#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""
Unified training and inference pipeline with resume capability.

Usage:
    python run.py -c path/to/experiment.yaml
    python run.py -d /path/to/run   # uses /path/to/run/experiment.yaml;
                                     # output_base_dir is set to /path/to/run

Multi-GPU: set ``nproc_per_node`` in ``experiment.yaml`` or pass ``--nproc-per-node N`` (default 1).
That value is used for both the ``training.py`` and ``inference.py`` subprocesses
(``torch.distributed.run`` when ``> 1``). Do not wrap ``run.py`` itself with ``torchrun``.
Step 4 calls ``collect_stats.py`` with ``-p`` (max concurrent ``calc_metrics`` subprocesses);
default is ``4 × --nproc-per-node`` unless ``-p`` / ``--stats-parallel`` overrides.

Predictor head is selected in the experiment YAML via ``predictor.type`` (fusion-only).
Passed through to ``training.py`` / ``inference.py``.

Pipeline:
    1. Training → <output_base_dir>/model/
    2. Inference on training data → <output_base_dir>/reg/
    3. Inference on test data → <output_base_dir>/test/
    4. If ``collect_stats`` is true (default): ``collect_stats.py`` (parallel calc_metrics + wide CSV)
       → <output_base_dir>/stats.wide.full.csv

Inference checkpoints are chosen from ``<output_base_dir>/model/checkpoint-*`` via optional
``inference_checkpoints``: sort ``checkpoint-N`` by ``N`` descending (newest first), then take
every ``checkpoint_stride``-th entry starting at the newest—like ``reversed(sorted)[::checkpoint_stride]``—
until ``pick_n`` checkpoints are collected (defaults: ``pick_n: 3``, ``checkpoint_stride: 1``).
Inference runs on those checkpoints in ascending ``N`` order.

Resume capability uses flag files in <output_base_dir>/.flags/
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional, TextIO

import yaml

from model.config import effective_inference_batch_size

# Import tee_log for output redirection
from tee_log import tee_output

# Directory containing this script — subprocess targets use absolute paths here.
BINDIR = Path(__file__).resolve().parent


def run_subprocess_teed(cmd: list[str]) -> None:
    """Run ``cmd``; copy child stdout/stderr through ``sys.stdout`` / ``sys.stderr`` (teed when active).

    Uses pipes plus threads so large output cannot deadlock; failed commands raise
    ``subprocess.CalledProcessError``.  Pump thread exceptions are re-raised in the
    calling thread so I/O failures are never silently swallowed.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    pump_exceptions: list[BaseException] = []

    def _pump(src: Optional[TextIO], dst: TextIO) -> None:
        if src is None:
            return
        try:
            while True:
                chunk = src.read(8192)
                if not chunk:
                    break
                dst.write(chunk)
                dst.flush()
        except Exception as exc:
            pump_exceptions.append(exc)
        finally:
            src.close()

    assert proc.stdout is not None and proc.stderr is not None
    t_out = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout))
    t_err = threading.Thread(target=_pump, args=(proc.stderr, sys.stderr))
    t_out.start()
    t_err.start()
    ret = proc.wait()
    t_out.join()
    t_err.join()
    if pump_exceptions:
        raise RuntimeError(f"Output pump failed: {pump_exceptions[0]}") from pump_exceptions[0]
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def get_logger() -> logging.Logger:
    """Get a simple logger that writes to ``sys.stdout``.

    Call only after ``tee_output`` is active so logs go to the tee (e.g. ``run.o``).
    """
    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)
    
    # Only add handler if not already added
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


class FlagManager:
    """Manage flag files for resume capability.
    
    Flag files are stored in <output_base_dir>/flags/ with visible names:
    - flags/training_done
    - flags/inference_train.data_name.checkpoint-XXXX_done
    - flags/inference_test.data_name.checkpoint-XXXX_done
    """
    
    def __init__(self, output_base_dir: str):
        self.flags_dir = os.path.join(output_base_dir, "flags")
        os.makedirs(self.flags_dir, exist_ok=True)
        self.logger = get_logger()
        self.logger.info(f"Flag directory: {self.flags_dir}")
    
    def _flag_path(self, name: str) -> str:
        """Get the full path for a flag file (visible name, no leading dot)."""
        return os.path.join(self.flags_dir, f"{name}_done")
    
    def is_done(self, name: str) -> bool:
        """Check if a step is already completed."""
        flag_path = self._flag_path(name)
        exists = os.path.exists(flag_path)
        if exists:
            self.logger.info(f"Found flag file (step completed): {flag_path}")
        return exists
    
    def mark_done(self, name: str) -> None:
        """Mark a step as completed."""
        flag_path = self._flag_path(name)
        with open(flag_path, "w") as f:
            f.write(f"Completed: {name}\n")
        self.logger.info(f"Created flag file: {flag_path}")
    
    def clear_flag(self, name: str) -> None:
        """Clear a flag (for re-running)."""
        flag_path = self._flag_path(name)
        if os.path.exists(flag_path):
            os.remove(flag_path)
            self.logger.info(f"Removed flag file: {flag_path}")


def load_experiment_config(config_path: str) -> dict[str, Any]:
    """Load experiment YAML (mapping at top level)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    if not isinstance(cfg, dict):
        raise ValueError("Invalid YAML config (expected a mapping at top level)")
    
    return cfg


def resolve_nproc_per_node(cli_value: Optional[int], cfg: dict[str, Any]) -> int:
    """GPU count for training and inference subprocesses: CLI overrides YAML ``nproc_per_node``; default 1."""
    raw = cli_value if cli_value is not None else cfg.get("nproc_per_node", 1)
    try:
        n = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError("nproc_per_node must be a positive integer") from e
    if n < 1:
        raise ValueError("nproc_per_node must be >= 1")
    return n


def get_checkpoints(model_dir: str) -> list[tuple[int, str]]:
    """Find checkpoint directories; return sorted list of (global_step, dir_name)."""
    if not os.path.exists(model_dir):
        return []

    checkpoints: list[tuple[int, str]] = []
    for entry in os.listdir(model_dir):
        if entry.startswith("checkpoint-"):
            match = re.match(r"checkpoint-(\d+)", entry)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoints.append((checkpoint_num, entry))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def resolve_inference_checkpoint_params(cfg: dict[str, Any]) -> tuple[int, int]:
    """Return (pick_n, checkpoint_stride) for inference checkpoint selection.

    YAML (optional)::

        inference_checkpoints:
          pick_n: 3
          checkpoint_stride: 1

    Checkpoints are ordered by ``N`` in ``checkpoint-N`` descending (newest first). Starting at
    the newest, take that checkpoint, skip the next ``checkpoint_stride - 1`` checkpoints in that
    order, take the next, repeat until ``pick_n`` picks (or the list is exhausted). This matches
    stepping through ``reversed(sorted_by_N)`` with stride ``checkpoint_stride`` and taking the
    first ``pick_n`` elements.

    """
    block = cfg.get("inference_checkpoints")
    pick_n = 3
    checkpoint_stride = 1
    if block is not None:
        if not isinstance(block, dict):
            raise ValueError(
                "inference_checkpoints must be a mapping with pick_n and/or checkpoint_stride"
            )
        if "last_k" in block and block["last_k"] is not None:
            raise ValueError("inference_checkpoints.last_k is not supported; use pick_n")
        if "pick_n" in block and block["pick_n"] is not None:
            pick_n = int(block["pick_n"])
        if "checkpoint_stride" in block and block["checkpoint_stride"] is not None:
            checkpoint_stride = int(block["checkpoint_stride"])
    if pick_n < 1:
        raise ValueError("inference_checkpoints.pick_n must be >= 1")
    if checkpoint_stride < 1:
        raise ValueError("inference_checkpoints.checkpoint_stride must be >= 1")
    return pick_n, checkpoint_stride


def select_inference_checkpoint_names(
    model_dir: str, pick_n: int, checkpoint_stride: int
) -> list[str]:
    """Checkpoint dir names to run inference on (ascending by global step)."""
    numbered = get_checkpoints(model_dir)
    if not numbered:
        return []
    descending = list(reversed(numbered))
    selected: list[tuple[int, str]] = []
    for i in range(0, len(descending), checkpoint_stride):
        if len(selected) >= pick_n:
            break
        selected.append(descending[i])
    selected.sort(key=lambda x: x[0])
    return [name for _, name in selected]


def run_training_step(
    cfg: dict[str, Any],
    flags: FlagManager,
    logger: logging.Logger,
    *,
    nproc_per_node: int = 1,
) -> str:
    """Run training step. Returns model directory path.

    When ``nproc_per_node`` > 1, launches ``training.py`` via ``torch.distributed.run``
    (same effect as ``torchrun --nproc_per_node=N training.py``).
    """
    output_base_dir = cfg["output_base_dir"]
    model_dir = os.path.join(output_base_dir, "model")
    
    if flags.is_done("training"):
        logger.info("Training already completed (flag found)")
        return model_dir
    
    logger.info("=" * 60)
    logger.info("Step 1: Running Training")
    logger.info("=" * 60)
    
    train_cfg = dict(cfg)
    train_cfg.pop("output_base_dir", None)
    train_cfg["output_training_dir"] = model_dir
    train_cfg.pop("nproc_per_node", None)

    # Remove inference-only keys
    for key in [
        "ckpt_path",
        "output_eval_dir",
        "save_csv",
        "save_pickle",
        "calc_metrics",
        "inference_batch_size",
        "inference_checkpoints",
        "collect_stats",
    ]:
        train_cfg.pop(key, None)
    
    os.makedirs(os.path.join(output_base_dir, "config"), exist_ok=True)
    train_config_path = os.path.join(output_base_dir, "config", "train.yaml")
    with open(train_config_path, "w") as f:
        yaml.dump(train_cfg, f)

    try:
        training_script = str(BINDIR / "training.py")
        if nproc_per_node == 1:
            cmd = [sys.executable, training_script, "-c", train_config_path]
        else:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc_per_node}",
                "--standalone",
                training_script,
                "-c",
                train_config_path,
            ]
        logger.info(f"Running: {' '.join(cmd)}")
        run_subprocess_teed(cmd)
        logger.info("Training completed successfully")
        flags.mark_done("training")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        raise

    return model_dir


def run_inference_step(
    cfg: dict[str, Any],
    checkpoint_name: str,
    checkpoint_path: str,
    data_entry: dict[str, Any],
    output_dir: str,
    flags: FlagManager,
    logger: logging.Logger,
    flag_prefix: str,
    *,
    nproc_per_node: int = 1,
) -> None:
    """Run inference for a single checkpoint and data entry.

    When ``nproc_per_node`` > 1, launches ``inference.py`` via ``torch.distributed.run``.
    Optional metrics: if experiment sets ``collect_stats: true`` (default), ``run.py`` runs
    ``collect_stats.py`` after inference; it does not use ``calc_metrics`` inside ``inference.py``.
    """
    data_name = data_entry["name"]
    flag_name = f"{flag_prefix}.{data_name}.{checkpoint_name}"

    if flags.is_done(flag_name):
        logger.info(f"Inference already completed: {flag_name}")
        return

    logger.info(f"Running inference: {checkpoint_name} on {data_name}")
    
    # Create inference config
    infer_cfg = {
        "ckpt_path": checkpoint_path,
        "output_eval_dir": output_dir,
        # May be omitted if per-entry `genome_fasta` is provided in test_data entries.
        "genome_fasta": cfg.get("genome_fasta"),
        "model_path": cfg["model_path"],
        "model_torch_dtype": cfg.get("model_torch_dtype", "bfloat16"),
        "atac_encoder_output_dim": cfg.get("atac_encoder_output_dim", 1024),
        "target_len": int(cfg.get("target_len", 32000)),
        "overlap_len": int(cfg.get("overlap_len", int(cfg.get("target_len", 32000)) // 2)),
        "chromosome": cfg.get("chromosome"),
        "test_data": [data_entry],
        "save_csv": cfg.get("save_csv", True),
        "save_pickle": cfg.get("save_pickle", True),
        "inference_batch_size": effective_inference_batch_size(cfg),
    }

    cq = cfg.get("cap_expression_quantile")
    if cq is not None:
        infer_cfg["cap_expression_quantile"] = cq

    # Add predictor if present
    if "predictor" in cfg:
        infer_cfg["predictor"] = cfg["predictor"]
    
    os.makedirs(os.path.join(cfg["output_base_dir"], "config"), exist_ok=True)
    infer_config_path = os.path.join(
        cfg["output_base_dir"],
        "config",
        "infer." + os.path.basename(output_dir) + ".yaml",
    )
    with open(infer_config_path, "w") as f:
        yaml.dump(infer_cfg, f)

    try:
        infer_script = str(BINDIR / "inference.py")
        if nproc_per_node == 1:
            cmd = [sys.executable, infer_script, infer_config_path]
        else:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc_per_node}",
                "--standalone",
                infer_script,
                infer_config_path,
            ]
        logger.info(f"Running: {' '.join(cmd)}")
        run_subprocess_teed(cmd)
        logger.info(f"Inference completed: {flag_name}")
        flags.mark_done(flag_name)
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed with exit code {e.returncode}")
        raise


def run_inference_on_entries(
    cfg: dict[str, Any],
    model_dir: str,
    flags: FlagManager,
    logger: logging.Logger,
    *,
    data_key: str,
    results_subdir: str,
    flag_prefix: str,
    pick_n: int,
    checkpoint_stride: int,
    skip_if_empty: bool = False,
    nproc_per_node: int = 1,
) -> None:
    """Run inference on ``cfg[data_key]`` using checkpoints from ``select_inference_checkpoint_names``."""
    use_checkpoints = select_inference_checkpoint_names(model_dir, pick_n, checkpoint_stride)
    if not use_checkpoints:
        logger.warning(
            "No checkpoints selected for inference "
            f"(model_dir={model_dir!r}, pick_n={pick_n}, checkpoint_stride={checkpoint_stride})"
        )
        return

    logger.info(
        f"Inference ({results_subdir!r}, {data_key}): pick_n={pick_n}, checkpoint_stride={checkpoint_stride} → "
        f"{use_checkpoints}"
    )

    output_base_dir = cfg["output_base_dir"]
    base_dir = os.path.join(output_base_dir, results_subdir)
    os.makedirs(base_dir, exist_ok=True)

    entries = cfg.get(data_key, [])
    if not entries:
        if skip_if_empty:
            logger.info(
                f"No {data_key} found in config, skipping inference for {results_subdir!r}"
            )
        else:
            logger.warning(f"No {data_key} found in config")
        return

    for checkpoint_name in use_checkpoints:
        checkpoint_path = os.path.join(model_dir, checkpoint_name, "model.safetensors")
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            continue

        for data_entry in entries:
            data_name = data_entry["name"]
            output_dir = os.path.join(base_dir, f"{data_name}.{checkpoint_name}")
            os.makedirs(output_dir, exist_ok=True)

            run_inference_step(
                cfg,
                checkpoint_name,
                checkpoint_path,
                data_entry,
                output_dir,
                flags,
                logger,
                flag_prefix,
                nproc_per_node=nproc_per_node,
            )


def run_collect_stats_step(
    output_base_dir: str,
    logger: logging.Logger,
    *,
    calc_metrics_parallel: int,
) -> None:
    """Run collect_stats.py: parallel calc_metrics on inference dirs, then wide CSV."""
    logger.info("=" * 60)
    logger.info("Step 4: calc_metrics (parallel) + stats.wide CSV(s)")
    logger.info("=" * 60)

    cmd = [
        sys.executable,
        str(BINDIR / "collect_stats.py"),
        output_base_dir,
        "-p",
        str(calc_metrics_parallel),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    run_subprocess_teed(cmd)


def _resolve_experiment_config_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Return absolute path to experiment.yaml (-c path or <output_dir>/experiment.yaml)."""
    if args.config_path is not None:
        return os.path.abspath(args.config_path)
    exp_path = os.path.join(os.path.abspath(args.output_dir), "experiment.yaml")
    if not os.path.isfile(exp_path):
        parser.error(f"config not found: {exp_path}")
    return exp_path


def main():
    parser = argparse.ArgumentParser(
        description="Unified training and inference pipeline with resume capability."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "-c",
        "--config",
        dest="config_path",
        metavar="PATH",
        help="Path to experiment.yaml",
    )
    src.add_argument(
        "-d",
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        help="Run directory: load DIR/experiment.yaml and set output_base_dir to DIR",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run by clearing all flags"
    )
    parser.add_argument(
        "-r",
        "--reuse-model",
        action="store_true",
        help="Skip training: use checkpoints under <output_base_dir>/model/ and mark training completed",
    )
    parser.add_argument(
        "-n",
        "--nproc-per-node",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of GPUs for training.py and inference.py subprocesses (torch.distributed.run). "
            "Default: experiment YAML key nproc_per_node, else 1. With --reuse-model, training is skipped "
            "but inference still uses this value."
        ),
    )
    parser.add_argument(
        "-p",
        "--stats-parallel",
        type=int,
        default=None,
        dest="stats_parallel",
        metavar="P",
        help=(
            "Max concurrent calc_metrics subprocesses passed to collect_stats.py "
            "(default: 4 × --nproc-per-node)."
        ),
    )

    args = parser.parse_args()
    
    config_path = _resolve_experiment_config_path(args, parser)
    cfg = load_experiment_config(config_path)
    if args.output_dir is not None:
        cfg["output_base_dir"] = os.path.abspath(args.output_dir)
    try:
        output_base_dir = os.path.abspath(cfg["output_base_dir"])
    except (KeyError, TypeError):
        parser.error(
            "config must set output_base_dir (or use -d /path/to/run)"
        )

    from valid_config import validate_config_dict

    ok, verrors = validate_config_dict(cfg)
    if not ok:
        parser.error("config validation failed:\n  - " + "\n  - ".join(verrors))

    try:
        nproc_per_node = resolve_nproc_per_node(args.nproc_per_node, cfg)
    except ValueError as e:
        parser.error(str(e))

    if args.force:
        import shutil

        flags_dir = os.path.join(output_base_dir, "flags")
        if os.path.exists(flags_dir):
            shutil.rmtree(flags_dir)
        os.makedirs(flags_dir, exist_ok=True)

    # Tee stdout/stderr first so logging and subprocess pumps hit run.o / run.e
    with tee_output(output_base_dir, "run.o", "run.e"):
        flags = FlagManager(output_base_dir)
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("Starting Run Pipeline")
        logger.info(f"Config: {config_path}")
        logger.info(f"Output directory: {output_base_dir}")
        logger.info("=" * 60)
        
        if args.force:
            logger.info("Force mode: cleared all flags")
        if args.reuse_model:
            logger.info("Reuse-model mode: will skip training and use existing checkpoints")
        else:
            logger.info(
                f"Training subprocess: nproc_per_node={nproc_per_node} "
                f"({'torch.distributed.run' if nproc_per_node > 1 else 'single process'})"
            )
        logger.info(
            f"Inference subprocess(es): nproc_per_node={nproc_per_node} "
            f"({'torch.distributed.run' if nproc_per_node > 1 else 'single process'})"
        )

        try:
            pick_n, checkpoint_stride = resolve_inference_checkpoint_params(cfg)
        except ValueError as e:
            parser.error(str(e))

        stats_parallel = (
            int(args.stats_parallel)
            if args.stats_parallel is not None
            else max(1, 4 * nproc_per_node)
        )

        _run_pipeline(
            cfg,
            flags,
            logger,
            reuse_model=args.reuse_model,
            nproc_per_node=nproc_per_node,
            inference_pick_n=pick_n,
            inference_checkpoint_stride=checkpoint_stride,
            calc_metrics_parallel=stats_parallel,
        )


def _run_pipeline(
    cfg: dict[str, Any],
    flags: FlagManager,
    logger: logging.Logger,
    *,
    reuse_model: bool = False,
    nproc_per_node: int = 1,
    inference_pick_n: int = 3,
    inference_checkpoint_stride: int = 1,
    calc_metrics_parallel: int = 4,
) -> None:
    """Execute the pipeline steps."""
    output_base_dir = cfg["output_base_dir"]

    try:
        os.makedirs(os.path.join(output_base_dir, "config"), exist_ok=True)
        model_dir = os.path.join(output_base_dir, "model")

        if reuse_model:
            logger.info("=" * 60)
            logger.info("Step 1: Skipping training (--reuse-model)")
            logger.info("=" * 60)
            if not os.path.isdir(model_dir):
                raise ValueError(f"--reuse-model: model directory not found: {model_dir}")
            if not get_checkpoints(model_dir):
                raise ValueError(
                    f"--reuse-model: no checkpoint-* directories under {model_dir}"
                )
            flags.mark_done("training")
            logger.info("Training step marked completed (flag); using existing checkpoints")
        else:
            model_dir = run_training_step(cfg, flags, logger, nproc_per_node=nproc_per_node)
        
        # Step 2: Inference on training data
        logger.info("=" * 60)
        logger.info("Step 2: Inference on Training Data")
        logger.info("=" * 60)
        run_inference_on_entries(
            cfg,
            model_dir,
            flags,
            logger,
            data_key="training_data",
            results_subdir="reg",
            flag_prefix="inference_train",
            pick_n=inference_pick_n,
            checkpoint_stride=inference_checkpoint_stride,
            skip_if_empty=False,
            nproc_per_node=nproc_per_node,
        )

        # Step 3: Inference on test data
        logger.info("=" * 60)
        logger.info("Step 3: Inference on Test Data")
        logger.info("=" * 60)
        run_inference_on_entries(
            cfg,
            model_dir,
            flags,
            logger,
            data_key="test_data",
            results_subdir="test",
            flag_prefix="inference_test",
            pick_n=inference_pick_n,
            checkpoint_stride=inference_checkpoint_stride,
            skip_if_empty=True,
            nproc_per_node=nproc_per_node,
        )

        if cfg.get("collect_stats", True):
            run_collect_stats_step(
                output_base_dir,
                logger,
                calc_metrics_parallel=calc_metrics_parallel,
            )
        else:
            logger.info("Skipping collect_stats (collect_stats: false in config)")

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

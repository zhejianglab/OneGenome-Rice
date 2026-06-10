#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-04-02
"""
Validate YAML config format and required input paths.

Checks:
- YAML top-level must be a mapping (dict)
- Dataset blocks must be structured lists (training_data or test_data)
- All referenced *input* file paths exist:
  - rna_path_plus / rna_path_minus / atac_path
  - genome_fasta (global or per-entry override)
  - model_path (dir or file)
  - ckpt_path if provided
- Optional: ``training.scale_gradient_accumulation_for_world_size`` (reference world size is fixed to 1 in code)

Does NOT require output directories to exist:
- output_base_dir, output_training_dir, output_eval_dir

Usage:
  python valid_config.py path/to/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from model.config import parse_dataset_block


def _is_existing_path(p: str) -> bool:
    try:
        return Path(p).exists()
    except OSError:
        return False


def _validate_model_path(cfg: dict[str, Any], errors: list[str]) -> None:
    if "model_path" not in cfg or cfg["model_path"] is None:
        errors.append("missing required key: model_path")
        return
    mp = str(cfg["model_path"])
    if not _is_existing_path(mp):
        errors.append(f"model_path not found: {mp}")


def _validate_ckpt_path_if_present(cfg: dict[str, Any], errors: list[str]) -> None:
    if "ckpt_path" not in cfg or cfg["ckpt_path"] is None:
        return
    cp = str(cfg["ckpt_path"])
    if not os.path.isfile(cp):
        errors.append(f"ckpt_path not found (expected file): {cp}")


def _validate_yaml_mapping(cfg: Any, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(cfg, dict):
        errors.append("invalid YAML: expected a mapping at top level")
        return None
    return cfg


def _validate_dataset_block(cfg: dict[str, Any], errors: list[str]) -> None:
    # parse_dataset_block performs structure checks and file existence checks for:
    # - rna_path_plus/rna_path_minus/atac_path
    # - genome_fasta (global or per-entry)
    # and returns normalized per-entry overrides for chromosome and genome_fasta.
    try:
        parse_dataset_block(cfg)
    except Exception as e:
        errors.append(f"dataset block invalid: {type(e).__name__}: {e}")


def _validate_inference_checkpoints(cfg: dict[str, Any], errors: list[str]) -> None:
    block = cfg.get("inference_checkpoints")
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(
            "inference_checkpoints must be a mapping (e.g. pick_n, checkpoint_stride)"
        )
        return
    if block.get("last_k") is not None:
        errors.append("inference_checkpoints.last_k is not supported; use pick_n")
    pick_n = 3
    if block.get("pick_n") is not None:
        try:
            pick_n = int(block["pick_n"])
        except (TypeError, ValueError):
            errors.append("inference_checkpoints.pick_n must be an integer")
            return

    checkpoint_stride = 1
    if block.get("checkpoint_stride") is not None:
        try:
            checkpoint_stride = int(block["checkpoint_stride"])
        except (TypeError, ValueError):
            errors.append("inference_checkpoints.checkpoint_stride must be an integer")
            return
    if pick_n < 1:
        errors.append("inference_checkpoints.pick_n must be >= 1")
    if checkpoint_stride < 1:
        errors.append("inference_checkpoints.checkpoint_stride must be >= 1")


def _validate_training_checkpoint_schedule(cfg: dict[str, Any], errors: list[str]) -> None:
    tcfg = cfg.get("training")
    if tcfg is None:
        return
    if not isinstance(tcfg, dict):
        errors.append("training must be a mapping")
        return

    save_num_raw = tcfg.get("save_num_per_epoch")
    save_per_n_raw = tcfg.get("save_per_n_epoch")

    def _as_pos_int(v: Any, key: str) -> int | None:
        if v is None:
            return None
        try:
            i = int(v)
        except (TypeError, ValueError):
            errors.append(f"{key} must be an integer")
            return None
        if i < 1:
            errors.append(f"{key} must be >= 1")
            return None
        return i

    k = _as_pos_int(save_num_raw, "training.save_num_per_epoch")
    n = _as_pos_int(save_per_n_raw, "training.save_per_n_epoch")

    if k is not None and n is not None and not (k == 1 and n == 1):
        errors.append(
            "training.save_per_n_epoch is exclusive with training.save_num_per_epoch "
            "(unless both are set to 1)"
        )

    save_strategy = tcfg.get("save_strategy")
    if save_strategy is not None:
        s = str(save_strategy).strip().lower()
        if s == "steps" and k is None:
            errors.append(
                "training.save_strategy 'steps' requires training.save_num_per_epoch "
                "(save_steps is not supported)"
            )

    gas = tcfg.get("gradient_accumulation_steps")
    if gas is not None:
        try:
            gi = int(gas)
        except (TypeError, ValueError):
            errors.append("training.gradient_accumulation_steps must be an integer")
        else:
            if gi < 1:
                errors.append("training.gradient_accumulation_steps must be >= 1")

    raw_scale = tcfg.get("scale_gradient_accumulation_for_world_size")
    scale_ga = False
    if raw_scale is None:
        scale_ga = False
    elif isinstance(raw_scale, bool):
        scale_ga = raw_scale
    elif isinstance(raw_scale, int):
        scale_ga = raw_scale != 0
    elif isinstance(raw_scale, str):
        s = raw_scale.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            scale_ga = True
        elif s in ("0", "false", "no", "n", "off", ""):
            scale_ga = False
        else:
            errors.append(
                "training.scale_gradient_accumulation_for_world_size must be a boolean "
                f"(got {raw_scale!r})"
            )
    else:
        errors.append(
            "training.scale_gradient_accumulation_for_world_size must be a boolean "
            f"(got {type(raw_scale).__name__})"
        )


def _normalize_predictor_type(cfg: dict[str, Any]) -> str:
    """Match ``model.pipeline._normalize_predictor_type`` (predictor block only)."""
    block = cfg.get("predictor")
    if isinstance(block, dict):
        if "type" in block and block["type"] is not None:
            s = str(block["type"]).strip().lower()
            return s if s else "fusion"
    return "fusion"


def _validate_predictor_type(cfg: dict[str, Any], errors: list[str]) -> None:
    # This repo only supports fusion; predictor.type may be omitted (defaults to fusion).
    block = cfg.get("predictor")
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append("predictor must be a mapping (e.g. {type: fusion})")
        return
    raw = block.get("type")
    if raw is None:
        return
    s = str(raw).strip().lower()
    if not s:
        return
    if s != "fusion":
        errors.append("predictor.type must be 'fusion' (v0 and other predictor types are not supported)")


def _validate_misc_paths(cfg: dict[str, Any], errors: list[str]) -> None:
    # Optional inference metrics inputs (if user passes them explicitly)
    for k in ("metrics_plus_input", "metrics_minus_input"):
        if k in cfg and cfg[k] is not None:
            p = str(cfg[k])
            if not os.path.isfile(p):
                errors.append(f"{k} not found (expected file): {p}")

    # Optional genome_fasta global (if set, must exist as a file)
    # Note: if omitted, per-entry genome_fasta must be set; parse_dataset_block enforces that.
    if "genome_fasta" in cfg and cfg["genome_fasta"] is not None:
        p = str(cfg["genome_fasta"])
        if not os.path.isfile(p):
            errors.append(f"genome_fasta not found (expected file): {p}")


def validate_config_dict(
    cfg: Any,
) -> tuple[bool, list[str]]:
    """Run validation checks on a loaded config mapping.

    """
    errors: list[str] = []
    cfg = _validate_yaml_mapping(cfg, errors)
    if cfg is None:
        return False, errors

    _validate_model_path(cfg, errors)
    _validate_ckpt_path_if_present(cfg, errors)
    _validate_dataset_block(cfg, errors)
    _validate_predictor_type(cfg, errors)
    _validate_inference_checkpoints(cfg, errors)
    _validate_training_checkpoint_schedule(cfg, errors)
    # target_len / overlap_len sanity
    if "target_len" in cfg and cfg["target_len"] is not None:
        try:
            tl = int(cfg["target_len"])
        except (TypeError, ValueError):
            errors.append("target_len must be an integer")
        else:
            if tl < 1:
                errors.append("target_len must be >= 1")
    if "overlap_len" in cfg and cfg["overlap_len"] is not None:
        try:
            ol = int(cfg["overlap_len"])
        except (TypeError, ValueError):
            errors.append("overlap_len must be an integer")
        else:
            if ol < 0:
                errors.append("overlap_len must be >= 0")
    _validate_misc_paths(cfg, errors)

    return (len(errors) == 0), errors


def validate_config(
    path: str,
) -> tuple[bool, list[str]]:
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        return False, [f"failed to read/parse YAML: {type(e).__name__}: {e}"]

    return validate_config_dict(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate config format and required input file paths."
    )
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()

    ok, errors = validate_config(args.config)
    if ok:
        print(f"OK: config valid: {args.config}")
        return 0

    print(f"ERROR: config invalid: {args.config}")
    for e in errors:
        print(f"- {e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


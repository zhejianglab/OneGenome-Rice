#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""
Unified ATAC-conditioned RNA training entrypoint.

Loads YAML config and runs `model.pipeline.run_training`. Set `predictor.type` in the
config to `fusion` (default if omitted).

Stdout is teed to output_training_dir/train.o; stderr is teed to output_training_dir/train.e
(both opened in append mode).

Usage:
  python training.py -c path/to/config.yaml
  python training.py -d /path/to/run   # uses /path/to/run/config/training.yaml;
                                       # output_training_dir is set to /path/to/run
"""

import argparse
import logging
import os
import sys

import yaml

def _load_training_config(args, parser):
    if args.config_path is not None:
        with open(args.config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        config_path = os.path.join(args.output_dir, "config", "training.yaml")
        if not os.path.isfile(config_path):
            parser.error(f"config not found: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["output_training_dir"] = os.path.abspath(args.output_dir)
    if not isinstance(cfg, dict):
        parser.error("invalid YAML config (expected a mapping at top level)")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="ATAC-conditioned RNA prediction training (YAML config, predictor.type fusion).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "-c",
        "--config",
        dest="config_path",
        metavar="PATH",
        help="Path to YAML training config",
    )
    src.add_argument(
        "-d",
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        help="Run directory: load DIR/config/training.yaml and set output_training_dir to DIR",
    )
    args = parser.parse_args()
    cfg = _load_training_config(args, parser)
    train_out = cfg.get("output_training_dir")
    if not train_out:
        parser.error(
            "config must set output_training_dir (or use -d /path/to/run)"
        )

    # Heavy imports after argparse: keeps `--help` fast and avoids importing torch/transformers unless needed.
    # Configure env and seeds before other `model` imports (side effects in model.env).
    import model.env  # noqa: F401
    from model.pipeline import run_training
    from tee_log import tee_output

    from valid_config import validate_config_dict

    ok, verrors = validate_config_dict(cfg)
    if not ok:
        parser.error("config validation failed:\n  - " + "\n  - ".join(verrors))

    log_dir = os.path.abspath(train_out)

    with tee_output(log_dir, "train.o", "train.e"):
        # logging.basicConfig() in model.env ran at import time and captured the
        # original sys.stderr (pipe) as the StreamHandler stream, before tee_output
        # replaced sys.stderr with a TeeStream.  Re-point every root-logger
        # StreamHandler to the current sys.stderr so logging output reaches
        # train.e as well as the pipe that run.py pumps to run.e / screen.
        for _handler in logging.root.handlers:
            if isinstance(_handler, logging.StreamHandler):
                _handler.stream = sys.stderr
        run_training(cfg)


if __name__ == "__main__":
    main()

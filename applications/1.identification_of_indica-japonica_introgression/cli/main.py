from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config, require_keys
from src.common.schema import normalize_and_validate_config, get_indent
from src.pipelines.run import run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified domain-based project CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to unified config file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["prepare", "extract", "train", "test", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset names from data_process.datasets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, cli_args=sys.argv[1:])
    require_keys(
        config,
        [
            "model",
            "data_process",
            "embedding",
            "environment",
            "classifiers",
            "output",
        ],
    )
    config = normalize_and_validate_config(config)
    print("=" * 100)
    print(f"{get_indent('PIPELINE')} run.name={config.run.name}, isolate_embeddings={config.run.isolate_embeddings}")
    outputs = run_stage(config, stage=args.stage, datasets=args.datasets)
    print(f"{get_indent('PIPELINE')} Completed stage: {args.stage}")
    print("=" * 50)
    print("=" * 50)
    for path in outputs:
        print(f"- {path}")
    print("=" * 100)


if __name__ == "__main__":
    main()


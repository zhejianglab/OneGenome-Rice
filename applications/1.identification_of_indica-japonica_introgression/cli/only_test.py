from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_config
from src.pipelines.only_test import only_test_pipeline
from src.common.schema import get_indent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Only-test pipeline for external RF and embedding files")
    parser.add_argument(
        "--config",
        type=str,
        default="config/only_test.yaml",
        help="Path to only_test config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, cli_args=sys.argv[1:])
    outputs = only_test_pipeline(config)
    print(f"{get_indent('ONLY_TEST')} Completed")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()


from __future__ import annotations

from pathlib import Path

from src.datasets.prepare import run_prepare
from src.common.config import Config


def prepare_pipeline(config: Config, datasets: list[str] | None = None) -> list[Path]:
    return run_prepare(config, datasets=datasets)


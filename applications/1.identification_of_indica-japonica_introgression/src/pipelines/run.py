from __future__ import annotations

from pathlib import Path

from src.pipelines.extract import extract_pipeline
from src.pipelines.prepare import prepare_pipeline
from src.pipelines.test import test_pipeline
from src.pipelines.train import train_pipeline
from src.common.config import Config
from src.common.schema import get_indent

def run_stage(config: Config, stage: str, datasets: list[str] | None = None) -> list[Path]:
    print(f"{get_indent('PIPELINE')} Stage start: {stage}")
    if stage == "prepare":
        out = prepare_pipeline(config, datasets=datasets)
        print("=" * 50)
        print(f"{get_indent('PIPELINE')} Stage done: {stage}, outputs={len(out)}")
        return out
    if stage == "extract":
        out = extract_pipeline(config, datasets=datasets)
        print("=" * 50)
        print(f"{get_indent('PIPELINE')} Stage done: {stage}, outputs={len(out)}")
        return out
    if stage == "train":
        out = [train_pipeline(config, datasets=datasets)]
        print("=" * 50)
        print(f"{get_indent('PIPELINE')} Stage done: {stage}, outputs={len(out)}")
        return out
    if stage == "test":
        out = test_pipeline(config, datasets=datasets)
        print("=" * 50)
        print(f"{get_indent('PIPELINE')} Stage done: {stage}, outputs={len(out)}")
        return out
    if stage == "all":
        out: list[Path] = []
        print(f"{get_indent('PIPELINE')} Running full workflow: prepare -> extract -> train -> test")
        print("=" * 50)
        out.extend(prepare_pipeline(config, datasets=datasets))
        out.extend(extract_pipeline(config, datasets=datasets))
        out.append(train_pipeline(config, datasets=datasets))
        out.extend(test_pipeline(config, datasets=datasets))
        print("=" * 50)
        print(f"{get_indent('PIPELINE')} Stage done: {stage}, outputs={len(out)}")
        return out
    raise ValueError(f"Unsupported stage: {stage}")


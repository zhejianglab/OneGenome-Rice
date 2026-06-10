from __future__ import annotations

from pathlib import Path

from src.common.config import Config


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def model_dir(config: Config) -> Path:
    return resolve(config.model.path)


def dataset_root(config: Config) -> Path:
    return resolve(config.data_process.dataset_dir)


def embedding_root(config: Config) -> Path:
    base = resolve(config.embedding.output_dir) / config.model.name
    if config.run.isolate_embeddings:
        return base / "runs" / str(config.run.name)
    return base


def result_root(config: Config) -> Path:
    return resolve(config.output.result_dir) / config.model.name / "runs" / str(config.run.name)


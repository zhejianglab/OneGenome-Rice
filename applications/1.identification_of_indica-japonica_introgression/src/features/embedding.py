from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.common.config import Config
from src.features.embedding_backend_builtin import collate_fn, extract_embeddings
from src.io.naming import embedding_filename
from src.io.paths import embedding_root


class JsonlSequenceDataset(torch.utils.data.Dataset):
    """Each row: sequence, label tensor inputs, plus metadata aligned with embeddings."""

    def __init__(self, file_path: str | Path):
        self.data: list[tuple[str, Any, str, str, str, list[int], str]] = []
        with Path(file_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                seq_start_end = item.get("seq_start_end", [0, 0])
                if not isinstance(seq_start_end, list) or len(seq_start_end) != 2:
                    seq_start_end = [0, 0]
                self.data.append(
                    (
                        item["sequence"],
                        item["label"],
                        str(item.get("variety", "")),
                        str(item.get("source", "")),
                        str(item.get("seq_id", "")),
                        [int(seq_start_end[0]), int(seq_start_end[1])],
                        str(item.get("source_group", "")),
                    )
                )
        self._seq_number = 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def get_seq_number(self) -> int:
        return self._seq_number


def embedding_cache_path(
    config: Config,
    dataset_tag_name: str,
    window_size: int,
    step_size: int,
    layer: int,
    split: str,
) -> Path:
    return embedding_root(config) / embedding_filename(
        dataset_tag_name,
        window_size,
        step_size,
        layer,
        split,
    )


def extract_and_save_embeddings(
    model: object,
    tokenizer: object,
    device: str,
    config: Config,
    dataset_tag_name: str,
    split_jsonl_path: str | Path,
    split: str,
    window_size: int,
    step_size: int,
    layers_to_extract: list[int] | None = None,
) -> dict[int, Path]:
    dataset = JsonlSequenceDataset(split_jsonl_path)
    layers = layers_to_extract if layers_to_extract is not None else config.embedding.layers
    batch_size = config.environment.get("batch_size", 8)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )
    gpu_list = config.environment.get("gpu_list", [0])
    gpu_id = gpu_list[0] if gpu_list else 0
    config.model_config = {
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": config.model.hidden_layers,
    }
    embeddings_dict, labels, meta = extract_embeddings(
        model,
        loader,
        device,
        gpu_id,
        dataset_tag_name,
        layers,
        dataset.get_seq_number(),
        config,
        tqdm_print=f"Extract {dataset_tag_name} {split}",
    )
    saved_paths: dict[int, Path] = {}
    for layer, emb in embeddings_dict.items():
        out_path = embedding_cache_path(
            config,
            dataset_tag_name,
            window_size,
            step_size,
            layer,
            split,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": emb,
                "labels": labels,
                "sequences": meta["sequences"],
                "variety": meta["variety"],
                "source": meta["source"],
                "seq_id": meta["seq_id"],
                "seq_start_end": meta["seq_start_end"],
                "source_group": meta["source_group"],
            },
            out_path,
        )
        saved_paths[layer] = out_path
    return saved_paths


def expected_embedding_paths(
    config: Config,
    dataset_tag_name: str,
    split: str,
    window_size: int,
    step_size: int,
) -> dict[int, Path]:
    """Return expected cache paths for all configured layers."""
    return {
        layer: embedding_cache_path(
            config=config,
            dataset_tag_name=dataset_tag_name,
            window_size=window_size,
            step_size=step_size,
            layer=layer,
            split=split,
        )
        for layer in config.embedding.layers
    }


def missing_embedding_layers(
    config: Config,
    dataset_tag_name: str,
    split: str,
    window_size: int,
    step_size: int,
) -> list[int]:
    """Return layers whose cache files are missing for the given split."""
    expected = expected_embedding_paths(
        config=config,
        dataset_tag_name=dataset_tag_name,
        split=split,
        window_size=window_size,
        step_size=step_size,
    )
    return [layer for layer, path in expected.items() if not path.exists()]


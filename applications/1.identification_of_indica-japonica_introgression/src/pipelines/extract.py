from __future__ import annotations

from pathlib import Path

from src.features.embedding import (
    extract_and_save_embeddings,
    expected_embedding_paths,
    missing_embedding_layers,
)
from src.io.naming import dataset_tag
from src.io.paths import dataset_root
from src.models.loader import load_model_and_tokenizer
from src.common.config import Config
from src.common.schema import get_indent

def extract_pipeline(config: Config, datasets: list[str] | None = None) -> list[Path]:
    model = None
    tokenizer = None
    device = None
    selected = datasets or list(config.data_process.datasets.__dict__.keys())
    outputs: list[Path] = []
    print("=" * 50)
    print(f"{get_indent('EXTRACT')} Stage start: datasets={selected}")
    for dataset_name in selected:
        label_cfg = config.data_process.datasets[dataset_name].label
        class_names = list(label_cfg.__dict__.keys())
        ds_tag = dataset_tag(dataset_name, class_names)
        ds_cfg = config.data_process.datasets[dataset_name]
        print(f"{get_indent('EXTRACT')} Dataset start: {dataset_name} -> {ds_tag}")
        for split in ["train", "test"]:
            split_cfg = ds_cfg[split]
            split_path = dataset_root(config) / ds_tag / f"{split}.jsonl"
            print(f"{get_indent('EXTRACT')} Split check: dataset={ds_tag}, split={split}, jsonl={split_path}")
            missing_layers = missing_embedding_layers(
                config=config,
                dataset_tag_name=ds_tag,
                split=split,
                window_size=split_cfg.window_size,
                step_size=split_cfg.step_size,
            )
            if not missing_layers:
                print(f"{get_indent('EXTRACT')} Cache hit: {ds_tag} {split}, all layers exist.")
                outputs.extend(
                    expected_embedding_paths(
                        config=config,
                        dataset_tag_name=ds_tag,
                        split=split,
                        window_size=split_cfg.window_size,
                        step_size=split_cfg.step_size,
                    ).values()
                )
                continue

            print(
                f"{get_indent('EXTRACT')} Cache miss: {ds_tag} {split}, extracting missing layers: {missing_layers}"
            )
            if model is None or tokenizer is None or device is None:
                print(f"{get_indent('EXTRACT')} Loading model/tokenizer for embedding extraction...")
                model, tokenizer, device = load_model_and_tokenizer(config)
                print(f"{get_indent('EXTRACT')} Model ready on device={device}")
            print(f"{get_indent('EXTRACT')} Start extraction: dataset={ds_tag}, split={split}, layers={missing_layers}")
            saved = extract_and_save_embeddings(
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=config,
                dataset_tag_name=ds_tag,
                split_jsonl_path=split_path,
                split=split,
                window_size=split_cfg.window_size,
                step_size=split_cfg.step_size,
                layers_to_extract=missing_layers,
            )
            outputs.extend(saved.values())
            print(f"{get_indent('EXTRACT')} Extraction done: dataset={ds_tag}, split={split}, saved={len(saved)}")
        print(f"{get_indent('EXTRACT')} Dataset done: {dataset_name}")
    print(f"{get_indent('EXTRACT')} Stage done: outputs={len(outputs)}")
    print("=" * 50)
    return outputs


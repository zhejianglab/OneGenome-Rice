from __future__ import annotations

from pathlib import Path

import torch
import os

from src.classifiers.rf import (
    load_rf_models,
    pool_embeddings,
    save_rf_models,
    train_rf_models,
)
from src.common.config import Config
from src.features.embedding import (
    embedding_cache_path,
    extract_and_save_embeddings,
    missing_embedding_layers,
)
from src.io.naming import dataset_tag, rf_model_filename
from src.io.paths import dataset_root, result_root
from src.models.loader import build_device, load_model_and_tokenizer
from src.common.schema import get_indent


def _dataset_tag_and_classes(config: Config, dataset_name: str) -> tuple[str, list[str]]:
    class_names = list(config.data_process.datasets[dataset_name].label.__dict__.keys())
    return dataset_tag(dataset_name, class_names), class_names


def train_pipeline(config: Config, datasets: list[str] | None = None) -> Path:
    model = None
    tokenizer = None
    device = build_device(config)
    selected = datasets or list(config.data_process.datasets.__dict__.keys())
    layers = config.embedding.layers
    cat_dim = config.embedding.get("pooled_embeddings_cat_dim", 0)
    rf_cfg = config.classifiers.RF
    result_path = []
    print("=" * 50)
    print(f"{get_indent('TRAIN')} Stage start: datasets={selected}, layers={layers}")

    for dataset_name in selected:
        ds_tag, _ = _dataset_tag_and_classes(config, dataset_name)
        train_cfg = config.data_process.datasets[dataset_name].train
        train_jsonl = dataset_root(config) / ds_tag / "train.jsonl"
        print(f"{get_indent('TRAIN')} Dataset start: {dataset_name} -> {ds_tag}")

        # Ensure embeddings exist for train.
        for split, split_jsonl, cfg_part in [
            ("train", train_jsonl, train_cfg)
        ]:
            missing_layers = missing_embedding_layers(
                config=config,
                dataset_tag_name=ds_tag,
                split=split,
                window_size=cfg_part.window_size,
                step_size=cfg_part.step_size,
            )
            if missing_layers:
                print(
                    f"{get_indent('TRAIN')} Cache miss: {ds_tag} {split}, extracting missing layers: {missing_layers}"
                )
                if model is None or tokenizer is None:
                    print(f"{get_indent('TRAIN')} Loading model/tokenizer for missing embedding extraction...")
                    model, tokenizer, device = load_model_and_tokenizer(config)
                    print(f"{get_indent('TRAIN')} Model ready on device={device}")
                extract_and_save_embeddings(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    config=config,
                    dataset_tag_name=ds_tag,
                    split_jsonl_path=split_jsonl,
                    split=split,
                    window_size=cfg_part.window_size,
                    step_size=cfg_part.step_size,
                    layers_to_extract=missing_layers,
                )
            else:
                print(f"{get_indent('TRAIN')} Cache hit: {ds_tag} {split}, all layers exist.")

        for layer in layers:
            print(f"{get_indent('TRAIN')} Layer start: dataset={ds_tag}, layer={layer}")
            model_path = result_root(config) / "last_epoch_model" / rf_model_filename(
                ds_tag,
                train_cfg.window_size,
                train_cfg.step_size,
                layer,
            )
            os.makedirs(model_path.parent, exist_ok=True)
            train_path = embedding_cache_path(
                config,
                ds_tag,
                train_cfg.window_size,
                train_cfg.step_size,
                layer,
                "train",
            )
            train_data = torch.load(train_path, map_location=device)
            x_train = pool_embeddings(train_data["embeddings"], cat_dim)
            y_train = train_data["labels"]
            if model_path.exists():
                print(f"{get_indent('TRAIN')} Skip trained model: {model_path}")
                rf_models = load_rf_models(model_path)
                result_path.append(model_path)
            else:
                print(f"{get_indent('TRAIN')} Training RF: dataset={ds_tag}, layer={layer}")
                rf_models = train_rf_models(
                    embeddings=x_train,
                    labels=y_train,
                    n_estimators=rf_cfg.n_estimators,
                    random_state=rf_cfg.get("random_state", config.environment.seed),
                )
                save_rf_models(
                    rf_models=rf_models,
                    rf_path=model_path,
                )
                result_path.append(model_path)
                print(f"{get_indent('TRAIN')} RF saved: {model_path}")
            print(f"{get_indent('TRAIN')} Layer done: dataset={ds_tag}, layer={layer}")
        print(f"{get_indent('TRAIN')} Dataset done: {dataset_name}")
    print(f"{get_indent('TRAIN')} Stage done: outputs={len(result_path)}")
    print("=" * 50)
    return result_path


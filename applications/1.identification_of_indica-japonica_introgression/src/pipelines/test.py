from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import os

from src.classifiers.rf import load_rf_models, predict_by_models
from src.common.config import Config
from src.evaluation.report import build_training_metrics, checkout_training_results, generate_results_df
from src.io.naming import (
    dataset_tag,
    embedding_filename,
    rf_model_filename,
    rf_test_result_filename,
    result_test_dataset_name,
)
from src.features.embedding import (
    extract_and_save_embeddings,
    missing_embedding_layers,
)
from src.io.paths import embedding_root, result_root, dataset_root
from src.models.loader import build_device, load_model_and_tokenizer
from src.common.schema import get_indent


def test_pipeline(config: Config, datasets: list[str] | None = None) -> list[Path]:
    selected = datasets or list(config.data_process.datasets.__dict__.keys())
    new_metrics: list[dict] = []
    out_files: list[Path] = []
    layers = config.embedding.layers
    threshold = config.classifiers.prediction.threshold
    cat_dim = config.embedding.get("pooled_embeddings_cat_dim", 0)
    device = build_device(config)
    result_path = result_root(config) / "training_results.tsv"
    existing_metrics = checkout_training_results(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print(f"{get_indent('TEST')} Stage start: datasets={selected}, layers={layers}, threshold={threshold}")

    for dataset_name in selected:
        class_names = list(config.data_process.datasets[dataset_name].label.__dict__.keys())
        ds_tag = dataset_tag(dataset_name, class_names)
        train_cfg = config.data_process.datasets[dataset_name].train
        test_cfg = config.data_process.datasets[dataset_name].test
        test_jsonl = dataset_root(config) / ds_tag / "test.jsonl"
        print(f"{get_indent('TEST')} Dataset start: {dataset_name} -> {ds_tag}")

        # Ensure embeddings exist for test.
        for split, split_jsonl, cfg_part in [
            ("test", test_jsonl, test_cfg),
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
                    f"{get_indent('TEST')} Cache miss: {ds_tag} {split}, extracting missing layers: {missing_layers}"
                )
                if model is None or tokenizer is None:
                    print(f"{get_indent('TEST')} Loading model/tokenizer for missing embedding extraction...")
                    model, tokenizer, device = load_model_and_tokenizer(config)
                    print(f"{get_indent('TEST')} Model ready on device={device}")
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
                print(f"{get_indent('TEST')} Cache hit: {ds_tag} {split}, all layers exist.")

        for layer in layers:
            print(f"{get_indent('TEST')} Layer start: dataset={ds_tag}, layer={layer}")
            new_metrics_skip = False
            test_result_df_skip = False

            test_dataset_name = result_test_dataset_name(
                ds_tag,
                test_cfg.window_size,
                test_cfg.step_size,
            )
            model_path = (
                result_root(config)
                / "last_epoch_model"
                / rf_model_filename(
                    ds_tag,
                    train_cfg.window_size,
                    train_cfg.step_size,
                    layer,
                )
            )
            existing_key = (test_dataset_name, model_path.stem, layer, threshold)
            if model_path.exists() and existing_key in existing_metrics:
                print(f"{get_indent('TEST')} Reuse existing metrics: dataset={ds_tag}, use_model={model_path.stem}, layer={layer}, threshold={threshold}")
                new_metrics_skip = True

            file_output_path = result_root(config) / model_path.stem / rf_test_result_filename(
                ds_tag,
                test_cfg.window_size,
                test_cfg.step_size,
                layer,
                threshold,
            )
            if file_output_path.exists():
                print(f"{get_indent('TEST')} Skip existing result: {file_output_path}")
                out_files.append(file_output_path)
                test_result_df_skip = True

            if new_metrics_skip and test_result_df_skip:
                print(f"{get_indent('TEST')} Skipping layer: dataset={ds_tag}, layer={layer}")
                continue
            
            test_embed_path = (
                embedding_root(config)
                / embedding_filename(
                    ds_tag,
                    test_cfg.window_size,
                    test_cfg.step_size,
                    layer,
                    "test",
                )
            )
            test_data = torch.load(test_embed_path, map_location=device)
            x_test = test_data["embeddings"]
            y_test = test_data["labels"]

            rf_models = load_rf_models(model_path)
            _, y_prob = predict_by_models(x_test, rf_models, cat_dim)

            if not new_metrics_skip:
                new_metrics.append(
                    build_training_metrics(
                        dataset_name=test_dataset_name,
                        layer=layer,
                        y_prob=y_prob,
                        y_true=y_test.cpu().numpy(),
                        model_name=model_path.stem,
                        threshold=threshold
                    )
                )
            if not test_result_df_skip:
                n = int(y_test.shape[0])

                def _col_str(name: str, default: str = "") -> list[str]:
                    v = test_data.get(name)
                    if v is None or len(v) != n:
                        return [default] * n
                    return [str(x) for x in v]

                def _col_pairs(name: str) -> list[list[int]]:
                    v = test_data.get(name)
                    if v is None or len(v) != n:
                        return [[0, 0] for _ in range(n)]
                    out: list[list[int]] = []
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            out.append([int(item[0]), int(item[1])])
                        else:
                            out.append([0, 0])
                    return out

                seq_se = _col_pairs("seq_start_end")
                row_metadata = [
                    {
                        "chrom": _col_str("seq_id", "")[i],
                        "start": seq_se[i][0],
                        "end": seq_se[i][1],
                        "variety": _col_str("variety", "")[i],
                    }
                    for i in range(n)
                ]
                y_pred = (y_prob >= threshold).astype(int)
                test_result_df = generate_results_df(
                    row_metadata=row_metadata,
                    labels=y_pred,
                    probs=y_prob,
                    ground_truth_label=y_test.cpu().numpy(),
                )
                os.makedirs(file_output_path.parent, exist_ok=True)
                test_result_df.to_csv(file_output_path, sep="\t", index=False)

                out_files.append(file_output_path)
            print(f"{get_indent('TEST')} Layer done: dataset={ds_tag}, layer={layer}, saved={file_output_path}")
        print(f"{get_indent('TEST')} Dataset done: {dataset_name}")

    if new_metrics:
        df_new = pd.DataFrame(new_metrics).sort_values(["task", "layer"])
        write_header = not result_path.exists()
        df_new.to_csv(
            result_path,
            sep="\t",
            index=False,
            mode="a" if result_path.exists() else "w",
            header=write_header,
        )
        print(f"{get_indent('TEST')} Appended metrics: file={result_path}, appended_rows={len(df_new)}")
    else:
        print(f"{get_indent('TEST')} No new metrics to append: file={result_path}")
    print(f"{get_indent('TEST')} Stage done: outputs={len(out_files)}")
    print("=" * 50)
    return out_files


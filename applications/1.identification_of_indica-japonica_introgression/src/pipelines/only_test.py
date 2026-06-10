from __future__ import annotations

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.classifiers.rf import load_rf_models, predict_by_models
from src.common.config import Config
from src.evaluation.report import build_training_metrics, generate_results_df
from src.io.paths import resolve
from src.common.schema import get_indent


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return []


def _format_threshold(threshold: float) -> str:
    return f"{threshold:.4f}".rstrip("0").rstrip(".")


def _col_str(data: dict, key: str, n: int, default: str = "") -> list[str]:
    value = data.get(key)
    if value is None or not isinstance(value, list) or len(value) != n:
        return [default] * n
    return [str(v) for v in value]


def _col_pairs(data: dict, key: str, n: int) -> list[list[int]]:
    value = data.get(key)
    if value is None or not isinstance(value, list) or len(value) != n:
        return [[0, 0] for _ in range(n)]
    out: list[list[int]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append([int(item[0]), int(item[1])])
        else:
            out.append([0, 0])
    return out


def _row_metadata(test_data: dict, n: int) -> list[dict]:
    seq_se = _col_pairs(test_data, "seq_start_end", n)
    return [
        {
            "chrom": _col_str(test_data, "seq_id", n)[i],
            "start": seq_se[i][0],
            "end": seq_se[i][1],
            "variety": _col_str(test_data, "variety", n)[i],
            "source": _col_str(test_data, "source", n)[i],
            "source_group": _col_str(test_data, "source_group", n)[i],
        }
        for i in range(n)
    ]


def only_test_pipeline(config: Config) -> list[Path]:
    if not hasattr(config, "test"):
        raise ValueError("Missing `test` section in only_test config")

    classifier_paths = [resolve(p) for p in _as_list(getattr(config.test, "classifier_paths", []))]
    embedding_paths = [resolve(p) for p in _as_list(getattr(config.test, "embedding_paths", []))]
    if not classifier_paths or not embedding_paths:
        raise ValueError("`test.classifier_paths` and `test.embedding_paths` must be non-empty")

    output_dir = resolve(getattr(config.test, "output_dir", "results_path/only_test"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pairing = str(getattr(config.test, "pairing", "cross")).strip().lower()
    if pairing not in {"cross", "zip"}:
        raise ValueError("`test.pairing` must be `cross` or `zip`")
    if pairing == "zip" and len(classifier_paths) != len(embedding_paths):
        raise ValueError("`test.pairing=zip` requires equal lengths of classifier_paths and embedding_paths")

    thresholds = getattr(config.test, "thresholds", None)
    if thresholds is None:
        threshold = float(getattr(config.test, "threshold", 0.5))
        thresholds = [threshold]
    thresholds = [float(t) for t in thresholds]
    if not thresholds:
        thresholds = [0.5]

    cat_dim = int(getattr(config.test, "pooled_embeddings_cat_dim", 0))
    device = str(getattr(config.test, "device", "cpu"))

    pairs: list[tuple[Path, Path]] = []
    if pairing == "zip":
        pairs.extend(zip(classifier_paths, embedding_paths))
    else:
        for clf in classifier_paths:
            for emb in embedding_paths:
                pairs.append((clf, emb))

    summary_rows: list[dict] = []
    outputs: list[Path] = []
    print(
        f"{get_indent('ONLY_TEST')} Start: pairs={len(pairs)}, thresholds={thresholds}, "
        f"output_dir={output_dir}, pairing={pairing}"
    )

    for model_path, embed_path in pairs:
        if not model_path.exists():
            raise FileNotFoundError(f"Classifier file not found: {model_path}")
        if not embed_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embed_path}")
        print(f"{get_indent('ONLY_TEST')} Pair start: model={model_path.name}, embedding={embed_path.name}")
        test_data = torch.load(embed_path, map_location=device)
        if "embeddings" not in test_data or "labels" not in test_data:
            raise ValueError(f"Embedding file missing required keys `embeddings`/`labels`: {embed_path}")

        x_test = test_data["embeddings"]
        y_test = test_data["labels"]
        rf_models = load_rf_models(model_path)
        _, y_prob = predict_by_models(x_test, rf_models, cat_dim=cat_dim)
        y_true_np = y_test.cpu().numpy()
        n = int(y_test.shape[0])
        row_meta = _row_metadata(test_data, n)


        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            pred_df = generate_results_df(
                row_metadata=row_meta,
                labels=y_pred,
                probs=y_prob,
                ground_truth_label=y_true_np,
            )
            out_name = (
                f"{embed_path.stem}"
                f"_thr{_format_threshold(thr)}.tsv"
            )
            out_file = output_dir / model_path.stem / out_name
            os.makedirs(out_file.parent, exist_ok=True)
            pred_df.to_csv(out_file, sep="\t", index=False)
            outputs.append(out_file)

            metrics = build_training_metrics(
                dataset_name=embed_path.stem,
                layer=-1,
                y_prob=y_prob,
                y_true=y_true_np,
                model_name=model_path.stem,
                threshold=thr,
            )
            metrics["classifier_path"] = str(model_path)
            metrics["embedding_path"] = str(embed_path)
            metrics["output_file"] = str(out_file)
            summary_rows.append(metrics)
            print(
                f"{get_indent('ONLY_TEST')} Pair done: model={model_path.name}, embedding={embed_path.name}, "
                f"threshold={thr}, output={out_file.name}"
            )

    summary_path = output_dir / "summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    outputs.append(summary_path)
    print(f"{get_indent('ONLY_TEST')} Done: files={len(outputs)}, summary={summary_path}")
    return outputs


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics_by_threshold


def build_training_metrics(
    dataset_name: str,
    layer: int,
    y_prob: np.ndarray,
    y_true: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    base = compute_metrics_by_threshold(y_prob, y_true, threshold)
    if layer == -1:
        return {
            "task": dataset_name,
            "classifier": model_name,
            "accuracy": base["all"]["accuracy"],
            "roc_auc": base["all"]["auc_roc"],
            "precision": base["all"]["precision"],
            "recall": base["all"]["recall"],
            "f1": base["all"]["f1"],
            "mcc": base["all"]["mcc"],
            "filtered_accuracy": base["filtered"]["accuracy"],
            "filtered_roc_auc": base["filtered"]["auc_roc"],
            "filtered_precision": base["filtered"]["precision"],
            "filtered_recall": base["filtered"]["recall"],
            "filtered_f1": base["filtered"]["f1"],
            "filtered_mcc": base["filtered"]["mcc"],
            "filtered_samples": base["filtered_samples"],
            "total_samples": base["total_samples"],
            "threshold": base["threshold"],
        }
    else:
        return {
            "task": dataset_name,
            "classifier": model_name,
            "layer": layer,
            "accuracy": base["all"]["accuracy"],
            "roc_auc": base["all"]["auc_roc"],
            "precision": base["all"]["precision"],
            "recall": base["all"]["recall"],
            "f1": base["all"]["f1"],
            "mcc": base["all"]["mcc"],
            "filtered_accuracy": base["filtered"]["accuracy"],
            "filtered_roc_auc": base["filtered"]["auc_roc"],
            "filtered_precision": base["filtered"]["precision"],
            "filtered_recall": base["filtered"]["recall"],
            "filtered_f1": base["filtered"]["f1"],
            "filtered_mcc": base["filtered"]["mcc"],
            "filtered_samples": base["filtered_samples"],
            "total_samples": base["total_samples"],
            "threshold": base["threshold"],
        }


def write_tsv(rows: list[dict], path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    return out


def checkout_training_results(result_path):
    existing_metrics: dict[tuple[str, int], dict] = {}
    if result_path.exists():
        existing_df = pd.read_csv(result_path, sep="\t")
        for _, row in existing_df.iterrows():
            existing_metrics[(str(row["task"]), str(row["classifier"]), int(row["layer"]), float(row["threshold"]))] = row.to_dict()
    return existing_metrics



def _cell_for_metadata(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, int) for x in value):
        return f"{value[0]}-{value[1]}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def generate_results_df(
    row_metadata: list[dict[str, Any]],
    labels: np.ndarray,
    probs: np.ndarray,
    ground_truth_label: np.ndarray,
) -> pd.DataFrame:
    """One dict per sample (e.g. sequence, variety, source, …); prediction columns are appended."""
    rows: list[dict[str, Any]] = []
    for meta, label, prob, gt_label in zip(row_metadata, labels, probs, ground_truth_label):
        row = {k: _cell_for_metadata(v) for k, v in meta.items()}
        row["ground_truth"] = ",".join(map(str, gt_label.tolist()))
        row["label"] = ",".join(map(str, label.tolist()))
        row["prob"] = ",".join(f"{float(x):.6f}" for x in prob.tolist())
        rows.append(row)

    return pd.DataFrame(rows)
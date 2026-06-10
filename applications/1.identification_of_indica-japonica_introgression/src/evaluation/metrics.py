from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_macro_auc(labels: np.ndarray, logits: np.ndarray) -> float:
    """Compute macro AUC over columns with both positive and negative labels."""
    auc_scores: list[float] = []
    for i in range(labels.shape[1]):
        unique_labels = np.unique(labels[:, i])
        if unique_labels.size < 2:
            continue
        auc_scores.append(roc_auc_score(labels[:, i], logits[:, i]))
    return float(np.mean(auc_scores)) if auc_scores else float("nan")


def _compute_metrics(logits: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> dict:
    """Compute multilabel metrics from probabilities/logits, labels and predictions."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    if not (logits.shape == labels.shape == preds.shape):
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, labels={labels.shape}, preds={preds.shape}"
        )

    def calculate_mcc(inner_labels: np.ndarray, inner_preds: np.ndarray) -> float:
        mcc_scores: list[float] = []
        for i in range(inner_labels.shape[1]):
            mcc_scores.append(matthews_corrcoef(inner_labels[:, i], inner_preds[:, i]))
        return float(np.mean(mcc_scores))

    return {
        "accuracy": accuracy_score(labels, preds),
        "auc_roc": _safe_macro_auc(labels, logits),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "mcc": calculate_mcc(labels, preds),
    }


def compute_metrics_by_threshold(y_probs: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
    """Evaluate metrics from a TSV dataframe with columns: ground_truth, prob."""
    y_probs = np.asarray(y_probs)
    y_true = np.asarray(y_true)
    y_preds = (y_probs >= threshold).astype(int)

    result = _compute_metrics(y_probs, y_true, y_preds)
    filter_mask = ~((y_preds == 0).all(axis=1) | (y_preds == 1).all(axis=1))
    if filter_mask.any():
        result_filter = _compute_metrics(
            y_probs[filter_mask],
            y_true[filter_mask],
            y_preds[filter_mask],
        )
    else:
        result_filter = {
            "accuracy": None,
            "auc_roc": None,
            "f1": None,
            "recall": None,
            "precision": None,
            "mcc": None,
        }

    final_result = {
        "all": result,
        "filtered": result_filter,
        "filtered_samples": int(filter_mask.sum()),
        "total_samples": len(y_true),
        "threshold": threshold,
    }
    return final_result

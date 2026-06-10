from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from src.common.config import Config
from src.io.naming import rf_model_filename
from src.io.paths import result_root


def pool_embeddings(embeddings: torch.Tensor, cat_dim: int) -> torch.Tensor:
    """Pool/reshape embeddings before RF training or prediction."""
    if len(embeddings.shape) == 2:
        return embeddings
    if cat_dim == 1:
        return torch.reshape(embeddings, (embeddings.shape[0], -1))
    return embeddings.mean(dim=1)


def predict_with_rf_models(
    embeddings: np.ndarray,
    rf_models: list[RandomForestClassifier],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict multi-label outputs with one RF model per label."""
    preds_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    for rf in rf_models:
        pred_i = rf.predict(embeddings)
        prob_i = rf.predict_proba(embeddings)[:, 1]
        preds_list.append(pred_i)
        probs_list.append(prob_i)
    probs = np.column_stack(probs_list)
    preds = np.column_stack(preds_list)
    return preds, probs


def train_rf_models(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_estimators: int,
    random_state: int,
) -> list[RandomForestClassifier]:
    x_np = embeddings.cpu().numpy()
    y_np = labels.cpu().numpy()
    models: list[RandomForestClassifier] = []
    for i in range(y_np.shape[1]):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(x_np, y_np[:, i])
        models.append(rf)
    return models


def save_rf_models(
    rf_models: list[RandomForestClassifier],
    rf_path: str | Path,
) -> Path:
    joblib.dump(rf_models, rf_path)
    return rf_path


def load_rf_models(model_path: str | Path) -> list[RandomForestClassifier]:
    return joblib.load(model_path)


def predict_by_models(
    embeddings: torch.Tensor,
    rf_models: list[RandomForestClassifier],
    cat_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_np = pool_embeddings(embeddings, cat_dim).cpu().numpy()
    return predict_with_rf_models(x_np, rf_models)


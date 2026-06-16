import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef
from typing import Dict

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        # similarity
        sim = torch.matmul(features, features.T) / self.temperature

        # ✅ 数值稳定（来自版本2）
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # ✅ 多标签支持（来自版本1）
        if labels.ndim == 2:
            # labels = labels.float()
            # mask = (labels @ labels.T > 0).float()
            labels = labels.float()

            inter = labels @ labels.T

            union = (
                labels.sum(1, keepdim=True)
                + labels.sum(1, keepdim=True).T
                - inter
            )

            mask = inter / (union + 1e-9)
        else:
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float()

        # 去掉自身
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(device)
        mask = mask * logits_mask

        # log_prob
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

        loss = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return loss.mean()


def compute_metrics(eval_pred, multi_label=False, threshold=0.5):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    def calculate_mcc(labels, probs):
        # 处理一维和二维情况
        if labels.ndim == 1:
            return matthews_corrcoef(labels, probs)
        mcc_scores = []
        for i in range(labels.shape[1]):
            mcc_scores.append(matthews_corrcoef(labels[:, i], probs[:, i]))
        mcc = np.mean(mcc_scores)
        return mcc

    if multi_label:
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= threshold).astype(int)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "auc_roc": roc_auc_score(labels, probs, average="macro"),
            "mcc": calculate_mcc(labels, preds)
        }

    if logits.ndim == 2 and logits.shape[1] > 1:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        preds = np.argmax(logits, axis=1)
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "auc_roc": roc_auc_score(labels, probs[:, 1]),
            "mcc": calculate_mcc(labels, preds)
        }

    return {}


def serialize_label_array(arr):
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    if isinstance(arr, (list, tuple, np.ndarray)):
        return ",".join(str(x) for x in np.asarray(arr).ravel())
    return str(arr)


def build_prediction_results_dataframe(logits, labels, num_samples):
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if len(logits) > num_samples:
        logits = logits[:num_samples]
    if len(labels) > num_samples:
        labels = labels[:num_samples]

    n, num_classes = logits.shape[0], logits.shape[1] if logits.ndim > 1 else 1
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    df = pd.DataFrame(
        logits,
        columns=[f"logit_{i}" for i in range(num_classes)],
    )
    df.insert(0, "labels", [serialize_label_array(labels[i]) for i in range(n)])
    df.insert(0, "sample_idx", np.arange(n))
    return df


def load_checkpoint_state_dict(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    pytorch_path = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path)
    if os.path.exists(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu")
    raise FileNotFoundError(f"No model.safetensors/pytorch_model.bin found in: {ckpt_dir}")

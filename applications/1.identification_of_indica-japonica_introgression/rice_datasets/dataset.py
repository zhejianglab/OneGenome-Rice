import os
import torch
from torch.utils.data import Dataset
import json
import numpy as np

class RiceDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, seq_key="sequence", label_key="label", max_length=8192):
        self.input_ids_list = []
        self.attention_masks_list = []
        self.labels_list = []

        # 预处理：在初始化时一次性tokenize所有序列
        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line)
                seq = data.get(seq_key)
                label = data.get(label_key)
                if seq is None or label is None:
                    raise ValueError(f"Missing '{seq_key}' or '{label_key}' in {jsonl_path}")
                
                # Tokenize
                enc = tokenizer(
                    seq,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )
                self.input_ids_list.append(enc["input_ids"].squeeze(0))
                self.attention_masks_list.append(enc["attention_mask"].squeeze(0))
                
                # 处理标签
                if isinstance(label, list):
                    label = np.array(label, dtype=np.float32)
                self.labels_list.append(torch.tensor(label))

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        # O(1)查表操作，无计算成本
        return {
            "input_ids": self.input_ids_list[idx],
            "attention_mask": self.attention_masks_list[idx],
            "labels": self.labels_list[idx]
        }


class EmbeddingsDataset(Dataset):
    """加载预计算嵌入的数据集"""
    def __init__(self, pt_path):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Embedding file not found: {pt_path}")

        data = torch.load(pt_path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(f"Embedding file must contain a dict, got {type(data)}: {pt_path}")

        if "embeddings" not in data or "labels" not in data:
            raise KeyError(
                f"Embedding file must contain 'embeddings' and 'labels' keys: {pt_path}"
            )

        self.embeddings = data["embeddings"]
        self.labels = data["labels"]

        if not torch.is_tensor(self.embeddings):
            raise TypeError(
                f"'embeddings' must be a torch.Tensor, got {type(self.embeddings)}: {pt_path}"
            )
        if not torch.is_tensor(self.labels):
            raise TypeError(
                f"'labels' must be a torch.Tensor, got {type(self.labels)}: {pt_path}"
            )
        if len(self.embeddings) != len(self.labels):
            raise ValueError(
                f"Embeddings and labels length mismatch: {len(self.embeddings)} vs {len(self.labels)}"
            )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "labels": self.labels[idx]
        }


# 工厂函数：统一数据集构建
def create_rice_dataset(dataset_info, split, tokenizer):
    """创建基于token的RiceDataset"""
    jsonl_path = os.path.join(dataset_info["dataset_dir"], f"{split}.jsonl")
    seq_key = dataset_info["seq_key"]
    label_key = dataset_info["label_key"]
    max_length = dataset_info["max_length"]
    return RiceDataset(jsonl_path, tokenizer, seq_key=seq_key, label_key=label_key, max_length=max_length)

def create_embeddings_dataset(embedding_dir, split):
    """创建基于embedding的EmbeddingsDataset"""
    pt_path = os.path.join(embedding_dir, f"{split}.pt")
    return EmbeddingsDataset(pt_path)

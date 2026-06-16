"""
HuggingFace datasets + Arrow 落盘缓存 tokenized jsonl，与 RiceDataset 行为对齐。
需在项目根运行且已安装 datasets、pyarrow。
"""
from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch
from datasets import load_dataset, load_from_disk

if TYPE_CHECKING:
    from datasets import Dataset


META_FILENAME = "token_cache_meta.json"


def tokenizer_config_for_e2e(config: dict) -> dict:
    """
    端到端训练/推理必选：返回顶层 config['tokenizer'] dict。
    必须显式包含 use_arrow_token_cache；force_rebuild_arrow_cache 缺省为 False。
    """
    tok = config.get("tokenizer")
    if not isinstance(tok, dict):
        raise ValueError(
            "End-to-end mode requires a top-level 'tokenizer' mapping in the config YAML "
            "(sibling of 'data', 'embedding', 'train'). "
            "Expected keys include: use_arrow_token_cache (required), "
            "tokenize_dir (required when use_arrow_token_cache is true), "
            "force_rebuild_arrow_cache (optional, default false)."
        )
    if "use_arrow_token_cache" not in tok:
        raise ValueError(
            "tokenizer.use_arrow_token_cache must be set explicitly in the config (true or false)."
        )
    return tok


def _dataset_dir_abs(dataset_info: dict) -> str:
    return os.path.abspath(dataset_info["dataset_dir"])


def _tokenize_base_abs(config: dict, project_root: str) -> str:
    tok_cfg = tokenizer_config_for_e2e(config)
    raw = tok_cfg.get("tokenize_dir")
    if raw is None or str(raw).strip() == "":
        raise ValueError(
            "tokenizer.tokenize_dir is required when tokenizer.use_arrow_token_cache is true. "
            "Set a non-empty directory; caches are stored under "
            "{tokenize_dir}/{model_name}/{dataset_name}/<split>/. "
            "Relative paths are resolved against project_root."
        )
    s = os.path.expanduser(str(raw).strip())
    if os.path.isabs(s):
        return os.path.abspath(s)
    return os.path.abspath(os.path.join(project_root, s))


def _split_cache_dir(config: dict, project_root: str, split_key: str) -> str:
    base = _tokenize_base_abs(config, project_root)
    model_name = config["model"]["model_name"]
    dataset_name = config["data"]["dataset_name"]
    return os.path.join(base, model_name, dataset_name, split_key)


def _arrow_ready(split_dir: str) -> bool:
    return os.path.isfile(os.path.join(split_dir, "dataset_info.json"))


def _labels_to_serializable(lab: Any):
    if isinstance(lab, list):
        return [float(x) for x in lab]
    if isinstance(lab, bool):
        return int(lab)
    if isinstance(lab, int):
        return lab
    if isinstance(lab, float):
        return lab
    return lab


def load_or_build_arrow_split(
    config: dict,
    dataset_info: dict,
    tokenizer,
    split_key: str,
    *,
    project_root: str,
    force_rebuild: bool = False,
    map_batch_size: int = 256,
    num_proc: int = 4,
) -> "Dataset":
    """
    返回 HuggingFace Dataset（含 input_ids, attention_mask, labels 列）。

    要求 tokenizer.use_arrow_token_cache 为 true 时 tokenizer.tokenize_dir 非空；缓存目录为：
    {tokenize_dir_abs}/{model_name}/{dataset_name}/{split_key}/

    jsonl 仍来自 dataset_info["dataset_dir"]/{split_key}.jsonl。
    project_root：用于将相对 tokenize_dir 解析为绝对路径。
    """
    jsonl_path = os.path.join(_dataset_dir_abs(dataset_info), f"{split_key}.jsonl")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Missing jsonl for split {split_key!r}: {jsonl_path}")

    split_dir = _split_cache_dir(config, project_root, split_key)

    if not force_rebuild and _arrow_ready(split_dir):
        return load_from_disk(split_dir)

    print(f"Building Arrow token cache: {split_dir}")
    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)

    seq_key = dataset_info["seq_key"]
    label_key = dataset_info["label_key"]
    max_length = int(dataset_info["max_length"])
    raw = load_dataset("json", data_files=str(jsonl_path), split='train')

    def _tok_batch(examples: Dict[str, list]):
        enc = tokenizer(
            examples[seq_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        labels_out = [_labels_to_serializable(lab) for lab in examples[label_key]]
        enc["labels"] = labels_out
        return enc

    tok = raw.map(
        _tok_batch,
        batched=True,
        batch_size=map_batch_size,
        remove_columns=raw.column_names,
        num_proc=num_proc 
    )

    os.makedirs(split_dir, exist_ok=True)
    tok.save_to_disk(split_dir)
    return load_from_disk(split_dir)


def make_token_batch_collator() -> Callable:
    """将 HF Dataset 行（list 字段）堆叠为与 RiceDataset + DataLoader 一致的 tensor batch。"""

    def collate(features: list) -> dict:
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        }
        labs = [f["labels"] for f in features]
        if len(labs) == 0:
            raise ValueError("empty batch")
        if isinstance(labs[0], list):
            batch["labels"] = torch.tensor(labs, dtype=torch.float32)
        else:
            batch["labels"] = torch.tensor(labs, dtype=torch.long)
        return batch

    return collate

# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""Load Hugging Face base model and tokenizer from training config."""

import torch
from transformers import AutoModel, AutoTokenizer

from model.distributed import dist_print


def load_model_and_tokenizer(cfg: dict):
    model_path = cfg["model_path"]

    model_kwargs = dict(
        trust_remote_code=True,
        revision="main",
        attn_implementation="flash_attention_2",
    )
    dtype_str = cfg.get("model_torch_dtype")
    if dtype_str:
        model_kwargs["torch_dtype"] = getattr(torch, dtype_str)

    tok_kwargs = dict(
        trust_remote_code=True,
        revision="main",
        padding_side="right",
    )
    tok_dtype_str = cfg.get("tokenizer_torch_dtype")
    if tok_dtype_str:
        tok_kwargs["torch_dtype"] = getattr(torch, tok_dtype_str)

    dist_print(">>> Loading pretrained model and tokenizer...")
    base_model = AutoModel.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    return base_model, tokenizer

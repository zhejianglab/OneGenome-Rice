from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from src.common.config import Config
from src.io.paths import model_dir


def build_device(config: Config) -> str:
    preferred = config.environment.get("device", "cuda")
    if preferred == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return preferred


def load_model_and_tokenizer(config: Config) -> tuple[object, object, str]:
    device = build_device(config)
    model_path = model_dir(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).eval()
    if device == "cpu":
        model = model.to(device)
    return model, tokenizer, device


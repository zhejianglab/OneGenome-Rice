from __future__ import annotations

from datetime import datetime

from src.common.config import Config
from datetime import datetime

def normalize_and_validate_config(config: Config) -> Config:
    """Normalize unified config schema for domain pipelines."""
    required_top = ["model", "data_process", "embedding", "environment", "classifiers", "output"]
    for key in required_top:
        if not hasattr(config, key):
            raise ValueError(f"Missing required top-level section: {key}")

    if not hasattr(config.data_process, "datasets"):
        raise ValueError("Missing `data_process.datasets`")

    # Ensure embedding.layers exists and is list[int]
    if not hasattr(config.embedding, "layers"):
        config.embedding.layers = [config.model.hidden_layers]
    if not isinstance(config.embedding.layers, list):
        raise ValueError("`embedding.layers` must be a list")
    if not config.embedding.layers:
        raise ValueError("`embedding.layers` must not be empty")

    # Ensure classifier defaults
    if not hasattr(config.classifiers, "RF"):
        config.classifiers.RF = Config({"n_estimators": 100, "random_state": 42})
    if not hasattr(config.classifiers, "prediction"):
        config.classifiers.prediction = Config({"threshold": 0.5})

    # Ensure run defaults
    if not hasattr(config, "run"):
        config.run = Config({"name": "default", "isolate_embeddings": False})
    if not hasattr(config.run, "name"):
        config.run.name = "default"
    if not hasattr(config.run, "isolate_embeddings"):
        config.run.isolate_embeddings = False
    if not isinstance(config.run.isolate_embeddings, bool):
        raise ValueError("`run.isolate_embeddings` must be bool")
    if str(config.run.name).strip().lower() == "auto":
        config.run.name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not str(config.run.name).strip():
        raise ValueError("`run.name` must be non-empty")

    # Optional test-stage path overrides (empty string = use defaults)
    if not hasattr(config, "test"):
        config.test = Config({})
    if not hasattr(config.test, "classifier_path"):
        config.test.classifier_path = ""
    if not hasattr(config.test, "embedding_path"):
        config.test.embedding_path = ""

    # Ensure each dataset split has required keys
    for ds_name, ds_cfg in config.data_process.datasets.__dict__.items():
        if not hasattr(ds_cfg, "label"):
            raise ValueError(f"Dataset `{ds_name}` missing label section")
        label_keys = list(ds_cfg.label.__dict__.keys())
        if not label_keys:
            raise ValueError(f"Dataset `{ds_name}` label section is empty")

        for split in ["train", "test"]:
            if not hasattr(ds_cfg, split):
                raise ValueError(f"Dataset `{ds_name}` missing `{split}` section")
            split_cfg = getattr(ds_cfg, split)
            if not hasattr(split_cfg, "window_size") or not hasattr(split_cfg, "step_size"):
                raise ValueError(f"Dataset `{ds_name}` `{split}` missing window_size or step_size")
            if split_cfg.window_size <= 0 or split_cfg.step_size <= 0:
                raise ValueError(f"Dataset `{ds_name}` `{split}` has non-positive window/step size")

            split_keys = set(split_cfg.__dict__.keys())
            meta_keys = {"window_size", "step_size", "unique_windows"}
            split_label_keys = split_keys - meta_keys
            if set(label_keys) != split_label_keys:
                raise ValueError(
                    f"Dataset `{ds_name}` `{split}` label keys mismatch: "
                    f"labels={sorted(label_keys)}, split={sorted(split_label_keys)}"
                )

            for cls_key in label_keys:
                cls_files = getattr(split_cfg, cls_key)
                if not isinstance(cls_files, list) or not cls_files:
                    raise ValueError(
                        f"Dataset `{ds_name}` `{split}` class `{cls_key}` must be a non-empty file list"
                    )

    return config



# 定义层级关系，key是名称，value是层级数
LEVEL_MAP = {
    "PIPELINE": 0,
    "PREPARE": 1,
    "EXTRACT": 1,
    "TRAIN": 1,
    "TEST": 1,
    "ONLY_TEST": 0
}

def get_indent(name):
    """
    生成缩进字符串
    """
    level = LEVEL_MAP.get(name, 0)
    indent_str = " " * (level * 4)
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{indent_str}{now_time} [{name}] "
#!/usr/bin/env python3
"""离线构建端到端 Arrow token 缓存（与 scripts/train.py 中 load_or_build_arrow_split 一致）。"""
import argparse
import os
import sys
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rice_datasets.arrow_cache import load_or_build_arrow_split, tokenizer_config_for_e2e
from scripts.run_train import read_config, path_template, read_dataset_info


def main(config_path, split_mode):
    config = read_config(config_path)
    if config.get("embedding", {}).get("use_extracted_embeddings", False):
        raise SystemExit(
            "embedding.use_extracted_embeddings is true; Arrow token cache applies only to end-to-end mode.")
    tok_cfg = tokenizer_config_for_e2e(config)
    if not tok_cfg["use_arrow_token_cache"]:
        raise SystemExit(
            "tokenizer.use_arrow_token_cache is false; preprocess_tokenize only builds Arrow caches.")

    paths = path_template(config)
    dataset_info = read_dataset_info(config["data"]["dataset_name"], paths)
    tokenizer = AutoTokenizer.from_pretrained(
        paths["model_path"], trust_remote_code=True, token=config["model"]["token"]
    )
    split_names = config["data"]["split_name"]
    force_rebuild = tok_cfg["force_rebuild_arrow_cache"]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for role in split_mode:
        load_or_build_arrow_split(
            config,
            dataset_info,
            tokenizer,
            split_names[role],
            project_root=repo_root,
            force_rebuild=force_rebuild,
        )
        print(f"OK split={split_names[role]!r}")


if __name__ == "__main__":
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _default_cfg = os.path.join(_repo_root, "configs", "config_tuning.yaml")

    parser = argparse.ArgumentParser(
        description="Build HuggingFace Arrow token caches for train/eval/test jsonl.")
    parser.add_argument(
        "--config",
        type=str,
        default=_default_cfg,
        help="Training yaml (same as train.py).",
    )
    parser.add_argument("--split-mode", type=str, default="train,eval,test", help="Split mode to tokenize.")
    args = parser.parse_args()

    if not os.path.isabs(args.config):
        args.config = os.path.join(_repo_root, args.config)
    split_mode_list = [mode.strip() for mode in args.split_mode.split(",")]
    main(args.config, split_mode_list)

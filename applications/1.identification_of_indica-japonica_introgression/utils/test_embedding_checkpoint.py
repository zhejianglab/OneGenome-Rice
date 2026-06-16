import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from rice_datasets.dataset import create_embeddings_dataset
from models.model import EmbeddingModel
from utils.trainer_utils import compute_metrics, load_checkpoint_state_dict


def load_full_checkpoint(model: torch.nn.Module, checkpoint_dir: str) -> str:
    state_dict = load_checkpoint_state_dict(checkpoint_dir)
    source = os.path.join(checkpoint_dir, "model.safetensors") if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) else os.path.join(checkpoint_dir, "pytorch_model.bin")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint keys mismatch: "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    return source


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test-only script for embedding model checkpoint on a specific test.pt."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default= "results/rice_1B_stage2_8k_hf/varieties_classification_jap1-ind1_8k_8k_20260326/12layer/0_lossRatio-0.1_lr-cosine_noBN_batchSize-256_epoch-1000/checkpoints/checkpoint-100521",
        help="Checkpoint directory containing model.safetensors or pytorch_model.bin.",
    )
    parser.add_argument(
        "--test-pt",
        default="results/rice_1B_stage2_8k_hf/varieties_classification_jap1-ind1_8k_8k_20260326/12layer/embeddings/test.pt",
        help="Path to test.pt containing {'embeddings', 'labels'}.",
    )
    parser.add_argument(
        "--use-batchnorm",
        action="store_true",
        default=False,
        help="Whether to use batch normalization in projection head.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.69,
        help="Sigmoid threshold used for multi-label metrics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--output-json",
        default="results/rice_1B_stage2_8k_hf/varieties_classification_jap1-ind1_8k_8k_20260326/12layer/0_lossRatio-0.1_lr-cosine_noBN_batchSize-256_epoch-1000/metrics/test_metrics1.json",
        help="Path to save metrics JSON. If not provided, will not save.",
    )
    parser.add_argument(
        "--proj-dims",
        nargs="+",
        type=int,
        default=[1024, 512, 128],
        help="Projection dimensions, e.g. 1024 512 128.",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of classification labels.",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        default=True,
        help="Whether this is a multi-label classification task.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = (
        args.checkpoint_dir
        if os.path.isabs(args.checkpoint_dir)
        else os.path.join(PROJECT_ROOT, args.checkpoint_dir)
    )
    test_pt = args.test_pt if os.path.isabs(args.test_pt) else os.path.join(PROJECT_ROOT, args.test_pt)
    output_json = None
    if args.output_json:
        output_json = (
            args.output_json if os.path.isabs(args.output_json) else os.path.join(PROJECT_ROOT, args.output_json)
        )
    
    # 使用 create_embeddings_dataset 工厂函数
    embedding_dir = os.path.dirname(test_pt)
    split = os.path.splitext(os.path.basename(test_pt))[0]
    dataset = create_embeddings_dataset(embedding_dir, split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 使用命令行参数创建模型
    model = EmbeddingModel(
        proj_dims=args.proj_dims,
        num_labels=args.num_labels,
        is_bn=args.use_batchnorm,
        dropout=0.02,
    )
    ckpt_source = load_full_checkpoint(model, checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device).float()
            labels = batch["labels"].cpu().numpy()
            _, logits = model(embeddings)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(
        (all_logits, all_labels),
        multi_label=args.multi_label,
        threshold=args.threshold,
    )

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        
        # 创建包含元数据的结果字典
        result = metrics
        result.update({
            "time": datetime.now().isoformat(),
            "checkpoint": checkpoint_dir,
            "threshold": args.threshold
        })

        
        # 读取现有的JSON文件（如果存在）
        results = []
        if os.path.exists(output_json):
            try:
                with open(output_json, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    # 处理两种可能的格式：列表或单个字典
                    if isinstance(existing_data, list):
                        results = existing_data
                    else:
                        results = [existing_data]
            except (json.JSONDecodeError, IOError):
                results = []
        
        # 追加新结果
        results.append(result)
        
        # 保存更新后的结果
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("========== Test-only Metrics ==========")
    print(f"checkpoint: {checkpoint_dir}")
    print(f"checkpoint_file: {ckpt_source}")
    print(f"test_pt: {test_pt}")
    print(f"proj_dims: {args.proj_dims}")
    print(f"num_labels: {args.num_labels}")
    print(f"use_batchnorm: {args.use_batchnorm}")
    print(f"threshold: {args.threshold}")
    print(f"multi_label: {args.multi_label}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    if output_json:
        print(f"saved_json: {output_json}")
    print("======================================")


if __name__ == "__main__":
    main()

import os
import sys
import time
import argparse
import json
import torch
import yaml
import random
import numpy as np
from datetime import datetime
from transformers import TrainingArguments, AutoTokenizer, set_seed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import format_time
from utils.trainer_utils import compute_metrics, build_prediction_results_dataframe
from rice_datasets.arrow_cache import (
    load_or_build_arrow_split,
    make_token_batch_collator,
    tokenizer_config_for_e2e,
)
from rice_datasets.dataset import create_rice_dataset, create_embeddings_dataset
from trainer.trainer import ContrastiveTrainer
from models.model import FullModel, EmbeddingModel


# =========================
# 1. 基础配置与环境清理
# =========================
SEED = 42
set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def is_distributed_env():
    """是否为 accelerate/torchrun 多进程启动（非单进程残留环境变量）。"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size() > 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return (
        world_size > 1
        and os.environ.get("LOCAL_RANK", "-1") != "-1"
        and os.environ.get("MASTER_ADDR") is not None
    )


def _clear_stale_distributed_env():
    """单进程推理时清理残留的分布式环境变量；真实多卡启动时保留。"""
    if is_distributed_env():
        return
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        os.environ.pop(var, None)


_clear_stale_distributed_env()


def read_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = yaml.safe_load(os.path.expandvars(yaml.dump(cfg)))
    return cfg


def path_template(config):

    if config["data"].get("data_dir") is None:
        data_dir = "data"
    else:
        data_dir = config["data"]["data_dir"]

    values = {
        "model_name": config["model"]["model_name"],
        "dataset_name": config["data"]["dataset_name"],
    }

    if config["train"].get("output_dir"):
        result_dir = config["train"]["output_dir"]
    else:
        result_dir = "results"

    use_extracted_embeddings = config["embedding"].get(
        "use_extracted_embeddings", False)

    if use_extracted_embeddings:
        layer = config["embedding"].get("layer", 12)
        values["layer"] = layer
        layer_path = os.path.join(
            result_dir, values["model_name"], values['dataset_name'], f"{layer}layer")
        if not config["embedding"].get("embedding_dir"):
            embedding_path = os.path.join(layer_path, "embeddings")
        else:
            embedding_path = config["embedding"]["embedding_dir"]
    else:
        layer_path = os.path.join(
            result_dir, values["model_name"], values['dataset_name'])
        embedding_path = None

    experiment_name = config["train"].get(
        "run_name", time.strftime("%Y%m%d-%H%M%S"))
    experiment_dir = os.path.join(layer_path, experiment_name)
    metrics_dir = os.path.join(experiment_dir, "metrics")

    os.makedirs(metrics_dir, exist_ok=True)


    return {
        'data_dir': data_dir,
        'model_path': config["model"]["model_path"],
        'result_layer_path': layer_path,
        'embedding_path': embedding_path,
        'result_metrics_dir': metrics_dir,
        'use_extracted_embeddings': use_extracted_embeddings
    }


def read_dataset_info(dataset_name, paths):
    info_path = os.path.join(paths['data_dir'], "datasets_info.yaml")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"datasets_info.yaml not found at: {info_path}")
    with open(info_path) as f:
        datasets_info = yaml.safe_load(f)
    if dataset_name not in datasets_info.get("dataset_feature", {}):
        raise ValueError(f"Dataset '{dataset_name}' not found in {info_path}")
    dataset_info = datasets_info["dataset_feature"][dataset_name]
    return {
        "dataset_dir": os.path.join(paths['data_dir'], dataset_name),
        "seq_key": dataset_info.get("seq_key", "sequence"),
        "label_key": dataset_info.get("label_key", "label"),
        "max_length": dataset_info.get("max_length", 8192),
        "eval_task": dataset_info.get("eval_task", "labels"),
        "split_name": dataset_info.get("split_name", {"train": "train", "eval": "eval", "test": "test"})
    }


def dataset_tokenize(config, dataset_info, paths):
    use_extracted_embeddings = paths['use_extracted_embeddings']

    if use_extracted_embeddings:
        if paths['embedding_path'] is None:
            raise ValueError(
                "embedding_path is None when use_extracted_embeddings is True")
        # 加载预计算的 Embedding
        test_ds = create_embeddings_dataset(
            paths['embedding_path'], config['embedding']['split_name']['test'])
        return test_ds, None

    # 端到端模式：Tokenize
    token = config["model"]["token"]
    tokenizer = AutoTokenizer.from_pretrained(
        paths["model_path"],
        trust_remote_code=True,
        token=token
    )
    split_names = dataset_info.get("split_name", config['data'].get(
        'split_name', {'train': 'train', 'eval': 'eval', 'test': 'test'}))

    tok_cfg = tokenizer_config_for_e2e(config)
    use_arrow = bool(tok_cfg["use_arrow_token_cache"])
    force_rebuild = bool(tok_cfg.get("force_rebuild_arrow_cache", False))

    if use_arrow:
        _repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        test_ds = load_or_build_arrow_split(
            config, 
            dataset_info, 
            tokenizer, 
            split_names["test"],
            project_root=_repo_root, 
            force_rebuild=force_rebuild
        )
        return test_ds, make_token_batch_collator()

    test_ds = create_rice_dataset(dataset_info, split_names['test'], tokenizer)
    return test_ds, None


def peft_lora_dict_for_backbone(config):
    lora = config.get("lora")
    if not lora or not lora.get("use_lora", False):
        return None
    return {k: v for k, v in lora.items() if k != "use_lora"}


def model(config, paths):
    use_extracted_embeddings = paths['use_extracted_embeddings']
    head_dropout = float(config["train"].get("dropout", 0.1))

    if use_extracted_embeddings:
        model_obj = EmbeddingModel(
            proj_dims=config["projection"]["dims"],
            num_labels=config["task"]["num_labels"],
            is_bn=config["projection"].get("use_batchnorm", False),
            dropout=head_dropout,
        )
    else:
        model_obj = FullModel(
            model_path=config["model"]["model_path"],
            proj_dims=config["projection"]["dims"],
            num_labels=config["task"]["num_labels"],
            token=config["model"]["token"],
            lora_config=peft_lora_dict_for_backbone(config),
            pooling_strategy=config["model"].get(
                "pooling_strategy", "masked_mean"),
            dropout=head_dropout,
            is_bn=config["projection"].get("use_batchnorm", False),
        )
    return model_obj


def inference_args(config, paths):
    dataloader_num_workers = int(
        config["train"].get("dataloader_num_workers", 0))
    dataloader_persistent_workers = bool(
        config["train"].get("dataloader_persistent_workers",
                            dataloader_num_workers > 0)
    )
    if dataloader_num_workers == 0 and dataloader_persistent_workers:
        dataloader_persistent_workers = False
    if is_distributed_env():
        # 多卡时限制 worker 数，避免 num_processes * num_workers 过大
        dataloader_num_workers = min(dataloader_num_workers, 8)
        dataloader_persistent_workers = False

    fp16 = bool(config["train"].get("fp16", True))
    bf16 = bool(config["train"].get("bf16", False))
    if fp16 and bf16:
        fp16 = False
        print("Warning: Both fp16 and bf16 are enabled. Disabling fp16 in favor of bf16.")

    ddp_find_unused_parameters = config["train"].get(
        "ddp_find_unused_parameters")
    ddp_backend = config["train"].get("ddp_backend")

    training_args_kwargs = dict(
        output_dir=paths["result_layer_path"],
        per_device_eval_batch_size=int(config["train"]["batch_size"]),
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=bool(
            config["train"].get("dataloader_pin_memory",
                                torch.cuda.is_available())
        ),
        dataloader_persistent_workers=dataloader_persistent_workers,
        remove_unused_columns=False,
        report_to=[],
        seed=SEED,
        logging_steps=10,
        disable_tqdm=is_distributed_env() and int(
            os.environ.get("LOCAL_RANK", "0")) != 0,
        eval_do_concat_batches=True,
    )

    if is_distributed_env():
        if ddp_find_unused_parameters is not None:
            training_args_kwargs["ddp_find_unused_parameters"] = bool(
                ddp_find_unused_parameters)
        if ddp_backend:
            training_args_kwargs["ddp_backend"] = ddp_backend

    return TrainingArguments(**training_args_kwargs)


def create_trainer(config, model, args, test_ds, data_collator=None):
    threshold = config["train"]["sigmoid_threshold"]
    task_class = config["task"]["task_class"]

    if task_class == "labels":
        multi_label = True
    elif task_class == "classification":
        multi_label = False
    elif task_class == "regression":
        multi_label = False
    else:
        raise ValueError(f"Unsupported task_class: {task_class}")

    def metric_fn(eval_pred): return compute_metrics(
        eval_pred, multi_label=multi_label, threshold=threshold)

    trainer = ContrastiveTrainer(
        model=model,
        args=args,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=metric_fn,
        multi_label=multi_label,
        lambda_contrastive=config["train"]["lambda_contrastive"],
    )
    return trainer


def _load_checkpoint(trainer, checkpoint_dir):
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        print("No checkpoint directory provided or not found. Using initialized weights.")
        return

    print(f"Loading checkpoint from: {checkpoint_dir}")
    safetensor_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(safetensor_path):
        from safetensors.torch import load_model
        load_model(trainer.model, safetensor_path)
        print("Model weights loaded successfully using safetensors.")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully using torch.load.")
    else:
        raise FileNotFoundError(
            f"No model weights (safetensors or bin) found in {checkpoint_dir}")


def run_inference(trainer, test_ds, paths, config, checkpoint_dir):
    threshold = config["train"]["sigmoid_threshold"]
    _load_checkpoint(trainer, checkpoint_dir)
    trainer.model.eval()

    num_samples = len(test_ds)
    if trainer.is_world_process_zero():
        print(f"\n{'='*60}")
        print("Starting Inference...")
        print(
            f"Samples: {num_samples} | Batch Size: {trainer.args.per_device_eval_batch_size}"
            f" | World Size: {trainer.args.world_size}"
        )
        print(f"{'='*60}\n")

    with torch.no_grad():
        predictions = trainer.predict(test_ds)

    if not trainer.is_world_process_zero():
        return

    logits = predictions.predictions
    labels = predictions.label_ids
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logits_path = os.path.join(
        paths["result_metrics_dir"], f"inference_results_{timestamp}.tsv")
    df_results = build_prediction_results_dataframe(
        logits, labels, num_samples)
    df_results.to_csv(logits_path, sep="\t", index=False)
    print(
        f"\nSaved logits to: {logits_path} ({len(df_results)} rows, with sample_idx)")

    metrics = predictions.metrics
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    result = {
        "accuracy": metrics.get("test_accuracy"),
        "f1": metrics.get("test_f1"),
        "recall": metrics.get("test_recall"),
        "precision": metrics.get("test_precision"),
        "auc_roc": metrics.get("test_auc_roc"),
        "mcc": metrics.get("test_mcc"),
        "num_samples": num_samples,
        "time": datetime.now().isoformat(),
        "checkpoint": checkpoint_dir,
        "threshold": threshold,
        "runtime": metrics.get("test_runtime"),
        "world_size": trainer.args.world_size,
    }
    metrics_path = os.path.join(
        paths["result_metrics_dir"], f"test_metrics_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved result to: {metrics_path}")
    print(f"{'='*60}\n")


def main(config_path, checkpoint_dir):
    start_time = time.time()
    print(f"\n{'+'*30} Starting Inference Pipeline... {'+'*30}\n")

    # 1. 读取配置
    print(f"\n{'+'*30} Step 1/5: Reading configuration... {'+'*30}")
    config = read_config(config_path)

    # 2. 设置路径
    print(f"\n{'+'*30} Step 2/5: Setting up paths... {'+'*30}")
    paths = path_template(config)

    # 3. 加载数据集
    print(f"\n{'+'*30} Step 3/5: Loading test dataset... {'+'*30}")
    dataset_info = read_dataset_info(config["data"]["dataset_name"], paths)
    test_ds, data_collator = dataset_tokenize(config, dataset_info, paths)

    # 4. 初始化模型
    print(f"\n{'+'*30} Step 4/5: Initializing model... {'+'*30}")
    model_instance = model(config, paths)

    # 5. 初始化 Trainer
    print(f"\n{'+'*30} Step 5/5: Setting up Trainer... {'+'*30}")
    infer_args = inference_args(config, paths)
    trainer = create_trainer(config, model_instance,
                             infer_args, test_ds, data_collator)

    # 6. 运行推理
    print(f"\n{'+'*30} Final Step: Running Inference... {'+'*30}")
    run_inference(trainer, test_ds, paths, config, checkpoint_dir)

    print(f"\n{'+'*30} Total Inference Time: {format_time(time.time() - start_time)} {'+'*30}\n")


if __name__ == "__main__":
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _default_cfg = os.path.join(_repo_root, "configs", "config_tuning.yaml")

    parser = argparse.ArgumentParser(
        description="Run Inference for RICE model")
    parser.add_argument("--config", type=str,
                        default=_default_cfg, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to the checkpoint folder")
    args = parser.parse_args()

    if not os.path.isabs(args.config):
        args.config = os.path.join(_repo_root, args.config)

    main(args.config, args.checkpoint)

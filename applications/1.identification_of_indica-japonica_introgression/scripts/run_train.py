import os
import sys
import time
import argparse
import json
import glob
import torch
import yaml
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoTokenizer, set_seed
from transformers.trainer_callback import EarlyStoppingCallback
from datetime import datetime
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


# =========================
# 1. 读取config
# =========================
def read_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = yaml.safe_load(os.path.expandvars(yaml.dump(cfg)))
    return cfg


# =========================
# 2. 路径模板
#  ========================
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

    # 根据训练模式构建路径
    use_extracted_embeddings = config["embedding"].get(
        "use_extracted_embeddings", False)

    if use_extracted_embeddings:
        # Embedding-only 模式：包含 layer 信息
        layer = config["embedding"].get("layer", 12)
        values["layer"] = layer
        layer_path = os.path.join(
            result_dir, values["model_name"], values['dataset_name'], f"{layer}layer")

        if not config["embedding"].get("embedding_dir"):
            embedding_path = os.path.join(layer_path, "embeddings")
        else:
            embedding_path = config["embedding"]["embedding_dir"]
    else:
        # LoRA 端到端模式：不需要 layer 信息
        layer_path = os.path.join(
            result_dir, values["model_name"], values['dataset_name'])
        embedding_path = None

    experiment_name = config["train"].get(
        "run_name", time.strftime("%Y%m%d-%H%M%S"))
    experiment_dir = os.path.join(layer_path, experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    logs_dir = os.path.join(experiment_dir, "logs")
    metrics_dir = os.path.join(experiment_dir, "metrics")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    return {
        'data_dir': data_dir,
        'model_path': config["model"]["model_path"],
        'result_layer_path': layer_path,
        'embedding_path': embedding_path,
        'result_checkpoints_dir': checkpoints_dir,
        'result_logs_dir': logs_dir,
        'result_metrics_dir': metrics_dir,
        'use_extracted_embeddings': use_extracted_embeddings
    }

# =========================
# 3. 读取 dataset info
# =========================


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
        "eval_task": dataset_info.get("eval_task", "classification"),
        "split_name": dataset_info.get("split_name", {"train": "train", "eval": "eval", "test": "test"})
    }


# =========================
# 4. tokenizer & datasets
# =========================
def dataset_tokenize(config, dataset_info, paths):

    use_extracted_embeddings = paths['use_extracted_embeddings']
    if use_extracted_embeddings:
        # 模式2: 使用预计算嵌入
        if paths['embedding_path'] is None:
            raise ValueError(
                "embedding_path is None when use_extracted_embeddings is True")
        train_ds = create_embeddings_dataset(
            paths['embedding_path'], config['embedding']['split_name']['train'])
        eval_ds = create_embeddings_dataset(
            paths['embedding_path'], config['embedding']['split_name']['eval'])
        test_ds = create_embeddings_dataset(
            paths['embedding_path'], config['embedding']['split_name']['test'])
        return train_ds, eval_ds, test_ds, None

    # 模式1: 端到端 token（可选 HF Arrow 落盘缓存）
    token = config["model"]["token"]
    tokenizer = AutoTokenizer.from_pretrained(
        paths["model_path"], trust_remote_code=True, token=token
    )
    split_names = dataset_info.get("split_name", config['data'].get('split_name', {
        'train': 'train',
        'eval': 'eval',
        'test': 'test'
    }))
    tok_cfg = tokenizer_config_for_e2e(config)
    use_arrow = bool(tok_cfg["use_arrow_token_cache"])
    force_rebuild = bool(tok_cfg.get("force_rebuild_arrow_cache", False))
    if use_arrow:
        _repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        train_ds = load_or_build_arrow_split(
            config,
            dataset_info,
            tokenizer,
            split_names["train"],
            project_root=_repo_root,
            force_rebuild=force_rebuild,
        )
        eval_ds = load_or_build_arrow_split(
            config,
            dataset_info,
            tokenizer,
            split_names["eval"],
            project_root=_repo_root,
            force_rebuild=force_rebuild,
        )
        test_ds = load_or_build_arrow_split(
            config,
            dataset_info,
            tokenizer,
            split_names["test"],
            project_root=_repo_root,
            force_rebuild=force_rebuild,
        )
        return train_ds, eval_ds, test_ds, make_token_batch_collator()

    train_ds = create_rice_dataset(
        dataset_info, split_names['train'], tokenizer)
    eval_ds = create_rice_dataset(dataset_info, split_names['eval'], tokenizer)
    test_ds = create_rice_dataset(dataset_info, split_names['test'], tokenizer)
    return train_ds, eval_ds, test_ds, None


def peft_lora_dict_for_backbone(config):
    """lora.use_lora 为真时返回供 BackboneModule / LoraConfig 使用的字段（不含 use_lora）。"""
    lora = config.get("lora")
    if not lora or not lora.get("use_lora", False):
        return None
    return {k: v for k, v in lora.items() if k != "use_lora"}


def model(config, paths):
    use_extracted_embeddings = paths['use_extracted_embeddings']
    head_dropout = float(config["train"].get("dropout", 0.1))
    if use_extracted_embeddings:
        # 模式2: 仅训练Projection + Classifier (Embedding-only 模式)
        model_obj = EmbeddingModel(
            proj_dims=config["projection"]["dims"],
            num_labels=config["task"]["num_labels"],
            is_bn=config["projection"].get("use_batchnorm", False),
            dropout=head_dropout,
        )
    else:
        # 模式1: 端到端训练 (LoRA 微调模式)
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

# =========================
# 6. Trainer arguments (with best practices)
# =========================
def trainer_args(config, paths):
    warmup_ratio = config["train"].get("warmup_ratio")
    warmup_steps = config["train"].get("warmup_steps")
    fp16 = bool(config["train"].get("fp16", True))
    bf16 = bool(config["train"].get("bf16", False))
    if fp16 and bf16:
        raise ValueError("Only one of train.fp16 and train.bf16 can be true.")

    dataloader_num_workers = int(
        config["train"].get("dataloader_num_workers", 0))
    dataloader_persistent_workers = bool(
        config["train"].get("dataloader_persistent_workers",
                            dataloader_num_workers > 0)
    )
    if dataloader_num_workers == 0 and dataloader_persistent_workers:
        dataloader_persistent_workers = False

    ddp_find_unused_parameters = config["train"].get(
        "ddp_find_unused_parameters")
    ddp_backend = config["train"].get("ddp_backend")

    run_name = config["train"].get(
        "run_name") or time.strftime("%Y%m%d-%H%M%S")

    training_args_kwargs = dict(
        output_dir=paths['result_checkpoints_dir'],
        eval_strategy=config["train"]["eval_strategy"],
        eval_steps=config["train"]["eval_steps"],
        save_strategy=config["train"]["save_strategy"],
        save_steps=config["train"]["save_steps"],
        learning_rate=float(config["train"]["lr"]),
        per_device_train_batch_size=int(config["train"]["batch_size"]),
        per_device_eval_batch_size=int(config["train"]["batch_size"]),
        num_train_epochs=config["train"]["epochs"],
        lr_scheduler_type=config["train"].get("lr_scheduler_type", "linear"),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
        fp16=fp16,
        bf16=bf16,
        logging_dir=paths["result_logs_dir"],
        logging_steps=int(config["train"].get("logging_steps", 10)),
        save_total_limit=2,
        report_to=["tensorboard"],
        run_name=run_name,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=bool(
            config["train"].get("dataloader_pin_memory",
                                torch.cuda.is_available())
        ),
        dataloader_persistent_workers=dataloader_persistent_workers,
        seed=SEED,  # 设置随机种子
        # 梯度累积
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
        load_best_model_at_end=True,  # 训练完后加载最佳模型
        metric_for_best_model="accuracy",  # 用 accuracy 作为最佳模型指标
        greater_is_better=True,  # accuracy 越大越好
        remove_unused_columns=False,
    )

    if warmup_ratio is not None and warmup_ratio > 0:
        training_args_kwargs["warmup_ratio"] = float(warmup_ratio)
    elif warmup_steps is not None and warmup_steps > 0:
        training_args_kwargs["warmup_steps"] = int(warmup_steps)
    else:
        training_args_kwargs["warmup_ratio"] = 0.05  # 默认 warmup 比例，适用于大多数情况

    if is_distributed_env():
        if ddp_find_unused_parameters is not None:
            training_args_kwargs["ddp_find_unused_parameters"] = bool(
                ddp_find_unused_parameters)
        if ddp_backend:
            training_args_kwargs["ddp_backend"] = ddp_backend

    training_args = TrainingArguments(**training_args_kwargs)
    return training_args


def find_latest_checkpoint(checkpoints_dir):
    """在 checkpoints 目录中查找最新的 checkpoint-* 目录。"""
    if not checkpoints_dir or not os.path.isdir(checkpoints_dir):
        return None

    candidates = []
    for entry in glob.glob(os.path.join(checkpoints_dir, "checkpoint-*")):
        if not os.path.isdir(entry):
            continue
        base = os.path.basename(entry)
        if not base.startswith("checkpoint-"):
            continue
        suffix = base.split("checkpoint-", 1)[-1]
        if suffix.isdigit():
            candidates.append((int(suffix), entry))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_checkpoint_path(checkpoint_path_spec, checkpoints_dir):
    """
    解析续跑 checkpoint 路径。
    checkpoint_path_spec: None 表示不续跑；True/'auto'/'latest' 自动查找最新；字符串为具体路径。
    """
    if checkpoint_path_spec is None:
        return None

    if isinstance(checkpoint_path_spec, bool):
        if not checkpoint_path_spec:
            return None
        checkpoint_path_spec = "auto"

    if isinstance(checkpoint_path_spec, str):
        value = checkpoint_path_spec.strip()
        if not value or value.lower() in {"false", "none", "0", "no"}:
            return None
        if value.lower() in {"true", "auto", "latest"}:
            ckpt = find_latest_checkpoint(checkpoints_dir)
            if ckpt is None:
                print(
                    f"[Checkpoint] No checkpoint found under {checkpoints_dir}; "
                    "starting training from scratch."
                )
                return None
            return ckpt

        ckpt_dir = os.path.expanduser(value)
        if not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.abspath(ckpt_dir)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        return ckpt_dir

    raise ValueError(
        f"Unsupported resume_checkpoint_path: {checkpoint_path_spec!r}. "
        "Use auto/latest or a checkpoint directory path."
    )


def validate_checkpoint_path(ckpt_dir):
    """确认 checkpoint 目录包含 Trainer 续跑所需的状态文件。"""
    trainer_state = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.isfile(trainer_state):
        raise FileNotFoundError(
            f"Invalid checkpoint (missing trainer_state.json): {ckpt_dir}"
        )

    weight_files = [
        os.path.join(ckpt_dir, "model.safetensors"),
        os.path.join(ckpt_dir, "pytorch_model.bin"),
    ]
    if not any(os.path.isfile(path) for path in weight_files):
        raise FileNotFoundError(
            f"Invalid checkpoint (missing model weights): {ckpt_dir}"
        )
    return ckpt_dir


def resolve_resume_checkpoint_path(config, checkpoints_dir):
    """
    根据配置 train.resume_checkpoint_path 解析续跑路径：
    - auto：在 checkpoints_dir 下自动找最新 checkpoint
    - 路径字符串：使用该 checkpoint 目录
    - false：不续跑
    """
    spec = config.get("train", {}).get("resume_checkpoint_path", "auto")
    return resolve_checkpoint_path(spec, checkpoints_dir)


# =========================
# 7. 自定义 Trainer
# =========================
def trainer(config, model, training_args, train_ds, eval_ds, data_collator=None):
    threshold = config["train"]["sigmoid_threshold"]

    if config["task"]["task_class"] == "labels":
        multi_label = True
    elif config["task"]["task_class"] == "classification":
        multi_label = False
    elif config["task"]["task_class"] == "regression":
        multi_label = False
    else:
        raise ValueError(
            f"Unsupported task_class: {config['task']['task_class']}")

    def metric_fn(eval_pred): return compute_metrics(
        eval_pred, multi_label=multi_label, threshold=threshold)

    early_stopping_patience = config["train"].get("early_stopping_patience")
    callbacks = []
    if early_stopping_patience is not None and int(early_stopping_patience) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
                early_stopping_threshold=0.0,
            )
        )
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=metric_fn,
        multi_label=multi_label,
        lambda_contrastive=config["train"]["lambda_contrastive"],
        callbacks=callbacks,
    )
    return trainer

# =========================
# 8. 训练 (with best model checkpoint)
# =========================
def train(trainer, config, training_args, resume_checkpoint_path=None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")

    num_samples = len(trainer.train_dataset)
    bs = training_args.per_device_train_batch_size
    gas = training_args.gradient_accumulation_steps
    epochs = int(training_args.num_train_epochs)
    approx_total_updates = max(1, (num_samples * epochs) // (bs * gas))
    warmup_effective = training_args.get_warmup_steps(approx_total_updates)

    print(f"\n{'='*60}")
    print(f"Starting training with:")
    print(f"  - Seed: {SEED}")
    print(f"  - CUDA_VISIBLE_DEVICES: {visible_devices}")
    print(f"  - World Size: {world_size}")
    print(f"  - Local Rank: {local_rank}")
    print(
        f"  - Precision: {'bf16' if training_args.bf16 else ('fp16' if training_args.fp16 else 'fp32')}")
    print(
        f"  - Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}")
    print(f"  - Dataloader Workers: {training_args.dataloader_num_workers}")
    print(
        f"  - Train samples: {num_samples}; approx. total optimizer steps: {approx_total_updates}")
    if training_args.warmup_ratio and training_args.warmup_ratio > 0:
        print(
            f"  - Warmup Ratio: {training_args.warmup_ratio} "
            f"(effective warmup steps ≈ {warmup_effective} at {approx_total_updates} steps)"
        )
    else:
        print(
            f"  - Warmup Steps: {training_args.warmup_steps} "
            f"(effective ≈ {warmup_effective} at {approx_total_updates} steps)"
        )
    early_stopping_patience = config["train"].get("early_stopping_patience")
    if early_stopping_patience is None or int(early_stopping_patience) <= 0:
        print("  - Early Stopping: Disabled")
    else:
        print(f"  - Early Stopping Patience: {int(early_stopping_patience)}")
    if resume_checkpoint_path:
        print(f"  - Resume Checkpoint Path: {resume_checkpoint_path}")
    else:
        print("  - Resume Checkpoint Path: Disabled")
    print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint_path)

    print(f"\n{'='*60}")
    print(f"Checkpoints directory: {training_args.output_dir}")
    print(f"Best Model Checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"{'='*60}\n")


# =========================
# 9. 验证推理 (on best model)
# =========================
def eval_inference(trainer, eval_ds, paths, threshold=0.5):
    metrics = trainer.evaluate(eval_ds)
    print(f"\n{'='*60}")
    print("Eval Metrics (using best model):")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result = {
        "accuracy": metrics.get("eval_accuracy"),
        "f1": metrics.get("eval_f1"),
        "recall": metrics.get("eval_recall"),
        "precision": metrics.get("eval_precision"),
        "auc_roc": metrics.get("eval_auc_roc"),
        "mcc": metrics.get("eval_mcc"),
        "time": datetime.now().isoformat(),
        "checkpoint": trainer.state.best_model_checkpoint,
        "threshold": threshold,
        "loss": metrics.get("eval_loss"),
        "runtime": metrics.get("eval_runtime"),
        "samples_per_second": metrics.get("eval_samples_per_second"),
        "steps_per_second": metrics.get("eval_steps_per_second"),
        "epoch": metrics.get("epoch"),
    }
    metrics_path = os.path.join(paths["result_metrics_dir"], f"eval_metrics_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved eval metrics to: {metrics_path}")
    print(f"{'='*60}\n")


# =========================
# 10. 测试推理 (on best model)
# =========================
def test_inference(trainer, test_ds, paths, threshold=0.5):
    trainer.model.eval()
    num_samples = len(test_ds)
    if trainer.is_world_process_zero():
        print(f"\n{'='*60}")
        print("Test Metrics (using best model):")
        print(
            f"Samples: {num_samples} | Batch Size: {trainer.args.per_device_eval_batch_size}")
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
        f"Saved logits to: {logits_path} ({len(df_results)} rows, with sample_idx)")

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
        "checkpoint": trainer.state.best_model_checkpoint,
        "threshold": threshold,
        "runtime": metrics.get("test_runtime"),
        "world_size": trainer.args.world_size,
    }
    metrics_path = os.path.join(
        paths["result_metrics_dir"], "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved test metrics to: {metrics_path}")
    print(f"{'='*60}\n")


def main(config_path):
    _local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    start_time = time.time()
    print(f"\n{'+'*30} 【{_local_rank}】 Starting training pipeline...{'+'*30}\n")

    # 1. 读取配置和GPU设置
    step_start = time.time()
    print(f"\n{'+'*30}【{_local_rank}】 Step 1/9: Reading configuration and setting up GPU...{'+'*30}\n")
    config = read_config(config_path)
    print(f"\n{'+'*30}【{_local_rank}】 Step 1/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")

    # 2. 路径模板和训练模式信息
    step_start = time.time()
    print(f"\n{'+'*30}【{_local_rank}】 Step 2/9: Setting up paths and training mode...{'+'*30}\n")
    paths = path_template(config)

    resume_checkpoint_path = resolve_resume_checkpoint_path(
        config, paths["result_checkpoints_dir"]
    )
    if resume_checkpoint_path is not None:
        resume_checkpoint_path = validate_checkpoint_path(resume_checkpoint_path)
        print(f"\n{'='*60}")
        print("Resuming training from checkpoint path:")
        print(f"  {resume_checkpoint_path}")
        print(f"{'='*60}\n")

    # 打印训练模式信息
    print(f"\n{'='*60}")
    if paths['use_extracted_embeddings']:
        print("Training Mode: EMBEDDING-ONLY (using pre-computed embeddings)")
    else:
        print("Training Mode: END-TO-END (FullModel)")
        if (config.get("lora") or {}).get("use_lora"):
            print(
                f"  - LoRA enabled with r={config['lora'].get('r', 16)}, alpha={config['lora'].get('alpha', 32)}")
        else:
            print("  - LoRA disabled (full model fine-tuning)")
    print(f"{'='*60}\n")
    print(f"\n{'+'*30}【{_local_rank}】 Step 2/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")

    # 3. 数据集信息
    step_start = time.time()
    print(
        f"\n{'+'*30}【{_local_rank}】 Step 3/9: Reading dataset information...{'+'*30}\n")
    dataset_info = read_dataset_info(config["data"]["dataset_name"], paths)
    print(f"\n{'+'*30}【{_local_rank}】 Step 3/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")

    # 4. tokenizer & datasets
    step_start = time.time()
    print(
        f"\n{'+'*30}【{_local_rank}】 Step 4/9: Tokenizing and loading datasets...{'+'*30}\n")
    train_ds, eval_ds, test_ds, data_collator = dataset_tokenize(
        config,
        dataset_info,
        paths=paths
    )
    print(f"\n{'+'*30}【{_local_rank}】 Step 4/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")

    # 5. model
    step_start = time.time()
    print(f"\n{'+'*30}【{_local_rank}】 Step 5/9: Initializing model...{'+'*30}\n")
    model_instance = model(config, paths)
    print(f"\n{'+'*30}【{_local_rank}】 Step 5/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")

    # 6. Trainer arguments
    step_start = time.time()
    print(
        f"\n{'+'*30}【{_local_rank}】 Step 6/9 : Setting up trainer arguments...{'+'*30}\n")
    training_args = trainer_args(config, paths)
    print(f"\n{'+'*30}【{_local_rank}】 Step 6/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")

    # 7. Trainer
    step_start = time.time()
    print(f"\n{'+'*30}【{_local_rank}】 Step 7/9: Initializing trainer...{'+'*30}\n")
    trainer_instance = trainer(
        config, model_instance, training_args, train_ds, eval_ds, data_collator=data_collator
    )
    print(f"\n{'+'*30}【{_local_rank}】 Step 7/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")

    # 8. 训练
    step_start = time.time()
    print(f"\n{'+'*30}【{_local_rank}】 Step 8/9: Starting training...{'+'*30}\n")
    train(
        trainer_instance,
        config,
        training_args,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    print(f"\n{'+'*30}【{_local_rank}】 Step 8/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")


    # 9. 验证推理
    step_start = time.time()
    print(f"\n{'+'*30}【{_local_rank}】 Step 9/9: Running eval inference...{'+'*30}\n")
    eval_inference(trainer_instance, eval_ds, paths,
                   threshold=config["train"]["sigmoid_threshold"])
    print(f"\n{'+'*30}【{_local_rank}】 Step 9/9 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")


    # 10. 测试推理
    step_start = time.time()
    print(f"\n{'+'*30}【{_local_rank}】 Step 10/10: Running test inference...{'+'*30}\n")
    test_inference(trainer_instance, test_ds, paths,
                   threshold=config["train"]["sigmoid_threshold"])
    print(f"\n{'+'*30}【{_local_rank}】 Step 10/10 Completed in {format_time(time.time() - step_start)}{'+'*30}\n")

    total_time = time.time() - start_time
    print(f"\n{'+'*30} 【{_local_rank}】 Total training time: {format_time(total_time)} {'+'*30}\n")


if __name__ == "__main__":
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _default_cfg = os.path.join(_repo_root, "configs", "config_tuning.yaml")

    parser = argparse.ArgumentParser(
        description="Train contrastive rice model")
    parser.add_argument(
        "--config",
        type=str,
        default=_default_cfg,
        help="Path to training config yaml (absolute or relative to project root).",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.config):
        args.config = os.path.join(_repo_root, args.config)

    main(args.config)

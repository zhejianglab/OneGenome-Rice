# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""End-to-end training pipeline (YAML-driven, predictor selected by config)."""

import math
import os
import traceback

import torch
import torch.distributed as dist
import yaml
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from model.config import effective_per_device_train_batch_size, parse_dataset_block
from model.distributed import DistributedSamplerCallback, dist_print, setup_distributed, setup_sync_batchnorm
from model.encoder_transformer import ATAC_TransformerEncoder
from model.env import SEED
from model.index import build_index
from model.load_pretrained import load_model_and_tokenizer
from model.predictor_fusion import MultiModalPredictorFusion

# When `scale_gradient_accumulation_for_world_size` is true, YAML `gradient_accumulation_steps`
# is interpreted for a world_size=1 recipe; scaling matches global batch to that reference.
GRADIENT_ACCUMULATION_REFERENCE_WORLD_SIZE = 1


def _predictor_config_block(cfg: dict) -> dict:
    """Return the `predictor:` mapping from config, or {} if missing."""
    block = cfg.get("predictor")
    return block if isinstance(block, dict) else {}


def _predictor_type_raw(cfg: dict):
    """Read raw predictor type from config (explicit key only)."""
    block = _predictor_config_block(cfg)
    if "type" in block:
        v = block["type"]
        if v is not None:
            return v
    return "fusion"


def _normalize_predictor_type(cfg: dict) -> str:
    """Return predictor.type (lowercased); default `fusion` if unset.

    Reads `predictor.type`.
    """
    raw = _predictor_type_raw(cfg)
    if isinstance(raw, str):
        s = raw.strip().lower()
    else:
        s = str(raw).strip().lower()
    if not s:
        return "fusion"
    return s


def _mapping_yaml_bool(mapping: dict, key: str, *, default: bool = False, label: str) -> bool:
    """Parse a boolean from a config mapping (avoids ``bool(\"false\")`` being true)."""
    if key not in mapping:
        return default
    v = mapping[key]
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
        raise ValueError(f"{label}.{key} must be a boolean, got {v!r}")
    raise ValueError(f"{label}.{key} must be a boolean, got {type(v).__name__}: {v!r}")


def _training_yaml_bool(tcfg: dict, key: str, *, default: bool = False) -> bool:
    """Parse ``training.<key>`` boolean."""
    return _mapping_yaml_bool(tcfg, key, default=default, label="training")


def _fusion_predictor_kwargs(cfg: dict) -> dict:
    """Fusion-only: entropy fracs + optional last-layer unfreeze (``predictor:`` block)."""
    block = _predictor_config_block(cfg)
    return {
        "fusion_gate_entropy_frac": float(block.get("fusion_gate_entropy_frac", 0.0)),
        "skip_gate_entropy_frac": float(block.get("skip_gate_entropy_frac", 0.0)),
        "unfreeze_base_last_layer": _mapping_yaml_bool(
            block, "unfreeze_base_last_layer", default=False, label="predictor"
        ),
    }


def _scaled_gradient_accumulation_steps(grad_accum_yaml: int, *, world_size: int) -> int:
    """Scale grad accum so global batch ~ matches a world_size=1 reference: G' = G * 1 / ws (min 1).

    Global samples per optimizer step: per_device_bs * world_size * gradient_accumulation_steps.
    With fixed per_device_bs, G' * ws ≈ G_yaml * GRADIENT_ACCUMULATION_REFERENCE_WORLD_SIZE.
    """
    g = int(grad_accum_yaml)
    if g < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    ws = max(1, int(world_size))
    ref = GRADIENT_ACCUMULATION_REFERENCE_WORLD_SIZE
    scaled = int(round(g * ref / ws))
    return max(1, scaled)


def _num_update_steps_per_epoch(len_dataloader: int, gradient_accumulation_steps: int) -> int:
    """Optimizer steps per epoch; matches HuggingFace `Trainer.set_initial_training_values`."""
    ga = int(gradient_accumulation_steps)
    return max(len_dataloader // ga + int(len_dataloader % ga > 0), 1)


def _train_dataloader_length(train_dataset, per_device_train_batch_size: int) -> int:
    """Len of train DataLoader using the same sampler/batch rules as `CustomTrainer.get_train_dataloader`."""
    if dist.is_initialized():
        sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED)
    else:
        generator = torch.Generator()
        generator.manual_seed(SEED)
        sampler = RandomSampler(train_dataset, generator=generator)
    dl = DataLoader(
        train_dataset,
        batch_size=int(per_device_train_batch_size),
        num_workers=0,
        sampler=sampler,
        shuffle=False,
    )
    return len(dl)


def _save_steps_for_num_per_epoch(
    len_dataloader: int, gradient_accumulation_steps: int, k: int
) -> int:
    """Approximately k checkpoint saves per epoch via `save_strategy='steps'`."""
    n = _num_update_steps_per_epoch(len_dataloader, gradient_accumulation_steps)
    return max(1, n // int(k))


def build_multimodal_model(cfg: dict, base_model):
    """Construct ATAC-conditioned predictor from YAML. Used by training and inference."""
    ptype = _normalize_predictor_type(cfg)
    if ptype == "fusion":
        fusion_kw = _fusion_predictor_kwargs(cfg)
        atac_encoder = ATAC_TransformerEncoder(base_model)
        model = MultiModalPredictorFusion(
            base_model,
            atac_encoder,
            fusion_gate_entropy_frac=fusion_kw["fusion_gate_entropy_frac"],
            skip_gate_entropy_frac=fusion_kw["skip_gate_entropy_frac"],
            unfreeze_base_last_layer=fusion_kw["unfreeze_base_last_layer"],
        )
    else:
        raise ValueError(
            f"Unknown predictor type {ptype!r}. Use 'fusion' "
            "(under predictor.type)."
        )
    return model


def _resolve_output_training_dir(cfg: dict) -> str:
    """Directory for checkpoints, logs, and train_index CSV."""
    out = cfg.get("output_training_dir")
    if not out:
        raise ValueError(
            "Config must set output_training_dir"
        )
    return os.path.abspath(str(out))


def run_training(cfg: dict):
    setup_distributed()

    output_training_dir = _resolve_output_training_dir(cfg)
    os.makedirs(output_training_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = output_training_dir

    # Heavy imports are delayed so inference/help can import this module
    # without pulling in transformers.Trainer (which can trigger TF/Keras imports).
    from transformers import TrainingArguments
    from model.trainer import (
        CustomTrainer,
        CustomTrainerFusion,
        CustomTrainerDiscriminativeLR,
        CustomTrainerDiscriminativeLRFusion,
        FractionalEpochSchedulerCallback,
    )

    base_model, tokenizer = load_model_and_tokenizer(cfg)

    train_dataset = None
    try:
        (
            rna_files,
            atac_files,
            cell_types,
            chromosome,
            chromosome_per_cell_type,
            genome_fasta,
            genome_fasta_per_cell_type,
        ) = parse_dataset_block(cfg)

        dist_print("\n>>> Building paired index...")
        train_index_path = os.path.join(output_training_dir, "train_index_paired.csv")
        target_len = int(cfg.get("target_len", 32000))
        overlap_len = int(cfg.get("overlap_len", target_len // 2))
        train_index_df = build_index(
            genome_fasta,
            train_index_path,
            rna_files,
            atac_files,
            cell_types,
            chromosome,
            chromosome_per_cell_type=chromosome_per_cell_type,
            fasta_path_per_cell_type=genome_fasta_per_cell_type,
            window_size=target_len,
            overlap=overlap_len,
            cap_expression_quantile=cfg.get("cap_expression_quantile"),
        )

        if dist.is_initialized():
            dist.barrier()

        ptype = _normalize_predictor_type(cfg)
        dist_print(f">>> Predictor type: {ptype}")

        model = build_multimodal_model(cfg, base_model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for name, param in model.named_parameters():
            param.data = param.data.to(torch.bfloat16 if "base" in name else torch.float32)

        model = setup_sync_batchnorm(model)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        dist_print(
            f"Parameters – total: {total:,} | trainable: {trainable:,} "
            f"({trainable / total * 100:.2f}%)"
        )

        from model.dataset import LazyGenomicDataset as _LazyGenomicDataset

        n_rows = len(train_index_df)
        dist_print(
            f"\n>>> Creating dataset (index rows: {n_rows})"
        )
        train_dataset = _LazyGenomicDataset(
            train_index_df, tokenizer, max_length=int(cfg.get("target_len", 32000))
        )

        tcfg = cfg.get("training", {})
        per_device_bs = effective_per_device_train_batch_size(cfg)
        grad_accum_yaml = int(tcfg.get("gradient_accumulation_steps", 11))
        scale_ga_for_ws = _training_yaml_bool(
            tcfg, "scale_gradient_accumulation_for_world_size", default=False
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        grad_accum = (
            _scaled_gradient_accumulation_steps(grad_accum_yaml, world_size=world_size)
            if scale_ga_for_ws
            else grad_accum_yaml
        )
        if scale_ga_for_ws and grad_accum != grad_accum_yaml:
            dist_print(
                ">>> Gradient accumulation scaled for world_size: "
                f"gradient_accumulation_steps {grad_accum_yaml} → {grad_accum} "
                f"(reference world_size={GRADIENT_ACCUMULATION_REFERENCE_WORLD_SIZE}, "
                f"current world_size={world_size}; global batch ~ matched to single-process recipe)"
            )
        elif scale_ga_for_ws:
            dist_print(
                ">>> scale_gradient_accumulation_for_world_size: no change to "
                f"gradient_accumulation_steps={grad_accum} "
                f"(reference world_size={GRADIENT_ACCUMULATION_REFERENCE_WORLD_SIZE}, "
                f"current world_size={world_size})"
            )
        save_strategy = tcfg.get("save_strategy", "epoch")
        save_num_raw = tcfg.get("save_num_per_epoch")
        save_per_n_epoch_raw = tcfg.get("save_per_n_epoch")
        use_discriminative_lr = bool(tcfg.get("discriminative_lr", False))

        # LR scheduler configuration (Hugging Face TrainingArguments).
        # Defaults match HF defaults (linear warmup+decay); `warmup_ratio` is already wired below.
        lr_scheduler_type = tcfg.get("lr_scheduler_type", "linear")
        lr_scheduler_kwargs_raw = tcfg.get("lr_scheduler_kwargs", {}) or {}
        if not isinstance(lr_scheduler_kwargs_raw, dict):
            raise ValueError("training.lr_scheduler_kwargs must be a mapping (dict)")
        lr_scheduler_kwargs = dict(lr_scheduler_kwargs_raw)
        lr_st = str(lr_scheduler_type).strip().lower()
        # `min_lr_rate` is only valid for HF cosine-with-floor schedulers. Passing it to
        # `get_linear_schedule_with_warmup` (and others) raises TypeError.
        schedulers_min_lr_rate = ("cosine_with_min_lr", "cosine_warmup_with_min_lr")
        if lr_st in schedulers_min_lr_rate:
            if "min_lr_rate" not in lr_scheduler_kwargs and "min_lr" not in lr_scheduler_kwargs:
                lr_scheduler_kwargs["min_lr_rate"] = float(tcfg.get("min_lr_rate", 0.0))
        else:
            lr_scheduler_kwargs.pop("min_lr_rate", None)

        if save_num_raw is not None and save_per_n_epoch_raw is not None:
            try:
                k = int(save_num_raw)
                n = int(save_per_n_epoch_raw)
            except (TypeError, ValueError):
                raise ValueError(
                    "training.save_num_per_epoch and training.save_per_n_epoch must be integers"
                )
            if not (k == 1 and n == 1):
                raise ValueError(
                    "training.save_per_n_epoch is exclusive with training.save_num_per_epoch "
                    "(unless both are set to 1)"
                )

        num_train_epochs = float(tcfg.get("num_train_epochs", 20))
        len_dl = _train_dataloader_length(train_dataset, per_device_bs)
        n_opt = _num_update_steps_per_epoch(len_dl, grad_accum)
        max_steps = int(math.ceil(num_train_epochs * n_opt))
        dist_print(
            f">>> Training budget: len_dataloader={len_dl}, optimizer steps/epoch={n_opt}, "
            f"max_steps={max_steps} (ceil(num_train_epochs * steps/epoch))"
        )
        max_grad_norm_raw = tcfg.get("max_grad_norm", 1.0)
        max_grad_norm = float(max_grad_norm_raw)
        if max_grad_norm < 0:
            raise ValueError("training.max_grad_norm must be >= 0 (0 disables gradient clipping in HF Trainer)")
        ta_kwargs = dict(
            output_dir=output_training_dir,
            learning_rate=tcfg.get("learning_rate", 8e-5),
            per_device_train_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            max_grad_norm=max_grad_norm,
            weight_decay=tcfg.get("weight_decay", 0.01),
            logging_dir=os.path.join(output_training_dir, "logs"),
            logging_steps=tcfg.get("logging_steps", 2),
            eval_strategy="no",
            # Save timing is handled by FractionalEpochSchedulerCallback so saves occur
            # on the first optimizer step that crosses the target fractional epoch.
            save_strategy="no",
            bf16=True,
            fp16=False,
            optim=tcfg.get("optim", "adafactor"),
            seed=SEED,
            ddp_find_unused_parameters=True,
            report_to="wandb",
            run_name=tcfg.get("run_name", "atac_strand_paired"),
            save_total_limit=tcfg.get("save_total_limit", 2),
            warmup_ratio=tcfg.get("warmup_ratio", 0.05),
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        save_every_epochs = 1.0
        if save_num_raw is not None:
            k = int(save_num_raw)
            if k < 1:
                raise ValueError("training.save_num_per_epoch must be a positive integer")
            save_every_epochs = 1.0 / float(k)
            dist_print(
                f">>> Checkpoints: save_num_per_epoch={k} → save every {save_every_epochs:.6g} epoch "
                f"(~{n_opt} optimizer steps/epoch, len_dataloader={len_dl})"
            )
        elif save_per_n_epoch_raw is not None:
            n = int(save_per_n_epoch_raw)
            if n < 1:
                raise ValueError("training.save_per_n_epoch must be a positive integer")
            save_every_epochs = float(n)
            dist_print(
                f">>> Checkpoints: save_per_n_epoch={n} → save on first step crossing every {n} epochs"
            )
        elif save_strategy == "steps":
            raise ValueError(
                "training.save_strategy 'steps' requires training.save_num_per_epoch (save_steps is removed). "
                "Example: save_num_per_epoch: 4"
            )
        else:
            dist_print(">>> Checkpoints: default save on first step crossing each integer epoch")
        training_args = TrainingArguments(**ta_kwargs)

        if use_discriminative_lr:
            trainer_cls = CustomTrainerDiscriminativeLRFusion
        else:
            trainer_cls = CustomTrainerFusion

        lr_backbone = tcfg.get("learning_rate_backbone")
        lr_head = tcfg.get("learning_rate_head")
        callbacks = [
            DistributedSamplerCallback(),
            FractionalEpochSchedulerCallback(
                total_epochs=num_train_epochs,
                save_every_epochs=save_every_epochs,
            ),
        ]
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
        if use_discriminative_lr:
            trainer_kwargs["learning_rate_backbone"] = (
                ta_kwargs["learning_rate"] if lr_backbone is None else float(lr_backbone)
            )
            trainer_kwargs["learning_rate_head"] = (
                ta_kwargs["learning_rate"] if lr_head is None else float(lr_head)
            )

        trainer = trainer_cls(**trainer_kwargs)

        dist_print("\n>>> Validating data shapes (first sample)")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            dist_print(f"  input_ids  : {sample['input_ids'].shape}")
            dist_print(f"  atac_signal: {sample['atac_signal'].shape}")
            dist_print(f"  labels     : {sample['labels'].shape}")
            model.eval()
            with torch.no_grad():
                out = model(
                    input_ids=sample["input_ids"].unsqueeze(0).to(device),
                    atac_signal=sample["atac_signal"].unsqueeze(0).to(device),
                    labels=sample["labels"].unsqueeze(0).to(device),
                )
            dist_print(
                f"  logits: {out['logits'].shape} | loss: {out['loss'].item():.6f}"
            )

        dist_print("\n>>> Starting training")
        trainer.train()
        dist_print("Training complete. Checkpoints saved to: " + output_training_dir)

    except Exception as e:
        dist_print(f"Error: {e}")
        dist_print(traceback.format_exc())
        raise
    finally:
        if train_dataset is not None and hasattr(train_dataset, "close"):
            train_dataset.close()
            dist_print(">>> train_dataset closed")
        if dist.is_initialized():
            destroy_process_group()
            dist_print(">>> DDP process group destroyed")

    dist_print("Training pipeline complete.")


def run_training_from_yaml(yaml_path: str) -> None:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    run_training(cfg)

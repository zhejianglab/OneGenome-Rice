# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""Hugging Face Trainer subclasses for multimodal genomic training."""

import logging
import os

import pandas as pd
import math
import torch
import torch.distributed as dist
from safetensors.torch import save_model as safe_save_model
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback

from model.distributed import is_main_process
from model.env import SEED


def _samples_seen_at_global_step(trainer: Trainer, global_step: int) -> int:
    """Training windows (samples) implied by an optimizer global_step (same formula as HF step batching)."""
    args = trainer.args
    per_step = (
        int(args.per_device_train_batch_size)
        * int(args.gradient_accumulation_steps)
        * int(args.world_size)
    )
    return int(global_step) * per_step


def _samples_seen(trainer: Trainer) -> int:
    """Cumulative training windows (samples) seen so far; approximate if last batch is partial."""
    return _samples_seen_at_global_step(trainer, int(trainer.state.global_step))


class FractionalEpochSchedulerCallback(TrainerCallback):
    """Trigger checkpoint saves and training stop when fractional epoch thresholds are crossed.

    This avoids waiting for ``on_epoch_end`` when the desired save/stop boundary is a
    fractional epoch such as 0.25, 0.5, 2.5, etc. The callback fires on the first
    optimizer step whose ``state.epoch`` meets or exceeds the next target.
    """

    def __init__(self, total_epochs: float, save_every_epochs: float | None):
        self.total_epochs = float(total_epochs)
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be > 0")
        self.save_every_epochs = (
            None if save_every_epochs is None else float(save_every_epochs)
        )
        if self.save_every_epochs is not None and self.save_every_epochs <= 0:
            raise ValueError("save_every_epochs must be > 0")
        self._next_save_epoch = self.save_every_epochs

    @staticmethod
    def _epoch_float(state) -> float | None:
        if state.epoch is None:
            return None
        try:
            return float(state.epoch)
        except (TypeError, ValueError):
            return None

    def on_train_begin(self, args, state, control, **kwargs):
        self._next_save_epoch = self.save_every_epochs
        return control

    def on_step_end(self, args, state, control, **kwargs):
        epoch_f = self._epoch_float(state)
        if epoch_f is None:
            return control

        eps = 1e-12
        if self._next_save_epoch is not None:
            crossed_save = False
            while self._next_save_epoch <= self.total_epochs + eps and epoch_f + eps >= self._next_save_epoch:
                crossed_save = True
                self._next_save_epoch += self.save_every_epochs
            if crossed_save:
                control.should_save = True

        if epoch_f + eps >= self.total_epochs:
            control.should_training_stop = True
            control.should_epoch_stop = True

        return control


class CustomTrainer(Trainer):
    """Default trainer: multimodal forward, DDP dataloader, CSV loss log, safetensors save."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            atac_signal=inputs["atac_signal"],
            labels=inputs["labels"],
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        # Both paths use an explicit sampler with shuffling; DataLoader(shuffle=...) must stay False.
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True, seed=SEED)
        else:
            generator = torch.Generator()
            generator.manual_seed(SEED)
            sampler = RandomSampler(self.train_dataset, generator=generator)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=1,
            pin_memory=True,
            collate_fn=self._collate_fn,
            sampler=sampler,
            shuffle=False,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        return None

    def _collate_fn(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "atac_signal": torch.stack([b["atac_signal"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    def log(self, logs, start_time=None):
        if self.state.epoch is not None:
            try:
                logs["epoch"] = float(self.state.epoch)
            except (TypeError, ValueError):
                pass
        if "loss" in logs:
            logs["samples_seen"] = _samples_seen(self)
        super().log(logs, start_time)
        self._append_training_loss_csv(logs)

    def _append_training_loss_csv(self, logs):
        """Write one row to train_loss_per_log.csv when this log line has training loss.

        Includes ``grad_norm`` when Hugging Face ``Trainer`` provides it (same as console/W&B).
        """
        if not is_main_process():
            return
        if "loss" not in logs or "epoch" not in logs:
            return
        csv_path = os.path.join(self.args.output_dir, "train_loss_per_log.csv")
        epoch_value = logs.get("epoch")
        if self.state.epoch is not None:
            try:
                epoch_value = float(self.state.epoch)
            except (TypeError, ValueError):
                pass
        row = {
            "global_step": int(self.state.global_step),
            "samples_seen": _samples_seen(self),
            "epoch": epoch_value,
            "loss": logs["loss"],
        }
        gn = logs.get("grad_norm")
        if gn is not None:
            try:
                row["grad_norm"] = round(float(gn), 8)
            except (TypeError, ValueError):
                row["grad_norm"] = gn
        else:
            row["grad_norm"] = None
        self._extend_csv_row_with_aux_losses(row, logs)
        pd.DataFrame([row]).to_csv(
            csv_path, mode="a", header=not os.path.exists(csv_path), index=False
        )

    def _extend_csv_row_with_aux_losses(self, row: dict, logs: dict) -> None:
        """H2 adds cls_loss / reg_loss; other heads may add auxiliary losses too."""
        pass

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.getLogger(__name__).info(
            f"Saving model (safetensors) to {output_dir}"
        )
        safe_save_model(self.model, os.path.join(output_dir, "model.safetensors"))
        cfg_src = self.model if hasattr(self.model, "config") else self.model.base
        if hasattr(cfg_src, "config"):
            cfg_src.config.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def _save_checkpoint(self, model, trial):
        """Name checkpoint dirs `checkpoint-{samples_seen}` instead of `checkpoint-{global_step}`."""
        from transformers.trainer_callback import ExportableState
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy
        from transformers.training_args import ParallelMode
        from transformers.utils import is_sagemaker_mp_enabled, is_torch_xla_available

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{_samples_seen(self)}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if (
            self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]
            and self.state.best_global_step
        ):
            if is_torch_xla_available():
                import torch_xla.core.xla_model as xm

                xm.rendezvous("load_best_model_at_end")
            elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                import smdistributed.modelparallel.torch as smp

                smp.barrier()

            best_folder = f"{PREFIX_CHECKPOINT_DIR}-{_samples_seen_at_global_step(self, self.state.best_global_step)}"
            best_checkpoint_dir = os.path.join(run_dir, best_folder)
            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            self._save_rng_state(output_dir)

        if self.args.should_save:
            for cb in [
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


def _split_trainable_params_discriminative_lr(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainable params into (backbone_last_layer, head_and_everything_else).

    This project typically freezes the base model and unfreezes only the last transformer layer
    (see predictor conventions in `model/`). For discriminative LR, we treat that last
    transformer layer as the "backbone" group, and all other trainable parameters as the "head"
    group (ATAC encoder + decoder, etc.).

    The split is based on **parameter identity** (not names) to remain robust to module aliasing
    (e.g. predictors exposing `self.layers` that references modules also reachable via `self.base`).
    """
    backbone_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    # Best-effort: locate last transformer layer via the predictor convention `model.layers[-1]`.
    last_layer_param_ids: set[int] = set()
    layers = getattr(model, "layers", None)
    try:
        if layers is not None and len(layers) > 0:
            for p in layers[-1].parameters():
                last_layer_param_ids.add(id(p))
    except Exception:
        # If we cannot introspect layers, fall back to "everything is head".
        last_layer_param_ids = set()

    for p in model.parameters():
        if not p.requires_grad:
            continue
        (backbone_params if id(p) in last_layer_param_ids else head_params).append(p)

    return backbone_params, head_params


class CustomTrainerDiscriminativeLR(CustomTrainer):
    """Trainer that applies discriminative learning rates (backbone last layer vs head).

    Config wiring is done in `model/pipeline.py`:
    - training.discriminative_lr: bool
    - training.learning_rate_backbone: optional float (defaults to training.learning_rate)
    - training.learning_rate_head: optional float (defaults to training.learning_rate)
    """

    def __init__(
        self,
        *args,
        learning_rate_backbone: float | None = None,
        learning_rate_head: float | None = None,
        **kwargs,
    ):
        self.learning_rate_backbone = (
            None if learning_rate_backbone is None else float(learning_rate_backbone)
        )
        self.learning_rate_head = (
            None if learning_rate_head is None else float(learning_rate_head)
        )
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        # If an optimizer was already provided/created, keep HF behavior.
        if self.optimizer is not None:
            return self.optimizer

        backbone_params, head_params = _split_trainable_params_discriminative_lr(self.model)

        # Fall back to TrainingArguments.learning_rate if unset.
        lr_base = float(self.args.learning_rate) if self.learning_rate_backbone is None else float(self.learning_rate_backbone)
        lr_head = float(self.args.learning_rate) if self.learning_rate_head is None else float(self.learning_rate_head)

        if len(backbone_params) == 0:
            logging.getLogger(__name__).warning(
                "Discriminative LR enabled but no backbone params were detected; "
                "all trainable params will use learning_rate_head."
            )
            lr_base = lr_head

        # Weight decay handling:
        # Use the HF helper if present (version-dependent). If not, apply weight_decay uniformly.
        weight_decay = float(getattr(self.args, "weight_decay", 0.0) or 0.0)
        logger = logging.getLogger(__name__)
        logger.info(
            "Creating optimizer with discriminative LR: "
            f"backbone_lr={lr_base:g} (params={len(backbone_params)}), "
            f"head_lr={lr_head:g} (params={len(head_params)}), "
            f"weight_decay={weight_decay:g}"
        )

        # Use Trainer's standard optimizer class selection (Adafactor/AdamW, etc.),
        # but feed our custom param groups.
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_base, "weight_decay": weight_decay})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer


class CustomTrainerFusion(CustomTrainer):
    """Fusion predictor: log gate entropies and entropy-regularization terms (interval-averaged like H2)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fusion_h4_sum = None
        self._fusion_h2_sum = None
        self._fusion_mse_sum = None
        self._fusion_log_interval = None

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        self._fusion_log_interval = None
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            self._fusion_log_interval = self.state.global_step - self._globalstep_last_logged
        super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            atac_signal=inputs["atac_signal"],
            labels=inputs["labels"],
        )
        loss = outputs["loss"]
        if outputs.get("fusion_gate_entropy") is not None:
            if self._fusion_h4_sum is None:
                z = torch.tensor(0.0, device=self.args.device, dtype=torch.float32)
                self._fusion_h4_sum = z.clone()
                self._fusion_h2_sum = z.clone()
                self._fusion_mse_sum = z.clone()
            self._fusion_h4_sum = self._fusion_h4_sum + outputs["fusion_gate_entropy"].detach().float()
            self._fusion_h2_sum = self._fusion_h2_sum + outputs["skip_gate_entropy"].detach().float()
            if outputs.get("mse_loss") is not None:
                self._fusion_mse_sum = self._fusion_mse_sum + outputs["mse_loss"].detach().float()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs and self._fusion_h4_sum is not None:
            interval = self._fusion_log_interval
            if interval is not None and interval > 0:
                h4 = self._nested_gather(self._fusion_h4_sum).mean()
                h2 = self._nested_gather(self._fusion_h2_sum).mean()
                mse = self._nested_gather(self._fusion_mse_sum).mean()
                denom = float(interval) * float(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
                # Report normalized entropies (H / log(K)) so values live in [0,1].
                logs["fusion_gate_entropy"] = round(((h4 / denom) / math.log(4.0)).item(), 6)
                logs["skip_gate_entropy"] = round(((h2 / denom) / math.log(2.0)).item(), 6)
                logs["mse_loss"] = round((mse / denom).item(), 6)
                self._fusion_h4_sum.zero_()
                self._fusion_h2_sum.zero_()
                self._fusion_mse_sum.zero_()
        self._fusion_log_interval = None
        super().log(logs, start_time)

    def _extend_csv_row_with_aux_losses(self, row: dict, logs: dict) -> None:
        for k in (
            "fusion_gate_entropy",
            "skip_gate_entropy",
            "mse_loss",
        ):
            if k in logs:
                row[k] = logs[k]


class CustomTrainerDiscriminativeLRFusion(CustomTrainerDiscriminativeLR, CustomTrainerFusion):
    """Fusion variant of discriminative LR trainer (keeps fusion gate logging)."""

    pass

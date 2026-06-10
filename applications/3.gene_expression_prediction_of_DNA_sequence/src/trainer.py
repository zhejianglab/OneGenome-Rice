# 标准库（内置模块）
from typing import Optional, Union, Dict, Any

import sys
import json
import os
import time

# 第三方库（pip 安装的包）
import torch
from torch.utils.data import Dataset, Subset, DataLoader, DistributedSampler, TensorDataset
from collections import defaultdict

# Hugging Face Transformers
from transformers import (
    Trainer,
    TrainerCallback
    )

# 从自定义仓库中导入模块
from src.util import dist_print



# 自定义训练器
class DistributedSamplerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(int(state.epoch))



class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chrom2id = {f"chr{i}": i for i in range(1,23)}
        self._accumulated_per_head_losses = defaultdict(list)
    
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(input_ids=inputs["input_ids"], labels=inputs["labels"])
        loss = outputs["loss"]
        per_head_losses = outputs.get("per_head_losses", {})

        # 不在这里做多卡/accum 缩放——在 training_step 中统一处理并保存
        extra = {"per_head_losses": per_head_losses}
        return (loss, extra) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        # === 注入分任务 loss（求和即为平均，因每项已 / accum_steps 并已多卡平均）===
        for name, losses in self._accumulated_per_head_losses.items():
            if not losses:
                continue
            try:
                stacked = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in losses])
                total = float(stacked.sum().item())
            except Exception:
                # best-effort float sum
                s = 0.0
                for l in losses:
                    try:
                        s += float(l)
                    except Exception:
                        pass
                total = float(s)
            logs[f"loss_{name}"] = total

        # 清空缓冲区
        self._accumulated_per_head_losses.clear()

        # 调用父类 log（包含 loss, lr, etc.）
        super().log(logs, *args, **kwargs)
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, extra = self.compute_loss(model, inputs, return_outputs=True)

        # 缩放总 loss（HF 原生逻辑）
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # 反向传播
        self.accelerator.backward(loss)

        # === 关键：累积分任务 loss（做相同缩放）===
        per_head_losses = extra.get("per_head_losses", {})
        world_size = 1
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = float(torch.distributed.get_world_size())
        except Exception:
            world_size = 1.0

        device_for_reduce = None
        try:
            device_for_reduce = next(model.parameters()).device
        except Exception:
            device_for_reduce = torch.device("cpu")

        for name, val in per_head_losses.items():
            # 标准化为 tensor
            try:
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32, device=device_for_reduce)
                else:
                    val = val.detach().to(device_for_reduce)
            except Exception:
                # fallback: convert to float -> tensor
                try:
                    val = torch.tensor(float(val), dtype=torch.float32, device=device_for_reduce)
                except Exception:
                    continue

            # 多卡聚合（sum -> / world_size 即为平均）
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
                    torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
                    val = val / world_size
            except Exception:
                # ignore reduce errors, keep local value
                pass

            # HF 原生对总 loss 做的 mean()/grad_accum 缩放也应同样应用于 per-head
            try:
                if self.args.n_gpu > 1:
                    val = val.mean()
            except Exception:
                pass
            if self.args.gradient_accumulation_steps > 1:
                val = val / self.args.gradient_accumulation_steps

            # 存为 CPU tensor（便于后续 stack & logging）
            self._accumulated_per_head_losses[name].append(val.detach().cpu())

        return loss.detach()
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=self._collate_fn, 
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset or self.eval_dataset
        )

        sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            collate_fn=self._collate_fn,  
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    def _get_train_sampler(self):
        # Use the trainer's built-in method for creating DistributedSampler
        if self.train_dataset is None:
            return None
            
        # For newer versions of transformers, use the built-in method
        return DistributedSampler(
            self.train_dataset,
            shuffle=True,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset):
        return DistributedSampler(
            eval_dataset,
            shuffle=False,
        )


    def _collate_fn(self, batch):
        """
        Compact collate: stack tensors and return position metadata.
        """
        # stack tensors
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])

        # position metadata (keep original list for backward compatibility)
        position = [b["position"] for b in batch]
        chroms = [p[0] for p in position]
        starts = [p[1] for p in position]
        ends = [p[2] for p in position]

        position_chrom = torch.tensor([self.chrom2id.get(c, 0) for c in chroms], dtype=torch.long)
        position_start = torch.tensor(starts, dtype=torch.long)
        position_end = torch.tensor(ends, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position": position,
            "position_chrom": position_chrom,
            "position_start": position_start,
            "position_end": position_end,
        }
    

    
    
class LocalLoggerCallback(TrainerCallback):
    def __init__(self, log_file_path: str, metrics_file_path: Optional[str] = None):
        super().__init__()
        self.log_file_path = log_file_path
        self.metrics_file_path = metrics_file_path or log_file_path.replace(".log", "_metrics.jsonl")
        self._model_ref = None  # 缓存 model 引用

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return

        _step = state.global_step
        _epoch = state.epoch or 0

        # 直接从 logs 提取所有字段
        content = []
        for k, v in logs.items():
            if k in {"epoch", "step", "runtime"}:
                continue
            if isinstance(v, float):
                content.append(f"{k}: {v:.6f}")
            else:
                content.append(f"{k}: {v}")

        message = f"📌 TRAINER LOG | Step: {_step} | Epoch: {_epoch:.2f} | " + " | ".join(content)
        dist_print(message)

        # --- 写入结构化 eval 日志 ---
        if logs and ("eval_loss" in logs or any(k.startswith("eval_") for k in logs.keys())):
            try:
                log_entry = {
                    "step": _step,
                    "epoch": round(_epoch, 4),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **{
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in logs.items()
                    }
                }
                with open(self.metrics_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                dist_print(f"[LocalLoggerCallback] 写入 metrics 失败: {e}")
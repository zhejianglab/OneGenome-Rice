# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""Distributed training helpers and Hugging Face Trainer callback."""

import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import init_process_group
from transformers import TrainerCallback


class DistributedSamplerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        if train_dataloader is not None and hasattr(train_dataloader.sampler, "set_epoch"):
            epoch = int(state.epoch) if state.epoch is not None else 0
            train_dataloader.sampler.set_epoch(epoch)
            if dist.is_initialized() and dist.get_rank() == 0:
                logging.info(f"[DistributedSamplerCallback] set_epoch({epoch}) called.")


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def dist_print(*args, **kwargs):
    if is_main_process():
        logging.info(*args, **kwargs)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist_print(
            f"[Distributed] rank={dist.get_rank()}, "
            f"world_size={dist.get_world_size()}, local_rank={local_rank}"
        )
        return local_rank, dist.get_world_size(), True
    else:
        dist_print("[Distributed] Single-GPU run")
        return 0, 1, False


def setup_sync_batchnorm(model, gpus_per_node: int = 8):
    """Convert BatchNorm layers to SyncBatchNorm with per-node process groups."""
    if not dist.is_initialized():
        dist_print("SyncBatchNorm: non-distributed environment, skipping conversion")
        return model

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size % gpus_per_node != 0:
        dist_print(
            f"SyncBatchNorm: world_size={world_size} not divisible by {gpus_per_node}, "
            f"falling back to world group"
        )
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    num_nodes = world_size // gpus_per_node
    bn_group = None
    bn_group_ranks = None
    for node_idx in range(num_nodes):
        ranks = list(range(node_idx * gpus_per_node, (node_idx + 1) * gpus_per_node))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            bn_group = group
            bn_group_ranks = ranks

    dist_print(f"SyncBatchNorm: rank={rank} assigned to per-node group {bn_group_ranks}")
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=bn_group)
    return model

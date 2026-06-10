# 标准库（内置模块）
import os
import random
import logging
import datetime
import sys
from typing import Optional, Union
import argparse

# 第三方库（pip 安装的包）
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, barrier


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_logging(
    output_base_dir: str,
    timestamp: Optional[str] = None,
    log_level: int = logging.INFO,
    log_filename: str = None,
) -> str:
    """
    配置日志系统：仅 rank 0 写入日志文件，所有 rank 输出到控制台。
    返回日志文件路径（所有进程都返回相同值）。
    """
    # 等待所有进程进入，确保 dist 初始化完成
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # 只有 rank 0 生成 timestamp
    if rank == 0:
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = ""

    # 广播 timestamp 给所有进程
    if dist.is_available() and dist.is_initialized():
        timestamp_list = [timestamp]
        dist.broadcast_object_list(timestamp_list, src=0)
        timestamp = timestamp_list[0]

    # 定义日志目录
    log_dir = os.path.join(output_base_dir, "logs")
    log_filepath = None

    # 获取 logger
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()  # 清除已有 handler

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )

    # 所有进程都添加控制台输出（可选）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 仅 rank 0 创建目录并添加文件 handler
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)

        if log_filename is None:
            log_filename = f"training_{timestamp}.log"
        else:
            log_filename = f"{log_filename}_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 广播 log_filepath 给所有进程（确保 callback 能拿到）
    if dist.is_available() and dist.is_initialized():
        log_filepath_list = [log_filepath]
        dist.broadcast_object_list(log_filepath_list, src=0)
        log_filepath = log_filepath_list[0]

    # 所有进程都设置日志级别
    logger.setLevel(log_level)

    # rank 0 打印日志路径
    if rank == 0:
        logger.info(f"✅ 日志系统初始化完成，日志文件: {log_filepath}")

    return log_filepath  # 所有进程都返回相同的路径

def dist_print(*args, **kwargs) -> None:
    """
    Print only from the main process (rank 0) in distributed training.
    Prevents duplicate outputs in multi-GPU settings.

    Args:
        *args: Arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    # 检查分布式训练是否已初始化
    if dist.is_available() and dist.is_initialized():
        # 只在 rank 0 进程打印
        if dist.get_rank() == 0:
            logging.info(*args, **kwargs)
            sys.stdout.flush()
    else:
        # 单卡运行时直接打印
        logging.info(*args, **kwargs)
        sys.stdout.flush()


def setup_distributed():
    """
    初始化分布式训练环境（DDP），自动判断是否为多卡模式。
    
    Returns:
        tuple: (local_rank, world_size, is_distributed)
            - local_rank (int): 当前进程的本地 GPU 编号（单卡为 0）
            - world_size (int): 总进程数（单卡为 1）
            - is_distributed (bool): 是否启用了分布式
    """
    # 检查是否已在分布式环境中
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if dist.is_initialized():
            # 已初始化，直接返回信息
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = dist.get_world_size()
            is_distributed = True
            logging.info(f"✅ [Distributed] 已初始化！rank={dist.get_rank()}, world_size={world_size}, local_rank={local_rank}")
        else:
            try:
                # 初始化进程组
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                init_process_group(backend=backend, init_method="env://")

                local_rank = int(os.environ["LOCAL_RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                is_distributed = True

                # 设置当前 GPU 设备
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)

                # 同步所有进程
                barrier()

                logging.info(
                    f"✅ [Distributed] 初始化成功！"
                    f" rank={dist.get_rank()}, world_size={world_size}, local_rank={local_rank}"
                )
            except Exception as e:
                logging.error(f"❌ [Distributed] 初始化失败: {e}")
                raise
    else:
        # 单卡模式
        local_rank = 0
        world_size = 1
        is_distributed = False
        logging.info("✅ [Distributed] 未检测到分布式环境变量，使用单卡模式。")

    return local_rank, world_size, is_distributed
    

def get_index(index_path):

    if os.path.exists(index_path):
        dist_print(f"🏷️ 索引文件已存在，直接加载: {index_path}")
        return pd.read_csv(index_path)
    else:
        dist_print(f"❌ 索引文件 {index_path} 不存在，请确认。")
        
        

def setup_sync_batchnorm(model, is_distributed: bool, gpus_per_node: int = 8):
    """
    为模型设置 SyncBatchNorm，支持分布式训练中的节点内同步。
    
    在多节点分布式训练中，只在每个节点内的 GPU 之间同步 BatchNorm 统计信息，
    耻于跨节点同步，以减少通信开销并提高训练效率。
    
    Args:
        model (torch.nn.Module): 需要处理的模型
        is_distributed (bool): 是否启用分布式训练
        gpus_per_node (int): 每个节点的 GPU 数量，默认为 8
        
    Returns:
        torch.nn.Module: 处理后的模型
        
    Example:
        model = setup_sync_batchnorm(model, is_distributed=True, gpus_per_node=8)
    """
    
    if is_distributed and dist.is_initialized():
        # world_size = 总进程数 = 节点数 * 每节点 GPU 数
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 验证 world_size 能被 gpus_per_node 整除
        assert world_size % gpus_per_node == 0, (
            f"world_size={world_size} 不能被 gpus_per_node={gpus_per_node} 整除，"
            "请检查多机多卡启动参数 (--nnodes, --nproc_per_node)"
        )

        num_nodes = world_size // gpus_per_node

        # 为每个"节点"创建一个进程组： [0..7], [8..15], [16..23], ...
        bn_group = None
        bn_group_ranks = None
        for node_idx in range(num_nodes):
            ranks = list(range(node_idx * gpus_per_node,
                               (node_idx + 1) * gpus_per_node))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                bn_group = group
                bn_group_ranks = ranks

        dist_print(
            f"🔗 SyncBatchNorm 使用每节点进程组，当前 rank={rank} "
            f"所在组 ranks={bn_group_ranks}"
        )

        # ✅ 关键：只在"本节点的 8 个 rank"内同步 BN
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model,
            process_group=bn_group,
        )
    else:
        # 单机单卡、单机多卡但没初始化 dist 的情况：退化成普通 SyncBN（或根本不用）
        dist_print("⚠️ 非分布式 / 未初始化 dist，使用默认 SyncBatchNorm（或可直接跳过）")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    return model
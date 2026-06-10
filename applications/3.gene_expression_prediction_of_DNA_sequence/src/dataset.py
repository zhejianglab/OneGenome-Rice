# 标准库（内置模块）
import os
import json
import argparse

# 第三方库（pip 安装的包）
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import pyBigWig
import pyfaidx
import logging
import pandas as pd
from typing import Callable, Dict, List

# 从自定义仓库中导入模块
from src.util import dist_print

# 配置日志格式
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def load_bigwig_signal(bw_path, chromosome, start, end, max_length=None):
    """
    从 BigWig 文件中加载指定区域的信号轨迹
    
    Args:
        bw_path (str): BigWig 文件路径
        chromosome (str): 染色体名称
        start (int): 起始位置
        end (int): 终止位置
        max_length (int, optional): 最大长度，仅用于截断（不进行填充）
    
    Returns:
        np.ndarray: 信号值数组
    """
    bw = None
    try:
        bw = pyBigWig.open(bw_path)
        raw_values = np.array(bw.values(chromosome, start, end))
        raw_values = np.nan_to_num(raw_values, nan=0.0)  # NaN 转换为 0.0
        
        if max_length is not None and len(raw_values) > max_length:
            values = raw_values[:max_length]
        else:
            values = raw_values
            
        return values
    finally:
        if bw is not None:
            bw.close()

def load_fasta_sequence(fasta, chromosome, start, end, max_length=None):
    """
    从 FASTA 文件中加载指定区域的序列
    
    Args:
        fasta (pyfaidx.Fasta): 已打开的 FASTA 对象
        chromosome (str): 染色体名称
        start (int): 起始位置
        end (int): 终止位置
        max_length (int, optional): 最大长度，仅用于截断（不进行填充）
    
    Returns:
        str: DNA 序列
    """
    #如果染色体为整数，则转换为字符串
    chromosome_str = str(chromosome)
    raw_seq = str(fasta[chromosome_str][start:end])
    
    if max_length is not None and len(raw_seq) > max_length:
        seq = raw_seq[:max_length]
    else:
        seq = raw_seq
        
    return seq


# class LazyGenomicDataset(Dataset):
#     def __init__(self, index_df, meta_json, tokenizer, max_length=32768, apply_scaling=False):
#         self.index_df = index_df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self._fasta = None  
#         self.apply_scaling = apply_scaling  
        
#         # 在初始化时打开并读取meta_json文件
#         with open(meta_json, 'r') as f:
#             self.meta_data = json.load(f)

#         # 从meta_data中获取fasta路径和bigwig目录
#         self.fasta_path = self.meta_data["summary"]["fasta_path"]
#         self.bigwig_dir = self.meta_data["summary"]["bigwig_dir"]

#     def print_attributes(self):
#         """
#         打印数据集的主要属性信息
#         """
#         print("=== LazyGenomicDataset 属性信息 ===")
#         print(f"数据集大小: {len(self.index_df)} 样本")
#         print(f"最大序列长度限制: {self.max_length}")
#         print(f"是否应用标签缩放: {self.apply_scaling}")
#         print(f"FASTA文件路径: {self.fasta_path}")
#         print(f"BigWig目录: {self.bigwig_dir}")
#         print(f"染色体数量: {len(self.index_df['chromosome'].unique())}")
#         print(f"文件数量: {len(self.index_df['file_name'].unique())}")
        
        
#         print("=== 前5行索引数据预览 ===")
#         print(self.index_df.head())

#     def _get_fasta(self):
#         if self._fasta is None:
#             self._fasta = pyfaidx.Fasta(self.fasta_path)
#         return self._fasta

#     def __len__(self):
#         return self.index_df.shape[0]

#     def __getitem__(self, idx):
#         row = self.index_df.iloc[idx]
#         fasta = self._get_fasta()

#         # 使用meta信息构建完整的bw_path
#         bw_path = os.path.join(self.bigwig_dir, row["file_name"])
        
#         # 加载序列和信号值
#         seq = load_fasta_sequence(fasta, row["chromosome"], row["start"],
#                                     row["end"], self.max_length)
        
#         signal_values = load_bigwig_signal(bw_path, row["chromosome"], 
#                                             row["start"], row["end"], self.max_length)

#         # 根据参数决定是否进行标签缩放
#         if self.apply_scaling:
#             scaled_labels = targets_scaling(signal_values, row["track_mean"])
#         else:
#             scaled_labels = signal_values  

#         # 分词
#         seq_prefixed = f"<{row['chromosome']}>" + row['prefix_token'] + seq # 添加染色体编号信息，添加前缀特殊token

#         encodings = self.tokenizer(
#             seq_prefixed,
#             padding="max_length",
#             max_length=self.max_length+2, # 添加2个前缀token
#             truncation=True,
#             return_tensors="pt",
#             return_attention_mask=False
#         )

#         return {
#             "input_ids": encodings["input_ids"].squeeze(0),
#             "labels": torch.tensor(scaled_labels, dtype=torch.float32),
#             "batch_name": row["prefix_token"],
#             "track_mean": torch.tensor(row["track_mean"], dtype=torch.float32),
#             # "track_mean": torch.tensor(1.0, dtype=torch.float32),
#             "sequence": seq_prefixed,
#             "position": (row["chromosome"], row["start"], row["end"]),
#             "file_name": row["file_name"]
#         }

#     def close(self):
#         if self._fasta is not None:
#             self._fasta.close()



def sample_viewer(sample):

    # 兼容 torch 张量的导出
    def to_numpy(x):
        try:
            # torch.Tensor
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(x, np.ndarray):
            return x
        try:
            return np.array(x)
        except Exception:
            return np.array([x])

    def scalar(x):
        # 从 numpy/torch/列表等取标量
        if x is None:
            return None
        x_np = to_numpy(x)
        if x_np.size == 1:
            return x_np.item()
        # 如果是字节或字符串数组，返回第一个元素字符串
        try:
            return x_np[0]
        except Exception:
            return x

    # 固定模态顺序（与 metrics.py 保持一致）
    modalities = ["total_RNA-seq_+", "total_RNA-seq_-", "ATAC-seq_."]

    input_ids = sample.get("input_ids", None)
    labels = sample.get("labels", None)
    sequence = sample.get("sequence", "")
    position = sample.get("position", None)
    files = sample.get("files", None)
    track_means = sample.get("track_means", None)
    biosample = sample.get("biosample", "")

    # 解析 position（兼容多种格式）
    chrom = start = end = None
    try:
        if position is None:
            chrom = start = end = None
        else:
            # 常见：position = (chrom, start, end)
            if isinstance(position, (list, tuple)) and len(position) == 3 and not any(isinstance(el, (list, tuple, np.ndarray)) for el in position):
                chrom = scalar(position[0])
                start = scalar(position[1])
                end = scalar(position[2])
            else:
                # 可能是单样本 tuple-of-lists (pos0_list, pos1_list, pos2_list)：尝试取第0元素
                try:
                    chrom = scalar(position[0])
                    start = scalar(position[1])
                    end = scalar(position[2])
                except Exception:
                    # 兜底
                    chrom = scalar(position)
    except Exception:
        chrom = start = end = None

    # 将 labels 转为 numpy，期望最终形状为 [L, C]
    lab = to_numpy(labels)
    if lab.ndim == 1:
        lab = lab[:, None]
    elif lab.ndim == 2:
        # 若是 [C, L] 并且 C==3（模态数），则转置为 [L, C]
        if lab.shape[0] == len(modalities) and lab.shape[1] != len(modalities):
            lab = lab.T
    else:
        # 高维时尝试压平前几维为 L，最后一维为通道
        lab = lab.reshape(-1, lab.shape[-1])

    seq_len = lab.shape[0]
    channels = lab.shape[1]

    # 通道名称优先使用固定 modalities（若匹配），否则使用 files 或通用名
    if channels == len(modalities):
        channel_names = modalities
    elif files:
        channel_names = [os.path.basename(f) for f in files]
    else:
        channel_names = [f"track_{i}" for i in range(channels)]

    # 颜色表
    default_colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    colors = [default_colors[i % len(default_colors)] for i in range(channels)]

    # --- 作图：每个模态一个子图（不绘制 track_means 虚线） ---
    nrows = channels
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 3 * max(1, nrows)), sharex=True)
    if nrows == 1:
        axes = [axes]
    title_pos = f"{chrom}:{start}-{end}" if chrom is not None else f"{biosample or ''}"
    fig.suptitle(f"Genomic Track Visualization  {title_pos}", fontsize=12)

    x = np.arange(seq_len) + (int(start) if start is not None else 0)
    for c, ax in enumerate(axes):
        ax.plot(x, lab[:, c], color=colors[c], linewidth=1)
        ax.set_ylabel('Signal')
        ax.set_title(f"{channel_names[c]}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('Genomic Position')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # 打印统计信息
    print(f"Sample Position: {chrom}:{start}-{end}")
    print(f"Biosample: {biosample}")
    try:
        seq_body = sequence if isinstance(sequence, str) else "".join(map(str, to_numpy(sequence)))[:50]
    except Exception:
        seq_body = ""
    # print(f"Sequence (preview, len ~50): {seq_body[:50]}")
    print(f"Labels shape: {lab.shape}")

    for c in range(channels):
        vals = lab[:, c]
        mn, mx = float(np.nanmin(vals)), float(np.nanmax(vals))
        mean = float(np.nanmean(vals))
        zero_ratio = float((vals == 0).mean()) * 100.0
        name = channel_names[c]
        tm_str = ""
        if track_means is not None and len(track_means) > c:
            try:
                tm = float(to_numpy(track_means)[c])
                tm_str = f", track_mean={tm:.6f}"
            except Exception:
                pass
        print(f"Channel {c} ({name}): min={mn:.6f}, max={mx:.6f}, mean={mean:.6f}{tm_str}, zeros={zero_ratio:.2f}%")
 
    



    
# class MultiTrackDataset(Dataset):
#     """
#     Multi-track genomic dataset.

#     Args:
#         sequence_split_df (pd.DataFrame): windows with columns ['chromosome','start','end'].
#         labels_meta_df (pd.DataFrame): metadata for tracks (contains 'target_file_name', 'nonzero_mean', ...).
#         index_stat (str): path to index_stat.json (contains fasta and bigwig dir info).
#         tokenizer: tokenizer for sequence to input_ids.
#         max_length (int): maximum sequence length / signal length.
#     """
#     def __init__(self, 
#                  sequence_split_df, 
#                  labels_meta_df, 
#                  index_stat, 
#                  tokenizer, 
#                  max_length=32768):
#         # window index (0-based, half-open)
#         self.sequence_split_df = sequence_split_df.reset_index(drop=True)
#         self.labels_meta_df = labels_meta_df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.index_stat = index_stat
        
#         # Lazy FASTA handle
#         self._fasta = None

#         # Cache for opened BigWig handles (filename -> pyBigWig object)
#         self._bw_handles = {}
        
#         # Paths from index_stat
#         self.fasta_path = self.index_stat["inputs"]["genome_fasta"]
#         self.bigwig_dir = self.index_stat["inputs"]["processed_bw_dir"]
        
#         self.target_files = self.index_stat["counts"]["target_file_name"]
#         self.nonzero_means = self.index_stat["counts"]["nonzero_mean"]

#     def print_stat(self):
#         """
#         Print main attributes of the dataset.
#         """
#         print("=== MultiTrackDataset Attributes ===")
#         print(f"Dataset size: {len(self.sequence_split_df)} samples")
#         print(f"Number of tracks: {len(self.target_files)}")
#         print(f"Number of heads: {len(self.index_stat['counts']['heads'])}")
#         print(f"Number of biosamples: {len(self.index_stat['counts']['biosample_order'])}")
#         print(f"Max sequence length: {self.max_length}")
#         print(f"FASTA path: {self.fasta_path}")
#         print(f"BigWig directory: {self.bigwig_dir}")
#         print(f"Number of chromosomes: {len(self.sequence_split_df['chromosome'].unique())}")
        
#         # 显示前5个序列切片
#         print("=== First 5 rows of sequence_split_df preview ===")
#         print(self.sequence_split_df.head())
#         # 显示前5个标签元数据
#         print("=== First 5 rows of labels_meta_df preview ===")
#         print(self.labels_meta_df.head())

#     def _get_fasta(self):
#         if self._fasta is None:
#             self._fasta = pyfaidx.Fasta(self.fasta_path)
#         return self._fasta
    
#     def _get_bw(self, filename):
#         """Get cached BigWig handle or open and cache it. Return None on failure."""
#         if filename in self._bw_handles:
#             return self._bw_handles[filename]
#         path = os.path.join(self.bigwig_dir, filename)
#         try:
#             bw = pyBigWig.open(path)
#             self._bw_handles[filename] = bw
#             return bw
#         except Exception as e:
#             logging.warning(f"Failed to open BigWig {path}: {e}")
#             return None
#     def __len__(self):
#         return self.sequence_split_df.shape[0]
    
#     def __getitem__(self, idx):
#         """Return one sample: sequence, input_ids, labels [L, C], track_means."""
#         row = self.sequence_split_df.iloc[idx]
#         fasta = self._get_fasta()

#         # 🔴 获取并确保 chromosome 是字符串类型
#         chromosome_val = row["chromosome"]
#         if not isinstance(chromosome_val, str):
#             chromosome_val = str(chromosome_val)

#         # Load sequence
#         seq = load_fasta_sequence(fasta, chromosome_val, row["start"], row["end"], self.max_length)

#         # Tokenize
#         encodings = self.tokenizer(
#             seq,
#             padding="max_length",
#             max_length=self.max_length,
#             truncation=True,
#             return_tensors="pt",
#             return_attention_mask=False
#         )

#         # Load all track values and pad/truncate to max_length
#         track_values = []
#         for bw_file in self.target_files:
#             bw = self._get_bw(bw_file)
#             if bw is None:
#                 vals = np.zeros(self.max_length, dtype=np.float32) if self.max_length else np.array([], dtype=np.float32)
#             else:
#                 vals = np.array(bw.values(str(row["chromosome"]), row["start"], row["end"]))
#                 vals = np.nan_to_num(vals, nan=0.0)
#                 if self.max_length is not None and len(vals) > self.max_length:
#                     vals = vals[:self.max_length]
#             track_values.append(vals)

#         # Stack to tensor shape [L, num_tracks]
#         tensor_list = [torch.tensor(tv, dtype=torch.float32) for tv in track_values]
#         labels = torch.empty((0, 0), dtype=torch.float32) if len(tensor_list) == 0 else torch.stack(tensor_list, dim=-1)

#         # # Track means (one per track)
#         # track_means = torch.tensor(self.nonzero_means, dtype=torch.float32)

#         return {
#             "position": (row["chromosome"], row["start"], row["end"]),
#             "sequence": seq,
#             "input_ids": encodings["input_ids"].squeeze(0),
#             "labels": labels,
#             # "track_means": track_means,
#         }

#     def close(self):
#         """Close opened FASTA and BigWig handles and clear cache."""
#         if self._fasta is not None:
#             self._fasta.close()
#         for fname, bw in list(self._bw_handles.items()):
#             try:
#                 if bw is not None:
#                     bw.close()
#             except Exception:
#                 pass
#         self._bw_handles.clear()


class MultiTrackDataset(Dataset):
    """
    Multi-track genomic dataset supporting multiple sequence splits and multiple index stats.

    Args:
        sequence_split_dfs (list or dict): 
            - list: List of pd.DataFrame with columns ['chromosome','start','end']
            - dict: Dictionary of {dataset_name: pd.DataFrame}
        index_stats (list or dict or str):
            - list: List of index_stat.json paths or loaded dicts
            - dict: Dictionary of {dataset_name: index_stat dict or path}
            - str: Single index_stat.json path (for backward compatibility)
        tokenizer: tokenizer for sequence to input_ids.
        max_length (int): maximum sequence length / signal length.
        mode (str): 'single' for single dataset, 'multi' for multiple datasets
    """
    def __init__(self, 
                 sequence_split_dfs, 
                 index_stats, 
                 tokenizer, 
                 max_length=32768,
                 mode='multi'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # 处理多数据集输入
        self._init_datasets(sequence_split_dfs, index_stats)
        
        # Lazy FASTA handle (支持多个FASTA)
        self._fasta_handles = {}
        
        # Cache for opened BigWig handles (dataset_name -> {filename -> pyBigWig object})
        self._bw_handles = {}
        
    def _init_datasets(self, sequence_split_dfs, index_stats):
        """初始化多数据集"""
        if self.mode == 'single':
            # 向后兼容：单数据集模式
            if isinstance(sequence_split_dfs, pd.DataFrame):
                sequence_split_dfs = {'default': sequence_split_dfs}
            if isinstance(index_stats, (str, dict)):
                index_stats = {'default': index_stats}
        
        # 确保输入格式正确
        if isinstance(sequence_split_dfs, list):
            # 列表转字典，使用默认名称
            self.datasets = {f'dataset_{i}': df for i, df in enumerate(sequence_split_dfs)}
        elif isinstance(sequence_split_dfs, dict):
            self.datasets = sequence_split_dfs
        else:
            raise ValueError("sequence_split_dfs should be list or dict")
            
        if isinstance(index_stats, list):
            # 列表转字典
            self.index_stats = {f'dataset_{i}': stat for i, stat in enumerate(index_stats)}
        elif isinstance(index_stats, dict):
            self.index_stats = index_stats
        else:
            # 单个index_stat
            self.index_stats = {'default': index_stats}
            
        # 检查数据集和index_stats是否匹配
        if set(self.datasets.keys()) != set(self.index_stats.keys()):
            print(f"Warning: dataset names mismatch. Datasets: {set(self.datasets.keys())}, "
                  f"Index stats: {set(self.index_stats.keys())}")
            # 只保留两者都有的
            common_keys = set(self.datasets.keys()) & set(self.index_stats.keys())
            self.datasets = {k: self.datasets[k] for k in common_keys}
            self.index_stats = {k: self.index_stats[k] for k in common_keys}
        
        # 重置所有数据集的索引并存储
        self.sequence_split_dfs = {}
        self.dataset_offsets = {}
        self.cumulative_sizes = []
        self.dataset_names = []
        
        total_size = 0
        for name, df in self.datasets.items():
            # 重置索引
            self.sequence_split_dfs[name] = df.reset_index(drop=True)
            
            # 加载或处理index_stat
            if isinstance(self.index_stats[name], str):
                # 如果是文件路径，加载JSON
                with open(self.index_stats[name], 'r') as f:
                    self.index_stats[name] = json.load(f)
            
            # 存储数据集信息
            dataset_size = len(self.sequence_split_dfs[name])
            self.dataset_offsets[name] = total_size
            self.cumulative_sizes.append(total_size + dataset_size)
            self.dataset_names.append(name)
            total_size += dataset_size
            
        # 存储总大小
        self.total_size = total_size
        
        # 验证每个数据集的必需字段
        for name in self.datasets.keys():
            self._validate_dataset(name)
    
    def _validate_dataset(self, dataset_name):
        """验证数据集的必需字段"""
        stat = self.index_stats[dataset_name]
        required_inputs = ["genome_fasta", "processed_bw_dir"]
        required_counts = ["target_file_name", "nonzero_mean"]
        
        for req in required_inputs:
            if req not in stat.get("inputs", {}):
                raise ValueError(f"Dataset {dataset_name}: Missing required field 'inputs.{req}' in index_stat")
        
        for req in required_counts:
            if req not in stat.get("counts", {}):
                raise ValueError(f"Dataset {dataset_name}: Missing required field 'counts.{req}' in index_stat")
    
    def _get_fasta(self, dataset_name):
        """获取缓存的FASTA句柄"""
        if dataset_name not in self._fasta_handles:
            fasta_path = self.index_stats[dataset_name]["inputs"]["genome_fasta"]
            try:
                self._fasta_handles[dataset_name] = pyfaidx.Fasta(fasta_path)
            except Exception as e:
                logging.error(f"Failed to open FASTA for dataset {dataset_name}: {e}")
                raise
        return self._fasta_handles[dataset_name]
    
    def _get_bw(self, dataset_name, filename):
        """获取缓存的BigWig句柄"""
        if dataset_name not in self._bw_handles:
            self._bw_handles[dataset_name] = {}
            
        if filename in self._bw_handles[dataset_name]:
            return self._bw_handles[dataset_name][filename]
            
        bigwig_dir = self.index_stats[dataset_name]["inputs"]["processed_bw_dir"]
        path = os.path.join(bigwig_dir, filename)
        try:
            bw = pyBigWig.open(path)
            self._bw_handles[dataset_name][filename] = bw
            return bw
        except Exception as e:
            logging.warning(f"Dataset {dataset_name}: Failed to open BigWig {path}: {e}")
            return None
    
    def _get_dataset_index(self, idx):
        """根据全局索引获取数据集名称和局部索引"""
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")
        
        # 查找数据集
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_sizes) and idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
            
        if dataset_idx >= len(self.dataset_names):
            raise IndexError(f"Index {idx} out of range")
            
        dataset_name = self.dataset_names[dataset_idx]
        local_idx = idx - (self.cumulative_sizes[dataset_idx-1] if dataset_idx > 0 else 0)
        
        return dataset_name, local_idx
    
    def print_stat(self):
        """打印所有数据集的统计信息"""
        print("=== MultiTrackDataset (Multi-Dataset Mode) ===")
        print(f"Total size: {self.total_size} samples")
        print(f"Number of datasets: {len(self.datasets)}")
        print("\n=== Per Dataset Statistics ===")
        
        for name in self.dataset_names:
            df = self.sequence_split_dfs[name]
            stat = self.index_stats[name]
            
            print(f"\n--- Dataset: {name} ---")
            print(f"  Size: {len(df)} samples")
            print(f"  Number of tracks: {len(stat['counts']['target_file_name'])}")
            print(f"  Number of heads: {len(stat['counts'].get('heads', []))}")
            print(f"  Number of biosamples: {len(stat['counts'].get('biosample_order', []))}")
            print(f"  FASTA path: {stat['inputs']['genome_fasta']}")
            print(f"  BigWig directory: {stat['inputs']['processed_bw_dir']}")
            print(f"  Number of chromosomes: {len(df['chromosome'].unique())}")
            
            # 显示前3个序列切片
            print(f"  First 3 rows preview:")
            print(df.head(3).to_string())
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """返回一个样本，支持跨数据集索引"""
        # 获取数据集和局部索引
        dataset_name, local_idx = self._get_dataset_index(idx)
        
        # 获取数据集特定的数据
        df = self.sequence_split_dfs[dataset_name]
        row = df.iloc[local_idx]
        stat = self.index_stats[dataset_name]
        
        # 获取FASTA
        fasta = self._get_fasta(dataset_name)
        
        # 确保chromosome是字符串类型
        chromosome_val = str(row["chromosome"])
        
        # 加载序列
        seq = load_fasta_sequence(fasta, chromosome_val, row["start"], row["end"], self.max_length)
        
        # Tokenize
        encodings = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False
        )
        
        # 加载所有track值
        target_files = stat["counts"]["target_file_name"]
        track_values = []
        
        for bw_file in target_files:
            bw = self._get_bw(dataset_name, bw_file)
            if bw is None:
                vals = np.zeros(self.max_length, dtype=np.float32)
            else:
                vals = np.array(bw.values(chromosome_val, row["start"], row["end"]))
                vals = np.nan_to_num(vals, nan=0.0)
                if self.max_length is not None and len(vals) > self.max_length:
                    vals = vals[:self.max_length]
            track_values.append(vals)
        
        # 堆叠为张量 [L, num_tracks]
        tensor_list = [torch.tensor(tv, dtype=torch.float32) for tv in track_values]
        labels = torch.stack(tensor_list, dim=-1) if tensor_list else torch.empty((self.max_length, 0), dtype=torch.float32)
        
        return {
            "position": (row["chromosome"], row["start"], row["end"]),
            "sequence": seq,
            "input_ids": encodings["input_ids"].squeeze(0),
            "labels": labels,
            "dataset_name": dataset_name,  # 添加数据集名称信息
            "dataset_idx": local_idx,      # 添加数据集内索引
        }
    
    def get_dataset_info(self, dataset_name):
        """获取特定数据集的信息"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        df = self.sequence_split_dfs[dataset_name]
        stat = self.index_stats[dataset_name]
        
        return {
            "name": dataset_name,
            "size": len(df),
            "num_tracks": len(stat["counts"]["target_file_name"]),
            "fasta_path": stat["inputs"]["genome_fasta"],
            "bigwig_dir": stat["inputs"]["processed_bw_dir"],
            "chromosomes": df["chromosome"].unique().tolist(),
            "start_range": (df["start"].min(), df["start"].max()),
            "end_range": (df["end"].min(), df["end"].max()),
        }
    
    def close(self):
        """关闭所有打开的FASTA和BigWig句柄"""
        # 关闭所有FASTA句柄
        for name, fasta in self._fasta_handles.items():
            try:
                if fasta is not None:
                    fasta.close()
            except Exception as e:
                logging.warning(f"Error closing FASTA for dataset {name}: {e}")
        
        # 关闭所有BigWig句柄
        for dataset_name, handles in self._bw_handles.items():
            for filename, bw in handles.items():
                try:
                    if bw is not None:
                        bw.close()
                except Exception as e:
                    logging.warning(f"Error closing BigWig {filename} for dataset {dataset_name}: {e}")
        
        self._fasta_handles.clear()
        self._bw_handles.clear()
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.close()


def create_targets_scaling_fn(
    index_stat: dict
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a scaling function that uses the track means to normalise and softclip the targets.
    """
    # Open bigwig files and compute track statistics
    track_means = index_stat["counts"]["nonzero_means"] 
    print(f"Track means: {track_means}")
    print(f"Number of tracks: {track_means.shape}")

    # Create tensor from computed means
    track_means_tensor = torch.tensor(track_means, dtype=torch.float32)

    def transform_fn(x: torch.Tensor) -> torch.Tensor:
        # Move constants to correct device then normalize
        means = track_means_tensor.to(x.device)
        scaled = x / means

        # Smooth clipping: if > 10, apply formula
        clipped = torch.where(
            scaled > 10.0,
            2.0 * torch.sqrt(scaled * 10.0) - 10.0,
            scaled,
        )
        return clipped

    return transform_fn



class MultiTrackDataset2(Dataset):
    """
    Multi-track genomic dataset supporting multiple sequence splits and multiple index stats.

    Args:
        sequence_split_dfs (list or dict): 
            - list: List of pd.DataFrame with columns ['chromosome','start','end']
            - dict: Dictionary of {dataset_name: pd.DataFrame}
        index_stats (list or dict or str):
            - list: List of index_stat.json paths or loaded dicts
            - dict: Dictionary of {dataset_name: index_stat dict or path}
            - str: Single index_stat.json path (for backward compatibility)
        tokenizer: tokenizer for sequence to input_ids.
        max_length (int): maximum sequence length / signal length.
        mode (str): 'single' for single dataset, 'multi' for multiple datasets
    """
    def __init__(self, 
                 sequence_split_dfs, 
                 index_stats, 
                 tokenizer, 
                 transform_fn: Callable[[torch.Tensor], torch.Tensor],
                 max_length=32768,
                 mode='multi'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.transform_fn = transform_fn
        # 处理多数据集输入
        self._init_datasets(sequence_split_dfs, index_stats)
        
        # Lazy FASTA handle (支持多个FASTA)
        self._fasta_handles = {}
        
        # Cache for opened BigWig handles (dataset_name -> {filename -> pyBigWig object})
        self._bw_handles = {}
        
    def _init_datasets(self, sequence_split_dfs, index_stats):
        """初始化多数据集"""
        if self.mode == 'single':
            # 向后兼容：单数据集模式
            if isinstance(sequence_split_dfs, pd.DataFrame):
                sequence_split_dfs = {'default': sequence_split_dfs}
            if isinstance(index_stats, (str, dict)):
                index_stats = {'default': index_stats}
        
        # 确保输入格式正确
        if isinstance(sequence_split_dfs, list):
            # 列表转字典，使用默认名称
            self.datasets = {f'dataset_{i}': df for i, df in enumerate(sequence_split_dfs)}
        elif isinstance(sequence_split_dfs, dict):
            self.datasets = sequence_split_dfs
        else:
            raise ValueError("sequence_split_dfs should be list or dict")
            
        if isinstance(index_stats, list):
            # 列表转字典
            self.index_stats = {f'dataset_{i}': stat for i, stat in enumerate(index_stats)}
        elif isinstance(index_stats, dict):
            self.index_stats = index_stats
        else:
            # 单个index_stat
            self.index_stats = {'default': index_stats}
            
        # 检查数据集和index_stats是否匹配
        if set(self.datasets.keys()) != set(self.index_stats.keys()):
            print(f"Warning: dataset names mismatch. Datasets: {set(self.datasets.keys())}, "
                  f"Index stats: {set(self.index_stats.keys())}")
            # 只保留两者都有的
            common_keys = set(self.datasets.keys()) & set(self.index_stats.keys())
            self.datasets = {k: self.datasets[k] for k in common_keys}
            self.index_stats = {k: self.index_stats[k] for k in common_keys}
        
        # 重置所有数据集的索引并存储
        self.sequence_split_dfs = {}
        self.dataset_offsets = {}
        self.cumulative_sizes = []
        self.dataset_names = []
        
        total_size = 0
        for name, df in self.datasets.items():
            # 重置索引
            self.sequence_split_dfs[name] = df.reset_index(drop=True)
            
            # 加载或处理index_stat
            if isinstance(self.index_stats[name], str):
                # 如果是文件路径，加载JSON
                with open(self.index_stats[name], 'r') as f:
                    self.index_stats[name] = json.load(f)
            
            # 存储数据集信息
            dataset_size = len(self.sequence_split_dfs[name])
            self.dataset_offsets[name] = total_size
            self.cumulative_sizes.append(total_size + dataset_size)
            self.dataset_names.append(name)
            total_size += dataset_size
            
        # 存储总大小
        self.total_size = total_size
        
        # 验证每个数据集的必需字段
        for name in self.datasets.keys():
            self._validate_dataset(name)
    
    def _validate_dataset(self, dataset_name):
        """验证数据集的必需字段"""
        stat = self.index_stats[dataset_name]
        required_inputs = ["genome_fasta", "processed_bw_dir"]
        required_counts = ["target_file_name", "nonzero_mean"]
        
        for req in required_inputs:
            if req not in stat.get("inputs", {}):
                raise ValueError(f"Dataset {dataset_name}: Missing required field 'inputs.{req}' in index_stat")
        
        for req in required_counts:
            if req not in stat.get("counts", {}):
                raise ValueError(f"Dataset {dataset_name}: Missing required field 'counts.{req}' in index_stat")
    
    def _get_fasta(self, dataset_name):
        """获取缓存的FASTA句柄"""
        if dataset_name not in self._fasta_handles:
            fasta_path = self.index_stats[dataset_name]["inputs"]["genome_fasta"]
            try:
                self._fasta_handles[dataset_name] = pyfaidx.Fasta(fasta_path)
            except Exception as e:
                logging.error(f"Failed to open FASTA for dataset {dataset_name}: {e}")
                raise
        return self._fasta_handles[dataset_name]
    
    def _get_bw(self, dataset_name, filename):
        """获取缓存的BigWig句柄"""
        if dataset_name not in self._bw_handles:
            self._bw_handles[dataset_name] = {}
            
        if filename in self._bw_handles[dataset_name]:
            return self._bw_handles[dataset_name][filename]
            
        bigwig_dir = self.index_stats[dataset_name]["inputs"]["processed_bw_dir"]
        path = os.path.join(bigwig_dir, filename)
        try:
            bw = pyBigWig.open(path)
            self._bw_handles[dataset_name][filename] = bw
            return bw
        except Exception as e:
            logging.warning(f"Dataset {dataset_name}: Failed to open BigWig {path}: {e}")
            return None
    
    def _get_dataset_index(self, idx):
        """根据全局索引获取数据集名称和局部索引"""
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")
        
        # 查找数据集
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_sizes) and idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
            
        if dataset_idx >= len(self.dataset_names):
            raise IndexError(f"Index {idx} out of range")
            
        dataset_name = self.dataset_names[dataset_idx]
        local_idx = idx - (self.cumulative_sizes[dataset_idx-1] if dataset_idx > 0 else 0)
        
        return dataset_name, local_idx
    
    def print_stat(self):
        """打印所有数据集的统计信息"""
        print("=== MultiTrackDataset (Multi-Dataset Mode) ===")
        print(f"Total size: {self.total_size} samples")
        print(f"Number of datasets: {len(self.datasets)}")
        print("\n=== Per Dataset Statistics ===")
        
        for name in self.dataset_names:
            df = self.sequence_split_dfs[name]
            stat = self.index_stats[name]
            
            print(f"\n--- Dataset: {name} ---")
            print(f"  Size: {len(df)} samples")
            print(f"  Number of tracks: {len(stat['counts']['target_file_name'])}")
            print(f"  Number of heads: {len(stat['counts'].get('heads', []))}")
            print(f"  Number of biosamples: {len(stat['counts'].get('biosample_order', []))}")
            print(f"  FASTA path: {stat['inputs']['genome_fasta']}")
            print(f"  BigWig directory: {stat['inputs']['processed_bw_dir']}")
            print(f"  Number of chromosomes: {len(df['chromosome'].unique())}")
            
            # 显示前3个序列切片
            print(f"  First 3 rows preview:")
            print(df.head(3).to_string())
    
    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        """返回一个样本，支持跨数据集索引"""
        # 获取数据集和局部索引
        dataset_name, local_idx = self._get_dataset_index(idx)
        
        # 获取数据集特定的数据
        df = self.sequence_split_dfs[dataset_name]
        row = df.iloc[local_idx]
        stat = self.index_stats[dataset_name]
        
        # 获取FASTA
        fasta = self._get_fasta(dataset_name)
        
        # 确保chromosome是字符串类型
        chromosome_val = str(row["chromosome"])
        
        # 加载序列
        seq = load_fasta_sequence(fasta, chromosome_val, row["start"], row["end"], self.max_length)
        
        # Tokenize
        encodings = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False
        )
        
        # 加载所有track值
        target_files = stat["counts"]["target_file_name"]
        track_values = []
        
        for bw_file in target_files:
            bw = self._get_bw(dataset_name, bw_file)
            if bw is None:
                vals = np.zeros(self.max_length, dtype=np.float32)
            else:
                vals = np.array(bw.values(chromosome_val, row["start"], row["end"]))
                vals = np.nan_to_num(vals, nan=0.0)
                vals = self.transform_fn(vals)  # 应用变换函数
                if self.max_length is not None and len(vals) > self.max_length:
                    vals = vals[:self.max_length]
            track_values.append(vals)
        
        labels=self.transform_fn(track_values)
        
        # 堆叠为张量 [L, num_tracks]
        #tensor_list = [torch.tensor(tv, dtype=torch.float32) for tv in track_values]
        #labels = torch.stack(tensor_list, dim=-1) if tensor_list else torch.empty((self.max_length, 0), dtype=torch.float32)
        
        return {
            "position": (row["chromosome"], row["start"], row["end"]),
            "sequence": seq,
            "input_ids": encodings["input_ids"].squeeze(0),
            "labels": labels,
            "dataset_name": dataset_name,  # 添加数据集名称信息
            "dataset_idx": local_idx,      # 添加数据集内索引
        }
    
    def get_dataset_info(self, dataset_name):
        """获取特定数据集的信息"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        df = self.sequence_split_dfs[dataset_name]
        stat = self.index_stats[dataset_name]
        
        return {
            "name": dataset_name,
            "size": len(df),
            "num_tracks": len(stat["counts"]["target_file_name"]),
            "fasta_path": stat["inputs"]["genome_fasta"],
            "bigwig_dir": stat["inputs"]["processed_bw_dir"],
            "chromosomes": df["chromosome"].unique().tolist(),
            "start_range": (df["start"].min(), df["start"].max()),
            "end_range": (df["end"].min(), df["end"].max()),
        }
    
    def close(self):
        """关闭所有打开的FASTA和BigWig句柄"""
        # 关闭所有FASTA句柄
        for name, fasta in self._fasta_handles.items():
            try:
                if fasta is not None:
                    fasta.close()
            except Exception as e:
                logging.warning(f"Error closing FASTA for dataset {name}: {e}")
        
        # 关闭所有BigWig句柄
        for dataset_name, handles in self._bw_handles.items():
            for filename, bw in handles.items():
                try:
                    if bw is not None:
                        bw.close()
                except Exception as e:
                    logging.warning(f"Error closing BigWig {filename} for dataset {dataset_name}: {e}")
        
        self._fasta_handles.clear()
        self._bw_handles.clear()
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.close()


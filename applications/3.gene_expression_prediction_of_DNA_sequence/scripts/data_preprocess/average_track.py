#!/usr/bin/env python3
"""
优化版：分染色体流式处理 bigWig 平均值，内存友好
修改：平均后为0的位置不写入 bigWig（留空 = NaN），减小文件体积
"""

import os
import warnings
from tqdm import tqdm
import pyBigWig
import numpy as np
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import time
import gc
import pandas as pd
import logging
from datetime import datetime
import shutil

# ----------------------------------------------------------
# 单进程：读取一个 bigWig 文件在指定染色体上的信号（返回 numpy array）
# ----------------------------------------------------------
def read_chrom_signal(args: Tuple[str, str, int, int]) -> Tuple[str, np.ndarray]:
    """
    读取单个 bigWig 文件中某染色体的信号
    
    Args:
        args: (bigwig_path, chrom, start, end)
    
    Returns:
        (chrom, signal_array)
    """
    bigwig_path, chrom, start, end = args
    try:
        bw = pyBigWig.open(bigwig_path)
        if bw is None:
            raise RuntimeError(f"无法打开: {bigwig_path}")
        if chrom not in bw.chroms():
            bw.close()
            return chrom, None
        # 读取信号
        signal = np.array(bw.values(chrom, start, end))
        signal = np.nan_to_num(signal, nan=0.0)
        bw.close()
        return chrom, signal
    except Exception as e:
        print(f"Error reading {bigwig_path} for {chrom}: {e}")
        return chrom, None


# ----------------------------------------------------------
# 对单条染色体计算平均信号并写入 bigWig（主流程）
# ----------------------------------------------------------
def process_chromosome(
    bigwig_paths: List[str],
    chrom: str,
    chrom_length: int,
    window_size: int,
    step: int,
    output_bw: pyBigWig,
    verbose: bool = True
):
    """
    处理单条染色体：并行读取多个文件，逐窗口平均，只写入非零平均值
    """
    positions = []
    start = 0
    while start <= chrom_length - window_size:
        end = start + window_size
        positions.append((start, end))
        start += step

    if not positions:
        if verbose:
            print(f"No windows for {chrom}")
        return

    # 并行读取所有文件在该染色体上的完整信号
    args_list = [(bw_path, chrom, 0, chrom_length) for bw_path in bigwig_paths]
    
    with Pool(processes=min(cpu_count(), len(bigwig_paths))) as pool:
        results = list(tqdm(
            pool.imap(read_chrom_signal, args_list),
            total=len(args_list),
            desc=f"Reading {chrom}",
            disable=not verbose
        ))

    # 过滤无效结果
    signals = [sig for ch, sig in results if ch == chrom and sig is not None]
    if len(signals) == 0:
        if verbose:
            print(f"⚠️ No valid signal for {chrom}")
        return

    if len(signals) != len(bigwig_paths):
        warnings.warn(f"Only {len(signals)}/{len(bigwig_paths)} files have {chrom}")

    # 逐窗口处理并写入非零平均值
    pbar = tqdm(positions, desc=f"Writing {chrom}", disable=not verbose)
    for start, end in pbar:
        # 提取每个文件在 [start, end] 的信号
        window_vals = []
        for sig in signals:
            if end <= len(sig):
                window_vals.append(sig[start:end])
            else:
                # 边界填充
                pad_len = end - len(sig)
                padded = np.pad(sig[start:], (0, pad_len), mode='constant', constant_values=0)
                window_vals.append(padded)

        if not window_vals:
            continue

        # 计算平均
        mean_vals = np.mean(window_vals, axis=0, dtype=np.float32)

        # === 关键修改：跳过全零区域，只写非零值 ===
        non_zero_mask = mean_vals != 0.0
        if not np.any(non_zero_mask):
            continue  # 整个窗口平均为0，不写入（bigWig 中即为 NaN）

        # 获取非零位置和对应值
        offsets = np.where(non_zero_mask)[0]
        write_starts = (start + offsets).tolist()
        write_values = mean_vals[non_zero_mask].tolist()

        try:
            output_bw.addEntries(
                [chrom] * len(write_values),
                write_starts,
                ends=[s + 1 for s in write_starts],
                values=write_values
            )
        except Exception as e:
            print(f"Failed to write {chrom}:{start}-{end}: {e}")

    # 手动释放
    del signals, window_vals, mean_vals
    gc.collect()


# ----------------------------------------------------------
# 主函数：按染色体循环处理
# ----------------------------------------------------------
def main():
    # Setup logging
    log_filename = f"average_track_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info("Starting average track processing")

    # 示例参数
    processed_data_dir = "/mnt/a100-nas-new/public/RNASEQ/renormalized_file"
    csv_path = "/mnt/a100-nas-new/personal/yecheng/worksapce/Genos-Reg/data/external/media.csv"
    chrom_sizes_file = "/mnt/a100-nas-new/personal/yecheng/worksapce/Genos-Reg/data/external/chrom.sizes"
    output_dir = "/mnt/zzbnew/Public/RNASEQ/averaged_file"
    output_type = "RNA_SEQ" # ATAC
    os.makedirs(output_dir, exist_ok=True)

    index_list = [299, 300, 304, 307, 309, 310, 358, 359, 378, 380, 
                  381, 383, 384, 386, 387, 390, 391, 392, 401, 403, 
                  404, 407, 429, 431, 434, 435, 437, 442, 446, 449, 
                  450, 456, 457, 458, 462, 463, 464, 466, 467, 468, 
                  469, 473, 474, 475, 490, 491, 492, 493, 494, 495, 496]
    
    # 读取 CSV，筛选
    df = pd.read_csv(csv_path)
    df = df[(df["track_index"].isin(index_list)) &
            (df["data_source"] == "encode") &
            (df["output_type"] == output_type) &
            (df['organism'] == 'human')]

    # 构建任务
    tasks = []
    for _, row in df.iterrows():
        file_accessions = [acc.strip() for acc in row['File accession'].split(',') if acc != '']
        input_files = [
            f"{processed_data_dir}/re-normalized_{name}.bigWig"
            for name in file_accessions
        ]
        tasks.append((input_files, row['track_index']))

    # 读取染色体大小 
    chrom_sizes = {}
    with open(chrom_sizes_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.split()
                chrom_sizes[parts[0]] = int(parts[1])

    # 处理每个任务
    for input_files, track_idx in tasks:
        start_time_i = time.time()
        # output_file = f"{output_dir}/{output_type}_track_avg_{track_idx}.bw"
        output_file = f"{output_dir}/rnaseq_track{track_idx}_mean.bigWig"
        logger.info(f"🚀 Processing track {track_idx} -> {output_file}") 

        # 如果只有一个文件，则直接复制
        if len(input_files) == 1:
            logger.info(f"Only one file for track {track_idx}, copying directly")
            shutil.copy2(input_files[0], output_file)
            logger.info(f"✅ Copied {input_files[0]} to {output_file}")
            logger.info(f"Time for track {track_idx}: {time.time() - start_time_i:.2f}s")
            continue

        # 打开输出 bigWig
        out_bw = pyBigWig.open(output_file, 'w')
        header = [(chrom, chrom_sizes[chrom]) for chrom in chrom_sizes.keys() if chrom in chrom_sizes]
        out_bw.addHeader(header)

        # 参数
        window_size = 32_000
        overlap = 0
        step = window_size - overlap

        # 按染色体处理
        for chrom, length in chrom_sizes.items():
            start_time_i_j = time.time()
            if chrom.startswith('chr') and chrom[3:].isdigit() and int(chrom[3:]) <= 22:  # chr1-chr22
                try:
                    process_chromosome(
                        bigwig_paths=input_files,
                        chrom=chrom,
                        chrom_length=length,
                        window_size=window_size,
                        step=step,
                        output_bw=out_bw,
                        verbose=True
                    )
                except Exception as e:
                    logger.error(f"Error processing {chrom}: {e}")
            logger.info(f"Time for track {track_idx} chrom {chrom}: {time.time() - start_time_i_j:.2f}s")
        out_bw.close()
        logger.info(f"✅ Saved {output_file}")
        logger.info(f"Time for track {track_idx}: {time.time() - start_time_i:.2f}s")
        gc.collect()
    
    logger.info(f"Total Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()

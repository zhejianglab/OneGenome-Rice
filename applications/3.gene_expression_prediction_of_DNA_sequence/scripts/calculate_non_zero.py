import pandas as pd
import numpy as np
import pyBigWig
from tqdm import tqdm
import multiprocessing as mp

def calculate_single_nonzero_mean(args):
    """计算单个文件的非零平均数（用于多进程）"""
    bw_path, use_weighted = args
    try:
        with pyBigWig.open(bw_path) as bw:
            chromosomes = bw.chroms()
            
            if use_weighted:
                # 加权计算
                weighted_sum = 0.0
                total_length = 0
                for chrom in chromosomes:
                    intervals = bw.intervals(chrom)
                    if intervals is None:
                        continue
                    for start, end, value in intervals:
                        if value != 0:
                            length = end - start
                            weighted_sum += value * length
                            total_length += length
                return weighted_sum / total_length if total_length > 0 else 0.0
            else:
                # 简单平均
                nonzero_vals = []
                for chrom in chromosomes:
                    intervals = bw.intervals(chrom)
                    if intervals is None:
                        continue
                    for _, _, value in intervals:
                        if value != 0:
                            nonzero_vals.append(value)
                return np.mean(nonzero_vals) if nonzero_vals else 0.0
                
    except Exception as e:
        print(f"处理 {bw_path} 出错: {e}")
        return None

def batch_calculate_nonzero_mean(bigwig_paths, use_weighted=True, num_processes=4):
    """
    批量计算多个BigWig文件的非零平均数
    
    参数:
    bigwig_paths: BigWig文件路径列表
    use_weighted: 是否使用加权平均（推荐True）
    num_processes: 进程数
    """
    if num_processes > 1:
        # 多进程计算
        with mp.Pool(num_processes) as pool:
            args = [(path, use_weighted) for path in bigwig_paths]
            results = list(tqdm(pool.imap(calculate_single_nonzero_mean, args), 
                               total=len(bigwig_paths), 
                               desc="计算非零平均数"))
    else:
        # 单进程计算
        results = []
        for path in tqdm(bigwig_paths, desc="计算非零平均数"):
            results.append(calculate_single_nonzero_mean((path, use_weighted)))
    
    return results

# 使用示例
import sys

if len(sys.argv) < 2:
    print("用法: python Untitled-1.py file1.bigWig file2.bigWig ...")
    sys.exit(1)

bigwig_files = sys.argv[1:]
nonzero_means = batch_calculate_nonzero_mean(bigwig_files, use_weighted=True)

# 创建结果DataFrame
df = pd.DataFrame({
    'file': bigwig_files,
    'nonzero_mean': nonzero_means
})
print(df)
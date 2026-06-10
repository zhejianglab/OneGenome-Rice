#!/usr/bin/env python3
"""
根据 CSV 中每行的 expression_values 数据生成 bigWig 文件
支持变长窗口（由 start 和 end 决定），并支持多染色体
"""
import argparse
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import pyBigWig
import os
import numpy as np

def str2arr(s: str):
    """把 "[a,b,c]" 字符串 → [a,b,c] 列表"""
    try:
        return json.loads(s.strip())
    except Exception as e:
        raise ValueError(f"无法解析表达式字符串: {s}, 错误: {e}")

def df_value_extract(df, expression_parsed_col):
    print("📌 步骤3: 累加重叠位置的表达值...")
    pos_data = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # pos_data[chrom][pos] = [sum, count]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理每一行"):
        chrom = row['chromosome']
        start = int(row['start'])
        end = int(row['end'])
        values = list(map(float, row[expression_parsed_col]))
        # 确保长度一致（再次确认）
        if len(values) != (end - start):
            continue  # 已过滤，但保险起见

        for j, val in enumerate(values):
            current_pos = start + j
            pos_data[chrom][current_pos][0] += val  # 累加
            pos_data[chrom][current_pos][1] += 1    # 计数
    
    averages = {}
    for chrom in sorted(pos_data.keys()):
        chrom_data = pos_data[chrom]
        sorted_positions = sorted(chrom_data.keys())
        averages[chrom] = [s / c for s, c in (chrom_data[pos] for pos in sorted_positions)]
    

    return averages


def merge_arrays_by_sorted_keys(dictionary):
    # 按键排序（自然排序）
    sorted_keys = sorted(dictionary.keys(), key=lambda x: int(x[3:]) if x[3:].isdigit() else x)
    return np.concatenate([dictionary[key] for key in sorted_keys])


def main():
    # ================== 命令行参数配置 ==================
    parser = argparse.ArgumentParser(description='Convert CSV with variable-length expression values per row to bigWig format.')
    parser.add_argument('--csv', required=True, help='Input CSV file path (must have chromosome, start, end, and expression_values columns)')
    parser.add_argument('--output', required=True, help='Output bigWig file path')
    parser.add_argument('--chrom_sizes', default="/mnt/zzbnew/peixunban/yecheng/workspace/downstream-task/rna-seq_coverage/scripts/evaluation/chrom.sizes", help='Chromosome sizes file path (two columns: chrom size)')
    parser.add_argument('--expression_col', default='expression_values', help='Column name containing expression values (default: expression_values)')
    
    args = parser.parse_args()

    # ================== 参数打印 ==================
    print("⚙️ 配置参数:")
    print(f"  - 输入 CSV: {args.csv}")
    print(f"  - 输出 bigWig: {args.output}")
    print(f"  - 染色体大小文件: {args.chrom_sizes}")
    print(f"  - 表达值列名: {args.expression_col}\n")

    # 1. 读取并解析 CSV
    print("📌 步骤1: 读取 CSV 文件...")
    df = pd.read_csv(args.csv)

    required_cols = {'chromosome', 'start', 'end', args.expression_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"错误：CSV 文件缺少必要列: {missing}")

    # 解析 expression_values 列
    print("📌 步骤2: 解析表达值...")
    df['parsed_expression'] = df[args.expression_col].apply(str2arr)
    df['parsed_true_expression'] = df['true_expression'].apply(str2arr)
    df['calculated_length'] = df['end'] - df['start']
    df['parsed_length'] = df['parsed_expression'].apply(len)

    # 检查长度是否匹配
    mismatch_mask = df['calculated_length'] != df['parsed_length']
    if mismatch_mask.any():
        print("⚠️ 警告：以下行的 (end - start) 与 expression_values 长度不一致，将被过滤：")
        print(df[mismatch_mask][['chromosome', 'start', 'end', 'calculated_length', 'parsed_length']])
        df = df[~mismatch_mask].copy()
        if len(df) == 0:
            raise ValueError("所有行的长度都不匹配，无法继续")

    y_prep_dict = df_value_extract(df, "parsed_expression")
    y_pred_array = merge_arrays_by_sorted_keys(y_prep_dict)
    y_true_dict = df_value_extract(df, "parsed_true_expression")
    y_true_array = merge_arrays_by_sorted_keys(y_true_dict)

    np.save(args.output+'.npy', y_pred_array)
    np.save(args.output+'_true.npy', y_true_array)

    # # 2. 累加每个染色体上每个位置的值（用于平均）
    # print("📌 步骤3: 累加重叠位置的表达值...")
    # pos_data = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # pos_data[chrom][pos] = [sum, count]

    # for _, row in tqdm(df.iterrows(), total=len(df), desc="处理每一行"):
    #     chrom = row['chromosome']
    #     start = int(row['start'])
    #     end = int(row['end'])
    #     values = list(map(float, row['parsed_expression']))

    #     # 确保长度一致（再次确认）
    #     if len(values) != (end - start):
    #         continue  # 已过滤，但保险起见

    #     for j, val in enumerate(values):
    #         current_pos = start + j
    #         pos_data[chrom][current_pos][0] += val  # 累加
    #         pos_data[chrom][current_pos][1] += 1    # 计数

    # # 3. 读取染色体大小
    # print("📌 步骤4: 读取染色体大小...")
    # chrom_sizes = {}
    # try:
    #     with open(args.chrom_sizes, 'r') as f:
    #         for line in f:
    #             line = line.strip()
    #             if line and not line.startswith('#'):
    #                 parts = line.split()
    #                 if len(parts) >= 2:
    #                     chrom, size = parts[0], int(parts[1])
    #                     chrom_sizes[chrom] = size
    # except Exception as e:
    #     raise IOError(f"无法读取染色体大小文件 {args.chrom_sizes}: {e}")

    # # 检查数据中出现的染色体是否在 chrom_sizes 中
    # for chrom in pos_data:
    #     if str(chrom) not in chrom_sizes:
    #         raise ValueError(f"错误：染色体 '{chrom}' 在 {args.chrom_sizes} 中未定义大小")

    # # 4. 写入 bigWig 文件
    # print("📌 步骤5: 生成 bigWig 文件...")
    # bw = pyBigWig.open(args.output, 'w')
    # if bw is None:
    #     raise IOError(f"无法创建 bigWig 文件: {args.output}")

    # # 构建 header：只包含出现在数据中的染色体
    # header = [(str(chrom), chrom_sizes[str(chrom)]) for chrom in pos_data.keys() if str(chrom) in chrom_sizes]
    # if not header:
    #     raise ValueError("没有有效的染色体可以写入")
    # bw.addHeader(header)

    # # 写入每个染色体的数据
    # for chrom in sorted(pos_data.keys()):
    #     chrom_data = pos_data[chrom]
    #     sorted_positions = sorted(chrom_data.keys())
    #     averages = [s / c for s, c in (chrom_data[pos] for pos in sorted_positions)]
        
    #     starts = sorted_positions
    #     ends = [pos + 1 for pos in starts]  # 每个位点占 1bp

    #     try:
    #         bw.addEntries([str(chrom)] * len(starts), starts, ends=ends, values=averages)
    #     except Exception as e:
    #         print(f"❌ 写入染色体 {chrom} 时出错: {e}")
    #         raise

    # bw.close() 
    # print(f"✅ 成功生成 bigWig 文件: {args.output}")
    
    # # 尝试删除原始 CSV 文件
    # try:
    #     os.remove(args.csv)
    #     print(f"🗑️ 已删除原始 CSV 文件: {args.csv}")
    # except Exception as e:
    #     print(f"⚠️ 无法删除原始 CSV 文件 {args.csv}: {e}")


if __name__ == "__main__":
    main()
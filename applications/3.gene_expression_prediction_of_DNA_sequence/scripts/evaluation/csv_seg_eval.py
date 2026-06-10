#!/usr/bin/env python3
"""
根据 CSV 中每行的 expression_values 数据生成 bigWig 文件
支持变长窗口（由 start 和 end 决定），并支持多染色体
添加每行区间的预测值与真实值的相关性统计
"""
import argparse
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import pyBigWig
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

def str2arr(s: str):
    """把 "[a,b,c]" 字符串 → [a,b,c] 列表"""
    try:
        return json.loads(s.strip())
    except Exception as e:
        raise ValueError(f"无法解析表达式字符串: {s}, 错误: {e}")

def df_value_extract(df, expression_parsed_col):
    """
    从DataFrame中提取并累加每个位置的表达值
    """
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

    return pos_data

def compute_correlation_metrics(pred_values, true_values):
    """
    计算预测值和真实值之间的多种相关性指标
    支持一维数组（所有值串联）和二维数组（每行一个区间）
    """
    metrics = {}
    
    # 转换为numpy数组
    pred_array = np.array(pred_values)
    true_array = np.array(true_values)
    
    # 检查数组维度
    if pred_array.ndim == 1 and true_array.ndim == 1:
        # 一维数组：所有值串联
        pred_1d = pred_array
        true_1d = true_array
        
        # 过滤掉全零的区域（避免除以零）
        non_zero_mask = (true_1d > 0) & (pred_1d > 0)
        pred_non_zero = pred_1d[non_zero_mask]
        true_non_zero = true_1d[non_zero_mask]
        
        # 1. 皮尔逊相关系数
        if len(pred_non_zero) > 1 and len(true_non_zero) > 1:
            if len(np.unique(pred_non_zero)) > 1 and len(np.unique(true_non_zero)) > 1:
                pearson_corr, _ = stats.pearsonr(pred_non_zero, true_non_zero)
                metrics['pearson'] = pearson_corr
            else:
                metrics['pearson'] = np.nan
        else:
            metrics['pearson'] = np.nan
        
        # 2. 斯皮尔曼相关系数
        if len(pred_non_zero) > 1 and len(true_non_zero) > 1:
            if len(np.unique(pred_non_zero)) > 1 and len(np.unique(true_non_zero)) > 1:
                spearman_corr, _ = stats.spearmanr(pred_non_zero, true_non_zero)
                metrics['spearman'] = spearman_corr
            else:
                metrics['spearman'] = np.nan
        else:
            metrics['spearman'] = np.nan
        
    elif pred_array.ndim == 2 and true_array.ndim == 2:
        # 二维数组：每行一个区间
        pred_2d = pred_array
        true_2d = true_array
        
        # 逐行计算相关性
        pearson_corrs = []
        spearman_corrs = []
        
        for p_row, t_row in zip(pred_2d, true_2d):
            # 过滤零值
            non_zero_mask = (t_row > 0) & (p_row > 0)
            p_row_nonzero = p_row[non_zero_mask]
            t_row_nonzero = t_row[non_zero_mask]
            
            if len(p_row_nonzero) > 1 and len(t_row_nonzero) > 1:
                if len(np.unique(p_row_nonzero)) > 1 and len(np.unique(t_row_nonzero)) > 1:
                    # Pearson
                    corr, _ = stats.pearsonr(p_row_nonzero, t_row_nonzero)
                    pearson_corrs.append(corr)
                    
                    # Spearman
                    corr, _ = stats.spearmanr(p_row_nonzero, t_row_nonzero)
                    spearman_corrs.append(corr)
        
        # 统计汇总
        if pearson_corrs:
            metrics['pearson_mean'] = np.mean(pearson_corrs)
            metrics['pearson_median'] = np.median(pearson_corrs)
            metrics['pearson_std'] = np.std(pearson_corrs)
            metrics['pearson_min'] = np.min(pearson_corrs)
            metrics['pearson_max'] = np.max(pearson_corrs)
            metrics['pearson_valid_rows'] = len(pearson_corrs)
            metrics['pearson'] = metrics['pearson_mean']  # 向后兼容
        else:
            metrics['pearson'] = np.nan
            
        if spearman_corrs:
            metrics['spearman_mean'] = np.mean(spearman_corrs)
            metrics['spearman_median'] = np.median(spearman_corrs)
            metrics['spearman_std'] = np.std(spearman_corrs)
            metrics['spearman_valid_rows'] = len(spearman_corrs)
            metrics['spearman'] = metrics['spearman_mean']  # 向后兼容
        else:
            metrics['spearman'] = np.nan
    else:
        # 维度不匹配，降为一维处理
        pred_flat = pred_array.flatten()
        true_flat = true_array.flatten()
        return compute_correlation_metrics(pred_flat, true_flat)
    
    # 3. 均方误差和平均绝对误差（使用所有值）
    mse = np.mean((pred_array - true_array) ** 2)
    mae = np.mean(np.abs(pred_array - true_array))
    
    metrics['mse'] = mse
    metrics['mae'] = mae
    metrics['rmse'] = np.sqrt(mse)
    
    # 4. R²得分
    ss_res = np.sum((true_array - pred_array) ** 2)
    ss_tot = np.sum((true_array - np.mean(true_array)) ** 2)
    if ss_tot > 0:
        metrics['r2_score'] = 1 - (ss_res / ss_tot)
    else:
        metrics['r2_score'] = np.nan
    
    # 5. 平均值和方差比较
    metrics['pred_mean'] = np.mean(pred_array)
    metrics['true_mean'] = np.mean(true_array)
    metrics['pred_std'] = np.std(pred_array)
    metrics['true_std'] = np.std(true_array)
    metrics['pred_max'] = np.max(pred_array)
    metrics['true_max'] = np.max(true_array)
    metrics['pred_min'] = np.min(pred_array)
    metrics['true_min'] = np.min(true_array)
    
    # 6. 零值比例
    metrics['pred_zero_ratio'] = np.mean(pred_array == 0)
    metrics['true_zero_ratio'] = np.mean(true_array == 0)
    
    # 7. 中位数和分位数
    metrics['pred_median'] = np.median(pred_array)
    metrics['true_median'] = np.median(true_array)
    metrics['pred_q25'] = np.percentile(pred_array, 25)
    metrics['pred_q75'] = np.percentile(pred_array, 75)
    metrics['true_q25'] = np.percentile(true_array, 25)
    metrics['true_q75'] = np.percentile(true_array, 75)
    
    return metrics, [], []  # 返回空列表以保持兼容性

def generate_correlation_report(df, output_base_path):
    """
    生成相关性统计报告和可视化
    """
    print("📊 计算相关性统计指标...")
    
    # 提取每行的预测值和真实值
    row_metrics = []
    all_pearson_corrs = []
    all_spearman_corrs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="逐行计算相关性"):
        pred_values = np.array(row['parsed_expression'], dtype=float)
        true_values = np.array(row['parsed_true_expression'], dtype=float)
        
        # 跳过长度不匹配的行
        if len(pred_values) != len(true_values):
            continue
            
        # 单行指标
        if len(pred_values) > 1:
            # 过滤零值
            non_zero_mask = (true_values > 0) & (pred_values > 0)
            pred_nonzero = pred_values[non_zero_mask]
            true_nonzero = true_values[non_zero_mask]
            
            if len(pred_nonzero) > 1 and len(true_nonzero) > 1:
                if len(np.unique(pred_nonzero)) > 1 and len(np.unique(true_nonzero)) > 1:
                    pearson_corr, _ = stats.pearsonr(pred_nonzero, true_nonzero)
                    spearman_corr, _ = stats.spearmanr(pred_nonzero, true_nonzero)
                    
                    all_pearson_corrs.append(pearson_corr)
                    all_spearman_corrs.append(spearman_corr)
                    
                    row_metrics.append({
                        'chromosome': row['chromosome'],
                        'start': row['start'],
                        'end': row['end'],
                        'length': len(pred_values),
                        'pearson_corr': pearson_corr,
                        'spearman_corr': spearman_corr,
                        'mse': np.mean((pred_values - true_values) ** 2),
                        'mae': np.mean(np.abs(pred_values - true_values)),
                        'pred_mean': np.mean(pred_values),
                        'true_mean': np.mean(true_values),
                        'pred_max': np.max(pred_values),
                        'true_max': np.max(true_values),
                        'pred_zero_ratio': np.mean(pred_values == 0),
                        'true_zero_ratio': np.mean(true_values == 0),
                        'non_zero_count': np.sum(non_zero_mask),
                    })
    
    # 创建详细的DataFrame
    row_metrics_df = pd.DataFrame(row_metrics)
    
    # 整体指标
    overall_metrics = {}
    
    if all_pearson_corrs:
        overall_metrics['pearson_mean'] = np.mean(all_pearson_corrs)
        overall_metrics['pearson_median'] = np.median(all_pearson_corrs)
        overall_metrics['pearson_std'] = np.std(all_pearson_corrs)
        overall_metrics['pearson_min'] = np.min(all_pearson_corrs)
        overall_metrics['pearson_max'] = np.max(all_pearson_corrs)
        overall_metrics['pearson_q25'] = np.percentile(all_pearson_corrs, 25)
        overall_metrics['pearson_q75'] = np.percentile(all_pearson_corrs, 75)
        overall_metrics['pearson'] = overall_metrics['pearson_mean']
    
    if all_spearman_corrs:
        overall_metrics['spearman_mean'] = np.mean(all_spearman_corrs)
        overall_metrics['spearman_median'] = np.median(all_spearman_corrs)
        overall_metrics['spearman_std'] = np.std(all_spearman_corrs)
        overall_metrics['spearman'] = overall_metrics['spearman_mean']
    
    # 保存逐行详细结果
    if len(row_metrics_df) > 0:
        detailed_csv_path = f"{output_base_path}_per_row_metrics.csv"
        row_metrics_df.to_csv(detailed_csv_path, index=False)
        print(f"✅ 逐行详细指标已保存至: {detailed_csv_path}")
        
        # 按染色体分组统计
        chrom_stats = row_metrics_df.groupby('chromosome').agg({
            'pearson_corr': ['mean', 'std', 'count'],
            'spearman_corr': ['mean', 'std'],
            'length': ['mean', 'sum'],
            'mse': ['mean'],
            'mae': ['mean']
        }).round(4)
        
        chrom_stats_path = f"{output_base_path}_chromosome_stats.csv"
        chrom_stats.to_csv(chrom_stats_path)
        print(f"✅ 染色体分组统计已保存至: {chrom_stats_path}")
    
    # 生成可视化图表
    if all_pearson_corrs and all_spearman_corrs:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Pearson相关系数分布
        axes[0, 0].hist(all_pearson_corrs, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Pearson Correlation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Pearson Correlation Distribution\nMean: {overall_metrics.get("pearson_mean", np.nan):.3f}')
        axes[0, 0].axvline(x=overall_metrics.get('pearson_mean', 0), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spearman相关系数分布
        axes[0, 1].hist(all_spearman_corrs, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Spearman Correlation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Spearman Correlation Distribution\nMean: {overall_metrics.get("spearman_mean", np.nan):.3f}')
        axes[0, 1].axvline(x=overall_metrics.get('spearman_mean', 0), color='red', linestyle='--', label='Mean')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 两种相关系数的散点图
        axes[0, 2].scatter(all_pearson_corrs, all_spearman_corrs, alpha=0.5, s=10)
        axes[0, 2].set_xlabel('Pearson Correlation')
        axes[0, 2].set_ylabel('Spearman Correlation')
        axes[0, 2].set_title('Pearson vs Spearman Correlation')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[0, 2].axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # 4. 相关系数与序列长度的关系
        if len(row_metrics_df) > 0:
            axes[1, 0].scatter(row_metrics_df['length'], row_metrics_df['pearson_corr'], alpha=0.5, s=10)
            axes[1, 0].set_xlabel('Sequence Length')
            axes[1, 0].set_ylabel('Pearson Correlation')
            axes[1, 0].set_title('Correlation vs Sequence Length')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加趋势线
            try:
                z = np.polyfit(row_metrics_df['length'], row_metrics_df['pearson_corr'], 1)
                p = np.poly1d(z)
                axes[1, 0].plot(row_metrics_df['length'], p(row_metrics_df['length']), "r--", alpha=0.8)
            except:
                pass
        
        # 5. MSE分布
        if 'mse' in row_metrics_df.columns and len(row_metrics_df['mse']) > 0:
            mse_values = row_metrics_df['mse']
            # 去除极端值以便更好地可视化
            if len(mse_values) > 10:
                mse_clipped = mse_values.clip(upper=np.percentile(mse_values, 95))
            else:
                mse_clipped = mse_values
                
            axes[1, 1].hist(mse_clipped, bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_xlabel('MSE')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Mean Squared Error Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 预测均值与真实均值的关系
        if len(row_metrics_df) > 0:
            axes[1, 2].scatter(row_metrics_df['pred_mean'], row_metrics_df['true_mean'], alpha=0.5, s=10)
            max_val = max(row_metrics_df['pred_mean'].max(), row_metrics_df['true_mean'].max())
            axes[1, 2].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
            axes[1, 2].set_xlabel('Predicted Mean')
            axes[1, 2].set_ylabel('True Mean')
            axes[1, 2].set_title('Predicted vs True Mean Expression')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{output_base_path}_correlation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 相关性可视化图已保存至: {plot_path}")
    
    # 保存整体统计报告
    report_path = f"{output_base_path}_correlation_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("         基因组信号预测相关性统计报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"数据集信息:\n")
        f.write(f"  总行数: {len(df)}\n")
        f.write(f"  有效计算相关性的行数: {len(all_pearson_corrs)}\n")
        f.write(f"  染色体数量: {df['chromosome'].nunique()}\n")
        f.write(f"  总碱基数: {(df['end'] - df['start']).sum():,}\n\n")
        
        f.write("整体统计指标:\n")
        for key, value in overall_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        if len(row_metrics_df) > 0:
            f.write(f"\n逐行指标统计:\n")
            f.write(f"  Pearson相关系数范围: [{row_metrics_df['pearson_corr'].min():.4f}, {row_metrics_df['pearson_corr'].max():.4f}]\n")
            f.write(f"  Spearman相关系数范围: [{row_metrics_df['spearman_corr'].min():.4f}, {row_metrics_df['spearman_corr'].max():.4f}]\n")
            f.write(f"  平均MSE: {row_metrics_df['mse'].mean():.4f}\n")
            f.write(f"  平均MAE: {row_metrics_df['mae'].mean():.4f}\n")
            f.write(f"  平均序列长度: {row_metrics_df['length'].mean():.1f} bp\n")
        
        f.write(f"\n预测值与真实值分布对比:\n")
        # 将所有值串联成一维数组
        all_pred = []
        all_true = []
        
        for _, row in df.iterrows():
            all_pred.extend(row['parsed_expression'])
            all_true.extend(row['parsed_true_expression'])
        
        overall_metrics_detailed, _, _ = compute_correlation_metrics(all_pred, all_true)
        
        for key in ['pearson', 'spearman', 'pred_mean', 'true_mean', 'pred_std', 'true_std', 
                   'pred_max', 'true_max', 'pred_min', 'true_min',
                   'pred_zero_ratio', 'true_zero_ratio', 'mse', 'mae', 'rmse', 'r2_score',
                   'pred_median', 'true_median']:
            if key in overall_metrics_detailed:
                value = overall_metrics_detailed[key]
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
    
    # 在generate_correlation_report函数中添加：
    print("\n🔍 诊断信息:")
    print(f"  预测值统计:")
    print(f"    均值: {np.mean(all_pred):.4f}, 标准差: {np.std(all_pred):.4f}")
    print(f"    中位数: {np.median(all_pred):.4f}, 99%分位数: {np.percentile(all_pred, 99):.4f}")
    print(f"    最大值: {np.max(all_pred):.4f}, 最小值: {np.min(all_pred):.4f}")
    print(f"    零值比例: {np.mean(np.array(all_pred) == 0):.4f}")

    print(f"\n  真实值统计:")
    print(f"    均值: {np.mean(all_true):.4f}, 标准差: {np.std(all_true):.4f}")
    print(f"    中位数: {np.median(all_true):.4f}, 99%分位数: {np.percentile(all_true, 99):.4f}")
    print(f"    最大值: {np.max(all_true):.4f}, 最小值: {np.min(all_true):.4f}")
    print(f"    零值比例: {np.mean(np.array(all_true) == 0):.4f}")

    # 检查是否存在异常大的预测值
    extreme_threshold = np.percentile(all_true, 99.9) * 100  # 真实值99.9%分位数的100倍
    extreme_pred_mask = np.array(all_pred) > extreme_threshold
    if np.any(extreme_pred_mask):
        extreme_count = np.sum(extreme_pred_mask)
        print(f"\n⚠️  发现 {extreme_count} 个异常大的预测值 (> {extreme_threshold:.2f})")
        print(f"    这些值范围: [{np.min(np.array(all_pred)[extreme_pred_mask]):.2f}, "
            f"{np.max(np.array(all_pred)[extreme_pred_mask]):.2f}]")
    
    print(f"✅ 详细统计报告已保存至: {report_path}")
    
    return overall_metrics


def analyze_extreme_values(predictions, true_values, output_path):
    """
    分析极端值对统计指标的影响
    """
    # 分位数分析
    pred_quantiles = np.percentile(predictions, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    true_quantiles = np.percentile(true_values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    
    # 极端值检测
    pred_q99 = pred_quantiles[-2]  # 99分位数
    extreme_mask = predictions > pred_q99
    extreme_count = np.sum(extreme_mask)
    extreme_ratio = extreme_count / len(predictions)
    
    # 去除极端值后的统计
    pred_filtered = predictions[~extreme_mask]
    true_filtered = true_values[~extreme_mask]
    
    if len(pred_filtered) > 0:
        pearson_filtered, _ = stats.pearsonr(pred_filtered, true_filtered)
        spearman_filtered, _ = stats.spearmanr(pred_filtered, true_filtered)
        mse_filtered = np.mean((pred_filtered - true_filtered) ** 2)
    else:
        pearson_filtered = np.nan
        spearman_filtered = np.nan
        mse_filtered = np.nan
    
    # 保存分析结果
    with open(output_path, 'w') as f:
        f.write("极端值分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总数据点数: {len(predictions):,}\n")
        f.write(f"极端值数量 (>99%分位数): {extreme_count:,} ({extreme_ratio*100:.2f}%)\n")
        f.write(f"99%分位数: {pred_q99:.2f}\n\n")
        
        f.write("预测值分位数:\n")
        for q, val in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], pred_quantiles):
            f.write(f"  {q}%: {val:.6f}\n")
        
        f.write("\n真实值分位数:\n")
        for q, val in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], true_quantiles):
            f.write(f"  {q}%: {val:.6f}\n")
        
        f.write("\n去除极端值后的指标:\n")
        f.write(f"  Pearson相关系数: {pearson_filtered:.4f}\n")
        f.write(f"  Spearman相关系数: {spearman_filtered:.4f}\n")
        f.write(f"  MSE: {mse_filtered:.4f}\n")
        
        # 分析极端值的影响
        if extreme_count > 0:
            f.write("\n极端值分析:\n")
            f.write(f"  极端预测值范围: [{predictions[extreme_mask].min():.2f}, {predictions[extreme_mask].max():.2f}]\n")
            f.write(f"  对应真实值范围: [{true_values[extreme_mask].min():.2f}, {true_values[extreme_mask].max():.2f}]\n")
            
            # 计算极端值对MSE的贡献
            mse_total = np.mean((predictions - true_values) ** 2)
            mse_extreme = np.mean((predictions[extreme_mask] - true_values[extreme_mask]) ** 2)
            mse_contribution = mse_extreme * extreme_count / (mse_total * len(predictions))
            f.write(f"  极端值对总MSE的贡献: {mse_contribution*100:.1f}%\n")
    
    return {
        'extreme_ratio': extreme_ratio,
        'pearson_filtered': pearson_filtered,
        'spearman_filtered': spearman_filtered,
        'mse_filtered': mse_filtered
    }


def main():
    # ================== 命令行参数配置 ==================
    parser = argparse.ArgumentParser(description='Convert CSV with variable-length expression values per row to bigWig format.')
    parser.add_argument('--csv', required=True, help='Input CSV file path (must have chromosome, start, end, and expression_values columns)')
    parser.add_argument('--output', required=True, help='Output bigWig file path')
    parser.add_argument('--chrom_sizes', default="/mnt/zzbnew/peixunban/yecheng/workspace/downstream-task/rna-seq_coverage/scripts/evaluation/chrom.sizes", help='Chromosome sizes file path (two columns: chrom size)')
    parser.add_argument('--expression_col', default='expression_values', help='Column name containing expression values (default: expression_values)')
    parser.add_argument('--skip_bigwig', action='store_true', help='跳过bigWig文件生成，只进行相关性分析')
    
    args = parser.parse_args()

    # ================== 参数打印 ==================
    print("⚙️ 配置参数:")
    print(f"  - 输入 CSV: {args.csv}")
    print(f"  - 输出 bigWig: {args.output}")
    print(f"  - 染色体大小文件: {args.chrom_sizes}")
    print(f"  - 表达值列名: {args.expression_col}")
    print(f"  - 跳过bigWig生成: {args.skip_bigwig}\n")

    # 1. 读取并解析 CSV
    print("📌 步骤1: 读取 CSV 文件...")
    df = pd.read_csv(args.csv)

    required_cols = {'chromosome', 'start', 'end', args.expression_col, 'true_expression'}
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
        mismatch_df = df[mismatch_mask][['chromosome', 'start', 'end', 'calculated_length', 'parsed_length']]
        print(mismatch_df.head())
        print(f"总计 {mismatch_mask.sum()} 行不匹配")
        df = df[~mismatch_mask].copy()
        if len(df) == 0:
            raise ValueError("所有行的长度都不匹配，无法继续")

    # 2. 进行相关性分析
    output_base = Path(args.output).stem
    output_dir = Path(args.output).parent
    metrics_output_base = output_dir / output_base
    
    correlation_metrics = generate_correlation_report(df, str(metrics_output_base))
    
    # 3. 生成bigWig文件（如果不需要则跳过）
    if not args.skip_bigwig:
        print("📌 步骤3: 生成bigWig文件...")
        
        # 提取并累加位置数据
        pred_pos_data = df_value_extract(df, "parsed_expression")
        
        # 读取染色体大小
        print("📌 步骤4: 读取染色体大小...")
        chrom_sizes = {}
        try:
            with open(args.chrom_sizes, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            chrom, size = parts[0], int(parts[1])
                            chrom_sizes[chrom] = size
        except Exception as e:
            raise IOError(f"无法读取染色体大小文件 {args.chrom_sizes}: {e}")

        # 检查数据中出现的染色体是否在 chrom_sizes 中
        for chrom in pred_pos_data:
            if str(chrom) not in chrom_sizes:
                print(f"⚠️ 警告：染色体 '{chrom}' 在 {args.chrom_sizes} 中未定义大小，跳过该染色体")
                continue

        # 写入 bigWig 文件
        print("📌 步骤5: 写入bigWig文件...")
        bw = pyBigWig.open(args.output, 'w')
        if bw is None:
            raise IOError(f"无法创建 bigWig 文件: {args.output}")

        # 构建 header：只包含出现在数据中的染色体
        header = [(str(chrom), chrom_sizes[str(chrom)]) for chrom in pred_pos_data.keys() 
                 if str(chrom) in chrom_sizes]
        if not header:
            raise ValueError("没有有效的染色体可以写入")
        bw.addHeader(header)

        # 写入每个染色体的数据
        for chrom in sorted(pred_pos_data.keys()):
            if str(chrom) not in chrom_sizes:
                continue
                
            chrom_data = pred_pos_data[chrom]
            sorted_positions = sorted(chrom_data.keys())
            averages = [s / c for s, c in (chrom_data[pos] for pos in sorted_positions)]
            
            starts = sorted_positions
            ends = [pos + 1 for pos in starts]  # 每个位点占 1bp

            try:
                bw.addEntries([str(chrom)] * len(starts), starts, ends=ends, values=averages)
                print(f"  ✅ 染色体 {chrom}: 写入 {len(starts)} 个位置")
            except Exception as e:
                print(f"❌ 写入染色体 {chrom} 时出错: {e}")
                continue

        bw.close() 
        print(f"✅ 成功生成 bigWig 文件: {args.output}")
    
    # 保存numpy数组
    print("📌 步骤6: 保存numpy数组...")
    # 提取所有预测值和真实值
    all_pred_values = []
    all_true_values = []
    
    for _, row in df.iterrows():
        all_pred_values.extend(row['parsed_expression'])
        all_true_values.extend(row['parsed_true_expression'])
    
    np.save(f"{args.output}.npy", np.array(all_pred_values))
    np.save(f"{args.output}_true.npy", np.array(all_true_values))
    print(f"✅ numpy数组已保存: {args.output}.npy 和 {args.output}_true.npy")
    
    print("\n" + "=" * 60)
    print("🎉 处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
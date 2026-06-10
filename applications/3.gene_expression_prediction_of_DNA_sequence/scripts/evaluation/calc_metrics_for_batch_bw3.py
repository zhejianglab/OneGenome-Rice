#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate metrics between multiple predicted and raw bigwig files
支持命令行参数配置所有路径和参数
"""
import argparse
import pyfaidx
import numpy as np
import pyBigWig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr 
import time
from math import sqrt
import os
import matplotlib.pyplot as plt
import re
import pyfaidx

def load_data_from_bigwig(bigwig_path, fasta_path, chromosome="chr19"):
    """
    Load data from a single bigwig file
    """
    genome = pyfaidx.Fasta(fasta_path)
    bw = pyBigWig.open(bigwig_path)
    
    for ref in genome.keys():  # 或者 genome.references
            # 匹配以chrom开头的染色体名
            if ref.startswith(chromosome):
                real_chrom_names=ref

    chromosome=real_chrom_names
    
    chrom_length = bw.chroms().get(chromosome)
    if not chrom_length:
        raise ValueError(f"Chromosome {chromosome} not found in {bigwig_path}")
    
    # Get all values for the chromosome
    values = np.array(bw.values(chromosome, 0, chrom_length))
    values = np.nan_to_num(values, nan=0.0)
    
    # Get sequence
    sequence = str(genome[chromosome][0:chrom_length])
    
    if len(sequence) != len(values):
        print(f"Warning: Length mismatch in {bigwig_path}")
    
    bw.close()
    genome.close()
    
    return sequence, values

def compute_metrics(y_pred, y_true, plot_path=None, title=None):
    """
    Compute zero-inflated regression metrics
    If plot_path is provided, create and save a scatter plot of non-zero true vs pred
    """
    # Convert to numpy arrays
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    
    # Filter out NaN and Inf values
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    
    if len(y_pred) == 0:
        return {metric: float("nan") for metric in [
            "mae", "mse", "rmse", "r2", "pearson", "spearman",
            "zero_auroc", "zero_auprc", "nonzero_auroc", "nonzero_auprc",
            "nonzero_pearson", "nonzero_spearman", "zero_ratio", "scatter_plot"
        ]}
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)
    log_transformed_pearson, _ = pearsonr(np.log(y_true+1), np.log(y_pred+1))
    spearman, _ = spearmanr(y_true, y_pred)
    
    # Zero-inflated metrics
    true_zero = (y_true == 0)
    true_nonzero = (y_true != 0)
    zero_ratio = np.mean(true_zero) * 100
    
    # AUROC & AUPRC
    try:
        zero_auroc = roc_auc_score(true_zero, -y_pred)
        zero_auprc = average_precision_score(true_zero, -y_pred)
        nonzero_auprc = average_precision_score(true_nonzero, y_pred)
    except ValueError as e:
        print(f"Warning: Cannot compute AUROC/AUPRC: {e}")
        zero_auroc = zero_auprc = nonzero_auprc = np.nan
    
    # Non-zero region metrics
    non_zero_mask = ~true_zero
    y_true_nonzero = y_true[non_zero_mask]
    y_pred_nonzero = y_pred[non_zero_mask]
    
    # default values if insufficient points
    nonzero_pearson = np.nan
    log_transformed_nonzero_pearson = np.nan
    nonzero_spearman = np.nan

    non_zero_mae = non_zero_mse = non_zero_rmse = np.nan
    if y_true_nonzero.size > 0:
        non_zero_mae = mean_absolute_error(y_true_nonzero, y_pred_nonzero)
        non_zero_mse = mean_squared_error(y_true_nonzero, y_pred_nonzero)
        non_zero_rmse = sqrt(non_zero_mse)

    if len(y_true_nonzero) >= 2:
        try:
            nonzero_pearson, _ = pearsonr(y_true_nonzero, y_pred_nonzero)
            log_transformed_nonzero_pearson, _ = pearsonr(np.log(y_true_nonzero+1), np.log(y_pred_nonzero+1))
            nonzero_spearman, _ = spearmanr(y_true_nonzero, y_pred_nonzero)
        except Exception:
            nonzero_pearson = log_transformed_nonzero_pearson = nonzero_spearman = np.nan

    scatter_path = None
    # Plot scatter if requested and there are non-zero points
    if plot_path is not None and y_true_nonzero.size > 0:
        try:
            # ensure output directory exists
            plot_dir = os.path.dirname(plot_path)
            if plot_dir:
                os.makedirs(plot_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(y_true_nonzero, y_pred_nonzero, color='blue', alpha=0.3, s=5)
            # 1:1 line
            mn = min(y_true_nonzero.min(), y_pred_nonzero.min())
            mx = max(y_true_nonzero.max(), y_pred_nonzero.max())
            ax.plot([mn, mx], [mn, mx], color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('y_true_nonzero')
            ax.set_ylabel('y_pred_nonzero')
            if title:
                ax.set_title(str(title))
            # correlation annotations
            ann_text = f"n={len(y_true_nonzero)}\npearson={np.nan_to_num(nonzero_pearson):.3f}\nspearman={np.nan_to_num(nonzero_spearman):.3f}\nrmse={np.nan_to_num(non_zero_rmse):.3f}"
            ax.text(0.05, 0.95, ann_text, transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            scatter_path = plot_path
        except Exception as e:
            print(f"Warning: Failed to create scatter plot {plot_path}: {e}")
    
    return {
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "pearson": round(float(pearson), 6),
        "log_transformed_pearson": round(float(log_transformed_pearson), 6),
        "spearman": round(spearman, 6),
        "zero_auroc": round(zero_auroc, 6) if not np.isnan(zero_auroc) else float('nan'),
        "zero_auprc": round(zero_auprc, 6) if not np.isnan(zero_auprc) else float('nan'),
        "nonzero_auprc": round(nonzero_auprc, 6) if not np.isnan(nonzero_auprc) else float('nan'),
        "non_zero_mae": round(non_zero_mae, 6) if not np.isnan(non_zero_mae) else float('nan'),
        "non_zero_mse": round(non_zero_mse, 6) if not np.isnan(non_zero_mse) else float('nan'),
        "non_zero_rmse": round(non_zero_rmse, 6) if not np.isnan(non_zero_rmse) else float('nan'),
        "nonzero_pearson": round(float(nonzero_pearson), 6) if not np.isnan(nonzero_pearson) else float('nan'),
        "nonzero_log_transformed_pearson": round(float(log_transformed_nonzero_pearson), 6) if not np.isnan(log_transformed_nonzero_pearson) else float('nan'),
        "nonzero_spearman": round(nonzero_spearman, 6) if not np.isnan(nonzero_spearman) else float('nan'),
        "zero_ratio": round(zero_ratio, 4),
        "scatter_plot": scatter_path
    }

def parse_file_list(file_list_str):
    """Parse comma-separated file list string into list"""
    return [f.strip() for f in file_list_str.split(",")]

def main():    
    # ================== 命令行参数配置 ==================
    parser = argparse.ArgumentParser(description='Calculate metrics between predicted and raw bigwig files.')
    parser.add_argument('--fasta', default="./hg38_cleaned.fa", help='Reference genome FASTA file path')
    parser.add_argument('--chrom', default='chr19', help='Target chromosome (default: chr19)')
    parser.add_argument('--pred_files', required=True, 
                       help='Comma-separated list of predicted bigwig files')
    parser.add_argument('--raw_files', required=True,
                       help='Comma-separated list of raw bigwig files')
    parser.add_argument('--titles', default="default title",
                       help='Comma-separated list of titles for each file pair')
    parser.add_argument('--output', required=True,
                       help='Output text file path for metrics')
    
    args = parser.parse_args()
    
    # 解析文件列表
    pred_files = parse_file_list(args.pred_files)
    raw_files = parse_file_list(args.raw_files)
    rna_seq_titles = pred_files
    
    # ================== 参数验证 ==================
    if len(pred_files) != len(raw_files) or len(pred_files) != len(rna_seq_titles):
        raise ValueError("Number of pred files, raw files, and RNA-seq titles must all match")
    
    # ================== 打印配置 ==================
    print("⚙️ Configuration:")
    print(f"  - Reference FASTA: {args.fasta}")
    print(f"  - Chromosome: {args.chrom}")
    print(f"  - Predicted files: {pred_files}")
    print(f"  - Raw files: {raw_files}")
    print(f"  - Titles: {rna_seq_titles}")
    print(f"  - Output file: {args.output}\n")
    
    # ================== 主流程 ==================
    start_time = time.time()
    
    plot_dir = os.path.join(os.path.dirname(args.output) or ".", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load all data
    print("Loading predicted and raw bigwig files...")
    all_pred_values = []
    all_raw_values = []
    
    for i, (pred_path, raw_path) in enumerate(zip(pred_files, raw_files)):
        print(f"Loading file pair {i+1}/{len(pred_files)} ({rna_seq_titles[i]})...")
        #_, pred_values = load_data_from_bigwig(pred_path, args.fasta, args.chrom)
        _, raw_values = load_data_from_bigwig(raw_path, args.fasta, args.chrom)
        pred_values = np.load(pred_path).tolist()
        # raw_values = np.load(raw_path).tolist()
        raw_values = raw_values[0:len(pred_values)]
        all_pred_values.append(pred_values)
        all_raw_values.append(raw_values)
    
    
    # Compute metrics for each pair and average them
    print("Computing metrics for each pair...")
    individual_metrics = []
    for i, (pred_vals, raw_vals) in enumerate(zip(all_pred_values, all_raw_values)):
        print(f"Computing metrics for pair {i+1} ({rna_seq_titles[i]})...")
        # safe name for file
        safe_title = re.sub(r"[^0-9A-Za-z_.+\-]", "_", rna_seq_titles[i])
        plot_path = os.path.join(plot_dir, f"pair_{i+1}_{safe_title}.png")
        pair_metrics = compute_metrics(pred_vals, raw_vals, plot_path=plot_path, title=rna_seq_titles[i])
        individual_metrics.append(pair_metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    metric_keys = list(individual_metrics[0].keys())
    for key in metric_keys:
        vals = []
        for m in individual_metrics:
            v = m.get(key, None)
            try:
                # try convert to float (will raise for strings like paths)
                fv = float(v)
            except Exception:
                continue
            if np.isnan(fv):
                continue
            vals.append(fv)
        if vals:
            avg_metrics[f"avg_{key}"] = round(float(np.mean(vals)), 6)
        else:
            avg_metrics[f"avg_{key}"] = float('nan')
    

        
    print("\nAverage metrics across individual pairs:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value}")
    
    # Save results to txt file
    with open(args.output, 'w') as f:
        
        f.write("\nAverage metrics across individual pairs:\n")
        f.write("-" * 50 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nIndividual pair metrics:\n")
        f.write("-" * 50 + "\n")
        for i, (pair_metrics, title) in enumerate(zip(individual_metrics, rna_seq_titles)):
            f.write(f"\nPair {i+1} ({title}):\n")
            for key, value in pair_metrics.items():
                f.write(f"  {key}: {value}\n")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Metrics saved to {args.output}")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 





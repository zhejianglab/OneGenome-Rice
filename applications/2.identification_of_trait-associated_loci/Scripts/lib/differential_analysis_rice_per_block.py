#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run per-block differential attention analysis for forward and reverse-complement matrices.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(attn_dir, file_name='hap1_attention_collapsed.csv'):
    attn_file = os.path.join(attn_dir, file_name)
    if not os.path.exists(attn_file):
        raise FileNotFoundError(f"File not found: {attn_file}")
    attn = pd.read_csv(attn_file, index_col=0)
    
    meta_file = os.path.join(attn_dir, 'metadata.csv')
    metadata = pd.read_csv(meta_file, index_col=0)
    
    common_samples = attn.index.intersection(metadata.index)
    return attn.loc[common_samples], metadata.loc[common_samples]


def differential_analysis(attn, metadata, group_col='sample_type', group_a=1, group_b=2, min_samples=10):
    mask_a = metadata[group_col] == group_a
    mask_b = metadata[group_col] == group_b
    
    group_a_data = attn.loc[mask_a]
    group_b_data = attn.loc[mask_b]
    
    n_a, n_b = len(group_a_data), len(group_b_data)
    if n_a < min_samples or n_b < min_samples:
        raise ValueError(f"Insufficient samples. Group {group_a}: {n_a}, Group {group_b}: {n_b}")
    
    results = []
    
    for pos in attn.columns.tolist():
        vals_a = group_a_data[pos].dropna().values
        vals_b = group_b_data[pos].dropna().values
        
        if len(vals_a) < min_samples or len(vals_b) < min_samples:
            results.append({'position': pos, 'pvalue': np.nan, 'log2fc': np.nan, 'padj': np.nan})
            continue
            
        mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
        
        if mean_a > 0 and mean_b > 0: log2fc = np.log2(mean_b / mean_a)
        elif mean_a == 0 and mean_b > 0: log2fc = np.inf
        elif mean_a > 0 and mean_b == 0: log2fc = -np.inf
        else: log2fc = 0
            
        try:
            stat, pvalue = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        except:
            stat, pvalue = np.nan, np.nan
            
        results.append({
            'position': pos, 'n_a': len(vals_a), 'n_b': len(vals_b),
            'mean_a': mean_a, 'mean_b': mean_b, 'log2fc': log2fc, 'pvalue': pvalue
        })
        
    df = pd.DataFrame(results)
    valid_pvals = df['pvalue'].notna()
    if valid_pvals.sum() > 0:
        df.loc[valid_pvals, 'padj'] = multipletests(df.loc[valid_pvals, 'pvalue'], method='fdr_bh')[1]
    else: df['padj'] = np.nan
    
    df['significant'] = (df['padj'] < 0.05) & (df['log2fc'].abs() > 1)
    return df


def plot_manhattan(results, output_file, title='Manhattan Plot', padj_thresh=0.05, fc_thresh=1):
    df = results[results['pvalue'].notna()].copy()
    if len(df) == 0: return

    df['pos_int'] = pd.to_numeric(df['position'].astype(str).str.replace('pos_', ''), errors='coerce')
    df = df.sort_values('pos_int')

    df['-log10p'] = -np.log10(df['padj'] + 1e-300)
    df['color'] = 'gray'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    fig.patch.set_facecolor('white')

    ax1.plot(df['pos_int'], df['-log10p'], color='gray', lw=1.5)
    ax1.axhline(-np.log10(padj_thresh), color='red', linestyle='--', linewidth=1)
    ax1.set_ylabel('-log10(Adjusted P)', fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight='bold')

    ax2.plot(df['pos_int'], df['log2fc'], color='black', lw=1.5)
    ax2.axhline(fc_thresh, color='red', linestyle='--', linewidth=1)
    ax2.axhline(-fc_thresh, color='red', linestyle='--', linewidth=1)
    ax2.axhline(0, color='gray', linewidth=1.2, alpha=0.3)
    ax2.set_xlabel('Genomic Position', fontsize=11)
    ax2.set_ylabel('log2(Fold Change)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def process_matrix(file_name, prefix, block_dir, block_output, args):
    attn, metadata = load_data(block_dir, file_name)
    results = differential_analysis(attn, metadata, group_a=args.group_a, group_b=args.group_b, min_samples=args.min_samples)
    
    results.to_csv(os.path.join(block_output, 'tables', f'{prefix}_all_results.csv'), index=False)
    
    plot_manhattan(
        results,
        os.path.join(block_output, 'figures', f'{prefix}_manhattan.png'),
        title=f'{prefix} - Manhattan Plot'
    )
    return results['significant'].sum()


def process_single_block(block_dir, block_name, args):
    block_output = os.path.join(args.output_dir, block_name)
    os.makedirs(os.path.join(block_output, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(block_output, 'tables'), exist_ok=True)
    
    try:
        n_sig_fwd = process_matrix('hap1_attention_collapsed.csv', f"fwd_group{args.group_a}vs{args.group_b}", block_dir, block_output, args)
        print(f"  {block_name} forward analysis finished, significant sites: {n_sig_fwd}")
        
        rev_file = os.path.join(block_dir, 'hap1_attention_collapsed_revcomp.csv')
        if os.path.exists(rev_file):
            n_sig_rev = process_matrix('hap1_attention_collapsed_revcomp.csv', f"revcomp_group{args.group_a}vs{args.group_b}", block_dir, block_output, args)
            print(f"  {block_name} reverse-complement analysis finished, significant sites: {n_sig_rev}")
            
    except Exception as e:
        print(f"  Skipping {block_name}: {e}")

def find_block_directories(input_dir):
    block_dirs = []
    input_path = Path(input_dir)
    for subdir in input_path.iterdir():
        if subdir.is_dir() and (subdir / "hap1_attention_collapsed.csv").exists():
            block_dirs.append((subdir, subdir.name))
    return sorted(block_dirs, key=lambda item: natural_block_key(item[1]))


def extract_block_id(name):
    match = re.search(r'block_(\d+)', str(name))
    return int(match.group(1)) if match else None


def natural_block_key(name):
    block_id = extract_block_id(name)
    if block_id is not None:
        return (0, block_id, name)
    return (1, name)

def main():
    parser = argparse.ArgumentParser(description='Run differential analysis across block directories.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing block subdirectories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--group_a', type=int, default=0, help='Control group label.')
    parser.add_argument('--group_b', type=int, default=1, help='Case group label.')
    parser.add_argument('--min_samples', type=int, default=10, help='Minimum samples per group.')
    parser.add_argument('--block_start', type=int, default=1, help='First block ID to process, inclusive and 1-based.')
    parser.add_argument('--block_end', type=int, default=0, help='Last block ID to process, inclusive and 1-based; 0 means all.')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    block_dirs = find_block_directories(args.input_dir)
    
    for block_dir, block_name in block_dirs:
        block_id = extract_block_id(block_name)
        if block_id is not None:
            if block_id < max(1, args.block_start):
                continue
            if args.block_end > 0 and block_id > args.block_end:
                continue
        process_single_block(block_dir, block_name, args)

if __name__ == '__main__':
    main()

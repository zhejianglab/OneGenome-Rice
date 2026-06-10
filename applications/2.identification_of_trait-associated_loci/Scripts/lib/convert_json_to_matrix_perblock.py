#!/usr/bin/env python3
"""
Convert normalized attention JSON files into matrices for differential analysis.
Supports forward and reverse-complement attention arrays.
"""

import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing normalized JSON or CSV sidecar files.")
    parser.add_argument("--bed_file", type=str, default="", help="BED file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--file_pattern", type=str, default="block_*_normalized.json", help="JSON glob pattern.")
    parser.add_argument("--input_format", choices=["auto", "json", "csv"], default="auto", help="Input format.")
    parser.add_argument("--block_start", type=int, default=1, help="First block ID to process, inclusive and 1-based.")
    parser.add_argument("--block_end", type=int, default=0, help="Last block ID to process, inclusive and 1-based; 0 means all.")
    return parser.parse_args()

def extract_block_name(filename):
    stem = Path(filename).stem
    match = re.match(r'(block_\d+)', stem)
    if match: return match.group(1)
    return stem

def extract_block_id(name):
    match = re.search(r'block_(\d+)', str(name))
    return int(match.group(1)) if match else None

def natural_file_key(path_obj):
    block_id = extract_block_id(path_obj.name)
    if block_id is not None:
        return (0, block_id, path_obj.name)
    return (1, path_obj.name)

def load_bed_positions(bed_file):
    bed_df = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
    block_positions = {}
    for idx, row in bed_df.iterrows():
        block_name = f"block_{idx+1}"
        block_positions[block_name] = {
            'chrom': str(row['chrom']), 'start': int(row['start']), 'end': int(row['end'])
        }
    return block_positions

def process_single_json(json_file, block_name, pos_info):
    start = pos_info['start']
    end = pos_info['end']
    
    with open(json_file, 'r') as f:
        samples = json.load(f)
    
    fwd_data = []
    rev_data = []
    
    for sample in samples:
        sample_id = sample['spec']
        label = sample['label']
        
        scores_fwd = sample.get('sequence_attention', [])
        expected_len = end - start
        
        if len(scores_fwd) != expected_len:
            if len(scores_fwd) > expected_len: scores_fwd = scores_fwd[:expected_len]
            else: scores_fwd = scores_fwd + [np.nan] * (expected_len - len(scores_fwd))
            
        for i, score in enumerate(scores_fwd):
            fwd_data.append({
                'sample_id': sample_id, 'label': label, 'chrom': pos_info['chrom'],
                'position': start + i + 1, 'attention': score
            })
            
        if 'sequence_revcomp_attention' in sample:
            scores_rev = sample['sequence_revcomp_attention']
            if len(scores_rev) != expected_len:
                if len(scores_rev) > expected_len: scores_rev = scores_rev[:expected_len]
                else: scores_rev = scores_rev + [np.nan] * (expected_len - len(scores_rev))
                
            for i, score in enumerate(scores_rev):
                rev_data.append({
                    'sample_id': sample_id, 'label': label, 'chrom': pos_info['chrom'],
                    'position': start + i + 1, 'attention': score
                })
                
    return pd.DataFrame(fwd_data), pd.DataFrame(rev_data) if rev_data else None

def find_csv_sidecar(json_path, block_name, direction):
    patterns = [
        f"{block_name}_{direction}_attention.csv",
        f"{block_name}_{direction}_attention.csv.gz",
    ]
    for pattern in patterns:
        candidate = json_path / pattern
        if candidate.exists():
            return candidate
    return None

def process_single_csv(json_path, block_name, pos_info):
    fwd_file = find_csv_sidecar(json_path, block_name, "fwd")
    if fwd_file is None:
        raise FileNotFoundError(f"Missing fwd CSV sidecar for {block_name}")

    rev_file = find_csv_sidecar(json_path, block_name, "revcomp")
    meta_file = json_path / f"{block_name}_metadata.csv"

    fwd_matrix = pd.read_csv(fwd_file, index_col=0)
    rev_matrix = pd.read_csv(rev_file, index_col=0) if rev_file else None

    if meta_file.exists():
        metadata = pd.read_csv(meta_file, index_col=0)
    else:
        metadata = pd.DataFrame({"sample_type": np.nan}, index=fwd_matrix.index)
        metadata.index.name = "sample_id"

    expected_columns = [f"pos_{pos}" for pos in range(pos_info["start"] + 1, pos_info["end"] + 1)]
    fwd_matrix = fwd_matrix.reindex(columns=expected_columns)
    if rev_matrix is not None:
        rev_matrix = rev_matrix.reindex(columns=expected_columns)

    return fwd_matrix, rev_matrix, metadata

def create_attention_matrix(df):
    pivot_df = df.pivot_table(index='sample_id', columns='position', values='attention', aggfunc='first')
    pivot_df.columns = [f"pos_{col}" for col in pivot_df.columns]
    return pivot_df

def create_metadata(df):
    sample_labels = df.groupby('sample_id')['label'].first()
    metadata = pd.DataFrame({'sample_id': sample_labels.index, 'sample_type': sample_labels.values})
    return metadata.set_index('sample_id')

def save_block_results(block_name, fwd_df, rev_df, output_dir):
    block_dir = os.path.join(output_dir, block_name)
    os.makedirs(block_dir, exist_ok=True)
    
    fwd_matrix = create_attention_matrix(fwd_df)
    metadata = create_metadata(fwd_df)
    
    fwd_matrix.to_csv(os.path.join(block_dir, "hap1_attention_collapsed.csv"))
    
    # hap2 boilerplate for downstream script
    hap2_matrix = pd.DataFrame(np.nan, index=fwd_matrix.index, columns=fwd_matrix.columns)
    hap2_matrix.to_csv(os.path.join(block_dir, "hap2_attention_collapsed.csv"))
    
    metadata.to_csv(os.path.join(block_dir, "metadata.csv"))
    
    if rev_df is not None:
        rev_matrix = create_attention_matrix(rev_df)
        rev_matrix.to_csv(os.path.join(block_dir, "hap1_attention_collapsed_revcomp.csv"))

    summary = {
        'block_name': block_name, 'samples': int(len(fwd_matrix)),
        'positions': int(len(fwd_matrix.columns)),
        'has_revcomp': rev_df is not None
    }
    with open(os.path.join(block_dir, "block_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

def save_block_matrices(block_name, fwd_matrix, rev_matrix, metadata, output_dir):
    block_dir = os.path.join(output_dir, block_name)
    os.makedirs(block_dir, exist_ok=True)

    fwd_matrix.to_csv(os.path.join(block_dir, "hap1_attention_collapsed.csv"))
    hap2_matrix = pd.DataFrame(np.nan, index=fwd_matrix.index, columns=fwd_matrix.columns)
    hap2_matrix.to_csv(os.path.join(block_dir, "hap2_attention_collapsed.csv"))
    metadata.to_csv(os.path.join(block_dir, "metadata.csv"))

    if rev_matrix is not None:
        rev_matrix.to_csv(os.path.join(block_dir, "hap1_attention_collapsed_revcomp.csv"))

    summary = {
        'block_name': block_name,
        'samples': int(len(fwd_matrix)),
        'positions': int(len(fwd_matrix.columns)),
        'has_revcomp': rev_matrix is not None
    }
    with open(os.path.join(block_dir, "block_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    block_positions = load_bed_positions(args.bed_file)
    json_path = Path(args.json_dir)
    block_start = max(1, int(args.block_start))
    block_end = int(args.block_end)

    if args.input_format in ("auto", "csv"):
        csv_files = sorted(json_path.glob("block_*_fwd_attention.csv*"), key=natural_file_key)
        if csv_files:
            for csv_file in tqdm(csv_files, desc="Converting CSV sidecars to matrices"):
                block_name = extract_block_name(csv_file.name)
                block_id = extract_block_id(block_name)
                if block_id is None:
                    continue
                if block_id < block_start:
                    continue
                if block_end > 0 and block_id > block_end:
                    continue
                if block_name not in block_positions:
                    continue
                pos_info = block_positions[block_name]
                fwd_matrix, rev_matrix, metadata = process_single_csv(json_path, block_name, pos_info)
                save_block_matrices(block_name, fwd_matrix, rev_matrix, metadata, args.output_dir)
            return
        if args.input_format == "csv":
            raise FileNotFoundError(f"No block_*_fwd_attention.csv* files found in {json_path}")

    json_files = sorted(json_path.glob(args.file_pattern), key=natural_file_key)

    for json_file in tqdm(json_files, desc="Converting JSON files to matrices"):
        block_name = extract_block_name(json_file.name)
        block_id = extract_block_id(block_name)
        if block_id is None:
            continue
        if block_id < block_start:
            continue
        if block_end > 0 and block_id > block_end:
            continue
        if block_name not in block_positions: continue

        pos_info = block_positions[block_name]
        fwd_df, rev_df = process_single_json(json_file, block_name, pos_info)

        save_block_results(block_name, fwd_df, rev_df, args.output_dir)

if __name__ == "__main__":
    main()

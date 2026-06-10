#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalize attention-score arrays against precomputed genomic coordinate lists.
- Inserted bases sharing the same reference coordinate are averaged.
- Forward and reverse-complement mappings are normalized independently.
- All input paths are passed explicitly through command-line arguments.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Normalize attention scores by exact genomic coordinates.')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing attention JSON files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    # Default origin data
    parser.add_argument('--bed_file', default="")
    parser.add_argument('--fasta_file', default="")
    parser.add_argument('--file_pattern', type=str, default='block_*_attn.json', help='Input attention JSON glob pattern.')
    parser.add_argument('--block_start', type=int, default=1, help='First block ID to process, inclusive and 1-based.')
    parser.add_argument('--block_end', type=int, default=0, help='Last block ID to process, inclusive and 1-based; 0 means all.')
    parser.add_argument('--write_csv', action='store_true', help='Also write normalized attention matrix CSV sidecars.')
    parser.add_argument('--no_json', action='store_true', help='Write only CSV sidecars and skip normalized JSON output.')
    parser.add_argument('--csv_compression', choices=['none', 'gzip'], default='gzip', help='CSV sidecar compression mode.')
    return parser.parse_args()


def extract_block_id(name):
    match = re.search(r'block_(\d+)', str(name))
    return int(match.group(1)) if match else None


def natural_file_key(path_obj):
    block_id = extract_block_id(path_obj.name)
    if block_id is not None:
        return (0, block_id, path_obj.name)
    return (1, path_obj.name)


def csv_suffix(args):
    return '.csv.gz' if args.csv_compression == 'gzip' else '.csv'


def write_matrix_csv(block_name, matrix_data, output_dir, suffix, direction, columns):
    if not matrix_data:
        return None
    out_path = os.path.join(output_dir, f"{block_name}_{direction}_attention{suffix}")
    matrix_df = pd.DataFrame.from_dict(matrix_data, orient='index')
    matrix_df.index.name = 'sample_id'
    matrix_df.columns = columns
    compression = 'gzip' if suffix.endswith('.gz') else None
    matrix_df.to_csv(out_path, compression=compression)
    return out_path


def normalize_attention_scores_with_poslist(scores, pos_list, start, end):
    """
    Convert model attention scores into a complete 1-based reference coordinate array.
    """
    ref_len = end - start
    
    sum_dict = {pos: 0.0 for pos in range(start + 1, end + 1)}
    cnt_dict = {pos: 0 for pos in range(start + 1, end + 1)}
    
    length = min(len(scores), len(pos_list))
    
    for i in range(length):
        p = pos_list[i]
        s = scores[i]
        if p in sum_dict:
            sum_dict[p] += float(s)
            cnt_dict[p] += 1
            
    normalized = []
    for pos in range(start + 1, end + 1):
        if cnt_dict[pos] > 0:
            normalized.append(sum_dict[pos] / cnt_dict[pos])
        else:
            normalized.append(0.0)
            
    return normalized

def process_block(block_name, json_file, bed_row, output_dir, args):
    """Process one block."""
    chrom = str(bed_row['chrom'])
    # BED is 0-based; output positions are 1-based reference coordinates.
    start = int(bed_row['start'])
    end = int(bed_row['end'])
    
    print(f"\n{block_name}: chr{chrom}:{start}-{end}")
    
    with open(json_file, 'r') as f:
        attention_data = json.load(f)
    
    normalized_data = []
    fwd_matrix_data = {}
    rev_matrix_data = {}
    sample_labels = {}
    
    for item in tqdm(attention_data, desc=f"  {block_name} normalize", leave=False):
        sample_id = item['spec']
        label = item['label']
        sample_labels[sample_id] = label
        
        scores_fwd = item.get('sequence_attention', [])
        pos_fwd = item.get('pos_list', [])
        
        norm_fwd = normalize_attention_scores_with_poslist(scores_fwd, pos_fwd, start, end)
        fwd_matrix_data[sample_id] = norm_fwd
        
        res = {
            'label': label,
            'spec': sample_id,
            'loc': block_name,
            'sequence_attention': norm_fwd
        }
        
        if 'sequence_revcomp_attention' in item and 'pos_list_revcomp' in item:
            scores_rev = item['sequence_revcomp_attention']
            pos_rev = item['pos_list_revcomp']
            norm_rev = normalize_attention_scores_with_poslist(scores_rev, pos_rev, start, end)
            res['sequence_revcomp_attention'] = norm_rev
            rev_matrix_data[sample_id] = norm_rev
            
        normalized_data.append(res)
    
    output_file = ""
    if not args.no_json:
        output_file = os.path.join(output_dir, f"{block_name}_normalized.json")
        with open(output_file, 'w') as f:
            json.dump(normalized_data, f, indent=2)

    fwd_csv = rev_csv = metadata_csv = None
    if args.write_csv:
        suffix = csv_suffix(args)
        columns = [f"pos_{pos}" for pos in range(start + 1, end + 1)]
        fwd_csv = write_matrix_csv(block_name, fwd_matrix_data, output_dir, suffix, "fwd", columns)
        rev_csv = write_matrix_csv(block_name, rev_matrix_data, output_dir, suffix, "revcomp", columns)
        metadata_csv = os.path.join(output_dir, f"{block_name}_metadata.csv")
        metadata = pd.DataFrame({
            'sample_id': list(sample_labels.keys()),
            'sample_type': list(sample_labels.values()),
        }).set_index('sample_id')
        metadata.to_csv(metadata_csv)
    
    return {
        'block': block_name,
        'samples': len(normalized_data),
        'ref_len': end - start,
        'output': output_file,
        'fwd_csv': fwd_csv,
        'rev_csv': rev_csv,
        'metadata_csv': metadata_csv
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    bed_df = pd.read_csv(args.bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
    
    results = []
    
    block_start = max(1, int(args.block_start))
    block_end = int(args.block_end)
    if block_end <= 0:
        block_end = len(bed_df)

    json_path = Path(args.json_dir)
    json_files = {extract_block_id(p.name): p for p in sorted(json_path.glob(args.file_pattern), key=natural_file_key)}

    for block_id, row in bed_df.iterrows():
        block_no = block_id + 1
        if block_no < block_start or block_no > block_end:
            continue

        block_name = f"block_{block_id + 1}"
        json_file = json_files.get(block_no)
        
        if json_file is None or not os.path.exists(json_file):
            print(f"\nSkipping {block_name}: attention JSON not found")
            continue
            
        try:
            result = process_block(block_name, str(json_file), row, args.output_dir, args)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error while processing {block_name}: {e}")

    print("\nAttention-score normalization finished.")

if __name__ == '__main__':
    main()

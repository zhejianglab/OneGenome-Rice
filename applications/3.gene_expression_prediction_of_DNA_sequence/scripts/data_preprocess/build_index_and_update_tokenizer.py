#!/usr/bin/env python3
"""
Build index for genomic data processing.

This script creates an index of genomic regions from BigWig files for use in
machine learning pipelines. It generates sliding windows across specified
chromosomes and calculates statistics for efficient data loading.
"""

import os
import argparse
import logging
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
from tqdm import tqdm
import json
import datetime
import random
from transformers import AutoTokenizer 

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def build_index(bigwig_dir, track_index, fasta_path, chromosomes, output_index_path, window_size, overlap):
    """
    Build training index file (calculating non-zero mean per chromosome).

    Args:
        bigwig_dir (str): Directory containing BigWig files
        fasta_path (str): Path to reference genome FASTA file
        chromosomes (list): List of chromosome names to process
        output_index_path (str): Path to output index CSV file
        window_size (int): Size of genomic windows
        overlap (int): Overlap between consecutive windows

    Returns:
        pandas.DataFrame: Index dataframe
    """
    output_base_dir = os.path.dirname(output_index_path)
    
    # Check if index already exists
    if os.path.exists(output_index_path):
        logging.info(f"Index file already exists, loading: {output_index_path}")
        return pd.read_csv(output_index_path)

    # Validate genome FASTA file
    try:
        genome = pyfaidx.Fasta(fasta_path)
        logging.info(f"✅ Successfully loaded genome FASTA: {fasta_path}")
        
        # Check if all requested chromosomes exist in the genome
        available_chromosomes = list(genome.keys())
        missing_chromosomes = [chrom for chrom in chromosomes if chrom not in available_chromosomes]
        if missing_chromosomes:
            logging.warning(f"⚠️  The following chromosomes are not found in the genome: {missing_chromosomes}")
        
        # Close the genome file immediately after validation
        genome.close()
    except Exception as e:
        raise ValueError(f"Failed to load genome FASTA file {fasta_path}: {str(e)}")

    
    # Find BigWig files
    if track_index:
        bigwig_files = sorted([
            os.path.join(bigwig_dir, f"RNA-SEQ_track_avg_{idx}.bigWig") 
            for idx in track_index
        ])
    else:
        bigwig_files = sorted([
        os.path.join(bigwig_dir, f) 
        for f in os.listdir(bigwig_dir)
        if f.lower().endswith('.bigwig') or f.lower().endswith('.bw')
    ])
    
    if not bigwig_files:
        raise ValueError(f"No BigWig files found in directory: {bigwig_dir}")

    index_data = []          # Collect all windows
    file_stats = []          # Sample count and stats per file
    chrom_counts = {chrom: 0 for chrom in chromosomes}

    # Process each BigWig file
    file_pbar = tqdm(bigwig_files, desc="Processing files")
    for bw_path in file_pbar:
        try:
            file_name = os.path.basename(bw_path)
            track_index = file_name.split('_')[-1].split('.')[0]
            file_pbar.set_postfix(file=file_name[:20])

            bw = pyBigWig.open(bw_path)
            
            # Pre-calculate non-zero mean for each chromosome
            chrom_means = {}
            for chromosome in tqdm(chromosomes, desc="Pre-calculate non-zero mean for each chromosome"):
                if chromosome not in bw.chroms():
                    chrom_means[chromosome] = 1.0
                    continue
                
                total_sum = 0
                total_bases = 0
                intervals = bw.intervals(chromosome)
                
                if intervals:
                    for start, end, value in intervals:
                        if value != 0:
                            span = end - start
                            total_sum += value * span
                            total_bases += span
                
                chrom_means[chromosome] = (
                    total_sum / total_bases if total_bases else 1.0
                )

            # Generate windows for each chromosome
            for chromosome in tqdm(chromosomes, desc=f"Generate windows for each chromosome"):
                if chromosome not in bw.chroms():
                    continue
                    
                chrom_length = bw.chroms()[chromosome]
                if chrom_length < window_size:
                    continue

                # Calculate window positions
                step_size = window_size - overlap
                starts = list(range(0, chrom_length - window_size + 1, step_size))
                last_start = chrom_length - window_size
                if last_start not in starts:
                    starts.append(last_start)

                track_mean = chrom_means[chromosome]
                for start in starts:
                    end = start + window_size
                    index_data.append({
                        "bw_path": bw_path,  # Keep for internal processing but not saved to CSV
                        "fasta_path": fasta_path,  # Keep for internal processing but not saved to CSV
                        "chromosome": chromosome,
                        "start": start,
                        "end": end,
                        "file_name": file_name,
                        "prefix_token": f"<output_type:RNA_SEQ|track_index:{track_index}>",
                        "track_mean": track_mean
                    })
                    chrom_counts[chromosome] += 1

            # Record file statistics
            file_stats.append({
                "file": file_name,
                "samples": sum(1 for item in index_data if item["file_name"] == file_name)
            })

            bw.close()
        except Exception as e:
            file_pbar.set_postfix_str(f"Error: {str(e)[:20]}")
            continue

    file_pbar.close()

    # # Pre-filter invalid samples
    # logging.info("🔎 Pre-scanning to filter invalid samples...")
    # valid_indices = []

    # # Create temporary dataframe for filtering
    temp_df = pd.DataFrame(index_data)

    # # Group by file paths to minimize file I/O
    # grouped = temp_df.groupby(['fasta_path', 'bw_path']).groups

    # logging.info(f"🔎 {len(grouped)} unique file combinations...")

    # total_processed = 0
    # successful_groups = 0

    # for (fasta_path, bw_path), indices in grouped.items():
    #     try:
    #         # Open files once for this group
    #         logging.debug(f"Opening files: {os.path.basename(fasta_path)} and {os.path.basename(bw_path)}")
    #         bw = pyBigWig.open(bw_path)
    #         fasta = pyfaidx.Fasta(fasta_path)
            
    #         group_valid_count = 0
    #         # Process all indices for this file combination
    #         for idx in tqdm(indices, desc=f"Processing {os.path.basename(bw_path)}", leave=False):
    #             row = temp_df.iloc[idx]
    #             try:
    #                 # Check sequence and signal data
    #                 seq = str(fasta[row["chromosome"]][row["start"]:row["end"]])
    #                 values = np.array(bw.values(row["chromosome"], row["start"], row["end"]))
                    
    #                 # Convert NaN values to 0
    #                 values = np.nan_to_num(values, nan=0.0)
                    
    #                 # If we get here, the sample is valid
    #                 valid_indices.append(idx)
    #                 group_valid_count += 1
    #                 total_processed += 1
                    
    #             except Exception as e:
    #                 # Log individual sample errors if needed for debugging
    #                 logging.debug(f"Invalid sample at {row['chromosome']}:{row['start']}-{row['end']} in {row['file_name']}: {str(e)}")
    #                 continue
            
    #         logging.info(f"  ✅ Processed {group_valid_count}/{len(indices)} valid samples from {os.path.basename(bw_path)}")
    #         successful_groups += 1
            
    #         # Clean up files for this group
    #         bw.close()
    #         fasta.close()
            
    #     except Exception as e:
    #         logging.warning(f"❌ Failed to process file combination {os.path.basename(bw_path)}: {str(e)}")
    #         # Continue with other file combinations
    #         continue

    # logging.info(f"🏷️ Successfully processed {successful_groups}/{len(grouped)} file combinations")
    # logging.info(f"🏷️ Valid samples: {len(valid_indices)}/{len(temp_df)}")
    # filtered_data = temp_df.iloc[valid_indices]

    filtered_data = temp_df
    filtered_data = filtered_data.sort_values(by=['chromosome', 'start'], ascending=[True, True])

    # Save comprehensive statistics as JSON
    if output_base_dir:
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Generate comprehensive statistics from filtered_data
        statistics = {
            "summary": {
                "total_samples": len(filtered_data),
                "total_chromosomes": len(chromosomes),
                "window_size": window_size,
                "overlap": overlap,
                "total_unique_files": len(bigwig_files),
                "fasta_path": fasta_path,  # Store paths in statistics
                "bigwig_dir": bigwig_dir
            },
            "chromosome_stats": {},
            "file_stats": {},
            "processing_info": {
                # "successful_file_groups": successful_groups,
                # "total_file_groups": len(grouped),
                "validation_time": str(datetime.datetime.now())
            }
        }
        
        # Generate chromosome statistics
        if len(filtered_data) > 0:
            chrom_counts = filtered_data['chromosome'].value_counts().to_dict()
            # Sort by chromosome name for consistent output
            statistics["chromosome_stats"] = dict(sorted(chrom_counts.items()))
            
            # Generate file statistics
            file_counts = filtered_data['file_name'].value_counts().to_dict()
            # Sort by file name for consistent output
            statistics["file_stats"] = dict(sorted(file_counts.items()))
        
        # Save as JSON
        stats_json_path = os.path.join(output_base_dir, "train_indices_statistics.json")
        with open(stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📊 Statistics saved to {stats_json_path}")

    # Write index (without fasta_path and bw_path columns)
    df = pd.DataFrame(filtered_data)
    # Remove internal processing columns
    df = df.drop(columns=['fasta_path', 'bw_path'])
    
    df.to_csv(
        output_index_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        lineterminator='\n',
        encoding='utf-8'
    )
    
    logging.info(f"✅ Index building completed, {len(df)} samples, index file saved to {output_index_path}")
    return df

def generate_validation_index(train_index_df, val_ratio, window_size, output_val_index_path):
    """
    Generate validation index by sampling window-sized fragments from training samples.
    Consecutive windows are defined as those with a gap of one window between them.
    
    Args:
        train_index_df (pd.DataFrame): Training index dataframe
        val_ratio (float): Validation ratio (0.0 to 1.0)
        window_size (int): Size of genomic windows
        output_val_index_path (str): Path to output validation index CSV file
        
    Returns:
        pd.DataFrame: Validation index dataframe
    """
    logging.info(f"🎲 Generating validation index with ratio {val_ratio}...")
    
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("Validation ratio must be between 0 and 1 (exclusive)")
    
    # Sort by chromosome and start position
    train_index_df = train_index_df.sort_values(by=['chromosome', 'start']).reset_index(drop=True)
    
    val_samples = []
    
    # Group by chromosome and file
    grouped = train_index_df.groupby(['chromosome', 'file_name'])
    
    total_candidates = 0
    total_val_samples = 0
    
    for (chromosome, file_name), group in grouped:
        # Sort by start position within each group
        group = group.sort_values('start').reset_index(drop=True)
        
        # Find consecutive windows (where a window exists between two windows)
        consecutive_triples = []
        for i in range(len(group) - 2):  # Need at least 3 windows for a triple
            first_row = group.iloc[i]
            next2_row = group.iloc[i + 2]
            
            # Check if they form a consecutive sequence with one window in between
            # (end of first = start of middle) AND (end of middle = start of last)
            if first_row['end'] == next2_row['start']:
                consecutive_triples.append(i)
        
        total_candidates += len(consecutive_triples)
        
        # Sample validation windows from consecutive triples
        num_val_needed = int(len(consecutive_triples) * val_ratio)
        if num_val_needed > 0:
            # Randomly sample triples
            sampled_indices = random.sample(consecutive_triples, min(num_val_needed, len(consecutive_triples)))
            
            # For each sampled triple, create a validation window
            for idx in sampled_indices:
                first_row = group.iloc[idx]
                next2_row = group.iloc[idx + 2]
                
                # Create validation sample by taking a window_size fragment
                # Randomly sample a start position within the range covered by all three windows
                min_start = first_row['start']
                max_start = first_row['end'] 
                
                # Ensure we don't go beyond the window boundaries
                if max_start >= min_start:
                    # Randomly sample start position within valid range
                    val_start = random.randint(min_start, max_start)
                    val_end = val_start + window_size
                    
                    # Ensure the validation window is within the combined range
                    if val_end <= next2_row['end']:
                        val_sample = first_row.copy()
                        val_sample['start'] = val_start
                        val_sample['end'] = val_end
                        val_samples.append(val_sample)
                        total_val_samples += 1
    
    if total_val_samples == 0:
        logging.warning("⚠️  No validation samples generated. Consider adjusting the validation ratio or checking data.")
        # Create a minimal validation set by sampling from training data
        num_val_needed = max(1, int(len(train_index_df) * val_ratio))
        sampled_indices = random.sample(range(len(train_index_df)), min(num_val_needed, len(train_index_df)))
        for idx in sampled_indices:
            val_samples.append(train_index_df.iloc[idx].copy())
        total_val_samples = len(val_samples)
    
    # Create validation dataframe
    val_df = pd.DataFrame(val_samples)
    if len(val_df) > 0:
        val_df = val_df.sort_values(by=['chromosome', 'start'], ascending=[True, True])
    
    # Save validation index
    val_df.to_csv(
        output_val_index_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        lineterminator='\n',
        encoding='utf-8'
    )
    
    # Save validation statistics
    output_base_dir = os.path.dirname(output_val_index_path)
    if output_base_dir:
        val_statistics = {
            "summary": {
                "total_samples": len(val_df),
                "window_size": window_size,
                "val_ratio": val_ratio,
                "total_candidates": total_candidates
            },
            "chromosome_stats": {},
            "file_stats": {},
            "processing_info": {
                "generation_time": str(datetime.datetime.now())
            }
        }
        
        # Generate chromosome statistics
        if len(val_df) > 0:
            chrom_counts = val_df['chromosome'].value_counts().to_dict()
            val_statistics["chromosome_stats"] = dict(sorted(chrom_counts.items()))
            
            # Generate file statistics
            file_counts = val_df['file_name'].value_counts().to_dict()
            val_statistics["file_stats"] = dict(sorted(file_counts.items()))
        
        # Save as JSON
        val_stats_json_path = os.path.join(output_base_dir, "val_indices_statistics.json")
        with open(val_stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_statistics, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📊 Validation statistics saved to {val_stats_json_path}")
    
    logging.info(f"✅ Validation index generation completed, {len(val_df)} samples, index file saved to {output_val_index_path}")
    return val_df



def update_tokenizer(bigwig_dir, track_index, input_dir, output_dir):
    """
    Updates the tokenizer by adding special tokens derived from bigwig file names.

    Args:
        bigwig_dir (str or Path): Directory containing .bigwig or .bw files.
        input_dir (str or Path): Path to the pretrained tokenizer to load.
        output_dir (str or Path): Path to save the updated tokenizer.
    """
    # Build special tokens from bigwig file names
    bigwig_dir = Path(bigwig_dir)
    if track_index:
        prefix_tokens = [
        f"<output_type:RNA_SEQ|track_index:{idx}>"
        for idx in track_index
    ]
    else:
        prefix_tokens = [
            f"<output_type:RNA_SEQ|track_index:{f.stem.split('_')[-1]}>"
            for f in bigwig_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ['.bigwig', '.bw']
        ]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_dir)

    # Add special tokens
    tokenizer.add_tokens([f"<chr{x}>" for x in range(1,23)]+["chrX", "chrY"], special_tokens=True) # 给染色体编号也加入特殊token
    tokenizer.add_tokens(prefix_tokens, special_tokens=True)
    print(f"✅ Added {len(prefix_tokens)} special tokens: {prefix_tokens}")
    print(f"✅ Current vocabulary size: {len(tokenizer)}")

    # Print token IDs of newly added special tokens
    for token in prefix_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  Token: {token:<15} → Token ID: {token_id:<6}")

    # Save updated tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"\n💾 Tokenizer saved to: {output_dir}")


def parse_track_index(value):
    """
    Parse a comma-separated string into a list of strings.
    If value is None, return None.
    Example: "27,33,133" -> ['27', '33', '133']
    """
    if not value:
        return None
    return value.split(',')
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build index for genomic data processing")
    
    # Data paths
    parser.add_argument("--genome_fasta", type=str, required=True,
                        help="Reference genome FASTA file path")
    parser.add_argument("--bigwig_dir", type=str, required=True,
                        help="Directory containing BigWig signal files")
    
    # Output settings
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Output base directory")
    
    # Chromosome settings
    parser.add_argument("--chromosomes", type=str, nargs='+', 
                        default=[f"chr{i}" for i in range(1, 23)], # + ["chrX", "chrY"],
                        help="Chromosomes to process")
    
    # Window parameters
    parser.add_argument("--window_size", type=int, default=32768,
                        help="Size of genomic windows (default: 32768)")
    parser.add_argument("--overlap", type=int, default=16384,
                        help="Overlap between consecutive windows (default: 16384)")
    
    # Validation parameters
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation ratio (default: 0.1)")
    
    # tokenizer dir
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--updated_tokenizer_dir", type=str, required=True)

    # track index
    parser.add_argument("--track_index", type=parse_track_index, default=None,
                        help="Comma-separated track indices (e.g., 27,33,133). If provided, only these tracks will be processed.")

    # pre-calculated track mean path
    parser.add_argument("--output_json_file_path", type=str,required=True)

    return parser.parse_args()


def csv_to_nested_json_pandas(df, json_file_path):
    """
    使用 pandas 读取 CSV，按 prefix_token 和 chromosome 分组，
    提取 track_mean 并生成嵌套 JSON。
    """

    # 2. 确保相关列是字符串或数值（防止类型问题）
    df['prefix_token'] = df['prefix_token'].astype(str)
    df['chromosome'] = df['chromosome'].astype(str)
    df['track_mean'] = pd.to_numeric(df['track_mean'], errors='coerce')  # 转为 float

    # 3. 按 prefix_token 和 chromosome 分组，取 track_mean（假设每组只有一个值）
    grouped = df.groupby(['prefix_token', 'chromosome'])['track_mean'].first().unstack(fill_value=0.0)

    # 4. 转为嵌套字典
    result = grouped.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()

    # 5. 保存为 JSON
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON 已保存至: {json_file_path}")
    return result


def main():
    
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Log chromosome information
    logging.info(f"Processing chromosomes: {args.chromosomes}")
    logging.info(f"Window size: {args.window_size}, Overlap: {args.overlap}")
    logging.info(f"Validation ratio: {args.val_ratio}")
    
    # Build data index
    logging.info("Building training sample index...")
    train_index_path = os.path.join(args.output_base_dir, "train_indices.csv") # 可自行修改
    train_df = build_index(
        bigwig_dir=args.bigwig_dir,
        track_index=args.track_index,
        fasta_path=args.genome_fasta,
        chromosomes=args.chromosomes,
        output_index_path=train_index_path,
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    # # Generate validation index 
    # logging.info("Generating validation sample index...")
    # val_index_path = os.path.join(args.output_base_dir, "val_indices.csv") # 可自行修改
    # generate_validation_index(
    #     train_index_df=train_df,
    #     val_ratio=args.val_ratio,
    #     window_size=args.window_size,
    #     output_val_index_path=val_index_path
    # )
    
    logging.info("🎉 Index building completed!")

    update_tokenizer(args.bigwig_dir, args.track_index, args.tokenizer_dir, args.updated_tokenizer_dir)
    logging.info("🎉 Tokenizer updating completed!")

    output_json_file_path = os.path.join(args.output_base_dir, "pre_calc_track_mean.json")
    csv_to_nested_json_pandas(train_df, output_json_file_path)
    logging.info("🎉 Tokenizer updating completed!")

if __name__ == "__main__":
    main()
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


def build_multimodal_index(meta_csv, assay_titles, biosample_names, bigwig_dir, fasta_path, chromosomes, output_index_path, window_size, overlap):
    """
    Build training index file (calculating non-zero mean per chromosome).

    Args:
        meta_csv (str): Path to metadata CSV file
        bigwig_dir (str): Directory containing BigWig files
        assay_titles (list): List of assay titles
        biosample_names (list): List of biosample names
        fasta_path (str): Path to reference genome FASTA file
        chromosomes (list): List of chromosome names to process
        output_index_path (str): Path to output index CSV file
        window_size (int): Size of genomic windows
        overlap (int): Overlap between consecutive windows

    Returns:
        pandas.DataFrame: Index dataframe
    """
    meta_df = pd.read_csv(meta_csv)
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

    # Build multi-modal track dict
    track_dict = {}
    for biosample_name in biosample_names:
        # Filter metadata once per biosample for efficiency
        biosample_mask = meta_df['biosample_name'] == biosample_name
        biosample_data = meta_df[biosample_mask]
        
        if biosample_data.empty:
            logging.warning(f"No metadata found for biosample: {biosample_name}")
            continue
            
        biosample_type = biosample_data['biosample_type'].values[0]
        biosample_key = f"{biosample_type}-{biosample_name.replace(' ', '_')}"
        
        # Initialize nested dictionary for this biosample
        track_dict[biosample_key] = {}
        
        for assay in assay_titles:
            if assay == "total RNA-seq":
                # Handle positive strand
                plus_mask = (biosample_mask & 
                            (meta_df['Assay title'] == assay) & 
                            (meta_df['strand'] == '+'))
                plus_data = meta_df[plus_mask]
                
                if not plus_data.empty:
                    idx = plus_data['track_index'].values[0]
                    bigwig_file = os.path.join(bigwig_dir, f"RNA_SEQ_track_avg_{idx}.bigWig")
                    if os.path.exists(bigwig_file):
                        # bigwig_file只保留文件名
                        track_dict[biosample_key]['total_RNA-seq_+'] = os.path.basename(bigwig_file)
                    else:
                        logging.warning(f"BigWig file not found: {bigwig_file} for biosample {biosample_key}, assay {assay} and strand +")
                else:
                    logging.warning(f"No metadata entry found for biosample {biosample_key}, assay {assay} and strand +")
                
                # Handle negative strand
                minus_mask = (biosample_mask & 
                            (meta_df['Assay title'] == assay) & 
                            (meta_df['strand'] == '-'))
                minus_data = meta_df[minus_mask]
                
                if not minus_data.empty:
                    idx = minus_data['track_index'].values[0]
                    bigwig_file = os.path.join(bigwig_dir, f"RNA_SEQ_track_avg_{idx}.bigWig")
                    if os.path.exists(bigwig_file):
                        track_dict[biosample_key]['total_RNA-seq_-'] = os.path.basename(bigwig_file)
                    else:
                        logging.warning(f"BigWig file not found: {bigwig_file} for biosample {biosample_key}, assay {assay} and strand -")
                else:
                    logging.warning(f"No metadata entry found for biosample {biosample_key}, assay {assay} and strand -")

            elif assay == "ATAC-seq":
                # Handle ATAC-seq data
                atac_mask = (biosample_mask & 
                            (meta_df['Assay title'] == assay) & 
                            (meta_df['strand'] == '.'))
                atac_data = meta_df[atac_mask]
                
                if not atac_data.empty:
                    idx = atac_data['track_index'].values[0]
                    bigwig_file = os.path.join(bigwig_dir, f"ATAC_track_avg_{idx}.bigWig")
                    if os.path.exists(bigwig_file):
                        track_dict[biosample_key]['ATAC-seq_.'] = os.path.basename(bigwig_file)
                    else:
                        logging.warning(f"BigWig file not found: {bigwig_file} for biosample {biosample_key}, assay {assay} and strand .")
                else:
                    logging.warning(f"No metadata entry found for biosample {biosample_key}, assay {assay} and strand .")

    # Create index data with genomic windows
    index_data = []
    
    # Process each chromosome
    for chromosome in tqdm(chromosomes, desc="Processing chromosomes"):
        try:
            # Get chromosome length from any available bigwig file (assuming all have same chrom sizes)
            sample_bw_file = next(iter([v for d in track_dict.values() for v in d.values()]))
            sample_bw_path = os.path.join(bigwig_dir, sample_bw_file)
            bw = pyBigWig.open(sample_bw_path)
            
            if chromosome not in bw.chroms():
                logging.warning(f"Chromosome {chromosome} not found in bigwig files")
                bw.close()
                continue
                
            chrom_length = bw.chroms()[chromosome]
            bw.close()
            
            if chrom_length < window_size:
                continue

            # Calculate window positions
            step_size = window_size - overlap
            starts = list(range(0, chrom_length - window_size + 1, step_size))
            last_start = chrom_length - window_size
            if last_start not in starts:
                starts.append(last_start)

            # For each window, add entries for all biosamples
            for start in starts:
                end = start + window_size
                # Add an entry for each biosample
                for biosample_key in track_dict:
                    entry = {
                        "chromosome": chromosome,
                        "start": start,
                        "end": end,
                        "biosample": biosample_key,
                    }
                    # Add track paths for this biosample
                    entry.update(track_dict[biosample_key])
                    index_data.append(entry)
                    
        except Exception as e:
            logging.warning(f"Error processing chromosome {chromosome}: {str(e)}")
            continue

    # Create initial dataframe
    df = pd.DataFrame(index_data)
    
    # Calculate track means for each track in each biosample
    if not df.empty:
        logging.info("Calculating track means for each biosample...")
        
        # Get unique combinations of biosample and track types
        track_columns = [col for col in df.columns if col not in ['chromosome', 'start', 'end', 'biosample']]
        
        # For each track column, calculate means
        for track_col in track_columns:
            mean_column_name = f"{track_col}_mean"
            df[mean_column_name] = 1.0  # Default value
            
            # Group by track file to minimize file operations
            track_files = df[track_col].unique()
            
            for track_file in tqdm(track_files, desc=f"Calculating means for {track_col}"):
                if pd.isna(track_file):
                    continue
                    
                track_file_path = os.path.join(bigwig_dir, track_file)
                if not os.path.exists(track_file_path):
                    continue
                    
                try:
                    bw = pyBigWig.open(track_file_path)
                    
                    # Update all rows with this track file
                    track_mask = df[track_col] == track_file
                    chromosomes_in_df = df[track_mask]['chromosome'].unique()
                    
                    for chromosome in chromosomes_in_df:
                        if chromosome not in bw.chroms():
                            continue
                            
                        # Calculate mean for this chromosome
                        total_sum = 0
                        total_bases = 0
                        intervals = bw.intervals(chromosome)
                        
                        if intervals:
                            for interval_start, interval_end, value in intervals:
                                if value != 0:
                                    span = interval_end - interval_start
                                    total_sum += value * span
                                    total_bases += span
                        
                        track_mean = total_sum / total_bases if total_bases else 1.0
                        
                        # Update all rows matching this track file and chromosome
                        chrom_track_mask = track_mask & (df['chromosome'] == chromosome)
                        df.loc[chrom_track_mask, mean_column_name] = track_mean
                    
                    bw.close()
                except Exception as e:
                    logging.warning(f"Error calculating mean for {track_file}: {str(e)}")
                    continue

    # Save comprehensive statistics as JSON
    if output_base_dir:
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Generate comprehensive statistics
        statistics = {
            "summary": {
                "total_samples": len(df),
                "total_biosamples": len(track_dict),
                "total_chromosomes": len(chromosomes),
                "window_size": window_size,
                "overlap": overlap,
                "fasta_path": fasta_path,
                "bigwig_dir": bigwig_dir
            },
            "biosample_stats": {},
            "chromosome_stats": {},
            "track_types": [],  # Add track column types
            "processing_info": {
                "validation_time": str(datetime.datetime.now())
            }
        }
        
        # Generate biosample statistics
        if len(df) > 0:
            biosample_counts = df['biosample'].value_counts().to_dict()
            statistics["biosample_stats"] = dict(sorted(biosample_counts.items()))
            
            # Generate chromosome statistics
            chrom_counts = df['chromosome'].value_counts().to_dict()
            statistics["chromosome_stats"] = dict(sorted(chrom_counts.items()))
            
            # Add track column types (exclude common columns)
            common_columns = ['chromosome', 'start', 'end', 'biosample']
            track_columns = [col for col in df.columns if col not in common_columns and not col.endswith('_mean')]
            statistics["track_types"] = sorted(track_columns)
        
        # Save as JSON
        stats_json_path = os.path.join(output_base_dir, "multimodal_train_indices_statistics.json")
        with open(stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📊 Statistics saved to {stats_json_path}")

    # Save index to CSV
    df.to_csv(
        output_index_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        lineterminator='\n',
        encoding='utf-8'
    )
    
    logging.info(f"✅ Multimodal index building completed, {len(df)} samples, index file saved to {output_index_path}")
    return df

# def update_tokenizer_with_biosamples(input_dir, output_dir, biosample_tokens=None):
#     """
#     Updates the tokenizer by adding special tokens for chromosomes and biosamples.

#     Args:
#         input_dir (str or Path): Path to the pretrained tokenizer to load.
#         output_dir (str or Path): Path to save the updated tokenizer.
#         biosample_tokens (list): List of biosample tokens to add.
#     """
#     # Load tokenizer
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(input_dir)
#         print(f"✅ Loaded tokenizer from: {input_dir}")
#     except Exception as e:
#         print(f"❌ Failed to load tokenizer from {input_dir}: {e}")
#         raise

#     # Add chromosome tokens
#     chromosome_tokens = [f"<chr{x}>" for x in range(1, 23)] + ["<chrX>", "<chrY>"]
    

#     # Add all special tokens    
#     tokenizer.add_tokens(chromosome_tokens, special_tokens=True)
#     print(f"✅ Added {len(chromosome_tokens)} chromosome tokens")
#     if biosample_tokens:
#         tokenizer.add_tokens(biosample_tokens, special_tokens=True)
#         print(f"✅ Added {len(biosample_tokens)} biosample tokens")
#     print(f"✅ Current vocabulary size: {len(tokenizer)}")

#     # Print token IDs of newly added special tokens
#     for token in chromosome_tokens + (biosample_tokens if biosample_tokens else []):
#         token_id = tokenizer.convert_tokens_to_ids(token)
#         print(f"  Token: {token:<30} → Token ID: {token_id:<6}")

#     # Ensure output directory exists
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
#     print(f"📁 Ensuring output directory exists: {output_dir}")
    
#     # Save updated tokenizer
#     try:
#         tokenizer.save_pretrained(output_dir)
#         print(f"✅ Tokenizer successfully saved to: {output_dir}")
        
#         # Verify the tokenizer was saved
#         if os.path.exists(output_dir):
#             files = os.listdir(output_dir)
#             print(f"📁 Files in updated tokenizer directory: {files}")
#         else:
#             print(f"❌ Output directory was not created: {output_dir}")
#     except Exception as e:
#         print(f"❌ Failed to save tokenizer to {output_dir}: {e}")
#         raise
    
#     return tokenizer  # Return the tokenizer for potential further use


def update_tokenizer_with_biosamples(input_dir, output_dir, biosample_tokens=None):
    """
    Updates the tokenizer by adding special tokens for chromosomes and biosamples.

    Args:
        input_dir (str or Path): Path to the pretrained tokenizer to load.
        output_dir (str or Path): Path to save the updated tokenizer.
        biosample_tokens (list): List of biosample tokens to add.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_dir)

    # Add chromosome tokens
    chromosome_tokens = [f"<chr{x}>" for x in range(1, 23)] + ["<chrX>", "<chrY>"]
    
    # Add all special tokens    
    tokenizer.add_tokens(chromosome_tokens, special_tokens=True)
    print(f"✅ Added {len(chromosome_tokens)} chromosome tokens")
    if biosample_tokens:
        tokenizer.add_tokens(biosample_tokens, special_tokens=True)
        print(f"✅ Added {len(biosample_tokens)} biosample tokens")
    print(f"✅ Current vocabulary size: {len(tokenizer)}")


    # Print token IDs of newly added special tokens
    for token in chromosome_tokens + (biosample_tokens if biosample_tokens else []):
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  Token: {token:<30} → Token ID: {token_id:<6}")

    # Save updated tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"\n💾 Tokenizer saved to: {output_dir}")



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build index for genomic data processing")
    
    # Data paths
    parser.add_argument("--genome_fasta", type=str, required=True,
                        help="Reference genome FASTA file path")
    parser.add_argument("--bigwig_dir", type=str, required=True,
                        help="Directory containing BigWig signal files")
    parser.add_argument("--meta_csv", type=str, required=True,
                        help="Path to metadata CSV file")
    
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

    # Biosample and assay parameters
    parser.add_argument("--assay_titles", type=str, nargs='+', required=True,
                        help="List of assay titles to process")
    parser.add_argument("--biosample_names", type=str, nargs='+', required=True,
                        help="List of biosample names to process")

    # pre-calculated track mean path
    parser.add_argument("--output_json_file_path", type=str, required=True)

    return parser.parse_args()

def csv_to_nested_json_pandas(df, json_file_path):
    """
    Using pandas to read CSV, group by biosample and chromosome,
    extract _mean values for each track_col and generate nested JSON.
    """
    
    # Ensure relevant columns are strings or numbers (prevent type issues)
    df['biosample'] = df['biosample'].astype(str)
    df['chromosome'] = df['chromosome'].astype(str)
    
    # Find all track_mean columns
    track_mean_columns = [col for col in df.columns if col.endswith('_mean')]
    
    # Create a dictionary containing all biosamples' results
    result = {}
    
    # Process each biosample separately
    for biosample in df['biosample'].unique():
        # Filter data for current biosample and explicitly make a copy
        biosample_df = df[df['biosample'] == biosample].copy()
        
        # Create a sub-dictionary for the current biosample
        biosample_result = {}
        
        # Process each track_mean column separately
        for track_col in track_mean_columns:
            # Convert track_mean column to numeric type
            biosample_df[track_col] = pd.to_numeric(biosample_df[track_col], errors='coerce')
            
            # Group by chromosome and take the first value of the corresponding track_mean
            grouped = biosample_df.groupby('chromosome')[track_col].first()
            
            # Convert to dictionary and remove NaN values
            track_result = grouped.dropna().to_dict()
            
            # Only add when track_result is not empty
            if track_result:
                biosample_result[track_col] = track_result
        
        # Only add when biosample_result is not empty
        if biosample_result:
            result[biosample] = biosample_result

    # Ensure output directory exists
    output_dir = os.path.dirname(json_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON has been saved to: {json_file_path}")
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
    logging.info(f"Assay titles: {args.assay_titles}")
    logging.info(f"Biosample names: {args.biosample_names}")
    
    # Build multimodal data index
    logging.info("Building multimodal training sample index...")
    train_index_path = os.path.join(args.output_base_dir, "train_indices.csv")
    train_df = build_multimodal_index(
        meta_csv=args.meta_csv,
        assay_titles=args.assay_titles,
        biosample_names=args.biosample_names,
        bigwig_dir=args.bigwig_dir,
        fasta_path=args.genome_fasta,
        chromosomes=args.chromosomes,
        output_index_path=train_index_path,
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    logging.info("🎉 Index building completed!")

    # Extract biosample names from the index dataframe for tokenizer
    biosample_tokens = []
    if 'biosample' in train_df.columns:
        # Extract unique biosample names from the biosample column
        unique_biosamples = train_df['biosample'].unique().tolist()
        # Create tokens in the format <biosample:type-name>
        biosample_tokens = [f"<biosample:{name}>" for name in unique_biosamples]
        logging.info(f"🧬 Generated {len(biosample_tokens)} biosample tokens")
    else:
        logging.warning("No 'biosample' column found in dataframe")
    
    # Update tokenizer with biosample tokens
    logging.info(f"🔄 Updating tokenizer from {args.tokenizer_dir} to {args.updated_tokenizer_dir}")
    update_tokenizer_with_biosamples(args.tokenizer_dir, args.updated_tokenizer_dir, biosample_tokens)
    logging.info("🎉 Tokenizer updating completed!")

    output_json_file_path = os.path.join(args.output_base_dir, "pre_calc_track_mean.json")
    csv_to_nested_json_pandas(train_df, output_json_file_path)
    logging.info("🎉 Track mean JSON generation completed!")

if __name__ == "__main__":
    main()
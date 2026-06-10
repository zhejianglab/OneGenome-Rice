#!/usr/bin/env python3
"""
Minimal utilities:
- Split FASTA chromosomes into sliding windows and save `sequence_split_train.csv`.
- Filter provided metadata CSV by assay titles and biosample names and save `bigWig_labels_meta.csv`.

This file intentionally keeps functionality small and focused.
"""

import os
import argparse
import logging
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm
import json
import datetime


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def split_fasta_windows(fasta_path, chromosomes, window_size, overlap, output_dir):
    """Split chromosomes into windows and save CSV with columns: chromosome,start,end.

    Coordinates are 0-based, half-open: start inclusive, end exclusive.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'sequence_split_train.csv') # for training set

    logging.info(f"Loading FASTA: {fasta_path}")
    genome = Fasta(fasta_path)

    rows = []
    for chrom in tqdm(chromosomes, desc='Splitting chromosomes'):
        if chrom not in genome:
            logging.warning(f"Chromosome {chrom} not found in FASTA, skipping")
            continue

        chrom_len = len(genome[chrom])
        if chrom_len < window_size:
            logging.info(f"Chromosome {chrom} length {chrom_len} < window_size {window_size}, skipping")
            continue

        step = window_size - overlap
        starts = list(range(0, chrom_len - window_size + 1, step))
        last_start = chrom_len - window_size
        if last_start not in starts:
            starts.append(last_start)

        for s in starts:
            e = s + window_size
            rows.append((chrom, int(s), int(e)))

    df = pd.DataFrame(rows, columns=['chromosome', 'start', 'end'])
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved sequence splits to {out_csv} ({len(df)} rows)")
    genome.close()
    return out_csv


def extract_meta_rows(meta_csv, assay_titles, biosample_names, output_dir):
    """Filter `meta_csv` for rows where 'Assay title' in assay_titles AND 'biosample_name' in biosample_names.

    The resulting CSV is saved as `bigWig_labels_meta.csv` in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'bigWig_labels_meta.csv')

    logging.info(f"Loading metadata CSV: {meta_csv}")
    df = pd.read_csv(meta_csv)
    #df = df[(df['organism'] == 'human')&(df['output_type'].isin(['ATAC','RNA_SEQ']))]
    df = df[df['output_type'].isin(['ATAC','RNA_SEQ'])]


    # Defensive: ensure columns exist
    if 'Assay title' not in df.columns or 'biosample_name' not in df.columns:
        raise ValueError("`meta_csv` must contain columns 'Assay title' and 'biosample_name'")

    mask = df['Assay title'].isin(assay_titles) & df['biosample_name'].isin(biosample_names)
    filtered = df[mask].copy()

    # Remove rows where output_type is RNA_SEQ but strand is '.' (invalid for RNA-seq)
    if 'output_type' in filtered.columns and 'strand' in filtered.columns:
        bad_mask = (filtered['output_type'] == 'RNA_SEQ') & (filtered['strand'] == '.')
        if bad_mask.any():
            logging.info(f"Removing {bad_mask.sum()} rows where output_type=='RNA_SEQ' and strand=='.'")
            filtered = filtered[~bad_mask].copy()

    # Add `target_file_name` and `num_file_accession` columns
    # Ensure required columns exist
    if 'output_type' not in filtered.columns or 'track_index' not in filtered.columns:
        raise ValueError("Filtered metadata must contain 'output_type' and 'track_index' columns")

    # def make_target(row):
    #     out_type = str(row['output_type'])
    #     idx = row['track_index']
    #     if pd.isna(idx):
    #         idx_str = ''
    #     else:
    #         try:
    #             if float(idx).is_integer():
    #                 idx_str = str(int(float(idx)))
    #             else:
    #                 idx_str = str(idx)
    #         except Exception:
    #             idx_str = str(idx)
    #     return f"{out_type}_track_avg_{idx_str}.bigWig"

    filtered['target_file_name'] = "re-normalized_" + filtered['target_file_name']

    def count_accessions(val):
        if pd.isna(val):
            return 0
        s = str(val).strip()
        if s == '':
            return 0
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return len(parts)

    if 'File accession' in filtered.columns:
        filtered['num_file_accession'] = filtered['File accession'].apply(count_accessions)
    else:
        logging.warning("'File accession' column not found in meta; setting num_file_accession=0")
        filtered['num_file_accession'] = 0

    # Move the two new columns to be the first two columns in the DataFrame
    cols = filtered.columns.tolist()
    # Ensure columns order: target_file_name, num_file_accession, then the rest preserving original order
    remaining = [c for c in cols if c not in ('target_file_name', 'num_file_accession')]
    new_order = ['target_file_name', 'num_file_accession'] + remaining
    filtered = filtered[new_order]

    # 对 filtered 标签进行重排序，先按照 'Assay title' 和 'strand' 的组合排序，再按照 'biosample_name' 排序
    filtered = filtered.sort_values(by=['Assay title', 'strand', 'biosample_name']).reset_index(drop=True)
    filtered.to_csv(out_csv, index=False)
    logging.info(f"Saved filtered metadata to {out_csv} ({len(filtered)} rows)")
    
    return out_csv


def build_and_save_index_stat(args, sequence_csv, meta_csv):
    """Save input parameters and simple output statistics to index_stat.json in output_base_dir."""
    stats = {
        "inputs": {
            "genome_fasta": args.genome_fasta,
            "chromosomes": args.chromosomes,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "meta_csv": args.meta_csv,
            "assay_titles": args.assay_titles,
            "biosample_names": args.biosample_names,
            "processed_bw_dir": args.processed_bw_dir,
        },
        "outputs": {
            "sequence_split_csv": os.path.basename(sequence_csv) if sequence_csv else None,
            "bigWig_labels_meta_csv": os.path.basename(meta_csv) if meta_csv else None,
        },
        "counts": {},
        "created_at": datetime.datetime.now().isoformat()
    }

    # counts: number of windows (samples) from sequence CSV (total + per-chromosome)
    try:
        df_seq = pd.read_csv(sequence_csv)
        total_samples = int(len(df_seq))
        stats["counts"]["num_samples"] = total_samples

        # per-chromosome counts
        try:
            vc = df_seq['chromosome'].value_counts().to_dict()
            # ensure plain python ints
            by_chrom = {str(k): int(v) for k, v in vc.items()}
            stats["counts"]["num_samples_by_chromosome"] = by_chrom
        except Exception:
            stats["counts"]["num_samples_by_chromosome"] = None

    except Exception:
        stats["counts"]["num_samples"] = None
        stats["counts"]["num_samples_by_chromosome"] = None

    # counts: modalities (unique combinations of Assay title and strand) and unique biosamples from meta CSV
    try:
        df_meta = pd.read_csv(meta_csv)
        # modalities: unique pairs of ('Assay title', 'strand')
        if ('Assay title' in df_meta.columns) and ('strand' in df_meta.columns):
            modalities_df = df_meta[['Assay title', 'strand']].drop_duplicates().fillna('')
            # build heads as "assay_strand" with spaces replaced by '_'
            def _norm(x):
                return str(x).replace(' ', '_')
            heads = [f"{_norm(a)}_{_norm(s.replace('.',''))}" for a, s in modalities_df.values.tolist()]
            stats["counts"]["num_modalities"] = int(len(heads))
            stats["counts"]["heads"] = heads
        else:
            stats["counts"]["num_modalities"] = None
            stats["counts"]["heads"] = []

        # biosample count and biosample order (unique preserving appearance)
        if 'biosample_name' in df_meta.columns:
            stats["counts"]["num_biosamples"] = int(df_meta['biosample_name'].nunique())
            # preserve order of first occurrence
            biosample_order = df_meta['biosample_name'].drop_duplicates().tolist()
            stats["counts"]["biosample_order"] = biosample_order
        else:
            stats["counts"]["num_biosamples"] = None
            stats["counts"]["biosample_order"] = []

        # include target_file_name and nonzero_mean lists if present 
        if 'target_file_name' in df_meta.columns:
            # ensure plain python strings
            stats["counts"]["target_file_name"] = [str(x) if not pd.isna(x) else "" for x in df_meta['target_file_name'].tolist()]
        else:
            stats["counts"]["target_file_name"] = []

        if 'nonzero_mean' in df_meta.columns:
            # convert to floats, replace NaN with None or 0.0 (choose 0.0 here)
            stats["counts"]["nonzero_mean"] = [float(x) if not pd.isna(x) else 0.0 for x in df_meta['nonzero_mean'].tolist()]
        else:
            stats["counts"]["nonzero_mean"] = []
            
    except Exception:
        stats["counts"]["num_modalities"] = None
        stats["counts"]["num_biosamples"] = None
        stats["counts"]["heads"] = []
        stats["counts"]["biosample_order"] = []
        stats["counts"]["target_file_name"] = []
        stats["counts"]["nonzero_mean"] = []

    out_path = os.path.join(args.output_base_dir, 'index_stat.json')
    os.makedirs(args.output_base_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved index statistics to {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description='Split FASTA and extract metadata (minimal)')

    # FASTA splitting args
    p.add_argument('--genome_fasta', required=True, help='Path to reference FASTA')
    p.add_argument('--chromosomes', required=True, nargs='+', help='Chromosomes to process (e.g. chr1 chr2)')
    p.add_argument('--window_size', type=int, default=32768, help='Window size')
    p.add_argument('--overlap', type=int, default=16384, help='Window overlap')

    # Metadata extraction args
    p.add_argument('--meta_csv', required=True, help='Path to metadata CSV to filter')
    p.add_argument('--assay_titles', required=True, nargs='+', help='Assay titles to keep')
    p.add_argument('--biosample_names', required=True, nargs='+', help='Biosample names to keep')
    p.add_argument('--processed_bw_dir', required=True, help='Directory where processed bigWig files are stored')
    
    # Output
    p.add_argument('--output_base_dir', required=True, help='Directory to save outputs')

    return p.parse_args()


def main():
    args = parse_args()

    logging.info('Starting minimal processing')

    split_csv = split_fasta_windows(
        fasta_path=args.genome_fasta,
        chromosomes=args.chromosomes,
        window_size=args.window_size,
        overlap=args.overlap,
        output_dir=args.output_base_dir,
    )

    meta_csv = extract_meta_rows(
        meta_csv=args.meta_csv,
        assay_titles=args.assay_titles,
        biosample_names=args.biosample_names,
        output_dir=args.output_base_dir,
    )

    stat_path = build_and_save_index_stat(args, split_csv, meta_csv)
    
    logging.info('Done. Outputs:')
    logging.info(f' - {split_csv}')
    logging.info(f' - {meta_csv}')
    logging.info(f' - {stat_path}')


if __name__ == '__main__':
    main()

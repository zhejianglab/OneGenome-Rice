# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""Genomic window index and track-mean computation for paired ATAC/RNA training."""

import os

import pandas as pd
import pyBigWig
import pyfaidx
import torch.distributed as dist

from model.distributed import dist_print, is_main_process
from model.scaling import LabelScaler


def build_index(
    fasta_path,
    output_index_path,
    rna_files,
    atac_files,
    cell_types,
    chromosome,
    chromosome_per_cell_type=None,
    fasta_path_per_cell_type=None,
    window_size=32000,
    overlap=16000,
    cap_expression_quantile=None,
):
    if os.path.exists(output_index_path):
        dist_print(f"Index already exists, loading: {output_index_path}")
        return pd.read_csv(output_index_path)

    def _chr_for_cell(cell: str) -> str:
        if chromosome_per_cell_type and cell in chromosome_per_cell_type:
            return chromosome_per_cell_type[cell]
        return chromosome

    def _fasta_for_cell(cell: str) -> str:
        if fasta_path_per_cell_type and cell in fasta_path_per_cell_type:
            return fasta_path_per_cell_type[cell]
        return fasta_path

    scaler_cache: dict = {}
    for (cell, strand), path in rna_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"RNA file not found: {path}")
        bw = pyBigWig.open(path)
        current_chr = _chr_for_cell(cell)
        intervals = []
        if current_chr in bw.chroms():
            iv = bw.intervals(current_chr)
            if iv:
                intervals = iv
        bw.close()
        scaler = LabelScaler.fit(intervals, cap_expression_quantile=cap_expression_quantile)
        scaler_cache[(cell, strand)] = scaler
        cap_msg = (
            f", cap_threshold={scaler.cap_threshold:.4g}"
            if scaler.cap_threshold is not None
            else ""
        )
        dist_print(f"[{cell} {strand}] track_mean={scaler.track_mean:.4f}{cap_msg}")

    # FASTA is only needed later by the Dataset; we keep this open/close here as a
    # lightweight validation that the global FASTA is readable when used.
    genome = pyfaidx.Fasta(fasta_path)
    paired_index_data = []

    for cell_type in cell_types:
        atac_path = atac_files[cell_type]
        if not os.path.exists(atac_path):
            raise FileNotFoundError(f"ATAC file not found: {atac_path}")

        rna_plus_path = rna_files[(cell_type, "+")]
        rna_minus_path = rna_files[(cell_type, "-")]

        current_chr = _chr_for_cell(cell_type)
        bw_rna = pyBigWig.open(rna_plus_path)
        chrom_length = bw_rna.chroms().get(current_chr, 0)
        bw_rna.close()
        if chrom_length < window_size:
            continue

        step_size = window_size - overlap
        starts = list(range(0, chrom_length - window_size + 1, step_size))
        last_start = chrom_length - window_size
        if last_start not in starts:
            starts.append(last_start)

        for start in starts:
            end = start + window_size
            paired_index_data.append(
                {
                    "cell_type": cell_type,
                    "chromosome": current_chr,
                    "start": start,
                    "end": end,
                    "fasta_path": _fasta_for_cell(cell_type),
                    "atac_path": atac_path,
                    "rna_path_plus": rna_plus_path,
                    "rna_path_minus": rna_minus_path,
                    "track_mean_plus": scaler_cache[(cell_type, "+")].track_mean,
                    "track_mean_minus": scaler_cache[(cell_type, "-")].track_mean,
                    "cap_threshold_plus": scaler_cache[(cell_type, "+")].cap_threshold
                    if scaler_cache[(cell_type, "+")].cap_threshold is not None
                    else float("nan"),
                    "cap_threshold_minus": scaler_cache[(cell_type, "-")].cap_threshold
                    if scaler_cache[(cell_type, "-")].cap_threshold is not None
                    else float("nan"),
                    "batch_name_plus": rna_plus_path.split("/")[-1].split("_")[0],
                    "batch_name_minus": rna_minus_path.split("/")[-1].split("_")[0],
                }
            )

    genome.close()
    df = pd.DataFrame(paired_index_data)

    if is_main_process():
        df.to_csv(output_index_path, index=False)
        if chromosome_per_cell_type:
            dist_print(
                f"Index built: {len(df)} samples "
                f"(chromosome_per_cell_type={chromosome_per_cell_type}, "
                f"cell_types={cell_types})"
            )
        else:
            dist_print(
                f"Index built: {len(df)} samples (chromosome={chromosome}, "
                f"cell_types={cell_types})"
            )

    if dist.is_initialized():
        dist.barrier()
    return df

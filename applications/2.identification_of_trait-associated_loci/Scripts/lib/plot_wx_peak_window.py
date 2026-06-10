#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a 21-bp summed-attention window centered at a selected Wx-region coordinate."
    )
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--bed-file", required=True)
    parser.add_argument("--sample-table", required=True)
    parser.add_argument("--peak-center", type=int, default=1767006)
    parser.add_argument("--window-bp", type=int, default=10, help="Half window size; total width is 2N+1 bp.")
    parser.add_argument("--out-png", required=True)
    return parser.parse_args()


def read_bed_intervals(bed_file: Path) -> List[Tuple[int, int]]:
    bed = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
    return [(int(row["start"]) + 1, int(row["end"])) for _, row in bed.iterrows()]


def compute_effective_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    effective: List[Tuple[int, int]] = []
    for i, (start, end) in enumerate(intervals):
        left = start
        right = end
        if i > 0:
            prev_start, prev_end = intervals[i - 1]
            ov_start = max(prev_start, start)
            ov_end = min(prev_end, end)
            if ov_start <= ov_end:
                left = max(left, (ov_start + ov_end) // 2 + 1)
        if i < len(intervals) - 1:
            next_start, next_end = intervals[i + 1]
            ov_start = max(start, next_start)
            ov_end = min(end, next_end)
            if ov_start <= ov_end:
                right = min(right, (ov_start + ov_end) // 2)
        if left <= right:
            effective.append((left, right))
        else:
            effective.append((start, end))
    return effective


def parse_pos(col: str) -> Optional[int]:
    match = re.search(r"(\d+)$", col)
    return int(match.group(1)) if match else None


def load_selected_matrix(csv_path: Path, eff_start: int, eff_end: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    pos_cols = header[1:]
    selected = []
    positions = []
    for col in pos_cols:
        pos = parse_pos(col)
        if pos is not None and eff_start <= pos <= eff_end:
            selected.append(col)
            positions.append(pos)
    df = pd.read_csv(csv_path, usecols=["sample_id"] + selected)
    return np.array(positions, dtype=np.int64), df["sample_id"].astype(str).tolist(), df[selected].to_numpy(dtype=np.float64)


def build_sum_signal(matrix_dir: Path, bed_file: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    intervals = read_bed_intervals(bed_file)
    effective = compute_effective_intervals(intervals)
    sample_ids: List[str] = []
    parts: List[np.ndarray] = []
    pos_parts: List[np.ndarray] = []
    for idx, (eff_start, eff_end) in enumerate(effective, start=1):
        block_dir = matrix_dir / f"block_{idx}"
        fwd_csv = block_dir / "hap1_attention_collapsed.csv"
        rev_csv = block_dir / "hap1_attention_collapsed_revcomp.csv"
        if not fwd_csv.exists() or not rev_csv.exists():
            continue
        pos_fwd, sids_fwd, vals_fwd = load_selected_matrix(fwd_csv, eff_start, eff_end)
        pos_rev, sids_rev, vals_rev = load_selected_matrix(rev_csv, eff_start, eff_end)
        if pos_fwd.size == 0 or pos_rev.size == 0:
            continue
        if not np.array_equal(pos_fwd, pos_rev) or sids_fwd != sids_rev:
            raise ValueError(f"Mismatch between forward and reverse matrices in {block_dir}")
        if not sample_ids:
            sample_ids = sids_fwd
        elif sample_ids != sids_fwd:
            raise ValueError(f"Sample order mismatch across blocks in {block_dir}")
        pos_parts.append(pos_fwd)
        parts.append(vals_fwd + vals_rev)
    if not parts:
        raise RuntimeError(f"No valid blocks found under {matrix_dir}")
    return np.concatenate(pos_parts), sample_ids, np.concatenate(parts, axis=1)


def main():
    args = parse_args()
    positions, sample_ids, sum_values = build_sum_signal(Path(args.matrix_dir), Path(args.bed_file))
    sample_df = pd.read_csv(args.sample_table)
    selected_ids = sample_df["sample_id"].astype(str).tolist()

    lo = args.peak_center - args.window_bp
    hi = args.peak_center + args.window_bp
    mask = (positions >= lo) & (positions <= hi)
    if mask.sum() != 2 * args.window_bp + 1:
        raise RuntimeError(f"Expected {2 * args.window_bp + 1} bp around peak, found {mask.sum()}")

    style_map = {
        0: [("#75bd42", "-", "o"), ("#75bd42", "--", "s")],
        1: [("#4874cb", "-", "o"), ("#4874cb", "--", "s")],
    }
    used_style = {0: 0, 1: 0}

    fig, ax = plt.subplots(figsize=(10.0, 2.5))
    for _, row in sample_df.iterrows():
        sample_id = str(row["sample_id"])
        trait = int(row["trait"])
        idx = sample_ids.index(sample_id)
        color, linestyle, marker = style_map[trait][used_style[trait]]
        used_style[trait] += 1
        label = f"{'Glutinous' if trait == 0 else 'Non-glutinous'} | {sample_id}"
        ax.plot(
            positions[mask],
            sum_values[idx, mask],
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            alpha=0.98,
            marker=marker,
            markersize=4.2,
            label=label,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Genomic position (bp)", ha="right", x=1.0)
    ax.set_ylabel("Summed attention")
    tick_positions = [int(p) for p in positions[mask] if int(p) % 5 == 0]
    ax.set_xticks(tick_positions)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.get_xaxis().get_offset_text().set_visible(False)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()

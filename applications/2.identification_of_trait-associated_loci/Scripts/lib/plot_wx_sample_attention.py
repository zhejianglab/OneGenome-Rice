#!/usr/bin/env python3
import argparse
import csv
import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WX_LOCUS = "LOC_Os06g04200"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot summed per-sample attention traces for 2 glutinous and 2 non-glutinous samples across the Wx region."
    )
    parser.add_argument("--matrix-dir", required=True, help="Matrix directory for the Wx region, e.g. 15.json_2_matrix/display_demo_1.wx")
    parser.add_argument("--bed-file", required=True, help="Region BED file for 1.wx")
    parser.add_argument("--pheno-file", required=True)
    parser.add_argument("--gff", required=True)
    parser.add_argument("--chrom", default="6")
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-table", required=True)
    return parser.parse_args()


def parse_gff_attributes(attr_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in attr_text.strip().split(";"):
        if not item:
            continue
        item = item.strip()
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
        elif " " in item:
            k, v = item.split(" ", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def normalize_chrom_name(chrom: str) -> str:
    s = str(chrom).strip().lower()
    s = re.sub(r"^chrom", "chr", s)
    m = re.match(r"^chr0*(\d+)$", s)
    if m:
        return f"chr{int(m.group(1))}"
    m = re.match(r"^0*(\d+)$", s)
    if m:
        return f"chr{int(m.group(1))}"
    return s


def chrom_matches(seqid: str, chrom_filter: str) -> bool:
    return normalize_chrom_name(seqid) == normalize_chrom_name(chrom_filter)


def load_wx_coords(gff_path: Path, chrom: str) -> Tuple[int, int]:
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, _, feature_type, start, end, _, _, _, attrs = parts
            if feature_type != "gene" or not chrom_matches(seqid, chrom):
                continue
            attr_map = parse_gff_attributes(attrs)
            if attr_map.get("Alias") == WX_LOCUS or attr_map.get("ID") == WX_LOCUS or attr_map.get("Name") == WX_LOCUS:
                return int(start), int(end)
    raise ValueError(f"Could not find {WX_LOCUS} in {gff_path}")


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
    m = re.search(r"(\d+)$", col)
    return int(m.group(1)) if m else None


def load_selected_matrix(csv_path: Path, eff_start: int, eff_end: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    pos_cols = header[1:]
    pos_vals = [parse_pos(c) for c in pos_cols]
    selected = [c for c, p in zip(pos_cols, pos_vals) if p is not None and eff_start <= p <= eff_end]
    positions = np.array([parse_pos(c) for c in selected], dtype=np.int64)
    df = pd.read_csv(csv_path, usecols=["sample_id"] + selected)
    sample_ids = df["sample_id"].astype(str).tolist()
    values = df[selected].to_numpy(dtype=np.float64)
    return positions, sample_ids, values


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
        if not np.array_equal(pos_fwd, pos_rev):
            raise ValueError(f"Position mismatch in {block_dir}")
        if sids_fwd != sids_rev:
            raise ValueError(f"Sample order mismatch in {block_dir}")
        if not sample_ids:
            sample_ids = sids_fwd
        elif sample_ids != sids_fwd:
            raise ValueError(f"Sample order mismatch across blocks in {block_dir}")
        pos_parts.append(pos_fwd)
        parts.append(vals_fwd + vals_rev)

    if not parts:
        raise RuntimeError(f"No valid blocks found under {matrix_dir}")

    positions = np.concatenate(pos_parts)
    values = np.concatenate(parts, axis=1)
    return positions, sample_ids, values


def main():
    args = parse_args()
    wx_start, wx_end = load_wx_coords(Path(args.gff), args.chrom)
    positions, sample_ids, sum_values = build_sum_signal(Path(args.matrix_dir), Path(args.bed_file))
    intervals = read_bed_intervals(Path(args.bed_file))
    region_start = intervals[0][0]
    region_end = intervals[-1][1]
    center_mask = (positions >= region_start + 4000) & (positions <= region_end - 4000)
    positions = positions[center_mask]
    sum_values = sum_values[:, center_mask]
    wx_start = max(wx_start, region_start + 4000)
    wx_end = min(wx_end, region_end - 4000)

    pheno = pd.read_csv(args.pheno_file, sep="\t")
    pheno["SampleID"] = pheno["SampleID"].astype(str)
    group_map = dict(zip(pheno["SampleID"], pheno["Trait"]))
    wx_mask = (positions >= wx_start) & (positions <= wx_end)
    if wx_mask.sum() == 0:
        raise RuntimeError("No stitched positions overlap the Wx locus")

    mean_wx = sum_values[:, wx_mask].mean(axis=1)
    choice_df = pd.DataFrame({"sample_id": sample_ids, "trait": [group_map[s] for s in sample_ids], "mean_wx_sum": mean_wx})
    g0 = choice_df[choice_df["trait"] == 0].sort_values("mean_wx_sum", ascending=True).head(2).copy()
    g1 = choice_df[choice_df["trait"] == 1].sort_values("mean_wx_sum", ascending=False).head(2).copy()
    picked = pd.concat([g0, g1], ignore_index=True)
    picked.to_csv(args.out_table, index=False)

    style_map = {
        0: [("#f8c291", "-"), ("#f8c291", "--")],
        1: [("#7fb3d5", "-"), ("#7fb3d5", "--")],
    }
    used_style = {0: 0, 1: 0}
    fig, ax = plt.subplots(figsize=(15, 4.5))
    for _, row in picked.iterrows():
        sample_id = row["sample_id"]
        trait = int(row["trait"])
        idx = sample_ids.index(sample_id)
        color, linestyle = style_map[trait][used_style[trait]]
        used_style[trait] += 1
        label = f"{'Glutinous' if trait == 0 else 'Non-glutinous'} | {sample_id}"
        ax.plot(
            positions,
            sum_values[idx, :],
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            alpha=0.78,
            label=label,
        )

    ax.axvspan(wx_start, wx_end, color="#d62728", alpha=0.08)
    ax.set_title("Wx Region Attention Signal Visualization (sum of forward + reverse)")
    ax.set_xlabel("Genomic position (bp)")
    ax.set_ylabel("Summed attention")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote plot: {args.out_png}")
    print(f"Wrote sample table: {args.out_table}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import gzip
import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.stats import entropy, iqr, kurtosis, skew


@dataclass
class BlockInfo:
    name: str
    path: str
    start: int
    end: int
    eff_start: int
    eff_end: int


@dataclass
class GeneInfo:
    seqid: str
    start: int
    end: int
    strand: str
    gene_id: str
    alias: str
    name: str


@dataclass
class MatrixSource:
    tag: str
    files: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract gene-level metrics from overlapping block score matrices. "
            "For each block, only the effective interval between overlap midpoints is used."
        )
    )
    parser.add_argument(
        "--gff",
        default=None,
        help="Path to input GFF/GFF3 file (required unless using --plot-from-out-dir)",
    )
    parser.add_argument(
        "--matrix-dir",
        nargs="+",
        required=False,
        default=None,
        help=(
            "One or more directories containing block_* folders with score matrices. "
            "Supports shell-expanded paths and wildcard patterns. "
            "Required unless using --plot-from-out-dir."
        ),
    )
    parser.add_argument(
        "--matrix-files",
        default="hap1_attention_collapsed.csv,hap1_attention_collapsed_revcomp.csv",
        help=(
            "Comma-separated matrix filenames to process independently and summarize "
            "(default: hap1_attention_collapsed.csv,hap1_attention_collapsed_revcomp.csv)"
        ),
    )
    parser.add_argument(
        "--add-direction-sum",
        action="store_true",
        help="Add an extra synthetic direction by summing base-wise scores of the first two matrix files",
    )
    parser.add_argument(
        "--sum-tag",
        default="sum_direction",
        help="Tag name for summed direction output (default: sum_direction)",
    )
    parser.add_argument(
        "--chrom",
        default=None,
        help="Chromosome/seqid filter for GFF (e.g. Chr1). If omitted, use all seqids.",
    )
    parser.add_argument(
        "--feature-type",
        default="gene",
        help="GFF feature type used as genes (default: gene)",
    )
    parser.add_argument(
        "--flank-length",
        type=int,
        default=0,
        help="Extend gene region by this many bp on both upstream and downstream sides (default: 0)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (required unless using --plot-from-out-dir)",
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Plot horizontal bar charts for all 17 metrics using shared gene labels",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Output directory for plots (default: <out-dir>/metric_plots)",
    )
    parser.add_argument(
        "--plot-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Plot file format (default: png)",
    )
    parser.add_argument(
        "--plot-topn",
        type=int,
        default=0,
        help="Only plot TopN genes by combined rank; 0 means all genes (default: 0)",
    )
    parser.add_argument(
        "--plot-label",
        default="gene_id",
        choices=["gene_id", "alias"],
        help="Label field for y-axis gene names in plots (default: gene_id)",
    )
    parser.add_argument(
        "--plot-from-out-dir",
        default=None,
        help=(
            "Read existing <out-dir>/gene_level_17metrics_directional_pvalues.csv "
            "and only regenerate plots (skip metric and p-value recomputation)."
        ),
    )
    parser.add_argument(
        "--plot-sum-separately",
        action="store_true",
        help="If set, plot summed direction in a separate directory (metric_plots/sum_*) without other directions",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for block-level processing (default: 1)",
    )
    parser.add_argument(
        "--csv-chunk-rows",
        type=int,
        default=256,
        help="Rows per CSV chunk when computing column means to reduce memory usage (default: 256)",
    )
    parser.add_argument(
        "--base-intermediate-dir",
        default=None,
        help=(
            "Directory for base-level cached matrices (*.npz). "
            "If provided, cache will be reused across runs and created on demand."
        ),
    )
    parser.add_argument(
        "--build-base-intermediate-only",
        action="store_true",
        help=(
            "Build base-level cache files for all blocks/directions and exit. "
            "Useful before sweeping multiple --flank-length values."
        ),
    )
    parser.add_argument(
        "--pheno-file",
        default=None,
        help="Optional external phenotype/group file path. If omitted, metadata.csv in each block folder is used.",
    )
    return parser.parse_args()


def parse_pos(col: str) -> Optional[int]:
    m = re.search(r"(\d+)$", col)
    if not m:
        return None
    return int(m.group(1))


def read_block_span(matrix_csv: str) -> Tuple[int, int]:
    with open(matrix_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if len(header) < 2:
        raise ValueError(f"Matrix file has no position columns: {matrix_csv}")

    first_pos = parse_pos(header[1])
    last_pos = parse_pos(header[-1])
    if first_pos is None or last_pos is None:
        raise ValueError(f"Cannot parse position columns in: {matrix_csv}")
    return first_pos, last_pos


def resolve_matrix_dirs(matrix_dir_args: List[str]) -> List[str]:
    resolved: List[str] = []
    seen = set()

    for item in matrix_dir_args:
        parts = [x.strip() for x in str(item).split(",") if x.strip()]
        for p in parts:
            matches = glob.glob(p)
            cand = matches if matches else [p]
            for c in cand:
                if os.path.isdir(c) and c not in seen:
                    resolved.append(c)
                    seen.add(c)

    if not resolved:
        raise ValueError("No valid matrix directories found from --matrix-dir inputs")
    return resolved


def sanitize_name_for_file(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def get_block_cache_path(base_intermediate_dir: str, block: BlockInfo, source_tag: str) -> str:
    os.makedirs(base_intermediate_dir, exist_ok=True)
    bname = sanitize_name_for_file(block.name)
    tname = sanitize_name_for_file(source_tag)
    return os.path.join(base_intermediate_dir, f"{bname}__{tname}.npz")


def load_effective_matrix_single(
    block: BlockInfo,
    matrix_file: str,
    csv_chunk_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix_csv = os.path.join(block.path, matrix_file)
    if not os.path.exists(matrix_csv):
        return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, 0), dtype=np.float64)

    with open(matrix_csv, "r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))

    if len(header) < 2 or header[0] != "sample_id":
        return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, 0), dtype=np.float64)

    pos_cols = header[1:]
    pos_vals = [parse_pos(c) for c in pos_cols]
    selected = [c for c, p in zip(pos_cols, pos_vals) if p is not None and block.eff_start <= p <= block.eff_end]
    if not selected:
        return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, 0), dtype=np.float64)

    positions = np.array([parse_pos(c) for c in selected], dtype=np.int64)

    sid_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []
    usecols = ["sample_id"] + selected
    for chunk in pd.read_csv(matrix_csv, usecols=usecols, chunksize=csv_chunk_rows):
        sid_chunks.append(chunk["sample_id"].astype(str).to_numpy())
        val_chunks.append(chunk[selected].to_numpy(dtype=np.float64))

    if not sid_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, positions.size), dtype=np.float64)

    sample_ids = np.concatenate(sid_chunks)
    values = np.vstack(val_chunks)
    return positions, sample_ids, values


def load_effective_matrix_sum(
    block: BlockInfo,
    matrix_file_a: str,
    matrix_file_b: str,
    csv_chunk_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_a = os.path.join(block.path, matrix_file_a)
    csv_b = os.path.join(block.path, matrix_file_b)
    if not (os.path.exists(csv_a) and os.path.exists(csv_b)):
        return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, 0), dtype=np.float64)

    with open(csv_a, "r", encoding="utf-8", newline="") as fa:
        header_a = next(csv.reader(fa))
    with open(csv_b, "r", encoding="utf-8", newline="") as fb:
        header_b = next(csv.reader(fb))

    if len(header_a) < 2 or len(header_b) < 2 or header_a[0] != "sample_id" or header_b[0] != "sample_id":
        return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, 0), dtype=np.float64)

    cols_a = header_a[1:]
    cols_b = header_b[1:]
    posa = [parse_pos(c) for c in cols_a]
    posb = [parse_pos(c) for c in cols_b]

    map_a = {int(p): c for c, p in zip(cols_a, posa) if p is not None and block.eff_start <= p <= block.eff_end}
    map_b = {int(p): c for c, p in zip(cols_b, posb) if p is not None and block.eff_start <= p <= block.eff_end}
    common_pos = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common_pos:
        return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, 0), dtype=np.float64)

    sel_a = [map_a[p] for p in common_pos]
    sel_b = [map_b[p] for p in common_pos]

    sid_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    it_a = pd.read_csv(csv_a, usecols=["sample_id"] + sel_a, chunksize=csv_chunk_rows)
    it_b = pd.read_csv(csv_b, usecols=["sample_id"] + sel_b, chunksize=csv_chunk_rows)

    while True:
        try:
            ca = next(it_a)
            cb = next(it_b)
        except StopIteration:
            break

        sid_a = ca["sample_id"].astype(str).to_numpy()
        sid_b = cb["sample_id"].astype(str).to_numpy()

        if sid_a.shape == sid_b.shape and np.all(sid_a == sid_b):
            summed = ca[sel_a].to_numpy(dtype=np.float64) + cb[sel_b].to_numpy(dtype=np.float64)
            sid_parts.append(sid_a)
            val_parts.append(summed)
        else:
            ma = ca.rename(columns={c: f"a__{c}" for c in sel_a})
            mb = cb.rename(columns={c: f"b__{c}" for c in sel_b})
            mm = ma.merge(mb, on="sample_id", how="inner")
            if mm.empty:
                continue
            sid = mm["sample_id"].astype(str).to_numpy()
            aa = mm[[f"a__{c}" for c in sel_a]].to_numpy(dtype=np.float64)
            bb = mm[[f"b__{c}" for c in sel_b]].to_numpy(dtype=np.float64)
            sid_parts.append(sid)
            val_parts.append(aa + bb)

    if not sid_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, len(common_pos)), dtype=np.float64)

    return np.array(common_pos, dtype=np.int64), np.concatenate(sid_parts), np.vstack(val_parts)


def load_or_build_block_base_matrix(
    block: BlockInfo,
    matrix_source: MatrixSource,
    csv_chunk_rows: int,
    base_intermediate_dir: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if base_intermediate_dir:
        cache_path = get_block_cache_path(base_intermediate_dir, block, matrix_source.tag)
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            return data["positions"], data["sample_ids"], data["values"]

    if len(matrix_source.files) == 1:
        positions, sample_ids, values = load_effective_matrix_single(block, matrix_source.files[0], csv_chunk_rows)
    else:
        positions, sample_ids, values = load_effective_matrix_sum(
            block,
            matrix_source.files[0],
            matrix_source.files[1],
            csv_chunk_rows,
        )

    if base_intermediate_dir:
        cache_path = get_block_cache_path(base_intermediate_dir, block, matrix_source.tag)
        np.savez_compressed(
            cache_path,
            positions=positions.astype(np.int64, copy=False),
            sample_ids=sample_ids.astype(str, copy=False),
            values=values.astype(np.float32, copy=False),
        )

    return positions, sample_ids, values


def build_matrix_sources(matrix_files: List[str], add_direction_sum: bool, sum_tag: str) -> List[MatrixSource]:
    sources = [MatrixSource(tag=matrix_tag(f), files=[f]) for f in matrix_files]
    if add_direction_sum:
        if len(matrix_files) < 2:
            raise ValueError("--add-direction-sum requires at least two files in --matrix-files")
        sources.append(MatrixSource(tag=sum_tag, files=[matrix_files[0], matrix_files[1]]))
    return sources


def load_blocks(matrix_dirs: List[str], matrix_file: str) -> List[BlockInfo]:
    raw: List[Tuple[str, str, int, int]] = []

    for matrix_dir in matrix_dirs:
        dir_tag = os.path.basename(os.path.normpath(matrix_dir))
        for d in os.listdir(matrix_dir):
            block_path = os.path.join(matrix_dir, d)
            if not (os.path.isdir(block_path) and d.startswith("block_")):
                continue
            matrix_csv = os.path.join(block_path, matrix_file)
            if not os.path.exists(matrix_csv):
                continue
            start, end = read_block_span(matrix_csv)
            uniq_name = f"{dir_tag}/{d}"
            raw.append((uniq_name, block_path, start, end))

    if not raw:
        raise ValueError(f"No valid block folders found under input directories: {matrix_dirs}")

    raw.sort(key=lambda x: (x[2], x[3]))

    blocks: List[BlockInfo] = []
    for i, (name, path, start, end) in enumerate(raw):
        left = start
        right = end

        if i > 0:
            prev_start, prev_end = raw[i - 1][2], raw[i - 1][3]
            ov_start = max(prev_start, start)
            ov_end = min(prev_end, end)
            if ov_start <= ov_end:
                mid = (ov_start + ov_end) // 2
                left = max(left, mid + 1)

        if i < len(raw) - 1:
            next_start, next_end = raw[i + 1][2], raw[i + 1][3]
            ov_start = max(start, next_start)
            ov_end = min(end, next_end)
            if ov_start <= ov_end:
                mid = (ov_start + ov_end) // 2
                right = min(right, mid)

        if left > right:
            raise ValueError(
                f"Invalid effective interval for {name}: {left}-{right}. "
                "Check block overlap/order."
            )

        blocks.append(
            BlockInfo(
                name=name,
                path=path,
                start=start,
                end=end,
                eff_start=left,
                eff_end=right,
            )
        )

    return blocks


def matrix_tag(matrix_file: str) -> str:
    base = os.path.basename(matrix_file)
    stem = re.sub(r"\.csv$", "", base)
    return re.sub(r"[^A-Za-z0-9_]+", "_", stem)


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
            # GTF-like fallback: key "value"
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


def chrom_matches(seqid: str, chrom_filter: Optional[str]) -> bool:
    if not chrom_filter:
        return True
    return normalize_chrom_name(seqid) == normalize_chrom_name(chrom_filter)


def load_genes(gff_path: str, feature_type: str, chrom: Optional[str]) -> List[GeneInfo]:
    genes: List[GeneInfo] = []
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Prefer strict GFF tab split, fallback to whitespace split for irregular spacing files.
            parts = line.split("\t")
            if len(parts) < 9:
                parts = re.split(r"\s+", line, maxsplit=8)
            if len(parts) < 9:
                continue

            seqid, _, ftype, start, end, _, strand, _, attrs = parts
            if ftype != feature_type:
                continue
            if not chrom_matches(seqid, chrom):
                continue

            attr_map = parse_gff_attributes(attrs)
            gene_id = attr_map.get("ID", "")
            alias = attr_map.get("Alias", "")
            name = attr_map.get("Name", "")
            genes.append(
                GeneInfo(
                    seqid=seqid,
                    start=int(start),
                    end=int(end),
                    strand=strand,
                    gene_id=gene_id,
                    alias=alias,
                    name=name,
                )
            )

    genes.sort(key=lambda g: (g.seqid, g.start, g.end))
    return genes


def get_percent_mean(x: np.ndarray, percent: float = 0.05, top: bool = True) -> float:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    n = max(1, int(x.size * percent))
    if top:
        return np.sort(x)[-n:].mean()
    return np.sort(x)[:n].mean()


def analyze_peaks(x: np.ndarray) -> Tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if x.size < 3:
        return 0.0, 0.0, np.nan

    peaks, props = find_peaks(x, height=float(np.mean(x)))
    peak_count = float(peaks.size)
    peak_density = peak_count / float(x.size) if x.size > 0 else 0.0
    if peaks.size == 0:
        peak_mean = np.nan
    else:
        peak_mean = float(np.mean(props["peak_heights"]))

    return peak_count, peak_density, peak_mean


def calc_entropy(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    x = x[x > 0]
    if x.size == 0:
        return 0.0
    probs = x / x.sum()
    return float(entropy(probs))


def calc_metrics(x: np.ndarray) -> Dict[str, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {
            "mean": np.nan,
            "max": np.nan,
            "std": np.nan,
            "cv": np.nan,
            "median": np.nan,
            "mode": np.nan,
            "iqr": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "top5_percent_mean": np.nan,
            "low5_percent_mean": np.nan,
            "percentile_90": np.nan,
            "percentile_10": np.nan,
            "peak_count": np.nan,
            "peak_density": np.nan,
            "peak_mean": np.nan,
            "shannon_entropy": np.nan,
        }

    mode_val = np.nan
    mode_res = stats.mode(x, keepdims=True)
    if mode_res.count.size > 0 and mode_res.count[0] > 0:
        mode_val = float(mode_res.mode[0])

    peak_count, peak_density, peak_mean = analyze_peaks(x)

    mean_val = float(np.mean(x))
    std_val = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    cv_val = std_val / mean_val if mean_val != 0 else np.nan

    return {
        "mean": mean_val,
        "max": float(np.max(x)),
        "std": std_val,
        "cv": cv_val,
        "median": float(np.median(x)),
        "mode": mode_val,
        "iqr": float(iqr(x, nan_policy="omit")),
        "skewness": float(skew(x, nan_policy="omit")) if x.size > 2 else np.nan,
        "kurtosis": float(kurtosis(x, nan_policy="omit")) if x.size > 3 else np.nan,
        "top5_percent_mean": float(get_percent_mean(x, 0.05, True)),
        "low5_percent_mean": float(get_percent_mean(x, 0.05, False)),
        "percentile_90": float(np.quantile(x, 0.9)),
        "percentile_10": float(np.quantile(x, 0.1)),
        "peak_count": peak_count,
        "peak_density": peak_density,
        "peak_mean": peak_mean,
        "shannon_entropy": calc_entropy(x),
    }


def bh_correction(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, n + 1, dtype=float)

    q_sorted = p_sorted * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    rejected = q <= alpha
    return rejected, q


def load_group_map(pheno_file: str) -> Dict[str, str]:
    pheno = pd.read_csv(pheno_file, sep=r"\s+", engine="python")
    if pheno.shape[1] < 2:
        raise ValueError(f"Invalid phenotype file format: {pheno_file}")

    sid_col = pheno.columns[0]
    trait_col = pheno.columns[1]

    group_map: Dict[str, str] = {}
    for _, row in pheno[[sid_col, trait_col]].dropna().iterrows():
        sid = str(row[sid_col]).strip()
        trait = row[trait_col]
        # Keep consistent with case/control idea in 1.calculate_17metrics.py.
        # Here 0 -> control; non-zero -> case (supports trait coded as 1 or 2).
        group_map[sid] = "control" if float(trait) == 0 else "case"

    if not group_map:
        raise ValueError(f"No valid sample groups loaded from: {pheno_file}")
    return group_map


def load_group_map_from_metadata(matrix_dirs: List[str]) -> Dict[str, str]:
    group_map: Dict[str, str] = {}

    for matrix_dir in matrix_dirs:
        for d in os.listdir(matrix_dir):
            block_path = os.path.join(matrix_dir, d)
            if not (os.path.isdir(block_path) and d.startswith("block_")):
                continue

            meta_csv = os.path.join(block_path, "metadata.csv")
            if not os.path.exists(meta_csv):
                continue

            meta = pd.read_csv(meta_csv)
            if meta.shape[1] < 2:
                continue

            sid_col = meta.columns[0]
            group_col = meta.columns[1]

            for _, row in meta[[sid_col, group_col]].dropna().iterrows():
                sid = str(row[sid_col]).strip()
                grp_val = row[group_col]
                grp = "control" if float(grp_val) == 0 else "case"

                if sid in group_map and group_map[sid] != grp:
                    raise ValueError(
                        f"Conflicting group assignments for sample {sid}: {group_map[sid]} vs {grp}"
                    )
                group_map[sid] = grp

    if not group_map:
        raise ValueError(f"No valid groups found in metadata.csv under: {matrix_dirs}")

    return group_map


def extract_gene_metrics_for_matrix(
    blocks: List[BlockInfo],
    genes_in_window: List[GeneInfo],
    matrix_source: MatrixSource,
    flank_length: int,
    workers: int,
    csv_chunk_rows: int,
    base_intermediate_dir: Optional[str],
) -> pd.DataFrame:
    gene_values: Dict[str, List[float]] = defaultdict(list)
    gene_pos_count: Dict[str, int] = defaultdict(int)
    gene_meta: Dict[str, GeneInfo] = {}

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if csv_chunk_rows < 1:
        raise ValueError("csv_chunk_rows must be >= 1")

    def load_block_profile(block: BlockInfo, matrix_file: str) -> Tuple[np.ndarray, np.ndarray]:
        matrix_csv = os.path.join(block.path, matrix_file)
        if not os.path.exists(matrix_csv):
            raise ValueError(f"Missing matrix file in block {block.name}: {matrix_file}")

        with open(matrix_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        if len(header) < 2 or header[0] != "sample_id":
            raise ValueError(f"Missing/invalid sample_id header in {matrix_csv}")

        pos_cols = header[1:]
        positions_list = [parse_pos(c) for c in pos_cols]
        valid_idx = [i for i, p in enumerate(positions_list) if p is not None]
        if not valid_idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        positions = np.array([positions_list[i] for i in valid_idx], dtype=np.int64)
        pos_cols = [pos_cols[i] for i in valid_idx]

        eff_mask = (positions >= block.eff_start) & (positions <= block.eff_end)
        if not np.any(eff_mask):
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        eff_positions = positions[eff_mask]
        eff_cols = [c for c, ok in zip(pos_cols, eff_mask) if ok]

        col_sum = np.zeros(len(eff_cols), dtype=np.float64)
        col_cnt = np.zeros(len(eff_cols), dtype=np.int64)
        for chunk in pd.read_csv(
            matrix_csv,
            usecols=eff_cols,
            chunksize=csv_chunk_rows,
            dtype=np.float32,
        ):
            arr = chunk.to_numpy(dtype=np.float64)
            valid = ~np.isnan(arr)
            col_sum += np.nansum(arr, axis=0)
            col_cnt += valid.sum(axis=0)

        profile = np.divide(
            col_sum,
            col_cnt,
            out=np.full_like(col_sum, np.nan, dtype=np.float64),
            where=col_cnt > 0,
        )
        return eff_positions, profile

    def sum_profiles(pos_a: np.ndarray, val_a: np.ndarray, pos_b: np.ndarray, val_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if pos_a.size == 0 or pos_b.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        if pos_a.shape == pos_b.shape and np.array_equal(pos_a, pos_b):
            return pos_a, val_a + val_b

        da = {int(p): float(v) for p, v in zip(pos_a, val_a) if np.isfinite(v)}
        db = {int(p): float(v) for p, v in zip(pos_b, val_b) if np.isfinite(v)}
        common = sorted(set(da.keys()) & set(db.keys()))
        if not common:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        vals = np.array([da[p] + db[p] for p in common], dtype=np.float64)
        return np.array(common, dtype=np.int64), vals

    def process_block(b: BlockInfo) -> Tuple[Dict[str, List[float]], Dict[str, int], Dict[str, GeneInfo]]:
        local_values: Dict[str, List[float]] = defaultdict(list)
        local_count: Dict[str, int] = defaultdict(int)
        local_meta: Dict[str, GeneInfo] = {}

        if base_intermediate_dir:
            eff_positions, _, values = load_or_build_block_base_matrix(
                block=b,
                matrix_source=matrix_source,
                csv_chunk_rows=csv_chunk_rows,
                base_intermediate_dir=base_intermediate_dir,
            )
            if eff_positions.size == 0 or values.size == 0:
                return local_values, local_count, local_meta
            with np.errstate(invalid="ignore"):
                profile = np.nanmean(values.astype(np.float64, copy=False), axis=0)
        elif len(matrix_source.files) == 1:
            eff_positions, profile = load_block_profile(b, matrix_source.files[0])
        else:
            pos_a, val_a = load_block_profile(b, matrix_source.files[0])
            pos_b, val_b = load_block_profile(b, matrix_source.files[1])
            eff_positions, profile = sum_profiles(pos_a, val_a, pos_b, val_b)

        if eff_positions.size == 0:
            return local_values, local_count, local_meta

        overlap_genes = [
            g
            for g in genes_in_window
            if not ((g.end + flank_length) < b.eff_start or (g.start - flank_length) > b.eff_end)
        ]

        for g in overlap_genes:
            ext_start = max(1, g.start - flank_length)
            ext_end = g.end + flank_length

            gmask = (eff_positions >= ext_start) & (eff_positions <= ext_end)
            if not np.any(gmask):
                continue

            vals = profile[gmask]
            if vals.size == 0:
                continue

            gid = g.gene_id if g.gene_id else f"{g.seqid}:{g.start}-{g.end}"
            local_values[gid].extend(vals.tolist())
            local_count[gid] += int(vals.size)
            local_meta[gid] = g

        return local_values, local_count, local_meta

    if workers == 1:
        block_results = [process_block(b) for b in blocks]
    else:
        block_results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_block, b) for b in blocks]
            for fu in as_completed(futures):
                block_results.append(fu.result())

    for local_values, local_count, local_meta in block_results:
        for gid, vals in local_values.items():
            gene_values[gid].extend(vals)
        for gid, cnt in local_count.items():
            gene_pos_count[gid] += cnt
        gene_meta.update(local_meta)

    rows = []
    for gid, vals in gene_values.items():
        g = gene_meta[gid]
        arr = np.array(vals, dtype=float)
        met = calc_metrics(arr)

        row = {
            "gene_id": gid,
            "alias": g.alias,
            "name": g.name,
            "seqid": g.seqid,
            "start": g.start,
            "end": g.end,
            "strand": g.strand,
            "gene_length": g.end - g.start + 1,
            "flank_length": flank_length,
            "region_start_with_flank": max(1, g.start - flank_length),
            "region_end_with_flank": g.end + flank_length,
            "covered_positions": gene_pos_count[gid],
        }
        row.update(met)
        rows.append(row)

    if not rows:
        raise ValueError(
            f"No gene values were extracted for matrix source {matrix_source.tag}. "
            "Check chrom/block coordinates/matrix file/flank length."
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.sort_values("mean", ascending=False).reset_index(drop=True)
    return metrics_df


def build_gene_label(row: pd.Series, label_mode: str) -> str:
    gid = str(row.get("gene_id", "")).strip()
    if label_mode == "alias":
        alias = str(row.get("alias", "")).strip()
        return alias if alias else gid
    return gid


def plot_single_direction_metrics(
    pval_df: pd.DataFrame,
    metric_names: List[str],
    direction_tag: str,
    plot_dir: str,
    plot_format: str,
    plot_topn: int,
    label_mode: str,
) -> None:
    """Plot metrics for a single direction (e.g., sum_direction) without comparison to other directions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for --plot-metrics. Please install matplotlib."
        ) from e

    os.makedirs(plot_dir, exist_ok=True)

    threshold = -np.log10(0.05)

    for metric in metric_names:
        q_col = f"{metric}_p_corrected_{direction_tag}"

        # Check if columns exist for this direction
        if q_col not in pval_df.columns:
            continue

        current = pval_df.copy()

        # Always sort by corrected p-value and then apply TopN strictly.
        qvals = pd.to_numeric(current[q_col], errors="coerce")
        current["_min_q"] = qvals.fillna(np.inf)
        current = current.sort_values("_min_q", ascending=True)

        if plot_topn > 0:
            current = current.iloc[:plot_topn].copy()

        n_gene = len(current)
        fig_h = max(5.0, min(24.0, n_gene * 0.4 + 2))
        fig, ax = plt.subplots(figsize=(12, fig_h))

        if n_gene == 0:
            ax.text(
                0.5,
                0.5,
                "No significant genes\n(P < 0.05)",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xlabel("-log10(P-value)")
            ax.set_ylabel("Genes")
            ax.set_title(f"{metric} ({direction_tag}, BH corrected)")
            fig.tight_layout()
            out_png = os.path.join(plot_dir, f"{metric}_{direction_tag}_bar.{plot_format}")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            continue

        labels = current.apply(lambda r: build_gene_label(r, label_mode), axis=1).tolist()
        y = np.arange(n_gene)

        qvals_array = np.array(pd.to_numeric(current[q_col], errors="coerce").to_numpy(dtype=float), copy=True)
        qvals_array[~np.isfinite(qvals_array)] = 1.0
        qvals_array = np.clip(qvals_array, 1e-300, 1.0)
        neg_log10_vals = -np.log10(qvals_array)

        ax.barh(
            y,
            neg_log10_vals,
            height=0.7,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            color="#3498db",
        )

        ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5, label="P=0.05")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8 if n_gene <= 80 else 6)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(P-value)")
        ax.set_ylabel("Genes")
        ax.set_title(f"{metric} ({direction_tag}, BH corrected, genes={n_gene})")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(loc="best")

        fig.tight_layout()
        out_png = os.path.join(plot_dir, f"{metric}_{direction_tag}_bar.{plot_format}")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)


def plot_metric_comparisons(
    pval_df: pd.DataFrame,
    metric_names: List[str],
    tags: List[str],
    plot_dir: str,
    plot_format: str,
    plot_topn: int,
    label_mode: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for --plot-metrics. Please install matplotlib."
        ) from e

    os.makedirs(plot_dir, exist_ok=True)

    threshold = -np.log10(0.05)

    for metric in metric_names:
        plot_tags = [t for t in tags if f"{metric}_p_corrected_{t}" in pval_df.columns]
        if not plot_tags:
            continue

        sig_mask = np.zeros(len(pval_df), dtype=bool)
        min_q = pd.Series(np.inf, index=pval_df.index, dtype=float)
        for t in plot_tags:
            sig_col = f"{metric}_significant_{t}"
            q_col = f"{metric}_p_corrected_{t}"
            if sig_col in pval_df.columns:
                sig_mask = sig_mask | pval_df[sig_col].fillna(False).to_numpy(dtype=bool)
            if q_col in pval_df.columns:
                q = pd.to_numeric(pval_df[q_col], errors="coerce")
                min_q = pd.Series(
                    np.minimum(min_q.to_numpy(dtype=float), q.fillna(np.inf).to_numpy(dtype=float)),
                    index=min_q.index,
                )

        current = pval_df[sig_mask].copy()
        if len(current) == 0:
            current = pval_df.copy()

        current["_min_q"] = min_q.loc[current.index]
        current = current.sort_values("_min_q", ascending=True)
        if plot_topn > 0:
            current = current.head(plot_topn)

        n_gene = len(current)
        fig_h = max(5.0, min(24.0, n_gene * 0.4 + 2))
        fig, ax = plt.subplots(figsize=(14, fig_h))

        if n_gene == 0:
            ax.text(
                0.5,
                0.5,
                "No significant genes\n(P < 0.05)",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_xlabel("-log10(P-value)")
            ax.set_ylabel("Genes")
            ax.set_title(f"{metric} (BH corrected)")
            fig.tight_layout()
            out_png = os.path.join(plot_dir, f"{metric}_horizontal_bar.{plot_format}")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            continue

        labels = current.apply(lambda r: build_gene_label(r, label_mode), axis=1).tolist()
        y = np.arange(n_gene)

        n_tag = len(plot_tags)
        bar_h = 0.8 / max(1, n_tag)

        for i, t in enumerate(plot_tags):
            q_col = f"{metric}_p_corrected_{t}"
            qvals = np.array(pd.to_numeric(current[q_col], errors="coerce").to_numpy(dtype=float), copy=True)
            qvals[~np.isfinite(qvals)] = 1.0
            qvals = np.clip(qvals, 1e-300, 1.0)
            neg_log10_vals = -np.log10(qvals)

            offset = (i - (n_tag - 1) / 2.0) * bar_h
            ax.barh(
                y + offset,
                neg_log10_vals,
                height=bar_h * 0.9,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                label=t,
            )

        ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5, label="P=0.05")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8 if n_gene <= 80 else 6)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(P-value)")
        ax.set_ylabel("Genes")
        ax.set_title(f"{metric} (BH corrected, genes={n_gene})")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(loc="best")

        fig.tight_layout()
        out_png = os.path.join(plot_dir, f"{metric}_horizontal_bar.{plot_format}")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)


def build_pvalue_table_for_direction(
    blocks: List[BlockInfo],
    genes_in_window: List[GeneInfo],
    matrix_source: MatrixSource,
    flank_length: int,
    group_map: Dict[str, str],
    metric_names: List[str],
    workers: int,
    csv_chunk_rows: int,
    base_intermediate_dir: Optional[str],
) -> pd.DataFrame:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    block_lookup = {(b.path, b.name): b for b in blocks}

    @lru_cache(maxsize=16)
    def get_block_base_matrix(block_path: str, block_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        block = block_lookup.get((block_path, block_name))
        if block is None:
            return np.array([], dtype=np.int64), np.array([], dtype=str), np.empty((0, 0), dtype=np.float64)
        return load_or_build_block_base_matrix(
            block=block,
            matrix_source=matrix_source,
            csv_chunk_rows=csv_chunk_rows,
            base_intermediate_dir=base_intermediate_dir,
        )

    def process_gene(g: GeneInfo) -> Dict[str, object]:
        ext_start = max(1, g.start - flank_length)
        ext_end = g.end + flank_length

        sample_values: Dict[str, List[float]] = defaultdict(list)

        if base_intermediate_dir:
            for b in blocks:
                ov_start = max(ext_start, b.eff_start)
                ov_end = min(ext_end, b.eff_end)
                if ov_start > ov_end:
                    continue

                pos, sample_ids, values = get_block_base_matrix(b.path, b.name)
                if pos.size == 0 or values.size == 0:
                    continue

                mask = (pos >= ov_start) & (pos <= ov_end)
                if not np.any(mask):
                    continue

                sub = values[:, mask]
                for sid, row in zip(sample_ids, sub):
                    good = row[np.isfinite(row)]
                    if good.size > 0:
                        sample_values[str(sid)].extend(good.tolist())
        else:
            for b in blocks:
                ov_start = max(ext_start, b.eff_start)
                ov_end = min(ext_end, b.eff_end)
                if ov_start > ov_end:
                    continue

                if len(matrix_source.files) == 1:
                    matrix_csv = os.path.join(b.path, matrix_source.files[0])
                    if not os.path.exists(matrix_csv):
                        continue

                    with open(matrix_csv, "r", encoding="utf-8", newline="") as f:
                        header = next(csv.reader(f))

                    if len(header) < 2 or header[0] != "sample_id":
                        continue

                    pos_cols = header[1:]
                    pos_vals = [parse_pos(c) for c in pos_cols]
                    selected = [c for c, p in zip(pos_cols, pos_vals) if p is not None and ov_start <= p <= ov_end]
                    if not selected:
                        continue

                    usecols = ["sample_id"] + selected
                    for chunk in pd.read_csv(matrix_csv, usecols=usecols, chunksize=csv_chunk_rows):
                        sid_arr = chunk["sample_id"].astype(str).to_numpy()
                        arr = chunk[selected].to_numpy(dtype=np.float64)
                        for sid, arr_row in zip(sid_arr, arr):
                            good = arr_row[np.isfinite(arr_row)]
                            if good.size > 0:
                                sample_values[sid].extend(good.tolist())
                else:
                    csv_a = os.path.join(b.path, matrix_source.files[0])
                    csv_b = os.path.join(b.path, matrix_source.files[1])
                    if not (os.path.exists(csv_a) and os.path.exists(csv_b)):
                        continue

                    with open(csv_a, "r", encoding="utf-8", newline="") as fa:
                        header_a = next(csv.reader(fa))
                    with open(csv_b, "r", encoding="utf-8", newline="") as fb:
                        header_b = next(csv.reader(fb))

                    if len(header_a) < 2 or len(header_b) < 2 or header_a[0] != "sample_id" or header_b[0] != "sample_id":
                        continue

                    cols_a = header_a[1:]
                    cols_b = header_b[1:]
                    posa = [parse_pos(c) for c in cols_a]
                    posb = [parse_pos(c) for c in cols_b]

                    map_a = {int(p): c for c, p in zip(cols_a, posa) if p is not None and ov_start <= p <= ov_end}
                    map_b = {int(p): c for c, p in zip(cols_b, posb) if p is not None and ov_start <= p <= ov_end}
                    common_pos = sorted(set(map_a.keys()) & set(map_b.keys()))
                    if not common_pos:
                        continue

                    sel_a = [map_a[p] for p in common_pos]
                    sel_b = [map_b[p] for p in common_pos]

                    it_a = pd.read_csv(csv_a, usecols=["sample_id"] + sel_a, chunksize=csv_chunk_rows)
                    it_b = pd.read_csv(csv_b, usecols=["sample_id"] + sel_b, chunksize=csv_chunk_rows)

                    while True:
                        try:
                            ca = next(it_a)
                            cb = next(it_b)
                        except StopIteration:
                            break

                        sid_a = ca["sample_id"].astype(str).to_numpy()
                        sid_b = cb["sample_id"].astype(str).to_numpy()

                        if sid_a.shape == sid_b.shape and np.all(sid_a == sid_b):
                            arr = ca[sel_a].to_numpy(dtype=np.float64) + cb[sel_b].to_numpy(dtype=np.float64)
                            for sid, arr_row in zip(sid_a, arr):
                                good = arr_row[np.isfinite(arr_row)]
                                if good.size > 0:
                                    sample_values[sid].extend(good.tolist())
                        else:
                            ma = ca.rename(columns={c: f"a__{c}" for c in sel_a})
                            mb = cb.rename(columns={c: f"b__{c}" for c in sel_b})
                            mm = ma.merge(mb, on="sample_id", how="inner")
                            if mm.empty:
                                continue
                            sid = mm["sample_id"].astype(str).to_numpy()
                            aa = mm[[f"a__{c}" for c in sel_a]].to_numpy(dtype=np.float64)
                            bb = mm[[f"b__{c}" for c in sel_b]].to_numpy(dtype=np.float64)
                            arr = aa + bb
                            for s, arr_row in zip(sid, arr):
                                good = arr_row[np.isfinite(arr_row)]
                                if good.size > 0:
                                    sample_values[s].extend(good.tolist())

        sample_rows = []
        for sid, vals in sample_values.items():
            grp = group_map.get(sid)
            if grp is None:
                continue
            met = calc_metrics(np.asarray(vals, dtype=float))
            met_obj: Dict[str, object] = {k: v for k, v in met.items()}
            met_obj["sample_id"] = sid
            met_obj["group"] = grp
            sample_rows.append(met_obj)

        row: Dict[str, object] = {
            "gene_id": g.gene_id if g.gene_id else f"{g.seqid}:{g.start}-{g.end}",
            "alias": g.alias,
            "name": g.name,
            "seqid": g.seqid,
            "start": g.start,
            "end": g.end,
            "strand": g.strand,
        }

        if not sample_rows:
            for m in metric_names:
                row[f"{m}_p"] = np.nan
            return row

        sdf = pd.DataFrame(sample_rows)
        for m in metric_names:
            case = sdf.loc[sdf["group"] == "case", m].dropna()
            ctrl = sdf.loc[sdf["group"] == "control", m].dropna()
            if len(case) > 0 and len(ctrl) > 0:
                _, p_val = stats.mannwhitneyu(case, ctrl, alternative="two-sided")
            else:
                p_val = np.nan
            row[f"{m}_p"] = p_val

        return row

    if workers == 1:
        rows = [process_gene(g) for g in genes_in_window]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_gene, g) for g in genes_in_window]
            for fu in as_completed(futures):
                rows.append(fu.result())

    pval_df = pd.DataFrame(rows)

    for m in metric_names:
        p_col = f"{m}_p"
        pc_col = f"{m}_p_corrected"
        s_col = f"{m}_significant"

        pval_df[pc_col] = np.nan
        pval_df[s_col] = False

        valid = pval_df[p_col].notna() & np.isfinite(pval_df[p_col])
        if valid.any():
            rejected, p_corr = bh_correction(pval_df.loc[valid, p_col].to_numpy(dtype=float), alpha=0.05)
            pval_df.loc[valid, pc_col] = p_corr
            pval_df.loc[valid, s_col] = rejected

    return pval_df


def infer_tags_from_pvalue_table(pval_df: pd.DataFrame, metric_names: List[str]) -> List[str]:
    tags: List[str] = []
    seen = set()
    for m in metric_names:
        prefix = f"{m}_p_corrected_"
        for c in pval_df.columns:
            if c.startswith(prefix):
                t = c[len(prefix):]
                if t and t not in seen:
                    seen.add(t)
                    tags.append(t)
    return tags


def render_metric_plots(
    pval_df: pd.DataFrame,
    metric_names: List[str],
    tags: List[str],
    out_dir: str,
    plot_dir: Optional[str],
    plot_format: str,
    plot_topn: int,
    plot_sum_separately: bool,
    plot_label: str,
) -> str:
    final_plot_dir = plot_dir if plot_dir else os.path.join(out_dir, "metric_plots")

    if plot_sum_separately:
        sum_tags = [t for t in tags if t.startswith("sum_")]

        if sum_tags:
            for sum_tag in sum_tags:
                sum_plot_dir = os.path.join(final_plot_dir, f"plots_{sum_tag}")
                plot_single_direction_metrics(
                    pval_df=pval_df,
                    metric_names=metric_names,
                    direction_tag=sum_tag,
                    plot_dir=sum_plot_dir,
                    plot_format=plot_format,
                    plot_topn=plot_topn,
                    label_mode=plot_label,
                )
                print(f"Saved sum_direction plots: {sum_plot_dir}")

            non_sum_tags = [t for t in tags if not t.startswith("sum_")]
            if non_sum_tags:
                other_plot_dir = os.path.join(final_plot_dir, "plots_others")
                plot_metric_comparisons(
                    pval_df=pval_df,
                    metric_names=metric_names,
                    tags=non_sum_tags,
                    plot_dir=other_plot_dir,
                    plot_format=plot_format,
                    plot_topn=plot_topn,
                    label_mode=plot_label,
                )
                print(f"Saved other_directions plots: {other_plot_dir}")
        else:
            plot_metric_comparisons(
                pval_df=pval_df,
                metric_names=metric_names,
                tags=tags,
                plot_dir=final_plot_dir,
                plot_format=plot_format,
                plot_topn=plot_topn,
                label_mode=plot_label,
            )
            print(f"Saved plots:   {final_plot_dir}")
    else:
        plot_metric_comparisons(
            pval_df=pval_df,
            metric_names=metric_names,
            tags=tags,
            plot_dir=final_plot_dir,
            plot_format=plot_format,
            plot_topn=plot_topn,
            label_mode=plot_label,
        )
        print(f"Saved plots:   {final_plot_dir}")

    return final_plot_dir


def main() -> None:
    args = parse_args()

    if args.flank_length < 0:
        raise ValueError("--flank-length must be >= 0")
    if args.plot_topn < 0:
        raise ValueError("--plot-topn must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.csv_chunk_rows < 1:
        raise ValueError("--csv-chunk-rows must be >= 1")

    matrix_files = [x.strip() for x in args.matrix_files.split(",") if x.strip()]
    if not matrix_files:
        raise ValueError("No matrix files provided via --matrix-files")
    matrix_sources = build_matrix_sources(matrix_files, args.add_direction_sum, args.sum_tag)
    tags = [s.tag for s in matrix_sources]

    metric_names = [
        "mean",
        "max",
        "std",
        "cv",
        "median",
        "mode",
        "iqr",
        "skewness",
        "kurtosis",
        "top5_percent_mean",
        "low5_percent_mean",
        "percentile_90",
        "percentile_10",
        "peak_count",
        "peak_density",
        "peak_mean",
        "shannon_entropy",
    ]

    if args.plot_from_out_dir:
        in_out_dir = args.plot_from_out_dir
        pval_path = os.path.join(in_out_dir, "gene_level_17metrics_directional_pvalues.csv")
        if not os.path.exists(pval_path):
            raise ValueError(f"P-value file not found for plot-only mode: {pval_path}")

        pval_merged = pd.read_csv(pval_path)
        tags = infer_tags_from_pvalue_table(pval_merged, metric_names)
        if not tags:
            raise ValueError(
                "No direction tags inferred from p-value table. "
                "Expected columns like <metric>_p_corrected_<tag>."
            )

        print(f"Loaded existing pvalues: {pval_path}")
        print(f"Detected direction tags: {', '.join(tags)}")
        final_plot_dir = render_metric_plots(
            pval_df=pval_merged,
            metric_names=metric_names,
            tags=tags,
            out_dir=in_out_dir,
            plot_dir=args.plot_dir,
            plot_format=args.plot_format,
            plot_topn=args.plot_topn,
            plot_sum_separately=args.plot_sum_separately,
            plot_label=args.plot_label,
        )
        print(f"Updated plots only from existing outputs: {final_plot_dir}")
        return

    if not args.matrix_dir:
        raise ValueError("--matrix-dir is required unless using --plot-from-out-dir")
    if not args.build_base_intermediate_only:
        if not args.out_dir:
            raise ValueError("--out-dir is required unless using --plot-from-out-dir")
        if not args.gff:
            raise ValueError("--gff is required unless using --plot-from-out-dir")
    if args.build_base_intermediate_only and not args.base_intermediate_dir:
        raise ValueError("--build-base-intermediate-only requires --base-intermediate-dir")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.base_intermediate_dir:
        os.makedirs(args.base_intermediate_dir, exist_ok=True)

    matrix_dirs = resolve_matrix_dirs(args.matrix_dir)
    print(f"Resolved matrix dirs: {len(matrix_dirs)}")

    blocks = load_blocks(matrix_dirs, matrix_files[0])
    print(f"Loaded {len(blocks)} blocks")

    if args.build_base_intermediate_only:
        for ms in matrix_sources:
            print(f"Building base cache for source: {ms.tag}")
            for b in blocks:
                load_or_build_block_base_matrix(
                    block=b,
                    matrix_source=ms,
                    csv_chunk_rows=args.csv_chunk_rows,
                    base_intermediate_dir=args.base_intermediate_dir,
                )
        print(f"Base cache ready: {args.base_intermediate_dir}")
        return

    genes = load_genes(args.gff, args.feature_type, args.chrom)
    if not genes:
        raise ValueError("No genes loaded from GFF with current filters")
    print(f"Loaded {len(genes)} genes from GFF")

    min_eff = min(b.eff_start for b in blocks)
    max_eff = max(b.eff_end for b in blocks)
    genes_in_window = [
        g
        for g in genes
        if not ((g.end + args.flank_length) < min_eff or (g.start - args.flank_length) > max_eff)
    ]
    print(f"Genes overlapping effective block window: {len(genes_in_window)}")

    per_dir_metrics: Dict[str, pd.DataFrame] = {}
    per_dir_ranks: Dict[str, pd.DataFrame] = {}

    base_cols = [
        "gene_id",
        "alias",
        "name",
        "seqid",
        "start",
        "end",
        "strand",
        "gene_length",
        "flank_length",
        "region_start_with_flank",
        "region_end_with_flank",
        "covered_positions",
    ]

    for matrix_source in matrix_sources:
        tag = matrix_source.tag
        print(f"Processing matrix source: {tag} ({','.join(matrix_source.files)})")

        metrics_df = extract_gene_metrics_for_matrix(
            blocks=blocks,
            genes_in_window=genes_in_window,
            matrix_source=matrix_source,
            flank_length=args.flank_length,
            workers=args.workers,
            csv_chunk_rows=args.csv_chunk_rows,
            base_intermediate_dir=args.base_intermediate_dir,
        )

        rank_df = metrics_df[base_cols].copy()
        for m in metric_names:
            rank_df[f"{m}_rank"] = metrics_df[m].rank(method="min", ascending=False)
        rank_cols = [f"{m}_rank" for m in metric_names]
        rank_df["mean_rank"] = rank_df[rank_cols].mean(axis=1)
        rank_df = rank_df.sort_values("mean_rank", ascending=True).reset_index(drop=True)

        per_dir_metrics[tag] = metrics_df
        per_dir_ranks[tag] = rank_df

        out_metrics = os.path.join(args.out_dir, f"gene_level_17metrics.{tag}.csv")
        out_ranks = os.path.join(args.out_dir, f"gene_level_metric_ranks.{tag}.csv")
        metrics_df.to_csv(out_metrics, index=False)
        rank_df.to_csv(out_ranks, index=False)
        print(f"Saved metrics ({tag}): {out_metrics}")
        print(f"Saved ranks   ({tag}): {out_ranks}")

    merged = None
    for tag, metrics_df in per_dir_metrics.items():
        rename_map = {m: f"{m}_{tag}" for m in metric_names}
        rename_map["covered_positions"] = f"covered_positions_{tag}"
        rename_map["mean"] = f"mean_{tag}"
        mdf = metrics_df.rename(columns=rename_map)
        keep_cols = base_cols[:-1] + [f"covered_positions_{tag}"] + [f"{m}_{tag}" for m in metric_names if m != "mean"] + [f"mean_{tag}"]
        mdf = mdf[keep_cols]

        if merged is None:
            merged = mdf
        else:
            merged = merged.merge(mdf, on=base_cols[:-1], how="outer")

    if merged is None:
        raise ValueError("No matrix results to summarize")

    summary = merged.copy()
    for m in metric_names:
        dir_rank_cols = []
        for tag, metrics_df in per_dir_metrics.items():
            col = f"{m}_{tag}"
            if col in summary.columns:
                rank_col = f"{m}_rank_{tag}"
                summary[rank_col] = summary[col].rank(method="min", ascending=False)
                dir_rank_cols.append(rank_col)
        summary[f"{m}_rank_combined"] = summary[dir_rank_cols].mean(axis=1)

    mean_rank_cols = [f"{m}_rank_combined" for m in metric_names]
    summary["mean_rank_combined"] = summary[mean_rank_cols].mean(axis=1)
    summary = summary.sort_values("mean_rank_combined", ascending=True).reset_index(drop=True)

    if args.pheno_file:
        group_map = load_group_map(args.pheno_file)
        print(f"Loaded groups from phenotype file: {args.pheno_file}")
    else:
        group_map = load_group_map_from_metadata(matrix_dirs)
        print("Loaded groups from block metadata.csv files")
    pval_merged = None
    for matrix_source in matrix_sources:
        tag = matrix_source.tag
        print(f"Computing case/control p-values for source: {tag}")
        pdir = build_pvalue_table_for_direction(
            blocks=blocks,
            genes_in_window=genes_in_window,
            matrix_source=matrix_source,
            flank_length=args.flank_length,
            group_map=group_map,
            metric_names=metric_names,
            workers=args.workers,
            csv_chunk_rows=args.csv_chunk_rows,
            base_intermediate_dir=args.base_intermediate_dir,
        )

        rename = {}
        for m in metric_names:
            rename[f"{m}_p"] = f"{m}_p_{tag}"
            rename[f"{m}_p_corrected"] = f"{m}_p_corrected_{tag}"
            rename[f"{m}_significant"] = f"{m}_significant_{tag}"
        pdir = pdir.rename(columns=rename)

        key_cols = ["gene_id", "alias", "name", "seqid", "start", "end", "strand"]
        keep_cols = key_cols + [
            c
            for c in pdir.columns
            if c not in key_cols
        ]
        pdir = pdir[keep_cols]

        if pval_merged is None:
            pval_merged = pdir
        else:
            pval_merged = pval_merged.merge(pdir, on=key_cols, how="outer")

    if pval_merged is None:
        raise ValueError("Failed to compute p-value table")

    pval_merged = pval_merged.merge(
        summary[["gene_id", "mean_rank_combined"]],
        on="gene_id",
        how="left",
    )
    pval_merged = pval_merged.sort_values("mean_rank_combined", ascending=True)

    out_pval = os.path.join(args.out_dir, "gene_level_17metrics_directional_pvalues.csv")
    pval_merged.to_csv(out_pval, index=False)
    print(f"Saved pvalues: {out_pval}")

    if args.plot_metrics:
        render_metric_plots(
            pval_df=pval_merged,
            metric_names=metric_names,
            tags=tags,
            out_dir=args.out_dir,
            plot_dir=args.plot_dir,
            plot_format=args.plot_format,
            plot_topn=args.plot_topn,
            plot_sum_separately=args.plot_sum_separately,
            plot_label=args.plot_label,
        )

    block_df = pd.DataFrame(
        [
            {
                "block": b.name,
                "raw_start": b.start,
                "raw_end": b.end,
                "effective_start": b.eff_start,
                "effective_end": b.eff_end,
                "effective_length": b.eff_end - b.eff_start + 1,
            }
            for b in blocks
        ]
    )

    out_blocks = os.path.join(args.out_dir, "block_effective_ranges.csv")
    out_summary = os.path.join(args.out_dir, "gene_level_metrics_summary_combined.csv")

    block_df.to_csv(out_blocks, index=False)
    summary.to_csv(out_summary, index=False)

    print(f"Saved blocks:  {out_blocks}")
    print(f"Saved summary: {out_summary}")


if __name__ == "__main__":
    main()

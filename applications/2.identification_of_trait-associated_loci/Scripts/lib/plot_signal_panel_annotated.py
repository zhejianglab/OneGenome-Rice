#!/usr/bin/env python3
import argparse
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


@dataclass
class GeneFeature:
    seqid: str
    start: int
    end: int
    gene_id: str
    alias: str
    name: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot stacked region signal panels with midpoint-partitioned blocks and gene tracks."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--region-dir", nargs="+", required=True)
    parser.add_argument("--bed-file", nargs="+", required=True)
    parser.add_argument("--matrix-dir", nargs="+")
    parser.add_argument("--gff", required=True)
    parser.add_argument("--chrom", default="6")
    parser.add_argument("--metric", choices=["log2fc", "neglog10_padj"], default="neglog10_padj")
    parser.add_argument("--display-title", nargs="+")
    parser.add_argument("--out-png", required=True)
    return parser.parse_args()


def parse_gff_attributes(attr_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in attr_text.strip().split(";"):
        if not item:
            continue
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip().strip('"')
        elif " " in item:
            key, value = item.split(" ", 1)
            out[key.strip()] = value.strip().strip('"')
    return out


def normalize_chrom_name(chrom: str) -> str:
    value = str(chrom).strip().lower()
    value = re.sub(r"^chrom", "chr", value)
    match = re.match(r"^chr0*(\d+)$", value)
    if match:
        return f"chr{int(match.group(1))}"
    match = re.match(r"^0*(\d+)$", value)
    if match:
        return f"chr{int(match.group(1))}"
    return value


def chrom_matches(seqid: str, chrom_filter: str) -> bool:
    return normalize_chrom_name(seqid) == normalize_chrom_name(chrom_filter)


def load_genes(gff_path: Path, chrom: str) -> List[GeneFeature]:
    genes: List[GeneFeature] = []
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, _, feature_type, start, end, _, _, _, attrs = parts
            if feature_type != "gene":
                continue
            if not chrom_matches(seqid, chrom):
                continue
            attr_map = parse_gff_attributes(attrs)
            genes.append(
                GeneFeature(
                    seqid=seqid,
                    start=int(start),
                    end=int(end),
                    gene_id=attr_map.get("ID", ""),
                    alias=attr_map.get("Alias", ""),
                    name=attr_map.get("Name", ""),
                )
            )
    genes.sort(key=lambda item: (item.start, item.end))
    return genes


def choose_gene_label(gene: GeneFeature) -> str:
    for value in [gene.alias, gene.name, gene.gene_id]:
        if value and value.startswith("LOC_Os"):
            return value
    return gene.alias or gene.name or gene.gene_id


def read_block_csv(base_dir: Path, block_id: int, direction: str) -> pd.DataFrame:
    csv_path = base_dir / f"block_{block_id}" / "tables" / f"{direction}_group0vs1_all_results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "position" in df.columns:
        df["pos"] = df["position"].astype(str).str.replace("pos_", "", regex=False).astype(int)
    elif "pos" not in df.columns:
        return pd.DataFrame()
    return df


def read_matrix_positions(matrix_dir: Path, block_id: int) -> np.ndarray:
    block_dir = matrix_dir / f"block_{block_id}"
    for filename in ["hap1_attention_collapsed.csv", "hap1_attention_collapsed_revcomp.csv"]:
        csv_path = block_dir / filename
        if not csv_path.exists():
            continue
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        pos_cols = [c for c in cols if c.startswith("pos_")]
        if pos_cols:
            return np.array([int(c.replace("pos_", "")) for c in pos_cols], dtype=int)
    return np.array([], dtype=int)


def prepare_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return df
    if metric == "log2fc":
        if "log2fc" not in df.columns:
            return pd.DataFrame()
        value_col = "log2fc"
    else:
        if "padj" not in df.columns:
            return pd.DataFrame()
        df = df[df["padj"].notna()].copy()
        if df.empty:
            return df
        df["neglog10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
        value_col = "neglog10_padj"
    return df[df[value_col].notna()].copy()


def read_bed_intervals(bed_file: Path) -> List[Tuple[int, int]]:
    bed = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end"])
    intervals = []
    for _, row in bed.iterrows():
        intervals.append((int(row["start"]) + 1, int(row["end"])))
    return intervals


def compute_effective_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    effective: List[Tuple[int, int]] = []
    for idx, (start, end) in enumerate(intervals):
        left = start
        right = end
        if idx > 0:
            prev_start, prev_end = intervals[idx - 1]
            ov_start = max(prev_start, start)
            ov_end = min(prev_end, end)
            if ov_start <= ov_end:
                left = max(left, (ov_start + ov_end) // 2 + 1)
        if idx < len(intervals) - 1:
            next_start, next_end = intervals[idx + 1]
            ov_start = max(start, next_start)
            ov_end = min(end, next_end)
            if ov_start <= ov_end:
                right = min(right, (ov_start + ov_end) // 2)
        if left <= right:
            effective.append((left, right))
        else:
            effective.append((start, end))
    return effective


def region_label_from_dir(region_dir: Path) -> str:
    match = re.search(r"_(\d+\.[^.]+)$", region_dir.name)
    if match:
        return match.group(1)
    return region_dir.name


def display_region_title(region_name: str) -> str:
    match = re.match(r"^(\d+)\.", region_name)
    if match:
        return f"Region {int(match.group(1))}"
    return region_name


def metric_config(metric: str):
    if metric == "log2fc":
        return {
            "ylabel": "log2FC",
            "fwd_color": "#4874CB",
            "rev_color": "#75BD42",
            "baseline": 0.0,
            "thresholds": [1.0, -1.0],
            "threshold_color": "#8b0000",
        }
    return {
        "ylabel": "-log10(adjusted P)",
        "fwd_color": "#4874CB",
        "rev_color": "#75BD42",
        "baseline": 0.0,
        "thresholds": [-np.log10(0.05)],
        "threshold_color": "#8c564b",
        "display_floor": 0.05,
    }


def load_region_curves(region_dir: Path, bed_file: Path, metric: str, matrix_dir: Optional[Path] = None):
    intervals = read_bed_intervals(bed_file)
    effective = compute_effective_intervals(intervals)
    rows = {"fwd": [], "revcomp": []}
    for idx, (eff_start, eff_end) in enumerate(effective, start=1):
        scaffold_pos = None
        if matrix_dir is not None:
            matrix_pos = read_matrix_positions(matrix_dir, idx)
            if matrix_pos.size:
                scaffold_pos = matrix_pos[(matrix_pos >= eff_start) & (matrix_pos <= eff_end)]
        if scaffold_pos is None:
            scaffold_pos = np.arange(eff_start, eff_end + 1, dtype=int)
        for direction in ["fwd", "revcomp"]:
            df = read_block_csv(region_dir, idx, direction)
            if df.empty:
                scaffold_df = pd.DataFrame({"pos": scaffold_pos, metric: np.zeros(len(scaffold_pos), dtype=float)})
                rows[direction].append(scaffold_df)
                continue
            df = prepare_metric(df, metric)
            if df.empty:
                scaffold_df = pd.DataFrame({"pos": scaffold_pos, metric: np.zeros(len(scaffold_pos), dtype=float)})
                rows[direction].append(scaffold_df)
                continue
            df = df[(df["pos"] >= eff_start) & (df["pos"] <= eff_end)][["pos", metric]].copy()
            merged = pd.DataFrame({"pos": scaffold_pos}).merge(df, on="pos", how="left")
            merged[metric] = merged[metric].fillna(0.0)
            rows[direction].append(merged)

    out = {}
    for direction in ["fwd", "revcomp"]:
        if rows[direction]:
            out[direction] = (
                pd.concat(rows[direction], ignore_index=True)
                .sort_values("pos")
                .drop_duplicates(subset=["pos"], keep="first")
            )
        else:
            out[direction] = pd.DataFrame(columns=["pos", metric])

    region_start = min(start for start, _ in effective)
    region_end = max(end for _, end in effective)
    return out, region_start, region_end


def assign_gene_lanes(genes: List[GeneFeature]) -> List[Tuple[GeneFeature, int]]:
    placed: List[Tuple[GeneFeature, int]] = []
    for idx, gene in enumerate(sorted(genes, key=lambda item: (item.start, item.end))):
        placed.append((gene, 1 if idx % 2 == 0 else 0))
    return placed


def plot_gene_track(ax, genes: List[GeneFeature], region_start: int, region_end: int):
    visible = [gene for gene in genes if gene.end >= region_start and gene.start <= region_end]
    placed = assign_gene_lanes(visible)
    lane_count = 2 if placed else 1
    rect_height = 0.34
    lane_pitch = 2.20
    label_pad = 0.001

    ax.set_xlim(region_start, region_end)
    ax.set_ylim(-0.10, lane_count * lane_pitch + 0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=17, pad=1, length=2)
    ax.grid(axis="x", color="#d9d9d9", linestyle="-", linewidth=0.6, alpha=0.75)

    for gene, lane in placed:
        y0 = (lane_count - lane - 1) * lane_pitch + 0.16
        x0 = max(gene.start, region_start)
        x1 = min(gene.end, region_end)
        rect = Rectangle(
            (x0, y0),
            max(x1 - x0, 1),
            rect_height,
            facecolor="#F2BA02",
            edgecolor="#F2BA02",
            linewidth=0.8,
            alpha=0.98,
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0 + rect_height + label_pad,
            choose_gene_label(gene),
            ha="left",
            va="bottom",
            fontsize=14.5,
            color="#1f1f1f",
            clip_on=False,
        )
    ax.set_xlabel("Genomic Position (Mb)", ha="right", x=1.0, fontsize=19)


def plot_signal_axis(ax, curves: Dict[str, pd.DataFrame], metric: str, cfg: Dict[str, object], ylim: Tuple[float, float], show_legend: bool):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    seen = {"fwd": False, "revcomp": False}
    for direction, color, style in [("fwd", cfg["fwd_color"], "-"), ("revcomp", cfg["rev_color"], "-")]:
        df = curves[direction]
        if df.empty:
            continue
        values = df[metric].to_numpy(dtype=float)
        display_floor = cfg.get("display_floor")
        if display_floor is not None:
            values = np.where(values <= 0, float(display_floor), values)
        ax.plot(
            df["pos"],
            values,
            color=color,
            linestyle=style,
            linewidth=0.95,
            alpha=0.95 if direction == "fwd" else 0.75,
            label=("Forward attention" if direction == "fwd" else "Reverse-complement attention") if not seen[direction] else "",
        )
        seen[direction] = True

    for threshold in cfg["thresholds"]:
        ax.axhline(threshold, color=cfg["threshold_color"], linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(cfg["baseline"], color="black", linestyle="-", linewidth=0.8, alpha=0.25)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#d9d9d9", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=17)

    handles, labels = ax.get_legend_handles_labels()
    if handles and show_legend:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True, fontsize=17)


def apply_xaxis_formatter(ax):
    xmin, xmax = ax.get_xlim()
    tick_step = 10_000
    start = int(np.ceil(xmin / tick_step) * tick_step)
    end = int(np.floor((xmax - 1) / tick_step) * tick_step)
    ticks = np.arange(start, end + 1, tick_step, dtype=int)
    ticks = ticks[(ticks >= xmin) & (ticks < xmax)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.2f}"))
    ax.xaxis.get_offset_text().set_visible(False)


def find_global_peak(payload, metric: str) -> Optional[Tuple[str, int, float]]:
    if metric != "neglog10_padj":
        return None
    best: Optional[Tuple[str, int, float]] = None
    for region_name, curves, _, _ in payload:
        merged = curves["fwd"].merge(
            curves["revcomp"],
            on="pos",
            how="outer",
            suffixes=("_fwd", "_rev"),
        ).sort_values("pos")
        if merged.empty:
            continue
        fwd_col = f"{metric}_fwd" if f"{metric}_fwd" in merged.columns else None
        rev_col = f"{metric}_rev" if f"{metric}_rev" in merged.columns else None
        fwd = merged[fwd_col].to_numpy(dtype=float) if fwd_col else np.full(len(merged), np.nan)
        rev = merged[rev_col].to_numpy(dtype=float) if rev_col else np.full(len(merged), np.nan)
        combined = np.nanmax(np.vstack([fwd, rev]), axis=0)
        if not np.isfinite(combined).any():
            continue
        idx = int(np.nanargmax(combined))
        peak = float(combined[idx])
        pos = int(merged.iloc[idx]["pos"])
        if best is None or peak > best[2]:
            best = (region_name, pos, peak)
    return best


def main():
    args = parse_args()
    if len(args.region_dir) != len(args.bed_file):
        raise ValueError("--region-dir and --bed-file must have the same length")
    if args.matrix_dir is not None and len(args.matrix_dir) != len(args.region_dir):
        raise ValueError("--matrix-dir and --region-dir must have the same length")
    if args.display_title is not None and len(args.display_title) != len(args.region_dir):
        raise ValueError("--display-title and --region-dir must have the same length")

    genes = load_genes(Path(args.gff), args.chrom)
    cfg = metric_config(args.metric)
    payload = []
    all_values = []

    matrix_dirs = args.matrix_dir if args.matrix_dir is not None else [None] * len(args.region_dir)
    for region_dir_str, bed_str, matrix_dir_str in zip(args.region_dir, args.bed_file, matrix_dirs):
        region_dir = Path(region_dir_str)
        bed_file = Path(bed_str)
        matrix_dir = Path(matrix_dir_str) if matrix_dir_str else None
        curves, region_start, region_end = load_region_curves(region_dir, bed_file, args.metric, matrix_dir=matrix_dir)
        region_name = region_label_from_dir(region_dir)
        payload.append((region_name, curves, region_start, region_end))
        for direction in ["fwd", "revcomp"]:
            if not curves[direction].empty:
                all_values.append(curves[direction][args.metric].to_numpy(dtype=float))

    if all_values:
        y_all = np.concatenate(all_values)
        if args.metric == "log2fc":
            ymax = np.nanmax(np.abs(y_all)) if y_all.size else 1.0
            ylim = (-ymax * 1.08, ymax * 1.08)
        else:
            ymax = np.nanmax(y_all) if y_all.size else 1.0
            ylim = (0.0, ymax * 1.08)
    else:
        ylim = (-1.0, 1.0) if args.metric == "log2fc" else (0.0, 1.0)

    global_peak = find_global_peak(payload, args.metric)

    fig_height = max(6.0, 4.0 * len(payload))
    fig = plt.figure(figsize=(24, fig_height))
    gs = GridSpec(
        nrows=len(payload) * 3,
        ncols=1,
        height_ratios=[4, 1, 0.9] * len(payload),
        hspace=0.10,
        figure=fig,
    )

    display_titles = args.display_title if args.display_title is not None else [None] * len(payload)
    for idx, ((region_name, curves, region_start, region_end), custom_title) in enumerate(zip(payload, display_titles)):
        signal_ax = fig.add_subplot(gs[idx * 3, 0])
        gene_ax = fig.add_subplot(gs[idx * 3 + 1, 0], sharex=signal_ax)
        spacer_ax = fig.add_subplot(gs[idx * 3 + 2, 0])
        spacer_ax.axis("off")

        plot_signal_axis(signal_ax, curves, args.metric, cfg, ylim, show_legend=(idx == 0))

        if global_peak is not None and region_name == global_peak[0]:
            signal_ax.axvline(global_peak[1], color="#e31a1c", linewidth=1.6, alpha=0.95, linestyle="--")

        plot_gene_track(gene_ax, genes, region_start, region_end)
        apply_xaxis_formatter(gene_ax)

        signal_ax.set_ylabel(cfg["ylabel"], fontsize=15)
        shown_title = custom_title if custom_title else display_region_title(region_name)
        signal_ax.text(0.003, 0.96, shown_title, transform=signal_ax.transAxes, ha="left", va="top", fontsize=21)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted panel: {out_png}")


if __name__ == "__main__":
    main()

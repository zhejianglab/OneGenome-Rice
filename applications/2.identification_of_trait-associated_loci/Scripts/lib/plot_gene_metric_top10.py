#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WX = "LOC_Os06g04200"
ALK = "LOC_Os06g12450"
METRICS = [
    ("Peak Density", "Peak density", "peak_density_p_corrected_sum_direction"),
    ("Shannon Entropy", "Shannon entropy", "shannon_entropy_p_corrected_sum_direction"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a 1x2 top-N summary for selected sum-direction gene metrics.")
    parser.add_argument("--long-csv", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def display_label(gene: str) -> str:
    if gene == WX:
        return "LOC_Os06g04200\n($\\it{Wx}$)"
    if gene == ALK:
        return "LOC_Os06g12450\n($\\it{ALK}$)"
    return gene


def bar_color(gene: str) -> str:
    if gene == WX:
        return "#EE822F"
    if gene == ALK:
        return "#4874CB"
    return "#bdbdbd"


def tick_color(gene: str) -> str:
    if gene == WX:
        return "#B65C18"
    if gene == ALK:
        return "#274C96"
    return "#222222"


def neglog10(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.mask(numeric <= 0, np.nextafter(0, 1))
    return -np.log10(numeric)


def load_panels(input_csv: str, top_n: int):
    df = pd.read_csv(input_csv)

    if {"direction", "metric", "gene_label", "neglog10_p_adj"}.issubset(df.columns):
        df = df[df["direction"] == "Sum"].copy()
        panels = []
        for metric_key, metric_label, _ in METRICS:
            one = (
                df[df["metric"] == metric_key]
                .copy()
                .sort_values("neglog10_p_adj", ascending=False)
                .head(top_n)
            )
            panels.append((metric_label, one))
        return panels

    if "alias" not in df.columns:
        raise ValueError(
            "Input must be either a long metric table with direction/metric/gene_label/neglog10_p_adj "
            "or the wide gene_level_17metrics_directional_pvalues.csv table with an alias column."
        )

    panels = []
    for metric_key, metric_label, p_col in METRICS:
        if p_col not in df.columns:
            raise ValueError(f"Missing required column for wide-table plotting: {p_col}")
        one = pd.DataFrame(
            {
                "metric": metric_key,
                "direction": "Sum",
                "gene_label": df["alias"].astype(str),
                "neglog10_p_adj": neglog10(df[p_col]),
            }
        )
        one = one.dropna(subset=["neglog10_p_adj"]).sort_values("neglog10_p_adj", ascending=False).head(top_n)
        panels.append((metric_label, one))
    return panels


def main():
    args = parse_args()
    panels = load_panels(args.long_csv, args.top_n)
    normalized_panels = []
    for metric_label, one in panels:
        one["gene_label"] = one["gene_label"].astype(str)
        one["shown_label"] = one["gene_label"].map(display_label)
        one["bar_color"] = one["gene_label"].map(bar_color)
        normalized_panels.append((metric_label, one))
    panels = normalized_panels

    export_rows = []
    for metric_label, one in panels:
        export = one[["metric", "direction", "gene_label", "neglog10_p_adj"]].copy()
        export_rows.append(export)
    pd.concat(export_rows, ignore_index=True).to_csv(args.out_csv, index=False)

    xmax = max(one["neglog10_p_adj"].max() for _, one in panels) * 1.10
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), sharex=True)

    for ax, (metric_label, one) in zip(axes, panels):
        labels = one["shown_label"].tolist() + ["..."]
        values = one["neglog10_p_adj"].tolist() + [0.02 * xmax]
        colors = one["bar_color"].tolist() + ["#d9d9d9"]
        y = np.arange(len(labels))

        ax.barh(y, values, color=colors, edgecolor="none", height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(metric_label, fontsize=14)
        ax.set_xlabel("-log10(BH-adjusted P)")
        ax.grid(axis="x", linestyle=":", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

        for tick, gene in zip(ax.get_yticklabels()[:-1], one["gene_label"].tolist()):
            tick.set_color(tick_color(gene))
        ax.get_yticklabels()[-1].set_color("#555555")

    fig.tight_layout(w_pad=2.0)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_png}")
    print(f"Wrote table: {args.out_csv}")


if __name__ == "__main__":
    main()

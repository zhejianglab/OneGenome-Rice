#!/usr/bin/env python3
"""
expr_analysis.py
计算基因平均覆盖度 → 差异表达 → 火山图/MA 图/热图
示例：
    ./expr_analysis.py pred.bw raw.bw gencode.feather -o result_dir --norm quantile
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyBigWig
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from tqdm.auto import tqdm

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# ------------ 读取 GTF ------------
def load_gene_regions(gtf_feather: Path, chrom: str = "chr19") -> pd.DataFrame:
    gtf = pd.read_feather(gtf_feather)
    genes = (
        gtf.query("Feature == 'gene' and Chromosome == @chrom")
        .loc[:, ["gene_name", "Chromosome", "Start", "End"]]
        .drop_duplicates()
        .copy()
    )
    genes["Start"] -= 1  # 0-based
    return genes

# ------------ BigWig 向量化均值 ------------
def _bw_mean_vectorized(bw_path: Path, chrom: str, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    with pyBigWig.open(str(bw_path)) as bw:
        means = np.empty(len(starts), dtype=np.float64)
        for i, (s, e) in enumerate(tqdm(zip(starts, ends), total=len(starts), desc=f"Reading {bw_path.name}")):
            try:
                vals = np.array(bw.values(chrom, s, e), dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                means[i] = vals.mean() if len(vals) else np.nan
            except Exception:
                means[i] = np.nan
    return means

def build_expression_matrix(pred_bw: Path, raw_bw: Path, gene_df: pd.DataFrame, chrom: str) -> pd.DataFrame:
    sub = gene_df.query("Chromosome == @chrom")
    pred_mean = _bw_mean_vectorized(pred_bw, chrom, sub["Start"].values, sub["End"].values)
    raw_mean = _bw_mean_vectorized(raw_bw, chrom, sub["Start"].values, sub["End"].values)
    return pd.DataFrame(
        {
            "gene_name": sub["gene_name"].values,
            "Predicted_Expression": pred_mean,
            "Raw_Expression": raw_mean,
        }
    )

def build_expression_matrix_ChromNameCh(pred_bw: Path, raw_bw: Path, gene_df: pd.DataFrame, chrom: str) -> pd.DataFrame:
    import re
    sub = gene_df.query("Chromosome == @chrom")
    chrom_num=re.findall(r'\d+', chrom)
    pred_mean = _bw_mean_vectorized(pred_bw, chrom_num[0], sub["Start"].values, sub["End"].values)
    raw_mean = _bw_mean_vectorized(raw_bw, chrom_num[0], sub["Start"].values, sub["End"].values)
    return pd.DataFrame(
        {
            "gene_name": sub["gene_name"].values,
            "Predicted_Expression": pred_mean,
            "Raw_Expression": raw_mean,
        }
    )

# ------------ 差异表达 ------------
def normalize_and_calculate_fc(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    # 检查必要列是否存在
    required_columns = ['gene_name', 'Predicted_Expression', 'Raw_Expression'] # 实际上只有Raw_Expression列存在0
    # 处理缺失值：将空值替换为0
    df[required_columns[1:]] = df[required_columns[1:]].fillna(0)
    
    pseudocount = 1e-6
    df[["Predicted_Expression", "Raw_Expression"]] += pseudocount

    # 标准化
    if method == "zscore":
        df[["Predicted_Expression", "Raw_Expression"]] = StandardScaler().fit_transform(
            df[["Predicted_Expression", "Raw_Expression"]]
        )
    elif method == "quantile":
        
        qt = QuantileTransformer(
            n_quantiles=min(1000, len(df)), random_state=42, output_distribution="normal"
        )
        df[["Predicted_Expression", "Raw_Expression"]] = qt.fit_transform(
            df[["Predicted_Expression", "Raw_Expression"]]
        )
    else:
        raise ValueError(method)

    # 右位移
    min_val = df[["Predicted_Expression", "Raw_Expression"]].min().min()
    # if min_val <= 0:
    df[["Predicted_Expression", "Raw_Expression"]] += abs(min_val) + pseudocount

    df["log2FC"] = np.log2(df["Predicted_Expression"] / df["Raw_Expression"])
    df["mean_expression"] = df[["Predicted_Expression", "Raw_Expression"]].mean(axis=1)
    df["Significant"] = (np.abs(df["log2FC"]) > 2) & (df["mean_expression"] > 1)
    return df

# ------------ 画图 ------------
def plot_volcano(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(12, 8))
    df["noise"] = np.random.normal(0, 0.1, len(df))
    sns.scatterplot(
        data=df, x="log2FC", y="noise", hue="Significant",
        palette={True: "#ff3333", False: "#999999"}, alpha=0.7, s=60
    )
    plt.axvline(x=2, ls="--", c="#333", alpha=0.5)
    plt.axvline(x=-2, ls="--", c="#333", alpha=0.5)
    plt.yticks([])
    plt.xlabel("log2(Fold Change)")
    plt.title("Volcano Plot")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    logging.info(f"Saved {out}")

def plot_ma(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, x="mean_expression", y="log2FC", hue="Significant",
        palette={True: "#ff3333", False: "#999999"}, alpha=0.7, s=60
    )
    plt.xscale("log")
    plt.axhline(y=2, ls="--", c="#333", alpha=0.5)
    plt.axhline(y=-2, ls="--", c="#333", alpha=0.5)
    plt.xlabel("Mean Expression (log10)")
    plt.ylabel("log2 Fold Change")
    plt.title("MA Plot")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    logging.info(f"Saved {out}")

def plot_heatmap(df: pd.DataFrame, out: Path, top_n: int = 50) -> None:
    top = (
        df.sort_values(by="log2FC", key=np.abs, ascending=False)
        .head(top_n)
        .groupby("gene_name")[["Predicted_Expression", "Raw_Expression"]]
        .mean()
    )
    plt.figure(figsize=(15, 10))
    sns.heatmap(top, cmap="coolwarm", linewidths=0.5, cbar_kws={"label": "Expression"})
    plt.title(f"Top {top_n} Differentially Expressed Genes")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    logging.info(f"Saved {out}")

# ------------ CLI & 统计 ------------
def main():
    parser = argparse.ArgumentParser(description="Gene expression & differential analysis")
    parser.add_argument("--pred_bw", type=Path, help="Predicted bigWig")
    parser.add_argument("--raw_bw", type=Path, help="Raw bigWig")
    parser.add_argument("--gtf_feather", type=Path, help="GTF feather")
    parser.add_argument("-o", "--out_dir", type=Path, default=Path("expr_result"))
    parser.add_argument("-c", "--chrom", default="chr19", help="Chromosome")
    parser.add_argument("--norm", choices=["zscore", "quantile"], default="zscore",
                        help="Normalization method (default: zscore)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取表达矩阵
    gene_df = load_gene_regions(args.gtf_feather, chrom=args.chrom)
    expr_df = build_expression_matrix_ChromNameCh(args.pred_bw, args.raw_bw, gene_df, chrom=args.chrom)
    expr_csv = args.out_dir / "expr_matrix.csv"
    expr_df.to_csv(expr_csv, index=False)
    logging.info(f"Expression matrix -> {expr_csv}")

    # 2. 差异表达
    de_df = normalize_and_calculate_fc(expr_df, method=args.norm)
    all_csv = args.out_dir / "differential_expression_results_all.csv"
    de_df.to_csv(all_csv, index=False)

    sig_df = de_df[de_df["Significant"]].sort_values(
        by="log2FC", key=np.abs, ascending=False
    )
    sig_csv = args.out_dir / "differential_expression_results_significant.csv"
    sig_df.to_csv(sig_csv, index=False)

    #  3. 统计 + 相关系数 
    stats_txt = args.out_dir / "gene-level_stats.txt"
    with stats_txt.open("w") as f:
        # 原始 log2FC 统计
        total_genes = len(de_df)
        total_genes_1 = de_df[de_df["mean_expression"] > 1].shape[0]
        up2   = de_df[(de_df["log2FC"] > 2)  & (de_df["mean_expression"] > 1)].shape[0]
        down2 = de_df[(de_df["log2FC"] < -2) & (de_df["mean_expression"] > 1)].shape[0]

        f.write("Only consider those mean expression > 1.\n")
        f.write(f"Total genes: {total_genes} -> {total_genes_1}\n")
        f.write(f"Upregulated (log2FC > 2): {up2}\n")
        f.write(f"Downregulated (log2FC < -2): {down2}\n")

        # 计算相关系数（无标准化）
        corr_df = expr_df[["Predicted_Expression", "Raw_Expression"]].dropna()
        if len(corr_df) < 2:
            f.write("Too few genes to compute correlations.\n")
        else:
            from scipy.stats import pearsonr, spearmanr
            pear_r,  _ = pearsonr(corr_df["Predicted_Expression"], corr_df["Raw_Expression"])
            spear_r, _ = spearmanr(corr_df["Predicted_Expression"], corr_df["Raw_Expression"])
            log_pred = np.log1p(corr_df["Predicted_Expression"])
            log_raw  = np.log1p(corr_df["Raw_Expression"])
            log_pear_r, _ = pearsonr(log_pred, log_raw)

            f.write("Gene-level correlation (no normalization)\n")
            f.write(f"Pearson (raw): {pear_r:.4f}\n")
            f.write(f"Spearman (raw): {spear_r:.4f}\n")
            f.write(f"Pearson (log1p): {log_pear_r:.4f}\n")

    logging.info(f"Stats & correlations saved -> {stats_txt}")

    # 4. 画图 
    plot_volcano(de_df, args.out_dir / "volcano_plot.png")
    plot_ma(de_df, args.out_dir / "ma_plot.png")
    plot_heatmap(de_df, args.out_dir / "heatmap.png")

    logging.info("All finished!")

if __name__ == "__main__":
    main()
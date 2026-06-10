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
import numpy as np
import json
from collections import defaultdict

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

def str2arr(s: str):
    """把 "[a,b,c]" 字符串 → [a,b,c] 列表"""
    try:
        return json.loads(s.strip())
    except Exception as e:
        raise ValueError(f"无法解析表达式字符串: {s}, 错误: {e}")

# ------------ 读取 GTF ------------
def load_gtf_file(gtf_path: Union[str, Path], feature_type: str = "gene", 
                  attributes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    直接读取GTF文件并解析为DataFrame
    
    参数:
    gtf_path: GTF文件路径（支持.gz压缩格式）
    feature_type: 要筛选的特征类型，如"gene", "exon", "transcript"等
    attributes: 要提取的属性列表，如["gene_id", "gene_name", "gene_biotype"]
                如果为None，则提取所有属性
    
    返回:
    包含GTF数据的DataFrame
    """
    gtf_path = Path(gtf_path)
    
    lines = []
    with open(gtf_path, 'r') as f:
        lines = f.readlines()
    data_lines = [line.strip().split('\t') for line in lines if not line.startswith('#')]

    # GTF文件列名
    gtf_columns = [
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ]

    gtf_df = pd.DataFrame(data_lines, columns=gtf_columns)
    gtf_df["start"] = pd.to_numeric(gtf_df["start"], errors='coerce')
    gtf_df["end"] = pd.to_numeric(gtf_df["end"], errors='coerce')

    if feature_type:
        gtf_df = gtf_df[gtf_df["feature"] == feature_type].copy()

    def parse_attributes(attr_string):
        """解析GTF属性字符串为字典"""
        attr_dict = {}
        if pd.isna(attr_string):
            return attr_dict
        
        # 分割属性对
        attributes_list = attr_string.strip(';').split(';')
        
        for attr in attributes_list:
            attr = attr.strip()
            if not attr:
                continue
                
            # 分割键值对
            if ' ' in attr:
                key, value = attr.split(' ', 1)
                # 去掉值中的引号
                value = value.strip().strip('"')
                attr_dict[key] = value
        
        return attr_dict

    parsed_attrs = gtf_df["attribute"].apply(parse_attributes)

    # 提取指定的属性作为新列
    if attributes is None:
        # 提取所有唯一的属性键
        all_attr_keys = set()
        for attr_dict in parsed_attrs:
            all_attr_keys.update(attr_dict.keys())
        attributes = list(all_attr_keys)

    for attr_key in attributes:
        gtf_df[attr_key] = parsed_attrs.apply(lambda x: x.get(attr_key, None))

    # 重命名列以保持兼容性
    column_mapping = {
        "seqname": "Chromosome",
        "start": "Start",
        "end": "End",
        "strand": "Strand"
    }
    
    # 应用列重命名
    gtf_df.rename(columns={k: v for k, v in column_mapping.items() if k in gtf_df.columns}, 
                  inplace=True)
    
    print(f"✅ 成功加载 {len(gtf_df)} 行数据")
    return gtf_df

# 可选：如果你需要更高效的计算方式（向量化版本）
def build_expression_matrix_ChromNameCh_vectorized(
    pred_dict: dict, 
    raw_dict: dict, 
    gene_df: pd.DataFrame, 
    chrom_start_pos: int = 0
) -> pd.DataFrame:
    """
    向量化版本，更高效
    """
    # 筛选指定染色体的基因
    sub = gene_df
    
    if len(sub) == 0:
        return pd.DataFrame(columns=["transcript_id", "Predicted_Expression", "Raw_Expression"])
    
    # 获取基因区间
    starts = sub["Start"].values - chrom_start_pos
    ends = sub["End"].values - chrom_start_pos
    chrom =sub["Chromosome"].values
    
    # 计算每个区间的平均值
    def compute_means(dict, chrom, starts, ends):
        means = []
        for chrom, start, end in zip(chrom, starts, ends):
            if start < end:
                means.append(np.mean(dict[chrom][start:end]))
            else:
                means.append(np.nan)
        return np.array(means)

    pred_mean = compute_means(pred_dict, chrom, starts, ends)
    raw_mean = compute_means(raw_dict, chrom, starts, ends)

    return pd.DataFrame(
        {
            "gene_name": sub["transcript_id"].values,
            "Predicted_Expression": pred_mean,
            "Raw_Expression": raw_mean,
        }
    )

def df_value_extract(df, expression_parsed_col):
    print("📌 步骤3: 累加重叠位置的表达值...")
    pos_data = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # pos_data[chrom][pos] = [sum, count]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理每一行"):
        chrom = row['chromosome']
        start = int(row['start'])
        end = int(row['end'])
        values = list(map(float, row[expression_parsed_col]))
        # 确保长度一致（再次确认）
        if len(values) != (end - start):
            continue  # 已过滤，但保险起见

        for j, val in enumerate(values):
            current_pos = start + j
            pos_data[chrom][current_pos][0] += val  # 累加
            pos_data[chrom][current_pos][1] += 1    # 计数
    
    averages = {}
    for chrom in sorted(pos_data.keys()):
        chrom_data = pos_data[chrom]
        sorted_positions = sorted(chrom_data.keys())
        averages[chrom] = [s / c for s, c in (chrom_data[pos] for pos in sorted_positions)]
    

    return averages

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
    parser.add_argument("--pred_csv", type=Path, help="Predicted CSV")
    parser.add_argument("--gtf_file", type=Path, help="GTF file path")
    parser.add_argument("--genes", type=Path, help="gene list (one gene per line)", default=None)
    parser.add_argument("-o", "--out_dir", type=Path, default=Path("expr_result"))
    parser.add_argument("-c", "--chrom", default="chr19", help="Chromosome")
    parser.add_argument("--norm", choices=["zscore", "quantile"], default="zscore",
                        help="Normalization method (default: zscore)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取表达矩阵
    gene_df = load_gtf_file(args.gtf_file, feature_type="transcript", attributes=["transcript_id"])
    if args.genes:
        gene_list = pd.read_csv(args.genes, header=1, names=["gene_name"])
        gene_df = gene_df[gene_df["transcript_id"].isin(gene_list["gene_name"])].copy()
        logging.info(f"Filtered genes: {len(gene_df)}")       

    df_pred = pd.read_csv(args.pred_csv)
    df_pred['parsed_expression'] = df_pred["predicted_expression"].apply(str2arr)
    y_prep_dict = df_value_extract(df_pred, "parsed_expression")
    
    df_raw = pd.read_csv(args.pred_csv)
    df_raw['parsed_expression'] = df_pred["true_expression"].apply(str2arr)
    y_raw_dict = df_value_extract(df_raw, "parsed_expression")
   
    expr_df = build_expression_matrix_ChromNameCh_vectorized(y_prep_dict, y_raw_dict, gene_df)


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
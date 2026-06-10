import pandas as pd
from typing import Tuple, List

def load_significant_genes(csv_path: str, gene_col: str = 'gene_name') -> List[str]:
    """
    从 CSV 文件中读取显著差异表达基因列表。
    
    Args:
        csv_path (str): CSV 文件路径
        gene_col (str): 基因名所在的列名，默认为 'gene_name'
    
    Returns:
        List[str]: 基因名列表
    
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果指定的列不存在
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"文件未找到: {csv_path}") from e
    except Exception as e:
        raise Exception(f"读取 CSV 文件失败: {csv_path}, 错误: {e}") from e

    if gene_col not in df.columns:
        raise ValueError(f"列 '{gene_col}' 不存在于文件中。可用列: {list(df.columns)}")

    genes = df[gene_col].dropna().astype(str).tolist()
    if not genes:
        print(f"警告: 文件 {csv_path} 中 '{gene_col}' 列无有效基因名。")
    return genes


def calculate_recall_precision(
    true_genes: List[str], 
    predicted_genes: List[str]
) -> Tuple[float, float]:
    """
    计算 Recall 和 Precision。
    
    Args:
        true_genes (List[str]): 真实差异基因列表
        predicted_genes (List[str]): 预测差异基因列表
    
    Returns:
        Tuple[float, float]: (Recall, Precision)
    """
    if not true_genes:
        raise ValueError("真实基因列表为空，无法计算 Recall。")
    if not predicted_genes:
        raise ValueError("预测基因列表为空，无法计算 Precision。")

    true_set = set(true_genes)
    pred_set = set(predicted_genes)

    tp = len(true_set & pred_set)  # 交集：真正例
    recall = tp / len(true_set)
    precision = tp / len(pred_set)

    return recall, precision


def main():
    """主函数：计算预测结果相对于真实值的 Recall 和 Precision。"""
    # --- 配置参数 ---
    GT_CSV_PATH = "/mnt/zzbnew/peixunban/yecheng/workspace/Geno2Transcript/output/133vs27/GT/gene-level_stats/differential_expression_results_significant.csv"
    PREDICT_CSV_PATH = "/mnt/zzbnew/peixunban/yecheng/data/pred_track_bw_1013_/ljy_10b_frozen_1013_133vs27/gene-level_stats/differential_expression_results_significant.csv"
    GENE_COLUMN = "gene_name"
    # ----------------

    try:
        # 加载真实和预测的差异基因
        true_diff_genes = load_significant_genes(GT_CSV_PATH, GENE_COLUMN)
        predict_diff_genes = load_significant_genes(PREDICT_CSV_PATH, GENE_COLUMN)

        # 计算指标
        recall, precision = calculate_recall_precision(true_diff_genes, predict_diff_genes)

        # 输出结果
        print(f"真实差异基因数: {len(true_diff_genes)}")
        print(f"预测差异基因数: {len(predict_diff_genes)}")
        print(f"共同基因数 (TP): {len(set(true_diff_genes) & set(predict_diff_genes))}")
        print(f"Recall:  {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        raise


if __name__ == "__main__":
    main()
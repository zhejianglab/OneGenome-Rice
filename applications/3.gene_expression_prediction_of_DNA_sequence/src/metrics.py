# 标准库（内置模块）
import numpy as np

# 第三方库（pip 安装的包）
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score, 
    average_precision_score,
    r2_score
)
from scipy.stats import pearsonr, spearmanr
from transformers import EvalPrediction
from src.util import dist_print

def evaluate_zero_inflated(y_true, y_pred, epsilon=1e-8):
    """
    计算零膨胀回归任务的评测指标（含非零区域的相关系数 和 样本级平均指标）

    参数:
    y_true: list of arrays 或 2D array, 形状为 [N, L]，每个样本长度 L
    y_pred: list of arrays 或 2D array, 形状为 [N, L]
    epsilon: 判断预测为0的阈值（预测值 < epsilon 视为预测0）

    返回:
    metrics: 包含所有指标的字典
    """
    # 转换输入为 list of arrays（确保结构清晰）
    if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
        y_true = [y_true[i] for i in range(len(y_true))]
    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
        y_pred = [y_pred[i] for i in range(len(y_pred))]

    # 展平用于全局指标计算
    y_true_flatten = np.concatenate(y_true)
    y_pred_flatten = np.concatenate(y_pred)

    if len(y_true_flatten) != len(y_pred_flatten):
        raise ValueError("y_true_flatten 和 y_pred_flatten 长度必须一致")

    # --------------------------
    # 1. 零值判断与基础准备
    # --------------------------
    true_zero = (y_true_flatten == 0)
    true_nonzero = (y_true_flatten != 0)

    # --------------------------
    # 2. 整体误差指标（Global）
    # --------------------------
    mse = mean_squared_error(y_true_flatten, y_pred_flatten)
    mae = mean_absolute_error(y_true_flatten, y_pred_flatten)
    r2 = r2_score(y_true_flatten, y_pred_flatten)
    pearson, _ = pearsonr(y_true_flatten, y_pred_flatten)
    try:
        log1p_pearson, _ = pearsonr(np.log1p(y_true_flatten), np.log1p(y_pred_flatten))
    except:
        log1p_pearson = np.nan
    spearman, _ = spearmanr(y_true_flatten, y_pred_flatten)

    # --------------------------
    # 3. 零值识别指标
    # --------------------------
    try:
        auroc_zero = roc_auc_score(true_zero, -y_pred_flatten)
        auprc_zero = average_precision_score(true_zero, -y_pred_flatten)
        auprc_nonzero = average_precision_score(true_nonzero, y_pred_flatten)
    except ValueError as e:
        print(f"无法计算 AUROC/AUPRC: {e}")
        auroc_zero = np.nan
        auprc_zero = np.nan
        auprc_nonzero = np.nan

    # --------------------------
    # 4. 非零区域指标
    # --------------------------
    non_zero_mask = ~true_zero
    y_true_nonzero = y_true_flatten[non_zero_mask]
    y_pred_nonzero = y_pred_flatten[non_zero_mask]
    n_nonzero = len(y_true_nonzero)

    if n_nonzero < 2:
        non_zero_mse = np.nan
        non_zero_mae = np.nan
        non_zero_pearson = np.nan
        non_zero_spearman = np.nan
        non_zero_log1p_pearson = np.nan
    else:
        non_zero_mse = mean_squared_error(y_true_nonzero, y_pred_nonzero)
        non_zero_mae = mean_absolute_error(y_true_nonzero, y_pred_nonzero)
        non_zero_pearson, _ = pearsonr(y_true_nonzero, y_pred_nonzero)
        non_zero_spearman, _ = spearmanr(y_true_nonzero, y_pred_nonzero)
        try:
            non_zero_log1p_pearson, _ = pearsonr(np.log1p(y_true_nonzero), np.log1p(y_pred_nonzero))
        except:
            non_zero_log1p_pearson = np.nan

    # --------------------------
    # 5. 样本级平均指标 (Per-sample Mean)
    # --------------------------
    sample_mses = []
    sample_maes = []
    sample_r2s = []
    # sample_pearsons = []
    # sample_log1p_pearsons = []
    # sample_spearmans = []

    for yt, yp in zip(y_true, y_pred):
        if len(yt) < 2:
            continue

        # 基础指标
        sample_mses.append(mean_squared_error(yt, yp))
        sample_maes.append(mean_absolute_error(yt, yp))
        # sample_r2s.append(r2_score(yt, yp))

        # # 相关系数（需要至少2个点）
        # if len(yt) >= 2:
        #     try:
        #         pcc, _ = pearsonr(yt, yp)
        #         sample_pearsons.append(pcc)
        #     except:
        #         pass

        #     try:
        #         spc, _ = spearmanr(yt, yp)
        #         sample_spearmans.append(spc)
        #     except:
        #         pass

        #     try:
        #         log1p_pcc, _ = pearsonr(np.log1p(yt), np.log1p(yp))
        #         sample_log1p_pearsons.append(log1p_pcc)
        #     except:
        #         pass

    # 取平均（注意：可能为空）
    sample_mean_mse = np.mean(sample_mses) if sample_mses else np.nan
    sample_mean_mae = np.mean(sample_maes) if sample_maes else np.nan
    # sample_mean_r2 = np.mean(sample_r2s) if sample_r2s else np.nan
    # sample_mean_pearson = np.mean(sample_pearsons) if sample_pearsons else np.nan
    # sample_mean_spearman = np.mean(sample_spearmans) if sample_spearmans else np.nan
    # sample_mean_log1p_pearson = np.mean(sample_log1p_pearsons) if sample_log1p_pearsons else np.nan

    # --------------------------
    # 6. 整理所有指标
    # --------------------------
    metrics = {
        # 整体指标（Global）
        "mse": round(float(mse), 6),
        "mae": round(float(mae), 6),
        "r2_score": round(float(r2), 6),
        "pearson": round(float(pearson), 6),
        "log1p_pearson": round(float(log1p_pearson), 6),
        "spearman": round(float(spearman), 6),

        # 零值识别指标
        "zero_auroc": round(float(auroc_zero), 6),
        "zero_auprc": round(float(auprc_zero), 6),
        "nonzero_auprc": round(float(auprc_nonzero), 6),

        # 非零区域指标
        "nonzero_mse": round(float(non_zero_mse), 6),
        "eval_nonzero_mae": round(float(non_zero_mae), 6),
        "nonzero_pearson": round(float(non_zero_pearson), 6),
        "nonzero_log1p_pearson": round(float(non_zero_log1p_pearson), 6),
        "nonzero_spearman": round(float(non_zero_spearman), 6),

        # 样本平均指标（Per-sample Mean）
        "sample_mean_mse": round(float(sample_mean_mse), 6),
        "sample_mean_mae": round(float(sample_mean_mae), 6),
        # "sample_mean_r2_score": round(float(sample_mean_r2), 6),
        # "sample_mean_pearson": round(float(sample_mean_pearson), 6),
        # "sample_mean_log1p_pearson": round(float(sample_mean_log1p_pearson), 6),
        # "sample_mean_spearman": round(float(sample_mean_spearman), 6),

        # 辅助信息
        "zero_ratio": round(float(np.mean(true_zero) * 100), 4),
    }

    return metrics


# def compute_metrics(eval_pred: EvalPrediction, val_chromosomes, tokenizer):
#     print(val_chromosomes)

#     preds = eval_pred.predictions  # [B, L]
#     labels = eval_pred.label_ids   # [B, L]
#     input_ids = eval_pred.inputs    # [B, L+2]
    
#     input_chroms = [tokenizer.convert_ids_to_tokens([seq[0]])[0][1:-1] for seq in input_ids]
#     input_track_index = [tokenizer.convert_ids_to_tokens([seq[1]])[0][1:-1].split(":")[-1] for seq in input_ids]

#     # 对每个样本去头去尾 [100:-100]
#     preds_trimmed = [np.asarray(seq, dtype=np.float32)[100:-100] for seq in preds]
#     labels_trimmed = [np.asarray(seq, dtype=np.float32)[100:-100] for seq in labels]

#     # 不分组
#     if val_chromosomes is None:
#         return evaluate_zero_inflated(labels_trimmed, preds_trimmed)
    
#     final_metrics = {}

#     # 构建 (chrom, track_index) -> indices 的映射
#     group_map = {}
#     for i, (chrom, track_idx) in enumerate(zip(input_chroms, input_track_index)):
#         if chrom not in val_chromosomes:
#             continue
#         key = (chrom, track_idx)
#         if key not in group_map:
#             group_map[key] = []
#         group_map[key].append(i)

#     # 遍历每个 (chrom, track_index) 组合
#     for (chrom, track_idx), indices in group_map.items():
#         if not indices:
#             continue

#         group_preds = [preds_trimmed[i] for i in indices]
#         group_labels = [labels_trimmed[i] for i in indices]

#         group_metrics = evaluate_zero_inflated(group_labels, group_preds)

#         for k, v in group_metrics.items():
#             final_metrics[f"{chrom}_track{track_idx}_{k}"] = v

#     dist_print(final_metrics)
#     return final_metrics


def compute_multimodal_metrics(eval_pred: EvalPrediction, val_chromosomes, tokenizer):
    """
    计算多模态预测的评估指标
    
    Args:
        eval_pred: EvalPrediction对象，包含predictions, label_ids和inputs
        val_chromosomes: 验证染色体列表
        tokenizer: 分词器
        
    Returns:
        dict: 包含各模态评估指标的字典
    """
    # print(val_chromosomes)

    # 解析模型输出和标签
    preds = eval_pred.predictions  # 模型预测输出，形状为 [B, L, 3]
    labels = eval_pred.label_ids   # 真实标签，形状为 [B, L, 3]
    input_ids = eval_pred.inputs   # 输入ID

    # 从input_ids提取染色体和biosample信息
    input_chroms = [tokenizer.convert_ids_to_tokens([seq[0]])[0][1:-1] for seq in input_ids]
    input_biosample = [tokenizer.convert_ids_to_tokens([seq[1]])[0][1:-1] for seq in input_ids]
    
    # 固定模态名称
    modalities = ["total_RNA-seq_+", "total_RNA-seq_-", "ATAC-seq_."]
    
    # 存储各模态的结果
    modality_preds = {modality: [] for modality in modalities}
    modality_labels = {modality: [] for modality in modalities}
    
    # 处理预测结果，将 [B, L, 3] 张量拆分为三个模态
    for i in range(len(modalities)):
        # 对每个样本去头去尾 [100:-100]
        modality_preds[modalities[i]] = [np.asarray(seq[:, i].cpu() if hasattr(seq, 'cpu') else seq[:, i], dtype=np.float32)[100:-100] 
                                       for seq in preds]
    
    # 处理标签数据
    for i in range(len(modalities)):
        # 对每个样本去头去尾 [100:-100]
        modality_labels[modalities[i]] = [np.asarray(seq[:, i].cpu() if hasattr(seq, 'cpu') else seq[:, i], dtype=np.float32)[100:-100] 
                                        for seq in labels]

    # 分组计算指标
    final_metrics = {}

    # 构建 (biosample, chrom) -> indices 的映射
    group_map = {}
    for i, (chrom, biosample) in enumerate(zip(input_chroms, input_biosample)):
        if chrom not in val_chromosomes:
            continue
        key = (biosample, chrom)
        if key not in group_map:
            group_map[key] = []
        group_map[key].append(i)

    # 遍历每个 (biosample, chrom) 组合
    for (biosample, chrom), indices in group_map.items():
        if not indices:
            continue

        # 为每个模态计算指标
        for modality in modalities:
            # 提取该组的预测和标签
            group_preds = [modality_preds[modality][i] for i in indices]
            group_labels = [modality_labels[modality][i] for i in indices]

            # 使用evaluate_zero_inflated计算指标
            group_metrics = evaluate_zero_inflated(group_labels, group_preds)

            # 为指标添加组和模态前缀，按照 biosample/modality/chrom/k 的格式
            for k, v in group_metrics.items():
                final_metrics[f"{biosample}/{modality}/{chrom}/{k}"] = v

    dist_print(final_metrics)
    return final_metrics
#!/usr/bin/env bash
set -euo pipefail
#-------------------------------------for train data evaluation-------------------------------------#
# 配置
SCHROM=$1
CHROM_SIZES="/mnt/zzb/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref/P4-new.chrom.sizes"
EXPR_COL="predicted_expression"
CSV_DIR="outputs/0407-${SCHROM}-traindata"
OUT_DIR="$CSV_DIR"
METRICS_SCRIPT="scripts/evaluation/calc_metrics_for_batch_bw2.py"
DIFF_SCRIPT="scripts/evaluation/diff_exp_analysis_for_single_bw.py"
FASTA="/mnt/zzb/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref/P4-new.fasta"
GTF_FEATHER="data/external/osa1_r7.all_models.gff3.feather"
CHROM="${SCHROM}"

# 1) CSV -> npy（批量）
shopt -s nullglob
csvs=("$CSV_DIR"/*predictions.csv)
if [ ${#csvs[@]} -eq 0 ]; then
    echo "未找到 CSV 文件：$CSV_DIR"
    exit 0
fi

for csv in "${csvs[@]}"; do
    out_txt="${csv%.csv}"
    echo "Convert: $csv -> $out_txt"
    python scripts/evaluation/csv2bw2.py \
        --csv "$csv" \
        --output "$out_txt".bw \
        --chrom_sizes "$CHROM_SIZES" \
        --expression_col "$EXPR_COL"
done

# 2) 碱基级指标计算（按原始脚本的映射精简为列表）
# metrics_tasks=(
# "$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions.bw|$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions_track-level_stats.txt|data/processed/renorm_bigwig_output/re-normalized_Z_P14_1.forward.bw"
# "$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions.bw|$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions_track-level_stats.txt|data/processed/renorm_bigwig_output/re-normalized_Z_P14_1.reverse.bw "
# )
metrics_tasks=(
"$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions.bw.npy|$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions_track-level_stats.txt|$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions.bw_true.npy"
"$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions.bw.npy|$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions_track-level_stats.txt|$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions.bw_true.npy"
)

echo "Run base-level metrics..."
for task in "${metrics_tasks[@]}"; do
    IFS='|' read -r pred_bw out_txt raw_bw <<< "$task"
    if [ ! -f "$pred_bw" ]; then
        echo "跳过（未找到预测 bw）: $pred_bw"
        continue
    fi
    echo "Metrics: $pred_bw vs $raw_bw -> $out_txt"
    # mkdir -p "$(dirname "$out_txt")"
    python "$METRICS_SCRIPT" \
        --pred_files "$pred_bw" \
        --raw_files "$raw_bw" \
        --output "$out_txt" \
        --fasta "$FASTA" \
        --chrom "$CHROM"
done

python ../RiceModel3-SFT-track_prediction/scripts/evaluation/csv_seg_eval.py \
 --csv "$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions.csv"\
 --output $OUT_DIR/minus --skip_bigwig \
 --chrom_sizes "$CHROM_SIZES" \
 --expression_col "predicted_expression"

python ../RiceModel3-SFT-track_prediction/scripts/evaluation/csv_seg_eval.py \
 --csv "$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions.csv"\
 --output $OUT_DIR/plus --skip_bigwig \
 --chrom_sizes "$CHROM_SIZES" \
 --expression_col "predicted_expression"

#-------------------------------------for test data evaluation-------------------------------------#

# 配置
CHROM_SIZES="/mnt/zzb/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref/P7-new.chrom.sizes"
EXPR_COL="predicted_expression"
CSV_DIR="outputs/0407-${SCHROM}"
OUT_DIR="$CSV_DIR"
METRICS_SCRIPT="scripts/evaluation/calc_metrics_for_batch_bw3.py"
DIFF_SCRIPT="scripts/evaluation/diff_exp_analysis_for_single_csv.py"
FASTA="/mnt/zzb/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref/P7-new.fasta"
GTF_FILE="/mnt/zzb/default/Workspace/Rice-Genome/application/RNAseq/riceRNAseqData/18k/ref/P7_EVM.all.gtf"
CHROM=${SCHROM}

# # 1) CSV -> npy（批量）
shopt -s nullglob
csvs=("$CSV_DIR"/*.csv)
if [ ${#csvs[@]} -eq 0 ]; then
     echo "未找到 CSV 文件：$CSV_DIR"
     exit 0
fi

for csv in "${csvs[@]}"; do
     out_txt="${csv%.csv}"
     echo "Convert: $csv -> $out_txt"
     python scripts/evaluation/csv2bw2.py \
         --csv "$csv" \
         --output "$out_txt".bw \
         --chrom_sizes "$CHROM_SIZES" \
         --expression_col "$EXPR_COL"
done

# 2) 碱基级指标计算（按原始脚本的映射精简为列表）
metrics_tasks=(
"$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions.bw.npy|$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions_track-level_stats.txt|data/processed/renorm_bigwig_output/CSQ_P7_1.bw"
"$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions.bw.npy|$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions_track-level_stats.txt|data/processed/renorm_bigwig_output/YG_P7_1.bw "
)
# metrics_tasks=(
# "$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions.bw.npy|$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions_track-level_stats.txt|$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions.bw_true.npy"
# "$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions.bw.npy|$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions_track-level_stats.txt|$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions.bw_true.npy"
# )

echo "Run base-level metrics..."
for task in "${metrics_tasks[@]}"; do
    IFS='|' read -r pred_bw out_txt raw_bw <<< "$task"
    if [ ! -f "$pred_bw" ]; then
        echo "跳过（未找到预测 bw）: $pred_bw"
        continue
    fi
    echo "Metrics: $pred_bw vs $raw_bw -> $out_txt"
    # mkdir -p "$(dirname "$out_txt")"
    python "$METRICS_SCRIPT" \
        --pred_files "$pred_bw" \
        --raw_files "$raw_bw" \
        --output "$out_txt" \
        --fasta "$FASTA" \
        --chrom "$CHROM"
done

python scripts/evaluation/csv_seg_eval2.py \
 --csv "$OUT_DIR/biosample__Panicle1_all_RNAseq-_predictions.csv"\
 --bw "data/processed/renorm_bigwig_output/CSQ_P7_1.bw" \
 --fasta "$FASTA" \
 --output $OUT_DIR/minus --skip_bigwig \
 --chrom_sizes "$CHROM_SIZES" \
 --expression_col "predicted_expression"

python scripts/evaluation/csv_seg_eval2.py \
 --csv "$OUT_DIR/biosample__Panicle1_all_RNAseq+_predictions.csv"\
 --bw "data/processed/renorm_bigwig_output/YG_P7_1.bw" \
 --fasta "$FASTA" \
 --output $OUT_DIR/plus --skip_bigwig \
 --chrom_sizes "$CHROM_SIZES" \
 --expression_col "predicted_expression"



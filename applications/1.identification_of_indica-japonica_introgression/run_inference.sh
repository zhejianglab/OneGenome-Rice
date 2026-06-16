#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME="env_introgression_analysis"

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

export MNT_DATA="/mnt/rice/data"
export MNT_DEFAULT="/mnt/rice/default"

cd ${MNT_DEFAULT}/Workspace/OneGenome-Rice/applications/1.identification_of_indica-japonica_introgression

export DATASET_NAME="varieties_classification_YF47_100bp_100bp_20260604"
export RUN_NAME="test_run_$(date +%Y%m%d_%H%M%S)"

export CONFIG_FILE="configs/config_tuning.yaml"
export CHECKPOINT_DIR="${MNT_DEFAULT}/Workspace/OneGenome-Rice/applications/1.identification_of_indica-japonica_introgression/results/rice_1B_stage2_8k_hf/varieties_classification_jap25-ind25_8k_7k_20260529/1_lossRatio-0.1_lr-cosine_batchSize-4_warmupR-0.05_epoch-4_noNormalize/checkpoints/checkpoint-26500"

# export CUDA_VISIBLE_DEVICES="0"
# 使用单卡推理
# python scripts/run_inference.py \
#     --config ${CONFIG_FILE} \
#     --checkpoint ${CHECKPOINT_DIR}

# 使用accelerate.launch进行多卡推理
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
accelerate launch --num_processes=8 scripts/run_inference.py \
    --config ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_DIR}
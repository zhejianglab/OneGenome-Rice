#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME="env_introgression_analysis"

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

export MNT_DATA="/mnt/rice/data"
export MNT_DEFAULT="/mnt/rice/default"

cd ${MNT_DEFAULT}/Workspace/OneGenome-Rice/applications/1.identification_of_indica-japonica_introgression

export DATASET_NAME="varieties_classification_jap25-ind25_8k_7k_20260529"
export RUN_NAME="1_lossRatio-0.1_lr-cosine_batchSize-4_warmupR-0.05_epoch-4_noNormalize"

export CONFIG_FILE="configs/config_tuning.yaml"
# 使用单卡训练
# python scripts/run_train.py \
#     --config "${CONFIG_FILE}"
# 使用torchrun进行多卡训练
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
torchrun --nproc_per_node=8 scripts/run_train.py \
    --config "${CONFIG_FILE}"
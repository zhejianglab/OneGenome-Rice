#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME="env_introgression_analysis"

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

export MNT_DATA="/mnt/rice/data"
export MNT_DEFAULT="/mnt/rice/default"

cd ${MNT_DEFAULT}/Workspace/OneGenome-Rice/applications/1.identification_of_indica-japonica_introgression

export DATASET_NAME="varieties_classification_YF47_100bp_100bp_20260604"
export RUN_NAME="default"

export CONFIG_FILE="configs/config_tuning.yaml"
# 模式选择：train，eval，test，或组合
export SPLIT_MODE="train,eval,test"
python scripts/run_tokenize.py \
    --config "${CONFIG_FILE}" \
    --split-mode "${SPLIT_MODE}"
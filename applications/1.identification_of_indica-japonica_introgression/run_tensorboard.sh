#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME="env_introgression_analysis"

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

export RESULT_DIR="/mnt/rice/default/Workspace/OneGenome-Rice/applications/1.identification_of_indica-japonica_introgression/results/rice_1B_stage2_8k_hf"

tensorboard --logdir ${RESULT_DIR} --port 6006
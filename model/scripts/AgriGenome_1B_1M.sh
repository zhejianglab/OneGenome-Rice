#!/bin/bash

# Runs AgriGenome model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_MODE=offline
export WANDB_API_KEY=

# ================begin to edit================
export WANDB_NAME=AgriGenome_1B_1M
SEQ_LENGTH=1048576
MAX_LENGTH=1048576
TRAIN_SAMPLES=40191535
LAST_TRAIN_SAMPLES=40128485
LR_DECAY_SAMPLES=$(((TRAIN_SAMPLES - LAST_TRAIN_SAMPLES) * 80 / 100))
CHECKPOINT_PATH=/mnt/rice/AgriGenome_1B_1M/$WANDB_NAME
# ================end================

TOKENIZER_TYPE=SentencePieceTokenizer
TOKENIZER_MODEL=/mnt/rice/tokenizer/one_hot.bpe.model
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024




DISTRIBUTED_ARGS=" \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=8 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

MODEL_ARGS=" \
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_LENGTH \
    --num-layers 12 \
    --hidden-size 1024 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --no-position-embedding \
    --rotary-base 50000000"



MOE_ARGS=" \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-3 \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --moe-router-dtype fp32 \
    --moe-z-loss-coeff 1e-3 \
    --moe-permute-fusion"

DATA_ARGS=" \
    --num-workers 8 \
    --dataloader-type cyclic \
    --tokenizer-type ${TOKENIZER_TYPE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path /mnt/rice/1M/positive/13-65_1m_text_document 
    --split 1000,0,0 \
    --loss-mask-tokens N \
    --no-create-attention-mask-in-dataloader"

TRAINING_ARGS=" \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 9.7e-5 \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-decay-style cosine \
    --min-lr 9.7e-6 \
    --weight-decay 0.1 \
    --lr-warmup-fraction 0.05 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --disable-bf16-reduced-precision-matmul"



MODEL_PARALLEL_ARGS=" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --sequence-parallel \
    --context-parallel-size 16 \
    --use-distributed-optimizer"





LOGGING_ARGS=" \
    --log-interval 1 \
    --save-interval 30 \
    --eval-interval 50000000 \
    --eval-iters 0 \
    --save $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --wandb-project ${WANDB_PROJECT:-"AgriGenome"} \
    --wandb-exp-name ${WANDB_NAME:-"AgriGenome-1B"} \
    --moe-per-layer-logging \
    --no-load-optim \
    --no-load-rng \
    --load /mnt/rice/AgriGenome_1B_128k \
    --log-throughput"





torchrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    ${MODEL_ARGS} \
    ${MOE_ARGS} \
    ${DATA_ARGS} \
    ${TRAINING_ARGS} \
    ${MODEL_PARALLEL_ARGS} \
    ${LOGGING_ARGS}

#!/bin/bash

# Runs Mixtral 8x7B model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NVTE_COMM_OVERLAP=0
export NCCL_P2P_DISABLE=1
export NCCL_P2P_DIRECT_DISABLE=1
OUTPUT_PATH=output
# TOKENIZER_MODEL=$2
# DATA_PATH=$3

#DATASET_PATH=/mnt/zzb/Public/DataSet/human_genome_assemblies/demo_data/32k/HG00673.hap1.phase1.gene_10k.32k_text_document
#VALID_DATASET_PATH=/mnt/zzb/Public/DataSet/human_genome_assemblies/demo_data/32k/HG00673.hap1.phase1.gene_10k.32k_text_document
#TOKENIZER_PATH=/mnt/zzb/Public/DataSet/human_genome_assemblies/demo_data/one-hot

#DATA_PATH=/mnt/zzb/peixunban/liujunchen/zzb2/all_data/txj-0627-0/data/HG00658.gene_10k.8k.chr1_22_XY_text_document
#TOKENIZER_PATH=/mnt/zzb/peixunban/liujunchen/zzb2/all_data/txj-0627-0/tokenizer/


DATA_PATH=/mnt/data/users/datasets/NA21093.hap1_text_document
TOKENIZER_PATH=/mnt/data/users/genetokenizer/one_hot.bpe.model

DISTRIBUTED_ARGS=(
    --nnodes 1 
    --nproc_per_node 4 
    --node_rank 0  
    --master_addr localhost 
    --master_port 29600
)

PR=${PR:-bf16}

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16"
    export USE_BLOCK_FP8=true
    export SAVE_MEMORY=true 
#    pr_options=" \
#        --bf16 \
#        --fp8-format hybrid \
#        --fp8-recipe delayed \
#        --fp8-param-gather \
#        --fp8-amax-compute-algo max \
#        --fp8-amax-history-len 1024"
fi



SEQ_LENG=32000
let NUM_LAYERS=18
#    --use-mcore-models
MODEL_ARGS=(
    --transformer-impl transformer_engine ###default
    --disable-bias-linear
    --seq-length $SEQ_LENG
    --max-position-embeddings $SEQ_LENG
    --num-layers $NUM_LAYERS
    --hidden-size 1280 ###args.hidden_size % args.num_attention_heads == 0
    --ffn-hidden-size 2048
    --num-attention-heads 10
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --use-rotary-position-embeddings 
    --no-bias-swiglu-fusion 
    --no-rope-fusion 
    --rotary-base 100000000
    --rotary-scaling-factor 1
    --rotary-seq-len-interpolation-factor 1
    --swiglu
    --untie-embeddings-and-output-weights
    --moe-router-dtype fp32 ##moe 
    --moe-permute-fusion 
##    --multi-latent-attention
    --bf16
)


MOE_INTERMEDIATE_SIZE=1024
# Q_LORA_RANK=1536 # 后训练组删除
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
NUM_EXPERTS=32
ROUTER_TOPK=3
NUM_SHARED_EXPERTS=1
MOE_LAYER_FREQ=1
MOE_FIRST_K_DENSE_REPLACE=2
RMS_NORM_EPS=1e-6

##export UB_SKIPMC=1 ###if moe grouped gemm is set
##    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} ##无此参数
MOE_ARGS=(
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} 
    --moe-router-topk ${ROUTER_TOPK} 
    --num-experts ${NUM_EXPERTS} 
    --moe-layer-freq ${MOE_LAYER_FREQ}  
    --moe-aux-loss-coeff 0.001 
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} ))  
    --kv-lora-rank ${KV_LORA_RANK} 
    --qk-head-dim ${QK_NOPE_HEAD_DIM} 
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} 
    --v-head-dim ${V_HEAD_DIM} 
    --moe-grouped-gemm ###
)

DATA_ARGS=(
    --tokenizer-type SentencePieceTokenizer  ##Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_PATH} ##/one-hot.bpe.model
    --vocab-file $TOKENIZER_PATH/tokenizer.model
    --data-path $DATA_PATH
    --split 10,1,1
    --no-create-attention-mask-in-dataloader
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --lr 5e-5
    --train-iters 40
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 10
    --clip-grad 1.0
)


let TP=2
let PP=1
let EP=2
AC=full
DO=true ###use distributed optimizer
FL=true
SP=true

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
#    --expert-tensor-parallel-size $TP
    --expert-model-parallel-size $EP
    --use-distributed-optimizer
    --context-parallel-size 1
##
##    --sequence-parallel

)


if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    fl_options=" --attention-backend flash " ###Disabling FlashAttention as it does not support MLA
    
#    export NVTE_FUSED_ATTN_BACKEND=0 ##
#    fl_options="\
#		    --attention-backend fused"
    echo $fl_options
elif [ $FL = false ]; then
    #export NVTE_FUSED_ATTN=1
    fl_options=" --attention-backend unfused "
fi



# For MoE Stability
if [[ ${WARMUP_ROUTER:-0} -gt 0 ]]; then
    moe_options=" ${moe_options}  --moe-warmup-router  ${WARMUP_ROUTER}  "
fi

if [ ! -z ${APPLY_NORM_HEAD} ];then
    moe_options=" ${moe_options}  --moe-apply-norm-head "
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option=""
#    --overlap-grad-reduce \
#    --overlap-param-gather

#if [ $TP_COMM_OVERLAP -eq 1 ]; then
#    comm_overlap_option=" --tp-comm-overlap --overlap-grad-reduce --overlap-param-gather"
#fi

let MP_AC_LAYERS=1

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
        --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-granularity selective \
        --recompute-modules ${RECOMPUTE_MODULES:-"core_attn moe_act layernorm mla_up_proj mlp moe"} \
    "
    if [[ ${MOE_PERMUTE_CHECKPOINT:-none} != none ]]; then
        activation_checkpoint_options=" ${activation_checkpoint_options} \
            --moe-perm-checkpoint ${MOE_PERMUTE_CHECKPOINT} 
        "
    fi
elif [ $AC = permckpt ]; then
    activation_checkpoint_options=" \
        --recompute-granularity selective \
        --recompute-beside-moe \
        --recompute-modules moe \
        --moe-perm-checkpoint ${MOE_PERMUTE_CHECKPOINT:-half} \
    "
elif [ $AC = moeckpt ]; then
    activation_checkpoint_options=" \
        --recompute-beside-moe \
    "
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi
#echo $activation_checkpoint_options

USE_FSDP=false
# User custom FSDP from Megatron Core
if [[ ${USE_FSDP} = true ]] ; then
    fsdp_options="\
        --use-custom-fsdp \
        --data-parallel-sharding-strategy optim_grads_params \
        --no-gradient-accumulation-fusion \
        --calculate-per-token-loss \
        "
    unset CUDA_MAX_CONNECTIONS
    unset CUDA_DEVICE_MAX_CONNECTIONS
fi

# Precision Aware Optimizer
OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-true}
PAO_LEVEL=${PAO:-none}

if [[ $PAO_LEVEL = none ]]; then
    new_options=" ${new_options} \
    "
    OFFLOAD_OPTIMIZER=false
elif [[ $PAO_LEVEL = moments ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
    "
elif [[ $PAO_LEVEL = grads ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
        --main-grads-dtype bf16 \
    "
elif [[ $PAO_LEVEL = weights ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
        --main-grads-dtype bf16 \
        --main-params-dtype fp16 \
    "
else
    echo "PAO_LEVEL=${PAO_LEVEL} is not a valid option. Valid options include: none, moments, grads, weights"
    exit 1
fi
if [[ $OFFLOAD_OPTIMIZER = true ]]; then
    new_options=" ${new_options} \
        --optimizer-cpu-offload \
    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

LOGGING_ARGS=(
    --log-interval 1 
    --save-interval 1000 
    --eval-interval 1000 
    --eval-iters 10 
    --save $OUTPUT_PATH 
#    --load $OUTPUT_PATH 
    --tensorboard-dir "${OUTPUT_PATH}/tensorboard" 
    --no-load-optim 
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${fl_options} \
    ${sp_options} \
    ${activation_checkpoint_options} \
    ${new_options} \
    ${comm_overlap_option} \
    ${moe_options} \
    ${fsdp_options} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
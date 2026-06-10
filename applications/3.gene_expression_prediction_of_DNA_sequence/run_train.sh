#source .env
export WANDB_API_KEY=b63e3d8ba6a07e60b6deffa1f6950391087c5c96
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NVTE_COMM_OVERLAP=0
export NCCL_P2P_DISABLE=1
export NCCL_P2P_DIRECT_DISABLE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export nproc_per_node=8

DISTRIBUTED_ARGS=(
    --nnodes 1 
    --nproc_per_node $nproc_per_node
    --node_rank 0  
    --master_addr localhost 
    --master_port 29520
)

# export nproc_per_node=4
# DISTRIBUTED_ARGS=(
#     --nnodes $WORLD_SIZE
#     --nproc_per_node $nproc_per_node 
#     --node_rank $RANK  
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PROT
# )

torchrun ${DISTRIBUTED_ARGS[@]} train.py \
    --model_path /path/to/baic_model \
    --tokenizer_dir /path/to/baic_model \
    --sequence_split_train_multi data/indices/test_*_multitrack/sequence_split_train.csv \
    --sequence_split_val data/indices/valid_*_multitrack/sequence_split_train.csv \
    --index_stat_multi_json data/indices/test_*_multitrack/index_stat.json \
    --nonzero_means 0 0 \
    --train_chromosomes Chr01 Chr02 Chr03 Chr04 Chr05 Chr06 Chr07 Chr08 Chr09 Chr10 Chr11 Chr12 \
    --output_base_dir /path/to/output/$(date +%Y%m%d%H%M) \
    --lr 0.00005 \
    --batch_size_per_device 1 \
    --gradient_accumulation_steps 10 \
    --num_train_epochs 20 \
    --loss_func mse \
    --max_sequence_length 32768 \
    --use_flash_attn \
    --gpus_per_node $nproc_per_node \
    --val_chromosomes Chr01 Chr02 Chr03 Chr04 Chr05 Chr06 Chr07 Chr08 Chr09 Chr10 Chr11 Chr12 \
    --use_wandb 


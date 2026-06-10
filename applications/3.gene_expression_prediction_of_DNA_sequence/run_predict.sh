#source .env

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NVTE_COMM_OVERLAP=0
export NCCL_P2P_DISABLE=1
export NCCL_P2P_DIRECT_DISABLE=1

export CUDA_VISIBLE_DEVICES=0,1

DISTRIBUTED_ARGS=(
    --nnodes 1
    --nproc_per_node 2
    --node_rank 0
    --master_addr localhost
    --master_port 29501
)

#Predict on test set
torchrun ${DISTRIBUTED_ARGS[@]} predict.py \
  --model_path /path/to/basic_model \
  --tokenizer_path /path/to/basic_model \
  --ckpt_path /path/to/finetuned_model/model.safetensors \
  --sequence_split_test data/indices/valid_*_multitrack/sequence_split_train.csv \
  --index_stat_json data/indices/valid_*_multitrack/index_stat.json \
  --bigWig_labels_meta data/indices/valid_*_multitrack/bigWig_labels_meta.csv \
  --max_predict_samples 500 \
  --output_base_dir outputs/valid_data \
  --test_chromosomes Chr06 \
  --batch_size 3 \
  --num_workers 8 \
  --use_flash_attn 

# predict on train set
torchrun ${DISTRIBUTED_ARGS[@]} predict.py \
  --model_path /path/to/basic_model \
  --tokenizer_path /path/to/basic_model \
  --ckpt_path /path/to/finetuned_model/model.safetensors \
  --sequence_split_test data/indices/test_CSQ_YG_P4_multitrack/sequence_split_train.csv \
  --index_stat_json data/indices/test_CSQ_YG_P4_multitrack/index_stat.json \
  --bigWig_labels_meta data/indices/test_CSQ_YG_P4_multitrack/bigWig_labels_meta.csv \
  --max_predict_samples 500 \
  --output_base_dir outputs/traindata \
  --test_chromosomes Chr06 \
  --batch_size 3 \
  --num_workers 8 \
  --use_flash_attn 



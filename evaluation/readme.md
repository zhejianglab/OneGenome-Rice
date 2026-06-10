# Benchmark code Quick Start

## Environment Setup

**Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage
1. **Modify the config file and run evaluation**:
   ```bash
   python benchmarks.py --config config.yaml
   ```

2. **Configuration Parameters**:
- Model Configuration
   ```yaml
   # Specify the model path for embedding extraction.
   # The model name will be inferred from the last directory in the path.
   model_path: "path_to_your_model_checkpoint"
   # Note: Some model architectures may not be supported by this framework.
   # If embedding extraction fails, you can manually extract embeddings.
   # In that case:
   #   - Remove model_path
   #   - Specify model_name and num_layers instead (see advanced usage below)

   # If the model and number of layers can be correctly detected,
   # remove the following two parameters.
   model_name: "model_name"
   num_layers: 12
   ```
- Dataset Configuration
   ```yaml
   # Path to dataset directory
   dataset_path: "/data/dataset_path"
   # Path to dataset feature configuration
   datasets_feature_path: "benchmarks/datasets_info.yaml"
   # Evaluation datasets
   # Add the datasets you want to evaluate.
   # For custom datasets, see the section below.
   eval_datasets:
     - demo_coding_vs_intergenomic_seqs
     - human_enhancers_cohn
     - splice_sites_all
   ```
- Task Scheduling
   ```yaml
   # GPU configuration: specify GPU IDs as a list.
   # If not specified, all available GPUs will be used.
   gpu_list: [0,1]
   # If not specified, all layers will be evaluated.
   layer_to_eval: [12]
   
   # Task scheduling parameters (see details below)
   process_power_per_gpu: 5       # Total capacity per GPU
   embedding_process_power: 4     # Cost of embedding extraction task
   eval_process_power: 1          # Cost of evaluation task

   # Data split size for embedding extraction
   embedding_extract_split: 50000
   ```
- Embedding Extraction
   ```yaml
   # Output directory for embeddings (may require large storage)
   embedding_output_dir: "/results/embeddings"
   # Batch size for embedding extraction
   batch_size: 8
   ```
- Downstream Tasks

   ```yaml
   # Default classifier type: MLP, XGB (XGBoost), or RF (Random Forest)
   classifer_type: "MLP"

   # MLP configuration
   mlp_dropout: 0.2
   mlp_lr: 0.0001
   mlp_epochs: 100
   save_last_epoch_model: False

   # Random Forest configuration
   rf_n_estimators: 100

   # XGBoost configuration
   xgb_n_estimators: 100
   xgb_eval_metric: 'mlogloss'
   xgb_learning_rate: 0.1
   xgb_max_depth: 6

   # Strategy for combining multiple sequence embeddings per sample
   # (1: concatenation, e.g., 2×1024 → 2048; 0: mean pooling)
   pooled_embeddings_cat_dim: 0
   ```
- Report Generation
   ```yaml
   # Path to save results
   eval_result_path: "results_path" 

   # Supported classification metrics:
   # ["accuracy","roc_auc","precision","recall","f1","mcc"]

   # Supported regression metrics:
   # ["r2","pearsonr","spearmanr"]

   # Extract best results per dataset:
   # - Single metric: select the best layer and report all metrics for that layer
   # - Multiple metrics: select the best value for each metric across all layers
   target_best_table:
   - "accuracy"
   - "roc_auc"
   - ["roc_auc", "accuracy"]
   - "pearsonr"

   # Plot layer-wise performance curves
   # - Single metric: plot directly
   # - Multiple metrics: first is primary metric, others shown in legend
   target_layer_line_figure:
   - ["roc_auc", "accuracy"]
   - ["accuracy", "roc_auc"]
   - "pearsonr"
   ```
- Disable if too many experiments are generated
    ```yaml
    wandb_report: False
    wandb_project: "mlp"
    wandb_entity: "wandb_entity"
    wandb_online: False
   ```
- Other
    ```yaml
    seed: 42
   ```

## Advanced Usage

### 1. Using Pre-extracted Embeddings Only
#### 1.1 Modify`config.yaml`
```yaml
# Comment out model_path
# model_path: "/path/to/model"
model_name: "your_model_name"  # Used for indexing results
num_layers: 12                 # Used to locate embedding files
```
#### 1.2 Embedding File Format
- File name
`model_name/datasetname-[x]layer_[train/eval/test].pt`
- Data format
`{'embeddings':pooled_embeddings, 'labels':all_labels}`


### 2. Custom Dataset
#### 2.1 Dataset Format
- filename: `model_name/[train/eval/test].jsonl`
- Each line format in jsonl: `{"seq": "ATCG", "label": 0}`；Label format: Classification: int, Multi-label: [int, int], Regression: [float, float]

#### 2.2 Add Dataset Config (benchmarks/datasets_info.yaml)

```yaml
  Human_classify_1048576:
    seq_for_item: 1   
    seq_key: seq      
    label_key: label 
    eval_task: classification  
    data_split: ['train', 'test', 'eval']  
    dataset_ratio:  [0.79, 0.09, 0.12]  
    classifer_type: "XGB"  
    sample_num:  28802    
    min_length:  1048576  
    max_length:  1048576   
    dataset_ratio:  [0.79, 0.09, 0.12] 
    label_train_counter:  [(0, 4507), (1, 10582), (2, 7837)]
    label_test_counter:  [(0, 392), (1, 1174), (2, 979)]   
    label_eval_counter:  [(0, 784), (1, 1568), (2, 979)]
```

### 3. Task Scheduling Logic

Task scheduling is mainly controlled by four parameters in the `config.yaml` file:  
`process_power_per_gpu`, `embedding_process_power`, `eval_process_power`, and `embedding_extract_split`. All of them are configurable.

- `process_power_per_gpu`, `embedding_process_power`, and `eval_process_power` are used to manage task allocation and scheduling.
- `embedding_extract_split` controls how embedding tasks are partitioned.

#### 3.1 Task Allocation

- `process_power_per_gpu`: total capacity (weight) available on each GPU  
- `embedding_process_power`: weight consumed by each embedding extraction task  
- `eval_process_power`: weight consumed by each downstream evaluation task after embeddings are obtained  

For example, given:
process_power_per_gpu: 5
embedding_process_power: 2
eval_process_power: 1

Assume no precomputed embeddings are available, with 2 GPUs and 3 datasets (`dataset_A`, `dataset_B`, `dataset_C`). The scheduling process proceeds as follows:

**Initial state**:
```
GPU-0: load 0/5 [idle]
GPU-1: load 0/5 [idle]
embedding queue: [dataset_A, dataset_B, dataset_C]
eval queue: []
```

**First round (embedding tasks prioritized)**:
- GPU-0 is assigned embedding extraction for `dataset_A`, load becomes 2/5  
- GPU-1 is assigned embedding extraction for `dataset_B`, load becomes 2/5  
```
GPU-0: load 2/5 [dataset_A embedding]
GPU-1: load 2/5 [dataset_B embedding]
embedding queue: [dataset_C]
eval queue: []
```

**Second round**:
- GPU-0 still has capacity (3 units remaining), so it can take another embedding task  
- GPU-0 is assigned embedding extraction for `dataset_C`, load becomes 4/5  
```
GPU-0: load 4/5 [dataset_A embedding, dataset_C embedding]
GPU-1: load 2/5 [dataset_B embedding]
embedding queue: []
eval queue: []
```

**Task completion and cascading scheduling**:
- When `dataset_A` embedding is completed, GPU-0 releases 2 units (load becomes 2/5)  
- `dataset_A` is automatically added to the eval queue  
- GPU-0 can then take the eval task for `dataset_A` (cost = 1), load becomes 3/5  
```
GPU-0: load 3/5 [dataset_C embedding, dataset_A evaluation]
GPU-1: load 2/5 [dataset_B embedding]
embedding queue: []
eval queue: []
```

**Final optimized state**:
As tasks continue to complete, GPU resources are efficiently utilized:
```
GPU-0: load 3/5 [dataset_B evaluation, dataset_C evaluation, other tasks]
GPU-1: load 2/5 [new embedding tasks or multiple evaluation tasks]
```

This weighted scheduling mechanism ensures:
1. **No resource overload**: total load per GPU never exceeds its capacity  
2. **Task prioritization**: embedding tasks (higher weight) are prioritized  
3. **Load balancing**: new tasks are assigned to the least-loaded GPU  
4. **Cascading execution**: evaluation tasks are automatically triggered after embedding completion  

#### 3.2 Embedding Task Splitting

To optimize memory usage and improve parallel efficiency, the system adopts a **data partitioning mechanism** for embedding extraction, controlled by the `embedding_extract_split` parameter.

##### **Splitting Strategy**
- `embedding_extract_split: 50000` means each subtask processes up to 50,000 samples  
- Large datasets are automatically divided into multiple smaller subtasks  
- Each subtask is independently scheduled on GPUs, improving parallelism  

This mechanism enables efficient processing of datasets of arbitrary size while maintaining optimal resource utilization and system stability.

---

**Note**: Before using this tool, ensure that you have sufficient computational resources and storage space, especially when working with large models and datasets. Evaluation results may vary slightly across different platforms or environments.

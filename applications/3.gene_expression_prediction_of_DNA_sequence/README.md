# Gene Expression Prediction of DNA Sequence

## 1. Overview
A central challenge in predictive genomics is linking static DNA sequence to dynamic, context-specific gene expression. This repository implements a scalable, multi-modal deep learning framework for single-nucleotide resolution RNA-seq prediction. Given a genomic DNA sequence window, the model learns to predict strand-specific transcriptional output by jointly modeling sequence context and regulatory signals. 

The architecture leverages a pre-trained DNA foundation model as a sequence encoder, paired with a U-Net-style regression head designed for multi-track genomic signal prediction. The framework supports full-parameter fine-tuning, distributed data-parallel training, and efficient inference, enabling downstream applications such as cis-regulatory variant effect prediction, allele-specific expression modeling, and transcriptome-informed breeding design.


## Technology stack
- **PyTorch** with **PyTorch Distributed (DDP)** and **NCCL** on CUDA for multi-GPU.
- **Hugging Face Transformers** (`Trainer`) for the training loop and integration with a frozen/unfrozen DNA LM.
- **Genomics I/O:** `pyBigWig`, `pyfaidx`; configs via **PyYAML**.

## Data prepartion
The data_prepare.sh script orchestrates the full preprocessing pipeline:
Signal Normalization: Renormalizes raw BigWig tracks to a common scale using renorm_bigwig.py.
Window Tiling: Splits genomes into overlapping 32,768 bp windows (16,384 bp stride) using sequence_split_and_meta_extract2.py.
Metadata Generation: Creates index_stat.json (non-zero mean & track statistics) and bigWig_labels_meta.csv for strand-specific track mapping.
Chromosome Partitioning: Automatically separates training, validation, and test sets by accession/chromosome.

### Run
- The BigWig file needs to be renamed as tissue_species_1.bw.
- Run data_prepare.sh first to generate training and validation datasets.

### Output directories:
data/processed/renorm_bigwig_output/: Normalized BigWig files
data/indices/{split}_multitrack/: Window indices, stats, and metadata CSVs

## Training and prediction
### Model Training
Training is executed via run_train.sh, which launches train.py using torchrun for Distributed Data Parallel (DDP) synchronization.
Key Architectural & Optimization Details
Encoder: Pre-trained DNA language model (32k context) with only the final transformer block unfrozen by default.
Decoder: U-Net-style encoder-decoder with configurable projection (1024), 4 downsampling blocks, and bottleneck dimension (1536).
Precision: bfloat16 mixed precision + FlashAttention-2 for accelerated self-attention.
Optimizer: Adafactor with cosine learning rate schedule (10% linear warmup), weight decay 0.01, and gradient clipping (max_norm=1.0).
Loss: Selectable across mse, poisson, tweedie, or poisson-multinomial (default: mse).
Logging: Real-time metrics via Weights & Biases and structured JSONL logs; per-track loss aggregation handled by CustomTrainer.


### Key arguments passed to train.py:
| Argument | Description |
|---|---|
| `--model_path` / `--tokenizer_dir` | Path to pre-trained foundation model |
| `--sequence_split_train_multi` | Comma-separated window index CSVs |
| `--index_stat_multi_json` | Corresponding `index_stat.json` paths |
| `--train_chromosomes` / `--val_chromosomes` | Chromosome lists for train/val splits |
| `--output_base_dir` | Checkpoint & log output directory |
| `--lr`, `--batch_size_per_device`, `--num_train_epochs` | Optimization hyperparameters |
| `--use_flash_attn`, `--use_wandb` | Enable acceleration & experiment tracking |

### Inference & Visualization
Inference is handled by predict.py, launched via run_predict.sh. It processes held-out chromosomes or accessions, outputs base-resolution prediction tracks, and supports batched sliding-window evaluation.

### Key inference arguments:
| Argument | Description |
|---|---|
| `--ckpt_path` | Path to fine-tuned `.safetensors` checkpoint |
| `--sequence_split_test` | Test set window index CSV |
| `--test_chromosomes` | Chromosomes to evaluate |
| `--output_base_dir` | Prediction output directory |
| `--batch_size`, `--num_workers` | Inference throughput settings |

### Visualization
Use src/viewer.py to plot predicted vs. observed tracks, compare reference vs. mutant sequences, or generate publication-ready genomic browser-style figures:


## File Structure
```text
├── data_prepare.sh # End-to-end data preprocessing & window indexing
├── train.py # Core training script with DDP & custom Trainer
├── predict.py # Inference script for chromosome/window-level prediction
├── run_train.sh # Shell wrapper for distributed training launch
├── run_predict.sh # Shell wrapper for distributed inference launch
├── requirements.txt # Python dependencies
├── README.md # This file
├── scripts/ # Data preprocessing & index generation utilities
└── src/ # Core modules
    ├── model.py # GenOmics architecture (DNA encoder + U-Net decoder)
    ├── trainer.py # Custom Trainer (DDP sync, per-head loss, collation)
    ├── dataset.py # Multi-track BigWig/FASTA dataset & collate functions
    ├── metrics.py # Pearson/Spearman/R² evaluation utilities
    ├── viewer.py # Prediction visualization & track comparison tools
    └── util.py # DDP setup, logging, distributed synchronization
```

## 3. Environment Setup
We recommend using a clean Conda environment with CUDA 12.1+ and PyTorch 2.0+ to leverage `bfloat16` mixed precision and FlashAttention-2.


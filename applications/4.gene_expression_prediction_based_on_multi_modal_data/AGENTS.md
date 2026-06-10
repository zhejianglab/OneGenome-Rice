# AGENTS.md — ATAC-conditioned RNA prediction

This document is the **authoritative guide** for agents and humans working in this repository. It reflects the **current** layout, entrypoints, configuration, and training/inference behavior.

---

## 1. What this project does

The code trains and runs models that predict **strand-specific RNA-seq coverage** along a genomic window from:

- **DNA** — reference sequence for the same window (tokenized via a Hugging Face DNA language model).
- **ATAC-seq** — chromatin accessibility in BigWig format for the same window.

**Predictor:** only **`predictor.type: fusion`** is supported (default if `predictor` or `type` is omitted). Other predictor types are rejected by `valid_config.py`.

---

## 2. Technology stack

| Area | Choice |
|------|--------|
| Deep learning | **PyTorch** |
| Training loop | **Hugging Face `transformers.Trainer`** (custom subclasses in `model/trainer.py`) |
| DNA backbone | **Hugging Face** pretrained model directory (`model_path`) |
| Checkpoints | **`safetensors`** (saved by the custom trainer) |
| Genomics I/O | **`pyBigWig`**, **`pyfaidx`** |
| Config | **YAML** (`pyyaml`) |

Optional / environment-specific: **Flash Attention** may be used by the installed base model; it is not a separate hard dependency of this repo’s Python entrypoints. Declare actual dependencies in your deployment environment (see `pyproject.toml` for formatter config only; there is no full lockfile in-tree).

**Preprocessing** (outside the core training code): shell and Python helpers under `ATAC/` and `RNA/` (alignment, BigWig generation, etc.).

---

## 3. Repository layout (current)

Top-level **Python entrypoints and tools**:

| File | Role |
|------|------|
| **`run.py`** | **Primary workflow:** train → inference on `training_data` → inference on `test_data` → optional `collect_stats.py`. Resume via flag files under `<output_base_dir>/flags/`. Do **not** wrap `run.py` with `torchrun`. |
| **`training.py`** | Standalone training: loads YAML, validates with `valid_config`, tees logs, calls `model.pipeline.run_training`. Also invoked **by `run.py`** as a subprocess. |
| **`inference.py`** | Standalone inference from a YAML config. Also invoked **by `run.py`** as a subprocess. |
| **`collect_stats.py`** | Scans `reg/` and `test/` under a run directory, runs `calc_metrics.py` in parallel, builds **`stats.wide.full.csv`**. Used when `collect_stats: true` in experiment config. |
| **`calc_metrics.py`** | Metrics from paired pickle outputs (R², Pearson, Pearson on `log1p` values; per ± strand). |
| **`valid_config.py`** | Validates YAML (structure + input path existence). CLI: `python valid_config.py path/to/config.yaml`. |
| **`tee_log.py`** | Tee `stdout`/`stderr` to files (used by `training.py` and `run.py`). |

**Package `model/`** (library shared by training and inference):

| Module | Role |
|--------|------|
| `pipeline.py` | `run_training`, `build_multimodal_model`, dataset wiring, **fusion-only** predictor selection. |
| `trainer.py` | `CustomTrainer`, fusion + discriminative-LR variants, **`FractionalEpochSchedulerCallback`**, safetensors save, CSV loss log, checkpoint folder naming. |
| `predictor_fusion.py` | `MultiModalPredictorFusion` — fusion head (cross-attn, gates, U-Net-style decoder). |
| `encoder_transformer.py` | `ATAC_TransformerEncoder` — RoPE-aligned ATAC encoder → `[B, 1024, L]`. |
| `dataset.py` | `LazyGenomicDataset` (training), `InferenceDataset` (inference). |
| `index.py` | `build_index` — windows, BigWig values, CSV index. |
| `config.py` | `parse_dataset_block`, batch-size helpers, dataset YAML parsing (including legacy layouts if present in YAML). |
| `distributed.py` | DDP setup, `dist_print`, SyncBatchNorm helper, `DistributedSamplerCallback`. |
| `load_pretrained.py` | Load tokenizer + base model from `model_path`. |
| `env.py` | Seeds, logging, WANDB-related env defaults. |
| `scaling.py` | `LabelScaler` for RNA tracks at inference. |

**Example configs:** `config.example/`

- `experiment.yaml` — unified template for **`run.py`** (`output_base_dir`, `training_data`, `test_data`, training block, inference flags, `inference_checkpoints`, `collect_stats`).
- `training.yaml` — template for **`training.py`** only (`output_training_dir`, etc.).
- `infer.yaml` — template for **`inference.py`** only (`ckpt_path`, `output_eval_dir`, `test_data`, …).

**Preprocessing pipelines:** `ATAC/`, `RNA/`, `SRR_download.sh` (see comments inside those trees).

---

## 4. Recommended workflow: `run.py`

### 4.1 What `run.py` does

1. **Training** — writes `<output_base_dir>/config/train.yaml` with `output_training_dir` set to `<output_base_dir>/model`, then runs `training.py` on that file.
2. **Inference (train split)** — results under `<output_base_dir>/reg/`.
3. **Inference (test split)** — results under `<output_base_dir>/test/` (skipped if `test_data` is empty).
4. **Stats** — if `collect_stats` is true (default in `experiment.yaml`), runs `collect_stats.py` on `<output_base_dir>`.

**Resume:** completion is recorded under `<output_base_dir>/flags/` (e.g. `training_done`, `inference_train.<sample>.<checkpoint>_done`). Use **`--force`** to clear flags and rerun from scratch.

**GPU count:** set `nproc_per_node` in YAML or pass **`--nproc-per-node N`**. When `N > 1`, `run.py` launches `training.py` / `inference.py` via `python -m torch.distributed.run --nproc_per_node=N --standalone` (single-node only).

### 4.2 CLI reference

```text
python run.py -c PATH/to/experiment.yaml     # load config; require output_base_dir in YAML (unless -d)
python run.py -d /path/to/run               # load /path/to/run/experiment.yaml; set output_base_dir to that directory
python run.py ... --force                   # delete <output_base_dir>/flags and rerun
python run.py ... --reuse-model             # skip training; require existing checkpoints under <output_base_dir>/model/
python run.py ... --nproc-per-node N        # override YAML nproc_per_node
python run.py ... --stats-parallel P      # max concurrent calc_metrics in collect_stats (default scales with N)
```

**Important:** Do **not** wrap `run.py` with `torchrun`; multi-process launch is internal.

### 4.3 Run directory layout (after a full run)

```text
<output_base_dir>/
  experiment.yaml          # optional copy from user; run may write generated configs under config/
  config/
    train.yaml             # generated training config (from run.py)
    infer.<subdir>.yaml    # generated per inference job
  model/                   # checkpoints + HF tokenizer/config exports
    checkpoint-<N>/       # N = training samples seen (see model/trainer.py)
    train_loss_per_log.csv
    train.o / train.e    # when training.py tees (paths depend on training output dir)
  reg/                     # inference on training_data
    <sample>.checkpoint-<N>/
  test/                    # inference on test_data (if any)
    <sample>.checkpoint-<N>/
  flags/                   # resume markers
  stats.wide.full.csv      # if collect_stats ran
  run.o / run.e            # tee from run.py
```

### 4.4 Checkpoint selection for inference (`run.py` only)

Under `inference_checkpoints`:

- **`pick_n`** — how many checkpoints to use.
- **`checkpoint_stride`** — stride when stepping through checkpoints ordered by **decreasing** checkpoint index (newest first), then inference runs in **ascending** checkpoint order.

Unsupported: `last_k` (validator error).

---

## 5. Standalone training (`training.py`)

```bash
python training.py -c /path/to/training.yaml
python training.py -d /path/to/run    # loads <run>/config/training.yaml; sets output_training_dir to <run>
```

Multi-GPU (single machine), **without** `run.py`:

```bash
torchrun --nproc_per_node=4 training.py -c /path/to/training.yaml
```

Training logic lives in **`model/pipeline.py`** (`run_training`): builds index, `LazyGenomicDataset`, `TrainingArguments`, and a **CustomTrainerFusion** or **CustomTrainerDiscriminativeLRFusion** depending on `training.discriminative_lr`.

**Checkpoint schedule:** the pipeline uses **`FractionalEpochSchedulerCallback`** and typically `save_strategy: "no"` on `TrainingArguments`, driving saves from the callback. See `model/pipeline.py` and `model/trainer.py` for exact semantics (`save_num_per_epoch`, `save_per_n_epoch`, etc.).

**Checkpoint directory names:** `_save_checkpoint` in `model/trainer.py` names folders `checkpoint-<samples_seen>` (samples implied by HF global step × per-step batch), not necessarily “step” in the old sense.

---

## 6. Standalone inference (`inference.py`)

```bash
python inference.py /path/to/infer.yaml
```

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=4 inference.py /path/to/infer.yaml
```

Requires `ckpt_path` (safetensors file), `output_eval_dir`, `model_path`, dtypes, `atac_encoder_output_dim`, and a **`test_data`** list (same entry shape as training: `name`, `rna_path_plus`, `rna_path_minus`, `atac_path`, optional `chromosome` / `genome_fasta`).

Optional: `calc_metrics: true` runs metrics inside inference (the **`run.py`** path prefers **`collect_stats.py`** instead).

---

## 7. Configuration

### 7.1 Validation

- **`run.py`** and **`training.py`** call **`validate_config_dict`** from `valid_config.py` before running.
- CLI check: `python valid_config.py your.yaml`.

Checks include: top-level mapping, **`model_path`** exists, optional **`ckpt_path`** file exists, dataset entries and BigWig/FASTA paths, **`predictor.type` must be `fusion`**, `inference_checkpoints` shape, mutual exclusion of `save_num_per_epoch` vs `save_per_n_epoch`, and basic `target_len` / `overlap_len` sanity.

### 7.2 Dataset entries

Preferred layout: **`training_data`** / **`test_data`** as lists of objects:

```yaml
name: "sample_id"
rna_path_plus: "/path/to/plus.bw"
rna_path_minus: "/path/to/minus.bw"
atac_path: "/path/to/atac.bw"
# optional: chromosome, genome_fasta
```

Legacy key layouts may still be parsed by **`model/config.py`** for older YAML; prefer the list form for new work.

### 7.3 Predictor (`fusion` only)

Under `predictor:`:

- **`type: fusion`** (or omit `type`).
- **`fusion_gate_entropy_frac`**, **`skip_gate_entropy_frac`** — optional entropy regularization on the fusion and skip gates (see `model/predictor_fusion.py`).
- **`unfreeze_base_last_layer`** — if true, gradients flow through the **last** transformer block of the base model; otherwise the base is frozen and only the head/ATAC path trains.

### 7.4 Training block (`training:`)

Common keys: `learning_rate`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `num_train_epochs`, `weight_decay`, `max_grad_norm`, `logging_steps`, `optim`, `warmup_ratio`, `save_total_limit`, `discriminative_lr`, `learning_rate_backbone`, `learning_rate_head`, LR scheduler keys (`lr_scheduler_type`, `min_lr_rate`, `lr_scheduler_kwargs`), etc.

**Saving:** `save_num_per_epoch` vs `save_per_n_epoch` are mutually exclusive (unless both are 1). If `save_strategy: steps` is used, `save_num_per_epoch` is required — details enforced in `valid_config.py` and implemented in `model/pipeline.py`.

**Gradient accumulation scaling:** optional `scale_gradient_accumulation_for_world_size` (boolean) — see comments in `config.example/training.yaml` / `experiment.yaml`.

### 7.5 Experiment-only keys (`run.py`)

- **`output_base_dir`** — run root (required unless using `run.py -d`, which sets it).
- **`inference_checkpoints`** — `pick_n`, `checkpoint_stride`.
- **`collect_stats`** — default true in example; set false to skip `collect_stats.py`.

---

## 8. Model architecture (concise)

1. **DNA backbone** — Hugging Face causal LM; hidden states \([B, L, H]\) projected effectively to **1024** channels per position for the fusion head (see `MultiModalPredictorFusion._encode_dna`).
2. **ATAC encoder** — **`ATAC_TransformerEncoder`**: per-position ATAC value, **RoPE** aligned to DNA `position_ids`, **6 layers**, **d_low=192**, **4 heads**, output **1024** channels, shape `[B, 1024, L]` (`model/encoder_transformer.py`).
3. **Fusion head** — **`MultiModalPredictorFusion`**: downsample DNA and ATAC to **L/4**, **bidirectional cross-attention**, **4-way softmax fusion gate**, dilated conv bottleneck, **U-Net** upsampling with ATAC skips, **2-way skip gate** at full resolution, final conv to **2 channels** (± strand), **softplus** outputs with a learned scale (`model/predictor_fusion.py`).

**Loss:** MSE on predictions vs labels; optional entropy terms subtracted scaled by detached MSE (training stability).

---

## 9. Distributed execution

| Mode | Supported? |
|------|------------|
| Single GPU / CPU | Yes |
| Single-node multi-GPU | Yes — `run.py --nproc-per-node N`, or `torchrun` on `training.py` / `inference.py` |
| Multi-node | **Not** wired in `run.py` (`--standalone`). Advanced users could adapt subprocess commands; not documented here. |

`model/distributed.py` sets up NCCL when launched under distributed; rank-0 logging uses `dist_print`.

---

## 10. Outputs and metrics

- **Training:** `model.safetensors`, `config.json`, tokenizer files under each checkpoint dir; **`train_loss_per_log.csv`** in `output_training_dir`.
- **Inference:** per-batch CSVs and/or `plus_predictions.pickle` / `minus_predictions.pickle` under each `output_eval_dir`.
- **Aggregate metrics:** `collect_stats.py` → **`stats.wide.full.csv`** at run root.

---

## 11. Code style (for contributors)

- Comments and user-facing strings in **English**.
- Naming: `snake_case` functions, `PascalCase` classes, private helpers with a leading underscore.
- Prefer **safetensors** for checkpoints; match existing dtype conventions (bf16 base vs fp32 head modules as in `model/pipeline.py` / predictor code).

---

## 12. Troubleshooting

| Issue | Suggestion |
|-------|------------|
| Checkpoint load / size mismatch | Same `predictor` block and `atac_encoder_output_dim` / model_path as training; only **fusion** is supported. |
| CUDA OOM | Lower `per_device_train_batch_size` or `inference_batch_size`; raise `gradient_accumulation_steps`. |
| BigWig / FASTA chr mismatch | Ensure chromosome names match between FASTA index and BigWigs. |
| NaN loss | Lower LR; check BigWigs for NaNs; inspect caps / `cap_expression_quantile`. |

---

## 13. Security and environment

- Training configures **W&B offline** by default in `model/env.py` when applicable.
- Prefer **absolute paths** in YAML for reproducibility.

---

## 14. Agent tooling (conda / packages)

- You may use a conda env (e.g. **`model`**) to inspect installed package behavior when debugging.
- **Ask before installing** new packages; prefer **conda** over **pip** when both work.

---

## 15. Repository scope

This tree is intended to be **self-contained**. Prefer updating in-repo imports and entrypoints directly rather than adding compatibility shims for external callers.

---

## 16. Related docs

- **`README.md`** — short quick start and stack summary.
- **`config.example/*.yaml`** — commented templates keyed to the current code paths.

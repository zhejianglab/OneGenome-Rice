# ATAC-conditioned RNA prediction

This repository trains and runs models that predict **strand-specific RNA-seq coverage** from a **DNA sequence window** and matching **ATAC-seq** (chromatin accessibility) in the same window. It targets plant and other genomics setups where BigWig tracks and a reference FASTA are available.

For a biologically oriented narrative of the task, architecture, and evaluation, see [`senario.md`](senario.md).

## What it does

- **Inputs:** reference genome (FASTA), ATAC BigWig, plus/minus strand RNA BigWigs, YAML experiment config.
- **Outputs:** checkpoints (`safetensors`), training logs, inference CSV/pickle predictions; optional metrics and plots via helper scripts.
- **Heads:** `predictor.type: fusion` (default if omitted). See `config.example/training.yaml` and `AGENTS.md`.

## Technology stack

- **PyTorch** with **PyTorch Distributed (DDP)** and **NCCL** on CUDA for multi-GPU.
- **Hugging Face Transformers** (`Trainer`) for the training loop and integration with a frozen/unfrozen DNA LM.
- **Genomics I/O:** `pyBigWig`, `pyfaidx`; configs via **PyYAML**.

Preprocessing pipelines for ATAC/RNA live under `ATAC/` and `RNA/` (see below). Full dependency and tooling lists are in `AGENTS.md`.

## Repository layout (high level)

| Path | Role |
|------|------|
| `run.py` | **Primary entrypoint**: end-to-end pipeline runner (train → infer on train/test → metrics summary) with resume flags. |
| `training.py` | Training subprocess entrypoint used by `run.py` (can be used standalone, but prefer `run.py`). |
| `inference.py` | Inference subprocess entrypoint used by `run.py` (can be used standalone, but prefer `run.py`). |
| `model/` | Core library: dataset, index, encoders, predictors, `pipeline.py`, DDP helpers. |
| `config.example/` | Example training and inference YAML. |
| `ATAC/` | **Preprocessing** (raw ATAC-seq reads → accessibility BigWig). Main entry: `ATAC/run_atac.py`; output feeds YAML `atac_path`. |
| `RNA/` | **Preprocessing** (raw RNA-seq reads → strand-specific expression BigWigs). Main entry: `RNA/run.py`; outputs feed YAML `rna_path_plus` / `rna_path_minus`. |
| `AGENTS.md` | Detailed project manual (config keys, metrics, troubleshooting). |
| `requirement.preprocess.txt` | Pinned **preprocess** (ATAC/RNA) Python packages; use with **Python 3.12.0** (conda: `python=3.12.0`). |
| `requirement.model.txt` | Pinned **model/training** (`run.py`, `training.py`, `inference.py`, `model/`) Python packages; use with **Python 3.10.12** (conda: `python=3.10.12`). |

## Preprocessing environment (ATAC / RNA)

Use a dedicated conda (or similar) environment for the helpers under `ATAC/` and `RNA/` with **`python=3.12.0`**. Package pins that match the reference preprocess stack are in **`requirement.preprocess.txt`** (see the header comments there for how the list was produced).

## Model training environment

Use a separate environment for training and inference with **`python=3.10.12`**. Package pins that match the reference model stack are in **`requirement.model.txt`** (see the header comment there for the Python version). For config keys, metrics, and troubleshooting beyond pinned versions, see **`AGENTS.md`**.

## `ATAC/` — ATAC-seq preprocessing

Turn **paired-end FASTQ** into a **single-track BigWig** suitable for `atac_path` in your YAML.

**Entrypoint:** `ATAC/run_atac.py` (orchestrates the shell/Python helpers in the same folder).

**Inputs**

- **Reads:** `<input_prefix>_R1.fastq.gz` and `<input_prefix>_R2.fastq.gz`
- **Reference:** Bowtie2 index **prefix** passed as `ref` (same style as Bowtie2 `-x`)

**Typical command**

```bash
python ATAC/run_atac.py <input_prefix> <ref_prefix> <output_prefix> -p 8
```

**Pipeline steps (in order)**

1. **Trimming** — `trimmomatic_atac.sh` (Trimmomatic on PE reads).
2. **Merge PE → SE** — Concatenate trimmed R1/R2 (paired + unpaired) into one `<output_prefix>_SE.fastq.gz` via `pigz`.
3. **Align** — `align.bt2_se.sh`: Bowtie2 **single-end** alignment to sorted BAM (`*_SE.align.sorted.bam`).
4. **QC stats** — `samtools stats` on the sorted BAM.
5. **BigWig** — `bam2bw_atac.py`: `bamCoverage`-style coverage (defaults: mapping quality ≥ 30, bin size 1, normalization **RPGC** unless overridden with `-n` / `-q` / `-b`).

Logs for the driver are teed to `<output_prefix>.log` (see `tee_log` usage in the script).

## `RNA/` — RNA-seq preprocessing

Turn **paired-end FASTQ** into **strand-specific BigWigs** (plus and minus) suitable for `rna_path_plus` / `rna_path_minus` in your YAML.

**Entrypoint:** `RNA/run.py`.

**Inputs**

- **Reads:** `<input_prefix>_R1.fastq.gz` and `<input_prefix>_R2.fastq.gz`
- **Reference:** HISAT2 index **prefix** for `align_hisat.py`, plus files expected by `bam2bw_RNA.py` (e.g. reference exclude BED — see script help/comments).

**Typical command**

```bash
python RNA/run.py <input_prefix> <ref_prefix> <output_prefix> -p 8
```

Use `-s` / `--rna-strandness` when the library is stranded (`F`, `R`, `FR`, `RF`; default is unstranded `None`).

**Pipeline steps (in order)**

1. **Align** — `align_hisat.py`: HISAT2 paired-end alignment, then **sorted BAM** (`<output_prefix>.bam`).
2. **BigWig** — `bam2bw_RNA.py`: strand-aware coverage tracks via deepTools-style `bamCoverage` (defaults: **CPM** normalization, configurable `-n`, `-q`, `-b`).

Logs are teed to `<output_prefix>.log`.

**Note:** `bam2bw_RNA.py` may contain **machine-specific** paths for the Python/deepTools environment at the top of the file; adjust them on your cluster before relying on this step.

## Quick start

1. Copy and edit `config.example/experiment.yaml` (paths to genome, BigWigs, Hugging Face model dir, `output_base_dir`, etc.).
2. Run the full pipeline (training + inference + optional stats):

```bash
python run.py -c config.example/experiment.yaml
```

**Run directory mode:** load `DIR/experiment.yaml` and force `output_base_dir` to `DIR`:

```bash
python run.py -d /path/to/run
```

**Skip training / reuse existing checkpoints** (expects `DIR/model/checkpoint-*` to exist):

```bash
python run.py -d /path/to/run --reuse-model
```

## Multi-GPU (recommended: via `run.py` only)

`run.py` launches `training.py` / `inference.py` internally (using `torch.distributed.run` when \(N>1\)). **Do not wrap `run.py` itself with `torchrun`.**

### Single machine

Set GPU count either in `experiment.yaml` (`nproc_per_node`) or on the command line:

```bash
python run.py -c /path/to/experiment.yaml --nproc-per-node 4
```

Replace `4` with your GPU count. `run.py` will use that value for both training and inference subprocesses.

## More documentation

- **`AGENTS.md`** — Configuration reference, metrics, checkpoint layout, troubleshooting, and full repository structure.

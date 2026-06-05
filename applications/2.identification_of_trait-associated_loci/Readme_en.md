# Differential Locus Discovery from Bidirectional Genomic Attention

## 1. Overview

This repository demonstrates a reproducible workflow for identifying rice candidate loci from bidirectional attention signals produced by OneGenomeRice. The workflow reconstructs sample-specific sequences from variants, extracts forward and reverse-complement attention, performs position-level group comparisons, and summarizes gene-level differential signals in selected candidate regions.

## 2. Data

All required inputs are placed under `Data/`.

| File | Description |
|:--|:--|
| `glutinous.vcf.gz` | VCF subset for the glutinous/non-glutinous samples and four Chr6 candidate regions |
| `glutinous.vcf.gz.tbi` | Tabix index for `glutinous.vcf.gz` |
| `glutinous.tsv` | Glutinous/non-glutinous phenotype table with `SampleID` and `Trait` |
| `glutinous_regions.bed` | Four selected 40 kb genomic intervals for the glutinous workflow |
| `osa1_r7.asm.fa.gz` | OSA1/R7 reference genome FASTA used for pseudo-sequence reconstruction |
| `osa1_r7.asm.fa.gz.fai` / `osa1_r7.asm.fa.gz.gzi` | FASTA random-access indexes generated from `osa1_r7.asm.fa.gz` |
| `chr06.gff3.gz` | Chr6 MSU Rice Genome Annotation Project osa1r6 gene annotation used for gene-level scoring |
| `chr03.gff3.gz` | Chr3 MSU Rice Genome Annotation Project osa1r6 gene annotation used for the grain-length display |
| `grain_length.vcf.gz` | Chr3 grain-length VCF subset used by the grain-length workflow |
| `grain_length.vcf.gz.tbi` | Tabix index for `grain_length.vcf.gz` |
| `grain_length.tsv` | Grain-length phenotype table |
| `OneGenomeRice_model/` | Optional symlink or local directory for the pretrained OneGenomeRice 8 kb Hugging Face model |

The reference FASTA is not required to be committed with the repository. If it is missing, `0.env_check.sh` prints the configured download URL and can download, BGZF-compress, and index it automatically:

```bash
bash 0.env_check.sh --download-reference
```

The model path is configured in `default_config.json` under `paths.model`. Users can either edit this value to an absolute model path or create a symlink at `Data/OneGenomeRice_model`.

## 3. Workflow

The root directory contains three entry-point scripts.

| Step | Command | Output |
|:--|:--|:--|
| 0 | `bash 0.env_check.sh` | Checks input files, model files, Python modules, reference indexes, and CUDA visibility |
| 1 | `bash 1.glutinous.sh` | Generates or reuses glutinous-rice attention outputs from `Data/glutinous.vcf.gz`, runs differential tests when needed, and writes the four-region significance figure |
| 1 | `bash 1.grain_length.sh` | Runs the default 50-sample grain-length smoke workflow and writes the Chr3 annotated association figure |

The main output directories are:

| Directory | Content |
|:--|:--|
| `Results/glutinous/` | Glutinous-rice attention outputs, differential tests, tables, and figures |
| `Results/grain_length/` | Grain-length attention outputs, model tables, and figures |

## 4. Methods

The four 40 kb regions are split into 8 kb windows with a 4 kb stride. During plotting and gene-level summarization, only the effective non-overlapping middle interval of each overlapping block is used, avoiding duplicated visualization or scoring of overlapping coordinates.

For each base position, forward and reverse-complement attention matrices are compared between phenotype groups using a Mann-Whitney U test, followed by Benjamini-Hochberg correction. The signal figures report `-log10(adjusted P)` and `log2FC` tracks for both attention directions.

For gene-level prioritization, the workflow uses gene bodies from `chr06.gff3.gz` without upstream/downstream extension. Forward, reverse-complement, and summed-direction matrices are evaluated with the ATLAS-style summary statistics implemented in this repository. The final display focuses on summed-direction Peak Density and Shannon Entropy rankings.

The grain-length extension records its own VCF at `Data/grain_length.vcf.gz`; it does not reuse glutinous-rice VCF or attention outputs. The default entry point runs an end-to-end 50-sample smoke workflow: it regenerates pseudo-sequences, extracts bidirectional attention, builds matrices under `Results/grain_length/smoke_test/attention/04_attention_matrices/`, runs the middle-4 kb extraction, fits the PCA-corrected linear model against `Data/grain_length.tsv`, computes bidirectional consistency scores, and redraws the Chr3 annotated Manhattan-style figure using `Data/chr03.gff3.gz`.

Run it in a CUDA-enabled environment:

```bash
bash 1.grain_length.sh --gpus 0,1
```

When multiple GPU IDs are supplied, the grain-length attention step is split by block range and run concurrently. By default, the number of attention workers equals the number of GPU IDs; override it with `--workers N` if needed.

The default grain-length command keeps 50 randomly selected matched samples and writes isolated outputs under `Results/grain_length/smoke_test/`, so the formal grain-length matrices and figures are not overwritten. The legacy `--smoke-test` flag is still accepted but is no longer required. Use `--smoke-samples N` and `--sample-seed N` to change the subset size or make a different deterministic draw.

To process all matched grain-length samples and write formal outputs under `Results/grain_length/`, add `--full_sample`:

```bash
bash 1.grain_length.sh --gpus 0,1 --full_sample
```

If complete grain-length attention matrices are already available locally, pass them with `--matrix-dir /path/to/matrices`. A precomputed association table can still be plotted explicitly with `--mode precomputed --observed-csv /path/to/observed_consistency_pvalues.csv`, but no precomputed table is part of the default open-source input set.

## 5. Environment

Run the workflow inside an environment that can execute OneGenomeRice inference and the downstream scientific Python stack. The entry-point scripts use the currently active Python environment by default. If preferred, users can set `CONDA_ENV` or fill `environment.conda_env` and `environment.conda_sh` in `default_config.json` to run through `conda run`.

Required Python modules:

```text
Bio
cyvcf2
matplotlib
numpy
pandas
pysam
scipy
seaborn
statsmodels
torch
tqdm
transformers
```

## 6. Usage

Run the workflow from the repository root:

```bash
bash 0.env_check.sh
bash 1.glutinous.sh --gpus 0,1
bash 1.grain_length.sh --gpus 0,1
```

If the reference FASTA is not present after cloning the package, download and index it automatically:

```bash
bash 0.env_check.sh --download-reference
```

If `osa1_r7.asm.fa.gz` already exists but `.fai` or `.gzi` is missing, rebuild the indexes with:

```bash
bash 0.env_check.sh --repair-reference-index
```

If a user manually downloads `osa1_r7.asm.fa.gz` with `curl` or `wget`, the environment check will verify whether it supports random access. If needed, the repair step converts a plain gzip FASTA to BGZF before indexing.

If only one GPU is available:

```bash
bash 1.glutinous.sh --gpus 0 --workers 1
```

The primary display figures are:

| Figure | Description |
|:--|:--|
| `Results/glutinous/figures/glutinous_differential_attention_padj.png` | Four-region glutinous adjusted-P differential attention signal |
| `Results/grain_length/smoke_test/figures/grain_length_consistency_manhattan_annotated.png` | Default 50-sample grain-length smoke association landscape with Chr3 gene annotation |
| `Results/grain_length/figures/grain_length_consistency_manhattan_annotated.png` | Full-sample grain-length association landscape, produced with `bash 1.grain_length.sh --full_sample` |

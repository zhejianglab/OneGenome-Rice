#!/usr/bin/env python3
"""Run the grain-length GS3 association display and optional matrix-based model."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "0420_display_mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def rel(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(config_path: Path) -> tuple[Path, dict]:
    root = config_path.resolve().parent
    return root, json.loads(config_path.read_text(encoding="utf-8"))


def results_root(root: Path, cfg: dict) -> Path:
    return rel(root, cfg["paths"]["results_dir"])


def grain_length_results(root: Path, cfg: dict) -> Path:
    return results_root(root, cfg) / "grain_length"


def standardize_id(df: pd.DataFrame) -> pd.DataFrame:
    id_map = {
        "sample_id": "SampleID",
        "Sample_ID": "SampleID",
        "sampleID": "SampleID",
        "ID": "SampleID",
        "id": "SampleID",
    }
    df = df.rename(columns=id_map)
    if "SampleID" not in df.columns:
        raise ValueError("Input table must contain SampleID or sample_id")
    df["SampleID"] = df["SampleID"].astype(str).str.strip()
    return df


def read_pheno(pheno_file: Path, pheno_col: str) -> pd.DataFrame:
    if not pheno_file.is_file():
        raise FileNotFoundError(f"Phenotype file not found: {pheno_file}")
    pheno = pd.read_csv(pheno_file, sep=r"\s+|,", engine="python", dtype=str)
    pheno = standardize_id(pheno)
    if pheno_col not in pheno.columns:
        raise ValueError(f"Phenotype column '{pheno_col}' not found in {pheno_file}")
    print(f"Phenotype samples: {len(pheno)}; column: {pheno_col}")
    return pheno


def trim_csv_columns(input_csv: Path, output_csv: Path, slice_start: int, slice_end: int) -> None:
    i0 = slice_start - 1
    i1 = slice_end
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", newline="", encoding="utf-8") as fin, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow([row[0]] + row[i0:i1])


def matrix_inputs_ready(matrix_dir: Path, start_block: int, end_block: int) -> bool:
    if not matrix_dir.is_dir():
        return False
    for block_idx in range(start_block, end_block + 1):
        block_dir = matrix_dir / f"block_{block_idx}"
        if not (block_dir / "hap1_attention_collapsed.csv").exists():
            return False
        if not (block_dir / "hap1_attention_collapsed_revcomp.csv").exists():
            return False
    return True


def middle_blocks_ready(block_4k_root: Path, start_block: int, end_block: int) -> bool:
    if not block_4k_root.is_dir():
        return False
    for block_idx in range(start_block, end_block + 1):
        block_dir = block_4k_root / f"block_{block_idx}"
        if not (block_dir / "hap1_attention_collapsed_4k.csv").exists():
            return False
        if not (block_dir / "hap1_attention_collapsed_revcomp_4k.csv").exists():
            return False
    return True


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}; see {log_path}")


def parse_gpus(gpu_arg: str | None) -> list[str]:
    gpu_list = gpu_arg
    if gpu_list is None:
        gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return [item.strip() for item in gpu_list.split(",") if item.strip()] or ["0"]


def split_block_ranges(start_block: int, end_block: int, shards: int) -> list[tuple[int, int]]:
    block_count = end_block - start_block + 1
    shard_count = max(1, min(shards, block_count))
    ranges: list[tuple[int, int]] = []
    next_start = start_block
    for shard_idx in range(shard_count):
        remaining_blocks = end_block - next_start + 1
        remaining_shards = shard_count - shard_idx
        size = (remaining_blocks + remaining_shards - 1) // remaining_shards
        shard_end = next_start + size - 1
        ranges.append((next_start, shard_end))
        next_start = shard_end + 1
    return ranges


def ensure_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Full grain-length attention requires CUDA, but torch.cuda.is_available() is False.")


def run_attention_from_vcf(
    root: Path,
    cfg: dict,
    result_dir: Path,
    pheno_file: Path,
    pheno_col: str,
    output_matrix_dir: Path,
    gpus: list[str],
    workers: int,
    start_block: int,
    end_block: int,
    sample_limit: int,
    sample_seed: int,
    env: dict[str, str],
) -> Path:
    paths = cfg["paths"]
    scripts = root / "Scripts" / "lib"
    attention_root = result_dir / "attention"
    out1 = attention_root / "01_pseudo_sequences"
    out2 = attention_root / "02_raw_attention"
    out3 = attention_root / "03_normalized_attention"
    out4 = output_matrix_dir
    for path in (out1, out2, out3, out4):
        path.mkdir(parents=True, exist_ok=True)
    ensure_cuda_available()

    py = sys.executable
    run_logged(
        [
            py,
            str(scripts / "generate_jsonl_indel_rice.py"),
            "--bed",
            str(rel(root, paths["grain_length_regions"])),
            "--pheno",
            str(pheno_file),
            "--vcf",
            str(rel(root, paths["grain_length_vcf"])),
            "--fasta",
            str(rel(root, paths["reference_fasta"])),
            "--pheno-col",
            pheno_col,
            "--out",
            str(out1),
            "--sample-limit",
            str(sample_limit),
            "--sample-seed",
            str(sample_seed),
        ],
        out1 / "run.log",
        env,
    )

    block_ranges = split_block_ranges(start_block, end_block, workers)
    print(
        "Running grain-length attention with "
        f"{len(block_ranges)} worker(s); GPUs={','.join(gpus)}; "
        f"blocks={start_block}-{end_block}",
        flush=True,
    )

    def run_attention_shard(shard_idx: int, block_start: int, block_end: int, gpu_id: str) -> None:
        model_env = env.copy()
        model_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            py,
            str(scripts / "calc_flash_attention_run.py"),
            "--model_path",
            str(rel(root, paths["model"])),
            "--input_dir",
            str(out1),
            "--output_dir",
            str(out2),
            "--bi_direction",
            "--block_start",
            str(block_start),
            "--block_end",
            str(block_end),
        ]
        log_path = out2 / f"run.gpu{gpu_id}.blocks_{block_start}_{block_end}.log"
        run_logged(cmd, log_path, model_env)
        print(f"grain_length blocks {block_start}-{block_end} finished on GPU {gpu_id}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(block_ranges)) as executor:
        futures = []
        for shard_idx, (block_start, block_end) in enumerate(block_ranges, start=1):
            gpu_id = gpus[(shard_idx - 1) % len(gpus)]
            futures.append(executor.submit(run_attention_shard, shard_idx, block_start, block_end, gpu_id))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    run_logged(
        [
            py,
            str(scripts / "atten_score_indel_normalizeV2.py"),
            "--json_dir",
            str(out2),
            "--output_dir",
            str(out3),
            "--bed_file",
            str(rel(root, paths["grain_length_regions"])),
        ],
        out3 / "run.log",
        env,
    )

    run_logged(
        [
            py,
            str(scripts / "convert_json_to_matrix_perblock.py"),
            "--json_dir",
            str(out3),
            "--output_dir",
            str(out4),
            "--bed_file",
            str(rel(root, paths["grain_length_regions"])),
        ],
        out4 / "run.log",
        env,
    )
    return out4


def extract_middle_blocks(
    matrix_dir: Path,
    block_4k_root: Path,
    start_block: int,
    end_block: int,
    slice_start: int,
    slice_end: int,
) -> Path:
    if not matrix_inputs_ready(matrix_dir, start_block, end_block):
        raise FileNotFoundError(f"Complete grain-length matrix inputs were not found under {matrix_dir}")
    block_4k_root.mkdir(parents=True, exist_ok=True)
    for block_idx in range(start_block, end_block + 1):
        block_tag = f"block_{block_idx}"
        block_dir = matrix_dir / block_tag
        for source_name, target_name in [
            ("hap1_attention_collapsed.csv", "hap1_attention_collapsed_4k.csv"),
            ("hap1_attention_collapsed_revcomp.csv", "hap1_attention_collapsed_revcomp_4k.csv"),
        ]:
            trim_csv_columns(block_dir / source_name, block_4k_root / block_tag / target_name, slice_start, slice_end)
        print(f"Extracted middle columns for {block_tag}", flush=True)
    return block_4k_root


def merge_files(filename: str, base_dir: Path, blocks: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for block in blocks:
        file_path = base_dir / block / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing block matrix: {file_path}")
        df = standardize_id(pd.read_csv(file_path, dtype=str))
        merged = df if merged is None else merged.merge(df, on="SampleID", how="inner")
    if merged is None:
        raise ValueError("No block matrices were merged")
    print(f"Merged {filename}: {merged.shape[0]} samples x {merged.shape[1] - 1} positions")
    return merged


def compute_pcs(attn_sub: pd.DataFrame, n_pcs: int) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    values = (
        attn_sub.drop(columns=["SampleID"])
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values
    )
    n_components = min(n_pcs, values.shape[0], values.shape[1])
    if n_components < 1:
        return pd.DataFrame(index=attn_sub.index)
    pcs = PCA(n_components=n_components).fit_transform(values)
    return pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(n_components)])


def run_association(attn_df: pd.DataFrame, pheno_df: pd.DataFrame, pheno_col: str, n_pcs: int) -> pd.DataFrame:
    import statsmodels.api as sm

    common_ids = sorted(set(attn_df["SampleID"]) & set(pheno_df["SampleID"]))
    if not common_ids:
        raise ValueError("No common SampleID found between attention matrix and phenotype")

    attn_sub = attn_df[attn_df["SampleID"].isin(common_ids)].sort_values("SampleID").reset_index(drop=True)
    pheno_sub = pheno_df[pheno_df["SampleID"].isin(common_ids)].sort_values("SampleID").reset_index(drop=True)
    y_all = pd.to_numeric(pheno_sub[pheno_col], errors="coerce").values
    valid_idx = ~np.isnan(y_all)
    y = y_all[valid_idx]
    attn_sub = attn_sub.iloc[valid_idx].reset_index(drop=True)
    features = attn_sub.drop(columns=["SampleID"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    pc_df = compute_pcs(attn_sub, n_pcs=n_pcs)

    pvals: list[float] = []
    betas: list[float] = []
    for col in tqdm(features.columns, desc="Association", unit="pos"):
        design = pd.DataFrame({"attention": np.nan_to_num(features[col].values)})
        for pc in pc_df.columns:
            design[pc] = pc_df[pc].values
        design = sm.add_constant(design)
        try:
            model = sm.OLS(y, design).fit()
            pvals.append(float(model.pvalues["attention"]))
            betas.append(float(model.params["attention"]))
        except Exception:
            pvals.append(np.nan)
            betas.append(np.nan)
    return pd.DataFrame({"feature": features.columns, "beta": betas, "pval": pvals})


def consistency_combined_pvalue(p1: np.ndarray, p2: np.ndarray, beta1: np.ndarray, beta2: np.ndarray) -> np.ndarray:
    p1 = np.clip(p1, 1e-300, 1)
    p2 = np.clip(p2, 1e-300, 1)
    geo_mean = np.sqrt(p1 * p2)
    consistency = np.clip(np.minimum(p1, p2) / np.maximum(p1, p2), 1e-12, 1)
    combined_p = geo_mean / consistency
    combined_p[np.sign(beta1) != np.sign(beta2)] = 1
    return np.clip(combined_p, 1e-300, 1)


def run_linear_model(
    block_4k_root: Path,
    pheno_file: Path,
    pheno_col: str,
    n_pcs: int,
    start_block: int,
    end_block: int,
    table_dir: Path,
) -> pd.DataFrame:
    blocks = [f"block_{i}" for i in range(start_block, end_block + 1)]
    pheno = read_pheno(pheno_file, pheno_col)
    forward = run_association(merge_files("hap1_attention_collapsed_4k.csv", block_4k_root, blocks), pheno, pheno_col, n_pcs)
    reverse = run_association(merge_files("hap1_attention_collapsed_revcomp_4k.csv", block_4k_root, blocks), pheno, pheno_col, n_pcs)

    merged = forward.rename(columns={"beta": "forward_beta", "pval": "forward_p"})
    merged["reverse_beta"] = reverse["beta"].values
    merged["reverse_p"] = reverse["pval"].values
    merged["combined_p"] = consistency_combined_pvalue(
        merged["forward_p"].values,
        merged["reverse_p"].values,
        merged["forward_beta"].values,
        merged["reverse_beta"].values,
    )
    merged["position"] = merged["feature"].str.replace("pos_", "", regex=False).astype(int)
    merged = merged.sort_values("position")
    table_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(table_dir / "observed_consistency_pvalues.csv", index=False)
    return merged


def parse_gff_attributes(attr_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in attr_text.strip().split(";"):
        if item and "=" in item:
            key, value = item.split("=", 1)
            attrs[key.strip()] = value.strip().strip('"')
    return attrs


def normalize_chrom_name(chrom: str | int) -> str:
    value = str(chrom).strip().lower()
    value = re.sub(r"^chrom", "chr", value)
    match = re.match(r"^chr0*(\d+)$", value)
    if match:
        return f"chr{int(match.group(1))}"
    match = re.match(r"^0*(\d+)$", value)
    if match:
        return f"chr{int(match.group(1))}"
    return value


def parse_gff3_genes(gff_path: Path, chrom: str | int) -> list[dict[str, str | int]]:
    if not gff_path.exists():
        raise FileNotFoundError(f"GFF3 file not found: {gff_path}")
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    chrom_key = normalize_chrom_name(chrom)
    genes: list[dict[str, str | int]] = []
    with opener(gff_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            if normalize_chrom_name(parts[0]) != chrom_key:
                continue
            attrs = parse_gff_attributes(parts[8])
            label = attrs.get("Alias") or attrs.get("Name") or attrs.get("ID") or ""
            genes.append({"start": int(parts[3]), "end": int(parts[4]), "id": label})
    return genes


def prepare_plot_df(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    if "position" not in df.columns:
        df["position"] = df["feature"].str.replace("pos_", "", regex=False).astype(int)
    df["combined_p"] = pd.to_numeric(df["combined_p"], errors="coerce").clip(lower=1e-300, upper=1)
    df["score"] = -np.log10(df["combined_p"])
    return df[df["position"].notna() & df["score"].notna()].sort_values("position")


def plot_annotated(merged: pd.DataFrame, out_png: Path, gff_file: Path, chrom: str | int) -> Path:
    df = prepare_plot_df(merged)
    x_bp = df["position"].values
    x_mb = x_bp / 1_000_000
    y = df["score"].values
    genes = parse_gff3_genes(gff_file, chrom)
    plot_genes = [g for g in genes if int(g["end"]) >= x_bp.min() and int(g["start"]) <= x_bp.max()]

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.vlines(x=x_mb, ymin=0, ymax=y, linewidth=0.4, color=(72 / 255, 116 / 255, 203 / 255))
    if len(y):
        max_idx = int(np.argmax(y))
        ax.scatter(x_mb[max_idx], y[max_idx], color=(238 / 255, 130 / 255, 47 / 255), s=50, zorder=5)
        ax.annotate(
            f"Pos: {int(x_bp[max_idx])}",
            xy=(x_mb[max_idx], y[max_idx]),
            xytext=(x_mb[max_idx] + 0.003, y[max_idx] + 0.2),
            ha="center",
            fontsize=14,
            color=(238 / 255, 130 / 255, 47 / 255),
            fontweight="bold",
        )

    y_max = max(float(np.nanmax(y)), 1.0) if len(y) else 1.0
    gene_y = -y_max * 0.05
    gene_color = (242 / 255, 186 / 255, 2 / 255)
    for i, gene in enumerate(plot_genes):
        start_mb = int(gene["start"]) / 1_000_000
        end_mb = int(gene["end"]) / 1_000_000
        mid_mb = (start_mb + end_mb) / 2
        ax.hlines(y=gene_y, xmin=start_mb, xmax=end_mb, color=gene_color, linewidth=8, zorder=10)
        offset = 1.5 if i % 2 == 0 else 3.5
        ax.text(mid_mb, gene_y * offset, str(gene["id"]), ha="center", va="top", fontsize=10, color=gene_color)

    ax.set_title("Grain-Length Bidirectional Attention Association Landscape", loc="left", x=0.03, y=0.95, fontsize=18)
    ax.set_xlabel("Genomic Position (Mb)", x=1.0, ha="right", fontsize=16)
    ax.set_ylabel("Consistency-weighted Association Score", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(gene_y * 8, y_max * 1.1)
    ax.tick_params(axis="both", which="major", labelsize=14)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Annotated {len(plot_genes)} Chr{chrom} genes")
    return out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grain-length display and PCA linear-model rerun.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["precomputed", "model"], default="model")
    parser.add_argument("--matrix-dir", help="Directory containing block_1..block_N attention matrices.")
    parser.add_argument("--observed-csv", help="Optional precomputed observed_consistency_pvalues.csv for --mode precomputed.")
    parser.add_argument("--pheno-file", help="Phenotype table used when --mode model is selected.")
    parser.add_argument("--pheno-col", help="Phenotype column name.")
    parser.add_argument("--gff-file", help="Chr3 GFF3 file for annotation.")
    parser.add_argument("--table-dir", help="Output directory for grain-length tables.")
    parser.add_argument("--out-png", help="Output PNG path.")
    parser.add_argument("--block-4k-dir", help="Output directory for trimmed middle-4k matrices.")
    parser.add_argument("--skip-extract", action="store_true", help="Use an existing --block-4k-dir in model mode.")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs for full attention generation. Default: CUDA_VISIBLE_DEVICES or 0.")
    parser.add_argument("--workers", type=int, default=None, help="Number of attention shards to run concurrently. Default: number of GPUs.")
    parser.add_argument("--force-attention", action="store_true", help="Regenerate grain-length attention matrices from the VCF before model mode.")
    parser.add_argument("--smoke-test", action="store_true", help="Compatibility flag; grain-length model mode uses the 50-sample smoke workflow by default.")
    parser.add_argument("--full_sample", "--full-sample", action="store_true", help="Run the formal grain-length workflow on all matched samples.")
    parser.add_argument("--smoke-samples", type=int, default=50, help="Number of samples to keep in the default smoke workflow.")
    parser.add_argument("--sample-seed", type=int, default=20260420, help="Random seed used by smoke-workflow sample selection.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root, cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    gl_cfg = cfg["workflow"].get("grain_length", {})
    results_base = results_root(root, cfg)
    results = grain_length_results(root, cfg)
    smoke_mode = args.mode == "model" and not args.full_sample
    if args.full_sample and args.smoke_test:
        print("--full_sample requested; ignoring --smoke-test.", flush=True)
    run_results = results / "smoke_test" if smoke_mode else results
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(results_base / ".mplconfig"))

    start_block = int(gl_cfg.get("start_block", 1))
    end_block = int(gl_cfg.get("end_block", 10))
    slice_start = int(gl_cfg.get("slice_start", 2002))
    slice_end = int(gl_cfg.get("slice_end", 6001))
    n_pcs = int(gl_cfg.get("n_pcs", 5))
    chrom = gl_cfg.get("chromosome", 3)

    matrix_dir = rel(
        root,
        args.matrix_dir
        or (run_results / "attention" / "04_attention_matrices" if smoke_mode else paths.get("grain_length_matrix_dir", "Data/grain_length_attention_matrices")),
    )
    observed_csv = rel(root, args.observed_csv) if args.observed_csv else None
    pheno_file = rel(root, args.pheno_file or paths.get("grain_length_phenotype", paths["phenotype"]))
    pheno_col = args.pheno_col or gl_cfg.get("phenotype_column", "grain_length")
    gff_file = rel(root, args.gff_file or paths.get("grain_length_gene_annotation", "Data/chr03.gff3.gz"))
    table_dir = rel(root, args.table_dir or run_results / "tables")
    out_png = rel(root, args.out_png or run_results / "figures" / "grain_length_consistency_manhattan_annotated.png")
    block_4k_dir = rel(root, args.block_4k_dir or run_results / "attention" / "block_4k")
    for path in (run_results, run_results / "attention", table_dir, out_png.parent, block_4k_dir):
        path.mkdir(parents=True, exist_ok=True)
    sample_limit = max(0, int(args.smoke_samples)) if smoke_mode else 0
    if smoke_mode:
        print(
            f"Smoke workflow enabled: using {sample_limit} random sample(s); outputs under {run_results}",
            flush=True,
        )

    if args.mode == "model":
        use_existing_middle = middle_blocks_ready(block_4k_dir, start_block, end_block)
        if args.force_attention or not matrix_inputs_ready(matrix_dir, start_block, end_block):
            gpus = parse_gpus(args.gpus)
            workers = args.workers or len(gpus)
            workers = max(1, min(workers, end_block - start_block + 1))
            print("Grain-length attention matrices not found; generating them from the VCF.", flush=True)
            matrix_dir = run_attention_from_vcf(
                root,
                cfg,
                run_results,
                pheno_file,
                pheno_col,
                matrix_dir,
                gpus,
                workers,
                start_block,
                end_block,
                sample_limit,
                args.sample_seed,
                env,
            )
            use_existing_middle = False
        if args.skip_extract:
            block_root = block_4k_dir
            if not block_root.is_dir():
                raise FileNotFoundError(f"--skip-extract requested but missing: {block_root}")
        elif use_existing_middle:
            block_root = block_4k_dir
            print(f"Reusing existing grain-length middle-4k matrices: {block_root}")
        else:
            block_root = extract_middle_blocks(matrix_dir, block_4k_dir, start_block, end_block, slice_start, slice_end)
        merged = run_linear_model(block_root, pheno_file, pheno_col, n_pcs, start_block, end_block, table_dir)
    else:
        if observed_csv is None or not observed_csv.is_file():
            raise FileNotFoundError("--mode precomputed requires --observed-csv pointing to an existing CSV")
        table_dir.mkdir(parents=True, exist_ok=True)
        target_csv = table_dir / "observed_consistency_pvalues.csv"
        if observed_csv.resolve() != target_csv.resolve():
            shutil.copy2(observed_csv, target_csv)
        merged = pd.read_csv(observed_csv)
        print(f"Loaded precomputed grain-length association table: {observed_csv}")

    plot_annotated(merged, out_png, gff_file, chrom)
    print(f"Grain-length figure written to {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

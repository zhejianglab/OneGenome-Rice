#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-24
"""
ATAC-conditioned RNA inference script driven by a YAML config file.

Usage:
    python inference.py <config>.yaml
    torchrun --standalone --nproc_per_node=4 inference.py <config>.yaml

Multi-GPU: launch with ``torchrun`` (NCCL). Each rank processes a disjoint shard of
windows; rank 0 merges shards and writes outputs. Do not set ``WORLD_SIZE`` manually.

Model components (`build_multimodal_model`, `InferenceDataset`, `LabelScaler`)
are imported from the `model/` package to keep training and inference
architecturally in sync. Set `predictor.type: fusion` (default if omitted).
"""

import argparse
import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

from model.dataset import InferenceDataset as PairedWindowInferenceDataset
from model.distributed import setup_distributed
from model.config import effective_inference_batch_size, parse_dataset_block
from model.index import build_index as _build_index
from model.load_pretrained import load_model_and_tokenizer
from model.pipeline import (
    build_multimodal_model,
    _normalize_predictor_type,
)
from model.scaling import LabelScaler

from calc_metrics import calculate_and_save_metrics

InferenceDatasetType = PairedWindowInferenceDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Output serialization
# =============================================================================

def save_result(plus_results, minus_results, output_dir, save_csv=False, save_pickle=False):
    def build_dataframes(results, stringify_arrays=False):
        dfs = {}
        for batch, res in results.items():
            rows = []
            for r in res:
                row = dict(r)
                if stringify_arrays:
                    row["true_expression"] = str(np.asarray(row["true_expression"]).tolist())
                    row["predicted_expression"] = str(np.asarray(row["predicted_expression"]).tolist())
                rows.append(row)
            dfs[batch] = pd.DataFrame(rows)
        return dfs

    plus_dfs = build_dataframes(plus_results)
    minus_dfs = build_dataframes(minus_results)

    if save_csv:
        plus_csv_dfs = build_dataframes(plus_results, stringify_arrays=True)
        minus_csv_dfs = build_dataframes(minus_results, stringify_arrays=True)
        for batch, df in plus_csv_dfs.items():
            path = os.path.join(output_dir, f"{batch}_plus_predictions.csv")
            df.to_csv(path, index=False)
            logger.info(f"Saved + strand: {path} ({len(df)} samples)")
        for batch, df in minus_csv_dfs.items():
            path = os.path.join(output_dir, f"{batch}_minus_predictions.csv")
            df.to_csv(path, index=False)
            logger.info(f"Saved - strand: {path} ({len(df)} samples)")

    if save_pickle:
        fmt_text = (
            "Pickle format:\n"
            "- Object type: dict[str, pandas.DataFrame]\n"
            "- Dict key: batch name\n"
            "- DataFrame columns:\n"
            "  chromosome (str), start (int), end (int), sequence (str),\n"
            "  true_expression (numpy.ndarray, shape=(L,), dtype=float),\n"
            "  predicted_expression (numpy.ndarray, shape=(L,), dtype=float)\n"
            "- Arrays remain numpy format in pickle (not stringified)\n"
        )
        for strand, dfs in (("plus", plus_dfs), ("minus", minus_dfs)):
            p = os.path.join(output_dir, f"{strand}_predictions.pickle")
            with open(p, "wb") as fh:
                pickle.dump(dfs, fh)
            with open(p + ".fmt", "w") as fh:
                fh.write(fmt_text)


def _merge_shard_dicts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-rank ``dict[batch_name, list[rows]]`` into one dict (concatenate lists)."""
    merged: Dict[str, Any] = {}
    for p in parts:
        for k, rows in p.items():
            merged.setdefault(k, []).extend(rows)
    return merged


def _sort_result_rows(results: Dict[str, Any]) -> None:
    """Sort each batch's rows by genomic position (in-place)."""
    for rows in results.values():
        rows.sort(key=lambda r: (str(r["chromosome"]), int(r["start"]), int(r["end"])))


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="ATAC-conditioned RNA inference (YAML config)."
    )
    ap.add_argument(
        "config",
        metavar="YAML",
        help="Path to inference YAML config",
    )
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    from valid_config import validate_config_dict

    ok, verrors = validate_config_dict(cfg)
    if not ok:
        print("ERROR: config validation failed:", file=sys.stderr)
        for e in verrors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    CKPT_PATH  = cfg["ckpt_path"]
    OUTPUT_DIR = cfg["output_eval_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    local_rank, world_size, dist_enabled = setup_distributed()
    rank = dist.get_rank() if dist_enabled else 0
    if dist_enabled and not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU inference via torchrun requires CUDA (NCCL).")
    dataset: Optional[InferenceDatasetType] = None
    try:
        dataset = _run_inference_body(
            cfg,
            CKPT_PATH,
            OUTPUT_DIR,
            local_rank,
            world_size,
            dist_enabled,
            rank,
        )
    finally:
        if dataset is not None:
            dataset.close()
        if dist_enabled and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_inference_body(
    cfg: dict[str, Any],
    CKPT_PATH: str,
    OUTPUT_DIR: str,
    local_rank: int,
    world_size: int,
    dist_enabled: bool,
    rank: int,
) -> Optional[InferenceDatasetType]:
    save_csv     = cfg.get("save_csv", True)
    save_pickle  = cfg.get("save_pickle", True)
    calc_metrics = cfg.get("calc_metrics", False)
    metrics_plus_input  = cfg.get("metrics_plus_input",  os.path.join(OUTPUT_DIR, "plus_predictions.pickle"))
    metrics_minus_input = cfg.get("metrics_minus_input", os.path.join(OUTPUT_DIR, "minus_predictions.pickle"))
    metrics_output      = cfg.get("metrics_output",      os.path.join(OUTPUT_DIR, "strand_metrics.txt"))

    try:
        inference_batch_size = effective_inference_batch_size(cfg)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "inference_batch_size / batch_size must be integers (see effective_inference_batch_size)"
        ) from e
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be >= 1")

    # --- Model and tokenizer --------------------------------------------------
    if rank == 0:
        logger.info("Loading tokenizer and base model...")
    base_model, tokenizer = load_model_and_tokenizer(cfg)

    # --- Index ----------------------------------------------------------------
    if rank == 0:
        logger.info("Building genomic index...")
    (
        rna_files,
        atac_files,
        cell_types,
        chromosome,
        chromosome_per_cell_type,
        genome_fasta,
        genome_fasta_per_cell_type,
    ) = parse_dataset_block(cfg)
    target_len = int(cfg.get("target_len", 32000))
    overlap_len = int(cfg.get("overlap_len", target_len // 2))
    index_df = _build_index(
        genome_fasta,
        os.path.join(OUTPUT_DIR, "infer_index.csv"),
        rna_files,
        atac_files,
        cell_types,
        chromosome,
        chromosome_per_cell_type=chromosome_per_cell_type,
        fasta_path_per_cell_type=genome_fasta_per_cell_type,
        window_size=target_len,
        overlap=overlap_len,
        cap_expression_quantile=cfg.get("cap_expression_quantile"),
    )

    # --- Model ----------------------------------------------------------------
    if rank == 0:
        logger.info("Initializing model...")
    model = build_multimodal_model(cfg, base_model)
    ptype = _normalize_predictor_type(cfg)

    if dist_enabled:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if rank == 0:
        logger.info(f"Loading checkpoint: {CKPT_PATH}")
    from safetensors.torch import load_file as safe_load
    state_dict = safe_load(CKPT_PATH)
    model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        logger.info("Checkpoint loaded (non-strict mode)")

    model.eval()

    # --- Dataset ---
    dataset = PairedWindowInferenceDataset(index_df, tokenizer, max_length=target_len)

    # --- Inference loop -------------------------------------------------------
    n = len(dataset)
    local_indices = list(range(rank, n, world_size))
    n_local = len(local_indices)
    if rank == 0:
        logger.info(
            f"Starting inference on {n} regions (inference_batch_size={inference_batch_size}"
            f"{f', world_size={world_size}' if dist_enabled else ''})..."
        )
    plus_results, minus_results = {}, {}

    processed = 0
    n_batches = (n_local + inference_batch_size - 1) // inference_batch_size if n_local else 0
    for batch_start in tqdm(
        range(0, n_local, inference_batch_size),
        desc=f"Inference[r{rank}]",
        total=n_batches,
        miniters=10,
        mininterval=60,
        disable=(rank != 0),
    ):
        idx_chunk = local_indices[batch_start : batch_start + inference_batch_size]
        samples = [dataset[i] for i in idx_chunk]
        inp = torch.stack([s["input_ids"] for s in samples], dim=0).to(device)
        atac = torch.stack([s["atac_signal"] for s in samples], dim=0).to(device)

        with torch.no_grad():
            out = model(inp, atac)
            logits_b = out["logits"].float().cpu().numpy()

        for sample, logits in zip(samples, logits_b):
            L = len(sample["sequence"])
            scaler_p = LabelScaler(float(sample["track_mean_plus"]), None)
            scaler_m = LabelScaler(float(sample["track_mean_minus"]), None)
            pred_p = scaler_p.inverse_transform(logits[0, :L])
            pred_m = scaler_m.inverse_transform(logits[1, :L])
            true_p = np.array(sample["raw_rna_plus"][:L], dtype=np.float64, copy=True)
            true_m = np.array(sample["raw_rna_minus"][:L], dtype=np.float64, copy=True)

            chrom, start_pos, end_pos = sample["position"]
            seq = sample["sequence"]
            bp = sample["batch_name_plus"]
            bm = sample["batch_name_minus"]

            plus_results.setdefault(bp, []).append({
                "chromosome": chrom, "start": start_pos, "end": end_pos, "sequence": seq,
                "true_expression": np.array(true_p, copy=True),
                "predicted_expression": np.array(pred_p, copy=True),
            })

            minus_results.setdefault(bm, []).append({
                "chromosome": chrom, "start": start_pos, "end": end_pos, "sequence": seq,
                "true_expression": np.array(true_m, copy=True),
                "predicted_expression": np.array(pred_m, copy=True),
            })

        processed += len(samples)
        if rank == 0 and (processed % 100 == 0 or processed == n_local):
            logger.info(f"Processed {processed} / {n_local} samples (local shard)")

    if dist_enabled:
        logger.info(f"[rank{rank}] Gathering shard results to rank 0 (in RAM)...")
        gathered: Optional[List[Any]] = [None] * world_size if rank == 0 else None
        dist.gather_object((plus_results, minus_results), gathered, dst=0)
        if rank == 0:
            logger.info("Merging gathered shards...")
            plus_parts: List[Dict[str, Any]] = [g[0] for g in gathered]
            minus_parts: List[Dict[str, Any]] = [g[1] for g in gathered]
            plus_results = _merge_shard_dicts(plus_parts)
            minus_results = _merge_shard_dicts(minus_parts)
            _sort_result_rows(plus_results)
            _sort_result_rows(minus_results)
            logger.info("Shard merge complete.")

    # --- Save (rank 0 only); then sync so other ranks do not exit before save finishes ---
    if rank == 0:
        logger.info("Saving results...")
        save_result(
            plus_results=plus_results,
            minus_results=minus_results,
            output_dir=OUTPUT_DIR,
            save_csv=save_csv,
            save_pickle=save_pickle,
        )

        if calc_metrics:
            logger.info("Calculating metrics...")
            calculate_and_save_metrics(
                metrics_plus_input,
                metrics_minus_input,
                metrics_output,
            )

        logger.info(f"Inference complete. Results saved to: {OUTPUT_DIR}")

    if dist_enabled:
        dist.barrier()
    return dataset

if __name__ == "__main__":
    main()

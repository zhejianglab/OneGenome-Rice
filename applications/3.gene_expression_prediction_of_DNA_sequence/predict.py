import os
from math import sqrt
from pathlib import Path
import argparse  
import time
import numpy as np
import re
from glob import glob
import json

import pandas as pd
import torch
from safetensors.torch import load_file
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Union

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig

from src.util import setup_distributed, setup_seed, is_main_process, dist_print, setup_logging, get_index
from src.model import GenOmics, load_finetuned_model
from src.dataset import MultiTrackDataset


    
def predict_and_save_result(model, test_dataset, output_dir, batch_size, num_workers):


    os.makedirs(output_dir, exist_ok=True)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    device = torch.device(
        f"cuda:{rank}" if dist.is_initialized() and torch.cuda.is_available()
        else "cuda:0" if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device).eval()

    sampler = DistributedSampler(test_dataset, shuffle=False) if dist.is_initialized() else None
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    modalities = ["Panicle1_all_RNAseq-","Panicle1_all_RNAseq+"]

    tmp_dir = os.path.join(output_dir, f"tmp_rank_{rank}")
    os.makedirs(tmp_dir, exist_ok=True)

    def _safe_name(x: str) -> str:
            return re.sub(r"[^0-9A-Za-z_.+\-]", "_", str(x))

    def safe_int(x):
        """Try to coerce x to int, return None on failure (no exception)."""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            try:
                return int(x.item())
            except Exception:
                return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, str):
            s = re.sub(r"[^\d\-]", "", x)
            if s == "" or s == "-" or s == "+":
                return None
            try:
                return int(s)
            except Exception:
                return None
        try:
            return int(x)
        except Exception:
            return None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference rk{rank}", disable=not is_main_process()):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # call model (expected signature)
            try:
                out = model(input_ids=input_ids)
            except TypeError:
                out = model(input_ids=input_ids)

            # extract logits (tensor or dict)
            if isinstance(out, dict):
                logits = out.get("logits", out.get("predictions", None))
                if logits is None:
                    vals = [v for v in out.values() if isinstance(v, torch.Tensor)]
                    logits = vals[0] if vals else None
            elif isinstance(out, torch.Tensor):
                logits = out
            else:
                raise RuntimeError("Unexpected model output type")

            if logits is None:
                raise RuntimeError("Model did not return logits/predictions tensor")

            preds_np = logits.cpu().float().numpy()   # [B, L, C] or [B, C, L]
            labs_np = labels.cpu().numpy()           # [B, L, C] or [B, L]

            # normalize layout if needed
            if preds_np.ndim == 3 and preds_np.shape[1] == len(modalities) and preds_np.shape[2] != len(modalities):
                preds_np = np.transpose(preds_np, (0, 2, 1))
            if labs_np.ndim == 3 and labs_np.shape[1] == len(modalities) and labs_np.shape[2] != len(modalities):
                labs_np = np.transpose(labs_np, (0, 2, 1))

            B = preds_np.shape[0]
            for i in range(B):
                # parse position produced by dataset: dataset returns per-sample tuple (chrom, start, end)
                chrom = None; start = None; end = None
                if "position" in batch:
                    pos = batch["position"]
                    # detect tuple-of-lists: pos is (list_chroms, list_starts, list_ends)
                    is_tuple_of_lists = (
                        isinstance(pos, (tuple, list))
                        and len(pos) == 3
                        and (isinstance(pos[0], (list, tuple, np.ndarray, torch.Tensor))
                             or isinstance(pos[1], (list, tuple, np.ndarray, torch.Tensor))
                             or isinstance(pos[2], (list, tuple, np.ndarray, torch.Tensor)))
                    )
                    if is_tuple_of_lists:
                        try:
                            chrom = pos[0][i]
                            start = pos[1][i]
                            end = pos[2][i]
                        except Exception:
                            chrom = None; start = None; end = None
                    else:
                        # assume list-of-tuples or array-of-shape-(B,3): pos[i] is (chrom,start,end)
                        try:
                            cand = pos[i]
                            chrom, start, end = cand[0], cand[1], cand[2]
                        except Exception:
                            chrom = None; start = None; end = None
                else:
                    # separate position_* fields (not expected with current dataset but keep for robustness)
                    if "position_chrom" in batch:
                        pc = batch["position_chrom"][i]
                        chrom = pc.item() if isinstance(pc, torch.Tensor) else pc
                    if "position_start" in batch:
                        ps = batch["position_start"][i]
                        start = ps.item() if isinstance(ps, torch.Tensor) else ps
                    if "position_end" in batch:
                        pe = batch["position_end"][i]
                        end = pe.item() if isinstance(pe, torch.Tensor) else pe

                
                seq = batch.get("sequence", [None] * B)[i]

                preds_i = preds_np[i].copy()
                labs_i = labs_np[i].copy()

                if preds_i.ndim == 1:
                    preds_i = preds_i[:, None]
                if labs_i.ndim == 1:
                    labs_i = labs_i[:, None]

                C = preds_i.shape[-1]

                # stream-append to per-rank per-biosample+modality csv files (no cross-rank contention)
                for c_idx in range(C):
                    modality = modalities[c_idx] if c_idx < len(modalities) else f"ch{c_idx}"
                    entry = {
                        "chromosome": str(chrom.item() if isinstance(chrom, torch.Tensor) else chrom) if chrom is not None else None,
                        "start": safe_int(start),
                        "end": safe_int(end),
                        "sequence": seq,
                        "biosample":"biosample",
                        "modality": modality,
                        "predicted_expression": str(preds_i[:, c_idx].tolist()),
                        "true_expression": str(labs_i[:, c_idx].tolist() if labs_i.shape[-1] > c_idx else [])
                    }
                    fname = f"biosample__{_safe_name(modality)}__rk{rank}.csv"
                    fpath = os.path.join(tmp_dir, fname)
                    df = pd.DataFrame([entry])
                    df.to_csv(fpath, mode="a", header=not os.path.exists(fpath), index=False)

    # ensure all ranks finished writing
    if dist.is_initialized():
        dist.barrier()

    # main process: merge per-rank temp files into final per-biosample+modality CSVs
    if is_main_process():
        from collections import defaultdict
        tmp_pattern = os.path.join(output_dir, "tmp_rank_*", "*__*__rk*.csv")
        tmp_files = glob(tmp_pattern)
        grouped = defaultdict(list)
        for f in tmp_files:
            base = os.path.basename(f)
            try:
                bs, mod, _ = base.split("__", 2)
                biosample_name = bs
                modality_name = mod
            except Exception:
                df_tmp = pd.read_csv(f)
                if "biosample" in df_tmp.columns and "modality" in df_tmp.columns:
                    grouped_key = (str(df_tmp.loc[0, "biosample"]), str(df_tmp.loc[0, "modality"]))
                    grouped[grouped_key].append(f)
                    continue
                else:
                    continue
            grouped[(biosample_name, modality_name)].append(f)

        # merge and write final CSVs
        for (bs_name, mod_name), files in grouped.items():
            parts = []
            for f in files:
                try:
                    parts.append(pd.read_csv(f))
                except Exception:
                    continue
            if not parts:
                continue
            df_all = pd.concat(parts, ignore_index=True)
            # sort & dedup
            df_sorted = df_all.sort_values(by=["chromosome", "start"], ascending=[True, True])
            df_sorted = df_sorted.drop_duplicates(subset=["biosample", "chromosome", "start", "end", "modality"])
            out_fname = f"{Path(bs_name).stem}__{Path(mod_name).stem}_predictions.csv"
            out_path = os.path.join(output_dir, out_fname)
            df_sorted.to_csv(out_path, index=False)
            dist_print(f"[save] Merged and saved: {out_path} ({len(df_sorted)} rows)")

        # cleanup tmp dirs
        for r in range(world_size):
            td = os.path.join(output_dir, f"tmp_rank_{r}")
            try:
                for f in glob(os.path.join(td, "*")):
                    os.remove(f)
                if os.path.exists(td):
                    os.rmdir(td)
            except Exception:
                pass

    if dist.is_initialized():
        dist.barrier()

    dist_print("[done] Prediction finished")


def main():
    parser = argparse.ArgumentParser(description="RNA-seq Coverage Prediction CLI")

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Pretrained model path, e.g. /path/to/hyenadna-small-32k-seqlen-hf (HF format).")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Downstream checkpoint path (.bin or .safetensors).")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Enable FlashAttention-2.")
    parser.add_argument("--tokenizer_path", type=str,
                        help="Tokenizer path (HF format).")
    parser.add_argument("--proj_dim", type=int, default=1024,
                        help="U-Net input feature dimension.")
    parser.add_argument("--num_downsamples", type=int, default=4,
                        help="Number of downsampling blocks in the U-Net.")
    parser.add_argument("--bottleneck_dim", type=int, default=1536,
                        help="U-Net bottleneck dimension.")

    # Data
    parser.add_argument("--sequence_split_test", type=str, required=True,
                        help="Test split index CSV path.")
    parser.add_argument("--index_stat_json", type=str, required=True,
                        help="Index statistics JSON path (index_stat.json).")
    parser.add_argument("--bigWig_labels_meta", type=str, required=True,
                        help="BigWig labels metadata (CSV).")
    parser.add_argument("--test_chromosomes", type=str, nargs="+", default=["chr19"],
                        help="Chromosomes to predict, e.g. --test_chromosomes chr19 chr20")
    parser.add_argument("--max_predict_samples", type=int, default=None,
                        help="Debug: limit number of samples used for prediction (None means no limit).")

    # Output
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Output directory for predictions.")

    # Runtime
    parser.add_argument("--max_seq_len", type=int, default=32768,
                        help="Maximum sequence length (default: 32768).")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Inference batch size (default: 12).")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of DataLoader workers (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")

    args = parser.parse_args()

    try:
        setup_seed(args.seed)
    
        local_rank, world_size, is_distributed = setup_distributed()
        logfile_name = setup_logging(args.output_base_dir, log_filename="predict")
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        dist_print(f"[log] Writing logs to {logfile_name}")
        dist_print("[load] Loading model and tokenizer...")

        start_time = time.time()

        model = load_finetuned_model(
            model_class=GenOmics,
            model_path=args.model_path,
            ckpt_path=args.ckpt_path,
            use_flash_attn=args.use_flash_attn,
            device=device,
            torch_dtype=torch.bfloat16,
            model_init_kwargs={
                "index_stat":json.load(open(args.index_stat_json, "r")),
                "proj_dim":args.proj_dim,
                "num_downsamples":args.num_downsamples,
                "bottleneck_dim":args.bottleneck_dim
            }
        )

        end_time = time.time()
        loading_time = end_time - start_time
        dist_print(f"[load] Model loaded in {loading_time:.2f} seconds")

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=True,
            revision="main",
            padding_side='right'
        )
        dist_print("[load] Tokenizer loaded")

        dist_print(f"[data] Chromosomes to predict: {args.test_chromosomes}")

        dist_print("[data] Loading test index...")
        test_index_df = get_index(args.sequence_split_test)
        selected_test_index_df = test_index_df[test_index_df["chromosome"].str.extract(r'(Chr\d+)')[0].isin(args.test_chromosomes)].copy()
        if args.max_predict_samples is not None:
            selected_test_index_df = selected_test_index_df[:args.max_predict_samples]
        
        # Chromosomes are already defined in run_sequence_split_and_meta_extract.py, so no need to filter again here.
        #selected_test_index_df = test_index_df

        # If max_test_samples is set (legacy name from train.py), limit the number of test samples by random sampling.
        # if args.max_test_samples is not None:
        #     selected_test_index_df = selected_test_index_df.head(args.max_test_samples).reset_index(drop=True)
        #     dist_print(f"[debug] Predicting on the first {len(selected_test_index_df)} test samples")

        dist_print("[data] Building test dataset...")
        test_dataset = MultiTrackDataset(selected_test_index_df, 
                                        # pd.read_csv(args.bigWig_labels_meta), 
                                        json.load(open(args.index_stat_json, "r")),
                                        tokenizer, max_length=args.max_seq_len,mode="single")
        dist_print(f"[data] Test set size: {len(test_dataset):,} samples")

        dist_print("[infer] Starting prediction on test set...")
        predict_and_save_result(model, test_dataset, args.output_base_dir, args.batch_size, args.num_workers)
        dist_print("[infer] Prediction complete")

    except Exception as e:
        dist_print(f"[error] Error: {str(e)}")
        raise
    finally:
        for ds_name in ['train_dataset', 'val_dataset', 'test_dataset']:
            if ds_name in locals() and hasattr(locals()[ds_name], 'close'):
                locals()[ds_name].close()
                dist_print(f"[cleanup] Released resources: {ds_name}")

    dist_print("[done] Main process finished")


if __name__ == "__main__":
    main()

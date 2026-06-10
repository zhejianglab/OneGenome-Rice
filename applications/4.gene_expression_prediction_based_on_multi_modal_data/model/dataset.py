# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""Genomic datasets for training and inference."""

import traceback

import numpy as np
import pyBigWig
import pyfaidx
import torch
from torch.utils.data import Dataset

from model.distributed import dist_print, is_main_process
from model.scaling import LabelScaler, cap_threshold_from_index_row


class LazyGenomicDataset(Dataset):
    def __init__(self, index_df, tokenizer, max_length=32000):
        self.index_df = index_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self._fasta_cache = {}
        self._bw_cache = {}

        unique_paths = set()
        for _, row in index_df.iterrows():
            unique_paths.update(
                [row["rna_path_plus"], row["rna_path_minus"], row["atac_path"]]
            )
        for path in unique_paths:
            try:
                self._bw_cache[path] = pyBigWig.open(path, "r")
            except Exception as e:
                dist_print(f"[WARNING] Cannot open {path}: {e}")
                self._bw_cache[path] = None

    def _get_fasta(self, fasta_path: str):
        fasta_path = str(fasta_path)
        if fasta_path not in self._fasta_cache:
            self._fasta_cache[fasta_path] = pyfaidx.Fasta(fasta_path)
        return self._fasta_cache[fasta_path]

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        fasta = self._get_fasta(row["fasta_path"])
        target_len = int(self.max_length)

        try:
            bw_plus = self._bw_cache[row["rna_path_plus"]]
            bw_minus = self._bw_cache[row["rna_path_minus"]]
            bw_atac = self._bw_cache[row["atac_path"]]
            if not all([bw_plus, bw_minus, bw_atac]):
                raise ValueError("One or more BigWig handles not loaded")

            raw_seq = str(fasta[row["chromosome"]][row["start"] : row["end"]])
            raw_plus = np.array(bw_plus.values(row["chromosome"], row["start"], row["end"]))
            raw_minus = np.array(bw_minus.values(row["chromosome"], row["start"], row["end"]))
            raw_atac = np.array(bw_atac.values(row["chromosome"], row["start"], row["end"]))

            if len(raw_seq) > target_len:
                seq = raw_seq[:target_len]
            elif len(raw_seq) < target_len:
                seq = raw_seq + "N" * (target_len - len(raw_seq))
            else:
                seq = raw_seq

            def process_signal(raw_vals):
                vals = np.nan_to_num(raw_vals, nan=0.0)
                if len(vals) > target_len:
                    vals = vals[:target_len]
                elif len(vals) < target_len:
                    vals = np.pad(vals, (0, target_len - len(vals)), constant_values=0.0)
                return vals

            plus_vals = process_signal(raw_plus)
            minus_vals = process_signal(raw_minus)

            atac_vals = np.nan_to_num(raw_atac, nan=0.0)
            if atac_vals.size > 0:
                atac_vals = np.clip(atac_vals, 0, np.percentile(atac_vals, 99))
            else:
                atac_vals = np.zeros(target_len)
            atac_vals = process_signal(atac_vals)

            scaler_p = LabelScaler(
                float(row["track_mean_plus"]),
                cap_threshold_from_index_row(row, "cap_threshold_plus"),
            )
            scaler_m = LabelScaler(
                float(row["track_mean_minus"]),
                cap_threshold_from_index_row(row, "cap_threshold_minus"),
            )
            scaled_labels = np.stack(
                [scaler_p.transform(plus_vals), scaler_m.transform(minus_vals)], axis=0
            )

            encodings = self.tokenizer(
                seq,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=False,
            )

            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "atac_signal": torch.tensor(atac_vals, dtype=torch.bfloat16),
                "labels": torch.tensor(scaled_labels, dtype=torch.float32),
            }

        except Exception as e:
            if is_main_process():
                print(
                    "[CRITICAL WARNING] Failed to load sample "
                    f"idx={idx} | type={type(e).__name__} | error={repr(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
            seq = "N" * target_len
            encodings = self.tokenizer(
                seq,
                padding="max_length",
                max_length=target_len,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=False,
            )
            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "atac_signal": torch.zeros(target_len, dtype=torch.bfloat16),
                "labels": torch.zeros(2, target_len, dtype=torch.float32),
            }

    def close(self):
        for fa in self._fasta_cache.values():
            try:
                fa.close()
            except Exception:
                pass
        self._fasta_cache.clear()
        for bw in self._bw_cache.values():
            if bw is not None:
                try:
                    bw.close()
                except Exception:
                    pass
        self._bw_cache.clear()


def _process_signal(raw_vals, target_len):
    vals = np.nan_to_num(raw_vals, nan=0.0)
    if len(vals) > target_len:
        vals = vals[:target_len]
    elif len(vals) < target_len:
        vals = np.pad(vals, (0, target_len - len(vals)), constant_values=0.0)
    return vals


class InferenceDataset:
    """Dataset for inference: same signal loading as LazyGenomicDataset but returns
    additional metadata needed to build output records (position, batch names, track means)."""

    def __init__(self, index_df, tokenizer, max_length=32000):
        self.df = index_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.target_len = int(max_length)
        self._fasta_cache = {}
        self._bw_cache = {}

        paths = set()
        for _, row in self.df.iterrows():
            paths.update([row["rna_path_plus"], row["rna_path_minus"], row["atac_path"]])
        for path in paths:
            self._bw_cache[path] = pyBigWig.open(path, "r")

    def _get_fasta(self, fasta_path: str):
        fasta_path = str(fasta_path)
        if fasta_path not in self._fasta_cache:
            self._fasta_cache[fasta_path] = pyfaidx.Fasta(fasta_path)
        return self._fasta_cache[fasta_path]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        L = self.target_len
        fasta = self._get_fasta(row["fasta_path"])

        chrom, start, end = row["chromosome"], row["start"], row["end"]
        raw_seq = str(fasta[chrom][start:end])
        raw_plus = np.array(self._bw_cache[row["rna_path_plus"]].values(chrom, start, end))
        raw_minus = np.array(self._bw_cache[row["rna_path_minus"]].values(chrom, start, end))
        raw_atac = np.array(self._bw_cache[row["atac_path"]].values(chrom, start, end))

        seq = raw_seq[:L] if len(raw_seq) > L else raw_seq + "N" * (L - len(raw_seq))

        plus_vals = _process_signal(raw_plus, L)
        minus_vals = _process_signal(raw_minus, L)

        atac_clipped = np.clip(
            np.nan_to_num(raw_atac, nan=0.0), 0,
            np.percentile(raw_atac, 99) if raw_atac.size > 0 else 0.0,
        )
        atac_vals = _process_signal(atac_clipped, L)

        scaler_p = LabelScaler(
            float(row["track_mean_plus"]),
            cap_threshold_from_index_row(row, "cap_threshold_plus"),
        )
        scaler_m = LabelScaler(
            float(row["track_mean_minus"]),
            cap_threshold_from_index_row(row, "cap_threshold_minus"),
        )
        scaled_labels = np.stack(
            [scaler_p.transform(plus_vals), scaler_m.transform(minus_vals)], axis=0
        )

        enc = self.tokenizer(
            seq, padding="max_length", max_length=L,
            truncation=True, return_tensors="pt", return_attention_mask=False,
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "atac_signal": torch.tensor(atac_vals, dtype=torch.bfloat16),
            "labels": torch.tensor(scaled_labels, dtype=torch.float32),
            "track_mean_plus": row["track_mean_plus"],
            "track_mean_minus": row["track_mean_minus"],
            "raw_rna_plus": plus_vals.astype(np.float32, copy=False),
            "raw_rna_minus": minus_vals.astype(np.float32, copy=False),
            "sequence": seq,
            "position": (chrom, int(start), int(end)),
            "batch_name_plus": row["batch_name_plus"],
            "batch_name_minus": row["batch_name_minus"],
        }

    def close(self):
        for fa in self._fasta_cache.values():
            try:
                fa.close()
            except Exception:
                pass
        self._fasta_cache.clear()
        for bw in self._bw_cache.values():
            try:
                bw.close()
            except Exception:
                pass
        self._bw_cache.clear()

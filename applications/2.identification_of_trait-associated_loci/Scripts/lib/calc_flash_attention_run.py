#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory-efficient streaming attention extraction.
1. Uses causal attention for decoder-only models.
2. Supports bidirectional extraction with precomputed coordinate arrays.
3. Takes the model path explicitly from the command line.
"""

import argparse
import json
import math
import re
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

captured: Dict[str, torch.Tensor] = {}

@torch.no_grad()
def attn_column_sums_streaming(q, k, causal=True, block_rows=1024):
    """
    Return [B, L] column sums after averaging across attention heads.

    This uses a two-pass query-row chunking algorithm:
      Pass 1 accumulates the global row max and normalization denominator.
      Pass 2 computes exact probabilities from the complete max/denominator terms.

    This avoids the bias introduced by normalizing each key-column block before
    the full denominator has been accumulated.

    q, k: [B, H, L, D]
    block_rows: query rows per chunk; controls memory use without changing results.
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)

    q = q.to(torch.float32)
    k = k.to(torch.float32)

    out = torch.zeros((B, L), dtype=torch.float32, device=q.device)

    for b in range(B):
        # K_b: [H, D, L] after transpose for matrix multiplication.
        K_b = k[b].transpose(-1, -2)

        m = torch.full((H, L), -float('inf'), dtype=torch.float32, device=q.device)
        l = torch.zeros((H, L), dtype=torch.float32, device=q.device)

        for i0 in range(0, L, block_rows):
            i1 = min(i0 + block_rows, L)
            Q_chunk = q[b, :, i0:i1, :]               # [H, Br, D]
            S = torch.matmul(Q_chunk, K_b) * scale    # [H, Br, L]

            if causal:
                row_idx = torch.arange(i0, i1, device=q.device).view(1, -1, 1)
                col_idx = torch.arange(L, device=q.device).view(1, 1, -1)
                S = S.masked_fill(col_idx > row_idx, float('-inf'))

            block_max = S.max(dim=-1).values           # [H, Br]
            prev_m    = m[:, i0:i1]                    # [H, Br]
            new_m     = torch.maximum(prev_m, block_max)

            exp_term  = torch.exp(S - new_m.unsqueeze(-1))  # [H, Br, L]

            l[:, i0:i1] = (
                l[:, i0:i1] * torch.exp(prev_m - new_m)
                + exp_term.sum(dim=-1)
            )
            m[:, i0:i1] = new_m

            del S, exp_term

        col_sum_heads = torch.zeros((H, L), dtype=torch.float32, device=q.device)

        for i0 in range(0, L, block_rows):
            i1 = min(i0 + block_rows, L)
            Q_chunk = q[b, :, i0:i1, :]
            S = torch.matmul(Q_chunk, K_b) * scale

            if causal:
                row_idx = torch.arange(i0, i1, device=q.device).view(1, -1, 1)
                col_idx = torch.arange(L, device=q.device).view(1, 1, -1)
                S = S.masked_fill(col_idx > row_idx, float('-inf'))

            P = torch.exp(S - m[:, i0:i1].unsqueeze(-1)) / (
                l[:, i0:i1].unsqueeze(-1) + 1e-12
            )                                          # [H, Br, L]

            col_sum_heads += P.sum(dim=1)              # [H, L]
            del S, P

        out[b] = col_sum_heads.mean(dim=0)

    return out


def _get(module, names):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    return None

def _get_heads_from_model(model, last_attn_module):
    H_q = getattr(getattr(model, "config", object()), "num_attention_heads", None)
    H_kv = getattr(getattr(model, "config", object()), "num_key_value_heads", None)
    return int(H_q), int(H_kv)

def _attach_qk_hooks(last_attn_module, model):
    q_linear = _get(last_attn_module, ["q_proj", "wq", "query", "q"])
    k_linear = _get(last_attn_module, ["k_proj", "wk", "key", "k"])
    if q_linear is None or k_linear is None: return []

    H_q, H_kv = _get_heads_from_model(model, last_attn_module)
    group = H_q // H_kv
    hooks = []

    def _grab_q(module, inp, out):
        B, L, Dall_q = out.shape
        D = Dall_q // H_q
        q = out.view(B, L, H_q, D).permute(0, 2, 1, 3).contiguous()
        captured["q_linear"] = q.detach()

    def _grab_k(module, inp, out):
        B, L, Dall_k = out.shape
        D = Dall_k // H_kv
        k = out.view(B, L, H_kv, D).permute(0, 2, 1, 3).contiguous()
        if H_kv != H_q: k = k.repeat_interleave(group, dim=1)
        captured["k_linear"] = k.detach()

    hooks.append(q_linear.register_forward_hook(_grab_q))
    hooks.append(k_linear.register_forward_hook(_grab_k))
    return hooks


def _apply_rope_if_possible(q, k, last_attn_module, seq_len):
    try:
        rotary = getattr(last_attn_module, "rotary_emb", None) or getattr(last_attn_module, "rope", None)
        if rotary is None: return q, k
        B, H, L, D = q.shape
        if hasattr(rotary, "forward"):
            cos, sin = rotary(torch.empty(B*H, L, D, device=q.device, dtype=q.dtype), seq_len=L)
        elif hasattr(rotary, "get_cos_sin"):
            cos, sin = rotary.get_cos_sin(L, device=q.device, dtype=q.dtype)
        else: return q, k

        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0).expand(B, H, L, D)
            sin = sin.unsqueeze(0).unsqueeze(0).expand(B, H, L, D)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(0).expand(B, -1, -1, -1)
            sin = sin.unsqueeze(0).expand(B, -1, -1, -1)

        def _rope(a, cos, sin):
            D2 = a.shape[-1] // 2
            a1, a2 = a[..., :D2], a[..., D2:]
            rot = torch.cat([-a2, a1], dim=-1)
            return a * cos.to(a.dtype) + rot * sin.to(a.dtype)

        return _rope(q, cos, sin), _rope(k, cos, sin)
    except Exception:
        return q, k


def calc_attentions_streaming(seq: str, model, tokenizer, device, block_rows=1024) -> List[float]:
    captured.clear()
    tokenizer.model_max_length = int(1e9)
    inputs = tokenizer(seq, return_tensors="pt", truncation=False)
    if 'token_type_ids' in inputs: del inputs['token_type_ids']
    inputs = {k: v.to(device) for k, v in inputs.items()}

    last_attn = model.model.layers[-1].self_attn
    hooks = _attach_qk_hooks(last_attn, model)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        for h in hooks:
            try: h.remove()
            except: pass

    q, k = captured.get("q_linear", None), captured.get("k_linear", None)
    q, k = q.to(device), k.to(device)
    q, k = _apply_rope_if_possible(q, k, last_attn, q.shape[2])

    col_sums = attn_column_sums_streaming(q, k, causal=True, block_rows=block_rows)
    return col_sums[0].detach().cpu().float().numpy().tolist()


def calc_attentions_streaming_batch(seqs: List[str], model, tokenizer, device, block_rows=1024) -> List[List[float]]:
    """Extract attention in batches of sequences with identical token length."""
    if len(seqs) == 1:
        return [calc_attentions_streaming(seqs[0], model, tokenizer, device, block_rows)]

    captured.clear()
    tokenizer.model_max_length = int(1e9)
    inputs = tokenizer(seqs, return_tensors="pt", truncation=False, padding=False)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(device) for k, v in inputs.items()}

    last_attn = model.model.layers[-1].self_attn
    hooks = _attach_qk_hooks(last_attn, model)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        for h in hooks:
            try:
                h.remove()
            except:
                pass

    q, k = captured.get("q_linear", None), captured.get("k_linear", None)
    if q is None or k is None:
        raise ValueError("Q/K tensors were not captured; check the model attention projection layer names.")
    q, k = q.to(device), k.to(device)
    q, k = _apply_rope_if_possible(q, k, last_attn, q.shape[2])

    col_sums = attn_column_sums_streaming(q, k, causal=True, block_rows=block_rows)
    return [col_sums[i].detach().cpu().float().numpy().tolist() for i in range(col_sums.shape[0])]


def token_length(seq: str, tokenizer) -> int:
    return len(tokenizer(seq, truncation=False)["input_ids"])


def build_same_length_batches(samples, seq_key, tokenizer, batch_size):
    groups = {}
    for idx, sample in enumerate(samples):
        if seq_key not in sample:
            continue
        length = token_length(sample[seq_key], tokenizer)
        groups.setdefault(length, []).append(idx)

    for length in sorted(groups):
        indices = groups[length]
        for i in range(0, len(indices), batch_size):
            yield indices[i:i + batch_size]


def fill_attention_batches(samples, processed, seq_key, out_key, tokenizer, model, device, block_rows, batch_size, desc):
    batches = list(build_same_length_batches(samples, seq_key, tokenizer, batch_size))
    for indices in tqdm(batches, desc=desc, leave=False):
        seqs = [samples[i][seq_key] for i in indices]
        try:
            scores_list = calc_attentions_streaming_batch(seqs, model, tokenizer, device, block_rows)
        except Exception as e:
            print(f"Batch attention failed; falling back to single-sample calculation: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            scores_list = []
            for seq in seqs:
                scores_list.append(calc_attentions_streaming(seq, model, tokenizer, device, block_rows))

        for idx, scores in zip(indices, scores_list):
            processed[idx][out_key] = scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract position-level attention in streaming mode.")
    parser.add_argument("--model_path", type=str, required=True, help="Local model directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input JSON directory.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--file_pattern", type=str, default="*.json")
    parser.add_argument("--block_cols", type=int, default=1024,
                        help="Number of query rows to process per chunk.")
    parser.add_argument("--bi_direction", action="store_true", help="Also extract reverse-complement attention.")
    parser.add_argument("--block_start", type=int, default=1, help="First block ID to process, inclusive and 1-based.")
    parser.add_argument("--block_end", type=int, default=0, help="Last block ID to process, inclusive and 1-based; 0 means all.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for sequences with the same token length.")
    return parser.parse_args()


def _extract_block_id(path_obj: Path):
    match = re.search(r"block_(\d+)", path_obj.stem)
    return int(match.group(1)) if match else None


def _natural_file_key(path_obj: Path):
    block_id = _extract_block_id(path_obj)
    if block_id is not None:
        return (0, block_id, path_obj.name)
    return (1, path_obj.name)


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    
    all_json_files = sorted(Path(args.input_dir).glob(args.file_pattern), key=_natural_file_key)
    block_start = max(1, int(args.block_start))
    block_end = int(args.block_end)
    json_files = []
    for p in all_json_files:
        block_id = _extract_block_id(p)
        if block_id is None:
            continue
        if block_id < block_start:
            continue
        if block_end > 0 and block_id > block_end:
            continue
        json_files.append(p)

    for json_file in json_files:
        output_file = Path(args.output_dir) / f"{json_file.stem}_attn.json"
        with open(json_file, "r") as f:
            samples = json.load(f)
            
        processed = [
            {
                "label": sample["label"], "spec": sample["spec"], "loc": sample["loc"],
                "pos_list": sample["pos_list"]
            }
            for sample in samples
        ]

        effective_batch_size = max(1, int(args.batch_size))
        fill_attention_batches(
            samples, processed, "sequence", "sequence_attention",
            tokenizer, model, device, args.block_cols, effective_batch_size,
            desc=f"forward {json_file.name}"
        )

        if args.bi_direction:
            for idx, sample in enumerate(samples):
                if "pos_list_revcomp" in sample:
                    processed[idx]["pos_list_revcomp"] = sample["pos_list_revcomp"]
            fill_attention_batches(
                samples, processed, "sequence_revcomp", "sequence_revcomp_attention",
                tokenizer, model, device, args.block_cols, effective_batch_size,
                desc=f"revcomp {json_file.name}"
            )
                
        with open(output_file, 'w') as f:
            json.dump(processed, f)
        print("\nStreaming attention extraction finished.")

if __name__ == "__main__":
    main()

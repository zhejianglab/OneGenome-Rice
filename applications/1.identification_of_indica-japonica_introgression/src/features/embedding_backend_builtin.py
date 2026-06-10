from __future__ import annotations

import os
from typing import Any

import torch
from tqdm import tqdm

torch.use_deterministic_algorithms(True)


def collate_fn(batch, tokenizer):
    """Batch rows: (sequence, label, variety, source, seq_id, seq_start_end, source_group)."""
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    encoding = tokenizer(sequences, padding=True, return_tensors="pt")
    metas = {
        "sequences": sequences,
        "variety": [item[2] for item in batch],
        "source": [item[3] for item in batch],
        "seq_id": [item[4] for item in batch],
        "seq_start_end": [item[5] for item in batch],
        "source_group": [item[6] for item in batch],
    }
    return encoding, torch.tensor(labels), metas


def extract_embeddings(
    model: Any,
    dataloader,
    device: str,
    gpu_id: int,
    dataset_name: str,
    layer2extract: list[int],
    seq_number: int,
    config,
    tqdm_print: str,
):
    """Extract embeddings with masked mean pooling.

    Kept compatible with legacy extractor signature.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    all_embeddings = {layer: [] for layer in layer2extract}
    all_labels = []
    meta_keys = ("sequences", "variety", "source", "seq_id", "seq_start_end", "source_group")
    all_meta: dict[str, list[Any]] = {k: [] for k in meta_keys}
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f"{tqdm_print}",
            leave=False,
            position=gpu_id,
            mininterval=3,
            dynamic_ncols=True,
        ):
            mask = batch[0]["attention_mask"].unsqueeze(-1).to(device)
            outputs = model(
                **{
                    "input_ids": batch[0]["input_ids"].to(
                        device, non_blocking=(device == "cuda")
                    )
                },
                output_hidden_states=True,
            )
            for layer in all_embeddings.keys():
                hidden = outputs.hidden_states[layer]
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                pooled = pooled.float()
                all_embeddings[layer].append(pooled.cpu())
                del hidden, pooled
            del mask, outputs
            if device == "cuda":
                torch.cuda.empty_cache()
            all_labels.append(batch[1])
            m = batch[2]
            for k in meta_keys:
                all_meta[k].extend(m[k])

    for layer, embeddings in all_embeddings.items():
        hidden_state_length = config.model_config["hidden_size"]
        assert (
            embeddings[0].shape[-1] == hidden_state_length
        ), f"[ERROR] hidden_state_length is not correct: {embeddings[0].shape[-1]} != {hidden_state_length}"
        all_embeddings[layer] = torch.squeeze(
            torch.reshape(torch.cat(embeddings), (-1, seq_number, hidden_state_length))
        )
    stacked_labels = torch.cat(all_labels).cpu()
    n_rows = int(stacked_labels.shape[0])
    for k, vals in all_meta.items():
        if len(vals) != n_rows:
            raise RuntimeError(
                f"Metadata length mismatch for {k}: got {len(vals)}, expected {n_rows} (same as labels)"
            )
    return all_embeddings, stacked_labels, all_meta


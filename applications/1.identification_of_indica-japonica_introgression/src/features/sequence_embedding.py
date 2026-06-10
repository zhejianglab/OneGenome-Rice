from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SimpleSequenceDataset(Dataset):
    """Simple dataset wrapper for text sequences."""

    def __init__(self, sequences: List[str], labels: Optional[List[Any]] = None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


def collate_sequences_for_embedding(batch: List[Any], tokenizer) -> Tuple[Dict, Optional[torch.Tensor]]:
    """Collate function for embedding extraction."""
    if isinstance(batch[0], tuple):
        sequences, labels = zip(*batch)
        has_labels = True
    else:
        sequences = batch
        labels = None
        has_labels = False

    encoding = tokenizer(list(sequences), padding=True, return_tensors="pt")

    if has_labels:
        labels_tensor = torch.tensor([label if isinstance(label, (int, float)) else label for label in labels])
        return encoding, labels_tensor

    return encoding, None


def extract_embeddings_from_sequences(
    model,
    tokenizer,
    sequences: List[str],
    labels: Optional[List[Any]] = None,
    layers: List[int] = [12],
    batch_size: int = 8,
    device: str = "cuda",
) -> Tuple[Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    """Extract multi-layer embeddings from sequences with masked mean pooling."""
    if not sequences:
        raise ValueError("sequences list cannot be empty")

    dataset = SimpleSequenceDataset(sequences, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_sequences_for_embedding(batch, tokenizer),
    )

    embeddings_dict = {layer: [] for layer in layers}
    all_labels = [] if labels is not None else None

    model.eval()
    with torch.no_grad():
        for encoding, batch_labels in dataloader:
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            for layer in layers:
                hidden_states = outputs.hidden_states[layer]
                mask = attention_mask.unsqueeze(-1).float()
                masked_hidden = hidden_states * mask
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = mask.sum(dim=1)
                sum_mask = torch.where(sum_mask == 0, torch.ones_like(sum_mask), sum_mask)
                pooled = sum_hidden / sum_mask
                embeddings_dict[layer].append(pooled.cpu())

            if batch_labels is not None:
                all_labels.append(batch_labels)

    for layer in layers:
        embeddings_dict[layer] = torch.cat(embeddings_dict[layer], dim=0)

    if all_labels is not None:
        all_labels = torch.cat(all_labels, dim=0)

    return embeddings_dict, all_labels


# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""Parse training/inference YAML dataset sections into `build_index` arguments."""

from __future__ import annotations

import os
from typing import Any

# Canonical keys (first non-empty list wins): training uses `training_data`, inference uses `test_data`.
_DATASET_LIST_KEYS = ("training_data", "test_data")


def _dataset_entries_list(cfg: dict[str, Any]) -> list:
    for key in _DATASET_LIST_KEYS:
        raw = cfg.get(key)
        if isinstance(raw, list) and len(raw) > 0:
            return raw
    raise ValueError(
        "Config must define one non-empty list: `training_data` (training) or "
        "`test_data` (inference), each entry: name, rna_path_plus, rna_path_minus, atac_path."
    )


def _require_file(path: str, *, context: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{context}: file not found: {path}")


def parse_dataset_block(cfg: dict[str, Any]):
    """Return parsed dataset block for index building.

    **Structured lists only** (list of dicts with ``name``, ``rna_path_plus``, ``rna_path_minus``,
    ``atac_path``, optional ``chromosome``, optional ``genome_fasta``) under:

    - ``training_data`` — canonical for ``training.yaml`` / ``training.py``
    - ``test_data`` — canonical for ``infer.yaml`` / ``inference.py``

    File paths are checked with ``os.path.isfile`` before returning.
    """
    entries = _dataset_entries_list(cfg)
    first = entries[0]
    if not isinstance(first, dict):
        raise ValueError(
            "Dataset entries must be a list of mappings with keys name, rna_path_plus, "
            "rna_path_minus, atac_path (legacy string lists / cell_types are not supported)."
        )
    return _parse_structured_dataset_entries(cfg, entries)


def _parse_structured_dataset_entries(
    cfg: dict[str, Any], entries: list
) -> tuple[
    dict,
    dict,
    list[str],
    str,
    dict[str, str] | None,
    str,
    dict[str, str] | None,
]:
    default_chr = cfg.get("chromosome")
    default_chr_str = str(default_chr) if default_chr is not None else None
    default_fasta = cfg.get("genome_fasta")
    default_fasta_str = str(default_fasta) if default_fasta is not None else None

    rna_files: dict = {}
    atac_files: dict = {}
    cell_types: list[str] = []
    overrides: dict[str, str] = {}
    fasta_overrides: dict[str, str] = {}

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"dataset entry [{i}] must be a mapping")
        name = entry.get("name")
        if not name:
            raise ValueError(f"dataset entry [{i}] must include string 'name'")
        name = str(name)

        for key in ("rna_path_plus", "rna_path_minus", "atac_path"):
            if key not in entry or not entry[key]:
                raise ValueError(f"dataset entry {name!r} missing or empty {key!r}")
            p = str(entry[key])
            _require_file(
                p,
                context=f"dataset entry {name!r} ({key})",
            )

        cell_types.append(name)
        rna_files[(name, "+")] = entry["rna_path_plus"]
        rna_files[(name, "-")] = entry["rna_path_minus"]
        atac_files[name] = entry["atac_path"]

        ch = entry.get("chromosome")
        if ch is not None:
            ch = str(ch)
            if default_chr_str is None or ch != default_chr_str:
                overrides[name] = ch

        fasta = entry.get("genome_fasta")
        if fasta is not None:
            fasta = str(fasta)
            _require_file(fasta, context=f"dataset entry {name!r} (genome_fasta)")
            if default_fasta_str is None or fasta != default_fasta_str:
                fasta_overrides[name] = fasta

    if default_chr_str is None:
        unresolved = [n for n in cell_types if n not in overrides]
        if unresolved:
            raise ValueError(
                "Set top-level 'chromosome' or add 'chromosome' to each dataset entry "
                f"(missing for: {unresolved})"
            )
        default_chr_str = overrides[cell_types[0]]

    per_cell = overrides if overrides else None

    if default_fasta_str is None:
        unresolved_fasta = [n for n in cell_types if n not in fasta_overrides]
        if unresolved_fasta:
            raise ValueError(
                "Set top-level 'genome_fasta' or add 'genome_fasta' to each dataset entry "
                f"(missing for: {unresolved_fasta})"
            )
        default_fasta_str = fasta_overrides[cell_types[0]]

    per_cell_fasta = fasta_overrides if fasta_overrides else None
    return (
        rna_files,
        atac_files,
        cell_types,
        default_chr_str,
        per_cell,
        default_fasta_str,
        per_cell_fasta,
    )


def effective_per_device_train_batch_size(cfg: dict[str, Any]) -> int:
    """Return ``training.per_device_train_batch_size`` (required)."""
    tcfg = cfg.get("training")
    if (
        isinstance(tcfg, dict)
        and "per_device_train_batch_size" in tcfg
        and tcfg["per_device_train_batch_size"] is not None
    ):
        return int(tcfg["per_device_train_batch_size"])
    raise ValueError("Missing required key: training.per_device_train_batch_size")


def effective_inference_batch_size(cfg: dict[str, Any]) -> int:
    """Return ``inference_batch_size`` (required for inference configs)."""
    if "inference_batch_size" in cfg and cfg["inference_batch_size"] is not None:
        return int(cfg["inference_batch_size"])
    raise ValueError("Missing required key: inference_batch_size")

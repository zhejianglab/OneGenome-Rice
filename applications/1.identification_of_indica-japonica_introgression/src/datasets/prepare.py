from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import yaml

from src.io.naming import dataset_tag
from src.io.paths import dataset_root, resolve
from src.common.config import Config
from src.common.schema import get_indent

def read_fasta(file_path: str | Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []
    path_str = str(file_path)
    open_func = gzip.open if path_str.endswith(".gz") else open
    mode = "rt" if path_str.endswith(".gz") else "r"
    with open_func(path_str, mode, encoding="utf-8") as fasta:
        for line in fasta:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)
    return sequences


def trim_n(sequence: str) -> tuple[str, list[int]]:
    sequence = sequence.upper()
    start = 0
    end = len(sequence)
    while start < end and sequence[start] == "N":
        start += 1
    while end > start and sequence[end - 1] == "N":
        end -= 1
    return sequence[start:end], [start, end]


def sliding_window(sequence: str, window_size: int, step_size: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for start in range(0, len(sequence) - window_size + 1, step_size):
        end = start + window_size
        windows.append({"window": sequence[start:end], "seq_start_end": [start, end]})
    return windows


def _collect_windows(file_paths: list[str], window_size: int, step_size: int, unique: bool) -> list[dict[str, Any]]:
    all_windows: list[dict[str, Any]] = []
    for fp in file_paths:
        fasta_path = resolve(fp)
        sequences = read_fasta(fasta_path)
        for seq_id, seq in sequences.items():
            trimmed, span = trim_n(seq)
            for win in sliding_window(trimmed, window_size, step_size):
                start, end = win["seq_start_end"]
                all_windows.append(
                    {
                        "window": win["window"],
                        "file": str(fasta_path),
                        "seq_id": seq_id,
                        "seq_start_end": [span[0] + start, span[0] + end],
                    }
                )
    if not unique:
        return all_windows
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for row in all_windows:
        if row["window"] in seen:
            continue
        seen.add(row["window"])
        uniq.append(row)
    return uniq


def resolve_fasta_path(file_path: str, fasta_root: str | Path) -> Path:
    """Resolve FASTA path with fasta_root fallback for relative paths.

    Rules:
    1. Absolute path: use as-is.
    2. Relative path that already resolves from repo root: use it directly.
    3. Otherwise, prepend `fasta_root`.
    """
    p = Path(file_path).expanduser()
    if p.is_absolute():
        return p.resolve()

    direct = resolve(file_path)
    if direct.exists():
        return direct

    return resolve(Path(fasta_root) / p)


def build_dataset_split_records(
    dataset_name: str,
    split: str,
    config: Config,
) -> tuple[str, list[dict[str, Any]], int, int]:
    ds_cfg = config.data_process.datasets[dataset_name]
    split_cfg = ds_cfg[split]
    label_cfg = ds_cfg.label
    class_names = list(label_cfg.__dict__.keys())
    ds_tag = dataset_tag(dataset_name, class_names)
    window_size = split_cfg.window_size
    step_size = split_cfg.step_size
    unique = split_cfg.get("unique_windows", False)
    fasta_root = config.data_process.fasta_root
    records: list[dict[str, Any]] = []
    print(
        f"{get_indent('PREPARE')} Start slicing: dataset={dataset_name}, split={split}, "
        f"window={window_size}, step={step_size}, unique={unique}"
    )

    for class_name in class_names:
        label = label_cfg[class_name]
        files = [str(resolve_fasta_path(fp, fasta_root=fasta_root)) for fp in split_cfg[class_name]]
        print(f"{get_indent('PREPARE')} Processing class={class_name}, files={len(files)}")
        windows = _collect_windows(files, window_size, step_size, unique)
        print(f"{get_indent('PREPARE')} Collected windows: class={class_name}, count={len(windows)}")
        for w in windows:
            records.append(
                {
                    "sequence": w["window"],
                    "label": label,
                    "variety": class_name,
                    "source": w["file"],
                    "seq_id": w["seq_id"],
                    "seq_start_end": w["seq_start_end"],
                    "source_group": f"{dataset_name}_{split}",
                }
            )
    print(f"{get_indent('PREPARE')} Split built: dataset={dataset_name}, split={split}, records={len(records)}")
    return ds_tag, records, window_size, step_size


def write_split_jsonl(dataset_dir: Path, split: str, records: list[dict[str, Any]]) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = dataset_dir / f"{split}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out


def update_datasets_info(config: Config, dataset_tag_name: str) -> Path:
    info_path = resolve(config.data_process.dataset_feature_path)
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data.setdefault("support_dataset", [])
    if dataset_tag_name not in data["support_dataset"]:
        data["support_dataset"].append(dataset_tag_name)
    data.setdefault("long_sequence_dataset", ["none"])
    data.setdefault("super_long_sequence_dataset", ["none"])
    data.setdefault("dataset_feature", {})
    data["dataset_feature"][dataset_tag_name] = {
        "seq_for_item": 1,
        "seq_key": "sequence",
        "label_key": "label",
        "eval_task": "labels",
        "data_split": ["train", "test"],
    }
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with info_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return info_path


def run_prepare(config: Config, datasets: list[str] | None = None) -> list[Path]:
    out_paths: list[Path] = []
    selected = datasets or list(config.data_process.datasets.__dict__.keys())
    root = dataset_root(config)
    print("=" * 50)
    print(f"{get_indent('PREPARE')} Stage start: datasets={selected}")
    for dataset_name in selected:
        ds_cfg = config.data_process.datasets[dataset_name]
        label_cfg = ds_cfg.label
        class_names = list(label_cfg.__dict__.keys())
        ds_tag = dataset_tag(dataset_name, class_names)
        out_dir = root / ds_tag
        print(f"{get_indent('PREPARE')} Dataset start: {dataset_name} -> {ds_tag}")
        for split in ["train", "test"]:
            split_path = out_dir / f"{split}.jsonl"
            if split_path.exists():
                print(f"{get_indent('PREPARE')} Skip existing split: {split_path}")
                out_paths.append(split_path)
                continue
            _, records, _, _ = build_dataset_split_records(dataset_name, split, config)
            print(f"{get_indent('PREPARE')} Writing split file: {split_path}")
            out_paths.append(write_split_jsonl(out_dir, split, records))
        update_datasets_info(config, ds_tag)
        print(f"{get_indent('PREPARE')} Dataset done: {dataset_name}")
    print(f"{get_indent('PREPARE')} Stage done: outputs={len(out_paths)}")
    print("=" * 50)
    return out_paths


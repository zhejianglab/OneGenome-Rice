#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-31
"""
Collect per-run metrics from reg/ and test/ inference outputs into a wide table.

1. For each subdirectory under reg/ and test/ that contains both
   plus_predictions.pickle and minus_predictions.pickle, runs calc_metrics.py
   (in parallel, up to -p workers). If ``<DIR>/experiment.yaml`` exists, it is passed
   to calc_metrics as ``-c`` so ``dataset_type`` (and metric defaults) come from YAML only.
2. Reads per-directory metric text files and writes a wide CSV.

Metric files (per inference output directory):
  strand_metrics.txt — legacy / full-only runs (unprefixed columns in the wide CSV)
  strands_metrics.txt — legacy typo filename, treated like strand_metrics.txt

Output columns:
  model_id, split, sample, checkpoint, <all metric keys unioned across files>

Missing metrics are filled with NA.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_DIR_RE = re.compile(r"^(?P<sample>.+)\.(?P<checkpoint>checkpoint-\d+)$")
_CKPT_NUM_RE = re.compile(r"^checkpoint-(?P<num>\d+)$")


def _parse_metrics_file(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            metrics[k] = v
    return metrics


def _parse_prefix_dirname(dirname: str) -> tuple[str, str]:
    """
    Default naming (from run.py): <sample>.<checkpoint-XXXX>
    If it doesn't match, return (dirname, "").
    """
    m = _DIR_RE.match(dirname)
    if not m:
        return dirname, "NA"
    sample = m.group("sample")
    ckpt_raw = m.group("checkpoint")
    m2 = _CKPT_NUM_RE.match(ckpt_raw)
    if not m2:
        return sample, "NA"
    return sample, f"ckpt-{m2.group('num')}"


def _prefix_dir_sort_key(p: Path) -> tuple[str, int, str]:
    """
    Sort inference output dirs by (sample, checkpoint_step) instead of lexicographic dirname.

    Expected dirname: <sample>.checkpoint-<num> (from run.py). Unparseable names go last,
    still grouped by their raw name for stability.
    """
    name = p.name
    m = _DIR_RE.match(name)
    if not m:
        return (name, 1_000_000_000_000, name)
    sample = m.group("sample")
    ckpt_raw = m.group("checkpoint")
    m2 = _CKPT_NUM_RE.match(ckpt_raw)
    if not m2:
        return (sample, 1_000_000_000_000, name)
    return (sample, int(m2.group("num")), name)


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)
    s = str(value).strip()
    if not s or s.upper() == "NA":
        return "NA"
    # Keep NaN verbatim, but normalize casing
    if s.lower() == "nan":
        return "NaN"
    try:
        return f"{float(s):.4f}"
    except Exception:
        return s


def _calc_metrics_script() -> Path:
    return Path(__file__).resolve().parent / "calc_metrics.py"


def discover_calc_metrics_jobs(base: Path) -> list[tuple[Path, Path, Path]]:
    """Return (plus_pickle, minus_pickle, strand_metrics_base_path) per inference output dir."""
    jobs: list[tuple[Path, Path, Path]] = []
    base = base.resolve()
    for split in ("reg", "test"):
        split_dir = base / split
        if not split_dir.is_dir():
            continue
        for d in sorted(split_dir.iterdir()):
            if not d.is_dir():
                continue
            plus = d / "plus_predictions.pickle"
            minus = d / "minus_predictions.pickle"
            out_base = d / "strand_metrics.txt"
            if plus.is_file() and minus.is_file():
                jobs.append((plus, minus, out_base))
    return jobs


def run_calc_metrics_parallel(
    jobs: list[tuple[Path, Path, Path]],
    parallel: int,
    calc_script: Path,
    *,
    experiment_yaml: Path | None,
) -> None:
    if not jobs:
        return
    workers = max(1, min(int(parallel), len(jobs)))

    def _one(job: tuple[Path, Path, Path]) -> None:
        p_plus, p_minus, p_out = job
        cmd = [
            sys.executable,
            str(calc_script),
            str(p_plus),
            str(p_minus),
            str(p_out),
        ]
        if experiment_yaml is not None and experiment_yaml.is_file():
            cmd.extend(["-c", str(experiment_yaml)])
        subprocess.run(cmd, check=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, jobs))


def _metric_txt_files_in_dir(d: Path) -> list[tuple[Path, str]]:
    """Paths and CSV column prefixes (empty string = legacy single full metrics file)."""
    out: list[tuple[Path, str]] = []
    legacy = d / "strand_metrics.txt"
    typo = d / "strands_metrics.txt"
    if legacy.is_file():
        out.append((legacy, ""))
    elif typo.is_file():
        out.append((typo, ""))
    return out


def collect_rows(model_base_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    model_base_dir = model_base_dir.resolve()
    model_id = model_base_dir.name

    rows: list[dict[str, Any]] = []
    all_metric_keys: set[str] = set()

    for split in ("reg", "test"):
        split_dir = model_base_dir / split
        if not split_dir.is_dir():
            continue

        for prefix_dir in sorted(
            (p for p in split_dir.iterdir() if p.is_dir()), key=_prefix_dir_sort_key
        ):
            metric_files = _metric_txt_files_in_dir(prefix_dir)
            if not metric_files:
                continue
            sample, checkpoint = _parse_prefix_dirname(prefix_dir.name)
            row: dict[str, Any] = {
                "model_id": model_id,
                "split": split,
                "sample": sample,
                "checkpoint": checkpoint,
            }
            for mpath, tag in metric_files:
                metrics = _parse_metrics_file(mpath)
                if tag:
                    for k, v in metrics.items():
                        key = f"{tag}_{k}"
                        row[key] = v
                        all_metric_keys.add(key)
                else:
                    row.update(metrics)
                    all_metric_keys.update(metrics.keys())
            rows.append(row)

    metric_cols = sorted(all_metric_keys)
    return rows, metric_cols


def write_wide_csv(rows: list[dict[str, Any]], metric_cols: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_id", "split", "sample", "checkpoint"] + list(metric_cols)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "NA") for k in fieldnames}
            # Format metric values and fill missing with NA.
            for k in metric_cols:
                out[k] = _format_metric_value(out.get(k))
            # Ensure id cols are always present (and checkpoint is non-empty).
            if not out.get("checkpoint"):
                out["checkpoint"] = "NA"
            w.writerow(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parallel calc_metrics on reg/test dirs, then collect strand_metrics*.txt into a wide CSV."
    )
    p.add_argument(
        "model_base_dir",
        metavar="DIR",
        help="Base directory that contains reg/ and/or test/ subdirectories",
    )
    p.add_argument(
        "-o",
        "--out",
        metavar="PATH",
        default=None,
        help=(
            "Output CSV path. Default: <DIR>/stats.wide.full.csv."
        ),
    )
    p.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=None,
        metavar="P",
        help="Max concurrent calc_metrics subprocesses (default when omitted: 4)",
    )
    args = p.parse_args()

    base = Path(args.model_base_dir).resolve()
    exp_yaml = base / "experiment.yaml"

    parallel = args.parallel if args.parallel is not None else 4
    if parallel < 1:
        p.error("--parallel must be >= 1")

    experiment_yaml = exp_yaml if exp_yaml.is_file() else None

    jobs = discover_calc_metrics_jobs(base)
    calc_script = _calc_metrics_script()
    if jobs:
        print(f"calc_metrics: {len(jobs)} job(s), up to {parallel} concurrent subprocess(es)")
        run_calc_metrics_parallel(
            jobs,
            parallel,
            calc_script,
            experiment_yaml=experiment_yaml,
        )

    rows, metric_cols = collect_rows(base)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = base / "stats.wide.full.csv"
    write_wide_csv(rows, metric_cols, out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()

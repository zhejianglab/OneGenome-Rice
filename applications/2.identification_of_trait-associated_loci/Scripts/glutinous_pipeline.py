#!/usr/bin/env python3
"""Run glutinous-rice attention analysis and produce the region significance plot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def rel(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(config_path: Path) -> tuple[Path, dict]:
    root = config_path.resolve().parent
    return root, json.loads(config_path.read_text(encoding="utf-8"))


def results_root(root: Path, cfg: dict) -> Path:
    return rel(root, cfg["paths"]["results_dir"])


def glutinous_results(root: Path, cfg: dict) -> Path:
    return results_root(root, cfg) / "glutinous"


def read_regions(path: Path) -> list[tuple[str, int, int]]:
    regions: list[tuple[str, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        chrom, start, end, *_ = line.split()
        regions.append((chrom, int(start), int(end)))
    if len(regions) != 4:
        raise ValueError(f"Expected four candidate regions, found {len(regions)} in {path}")
    return regions


def write_sliding_windows(out_bed: Path, chrom: str, start: int, end: int, window_size: int, stride: int) -> None:
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    pos = start
    while pos < end:
        block_end = min(pos + window_size, end)
        lines.append(f"{chrom}\t{pos}\t{block_end}\n")
        pos += stride
    out_bed.write_text("".join(lines), encoding="utf-8")


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}; see {log_path}")


def prepare_region_beds(root: Path, cfg: dict) -> list[Path]:
    paths = cfg["paths"]
    workflow = cfg["workflow"]
    attention = glutinous_results(root, cfg) / "attention"
    bed_dir = attention / "region_beds"
    bed_dir.mkdir(parents=True, exist_ok=True)
    beds: list[Path] = []
    for idx, (chrom, start, end) in enumerate(read_regions(rel(root, paths["candidate_regions"])), start=1):
        bed = bed_dir / f"region_{idx}.bed"
        if not bed.exists():
            write_sliding_windows(bed, chrom, start, end, int(workflow["window_size"]), int(workflow["stride"]))
        beds.append(bed)
    return beds


def count_blocks(bed_file: Path) -> int:
    count = 0
    for line in bed_file.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            count += 1
    return count


def matrix_ready(matrix_dir: Path, bed_files: list[Path]) -> bool:
    for region_idx, bed in enumerate(bed_files, start=1):
        region_dir = matrix_dir / f"region_{region_idx}"
        for block_idx in range(1, count_blocks(bed) + 1):
            block = region_dir / f"block_{block_idx}"
            if not (block / "hap1_attention_collapsed.csv").exists():
                return False
            if not (block / "hap1_attention_collapsed_revcomp.csv").exists():
                return False
    return True


def diff_ready(diff_dir: Path, bed_files: list[Path]) -> bool:
    for region_idx, bed in enumerate(bed_files, start=1):
        region_dir = diff_dir / f"region_{region_idx}"
        for block_idx in range(1, count_blocks(bed) + 1):
            tables = region_dir / f"block_{block_idx}" / "tables"
            if not (tables / "fwd_group0vs1_all_results.csv").exists():
                return False
            if not (tables / "revcomp_group0vs1_all_results.csv").exists():
                return False
    return True


def run_differential_only(root: Path, cfg: dict, bed_files: list[Path], env: dict[str, str]) -> None:
    scripts = root / "Scripts" / "lib"
    attention = glutinous_results(root, cfg) / "attention"
    matrix_dir = attention / "04_attention_matrices"
    diff_dir = attention / "05_differential_sites"
    py = sys.executable
    for region_idx, _bed in enumerate(bed_files, start=1):
        input_dir = matrix_dir / f"region_{region_idx}"
        output_dir = diff_dir / f"region_{region_idx}"
        run_logged(
            [
                py,
                str(scripts / "differential_analysis_rice_per_block.py"),
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
            ],
            output_dir / "run.log",
            env,
        )


def plot_significance(root: Path, cfg: dict, bed_files: list[Path], env: dict[str, str]) -> Path:
    paths = cfg["paths"]
    workflow = cfg["workflow"]
    scripts = root / "Scripts" / "lib"
    results = glutinous_results(root, cfg)
    attention = results / "attention"
    diff_dir = attention / "05_differential_sites"
    matrix_dir = attention / "04_attention_matrices"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    region_dirs = [diff_dir / f"region_{i}" for i in range(1, 5)]
    matrix_dirs = [matrix_dir / f"region_{i}" for i in range(1, 5)]
    out_png = figures / "glutinous_differential_attention_padj.png"
    run(
        [
            sys.executable,
            str(scripts / "plot_signal_panel_annotated.py"),
            "--run-name",
            cfg["project"]["name"],
            "--gff",
            str(rel(root, paths["gene_annotation"])),
            "--chrom",
            str(workflow["chromosome"]),
            "--region-dir",
            *map(str, region_dirs),
            "--bed-file",
            *map(str, bed_files),
            "--matrix-dir",
            *map(str, matrix_dirs),
            "--display-title",
            *workflow["region_names"],
            "--metric",
            "neglog10_padj",
            "--out-png",
            str(out_png),
        ],
        env,
    )
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs. Default: visible CUDA devices or 0.")
    parser.add_argument("--workers", type=int, default=None, help="Number of regions to run concurrently.")
    parser.add_argument("--force", action="store_true", help="Recompute attention even if matrix outputs already exist.")
    args = parser.parse_args()

    root, cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    results_base = results_root(root, cfg)
    results = glutinous_results(root, cfg)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(results_base / ".mplconfig"))

    bed_files = prepare_region_beds(root, cfg)
    attention = results / "attention"
    matrix_dir = attention / "04_attention_matrices"
    diff_dir = attention / "05_differential_sites"

    if args.force or not matrix_ready(matrix_dir, bed_files):
        print("Glutinous attention matrices not found; running full attention pipeline.", flush=True)
        cmd = [sys.executable, str(root / "Scripts" / "calc_attention.py"), "--config", str(Path(args.config).resolve())]
        if args.gpus:
            cmd.extend(["--gpus", args.gpus])
        if args.workers:
            cmd.extend(["--workers", str(args.workers)])
        run(cmd, env)
    else:
        print("Reusing existing glutinous attention matrices.", flush=True)
        if not diff_ready(diff_dir, bed_files):
            print("Differential-test outputs are missing; running differential tests from existing matrices.", flush=True)
            run_differential_only(root, cfg, bed_files, env)

    if not diff_ready(diff_dir, bed_files):
        raise FileNotFoundError("Glutinous differential-test outputs are still missing after computation.")

    out_png = plot_significance(root, cfg, bed_files, env)
    print(f"Glutinous figure written to {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the bidirectional attention and differential-analysis pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


def rel(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(config_path: Path) -> tuple[Path, dict]:
    root = config_path.resolve().parent
    return root, json.loads(config_path.read_text(encoding="utf-8"))


def glutinous_results(root: Path, cfg: dict) -> Path:
    return rel(root, cfg["paths"]["results_dir"]) / "glutinous"


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


def write_sliding_windows(
    out_bed: Path,
    chrom: str,
    start: int,
    end: int,
    window_size: int,
    stride: int,
) -> None:
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    pos = start
    while pos < end:
        block_end = min(pos + window_size, end)
        lines.append(f"{chrom}\t{pos}\t{block_end}\n")
        pos += stride
    out_bed.write_text("".join(lines), encoding="utf-8")


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}; see {log_path}")


def run_region(
    root: Path,
    cfg: dict,
    region_index: int,
    region_bed: Path,
    gpu_id: str,
) -> None:
    scripts = root / "Scripts" / "lib"
    paths = cfg["paths"]
    base = glutinous_results(root, cfg) / "attention"
    region_name = f"region_{region_index}"

    out1 = base / "01_pseudo_sequences" / region_name
    out2 = base / "02_raw_attention" / region_name
    out3 = base / "03_normalized_attention" / region_name
    out4 = base / "04_attention_matrices" / region_name
    out5 = base / "05_differential_sites" / region_name
    for path in (out1, out2, out3, out4, out5):
        path.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(rel(root, paths["results_dir"]) / ".mplconfig"))

    run_logged(
        [
            py,
            str(scripts / "generate_jsonl_indel_rice.py"),
            "--bed",
            str(region_bed),
            "--pheno",
            str(rel(root, paths["phenotype"])),
            "--vcf",
            str(rel(root, paths["vcf"])),
            "--fasta",
            str(rel(root, paths["reference_fasta"])),
            "--out",
            str(out1),
        ],
        out1 / "run.log",
        env,
    )

    model_env = env.copy()
    model_env["CUDA_VISIBLE_DEVICES"] = gpu_id
    run_logged(
        [
            py,
            str(scripts / "calc_flash_attention_run.py"),
            "--model_path",
            str(rel(root, paths["model"])),
            "--input_dir",
            str(out1),
            "--output_dir",
            str(out2),
            "--bi_direction",
        ],
        out2 / "run.log",
        model_env,
    )

    run_logged(
        [
            py,
            str(scripts / "atten_score_indel_normalizeV2.py"),
            "--json_dir",
            str(out2),
            "--output_dir",
            str(out3),
            "--bed_file",
            str(region_bed),
        ],
        out3 / "run.log",
        env,
    )

    run_logged(
        [
            py,
            str(scripts / "convert_json_to_matrix_perblock.py"),
            "--json_dir",
            str(out3),
            "--output_dir",
            str(out4),
            "--bed_file",
            str(region_bed),
        ],
        out4 / "run.log",
        env,
    )

    run_logged(
        [
            py,
            str(scripts / "differential_analysis_rice_per_block.py"),
            "--input_dir",
            str(out4),
            "--output_dir",
            str(out5),
        ],
        out5 / "run.log",
        env,
    )

    print(f"{region_name} finished on GPU {gpu_id}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs. Default: visible CUDA devices or 0.")
    parser.add_argument("--workers", type=int, default=None, help="Number of regions to run concurrently.")
    args = parser.parse_args()

    root, cfg = load_config(Path(args.config))
    paths = cfg["paths"]
    workflow = cfg["workflow"]

    base = glutinous_results(root, cfg) / "attention"
    bed_dir = base / "region_beds"
    bed_dir.mkdir(parents=True, exist_ok=True)

    regions = read_regions(rel(root, paths["candidate_regions"]))
    region_beds: list[Path] = []
    for idx, (chrom, start, end) in enumerate(regions, start=1):
        bed = bed_dir / f"region_{idx}.bed"
        write_sliding_windows(
            bed,
            chrom,
            start,
            end,
            int(workflow["window_size"]),
            int(workflow["stride"]),
        )
        region_beds.append(bed)
        print(f"Prepared {bed}", flush=True)

    gpu_list = args.gpus
    if gpu_list is None:
        gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpus = [item.strip() for item in gpu_list.split(",") if item.strip()] or ["0"]
    workers = args.workers or min(len(gpus), len(region_beds))
    workers = max(1, min(workers, len(region_beds)))

    print(f"Running {len(region_beds)} regions with {workers} worker(s); GPUs={','.join(gpus)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i, bed in enumerate(region_beds, start=1):
            gpu = gpus[(i - 1) % len(gpus)]
            futures.append(executor.submit(run_region, root, cfg, i, bed, gpu))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("Attention calculation and differential testing finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

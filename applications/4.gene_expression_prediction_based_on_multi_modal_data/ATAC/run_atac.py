#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""
ATAC-seq pipeline driver (Python entrypoint).

Mirrors ATAC/run.sh: Trimmomatic -> merge PE to SE -> Bowtie2 SE align ->
samtools stats -> bamCoverage BigWig via bam2bw_atac.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from tee_log import tee_output


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ATAC-seq trimming, merge, alignment, and bigWig generation pipeline."
    )
    parser.add_argument("input_prefix", help="Prefix for input FASTQ files (expects *_R1.fastq.gz and *_R2.fastq.gz)")
    parser.add_argument("ref", help="Reference prefix (e.g. ref/MH63/MH63)")
    parser.add_argument("output_prefix", help="Prefix for output files")
    parser.add_argument(
        "-p",
        "--threads",
        type=int,
        default=8,
        help="Number of processors to use (default: 8)",
    )
    parser.add_argument(
        "-q",
        "--minMappingQuality",
        type=int,
        default=30,
        help="Minimum mapping quality for bamCoverage (default: 30)",
    )
    parser.add_argument(
        "-b",
        "--binSize",
        type=int,
        default=1,
        help="Bin size for bamCoverage (default: 1)",
    )
    parser.add_argument(
        "-n",
        "--normalizeUsing",
        choices=["RPKM", "CPM", "BPM", "RPGC", "None"],
        default="RPGC",
        help="Normalization method for bamCoverage (default: RPGC)",
    )
    return parser.parse_args()


def _log_path_for_tee(output_prefix: str) -> tuple[str, str]:
    """Directory and basename for tee_output(...); both streams use basename -> one <prefix>.log."""
    log_path = os.path.abspath(f"{output_prefix}.log")
    parent = os.path.dirname(log_path) or "."
    name = os.path.basename(log_path)
    return parent, name


def run_cmd(cmd: list[str]) -> None:
    """Stream child stdout/stderr through sys.stdout/sys.stderr so TeeStream sees all output."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def forward(src, dst) -> None:
        assert src is not None
        for line in src:
            dst.write(line)
            dst.flush()

    t_out = threading.Thread(target=forward, args=(proc.stdout, sys.stdout), daemon=True)
    t_err = threading.Thread(target=forward, args=(proc.stderr, sys.stderr), daemon=True)
    t_out.start()
    t_err.start()
    rc = proc.wait()
    t_out.join(timeout=30)
    t_err.join(timeout=30)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _run_pipeline(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    trim_script = script_dir / "trimmomatic_atac.sh"
    align_script = script_dir / "align.bt2_se.sh"
    bam2bw_py = script_dir / "bam2bw_atac.py"

    output_prefix = args.output_prefix
    threads = args.threads

    logging.info("Starting ATAC-seq pipeline")
    logging.info("Workdir: %s", Path.cwd())
    logging.info("Log (tee): %s", os.path.abspath(f"{output_prefix}.log"))
    logging.info("Args: %s", args)
    logging.info("Input: %s", args.input_prefix)
    logging.info("Reference: %s", args.ref)
    logging.info("Output: %s", output_prefix)

    for path in (trim_script, align_script, bam2bw_py):
        if not path.is_file():
            logging.error("Missing script: %s", path)
            return 1

    # Step 1: Trimming with Trimmomatic (same as run.sh)
    logging.info("Step 1: Trimming...")
    run_cmd(["bash", str(trim_script), "-t", str(threads), args.input_prefix, output_prefix])

    # Step 2: Merge paired-end to single-end (same pigz pipeline as run.sh)
    logging.info("Step 2: Merging PE to SE...")
    outs = [
        f"{output_prefix}_R1_paired.fastq.gz",
        f"{output_prefix}_R2_paired.fastq.gz",
        f"{output_prefix}_R1_unpaired.fastq.gz",
        f"{output_prefix}_R2_unpaired.fastq.gz",
    ]
    se_out = f"{output_prefix}_SE.fastq.gz"
    pigz_in = " ".join(shlex.quote(o) for o in outs)
    merge_cmd = (
        f"pigz -p {threads} -dc {pigz_in} | pigz -p {threads} > {shlex.quote(se_out)}"
    )
    run_cmd(["bash", "-c", merge_cmd])

    # Step 3: Single-end alignment with Bowtie2 (same arguments as run.sh)
    logging.info("Step 3: Aligning SE reads...")
    run_cmd(
        [
            "bash",
            str(align_script),
            output_prefix,
            args.ref,
            f"{output_prefix}_SE",
        ]
    )

    sorted_bam = f"{output_prefix}_SE.align.sorted.bam"
    stat_out = f"{sorted_bam}.stat"
    stats_cmd = (
        f"samtools stats {shlex.quote(sorted_bam)} -@ {threads} > {shlex.quote(stat_out)}"
    )
    run_cmd(["bash", "-c", stats_cmd])

    # Step 5: Convert BAM to BigWig (bam2bw_atac.py; same threads as -p/--threads)
    logging.info("Step 5: Converting to BigWig...")
    run_cmd(
        [
            sys.executable,
            str(bam2bw_py),
            f"{output_prefix}_SE",
            args.ref,
            output_prefix,
            "--minMappingQuality",
            str(args.minMappingQuality),
            "--binSize",
            str(args.binSize),
            "--normalizeUsing",
            args.normalizeUsing,
            "--numberOfProcessors",
            str(threads),
        ]
    )

    logging.info("Pipeline complete!")
    return 0


def main() -> int:
    args = parse_args()
    log_dir, log_name = _log_path_for_tee(args.output_prefix)
    with tee_output(log_dir, log_name, log_name):
        return _run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

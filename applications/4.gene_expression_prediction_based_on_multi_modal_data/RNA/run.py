#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-27

import argparse
import logging
import os
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
    parser = argparse.ArgumentParser(description="RNA alignment and bigWig generation pipeline.")
    parser.add_argument("input_prefix", help="Input prefix for FASTQ files")
    parser.add_argument("ref", help="Reference prefix")
    parser.add_argument("output_prefix", help="Output prefix")
    parser.add_argument("-p", "--threads", type=int, default=8, help="Number of processors (default: 8)")
    parser.add_argument(
        "-s","--rna-strandness",
        choices=["F", "R", "FR", "RF", "None"],
        default="None",
        help="RNA strandness passed to align_hisat.py (default: None)",
    )
    parser.add_argument(
        "-n", "--normalizeUsing",
        choices=["RPKM", "CPM", "BPM", "RPGC", "None"],
        default="CPM",
        help="Normalization method passed to bam2bw_RNA.py (default: CPM)",
    )
    parser.add_argument(
        "-q", "--minMappingQuality", type=int, default=30, help="Minimum mapping quality (default: 30)")
    parser.add_argument(
        "-b", "--binSize", type=int, default=1, help="Bin size for bamCoverage (default: 1)")

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
    align_script = script_dir / "align_hisat.py"
    bam2bw_script = script_dir / "bam2bw_RNA.py"
    logging.info("Workdir: %s", Path.cwd())
    logging.info("Log (tee): %s", os.path.abspath(f"{args.output_prefix}.log"))
    logging.info("Args: %s", args)

    logging.info("RNA data alignment")
    logging.info("Workdir: %s", Path.cwd())
    logging.info("Input prefix: %s", args.input_prefix)
    logging.info("Reference: %s", args.ref)
    logging.info("Output prefix: %s", args.output_prefix)

    if not align_script.is_file():
        logging.error("Missing script: %s", align_script)
        return 1
    if not bam2bw_script.is_file():
        logging.error("Missing script: %s", bam2bw_script)
        return 1

    logging.info("Step 1: alignment")
    align_cmd: list[str] = [
        sys.executable,
        str(align_script),
        args.input_prefix,
        args.ref,
        args.output_prefix,
        "--threads",
        str(args.threads),
    ]
    if args.rna_strandness != "None":
        align_cmd.extend(["--rna-strandness", args.rna_strandness])
    run_cmd(align_cmd)

    logging.info("Step 2: bam2bw")
    bw_cmd: list[str] = [
        sys.executable,
        str(bam2bw_script),
        args.output_prefix,
        args.ref,
        args.output_prefix,
        "--minMappingQuality",
        str(args.minMappingQuality),
        "--binSize",
        str(args.binSize),
        "--normalizeUsing",
        args.normalizeUsing,
        "--numberOfProcessors",
        str(args.threads),
    ]
    run_cmd(bw_cmd)

    logging.info("Done: %s", args.output_prefix)
    return 0


def main() -> int:
    args = parse_args()
    log_dir, log_name = _log_path_for_tee(args.output_prefix)
    with tee_output(log_dir, log_name, log_name):
        return _run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())


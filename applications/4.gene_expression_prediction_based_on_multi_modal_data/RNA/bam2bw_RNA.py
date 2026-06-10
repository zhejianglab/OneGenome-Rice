#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-24

import argparse
import logging
import os
import subprocess
import sys


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Same bamCoverage entry as RNA/bam2bw_debug.py (seq env python + deeptools on sys.path).
SEQ_ENV_PYTHON = "/mnt/zzb/default/Workspace/xuyu/miniconda3/envs/seq/bin/python3.12"
SEQ_ENV_SITE_PACKAGES = "/mnt/zzb/default/Workspace/xuyu/software/source/deepTools/"
SEQ_ENV_BAMCOVERAGE_MAIN_CALL = (
    "import sys\n"
    f"sys.path.insert(0, {SEQ_ENV_SITE_PACKAGES!r})\n"
    "from deeptools.bamCoverage import main\n"
    "sys.exit(main())\n"
)


def _bamcoverage_cmd_prefix() -> list[str]:
    return [SEQ_ENV_PYTHON, "-c", SEQ_ENV_BAMCOVERAGE_MAIN_CALL]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strand-specific RNA bigWig files using bamCoverage.")
    parser.add_argument("input_prefix", help="Input prefix; expects <input_prefix>.bam")
    parser.add_argument("ref", help="Reference prefix; expects <ref>.exclude.bed")
    parser.add_argument("output_prefix", help="Output prefix for generated bigWig files")
    parser.add_argument("-q", "--minMappingQuality", type=int, default=0, help="Minimum mapping quality (default: 0)")
    parser.add_argument("-b", "--binSize", type=int, default=1, help="Bin size for bamCoverage (default: 1)")
    parser.add_argument(
        "-n",
        "--normalizeUsing",
        choices=["RPKM", "CPM", "BPM", "RPGC", "None"],
        default="CPM",
        help="Normalization method (default: CPM)",
    )
    parser.add_argument("-t", "--numberOfProcessors", type=int, default=8, help="Number of processors (default: 8)")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> int:
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        logging.error("bamCoverage failed with exit code %s", e.returncode)
        return e.returncode
    except FileNotFoundError:
        logging.error("Python interpreter not found: %s", SEQ_ENV_PYTHON)
        return 127


def main() -> int:
    args = parse_args()
    logging.info("Workdir: %s", os.getcwd())
    logging.info("Args: %s", vars(args))

    bam = f"{args.input_prefix}.bam"
    exclude_bed = f"{args.ref}.exclude.bed"

    if not os.path.isfile(bam):
        logging.error("BAM file not found: %s", bam)
        return 1
    if not os.path.isfile(exclude_bed):
        logging.error("Exclude BED not found: %s", exclude_bed)
        return 1

    logging.info("Input: %s", bam)
    logging.info("Reference: %s", args.ref)
    logging.info("Exclude: %s", exclude_bed)
    logging.info("Output prefix: %s", args.output_prefix)

    output_plus = f"{args.output_prefix}_plus.q{args.minMappingQuality}.bin{args.binSize}.{args.normalizeUsing}.bw"
    output_minus = f"{args.output_prefix}_minus.q{args.minMappingQuality}.bin{args.binSize}.{args.normalizeUsing}.bw"


    logging.info("Generating forward strand coverage...")
    cmd_plus = _bamcoverage_cmd_prefix() + [
        "--bam",
        bam,
        "--outFileName",
        output_plus,
        "--minMappingQuality",
        str(args.minMappingQuality),
        "--filterRNAstrand",
        "forward",
        "--binSize",
        str(args.binSize),
        "--normalizeUsing",
        args.normalizeUsing,
        "--blackListFileName",
        exclude_bed,
        "--numberOfProcessors",
        str(args.numberOfProcessors),
    ]
    rc = run_cmd(cmd_plus)
    if rc != 0:
        raise RuntimeError(f"bamCoverage on forward strand failed with exit code {rc}")

    logging.info("Generating reverse strand coverage...")
    cmd_minus = _bamcoverage_cmd_prefix() + [
        "--bam",
        bam,
        "--outFileName",
        output_minus,
        "--minMappingQuality",
        str(args.minMappingQuality),
        "--filterRNAstrand",
        "reverse",
        "--binSize",
        str(args.binSize),
        "--normalizeUsing",
        args.normalizeUsing,
        "--blackListFileName",
        exclude_bed,
        "--numberOfProcessors",
        str(args.numberOfProcessors),
    ]
    rc = run_cmd(cmd_minus)
    if rc != 0:
        raise RuntimeError(f"bamCoverage on reverse strand failed with exit code {rc}")

    logging.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

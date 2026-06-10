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

# Same bamCoverage entry as RNA/bam2bw_RNA.py (seq env python + deeptools on sys.path).
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
    parser = argparse.ArgumentParser(description="Generate normalized ATAC bigWig using bamCoverage.")
    parser.add_argument("input_prefix", help="Input prefix; expects <input_prefix>.align.sorted.bam")
    parser.add_argument("ref", help="Reference prefix; expects <ref>.genome_size and <ref>.exclude.bed")
    parser.add_argument("output_prefix", help="Output prefix for generated bigWig")
    parser.add_argument("-q", "--minMappingQuality", type=int, default=30, help="Minimum mapping quality (default: 30)")
    parser.add_argument("-b", "--binSize", type=int, default=1, help="Bin size for bamCoverage (default: 1)")
    parser.add_argument(
        "-n",
        "--normalizeUsing",
        choices=["RPKM", "CPM", "BPM", "RPGC", "None"],
        default="RPGC",
        help="Normalization method (default: RPGC)",
    )
    parser.add_argument("-t", "--numberOfProcessors", type=int, default=8, help="Number of processors (default: 8)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.info("Workdir: %s", os.getcwd())
    logging.info("Args: %s", vars(args))

    bam = f"{args.input_prefix}.align.sorted.bam"
    genome_size_file = f"{args.ref}.genome_size"
    exclude_bed = f"{args.ref}.exclude.bed"

    if not os.path.isfile(bam):
        logging.error("BAM file not found: %s", bam)
        return 1

    try:
        with open(genome_size_file, "r", encoding="utf-8") as f:
            genome_size = f.read().strip()
    except FileNotFoundError:
        logging.error("Genome size file not found: %s", genome_size_file)
        return 1

    if not genome_size:
        logging.error("Genome size file is empty: %s", genome_size_file)
        return 1

    output_bw = f"{args.output_prefix}.q{args.minMappingQuality}.bin{args.binSize}.{args.normalizeUsing}.bw"

    logging.info("Generating coverage ...")

    cmd = _bamcoverage_cmd_prefix() + [
        "--bam",
        bam,
        "--outFileName",
        output_bw,
        "--minMappingQuality",
        str(args.minMappingQuality),
        "--binSize",
        str(args.binSize),
        "--normalizeUsing",
        args.normalizeUsing,
        "--effectiveGenomeSize",
        genome_size,
        "--blackListFileName",
        exclude_bed,
        "--numberOfProcessors",
        str(args.numberOfProcessors),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("bamCoverage failed with exit code %s", e.returncode)
        return e.returncode
    except FileNotFoundError:
        logging.error("Python interpreter not found: %s", SEQ_ENV_PYTHON)
        return 127

    logging.info("Done: %s", output_bw)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-27

import argparse
import logging
import subprocess
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align paired-end FASTQ with HISAT2 and sort to BAM with samtools.",
    )
    parser.add_argument("input_prefix", help="Expects <input_prefix>_R1.fastq.gz and _R2.fastq.gz")
    parser.add_argument("ref", help="HISAT2 index prefix (-x)")
    parser.add_argument("output_prefix", help="Output prefix for <output_prefix>.bam")
    parser.add_argument("-p", "--threads", type=int, default=8, help="Threads for hisat2 and samtools (default: 8)")
    parser.add_argument(
        "-s", "--rna-strandness",
        dest="rna_strandness",
        choices=["F", "R", "FR", "RF", "None"],
        default="None",
        help="Passed to hisat2 --rna-strandness when not None (default: None, unstranded)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_prefix = args.input_prefix
    ref = args.ref
    output_prefix = args.output_prefix
    threads = args.threads

    r1 = f"{input_prefix}_R1.fastq.gz"
    r2 = f"{input_prefix}_R2.fastq.gz"

    logging.info("Workdir: %s", Path.cwd())
    logging.info("Args: %s", " ".join(sys.argv))

    if not Path(r1).is_file() or not Path(r2).is_file():
        logging.error("Input files not found: %s and/or %s", r1, r2)
        return 1

    hisat_cmd: list[str] = [
        "hisat2",
        "-p",
        str(threads),
        "-x",
        ref,
        "-1",
        r1,
        "-2",
        r2,
    ]
    if args.rna_strandness != "None":
        hisat_cmd.extend(["--rna-strandness", args.rna_strandness])

    sort_cmd: list[str] = [
        "samtools",
        "sort",
        "-@",
        str(threads),
        "-T",
        output_prefix,
        "--write-index",
        "-o",
        f"{output_prefix}.bam",
        "-",
    ]

    logging.info("Starting alignment...")
    p1 = subprocess.Popen(hisat_cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(sort_cmd, stdin=p1.stdout)
    if p1.stdout is not None:
        p1.stdout.close()
    rc2 = p2.wait()
    rc1 = p1.wait()
    if rc1 != 0:
        logging.error("hisat2 failed with exit code %s", rc1)
        return rc1
    if rc2 != 0:
        logging.error("samtools sort failed with exit code %s", rc2)
        return rc2

    logging.info("calc stats...")
    stats_cmd = [
        "samtools",
        "stats",
        f"{output_prefix}.bam",
        "-@",
        str(threads),
    ]
    stat_path = f"{output_prefix}.bam.stat"
    with open(stat_path, "w", encoding="utf-8") as f:
        subprocess.run(stats_cmd, check=True, stdout=f)

    logging.info("Done: %s.bam", output_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-24

set -euo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <input_prefix> <ref> <output_prefix>"
    exit 1
fi

INPUT_PREFIX=$1
REF=$2
OUTPUT_PREFIX=$3

READS="${INPUT_PREFIX}_SE.fastq.gz"

if [ ! -f "$READS" ]; then
    echo "Error: Input files not found"
    exit 1
fi

THREADS=8

echo "Aligning $INPUT_PREFIX..."

bowtie2 \
    -q --no-unal --threads $THREADS --sensitive \
    -x $REF -U $READS \
    -k 2 --local \
    2> ${OUTPUT_PREFIX}.bowtie2.log | \
    samtools sort -@ $THREADS -m 2G -T ${OUTPUT_PREFIX} -o ${OUTPUT_PREFIX}.align.sorted.bam - --write-index


echo "Done: ${OUTPUT_PREFIX}.align.sorted.bam"
grep "overall alignment rate" ${OUTPUT_PREFIX}.bowtie2.log

exit 0


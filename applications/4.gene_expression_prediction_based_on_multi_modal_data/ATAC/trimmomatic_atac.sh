#!/bin/bash
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-24

#===============================================================================
# Script: trim_atac.sh
# Description: Trimmomatic wrapper for HiSeq XTen PE150 ATAC-seq data
# Usage: ./trim_atac.sh [-t THREADS] <input_prefix> <output_prefix>
# Example: ./trim_atac.sh sample01 sample01_trimmed
# Example: ./trim_atac.sh -t 16 sample01 sample01_clean
#===============================================================================

THREADS=8
usage() {
    echo "Usage: $0 [-t THREADS] <input_prefix> <output_prefix>"
    echo "  -t THREADS   Trimmomatic -threads (default: 8)"
    echo ""
    echo "Expected input files: <input_prefix>_R1.fastq.gz and <input_prefix>_R2.fastq.gz"
}

while getopts "t:h" opt; do
    case $opt in
        t)
            THREADS=$OPTARG
            ;;
        h)
            usage
            exit 0
            ;;
        ?)
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -ne 2 ]; then
    usage
    exit 1
fi

if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: -t must be a positive integer, got: $THREADS"
    exit 1
fi

INPUT_PREFIX=$1
OUTPUT_PREFIX=$2

echo "workdir: $(pwd)"
echo "input: ${INPUT_PREFIX}" 
echo "output: ${OUTPUT_PREFIX}"

# Input files
R1="${INPUT_PREFIX}_R1.fastq.gz"
R2="${INPUT_PREFIX}_R2.fastq.gz"

# Check if input files exist
if [ ! -f "$R1" ]; then
    echo "Error: Input file not found: $R1"
    exit 1
fi

if [ ! -f "$R2" ]; then
    echo "Error: Input file not found: $R2"
    exit 1
fi


# Output files
R1_PAIRED="${OUTPUT_PREFIX}_R1_paired.fastq.gz"
R1_UNPAIRED="${OUTPUT_PREFIX}_R1_unpaired.fastq.gz"
R2_PAIRED="${OUTPUT_PREFIX}_R2_paired.fastq.gz"
R2_UNPAIRED="${OUTPUT_PREFIX}_R2_unpaired.fastq.gz"

# Trimmomatic settings
BINDIR="/mnt/zzb/default/Workspace/xuyu/software/source/Trimmomatic-0.40"
ADAPTER_FILE="$BINDIR/adapters/NexteraPE-PE.fa"

# Check if adapter file exists, create if not
if [ ! -f "$ADAPTER_FILE" ]; then
    echo "adapter file not exist"
    echo "should be at $ADAPTER_FILE"
    exit 1
fi

# Run Trimmomatic
echo "$(date) Starting Trimmomatic..."
java -jar  $BINDIR/trimmomatic-0.40.jar PE \
    -threads $THREADS -phred33 \
    -summary ${OUTPUT_PREFIX}.trim.stat \
    "$R1" "$R2" \
    "$R1_PAIRED" "$R1_UNPAIRED" \
    "$R2_PAIRED" "$R2_UNPAIRED" \
    ILLUMINACLIP:NexteraPE-PE.fa:2:30:10:2 \
    MINLEN:35

    # ILLUMINACLIP:${ADAPTER_FILE}:2:30:10:2:keepBothReads \

   # use loose creteria: no quality filter
   #  LEADING:3 \
   #  TRAILING:3 \
   #  SLIDINGWINDOW:4:15 \
# Check exit status
if [ $? -eq 0 ]; then
	echo "Trimmomatic completed successfully!"
    echo "$(date) DONE"
    exit 0
else
    echo "Error: Trimmomatic failed!"
    exit 1
fi


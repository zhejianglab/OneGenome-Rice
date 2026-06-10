#!/bin/bash

source .env
# Example script for running the multimodal index builder

# Define your specific paths
GENOME_FASTA="data/external/hg38_cleaned.fa"
BIGWIG_DIR="data/processed/averaged_encode_track/"
META_CSV="data/external/media.csv"
OUTPUT_DIR="data/indices/test_NK_3track_chr19/"
TOKENIZER_PATH=$Genos_10b
UPDATED_TOKENIZER_PATH="data/tokenizer/test_NK_3track/"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🚀 Starting multimodal index building..."

/mnt/yecheng/conda/bin/python scripts/data_preprocess/build_multimodal_index_and_update_tokenizer.py \
    --genome_fasta $GENOME_FASTA \
    --bigwig_dir $BIGWIG_DIR \
    --meta_csv $META_CSV \
    --output_base_dir $OUTPUT_DIR \
    --chromosomes chr19 \
    --window_size 32768 \
    --overlap 16384 \
    --tokenizer_dir $TOKENIZER_PATH \
    --updated_tokenizer_dir $UPDATED_TOKENIZER_PATH \
    --assay_titles "total RNA-seq" "ATAC-seq" \
    --biosample_names "natural killer cell" \
    --output_json_file_path $OUTPUT_DIR/precomputed_track_means.json


if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    
    # List output files
    echo "📁 Output files:"
    ls -la $OUTPUT_DIR
else
    echo "❌ Pipeline failed!"
    exit 1
fi
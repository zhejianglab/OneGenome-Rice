

python scripts/data_preprocess/sequence_split_and_meta_extract.py \
  --genome_fasta data/external/osa1_r7.asm.ch.fa \
  --chromosomes 7 12 \
  --window_size 32768 \
  --overlap 16384 \
  --meta_csv data/external/media.csv \
  --assay_titles "total RNA-seq" \
  --biosample_names "NIP_Panicle1" \
  --output_base_dir "data/indices/test_multitrack" \
  --processed_bw_dir data/processed/renorm_bigwig_output

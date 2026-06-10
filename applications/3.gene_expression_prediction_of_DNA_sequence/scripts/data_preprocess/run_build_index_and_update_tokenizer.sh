source .env

/mnt/yecheng/conda/bin/python scripts/data_preprocess/build_index_and_update_tokenizer.py \
    --genome_fasta data/external/hg38_cleaned.fa \
    --bigwig_dir data/processed/averaged_encode_track \
    --track_index 27,133,298,404 \
    --output_base_dir data/indices/test_4_track_133_27_298_404 \
    --window_size 32768 \
    --overlap 16384 \
    --tokenizer_dir $Genos_10b \
    --updated_tokenizer_dir data/tokenizer/test_4_track_133_27_298_404 \
    --output_json_file_path data/indices/test_4_track_133_27_298_404/pre_calc_track_mean.json




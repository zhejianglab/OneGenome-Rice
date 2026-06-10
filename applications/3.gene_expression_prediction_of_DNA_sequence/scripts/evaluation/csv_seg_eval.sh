if [ ! -d outputs/0105-chr7 ]; then
  echo "Creating output directory outputs/0105-chr7"
  mkdir -p outputs/0105-chr7
fi


python ../RiceModel3-SFT-track_prediction/scripts/evaluation/csv_seg_eval.py \
 --csv "outputs/0105-chr7/biosample__Panicle1_all_RNAseq-_predictions.csv"\
 --output outputs/0105-chr7/chr7_minus --skip_bigwig \
 --chrom_sizes "data/external/chrom.sizes" \
 --expression_col "predicted_expression"

python ../RiceModel3-SFT-track_prediction/scripts/evaluation/csv_seg_eval.py \
 --csv "outputs/0105-chr7/biosample__Panicle1_all_RNAseq+_predictions.csv"\
 --output outputs/0105-chr7/chr7_plus --skip_bigwig \
 --chrom_sizes "data/external/chrom.sizes" \
 --expression_col "predicted_expression"

#! /bin/bash


dir="/path/to/bw_dataset"
rel_dir="/path/to/dataset"
outdir="/path/to/training_dir"
train_species_file=("P1" "P4" "P6")
valid_species_file=("P7")
test_species_file=("P11")
tissue=("CSQ" "YG")
changeMinus=0

rm bwList

for species_file in ${train_species_file[@]};do
    for atissue in ${tissue[@]};do
        ls ${dir}/${atissue}_${species_file}_1.bw >> bwList
    done
done


for species_file in ${valid_species_file[@]};do
    for atissue in ${tissue[@]};do
        ls ${dir}/${atissue}_${species_file}_1.bw >> bwList
    done
done

for species_file in ${test_species_file[@]};do
    for atissue in ${tissue[@]};do
        ls ${dir}/${atissue}_${species_file}_1.bw >> bwList
    done
done

#if the value in minus bw file is minus, change it
if [ $changeMinus -eq 1 ];then
    for bw_file in $(ls $dir/*minus*.bw);do
        file_name=$(basename $bw_file)
        name=${file_name%.*}
        species=${name%%_*}
        $dir/trans_minus2plus.sh $bw_file $dir/${name}2.bw ref/${species}*.chrom.sizes
        #rm $bw_file
    done
fi

# python ${outdir}/scripts/data_preprocess/renorm_bigwig.py  \
#      --input_list bwList \
#      --output_dir ${outdir}/data/processed/renorm_bigwig_output \
#      --common_length 100 \
#      --processes 16

for bw_file in $(cat bwList);do
    cp $bw_file ${outdir}/data/processed/renorm_bigwig_output/
done

tissues=""
for atissue in ${tissue[@]};do
    tissues=${tissues}_${atissue}
done

tissues=${tissues:1}

rm bwList
# 计算非零均值

for atissue in ${tissue[@]};do
    for species_file in ${train_species_file[@]};do
        ls ${dir}/${atissue}_${species_file}_1.bw >> bwList_${atissue}
    done
done

python ${outdir}/scripts/data_preprocess/csv.generator.py --tissues ${tissue[@]} --species_range ${train_species_file[@]} --bwlist bwList_* --output_dir ./ref
rm bwList_*

for atissue in ${tissue[@]};do
    for species_file in ${valid_species_file[@]};do
        ls ${dir}/${atissue}_${species_file}_1.bw >> bwList_${atissue}
    done
done

python ${outdir}/scripts/data_preprocess/csv.generator.py --tissues ${tissue[@]} --species_range ${valid_species_file[@]} --bwlist bwList_* --output_dir ./ref
rm bwList_* 

for atissue in ${tissue[@]};do
    for species_file in ${test_species_file[@]};do
        ls ${dir}/${atissue}_${species_file}_1.bw >> bwList_${atissue}
    done
done

python ${outdir}/scripts/data_preprocess/csv.generator.py --tissues ${tissue[@]} --species_range ${test_species_file[@]} --bwlist bwList_* --output_dir ./ref
rm bwList_*

#generate index index_stat.json and sequence_split_train.csv file
for species_file in ${train_species_file[@]};do
    species_file_dir=${dir}/ref/${species_file}-new.fasta
    species=$species_file
    python ${outdir}/scripts/data_preprocess/sequence_split_and_meta_extract2.py \
        --genome_fasta $species_file_dir \
        --chromosomes Chr01 Chr02 Chr03 Chr04 Chr05 Chr06 Chr07 Chr08 Chr09 Chr10 Chr11 Chr12 \
        --window_size 32768 \
        --overlap 16384 \
        --meta_csv ref/${tissues}_${species}.csv \
        --assay_titles "total RNA-seq" \
        --biosample_names "${tissue[@]}" \
        --output_base_dir ${outdir}/data/indices/test_${tissues}_${species}_multitrack \
        --processed_bw_dir ${outdir}/data/processed/renorm_bigwig_output
done

#generate index index_stat.json and sequence_split_train.csv file for validation
for species_file in ${valid_species_file[@]};do
    species_file_dir=${dir}/ref/${species_file}-new.fasta
    species=$species_file
    python ${outdir}/scripts/data_preprocess/sequence_split_and_meta_extract2.py \
        --genome_fasta $species_file_dir \
        --chromosomes Chr01 Chr02 Chr03 Chr04 Chr05 Chr06 Chr07 Chr08 Chr09 Chr10 Chr11 Chr12 \
        --window_size 32768 \
        --overlap 16384 \
        --meta_csv ref/${tissues}_${species}.csv \
        --assay_titles "total RNA-seq" \
        --biosample_names "${tissue[@]}" \
        --output_base_dir ${outdir}/data/indices/valid_${tissues}_${species}_multitrack \
        --processed_bw_dir ${outdir}/data/processed/renorm_bigwig_output
done

#generate index index_stat.json and sequence_split_train.csv file for test
for species_file in ${test_species_file[@]};do
    species_file_dir=${dir}/ref/${species_file}-new.fasta
    species=$species_file
    python ${outdir}/scripts/data_preprocess/sequence_split_and_meta_extract2.py \
        --genome_fasta $species_file_dir \
        --chromosomes Chr01 Chr02 Chr03 Chr04 Chr05 Chr06 Chr07 Chr08 Chr09 Chr10 Chr11 Chr12 \
        --window_size 32768 \
        --overlap 16384 \
        --meta_csv ref/${tissues}_${species}.csv \
        --assay_titles "total RNA-seq" \
        --biosample_names "${tissue[@]}" \
        --output_base_dir ${outdir}/data/indices/test_${tissues}_${species}_multitrack \
        --processed_bw_dir ${outdir}/data/processed/renorm_bigwig_output
done


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate sample-specific consensus sequences from a VCF and record reference
genome coordinate arrays (`pos_list`) for forward and reverse-complement inputs.
"""

import argparse
import json
import os
import pysam
import numpy as np
import pandas as pd
from cyvcf2 import VCF
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate block-level sample consensus sequence JSON files from a VCF."
    )

    parser.add_argument("--bed", required=True, help="BED file with block/window intervals.")
    parser.add_argument("--pheno", required=True, help="Phenotype file.")
    parser.add_argument("--vcf", required=True, help="VCF file path.")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA.")
    parser.add_argument("--pheno-col", default="Trait", help="Phenotype column name, for example Trait.")
    parser.add_argument("--out", default="json_blocks", help="Output directory.")
    parser.add_argument("--block_start", type=int, default=1, help="First block ID to process, inclusive and 1-based.")
    parser.add_argument("--block_end", type=int, default=0, help="Last block ID to process, inclusive and 1-based; 0 means all.")
    parser.add_argument("--sample-limit", type=int, default=0, help="Randomly keep at most this many matched samples; 0 means all.")
    parser.add_argument("--sample-seed", type=int, default=20260420, help="Random seed used with --sample-limit.")
    
    return parser.parse_args()


chrom_map = {
    '1': '1', '2': '2', '3': '3',
    '4': '4', '5': '5', '6': '6',
    '7': '7', '8': '8', '9': '9',
    '10': '10', '11': '11', '12': '12'
}


def classify_variant(ref, alt):
    """Classify a variant by allele lengths."""
    ref = str(ref).upper()
    alt = str(alt).upper()

    len_ref = len(ref)
    len_alt = len(alt)

    if len_ref == 1 and len_alt == 1:
        return 'SNP'
    elif len_ref < len_alt:
        return 'INS'
    elif len_ref > len_alt:
        return 'DEL'
    else:
        return 'COMPLEX'

def get_reverse_complement(seq):
    """Return the reverse-complement sequence."""
    complement_map = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(complement_map)[::-1]

def encode_label(value):
    """Keep binary labels as ints while allowing continuous phenotypes."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return int(number)
    return number

def apply_variant_to_sequence_with_pos(ref_seq, start_pos, variant_list):
    """
    Apply variants to a reference sequence and update the coordinate array.
    """
    seq_list = list(ref_seq)
    pos_list = [start_pos + i for i in range(len(ref_seq))]
    
    # Apply variants from right to left to avoid coordinate shifts.
    variant_list = sorted(variant_list, key=lambda x: x[0], reverse=True)

    for offset, var_type, ref, alt in variant_list:
        if offset < 0 or offset >= len(seq_list):
            continue
            
        current_genomic_pos = start_pos + offset

        if var_type == 'SNP':
            seq_list[offset] = alt
            
        elif var_type == 'INS':
            seq_list[offset] = alt[0]
            insert_bases = alt[1:]
            for i, base in enumerate(insert_bases):
                seq_list.insert(offset + 1 + i, base)
                # Inserted bases share the preceding reference coordinate.
                pos_list.insert(offset + 1 + i, current_genomic_pos)

        elif var_type == 'DEL':
            seq_list[offset] = alt[0] if len(alt) > 0 else 'N'
            for i in range(1, len(ref)):
                if offset + i < len(seq_list):
                    seq_list[offset + i] = 'N'

        elif var_type == 'COMPLEX':
            seq_list[offset] = alt[0] if len(alt) > 0 else 'N'
            if len(ref) > 1:
                for i in range(1, len(ref)):
                    if offset + i < len(seq_list):
                        seq_list[offset + i] = 'N'

    return ''.join(seq_list), pos_list


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Reading phenotype data ...")
    pheno = pd.read_csv(args.pheno, sep=r"\s+|,", engine="python")
    pheno = pheno.dropna(subset=[args.pheno_col])
    pheno = pheno.set_index("SampleID")

    print("Reading BED file ...")
    bed_df = pd.read_csv(args.bed, sep="\t", header=None, names=["chrom", "start", "end"])

    print("Opening FASTA file ...")
    print(f"DEBUG: args.fasta is {repr(args.fasta)}")
    fasta = pysam.FastaFile(args.fasta)

    print("Reading VCF file ...")
    vcf = VCF(args.vcf)
    vcf_samples = vcf.samples
    
    pheno_samples = [s for s in pheno.index if s in vcf_samples]
    if args.sample_limit > 0 and len(pheno_samples) > args.sample_limit:
        rng = np.random.default_rng(args.sample_seed)
        keep_idx = sorted(rng.choice(len(pheno_samples), size=args.sample_limit, replace=False).tolist())
        pheno_samples = [pheno_samples[i] for i in keep_idx]
    sample_to_idx = {sid: i for i, sid in enumerate(vcf_samples)}
    pheno_idx = [sample_to_idx[s] for s in pheno_samples]
    labels = pheno.loc[pheno_samples, args.pheno_col].values
    
    print(f"Valid sample count: {len(pheno_samples)}")
    if not pheno_samples:
        raise ValueError(
            "No phenotype SampleID values matched VCF samples. "
            "Check the phenotype delimiter and sample naming."
        )

    print("\n" + "="*60)
    print("Processing blocks...")
    
    block_start = max(1, int(args.block_start))
    block_end = int(args.block_end)
    if block_end <= 0:
        block_end = len(bed_df)

    for block_id, row in tqdm(bed_df.iterrows(), total=len(bed_df), desc="Processing blocks"):
        block_no = block_id + 1
        if block_no < block_start or block_no > block_end:
            continue

        chrom = str(row['chrom']).replace('chr', '')
        start = int(row['start'])
        end = int(row['end'])
        block_name = f"block_{block_no}"

        if chrom not in chrom_map:
            continue

        try:
            ref_seq = fasta.fetch(chrom_map[chrom], start, end).upper()
        except:
            continue

        variants_in_block = []
        vcf_chrom_names = [chrom, chrom_map.get(chrom, chrom)]
        
        for vcf_chrom in vcf_chrom_names:
            try:
                for variant in vcf(f"{vcf_chrom}:{start}-{end}"):
                    variants_in_block.append(variant)
                if len(variants_in_block) > 0:
                    break
            except: pass

        if not variants_in_block:
            continue

        json_list = []
        # vcf start is 1-based, bed was 0-based. Let's make pos 1-based natively
        start_1based = start + 1

        for i, sample in enumerate(tqdm(pheno_samples, desc=f"  {block_name} samples", leave=False)):
            sample_idx = pheno_idx[i]
            sample_variants = []

            for variant in variants_in_block:
                gt = variant.genotypes[sample_idx]
                allele1, allele2 = gt[0], gt[1]

                if allele1 == -1 or allele2 == -1:
                    applied_allele = 'N' * len(variant.REF)
                    var_type = 'COMPLEX'
                elif allele1 == 0 and allele2 == 0:
                    continue 
                else:
                    if allele1 == 0:
                        allele_idx = allele2
                    elif allele2 == 0:
                        allele_idx = allele1
                    else:
                        allele_idx = max(allele1, allele2)
                    
                    if allele_idx == 0:
                        continue
                    else:
                        alt_index = allele_idx - 1
                        if alt_index < len(variant.ALT):
                            applied_allele = variant.ALT[alt_index]
                        else:
                            applied_allele = 'N' * len(variant.REF)
                            var_type = 'COMPLEX'
                            offset = variant.POS - start_1based
                            sample_variants.append((offset, var_type, variant.REF, applied_allele))
                            continue
                    
                    if applied_allele in ['DEL', '<DEL>', '*', '.']:
                        applied_allele = 'N' * len(variant.REF)
                        var_type = 'DEL'
                    else:
                        var_type = classify_variant(variant.REF, applied_allele)

                offset = variant.POS - start_1based
                sample_variants.append((offset, var_type, variant.REF, applied_allele))

            consensus_seq, pos_list = apply_variant_to_sequence_with_pos(ref_seq, start_1based, sample_variants)
            
            consensus_seq_revcomp = get_reverse_complement(consensus_seq)
            pos_list_revcomp = pos_list[::-1]

            json_list.append({
                "label": encode_label(labels[i]),
                "spec": sample,
                "loc": block_name,
                "sequence": consensus_seq,
                "pos_list": pos_list,
                "sequence_revcomp": consensus_seq_revcomp,
                "pos_list_revcomp": pos_list_revcomp
            })

        out_path = os.path.join(args.out, f"{block_name}.json")
        with open(out_path, "w") as f:
            json.dump(json_list, f)

    fasta.close()
    vcf.close()
    print("Finished generating sample consensus sequences and coordinate arrays from VCF.\n")

if __name__ == "__main__":
    main()

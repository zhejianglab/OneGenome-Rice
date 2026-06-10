#!/usr/bin/env python3
"""Generate per-sample RNA-seq metadata CSV files.

The important detail here is that nonzero_mean is computed for each concrete
BigWig file, not once per tissue list and then broadcast to every sample.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig


def read_bw_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def calc_nonzero_mean(bw_path, chunk_size=1_000_000):
    total = 0.0
    count = 0

    bw = pyBigWig.open(str(bw_path))
    if bw is None:
        raise RuntimeError(f"Could not open BigWig: {bw_path}")

    try:
        for chrom, length in bw.chroms().items():
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                values = np.array(bw.values(chrom, start, end), dtype=np.float64)
                values = values[(~np.isnan(values)) & (values != 0)]
                if values.size:
                    total += float(values.sum())
                    count += int(values.size)
    finally:
        bw.close()

    return total / count if count else 0.0


def create_template_row(tissue, track, species, nonzero_mean):
    return {
        "target_file_name": f"{tissue}_{species}_1.bw",
        "track_index": track,
        "data_source": "biobigdata",
        "output_type": "RNA_SEQ",
        "organism": tissue,
        "biosample_name": tissue,
        "Assay title": "total RNA-seq",
        "strand": "+",
        "nonzero_mean": nonzero_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fixed tissue-combination CSV files")
    parser.add_argument("--tissues", required=True, nargs="+")
    parser.add_argument("--output_dir", default="./tissue_combinations")
    parser.add_argument("--species_range", required=True, nargs="+")
    parser.add_argument("--bwlist", required=True, nargs="+")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if len(args.bwlist) != len(args.tissues):
        raise SystemExit(
            f"--bwlist count must match --tissues count: {len(args.bwlist)} != {len(args.tissues)}"
        )

    tissue_bw_paths = {
        tissue: read_bw_list(list_path)
        for tissue, list_path in zip(args.tissues, args.bwlist)
    }
    for tissue, paths in tissue_bw_paths.items():
        if len(paths) != len(args.species_range):
            raise SystemExit(
                f"{tissue}: bwlist has {len(paths)} files but species_range has "
                f"{len(args.species_range)} samples"
            )

    combo_name = "_".join(args.tissues)
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    mean_cache = {}
    for sample_idx, species in enumerate(args.species_range):
        rows = []
        for track, tissue in enumerate(args.tissues, start=1):
            bw_path = Path(tissue_bw_paths[tissue][sample_idx])
            if bw_path not in mean_cache:
                mean_cache[bw_path] = calc_nonzero_mean(bw_path)
            rows.append(create_template_row(tissue, track, species, mean_cache[bw_path]))

        df = pd.DataFrame(rows)
        output_path = Path(args.output_dir) / f"{combo_name}_{species}.csv"
        if args.dry_run:
            print(f"[dry-run] would create {output_path}")
            print(df.to_string(index=False))
        else:
            df.to_csv(output_path, index=False)
            print(f"Created {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()

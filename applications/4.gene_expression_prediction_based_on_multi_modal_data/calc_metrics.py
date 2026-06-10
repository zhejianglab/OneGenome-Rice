#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-24
"""
Compute strand-level metrics from inference outputs.

Usage:
    python calc_metrics.py <plus_input> <minus_input> <metrics_output_base>
    python calc_metrics.py ... [-c infer_or_experiment.yaml]

``metrics_output_base`` is the path used when only ``full`` metrics are written
(typically .../strand_metrics.txt).
"""

import argparse
import ast
import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _coerce_to_1d_array(value):
    """Convert CSV/pickle cell value into a 1D numpy array."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=False).reshape(-1)
    if isinstance(value, list):
        return np.asarray(value).reshape(-1)
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return np.asarray(parsed).reshape(-1)
    return np.asarray(value).reshape(-1)


def _flatten_from_df_dict(
    df_dict,
    *,
    region: Literal["full"] = "full",
):
    true_vals, pred_vals = [], []
    for df in df_dict.values():
        if not isinstance(df, pd.DataFrame):
            continue
        for _, row in df.iterrows():
            t = _coerce_to_1d_array(row["true_expression"])
            p = _coerce_to_1d_array(row["predicted_expression"])
            m = None
            if "region_mask" in row and row["region_mask"] is not None:
                m = _coerce_to_1d_array(row["region_mask"]).astype(np.float64, copy=False)
            if m is not None:
                keep = m > 0.5
                t = t[keep]
                p = p[keep]
            true_vals.extend(t)
            pred_vals.extend(p)
    return np.asarray(true_vals), np.asarray(pred_vals)


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    """Compute log1p for predictions/targets that should be non-negative.

    If negative values appear due to scaling/artifacts, clamp to 0 to avoid NaNs.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.maximum(x, 0.0))


def _calc(true, pred, name):
    if len(true) == 0 or len(pred) == 0:
        return {}

    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    m = np.isfinite(true) & np.isfinite(pred)
    true = true[m]
    pred = pred[m]
    if len(true) == 0:
        return {}

    r2 = r2_score(true, pred)
    pcc = pearsonr(true, pred)[0] if np.std(true) > 1e-8 and np.std(pred) > 1e-8 else np.nan

    true_l = _safe_log1p(true)
    pred_l = _safe_log1p(pred)
    pcc_log1p = (
        pearsonr(true_l, pred_l)[0]
        if np.std(true_l) > 1e-8 and np.std(pred_l) > 1e-8
        else np.nan
    )
    return {
        f"{name}_R2": r2,
        f"{name}_Pearson": pcc,
        f"{name}_Pearson_log1p": pcc_log1p,
    }


def _load_from_pickle(path):
    with open(path, "rb") as f:
        plus_dfs = pickle.load(f)
    logger.info("Loaded prediction pickle: %s", path)
    return plus_dfs


def _load_from_csv(path):
    df = pd.read_csv(path)
    key = os.path.basename(path).replace(".csv", "")
    logger.info("Loaded prediction CSV: %s", path)
    return {key: df}


def _load_input(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.endswith(".pickle"):
        return _load_from_pickle(path)
    if path.endswith(".csv"):
        return _load_from_csv(path)
    raise ValueError(f"Unsupported input format for {path}. Use .pickle or .csv")


def resolve_metric_output_paths(
    metrics_output_base: str, regions: Sequence[str]
) -> dict[str, str]:
    """Map region name -> output file path (see module docstring)."""
    base = Path(metrics_output_base)
    regions_set = set(regions)
    if regions_set == {"full"}:
        return {"full": str(base)}
    raise ValueError(f"Unsupported metric regions: {regions!r}")


def calculate_metrics_from_files(
    plus_input,
    minus_input,
    *,
    regions: Sequence[str] = ("full",),
):
    plus_dfs = _load_input(plus_input)
    minus_dfs = _load_input(minus_input)
    regions = ("full",)
    out: dict[str, dict] = {}
    for region in regions:
        if region == "full":
            t_p, p_p = _flatten_from_df_dict(plus_dfs, region="full")
            t_m, p_m = _flatten_from_df_dict(minus_dfs, region="full")
        else:
            raise ValueError(region)
        m: dict = {}
        m.update(_calc(t_p, p_p, "plus"))
        m.update(_calc(t_m, p_m, "minus"))
        out[region] = m
    return out


def save_metrics_txt(metrics, metrics_output):
    out_dir = os.path.dirname(metrics_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(metrics_output, "w") as f:
        for k, v in sorted(metrics.items()):
            line = f"{k}: {v:.6f}" if not np.isnan(v) else f"{k}: NaN"
            f.write(line + "\n")
            logger.info(line)
    logger.info("Saved metrics file: %s", metrics_output)


def calculate_and_save_metrics(
    plus_input,
    minus_input,
    metrics_output_base,
):
    by_region = calculate_metrics_from_files(plus_input, minus_input, regions=("full",))
    paths = resolve_metric_output_paths(metrics_output_base, ["full"])
    save_metrics_txt(by_region["full"], paths["full"])
    return by_region


def main():
    ap = argparse.ArgumentParser(description="Strand-level metrics from inference pickles/CSVs.")
    ap.add_argument("plus_input", help="Plus-strand predictions (.pickle or .csv)")
    ap.add_argument("minus_input", help="Minus-strand predictions (.pickle or .csv)")
    ap.add_argument(
        "metrics_output_base",
        help="Output path for metrics (full only)",
    )
    ap.add_argument(
        "-c",
        "--config",
        metavar="YAML",
        help="Optional YAML path (currently unused by calc_metrics).",
    )
    args = ap.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            _ = yaml.safe_load(f) or {}

    calculate_and_save_metrics(
        args.plus_input,
        args.minus_input,
        args.metrics_output_base,
    )


if __name__ == "__main__":
    main()

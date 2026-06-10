# Author: Yu XU <xuyu@genomics.cn>
# Created: 2026-03-28
"""Label scaling and prediction unscaling (shared by training and inference)."""

from __future__ import annotations

import numpy as np


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Length-weighted q-th quantile (non-zero-filtered values; same CDF rule as bw_hist.py)."""
    sorter = np.argsort(values)
    sv = values[sorter]
    sw = weights[sorter]
    cumw = np.cumsum(sw)
    total = float(cumw[-1])
    if total <= 0:
        return float(sv[0])
    frac = (cumw - sw / 2.0) / total
    return float(np.interp(q, frac, sv))


class LabelScaler:
    """Fits cap_threshold + track_mean from BigWig intervals, then transforms labels."""

    def __init__(self, track_mean: float, cap_threshold: float | None = None):
        self.track_mean = float(track_mean)
        self.cap_threshold = float(cap_threshold) if cap_threshold is not None else None

    @classmethod
    def fit(cls, intervals, cap_expression_quantile: float | None = None) -> LabelScaler:
        """Fit from pyBigWig ``intervals(chrom)`` list of ``(start, end, value)``.

        Uses non-zero intervals only (same convention as bw_hist.py). If ``cap_expression_quantile``
        is set, computes a length-weighted quantile on non-zero values, caps values, then
        computes ``track_mean`` as the length-weighted mean of capped non-zero values.
        """
        if not intervals:
            return cls(track_mean=1.0, cap_threshold=None)

        vals_list: list[float] = []
        spans_list: list[float] = []
        for start, end, value in intervals:
            if value is None:
                continue
            v = float(value)
            if not np.isfinite(v):
                continue
            span = float(end - start)
            if span <= 0:
                continue
            if v == 0.0:
                continue
            vals_list.append(v)
            spans_list.append(span)

        if not vals_list:
            return cls(track_mean=1.0, cap_threshold=None)

        vals = np.asarray(vals_list, dtype=np.float64)
        spans = np.asarray(spans_list, dtype=np.float64)

        cap_threshold: float | None = None
        if cap_expression_quantile is not None:
            q = float(cap_expression_quantile)
            if not 0.0 <= q <= 1.0:
                raise ValueError(
                    f"cap_expression_quantile must be in [0, 1], got {cap_expression_quantile!r}"
                )
            cap_threshold = _weighted_quantile(vals, spans, q)
            vals = np.minimum(vals, cap_threshold)

        track_mean = float(np.sum(vals * spans) / np.sum(spans))
        return cls(track_mean=track_mean, cap_threshold=cap_threshold)

    def transform(self, v: np.ndarray) -> np.ndarray:
        """Upper-cap (if fitted) → normalize by track_mean → power + squash."""
        v = np.asarray(v, dtype=np.float64)
        if self.cap_threshold is not None:
            v = np.minimum(v, self.cap_threshold)
        v = v / self.track_mean
        v = v**0.75
        return np.where(v > 10.0, 2 * np.sqrt(v * 10.0) - 10.0, v)

    def inverse_transform(self, v: np.ndarray) -> np.ndarray:
        """Inverse squash → inverse power → multiply by track_mean."""
        v = np.asarray(v, dtype=np.float64)
        v = np.where(v > 10.0, (v + 10.0) ** 2 / (4 * 10.0), v)
        v = v ** (1.0 / 0.75)
        return np.nan_to_num(v * self.track_mean, nan=0.0)


def cap_threshold_from_index_row(row, col: str):
    """Return ``None`` if column missing or NaN (backward-compatible with old index CSV)."""
    if col not in row.index:
        return None
    v = row[col]
    try:
        import pandas as pd

        if pd.isna(v):
            return None
    except (ImportError, TypeError, ValueError):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
    return float(v)

"""Data Cleaning and Data Quality — acmnxt.core.dq

Functions:
- clean_time(df): sort by time, drop duplicate timestamps, enforce monotonic Ts
- resample_numeric(df, rule='1min'): resample numeric columns uniformly
- compute_dq(df): per-tag DQ metrics and per-sample dq_bad mask

All timestamps must be UTC. Non-numeric columns are ignored.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def clean_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, drop duplicate timestamps (keep last), ensure monotonic.

    Args:
        df: DataFrame with DateTimeIndex.

    Returns:
        Cleaned DataFrame with strictly increasing timestamps.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("clean_time requires DateTimeIndex")
    out = df.copy()
    out = out.sort_index()
    # Drop exact duplicate timestamps (keep last occurrence)
    out = out[~out.index.duplicated(keep="last")]
    return out


def resample_numeric(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """Uniformly resample numeric columns using mean aggregation.

    Args:
        df: DataFrame with DateTimeIndex.
        rule: Pandas offset alias, e.g., '1min'.

    Returns:
        Numeric-only DataFrame, resampled.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("resample_numeric requires DateTimeIndex")
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return num
    return num.resample(rule).mean()


def compute_dq(
    df: pd.DataFrame,
    flatline_eps: float = 1e-12,
    spike_z: float = 6.0,
    nan_pct_bad: float = 0.2,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute per-tag DQ metrics and overall dq_bad mask per timestamp.

    Metrics per tag:
      - nan_pct: fraction of NaNs
      - flatline_ratio: fraction of first differences with |d1| < eps
      - spike_ratio: fraction of |d1| beyond robust MAD threshold
      - dropout_runs: count of NaN runs (len>=2)
      - dq_flag: True if any metric exceeds heuristic bad thresholds

    Args:
        df: Numeric DataFrame with DateTimeIndex.
        flatline_eps: Absolute epsilon to consider flatline in first diff.
        spike_z: Robust z threshold (MAD-based) for spikes in first diff.
        nan_pct_bad: Threshold to flag tag as bad due to NaNs.

    Returns:
        (metrics_df, dq_bad_series)
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=bool)
    num = df.select_dtypes(include=[np.number])
    metrics = []
    for col in num.columns:
        s = num[col]
        nan_mask = s.isna()
        nan_pct = float(nan_mask.mean())
        # Fill for diff computation to avoid NaN propagating
        filled = s.ffill().bfill()
        d1 = filled.diff().abs()
        flatline_ratio = float((d1 < flatline_eps).mean())
        # Robust spike threshold on d1
        med = float(d1.median()) if not d1.isna().all() else 0.0
        mad = float((d1 - med).abs().median()) if not d1.isna().all() else 0.0
        thr = med + spike_z * 1.4826 * (mad + 1e-12)
        spike_ratio = float((d1 > thr).mean())
        # Count NaN runs (len>=2)
        runs = 0
        in_run = False
        run_len = 0
        for flag in nan_mask.astype(bool):
            if flag:
                run_len += 1
                if run_len == 2:
                    in_run = True
            else:
                if in_run:
                    runs += 1
                in_run = False
                run_len = 0
        if in_run:
            runs += 1
        dq_flag = (nan_pct >= nan_pct_bad) or (flatline_ratio > 0.95) or (spike_ratio > 0.10)
        metrics.append({
            "tag": col,
            "nan_pct": nan_pct,
            "flatline_ratio": flatline_ratio,
            "spike_ratio": spike_ratio,
            "dropout_runs": runs,
            "dq_flag": dq_flag,
        })
    mdf = pd.DataFrame(metrics).set_index("tag").sort_values(["dq_flag", "nan_pct", "flatline_ratio", "spike_ratio"], ascending=False)
    # Per-sample dq_bad if any tag NaN at that timestamp
    dq_bad = num.isna().any(axis=1).rename("dq_bad")
    return mdf, dq_bad

"""Feature Engineering — acmnxt.core.features

Pipeline sketch:
    raw df ──> clean/resample ──> select tags ──> build_features ──> features.parquet

Dataflow:
- Input: pd.DataFrame with DateTimeIndex (UTC), numeric tag columns only.
- Output: pd.DataFrame of same row count; NaN warmup masked but rows retained.

Functions to implement (see TASKS.md):
- build_features(df, cfg): rolling mean/std, deltas, z-scores (multi-window)
- Frequency features via Welch PSD; dominant freq; spectral flatness
- Context features: lags and AR(1) residuals

All computations must be vectorized where possible.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, windows: Iterable[int] = (5, 15, 60)) -> pd.DataFrame:
    """Builds a set of time-series features from numeric tag dataframe.

    Args:
        df: Numeric DataFrame with DateTimeIndex in UTC.
        windows: Rolling windows (in samples) to compute statistics.

    Returns:
        DataFrame with engineered features. Same index as input; NaN warmups retained.

    Raises:
        ValueError: If index is not a DateTimeIndex or df is empty.

    Example:
        >>> out = build_features(pd.DataFrame({"a": [1,2,3]}, index=pd.date_range('2020-01-01', periods=3, freq='T')))
        >>> out.shape[0] == 3
        True
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input df must have DateTimeIndex")
    if df.empty:
        raise ValueError("Input df is empty")

    num = df.select_dtypes(include=[np.number])
    feats = []

    # Identity features (baseline)
    base = num.add_prefix("raw_")
    feats.append(base)

    # Deltas
    deltas = num.diff().add_prefix("d1_")
    feats.append(deltas)

    # Rolling stats and z-scores
    for w in windows:
        roll_mean = num.rolling(w, min_periods=1).mean().add_prefix(f"r{w}_mean_")
        roll_std = num.rolling(w, min_periods=1).std(ddof=0).add_prefix(f"r{w}_std_")
        feats.extend([roll_mean, roll_std])
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (num - roll_mean.values) / (roll_std.values + 1e-9)
        feats.append(pd.DataFrame(z, index=num.index, columns=[f"r{w}_z_{c}" for c in num.columns]))

    # NOTE: Frequency and AR(1) residuals to be added in M2 (see TASKS.md)

    out = pd.concat(feats, axis=1)
    out.index = df.index
    return out


"""H1 — AR(1) Residual Z-Score

Fast trend anomaly detector using AR(1) residuals normalized to [0, 1].

Input: numeric matrix (features or selected raw tags)
Output: pd.Series of scores in [0, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_h1(df: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    """Compute H1 AR(1) residual z-score anomaly.

    Args:
        df: Numeric DataFrame with DateTimeIndex.
        alpha: AR(1) factor between 0 and 1.

    Returns:
        pd.Series of scores in [0, 1], indexed like df.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    x = df.select_dtypes(include=[np.number])
    # AR(1) residuals per column, aggregated by RMS
    resid = x - x.shift(1).bfill() * alpha
    z = (resid - resid.rolling(60, min_periods=1).mean()) / (resid.rolling(60, min_periods=1).std(ddof=0) + 1e-9)
    rms = np.sqrt(np.nanmean(np.square(z), axis=1))
    # Map to [0,1] via logistic
    scores = 1 / (1 + np.exp(-rms))
    return pd.Series(scores, index=df.index, name="H1_AR1")

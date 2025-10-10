from __future__ import annotations

import numpy as np
import pandas as pd


def score_h1(df: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    x = df.select_dtypes(include=[float, int])
    resid = x - x.shift(1).bfill() * alpha
    z = (resid - resid.rolling(60, min_periods=1).mean()) / (
        resid.rolling(60, min_periods=1).std(ddof=0) + 1e-9
    )
    rms = np.sqrt(np.nanmean(np.square(z), axis=1))
    s = 1 / (1 + np.exp(-rms))
    return pd.Series(s, index=df.index, name="H1_Forecast")


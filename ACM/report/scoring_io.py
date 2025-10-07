from __future__ import annotations

import pandas as pd
import numpy as np


def load_scored(path_or_df) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    elif isinstance(path_or_df, str):
        if path_or_df.lower().endswith(".parquet"):
            df = pd.read_parquet(path_or_df)
        else:
            df = pd.read_csv(path_or_df)
    else:
        df = pd.DataFrame()

    # Assume scored has Ts or index; attempt to coerce
    if "Ts" in df.columns:
        df["Ts"] = pd.to_datetime(df["Ts"], errors="coerce")
        df = df.dropna(subset=["Ts"]).set_index("Ts").sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        # fabricate time index
        df.index = pd.date_range("2000-01-01", periods=len(df), freq="T")

    # Ensure a fused score exists; if not, synthesize from top-variance numeric cols
    if "FusedScore" not in df.columns:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= 1:
            z = (num - num.mean()) / (num.std() + 1e-9)
            df["FusedScore"] = z.mean(axis=1).clip(lower=0)
        else:
            df["FusedScore"] = 0.0
    return df


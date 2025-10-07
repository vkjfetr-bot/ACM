from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple


def _coerce_ts(df: pd.DataFrame) -> pd.DataFrame:
    if "Ts" in df.columns:
        df = df.copy()
        df["Ts"] = pd.to_datetime(df["Ts"], errors="coerce")
        df = df.dropna(subset=["Ts"])  # drop bad timestamps
        return df.set_index("Ts").sort_index()
    # fabricate a time index if missing
    idx = pd.date_range("2000-01-01", periods=len(df), freq="T")
    df = df.copy()
    df.index = idx
    return df


def load_train_test(train_csv: str, test_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    tr = _coerce_ts(tr)
    te = _coerce_ts(te)
    # keep numeric columns only
    tr = tr.select_dtypes(include=[np.number])
    te = te.select_dtypes(include=[np.number])
    return tr, te


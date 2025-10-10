from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def ensure_datetime_index(df: pd.DataFrame, ts_col: str = "Ts") -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        cols_lower = {c.lower(): c for c in df.columns}
        candidates = [ts_col.lower(), "ts", "timestamp", "time", "datetime"]
        found = next((cols_lower.get(c) for c in candidates if c in cols_lower), None)
        if found is None:
            # assume first column is time-like
            found = df.columns[0]
        idx = pd.DatetimeIndex(pd.to_datetime(df[found], errors="coerce", utc=True))
        df = df.drop(columns=[found]).copy()
    # normalize to UTC then tz-naive for plots
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx
    return df.sort_index()

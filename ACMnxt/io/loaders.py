"""Data Loaders — CSV/XLSX to DataFrame with DateTimeIndex.

Handles timezone normalization to UTC and flexible timestamp columns.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    return df


def ensure_datetime_index(df: pd.DataFrame, ts_col: str = "Ts", tz: str | None = None) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        cols_lower = {c.lower(): c for c in df.columns}
        candidates = [ts_col.lower(), "ts", "timestamp", "time", "datetime", "date_time"]
        found = next((cols_lower[c] for c in candidates if c in cols_lower), None)
        if found is None:
            # fallback: try first column
            found = df.columns[0]
        ts = pd.to_datetime(df[found], utc=True, errors="coerce")
        df = df.drop(columns=[found]).copy()
        df.index = ts
    if tz:
        df.index = df.index.tz_convert(tz)
        df.index = df.index.tz_convert("UTC")
    else:
        # Ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    return df.sort_index()

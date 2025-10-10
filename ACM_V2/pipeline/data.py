"""Data ingestion, cleaning, and quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataWindow:
    raw: pd.DataFrame
    clean: pd.DataFrame
    dq: pd.DataFrame
    tags: Iterable[str]


def load_raw_data(
    source: Path,
    *,
    equip: str,
    t0: Optional[datetime] = None,
    t1: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load raw data for an equipment asset.

    Currently reads from a CSV file; replace with SQL call when available.
    """
    if not source.exists():
        raise FileNotFoundError(f"CSV source not found: {source}")
    df = pd.read_csv(source)
    # TODO: Replace file read with usp_GetEquipmentWindow(@equip, @t0, @t1).
    if "Ts" in df.columns:
        df["Ts"] = pd.to_datetime(df["Ts"])
        df = df.set_index("Ts")
    else:
        df.index = pd.to_datetime(df.index)
    if t0:
        df = df[df.index >= t0]
    if t1:
        df = df[df.index <= t1]
    df = df.sort_index()
    return df


def resample_and_clean(
    df: pd.DataFrame,
    *,
    resample_rule: str,
    clamp_sigma: float = 6.0,
) -> Tuple[pd.DataFrame, Iterable[str]]:
    """Resample numeric columns, fill gaps, and clamp outliers."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.resample(resample_rule).mean()
    numeric_df = numeric_df.interpolate(method="time").ffill().bfill()
    mu = numeric_df.rolling(window=15, min_periods=1).mean()
    sigma = numeric_df.rolling(window=15, min_periods=1).std().fillna(0)
    upper = mu + clamp_sigma * sigma
    lower = mu - clamp_sigma * sigma
    numeric_df = numeric_df.clip(lower=lower, upper=upper)
    tags = numeric_df.columns.tolist()
    return numeric_df, tags


def compute_dq_metrics(df: pd.DataFrame, tags: Iterable[str]) -> pd.DataFrame:
    """Compute data-quality metrics per tag."""
    rows = []
    for tag in tags:
        s = pd.to_numeric(df.get(tag), errors="coerce")
        total = len(s)
        if total == 0:
            rows.append({"Tag": tag, "flatline_pct": 0.0, "dropout_pct": 0.0, "nan_pct": 0.0, "presence_ratio": 0.0, "spikes_pct": 0.0})
            continue
        nan_mask = s.isna()
        presence_ratio = float((~nan_mask).sum()) / float(total) if total else 0.0
        nan_pct = float(nan_mask.sum()) / float(total) * 100.0 if total else 0.0
        clean = s.fillna(method="ffill").fillna(method="bfill")
        diffs = clean.diff().abs()
        flatline_pct = float((diffs <= 1e-9).sum()) / max(total - 1, 1) * 100.0
        dropout_pct = float((clean == 0).sum()) / float(total) * 100.0 if total else 0.0
        if clean.std(ddof=0) > 1e-9:
            z = (clean - clean.mean()) / clean.std(ddof=0)
            spikes_pct = float((z.abs() > 4).sum()) / float(total) * 100.0
        else:
            spikes_pct = 0.0
        rows.append(
            {
                "Tag": tag,
                "flatline_pct": round(flatline_pct, 3),
                "dropout_pct": round(dropout_pct, 3),
                "nan_pct": round(nan_pct, 3),
                "presence_ratio": round(presence_ratio, 3),
                "spikes_pct": round(spikes_pct, 3),
            }
        )
    return pd.DataFrame(rows)


def prepare_window(
    source: Path,
    *,
    equip: str,
    resample_rule: str,
    clamp_sigma: float = 6.0,
    t0: Optional[datetime] = None,
    t1: Optional[datetime] = None,
) -> DataWindow:
    """Convenience function to load, clean, and compute DQ in one call."""
    raw = load_raw_data(source, equip=equip, t0=t0, t1=t1)
    clean, tags = resample_and_clean(raw, resample_rule=resample_rule, clamp_sigma=clamp_sigma)
    dq = compute_dq_metrics(clean, tags)
    return DataWindow(raw=raw, clean=clean, dq=dq, tags=tags)

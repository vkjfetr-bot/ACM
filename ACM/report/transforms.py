from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict


def align_and_clip(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    # Align to overlapping time range and shared columns
    cols = sorted(set(train_df.columns) & set(test_df.columns))
    if not cols:
        return pd.DataFrame(index=test_df.index)
    tr = train_df[cols]
    te = test_df[cols]
    # concatenate for simple downstream refs
    return te.copy()


def compute_regime_spans(scored: pd.DataFrame) -> List[Dict]:
    spans: List[Dict] = []
    if "Regime" not in scored.columns:
        return spans
    r = scored["Regime"].fillna(method="ffill").fillna(method="bfill").astype(int)
    cur = None
    start = None
    for i, (ts, val) in enumerate(r.items()):
        if cur is None:
            cur = val
            start = ts
        elif val != cur:
            spans.append({"start": start, "end": ts, "regime_id": int(cur)})
            cur = val
            start = ts
    if cur is not None:
        spans.append({"start": start, "end": r.index[-1], "regime_id": int(cur)})
    return spans


def episodeize(anom_mask: pd.Series, min_gap: pd.Timedelta) -> List[Dict]:
    spans: List[Dict] = []
    if anom_mask is None or anom_mask.empty:
        return spans
    anom = anom_mask.astype(bool)
    if not isinstance(anom.index, pd.DatetimeIndex):
        # attach dummy time index at 1-minute
        anom.index = pd.date_range("2000-01-01", periods=len(anom), freq="T")
    in_ep = False
    start = None
    for ts, flag in anom.items():
        if flag and not in_ep:
            in_ep = True
            start = ts
        elif not flag and in_ep:
            spans.append({"start": start, "end": ts})
            in_ep = False
    if in_ep and start is not None:
        spans.append({"start": start, "end": anom.index[-1]})
    return spans


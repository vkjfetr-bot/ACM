from __future__ import annotations

import pandas as pd
import numpy as np


def compute(df: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["tag", "coverage", "dropout", "flatline", "spikes"])
    rows = []
    for t in df.columns:
        s = pd.to_numeric(df[t], errors="coerce").astype(float)
        n = len(s)
        if n == 0:
            continue
        coverage = 100.0 * (1.0 - s.isna().mean())
        dropout = 100.0 * s.isna().mean()
        flatline = 100.0 * (s.diff().abs() < 1e-12).sum() / max(1, n)
        d = s.diff().dropna().abs()
        iqr = (d.quantile(0.75) - d.quantile(0.25)) + 1e-9
        spikes = int((d > 5 * iqr).sum())
        rows.append({"tag": t, "coverage": coverage, "dropout": dropout, "flatline": flatline, "spikes": spikes})
    return pd.DataFrame(rows)


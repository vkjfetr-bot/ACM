"""Fusion — Weighted Geometric Mean of Hypothesis Scores.

Combines H1/H2/H3 scores into a fused score in [0,1], handling missing series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fuse_scores(scores: dict[str, pd.Series], weights: dict[str, float] | None = None) -> pd.Series:
    if not scores:
        return pd.Series(dtype=float)
    # Align indices and fill missing with neutral 1.0
    idx = None
    for s in scores.values():
        idx = s.index if idx is None else idx.union(s.index)
    aligned = {k: s.reindex(idx).fillna(1.0).clip(0, 1) for k, s in scores.items()}
    if weights is None:
        weights = {k: 1.0 for k in aligned}
    total_w = sum(weights.values()) + 1e-9
    w = np.array([weights[k] for k in aligned]) / total_w
    mat = np.vstack([aligned[k].values for k in aligned])
    # Geometric mean with weights: exp(sum(w_i * log(x_i)))
    with np.errstate(divide="ignore"):
        fused = np.exp(np.sum(w[:, None] * np.log(mat + 1e-12), axis=0))
    return pd.Series(fused, index=idx, name="fused")


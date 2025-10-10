from __future__ import annotations

import numpy as np
import pandas as pd


def fuse_scores(scores: dict[str, pd.Series], weights: dict[str, float] | None = None) -> pd.Series:
    if not scores:
        return pd.Series(dtype=float)
    # unify index
    idx = None
    for s in scores.values():
        idx = s.index if idx is None else idx.union(s.index)
    aligned = {k: s.reindex(idx).fillna(0.5).clip(0, 1) for k, s in scores.items()}
    if weights is None:
        weights = {k: 1.0 for k in aligned}
    w = np.array([weights[k] for k in aligned], dtype=float)
    w = w / (w.sum() + 1e-9)
    mat = np.vstack([aligned[k].values for k in aligned])
    fused = np.exp(np.sum(w[:, None] * np.log(mat + 1e-9), axis=0))
    return pd.Series(fused, index=idx, name="FusedScore")


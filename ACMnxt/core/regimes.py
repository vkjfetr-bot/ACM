"""Regimes — Auto-K KMeans with Silhouette Sweep

Find stable cluster regimes and assign labels; persist artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_assign_regimes(x: pd.DataFrame, k_range=range(2, 8), out_dir: str | Path | None = None, max_rows: Optional[int] = 20000) -> Tuple[pd.Series, KMeans]:
    num = x.select_dtypes(include=[np.number])
    if num.empty:
        return pd.Series(dtype=int, index=x.index, name="regime"), KMeans(n_clusters=1)
    arr = num.values
    # Subsample for silhouette to avoid O(N) fits on huge frames
    sample_idx = None
    if max_rows is not None and len(arr) > max_rows:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(arr), size=max_rows, replace=False)
        sample = arr[sample_idx]
    else:
        sample = arr
    best_k, best_score, best_model = None, -1.0, None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        # Fit on sample, score on sample
        labels_sample = km.fit_predict(sample)
        score = silhouette_score(sample, labels_sample) if len(np.unique(labels_sample)) > 1 else -1
        if score > best_score:
            best_k, best_score, best_model = k, score, km
    assert best_model is not None
    labels = pd.Series(best_model.predict(arr), index=x.index, name="regime")
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, out / "acm_regimes.joblib")
    return labels, best_model

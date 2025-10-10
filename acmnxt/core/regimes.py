from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_assign_regimes(x: pd.DataFrame, k_range=range(2, 8), out_dir: str | Path | None = None, max_rows: Optional[int] = 20000) -> Tuple[pd.Series, KMeans]:
    num = x.select_dtypes(include=[float, int])
    if num.empty:
        return pd.Series(dtype=int, index=x.index, name="Regime"), KMeans(n_clusters=1)
    arr = num.values
    # sampling for silhouette
    sample = arr
    if max_rows is not None and len(arr) > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(arr), size=max_rows, replace=False)
        sample = arr[idx]
    best = None; best_score = -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(sample)
        sc = silhouette_score(sample, labs) if len(np.unique(labs)) > 1 else -1
        if sc > best_score:
            best = km; best_score = sc
    labels = pd.Series(best.predict(arr), index=x.index, name="Regime")
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, out / "acm_regimes.joblib")
    return labels, best


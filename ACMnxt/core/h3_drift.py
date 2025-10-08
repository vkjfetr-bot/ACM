"""H3 — Embedding Drift Distance

Regime-shift detection by measuring drift from reference center in PCA space.
Assumes PCA artifacts from H2.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def score_drift(df: pd.DataFrame, art_dir: str | Path) -> pd.Series:
    """Compute drift distance in PCA space relative to training center.

    Loads scaler+pca from art_dir and computes L2 distance to training mean
    in PCA space. Scores are normalized to [0,1] via logistic on robust z.
    """
    scaler = joblib.load(Path(art_dir) / "acm_scaler.joblib")
    pca = joblib.load(Path(art_dir) / "acm_pca.joblib")
    x = df.select_dtypes(include=[float, int]).values
    xs = scaler.transform(x)
    zp = pca.transform(xs)
    # distance from origin approximates drift if training mean ~0 after scaling
    d = np.linalg.norm(zp, axis=1)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-9
    z = (d - med) / (1.4826 * mad)
    s = 1 / (1 + np.exp(-z))
    return pd.Series(s, index=df.index, name="H3_Drift")


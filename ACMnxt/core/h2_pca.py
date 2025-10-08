"""H2 — PCA Reconstruction Error

Multivariate anomaly via PCA reconstruction error normalized to [0,1].
Provides fit_pca() and score_h2() using cached scaler/PCA artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAArtifacts:
    scaler: StandardScaler
    pca: PCA


def fit_pca(df: pd.DataFrame, n_components: int = 5, out_dir: str | Path | None = None) -> PCAArtifacts:
    x = df.select_dtypes(include=[np.number]).astype(np.float32).values
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(xs)
    arts = PCAArtifacts(scaler=scaler, pca=pca)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, out / "acm_scaler.joblib")
        joblib.dump(pca, out / "acm_pca.joblib")
    return arts


def score_h2(df: pd.DataFrame, artifacts: PCAArtifacts | None = None, art_dir: str | Path | None = None) -> pd.Series:
    if artifacts is None:
        if art_dir is None:
            raise ValueError("Provide artifacts or art_dir to load them")
        scaler = joblib.load(Path(art_dir) / "acm_scaler.joblib")
        pca = joblib.load(Path(art_dir) / "acm_pca.joblib")
        artifacts = PCAArtifacts(scaler=scaler, pca=pca)
    x = df.select_dtypes(include=[np.number]).values
    xs = artifacts.scaler.transform(x.astype(np.float32))
    xh = artifacts.pca.inverse_transform(artifacts.pca.transform(xs))
    err = np.mean((xs - xh) ** 2, axis=1)
    # Normalize to [0,1] robustly via MAD
    med = np.median(err)
    mad = np.median(np.abs(err - med)) + 1e-9
    z = (err - med) / (1.4826 * mad)
    score = 1 / (1 + np.exp(-z))
    return pd.Series(score, index=df.index, name="H2_PCA")

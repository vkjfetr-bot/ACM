"""Regime-aware anomaly detectors and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from scipy.stats import genpareto  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genpareto = None


@dataclass
class DetectorConfig:
    per_regime: str
    fusion_weights: Tuple[float, float, float]


def train_regime_detectors(
    X: np.ndarray,
    index: pd.Index,
    regimes: np.ndarray,
    *,
    config: DetectorConfig,
) -> Dict[int, IsolationForest]:
    models: Dict[int, IsolationForest] = {}
    for regime in np.unique(regimes):
        mask = regimes == regime
        if mask.sum() < 20:
            continue
        subset = X[mask]
        model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=42,
        )
        model.fit(subset)
        models[int(regime)] = model
    return models


def score_regime_detectors(
    X: np.ndarray,
    index: pd.Index,
    regimes: np.ndarray,
    models: Dict[int, IsolationForest],
) -> pd.DataFrame:
    pt_scores = np.ones(len(index))
    for regime, model in models.items():
        mask = regimes == regime
        if not mask.any():
            continue
        subset = X[mask]
        preds = -model.score_samples(subset)
        pt_scores[mask] = _normalise(preds)
    df = pd.DataFrame({"pt_score": pt_scores}, index=index)
    return df


def sequence_score(latent: np.ndarray, index: pd.Index, window: int = 30) -> pd.Series:
    if latent.shape[0] < window * 2:
        return pd.Series(np.zeros(latent.shape[0]), index=index)
    errs = np.zeros(latent.shape[0])
    for idx in range(window, latent.shape[0]):
        history = latent[idx - window : idx]
        target = latent[idx]
        mu = history.mean(axis=0)
        errs[idx] = np.linalg.norm(target - mu)
    return pd.Series(_normalise(errs), index=index)


def window_score(scores: pd.Series, window: int = 30) -> pd.Series:
    win = scores.rolling(window=window, min_periods=1).mean()
    return _normalise_series(win)


def calibrate_thresholds(
    fused: pd.Series,
    regimes: np.ndarray,
    *,
    alpha: float,
    fallback_quantile: float,
) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    for regime in np.unique(regimes):
        mask = regimes == regime
        data = fused[mask]
        if len(data) < 20:
            thresholds[int(regime)] = float(data.quantile(fallback_quantile)) if len(data) else 0.95
            continue
        tail = data.nlargest(max(int(len(data) * 0.05), 10))
        if genpareto is not None and len(tail) > 10:
            threshold = _gpd_threshold(tail.values, alpha) or data.quantile(fallback_quantile)
        else:
            threshold = data.quantile(fallback_quantile)
        thresholds[int(regime)] = float(threshold)
    return thresholds


def _gpd_threshold(samples: np.ndarray, alpha: float) -> float:
    try:
        params = genpareto.fit(samples - samples.min())
        c, loc, scale = params
        q = genpareto.ppf(1 - alpha, c, loc=loc, scale=scale)
        return float(samples.min() + q)
    except Exception:
        return float(np.quantile(samples, 0.99))


def _normalise(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=np.nanmax(data[np.isfinite(data)]) if np.isfinite(data).any() else 1.0)
    mn = np.nanmin(data)
    mx = np.nanmax(data)
    if mx - mn < 1e-9:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)


def _normalise_series(series: pd.Series) -> pd.Series:
    arr = _normalise(series.values)
    return pd.Series(arr, index=series.index)

"""Drift detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DriftThresholds:
    psi_warn: float
    psi_alert: float
    adwin_delta: float


def regime_population_stability(current: Sequence[float], reference: Sequence[float]) -> float:
    current = np.asarray(current, dtype=float)
    reference = np.asarray(reference, dtype=float)
    current = current / current.sum() if current.sum() else current
    reference = reference / reference.sum() if reference.sum() else reference
    psi = np.nansum((current - reference) * np.log((current + 1e-9) / (reference + 1e-9)))
    return float(psi)


def detect_regime_drift(
    regimes: np.ndarray,
    reference_hist: Optional[Dict[int, float]],
    thresholds: DriftThresholds,
) -> Dict[str, object]:
    counts = pd.Series(regimes).value_counts(normalize=True)
    if not counts.empty and reference_hist:
        aligned_ref = np.array([reference_hist.get(int(k), 1e-6) for k in counts.index])
    else:
        aligned_ref = np.ones_like(counts.values) / len(counts)
    psi = regime_population_stability(counts.values, aligned_ref)
    level = "ok"
    if psi >= thresholds.psi_alert:
        level = "alert"
    elif psi >= thresholds.psi_warn:
        level = "warn"
    return {"metric": "psi_regime", "value": psi, "threshold": thresholds.psi_alert, "level": level}


def simple_adwin(series: pd.Series, delta: float) -> Dict[str, object]:
    """Approximate ADWIN using rolling mean differences."""
    window = max(int(len(series) * 0.2), 30)
    if len(series) < window * 2:
        return {"metric": "adwin", "value": 0.0, "threshold": delta, "level": "ok"}
    pre = series.iloc[:-window].mean()
    post = series.iloc[-window:].mean()
    diff = abs(pre - post)
    level = "alert" if diff >= delta else "ok"
    return {"metric": "adwin", "value": float(diff), "threshold": delta, "level": level}

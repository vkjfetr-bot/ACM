# core/drift.py
"""
Change-point and drift detection module.

Implements online detectors to identify subtle but persistent shifts in a time series,
typically the fused anomaly score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict

from . import fuse


class CUSUMDetector:
    """
    Online change-point detection using the CUSUM algorithm.
    Detects small, sustained drifts from a baseline mean.
    """
    def __init__(self, threshold: float = 2.0, drift: float = 0.1):
        self.threshold = threshold
        self.drift = drift
        self.mean = 0.0
        self.std = 1.0
        self.sum_pos = 0.0
        self.sum_neg = 0.0

    def fit(self, x: np.ndarray) -> "CUSUMDetector":
        # v11.1.2: Use robust statistics (median/MAD) for CUSUM baseline
        # This allows CUSUM to work correctly even when training data contains faults
        self.mean = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - self.mean)))
        self.std = mad * 1.4826  # Scale MAD to be consistent with std for normal distribution
        
        # DRIFT-AUDIT-01: Guard against non-finite mean (e.g., all-NaN input)
        if not np.isfinite(self.mean):
            self.mean = 0.0
        if not np.isfinite(self.std) or self.std < 1e-9:
            self.std = 1.0
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros_like(x, dtype=np.float32)
        x_norm = (x - self.mean) / self.std
        # DRIFT-AUDIT-02: Handle NaN/inf in normalized values to prevent accumulation errors
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        for i, val in enumerate(x_norm):
            self.sum_pos = max(0.0, self.sum_pos + val - self.drift)
            self.sum_neg = max(0.0, self.sum_neg - val - self.drift)
            scores[i] = max(self.sum_pos, self.sum_neg)
        return scores


# ============================================================================
# DRIFT-01: Multi-Feature Drift Detection Helpers (moved from acm_main.py)
# ============================================================================

def compute_drift_trend(drift_series: np.ndarray, window: int = 20) -> float:
    """
    Compute drift trend as the slope of linear regression over the last `window` points.
    
    Positive slope indicates upward drift (degradation), negative indicates recovery.
    Returns normalized slope (slope per sample).
    
    Args:
        drift_series: Array of drift/CUSUM z-scores
        window: Number of recent points to use for trend calculation
    
    Returns:
        Slope of linear regression fit. Positive = worsening, negative = improving.
    """
    if len(drift_series) < 2:
        return 0.0
    
    # Use last `window` points
    recent = drift_series[-window:] if len(drift_series) >= window else drift_series
    if len(recent) < 2:
        return 0.0
    
    # Remove NaNs
    valid_mask = ~np.isnan(recent)
    if valid_mask.sum() < 2:
        return 0.0
    
    x = np.arange(len(recent))[valid_mask]
    y = recent[valid_mask]
    
    # Linear regression: y = slope * x + intercept
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except Exception:
        return 0.0


def compute_regime_volatility(regime_labels: np.ndarray, window: int = 20) -> float:
    """
    Compute regime volatility as the fraction of regime transitions in the last `window` points.
    
    High volatility suggests unstable operating conditions or noisy regime assignments.
    
    Args:
        regime_labels: Array of regime label assignments (integers)
        window: Number of recent points to analyze
    
    Returns:
        Value in [0, 1] where 0 = completely stable, 1 = regime changes every step.
    """
    if len(regime_labels) < 2:
        return 0.0
    
    # Use last `window` points
    recent = regime_labels[-window:] if len(regime_labels) >= window else regime_labels
    if len(recent) < 2:
        return 0.0
    
    # Count transitions (label changes)
    transitions = np.sum(recent[1:] != recent[:-1])
    return float(transitions) / (len(recent) - 1)


def compute(score_df: pd.DataFrame, score_out: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes drift and change-point scores on the fused anomaly score.
    """
    frame = score_out["frame"]
    if "fused" not in frame.columns:
        return score_out

    fused_score = frame["fused"].to_numpy(copy=False)

    # CUSUM detector for online change-point detection
    drift_cfg = cfg.get("drift", {}) or {}
    cusum_cfg = (drift_cfg.get("cusum", {}) or {})
    
    # BUGFIX v11.1.5: Split calibration and scoring to avoid data leakage
    # Fit CUSUM on first 50% of data, score on all data
    # This prevents the detector from "seeing" future data during calibration
    n = len(fused_score)
    calibration_end = max(10, n // 2)  # At least 10 points for calibration
    calibration_window = fused_score[:calibration_end]
    
    detector = CUSUMDetector(
        threshold=float(cusum_cfg.get("threshold", 2.0)),
        drift=float(cusum_cfg.get("drift", 0.1)),
    ).fit(calibration_window)  # Calibrate on first half only

    frame["cusum_raw"] = detector.score(fused_score)

    # Apply exponential smoothing to reduce stepped appearance
    smoothing_alpha = float(cusum_cfg.get("smoothing_alpha", 0.3))
    cusum_smooth = pd.Series(frame["cusum_raw"]).ewm(alpha=smoothing_alpha, adjust=False).mean().to_numpy()
    
    # Calibrate the smoothed CUSUM score to a z-score for fusion/reporting
    cal_cusum = fuse.ScoreCalibrator(q=0.98).fit(cusum_smooth)
    frame["cusum_z"] = cal_cusum.transform(cusum_smooth)

    score_out["frame"] = frame
    return score_out


# ============================================================================
# DRIFT-02: Alert Mode Classification (moved from acm_main.py v11.2)
# ============================================================================

def compute_drift_alert_mode(
    frame: pd.DataFrame,
    cfg: Dict[str, Any],
    regime_quality_ok: bool = False,
    equip: str = "",
) -> pd.DataFrame:
    """Compute drift alert mode using multi-feature detection or simple threshold.
    
    This helper determines whether the system is in DRIFT mode (gradual degradation
    requiring retraining) or FAULT mode (transient anomaly) using:
    - Multi-feature detection: drift trend, fused level, regime volatility with hysteresis
    - Simple threshold: P95 drift exceeds configured threshold
    
    Args:
        frame: Scored frame DataFrame with drift_z, cusum_z, fused columns
        cfg: Config dictionary with drift settings
        regime_quality_ok: Whether regime clustering is of sufficient quality
        equip: Equipment name for logging
        
    Returns:
        Frame with alert_mode column added ('DRIFT' or 'FAULT')
    """
    from .observability import Console
    
    # Find the drift column
    drift_col = "cusum_z" if "cusum_z" in frame.columns else ("drift_z" if "drift_z" in frame.columns else None)
    
    # Retrieve multi-feature drift configuration
    drift_cfg = (cfg or {}).get("drift", {})
    multi_feat_cfg = drift_cfg.get("multi_feature", {})
    multi_feat_enabled = bool(multi_feat_cfg.get("enabled", False))
    
    if drift_col is None:
        frame["alert_mode"] = "FAULT"
        return frame
    
    try:
        drift_array = frame[drift_col].to_numpy(dtype=np.float32)
        
        if multi_feat_enabled:
            # DRIFT-01: Multi-feature logic with hysteresis
            trend_window = int(multi_feat_cfg.get("trend_window", 20))
            trend_threshold = float(multi_feat_cfg.get("trend_threshold", 0.05))
            fused_drift_min = float(multi_feat_cfg.get("fused_drift_min", 2.0))
            fused_drift_max = float(multi_feat_cfg.get("fused_drift_max", 5.0))
            regime_volatility_max = float(multi_feat_cfg.get("regime_volatility_max", 0.3))
            hysteresis_on = float(multi_feat_cfg.get("hysteresis_on", 3.0))
            hysteresis_off = float(multi_feat_cfg.get("hysteresis_off", 1.5))
            
            # Compute features (using local helpers)
            drift_trend = compute_drift_trend(drift_array, window=trend_window)
            fused_p95 = float(np.nanpercentile(frame["fused"].to_numpy(dtype=np.float32), 95)) if "fused" in frame.columns else 0.0
            
            # Compute regime volatility if regime labels exist
            regime_volatility = 0.0
            if "regime_label" in frame.columns and regime_quality_ok:
                regime_labels = frame["regime_label"].to_numpy()
                regime_volatility = compute_regime_volatility(regime_labels, window=trend_window)
            
            drift_p95 = float(np.nanpercentile(drift_array, 95))
            
            # Previous alert mode (for hysteresis) - assume "FAULT" if unavailable
            prev_alert_mode = "FAULT"
            
            # Composite rule for drift detection
            is_drift_condition = (
                abs(drift_trend) > trend_threshold and
                fused_drift_min <= fused_p95 <= fused_drift_max and
                regime_volatility < regime_volatility_max
            )
            
            # Hysteresis logic
            if prev_alert_mode == "DRIFT":
                alert_mode = "DRIFT" if drift_p95 > hysteresis_off else "FAULT"
            else:
                alert_mode = "DRIFT" if (drift_p95 > hysteresis_on and is_drift_condition) else "FAULT"
            
            frame["alert_mode"] = alert_mode
            Console.info(f"Drift: {drift_col} P95={drift_p95:.3f} | trend={drift_trend:.4f} | fused={fused_p95:.3f} | mode={alert_mode}", component="DRIFT")
        else:
            # Fallback to legacy simple threshold
            drift_p95 = float(np.nanpercentile(drift_array, 95))
            drift_threshold = float(drift_cfg.get("p95_threshold", 2.0))
            frame["alert_mode"] = "DRIFT" if drift_p95 > drift_threshold else "FAULT"
            Console.info(f"Drift: {drift_col} P95={drift_p95:.3f} | threshold={drift_threshold:.1f} | mode={frame['alert_mode'].iloc[-1]}", component="DRIFT")
    except Exception as e:
        Console.warn(f"Detection failed: {e}", component="DRIFT",
                     equip=equip, error_type=type(e).__name__, error=str(e)[:200])
        frame["alert_mode"] = "FAULT"
    
    return frame

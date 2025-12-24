"""
AR(1) Baseline Detector
=======================

Per-sensor AR(1) baseline model for residual scoring.
Extracted from core/forecasting.py for modularity.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Literal, Optional
import numpy as np
import pandas as pd
from core.observability import Console, Span

# Minimum samples required for AR(1) model coefficient estimation
MIN_AR1_SAMPLES = 3
# Minimum samples for stable forecast generation (used for warnings)
MIN_FORECAST_SAMPLES = 20

class AR1Detector:
    """
    Per-sensor AR(1) baseline model for residual scoring.
    
    Calculates AR(1) coefficients (phi) and mean (mu) for each sensor.
    Scores new data by calculating the absolute z-score of the residuals, 
    normalized by the TRAIN-time residual standard deviation.
    
    Usage:
        detector = AR1Detector(ar1_cfg={})
        detector.fit(train_df)
        scores = detector.score(test_df)
    """
    
    def __init__(self, ar1_cfg: Dict[str, Any] | None = None):
        """
        Initialize the AR(1) detector.
        
        Args:
            ar1_cfg: Configuration dict with optional keys:
                - eps (float): Numeric stability epsilon (default: 1e-9)
                - phi_cap (float): Max absolute phi value (default: 0.999)
                - sd_floor (float): Min std dev (default: 1e-6)
                - fuse (str): Fusion strategy "mean"|"median"|"p95" (default: "mean")
        """
        self.cfg = ar1_cfg or {}
        self._eps: float = float(self.cfg.get("eps", 1e-9))
        self._phi_cap: float = float(self.cfg.get("phi_cap", 0.999))
        self._sd_floor: float = float(self.cfg.get("sd_floor", 1e-6))
        self._fuse: Literal["mean", "median", "p95"] = self.cfg.get("fuse", "mean")
        
        # Trained parameters per column: (phi, mu)
        self.phimap: Dict[str, Tuple[float, float]] = {}
        # TRAIN residual std per column for normalization
        self.sdmap: Dict[str, float] = {}
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> "AR1Detector":
        """
        Fit the AR(1) model for each column in the training data.
        
        Args:
            X: Training feature matrix
            
        Returns:
            self for chaining
        """
        with Span("fit.ar1", n_samples=len(X), n_features=X.shape[1] if len(X) > 0 else 0):
            self.phimap = {}
            self.sdmap = {}
        
        if not isinstance(X, pd.DataFrame) or X.shape[0] == 0:
            self._is_fitted = True
            return self
        
        # Collect warnings to batch-report at end (avoid 100s of individual SQL inserts)
        near_constant_cols = []
        clamped_cols = []
        insufficient_cols = []
        
        for c in X.columns:
            col = X[c].to_numpy(copy=False, dtype=np.float32)
            finite = np.isfinite(col)
            x = col[finite]
            
            if x.size < MIN_AR1_SAMPLES:
                mu = float(np.nanmean(col)) if x.size else 0.0
                if not np.isfinite(mu):
                    mu = 0.0
                phi = 0.0
                self.phimap[c] = (phi, mu)
                resid = (x - mu) if x.size else np.array([0.0], dtype=np.float32)
                sd = float(np.std(resid)) if resid.size else self._sd_floor
                self.sdmap[c] = max(sd, self._sd_floor)
                continue
            
            mu = float(np.nanmean(x))
            if not np.isfinite(mu):
                mu = 0.0
            xc = x - mu
            var_xc = float(np.var(xc)) if xc.size else 0.0
            phi = 0.0
            
            if np.isfinite(var_xc) and var_xc >= 1e-8:
                num = float(np.dot(xc[1:], xc[:-1]))
                den = float(np.dot(xc[:-1], xc[:-1]))
                if abs(den) >= 1e-9:
                    phi = num / den
            else:
                near_constant_cols.append(c)
            
            if abs(phi) > self._phi_cap:
                original_phi = phi
                phi = float(np.sign(phi) * self._phi_cap)
                clamped_cols.append((c, original_phi, phi))
            
            if len(x) < MIN_FORECAST_SAMPLES:
                insufficient_cols.append((c, len(x)))
            
            self.phimap[c] = (phi, mu)
            
            # Compute TRAIN residuals & std for normalization during score()
            x_shift = np.empty_like(x, dtype=np.float32)
            x_shift[0] = mu
            x_shift[1:] = x[:-1]
            pred = (x_shift - mu) * phi + mu
            resid = x - pred
            resid_for_sd = resid[1:] if resid.size > 1 else resid
            sd = float(np.std(resid_for_sd))
            self.sdmap[c] = max(sd, self._sd_floor)
        
        # Emit batched warnings (single SQL insert instead of 100s)
        if near_constant_cols:
            n = len(near_constant_cols)
            sample = near_constant_cols[:3]
            Console.warn(f"{n} near-constant columns (phi=0): {sample}{'...' if n > 3 else ''}", component="AR1")
        if clamped_cols:
            n = len(clamped_cols)
            Console.warn(f"{n} columns with phi clamped to +/-{self._phi_cap}", component="AR1")
        if insufficient_cols:
            n = len(insufficient_cols)
            Console.warn(f"{n} columns with <{MIN_FORECAST_SAMPLES} samples (unstable coefficients)", component="AR1")
        
        self._is_fitted = True
        return self
    
    def score(self, X: pd.DataFrame, return_per_sensor: bool = False) -> np.ndarray | Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate absolute z-scores of residuals using TRAIN-time residual std.
        
        Args:
            X: Scoring feature matrix
            return_per_sensor: If True, also return DataFrame of per-sensor |z|
            
        Returns:
            Fused absolute z-scores (len == len(X))
            Optionally: (fused_scores, per_sensor_df) when return_per_sensor=True
        """
        with Span("score.ar1", n_samples=len(X), n_features=X.shape[1] if len(X) > 0 else 0):
            if not self._is_fitted:
                return np.zeros(len(X), dtype=np.float32)
        
        per_cols: Dict[str, np.ndarray] = {}
        n = len(X)
        
        if n == 0 or X.shape[1] == 0:
            return (np.zeros(0, dtype=np.float32), pd.DataFrame(index=X.index)) if return_per_sensor else np.zeros(0, dtype=np.float32)
        
        # EARLY EXIT: If >80% of columns are near-constant, return zeros to prevent hang on pathological data
        near_constant_count = sum(1 for (phi, _) in self.phimap.values() if abs(phi) < 1e-6)
        total_count = len(self.phimap)
        if total_count > 0 and near_constant_count > total_count * 0.8:
            Console.warn(f"Early exit: {near_constant_count}/{total_count} columns near-constant (>{80}%) - returning zero scores to prevent hang", component="AR1")
            return (np.zeros(n, dtype=np.float32), pd.DataFrame(index=X.index)) if return_per_sensor else np.zeros(n, dtype=np.float32)
        
        for c in X.columns:
            series = X[c].to_numpy(copy=False, dtype=np.float32)
            
            # Data quality check: skip columns with >50% NaN
            nan_count = np.isnan(series).sum()
            nan_fraction = nan_count / len(series) if len(series) > 0 else 0.0
            
            if nan_fraction > 0.5:
                Console.warn(f"Column '{c}': {nan_fraction*100:.1f}% NaN (>{50}%) - skipping column", component="AR1")
                continue
            
            # Log high imputation rates for visibility
            if nan_fraction > 0.2:
                Console.warn(f"Column '{c}': {nan_fraction*100:.1f}% NaN - imputing to mu (high imputation rate)", component="AR1")
            
            # FOR-CODE-03: Renamed ph -> phi for clarity (autoregressive coefficient)
            phi, mu = self.phimap.get(c, (0.0, float(np.nanmean(series))))
            if not np.isfinite(mu):
                mu = 0.0
            
            sd_train = self.sdmap.get(c, self._sd_floor)
            if not np.isfinite(sd_train) or sd_train <= self._sd_floor:
                sd_train = self._sd_floor
            
            # Impute NaNs to mu for prediction path
            series_finite = series.copy()
            if np.isnan(series_finite).any():
                series_finite = np.where(np.isfinite(series_finite), series_finite, mu).astype(np.float32, copy=False)
            
            # One-step AR(1) prediction
            pred = np.empty_like(series_finite, dtype=np.float32)
            first_obs = series_finite[0] if series_finite.size else mu
            pred[0] = first_obs if np.isfinite(first_obs) else mu
            if n > 1:
                pred[1:] = (series_finite[:-1] - mu) * phi + mu
            
            resid = series - pred  # Keep NaNs where original series had NaNs
            z = np.abs(resid) / sd_train
            per_cols[c] = z.astype(np.float32, copy=False)
        
        if not per_cols:
            return (np.zeros(n, dtype=np.float32), pd.DataFrame(index=X.index)) if return_per_sensor else np.zeros(n, dtype=np.float32)
        
        col_names = list(per_cols.keys())
        matrix = np.column_stack([per_cols[name] for name in col_names]) if col_names else np.zeros((n, 0), dtype=np.float32)
        
        with np.errstate(all="ignore"):
            if self._fuse == "median":
                fused = np.nanmedian(matrix, axis=1).astype(np.float32)
            elif self._fuse == "p95":
                fused = np.nanpercentile(matrix, 95, axis=1).astype(np.float32)
            else:
                fused = np.nanmean(matrix, axis=1).astype(np.float32)
        
            if return_per_sensor:
                Z = pd.DataFrame({name: matrix[:, i] for i, name in enumerate(col_names)}, index=X.index)
                return fused, Z
            return fused
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize detector state for persistence."""
        return {"phimap": self.phimap, "sdmap": self.sdmap, "cfg": self.cfg}
    
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AR1Detector":
        """Deserialize detector state from dict."""
        inst = cls(payload.get("cfg"))
        inst.phimap = dict(payload.get("phimap", {}))
        inst.sdmap = dict(payload.get("sdmap", {}))

# core/correlation.py
"""
Multivariate correlation-break detection module.

- MahalanobisDetector: classic distance in feature space with ridge-regularized
  covariance (pinv for stability).

- PCASubspaceDetector: stable PCA monitoring that drops constant/non-finite
  columns, scales in float64, and returns finite SPE (Q) and Hotelling T².
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from utils.logger import Console
except ImportError as e:
    # If logger import fails, something is seriously wrong - fail fast
    raise SystemExit(f"FATAL: Cannot import utils.logger.Console: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# Robust StandardScaler with variance floor
# ──────────────────────────────────────────────────────────────────────────────
class RobustStandardScaler(StandardScaler):
    """StandardScaler with variance floor to prevent division-by-zero explosions.
    
    Prevents catastrophic Z-score spikes (±100σ) when batch variance collapses.
    Critical fix for batch continuity: per-batch standardization on narrow slices
    can create near-zero variance, exploding Z-scores at batch boundaries.
    
    Parameters
    ----------
    epsilon : float, default=1e-6
        Minimum allowed standard deviation. Values below this are clamped.
    """
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        if self.with_std and hasattr(self, 'scale_') and self.scale_ is not None:  # type: ignore[attr-defined]
            # Floor scale_ to prevent division by near-zero variance
            # This prevents Z-scores >100 when variance collapses
            self.scale_ = np.maximum(self.scale_, self.epsilon)
        return self
    
    def partial_fit(self, X, y=None, sample_weight=None):
        super().partial_fit(X, y, sample_weight)
        if self.with_std and hasattr(self, 'scale_') and self.scale_ is not None:  # type: ignore[attr-defined]
            self.scale_ = np.maximum(self.scale_, self.epsilon)
        return self

# ──────────────────────────────────────────────────────────────────────────────
# Mahalanobis
# ──────────────────────────────────────────────────────────────────────────────
class MahalanobisDetector:
    """Detects correlation breaks using squared Mahalanobis distance."""

    def __init__(self, regularization: float = 1e-3):
        self.l2: float = float(regularization)
        self.mu: Optional[np.ndarray] = None
        self.S_inv: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame) -> "MahalanobisDetector":
        Xn = X.to_numpy(dtype=np.float64, copy=False)
        
        # COR-01: Guard against insufficient samples for covariance estimation
        # Covariance matrix requires at least 2 samples to be well-defined
        if Xn.shape[0] < 2:
            Console.warn(
                f"[MHAL] Insufficient samples for covariance estimation (n={Xn.shape[0]}). "
                f"Falling back to identity covariance (Mahalanobis = Euclidean distance)."
            )
            # Fallback: use mean (or zeros if no data) and identity covariance
            self.mu = Xn.mean(axis=0) if Xn.shape[0] > 0 else np.zeros(Xn.shape[1], dtype=np.float64)
            self.S_inv = np.eye(Xn.shape[1], dtype=np.float64)
            self.cond_num = 1.0  # Identity matrix is perfectly conditioned
            return self
        
        # ANA-07: Audit NaN during TRAIN phase
        nan_count = int(np.sum(~np.isfinite(Xn)))
        total_elements = Xn.size
        nan_rate = nan_count / total_elements if total_elements > 0 else 0.0
        if nan_rate > 0.001:  # More than 0.1%
            Console.warn(f"[MHAL] TRAIN phase NaN audit: {nan_count}/{total_elements} ({nan_rate:.3%}) non-finite values")
        
        Xn = np.nan_to_num(Xn, copy=False)

        self.mu = Xn.mean(axis=0)

        S = np.cov(Xn, rowvar=False)
        if not np.all(np.isfinite(S)):
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S += self.l2 * np.eye(S.shape[0], dtype=np.float64)

        self.S_inv = np.linalg.pinv(S)
        # ANA-07: Store condition number for export to calibration_summary
        self.cond_num = np.linalg.cond(S)
        
        # ANA-07: Lower action thresholds (warn at 1e8, increase reg at 1e10)
        if self.cond_num > 1e10:
            # Auto-increase regularization more aggressively (100x instead of 10x)
            # This helps with very ill-conditioned covariance matrices
            old_reg = self.l2
            self.l2 = self.l2 * 100.0  # Increased from 10x to 100x
            Console.warn(f"[MHAL] CRITICAL condition number ({self.cond_num:.2e}) detected. Auto-increasing regularization: {old_reg:.2e} -> {self.l2:.2e}")
            # Re-compute with increased regularization
            S = np.cov(Xn, rowvar=False)
            if not np.all(np.isfinite(S)):
                S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            S += self.l2 * np.eye(S.shape[0], dtype=np.float64)
            self.S_inv = np.linalg.pinv(S)
            self.cond_num = np.linalg.cond(S)
            Console.info(f"[MHAL] After re-regularization: cond_num={self.cond_num:.2e}")
            # If still too high, increase again
            if self.cond_num > 1e10:
                old_reg = self.l2
                self.l2 = self.l2 * 10.0
                Console.warn(f"[MHAL] Still critical after 100x increase. Applying additional 10x: {old_reg:.2e} -> {self.l2:.2e}")
                S = np.cov(Xn, rowvar=False)
                if not np.all(np.isfinite(S)):
                    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
                S += self.l2 * np.eye(S.shape[0], dtype=np.float64)
                self.S_inv = np.linalg.pinv(S)
                self.cond_num = np.linalg.cond(S)
                Console.info(f"[MHAL] Final condition number after 1000x total increase: {self.cond_num:.2e}")
        elif self.cond_num > 1e8:
            Console.warn(f"[MHAL] High condition number ({self.cond_num:.2e}). Consider increasing regularization (current: {self.l2:.2e}).")
        else:
            Console.info(f"[MHAL] Good condition number ({self.cond_num:.2e}) with regularization {self.l2:.2e}.")
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.mu is None or self.S_inv is None:
            raise RuntimeError("MahalanobisDetector not fitted. Call .fit() first.")

        Xn = X.to_numpy(dtype=np.float64, copy=False)
        
        # ANA-07: Audit NaN during SCORE phase
        nan_count = int(np.sum(~np.isfinite(Xn)))
        total_elements = Xn.size
        nan_rate = nan_count / total_elements if total_elements > 0 else 0.0
        if nan_rate > 0.001:  # More than 0.1%
            Console.warn(f"[MHAL] SCORE phase NaN audit: {nan_count}/{total_elements} ({nan_rate:.3%}) non-finite values")
        
        Xn = np.nan_to_num(Xn, copy=False)

        d = Xn - self.mu
        m = np.einsum("ij,jk,ik->i", d, self.S_inv, d)
        m = np.maximum(m, 0.0)
        
        # ANA-07: Final NaN check on output scores
        result = m.astype(np.float32, copy=False)
        output_nan_count = int(np.sum(~np.isfinite(result)))
        if output_nan_count > 0:
            Console.warn(f"[MHAL] Output NaN audit: {output_nan_count}/{len(result)} ({output_nan_count/len(result):.3%}) non-finite scores detected")
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return result


# ──────────────────────────────────────────────────────────────────────────────
# PCA Subspace
# ──────────────────────────────────────────────────────────────────────────────
class PCASubspaceDetector:
    """
    PCA subspace monitoring returning:
      - SPE (Q): squared prediction error (distance to principal subspace)
      - T²     : Hotelling's T-squared (distance within the subspace)
    """

    def __init__(self, pca_cfg: Dict[str, Any] | None = None):
        self.cfg: Dict[str, Any] = (pca_cfg or {})
        # Use RobustStandardScaler to prevent variance collapse
        self.scaler = RobustStandardScaler(epsilon=1e-6, with_mean=True, with_std=True)
        self.pca: Optional[PCA] | Any = None  # Allow IncrementalPCA too

        self.keep_cols: List[str] = []
        self.col_medians: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame) -> "PCASubspaceDetector":
        """Fit scaler + PCA on TRAIN safely."""
        df = X.copy()

        # Drop fully non-finite columns, then near-constants
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        std = df.astype("float64").std(axis=0)
        df = df.loc[:, std > 1e-6]

        # If everything got dropped, inject a safe dummy feature
        if df.shape[1] == 0:
            df = pd.DataFrame({"_dummy": np.zeros(len(df), dtype=np.float64)}, index=df.index)

        # Impute + clip (guard wild magnitudes)
        self.col_medians = df.median()
        df = df.fillna(self.col_medians).clip(lower=-1e6, upper=1e6)

        self.keep_cols = list(df.columns)
        
        # COR-02: Guard against insufficient samples for PCA after feature filtering
        # PCA requires at least 2 samples to compute meaningful components
        if df.shape[0] < 2:
            Console.warn(
                f"[PCA] Insufficient samples after feature filtering (n={df.shape[0]}). "
                f"Skipping PCA fit - score() will return zeros."
            )
            # Set PCA to None to signal fallback mode
            self.pca = None
            self.keep_cols = []
            return self

        # Scale
        Xs = self.scaler.fit_transform(df.values.astype(np.float64, copy=False))

        # Choose a safe n_components
        n_samples, n_features = Xs.shape
        user_nc = self.cfg.get("n_components", 5)
        try:
            user_nc = int(user_nc)
        except Exception:
            user_nc = 5
        k = int(max(1, min(user_nc, n_features, max(1, n_samples - 1))))
        
        # Incremental path (CPU only)
        if self.cfg.get("incremental", False):
            from sklearn.decomposition import IncrementalPCA
            self.pca = IncrementalPCA(  # type: ignore[assignment]
                n_components=k,
                batch_size=max(256, int(self.cfg.get("batch_size", 4096))),
            )
            # partial_fit in batches
            bs = self.pca.batch_size or 256  # type: ignore[attr-defined]
            for i in range(0, n_samples, bs):
                self.pca.partial_fit(Xs[i:i+bs])  # type: ignore[attr-defined]
        else:
            # Default batch path
            self.pca = PCA(n_components=k, svd_solver="full", random_state=17)
            self.pca.fit(Xs)
            
        return self

    def score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (SPE, T²) as float32 arrays."""
        if self.pca is None or self.col_medians is None:
            n = len(X)
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

        # Align columns, impute with TRAIN medians
        df = X.copy().reindex(columns=self.keep_cols)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(self.col_medians)
        Xs = self.scaler.transform(df.values.astype(np.float64, copy=False))

        # Project → reconstruct → SPE
        Z = self.pca.transform(Xs)
        X_hat = self.pca.inverse_transform(Z)
        spe = np.sum((Xs - X_hat) ** 2, axis=1)
        # Clamp outrageous but finite values to keep float32 cast safe
        spe = np.clip(spe, 0.0, 1e9)

        # Hotelling T² (guard tiny eigenvalues)
        ev = getattr(self.pca, "explained_variance_", None)
        if ev is None:
            t2 = np.sum(Z ** 2, axis=1)
        else:
            ev = np.maximum(np.asarray(ev, dtype=np.float64), 1e-12)
            # Compute per-component then clamp to avoid huge-but-finite spikes
            t2_comp = (Z ** 2) / ev
            # Robust per-component clamp (protects rare degenerate modes)
            t2_comp = np.clip(t2_comp, 0.0, 1e9)
            t2 = np.sum(t2_comp, axis=1)

        # Final sanitization and clamp before float32 cast
        spe = np.nan_to_num(spe, nan=0.0, posinf=1e9, neginf=0.0)
        t2  = np.nan_to_num(t2,  nan=0.0, posinf=1e9, neginf=0.0)
        t2  = np.clip(t2, 0.0, 1e9)

        return spe.astype(np.float32, copy=False), t2.astype(np.float32, copy=False)

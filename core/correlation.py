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
    from core.observability import Console, Heartbeat, Span
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
# Mahalanobis (DEPRECATED - use PCA-T² instead)
# ──────────────────────────────────────────────────────────────────────────────
class MahalanobisDetector:
    """
    DEPRECATED: This detector is redundant with PCASubspaceDetector's T² statistic.
    
    Both compute the same Mahalanobis distance:
    - MHAL: D² = (x-μ)ᵀ Σ⁻¹ (x-μ) in raw feature space (numerically unstable)
    - PCA-T²: T² = Σᵢ zᵢ²/λᵢ in orthogonal PCA space (stable, guaranteed)
    
    PCA-T² is mathematically superior because:
    1. Covariance is diagonal in PCA space → no matrix inversion issues
    2. Dimensionality reduction → better numerical stability  
    3. Same statistical interpretation (both follow χ² distribution)
    
    DEPRECATED: This detector is no longer used in ACM v10.2.0+.
    Use pca_t2_z instead (same Mahalanobis distance, better stability).
    
    Parameters
    ----------
    regularization : float, default=1e-3
        Legacy parameter (ignored). Kept for backward compatibility.
    variance_retained : float, default=0.95
        Fraction of variance to retain when selecting PCA components.
    min_components : int, default=2
        Minimum number of PCA components to retain.
    """

    def __init__(self, regularization: float = 1e-3, variance_retained: float = 0.95, min_components: int = 2):
        self.l2: float = float(regularization)  # Legacy, not used
        self.variance_retained: float = float(variance_retained)
        self.min_components: int = int(min_components)
        
        # Pipeline components
        self.scaler: Optional[RobustStandardScaler] = None
        self.pca: Optional[PCA] = None
        self.n_components_: int = 0
        self.keep_cols: List[str] = []
        self.col_medians: Optional[pd.Series] = None
        
        # Diagnostics
        self.cond_num: float = 1.0  # Now always well-controlled
        self.eigenvalues_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame) -> "MahalanobisDetector":
        """
        Fit the PCA-based Mahalanobis detector.
        
        Process:
        1. Clean pathological features (constant, all-NaN)
        2. Standardize (zero mean, unit variance)
        3. PCA to orthogonal space (eliminates multicollinearity)
        4. Store eigenvalues for Mahalanobis computation
        """
        df = X.copy()
        n_samples_orig, n_features_orig = df.shape
        
        # Step 1: Clean features
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        std = df.astype("float64").std(axis=0)
        df = df.loc[:, std > 1e-8]  # Remove near-constant columns
        
        n_dropped = n_features_orig - df.shape[1]
        if n_dropped > 0:
            Console.debug(f"Dropped {n_dropped} constant/all-NaN columns", component="MHAL")
        
        # Fallback if everything got dropped
        if df.shape[1] == 0:
            df = pd.DataFrame({"_dummy": np.zeros(len(df), dtype=np.float64)}, index=df.index)
        
        # Impute remaining NaNs with median
        self.col_medians = df.median()
        df = df.fillna(self.col_medians)
        self.keep_cols = list(df.columns)
        
        n_samples, n_features = df.shape
        
        # Guard: insufficient samples
        if n_samples < 2:
            Console.warn(f"Insufficient samples (n={n_samples}). Falling back to identity scoring.", component="MHAL")
            self.pca = None
            self.cond_num = 1.0
            return self
        
        # Step 2: Standardize
        self.scaler = RobustStandardScaler(epsilon=1e-6, with_mean=True, with_std=True)
        X_scaled = self.scaler.fit_transform(df.values.astype(np.float64))
        
        # Step 3: Determine optimal number of components
        max_components = min(n_samples - 1, n_features)
        
        # Fit PCA with all possible components first to determine variance explained
        pca_full = PCA(n_components=max_components, random_state=42)
        pca_full.fit(X_scaled)
        
        # Select components to retain target variance
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.searchsorted(cumulative_variance, self.variance_retained) + 1
        n_components = max(self.min_components, n_components)
        n_components = min(n_components, max_components)
        self.n_components_ = int(n_components)
        
        # Refit with selected components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(X_scaled)
        
        # Store eigenvalues (variances of principal components)
        self.eigenvalues_ = self.pca.explained_variance_
        
        # Compute condition number in PCA space (now guaranteed to be reasonable)
        self.cond_num = float(self.eigenvalues_.max() / max(self.eigenvalues_.min(), 1e-12))
        var_explained = self.pca.explained_variance_ratio_.sum()
        
        Console.info(
            f"[MHAL] PCA-based fit: {n_components}/{n_features} components, "
            f"var={var_explained:.1%}, cond={self.cond_num:.2e}"
        )
        
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute Mahalanobis distance in PCA space.
        
        In PCA space, covariance is diagonal with eigenvalues on diagonal.
        Mahalanobis distance simplifies to:
            D² = Σᵢ (zᵢ)² / λᵢ
        
        where zᵢ are PCA scores and λᵢ are eigenvalues.
        This is equivalent to Hotelling's T² statistic.
        """
        if self.pca is None or self.scaler is None:
            # Fallback mode: return zeros
            return np.zeros(len(X), dtype=np.float32)
        
        # Align columns and impute with training medians
        df = X.copy().reindex(columns=self.keep_cols)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(self.col_medians)
        
        # Transform through pipeline
        X_scaled = self.scaler.transform(df.values.astype(np.float64))
        Z = self.pca.transform(X_scaled)
        
        # Compute Mahalanobis distance squared in PCA space
        # D² = Σᵢ (zᵢ)² / λᵢ (weighted sum of squared PCA scores)
        if self.eigenvalues_ is None:
            return np.zeros(len(X), dtype=np.float32)
        eigenvalues_safe = np.maximum(self.eigenvalues_, 1e-12)
        D_squared = np.sum((Z ** 2) / eigenvalues_safe, axis=1)
        
        # Ensure non-negative and finite
        # CAL-FIX-01: Reduced clip from 1e9 to 1e6 to prevent calibrator extreme thresholds
        D_squared = np.maximum(D_squared, 0.0)
        D_squared = np.clip(D_squared, 0.0, 1e6)
        
        result = D_squared.astype(np.float32)
        
        # Final NaN check
        nan_count = int(np.sum(~np.isfinite(result)))
        if nan_count > 0:
            Console.warn(f"Output NaN audit: {nan_count}/{len(result)} non-finite scores detected", component="MHAL")
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)
        
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
        with Span("fit.pca", n_samples=len(X), n_features=X.shape[1] if len(X) > 0 else 0):
            from core.observability import Console, Heartbeat
            Console.info(f"Fit start: train shape={X.shape}", component="PCA")
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
                    f"Insufficient samples after feature filtering (n={df.shape[0]}). "
                    f"Skipping PCA fit - score() will return zeros.", component="PCA"
                )
                # Set PCA to None to signal fallback mode
                self.pca = None
                self.keep_cols = []
                return self

            # Scale
            data_float = df.values.astype(np.float64, copy=False)
            Xs = self.scaler.fit_transform(data_float)

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
                    batch_end = min(i+bs, n_samples)
                    self.pca.partial_fit(Xs[i:batch_end])  # type: ignore[attr-defined]
            else:
                # Default batch path
                self.pca = PCA(n_components=k, svd_solver="full", random_state=17)
                self.pca.fit(Xs)
                
            Console.info(f"Fit complete in {Span.__name__}: {k} components, {n_samples} samples, {n_features} features", component="PCA" if "skip_loki" not in locals() else None, skip_loki=True)
            return self

    def score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (SPE, T²) as float32 arrays."""
        with Span("score.pca", n_samples=len(X), n_features=X.shape[1] if len(X) > 0 else 0):
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
            # CAL-FIX-01: Reduced clip from 1e9 to 1e6 to prevent calibrator extreme thresholds
            spe = np.clip(spe, 0.0, 1e6)

            # Hotelling T² (guard tiny eigenvalues)
            ev = getattr(self.pca, "explained_variance_", None)
            if ev is None:
                t2 = np.sum(Z ** 2, axis=1)
            else:
                ev = np.maximum(np.asarray(ev, dtype=np.float64), 1e-12)
                # Compute per-component then clamp to avoid huge-but-finite spikes
                t2_comp = (Z ** 2) / ev
                # CAL-FIX-01: Reduced clip from 1e9 to 1e6
                t2_comp = np.clip(t2_comp, 0.0, 1e6)
                t2 = np.sum(t2_comp, axis=1)

            # Final sanitization and clamp before float32 cast
            # CAL-FIX-01: Reduced clip from 1e9 to 1e6 to prevent calibrator extreme thresholds
            spe = np.nan_to_num(spe, nan=0.0, posinf=1e6, neginf=0.0)
            t2  = np.nan_to_num(t2,  nan=0.0, posinf=1e6, neginf=0.0)
            t2  = np.clip(t2, 0.0, 1e6)

            return spe.astype(np.float32, copy=False), t2.astype(np.float32, copy=False)

# core/omr.py
"""
Overall Model Residual (OMR) - Multivariate health indicator.

The OMR captures equipment health by modeling the normal relationships between
all sensors and detecting when actual behavior deviates from the learned baseline.
Unlike univariate detectors (AR1, PCA SPE), OMR captures multivariate correlations.

Key features:
- Fits a multivariate model (PLS, Linear, or PCA) on healthy baseline data
- Computes reconstruction error as health indicator
- Tracks per-sensor contributions to identify root causes
- Supports multiple model architectures with auto-selection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Literal
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class ModelType(str, Enum):
    """Supported model types for OMR."""
    PLS = "pls"
    LINEAR = "linear"
    PCA = "pca"
    AUTO = "auto"


@dataclass
class OMRModel:
    """Container for trained OMR model and metadata."""
    model: Any  # PLSRegression, Ridge, or PCA (None when using stored linear ensemble)
    scaler: StandardScaler
    model_type: str  # "pls", "linear", "pca"
    feature_names: List[str]
    train_residual_std: float  # For z-score normalization
    n_components: int  # Number of latent components (PLS/PCA)
    linear_models: Optional[List[Dict[str, Any]]] = None  # Stored ridge sub-models for linear mode
    train_samples: int = 0  # Track training sample count
    train_features: int = 0  # Track training feature count
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        import joblib
        from io import BytesIO
        
        # Serialize sklearn model to bytes
        model_bytes = BytesIO()
        joblib.dump(self.model, model_bytes)
        model_bytes.seek(0)
        
        # Serialize scaler to bytes
        scaler_bytes = BytesIO()
        joblib.dump(self.scaler, scaler_bytes)
        scaler_bytes.seek(0)
        
        payload = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "train_residual_std": self.train_residual_std,
            "n_components": self.n_components,
            "model_bytes": model_bytes.read(),
            "scaler_bytes": scaler_bytes.read(),
            "train_samples": self.train_samples,
            "train_features": self.train_features,
        }
        if self.linear_models is not None:
            payload["linear_models"] = [
                {
                    "indices": model_entry["indices"].tolist(),
                    "coef": model_entry["coef"].tolist(),
                    "intercept": model_entry["intercept"],
                }
                for model_entry in self.linear_models
            ]
        return payload


class OMRDetector:
    """
    Overall Model Residual detector using multivariate modeling.
    
    Strategy:
    1. Fit a multivariate model (PLS/Linear/PCA) on healthy training data
    2. For each timestep, predict sensor values from all other sensors
    3. Compute reconstruction error: ||x - x_reconstructed||
    4. Normalize by training residual std to get z-score
    5. Track per-sensor squared contributions to identify culprits
    
    Model Selection:
    - PLS: Best for high correlation, moderate sample size (default)
    - Linear: Fast, works well with sufficient samples
    - PCA: Best for dimensionality reduction, captures variance
    """
    
    # Constants
    MIN_RESIDUAL_STD = 1e-6  # Prevent division by zero
    MAX_Z_SCORE = 10.0  # Clip extreme z-scores
    DEFAULT_N_COMPONENTS = 5
    DEFAULT_ALPHA = 1.0
    DEFAULT_MIN_SAMPLES = 100
    
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize OMR detector.
        
        Args:
            cfg: Configuration dict with omr section
        """
        self.cfg = cfg or {}
        omr_cfg = self.cfg.get("omr", {})
        
        # Model selection
        self.model_type = omr_cfg.get("model_type", "auto")
        self.n_components = int(omr_cfg.get("n_components", self.DEFAULT_N_COMPONENTS))
        self.alpha = float(omr_cfg.get("alpha", self.DEFAULT_ALPHA))
        
        # Minimum samples for training
        self.min_samples = int(omr_cfg.get("min_samples", self.DEFAULT_MIN_SAMPLES))
        
        # Z-score clipping (configurable)
        self.max_z_score = float(omr_cfg.get("max_z_score", self.MAX_Z_SCORE))
        
        self._is_fitted = False
        self.model: Optional[OMRModel] = None
    
    def _select_model_type(self, n_samples: int, n_features: int) -> str:
        """
        Auto-select model type based on data characteristics.
        
        Args:
            n_samples: Number of training samples
            n_features: Number of features
            
        Returns:
            Model type: "pls", "linear", or "pca"
        """
        if self.model_type != "auto":
            return self.model_type
        
        # Decision tree for model selection
        if n_features > n_samples:
            # More features than samples - use PCA for dimensionality reduction
            return ModelType.PCA.value
        elif n_samples > 1000 and n_features < 20:
            # Large samples, moderate features - linear is fast
            return ModelType.LINEAR.value
        else:
            # Default: PLS works well for correlated sensor data
            return ModelType.PLS.value
    
    def _validate_input(self, X: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate input data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            (is_valid, error_message)
        """
        if X.empty:
            return False, "Empty input DataFrame"
        
        if X.shape[1] == 0:
            return False, "No features in input DataFrame"
        
        # Check for all-NaN columns
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            return False, f"All-NaN columns detected: {all_nan_cols}"
        
        return True, None
    
    def _prepare_data(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data for modeling (handle missing values).
        
        Args:
            X: Input DataFrame
            
        Returns:
            (cleaned_array, feature_names)
        """
        # Handle missing values with median imputation
        X_clean = X.fillna(X.median())
        
        # For remaining NaNs (e.g., all-NaN columns), fill with 0
        X_clean = X_clean.fillna(0)
        
        return X_clean.values, list(X.columns)
    
    def _compute_optimal_components(
        self, 
        n_samples: int, 
        n_features: int,
        model_type: str
    ) -> int:
        """
        Compute optimal number of components based on data dimensions.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            model_type: Selected model type
            
        Returns:
            Optimal number of components
        """
        if model_type not in {ModelType.PLS.value, ModelType.PCA.value}:
            return 0  # Not applicable for linear models
        
        # Start with configured components
        max_components = self.n_components
        
        # Constrain by data dimensions
        max_components = min(max_components, n_features, n_samples - 1)
        
        # For PLS/PCA, need at least 2 features
        if n_features > 1:
            max_components = min(max_components, n_features - 1)
        
        # Ensure at least 1 component
        return max(1, max_components)
    
    def _fit_pls_model(self, X_scaled: np.ndarray, n_components: int) -> Tuple[Any, np.ndarray]:
        """Fit PLS model and return model + reconstructions."""
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(X_scaled, X_scaled)
        X_recon = model.predict(X_scaled)
        return model, X_recon
    
    def _fit_linear_model(self, X_scaled: np.ndarray, n_features: int) -> Tuple[None, np.ndarray, List[Dict[str, Any]]]:
        """Fit linear ensemble model and return reconstructions + metadata."""
        reconstructions = []
        linear_models = []
        col_indices = np.arange(n_features)
        
        for target_idx in range(n_features):
            other_idx = np.delete(col_indices, target_idx)
            X_others = X_scaled[:, other_idx]
            y_target = X_scaled[:, target_idx]
            
            ridge = Ridge(alpha=self.alpha)
            ridge.fit(X_others, y_target)
            y_pred = ridge.predict(X_others)
            
            reconstructions.append(y_pred)
            linear_models.append({
                "indices": other_idx.astype(np.int32),
                "coef": ridge.coef_.astype(np.float32),
                "intercept": float(ridge.intercept_),
            })
        
        X_recon = np.column_stack(reconstructions)
        return None, X_recon, linear_models
    
    def _fit_pca_model(self, X_scaled: np.ndarray, n_components: int) -> Tuple[Any, np.ndarray]:
        """Fit PCA model and return model + reconstructions."""
        model = PCA(n_components=n_components, random_state=42)
        X_latent = model.fit_transform(X_scaled)
        X_recon = model.inverse_transform(X_latent)
        return model, X_recon
    
    def fit(self, X: pd.DataFrame, regime_labels: Optional[np.ndarray] = None) -> "OMRDetector":
        """
        Fit OMR model on healthy training data.
        
        Args:
            X: Training data (n_samples, n_features)
            regime_labels: Optional regime labels to filter healthy data
            
        Returns:
            self (fitted)
        """
        from utils.logger import Console
        
        # Validate input
        is_valid, error_msg = self._validate_input(X)
        if not is_valid:
            Console.info(f"[OMR] Skipping fit: {error_msg}")
            return self
        
        # Filter to healthy regime if labels provided
        if regime_labels is not None and len(regime_labels) == len(X):
            healthy_regime = int(np.min(regime_labels))
            healthy_mask = regime_labels == healthy_regime
            n_healthy = int(np.sum(healthy_mask))
            
            if n_healthy >= self.min_samples:
                X = X.iloc[healthy_mask]
                Console.info(f"[OMR] Filtered to healthy regime {healthy_regime}: {n_healthy} samples")
        
        # Prepare data
        X_clean, feature_names = self._prepare_data(X)
        n_samples, n_features = X_clean.shape
        
        # Check minimum samples
        min_required = max(20, min(self.min_samples // 2, n_features))
        if n_samples < min_required:
            Console.warn(f"[OMR] Insufficient samples ({n_samples}/{min_required}), skipping fit")
            return self
        
        if n_samples < self.min_samples:
            Console.info(f"[OMR] Proceeding with reduced sample count ({n_samples} < {self.min_samples})")
        
        # Auto-select model type
        selected_model = self._select_model_type(n_samples, n_features)
        Console.info(f"[OMR] Selected model type: {selected_model.upper()}")
        
        # Fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # Compute optimal components
        n_components = self._compute_optimal_components(n_samples, n_features, selected_model)
        
        linear_models: Optional[List[Dict[str, Any]]] = None
        model = None
        
        try:
            if selected_model == ModelType.PLS.value:
                model, X_recon = self._fit_pls_model(X_scaled, n_components)
                
            elif selected_model == ModelType.LINEAR.value:
                model, X_recon, linear_models = self._fit_linear_model(X_scaled, n_features)
                
            elif selected_model == ModelType.PCA.value:
                model, X_recon = self._fit_pca_model(X_scaled, n_components)
                
            else:
                raise ValueError(f"Unknown model type: {selected_model}")
            
            # Compute residuals
            residuals = X_scaled - X_recon
            residual_norm = np.linalg.norm(residuals, axis=1)
            train_residual_std = float(np.std(residual_norm))
            
            # Enforce lower bound to prevent division by zero
            train_residual_std = max(train_residual_std, self.MIN_RESIDUAL_STD)
            
            self.model = OMRModel(
                model=model,
                scaler=scaler,
                model_type=selected_model,
                feature_names=feature_names,
                train_residual_std=train_residual_std,
                n_components=n_components,
                linear_models=linear_models if selected_model == ModelType.LINEAR.value else None,
                train_samples=n_samples,
                train_features=n_features,
            )
            self._is_fitted = True
            
            Console.info(
                f"[OMR] Fitted {selected_model.upper()} model: "
                f"{n_samples} samples, {n_features} features, "
                f"{n_components} components, std={train_residual_std:.3f}"
            )
            
        except Exception as e:
            Console.error(f"[OMR] Model fitting failed: {e}")
            import traceback
            Console.error(traceback.format_exc())
            return self
        
        return self
    
    def _reconstruct_data(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reconstruct data using fitted model.
        
        Args:
            X_scaled: Scaled input data
            
        Returns:
            Reconstructed data
        """
        if self.model is None:
            return X_scaled.copy()
        
        if self.model.model_type == ModelType.PLS.value:
            return self.model.model.predict(X_scaled)
            
        elif self.model.model_type == ModelType.LINEAR.value:
            if not self.model.linear_models:
                return X_scaled.copy()
            
            reconstructions = []
            for model_entry in self.model.linear_models:
                other_idx = model_entry["indices"]
                X_others = X_scaled[:, other_idx]
                y_pred = X_others @ model_entry["coef"] + model_entry["intercept"]
                reconstructions.append(y_pred)
            
            return np.column_stack(reconstructions) if reconstructions else X_scaled.copy()
            
        elif self.model.model_type == ModelType.PCA.value:
            X_latent = self.model.model.transform(X_scaled)
            return self.model.model.inverse_transform(X_latent)
        
        return X_scaled.copy()
    
    def score(
        self, 
        X: pd.DataFrame, 
        return_contributions: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute OMR z-scores (reconstruction error normalized by training std).
        
        Args:
            X: Scoring data (n_samples, n_features)
            return_contributions: If True, also return per-sensor contributions
            
        Returns:
            omr_z: OMR z-scores (n_samples,)
            contributions: Optional DataFrame of per-sensor squared residuals (n_samples, n_features)
        """
        if not self._is_fitted or self.model is None:
            zeros = np.zeros(len(X), dtype=np.float32)
            if return_contributions:
                empty_contrib = pd.DataFrame(
                    np.zeros((len(X), len(X.columns))),
                    index=X.index,
                    columns=X.columns
                )
                return zeros, empty_contrib
            return zeros
        
        # Validate and prepare data
        is_valid, _ = self._validate_input(X)
        if not is_valid:
            zeros = np.zeros(len(X), dtype=np.float32)
            if return_contributions:
                empty_contrib = pd.DataFrame(
                    np.zeros((len(X), len(X.columns))),
                    index=X.index,
                    columns=X.columns
                )
                return zeros, empty_contrib
            return zeros
        
        X_clean, _ = self._prepare_data(X)
        
        # Scale
        X_scaled = self.model.scaler.transform(X_clean)
        
        # Reconstruct
        try:
            X_recon = self._reconstruct_data(X_scaled)
                
        except Exception as e:
            from utils.logger import Console
            Console.error(f"[OMR] Reconstruction failed: {e}")
            zeros = np.zeros(len(X), dtype=np.float32)
            if return_contributions:
                empty_contrib = pd.DataFrame(
                    np.zeros((len(X), len(X.columns))),
                    index=X.index,
                    columns=X.columns
                )
                return zeros, empty_contrib
            return zeros
        
        # Compute residuals
        residuals = X_scaled - X_recon
        
        # Per-sensor squared contributions (for root cause analysis)
        squared_residuals = residuals ** 2
        
        # Overall residual norm (L2)
        residual_norm = np.linalg.norm(residuals, axis=1)
        
        # Normalize by training std to get z-score
        omr_z = residual_norm / self.model.train_residual_std
        
        # Clip z-scores to prevent extreme values
        omr_z = np.clip(omr_z, -self.max_z_score, self.max_z_score)
        omr_z = omr_z.astype(np.float32)
        
        if return_contributions:
            # Return per-sensor contributions (squared residuals)
            contrib_df = pd.DataFrame(
                squared_residuals,
                index=X.index,
                columns=self.model.feature_names
            )
            return omr_z, contrib_df
        
        return omr_z
    
    def get_top_contributors(
        self,
        contributions: pd.DataFrame,
        timestamp: pd.Timestamp,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top N sensor contributors for a specific timestamp.
        
        Args:
            contributions: Per-sensor squared residuals DataFrame
            timestamp: Timestamp to analyze
            top_n: Number of top contributors to return
            
        Returns:
            List of (sensor_name, contribution) tuples sorted by contribution
        """
        if timestamp not in contributions.index:
            return []
        
        row = contributions.loc[timestamp]
        top_sensors = row.nlargest(n=top_n)
        
        return [(str(sensor), float(value)) for sensor, value in top_sensors.items()]
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the fitted model.
        
        Returns:
            Dictionary with model diagnostics
        """
        if not self._is_fitted or self.model is None:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "model_type": self.model.model_type,
            "n_features": self.model.train_features,
            "n_samples": self.model.train_samples,
            "n_components": self.model.n_components,
            "train_residual_std": self.model.train_residual_std,
            "feature_names": self.model.feature_names,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        if self.model is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "model": self.model.to_dict()
        }
    
    @classmethod
    def from_dict(cls, payload: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> "OMRDetector":
        """Deserialize from dict."""
        import joblib
        from io import BytesIO
        
        inst = cls(cfg)
        if payload.get("fitted"):
            model_dict = payload["model"]
            
            # Deserialize sklearn model and scaler
            model_obj = joblib.load(BytesIO(model_dict["model_bytes"]))
            scaler_obj = joblib.load(BytesIO(model_dict["scaler_bytes"]))
            
            # Reconstruct OMRModel
            linear_models = None
            if "linear_models" in model_dict:
                linear_models = [
                    {
                        "indices": np.array(entry["indices"], dtype=np.int32),
                        "coef": np.array(entry["coef"], dtype=np.float32),
                        "intercept": float(entry["intercept"]),
                    }
                    for entry in model_dict["linear_models"]
                ]

            inst.model = OMRModel(
                model=model_obj,
                scaler=scaler_obj,
                model_type=model_dict["model_type"],
                feature_names=model_dict["feature_names"],
                train_residual_std=model_dict["train_residual_std"],
                n_components=model_dict["n_components"],
                linear_models=linear_models,
                train_samples=model_dict.get("train_samples", 0),
                train_features=model_dict.get("train_features", 0),
            )
            inst._is_fitted = True
        return inst
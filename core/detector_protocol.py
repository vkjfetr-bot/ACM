"""
ACM v11.0.0 - Detector Protocol

Defines the standardized interface all detectors must implement.

TRAIN-SCORE SEPARATION CONTRACT:
1. fit_baseline() learns ONLY from training data (baseline period)
2. score() uses ONLY parameters learned during fit_baseline()
3. A batch being scored CANNOT influence its own anomaly scores
4. All normalization parameters (mean, std, thresholds) come from training data

VIOLATIONS (will fail contract validation):
- Using score batch mean/std for normalization
- Updating model parameters during scoring
- Adaptive thresholds based on current batch
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class DetectorOutput:
    """
    Standardized output from all detectors.
    
    All detectors MUST return this structure to ensure consistent
    downstream processing in fusion, episodes, and health tracking.
    
    Attributes:
        timestamp: Timestamp column from input data
        z_score: Standardized anomaly score (higher = more anomalous)
        raw_score: Raw detector-specific score before z-normalization
        is_anomaly: Boolean anomaly flag (z_score > threshold)
        confidence: Confidence in the score (0-1), lower when data is missing/uncertain
        detector_name: Name of detector for identification
        feature_contributions: Per-feature contribution to anomaly (optional)
    """
    timestamp: pd.Series
    z_score: pd.Series
    raw_score: pd.Series
    is_anomaly: pd.Series
    confidence: pd.Series
    detector_name: str
    feature_contributions: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Validate output consistency."""
        # Ensure all series have same length
        n = len(self.timestamp)
        if len(self.z_score) != n:
            raise ValueError(f"z_score length {len(self.z_score)} != timestamp length {n}")
        if len(self.raw_score) != n:
            raise ValueError(f"raw_score length {len(self.raw_score)} != timestamp length {n}")
        if len(self.is_anomaly) != n:
            raise ValueError(f"is_anomaly length {len(self.is_anomaly)} != timestamp length {n}")
        if len(self.confidence) != n:
            raise ValueError(f"confidence length {len(self.confidence)} != timestamp length {n}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy manipulation."""
        df = pd.DataFrame({
            "Timestamp": self.timestamp,
            f"{self.detector_name}_z": self.z_score,
            f"{self.detector_name}_raw": self.raw_score,
            f"{self.detector_name}_anomaly": self.is_anomaly,
            f"{self.detector_name}_confidence": self.confidence,
        })
        return df
    
    @property
    def anomaly_count(self) -> int:
        """Count of detected anomalies."""
        return int(self.is_anomaly.sum())
    
    @property
    def mean_z_score(self) -> float:
        """Mean z-score (useful for summary)."""
        return float(self.z_score.mean())
    
    @property
    def max_z_score(self) -> float:
        """Maximum z-score (peak anomaly)."""
        return float(self.z_score.max())


@dataclass
class DetectorMetadata:
    """
    Metadata about a fitted detector.
    
    Used for persistence, versioning, and audit trail.
    """
    detector_name: str
    detector_version: str
    fitted_at: datetime
    n_training_samples: int
    training_start: datetime
    training_end: datetime
    feature_columns: List[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize for SQL storage."""
        return {
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "fitted_at": self.fitted_at.isoformat(),
            "n_training_samples": self.n_training_samples,
            "training_start": self.training_start.isoformat(),
            "training_end": self.training_end.isoformat(),
            "feature_columns": self.feature_columns,
            "hyperparameters": self.hyperparameters,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DetectorMetadata":
        """Deserialize from SQL storage."""
        return cls(
            detector_name=data["detector_name"],
            detector_version=data["detector_version"],
            fitted_at=datetime.fromisoformat(data["fitted_at"]),
            n_training_samples=data["n_training_samples"],
            training_start=datetime.fromisoformat(data["training_start"]),
            training_end=datetime.fromisoformat(data["training_end"]),
            feature_columns=data["feature_columns"],
            hyperparameters=data.get("hyperparameters", {}),
        )


class DetectorProtocol(ABC):
    """
    Abstract base class for all ACM detectors.
    
    All detectors MUST implement this interface to ensure:
    1. Consistent API across all detectors
    2. Train-score separation (critical for valid anomaly detection)
    3. Standardized output format for fusion
    4. Serialization/deserialization for persistence
    
    Example Implementation:
    ```python
    class MyDetector(DetectorProtocol):
        def __init__(self, threshold: float = 3.0):
            self._threshold = threshold
            self._fitted = False
            self._mean = None
            self._std = None
        
        @property
        def name(self) -> str:
            return "my_detector"
        
        @property
        def is_fitted(self) -> bool:
            return self._fitted
        
        def fit_baseline(self, X_train: pd.DataFrame) -> "MyDetector":
            self._mean = X_train.mean()
            self._std = X_train.std()
            self._fitted = True
            return self
        
        def score(self, X_score: pd.DataFrame) -> DetectorOutput:
            if not self.is_fitted:
                raise RuntimeError("Must call fit_baseline() first")
            z = (X_score - self._mean) / self._std
            return DetectorOutput(
                timestamp=X_score.index.to_series(),
                z_score=z.mean(axis=1),
                raw_score=X_score.mean(axis=1),
                is_anomaly=z.mean(axis=1).abs() > self._threshold,
                confidence=pd.Series(1.0, index=X_score.index),
                detector_name=self.name
            )
    ```
    """
    
    # Class-level version for tracking detector implementations
    VERSION: str = "1.0.0"
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Detector name for logging and output columns.
        
        Convention: lowercase, underscore-separated
        Examples: "ar1", "pca_spe", "pca_t2", "iforest", "gmm", "omr"
        """
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Whether detector has been fitted.
        
        Returns True only if fit_baseline() has been called successfully.
        """
        pass
    
    @abstractmethod
    def fit_baseline(self, X_train: pd.DataFrame) -> "DetectorProtocol":
        """
        Fit detector on baseline (training) data.
        
        CONTRACT:
        - MUST learn all parameters from X_train only
        - MUST store all statistics needed for score()
        - MUST NOT be called during scoring
        - MUST return self for method chaining
        
        Args:
            X_train: Baseline data with normalized sensor columns.
                     Index should be timestamps.
                     Columns should be numeric sensor/feature values.
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If X_train is empty or invalid
        """
        pass
    
    @abstractmethod
    def score(self, X_score: pd.DataFrame) -> DetectorOutput:
        """
        Score new data using parameters from fit_baseline().
        
        CONTRACT:
        - MUST use only parameters learned during fit_baseline()
        - MUST NOT update any model parameters
        - MUST NOT use batch statistics for normalization
        - MUST return standardized DetectorOutput
        
        Args:
            X_score: Data to score (same columns as X_train)
            
        Returns:
            DetectorOutput with standardized scores
            
        Raises:
            RuntimeError: If fit_baseline() has not been called
            ValueError: If X_score columns don't match X_train
        """
        pass
    
    def validate_input(self, X: pd.DataFrame, context: str = "input") -> None:
        """
        Validate input data structure.
        
        Override in subclass for detector-specific validation.
        
        Args:
            X: Input DataFrame
            context: Description for error messages ("train" or "score")
            
        Raises:
            ValueError: If input is invalid
        """
        if X is None:
            raise ValueError(f"Detector {self.name}: {context} data is None")
        if X.empty:
            raise ValueError(f"Detector {self.name}: {context} DataFrame is empty")
        if not X.select_dtypes(include=[np.number]).columns.any():
            raise ValueError(f"Detector {self.name}: {context} has no numeric columns")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get fitted parameters for serialization.
        
        Override in subclass to include detector-specific parameters.
        
        Returns:
            Dictionary of parameters that can be JSON-serialized
        """
        return {
            "name": self.name,
            "version": self.VERSION,
            "is_fitted": self.is_fitted,
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set parameters from deserialization.
        
        Override in subclass to restore detector-specific parameters.
        
        Args:
            params: Dictionary from get_params()
        """
        pass
    
    def get_metadata(self, X_train: pd.DataFrame) -> DetectorMetadata:
        """
        Get metadata about the fitted detector.
        
        Args:
            X_train: Training data used for fitting
            
        Returns:
            DetectorMetadata instance
        """
        return DetectorMetadata(
            detector_name=self.name,
            detector_version=self.VERSION,
            fitted_at=datetime.now(),
            n_training_samples=len(X_train),
            training_start=X_train.index.min() if isinstance(X_train.index, pd.DatetimeIndex) else datetime.now(),
            training_end=X_train.index.max() if isinstance(X_train.index, pd.DatetimeIndex) else datetime.now(),
            feature_columns=list(X_train.columns),
            hyperparameters=self.get_params(),
        )


def validate_train_score_separation(detector: DetectorProtocol) -> bool:
    """
    Validate that a detector properly implements train-score separation.
    
    This is a development-time check, not meant for production use.
    
    Args:
        detector: A detector instance to validate
        
    Returns:
        True if detector passes separation contract
        
    Raises:
        AssertionError: If detector violates contract
    """
    # Generate synthetic data
    np.random.seed(42)
    n_train, n_score = 100, 50
    n_features = 5
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
        index=pd.date_range("2024-01-01", periods=n_train, freq="h")
    )
    
    X_score = pd.DataFrame(
        np.random.randn(n_score, n_features) * 2 + 1,  # Different distribution
        columns=[f"feat_{i}" for i in range(n_features)],
        index=pd.date_range("2024-01-05", periods=n_score, freq="h")
    )
    
    # Fit and score
    detector.fit_baseline(X_train)
    output1 = detector.score(X_score)
    
    # Score again - should get identical results (deterministic)
    output2 = detector.score(X_score)
    
    # Verify determinism
    assert np.allclose(output1.z_score.values, output2.z_score.values, equal_nan=True), \
        f"Detector {detector.name} is not deterministic"
    
    # Verify that training data gives different scores than score data
    # (they come from different distributions)
    train_output = detector.score(X_train)
    assert not np.allclose(train_output.z_score.mean(), output1.z_score.mean()), \
        f"Detector {detector.name} may be using batch statistics"
    
    return True


# Export public interface
__all__ = [
    "DetectorOutput",
    "DetectorMetadata", 
    "DetectorProtocol",
    "validate_train_score_separation",
]

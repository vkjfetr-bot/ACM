"""
ACM v11.0.0 - Baseline Normalizer

Centralized baseline normalization for all detectors.

PURPOSE:
Computes statistics ONCE from training data and applies consistently
across all detectors and modules. This prevents each detector from
having its own normalization which can lead to inconsistencies.

USAGE:
    # Fit on training data
    normalizer = BaselineNormalizer().fit(train_df, sensor_cols)
    
    # Normalize for detectors
    X_train_norm = normalizer.normalize(train_df)
    X_score_norm = normalizer.normalize(score_df)
    
    # All detectors use the same normalized data
    ar1_detector.fit_baseline(X_train_norm)
    pca_detector.fit_baseline(X_train_norm)
    
    # Persist for later runs
    normalizer.to_sql(sql_client, equip_id)
    
    # Load from persistence
    normalizer = BaselineNormalizer.from_sql(sql_client, equip_id)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal
import pandas as pd
import numpy as np
from datetime import datetime
import json


@dataclass
class SensorBaseline:
    """
    Baseline statistics for a single sensor.
    
    Stores multiple normalization parameters to support different
    normalization methods without re-computing.
    """
    # Sensor identification
    name: str
    
    # Central tendency
    mean: float
    median: float
    
    # Dispersion
    std: float
    mad: float  # Median Absolute Deviation (robust)
    iqr: float  # Interquartile Range (robust)
    
    # Range
    min_val: float
    max_val: float
    p01: float  # 1st percentile
    p05: float  # 5th percentile
    p95: float  # 95th percentile
    p99: float  # 99th percentile
    
    # Quality metrics
    valid_count: int  # Non-null values
    null_count: int   # Null values
    null_pct: float   # Percentage null
    
    def to_dict(self) -> dict:
        """Serialize for JSON/SQL storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SensorBaseline":
        """Deserialize from JSON/SQL storage."""
        return cls(**data)
    
    @classmethod
    def compute(cls, series: pd.Series, name: str) -> "SensorBaseline":
        """
        Compute baseline statistics from a pandas Series.
        
        Args:
            series: Sensor data (with potential NaN values)
            name: Sensor name
            
        Returns:
            SensorBaseline instance
        """
        valid = series.dropna()
        n_valid = len(valid)
        n_null = series.isna().sum()
        
        if n_valid == 0:
            # All null - return zero baseline
            return cls(
                name=name,
                mean=0.0, median=0.0,
                std=1.0, mad=1.0, iqr=1.0,  # Use 1.0 to avoid division by zero
                min_val=0.0, max_val=0.0,
                p01=0.0, p05=0.0, p95=0.0, p99=0.0,
                valid_count=0, null_count=int(n_null), null_pct=100.0
            )
        
        # Compute all statistics
        q1, q3 = valid.quantile([0.25, 0.75])
        
        return cls(
            name=name,
            mean=float(valid.mean()),
            median=float(valid.median()),
            std=float(valid.std()) if n_valid > 1 else 1.0,
            mad=float((valid - valid.median()).abs().median()) or 1.0,
            iqr=float(q3 - q1) or 1.0,
            min_val=float(valid.min()),
            max_val=float(valid.max()),
            p01=float(valid.quantile(0.01)),
            p05=float(valid.quantile(0.05)),
            p95=float(valid.quantile(0.95)),
            p99=float(valid.quantile(0.99)),
            valid_count=int(n_valid),
            null_count=int(n_null),
            null_pct=float(n_null / (n_valid + n_null) * 100)
        )


@dataclass
class BaselineStatistics:
    """
    Baseline statistics for all sensors.
    
    Container for SensorBaseline instances with metadata.
    """
    sensor_stats: Dict[str, SensorBaseline] = field(default_factory=dict)
    computed_at: Optional[datetime] = None
    n_samples: int = 0
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    version: str = "1.0.0"
    
    def to_dict(self) -> dict:
        """Serialize for JSON/SQL storage."""
        return {
            "sensor_stats": {k: v.to_dict() for k, v in self.sensor_stats.items()},
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
            "n_samples": self.n_samples,
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BaselineStatistics":
        """Deserialize from JSON/SQL storage."""
        return cls(
            sensor_stats={k: SensorBaseline.from_dict(v) for k, v in data["sensor_stats"].items()},
            computed_at=datetime.fromisoformat(data["computed_at"]) if data.get("computed_at") else None,
            n_samples=data.get("n_samples", 0),
            time_start=datetime.fromisoformat(data["time_start"]) if data.get("time_start") else None,
            time_end=datetime.fromisoformat(data["time_end"]) if data.get("time_end") else None,
            version=data.get("version", "1.0.0"),
        )


# Normalization method type
NormMethod = Literal["z-score", "robust", "minmax", "percentile"]


class BaselineNormalizer:
    """
    Centralized baseline normalization.
    
    Computes statistics once from training data and applies consistently
    across all detectors and modules.
    
    NORMALIZATION METHODS:
    - "z-score": (x - mean) / std  [Standard, sensitive to outliers]
    - "robust": (x - median) / mad  [Robust to outliers]
    - "minmax": (x - min) / (max - min)  [Bounded 0-1]
    - "percentile": (x - p05) / (p95 - p05)  [Bounded, robust]
    
    USAGE:
        normalizer = BaselineNormalizer().fit(train_df, sensor_cols)
        normalized = normalizer.normalize(score_df, method="z-score")
    """
    
    VERSION: str = "1.0.0"
    
    def __init__(self, default_method: NormMethod = "z-score"):
        """
        Initialize normalizer.
        
        Args:
            default_method: Default normalization method
        """
        self.baseline: Optional[BaselineStatistics] = None
        self.default_method = default_method
        self._fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted."""
        return self._fitted and self.baseline is not None
    
    @property
    def sensor_names(self) -> List[str]:
        """List of sensor names in baseline."""
        if not self.is_fitted:
            return []
        return list(self.baseline.sensor_stats.keys())
    
    def fit(self, X_train: pd.DataFrame, sensor_cols: Optional[List[str]] = None) -> "BaselineNormalizer":
        """
        Compute baseline statistics from training data.
        
        Args:
            X_train: Training DataFrame with timestamp index
            sensor_cols: List of columns to include. If None, uses all numeric columns.
            
        Returns:
            self for method chaining
        """
        if X_train.empty:
            raise ValueError("Cannot fit on empty DataFrame")
        
        # Determine columns to use
        if sensor_cols is None:
            sensor_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Compute statistics for each sensor
        stats = {}
        for col in sensor_cols:
            if col not in X_train.columns:
                continue
            stats[col] = SensorBaseline.compute(X_train[col], col)
        
        if not stats:
            raise ValueError("No valid sensor columns found")
        
        # Determine time range
        time_start = None
        time_end = None
        if isinstance(X_train.index, pd.DatetimeIndex):
            time_start = X_train.index.min().to_pydatetime()
            time_end = X_train.index.max().to_pydatetime()
        
        self.baseline = BaselineStatistics(
            sensor_stats=stats,
            computed_at=datetime.now(),
            n_samples=len(X_train),
            time_start=time_start,
            time_end=time_end,
            version=self.VERSION,
        )
        self._fitted = True
        
        return self
    
    def normalize(self, X: pd.DataFrame, method: Optional[NormMethod] = None) -> pd.DataFrame:
        """
        Normalize data using baseline statistics.
        
        Args:
            X: Data to normalize
            method: Normalization method (uses default if None)
            
        Returns:
            Normalized DataFrame (same shape as input)
            
        Raises:
            RuntimeError: If not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before normalize()")
        
        method = method or self.default_method
        result = X.copy()
        
        for col, stats in self.baseline.sensor_stats.items():
            if col not in result.columns:
                continue
            
            # Apply normalization based on method
            if method == "z-score":
                denom = stats.std if stats.std > 1e-10 else 1.0
                result[col] = (result[col] - stats.mean) / denom
                
            elif method == "robust":
                denom = stats.mad if stats.mad > 1e-10 else 1.0
                result[col] = (result[col] - stats.median) / denom
                
            elif method == "minmax":
                denom = stats.max_val - stats.min_val
                denom = denom if denom > 1e-10 else 1.0
                result[col] = (result[col] - stats.min_val) / denom
                
            elif method == "percentile":
                denom = stats.p95 - stats.p05
                denom = denom if denom > 1e-10 else 1.0
                result[col] = (result[col] - stats.p05) / denom
        
        return result
    
    def denormalize(self, X: pd.DataFrame, method: Optional[NormMethod] = None) -> pd.DataFrame:
        """
        Reverse normalization (convert back to original scale).
        
        Args:
            X: Normalized data
            method: Normalization method used (uses default if None)
            
        Returns:
            Denormalized DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before denormalize()")
        
        method = method or self.default_method
        result = X.copy()
        
        for col, stats in self.baseline.sensor_stats.items():
            if col not in result.columns:
                continue
            
            if method == "z-score":
                result[col] = result[col] * stats.std + stats.mean
                
            elif method == "robust":
                result[col] = result[col] * stats.mad + stats.median
                
            elif method == "minmax":
                result[col] = result[col] * (stats.max_val - stats.min_val) + stats.min_val
                
            elif method == "percentile":
                result[col] = result[col] * (stats.p95 - stats.p05) + stats.p05
        
        return result
    
    def get_sensor_stats(self, sensor_name: str) -> Optional[SensorBaseline]:
        """
        Get baseline statistics for a specific sensor.
        
        Args:
            sensor_name: Name of sensor
            
        Returns:
            SensorBaseline or None if not found
        """
        if not self.is_fitted:
            return None
        return self.baseline.sensor_stats.get(sensor_name)
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """
        Get summary of baseline quality.
        
        Returns:
            Dictionary with quality metrics
        """
        if not self.is_fitted:
            return {"fitted": False}
        
        null_pcts = [s.null_pct for s in self.baseline.sensor_stats.values()]
        
        return {
            "fitted": True,
            "n_sensors": len(self.baseline.sensor_stats),
            "n_samples": self.baseline.n_samples,
            "time_range_hours": (
                (self.baseline.time_end - self.baseline.time_start).total_seconds() / 3600
                if self.baseline.time_start and self.baseline.time_end else 0
            ),
            "avg_null_pct": np.mean(null_pcts) if null_pcts else 0,
            "max_null_pct": max(null_pcts) if null_pcts else 0,
            "sensors_with_nulls": sum(1 for p in null_pcts if p > 0),
        }
    
    def to_dict(self) -> dict:
        """Serialize for JSON/SQL storage."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted normalizer")
        return {
            "baseline": self.baseline.to_dict(),
            "default_method": self.default_method,
            "version": self.VERSION,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BaselineNormalizer":
        """Deserialize from JSON/SQL storage."""
        normalizer = cls(default_method=data.get("default_method", "z-score"))
        normalizer.baseline = BaselineStatistics.from_dict(data["baseline"])
        normalizer._fitted = True
        return normalizer
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "BaselineNormalizer":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Export public interface
__all__ = [
    "SensorBaseline",
    "BaselineStatistics",
    "BaselineNormalizer",
    "NormMethod",
]

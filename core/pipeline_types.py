# core/pipeline_types.py
"""
ACM Pipeline Types - v11.0.0

Central type definitions for the ACM pipeline.
These types establish contracts between pipeline stages and ensure
clean separation between ONLINE (real-time) and OFFLINE (batch) modes.

Phase 1 Implementation Items:
- P1.1: PipelineMode enum (ONLINE/OFFLINE)
- P1.2: DataContract dataclass
- P1.3: SensorValidator (basic structure)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# P1.1 - PipelineMode Enum
# =============================================================================

class PipelineMode(Enum):
    """
    Defines the execution mode for the ACM pipeline.
    
    ONLINE: Real-time streaming processing
        - Single observation at a time
        - No batch aggregations
        - Incremental model updates only
        - Lower latency requirements
        
    OFFLINE: Batch processing mode  
        - Multiple observations processed together
        - Full batch aggregations allowed
        - Full model refit allowed
        - Higher throughput focus
    """
    ONLINE = auto()
    OFFLINE = auto()
    
    @classmethod
    def from_env(cls) -> "PipelineMode":
        """Detect pipeline mode from environment variables."""
        import os
        # BATCH mode is set by sql_batch_runner.py
        if os.getenv("ACM_BATCH_MODE", "").lower() in ("1", "true", "yes"):
            return cls.OFFLINE
        return cls.ONLINE
    
    @classmethod  
    def from_config(cls, cfg: Dict[str, Any]) -> "PipelineMode":
        """Detect pipeline mode from configuration."""
        mode_str = cfg.get("pipeline", {}).get("mode", "offline").lower()
        if mode_str == "online":
            return cls.ONLINE
        return cls.OFFLINE
    
    @property
    def allows_batch_aggregation(self) -> bool:
        """Whether this mode allows batch-level aggregations."""
        return self == PipelineMode.OFFLINE
    
    @property
    def allows_model_refit(self) -> bool:
        """Whether this mode allows full model refit."""
        return self == PipelineMode.OFFLINE
    
    @property
    def max_latency_ms(self) -> int:
        """Maximum acceptable latency for this mode."""
        return 100 if self == PipelineMode.ONLINE else 300000  # 100ms vs 5min


# =============================================================================
# P1.2 - DataContract
# =============================================================================

@dataclass
class SensorMeta:
    """Metadata for a single sensor column."""
    name: str
    unit: str = ""
    expected_range: Tuple[float, float] = (-np.inf, np.inf)
    is_required: bool = True
    sensor_type: str = "continuous"  # continuous, binary, categorical


@dataclass  
class DataContract:
    """
    Contract defining expected data characteristics for pipeline stages.
    
    The DataContract ensures that:
    1. Required sensors are present
    2. Data types are correct
    3. Time range is valid
    4. Data quality meets minimum thresholds
    
    Use validate() to check incoming data against the contract.
    """
    
    # Required sensor columns
    required_sensors: List[str] = field(default_factory=list)
    
    # Optional sensor columns (used if present)
    optional_sensors: List[str] = field(default_factory=list)
    
    # Timestamp column name
    timestamp_col: str = "Timestamp"
    
    # Expected minimum rows
    min_rows: int = 100
    
    # Maximum allowed null fraction per column
    max_null_fraction: float = 0.3
    
    # Maximum allowed constant columns (zero variance)
    max_constant_fraction: float = 0.5
    
    # Expected time range (optional)
    expected_start: Optional[pd.Timestamp] = None
    expected_end: Optional[pd.Timestamp] = None
    
    # Equipment context
    equip_id: int = 0
    equip_code: str = ""
    
    # Sensor metadata (optional detailed specs)
    sensor_meta: Dict[str, SensorMeta] = field(default_factory=dict)
    
    # Contract version for compatibility checking
    version: str = "1.0"
    
    def validate(self, df: pd.DataFrame) -> "ValidationResult":
        """
        Validate a DataFrame against this contract.
        
        Returns ValidationResult with pass/fail and detailed issues.
        """
        issues: List[str] = []
        warnings: List[str] = []
        
        # Check timestamp column
        if self.timestamp_col not in df.columns:
            issues.append(f"Missing timestamp column: {self.timestamp_col}")
        
        # Check required sensors
        present_cols = set(df.columns)
        missing_required = set(self.required_sensors) - present_cols
        if missing_required:
            issues.append(f"Missing required sensors: {sorted(missing_required)}")
        
        # Check optional sensors (warn only)
        missing_optional = set(self.optional_sensors) - present_cols
        if missing_optional:
            warnings.append(f"Missing optional sensors: {sorted(missing_optional)}")
        
        # Check minimum rows
        if len(df) < self.min_rows:
            issues.append(f"Insufficient rows: {len(df)} < {self.min_rows}")
        
        # Check null fractions
        if len(df) > 0:
            null_fractions = df.isnull().mean()
            high_null_cols = null_fractions[null_fractions > self.max_null_fraction].index.tolist()
            if high_null_cols:
                warnings.append(f"High null fraction columns: {high_null_cols}")
        
        # Check constant columns (zero variance)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and len(df) > 1:
            variances = df[numeric_cols].var()
            constant_cols = variances[variances == 0].index.tolist()
            constant_frac = len(constant_cols) / len(numeric_cols) if len(numeric_cols) > 0 else 0
            if constant_frac > self.max_constant_fraction:
                warnings.append(f"Too many constant columns: {len(constant_cols)}/{len(numeric_cols)}")
        
        # Check time range if specified
        if self.expected_start and self.timestamp_col in df.columns:
            try:
                actual_start = pd.Timestamp(df[self.timestamp_col].min())
                if actual_start < self.expected_start:
                    warnings.append(f"Data starts before expected: {actual_start} < {self.expected_start}")
            except Exception:
                pass
        
        if self.expected_end and self.timestamp_col in df.columns:
            try:
                actual_end = pd.Timestamp(df[self.timestamp_col].max())
                if actual_end > self.expected_end:
                    warnings.append(f"Data ends after expected: {actual_end} > {self.expected_end}")
            except Exception:
                pass
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            rows_validated=len(df),
            columns_validated=len(df.columns)
        )
    
    def get_available_sensors(self, df: pd.DataFrame) -> List[str]:
        """Get list of contract sensors that are available in the DataFrame."""
        all_expected = set(self.required_sensors) | set(self.optional_sensors)
        return [c for c in df.columns if c in all_expected]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize contract to dictionary."""
        return {
            "version": self.version,
            "required_sensors": self.required_sensors,
            "optional_sensors": self.optional_sensors,
            "timestamp_col": self.timestamp_col,
            "min_rows": self.min_rows,
            "max_null_fraction": self.max_null_fraction,
            "max_constant_fraction": self.max_constant_fraction,
            "equip_id": self.equip_id,
            "equip_code": self.equip_code,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataContract":
        """Deserialize contract from dictionary."""
        return cls(
            required_sensors=data.get("required_sensors", []),
            optional_sensors=data.get("optional_sensors", []),
            timestamp_col=data.get("timestamp_col", "Timestamp"),
            min_rows=data.get("min_rows", 100),
            max_null_fraction=data.get("max_null_fraction", 0.3),
            max_constant_fraction=data.get("max_constant_fraction", 0.5),
            equip_id=data.get("equip_id", 0),
            equip_code=data.get("equip_code", ""),
            version=data.get("version", "1.0"),
        )
    
    def signature(self) -> str:
        """Compute a hash signature for this contract."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class ValidationResult:
    """Result of validating data against a DataContract."""
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rows_validated: int = 0
    columns_validated: int = 0
    
    def __bool__(self) -> bool:
        return self.passed
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        parts = [f"Validation {status}: {self.rows_validated} rows, {self.columns_validated} cols"]
        if self.issues:
            parts.append(f"  Issues: {self.issues}")
        if self.warnings:
            parts.append(f"  Warnings: {self.warnings}")
        return "\n".join(parts)


# =============================================================================
# P1.3 - SensorValidator  
# =============================================================================

class SensorValidator:
    """
    Validates sensor data quality before pipeline processing.
    
    Checks:
    - Required sensors present
    - Physical range validation
    - Stale data detection
    - Outlier flagging
    
    Thread-safe and stateless per validation call.
    """
    
    def __init__(self, contract: Optional[DataContract] = None):
        """
        Initialize validator with optional data contract.
        
        Args:
            contract: DataContract defining expected sensors and constraints
        """
        self.contract = contract
        
        # Default physical limits for common sensor types
        self.physical_limits: Dict[str, Tuple[float, float]] = {
            "temperature": (-100.0, 1500.0),  # Celsius
            "pressure": (0.0, 1000.0),  # bar
            "vibration": (0.0, 1000.0),  # mm/s or g
            "flow": (0.0, 100000.0),  # m3/h
            "speed": (0.0, 50000.0),  # RPM
            "power": (0.0, 1000000.0),  # kW
            "current": (0.0, 10000.0),  # A
            "voltage": (0.0, 100000.0),  # V
        }
        
        # Stale data threshold (seconds)
        self.stale_threshold_seconds: float = 3600.0  # 1 hour
    
    def validate(self, 
                 df: pd.DataFrame,
                 timestamp_col: str = "Timestamp") -> ValidationResult:
        """
        Validate sensor data quality.
        
        Args:
            df: DataFrame with sensor data
            timestamp_col: Name of timestamp column
            
        Returns:
            ValidationResult with pass/fail and issues
        """
        issues: List[str] = []
        warnings: List[str] = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return ValidationResult(passed=False, issues=issues)
        
        # Validate against contract if available
        if self.contract:
            contract_result = self.contract.validate(df)
            issues.extend(contract_result.issues)
            warnings.extend(contract_result.warnings)
        
        # Check for duplicate timestamps
        if timestamp_col in df.columns:
            dup_count = df[timestamp_col].duplicated().sum()
            if dup_count > 0:
                warnings.append(f"Duplicate timestamps: {dup_count}")
        
        # Check for stale data
        if timestamp_col in df.columns:
            try:
                latest = pd.Timestamp(df[timestamp_col].max())
                now = pd.Timestamp.now()
                staleness = (now - latest).total_seconds()
                if staleness > self.stale_threshold_seconds:
                    warnings.append(f"Stale data: latest is {staleness/3600:.1f} hours old")
            except Exception:
                pass
        
        # Check for physical limit violations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            sensor_type = self._infer_sensor_type(col)
            if sensor_type and sensor_type in self.physical_limits:
                lo, hi = self.physical_limits[sensor_type]
                violations = ((df[col] < lo) | (df[col] > hi)).sum()
                if violations > 0:
                    warnings.append(f"Physical limit violations in {col}: {violations}")
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            rows_validated=len(df),
            columns_validated=len(df.columns)
        )
    
    def _infer_sensor_type(self, column_name: str) -> Optional[str]:
        """Infer sensor type from column name."""
        name_lower = column_name.lower()
        
        if any(kw in name_lower for kw in ["temp", "tmp"]):
            return "temperature"
        if any(kw in name_lower for kw in ["press", "psi", "bar"]):
            return "pressure"
        if any(kw in name_lower for kw in ["vib", "vibration"]):
            return "vibration"
        if any(kw in name_lower for kw in ["flow", "gpm"]):
            return "flow"
        if any(kw in name_lower for kw in ["speed", "rpm"]):
            return "speed"
        if any(kw in name_lower for kw in ["power", "kw", "mw"]):
            return "power"
        if any(kw in name_lower for kw in ["current", "amp"]):
            return "current"
        if any(kw in name_lower for kw in ["volt", "voltage"]):
            return "voltage"
        
        return None
    
    def filter_valid_sensors(self, 
                             df: pd.DataFrame,
                             min_variance: float = 1e-10) -> pd.DataFrame:
        """
        Filter DataFrame to only valid sensor columns.
        
        Removes:
        - Non-numeric columns (except timestamp)
        - Constant columns (zero variance)
        - Columns with too many nulls
        
        Args:
            df: Input DataFrame
            min_variance: Minimum variance threshold
            
        Returns:
            Filtered DataFrame with valid sensors only
        """
        timestamp_col = self.contract.timestamp_col if self.contract else "Timestamp"
        
        # Keep timestamp if present
        keep_cols = []
        if timestamp_col in df.columns:
            keep_cols.append(timestamp_col)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter by variance and null fraction
        max_null = self.contract.max_null_fraction if self.contract else 0.3
        
        for col in numeric_cols:
            if col == timestamp_col:
                continue
            
            null_frac = df[col].isnull().mean()
            if null_frac > max_null:
                continue
            
            variance = df[col].var()
            if variance is None or variance < min_variance:
                continue
            
            keep_cols.append(col)
        
        return df[keep_cols].copy()


# =============================================================================
# Feature Matrix Types (used in Phase 2+)
# =============================================================================

@dataclass
class FeatureMatrix:
    """
    Standardized container for processed feature data.
    
    Separates different feature types to prevent data leakage
    between regime detection and anomaly detection.
    """
    
    # Core sensor features (used by detectors)
    sensor_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Regime detection inputs (no detector outputs!)
    regime_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Statistical/derived features
    stat_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Timestamp index
    timestamps: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    
    # Feature metadata
    feature_names: List[str] = field(default_factory=list)
    
    # Processing metadata
    created_at: datetime = field(default_factory=datetime.now)
    source_rows: int = 0
    
    # Forbidden patterns for regime inputs (prevent leakage)
    _FORBIDDEN_REGIME_PATTERNS: List[str] = field(
        default_factory=lambda: ["_z", "pca_", "iforest_", "gmm_", "omr_", "ar1_", "_score"]
    )
    
    def get_regime_inputs(self) -> pd.DataFrame:
        """
        Get features for regime detection.
        
        CRITICAL: This method validates that no detector outputs
        leak into regime inputs (which would cause data leakage).
        """
        if self.regime_features.empty:
            return self.regime_features
        
        # Validate no forbidden patterns in columns
        for col in self.regime_features.columns:
            col_lower = col.lower()
            for pattern in self._FORBIDDEN_REGIME_PATTERNS:
                if pattern in col_lower:
                    raise ValueError(
                        f"Data leakage detected: column '{col}' contains "
                        f"forbidden pattern '{pattern}' in regime inputs"
                    )
        
        return self.regime_features
    
    def get_detector_inputs(self) -> pd.DataFrame:
        """Get features for anomaly detectors."""
        return self.sensor_features
    
    def signature(self) -> str:
        """Compute hash of feature matrix structure."""
        content = json.dumps({
            "sensor_cols": list(self.sensor_features.columns),
            "regime_cols": list(self.regime_features.columns),
            "stat_cols": list(self.stat_features.columns),
            "source_rows": self.source_rows,
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PipelineMode",
    "DataContract", 
    "ValidationResult",
    "SensorMeta",
    "SensorValidator",
    "FeatureMatrix",
]

"""
Standardized Feature Matrix for ACM v11.0.0

Provides a canonical schema for feature data flowing through the pipeline.
Ensures consistent column naming and categorization across all detectors.

Phase 1.6 Implementation - CRITICAL PREREQUISITE for Phase 3
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
import pandas as pd
import numpy as np

from core.observability import Console


# =============================================================================
# Schema Definition
# =============================================================================

@dataclass
class FeatureSchema:
    """
    Defines canonical column schema for feature matrix.
    
    All columns follow a prefix convention:
    - raw_*: Original sensor values
    - norm_*: Normalized sensor values
    - feat_*: Engineered features
    - lag_*: Lag features
    - roll_*: Rolling statistics
    - regime_*: Regime-related columns
    """
    
    # Required columns (always present)
    TIMESTAMP_COL: str = "Timestamp"
    EQUIP_ID_COL: str = "EquipID"
    
    # Column prefixes for categorization
    RAW_SENSOR_PREFIX: str = "raw_"      # Original sensor values
    NORM_SENSOR_PREFIX: str = "norm_"    # Normalized sensor values
    FEATURE_PREFIX: str = "feat_"        # Engineered features
    LAG_PREFIX: str = "lag_"             # Lag features
    ROLL_PREFIX: str = "roll_"           # Rolling statistics
    REGIME_PREFIX: str = "regime_"       # Regime-related columns
    
    # Excluded from regime discovery (detector outputs)
    DETECTOR_PREFIXES: Set[str] = field(default_factory=lambda: {
        "ar1_", "pca_", "iforest_", "gmm_", "omr_", "fused_", "corr_"
    })
    
    # Health-related columns to exclude from regime discovery
    HEALTH_PATTERNS: Set[str] = field(default_factory=lambda: {
        "health", "Health", "fused", "Fused", "episode", "Episode"
    })
    
    def is_raw_sensor(self, col: str) -> bool:
        """Check if column is a raw sensor value."""
        return col.startswith(self.RAW_SENSOR_PREFIX)
    
    def is_normalized(self, col: str) -> bool:
        """Check if column is normalized."""
        return col.startswith(self.NORM_SENSOR_PREFIX)
    
    def is_feature(self, col: str) -> bool:
        """Check if column is an engineered feature."""
        return col.startswith(self.FEATURE_PREFIX)
    
    def is_detector_output(self, col: str) -> bool:
        """Check if column is a detector output."""
        return any(col.startswith(prefix) for prefix in self.DETECTOR_PREFIXES)
    
    def is_regime_excluded(self, col: str) -> bool:
        """Check if column should be excluded from regime discovery."""
        # Check detector prefixes
        if self.is_detector_output(col):
            return True
        
        # Check health patterns
        for pattern in self.HEALTH_PATTERNS:
            if pattern in col:
                return True
        
        return False
    
    def categorize_column(self, col: str) -> str:
        """Get category for a column."""
        if col in (self.TIMESTAMP_COL, self.EQUIP_ID_COL):
            return "metadata"
        if self.is_raw_sensor(col):
            return "raw_sensor"
        if self.is_normalized(col):
            return "normalized"
        if self.is_feature(col):
            return "feature"
        if col.startswith(self.LAG_PREFIX):
            return "lag"
        if col.startswith(self.ROLL_PREFIX):
            return "rolling"
        if col.startswith(self.REGIME_PREFIX):
            return "regime"
        if self.is_detector_output(col):
            return "detector"
        return "other"


# =============================================================================
# Feature Matrix
# =============================================================================

@dataclass
class FeatureMatrix:
    """
    Standardized feature matrix with schema enforcement.
    
    Wraps a DataFrame and provides:
    - Schema validation
    - Column categorization
    - Subset extraction for different pipeline stages
    
    Example:
        matrix = FeatureMatrix(df)
        regime_inputs = matrix.get_regime_inputs()
        detector_inputs = matrix.get_detector_inputs()
    """
    
    data: pd.DataFrame
    schema: FeatureSchema = field(default_factory=FeatureSchema)
    
    # Metadata (computed in __post_init__)
    n_rows: int = field(init=False)
    n_raw_sensors: int = field(init=False)
    n_normalized: int = field(init=False)
    n_features: int = field(init=False)
    sensor_names: List[str] = field(init=False)
    feature_names: List[str] = field(init=False)
    
    def __post_init__(self):
        """Validate schema and extract metadata."""
        self._validate_schema()
        self._extract_metadata()
    
    def _validate_schema(self) -> None:
        """Ensure required columns exist."""
        if self.schema.TIMESTAMP_COL not in self.data.columns:
            raise ValueError(f"Missing required column: {self.schema.TIMESTAMP_COL}")
    
    def _extract_metadata(self) -> None:
        """Extract metadata from columns."""
        cols = self.data.columns.tolist()
        
        self.n_rows = len(self.data)
        
        # Categorize columns
        self.sensor_names = [c for c in cols if self.schema.is_raw_sensor(c)]
        self.feature_names = [c for c in cols if self.schema.is_feature(c)]
        
        normalized = [c for c in cols if self.schema.is_normalized(c)]
        
        self.n_raw_sensors = len(self.sensor_names)
        self.n_normalized = len(normalized)
        self.n_features = len(self.feature_names)
    
    # -------------------------------------------------------------------------
    # Column Extraction
    # -------------------------------------------------------------------------
    
    def get_columns_by_category(self, category: str) -> List[str]:
        """Get column names for a category."""
        return [c for c in self.data.columns 
                if self.schema.categorize_column(c) == category]
    
    def get_all_sensor_columns(self) -> List[str]:
        """Get all sensor columns (raw and normalized)."""
        raw = self.get_columns_by_category("raw_sensor")
        norm = self.get_columns_by_category("normalized")
        return raw + norm
    
    def get_regime_inputs(self) -> pd.DataFrame:
        """
        Get columns suitable for regime discovery.
        
        Excludes:
        - Detector outputs (ar1_z, pca_spe_z, etc.)
        - Health-related columns
        - Fused scores
        """
        excluded = set()
        
        for col in self.data.columns:
            if self.schema.is_regime_excluded(col):
                excluded.add(col)
        
        valid_cols = [c for c in self.data.columns if c not in excluded]
        return self.data[valid_cols].copy()
    
    def get_detector_inputs(self) -> pd.DataFrame:
        """
        Get columns suitable for detector scoring.
        
        Returns timestamp + raw sensors + features (no detector outputs).
        """
        cols = [self.schema.TIMESTAMP_COL]
        
        # Add raw sensors
        cols.extend(self.sensor_names)
        
        # Add normalized if available
        cols.extend(self.get_columns_by_category("normalized"))
        
        # Add features
        cols.extend(self.feature_names)
        
        # Filter to columns that exist
        valid_cols = [c for c in cols if c in self.data.columns]
        return self.data[valid_cols].copy()
    
    def get_numeric_columns(self, exclude_timestamp: bool = True) -> List[str]:
        """Get all numeric columns."""
        numeric = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_timestamp:
            numeric = [c for c in numeric if c != self.schema.TIMESTAMP_COL]
        
        return numeric
    
    # -------------------------------------------------------------------------
    # Data Access
    # -------------------------------------------------------------------------
    
    @property
    def timestamps(self) -> pd.Series:
        """Get timestamp column."""
        return self.data[self.schema.TIMESTAMP_COL]
    
    @property
    def equip_id(self) -> Optional[int]:
        """Get equipment ID if present."""
        if self.schema.EQUIP_ID_COL in self.data.columns:
            unique_ids = self.data[self.schema.EQUIP_ID_COL].unique()
            if len(unique_ids) == 1:
                return int(unique_ids[0])
        return None
    
    def get_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get (min, max) timestamps."""
        ts = self.timestamps
        return ts.min(), ts.max()
    
    def slice_by_time(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None
    ) -> "FeatureMatrix":
        """Get a time-sliced copy of the matrix."""
        mask = pd.Series(True, index=self.data.index)
        
        if start is not None:
            mask &= self.timestamps >= start
        if end is not None:
            mask &= self.timestamps <= end
        
        return FeatureMatrix(
            data=self.data[mask].copy(),
            schema=self.schema
        )
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    def validate(self) -> List[str]:
        """
        Validate the feature matrix.
        
        Returns list of validation issues (empty if valid).
        """
        issues = []
        
        # Check required columns
        if self.schema.TIMESTAMP_COL not in self.data.columns:
            issues.append(f"Missing required column: {self.schema.TIMESTAMP_COL}")
        
        # Check for empty data
        if self.data.empty:
            issues.append("DataFrame is empty")
        
        # Check for NaN-only columns
        for col in self.data.columns:
            if self.data[col].isna().all():
                issues.append(f"Column '{col}' is all NaN")
        
        # Check timestamp ordering
        if self.schema.TIMESTAMP_COL in self.data.columns:
            ts = self.data[self.schema.TIMESTAMP_COL]
            if not ts.is_monotonic_increasing:
                issues.append("Timestamps are not monotonically increasing")
        
        return issues
    
    def is_valid(self) -> bool:
        """Check if matrix is valid."""
        return len(self.validate()) == 0
    
    # -------------------------------------------------------------------------
    # Summary & Display
    # -------------------------------------------------------------------------
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        time_range = self.get_time_range()
        
        return {
            "n_rows": self.n_rows,
            "n_columns": len(self.data.columns),
            "n_raw_sensors": self.n_raw_sensors,
            "n_normalized": self.n_normalized,
            "n_features": self.n_features,
            "time_start": time_range[0],
            "time_end": time_range[1],
            "equip_id": self.equip_id,
            "column_categories": self.get_column_breakdown(),
        }
    
    def get_column_breakdown(self) -> Dict[str, int]:
        """Get count of columns by category."""
        breakdown: Dict[str, int] = {}
        
        for col in self.data.columns:
            category = self.schema.categorize_column(col)
            breakdown[category] = breakdown.get(category, 0) + 1
        
        return breakdown
    
    def __repr__(self) -> str:
        return (
            f"FeatureMatrix(rows={self.n_rows}, "
            f"sensors={self.n_raw_sensors}, "
            f"features={self.n_features})"
        )


# =============================================================================
# Builder
# =============================================================================

class FeatureMatrixBuilder:
    """
    Builder for constructing FeatureMatrix from raw data.
    
    Handles column renaming to match schema conventions.
    
    Example:
        builder = FeatureMatrixBuilder()
        builder.set_raw_sensors(df, ["temp", "pressure"])
        builder.add_features(features_df)
        matrix = builder.build()
    """
    
    def __init__(self, schema: Optional[FeatureSchema] = None):
        self.schema = schema or FeatureSchema()
        self._data: Dict[str, pd.Series] = {}
        self._timestamps: Optional[pd.Series] = None
        self._equip_id: Optional[int] = None
    
    def set_timestamps(self, timestamps: pd.Series) -> "FeatureMatrixBuilder":
        """Set timestamp column."""
        self._timestamps = timestamps
        return self
    
    def set_equip_id(self, equip_id: int) -> "FeatureMatrixBuilder":
        """Set equipment ID."""
        self._equip_id = equip_id
        return self
    
    def add_raw_sensors(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str]
    ) -> "FeatureMatrixBuilder":
        """Add raw sensor columns with proper prefix."""
        for col in sensor_cols:
            if col in df.columns:
                prefixed = f"{self.schema.RAW_SENSOR_PREFIX}{col}"
                self._data[prefixed] = df[col].copy()
        return self
    
    def add_normalized_sensors(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str]
    ) -> "FeatureMatrixBuilder":
        """Add normalized sensor columns with proper prefix."""
        for col in sensor_cols:
            if col in df.columns:
                prefixed = f"{self.schema.NORM_SENSOR_PREFIX}{col}"
                self._data[prefixed] = df[col].copy()
        return self
    
    def add_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> "FeatureMatrixBuilder":
        """Add feature columns with proper prefix."""
        if feature_cols is None:
            feature_cols = df.columns.tolist()
        
        for col in feature_cols:
            if col in df.columns:
                # Skip if already prefixed
                if any(col.startswith(p) for p in 
                       [self.schema.RAW_SENSOR_PREFIX, 
                        self.schema.NORM_SENSOR_PREFIX,
                        self.schema.FEATURE_PREFIX]):
                    self._data[col] = df[col].copy()
                else:
                    prefixed = f"{self.schema.FEATURE_PREFIX}{col}"
                    self._data[prefixed] = df[col].copy()
        return self
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        lag_cols: List[str]
    ) -> "FeatureMatrixBuilder":
        """Add lag feature columns."""
        for col in lag_cols:
            if col in df.columns:
                prefixed = f"{self.schema.LAG_PREFIX}{col}"
                self._data[prefixed] = df[col].copy()
        return self
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        roll_cols: List[str]
    ) -> "FeatureMatrixBuilder":
        """Add rolling statistic columns."""
        for col in roll_cols:
            if col in df.columns:
                prefixed = f"{self.schema.ROLL_PREFIX}{col}"
                self._data[prefixed] = df[col].copy()
        return self
    
    def build(self) -> FeatureMatrix:
        """Build the FeatureMatrix."""
        if self._timestamps is None:
            raise ValueError("Timestamps must be set before building")
        
        # Create DataFrame
        data = pd.DataFrame(self._data)
        data[self.schema.TIMESTAMP_COL] = self._timestamps.values
        
        if self._equip_id is not None:
            data[self.schema.EQUIP_ID_COL] = self._equip_id
        
        # Reorder columns
        meta_cols = [self.schema.TIMESTAMP_COL]
        if self.schema.EQUIP_ID_COL in data.columns:
            meta_cols.append(self.schema.EQUIP_ID_COL)
        
        other_cols = [c for c in data.columns if c not in meta_cols]
        data = data[meta_cols + sorted(other_cols)]
        
        return FeatureMatrix(data=data, schema=self.schema)
    
    def reset(self) -> "FeatureMatrixBuilder":
        """Reset builder state."""
        self._data.clear()
        self._timestamps = None
        self._equip_id = None
        return self


# =============================================================================
# Factory Functions
# =============================================================================

def from_dataframe(
    df: pd.DataFrame,
    sensor_cols: List[str],
    timestamp_col: str = "Timestamp",
    equip_id_col: Optional[str] = "EquipID",
    auto_prefix: bool = True
) -> FeatureMatrix:
    """
    Create FeatureMatrix from a DataFrame.
    
    Args:
        df: Source DataFrame
        sensor_cols: List of sensor column names
        timestamp_col: Name of timestamp column
        equip_id_col: Name of equipment ID column (or None)
        auto_prefix: Whether to add prefixes to columns
        
    Returns:
        FeatureMatrix instance
    """
    schema = FeatureSchema()
    
    if auto_prefix:
        # Rename columns to follow schema
        renamed = {}
        
        for col in df.columns:
            if col == timestamp_col:
                renamed[col] = schema.TIMESTAMP_COL
            elif col == equip_id_col:
                renamed[col] = schema.EQUIP_ID_COL
            elif col in sensor_cols:
                renamed[col] = f"{schema.RAW_SENSOR_PREFIX}{col}"
            elif not any(col.startswith(p) for p in 
                        [schema.RAW_SENSOR_PREFIX, schema.NORM_SENSOR_PREFIX,
                         schema.FEATURE_PREFIX, schema.LAG_PREFIX, schema.ROLL_PREFIX]):
                # Non-sensor columns become features
                if col not in (timestamp_col, equip_id_col):
                    renamed[col] = f"{schema.FEATURE_PREFIX}{col}"
        
        df = df.rename(columns=renamed)
    
    return FeatureMatrix(data=df.copy(), schema=schema)


def merge_matrices(*matrices: FeatureMatrix) -> FeatureMatrix:
    """
    Merge multiple FeatureMatrix objects by timestamp.
    
    Args:
        *matrices: FeatureMatrix instances to merge
        
    Returns:
        Merged FeatureMatrix
    """
    if not matrices:
        raise ValueError("At least one matrix required")
    
    if len(matrices) == 1:
        return matrices[0]
    
    schema = matrices[0].schema
    
    # Start with first matrix
    result = matrices[0].data.copy()
    
    # Merge subsequent matrices
    for matrix in matrices[1:]:
        # Get non-overlapping columns
        new_cols = [c for c in matrix.data.columns 
                    if c not in result.columns or c == schema.TIMESTAMP_COL]
        
        if len(new_cols) > 1:  # More than just timestamp
            result = pd.merge(
                result,
                matrix.data[new_cols],
                on=schema.TIMESTAMP_COL,
                how="outer"
            )
    
    # Sort by timestamp
    result = result.sort_values(schema.TIMESTAMP_COL).reset_index(drop=True)
    
    return FeatureMatrix(data=result, schema=schema)

"""
Table Schema Validation for ACM v11.0.0

Provides schema definitions and validation for SQL table writes.
Ensures data integrity before persisting to database.

Phase 1.8 Implementation
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Type, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

from core.observability import Console
from utils.version import __version__


# =============================================================================
# Schema Definition Classes
# =============================================================================

@dataclass
class ColumnSpec:
    """
    Specification for a single column.
    
    Attributes:
        python_type: Expected Python/pandas type
        nullable: Whether NULL values are allowed
        default: Default value if missing (None means no default)
    """
    python_type: Type
    nullable: bool = True
    default: Optional[Any] = None


@dataclass
class TableSchema:
    """
    Schema definition for SQL table validation.
    
    Attributes:
        required_columns: Columns that must exist (column_name -> ColumnSpec)
        optional_columns: Columns that may exist
        auto_add_columns: Columns to auto-add with defaults if missing
        key_columns: Primary/unique key columns for MERGE operations
    """
    required_columns: Dict[str, ColumnSpec] = field(default_factory=dict)
    optional_columns: Dict[str, ColumnSpec] = field(default_factory=dict)
    auto_add_columns: Dict[str, Any] = field(default_factory=dict)  # column -> default value
    key_columns: List[str] = field(default_factory=list)
    
    def validate(self, df: pd.DataFrame, table_name: str) -> List[str]:
        """
        Validate DataFrame against schema.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required columns exist
        for col, spec in self.required_columns.items():
            if col not in df.columns:
                errors.append(f"Missing required column '{col}' for {table_name}")
            elif not spec.nullable:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}' has {null_count} NULL values but is NOT NULL")
        
        return errors
    
    def apply_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply default values for missing columns.
        
        Returns:
            DataFrame with defaults applied (modifies in place)
        """
        for col, default in self.auto_add_columns.items():
            if col not in df.columns:
                df[col] = default
        
        # Apply column-level defaults
        for col, spec in self.required_columns.items():
            if spec.default is not None and col in df.columns:
                df[col] = df[col].fillna(spec.default)
        
        return df


# =============================================================================
# Schema Registry
# =============================================================================

# Common column specs
_INT_NOT_NULL = ColumnSpec(int, nullable=False)
_INT_NULLABLE = ColumnSpec(int, nullable=True)
_FLOAT_NULLABLE = ColumnSpec(float, nullable=True)
_FLOAT_NOT_NULL = ColumnSpec(float, nullable=False)
_STR_NOT_NULL = ColumnSpec(str, nullable=False)
_STR_NULLABLE = ColumnSpec(str, nullable=True)
_DATETIME_NOT_NULL = ColumnSpec(datetime, nullable=False)
_DATETIME_NULLABLE = ColumnSpec(datetime, nullable=True)


TABLE_SCHEMAS: Dict[str, TableSchema] = {
    # =========================================================================
    # Core Scoring Tables
    # =========================================================================
    "ACM_Scores_Wide": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Timestamp": _DATETIME_NOT_NULL,
        },
        optional_columns={
            "ar1_z": _FLOAT_NULLABLE,
            "pca_spe_z": _FLOAT_NULLABLE,
            "pca_t2_z": _FLOAT_NULLABLE,
            "iforest_z": _FLOAT_NULLABLE,
            "gmm_z": _FLOAT_NULLABLE,
            "omr_z": _FLOAT_NULLABLE,
            "fused_z": _FLOAT_NULLABLE,
        },
        auto_add_columns={
            "ACMVersion": __version__,
        },
        key_columns=["RunID", "EquipID", "Timestamp"],
    ),
    
    # =========================================================================
    # Health Tables
    # =========================================================================
    "ACM_HealthTimeline": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Timestamp": _DATETIME_NOT_NULL,
            "Health": _FLOAT_NOT_NULL,
        },
        optional_columns={
            "HealthZone": _STR_NULLABLE,
            "Regime": _STR_NULLABLE,
        },
        auto_add_columns={
            "ACMVersion": __version__,
        },
        key_columns=["RunID", "EquipID", "Timestamp"],
    ),
    
    "ACM_HealthForecast": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "ForecastTimestamp": _DATETIME_NOT_NULL,
        },
        optional_columns={
            "Method": _STR_NULLABLE,
            "Health": _FLOAT_NULLABLE,
            "HealthLower": _FLOAT_NULLABLE,
            "HealthUpper": _FLOAT_NULLABLE,
            "Confidence": _FLOAT_NULLABLE,
        },
        auto_add_columns={
            "ACMVersion": __version__,
        },
        key_columns=["RunID", "EquipID", "ForecastTimestamp", "Method"],
    ),
    
    # =========================================================================
    # Episode Tables
    # =========================================================================
    "ACM_Episodes": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "EpisodeID": _INT_NOT_NULL,
            "StartTime": _DATETIME_NOT_NULL,
        },
        optional_columns={
            "EndTime": _DATETIME_NULLABLE,
            "DurationMinutes": _FLOAT_NULLABLE,
            "MaxSeverity": _FLOAT_NULLABLE,
            "TopSensor": _STR_NULLABLE,
        },
        auto_add_columns={
            "ACMVersion": __version__,
        },
        key_columns=["RunID", "EquipID", "EpisodeID"],
    ),
    
    "ACM_EpisodeDiagnostics": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "StartTime": _DT_NOT_NULL,
        },
        optional_columns={
            "EpisodeID": _INT_NULLABLE,
            "EndTime": _DT_NULLABLE,
            "DurationHours": _FLOAT_NULLABLE,
            "PeakZ": _FLOAT_NULLABLE,
            "AvgZ": _FLOAT_NULLABLE,
            "Severity": _STR_NULLABLE,
            "TopSensor1": _STR_NULLABLE,
            "TopSensor2": _STR_NULLABLE,
            "TopSensor3": _STR_NULLABLE,
            "RegimeAtStart": _STR_NULLABLE,
            "AlertMode": _STR_NULLABLE,
        },
        auto_add_columns={},
        key_columns=["RunID", "EquipID", "EpisodeID"],
    ),
    
    # =========================================================================
    # RUL Tables
    # =========================================================================
    "ACM_RUL": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
        },
        optional_columns={
            "RUL_Hours": _FLOAT_NULLABLE,
            "P10_LowerBound": _FLOAT_NULLABLE,
            "P50_Median": _FLOAT_NULLABLE,
            "P90_UpperBound": _FLOAT_NULLABLE,
            "Confidence": _FLOAT_NULLABLE,
            "Method": _STR_NULLABLE,
            "TopSensor1": _STR_NULLABLE,
            "TopSensor2": _STR_NULLABLE,
            "TopSensor3": _STR_NULLABLE,
        },
        auto_add_columns={
            "ACMVersion": __version__,
        },
        key_columns=["RunID", "EquipID"],
    ),
    
    # =========================================================================
    # Regime Tables
    # =========================================================================
    "ACM_RegimeTimeline": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Timestamp": _DATETIME_NOT_NULL,
        },
        optional_columns={
            "Regime": _STR_NULLABLE,
            "RegimeID": _INT_NULLABLE,
            "Confidence": _FLOAT_NULLABLE,
        },
        auto_add_columns={},
        key_columns=["RunID", "EquipID", "Timestamp"],
    ),
    
    # =========================================================================
    # Forecast Tables
    # =========================================================================
    "ACM_FailureForecast": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
        },
        optional_columns={
            "Method": _STR_NULLABLE,
            "ThresholdUsed": _FLOAT_NULLABLE,
            "FailureProbability": _FLOAT_NULLABLE,
            "TimeToThreshold_Hours": _FLOAT_NULLABLE,
        },
        auto_add_columns={
            "ACMVersion": __version__,
        },
        key_columns=["RunID", "EquipID", "Method"],
    ),
    
    "ACM_SensorForecast": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Sensor": _STR_NOT_NULL,
        },
        optional_columns={
            "Method": _STR_NULLABLE,
            "ForecastValue": _FLOAT_NULLABLE,
            "LowerBound": _FLOAT_NULLABLE,
            "UpperBound": _FLOAT_NULLABLE,
        },
        auto_add_columns={
            "ACMVersion": __version__,
        },
        key_columns=["RunID", "EquipID", "Sensor"],
    ),
    
    # =========================================================================
    # v11.0.0 New Tables
    # =========================================================================
    "ACM_ActiveModels": TableSchema(
        required_columns={
            "EquipID": _INT_NOT_NULL,
        },
        optional_columns={
            "ActiveRegimeVersion": _INT_NULLABLE,
            "RegimeMaturityState": _STR_NULLABLE,
            "RegimePromotedAt": _DATETIME_NULLABLE,
            "ActiveThresholdVersion": _INT_NULLABLE,
            "ThresholdPromotedAt": _DATETIME_NULLABLE,
            "ActiveForecastVersion": _INT_NULLABLE,
            "ForecastPromotedAt": _DATETIME_NULLABLE,
            "LastUpdatedAt": _DATETIME_NULLABLE,
            "LastUpdatedBy": _STR_NULLABLE,
        },
        key_columns=["EquipID"],
    ),
    
    "ACM_RegimeDefinitions": TableSchema(
        required_columns={
            "EquipID": _INT_NOT_NULL,
            "RegimeVersion": _INT_NOT_NULL,
            "RegimeID": _INT_NOT_NULL,
            "RegimeName": _STR_NOT_NULL,
            "CentroidJSON": _STR_NOT_NULL,
            "FeatureColumns": _STR_NOT_NULL,
            "DataPointCount": _INT_NOT_NULL,
        },
        optional_columns={
            "SilhouetteScore": _FLOAT_NULLABLE,
            "CreatedAt": _DATETIME_NULLABLE,
            "CreatedByRunID": _STR_NULLABLE,
        },
        key_columns=["EquipID", "RegimeVersion", "RegimeID"],
    ),
    
    "ACM_DataContractValidation": TableSchema(
        required_columns={
            "RunID": _INT_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Passed": _INT_NOT_NULL,
            "RowsValidated": _INT_NOT_NULL,
            "ColumnsValidated": _INT_NOT_NULL,
        },
        optional_columns={
            "IssuesJSON": _STR_NULLABLE,
            "WarningsJSON": _STR_NULLABLE,
            "ContractSignature": _STR_NULLABLE,
            "ValidatedAt": _DATETIME_NULLABLE,
        },
        key_columns=["RunID", "EquipID"],
    ),
    
    "ACM_SeasonalPatterns": TableSchema(
        required_columns={
            "EquipID": _INT_NOT_NULL,
            "SensorName": _STR_NOT_NULL,
            "PatternType": _STR_NOT_NULL,
            "PeriodHours": _FLOAT_NOT_NULL,
            "Amplitude": _FLOAT_NOT_NULL,
        },
        optional_columns={
            "PhaseShift": _FLOAT_NULLABLE,
            "Confidence": _FLOAT_NULLABLE,
            "DetectedAt": _DATETIME_NULLABLE,
            "DetectedByRunID": _STR_NULLABLE,
        },
        key_columns=["EquipID", "SensorName", "PatternType"],
    ),
    
    "ACM_AssetProfiles": TableSchema(
        required_columns={
            "EquipID": _INT_NOT_NULL,
            "EquipType": _STR_NOT_NULL,
            "SensorNamesJSON": _STR_NOT_NULL,
            "SensorMeansJSON": _STR_NOT_NULL,
            "SensorStdsJSON": _STR_NOT_NULL,
        },
        optional_columns={
            "RegimeCount": _INT_NULLABLE,
            "TypicalHealth": _FLOAT_NULLABLE,
            "DataHours": _FLOAT_NULLABLE,
            "LastUpdatedAt": _DATETIME_NULLABLE,
            "LastUpdatedByRunID": _STR_NULLABLE,
        },
        key_columns=["EquipID"],
    ),
    
    # =========================================================================
    # Root Cause Tables (Tier 3) - Added Dec 25, 2025
    # =========================================================================
    "ACM_DetectorCorrelation": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Detector1": _STR_NOT_NULL,
            "Detector2": _STR_NOT_NULL,
            "Correlation": _FLOAT_NOT_NULL,
        },
        optional_columns={},
        key_columns=["RunID", "EquipID", "Detector1", "Detector2"],
    ),
    
    "ACM_DriftSeries": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Timestamp": _DATETIME_NOT_NULL,
            "DriftValue": _FLOAT_NOT_NULL,
        },
        optional_columns={
            "DriftState": _STR_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "Timestamp"],
    ),
    
    "ACM_SensorCorrelations": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Sensor1": _STR_NOT_NULL,
            "Sensor2": _STR_NOT_NULL,
            "Correlation": _FLOAT_NOT_NULL,
        },
        optional_columns={
            "CorrelationType": _STR_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "Sensor1", "Sensor2"],
    ),
    
    "ACM_FeatureDropLog": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "FeatureName": _STR_NOT_NULL,
            "DropReason": _STR_NOT_NULL,
        },
        optional_columns={
            "DropValue": _FLOAT_NULLABLE,
            "Threshold": _FLOAT_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "FeatureName"],
    ),
    
    # =========================================================================
    # Model Quality Tables (Tier 4) - Added Dec 25, 2025
    # =========================================================================
    "ACM_SensorNormalized_TS": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Timestamp": _DATETIME_NOT_NULL,
            "SensorName": _STR_NOT_NULL,
        },
        optional_columns={
            "RawValue": _FLOAT_NULLABLE,
            "NormalizedValue": _FLOAT_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "Timestamp", "SensorName"],
    ),
    
    "ACM_CalibrationSummary": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "DetectorType": _STR_NOT_NULL,
        },
        optional_columns={
            "CalibrationScore": _FLOAT_NULLABLE,
            "TrainR2": _FLOAT_NULLABLE,
            "MeanAbsError": _FLOAT_NULLABLE,
            "P95Error": _FLOAT_NULLABLE,
            "DatapointsUsed": _INT_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "DetectorType"],
    ),
    
    "ACM_PCA_Metrics": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "ComponentIndex": _INT_NOT_NULL,
        },
        optional_columns={
            "ExplainedVariance": _FLOAT_NULLABLE,
            "CumulativeVariance": _FLOAT_NULLABLE,
            "Eigenvalue": _FLOAT_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "ComponentIndex"],
    ),
    
    # =========================================================================
    # Advanced Analytics Tables (Tier 6) - Added Dec 25, 2025
    # =========================================================================
    "ACM_RegimeOccupancy": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "RegimeLabel": _STR_NOT_NULL,
            "DwellTimeHours": _FLOAT_NOT_NULL,
            "DwellFraction": _FLOAT_NOT_NULL,
        },
        optional_columns={
            "EntryCount": _INT_NULLABLE,
            "AvgDwellMinutes": _FLOAT_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "RegimeLabel"],
    ),
    
    "ACM_RegimeTransitions": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "FromRegime": _STR_NOT_NULL,
            "ToRegime": _STR_NOT_NULL,
            "TransitionCount": _INT_NOT_NULL,
        },
        optional_columns={
            "TransitionProbability": _FLOAT_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "FromRegime", "ToRegime"],
    ),
    
    "ACM_ContributionTimeline": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "Timestamp": _DATETIME_NOT_NULL,
            "DetectorType": _STR_NOT_NULL,
            "ContributionPct": _FLOAT_NOT_NULL,
        },
        optional_columns={},
        key_columns=["RunID", "EquipID", "Timestamp", "DetectorType"],
    ),
    
    "ACM_RegimePromotionLog": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "RegimeLabel": _STR_NOT_NULL,
            "FromState": _STR_NOT_NULL,
            "ToState": _STR_NOT_NULL,
        },
        optional_columns={
            "Reason": _STR_NULLABLE,
            "DataPointsAtPromotion": _INT_NULLABLE,
            "PromotedAt": _DATETIME_NULLABLE,
        },
        key_columns=["RunID", "EquipID", "RegimeLabel"],
    ),
    
    "ACM_DriftController": TableSchema(
        required_columns={
            "RunID": _STR_NOT_NULL,
            "EquipID": _INT_NOT_NULL,
            "ControllerState": _STR_NOT_NULL,
        },
        optional_columns={
            "Threshold": _FLOAT_NULLABLE,
            "Sensitivity": _FLOAT_NULLABLE,
            "LastDriftValue": _FLOAT_NULLABLE,
            "LastDriftTime": _DATETIME_NULLABLE,
            "ResetCount": _INT_NULLABLE,
        },
        key_columns=["RunID", "EquipID"],
    ),
}


# =============================================================================
# Validation Functions
# =============================================================================

def validate_dataframe(df: pd.DataFrame, table_name: str, 
                       raise_on_error: bool = True) -> List[str]:
    """
    Validate DataFrame against table schema.
    
    Args:
        df: DataFrame to validate
        table_name: Target SQL table name
        raise_on_error: Whether to raise exception on validation failure
        
    Returns:
        List of validation errors (empty if valid)
        
    Raises:
        ValueError: If validation fails and raise_on_error is True
    """
    if table_name not in TABLE_SCHEMAS:
        # No schema defined - allow write with warning
        Console.warn(f"No schema defined for table '{table_name}'", 
                    component="SCHEMA", table=table_name)
        return []
    
    schema = TABLE_SCHEMAS[table_name]
    errors = schema.validate(df, table_name)
    
    if errors and raise_on_error:
        error_msg = f"Schema validation failed for {table_name}: " + "; ".join(errors)
        Console.error(error_msg, component="SCHEMA", table=table_name, 
                     error_count=len(errors))
        raise ValueError(error_msg)
    
    return errors


def apply_schema_defaults(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Apply schema defaults to DataFrame.
    
    Adds missing auto-add columns and fills NULL defaults.
    
    Args:
        df: DataFrame to process
        table_name: Target SQL table name
        
    Returns:
        DataFrame with defaults applied
    """
    if table_name not in TABLE_SCHEMAS:
        return df
    
    schema = TABLE_SCHEMAS[table_name]
    return schema.apply_defaults(df)


def get_key_columns(table_name: str) -> List[str]:
    """
    Get key columns for a table (for MERGE operations).
    
    Args:
        table_name: Target SQL table name
        
    Returns:
        List of key column names
    """
    if table_name not in TABLE_SCHEMAS:
        return []
    
    return TABLE_SCHEMAS[table_name].key_columns


def has_schema(table_name: str) -> bool:
    """Check if a schema is defined for a table."""
    return table_name in TABLE_SCHEMAS


def list_validated_tables() -> List[str]:
    """Return list of tables with schema validation."""
    return list(TABLE_SCHEMAS.keys())


# =============================================================================
# Schema Registration (for dynamic schema additions)
# =============================================================================

def register_schema(table_name: str, schema: TableSchema) -> None:
    """
    Register a new table schema.
    
    Args:
        table_name: SQL table name
        schema: TableSchema definition
    """
    if table_name in TABLE_SCHEMAS:
        Console.warn(f"Overwriting existing schema for '{table_name}'", 
                    component="SCHEMA", table=table_name)
    
    TABLE_SCHEMAS[table_name] = schema
    Console.info(f"Registered schema for '{table_name}'", 
                component="SCHEMA", table=table_name,
                required_cols=len(schema.required_columns),
                key_cols=len(schema.key_columns))

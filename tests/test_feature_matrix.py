"""
Tests for core/feature_matrix.py - Standardized Feature Matrix

v11.0.0 Phase 1.6 Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from core.feature_matrix import (
    FeatureSchema,
    FeatureMatrix,
    FeatureMatrixBuilder,
    from_dataframe,
    merge_matrices,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_timestamps():
    """Generate sample timestamps."""
    return pd.date_range("2024-01-01", periods=100, freq="1h")


@pytest.fixture
def raw_df(sample_timestamps):
    """Generate raw DataFrame with sensor data."""
    n = len(sample_timestamps)
    np.random.seed(42)
    
    return pd.DataFrame({
        "Timestamp": sample_timestamps,
        "EquipID": 1,
        "temp": 50 + np.random.randn(n) * 2,
        "pressure": 100 + np.random.randn(n) * 5,
        "vibration": 0.5 + np.random.randn(n) * 0.1,
    })


@pytest.fixture
def prefixed_df(sample_timestamps):
    """DataFrame with properly prefixed columns."""
    n = len(sample_timestamps)
    np.random.seed(42)
    
    return pd.DataFrame({
        "Timestamp": sample_timestamps,
        "EquipID": 1,
        "raw_temp": 50 + np.random.randn(n) * 2,
        "raw_pressure": 100 + np.random.randn(n) * 5,
        "norm_temp": np.random.randn(n),
        "feat_temp_roll_mean": np.random.randn(n),
        "feat_pressure_lag_1": np.random.randn(n),
    })


@pytest.fixture
def schema():
    """Default schema instance."""
    return FeatureSchema()


# =============================================================================
# FeatureSchema Tests
# =============================================================================

class TestFeatureSchema:
    """Tests for FeatureSchema."""
    
    def test_default_values(self, schema):
        """Default schema values."""
        assert schema.TIMESTAMP_COL == "Timestamp"
        assert schema.EQUIP_ID_COL == "EquipID"
        assert schema.RAW_SENSOR_PREFIX == "raw_"
    
    def test_is_raw_sensor(self, schema):
        """Identify raw sensor columns."""
        assert schema.is_raw_sensor("raw_temp") is True
        assert schema.is_raw_sensor("temp") is False
        assert schema.is_raw_sensor("norm_temp") is False
    
    def test_is_normalized(self, schema):
        """Identify normalized columns."""
        assert schema.is_normalized("norm_temp") is True
        assert schema.is_normalized("raw_temp") is False
    
    def test_is_feature(self, schema):
        """Identify feature columns."""
        assert schema.is_feature("feat_temp_mean") is True
        assert schema.is_feature("temp_mean") is False
    
    def test_is_detector_output(self, schema):
        """Identify detector output columns."""
        assert schema.is_detector_output("ar1_z") is True
        assert schema.is_detector_output("pca_spe_z") is True
        assert schema.is_detector_output("iforest_z") is True
        assert schema.is_detector_output("raw_temp") is False
    
    def test_is_regime_excluded(self, schema):
        """Identify columns excluded from regime discovery."""
        # Detector outputs should be excluded
        assert schema.is_regime_excluded("ar1_z") is True
        assert schema.is_regime_excluded("fused_z") is True
        
        # Health columns should be excluded
        assert schema.is_regime_excluded("health_score") is True
        assert schema.is_regime_excluded("Health") is True
        
        # Normal columns should not be excluded
        assert schema.is_regime_excluded("raw_temp") is False
        assert schema.is_regime_excluded("feat_temp_mean") is False
    
    def test_categorize_column(self, schema):
        """Categorize columns correctly."""
        assert schema.categorize_column("Timestamp") == "metadata"
        assert schema.categorize_column("EquipID") == "metadata"
        assert schema.categorize_column("raw_temp") == "raw_sensor"
        assert schema.categorize_column("norm_temp") == "normalized"
        assert schema.categorize_column("feat_temp_mean") == "feature"
        assert schema.categorize_column("lag_temp_1") == "lag"
        assert schema.categorize_column("roll_temp_mean") == "rolling"
        assert schema.categorize_column("ar1_z") == "detector"


# =============================================================================
# FeatureMatrix Basic Tests
# =============================================================================

class TestFeatureMatrixBasic:
    """Basic tests for FeatureMatrix."""
    
    def test_create_from_prefixed_df(self, prefixed_df):
        """Create from properly prefixed DataFrame."""
        matrix = FeatureMatrix(data=prefixed_df)
        
        assert matrix.n_rows == len(prefixed_df)
        assert matrix.n_raw_sensors == 2  # raw_temp, raw_pressure
        assert matrix.n_features == 2  # feat_temp_roll_mean, feat_pressure_lag_1
    
    def test_missing_timestamp_raises(self):
        """Missing timestamp column raises error."""
        df = pd.DataFrame({"temp": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required column"):
            FeatureMatrix(data=df)
    
    def test_sensor_names(self, prefixed_df):
        """Extract sensor names."""
        matrix = FeatureMatrix(data=prefixed_df)
        
        assert "raw_temp" in matrix.sensor_names
        assert "raw_pressure" in matrix.sensor_names
    
    def test_feature_names(self, prefixed_df):
        """Extract feature names."""
        matrix = FeatureMatrix(data=prefixed_df)
        
        assert "feat_temp_roll_mean" in matrix.feature_names
        assert "feat_pressure_lag_1" in matrix.feature_names
    
    def test_repr(self, prefixed_df):
        """String representation."""
        matrix = FeatureMatrix(data=prefixed_df)
        repr_str = repr(matrix)
        
        assert "FeatureMatrix" in repr_str
        assert "rows=" in repr_str


# =============================================================================
# FeatureMatrix Column Extraction Tests
# =============================================================================

class TestFeatureMatrixExtraction:
    """Tests for column extraction methods."""
    
    def test_get_columns_by_category(self, prefixed_df):
        """Get columns by category."""
        matrix = FeatureMatrix(data=prefixed_df)
        
        raw_cols = matrix.get_columns_by_category("raw_sensor")
        assert len(raw_cols) == 2
        
        feature_cols = matrix.get_columns_by_category("feature")
        assert len(feature_cols) == 2
    
    def test_get_regime_inputs(self, sample_timestamps):
        """Get regime-suitable inputs."""
        n = len(sample_timestamps)
        np.random.seed(42)
        
        df = pd.DataFrame({
            "Timestamp": sample_timestamps,
            "raw_temp": np.random.randn(n),
            "ar1_z": np.random.randn(n),  # Should be excluded
            "pca_spe_z": np.random.randn(n),  # Should be excluded
            "health_score": np.random.randn(n),  # Should be excluded
        })
        
        matrix = FeatureMatrix(data=df)
        regime_inputs = matrix.get_regime_inputs()
        
        assert "raw_temp" in regime_inputs.columns
        assert "ar1_z" not in regime_inputs.columns
        assert "pca_spe_z" not in regime_inputs.columns
        assert "health_score" not in regime_inputs.columns
    
    def test_get_detector_inputs(self, prefixed_df):
        """Get detector-suitable inputs."""
        matrix = FeatureMatrix(data=prefixed_df)
        detector_inputs = matrix.get_detector_inputs()
        
        # Should have timestamp, sensors, and features
        assert "Timestamp" in detector_inputs.columns
        assert "raw_temp" in detector_inputs.columns
        assert "feat_temp_roll_mean" in detector_inputs.columns
    
    def test_get_numeric_columns(self, prefixed_df):
        """Get numeric columns."""
        matrix = FeatureMatrix(data=prefixed_df)
        numeric = matrix.get_numeric_columns()
        
        assert "raw_temp" in numeric
        assert "Timestamp" not in numeric  # Excluded by default


# =============================================================================
# FeatureMatrix Data Access Tests
# =============================================================================

class TestFeatureMatrixDataAccess:
    """Tests for data access methods."""
    
    def test_timestamps(self, prefixed_df, sample_timestamps):
        """Access timestamp series."""
        matrix = FeatureMatrix(data=prefixed_df)
        
        assert len(matrix.timestamps) == len(sample_timestamps)
    
    def test_equip_id(self, prefixed_df):
        """Access equipment ID."""
        matrix = FeatureMatrix(data=prefixed_df)
        
        assert matrix.equip_id == 1
    
    def test_get_time_range(self, prefixed_df, sample_timestamps):
        """Get time range."""
        matrix = FeatureMatrix(data=prefixed_df)
        start, end = matrix.get_time_range()
        
        assert start == sample_timestamps[0]
        assert end == sample_timestamps[-1]
    
    def test_slice_by_time(self, prefixed_df, sample_timestamps):
        """Slice by time range."""
        matrix = FeatureMatrix(data=prefixed_df)
        
        mid = sample_timestamps[50]
        sliced = matrix.slice_by_time(start=mid)
        
        assert len(sliced.data) == 50  # Second half


# =============================================================================
# FeatureMatrix Validation Tests
# =============================================================================

class TestFeatureMatrixValidation:
    """Tests for validation methods."""
    
    def test_validate_valid_matrix(self, prefixed_df):
        """Valid matrix passes validation."""
        matrix = FeatureMatrix(data=prefixed_df)
        issues = matrix.validate()
        
        assert len(issues) == 0
        assert matrix.is_valid() is True
    
    def test_validate_empty_data(self):
        """Empty data fails validation."""
        df = pd.DataFrame(columns=["Timestamp", "raw_temp"])
        matrix = FeatureMatrix(data=df)
        issues = matrix.validate()
        
        assert any("empty" in i.lower() for i in issues)
        assert matrix.is_valid() is False
    
    def test_validate_nan_only_column(self, sample_timestamps):
        """All-NaN column fails validation."""
        df = pd.DataFrame({
            "Timestamp": sample_timestamps,
            "raw_temp": [np.nan] * len(sample_timestamps),
        })
        matrix = FeatureMatrix(data=df)
        issues = matrix.validate()
        
        assert any("NaN" in i for i in issues)
    
    def test_summary(self, prefixed_df):
        """Generate summary."""
        matrix = FeatureMatrix(data=prefixed_df)
        summary = matrix.summary()
        
        assert summary["n_rows"] == len(prefixed_df)
        assert summary["n_raw_sensors"] == 2
        assert "column_categories" in summary


# =============================================================================
# FeatureMatrixBuilder Tests
# =============================================================================

class TestFeatureMatrixBuilder:
    """Tests for FeatureMatrixBuilder."""
    
    def test_basic_build(self, sample_timestamps):
        """Build basic matrix."""
        n = len(sample_timestamps)
        np.random.seed(42)
        
        df = pd.DataFrame({
            "temp": np.random.randn(n),
            "pressure": np.random.randn(n),
        })
        
        builder = FeatureMatrixBuilder()
        builder.set_timestamps(sample_timestamps)
        builder.set_equip_id(1)
        builder.add_raw_sensors(df, ["temp", "pressure"])
        
        matrix = builder.build()
        
        assert matrix.n_rows == n
        assert matrix.n_raw_sensors == 2
        assert matrix.equip_id == 1
    
    def test_add_features(self, sample_timestamps):
        """Add feature columns."""
        n = len(sample_timestamps)
        np.random.seed(42)
        
        sensors = pd.DataFrame({
            "temp": np.random.randn(n),
        })
        
        features = pd.DataFrame({
            "temp_mean": np.random.randn(n),
            "temp_std": np.random.randn(n),
        })
        
        builder = FeatureMatrixBuilder()
        builder.set_timestamps(sample_timestamps)
        builder.add_raw_sensors(sensors, ["temp"])
        builder.add_features(features, ["temp_mean", "temp_std"])
        
        matrix = builder.build()
        
        assert matrix.n_raw_sensors == 1
        assert matrix.n_features == 2
        assert "feat_temp_mean" in matrix.feature_names
    
    def test_reset(self, sample_timestamps):
        """Reset builder state."""
        builder = FeatureMatrixBuilder()
        builder.set_timestamps(sample_timestamps)
        builder.reset()
        
        with pytest.raises(ValueError, match="Timestamps must be set"):
            builder.build()
    
    def test_chained_building(self, sample_timestamps):
        """Chain builder methods."""
        n = len(sample_timestamps)
        df = pd.DataFrame({
            "temp": np.random.randn(n),
        })
        
        matrix = (
            FeatureMatrixBuilder()
            .set_timestamps(sample_timestamps)
            .set_equip_id(5)
            .add_raw_sensors(df, ["temp"])
            .build()
        )
        
        assert matrix.equip_id == 5


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFromDataframe:
    """Tests for from_dataframe factory function."""
    
    def test_with_auto_prefix(self, raw_df):
        """Convert with auto-prefixing."""
        matrix = from_dataframe(
            raw_df,
            sensor_cols=["temp", "pressure", "vibration"],
            auto_prefix=True
        )
        
        assert matrix.n_raw_sensors == 3
        assert "raw_temp" in matrix.sensor_names
    
    def test_without_auto_prefix(self, prefixed_df):
        """Convert without auto-prefixing."""
        matrix = from_dataframe(
            prefixed_df,
            sensor_cols=["raw_temp", "raw_pressure"],
            auto_prefix=False
        )
        
        assert matrix.n_raw_sensors == 2


class TestMergeMatrices:
    """Tests for merge_matrices function."""
    
    def test_merge_two_matrices(self, sample_timestamps):
        """Merge two matrices."""
        n = len(sample_timestamps)
        np.random.seed(42)
        
        df1 = pd.DataFrame({
            "Timestamp": sample_timestamps,
            "raw_temp": np.random.randn(n),
        })
        
        df2 = pd.DataFrame({
            "Timestamp": sample_timestamps,
            "raw_pressure": np.random.randn(n),
        })
        
        matrix1 = FeatureMatrix(data=df1)
        matrix2 = FeatureMatrix(data=df2)
        
        merged = merge_matrices(matrix1, matrix2)
        
        assert "raw_temp" in merged.data.columns
        assert "raw_pressure" in merged.data.columns
        assert merged.n_rows == n
    
    def test_merge_single(self, prefixed_df):
        """Merge single matrix returns same."""
        matrix = FeatureMatrix(data=prefixed_df)
        merged = merge_matrices(matrix)
        
        assert merged.n_rows == matrix.n_rows
    
    def test_merge_empty_raises(self):
        """Merge with no matrices raises."""
        with pytest.raises(ValueError, match="At least one matrix"):
            merge_matrices()


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_row(self):
        """Single row matrix."""
        df = pd.DataFrame({
            "Timestamp": [datetime.now()],
            "raw_temp": [50.0],
        })
        
        matrix = FeatureMatrix(data=df)
        
        assert matrix.n_rows == 1
        assert matrix.is_valid()
    
    def test_no_sensors(self, sample_timestamps):
        """Matrix with no sensor columns."""
        df = pd.DataFrame({
            "Timestamp": sample_timestamps,
            "other_col": range(len(sample_timestamps)),
        })
        
        matrix = FeatureMatrix(data=df)
        
        assert matrix.n_raw_sensors == 0
    
    def test_mixed_dtypes(self, sample_timestamps):
        """Matrix with mixed column types."""
        n = len(sample_timestamps)
        
        df = pd.DataFrame({
            "Timestamp": sample_timestamps,
            "raw_temp": np.random.randn(n),
            "category": ["A"] * n,
        })
        
        matrix = FeatureMatrix(data=df)
        numeric = matrix.get_numeric_columns()
        
        assert "raw_temp" in numeric
        assert "category" not in numeric

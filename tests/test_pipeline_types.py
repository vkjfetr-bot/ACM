# tests/test_pipeline_types.py
"""
Tests for Phase 1 pipeline types (P1.1, P1.2, P1.3).
"""
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the new types
from core.pipeline_types import (
    PipelineMode,
    DataContract,
    ValidationResult,
    SensorMeta,
    SensorValidator,
    FeatureMatrix,
)


class TestPipelineMode:
    """Tests for P1.1 - PipelineMode enum."""
    
    def test_mode_values_exist(self):
        """Verify ONLINE and OFFLINE modes exist."""
        assert PipelineMode.ONLINE is not None
        assert PipelineMode.OFFLINE is not None
        assert PipelineMode.ONLINE != PipelineMode.OFFLINE
    
    def test_from_env_batch_mode(self):
        """Test detection from environment variable."""
        # Test OFFLINE detection
        os.environ["ACM_BATCH_MODE"] = "1"
        assert PipelineMode.from_env() == PipelineMode.OFFLINE
        
        os.environ["ACM_BATCH_MODE"] = "true"
        assert PipelineMode.from_env() == PipelineMode.OFFLINE
        
        # Test ONLINE detection (default)
        os.environ["ACM_BATCH_MODE"] = "0"
        assert PipelineMode.from_env() == PipelineMode.ONLINE
        
        del os.environ["ACM_BATCH_MODE"]
        assert PipelineMode.from_env() == PipelineMode.ONLINE
    
    def test_from_config(self):
        """Test detection from config dict."""
        assert PipelineMode.from_config({"pipeline": {"mode": "online"}}) == PipelineMode.ONLINE
        assert PipelineMode.from_config({"pipeline": {"mode": "offline"}}) == PipelineMode.OFFLINE
        assert PipelineMode.from_config({}) == PipelineMode.OFFLINE  # Default
    
    def test_mode_properties(self):
        """Test mode capability properties."""
        # OFFLINE allows batch operations
        assert PipelineMode.OFFLINE.allows_batch_aggregation is True
        assert PipelineMode.OFFLINE.allows_model_refit is True
        
        # ONLINE restricts operations
        assert PipelineMode.ONLINE.allows_batch_aggregation is False
        assert PipelineMode.ONLINE.allows_model_refit is False
        
        # Latency constraints
        assert PipelineMode.ONLINE.max_latency_ms < PipelineMode.OFFLINE.max_latency_ms


class TestDataContract:
    """Tests for P1.2 - DataContract dataclass."""
    
    @pytest.fixture
    def sample_contract(self):
        """Create a sample data contract."""
        return DataContract(
            required_sensors=["Temp1", "Pressure1", "Flow1"],
            optional_sensors=["Temp2", "Vibration1"],
            timestamp_col="Timestamp",
            min_rows=50,
            max_null_fraction=0.2,
            equip_id=1,
            equip_code="TEST_EQUIP",
        )
    
    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid DataFrame that passes contract."""
        n = 100
        return pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "Temp1": np.random.randn(n) * 10 + 50,
            "Pressure1": np.random.randn(n) * 2 + 10,
            "Flow1": np.random.randn(n) * 5 + 100,
            "Temp2": np.random.randn(n) * 10 + 55,  # Optional, present
        })
    
    def test_contract_validation_pass(self, sample_contract, valid_dataframe):
        """Test that valid data passes contract validation."""
        result = sample_contract.validate(valid_dataframe)
        
        assert result.passed is True
        assert len(result.issues) == 0
        assert result.rows_validated == 100
        assert result.columns_validated == 5
    
    def test_contract_missing_required_sensors(self, sample_contract):
        """Test that missing required sensors fails validation."""
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
            "Temp1": np.random.randn(100),
            # Missing Pressure1 and Flow1
        })
        
        result = sample_contract.validate(df)
        
        assert result.passed is False
        assert any("Missing required sensors" in issue for issue in result.issues)
    
    def test_contract_missing_optional_sensors_warns(self, sample_contract, valid_dataframe):
        """Test that missing optional sensors only warns."""
        # Remove optional Vibration1 (Temp2 is present)
        result = sample_contract.validate(valid_dataframe)
        
        # Should pass but have warning about Vibration1
        assert result.passed is True
        assert any("Vibration1" in str(w) for w in result.warnings)
    
    def test_contract_insufficient_rows(self, sample_contract):
        """Test that insufficient rows fails validation."""
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "Temp1": np.random.randn(10),
            "Pressure1": np.random.randn(10),
            "Flow1": np.random.randn(10),
        })
        
        result = sample_contract.validate(df)
        
        assert result.passed is False
        assert any("Insufficient rows" in issue for issue in result.issues)
    
    def test_contract_high_null_fraction_warns(self, sample_contract):
        """Test that high null fraction warns."""
        n = 100
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "Temp1": np.random.randn(n),
            "Pressure1": np.random.randn(n),
            "Flow1": np.concatenate([np.random.randn(50), [np.nan] * 50]),  # 50% null
        })
        
        result = sample_contract.validate(df)
        
        # Should pass (issues) but warn about high nulls
        assert any("High null fraction" in str(w) for w in result.warnings)
    
    def test_contract_serialization(self, sample_contract):
        """Test contract to/from dict serialization."""
        data = sample_contract.to_dict()
        
        assert data["required_sensors"] == ["Temp1", "Pressure1", "Flow1"]
        assert data["equip_code"] == "TEST_EQUIP"
        
        # Roundtrip
        restored = DataContract.from_dict(data)
        assert restored.required_sensors == sample_contract.required_sensors
        assert restored.equip_code == sample_contract.equip_code
    
    def test_contract_signature(self, sample_contract):
        """Test that contract signature is stable."""
        sig1 = sample_contract.signature()
        sig2 = sample_contract.signature()
        
        assert sig1 == sig2
        assert len(sig1) == 12  # MD5 truncated
        
        # Different contract should have different signature
        other = DataContract(required_sensors=["Other"])
        assert other.signature() != sig1


class TestSensorValidator:
    """Tests for P1.3 - SensorValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a sensor validator."""
        contract = DataContract(
            required_sensors=["Temp1", "Pressure1"],
            min_rows=10,
        )
        return SensorValidator(contract=contract)
    
    def test_validator_empty_dataframe(self, validator):
        """Test validation of empty DataFrame."""
        result = validator.validate(pd.DataFrame())
        
        assert result.passed is False
        assert any("empty" in issue.lower() for issue in result.issues)
    
    def test_validator_duplicate_timestamps(self, validator):
        """Test detection of duplicate timestamps."""
        df = pd.DataFrame({
            "Timestamp": ["2024-01-01"] * 20,  # All duplicates
            "Temp1": np.random.randn(20),
            "Pressure1": np.random.randn(20),
        })
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        result = validator.validate(df)
        
        assert any("Duplicate timestamps" in str(w) for w in result.warnings)
    
    def test_validator_infer_sensor_type(self, validator):
        """Test sensor type inference from column names."""
        assert validator._infer_sensor_type("Temp1") == "temperature"
        assert validator._infer_sensor_type("Bearing_Temp") == "temperature"
        assert validator._infer_sensor_type("Inlet_Pressure") == "pressure"
        assert validator._infer_sensor_type("Vibration_X") == "vibration"
        assert validator._infer_sensor_type("Motor_Speed_RPM") == "speed"
        assert validator._infer_sensor_type("UnknownSensor") is None
    
    def test_filter_valid_sensors(self, validator):
        """Test filtering to valid sensors only."""
        n = 50
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "Temp1": np.random.randn(n) + 50,  # Valid
            "Pressure1": [100.0] * n,  # Constant - should be removed
            "Sensor2": np.random.randn(n),  # Valid
            "BadSensor": [np.nan] * n,  # All null - should be removed
        })
        
        filtered = validator.filter_valid_sensors(df)
        
        assert "Timestamp" in filtered.columns
        assert "Temp1" in filtered.columns
        assert "Sensor2" in filtered.columns
        assert "Pressure1" not in filtered.columns  # Constant
        assert "BadSensor" not in filtered.columns  # All null


class TestFeatureMatrix:
    """Tests for FeatureMatrix data leakage prevention."""
    
    def test_regime_inputs_clean(self):
        """Test that clean regime inputs pass validation."""
        matrix = FeatureMatrix(
            regime_features=pd.DataFrame({
                "Temp1": [1, 2, 3],
                "Pressure1": [4, 5, 6],
            }),
            source_rows=3,
        )
        
        # Should not raise
        regime_inputs = matrix.get_regime_inputs()
        assert len(regime_inputs) == 3
    
    def test_regime_inputs_detect_leakage(self):
        """Test that detector outputs in regime inputs raise error."""
        matrix = FeatureMatrix(
            regime_features=pd.DataFrame({
                "Temp1": [1, 2, 3],
                "pca_spe_z": [0.1, 0.2, 0.3],  # Forbidden!
            }),
            source_rows=3,
        )
        
        with pytest.raises(ValueError, match="Data leakage detected"):
            matrix.get_regime_inputs()
    
    def test_regime_inputs_detect_various_patterns(self):
        """Test detection of various forbidden patterns."""
        forbidden_cols = ["ar1_z", "iforest_score", "gmm_z", "omr_residual"]
        
        for col in forbidden_cols:
            matrix = FeatureMatrix(
                regime_features=pd.DataFrame({col: [1, 2, 3]}),
                source_rows=3,
            )
            with pytest.raises(ValueError):
                matrix.get_regime_inputs()
    
    def test_feature_matrix_signature(self):
        """Test signature computation."""
        matrix = FeatureMatrix(
            sensor_features=pd.DataFrame({"A": [1, 2]}),
            regime_features=pd.DataFrame({"B": [3, 4]}),
            source_rows=2,
        )
        
        sig = matrix.signature()
        assert len(sig) == 12
        
        # Same structure = same signature
        matrix2 = FeatureMatrix(
            sensor_features=pd.DataFrame({"A": [5, 6]}),  # Different values
            regime_features=pd.DataFrame({"B": [7, 8]}),
            source_rows=2,
        )
        assert matrix2.signature() == sig  # Same columns, same signature


class TestValidationResult:
    """Tests for ValidationResult."""
    
    def test_bool_conversion(self):
        """Test that ValidationResult converts to bool correctly."""
        passed = ValidationResult(passed=True)
        failed = ValidationResult(passed=False, issues=["test issue"])
        
        assert bool(passed) is True
        assert bool(failed) is False
        
        # Can use in if statements
        if passed:
            assert True
        else:
            assert False, "Should be truthy"
    
    def test_summary_output(self):
        """Test human-readable summary."""
        result = ValidationResult(
            passed=False,
            issues=["Missing sensor X"],
            warnings=["High null fraction"],
            rows_validated=100,
            columns_validated=5,
        )
        
        summary = result.summary()
        
        assert "FAILED" in summary
        assert "100 rows" in summary
        assert "Missing sensor X" in summary
        assert "High null fraction" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

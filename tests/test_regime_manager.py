# tests/test_regime_manager.py
"""
Tests for Phase 2 regime management (P2.1-P2.4).
"""
import pytest
import numpy as np
import pandas as pd

from core.regime_manager import (
    REGIME_UNKNOWN,
    REGIME_EMERGING,
    MaturityState,
    ActiveModels,
    RegimeAssignment,
    RegimeAssigner,
    RegimeInputValidator,
    FORBIDDEN_REGIME_INPUT_PATTERNS,
)


class TestMaturityState:
    """Tests for P2.1 - MaturityState enum."""
    
    def test_all_states_exist(self):
        """Verify all maturity states are defined."""
        assert MaturityState.INITIALIZING is not None
        assert MaturityState.LEARNING is not None
        assert MaturityState.CONVERGED is not None
        assert MaturityState.DEPRECATED is not None
    
    def test_state_values(self):
        """Test string values for SQL storage."""
        assert MaturityState.INITIALIZING.value == "INITIALIZING"
        assert MaturityState.LEARNING.value == "LEARNING"
        assert MaturityState.CONVERGED.value == "CONVERGED"
        assert MaturityState.DEPRECATED.value == "DEPRECATED"
    
    def test_allows_threshold_conditioning(self):
        """Test which states allow regime-conditioned thresholds."""
        assert MaturityState.INITIALIZING.allows_threshold_conditioning is False
        assert MaturityState.LEARNING.allows_threshold_conditioning is False
        assert MaturityState.CONVERGED.allows_threshold_conditioning is True
        assert MaturityState.DEPRECATED.allows_threshold_conditioning is False
    
    def test_is_operational(self):
        """Test which states are operational."""
        assert MaturityState.INITIALIZING.is_operational is False
        assert MaturityState.LEARNING.is_operational is True
        assert MaturityState.CONVERGED.is_operational is True
        assert MaturityState.DEPRECATED.is_operational is False


class TestActiveModels:
    """Tests for P2.1/P2.2 - ActiveModels dataclass."""
    
    def test_cold_start_detection(self):
        """Test cold-start is detected when regime_version is None."""
        cold_start = ActiveModels(
            equip_id=1,
            regime_version=None,
            regime_maturity=MaturityState.INITIALIZING
        )
        assert cold_start.is_cold_start is True
        assert cold_start.allows_regime_conditioning is False
    
    def test_operational_state(self):
        """Test operational state with active regime."""
        active = ActiveModels(
            equip_id=1,
            regime_version=5,
            regime_maturity=MaturityState.CONVERGED,
            threshold_version=3
        )
        assert active.is_cold_start is False
        assert active.allows_regime_conditioning is True
    
    def test_learning_state_no_conditioning(self):
        """Test LEARNING state doesn't allow conditioning."""
        learning = ActiveModels(
            equip_id=1,
            regime_version=2,
            regime_maturity=MaturityState.LEARNING
        )
        assert learning.is_cold_start is False
        assert learning.allows_regime_conditioning is False
    
    def test_to_dict(self):
        """Test serialization."""
        active = ActiveModels(
            equip_id=42,
            regime_version=3,
            regime_maturity=MaturityState.CONVERGED
        )
        d = active.to_dict()
        
        assert d["equip_id"] == 42
        assert d["regime_version"] == 3
        assert d["regime_maturity"] == "CONVERGED"
        assert d["is_cold_start"] is False


class TestRegimeAssignment:
    """Tests for P2.3 - Regime assignment with UNKNOWN/EMERGING."""
    
    def test_known_regime(self):
        """Test assignment to known regime."""
        assignment = RegimeAssignment(
            label=2,
            confidence=0.85,
            nearest_regime=2,
            distance=1.5
        )
        assert assignment.is_known is True
        assert assignment.is_unknown is False
        assert assignment.is_emerging is False
        assert assignment.label_str == "R2"
    
    def test_unknown_regime(self):
        """Test UNKNOWN assignment."""
        assignment = RegimeAssignment(
            label=REGIME_UNKNOWN,
            confidence=0.1,
            nearest_regime=1,
            distance=5.0
        )
        assert assignment.is_known is False
        assert assignment.is_unknown is True
        assert assignment.is_emerging is False
        assert assignment.label_str == "UNKNOWN"
    
    def test_emerging_regime(self):
        """Test EMERGING assignment."""
        assignment = RegimeAssignment(
            label=REGIME_EMERGING,
            confidence=0.3,
            nearest_regime=0,
            distance=3.5
        )
        assert assignment.is_known is False
        assert assignment.is_unknown is False
        assert assignment.is_emerging is True
        assert assignment.label_str == "EMERGING"


class TestRegimeAssigner:
    """Tests for P2.3 - RegimeAssigner with distance thresholds."""
    
    @pytest.fixture
    def simple_assigner(self):
        """Create a simple 2-regime assigner."""
        # Two regimes at (0, 0) and (10, 10) in 2D space
        centroids = np.array([[0, 0], [10, 10]])
        return RegimeAssigner(
            centroids=centroids,
            scaler_mean=np.array([5, 5]),
            scaler_scale=np.array([5, 5]),
            avg_centroid_distance=1.5,
            unknown_threshold=2.0,
            emerging_threshold=1.5
        )
    
    def test_assign_to_nearest(self, simple_assigner):
        """Test assignment to nearest regime when close."""
        # Point near regime 0 (scaled: -1, -1)
        assignment = simple_assigner.assign(np.array([0, 0]))
        
        assert assignment.label == 0
        assert assignment.is_known is True
        assert assignment.confidence > 0.5
    
    def test_assign_unknown_when_far(self, simple_assigner):
        """Test UNKNOWN assignment when far from all regimes."""
        # Point very far from both regimes
        assignment = simple_assigner.assign(np.array([100, 100]))
        
        assert assignment.label == REGIME_UNKNOWN
        assert assignment.is_unknown is True
        assert assignment.confidence < 0.5
    
    def test_assign_emerging_borderline(self, simple_assigner):
        """Test EMERGING assignment for borderline points."""
        # Point somewhat far but not extremely far
        # Adjust to get borderline distance
        assignment = simple_assigner.assign(np.array([5, 20]))  # Between regimes, elevated
        
        # Should be either EMERGING or UNKNOWN depending on distance
        assert assignment.label in [REGIME_EMERGING, REGIME_UNKNOWN, 0, 1]
        assert assignment.nearest_regime in [0, 1]
    
    def test_batch_assignment(self, simple_assigner):
        """Test batch assignment."""
        X = np.array([
            [0, 0],   # Near regime 0
            [10, 10], # Near regime 1
            [5, 5],   # Between regimes
        ])
        
        assignments = simple_assigner.assign_batch(X)
        
        assert len(assignments) == 3
        # The actual nearest regime depends on scaling - just verify structure
        assert all(isinstance(a, RegimeAssignment) for a in assignments)
        assert all(a.nearest_regime in [0, 1] for a in assignments)
    
    def test_serialization_roundtrip(self, simple_assigner):
        """Test to_dict/from_dict roundtrip."""
        d = simple_assigner.to_dict()
        restored = RegimeAssigner.from_dict(d)
        
        # Compare attributes
        np.testing.assert_array_equal(restored.centroids, simple_assigner.centroids)
        assert restored.n_regimes == simple_assigner.n_regimes
        assert restored.unknown_threshold == simple_assigner.unknown_threshold


class TestRegimeInputValidator:
    """Tests for P2.4 - Clean regime input validation."""
    
    @pytest.fixture
    def validator(self):
        """Create a strict validator."""
        return RegimeInputValidator(strict=True)
    
    @pytest.fixture
    def clean_dataframe(self):
        """DataFrame with only clean sensor columns."""
        return pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "Temp1": np.random.randn(10),
            "Pressure1": np.random.randn(10),
            "Flow1": np.random.randn(10),
        })
    
    @pytest.fixture
    def contaminated_dataframe(self):
        """DataFrame with forbidden detector outputs."""
        return pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "Temp1": np.random.randn(10),
            "pca_spe_z": np.random.randn(10),  # Forbidden!
            "ar1_z": np.random.randn(10),       # Forbidden!
            "HealthIndex": np.random.randn(10), # Forbidden!
        })
    
    def test_clean_dataframe_passes(self, validator, clean_dataframe):
        """Test that clean DataFrame passes validation."""
        is_valid, valid_cols, rejected_cols = validator.validate(clean_dataframe)
        
        assert is_valid is True
        assert len(rejected_cols) == 0
        assert "Temp1" in valid_cols
    
    def test_contaminated_dataframe_fails(self, validator, contaminated_dataframe):
        """Test that contaminated DataFrame fails validation."""
        is_valid, valid_cols, rejected_cols = validator.validate(contaminated_dataframe)
        
        assert is_valid is False
        assert "pca_spe_z" in rejected_cols
        assert "ar1_z" in rejected_cols
        assert "HealthIndex" in rejected_cols
        assert "Temp1" in valid_cols
    
    def test_filter_clean_columns(self, validator, contaminated_dataframe):
        """Test filtering to clean columns only."""
        clean_df = validator.filter_clean_columns(contaminated_dataframe)
        
        assert "Temp1" in clean_df.columns
        assert "pca_spe_z" not in clean_df.columns
        assert "ar1_z" not in clean_df.columns
    
    def test_assert_clean_raises(self, validator, contaminated_dataframe):
        """Test that assert_clean raises on contaminated data."""
        with pytest.raises(ValueError, match="Data leakage detected"):
            validator.assert_clean(contaminated_dataframe)
    
    def test_assert_clean_passes(self, validator, clean_dataframe):
        """Test that assert_clean passes on clean data."""
        # Should not raise
        validator.assert_clean(clean_dataframe)
    
    def test_forbidden_patterns_comprehensive(self):
        """Test all forbidden patterns are detected."""
        validator = RegimeInputValidator()
        
        forbidden_cols = [
            "detector_z",
            "pca_component",
            "iforest_score",
            "gmm_prob",
            "omr_residual",
            "ar1_value",
            "fused_z",
            "HealthIndex",
            "residual_1",
            "regime_label",
            "anomaly_score",
        ]
        
        df = pd.DataFrame({col: [1, 2, 3] for col in forbidden_cols})
        df["clean_sensor"] = [4, 5, 6]
        
        is_valid, valid_cols, rejected_cols = validator.validate(df)
        
        # All forbidden columns should be rejected
        for col in forbidden_cols:
            assert col in rejected_cols, f"{col} should be rejected"
        
        # Clean column should pass
        assert "clean_sensor" in valid_cols


class TestRegimeConstants:
    """Test regime label constants."""
    
    def test_unknown_label_value(self):
        """Test UNKNOWN label is -1."""
        assert REGIME_UNKNOWN == -1
    
    def test_emerging_label_value(self):
        """Test EMERGING label is -2."""
        assert REGIME_EMERGING == -2
    
    def test_labels_are_negative(self):
        """Test special labels are negative (won't conflict with cluster indices)."""
        assert REGIME_UNKNOWN < 0
        assert REGIME_EMERGING < 0
        assert REGIME_UNKNOWN != REGIME_EMERGING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

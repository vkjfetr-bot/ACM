"""
Tests for ACM v11.0.0 Calibrated Fusion Module.

Tests P3.4 (Calibrated Fusion), P3.5 (Fusion Quality), and P3.6 (Detector Correlation).
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from core.detector_protocol import DetectorOutput
from core.calibrated_fusion import (
    FusionWeights,
    FusionResult,
    FusionQualityMetrics,
    CalibratedFusion,
    CorrelationResult,
    DetectorCorrelationTracker,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_detector_outputs() -> dict:
    """Create sample detector outputs for testing."""
    np.random.seed(42)
    n = 100
    
    return {
        "ar1": DetectorOutput(
            timestamp=pd.Series(range(n)),
            z_score=pd.Series(np.random.randn(n)),
            raw_score=pd.Series(np.random.randn(n)),
            is_anomaly=pd.Series([False] * n),
            confidence=pd.Series([1.0] * n),
            detector_name="ar1",
        ),
        "pca_spe": DetectorOutput(
            timestamp=pd.Series(range(n)),
            z_score=pd.Series(np.random.randn(n) * 1.5),
            raw_score=pd.Series(np.random.randn(n)),
            is_anomaly=pd.Series([False] * n),
            confidence=pd.Series([1.0] * n),
            detector_name="pca_spe",
        ),
        "iforest": DetectorOutput(
            timestamp=pd.Series(range(n)),
            z_score=pd.Series(np.random.randn(n) * 0.8),
            raw_score=pd.Series(np.random.randn(n)),
            is_anomaly=pd.Series([False] * n),
            confidence=pd.Series([1.0] * n),
            detector_name="iforest",
        ),
    }


@pytest.fixture
def outputs_with_nan(sample_detector_outputs) -> dict:
    """Detector outputs with some NaN values."""
    outputs = sample_detector_outputs.copy()
    
    # Add NaN to ar1
    ar1_z = outputs["ar1"].z_score.copy()
    ar1_z.iloc[10:15] = np.nan
    outputs["ar1"] = DetectorOutput(
        timestamp=outputs["ar1"].timestamp,
        z_score=ar1_z,
        raw_score=outputs["ar1"].raw_score,
        is_anomaly=outputs["ar1"].is_anomaly,
        confidence=outputs["ar1"].confidence,
        detector_name="ar1",
    )
    
    return outputs


# =============================================================================
# FUSION WEIGHTS TESTS
# =============================================================================

class TestFusionWeights:
    """Tests for FusionWeights dataclass."""
    
    def test_weights_normalized(self):
        """Weights should be normalized to sum to 1."""
        weights = FusionWeights(
            detector_weights={"a": 2.0, "b": 3.0, "c": 5.0},
            calibration_method="test",
            calibrated_at=datetime.now(),
        )
        
        total = sum(weights.detector_weights.values())
        assert abs(total - 1.0) < 1e-10
    
    def test_get_weight_default(self):
        """get_weight returns default for missing detector."""
        weights = FusionWeights(
            detector_weights={"a": 0.5, "b": 0.5},
            calibration_method="test",
            calibrated_at=datetime.now(),
        )
        
        assert weights.get_weight("a") == 0.5
        assert weights.get_weight("unknown", default=0.1) == 0.1
    
    def test_equal_weights(self):
        """equal_weights creates balanced weights."""
        weights = FusionWeights.equal_weights(["a", "b", "c", "d"])
        
        assert len(weights.detector_weights) == 4
        for w in weights.detector_weights.values():
            assert abs(w - 0.25) < 1e-10
        assert weights.calibration_method == "equal"
    
    def test_serialization_roundtrip(self):
        """to_dict/from_dict roundtrip preserves data."""
        original = FusionWeights(
            detector_weights={"ar1": 0.3, "pca": 0.7},
            calibration_method="performance",
            calibrated_at=datetime(2024, 1, 15, 12, 0, 0),
            metadata={"source": "test"},
        )
        
        data = original.to_dict()
        restored = FusionWeights.from_dict(data)
        
        assert restored.calibration_method == original.calibration_method
        assert restored.metadata == original.metadata


# =============================================================================
# CALIBRATED FUSION TESTS
# =============================================================================

class TestCalibratedFusion:
    """Tests for CalibratedFusion class."""
    
    def test_basic_fusion(self, sample_detector_outputs):
        """Basic fusion produces valid results."""
        fusion = CalibratedFusion()
        result = fusion.fuse(sample_detector_outputs)
        
        assert isinstance(result, FusionResult)
        assert len(result.fused_z) == 100
        assert len(result.confidence) == 100
        assert len(result.contributing_detectors) == 3
    
    def test_fusion_with_weights(self, sample_detector_outputs):
        """Fusion with custom weights."""
        fusion = CalibratedFusion()
        weights = FusionWeights(
            detector_weights={"ar1": 0.5, "pca_spe": 0.3, "iforest": 0.2},
            calibration_method="custom",
            calibrated_at=datetime.now(),
        )
        
        result = fusion.fuse(sample_detector_outputs, weights=weights)
        
        assert result.weights_used is not None
        assert result.weights_used.calibration_method == "custom"
    
    def test_fusion_with_nan(self, outputs_with_nan):
        """Fusion handles NaN values gracefully."""
        fusion = CalibratedFusion()
        result = fusion.fuse(outputs_with_nan)
        
        # Should still produce valid fused scores
        assert not result.fused_z.isna().all()
        
        # Check that NaN rows have fewer contributing detectors
        # (rows 10-15 have NaN in ar1, so only 2 detectors contribute)
        n_contrib_nan_rows = result.n_contributing.iloc[10:15].mean()
        n_contrib_other = result.n_contributing.iloc[20:30].mean()
        assert n_contrib_nan_rows < n_contrib_other
    
    def test_confidence_degradation_missing_detectors(self, sample_detector_outputs):
        """Confidence decreases when detectors are missing."""
        fusion = CalibratedFusion(nan_penalty=0.2)
        fusion.set_expected_detectors(["ar1", "pca_spe", "iforest", "gmm", "omr"])
        
        result = fusion.fuse(sample_detector_outputs)
        
        # 2 detectors missing (gmm, omr) -> expect lower confidence
        assert result.mean_confidence < 1.0
        assert len(result.missing_detectors) == 2
    
    def test_disagreement_penalty(self):
        """Disagreement between detectors reduces confidence."""
        np.random.seed(42)
        n = 100
        
        # Create outputs where detectors strongly disagree
        outputs = {
            "det1": DetectorOutput(
                timestamp=pd.Series(range(n)),
                z_score=pd.Series([5.0] * n),  # High
                raw_score=pd.Series([5.0] * n),
                is_anomaly=pd.Series([True] * n),
                confidence=pd.Series([1.0] * n),
                detector_name="det1",
            ),
            "det2": DetectorOutput(
                timestamp=pd.Series(range(n)),
                z_score=pd.Series([-2.0] * n),  # Low/negative
                raw_score=pd.Series([-2.0] * n),
                is_anomaly=pd.Series([False] * n),
                confidence=pd.Series([1.0] * n),
                detector_name="det2",
            ),
        }
        
        fusion = CalibratedFusion(disagreement_penalty=0.3)
        result = fusion.fuse(outputs)
        
        # Disagreement should result in lower agreement score
        assert result.mean_agreement < 0.5
    
    def test_empty_outputs_raises(self):
        """Empty detector outputs raises ValueError."""
        fusion = CalibratedFusion()
        
        with pytest.raises(ValueError, match="No detector outputs"):
            fusion.fuse({})
    
    def test_min_detectors_raises(self):
        """Too few detectors raises ValueError."""
        fusion = CalibratedFusion(min_detectors=3)
        
        outputs = {
            "det1": DetectorOutput(
                timestamp=pd.Series([1, 2, 3]),
                z_score=pd.Series([1.0, 2.0, 3.0]),
                raw_score=pd.Series([1.0, 2.0, 3.0]),
                is_anomaly=pd.Series([False, False, True]),
                confidence=pd.Series([1.0, 1.0, 1.0]),
                detector_name="det1",
            ),
        }
        
        with pytest.raises(ValueError, match="at least 3 detectors"):
            fusion.fuse(outputs)
    
    def test_fuse_from_dataframe(self):
        """fuse_from_dataframe works with z-score columns."""
        np.random.seed(42)
        n = 50
        
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "ar1_z": np.random.randn(n),
            "pca_spe_z": np.random.randn(n) * 1.5,
            "iforest_z": np.random.randn(n) * 0.8,
        })
        
        fusion = CalibratedFusion(min_detectors=2)
        result = fusion.fuse_from_dataframe(df)
        
        assert len(result.fused_z) == n
        assert "ar1" in result.contributing_detectors
        assert "pca_spe" in result.contributing_detectors
    
    def test_quality_metrics(self, sample_detector_outputs):
        """get_quality_metrics extracts correct values."""
        fusion = CalibratedFusion()
        result = fusion.fuse(sample_detector_outputs)
        
        metrics = fusion.get_quality_metrics(result, run_id="test-123", equip_id=1)
        
        assert metrics.run_id == "test-123"
        assert metrics.equip_id == 1
        assert len(metrics.detectors_used) == 3
        assert 0 <= metrics.agreement_mean <= 1
        assert 0 <= metrics.confidence_mean <= 1


# =============================================================================
# FUSION RESULT TESTS
# =============================================================================

class TestFusionResult:
    """Tests for FusionResult dataclass."""
    
    def test_to_dataframe(self, sample_detector_outputs):
        """to_dataframe creates proper DataFrame."""
        fusion = CalibratedFusion()
        result = fusion.fuse(sample_detector_outputs)
        
        df = result.to_dataframe()
        
        assert "timestamp" in df.columns
        assert "fused_z" in df.columns
        assert "fusion_confidence" in df.columns
        assert "detector_agreement" in df.columns
        assert "n_contributing" in df.columns
    
    def test_length_validation(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length"):
            FusionResult(
                timestamp=pd.Series([1, 2, 3]),
                fused_z=pd.Series([1, 2]),  # Wrong length
                confidence=pd.Series([1, 1, 1]),
                detector_agreement=pd.Series([1, 1, 1]),
                n_contributing=pd.Series([1, 1, 1]),
                contributing_detectors=["a"],
            )
    
    def test_properties(self, sample_detector_outputs):
        """Properties compute correctly."""
        fusion = CalibratedFusion()
        result = fusion.fuse(sample_detector_outputs)
        
        assert isinstance(result.mean_confidence, float)
        assert isinstance(result.mean_agreement, float)
        assert isinstance(result.disagreement_rate, float)
        assert 0 <= result.disagreement_rate <= 1


# =============================================================================
# WEIGHT CALIBRATION TESTS
# =============================================================================

class TestWeightCalibration:
    """Tests for weight calibration methods."""
    
    def test_calibrate_from_performance(self):
        """Performance-based calibration produces valid weights."""
        historical = pd.DataFrame({
            "detector": ["ar1", "ar1", "pca", "pca", "iforest", "iforest"],
            "true_positive_rate": [0.9, 0.85, 0.95, 0.92, 0.7, 0.75],
            "false_positive_rate": [0.1, 0.12, 0.05, 0.08, 0.2, 0.18],
        })
        
        fusion = CalibratedFusion()
        weights = fusion.calibrate_from_performance(historical)
        
        assert "ar1" in weights.detector_weights
        assert "pca" in weights.detector_weights
        assert "iforest" in weights.detector_weights
        
        # PCA should have highest weight (best TPR, lowest FPR)
        assert weights.detector_weights["pca"] > weights.detector_weights["iforest"]
    
    def test_calibrate_from_correlation(self):
        """Correlation-based calibration reduces weight of correlated detectors."""
        correlations = {
            "ar1": {"pca": 0.3, "iforest": 0.2},
            "pca": {"ar1": 0.3, "iforest": 0.95},  # Highly correlated with iforest
            "iforest": {"ar1": 0.2, "pca": 0.95},
        }
        
        fusion = CalibratedFusion()
        weights = fusion.calibrate_from_correlation(correlations)
        
        # ar1 should have highest weight (lowest avg correlation)
        assert weights.detector_weights["ar1"] > weights.detector_weights["pca"]
        assert weights.detector_weights["ar1"] > weights.detector_weights["iforest"]


# =============================================================================
# DETECTOR CORRELATION TESTS
# =============================================================================

class TestDetectorCorrelationTracker:
    """Tests for DetectorCorrelationTracker."""
    
    def test_compute_correlations(self, sample_detector_outputs):
        """Compute correlations between detectors."""
        tracker = DetectorCorrelationTracker()
        results = tracker.compute_correlations(sample_detector_outputs)
        
        # Should have 3 pairs: ar1-pca_spe, ar1-iforest, pca_spe-iforest
        assert len(results) == 3
        
        for r in results:
            assert isinstance(r, CorrelationResult)
            assert -1 <= r.correlation <= 1
    
    def test_identify_redundant(self):
        """Identify redundant detectors."""
        tracker = DetectorCorrelationTracker(redundancy_threshold=0.9)
        
        results = [
            CorrelationResult("a", "b", 0.95, True),
            CorrelationResult("a", "c", 0.3, False),
            CorrelationResult("b", "c", 0.98, True),
        ]
        
        redundant = tracker.identify_redundant_detectors(results)
        
        # "b" is redundant with "a", and "c" is redundant with "b"
        assert "b" in redundant  # max("a", "b")
        assert "c" in redundant  # max("b", "c")
    
    def test_correlation_matrix(self):
        """get_correlation_matrix produces valid matrix."""
        tracker = DetectorCorrelationTracker()
        
        results = [
            CorrelationResult("ar1", "pca", 0.5, False),
            CorrelationResult("ar1", "iforest", 0.3, False),
            CorrelationResult("pca", "iforest", 0.7, False),
        ]
        
        matrix = tracker.get_correlation_matrix(results)
        
        assert matrix.shape == (3, 3)
        assert matrix.loc["ar1", "pca"] == 0.5
        assert matrix.loc["pca", "ar1"] == 0.5  # Symmetric
        assert matrix.loc["ar1", "ar1"] == 1.0  # Diagonal
    
    def test_highly_correlated_detection(self):
        """Detect highly correlated (redundant) detector pairs."""
        np.random.seed(42)
        n = 100
        base_signal = np.random.randn(n)
        
        # Create outputs where two detectors are nearly identical
        outputs = {
            "det1": DetectorOutput(
                timestamp=pd.Series(range(n)),
                z_score=pd.Series(base_signal),
                raw_score=pd.Series(base_signal),
                is_anomaly=pd.Series([False] * n),
                confidence=pd.Series([1.0] * n),
                detector_name="det1",
            ),
            "det2": DetectorOutput(
                timestamp=pd.Series(range(n)),
                z_score=pd.Series(base_signal + np.random.randn(n) * 0.01),  # Nearly identical
                raw_score=pd.Series(base_signal),
                is_anomaly=pd.Series([False] * n),
                confidence=pd.Series([1.0] * n),
                detector_name="det2",
            ),
            "det3": DetectorOutput(
                timestamp=pd.Series(range(n)),
                z_score=pd.Series(np.random.randn(n)),  # Independent
                raw_score=pd.Series(np.random.randn(n)),
                is_anomaly=pd.Series([False] * n),
                confidence=pd.Series([1.0] * n),
                detector_name="det3",
            ),
        }
        
        tracker = DetectorCorrelationTracker(redundancy_threshold=0.95)
        results = tracker.compute_correlations(outputs)
        
        # Find det1-det2 correlation
        det1_det2 = next(r for r in results if 
                        (r.detector1 == "det1" and r.detector2 == "det2") or
                        (r.detector1 == "det2" and r.detector2 == "det1"))
        
        assert det1_det2.correlation > 0.99
        assert det1_det2.is_redundant == True  # Use == for numpy bool


# =============================================================================
# QUALITY METRICS TESTS
# =============================================================================

class TestFusionQualityMetrics:
    """Tests for FusionQualityMetrics."""
    
    def test_to_dict(self):
        """to_dict produces valid SQL-ready dictionary."""
        metrics = FusionQualityMetrics(
            run_id="test-123",
            equip_id=1,
            detectors_used=["ar1", "pca", "iforest"],
            missing_detectors=["gmm"],
            agreement_mean=0.85,
            agreement_std=0.1,
            confidence_mean=0.75,
            disagreement_rate=0.05,
            weights_used={"ar1": 0.3, "pca": 0.5, "iforest": 0.2},
            calibration_method="performance",
        )
        
        data = metrics.to_dict()
        
        assert data["RunID"] == "test-123"
        assert data["EquipID"] == 1
        assert data["DetectorCount"] == 3
        assert "ar1" in data["DetectorsUsed"]
        assert data["MissingDetectors"] == "gmm"
        assert isinstance(data["WeightsUsed"], str)  # JSON string


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

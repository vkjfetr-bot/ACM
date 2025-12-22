"""
Tests for ACM v11.0.0 Detector Protocol and Baseline Normalizer.

Tests P3.2 (BaselineNormalizer) and P3.3 (DetectorProtocol).
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.detector_protocol import (
    DetectorOutput,
    DetectorMetadata,
    DetectorProtocol,
    validate_train_score_separation,
)
from core.baseline_normalizer import (
    SensorBaseline,
    BaselineStatistics,
    BaselineNormalizer,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="h")
    
    return pd.DataFrame({
        "sensor_a": np.random.randn(n_samples) * 10 + 100,
        "sensor_b": np.random.randn(n_samples) * 5 + 50,
        "sensor_c": np.random.randn(n_samples) * 2 + 20,
    }, index=dates)


@pytest.fixture
def sample_score_data() -> pd.DataFrame:
    """Create sample scoring data (different distribution)."""
    np.random.seed(123)
    n_samples = 50
    dates = pd.date_range("2024-01-10", periods=n_samples, freq="h")
    
    return pd.DataFrame({
        "sensor_a": np.random.randn(n_samples) * 10 + 110,  # Shifted mean
        "sensor_b": np.random.randn(n_samples) * 5 + 50,
        "sensor_c": np.random.randn(n_samples) * 2 + 25,   # Shifted mean
    }, index=dates)


# =============================================================================
# SENSOR BASELINE TESTS
# =============================================================================

class TestSensorBaseline:
    """Tests for SensorBaseline dataclass."""
    
    def test_compute_normal_series(self):
        """Compute baseline from normal distribution."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(1000) * 10 + 100, name="test")
        
        baseline = SensorBaseline.compute(series, "test")
        
        assert baseline.name == "test"
        assert 99 < baseline.mean < 101  # Near 100
        assert 9 < baseline.std < 11  # Near 10
        assert baseline.valid_count == 1000
        assert baseline.null_count == 0
        assert baseline.null_pct == 0.0
    
    def test_compute_with_nulls(self):
        """Compute baseline from series with NaN values."""
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, np.nan], name="test")
        
        baseline = SensorBaseline.compute(series, "test")
        
        assert baseline.valid_count == 4
        assert baseline.null_count == 2
        assert abs(baseline.null_pct - 33.33) < 1  # ~33%
        assert baseline.mean == 3.0  # (1+2+4+5)/4
    
    def test_compute_all_null(self):
        """Compute baseline from all-null series."""
        series = pd.Series([np.nan, np.nan, np.nan], name="test")
        
        baseline = SensorBaseline.compute(series, "test")
        
        assert baseline.valid_count == 0
        assert baseline.null_pct == 100.0
        assert baseline.std == 1.0  # Default to avoid division by zero
    
    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = SensorBaseline(
            name="test",
            mean=100.0, median=99.0,
            std=10.0, mad=8.0, iqr=12.0,
            min_val=50.0, max_val=150.0,
            p01=55.0, p05=60.0, p95=140.0, p99=145.0,
            valid_count=1000, null_count=0, null_pct=0.0
        )
        
        data = original.to_dict()
        restored = SensorBaseline.from_dict(data)
        
        assert restored.name == original.name
        assert restored.mean == original.mean
        assert restored.std == original.std


# =============================================================================
# BASELINE NORMALIZER TESTS
# =============================================================================

class TestBaselineNormalizer:
    """Tests for BaselineNormalizer class."""
    
    def test_fit_basic(self, sample_train_data):
        """Test basic fit operation."""
        normalizer = BaselineNormalizer()
        
        result = normalizer.fit(sample_train_data)
        
        assert result is normalizer  # Returns self
        assert normalizer.is_fitted
        assert len(normalizer.sensor_names) == 3
        assert "sensor_a" in normalizer.sensor_names
    
    def test_fit_with_column_filter(self, sample_train_data):
        """Test fit with specific columns."""
        normalizer = BaselineNormalizer()
        
        normalizer.fit(sample_train_data, sensor_cols=["sensor_a", "sensor_b"])
        
        assert len(normalizer.sensor_names) == 2
        assert "sensor_c" not in normalizer.sensor_names
    
    def test_fit_empty_raises(self):
        """Fit on empty DataFrame raises error."""
        normalizer = BaselineNormalizer()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            normalizer.fit(empty_df)
    
    def test_normalize_zscore(self, sample_train_data):
        """Test z-score normalization."""
        normalizer = BaselineNormalizer(default_method="z-score")
        normalizer.fit(sample_train_data)
        
        normalized = normalizer.normalize(sample_train_data)
        
        # Z-scored training data should have mean ~0, std ~1
        assert abs(normalized["sensor_a"].mean()) < 0.1
        assert abs(normalized["sensor_a"].std() - 1.0) < 0.1
    
    def test_normalize_robust(self, sample_train_data):
        """Test robust normalization."""
        normalizer = BaselineNormalizer(default_method="robust")
        normalizer.fit(sample_train_data)
        
        normalized = normalizer.normalize(sample_train_data, method="robust")
        
        # Should be normalized
        assert abs(normalized["sensor_a"].median()) < 0.1
    
    def test_normalize_minmax(self, sample_train_data):
        """Test min-max normalization."""
        normalizer = BaselineNormalizer()
        normalizer.fit(sample_train_data)
        
        normalized = normalizer.normalize(sample_train_data, method="minmax")
        
        # Training data should be in [0, 1] range
        assert normalized["sensor_a"].min() >= -0.01
        assert normalized["sensor_a"].max() <= 1.01
    
    def test_normalize_before_fit_raises(self, sample_train_data):
        """Normalize before fit raises error."""
        normalizer = BaselineNormalizer()
        
        with pytest.raises(RuntimeError, match="fit"):
            normalizer.normalize(sample_train_data)
    
    def test_denormalize_roundtrip(self, sample_train_data):
        """Test normalize/denormalize roundtrip."""
        normalizer = BaselineNormalizer(default_method="z-score")
        normalizer.fit(sample_train_data)
        
        normalized = normalizer.normalize(sample_train_data)
        denormalized = normalizer.denormalize(normalized)
        
        # Should be approximately equal to original
        pd.testing.assert_frame_equal(
            sample_train_data, denormalized, 
            check_exact=False, atol=1e-10
        )
    
    def test_normalize_score_data(self, sample_train_data, sample_score_data):
        """Test normalizing score data with train baseline."""
        normalizer = BaselineNormalizer()
        normalizer.fit(sample_train_data)
        
        # Normalize score data using train baseline
        normalized = normalizer.normalize(sample_score_data)
        
        # Score data has different distribution, so mean won't be 0
        # sensor_a was shifted by +10, so normalized mean should be ~1
        assert normalized["sensor_a"].mean() > 0.5
    
    def test_serialization_roundtrip(self, sample_train_data):
        """Test to_dict/from_dict roundtrip."""
        original = BaselineNormalizer(default_method="robust")
        original.fit(sample_train_data)
        
        data = original.to_dict()
        restored = BaselineNormalizer.from_dict(data)
        
        assert restored.is_fitted
        assert restored.default_method == "robust"
        assert restored.sensor_names == original.sensor_names
    
    def test_json_roundtrip(self, sample_train_data):
        """Test to_json/from_json roundtrip."""
        original = BaselineNormalizer()
        original.fit(sample_train_data)
        
        json_str = original.to_json()
        restored = BaselineNormalizer.from_json(json_str)
        
        assert restored.is_fitted
        assert set(restored.sensor_names) == set(original.sensor_names)
    
    def test_get_quality_summary(self, sample_train_data):
        """Test quality summary generation."""
        normalizer = BaselineNormalizer()
        normalizer.fit(sample_train_data)
        
        summary = normalizer.get_quality_summary()
        
        assert summary["fitted"] is True
        assert summary["n_sensors"] == 3
        assert summary["n_samples"] == 200


# =============================================================================
# DETECTOR OUTPUT TESTS
# =============================================================================

class TestDetectorOutput:
    """Tests for DetectorOutput dataclass."""
    
    def test_basic_creation(self):
        """Create basic DetectorOutput."""
        n = 100
        output = DetectorOutput(
            timestamp=pd.Series(pd.date_range("2024-01-01", periods=n, freq="h")),
            z_score=pd.Series(np.random.randn(n)),
            raw_score=pd.Series(np.random.randn(n) * 10),
            is_anomaly=pd.Series(np.random.choice([True, False], n)),
            confidence=pd.Series(np.random.rand(n)),
            detector_name="test_detector"
        )
        
        assert len(output.timestamp) == n
        assert output.detector_name == "test_detector"
    
    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="length"):
            DetectorOutput(
                timestamp=pd.Series([1, 2, 3]),
                z_score=pd.Series([1, 2]),  # Wrong length
                raw_score=pd.Series([1, 2, 3]),
                is_anomaly=pd.Series([True, False, True]),
                confidence=pd.Series([1, 1, 1]),
                detector_name="test"
            )
    
    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        n = 10
        output = DetectorOutput(
            timestamp=pd.Series(pd.date_range("2024-01-01", periods=n, freq="h")),
            z_score=pd.Series(np.ones(n)),
            raw_score=pd.Series(np.ones(n) * 2),
            is_anomaly=pd.Series([False] * n),
            confidence=pd.Series(np.ones(n)),
            detector_name="my_det"
        )
        
        df = output.to_dataframe()
        
        assert "my_det_z" in df.columns
        assert "my_det_raw" in df.columns
        assert "my_det_anomaly" in df.columns
        assert "my_det_confidence" in df.columns
    
    def test_anomaly_count(self):
        """Test anomaly count property."""
        output = DetectorOutput(
            timestamp=pd.Series([1, 2, 3, 4, 5]),
            z_score=pd.Series([1, 2, 3, 4, 5]),
            raw_score=pd.Series([1, 2, 3, 4, 5]),
            is_anomaly=pd.Series([True, False, True, True, False]),
            confidence=pd.Series([1, 1, 1, 1, 1]),
            detector_name="test"
        )
        
        assert output.anomaly_count == 3


# =============================================================================
# DETECTOR PROTOCOL TESTS
# =============================================================================

class MockDetector(DetectorProtocol):
    """Mock detector for testing protocol compliance."""
    
    def __init__(self, threshold: float = 3.0):
        self._threshold = threshold
        self._mean = None
        self._std = None
        self._fitted = False
    
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    def fit_baseline(self, X_train: pd.DataFrame) -> "MockDetector":
        self.validate_input(X_train, "train")
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        self._mean = X_train[numeric_cols].mean()
        self._std = X_train[numeric_cols].std().replace(0, 1)
        self._fitted = True
        return self
    
    def score(self, X_score: pd.DataFrame) -> DetectorOutput:
        if not self.is_fitted:
            raise RuntimeError("Must call fit_baseline() first")
        
        self.validate_input(X_score, "score")
        
        numeric_cols = X_score.select_dtypes(include=[np.number]).columns
        # Use only columns that exist in baseline
        common_cols = [c for c in numeric_cols if c in self._mean.index]
        
        z_scores = (X_score[common_cols] - self._mean[common_cols]) / self._std[common_cols]
        combined_z = z_scores.abs().mean(axis=1)
        
        return DetectorOutput(
            timestamp=pd.Series(X_score.index if isinstance(X_score.index, pd.DatetimeIndex) 
                              else range(len(X_score))),
            z_score=combined_z,
            raw_score=X_score[common_cols].mean(axis=1),
            is_anomaly=combined_z > self._threshold,
            confidence=pd.Series(1.0, index=X_score.index),
            detector_name=self.name
        )


class TestDetectorProtocol:
    """Tests for DetectorProtocol ABC."""
    
    def test_mock_detector_implements_protocol(self, sample_train_data, sample_score_data):
        """Verify mock detector properly implements protocol."""
        detector = MockDetector(threshold=2.0)
        
        assert not detector.is_fitted
        assert detector.name == "mock"
        
        detector.fit_baseline(sample_train_data)
        
        assert detector.is_fitted
        
        output = detector.score(sample_score_data)
        
        assert isinstance(output, DetectorOutput)
        assert output.detector_name == "mock"
    
    def test_score_before_fit_raises(self, sample_score_data):
        """Score before fit raises RuntimeError."""
        detector = MockDetector()
        
        with pytest.raises(RuntimeError, match="fit_baseline"):
            detector.score(sample_score_data)
    
    def test_validate_input_empty(self):
        """Empty input raises ValueError."""
        detector = MockDetector()
        
        with pytest.raises(ValueError, match="empty"):
            detector.validate_input(pd.DataFrame())
    
    def test_get_params(self, sample_train_data):
        """Test get_params method."""
        detector = MockDetector()
        detector.fit_baseline(sample_train_data)
        
        params = detector.get_params()
        
        assert params["name"] == "mock"
        assert params["is_fitted"] is True
    
    def test_deterministic_scoring(self, sample_train_data, sample_score_data):
        """Verify scoring is deterministic (same input = same output)."""
        detector = MockDetector()
        detector.fit_baseline(sample_train_data)
        
        output1 = detector.score(sample_score_data)
        output2 = detector.score(sample_score_data)
        
        # Should get identical results
        pd.testing.assert_series_equal(output1.z_score, output2.z_score)
    
    def test_train_score_separation(self, sample_train_data, sample_score_data):
        """Verify train-score separation contract."""
        detector = MockDetector()
        
        # The validation function tests the contract
        assert validate_train_score_separation(detector)


# =============================================================================
# DETECTOR METADATA TESTS
# =============================================================================

class TestDetectorMetadata:
    """Tests for DetectorMetadata dataclass."""
    
    def test_basic_creation(self):
        """Create basic metadata."""
        meta = DetectorMetadata(
            detector_name="test",
            detector_version="1.0.0",
            fitted_at=datetime.now(),
            n_training_samples=1000,
            training_start=datetime(2024, 1, 1),
            training_end=datetime(2024, 1, 31),
            feature_columns=["a", "b", "c"]
        )
        
        assert meta.detector_name == "test"
        assert meta.n_training_samples == 1000
    
    def test_serialization_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = DetectorMetadata(
            detector_name="test",
            detector_version="1.0.0",
            fitted_at=datetime(2024, 1, 15, 12, 0, 0),
            n_training_samples=1000,
            training_start=datetime(2024, 1, 1),
            training_end=datetime(2024, 1, 31),
            feature_columns=["a", "b", "c"],
            hyperparameters={"threshold": 3.0}
        )
        
        data = original.to_dict()
        restored = DetectorMetadata.from_dict(data)
        
        assert restored.detector_name == original.detector_name
        assert restored.n_training_samples == original.n_training_samples
        assert restored.feature_columns == original.feature_columns
        assert restored.hyperparameters == original.hyperparameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

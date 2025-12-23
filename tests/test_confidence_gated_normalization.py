"""Tests for P2.11: Confidence-Gated Normalization.

This module tests the ConfidenceGatedNormalizer class and the convenience function
normalize_with_confidence_gating() that provide regime-conditioned normalization
with confidence fallback.
"""

import numpy as np
import pandas as pd
import pytest

from core.fast_features import (
    ConfidenceGatedNormalizer,
    NormalizationResult,
    RegimeNormStats,
    normalize_with_confidence_gating,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sensor_cols():
    """Standard sensor columns for testing."""
    return ['temp', 'pressure', 'vibration']


@pytest.fixture
def train_data(sensor_cols):
    """Training data with two distinct regimes."""
    np.random.seed(42)
    n_samples = 200
    
    # Regime 0: Lower mean, lower variance
    regime_0 = pd.DataFrame({
        'temp': np.random.normal(100, 5, n_samples // 2),
        'pressure': np.random.normal(50, 2, n_samples // 2),
        'vibration': np.random.normal(10, 1, n_samples // 2),
    })
    
    # Regime 1: Higher mean, higher variance
    regime_1 = pd.DataFrame({
        'temp': np.random.normal(150, 10, n_samples // 2),
        'pressure': np.random.normal(80, 5, n_samples // 2),
        'vibration': np.random.normal(20, 3, n_samples // 2),
    })
    
    return pd.concat([regime_0, regime_1], ignore_index=True)


@pytest.fixture
def regime_labels():
    """Regime labels: first half = 0, second half = 1."""
    return pd.Series([0] * 100 + [1] * 100)


@pytest.fixture
def fitted_normalizer(train_data, sensor_cols, regime_labels):
    """Pre-fitted normalizer for testing."""
    normalizer = ConfidenceGatedNormalizer(confidence_threshold=0.7)
    
    # Fit global
    normalizer.fit_global(train_data, sensor_cols)
    
    # Fit regime 0
    r0_mask = regime_labels == 0
    normalizer.fit_regime(0, train_data.loc[r0_mask], sensor_cols)
    
    # Fit regime 1
    r1_mask = regime_labels == 1
    normalizer.fit_regime(1, train_data.loc[r1_mask], sensor_cols)
    
    return normalizer


# =============================================================================
# RegimeNormStats Tests
# =============================================================================

class TestRegimeNormStats:
    """Tests for RegimeNormStats dataclass."""
    
    def test_creation(self):
        """Test basic creation."""
        stats = RegimeNormStats(
            regime_label=0,
            mean=pd.Series({'a': 1.0, 'b': 2.0}),
            std=pd.Series({'a': 0.5, 'b': 0.8}),
            p05=pd.Series({'a': 0.1, 'b': 0.5}),
            p95=pd.Series({'a': 2.0, 'b': 3.5}),
            sample_count=100
        )
        
        assert stats.regime_label == 0
        assert stats.sample_count == 100
        assert stats.mean['a'] == 1.0
    
    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict."""
        original = RegimeNormStats(
            regime_label=1,
            mean=pd.Series({'temp': 100.0, 'pressure': 50.0}),
            std=pd.Series({'temp': 5.0, 'pressure': 2.0}),
            p05=pd.Series({'temp': 90.0, 'pressure': 46.0}),
            p95=pd.Series({'temp': 110.0, 'pressure': 54.0}),
            sample_count=50
        )
        
        data = original.to_dict()
        restored = RegimeNormStats.from_dict(data)
        
        assert restored.regime_label == original.regime_label
        assert restored.sample_count == original.sample_count
        pd.testing.assert_series_equal(restored.mean, original.mean)
        pd.testing.assert_series_equal(restored.std, original.std)


# =============================================================================
# ConfidenceGatedNormalizer Tests
# =============================================================================

class TestConfidenceGatedNormalizerInit:
    """Tests for normalizer initialization."""
    
    def test_default_parameters(self):
        """Test default initialization parameters."""
        normalizer = ConfidenceGatedNormalizer()
        
        assert normalizer.confidence_threshold == 0.7
        assert normalizer.min_regime_samples == 50
        assert normalizer.epsilon == 1e-10
        assert not normalizer._is_fitted
    
    def test_custom_parameters(self):
        """Test custom initialization parameters."""
        normalizer = ConfidenceGatedNormalizer(
            confidence_threshold=0.8,
            min_regime_samples=100,
            epsilon=1e-8
        )
        
        assert normalizer.confidence_threshold == 0.8
        assert normalizer.min_regime_samples == 100
        assert normalizer.epsilon == 1e-8


class TestConfidenceGatedNormalizerFit:
    """Tests for fitting the normalizer."""
    
    def test_fit_global(self, train_data, sensor_cols):
        """Test global statistics fitting."""
        normalizer = ConfidenceGatedNormalizer()
        normalizer.fit_global(train_data, sensor_cols)
        
        assert normalizer._is_fitted
        assert normalizer._global_stats is not None
        assert normalizer._global_stats.sample_count == len(train_data)
        assert set(normalizer._sensor_cols) == set(sensor_cols)
    
    def test_fit_global_chaining(self, train_data, sensor_cols):
        """Test that fit_global returns self for chaining."""
        normalizer = ConfidenceGatedNormalizer()
        result = normalizer.fit_global(train_data, sensor_cols)
        
        assert result is normalizer
    
    def test_fit_global_empty_raises(self, sensor_cols):
        """Test that fitting on empty DataFrame raises."""
        normalizer = ConfidenceGatedNormalizer()
        
        with pytest.raises(ValueError, match="Cannot fit on empty DataFrame"):
            normalizer.fit_global(pd.DataFrame(), sensor_cols)
    
    def test_fit_global_no_valid_cols_raises(self, train_data):
        """Test that fitting with no valid columns raises."""
        normalizer = ConfidenceGatedNormalizer()
        
        with pytest.raises(ValueError, match="No valid sensor columns found"):
            normalizer.fit_global(train_data, ['nonexistent_col'])
    
    def test_fit_regime_before_global_raises(self, train_data, sensor_cols):
        """Test that fit_regime before fit_global raises."""
        normalizer = ConfidenceGatedNormalizer()
        
        with pytest.raises(RuntimeError, match="Must call fit_global"):
            normalizer.fit_regime(0, train_data[:50], sensor_cols)
    
    def test_fit_regime_negative_label_raises(self, train_data, sensor_cols):
        """Test that negative regime label raises."""
        normalizer = ConfidenceGatedNormalizer()
        normalizer.fit_global(train_data, sensor_cols)
        
        with pytest.raises(ValueError, match="Regime label must be non-negative"):
            normalizer.fit_regime(-1, train_data[:50], sensor_cols)
    
    def test_fit_regime_insufficient_samples_skipped(self, train_data, sensor_cols):
        """Test that regimes with insufficient samples are skipped."""
        normalizer = ConfidenceGatedNormalizer(min_regime_samples=100)
        normalizer.fit_global(train_data, sensor_cols)
        
        # Only 20 samples - should be skipped
        normalizer.fit_regime(0, train_data[:20], sensor_cols)
        
        assert not normalizer.has_regime_stats(0)
    
    def test_fit_regime_sufficient_samples(self, train_data, sensor_cols):
        """Test that regimes with sufficient samples are fitted."""
        normalizer = ConfidenceGatedNormalizer(min_regime_samples=50)
        normalizer.fit_global(train_data, sensor_cols)
        
        # 100 samples - should be fitted
        normalizer.fit_regime(0, train_data[:100], sensor_cols)
        
        assert normalizer.has_regime_stats(0)
        assert normalizer._regime_stats[0].sample_count == 100


class TestConfidenceGatedNormalizerNormalize:
    """Tests for the normalize method."""
    
    def test_normalize_not_fitted_raises(self):
        """Test that normalize before fitting raises."""
        normalizer = ConfidenceGatedNormalizer()
        df = pd.DataFrame({'temp': [100, 105]})
        
        with pytest.raises(RuntimeError, match="Normalizer not fitted"):
            normalizer.normalize(
                df,
                regime_labels=pd.Series([0, 0]),
                confidences=pd.Series([0.9, 0.9])
            )
    
    def test_normalize_empty_df(self, fitted_normalizer):
        """Test normalizing empty DataFrame."""
        result = fitted_normalizer.normalize(
            pd.DataFrame(),
            regime_labels=pd.Series(dtype=int),
            confidences=pd.Series(dtype=float)
        )
        
        assert isinstance(result, NormalizationResult)
        assert result.z_scores.empty
    
    def test_normalize_all_low_confidence_uses_global(self, fitted_normalizer, train_data, sensor_cols):
        """Test that low confidence values use global normalization."""
        score_data = train_data[:10].copy()
        regime_labels = pd.Series([0] * 10)
        confidences = pd.Series([0.5] * 10)  # Below 0.7 threshold
        
        result = fitted_normalizer.normalize(score_data, regime_labels, confidences)
        
        # All rows should use global
        assert (result.method_used == 'global').all()
    
    def test_normalize_all_high_confidence_uses_regime(self, fitted_normalizer, train_data):
        """Test that high confidence values use regime normalization."""
        score_data = train_data[:10].copy()
        regime_labels = pd.Series([0] * 10)
        confidences = pd.Series([0.9] * 10)  # Above 0.7 threshold
        
        result = fitted_normalizer.normalize(score_data, regime_labels, confidences)
        
        # All rows should use regime_0
        assert (result.method_used == 'regime_0').all()
    
    def test_normalize_mixed_confidence(self, fitted_normalizer, train_data):
        """Test mixed confidence levels."""
        score_data = train_data[:4].copy()
        regime_labels = pd.Series([0, 0, 1, 1])
        confidences = pd.Series([0.5, 0.9, 0.3, 0.85])  # Mixed
        
        result = fitted_normalizer.normalize(score_data, regime_labels, confidences)
        
        expected_methods = pd.Series(['global', 'regime_0', 'global', 'regime_1'])
        pd.testing.assert_series_equal(
            result.method_used.reset_index(drop=True),
            expected_methods.reset_index(drop=True)
        )
    
    def test_normalize_unknown_regime_uses_global(self, fitted_normalizer, train_data):
        """Test that unknown regime labels fall back to global."""
        score_data = train_data[:3].copy()
        regime_labels = pd.Series([0, 99, -1])  # 99 and -1 are unknown
        confidences = pd.Series([0.9, 0.9, 0.9])
        
        result = fitted_normalizer.normalize(score_data, regime_labels, confidences)
        
        expected_methods = pd.Series(['regime_0', 'global', 'global'])
        pd.testing.assert_series_equal(
            result.method_used.reset_index(drop=True),
            expected_methods.reset_index(drop=True)
        )
    
    def test_normalize_produces_reasonable_z_scores(self, fitted_normalizer, sensor_cols):
        """Test that z-scores are reasonable for known data."""
        # Create data exactly at regime 0's mean
        global_stats = fitted_normalizer._global_stats
        regime_0_stats = fitted_normalizer._regime_stats[0]
        
        # Data at regime 0's mean
        score_data = pd.DataFrame({
            col: [regime_0_stats.mean[col]] for col in sensor_cols
        })
        regime_labels = pd.Series([0])
        confidences = pd.Series([0.95])
        
        result = fitted_normalizer.normalize(score_data, regime_labels, confidences)
        
        # Z-scores should be near zero for regime-specific normalization
        for col in sensor_cols:
            assert abs(result.z_scores.iloc[0][col]) < 0.1, f"Z-score for {col} should be near 0"


class TestConfidenceGatedNormalizerSerialization:
    """Tests for serialization."""
    
    def test_serialization_roundtrip(self, fitted_normalizer, train_data, sensor_cols):
        """Test full serialization/deserialization roundtrip."""
        data = fitted_normalizer.to_dict()
        restored = ConfidenceGatedNormalizer.from_dict(data)
        
        assert restored.confidence_threshold == fitted_normalizer.confidence_threshold
        assert restored.min_regime_samples == fitted_normalizer.min_regime_samples
        assert restored._is_fitted == fitted_normalizer._is_fitted
        assert restored._sensor_cols == fitted_normalizer._sensor_cols
        
        # Check regime stats preserved
        assert set(restored._regime_stats.keys()) == set(fitted_normalizer._regime_stats.keys())
    
    def test_get_stats_summary(self, fitted_normalizer):
        """Test stats summary generation."""
        summary = fitted_normalizer.get_stats_summary()
        
        assert summary['is_fitted'] is True
        assert summary['confidence_threshold'] == 0.7
        assert summary['regime_count'] == 2
        assert 0 in summary['regime_samples']
        assert 1 in summary['regime_samples']


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestNormalizeWithConfidenceGating:
    """Tests for normalize_with_confidence_gating() function."""
    
    def test_empty_df(self, sensor_cols):
        """Test with empty DataFrame."""
        global_mean = pd.Series({'temp': 100, 'pressure': 50, 'vibration': 10})
        global_std = pd.Series({'temp': 5, 'pressure': 2, 'vibration': 1})
        
        z_scores, methods = normalize_with_confidence_gating(
            pd.DataFrame(),
            sensor_cols,
            regime_labels=pd.Series(dtype=int),
            confidences=pd.Series(dtype=float),
            global_mean=global_mean,
            global_std=global_std
        )
        
        assert z_scores.empty
        assert methods.empty
    
    def test_all_global(self, sensor_cols):
        """Test with all low confidence (global normalization)."""
        df = pd.DataFrame({
            'temp': [100, 105, 110],
            'pressure': [50, 52, 48],
            'vibration': [10, 11, 9],
        })
        global_mean = pd.Series({'temp': 100, 'pressure': 50, 'vibration': 10})
        global_std = pd.Series({'temp': 5, 'pressure': 2, 'vibration': 1})
        
        regime_labels = pd.Series([0, 0, 0])
        confidences = pd.Series([0.5, 0.3, 0.6])  # All below 0.7
        
        z_scores, methods = normalize_with_confidence_gating(
            df, sensor_cols, regime_labels, confidences,
            global_mean, global_std
        )
        
        assert (methods == 'global').all()
        # First row: temp at mean should have z=0
        assert abs(z_scores.iloc[0]['temp']) < 0.01
    
    def test_mixed_regime_and_global(self, sensor_cols):
        """Test with mixed high and low confidence."""
        df = pd.DataFrame({
            'temp': [100, 100, 150, 150],
            'pressure': [50, 50, 80, 80],
            'vibration': [10, 10, 20, 20],
        })
        global_mean = pd.Series({'temp': 125, 'pressure': 65, 'vibration': 15})
        global_std = pd.Series({'temp': 25, 'pressure': 15, 'vibration': 5})
        
        regime_means = {
            0: pd.Series({'temp': 100, 'pressure': 50, 'vibration': 10}),
            1: pd.Series({'temp': 150, 'pressure': 80, 'vibration': 20}),
        }
        regime_stds = {
            0: pd.Series({'temp': 5, 'pressure': 2, 'vibration': 1}),
            1: pd.Series({'temp': 10, 'pressure': 5, 'vibration': 3}),
        }
        
        regime_labels = pd.Series([0, 0, 1, 1])
        confidences = pd.Series([0.5, 0.9, 0.3, 0.95])  # Alternating
        
        z_scores, methods = normalize_with_confidence_gating(
            df, sensor_cols, regime_labels, confidences,
            global_mean, global_std,
            regime_means=regime_means,
            regime_stds=regime_stds
        )
        
        expected_methods = pd.Series(['global', 'regime_0', 'global', 'regime_1'])
        pd.testing.assert_series_equal(methods.reset_index(drop=True), expected_methods)
        
        # Row 1 (regime_0, temp at mean): z should be ~0
        assert abs(z_scores.iloc[1]['temp']) < 0.1
        
        # Row 3 (regime_1, temp at mean): z should be ~0
        assert abs(z_scores.iloc[3]['temp']) < 0.1
    
    def test_missing_regime_stats_falls_back(self, sensor_cols):
        """Test that missing regime stats fall back to global."""
        df = pd.DataFrame({
            'temp': [100, 150],
            'pressure': [50, 80],
            'vibration': [10, 20],
        })
        global_mean = pd.Series({'temp': 125, 'pressure': 65, 'vibration': 15})
        global_std = pd.Series({'temp': 25, 'pressure': 15, 'vibration': 5})
        
        # Only regime 0 has stats
        regime_means = {0: pd.Series({'temp': 100, 'pressure': 50, 'vibration': 10})}
        regime_stds = {0: pd.Series({'temp': 5, 'pressure': 2, 'vibration': 1})}
        
        regime_labels = pd.Series([0, 1])  # Regime 1 has no stats
        confidences = pd.Series([0.9, 0.9])  # Both high confidence
        
        z_scores, methods = normalize_with_confidence_gating(
            df, sensor_cols, regime_labels, confidences,
            global_mean, global_std,
            regime_means=regime_means,
            regime_stds=regime_stds
        )
        
        # Row 0: regime_0
        # Row 1: global (regime 1 missing)
        expected_methods = pd.Series(['regime_0', 'global'])
        pd.testing.assert_series_equal(methods.reset_index(drop=True), expected_methods)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests simulating real usage patterns."""
    
    def test_full_workflow(self, sensor_cols):
        """Test complete workflow: fit, normalize, serialize, restore."""
        np.random.seed(123)
        
        # Create training data with clear regime separation
        n_samples = 300
        regime_0_data = pd.DataFrame({
            'temp': np.random.normal(100, 5, n_samples // 3),
            'pressure': np.random.normal(50, 2, n_samples // 3),
            'vibration': np.random.normal(10, 1, n_samples // 3),
        })
        regime_1_data = pd.DataFrame({
            'temp': np.random.normal(150, 8, n_samples // 3),
            'pressure': np.random.normal(80, 4, n_samples // 3),
            'vibration': np.random.normal(20, 2, n_samples // 3),
        })
        regime_2_data = pd.DataFrame({
            'temp': np.random.normal(200, 12, n_samples // 3),
            'pressure': np.random.normal(100, 6, n_samples // 3),
            'vibration': np.random.normal(30, 3, n_samples // 3),
        })
        
        train_data = pd.concat([regime_0_data, regime_1_data, regime_2_data], ignore_index=True)
        
        # Fit normalizer
        normalizer = ConfidenceGatedNormalizer(confidence_threshold=0.75)
        normalizer.fit_global(train_data, sensor_cols)
        normalizer.fit_regime(0, regime_0_data, sensor_cols)
        normalizer.fit_regime(1, regime_1_data, sensor_cols)
        normalizer.fit_regime(2, regime_2_data, sensor_cols)
        
        # Create score data
        score_data = pd.DataFrame({
            'temp': [100, 150, 200, 125],
            'pressure': [50, 80, 100, 65],
            'vibration': [10, 20, 30, 15],
        })
        regime_labels = pd.Series([0, 1, 2, 0])
        confidences = pd.Series([0.95, 0.6, 0.9, 0.4])  # Mixed
        
        # Normalize
        result = normalizer.normalize(score_data, regime_labels, confidences)
        
        # Check methods used
        assert result.method_used.iloc[0] == 'regime_0'  # High confidence
        assert result.method_used.iloc[1] == 'global'     # Low confidence
        assert result.method_used.iloc[2] == 'regime_2'  # High confidence
        assert result.method_used.iloc[3] == 'global'     # Low confidence
        
        # Serialize and restore
        serialized = normalizer.to_dict()
        restored = ConfidenceGatedNormalizer.from_dict(serialized)
        
        # Verify restored produces same results
        result2 = restored.normalize(score_data, regime_labels, confidences)
        pd.testing.assert_frame_equal(result.z_scores, result2.z_scores)
        pd.testing.assert_series_equal(result.method_used, result2.method_used)
    
    def test_edge_case_all_same_value(self, sensor_cols):
        """Test with constant sensor values (zero std edge case)."""
        # All same value - std should be epsilon
        train_data = pd.DataFrame({
            'temp': [100.0] * 100,
            'pressure': [50.0] * 100,
            'vibration': [10.0] * 100,
        })
        
        normalizer = ConfidenceGatedNormalizer()
        normalizer.fit_global(train_data, sensor_cols)
        
        # Verify std is epsilon, not zero
        for col in sensor_cols:
            assert normalizer._global_stats.std[col] > 0
        
        # Normalize slightly different values
        score_data = pd.DataFrame({
            'temp': [100.1],
            'pressure': [50.1],
            'vibration': [10.1],
        })
        
        result = normalizer.normalize(
            score_data,
            regime_labels=pd.Series([0]),
            confidences=pd.Series([0.5])
        )
        
        # Should not produce inf or NaN
        assert not result.z_scores.isna().any().any()
        assert not np.isinf(result.z_scores.values).any()

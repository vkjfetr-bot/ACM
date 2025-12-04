# tests/test_drift.py
"""
Comprehensive unit tests for drift detection logic.
Covers CUSUMDetector, compute(), and multi-feature drift detection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.drift import CUSUMDetector, compute
from core.fuse import ScoreCalibrator


class TestCUSUMDetector:
    """Tests for the CUSUMDetector class."""
    
    def test_fit_basic(self):
        """Test that fit() correctly computes mean and std."""
        detector = CUSUMDetector(threshold=2.0, drift=0.1)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        detector.fit(x)
        
        assert np.isclose(detector.mean, 3.0), f"Expected mean=3.0, got {detector.mean}"
        expected_std = np.std(x, ddof=0)  # np.nanstd uses ddof=0 by default
        assert np.isclose(detector.std, expected_std), f"Expected std={expected_std}, got {detector.std}"
    
    def test_fit_with_nans(self):
        """Test that fit() handles NaN values correctly using nanmean/nanstd."""
        detector = CUSUMDetector()
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        detector.fit(x)
        
        expected_mean = np.nanmean(x)
        expected_std = np.nanstd(x)
        assert np.isclose(detector.mean, expected_mean), f"Expected mean={expected_mean}, got {detector.mean}"
        assert np.isclose(detector.std, expected_std), f"Expected std={expected_std}, got {detector.std}"
    
    def test_fit_all_nans(self):
        """Test fit() with all NaN values - should use fallback std=1.0 and mean=0.0."""
        detector = CUSUMDetector()
        x = np.array([np.nan, np.nan, np.nan])
        detector.fit(x)
        
        # Mean should fallback to 0.0, std should fall back to 1.0
        assert detector.mean == 0.0, f"Expected mean=0.0 fallback, got {detector.mean}"
        assert detector.std == 1.0, f"Expected std=1.0 fallback, got {detector.std}"
    
    def test_fit_constant_series(self):
        """Test fit() with constant series (std=0) - should use fallback std=1.0."""
        detector = CUSUMDetector()
        x = np.array([5.0, 5.0, 5.0, 5.0])
        detector.fit(x)
        
        assert detector.mean == 5.0, f"Expected mean=5.0, got {detector.mean}"
        assert detector.std == 1.0, f"Expected std=1.0 fallback for constant series, got {detector.std}"
    
    def test_fit_very_small_std(self):
        """Test fit() with very small std - should use fallback std=1.0."""
        detector = CUSUMDetector()
        x = np.array([1.0, 1.0 + 1e-12, 1.0 - 1e-12])
        detector.fit(x)
        
        # std will be very close to zero, should fallback to 1.0
        assert detector.std == 1.0, f"Expected std=1.0 fallback for tiny std, got {detector.std}"
    
    def test_score_detects_upward_drift(self):
        """Test that score() detects an upward drift (sustained positive shift)."""
        detector = CUSUMDetector(threshold=2.0, drift=0.1)
        
        # Fit on baseline (mean=0, std=1)
        baseline = np.zeros(100)
        detector.fit(baseline)
        
        # Score on drifted data (mean shift to +1)
        drifted = np.ones(50)
        scores = detector.score(drifted)
        
        # CUSUM should accumulate and scores should increase
        assert scores[-1] > scores[0], "CUSUM should accumulate for upward drift"
        assert scores[-1] > 0, "Final score should be positive"
    
    def test_score_detects_downward_drift(self):
        """Test that score() detects a downward drift (sustained negative shift)."""
        detector = CUSUMDetector(threshold=2.0, drift=0.1)
        
        # Fit on baseline (mean=0, std=1)
        baseline = np.zeros(100)
        detector.fit(baseline)
        
        # Score on drifted data (mean shift to -1)
        drifted = -np.ones(50)
        scores = detector.score(drifted)
        
        # CUSUM should accumulate for negative drift
        assert scores[-1] > scores[0], "CUSUM should accumulate for downward drift"
        assert scores[-1] > 0, "Final score should be positive (abs value)"
    
    def test_score_stable_data(self):
        """Test that score() remains low for stable data (no drift)."""
        detector = CUSUMDetector(threshold=2.0, drift=0.1)
        
        # Fit on baseline
        baseline = np.zeros(100)
        detector.fit(baseline)
        
        # Score on same distribution
        stable = np.random.randn(100) * 0.1  # Low noise around mean
        detector.sum_pos = 0.0
        detector.sum_neg = 0.0
        scores = detector.score(stable)
        
        # Scores should remain relatively low
        assert np.max(scores) < 10, f"Max score {np.max(scores)} too high for stable data"
    
    def test_score_resets_cumsum_state(self):
        """Test that sum_pos and sum_neg are properly updated during scoring."""
        detector = CUSUMDetector(threshold=2.0, drift=0.1)
        detector.fit(np.zeros(10))
        
        # Initial state
        assert detector.sum_pos == 0.0
        assert detector.sum_neg == 0.0
        
        # Score some positive values
        detector.score(np.array([1.0, 1.0, 1.0]))
        
        # sum_pos should have accumulated, sum_neg should stay near 0
        assert detector.sum_pos > 0, "sum_pos should accumulate for positive values"
    
    def test_score_handles_nans_in_input(self):
        """Test score() behavior with NaN values in input - should replace with 0."""
        detector = CUSUMDetector(threshold=2.0, drift=0.1)
        detector.fit(np.zeros(10))
        detector.mean = 0.0
        detector.std = 1.0
        
        x = np.array([1.0, np.nan, 2.0])
        scores = detector.score(x)
        
        # After DRIFT-AUDIT-02 fix, NaN should be replaced with 0
        assert len(scores) == 3, "Output length should match input length"
        assert np.all(np.isfinite(scores)), "All scores should be finite after NaN handling"
    
    def test_threshold_and_drift_params(self):
        """Test that threshold and drift parameters are correctly stored."""
        detector = CUSUMDetector(threshold=3.5, drift=0.2)
        
        assert detector.threshold == 3.5
        assert detector.drift == 0.2


class TestComputeFunction:
    """Tests for the drift.compute() function."""
    
    def test_compute_adds_cusum_columns(self):
        """Test that compute() adds cusum_raw and cusum_z columns."""
        # Create mock score output with fused column
        frame = pd.DataFrame({
            'fused': np.random.randn(100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h')
        })
        frame = frame.set_index('timestamp')
        
        score_out = {'frame': frame}
        cfg = {}
        
        result = compute(frame, score_out, cfg)
        
        assert 'cusum_raw' in result['frame'].columns, "cusum_raw column should be added"
        assert 'cusum_z' in result['frame'].columns, "cusum_z column should be added"
    
    def test_compute_without_fused_column(self):
        """Test that compute() returns unchanged when no fused column."""
        frame = pd.DataFrame({
            'other': np.random.randn(100)
        })
        score_out = {'frame': frame}
        cfg = {}
        
        result = compute(frame, score_out, cfg)
        
        assert 'cusum_raw' not in result['frame'].columns, "cusum_raw should not be added without fused"
        assert 'cusum_z' not in result['frame'].columns, "cusum_z should not be added without fused"
    
    def test_compute_respects_config(self):
        """Test that compute() respects drift configuration parameters."""
        frame = pd.DataFrame({
            'fused': np.random.randn(100)
        })
        score_out = {'frame': frame}
        cfg = {
            'drift': {
                'cusum': {
                    'threshold': 5.0,
                    'drift': 0.5,
                    'smoothing_alpha': 0.5
                }
            }
        }
        
        result = compute(frame, score_out, cfg)
        
        # Just verify it runs without error with custom config
        assert 'cusum_z' in result['frame'].columns
    
    def test_compute_empty_config(self):
        """Test compute() with empty or None config values."""
        frame = pd.DataFrame({
            'fused': np.random.randn(50)
        })
        score_out = {'frame': frame}
        
        # Test with None nested configs
        cfg = {'drift': None}
        result = compute(frame, score_out, cfg)
        assert 'cusum_z' in result['frame'].columns
        
        # Test with empty nested configs
        cfg = {'drift': {'cusum': None}}
        result = compute(frame, score_out, cfg)
        assert 'cusum_z' in result['frame'].columns


class TestScoreCalibrator:
    """Tests for ScoreCalibrator used in drift detection."""
    
    def test_calibrator_basic(self):
        """Test basic ScoreCalibrator fit and transform."""
        cal = ScoreCalibrator(q=0.95)
        x = np.random.randn(1000)
        cal.fit(x)
        
        z = cal.transform(x)
        
        # Z-scores should have approximately zero median after transformation
        assert len(z) == len(x), "Transform should preserve length"
        assert np.all(np.isfinite(z)), "All z-scores should be finite"
    
    def test_calibrator_handles_empty(self):
        """Test ScoreCalibrator with empty input."""
        cal = ScoreCalibrator(q=0.95)
        x = np.array([])
        cal.fit(x)
        
        # Should not crash, defaults should be set
        assert cal.med == 0.0
        # scale is the key value used for z-score computation
        assert cal.scale == 1.0 or cal.scale >= 1e-3  # minimum enforced by FUSE-FIX-01
    
    def test_calibrator_handles_constant(self):
        """Test ScoreCalibrator with constant input."""
        cal = ScoreCalibrator(q=0.95)
        x = np.full(100, 5.0)
        cal.fit(x)
        
        # MAD will be 0, should fallback to std or 1.0
        z = cal.transform(x)
        
        # All z-scores should be finite (not inf from division by zero)
        assert np.all(np.isfinite(z)), "Z-scores should be finite for constant input"


class TestDriftTrendComputation:
    """Tests for _compute_drift_trend helper."""
    
    def test_drift_trend_positive_slope(self):
        """Test drift trend with clear upward trend."""
        from core.acm_main import _compute_drift_trend
        
        # Linear upward trend
        x = np.arange(50, dtype=float)
        trend = _compute_drift_trend(x, window=20)
        
        assert trend > 0, f"Expected positive trend for upward data, got {trend}"
        assert np.isclose(trend, 1.0, atol=0.01), f"Expected slope ~1.0, got {trend}"
    
    def test_drift_trend_negative_slope(self):
        """Test drift trend with clear downward trend."""
        from core.acm_main import _compute_drift_trend
        
        # Linear downward trend
        x = np.arange(50, 0, -1, dtype=float)
        trend = _compute_drift_trend(x, window=20)
        
        assert trend < 0, f"Expected negative trend for downward data, got {trend}"
    
    def test_drift_trend_flat(self):
        """Test drift trend with flat data."""
        from core.acm_main import _compute_drift_trend
        
        x = np.full(50, 5.0)
        trend = _compute_drift_trend(x, window=20)
        
        assert np.isclose(trend, 0.0, atol=0.01), f"Expected ~0 trend for flat data, got {trend}"
    
    def test_drift_trend_short_series(self):
        """Test drift trend with very short series."""
        from core.acm_main import _compute_drift_trend
        
        # Series shorter than window
        x = np.array([1.0])
        trend = _compute_drift_trend(x, window=20)
        assert trend == 0.0, "Should return 0 for single-element series"
        
        x = np.array([])
        trend = _compute_drift_trend(x, window=20)
        assert trend == 0.0, "Should return 0 for empty series"
    
    def test_drift_trend_with_nans(self):
        """Test drift trend with NaN values."""
        from core.acm_main import _compute_drift_trend
        
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        trend = _compute_drift_trend(x, window=5)
        
        # Should still compute trend ignoring NaNs
        assert np.isfinite(trend), "Trend should be finite even with NaN values"
    
    def test_drift_trend_all_nans(self):
        """Test drift trend with all NaN values."""
        from core.acm_main import _compute_drift_trend
        
        x = np.array([np.nan, np.nan, np.nan])
        trend = _compute_drift_trend(x, window=3)
        
        assert trend == 0.0, "Should return 0 for all-NaN input"


class TestRegimeVolatility:
    """Tests for _compute_regime_volatility helper."""
    
    def test_volatility_stable(self):
        """Test volatility with stable regime."""
        from core.acm_main import _compute_regime_volatility
        
        # All same regime
        labels = np.zeros(50, dtype=int)
        vol = _compute_regime_volatility(labels, window=20)
        
        assert vol == 0.0, f"Expected 0 volatility for stable regime, got {vol}"
    
    def test_volatility_high(self):
        """Test volatility with frequent transitions."""
        from core.acm_main import _compute_regime_volatility
        
        # Alternating regimes
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        vol = _compute_regime_volatility(labels, window=10)
        
        # All transitions = max volatility
        expected = 9.0 / 9.0  # 9 transitions in 10 points
        assert np.isclose(vol, expected), f"Expected volatility={expected}, got {vol}"
    
    def test_volatility_short_series(self):
        """Test volatility with very short series."""
        from core.acm_main import _compute_regime_volatility
        
        labels = np.array([0])
        vol = _compute_regime_volatility(labels, window=20)
        assert vol == 0.0, "Should return 0 for single-element series"
        
        labels = np.array([])
        vol = _compute_regime_volatility(labels, window=20)
        assert vol == 0.0, "Should return 0 for empty series"


class TestNumericalStability:
    """Tests for numerical stability edge cases."""
    
    def test_cusum_with_extreme_values(self):
        """Test CUSUM detector with extreme values."""
        detector = CUSUMDetector(threshold=2.0, drift=0.1)
        
        # Fit on normal data
        detector.fit(np.zeros(100))
        detector.mean = 0.0
        detector.std = 1.0
        
        # Score extreme values
        extreme = np.array([1e10, -1e10, 0, 1e10])
        scores = detector.score(extreme)
        
        # Should not produce inf or nan
        assert np.all(np.isfinite(scores)), "Scores should be finite for extreme values"
    
    def test_cusum_with_inf_values(self):
        """Test CUSUM detector with inf values."""
        detector = CUSUMDetector()
        
        # Fit with inf should handle gracefully
        x = np.array([1.0, np.inf, 3.0])
        detector.fit(x)
        
        # std might be inf, should fallback to 1.0
        # Note: np.nanstd doesn't exclude inf, so this tests the isfinite check
        if not np.isfinite(detector.std):
            assert detector.std == 1.0
    
    def test_calibrator_z_score_clipping(self):
        """Test that z-scores are properly clipped."""
        cal = ScoreCalibrator(q=0.95)
        
        # Normal data
        x = np.random.randn(1000)
        cal.fit(x)
        
        # Transform with extreme outlier
        extreme = np.array([100.0])
        z = cal.transform(extreme)
        
        # Should be clipped (default clip_z is 8.0, then hard limit to 10)
        assert z[0] <= 10.0, f"Z-score should be clipped, got {z[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

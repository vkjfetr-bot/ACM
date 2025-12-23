"""
Tests for core/confidence_model.py - Unified Confidence Model

v11.0.0 Phase 4.6 Tests
"""

import pytest
import numpy as np

from core.confidence_model import (
    ConfidenceSignals,
    CombinedConfidence,
    ConfidenceWeights,
    ConfidenceModel,
    ConfidenceTracker,
    combine_confidence_scores,
)


# =============================================================================
# ConfidenceSignals Tests
# =============================================================================

class TestConfidenceSignals:
    """Tests for ConfidenceSignals dataclass."""
    
    def test_default_values(self):
        """Default signals have moderate confidence."""
        signals = ConfidenceSignals()
        
        assert signals.regime_confidence == 0.5
        assert signals.detector_agreement == 0.5
        assert signals.data_quality == 0.5
        assert signals.model_maturity == 0.5
        assert signals.drift_indicator == 1.0
        assert signals.data_recency == 1.0
    
    def test_custom_values(self):
        """Custom signal values."""
        signals = ConfidenceSignals(
            regime_confidence=0.9,
            detector_agreement=0.8,
            data_quality=0.95,
            model_maturity=0.7,
            drift_indicator=1.0,
            data_recency=0.9
        )
        
        assert signals.regime_confidence == 0.9
        assert signals.data_quality == 0.95
    
    def test_to_dict(self):
        """Convert to dictionary."""
        signals = ConfidenceSignals(regime_confidence=0.85)
        d = signals.to_dict()
        
        assert d["regime_confidence"] == 0.85
        assert "detector_agreement" in d
        assert len(d) == 6
    
    def test_from_dict(self):
        """Create from dictionary."""
        d = {
            "regime_confidence": 0.9,
            "detector_agreement": 0.8,
        }
        signals = ConfidenceSignals.from_dict(d)
        
        assert signals.regime_confidence == 0.9
        assert signals.detector_agreement == 0.8
        # Defaults for missing keys
        assert signals.data_quality == 0.5
    
    def test_default_healthy(self):
        """Factory for healthy operation."""
        signals = ConfidenceSignals.default_healthy()
        
        assert signals.regime_confidence == 0.9
        assert signals.detector_agreement == 0.9
        assert signals.drift_indicator == 1.0
    
    def test_default_degraded(self):
        """Factory for degraded operation."""
        signals = ConfidenceSignals.default_degraded()
        
        assert signals.regime_confidence < 0.7
        assert signals.detector_agreement < 0.7
    
    def test_min_signal(self):
        """Find minimum signal."""
        signals = ConfidenceSignals(
            regime_confidence=0.9,
            detector_agreement=0.8,
            data_quality=0.3,  # Lowest
            model_maturity=0.7,
            drift_indicator=1.0,
            data_recency=0.9
        )
        
        name, value = signals.min_signal()
        assert name == "data_quality"
        assert value == 0.3
    
    def test_max_signal(self):
        """Find maximum signal."""
        signals = ConfidenceSignals(
            regime_confidence=0.5,
            detector_agreement=0.6,
            data_quality=0.7,
            model_maturity=0.5,
            drift_indicator=1.0,  # Highest
            data_recency=0.8
        )
        
        name, value = signals.max_signal()
        assert name == "drift_indicator"
        assert value == 1.0


# =============================================================================
# CombinedConfidence Tests
# =============================================================================

class TestCombinedConfidence:
    """Tests for CombinedConfidence dataclass."""
    
    def test_level_high(self):
        """Level = HIGH for confidence >= 0.8."""
        result = CombinedConfidence(
            confidence=0.85,
            signals=ConfidenceSignals(),
            limiting_factor="regime_confidence",
            limiting_value=0.5,
            is_trustworthy=True
        )
        assert result.level == "HIGH"
    
    def test_level_medium(self):
        """Level = MEDIUM for 0.5 <= confidence < 0.8."""
        result = CombinedConfidence(
            confidence=0.65,
            signals=ConfidenceSignals(),
            limiting_factor="regime_confidence",
            limiting_value=0.5,
            is_trustworthy=True
        )
        assert result.level == "MEDIUM"
    
    def test_level_low(self):
        """Level = LOW for confidence < 0.5."""
        result = CombinedConfidence(
            confidence=0.35,
            signals=ConfidenceSignals(),
            limiting_factor="data_quality",
            limiting_value=0.2,
            is_trustworthy=False
        )
        assert result.level == "LOW"
    
    def test_to_dict(self):
        """Convert to dictionary."""
        result = CombinedConfidence(
            confidence=0.75,
            signals=ConfidenceSignals(regime_confidence=0.9),
            limiting_factor="model_maturity",
            limiting_value=0.4,
            is_trustworthy=True
        )
        
        d = result.to_dict()
        
        assert d["Confidence"] == 0.75
        assert d["ConfidenceLevel"] == "MEDIUM"
        assert d["LimitingFactor"] == "model_maturity"
        assert d["LimitingValue"] == 0.4
        assert d["IsTrustworthy"] is True
        assert d["RegimeConfidence"] == 0.9
    
    def test_explain(self):
        """Generate explanation text."""
        result = CombinedConfidence(
            confidence=0.70,
            signals=ConfidenceSignals(),
            limiting_factor="data_quality",
            limiting_value=0.3,
            is_trustworthy=True
        )
        
        explanation = result.explain()
        
        assert "0.70" in explanation
        assert "MEDIUM" in explanation
        assert "data_quality" in explanation


# =============================================================================
# ConfidenceWeights Tests
# =============================================================================

class TestConfidenceWeights:
    """Tests for ConfidenceWeights dataclass."""
    
    def test_default_weights_sum_to_one(self):
        """Default weights sum to 1.0."""
        weights = ConfidenceWeights()
        total = (
            weights.regime_confidence +
            weights.detector_agreement +
            weights.data_quality +
            weights.model_maturity +
            weights.drift_indicator +
            weights.data_recency
        )
        assert abs(total - 1.0) < 0.01
    
    def test_custom_weights_normalized(self):
        """Custom weights are normalized to sum to 1.0."""
        weights = ConfidenceWeights(
            regime_confidence=0.5,
            detector_agreement=0.5,
            data_quality=0.5,
            model_maturity=0.5,
            drift_indicator=0.5,
            data_recency=0.5
        )
        # Should be normalized: 3.0 -> 1.0
        total = (
            weights.regime_confidence +
            weights.detector_agreement +
            weights.data_quality +
            weights.model_maturity +
            weights.drift_indicator +
            weights.data_recency
        )
        assert abs(total - 1.0) < 0.01
    
    def test_to_dict(self):
        """Convert to dictionary."""
        weights = ConfidenceWeights()
        d = weights.to_dict()
        
        assert "regime_confidence" in d
        assert len(d) == 6


# =============================================================================
# ConfidenceModel Tests
# =============================================================================

class TestConfidenceModel:
    """Tests for ConfidenceModel class."""
    
    @pytest.fixture
    def model(self):
        return ConfidenceModel()
    
    def test_compute_perfect_signals(self, model):
        """Perfect signals produce high confidence."""
        signals = ConfidenceSignals(
            regime_confidence=1.0,
            detector_agreement=1.0,
            data_quality=1.0,
            model_maturity=1.0,
            drift_indicator=1.0,
            data_recency=1.0
        )
        
        result = model.compute(signals)
        
        assert result.confidence >= 0.95
        assert result.is_trustworthy is True
        assert result.level == "HIGH"
    
    def test_compute_poor_signals(self, model):
        """Poor signals produce low confidence."""
        signals = ConfidenceSignals(
            regime_confidence=0.2,
            detector_agreement=0.3,
            data_quality=0.2,
            model_maturity=0.1,
            drift_indicator=0.3,
            data_recency=0.5
        )
        
        result = model.compute(signals)
        
        assert result.confidence < 0.5
        assert result.is_trustworthy is False
        assert result.level == "LOW"
    
    def test_compute_mixed_signals(self, model):
        """Mixed signals produce medium confidence."""
        signals = ConfidenceSignals(
            regime_confidence=0.9,
            detector_agreement=0.8,
            data_quality=0.5,  # Low
            model_maturity=0.7,
            drift_indicator=1.0,
            data_recency=0.9
        )
        
        result = model.compute(signals)
        
        # Should be between low and high
        assert 0.5 < result.confidence < 0.9
        assert result.limiting_factor == "data_quality"
    
    def test_critical_signal_cap(self, model):
        """Very low signal caps overall confidence."""
        signals = ConfidenceSignals(
            regime_confidence=0.9,
            detector_agreement=0.9,
            data_quality=0.15,  # Below critical threshold (0.3)
            model_maturity=0.9,
            drift_indicator=1.0,
            data_recency=0.9
        )
        
        result = model.compute(signals)
        
        # Should be capped at 0.5
        assert result.confidence <= 0.5
        assert result.limiting_factor == "data_quality"
    
    def test_custom_thresholds(self):
        """Custom thresholds work correctly."""
        model = ConfidenceModel(
            trustworthy_threshold=0.8,
            critical_signal_threshold=0.5,
            critical_signal_cap=0.6
        )
        
        # All signals at 0.7 - above new critical threshold
        signals = ConfidenceSignals(
            regime_confidence=0.7,
            detector_agreement=0.7,
            data_quality=0.7,
            model_maturity=0.7,
            drift_indicator=0.7,
            data_recency=0.7
        )
        
        result = model.compute(signals)
        
        assert result.confidence == pytest.approx(0.7, abs=0.01)
        assert result.is_trustworthy is False  # Below 0.8 threshold
    
    def test_from_run_context_unknown_regime(self, model):
        """Unknown regime produces zero regime confidence."""
        result = model.from_run_context(
            regime_label=-1,  # Unknown
            regime_stability=0.8,
            detector_z_scores={"ar1": 3.0, "pca": 3.0},
            sensor_validity={"temp": True, "pressure": True},
            model_age_hours=500,
            drift_detected=False,
            data_age_minutes=2.0
        )
        
        # Regime confidence should be 0
        assert result.signals.regime_confidence == 0.0
        assert result.limiting_factor == "regime_confidence"
    
    def test_from_run_context_drift_detected(self, model):
        """Drift detection reduces confidence."""
        no_drift = model.from_run_context(
            regime_label=1,
            regime_stability=0.9,
            drift_detected=False,
            model_age_hours=1000,  # Mature model so not capped
        )
        
        with_drift = model.from_run_context(
            regime_label=1,
            regime_stability=0.9,
            drift_detected=True,
            model_age_hours=1000,  # Mature model so not capped
        )
        
        assert with_drift.confidence < no_drift.confidence
        assert with_drift.signals.drift_indicator == 0.3
    
    def test_from_run_context_stale_data(self, model):
        """Stale data reduces confidence."""
        fresh = model.from_run_context(
            regime_label=1,
            regime_stability=0.9,
            data_age_minutes=2.0,
        )
        
        stale = model.from_run_context(
            regime_label=1,
            regime_stability=0.9,
            data_age_minutes=60.0,  # 1 hour old
        )
        
        assert stale.signals.data_recency < fresh.signals.data_recency
        assert stale.signals.data_recency == 0.0


class TestConfidenceModelHelpers:
    """Tests for helper methods in ConfidenceModel."""
    
    @pytest.fixture
    def model(self):
        return ConfidenceModel()
    
    def test_detector_agreement_single(self, model):
        """Single detector = perfect agreement."""
        agreement = model._compute_detector_agreement({"ar1": 5.0})
        assert agreement == 1.0
    
    def test_detector_agreement_same_values(self, model):
        """Same z-scores = perfect agreement."""
        agreement = model._compute_detector_agreement({
            "ar1": 3.0, "pca": 3.0, "iforest": 3.0
        })
        assert agreement == 1.0
    
    def test_detector_agreement_high_variance(self, model):
        """High variance = low agreement."""
        agreement = model._compute_detector_agreement({
            "ar1": 1.0, "pca": 10.0, "iforest": 5.0
        })
        assert agreement < 0.7
    
    def test_data_quality_all_valid(self, model):
        """All valid sensors = perfect quality."""
        quality = model._compute_data_quality({
            "temp": True, "pressure": True, "vibration": True
        })
        assert quality == 1.0
    
    def test_data_quality_half_valid(self, model):
        """Half valid = 0.5 quality."""
        quality = model._compute_data_quality({
            "temp": True, "pressure": False
        })
        assert quality == 0.5
    
    def test_data_quality_none(self, model):
        """No sensor info = moderate quality."""
        quality = model._compute_data_quality(None)
        assert quality == 0.5
    
    def test_model_maturity_new(self, model):
        """New model = low maturity."""
        maturity = model._compute_model_maturity(0.0)
        assert maturity == 0.0
    
    def test_model_maturity_mature(self, model):
        """30+ days = full maturity."""
        maturity = model._compute_model_maturity(720.0)  # 30 days
        assert maturity == 1.0
    
    def test_model_maturity_partial(self, model):
        """Partial maturity."""
        maturity = model._compute_model_maturity(360.0)  # 15 days
        assert maturity == 0.5
    
    def test_data_recency_fresh(self, model):
        """Fresh data = high recency."""
        recency = model._compute_data_recency(2.0)
        assert recency == 1.0
    
    def test_data_recency_stale(self, model):
        """Stale data = low recency."""
        recency = model._compute_data_recency(60.0)
        assert recency == 0.0
    
    def test_data_recency_moderate(self, model):
        """Moderate age = moderate recency."""
        recency = model._compute_data_recency(30.0)
        assert 0.3 < recency < 0.7


# =============================================================================
# ConfidenceTracker Tests
# =============================================================================

class TestConfidenceTracker:
    """Tests for ConfidenceTracker class."""
    
    def _make_confidence(self, value: float, limiting: str = "data_quality") -> CombinedConfidence:
        """Helper to create CombinedConfidence."""
        return CombinedConfidence(
            confidence=value,
            signals=ConfidenceSignals(),
            limiting_factor=limiting,
            limiting_value=value,
            is_trustworthy=value >= 0.6
        )
    
    def test_empty_tracker(self):
        """Empty tracker returns zeros."""
        tracker = ConfidenceTracker()
        
        assert tracker.mean_confidence == 0.0
        assert tracker.min_confidence == 0.0
        assert tracker.max_confidence == 0.0
    
    def test_add_values(self):
        """Add and track values."""
        tracker = ConfidenceTracker()
        
        tracker.add(self._make_confidence(0.7))
        tracker.add(self._make_confidence(0.8))
        tracker.add(self._make_confidence(0.6))
        
        assert tracker.mean_confidence == pytest.approx(0.7, abs=0.01)
        assert tracker.min_confidence == 0.6
        assert tracker.max_confidence == 0.8
    
    def test_window_size(self):
        """Tracker respects window size."""
        tracker = ConfidenceTracker(window_size=5)
        
        # Add more than window size
        for i in range(10):
            tracker.add(self._make_confidence(0.5))
        
        # Should only have 5 values
        assert len(tracker._history) == 5
    
    def test_trend_stable(self):
        """Stable trend detection."""
        tracker = ConfidenceTracker()
        
        # All same value
        for _ in range(20):
            tracker.add(self._make_confidence(0.7))
        
        assert tracker.trend == "STABLE"
    
    def test_trend_improving(self):
        """Improving trend detection."""
        tracker = ConfidenceTracker()
        
        # First half low, second half high
        for i in range(20):
            value = 0.5 if i < 10 else 0.8
            tracker.add(self._make_confidence(value))
        
        assert tracker.trend == "IMPROVING"
    
    def test_trend_degrading(self):
        """Degrading trend detection."""
        tracker = ConfidenceTracker()
        
        # First half high, second half low
        for i in range(20):
            value = 0.8 if i < 10 else 0.5
            tracker.add(self._make_confidence(value))
        
        assert tracker.trend == "DEGRADING"
    
    def test_most_common_limiting_factor(self):
        """Find most common limiting factor."""
        tracker = ConfidenceTracker()
        
        tracker.add(self._make_confidence(0.7, "data_quality"))
        tracker.add(self._make_confidence(0.6, "data_quality"))
        tracker.add(self._make_confidence(0.5, "data_quality"))
        tracker.add(self._make_confidence(0.8, "regime_confidence"))
        
        assert tracker.most_common_limiting_factor() == "data_quality"
    
    def test_clear(self):
        """Clear history."""
        tracker = ConfidenceTracker()
        tracker.add(self._make_confidence(0.7))
        tracker.add(self._make_confidence(0.8))
        
        tracker.clear()
        
        assert len(tracker._history) == 0
        assert tracker.mean_confidence == 0.0
    
    def test_summary(self):
        """Generate summary."""
        tracker = ConfidenceTracker()
        
        for i in range(15):
            tracker.add(self._make_confidence(0.6 + i * 0.02, "data_quality"))
        
        summary = tracker.summary()
        
        assert summary["n_observations"] == 15
        assert "mean_confidence" in summary
        assert "trend" in summary
        assert summary["most_common_limiting_factor"] == "data_quality"


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestCombineConfidenceScores:
    """Tests for combine_confidence_scores function."""
    
    def test_equal_weighting(self):
        """Default equal weighting."""
        combined = combine_confidence_scores(
            health_confidence=0.8,
            episode_confidence=0.8,
            rul_confidence=0.8,
        )
        
        assert combined == pytest.approx(0.8, abs=0.01)
    
    def test_custom_weights(self):
        """Custom weights."""
        combined = combine_confidence_scores(
            health_confidence=1.0,
            episode_confidence=0.0,
            rul_confidence=0.0,
            weights={"health": 1.0, "episode": 0.0, "rul": 0.0}
        )
        
        assert combined == 1.0
    
    def test_clipping(self):
        """Result is clipped to [0, 1]."""
        combined = combine_confidence_scores(
            health_confidence=1.5,  # Invalid but test clipping
            episode_confidence=1.5,
            rul_confidence=1.5,
        )
        
        assert combined <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestConfidenceIntegration:
    """Integration tests for confidence model workflow."""
    
    def test_full_workflow(self):
        """Complete confidence computation workflow."""
        model = ConfidenceModel()
        tracker = ConfidenceTracker()
        
        # Simulate 20 runs with varying conditions
        for i in range(20):
            regime_stability = 0.7 + np.random.random() * 0.3
            
            result = model.from_run_context(
                regime_label=1,
                regime_stability=regime_stability,
                detector_z_scores={"ar1": 3.0, "pca": 3.5},
                sensor_validity={"temp": True, "pressure": True},
                model_age_hours=500 + i * 10,
                drift_detected=False,
                data_age_minutes=5.0
            )
            
            tracker.add(result)
        
        # Should have stable, trustworthy confidence
        assert tracker.mean_confidence > 0.6
        assert tracker.trend in ("STABLE", "IMPROVING")
    
    def test_degradation_scenario(self):
        """Confidence degrades with worsening conditions."""
        model = ConfidenceModel()
        
        # Good conditions
        good = model.from_run_context(
            regime_label=1,
            regime_stability=0.95,
            detector_z_scores={"ar1": 3.0, "pca": 3.0},
            sensor_validity={"temp": True, "pressure": True},
            model_age_hours=1000,
            drift_detected=False,
            data_age_minutes=2.0
        )
        
        # Bad conditions
        bad = model.from_run_context(
            regime_label=-1,  # Unknown regime
            regime_stability=0.0,
            detector_z_scores={"ar1": 1.0, "pca": 8.0},  # Disagreement
            sensor_validity={"temp": False, "pressure": False},
            model_age_hours=10,  # New model
            drift_detected=True,
            data_age_minutes=50.0  # Stale
        )
        
        assert good.confidence > bad.confidence
        assert good.is_trustworthy is True
        assert bad.is_trustworthy is False

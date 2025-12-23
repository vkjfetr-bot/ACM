"""
Tests for core/rul_reliability.py - RUL Reliability Gate

v11.0.0 Phase 4.4 Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.rul_reliability import (
    RULStatus,
    RULPrerequisites,
    RULResult,
    RULReliabilityGate,
    create_rul_result_from_legacy,
)


# =============================================================================
# RULStatus Tests
# =============================================================================

class TestRULStatus:
    """Tests for RULStatus enum."""
    
    def test_all_values_exist(self):
        """All expected status values exist."""
        assert RULStatus.RELIABLE.value == "RELIABLE"
        assert RULStatus.NOT_RELIABLE.value == "NOT_RELIABLE"
        assert RULStatus.INSUFFICIENT_DATA.value == "INSUFFICIENT_DATA"
        assert RULStatus.NO_DEGRADATION.value == "NO_DEGRADATION"
        assert RULStatus.REGIME_UNSTABLE.value == "REGIME_UNSTABLE"
        assert RULStatus.DETECTOR_DISAGREEMENT.value == "DETECTOR_DISAGREEMENT"
    
    def test_allows_numeric_rul(self):
        """Only RELIABLE status allows numeric RUL display."""
        assert RULStatus.RELIABLE.allows_numeric_rul is True
        assert RULStatus.NOT_RELIABLE.allows_numeric_rul is False
        assert RULStatus.INSUFFICIENT_DATA.allows_numeric_rul is False
        assert RULStatus.NO_DEGRADATION.allows_numeric_rul is False
        assert RULStatus.REGIME_UNSTABLE.allows_numeric_rul is False
        assert RULStatus.DETECTOR_DISAGREEMENT.allows_numeric_rul is False
    
    def test_display_text(self):
        """Each status has readable display text."""
        for status in RULStatus:
            text = status.display_text
            assert isinstance(text, str)
            assert len(text) > 5


# =============================================================================
# RULPrerequisites Tests
# =============================================================================

class TestRULPrerequisites:
    """Tests for RULPrerequisites dataclass."""
    
    def test_default_values(self):
        """Default prerequisites have sensible values."""
        prereqs = RULPrerequisites()
        
        assert prereqs.min_data_points == 500
        assert prereqs.min_degradation_episodes == 2
        assert prereqs.min_health_trend_points == 50
        assert prereqs.min_regime_stability_hours == 24.0
        assert prereqs.max_data_gap_hours == 48.0
        assert prereqs.min_detector_agreement == 0.6
        assert prereqs.min_health_decline == 10.0
        assert prereqs.min_health_confidence == 0.5
    
    def test_custom_values(self):
        """Custom prerequisites override defaults."""
        prereqs = RULPrerequisites(
            min_data_points=1000,
            min_degradation_episodes=5,
            min_regime_stability_hours=48.0
        )
        
        assert prereqs.min_data_points == 1000
        assert prereqs.min_degradation_episodes == 5
        assert prereqs.min_regime_stability_hours == 48.0
        # Defaults still apply to unspecified
        assert prereqs.min_health_trend_points == 50
    
    def test_to_dict_round_trip(self):
        """Prerequisites survive dict serialization."""
        original = RULPrerequisites(
            min_data_points=750,
            max_data_gap_hours=72.0
        )
        
        as_dict = original.to_dict()
        restored = RULPrerequisites.from_dict(as_dict)
        
        assert restored.min_data_points == 750
        assert restored.max_data_gap_hours == 72.0
    
    def test_from_dict_with_missing_keys(self):
        """from_dict handles missing keys with defaults."""
        partial_dict = {"min_data_points": 300}
        prereqs = RULPrerequisites.from_dict(partial_dict)
        
        assert prereqs.min_data_points == 300
        assert prereqs.min_degradation_episodes == 2  # default


# =============================================================================
# RULResult Tests
# =============================================================================

class TestRULResult:
    """Tests for RULResult dataclass."""
    
    def test_reliable_result(self):
        """Reliable result has correct properties."""
        result = RULResult(
            status=RULStatus.RELIABLE,
            rul_hours=120.0,
            p10_lower=80.0,
            p50_median=120.0,
            p90_upper=200.0,
            confidence=0.85,
            method="monte_carlo"
        )
        
        assert result.is_reliable is True
        assert result.status == RULStatus.RELIABLE
        assert result.rul_hours == 120.0
    
    def test_not_reliable_result(self):
        """Not reliable result blocks numeric display."""
        result = RULResult(
            status=RULStatus.NOT_RELIABLE,
            rul_hours=None,
            confidence=0.0,
            method="none",
            prerequisite_failures=["Only 100 data points (need 500)"]
        )
        
        assert result.is_reliable is False
        assert "not reliable" in result.display_rul.lower()
    
    def test_display_rul_hours(self):
        """Display RUL formats hours correctly."""
        result = RULResult(
            status=RULStatus.RELIABLE,
            rul_hours=12.5,
            confidence=0.8,
            method="linear"
        )
        assert "12.5 hours" in result.display_rul
    
    def test_display_rul_days(self):
        """Display RUL formats days correctly."""
        result = RULResult(
            status=RULStatus.RELIABLE,
            rul_hours=72.0,  # 3 days
            confidence=0.8,
            method="linear"
        )
        assert "3.0 days" in result.display_rul
    
    def test_display_rul_weeks(self):
        """Display RUL formats weeks correctly."""
        result = RULResult(
            status=RULStatus.RELIABLE,
            rul_hours=336.0,  # 2 weeks
            confidence=0.8,
            method="linear"
        )
        assert "2.0 weeks" in result.display_rul
    
    def test_display_rul_none(self):
        """Display RUL handles None value."""
        result = RULResult(
            status=RULStatus.RELIABLE,
            rul_hours=None,
            confidence=0.8,
            method="linear"
        )
        assert result.display_rul == "N/A"
    
    def test_to_dict(self):
        """Result converts to SQL-compatible dict."""
        result = RULResult(
            status=RULStatus.RELIABLE,
            rul_hours=100.0,
            p10_lower=80.0,
            p50_median=100.0,
            p90_upper=150.0,
            confidence=0.75,
            method="monte_carlo",
            prerequisite_failures=[],
            data_quality_score=0.95,
            regime_stability_score=1.0,
            detector_agreement_score=0.85
        )
        
        d = result.to_dict()
        
        assert d["RULStatus"] == "RELIABLE"
        assert d["RUL_Hours"] == 100.0
        assert d["P10_LowerBound"] == 80.0
        assert d["P50_Median"] == 100.0
        assert d["P90_UpperBound"] == 150.0
        assert d["Confidence"] == 0.75
        assert d["Method"] == "monte_carlo"
        assert d["PrerequisiteFailures"] is None  # Empty list
        assert d["DataQualityScore"] == 0.95
    
    def test_to_dict_with_failures(self):
        """to_dict formats failures as semicolon-separated string."""
        result = RULResult(
            status=RULStatus.NOT_RELIABLE,
            prerequisite_failures=["Failure 1", "Failure 2", "Failure 3"]
        )
        
        d = result.to_dict()
        assert d["PrerequisiteFailures"] == "Failure 1; Failure 2; Failure 3"
    
    def test_not_reliable_factory(self):
        """not_reliable factory creates correct result."""
        result = RULResult.not_reliable(
            failures=["Test failure"],
            data_quality=0.5,
            regime_stability=0.3
        )
        
        assert result.status == RULStatus.NOT_RELIABLE
        assert result.rul_hours is None
        assert result.confidence == 0.0
        assert len(result.prerequisite_failures) == 1
        assert result.data_quality_score == 0.5
    
    def test_no_degradation_factory(self):
        """no_degradation factory creates correct result."""
        result = RULResult.no_degradation()
        
        assert result.status == RULStatus.NO_DEGRADATION
        assert result.rul_hours is None
        assert result.confidence == 0.95
        assert result.data_quality_score == 1.0


# =============================================================================
# RULReliabilityGate Tests
# =============================================================================

class TestRULReliabilityGate:
    """Tests for RULReliabilityGate class."""
    
    @pytest.fixture
    def gate(self):
        """Standard gate with default prerequisites."""
        return RULReliabilityGate()
    
    @pytest.fixture
    def gate_relaxed(self):
        """Gate with relaxed prerequisites for testing."""
        return RULReliabilityGate(RULPrerequisites(
            min_data_points=10,
            min_degradation_episodes=1,
            min_health_trend_points=5,
            min_regime_stability_hours=1.0,
            max_data_gap_hours=100.0,
            min_detector_agreement=0.3,
            min_health_decline=5.0,
            min_health_confidence=0.2
        ))
    
    @pytest.fixture
    def good_data(self):
        """DataFrame with sufficient data points."""
        n = 600
        timestamps = pd.date_range("2024-01-01", periods=n, freq="10min")
        return pd.DataFrame({
            "Timestamp": timestamps,
            "sensor1": np.random.randn(n),
            "sensor2": np.random.randn(n),
        })
    
    @pytest.fixture
    def sparse_data(self):
        """DataFrame with insufficient data points."""
        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="10min")
        return pd.DataFrame({
            "Timestamp": timestamps,
            "sensor1": np.random.randn(n),
        })
    
    @pytest.fixture
    def health_trend(self):
        """Health trend with sufficient points."""
        n = 100
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "health_pct": np.linspace(100, 70, n)  # Declining health
        })
    
    @pytest.fixture
    def healthy_trend(self):
        """Health trend showing healthy equipment."""
        n = 100
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "health_pct": np.full(n, 98.0)  # Steady healthy
        })
    
    def test_init_default_prereqs(self, gate):
        """Gate initializes with default prerequisites."""
        assert gate.prereqs.min_data_points == 500
    
    def test_init_custom_prereqs(self):
        """Gate accepts custom prerequisites."""
        custom = RULPrerequisites(min_data_points=1000)
        gate = RULReliabilityGate(custom)
        assert gate.prereqs.min_data_points == 1000
    
    def test_all_prerequisites_pass(self, gate_relaxed, good_data, health_trend):
        """Returns None when all prerequisites pass."""
        # Create episodes
        episodes = [object(), object()]  # Mock episodes
        
        # Create stable regime history
        regime_history = pd.Series(
            [1, 1, 1, 1, 1],
            index=pd.date_range("2024-01-01", periods=5, freq="6h")
        )
        
        result = gate_relaxed.check_prerequisites(
            data=good_data,
            episodes=episodes,
            health_trend=health_trend,
            current_regime=1,
            regime_history=regime_history,
            detector_z_scores={"ar1": 3.0, "pca": 3.2},
            current_health_pct=75.0,
            current_confidence=0.8
        )
        
        # None means prerequisites passed
        assert result is None
    
    def test_insufficient_data_points(self, gate, sparse_data, health_trend):
        """Fails when data points below threshold."""
        result = gate.check_prerequisites(
            data=sparse_data,
            episodes=[object(), object()],
            health_trend=health_trend,
            current_regime=1,
        )
        
        assert result is not None
        assert result.status == RULStatus.NOT_RELIABLE
        assert any("data points" in f.lower() for f in result.prerequisite_failures)
    
    def test_insufficient_episodes(self, gate, good_data, health_trend):
        """Fails when too few degradation episodes."""
        result = gate.check_prerequisites(
            data=good_data,
            episodes=[],  # No episodes
            health_trend=health_trend,
            current_regime=1,
        )
        
        # Should return NO_DEGRADATION since equipment is healthy
        # (avg health > 90)
        # But our health_trend has declining health to 70, so it should fail
        assert result is not None
        if result.status == RULStatus.NOT_RELIABLE:
            assert any("episodes" in f.lower() for f in result.prerequisite_failures)
    
    def test_no_degradation_healthy_equipment(self, gate_relaxed, good_data, healthy_trend):
        """Returns NO_DEGRADATION for healthy equipment with no episodes."""
        result = gate_relaxed.check_prerequisites(
            data=good_data,
            episodes=[],  # No episodes
            health_trend=healthy_trend,  # 98% health
            current_regime=1,
        )
        
        assert result is not None
        assert result.status == RULStatus.NO_DEGRADATION
        assert result.confidence == 0.95
    
    def test_unknown_regime(self, gate, good_data, health_trend):
        """Fails when current regime is unknown."""
        result = gate.check_prerequisites(
            data=good_data,
            episodes=[object(), object()],
            health_trend=health_trend,
            current_regime=-1,  # Unknown
        )
        
        assert result is not None
        assert result.status == RULStatus.NOT_RELIABLE
        assert any("UNKNOWN" in f for f in result.prerequisite_failures)
    
    def test_regime_unstable(self, gate, good_data, health_trend):
        """Fails when regime recently changed."""
        # Regime changed 1 hour ago (need 24 hours)
        now = datetime.now()
        regime_history = pd.Series(
            [0, 0, 0, 1],  # Changed to regime 1 recently
            index=pd.DatetimeIndex([
                now - timedelta(hours=10),
                now - timedelta(hours=5),
                now - timedelta(hours=2),
                now - timedelta(hours=0.5)  # 30 min ago
            ])
        )
        
        result = gate.check_prerequisites(
            data=good_data,
            episodes=[object(), object()],
            health_trend=health_trend,
            current_regime=1,
            regime_history=regime_history,
        )
        
        assert result is not None
        assert result.status == RULStatus.NOT_RELIABLE
        assert any("stable" in f.lower() for f in result.prerequisite_failures)
    
    def test_data_gap_too_large(self, gate, health_trend):
        """Fails when data has large gaps."""
        # Create data with 72h gap (exceeds 48h limit)
        # First batch: 300 points at 10min intervals = 50h
        times = list(pd.date_range("2024-01-01", periods=300, freq="10min"))
        # Second batch: starts 4 days later (96h gap)
        times.extend(pd.date_range("2024-01-05", periods=300, freq="10min"))
        
        data = pd.DataFrame({
            "Timestamp": times,
            "sensor1": np.random.randn(len(times)),
        })
        
        # Create gate with small max gap to trigger failure
        gap_gate = RULReliabilityGate(RULPrerequisites(
            min_data_points=100,  # We have 600
            min_degradation_episodes=1,
            min_health_trend_points=10,
            max_data_gap_hours=24.0,  # Our gap is ~46h
        ))
        
        result = gap_gate.check_prerequisites(
            data=data,
            episodes=[object(), object()],
            health_trend=health_trend,
            current_regime=1,
        )
        
        assert result is not None
        assert result.status == RULStatus.NOT_RELIABLE
        assert any("gap" in f.lower() for f in result.prerequisite_failures)
    
    def test_detector_disagreement(self, gate_relaxed, good_data, health_trend):
        """Fails when detectors significantly disagree."""
        # Create gate with strict agreement requirement
        strict_gate = RULReliabilityGate(RULPrerequisites(
            min_data_points=10,
            min_degradation_episodes=1,
            min_health_trend_points=5,
            min_detector_agreement=0.9  # Very strict
        ))
        
        regime_history = pd.Series(
            [1, 1, 1],
            index=pd.date_range("2024-01-01", periods=3, freq="24h")
        )
        
        # Detectors with wildly different z-scores
        result = strict_gate.check_prerequisites(
            data=good_data,
            episodes=[object(), object()],
            health_trend=health_trend,
            current_regime=1,
            regime_history=regime_history,
            detector_z_scores={"ar1": 1.0, "pca": 8.0, "iforest": 0.5},  # Disagreement
        )
        
        assert result is not None
        assert result.status == RULStatus.NOT_RELIABLE
        assert any("agreement" in f.lower() for f in result.prerequisite_failures)
    
    def test_low_health_confidence(self, gate, good_data, health_trend):
        """Fails when health confidence is too low."""
        regime_history = pd.Series(
            [1, 1, 1],
            index=pd.date_range("2024-01-01", periods=3, freq="24h")
        )
        
        result = gate.check_prerequisites(
            data=good_data,
            episodes=[object(), object()],
            health_trend=health_trend,
            current_regime=1,
            regime_history=regime_history,
            current_confidence=0.1,  # Very low
        )
        
        assert result is not None
        assert result.status == RULStatus.NOT_RELIABLE
        assert any("confidence" in f.lower() for f in result.prerequisite_failures)
    
    def test_wrap_rul_prediction(self, gate):
        """wrap_rul_prediction creates RELIABLE result."""
        result = gate.wrap_rul_prediction(
            rul_hours=150.0,
            p10=100.0,
            p50=150.0,
            p90=250.0,
            confidence=0.82,
            method="monte_carlo",
            data_quality=0.95,
            regime_stability=1.0,
            detector_agreement=0.88
        )
        
        assert result.status == RULStatus.RELIABLE
        assert result.rul_hours == 150.0
        assert result.p10_lower == 100.0
        assert result.p50_median == 150.0
        assert result.p90_upper == 250.0
        assert result.confidence == 0.82
        assert result.method == "monte_carlo"
        assert result.is_reliable is True
        assert len(result.prerequisite_failures) == 0


class TestRULReliabilityGateHelpers:
    """Tests for helper methods in RULReliabilityGate."""
    
    @pytest.fixture
    def gate(self):
        return RULReliabilityGate()
    
    def test_regime_duration_empty_history(self, gate):
        """Returns 0 for empty regime history."""
        duration = gate._regime_duration_hours(pd.Series([], dtype=int), 1)
        assert duration == 0.0
    
    def test_regime_duration_same_regime_entire_history(self, gate):
        """Returns full span when regime never changed."""
        times = pd.date_range("2024-01-01", periods=25, freq="1h")
        history = pd.Series([1] * 25, index=times)
        
        duration = gate._regime_duration_hours(history, 1)
        assert duration == 24.0  # 25 points over 24 hours
    
    def test_regime_duration_after_change(self, gate):
        """Returns duration since last change."""
        times = pd.date_range("2024-01-01", periods=10, freq="1h")
        # Changed from 0 to 1 at index 5
        history = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], index=times)
        
        duration = gate._regime_duration_hours(history, 1)
        # From index 5 to 9 = 4 hours
        assert duration == pytest.approx(4.0, abs=0.1)
    
    def test_max_data_gap_no_gaps(self, gate):
        """Returns small gap for continuous data."""
        times = pd.date_range("2024-01-01", periods=100, freq="10min")
        data = pd.DataFrame({"Timestamp": times, "value": range(100)})
        
        gap = gate._max_data_gap_hours(data)
        assert gap < 1.0  # 10 min intervals = ~0.167 hours
    
    def test_max_data_gap_with_large_gap(self, gate):
        """Detects large gaps in data."""
        # First batch ends at about 8 hours (50 points * 10 min)
        times = list(pd.date_range("2024-01-01", periods=50, freq="10min"))
        # Second batch starts 2 days later - creating ~40h gap
        times.extend(pd.date_range("2024-01-03", periods=50, freq="10min"))
        
        data = pd.DataFrame({"Timestamp": times, "value": range(100)})
        
        gap = gate._max_data_gap_hours(data)
        assert gap > 35  # Should detect the ~40 hour gap
    
    def test_max_data_gap_empty_data(self, gate):
        """Returns 0 for empty data."""
        gap = gate._max_data_gap_hours(pd.DataFrame())
        assert gap == 0.0
    
    def test_calculate_detector_agreement_single_detector(self, gate):
        """Single detector = perfect agreement."""
        agreement = gate._calculate_detector_agreement({"ar1": 5.0})
        assert agreement == 1.0
    
    def test_calculate_detector_agreement_same_values(self, gate):
        """Same z-scores = perfect agreement."""
        agreement = gate._calculate_detector_agreement({
            "ar1": 3.0, "pca": 3.0, "iforest": 3.0
        })
        assert agreement == 1.0
    
    def test_calculate_detector_agreement_high_variance(self, gate):
        """High variance z-scores = low agreement."""
        agreement = gate._calculate_detector_agreement({
            "ar1": 0.5, "pca": 10.0, "iforest": 5.0
        })
        assert agreement < 0.7
    
    def test_calculate_detector_agreement_with_nan(self, gate):
        """NaN values are filtered out."""
        agreement = gate._calculate_detector_agreement({
            "ar1": 3.0, "pca": np.nan, "iforest": 3.0
        })
        assert agreement == 1.0  # Only valid values compared
    
    def test_calculate_regime_stability_unknown_regime(self, gate):
        """Unknown regime = zero stability."""
        stability = gate._calculate_regime_stability(None, -1)
        assert stability == 0.0
    
    def test_calculate_regime_stability_long_stable(self, gate):
        """Long stable regime = high stability."""
        times = pd.date_range("2024-01-01", periods=48, freq="1h")
        history = pd.Series([1] * 48, index=times)
        
        stability = gate._calculate_regime_stability(history, 1)
        assert stability == 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestRULReliabilityGateIntegration:
    """Integration tests for complete RUL reliability workflow."""
    
    def test_full_workflow_prerequisites_pass(self):
        """Complete workflow when prerequisites pass."""
        gate = RULReliabilityGate(RULPrerequisites(
            min_data_points=50,
            min_degradation_episodes=1,
            min_health_trend_points=10,
            min_regime_stability_hours=2.0,
        ))
        
        # Create sufficient data
        n = 100
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="10min"),
            "sensor": np.random.randn(n)
        })
        
        health_trend = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            "health_pct": np.linspace(100, 60, 50)
        })
        
        regime_history = pd.Series(
            [1] * 10,
            index=pd.date_range("2024-01-01", periods=10, freq="1h")
        )
        
        # Check prerequisites
        result = gate.check_prerequisites(
            data=data,
            episodes=[object()],
            health_trend=health_trend,
            current_regime=1,
            regime_history=regime_history,
            detector_z_scores={"ar1": 4.0, "pca": 4.5},
            current_health_pct=60.0,
            current_confidence=0.8
        )
        
        # Prerequisites passed
        assert result is None
        
        # Compute and wrap RUL
        final = gate.wrap_rul_prediction(
            rul_hours=200.0,
            p10=150.0,
            p50=200.0,
            p90=300.0,
            confidence=0.75,
            method="monte_carlo"
        )
        
        assert final.is_reliable
        assert final.rul_hours == 200.0
    
    def test_full_workflow_prerequisites_fail(self):
        """Complete workflow when prerequisites fail."""
        gate = RULReliabilityGate()  # Default (strict) prerequisites
        
        # Create insufficient data
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=10, freq="10min"),
            "sensor": np.random.randn(10)
        })
        
        health_trend = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "health_pct": [80, 75, 70, 65, 60]
        })
        
        # Check prerequisites
        result = gate.check_prerequisites(
            data=data,
            episodes=[],
            health_trend=health_trend,
            current_regime=-1,  # Unknown
        )
        
        # Prerequisites failed - use this result directly
        assert result is not None
        assert not result.is_reliable
        assert len(result.prerequisite_failures) > 0
        
        # Don't compute RUL - use failure result as final
        final = result
        assert final.rul_hours is None
        display = final.display_rul
        assert display is not None
        assert "not reliable" in display.lower() or "unknown" in display.lower()


# =============================================================================
# Legacy Migration Helper Tests
# =============================================================================

class TestCreateRULResultFromLegacy:
    """Tests for legacy migration helper."""
    
    def test_reliable_legacy(self):
        """Creates RELIABLE result from valid legacy output."""
        result = create_rul_result_from_legacy(
            rul_hours=100.0,
            p10=80.0,
            p50=100.0,
            p90=150.0,
            confidence=0.8,
            method="linear"
        )
        
        assert result.status == RULStatus.RELIABLE
        assert result.rul_hours == 100.0
    
    def test_not_reliable_none_rul(self):
        """Creates NOT_RELIABLE for None RUL."""
        result = create_rul_result_from_legacy(
            rul_hours=None,
            p10=None,
            p50=None,
            p90=None,
            confidence=0.5,
            method="none"
        )
        
        assert result.status == RULStatus.NOT_RELIABLE
    
    def test_not_reliable_low_confidence(self):
        """Creates NOT_RELIABLE for low confidence."""
        result = create_rul_result_from_legacy(
            rul_hours=100.0,
            p10=80.0,
            p50=100.0,
            p90=150.0,
            confidence=0.1,  # Very low
            method="linear"
        )
        
        assert result.status == RULStatus.NOT_RELIABLE
    
    def test_status_override(self):
        """Status override takes precedence."""
        result = create_rul_result_from_legacy(
            rul_hours=100.0,
            p10=80.0,
            p50=100.0,
            p90=150.0,
            confidence=0.8,
            method="linear",
            status_override=RULStatus.REGIME_UNSTABLE
        )
        
        assert result.status == RULStatus.REGIME_UNSTABLE


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_data_frame(self):
        """Handles empty DataFrame gracefully."""
        gate = RULReliabilityGate()
        
        result = gate.check_prerequisites(
            data=pd.DataFrame(),
            episodes=[],
            health_trend=pd.DataFrame(),
            current_regime=1
        )
        
        assert result is not None
        # Should fail due to insufficient data
        assert result.status in (RULStatus.NOT_RELIABLE, RULStatus.NO_DEGRADATION)
    
    def test_nan_z_scores(self):
        """Handles NaN z-scores in detector agreement."""
        gate = RULReliabilityGate()
        
        agreement = gate._calculate_detector_agreement({
            "ar1": np.nan,
            "pca": np.nan,
            "iforest": np.nan
        })
        
        # All NaN = single valid detector = perfect agreement
        assert agreement == 1.0
    
    def test_inf_z_scores(self):
        """Handles inf z-scores in detector agreement."""
        gate = RULReliabilityGate()
        
        agreement = gate._calculate_detector_agreement({
            "ar1": np.inf,
            "pca": 3.0,
            "iforest": 3.0
        })
        
        # Inf filtered out, remaining two agree perfectly
        assert agreement == 1.0
    
    def test_zero_z_scores(self):
        """Handles all-zero z-scores."""
        gate = RULReliabilityGate()
        
        agreement = gate._calculate_detector_agreement({
            "ar1": 0.0,
            "pca": 0.0,
            "iforest": 0.0
        })
        
        assert agreement == 1.0  # Perfect agreement at zero
    
    def test_negative_rul_hours(self):
        """Handles negative RUL (already failed)."""
        result = RULResult(
            status=RULStatus.RELIABLE,
            rul_hours=-10.0,  # Already past failure
            confidence=0.5,
            method="linear"
        )
        
        # Should still display (negative = past due)
        display = result.display_rul
        assert display is not None
        assert "-10.0 hours" in display

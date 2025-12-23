"""
Tests for UnifiedAttribution (P5.4) in core/sensor_attribution.py

Tests the new v11.0.0 unified sensor attribution using frozen baseline artifacts.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from core.sensor_attribution import (
    DeviationDirection,
    SensorStats,
    SensorContribution,
    AttributionResult,
    UnifiedAttribution,
    BaselineNormalizerProtocol
)


# =============================================================================
# MOCK BASELINE NORMALIZER
# =============================================================================


class MockBaselineNormalizer:
    """Mock normalizer that implements BaselineNormalizerProtocol."""
    
    def __init__(self, stats: Dict[str, SensorStats]):
        self._stats = stats
    
    def get_sensor_stats(self, sensor_name: str) -> Optional[SensorStats]:
        return self._stats.get(sensor_name)
    
    @property
    def sensor_names(self) -> List[str]:
        return list(self._stats.keys())


# =============================================================================
# TEST: DeviationDirection Enum
# =============================================================================


class TestDeviationDirection:
    """Tests for DeviationDirection enum."""
    
    def test_values_exist(self):
        """Test all expected values exist."""
        assert DeviationDirection.HIGH.value == "HIGH"
        assert DeviationDirection.LOW.value == "LOW"
        assert DeviationDirection.VOLATILE.value == "VOLATILE"
        assert DeviationDirection.NORMAL.value == "NORMAL"
    
    def test_enum_count(self):
        """Test correct number of values."""
        assert len(DeviationDirection) == 4


# =============================================================================
# TEST: SensorStats
# =============================================================================


class TestSensorStats:
    """Tests for SensorStats dataclass."""
    
    def test_basic_creation(self):
        """Test basic creation with required fields."""
        stats = SensorStats(mean=100.0, std=10.0)
        assert stats.mean == 100.0
        assert stats.std == 10.0
        assert stats.min_val == 0.0
        assert stats.max_val == 0.0
    
    def test_compute_z_score_positive(self):
        """Test z-score computation for value above mean."""
        stats = SensorStats(mean=100.0, std=10.0)
        z = stats.compute_z_score(120.0)
        assert z == pytest.approx(2.0, rel=1e-6)
    
    def test_compute_z_score_negative(self):
        """Test z-score computation for value below mean."""
        stats = SensorStats(mean=100.0, std=10.0)
        z = stats.compute_z_score(80.0)
        assert z == pytest.approx(-2.0, rel=1e-6)
    
    def test_compute_z_score_nan(self):
        """Test z-score computation for NaN value."""
        stats = SensorStats(mean=100.0, std=10.0)
        z = stats.compute_z_score(np.nan)
        assert z == 0.0
    
    def test_compute_z_score_zero_std(self):
        """Test z-score computation with zero std (should not raise)."""
        stats = SensorStats(mean=100.0, std=0.0)
        z = stats.compute_z_score(120.0)
        # Should handle division by near-zero
        assert not np.isnan(z)
        assert not np.isinf(z)


# =============================================================================
# TEST: SensorContribution
# =============================================================================


class TestSensorContribution:
    """Tests for SensorContribution dataclass."""
    
    def test_basic_creation(self):
        """Test creation with sensor name only."""
        contrib = SensorContribution(sensor_name="Temp1")
        assert contrib.sensor_name == "Temp1"
        assert contrib.contribution_pct == 0.0
        assert contrib.z_score == 0.0
        assert contrib.direction == DeviationDirection.NORMAL
        assert contrib.baseline_deviation == 0.0
    
    def test_full_creation(self):
        """Test creation with all fields."""
        contrib = SensorContribution(
            sensor_name="Vibration",
            contribution_pct=45.5,
            z_score=3.2,
            direction=DeviationDirection.HIGH,
            baseline_deviation=3.2
        )
        assert contrib.sensor_name == "Vibration"
        assert contrib.contribution_pct == 45.5
        assert contrib.z_score == 3.2
        assert contrib.direction == DeviationDirection.HIGH
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        contrib = SensorContribution(
            sensor_name="Pressure",
            contribution_pct=25.333,
            z_score=-2.567,
            direction=DeviationDirection.LOW,
            baseline_deviation=2.567
        )
        d = contrib.to_dict()
        assert d["sensor_name"] == "Pressure"
        assert d["contribution_pct"] == 25.33  # Rounded
        assert d["z_score"] == -2.567  # Rounded
        assert d["direction"] == "LOW"
        assert d["baseline_deviation"] == 2.567


# =============================================================================
# TEST: AttributionResult
# =============================================================================


class TestAttributionResult:
    """Tests for AttributionResult dataclass."""
    
    def test_empty_result(self):
        """Test empty result creation."""
        result = AttributionResult()
        assert result.timestamp is None
        assert result.total_z_score == 0.0
        assert result.contributions == []
        assert result.top_3_sensors == []
        assert result.explanation == ""
    
    def test_full_result(self):
        """Test full result creation."""
        ts = pd.Timestamp("2024-01-15 10:30:00")
        contribs = [
            SensorContribution(sensor_name="Temp1", contribution_pct=50.0),
            SensorContribution(sensor_name="Pressure", contribution_pct=30.0)
        ]
        result = AttributionResult(
            timestamp=ts,
            total_z_score=4.5,
            contributions=contribs,
            top_3_sensors=["Temp1", "Pressure"],
            explanation="Elevated anomaly"
        )
        assert result.timestamp == ts
        assert result.total_z_score == 4.5
        assert len(result.contributions) == 2
        assert result.top_3_sensors == ["Temp1", "Pressure"]
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AttributionResult(
            total_z_score=3.5,
            top_3_sensors=["A", "B", "C"],
            explanation="Test explanation"
        )
        d = result.to_dict()
        assert d["timestamp"] is None
        assert d["total_z_score"] == 3.5
        assert d["top_3_sensors"] == ["A", "B", "C"]
        assert d["explanation"] == "Test explanation"
    
    def test_to_sql_row(self):
        """Test conversion to SQL row format."""
        ts = pd.Timestamp("2024-01-15 10:30:00")
        result = AttributionResult(
            timestamp=ts,
            total_z_score=4.2,
            top_3_sensors=["Sensor1", "Sensor2", "Sensor3"],
            explanation="Critical anomaly driven by Sensor1"
        )
        row = result.to_sql_row(equip_id=1, run_id="run_001")
        assert row["EquipID"] == 1
        assert row["RunID"] == "run_001"
        assert row["Timestamp"] == ts
        assert row["TotalZScore"] == 4.2
        assert row["TopSensor1"] == "Sensor1"
        assert row["TopSensor2"] == "Sensor2"
        assert row["TopSensor3"] == "Sensor3"
    
    def test_to_sql_row_partial_sensors(self):
        """Test SQL row with fewer than 3 sensors."""
        result = AttributionResult(
            total_z_score=2.0,
            top_3_sensors=["OnlyOne"]
        )
        row = result.to_sql_row(equip_id=5, run_id="run_002")
        assert row["TopSensor1"] == "OnlyOne"
        assert row["TopSensor2"] is None
        assert row["TopSensor3"] is None


# =============================================================================
# TEST: UnifiedAttribution Initialization
# =============================================================================


class TestUnifiedAttributionInit:
    """Tests for UnifiedAttribution initialization."""
    
    def test_default_init(self):
        """Test default initialization without normalizer."""
        attr = UnifiedAttribution()
        assert attr.normalizer is None
        assert attr.z_threshold_high == 2.0
        assert attr.z_threshold_low == -2.0
    
    def test_init_with_normalizer(self):
        """Test initialization with mock normalizer."""
        stats = {"Temp": SensorStats(mean=100.0, std=10.0)}
        normalizer = MockBaselineNormalizer(stats)
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        assert attr.normalizer is not None
    
    def test_init_custom_thresholds(self):
        """Test initialization with custom z-score thresholds."""
        attr = UnifiedAttribution(z_threshold_high=3.0, z_threshold_low=-3.0)
        assert attr.z_threshold_high == 3.0
        assert attr.z_threshold_low == -3.0


# =============================================================================
# TEST: UnifiedAttribution.attribute()
# =============================================================================


class TestUnifiedAttributionAttribute:
    """Tests for UnifiedAttribution.attribute() method."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sensor data."""
        return pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "Temp1": [120.0],
            "Pressure": [85.0],
            "Vibration": [45.0]
        })
    
    @pytest.fixture
    def normalizer(self):
        """Create mock normalizer with baseline stats."""
        stats = {
            "Temp1": SensorStats(mean=100.0, std=10.0),
            "Pressure": SensorStats(mean=100.0, std=10.0),
            "Vibration": SensorStats(mean=30.0, std=5.0)
        }
        return MockBaselineNormalizer(stats)
    
    def test_attribute_empty_data(self):
        """Test attribution with empty data."""
        attr = UnifiedAttribution()
        result = attr.attribute(raw_data=pd.DataFrame(), fused_z=3.0)
        assert result.total_z_score == 3.0
        assert "No data" in result.explanation
    
    def test_attribute_without_normalizer(self, sample_data):
        """Test attribution using column-level stats (no normalizer)."""
        attr = UnifiedAttribution()
        result = attr.attribute(raw_data=sample_data, fused_z=3.5)
        assert result.total_z_score == 3.5
        assert len(result.contributions) > 0
    
    def test_attribute_with_normalizer(self, sample_data, normalizer):
        """Test attribution using frozen baseline stats."""
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=sample_data, fused_z=4.0)
        
        assert result.total_z_score == 4.0
        assert len(result.contributions) == 3
        
        # Temp1: (120 - 100) / 10 = 2.0 (exactly at threshold)
        # Pressure: (85 - 100) / 10 = -1.5 (NORMAL)
        # Vibration: (45 - 30) / 5 = 3.0 (HIGH)
        temp_contrib = next(c for c in result.contributions if c.sensor_name == "Temp1")
        vib_contrib = next(c for c in result.contributions if c.sensor_name == "Vibration")
        
        assert temp_contrib.z_score == pytest.approx(2.0, rel=1e-6)
        assert vib_contrib.z_score == pytest.approx(3.0, rel=1e-6)
        assert vib_contrib.direction == DeviationDirection.HIGH
    
    def test_attribute_timestamp_extraction(self, sample_data, normalizer):
        """Test timestamp is correctly extracted."""
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=sample_data, fused_z=3.0)
        assert result.timestamp == pd.Timestamp("2024-01-15 10:00:00")
    
    def test_attribute_top_3_sensors(self, sample_data, normalizer):
        """Test top 3 sensors are correctly identified."""
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=sample_data, fused_z=3.0)
        
        assert len(result.top_3_sensors) == 3
        # Vibration should be first (z=3.0 highest deviation)
        assert result.top_3_sensors[0] == "Vibration"
    
    def test_attribute_custom_sensor_cols(self, sample_data, normalizer):
        """Test attribution with specified sensor columns."""
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(
            raw_data=sample_data,
            fused_z=3.0,
            sensor_cols=["Temp1", "Pressure"]  # Exclude Vibration
        )
        
        assert len(result.contributions) == 2
        sensor_names = [c.sensor_name for c in result.contributions]
        assert "Vibration" not in sensor_names
    
    def test_attribute_explanation_critical(self, sample_data, normalizer):
        """Test explanation for critical severity."""
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=sample_data, fused_z=6.0)
        assert "Critical" in result.explanation
    
    def test_attribute_explanation_elevated(self, sample_data, normalizer):
        """Test explanation for elevated severity."""
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=sample_data, fused_z=4.0)
        assert "Elevated" in result.explanation
    
    def test_attribute_explanation_minor(self, sample_data, normalizer):
        """Test explanation for minor severity."""
        # Create data with smaller deviations
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "Temp1": [105.0],  # z = 0.5
            "Pressure": [95.0],  # z = -0.5
        })
        stats = {
            "Temp1": SensorStats(mean=100.0, std=10.0),
            "Pressure": SensorStats(mean=100.0, std=10.0)
        }
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=2.5)
        assert "Minor" in result.explanation


# =============================================================================
# TEST: UnifiedAttribution Direction Detection
# =============================================================================


class TestUnifiedAttributionDirection:
    """Tests for direction detection logic."""
    
    def test_direction_high(self):
        """Test HIGH direction detection."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "Sensor1": [130.0]  # z = 3.0 (above threshold)
        })
        stats = {"Sensor1": SensorStats(mean=100.0, std=10.0)}
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=3.0)
        
        assert result.contributions[0].direction == DeviationDirection.HIGH
    
    def test_direction_low(self):
        """Test LOW direction detection."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "Sensor1": [70.0]  # z = -3.0 (below threshold)
        })
        stats = {"Sensor1": SensorStats(mean=100.0, std=10.0)}
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=3.0)
        
        assert result.contributions[0].direction == DeviationDirection.LOW
    
    def test_direction_normal(self):
        """Test NORMAL direction for values within thresholds."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "Sensor1": [105.0]  # z = 0.5 (within threshold)
        })
        stats = {"Sensor1": SensorStats(mean=100.0, std=10.0)}
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=1.0)
        
        assert result.contributions[0].direction == DeviationDirection.NORMAL
    
    def test_custom_thresholds_affect_direction(self):
        """Test custom thresholds change direction detection."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "Sensor1": [125.0]  # z = 2.5
        })
        stats = {"Sensor1": SensorStats(mean=100.0, std=10.0)}
        normalizer = MockBaselineNormalizer(stats)
        
        # With default threshold (2.0), this would be HIGH
        attr_default = UnifiedAttribution(baseline_normalizer=normalizer)
        result_default = attr_default.attribute(raw_data=data, fused_z=2.5)
        assert result_default.contributions[0].direction == DeviationDirection.HIGH
        
        # With higher threshold (3.0), this would be NORMAL
        attr_high = UnifiedAttribution(baseline_normalizer=normalizer, z_threshold_high=3.0)
        result_high = attr_high.attribute(raw_data=data, fused_z=2.5)
        assert result_high.contributions[0].direction == DeviationDirection.NORMAL


# =============================================================================
# TEST: UnifiedAttribution Contribution Calculation
# =============================================================================


class TestUnifiedAttributionContribution:
    """Tests for contribution percentage calculation."""
    
    def test_contribution_percentages_sum_to_100(self):
        """Test that contribution percentages sum to 100."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "Sensor1": [130.0],  # z = 3.0
            "Sensor2": [120.0],  # z = 2.0
            "Sensor3": [110.0]   # z = 1.0
        })
        stats = {
            "Sensor1": SensorStats(mean=100.0, std=10.0),
            "Sensor2": SensorStats(mean=100.0, std=10.0),
            "Sensor3": SensorStats(mean=100.0, std=10.0)
        }
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=3.0)
        
        total_pct = sum(c.contribution_pct for c in result.contributions)
        assert total_pct == pytest.approx(100.0, rel=1e-6)
    
    def test_larger_deviation_gets_higher_contribution(self):
        """Test that larger deviation gets higher contribution percentage."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "BigDev": [150.0],   # z = 5.0
            "SmallDev": [110.0]  # z = 1.0
        })
        stats = {
            "BigDev": SensorStats(mean=100.0, std=10.0),
            "SmallDev": SensorStats(mean=100.0, std=10.0)
        }
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=5.0)
        
        big_contrib = next(c for c in result.contributions if c.sensor_name == "BigDev")
        small_contrib = next(c for c in result.contributions if c.sensor_name == "SmallDev")
        
        assert big_contrib.contribution_pct > small_contrib.contribution_pct
        # BigDev should have ~83% (5/(5+1) = 83.3%)
        assert big_contrib.contribution_pct == pytest.approx(83.33, rel=0.1)
    
    def test_negative_deviation_counted_as_absolute(self):
        """Test that negative deviations use absolute value for contribution."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "HighSensor": [130.0],  # z = +3.0
            "LowSensor": [70.0]     # z = -3.0
        })
        stats = {
            "HighSensor": SensorStats(mean=100.0, std=10.0),
            "LowSensor": SensorStats(mean=100.0, std=10.0)
        }
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=3.0)
        
        # Both should have equal contribution (|3.0| == |−3.0|)
        high = next(c for c in result.contributions if c.sensor_name == "HighSensor")
        low = next(c for c in result.contributions if c.sensor_name == "LowSensor")
        
        assert high.contribution_pct == pytest.approx(50.0, rel=1e-6)
        assert low.contribution_pct == pytest.approx(50.0, rel=1e-6)


# =============================================================================
# TEST: UnifiedAttribution.attribute_episode()
# =============================================================================


class TestUnifiedAttributionEpisode:
    """Tests for attribute_episode() method."""
    
    @pytest.fixture
    def episode_data(self):
        """Create sample episode data with multiple rows."""
        return pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-15 10:00:00", periods=5, freq="1min"),
            "Temp1": [110.0, 120.0, 150.0, 130.0, 115.0],  # Peak at row 2
            "Pressure": [95.0, 90.0, 85.0, 88.0, 92.0],
            "fused_z": [2.0, 3.0, 5.0, 4.0, 2.5]  # Peak at row 2 (5.0)
        })
    
    @pytest.fixture
    def normalizer(self):
        """Create mock normalizer."""
        stats = {
            "Temp1": SensorStats(mean=100.0, std=10.0),
            "Pressure": SensorStats(mean=100.0, std=10.0)
        }
        return MockBaselineNormalizer(stats)
    
    def test_episode_uses_peak_row(self, episode_data, normalizer):
        """Test that episode attribution uses the peak fused_z row."""
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute_episode(episode_data)
        
        # Should use peak z-score
        assert result.total_z_score == 5.0
        
        # Should use peak row values (Temp1=150, Pressure=85)
        temp_contrib = next(c for c in result.contributions if c.sensor_name == "Temp1")
        assert temp_contrib.z_score == pytest.approx(5.0, rel=1e-6)  # (150-100)/10
    
    def test_episode_empty_data(self, normalizer):
        """Test episode attribution with empty data."""
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute_episode(pd.DataFrame())
        assert "Empty episode" in result.explanation
    
    def test_episode_without_fused_z_column(self, normalizer):
        """Test episode attribution without fused_z column."""
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-15 10:00:00", periods=3, freq="1min"),
            "Temp1": [110.0, 120.0, 130.0]
        })
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute_episode(data)
        
        # Should use last row
        assert result.total_z_score == 0.0  # No fused_z available
        temp_contrib = next(c for c in result.contributions if c.sensor_name == "Temp1")
        assert temp_contrib.z_score == pytest.approx(3.0, rel=1e-6)  # Last row: (130-100)/10


# =============================================================================
# TEST: BaselineNormalizerProtocol
# =============================================================================


class TestBaselineNormalizerProtocol:
    """Tests for BaselineNormalizerProtocol."""
    
    def test_mock_normalizer_implements_protocol(self):
        """Test that mock normalizer implements the protocol."""
        stats = {"Sensor1": SensorStats(mean=100.0, std=10.0)}
        normalizer = MockBaselineNormalizer(stats)
        
        assert isinstance(normalizer, BaselineNormalizerProtocol)
    
    def test_protocol_get_sensor_stats(self):
        """Test protocol get_sensor_stats method."""
        stats = {"Sensor1": SensorStats(mean=100.0, std=10.0)}
        normalizer = MockBaselineNormalizer(stats)
        
        result = normalizer.get_sensor_stats("Sensor1")
        assert result is not None
        assert result.mean == 100.0
        
        result_missing = normalizer.get_sensor_stats("Unknown")
        assert result_missing is None
    
    def test_protocol_sensor_names(self):
        """Test protocol sensor_names property."""
        stats = {
            "A": SensorStats(mean=1.0, std=1.0),
            "B": SensorStats(mean=2.0, std=1.0)
        }
        normalizer = MockBaselineNormalizer(stats)
        
        names = normalizer.sensor_names
        assert set(names) == {"A", "B"}


# =============================================================================
# TEST: Integration with Missing Sensor Stats
# =============================================================================


class TestUnifiedAttributionMissingSensorStats:
    """Tests for handling sensors without baseline stats."""
    
    def test_sensor_without_stats_uses_column_stats(self):
        """Test sensor without baseline stats uses column-level statistics."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "KnownSensor": [130.0],   # Has baseline stats
            "UnknownSensor": [200.0]  # No baseline stats
        })
        # Only provide stats for KnownSensor
        stats = {"KnownSensor": SensorStats(mean=100.0, std=10.0)}
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=3.0)
        
        # Both sensors should be in contributions
        sensor_names = [c.sensor_name for c in result.contributions]
        assert "KnownSensor" in sensor_names
        assert "UnknownSensor" in sensor_names
    
    def test_nan_values_excluded(self):
        """Test that NaN sensor values are excluded from attribution."""
        data = pd.DataFrame({
            "Timestamp": [pd.Timestamp("2024-01-15 10:00:00")],
            "ValidSensor": [130.0],
            "NanSensor": [np.nan]
        })
        stats = {
            "ValidSensor": SensorStats(mean=100.0, std=10.0),
            "NanSensor": SensorStats(mean=100.0, std=10.0)
        }
        normalizer = MockBaselineNormalizer(stats)
        
        attr = UnifiedAttribution(baseline_normalizer=normalizer)
        result = attr.attribute(raw_data=data, fused_z=3.0)
        
        sensor_names = [c.sensor_name for c in result.contributions]
        assert "ValidSensor" in sensor_names
        assert "NanSensor" not in sensor_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

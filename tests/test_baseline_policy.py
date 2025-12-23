"""Tests for P5.5: Baseline Window Policy.

Tests the BaselinePolicy, BaselineRequirements, and BaselineAssessment classes
from core/baseline_policy.py.
"""

import numpy as np
import pandas as pd
import pytest

from core.baseline_policy import (
    BaselineQuality,
    BaselineRequirements,
    BaselineAssessment,
    BaselinePolicy,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def excellent_baseline_data():
    """Create baseline data that meets all requirements."""
    n = 600
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    data = pd.DataFrame({
        "Timestamp": timestamps,
        "sensor1": np.random.normal(100, 5, n),
        "sensor2": np.random.normal(50, 2, n),
        "sensor3": np.random.normal(200, 10, n),
    })
    return data


@pytest.fixture
def marginal_baseline_data():
    """Create baseline data with some issues."""
    n = 300  # Below default 500 rows
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    data = pd.DataFrame({
        "Timestamp": timestamps,
        "sensor1": np.random.normal(100, 5, n),
        "sensor2": np.random.normal(50, 2, n),
    })
    # Add some nulls
    data.loc[50:60, "sensor1"] = np.nan
    return data


@pytest.fixture
def insufficient_baseline_data():
    """Create baseline data that is insufficient."""
    n = 100  # Way below requirements
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    data = pd.DataFrame({
        "Timestamp": timestamps,
        "sensor1": np.random.normal(100, 5, n),
    })
    return data


@pytest.fixture
def regime_labels_diverse():
    """Create diverse regime labels."""
    return pd.Series([0] * 200 + [1] * 200 + [2] * 200)


@pytest.fixture
def regime_labels_single():
    """Create single regime labels."""
    return pd.Series([0] * 600)


# =============================================================================
# BaselineQuality Tests
# =============================================================================

class TestBaselineQuality:
    """Tests for BaselineQuality enum."""

    def test_quality_values(self):
        """Test quality enum values."""
        assert BaselineQuality.EXCELLENT.value == "EXCELLENT"
        assert BaselineQuality.ADEQUATE.value == "ADEQUATE"
        assert BaselineQuality.MARGINAL.value == "MARGINAL"
        assert BaselineQuality.INSUFFICIENT.value == "INSUFFICIENT"


# =============================================================================
# BaselineRequirements Tests
# =============================================================================

class TestBaselineRequirements:
    """Tests for BaselineRequirements dataclass."""

    def test_default_requirements(self):
        """Test default requirement values."""
        reqs = BaselineRequirements()

        assert reqs.min_rows == 500
        assert reqs.min_hours == 168.0  # 7 days
        assert reqs.max_gap_hours == 24.0
        assert reqs.min_sensor_coverage == 0.9
        assert reqs.require_regime_diversity is True
        assert reqs.min_regimes == 1

    def test_custom_requirements(self):
        """Test custom requirement values."""
        reqs = BaselineRequirements(
            min_rows=1000,
            min_hours=336,  # 14 days
            max_gap_hours=12.0
        )

        assert reqs.min_rows == 1000
        assert reqs.min_hours == 336
        assert reqs.max_gap_hours == 12.0


# =============================================================================
# BaselineAssessment Tests
# =============================================================================

class TestBaselineAssessment:
    """Tests for BaselineAssessment dataclass."""

    def test_assessment_creation(self):
        """Test assessment creation."""
        assessment = BaselineAssessment(
            quality=BaselineQuality.EXCELLENT,
            actual_rows=600,
            actual_hours=600.0,
            max_gap_hours=1.0,
            sensor_coverage=0.95,
            regime_count=3,
            violations=[],
            can_proceed=True
        )

        assert assessment.quality == BaselineQuality.EXCELLENT
        assert assessment.actual_rows == 600
        assert assessment.can_proceed is True

    def test_to_sql_row(self):
        """Test SQL row conversion."""
        assessment = BaselineAssessment(
            quality=BaselineQuality.ADEQUATE,
            actual_rows=400,
            actual_hours=400.0,
            max_gap_hours=5.0,
            sensor_coverage=0.92,
            regime_count=2,
            violations=["Only 400 rows (need 500)"],
            can_proceed=True
        )

        row = assessment.to_sql_row(run_id=1, equip_id=123, equipment_type="FD_FAN")

        assert row["RunID"] == 1
        assert row["EquipID"] == 123
        assert row["EquipmentType"] == "FD_FAN"
        assert row["Quality"] == "ADEQUATE"
        assert row["ActualRows"] == 400
        assert row["CanProceed"] is True
        assert "400 rows" in row["Violations"]


# =============================================================================
# BaselinePolicy Init Tests
# =============================================================================

class TestBaselinePolicyInit:
    """Tests for BaselinePolicy initialization."""

    def test_default_init(self):
        """Test default initialization has equipment types."""
        policy = BaselinePolicy()

        assert "*" in policy.requirements
        assert "GAS_TURBINE" in policy.requirements
        assert "FD_FAN" in policy.requirements

    def test_custom_requirements(self):
        """Test custom requirements initialization."""
        custom = {
            "*": BaselineRequirements(min_rows=200),
            "CUSTOM": BaselineRequirements(min_rows=1000)
        }

        policy = BaselinePolicy(equipment_requirements=custom)

        assert policy.requirements["*"].min_rows == 200
        assert policy.requirements["CUSTOM"].min_rows == 1000


class TestBaselinePolicyGetRequirements:
    """Tests for getting requirements."""

    def test_get_known_equipment(self):
        """Test getting requirements for known equipment."""
        policy = BaselinePolicy()

        reqs = policy.get_requirements("GAS_TURBINE")

        assert reqs.min_rows == 1000
        assert reqs.min_hours == 336

    def test_get_unknown_equipment(self):
        """Test getting requirements for unknown equipment uses default."""
        policy = BaselinePolicy()

        reqs = policy.get_requirements("UNKNOWN_EQUIPMENT")

        assert reqs == policy.requirements["*"]

    def test_set_requirements(self):
        """Test setting requirements for equipment type."""
        policy = BaselinePolicy()

        new_reqs = BaselineRequirements(min_rows=2000, min_hours=720)
        policy.set_requirements("NEW_TYPE", new_reqs)

        assert policy.get_requirements("NEW_TYPE").min_rows == 2000


# =============================================================================
# BaselinePolicy Assess Tests
# =============================================================================

class TestBaselinePolicyAssess:
    """Tests for BaselinePolicy.assess()."""

    def test_assess_excellent_data(self, excellent_baseline_data, regime_labels_diverse):
        """Test assessment of excellent baseline data."""
        policy = BaselinePolicy()

        assessment = policy.assess(
            excellent_baseline_data,
            "FD_FAN",
            regime_labels_diverse
        )

        assert assessment.quality == BaselineQuality.EXCELLENT
        assert assessment.can_proceed is True
        assert len(assessment.violations) == 0

    def test_assess_marginal_data(self, marginal_baseline_data):
        """Test assessment of marginal baseline data."""
        policy = BaselinePolicy()

        assessment = policy.assess(
            marginal_baseline_data,
            "FD_FAN",
            regime_labels=None
        )

        assert assessment.quality in [BaselineQuality.MARGINAL, BaselineQuality.ADEQUATE]
        assert len(assessment.violations) > 0
        # Still has over 50% of required rows
        assert assessment.can_proceed is True

    def test_assess_insufficient_data(self, insufficient_baseline_data):
        """Test assessment of insufficient baseline data."""
        policy = BaselinePolicy()

        assessment = policy.assess(
            insufficient_baseline_data,
            "GAS_TURBINE",  # Higher requirements
            regime_labels=None
        )

        assert assessment.quality == BaselineQuality.INSUFFICIENT
        assert assessment.can_proceed is False
        assert len(assessment.violations) > 0

    def test_assess_row_count_violation(self):
        """Test row count violation detection."""
        policy = BaselinePolicy()

        # Only 200 rows
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2025-01-01", periods=200, freq="h"),
            "sensor1": np.random.normal(100, 5, 200)
        })

        assessment = policy.assess(data, "FD_FAN")

        assert any("rows" in v for v in assessment.violations)

    def test_assess_time_span_violation(self):
        """Test time span violation detection."""
        policy = BaselinePolicy()

        # 600 rows but only 10 hours
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2025-01-01", periods=600, freq="min"),
            "sensor1": np.random.normal(100, 5, 600)
        })

        assessment = policy.assess(data, "FD_FAN")

        assert any("data" in v and "need" in v for v in assessment.violations)

    def test_assess_gap_violation(self):
        """Test gap violation detection."""
        policy = BaselinePolicy()

        # Create data with 48-hour gap
        timestamps1 = pd.date_range("2025-01-01", periods=300, freq="h")
        timestamps2 = pd.date_range("2025-01-15 00:00:00", periods=300, freq="h")
        timestamps = pd.concat([pd.Series(timestamps1), pd.Series(timestamps2)], ignore_index=True)

        data = pd.DataFrame({
            "Timestamp": timestamps,
            "sensor1": np.random.normal(100, 5, 600)
        })

        assessment = policy.assess(data, "FD_FAN")

        assert any("Gap" in v or "exceeds" in v for v in assessment.violations)

    def test_assess_sensor_coverage_violation(self):
        """Test sensor coverage violation detection."""
        policy = BaselinePolicy()

        n = 600
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "sensor1": np.random.normal(100, 5, n),
            "sensor2": np.random.normal(50, 2, n),
        })
        # Set 50% of sensor1 to null
        data.loc[:300, "sensor1"] = np.nan

        assessment = policy.assess(data, "FD_FAN")

        assert any("coverage" in v.lower() for v in assessment.violations)

    def test_assess_regime_diversity_violation(self, excellent_baseline_data):
        """Test regime diversity violation detection."""
        policy = BaselinePolicy()

        # No regime labels provided
        assessment = policy.assess(
            excellent_baseline_data,
            "FD_FAN",
            regime_labels=None
        )

        # Should have violation about regimes
        assert any("regime" in v.lower() for v in assessment.violations)

    def test_assess_empty_data(self):
        """Test assessment of empty data."""
        policy = BaselinePolicy()

        data = pd.DataFrame({"Timestamp": [], "sensor1": []})

        assessment = policy.assess(data, "FD_FAN")

        assert assessment.quality == BaselineQuality.INSUFFICIENT
        assert assessment.can_proceed is False


class TestBaselinePolicyQualityComputation:
    """Tests for quality computation logic."""

    def test_excellent_no_violations(self, excellent_baseline_data, regime_labels_diverse):
        """Test EXCELLENT when no violations."""
        policy = BaselinePolicy()

        assessment = policy.assess(
            excellent_baseline_data,
            "FD_FAN",
            regime_labels_diverse
        )

        assert assessment.quality == BaselineQuality.EXCELLENT

    def test_adequate_one_violation_with_70pct_rows(self):
        """Test ADEQUATE with one violation and 70% rows."""
        policy = BaselinePolicy()

        # 400 rows = 80% of 500 requirement
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2025-01-01", periods=400, freq="h"),
            "sensor1": np.random.normal(100, 5, 400)
        })
        regime_labels = pd.Series([0] * 400)  # Single regime but meets min

        assessment = policy.assess(data, "FD_FAN", regime_labels)

        # Should be ADEQUATE (one violation: rows, but has 80% of requirement)
        assert assessment.quality in [BaselineQuality.ADEQUATE, BaselineQuality.MARGINAL]

    def test_marginal_with_50pct_rows(self):
        """Test MARGINAL with 50% of required rows."""
        policy = BaselinePolicy()

        # 300 rows = 60% of 500 requirement, but short on time
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2025-01-01", periods=300, freq="h"),
            "sensor1": np.random.normal(100, 5, 300)
        })

        assessment = policy.assess(data, "FD_FAN")

        # Multiple violations but has > 50% rows
        assert assessment.quality in [BaselineQuality.MARGINAL, BaselineQuality.ADEQUATE]
        assert assessment.can_proceed is True

    def test_insufficient_below_50pct_rows(self):
        """Test INSUFFICIENT when below 50% of required rows."""
        policy = BaselinePolicy()

        # 200 rows = 40% of 500 requirement
        data = pd.DataFrame({
            "Timestamp": pd.date_range("2025-01-01", periods=200, freq="h"),
            "sensor1": np.random.normal(100, 5, 200)
        })

        assessment = policy.assess(data, "FD_FAN")

        assert assessment.quality == BaselineQuality.INSUFFICIENT
        assert assessment.can_proceed is False


class TestBaselinePolicySummary:
    """Tests for policy summary."""

    def test_get_summary(self):
        """Test getting policy summary."""
        policy = BaselinePolicy()

        summary = policy.get_summary()

        assert "*" in summary
        assert "GAS_TURBINE" in summary
        assert summary["GAS_TURBINE"]["min_rows"] == 1000
        assert summary["FD_FAN"]["min_hours"] == 168

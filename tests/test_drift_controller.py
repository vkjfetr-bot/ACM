"""Tests for P5.1-P5.3: Drift/Novelty Control Plane.

Tests the DriftController, NoveltyTracker, and DriftEventManager classes
from core/drift_controller.py.
"""

import numpy as np
import pandas as pd
import pytest
from typing import List

from core.drift_controller import (
    DriftAction,
    DriftThresholds,
    DriftSignal,
    DriftController,
    NoveltyPressure,
    NoveltyTracker,
    DriftEvent,
    DriftEventManager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def drift_thresholds():
    """Standard drift thresholds for testing."""
    return DriftThresholds(
        psi_warning=0.1,
        psi_critical=0.25,
        kl_warning=0.1,
        kl_critical=0.5,
        novelty_warning=0.3,
        novelty_critical=0.5
    )


@pytest.fixture
def psi_normal():
    """PSI values below warning threshold."""
    return pd.DataFrame({
        'sensor1': [0.05],
        'sensor2': [0.08],
        'sensor3': [0.03]
    })


@pytest.fixture
def psi_warning():
    """PSI values triggering warning."""
    return pd.DataFrame({
        'sensor1': [0.15],  # Warning
        'sensor2': [0.05],  # Normal
        'sensor3': [0.12]   # Warning
    })


@pytest.fixture
def psi_critical():
    """PSI values triggering critical."""
    return pd.DataFrame({
        'sensor1': [0.30],  # Critical
        'sensor2': [0.15],  # Warning
        'sensor3': [0.05]   # Normal
    })


@pytest.fixture
def sample_timestamp():
    """Sample timestamp for testing."""
    return pd.Timestamp("2025-01-15 10:00:00")


# =============================================================================
# DriftThresholds Tests
# =============================================================================

class TestDriftThresholds:
    """Tests for DriftThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values."""
        t = DriftThresholds()

        assert t.psi_warning == 0.1
        assert t.psi_critical == 0.25
        assert t.kl_warning == 0.1
        assert t.kl_critical == 0.5
        assert t.novelty_warning == 0.3
        assert t.novelty_critical == 0.5

    def test_custom_values(self):
        """Test custom threshold values."""
        t = DriftThresholds(
            psi_warning=0.2,
            psi_critical=0.4,
            novelty_warning=0.5
        )

        assert t.psi_warning == 0.2
        assert t.psi_critical == 0.4
        assert t.novelty_warning == 0.5
        # Others should be default
        assert t.kl_warning == 0.1


# =============================================================================
# DriftSignal Tests
# =============================================================================

class TestDriftSignal:
    """Tests for DriftSignal dataclass."""

    def test_signal_creation(self, sample_timestamp):
        """Test signal creation."""
        signal = DriftSignal(
            timestamp=sample_timestamp,
            signal_type="PSI",
            severity="WARNING",
            value=0.15,
            affected_sensors=["sensor1", "sensor2"],
            recommended_action=DriftAction.LOG_WARNING
        )

        assert signal.signal_type == "PSI"
        assert signal.severity == "WARNING"
        assert signal.value == 0.15
        assert len(signal.affected_sensors) == 2

    def test_to_dict(self, sample_timestamp):
        """Test signal serialization."""
        signal = DriftSignal(
            timestamp=sample_timestamp,
            signal_type="NOVELTY",
            severity="CRITICAL",
            value=0.55,
            affected_sensors=[],
            recommended_action=DriftAction.TRIGGER_REPLAY
        )

        d = signal.to_dict()

        assert d["signal_type"] == "NOVELTY"
        assert d["severity"] == "CRITICAL"
        assert d["value"] == 0.55
        assert d["recommended_action"] == "TRIGGER_REPLAY"


# =============================================================================
# DriftController Tests
# =============================================================================

class TestDriftControllerInit:
    """Tests for DriftController initialization."""

    def test_default_init(self):
        """Test default initialization."""
        controller = DriftController()

        assert controller.thresholds is not None
        assert controller.replay_queued is False
        assert controller.get_confidence_penalty() == 0.0

    def test_custom_thresholds(self, drift_thresholds):
        """Test initialization with custom thresholds."""
        controller = DriftController(thresholds=drift_thresholds)

        assert controller.thresholds.psi_warning == 0.1


class TestDriftControllerEvaluatePSI:
    """Tests for PSI evaluation."""

    def test_no_signals_when_normal(self, psi_normal, sample_timestamp):
        """Test no signals when PSI is normal."""
        controller = DriftController()

        signals = controller.evaluate(
            psi_values=psi_normal,
            kl_values=None,
            unknown_regime_pct=0.0,
            timestamp=sample_timestamp
        )

        assert len(signals) == 0

    def test_warning_signals(self, psi_warning, sample_timestamp):
        """Test warning signals are generated."""
        controller = DriftController()

        signals = controller.evaluate(
            psi_values=psi_warning,
            kl_values=None,
            unknown_regime_pct=0.0,
            timestamp=sample_timestamp
        )

        assert len(signals) == 2
        assert all(s.severity == "WARNING" for s in signals)
        assert all(s.signal_type == "PSI" for s in signals)

    def test_critical_signals(self, psi_critical, sample_timestamp):
        """Test critical signals are generated."""
        controller = DriftController()

        signals = controller.evaluate(
            psi_values=psi_critical,
            kl_values=None,
            unknown_regime_pct=0.0,
            timestamp=sample_timestamp
        )

        # Should have 1 critical, 1 warning
        critical = [s for s in signals if s.severity == "CRITICAL"]
        warning = [s for s in signals if s.severity == "WARNING"]

        assert len(critical) == 1
        assert len(warning) == 1


class TestDriftControllerEvaluateNovelty:
    """Tests for novelty evaluation."""

    def test_no_signal_when_low_novelty(self, sample_timestamp):
        """Test no signal when novelty is low."""
        controller = DriftController()

        signals = controller.evaluate(
            psi_values=pd.DataFrame(),
            kl_values=None,
            unknown_regime_pct=0.1,
            timestamp=sample_timestamp
        )

        assert len(signals) == 0

    def test_warning_signal_at_warning_threshold(self, sample_timestamp):
        """Test warning signal at 35% unknown."""
        controller = DriftController()

        signals = controller.evaluate(
            psi_values=pd.DataFrame(),
            kl_values=None,
            unknown_regime_pct=0.35,
            timestamp=sample_timestamp
        )

        assert len(signals) == 1
        assert signals[0].signal_type == "NOVELTY"
        assert signals[0].severity == "WARNING"
        assert signals[0].recommended_action == DriftAction.REDUCE_CONFIDENCE

    def test_critical_signal_at_critical_threshold(self, sample_timestamp):
        """Test critical signal at 55% unknown."""
        controller = DriftController()

        signals = controller.evaluate(
            psi_values=pd.DataFrame(),
            kl_values=None,
            unknown_regime_pct=0.55,
            timestamp=sample_timestamp
        )

        assert len(signals) == 1
        assert signals[0].severity == "CRITICAL"
        assert signals[0].recommended_action == DriftAction.TRIGGER_REPLAY


class TestDriftControllerCallbacks:
    """Tests for callback execution."""

    def test_replay_trigger_callback(self, sample_timestamp):
        """Test replay trigger callback is called."""
        triggered_signals: List[DriftSignal] = []

        def on_replay(signal):
            triggered_signals.append(signal)

        controller = DriftController(on_replay_trigger=on_replay)

        controller.evaluate(
            psi_values=pd.DataFrame({'sensor1': [0.30]}),  # Critical PSI
            kl_values=None,
            unknown_regime_pct=0.0,
            timestamp=sample_timestamp
        )

        assert len(triggered_signals) == 1
        assert controller.replay_queued is True

    def test_replay_not_triggered_twice(self, sample_timestamp):
        """Test replay is only triggered once."""
        trigger_count = [0]

        def on_replay(signal):
            trigger_count[0] += 1

        controller = DriftController(on_replay_trigger=on_replay)

        # First evaluation
        controller.evaluate(
            psi_values=pd.DataFrame({'sensor1': [0.30]}),
            kl_values=None,
            unknown_regime_pct=0.0,
            timestamp=sample_timestamp
        )

        # Second evaluation
        controller.evaluate(
            psi_values=pd.DataFrame({'sensor1': [0.35]}),
            kl_values=None,
            unknown_regime_pct=0.0,
            timestamp=sample_timestamp + pd.Timedelta(minutes=5)
        )

        assert trigger_count[0] == 1  # Only triggered once

    def test_confidence_reduction_callback(self, sample_timestamp):
        """Test confidence reduction callback."""
        penalties = []

        def on_confidence(penalty):
            penalties.append(penalty)

        controller = DriftController(on_confidence_reduction=on_confidence)

        controller.evaluate(
            psi_values=pd.DataFrame(),
            kl_values=None,
            unknown_regime_pct=0.35,  # Warning level novelty
            timestamp=sample_timestamp
        )

        assert len(penalties) == 1
        assert penalties[0] > 0


class TestDriftControllerConfidencePenalty:
    """Tests for confidence penalty accumulation."""

    def test_penalty_accumulates(self, sample_timestamp):
        """Test confidence penalty accumulates."""
        controller = DriftController()

        # First warning
        controller.evaluate(
            psi_values=pd.DataFrame(),
            kl_values=None,
            unknown_regime_pct=0.35,
            timestamp=sample_timestamp
        )

        penalty1 = controller.get_confidence_penalty()
        assert penalty1 == 0.1

        # Second warning
        controller.evaluate(
            psi_values=pd.DataFrame(),
            kl_values=None,
            unknown_regime_pct=0.35,
            timestamp=sample_timestamp + pd.Timedelta(minutes=5)
        )

        penalty2 = controller.get_confidence_penalty()
        assert penalty2 == 0.2

    def test_penalty_capped_at_50_percent(self, sample_timestamp):
        """Test penalty is capped at 0.5."""
        controller = DriftController()

        # Multiple warning signals (WARNING triggers REDUCE_CONFIDENCE)
        for i in range(10):
            controller.evaluate(
                psi_values=pd.DataFrame(),
                kl_values=pd.DataFrame({'sensor1': [0.35]}),  # KL Warning triggers confidence reduction
                unknown_regime_pct=0.0,
                timestamp=sample_timestamp + pd.Timedelta(minutes=i)
            )

        assert controller.get_confidence_penalty() == 0.5

    def test_penalty_reset(self, sample_timestamp):
        """Test penalty can be reset."""
        controller = DriftController()

        controller.evaluate(
            psi_values=pd.DataFrame(),
            kl_values=None,
            unknown_regime_pct=0.35,
            timestamp=sample_timestamp
        )

        assert controller.get_confidence_penalty() > 0

        controller.reset_confidence_penalty()

        assert controller.get_confidence_penalty() == 0.0


# =============================================================================
# NoveltyPressure Tests
# =============================================================================

class TestNoveltyPressure:
    """Tests for NoveltyPressure dataclass."""

    def test_pressure_creation(self, sample_timestamp):
        """Test pressure creation."""
        pressure = NoveltyPressure(
            timestamp=sample_timestamp,
            unknown_pct=0.15,
            emerging_pct=0.05,
            out_of_distribution=2.5,
            trend="STABLE"
        )

        assert pressure.unknown_pct == 0.15
        assert pressure.emerging_pct == 0.05
        assert pressure.trend == "STABLE"

    def test_to_dict(self, sample_timestamp):
        """Test pressure serialization."""
        pressure = NoveltyPressure(
            timestamp=sample_timestamp,
            unknown_pct=0.20,
            emerging_pct=0.10,
            out_of_distribution=3.0,
            trend="INCREASING"
        )

        d = pressure.to_dict()

        assert d["UnknownPct"] == 0.20
        assert d["EmergingPct"] == 0.10
        assert d["Trend"] == "INCREASING"


# =============================================================================
# NoveltyTracker Tests
# =============================================================================

class TestNoveltyTrackerInit:
    """Tests for NoveltyTracker initialization."""

    def test_default_init(self):
        """Test default initialization."""
        tracker = NoveltyTracker()

        assert tracker.window_hours == 24.0
        assert len(tracker.history) == 0

    def test_custom_window(self):
        """Test custom window initialization."""
        tracker = NoveltyTracker(window_hours=48.0)

        assert tracker.window_hours == 48.0


class TestNoveltyTrackerCompute:
    """Tests for NoveltyTracker.compute()."""

    def test_compute_basic(self):
        """Test basic computation."""
        tracker = NoveltyTracker(window_hours=1000.0)  # Large window to include all data

        # Create data with some unknown regimes
        n = 100
        timestamps = pd.date_range("2025-01-15", periods=n, freq="h")
        labels = pd.Series([0] * 80 + [-1] * 20)  # 20% unknown

        pressure = tracker.compute(
            regime_labels=labels,
            regime_distances=None,
            timestamps=timestamps
        )

        assert 0.15 < pressure.unknown_pct < 0.25  # Around 20%
        assert pressure.emerging_pct == 0.0
        assert pressure.trend == "STABLE"

    def test_trend_detection_increasing(self):
        """Test increasing trend detection."""
        tracker = NoveltyTracker(window_hours=1.0, trend_window=3)

        base_time = pd.Timestamp("2025-01-15 00:00:00")

        # Simulate increasing novelty
        for i in range(5):
            unknown_count = 10 + i * 5  # 10, 15, 20, 25, 30
            n = 100
            timestamps = pd.date_range(
                base_time + pd.Timedelta(hours=i),
                periods=n,
                freq="min"
            )
            labels = pd.Series([0] * (n - unknown_count) + [-1] * unknown_count)

            pressure = tracker.compute(
                regime_labels=labels,
                regime_distances=None,
                timestamps=timestamps
            )

        # After increasing novelty, trend should be INCREASING
        assert pressure.trend == "INCREASING"

    def test_get_current_pressure(self):
        """Test get_current_pressure returns latest."""
        tracker = NoveltyTracker(window_hours=1000.0)  # Large window to include all data

        n = 100
        timestamps = pd.date_range("2025-01-15", periods=n, freq="h")
        labels = pd.Series([0] * 90 + [-1] * 10)

        tracker.compute(labels, None, timestamps)

        current = tracker.get_current_pressure()

        assert current is not None
        assert current.unknown_pct == 0.1


# =============================================================================
# DriftEvent Tests
# =============================================================================

class TestDriftEvent:
    """Tests for DriftEvent dataclass."""

    def test_event_creation(self, sample_timestamp):
        """Test event creation."""
        event = DriftEvent(
            id="DRIFT_123_20250115100000",
            equip_id=123,
            detected_at=sample_timestamp,
            event_type="PSI",
            severity="CRITICAL",
            psi_value=0.30,
            kl_value=None,
            affected_sensors=["sensor1"],
            confidence_reduction=0.3,
            models_invalidated=["AR1", "PCA"]
        )

        assert event.id == "DRIFT_123_20250115100000"
        assert event.is_active is True

    def test_to_sql_row(self, sample_timestamp):
        """Test SQL row conversion."""
        event = DriftEvent(
            id="DRIFT_123_20250115100000",
            equip_id=123,
            detected_at=sample_timestamp,
            event_type="NOVELTY",
            severity="WARNING",
            psi_value=None,
            kl_value=None,
            affected_sensors=["sensor1", "sensor2"],
            confidence_reduction=0.1,
            models_invalidated=[]
        )

        row = event.to_sql_row()

        assert row["DriftEventID"] == "DRIFT_123_20250115100000"
        assert row["EquipID"] == 123
        assert row["EventType"] == "NOVELTY"
        assert row["AffectedSensors"] == "sensor1,sensor2"

    def test_is_active_resolved(self, sample_timestamp):
        """Test is_active for resolved event."""
        event = DriftEvent(
            id="DRIFT_123_20250115100000",
            equip_id=123,
            detected_at=sample_timestamp,
            event_type="PSI",
            severity="WARNING",
            psi_value=0.15,
            kl_value=None,
            affected_sensors=["sensor1"],
            confidence_reduction=0.1,
            models_invalidated=[],
            resolved_at=sample_timestamp + pd.Timedelta(hours=1),
            resolution="ACKNOWLEDGED"
        )

        assert event.is_active is False


# =============================================================================
# DriftEventManager Tests
# =============================================================================

class TestDriftEventManagerInit:
    """Tests for DriftEventManager initialization."""

    def test_default_init(self):
        """Test default initialization."""
        manager = DriftEventManager()

        assert len(manager.active_events) == 0
        assert len(manager.resolved_events) == 0


class TestDriftEventManagerCreateEvent:
    """Tests for DriftEventManager.create_event()."""

    def test_create_event_from_signal(self, sample_timestamp):
        """Test creating event from signal."""
        manager = DriftEventManager()

        signal = DriftSignal(
            timestamp=sample_timestamp,
            signal_type="PSI",
            severity="CRITICAL",
            value=0.30,
            affected_sensors=["sensor1"],
            recommended_action=DriftAction.TRIGGER_REPLAY
        )

        event = manager.create_event(signal, equip_id=123)

        assert event.equip_id == 123
        assert event.event_type == "PSI"
        assert event.severity == "CRITICAL"
        assert event.confidence_reduction == 0.3
        assert event.id in manager.active_events

    def test_create_warning_event(self, sample_timestamp):
        """Test warning event has lower confidence reduction."""
        manager = DriftEventManager()

        signal = DriftSignal(
            timestamp=sample_timestamp,
            signal_type="NOVELTY",
            severity="WARNING",
            value=0.35,
            affected_sensors=[],
            recommended_action=DriftAction.REDUCE_CONFIDENCE
        )

        event = manager.create_event(signal, equip_id=456)

        assert event.confidence_reduction == 0.1  # Warning = 0.1


class TestDriftEventManagerResolve:
    """Tests for DriftEventManager.resolve_event()."""

    def test_resolve_event(self, sample_timestamp):
        """Test resolving an event."""
        manager = DriftEventManager()

        signal = DriftSignal(
            timestamp=sample_timestamp,
            signal_type="PSI",
            severity="WARNING",
            value=0.15,
            affected_sensors=["sensor1"],
            recommended_action=DriftAction.LOG_WARNING
        )

        event = manager.create_event(signal, equip_id=123)
        event_id = event.id

        # Resolve it
        result = manager.resolve_event(event_id, "ACKNOWLEDGED")

        assert result is True
        assert event_id not in manager.active_events
        assert len(manager.resolved_events) == 1
        assert manager.resolved_events[0].resolution == "ACKNOWLEDGED"

    def test_resolve_nonexistent_event(self):
        """Test resolving non-existent event returns False."""
        manager = DriftEventManager()

        result = manager.resolve_event("NONEXISTENT", "ACKNOWLEDGED")

        assert result is False


class TestDriftEventManagerConfidencePenalty:
    """Tests for confidence penalty calculation."""

    def test_penalty_from_active_events(self, sample_timestamp):
        """Test confidence penalty calculation."""
        manager = DriftEventManager()

        # Create critical event
        signal1 = DriftSignal(
            timestamp=sample_timestamp,
            signal_type="PSI",
            severity="CRITICAL",
            value=0.30,
            affected_sensors=["sensor1"],
            recommended_action=DriftAction.TRIGGER_REPLAY
        )
        manager.create_event(signal1, equip_id=123)

        # Create warning event
        signal2 = DriftSignal(
            timestamp=sample_timestamp + pd.Timedelta(minutes=1),
            signal_type="PSI",
            severity="WARNING",
            value=0.15,
            affected_sensors=["sensor2"],
            recommended_action=DriftAction.LOG_WARNING
        )
        manager.create_event(signal2, equip_id=123)

        penalty = manager.get_confidence_penalty(equip_id=123)

        assert penalty == 0.4  # 0.3 (critical) + 0.1 (warning)

    def test_penalty_capped(self, sample_timestamp):
        """Test penalty is capped at 0.5."""
        manager = DriftEventManager()

        # Create multiple critical events
        for i in range(5):
            signal = DriftSignal(
                timestamp=sample_timestamp + pd.Timedelta(minutes=i),
                signal_type="PSI",
                severity="CRITICAL",
                value=0.30,
                affected_sensors=[f"sensor{i}"],
                recommended_action=DriftAction.TRIGGER_REPLAY
            )
            manager.create_event(signal, equip_id=123)

        penalty = manager.get_confidence_penalty(equip_id=123)

        assert penalty == 0.5  # Capped

    def test_penalty_excludes_resolved(self, sample_timestamp):
        """Test resolved events don't contribute to penalty."""
        manager = DriftEventManager()

        signal = DriftSignal(
            timestamp=sample_timestamp,
            signal_type="PSI",
            severity="CRITICAL",
            value=0.30,
            affected_sensors=["sensor1"],
            recommended_action=DriftAction.TRIGGER_REPLAY
        )
        event = manager.create_event(signal, equip_id=123)

        # Before resolve
        assert manager.get_confidence_penalty(123) == 0.3

        # Resolve
        manager.resolve_event(event.id, "RETRAINED")

        # After resolve
        assert manager.get_confidence_penalty(123) == 0.0


class TestDriftEventManagerSummary:
    """Tests for event summary."""

    def test_get_event_summary(self, sample_timestamp):
        """Test event summary generation."""
        manager = DriftEventManager()

        # Create events with different timestamps to get unique IDs
        signal1 = DriftSignal(sample_timestamp, "PSI", "CRITICAL", 0.30, ["s1"], DriftAction.TRIGGER_REPLAY)
        signal2 = DriftSignal(sample_timestamp + pd.Timedelta(seconds=1), "NOVELTY", "WARNING", 0.35, [], DriftAction.REDUCE_CONFIDENCE)

        manager.create_event(signal1, equip_id=123)
        manager.create_event(signal2, equip_id=123)

        summary = manager.get_event_summary(equip_id=123)

        assert summary["active_count"] == 2
        assert summary["resolved_count"] == 0
        assert summary["active_by_type"]["PSI"] == 1
        assert summary["active_by_type"]["NOVELTY"] == 1
        assert summary["active_by_severity"]["CRITICAL"] == 1
        assert summary["active_by_severity"]["WARNING"] == 1

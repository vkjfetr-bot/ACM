"""
Tests for ACM v11.0.0 Health State Module.

Tests the time-evolving health state machine with hysteresis.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.health_state import (
    HealthState,
    HealthSnapshot,
    HealthThresholds,
    HealthTracker,
    HealthTransition,
    compute_health_from_z_scores,
)


# ============================================================================
# HealthState Enum Tests
# ============================================================================

class TestHealthState:
    """Tests for HealthState enum."""
    
    def test_health_states_exist(self):
        """All required health states are defined."""
        assert HealthState.HEALTHY
        assert HealthState.DEGRADED
        assert HealthState.CRITICAL
        assert HealthState.UNKNOWN
        assert HealthState.RECOVERING
    
    def test_severity_level_ordering(self):
        """Severity levels are ordered correctly."""
        assert HealthState.HEALTHY.severity_level < HealthState.DEGRADED.severity_level
        assert HealthState.DEGRADED.severity_level < HealthState.CRITICAL.severity_level
        assert HealthState.UNKNOWN.severity_level == -1  # Unknown has negative severity
    
    def test_state_values(self):
        """State values are strings."""
        assert HealthState.HEALTHY.value == "HEALTHY"
        assert HealthState.DEGRADED.value == "DEGRADED"
        assert HealthState.CRITICAL.value == "CRITICAL"
        assert HealthState.UNKNOWN.value == "UNKNOWN"
        assert HealthState.RECOVERING.value == "RECOVERING"
    
    def test_is_actionable(self):
        """Only DEGRADED and CRITICAL require action."""
        assert HealthState.HEALTHY.is_actionable() == False
        assert HealthState.DEGRADED.is_actionable() == True
        assert HealthState.CRITICAL.is_actionable() == True
        assert HealthState.UNKNOWN.is_actionable() == False
        assert HealthState.RECOVERING.is_actionable() == False


# ============================================================================
# HealthSnapshot Tests
# ============================================================================

class TestHealthSnapshot:
    """Tests for HealthSnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """HealthSnapshot can be created with required fields."""
        snapshot = HealthSnapshot(
            timestamp=pd.Timestamp("2024-01-01 10:00:00"),
            state=HealthState.HEALTHY,
            health_pct=95.0,
            confidence=1.0,
            state_duration_hours=2.5,
        )
        assert snapshot.state == HealthState.HEALTHY
        assert snapshot.health_pct == 95.0
        assert snapshot.confidence == 1.0
    
    def test_snapshot_with_all_fields(self):
        """HealthSnapshot with all optional fields."""
        snapshot = HealthSnapshot(
            timestamp=pd.Timestamp("2024-01-01 10:00:00"),
            state=HealthState.DEGRADED,
            health_pct=65.0,
            confidence=0.85,
            state_duration_hours=1.0,
            previous_state=HealthState.HEALTHY,
            transition_reason="fused_z=3.5",
            fused_z_mean=3.5,
            fused_z_max=4.0,
            active_episode_count=1,
            worst_detector="ar1_z",
            worst_detector_z=4.2,
            equip_id=1,
            regime=2,
        )
        assert snapshot.previous_state == HealthState.HEALTHY
        assert snapshot.fused_z_mean == 3.5
        assert snapshot.worst_detector == "ar1_z"
        assert snapshot.regime == 2
    
    def test_snapshot_to_dict(self):
        """Snapshot serialization for SQL."""
        snapshot = HealthSnapshot(
            timestamp=pd.Timestamp("2024-01-01"),
            state=HealthState.CRITICAL,
            health_pct=25.0,
            confidence=0.9,
            state_duration_hours=0.5,
        )
        d = snapshot.to_dict()
        assert d["HealthState"] == "CRITICAL"
        assert d["HealthPct"] == 25.0
        assert d["Confidence"] == 0.9


# ============================================================================
# HealthThresholds Tests
# ============================================================================

class TestHealthThresholds:
    """Tests for HealthThresholds configuration."""
    
    def test_default_thresholds(self):
        """Default thresholds are sensible."""
        thresh = HealthThresholds()
        assert thresh.healthy_to_degraded == 3.0
        assert thresh.degraded_to_critical == 5.5
        assert thresh.degraded_to_healthy == 2.0  # Hysteresis
        assert thresh.critical_to_degraded == 4.0  # Hysteresis
    
    def test_custom_thresholds(self):
        """Custom thresholds can be set."""
        thresh = HealthThresholds(
            healthy_to_degraded=4.0,
            degraded_to_critical=6.0,
            degraded_to_healthy=3.0,
            critical_to_degraded=5.0,
        )
        assert thresh.healthy_to_degraded == 4.0
        assert thresh.critical_to_degraded == 5.0
    
    def test_from_config(self):
        """Thresholds can be loaded from config dict."""
        cfg = {
            "health": {
                "threshold_degraded": 3.5,
                "threshold_critical": 6.0,
                "threshold_healthy": 2.5,
                "threshold_recovering": 4.5,
            }
        }
        thresh = HealthThresholds.from_config(cfg)
        assert thresh.healthy_to_degraded == 3.5
        assert thresh.degraded_to_critical == 6.0
        assert thresh.degraded_to_healthy == 2.5
        assert thresh.critical_to_degraded == 4.5
    
    def test_get_entry_threshold(self):
        """Entry thresholds are returned correctly."""
        thresh = HealthThresholds()
        assert thresh.get_entry_threshold(HealthState.DEGRADED) == 3.0
        assert thresh.get_entry_threshold(HealthState.CRITICAL) == 5.5
        assert thresh.get_entry_threshold(HealthState.HEALTHY) == 0.0
    
    def test_get_exit_threshold(self):
        """Exit thresholds apply hysteresis."""
        thresh = HealthThresholds()
        assert thresh.get_exit_threshold(HealthState.DEGRADED) == 2.0
        assert thresh.get_exit_threshold(HealthState.CRITICAL) == 4.0
    
    def test_get_min_duration(self):
        """Minimum durations are state-specific."""
        thresh = HealthThresholds()
        assert thresh.get_min_duration(HealthState.HEALTHY) == 0.0
        assert thresh.get_min_duration(HealthState.DEGRADED) > 0
        assert thresh.get_min_duration(HealthState.CRITICAL) > 0
        assert thresh.get_min_duration(HealthState.RECOVERING) > 0


# ============================================================================
# HealthTracker Tests
# ============================================================================

class TestHealthTracker:
    """Tests for HealthTracker state machine."""
    
    def test_tracker_initialization(self):
        """Tracker initializes with UNKNOWN state."""
        tracker = HealthTracker(equip_id=1)
        assert tracker.equip_id == 1
        assert tracker.current_state == HealthState.UNKNOWN
        assert len(tracker.history) == 0
    
    def test_update_from_unknown(self):
        """First update transitions from UNKNOWN."""
        tracker = HealthTracker(equip_id=1)
        snapshot = tracker.update(
            timestamp=pd.Timestamp("2024-01-01"),
            fused_z=1.0,  # Below threshold, should be healthy
            confidence=0.9,
        )
        assert tracker.current_state == HealthState.HEALTHY
        assert snapshot.state == HealthState.HEALTHY
        assert len(tracker.history) == 1
    
    def test_update_maintains_healthy(self):
        """Low z-scores maintain HEALTHY state."""
        tracker = HealthTracker(equip_id=1)
        tracker.update(timestamp=pd.Timestamp("2024-01-01"), fused_z=1.0, confidence=0.9)
        tracker.update(timestamp=pd.Timestamp("2024-01-02"), fused_z=1.5, confidence=0.9)
        tracker.update(timestamp=pd.Timestamp("2024-01-03"), fused_z=2.0, confidence=0.9)
        
        assert tracker.current_state == HealthState.HEALTHY
        assert len(tracker.history) == 3
    
    def test_transition_to_degraded(self):
        """High z-score triggers transition to DEGRADED."""
        tracker = HealthTracker(equip_id=1)
        tracker.update(timestamp=pd.Timestamp("2024-01-01"), fused_z=1.0, confidence=0.9)
        tracker.update(timestamp=pd.Timestamp("2024-01-02"), fused_z=4.0, confidence=0.9)
        
        assert tracker.current_state == HealthState.DEGRADED
        # 2 transitions: UNKNOWN->HEALTHY, HEALTHY->DEGRADED
        assert len(tracker.transitions) == 2
        assert tracker.transitions[-1].from_state == HealthState.HEALTHY
        assert tracker.transitions[-1].to_state == HealthState.DEGRADED
    
    def test_transition_to_critical(self):
        """Very high z-score triggers transition to CRITICAL."""
        tracker = HealthTracker(equip_id=1)
        tracker.update(timestamp=pd.Timestamp("2024-01-01"), fused_z=1.0, confidence=0.9)
        tracker.update(timestamp=pd.Timestamp("2024-01-02"), fused_z=7.0, confidence=0.9)
        
        assert tracker.current_state == HealthState.CRITICAL
    
    def test_hysteresis_prevents_oscillation(self):
        """Hysteresis prevents rapid state changes."""
        thresh = HealthThresholds(
            healthy_to_degraded=3.0,
            degraded_to_healthy=2.0,  # Need to go below 2.0 to return to healthy
        )
        tracker = HealthTracker(equip_id=1, thresholds=thresh)
        
        # Start healthy
        tracker.update(timestamp=pd.Timestamp("2024-01-01 00:00"), fused_z=1.0, confidence=0.9)
        assert tracker.current_state == HealthState.HEALTHY
        
        # Go degraded
        tracker.update(timestamp=pd.Timestamp("2024-01-01 01:00"), fused_z=4.0, confidence=0.9)
        assert tracker.current_state == HealthState.DEGRADED
        
        # Drop to 2.5 (below entry threshold 3.0, but above exit threshold 2.0)
        # After min duration
        tracker.update(timestamp=pd.Timestamp("2024-01-01 02:00"), fused_z=2.5, confidence=0.9)
        # Should stay DEGRADED due to hysteresis
        assert tracker.current_state == HealthState.DEGRADED
    
    def test_recovery_detection(self):
        """Recovery from critical goes through RECOVERING state."""
        thresh = HealthThresholds(
            min_critical_duration=0.0,  # No minimum for test
        )
        tracker = HealthTracker(equip_id=1, thresholds=thresh)
        
        # Go critical
        tracker.update(timestamp=pd.Timestamp("2024-01-01 00:00"), fused_z=1.0, confidence=0.9)
        tracker.update(timestamp=pd.Timestamp("2024-01-01 01:00"), fused_z=7.0, confidence=0.9)
        assert tracker.current_state == HealthState.CRITICAL
        
        # Recover - should go to RECOVERING
        tracker.update(timestamp=pd.Timestamp("2024-01-01 05:00"), fused_z=3.5, confidence=0.9)
        assert tracker.current_state == HealthState.RECOVERING
    
    def test_recovery_cooldown(self):
        """Recovery requires cooldown before returning to healthy."""
        thresh = HealthThresholds(
            min_critical_duration=0.0,
            min_recovery_duration=6.0,  # 6 hour cooldown
            critical_to_degraded=4.0,   # Need to drop below 4.0 to exit critical
        )
        tracker = HealthTracker(equip_id=1, thresholds=thresh)
        
        tracker.update(timestamp=pd.Timestamp("2024-01-01 00:00"), fused_z=1.0, confidence=0.9)
        tracker.update(timestamp=pd.Timestamp("2024-01-01 01:00"), fused_z=7.0, confidence=0.9)
        assert tracker.current_state == HealthState.CRITICAL
        
        # Try to recover with z below exit threshold (4.0)
        # This should transition to RECOVERING
        tracker.update(timestamp=pd.Timestamp("2024-01-01 05:00"), fused_z=3.0, confidence=0.9)
        # Should be RECOVERING
        assert tracker.current_state == HealthState.RECOVERING
        
        # After cooldown (6+ hours from entering RECOVERING)
        tracker.update(timestamp=pd.Timestamp("2024-01-01 12:00"), fused_z=1.0, confidence=0.9)
        assert tracker.current_state == HealthState.HEALTHY
    
    def test_worst_detector_tracking(self):
        """Worst detector is tracked in snapshot."""
        tracker = HealthTracker(equip_id=1)
        
        detector_z_scores = {
            "ar1_z": 2.1,
            "pca_spe_z": 3.5,
            "iforest_z": 1.8,
        }
        
        snapshot = tracker.update(
            timestamp=pd.Timestamp("2024-01-01"),
            fused_z=2.5,
            confidence=0.9,
            detector_z_scores=detector_z_scores,
        )
        
        assert snapshot.worst_detector == "pca_spe_z"
        assert snapshot.worst_detector_z == 3.5
    
    def test_to_dict_serialization(self):
        """Tracker can be serialized."""
        tracker = HealthTracker(equip_id=1)
        tracker.update(timestamp=pd.Timestamp("2024-01-01"), fused_z=1.0, confidence=0.9)
        
        d = tracker.to_dict()
        assert d["equip_id"] == 1
        assert d["current_state"] == "HEALTHY"
        assert "state_entered_at" in d
        assert "thresholds" in d
    
    def test_from_dict_deserialization(self):
        """Tracker can be restored from serialized state."""
        tracker = HealthTracker(equip_id=1)
        tracker.update(timestamp=pd.Timestamp("2024-01-01"), fused_z=4.0, confidence=0.9)
        
        d = tracker.to_dict()
        restored = HealthTracker.from_dict(d)
        
        assert restored.equip_id == 1
        assert restored.current_state == HealthState.DEGRADED
    
    def test_get_statistics(self):
        """Tracker provides statistics."""
        tracker = HealthTracker(equip_id=1)
        tracker.update(timestamp=pd.Timestamp("2024-01-01"), fused_z=1.0, confidence=0.9)
        tracker.update(timestamp=pd.Timestamp("2024-01-02"), fused_z=4.0, confidence=0.9)
        
        stats = tracker.get_statistics()
        # 2 transitions: UNKNOWN->HEALTHY, HEALTHY->DEGRADED
        assert stats["transition_count"] == 2
        assert "time_in_state" in stats
        assert "history_length" in stats
    
    def test_get_recent_history(self):
        """Recent history can be retrieved."""
        tracker = HealthTracker(equip_id=1)
        for i in range(20):
            tracker.update(
                timestamp=pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
                fused_z=1.0,
                confidence=0.9,
            )
        
        recent = tracker.get_recent_history(n=5)
        assert len(recent) == 5
    
    def test_reset(self):
        """Tracker can be reset."""
        tracker = HealthTracker(equip_id=1)
        tracker.update(timestamp=pd.Timestamp("2024-01-01"), fused_z=4.0, confidence=0.9)
        
        tracker.reset()
        
        assert tracker.current_state == HealthState.UNKNOWN
        assert len(tracker.history) == 0
        assert len(tracker.transitions) == 0
    
    def test_set_initial_state(self):
        """Initial state can be set explicitly."""
        tracker = HealthTracker(equip_id=1)
        tracker.set_initial_state(
            state=HealthState.DEGRADED,
            timestamp=pd.Timestamp("2024-01-01"),
            confidence=0.7,
        )
        
        assert tracker.current_state == HealthState.DEGRADED
        assert tracker.state_confidence == 0.7


# ============================================================================
# compute_health_from_z_scores Tests
# ============================================================================

class TestComputeHealthFromZScores:
    """Tests for the z-score fusion function."""
    
    def test_empty_returns_zero(self):
        """Empty z_scores returns (0.0, 0.0)."""
        fused_z, confidence = compute_health_from_z_scores({})
        assert fused_z == 0.0
        assert confidence == 0.0
    
    def test_single_detector(self):
        """Single detector returns its value with moderate confidence."""
        fused_z, confidence = compute_health_from_z_scores({"ar1_z": 3.0})
        assert fused_z == 3.0
        assert confidence == 0.5  # Moderate for single detector
    
    def test_multiple_detectors_agreement(self):
        """Agreement increases confidence."""
        fused_z, confidence = compute_health_from_z_scores({
            "ar1_z": 3.0,
            "pca_spe_z": 3.0,
            "iforest_z": 3.0,
        })
        assert fused_z == 3.0
        assert confidence > 0.5  # High agreement
    
    def test_multiple_detectors_disagreement(self):
        """Disagreement decreases confidence."""
        fused_z, confidence = compute_health_from_z_scores({
            "ar1_z": 1.0,
            "pca_spe_z": 5.0,
            "iforest_z": 3.0,
        })
        # Mean of 1, 5, 3 = 3.0
        assert abs(fused_z - 3.0) < 0.01
        # High std = low confidence
        assert confidence < 0.5
    
    def test_nan_values_excluded(self):
        """NaN values are excluded from computation."""
        fused_z, confidence = compute_health_from_z_scores({
            "ar1_z": 3.0,
            "pca_spe_z": np.nan,
            "iforest_z": 3.0,
        })
        assert fused_z == 3.0
    
    def test_all_nan_returns_zero(self):
        """All NaN values returns (0.0, 0.0)."""
        fused_z, confidence = compute_health_from_z_scores({
            "ar1_z": np.nan,
            "pca_spe_z": np.nan,
        })
        assert fused_z == 0.0
        assert confidence == 0.0
    
    def test_custom_weights(self):
        """Custom weights affect fusion."""
        fused_z, confidence = compute_health_from_z_scores(
            z_scores={"ar1_z": 2.0, "pca_spe_z": 4.0},
            weights={"ar1_z": 2.0, "pca_spe_z": 1.0},
        )
        # Weighted mean: (2*2 + 4*1) / 3 = 8/3 = 2.67
        assert abs(fused_z - 2.67) < 0.1


# ============================================================================
# HealthTransition Tests
# ============================================================================

class TestHealthTransition:
    """Tests for HealthTransition tracking."""
    
    def test_transition_creation(self):
        """Transition records state change details."""
        transition = HealthTransition(
            timestamp=pd.Timestamp("2024-01-01"),
            from_state=HealthState.HEALTHY,
            to_state=HealthState.DEGRADED,
            reason="fused_z=4.0",
            confidence=0.9,
            duration_in_previous_hours=2.0,
        )
        assert transition.from_state == HealthState.HEALTHY
        assert transition.to_state == HealthState.DEGRADED
        assert transition.reason == "fused_z=4.0"
        assert transition.duration_in_previous_hours == 2.0


# ============================================================================
# Edge Cases and Robustness
# ============================================================================

class TestHealthStateEdgeCases:
    """Edge case and robustness tests."""
    
    def test_tracker_with_nan_z_score(self):
        """Tracker handles NaN z-scores gracefully."""
        tracker = HealthTracker(equip_id=1)
        tracker.update(timestamp=pd.Timestamp("2024-01-01"), fused_z=1.0, confidence=0.9)
        
        # NaN update - should be treated as 0
        snapshot = tracker.update(timestamp=pd.Timestamp("2024-01-02"), fused_z=np.nan, confidence=0.9)
        # NaN treated as 0 means healthy
        assert tracker.current_state == HealthState.HEALTHY
    
    def test_tracker_with_inf_z_score(self):
        """Tracker handles infinite z-scores gracefully."""
        tracker = HealthTracker(equip_id=1)
        snapshot = tracker.update(
            timestamp=pd.Timestamp("2024-01-01"),
            fused_z=np.inf,
            confidence=0.9,
        )
        # Inf treated as 0
        assert tracker.current_state == HealthState.HEALTHY
    
    def test_rapid_updates(self):
        """Tracker handles many rapid updates."""
        tracker = HealthTracker(equip_id=1)
        
        np.random.seed(42)
        for i in range(100):
            z = np.random.uniform(0, 10)
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
            tracker.update(timestamp=ts, fused_z=z, confidence=0.9)
        
        assert len(tracker.history) == 100
        assert tracker.current_state != HealthState.UNKNOWN
    
    def test_health_pct_healthy_range(self):
        """HEALTHY state gives 80-100% health."""
        tracker = HealthTracker(equip_id=1)
        snapshot = tracker.update(
            timestamp=pd.Timestamp("2024-01-01"),
            fused_z=0.5,  # Very low z
            confidence=0.9,
        )
        assert 80 <= snapshot.health_pct <= 100
    
    def test_health_pct_critical_range(self):
        """CRITICAL state gives 0-40% health."""
        tracker = HealthTracker(equip_id=1)
        snapshot = tracker.update(
            timestamp=pd.Timestamp("2024-01-01"),
            fused_z=8.0,  # High z
            confidence=0.9,
        )
        assert 0 <= snapshot.health_pct <= 40
    
    def test_active_episode_count(self):
        """Active episodes are tracked in snapshot."""
        tracker = HealthTracker(equip_id=1)
        
        fake_episodes = [{"id": 1}, {"id": 2}]
        snapshot = tracker.update(
            timestamp=pd.Timestamp("2024-01-01"),
            fused_z=4.0,
            confidence=0.9,
            active_episodes=fake_episodes,
        )
        
        assert snapshot.active_episode_count == 2
    
    def test_regime_tracking(self):
        """Regime is tracked in snapshot."""
        tracker = HealthTracker(equip_id=1)
        snapshot = tracker.update(
            timestamp=pd.Timestamp("2024-01-01"),
            fused_z=1.0,
            confidence=0.9,
            regime=3,
        )
        
        assert snapshot.regime == 3

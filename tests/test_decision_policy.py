"""
Tests for Decision Policy (P5.8) in core/decision_policy.py

Tests the operational decision contract and policy mapping.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from core.decision_policy import (
    RecommendedAction,
    HealthState,
    RULStatus,
    DecisionContract,
    DecisionPolicy,
    DecisionHistory
)


# =============================================================================
# TEST: RecommendedAction Enum
# =============================================================================


class TestRecommendedAction:
    """Tests for RecommendedAction enum."""
    
    def test_values_exist(self):
        """Test all expected values exist."""
        assert RecommendedAction.NO_ACTION.value == "NO_ACTION"
        assert RecommendedAction.MONITOR.value == "MONITOR"
        assert RecommendedAction.INVESTIGATE.value == "INVESTIGATE"
        assert RecommendedAction.SCHEDULE_MAINTENANCE.value == "SCHEDULE_MAINTENANCE"
        assert RecommendedAction.IMMEDIATE_ACTION.value == "IMMEDIATE_ACTION"
    
    def test_comparison_ordering(self):
        """Test action comparison for escalation."""
        assert RecommendedAction.NO_ACTION < RecommendedAction.MONITOR
        assert RecommendedAction.MONITOR < RecommendedAction.INVESTIGATE
        assert RecommendedAction.INVESTIGATE < RecommendedAction.SCHEDULE_MAINTENANCE
        assert RecommendedAction.SCHEDULE_MAINTENANCE < RecommendedAction.IMMEDIATE_ACTION
    
    def test_max_function(self):
        """Test max() works for escalation."""
        action1 = RecommendedAction.MONITOR
        action2 = RecommendedAction.INVESTIGATE
        assert max(action1, action2) == RecommendedAction.INVESTIGATE
    
    def test_comparison_equality(self):
        """Test equality comparison."""
        assert RecommendedAction.MONITOR <= RecommendedAction.MONITOR
        assert RecommendedAction.MONITOR >= RecommendedAction.MONITOR


# =============================================================================
# TEST: HealthState and RULStatus Enums
# =============================================================================


class TestHealthState:
    """Tests for HealthState enum."""
    
    def test_values_exist(self):
        """Test all expected values exist."""
        assert HealthState.HEALTHY.value == "HEALTHY"
        assert HealthState.DEGRADED.value == "DEGRADED"
        assert HealthState.CRITICAL.value == "CRITICAL"
        assert HealthState.UNKNOWN.value == "UNKNOWN"


class TestRULStatus:
    """Tests for RULStatus enum."""
    
    def test_values_exist(self):
        """Test all expected values exist."""
        assert RULStatus.RELIABLE.value == "RELIABLE"
        assert RULStatus.NOT_RELIABLE.value == "NOT_RELIABLE"
        assert RULStatus.INSUFFICIENT_DATA.value == "INSUFFICIENT_DATA"


# =============================================================================
# TEST: DecisionContract
# =============================================================================


class TestDecisionContract:
    """Tests for DecisionContract dataclass."""
    
    @pytest.fixture
    def sample_contract(self):
        """Create sample decision contract."""
        return DecisionContract(
            timestamp=pd.Timestamp("2024-01-15 10:30:00"),
            equip_id=1,
            state="DEGRADED",
            state_confidence=0.85,
            rul_status="RELIABLE",
            rul_hours=72.0,
            rul_confidence=0.75,
            active_episodes=2,
            worst_episode_severity="MEDIUM",
            recommended_action=RecommendedAction.SCHEDULE_MAINTENANCE,
            action_reason="RUL 72h < 168h threshold",
            action_urgency_hours=72.0,
            system_confidence=0.8,
            limiting_factor="novelty_pressure"
        )
    
    def test_basic_creation(self, sample_contract):
        """Test basic creation."""
        assert sample_contract.state == "DEGRADED"
        assert sample_contract.rul_hours == 72.0
        assert sample_contract.recommended_action == RecommendedAction.SCHEDULE_MAINTENANCE
    
    def test_to_dict(self, sample_contract):
        """Test conversion to dictionary."""
        d = sample_contract.to_dict()
        assert d["state"] == "DEGRADED"
        assert d["rul_hours"] == 72.0
        assert d["recommended_action"] == "SCHEDULE_MAINTENANCE"
        assert d["system_confidence"] == 0.8
    
    def test_to_sql_row(self, sample_contract):
        """Test conversion to SQL row format."""
        row = sample_contract.to_sql_row(run_id=123)
        assert row["RunID"] == 123
        assert row["EquipID"] == 1
        assert row["HealthState"] == "DEGRADED"
        assert row["RecommendedAction"] == "SCHEDULE_MAINTENANCE"
        assert row["RULHours"] == 72.0
    
    def test_is_actionable_true(self, sample_contract):
        """Test is_actionable returns True for actionable actions."""
        assert sample_contract.is_actionable() is True
    
    def test_is_actionable_false(self):
        """Test is_actionable returns False for non-actionable actions."""
        contract = DecisionContract(
            timestamp=pd.Timestamp.now(),
            equip_id=1,
            state="HEALTHY",
            state_confidence=0.95,
            rul_status="RELIABLE",
            rul_hours=500.0,
            rul_confidence=0.9,
            active_episodes=0,
            worst_episode_severity=None,
            recommended_action=RecommendedAction.NO_ACTION,
            action_reason="State is HEALTHY",
            action_urgency_hours=None,
            system_confidence=0.95,
            limiting_factor="none"
        )
        assert contract.is_actionable() is False
    
    def test_repr(self, sample_contract):
        """Test string representation."""
        s = repr(sample_contract)
        assert "DEGRADED" in s
        assert "SCHEDULE_MAINTENANCE" in s


# =============================================================================
# TEST: DecisionPolicy Initialization
# =============================================================================


class TestDecisionPolicyInit:
    """Tests for DecisionPolicy initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        policy = DecisionPolicy()
        assert policy.rul_immediate_hours == 24.0
        assert policy.rul_schedule_hours == 168.0
        assert policy.min_confidence_for_action == 0.5
    
    def test_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        policy = DecisionPolicy(
            rul_immediate_hours=12.0,
            rul_schedule_hours=72.0,
            min_confidence_for_action=0.7
        )
        assert policy.rul_immediate_hours == 12.0
        assert policy.rul_schedule_hours == 72.0
        assert policy.min_confidence_for_action == 0.7
    
    def test_default_state_actions(self):
        """Test default state to action mapping."""
        policy = DecisionPolicy()
        assert policy.state_actions["HEALTHY"] == RecommendedAction.NO_ACTION
        assert policy.state_actions["DEGRADED"] == RecommendedAction.MONITOR
        assert policy.state_actions["CRITICAL"] == RecommendedAction.INVESTIGATE
        assert policy.state_actions["UNKNOWN"] == RecommendedAction.MONITOR


# =============================================================================
# TEST: DecisionPolicy.evaluate()
# =============================================================================


class TestDecisionPolicyEvaluate:
    """Tests for DecisionPolicy.evaluate() method."""
    
    @pytest.fixture
    def policy(self):
        """Create default policy."""
        return DecisionPolicy()
    
    def test_healthy_state_no_action(self, policy):
        """Test HEALTHY state returns NO_ACTION."""
        contract = policy.evaluate(
            health_state="HEALTHY",
            health_confidence=0.95,
            rul_status="RELIABLE",
            rul_hours=500.0,
            rul_confidence=0.9,
            active_episodes=0,
            worst_severity=None,
            system_confidence=0.95,
            limiting_factor="none"
        )
        assert contract.recommended_action == RecommendedAction.NO_ACTION
    
    def test_degraded_state_monitor(self, policy):
        """Test DEGRADED state returns MONITOR."""
        contract = policy.evaluate(
            health_state="DEGRADED",
            health_confidence=0.85,
            rul_status="NOT_RELIABLE",
            rul_hours=None,
            rul_confidence=0.3,
            active_episodes=1,
            worst_severity="LOW",
            system_confidence=0.85,
            limiting_factor="none"
        )
        assert contract.recommended_action == RecommendedAction.MONITOR
    
    def test_critical_state_investigate(self, policy):
        """Test CRITICAL state returns INVESTIGATE."""
        contract = policy.evaluate(
            health_state="CRITICAL",
            health_confidence=0.9,
            rul_status="NOT_RELIABLE",
            rul_hours=None,
            rul_confidence=0.2,
            active_episodes=3,
            worst_severity="HIGH",
            system_confidence=0.9,
            limiting_factor="none"
        )
        assert contract.recommended_action == RecommendedAction.INVESTIGATE
    
    def test_rul_immediate_escalation(self, policy):
        """Test RUL < 24h triggers IMMEDIATE_ACTION."""
        contract = policy.evaluate(
            health_state="HEALTHY",
            health_confidence=0.7,
            rul_status="RELIABLE",
            rul_hours=12.0,  # < 24h
            rul_confidence=0.8,
            active_episodes=0,
            worst_severity=None,
            system_confidence=0.8,
            limiting_factor="none"
        )
        assert contract.recommended_action == RecommendedAction.IMMEDIATE_ACTION
        assert "12h < 24h" in contract.action_reason
    
    def test_rul_schedule_escalation(self, policy):
        """Test RUL < 168h triggers SCHEDULE_MAINTENANCE."""
        contract = policy.evaluate(
            health_state="HEALTHY",
            health_confidence=0.7,
            rul_status="RELIABLE",
            rul_hours=100.0,  # < 168h
            rul_confidence=0.8,
            active_episodes=0,
            worst_severity=None,
            system_confidence=0.8,
            limiting_factor="none"
        )
        assert contract.recommended_action == RecommendedAction.SCHEDULE_MAINTENANCE
        assert "100h < 168h" in contract.action_reason
    
    def test_rul_not_reliable_no_escalation(self, policy):
        """Test unreliable RUL does not trigger escalation."""
        contract = policy.evaluate(
            health_state="HEALTHY",
            health_confidence=0.9,
            rul_status="NOT_RELIABLE",
            rul_hours=12.0,  # Would be IMMEDIATE if reliable
            rul_confidence=0.2,
            active_episodes=0,
            worst_severity=None,
            system_confidence=0.8,
            limiting_factor="rul"
        )
        assert contract.recommended_action == RecommendedAction.NO_ACTION
    
    def test_episode_severity_escalation(self, policy):
        """Test HIGH severity episode triggers INVESTIGATE."""
        contract = policy.evaluate(
            health_state="HEALTHY",
            health_confidence=0.9,
            rul_status="NOT_RELIABLE",
            rul_hours=None,
            rul_confidence=0.2,
            active_episodes=1,
            worst_severity="HIGH",
            system_confidence=0.9,
            limiting_factor="none"
        )
        assert contract.recommended_action == RecommendedAction.INVESTIGATE
        assert "HIGH" in contract.action_reason
    
    def test_low_confidence_dampening(self, policy):
        """Test low confidence dampens action recommendation."""
        contract = policy.evaluate(
            health_state="HEALTHY",
            health_confidence=0.7,
            rul_status="RELIABLE",
            rul_hours=12.0,  # Would be IMMEDIATE normally
            rul_confidence=0.6,
            active_episodes=0,
            worst_severity=None,
            system_confidence=0.4,  # Below 0.5 threshold
            limiting_factor="baseline_quality"
        )
        # Should be dampened from IMMEDIATE to INVESTIGATE
        assert contract.recommended_action == RecommendedAction.INVESTIGATE
        assert "confidence" in contract.action_reason.lower()
    
    def test_urgency_hours_set(self, policy):
        """Test urgency hours is set correctly."""
        contract = policy.evaluate(
            health_state="HEALTHY",
            health_confidence=0.9,
            rul_status="RELIABLE",
            rul_hours=50.0,
            rul_confidence=0.8,
            active_episodes=0,
            worst_severity=None,
            system_confidence=0.9,
            limiting_factor="none"
        )
        assert contract.action_urgency_hours == 50.0
    
    def test_equip_id_and_timestamp(self, policy):
        """Test equip_id and timestamp are set correctly."""
        ts = pd.Timestamp("2024-01-15 10:00:00")
        contract = policy.evaluate(
            health_state="HEALTHY",
            health_confidence=0.9,
            rul_status="RELIABLE",
            rul_hours=500.0,
            rul_confidence=0.9,
            active_episodes=0,
            worst_severity=None,
            system_confidence=0.9,
            limiting_factor="none",
            equip_id=42,
            timestamp=ts
        )
        assert contract.equip_id == 42
        assert contract.timestamp == ts


# =============================================================================
# TEST: DecisionPolicy Configuration
# =============================================================================


class TestDecisionPolicyConfiguration:
    """Tests for policy configuration methods."""
    
    def test_get_action_for_state(self):
        """Test get_action_for_state method."""
        policy = DecisionPolicy()
        assert policy.get_action_for_state("HEALTHY") == RecommendedAction.NO_ACTION
        assert policy.get_action_for_state("UNKNOWN_STATE") == RecommendedAction.MONITOR
    
    def test_set_action_for_state(self):
        """Test set_action_for_state method."""
        policy = DecisionPolicy()
        policy.set_action_for_state("HEALTHY", RecommendedAction.MONITOR)
        assert policy.state_actions["HEALTHY"] == RecommendedAction.MONITOR
    
    def test_to_dict(self):
        """Test policy serialization."""
        policy = DecisionPolicy()
        d = policy.to_dict()
        assert "state_actions" in d
        assert "rul_immediate_hours" in d
        assert d["rul_immediate_hours"] == 24.0
        assert d["state_actions"]["HEALTHY"] == "NO_ACTION"
    
    def test_from_dict(self):
        """Test policy deserialization."""
        config = {
            "state_actions": {
                "HEALTHY": "MONITOR",
                "DEGRADED": "INVESTIGATE"
            },
            "rul_immediate_hours": 12.0,
            "min_confidence_for_action": 0.6
        }
        policy = DecisionPolicy.from_dict(config)
        assert policy.state_actions["HEALTHY"] == RecommendedAction.MONITOR
        assert policy.state_actions["DEGRADED"] == RecommendedAction.INVESTIGATE
        assert policy.rul_immediate_hours == 12.0
        assert policy.min_confidence_for_action == 0.6


# =============================================================================
# TEST: DecisionHistory
# =============================================================================


class TestDecisionHistory:
    """Tests for DecisionHistory class."""
    
    @pytest.fixture
    def sample_decisions(self):
        """Create sample decisions."""
        decisions = []
        base_time = pd.Timestamp("2024-01-15 10:00:00")
        actions = [
            RecommendedAction.NO_ACTION,
            RecommendedAction.MONITOR,
            RecommendedAction.MONITOR,
            RecommendedAction.INVESTIGATE,
            RecommendedAction.SCHEDULE_MAINTENANCE
        ]
        for i, action in enumerate(actions):
            decisions.append(DecisionContract(
                timestamp=base_time + pd.Timedelta(hours=i),
                equip_id=1,
                state="DEGRADED",
                state_confidence=0.8,
                rul_status="RELIABLE",
                rul_hours=100 - i*10,
                rul_confidence=0.7,
                active_episodes=i,
                worst_episode_severity="MEDIUM" if i > 0 else None,
                recommended_action=action,
                action_reason="Test",
                action_urgency_hours=None,
                system_confidence=0.8,
                limiting_factor="none"
            ))
        return decisions
    
    def test_add_and_get_recent(self, sample_decisions):
        """Test adding and retrieving decisions."""
        history = DecisionHistory()
        for d in sample_decisions:
            history.add(d)
        
        recent = history.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].recommended_action == RecommendedAction.SCHEDULE_MAINTENANCE
    
    def test_max_history_limit(self):
        """Test max history limit is enforced."""
        history = DecisionHistory(max_history=5)
        
        for i in range(10):
            history.add(DecisionContract(
                timestamp=pd.Timestamp.now() + pd.Timedelta(minutes=i),
                equip_id=1,
                state="HEALTHY",
                state_confidence=0.9,
                rul_status="RELIABLE",
                rul_hours=500.0,
                rul_confidence=0.9,
                active_episodes=0,
                worst_episode_severity=None,
                recommended_action=RecommendedAction.NO_ACTION,
                action_reason="Test",
                action_urgency_hours=None,
                system_confidence=0.9,
                limiting_factor="none"
            ))
        
        assert len(history.decisions) == 5
    
    def test_get_action_counts(self, sample_decisions):
        """Test action count aggregation."""
        history = DecisionHistory()
        for d in sample_decisions:
            history.add(d)
        
        counts = history.get_action_counts()
        assert counts["NO_ACTION"] == 1
        assert counts["MONITOR"] == 2
        assert counts["INVESTIGATE"] == 1
        assert counts["SCHEDULE_MAINTENANCE"] == 1
    
    def test_has_escalating_trend_true(self, sample_decisions):
        """Test escalating trend detection."""
        history = DecisionHistory()
        for d in sample_decisions:
            history.add(d)
        
        assert history.has_escalating_trend(window=5) is True
    
    def test_has_escalating_trend_false(self):
        """Test no escalating trend when stable."""
        history = DecisionHistory()
        base_time = pd.Timestamp("2024-01-15 10:00:00")
        
        for i in range(5):
            history.add(DecisionContract(
                timestamp=base_time + pd.Timedelta(hours=i),
                equip_id=1,
                state="HEALTHY",
                state_confidence=0.9,
                rul_status="RELIABLE",
                rul_hours=500.0,
                rul_confidence=0.9,
                active_episodes=0,
                worst_episode_severity=None,
                recommended_action=RecommendedAction.MONITOR,  # Same action
                action_reason="Test",
                action_urgency_hours=None,
                system_confidence=0.9,
                limiting_factor="none"
            ))
        
        assert history.has_escalating_trend(window=5) is False
    
    def test_get_time_at_action(self, sample_decisions):
        """Test time calculation at action level."""
        history = DecisionHistory()
        for d in sample_decisions:
            history.add(d)
        
        # MONITOR appears twice with 1h between decisions
        monitor_hours = history.get_time_at_action(RecommendedAction.MONITOR)
        assert monitor_hours == pytest.approx(2.0, rel=0.01)  # 2 hours at MONITOR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Decision Policy for ACM v11.0.0 (P5.8)

Operational Decision Contract: Maps analytics outputs to discrete operator actions.

Key Features:
- DecisionContract: Compact output for operators to act on
- RecommendedAction: Discrete action types (NO_ACTION, MONITOR, INVESTIGATE, etc.)
- DecisionPolicy: Configurable mapping from analytics to actions
- Action escalation based on RUL, episodes, and health state

This is the ONLY output operators should act on. It decouples analytics
from operational decisions, allowing policy changes without retraining models.

Usage:
    policy = DecisionPolicy()
    
    contract = policy.evaluate(
        health_state="DEGRADED",
        health_confidence=0.85,
        rul_status="RELIABLE",
        rul_hours=72.0,
        ...
    )
    
    print(contract.recommended_action)  # SCHEDULE_MAINTENANCE
    print(contract.action_reason)       # "RUL 72h < 7d threshold"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd


class RecommendedAction(Enum):
    """Discrete actions for operators.
    
    Ordered by severity for escalation comparison.
    """
    NO_ACTION = "NO_ACTION"                      # Normal operation
    MONITOR = "MONITOR"                          # Increased monitoring frequency
    INVESTIGATE = "INVESTIGATE"                  # Requires operator review
    SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"  # Plan maintenance window
    IMMEDIATE_ACTION = "IMMEDIATE_ACTION"        # Urgent intervention required
    
    def __lt__(self, other: "RecommendedAction") -> bool:
        """Enable comparison for escalation."""
        order = [
            RecommendedAction.NO_ACTION,
            RecommendedAction.MONITOR,
            RecommendedAction.INVESTIGATE,
            RecommendedAction.SCHEDULE_MAINTENANCE,
            RecommendedAction.IMMEDIATE_ACTION
        ]
        return order.index(self) < order.index(other)
    
    def __gt__(self, other: "RecommendedAction") -> bool:
        return other < self
    
    def __le__(self, other: "RecommendedAction") -> bool:
        return self == other or self < other
    
    def __ge__(self, other: "RecommendedAction") -> bool:
        return self == other or self > other


class HealthState(Enum):
    """Equipment health states."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class RULStatus(Enum):
    """RUL reliability status."""
    RELIABLE = "RELIABLE"
    NOT_RELIABLE = "NOT_RELIABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class DecisionContract:
    """
    Compact operational output contract.
    
    This is the ONLY output operators should act on.
    Decouples analytics from operational decisions.
    
    Attributes:
        timestamp: When this decision was made
        equip_id: Equipment identifier
        
        state: Current health state (HEALTHY, DEGRADED, CRITICAL, UNKNOWN)
        state_confidence: Confidence in state assessment (0-1)
        
        rul_status: RUL reliability (RELIABLE, NOT_RELIABLE, INSUFFICIENT_DATA)
        rul_hours: Predicted hours to failure (if reliable)
        rul_confidence: Confidence in RUL prediction (0-1)
        
        active_episodes: Number of currently active anomaly episodes
        worst_episode_severity: Severity of worst active episode
        
        recommended_action: What the operator should do
        action_reason: Human-readable explanation for action
        action_urgency_hours: Time available before action needed
        
        system_confidence: Overall confidence in this output (0-1)
        limiting_factor: What's reducing confidence
    """
    timestamp: pd.Timestamp
    equip_id: int
    
    # State summary
    state: str
    state_confidence: float
    
    # RUL summary
    rul_status: str
    rul_hours: Optional[float]
    rul_confidence: float
    
    # Episode summary
    active_episodes: int
    worst_episode_severity: Optional[str]
    
    # Action recommendation
    recommended_action: RecommendedAction
    action_reason: str
    action_urgency_hours: Optional[float]
    
    # System health
    system_confidence: float
    limiting_factor: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": str(self.timestamp),
            "equip_id": self.equip_id,
            "state": self.state,
            "state_confidence": round(self.state_confidence, 3),
            "rul_status": self.rul_status,
            "rul_hours": round(self.rul_hours, 1) if self.rul_hours else None,
            "rul_confidence": round(self.rul_confidence, 3),
            "active_episodes": self.active_episodes,
            "worst_episode_severity": self.worst_episode_severity,
            "recommended_action": self.recommended_action.value,
            "action_reason": self.action_reason,
            "action_urgency_hours": round(self.action_urgency_hours, 1) if self.action_urgency_hours else None,
            "system_confidence": round(self.system_confidence, 3),
            "limiting_factor": self.limiting_factor
        }
    
    def to_sql_row(self, run_id: int) -> Dict[str, Any]:
        """Convert to SQL row format for ACM_DecisionOutput table."""
        return {
            "RunID": run_id,
            "EquipID": self.equip_id,
            "Timestamp": self.timestamp,
            "HealthState": self.state,
            "StateConfidence": self.state_confidence,
            "RULStatus": self.rul_status,
            "RULHours": self.rul_hours,
            "RULConfidence": self.rul_confidence,
            "ActiveEpisodes": self.active_episodes,
            "WorstEpisodeSeverity": self.worst_episode_severity,
            "RecommendedAction": self.recommended_action.value,
            "ActionReason": self.action_reason[:500] if self.action_reason else "",
            "ActionUrgencyHours": self.action_urgency_hours,
            "SystemConfidence": self.system_confidence,
            "LimitingFactor": self.limiting_factor[:50] if self.limiting_factor else ""
        }
    
    def is_actionable(self) -> bool:
        """Check if this contract requires operator attention."""
        return self.recommended_action in [
            RecommendedAction.INVESTIGATE,
            RecommendedAction.SCHEDULE_MAINTENANCE,
            RecommendedAction.IMMEDIATE_ACTION
        ]
    
    def __repr__(self) -> str:
        return (
            f"DecisionContract(state={self.state}, action={self.recommended_action.value}, "
            f"confidence={self.system_confidence:.0%})"
        )


@dataclass
class DecisionPolicy:
    """
    Maps analytics outputs to operational decisions.
    
    Can be modified WITHOUT retraining models. This allows operators to
    adjust thresholds and escalation rules independently of the analytics.
    
    Attributes:
        state_actions: Mapping from health state to default action
        rul_immediate_hours: RUL threshold for immediate action (default 24h)
        rul_schedule_hours: RUL threshold for scheduled maintenance (default 168h/7d)
        min_confidence_for_action: Minimum confidence to recommend action (default 0.5)
    """
    # State → Action mapping
    state_actions: Dict[str, RecommendedAction] = field(default_factory=lambda: {
        "HEALTHY": RecommendedAction.NO_ACTION,
        "DEGRADED": RecommendedAction.MONITOR,
        "CRITICAL": RecommendedAction.INVESTIGATE,
        "UNKNOWN": RecommendedAction.MONITOR,
    })
    
    # RUL thresholds for escalation
    rul_immediate_hours: float = 24.0
    rul_schedule_hours: float = 168.0  # 7 days
    
    # Confidence thresholds
    min_confidence_for_action: float = 0.5
    
    # Episode severity escalation
    episode_severity_escalation: Dict[str, RecommendedAction] = field(default_factory=lambda: {
        "LOW": RecommendedAction.MONITOR,
        "MEDIUM": RecommendedAction.MONITOR,
        "HIGH": RecommendedAction.INVESTIGATE,
        "CRITICAL": RecommendedAction.INVESTIGATE,
    })
    
    def evaluate(
        self,
        health_state: str,
        health_confidence: float,
        rul_status: str,
        rul_hours: Optional[float],
        rul_confidence: float,
        active_episodes: int,
        worst_severity: Optional[str],
        system_confidence: float,
        limiting_factor: str,
        equip_id: int = 0,
        timestamp: Optional[pd.Timestamp] = None
    ) -> DecisionContract:
        """
        Apply policy to determine recommended action.
        
        Decision logic:
        1. Start with state-based action
        2. Escalate based on RUL if reliable
        3. Escalate based on episode severity
        4. Dampen if low confidence
        
        Args:
            health_state: Current health state (HEALTHY, DEGRADED, CRITICAL, UNKNOWN)
            health_confidence: Confidence in health state (0-1)
            rul_status: RUL reliability status
            rul_hours: Predicted hours to failure (if reliable)
            rul_confidence: Confidence in RUL (0-1)
            active_episodes: Number of active anomaly episodes
            worst_severity: Severity of worst active episode
            system_confidence: Overall system confidence (0-1)
            limiting_factor: What's reducing confidence
            equip_id: Equipment identifier
            timestamp: When this decision is made (default: now)
        
        Returns:
            DecisionContract with recommended action
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Step 1: Start with state-based action
        action = self.state_actions.get(health_state, RecommendedAction.MONITOR)
        reason = f"State is {health_state}"
        urgency: Optional[float] = None
        
        # Step 2: Escalate based on RUL if reliable
        if rul_status == "RELIABLE" and rul_hours is not None:
            if rul_hours < self.rul_immediate_hours:
                action = max(action, RecommendedAction.IMMEDIATE_ACTION)
                reason = f"RUL {rul_hours:.0f}h < {self.rul_immediate_hours:.0f}h threshold"
                urgency = rul_hours
            elif rul_hours < self.rul_schedule_hours:
                action = max(action, RecommendedAction.SCHEDULE_MAINTENANCE)
                reason = f"RUL {rul_hours:.0f}h < {self.rul_schedule_hours:.0f}h threshold"
                urgency = rul_hours
        
        # Step 3: Escalate based on episodes
        if worst_severity is not None:
            episode_action = self.episode_severity_escalation.get(
                worst_severity, RecommendedAction.MONITOR
            )
            if episode_action > action:
                action = episode_action
                reason = f"{worst_severity} severity episode active"
        
        # Step 4: Dampen if low confidence
        if system_confidence < self.min_confidence_for_action:
            if action in [RecommendedAction.IMMEDIATE_ACTION, RecommendedAction.SCHEDULE_MAINTENANCE]:
                original_action = action
                action = RecommendedAction.INVESTIGATE
                reason += f" (confidence {system_confidence:.0%} too low for {original_action.value})"
        
        return DecisionContract(
            timestamp=timestamp,
            equip_id=equip_id,
            state=health_state,
            state_confidence=health_confidence,
            rul_status=rul_status,
            rul_hours=rul_hours,
            rul_confidence=rul_confidence,
            active_episodes=active_episodes,
            worst_episode_severity=worst_severity,
            recommended_action=action,
            action_reason=reason,
            action_urgency_hours=urgency,
            system_confidence=system_confidence,
            limiting_factor=limiting_factor
        )
    
    def get_action_for_state(self, state: str) -> RecommendedAction:
        """Get default action for a health state."""
        return self.state_actions.get(state, RecommendedAction.MONITOR)
    
    def set_action_for_state(self, state: str, action: RecommendedAction) -> None:
        """Override action for a health state."""
        self.state_actions[state] = action
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for serialization."""
        return {
            "state_actions": {k: v.value for k, v in self.state_actions.items()},
            "rul_immediate_hours": self.rul_immediate_hours,
            "rul_schedule_hours": self.rul_schedule_hours,
            "min_confidence_for_action": self.min_confidence_for_action,
            "episode_severity_escalation": {k: v.value for k, v in self.episode_severity_escalation.items()}
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DecisionPolicy":
        """Create policy from dictionary."""
        policy = cls()
        
        if "state_actions" in config:
            policy.state_actions = {
                k: RecommendedAction(v) for k, v in config["state_actions"].items()
            }
        
        if "rul_immediate_hours" in config:
            policy.rul_immediate_hours = float(config["rul_immediate_hours"])
        
        if "rul_schedule_hours" in config:
            policy.rul_schedule_hours = float(config["rul_schedule_hours"])
        
        if "min_confidence_for_action" in config:
            policy.min_confidence_for_action = float(config["min_confidence_for_action"])
        
        if "episode_severity_escalation" in config:
            policy.episode_severity_escalation = {
                k: RecommendedAction(v) for k, v in config["episode_severity_escalation"].items()
            }
        
        return policy


class DecisionHistory:
    """
    Track decision history for trend analysis and alerting.
    
    Attributes:
        max_history: Maximum number of decisions to retain
        decisions: List of historical decisions
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.decisions: List[DecisionContract] = []
    
    def add(self, decision: DecisionContract) -> None:
        """Add a decision to history."""
        self.decisions.append(decision)
        if len(self.decisions) > self.max_history:
            self.decisions = self.decisions[-self.max_history:]
    
    def get_recent(self, n: int = 10) -> List[DecisionContract]:
        """Get most recent N decisions."""
        return self.decisions[-n:]
    
    def get_action_counts(self) -> Dict[str, int]:
        """Get count of each action type in history."""
        counts: Dict[str, int] = {}
        for d in self.decisions:
            action = d.recommended_action.value
            counts[action] = counts.get(action, 0) + 1
        return counts
    
    def has_escalating_trend(self, window: int = 5) -> bool:
        """Check if actions are escalating over recent window."""
        if len(self.decisions) < window:
            return False
        
        recent = self.decisions[-window:]
        for i in range(1, len(recent)):
            if recent[i].recommended_action > recent[i-1].recommended_action:
                return True
        return False
    
    def get_time_at_action(self, action: RecommendedAction) -> float:
        """Get total hours spent at a specific action level."""
        total_hours = 0.0
        for i in range(len(self.decisions) - 1):
            if self.decisions[i].recommended_action == action:
                delta = self.decisions[i+1].timestamp - self.decisions[i].timestamp
                total_hours += delta.total_seconds() / 3600
        return total_hours

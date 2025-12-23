"""
ACM v11.0.0 - Health State Module.

Implements time-evolving health state machine with:
- Discrete states (HEALTHY, DEGRADED, CRITICAL, UNKNOWN, RECOVERING)
- Hysteresis to prevent oscillation
- Confidence attached to every state
- Recovery logic with cooldown
- State persistence across runs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd


class HealthState(Enum):
    """
    Discrete health states with clear semantics.
    
    State meanings:
    - HEALTHY: Normal operation, no action needed
    - DEGRADED: Deviation detected, monitor closely
    - CRITICAL: Significant issue, action required
    - UNKNOWN: Insufficient data to determine state
    - RECOVERING: Returning from degraded/critical, in cooldown
    """
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    RECOVERING = "RECOVERING"
    
    @property
    def severity_level(self) -> int:
        """Numeric severity for comparison."""
        return {
            HealthState.HEALTHY: 0,
            HealthState.RECOVERING: 1,
            HealthState.DEGRADED: 2,
            HealthState.CRITICAL: 3,
            HealthState.UNKNOWN: -1,
        }[self]
    
    def is_actionable(self) -> bool:
        """Whether this state requires operator attention."""
        return self in (HealthState.DEGRADED, HealthState.CRITICAL)


@dataclass
class HealthSnapshot:
    """
    Point-in-time health assessment.
    
    Captures all relevant information about equipment health
    at a specific moment, including confidence and transition info.
    """
    timestamp: pd.Timestamp
    state: HealthState
    health_pct: float           # 0-100 for backwards compatibility
    confidence: float           # 0-1, how confident in this state
    state_duration_hours: float # How long in current state
    
    # Transition info
    previous_state: Optional[HealthState] = None
    transition_reason: Optional[str] = None
    
    # Underlying signals
    fused_z_mean: float = 0.0
    fused_z_max: float = 0.0
    active_episode_count: int = 0
    worst_detector: str = ""
    worst_detector_z: float = 0.0
    
    # Equipment context
    equip_id: int = 0
    regime: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for SQL persistence."""
        return {
            "Timestamp": self.timestamp,
            "HealthState": self.state.value,
            "HealthPct": round(self.health_pct, 2),
            "Confidence": round(self.confidence, 4),
            "StateDurationHours": round(self.state_duration_hours, 2),
            "PreviousState": self.previous_state.value if self.previous_state else None,
            "TransitionReason": self.transition_reason,
            "FusedZMean": round(self.fused_z_mean, 4),
            "FusedZMax": round(self.fused_z_max, 4),
            "ActiveEpisodeCount": self.active_episode_count,
            "WorstDetector": self.worst_detector,
            "WorstDetectorZ": round(self.worst_detector_z, 4),
            "EquipID": self.equip_id,
            "Regime": self.regime,
        }


@dataclass
class HealthTransition:
    """Record of a health state transition."""
    timestamp: pd.Timestamp
    from_state: HealthState
    to_state: HealthState
    reason: str
    confidence: float
    duration_in_previous_hours: float


@dataclass 
class HealthThresholds:
    """
    Configurable thresholds for health state transitions.
    
    Uses hysteresis: thresholds differ based on direction
    to prevent rapid oscillation.
    """
    # Z-score thresholds for state entry (going worse)
    healthy_to_degraded: float = 3.0
    degraded_to_critical: float = 5.5
    
    # Z-score thresholds for state exit (improving)
    degraded_to_healthy: float = 2.0  # Lower to go back
    critical_to_degraded: float = 4.0  # Lower to go back
    
    # Minimum time in state before transition (hours)
    min_healthy_duration: float = 0.0      # Can leave immediately
    min_degraded_duration: float = 0.5     # 30 min minimum
    min_critical_duration: float = 2.0     # 2 hour minimum
    min_recovery_duration: float = 6.0     # 6 hour cooldown
    
    # Health percentage mappings
    healthy_min_pct: float = 80.0
    degraded_min_pct: float = 40.0
    critical_min_pct: float = 0.0
    
    def get_entry_threshold(self, target: HealthState) -> float:
        """Get z-score threshold to enter a state (getting worse)."""
        if target == HealthState.DEGRADED:
            return self.healthy_to_degraded
        elif target == HealthState.CRITICAL:
            return self.degraded_to_critical
        return 0.0
    
    def get_exit_threshold(self, current: HealthState) -> float:
        """Get z-score threshold to exit a state (improving)."""
        if current == HealthState.DEGRADED:
            return self.degraded_to_healthy
        elif current == HealthState.CRITICAL:
            return self.critical_to_degraded
        return float('inf')
    
    def get_min_duration(self, state: HealthState) -> float:
        """Get minimum duration in hours for a state."""
        return {
            HealthState.HEALTHY: self.min_healthy_duration,
            HealthState.DEGRADED: self.min_degraded_duration,
            HealthState.CRITICAL: self.min_critical_duration,
            HealthState.RECOVERING: self.min_recovery_duration,
            HealthState.UNKNOWN: 0.0,
        }[state]
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "HealthThresholds":
        """Create thresholds from config dict."""
        health_cfg = cfg.get("health", {})
        return cls(
            healthy_to_degraded=float(health_cfg.get("threshold_degraded", 3.0)),
            degraded_to_critical=float(health_cfg.get("threshold_critical", 5.5)),
            degraded_to_healthy=float(health_cfg.get("threshold_healthy", 2.0)),
            critical_to_degraded=float(health_cfg.get("threshold_recovering", 4.0)),
            min_degraded_duration=float(health_cfg.get("min_degraded_hours", 0.5)),
            min_critical_duration=float(health_cfg.get("min_critical_hours", 2.0)),
            min_recovery_duration=float(health_cfg.get("recovery_cooldown_hours", 6.0)),
        )


class HealthTracker:
    """
    Time-evolving health state machine.
    
    Key improvements over v10:
    1. Explicit state machine with hysteresis
    2. Confidence attached to every state
    3. Recovery logic with cooldown
    4. State persistence across runs
    5. UNKNOWN state when data is insufficient
    
    Usage:
        tracker = HealthTracker()
        tracker.set_initial_state(HealthState.UNKNOWN)
        
        for row in data:
            snapshot = tracker.update(
                timestamp=row.Timestamp,
                fused_z=row.fused_z,
                confidence=row.confidence,
                detector_z_scores={"ar1_z": 2.1, "pca_spe_z": 3.5},
            )
            # snapshot contains current state, health %, confidence
    """
    
    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        equip_id: int = 0,
    ):
        """
        Initialize health tracker.
        
        Args:
            thresholds: Custom thresholds or use defaults
            equip_id: Equipment ID for context
        """
        self.thresholds = thresholds or HealthThresholds()
        self.equip_id = equip_id
        
        # Current state
        self.current_state = HealthState.UNKNOWN
        self.state_entered_at: Optional[pd.Timestamp] = None
        self.state_confidence: float = 0.0
        
        # History tracking
        self.history: List[HealthSnapshot] = []
        self.transitions: List[HealthTransition] = []
        
        # Statistics
        self._transition_count = 0
        self._time_in_state: Dict[HealthState, float] = {s: 0.0 for s in HealthState}
    
    def set_initial_state(
        self,
        state: HealthState,
        timestamp: Optional[pd.Timestamp] = None,
        confidence: float = 0.5,
    ) -> None:
        """
        Set the initial state (e.g., from persisted state or cold start).
        
        Args:
            state: Initial health state
            timestamp: When this state was entered
            confidence: Confidence in initial state
        """
        self.current_state = state
        self.state_entered_at = timestamp or pd.Timestamp.now()
        self.state_confidence = confidence
    
    def update(
        self,
        timestamp: pd.Timestamp,
        fused_z: float,
        confidence: float,
        detector_z_scores: Optional[Dict[str, float]] = None,
        active_episodes: Optional[List[Any]] = None,
        regime: int = -1,
    ) -> HealthSnapshot:
        """
        Update health state based on new evidence.
        
        Implements hysteresis: thresholds differ based on direction.
        
        Args:
            timestamp: Current timestamp
            fused_z: Fused anomaly z-score
            confidence: Confidence in the fused score (0-1)
            detector_z_scores: Per-detector z-scores for attribution
            active_episodes: List of currently active episodes
            regime: Current operating regime
            
        Returns:
            HealthSnapshot with current state assessment
        """
        if not np.isfinite(fused_z):
            fused_z = 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Determine target state from z-score
        target_state = self._z_to_state(fused_z)
        
        # Check if transition is allowed (duration, hysteresis)
        new_state, transition_allowed = self._check_transition(
            target_state, timestamp, fused_z
        )
        
        # Handle recovery state
        if (self.current_state == HealthState.CRITICAL and 
            new_state == HealthState.DEGRADED):
            new_state = HealthState.RECOVERING
        
        # Find worst detector
        worst_detector = ""
        worst_detector_z = 0.0
        if detector_z_scores:
            for det, z in detector_z_scores.items():
                if np.isfinite(z) and z > worst_detector_z:
                    worst_detector = det
                    worst_detector_z = z
        
        # Compute state duration
        state_duration = self._state_duration(timestamp)
        
        # Determine transition info
        is_transition = new_state != self.current_state
        previous = self.current_state if is_transition else None
        reason = f"fused_z={fused_z:.2f}" if is_transition else None
        
        # Create snapshot
        snapshot = HealthSnapshot(
            timestamp=timestamp,
            state=new_state,
            health_pct=self._state_to_pct(new_state, fused_z),
            confidence=self._compute_state_confidence(new_state, confidence, fused_z),
            state_duration_hours=state_duration if not is_transition else 0.0,
            previous_state=previous,
            transition_reason=reason,
            fused_z_mean=fused_z,
            fused_z_max=fused_z,  # For single point, same as mean
            active_episode_count=len(active_episodes) if active_episodes else 0,
            worst_detector=worst_detector,
            worst_detector_z=worst_detector_z,
            equip_id=self.equip_id,
            regime=regime,
        )
        
        # Record transition if occurred
        if is_transition:
            transition = HealthTransition(
                timestamp=timestamp,
                from_state=self.current_state,
                to_state=new_state,
                reason=reason or "",
                confidence=confidence,
                duration_in_previous_hours=state_duration,
            )
            self.transitions.append(transition)
            self._transition_count += 1
            
            # Update time tracking
            if self.state_entered_at is not None:
                self._time_in_state[self.current_state] += state_duration
            
            # Update current state
            self.current_state = new_state
            self.state_entered_at = timestamp
            self.state_confidence = confidence
        
        self.history.append(snapshot)
        return snapshot
    
    def _z_to_state(self, z: float) -> HealthState:
        """Map z-score to target health state."""
        if z >= self.thresholds.degraded_to_critical:
            return HealthState.CRITICAL
        elif z >= self.thresholds.healthy_to_degraded:
            return HealthState.DEGRADED
        else:
            return HealthState.HEALTHY
    
    def _check_transition(
        self,
        target: HealthState,
        timestamp: pd.Timestamp,
        z_score: float,
    ) -> Tuple[HealthState, bool]:
        """
        Check if state transition is allowed.
        
        Implements:
        1. Minimum duration in current state
        2. Hysteresis (different thresholds for up/down)
        3. Cooldown after critical state
        
        Returns:
            (new_state, transition_allowed)
        """
        # If in UNKNOWN, allow any transition
        if self.current_state == HealthState.UNKNOWN:
            return target, True
        
        # Same state - no transition needed
        if target == self.current_state:
            return self.current_state, False
        
        # Check minimum duration
        duration_hours = self._state_duration(timestamp)
        min_duration = self.thresholds.get_min_duration(self.current_state)
        
        if duration_hours < min_duration:
            return self.current_state, False  # Stay in current state
        
        # RECOVERING must complete cooldown
        if self.current_state == HealthState.RECOVERING:
            cooldown = self.thresholds.min_recovery_duration
            if duration_hours < cooldown:
                return HealthState.RECOVERING, False
            # After cooldown, can transition to healthy
            if target == HealthState.HEALTHY:
                return HealthState.HEALTHY, True
            return target, True
        
        # Apply hysteresis for improvement transitions
        if target.severity_level < self.current_state.severity_level:
            # Improving - need to pass exit threshold
            exit_threshold = self.thresholds.get_exit_threshold(self.current_state)
            if z_score > exit_threshold:
                return self.current_state, False  # Not improved enough
        
        return target, True
    
    def _state_duration(self, current_time: pd.Timestamp) -> float:
        """Get duration in current state in hours."""
        if self.state_entered_at is None:
            return 0.0
        delta = current_time - self.state_entered_at
        return delta.total_seconds() / 3600
    
    def _state_to_pct(self, state: HealthState, z: float) -> float:
        """
        Convert state to backwards-compatible percentage.
        
        Mapping:
        - HEALTHY: 80-100%
        - DEGRADED: 40-80%
        - CRITICAL: 0-40%
        - RECOVERING: 50%
        - UNKNOWN: 50%
        """
        if state == HealthState.HEALTHY:
            # 80-100%, decreases as z approaches threshold
            z_factor = min(z / self.thresholds.healthy_to_degraded, 1.0)
            return 100 - (z_factor * 20)
        
        elif state == HealthState.DEGRADED:
            # 40-80%, linear interpolation
            z_range = self.thresholds.degraded_to_critical - self.thresholds.healthy_to_degraded
            z_normalized = (z - self.thresholds.healthy_to_degraded) / max(z_range, 0.1)
            z_normalized = np.clip(z_normalized, 0.0, 1.0)
            return 80 - (z_normalized * 40)
        
        elif state == HealthState.CRITICAL:
            # 0-40%, linear interpolation
            z_above = z - self.thresholds.degraded_to_critical
            pct = 40 - min(z_above * 8, 40)
            return max(pct, 0.0)
        
        elif state == HealthState.RECOVERING:
            return 50.0  # Fixed during recovery
        
        return 50.0  # UNKNOWN
    
    def _compute_state_confidence(
        self,
        state: HealthState,
        base_confidence: float,
        z: float,
    ) -> float:
        """
        Compute confidence in the current state assessment.
        
        Confidence is reduced when:
        - Z-score is near a threshold (borderline)
        - State duration is very short
        - Base confidence from fusion is low
        """
        confidence = base_confidence
        
        # Reduce confidence near thresholds
        thresholds = [
            self.thresholds.healthy_to_degraded,
            self.thresholds.degraded_to_critical,
        ]
        for thresh in thresholds:
            distance = abs(z - thresh)
            if distance < 0.5:
                # Within 0.5 of threshold - reduce confidence
                confidence *= (0.5 + distance)
        
        # Unknown state always has lower confidence
        if state == HealthState.UNKNOWN:
            confidence *= 0.5
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    # ===== Persistence Methods =====
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state for persistence."""
        return {
            "current_state": self.current_state.value,
            "state_entered_at": self.state_entered_at.isoformat() if self.state_entered_at else None,
            "state_confidence": self.state_confidence,
            "equip_id": self.equip_id,
            "transition_count": self._transition_count,
            "time_in_state": {s.value: t for s, t in self._time_in_state.items()},
            "thresholds": {
                "healthy_to_degraded": self.thresholds.healthy_to_degraded,
                "degraded_to_critical": self.thresholds.degraded_to_critical,
                "degraded_to_healthy": self.thresholds.degraded_to_healthy,
                "critical_to_degraded": self.thresholds.critical_to_degraded,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthTracker":
        """Restore tracker from persisted state."""
        thresholds = HealthThresholds(
            healthy_to_degraded=data.get("thresholds", {}).get("healthy_to_degraded", 3.0),
            degraded_to_critical=data.get("thresholds", {}).get("degraded_to_critical", 5.5),
            degraded_to_healthy=data.get("thresholds", {}).get("degraded_to_healthy", 2.0),
            critical_to_degraded=data.get("thresholds", {}).get("critical_to_degraded", 4.0),
        )
        
        tracker = cls(
            thresholds=thresholds,
            equip_id=data.get("equip_id", 0),
        )
        
        state_str = data.get("current_state", "UNKNOWN")
        try:
            tracker.current_state = HealthState(state_str)
        except ValueError:
            tracker.current_state = HealthState.UNKNOWN
        
        entered_at = data.get("state_entered_at")
        if entered_at:
            tracker.state_entered_at = pd.Timestamp(entered_at)
        
        tracker.state_confidence = data.get("state_confidence", 0.5)
        tracker._transition_count = data.get("transition_count", 0)
        
        return tracker
    
    # ===== Statistics Methods =====
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "current_state": self.current_state.value,
            "transition_count": self._transition_count,
            "time_in_state": {s.value: round(t, 2) for s, t in self._time_in_state.items()},
            "history_length": len(self.history),
            "transitions_length": len(self.transitions),
        }
    
    def get_recent_history(self, n: int = 10) -> List[HealthSnapshot]:
        """Get last N snapshots."""
        return self.history[-n:] if self.history else []
    
    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.current_state = HealthState.UNKNOWN
        self.state_entered_at = None
        self.state_confidence = 0.0
        self.history.clear()
        self.transitions.clear()
        self._transition_count = 0
        self._time_in_state = {s: 0.0 for s in HealthState}


# ============================================================================
# Convenience Functions
# ============================================================================

def compute_health_from_z_scores(
    z_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Compute weighted fused z-score and confidence from detector z-scores.
    
    Args:
        z_scores: Dict of detector_name -> z_score
        weights: Optional weights per detector
        
    Returns:
        (fused_z, confidence)
    """
    if not z_scores:
        return 0.0, 0.0
    
    valid_scores = {k: v for k, v in z_scores.items() if np.isfinite(v)}
    if not valid_scores:
        return 0.0, 0.0
    
    if weights is None:
        weights = {k: 1.0 for k in valid_scores}
    
    total_weight = sum(weights.get(k, 1.0) for k in valid_scores)
    if total_weight <= 0:
        total_weight = len(valid_scores)
    
    fused_z = sum(v * weights.get(k, 1.0) for k, v in valid_scores.items()) / total_weight
    
    # Confidence based on detector agreement
    if len(valid_scores) > 1:
        values = list(valid_scores.values())
        std = np.std(values)
        agreement = 1 / (1 + std)  # High std = low agreement
        missing_penalty = 1.0 - (0.1 * (len(z_scores) - len(valid_scores)))
        confidence = agreement * max(0.0, missing_penalty)
    else:
        confidence = 0.5  # Single detector, moderate confidence
    
    return float(fused_z), float(np.clip(confidence, 0.0, 1.0))

"""
Drift and Novelty Control Plane for ACM v11.

P5.1 - Promotes drift signals from passive logging to active control-plane triggers.
P5.2 - Tracks novelty pressure as a first-class metric independent of regime labels.
P5.3 - Manages drift events as persisted, resolvable objects.

This module provides:
- DriftController: Central controller evaluating drift/novelty signals
- NoveltyTracker: Continuous novelty pressure tracking
- DriftEventManager: Drift event lifecycle management

Usage:
    from core.drift_controller import DriftController, DriftThresholds

    controller = DriftController(
        thresholds=DriftThresholds(),
        on_replay_trigger=lambda signal: queue_offline_replay(signal)
    )

    signals = controller.evaluate(
        psi_values=psi_df,
        kl_values=kl_df,
        unknown_regime_pct=0.15,
        timestamp=pd.Timestamp.now()
    )
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Callable, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

__all__ = [
    "DriftAction",
    "DriftThresholds",
    "DriftSignal",
    "DriftController",
    "NoveltyPressure",
    "NoveltyTracker",
    "DriftEvent",
    "DriftEventManager",
]


# =============================================================================
# P5.1 - Drift/Novelty Control Plane
# =============================================================================

class DriftAction(Enum):
    """Actions triggered by drift detection."""
    NONE = "NONE"                    # No action needed
    LOG_WARNING = "LOG_WARNING"      # Log and continue
    REDUCE_CONFIDENCE = "REDUCE_CONFIDENCE"  # Dampen output confidence
    TRIGGER_REPLAY = "TRIGGER_REPLAY"  # Queue offline replay
    HALT_PREDICTIONS = "HALT_PREDICTIONS"  # Stop making predictions


@dataclass
class DriftThresholds:
    """
    Thresholds for drift severity levels.

    Attributes:
        psi_warning: PSI value triggering warning (default 0.1)
        psi_critical: PSI value triggering critical action (default 0.25)
        kl_warning: KL divergence warning threshold (default 0.1)
        kl_critical: KL divergence critical threshold (default 0.5)
        novelty_warning: % unknown regime triggering warning (default 0.3 = 30%)
        novelty_critical: % unknown regime triggering critical action (default 0.5 = 50%)
    """
    psi_warning: float = 0.1      # PSI > 0.1 = warning
    psi_critical: float = 0.25    # PSI > 0.25 = critical
    kl_warning: float = 0.1       # KL divergence warning
    kl_critical: float = 0.5      # KL divergence critical
    novelty_warning: float = 0.3  # % unknown regime > 30%
    novelty_critical: float = 0.5 # % unknown regime > 50%


@dataclass
class DriftSignal:
    """
    A detected drift or novelty signal.

    Attributes:
        timestamp: When signal was detected
        signal_type: Type of signal (PSI, KL, NOVELTY, COVARIATE)
        severity: Severity level (WARNING, CRITICAL)
        value: The measured value that triggered the signal
        affected_sensors: List of sensors affected
        recommended_action: Action to take based on signal
    """
    timestamp: pd.Timestamp
    signal_type: str  # "PSI", "KL", "NOVELTY", "COVARIATE"
    severity: str     # "WARNING", "CRITICAL"
    value: float
    affected_sensors: List[str]
    recommended_action: DriftAction

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/persistence."""
        return {
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            "signal_type": self.signal_type,
            "severity": self.severity,
            "value": self.value,
            "affected_sensors": self.affected_sensors,
            "recommended_action": self.recommended_action.value,
        }


class DriftController:
    """
    Central controller for drift and novelty signals.

    Promotes drift from passive logging to active control plane triggers.
    Evaluates PSI, KL divergence, and novelty metrics to determine actions.

    Parameters:
        thresholds: DriftThresholds configuration
        on_replay_trigger: Callback when replay is triggered
        on_confidence_reduction: Callback when confidence should be reduced

    Example:
        >>> controller = DriftController()
        >>> signals = controller.evaluate(psi_df, kl_df, 0.15, pd.Timestamp.now())
        >>> for s in signals:
        ...     print(f"{s.signal_type}: {s.severity} -> {s.recommended_action}")
    """

    def __init__(
        self,
        thresholds: Optional[DriftThresholds] = None,
        on_replay_trigger: Optional[Callable[[DriftSignal], None]] = None,
        on_confidence_reduction: Optional[Callable[[float], None]] = None
    ):
        self.thresholds = thresholds or DriftThresholds()
        self.on_replay_trigger = on_replay_trigger
        self.on_confidence_reduction = on_confidence_reduction
        self.signal_history: List[DriftSignal] = []
        self.replay_queued = False
        self._confidence_penalty = 0.0

    def evaluate(
        self,
        psi_values: pd.DataFrame,
        kl_values: Optional[pd.DataFrame],
        unknown_regime_pct: float,
        timestamp: pd.Timestamp
    ) -> List[DriftSignal]:
        """
        Evaluate all drift/novelty signals and determine actions.

        Parameters:
            psi_values: DataFrame with PSI values per sensor column
            kl_values: Optional DataFrame with KL divergence values
            unknown_regime_pct: Fraction of recent rows in unknown regime (0-1)
            timestamp: Current timestamp for signal recording

        Returns:
            List of DriftSignal objects representing detected drift/novelty
        """
        signals = []

        # Evaluate PSI per sensor
        if psi_values is not None and not psi_values.empty:
            signals.extend(self._evaluate_psi(psi_values, timestamp))

        # Evaluate KL divergence per sensor
        if kl_values is not None and not kl_values.empty:
            signals.extend(self._evaluate_kl(kl_values, timestamp))

        # Evaluate novelty pressure
        signals.extend(self._evaluate_novelty(unknown_regime_pct, timestamp))

        # Store and execute actions
        self.signal_history.extend(signals)
        self._execute_actions(signals)

        return signals

    def _evaluate_psi(
        self, psi_values: pd.DataFrame, timestamp: pd.Timestamp
    ) -> List[DriftSignal]:
        """Evaluate PSI values for each sensor."""
        signals = []

        for col in psi_values.columns:
            psi = psi_values[col].iloc[-1] if len(psi_values) > 0 else 0

            if pd.isna(psi):
                continue

            if psi > self.thresholds.psi_critical:
                signals.append(DriftSignal(
                    timestamp=timestamp,
                    signal_type="PSI",
                    severity="CRITICAL",
                    value=float(psi),
                    affected_sensors=[str(col)],
                    recommended_action=DriftAction.TRIGGER_REPLAY
                ))
            elif psi > self.thresholds.psi_warning:
                signals.append(DriftSignal(
                    timestamp=timestamp,
                    signal_type="PSI",
                    severity="WARNING",
                    value=float(psi),
                    affected_sensors=[str(col)],
                    recommended_action=DriftAction.LOG_WARNING
                ))

        return signals

    def _evaluate_kl(
        self, kl_values: pd.DataFrame, timestamp: pd.Timestamp
    ) -> List[DriftSignal]:
        """Evaluate KL divergence values for each sensor."""
        signals = []

        for col in kl_values.columns:
            kl = kl_values[col].iloc[-1] if len(kl_values) > 0 else 0

            if pd.isna(kl):
                continue

            if kl > self.thresholds.kl_critical:
                signals.append(DriftSignal(
                    timestamp=timestamp,
                    signal_type="KL",
                    severity="CRITICAL",
                    value=float(kl),
                    affected_sensors=[str(col)],
                    recommended_action=DriftAction.TRIGGER_REPLAY
                ))
            elif kl > self.thresholds.kl_warning:
                signals.append(DriftSignal(
                    timestamp=timestamp,
                    signal_type="KL",
                    severity="WARNING",
                    value=float(kl),
                    affected_sensors=[str(col)],
                    recommended_action=DriftAction.REDUCE_CONFIDENCE
                ))

        return signals

    def _evaluate_novelty(
        self, unknown_regime_pct: float, timestamp: pd.Timestamp
    ) -> List[DriftSignal]:
        """Evaluate novelty pressure from unknown regime percentage."""
        signals = []

        if unknown_regime_pct > self.thresholds.novelty_critical:
            signals.append(DriftSignal(
                timestamp=timestamp,
                signal_type="NOVELTY",
                severity="CRITICAL",
                value=float(unknown_regime_pct),
                affected_sensors=[],
                recommended_action=DriftAction.TRIGGER_REPLAY
            ))
        elif unknown_regime_pct > self.thresholds.novelty_warning:
            signals.append(DriftSignal(
                timestamp=timestamp,
                signal_type="NOVELTY",
                severity="WARNING",
                value=float(unknown_regime_pct),
                affected_sensors=[],
                recommended_action=DriftAction.REDUCE_CONFIDENCE
            ))

        return signals

    def _execute_actions(self, signals: List[DriftSignal]) -> None:
        """Execute recommended actions from signals."""
        for signal in signals:
            if signal.recommended_action == DriftAction.TRIGGER_REPLAY:
                if not self.replay_queued and self.on_replay_trigger:
                    self.on_replay_trigger(signal)
                    self.replay_queued = True

            elif signal.recommended_action == DriftAction.REDUCE_CONFIDENCE:
                # Accumulate confidence penalty
                penalty = 0.3 if signal.severity == "CRITICAL" else 0.1
                self._confidence_penalty = min(0.5, self._confidence_penalty + penalty)
                if self.on_confidence_reduction:
                    self.on_confidence_reduction(self._confidence_penalty)

    def get_confidence_penalty(self) -> float:
        """Get current confidence penalty from accumulated drift signals."""
        return self._confidence_penalty

    def reset_replay_queue(self) -> None:
        """Reset replay queue after replay completes."""
        self.replay_queued = False

    def reset_confidence_penalty(self) -> None:
        """Reset confidence penalty after model retraining."""
        self._confidence_penalty = 0.0

    def get_signal_summary(
        self, hours: int = 24
    ) -> Dict[str, Any]:
        """Get summary of recent signals."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        recent = [s for s in self.signal_history
                  if s.timestamp > cutoff]

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for s in recent:
            by_type[s.signal_type] = by_type.get(s.signal_type, 0) + 1
            by_severity[s.severity] = by_severity.get(s.severity, 0) + 1

        return {
            "total_signals": len(recent),
            "by_type": by_type,
            "by_severity": by_severity,
            "replay_queued": self.replay_queued,
            "confidence_penalty": self._confidence_penalty,
        }


# =============================================================================
# P5.2 - Novelty Pressure Tracking
# =============================================================================

@dataclass
class NoveltyPressure:
    """
    Track novelty pressure independent of regime labels.

    Unlike regime labels (which are discrete), novelty pressure
    is a continuous measure of how "unusual" recent data is.

    Attributes:
        timestamp: When measurement was taken
        unknown_pct: % of recent rows in UNKNOWN regime (-1)
        emerging_pct: % in EMERGING regime (not yet promoted)
        out_of_distribution: Average distance from known regime centroids
        trend: Direction of novelty pressure (INCREASING, STABLE, DECREASING)
    """
    timestamp: pd.Timestamp
    unknown_pct: float           # % of recent rows in UNKNOWN regime
    emerging_pct: float          # % in EMERGING (not yet promoted)
    out_of_distribution: float   # Average distance from known regime centroids
    trend: str                   # "INCREASING", "STABLE", "DECREASING"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQL persistence."""
        return {
            "Timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            "UnknownPct": self.unknown_pct,
            "EmergingPct": self.emerging_pct,
            "OutOfDistribution": self.out_of_distribution,
            "Trend": self.trend,
        }


class NoveltyTracker:
    """
    Track novelty pressure as a first-class metric.

    Provides continuous monitoring of how unusual recent data is,
    independent of discrete regime labels.

    Parameters:
        window_hours: Lookback window for computing novelty (default 24h)
        trend_window: Number of measurements for trend computation (default 5)
        slope_threshold: Slope threshold for trend detection (default 0.05)

    Example:
        >>> tracker = NoveltyTracker(window_hours=24)
        >>> pressure = tracker.compute(regime_labels, regime_distances, timestamps)
        >>> print(f"Unknown: {pressure.unknown_pct:.1%}, Trend: {pressure.trend}")
    """

    def __init__(
        self,
        window_hours: float = 24.0,
        trend_window: int = 5,
        slope_threshold: float = 0.05
    ):
        self.window_hours = window_hours
        self.trend_window = trend_window
        self.slope_threshold = slope_threshold
        self.history: List[NoveltyPressure] = []

    def compute(
        self,
        regime_labels: pd.Series,
        regime_distances: Optional[pd.DataFrame],
        timestamps: pd.Series
    ) -> NoveltyPressure:
        """
        Compute current novelty pressure.

        Parameters:
            regime_labels: Series of regime labels (-1 = UNKNOWN, -2 = EMERGING)
            regime_distances: DataFrame of distances to each regime centroid
            timestamps: Series of timestamps corresponding to labels

        Returns:
            NoveltyPressure measurement
        """
        # Filter to window
        max_ts = timestamps.max()
        window_start = max_ts - pd.Timedelta(hours=self.window_hours)
        mask = timestamps >= window_start

        labels = regime_labels[mask]

        # % in UNKNOWN (-1) or EMERGING (-2)
        unknown_pct = float((labels == -1).mean()) if len(labels) > 0 else 0.0
        emerging_pct = float((labels == -2).mean()) if len(labels) > 0 else 0.0

        # Average distance from nearest known regime
        if regime_distances is not None and not regime_distances.empty:
            masked_distances = regime_distances.loc[mask] if mask.any() else regime_distances
            if not masked_distances.empty:
                min_distances = masked_distances.min(axis=1)
                out_of_distribution = float(min_distances.mean())
            else:
                out_of_distribution = 0.0
        else:
            out_of_distribution = 0.0

        # Compute trend from history
        trend = self._compute_trend()

        pressure = NoveltyPressure(
            timestamp=max_ts,
            unknown_pct=unknown_pct,
            emerging_pct=emerging_pct,
            out_of_distribution=out_of_distribution,
            trend=trend
        )

        self.history.append(pressure)
        return pressure

    def _compute_trend(self) -> str:
        """Determine if novelty pressure is increasing/decreasing."""
        if len(self.history) < self.trend_window:
            return "STABLE"

        recent = [h.unknown_pct + h.emerging_pct for h in self.history[-self.trend_window:]]
        if len(recent) < 2:
            return "STABLE"

        # Simple linear slope
        slope = (recent[-1] - recent[0]) / len(recent)

        if slope > self.slope_threshold:
            return "INCREASING"
        elif slope < -self.slope_threshold:
            return "DECREASING"
        return "STABLE"

    def get_current_pressure(self) -> Optional[NoveltyPressure]:
        """Get most recent novelty pressure measurement."""
        return self.history[-1] if self.history else None

    def get_history(self, hours: int = 24) -> List[NoveltyPressure]:
        """Get novelty pressure history for specified hours."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        return [p for p in self.history if p.timestamp > cutoff]


# =============================================================================
# P5.3 - Drift Events as Objects
# =============================================================================

@dataclass
class DriftEvent:
    """
    Persisted drift event with full context.

    Represents a drift event that can be tracked through its lifecycle
    from detection to resolution.

    Attributes:
        id: Unique event identifier
        equip_id: Equipment this event applies to
        detected_at: When event was detected
        event_type: Type (COVARIATE, CONCEPT, NOVELTY)
        severity: WARNING or CRITICAL
        psi_value: PSI value if applicable
        kl_value: KL divergence if applicable
        affected_sensors: List of affected sensor names
        confidence_reduction: How much to reduce output confidence (0-1)
        models_invalidated: List of model names that need retraining
        resolved_at: When event was resolved (None if active)
        resolution: How event was resolved (RETRAINED, FALSE_ALARM, ACKNOWLEDGED)
    """
    id: str
    equip_id: int
    detected_at: pd.Timestamp
    event_type: str  # "COVARIATE", "CONCEPT", "NOVELTY"
    severity: str    # "WARNING", "CRITICAL"

    # Evidence
    psi_value: Optional[float]
    kl_value: Optional[float]
    affected_sensors: List[str]

    # Impact
    confidence_reduction: float  # How much to reduce output confidence
    models_invalidated: List[str]  # Which models need retraining

    # Resolution
    resolved_at: Optional[pd.Timestamp] = None
    resolution: Optional[str] = None  # "RETRAINED", "FALSE_ALARM", "ACKNOWLEDGED"

    def to_sql_row(self) -> Dict[str, Any]:
        """Convert to dict for SQL insert."""
        return {
            "DriftEventID": self.id,
            "EquipID": self.equip_id,
            "DetectedAt": self.detected_at,
            "EventType": self.event_type,
            "Severity": self.severity,
            "PSIValue": self.psi_value,
            "KLValue": self.kl_value,
            "AffectedSensors": ",".join(self.affected_sensors),
            "ConfidenceReduction": self.confidence_reduction,
            "ModelsInvalidated": ",".join(self.models_invalidated),
            "ResolvedAt": self.resolved_at,
            "Resolution": self.resolution
        }

    @property
    def is_active(self) -> bool:
        """Check if event is still active (not resolved)."""
        return self.resolved_at is None


class DriftEventManager:
    """
    Manage drift events as first-class objects.

    Handles event creation, resolution, and confidence penalty calculation.

    Example:
        >>> manager = DriftEventManager()
        >>> event = manager.create_event(signal, equip_id=123)
        >>> print(f"Created: {event.id}")
        >>> manager.resolve_event(event.id, "ACKNOWLEDGED")
    """

    def __init__(self):
        self.active_events: Dict[str, DriftEvent] = {}
        self.resolved_events: List[DriftEvent] = []

    def create_event(
        self,
        signal: DriftSignal,
        equip_id: int,
        models_invalidated: Optional[List[str]] = None
    ) -> DriftEvent:
        """
        Create drift event from signal.

        Parameters:
            signal: DriftSignal that triggered event creation
            equip_id: Equipment ID
            models_invalidated: Optional list of model names to invalidate

        Returns:
            Created DriftEvent
        """
        event_id = f"DRIFT_{equip_id}_{signal.timestamp.strftime('%Y%m%d%H%M%S')}"

        # Determine confidence reduction based on severity
        confidence_reduction = 0.3 if signal.severity == "CRITICAL" else 0.1

        event = DriftEvent(
            id=event_id,
            equip_id=equip_id,
            detected_at=signal.timestamp,
            event_type=signal.signal_type,
            severity=signal.severity,
            psi_value=signal.value if signal.signal_type == "PSI" else None,
            kl_value=signal.value if signal.signal_type == "KL" else None,
            affected_sensors=signal.affected_sensors,
            confidence_reduction=confidence_reduction,
            models_invalidated=models_invalidated or []
        )

        self.active_events[event.id] = event
        return event

    def resolve_event(
        self,
        event_id: str,
        resolution: str,
        resolved_at: Optional[pd.Timestamp] = None
    ) -> bool:
        """
        Mark drift event as resolved.

        Parameters:
            event_id: Event ID to resolve
            resolution: Resolution type (RETRAINED, FALSE_ALARM, ACKNOWLEDGED)
            resolved_at: Resolution timestamp (defaults to now)

        Returns:
            True if event was found and resolved, False otherwise
        """
        if event_id not in self.active_events:
            return False

        event = self.active_events.pop(event_id)
        event.resolved_at = resolved_at or pd.Timestamp.now()
        event.resolution = resolution
        self.resolved_events.append(event)
        return True

    def get_confidence_penalty(self, equip_id: int) -> float:
        """
        Get total confidence penalty from active drift events.

        Parameters:
            equip_id: Equipment ID

        Returns:
            Confidence penalty (0-0.5, capped)
        """
        penalty = 0.0
        for event in self.active_events.values():
            if event.equip_id == equip_id and event.is_active:
                penalty += event.confidence_reduction
        return min(0.5, penalty)  # Cap at 50% reduction

    def get_active_events(self, equip_id: Optional[int] = None) -> List[DriftEvent]:
        """Get active drift events, optionally filtered by equipment."""
        if equip_id is None:
            return list(self.active_events.values())
        return [e for e in self.active_events.values() if e.equip_id == equip_id]

    def get_event_summary(
        self, equip_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get summary of drift events."""
        active = self.get_active_events(equip_id)
        resolved = [e for e in self.resolved_events
                    if equip_id is None or e.equip_id == equip_id]

        return {
            "active_count": len(active),
            "resolved_count": len(resolved),
            "total_confidence_penalty": self.get_confidence_penalty(equip_id) if equip_id else 0.0,
            "active_by_type": {
                t: sum(1 for e in active if e.event_type == t)
                for t in ["PSI", "KL", "NOVELTY", "COVARIATE"]
            },
            "active_by_severity": {
                s: sum(1 for e in active if e.severity == s)
                for s in ["WARNING", "CRITICAL"]
            }
        }

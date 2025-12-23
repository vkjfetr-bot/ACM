"""
RUL Reliability Gate - prerequisite checking for RUL predictions.

v11.0.0 - Phase 4.4 Implementation

CRITICAL PRINCIPLE: RUL predictions MUST NOT be presented to operators
unless prerequisites are met. An unreliable RUL prediction can cause:
- False alarms leading to unnecessary maintenance
- Missed failures due to overconfidence in bad predictions
- Loss of operator trust in the system

This module gates RUL predictions, returning NOT_RELIABLE when
prerequisites fail, instead of a misleading numeric value.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np


class RULStatus(Enum):
    """RUL prediction reliability status."""
    
    RELIABLE = "RELIABLE"
    """High confidence prediction - all prerequisites met."""
    
    NOT_RELIABLE = "NOT_RELIABLE"
    """Insufficient evidence to make reliable prediction."""
    
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    """Not enough historical data for prediction."""
    
    NO_DEGRADATION = "NO_DEGRADATION"
    """Equipment is healthy - no RUL prediction needed."""
    
    REGIME_UNSTABLE = "REGIME_UNSTABLE"
    """Recent regime changes invalidate prediction."""
    
    DETECTOR_DISAGREEMENT = "DETECTOR_DISAGREEMENT"
    """Detectors disagree significantly about health state."""
    
    @property
    def allows_numeric_rul(self) -> bool:
        """Whether this status allows showing a numeric RUL value."""
        return self in (RULStatus.RELIABLE,)
    
    @property
    def display_text(self) -> str:
        """Human-readable display text for dashboards."""
        return {
            RULStatus.RELIABLE: "RUL prediction available",
            RULStatus.NOT_RELIABLE: "RUL prediction not reliable",
            RULStatus.INSUFFICIENT_DATA: "Insufficient data for RUL",
            RULStatus.NO_DEGRADATION: "Equipment healthy - no RUL needed",
            RULStatus.REGIME_UNSTABLE: "Regime unstable - RUL uncertain",
            RULStatus.DETECTOR_DISAGREEMENT: "Detector disagreement",
        }[self]


@dataclass
class RULPrerequisites:
    """
    Prerequisites for reliable RUL prediction.
    
    These thresholds are based on empirical analysis of RUL prediction
    accuracy. Predictions made without meeting these prerequisites have
    historically shown >50% error rates.
    """
    
    # Minimum data points for trend analysis
    min_data_points: int = 500
    
    # Minimum degradation episodes to establish pattern
    min_degradation_episodes: int = 2
    
    # Minimum health trend points for extrapolation
    min_health_trend_points: int = 50
    
    # Minimum regime stability (hours) before trusting RUL
    min_regime_stability_hours: float = 24.0
    
    # Maximum allowable data gap (hours) in recent history
    max_data_gap_hours: float = 48.0
    
    # Minimum detector agreement (0-1) for reliable fusion
    min_detector_agreement: float = 0.6
    
    # Minimum health decline (percentage points) to warrant RUL
    min_health_decline: float = 10.0
    
    # Minimum confidence from health tracker
    min_health_confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "min_data_points": self.min_data_points,
            "min_degradation_episodes": self.min_degradation_episodes,
            "min_health_trend_points": self.min_health_trend_points,
            "min_regime_stability_hours": self.min_regime_stability_hours,
            "max_data_gap_hours": self.max_data_gap_hours,
            "min_detector_agreement": self.min_detector_agreement,
            "min_health_decline": self.min_health_decline,
            "min_health_confidence": self.min_health_confidence,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RULPrerequisites":
        """Create from dictionary."""
        return cls(
            min_data_points=d.get("min_data_points", 500),
            min_degradation_episodes=d.get("min_degradation_episodes", 2),
            min_health_trend_points=d.get("min_health_trend_points", 50),
            min_regime_stability_hours=d.get("min_regime_stability_hours", 24.0),
            max_data_gap_hours=d.get("max_data_gap_hours", 48.0),
            min_detector_agreement=d.get("min_detector_agreement", 0.6),
            min_health_decline=d.get("min_health_decline", 10.0),
            min_health_confidence=d.get("min_health_confidence", 0.5),
        )


@dataclass
class RULResult:
    """
    RUL prediction result with reliability status.
    
    CRITICAL: If status != RELIABLE, numeric RUL values should NOT
    be displayed to operators. The rul_hours and percentiles may
    be None or unreliable.
    """
    
    # Reliability status - MUST check before using numeric values
    status: RULStatus
    
    # RUL prediction (only valid if status == RELIABLE)
    rul_hours: Optional[float] = None
    
    # Percentile bounds (only valid if status == RELIABLE)
    p10_lower: Optional[float] = None
    p50_median: Optional[float] = None
    p90_upper: Optional[float] = None
    
    # Confidence in prediction (0-1)
    confidence: float = 0.0
    
    # Prediction method used
    method: str = "none"
    
    # Diagnostic information
    prerequisite_failures: List[str] = field(default_factory=list)
    data_quality_score: float = 0.0
    regime_stability_score: float = 0.0
    detector_agreement_score: float = 0.0
    
    @property
    def is_reliable(self) -> bool:
        """Whether this prediction can be shown to operators."""
        return self.status == RULStatus.RELIABLE
    
    @property
    def display_rul(self) -> Optional[str]:
        """Display string for RUL, accounting for reliability."""
        if not self.is_reliable:
            return self.status.display_text
        if self.rul_hours is None:
            return "N/A"
        if self.rul_hours < 24:
            return f"{self.rul_hours:.1f} hours"
        elif self.rul_hours < 168:  # 7 days
            return f"{self.rul_hours / 24:.1f} days"
        else:
            return f"{self.rul_hours / 168:.1f} weeks"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SQL writes."""
        return {
            "RULStatus": self.status.value,
            "RUL_Hours": self.rul_hours,
            "P10_LowerBound": self.p10_lower,
            "P50_Median": self.p50_median,
            "P90_UpperBound": self.p90_upper,
            "Confidence": self.confidence,
            "Method": self.method,
            "PrerequisiteFailures": "; ".join(self.prerequisite_failures) if self.prerequisite_failures else None,
            "DataQualityScore": self.data_quality_score,
            "RegimeStabilityScore": self.regime_stability_score,
            "DetectorAgreementScore": self.detector_agreement_score,
        }
    
    @classmethod
    def not_reliable(cls, failures: List[str], 
                     data_quality: float = 0.0,
                     regime_stability: float = 0.0) -> "RULResult":
        """Factory for NOT_RELIABLE result."""
        return cls(
            status=RULStatus.NOT_RELIABLE,
            rul_hours=None,
            p10_lower=None,
            p50_median=None,
            p90_upper=None,
            confidence=0.0,
            method="none",
            prerequisite_failures=failures,
            data_quality_score=data_quality,
            regime_stability_score=regime_stability,
        )
    
    @classmethod
    def no_degradation(cls) -> "RULResult":
        """Factory for NO_DEGRADATION result (healthy equipment)."""
        return cls(
            status=RULStatus.NO_DEGRADATION,
            rul_hours=None,
            p10_lower=None,
            p50_median=None,
            p90_upper=None,
            confidence=0.95,  # High confidence it's healthy
            method="none",
            prerequisite_failures=[],
            data_quality_score=1.0,
            regime_stability_score=1.0,
        )


class RULReliabilityGate:
    """
    Gate RUL predictions on prerequisites.
    
    CRITICAL: If prerequisites fail, return RUL_NOT_RELIABLE instead of
    a numeric prediction that might mislead operators.
    
    Usage:
        gate = RULReliabilityGate()
        result = gate.check_prerequisites(data, episodes, health_trend, ...)
        
        if result is not None:
            # Prerequisites failed - use result as final RUL output
            return result
        
        # Prerequisites passed - compute actual RUL
        rul = compute_rul(...)
        return RULResult(status=RULStatus.RELIABLE, rul_hours=rul, ...)
    """
    
    def __init__(self, prereqs: Optional[RULPrerequisites] = None):
        """
        Initialize gate with prerequisites.
        
        Args:
            prereqs: Prerequisite thresholds. Uses defaults if None.
        """
        self.prereqs = prereqs or RULPrerequisites()
    
    def check_prerequisites(
        self,
        data: pd.DataFrame,
        episodes: List[Any],
        health_trend: pd.DataFrame,
        current_regime: int,
        regime_history: Optional[pd.Series] = None,
        detector_z_scores: Optional[Dict[str, float]] = None,
        current_health_pct: Optional[float] = None,
        current_confidence: Optional[float] = None,
    ) -> Optional[RULResult]:
        """
        Check all prerequisites for reliable RUL.
        
        Args:
            data: Historical sensor data DataFrame
            episodes: List of degradation episodes
            health_trend: DataFrame with health history (needs 'health_pct' column)
            current_regime: Current operating regime ID (-1 = UNKNOWN)
            regime_history: Series of regime IDs over time (index = timestamps)
            detector_z_scores: Dict mapping detector names to current z-scores
            current_health_pct: Current health percentage (0-100)
            current_confidence: Current confidence from health tracker
        
        Returns:
            RULResult if prerequisites failed (use as final result)
            None if all prerequisites passed (proceed to compute RUL)
        """
        failures: List[str] = []
        
        # Calculate quality scores
        data_quality = min(1.0, len(data) / self.prereqs.min_data_points) if len(data) > 0 else 0.0
        regime_stability = self._calculate_regime_stability(regime_history, current_regime)
        detector_agreement = self._calculate_detector_agreement(detector_z_scores)
        
        # --- Check 1: Minimum data points ---
        if len(data) < self.prereqs.min_data_points:
            failures.append(
                f"Only {len(data)} data points (need {self.prereqs.min_data_points})"
            )
        
        # --- Check 2: Minimum degradation episodes ---
        if len(episodes) < self.prereqs.min_degradation_episodes:
            failures.append(
                f"Only {len(episodes)} degradation episodes "
                f"(need {self.prereqs.min_degradation_episodes})"
            )
        
        # --- Check 3: Minimum health trend points ---
        if len(health_trend) < self.prereqs.min_health_trend_points:
            failures.append(
                f"Only {len(health_trend)} health trend points "
                f"(need {self.prereqs.min_health_trend_points})"
            )
        
        # --- Check 4: Regime stability ---
        if current_regime == -1:
            failures.append("Current regime is UNKNOWN")
        elif regime_history is not None and not regime_history.empty:
            regime_duration = self._regime_duration_hours(regime_history, current_regime)
            if regime_duration < self.prereqs.min_regime_stability_hours:
                failures.append(
                    f"Regime only stable for {regime_duration:.1f}h "
                    f"(need {self.prereqs.min_regime_stability_hours}h)"
                )
        
        # --- Check 5: Data gaps ---
        if not data.empty:
            max_gap = self._max_data_gap_hours(data)
            if max_gap > self.prereqs.max_data_gap_hours:
                failures.append(
                    f"Data gap of {max_gap:.1f}h exceeds {self.prereqs.max_data_gap_hours}h"
                )
        
        # --- Check 6: Detector agreement ---
        if detector_z_scores and len(detector_z_scores) >= 2:
            if detector_agreement < self.prereqs.min_detector_agreement:
                failures.append(
                    f"Detector agreement {detector_agreement:.2f} below "
                    f"threshold {self.prereqs.min_detector_agreement}"
                )
        
        # --- Check 7: Health confidence ---
        if current_confidence is not None:
            if current_confidence < self.prereqs.min_health_confidence:
                failures.append(
                    f"Health confidence {current_confidence:.2f} below "
                    f"threshold {self.prereqs.min_health_confidence}"
                )
        
        # --- Special case: No degradation detected ---
        if len(episodes) == 0:
            # Check if equipment is actually healthy
            health_col = "health_pct" if "health_pct" in health_trend.columns else None
            if health_col:
                avg_health = health_trend[health_col].mean()
            elif current_health_pct is not None:
                avg_health = current_health_pct
            else:
                avg_health = 100.0
            
            if avg_health > (100 - self.prereqs.min_health_decline):
                return RULResult.no_degradation()
        
        # --- Return failure result if any prerequisites failed ---
        if failures:
            return RULResult.not_reliable(
                failures=failures,
                data_quality=data_quality,
                regime_stability=regime_stability,
            )
        
        # All prerequisites passed - caller should compute RUL
        return None
    
    def _regime_duration_hours(
        self, 
        regime_history: pd.Series, 
        current_regime: int
    ) -> float:
        """Calculate how long we've been in current regime."""
        if regime_history.empty:
            return 0.0
        
        # Ensure we have a datetime index
        if not isinstance(regime_history.index, pd.DatetimeIndex):
            return float(len(regime_history))  # Fallback: count as hours
        
        # Find where regime changes
        changes = regime_history != regime_history.shift()
        
        if not changes.any():
            # Entire history is same regime
            total_span = (regime_history.index[-1] - regime_history.index[0])
            return total_span.total_seconds() / 3600
        
        # Find last change to current regime
        change_indices = changes[changes].index
        last_change_idx = change_indices[-1]
        
        # Calculate duration from last change
        duration = (regime_history.index[-1] - last_change_idx)
        return duration.total_seconds() / 3600
    
    def _max_data_gap_hours(self, data: pd.DataFrame) -> float:
        """Find maximum time gap in data."""
        if data.empty or len(data) < 2:
            return 0.0
        
        # Try to find timestamp column
        time_col = None
        for col in ["Timestamp", "timestamp", "EntryDateTime", "Time", "time"]:
            if col in data.columns:
                time_col = col
                break
        
        if time_col is None and isinstance(data.index, pd.DatetimeIndex):
            times = data.index.to_series()
        elif time_col is not None:
            times = pd.to_datetime(data[time_col])
        else:
            return 0.0  # Can't determine gaps
        
        # Calculate gaps
        times_sorted = times.sort_values()
        gaps = times_sorted.diff().dropna()
        
        if gaps.empty:
            return 0.0
        
        max_gap = gaps.max()
        
        # Convert to hours
        if hasattr(max_gap, "total_seconds"):
            return max_gap.total_seconds() / 3600
        else:
            return float(max_gap) / 3600
    
    def _calculate_regime_stability(
        self,
        regime_history: Optional[pd.Series],
        current_regime: int
    ) -> float:
        """
        Calculate regime stability score (0-1).
        
        1.0 = Stable (same regime for >24h)
        0.0 = Unstable (regime just changed or unknown)
        """
        if current_regime == -1:
            return 0.0
        
        if regime_history is None or regime_history.empty:
            return 0.5  # Unknown, assume moderate stability
        
        duration_h = self._regime_duration_hours(regime_history, current_regime)
        
        # Saturate at min_regime_stability_hours
        stability = min(1.0, duration_h / self.prereqs.min_regime_stability_hours)
        return stability
    
    def _calculate_detector_agreement(
        self,
        detector_z_scores: Optional[Dict[str, float]]
    ) -> float:
        """
        Calculate detector agreement score (0-1).
        
        1.0 = All detectors agree (similar z-scores)
        0.0 = Complete disagreement
        
        Uses coefficient of variation: lower CV = higher agreement.
        """
        if not detector_z_scores or len(detector_z_scores) < 2:
            return 1.0  # Single detector = perfect agreement
        
        # Filter out NaN/inf values
        valid_scores = [
            z for z in detector_z_scores.values()
            if z is not None and np.isfinite(z)
        ]
        
        if len(valid_scores) < 2:
            return 1.0
        
        scores_array = np.array(valid_scores)
        mean_z = np.mean(scores_array)
        std_z = np.std(scores_array)
        
        if mean_z == 0:
            # All zeros = perfect agreement
            return 1.0 if std_z == 0 else 0.5
        
        # Coefficient of variation
        cv = std_z / abs(mean_z)
        
        # Convert CV to agreement score (CV of 0 = 1.0, CV of 2+ = 0.0)
        agreement = max(0.0, 1.0 - cv / 2.0)
        return agreement
    
    def wrap_rul_prediction(
        self,
        rul_hours: float,
        p10: float,
        p50: float,
        p90: float,
        confidence: float,
        method: str,
        data_quality: float = 1.0,
        regime_stability: float = 1.0,
        detector_agreement: float = 1.0,
    ) -> RULResult:
        """
        Wrap a computed RUL prediction in a RULResult.
        
        Call this AFTER check_prerequisites returns None.
        
        Args:
            rul_hours: Predicted RUL in hours
            p10: 10th percentile (lower bound)
            p50: 50th percentile (median)
            p90: 90th percentile (upper bound)
            confidence: Prediction confidence (0-1)
            method: Prediction method name
            data_quality: Data quality score (0-1)
            regime_stability: Regime stability score (0-1)
            detector_agreement: Detector agreement score (0-1)
        
        Returns:
            RULResult with RELIABLE status
        """
        return RULResult(
            status=RULStatus.RELIABLE,
            rul_hours=rul_hours,
            p10_lower=p10,
            p50_median=p50,
            p90_upper=p90,
            confidence=confidence,
            method=method,
            prerequisite_failures=[],
            data_quality_score=data_quality,
            regime_stability_score=regime_stability,
            detector_agreement_score=detector_agreement,
        )


def create_rul_result_from_legacy(
    rul_hours: Optional[float],
    p10: Optional[float],
    p50: Optional[float],
    p90: Optional[float],
    confidence: float,
    method: str,
    status_override: Optional[RULStatus] = None,
) -> RULResult:
    """
    Create RULResult from legacy RUL computation output.
    
    This is a migration helper for converting old RUL outputs
    to the new RULResult format.
    
    Args:
        rul_hours: RUL prediction (may be None)
        p10, p50, p90: Percentile bounds (may be None)
        confidence: Prediction confidence
        method: Method name
        status_override: Force a specific status
    
    Returns:
        RULResult with inferred or overridden status
    """
    # Determine status from values
    if status_override is not None:
        status = status_override
    elif rul_hours is None:
        status = RULStatus.NOT_RELIABLE
    elif confidence < 0.3:
        status = RULStatus.NOT_RELIABLE
    else:
        status = RULStatus.RELIABLE
    
    return RULResult(
        status=status,
        rul_hours=rul_hours,
        p10_lower=p10,
        p50_median=p50,
        p90_upper=p90,
        confidence=confidence,
        method=method,
        prerequisite_failures=[],
        data_quality_score=1.0 if status == RULStatus.RELIABLE else 0.5,
        regime_stability_score=1.0 if status == RULStatus.RELIABLE else 0.5,
    )

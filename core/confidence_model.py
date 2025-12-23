"""
Unified Confidence Model - combines multiple signals into confidence scores.

v11.0.0 - Phase 4.6 Implementation

This module provides a unified confidence model that combines multiple
signals (regime stability, detector agreement, data quality, etc.) into
a single confidence score for all ACM outputs.

The confidence score is crucial for:
- Determining whether to show RUL predictions
- Setting alert thresholds
- Operator trust in ACM outputs
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import numpy as np


@dataclass
class ConfidenceSignals:
    """
    Individual confidence signals that contribute to overall confidence.
    
    Each signal is a float in [0, 1] where:
    - 1.0 = maximum confidence from this signal
    - 0.0 = minimum confidence (this signal is failing)
    """
    
    # Regime-related signals
    regime_confidence: float = 0.5
    """How stable/certain is the current regime (0 = UNKNOWN, 1 = stable)."""
    
    # Detector-related signals
    detector_agreement: float = 0.5
    """How much detectors agree (0 = disagreement, 1 = perfect agreement)."""
    
    # Data quality signals
    data_quality: float = 0.5
    """Sensor validity and completeness (0 = bad data, 1 = perfect data)."""
    
    # Model maturity signals
    model_maturity: float = 0.5
    """How trained are the models (0 = new, 1 = fully trained)."""
    
    # Drift/novelty signals
    drift_indicator: float = 1.0
    """Inverse of drift severity (0 = severe drift, 1 = no drift)."""
    
    # Data recency signals
    data_recency: float = 1.0
    """How recent is the data (0 = stale, 1 = fresh)."""
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "regime_confidence": self.regime_confidence,
            "detector_agreement": self.detector_agreement,
            "data_quality": self.data_quality,
            "model_maturity": self.model_maturity,
            "drift_indicator": self.drift_indicator,
            "data_recency": self.data_recency,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ConfidenceSignals":
        """Create from dictionary."""
        return cls(
            regime_confidence=d.get("regime_confidence", 0.5),
            detector_agreement=d.get("detector_agreement", 0.5),
            data_quality=d.get("data_quality", 0.5),
            model_maturity=d.get("model_maturity", 0.5),
            drift_indicator=d.get("drift_indicator", 1.0),
            data_recency=d.get("data_recency", 1.0),
        )
    
    @classmethod
    def default_healthy(cls) -> "ConfidenceSignals":
        """Create signals for healthy/normal operation."""
        return cls(
            regime_confidence=0.9,
            detector_agreement=0.9,
            data_quality=0.95,
            model_maturity=0.8,
            drift_indicator=1.0,
            data_recency=1.0,
        )
    
    @classmethod
    def default_degraded(cls) -> "ConfidenceSignals":
        """Create signals for degraded operation."""
        return cls(
            regime_confidence=0.5,
            detector_agreement=0.6,
            data_quality=0.7,
            model_maturity=0.5,
            drift_indicator=0.7,
            data_recency=0.8,
        )
    
    def min_signal(self) -> tuple[str, float]:
        """Return the name and value of the lowest signal."""
        signals = self.to_dict()
        min_name = min(signals, key=signals.get)
        return min_name, signals[min_name]
    
    def max_signal(self) -> tuple[str, float]:
        """Return the name and value of the highest signal."""
        signals = self.to_dict()
        max_name = max(signals, key=signals.get)
        return max_name, signals[max_name]


@dataclass
class CombinedConfidence:
    """
    Combined confidence with breakdown of contributing signals.
    
    This is the final confidence output used by ACM components.
    """
    
    confidence: float
    """Overall confidence score [0, 1]."""
    
    signals: ConfidenceSignals
    """Individual signal contributions."""
    
    limiting_factor: str
    """Which signal is pulling down confidence the most."""
    
    limiting_value: float
    """Value of the limiting factor."""
    
    is_trustworthy: bool
    """Whether confidence exceeds trustworthy threshold."""
    
    @property
    def level(self) -> str:
        """Confidence level as string: HIGH, MEDIUM, LOW."""
        if self.confidence >= 0.8:
            return "HIGH"
        elif self.confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/SQL."""
        return {
            "Confidence": self.confidence,
            "ConfidenceLevel": self.level,
            "LimitingFactor": self.limiting_factor,
            "LimitingValue": self.limiting_value,
            "IsTrustworthy": self.is_trustworthy,
            "RegimeConfidence": self.signals.regime_confidence,
            "DetectorAgreement": self.signals.detector_agreement,
            "DataQuality": self.signals.data_quality,
            "ModelMaturity": self.signals.model_maturity,
            "DriftIndicator": self.signals.drift_indicator,
            "DataRecency": self.signals.data_recency,
        }
    
    def explain(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Confidence: {self.confidence:.2f} ({self.level})",
            f"Limiting factor: {self.limiting_factor} = {self.limiting_value:.2f}",
            f"Trustworthy: {'Yes' if self.is_trustworthy else 'No'}",
        ]
        return "\n".join(lines)


@dataclass
class ConfidenceWeights:
    """
    Weights for combining confidence signals.
    
    Weights should sum to 1.0 for interpretability.
    """
    
    regime_confidence: float = 0.20
    detector_agreement: float = 0.25
    data_quality: float = 0.20
    model_maturity: float = 0.15
    drift_indicator: float = 0.10
    data_recency: float = 0.10
    
    def __post_init__(self):
        """Validate weights sum to ~1.0."""
        total = (
            self.regime_confidence +
            self.detector_agreement +
            self.data_quality +
            self.model_maturity +
            self.drift_indicator +
            self.data_recency
        )
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.regime_confidence /= total
            self.detector_agreement /= total
            self.data_quality /= total
            self.model_maturity /= total
            self.drift_indicator /= total
            self.data_recency /= total
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "regime_confidence": self.regime_confidence,
            "detector_agreement": self.detector_agreement,
            "data_quality": self.data_quality,
            "model_maturity": self.model_maturity,
            "drift_indicator": self.drift_indicator,
            "data_recency": self.data_recency,
        }


class ConfidenceModel:
    """
    Unified confidence model for all ACM outputs.
    
    Combines multiple signals into a single confidence score:
    1. Regime confidence (regime stability, not UNKNOWN)
    2. Detector agreement (fusion quality)
    3. Data quality (sensor validity, no gaps)
    4. Model maturity (training data quantity)
    5. Drift indicator (no recent drift)
    6. Data recency (fresh data)
    
    The model uses weighted averaging with a minimum gate:
    if any signal falls below a critical threshold, overall
    confidence is capped.
    
    Usage:
        model = ConfidenceModel()
        signals = ConfidenceSignals(
            regime_confidence=0.8,
            detector_agreement=0.9,
            data_quality=0.95,
            model_maturity=0.7,
            drift_indicator=1.0,
            data_recency=1.0
        )
        result = model.compute(signals)
        
        if result.is_trustworthy:
            # Use predictions with full confidence
            pass
        else:
            # Warn user about low confidence
            print(f"Low confidence: {result.limiting_factor}")
    """
    
    def __init__(
        self,
        weights: Optional[ConfidenceWeights] = None,
        trustworthy_threshold: float = 0.6,
        critical_signal_threshold: float = 0.3,
        critical_signal_cap: float = 0.5,
    ):
        """
        Initialize confidence model.
        
        Args:
            weights: Weights for combining signals. Uses defaults if None.
            trustworthy_threshold: Confidence above this is "trustworthy".
            critical_signal_threshold: If any signal < this, cap confidence.
            critical_signal_cap: Cap confidence at this if critical threshold hit.
        """
        self.weights = weights or ConfidenceWeights()
        self.trustworthy_threshold = trustworthy_threshold
        self.critical_signal_threshold = critical_signal_threshold
        self.critical_signal_cap = critical_signal_cap
    
    def compute(self, signals: ConfidenceSignals) -> CombinedConfidence:
        """
        Compute combined confidence from signals.
        
        Args:
            signals: Individual confidence signals
        
        Returns:
            CombinedConfidence with overall score and breakdown
        """
        # Compute weighted average
        weighted_sum = (
            signals.regime_confidence * self.weights.regime_confidence +
            signals.detector_agreement * self.weights.detector_agreement +
            signals.data_quality * self.weights.data_quality +
            signals.model_maturity * self.weights.model_maturity +
            signals.drift_indicator * self.weights.drift_indicator +
            signals.data_recency * self.weights.data_recency
        )
        
        # Find limiting factor (lowest signal)
        limiting_factor, limiting_value = signals.min_signal()
        
        # Apply minimum gate: if any signal is very low, cap confidence
        if limiting_value < self.critical_signal_threshold:
            weighted_sum = min(weighted_sum, self.critical_signal_cap)
        
        # Clip to valid range
        confidence = float(np.clip(weighted_sum, 0.0, 1.0))
        
        return CombinedConfidence(
            confidence=confidence,
            signals=signals,
            limiting_factor=limiting_factor,
            limiting_value=limiting_value,
            is_trustworthy=confidence >= self.trustworthy_threshold,
        )
    
    def from_run_context(
        self,
        regime_label: int,
        regime_stability: float,
        detector_z_scores: Optional[Dict[str, float]] = None,
        sensor_validity: Optional[Dict[str, bool]] = None,
        model_age_hours: float = 0.0,
        drift_detected: bool = False,
        data_age_minutes: float = 0.0,
    ) -> CombinedConfidence:
        """
        Build confidence from pipeline run context.
        
        This is a convenience method that converts raw pipeline state
        into confidence signals.
        
        Args:
            regime_label: Current regime (-1 = UNKNOWN)
            regime_stability: How stable the regime has been (0-1)
            detector_z_scores: Dict mapping detector name to z-score
            sensor_validity: Dict mapping sensor name to validity flag
            model_age_hours: How long models have been training
            drift_detected: Whether drift was detected recently
            data_age_minutes: How old the most recent data is
        
        Returns:
            CombinedConfidence
        """
        # Build signals
        signals = ConfidenceSignals(
            regime_confidence=0.0 if regime_label == -1 else regime_stability,
            detector_agreement=self._compute_detector_agreement(detector_z_scores),
            data_quality=self._compute_data_quality(sensor_validity),
            model_maturity=self._compute_model_maturity(model_age_hours),
            drift_indicator=0.3 if drift_detected else 1.0,
            data_recency=self._compute_data_recency(data_age_minutes),
        )
        
        return self.compute(signals)
    
    def _compute_detector_agreement(
        self,
        detector_z_scores: Optional[Dict[str, float]]
    ) -> float:
        """Compute detector agreement from z-scores."""
        if not detector_z_scores or len(detector_z_scores) < 2:
            return 1.0  # Single detector = perfect agreement
        
        # Filter valid values
        valid_scores = [
            z for z in detector_z_scores.values()
            if z is not None and np.isfinite(z)
        ]
        
        if len(valid_scores) < 2:
            return 1.0
        
        # Use coefficient of variation
        scores_array = np.array(valid_scores)
        mean_z = np.mean(scores_array)
        std_z = np.std(scores_array)
        
        if abs(mean_z) < 1e-10:
            return 1.0 if std_z < 0.1 else 0.5
        
        cv = std_z / abs(mean_z)
        
        # Convert CV to agreement (CV of 0 = 1.0, CV of 2+ = 0.0)
        agreement = max(0.0, 1.0 - cv / 2.0)
        return agreement
    
    def _compute_data_quality(
        self,
        sensor_validity: Optional[Dict[str, bool]]
    ) -> float:
        """Compute data quality from sensor validity."""
        if not sensor_validity:
            return 0.5  # Unknown quality
        
        valid_count = sum(1 for v in sensor_validity.values() if v)
        total = len(sensor_validity)
        
        if total == 0:
            return 0.5
        
        return valid_count / total
    
    def _compute_model_maturity(self, model_age_hours: float) -> float:
        """Compute model maturity from training age."""
        # Full maturity at 30 days (720 hours)
        maturity_hours = 720.0
        maturity = min(1.0, model_age_hours / maturity_hours)
        return maturity
    
    def _compute_data_recency(self, data_age_minutes: float) -> float:
        """Compute data recency from data age."""
        # Full recency if < 5 minutes, decay to 0 at 60 minutes
        if data_age_minutes <= 5:
            return 1.0
        elif data_age_minutes >= 60:
            return 0.0
        else:
            # Linear decay from 5 to 60 minutes
            return max(0.0, 1.0 - (data_age_minutes - 5) / 55)


class ConfidenceTracker:
    """
    Track confidence over time for trend analysis.
    
    Useful for detecting gradual degradation in confidence
    that might indicate model staleness or data issues.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize tracker.
        
        Args:
            window_size: Number of recent values to track
        """
        self.window_size = window_size
        self._history: List[CombinedConfidence] = []
    
    def add(self, confidence: CombinedConfidence) -> None:
        """Add a confidence measurement."""
        self._history.append(confidence)
        
        # Trim to window size
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]
    
    @property
    def mean_confidence(self) -> float:
        """Mean confidence over window."""
        if not self._history:
            return 0.0
        return np.mean([c.confidence for c in self._history])
    
    @property
    def min_confidence(self) -> float:
        """Minimum confidence over window."""
        if not self._history:
            return 0.0
        return min(c.confidence for c in self._history)
    
    @property
    def max_confidence(self) -> float:
        """Maximum confidence over window."""
        if not self._history:
            return 0.0
        return max(c.confidence for c in self._history)
    
    @property
    def trend(self) -> str:
        """Confidence trend: IMPROVING, STABLE, DEGRADING."""
        if len(self._history) < 10:
            return "STABLE"
        
        # Compare first and second half
        mid = len(self._history) // 2
        first_half = np.mean([c.confidence for c in self._history[:mid]])
        second_half = np.mean([c.confidence for c in self._history[mid:]])
        
        diff = second_half - first_half
        
        if diff > 0.05:
            return "IMPROVING"
        elif diff < -0.05:
            return "DEGRADING"
        else:
            return "STABLE"
    
    def most_common_limiting_factor(self) -> Optional[str]:
        """Find the most common limiting factor in recent history."""
        if not self._history:
            return None
        
        factors = {}
        for c in self._history:
            factor = c.limiting_factor
            factors[factor] = factors.get(factor, 0) + 1
        
        return max(factors, key=factors.get)
    
    def clear(self) -> None:
        """Clear history."""
        self._history = []
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "n_observations": len(self._history),
            "mean_confidence": self.mean_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "trend": self.trend,
            "most_common_limiting_factor": self.most_common_limiting_factor(),
        }


def combine_confidence_scores(
    health_confidence: float,
    episode_confidence: float,
    rul_confidence: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Combine confidence scores from different components.
    
    Args:
        health_confidence: Confidence from health tracker
        episode_confidence: Confidence from episode manager
        rul_confidence: Confidence from RUL estimator
        weights: Optional weights (default equal weighting)
    
    Returns:
        Combined confidence score
    """
    if weights is None:
        weights = {"health": 0.4, "episode": 0.3, "rul": 0.3}
    
    combined = (
        health_confidence * weights.get("health", 0.33) +
        episode_confidence * weights.get("episode", 0.33) +
        rul_confidence * weights.get("rul", 0.34)
    )
    
    return float(np.clip(combined, 0.0, 1.0))

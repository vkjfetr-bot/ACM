# core/confidence.py
"""
Unified Confidence Model for V11

All ACM outputs must have explicit confidence scores (0-1).
This module provides consistent confidence computation across:
- Health scores
- RUL estimates  
- Episode detection
- Regime assignments

V11 Rules Implemented:
- Rule #10: RUL must be gated or suppressed
- Rule #17: Confidence must always be exposed
- Rule #20: If unsure, the system must say "not reliable"

Confidence Sources:
1. Model maturity (LEARNING=0.5, CONVERGED=1.0)
2. Data quality (sample size, coverage)
3. Prediction uncertainty (std dev, quantile spread)
4. Regime assignment confidence (distance to centroid)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np

from core.observability import Console


class ReliabilityStatus(str, Enum):
    """RUL reliability status for gating."""
    RELIABLE = "RELIABLE"           # All prerequisites met
    NOT_RELIABLE = "NOT_RELIABLE"   # Missing prerequisites
    LEARNING = "LEARNING"           # Model not yet converged
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Not enough history


@dataclass
class ConfidenceFactors:
    """Components that contribute to overall confidence."""
    maturity_factor: float = 1.0      # Model maturity (0.5 for LEARNING, 1.0 for CONVERGED)
    data_quality_factor: float = 1.0  # Data completeness and sample size
    prediction_factor: float = 1.0    # Prediction uncertainty (narrow CI = high)
    regime_factor: float = 1.0        # Regime assignment confidence
    
    def overall(self) -> float:
        """Compute overall confidence as geometric mean of factors."""
        factors = [
            self.maturity_factor,
            self.data_quality_factor,
            self.prediction_factor,
            self.regime_factor,
        ]
        # Geometric mean - all factors contribute equally
        product = 1.0
        for f in factors:
            product *= max(0.0, min(1.0, f))
        return product ** (1.0 / len(factors))
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'MaturityFactor': self.maturity_factor,
            'DataQualityFactor': self.data_quality_factor,
            'PredictionFactor': self.prediction_factor,
            'RegimeFactor': self.regime_factor,
            'OverallConfidence': self.overall(),
        }


def compute_maturity_confidence(maturity_state: str) -> float:
    """
    Confidence factor based on model maturity.
    
    COLDSTART: 0.2 - No reliable baseline
    LEARNING: 0.5 - Building baseline, not production-ready
    CONVERGED: 1.0 - Full confidence in model
    DEPRECATED: 0.3 - Model is stale
    """
    maturity_map = {
        'COLDSTART': 0.2,
        'LEARNING': 0.5,
        'CONVERGED': 1.0,
        'DEPRECATED': 0.3,
    }
    return maturity_map.get(str(maturity_state).upper(), 0.5)


def compute_data_quality_confidence(
    sample_count: int,
    min_samples: int = 100,
    optimal_samples: int = 1000,
    coverage_ratio: float = 1.0,
) -> float:
    """
    Confidence factor based on data quality.
    
    Args:
        sample_count: Number of samples in training window
        min_samples: Minimum for any confidence (below = 0.1)
        optimal_samples: Sample count for full confidence
        coverage_ratio: Fraction of expected time range covered (0-1)
    
    Returns:
        Confidence factor 0-1
    """
    if sample_count < min_samples:
        sample_factor = 0.1
    elif sample_count >= optimal_samples:
        sample_factor = 1.0
    else:
        # Linear interpolation
        sample_factor = 0.1 + 0.9 * (sample_count - min_samples) / (optimal_samples - min_samples)
    
    # Combine with coverage
    return sample_factor * max(0.1, min(1.0, coverage_ratio))


def compute_prediction_confidence(
    p10: float,
    p50: float,
    p90: float,
    max_acceptable_spread: float = 100.0,
) -> float:
    """
    Confidence factor based on prediction uncertainty.
    
    Narrow confidence intervals = high confidence.
    Wide intervals = low confidence.
    
    Args:
        p10: 10th percentile prediction
        p50: 50th percentile (median) prediction
        p90: 90th percentile prediction
        max_acceptable_spread: Maximum P90-P10 spread for full confidence
    
    Returns:
        Confidence factor 0-1
    """
    if p50 <= 0:
        return 0.1  # Invalid prediction
    
    spread = abs(p90 - p10)
    relative_spread = spread / max(abs(p50), 1.0)
    
    # Sigmoid-like mapping: small spread = high confidence
    # relative_spread of 0 -> 1.0, relative_spread of 2.0 -> 0.5
    confidence = 1.0 / (1.0 + relative_spread)
    return max(0.1, min(1.0, confidence))


def check_rul_reliability(
    maturity_state: str,
    training_rows: int,
    training_days: float,
    health_history_days: float,
    min_training_rows: int = 500,
    min_training_days: float = 7.0,
    min_health_history_days: float = 3.0,
) -> Tuple[ReliabilityStatus, str]:
    """
    Gate RUL predictions based on prerequisites.
    
    V11 Rule #10: RUL must be gated or suppressed.
    V11 Rule #20: If unsure, the system must say "not reliable".
    
    Args:
        maturity_state: Current model maturity (LEARNING, CONVERGED, etc.)
        training_rows: Number of training samples
        training_days: Days of training data
        health_history_days: Days of health history available
        min_training_rows: Minimum rows required
        min_training_days: Minimum training days required
        min_health_history_days: Minimum health history required
    
    Returns:
        Tuple of (status, reason)
    """
    maturity_upper = str(maturity_state).upper()
    
    # Check maturity first
    if maturity_upper == 'COLDSTART':
        return ReliabilityStatus.NOT_RELIABLE, "Model in COLDSTART state - no baseline established"
    
    if maturity_upper == 'LEARNING':
        return ReliabilityStatus.LEARNING, "Model still LEARNING - predictions may be unreliable"
    
    if maturity_upper == 'DEPRECATED':
        return ReliabilityStatus.NOT_RELIABLE, "Model DEPRECATED - needs refresh"
    
    # Check data requirements for CONVERGED model
    if training_rows < min_training_rows:
        return ReliabilityStatus.INSUFFICIENT_DATA, f"Insufficient training data: {training_rows} < {min_training_rows} rows"
    
    if training_days < min_training_days:
        return ReliabilityStatus.INSUFFICIENT_DATA, f"Insufficient training period: {training_days:.1f} < {min_training_days} days"
    
    if health_history_days < min_health_history_days:
        return ReliabilityStatus.INSUFFICIENT_DATA, f"Insufficient health history: {health_history_days:.1f} < {min_health_history_days} days"
    
    return ReliabilityStatus.RELIABLE, "All prerequisites met"


def compute_health_confidence(
    fused_z: float,
    regime_confidence: float = 1.0,
    maturity_state: str = "CONVERGED",
    sample_count: int = 1000,
) -> float:
    """
    Compute confidence for a health score.
    
    Args:
        fused_z: Fused anomaly Z-score
        regime_confidence: Confidence in regime assignment (0-1)
        maturity_state: Model maturity state
        sample_count: Training sample count
    
    Returns:
        Confidence 0-1
    """
    factors = ConfidenceFactors(
        maturity_factor=compute_maturity_confidence(maturity_state),
        data_quality_factor=compute_data_quality_confidence(sample_count),
        prediction_factor=1.0,  # Health is direct measurement, not prediction
        regime_factor=regime_confidence,
    )
    return factors.overall()


def compute_episode_confidence(
    episode_duration_seconds: float,
    peak_z: float,
    regime_confidence: float = 1.0,
    maturity_state: str = "CONVERGED",
    min_duration_seconds: float = 60.0,
) -> float:
    """
    Compute confidence for an episode detection.
    
    Short episodes or low peak Z = lower confidence.
    
    Args:
        episode_duration_seconds: Duration of episode
        peak_z: Peak Z-score during episode
        regime_confidence: Confidence in regime assignment
        maturity_state: Model maturity state
        min_duration_seconds: Minimum duration for full confidence
    
    Returns:
        Confidence 0-1
    """
    # Duration factor: longer episodes = more confident
    if episode_duration_seconds < min_duration_seconds:
        duration_factor = max(0.3, episode_duration_seconds / min_duration_seconds)
    else:
        duration_factor = 1.0
    
    # Peak Z factor: higher peak = more confident it's real
    # Z < 2 might be noise, Z > 5 is clearly anomalous
    z_factor = min(1.0, max(0.3, (peak_z - 1.0) / 4.0))
    
    factors = ConfidenceFactors(
        maturity_factor=compute_maturity_confidence(maturity_state),
        data_quality_factor=1.0,  # Episodes are from scored data
        prediction_factor=duration_factor * z_factor,
        regime_factor=regime_confidence,
    )
    return factors.overall()


def compute_rul_confidence(
    p10: float,
    p50: float,
    p90: float,
    maturity_state: str,
    training_rows: int,
    training_days: float,
) -> Tuple[float, ReliabilityStatus, str]:
    """
    Compute confidence and reliability for RUL estimate.
    
    Returns:
        Tuple of (confidence, reliability_status, reason)
    """
    # First check reliability gate
    status, reason = check_rul_reliability(
        maturity_state=maturity_state,
        training_rows=training_rows,
        training_days=training_days,
        health_history_days=training_days,  # Use training days as proxy
    )
    
    # If not reliable, return low confidence
    if status != ReliabilityStatus.RELIABLE:
        base_confidence = 0.3 if status == ReliabilityStatus.LEARNING else 0.1
        return base_confidence, status, reason
    
    # Compute confidence factors for reliable predictions
    factors = ConfidenceFactors(
        maturity_factor=compute_maturity_confidence(maturity_state),
        data_quality_factor=compute_data_quality_confidence(training_rows),
        prediction_factor=compute_prediction_confidence(p10, p50, p90),
        regime_factor=1.0,  # RUL is equipment-level, not regime-specific
    )
    
    return factors.overall(), status, reason

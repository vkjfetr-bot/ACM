"""
Failure Probability Estimation (v10.0.0)

Pure functions for converting health forecasts to failure probabilities.
Replaces duplicate logic from forecasting.py (lines 238-290, 1044-1104).

Key Features:
- Health-to-failure probability mapping via normal CDF
- Survival curve computation (reliability function)
- Hazard rate (instantaneous failure rate)
- Mean Time To Failure (MTTF) estimation
- Research-backed thresholds and transformations

References:
- Jardine & Tsang (2013): "Maintenance, Replacement, and Reliability" - failure modeling
- ISO 13381-1:2015: Prognostics and health management - failure threshold definitions
- Reliability Theory (Barlow & Proschan 1975): Survival/hazard relationships
"""

from typing import Optional
import numpy as np
from scipy import stats
from scipy import integrate


def health_to_failure_probability(
    health_forecast: np.ndarray,
    failure_threshold: float = 70.0,
    health_std: float = 10.0
) -> np.ndarray:
    """
    Convert health forecast to failure probability via normal CDF.
    
    Mathematical Model:
    - P(failure | health) = P(health < threshold)
    - Assume health ~ Normal(μ_forecast, σ_health)
    - P(failure) = Φ((threshold - μ_forecast) / σ_health)
    
    Where Φ is the standard normal CDF.
    
    Interpretation:
    - health >> threshold → P(failure) ≈ 0 (healthy)
    - health ≈ threshold → P(failure) ≈ 0.5 (marginal)
    - health << threshold → P(failure) ≈ 1 (critical)
    
    References:
    - ISO 13381-1:2015: Failure threshold typically 40-80% of nominal health
    - Normal approximation valid when health variance is moderate
    
    Args:
        health_forecast: Forecasted health values (0-100 scale)
        failure_threshold: Health level below which failure is imminent (default 70.0)
        health_std: Standard deviation of health noise (default 10.0)
    
    Returns:
        Failure probabilities (0-1 scale), same shape as health_forecast
    """
    if health_std <= 0:
        health_std = 10.0  # Guard against invalid std
    
    # Z-score: (threshold - forecast) / std
    # Positive z-score means forecast is below threshold (higher failure risk)
    z_scores = (failure_threshold - health_forecast) / health_std
    
    # CDF gives P(X < threshold) = P(failure)
    failure_probs = stats.norm.cdf(z_scores)
    
    # Clamp to [0, 1]
    failure_probs = np.clip(failure_probs, 0.0, 1.0)
    
    return failure_probs


def compute_survival_curve(failure_probabilities: np.ndarray) -> np.ndarray:
    """
    Compute survival curve (reliability function) from failure probabilities.
    
    Mathematical Definition:
    - S(t) = 1 - F(t) = P(T > t)
    - Where F(t) is cumulative failure probability
    - S(t) represents probability equipment survives to time t
    
    Properties:
    - S(0) = 1 (certain survival at t=0)
    - S(∞) = 0 (eventual failure)
    - S(t) is monotonically decreasing
    
    References:
    - Reliability Theory (Barlow & Proschan 1975)
    - Survival Analysis (Cox & Oakes 1984)
    
    Args:
        failure_probabilities: Cumulative failure probabilities F(t)
    
    Returns:
        Survival probabilities S(t) = 1 - F(t)
    """
    survival_probs = 1.0 - failure_probabilities
    
    # Guard against numerical issues (S(t) must be in [0, 1])
    survival_probs = np.clip(survival_probs, 0.0, 1.0)
    
    # Enforce monotonicity (survival can only decrease)
    for i in range(1, len(survival_probs)):
        if survival_probs[i] > survival_probs[i - 1]:
            survival_probs[i] = survival_probs[i - 1]
    
    return survival_probs


def compute_hazard_rate(
    failure_probabilities: np.ndarray,
    dt_hours: float = 1.0
) -> np.ndarray:
    """
    Compute discrete hazard rate (instantaneous failure rate) from cumulative failure probabilities.
    
    Mathematical Derivation:
    - Continuous: λ(t) = f(t) / S(t) = -d[ln S(t)] / dt
    - Discrete: λ[i] = [F[i] - F[i-1]] / [S[i-1] * dt]
    - Where F = cumulative failure prob, S = survival prob
    
    Physical Interpretation:
    - λ(t) = probability of failure in next dt, given survival to t
    - Units: failures per hour
    - Increasing λ(t) indicates accelerating degradation (wear-out phase)
    - Constant λ(t) indicates random failures (useful life phase)
    
    References:
    - Reliability Engineering (Ebeling 2010): Hazard rate fundamentals
    - ISO 13381-1:2015: Prognostics terminology
    
    Args:
        failure_probabilities: Cumulative failure probabilities F(t)
        dt_hours: Time interval between observations (hours)
    
    Returns:
        Hazard rates λ(t) in failures per hour
    """
    n = len(failure_probabilities)
    if n == 0:
        return np.array([])
    
    # Compute survival probabilities
    survival_probs = compute_survival_curve(failure_probabilities)
    
    # Initialize hazard rate array
    hazard_rates = np.zeros(n)
    
    # First point: use forward difference
    if n > 1:
        delta_f = failure_probabilities[1] - failure_probabilities[0]
        s_prev = max(survival_probs[0], 1e-6)  # Guard against division by zero
        hazard_rates[0] = max(0.0, delta_f / (s_prev * dt_hours))
    
    # Subsequent points: use backward difference
    for i in range(1, n):
        delta_f = failure_probabilities[i] - failure_probabilities[i - 1]
        s_prev = max(survival_probs[i - 1], 1e-6)
        hazard_rates[i] = max(0.0, delta_f / (s_prev * dt_hours))
    
    # Clamp extreme values (hazard rate should be reasonable)
    hazard_rates = np.clip(hazard_rates, 0.0, 1.0)  # Max 1 failure/hour
    
    return hazard_rates


def mean_time_to_failure(
    survival_probabilities: np.ndarray,
    dt_hours: float = 1.0
) -> float:
    """
    Estimate Mean Time To Failure (MTTF) from survival curve.
    
    Mathematical Definition:
    - MTTF = ∫[0,∞] S(t) dt
    - Discrete approximation: MTTF ≈ Σ S[i] * dt
    
    Physical Interpretation:
    - Expected remaining lifetime in hours
    - Area under survival curve
    - Lower MTTF indicates imminent failure
    
    Limitations:
    - Assumes survival curve extends to S(t) ≈ 0
    - Truncated forecasts underestimate MTTF
    - Use with caution for short forecast horizons
    
    References:
    - Reliability Engineering (Ebeling 2010): MTTF definition
    - Jardine & Tsang (2013): MTTF estimation from prognostics
    
    Args:
        survival_probabilities: Survival curve S(t)
        dt_hours: Time interval between observations (hours)
    
    Returns:
        Mean Time To Failure in hours
    """
    if len(survival_probabilities) == 0:
        return 0.0
    
    # Trapezoidal integration: MTTF = ∫ S(t) dt
    mttf = float(integrate.trapezoid(survival_probabilities, dx=dt_hours))
    
    # Guard against negative/invalid values
    mttf = max(0.0, mttf)
    
    return mttf


def compute_failure_statistics(
    health_forecast: np.ndarray,
    failure_threshold: float = 70.0,
    health_std: float = 10.0,
    dt_hours: float = 1.0
) -> dict:
    """
    Compute comprehensive failure statistics from health forecast.
    
    Aggregates all failure probability metrics in one pass:
    - Failure probabilities F(t)
    - Survival probabilities S(t)
    - Hazard rates λ(t)
    - Mean Time To Failure (MTTF)
    - Time to 50% failure probability (median RUL proxy)
    
    Args:
        health_forecast: Forecasted health values
        failure_threshold: Health threshold for failure definition
        health_std: Standard deviation of health noise
        dt_hours: Time interval between observations (hours)
    
    Returns:
        Dictionary with keys:
        - 'failure_probs': F(t) array
        - 'survival_probs': S(t) array
        - 'hazard_rates': λ(t) array
        - 'mttf_hours': Mean Time To Failure
        - 'median_rul_hours': Time to 50% failure probability
        - 'max_failure_prob': Maximum failure probability in forecast horizon
    """
    # Compute failure probabilities
    failure_probs = health_to_failure_probability(
        health_forecast,
        failure_threshold=failure_threshold,
        health_std=health_std
    )
    
    # Compute survival curve
    survival_probs = compute_survival_curve(failure_probs)
    
    # Compute hazard rates
    hazard_rates = compute_hazard_rate(failure_probs, dt_hours=dt_hours)
    
    # Compute MTTF
    mttf = mean_time_to_failure(survival_probs, dt_hours=dt_hours)
    
    # Find median RUL (time to 50% failure probability)
    median_rul_hours = 0.0
    for i, fp in enumerate(failure_probs):
        if fp >= 0.5:
            median_rul_hours = i * dt_hours
            break
    if median_rul_hours == 0.0 and len(failure_probs) > 0:
        # If never exceeds 50%, use last index
        median_rul_hours = len(failure_probs) * dt_hours
    
    # Maximum failure probability
    max_failure_prob = float(np.max(failure_probs)) if len(failure_probs) > 0 else 0.0
    
    return {
        'failure_probs': failure_probs,
        'survival_probs': survival_probs,
        'hazard_rates': hazard_rates,
        'mttf_hours': mttf,
        'median_rul_hours': median_rul_hours,
        'max_failure_prob': max_failure_prob
    }


def health_to_rul_simple(
    current_health: float,
    failure_threshold: float = 70.0,
    degradation_rate: float = 0.1
) -> float:
    """
    Simple linear RUL estimate (no uncertainty).
    
    Formula:
    - RUL = (current_health - failure_threshold) / degradation_rate
    
    Assumptions:
    - Linear degradation at constant rate
    - No uncertainty quantification
    - Deterministic model
    
    Use Case:
    - Quick RUL estimate for dashboards
    - Fallback when Monte Carlo simulation unavailable
    
    Args:
        current_health: Current health index value
        failure_threshold: Health threshold for failure
        degradation_rate: Health degradation per hour (positive = declining)
    
    Returns:
        Remaining Useful Life in hours (0 if already below threshold)
    """
    if current_health <= failure_threshold:
        return 0.0
    
    if degradation_rate <= 0:
        return float('inf')  # Not degrading
    
    rul_hours = (current_health - failure_threshold) / degradation_rate
    
    return max(0.0, rul_hours)

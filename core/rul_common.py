"""
Common helpers for RUL estimation modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from math import erf, sqrt

def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF that accepts scalars or numpy arrays."""
    arr = np.asarray(x, dtype=float)
    scaled = arr / sqrt(2.0)
    # math.erf only supports scalars; vectorize to handle numpy arrays.
    erf_vec = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(scaled))

@dataclass
class RULConfig:
    health_threshold: float = 70.0
    min_points: int = 20
    max_forecast_hours: float = 24.0
    maintenance_risk_low: float = 0.2
    maintenance_risk_high: float = 0.5
    
    # Enhanced specific (optional here, but good to have base)
    learning_rate: float = 0.1
    min_model_weight: float = 0.05
    enable_online_learning: bool = True
    calibration_window: int = 50

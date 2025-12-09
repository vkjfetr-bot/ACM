"""
Forecast Quality Metrics (v10.0.0)

Statistical metrics for evaluating forecast quality and model performance.
Replaces duplicate logic from forecasting.py (lines 1189-1339).

Key Features:
- Standard regression metrics (MAE, RMSE, MAPE)
- Forecast-specific metrics (bias, directional accuracy)
- Confidence interval coverage and sharpness
- Theil's U statistic for model comparison
- Comprehensive diagnostics for model validation

References:
- Hyndman & Koehler (2006): "Another look at measures of forecast accuracy"
- Theil (1966): Applied Economic Forecasting - Theil's U statistic
- Diebold & Mariano (1995): Forecast accuracy tests
"""

from typing import Dict, Optional
import numpy as np
from scipy import stats

from utils.logger import Console


def compute_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).
    
    Formula: MAE = (1/n) * Σ|actual - forecast|
    
    Interpretation:
    - Average magnitude of forecast errors
    - Units same as original data
    - Less sensitive to outliers than RMSE
    
    Args:
        actual: Ground truth values
        forecast: Forecasted values
    
    Returns:
        MAE value (lower is better)
    """
    valid = np.isfinite(actual) & np.isfinite(forecast)
    if not np.any(valid):
        return float('nan')
    
    errors = np.abs(actual[valid] - forecast[valid])
    mae = float(np.mean(errors))
    
    return mae


def compute_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE).
    
    Formula: RMSE = sqrt((1/n) * Σ(actual - forecast)²)
    
    Interpretation:
    - Penalizes large errors more than MAE
    - Units same as original data
    - Standard regression metric
    
    Args:
        actual: Ground truth values
        forecast: Forecasted values
    
    Returns:
        RMSE value (lower is better)
    """
    valid = np.isfinite(actual) & np.isfinite(forecast)
    if not np.any(valid):
        return float('nan')
    
    errors = actual[valid] - forecast[valid]
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    return rmse


def compute_mape(actual: np.ndarray, forecast: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    
    Formula: MAPE = (100/n) * Σ|(actual - forecast) / actual|
    
    Interpretation:
    - Percentage error metric (scale-independent)
    - 0% = perfect forecast, 100% = large errors
    - Undefined when actual = 0 (uses epsilon guard)
    
    Args:
        actual: Ground truth values
        forecast: Forecasted values
        epsilon: Small constant to avoid division by zero
    
    Returns:
        MAPE percentage (lower is better)
    """
    valid = np.isfinite(actual) & np.isfinite(forecast)
    if not np.any(valid):
        return float('nan')
    
    actual_valid = actual[valid]
    forecast_valid = forecast[valid]
    
    # Guard against division by zero
    actual_safe = np.where(np.abs(actual_valid) < epsilon, epsilon, actual_valid)
    
    percentage_errors = np.abs((actual_valid - forecast_valid) / actual_safe) * 100.0
    mape = float(np.mean(percentage_errors))
    
    return mape


def compute_forecast_bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Forecast bias (directional error).
    
    Formula: Bias = (1/n) * Σ(actual - forecast)
    
    Interpretation:
    - Positive bias: forecast underestimates (optimistic)
    - Negative bias: forecast overestimates (pessimistic)
    - Zero bias: unbiased forecasts
    
    Args:
        actual: Ground truth values
        forecast: Forecasted values
    
    Returns:
        Bias value (positive = under-forecast, negative = over-forecast)
    """
    valid = np.isfinite(actual) & np.isfinite(forecast)
    if not np.any(valid):
        return float('nan')
    
    errors = actual[valid] - forecast[valid]
    bias = float(np.mean(errors))
    
    return bias


def compute_coverage_probability(
    actual: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> float:
    """
    Confidence interval coverage probability.
    
    Formula: Coverage = (1/n) * Σ I(actual ∈ [lower, upper])
    
    Where I is indicator function (1 if true, 0 if false).
    
    Interpretation:
    - Proportion of actual values within prediction intervals
    - Target: ~95% for 95% confidence intervals
    - Lower coverage = intervals too narrow (overconfident)
    - Higher coverage = intervals too wide (underconfident)
    
    Args:
        actual: Ground truth values
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
    
    Returns:
        Coverage probability (0-1 scale)
    """
    valid = (np.isfinite(actual) & 
             np.isfinite(lower_bound) & 
             np.isfinite(upper_bound))
    
    if not np.any(valid):
        return float('nan')
    
    actual_valid = actual[valid]
    lower_valid = lower_bound[valid]
    upper_valid = upper_bound[valid]
    
    within_interval = (actual_valid >= lower_valid) & (actual_valid <= upper_valid)
    coverage = float(np.mean(within_interval))
    
    return coverage


def compute_sharpness(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """
    Confidence interval sharpness (average width).
    
    Formula: Sharpness = (1/n) * Σ(upper - lower)
    
    Interpretation:
    - Average width of prediction intervals
    - Lower sharpness = more precise forecasts
    - Trade-off with coverage: narrow intervals may have poor coverage
    
    Args:
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
    
    Returns:
        Average interval width
    """
    valid = np.isfinite(lower_bound) & np.isfinite(upper_bound)
    
    if not np.any(valid):
        return float('nan')
    
    widths = upper_bound[valid] - lower_bound[valid]
    sharpness = float(np.mean(widths))
    
    return sharpness


def compute_directional_accuracy(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Directional accuracy (trend prediction).
    
    Formula: DA = (1/n) * Σ I(sign(Δactual) == sign(Δforecast))
    
    Interpretation:
    - Proportion of correctly predicted trend directions
    - 100% = all trend directions correct
    - 50% = random guessing
    - Useful for trend-following applications
    
    Args:
        actual: Ground truth values
        forecast: Forecasted values
    
    Returns:
        Directional accuracy (0-1 scale)
    """
    valid = np.isfinite(actual) & np.isfinite(forecast)
    
    if np.sum(valid) < 2:
        return float('nan')
    
    actual_valid = actual[valid]
    forecast_valid = forecast[valid]
    
    # Compute differences
    actual_diff = np.diff(actual_valid)
    forecast_diff = np.diff(forecast_valid)
    
    # Ignore near-zero changes (noise threshold)
    threshold = 0.01 * np.std(actual_diff) if np.std(actual_diff) > 0 else 1e-6
    significant = np.abs(actual_diff) > threshold
    
    if not np.any(significant):
        return 1.0  # All changes insignificant
    
    # Sign agreement
    actual_sign = np.sign(actual_diff[significant])
    forecast_sign = np.sign(forecast_diff[significant])
    
    directional_acc = float(np.mean(actual_sign == forecast_sign))
    
    return directional_acc


def compute_theils_u(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Theil's U statistic (forecast vs naive forecast comparison).
    
    Formula: U = sqrt(Σ(forecast - actual)²) / sqrt(Σ(actual[t] - actual[t-1])²)
    
    Interpretation:
    - U < 1: forecast better than naive (persistence) model
    - U = 1: forecast equivalent to naive model
    - U > 1: forecast worse than naive model (consider simpler methods)
    
    References:
    - Theil (1966): Applied Economic Forecasting
    
    Args:
        actual: Ground truth values
        forecast: Forecasted values
    
    Returns:
        Theil's U statistic
    """
    valid = np.isfinite(actual) & np.isfinite(forecast)
    
    if np.sum(valid) < 2:
        return float('nan')
    
    actual_valid = actual[valid]
    forecast_valid = forecast[valid]
    
    # Forecast error
    forecast_errors = forecast_valid - actual_valid
    forecast_mse = np.mean(forecast_errors ** 2)
    
    # Naive forecast error (persistence: forecast[t] = actual[t-1])
    naive_errors = actual_valid[1:] - actual_valid[:-1]
    naive_mse = np.mean(naive_errors ** 2)
    
    if naive_mse < 1e-10:
        return float('nan')  # Cannot compare when series is constant
    
    theils_u = float(np.sqrt(forecast_mse / naive_mse))
    
    return theils_u


def compute_comprehensive_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive forecast quality metrics in one pass.
    
    Includes:
    - MAE, RMSE, MAPE (standard regression metrics)
    - Bias (directional error)
    - Coverage and sharpness (confidence interval quality)
    - Directional accuracy (trend prediction)
    - Theil's U (model comparison)
    
    Args:
        actual: Ground truth values
        forecast: Forecasted values
        lower_bound: Optional lower confidence bound
        upper_bound: Optional upper confidence bound
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'mae': compute_mae(actual, forecast),
        'rmse': compute_rmse(actual, forecast),
        'mape': compute_mape(actual, forecast),
        'bias': compute_forecast_bias(actual, forecast),
        'directional_accuracy': compute_directional_accuracy(actual, forecast),
        'theils_u': compute_theils_u(actual, forecast),
        'n_samples': float(np.sum(np.isfinite(actual) & np.isfinite(forecast)))
    }
    
    # Add interval metrics if bounds provided
    if lower_bound is not None and upper_bound is not None:
        metrics['coverage_probability'] = compute_coverage_probability(actual, lower_bound, upper_bound)
        metrics['interval_sharpness'] = compute_sharpness(lower_bound, upper_bound)
    else:
        metrics['coverage_probability'] = float('nan')
        metrics['interval_sharpness'] = float('nan')
    
    return metrics


def log_metrics_summary(metrics: Dict[str, float], prefix: str = "[Metrics]") -> None:
    """
    Log metrics summary to console.
    
    Args:
        metrics: Dictionary of computed metrics
        prefix: Log message prefix
    """
    Console.info(
        f"{prefix} MAE={metrics.get('mae', float('nan')):.2f}, "
        f"RMSE={metrics.get('rmse', float('nan')):.2f}, "
        f"MAPE={metrics.get('mape', float('nan')):.1f}%, "
        f"Bias={metrics.get('bias', float('nan')):.2f}, "
        f"DA={metrics.get('directional_accuracy', float('nan')):.2%}, "
        f"TheilU={metrics.get('theils_u', float('nan')):.3f}, "
        f"Coverage={metrics.get('coverage_probability', float('nan')):.2%}, "
        f"Sharpness={metrics.get('interval_sharpness', float('nan')):.2f}, "
        f"N={int(metrics.get('n_samples', 0))}"
    )

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

from core.observability import Console


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


def compute_forecast_diagnostics(
    forecast_results: Dict[str, any],
    data_summary: Optional[any] = None
) -> Dict[str, any]:
    """
    Compute forecast diagnostics for ForecastEngine integration (M9).
    
    This function computes diagnostic metrics from forecast results without
    requiring actual future values (which aren't available at forecast time).
    It calculates:
    - Forecast statistics (mean, std, range)
    - RUL uncertainty metrics (spread, coefficient of variation)
    - Confidence interval width and asymmetry
    - Data quality indicators from DataSummary
    
    Args:
        forecast_results: Dictionary from ForecastEngine._generate_forecast_and_rul()
            Expected keys: rul_p10, rul_p50, rul_p90, rul_mean, rul_std,
                          forecast_values, forecast_lower, forecast_upper,
                          current_health, mttf_hours
        data_summary: Optional DataSummary object with dt_hours, n_samples, quality
    
    Returns:
        Flat dictionary suitable for storing in ForecastingState.last_forecast_json:
        - forecast_mean: Mean of health forecast values
        - forecast_std: Std dev of health forecast values
        - forecast_range: Max - min of forecast values
        - rul_spread: P90 - P10 (uncertainty width)
        - rul_cv: Coefficient of variation (std/mean) for RUL
        - ci_width_mean: Mean confidence interval width
        - ci_asymmetry: Upper-to-lower bound ratio (>1 = skewed upward)
        - current_health: Health at forecast start
        - mttf_hours: Mean time to failure
        - n_samples: Number of training samples (from DataSummary)
        - data_quality: Quality classification (from DataSummary)
        - dt_hours: Data cadence (from DataSummary)
        - computed_at: Timestamp when diagnostics computed
    """
    diagnostics: Dict[str, float] = {}
    
    # Extract forecast arrays
    forecast_values = forecast_results.get('forecast_values')
    forecast_lower = forecast_results.get('forecast_lower')
    forecast_upper = forecast_results.get('forecast_upper')
    
    # Forecast statistics
    if forecast_values is not None and len(forecast_values) > 0:
        forecast_arr = np.asarray(forecast_values)
        valid = np.isfinite(forecast_arr)
        if np.any(valid):
            diagnostics['forecast_mean'] = float(np.mean(forecast_arr[valid]))
            diagnostics['forecast_std'] = float(np.std(forecast_arr[valid]))
            diagnostics['forecast_range'] = float(np.max(forecast_arr[valid]) - np.min(forecast_arr[valid]))
        else:
            diagnostics['forecast_mean'] = float('nan')
            diagnostics['forecast_std'] = float('nan')
            diagnostics['forecast_range'] = float('nan')
    else:
        diagnostics['forecast_mean'] = float('nan')
        diagnostics['forecast_std'] = float('nan')
        diagnostics['forecast_range'] = float('nan')
    
    # RUL uncertainty metrics
    rul_p10 = forecast_results.get('rul_p10', float('nan'))
    rul_p50 = forecast_results.get('rul_p50', float('nan'))
    rul_p90 = forecast_results.get('rul_p90', float('nan'))
    rul_mean = forecast_results.get('rul_mean', float('nan'))
    rul_std = forecast_results.get('rul_std', float('nan'))
    
    # RUL spread (P90 - P10)
    if np.isfinite(rul_p10) and np.isfinite(rul_p90):
        diagnostics['rul_spread'] = float(rul_p90 - rul_p10)
    else:
        diagnostics['rul_spread'] = float('nan')
    
    # RUL coefficient of variation (CV = std / mean)
    if np.isfinite(rul_std) and np.isfinite(rul_mean) and rul_mean > 0:
        diagnostics['rul_cv'] = float(rul_std / rul_mean)
    else:
        diagnostics['rul_cv'] = float('nan')
    
    # Confidence interval metrics
    if forecast_lower is not None and forecast_upper is not None:
        lower_arr = np.asarray(forecast_lower)
        upper_arr = np.asarray(forecast_upper)
        valid = np.isfinite(lower_arr) & np.isfinite(upper_arr)
        
        if np.any(valid):
            widths = upper_arr[valid] - lower_arr[valid]
            diagnostics['ci_width_mean'] = float(np.mean(widths))
            
            # Asymmetry: ratio of upper to lower bound widths relative to forecast mean
            if forecast_values is not None and len(forecast_values) > 0:
                forecast_arr = np.asarray(forecast_values)
                valid_all = valid & np.isfinite(forecast_arr)
                if np.any(valid_all):
                    upper_widths = upper_arr[valid_all] - forecast_arr[valid_all]
                    lower_widths = forecast_arr[valid_all] - lower_arr[valid_all]
                    # Avoid division by zero
                    lower_mean = np.mean(np.abs(lower_widths))
                    upper_mean = np.mean(np.abs(upper_widths))
                    if lower_mean > 1e-6:
                        diagnostics['ci_asymmetry'] = float(upper_mean / lower_mean)
                    else:
                        diagnostics['ci_asymmetry'] = float('nan')
                else:
                    diagnostics['ci_asymmetry'] = float('nan')
            else:
                diagnostics['ci_asymmetry'] = float('nan')
        else:
            diagnostics['ci_width_mean'] = float('nan')
            diagnostics['ci_asymmetry'] = float('nan')
    else:
        diagnostics['ci_width_mean'] = float('nan')
        diagnostics['ci_asymmetry'] = float('nan')
    
    # Current health and MTTF
    diagnostics['current_health'] = float(forecast_results.get('current_health', float('nan')))
    diagnostics['mttf_hours'] = float(forecast_results.get('mttf_hours', float('nan')))
    diagnostics['rul_p50'] = float(rul_p50) if np.isfinite(rul_p50) else float('nan')
    
    # DataSummary metrics (M9 integration with M3)
    if data_summary is not None:
        diagnostics['n_samples'] = int(getattr(data_summary, 'n_samples', 0))
        diagnostics['dt_hours'] = float(getattr(data_summary, 'dt_hours', float('nan')))
        # Quality is enum, convert to string
        quality = getattr(data_summary, 'quality', None)
        diagnostics['data_quality'] = quality.value if quality else 'UNKNOWN'
    else:
        diagnostics['n_samples'] = 0
        diagnostics['dt_hours'] = float('nan')
        diagnostics['data_quality'] = 'UNKNOWN'
    
    # Timestamp
    from datetime import datetime
    diagnostics['computed_at'] = datetime.now().isoformat()
    
    return diagnostics


def log_forecast_diagnostics(diagnostics: Dict[str, any], prefix: str = "[ForecastDiag]") -> None:
    """
    Log forecast diagnostics summary to console.
    
    Args:
        diagnostics: Dictionary from compute_forecast_diagnostics()
        prefix: Log message prefix (deprecated, uses component now)
    """
    Console.info(
        f"RUL_P50={diagnostics.get('rul_p50', float('nan')):.1f}h, "
        f"RUL_Spread={diagnostics.get('rul_spread', float('nan')):.1f}h, "
        f"RUL_CV={diagnostics.get('rul_cv', float('nan')):.2f}, "
        f"CI_Width={diagnostics.get('ci_width_mean', float('nan')):.2f}, "
        f"Health={diagnostics.get('current_health', float('nan')):.1f}, "
        f"N={diagnostics.get('n_samples', 0)}, "
        f"Quality={diagnostics.get('data_quality', 'UNKNOWN')}",
        component="FORECAST"
    )


# =============================================================================
# CALIBRATION SCORE AND ROLLING BACKTEST (v10.1.0)
# =============================================================================
# Proper probabilistic calibration assessment and time-series backtesting
# Reference: Gneiting et al. (2007) "Probabilistic forecasts, calibration and sharpness"
# =============================================================================

def compute_calibration_score(
    actual: np.ndarray,
    forecast: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    coverage_levels: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute probabilistic calibration score for forecast intervals.
    
    Calibration measures whether predicted probabilities match observed frequencies.
    For 95% CI, ~95% of actuals should fall within bounds. Deviation from this
    indicates miscalibration (overconfident or underconfident forecasts).
    
    This implements proper calibration assessment per Gneiting et al. (2007):
    1. Check coverage at multiple probability levels (not just 95%)
    2. Compute calibration error as deviation from ideal coverage
    3. Return single calibration score (lower = better calibrated)
    
    Reference:
        Gneiting, Balabdaoui, Raftery (2007): "Probabilistic forecasts, 
        calibration and sharpness", JRSS-B 69(2):243-268
    
    Args:
        actual: Observed values
        forecast: Point forecasts (used to construct symmetric intervals)
        lower_bound: Lower confidence bounds (e.g., 2.5th percentile)
        upper_bound: Upper confidence bounds (e.g., 97.5th percentile)
        coverage_levels: Target coverage levels to check (default: [0.50, 0.80, 0.90, 0.95])
    
    Returns:
        Dictionary with:
        - calibration_error: Mean absolute deviation from ideal coverage (0 = perfect)
        - calibration_score: 1 - calibration_error (1 = perfect, 0 = completely miscalibrated)
        - coverage_95: Observed coverage at 95% level
        - overconfident: True if intervals too narrow (coverage < target)
        - coverage_by_level: Dict of observed coverage at each level
    """
    if coverage_levels is None:
        coverage_levels = [0.50, 0.80, 0.90, 0.95]
    
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    lower_bound = np.asarray(lower_bound, dtype=float)
    upper_bound = np.asarray(upper_bound, dtype=float)
    
    # Filter valid data
    valid = (np.isfinite(actual) & np.isfinite(forecast) & 
             np.isfinite(lower_bound) & np.isfinite(upper_bound))
    
    if np.sum(valid) < 10:
        return {
            'calibration_error': float('nan'),
            'calibration_score': float('nan'),
            'coverage_95': float('nan'),
            'overconfident': False,
            'coverage_by_level': {}
        }
    
    actual_v = actual[valid]
    forecast_v = forecast[valid]
    lower_v = lower_bound[valid]
    upper_v = upper_bound[valid]
    
    # Compute interval width (assumes symmetric intervals for width scaling)
    full_width = upper_v - lower_v
    half_width = full_width / 2.0
    
    # Check coverage at different levels by scaling the interval width
    # For 95% CI → scale factor 1.0
    # For 80% CI → scale factor ~0.60 (from normal quantiles: 1.28/1.96)
    # For 50% CI → scale factor ~0.34 (from normal quantiles: 0.67/1.96)
    
    scale_factors = {
        0.50: 0.6745 / 1.96,  # z_0.75 / z_0.975
        0.80: 1.282 / 1.96,   # z_0.90 / z_0.975
        0.90: 1.645 / 1.96,   # z_0.95 / z_0.975
        0.95: 1.0             # Full width
    }
    
    coverage_by_level = {}
    calibration_errors = []
    
    for level in coverage_levels:
        scale = scale_factors.get(level, level)  # Fallback to level as scale
        
        # Scale interval around forecast midpoint
        midpoint = (lower_v + upper_v) / 2.0
        scaled_lower = midpoint - half_width * scale
        scaled_upper = midpoint + half_width * scale
        
        # Compute observed coverage
        within = (actual_v >= scaled_lower) & (actual_v <= scaled_upper)
        observed_coverage = float(np.mean(within))
        
        coverage_by_level[f'coverage_{int(level*100):02d}'] = observed_coverage
        
        # Calibration error = |observed - expected|
        cal_error = abs(observed_coverage - level)
        calibration_errors.append(cal_error)
    
    # Aggregate calibration metrics
    calibration_error = float(np.mean(calibration_errors))
    calibration_score = 1.0 - calibration_error  # Higher = better
    
    # Check if overconfident (coverage below target)
    coverage_95 = coverage_by_level.get('coverage_95', float('nan'))
    overconfident = coverage_95 < 0.90 if np.isfinite(coverage_95) else False
    
    return {
        'calibration_error': calibration_error,
        'calibration_score': calibration_score,
        'coverage_95': coverage_95,
        'overconfident': overconfident,
        'coverage_by_level': coverage_by_level
    }


def rolling_backtest(
    health_series: np.ndarray,
    forecast_model,
    horizon: int = 24,
    min_train_samples: int = 100,
    step_size: int = 1
) -> Dict[str, float]:
    """
    Perform rolling backtest on forecast model with expanding window.
    
    Process:
    1. Train model on data up to time t
    2. Forecast h steps ahead
    3. Compare forecast to actual values at t+h
    4. Roll forward by step_size and repeat
    5. Aggregate metrics across all windows
    
    This is the gold standard for forecast validation because it:
    - Simulates real deployment (no future information leakage)
    - Tests model across multiple time points
    - Measures both point forecast and interval accuracy
    
    Reference:
        Hyndman & Athanasopoulos (2018): "Forecasting: Principles and Practice", 
        Chapter 5.4: Evaluating forecast accuracy
    
    Args:
        health_series: Full time series of health values
        forecast_model: Model with fit(data) and predict(steps) methods
            - fit(data: np.ndarray) -> self
            - predict(steps: int) -> ForecastResult with point_forecast, ci_lower, ci_upper
        horizon: Forecast horizon (steps ahead)
        min_train_samples: Minimum training samples before first backtest
        step_size: Steps to roll forward between tests (1 = exhaustive, higher = faster)
    
    Returns:
        Dictionary with aggregated backtest metrics:
        - mae: Mean absolute error across all backtests
        - rmse: Root mean squared error
        - mape: Mean absolute percentage error
        - directional_accuracy: Fraction of correct trend predictions
        - coverage_95: Fraction of actuals within 95% CI
        - n_backtests: Number of backtest windows
        - theils_u: Theil's U vs naive model
        - calibration_score: Probabilistic calibration quality
    """
    n = len(health_series)
    
    if n < min_train_samples + horizon:
        Console.warning(f"[Backtest] Insufficient data: {n} samples < {min_train_samples + horizon} required")
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'mape': float('nan'),
            'directional_accuracy': float('nan'),
            'coverage_95': float('nan'),
            'n_backtests': 0,
            'theils_u': float('nan'),
            'calibration_score': float('nan')
        }
    
    # Collect all forecasts and actuals
    all_actuals = []
    all_forecasts = []
    all_lower = []
    all_upper = []
    
    # Rolling window backtest
    for t in range(min_train_samples, n - horizon, step_size):
        try:
            # Train on data up to time t
            train_data = health_series[:t]
            forecast_model.fit(train_data)
            
            # Forecast horizon steps ahead
            result = forecast_model.predict(steps=horizon)
            
            # Get forecast at horizon
            if hasattr(result, 'point_forecast') and len(result.point_forecast) >= horizon:
                forecast_value = result.point_forecast[horizon - 1]
                
                # Get CI bounds if available
                ci_lower = result.ci_lower[horizon - 1] if hasattr(result, 'ci_lower') and result.ci_lower is not None else np.nan
                ci_upper = result.ci_upper[horizon - 1] if hasattr(result, 'ci_upper') and result.ci_upper is not None else np.nan
            else:
                continue  # Skip if prediction failed
            
            # Actual value at t + horizon
            actual_value = health_series[t + horizon - 1]
            
            all_actuals.append(actual_value)
            all_forecasts.append(forecast_value)
            all_lower.append(ci_lower)
            all_upper.append(ci_upper)
            
        except Exception:
            continue  # Skip failed backtests
    
    n_backtests = len(all_actuals)
    
    if n_backtests < 5:
        Console.warning(f"[Backtest] Only {n_backtests} successful backtests")
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'mape': float('nan'),
            'directional_accuracy': float('nan'),
            'coverage_95': float('nan'),
            'n_backtests': n_backtests,
            'theils_u': float('nan'),
            'calibration_score': float('nan')
        }
    
    # Convert to arrays
    actuals = np.array(all_actuals)
    forecasts = np.array(all_forecasts)
    lowers = np.array(all_lower)
    uppers = np.array(all_upper)
    
    # Compute metrics using existing functions
    mae = compute_mae(actuals, forecasts)
    rmse = compute_rmse(actuals, forecasts)
    mape = compute_mape(actuals, forecasts)
    directional_acc = compute_directional_accuracy(actuals, forecasts)
    theils_u = compute_theils_u(actuals, forecasts)
    
    # Coverage
    valid_ci = np.isfinite(lowers) & np.isfinite(uppers)
    if np.sum(valid_ci) > 5:
        coverage_95 = compute_coverage_probability(actuals[valid_ci], lowers[valid_ci], uppers[valid_ci])
        
        # Calibration score
        cal_result = compute_calibration_score(actuals[valid_ci], forecasts[valid_ci], 
                                               lowers[valid_ci], uppers[valid_ci])
        calibration_score = cal_result['calibration_score']
    else:
        coverage_95 = float('nan')
        calibration_score = float('nan')
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_acc,
        'coverage_95': coverage_95,
        'n_backtests': n_backtests,
        'theils_u': theils_u,
        'calibration_score': calibration_score
    }

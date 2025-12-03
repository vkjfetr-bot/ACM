# Forecasting.py Comprehensive Audit - Task List

**Generated**: December 3, 2025  
**Source**: Comprehensive Audit of forecasting.py  
**Priority Scale**: P0 (Critical) → P1 (Major) → P2 (Important) → P3 (Nice-to-have)

---

## 🔴 P0 - CRITICAL ISSUES (Must Fix Immediately)

### ✅ Task 1.1: Fix Initial Trend Calculation in Holt's Method
**File**: `core/forecasting.py` (Lines 682-708)  
**Issue**: Initial trend doesn't account for dt_hours, causing forecast divergence  
**Current Code**:
```python
trend = float(health_values.iloc[1] - health_values.iloc[0]) if n > 1 else 0.0
```
**Required Fix**:
```python
trend = float(health_values.iloc[1] - health_values.iloc[0]) / dt_hours if n > 1 else 0.0
```
**Impact**: High forecast error due to trend overestimation/underestimation by factor of dt_hours  
**Effort**: Low (1 line change)  
**Testing**: Validate forecast slopes match actual degradation rates

---

### ✅ Task 1.2: Correct Variance Growth Formula for Confidence Intervals
**File**: `core/forecasting.py` (Lines 733-735)  
**Issue**: Incorrect variance formula for Holt's Linear Trend method  
**Current Code**:
```python
var_mult = np.sqrt(1.0 + (alpha ** 2) * h + (beta ** 2) * (h ** 2))
horizon_std = std_error * var_mult
```
**Required Fix**: Implement correct Holt's variance formula
```python
def holt_variance_multiplier(h, alpha, beta):
    """Correct variance multiplier for Holt's Linear Trend."""
    if h == 1:
        return 1.0
    return 1.0 + (h - 1) * (alpha**2 + alpha*beta*h + beta**2 * h * (h + 1) / 2)
```
**Impact**: Confidence intervals incorrectly sized, undermining uncertainty quantification  
**Effort**: Low (function replacement)  
**Testing**: Verify CI coverage (95% CI should contain 95% of actuals)

---

### ✅ Task 1.3: Fix Hazard Rate Calculation Logic
**File**: `core/forecasting.py` (Lines 298-306)  
**Issue**: Treating failure probability incorrectly - confusing cumulative vs. incremental  
**Current Code**:
```python
lambda_raw_np = -np.log(1 - p_np) / dt_vec
```
**Required Fix**: Convert cumulative failure probability to hazard rate correctly
```python
def cumulative_prob_to_hazard(cum_prob_series, dt_hours):
    """Convert cumulative failure probability to hazard rate."""
    F = cum_prob_series.values
    n = len(F)
    
    # Ensure monotonic increasing
    F = np.maximum.accumulate(F)
    
    # Hazard rate calculation
    lambda_rate = np.zeros(n)
    lambda_rate[0] = -np.log(1 - F[0]) / dt_hours[0] if F[0] < 1 else 10.0
    
    for i in range(1, n):
        # Incremental probability
        dF = F[i] - F[i-1]
        # Survival at t-1
        S_prev = 1 - F[i-1]
        
        if S_prev > 1e-9:
            lambda_rate[i] = dF / (S_prev * dt_hours[i])
        else:
            lambda_rate[i] = 10.0  # Near-certain failure
    
    return lambda_rate
```
**Impact**: All downstream RUL calculations are incorrect  
**Effort**: Medium (function rewrite + validation)  
**Testing**: Verify hazard rates are physically plausible and monotonic

---

### ✅ Task 1.4: Implement Monte Carlo RUL Uncertainty Propagation
**File**: `core/forecasting.py` (Lines 1010-1042)  
**Issue**: RUL calculation uses arbitrary 50% threshold without uncertainty propagation  
**Current Code**:
```python
first_cross = next((h for h, p in zip(hours, cdf) if p >= 0.5), float(forecast_hours + 24))
```
**Required Fix**: Monte Carlo simulation for RUL with full uncertainty
```python
def estimate_rul_monte_carlo(
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    failure_threshold: float,
    n_simulations: int = 1000
) -> Dict[str, float]:
    """Monte Carlo RUL with full uncertainty quantification."""
    
    n_steps = len(forecast_mean)
    rul_samples = []
    
    for _ in range(n_simulations):
        # Sample forecast trajectory
        trajectory = np.random.normal(forecast_mean, forecast_std)
        
        # Find first crossing
        crossings = np.where(trajectory < failure_threshold)[0]
        
        if len(crossings) > 0:
            rul = crossings[0]
        else:
            rul = n_steps + 10  # Beyond horizon
        
        rul_samples.append(rul)
    
    rul_samples = np.array(rul_samples)
    
    return {
        "rul_median": float(np.median(rul_samples)),
        "rul_mean": float(np.mean(rul_samples)),
        "rul_p10": float(np.percentile(rul_samples, 10)),
        "rul_p90": float(np.percentile(rul_samples, 90)),
        "rul_std": float(np.std(rul_samples)),
        "failure_probability": float(np.mean(rul_samples <= n_steps))
    }
```
**Impact**: Underestimated risk, no quantification of RUL uncertainty  
**Effort**: Medium (new function + integration)  
**Testing**: Compare RUL distributions to deterministic results, validate coverage

---

## 🟠 P1 - MAJOR ISSUES (Fix Within Sprint)

### ✅ Task 2.1: Implement Comprehensive Forecast Quality Metrics
**File**: `core/forecasting.py` (Lines 235-317)  
**Issue**: Missing critical validation metrics (bias, coverage, sharpness, calibration)  
**Required Additions**:
```python
def compute_forecast_quality_comprehensive(
    prev_state: Optional[ForecastState],
    sql_client: SqlClient,
    equip_id: int,
    current_batch_time: datetime
) -> Dict[str, float]:
    """Enhanced quality metrics for forecast validation."""
    
    # ... existing RMSE, MAE, MAPE ...
    
    # 1. Bias
    bias = float(np.mean(errors))
    
    # 2. Coverage (95% CI should contain 95% of actuals)
    if "CI_Lower" in merged.columns and "CI_Upper" in merged.columns:
        in_ci = (
            (merged["HealthIndex"] >= merged["CI_Lower"]) &
            (merged["HealthIndex"] <= merged["CI_Upper"])
        )
        coverage_95 = float(in_ci.mean())
    else:
        coverage_95 = 0.0
    
    # 3. Interval width (sharpness)
    if "CI_Lower" in merged.columns and "CI_Upper" in merged.columns:
        interval_width = float((merged["CI_Upper"] - merged["CI_Lower"]).mean())
    else:
        interval_width = 0.0
    
    # 4. Directional accuracy
    if len(merged) >= 2:
        actual_trend = merged["HealthIndex"].diff().dropna()
        forecast_trend = merged["ForecastHealth"].diff().dropna()
        directional_accuracy = float(
            (np.sign(actual_trend) == np.sign(forecast_trend)).mean()
        )
    else:
        directional_accuracy = 0.0
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "bias": bias,
        "coverage_95": coverage_95,
        "interval_width": interval_width,
        "directional_accuracy": directional_accuracy,
        "n_samples": len(merged)
    }
```
**Impact**: Cannot properly assess model performance  
**Effort**: Low-Medium (extend existing function)  
**Testing**: Validate metrics on historical data with known characteristics

---

### ✅ Task 2.2: Fix Temporal Blending with Recency Weighting
**File**: `core/forecasting.py` (Lines 319-384)  
**Issue**: Exponential decay gives too much weight to old forecasts far in future  
**Required Fix**:
```python
def merge_forecast_horizons_improved(
    prev_horizon: pd.DataFrame,
    new_horizon: pd.DataFrame,
    current_time: datetime,
    blend_tau_hours: float = BLEND_TAU_HOURS
) -> pd.DataFrame:
    """Improved blending with recency priority."""
    
    # Calculate blend weights based on RECENCY not just time-to-horizon
    prev_age_hours = (current_time - prev_forecast_time).total_seconds() / 3600
    
    # Recency weight: exponential decay based on forecast age
    recency_weight = np.exp(-prev_age_hours / blend_tau_hours)
    
    # Horizon weight: closer points more certain
    horizon_hours = (merged["Timestamp"] - current_time).dt.total_seconds() / 3600
    horizon_weight = 1.0 / (1.0 + horizon_hours / 24.0)
    
    # Combined weight: new forecast gets priority
    w_prev = recency_weight * horizon_weight
    w_new = 1.0 - w_prev
    
    # ... rest of blending ...
```
**Impact**: Suboptimal forecast fusion, overweighting stale predictions  
**Effort**: Medium (rewrite blending logic)  
**Testing**: Validate on real data with multiple forecast updates

---

### ✅ Task 2.3: Replace Failure Probability Gaussian Assumption with Empirical Distribution
**File**: `core/forecasting.py` (Lines 767-782)  
**Issue**: Assumes Gaussian errors, but health degradation is often asymmetric/heavy-tailed  
**Required Fix**:
```python
def estimate_failure_probability_empirical(
    forecast_mean: float,
    forecast_std: float,
    failure_threshold: float,
    residual_history: np.ndarray
) -> float:
    """Non-parametric failure probability using empirical residual distribution."""
    
    n_samples = 10000
    
    # Bootstrap from empirical residuals
    sampled_residuals = np.random.choice(
        residual_history,
        size=n_samples,
        replace=True
    )
    
    # Generate forecast distribution
    forecast_samples = forecast_mean + sampled_residuals * (forecast_std / np.std(residual_history))
    
    # Estimate P(health < threshold)
    failure_prob = np.mean(forecast_samples < failure_threshold)
    
    return float(failure_prob)
```
**Impact**: Overconfident failure predictions  
**Effort**: Medium (new function + state tracking for residuals)  
**Testing**: Compare to Gaussian method, validate calibration

---

### ✅ Task 2.4: Improve Data Hash Stability
**File**: `core/forecasting.py` (Lines 140-165)  
**Issue**: Hash instability due to float precision, column ordering  
**Required Fix**:
```python
def compute_data_hash(df: pd.DataFrame) -> str:
    """Stable hash using sorted index + key columns only."""
    try:
        # Only hash columns that affect model training
        key_cols = ["Timestamp", "HealthIndex"]
        if not all(c in df.columns for c in key_cols):
            return ""
        
        # Sort by timestamp for determinism
        df_sorted = df[key_cols].sort_values("Timestamp").reset_index(drop=True)
        
        # Round floats to avoid precision drift
        df_sorted["HealthIndex"] = df_sorted["HealthIndex"].round(6)
        
        # Hash using stable representation
        return hashlib.sha256(
            df_sorted.to_json(orient="records", date_format="iso").encode()
        ).hexdigest()[:16]
    except Exception as e:
        Console.warn(f"[FORECAST] Hash computation failed: {e}")
        return ""
```
**Impact**: Unnecessary retraining due to hash changes  
**Effort**: Low (function rewrite)  
**Testing**: Verify identical data produces same hash

---

## 🟡 P2 - IMPORTANT IMPROVEMENTS (Next Sprint)

### ✅ Task 3.1: Implement Adaptive Hyperparameter Optimization
**File**: `core/forecasting.py` (New capability)  
**Issue**: Fixed α and β parameters, optimal values change with data characteristics  
**Required Implementation**:
```python
def adaptive_exponential_smoothing(
    series: pd.Series,
    initial_alpha: float = 0.3,
    initial_beta: float = 0.2
) -> Tuple[float, float, np.ndarray]:
    """Adaptive Holt's method with optimal α, β selection."""
    
    from scipy.optimize import minimize
    
    def forecast_error(params, series):
        alpha, beta = params
        # Fit model with these params
        # Return MSE
        return mse
    
    result = minimize(
        forecast_error,
        x0=[initial_alpha, initial_beta],
        args=(series,),
        bounds=[(0.01, 0.99), (0.01, 0.99)],
        method='L-BFGS-B'
    )
    
    optimal_alpha, optimal_beta = result.x
    
    # Refit with optimal parameters
    return optimal_alpha, optimal_beta, forecasts
```
**Impact**: Better forecast accuracy through optimal parameters  
**Effort**: High (optimization framework + validation)  
**Testing**: Compare adaptive vs. fixed parameters on diverse datasets

---

### ✅ Task 3.2: Add Regime-Specific Forecasting Models
**File**: `core/forecasting.py` (New capability)  
**Issue**: Single model for all operating regimes, ignoring context  
**Required Implementation**:
```python
def forecast_by_regime(
    health_series: pd.Series,
    regime_series: pd.Series,
    config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Fit regime-specific forecast models."""
    
    regimes = regime_series.unique()
    forecasts = {}
    
    for regime in regimes:
        # Filter data for this regime
        regime_mask = regime_series == regime
        regime_health = health_series[regime_mask]
        
        if len(regime_health) < MIN_FORECAST_SAMPLES:
            Console.warn(f"[FORECAST] Insufficient data for regime '{regime}', using global model")
            continue
        
        # Fit regime-specific model
        model = HoltForecast(alpha=config["alpha"], beta=config["beta"])
        model.fit(regime_health)
        
        # Forecast
        forecast = model.predict(steps=config["forecast_hours"])
        forecasts[regime] = forecast
    
    return forecasts
```
**Impact**: More accurate predictions by accounting for operating context  
**Effort**: High (regime tracking + model management)  
**Testing**: Validate improved accuracy within regimes

---

### ✅ Task 3.3: Implement Bootstrap Confidence Intervals
**File**: `core/forecasting.py` (Lines 733-745)  
**Issue**: CIs don't account for parameter estimation uncertainty  
**Required Implementation**:
```python
def forecast_with_bootstrap_ci(
    series: pd.Series,
    horizon: int,
    n_bootstrap: int = 100,
    alpha: float = 0.3,
    beta: float = 0.2
) -> pd.DataFrame:
    """Bootstrap forecast with full uncertainty quantification."""
    
    n = len(series)
    forecasts = []
    
    for _ in range(n_bootstrap):
        # Block bootstrap to preserve autocorrelation
        block_size = min(10, n // 10)
        resampled = block_bootstrap(series, block_size)
        
        # Fit model to resampled data
        model = HoltLinearTrend(alpha, beta)
        model.fit(resampled)
        
        # Forecast
        fc = model.predict(horizon)
        forecasts.append(fc)
    
    # Compute percentiles across bootstrap samples
    forecasts_array = np.array(forecasts)
    
    return pd.DataFrame({
        "forecast": np.mean(forecasts_array, axis=0),
        "ci_lower": np.percentile(forecasts_array, 2.5, axis=0),
        "ci_upper": np.percentile(forecasts_array, 97.5, axis=0),
        "std": np.std(forecasts_array, axis=0)
    })
```
**Impact**: More realistic uncertainty quantification  
**Effort**: High (bootstrap infrastructure)  
**Testing**: Validate CI coverage improvements

---

### ✅ Task 3.4: Enhance Retrain Logic with Diagnostics
**File**: `core/forecasting.py` (Lines 167-233)  
**Issue**: Retrain checks are independent, failures not tracked properly  
**Required Fix**:
```python
def should_retrain(...) -> Tuple[bool, str, Dict[str, Any]]:
    """Returns (should_retrain, reason, diagnostic_info)"""
    
    diagnostics = {
        "checks_performed": [],
        "checks_failed": [],
        "checks_skipped": []
    }
    
    # 1. Data hash check
    diagnostics["checks_performed"].append("data_hash")
    if current_hash != prev_state.data_hash:
        diagnostics["checks_failed"].append("data_hash")
        return True, "data_changed", diagnostics
    
    # 2. Performance degradation check
    diagnostics["checks_performed"].append("performance")
    quality_metrics = compute_forecast_quality(...)
    if quality_metrics["mape"] > 15.0:
        diagnostics["checks_failed"].append("performance")
        return True, "performance_degraded", diagnostics
    
    # 3. Drift check (if enabled)
    if config.get("enable_drift_check", False):
        diagnostics["checks_performed"].append("drift")
        drift_detected = check_forecast_drift(...)
        if drift_detected:
            diagnostics["checks_failed"].append("drift")
            return True, "drift_detected", diagnostics
    else:
        diagnostics["checks_skipped"].append("drift")
    
    return False, "no_retrain_needed", diagnostics
```
**Impact**: Better visibility into retrain decisions  
**Effort**: Medium (diagnostic tracking)  
**Testing**: Validate diagnostics logged correctly

---

## 🟢 P3 - ENHANCEMENTS (Future Work)

### ✅ Task 4.1: Improve Detector Forecasting with AR(1) Model
**File**: `core/forecasting.py` (Lines 828-867)  
**Issue**: Arbitrary exponential decay without physical justification  
**Required Implementation**:
```python
def forecast_detector_ar1(
    detector_history: pd.Series,
    horizon: int,
    detector_type: str
) -> pd.DataFrame:
    """AR(1) forecast for detector scores with type-specific params."""
    
    recent = detector_history.tail(168)  # Last week
    
    if len(recent) < 10:
        return exponential_decay_forecast(recent.iloc[-1], horizon)
    
    # AR(1): x_t = phi * x_{t-1} + mu + epsilon
    x = recent.values
    phi = np.corrcoef(x[:-1], x[1:])[0, 1]
    phi = np.clip(phi, 0.0, 0.99)
    
    mu = np.mean(x * (1 - phi))
    sigma = np.std(x[1:] - phi * x[:-1])
    
    # Forecast with variance growth
    forecasts = []
    x_t = recent.iloc[-1]
    
    for h in range(1, horizon + 1):
        x_t = phi * x_t + mu
        var_h = sigma**2 * (1 - phi**(2*h)) / (1 - phi**2)
        
        forecasts.append({
            "horizon": h,
            "forecast": float(x_t),
            "std": float(np.sqrt(var_h)),
            "ci_lower": float(x_t - 1.96 * np.sqrt(var_h)),
            "ci_upper": float(x_t + 1.96 * np.sqrt(var_h))
        })
    
    return pd.DataFrame(forecasts)
```
**Impact**: More realistic detector forecasts  
**Effort**: Medium (new forecasting method)  
**Testing**: Validate against actual detector trajectories

---

### ✅ Task 4.2: Add Vector Autoregression (VAR) for Sensor Forecasting
**File**: `core/forecasting.py` (Lines 881-1008)  
**Issue**: Sensors forecast independently, ignoring cross-correlations  
**Required Implementation**:
```python
from statsmodels.tsa.api import VAR

def forecast_sensors_var(
    sensor_df: pd.DataFrame,
    horizon: int,
    max_sensors: int = 10
) -> pd.DataFrame:
    """Multivariate sensor forecast with cross-correlations."""
    
    # Select top changing sensors
    sensor_variability = sensor_df.std() / (sensor_df.mean().abs() + 1e-6)
    top_sensors = sensor_variability.nlargest(max_sensors).index.tolist()
    
    # Prepare data
    data = sensor_df[top_sensors].dropna()
    
    if len(data) < 50:
        return forecast_sensors_univariate(data, horizon)
    
    # Fit VAR model
    model = VAR(data)
    results = model.fit(maxlags=5, ic='aic')
    
    # Forecast
    forecast = results.forecast(data.values[-results.k_ar:], steps=horizon)
    
    # Build forecast DataFrame with confidence intervals
    forecast_df = pd.DataFrame(
        forecast,
        columns=top_sensors,
        index=pd.date_range(
            start=data.index[-1] + pd.Timedelta(hours=1),
            periods=horizon,
            freq='h'
        )
    )
    
    # Add confidence intervals
    for col in top_sensors:
        std = results.resid[col].std()
        forecast_df[f"{col}_ci_lower"] = forecast_df[col] - 1.96 * std
        forecast_df[f"{col}_ci_upper"] = forecast_df[col] + 1.96 * std
    
    return forecast_df
```
**Impact**: Captures sensor dependencies for better predictions  
**Effort**: High (multivariate modeling)  
**Testing**: Validate cross-correlations preserved

---

### ✅ Task 4.3: Add Outlier Detection Before Forecasting
**File**: `core/forecasting.py` (New function)  
**Issue**: No robust outlier handling, bad data corrupts forecasts  
**Required Implementation**:
```python
def detect_and_remove_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 3.0
) -> pd.Series:
    """Robust outlier removal before forecasting."""
    
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        
    elif method == "zscore":
        mean = series.mean()
        std = series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std
    
    # Replace outliers with interpolated values
    mask = (series < lower) | (series > upper)
    series_clean = series.copy()
    series_clean[mask] = np.nan
    series_clean = series_clean.interpolate(method='linear', limit_direction='both')
    
    n_outliers = mask.sum()
    if n_outliers > 0:
        Console.warn(f"[FORECAST] Removed {n_outliers} outliers ({n_outliers/len(series)*100:.1f}%)")
    
    return series_clean
```
**Impact**: More robust forecasts against data quality issues  
**Effort**: Low (utility function)  
**Testing**: Validate on synthetic data with outliers

---

### ✅ Task 4.4: Add Comprehensive Model Diagnostics
**File**: `core/forecasting.py` (New function)  
**Issue**: No residual analysis to validate model assumptions  
**Required Implementation**:
```python
def validate_forecast_model(
    actual: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray
) -> Dict[str, Any]:
    """Comprehensive model diagnostics."""
    
    from scipy.stats import shapiro
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    diagnostics = {}
    
    # 1. Normality test
    _, p_shapiro = shapiro(residuals[:min(5000, len(residuals))])
    diagnostics["residuals_normal_p"] = float(p_shapiro)
    
    # 2. Autocorrelation
    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    diagnostics["residuals_autocorr_p"] = float(lb_result['lb_pvalue'].iloc[0])
    
    # 3. Heteroscedasticity
    n = len(residuals)
    var_first = np.var(residuals[:n//3])
    var_last = np.var(residuals[-n//3:])
    diagnostics["variance_ratio"] = float(var_last / var_first) if var_first > 0 else 1.0
    
    # 4. MAPE
    diagnostics["mape"] = float(np.mean(np.abs(residuals / actual)) * 100)
    
    # 5. Theil's U statistic
    naive_forecast = np.roll(actual, 1)
    naive_forecast[0] = actual[0]
    naive_mse = np.mean((actual - naive_forecast)**2)
    model_mse = np.mean(residuals**2)
    diagnostics["theil_u"] = float(np.sqrt(model_mse / naive_mse))
    
    return diagnostics
```
**Impact**: Better model validation and debugging  
**Effort**: Medium (diagnostic suite)  
**Testing**: Run on historical forecasts

---

### ✅ Task 4.5: Implement State Versioning and Migration
**File**: `core/forecasting.py` (Lines 1168-1213)  
**Issue**: No versioning strategy for forecast states, incompatibility on model changes  
**Required Implementation**:
```python
class ForecastStateV3(ForecastState):
    """Version 3 with backward compatibility."""
    
    VERSION = 3
    
    @classmethod
    def from_v2(cls, v2_state: ForecastState) -> "ForecastStateV3":
        """Migrate v2 state to v3 format."""
        return cls(
            model_type=f"{v2_state.model_type}_v3",
            alpha=v2_state.alpha,
            beta=v2_state.beta,
            # ... migrate all fields ...
            version=3
        )
    
    @classmethod
    def load_with_migration(cls, equip: str, equip_id: int, sql_client: SqlClient):
        """Load state and auto-migrate if needed."""
        state = load_forecast_state(equip, equip_id, sql_client)
        
        if state is None:
            return None
        
        version = getattr(state, 'VERSION', 1)
        
        if version < cls.VERSION:
            Console.info(f"[FORECAST] Migrating state from v{version} to v{cls.VERSION}")
            state = cls.from_v2(state)
        
        return state
```
**Impact**: Smooth model upgrades without breaking existing states  
**Effort**: Medium (migration framework)  
**Testing**: Test migration paths v1→v2→v3

---

## 📋 TECHNICAL DEBT & CLEANUP

### ✅ Task 5.1: Document All Magic Numbers
**File**: `core/forecasting.py` (Throughout)  
**Issue**: Unexplained constants (MIN_AR1_SAMPLES=3, BLEND_TAU_HOURS=12.0, etc.)  
**Required Action**: Add detailed comments explaining rationale for all magic numbers
```python
# Minimum samples required for AR(1) model coefficient estimation
# Rationale: Need at least 3 points for variance and autocorrelation estimation
# - 1 point: No variance
# - 2 points: One covariance pair, unstable
# - 3 points: Minimum for stable AR(1) coefficient (2 covariance pairs)
MIN_AR1_SAMPLES = 3
```
**Effort**: Low (documentation)

---

### ✅ Task 5.2: Reduce Logging Verbosity with Log Levels
**File**: `core/forecasting.py` (Throughout)  
**Issue**: Too many Console.info() calls, overwhelming in production  
**Required Fix**: Use appropriate log levels
```python
# Replace:
Console.info(f"[FORECAST] Generated {len(health_forecast_df)} hour health forecast")

# With:
logger.debug(f"[FORECAST] Generated {len(health_forecast_df)} hour health forecast")

# Keep INFO only for important events:
logger.info(f"[FORECAST] RUL: {rul_hours:.1f}h, Failure Prob: {max_failure_prob*100:.1f}%")
```
**Effort**: Low (find/replace with judgment)

---

### ✅ Task 5.3: Complete Type Hints Throughout
**File**: `core/forecasting.py` (Throughout)  
**Issue**: Incomplete or missing return type hints  
**Required Action**: Add complete type annotations with docstrings
```python
def smooth_failure_probability_hazard(
    prev_hazard_baseline: float,
    new_probability_series: pd.Series,
    dt_hours: Optional[float] = None,
    alpha: float = DEFAULT_HAZARD_SMOOTHING_ALPHA
) -> pd.DataFrame:  # Columns: [Timestamp, HazardRaw, HazardSmooth, Survival, FailureProb]
    """
    Returns:
        DataFrame with columns:
        - Timestamp: datetime
        - HazardRaw: float
        - HazardSmooth: float (EWMA smoothed)
        - Survival: float [0, 1]
        - FailureProb: float [0, 1]
    """
```
**Effort**: Low-Medium (systematic review)

---

### ✅ Task 5.4: Remove Unnecessary DataFrame Copies
**File**: `core/forecasting.py` (Throughout)  
**Issue**: Multiple .copy() calls that may not be needed  
**Required Action**: Profile and remove unnecessary copies
```python
# Line 663 - BEFORE:
health_values = pd.Series(health_series, copy=True).astype(float)

# AFTER (astype creates copy anyway):
health_values = health_series.astype(float)
health_values = health_values.fillna(median_val)
```
**Effort**: Low (code review + testing)

---

### ✅ Task 5.5: Enhance Error Messages with Context
**File**: `core/forecasting.py` (Throughout)  
**Issue**: Generic error messages without diagnostic context  
**Required Fix**:
```python
# BEFORE:
Console.warn(f"[ENHANCED_FORECAST] Health forecasting failed: {e}")

# AFTER:
Console.warn(
    f"[ENHANCED_FORECAST] Health forecasting failed for EquipID={equip_id}, "
    f"RunID={run_id}, DataPoints={len(health_series)}: {e}"
)
```
**Effort**: Low (systematic improvement)

---

### ✅ Task 5.6: Optimize SQL Queries with WHERE Clauses
**File**: `core/forecasting.py` (Lines 264-282, others)  
**Issue**: Loading entire tables and filtering in Python  
**Required Fix**:
```python
# BEFORE:
cur.execute("""
    SELECT Timestamp, HealthIndex
    FROM dbo.ACM_HealthTimeline
    WHERE EquipID = ?
    ORDER BY Timestamp
""", (equip_id,))
# Then filter in Python

# AFTER:
cur.execute("""
    SELECT Timestamp, HealthIndex
    FROM dbo.ACM_HealthTimeline
    WHERE EquipID = ? 
      AND Timestamp >= DATEADD(hour, -?, GETDATE())
    ORDER BY Timestamp
""", (equip_id, lookback_hours))
```
**Effort**: Medium (query optimization)

---

## 🎯 ADVANCED FEATURES (Long-term Vision)

### ✅ Task 6.1: Build Adaptive Forecast Engine Framework
**File**: New module `core/adaptive_forecasting.py`  
**Purpose**: Self-tuning forecast engine with continuous learning  
**Features**:
- Automatic model selection based on data characteristics
- Bayesian hyperparameter optimization
- Time series cross-validation
- Performance tracking and model switching
- Ensemble methods

**Effort**: Very High (3-4 sprints)  
**Dependencies**: Tasks 3.1, 3.2, 3.3 completed first

---

### ✅ Task 6.2: Implement Multi-Model Ensemble Forecasting
**Purpose**: Combine multiple forecast methods (Holt, ARIMA, Prophet, XGBoost)  
**Benefits**: Robust to model misspecification, better uncertainty quantification  
**Effort**: Very High (2-3 sprints)

---

### ✅ Task 6.3: Add Scenario-Based Forecasting
**Purpose**: "What-if" forecasting for different operating scenarios  
**Example**: "Forecast health if we increase load by 20%"  
**Effort**: Very High (requires physics-based modeling)

---

## 📊 TESTING REQUIREMENTS

### Test Coverage Goals
- **Unit tests**: All P0 and P1 fixes must have dedicated unit tests
- **Integration tests**: Validate end-to-end forecast pipeline
- **Regression tests**: Ensure fixes don't break existing functionality
- **Performance tests**: Validate optimization improvements

### Key Test Scenarios
1. **Synthetic data with known properties**: Test forecast accuracy on controlled datasets
2. **Historical data validation**: Backtest on real equipment data
3. **Edge cases**: Empty data, single point, constant values, missing data
4. **Stress testing**: Large datasets (10K+ points), many equipment in parallel

---

## 📈 SUCCESS METRICS

### P0 Tasks (Critical)
- [ ] Forecast MAPE improves by >20%
- [ ] CI coverage matches nominal level (95% CI contains 94-96% of actuals)
- [ ] RUL estimates have quantified uncertainty (P10, P50, P90 percentiles)

### P1 Tasks (Major)
- [ ] Forecast quality metrics dashboard implemented
- [ ] Temporal blending reduces forecast jumps by >30%
- [ ] Failure probability calibration score >0.8

### P2 Tasks (Important)
- [ ] Adaptive hyperparameters show >10% accuracy improvement
- [ ] Regime-specific models outperform global model by >15%
- [ ] Bootstrap CIs are 10-20% wider but better calibrated

---

## 📅 RECOMMENDED TIMELINE

### Sprint 1 (Week 1-2): P0 Critical Fixes
- Task 1.1: Fix trend calculation
- Task 1.2: Correct variance formula
- Task 1.3: Fix hazard rate calculation
- Task 1.4: Implement Monte Carlo RUL

### Sprint 2 (Week 3-4): P1 Major Improvements
- Task 2.1: Comprehensive quality metrics
- Task 2.2: Temporal blending fix
- Task 2.3: Empirical failure probability
- Task 2.4: Data hash stability

### Sprint 3 (Week 5-6): P2 Important Enhancements
- Task 3.1: Adaptive hyperparameters
- Task 3.2: Regime-specific models
- Task 3.3: Bootstrap CIs
- Task 3.4: Enhanced retrain logic

### Sprint 4+ (Week 7+): P3 & Advanced Features
- Detector AR(1) forecasting
- VAR sensor forecasting
- Outlier detection
- Model diagnostics
- Long-term: Adaptive forecast engine

---

## 🔍 REVIEW & SIGN-OFF

**Created By**: GitHub Copilot  
**Review Required**: Lead Data Scientist, DevOps Lead  
**Approval Required**: Technical Director  
**Target Start Date**: TBD  
**Estimated Completion**: 6-8 weeks for P0-P2 tasks

---

**END OF TASK LIST**

# Forecasting.py – Comprehensive Audit & Task Backlog

**Generated**: December 3, 2025
**Scope**: `core/forecasting.py`
**Priority Scale**: P0 (Critical) → P1 (Major) → P2 (Important) → P3 (Enhancement / Tech Debt)

---

## 0. High-Level Architecture & Responsibilities

These are cross-cutting design/composition tasks that sit **above** individual bug fixes.

---

###   Task A.1: Separate Orchestration from Forecast Logic

**File**: `core/forecasting.py`
**Issue**: CLI / orchestration logic and analytical code are interwoven (globals, side effects).

**Required Actions**:

1. **Introduce a `ForecastEngine` (or similar) class/function**:

   * Public entrypoint:

     ```python
     def run_forecast(
         cfg: Dict[str, Any],
         scores_df: pd.DataFrame,
         episodes_df: Optional[pd.DataFrame] = None,
         prev_state: Optional[ForecastState] = None,
     ) -> ForecastResult:
         ...
     ```
   * `ForecastResult` is a typed dataclass (e.g. `forecasts`, `rul`, `diagnostics`, `state_out`).

2. **Confine CLI / `if __name__ == "__main__"`**:

   * Only parse args, read config, call `run_forecast(...)`, and handle IO (files / SQL).
   * No analytics logic in CLI code.

3. **Remove module-level global objects**:

   * No global config, no global `sql_client`, no global cached DataFrames.
   * Pass dependencies via parameters or encapsulate inside `ForecastEngine`.

**Impact**: Easier testing, no cross-contamination between equipments, clearer separation of concerns.
**Effort**: Medium (refactor but mostly moving existing logic into class/methods).
**Testing**: Unit tests call `run_forecast()` directly with synthetic DataFrames.

---

###   Task A.2: Standardise Inputs & Outputs (I/O Contract)

**Issue**: Forecasting currently operates on mixed views (scores, detectors, sensors) in ad-hoc ways.

**Required Actions**:

1. **Define canonical inputs**:

   * **Primary target**: fused health / health index (see Task C.1).
   * **Context**: `regime_label`, episodes (optional), detector scores, sensor values.
   * Document expected input schemas in a docstring or separate markdown.

2. **Define standard outputs**:

   * `forecast_ts`: time series forecast (per horizon) with:

     * `Timestamp`, `HorizonHours`, `ForecastTarget`, `Std`, `CI_Lower`, `CI_Upper`, `ModelName`, `EquipID`, `RunID`.
   * `forecast_summary`: aggregated summary for dashboard:

     * Next threshold breach time, RUL percentiles, max failure probability over horizon, etc.
   * Optional sensor & detector forecast tables.

3. **Align with OutputManager**:

   * Use same time formatting and health index formulas as in `core/output_manager.py`.

**Impact**: Forecast outputs become drop-in with dashboards & SQL writers.
**Effort**: Medium.
**Testing**: Schema-validation tests for outputs (columns + dtypes).

---

###   Task A.3: Enforce Local-Time Policy and Index Contract

**Issue**: Mixed timezone handling; sometimes UTC; sometimes naive; indexing assumptions implicit.

**Required Actions**:

1. **No UTC timezone conversions**:

   * Remove any `tz_localize("UTC")`, `tz_convert(...)`, or `utc=True` usages in forecasting.
   * Work with **naive local timestamps** only, matching the rest of ACM.

2. **Standard index**:

   * Require `DatetimeIndex` on `scores_df`:

     ```python
     def _ensure_local_index(df: pd.DataFrame) -> pd.DataFrame:
         if not isinstance(df.index, pd.DatetimeIndex):
             if "Timestamp" in df.columns:
                 df = df.set_index("Timestamp")
             else:
                 raise ValueError("Forecast requires DatetimeIndex or Timestamp column.")
         if not df.index.is_monotonic_increasing:
             df = df.sort_index()
         return df
     ```

3. **dt_hours calculation**:

   * Compute sampling interval from index as median Δt (in hours).
   * Validate: log warning if Δt varies significantly; optionally resample.

4. **Horizon units**:

   * Use **hours** as the canonical horizon unit (config: `forecast.horizons_hours = [1, 4, 8, 24, 48]`).
   * Convert hours→steps using `dt_hours`.

**Impact**: Eliminates timezone bugs, provides consistent mapping from forecast horizon to index.
**Effort**: Medium.
**Testing**: Synthetic series with 1h, 10 min sampling; horizon mapping tests.

---

## 🔴 P0 – CRITICAL ISSUES (Must Fix Immediately)

---

###   Task 1.1: Fix Initial Trend Calculation in Holt's Method

**File**: `core/forecasting.py` (Lines 682–708)
**Issue**: Initial trend does not account for `dt_hours`, mis-scaling forecasts.

**Current Code**:

```python
trend = float(health_values.iloc[1] - health_values.iloc[0]) if n > 1 else 0.0
```

**Required Fix**:

```python
trend = float(health_values.iloc[1] - health_values.iloc[0]) / dt_hours if n > 1 else 0.0
```

**Additional Steps**:

1. Ensure `dt_hours` is computed as:

   ```python
   dt = np.median(np.diff(health_series.index.values).astype("timedelta64[s]").astype(float))
   dt_hours = dt / 3600.0
   ```
2. If `dt_hours <= 0` or NaN, raise a clear error or default to 1.0 with a warning.

**Impact**: Prevents trend from being off by factor of sampling interval; stabilises forecasts.
**Effort**: Low.
**Testing**:

* Use linear synthetic series (`y = 100 - 0.5*t`), confirm estimated trend ≈ -0.5 units/hour.

---

###   Task 1.2: Correct Variance Growth Formula for Confidence Intervals

**File**: `core/forecasting.py` (Lines 733–735)
**Issue**: Ad-hoc variance multiplier; CIs mathematically incorrect.

**Current Code**:

```python
var_mult = np.sqrt(1.0 + (alpha ** 2) * h + (beta ** 2) * (h ** 2))
horizon_std = std_error * var_mult
```

**Required Fix**: Factor out a proper helper:

```python
def holt_variance_multiplier(h: int, alpha: float, beta: float) -> float:
    """Correct variance multiplier for Holt's Linear Trend."""
    if h <= 1:
        return 1.0
    # Example formula from audit (adapt / validate against reference):
    return 1.0 + (h - 1) * (alpha**2 + alpha*beta*h + beta**2 * h * (h + 1) / 2)
```

Then:

```python
var_mult = holt_variance_multiplier(h, alpha, beta)
horizon_std = std_error * np.sqrt(var_mult)
```

**Impact**: CIs reflect true forecast uncertainty; avoids under/over-confidence.
**Effort**: Low.
**Testing**:

* Monte Carlo check: simulate Holt processes and compare empirical forecast error variance vs formula.

---

###   Task 1.3: Fix Hazard Rate Calculation Logic

**File**: `core/forecasting.py` (Lines 298–306)
**Issue**: Misinterprets cumulative failure probabilities as interval probabilities; wrong hazard.

**Current Code**:

```python
lambda_raw_np = -np.log(1 - p_np) / dt_vec
```

**Required Fix** (replace with cumulative→hazard conversion):

```python
def cumulative_prob_to_hazard(
    cum_prob_series: pd.Series,
    dt_hours_vec: np.ndarray
) -> np.ndarray:
    """Convert cumulative failure probability F(t) to discrete hazard λ(t)."""
    F = cum_prob_series.to_numpy(copy=True)
    n = len(F)

    # Enforce monotonic cumulative probabilities
    F = np.maximum.accumulate(F)
    F = np.clip(F, 0.0, 1.0)

    lambda_rate = np.zeros(n, dtype=float)

    if n == 0:
        return lambda_rate

    # First point: treat as if hazard over first dt
    if F[0] < 1:
        lambda_rate[0] = -np.log(max(1e-9, 1 - F[0])) / max(1e-6, dt_hours_vec[0])
    else:
        lambda_rate[0] = 10.0  # Sentinel for near-certain failure

    for i in range(1, n):
        dF = F[i] - F[i - 1]
        S_prev = 1.0 - F[i - 1]

        if S_prev > 1e-9 and dt_hours_vec[i] > 0:
            lambda_rate[i] = dF / (S_prev * dt_hours_vec[i])
        else:
            lambda_rate[i] = 10.0

    return lambda_rate
```

Then use `lambda_rate` as `lambda_raw_np` for smoothing and survival computations.

**Impact**: RUL and hazard-based metrics become mathematically consistent.
**Effort**: Medium.
**Testing**:

* Use synthetic exponential CDF; recovered hazard should be approximately constant.

---

###   Task 1.4: Implement Monte Carlo RUL Uncertainty Propagation

**File**: `core/forecasting.py` (Lines 1010–1042)
**Issue**: RUL uses a single CDF crossing at 50%; no uncertainty quantification.

**Current Code**:

```python
first_cross = next((h for h, p in zip(hours, cdf) if p >= 0.5), float(forecast_hours + 24))
```

**Required Fix**: Add a Monte Carlo RUL estimator:

```python
def estimate_rul_monte_carlo(
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    failure_threshold: float,
    n_simulations: int = 1000
) -> Dict[str, float]:
    """Monte Carlo RUL with full uncertainty quantification."""
    n_steps = len(forecast_mean)
    if n_steps == 0:
        return {
            "rul_median": float("nan"),
            "rul_mean": float("nan"),
            "rul_p10": float("nan"),
            "rul_p90": float("nan"),
            "rul_std": float("nan"),
            "failure_probability": 0.0,
        }

    rul_samples: list[float] = []

    for _ in range(n_simulations):
        trajectory = np.random.normal(forecast_mean, forecast_std)
        crossings = np.where(trajectory < failure_threshold)[0]

        if len(crossings) > 0:
            rul = float(crossings[0])
        else:
            rul = float(n_steps + 10)

        rul_samples.append(rul)

    rul_arr = np.asarray(rul_samples, dtype=float)

    return {
        "rul_median": float(np.median(rul_arr)),
        "rul_mean": float(np.mean(rul_arr)),
        "rul_p10": float(np.percentile(rul_arr, 10)),
        "rul_p90": float(np.percentile(rul_arr, 90)),
        "rul_std": float(np.std(rul_arr)),
        "failure_probability": float(np.mean(rul_arr <= n_steps)),
    }
```

**Integration**:

* Replace single `first_cross` with usage of this result.
* Expose all RUL stats in summary outputs.

**Impact**: RUL becomes a distribution, not a point guess.
**Effort**: Medium.
**Testing**:

* Synthetic monotonic degradation; RUL distribution shrinks as horizon is extended.

---

## 🟠 P1 – MAJOR ISSUES (Fix Within Sprint)

---

###   Task 2.1: Implement Comprehensive Forecast Quality Metrics

**File**: `core/forecasting.py` (Lines 235–317)
**Issue**: Only basic error metrics; no bias / coverage / sharpness / directional accuracy.

**Required Additions**:

```python
def compute_forecast_quality_comprehensive(
    prev_state: Optional[ForecastState],
    sql_client: SqlClient,
    equip_id: int,
    current_batch_time: datetime
) -> Dict[str, float]:
    """Enhanced quality metrics for forecast validation."""
    # Load actuals and forecasts (existing logic)
    # merged columns: ["Timestamp", "HealthIndex", "ForecastHealth", "CI_Lower", "CI_Upper", ...]

    # Existing metrics: rmse, mae, mape
    # ...
    errors = merged["ForecastHealth"] - merged["HealthIndex"]

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors / (merged["HealthIndex"].replace(0, np.nan)))) * 100)

    # 1. Bias
    bias = float(np.mean(errors))

    # 2. Coverage of 95% CI
    if {"CI_Lower", "CI_Upper"}.issubset(merged.columns):
        in_ci = (
            (merged["HealthIndex"] >= merged["CI_Lower"]) &
            (merged["HealthIndex"] <= merged["CI_Upper"])
        )
        coverage_95 = float(in_ci.mean())
        interval_width = float((merged["CI_Upper"] - merged["CI_Lower"]).mean())
    else:
        coverage_95 = 0.0
        interval_width = 0.0

    # 3. Directional accuracy
    if len(merged) >= 2:
        actual_trend = merged["HealthIndex"].diff().dropna()
        forecast_trend = merged["ForecastHealth"].diff().dropna().reindex(actual_trend.index)
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
        "n_samples": float(len(merged)),
    }
```

**Impact**: Enables robust monitoring of forecast quality; prerequisites for adaptive retrain logic.
**Effort**: Low–Medium.
**Testing**:

* Construct small examples where CI coverage and bias are known.

---

###   Task 2.2: Fix Temporal Blending with Recency Weighting

**File**: `core/forecasting.py` (Lines 319–384)
**Issue**: Stale forecasts still contribute too much to far-future points.

**Required Fix**:

```python
def merge_forecast_horizons_improved(
    prev_horizon: pd.DataFrame,
    new_horizon: pd.DataFrame,
    current_time: datetime,
    prev_forecast_time: datetime,
    blend_tau_hours: float = BLEND_TAU_HOURS,
) -> pd.DataFrame:
    """
    Blend old and new forecast horizons with recency + horizon-aware weights.
    Assumes both dataframes have a 'Timestamp' column and 'ForecastTarget' column.
    """
    if prev_horizon is None or prev_horizon.empty:
        return new_horizon

    merged = prev_horizon.merge(
        new_horizon,
        on="Timestamp",
        how="outer",
        suffixes=("_prev", "_new"),
    ).sort_values("Timestamp")

    horizon_hours = (merged["Timestamp"] - current_time).dt.total_seconds() / 3600.0
    horizon_hours = horizon_hours.clip(lower=0.0)

    prev_age_hours = (current_time - prev_forecast_time).total_seconds() / 3600.0
    recency_weight = np.exp(-prev_age_hours / blend_tau_hours)

    horizon_weight = 1.0 / (1.0 + horizon_hours / 24.0)

    w_prev = recency_weight * horizon_weight
    w_prev = np.clip(w_prev, 0.0, 0.9)  # ensure new forecast dominates
    w_new = 1.0 - w_prev

    merged["ForecastTarget"] = (
        merged["ForecastTarget_prev"].fillna(0.0) * w_prev
        + merged["ForecastTarget_new"].fillna(0.0) * w_new
    )

    # Blend std / CI similarly if available
    # ...

    return merged[["Timestamp", "ForecastTarget"] + [c for c in merged.columns if c.endswith("_CI")]]
```

**Impact**: Newer forecasts dominate; reduces weird jumps when new runs arrive.
**Effort**: Medium.
**Testing**:

* Scenario: forecast at t=0 and again at t=24; verify t=48 predictions are more influenced by recent run.

---

###   Task 2.3: Replace Pure Gaussian Failure Probability with Empirical Option

**File**: `core/forecasting.py` (Lines 767–782)
**Issue**: Gaussian error assumption often violates actual residual distribution.

**Required Fix**:

1. **Add mode switch in config**:

   * `forecast.failure_prob_mode = "gaussian" | "empirical"`.

2. **Implement empirical estimator**:

```python
def estimate_failure_probability_empirical(
    forecast_mean: float,
    forecast_std: float,
    failure_threshold: float,
    residual_history: np.ndarray,
    n_samples: int = 10000,
) -> float:
    """Non-parametric failure probability using empirical residual distribution."""
    residual_history = np.asarray(residual_history, dtype=float)
    residual_history = residual_history[np.isfinite(residual_history)]
    if residual_history.size < 10:
        # Fallback to Gaussian if not enough history
        z = (failure_threshold - forecast_mean) / max(forecast_std, 1e-6)
        return float(norm.cdf(z))

    res_std = residual_history.std()
    if res_std <= 0:
        res_std = 1.0

    sampled_residuals = np.random.choice(residual_history, size=n_samples, replace=True)
    scaled_residuals = sampled_residuals * (forecast_std / res_std)
    forecast_samples = forecast_mean + scaled_residuals

    failure_prob = np.mean(forecast_samples < failure_threshold)
    return float(failure_prob)
```

3. **Store residual history** in `ForecastState` so it’s available across runs.

**Impact**: Better calibrated failure probabilities in heavy-tailed / skewed scenarios.
**Effort**: Medium.
**Testing**:

* Synthetic skewed residuals; compare Gaussian vs empirical and check which is closer to empirical observed failure frequencies.

---

###   Task 2.4: Improve Data Hash Stability

**File**: `core/forecasting.py` (Lines 140–165)
**Issue**: Hash highly sensitive to non-material changes (column order, float noise).

**Required Fix**:

```python
def compute_data_hash(df: pd.DataFrame) -> str:
    """Stable hash using sorted index + key columns only."""
    try:
        key_cols = ["Timestamp", "HealthIndex"]
        if not all(c in df.columns for c in key_cols):
            return ""

        df_sorted = df[key_cols].sort_values("Timestamp").reset_index(drop=True)
        df_sorted["HealthIndex"] = df_sorted["HealthIndex"].astype(float).round(6)

        json_bytes = df_sorted.to_json(orient="records", date_format="iso").encode("utf-8")
        return hashlib.sha256(json_bytes).hexdigest()[:16]
    except Exception as e:
        Console.warn(f"[FORECAST] Hash computation failed: {e}")
        return ""
```

**Impact**: Avoids spurious retraining; only material data changes trigger new hash.
**Effort**: Low.
**Testing**:

* Reordered columns and tiny float differences shouldn’t change hash.

---

## 🟡 P2 – IMPORTANT IMPROVEMENTS (Next Sprint)

---

###   Task 3.1: Implement Adaptive Hyperparameter Optimisation for Holt

**File**: `core/forecasting.py` (New helper)
**Issue**: Fixed α, β are suboptimal across equipment & regimes.

**Required Implementation**:

```python
def adaptive_exponential_smoothing(
    series: pd.Series,
    initial_alpha: float = 0.3,
    initial_beta: float = 0.2,
) -> Tuple[float, float]:
    """Find optimal α, β for Holt's method via time-series CV."""
    from scipy.optimize import minimize
    from sklearn.model_selection import TimeSeriesSplit

    y = series.astype(float).to_numpy()
    n = len(y)
    if n < MIN_FORECAST_SAMPLES:
        return initial_alpha, initial_beta

    tscv = TimeSeriesSplit(n_splits=min(3, n // 10))

    def objective(params: np.ndarray) -> float:
        alpha, beta = params
        alpha = float(np.clip(alpha, 0.01, 0.99))
        beta = float(np.clip(beta, 0.01, 0.99))

        errors = []
        for train_idx, val_idx in tscv.split(y):
            train = y[train_idx]
            val = y[val_idx]
            if len(train) < MIN_FORECAST_SAMPLES or len(val) == 0:
                continue
            model = HoltLinearTrend(alpha=alpha, beta=beta)
            model.fit(train)
            fc = model.predict(len(val))
            mse = np.mean((val - fc)**2)
            errors.append(mse)

        if not errors:
            return 1e6
        return float(np.mean(errors))

    result = minimize(
        objective,
        x0=np.array([initial_alpha, initial_beta]),
        bounds=[(0.01, 0.99), (0.01, 0.99)],
        method="L-BFGS-B",
    )
    alpha_opt, beta_opt = result.x
    return float(alpha_opt), float(beta_opt)
```

**Integration**:

* When sufficient history exists and config allows, call this once to update α, β and store in `ForecastState`.

**Impact**: Improves accuracy across diverse equipment.
**Effort**: High (optimisation + performance considerations).
**Testing**:

* Benchmark fixed vs adaptive on multiple real histories.

---

###   Task 3.2: Add Regime-Specific Forecasting Models

**File**: `core/forecasting.py` (New functionality)
**Issue**: Same model across all operating regimes; regime info unused.

**Required Implementation**:

```python
def forecast_by_regime(
    health_series: pd.Series,
    regime_series: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Fit regime-specific Holt models where sufficient data exists."""
    regimes = regime_series.dropna().unique()
    forecasts: Dict[str, pd.DataFrame] = {}
    horizon = int(config["forecast_hours"])

    for regime in regimes:
        mask = regime_series == regime
        regime_health = health_series[mask]
        if len(regime_health) < MIN_FORECAST_SAMPLES:
            Console.warn(f"[FORECAST] Insufficient data for regime '{regime}', using global model")
            continue

        series_clean = detect_and_remove_outliers(regime_health)
        alpha, beta = adaptive_exponential_smoothing(series_clean)
        model = HoltLinearTrend(alpha=alpha, beta=beta)
        model.fit(series_clean)
        fc = model.predict(horizon)
        timestamps = pd.date_range(
            start=series_clean.index[-1] + (series_clean.index[-1] - series_clean.index[-2]),
            periods=horizon,
            freq=series_clean.index.freq or "H",
        )
        df_fc = pd.DataFrame({"Timestamp": timestamps, "ForecastHealth": fc})
        df_fc["Regime"] = regime
        forecasts[regime] = df_fc

    return forecasts
```

**Config**:

* `forecast.regime_mode = "off" | "diagnostics" | "per_regime_model"`.

**Impact**: More accurate forecasts within stable operational regimes.
**Effort**: High.
**Testing**:

* Synthetic 2-regime system with different degradation rates; per-regime forecasting should outperform global.

---

###   Task 3.3: Implement Bootstrap Confidence Intervals

**File**: `core/forecasting.py` (Around CI generation)
**Issue**: CI only accounts for model error, not parameter uncertainty.

**Required Implementation**:

```python
def forecast_with_bootstrap_ci(
    series: pd.Series,
    horizon: int,
    n_bootstrap: int = 100,
    alpha: float = 0.3,
    beta: float = 0.2,
) -> pd.DataFrame:
    """Bootstrap forecast with full uncertainty quantification."""
    series = series.astype(float)
    n = len(series)
    if n < MIN_FORECAST_SAMPLES:
        raise ValueError("Not enough samples for bootstrap CI.")

    forecasts = []
    for _ in range(n_bootstrap):
        block_size = min(10, max(5, n // 10))
        # block_bootstrap to be implemented or imported
        resampled = block_bootstrap(series, block_size)

        model = HoltLinearTrend(alpha=alpha, beta=beta)
        model.fit(resampled)
        fc = model.predict(horizon)
        forecasts.append(fc)

    forecasts_array = np.asarray(forecasts, dtype=float)

    return pd.DataFrame({
        "forecast": forecasts_array.mean(axis=0),
        "ci_lower": np.percentile(forecasts_array, 2.5, axis=0),
        "ci_upper": np.percentile(forecasts_array, 97.5, axis=0),
        "std": forecasts_array.std(axis=0),
    })
```

**Impact**: CIs reflect model + parameter uncertainty; more realistic risk bounds.
**Effort**: High.
**Testing**:

* Coverage tests on synthetic data.

---

###   Task 3.4: Enhance Retrain Logic with Diagnostics

**File**: `core/forecasting.py` (Lines 167–233)
**Issue**: Retrain decisions opaque; drift check disabled; no diagnostics.

**Required Fix**:

```python
def should_retrain(
    prev_state: Optional[ForecastState],
    current_hash: str,
    quality_metrics: Dict[str, float],
    config: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (should_retrain, reason, diagnostics)."""
    diagnostics: Dict[str, Any] = {
        "checks_performed": [],
        "checks_failed": [],
        "checks_skipped": [],
    }

    if prev_state is None:
        diagnostics["checks_performed"].append("no_previous_state")
        diagnostics["checks_failed"].append("no_previous_state")
        return True, "no_previous_state", diagnostics

    diagnostics["checks_performed"].append("data_hash")
    if current_hash and current_hash != prev_state.data_hash:
        diagnostics["checks_failed"].append("data_hash")
        return True, "data_changed", diagnostics

    diagnostics["checks_performed"].append("performance")
    mape = quality_metrics.get("mape", 0.0)
    mape_threshold = float(config.get("retrain_mape_threshold", 15.0))
    if mape > mape_threshold:
        diagnostics["checks_failed"].append("performance")
        return True, f"performance_degraded_mape_{mape:.1f}", diagnostics

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

**Impact**: Clear, auditable retrain decisions.
**Effort**: Medium.
**Testing**:

* Simulated scenarios triggering each condition; confirm reasons/diagnostics.

---

## 🟢 P3 – ENHANCEMENTS, TECH DEBT & PERFORMANCE

---

###   Task 4.1: Improve Detector Forecasting with AR(1) Model

**File**: `core/forecasting.py` (Lines 828–867)

**Issue**: Arbitrary exponential decay; no variance; no type-specific behaviour.

**Required Implementation**:

```python
def forecast_detector_ar1(
    detector_history: pd.Series,
    horizon: int,
    detector_type: str,
) -> pd.DataFrame:
    """AR(1)-based forecast for detector scores."""
    recent = detector_history.dropna().tail(168)
    if len(recent) < 10:
        return exponential_decay_forecast(recent.iloc[-1], horizon)

    x = recent.to_numpy(dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return exponential_decay_forecast(x[-1], horizon)

    phi = np.corrcoef(x[:-1], x[1:])[0, 1]
    phi = float(np.clip(phi, 0.0, 0.99))
    mu = float(np.mean(x * (1.0 - phi)))
    sigma = float(np.std(x[1:] - phi * x[:-1])) or 1e-6

    forecasts = []
    x_t = x[-1]
    for h in range(1, horizon + 1):
        x_t = phi * x_t + mu
        var_h = sigma**2 * (1 - phi**(2 * h)) / (1 - phi**2)
        std_h = float(np.sqrt(max(var_h, 0.0)))
        forecasts.append({
            "horizon": h,
            "forecast": float(x_t),
            "std": std_h,
            "ci_lower": float(x_t - 1.96 * std_h),
            "ci_upper": float(x_t + 1.96 * std_h),
        })

    return pd.DataFrame(forecasts)
```

**Impact**: Detector forecasts become statistically grounded with uncertainty.
**Effort**: Medium.
**Testing**:

* Simulated AR(1) detector sequences to validate.

---

###   Task 4.2: Add Vector Autoregression (VAR) for Sensor Forecasting

**File**: `core/forecasting.py` (Lines 881–1008)

**Issue**: Independent sensor forecasts ignore cross-correlation.

**Required Implementation**:

```python
from statsmodels.tsa.api import VAR

def forecast_sensors_var(
    sensor_df: pd.DataFrame,
    horizon: int,
    max_sensors: int = 10,
) -> pd.DataFrame:
    """Multivariate sensor forecast with cross-correlations (VAR)."""
    sensor_df = sensor_df.sort_index()
    variability = sensor_df.std() / (sensor_df.mean().abs() + 1e-6)
    top_sensors = variability.nlargest(max_sensors).index.tolist()

    data = sensor_df[top_sensors].dropna()
    if len(data) < 50:
        return forecast_sensors_univariate(data, horizon)

    model = VAR(data)
    results = model.fit(maxlags=5, ic="aic")
    fc = results.forecast(data.values[-results.k_ar:], steps=horizon)

    fc_index = pd.date_range(
        start=data.index[-1] + (data.index[-1] - data.index[-2]),
        periods=horizon,
        freq=data.index.freq or "H",
    )

    fc_df = pd.DataFrame(fc, index=fc_index, columns=top_sensors)
    for col in top_sensors:
        std_res = results.resid[col].std()
        fc_df[f"{col}_ci_lower"] = fc_df[col] - 1.96 * std_res
        fc_df[f"{col}_ci_upper"] = fc_df[col] + 1.96 * std_res

    return fc_df
```

**Impact**: Captures inter-sensor dependencies; more realistic future trajectories.
**Effort**: High.
**Testing**:

* Synthetic correlated sensors; VAR vs independent forecast comparison.

---

###   Task 4.3: Add Outlier Detection Before Forecasting

**File**: `core/forecasting.py` (New helper)

**Issue**: Outliers can completely destabilise exponential smoothing.

**Required Implementation**:

```python
def detect_and_remove_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 3.0,
) -> pd.Series:
    """Robust outlier removal before forecasting."""
    series = series.astype(float)
    if series.empty:
        return series

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    else:
        mean = series.mean()
        std = series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std

    mask = (series < lower) | (series > upper)
    series_clean = series.copy()
    series_clean[mask] = np.nan
    series_clean = series_clean.interpolate(method="linear", limit_direction="both")

    n_outliers = int(mask.sum())
    if n_outliers > 0:
        Console.warn(f"[FORECAST] Removed {n_outliers} outliers ({n_outliers / len(series) * 100:.1f}%)")

    return series_clean
```

**Impact**: Protects model from bad sensor or health spikes.
**Effort**: Low.
**Testing**:

* Inject synthetic spikes and verify they’re interpolated.

---

###   Task 4.4: Add Comprehensive Model Diagnostics

**File**: `core/forecasting.py` (New helper)

**Issue**: No residual-based validation (normality, autocorrelation, etc.).

**Required Implementation**:

```python
def validate_forecast_model(
    actual: np.ndarray,
    fitted: np.ndarray,
) -> Dict[str, Any]:
    """Comprehensive model diagnostics."""
    from scipy.stats import shapiro
    from statsmodels.stats.diagnostic import acorr_ljungbox

    actual = np.asarray(actual, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    residuals = actual - fitted

    diagnostics: Dict[str, Any] = {}

    if len(residuals) >= 10:
        _, p_shapiro = shapiro(residuals[: min(5000, len(residuals))])
        diagnostics["residuals_normal_p"] = float(p_shapiro)

        lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
        diagnostics["residuals_autocorr_p"] = float(lb["lb_pvalue"].iloc[0])
    else:
        diagnostics["residuals_normal_p"] = float("nan")
        diagnostics["residuals_autocorr_p"] = float("nan")

    n = len(residuals)
    if n >= 6:
        var_first = float(np.var(residuals[: n // 3]))
        var_last = float(np.var(residuals[-n // 3 :]))
        diagnostics["variance_ratio"] = float(var_last / var_first) if var_first > 1e-9 else 1.0
    else:
        diagnostics["variance_ratio"] = float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs(residuals / np.where(actual == 0, np.nan, actual))) * 100
    diagnostics["mape"] = float(mape)

    naive = np.roll(actual, 1)
    naive[0] = actual[0]
    naive_mse = float(np.mean((actual - naive) ** 2))
    model_mse = float(np.mean(residuals**2))
    diagnostics["theil_u"] = float(np.sqrt(model_mse / naive_mse)) if naive_mse > 0 else float("inf")

    return diagnostics
```

**Impact**: Helps decide if the model is fundamentally mis-specified.
**Effort**: Medium.
**Testing**:

* Validate that Theil’s U < 1 when the model is clearly better than naive.

---

###   Task 4.5: Implement State Versioning and Migration

**File**: `core/forecasting.py` (Lines 1168–1213)

**Issue**: No versioning; state schema changes will break loads.

**Required Implementation**:

```python
@dataclass
class ForecastState:
    VERSION: int = 2
    model_type: str = "ExponentialSmoothing_v2"
    alpha: float = 0.3
    beta: float = 0.2
    data_hash: str = ""
    residual_history: List[float] = field(default_factory=list)
    # ... other existing fields ...


@dataclass
class ForecastStateV3(ForecastState):
    VERSION: int = 3
    # add new fields here


def migrate_state_to_v3(old_state: ForecastState) -> ForecastStateV3:
    Console.info(f"[FORECAST] Migrating state from v{old_state.VERSION} to v3")
    return ForecastStateV3(
        model_type=f"{old_state.model_type}_v3",
        alpha=old_state.alpha,
        beta=old_state.beta,
        data_hash=old_state.data_hash,
        residual_history=getattr(old_state, "residual_history", []),
        # ... map other fields ...
    )


def load_forecast_state_with_migration(
    equip: str,
    equip_id: int,
    sql_client: SqlClient,
) -> Optional[ForecastState]:
    state = load_forecast_state(equip, equip_id, sql_client)
    if state is None:
        return None

    version = getattr(state, "VERSION", 1)
    if version < 3:
        state = migrate_state_to_v3(state)
    return state
```

**Impact**: Allows evolving state structure without breaking deployed systems.
**Effort**: Medium.
**Testing**:

* Mock old `ForecastState` objects and ensure they migrate cleanly.

---

###   Task 5.1: Document All Magic Numbers

**File**: `core/forecasting.py`

**Issue**: Constants have no documented rationale.

**Required Action**:

* For each constant (`MIN_AR1_SAMPLES`, `MIN_FORECAST_SAMPLES`, `BLEND_TAU_HOURS`, `DEFAULT_HAZARD_SMOOTHING_ALPHA`, etc.), add comments explaining:

```python
# Minimum samples required for AR(1) coefficient estimation.
# Rationale: need at least 3 points for stable variance & autocorrelation.
MIN_AR1_SAMPLES = 3
```

**Impact**: Future maintainers understand tuning decisions.
**Effort**: Low.

---

###   Task 5.2: Reduce Logging Verbosity with Levels

**File**: `core/forecasting.py`

**Issue**: Too many `Console.info()` logs; noisy in production.

**Required Fix**:

* Introduce a `logger` or extend `Console` to support `debug/info/warn/error`.
* Convert detailed step logs to `debug`.
* Keep high-level summary logs as `info`.

Example:

```python
logger.debug(f"[FORECAST] Generated {len(health_forecast_df)} horizon points")
logger.info(f"[FORECAST] RUL={rul_stats['rul_median']:.1f}h, "
            f"FailureProb={rul_stats['failure_probability']*100:.1f}%")
```

**Impact**: Cleaner logs; easier troubleshooting.
**Effort**: Low.

---

###   Task 5.3: Complete Type Hints & Schema Docstrings

**File**: `core/forecasting.py`

**Issue**: Missing/weak type hints and schema documentation.

**Required Action**:

* Add type hints for all public functions and ensure DataFrame schemas documented in docstrings:

```python
def smooth_failure_probability_hazard(
    prev_hazard_baseline: float,
    new_probability_series: pd.Series,
    dt_hours: Optional[np.ndarray] = None,
    alpha: float = DEFAULT_HAZARD_SMOOTHING_ALPHA,
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns:
        - Timestamp: datetime
        - HazardRaw: float
        - HazardSmooth: float
        - Survival: float [0, 1]
        - FailureProb: float [0, 1]
    """
```

**Impact**: Easier static analysis (mypy), better tooling support.
**Effort**: Low–Medium.

---

###   Task 5.4: Remove Unnecessary DataFrame Copies

**File**: `core/forecasting.py`

**Issue**: Multiple `.copy()` calls that don’t add safety.

**Required Action**:

* Review code paths, especially around `health_values`, `forecast_df`, etc.
* Replace patterns like:

```python
health_values = pd.Series(health_series, copy=True).astype(float)
```

with:

```python
health_values = pd.Series(health_series, copy=False).astype(float)
```

where safe, or:

```python
health_values = health_series.astype(float)
```

**Impact**: Lower memory footprint and faster execution.
**Effort**: Low.

---

###   Task 5.5: Enhance Error Messages with Context

**File**: `core/forecasting.py`

**Issue**: Error logs lack equip/run context.

**Required Fix**:

```python
Console.warn(
    f"[ENHANCED_FORECAST] Health forecasting failed for EquipID={equip_id}, "
    f"RunID={run_id}, DataPoints={len(health_series)}: {e}"
)
```

**Impact**: Faster debugging in multi-equip setups.
**Effort**: Low.

---

###   Task 5.6: Optimise SQL Queries with WHERE Clauses

**File**: `core/forecasting.py`

**Issue**: `SELECT *` and Python-side filters; inefficient for large histories.

**Required Fix**:

```python
# BEFORE:
cur.execute("""
    SELECT Timestamp, HealthIndex
    FROM dbo.ACM_HealthTimeline
    WHERE EquipID = ?
    ORDER BY Timestamp
""", (equip_id,))

# AFTER (example):
cur.execute("""
    SELECT Timestamp, HealthIndex
    FROM dbo.ACM_HealthTimeline
    WHERE EquipID = ?
      AND Timestamp >= DATEADD(hour, -?, @CurrentTime)
    ORDER BY Timestamp
""", (equip_id, lookback_hours))
```

* Parameterise `@CurrentTime` where appropriate.
* Configurable `forecast.lookback_hours`.

**Impact**: Lower load on SQL Server & Python.
**Effort**: Medium.

---

## 6. ADVANCED FEATURES & LONG-TERM VISION

These build on all above tasks and represent a **V2+** forecasting capability.

---

###   Task 6.1: Adaptive Forecast Engine Framework

**File**: `core/adaptive_forecasting.py` (new)

**Purpose**: Self-tuning Forecast Engine:

* Model library: Holt, ARIMA, Prophet, XGBoost etc.
* Data characteristics analysis (stationarity, seasonality, linearity).
* Hyperparameter / model selection via TS cross-validation or Bayesian optimisation.
* Performance history & automatic model switching.

**Effort**: Very High (multiple sprints).
**Dependencies**: Adaptive α, β; diagnostics; quality metrics.

---

###   Task 6.2: Multi-Model Ensemble Forecasting

**Purpose**: Combine multiple models into an ensemble for robustness:

* Weighted average of forecasts by past performance.
* Possibly quantile-wise combination for CIs.

**Effort**: Very High.

---

###   Task 6.3: Scenario-Based Forecasting

**Purpose**: “What-if” simulation:

* E.g. “If load increases 20%”, “if cooling water temperature increases by 3°C”.
* Needs linking of operating variables to health via causal or regression models.

**Effort**: Very High; depends on physics / domain modelling.

---

## 7. TESTING REQUIREMENTS

### 7.1 Coverage Goals

* **Unit tests** for all P0 & P1 functions.
* **Integration tests** for full pipeline: data load → forecast → RUL → outputs.
* **Regression tests** to ensure fixes don’t break current behaviours unintentionally.
* **Performance tests** for long histories and many equipments.

### 7.2 Key Test Scenarios

1. **Synthetic linear degradation** (known slope, known noise).
2. **Seasonal + trend** (regime & seasonality detection).
3. **Short history / cold start**.
4. **Data gaps & irregular timestamps**.
5. **Many outliers**.
6. **Backtest using real FD_FAN (or other) equipment data**.

---

## 8. SUCCESS METRICS

### P0

* Forecast MAPE improves by > 20% vs current.
* CI coverage at 95% nominal level: observed coverage between 0.94–0.96.
* RUL outputs provide P10/P50/P90; no longer single-point RUL.

### P1

* Forecast quality metrics available in tables / dashboard.
* Temporal blending clearly reduces jumpiness at batch boundaries.
* Failure probability calibration > 0.8 (via reliability diagrams).

### P2

* Adaptive α, β consistently beat fixed parameters across multiple assets.
* Regime-specific models show clear improvement inside regimes.

---

## 9. RECOMMENDED IMPLEMENTATION PHASING

* **Sprint 1 (Weeks 1–2)**:
  P0 Tasks 1.1–1.4 + Architecture/time-index cleanups (A.1–A.3).

* **Sprint 2 (Weeks 3–4)**:
  P1 Tasks 2.1–2.4, model diagnostics, improved blending.

* **Sprint 3 (Weeks 5–6)**:
  P2 Tasks 3.1–3.4, regime models, bootstrap CIs, retrain diagnostics.

* **Sprint 4+**:
  Detector AR(1), VAR sensors, outlier handling, state versioning, tech debt cleanup, and then AdaptiveEngine + ensembles.

---

**End of Comprehensive Forecasting.py Audit & Task List**

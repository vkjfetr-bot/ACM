## Unified RUL Engine – Comprehensive Audit & Task List

**File**: `rul_engine.py` 
**Generated**: 3 Dec 2025
**Severity Scale**: P0 (Critical) → P1 (Major) → P2 (Important) → P3 (Enhancement)

This mirrors the *Forecasting.py Comprehensive Audit & Task List* structure and level of detail.

---

## 🔴 P0 – CRITICAL ISSUES (Must Fix Immediately)

### 🧨 Task 1.1 – LearningState Weights Not Wired to Ensemble (Hard Bug)

**Location**: `RULModel.forecast()`

**Issue**: Ensemble weights are supposed to come from `LearningState` (`ar1_metrics`, `exp_metrics`, `weibull_metrics`), but the code references non-existent attributes (`self.learning_state.ar1`, `self.learning_state.exp`, `self.learning_state.weibull`). This will raise `AttributeError` and/or silently break adaptive weighting. 

```python
raw_weights = np.ones(len(model_names))
for i, name in enumerate(model_names):
    model_key = name.lower()
    if model_key == "ar1":
        raw_weights[i] = self.learning_state.ar1.weight      # ❌ no .ar1
    elif model_key == "exponential":
        raw_weights[i] = self.learning_state.exp.weight      # ❌ no .exp
    elif model_key == "weibull":
        raw_weights[i] = self.learning_state.weibull.weight  # ❌ no .weibull
```

**Required Fix** (wire to `*_metrics`):

```python
for i, name in enumerate(model_names):
    key = name.lower()
    if key == "ar1":
        raw_weights[i] = self.learning_state.ar1_metrics.weight
    elif key == "exponential":
        raw_weights[i] = self.learning_state.exp_metrics.weight
    elif key == "weibull":
        raw_weights[i] = self.learning_state.weibull_metrics.weight
```

**Impact**:

* Ensemble never actually “learns” from performance.
* May crash at runtime depending on Python env / logger behaviour.

**Effort**: Low (1 small block change).

**Testing**:

* Unit test: set `LearningState` weights to asymmetric values (e.g. `ar1=0.8, exp=0.1, weibull=0.1`) and assert `weights_dict` reflects this.
* Integration test: simulate degraded AR1 performance and verify weights shift after updating metrics.

---

### 🧨 Task 1.2 – `load_learning_state` Uses Cursor After Close

**Location**: `load_learning_state()`

**Issue**: Cursor is closed before `cur.description` is accessed to build the column name list. On some DB drivers, `cur.description` becomes invalid when closed. 

```python
cur = sql_client.cursor()
cur.execute("SELECT * FROM dbo.ACM_RUL_LearningState WHERE EquipID = ?", (equip_id,))
row = cur.fetchone()
cur.close()

if row:
    col_names = [desc[0] for desc in cur.description]  # ❌ cursor already closed
    row_dict = dict(zip(col_names, row))
    state = LearningState.from_sql_dict(equip_id, row_dict)
```

**Required Fix**:

```python
cur = sql_client.cursor()
cur.execute("SELECT * FROM dbo.ACM_RUL_LearningState WHERE EquipID = ?", (equip_id,))
row = cur.fetchone()
if row is not None:
    col_names = [desc[0] for desc in cur.description]
cur.close()
...
```

Or better: use a helper that returns `row_dict` directly.

**Impact**:

* RUL engine may always fall back to default `LearningState`, silently disabling learning.
* Potential exceptions in some drivers.

**Effort**: Low.

**Testing**:

* Unit test against a fake cursor that raises if `description` is accessed after `close()`.
* Smoke test: pre-populate `ACM_RUL_LearningState` row and ensure it loads without warning.

---

### 🧨 Task 1.3 – RUL Learning State Never Updated (Broken “Online Learning” Promise)

**Location**: `LearningState`, `run_rul()`

**Issue**:

* `LearningState` is loaded and saved, but `recent_errors`, `mae`, `rmse`, `bias`, `prediction_history`, and `calibration_factor` are never updated anywhere. 
* `run_rul()` explicitly has a TODO note: “future: after learning update” but currently just re-saves unchanged state. 

This means “online learning” & adaptive weights are non-functional – misleading from a product perspective.

**Required Fix**:

1. **Add update function**:

```python
def update_learning_state_from_realized(
    state: LearningState,
    realized_rul_hours: float,
    predicted_rul_hours_by_model: Dict[str, float],
    cfg: RULConfig,
) -> LearningState:
    """
    Update MAE/RMSE/bias and recent_errors for each model,
    then recompute weights using learning_rate.
    """
    for name, pred in predicted_rul_hours_by_model.items():
        err = pred - realized_rul_hours
        metrics = getattr(state, f"{name.lower()}_metrics")
        metrics.recent_errors.append(err)
        metrics.recent_errors = metrics.recent_errors[-cfg.calibration_window:]

        # Update MAE/RMSE/bias (exponential moving)
        alpha = cfg.learning_rate
        abs_err = abs(err)
        sq_err = err**2
        metrics.mae = (1 - alpha) * metrics.mae + alpha * abs_err
        metrics.rmse = np.sqrt((1 - alpha) * (metrics.rmse**2) + alpha * sq_err)
        metrics.bias = (1 - alpha) * metrics.bias + alpha * err

        # Recompute weight: inverse of RMSE (with floor)
        metrics.weight = 1.0 / max(metrics.rmse, 1e-3)
    ...
    # Normalize or rescale calibration_factor if needed
    return state
```

2. **Hook into pipeline**:

   * After we have actual failure time (future integration with event tables), call `update_learning_state_from_realized` and then `save_learning_state`.

**Impact**:

* Right now, the whole “learning” architecture is dead code.
* For a “unified RUL engine with online learning”, this is a correctness & product-expectation breach.

**Effort**: Medium (design + integration + tests).

**Testing**:

* Unit: use synthetic RUL realizations to confirm that worse models’ weights go down over time.
* Regression: ensure weights converge to the best model on stable synthetic data.

---

### 🧨 Task 1.4 – Maintenance Recommendation Uses Wrong Failure Probability Formula

**Location**: `build_maintenance_recommendation()`

**Issue**: `FailureProbAtWindowEnd` is computed from a nonsense formula using RUL only: 

```python
failure_prob_at_window_end = min(1.0, max(0.0, 1.0 - (rul_hours / max(rul_hours, 168))))
```

* If `rul_hours > 168`: denominator is `rul_hours`, so ratio = 1 → probability = 0.
* If `rul_hours < 168`: probability = `1 - RUL/168`, which has no relation to the actual failure curve we computed.
* Completely ignores `failure_curve` and the ensemble’s uncertainty.

Also, maintenance windows use `datetime.now()` instead of `current_time` from the forecast, which is wrong for back-tests or replay.

**Required Fix**:

1. **Use failure curve**:

```python
def compute_failure_prob_at_time(
    failure_curve: pd.DataFrame,
    target_time: pd.Timestamp
) -> float:
    if failure_curve is None or failure_curve.empty:
        return 0.0
    # Nearest timestamp
    idx = (failure_curve["Timestamp"] - target_time).abs().argmin()
    return float(failure_curve.iloc[idx]["FailureProb"])
```

2. **Change signature & logic**:

```python
def build_maintenance_recommendation(
    rul_multipath: Dict[str, Any],
    data_quality: str,
    confidence: float,
    cfg: RULConfig,
    run_id: str,
    equip_id: int,
    current_time: pd.Timestamp,
    failure_curve: Optional[pd.DataFrame],
) -> pd.DataFrame:
    ...

    now = current_time  # instead of datetime.now()
    ...
    preferred_window_end = now + timedelta(hours=rul_hours)
    failure_prob_at_window_end = compute_failure_prob_at_time(
        failure_curve, preferred_window_end
    )
```

**Impact**:

* Maintenance recommendations can be misleading, especially for long RUL.
* Historical replays will show inconsistent time windows.

**Effort**: Medium (signature change, call-site update, tests).

**Testing**:

* Synthetic failure_curve with monotonic increasing probability; verify `FailureProbAtWindowEnd` tracks that curve.
* Compare `now` vs `current_time` behaviour on back-tested runs.

---

### 🧨 Task 1.5 – Ensemble Fallback Behaviour on Total Failure is Too Silent

**Location**: `RULModel.forecast()`

**Issue**: If all models fail to predict (due to internal exceptions), it silently returns a flat 70 ± 10 forecast and `fit_status` with all False. 

```python
if not predictions:
    Console.warn("[RUL-Ensemble] No models available for prediction")
    ...
    return {
        "mean": np.full(n, 70.0),
        "std": np.full(n, 10.0),
        ...
    }
```

**Required Fix**:

* For P0: at least peg `data_quality` / `confidence` to *minimal* and propagate a flag to `run_rul()` so `ACM_RUL_Summary` clearly marks “fallback_rul” and `Confidence≈0.2`.
* Optionally: raise a specific exception that `run_rul()` can handle (and still write a clearly labeled default record).

**Impact**:

* Operators / downstream systems may treat this as a valid forecast instead of “model failure”.

**Effort**: Low-Medium (flag + summary fields).

**Testing**:

* Force failures in AR1/Exp/Weibull and assert summary shows low confidence & special method tag (e.g. `Method='fallback'`).

---

## 🟠 P1 – MAJOR ISSUES (Fix This Sprint)

### 🔶 Task 2.1 – Compute Failure Probabilities More Robustly (Non-Gaussian Option)

**Location**: `compute_failure_distribution()`

**Issue**: Assumes Gaussian health forecast errors and uses simple `Φ((threshold - mean)/std)` at each horizon. 

Problems:

* HealthIndex is bounded [0, 100] and often skewed.
* Gaussian assumption may underestimate tail risks.
* `t_future` parameter is currently unused (smell).

**Required Improvement** (similar to forecasting audit):

1. Keep Gaussian as *baseline* but allow plugging empirical residual history:

```python
def compute_failure_distribution(
    t_future: np.ndarray,
    health_mean: np.ndarray,
    health_std: np.ndarray,
    threshold: float,
    residual_history: Optional[np.ndarray] = None,
    method: str = "gaussian",
) -> np.ndarray:
    if method == "gaussian" or residual_history is None or len(residual_history) < 50:
        z = (threshold - health_mean) / (health_std + 1e-9)
        failure_prob = norm_cdf(z)
    else:
        # Empirical bootstrap
        n_samples = 10000
        base_std = np.std(residual_history) + 1e-9
        samples = np.random.choice(residual_history, size=(n_samples, len(health_mean)), replace=True)
        samples = samples * (health_std / base_std)
        simulated = health_mean[None, :] + samples
        failure_prob = (simulated < threshold).mean(axis=0)

    return np.clip(failure_prob, 0.0, 1.0)
```

**Impact**:

* Better calibrated failure curves, especially for heavy-tailed degradation.

**Effort**: Medium (optional residual input + tests).

**Testing**:

* Synthetic non-Gaussian residuals; verify empirical method improves calibration vs Gaussian.

---

### 🔶 Task 2.2 – Align Multipath RUL Naming With Actual Logic

**Location**: `compute_rul_multipath()`

**Issue**: Docstring claims “trajectory / hazard / energy” paths, but actual implementation uses: 

* “trajectory” → mean forecast crossing
* “conservative” → CI_Lower crossing
* “optimistic” → CI_Upper crossing

And then maps them as:

```python
"rul_hazard_hours": rul_conservative,  # actually CI_Lower
"rul_energy_hours": rul_optimistic,    # actually CI_Upper
```

No hazard or energy models are currently used.

**Required Fix** (two options):

1. **Rename to what it is** (simpler, more honest):

```python
"rul_expected_hours": rul_trajectory,
"rul_conservative_hours": rul_conservative,
"rul_optimistic_hours": rul_optimistic,
"DominantPath": dominant_path,  # "expected", "conservative", "optimistic"
```

and update SQL schema/docs accordingly.

2. **Or** actually implement hazard-based path using `failure_curve` (longer-term P2/P3).

**Impact**:

* Prevents misinterpretation of SQL columns by analytics and downstream users.

**Effort**: Medium (rename columns, docs, and DB alignment).

**Testing**:

* Ensure RUL summary & TS outputs align with updated names; migration script for existing tables.

---

### 🔶 Task 2.3 – Maintenance Recommendation Should Use Forecast Current Time

**Location**: `build_maintenance_recommendation()`

**Issue**: Uses `datetime.now()` instead of `current_time` from the RUL run. 

This breaks historical replays and will misalign maintenance windows when running backtests or scheduled offline jobs.

**Required Fix**:

* Add `current_time` param and replace `now = datetime.now()` with `now = current_time`.

**Impact**:

* Wrong maintenance windows in historical scenarios.

**Effort**: Low (signature change + one line + call-site update).

**Testing**:

* Backtest scenario: run on an old `current_time`; ensure window timestamps are anchored to that.

---

### 🔶 Task 2.4 – SQL Cleanup Assumes `CreatedAt` Columns

**Location**: `cleanup_old_forecasts()`

**Issue**: Uses `MAX(CreatedAt)` but that requires `CreatedAt` to be present in both `ACM_HealthForecast_TS` and `ACM_FailureForecast_TS`. 

```sql
SELECT DISTINCT RunID, 
       ROW_NUMBER() OVER (ORDER BY MAX(CreatedAt) DESC) AS rn
FROM dbo.{table}
WHERE EquipID = ?
GROUP BY RunID
```

If `CreatedAt` is missing or named differently, cleanup fails and may block RUL.

**Required Fix**:

* Either:

  * (a) enforce `CreatedAt` column via schema migration, or
  * (b) fallback to `MAX(Timestamp)` or `LastUpdate` when `CreatedAt` not present.

**Impact**:

* Potential runtime SQL error during cleanup, preventing RUL run.

**Effort**: Medium (schema + code or dynamic column detection).

**Testing**:

* Run against DB with/without `CreatedAt` and ensure cleanup is robust.

---

## 🟡 P2 – IMPORTANT IMPROVEMENTS

### 🟡 Task 3.1 – Strengthen Degradation Model Robustness

**Location**: `AR1Model`, `ExponentialDegradationModel`, `WeibullInspiredModel` 

**Issues**:

1. **Exponential model**:

   * Assumes monotonic exponential decay; if health is flat or improving (post-maintenance), `λ` may become negative, causing *growth* instead of decay.
   * `offset` derived from 5th percentile of last 20 points may be noisy.

2. **Weibull-inspired model**:

   * `self.k = float(-coeffs[0])` assumes negative slope; if data doesn’t fit this shape, `k` might be negative and then `h0 - k t^β` actually *increases*.
   * No clipping on `k`, no sanity checks on `beta` beyond 0.5–3 bounds.

**Required Improvements**:

* After fit, enforce physical constraints:

```python
# After solving coeffs in WeibullInspiredModel.fit
self.k = float(-coeffs[0])
if self.k < 0:
    # fallback to AR1-only behaviour or clamp
    Console.warn("[Weibull] Negative degradation rate, falling back or clamping")
    self.k = 0.0  # or treat as failed fit
```

* For Exponential: if fitted λ < 0 and residuals large, mark model as `fit_succeeded=False`.

**Impact**:

* Avoids clearly unphysical forecasts (health shooting upwards to 150%).

**Effort**: Medium (checks + thresholds).

**Testing**:

* Synthetic examples with improving health; ensure these models gracefully fail or clamp instead of producing nonsense.

---

### 🟡 Task 3.2 – Enforce Health Bounds (0–100) in All Forecasts

**Location**: `compute_rul()`, Degradation models, `compute_failure_distribution()`

**Issue**: None of the forecast paths clamp HealthIndex to [0, 100]. Negative health or >100 can appear due to noise or model misfit, which then flows into probabilities. 

**Required Fix**:

* After ensemble forecast:

```python
ensemble_mean = np.clip(ensemble_mean, 0.0, 100.0)
```

* Similarly, clamp CI_Lower and CI_Upper to [0, 100].

**Impact**:

* Physically consistent forecasts; better operator trust.

**Effort**: Low.

**Testing**:

* Inject noise causing >100 predictions and ensure outputs are clipped.

---

### 🟡 Task 3.3 – Tighten Data Quality Integration

**Location**: `_assess_data_quality()`, `compute_rul()`, `compute_confidence()` 

**Issue**:

* Data quality is computed (`OK/SPARSE/GAPPY/FLAT`) but only lightly used via `quality_score` in `compute_confidence`.
* No adjustments to forecast horizon or model selection based on quality.

**Required Improvements**:

* If `SPARSE` or `GAPPY`, reduce `max_forecast_hours` dynamically (e.g., half).
* If `FLAT`, consider skipping RUL and report “no degradation trend” with high RUL and low urgency.

**Impact**:

* More realistic forecasts when data is poor.

**Effort**: Medium (config-based policy).

**Testing**:

* Build small synthetic datasets to trigger each data_quality and validate behaviour changes.

---

### 🟡 Task 3.4 – Ensure RUL_TS Bounds Non-Negative & Consistent

**Location**: `make_rul_ts()` 

**Issue**:

* `LowerBound` is clamped to `>=0`, but `UpperBound` is not.
* As time moves forward (elapsed_hours > rul_final), `UpperBound` can become negative.

```python
"LowerBound": [max(0.0, lower_bound - elapsed)],
"UpperBound": [upper_bound - elapsed],
```

**Required Fix**:

```python
upper = max(0.0, upper_bound - elapsed)
...
"UpperBound": [... upper ...]
```

**Impact**:

* Prevents negative RUL bounds in SQL and dashboards.

**Effort**: Low.

**Testing**:

* Test at far horizons where elapsed_hours > upper_bound.

---

## 🟢 P3 – ENHANCEMENTS & TECHNICAL DEBT

### 🟢 Task 4.1 – Implement Real Multipath (Hazard / Energy) RUL

**Location**: `compute_rul_multipath()`, `compute_failure_distribution()`

**Idea** (long-term):

* Path 1: trajectory crossing (already implemented).
* Path 2 (hazard): derive RUL from integrated hazard from failure curve (e.g., time when cumulative failure probability reaches 50%).
* Path 3 (energy / degradation): integrate “health deficit” over time until a budget is exhausted.

This will align naming with actual math and provide richer views.

---

### 🟢 Task 4.2 – Expose Model Diagnostics in RUL Summary

**Location**: `make_rul_summary()` 

**Enhancements**:

* Add fields like `SamplingIntervalHours`, `ModelCount`, `ModelsFitted`, `FallbackFlag`.
* Include simple performance metrics once learning is implemented (e.g., `AR1_RMSE`, etc.) as optional columns.

---

### 🟢 Task 4.3 – Better Type Hints & Docstrings

**Location**: Entire file

**Issue**: Many functions are nicely typed, but some public APIs lack detailed return type docstrings (e.g., shapes and columns of DataFrames).

**Action**:

* Add detailed type hints + docstrings for all public helpers: `compute_rul`, `run_rul`, `build_sensor_attribution`, etc.

---

### 🟢 Task 4.4 – Config Surface Clean-Up

**Location**: `RULConfig`, `run_rul()`

**Issues**:

* `calibration_window`, `enable_online_learning` exist but are unused at present.
* Health bands for maintenance are fixed; might be better as DB config.

**Action**:

* Either wire them fully (preferred) or trim and document.

---

## 📊 TESTING REQUIREMENTS (RUL Engine)

### Unit Tests (Minimum)

1. **Model Fits**

   * AR1 with synthetic AR(1) data → verify recovered φ & forecast RMSE.
   * Exponential model on exponential decay data → λ close to true.
   * Weibull model on power-law data → β & k sensible.

2. **Ensemble**

   * Different `LearningState` weights → check ensemble_mean & weights structure.
   * Failure of one model → others still used.

3. **RUL Multipath**

   * Construct simple, monotonic decreasing health forecast crossing threshold at known times → verify RUL_trajectory / conservative / optimistic.

4. **IO / SQL**

   * `load_health_timeline` path: cache vs SQL; row limit/downsample.
   * `load_sensor_hotspots` behaviour with/without new columns.

5. **Maintenance Recommendation**

   * Different RUL bands produce correct `Urgency` & `Action`.

### Integration / Regression

* End-to-end `run_rul()` on:

  * Healthy asset (>90% health): verify `_healthy_rul_result` path.
  * Degrading asset with known synthetic failure time: RUL estimation error within tolerance.
* Time replay: run same health timeline with different `current_time` and ensure maintenance windows and RUL_TS align.

---

## 📈 SUCCESS METRICS

* **P0 fixes**

  * No AttributeErrors / cursor errors in production logs from RUL engine.
  * Ensemble weights demonstrably influence forecasts.
  * Maintenance recommendation uses actual failure curve probability & correct time base.

* **P1 fixes**

  * Failure curves calibrated within ±10% on synthetic data.
  * Column naming & semantics match actual maths (no “hazard” misnomers).

* **P2+**

  * LearningState updates result in improved RUL accuracy (>10% MAE improvement over static weights on backtests).

---

## 🗓 RECOMMENDED IMPLEMENTATION ORDER

**Sprint 1 (P0)**

* Task 1.1 (weights wiring)
* Task 1.2 (cursor bug)
* Task 1.3 (minimal learning state update hook & no-op safe wiring)
* Task 1.4 (maintenance probability + current_time)
* Task 1.5 (fallback labelling)

**Sprint 2 (P1)**

* Task 2.x (failure distribution, naming, cleanup robustness)
* Task 3.2 & 3.4 (bounds & health clamp – cheap wins)

**Sprint 3+ (P2/P3)**

* Physical constraints & hazard path
* Real online learning, advanced diagnostics & VAR-style interactions if needed.

---


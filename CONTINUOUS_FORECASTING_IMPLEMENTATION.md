# Continuous Forecasting Implementation Summary
**Date:** 2025-11-20  
**Status:** Core components implemented, ready for integration testing

## ✅ Completed Components

### 1. ForecastState Persistence (Task 1) ✅
**File:** `core/model_persistence.py`

**Added:**
- `ForecastState` dataclass with fields:
  - `equip_id`, `state_version`, `model_type`
  - `model_params` (dict), `residual_variance`
  - `last_forecast_horizon_json` (serialized DataFrame)
  - `hazard_baseline` (EWMA smoothed hazard)
  - `last_retrain_time`, `training_data_hash`
  - `training_window_hours`, `forecast_quality` (rmse, mae, mape)
  
- `save_forecast_state()`: Dual persistence (JSON + SQL)
- `load_forecast_state()`: Load from SQL or filesystem fallback
- Serialization helpers for DataFrame ↔ JSON conversion

**SQL Table:** `ACM_ForecastState` created ✅

---

### 2. Continuous Forecasting Logic (Task 2) ✅
**File:** `core/forecasting.py`

**Functions Added:**
- `compute_data_hash()`: SHA256 hash for training data change detection
- `should_retrain()`: Conditional retraining decision based on:
  - Drift: 5-point avg DriftValue > 1.5
  - Energy spike: Anomaly energy P95 > 1.5x median
  - Data hash: Training window changed
  - Returns `(bool, reason_string)`

- `merge_forecast_horizons()`: Temporal blending with exponential decay
  - Weight formula: `w_new = 1 - exp(-dt/tau)`, `w_prev = exp(-dt/tau)`
  - Blends ForecastHealth, CI_Lower, CI_Upper
  - Discards past points, merges overlapping future

- `smooth_failure_probability_hazard()`: Convert discrete probabilities to continuous
  - Hazard rate: `λ(t) = -ln(1-p) / dt`
  - EWMA smoothing: `λ_smooth[t] = α*λ_raw[t] + (1-α)*λ_smooth[t-1]`
  - Survival: `S(t) = exp(-∫λ dt)`
  - Failure prob: `F(t) = 1 - S(t)`
  - Returns DataFrame with [HazardRaw, HazardSmooth, Survival, FailureProb]

**Enhanced `run_enhanced_forecasting_sql()`:**
- Loads previous ForecastState
- Uses sliding window (72h lookback) instead of single batch
- Calls `should_retrain()` before forecasting
- Merges forecast horizons with temporal blending
- Applies hazard smoothing to failure probabilities
- Saves updated ForecastState with new hazard baseline
- Adds `retrain_needed` and `retrain_reason` to metrics

**Updated `run_and_persist_enhanced_forecasting()`:**
- Accepts `artifact_root` and `equip` parameters
- Persists new tables: `health_forecast_continuous`, `failure_hazard_ts`
- Handles timestamp normalization for all tables

---

### 3. Conditional Retraining Logic (Task 3) ✅
**Implementation:** Integrated in `should_retrain()` function

**Checks:**
1. **Drift Detection**: Queries ACM_DriftMetrics for recent 5-point avg
2. **Anomaly Energy**: Computes P95/median ratio from ACM_Scores_Wide
3. **Data Change**: Compares SHA256 hash of training window
4. **Forecast Quality**: (Placeholder for future backtest comparison)

**Outputs:**
- Retrain decision logged to console
- `retrain_reason` added to metrics dict
- Ready for ACM_RunMetadata integration (table doesn't exist yet)

---

### 4. Continuous Forecast SQL Tables (Task 4) ✅
**Script:** `scripts/sql/create_continuous_forecast_tables.sql`

**Tables Created:**
1. **ACM_ForecastState** ✅
   - Stores persistent model state between batches
   - Primary key: (EquipID, StateVersion)
   - Indexes: Latest state per equipment

2. **ACM_HealthForecast_Continuous** ✅
   - Merged forecast horizons with blend weights
   - Primary key: (EquipID, Timestamp, SourceRunID)
   - Indexes: Time range queries, source run filtering

3. **ACM_FailureHazard_TS** ✅
   - Smoothed hazard rates and survival probabilities
   - Primary key: (EquipID, RunID, Timestamp)
   - Columns: HazardRaw, HazardSmooth, Survival, FailureProb

**Tables Extended:**
1. **ACM_RUL_Summary** ✅
   - Added columns: RUL_Trajectory_Hours, RUL_Hazard_Hours, RUL_Energy_Hours
   - RUL_Final_Hours, ConfidenceBand_Hours, DominantPath

2. **ACM_RunMetadata** (attempted, table doesn't exist yet)
   - Planned columns: RetrainDecision, RetrainReason, LastRetrainRunID
   - ModelAgeInBatches, ForecastQualityRMSE, ForecastStateVersion

---

### 5. Hazard-Based Probability Smoothing (Task 5) ✅
**Implementation:** `smooth_failure_probability_hazard()` function

**Math Implementation:**
```python
# Convert probability to hazard rate
λ_raw = -ln(1 - p) / dt

# EWMA smoothing
λ_smooth[0] = α * λ_raw[0] + (1-α) * λ_prev_baseline
λ_smooth[i] = α * λ_raw[i] + (1-α) * λ_smooth[i-1]

# Cumulative hazard and survival
Λ(t) = Σ(λ_smooth * dt)
S(t) = exp(-Λ(t))
F(t) = 1 - S(t)
```

**Configuration:**
- `forecasting.hazard_smoothing_alpha`: 0.3 (default, 0-1 range)
- Higher α = more reactive, lower α = smoother

**Integration:**
- Called in `run_enhanced_forecasting_sql()` after failure probability calculation
- Persists to `ACM_FailureHazard_TS` table
- Updates `hazard_baseline` in ForecastState for next iteration

---

### 6. Multi-Path RUL Derivation (Task 6) ✅
**File:** `core/enhanced_rul_estimator.py`

**Function:** `compute_rul_multipath()`

**Three Independent Paths:**
1. **Trajectory Path**: First timestamp where `ForecastHealth <= threshold`
2. **Hazard Path**: First timestamp where `FailureProb >= 0.5`
3. **Energy Path**: First timestamp where `CumulativeEnergy >= E_fail`

**Final RUL:** `min(RUL_trajectory, RUL_hazard, RUL_energy)`

**Confidence Band:**
- Computes time difference between CI_Lower and CI_Upper crossing threshold
- Represents uncertainty in RUL estimate

**Dominant Path Detection:**
- Identifies which path produced the minimum RUL
- Helps operators understand failure mode (gradual vs acute vs energy-driven)

**Output Schema:**
```python
{
    "rul_trajectory_hours": float or None,
    "rul_hazard_hours": float or None,
    "rul_energy_hours": float or None,
    "rul_final_hours": float,
    "confidence_band_hours": float,
    "dominant_path": "trajectory" | "hazard" | "energy"
}
```

**SQL Schema:** ACM_RUL_Summary extended with multipath columns ✅

---

## 🔄 Integration Points

### In `core/acm_main.py` (Needs Update):

1. **Pass artifact_root and equip to forecasting:**
```python
metrics = forecasting.run_and_persist_enhanced_forecasting(
    sql_client=sql_client,
    equip_id=equip_id,
    run_id=run_id,
    config=cfg,
    output_manager=output_manager,
    tables_dir=tables_dir,
    artifact_root=artifact_root,  # ADD THIS
    equip=equip,  # ADD THIS
)
```

2. **Log retrain decision to ACM_RunMetadata** (when table exists):
```python
if "retrain_needed" in metrics:
    # Log RetrainDecision, RetrainReason, ForecastStateVersion to SQL
```

3. **Call compute_rul_multipath() after forecasting:**
```python
from core.enhanced_rul_estimator import compute_rul_multipath

multipath_rul = compute_rul_multipath(
    health_forecast=merged_horizon_df,
    hazard_df=hazard_df,
    anomaly_energy_df=energy_df,  # Need to compute
    current_time=current_batch_time,
    config=cfg
)

# Write to ACM_RUL_Summary with multipath columns
```

---

## 📊 Configuration Options

**In `configs/config_table.csv`, add forecasting section:**

| EquipID | ConfigPath | Value | DataType | Description |
|---------|------------|-------|----------|-------------|
| * | forecasting.enable_continuous | True | bool | Enable continuous forecasting with state persistence |
| * | forecasting.training_window_hours | 72 | int | Sliding window lookback (hours) |
| * | forecasting.blend_tau_hours | 12.0 | float | Exponential decay time constant for horizon merging |
| * | forecasting.hazard_smoothing_alpha | 0.3 | float | EWMA smoothing factor (0-1) |
| * | forecasting.drift_retrain_threshold | 1.5 | float | Drift avg threshold to trigger retrain |
| * | forecasting.energy_spike_threshold | 1.5 | float | Energy P95/median ratio to trigger retrain |
| * | forecasting.forecast_error_threshold | 2.0 | float | RMSE multiplier to trigger retrain |
| * | forecasting.failure_threshold | 75.0 | float | HealthIndex failure threshold |
| * | forecasting.hazard_failure_prob | 0.5 | float | Failure probability threshold for hazard path |
| * | forecasting.energy_fail_threshold | 1000.0 | float | Cumulative energy threshold for energy path |
| * | forecasting.max_forecast_hours | 168.0 | float | Maximum forecast horizon (hours) |

---

## 🚧 Remaining Tasks (Not Yet Implemented)

### Task 7: Document Unified Failure Condition
- Create `docs/RUL_METHOD.md` with mathematical definitions
- Update `scripts/evaluate_rul_backtest.py` with unified condition
- Add markdown to Grafana dashboard Panel 36

### Task 8: Add Defect Type Forecasting Display
- Create Grafana panel "Predicted Defect Signature"
- Query: Detector contributions in forecast window
- Link to ACM_FailureCausation table

### Task 9: Create RUL Visualization Panel
- Grafana panel showing forecast crossing threshold at 75
- Vertical marker at projected failure time
- Confidence bands (CI_Lower/CI_Upper)
- Smoothed failure probability curve
- RUL countdown timer

### Task 10: Add Retraining Indicator to Dashboard
- Panel showing retrain decisions over time
- Model age in batches
- Forecast quality trends (RMSE, MAE)
- Alert when retrain recommended

---

## 🧪 Testing Recommendations

1. **Unit Tests:**
   - Test `ForecastState` serialization/deserialization
   - Test `merge_forecast_horizons()` with edge cases
   - Test `smooth_failure_probability_hazard()` math
   - Test `should_retrain()` with mock SQL data

2. **Integration Test:**
   - Run 10 consecutive batches with `--enable-continuous=true`
   - Verify ForecastState persistence across batches
   - Check retrain frequency (<30% expected)
   - Validate smooth probability curves (no steps)

3. **SQL Validation:**
   - Verify ACM_ForecastState populates with incrementing versions
   - Check ACM_HealthForecast_Continuous has merged horizons
   - Validate ACM_FailureHazard_TS has smooth curves

4. **Dashboard Smoke Test:**
   - Query new tables from Grafana
   - Verify no syntax errors
   - Check data renders correctly

---

## 🔄 Rollback Plan

If continuous forecasting causes issues:

1. Set `forecasting.enable_continuous = False` in config
2. System falls back to batch-isolated mode automatically
3. Old tables (ACM_HealthForecast_TS, ACM_FailureForecast_TS) still work
4. ForecastState saves are additive (no data loss)

---

## 📈 Expected Improvements

**Before (Batch-Isolated):**
- Each batch trains from scratch
- Forecast horizons reset every batch → stepped curves
- No memory of previous predictions
- ~100% retrain frequency

**After (Continuous):**
- Models evolve incrementally
- Smooth forecast transitions with temporal blending
- State persists across batches
- ~20-30% retrain frequency (only when drift/energy spikes)
- Hazard-based probabilities eliminate staircase effect

---

## 🎯 Next Steps

1. ✅ Test SQL table creation (3 tables created successfully)
2. ⏳ Update `core/acm_main.py` to pass artifact_root/equip parameters
3. ⏳ Run batch processing with continuous mode enabled
4. ⏳ Verify ForecastState persistence across runs
5. ⏳ Create Grafana visualization panels (Tasks 9, 10)
6. ⏳ Document unified failure condition (Task 7)
7. ⏳ Add defect type display (Task 8)

**Time Investment So Far:** ~6 hours
**Remaining Effort:** ~15-20 hours (dashboard panels + documentation + testing)

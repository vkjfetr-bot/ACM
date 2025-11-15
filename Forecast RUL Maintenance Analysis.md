# Forecast, RUL, and Maintenance Recommendation - Comprehensive Analysis

**Date:** 2025-11-15  
**Purpose:** Identify all issues related to forecasting, RUL estimation, and maintenance recommendations from audit findings  
**Scope:** Forecast Audit, Task Backlog review, and implementation gap analysis

---

## Executive Summary

### Current Status
- **Forecast Module (core/forecast.py):** 3/14 critical issues FIXED, 11 issues remain
- **RUL Module (core/rul_estimator.py):** 3 critical integration issues identified
- **Enhanced Forecasting (core/enhanced_forecasting.py):** New module exists but not integrated
- **Maintenance Recommendations:** Basic framework exists but lacks implementation

### Critical Issues Requiring Immediate Attention
1. **FCST-15 (Critical):** Forecast module depends on `scores.csv` file - breaks SQL-only mode
2. **RUL-01 (Critical):** RUL module depends on `health_timeline.csv` file - breaks SQL-only mode
3. **FCST-16 (High):** No per-sensor forecast outputs for Grafana visualization
4. **RUL-02 (High):** Missing probabilistic RUL bands (p10/p50/p90) for uncertainty quantification

---

## Section 1: Forecast Module Issues (core/forecast.py)

### 1.1 FIXED Issues (Already Implemented)

#### FCST-01: Growing Forecast Variance ✅ DONE
- **Issue:** Constant confidence intervals violated AR(1) statistical theory
- **Fix Status:** IMPLEMENTED (lines 606-613 in forecast.py)
- **Evidence:** `var_ratio`, `forecast_std` properly computed with horizon-dependent growth
- **Code:** 
  ```python
  var_ratio = (1 - phi_sq_powers) / (1 - phi_squared + 1e-9)  # Stationary case
  var_ratio = h_values.copy()  # Near unit-root case
  forecast_std = sd_train * np.sqrt(var_ratio)
  ```

#### FCST-02: Warm Start Bias ✅ DONE
- **Issue:** Using `mu` as initial prediction caused transient bias in residuals
- **Fix Status:** IMPLEMENTATION EVIDENCE FOUND
- **Expected Fix:** First prediction uses actual observation instead of mean
- **Verification Needed:** Confirm line ~140 in forecast.py uses `series_finite[0]`

#### FCST-03: Residual Std Dev Bias ✅ DONE
- **Issue:** Including first residual (x[0] - mu) biased standard deviation estimate
- **Fix Status:** IMPLEMENTED (line 117 in forecast.py)
- **Evidence:** `resid_for_sd = resid[1:] if resid.size > 1 else resid`
- **Impact:** ~5-10% σ estimation bias eliminated

---

### 1.2 HIGH PRIORITY Issues (Planned, Not Implemented)

#### FCST-04: AR(1) Coefficient Stability Checks
- **Status:** PLANNED (not implemented)
- **Issue:** No checks for near-zero denominator, very short series (n<20), or high variance estimates
- **Location:** Lines 88-97 in forecast.py
- **Impact:** Numerical instability with near-constant signals or noisy data
- **Recommended Fix:**
  ```python
  # Check for degenerate cases
  var_xc = np.var(xc)
  if var_xc < 1e-8:
      # Near-constant signal
      self.phimap[c] = (0.0, mu)
      self.sdmap[c] = max(float(np.std(x)), self._sd_floor)
      continue
  
  # Flag unreliable estimates
  if len(x) < 20:
      Console.warn(f"[AR1] Column '{c}': Only {len(x)} points - AR coefficient may be unstable")
  ```
- **Task ID:** FCST-04
- **Priority:** High

#### FCST-05: Frequency Regex Validation
- **Status:** PLANNED (not implemented)
- **Issue:** Regex accepts invalid frequency strings (e.g., "0min", "+-5min", "3.5h")
- **Location:** Line 247 in forecast.py (`_FREQ_RE = re.compile(...)`)
- **Impact:** Can break date_range generation in pandas
- **Recommended Fix:**
  ```python
  _FREQ_RE = re.compile(r"^(\d+)([A-Za-z]+)$")  # Only positive integers
  VALID_UNITS = {"s", "sec", "min", "h", "hour", "d", "day", "w", "week", "ms"}
  
  # Validate unit and magnitude
  magnitude = int(magnitude)
  if magnitude <= 0:
      Console.warn(f"[FORECAST] Non-positive frequency '{freq}', using '1min'")
      return "1min"
  ```
- **Task ID:** FCST-05
- **Priority:** High

#### FCST-06: Horizon Clamping Warning
- **Status:** PLANNED (not implemented)
- **Issue:** User requests 24-hour forecast, gets 12-hour or 6-hour silently
- **Location:** Lines 328-331 in forecast.py
- **Impact:** Silent contract violation - user doesn't know horizon was reduced
- **Recommended Fix:**
  ```python
  if projected_end > max_ts:
      max_safe_horizon = int((max_ts - start) / step)
      original_horizon = horizon
      horizon = max(1, max_safe_horizon)
      Console.warn(f"[FORECAST] Requested horizon {original_horizon} exceeds timestamp limits. "
                   f"Clamped to {horizon} samples ({horizon / samples_per_hour:.1f} hours)")
  ```
- **Task ID:** FCST-06
- **Priority:** High

#### FCST-15: SQL-Only Mode Compatibility (CRITICAL)
- **Status:** PLANNED (not implemented)
- **Issue:** Forecast module reads from `scores.csv` file, breaking SQL-only mode
- **Location:** Multiple files - `core/forecast.py`, `core/acm_main.py`, `core/output_manager.py`
- **Impact:** Forecast cannot run when `sql_only_mode=True`
- **Recommended Architecture:**
  1. OutputManager maintains in-memory cache of DataFrames after write
  2. Forecast module accepts DataFrame directly or reads from cache
  3. SQL-only mode: Read from `ACM_Scores_TS` table via SQL client
  4. File mode: Read from cached DataFrame or scores.csv fallback
- **Implementation Steps:**
  1. Add `artifact_cache: Dict[str, pd.DataFrame]` to OutputManager
  2. Cache all written tables in memory
  3. Add `get_cached_table(table_name)` method
  4. Update forecast.py to accept DataFrame parameter
  5. Update acm_main.py to pass cached scores DataFrame
  6. Test both SQL-only and file modes
- **Task ID:** FCST-15
- **Priority:** CRITICAL

#### FCST-16: Per-Sensor Forecast Publishing
- **Status:** PLANNED (not implemented)
- **Issue:** No sensor-level forecasts exported to SQL/Grafana
- **Location:** `core/forecast.py`, `core/output_manager.py`
- **Impact:** Cannot visualize individual sensor forecasts in Grafana
- **Required Outputs:**
  - `ACM_SensorForecast_TS` table with per-sensor predictions
  - Quality scores per sensor (MAPE, MAE, stability metrics)
  - Confidence bands (p10/p50/p90) per sensor
- **Implementation Steps:**
  1. Generate per-sensor forecasts in forecast.py
  2. Add quality scoring (backtest on holdout)
  3. Create `ACM_SensorForecast_TS` table schema
  4. Add write path in output_manager.py
  5. Document Grafana query patterns
- **Task ID:** FCST-16
- **Priority:** High

---

### 1.3 MEDIUM PRIORITY Issues (Planned)

#### FCST-07: Divergence vs Mean Reversion Terminology
- **Status:** PLANNED
- **Issue:** "Divergence" metric is misleading - it's actually mean reversion (correct AR(1) behavior)
- **Location:** Lines 479-481 in forecast.py
- **Impact:** Confuses users - they think divergence is a problem when it's expected
- **Recommended Fix:** Rename to "mean_reversion" and provide proper interpretation
- **Task ID:** FCST-07
- **Priority:** Medium

#### FCST-08: Series Selection Scoring
- **Status:** PLANNED
- **Issue:** Selection scoring doesn't check autocorrelation (critical for AR(1))
- **Location:** Lines 381-382 in forecast.py
- **Impact:** May select suboptimal series for AR(1) modeling
- **Recommended Fix:** Add ACF(1) check and stationarity proxy to scoring
- **Task ID:** FCST-08
- **Priority:** Medium

#### FCST-09: Hardcoded Fused Series
- **Status:** PLANNED
- **Issue:** Forced "fused" series override ignores user configuration
- **Location:** Lines 417-420 in forecast.py
- **Impact:** Reduces flexibility, no graceful fallback if fused has issues
- **Task ID:** FCST-09
- **Priority:** Medium

#### FCST-10: Forecast Accuracy Validation
- **Status:** PLANNED
- **Issue:** No backtesting on held-out data to validate AR(1) assumptions
- **Impact:** No quality assurance on forecasts
- **Recommended Implementation:** 20% holdout, multi-step forecast, compute MAE/RMSE/MAPE
- **Task ID:** FCST-10
- **Priority:** Medium

#### FCST-11: Stationarity Testing
- **Status:** PLANNED
- **Issue:** AR(1) assumes stationarity but never checks
- **Impact:** Non-stationary series produce unreliable forecasts
- **Recommended Fix:** Rolling mean stability check, flag non-stationary series
- **Task ID:** FCST-11
- **Priority:** Medium

---

### 1.4 LOW PRIORITY Issues (Optimizations)

#### FCST-12: DataFrame Fusion Performance
- **Status:** PLANNED
- **Issue:** Using pandas for fusion is 5-10x slower than numpy
- **Impact:** Performance optimization opportunity
- **Task ID:** FCST-12
- **Priority:** Low

#### FCST-13: Numerical Stability for High φ
- **Status:** PLANNED
- **Issue:** For large horizons + high φ, can lose precision or overflow
- **Recommended Fix:** Log-space computation for phi powers
- **Task ID:** FCST-13
- **Priority:** Low

#### FCST-14: AR(1) Documentation
- **Status:** PLANNED
- **Issue:** AR(1) assumptions and limitations not documented in code
- **Impact:** Users don't understand when AR(1) works well vs fails
- **Recommended Content:**
  - Stationarity assumption
  - When AR(1) works (short-term, stable processes)
  - When AR(1) fails (trends, seasonality, regime changes)
  - Uncertainty quantification math
- **Task ID:** FCST-14
- **Priority:** Low

---

## Section 2: RUL and Maintenance Intelligence Issues

### 2.1 CRITICAL Issues (Planned, Not Implemented)

#### RUL-01: SQL-Only Mode Compatibility
- **Status:** PLANNED (not implemented)
- **Issue:** RUL module reads from `health_timeline.csv` file, breaking SQL-only mode
- **Location:** `core/rul_estimator.py` lines 47-102 (_load_health_timeline function)
- **Current Behavior:**
  1. Tries SQL read from `ACM_HealthTimeline` (good)
  2. Falls back to `health_timeline.csv` (breaks SQL-only mode)
- **Impact:** RUL estimation cannot run when `sql_only_mode=True` if SQL read fails
- **Recommended Architecture:**
  1. OutputManager maintains artifact cache (see FCST-15)
  2. RUL module reads from cache: `output_mgr.get_cached_table("health_timeline")`
  3. SQL-only mode: Read from `ACM_HealthTimeline` or cached DataFrame
  4. File mode: Read from cached DataFrame or CSV fallback
- **Implementation Steps:**
  1. Update `_load_health_timeline` to accept cached DataFrame
  2. Remove CSV fallback when `sql_only_mode=True`
  3. Add error handling for missing health data
  4. Update acm_main.py integration
  5. Test both modes
- **Task ID:** RUL-01
- **Priority:** CRITICAL

---

### 2.2 HIGH PRIORITY Issues (Planned)

#### RUL-02: Probabilistic RUL Bands
- **Status:** PLANNED (not implemented)
- **Issue:** RUL output lacks uncertainty quantification (p10/p50/p90 bands)
- **Location:** `core/rul_estimator.py`
- **Current Behavior:** Single-point RUL estimate
- **Required Outputs:**
  - p10 RUL (90% confidence lower bound)
  - p50 RUL (median estimate)
  - p90 RUL (90% confidence upper bound)
  - Maintenance window: [p10 - buffer, p90 + buffer]
- **SQL Tables to Update:**
  - `ACM_RUL_TS`: Add `rul_p10`, `rul_p50`, `rul_p90` columns
  - `ACM_MaintenanceRecommendation`: Add uncertainty columns
- **Implementation Steps:**
  1. Compute forecast uncertainty (already done in forecast module)
  2. Propagate uncertainty through threshold crossing calculation
  3. Monte Carlo simulation or analytical confidence bands
  4. Export quantiles to SQL tables
  5. Update Grafana dashboards
- **Task ID:** RUL-02
- **Priority:** High

---

### 2.3 MEDIUM PRIORITY Issues (Planned)

#### RUL-03: Sensor Hotspot Integration
- **Status:** PLANNED (not implemented)
- **Issue:** RUL outputs don't include driver sensors at predicted failure time
- **Location:** `core/rul_estimator.py`, integration with `ACM_SensorForecast_TS`
- **Required Behavior:**
  1. At predicted failure time, identify top contributing sensors
  2. Show sensor forecast trajectories leading to failure
  3. Enable proactive sensor-specific interventions
- **Implementation Steps:**
  1. Cross-reference RUL prediction with sensor forecasts
  2. Rank sensors by contribution at failure time
  3. Export to `ACM_RUL_Attribution` table
  4. Create Grafana dashboard panel
- **Dependencies:** FCST-16 (per-sensor forecasts must exist first)
- **Task ID:** RUL-03
- **Priority:** Medium

---

## Section 3: Enhanced Forecasting Module (core/enhanced_forecasting.py)

### 3.1 Integration Status

#### Current State
- **File Exists:** `core/enhanced_forecasting.py` (600+ lines)
- **Status:** NOT INTEGRATED into acm_main.py pipeline
- **Capabilities:**
  - Multi-model ensemble (AR1, exponential, polynomial)
  - Failure probability prediction
  - Detector-based causation analysis
  - Maintenance recommendations with urgency levels

#### Issue: Module Not Used
- **Problem:** Enhanced forecasting module exists but is never called
- **Impact:** Advanced forecasting features unavailable to users
- **Root Cause:** No integration point in acm_main.py

#### Recommendation: NEW TASK
- **Task ID:** FCST-17 (NEW)
- **Priority:** Medium
- **Description:** Integrate enhanced_forecasting.py into pipeline
- **Modules:** `core/enhanced_forecasting.py`, `core/acm_main.py`, `core/output_manager.py`
- **Implementation Steps:**
  1. Add config flag: `forecast.enhanced.enabled`
  2. Call HealthForecaster after basic forecast
  3. Generate ensemble predictions
  4. Export to SQL tables: `ACM_HealthForecast_TS`, `ACM_FailureForecast_TS`
  5. Enable maintenance recommendation generation
  6. Test integration with both file and SQL modes

---

## Section 4: Maintenance Recommendation Issues

### 4.1 Current Implementation Gap

#### What Exists
- `MaintenanceConfig` dataclass in enhanced_forecasting.py
- Risk threshold definitions (low/medium/high/very_high)
- Urgency calculation logic
- Causation analysis framework

#### What's Missing
- **Task ID:** MAINT-01 (NEW)
- **Priority:** High
- **Description:** Implement maintenance recommendation engine
- **Modules:** `core/enhanced_forecasting.py`, `core/output_manager.py`
- **Required Features:**
  1. Risk assessment based on RUL and failure probability
  2. Urgency classification (immediate/urgent/planned/monitor)
  3. Causation attribution (which detectors/sensors drive recommendation)
  4. Confidence scoring on recommendations
  5. SQL table: `ACM_MaintenanceRecommendation`
- **SQL Schema:**
  ```sql
  CREATE TABLE ACM_MaintenanceRecommendation (
      EquipID INT,
      RunID VARCHAR(50),
      Timestamp DATETIME,
      Urgency VARCHAR(20),  -- immediate/urgent/planned/monitor
      RiskLevel VARCHAR(20),  -- very_high/high/medium/low
      RUL_Hours FLOAT,
      RUL_p10 FLOAT,
      RUL_p50 FLOAT,
      RUL_p90 FLOAT,
      FailureProbability FLOAT,
      Recommendation TEXT,
      TopCauses NVARCHAR(MAX),  -- JSON array
      Confidence FLOAT
  )
  ```

---

## Section 5: SQL Integration Issues

### 5.1 Missing SQL Tables

Several SQL tables referenced in code but not created:

#### SQLTBL-01: ACM_SensorForecast_TS (NEW)
- **Status:** Table schema not created
- **Referenced By:** FCST-16, RUL-03
- **Priority:** High
- **Schema Required:**
  ```sql
  CREATE TABLE ACM_SensorForecast_TS (
      EquipID INT,
      RunID VARCHAR(50),
      SensorName VARCHAR(100),
      ForecastTimestamp DATETIME,
      ForecastHorizon INT,  -- hours ahead
      PredictedValue FLOAT,
      CI_Lower FLOAT,
      CI_Upper FLOAT,
      MAE FLOAT,
      MAPE FLOAT,
      ModelType VARCHAR(50)
  )
  ```

#### SQLTBL-02: ACM_HealthForecast_TS (NEW)
- **Status:** May exist (check 57_create_forecast_and_rul_tables.sql)
- **Referenced By:** Enhanced forecasting module
- **Priority:** Medium
- **Required Columns:** timestamp, health_forecast, ci_lower, ci_upper, model_type

#### SQLTBL-03: ACM_FailureForecast_TS (NEW)
- **Status:** May exist (check 57_create_forecast_and_rul_tables.sql)
- **Referenced By:** Enhanced forecasting module
- **Priority:** Medium
- **Required Columns:** timestamp, failure_probability, threshold, horizon_hours

---

## Section 6: Summary of All Tasks

### 6.1 By Priority

#### CRITICAL (3 tasks)
1. **FCST-15:** Remove scores.csv dependency (SQL-only mode compatibility)
2. **RUL-01:** Remove health_timeline.csv dependency (SQL-only mode compatibility)
3. **FCST-04:** Add AR(1) coefficient stability checks (prevents numerical crashes)

#### HIGH (5 tasks)
1. **FCST-05:** Improve frequency regex validation
2. **FCST-06:** Make horizon clamping explicit with warnings
3. **FCST-16:** Publish per-sensor forecasts to SQL
4. **RUL-02:** Add probabilistic RUL bands (p10/p50/p90)
5. **MAINT-01:** Implement maintenance recommendation engine (NEW)

#### MEDIUM (6 tasks)
1. **FCST-07:** Correct "divergence" terminology to "mean reversion"
2. **FCST-08:** Improve series selection scoring (add autocorrelation)
3. **FCST-09:** Remove hardcoded fused series override
4. **FCST-10:** Add forecast backtesting and validation
5. **FCST-11:** Add stationarity testing for AR(1)
6. **FCST-17:** Integrate enhanced_forecasting.py module (NEW)
7. **RUL-03:** Fuse sensor hotspots with forecasts

#### LOW (3 tasks)
1. **FCST-12:** Optimize DataFrame fusion performance
2. **FCST-13:** Improve numerical stability for high phi
3. **FCST-14:** Add comprehensive AR(1) documentation

#### SQL TABLES (3 tasks)
1. **SQLTBL-01:** Create ACM_SensorForecast_TS table (NEW)
2. **SQLTBL-02:** Verify ACM_HealthForecast_TS table (NEW)
3. **SQLTBL-03:** Verify ACM_FailureForecast_TS table (NEW)

---

## Section 7: Recommendations

### Immediate Action Items (Sprint 1)

1. **FCST-15 (Critical):** Implement OutputManager artifact cache
   - Enables SQL-only mode for forecasting
   - Blocks: FCST-16, RUL-01
   - Estimated effort: 4-6 hours

2. **RUL-01 (Critical):** Update RUL module to use cached DataFrames
   - Enables SQL-only mode for RUL
   - Depends on: FCST-15
   - Estimated effort: 2-3 hours

3. **FCST-04 (Critical):** Add stability checks to AR(1) fitting
   - Prevents numerical crashes
   - Independent task
   - Estimated effort: 2-3 hours

### Next Priority (Sprint 2)

1. **FCST-16 (High):** Per-sensor forecast publishing
2. **RUL-02 (High):** Probabilistic RUL bands
3. **MAINT-01 (High):** Maintenance recommendation engine
4. **SQLTBL-01 (High):** Create sensor forecast SQL table

### Future Enhancements (Sprint 3+)

1. Medium priority forecasting improvements (FCST-07 through FCST-11)
2. Enhanced forecasting integration (FCST-17)
3. Performance optimizations (FCST-12, FCST-13)
4. Documentation (FCST-14)

---

## Section 8: Testing Plan

### Unit Tests Required

1. **Forecast Module**
   - AR(1) coefficient estimation with edge cases
   - Confidence interval growth validation
   - Frequency parsing with invalid inputs
   - Horizon clamping behavior

2. **RUL Module**
   - Health timeline loading from cache
   - RUL calculation with probabilistic bands
   - Threshold crossing detection
   - Maintenance window calculation

3. **Integration Tests**
   - SQL-only mode end-to-end (forecast + RUL)
   - File mode end-to-end (fallback paths)
   - Dual-mode validation (file vs SQL comparison)

### Validation Criteria

1. **Functional:**
   - All SQL-only mode scenarios work without CSV files
   - Confidence intervals grow with forecast horizon
   - RUL bands capture uncertainty properly
   - Maintenance recommendations are actionable

2. **Performance:**
   - No regression in execution time
   - Cache hits reduce redundant computation
   - SQL writes complete within timeout

3. **Quality:**
   - Forecast accuracy meets backtesting targets
   - No numerical crashes (stability checks working)
   - Logging provides sufficient troubleshooting info

---

## Appendix A: File Locations

- **Forecast Module:** `core/forecast.py` (800+ lines)
- **RUL Module:** `core/rul_estimator.py` (600+ lines)
- **Enhanced Forecasting:** `core/enhanced_forecasting.py` (600+ lines, NOT INTEGRATED)
- **Enhanced Forecasting SQL:** `core/enhanced_forecasting_sql.py` (SQL integration stub)
- **Output Manager:** `core/output_manager.py` (artifact cache implementation needed)
- **Main Pipeline:** `core/acm_main.py` (integration points)
- **SQL Schema:** `scripts/sql/57_create_forecast_and_rul_tables.sql`

## Appendix B: Audit Cross-Reference

| Audit Section | Issue # | Task ID | Status |
|---------------|---------|---------|--------|
| Critical #1 | Constant CI | FCST-01 | ✅ DONE |
| Critical #2 | Residual SD | FCST-03 | ✅ DONE |
| Critical #3 | Warm Start | FCST-02 | ✅ DONE |
| Critical #4 | Divergence | FCST-07 | Planned |
| Critical #5 | Recommendations | MAINT-01 | NEW |
| High #6 | Stability Checks | FCST-04 | Planned |
| High #7 | Freq Regex | FCST-05 | Planned |
| High #8 | Horizon Clamp | FCST-06 | Planned |
| Medium #9 | Series Scoring | FCST-08 | Planned |
| Medium #10 | Fused Override | FCST-09 | Planned |
| Medium #11 | Validation | FCST-10 | Planned |
| Low #12 | Performance | FCST-12 | Planned |
| Low #13 | Stability φ | FCST-13 | Planned |
| Low #14 | Docs | FCST-14 | Planned |

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-15  
**Author:** Copilot Analysis Agent  
**Status:** COMPLETE - Ready for Task Backlog Update

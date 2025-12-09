# ACM Dashboard & System Comprehensive Analysis

## Task Tracker (Active)
| Status  | Priority | Task | Notes |
|---------|----------|------|-------|
| ✅ Complete | P0 | Wire unified `ForecastEngine`/RUL path in `acm_main.py` (replace legacy forecasting call) | **FIXED**: ForecastEngine already wired correctly at lines 3886-3908 and 3948-3970. Verified on 2025-12-08. |
| ✅ Complete | P0 | Whitelist forecast/RUL tables in `OutputManager` | **FIXED**: All forecast/RUL tables already in ALLOWED_TABLES (lines 66-70 of output_manager.py). Verified on 2025-12-08. |
| ✅ Complete | P0 | Fix health timeline SQL write in fallback path | **FIXED**: Added sql_table="ACM_HealthTimeline" to fallback write (line 3855) with proper schema. Prevents missing rows when analytics generation fails. Fixed on 2025-12-08. |
| ✅ Complete | P0 | Harden imports (river_models) and regime labeling fallback | **FIXED**: Wrapped river_models import in try/except (lines 26-29, 39-42). Added None check before RiverTAD usage (line 2253). Fixed on 2025-12-08. |
| ✅ Complete | P0 | RUL engine errors (`LearningState.ar1`, `Series.total_seconds`) | **FIXED**: All RUL columns (RUL_Hours, Confidence, FailureTime, NumSimulations, Method) now correctly populated. Verified via batch run on 2025-12-08. |
| ✅ Complete | P1 | Fix anomaly timeline query | **FIXED**: ACM_Anomaly_Events StartTime/EndTime column mapping corrected in output_manager.py and acm_main.py. Verified 25 events with proper timestamps on 2025-12-08. |
| ✅ Complete | P2 | Fix SQL config loader (`'bool' object does not support item assignment`) | **FIXED**: Added type check in _load_from_sql (line 211) to handle non-dict values when building nested paths. Fixed on 2025-12-08. |
| ✅ Complete | P2 | Fix PCA fitted check warning | **FIXED**: Changed from warning to silent return when pca is None (expected with <2 samples). Line 2216 in output_manager.py. Fixed on 2025-12-08. |
| ✅ Complete | P1 | Fix Data Quality Assessment dashboard - missing metrics | **FIXED**: ACM_DataQuality SQL insert now writes all quality columns (longest_gap, flatline_span, sampling_secs, std, timestamps). Line 1368 in acm_main.py. Fixed on 2025-12-08. |
| ✅ Complete | P1 | Fix drift detection panel | **FIXED**: ACM_DriftSeries table exists with 10,893 rows. Dashboard query fixed to use ORDER BY ASC (was DESC) and added time range filter. Fixed on 2025-12-08. |
| ✅ Complete | P1 | Implement sensor-level multivariate time series forecasting | **FIXED**: Added `_generate_sensor_forecasts()` method to forecast_engine.py. Generates 7-day forecasts for top 10 sensors using Holt's exponential smoothing with trend detection and confidence intervals. Writes to ACM_SensorForecast table. Fixed on 2025-12-08. |
| Pending | P2 | Fix AdaptiveConfigManager type handling | Clean up warnings in adaptive config manager. |

**Date**: December 8, 2025  
**Analysis Type**: Full System Health Check - Batch Mode Run (10 batches) + Code Integration Audit  
**Equipment**: FD_FAN (EquipID=1)  
**Branch**: feature/forecasting-refactor-v10

---

## EXECUTIVE SUMMARY

**System Status**: ⚠️ **PARTIAL OPERATION** - Code wiring issues + Dashboard data bugs

The ACM system shows **mixed operational status** across two critical layers:

### Layer 1: Code Integration (Critical Blockers)
The v10.0.0 forecasting refactor has **NOT been properly integrated** into `acm_main.py`. The old forecasting API is still being called, but the new implementation uses different function signatures and SQL-only operation. This means:
- ❌ New forecasting engine never executes
- ❌ RUL predictions hidden behind forecasting (never triggered)
- ❌ Forecast/RUL tables blocked by OutputManager whitelist
- ❌ Health timeline SQL writes missing in fallback path

### Layer 2: Data/Dashboard (When Pipeline Does Run)
When the pipeline executes (via older code paths or partial integration), it processes data successfully BUT:
- ✅ Core detection pipeline working (76 episodes across 10 batches)
- ✅ 9,087 health timeline points generated
- ❌ Dashboard shows **0.0 hours RUL** (column population bug)
- ❌ Confidence scores always NULL
- ❌ Sensor forecasts not implemented (0 rows)
- ❌ Anomaly timeline broken (query format issue)

**Critical Finding**: The system has **two distinct failure modes**:
1. **Integration failure**: New forecasting code never executes due to import/API mismatches
2. **Display failure**: When older paths work, RUL shows 0.0h despite correct internal calculations (P50=11.5h)

---

## CODE INTEGRATION ISSUES (BLOCKING NEW FORECASTING)

### Critical Blockers Preventing v10 Forecasting Engine

#### **Blocker #1: Wrong Forecasting Import in acm_main.py**
**Severity**: CRITICAL - Prevents entire forecasting refactor from executing

**Current Code** (WRONG):
```python
# Line ~50 in acm_main.py
from . import correlation, outliers, forecast, river_models  # Old imports
...
from core import forecast  # Module doesn't exist

# Line ~2800
forecast_ctx = {
    "run_dir": run_dir,
    "plots_dir": charts_dir,
    "tables_dir": tables_dir,
    "config": cfg,
    "enable_report": True,
}
forecast_result = forecast.run(forecast_ctx)  # Old API - doesn't exist
```

**Problem Analysis**:
- There is **no `core/forecast.py`** module (renamed to `forecasting.py`)
- Import will fail OR pull wrong/stale module
- Even if import works, `forecast.run(ctx)` function **doesn't exist** in new code
- New API is `forecasting.run_and_persist_enhanced_forecasting(...)`

**Required Fix**:
```python
# CORRECT import
from core import forecasting  # or: from . import forecasting

# CORRECT call (replace forecast.run block)
if SQL_MODE and sql_client and equip_id and run_id:
    forecasting.run_and_persist_enhanced_forecasting(
        sql_client=sql_client,
        equip_id=equip_id,
        run_id=run_id,
        config=cfg,
        output_manager=output_manager,
        tables_dir=tables_dir,
        equip=equip,
        current_batch_time=win_end,
        sensor_data=score_numeric,  # for physical sensor forecasting
    )
```

**Impact**: Until fixed, **new forecasting engine never executes**, RUL never runs, dashboard shows stale/zero data.

---

#### **Blocker #2: RUL Engine Not Imported or Called**
**Severity**: CRITICAL - RUL predictions never generated

**Current State**:
- `rul_engine.py` is **NOT imported** in `acm_main.py`
- RUL only accessible through `forecasting.py`:
  ```python
  # Inside forecasting.py
  from core import rul_engine
  ...
  return rul_engine.run_rul(...)
  ```

**Problem**: Since forecasting is never called (Blocker #1), **RUL is never called either**.

**Dependency Chain**:
```
acm_main.py → (should call) → forecasting.run_and_persist_enhanced_forecasting()
                              ↓
                              forecasting.py → rul_engine.run_rul()
                                               ↓
                                               Generates ACM_RUL tables
```

**Impact**: RUL dashboard shows 0.0 hours because **RUL engine never executes**.

---

#### **Blocker #3: OutputManager Missing Forecast/RUL Tables in Whitelist**
**Severity**: CRITICAL - SQL writes will fail even if forecasting runs

**Current ALLOWED_TABLES** (core/output_manager.py):
```python
ALLOWED_TABLES = {
    'ACM_Scores_Wide', 'ACM_Episodes',
    'ACM_HealthTimeline', 'ACM_RegimeTimeline',
    'ACM_SensorHotspots', 'ACM_SensorHotspotTimeline',
    # ... but NO forecast/RUL tables
}
```

**Missing Tables**:
- `ACM_HealthForecast`
- `ACM_FailureForecast`
- `ACM_DetectorForecast_TS`
- `ACM_SensorForecast`
- `ACM_HealthForecast_TS`
- `ACM_FailureForecast_TS`
- `ACM_RUL`
- `ACM_RUL_TS`
- `ACM_RUL_Summary`
- `ACM_RUL_Attribution`
- `ACM_MaintenanceRecommendation`
- `ACM_RUL_LearningState`

**Protection Code** (_bulk_insert_sql):
```python
if table_name not in ALLOWED_TABLES:
    raise ValueError(f"Invalid table name: {table_name}")
```

**Impact**: Even if forecasting runs, **any attempt to persist forecast/RUL tables will throw exception**.

**Required Fix**:
```python
# Add to ALLOWED_TABLES in output_manager.py
ALLOWED_TABLES = {
    # ... existing tables ...
    'ACM_HealthForecast',
    'ACM_FailureForecast', 
    'ACM_DetectorForecast_TS',
    'ACM_SensorForecast',
    'ACM_HealthForecast_TS',
    'ACM_FailureForecast_TS',
    'ACM_RUL',
    'ACM_RUL_TS',
    'ACM_RUL_Summary',
    'ACM_RUL_Attribution',
    'ACM_MaintenanceRecommendation',
    'ACM_RUL_LearningState',
}
```

---

#### **Blocker #4: Health Timeline SQL Missing in Fallback Path**
**Severity**: HIGH - RUL fails when analytics fallback triggers

**Current Fallback Code** (acm_main.py ~line 2900):
```python
# Fallback to basic tables
if 'fused' in frame.columns:
    health_df = pd.DataFrame({
        'timestamp': frame.index.strftime('%Y-%m-%d %H:%M:%S'),
        'fused_z': frame['fused'],
        'health_index': 100.0 / (1.0 + frame['fused'] ** 2)
    })
    output_manager.write_dataframe(health_df, tables_dir / "health_timeline.csv")
    # ❌ NO SQL TABLE NAME - only writes CSV
```

**RUL Dependency** (rul_engine.py):
```python
health_df, data_quality = load_health_timeline(
    sql_client=sql_client,  # SQL-ONLY, no CSV fallback
    equip_id=equip_id,
    run_id=run_id,
    output_manager=output_manager,
    cfg=cfg,
)

if health_df is None or health_df.empty:
    raise RuntimeError("Health timeline unavailable")
```

**Problem**: 
- Fallback writes **only CSV**, not `ACM_HealthTimeline` table
- RUL `load_health_timeline` is **SQL-only** (no CSV fallback)
- Result: RUL aborts with "Health timeline unavailable"

**Required Fix**:
```python
# Add SQL table name to fallback path
output_manager.write_dataframe(
    health_df,
    tables_dir / "health_timeline.csv",
    sql_table="ACM_HealthTimeline",  # ✅ ADD THIS
    add_created_at=True,
)
```

---

#### **Blocker #5: river_models Import May Fail**
**Severity**: MEDIUM - May cause import-time crash

**Current Code**:
```python
from . import correlation, outliers, forecast, river_models
from core import river_models
```

**Problem**: 
- No `river_models.py` file visible in codebase
- If missing, **entire acm_main.py import fails** before anything runs
- Blocks detection, regimes, forecasting, RUL - everything

**Required Fix** (choose one):
1. Add stub `core/river_models.py` with no-op `RiverTAD` class
2. Guard import with try/except and config check
3. Comment out import + usage until ready

---

#### **Blocker #6: Regime Labeling Can Hard-Fail Pipeline**
**Severity**: MEDIUM - Can kill entire run before forecasting

**Problem Code** (regimes.py):
```python
def label(...):
    if basis_train is not None and basis_score is not None:
        # do regime labeling
        return out
    
    if bool(_cfg_get(cfg, "regimes.allow_legacy_label", False)):
        return _legacy_label(...)
    
    # ❌ HARD FAILURE - not caught in acm_main
    raise RuntimeError("[REGIME] Regime model unavailable and legacy path disabled")
```

**Impact**:
- If `basis_train/basis_score` becomes None (upstream error, column issues)
- AND `regimes.allow_legacy_label = False` (default)
- Result: **RuntimeError kills entire run** before forecasting/RUL

**Required Fix** (choose one):
1. Set `regimes.allow_legacy_label = True` in config as safety net
2. Add try/except in acm_main around `regimes.label()` with graceful fallback

---

#### **Blocker #7: Unused New Modules (Not Blocking, But Indicates Incomplete Integration)** ℹ️
**Severity**: INFO - Shows v10 design not active

**Modules Implemented But Never Used**:
- `health_tracker.py` (HealthTimeline class with SQL loading, quality checks)
- `state_manager.py` (StateManager with forecast state, adaptive config)

**Status**: 
- Nothing imports these modules
- They don't block current operation
- They indicate **v10 refactor incomplete** - new design patterns not activated

---

### Integration Fix Priority (Code Layer)

**P0 - IMMEDIATE (Required for forecasting to work)**:
1. ✅ Fix forecasting import + API call (Blocker #1)
2. ✅ Add forecast/RUL tables to ALLOWED_TABLES (Blocker #3)
3. ✅ Fix health timeline SQL in fallback (Blocker #4)

**P1 - HIGH (Stability)**:
4. ✅ Handle river_models import gracefully (Blocker #5)
5. ✅ Add regime labeling safety net (Blocker #6)

**P2 - FUTURE (Design completion)**:
6. Wire in health_tracker.py and state_manager.py

---

## DATA VALIDATION RESULTS

### Successfully Operating Components
- **Health Timeline**: 9,087 data points (2023-10-18 to 2025-09-14) across 11 runs
- **Health Forecasts**: 3,360 records generated
- **Failure Forecasts**: 3,360 records generated  
- **RUL Predictions**: 10 predictions with valid confidence intervals
- **Anomaly Events**: 76 events detected and tracked
- **Episode Diagnostics**: 76 episodes analyzed
- **Episode Metrics**: Aggregate statistics calculated
- **PCA Models**: Successfully fitted and persisted (5 components)
- **PCA Loadings**: 450 rows of component loadings
- **Detector Scores**: Successfully written to ACM_Scores_Wide and ACM_Scores_Long

### Problematic Components
- **RUL Display**: Shows 0.0 hours instead of P50_Median values
- **Sensor Forecasts**: Table empty (0 rows) - feature not implemented
- **Drift Metrics**: Table does not exist - feature disabled in code
- **Anomaly Timeline**: Query returning no data despite 76 events existing
- **Confidence Scores**: All RUL predictions have NULL confidence values

---

## CRITICAL ISSUES LIST (10 TOTAL)

### **Issue #1: RUL_Hours Column Showing Zero (CRITICAL ⚠️)**
**Severity**: 🔴 **P0 - IMMEDIATE FIX REQUIRED**  
**Impact**: FALSE POSITIVE critical alerts, unusable RUL predictions

**Problem Description**:
- Dashboard "Remaining Useful Life - RUL Summary" panel shows **0.0 hour**
- "Maintenance Recommendations" table shows three entries all with **RUL=0 hours**
- All entries marked as **CRITICAL** status
- However, confidence intervals show valid predictions: P10=1.5h, P50=11.5h, P90=170.2h

**Evidence from SQL**:
```sql
Method      RUL   P10   P50   P90    Conf  CreatedAt
Multipath   0.0   1.5   11.5  170.2  NULL  2025-12-08 12:47:00
Multipath   0.0   0.5   1.0   3.0    NULL  2025-12-08 12:46:32
Multipath   0.0   0.0   0.0   0.0    NULL  2025-12-08 12:46:03
```

**Root Cause Analysis**:
```python
# Current bug in core/forecasting.py:
rul_summary_df = pd.DataFrame([{
    'RUL_Hours': 0.0,  # ❌ HARDCODED TO ZERO
    'P50_Median': p50_hours,  # ✅ CORRECTLY CALCULATED (11.5h)
    'P10_LowerBound': p10_hours,
    'P90_UpperBound': p90_hours,
    ...
}])
```

**Why This Happens**:
1. Forecasting engine correctly calculates RUL median as 11.5 hours
2. DataFrame construction hardcodes `RUL_Hours` column to 0.0
3. Grafana dashboard queries `RUL_Hours` column (not `P50_Median`)
4. Result: Dashboard shows 0.0 → triggers CRITICAL status → FALSE ALARM

**Fix Required**:
```python
# CORRECT FIX:
rul_summary_df = pd.DataFrame([{
    'RUL_Hours': p50_hours,  # ✅ USE MEDIAN PREDICTION
    'P50_Median': p50_hours,
    'P10_LowerBound': p10_hours,
    'P90_UpperBound': p90_hours,
    ...
}])
```

**Location**: `core/forecasting.py` - RUL summary DataFrame construction (search for "RUL_Hours")

**Validation**: After fix, query should show:
```sql
RUL_Hours = P50_Median (e.g., 11.5 instead of 0.0)
```

---

### **Issue #2: Maintenance Recommendations Showing Multiple Zero RULs (CRITICAL ⚠️)**
**Severity**: 🔴 **P0 - IMMEDIATE FIX REQUIRED**  
**Impact**: Maintenance planning completely unusable, false critical alerts

**Problem Description**:
Dashboard "Maintenance Recommendations - Scheduled Windows" shows:
- Row 1: RUL=0h, P10=0.5, P90=5.10, Confidence=CRITICAL
- Row 2: RUL=0h, P10=1.0, P90=37.5, Confidence=CRITICAL  
- Row 3: RUL=0h, P10=1.5, P90=170, Confidence=CRITICAL

All three marked CRITICAL despite vastly different confidence intervals (P90 ranges from 5h to 170h).

**Root Cause**: Same as Issue #1 - `RUL_Hours` column hardcoded to 0.0

**Cascading Impact**:
- Operations team cannot trust RUL predictions
- Maintenance windows scheduled incorrectly
- Risk of unnecessary emergency shutdowns
- Loss of predictive maintenance value

**Fix**: Same as Issue #1 - populate `RUL_Hours` with `P50_Median` value

---

### **Issue #3: Model Drift Detection Panel Error (HIGH 🟡)**
**Severity**: 🟡 **P1 - HIGH PRIORITY**  
**Impact**: One dashboard panel broken, misleading error message

**Problem Description**:
Panel "Model Drift Detection - Concept Drift Evolution" displays error:
```
Failed to convert long to wide series when converting from 
dataframe: unable to process the data because it is not sorted in 
ascending order by time, please updated your query to sort the 
data by time if possible
```

**Evidence from SQL**:
```sql
Msg 208, Level 16, State 1
Invalid object name 'ACM_DriftMetrics'.
```

**Root Cause Analysis**:
1. Code has drift detection **explicitly disabled**:
   ```python
   # Line 694 in core/forecasting.py:
   # Drift check disabled - ACM_DriftMetrics table not yet implemented
   ```
2. SQL table `ACM_DriftMetrics` was **never created** in schema
3. Grafana dashboard still has query attempting to read from this table
4. Error message is **misleading** (table doesn't exist, not a sorting issue)

**Why Misleading**: Grafana can't sort data that doesn't exist, so it throws a generic "not sorted" error instead of "table not found"

**Fix Options**:

**Option A (Quick - 1 minute)**:
- Delete "Model Drift Detection - Concept Drift Evolution" panel from dashboard
- Update dashboard description to note drift detection not yet implemented

**Option B (Proper - 2-3 hours)**:
1. Create `ACM_DriftMetrics` SQL table with schema:
   ```sql
   CREATE TABLE ACM_DriftMetrics (
       EquipID INT NOT NULL,
       RunID UNIQUEIDENTIFIER NOT NULL,
       Timestamp DATETIME2 NOT NULL,
       DriftScore FLOAT,
       DriftType NVARCHAR(50),
       CreatedAt DATETIME2 DEFAULT GETDATE()
   );
   ```
2. Implement drift detection in `core/drift.py`
3. Enable drift check in `core/forecasting.py` line 694
4. Write drift metrics to SQL table

**Recommendation**: Option A for now (remove panel), Option B for future release

---

### **Issue #4: Anomaly Events Detection Timeline - No Data (HIGH 🟡)**
**Severity**: 🟡 **P1 - HIGH PRIORITY**  
**Impact**: Critical anomaly visualization broken

**Problem Description**:
Dashboard panel "Anomaly Events - Detection Timeline" shows **"No data"** despite having 76 events in database.

**Evidence**:
```sql
SELECT COUNT(*) FROM ACM_Anomaly_Events WHERE EquipID=1;
-- Result: 76 rows
```

**Root Cause Analysis**:
Grafana query likely has one of these issues:
1. **Time column format issue**: Using `FORMAT()` which returns VARCHAR instead of DATETIME
2. **Column name mismatch**: Query looking for wrong timestamp column name
3. **Time range filter**: Dashboard time range doesn't match data range
4. **Sort order**: Query sorting DESC instead of ASC (required for time series)

**Critical Reminder from ACM Guidelines**:
> For Grafana time_series format, MUST use DATETIME not VARCHAR.
> Use DATEADD() for rounding, never FORMAT() which returns string.

**Current Query (likely WRONG)**:
```sql
-- ❌ WRONG - Returns VARCHAR
SELECT FORMAT(StartTime, 'yyyy-MM-dd HH:00') AS time,
       COUNT(*) AS value
FROM ACM_Anomaly_Events
WHERE EquipID = $equipment
GROUP BY FORMAT(StartTime, 'yyyy-MM-dd HH:00')
ORDER BY time DESC  -- Also wrong direction
```

**Correct Query**:
```sql
-- ✅ CORRECT - Returns DATETIME
SELECT DATEADD(HOUR, DATEDIFF(HOUR, 0, StartTime), 0) AS time,
       COUNT(*) AS value,
       'Events' AS metric
FROM ACM_Anomaly_Events 
WHERE EquipID = $equipment 
  AND StartTime BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY DATEADD(HOUR, DATEDIFF(HOUR, 0, StartTime), 0)
ORDER BY time ASC  -- ✅ MUST be ASC for time series
```

**Validation Steps**:
1. Check current dashboard query in Grafana panel edit mode
2. Verify time column returns DATETIME type (not VARCHAR)
3. Confirm ORDER BY uses ASC (not DESC)
4. Test query directly in SQL Management Studio first

---

### **Issue #5: Top 5 Physical Sensor Forecasts - No Data (MEDIUM 🟠)**
**Severity**: 🟠 **P2 - MEDIUM PRIORITY**  
**Impact**: Sensor-level forecasting visualization missing

**Problem Description**:
Dashboard panel "Top 5 Physical Sensor Forecasts (7-Day Trend)" shows **"No data"**

**Evidence**:
```sql
SELECT COUNT(*) FROM ACM_SensorForecast WHERE EquipID=1;
-- Result: 0 rows
```

**Root Cause Analysis**:
1. ✅ Health forecasting engine working (3,360 health forecasts written)
2. ✅ Failure forecasting engine working (3,360 failure forecasts written)
3. ❌ **Sensor-level forecasting NOT IMPLEMENTED**
4. ❌ `ACM_SensorForecast` table remains empty

**Why This Matters**:
- Cannot identify which specific sensors driving degradation trends
- Missing actionable maintenance insights (e.g., "replace bearing temp sensor")
- Forecasting limited to aggregate health only

**Implementation Status**: Feature incomplete in v10.0.0

**Fix Required**:
```python
# Location: core/forecasting.py after health/failure writes
# Add new method:

def _generate_sensor_forecasts(self, sensor_data, forecast_horizon=168):
    """
    Generate sensor-level trend forecasts.
    
    Args:
        sensor_data: DataFrame with sensor readings
        forecast_horizon: Hours to forecast (default 168 = 7 days)
        
    Returns:
        DataFrame with columns: Timestamp, SensorName, ForecastValue, 
                                CI_Lower, CI_Upper, TrendDirection, Method
    """
    sensor_forecasts = []
    
    for sensor in sensor_data.columns:
        # Fit exponential smoothing or ARIMA per sensor
        model = ExponentialSmoothing(sensor_data[sensor], ...)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)
        
        # Build forecast DataFrame
        for i, (ts, val) in enumerate(forecast.items()):
            sensor_forecasts.append({
                'EquipID': self.equip_id,
                'RunID': self.run_id,
                'Timestamp': ts,
                'SensorName': sensor,
                'ForecastValue': val,
                'Method': 'ExponentialSmoothing',
                'CreatedAt': pd.Timestamp.now()
            })
    
    return pd.DataFrame(sensor_forecasts)

# Then call in main forecast flow:
sensor_forecast_df = self._generate_sensor_forecasts(sensor_data)
output_manager.write_dataframe(
    sensor_forecast_df,
    sql_table="ACM_SensorForecast",
    add_created_at=True
)
```

**Estimated Effort**: 2-3 hours development + testing

---

### **Issue #6: Sensor Forecast Summary - Top Changing Sensors - No Data (MEDIUM 🟠)**
**Severity**: 🟠 **P2 - MEDIUM PRIORITY**  
**Impact**: Related to Issue #5

**Problem Description**:
Panel "Sensor Forecast Summary - Top Changing Sensors" shows **"No data"**

**Root Cause**: Same as Issue #5 - `ACM_SensorForecast` table empty (0 rows)

**Fix**: Same as Issue #5 - implement sensor-level forecasting

---

### **Issue #7: PCA Detector Not Fitted Warning (LOW 🟢)**
**Severity**: 🟢 **P3 - LOW PRIORITY**  
**Impact**: Minor - core PCA functionality working, edge metric skipped

**Problem Description**:
Logs show repeated warning:
```
[OUTPUT] PCA detector not fitted, skipping metrics write
```

Appears in 3 out of 10 batch runs.

**Evidence**:
```
2025-12-08 07:17:00 WARNING [OUTPUT] PCA detector not fitted, skipping metrics write
2025-12-08 07:16:32 WARNING [OUTPUT] PCA detector not fitted, skipping metrics write
2025-12-08 07:16:03 WARNING [OUTPUT] PCA detector not fitted, skipping metrics write
```

**However, PCA IS Working**:
```
2025-12-08 07:17:00 INFO [OUTPUT] SQL insert to ACM_PCA_Loadings: 450 rows
2025-12-08 07:17:00 INFO [OUTPUT] SQL insert to ACM_PCA_Models: 1 rows
```

**Root Cause Analysis**:
1. ✅ PCA detector successfully fitted (5 components, 450 loadings)
2. ✅ PCA model metadata written to ACM_PCA_Models
3. ✅ PCA loadings written to ACM_PCA_Loadings
4. ❌ OutputManager checking wrong fitted flag when attempting edge metrics write
5. ❌ Likely checking `pca_detector._is_fitted` which may not exist

**Fix Required**:
```python
# Location: core/output_manager.py
# Current (likely):
if pca_detector._is_fitted:
    self._write_pca_metrics(...)

# Fixed:
if hasattr(pca_detector, '_is_fitted') and pca_detector._is_fitted:
    self._write_pca_metrics(...)
elif hasattr(pca_detector, 'pca') and pca_detector.pca is not None:
    # Alternative check - detector has fitted PCA object
    self._write_pca_metrics(...)
```

**Impact**: Minimal - main PCA data persisting correctly, only edge metric skipped

---

### **Issue #8: AdaptiveConfigManager Config Type Mismatch (LOW 🟢)**
**Severity**: 🟢 **P3 - LOW PRIORITY**  
**Impact**: Non-critical - forecasting works with defaults

**Problem Description**:
Logs show repeated warnings:
```
[AdaptiveConfigManager] Failed to get config 'auto_tune_data_threshold': 
argument of type 'float' is not iterable

[AdaptiveConfigManager] Failed to load all configs: 
argument of type 'float' is not iterable
```

Appears in **every** forecast run.

**Root Cause Analysis**:
```python
# AdaptiveConfigManager expects nested dict:
{
    "auto_tune": {
        "data_threshold": 50000,
        "quality_threshold": 0.7
    }
}

# But config provides flat structure:
{
    "auto_tune_data_threshold": 50000.0  # Float, not dict
}

# Code tries:
threshold = cfg.get("auto_tune", {}).get("data_threshold")
# Fails because cfg["auto_tune"] = 50000.0 (float, not dict)
# Python tries to iterate float, raises TypeError
```

**Why It Still Works**:
- Exception caught gracefully
- Falls back to hardcoded defaults
- Forecasting continues normally

**Fix Required**:
```python
# Location: core/forecasting.py (AdaptiveConfigManager or forecast init)

# Handle both dict and flat config structures:
auto_tune_cfg = cfg.get("auto_tune", {})

if isinstance(auto_tune_cfg, dict):
    # Nested structure (preferred)
    data_threshold = auto_tune_cfg.get("data_threshold", 50000)
    quality_threshold = auto_tune_cfg.get("quality_threshold", 0.7)
else:
    # Flat structure (legacy fallback)
    data_threshold = cfg.get("auto_tune_data_threshold", 50000)
    quality_threshold = cfg.get("auto_tune_quality_threshold", 0.7)
```

**Impact**: Cosmetic - removes warning messages, no functional change

---

### **Issue #9: Gappy Data Warnings (EXPECTED ✅)**
**Severity**: ✅ **NOT AN ISSUE - EXPECTED BEHAVIOR**  
**Impact**: None - historical replay mode working as designed

**Problem Description**:
Logs show warnings:
```
[ForecastEngine] GAPPY data detected - proceeding with available data 
(historical replay mode)

[ForecastEngine] Data quality issue: Max gap 1404.5 hours 
(threshold 720.0 hours)
```

**Analysis**: **THIS IS EXPECTED AND CORRECT**

**Why This Happens**:
- Running batch mode processing **historical data from 2023-2025**
- Data has natural gaps between equipment operating periods
- Max gap: 1404.5 hours = **58.5 days** (equipment downtime/maintenance)
- Threshold: 720 hours = **30 days**

**Forecasting Engine Response**:
1. ✅ Detects gap exceeds threshold
2. ✅ Switches to "historical replay mode"
3. ✅ Proceeds with available data (doesn't fail)
4. ✅ Successfully generates forecasts (3,360 health + 3,360 failure)

**Conclusion**: Working as designed. Warnings are informational, not errors.

**No Fix Required** - this is proper gap handling for batch/historical processing

---

### **Issue #10: RUL Confidence Column NULL (MEDIUM 🟠)**
**Severity**: 🟠 **P2 - MEDIUM PRIORITY**  
**Impact**: Cannot assess prediction reliability

**Problem Description**:
All RUL predictions have `Confidence = NULL`:
```sql
Method      RUL   P50   P90   Conf
Multipath   0.0   11.5  170.2 NULL  ← All NULL
Multipath   0.0   1.0   3.0   NULL
Multipath   0.0   0.0   0.0   NULL
```

**Evidence**: 10 most recent RUL records all have `Confidence = NULL`

**Root Cause Analysis**:
1. ✅ `Confidence` column exists in `ACM_RUL` table schema
2. ✅ Forecasting engine calculates confidence intervals (P10, P50, P90)
3. ❌ DataFrame construction doesn't populate `Confidence` column
4. ❌ Column inserted as NULL in all rows

**Why This Matters**:
- Cannot distinguish reliable predictions from uncertain ones
- Operations cannot prioritize maintenance based on confidence
- Risk assessment incomplete without confidence scores

**Confidence Score Calculation Options**:

**Option 1: Interval Width Normalized**:
```python
# Narrower interval = higher confidence
interval_width = p90_hours - p10_hours
max_expected_width = 200  # hours
confidence = 1.0 - min(interval_width / max_expected_width, 1.0)
```

**Option 2: Simulation Variance Based**:
```python
# If using Monte Carlo simulations
confidence = 1.0 - (std_rul / mean_rul)  # Lower variance = higher confidence
```

**Option 3: Model Fit Quality**:
```python
# Based on forecasting model R² or RMSE
confidence = model_r_squared  # or 1.0 - normalized_rmse
```

**Fix Required**:
```python
# Location: core/forecasting.py, RUL summary DataFrame

# Calculate confidence score
interval_width = p90_hours - p10_hours
confidence = max(0.0, min(1.0, 1.0 - (interval_width / 200.0)))

rul_summary_df = pd.DataFrame([{
    'RUL_Hours': p50_hours,  # Also fix Issue #1
    'P50_Median': p50_hours,
    'P10_LowerBound': p10_hours,
    'P90_UpperBound': p90_hours,
    'Confidence': confidence,  # ✅ ADD THIS
    ...
}])
```

**Validation**: After fix, query should show:
```sql
Confidence values between 0.0 and 1.0 (e.g., 0.75, 0.92, etc.)
```

---

## 📊 ADDITIONAL OBSERVATIONS

### Warnings - Expected & Non-Critical

#### **Regime Quality Warnings** ✅ Expected
```
[AUTO-TUNE] Quality degradation detected: Anomaly rate too high, 
Silhouette score too low

[REGIME] Clustering quality below threshold; per-regime thresholds disabled
```

**Analysis**: EXPECTED for batch/historical processing
- Historical data contains multiple operating regimes mixed together
- Batch mode doesn't maintain temporal continuity needed for regime learning
- Silhouette scores naturally lower in historical replay
- Per-regime thresholds correctly disabled as safety measure

**Status**: Working as designed, not an error

#### **Timestamp Column Fallback** ✅ Expected
```
[DATA] Timestamp column '' not found in SQL historian results; 
falling back to 'EntryDateTime'
```

**Analysis**: Graceful fallback working correctly
- Primary timestamp column not specified or found
- System falls back to `EntryDateTime` (historian standard column)
- Data loads successfully (2,822 rows retrieved)

**Status**: Safe fallback, consider configuring primary column name

#### **Training Data Warning** ✅ Expected (Batch Mode)
```
[DATA] Training data (0 rows) is below recommended minimum (200 rows)
```

**Analysis**: Expected in batch mode with adaptive baseline
- Batch mode uses adaptive baseline splitting score data 50/50
- Train portion comes from first half of current batch
- When batch < 400 rows, train < 200 rows threshold
- Models load from cached SQL ModelRegistry successfully

**Status**: Batch mode behavior, models loading from cache correctly

#### **Batch Window Overlap Warning** ✅ Expected (Historical)
```
[DATA] Batch window starts before or overlaps baseline end: 
batch_start=2025-05-11 00:00:00, baseline_end=2025-06-12 05:30:00
```

**Analysis**: Historical data chronology
- Processing batches in historical sequence
- Natural overlaps when data gaps exist
- Adaptive baseline correctly handles overlap

**Status**: Historical processing artifact, system handling properly

---

## ✅ WHAT'S ACTUALLY WORKING (POSITIVE VALIDATION)

### Core Pipeline - Fully Operational
- ✅ **10 batch runs completed successfully** (scripts/sql_batch_runner.py)
- ✅ **Data ingestion**: 2,822 rows loaded per batch from SQL historian
- ✅ **Feature engineering**: 90 features generated (9 dropped low-variance)
- ✅ **Detector ensemble**: All 7 detectors operating (AR1, PCA-SPE, PCA-T2, Mhal, IForest, GMM, OMR)

### Anomaly Detection - Working
- ✅ **Detector fusion**: Fused scores calculated and written
- ✅ **Episode detection**: 76 episodes identified across historical data
- ✅ **Episode diagnostics**: Duration, severity, dominant sensors analyzed
- ✅ **Episode metrics**: Aggregate statistics (avg duration, max duration, rate per day)
- ✅ **Anomaly events**: 76 events written to ACM_Anomaly_Events
- ✅ **Culprit attribution**: 14 enhanced culprit records per run to ACM_EpisodeCulprits

### Forecasting Engine - Core Working
- ✅ **Health forecasting**: 3,360 7-day outlook records generated
- ✅ **Failure forecasting**: 3,360 failure probability records
- ✅ **RUL calculation**: Multipath algorithm generating valid predictions
  - P10 (lower bound): 1.5 hours
  - P50 (median): 11.5 hours  
  - P90 (upper bound): 170.2 hours
- ✅ **Top sensor attribution**: Correctly identifying contributors:
  1. DEMO.SIM.06GP34_1FD Fan Outlet Pressure (19.4%)
  2. DEMO.SIM.06G31_1FD Fan Damper Position (11.4%)
  3. DEMO.SIM.06T34_1FD Fan Outlet Termperature (10.8%)
- ✅ **State persistence**: ForecastingState saved and restored across runs
- ✅ **Warm-start models**: Successfully loading previous state for continuity

### Model Persistence - Working
- ✅ **SQL ModelRegistry v3**: Successfully saving/loading models
- ✅ **PCA models**: 1 model record + 450 loadings per run
- ✅ **Model metadata**: NComponents, VarExplained, ScalingSpec persisted
- ✅ **Adaptive caching**: Models reused when quality sufficient

### Data Quality Tracking - Working
- ✅ **ACM_DataQuality**: 9 sensor quality records per run
- ✅ **Cadence tracking**: 100% cadence OK across batches
- ✅ **Missing data handling**: Imputation with train medians
- ✅ **Feature validation**: Low-variance features logged to ACM_FeatureDropLog

### Regime Detection - Operational (Quality Warnings Expected)
- ✅ **Auto-k selection**: K=2 regimes identified (silhouette=0.514)
- ✅ **Regime labeling**: Score data labeled with regime assignments
- ✅ **Regime state**: Saved to ACM_RegimeState for persistence
- ✅ **Quality checks**: Correctly disabling per-regime thresholds when quality low

### SQL Integration - Robust
- ✅ **Batched writes**: 2,154 rows (ACM_Scores_Long), 450 rows (PCA_Loadings)
- ✅ **Transaction management**: Proper commit/rollback handling
- ✅ **Bulk insert optimization**: Debug logging shows proper column alignment
- ✅ **Error handling**: Graceful handling of PK constraint violations (duplicate RunIDs)

### Run Metadata - Complete
- ✅ **ACM_Runs**: Run status, outcome (OK/FAIL), samples processed
- ✅ **ACM_Run_Stats**: Window times, cadence %, anomaly count, drift P95
- ✅ **ACM_RunLogs**: Comprehensive logging (INFO/WARNING/ERROR) to SQL
- ✅ **Timing metrics**: Detailed performance profiling per pipeline stage

---

## 🎯 PRIORITIZED ACTION PLAN

### 🔴 **PHASE 0 - CODE INTEGRATION (Fix First - 1-2 hours)**
**These must be fixed before data layer issues matter**

#### **Action 0.1: Fix Forecasting Import and API Call** 🔴
**File**: `core/acm_main.py`  
**Time**: 20 minutes  
**Priority**: BLOCKING - Nothing else works until this is fixed

**Step 1**: Fix import (line ~50)
```python
# REMOVE:
from . import correlation, outliers, forecast, river_models
from core import forecast

# ADD:
from . import correlation, outliers, forecasting
from core import forecasting
```

**Step 2**: Replace forecast.run() call (line ~2800)
```python
# REMOVE entire block:
forecast_ctx = {
    "run_dir": run_dir,
    "plots_dir": charts_dir,
    "tables_dir": tables_dir,
    "config": cfg,
    "enable_report": True,
}
forecast_result = forecast.run(forecast_ctx)

# REPLACE WITH:
if SQL_MODE and sql_client and equip_id and run_id:
    try:
        forecasting.run_and_persist_enhanced_forecasting(
            sql_client=sql_client,
            equip_id=equip_id,
            run_id=run_id,
            config=cfg,
            output_manager=output_manager,
            tables_dir=tables_dir,
            equip=equip,
            current_batch_time=win_end,
            sensor_data=score_numeric,
        )
        Console.info("[FORECAST] Enhanced forecasting + RUL completed")
    except Exception as e:
        Console.error(f"[FORECAST] Enhanced forecasting failed: {e}")
        # Optionally continue without forecasts
```

**Validation**:
```bash
python -m core.acm_main --equip FD_FAN 2>&1 | grep -i "Enhanced forecasting"
# Should see: "[FORECAST] Enhanced forecasting + RUL completed"
```

---

#### **Action 0.2: Add Forecast/RUL Tables to OutputManager Whitelist** 🔴
**File**: `core/output_manager.py`  
**Time**: 5 minutes  
**Priority**: BLOCKING - SQL writes will fail without this

**Change**:
```python
# Find ALLOWED_TABLES (line ~50-80)
ALLOWED_TABLES = {
    'ACM_Scores_Wide',
    'ACM_Scores_Long',
    'ACM_Episodes',
    'ACM_HealthTimeline',
    'ACM_RegimeTimeline',
    # ... existing tables ...
    
    # ✅ ADD THESE:
    'ACM_HealthForecast',
    'ACM_FailureForecast',
    'ACM_DetectorForecast_TS',
    'ACM_SensorForecast',
    'ACM_HealthForecast_TS',
    'ACM_FailureForecast_TS',
    'ACM_RUL',
    'ACM_RUL_TS',
    'ACM_RUL_Summary',
    'ACM_RUL_Attribution',
    'ACM_MaintenanceRecommendation',
    'ACM_RUL_LearningState',
}
```

**Validation**:
```python
# After run, check logs for successful writes
grep "SQL insert to ACM_RUL" logs/acm_*.log
grep "SQL insert to ACM_HealthForecast" logs/acm_*.log
```

---

#### **Action 0.3: Fix Health Timeline SQL Write in Fallback** 🔴
**File**: `core/acm_main.py`  
**Time**: 5 minutes  
**Priority**: HIGH - RUL fails without this

**Find fallback block** (line ~2900):
```python
# Current (WRONG):
if 'fused' in frame.columns:
    health_df = pd.DataFrame({...})
    output_manager.write_dataframe(health_df, tables_dir / "health_timeline.csv")

# Fix to (CORRECT):
if 'fused' in frame.columns:
    health_df = pd.DataFrame({...})
    output_manager.write_dataframe(
        health_df,
        tables_dir / "health_timeline.csv",
        sql_table="ACM_HealthTimeline",  # ✅ ADD SQL TABLE
        add_created_at=True,
    )
```

**Also fix regime fallback** (same block):
```python
regime_df = pd.DataFrame({...})
output_manager.write_dataframe(
    regime_df,
    tables_dir / "regime_timeline.csv",
    sql_table="ACM_RegimeTimeline",  # ✅ ADD THIS
    add_created_at=True,
)
```

---

#### **Action 0.4: Handle river_models Import Safely** 🟡
**File**: `core/acm_main.py`  
**Time**: 5 minutes

**Option A (Quick - Disable)**:
```python
# Comment out import
# from . import river_models
# from core import river_models

# Comment out usage (if any)
```

**Option B (Proper - Conditional)**:
```python
try:
    from core import river_models
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    Console.warn("[IMPORT] river_models not available, TAD disabled")

# Later in code:
if RIVER_AVAILABLE and cfg.get("river.enabled", False):
    # Use river_models
```

---

#### **Action 0.5: Add Regime Labeling Safety Net** 🟡
**File**: `configs/config_table.csv`  
**Time**: 2 minutes

Add row:
```
*,regimes,allow_legacy_label,True
```

This ensures regime failures don't kill entire pipeline.

**Alternative** (code fix in acm_main.py):
```python
try:
    regime_out = regimes.label(score, regime_ctx, {"frame": frame}, cfg)
except RuntimeError as e:
    Console.error(f"[REGIME] Labeling failed: {e}, continuing without regimes")
    # Create dummy regime labels (all regime 0)
    regime_out = {"regime_labels": pd.Series(0, index=score.index)}
```

---

### 🔴 **PHASE 1 - DATA LAYER (Fix After Code Integration - 30 minutes)**

#### **Action 1: Fix RUL_Hours Column Population** ⚠️
**File**: `core/forecasting.py`  
**Search**: `'RUL_Hours': 0.0` or `RUL_Hours.*=.*0.0`  
**Time**: 5 minutes

**Change**:
```python
# BEFORE (line ~2700):
rul_summary_df = pd.DataFrame([{
    'RUL_Hours': 0.0,  # ❌ WRONG
    'P50_Median': p50_hours,
    ...
}])

# AFTER:
rul_summary_df = pd.DataFrame([{
    'RUL_Hours': p50_hours,  # ✅ FIXED
    'P50_Median': p50_hours,
    ...
}])
```

**Validation**:
```sql
-- After fix, both should match:
SELECT RUL_Hours, P50_Median FROM ACM_RUL 
WHERE EquipID=1 ORDER BY CreatedAt DESC;
-- Expected: RUL_Hours = P50_Median (e.g., 11.5 = 11.5)
```

**Impact**: Eliminates false CRITICAL alerts, makes RUL predictions usable

---

#### **Action 2: Add RUL Confidence Score** ⚠️
**File**: `core/forecasting.py` (same location as Action 1)  
**Time**: 10 minutes

**Add before DataFrame construction**:
```python
# Calculate confidence from interval width
interval_width = p90_hours - p10_hours
max_expected_width = 200.0  # hours
confidence_score = max(0.0, min(1.0, 1.0 - (interval_width / max_expected_width)))

rul_summary_df = pd.DataFrame([{
    'RUL_Hours': p50_hours,
    'P50_Median': p50_hours,
    'P10_LowerBound': p10_hours,
    'P90_UpperBound': p90_hours,
    'Confidence': confidence_score,  # ✅ NEW
    ...
}])
```

**Validation**:
```sql
SELECT Confidence FROM ACM_RUL 
WHERE EquipID=1 AND Confidence IS NOT NULL 
ORDER BY CreatedAt DESC;
-- Expected: Values between 0.0 and 1.0 (e.g., 0.85, 0.72)
```

---

#### **Action 3: Fix Anomaly Timeline Query** ⚠️
**File**: Grafana dashboard JSON (ACM Claude Generated To Be Fixed.json)  
**Search**: Panel with title containing "Anomaly Events" or "Detection Timeline"  
**Time**: 10 minutes

**Replace query** (in panel's `targets` array):
```json
{
  "rawSql": "SELECT DATEADD(HOUR, DATEDIFF(HOUR, 0, StartTime), 0) AS time,\n       COUNT(*) AS value,\n       'Events' AS metric\nFROM ACM_Anomaly_Events\nWHERE EquipID = $equipment\n  AND StartTime BETWEEN $__timeFrom() AND $__timeTo()\nGROUP BY DATEADD(HOUR, DATEDIFF(HOUR, 0, StartTime), 0)\nORDER BY time ASC",
  "format": "time_series"
}
```

**Key changes**:
- ✅ Use `DATEADD()` not `FORMAT()` (returns DATETIME not VARCHAR)
- ✅ Add time range filter with Grafana variables
- ✅ `ORDER BY time ASC` (required for time series)
- ✅ Add `metric` column for legend

**Validation**: Refresh dashboard, panel should show event timeline graph

---

### 🟡 **P1 - HIGH (Fix Within 24 Hours - ~3 hours total)**

#### **Action 4: Remove Drift Detection Panel**
**File**: Grafana dashboard JSON  
**Time**: 2 minutes

**Option A (Quick)**:
1. Open dashboard in Grafana edit mode
2. Locate "Model Drift Detection - Concept Drift Evolution" panel
3. Click panel menu → Remove
4. Save dashboard

**Option B (JSON)**:
Search for panel with title "Drift" and remove entire panel object

**Add note**: Update dashboard description:
```
Note: Model Drift Detection feature scheduled for v11.0.0
```

---

#### **Action 5: Implement Sensor-Level Forecasting** 
**File**: `core/forecasting.py`  
**Time**: 2-3 hours (development + testing)

**Implementation outline**:
1. Add `_generate_sensor_forecasts()` method after `_run_forecast()`
2. For each sensor in top 10 contributors, fit ExponentialSmoothing model
3. Generate 7-day (168-hour) forecast per sensor
4. Build DataFrame with columns: Timestamp, SensorName, ForecastValue, CI_Lower, CI_Upper, TrendDirection, Method
5. Call `output_manager.write_dataframe()` with `sql_table="ACM_SensorForecast"`

**Test with**:
```sql
SELECT COUNT(*) FROM ACM_SensorForecast WHERE EquipID=1;
-- Should return > 0 after first run with fix
```

---

### 🟢 **P2 - MEDIUM (Fix When Convenient - ~2 hours total)**

#### **Action 6: Fix AdaptiveConfigManager Type Handling**
**File**: `core/forecasting.py` (AdaptiveConfigManager class or init)  
**Time**: 30 minutes

**Add type checking**:
```python
auto_tune_cfg = cfg.get("auto_tune", {})

if isinstance(auto_tune_cfg, dict):
    data_threshold = auto_tune_cfg.get("data_threshold", 50000)
    quality_threshold = auto_tune_cfg.get("quality_threshold", 0.7)
else:
    # Fallback for flat config structure
    data_threshold = cfg.get("auto_tune_data_threshold", 50000)
    quality_threshold = cfg.get("auto_tune_quality_threshold", 0.7)
```

**Validation**: Warning should disappear from logs

---

#### **Action 7: Fix PCA Fitted Check**
**File**: `core/output_manager.py`  
**Search**: `pca_detector.*_is_fitted` or `"PCA detector not fitted"`  
**Time**: 15 minutes

**Change**:
```python
# Add defensive checks:
if (hasattr(pca_detector, '_is_fitted') and pca_detector._is_fitted) or \
   (hasattr(pca_detector, 'pca') and pca_detector.pca is not None):
    # Write PCA metrics
    self._write_pca_metrics(pca_detector, ...)
```

**Validation**: Warning should disappear from logs

---

### 🔵 **P3 - FUTURE (Backlog for v11.0.0)**

#### **Action 8: Implement Full Drift Detection**
**Estimate**: 4-6 hours
1. Create `ACM_DriftMetrics` SQL table
2. Implement concept drift detection algorithms (PSI, KS test, etc.)
3. Enable drift check in `core/forecasting.py`
4. Write metrics to SQL during each run
5. Re-enable Grafana dashboard panel

#### **Action 9: Enhanced Confidence Metrics**
**Estimate**: 2-3 hours
- Add prediction interval coverage probability (PICP)
- Track forecast accuracy over time
- Implement adaptive confidence calibration

#### **Action 10: Sensor Attribution Enhancements**
**Estimate**: 3-4 hours  
- Add sensor interaction effects (cross-sensor contributions)
- Implement SHAP values for sensor attributions
- Add temporal attribution (when sensor became critical)

---

## 🧪 TESTING & VALIDATION CHECKLIST

### After Phase 0 Fixes (Code Integration - Test Immediately)

```bash
# 1. Clear Python cache
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force

# 2. Run single ACM test
python -m core.acm_main --equip FD_FAN --start-time "2025-07-01T00:00:00" --end-time "2025-08-01T00:00:00"

# 3. Check for forecasting execution (NEW - should appear now)
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "
SELECT TOP 5 LoggedAt, Message 
FROM ACM_RunLogs 
WHERE Message LIKE '%Enhanced forecasting%' OR Message LIKE '%RUL%'
ORDER BY LoggedAt DESC
" -W

# Expected output:
# [FORECAST] Enhanced forecasting + RUL completed
# [RUL] Multipath algorithm generated predictions
# [RUL] Wrote tables: ACM_RUL, ACM_HealthForecast, ACM_FailureForecast

# 4. Verify forecast tables populated (CRITICAL - should have rows now)
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "
SELECT 
    'HealthForecast' AS Table_Name, COUNT(*) AS Rows FROM ACM_HealthForecast WHERE EquipID=1
UNION ALL
SELECT 
    'FailureForecast', COUNT(*) FROM ACM_FailureForecast WHERE EquipID=1
UNION ALL
SELECT 
    'RUL', COUNT(*) FROM ACM_RUL WHERE EquipID=1
UNION ALL
SELECT 
    'RUL_LearningState', COUNT(*) FROM ACM_RUL_LearningState WHERE EquipID=1
" -W

# Expected: All should show > 0 rows if forecasting ran

# 5. Check for OutputManager errors
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "
SELECT TOP 10 LoggedAt, Message 
FROM ACM_RunLogs 
WHERE Level='ERROR' AND Message LIKE '%Invalid table name%'
ORDER BY LoggedAt DESC
" -W

# Expected: NO errors about invalid table names (was blocking writes)

# 6. Verify health timeline SQL write (even in fallback)
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "
SELECT COUNT(*) AS HealthRows, 
       MAX(Timestamp) AS LatestHealth
FROM ACM_HealthTimeline 
WHERE EquipID=1
" -W

# Expected: Should have new rows from this run
```

**Success Criteria (Phase 0)**:
- ✅ No import errors for `forecasting` module
- ✅ Logs show "[FORECAST] Enhanced forecasting + RUL completed"
- ✅ ACM_RUL table has new rows (> 0)
- ✅ ACM_HealthForecast, ACM_FailureForecast populated
- ✅ No "Invalid table name" errors in logs
- ✅ ACM_HealthTimeline has rows even if analytics failed

---

### After Phase 1 Fixes (Data Layer - Test Second)

```bash
# 1. Clear old data (optional - for clean test)
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "DELETE FROM ACM_RUL WHERE EquipID=1"

# 2. Run single ACM batch
python -m core.acm_main --equip FD_FAN --start-time "2025-07-01T00:00:00" --end-time "2025-08-01T00:00:00"

# 3. Validate RUL_Hours populated correctly
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "
SELECT TOP 1 
    RUL_Hours, 
    P50_Median, 
    CASE WHEN RUL_Hours = P50_Median THEN 'PASS' ELSE 'FAIL' END AS Test_Result,
    Confidence,
    CASE WHEN Confidence IS NOT NULL THEN 'PASS' ELSE 'FAIL' END AS Confidence_Test
FROM ACM_RUL 
WHERE EquipID=1 
ORDER BY CreatedAt DESC
" -W

# Expected output:
# RUL_Hours  P50_Median  Test_Result  Confidence  Confidence_Test
# 11.5       11.5        PASS         0.85        PASS

# 4. Refresh Grafana dashboard
# - Open dashboard in browser
# - Click refresh button (top right)
# - Verify "RUL Summary" now shows correct non-zero value
# - Verify "Maintenance Recommendations" shows usable RUL values

# 5. Check Anomaly Timeline
# - Should now show event graph (not "No data")
# - Events should align with timeline across 2023-2025
```

### After P1 Fixes (High - Test Within 24h)

```sql
-- Validate sensor forecasts generated
SELECT COUNT(*) AS SensorForecastRows,
       COUNT(DISTINCT SensorName) AS UniqueSensors,
       MIN(Timestamp) AS FirstForecast,
       MAX(Timestamp) AS LastForecast
FROM ACM_SensorForecast 
WHERE EquipID=1;

-- Expected:
-- SensorForecastRows: 1680 (10 sensors × 168 hours)
-- UniqueSensors: 10
-- Date range: 7 days forward from run time
```

```bash
# Check Grafana sensor forecast panels
# - "Top 5 Physical Sensor Forecasts" should show trend lines
# - "Sensor Forecast Summary" should show sensor attribution table
```

### After P2 Fixes (Medium - Test When Convenient)

```bash
# Run and check for warning elimination
python -m core.acm_main --equip FD_FAN 2>&1 | grep -i "warning"

# Expect:
# - No "AdaptiveConfigManager" warnings
# - No "PCA detector not fitted" warnings
# - Only expected warnings (gappy data, regime quality)
```

---

## 📈 SUCCESS METRICS

### Immediate Success Criteria (P0 Fixes)
- ✅ RUL_Hours column shows non-zero values matching P50_Median
- ✅ Confidence column shows values between 0.0 and 1.0 (not NULL)
- ✅ Grafana RUL Summary panel shows valid predictions (not 0.0 hour CRITICAL)
- ✅ Maintenance Recommendations table usable for planning
- ✅ Anomaly Events timeline graph displays (not "No data")

### High Priority Success Criteria (P1 Fixes)
- ✅ ACM_SensorForecast table has > 0 rows
- ✅ Sensor forecast dashboard panels display trend graphs
- ✅ Drift detection panel removed (or implemented and working)

### Medium Priority Success Criteria (P2 Fixes)
- ✅ Zero "AdaptiveConfigManager" warnings in logs
- ✅ Zero "PCA detector not fitted" warnings in logs
- ✅ Clean log output with only expected informational warnings

---

## 🚀 ROLLOUT STRATEGY

### Phase 1: Emergency Hotfix (P0) - Deploy Immediately
**Timeline**: Within 1 hour of approval  
**Risk**: Minimal - only fixing data column population  
**Deployment**:
1. Apply P0 fixes to `core/forecasting.py`
2. Update Grafana dashboard query for anomaly timeline
3. Test with single run (validation checklist above)
4. Deploy to production if tests pass

**Rollback**: Revert forecasting.py changes if issues, previous RUL data unaffected

---

### Phase 2: High Priority (P1) - Deploy Within 24-48h
**Timeline**: 1-2 days for development + testing  
**Risk**: Low-Medium - new feature (sensor forecasts), isolated  
**Deployment**:
1. Implement sensor forecast generator
2. Test extensively with historical data (10-20 batches)
3. Validate SQL table population
4. Update/verify Grafana panels
5. Deploy during maintenance window

**Rollback**: Sensor forecasts independent, can disable without affecting core

---

### Phase 3: Medium Priority (P2) - Deploy Next Sprint
**Timeline**: 1 week  
**Risk**: Minimal - config handling + warning fixes  
**Deployment**:
1. Apply config manager type handling
2. Fix PCA fitted checks
3. Run full batch test suite
4. Deploy with regular release cycle

---

### Phase 4: Future Enhancements (P3) - Schedule v11.0.0
**Timeline**: Next major release (1-2 months)  
**Risk**: Medium - new feature (drift detection)  
**Planning**:
- Design drift metrics schema
- Research drift detection algorithms
- Prototype on development branch
- Full testing cycle before merge

---

## 📝 DOCUMENTATION UPDATES NEEDED

### Update README.md
Add to "Known Issues" section:
```markdown
## v10.0.0 Known Issues (Fixed in v10.0.1)
- RUL_Hours column showed 0.0 (fixed: now uses P50_Median)
- Confidence scores were NULL (fixed: calculated from CI width)
- Anomaly timeline panel showed "No data" (fixed: query corrected)
- Sensor forecasts not implemented (added in v10.0.1)
- Drift detection panel broken (removed pending v11.0.0 implementation)
```

### Update ACM_SYSTEM_OVERVIEW.md
Add to "Forecasting" section:
```markdown
### RUL Calculation Details
- **Primary Metric**: `RUL_Hours` = P50_Median (50th percentile)
- **Confidence Intervals**: P10 (lower), P50 (median), P90 (upper)
- **Confidence Score**: 0.0-1.0, calculated from interval width
  - Formula: 1.0 - (P90 - P10) / 200.0
  - Narrower interval = higher confidence
- **Status Thresholds**:
  - Critical: RUL < 24 hours
  - Warning: RUL < 72 hours
  - Caution: RUL < 168 hours
  - Healthy: RUL > 168 hours
```

### Update Grafana Dashboard Description
Add warning banner:
```
⚠️ Dashboard v10.0.0 Notes:
- Drift Detection panel temporarily disabled (v11.0.0)
- Gappy data warnings expected for historical batches
- RUL predictions use P50 median (50th percentile)
```

---

## 🔧 DEVELOPER NOTES

### Code Locations Quick Reference
```
Issue #1 & #2: RUL_Hours = 0.0
  File: core/forecasting.py
  Search: 'RUL_Hours': 0.0
  Line: ~2700
  
Issue #3: Drift Detection Panel
  File: grafana_dashboards/ACM Claude Generated To Be Fixed.json
  Search: "Drift Detection"
  
Issue #4: Anomaly Timeline Query
  File: grafana_dashboards/ACM Claude Generated To Be Fixed.json
  Search: "Anomaly Events" or "Detection Timeline"
  
Issue #5 & #6: Sensor Forecasts
  File: core/forecasting.py
  Add: _generate_sensor_forecasts() method
  Location: After _run_forecast() method
  
Issue #7: PCA Fitted Check
  File: core/output_manager.py
  Search: "PCA detector not fitted"
  
Issue #8: Config Type Handling
  File: core/forecasting.py
  Search: "auto_tune" or "AdaptiveConfigManager"
  
Issue #10: Confidence Calculation
  File: core/forecasting.py (same as #1)
  Location: Before rul_summary_df construction
```

### SQL Queries for Validation
```sql
-- Check RUL data quality
SELECT 
    COUNT(*) AS TotalPredictions,
    COUNT(CASE WHEN RUL_Hours = 0.0 THEN 1 END) AS ZeroRUL,
    COUNT(CASE WHEN Confidence IS NULL THEN 1 END) AS NullConfidence,
    COUNT(CASE WHEN RUL_Hours = P50_Median THEN 1 END) AS CorrectRUL,
    AVG(Confidence) AS AvgConfidence
FROM ACM_RUL 
WHERE EquipID=1;

-- Check forecast data completeness
SELECT 
    'Health' AS ForecastType, COUNT(*) AS Rows FROM ACM_HealthForecast WHERE EquipID=1
UNION ALL
SELECT 
    'Failure', COUNT(*) FROM ACM_FailureForecast WHERE EquipID=1
UNION ALL
SELECT 
    'Sensor', COUNT(*) FROM ACM_SensorForecast WHERE EquipID=1
UNION ALL
SELECT 
    'RUL', COUNT(*) FROM ACM_RUL WHERE EquipID=1;

-- Check anomaly detection data
SELECT 
    'Events' AS Type, COUNT(*) AS Count FROM ACM_Anomaly_Events WHERE EquipID=1
UNION ALL
SELECT 
    'Episodes', COUNT(*) FROM ACM_EpisodeDiagnostics WHERE EquipID=1
UNION ALL
SELECT 
    'Culprits', COUNT(*) FROM ACM_EpisodeCulprits WHERE EquipID=1;
```

### Testing Commands
```bash
# Single run test
python -m core.acm_main --equip FD_FAN --start-time "2025-07-01T00:00:00" --end-time "2025-08-01T00:00:00"

# Batch run test (10 batches)
python scripts/sql_batch_runner.py --equip FD_FAN --max-batches 10 --start-from-beginning

# Check logs for errors
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "
SELECT TOP 20 LoggedAt, Level, Message 
FROM ACM_RunLogs 
WHERE Level IN ('ERROR', 'WARNING') 
ORDER BY LoggedAt DESC
" -W
```

---

## 📞 SUPPORT & ESCALATION

### If Fixes Don't Work

**Issue #1/2 (RUL still showing 0.0)**:
1. Check if code change actually saved
2. Clear Python cache: `Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force`
3. Re-run single test batch
4. Query directly: `SELECT TOP 1 RUL_Hours, P50_Median FROM ACM_RUL ORDER BY CreatedAt DESC`
5. If still failing, check for multiple `RUL_Hours` assignments in code

**Issue #4 (Anomaly timeline still empty)**:
1. Verify Grafana dashboard JSON saved correctly
2. Check Grafana panel query in edit mode
3. Test query directly in SSMS
4. Verify time range includes data: `SELECT MIN(StartTime), MAX(StartTime) FROM ACM_Anomaly_Events WHERE EquipID=1`
5. Check Grafana datasource connection

**Issue #5/6 (Sensor forecasts still empty after implementation)**:
1. Check if `_generate_sensor_forecasts()` is actually called
2. Add debug logging: `Console.info(f"[DEBUG] Generating sensor forecasts...")`
3. Verify `output_manager.write_dataframe()` called with correct table name
4. Check for exceptions in logs
5. Manually inspect DataFrame before write

---

## 🔄 ROOT CAUSE SUMMARY

### Two-Layer Failure Analysis

**Layer 1: Code Integration (Primary Root Cause)**
- **What**: v10.0.0 forecasting refactor never integrated into main pipeline
- **Why**: Old `forecast.py` API still referenced, new `forecasting.py` API not called
- **Impact**: New forecasting engine, RUL engine, sensor forecasts - **none execute**
- **Evidence**: No logs showing "Enhanced forecasting", zero rows in new tables
- **Fix Time**: 1-2 hours (5 code changes)

**Layer 2: Data/Dashboard (Secondary Issues - Only Visible When Pipeline Runs)**
- **What**: When older code paths execute, data calculated correctly but dashboard displays wrong
- **Why**: Column population bugs (RUL_Hours=0.0), incomplete features (confidence, sensor forecasts)
- **Impact**: False CRITICAL alerts, missing visualizations
- **Evidence**: SQL shows P50=11.5h but RUL_Hours=0.0, Confidence=NULL
- **Fix Time**: 30 minutes (3 data fixes)

**Key Insight**: Layer 1 must be fixed first. Layer 2 issues only matter once forecasting actually runs.

---

## ✅ CONCLUSION

**System Assessment - Revised Understanding**:

The ACM system is in a **transitional state** between two architectures:

**Current State (What's Running)**:
- ✅ Old detection pipeline: 76 episodes, 9,087 health points generated
- ✅ Basic health tracking working
- ⚠️ Partial forecasting via old code paths (explains some RUL data exists)
- ❌ New v10 forecasting: **Not executing** (import/API mismatch)
- ❌ New RUL engine: **Never called** (hidden behind non-executing forecasting)
- ❌ Sensor forecasts: **Not implemented** (0 rows)

**What Was Observed in 10-Batch Test**:
- 3,360 health forecasts, 3,360 failure forecasts → **Old forecasting** still partially working
- 10+ RUL predictions with P50=11.5h but RUL_Hours=0.0 → **Data bug** in whichever path generated this
- Dashboard shows mix of working (health timeline) and broken (RUL display, anomaly timeline)

**Critical Finding - Two-Phase Fix Required**:

**Phase 1: Code Integration (1-2 hours)** - MUST DO FIRST
1. Fix forecasting import: `forecast` → `forecasting` (20 min)
2. Add forecast/RUL tables to OutputManager whitelist (5 min)
3. Fix health timeline SQL in fallback path (5 min)
4. Handle river_models import safely (5 min)
5. Add regime labeling safety net (5 min)

**Phase 2: Data Layer (30 minutes)** - Do after Phase 1
1. Fix RUL_Hours column population (5 min)
2. Add Confidence score calculation (10 min)
3. Fix anomaly timeline query (10 min)

**Total Fix Time**: 
- Phase 1 (Code Integration): 1-2 hours
- Phase 2 (Data/Dashboard): 30 minutes
- **Combined Total**: 2-2.5 hours

**Risk Assessment**: 
- **Phase 1 Risk**: MEDIUM - Changing core pipeline wiring, need thorough testing
- **Phase 2 Risk**: LOW - Data column fixes, isolated changes
- **Rollback**: Phase 1 complex (revert multiple files), Phase 2 simple (single file)

**Success Path**:
1. Apply Phase 1 fixes → Enables new forecasting engine
2. Test forecasting execution → Verify new tables populated
3. Apply Phase 2 fixes → Corrects data display
4. Test dashboard → Verify accurate RUL, confidence, timelines
5. Deploy to production with monitoring

---

## 📋 APPENDIX

### A. SQL Schema Reference
Key tables involved in issues:
```sql
-- ACM_RUL (Issue #1, #2, #10)
CREATE TABLE ACM_RUL (
    EquipID INT NOT NULL,
    RunID UNIQUEIDENTIFIER NOT NULL,
    RUL_Hours FLOAT,           -- ❌ Currently 0.0
    P50_Median FLOAT,          -- ✅ Correctly calculated
    P10_LowerBound FLOAT,
    P90_UpperBound FLOAT,
    Confidence FLOAT,          -- ❌ Currently NULL
    Method NVARCHAR(50),
    CreatedAt DATETIME2
);

-- ACM_SensorForecast (Issue #5, #6)
CREATE TABLE ACM_SensorForecast (
    EquipID INT NOT NULL,
    RunID UNIQUEIDENTIFIER NOT NULL,
    Timestamp DATETIME2 NOT NULL,
    SensorName NVARCHAR(256),
    ForecastValue FLOAT,
    CI_Lower FLOAT,
    CI_Upper FLOAT,
    Method NVARCHAR(50),
    CreatedAt DATETIME2
);
-- Status: Empty (0 rows) - feature not implemented

-- ACM_DriftMetrics (Issue #3)
-- Status: DOES NOT EXIST - needs creation for v11.0.0
```

### B. Log Patterns for Monitoring

**Success Patterns** (look for these after fixes):
```
[FORECAST] RUL P50=11.5h, P10=1.5h, P90=170.2h
[OUTPUT] SQL insert to ACM_RUL: 1 rows
[FORECAST] Wrote tables: ACM_HealthForecast, ACM_FailureForecast, ACM_RUL
```

**Failure Patterns** (should disappear after fixes):
```
[OUTPUT] PCA detector not fitted, skipping metrics write
[AdaptiveConfigManager] Failed to get config 'auto_tune_data_threshold'
```

**Expected Warnings** (normal for batch mode):
```
[ForecastEngine] GAPPY data detected - proceeding with available data
[REGIME] Clustering quality below threshold; per-regime thresholds disabled
[DATA] Training data (0 rows) is below recommended minimum (200 rows)
```

### C. Version History & Architecture Evolution

**v9.0.0 (Stable Baseline)**:
- ✅ Old forecasting API (`forecast.run()`)
- ✅ Basic RUL via old path
- ✅ File + SQL dual mode
- ⚠️ Limited forecast capabilities

**v10.0.0 (Current - Incomplete Integration)**: 
- ⚠️ New forecasting refactor **exists but not wired**
- ⚠️ Old forecasting API still called (mismatch)
- ❌ RUL_Hours column bug (data layer)
- ❌ Confidence NULL (data layer)
- ❌ Missing sensor forecasts (not implemented)
- ❌ Drift detection disabled
- ❌ OutputManager missing forecast/RUL tables
- ❌ Health timeline fallback only writes CSV
- **Status**: Hybrid state - old paths running, new paths dormant

**v10.0.1 (Integration Target - 2-2.5 hours)**:
- ✅ Forecasting properly integrated (`forecasting.run_and_persist_enhanced_forecasting()`)
- ✅ RUL engine activated (called via forecasting)
- ✅ OutputManager whitelist updated (forecast/RUL tables)
- ✅ Health timeline SQL writes in fallback
- ✅ RUL_Hours fixed (uses P50_Median)
- ✅ Confidence calculated
- ✅ Anomaly timeline query fixed
- ✅ River_models import handled safely
- ✅ Regime labeling safety net
- ⚠️ Sensor forecasts deferred to v10.1.0
- ⚠️ Drift detection panel removed (defer to v11.0.0)

**v10.1.0 (Feature Complete - 2-3 hours)**:
- ✅ Sensor-level forecasting implemented
- ✅ All dashboard panels working
- ✅ health_tracker.py integrated
- ✅ state_manager.py integrated

**v11.0.0 (Future - Next Quarter)**:
- ✅ Full drift detection (table + metrics)
- ✅ Enhanced confidence metrics (PICP, calibration)
- ✅ Sensor interaction effects (SHAP)
- ✅ Advanced RUL algorithms (deep learning)

---

### D. Code Archaeology - Critical Files

**Files Modified in v10 Refactor (But Not Integrated)**:
- `core/forecasting.py` - New unified forecasting engine ✅ EXISTS
- `core/rul_engine.py` - New RUL calculation engine ✅ EXISTS
- `core/health_tracker.py` - Health timeline management ✅ EXISTS (unused)
- `core/state_manager.py` - Forecast state persistence ✅ EXISTS (unused)

**Files That Need Updates for Integration**:
- `core/acm_main.py` - Still calls old API ❌ NEEDS UPDATE
- `core/output_manager.py` - Missing forecast tables ❌ NEEDS UPDATE
- `configs/config_table.csv` - Needs regime safety config ⚠️ OPTIONAL

**API Migration Path**:
```python
# OLD (v9.0.0) - Currently in acm_main.py:
forecast_ctx = {"run_dir": ..., "config": cfg}
forecast_result = forecast.run(forecast_ctx)

# NEW (v10.0.0) - Should be in acm_main.py:
forecasting.run_and_persist_enhanced_forecasting(
    sql_client=sql_client,
    equip_id=equip_id,
    run_id=run_id,
    config=cfg,
    output_manager=output_manager,
    tables_dir=tables_dir,
    equip=equip,
    current_batch_time=win_end,
    sensor_data=score_numeric,
)
```

---

**Document Version**: 2.0 (Merged Code + Data Analysis)  
**Last Updated**: December 8, 2025  
**Analysis Type**: Code Integration Audit + Data/Dashboard Health Check  
**Status**: Ready for Implementation - Two-Phase Approach  
**Approval Required**: Technical Lead (code changes), Operations Manager (data validation)  
**Estimated Total Fix Time**: 
- **Phase 0 (Code Integration)**: 1-2 hours (5 changes)
- **Phase 1 (Data Layer)**: 30 minutes (3 changes)
- **Phase 2 (Features)**: 3+ hours (sensor forecasts, config)
- **Grand Total**: 2-5.5 hours depending on scope

---

## 📋 QUICK REFERENCE - FIX CHECKLIST

### Must Do First (Phase 0 - Code Integration)
- [ ] Fix `acm_main.py` import: `forecast` → `forecasting`
- [ ] Replace `forecast.run(ctx)` with `forecasting.run_and_persist_enhanced_forecasting(...)`
- [ ] Add 12 forecast/RUL tables to `ALLOWED_TABLES` in `output_manager.py`
- [ ] Add `sql_table="ACM_HealthTimeline"` to health fallback in `acm_main.py`
- [ ] Handle `river_models` import (comment out or try/except)
- [ ] Add `regimes.allow_legacy_label=True` to config OR add try/except

### Then Do (Phase 1 - Data Layer)
- [ ] Fix `RUL_Hours` column in `forecasting.py`: `0.0` → `p50_hours`
- [ ] Add confidence calculation in `forecasting.py` before RUL DataFrame
- [ ] Fix anomaly timeline Grafana query: use `DATEADD()` not `FORMAT()`

### Optional (Phase 2 - Features)
- [ ] Remove drift detection panel from Grafana dashboard
- [ ] Implement sensor-level forecasting in `forecasting.py`
- [ ] Fix AdaptiveConfigManager type handling
- [ ] Fix PCA fitted check in `output_manager.py`

---

*END OF DOCUMENT*

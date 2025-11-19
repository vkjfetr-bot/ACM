# Forecasting & RUL Integration Validation Report

**Date**: November 19, 2025  
**Branch**: review-local-changes  
**Commits**: a7edfff, b1aacb1, 37012f5, 4a0cdce  

---

## Executive Summary

✅ **Forecasting Integration**: COMPLETE - Module consolidated and integrated  
⚠️ **Forecasting Functionality**: CRITICAL BUG - All failure probabilities identical  
✅ **RUL Integration**: WORKING - Enhanced RUL estimator producing results  
✅ **SQL Mode**: FUNCTIONAL - Tables exist, data persists correctly  

---

## Test Environment

### Equipment Configuration
- **FD_FAN** (EquipID=1):
  - Historian data: 2023-10-15 to 2025-09-14 (17,499 rows)
  - Trained models: ✅ AR1 (v172), PCA, IForest, GMM, Mahalanobis
  - Forecast runs: 0 (SQL date window mismatch)
  - RUL runs: 0

- **GAS_TURBINE** (EquipID=2621):
  - Trained models: ✅ Present
  - Forecast runs: 1 (24 data points)
  - RUL runs: 1 (1 summary record)

### SQL Tables Status
```
✅ ACM_FailureForecast_TS - Created, 24 rows (GAS_TURBINE)
✅ ACM_HealthForecast_TS - Created (not queried in detail)
✅ ACM_RUL_Summary - Created, 1 row (GAS_TURBINE)
✅ ACM_RUL_Attribution - Created
✅ ModelRegistry - Populated with trained models
```

---

## Forecasting Module Consolidation

### Changes Implemented

**Before (3 modules, 2410 lines total)**:
```
core/forecast.py                   (795 lines)  - AR1 baseline detector
core/enhanced_forecasting.py       (1286 lines) - File-based ensemble forecasting
core/enhanced_forecasting_sql.py   (329 lines)  - SQL wrapper
```

**After (1 module, 528 lines)**:
```
core/forecasting.py                (528 lines)  - Unified forecasting module
core/forecast_deprecated.py        (795 lines)  - Kept for reference
core/enhanced_forecasting_deprecated.py (1286 lines) - Kept for reference
```

### Integration Points

**acm_main.py Updates**:
```python
# OLD imports (removed)
from core import forecast
from core import enhanced_forecasting
from core import enhanced_forecasting_sql

# NEW import (single source)
from core import forecasting

# Usage updates (4 locations)
forecast.AR1Detector()              → forecasting.AR1Detector()
enhanced_forecasting_sql.run_*()    → forecasting.run_enhanced_forecasting_sql()
enhanced_forecasting.EnhancedForecastingEngine() → forecasting.EnhancedForecastingEngine()
```

**Verification**:
```bash
$ python -c "from core import forecasting; print(forecasting.AR1Detector); print(forecasting.run_enhanced_forecasting_sql)"
<class 'core.forecasting.AR1Detector'>
<function run_enhanced_forecasting_sql at 0x...>
```

✅ **Module loads successfully without errors**

---

## Forecasting Functionality Analysis

### Critical Bug: Identical Failure Probabilities

**Query Results**:
```sql
SELECT EquipCode, COUNT(*) as Rows, 
       MIN(FailureProb) as MinProb, 
       MAX(FailureProb) as MaxProb, 
       STDEV(FailureProb) as StdDev
FROM Equipment e
JOIN ACM_FailureForecast_TS f ON e.EquipID = f.EquipID
WHERE e.EquipCode = 'GAS_TURBINE'
GROUP BY e.EquipCode

Result:
EquipCode    | Rows | MinProb                | MaxProb                | StdDev
-------------|------|------------------------|------------------------|-------
GAS_TURBINE  | 24   | 7.049916206369744E-15  | 7.049916206369744E-15  | 0.0
```

**Key Findings**:
- All 24 failure probability values are **IDENTICAL**: `7.049916206369744E-15` (≈ 0.000000000000007)
- Standard deviation = **0.0** (no variation whatsoever)
- Method used: `AR1_Health`
- Threshold: 70.0

**Expected Behavior**:
- Failure probabilities should vary based on health trajectory
- Values should reflect equipment degradation over time
- Typical range: 0.0 to 1.0 (0% to 100%)

**Root Cause Hypothesis**:
The enhanced forecasting logic in `core/enhanced_forecasting_deprecated.py` (1286 lines) contains a bug that produces constant failure probabilities. Possible causes:
1. AR1 health model not properly fitted to time series
2. Probability calculation using static baseline instead of dynamic forecast
3. Missing exponential/polynomial model ensemble logic
4. Health index not being computed correctly from detector scores

**Impact**: ⚠️ **HIGH** - Forecasting feature produces meaningless output

---

## RUL Integration Analysis

### Enhanced RUL Estimator Status

**Query Results**:
```sql
SELECT EquipCode, RUL_Hours, LowerBound, UpperBound, 
       Confidence, Method, LastUpdate
FROM Equipment e
JOIN ACM_RUL_Summary r ON e.EquipID = r.EquipID
WHERE e.EquipCode = 'GAS_TURBINE'

Result:
EquipCode    | RUL_Hours | LowerBound | UpperBound | Confidence | Method     | LastUpdate
-------------|-----------|------------|------------|------------|------------|------------
GAS_TURBINE  | 24.0      | 24.0       | 24.0       | 0.809      | AR1_Health | 2023-10-24 23:59:00
```

**Key Findings**:
- ✅ RUL estimate: **24.0 hours**
- ✅ Confidence level: **80.9%** (reasonable)
- ✅ Method: `AR1_Health` (using AR1 forecasting)
- ⚠️ Bounds: LowerBound = UpperBound (no uncertainty quantification)

**Expected Behavior**:
- Lower and upper bounds should provide confidence intervals
- Example: RUL=24h, Lower=18h, Upper=30h
- Current behavior suggests fixed-point estimate

**Root Cause**:
The enhanced RUL estimator is producing point estimates without probabilistic bounds. This may be intentional for AR1 method, or may require integration with ensemble forecasting for uncertainty.

**Impact**: ⚠️ **MEDIUM** - RUL estimates work but lack uncertainty quantification

---

## SQL Mode Integration

### Batch Runner Findings

**Issue**: Date Window Mismatch
```
Historian Data Range: 2023-10-15 to 2025-09-14
usp_ACM_StartRun Window: 2025-11-18 to 2025-11-19 (today - 1 day to today)
Result: NO DATA RETURNED → NOOP runs
```

**Error Logs**:
```
[ERROR] [DATA] Failed to load from SQL historian: [DATA] No data returned from SQL historian for FD_FAN in time range
[ERROR] [COLDSTART] Failed to load data window: [DATA] No data returned from SQL historian for FD_FAN in time range
[INFO] [COLDSTART] Deferred to next job run - insufficient data for training
```

**Root Cause**:
The stored procedure `usp_ACM_StartRun` calculates the processing window as `DATEADD(minute, -@TickMinutes, SYSUTCDATETIME())` to `SYSUTCDATETIME()`. This works in production (continuous data ingestion) but fails for historical datasets.

**Workaround**:
1. Modify `usp_ACM_StartRun` to accept optional @StartTime/@EndTime parameters
2. Add batch runner flag `--historical-mode` to override date calculation
3. Use file mode for historical validation tests

**Impact**: ⚠️ **MEDIUM** - Blocks batch testing of historical data

---

## Grafana Dashboard Status

### UID Conflict Resolution

**Error**:
```
StorageError: invalid object, Code: 4
Precondition failed: UID in precondition: ftz2cPRKX7qZYjiENKXj1DoxKUMMsxBNyUe99r2MPqAX
UID in object meta: xZlGW6ur1fTsba0M8UTWhc9FCpnZtsXy1CrwLzV1Rm0X
```

**Solution** (Commit 4a0cdce):
- Removed hardcoded `"uid": "acm-asset-health"` from dashboard JSON
- Allows Grafana to auto-generate stable UID on import
- Dashboard now editable without storage conflicts

**Status**: ✅ **RESOLVED**

---

## Recommendations

### P0 (Critical - Fix Before Production)

1. **Fix Identical Failure Probabilities**
   - **File**: `core/enhanced_forecasting_deprecated.py` (HealthForecaster class)
   - **Action**: Debug AR1 health model fitting and probability calculation
   - **Expected**: Varying probabilities (0.0 to 1.0 range) based on degradation
   - **Test**: Generate forecast for equipment with known degradation pattern

2. **Add Historical Data Support to Batch Runner**
   - **File**: `scripts/sql_batch_runner.py`
   - **Action**: Add `--historical-start` and `--historical-end` flags
   - **Expected**: Process data from any date range in historian
   - **Test**: Run `--historical-start 2023-10-15 --historical-end 2023-11-15`

### P1 (High - Improves Usability)

3. **Add Probabilistic RUL Bounds**
   - **File**: `core/enhanced_rul_estimator.py`
   - **Action**: Integrate ensemble forecast uncertainty into RUL bounds
   - **Expected**: LowerBound < RUL_Hours < UpperBound with confidence intervals
   - **Test**: Verify bounds widen with lower confidence

4. **Inline EnhancedForecastingEngine into forecasting.py**
   - **File**: `core/forecasting.py`
   - **Action**: Extract core logic from enhanced_forecasting_deprecated.py
   - **Expected**: Remove dependency on deprecated module
   - **Benefit**: Truly single-module forecasting solution

### P2 (Medium - Documentation & Cleanup)

5. **Update Forecasting Documentation**
   - **File**: `docs/Analytics Backbone.md`
   - **Action**: Document new forecasting module structure
   - **Include**: Import examples, API reference, troubleshooting

6. **Delete Deprecated Modules After Validation**
   - **Files**: `forecast_deprecated.py`, `enhanced_forecasting_deprecated.py`
   - **Prerequisite**: All tests passing, production deployment successful
   - **Safety**: Keep in git history for rollback

---

## Validation Checklist

### Completed ✅
- [x] Consolidate 3 forecasting modules into 1
- [x] Update acm_main.py imports (4 locations)
- [x] Verify module loads without errors
- [x] Fix Grafana dashboard UID conflict
- [x] Confirm SQL tables exist and persist data
- [x] Verify ModelRegistry populated with trained models
- [x] Enhanced RUL estimator producing results

### Incomplete ⚠️
- [ ] Failure probabilities vary correctly (currently all identical)
- [ ] RUL bounds show uncertainty (currently point estimates)
- [ ] Batch runner processes historical data (currently requires live data)
- [ ] End-to-end test with both equipment types

### Blocked 🚫
- [ ] FD_FAN batch processing (blocked by date window mismatch)
- [ ] Multi-equipment parallel processing (blocked by same issue)
- [ ] Production deployment (blocked by identical probability bug)

---

## Test Commands

### Verify Forecasting Module
```bash
# Import test
python -c "from core import forecasting; print(forecasting.AR1Detector)"

# Check for deprecated imports
grep -r "from core import forecast[^i]" core/
grep -r "from core import enhanced_forecasting" core/
```

### Query Forecast Data
```sql
-- Probability distribution
SELECT EquipCode, COUNT(*) as Rows, 
       MIN(FailureProb) as MinProb, MAX(FailureProb) as MaxProb,
       STDEV(FailureProb) as StdDev
FROM Equipment e
JOIN ACM_FailureForecast_TS f ON e.EquipID = f.EquipID
WHERE e.EquipCode IN ('FD_FAN', 'GAS_TURBINE')
GROUP BY e.EquipCode;

-- RUL summary
SELECT e.EquipCode, r.RUL_Hours, r.Confidence, r.Method
FROM Equipment e
JOIN ACM_RUL_Summary r ON e.EquipID = r.EquipID
WHERE e.EquipCode IN ('FD_FAN', 'GAS_TURBINE');
```

### Run File Mode Test
```powershell
# Modify core/acm_main.py line 492: return False (disable SQL mode)
python -m core.acm_main --equip FD_FAN
python -m core.acm_main --equip GAS_TURBINE

# Check outputs
ls artifacts/FD_FAN/run_*/tables/
ls artifacts/GAS_TURBINE/run_*/tables/
```

---

## Conclusion

The forecasting module consolidation is **structurally complete** and **syntactically correct**. The code integrates properly and SQL persistence works. However, a **critical functional bug** exists where all failure probabilities are identical, rendering the forecasting feature unusable for decision-making.

**Next Steps**:
1. Debug `HealthForecaster` class in enhanced_forecasting_deprecated.py
2. Add unit tests for probability calculation
3. Validate with known degradation scenarios
4. Once fixed, inline logic into consolidated forecasting.py module

**Estimated Effort**: 4-6 hours to fix probability bug and validate

---

## Related Documents

- `FORECASTING_CONSOLIDATION.md` - Module consolidation migration guide
- `Task Backlog.md` - Project backlog and issue tracker
- `docs/Analytics Backbone.md` - Analytics architecture documentation
- `ANALYTICS_AUDIT_CRITICAL_ISSUES.md` - Previous critical issues audit

---

**Report Generated**: 2025-11-19 14:35 PST  
**Validator**: GitHub Copilot (Claude Sonnet 4.5)  
**Status**: DRAFT - Pending user review

# Forecasting Module Consolidation

**Date**: November 19, 2025  
**Status**: ⚠️ IN PROGRESS - Breaking Changes

---

## Problem Statement

Three separate forecasting modules existed with overlapping functionality:
1. `core/forecast.py` (795 lines) - Basic AR(1) detector and per-sensor forecasting
2. `core/enhanced_forecasting.py` (1,286 lines) - File-based enhanced forecasting
3. `core/enhanced_forecasting_sql.py` (329 lines) - SQL wrapper for enhanced forecasting

This caused:
- Confusion about which module to use
- Duplicate code maintenance
- Inconsistent behavior between file and SQL modes
- Import complexity in `acm_main.py`

---

## Solution: Single Unified Module

**New Structure**:
- ✅ `core/forecasting.py` - **Single source of truth** for all forecasting
- 📦 `core/forecast_deprecated.py` - Old AR(1) module (kept for reference)
- 📦 `core/enhanced_forecasting_deprecated.py` - Old file-based module (kept for reference)

---

## What Changed

### Files Renamed (Commit a7edfff)
```
core/forecast.py → core/forecast_deprecated.py
core/enhanced_forecasting.py → core/enhanced_forecasting_deprecated.py  
core/enhanced_forecasting_sql.py → core/forecasting.py
```

### Import Updates Needed

**OLD (acm_main.py)**:
```python
from core import forecast
from core import enhanced_forecasting
from core import enhanced_forecasting_sql

# Usage:
ar1_detector = forecast.AR1Detector()
enhanced_forecasting.EnhancedForecastingEngine()
enhanced_forecasting_sql.run_enhanced_forecasting_sql()
```

**NEW (acm_main.py)**:
```python
from core import forecasting

# Usage:
ar1_detector = forecasting.AR1Detector()  # TODO: Move AR1 into forecasting.py
forecasting.run_enhanced_forecasting_sql()  # Main entrypoint
```

---

## Migration Status

### ✅ Completed
- [x] Renamed old modules to `_deprecated.py`
- [x] Renamed `enhanced_forecasting_sql.py` → `forecasting.py`
- [x] Git commit preserves history with renames

### ⏳ TODO (Next Steps)
- [ ] Extract AR1Detector from `forecast_deprecated.py` 
- [ ] Add AR1Detector to `forecasting.py` as a helper class
- [ ] Update `acm_main.py` imports:
  - Remove `from core import forecast`
  - Remove `from core import enhanced_forecasting`
  - Remove `from core import enhanced_forecasting_sql`
  - Add `from core import forecasting`
- [ ] Update all `forecast.AR1Detector()` → `forecasting.AR1Detector()`
- [ ] Update all `enhanced_forecasting_sql.run_*` → `forecasting.run_*`
- [ ] Test pipeline end-to-end
- [ ] Delete deprecated files after validation

---

## AR1Detector Migration Plan

The AR1Detector class is still used in acm_main.py for per-sensor scoring:

**Current Usage Locations**:
- Line 1407: Cold-start AR1 detector (warmup mode)
- Line 1502: Regular AR1 detector (main pipeline)

**Migration Steps**:
1. Copy AR1Detector class from `forecast_deprecated.py` → `forecasting.py`
2. Simplify by removing unused features
3. Keep only essential methods: `__init__`, `fit`, `score`
4. Update docstrings to match consolidated module style

**Estimated Code**: ~150 lines (vs 795 in old module)

---

## Benefits After Consolidation

✅ **Single Import**: `from core import forecasting`  
✅ **Clear Entrypoint**: `forecasting.run_enhanced_forecasting_sql()`  
✅ **SQL-First**: Built for SQL mode, no file I/O dependencies  
✅ **Maintainable**: One module to update, test, and document  
✅ **Backward Compatible**: Deprecated files kept for reference  

---

## Testing Checklist

After completing migration, verify:
- [ ] `python -m core.acm_main --equip FD_FAN` runs without import errors
- [ ] AR1 detector scoring works (check `ar1_z` column in scores)
- [ ] Enhanced forecasting runs (check ACM_HealthForecast_TS table)
- [ ] Failure probabilities vary correctly (not identical)
- [ ] No references to old module names in logs

---

## Rollback Plan

If issues occur:
```bash
# Revert renames
git revert a7edfff

# Or manually:
mv core/forecasting.py core/enhanced_forecasting_sql.py
mv core/forecast_deprecated.py core/forecast.py
mv core/enhanced_forecasting_deprecated.py core/enhanced_forecasting.py
```

---

## Related Commits

- **a7edfff**: Fix Grafana dashboard + rename forecasting modules
- **3f260c4**: Remove cfg.rul.use_enhanced flag (made enhanced RUL default)
- **fe4476e**: Integrate enhanced RUL estimator

---

## Next Steps

1. Complete AR1Detector migration (copy → simplify → test)
2. Update acm_main.py imports
3. Run full batch test
4. Delete deprecated files after validation
5. Update documentation (README, docs/PROJECT_STRUCTURE.md)

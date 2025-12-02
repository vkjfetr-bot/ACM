# ACM SQL-Only Migration - Comprehensive Fix Plan

## Issue Summary
1. **File Logging**: Logger still creates files via `Console.set_output()` and `LOG_FILE` env var
2. **Artifacts Folder**: Extensive use of `artifacts/` folder for models, states, and outputs
3. **Broken Forecasting**: `run_enhanced_forecasting_sql` references undefined `engine` object

## Solution Architecture

### 1. SQL-Only Logging
- **Remove**: All file output logic from `utils/logger.py`
- **Keep**: stdout/stderr + SQL sink via `sql_logger.py`
- **Update**: `acm_main.py` to never call `Console.set_output()`

### 2. Remove Artifacts Folder
- **Model Storage**: Already in `ModelRegistry` table - remove filesystem fallback
- **Forecast State**: Store in `ACM_ForecastState` table (create if missing)
- **Config History**: Already in `ACM_Config` - remove file writes
- **Tables/Charts**: Already writing to SQL via `output_manager` - remove CSV generation

### 3. Fix Forecasting Engine
The issue is in `run_enhanced_forecasting_sql()` line ~1018:
```python
forecast_result = engine.forecaster.forecast(...)
```
But `engine` is never defined.

**Root Cause**: Incomplete refactoring from legacy `EnhancedForecastingEngine` class

**Solution**: The function already has access to `health_series` and `df_scores`. 
We need to:
- Create simple forecasting logic using exponential smoothing or AR models
- Calculate failure probabilities based on health threshold
- Estimate RUL from forecast trajectory
- Generate attribution from detector scores
- Write results to SQL tables

## Implementation Plan

### Phase 1: Remove File Logging (15 min)
1. Strip `_setup_file_output()` and `set_output()` from `utils/logger.py`
2. Remove `log_file` handling from `acm_main.py`
3. Ensure SQL sink is always enabled

### Phase 2: Remove Artifacts Dependencies (30 min)
1. Create `ACM_ForecastState` table if missing
2. Update `model_persistence.py` to use SQL-only for forecast state
3. Remove all `Path("artifacts")` references from `acm_main.py`
4. Update `forecasting.py` to not require `artifact_root`
5. Make `output_manager` skip CSV generation entirely

### Phase 3: Fix Forecasting Engine (45 min)
1. Implement proper forecasting logic in `run_enhanced_forecasting_sql()`
2. Use statsmodels or sklearn for exponential smoothing
3. Calculate failure probabilities from forecast trajectory
4. Estimate RUL using crossing-time analysis
5. Generate sensor attribution from detector contributions
6. Write all outputs to SQL tables

## Files to Modify
- `utils/logger.py` - Remove file output
- `core/acm_main.py` - Remove artifacts paths, log file config
- `core/forecasting.py` - Fix broken engine code
- `core/model_persistence.py` - SQL-only state persistence
- `core/output_manager.py` - Skip CSV generation
- `scripts/sql/create_forecast_state_table.sql` - New table (if needed)

## Testing
1. Run: `python -m core.acm_main --equip FD_FAN`
2. Verify: No files created in filesystem
3. Check: All dashboard panels populated from SQL

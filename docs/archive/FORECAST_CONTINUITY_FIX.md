# Fix for Stepped Forecast Appearance in Grafana Dashboard

## Problem Statement

The Health Forecast (Next 24h) chart in the Grafana dashboard was displaying a "stepped" appearance with green vertical bars instead of smooth forecast curves. This visual issue made the dashboard difficult to interpret.

## Root Cause Analysis

The stepped appearance was caused by **data continuity issues**:

1. **Forecast Gap**: Each pipeline run generates forecast data starting from the last timestamp + 1 hour. The `ACM_HealthForecast_TS` table only contained **future predictions**, not recent historical data.

2. **Batch Isolation**: When each new batch runs, it generates a fresh forecast starting from its own endpoint. Old forecasts are cleaned up (keeping only 2 most recent RunIDs).

3. **Visual Discontinuity**: Grafana displays this data with gaps between:
   - Historical health data (from `ACM_HealthTimeline`)
   - Current batch forecast (from `ACM_HealthForecast_TS`)
   - Previous batch forecast (if still retained)

The result: vertical jumps/steps in the visualization where data segments don't connect smoothly.

## Solution Implemented

### 1. Include Historical Health Data in Forecast Table

**Modified Files:**
- `core/rul_estimator.py` (lines ~402-441)
- `core/enhanced_rul_estimator.py` (lines ~628-700)

**Key Changes:**
- Added 24-hour lookback window to include recent historical health data
- Historical data is prepended to the forecast data in `ACM_HealthForecast_TS` and `ACM_FailureForecast_TS`
- Historical data uses actual health values with narrow confidence intervals (±2.0)
- Tagged historical rows with `Method = "Historical"` vs. `Method = "AR1_Health"` or `Method = "Ensemble_Adaptive"` for forecast rows

**Code Example (from enhanced_rul_estimator.py):**
```python
# CONTINUITY FIX: Include recent historical health data to bridge gap
lookback_hours = 24.0  # Include last 24h of history
cutoff_ts = forecast.index[0] - pd.Timedelta(hours=lookback_hours)
recent_history = hi[hi.index >= cutoff_ts].copy()

# Build forecast dataframe
forecast_df = pd.DataFrame({...})

# Build historical dataframe for recent data
if len(recent_history) > 0:
    hist_uncertainty = 2.0  # Small fixed uncertainty for historical data
    history_df = pd.DataFrame({
        "Timestamp": recent_history.index,
        "ForecastHealth": recent_history.values,
        "CiLower": np.clip(recent_history.values - hist_uncertainty, 0.0, 100.0),
        "CiUpper": np.clip(recent_history.values + hist_uncertainty, 0.0, 100.0),
        "ForecastStd": hist_uncertainty,
        "Method": "Historical",
        ...
    })
    # Concatenate history + forecast for seamless timeline
    health_forecast_df = pd.concat([history_df, forecast_df], ignore_index=True)
```

### 2. Cleanup Logic Verification

**Verified:**
- Cleanup logic runs BEFORE generating new forecast data (good timing)
- Keeps 2 most recent RunIDs by default (configurable via `ACM_FORECAST_RUNS_RETAIN` env var)
- No changes needed to cleanup logic

## Benefits

1. **Smooth Transitions**: Historical data provides bridge between past and forecast
2. **Visual Continuity**: No more vertical steps/gaps in dashboard charts
3. **Better Context**: Users can see actual recent performance alongside predictions
4. **Backward Compatible**: Existing queries work without modification
5. **Distinguishable Data**: Method column allows filtering historical vs forecast if needed

## Testing Instructions

### 1. Run Pipeline in File Mode

```powershell
# Quick test with GAS_TURBINE
python -m core.acm_main --equip GAS_TURBINE --enable-report

# Or use PowerShell shortcut
.\scripts\run_file_mode.ps1
```

### 2. Run Pipeline in SQL Mode

```powershell
# Ensure SQL connection configured in configs/sql_connection.ini
python -m core.acm_main --equip GAS_TURBINE --enable-report --mode sql
```

### 3. Verify Data in SQL (if using SQL mode)

```sql
-- Check that historical data is included
SELECT 
    Method, 
    COUNT(*) as RowCount,
    MIN(Timestamp) as FirstTimestamp,
    MAX(Timestamp) as LastTimestamp
FROM ACM_HealthForecast_TS
WHERE EquipID = <your_equip_id>
  AND RunID = (SELECT MAX(RunID) FROM ACM_HealthForecast_TS WHERE EquipID = <your_equip_id>)
GROUP BY Method
ORDER BY MIN(Timestamp);

-- Should show:
-- Method          | RowCount | FirstTimestamp (24h ago) | LastTimestamp (24h future)
-- Historical      | ~24-1440 | <24h before now>         | <now>
-- Ensemble_Adaptive | ~24-168 | <now + 1h>               | <now + 24h>
```

### 4. Verify Dashboard Visualization

1. Open Grafana dashboard: `grafana_dashboards/asset_health_dashboard.json`
2. Navigate to "Health Forecast (Next 24h)" panel
3. Select equipment (GAS_TURBINE or FD_FAN)
4. Set time range to include recent data + forecast window

**Expected Behavior:**
- ✅ Smooth continuous line from historical data through forecast
- ✅ Confidence intervals (CI Lower/Upper) shown as shaded area
- ✅ No vertical steps or gaps
- ✅ Clear transition point where historical becomes forecast (if distinguishing by Method)

**Problem Indicators (if still present):**
- ❌ Vertical green bars creating stepped appearance
- ❌ Gaps between data segments
- ❌ Disconnected forecast segments from different batches

### 5. Run Multiple Batches to Test Continuity

```powershell
# Run 3-5 sequential batches to verify smooth transitions
for ($i=1; $i -le 5; $i++) {
    Write-Host "Running batch $i..."
    python -m core.acm_main --equip GAS_TURBINE --enable-report
    Start-Sleep -Seconds 10
}
```

Check that each new forecast smoothly transitions from the previous run's historical data.

## Configuration Options

### Lookback Window

Currently hardcoded to 24 hours in both files. To adjust:

```python
# In core/rul_estimator.py and core/enhanced_rul_estimator.py
lookback_hours = 24.0  # Change this value (6.0, 12.0, 48.0, etc.)
```

### Historical Data Uncertainty

Currently set to ±2.0 health points. To adjust:

```python
# In core/rul_estimator.py and core/enhanced_rul_estimator.py
hist_uncertainty = 2.0  # Change this value for wider/narrower CI
```

### Forecast Retention

Control how many RunIDs to keep:

```powershell
# Windows PowerShell
$env:ACM_FORECAST_RUNS_RETAIN = "3"
python -m core.acm_main --equip GAS_TURBINE --enable-report
```

## Files Modified

1. **core/rul_estimator.py** (lines ~402-465)
   - Added historical data inclusion for health forecast
   - Added historical data inclusion for failure probability forecast
   - Imports scipy.stats.norm inline where needed

2. **core/enhanced_rul_estimator.py** (lines ~628-700)
   - Same changes as above for enhanced estimator
   - Includes model weights tracking for historical vs forecast data

## Rollback Plan

If issues arise, revert these specific sections:

1. In `core/rul_estimator.py`, replace lines 402-441 with original:
```python
health_forecast_df = pd.DataFrame(
    {
        "Timestamp": forecast.index,
        "ForecastHealth": forecast.values,
        "CiLower": ci_lower.values,
        "CiUpper": ci_upper.values,
        "ForecastStd": forecast_std,
        "Method": "AR1_Health",
    }
)
health_forecast_df = _insert_ids(health_forecast_df)
```

2. Similar revert for lines 448-465 (failure probability)
3. Repeat for `core/enhanced_rul_estimator.py` lines 628-700

## Future Enhancements

1. **Configurable Lookback**: Add to `configs/config_table.csv`:
   ```csv
   EquipID,forecasting.lookback_hours
   *,24.0
   GAS_TURBINE,48.0
   ```

2. **Smart Blending**: Weight recent historical data more heavily than older data

3. **Forecast Smoothing**: Apply moving average filter at historical-forecast boundary

4. **Dashboard Annotations**: Add vertical line marker showing where historical ends and forecast begins

## Related Documentation

- `docs/Analytics Backbone.md` - Overall analytics architecture
- `docs/BATCH_PROCESSING.md` - Batch mode and continuity patterns
- `core/rul_common.py` - Shared RUL/forecasting utilities
- `grafana_dashboards/asset_health_dashboard.json` - Dashboard configuration

## Contact

For questions or issues:
- Review `Task Backlog.md` for related items
- Check `CHANGELOG.md` for version history
- Consult `docs/ACM_WORKING_KNOWLEDGE_BASE.md` for system architecture

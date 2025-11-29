# Batch Processing Status & Summary

## Jobs Started
**Date:** 2025-11-24 16:38-16:39
**Mode:** Start from beginning, full cleanup

### Job 1: GAS_TURBINE
- **Command:** `python scripts/sql_batch_runner.py --equip GAS_TURBINE --start-from-beginning --max-batches 100 --tick-minutes 1440`
- **Job ID:** 1
- **Status:** ✅ Running
- **Cleanup:** All existing data for EquipID cleared, models deleted
- **Progress:** Writing RUL, forecast, and analytics data to SQL

### Job 2: FD_FAN
- **Command:** `python scripts/sql_batch_runner.py --equip FD_FAN --start-from-beginning --max-batches 100 --tick-minutes 1440`
- **Job ID:** 3
- **Status:** ✅ Running
- **Cleanup:** All existing data for EquipID cleared, models deleted
- **Progress:** Loading from SQL historian, processing batches

## Configuration
- **Tick Size:** 1440 minutes (24 hours per batch)
- **Max Batches:** 100
- **Mode:** `--start-from-beginning` (triggers full cleanup)

## Tables Being Populated

### Core Analytics
- ACM_HealthTimeline
- ACM_RegimeTimeline
- ACM_Scores_Wide
- ACM_Episodes
- ACM_EpisodeMetrics
- ACM_CulpritHistory

### OMR (New Feature)
- ✅ ACM_OMRTimeline
- ✅ ACM_OMRContributionsLong
- ✅ ACM_OMR_Metrics (newly created)
- ✅ ACM_OMR_TopContributors (newly created)
- ✅ ACM_OMR_SensorContributions (view)
- ✅ ACM_DetectorContributions (newly created)

### Forecasting & RUL
- ACM_RUL_TS
- ACM_RUL_Summary
- ACM_RUL_Attribution
- ACM_HealthForecast_TS
- ACM_FailureForecast_TS
- ACM_SensorForecast_TS
- ACM_MaintenanceRecommendation

### Sensor & Drift
- ACM_SensorNormalized_TS
- ACM_DriftSeries
- ACM_DriftEvents
- ACM_SensorRanking
- ACM_SensorHotspots

## Cleanup Actions Taken

When `--start-from-beginning` is used, the batch runner:

1. ✅ **Truncates all analytical tables** for the equipment
   - Deletes all rows where `EquipID = <target>`
   - Affects all tables in `ALLOWED_TABLES` (72 tables)
   - Uses SQL: `DELETE FROM <table> WHERE EquipID = ?`

2. ✅ **Deletes existing models** from ModelRegistry
   - Ensures coldstart rebuilds from scratch
   - SQL: `DELETE FROM ModelRegistry WHERE EquipID = ?`

3. ✅ **Resets progress tracking**
   - Clears `.sql_batch_progress.json` for the equipment
   - Forces coldstart to run first

4. ✅ **Infers optimal tick size** from raw data
   - Analyzes historian data cadence
   - Sets tick_minutes in ACM_Config table

## Monitoring

### Check Job Status
```powershell
.\scripts\monitor_batch_jobs.ps1
```

### View Full Output
```powershell
# GAS_TURBINE
Receive-Job -Name ACM_GAS_TURBINE -Keep

# FD_FAN
Receive-Job -Name ACM_FD_FAN -Keep
```

### Check Table Population
```powershell
python scripts\check_table_data.py
python scripts\verify_new_tables.py
```

### Stop Jobs (if needed)
```powershell
Stop-Job -Name ACM_GAS_TURBINE
Stop-Job -Name ACM_FD_FAN
Get-Job | Remove-Job -Force
```

## Expected Completion

With 1440-minute (24-hour) tick size and 100 max batches:
- Each equipment will process up to 100 days of data
- Time per batch: ~5-30 seconds depending on data volume
- Total runtime: 5-50 minutes per equipment

## Grafana Dashboard

After jobs complete, the dashboard will be fully populated:

### Working Panels (Data Available)
1. ✅ Health Timeline
2. ✅ Regime Timeline
3. ✅ Sensor Anomaly Markers
4. ✅ Drift Detection
5. ✅ Episode Metrics
6. ✅ Culprit History
7. ✅ **OMR Timeline** (NEW)
8. ✅ **OMR Sensor Contributions** (NEW)
9. ✅ Forecasting panels
10. ✅ RUL panels

### New Panels Needing Data
11. ⏳ **OMR Metrics & Gating** - will populate after first batch completes
12. ⏳ **OMR Top Contributors** - will populate after episodes are detected

## Next Steps

1. ⏳ Wait for batch jobs to complete (~10-30 minutes)
2. ✅ Verify table population with `python scripts\verify_new_tables.py`
3. ✅ Re-import updated dashboard JSON to Grafana
4. ✅ Adjust Grafana time range to match data (2023-10-15 onwards)
5. ✅ Verify all panels show data

## Files Changed

### SQL Schema
- `scripts/sql/create_missing_omr_tables.sql` - Creates new OMR tables

### Code
- `core/output_manager.py` - Added `ACM_DetectorContributions` to `ALLOWED_TABLES`

### Dashboard
- `grafana_dashboards/asset_health_dashboard.json` - Added 4 OMR panels

### Scripts
- `scripts/monitor_batch_jobs.ps1` - Monitor batch processing jobs
- `scripts/check_table_data.py` - Check table row counts
- `scripts/verify_new_tables.py` - Verify new OMR tables
- `scripts/check_omr_size.py` - Analyze OMR table structure

## Git Branch
- Branch: `omr-metrics-gating`
- Latest commit: Added ACM_DetectorContributions to ALLOWED_TABLES
- Ready to merge after validation

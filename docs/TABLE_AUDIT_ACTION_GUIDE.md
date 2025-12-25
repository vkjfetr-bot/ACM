# ACM Table Audit - Quick Action Guide

**Date:** 2024-12-25  
**Status:** ✅ ALLOWED_TABLES Updated - Partial Implementation Required

---

## TL;DR

The ACM table audit is **complete**. ALLOWED_TABLES has been expanded from 17 to 42 tables. 

**Current Status:**
- ✅ ~15 tables are already being written by existing code
- ⚠️ ~27 tables are now allowed but need implementation
- 📊 Result: Partial dashboard functionality (need to implement remaining tables)

---

## What Changed

### Before Audit
- 17 tables in ALLOWED_TABLES
- 42% dashboard coverage (broken dashboards)
- No operational visibility (no run tracking)

### After Audit  
- 42 tables in ALLOWED_TABLES ✅
- ~58% dashboard coverage (partially implemented)
- Partial operational visibility (Runs/RunMetrics/RefitRequests exist)

---

## Currently Implemented Tables (15 tables)

These tables are ALREADY being written by the code:

### From forecast_engine.py:
- ✅ ACM_RUL
- ✅ ACM_HealthForecast
- ✅ ACM_FailureForecast
- ✅ ACM_SensorForecast

### From acm_main.py:
- ✅ ACM_HealthTimeline
- ✅ ACM_RegimeTimeline
- ✅ ACM_DataQuality
- ✅ ACM_FeatureDropLog
- ✅ ACM_RefitRequests
- ✅ ACM_Runs
- ✅ ACM_RunMetrics

### From run_metadata_writer.py:
- ✅ ACM_RunMetadata (not in ALLOWED_TABLES but exists)
- ✅ ACM_RunTimers

### From episode_culprits_writer.py:
- ✅ ACM_EpisodeCulprits

### From config_history_writer.py:
- ✅ ACM_ConfigHistory (not in ALLOWED_TABLES but exists)

**Note:** ACM_Scores_Wide, ACM_Episodes, ACM_SensorDefects, ACM_SensorHotspots, ACM_EpisodeDiagnostics are likely also being written (check fuse.py and other modules).

---

## Tables Needing Implementation (27 tables)

These tables are NOW ALLOWED but need code to write them:

#### Priority 1: Logging Table (5 dashboards affected)
- [ ] **ACM_RunLogs** - Need to add SQL write for Console logs
  - Currently: Logs go to Console → Loki only
  - Need: Also write to ACM_RunLogs table for dashboard access
  - File: `core/observability.py` or create new log persistence layer

**Implementation approach:**
```python
# In observability.py or new core/log_writer.py
def write_log_to_sql(output_mgr, run_id, level, message, component, **kwargs):
    log_df = pd.DataFrame([{
        'RunID': run_id,
        'LogTime': pd.Timestamp.now(),
        'Level': level,
        'Component': component,
        'Message': message,
        'Details': json.dumps(kwargs)
    }])
    output_mgr.write_table('ACM_RunLogs', log_df)
```

#### Priority 2: Sensor Analytics Tables (2 dashboards affected)
- [ ] **ACM_ContributionCurrent** - Find and re-enable sensor contribution writes
- [ ] **ACM_ContributionTimeline** - Find and re-enable historical contribution writes
- [ ] **ACM_SensorHotspotTimeline** - Find and re-enable hotspot trend writes

**Files to check:**
```python
# core/fuse.py or core/acm_main.py
# Search for: contribution, hotspot
```

#### Priority 3: Regime & Drift Analytics (2 dashboards affected)
- [ ] **ACM_DriftSeries** - Re-enable drift time series writes
- [ ] **ACM_RegimeOccupancy** - Re-enable regime statistics writes
- [ ] **ACM_RegimeTransitions** - Re-enable regime transition writes
- [ ] **ACM_RegimeDwellStats** - Re-enable regime duration writes
- [ ] **ACM_RegimeStability** - Re-enable regime stability writes
- [ ] **ACM_HealthZoneByPeriod** - Re-enable health aggregate writes

**Files to check:**
```python
# core/drift.py - drift detection module
# core/regimes.py - regime detection module
# core/acm_main.py - main pipeline
```

#### Priority 4: Episode & Event Tracking (2 dashboards affected)
- [ ] **ACM_Anomaly_Events** - Create event-based anomaly tracking
- [ ] **ACM_EpisodeMetrics** - Re-enable episode metrics writes
- [ ] **ACM_DefectSummary** - Re-enable defect summary writes
- [ ] **ACM_ThresholdCrossings** - Re-enable threshold violation writes
- [ ] **ACM_AlertAge** - Re-enable alert aging writes

**Files to check:**
```python
# core/fuse.py - episode detection
# core/episode_culprits_writer.py - episode diagnostics
```

#### Priority 5: Operations Monitoring (1 dashboard affected)
- [ ] **ACM_ColdstartState** - Re-enable coldstart state writes
- [ ] **ACM_RefitRequests** - Already implemented? (check acm_main.py)

**Files to check:**
```python
# core/acm_main.py - search for coldstart, refit
```

#### Priority 6: Forecast Details (1 dashboard affected)
- [ ] **ACM_DetectorForecast_TS** - Re-enable detector-level forecast writes
- [ ] **ACM_FailureHazard_TS** - Re-enable hazard rate writes
- [ ] **ACM_HealthForecast_Continuous** - Re-enable continuous forecast writes

**Files to check:**
```python
# core/forecast_engine.py - forecasting module
# Search for: detector_forecast, hazard, continuous
```

#### Priority 7: Quality Metrics (no dashboards, but valuable)
- [ ] **ACM_DetectorCorrelation** - Re-enable detector correlation writes
- [ ] **ACM_CalibrationSummary** - Re-enable calibration writes
- [ ] **ACM_FeatureDropLog** - Already implemented? (check acm_main.py line 1738)

---

## How to Find Disabled Code

### Search Strategy

```bash
# Search for commented-out writes
cd /home/runner/work/ACM/ACM
grep -r "# output_mgr.write_table\|# write_table" core/

# Search for tables in git history
git log --all -p --grep="ACM_Runs\|ACM_RunLogs\|ACM_ContributionCurrent"

# Search for deleted write_table calls
git log --all -p -S "write_table.*ACM_Runs" -- core/

# Check what forecast_engine already writes
grep "write_table\|sql_table=" core/forecast_engine.py
```

### Expected Locations

| Table | Likely Location | Search Term |
|-------|----------------|-------------|
| ACM_Runs | run_metadata_writer.py | write_run_metadata |
| ACM_RunLogs | observability.py or acm_main.py | Console.* → SQL |
| ACM_RunTimers | run_metadata_writer.py | write_timer_stats |
| ACM_ContributionCurrent | fuse.py or acm_main.py | contribution |
| ACM_ContributionTimeline | fuse.py or acm_main.py | contribution |
| ACM_DriftSeries | drift.py | drift detection |
| ACM_RegimeOccupancy | regimes.py | regime stats |
| ACM_Anomaly_Events | fuse.py | episodes → events |

---

## Testing Strategy

### After Re-enabling Each Table

1. **Run single equipment test:**
```powershell
python -m core.acm_main --equip GAS_TURBINE
```

2. **Check if table was written:**
```sql
-- Connect to SQL Server
sqlcmd -S "localhost\INSTANCE" -d ACM -E

-- Check row count
SELECT COUNT(*) FROM ACM_Runs WHERE EquipID = 1
SELECT TOP 5 * FROM ACM_Runs ORDER BY CreatedAt DESC
```

3. **Verify dashboard shows data:**
- Open Grafana dashboard
- Check that panel shows recent data
- Verify no "No Data" errors

---

## Rollback Plan

If implementation proves too complex or breaks existing functionality:

```python
# In core/output_manager.py, revert to minimal set:
ALLOWED_TABLES = {
    # Keep only the 17 original tables
    'ACM_Scores_Wide',
    'ACM_HealthTimeline',
    'ACM_Episodes',
    'ACM_RegimeTimeline',
    'ACM_RUL',
    'ACM_HealthForecast',
    'ACM_FailureForecast',
    'ACM_SensorForecast',
    'ACM_DataQuality',
    'ACM_ForecastingState',
    'ACM_AdaptiveConfig',
    'ACM_SensorDefects',
    'ACM_SensorHotspots',
    'ACM_EpisodeCulprits',
    'ACM_EpisodeDiagnostics',
    'ACM_RegimeDefinitions',
    'ACM_ActiveModels',
}
```

---

## Success Criteria

✅ All 26 dashboard tables receiving fresh data  
✅ No performance degradation vs. 17-table baseline  
✅ All existing tests still pass  
✅ Dashboards load without errors  

---

## Timeline Estimate

| Phase | Tables | Effort | Duration |
|-------|--------|--------|----------|
| Priority 1 (Run management) | 3 | Medium | 1-2 days |
| Priority 2 (Sensor analytics) | 3 | Medium | 1-2 days |
| Priority 3 (Regime/drift) | 6 | High | 2-3 days |
| Priority 4 (Episodes/events) | 5 | Medium | 1-2 days |
| Priority 5 (Operations) | 2 | Low | 0.5-1 day |
| Priority 6 (Forecasts) | 3 | Medium | 1-2 days |
| Priority 7 (Quality metrics) | 3 | Low | 0.5-1 day |
| **Total** | **25 tables** | | **7-13 days** |

---

## References

- **Full audit report:** `docs/ACM_TABLE_ANALYTICS_AUDIT.md`
- **Table inventory:** `docs/TABLE_AUDIT.md`
- **ALLOWED_TABLES definition:** `core/output_manager.py` lines 56-108
- **Dashboard specifications:** `grafana_dashboards/*.json`

---

## Questions?

Check the comprehensive audit report for:
- Detailed impact assessment
- What was lost in cleanup
- Dashboard-by-dashboard table usage
- Orphaned table categorization
- Long-term recommendations

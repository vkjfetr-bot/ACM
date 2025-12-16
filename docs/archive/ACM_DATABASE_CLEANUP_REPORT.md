# ACM Database Cleanup Status Report
**Date**: December 4, 2025

## Summary
- **Total Tables**: 85
- **Status**: Mostly clean with some cleanup items remaining

---

## ✅ CLEANED UP
1. **Foreign Keys**: No invalid foreign keys detected
2. **Schema Consistency**: All tables have proper structure
3. **Stored Procedures**: Fixed `usp_ACM_FinalizeRun` to reference correct tables
4. **Equipment Table**: 6 records (FD_FAN, GAS_TURBINE, etc.)

---

## ⚠️ CLEANUP NEEDED

### 1. **Backup Tables (Safe to Delete)**
- `PCA_Components_BACKUP_20251203` - 1,160 rows
- `RunLog_BACKUP_20251203` - 1,871 rows  
- `Runs_BACKUP_20251203` - 3,951 rows

**Action**: These are dated backups from Dec 3. Safe to delete if you have other backups.

### 2. **Empty/Unused Tables (0-2 records)**
- `ACM_Drift_TS` - 0 rows (no drift data being written)
- `ACM_EnhancedFailureProbability_TS` - 0 rows (unused detector)
- `ACM_EnhancedMaintenanceRecommendation` - 0 rows (not populated)
- `ACM_FailureCausation` - 0 rows (abandoned feature)
- `ACM_Forecast_QualityMetrics` - 0 rows (not integrated)
- `ACM_HealthForecast_Continuous` - 0 rows (superseded by TS version)

**Action**: Consider these deprecated or not yet implemented. Safe to delete if not planned.

### 3. **State/Config Tables with Minimal Data (2 records)**
- `ACM_ColdstartState` - 2 rows (expected, tracks coldstart attempts)
- `ACM_MaintenanceRecommendation` - 2 rows (expected, sparse)
- `ACM_RegimeState` - 2 rows (expected, regime tracking)
- `ACM_RUL_LearningState` - 2 rows (expected, learning state)
- `ACM_SchemaVersion` - 2 rows (expected, schema tracking)

**Status**: ✅ Normal - these are state/configuration tables

### 4. **Incomplete Runs (4 records)**
- 4 runs in `ACM_Runs` have NULL `CompletedAt` timestamp
- These are NOOP runs that didn't complete finalization

**Action**: Can be marked with status or cleaned up if old

### 5. **Data Quality Issues Found**
- **EquipID Inconsistency**: 
  - GAS_TURBINE has EquipID=2621 in one run
  - GAS_TURBINE has EquipID=2621 in another (CORRECT)
  - But one FD_FAN run shows EquipID=1 (CORRECT)
  
- **Data Naming Issues**:
  - Some runs have EquipName='Gas Turbine Generator' (full name)
  - Others have EquipName='GAS_TURBINE' (code name)
  - Should standardize to equipment code

**Action**: Standardize equipment naming in ETL

---

## 📊 Table Statistics (Top Active Tables)

```
ACM_Scores_Wide          - 27,642 rows (main scoring output)
ACM_Episodes             - 15 rows (episode detection)
ACM_OMRContributionsLong - 17,431 rows (sensor contributions)
ACM_HealthTimeline       - 1,356 rows (health progression)
ACM_RegimeTimeline       - 649 rows (regime tracking)
ACM_RUL_TS               - 1,188 rows (RUL calculations)
ACM_HealthForecast_TS    - 594 rows (health forecasts)
ACM_FailureForecast_TS   - 572 rows (failure forecasts)
```

---

## 🔧 Recommended Actions (Priority Order)

### High Priority
1. ✅ **Fix FinalizeRun stored procedure** - DONE
2. **Delete backup tables** (if you have external backups):
   ```sql
   DROP TABLE PCA_Components_BACKUP_20251203;
   DROP TABLE RunLog_BACKUP_20251203;
   DROP TABLE Runs_BACKUP_20251203;
   ```

3. **Standardize Equipment Names** in ETL pipeline:
   - Use equipment code consistently (FD_FAN, GAS_TURBINE)
   - Not display names or mixed formats

### Medium Priority
4. **Delete unused table structs** (if not planned):
   ```sql
   DROP TABLE ACM_Drift_TS;
   DROP TABLE ACM_EnhancedFailureProbability_TS;
   DROP TABLE ACM_EnhancedMaintenanceRecommendation;
   DROP TABLE ACM_FailureCausation;
   DROP TABLE ACM_Forecast_QualityMetrics;
   DROP TABLE ACM_HealthForecast_Continuous;
   ```

5. **Mark incomplete runs** with error status or timestamp

### Low Priority
6. Archive old backup tables before deletion
7. Add data validation rules to prevent future inconsistencies

---

## Conclusion
✅ **Database is mostly clean** with ~85% of tables actively used and well-structured.
⚠️ **Cleanup items are minimal and optional** - backup tables and unused feature stubs.
🎯 **Main issue**: Equipment naming inconsistency in ETL - should fix in ACM pipeline.

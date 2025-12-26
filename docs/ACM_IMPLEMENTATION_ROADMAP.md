# ACM Implementation Roadmap - Tables + Refactoring

**Date:** December 25, 2025  
**Status:** Master tracking document for v11 completion  
**Merged From:** ACM_OUTPUT_TABLES_REFINED.md, ACM Main Refactoring Analysis, ACM_TABLES_COMPLETE_AUDIT.md

---

## Executive Summary

**Date Updated:** December 26, 2025  
**Goal:** Ensure all 42 ACM tables are created and populated through a properly refactored `acm_main.py`

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| **Tables in SQL** | 92 | 42 | ✅ Exceeded (many extra tables) |
| **Core Tables with data** | 35+ | 42 | 7-10 empty tables need population |
| **Helper functions extracted** | 22 | 30+ | 8 remaining |
| **Phase functions** | 0 | 7 | Not started |
| **acm_main.py lines** | ~4,900 | <300 | 93% reduction needed |

### Key Progress (December 26, 2025)
- ✅ ACM_SensorForecast: Now populating (1,512 rows) - sensor forecasting fixed
- ✅ ACM_SensorNormalized_TS: Now populating (472,446 rows) - new table
- ✅ ACM_EpisodeDiagnostics: Now populating (425 rows)
- ✅ ACM_DetectorCorrelation: Now populating (3,577 rows)
- ✅ ACM_SensorCorrelations: Now populating (191,844 rows)
- ✅ ACM_PCA_Metrics: Now populating (17 rows)

### Remaining Work
- **Empty Tables (need population logic)**: ACM_DriftSeries, ACM_FeatureDropLog, ACM_CalibrationSummary, ACM_RegimeOccupancy, ACM_RegimeTransitions, ACM_ContributionTimeline, ACM_RegimePromotionLog, ACM_DriftController, ACM_RegimeDefinitions, ACM_ActiveModels, ACM_DataContractValidation, ACM_SeasonalPatterns, ACM_AssetProfiles
- **Multivariate Forecast Error**: `MultivariateForecast` shape mismatch needs fix

---

## Part 1: Table Implementation Status

### TIER 1: Current State (6 tables) - User Question: "What is current health?"

| Table | SQL Exists | Rows | Schema in table_schemas.py | Status |
|-------|-----------|------|---------------------------|--------|
| ACM_HealthTimeline | ✅ | 27,689 | ✅ | ✅ COMPLETE |
| ACM_Scores_Wide | ✅ | 135,005 | ✅ | ✅ COMPLETE |
| ACM_Episodes | ✅ | 157 | ✅ | ✅ COMPLETE |
| ACM_RegimeTimeline | ✅ | 133,322 | ✅ | ✅ COMPLETE |
| ACM_SensorDefects | ✅ | 2,654 | ⚠️ Verify | ✅ COMPLETE |
| ACM_SensorHotspots | ✅ | 8,069 | ⚠️ Verify | ✅ COMPLETE |

**Status:** ✅ **ALL 6 TABLES EXIST AND POPULATED**

---

### TIER 2: Future State (4 tables) - User Question: "What will future health look like?"

| Table | SQL Exists | Rows | Schema in table_schemas.py | Status |
|-------|-----------|------|---------------------------|--------|
| ACM_RUL | ✅ | 359 | ✅ | ✅ COMPLETE |
| ACM_HealthForecast | ✅ | 91,056 | ✅ | ✅ COMPLETE |
| ACM_FailureForecast | ✅ | 91,056 | ✅ | ✅ COMPLETE |
| ACM_SensorForecast | ✅ | 1512 | ✅ | ✅ COMPLETE |

**Status:** ✅ **ALL 4 TABLES COMPLETE**

---

### TIER 3: Root Cause (6 tables) - User Question: "Why is this happening?"

| Table | SQL Exists | Rows | Schema in table_schemas.py | Status |
|-------|-----------|------|---------------------------|--------|
| ACM_EpisodeCulprits | ✅ | 5,108 | ⚠️ Add | ✅ COMPLETE |
| ACM_EpisodeDiagnostics | ✅ | 425 | ✅ | ✅ COMPLETE |
| ACM_DetectorCorrelation | ✅ | 3,577 | ⚠️ Add | ✅ COMPLETE |
| ACM_DriftSeries | ✅ | 0 | ⚠️ Add | ⚠️ EXISTS (empty) |
| ACM_SensorCorrelations | ✅ | 191,844 | ⚠️ Add | ✅ COMPLETE |
| ACM_FeatureDropLog | ✅ | 0 | ⚠️ Add | ⚠️ EXISTS (empty) |

**Status:** ✅ **4 of 6 POPULATED - 2 exist but empty (DriftSeries, FeatureDropLog)**

---

### TIER 4: Data & Model Management (10 tables)

| Table | SQL Exists | Rows | Schema in table_schemas.py | Status |
|-------|-----------|------|---------------------------|--------|
| ACM_BaselineBuffer | ✅ | 1,162,924 | ⚠️ Add | ✅ COMPLETE |
| ACM_HistorianData | ✅ | 204,067 | ⚠️ Add | ✅ COMPLETE |
| ACM_SensorNormalized_TS | ✅ | 472,446 | ✅ | ✅ COMPLETE |
| ACM_DataQuality | ✅ | 7,195 | ⚠️ Add | ✅ COMPLETE |
| ACM_ForecastingState | ✅ | 7 | ⚠️ Add | ✅ COMPLETE |
| ACM_CalibrationSummary | ✅ | 0 | ⚠️ Add | ⚠️ EXISTS (empty) |
| ACM_AdaptiveConfig | ✅ | 11 | ⚠️ Add | ✅ COMPLETE |
| ACM_RefitRequests | ✅ | 76 | ⚠️ Add | ✅ COMPLETE |
| ACM_PCA_Metrics | ✅ | 17 | ⚠️ Add | ✅ COMPLETE |
| ACM_RunMetadata | ✅ | 715 | ⚠️ Add | ✅ COMPLETE |

**Status:** ✅ **9 of 10 POPULATED - 1 exists but empty (CalibrationSummary)**

---

### TIER 5: Operations & Audit (6 tables)

| Table | SQL Exists | Rows | Schema in table_schemas.py | Status |
|-------|-----------|------|---------------------------|--------|
| ACM_Runs | ✅ | 716 | ❌ Formalize | ✅ COMPLETE |
| ACM_RunLogs | ✅ | 656,958 | ❌ Add | ✅ COMPLETE |
| ACM_RunTimers | ✅ | 203,050 | ❌ Add | ✅ COMPLETE |
| ACM_Config | ✅ | 342 | ❌ External | ✅ COMPLETE |
| ACM_ConfigHistory | ✅ | 3,495 | ❌ Add | ✅ COMPLETE |
| ACM_RunMetrics | ✅ | 104,064 | ❌ Add | ✅ COMPLETE |

**Status:** ✅ **ALL 6 TABLES EXIST AND POPULATED**

---

### TIER 6: Advanced Analytics (5 tables)

| Table | SQL Exists | Rows | Schema in table_schemas.py | Status |
|-------|-----------|------|---------------------------|--------|
| ACM_RegimeOccupancy | ✅ | 0 | ⚠️ Add | ⚠️ EXISTS (empty) |
| ACM_RegimeTransitions | ✅ | 0 | ⚠️ Add | ⚠️ EXISTS (empty) |
| ACM_ContributionTimeline | ✅ | 0 | ⚠️ Add | ⚠️ EXISTS (empty) |
| ACM_RegimePromotionLog | ✅ | 0 | ⚠️ Add | ⚠️ EXISTS (empty) |
| ACM_DriftController | ✅ | 0 | ⚠️ Add | ⚠️ EXISTS (empty) |

**Status:** ⚠️ **ALL 5 EXIST BUT EMPTY - Need population logic**

---

### TIER 7: V11 Features (5 tables)

| Table | SQL Exists | Rows | Schema in table_schemas.py | Status |
|-------|-----------|------|---------------------------|--------|
| ACM_RegimeDefinitions | ✅ | 0 | ✅ | ⚠️ EXISTS (empty) |
| ACM_ActiveModels | ✅ | 0 | ✅ | ⚠️ EXISTS (empty) |
| ACM_DataContractValidation | ✅ | 0 | ✅ | ⚠️ EXISTS (empty) |
| ACM_SeasonalPatterns | ✅ | 0 | ✅ | ⚠️ EXISTS (empty) |
| ACM_AssetProfiles | ✅ | 0 | ✅ | ⚠️ EXISTS (empty) |

**Status:** ⚠️ **ALL 5 EXIST BUT EMPTY - Need population logic**

---

## Part 2: Refactoring Status (from ACM Main Refactoring Analysis)

### Completed Work

| Wave | Description | Status | Result |
|------|-------------|--------|--------|
| Wave 1 | Dead code removal | ✅ COMPLETE | -335 lines (7.2%) |
| Wave 2 | File-mode branch removal | ✅ COMPLETE | All file-mode code removed |
| Wave 3 | Helper extraction | ✅ 22 helpers done | Testable structure added |

### 22 Extracted Helpers (All Done)

1. `_score_all_detectors()` - Score data through all detectors
2. `_calibrate_all_detectors()` - Calibrate detector outputs
3. `_fit_all_detectors()` - Fit all detectors on training data
4. `_get_detector_enable_flags()` - Get enable flags from fusion weights
5. `_deduplicate_index()` - Remove duplicate timestamps
6. `_rebuild_detectors_from_cache()` - Reconstruct detectors from cached models
7. `_update_baseline_buffer()` - Update baseline buffer with vectorized SQL writes
8. `_compute_stable_feature_hash()` - Stable cross-platform hash for training data
9. `_check_refit_request()` - Check and acknowledge SQL refit requests
10. `_load_cached_models_with_validation()` - Load and validate cached models
11. `_save_trained_models()` - Save trained models with versioning
12. `_write_fusion_metrics()` - Write fusion diagnostics to ACM_RunMetrics
13. `_log_dropped_features()` - Log dropped features to ACM_FeatureDropLog
14. `_write_data_quality()` - Write data quality metrics to ACM_DataQuality
15. `_normalize_episodes_schema()` - Normalize episodes with timestamps, regimes
16. `_write_pca_artifacts()` - Write PCA model, loadings, metrics
17. `_compute_drift_alert_mode()` - Compute drift alert mode
18. `_build_data_quality_records()` - Build per-sensor data quality records
19. `_build_health_timeline()` - Build health timeline DataFrame
20. `_build_regime_timeline()` - Build regime timeline DataFrame
21. Context dataclasses (RuntimeContext, DataContext, etc.)
22. Additional utility helpers

### Remaining Helper Extractions (8 more)

| Helper | Location | Lines | Priority |
|--------|----------|-------|----------|
| `_build_features()` | 3101-3175 | ~75 | Medium |
| `_impute_features()` | 3178-3240 | ~65 | Low |
| `_seed_baseline()` | 2850-2950 | ~100 | Medium |
| `_build_drift_ts()` | 4816-4830 | ~15 | Low |
| `_build_anomaly_events()` | 4835-4852 | ~18 | Low |
| `_build_regime_episodes()` | 4857-4870 | ~14 | Low |
| `_write_sensor_defects()` | Various | ~50 | High (for TIER 1) |
| `_write_regime_analytics()` | Various | ~80 | Medium (for TIER 6) |

### Phase Functions (Not Started - Wave 4)

| Phase | Function | Lines | Purpose |
|-------|----------|-------|---------|
| 1 | `_phase_initialize_runtime()` | ~343 | Startup, config, SQL connection |
| 2 | `_phase_load_data()` | ~732 | SmartColdstart, data validation |
| 3 | `_phase_fit_models()` | ~396 | Train/load detector models |
| 4 | `_phase_label_regimes()` | ~262 | Operating mode detection |
| 5 | `_phase_calibrate()` | ~174 | Calibrate detector outputs |
| 6 | `_phase_fuse_and_episodes()` | ~650 | Multi-detector fusion, episodes |
| 7 | `_phase_persist_results()` | ~490 | Write all outputs to SQL |

---

## Part 3: Implementation Priorities

### Summary: 15 Tables to Create

| Priority | Tables | Tier | Impact |
|----------|--------|------|--------|
| **P1** | EpisodeDiagnostics, DetectorCorrelation, DriftSeries, SensorCorrelations, FeatureDropLog | 3 | Root cause visibility |
| **P2** | SensorNormalized_TS, CalibrationSummary, PCA_Metrics | 4 | Model quality tracking |
| **P3** | RegimeOccupancy, RegimeTransitions, ContributionTimeline, RegimePromotionLog, DriftController | 6 | Advanced analytics |
| **P4** | RegimeDefinitions, ActiveModels, DataContractValidation, SeasonalPatterns, AssetProfiles | 7 | V11 features |

### Priority 1: TIER 3 Root Cause Tables (5 tables) - HIGH IMPACT

These tables answer "Why is this happening?" - critical for troubleshooting.

**Tasks:**
1. [ ] Create `ACM_EpisodeDiagnostics` - Schema exists in table_schemas.py
2. [ ] Create `ACM_DetectorCorrelation` - Add schema + create table
3. [ ] Create `ACM_DriftSeries` - Add schema + create table
4. [ ] Create `ACM_SensorCorrelations` - Add schema + create table
5. [ ] Create `ACM_FeatureDropLog` - Add schema + create table (helper `_log_dropped_features()` exists)

**SQL DDL Example:**
```sql
-- ACM_EpisodeDiagnostics (from table_schemas.py)
CREATE TABLE dbo.ACM_EpisodeDiagnostics (
    DiagID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    EpisodeID INT NOT NULL,
    RunID NVARCHAR(50),
    StartTime DATETIME NOT NULL,
    EndTime DATETIME,
    DurationHours FLOAT,
    Severity NVARCHAR(20),
    PeakZ FLOAT,
    CreatedAt DATETIME DEFAULT GETDATE()
);
```

### Priority 2: TIER 4 Model Quality Tables (3 tables) - MEDIUM IMPACT

**Tasks:**
1. [x] Create `ACM_SensorNormalized_TS` - ✅ COMPLETE (78,741 rows)
2. [ ] Create `ACM_CalibrationSummary` - Model quality over time
3. [ ] Create `ACM_PCA_Metrics` - PCA component metrics (write_pca_metrics() exists but table missing)

### Priority 3: TIER 6 Advanced Analytics (5 tables) - FUTURE

**Tasks:**
1. [ ] Create `ACM_RegimeOccupancy` - Operating mode utilization
2. [ ] Create `ACM_RegimeTransitions` - Mode switching patterns
3. [ ] Create `ACM_ContributionTimeline` - Historical sensor attribution
4. [ ] Create `ACM_RegimePromotionLog` - Regime maturity tracking
5. [ ] Create `ACM_DriftController` - Drift detection control

### Priority 4: TIER 7 V11 Features (5 tables) - V11 COMPLETION

Schemas exist in table_schemas.py, just need SQL tables.

**Tasks:**
1. [ ] Create `ACM_RegimeDefinitions` - Regime centroids and metadata
2. [ ] Create `ACM_ActiveModels` - Active model versions
3. [ ] Create `ACM_DataContractValidation` - Pipeline validation
4. [ ] Create `ACM_SeasonalPatterns` - Seasonal pattern detection
5. [ ] Create `ACM_AssetProfiles` - Asset similarity profiles

---

## Part 4: Investigation & Validation

### Investigate ACM_SensorForecast (0 rows)

Table exists but is empty. Possible causes:
1. Forecasting disabled in config
2. Write method not being called
3. Insufficient historical data

**Investigation:**
```powershell
# Check if forecast_engine writes to this table
grep -n "SensorForecast" core/forecast_engine.py
grep -n "SensorForecast" core/output_manager.py
```

### DDL Generation Strategy

**Option 1: Generate from table_schemas.py**
```python
# scripts/sql/generate_table_ddl.py
from core.table_schemas import TABLE_SCHEMAS

def generate_ddl():
    for table_name, schema in TABLE_SCHEMAS.items():
        print(f"CREATE TABLE dbo.{table_name} (...")
```

**Option 2: Manual DDL for missing tables**
```powershell
# Export current schema to understand what exists
python scripts/sql/export_comprehensive_schema.py --output artifacts/current_schema.md
```

### Database Views (Recommended - Create After Tables)

| View | Purpose | Source Tables |
|------|---------|---------------|
| ACM_CurrentHealth_View | Latest health per equipment | ACM_HealthTimeline |
| ACM_ActiveAnomalies_View | Currently active anomalies | ACM_Episodes, ACM_EpisodeCulprits |
| ACM_LatestRUL_View | Most recent RUL per equipment | ACM_RUL |
| ACM_ProblematicSensors_View | Sensors with issues | ACM_SensorDefects, ACM_DataQuality |
| ACM_FleetSummary_View | Fleet-wide health summary | Multiple tables |

### Validation Checklist (After Each Sprint)

```powershell
# 1. Run integration test
python scripts/sql_batch_runner.py --equip FD_FAN --start-from-beginning --max-workers 1 --max-batches 2

# 2. Verify tables populated
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "
SELECT t.name AS TableName, SUM(p.rows) AS TotalRows 
FROM sys.tables t JOIN sys.partitions p ON t.object_id = p.object_id 
WHERE t.name LIKE 'ACM_%' AND p.index_id IN (0,1) 
GROUP BY t.name ORDER BY t.name"

# 3. Check for errors
sqlcmd -S "localhost\B19CL3PCQLSERVER" -d ACM -E -Q "
SELECT TOP 10 LoggedAt, Level, Message FROM ACM_RunLogs 
WHERE Level IN ('ERROR','WARN') ORDER BY LoggedAt DESC"
```

---

## Part 5: Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| SQL Tables | 27 | 42 | `SELECT COUNT(*) FROM sys.tables WHERE name LIKE 'ACM_%'` |
| Tables with data | 26 | 42 | Row count query above |
| table_schemas.py entries | 14 | 42 | `grep -c "TableSchema" core/table_schemas.py` |
| OutputManager write methods | ~10 | 30+ | `grep -c "def write_" core/output_manager.py` |
| acm_main.py lines | ~4,900 | <300 | `wc -l core/acm_main.py` |
| Integration test outcome | OK | OK | Batch runner exit code 0 |

---

## Appendix: Quick Reference

### Table Count by Tier (Actual vs Target)

| Tier | Name | Target | Exist | Gap |
|------|------|--------|-------|-----|
| 1 | Current State | 6 | 6 | ✅ 0 |
| 2 | Future State | 4 | 4 | ⚠️ 0 (1 empty) |
| 3 | Root Cause | 6 | 1 | ❌ 5 |
| 4 | Data & Model | 10 | 7 | ❌ 3 |
| 5 | Operations | 6 | 6 | ✅ 0 |
| 6 | Advanced Analytics | 5 | 0 | ❌ 5 |
| 7 | V11 Features | 5 | 0 | ❌ 5 |
| **TOTAL** | | **42** | **27** | **15** |

### 14 Missing Tables (Sorted by Priority)

**P1 - Root Cause (5):**
1. ACM_EpisodeDiagnostics
2. ACM_DetectorCorrelation
3. ACM_DriftSeries
4. ACM_SensorCorrelations
5. ACM_FeatureDropLog

**P2 - Model Quality (2):**
~~6. ACM_SensorNormalized_TS~~ ✅ COMPLETE
6. ACM_CalibrationSummary
7. ACM_PCA_Metrics

**P3 - Advanced Analytics (5):**
9. ACM_RegimeOccupancy
10. ACM_RegimeTransitions
11. ACM_ContributionTimeline
12. ACM_RegimePromotionLog
13. ACM_DriftController

**P4 - V11 Features (5):**
14. ACM_RegimeDefinitions
15. ACM_ActiveModels
16. ACM_DataContractValidation
17. ACM_SeasonalPatterns
18. ACM_AssetProfiles

### Key Files to Modify

| File | Purpose | Changes Needed |
|------|---------|----------------|
| `core/table_schemas.py` | Schema definitions | Add ~28 missing schemas |
| `core/output_manager.py` | Write methods | Add ~15 write methods |
| `core/acm_main.py` | Pipeline | Use new write methods, extract phases |
| `scripts/sql/create_acm_tables.sql` | DDL | Create 15 new tables |

---

**Next Step:** Generate DDL script to create the 15 missing tables, starting with TIER 3 (Root Cause).

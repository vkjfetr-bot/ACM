# ACM Table Write Audit Report

**Generated**: 2025-01-13  
**Version**: v11.0.3  
**Purpose**: Track table write method status in ALLOWED_TABLES

---

## Executive Summary

| Category | Count | Notes |
|----------|-------|-------|
| **ACTIVE** | 45 | Tables with working write paths |
| **ORPHANED** | 1 | Write method exists but never called |
| **EXTERNAL** | 6 | Written by external processes/stored procs |

**Status**: As of v11.0.3, only 1 write method remains orphaned: `write_feature_drop_log()`.

---

## ORPHANED TABLES (1) - TO FIX

| Table | Write Method | Line | Expected Data | Fix Location |
|-------|--------------|------|---------------|--------------|
| `ACM_FeatureDropLog` | `write_feature_drop_log()` | 2689 | Features dropped during engineering | After PHASE 7: FEATURE ENGINEERING |

---

## RECENTLY FIXED (v11.0.3)

The following methods were wired up and are now called from `acm_main.py`:

| Table | Write Method | Called At |
|-------|--------------|-----------|
| `ACM_CalibrationSummary` | `write_calibration_summary()` | acm_main.py:4981 |
| `ACM_RegimeOccupancy` | `write_regime_occupancy()` | acm_main.py:4552 |
| `ACM_RegimeTransitions` | `write_regime_transitions()` | acm_main.py:4567 |
| `ACM_ContributionTimeline` | `write_contribution_timeline()` | acm_main.py:5563 |
| `ACM_RegimePromotionLog` | `write_regime_promotion_log()` | acm_main.py:4779 |
| `ACM_DriftController` | `write_drift_controller()` | acm_main.py:5391 |

---

## ACTIVE TABLES - Working Correctly

All write methods are called from the pipeline. See `acm_main.py` for call sites.

### Key Tables

| Table | Write Method | Purpose |
|-------|--------------|---------|
| `ACM_Scores_Wide` | `write_scores()` | Detector z-scores |
| `ACM_Episodes` | `write_episodes()` | Anomaly episodes |
| `ACM_HealthTimeline` | `generate_comprehensive_analytics()` | Health scores over time |
| `ACM_RegimeTimeline` | `generate_comprehensive_analytics()` | Regime labels over time |
| `ACM_RUL` | `ForecastEngine._write_rul()` | Remaining useful life |
| `ACM_SensorDefects` | `generate_comprehensive_analytics()` | Active sensor defects |
| `ACM_SensorHotspots` | `generate_comprehensive_analytics()` | Top contributing sensors |
| `ACM_CalibrationSummary` | `write_calibration_summary()` | Model calibration metrics |
| `ACM_RegimeOccupancy` | `write_regime_occupancy()` | Regime time distribution |
| `ACM_RegimeTransitions` | `write_regime_transitions()` | Regime transition matrix |
| `ACM_ContributionTimeline` | `write_contribution_timeline()` | Sensor contributions |
| `ACM_RegimePromotionLog` | `write_regime_promotion_log()` | Maturity state changes |
| `ACM_DriftController` | `write_drift_controller()` | Drift thresholds |

---

## EXTERNAL TABLES (6)

| Table | External Writer |
|-------|----------------|
| `ACM_BaselineBuffer` | `sql_batch_runner.py` coldstart logic |
| `ACM_HistorianData` | Historian integration / stored procedure |
| `ACM_Config` | `scripts/sql/populate_acm_config.py` |
| `ACM_ConfigHistory` | `config_history_writer.py` triggered by config changes |
| `ACM_RunLogs` | Observability stack (Loki) via `Console` class |
| `ACM_RefitRequests` | Model lifecycle manager on drift detection |

---

## Pending Fix: ACM_FeatureDropLog

**Purpose**: Document why features were dropped during engineering  
**Location to add call**: `acm_main.py` after feature engineering (PHASE 7)  
**Data source**: Features with low variance, high NaN rate, etc.

```python
# After feature engineering in _build_features()
dropped_features = [
    {'feature': 'sensor_X_lag1', 'reason': 'low_variance', 'value': 0.001},
    ...
]
output_manager.write_feature_drop_log(dropped_features)
```

---

## Verification Query

```sql
SELECT t.name AS TableName, 
       p.rows AS RowCount,
       CASE WHEN p.rows = 0 THEN 'EMPTY' ELSE 'HAS DATA' END AS Status
FROM sys.tables t
INNER JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
WHERE t.name LIKE 'ACM_%'
ORDER BY p.rows ASC, t.name;
```

---

## Appendix: Write Method Status (v11.0.3)

| Method | Target Table | Called? |
|--------|--------------|---------|
| `write_calibration_summary()` | ACM_CalibrationSummary | ✅ YES |
| `write_regime_occupancy()` | ACM_RegimeOccupancy | ✅ YES |
| `write_regime_transitions()` | ACM_RegimeTransitions | ✅ YES |
| `write_contribution_timeline()` | ACM_ContributionTimeline | ✅ YES |
| `write_regime_promotion_log()` | ACM_RegimePromotionLog | ✅ YES |
| `write_drift_controller()` | ACM_DriftController | ✅ YES |
| `write_feature_drop_log()` | ACM_FeatureDropLog | ❌ NO |

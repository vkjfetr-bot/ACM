# SQL Mode Continuous Learning - Data-Driven Retrain Implementation

**Date**: 2025-11-29  
**Branch**: feature/continuous-learning-architecture  
**Status**: Tasks 1 & 2 Complete, Task 3 Partially Complete

---

## Overview

Implemented comprehensive data-driven model retraining logic for SQL mode, addressing critical gaps where models only retrained on config changes. The system now monitors data quality metrics (anomaly rate, drift score, model age, regime quality) and triggers retraining automatically when thresholds are exceeded.

---

## Changes Implemented

### 1. Configuration Infrastructure ✅

Added 11 new configuration parameters to `configs/config_table.csv`:

```csv
models,auto_retrain.max_anomaly_rate,0.25,float
models,auto_retrain.max_drift_score,2.0,float
models,auto_retrain.max_model_age_hours,720,int
models,auto_retrain.min_regime_quality,0.3,float
models,auto_retrain.on_tuning_change,False,bool
models,max_model_age_days,30,int
models,max_model_age_hours,720,int
models,use_cache,True,bool
runtime,reuse_model_fit,True,bool
continuous_learning,enable_auto_tune,True,bool
continuous_learning,enable_quality_monitoring,True,bool
```

**Impact**: Provides tunable thresholds for all retrain triggers without code changes.

---

### 2. Quality Check Section Enhancement ✅

**File**: `core/acm_main.py` (lines 2058-2115)

**Changes**:
- Added model age validation (SQL mode temporal checks)
- Added regime quality validation (silhouette score thresholds)
- Multi-trigger aggregation: `config_changed OR model_age_trigger OR regime_quality_trigger`
- Invalidates cached models when any trigger fires
- Comprehensive logging of trigger reasons

**Code Sample**:
```python
# Check model age (SQL mode temporal validation)
model_age_trigger = False
if SQL_MODE and cached_manifest:
    created_at_str = cached_manifest.get("created_at")
    if created_at_str:
        model_age_hours = (datetime.now() - created_at).total_seconds() / 3600
        max_age_hours = auto_retrain_cfg.get("max_model_age_hours", 720)
        if model_age_hours > max_age_hours:
            model_age_trigger = True
            Console.warn(f"[RETRAIN-TRIGGER] Model age {model_age_hours:.1f}h exceeds {max_age_hours}h")
```

**Impact**: Prevents stale models from being reused indefinitely in SQL mode.

---

### 3. Auto-Tune Section Enhancement ✅

**File**: `core/acm_main.py` (lines 2729-2883)

**Changes**:
- Removed `if not SQL_MODE:` guard - auto-tune now runs in SQL mode
- Extract anomaly_rate and drift_score from `assess_model_quality` results
- Added anomaly_rate_trigger (threshold: 25% by default)
- Added drift_score_trigger (threshold: 2.0 by default)
- Aggregate all triggers: `needs_retraining = should_retrain OR anomaly_rate_trigger OR drift_score_trigger`
- Enhanced refit flag with detailed metrics

**Code Sample**:
```python
# Check anomaly rate trigger
anomaly_rate_trigger = False
anomaly_metrics = quality_report.get("metrics", {}).get("anomaly_metrics", {})
current_anomaly_rate = anomaly_metrics.get("anomaly_rate", 0.0)
max_anomaly_rate = auto_retrain_cfg.get("max_anomaly_rate", 0.25)
if current_anomaly_rate > max_anomaly_rate:
    anomaly_rate_trigger = True
    reasons.append(f"anomaly_rate={current_anomaly_rate:.2%} > {max_anomaly_rate:.2%}")
    Console.warn(f"[RETRAIN-TRIGGER] Anomaly rate {current_anomaly_rate:.2%} exceeds threshold {max_anomaly_rate:.2%}")
```

**Impact**: Models retrain when data quality degrades, not just on config changes.

---

### 4. SQL Mode Refit Request Logging ⏳

**File**: `core/acm_main.py` (lines 2864-2883)

**Changes**:
- Added SQL mode specific logging for refit requests
- Log detailed reasons including anomaly_rate and drift_score values
- Placeholder for ACM_RefitRequests table (to be implemented in Task 3)

**Code Sample**:
```python
else:
    # SQL mode: write to ACM_RefitRequests (to be implemented in Task 3)
    Console.warn("[MODEL] SQL mode refit request: ACM_RefitRequests table not yet implemented")
    Console.warn(f"[MODEL] Refit reasons: {', '.join(reasons)}")
    if anomaly_rate_trigger:
        Console.warn(f"[MODEL] Current anomaly rate: {current_anomaly_rate:.2%} (threshold: {max_anomaly_rate:.2%})")
    if drift_score_trigger:
        Console.warn(f"[MODEL] Current drift score: {drift_score:.2f} (threshold: {max_drift_score:.2f})")
```

**Status**: Partially complete - logging in place, SQL table creation pending.

---

## Task Status

### ✅ Task 1: Data-Driven Model Retrain Triggers - COMPLETE
- ✅ Anomaly rate trigger implemented
- ✅ Drift score trigger implemented  
- ✅ Model age trigger implemented
- ✅ Regime quality trigger implemented
- ✅ Config parameters added
- ✅ Multi-trigger aggregation logic

### ✅ Task 2: Wire assess_model_quality Results - COMPLETE
- ✅ Extract anomaly_metrics from quality_report
- ✅ Extract drift_score from quality_report
- ✅ Use metrics to drive force_retrain decisions
- ✅ Enhanced logging of trigger reasons
- ✅ Removed SQL_MODE guards from auto-tune

### ⏳ Task 3: SQL-Native Refit Mechanism - PARTIALLY COMPLETE
- ✅ Enhanced refit flag file with detailed metrics
- ✅ SQL mode specific logging for refit requests
- ⏳ ACM_RefitRequests table not yet created
- ⏳ Run-start query logic not yet implemented
- ⏳ Acknowledged flag clearing not yet implemented

---

## Testing Required

### Validation Steps

1. **Batch Mode Test** (20+ runs with continuous learning):
   ```powershell
   $env:ACM_BATCH_MODE=1
   $env:ACM_BATCH_NUM=1
   python -m core.acm_main --equip GAS_TURBINE --enable-report
   ```
   - Monitor logs for `[RETRAIN-TRIGGER]` messages
   - Verify models retrain when thresholds exceeded
   - Check ACM_ThresholdMetadata for evolving thresholds

2. **Anomaly Rate Trigger Test**:
   - Inject high anomaly rate data (>25%)
   - Verify `anomaly_rate_trigger = True`
   - Confirm refit flag written with anomaly_rate metric

3. **Drift Score Trigger Test**:
   - Simulate gradual process drift
   - Verify `drift_score_trigger = True` when score > 2.0
   - Confirm drift_score logged in refit reasons

4. **Model Age Trigger Test**:
   - Modify cached model `created_at` timestamp to be >720 hours old
   - Verify `model_age_trigger = True`
   - Confirm cached model invalidated

5. **Regime Quality Trigger Test**:
   - Force low silhouette score (<0.3)
   - Verify `regime_quality_trigger = True`
   - Confirm model retraining triggered

---

## Remaining Work

### Task 3: Complete SQL-Native Refit Mechanism

**SQL Table Schema**:
```sql
CREATE TABLE ACM_RefitRequests (
    RequestID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    RequestedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
    Reason NVARCHAR(MAX),
    AnomalyRate FLOAT NULL,
    DriftScore FLOAT NULL,
    ModelAgeHours FLOAT NULL,
    RegimeQuality FLOAT NULL,
    Acknowledged BIT DEFAULT 0,
    AcknowledgedAt DATETIME2 NULL,
    CONSTRAINT FK_RefitRequests_Equipment FOREIGN KEY (EquipID) REFERENCES ACM_Equipment(EquipID)
);

CREATE INDEX IX_RefitRequests_EquipID_Acknowledged ON ACM_RefitRequests(EquipID, Acknowledged);
```

**Implementation Steps**:
1. Create migration script: `scripts/sql/migrations/007_create_refit_requests.sql`
2. Add write logic in `core/acm_main.py` (replace placeholder logging)
3. Add read logic at run start to check for pending refit requests
4. Implement acknowledged flag update after processing
5. Test full refit request lifecycle

**Estimated Effort**: 3-4 hours

---

## Impact Assessment

### Before
- Models only retrained on config changes
- SQL mode had no temporal validation on cached models
- Data drift, anomaly spikes, regime quality degradation were ignored
- "Continuous learning" was effectively "config-driven retraining"

### After
- Models retrain on 5 distinct triggers:
  1. Config changes
  2. Model age exceeding threshold (720h default)
  3. Regime quality degradation (silhouette < 0.3)
  4. High anomaly rate (>25%)
  5. High drift score (>2.0)
- SQL mode now participates fully in quality monitoring
- Detailed logging of all trigger reasons
- Foundation for SQL-native refit request system

### Metrics to Monitor
- Retraining frequency in batch mode (expect more frequent retrains)
- Threshold evolution in ACM_ThresholdMetadata
- Model age distribution in ModelRegistry
- Anomaly rate trends in quality reports
- Drift score trends in quality reports

---

## Configuration Tuning Guide

### Anomaly Rate Threshold (`models.auto_retrain.max_anomaly_rate`)
- **Default**: 0.25 (25%)
- **Conservative**: 0.15 (15%) - More frequent retraining
- **Permissive**: 0.35 (35%) - Less frequent retraining
- **Guidance**: If false alarms are high, increase threshold; if faults missed, decrease

### Drift Score Threshold (`models.auto_retrain.max_drift_score`)
- **Default**: 2.0
- **Conservative**: 1.5 - Retrain on smaller shifts
- **Permissive**: 3.0 - Tolerate more drift
- **Guidance**: Typical drift_z values are 0-3; values >2 indicate significant process change

### Model Age Threshold (`models.auto_retrain.max_model_age_hours`)
- **Default**: 720 (30 days)
- **Conservative**: 168 (7 days) - Frequent refresh
- **Permissive**: 2160 (90 days) - Long-term stable processes
- **Guidance**: High-variability processes should use shorter windows

### Regime Quality Threshold (`models.auto_retrain.min_regime_quality`)
- **Default**: 0.3
- **Conservative**: 0.4 - Higher quality requirement
- **Permissive**: 0.2 - Accept lower silhouette scores
- **Guidance**: Silhouette scores: >0.5 good, 0.3-0.5 acceptable, <0.3 poor

---

## Related Documentation

- **Task Backlog**: `Task Backlog.md` (lines 145-225) - Updated with completion status
- **Analytics Backbone**: `docs/Analytics Backbone.md` - Retrain trigger architecture
- **Cold Start Mode**: `docs/COLDSTART_MODE.md` - Model initialization and caching
- **Config Table Spec**: `configs/config_table.csv` - New auto_retrain parameters

---

## Commit Summary

**Branch**: feature/continuous-learning-architecture

**Files Modified**:
- `core/acm_main.py` (lines 2058-2115, 2729-2883)
- `configs/config_table.csv` (+11 rows)
- `Task Backlog.md` (updated task status)
- `SQL_MODE_RETRAIN_IMPROVEMENTS.md` (this document)

**Key Commits**:
- Enhanced quality check with model age and regime quality triggers
- Implemented anomaly_rate and drift_score triggers in auto-tune section
- Added 11 auto_retrain config parameters
- Removed SQL_MODE guards from auto-tune logic
- Enhanced refit flag with detailed metrics
- Added SQL mode refit request logging (placeholder for Task 3)

**Next Steps**:
1. Test batch mode with 20+ runs
2. Implement ACM_RefitRequests table (Task 3)
3. Continue with Task 4: Disable legacy joblib cache in SQL mode

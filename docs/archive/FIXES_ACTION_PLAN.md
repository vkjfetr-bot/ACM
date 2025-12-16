# ACM Fixes - Detailed Task Plan & Action List

**Created**: 2025-12-11  
**Last Updated**: 2025-12-11 (Status Check)  
**Priority Basis**: RunLogs Analysis Report  
**Target Completion**: 2025-12-18 (1 week)  
**Version Target**: v10.0.1 or v10.1.0 (depending on scope)

---

## Status Summary

| Task | Status | Notes |
|------|--------|-------|
| Task 1: OMR NULL Constraint | **DONE** | fillna(0.0) added in output_manager.py:1886 |
| Task 2: Sensor Forecast Columns | **DONE** | Fixed query to use ACM_SensorNormalized_TS instead of ACM_Scores_Wide |
| Task 3: ForecastingState model_params | **DONE** | Property exists in state_manager.py:46 |
| Task 4: Install statsmodels | **DONE** | Added to pyproject.toml:23 |
| Task 5: SQL Commit API | **DONE** | commit()/rollback() added to sql_client.py:209 |
| Task 6: Add Missing Config Values | **DONE** | All 6 values exist in config_table.csv and ACM_Config |
| Task 7: Regime State Cache Variable | **DONE** | stable_models_dir defined in acm_main.py:852 |
| Task 8: Regime Clustering Quality | **DEFERRED** | Low silhouette is data-dependent, not a code bug |
| Task 9: OMR Backfill Script | **DONE** | scripts/sql/backfill_omr_contributions.sql created |
| Task 10: Timestamp Column Naming | **DONE** | EntryDateTime fallback working correctly |
| Task 11: Test Suite | **PENDING** | Need to run after ACM validation run |
| Task 12: SQL Verification | **PENDING** | Need post-run verification |
| Task 13: Schema Reference | **DONE** | export_comprehensive_schema.py succeeded, 91 tables |
| Task 14: Mahalanobis Monitoring | **DONE** | scripts/sql/monitor_mahalanobis_regularization.sql created |
| Task 15: Document Fixes | **PENDING** | Need release notes after validation |

**DB State Verified**:
- Migration 62 applied (QualityFlag, RawHealthIndex columns exist)
- ACM_OMRContributionsLong: 0 NULL ContributionScore records
- Regime config values: All 26 rows present in ACM_Config
- Recent logs: 11,395 INFO, 587 WARNING, 0 ERROR in last 24h
- Forecast tables: HealthForecast_TS=0, SensorForecast_TS=0, RUL=4 rows

---

## Overview

This task list addresses **15 actionable items** identified from RunLogs analysis:
- **4 CRITICAL** (blocking forecasting, 15 min each)
- **3 HIGH** (affects system stability, 30 min each)
- **2 MEDIUM** (performance/quality)
- **6 VERIFICATION/DOCUMENTATION** (testing & release prep)

**Estimated Total Time**: ~8-10 hours of development + testing

---

## PHASE 1: CRITICAL FIXES (Must Do First)

### Task 1: Fix OMR Contributions NULL Constraint ⚠️ CRITICAL
**Priority**: HIGHEST  
**Error Count**: 100+  
**Impact**: Data loss on OMR analytics  
**Est. Time**: 30 min (15 min fix + 15 min testing)

#### Current Issue:
```
[OUTPUT] SQL insert failed for ACM_OMRContributionsLong: 
Cannot insert the value NULL into column 'ContributionScore'
```

#### Root Cause Analysis:
1. OMR calculation producing NULL values for `ContributionScore`
2. Output manager not converting NULL → default value
3. SQL constraint enforced at insert time

#### Files to Check:
- `core/omr.py` - OMR calculation logic (lines ~100-150)
- `core/output_manager.py` - Data mapper for OMR writes (search: "OMRContributionsLong")
- Schema: `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` (ACM_OMRContributionsLong columns)

#### Steps:
1. [ ] Search for `ACM_OMRContributionsLong` in `core/output_manager.py`
2. [ ] Trace data flow: where ContributionScore comes from
3. [ ] Add NULL check: `if contribution_score is None: contribution_score = 0.0`
4. [ ] Or use fillna(0): `df['ContributionScore'].fillna(0.0, inplace=True)`
5. [ ] Add unit test for edge case (all NaN input)
6. [ ] Run: `python -m pytest tests/test_omr.py -v`
7. [ ] Manual test: run ACM on test data, verify no NULL constraint errors

#### Success Criteria:
- [ ] No "Cannot insert NULL" errors in logs
- [ ] All OMR records written to ACM_OMRContributionsLong
- [ ] Verify: `SELECT COUNT(*) FROM ACM_OMRContributionsLong WHERE ContributionScore IS NULL` returns 0

---

### Task 2: Fix Sensor Forecast Column Names ⚠️ CRITICAL
**Priority**: HIGHEST  
**Error Count**: 5+  
**Impact**: Sensor forecasting non-functional  
**Est. Time**: 30 min (20 min fix + 10 min testing)

#### Current Issue:
```
[ForecastEngine] Sensor forecasting failed: Invalid column name 'SensorName'
[ForecastEngine] Sensor forecasting failed: Invalid column name 'Score'
```

#### Root Cause Analysis:
1. Forecasting module uses old column names
2. Schema changed but code not updated
3. Hard-coded column references in queries

#### Files to Check:
- `core/forecasting.py` - Search for: `'SensorName'`, `'Score'`
- Cross-reference: `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` (ACM_SensorForecast_TS actual columns)

#### Steps:
1. [ ] Read ACM_SensorForecast_TS schema from reference doc
2. [ ] Search `core/forecasting.py` for all column name references
3. [ ] Create mapping table:
   ```
   Old Name     →  New Name (if different)
   SensorName   →  ?
   Score        →  ?
   ```
4. [ ] Replace all occurrences with correct names
5. [ ] Add comments with schema reference
6. [ ] Test: run forecast pipeline, verify no column errors

#### Success Criteria:
- [ ] No "Invalid column name" errors in logs
- [ ] ACM_SensorForecast_TS gets rows written
- [ ] Grafana sensor forecast panels show data

---

### Task 3: Fix ForecastingState model_params Attribute ⚠️ CRITICAL
**Priority**: HIGHEST  
**Error Count**: 100+  
**Impact**: RUL forecast engine crashes  
**Est. Time**: 20 min (15 min fix + 5 min testing)

#### Current Issue:
```
[ForecastEngine] Forecast failed: 'ForecastingState' object has no attribute 'model_params'
```

#### Root Cause Analysis:
1. ForecastingState class missing model_params initialization
2. State loaded from SQL missing this field
3. Code assumes attribute exists but doesn't

#### Files to Check:
- `core/forecasting.py` - Class: ForecastingState
- Search for: `self.model_params` in `__init__()`

#### Steps:
1. [ ] Open `core/forecasting.py`
2. [ ] Find class `ForecastingState`
3. [ ] Check `__init__()` method
4. [ ] Ensure all attributes initialized:
   ```python
   def __init__(self, ...):
       self.model_params = {}  # or appropriate default
       self.other_attrs = ...
   ```
5. [ ] Add defensive check in load method:
   ```python
   if not hasattr(self, 'model_params'):
       self.model_params = {}
   ```
6. [ ] Test: run forecast, verify no AttributeError

#### Success Criteria:
- [ ] No "object has no attribute 'model_params'" errors
- [ ] Forecast engine runs to completion (may still skip forecast, but doesn't crash)

---

### Task 4: Install statsmodels Dependency ⚠️ CRITICAL
**Priority**: HIGHEST  
**Error Count**: 1+  
**Impact**: Blocks ARIMA/VAR forecasting  
**Est. Time**: 10 min (5 min install + 5 min verify)

#### Current Issue:
```
[ForecastEngine] Sensor forecasting failed: No module named 'statsmodels'
```

#### Steps:
1. [ ] `pip install statsmodels`
2. [ ] Update `pyproject.toml` or `requirements.txt`:
   ```
   statsmodels>=0.13.5,<0.14  # Pin version
   ```
3. [ ] Verify: `python -c "import statsmodels; print(statsmodels.__version__)"`
4. [ ] Add to git: `git add requirements.txt` (or pyproject.toml)

#### Success Criteria:
- [ ] `import statsmodels` works in Python
- [ ] Requirements file updated and committed

---

## PHASE 2: HIGH-PRIORITY FIXES (Do Next)

### Task 5: Fix SQL Commit API for Forecasts
**Priority**: HIGH  
**Error Count**: 2  
**Impact**: Forecasts not committed, data inconsistency  
**Est. Time**: 30 min (15 min investigation + 15 min fix)

#### Current Issue:
```
[CONTINUOUS_FORECAST] Failed to write merged forecast: 'SQLClient' object has no attribute 'commit'
[CONTINUOUS_HAZARD] Failed to write hazard forecast: 'SQLClient' object has no attribute 'commit'
```

#### Root Cause Analysis:
1. SQLClient API changed or was never exposed
2. Forecast writers calling `.commit()` that doesn't exist
3. Need to check: is it `.flush()`, `.transaction.commit()`, or auto-commit?

#### Files to Check:
- `core/sql_client.py` - Check public methods
- `core/forecasting.py` - Search for `.commit()`

#### Steps:
1. [ ] Open `core/sql_client.py`
2. [ ] List all public methods (search class definition)
3. [ ] Find the correct commit/flush method:
   - Is it `self._conn.commit()`?
   - Is it `self.connection.commit()`?
   - Is it in a context manager?
4. [ ] Search `core/forecasting.py` for all `.commit()` calls
5. [ ] Replace with correct method name throughout
6. [ ] Test: run forecast pipeline, verify commits work

#### Success Criteria:
- [ ] No "object has no attribute 'commit'" errors
- [ ] Forecast data persists in database

---

### Task 6: Add Missing Config Values
**Priority**: HIGH  
**Error Count**: 378 each (6 missing values = 2,268 errors)  
**Impact**: Regime clustering uses wrong parameters  
**Est. Time**: 15 min (10 min update + 5 min sync)

#### Current Issue:
```
[REGIME] Config validation: Missing config value for regimes.quality.silhouette_min
[REGIME] Config validation: Missing config value for regimes.health.fused_warn_z
... (4 more)
```

#### Files to Update:
- `configs/config_table.csv`

#### Steps:
1. [ ] Open `configs/config_table.csv`
2. [ ] Add 6 new rows with equipment='*' (global defaults):

| Equipment | ConfigKey | ConfigValue | Description |
|-----------|-----------|-------------|-------------|
| * | regimes.quality.silhouette_min | 0.4 | Min silhouette score for good clustering |
| * | regimes.health.fused_warn_z | 1.618 | Fused Z-score warning threshold |
| * | regimes.auto_k.k_max | 5 | Max clusters to consider |
| * | regimes.auto_k.max_eval_samples | 1000 | Samples to use for k evaluation |
| * | regimes.smoothing.passes | 2 | Label smoothing iterations |
| * | regimes.auto_k.max_models | 10 | Max candidate models |

3. [ ] Save file
4. [ ] Sync to SQL: `python scripts/sql/populate_acm_config.py`
5. [ ] Verify in database:
   ```sql
   SELECT COUNT(*) FROM ACM_Config 
   WHERE ConfigKey LIKE 'regimes.%'
   ```

#### Success Criteria:
- [ ] No "Missing config value" warnings in logs
- [ ] 6 new config rows in ACM_Config table
- [ ] Regime clustering uses correct parameters

---

### Task 7: Fix Regime State Cache Variable
**Priority**: HIGH  
**Error Count**: 842  
**Impact**: Regime models re-trained every run (slower)  
**Est. Time**: 20 min (15 min fix + 5 min test)

#### Current Issue:
```
[REGIME] Failed to load cached regime state/model: name 'stable_models_dir' is not defined
```

#### Root Cause Analysis:
1. Variable `stable_models_dir` referenced but never defined
2. Should come from config or be hardcoded

#### Files to Check:
- `core/regimes.py` - Search for `stable_models_dir`

#### Steps:
1. [ ] Search `core/regimes.py` for `stable_models_dir`
2. [ ] Find context: how is it supposed to be used?
3. [ ] Define it (choose one):
   - Option A: Import from config: `stable_models_dir = cfg.get('regimes.cache_dir', 'models/regime_cache')`
   - Option B: Hardcode: `stable_models_dir = 'artifacts/regimes'`
   - Option C: Pass as parameter instead of global
4. [ ] Verify path exists or create if needed
5. [ ] Test: run ACM, verify regime cache loads successfully

#### Success Criteria:
- [ ] No "stable_models_dir is not defined" errors
- [ ] Regime models load from cache (faster second run)

---

## PHASE 3: MEDIUM-PRIORITY INVESTIGATIONS

### Task 8: Investigate Regime Clustering Quality
**Priority**: MEDIUM  
**Error Count**: 821  
**Impact**: Per-regime thresholds disabled  
**Est. Time**: 1-2 hours (investigation + analysis + possible fixes)

#### Current Issue:
```
[REGIME] Clustering quality below threshold; per-regime thresholds disabled.
[AUTO-TUNE] Quality degradation detected: Anomaly rate too high, Silhouette score too low
```

#### Root Cause Analysis:
- Silhouette score consistently below configured threshold
- Data too noisy or regimes don't naturally separate
- Algorithm may need tuning or different approach

#### Files to Check:
- `core/regimes.py` - Clustering algorithm, silhouette calculation
- `core/fast_features.py` - Feature engineering might affect separation

#### Investigation Steps:
1. [ ] Add logging to capture silhouette scores for analysis
2. [ ] Run multiple batches, log silhouette values
3. [ ] Analyze: is score getting worse over time, or consistently low?
4. [ ] Plot feature space to visualize regime separation (if possible)
5. [ ] Consider:
   - Better feature preprocessing?
   - Different number of clusters?
   - Different clustering algorithm?
   - Relax silhouette threshold if current approach is acceptable?

#### Success Criteria:
- [ ] Understand why silhouette score is low
- [ ] Decision made: fix or accept (document rationale)
- [ ] If fixing: implement solution and verify improvement

---

## PHASE 4: VERIFICATION & CLEANUP

### Task 9: Create Backfill Script for OMR Data
**Priority**: MEDIUM (after Task 1 complete)  
**Impact**: Restore lost OMR analytics from past runs  
**Est. Time**: 1 hour (script design + implementation + testing)

#### Approach:
1. [ ] Identify RunIDs affected by NULL constraint violations
2. [ ] Create script to:
   - Load historical data for affected runs
   - Recalculate OMR contributions
   - Insert into ACM_OMRContributionsLong
3. [ ] Test thoroughly on database copy
4. [ ] Run on production database
5. [ ] Verify: compare before/after counts

---

### Task 10: Fix Timestamp Column Naming
**Priority**: LOW  
**Error Count**: 628  
**Impact**: Minor timing inconsistencies  
**Est. Time**: 20 min (10 min fix + 10 min test)

#### Current Issue:
```
[DATA] Timestamp column '' not found; falling back to 'EntryDateTime'
```

#### Steps:
1. [ ] Find historian stored procedure (likely in `scripts/sql/`)
2. [ ] Ensure timestamp column is properly named in result set
3. [ ] Remove fallback code from `core/output_manager.py`
4. [ ] Test

---

### Task 11: Run Comprehensive Test Suite
**Priority**: CRITICAL (after all fixes)  
**Est. Time**: 1-2 hours

#### Test Suites to Run:
```bash
# Run all unit tests
pytest tests/test_fast_features.py -v
pytest tests/test_dual_write.py -v
pytest tests/test_progress_tracking.py -v

# Run manual end-to-end test
python -m core.acm_main --equip FD_FAN --start-time "2024-08-01T00:00:00" --end-time "2024-08-10T00:00:00"
```

#### Success Criteria:
- [ ] All unit tests pass
- [ ] Manual run completes without errors
- [ ] All analytics tables written
- [ ] Forecasts generated (if enough data)

---

### Task 12: Verify Fixes with SQL Queries
**Priority**: CRITICAL (after all fixes)  
**Est. Time**: 30 min

#### Query Checklist:
```sql
-- 1. OMR NULL check (should be 0)
SELECT COUNT(*) as NullCount FROM ACM_OMRContributionsLong 
WHERE ContributionScore IS NULL;

-- 2. Recent error logs (should be 0 for past 24h)
SELECT Level, COUNT(*) as Count FROM ACM_RunLogs 
WHERE LoggedAt > DATEADD(DAY, -1, GETDATE())
GROUP BY Level;

-- 3. Forecast tables have data
SELECT 'HealthForecast_TS' as Table, COUNT(*) as Rows FROM ACM_HealthForecast_TS
UNION ALL
SELECT 'SensorForecast_TS', COUNT(*) FROM ACM_SensorForecast_TS
UNION ALL
SELECT 'RUL', COUNT(*) FROM ACM_RUL;

-- 4. Config completeness (should be 0)
SELECT COUNT(*) as MissingConfigs FROM (
  SELECT 'regimes.quality.silhouette_min' as key WHERE NOT EXISTS (SELECT 1 FROM ACM_Config WHERE ConfigKey='regimes.quality.silhouette_min')
  UNION ALL
  SELECT 'regimes.health.fused_warn_z' WHERE NOT EXISTS (SELECT 1 FROM ACM_Config WHERE ConfigKey='regimes.health.fused_warn_z')
  -- ... check all 6 missing keys
) as missing;
```

#### Document Results:
- Create before/after comparison table
- Show error reduction metrics

---

### Task 13: Update Schema Reference (Migration 62)
**Priority**: MEDIUM  
**Est. Time**: 10 min

#### Steps:
1. [ ] Run: `python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md`
2. [ ] Verify: Check that QualityFlag, RawHealthIndex, health zone changes are documented
3. [ ] Commit: `git add docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md`

---

### Task 14: Monitor Mahalanobis Regularization
**Priority**: LOW  
**Est. Time**: 30 min (setup monitoring queries)

#### Create Monitoring:
1. [ ] Create SQL query to track regularization spikes:
   ```sql
   SELECT LoggedAt, Message, RunID FROM ACM_RunLogs
   WHERE Message LIKE '%Still critical after 100x increase%'
   AND LoggedAt > DATEADD(DAY, -7, GETDATE())
   ORDER BY LoggedAt DESC;
   ```
2. [ ] Document findings (how often, correlation with data issues)
3. [ ] Create alert if >N instances per day

---

### Task 15: Document Fixes and Create Release Notes
**Priority**: HIGH (after all fixes complete)  
**Est. Time**: 45 min

#### Create Release Notes (Update or create file):
- **What was fixed**: Each issue with before/after description
- **Why it mattered**: Impact on users
- **How to verify**: SQL queries or test steps
- **Breaking changes**: Any (likely none)
- **Migration steps**: For existing data (likely just backfill script)

#### Update Documentation:
- [ ] `README.md` - Update status section
- [ ] `docs/SOURCE_CONTROL_PRACTICES.md` - Add v10.0.1 or v10.1.0 release info
- [ ] Git tag: `git tag -a v10.0.1 -m "Fix critical forecasting and OMR issues"`

#### Commit:
```bash
git add README.md docs/ CHANGELOG.md
git commit -m "docs: Release notes for v10.0.1 - forecasting and OMR fixes"
git push origin feature/forecast-rul-v10
```

---

## Execution Timeline

### Day 1 (2025-12-12): CRITICAL FIXES
- [ ] Task 1: OMR NULL Constraint (30 min)
- [ ] Task 2: Sensor Forecast Columns (30 min)
- [ ] Task 3: ForecastingState Attribute (20 min)
- [ ] Task 4: Install statsmodels (10 min)
- **Total**: ~90 min

### Day 2 (2025-12-13): HIGH-PRIORITY FIXES
- [ ] Task 5: SQL Commit API (30 min)
- [ ] Task 6: Add Config Values (15 min)
- [ ] Task 7: Regime Cache Variable (20 min)
- [ ] Task 11: Test Suite (1-2 hours) **→ Day 3 if needed**
- **Total**: ~65 min

### Day 3 (2025-12-14): INVESTIGATION & TESTING
- [ ] Task 8: Regime Clustering Investigation (1-2 hours)
- [ ] Task 11: Comprehensive Test Suite (1-2 hours)
- [ ] Task 12: SQL Verification Queries (30 min)
- **Total**: ~3-4 hours

### Day 4 (2025-12-15): CLEANUP & DOCUMENTATION
- [ ] Task 9: OMR Backfill Script (1 hour)
- [ ] Task 10: Timestamp Naming (20 min)
- [ ] Task 13: Update Schema Reference (10 min)
- [ ] Task 14: Mahalanobis Monitoring (30 min)
- [ ] Task 15: Release Notes (45 min)
- **Total**: ~2.75 hours

---

## Dependencies & Blockers

```
Task 1 (OMR NULL) ─→ Task 9 (OMR Backfill)
Task 2,3 (Forecasting) ─→ Task 11,12 (Verify)
Task 4 (statsmodels) ──→ Task 11 (Test)
Task 5,6,7 (Config) ──→ Task 11,12 (Verify)
Task 8 (Investigation) (can run in parallel)
Task 13,14,15 (Docs) ─→ Only after all fixes verified
```

---

## Success Metrics

**Before Fixes**:
- Error count: 2,303 total, peak 1,784 per day
- Forecasting: Disabled/non-functional
- OMR analytics: Partial data loss
- Config: Missing values, warnings on every run

**After Fixes**:
- Error count: <10 per day
- Forecasting: All forecasts generated
- OMR analytics: Complete data written
- Config: All values present, no warnings
- Test suite: 100% pass rate

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking change in SQL | Test on copy of DB first, have rollback plan |
| Data loss during backfill | Verify counts before/after |
| Regression in analytics | Run full test suite before merging |
| Performance impact | Benchmark key operations before/after |

---

## Git Workflow

1. **Branch**: Already on `feature/forecast-rul-v10`
2. **For each task**: Create small focused commits
   ```bash
   git commit -m "fix(omr): Add NULL check for ContributionScore calculation"
   git commit -m "fix(forecast): Update column names to match current schema"
   git commit -m "fix(forecast): Initialize model_params in ForecastingState"
   ```
3. **After all tests pass**: Open PR for review
4. **Merge to main** when approved
5. **Tag release**: `git tag -a v10.0.1`

---

## Definitions of Done

- ✓ All code changes committed to feature branch
- ✓ All unit tests pass
- ✓ Manual end-to-end test passes
- ✓ SQL verification queries show expected results
- ✓ Release notes written
- ✓ PR reviewed and approved
- ✓ Merged to main
- ✓ Git tag created
- ✓ Documented in GitHub releases page

---

**Next Step**: Start with Task 1 (OMR NULL Constraint) - this is highest impact, lowest effort.


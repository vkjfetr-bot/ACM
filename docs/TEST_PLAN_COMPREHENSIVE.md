# Comprehensive Testing Plan - test/comprehensive-validation Branch

## Purpose
Validate ACM pipeline after critical fixes: RUL writes, batch overlap deduplication, and config corrections.

## Prerequisites
- [x] min_train_samples reverted to 200 (was 2)
- [x] ACM_Config SQL table synced with corrected config
- [x] All changes committed to main branch
- [x] Test branch `test/comprehensive-validation` created
- [x] Backup files deleted

## Test Scenarios

### 1. True Cold Start Test
**Objective**: Verify pipeline works from scratch with proper min_train_samples=200

**Steps**:
1. Clear all ACM_* tables (use `tools/truncate_acm_tables.py`)
2. Verify ACM_Config has min_train_samples=200 for EquipID 0,1,2621
3. Run single batch for FD_FAN:
   ```powershell
   python -m core.acm_main --equip FD_FAN
   ```
4. Check logs for:
   - SmartColdstart completion (should train models with 200+ samples)
   - Regime clustering quality (silhouette score > 0.2)
   - No errors in write_table() calls
   - RUL tables populated (ACM_RUL_Summary, ACM_RUL_TS, ACM_RUL_Attribution)

**Expected Results**:
- Pipeline completes successfully
- ACM_Runs table shows 1 new entry
- ACM_RegimeTimeline has entries with distinct RegimeLabels
- ACM_HealthTimeline has HealthIndex values
- No NULL constraint violations in logs

**Pass Criteria**:
- Exit code 0
- All ACM_* tables have data
- Regime silhouette > 0.2
- No SQL errors in logs

---

### 2. Batch Mode Test (100 Batches)
**Objective**: Validate batch overlap deduplication and _Latest views

**Steps**:
1. Run 100 batches for FD_FAN:
   ```powershell
   scripts/run_batch_mode.ps1 -Equipment FD_FAN -NumBatches 100 -StartBatch 1
   ```
2. Query base table vs Latest view row counts:
   ```sql
   -- Base table (will have duplicates)
   SELECT COUNT(*) AS BaseCount, COUNT(DISTINCT Timestamp) AS UniqueTimestamps
   FROM ACM_RegimeTimeline WHERE EquipID = 1;
   
   -- Latest view (should match unique timestamps)
   SELECT COUNT(*) AS LatestCount
   FROM ACM_RegimeTimeline_Latest WHERE EquipID = 1;
   ```
3. Verify Latest views deduplicate properly:
   - BaseCount > UniqueTimestamps (overlaps exist)
   - LatestCount == UniqueTimestamps (no duplicates)

**Expected Results**:
- 100 successful batch runs
- ACM_Runs has 100 new entries
- Base tables have overlapping timestamps (BaseCount > UniqueTimestamps)
- _Latest views have only unique timestamps (LatestCount == UniqueTimestamps)

**Pass Criteria**:
- All 100 batches complete (exit code 0)
- Latest views eliminate duplicates
- No wave patterns in Grafana (verify visually)

---

### 3. Grafana Visualization Test
**Objective**: Confirm wave patterns are gone after dashboard updates

**Steps**:
1. Open Grafana dashboard at localhost:3000
2. Check these panels:
   - **Regime Timeline**: Should show clean state blocks, no oscillations
   - **Health Index Over Time**: Should show smooth trends, no artificial waves
   - **Normalized Sensor Trends**: Should show actual sensor patterns, not sine waves
3. Verify all dashboards use _Latest views:
   ```powershell
   Get-ChildItem grafana_dashboards/*.json | Select-String "ACM_RegimeTimeline_Latest"
   ```

**Expected Results**:
- Regime Timeline shows stable regime assignments
- Health scores are smooth, no repeated dips
- Sensor data shows actual patterns, not waves
- All dashboards query _Latest views

**Pass Criteria**:
- Visual inspection confirms no wave artifacts
- All 8 updated dashboards use _Latest views
- No "No data" panels (unless legitimately empty)

---

### 4. RUL Table Validation
**Objective**: Ensure RUL tables populate correctly without NULL errors

**Steps**:
1. Query RUL tables after batch run:
   ```sql
   SELECT TOP 10 * FROM ACM_RUL_Summary WHERE EquipID = 1 ORDER BY LastUpdate DESC;
   SELECT COUNT(*) FROM ACM_RUL_TS WHERE EquipID = 1;
   SELECT COUNT(*) FROM ACM_RUL_Attribution WHERE EquipID = 1;
   SELECT COUNT(*) FROM ACM_MaintenanceRecommendation WHERE EquipID = 1;
   ```
2. Verify no NULL values in required columns:
   - Method (should be 'multipath_ensemble' or 'default')
   - LastUpdate (should be recent timestamp)
   - EarliestMaintenance (should be valid timestamp)

**Expected Results**:
- All RUL tables have data
- No NULL constraint violations
- Method column populated with valid values
- Timestamps are all recent (within last run window)

**Pass Criteria**:
- Row counts > 0 for all RUL tables
- No NULL values in required columns
- No "Cannot insert NULL" errors in logs

---

### 5. Model Persistence & Reload Test
**Objective**: Verify models persist and reload correctly in SQL mode

**Steps**:
1. Run single batch, let models train:
   ```powershell
   python -m core.acm_main --equip FD_FAN
   ```
2. Check ACM_PCA_Models and ACM_RegimeSummary for saved models
3. Run another batch immediately (should reload models):
   ```powershell
   python -m core.acm_main --equip FD_FAN
   ```
4. Check logs for "Loading cached model" or "Reusing model"

**Expected Results**:
- First run trains models from scratch
- Models saved to ACM_PCA_Models, ACM_RegimeSummary
- Second run reloads models (faster execution)
- Regime assignments consistent between runs

**Pass Criteria**:
- Models persist to SQL
- Second run faster than first (model reload works)
- Consistent regime labels across runs

---

### 6. Config Sync Validation
**Objective**: Confirm config changes propagate to SQL

**Steps**:
1. Verify current config in SQL:
   ```sql
   SELECT EquipID, ParameterKey, ParameterValue 
   FROM ACM_Config 
   WHERE ParameterKey = 'data.min_train_samples';
   ```
2. Expected values:
   - EquipID 0: 200
   - EquipID 1: 200
   - EquipID 2621: 200
3. If incorrect, re-sync:
   ```powershell
   python scripts/sql/populate_acm_config.py
   ```

**Expected Results**:
- min_train_samples = 200 for all EquipIDs
- No value = 2 in ACM_Config

**Pass Criteria**:
- Config matches config_table.csv
- Pipeline uses min_train_samples=200 (check logs)

---

## Full Integration Test

### Objective: End-to-end validation

**Steps**:
1. Clear all ACM_* tables
2. Run cold start for FD_FAN (single batch)
3. Run 100 batches for FD_FAN
4. Run 100 batches for GAS_TURBINE
5. Verify Grafana dashboards for both equipment
6. Check SQL table row counts and Latest view deduplication
7. Validate RUL tables populated correctly
8. Test model persistence across runs

**Expected Results**:
- 202 total runs (1 cold + 100 FD_FAN + 100 GAS_TURBINE + 1 GAS_TURBINE cold)
- All SQL tables populated
- Grafana shows clean data (no waves)
- RUL predictions available for both equipment
- No errors in logs

**Pass Criteria**:
- All 202 runs complete successfully
- 100% data in SQL tables (no file writes)
- Grafana visualizations correct
- Performance acceptable (<5 min per batch)

---

## Regression Checks

### Issues Previously Fixed
1. **RUL NULL constraint errors** → write_table() provides defaults
2. **Wave patterns in Grafana** → _Latest views deduplicate overlaps
3. **min_train_samples=2** → Reverted to 200 for statistical validity

### Regression Test Steps
1. Run 10 batches, check for any RUL errors
2. Query ACM_RegimeTimeline for duplicate timestamps at same EquipID
3. Verify config has min_train_samples=200

**Pass Criteria**:
- No RUL errors in logs
- _Latest views eliminate all duplicates
- Config remains at 200 (not reverted to 2)

---

## Test Summary Template

```
## Test Results - [Date]

### Environment
- Branch: test/comprehensive-validation
- Commit: [git rev-parse HEAD]
- SQL Server: localhost\B19CL3PCQLSERVER
- Database: ACM

### Test 1: Cold Start
- Status: [PASS/FAIL]
- Notes: ...

### Test 2: Batch Mode (100 batches)
- Status: [PASS/FAIL]
- Run Count: [actual]
- Duplicates Eliminated: [BaseCount - LatestCount]
- Notes: ...

### Test 3: Grafana Visualization
- Status: [PASS/FAIL]
- Dashboards Checked: [list]
- Wave Patterns: [YES/NO]
- Notes: ...

### Test 4: RUL Tables
- Status: [PASS/FAIL]
- ACM_RUL_Summary Rows: [count]
- ACM_RUL_TS Rows: [count]
- NULL Errors: [count]
- Notes: ...

### Test 5: Model Persistence
- Status: [PASS/FAIL]
- Models Saved: [YES/NO]
- Reload Successful: [YES/NO]
- Notes: ...

### Test 6: Config Sync
- Status: [PASS/FAIL]
- min_train_samples: [value]
- Notes: ...

### Full Integration Test
- Status: [PASS/FAIL]
- Total Runs: [count]
- Failures: [count]
- Overall Result: [PASS/FAIL]
- Notes: ...

### Overall Assessment
- [X] All tests passed
- [ ] Some tests failed (see notes)
- [ ] Major issues found (requires fixes)

### Next Steps
- [ ] Merge test branch to main
- [ ] Tag release: v8.x.x
- [ ] Deploy to production
- [ ] Monitor first 24h in production
```

---

## Exit Criteria

Branch can be merged to main when:
1. Cold start test PASSES
2. Batch mode test PASSES (100 batches complete)
3. Grafana visualization test PASSES (no waves)
4. RUL tables test PASSES (no NULL errors)
5. Model persistence test PASSES
6. Config sync test PASSES
7. Full integration test PASSES
8. No regressions detected

---

## Notes for Testers

- Always check logs for errors (`grep -i "error\|fail\|null" logs/*.log`)
- Visual Grafana inspection is critical (automated tests can't catch wave patterns)
- Compare run times: should be ~3-5 min per batch for FD_FAN, ~5-7 min for GAS_TURBINE
- If any test fails, do NOT merge to main - fix issues in test branch first
- Use `tools/check_counts.py` for quick SQL row count checks
- Keep backups before clearing ACM_* tables for cold start tests

---

## Troubleshooting

**Issue**: Cold start fails with "not enough samples"
- **Cause**: min_train_samples still 2 in memory (config not reloaded)
- **Fix**: Restart Python kernel or re-run config sync

**Issue**: Wave patterns still visible in Grafana
- **Cause**: Dashboards not updated or cache issue
- **Fix**: Re-run `scripts/update_grafana_dashboards.ps1`, restart Grafana

**Issue**: RUL NULL errors persist
- **Cause**: Old code without write_table() fix
- **Fix**: Verify you're on test/comprehensive-validation branch

**Issue**: Duplicate timestamps in _Latest views
- **Cause**: Views not created or query using base table
- **Fix**: Re-run `scripts/sql/54_create_latest_run_views.sql`

---

**Last Updated**: 2025-12-01  
**Branch**: test/comprehensive-validation  
**Author**: Copilot  
**Status**: Ready for testing

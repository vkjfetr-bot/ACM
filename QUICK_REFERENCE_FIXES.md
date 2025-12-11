# Quick Reference: Fixes Action Plan Summary

**Status**: Ready to Execute  
**Total Tasks**: 15  
**Estimated Effort**: 8-10 hours  
**Target Completion**: 2025-12-18  

---

## Priority Matrix

### 🔴 CRITICAL (Do These First - 1.5 hours)

| # | Task | File(s) | Est. Time | Impact |
|---|------|---------|-----------|--------|
| 1 | OMR NULL Constraint | `core/omr.py`, `core/output_manager.py` | 30 min | Data loss prevention |
| 2 | Sensor Forecast Columns | `core/forecasting.py` | 30 min | Enable forecasting |
| 3 | ForecastingState Attr | `core/forecasting.py` | 20 min | Fix crash |
| 4 | Install statsmodels | `pyproject.toml` | 10 min | Enable ARIMA/VAR |

### 🟠 HIGH (Do Next Week - 65 min)

| # | Task | File(s) | Est. Time | Impact |
|---|------|---------|-----------|--------|
| 5 | SQL Commit API | `core/forecasting.py`, `core/sql_client.py` | 30 min | Data integrity |
| 6 | Add Config Values | `configs/config_table.csv` | 15 min | Eliminate warnings |
| 7 | Regime Cache Variable | `core/regimes.py` | 20 min | Performance |

### 🟡 MEDIUM (This Month - 2-4 hours)

| # | Task | File(s) | Est. Time | Impact |
|---|------|---------|-----------|--------|
| 8 | Regime Clustering Investigation | `core/regimes.py` | 1-2 hours | Quality improvement |
| 9 | OMR Backfill Script | `scripts/` (new) | 1 hour | Restore historical data |

### ⚪ LOW (Nice to Have)

| # | Task | File(s) | Est. Time |
|---|------|---------|-----------|
| 10 | Timestamp Column Fix | `scripts/sql/`, `core/output_manager.py` | 20 min |
| 14 | Mahalanobis Monitoring | `scripts/` (new) | 30 min |

### 📋 VERIFICATION (After Fixes)

| # | Task | Est. Time |
|---|------|-----------|
| 11 | Test Suite | 1-2 hours |
| 12 | SQL Verification | 30 min |
| 13 | Update Schema Ref | 10 min |
| 15 | Release Notes | 45 min |

---

## Quick Start Checklist

### Before You Start:
- [ ] Create feature branch if not already on one
- [ ] Backup database
- [ ] Read full FIXES_ACTION_PLAN.md

### Day 1: Critical Fixes (4 tasks, ~90 min)
```bash
# Start with Task 1
# Implement: core/omr.py - Add NULL check for ContributionScore
# Test: pytest tests/test_omr.py -v

# Continue with Task 2
# Implement: core/forecasting.py - Update column names

# Task 3: ForecastingState initialization

# Task 4: pip install statsmodels
```

### Day 2: High Priority (3 tasks, ~65 min)
```bash
# Task 5: Fix SQL commit calls
# Task 6: Add 6 config values to configs/config_table.csv
# Task 7: Define stable_models_dir in regimes.py
```

### Day 3: Verify & Test (2-3 hours)
```bash
# Run full test suite
pytest tests/ -v

# Manual end-to-end test
python -m core.acm_main --equip FD_FAN

# SQL verification queries (provided in RUNLOGS_ANALYSIS.md)
```

### After Verification:
- [ ] Run OMR backfill script (Task 9)
- [ ] Document all changes (Task 15)
- [ ] Create git tag and release notes

---

## File Changes Summary

### Modified Files (Existing)
```
core/omr.py                          ← Add NULL checks
core/output_manager.py               ← Fix data mapping
core/forecasting.py                  ← Fix columns, state, API
core/sql_client.py                   ← Verify commit method
core/regimes.py                      ← Fix variable, investigate quality
configs/config_table.csv             ← Add 6 config values
pyproject.toml / requirements.txt    ← Add statsmodels
```

### New Files (Create)
```
scripts/OMR_backfill_script.sql      ← Restore historical data
scripts/regime_monitoring.sql        ← Track regularization
docs/RELEASE_v10.0.1.md              ← Release notes
```

---

## Critical Errors to Eliminate

### Before-and-After

| Error | Before | After | Task |
|-------|--------|-------|------|
| NULL constraint on ContributionScore | 100+ | 0 | 1 |
| Invalid column 'SensorName' | 5+ | 0 | 2 |
| 'model_params' AttributeError | 100+ | 0 | 3 |
| Missing statsmodels | 1+ | 0 | 4 |
| Missing 'commit' method | 2 | 0 | 5 |
| Config validation failures | 2,268 | 0 | 6 |
| Regime cache load failure | 842 | 0 | 7 |

---

## Success Criteria (Verify After Fixes)

```sql
-- All should return 0 or expected counts:

-- 1. No NULL ContributionScores
SELECT COUNT(*) FROM ACM_OMRContributionsLong 
WHERE ContributionScore IS NULL;  -- Should be 0

-- 2. No errors in recent logs
SELECT COUNT(*) FROM ACM_RunLogs 
WHERE Level='ERROR' 
AND LoggedAt > DATEADD(DAY, -1, GETDATE());  -- Should be 0

-- 3. Forecasts exist
SELECT COUNT(*) FROM ACM_HealthForecast_TS;  -- Should be > 0
SELECT COUNT(*) FROM ACM_SensorForecast_TS;  -- Should be > 0
SELECT COUNT(*) FROM ACM_RUL;  -- Should be > 0

-- 4. All config values present
SELECT COUNT(*) FROM ACM_Config 
WHERE ConfigKey IN (
  'regimes.quality.silhouette_min',
  'regimes.health.fused_warn_z',
  'regimes.auto_k.k_max',
  'regimes.auto_k.max_eval_samples',
  'regimes.smoothing.passes',
  'regimes.auto_k.max_models'
);  -- Should be 6 (or more if equipment-specific overrides)
```

---

## Git Commit Strategy

Make focused commits for each fix:

```bash
# Task 1: OMR fix
git commit -m "fix(omr): Add NULL check to prevent constraint violations"

# Task 2: Forecast columns
git commit -m "fix(forecast): Update column names to match current schema"

# Task 3: ForecastingState
git commit -m "fix(forecast): Initialize model_params attribute in ForecastingState"

# Task 4: Dependencies
git commit -m "chore: Add statsmodels to requirements"

# Task 5: SQL API
git commit -m "fix(forecast): Use correct SQL commit method"

# Task 6: Config
git commit -m "config: Add missing regime configuration values"

# Task 7: Cache
git commit -m "fix(regimes): Define stable_models_dir for model caching"

# After all fixes verified:
git tag -a v10.0.1 -m "Release v10.0.1: Critical forecasting and OMR fixes"
```

---

## Rollback Plan (If Issues Arise)

**Before starting**: Backup database
```bash
# Full backup
BACKUP DATABASE ACM TO DISK='D:\Backup\ACM_2025-12-12_BACKUP.bak';
```

**If critical issue**: Rollback to previous commit
```bash
git revert <commit-hash>
git push origin feature/forecast-rul-v10
```

---

## Documentation Files Generated

1. **RUNLOGS_ANALYSIS.md** (419 lines)
   - Full analysis of 339,940 log records
   - Top 20 issues with root causes
   - Impact assessment
   - Recommendations prioritized

2. **FIXES_ACTION_PLAN.md** (466 lines)
   - Detailed task breakdown
   - Step-by-step instructions for each fix
   - Timeline and dependencies
   - Success metrics
   - Git workflow

3. **QUICK_REFERENCE.md** (this file)
   - One-page summary
   - Priority matrix
   - Quick start checklist
   - File changes summary
   - Success verification SQL

---

## Next Actions

### Immediate (Today):
1. [ ] Review RUNLOGS_ANALYSIS.md for understanding
2. [ ] Review FIXES_ACTION_PLAN.md for detailed steps
3. [ ] Ensure database backup taken

### Tomorrow (Start Fixing):
1. [ ] Start with Task 1 (OMR NULL)
2. [ ] Make small, focused commits
3. [ ] Run tests after each fix
4. [ ] Verify no new errors in logs

### This Week:
1. [ ] Complete all 7 critical + high-priority tasks
2. [ ] Run full test suite
3. [ ] Execute SQL verification queries
4. [ ] Document and create release

---

## Contact/Questions

- **Analysis Date**: 2025-12-11
- **Analysis Based On**: 339,940 ACM RunLogs records
- **Data Period**: 2025-12-02 to 2025-12-11
- **Branch**: feature/forecast-rul-v10
- **Version Target**: v10.0.1 or v10.1.0

---

**Ready to proceed?** Start with FIXES_ACTION_PLAN.md Task 1 (OMR NULL Constraint).

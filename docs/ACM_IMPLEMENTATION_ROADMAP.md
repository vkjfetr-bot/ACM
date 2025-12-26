# ACM Implementation Roadmap

**Updated:** December 26, 2025  
**Branch:** feature/v11-refactor  
**Goal:** Complete v11 refactoring - reduce acm_main.py from 5,321 → <500 lines

---

## Current Status

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| acm_main.py lines | 5,464 | <500 | ⏳ 90% reduction needed |
| Helper functions | 45 extracted | 45 | ✅ Complete |
| Phase functions | 2 of 7 | 7 | ⏳ In progress |
| Error handling | safe_step() consolidated | Complete | ✅ Done |
| SQL tables with data | 35+ | 42 | ⏳ 13 empty tables |

---

## Priority 0: Error Handling Consolidation (COMPLETE)

Consolidated try/except sprawl with standardized error handling.

| Component | Description | Status |
|-----------|-------------|--------|
| `safe_step()` helper | Wraps operations with consistent error handling | ✅ Done |
| `RunOutcome` enum | OK, DEGRADED, NOOP, FAIL states | ✅ Done |
| `RunContext` dataclass | Consolidated context with degradations tracking | ✅ Done |
| `degradations` list | Track partial failures for DEGRADED outcome | ✅ Done |
| Persist phase blocks | 4 blocks converted to safe_step() | ✅ Done |
| Feature phase blocks | 2 blocks converted (build, impute) | ✅ Done |
| Regime phase blocks | 1 block converted (feature_basis) | ✅ Done |
| DEGRADED outcome | Analytics/forecast failures marked DEGRADED | ✅ Done |
| Metadata consistency | All debug logs have equip/run_id context | ✅ Done |

---

## Priority 1: Phase Function Extraction (Wave 4)

Extract 7 phase functions to transform `run_acm_pipeline()` from ~3,000 lines to ~100 lines.

| Phase | Function | Lines | Purpose | Status |
|-------|----------|-------|---------|--------|
| 1 | `_phase_initialize_runtime()` | ~343 | Startup, config, SQL connection | ⏳ |
| 2 | `_phase_load_data()` | ~732 | SmartColdstart, data validation | ⏳ |
| 3 | `_phase_fit_models()` | ~396 | Train/load detector models | ⏳ |
| 4 | `_phase_label_regimes()` | ~262 | Operating regime detection | ⏳ |
| 5 | `_phase_calibrate()` | ~174 | Calibrate detector outputs | ⏳ |
| 6 | `_phase_fuse_and_episodes()` | ~650 | Multi-detector fusion, episodes | ⏳ |
| 7 | `_write_sql_artifacts()` | ~120 | SQL artifact writes | ✅ Done |
| - | `_auto_tune_parameters()` | ~210 | Autonomous parameter tuning | ✅ Done |

**Target Structure:**
```python
def run_acm_pipeline(equip: str, ...) -> Dict[str, Any]:
    ctx = _phase_initialize_runtime(equip, cfg, sql_client)
    data = _phase_load_data(ctx)
    models = _phase_fit_models(ctx, data)
    regimes = _phase_label_regimes(ctx, data, models)
    scores = _phase_calibrate(ctx, data, models, regimes)
    episodes = _phase_fuse_and_episodes(ctx, scores)
    _phase_persist_results(ctx, scores, episodes)
    return ctx.result
```

---

## Priority 2: Empty Tables (13 remaining)

### TIER 3 - Root Cause (2 empty)
| Table | Rows | Needed Logic |
|-------|------|--------------|
| ACM_DriftSeries | 0 | Wire `_build_drift_ts()` to SQL write |
| ACM_FeatureDropLog | 0 | Wire `_log_dropped_features()` to SQL write |

### TIER 4 - Data & Model (1 empty)
| Table | Rows | Needed Logic |
|-------|------|--------------|
| ACM_CalibrationSummary | 0 | Add calibration summary write |

### TIER 6 - Advanced Analytics (5 empty)
| Table | Rows | Priority |
|-------|------|----------|
| ACM_RegimeOccupancy | 0 | Low |
| ACM_RegimeTransitions | 0 | Low |
| ACM_ContributionTimeline | 0 | Low |
| ACM_RegimePromotionLog | 0 | Low |
| ACM_DriftController | 0 | Low |

### TIER 7 - V11 Features (5 empty)
| Table | Rows | Priority |
|-------|------|----------|
| ACM_RegimeDefinitions | 0 | Low |
| ACM_ActiveModels | 0 | Low |
| ACM_DataContractValidation | 0 | Low |
| ACM_SeasonalPatterns | 0 | Low |
| ACM_AssetProfiles | 0 | Low |

---

## Validation Checklist

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

# 4. Syntax check
python -m py_compile core/acm_main.py
```

---

## Completed Work (Reference Only)

- ✅ Wave 1: Dead code removal (-335 lines)
- ✅ Wave 2: File-mode branch removal
- ✅ Wave 3: 30 helper functions extracted
- ✅ All TIER 1, 2, 5 tables populated (16 tables)
- ✅ Most TIER 3, 4 tables populated (19 tables)

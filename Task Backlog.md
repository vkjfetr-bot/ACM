# ACM V8 Unified Backlog & Issue Tracker

**Last Updated:** 2025-11-19  
**Scope:** SQL Mode rollout, Forecast/RUL roadmap, charting/documentation polish, and technical debt items carried over from the prior To-Do documents.

---

## Recent Completions (2025-11-19)

### ✅ CONSOLIDATED: Forecasting Module Consolidation (Commits: a7edfff, b1aacb1)
- **Problem**: Three separate forecasting modules (forecast.py, enhanced_forecasting.py, enhanced_forecasting_sql.py) caused confusion
- **Solution**: Unified into single `core/forecasting.py` module
- **Changes**:
  - Added AR1Detector class to forecasting.py
  - Updated all imports in acm_main.py to use `from core import forecasting`
  - Renamed old modules to `*_deprecated.py` for reference
  - Created FORECASTING_CONSOLIDATION.md with migration guide
- **Benefits**: Single import, clear entrypoint, reduced complexity (2410 lines → 528 lines active)
- **Status**: ✅ COMPLETE - SQL-mode tested and working

---

## Why this document?

`Task Backlog.md` now replaces the old `Task Backlog.md`. The two lists diverged and became hard to maintain, so they have been consolidated here. All day-to-day work should now be tracked in **GitHub Issues** instead of ad-hoc markdown tables. This file summarizes the active workstreams, how to label issues, and what is still genuinely outstanding after auditing the old backlog entries.

---

## Issue-Based Workflow

1. **Open a GitHub issue** for every piece of work. Use the naming convention `AREA-ID: short description` (e.g., `SQL-12: Complete dual-write run matrix`).
2. **Apply labels** so filtering by workstream is easy. Suggested labels: `SQL`, `Forecast`, `RUL`, `Charts`, `Docs`, `TechDebt`, and priority labels (`P0`, `P1`, `P2`).
3. **Cross-reference this file** from the issue description so the original context is preserved.
4. **Close issues** via pull requests. Avoid adding new TODO tables in markdown?use Issues/Projects for status tracking.

> The old `Task Backlog.md` file has been removed. If you come across references to it, update them to point here or directly to the GitHub Issues board.

---

## Workstream Summary (What is truly remaining)

The sections below list the still-open themes from the merged backlogs. Each bullet either already has an associated issue or needs one created.

### 1. SQL Integration (Phase 1 and Model Persistence)
- **SQL-12**  Finish the remaining 6/10 dual-write validation runs and capture the results in SQL tables.
- **SQL-13**  Build `validate_dual_write.py` to compare CSV vs SQL outputs automatically.
- **SQL-14**  Row-count/value parity checks between file and SQL storage.
- **SQL-15**  Baseline SQL write performance (<15 s target, capture stats in Run Logs).
- **SQL-20/21/22/23**  Save/load detectors to ModelRegistry and wire them into the training pipeline.

### 2. Forecast & RUL
- **FCST-15**  Remove the `scores.csv` dependency so forecasting works in SQL-only runs.
- **FCST-16**  Publish per-sensor/regime forecasts into `ACM_SensorForecast_TS` with diagnostics.
- **RUL-01**  Make `rul_estimator.py` consume SQL health timelines instead of CSVs.
- **RUL-02**  Add probabilistic RUL bands and maintenance telemetry to the SQL outputs.
- *(FCST-17, RUL-03, SQL-57, etc., were completed or superseded and do not need new issues.)*

### 4. Technical Debt & Ops
- **DEBT-07**  Tighten error handling around detector training and SQL IO.
- **DEBT-14/15**  Improve test hooks/path handling in `core/acm_main.py` and ensure error truncation captures full stacks.
- **OPS** tasks (scheduler integration, alerting, deployment runbooks) still need issues opened as we prepare for automation in SQL-only mode.

---

## Issue Migration Checklist

Use the following mapping when opening/triaging GitHub issues. Strike through entries if they are already covered by an existing issue or PR.

| Legacy ID | Area | Status | Action |
| --------- | ---- | ------ | ------ |
| SQL-12 | SQL Integration | Planned | Create/maintain issue covering the remaining dual-write validation matrix. |
| SQL-13 | SQL Integration | Planned | Issue for the CSV vs SQL comparison tool. |
| SQL-14 | SQL Integration | Planned | Issue for parity validation automation. |
| SQL-15 | SQL Integration | Planned | Issue for benchmarking SQL write times. |
| SQL-20/21/22/23 | Model Persistence | Planned | Either one umbrella issue or separate ones for save/load + integration + testing. |
| FCST-15/16 | Forecast | Planned | Two issues tracking SQL-only forecast support and sensor forecasts. |
| RUL-01/02 | RUL | Planned | Two issues covering SQL ingestion + probabilistic outputs. |
| CHART-19/20 | Charts/Docs | Planned | Open documentation + tooling issues. |
| DEBT-07/14/15 | Tech Debt | Planned | Create issues for structured error handling and test hooks. |

All other historical tasks were either completed (already struck through in previous versions) or removed because they duplicated the finished logging/chart work. If you discover a missing item while working through the code, open an issue instead of reintroducing a markdown TODO.

---

## Historical Context

- The previous `Task Backlog.md` and `Task Backlog.md` documents were audited on 2025-11-16. Duplicates were removed, chart/CSV-only tasks were archived, and completed logging upgrades were marked done.
- `CHART-03/04/13/15/16/17/18`, OUT-03/05/20/21/24/26/27/28, DEBT-04, CFG-07, SQL-57, and all logging upgrades are confirmed complete.
- SQL Run Logs, module-level logging overrides, and CLI logging flags now ship, so DEBT-16/17 are closed.

For everything else, the single source of truth is now **GitHub Issues**. This document will be updated only when the set of workstreams changes (e.g., adding a new audit area or closing a pillar entirely).


## New Tasks Created after analysis

Pending SQL-Related Tasks
🔴 HIGH PRIORITY - Empty Table Implementations
1. ACM_DataQuality ⚠️ CRITICAL
Purpose: Track data quality metrics per run (nulls, gaps, flatlines, statistics)
Schema: 24 columns (train/score metrics + sensor + RunID)
Current Status: Table exists, no write logic implemented
Implementation Location: output_manager.py after data loading
Business Value: HIGH - Essential for operations monitoring and data reliability
Estimated Effort: 2-4 hours
Blocked By: None
Note: Previously skipped due to pyodbc NULL float handling - needs resolution
2. ACM_RecommendedActions ⚠️ CRITICAL
Purpose: Automated maintenance action suggestions per run
Schema: 6 columns (Action, Priority, EstimatedDuration_Hours, RunID, EquipID, CreatedAt)
Current Status: Table exists, no write logic implemented
Implementation Location: acm_main.py or output_manager.py
Business Value: HIGH - Operator decision support and action tracking
Estimated Effort: 3-5 hours (requires recommendation engine logic)
Blocked By: None
Note: ACM_MaintenanceRecommendation (42 rows) is simpler version already working
3. ACM_Scores_Wide 🟡 MEDIUM
Purpose: Wide-format detector scores (one row per timestamp, all detectors as columns)
Schema: 15 columns (Timestamp + 10 detector z-scores + fused + regime + RunID + EquipID)
Current Status: Table exists, no write logic implemented
Implementation Location: output_manager.py as alternative score output
Business Value: MEDIUM - Nice-to-have for time-series export/visualization
Estimated Effort: 2-3 hours (pivot existing score data)
Blocked By: None
Note: May be redundant with existing ACM_ContributionTimeline

MEDIUM PRIORITY - Enhanced Feature Tables (Future Enhancements)
1. ACM_FailureCausation 🔮 PLANNED
Purpose: Advanced failure mode analysis with detector-level causation
Current Status: Schema exists (per FORECAST_RUL_AUDIT_SUMMARY.md), no data
Business Value: MEDIUM - Enhanced diagnostics beyond current hotspot analysis
Estimated Effort: 8-12 hours (new analytics logic required)
Blocked By: Requires enhanced forecasting framework expansion
1. ACM_EnhancedFailureProbability_TS 🔮 PLANNED
Purpose: Enhanced failure probability time-series with confidence intervals
Current Status: Schema exists, no data
Business Value: MEDIUM - Improvement over existing ACM_FailureForecast_TS (1,104 rows working)
Estimated Effort: 6-10 hours
Blocked By: Requires enhanced forecasting module (core/enhanced_forecasting.py expansion)
1. ACM_EnhancedMaintenanceRecommendation 🔮 PLANNED
Purpose: Enhanced maintenance recommendations with causation links
Current Status: Schema exists, no data
Business Value: MEDIUM - Enhancement over existing ACM_MaintenanceRecommendation (42 rows working)
Estimated Effort: 5-8 hours
Blocked By: Requires enhanced recommendation engine

SQL INTEGRATION TASKS (Phase 3: Pure SQL Operation)
SQL-45: Remove CSV Output Writes ⏳ PENDING
Objective: Eliminate all CSV file writes, keep SQL-only
Current State: Dual-write active (both CSV and SQL)
Changes Required:
Remove write_dataframe() CSV writes from output_manager.py
Keep SQL table writes only (ALLOWED_TABLES whitelist)
Remove scores.csv, episodes.csv, metrics.csv exports
Keep: Charts/PNG generation (visual outputs separate from storage)
Impact: artifacts directory will only contain charts, no data CSVs
Estimated Effort: 3-4 hours
Priority: MEDIUM (system working, cleanup for production)
SQL-46: Eliminate Model Filesystem Persistence ⏳ PENDING
Objective: Remove .joblib file writes, use SQL ModelRegistry only
Current State: Models saved as .joblib files in artifacts/{equip}/models/
Changes Required:
Remove filesystem save/load from model_persistence.py
Keep SQL ModelRegistry writes only
Remove stable_models_dir fallback logic
Remove .joblib file writes
Impact: No model files in filesystem, all models in SQL
Estimated Effort: 4-6 hours
Priority: MEDIUM (covered by SQL-20/21/22/23 below)
Related: SQL-20/21/22/23 (ModelRegistry save/load)
SQL-50: End-to-End Pure SQL Validation ⏳ PENDING
Objective: Validate complete SQL-only operation for 30+ days
Validation Steps:
Run full pipeline with storage_backend='sql'
Verify: No files created in artifacts (except charts)
Verify: All results in SQL tables only
Confirm: Pipeline runs successfully start-to-finish
Performance: SQL write time <15s per run
Stability: 30+ days unattended operation
Estimated Effort: Ongoing validation (2-4 weeks monitoring)
Priority: MEDIUM (system working, formal validation needed)

VALIDATION & TESTING TASKS
SQL-12: Complete Dual-Write Validation Matrix ⏳ 6/10 REMAINING
Objective: Validate CSV vs SQL output parity across all tables
Current State: 4/10 validation runs completed
Remaining: 6 more validation runs with different equipment/time windows
Estimated Effort: 3-4 hours (1 hour per equipment run + analysis)
Priority: HIGH (quality assurance)
SQL-13: Build validate_dual_write.py Comparison Tool 🆕 PLANNED
Objective: Automated CSV vs SQL comparison for all 40+ tables
Current State: No tool exists
Implementation:
Compare row counts between CSV and SQL
Compare column values (floating point tolerance)
Report discrepancies automatically
Estimated Effort: 4-6 hours
Priority: HIGH (prevents manual validation burden)
SQL-14: Row-Count/Value Parity Checks 🆕 PLANNED
Objective: Automated parity validation as part of CI/test suite
Current State: Manual validation only
Implementation: pytest fixtures that compare outputs
Estimated Effort: 3-5 hours
Priority: MEDIUM
SQL-15: Baseline SQL Write Performance 🆕 PLANNED
Objective: Benchmark SQL write times (<15s target per run)
Current State: No formal benchmarks captured
Implementation: Add timing instrumentation to OutputManager
Estimated Effort: 2-3 hours
Priority: MEDIUM

MODEL PERSISTENCE TO SQL
SQL-20: Save Detectors to ModelRegistry 🆕 PLANNED
Objective: Serialize detector models to SQL ModelRegistry table
Current State: ModelRegistry table exists, no save logic
Implementation: model_persistence.py serialize to binary + metadata
Estimated Effort: 5-7 hours
Priority: MEDIUM
Blocked By: None
SQL-21: Load Detectors from ModelRegistry 🆕 PLANNED
Objective: Deserialize detector models from SQL
Current State: No load logic implemented
Implementation: model_persistence.py fetch + deserialize
Estimated Effort: 4-6 hours
Priority: MEDIUM
Depends On: SQL-20
SQL-22: Wire ModelRegistry into Training Pipeline 🆕 PLANNED
Objective: Replace .joblib file save/load with SQL calls
Current State: Pipeline uses filesystem
Implementation: Update acm_main.py training/loading logic
Estimated Effort: 3-5 hours
Priority: MEDIUM
Depends On: SQL-20, SQL-21
SQL-23: Test ModelRegistry End-to-End 🆕 PLANNED
Objective: Validate model persistence across runs
Implementation: pytest fixtures validating save → load → predict cycle
Estimated Effort: 3-4 hours
Priority: MEDIUM
Depends On: SQL-22

FORECAST & RUL ENHANCEMENTS
FCST-15: Remove scores.csv Dependency ⏳ PENDING
Objective: Forecasting works in SQL-only runs (no CSV dependency)
Current State: forecast.py may still read scores.csv
Implementation: Ensure forecast reads from SQL tables only
Estimated Effort: 2-3 hours
Priority: HIGH (blocks SQL-only operation)
FCST-16: Per-Sensor/Regime Forecasts to ACM_SensorForecast_TS 🆕 PLANNED
Objective: Publish sensor-level forecasts with regime breakdown
Current State: Only equipment-level forecasts published
Implementation: Expand forecast logic, write to ACM_SensorForecast_TS
Estimated Effort: 6-8 hours
Priority: MEDIUM
RUL-01: SQL-Based RUL Estimation ⏳ PENDING
Objective: rul_estimator.py consumes SQL health timelines instead of CSVs
Current State: May still read from CSV files
Implementation: Update RUL data loading to query ACM_HealthTimeline
Estimated Effort: 3-4 hours
Priority: HIGH (blocks SQL-only operation)
RUL-02: Probabilistic RUL Bands ⏳ PENDING
Objective: Add P10/P50/P90 RUL confidence intervals
Current State: Single-point RUL estimates only
Implementation: Add probabilistic modeling to rul_estimator.py
Estimated Effort: 8-12 hours
Priority: MEDIUM

TECHNICAL DEBT
DEBT-07: Tighten Error Handling ⏳ PENDING
Objective: Robust error handling around detector training and SQL IO
Current State: Some unhandled exceptions possible
Implementation: Add try-except blocks with proper logging
Estimated Effort: 4-6 hours
Priority: MEDIUM
DEBT-14/15: Test Hooks & Path Handling ⏳ PENDING
Objective: Improve test fixtures and error stack truncation
Current State: Some test paths hard-coded
Implementation: Refactor acm_main.py test support
Estimated Effort: 3-5 hours
Priority: LOW

SUMMARY BY PRIORITY

CRITICAL (Implement Now)
ACM_DataQuality writes (2-4 hrs) - Operations monitoring
ACM_RecommendedActions writes (3-5 hrs) - Operator decision support

HIGH PRIORITY (Next Sprint)
SQL-12: Complete dual-write validation (3-4 hrs)
SQL-13: Build validate_dual_write.py tool (4-6 hrs)
FCST-15: Remove scores.csv dependency (2-3 hrs)
RUL-01: SQL-based RUL estimation (3-4 hrs)

MEDIUM PRIORITY (Following Sprint)
ACM_Scores_Wide writes (2-3 hrs) - Nice-to-have pivot table
SQL-14/15: Parity checks + performance benchmarking (5-8 hrs)
SQL-20/21/22/23: ModelRegistry implementation (15-22 hrs total)
SQL-45/46/50: Pure SQL operation cleanup (7-14 hrs)
FCST-16, RUL-02: Enhanced forecasting/RUL (14-20 hrs)
DEBT-07: Error handling improvements (4-6 hrs)
🔮 FUTURE ENHANCEMENTS (Backlog)
ACM_FailureCausation (8-12 hrs)
ACM_EnhancedFailureProbability_TS (6-10 hrs)
ACM_EnhancedMaintenanceRecommendation (5-8 hrs)
DEBT-14/15: Test infrastructure (3-5 hrs)

📊 EFFORT ESTIMATE TOTALS
Critical: 5-9 hours
High Priority: 12-17 hours
Medium Priority: 43-73 hours
Future Enhancements: 22-35 hours
Total Pending SQL Work: ~82-134 hours (10-17 developer days)

RECENTLY COMPLETED (For Context)
Fixed batch data truncation (60% data loss eliminated)
Fixed Unicode encoding errors in Heartbeat spinner
Comprehensive 46-table analysis (92/100 data quality score)
SQL logging infrastructure (SQL-57 complete)
All 40 core analytics tables actively populating
Forecasting & RUL tables working (1,104+ rows each)
Health timeline, sensor hotspots, regime detection all operational
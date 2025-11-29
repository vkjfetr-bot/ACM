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

### SQL MODE CONTINUOUS LEARNING ARCHITECTURE FIXES (2025-11-29)

**Context**: The continuous learning implementation for SQL mode is functionally safe but incomplete. Models rarely retrain based on data quality/drift, mixing two caching mechanisms, and baseline management is inconsistent. This creates a gap between what "continuous learning" suggests and what actually happens.

**Reference**: Analysis provided 2025-11-29 detailing 10 specific issues with SQL mode model retraining behavior.

---

#### 🔴 HIGH PRIORITY - Model Retraining Logic

**✅ SQL-CL-01: Implement Data-Driven Retrain Triggers** ⚠️ CRITICAL - COMPLETE
- **Problem**: Models only retrain when config changes or no cache exists. Data drift, anomaly rates, and regime quality degradation do NOT trigger retraining in SQL mode.
- **Solution Implemented** (2025-11-29):
  - ✅ Added config parameters: `models.auto_retrain.max_anomaly_rate=0.25`, `max_drift_score=2.0`, `max_model_age_hours=720`
  - ✅ Added anomaly_rate trigger to auto-tune section (lines 2756-2764)
  - ✅ Added drift_score trigger to auto-tune section (lines 2766-2774)
  - ✅ Added model_age and regime_quality triggers to quality check section (lines 2058-2115)
  - ✅ Removed `if not SQL_MODE:` guard from auto-tune section (line 2732)
  - ✅ Aggregated all triggers with proper reason logging
- **Impact**: Models now retrain based on data quality metrics, not just config changes
- **Files Modified**: 
  - `core/acm_main.py` (lines 2058-2115, 2729-2780)
  - `configs/config_table.csv` (added 11 auto_retrain config entries)
- **Status**: ✅ COMPLETE - Testing required
- **Next**: Run 20+ batch simulation to verify triggers fire correctly

**⏳ SQL-CL-02: SQL-Native Refit Flag Mechanism** ⚠️ CRITICAL - PARTIALLY COMPLETE
- **Problem**: Auto-tune writes `refit_requested.flag` only when `not SQL_MODE`. SQL deployments cannot trigger next-run retrain via quality policies.
- **Progress** (2025-11-29):
  - ✅ Added SQL mode logging for refit requests with detailed reasons
  - ✅ Enhanced refit flag write logic to include anomaly_rate and drift_score metrics
  - ⏳ SQL table `ACM_RefitRequests` not yet created
  - ⏳ Run-start query logic not yet implemented
- **Remaining Work**:
  - Create SQL table: `ACM_RefitRequests(EquipID, RequestedAt, Reason, AnomalyRate, DriftScore, Acknowledged)`
  - In SQL mode, write refit requests to this table instead of file
  - At run start, query table and set `refit_requested = True` if pending
  - Clear acknowledged flags after processing
- **Impact**: Quality-based retraining can work in SQL deployments
- **Estimated Effort**: 3-4 hours remaining
- **Priority**: P0 - Enables quality-driven retraining loop
- **Files**: `core/acm_main.py` (lines 2864-2883 placeholder logging), new migration script
- **Testing**: Force quality degradation, verify refit request written and honored

**✅ SQL-CL-03: Enhanced assess_model_quality Usage** 🟡 MEDIUM - COMPLETE
- **Problem**: `assess_model_quality` is called but metrics are discarded; only config signature comparison used.
- **Solution Implemented** (2025-11-29):
  - ✅ Extract `anomaly_metrics` and `drift_score` from `quality_report`
  - ✅ Use metrics to drive retrain decisions (anomaly_rate_trigger, drift_score_trigger)
  - ✅ Log all trigger reasons with detailed metrics
  - ✅ Aggregate `should_retrain`, `anomaly_rate_trigger`, `drift_score_trigger` into `needs_retraining` flag
  - ✅ Enhanced refit flag file to include anomaly_rate and drift_score values
- **Impact**: Quality metrics now directly influence retrain decisions, full observability
- **Files Modified**: `core/acm_main.py` (lines 2747-2780)
- **Status**: ✅ COMPLETE - Metrics fully wired into retrain logic

---

#### 🟡 MEDIUM PRIORITY - Caching & Validation

**SQL-CL-04: Remove Dual-Cache Confusion in SQL Mode** 🟡 MEDIUM
- **Problem**: Both `detectors.joblib` (legacy) and `ModelVersionManager` (SQL) are usable. Legacy requires identical train hash, SQL cache ignores it.
- **Current State**: Two sources of truth with different validation criteria
- **Fix Required**:
  - In SQL mode, disable legacy joblib cache by default: set `reuse_model_fit=False` in SQL configs
  - Or gate with `if not SQL_MODE and reuse_models:`
  - Rely solely on `ModelVersionManager` for SQL mode
  - Update validation criteria to handle drift (see SQL-CL-05)
- **Impact**: Single, consistent model cache mechanism per deployment mode
- **Estimated Effort**: 2-3 hours
- **Priority**: P1 - Reduces confusion and potential bugs
- **Files**: `core/acm_main.py` (cache loading sections ~1200-1400)
- **Testing**: Verify SQL mode never touches detectors.joblib

**SQL-CL-05: Extend check_model_validity Criteria** 🟡 MEDIUM
- **Problem**: Validation checks config signature + sensor list only. Model trained a year ago still accepted if config unchanged.
- **Current State**: No notion of training window, model age, or drift in validation
- **Fix Required**:
  - Extend manifest to include: `train_start`, `train_end`, `train_row_count`, `train_hash`, `baseline_version`
  - Add validation rules: reject if model age > `models.max_model_age_days`
  - Reject if drift indicators from last N runs exceed thresholds
  - Make thresholds configurable
- **Impact**: Prevents stale models from being reused indefinitely
- **Estimated Effort**: 5-7 hours
- **Priority**: P1 - Critical for long-running deployments
- **Files**: `core/model_persistence.py` (ModelVersionManager class)
- **Testing**: Create old model, verify rejection; test drift threshold enforcement

**SQL-CL-06: Fix models_were_trained Semantics** 🟢 LOW
- **Problem**: `models_were_trained = (not cached_models and detector_cache is None) or force_retrain` could save stale models if `force_retrain` set but fit skipped.
- **Current State**: Logically works but fragile to future refactors
- **Fix Required**:
  - Introduce explicit `detectors_fitted_this_run` boolean
  - Set only in the fit block after successful training
  - Define `models_were_trained = detectors_fitted_this_run`
  - Ensure fit block runs when `force_retrain=True` (invalidate caches first)
- **Impact**: More robust save logic, prevents edge case bugs
- **Estimated Effort**: 1-2 hours
- **Priority**: P2 - Safety improvement, not urgent
- **Files**: `core/acm_main.py` (model fitting and saving sections)
- **Testing**: Verify models only saved when actually fitted

---

#### 🟡 MEDIUM PRIORITY - Baseline Management

**SQL-CL-07: Remove Baseline Seed Logic in SQL Mode** 🟡 MEDIUM
- **Problem**: Even in SQL mode, fallback to `baseline_buffer.csv` or score-head seeding if `train_rows < min_points`. SmartColdstart already enforces sufficient baseline.
- **Current State**: Mixing SmartColdstart (SQL) with local CSV baseline seeding
- **Fix Required**:
  - For SQL mode, skip `baseline.seed` entirely after SmartColdstart returns `coldstart_complete=True`
  - Make `baseline.seed` strictly file-mode-only: wrap with `if not SQL_MODE:`
  - Baseline for SQL must come from SmartColdstart / ACM_BaselineBuffer only
- **Impact**: Single authoritative baseline source in SQL mode
- **Estimated Effort**: 2-3 hours
- **Priority**: P1 - Prevents baseline inconsistencies
- **Files**: `core/acm_main.py` (baseline section ~1000-1100)
- **Testing**: SQL mode run, verify no baseline_buffer.csv touched

**SQL-CL-08: Drop CSV Baseline in SQL Mode** 🟡 MEDIUM
- **Problem**: In SQL mode, both `baseline_buffer.csv` and `ACM_BaselineBuffer` maintained with similar content.
- **Current State**: Redundant, can diverge if CSV deleted or truncated
- **Fix Required**:
  - In SQL mode, drop CSV baseline entirely
  - Wrap `baseline_buffer.csv` read/write with `if not SQL_MODE:`
  - Use only `ACM_BaselineBuffer` and SmartColdstart logic for SQL
- **Impact**: Single source of truth for baseline data
- **Estimated Effort**: 2-3 hours
- **Priority**: P1 - Cleanup and consistency
- **Files**: `core/acm_main.py` (baseline buffer writes)
- **Testing**: SQL mode run, verify no baseline CSV created

---

#### 🟢 LOW PRIORITY - Config Loop & Observability

**SQL-CL-09: Close Auto-Tune Config Loop in SQL** 🟢 LOW
- **Problem**: Auto-tune computes `tuning_actions` but doesn't update `ACM_Config` or trigger retraining in SQL mode.
- **Current State**: Analysis done but loop not closed
- **Fix Required**:
  - Write `tuning_actions` to `ACM_ConfigHistory` with `PendingApply` flag
  - If `models.auto_retrain.on_tuning_change=True`, raise refit signal (via SQL-CL-02)
  - Next run picks up new config + retrains
- **Impact**: Fully automated tuning loop
- **Estimated Effort**: 4-6 hours
- **Priority**: P2 - Nice to have, not blocking
- **Files**: `core/acm_main.py` (auto-tune section)
- **Testing**: Force tuning action, verify config updated and retrain triggered

**SQL-CL-10: Improve Retrain Logging in SQL Mode** 🟢 LOW
- **Problem**: Logs say "Using cached models" or "retraining required" but don't explain WHY (age, drift, config, etc.)
- **Current State**: Hard to debug retrain behavior
- **Fix Required**:
  - When cached models accepted, log reason: config+columns unchanged, model_age=Xd < Yd, drift_ok=True
  - When retrain triggered, log actual reason: config change vs drift vs anomaly rate vs explicit refit
  - Add structured logging fields for parsing/monitoring
- **Impact**: Better observability and debugging
- **Estimated Effort**: 2-3 hours
- **Priority**: P2 - Quality of life
- **Files**: `core/acm_main.py` (cache and retrain sections)
- **Testing**: Review logs from multiple scenarios, verify clarity

---

**Total Estimated Effort**: 31-45 hours across 10 tasks
**Recommended Order**: SQL-CL-01 → SQL-CL-02 → SQL-CL-05 → SQL-CL-04 → SQL-CL-07 → SQL-CL-08 → SQL-CL-03 → SQL-CL-06 → SQL-CL-09 → SQL-CL-10

---

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

---

## 🚀 CONTINUOUS FORECASTING REDESIGN (2025-11-20)

### Problem Statement
Current architecture treats each batch in isolation:
- Models retrain from scratch every batch → wastes compute, discards learned patterns
- Forecasts regenerated independently → stepped/blocky probability curves
- No state persistence between batches → no temporal continuity
- Failure probability resets each run → unrealistic discontinuities
- RUL methodology unclear → "1 week" estimate unexplained
- Defect type not shown → operators don't know WHAT is failing

**User Requirements:**
1. Preserve prior state across batches (model evolution)
2. Temporal blending for smooth probability curves
3. Conditional retraining (add batch data to existing model when appropriate)
4. Clear failure condition definition on dashboard
5. Visual RUL projection with threshold crossing + confidence bands
6. Show predicted defect type/signature
7. ACM behaves as continuous live system (not batch-reset simulator)
8. Retrain-required indicator on dashboard
9. NO NEW SCRIPTS - modify existing code only (new SQL tables allowed)

### Architectural Changes

#### FORECAST-STATE-01: ForecastState Persistence 🔴 CRITICAL
**File:** `core/model_persistence.py`
**Changes:**
```python
@dataclass
class ForecastState:
    """Persistent state for continuous forecasting between batches."""
    equip_id: int
    state_version: int
    model_type: str  # "AR1", "ARIMA", "ETS"
    model_params: Dict[str, Any]  # {phi, mu, sigma} for AR1
    residual_variance: float
    last_forecast_horizon: pd.DataFrame  # Timestamp, ForecastHealth, CI_Lower, CI_Upper
    hazard_baseline: float  # EWMA smoothed hazard rate
    last_retrain_time: datetime
    training_data_hash: str
    training_window_hours: int
    forecast_quality: Dict[str, float]  # {rmse, mae, mape}
    
def save_forecast_state(state: ForecastState, artifact_root: Path, equip: str) -> None:
    """Serialize ForecastState to JSON in artifacts/{equip}/models/forecast_state.json"""
    
def load_forecast_state(artifact_root: Path, equip: str) -> Optional[ForecastState]:
    """Deserialize ForecastState from JSON. Returns None if not found."""
```

**SQL Table:** `ACM_ForecastState`
```sql
CREATE TABLE ACM_ForecastState (
    EquipID INT NOT NULL,
    StateVersion INT NOT NULL,
    ModelType NVARCHAR(50),
    ModelParamsJson NVARCHAR(MAX),  -- JSON serialized params
    ResidualVariance FLOAT,
    LastForecastHorizonJson NVARCHAR(MAX),  -- JSON array of forecast points
    HazardBaseline FLOAT,
    LastRetrainTime DATETIME2,
    TrainingDataHash NVARCHAR(64),
    TrainingWindowHours INT,
    ForecastQualityJson NVARCHAR(MAX),  -- {rmse, mae, mape}
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    PRIMARY KEY (EquipID, StateVersion)
)
```

**Effort:** 4-6 hours
**Files Modified:** `core/model_persistence.py`, `core/acm_main.py`

---

#### FORECAST-STATE-02: Continuous Forecasting Logic 🔴 CRITICAL
**File:** `core/forecasting.py`
**Function:** `run_enhanced_forecasting_sql()`

**Changes:**
1. **Load previous state:**
```python
prev_state = load_forecast_state(artifact_root, equip)
if prev_state:
    Console.info(f"[FORECAST] Loaded state v{prev_state.state_version}, last retrain {prev_state.last_retrain_time}")
```

2. **Sliding window instead of single batch:**
```python
# OLD: use only current batch health data
# NEW: combine last 72h from ACM_HealthTimeline + current batch
lookback_hours = 72
cutoff_time = current_batch_start - timedelta(hours=lookback_hours)
df_health_combined = load_health_timeline(sql_client, equip_id, since=cutoff_time)
```

3. **Conditional retraining:**
```python
def should_retrain(prev_state, current_metrics, config) -> Tuple[bool, str]:
    """
    Decide if full retrain needed.
    Returns: (retrain_needed, reason)
    """
    drift_threshold = config.get("forecasting", {}).get("drift_retrain_threshold", 1.5)
    energy_threshold = config.get("forecasting", {}).get("energy_spike_threshold", 1.5)
    error_threshold = config.get("forecasting", {}).get("forecast_error_threshold", 2.0)
    
    # Check drift
    drift_recent = get_recent_drift_metrics(sql_client, equip_id, window_hours=6)
    if drift_recent["DriftValue"].mean() > drift_threshold:
        return True, f"Drift exceeded {drift_threshold}"
    
    # Check anomaly energy spike
    energy_p95 = current_metrics.get("anomaly_energy_p95", 0)
    energy_median = current_metrics.get("anomaly_energy_median", 1)
    if energy_p95 > energy_threshold * energy_median:
        return True, f"Energy spike {energy_p95:.2f} > {energy_threshold}x median"
    
    # Check forecast quality degradation
    if prev_state and prev_state.forecast_quality:
        prev_rmse = prev_state.forecast_quality.get("rmse", 0)
        current_rmse = current_metrics.get("forecast_rmse", 0)
        if prev_rmse > 0 and current_rmse > error_threshold * prev_rmse:
            return True, f"Forecast error {current_rmse:.2f} > {error_threshold}x baseline"
    
    return False, "Model stable, incremental update"

# In main forecast logic:
retrain_needed, retrain_reason = should_retrain(prev_state, current_metrics, config)

if not retrain_needed and prev_state:
    # Incremental update: use existing model params, extend training window
    model_params = prev_state.model_params
    Console.info(f"[FORECAST] Incremental update: {retrain_reason}")
else:
    # Full retrain
    model_params = train_forecast_model(df_health_combined)
    Console.info(f"[FORECAST] Full retrain: {retrain_reason}")
```

4. **Horizon merging with temporal blending:**
```python
def merge_forecast_horizons(
    prev_horizon: pd.DataFrame,  # columns: Timestamp, ForecastHealth, CI_Lower, CI_Upper
    new_horizon: pd.DataFrame,
    current_time: datetime,
    blend_tau_hours: float = 12.0
) -> pd.DataFrame:
    """
    Merge overlapping forecast horizons with exponential decay weighting.
    
    Logic:
    - Keep all past points (Timestamp < current_time)
    - For overlapping future points, blend: w_new = 1 - exp(-dt/tau), w_prev = exp(-dt/tau)
    - Append non-overlapping new points
    """
    if prev_horizon.empty:
        return new_horizon
    
    # Filter to future points only
    prev_future = prev_horizon[prev_horizon["Timestamp"] >= current_time].copy()
    new_future = new_horizon[new_horizon["Timestamp"] >= current_time].copy()
    
    if prev_future.empty:
        return new_horizon
    
    # Merge on timestamp
    merged = pd.merge(
        prev_future, new_future,
        on="Timestamp", how="outer", suffixes=("_prev", "_new")
    ).sort_values("Timestamp")
    
    # Calculate blend weights
    dt_hours = (merged["Timestamp"] - current_time).dt.total_seconds() / 3600
    w_new = 1 - np.exp(-dt_hours / blend_tau_hours)
    w_prev = np.exp(-dt_hours / blend_tau_hours)
    
    # Blend values
    merged["ForecastHealth"] = (
        merged["ForecastHealth_new"].fillna(0) * w_new +
        merged["ForecastHealth_prev"].fillna(0) * w_prev
    )
    merged["CI_Lower"] = (
        merged["CI_Lower_new"].fillna(0) * w_new +
        merged["CI_Lower_prev"].fillna(0) * w_prev
    )
    merged["CI_Upper"] = (
        merged["CI_Upper_new"].fillna(0) * w_new +
        merged["CI_Upper_prev"].fillna(0) * w_prev
    )
    
    return merged[["Timestamp", "ForecastHealth", "CI_Lower", "CI_Upper"]]

# In main forecast logic:
merged_horizon = merge_forecast_horizons(
    prev_state.last_forecast_horizon if prev_state else pd.DataFrame(),
    new_forecast_df,
    current_batch_time,
    blend_tau_hours=config.get("forecasting", {}).get("blend_tau_hours", 12.0)
)
```

5. **Save updated state:**
```python
new_state = ForecastState(
    equip_id=equip_id,
    state_version=(prev_state.state_version + 1) if prev_state else 1,
    model_type="AR1",
    model_params=model_params,
    residual_variance=residual_variance,
    last_forecast_horizon=merged_horizon,
    hazard_baseline=smoothed_hazard,  # see FORECAST-STATE-03
    last_retrain_time=datetime.now() if retrain_needed else prev_state.last_retrain_time,
    training_data_hash=compute_hash(df_health_combined),
    training_window_hours=lookback_hours,
    forecast_quality={"rmse": rmse, "mae": mae, "mape": mape}
)
save_forecast_state(new_state, artifact_root, equip)
```

**Effort:** 10-14 hours
**Files Modified:** `core/forecasting.py`, `core/acm_main.py`

---

#### FORECAST-STATE-03: Hazard-Based Probability Smoothing 🔴 CRITICAL
**File:** `core/forecasting.py`
**New Function:** `smooth_failure_probability_hazard()`

**Implementation:**
```python
def smooth_failure_probability_hazard(
    prev_hazard_baseline: float,
    new_probability_series: pd.Series,  # Index: Timestamp, Values: discrete batch probabilities
    dt_hours: float = 1.0,
    alpha: float = 0.3
) -> pd.DataFrame:
    """
    Convert discrete batch probabilities to continuous hazard with EWMA smoothing.
    
    Math:
    - Hazard rate: lambda(t) = -ln(1 - p(t)) / dt
    - EWMA smoothing: lambda_smooth[t] = alpha * lambda_raw[t] + (1-alpha) * lambda_smooth[t-1]
    - Survival probability: S(t) = exp(-integral_0^t lambda_smooth(u) du)
    - Failure probability: F(t) = 1 - S(t)
    
    Returns: DataFrame with columns [Timestamp, HazardRaw, HazardSmooth, Survival, FailureProb]
    """
    df_result = pd.DataFrame(index=new_probability_series.index)
    
    # Convert probability to hazard rate
    p_clipped = new_probability_series.clip(1e-9, 1 - 1e-9)  # Avoid log(0)
    lambda_raw = -np.log(1 - p_clipped) / dt_hours
    df_result["HazardRaw"] = lambda_raw
    
    # EWMA smoothing
    lambda_smooth = np.zeros(len(lambda_raw))
    lambda_smooth[0] = alpha * lambda_raw.iloc[0] + (1 - alpha) * prev_hazard_baseline
    for i in range(1, len(lambda_raw)):
        lambda_smooth[i] = alpha * lambda_raw.iloc[i] + (1 - alpha) * lambda_smooth[i-1]
    df_result["HazardSmooth"] = lambda_smooth
    
    # Compute survival and failure probability
    cumulative_hazard = np.cumsum(lambda_smooth * dt_hours)
    df_result["Survival"] = np.exp(-cumulative_hazard)
    df_result["FailureProb"] = 1 - df_result["Survival"]
    df_result["Timestamp"] = df_result.index
    
    return df_result.reset_index(drop=True)

# In run_enhanced_forecasting_sql():
df_hazard = smooth_failure_probability_hazard(
    prev_hazard_baseline=prev_state.hazard_baseline if prev_state else 0.0,
    new_probability_series=failure_probs_df.set_index("Timestamp")["FailureProbability"],
    dt_hours=1.0,
    alpha=config.get("forecasting", {}).get("hazard_smoothing_alpha", 0.3)
)

# Write to ACM_FailureHazard_TS
output_manager.write_dataframe(
    df_hazard,
    tables_dir / "failure_hazard.csv",
    sql_table="ACM_FailureHazard_TS",
    add_created_at=True
)
```

**SQL Table:** `ACM_FailureHazard_TS`
```sql
CREATE TABLE ACM_FailureHazard_TS (
    Timestamp DATETIME2 NOT NULL,
    HazardRaw FLOAT,
    HazardSmooth FLOAT,
    Survival FLOAT,
    FailureProb FLOAT,
    RunID INT,
    EquipID INT,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    PRIMARY KEY (EquipID, RunID, Timestamp)
)
```

**Effort:** 6-8 hours
**Files Modified:** `core/forecasting.py`

---

#### FORECAST-STATE-04: Multi-Path RUL Derivation 🔴 CRITICAL
**File:** `core/enhanced_rul_estimator.py`
**Function:** `estimate_rul_and_failure()`

**Changes:**
```python
def compute_rul_multipath(
    health_forecast: pd.DataFrame,  # Timestamp, ForecastHealth, CI_Lower, CI_Upper
    hazard_df: pd.DataFrame,  # Timestamp, FailureProb
    anomaly_energy_df: pd.DataFrame,  # Timestamp, CumulativeEnergy
    current_time: datetime,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute RUL via three independent paths, take minimum.
    
    Path 1 (Trajectory): Find first Timestamp where ForecastHealth <= threshold
    Path 2 (Hazard): Find first Timestamp where FailureProb >= 0.5
    Path 3 (Energy): Find first Timestamp where CumulativeEnergy >= E_fail (calibrated)
    
    Returns: {
        "rul_trajectory_hours": float,
        "rul_hazard_hours": float,
        "rul_energy_hours": float,
        "rul_final_hours": float,  # min(t1, t2, t3)
        "confidence_band_hours": float,  # CI_Upper crossing - CI_Lower crossing
        "dominant_path": str  # "trajectory" | "hazard" | "energy"
    }
    """
    health_threshold = config.get("forecasting", {}).get("failure_threshold", 75.0)
    energy_fail_threshold = config.get("forecasting", {}).get("energy_fail_threshold", 1000.0)
    
    # Path 1: Trajectory crossing
    trajectory_crossing = health_forecast[health_forecast["ForecastHealth"] <= health_threshold]
    if not trajectory_crossing.empty:
        t1 = trajectory_crossing.iloc[0]["Timestamp"]
        rul_trajectory = (t1 - current_time).total_seconds() / 3600
    else:
        rul_trajectory = np.inf
    
    # Path 2: Hazard accumulation
    hazard_crossing = hazard_df[hazard_df["FailureProb"] >= 0.5]
    if not hazard_crossing.empty:
        t2 = hazard_crossing.iloc[0]["Timestamp"]
        rul_hazard = (t2 - current_time).total_seconds() / 3600
    else:
        rul_hazard = np.inf
    
    # Path 3: Anomaly energy threshold
    energy_crossing = anomaly_energy_df[anomaly_energy_df["CumulativeEnergy"] >= energy_fail_threshold]
    if not energy_crossing.empty:
        t3 = energy_crossing.iloc[0]["Timestamp"]
        rul_energy = (t3 - current_time).total_seconds() / 3600
    else:
        rul_energy = np.inf
    
    # Final RUL = minimum
    rul_final = min(rul_trajectory, rul_hazard, rul_energy)
    if rul_final == np.inf:
        rul_final = config.get("forecasting", {}).get("max_forecast_hours", 168.0)
    
    # Confidence band from trajectory CI
    ci_lower_crossing = health_forecast[health_forecast["CI_Lower"] <= health_threshold]
    ci_upper_crossing = health_forecast[health_forecast["CI_Upper"] <= health_threshold]
    if not ci_lower_crossing.empty and not ci_upper_crossing.empty:
        t_lower = (ci_lower_crossing.iloc[0]["Timestamp"] - current_time).total_seconds() / 3600
        t_upper = (ci_upper_crossing.iloc[0]["Timestamp"] - current_time).total_seconds() / 3600
        confidence_band = abs(t_upper - t_lower)
    else:
        confidence_band = 0.0
    
    # Determine dominant path
    if rul_final == rul_trajectory:
        dominant_path = "trajectory"
    elif rul_final == rul_hazard:
        dominant_path = "hazard"
    else:
        dominant_path = "energy"
    
    return {
        "rul_trajectory_hours": rul_trajectory if rul_trajectory != np.inf else None,
        "rul_hazard_hours": rul_hazard if rul_hazard != np.inf else None,
        "rul_energy_hours": rul_energy if rul_energy != np.inf else None,
        "rul_final_hours": rul_final,
        "confidence_band_hours": confidence_band,
        "dominant_path": dominant_path
    }

# Update ACM_RUL_Summary schema to include new columns
```

**SQL Table Update:** `ACM_RUL_Summary`
```sql
ALTER TABLE ACM_RUL_Summary ADD
    RUL_Trajectory_Hours FLOAT,
    RUL_Hazard_Hours FLOAT,
    RUL_Energy_Hours FLOAT,
    RUL_Final_Hours FLOAT,
    ConfidenceBand_Hours FLOAT,
    DominantPath NVARCHAR(20)
```

**Effort:** 8-10 hours
**Files Modified:** `core/enhanced_rul_estimator.py`, `core/forecasting.py`

---

#### FORECAST-STATE-05: Unified Failure Condition ✅ COMPLETED
**Files:** `docs/RUL_METHOD.md` (new), `scripts/evaluate_rul_backtest.py`, dashboard Panel 36

**Definition:**
```
FAILURE CONDITION (Unified):
A failure event is detected when ANY of the following occurs:

1. SUSTAINED LOW HEALTH: HealthIndex < 75 for >= 4 consecutive hours (>=4 data points @ 1h freq)
2. CRITICAL EPISODE: Episode with Severity='CRITICAL' logged in ACM_CulpritHistory
3. ACUTE ANOMALY: FusedZ >= 3.0 for >= 2 consecutive hours (>=2 data points @ 1h freq)

OPTIONAL PRE-FAILURE MARKER (Early Warning):
- DriftValue > 1.5 AND anomaly_energy_slope > threshold (indicates degradation trend)

RATIONALE:
- Condition 1: Captures gradual degradation (slow health decline)
- Condition 2: Captures known critical events (episode detection)
- Condition 3: Captures sudden acute failures (sensor spikes)
```

**Implementation in evaluate_rul_backtest.py:**
```python
def identify_unified_failures(
    sql_client,
    equip_id: int,
    health_threshold: float = 75.0,
    health_sustain_hours: int = 4,
    fused_z_threshold: float = 3.0,
    fused_z_sustain_hours: int = 2
) -> pd.DataFrame:
    """
    Identify failure events using unified condition.
    Returns DataFrame: [FailureTime, FailureType, Severity]
    """
    failures = []
    
    # Condition 1: Sustained low health
    health_failures = identify_health_threshold_failures(
        sql_client, equip_id, health_threshold, health_sustain_hours
    )
    for ts in health_failures:
        failures.append({"FailureTime": ts, "FailureType": "SUSTAINED_LOW_HEALTH", "Severity": "HIGH"})
    
    # Condition 2: Critical episodes
    critical_episodes = fetch_critical_episodes(sql_client, equip_id)
    for ts in critical_episodes:
        failures.append({"FailureTime": ts, "FailureType": "CRITICAL_EPISODE", "Severity": "CRITICAL"})
    
    # Condition 3: Acute FusedZ spikes
    fused_z_failures = identify_fused_z_spikes(
        sql_client, equip_id, fused_z_threshold, fused_z_sustain_hours
    )
    for ts in fused_z_failures:
        failures.append({"FailureTime": ts, "FailureType": "ACUTE_ANOMALY", "Severity": "HIGH"})
    
    df_failures = pd.DataFrame(failures)
    if df_failures.empty:
        return df_failures
    
    # Remove duplicates within 24h window (same failure event)
    df_failures = df_failures.sort_values("FailureTime")
    df_failures = df_failures[
        df_failures["FailureTime"].diff().dt.total_seconds() / 3600 > 24
    ]
    
    return df_failures
```

**Dashboard Update (Panel 36):**
Add markdown section explaining failure condition in plain language.

**Effort:** 4-5 hours
**Files Modified:** `scripts/evaluate_rul_backtest.py`, `grafana_dashboards/asset_health_dashboard.json`, `docs/RUL_METHOD.md` (new)

---

#### FORECAST-STATE-06: Defect Type Forecasting Display ✅ COMPLETED
**Grafana Panel:** New panel "Predicted Defect Signature"

**Query:**
```sql
-- Show detector contribution breakdown in forecast window
WITH LatestRun AS (
    SELECT MAX(RunID) AS RunID
    FROM ACM_Runs
    WHERE EquipID = $equipment
),
ForecastWindow AS (
    SELECT Timestamp
    FROM ACM_HealthForecast_Continuous
    WHERE EquipID = $equipment
      AND Timestamp >= DATEADD(HOUR, -24, GETDATE())
      AND Timestamp <= DATEADD(HOUR, 24, GETDATE())
)
SELECT 
    ct.Timestamp,
    ct.DetectorName,
    ct.ContributionPct,
    sh.SensorName,
    sh.AbsZScore
FROM ACM_ContributionTimeline ct
CROSS APPLY (SELECT RunID FROM LatestRun) lr
LEFT JOIN ACM_SensorHotspots sh 
    ON sh.RunID = lr.RunID AND sh.EquipID = $equipment
WHERE ct.EquipID = $equipment
  AND ct.RunID = lr.RunID
  AND ct.Timestamp IN (SELECT Timestamp FROM ForecastWindow)
ORDER BY ct.Timestamp, ct.ContributionPct DESC
```

**Visualization:** Stacked area chart showing detector contributions over forecast horizon + table showing top 5 sensor hotspots

**Effort:** 3-4 hours
**Files Modified:** `grafana_dashboards/asset_health_dashboard.json`

---

#### FORECAST-STATE-07: RUL Visualization Panel ✅ COMPLETED
**Grafana Panel:** New panel "RUL Projection with Failure Threshold"

**Query:**
```sql
-- Health forecast with threshold crossing marker
SELECT 
    Timestamp,
    ForecastHealth,
    CI_Lower,
    CI_Upper,
    75.0 AS FailureThreshold
FROM ACM_HealthForecast_Continuous
WHERE EquipID = $equipment
  AND Timestamp >= DATEADD(HOUR, -12, GETDATE())
  AND Timestamp <= DATEADD(HOUR, 48, GETDATE())
ORDER BY Timestamp

-- Add annotation query for projected failure time
SELECT 
    RUL_Final_Hours,
    DATEADD(HOUR, RUL_Final_Hours, GETDATE()) AS ProjectedFailureTime,
    DominantPath,
    ConfidenceBand_Hours
FROM ACM_RUL_Summary
WHERE EquipID = $equipment
  AND RunID = (SELECT MAX(RunID) FROM ACM_Runs WHERE EquipID = $equipment)
```

**Visualization:**
- Time series: ForecastHealth (solid line), CI_Lower/CI_Upper (shaded band), FailureThreshold (red dashed line at 75)
- Vertical marker at ProjectedFailureTime with label "Est. Failure: {RUL_Final_Hours}h ({DominantPath})"
- Shaded region for ConfidenceBand_Hours
- Stat panel showing countdown: "RUL: {RUL_Final_Hours}h ± {ConfidenceBand_Hours}h"

**Effort:** 4-5 hours
**Files Modified:** `grafana_dashboards/asset_health_dashboard.json`

---

#### FORECAST-STATE-08: Retraining Indicator Dashboard ✅ COMPLETED
**Grafana Panel:** New panel "Model Retraining Status"

**SQL Table Update:** `ACM_RunMetadata`
```sql
ALTER TABLE ACM_RunMetadata ADD
    RetrainDecision NVARCHAR(50),  -- "FULL_RETRAIN", "INCREMENTAL_UPDATE", "NO_RETRAIN"
    RetrainReason NVARCHAR(500),
    LastRetrainRunID INT,
    ModelAgeInBatches INT,
    ForecastQualityRMSE FLOAT
```

**Query:**
```sql
SELECT 
    rm.CreatedAt AS Timestamp,
    rm.RetrainDecision,
    rm.RetrainReason,
    rm.ModelAgeInBatches,
    rm.ForecastQualityRMSE,
    CASE 
        WHEN rm.RetrainDecision = 'FULL_RETRAIN' THEN 'Retrained'
        WHEN rm.ModelAgeInBatches > 10 AND rm.RetrainDecision != 'FULL_RETRAIN' THEN 'Retrain Recommended'
        ELSE 'Model Current'
    END AS RetrainStatus
FROM ACM_RunMetadata rm
WHERE rm.EquipID = $equipment
  AND rm.CreatedAt >= $__timeFrom
  AND rm.CreatedAt <= $__timeTo
ORDER BY rm.CreatedAt DESC
```

**Visualization:**
- Table showing last 10 runs with retrain decisions
- Stat panel: "Model Age: {ModelAgeInBatches} batches since retrain"
- Alert threshold annotation when ModelAgeInBatches > 10

**Effort:** 3-4 hours
**Files Modified:** `core/acm_main.py` (log retrain decision to ACM_RunMetadata), `grafana_dashboards/asset_health_dashboard.json`

---

#### FORECAST-STATE-09: Continuous Table Schema 🔴 CRITICAL
**SQL Table:** `ACM_HealthForecast_Continuous`
```sql
CREATE TABLE ACM_HealthForecast_Continuous (
    Timestamp DATETIME2 NOT NULL,
    ForecastHealth FLOAT NOT NULL,
    CI_Lower FLOAT,
    CI_Upper FLOAT,
    SourceRunID INT NOT NULL,  -- RunID that contributed this forecast point
    MergeWeight FLOAT,  -- Temporal blending weight (0-1)
    EquipID INT NOT NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    PRIMARY KEY (EquipID, Timestamp, SourceRunID)
)

CREATE INDEX IX_HealthForecast_TimeRange 
ON ACM_HealthForecast_Continuous(EquipID, Timestamp)
```

**Writer Logic in forecasting.py:**
```python
def write_continuous_health_forecast(
    merged_horizon: pd.DataFrame,  # Already has Timestamp, ForecastHealth, CI_Lower, CI_Upper
    run_id: int,
    equip_id: int,
    output_manager: Any,
    tables_dir: Path
) -> None:
    """
    Write merged forecast horizon to ACM_HealthForecast_Continuous.
    Append-only, no deletion of old forecasts (pruning handled separately).
    """
    df_write = merged_horizon.copy()
    df_write["SourceRunID"] = run_id
    df_write["EquipID"] = equip_id
    df_write["MergeWeight"] = 1.0  # Default weight, can be refined
    
    output_manager.write_dataframe(
        df_write,
        tables_dir / "health_forecast_continuous.csv",
        sql_table="ACM_HealthForecast_Continuous",
        add_created_at=True
    )
    
    # Prune forecasts older than 7 days (cleanup)
    if output_manager.sql_client:
        try:
            cur = output_manager.sql_client.cursor()
            cutoff_time = datetime.now() - timedelta(days=7)
            cur.execute("""
                DELETE FROM ACM_HealthForecast_Continuous
                WHERE EquipID = ? AND Timestamp < ?
            """, (equip_id, cutoff_time))
            output_manager.sql_client.conn.commit()
        except Exception as e:
            Console.warn(f"[FORECAST] Failed to prune old forecasts: {e}")
```

**Effort:** 3-4 hours
**Files Modified:** `core/forecasting.py`, SQL schema script

---

### Implementation Priority & Effort Summary

| Task ID | Description | Priority | Effort | Files Modified | Status |
|---------|-------------|----------|--------|----------------|--------|
| FORECAST-STATE-01 | ForecastState persistence class | 🔴 CRITICAL | 4-6h | model_persistence.py, acm_main.py | Completed |
| FORECAST-STATE-02 | Continuous forecasting logic | 🔴 CRITICAL | 10-14h | forecasting.py, acm_main.py | Completed |
| FORECAST-STATE-03 | Hazard-based probability smoothing | 🔴 CRITICAL | 6-8h | forecasting.py | Completed |
| FORECAST-STATE-04 | Multi-path RUL derivation | 🔴 CRITICAL | 8-10h | enhanced_rul_estimator.py, forecasting.py | Completed |
| FORECAST-STATE-05 | Unified failure condition | 🔴 CRITICAL | 4-5h | evaluate_rul_backtest.py, docs/RUL_METHOD.md, dashboard | Completed |
| FORECAST-STATE-06 | Defect type display | 🟡 MEDIUM | 3-4h | dashboard JSON | Completed |
| FORECAST-STATE-07 | RUL visualization panel | 🔴 CRITICAL | 4-5h | dashboard JSON | Completed |
| FORECAST-STATE-08 | Retraining indicator | 🟡 MEDIUM | 3-4h | acm_main.py, dashboard JSON | Completed (Panel added; DB write pending) |
| FORECAST-STATE-09 | Continuous table schema | 🔴 CRITICAL | 3-4h | forecasting.py, SQL scripts | Completed |

**Total Effort:** 45-63 hours (6-8 developer days)

**Critical Path:** STATE-01 → STATE-02 → STATE-03 → STATE-04 → STATE-09 → STATE-07

---

### Testing Strategy
1. **Unit tests:** Test forecast state save/load, horizon merging, hazard smoothing math
2. **Integration test:** Run 10 consecutive batches, verify state persistence, check probability curve smoothness
3. **Validation:** Compare retrain frequency (expect <30% of batches with full retrain)
4. **Dashboard smoke test:** Verify all new panels render without errors

### Rollback Plan
- ForecastState persistence is additive (doesn't break existing pipeline)
- If continuous forecasting fails, fallback to batch-isolated mode (controlled by config flag: `forecasting.enable_continuous: false`)
- Keep both ACM_FailureForecast_TS (old) and ACM_HealthForecast_Continuous (new) during transition

---

RECENTLY COMPLETED (For Context)
Fixed batch data truncation (60% data loss eliminated)
Fixed Unicode encoding errors in Heartbeat spinner
Comprehensive 46-table analysis (92/100 data quality score)
SQL logging infrastructure (SQL-57 complete)
All 40 core analytics tables actively populating
Forecasting & RUL tables working (1,104+ rows each)
Health timeline, sensor hotspots, regime detection all operational
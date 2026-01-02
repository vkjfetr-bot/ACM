# ACM Main Script - Code Reduction and Optimization Analysis

**Version**: v11.1.6  
**File**: `core/acm_main.py`  
**Total Lines**: 4804 (after Phase 4 cleanup - was 5906)  
**Analysis Date**: January 2, 2026  
**Analysis Type**: Working Code Optimization (Not Just "Unused" Detection)

---

## Completed Refactoring Summary

| Phase | Description | Lines Removed |
|-------|-------------|---------------|
| Phase 1 | Dead code deletion | 583 |
| Phase 2 | DataFrame builders → output_manager.py | 561 |
| Phase 3 | Log consolidation | ~260 |
| Phase 4 | Unused V11 scaffolding removal | 107 |
| **TOTAL** | | **~1100 lines (18.6%)** |

### Phase 4 Removals (V11 Bloat)

1. **RunContext dataclass** (41 lines) - Never instantiated, had TODO noting deferral
2. **Transfer Learning block** (65 lines) - `transfer_result` computed but never applied downstream
3. **ACM_FORCE_FILE_MODE escape hatch** (3 lines) - File mode deprecated, SQL-only mandatory
4. **MHAL stale comments** (2 lines) - Removal notices from v9.1.0

### ONLINE/OFFLINE Mode Status: ✅ CORRECTLY WIRED

The pipeline mode gates are working correctly:
- `ALLOWS_MODEL_REFIT` (line 3266): Fails fast if detectors missing in ONLINE mode
- `ALLOWS_REGIME_DISCOVERY` (line 3515): Passed to regimes.label() which respects it
- `regimes.py` (line 2689): RuntimeError if model missing and discovery not allowed

---

## Executive Summary

This analysis focuses on **reducing working code** through optimization, consolidation, and architectural improvements. The goal is NOT to remove "unused" code blindly, but to understand WHY each piece exists and whether it can be simplified or moved.

**Philosophy Shift**:
- ❌ OLD: "This function is unused → DELETE"
- ✅ NEW: "This function has purpose X → Can we achieve X more efficiently?"

**Key Categories**:
1. **REDUNDANT**: Multiple implementations of the same logic
2. **VERBOSE**: Working code that can be simplified without losing functionality
3. **MISPLACED**: Code that works but belongs in a different module
4. **CONSOLIDATABLE**: Duplicate patterns that can be unified

**Optimization Opportunities**: Additional refactoring possible (helper functions, SQL write consolidation)

---

## Logging Inventory Analysis (Updated Post-Phase 4)

### Current State

| Log Type | Count | Purpose |
|----------|-------|---------|
| Console.info | 35 | **Pipeline progress** - status updates for each phase |
| Console.warn | 76 | **Exception handling** - mostly in try/except blocks |
| Console.error | 11 | **Critical failures** - run-stopping errors |
| Console.ok | 1 | **Success milestones** - model promotion |
| **TOTAL** | 123 | |

### Logging Principles (SINGLE SOURCE OF TRUTH)

**Log Level Semantic Contract:**

| Level | When to Use | Goes to Loki? | Example |
|-------|-------------|---------------|---------|
| `Console.info` | Phase milestones, normal progress | ✅ YES | "Thresholds calculated: alert=3.5" |
| `Console.warn` | Exception handlers, fallbacks, skips | ✅ YES | "Cache invalid, re-fitting" |
| `Console.error` | Critical failures, run-stopping | ✅ YES | "DataContract validation FAILED" |
| `Console.ok` | Major success milestones (rare) | ✅ YES | "Model promoted LEARNING→CONVERGED" |
| `Console.status` | Console-only progress (no Loki) | ❌ NO | Timer output, spinners |
| `Console.header` | Section dividers (console-only) | ❌ NO | "===== PHASE 2 =====" |

### Console.info Audit (35 calls)

| # | Line | Component | Message Pattern | Verdict |
|---|------|-----------|-----------------|---------|
| 1 | 483 | CFG | `Loaded config from SQL for {equip}` | ⚠️ DUPE - 493 also logs config |
| 2 | 493 | CFG | `Config loaded: source={source} \| equip={equip}` | ✅ KEEP |
| 3 | 604 | THRESHOLD | `Adaptive thresholds disabled` | ✅ KEEP |
| 4 | 669 | THRESHOLD | `Thresholds calculated: per_regime=True...` | ✅ KEEP |
| 5 | 671 | THRESHOLD | `Thresholds calculated: alert=... warn=...` | ✅ KEEP |
| 6 | 714 | RUN | `Run started: {equip} (ID={id}) \| RunID=...` | ✅ KEEP (critical) |
| 7 | 822 | SCORE | `Scored {n} detectors: ar1, pca...` | ✅ KEEP |
| 8 | 997 | TRAIN | `Fitted {n} detectors in {t}s` | ✅ KEEP |
| 9 | 1283 | MODEL | `Using cached models v{version}` | ✅ KEEP |
| 10 | 1377 | MODEL | `Saved all trained models to v{version}` | ✅ KEEP |
| 11 | 1728 | DRIFT | `Drift: P95={x} \| trend={y}` | ✅ KEEP |
| 12 | 1734 | DRIFT | `Drift: P95={x} \| threshold={y}` | ⚠️ DUPE - merge with 1728 |
| 13 | 1777 | FEAT | `Feature building disabled in config` | ✅ KEEP |
| 14 | 1799 | FEAT | `Features built: window={w} \| train={shape}` | ✅ KEEP |
| 15 | 2009 | BASELINE | `Baseline: {used} \| extended={bool}` | ✅ KEEP |
| 16 | 2178 | AUTO-TUNE | `Auto-tune: {n} adjustments` | ✅ KEEP |
| 17 | 2472 | CFG | `Config: equip={equip} \| sig={hash}` | ⚠️ DUPE with 493 |
| 18 | 2537 | RUN | `Batch {n}/{total} \| {equip} \| mode=...` | ✅ KEEP (critical) |
| 19 | 2619 | RUN | `CLI overrides: {list}` | ✅ KEEP |
| 20 | 2660 | OUTPUT | `Output: sql_client=attached \| health=ok` | ✅ KEEP |
| 21 | 2759 | COLDSTART | `Coldstart deferred - insufficient data` | ✅ KEEP |
| 22 | 2861 | DATA | `Data: train={n} \| score={m} \| sensors={k}` | ✅ KEEP |
| 23 | 2947 | SEASON | `Seasonal: {n} patterns in {m} sensors` | ✅ KEEP |
| 24 | 3169 | REGIME | `Regime loaded: source={src} \| K={k}` | ✅ KEEP |
| 25 | 3310 | ADAPTIVE | `Adaptive: {result} \| adjustments={n}` | ✅ KEEP |
| 26 | 3498 | REGIME_STATE | `Regime state: saved_v{n} \| K={k}` | ✅ KEEP |
| 27 | 3551 | REGIME | `Regime analysis: occupancy={n} \| transitions={m}` | ✅ KEEP |
| 28 | 3777 | LIFECYCLE | `Model state: {maturity}` | ✅ KEEP |
| 29 | 3963 | CAL | `Calibration complete: q={q} \| clip_z={z}` | ✅ KEEP |
| 30 | 4107 | FUSE | `Fusion: detectors={n} \| episodes={m}` | ✅ KEEP |
| 31 | 4248 | THRESHOLD | `Threshold: reason={r} \| source={s}` | ✅ KEEP |
| 32 | 4285 | REGIME | `Regime: quality_ok={bool} \| states={dict}` | ✅ KEEP |
| 33 | 4733 | OUTPUTS | `Analytics: {status} \| tables={n}` | ✅ KEEP |
| 34 | 4759 | FORECAST | `Forecast: RUL={h}h \| confidence={c}` | ✅ KEEP |
| 35 | 4957 | RUN | `Finalized RunID={id} outcome={o}` | ✅ KEEP (critical) |

**Optimization Opportunities (Console.info):**
- Lines 483 + 493: Merge into single config log
- Lines 1728 + 1734: Merge into single drift log
- Line 2472: Remove - duplicates line 493

### Console.warn Categorization (76 calls)

| Category | Count | Examples | Action |
|----------|-------|----------|--------|
| **SQL Write Failures** | 12 | `DriftSeries write skipped`, `RegimeEpisodes write skipped` | ✅ KEEP - each table needs handler |
| **Model/Cache Issues** | 14 | `Cache invalid`, `Failed to load cached models`, `Incomplete cache` | ✅ KEEP - diagnostic |
| **Data/Validation** | 10 | `DataContract warnings`, `Score window empty`, `Low variance` | ✅ KEEP - data quality |
| **Config/Parse Fallbacks** | 8 | `Could not load from SQL`, `Failed to parse --start-time` | ✅ KEEP - user-facing |
| **Feature Skips** | 8 | `fast_features not available`, `Seasonality skipped`, `Drift skipped` | ⚠️ REVIEW - info level? |
| **Regime Issues** | 10 | `Quality degraded`, `Forcing retraining`, `Different feature columns` | ✅ KEEP - ML diagnostics |
| **Runtime Degradation** | 8 | `Steps skipped`, `Calibration summary failed`, `Output failed` | ✅ KEEP - operations |
| **Other** | 6 | `Observability init failed`, `Finalize skipped` | ✅ KEEP |

**Optimization Opportunities (Console.warn):**
- "Feature skip" category (8 logs) could be downgraded to info when feature is intentionally disabled
- Consider: if `enabled=False` in config → info, if exception → warn

### Console.error Audit (11 calls)

| Line | Component | Condition | Verdict |
|------|-----------|-----------|---------|
| 679 | THRESHOLD | Adaptive threshold calculation failed | ✅ KEEP |
| 2248 | safe_step | Generic step failure handler | ✅ KEEP |
| 2622 | RUN | Failed to start SQL run | ✅ KEEP (critical) |
| 2659 | OUTPUT | SQL health check failed | ✅ KEEP (critical) |
| 2807 | DATA | DataContract validation FAILED | ✅ KEEP (critical) |
| 3307 | ADAPTIVE | Failed to update config SQL | ✅ KEEP |
| 4705 | ANALYTICS | Error generating analytics | ✅ KEEP (critical) |
| 4784 | FORECAST | Unified forecasting engine failed | ✅ KEEP (critical) |
| 4786 | FORECAST | RUL estimation skipped | ⚠️ DUPE with 4784 |
| 4855 | RUN | Main exception | ✅ KEEP (critical) |
| 4986 | RUN | Finalize failed (finally) | ✅ KEEP |

**Optimization: Lines 4784 + 4786 should be merged into single error log.**

---

## SQL Write Delegation Status

### OutputManager Delegation (GOOD - 37 calls)

All SQL writes properly delegate to `output_manager.write_*()` methods:
- `write_threshold_metadata` (2 calls)
- `write_pca_metrics` (2 calls)
- `write_pca_model`, `write_pca_loadings`
- `write_feature_drop_log`, `write_refit_request`
- `write_drift_series`, `write_anomaly_events`, `write_regime_episodes`
- `write_run_stats`, `write_data_contract_validation`
- `write_dataframe` (generic - 6 calls)
- `write_regime_definitions`, `write_regime_occupancy`, `write_regime_transitions`
- `write_regime_promotion_log`, `write_active_models`
- `write_calibration_summary`, `write_fusion_metrics`
- `write_drift_controller`, `write_contribution_timeline`
- `write_scores`, `write_episodes`
- `write_detector_correlation`, `write_sensor_correlations`
- `write_sensor_normalized_ts`, `write_seasonal_patterns`, `write_asset_profile`

### Inline SQL Remaining (ACCEPTABLE - 3 locations)

| Location | Purpose | Action |
|----------|---------|--------|
| Line 725 `_sql_finalize_run` | Update ACM_Runs CompletedAt | ⚠️ Could move to OutputManager |
| Line 1939 `_seed_baseline` | Read from ACM_BaselineBuffer | ✅ KEEP - read operation |
| Line 2655 health check | SELECT 1 connectivity test | ✅ KEEP - diagnostic |

**Recommendation**: Move `_sql_finalize_run` logic to OutputManager.finalize_run()

---

## Dummy Filesystem Variables (CANNOT REMOVE)

These variables are set to `Path(".")` but ARE passed to functions:
- `run_dir` - used in weight tuning warm start (line 4016)
- `art_root` - passed to model loading/saving functions (10+ uses)
- `models_dir`, `stable_models_dir` - passed to regime loading

**Verdict**: Cannot remove without breaking function signatures. Low priority cleanup.

---

## Actionable Optimizations

### Immediate (Low Risk)

| Item | Lines | Change |
|------|-------|--------|
| Merge CFG info logs (483+493) | -1 | Single config load message |
| Merge DRIFT info logs (1728+1734) | -1 | Single drift status message |
| Remove duplicate CFG log (2472) | -1 | Already logged at 493 |
| Merge FORECAST error logs (4784+4786) | -1 | Single error with full context |
| **TOTAL** | -4 | |

### Medium Term

| Item | Lines | Change |
|------|-------|--------|
| Move `_sql_finalize_run` to OutputManager | -20 | Architectural cleanup |
| Downgrade "feature disabled" warns to info | 0 | Semantic correctness |

# AFTER (only copy if source is reused)
if source_will_be_reused:
    episodes = episodes.copy()
episodes["new_col"] = ...
```

**Estimated Savings**: 15-20 lines of `.copy()` calls can be removed

---

### 1.2 Redundant Type Checking - **Verbose Guards**

**Pattern Found**: Repeated `isinstance(df, pd.DataFrame)` checks

**Examples**:
```python
# Line 1817
episodes = (episodes if isinstance(episodes, pd.DataFrame) else pd.DataFrame()).copy()

# Line 2502
if not isinstance(train, pd.DataFrame):
    return train, score, []

# Line 2561
if isinstance(score, pd.DataFrame) and len(score) > 0:
```

**Analysis**:
- **WHY IT EXISTS**: Prevent AttributeError when function receives None or wrong type
- **IS IT NEEDED?**: If upstream guarantees DataFrame, no
- **CONSOLIDATION**: Use type hints + early validation instead of per-operation checks

**Recommendation**:
```python
# BEFORE: Check at every operation (scattered)
if isinstance(train, pd.DataFrame):
    train.fillna(...)
if isinstance(score, pd.DataFrame):
    score.fillna(...)

# AFTER: Validate once at function entry
def _impute_features(train: pd.DataFrame, score: pd.DataFrame, ...) -> ...:
    if not isinstance(train, pd.DataFrame) or not isinstance(score, pd.DataFrame):
        raise TypeError("train and score must be DataFrames")
    
    # Now all operations below can assume DataFrame
    train.fillna(...)
    score.fillna(...)
```

**Estimated Savings**: Consolidate 30+ scattered type checks into 10 function-entry validators

---

### 1.3 Repetitive Index Handling - **Duplicate Logic**

**Pattern Found**: Same index conversion logic repeated in multiple functions

**Examples**:
```python
# Pattern 1: tz_localize(None) appears 5+ times
idx_local = pd.DatetimeIndex(to_append.index).tz_localize(None)  # Line 1354
buf.index = pd.to_datetime(buf.index).tz_localize(None)  # Line 2556

# Pattern 2: ensure_local_index helper exists but not always used
def _ensure_local_index(df: pd.DataFrame) -> pd.DataFrame:  # Line 595
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    # ... tz handling
```

**Analysis**:
- **WHY IT EXISTS**: Centralized helper to normalize timestamps
- **IS IT NEEDED?**: Yes
- **PROBLEM**: Helper exists but is NOT consistently used (inline duplication)

**Evidence of Inconsistent Usage**:
- `_ensure_local_index()` defined at line 595
- Same logic repeated inline at lines 1354, 2556, 2524, etc.

**Recommendation**:
Replace all inline tz_localize logic with `_ensure_local_index()` helper

**Estimated Savings**: Remove 20-30 lines of duplicate index normalization

---

## Section 2: SQL Write Helpers - **Architectural Misplacement**

### 2.1 SQL Logic in acm_main.py - **Should Be in OutputManager**

**Pattern Found**: Direct SQL writes embedded in acm_main helpers

**Examples**:
```python
# Line 1718: _write_fusion_metrics - 80 lines
def _write_fusion_metrics(...) -> bool:
    # Builds metrics_rows
    # Inserts to ACM_RunMetrics via raw SQL
    insert_sql = """INSERT INTO dbo.ACM_RunMetrics ..."""
    cur.executemany(insert_sql, insert_records)

# Line 1804: _write_data_quality - 40 lines
def _write_data_quality(...) -> bool:
    # Builds quality records
    # Inserts to ACM_DataQuality via raw SQL
    insert_sql = """INSERT INTO dbo.ACM_DataQuality ..."""

# Line 1866: _log_dropped_features - 35 lines
def _log_dropped_features(...) -> bool:
    # Inserts to ACM_FeatureDropLog via raw SQL
```

**Analysis**:
- **WHY IT EXISTS**: SQL persistence is required for observability
- **IS IT NEEDED?**: Yes
- **PROBLEM**: SQL logic belongs in `core/output_manager.py`, not acm_main.py
- **CONSEQUENCE**: acm_main.py has 155+ lines of SQL write code that duplicates OutputManager's role

**Evidence**:
- `OutputManager` class exists (imported line 57)
- `OutputManager` has methods like `write_pca_metrics()`, `write_pca_loadings()`
- But fusion metrics, data quality, feature drops still use inline SQL in acm_main

**Recommendation**:
Move all SQL write logic to OutputManager:
```python
# BEFORE (acm_main.py)
def _write_fusion_metrics(...):
    insert_sql = """INSERT INTO ..."""
    cur.executemany(insert_sql, records)

# AFTER (move to output_manager.py)
class OutputManager:
    def write_fusion_metrics(self, metrics_rows: List[Dict]) -> bool:
        insert_sql = """INSERT INTO ..."""
        self.sql_client.cursor().executemany(insert_sql, records)

# THEN (acm_main.py becomes clean)
output_manager.write_fusion_metrics(metrics_rows)
```

**Estimated Savings**: Move 155 lines from acm_main.py to output_manager.py (architectural cleanup)

---

### 2.2 DataFrame Building Helpers - **Should Be in Output Manager**

**Pattern Found**: Helper functions that build DataFrames for SQL tables

**Examples**:
```python
# Line 2695: _build_health_timeline - 70 lines
def _build_health_timeline(frame, cfg, run_id, equip_id) -> pd.DataFrame:
    # Compute health index
    # Format for ACM_HealthTimeline
    return health_df

# Line 2824: _build_regime_timeline - 20 lines
def _build_regime_timeline(frame, regime_model, run_id, equip_id) -> pd.DataFrame:
    # Format for ACM_RegimeTimeline
    return regime_df

# Line 2842: _build_drift_ts - 20 lines
# Line 2858: _build_anomaly_events - 20 lines
# Line 2877: _build_regime_episodes - 20 lines
```

**Analysis**:
- **WHY IT EXISTS**: Transform pipeline data into SQL table schemas
- **IS IT NEEDED?**: Yes
- **PROBLEM**: These are OUTPUT formatting functions, not PIPELINE logic
- **BELONGS IN**: `output_manager.py` as private methods

**Recommendation**:
```python
# MOVE to output_manager.py
class OutputManager:
    def write_health_timeline(self, frame: pd.DataFrame, cfg: Dict, ...) -> int:
        health_df = self._build_health_timeline_df(frame, cfg)
        return self.write_dataframe(health_df, "ACM_HealthTimeline")
    
    def _build_health_timeline_df(self, frame, cfg) -> pd.DataFrame:
        # Current _build_health_timeline logic
        ...
```

**Estimated Savings**: Move 150 lines of DataFrame builders to output_manager.py

---

## Section 3: Verbose Error Handling - **Can Be Consolidated**

### 3.1 Try-Except Pyramid - **Nested Error Handling**

**Pattern Found**: Deeply nested try-except blocks

**Example Structure**:
```python
def main():
    try:
        # Initialize
        try:
            # Load data
            try:
                # Score detectors
            except Exception as e3:
                Console.error(...)
        except Exception as e2:
            Console.error(...)
    except Exception as e1:
        Console.error(...)
    finally:
        # Cleanup
```

**Analysis**:
- **WHY IT EXISTS**: Granular error handling for different pipeline phases
- **IS IT NEEDED?**: Yes, but can be flattened
- **PROBLEM**: Nesting makes control flow hard to trace

**Recommendation**:
Use phase functions with early returns:
```python
def main():
    try:
        ctx = _initialize()
        data_ctx = _load_data(ctx)
        if data_ctx is None:
            return  # Early exit on failure
        
        model_ctx = _train_models(data_ctx, ctx)
        if model_ctx is None:
            return
        
        # ... continue
    finally:
        _cleanup()
```

**Estimated Savings**: Reduce nesting depth from 4 levels to 1 (readability, not lines)

---

### 3.2 Duplicate Console Logging - **Verbose Status Messages**

**Pattern Found**: Excessive logging at every step

**Examples**:
```python
Console.info(f"Calculating adaptive thresholds from {len(fused_scores)} samples...", component="THRESHOLD")
Console.info(f"Per-regime thresholds: {threshold_results['fused_alert_z']}", component="THRESHOLD")
Console.info("SQL write completed for fused_alert_z", component="THRESHOLD")
Console.info("SQL write completed for fused_warn_z", component="THRESHOLD")
```

**Analysis**:
- **WHY IT EXISTS**: Observability and debugging
- **IS IT NEEDED?**: Partially
- **PROBLEM**: 4 log lines for a single operation (threshold calculation)

**Recommendation**:
Consolidate related logs:
```python
# BEFORE: 4 log lines
Console.info("Calculating...")
Console.info("Per-regime thresholds...")
Console.info("SQL write completed for alert_z")
Console.info("SQL write completed for warn_z")

# AFTER: 1 summary log
Console.info(
    f"Adaptive thresholds: alert={alert_z:.3f}, warn={warn_z:.3f} | "
    f"SQL writes: 2/2 OK | samples={len(fused_scores)}",
    component="THRESHOLD"
)
```

**Estimated Savings**: Reduce 200+ log lines to ~80 consolidated logs

---

## Section 4: Helper Function Analysis - **Purpose-Driven Evaluation**

### 4.1 RunContext Dataclass - **Unfinished v11 Feature (NOT Bloat)**

**Status**: ⚠️ **UNFINISHED** (Has explicit TODO)

**Code** (Lines 295-335):
```python
@dataclass
class RunContext:
    """
    NOTE (v11.1.6): This dataclass is defined but NOT YET fully integrated.
    Currently main() uses a local `degradations: List[str]` instead of 
    instantiating RunContext. Full integration is deferred to avoid a 
    large refactor. For now, the local list approach works correctly.
    
    TODO: Refactor main() to instantiate RunContext and pass it through
    the pipeline, replacing the scattered local variables.
    """
```

**Analysis**:
- **WHY IT EXISTS**: Part of v11 refactor to consolidate run context
- **IS IT NEEDED?**: YES - for v11 architecture goals
- **CURRENT STATE**: Partially designed, not yet integrated
- **RECOMMENDATION**: Either complete integration in v11.2 or defer to v12

**Verdict**: **KEEP** (unfinished != bloat if planned)

---

### 4.2 _compute_drift_trend & _compute_regime_volatility - **Future Multi-Feature Drift**

**Status**: ⚠️ **UNFINISHED** (But CALLED in multi-feature drift mode)

**Code** (Lines 422-468):
```python
def _compute_drift_trend(drift_series: np.ndarray, window: int = 20) -> float:
    """Compute drift trend as slope..."""

def _compute_regime_volatility(regime_labels: np.ndarray, window: int = 20) -> float:
    """Compute regime volatility..."""
```

**CRITICAL FINDING**: These ARE used! In `_compute_drift_alert_mode()`:
```python
# Line 2391
drift_trend = _compute_drift_trend(drift_array, window=trend_window)

# Line 2398
regime_volatility = _compute_regime_volatility(regime_labels, window=trend_window)
```

**Analysis**:
- **WHY IT EXISTS**: Multi-feature drift detection (DRIFT-01 spec)
- **IS IT NEEDED?**: YES - used when `multi_feature.enabled=True`
- **USAGE**: Called from `_compute_drift_alert_mode()` helper

**Verdict**: **KEEP** (NOT bloat - actively used in multi-feature mode)

---

### 4.3 _maybe_write_run_meta_json - **Legacy File-Mode Stub (TRUE Bloat)**

**Status**: ❌ **DELETABLE** (No-op in all cases)

**Code** (Lines 407-418):
```python
def _maybe_write_run_meta_json(local_vars: Dict[str, Any]) -> None:
    """
    Legacy file-mode metadata writer stub.
    
    In SQL-only mode (v10+), metadata is written to ACM_Runs table directly.
    This function is kept for backward compatibility but is a no-op in SQL mode.
    """
    if bool(local_vars.get('SQL_MODE')):
        return  # No-op in SQL mode
    # File mode is deprecated; do nothing
    pass
```

**Analysis**:
- **WHY IT EXISTS**: Backward compatibility with file mode
- **IS IT NEEDED?**: NO - file mode removed in v10
- **CURRENT STATE**: Does nothing in all execution paths

**Verdict**: **DELETE** (12 lines of dead code)

---

### 4.4 _calculate_adaptive_thresholds - **Architectural Bloat**

**Status**: 🟠 **REFACTORABLE** (Works but belongs in adaptive_thresholds.py)

**Code** (Lines 644-791): 148 lines

**Structure**:
```python
def _calculate_adaptive_thresholds(...) -> Dict[str, Any]:
    try:
        from core.adaptive_thresholds import calculate_thresholds_from_config
        
        # 1. Call imported function (10 lines)
        threshold_results = calculate_thresholds_from_config(...)
        
        # 2. Write to SQL (100 lines of SQL logic)
        if output_manager is not None:
            output_manager.write_threshold_metadata(...)
            output_manager.write_threshold_metadata(...)  # Again for warn_z
        
        # 3. Update config dict (30 lines)
        cfg["regimes"]["health"]["fused_alert_z"] = ...
        
        return threshold_results
    except Exception as e:
        # Error handling (8 lines)
```

**Analysis**:
- **WHY IT EXISTS**: Consolidates threshold calculation + SQL persistence
- **IS IT NEEDED?**: Yes, but NOT in acm_main.py
- **PROBLEM**: Duplicates OutputManager's role
- **SOLUTION**: Move SQL logic to OutputManager, keep only the orchestration call

**Refactor**:
```python
# BEFORE: 148 lines in acm_main.py
def _calculate_adaptive_thresholds(...):
    threshold_results = calculate_thresholds_from_config(...)
    # 100 lines of SQL writes
    # 30 lines of config updates
    return threshold_results

# AFTER: 15 lines in acm_main.py
def _calculate_adaptive_thresholds(...):
    threshold_results = calculate_thresholds_from_config(...)
    output_manager.write_adaptive_thresholds(threshold_results, ...)
    _update_config_from_thresholds(cfg, threshold_results)
    return threshold_results

# SQL logic moves to output_manager.py
class OutputManager:
    def write_adaptive_thresholds(self, results: Dict, ...) -> None:
        # 100 lines of SQL write logic
```

**Estimated Savings**: Reduce from 148 lines to 15 lines in acm_main.py (133 lines moved)

---

## Section 5: Optimization Summary Tables

### 5.1 Code Reduction Opportunities (By Category)

| Category | Current Lines | Optimized Lines | Savings | Method |
|----------|---------------|-----------------|---------|--------|
| SQL Write Helpers | 155 | 30 | **-125** | Move to OutputManager |
| DataFrame Builders | 150 | 30 | **-120** | Move to OutputManager |
| _calculate_adaptive_thresholds | 148 | 15 | **-133** | Refactor SQL to OutputManager |
| Duplicate Index Handling | 30 | 5 | **-25** | Use _ensure_local_index() |
| Type Checking Guards | 40 | 15 | **-25** | Consolidate to function entry |
| Console Logging | 200 | 80 | **-120** | Consolidate related logs |
| Unnecessary .copy() | 50 | 30 | **-20** | Remove when source discarded |
| **TOTAL ESTIMATED** | **773** | | **-568** | |

**Note**: These are working code lines, not dead code. Savings come from architectural improvements.

---

### 5.2 Unfinished v11 Features (NOT Bloat)

| Feature | Lines | Status | Verdict |
|---------|-------|--------|---------|
| RunContext | 41 | Defined, not integrated | FINISH in v11.2 or defer to v12 |
| _compute_drift_trend | 28 | **USED** in multi-feature drift | **KEEP** |
| _compute_regime_volatility | 17 | **USED** in multi-feature drift | **KEEP** |
| Seasonality (partial) | 40 | Computes but not wired to forecasting | FINISH integration |
| Asset Similarity (partial) | 30 | Profiles stored, transfer learning missing | FINISH or remove |
| Data Contract (partial) | 30 | Validates but doesn't fail-fast | FINISH or remove |

**Key Insight**: These aren't bloat - they're scaffolding for v11 features. Decision needed: finish or defer.

---

### 5.3 True Bloat (Deletable Dead Code)

| Item | Lines | Reason |
|------|-------|--------|
| _maybe_write_run_meta_json | 12 | No-op in all execution paths |
| MHAL removal comments | 11 | Feature removed in v9.1.0, stale comments |
| Filesystem dummy vars | 3 | Created but unused (SQL-only mode) |
| **TOTAL** | **26** | |

---

## Section 6: Architectural Recommendations

### 6.1 Immediate Wins (v11.2.0 - Low Risk)

**DELETE**:
1. `_maybe_write_run_meta_json()` - 12 lines
2. All MHAL comments - 11 instances
3. Filesystem dummy vars - 3 lines
**Savings**: 26 lines, zero risk

---

### 6.2 Medium-Term Refactoring (v11.2.0 - Moderate Risk)

**MOVE TO OUTPUT_MANAGER.PY**:
1. All SQL write helpers (_write_fusion_metrics, _write_data_quality, _log_dropped_features)
2. All DataFrame builders (_build_health_timeline, _build_regime_timeline, etc.)
3. SQL logic from _calculate_adaptive_thresholds

**Impact**:
- acm_main.py: **-455 lines**
- output_manager.py: +455 lines
- **Net**: Architectural cleanup, no code deletion

---

### 6.3 Optimization Refactoring (v11.3.0 - Requires Testing)

**CONSOLIDATE**:
1. Replace inline index handling with `_ensure_local_index()` (-25 lines)
2. Consolidate type checks to function entry (-25 lines)
3. Remove unnecessary `.copy()` calls (-20 lines)
4. Merge related Console.info() logs (-120 lines)

**Impact**:
- acm_main.py: **-190 lines**
- **Risk**: Requires regression testing

---

### 6.4 v11 Feature Completion (v12.0.0 - High Effort)

**FINISH OR REMOVE**:
1. RunContext integration - integrate or delete (41 lines)
2. Seasonality forecasting - wire to ForecastEngine or remove seasonal adjustment
3. Asset Similarity - implement cold-start transfer or remove profile storage
4. Data Contract - enforce fail-fast or remove validation

**Decision Required**: Each feature needs product/roadmap decision

---

## Section 7: Final Verdict

### 7.1 Total Optimization Potential

| Phase | Lines Saved | Risk Level | Timeline |
|-------|-------------|------------|----------|
| Delete dead code | 26 | ✅ None | Immediate |
| Move to OutputManager | 455 | 🟡 Low | v11.2 |
| Consolidation | 190 | 🟠 Medium | v11.3 |
| **TOTAL** | **671** | | |

**Baseline**: 5906 lines  
**Optimized**: ~5235 lines  
**Reduction**: 11.4%

---

### 7.2 Key Insights

1. **Most "bloat" is actually working code** - just misplaced architecturally
2. **"Unused" helpers are often used** - in conditional paths (drift mode, etc.)
3. **Unfinished v11 features aren't bloat** - they're scaffolding awaiting completion
4. **True dead code is minimal** - only 26 lines of no-ops and stale comments

---

### 7.3 Recommended Approach

**Phase 1 (This PR)**: Delete 26 lines of confirmed dead code  
**Phase 2 (v11.2)**: Move 455 lines to OutputManager (architectural cleanup)  
**Phase 3 (v11.3)**: Consolidate duplicate logic (190 lines)  
**Phase 4 (v12.0)**: Finish or remove partial v11 features

**Goal**: Not to make the file smaller by deleting features, but to make it **cleaner by organizing** what exists.

---

**END OF ANALYSIS**

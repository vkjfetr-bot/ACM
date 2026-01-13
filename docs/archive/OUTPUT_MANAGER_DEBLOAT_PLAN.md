# Output Manager Debloat Plan

**File**: `core/output_manager.py`  
**Current Size**: 3,676 lines (was 4,770)  
**Target Size**: ~2,500-3,000 lines  
**Total Reduction So Far**: 1,094 lines (23%)  
**Branch**: `feature/phase3-analytics-builder`

---

## Executive Summary

| Category | Lines | Action | Status |
|----------|-------|--------|--------|
| Dead Code | ~466 | DELETE - never called | ✅ DONE |
| Data Loading (wrong place) | ~277 | MOVE to `core/data_loader.py` | ✅ DONE |
| Analytics Generation | ~351 | MOVE to `core/analytics_builder.py` | ✅ DONE |
| Utility Functions (top-level) | ~150 | MOVE to `utils/` or inline | FUTURE |
| Upsert Consolidation | ~200 | Consolidate patterns | FUTURE |

---

## Phase 1: Delete Dead Code (~500 lines) [COMPLETE]

### 1.1 Truly Dead Methods (never called anywhere)

| Method | Lines | Reason Dead | Status |
|--------|-------|-------------|--------|
| `_build_health_timeline` | ~83 | Replaced by `_generate_health_timeline` | DELETED |
| `_build_regime_timeline` | ~38 | Replaced by `_generate_regime_timeline` | DELETED |
| `_build_drift_ts` | ~30 | Never called | DELETED |
| `_build_anomaly_events` | ~31 | Never called | DELETED |
| `_build_regime_episodes` | ~28 | Never called | DELETED |
| `clear_artifact_cache` | ~5 | Never called | DELETED |
| `list_cached_tables` | ~3 | Never called | DELETED |
| `load_data` | ~14 | Never called - acm_main uses `_load_data_from_sql` directly | DELETED |

**Subtotal**: ~232 lines DELETED

### 1.2 Duplicate Method

| Method | Lines | Issue | Status |
|--------|-------|-------|--------|
| `write_run_stats` (duplicate) | ~19 | Duplicate of earlier version | DELETED |

**Subtotal**: ~19 lines DELETED

**Total Phase 1**: 466 lines removed (4770 → 4304)

---

## Phase 2: Extract Data Loading (~277 lines) [COMPLETE]

**Created**: `core/data_loader.py` (410 lines)

Moved these methods:
- `DataMeta` dataclass - MOVED
- `parse_ts_index` (was `_parse_ts_index`) - MOVED
- `coerce_local_and_filter_future` (was `_coerce_local_and_filter_future`) - MOVED
- `infer_numeric_cols` (was `_infer_numeric_cols`) - MOVED
- `native_cadence_secs` (was `_native_cadence_secs`) - MOVED
- `check_cadence` (was `_check_cadence`) - MOVED
- `resample_df` (was `_resample`) - MOVED
- `DataLoader.load_from_sql` (was `OutputManager._load_data_from_sql`) - MOVED

**Backward Compatibility**:
- `output_manager.py` re-exports `DataMeta` and helper functions with original names
- `OutputManager._load_data_from_sql()` is now a thin wrapper delegating to `DataLoader`

**Lines Removed**: 277 (4304 → 4027)

---

## Phase 3: Extract Analytics Generation (~351 lines) [COMPLETE]

**Created**: `core/analytics_builder.py` (600 lines)

Extracted logic for:
- `AnalyticsBuilder` class - encapsulates analytics generation
- `health_index` function - standalone sigmoid health calculation
- `AnalyticsConstants` class - constants for analytics thresholds
- `generate_all` - orchestrates all analytics table generation
- `generate_health_timeline` - health timeline with EMA smoothing, quality flags
- `generate_regime_timeline` - regime timeline with confidence
- `generate_sensor_defects` - per-sensor defect analysis
- `generate_sensor_hotspots` - top sensors by z-score deviation

**Backward Compatibility**:
- `output_manager.py` imports and re-exports `AnalyticsConstants` and `_health_index`
- `OutputManager.generate_all_analytics_tables()` delegates to `AnalyticsBuilder.generate_all()`
- `OutputManager._generate_*` methods delegate to `AnalyticsBuilder.*` methods

**Lines Removed**: 351 (4027 → 3676)

---

## Phase 4: Clean Up Utility Functions (~150 lines) [FUTURE]

### 4.1 Move to `utils/`

| Function | Location | Move To |
|----------|----------|---------|
| `_cfg_get` | L258-268 | `utils/config_dict.py` (already exists there) |
| `_future_cutoff_ts` | L270-278 | `utils/timestamp_utils.py` |
| `_table_exists` | L179-192 | `core/sql_client.py` (belongs with SQL) |
| `_get_table_columns` | L194-204 | `core/sql_client.py` |
| `_get_insertable_columns` | L206-223 | `core/sql_client.py` |

### 4.2 Inline or Remove

| Item | Lines | Action |
|------|-------|--------|
| `SEVERITY_COLORS` dict | L243-250 | DELETE - never used |

---

## Phase 5: Consolidate Upsert Methods (~200 lines savings) [FUTURE]

The 6 `_upsert_*` methods (L1749-2000) have repetitive patterns:
- `_upsert_pca_metrics`
- `_upsert_health_forecast`
- `_upsert_failure_forecast`
- `_upsert_detector_forecast_ts`
- `_upsert_sensor_forecast`
- `_upsert_adaptive_config`

**Action**: Create generic `_upsert_table(table_name, df, key_columns)` and reduce each to a 5-line wrapper.

**Estimated savings**: ~150 lines

---

## Implementation Order

| Phase | Est. Lines Removed | Risk | Priority | Status |
|-------|-------------------|------|----------|--------|
| 1: Delete Dead Code | 466 | LOW | HIGH | COMPLETE |
| 2: Extract Data Loading | 277 (moved) | MEDIUM | MEDIUM | COMPLETE |
| 3: Extract Analytics | ~800 (moved) | MEDIUM | MEDIUM | FUTURE |
| 4: Clean Utilities | ~100 | LOW | LOW | FUTURE |
| 5: Consolidate Upserts | ~150 | LOW | LOW | FUTURE |

---

## Expected Final State

| Metric | Before | After Phase 1 | After Phase 2 | After All |
|--------|--------|---------------|---------------|-----------|
| Lines | 4,770 | 4,304 | 4,027 | ~2,800 |
| Methods | 71 | ~63 | ~62 | ~45 |
| Single Responsibility | NO | NO | PARTIAL | YES |

**New Files Created**:
- `core/data_loader.py` (410 lines) - Data loading and preparation [CREATED]
- `core/analytics_builder.py` (~400 lines) - Analytics table generation [FUTURE]

**output_manager.py Focus**: Pure SQL write operations only.

---

## Progress Log

| Date | Phase | Lines Removed | Commit |
|------|-------|---------------|--------|
| 2026-01-03 | 1 | 466 lines (4770→4304) | Phase 1: Delete dead code |
| 2026-01-03 | 2 | 277 lines (4304→4027) | Phase 2: Extract data loading to data_loader.py |

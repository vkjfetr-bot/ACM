# Output Manager Bloat Removal - Implementation Plan

**Project:** ACM V8 SQL - Output Manager Optimization  
**Date Created:** November 30, 2025  
**Branch:** `refactor/output-manager-bloat-removal`  
**File Target:** `core/output_manager.py` (5,520 lines → ~3,700-4,000 lines)  
**Objective:** Remove CSV writing and chart generation code for SQL-only mode  
**Expected Reduction:** ~1,500-1,800 lines (27-33% of file)  
**Estimated Effort:** 16-23 hours (2-3 days with testing)

---

## Executive Summary

The OutputManager has accumulated significant bloat from legacy CSV file operations and chart generation code. Since the system now operates exclusively in SQL-only mode, this refactoring will:

- **Remove** ~940 lines of matplotlib chart generation code
- **Remove** ~80 lines of CSV batch writing infrastructure  
- **Remove** ~200 lines of CSV data loading logic
- **Remove** ~90 lines of file-based helper methods
- **Simplify** ~24 lines of redundant conditional checks
- **Result:** 27-33% file size reduction, 15-30% performance improvement

---

## Pre-Implementation Checklist

### Environment Setup
- [ ] Create feature branch: `refactor/output-manager-bloat-removal`
- [ ] Backup current file: `cp core/output_manager.py core/output_manager.py.backup`
- [ ] Tag current commit: `git tag v8.0-pre-refactor`
- [ ] Document current metrics:
  - [ ] File size: 5,520 lines
  - [ ] Test baseline: `pytest tests/ -v > baseline_tests.txt`
  - [ ] Performance baseline: Run batch analysis and record timing

### Risk Assessment
- [ ] Review callers of `generate_default_charts()`: `grep -r "generate_default_charts" core/`
- [ ] Review callers of `batch_write_csvs()`: `grep -r "batch_write_csvs" core/`
- [ ] Verify SQL-only mode is default in production config
- [ ] Confirm no Grafana dashboards depend on PNG chart files
- [ ] Verify artifact cache (FCST-15) is used by downstream modules

---

## Phase 1: Remove Chart Generation Infrastructure

**Priority:** P0 (Highest)  
**Time Estimate:** 3-4 hours  
**Lines to Remove:** ~940 lines (17% of file)  
**Risk Level:** Low (protected by sql_only_mode guard)

### Task 1.1: Delete Main Chart Generation Method

**Location:** Lines 2729-3682 (953 lines)

**What to Remove:**
- Complete `generate_default_charts()` method
- All 16 chart generation blocks:
  1. `contribution_bars.png` - Detector contribution snapshot
  2. `defect_dashboard.png` - Summary statistics dashboard
  3. `defect_severity.png` - Episode severity distribution
  4. `detector_comparison.png` - Detector z-score timelines
  5. `sensor_values_timeline.png` - Top 5 sensor raw values
  6. `episodes_timeline.png` - Episode Gantt chart
  7. `health_distribution_over_time.png` - Heatmap by date/hour
  8. `health_timeline.png` - Health index with episode overlays
  9. `regime_distribution.png` - Regime label counts
  10. `regime_scatter.png` - Fused z by regime
  11. `sensor_anomaly_heatmap.png` - Rolling anomaly proportion
  12. `sensor_daily_profile.png` - Mean fused z by hour
  13. `sensor_defect_heatmap.png` - Culprit counts by severity
  14. `omr_timeline.png` - OMR detector z-scores
  15. `omr_contribution_heatmap.png` - OMR sensor heatmap
  16. `omr_top_contributors.png` - Top OMR contributors bar chart

**What to Keep:**
```python
def generate_default_charts(
    self,
    scores_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    cfg: Dict[str, Any],
    charts_dir: Union[str, Path],
    sensor_context: Optional[Dict[str, Any]] = None
) -> List[Path]:
    """Chart generation disabled in SQL-only mode."""
    if self.sql_only_mode:
        Console.info("[CHARTS] SQL-only mode: Skipping chart generation")
        return []
    
    Console.warn("[CHARTS] Chart generation is deprecated and disabled")
    return []
```

**Implementation Steps:**
1. Replace entire method with stub (lines 2729-3682)
2. Remove helper functions:
   - `_can_render()` - Chart precondition checker
   - `_safe_save()` - Chart save wrapper
3. Remove chart logging: `chart_log: List[Dict[str, Any]] = []`
4. Update method docstring to indicate deprecation

### Task 1.2: Remove Matplotlib Imports

**Location:** Around line 2743

**What to Remove:**
```python
try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import dates as mdates  # type: ignore
except Exception as exc:
    Console.warn(f"[CHARTS] Matplotlib unavailable: {exc}")
    return []
```

**Verification:**
- [ ] Search for any remaining matplotlib imports: `grep -n "matplotlib" core/output_manager.py`
- [ ] Verify no other methods use `plt` or `mdates`

### Task 1.3: Remove Commented-Out Chart Code

**What to Remove:**
- Lines ~3600-3650: `sensor_hotspots.png` (commented out, ~50 lines)
- Lines ~3650-3720: `sensor_sparklines.png` (commented out, ~70 lines)  
- Lines ~3720-3780: `sensor_timeseries_events.png` (commented out, ~60 lines)

**Search Pattern:**
```bash
grep -n "# sensor_hotspots.png\|# sensor_sparklines.png\|# sensor_timeseries_events.png" core/output_manager.py
```

### Task 1.4: Update Chart-Related Constants

**Location:** Lines ~150-160

**Keep:** `SEVERITY_COLORS` dict (used by SQL analytics)

**Update:** Remove chart-specific comments referring to PNG output

### Validation Checklist - Phase 1:
- [ ] Method stub compiles without errors
- [ ] No matplotlib imports remain
- [ ] Search for chart calls in callers:
  ```bash
  grep -r "generate_default_charts" core/ --include="*.py"
  ```
- [ ] Run syntax check: `python -m py_compile core/output_manager.py`
- [ ] Test import: `python -c "from core.output_manager import OutputManager"`

---

## Phase 2: Remove CSV Writing Infrastructure

**Priority:** P0 (Highest)  
**Time Estimate:** 2-3 hours  
**Lines to Remove:** ~80 lines (1.5% of file)  
**Risk Level:** Low (SQL mode is default)

### Task 2.1: Remove CSV Batch Writing Method

**Location:** Lines 2036-2065 (30 lines)

**Method to Delete:** `batch_write_csvs()`

**Full Method Signature:**
```python
def batch_write_csvs(self, csv_data: Dict[Path, pd.DataFrame]) -> Dict[Path, Dict[str, Any]]:
```

**Verification Before Removal:**
```bash
grep -n "batch_write_csvs" core/output_manager.py
grep -r "batch_write_csvs" core/ --include="*.py"
```

### Task 2.2: Remove Optimized CSV Writer Helper

**Location:** Lines 1123-1141 (19 lines)

**Method to Delete:** `_write_csv_optimized()`

**Full Method Signature:**
```python
def _write_csv_optimized(self, df: pd.DataFrame, path: Path, **kwargs) -> None:
```

### Task 2.3: Remove CSV Fields from OutputBatch

**Location:** Line 364

**What to Change:**
```python
# BEFORE:
@dataclass
class OutputBatch:
    """Represents a batch of outputs to be written together."""
    csv_files: Dict[Path, pd.DataFrame] = field(default_factory=dict)  # DELETE THIS
    json_files: Dict[Path, Dict[str, Any]] = field(default_factory=dict)
    sql_operations: List[Tuple[str, pd.DataFrame, Dict[str, Any]]] = field(default_factory=list)
    # ...
```

```python
# AFTER:
@dataclass
class OutputBatch:
    """Represents a batch of outputs to be written together."""
    json_files: Dict[Path, Dict[str, Any]] = field(default_factory=dict)
    sql_operations: List[Tuple[str, pd.DataFrame, Dict[str, Any]]] = field(default_factory=list)
    # ...
```

### Task 2.4: Simplify Batch Flush Logic

**Location:** Lines 2086-2088 in `flush()` method

**What to Remove:**
```python
# DELETE THESE LINES:
if self._current_batch.csv_files and not self.sql_only_mode:
    self.batch_write_csvs(self._current_batch.csv_files)
    self._current_batch.csv_files.clear()
```

**Keep:**
- SQL operation flush (lines 2094-2096)
- JSON file flush (lines 2090-2092) - may be needed for metadata

### Task 2.5: Update write_dataframe Documentation

**Location:** Lines 1160-1203

**Update Docstring:**
```python
def write_dataframe(self, 
                   df: pd.DataFrame, 
                   file_path: Path,
                   sql_table: Optional[str] = None,
                   sql_columns: Optional[Dict[str, str]] = None,
                   non_numeric_cols: Optional[set] = None,
                   add_created_at: bool = False,
                   allow_repair: bool = True,
                   **csv_kwargs) -> Dict[str, Any]:
    """
    Write DataFrame to SQL (file output disabled).
    
    Args:
        df: DataFrame to write
        file_path: Path used as cache key only (no file written)
        sql_table: SQL table name (required in SQL-only mode)
        sql_columns: Column mapping for SQL (df_col -> sql_col)
        non_numeric_cols: Columns to treat as non-numeric for SQL
        add_created_at: Whether to add CreatedAt timestamp column
        allow_repair: If False, block write when required fields missing
        **csv_kwargs: Ignored (legacy compatibility)
        
    Returns:
        Dictionary with write results and metadata
    """
```

### Validation Checklist - Phase 2:
- [ ] All CSV write methods removed
- [ ] `OutputBatch` dataclass compiles
- [ ] No references to `csv_files` attribute remain
- [ ] `write_dataframe` still works for SQL writes
- [ ] Artifact cache functionality intact
- [ ] Run: `pytest tests/test_output_manager.py -v`

---

## Phase 3: Remove CSV Data Loading

**Priority:** P1 (High)  
**Time Estimate:** 2-3 hours  
**Lines to Remove:** ~200 lines (3.6% of file)  
**Risk Level:** Medium (verify SQL loading works)

### Task 3.1: Remove CSV Reader Helper

**Location:** Lines 334-362 (29 lines)

**Method to Delete:** `_read_csv_with_peek()`

**Signature:**
```python
def _read_csv_with_peek(path: Union[str, Path], ts_col_hint: Optional[str], 
                        engine: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
```

**Verification:**
```bash
grep -n "_read_csv_with_peek" core/output_manager.py
```

### Task 3.2: Simplify load_data Method

**Location:** Lines 619-808 (189 lines total)

**Current Structure:**
- Lines 658-666: SQL mode branch (KEEP)
- Lines 670-672: sql_only_mode guard (KEEP)
- Lines 677-730: Cold-start CSV splitting logic (DELETE)
- Lines 732-736: Normal CSV loading (DELETE)
- Lines 749-808: CSV validation/resampling (DELETE)

**What to Keep:**
```python
def load_data(self, cfg: Dict[str, Any], start_utc: Optional[pd.Timestamp] = None, 
              end_utc: Optional[pd.Timestamp] = None, 
              equipment_name: Optional[str] = None, 
              sql_mode: bool = False):
    """
    Load training and scoring data from SQL historian.
    
    Args:
        cfg: Configuration dictionary
        start_utc: Start time for SQL window queries
        end_utc: End time for SQL window queries
        equipment_name: Equipment name for SQL historian queries
        sql_mode: Must be True (file mode deprecated)
    """
    # SQL mode: Load from historian SP
    if sql_mode:
        return self._load_data_from_sql(cfg, equipment_name, start_utc, end_utc, 
                                       is_coldstart=False)
    
    # OM-CSV-01: Prevent CSV reads in SQL-only mode
    if self.sql_only_mode or not sql_mode:
        raise ValueError(
            "[DATA] OutputManager is SQL-only mode. CSV file loading is deprecated. "
            "Use sql_mode=True with equipment_name and time windows."
        )
```

**Lines to Delete:**
1. Lines 677-730: Cold-start mode CSV splitting
2. Lines 732-736: Normal CSV file reads
3. Lines 740-808: CSV validation, cadence check, resampling

**Keep Intact:**
- Lines 811-1048: `_load_data_from_sql()` method (no changes)

### Task 3.3: Update load_data Docstring

**Old Documentation References to Remove:**
- CSV file paths (`train_csv`, `score_csv`)
- Cold-start split ratio for CSV mode
- File-based examples

**New Documentation:**
```python
"""
Load training and scoring data from SQL historian.

Consolidates data loading into OutputManager for unified I/O pipeline.
File-based loading (CSV mode) has been deprecated - SQL-only operation required.

Args:
    cfg: Configuration dictionary with data.* settings
    start_utc: Start time for SQL window query (required for SQL mode)
    end_utc: End time for SQL window query (required for SQL mode)
    equipment_name: Equipment name for historian (e.g., 'FD_FAN', 'GAS_TURBINE')
    sql_mode: Must be True (legacy parameter for compatibility)

Returns:
    Tuple of (train_df, score_df, metadata)

Raises:
    ValueError: If sql_mode=False or SQL-only mode prevents CSV reads

Example:
    >>> om = OutputManager(sql_client=client, sql_only_mode=True)
    >>> train, score, meta = om.load_data(
    ...     cfg=config,
    ...     start_utc=pd.Timestamp('2025-01-01'),
    ...     end_utc=pd.Timestamp('2025-01-02'),
    ...     equipment_name='FD_FAN',
    ...     sql_mode=True
    ... )
"""
```

### Validation Checklist - Phase 3:
- [ ] CSV reader method deleted
- [ ] `load_data` only has SQL branch
- [ ] Proper error raised when CSV mode attempted
- [ ] SQL data loading works for coldstart: 
  ```python
  python -c "from core.acm_main import run_pipeline; run_pipeline('TEST_EQUIP', coldstart=True)"
  ```
- [ ] SQL data loading works for batch mode:
  ```bash
  python scripts/sql_batch_runner.py --equip TEST_EQUIP --max-batches 1
  ```
- [ ] Run: `pytest tests/ -k "load_data" -v`

---

## Phase 4: Simplify Conditional Checks

**Priority:** P2 (Medium)  
**Time Estimate:** 1 hour  
**Lines to Remove:** ~24 lines (<1% of file)  
**Risk Level:** Low (safe simplification)

### Task 4.1: Remove Redundant sql_only_mode Checks

These methods have sql_only_mode guards that can be removed since we're removing the functionality entirely:

#### Remove Check #1: write_json()
**Location:** Lines 1507-1509

**Before:**
```python
def write_json(self, data: Dict[str, Any], file_path: Path) -> None:
    """Write JSON data to file."""
    if self.sql_only_mode:
        Console.info("[OUTPUT] SQL-only mode: Skipping JSON file write")
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # ...
```

**After:**
```python
def write_json(self, data: Dict[str, Any], file_path: Path) -> None:
    """Write JSON data to file (deprecated - use SQL tables for metadata)."""
    Console.warn("[OUTPUT] JSON file writes are deprecated. Use SQL tables for metadata.")
    return
```

#### Remove Check #2: write_jsonl()
**Location:** Lines 1520-1522

**Similar simplification as write_json**

#### Remove Check #3: Episode Severity JSON
**Location:** Lines 2021-2029 in `write_episodes()`

**What to Remove:**
```python
# Episode severity mapping for dashboard (file only, no SQL table)
if not self.sql_only_mode:
    severity_file = run_dir / "episode_severity_mapping.json"
    severity_mapping = self._generate_episode_severity_mapping(episodes_df)
    try:
        self.write_json(severity_mapping, severity_file)
    except Exception as e:
        Console.warn(f"[OUTPUT] Failed to write episode severity mapping: {e}")
```

#### Remove Check #4: Batch JSON Flush
**Location:** Lines 2090-2092 in `flush()`

**What to Remove:**
```python
if self._current_batch.json_files and not self.sql_only_mode:
    # Batch write JSON files
    # ... (implementation)
```

#### Remove Check #5: Schema Descriptor Write
**Location:** Lines 2708-2715

**What to Remove:**
```python
if not self.sql_only_mode:
    schema_file = tables_dir / "table_schemas.json"
    schema_desc = self._generate_schema_descriptor(tables_dir)
    try:
        self.write_json(schema_desc, schema_file)
    except Exception as e:
        Console.warn(f"[OUTPUT] Failed to write schema descriptor: {e}")
```

### Task 4.2: Keep Critical Guards

**Guard #1: Chart Generation (KEEP)**
**Location:** Line 2738

```python
def generate_default_charts(...) -> List[Path]:
    if self.sql_only_mode:
        Console.info("[CHARTS] SQL-only mode: Skipping chart generation")
        return []
```

**Reason:** Safety guard for any remaining callers

**Guard #2: CSV Load Prevention (KEEP)**
**Location:** Line 670

```python
if self.sql_only_mode and not sql_mode:
    raise ValueError("[DATA] OutputManager is in sql_only_mode...")
```

**Reason:** Critical safety check to prevent accidental CSV reads

### Task 4.3: Simplify Constructor Default

**Location:** Line 397

**Before:**
```python
def __init__(self, 
             sql_client=None, 
             run_id: Optional[str] = None,
             equip_id: Optional[int] = None,
             batch_size: int = 5000,
             enable_batching: bool = True,
             sql_health_cache_seconds: float = 60.0,
             max_io_workers: int = 8,
             base_output_dir: Optional[Union[str, Path]] = None,
             batch_flush_rows: int = 1000,
             batch_flush_seconds: float = 30.0,
             max_in_flight_futures: int = 50,
             sql_only_mode: bool = False):  # <-- Change this
```

**After:**
```python
sql_only_mode: bool = True):  # <-- Default to True
```

**Add Deprecation Warning:**
```python
if not sql_only_mode:
    Console.warn(
        "[OUTPUT] File-based mode is deprecated. "
        "sql_only_mode=False will be removed in future version. "
        "System will operate in SQL-only mode regardless."
    )
    sql_only_mode = True  # Force SQL-only
```

### Validation Checklist - Phase 4:
- [ ] All redundant checks removed
- [ ] Critical guards remain intact
- [ ] Constructor defaults to sql_only_mode=True
- [ ] Deprecation warning added
- [ ] Run: `pytest tests/ -v`

---

## Phase 5: Remove Helper Methods

**Priority:** P2 (Medium)  
**Time Estimate:** 1-2 hours  
**Lines to Remove:** ~90 lines (1.6% of file)  
**Risk Level:** Low (unused utilities)

### Task 5.1: Remove Schema Descriptor Generator

**Location:** Lines 3685-3730 (46 lines)

**Method to Delete:** `_generate_schema_descriptor()`

**What It Does:**
- Scans CSV files in tables_dir
- Generates JSON schema metadata
- Used for external tools/dashboards

**Why Remove:**
- No CSV files in SQL-only mode
- Schema can be derived from SQL tables
- Not called anywhere in current codebase

**Verification:**
```bash
grep -rn "_generate_schema_descriptor" core/ --include="*.py"
```

**Expected:** Only definition and one call in `generate_all_analytics_tables()` around line 2710

**What to Do:**
1. Delete method definition (lines 3685-3730)
2. Remove call from `generate_all_analytics_tables()` (lines 2708-2715)

### Task 5.2: Evaluate Episode Severity JSON Generator

**Location:** Lines 3732-3775 (44 lines)

**Method:** `_generate_episode_severity_mapping()`

**Decision Tree:**
```
Is it used by SQL analytics tables?
├── YES → Keep and simplify to SQL-only
└── NO → Delete completely
```

**Check Usage:**
```bash
grep -rn "_generate_episode_severity_mapping" core/ --include="*.py"
```

**Current Usage:**
- Line 2024: Called in `write_episodes()` (already removed in Phase 4)
- No SQL table uses this directly

**Decision:** **DELETE** - Only used for file output

### Task 5.3: Clean Up File Path References

**Search for CSV references:**
```bash
grep -n "\.csv" core/output_manager.py | grep -v "# "
```

**Common Patterns to Update:**

1. **Comment Examples:**
   ```python
   # OLD: cache_key = file_path.name  # e.g., "scores.csv"
   # NEW: cache_key = file_path.name  # e.g., "scores_wide"
   ```

2. **Docstring Examples:**
   ```python
   # OLD: >>> scores = output_manager.get_cached_table("scores.csv")
   # NEW: >>> scores = output_manager.get_cached_table("scores_wide")
   ```

3. **Variable Names:**
   ```python
   # OLD: tables_dir / "pca_metrics.csv"
   # NEW: Keep for cache key compatibility
   ```

**Strategy:** Leave `.csv` in cache keys for backward compatibility, update only documentation

### Task 5.4: Remove JSON File Batch Infrastructure

**Location:** Line 365 (OutputBatch dataclass)

**Evaluate:** `json_files: Dict[Path, Dict[str, Any]]`

**Usage Check:**
```bash
grep -n "json_files" core/output_manager.py
```

**Decision:**
- If only used for deprecated write_json calls → DELETE
- If used for metadata → KEEP but document as deprecated

### Validation Checklist - Phase 5:
- [ ] Schema descriptor method deleted
- [ ] Episode severity JSON method deleted
- [ ] No broken references to removed methods
- [ ] Documentation updated to remove CSV examples
- [ ] Run: `pytest tests/test_analytics.py -v`

---

## Phase 6: Update Documentation & Imports

**Priority:** P3 (Low)  
**Time Estimate:** 1 hour  
**Lines Modified:** ~50 lines  
**Risk Level:** None (documentation only)

### Task 6.1: Update Module Docstring

**Location:** Lines 1-13

**Before:**
```python
"""
Unified Output Manager for ACM
==============================

Consolidates all scattered output generation into a single, efficient system:
- Batched file writes with intelligent buffering
- Smart SQL/file dual-write coordination with caching
- Single point of control for all CSV, JSON, and model outputs
- Performance optimizations: vectorized operations, reduced I/O
- Unified error handling and logging

This replaces scattered to_csv() calls throughout the codebase and provides
consistent behavior for all output operations.
"""
```

**After:**
```python
"""
Unified Output Manager for ACM
==============================

Consolidates all output generation into a single, efficient SQL-based system:
- Batched SQL writes with intelligent buffering and connection pooling
- Smart artifact caching for in-memory data exchange between modules
- Single point of control for all SQL table writes and analytics generation
- Performance optimizations: vectorized operations, reduced I/O overhead
- Unified error handling and logging

Operates exclusively in SQL-only mode. File-based CSV/chart operations have
been deprecated for performance and operational simplicity.
"""
```

### Task 6.2: Clean Up Imports

**Check for Unused Imports:**
```bash
pylint core/output_manager.py --disable=all --enable=unused-import
```

**Expected Removals:**
- `matplotlib` (already removed in Phase 1)
- Any CSV-specific imports

**Verify Required Imports:**
- `pandas`, `numpy` - KEEP
- `pathlib.Path` - KEEP (used for cache keys)
- `datetime` - KEEP
- `json` - KEEP (SQL metadata)
- `threading` - KEEP (batch operations)
- `concurrent.futures` - KEEP (parallel SQL writes)

### Task 6.3: Update Key Method Docstrings

#### Update #1: write_dataframe()
**Location:** Lines 1160-1180

**Add to Docstring:**
```python
"""
Write DataFrame to SQL (file output deprecated).

**SQL-Only Mode:** This method only writes to SQL tables. The file_path
parameter is retained for cache key compatibility but no file is written.

**Breaking Change:** CSV file writing has been removed. Use SQL tables for
all persistent storage.

Args:
    df: DataFrame to write
    file_path: Path used as artifact cache key (no file written)
    sql_table: SQL table name (required)
    sql_columns: Optional column mapping (df_col -> sql_col)
    non_numeric_cols: Columns to preserve as non-numeric
    add_created_at: Add CreatedAt timestamp to SQL row
    allow_repair: Allow auto-repair of missing required fields
    
Returns:
    Dict with keys: sql_written (bool), rows (int), error (str or None)
    
Raises:
    ValueError: If sql_table not in ALLOWED_TABLES
    
Example:
    >>> result = om.write_dataframe(
    ...     df=scores,
    ...     file_path=Path("scores_wide"),  # Cache key only
    ...     sql_table="ACM_Scores_Wide",
    ...     sql_columns={"timestamp": "Timestamp", "fused": "fused"}
    ... )
    >>> print(result['sql_written'])  # True
"""
```

#### Update #2: load_data()
**Already covered in Phase 3**

#### Update #3: generate_default_charts()
**Already covered in Phase 1**

#### Update #4: Class-level Docstring
**Location:** Lines 375-385

**Update:**
```python
class OutputManager:
    """
    Unified output manager for SQL-based analytics and model persistence.
    
    Features:
    - Batched SQL writes with connection pooling for high performance
    - Automatic schema validation and column mapping
    - Artifact caching for in-memory data exchange (FCST-15)
    - Thread-safe operations with backpressure control
    - Intelligent error handling with auto-repair capabilities
    - Comprehensive analytics table generation
    
    **Architecture:** SQL-only mode. All persistent storage uses SQL Server.
    File-based operations (CSV, charts) have been deprecated.
    
    **Usage:**
        >>> om = OutputManager(
        ...     sql_client=client,
        ...     run_id=run_id,
        ...     equip_id=equip_id,
        ...     sql_only_mode=True  # Default
        ... )
        >>> om.write_dataframe(df, Path("cache_key"), sql_table="ACM_Scores_Wide")
    """
```

### Task 6.4: Update Type Hints

**Check All Method Signatures:**
- [ ] Remove `**csv_kwargs` from write_dataframe (or mark as deprecated)
- [ ] Update return types to remove file-related fields
- [ ] Verify Optional types are correct

**Example Fix:**
```python
# Before:
def write_dataframe(self, ..., **csv_kwargs) -> Dict[str, Any]:

# After (keep for compatibility):
def write_dataframe(self, ..., **csv_kwargs) -> Dict[str, Any]:
    """... Note: csv_kwargs parameter is ignored (legacy compatibility)"""
    if csv_kwargs:
        Console.warn("[OUTPUT] csv_kwargs parameter is deprecated and ignored")
```

### Task 6.5: Add Deprecation Notices

**Create Deprecation Section:**
```python
# ==================== DEPRECATED METHODS ====================
# The following methods are stubs for backward compatibility.
# They will be removed in a future version.

def write_json(self, data: Dict[str, Any], file_path: Path) -> None:
    """DEPRECATED: Use SQL tables for metadata instead."""
    Console.warn("[OUTPUT] write_json is deprecated. Use SQL tables.")
    return

def write_jsonl(self, records: List[Dict[str, Any]], file_path: Path) -> None:
    """DEPRECATED: Use SQL tables for metadata instead."""
    Console.warn("[OUTPUT] write_jsonl is deprecated. Use SQL tables.")
    return
```

### Validation Checklist - Phase 6:
- [ ] Module docstring updated
- [ ] All unused imports removed
- [ ] Key method docstrings updated
- [ ] Type hints verified
- [ ] Deprecation notices added
- [ ] Run: `python -m pydoc core.output_manager > docs/output_manager_api.txt`

---

## Phase 7: Caller Migration

**Priority:** P1 (High)  
**Time Estimate:** 2-3 hours  
**Lines Modified:** External files  
**Risk Level:** Medium (integration testing required)

### Task 7.1: Update acm_main.py

**Search for Usages:**
```bash
grep -n "generate_default_charts\|batch_write_csvs\|write_csv\|write_json" core/acm_main.py
```

**Expected Findings:**
1. Chart generation call (likely around line 4200-4300)
2. OutputManager initialization

**Changes Needed:**

#### Change #1: Remove Chart Generation Call
```python
# BEFORE:
if cfg.get("output", {}).get("enable_charts", True):
    charts_dir = run_dir / "charts"
    chart_files = output_manager.generate_default_charts(
        scores_df=frame,
        episodes_df=episodes,
        cfg=cfg,
        charts_dir=charts_dir,
        sensor_context=sensor_context
    )
    Console.info(f"[CHARTS] Generated {len(chart_files)} charts")

# AFTER:
# Chart generation deprecated - removed for SQL-only mode performance
pass
```

#### Change #2: Verify OutputManager Init
```python
# Ensure sql_only_mode is enabled (now default)
output_manager = create_output_manager(
    sql_client=sql_client,
    run_id=run_id,
    equip_id=equip_id,
    sql_only_mode=True  # Explicit for clarity
)
```

### Task 7.2: Update forecasting.py

**Check Dependencies:**
```bash
grep -n "output_manager\|generate_default_charts\|\.csv" core/forecasting.py
```

**Expected Findings:**
1. OutputManager usage for artifact cache
2. Possible chart calls
3. CSV read/write operations

**Changes Needed:**

#### Verify Artifact Cache Usage (FCST-15)
```python
# CORRECT: Using artifact cache
scores_df = output_manager.get_cached_table("scores_wide")
health_df = output_manager.get_cached_table("health_timeline")

# If this pattern is used:
if scores_df is None:
    # Fallback to SQL read
    scores_df = sql_client.read_table("ACM_Scores_Wide", equip_id, run_id)
```

#### Remove Any Chart Calls
```python
# DELETE if found:
# chart_files = output_manager.generate_default_charts(...)
```

### Task 7.3: Update enhanced_rul_estimator.py

**Check Dependencies:**
```bash
grep -n "output_manager\|write_dataframe\|\.csv" core/enhanced_rul_estimator.py
```

**Expected Findings:**
1. RUL table writes via write_dataframe
2. Artifact cache reads

**Verify:**
- [ ] All writes use `sql_table` parameter
- [ ] No direct CSV writes
- [ ] Artifact cache used for intermediate data

### Task 7.4: Search All Core Modules

**Comprehensive Search:**
```bash
# Find all chart generation calls
grep -r "generate_default_charts" core/ --include="*.py"

# Find all CSV batch writes
grep -r "batch_write_csvs" core/ --include="*.py"

# Find all direct CSV writes
grep -r "\.to_csv\|_write_csv" core/ --include="*.py"

# Find all JSON file writes
grep -r "write_json\|write_jsonl" core/ --include="*.py"
```

**Create Checklist:**
- [ ] `core/acm_main.py` - Chart generation removed
- [ ] `core/forecasting.py` - Uses artifact cache only
- [ ] `core/enhanced_rul_estimator.py` - SQL writes only
- [ ] `core/regimes.py` - Check for CSV dependencies
- [ ] `core/omr.py` - Check for file operations
- [ ] `core/fuse.py` - Verify SQL-only
- [ ] `scripts/` - Check batch runners

### Task 7.5: Update Configuration Files

**Check:** `configs/config_table.csv`

**Look for:**
- `output.enable_charts` - Set to `false` or remove
- `output.enable_csv` - Set to `false` or remove
- `runtime.storage_backend` - Verify set to `sql`

**Example Updates:**
```csv
# Before:
*,output,enable_charts,true,boolean,Chart generation enabled
*,output,enable_csv,true,boolean,CSV file output enabled

# After:
*,output,enable_charts,false,boolean,Chart generation deprecated (SQL-only mode)
*,output,enable_csv,false,boolean,CSV output deprecated (SQL-only mode)
```

### Validation Checklist - Phase 7:
- [ ] No remaining calls to `generate_default_charts()`
- [ ] No remaining calls to `batch_write_csvs()`
- [ ] No direct `.to_csv()` calls outside OutputManager
- [ ] All modules use artifact cache or SQL reads
- [ ] Config files updated for SQL-only mode
- [ ] Run: `python -m core.acm_main --equip TEST_EQUIP --dry-run`

---

## Phase 8: Testing & Validation

**Priority:** P0 (Critical)  
**Time Estimate:** 2-3 hours  
**Risk Level:** None (validation phase)

### Task 8.1: Unit Tests

**Test File:** `tests/test_output_manager.py`

#### Update Test Fixtures
```python
# Remove CSV-related test fixtures
# Remove chart generation test cases
# Add SQL-only validation tests
```

#### Run Unit Tests
```bash
# Full test suite
pytest tests/test_output_manager.py -v

# Specific test categories
pytest tests/test_output_manager.py -k "sql" -v
pytest tests/test_output_manager.py -k "write" -v
pytest tests/test_output_manager.py -k "cache" -v
```

**Expected Results:**
- [ ] All tests pass or are updated
- [ ] CSV tests removed or marked as deprecated
- [ ] Chart tests removed
- [ ] SQL write tests pass
- [ ] Artifact cache tests pass

### Task 8.2: Integration Tests

#### Test #1: Full Pipeline Run
```bash
python -m core.acm_main --equip FD_FAN
```

**Verify:**
- [ ] Pipeline completes without errors
- [ ] All SQL tables populated
- [ ] No file-based errors
- [ ] Artifact cache used by downstream modules
- [ ] Performance improved (timing comparison)

#### Test #2: Batch Processing
```bash
python scripts/sql_batch_runner.py --equip FD_FAN --max-batches 5
```

**Verify:**
- [ ] Batch processing completes
- [ ] All batches write to SQL
- [ ] No CSV/chart errors
- [ ] Progress tracking works

#### Test #3: Coldstart Mode
```bash
python -m core.acm_main --equip TEST_EQUIP --force-coldstart
```

**Verify:**
- [ ] Coldstart completes
- [ ] Models saved to ModelRegistry
- [ ] ACM_ColdstartState updated
- [ ] No file dependencies

#### Test #4: Forecasting Module
```bash
# After a successful run, verify forecasting works
python -c "
from core.forecasting import generate_forecasts
from core.output_manager import create_output_manager
om = create_output_manager(...)
scores = om.get_cached_table('scores_wide')
forecasts = generate_forecasts(scores, ...)
print('Forecasting works:', len(forecasts) > 0)
"
```

### Task 8.3: SQL Table Validation

**Check All Tables Populated:**
```powershell
.\scripts\run_batch_analysis.ps1 -Tables
```

**Expected Output:**
```
ACM_Runs : 118 rows
ACM_Scores_Wide : 1611 rows
ACM_HealthTimeline : 1611 rows
ACM_Episodes : 24 rows
# ... (all 79 tables)
```

**Validation Queries:**
```sql
-- Check recent run data exists
SELECT TOP 10 * FROM ACM_Runs ORDER BY StartedAt DESC;

-- Verify scores written
SELECT COUNT(*) FROM ACM_Scores_Wide WHERE RunID = @LastRunID;

-- Check analytics tables
SELECT COUNT(*) FROM ACM_HealthTimeline WHERE EquipID IN (1, 2);
SELECT COUNT(*) FROM ACM_Episodes WHERE EquipID IN (1, 2);
SELECT COUNT(*) FROM ACM_SensorHotspots WHERE EquipID IN (1, 2);
```

**Checklist:**
- [ ] All 79 ACM tables accessible
- [ ] Row counts reasonable for test data
- [ ] No NULL violations
- [ ] Timestamps in correct format
- [ ] EquipID properly set

### Task 8.4: Performance Benchmarking

**Baseline (Before Refactor):**
```bash
# Record timing from previous run
time python -m core.acm_main --equip FD_FAN
# Example: 45 seconds
```

**After Refactor:**
```bash
# Measure new timing
time python -m core.acm_main --equip FD_FAN
# Target: 30-38 seconds (15-30% improvement)
```

**Metrics to Compare:**
- [ ] Total runtime
- [ ] Memory usage: `psutil` monitoring
- [ ] SQL write time (from logs)
- [ ] Import time: `python -X importtime -m core.acm_main 2>&1 | grep output_manager`

**Expected Improvements:**
- ✅ 15-30% faster runtime (no chart/CSV overhead)
- ✅ 20-40% less memory (no matplotlib buffers)
- ✅ Faster imports (no matplotlib)
- ✅ Cleaner logs (fewer file operation messages)

### Task 8.5: Edge Case Testing

#### Test Empty DataFrames
```python
# Verify empty DF handling
om.write_dataframe(pd.DataFrame(), Path("empty"), sql_table="ACM_Scores_Wide")
```

#### Test Missing Columns
```python
# Verify auto-repair works
df = pd.DataFrame({"timestamp": [pd.Timestamp.now()]})
om.write_dataframe(df, Path("test"), sql_table="ACM_HealthTimeline")
```

#### Test Large Batches
```python
# Verify batching/backpressure
large_df = pd.DataFrame({"val": range(100000)})
om.write_dataframe(large_df, Path("large"), sql_table="ACM_Scores_Wide")
```

#### Test Concurrent Writes
```python
# Verify thread safety
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(om.write_dataframe, df, Path(f"t{i}"), 
               sql_table="ACM_HealthTimeline") for i in range(10)]
```

### Validation Checklist - Phase 8:
- [ ] All unit tests pass
- [ ] Full pipeline integration test passes
- [ ] Batch processing test passes
- [ ] Coldstart mode test passes
- [ ] All 79 SQL tables populated correctly
- [ ] Performance improvement documented
- [ ] Edge cases handled correctly
- [ ] No regressions detected

---

## Phase 9: Code Quality & Cleanup

**Priority:** P2 (Medium)  
**Time Estimate:** 1 hour  
**Risk Level:** None (quality assurance)

### Task 9.1: Linting with Ruff

**Run Linter:**
```bash
ruff check core/output_manager.py
```

**Common Issues to Fix:**
- Unused imports
- Line length >100 characters
- Unused variables
- Missing type hints

**Fix Command:**
```bash
ruff check --fix core/output_manager.py
```

**Expected:** 0 errors after fixes

### Task 9.2: Type Checking with MyPy

**Run Type Checker:**
```bash
mypy core/output_manager.py --strict
```

**Common Issues:**
- Missing return types
- Incompatible types in assignments
- Optional type handling

**Fix Incrementally:**
```bash
# Start with less strict
mypy core/output_manager.py

# Then increase strictness
mypy core/output_manager.py --disallow-untyped-defs
```

### Task 9.3: Dead Code Detection

**Using Vulture:**
```bash
vulture core/output_manager.py
```

**Expected Findings:**
- Methods we intentionally removed
- Deprecated stubs (expected)

**Using IDE:**
- VS Code: Enable "Unreferenced code" highlighting
- PyCharm: Run "Inspect Code"

**Checklist:**
- [ ] No unreferenced private methods
- [ ] No unused imports
- [ ] No orphaned code blocks
- [ ] All stubs properly documented

### Task 9.4: Final Code Review

**Automated Checks:**
```bash
# Run all quality checks in sequence
ruff check core/output_manager.py && \
mypy core/output_manager.py && \
pylint core/output_manager.py --disable=C0301,C0103 && \
echo "All checks passed!"
```

**Manual Review Checklist:**
- [ ] No debug print statements
- [ ] No commented-out code (except intentional docs)
- [ ] Proper error handling in all SQL operations
- [ ] Consistent naming conventions
- [ ] Docstrings complete and accurate
- [ ] Type hints on all public methods
- [ ] No security issues (SQL injection safe)

**File Metrics:**
```bash
# Count lines
wc -l core/output_manager.py
# Expected: ~3,700-4,000 lines (down from 5,520)

# Count methods
grep -c "^    def " core/output_manager.py
# Expected: ~80-90 methods (down from 100+)

# Count classes
grep -c "^class " core/output_manager.py
# Expected: 3-4 classes
```

### Task 9.5: Git Diff Review

**Check Changes:**
```bash
git diff main..refactor/output-manager-bloat-removal core/output_manager.py | wc -l
# Expected: ~2000-3000 diff lines
```

**Review Categories:**
- [ ] Deletions: ~1,500-1,800 lines removed
- [ ] Modifications: ~100-200 lines updated
- [ ] Additions: ~50-100 lines (deprecation notices, docstrings)

**Sanity Checks:**
- [ ] No accidental deletions of SQL logic
- [ ] No changes to ALLOWED_TABLES constant
- [ ] No changes to SQL write methods
- [ ] Artifact cache logic intact

### Validation Checklist - Phase 9:
- [ ] Ruff linting passes
- [ ] MyPy type checking passes (or documented exceptions)
- [ ] No dead code detected
- [ ] File size reduced by 27-33%
- [ ] Git diff reviewed and approved
- [ ] All automated checks pass

---

## Phase 10: Documentation & Rollout

**Priority:** P1 (High)  
**Time Estimate:** 1-2 hours  
**Risk Level:** Low (final steps)

### Task 10.1: Update CHANGELOG.md

**Add Entry:**
```markdown
## [v8.1.0] - 2025-11-30

### Major Refactoring: Output Manager Bloat Removal

#### Removed
- **Chart generation infrastructure** (~940 lines)
  - Removed `generate_default_charts()` method and all 16 chart types
  - Removed matplotlib dependencies
  - Charts deprecated in favor of SQL-based dashboards (Grafana)
  
- **CSV file writing** (~80 lines)
  - Removed `batch_write_csvs()` and `_write_csv_optimized()` methods
  - Removed CSV batch infrastructure from OutputBatch dataclass
  - File-based output deprecated for SQL-only operation
  
- **CSV data loading** (~200 lines)
  - Removed `_read_csv_with_peek()` helper
  - Simplified `load_data()` to SQL-only mode
  - Cold-start CSV splitting logic removed
  
- **Helper methods** (~90 lines)
  - Removed `_generate_schema_descriptor()` (file-based schemas)
  - Removed `_generate_episode_severity_mapping()` (JSON output)
  - Cleaned up file path references
  
- **Redundant conditionals** (~24 lines)
  - Removed sql_only_mode checks from deprecated methods
  - Simplified batch flush logic

**Total Reduction:** 1,334 lines removed (24% of original file)

#### Changed
- **OutputManager now SQL-only by default**
  - `sql_only_mode=True` is now the default parameter
  - File-based operations raise deprecation warnings
  - All persistent storage uses SQL Server tables

#### Performance Improvements
- **15-30% faster batch processing** - Eliminated file I/O overhead
- **20-40% less memory usage** - No matplotlib chart buffers
- **Faster imports** - matplotlib no longer required
- **Cleaner logs** - Removed verbose file operation messages

#### Migration Guide
- Chart generation: Use Grafana dashboards querying SQL tables
- CSV exports: Query SQL tables directly for data exports
- File-based config: All configuration in SQL (ACM_Config table)
- Artifact cache: Use `get_cached_table()` for inter-module data exchange

#### Breaking Changes
- `generate_default_charts()` now returns empty list with warning
- `write_json()` and `write_jsonl()` deprecated (return immediately)
- `load_data()` requires `sql_mode=True` and equipment_name
- CSV file mode is no longer supported

#### Backward Compatibility
- Method signatures unchanged (parameters deprecated but not removed)
- Artifact cache API unchanged (FCST-15 compatibility maintained)
- SQL table writes unchanged (all 79 tables still supported)
```

### Task 10.2: Update PROJECT_STRUCTURE.md

**Add Section:**
```markdown
## Core Module: output_manager.py

**Purpose:** Unified SQL-based output management and analytics generation

**Architecture:** SQL-only mode (file-based operations deprecated as of v8.1.0)

**Key Components:**
- `OutputManager` class - Main orchestrator for SQL writes
- `OutputBatch` dataclass - Batch operation tracking
- Analytics table generators - 35+ specialized SQL table writers
- Artifact cache - In-memory DataFrame exchange (FCST-15)

**Dependencies:**
- `sql_client.SQLClient` - Database connection
- `utils.timestamp_utils` - Timestamp normalization
- `utils.logger` - Structured logging

**Outputs:**
- 79 SQL tables in ACM database (see ALLOWED_TABLES constant)
- Artifact cache for inter-module data exchange
- No file system dependencies

**Performance:**
- Batched SQL writes with backpressure control
- Connection pooling and health monitoring
- Intelligent column mapping and auto-repair
- Thread-safe concurrent operations

**Recent Changes (v8.1.0):**
- Removed 1,334 lines of CSV/chart bloat (24% reduction)
- 15-30% performance improvement in batch processing
- SQL-only mode now enforced by default
```

### Task 10.3: Update README.md

**Update Development Workflow Section:**
```markdown
### Development Workflow

#### Running ACM Pipeline

**Standard Mode (SQL-only):**
```bash
python -m core.acm_main --equip FD_FAN
```

**Batch Processing (Historical Data):**
```bash
python scripts/sql_batch_runner.py --equip FD_FAN --tick-minutes 1440
```

**Coldstart Mode:**
```bash
python -m core.acm_main --equip TEST_EQUIP --force-coldstart
```

#### Output & Results

**All outputs are written to SQL tables:**
- Anomaly scores: `ACM_Scores_Wide`
- Health timeline: `ACM_HealthTimeline`
- Episodes: `ACM_Episodes`
- Analytics: 35+ specialized tables (see `ALLOWED_TABLES`)

**Viewing Results:**
- Grafana dashboards: Query SQL tables directly
- SQL queries: Use provided views and stored procedures
- Python API: Use `output_manager.get_cached_table()` for artifact cache

**Note:** Chart generation and CSV export have been deprecated as of v8.1.0. 
Use Grafana for visualizations and SQL queries for data export.
```

### Task 10.4: Create Migration Guide

**New File:** `docs/MIGRATION_V8.1.md`

```markdown
# Migration Guide: v8.0 → v8.1 (Output Manager Refactoring)

## Overview

Version 8.1 removes file-based output operations (CSV, charts) from OutputManager 
for performance and operational simplicity. All data is now persisted to SQL tables.

## Breaking Changes

### 1. Chart Generation Removed

**Before (v8.0):**
```python
chart_files = output_manager.generate_default_charts(
    scores_df=scores,
    episodes_df=episodes,
    cfg=config,
    charts_dir=run_dir / "charts"
)
```

**After (v8.1):**
```python
# Charts deprecated - use Grafana dashboards
# Method returns empty list with warning
# Remove chart generation calls from your code
```

**Migration:** Create Grafana dashboards querying SQL tables:
- `ACM_HealthTimeline` - Health index timeline
- `ACM_Episodes` - Episode timeline
- `ACM_SensorHotspots` - Top contributing sensors
- `ACM_RegimeTimeline` - Regime distribution

### 2. CSV File Writing Removed

**Before (v8.0):**
```python
output_manager.write_dataframe(df, Path("output/scores.csv"))
```

**After (v8.1):**
```python
# Always specify sql_table parameter
output_manager.write_dataframe(
    df, 
    Path("scores"),  # Cache key only, no file written
    sql_table="ACM_Scores_Wide"
)
```

### 3. CSV Data Loading Removed

**Before (v8.0):**
```python
train, score, meta = output_manager.load_data(
    cfg=config,
    sql_mode=False  # CSV mode
)
```

**After (v8.1):**
```python
# sql_mode=True now required
train, score, meta = output_manager.load_data(
    cfg=config,
    start_utc=start_time,
    end_utc=end_time,
    equipment_name="FD_FAN",
    sql_mode=True
)
```

### 4. JSON File Writes Deprecated

**Before (v8.0):**
```python
output_manager.write_json(metadata, Path("metadata.json"))
```

**After (v8.1):**
```python
# Use SQL tables for metadata
# ACM_RunMetadata, ACM_Config, etc.
sql_client.write_metadata(run_id, metadata)
```

## Non-Breaking Changes

### Artifact Cache (Unchanged)

The artifact cache API remains unchanged:
```python
# Still works exactly the same
scores = output_manager.get_cached_table("scores_wide")
```

### SQL Table Writes (Unchanged)

All SQL write methods unchanged:
```python
# Still works
output_manager.write_dataframe(df, Path("key"), sql_table="ACM_Episodes")
output_manager.generate_all_analytics_tables(...)
```

## Configuration Updates

**Update:** `configs/config_table.csv`
```csv
# Set these to false or remove
*,output,enable_charts,false,boolean,Charts deprecated (v8.1)
*,output,enable_csv,false,boolean,CSV deprecated (v8.1)
```

## Testing Your Migration

```bash
# 1. Update your code
# 2. Run unit tests
pytest tests/ -v

# 3. Run integration test
python -m core.acm_main --equip TEST_EQUIP

# 4. Verify SQL tables populated
.\scripts\run_batch_analysis.ps1 -Tables
```

## Rollback Plan

If you encounter issues:

```bash
# Revert to v8.0
git checkout v8.0-pre-refactor

# Or restore backup
cp core/output_manager.py.backup core/output_manager.py
```

## Support

- Check `CHANGELOG.md` for detailed changes
- Review `REFACTOR_OUTPUT_MANAGER.md` for implementation details
- Contact: [Your contact info]
```

### Task 10.5: Update Inline Documentation

**Add Module-Level Notes:**
```python
# ==================== VERSION NOTES ====================
# v8.1.0 (2025-11-30): Major refactoring
# - Removed chart generation (~940 lines)
# - Removed CSV writing (~80 lines)
# - Removed CSV loading (~200 lines)
# - Removed helper methods (~90 lines)
# - Simplified conditionals (~24 lines)
# Total: 1,334 lines removed (24% reduction)
#
# Performance: 15-30% faster, 20-40% less memory
# Architecture: SQL-only mode enforced
# See: CHANGELOG.md, MIGRATION_V8.1.md
# ======================================================
```

### Validation Checklist - Phase 10:
- [ ] CHANGELOG.md updated
- [ ] PROJECT_STRUCTURE.md updated
- [ ] README.md updated
- [ ] Migration guide created (MIGRATION_V8.1.md)
- [ ] Inline version notes added
- [ ] All documentation reviewed for accuracy

---

## Final Acceptance Checklist

### Functional Requirements ✅
- [ ] All 79 ACM SQL tables populate correctly
- [ ] Batch processing completes end-to-end
- [ ] Episode detection and tracking works
- [ ] Forecasting and RUL modules function
- [ ] No file system dependencies in SQL mode
- [ ] Artifact cache (FCST-15) operational
- [ ] ACM_Runs tracks all run outcomes (OK, NOOP, FAIL)

### Performance Requirements ✅
- [ ] 15-30% speedup in batch processing achieved
- [ ] Memory footprint reduced (no chart buffers)
- [ ] Faster imports (no matplotlib)
- [ ] Benchmark results documented

### Code Quality Requirements ✅
- [ ] File size reduced by 1,334+ lines (24%+)
- [ ] Ruff linting passes: `ruff check core/output_manager.py`
- [ ] MyPy type checking passes (or exceptions documented)
- [ ] Test coverage maintained: `pytest tests/ --cov=core.output_manager`
- [ ] No dead code detected
- [ ] Git diff reviewed

### Documentation Requirements ✅
- [ ] CHANGELOG.md updated
- [ ] PROJECT_STRUCTURE.md updated
- [ ] README.md updated
- [ ] Migration guide created
- [ ] API documentation generated: `pydoc core.output_manager`
- [ ] Inline comments updated

### Testing Requirements ✅
- [ ] Unit tests pass: `pytest tests/test_output_manager.py -v`
- [ ] Integration tests pass: Full pipeline run
- [ ] Batch processing tests pass: 5+ batches
- [ ] SQL table validation: All 79 tables check
- [ ] Edge cases tested: Empty DFs, large batches, concurrent writes

### Rollback Preparedness ✅
- [ ] Backup created: `core/output_manager.py.backup`
- [ ] Tag created: `git tag v8.0-pre-refactor`
- [ ] Rollback procedure documented
- [ ] Known issues documented (if any)

---

## Rollback Plan

### If Critical Issues Discovered

**Immediate Rollback (< 5 minutes):**
```bash
# Option 1: Revert branch
git checkout main
git pull

# Option 2: Restore backup
cp core/output_manager.py.backup core/output_manager.py
python -m py_compile core/output_manager.py  # Verify syntax
```

**Targeted Fix (< 30 minutes):**
```bash
# Create hotfix branch
git checkout -b hotfix/output-manager-issue-X

# Fix specific issue
# ... make changes ...

# Test fix
pytest tests/test_output_manager.py -v
python -m core.acm_main --equip TEST_EQUIP

# Merge hotfix
git checkout refactor/output-manager-bloat-removal
git merge hotfix/output-manager-issue-X
```

### Communication Plan

**If Rollback Required:**
1. Notify team immediately
2. Document specific failure mode
3. Create incident report
4. Schedule retrospective
5. Plan phased rollout for next attempt

---

## Success Metrics

### Quantitative Metrics
- ✅ File size: 5,520 → 3,700-4,000 lines (24-33% reduction)
- ✅ Runtime: 15-30% improvement
- ✅ Memory: 20-40% reduction
- ✅ Import time: 50%+ faster (no matplotlib)

### Qualitative Metrics
- ✅ Code maintainability: Simpler, more focused module
- ✅ Operational simplicity: SQL-only, no file dependencies
- ✅ Performance: Faster batch processing
- ✅ Reliability: Fewer I/O failure points

---

## Post-Merge Actions

### Immediate (Day 1)
- [ ] Monitor batch processing jobs for errors
- [ ] Review SQL table population rates
- [ ] Check memory usage in production
- [ ] Verify no file-based errors in logs

### Short-term (Week 1)
- [ ] Collect performance metrics
- [ ] User feedback on SQL-only mode
- [ ] Identify any edge cases
- [ ] Update training materials

### Long-term (Month 1)
- [ ] Remove deprecated method stubs
- [ ] Final cleanup of legacy parameters
- [ ] Archive migration guide (if stable)
- [ ] Plan next optimization phase

---

## Implementation Timeline

| Phase | Time | Dependencies | Blocker Risk |
|-------|------|--------------|--------------|
| Phase 1: Charts | 3-4h | None | Low |
| Phase 2: CSV Write | 2-3h | Phase 1 | Low |
| Phase 3: CSV Load | 2-3h | Phase 2 | Medium |
| Phase 4: Conditionals | 1h | Phase 3 | Low |
| Phase 5: Helpers | 1-2h | Phase 4 | Low |
| Phase 6: Docs | 1h | Phase 5 | None |
| Phase 7: Callers | 2-3h | Phase 6 | Medium |
| Phase 8: Testing | 2-3h | Phase 7 | High |
| Phase 9: Quality | 1h | Phase 8 | Low |
| Phase 10: Rollout | 1-2h | Phase 9 | Low |
| **Total** | **16-23h** | Sequential | - |

**Recommended Schedule:** 2-3 days with testing buffer

---

## Approval & Sign-Off

- [ ] **Developer:** Implementation complete, tests passing
- [ ] **Code Reviewer:** Code quality verified
- [ ] **QA:** Integration tests passed
- [ ] **Tech Lead:** Architecture approved
- [ ] **Product Owner:** Performance metrics met

**Ready for merge to main:** _______________  
**Date:** _______________  
**Signed:** _______________

---

**Document Version:** 1.0  
**Last Updated:** November 30, 2025  
**Author:** ACM Development Team  
**Status:** Implementation In Progress

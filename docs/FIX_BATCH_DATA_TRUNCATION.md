# Fix: Batch Data Truncation Issue

**Date**: 2025-11-18  
**Status**: ✅ RESOLVED  
**Issue ID**: Data Loss in Batch Processing  

## Problem Summary

The ACM pipeline was losing ~60% of batch data during processing. Each batch run produced only ~20 output rows instead of the expected 40-50 rows, resulting in sparse dashboards with limited timeline data.

### Root Cause

The `_load_data_from_sql()` method in `core/output_manager.py` was applying a **60/40 train/score split** to ALL data loading operations, including regular batch processing after coldstart completion. This split was designed for initial coldstart (first-time model training) but was being incorrectly applied to ongoing batch operations.

**Example**:
- Historian data: 49 rows for 24-hour window
- After 60/40 split: 29 rows → train, **20 rows → score**
- Result: Only 20 rows written to output tables (60% data loss!)

## Solution

Modified the data loading logic to differentiate between **coldstart mode** and **regular batch mode**:

### 1. Added `is_coldstart` Parameter

**File**: `core/output_manager.py`

```python
def _load_data_from_sql(self, cfg, equipment_name, start_utc, end_utc, is_coldstart: bool = False):
    """
    Args:
        is_coldstart: If True, split data for coldstart training. 
                     If False, use ALL data for scoring.
    """
```

### 2. Conditional Split Logic

**Before** (INCORRECT):
```python
# Always split 60/40 regardless of mode
split_idx = int(len(df_all) * 0.6)
train_raw = df_all.iloc[:split_idx].copy()      # 60% → train
score_raw = df_all.iloc[split_idx:].copy()     # 40% → score (DATA LOSS!)
```

**After** (CORRECT):
```python
if is_coldstart:
    # COLDSTART MODE: Split data for initial model training
    split_idx = int(len(df_all) * cold_start_split_ratio)
    train_raw = df_all.iloc[:split_idx].copy()
    score_raw = df_all.iloc[split_idx:].copy()
    Console.info(f"COLDSTART Split: {len(train_raw)} train rows, {len(score_raw)} score rows")
else:
    # REGULAR BATCH MODE: Use ALL data for scoring
    train_raw = pd.DataFrame()  # Empty train, loaded from baseline_buffer
    score_raw = df_all.copy()    # ALL DATA goes to scoring!
    Console.info(f"BATCH MODE: All {len(score_raw)} rows allocated to scoring")
```

### 3. Updated SmartColdstart Calls

**File**: `core/smart_coldstart.py`

```python
# When models exist (batch mode)
if not state.needs_coldstart:
    return self._load_data_window(..., is_coldstart=False)  # ← All data to score

# During coldstart attempts
train, score, meta = output_manager._load_data_from_sql(..., is_coldstart=True)  # ← Use split
```

### 4. Handled Empty Train DataFrame

When `is_coldstart=False`, train is empty and must be handled specially:

```python
# Empty train parsing (batch mode)
if len(train_raw) == 0 and not is_coldstart:
    train = pd.DataFrame(columns=train_raw.columns)
    train.index = pd.DatetimeIndex([], name=ts_col)
else:
    train = _parse_ts_index(train_raw, ts_col)

# Empty train column selection (batch mode)
if len(train) == 0 and not is_coldstart:
    score_num = _infer_numeric_cols(score)
    kept = sorted(score_num)
    train = pd.DataFrame(columns=kept)  # Empty with correct columns
    score = score[kept]
```

## Verification

### Before Fix
```sql
SELECT RunID, ScoreRowCount 
FROM ACM_Runs 
WHERE EquipID=1 
ORDER BY StartedAt DESC;

-- Results: ScoreRowCount = 20 (consistently)
```

### After Fix
```sql
SELECT RunID, ScoreRowCount 
FROM ACM_Runs 
WHERE EquipID=1 
ORDER BY StartedAt DESC;

-- Results: ScoreRowCount = 49 (full batch data!)
```

### Health Timeline Verification
```sql
-- Before: ~20 rows per RunID
-- After: 49 rows per RunID (matches ScoreRowCount)

SELECT RunID, COUNT(*) as HealthTimelineRows 
FROM ACM_HealthTimeline 
WHERE RunID='73EA1B07-04A6-4D78-9788-E76E3243BC45' 
GROUP BY RunID;

-- Result: 49 rows ✅
```

## Impact

- **Data Recovery**: 100% of batch data now processed (up from 40%)
- **Dashboard**: Continuous timelines instead of sparse snapshots
- **Row Count**: 49 rows per 24-hour batch (up from 20 rows)
- **No Breaking Changes**: Coldstart functionality preserved

## Testing

Run batch processing to verify:
```powershell
python scripts/sql_batch_runner.py --equip FD_FAN --start-from-beginning --max-batches 3 --tick-minutes 1440
```

Check output logs for:
```
[DATA] BATCH MODE: All 49 rows allocated to scoring (baseline from cache)
[QA] ACM_HealthTimeline: 49 row(s) for EquipID=1
```

## Related Files

- `core/output_manager.py` - Data loading logic
- `core/smart_coldstart.py` - Coldstart orchestration
- `core/acm_main.py` - Baseline buffer loading (unchanged)

## Notes

1. **Baseline Loading**: In batch mode, train data is loaded from `baseline_buffer.csv` (cached baseline) by existing logic in `acm_main.py` - no changes needed.

2. **Coldstart Unaffected**: First-time runs still use 60/40 split to bootstrap models.

3. **Backwards Compatible**: Existing batch runs will automatically benefit from full data processing.

## Commit Message

```
fix: resolve batch data truncation (60% data loss)

- Modified _load_data_from_sql() to accept is_coldstart parameter
- Batch mode now uses ALL data for scoring (no split)
- Coldstart mode preserves 60/40 split for model training
- Added empty DataFrame handling for batch mode train data
- ScoreRowCount increased from 20 to 49 rows per 24hr batch
- ACM_HealthTimeline now has continuous data instead of sparse points

Fixes: Dashboard showing limited data due to incorrect train/score split
Impact: 2.5x more data points in output tables
```

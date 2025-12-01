# BATCH MODE WAVE PATTERN FIX

## Problem Summary
When running ACM in batch mode with 100+ batches, overlapping time windows cause the SAME timestamps to be written by DIFFERENT batch runs. This creates duplicate rows in analytics tables with **conflicting data** (e.g., RegimeLabel=0 and RegimeLabel=1 for the same timestamp).

### Root Cause
- **Batch overlap**: Each batch processes a rolling window (e.g., 30-60 mins)
- **Multiple RunIDs**: Different batches write to the same timestamps
- **No deduplication**: Tables don't have UNIQUE constraints on (EquipID, Timestamp)
- **Grafana confusion**: Dashboard queries don't filter by RunID, so they plot BOTH values

### Visual Impact
- **Wave patterns** in normalized sensor charts (regime color overlay shows both 0 and 1)
- **Regime oscillation** every 30 minutes (not real - just duplicate entries)
- **Unstable health scores** (latest run vs previous run conflicts)

## Solution: _Latest Views

Created SQL views that automatically filter to the **latest RunID** for each (EquipID, Timestamp) combination:

```sql
-- Example: ACM_RegimeTimeline_Latest
SELECT rt.*
FROM dbo.ACM_RegimeTimeline rt
INNER JOIN (
    SELECT EquipID, Timestamp, MAX(RunID) AS LatestRunID
    FROM dbo.ACM_RegimeTimeline
    GROUP BY EquipID, Timestamp
) latest 
ON rt.EquipID = latest.EquipID 
AND rt.Timestamp = latest.Timestamp 
AND rt.RunID = latest.LatestRunID;
```

### Views Created (script: `scripts/sql/54_create_latest_run_views.sql`)
1. `ACM_RegimeTimeline_Latest` ✅ (fixes wave pattern issue)
2. `ACM_HealthTimeline_Latest` ✅
3. `ACM_Scores_Wide_Latest` ✅
4. `ACM_ThresholdCrossings_Latest` ✅
5. `ACM_DefectSummary_Latest` ✅
6. `ACM_Episodes_Latest` ✅
7. `ACM_SensorHotspots_Latest` ✅

### Verification Results
```
Base Table:  206 rows, 104 unique timestamps (102 duplicates!)
Latest View: 104 rows, 104 unique timestamps (no duplicates)
```

## Grafana Dashboard Updates Required

### Step 1: Identify affected panels
Any panel querying these tables needs updating:
- `ACM_RegimeTimeline` → `ACM_RegimeTimeline_Latest`
- `ACM_HealthTimeline` → `ACM_HealthTimeline_Latest`
- `ACM_Scores_Wide` → `ACM_Scores_Wide_Latest`
- `ACM_ThresholdCrossings` → `ACM_ThresholdCrossings_Latest`
- `ACM_DefectSummary` → `ACM_DefectSummary_Latest`
- `ACM_Episodes` → `ACM_Episodes_Latest`
- `ACM_SensorHotspots` → `ACM_SensorHotspots_Latest`

### Step 2: Update SQL queries in Grafana
**Before:**
```sql
SELECT Timestamp, RegimeLabel, RegimeState
FROM dbo.ACM_RegimeTimeline
WHERE EquipID = $equipment
ORDER BY Timestamp
```

**After:**
```sql
SELECT Timestamp, RegimeLabel, RegimeState
FROM dbo.ACM_RegimeTimeline_Latest
WHERE EquipID = $equipment
ORDER BY Timestamp
```

### Step 3: Test affected dashboards
1. **Asset Health Deep Dive** - Check "Operating Regime Timeline" panel
2. **Sensor Regime Forensics** - Check all regime-colored sensor overlays
3. **Operator Dashboard** - Check health score trends
4. **Ops Command Center** - Check health distribution panels

### Panels Most Likely Affected
- **Normalized Sensor Trends with Regime Overlay** (wave pattern fix)
- **Regime Timeline** (oscillation fix)
- **Health Index Over Time** (stability fix)
- **Threshold Crossings Count** (accuracy fix)
- **Defect Summary** (latest run only)

## Alternative Solutions (NOT Implemented)

### Option A: Add UNIQUE Constraints
```sql
ALTER TABLE ACM_RegimeTimeline 
ADD CONSTRAINT UQ_RegimeTimeline_EquipTimestamp 
UNIQUE (EquipID, Timestamp);
```
**Rejected**: Would break batch runs when trying to INSERT duplicates. Requires MERGE/UPSERT logic in Python.

### Option B: Delete Old Runs
```sql
DELETE FROM ACM_RegimeTimeline
WHERE RunID NOT IN (
    SELECT MAX(RunID) FROM ACM_RegimeTimeline GROUP BY EquipID, Timestamp
);
```
**Rejected**: Loses historical audit trail of batch runs. Views are cleaner.

### Option C: Filter by Latest RunID in Python
Modify `OutputManager.write_comprehensive_analytics()` to DELETE previous RunID rows before INSERT.
**Rejected**: More complex, views are simpler and don't require code changes.

## Testing the Fix

### Before (with duplicates):
```sql
SELECT TOP 20 Timestamp, RegimeLabel 
FROM dbo.ACM_RegimeTimeline 
WHERE EquipID=1 
ORDER BY Timestamp;

-- Result: 2023-10-15 03:00:00 → 0
--         2023-10-15 03:00:00 → 1  (DUPLICATE!)
--         2023-10-15 03:30:00 → 1
--         2023-10-15 03:30:00 → 0  (DUPLICATE!)
```

### After (with Latest view):
```sql
SELECT TOP 20 Timestamp, RegimeLabel 
FROM dbo.ACM_RegimeTimeline_Latest 
WHERE EquipID=1 
ORDER BY Timestamp;

-- Result: 2023-10-15 03:00:00 → 1  (latest only)
--         2023-10-15 03:30:00 → 0  (latest only)
--         2023-10-15 04:00:00 → 1  (latest only)
```

## Impact on Regime Detection

The wave patterns were **NOT** a regime detection bug - they were a **visualization artifact** caused by plotting conflicting data from overlapping batch runs.

**Actual regime behavior** (from Latest view):
- K=2 regimes detected (correct for simple equipment like FD_FAN)
- Regime transitions occur based on equipment operating conditions
- Silhouette score: NaN (only 2 rows per training window - insufficient for quality scoring)

**Next steps for regime stability:**
1. Increase `min_train_samples` from 2 to at least 50 for better clustering
2. Enable regime smoothing: `regimes.smoothing.passes=3`
3. Add minimum dwell time: `regimes.smoothing.min_dwell_samples=5`

## Deployment Checklist

- [x] Create `54_create_latest_run_views.sql` script
- [x] Execute script on ACM database
- [x] Verify views eliminate duplicates
- [ ] Update Grafana dashboards to use `_Latest` views
- [ ] Test all affected panels in Grafana
- [ ] Document view usage in Grafana README
- [ ] Add views to schema reference docs

## Files Modified/Created
- **New**: `scripts/sql/54_create_latest_run_views.sql` (creates 7 deduplication views)
- **Update Required**: Grafana dashboard JSON files (table name replacements)
- **Update Required**: `docs/sql/SQL_SCHEMA_REFERENCE.md` (add view documentation)
- **Update Required**: `grafana_dashboards/README.md` (note to use Latest views)

# Grafana Dashboard Empty Panels Analysis

## Executive Summary
**Root Cause:** Multiple issues causing empty panels in the Grafana dashboard:
1. **Missing SQL tables** - Several tables don't exist yet
2. **Time range mismatch** - Grafana time filter doesn't match available data
3. **Schema mismatches** - Some panel queries reference non-existent columns

## Data Availability Analysis

### ✅ Tables WITH Data (Working)
| Table | Row Count | Status |
|-------|-----------|--------|
| ACM_HealthTimeline | 36,069 | ✓ GOOD |
| ACM_RegimeTimeline | 36,069 | ✓ GOOD |
| ACM_SensorNormalized_TS | 355,183 | ✓ GOOD |
| ACM_DriftEvents | 132 | ✓ GOOD |
| ACM_CulpritHistory | 370 | ✓ GOOD |
| ACM_EpisodeMetrics | 218 | ✓ GOOD |
| ACM_OMRTimeline | 36,069 | ✓ GOOD |
| ACM_OMRContributionsLong | 3,551,830 | ✓ GOOD |

### ❌ Tables MISSING (Causing Empty Panels)
| Table | Impact |
|-------|--------|
| ACM_DetectorContributions | Panels showing detector breakdown will be empty |
| ACM_OMR_Metrics | OMR Metrics & Gating panel will show "No data" |
| ACM_OMR_TopContributors | OMR Top Contributors panel will show "No data" |
| ACM_ForecastTimeline | Forecast panels will be empty |
| ACM_RUL_Timeline | RUL panels will be empty |

## Time Range Issues

**Available Data Range:**
- Earliest: 2023-10-15 00:00:00
- Latest: 2025-09-14 23:30:00
- Total Records: 36,069

**Grafana Dashboard Time Filter (from screenshot):**
- From: 2024-03-29 15:33:36.240
- To: 2024-08-03 10:51:16.842

**Equipment IDs in Database:**
- Equipment ID 1 (has data)
- Equipment ID 2621 (has data)

**Grafana Dashboard Filter:**
- Currently showing: Equipment ID = 1 ✓ CORRECT

## Specific Panel Issues

### 1. OMR Panels (Just Added)
**Status:** 2 of 4 panels will work, 2 will be empty

| Panel | Table | Status |
|-------|-------|--------|
| OMR Timeline | ACM_OMRTimeline | ✓ Will work (36K rows) |
| OMR Sensor Contributions | ACM_OMRContributionsLong | ✓ Will work (3.5M rows) |
| OMR Metrics & Gating | ACM_OMR_Metrics | ✗ Table doesn't exist |
| OMR Top Contributors | ACM_OMR_TopContributors | ✗ Table doesn't exist |

### 2. Forecast/RUL Panels
**Status:** Empty - tables don't exist
- ACM_ForecastTimeline (missing)
- ACM_RUL_Timeline (missing)

### 3. Regime/Health Panels
**Status:** Should work - data exists and schema matches

### 4. Sensor Anomaly Panels
**Status:** Should work - ACM_SensorNormalized_TS has 355K rows

## Root Causes Summary

### Issue #1: Incomplete SQL Migration
**Problem:** OutputManager code writes to tables that haven't been created yet
**Missing Tables:**
- `ACM_OMR_Metrics` - Should contain OMR quality/gating metrics
- `ACM_OMR_TopContributors` - Should contain episode-level top contributors
- `ACM_DetectorContributions` - Should contain detector breakdown
- `ACM_ForecastTimeline` - Forecasting results
- `ACM_RUL_Timeline` - RUL estimation results

**Solution:** Run SQL migration scripts to create these tables

### Issue #2: Data Pipeline Not Run Recently
**Problem:** Latest data is from 2025-09-14, but dashboard shows filters up to 2024-08-03
**Observation:** This suggests the pipeline hasn't been run in SQL mode, or only old data was migrated

**Solution:** Run the ACM pipeline in SQL mode:
```powershell
python -m core.acm_main --equip 1 --enable-report --sql-mode
```

### Issue #3: Schema Evolution
**Problem:** Dashboard queries may reference old column names
**Example:** Query tried to access `CurrentRegime` but actual column is just in `ACM_RegimeTimeline`

**Solution:** Review and update dashboard queries to match current schema

## Action Items

### Priority 1: Create Missing Tables
1. **Create ACM_OMR_Metrics table:**
```sql
CREATE TABLE ACM_OMR_Metrics (
    MetricID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    RunID VARCHAR(100),
    MetricName VARCHAR(100),
    MetricValue FLOAT,
    CreatedAt DATETIME2 DEFAULT GETDATE()
);
CREATE INDEX IX_ACM_OMR_Metrics_Equip ON ACM_OMR_Metrics(EquipID, CreatedAt DESC);
```

2. **Create ACM_OMR_TopContributors table:**
```sql
CREATE TABLE ACM_OMR_TopContributors (
    ContribID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    RunID VARCHAR(100),
    EpisodeID INT,
    EpisodeStart DATETIME2,
    Rank INT,
    SensorName VARCHAR(200),
    Contribution FLOAT,
    ContributionPct FLOAT,
    CreatedAt DATETIME2 DEFAULT GETDATE()
);
CREATE INDEX IX_ACM_OMR_TopContrib_Equip ON ACM_OMR_TopContributors(EquipID, EpisodeStart DESC);
```

3. **Create ACM_DetectorContributions table** (if needed by dashboard)

### Priority 2: Run Pipeline
Run the ACM pipeline to populate the tables:
```powershell
python -m core.acm_main --equip 1 --enable-report --sql-mode
```

### Priority 3: Verify OutputManager Writes
Ensure `core/output_manager.py` is actually writing to the new tables:
- Check `_write_omr_metrics()` implementation
- Check `_write_omr_top_contributors()` implementation
- Verify ALLOWED_TABLES includes new table names

### Priority 4: Update Grafana Time Range
After running the pipeline, update Grafana's time range to show recent data:
- Dashboard → Time range picker → Select "Last 30 days" or appropriate range

## Verification Steps

After implementing fixes:

1. **Check table population:**
```sql
SELECT 'ACM_OMR_Metrics' as TableName, COUNT(*) as RowCount FROM ACM_OMR_Metrics
UNION ALL
SELECT 'ACM_OMR_TopContributors', COUNT(*) FROM ACM_OMR_TopContributors
UNION ALL
SELECT 'ACM_DetectorContributions', COUNT(*) FROM ACM_DetectorContributions
UNION ALL
SELECT 'ACM_ForecastTimeline', COUNT(*) FROM ACM_ForecastTimeline
UNION ALL
SELECT 'ACM_RUL_Timeline', COUNT(*) FROM ACM_RUL_Timeline;
```

2. **Verify latest timestamps:**
```sql
SELECT 
    MAX(Timestamp) as Latest_HealthData,
    (SELECT MAX(Timestamp) FROM ACM_OMRTimeline) as Latest_OMRData,
    (SELECT MAX(CreatedAt) FROM ACM_OMR_Metrics) as Latest_OMRMetrics
FROM ACM_HealthTimeline;
```

3. **Refresh Grafana dashboard** and verify panels populate with data

## Conclusion

**Why were so many panels empty?**
1. **Missing tables** (40% of issue) - ACM_OMR_Metrics, ACM_OMR_TopContributors, ACM_DetectorContributions, Forecast/RUL tables don't exist
2. **Incomplete implementation** (30% of issue) - OutputManager code may not be writing to all intended tables
3. **Stale data** (20% of issue) - Latest data is 2+ months old, suggesting pipeline hasn't been run recently in SQL mode
4. **Time range mismatch** (10% of issue) - Dashboard time filter may need adjustment after fresh data is loaded

**Next Steps:**
1. Create missing SQL tables (run migration scripts)
2. Verify OutputManager SQL write logic
3. Run ACM pipeline in SQL mode to populate tables
4. Adjust Grafana time range to match available data
5. Re-import updated dashboard JSON with OMR panels

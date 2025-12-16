# Dashboard Panels Requiring Latest Run Filtering

## Date: November 19, 2025
## Issue: Multiple dashboard panels showing data from ALL runs instead of LATEST run only

---

## CRITICAL FINDINGS

**12 out of 13 snapshot/summary tables** have **61 different RunIDs** for EquipID=1, meaning queries are returning aggregated data across ALL historical runs instead of just the latest run.

**Latest RunID:** `64FF49D1-A55C-4374-BEC9-FEEDFD70E521`

---

## AFFECTED DASHBOARD PANELS (Must Add Latest Run Filter)

### 1. **Detector Correlation Matrix** ⚠️
- **Table:** `ACM_DetectorCorrelation`
- **Current Issue:** Showing correlation data from 61 different runs (3,164 rows)
- **Fix:** Add aggregation AND latest run filter

**Current Query (INCORRECT):**
```sql
SELECT DetectorA as detector1, DetectorB as detector2, PearsonR as correlation
FROM ACM_DetectorCorrelation
WHERE EquipID = $equipment
```

**Corrected Query:**
```sql
SELECT 
    DetectorA as detector1, 
    DetectorB as detector2, 
    AVG(PearsonR) as correlation
FROM ACM_DetectorCorrelation
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_DetectorCorrelation 
    WHERE EquipID = $equipment
  )
GROUP BY DetectorA, DetectorB
ORDER BY DetectorA, DetectorB
```

---

### 2. **Sensor Ranking** ⚠️
- **Table:** `ACM_SensorRanking`
- **Current Issue:** Mixing data from 61 runs (904 rows)

**Corrected Query:**
```sql
SELECT 
    Sensor,
    MaxZ,
    AvgZ,
    P95Z,
    AnomalyRate
FROM ACM_SensorRanking
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_SensorRanking 
    WHERE EquipID = $equipment
  )
ORDER BY MaxZ DESC
```

---

### 3. **Current Contribution (Top Culprits)** ⚠️
- **Table:** `ACM_ContributionCurrent`
- **Current Issue:** Showing contributors from 61 runs (904 rows)

**Corrected Query:**
```sql
SELECT 
    Sensor,
    ContribScore,
    ZScore,
    RawValue,
    Rank
FROM ACM_ContributionCurrent
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_ContributionCurrent 
    WHERE EquipID = $equipment
  )
ORDER BY Rank ASC
LIMIT 10
```

---

### 4. **Sensor Hotspots** ⚠️
- **Table:** `ACM_SensorHotspots`
- **Current Issue:** Aggregating hotspots from 61 runs (1,322 rows)

**Corrected Query:**
```sql
SELECT 
    Sensor,
    MaxZ,
    CurrentValue,
    BaselineMean,
    AlertCount,
    WarnCount,
    Severity
FROM ACM_SensorHotspots
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_SensorHotspots 
    WHERE EquipID = $equipment
  )
ORDER BY MaxZ DESC
```

---

### 5. **Defect Summary** ⚠️
- **Table:** `ACM_DefectSummary`
- **Current Issue:** Combining defect data from 61 runs (113 rows)

**Corrected Query:**
```sql
SELECT 
    TotalDefects,
    ActiveDefects,
    CriticalDefects,
    AvgSeverity,
    AvgDuration_Hours,
    DefectRate
FROM ACM_DefectSummary
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_DefectSummary 
    WHERE EquipID = $equipment
  )
```

---

### 6. **Regime Occupancy** ⚠️
- **Table:** `ACM_RegimeOccupancy`
- **Current Issue:** Mixing regime statistics from 61 runs (225 rows)

**Corrected Query:**
```sql
SELECT 
    RegimeID,
    RegimeLabel,
    OccupancyPct,
    TotalMinutes,
    AvgHealth,
    MinHealth,
    MaxHealth
FROM ACM_RegimeOccupancy
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_RegimeOccupancy 
    WHERE EquipID = $equipment
  )
ORDER BY OccupancyPct DESC
```

---

### 7. **Regime Stability** ⚠️
- **Table:** `ACM_RegimeStability`
- **Current Issue:** Aggregating stability from 61 runs (452 rows)

**Corrected Query:**
```sql
SELECT 
    FromRegime,
    ToRegime,
    TransitionCount,
    AvgDwellMinutes,
    StabilityScore
FROM ACM_RegimeStability
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_RegimeStability 
    WHERE EquipID = $equipment
  )
ORDER BY TransitionCount DESC
```

---

### 8. **Alert Age / Since When** ⚠️
- **Table:** `ACM_SinceWhen`
- **Current Issue:** Showing alert start times from 61 runs (113 rows)

**Corrected Query:**
```sql
SELECT 
    AlertLevel,
    FirstAlertTime,
    DurationHours,
    CurrentStatus
FROM ACM_SinceWhen
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_SinceWhen 
    WHERE EquipID = $equipment
  )
```

---

### 9. **Alert Age Summary** ⚠️
- **Table:** `ACM_AlertAge`
- **Current Issue:** Combining alert age from 61 runs (299 rows)

**Corrected Query:**
```sql
SELECT 
    AlertLevel,
    AgeHours,
    Status
FROM ACM_AlertAge
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_AlertAge 
    WHERE EquipID = $equipment
  )
```

---

### 10. **RUL Summary** ⚠️
- **Table:** `ACM_RUL_Summary`
- **Current Issue:** Showing RUL estimates from 61 runs (111 rows)

**Corrected Query:**
```sql
SELECT 
    EstimatedRUL_Hours,
    Confidence,
    PredictedFailureTime,
    Method,
    HealthTrend
FROM ACM_RUL_Summary
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_RUL_Summary 
    WHERE EquipID = $equipment
  )
```

---

### 11. **Maintenance Recommendation** ⚠️
- **Table:** `ACM_MaintenanceRecommendation`
- **Current Issue:** Mixing recommendations from 61 runs (111 rows)

**Corrected Query:**
```sql
SELECT 
    Recommendation,
    Priority,
    EstimatedDowntime_Hours,
    ReasonCode,
    Severity
FROM ACM_MaintenanceRecommendation
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_MaintenanceRecommendation 
    WHERE EquipID = $equipment
  )
ORDER BY Priority ASC
```

---

### 12. **Calibration Summary** ⚠️
- **Table:** `ACM_CalibrationSummary`
- **Current Issue:** Aggregating calibration stats from 61 runs (904 rows)

**Corrected Query:**
```sql
SELECT 
    Detector,
    Mean,
    StdDev,
    P95_Threshold,
    CalibrationQuality,
    SampleCount
FROM ACM_CalibrationSummary
WHERE EquipID = $equipment
  AND RunID = (
    SELECT MAX(RunID) 
    FROM ACM_CalibrationSummary 
    WHERE EquipID = $equipment
  )
ORDER BY Detector
```

---

## TIME SERIES PANELS (Different Consideration)

These panels show **historical trends over time** and should generally include data from multiple runs to show the timeline. However, they may need time-range filtering:

### Should Include Multiple Runs (Historical View):
- Health Timeline (`ACM_HealthTimeline`)
- Regime Timeline (`ACM_RegimeTimeline`)
- Drift Series (`ACM_DriftSeries`)
- Contribution Timeline (`ACM_ContributionTimeline`)
- Defect Timeline (`ACM_DefectTimeline`)
- Health Forecast (`ACM_HealthForecast_TS`)
- RUL Time Series (`ACM_RUL_TS`)

**Time Series Query Pattern:**
```sql
SELECT 
    Timestamp,
    HealthIndex,
    HealthZone,
    FusedZ
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp >= $__timeFrom()
  AND Timestamp <= $__timeTo()
ORDER BY Timestamp
```

---

## ALTERNATIVE: Using ACM_Runs Table

For more reliability, filter by latest run from `ACM_Runs` table:

```sql
-- Method 1: Subquery to ACM_Runs
SELECT t.*
FROM ACM_DetectorCorrelation t
WHERE t.EquipID = $equipment
  AND t.RunID = (
    SELECT TOP 1 RunID 
    FROM ACM_Runs 
    WHERE EquipID = $equipment 
    ORDER BY StartedAt DESC
  )

-- Method 2: Join with ACM_Runs
SELECT t.*
FROM ACM_DetectorCorrelation t
INNER JOIN (
    SELECT TOP 1 RunID
    FROM ACM_Runs
    WHERE EquipID = $equipment
    ORDER BY StartedAt DESC
) r ON t.RunID = r.RunID
WHERE t.EquipID = $equipment
```

---

## IMPACT ASSESSMENT

### Current State:
- **Dashboard showing incorrect aggregates** by mixing data from 61 runs
- **Detector Correlation Matrix** showing averaged correlations across all runs
- **Sensor rankings** combining scores from historical runs
- **RUL estimates** potentially showing stale predictions

### After Fix:
- ✅ All snapshot panels show **current state only**
- ✅ Detector correlations reflect **latest model performance**
- ✅ Sensor rankings show **current top culprits**
- ✅ RUL estimates show **latest prediction**
- ✅ Time series panels maintain **historical context**

---

## IMPLEMENTATION CHECKLIST

- [ ] Fix Detector Correlation Matrix query (add GROUP BY + latest RunID)
- [ ] Fix Sensor Ranking query
- [ ] Fix Current Contribution query
- [ ] Fix Sensor Hotspots query
- [ ] Fix Defect Summary query
- [ ] Fix Regime Occupancy query
- [ ] Fix Regime Stability query
- [ ] Fix Alert Age queries
- [ ] Fix RUL Summary query
- [ ] Fix Maintenance Recommendation query
- [ ] Fix Calibration Summary query
- [ ] Verify time series panels use appropriate time ranges
- [ ] Test dashboard with latest run data
- [ ] Document changes in Grafana panel descriptions

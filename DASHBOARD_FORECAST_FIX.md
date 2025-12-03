# Dashboard Forecast Visualization Issue

## Problem
Grafana "Health Forecast (Stitched As-Of)" panel shows unrealistic spikes: 0→100→0→100 pattern.

## Root Cause
The dashboard query is selecting **all forecast runs** from `ACM_HealthForecast_TS` without filtering to the latest RunID. This causes:
- Old forecasts (from earlier batches when health was low) showing 0-20 range
- New forecasts (current health ~71% with positive trend) clamped at 100
- Time-based stitching creates saw-tooth pattern

## Evidence
```sql
SELECT TOP 10 Timestamp, CAST(ForecastHealth AS DECIMAL(10,2)) AS ForecastHealth 
FROM ACM_HealthForecast_TS WHERE EquipID=1 ORDER BY Timestamp DESC;
```
Returns all rows showing `100.00` for recent timestamps (clamped forecast values).

## Solution
Dashboard query MUST filter to only the latest RunID per equipment:

```sql
-- Option 1: Latest RunID per EquipID
SELECT f.Timestamp, f.ForecastHealth, f.CiLower, f.CiUpper
FROM ACM_HealthForecast_TS f
INNER JOIN (
    SELECT EquipID, MAX(RunID) AS LatestRunID
    FROM ACM_Runs
    WHERE Outcome = 'OK'
    GROUP BY EquipID
) latest ON f.EquipID = latest.EquipID AND f.RunID = latest.LatestRunID
WHERE f.EquipID = $EquipmentID
ORDER BY f.Timestamp;

-- Option 2: Use CreatedAt timestamp (if RunID ordering unreliable)
SELECT f.Timestamp, f.ForecastHealth, f.CiLower, f.CiUpper
FROM ACM_HealthForecast_TS f
WHERE f.EquipID = $EquipmentID
AND f.CreatedAt >= DATEADD(MINUTE, -10, GETDATE())  -- Only forecasts from last 10 min
ORDER BY f.Timestamp;

-- Option 3: Window function (most robust)
WITH RankedForecasts AS (
    SELECT *, 
           ROW_NUMBER() OVER (PARTITION BY EquipID, Timestamp ORDER BY CreatedAt DESC) AS rn
    FROM ACM_HealthForecast_TS
    WHERE EquipID = $EquipmentID
)
SELECT Timestamp, ForecastHealth, CiLower, CiUpper
FROM RankedForecasts
WHERE rn = 1
ORDER BY Timestamp;
```

## Grafana Panel Update
1. Navigate to "ACM Asset Health Dashboard 2" → "Health Forecast (Stitched As-Of)" panel
2. Edit panel → Query tab
3. Replace current query with Option 3 (most robust)
4. Test with different equipment IDs
5. Save dashboard

## Related Code
- Forecast clamping at `core/forecasting.py:1606`: `forecast_val = max(health_min, min(health_max, forecast_val))`
  - **This is correct** - prevents physically impossible values >100% or <0%
  - Dashboard must handle multiple forecast runs via RunID filtering

## Status
- ✅ Root cause identified
- ❌ Dashboard query needs update (Grafana admin action required)
- ✅ Code behavior is correct (clamping prevents invalid health values)

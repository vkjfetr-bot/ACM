-- Data Quality Monitoring Queries for Grafana Dashboard
-- Purpose: Track health volatility, extreme anomalies, and smoothing effectiveness

-- Query 1: Health Quality Flag Distribution (Stat Panel)
-- Shows count of quality issues in time range
SELECT 
    QualityFlag,
    COUNT(*) AS Count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS Percentage
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY QualityFlag
ORDER BY Count DESC;

-- Query 2: Smoothed vs Raw Health Comparison (Time Series Panel)
-- Shows effectiveness of smoothing
SELECT 
    Timestamp AS time,
    HealthIndex AS value,
    'Smoothed Health' AS metric
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
UNION ALL
SELECT 
    Timestamp AS time,
    RawHealthIndex AS value,
    'Raw Health (Unsmoothed)' AS metric
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
  AND RawHealthIndex IS NOT NULL
ORDER BY time ASC;

-- Query 3: Volatile Health Transitions (Table Panel)
-- Lists periods with excessive health changes
WITH HealthChanges AS (
    SELECT 
        Timestamp,
        HealthIndex,
        RawHealthIndex,
        LAG(HealthIndex) OVER (ORDER BY Timestamp) AS PrevHealth,
        ABS(HealthIndex - LAG(HealthIndex) OVER (ORDER BY Timestamp)) AS HealthChange,
        FusedZ,
        QualityFlag
    FROM ACM_HealthTimeline
    WHERE EquipID = $equipment
      AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
)
SELECT TOP 20
    FORMAT(Timestamp, 'yyyy-MM-dd HH:mm') AS 'Time',
    ROUND(PrevHealth, 1) AS 'Previous Health',
    ROUND(HealthIndex, 1) AS 'Current Health',
    ROUND(HealthChange, 1) AS 'Change (abs)',
    ROUND(FusedZ, 2) AS 'FusedZ',
    QualityFlag AS 'Quality'
FROM HealthChanges
WHERE HealthChange > 15.0  -- Significant changes only
ORDER BY HealthChange DESC;

-- Query 4: Extreme Anomaly Events (Time Series - Annotations)
-- Mark periods with extreme Z-scores
SELECT 
    Timestamp AS time,
    'Extreme Anomaly Detected' AS text,
    'extreme-anomaly' AS tags
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
  AND QualityFlag = 'EXTREME_ANOMALY';

-- Query 5: Health Volatility Trend (Time Series Panel)
-- Shows rate of change over time
WITH HealthChanges AS (
    SELECT 
        Timestamp,
        ABS(HealthIndex - LAG(HealthIndex) OVER (ORDER BY Timestamp)) AS HealthChange
    FROM ACM_HealthTimeline
    WHERE EquipID = $equipment
      AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
)
SELECT 
    Timestamp AS time,
    HealthChange AS value,
    'Health Volatility (% change)' AS metric
FROM HealthChanges
WHERE HealthChange IS NOT NULL
ORDER BY time ASC;

-- Query 6: Data Quality Summary (Table Panel - Current Status)
-- Overall health data quality assessment
SELECT 
    'Total Records' AS Metric,
    CAST(COUNT(*) AS NVARCHAR) AS Value
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
UNION ALL
SELECT 
    'Normal Quality',
    CAST(SUM(CASE WHEN QualityFlag = 'NORMAL' THEN 1 ELSE 0 END) AS NVARCHAR)
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
UNION ALL
SELECT 
    'Volatile Transitions',
    CAST(SUM(CASE WHEN QualityFlag = 'VOLATILE' THEN 1 ELSE 0 END) AS NVARCHAR)
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
UNION ALL
SELECT 
    'Extreme Anomalies',
    CAST(SUM(CASE WHEN QualityFlag = 'EXTREME_ANOMALY' THEN 1 ELSE 0 END) AS NVARCHAR)
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
UNION ALL
SELECT 
    'Avg Smoothed Health',
    CAST(ROUND(AVG(HealthIndex), 1) AS NVARCHAR) + '%'
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
UNION ALL
SELECT 
    'Avg Raw Health',
    CAST(ROUND(AVG(RawHealthIndex), 1) AS NVARCHAR) + '%'
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo()
  AND RawHealthIndex IS NOT NULL
UNION ALL
SELECT 
    'Max Volatility',
    CAST(ROUND(MAX(ABS(HealthIndex - LAG(HealthIndex) OVER (ORDER BY Timestamp))), 1) AS NVARCHAR) + '%'
FROM ACM_HealthTimeline
WHERE EquipID = $equipment
  AND Timestamp BETWEEN $__timeFrom() AND $__timeTo();

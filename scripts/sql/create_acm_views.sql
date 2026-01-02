-- ==============================================================================
-- ACM Dashboard Views v11.1.5
-- Purpose: Pre-built views optimized for Grafana dashboards
-- Created: 2025-02-02
-- ==============================================================================
-- Design Principles:
--   1. Each view answers a specific question for operators/engineers
--   2. All views join Equipment table for human-readable names
--   3. Time columns are DATETIME (not formatted strings) for Grafana time_series
--   4. Views are optimized for common dashboard queries
-- ==============================================================================

USE ACM;
GO

-- ==============================================================================
-- DROP EXISTING VIEWS
-- ==============================================================================
IF OBJECT_ID('dbo.vw_ACM_ResourceSummary', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_ResourceSummary;
IF OBJECT_ID('dbo.vw_ACM_CurrentHealth', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_CurrentHealth;
IF OBJECT_ID('dbo.vw_ACM_HealthHistory', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_HealthHistory;
IF OBJECT_ID('dbo.vw_ACM_ActiveDefects', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_ActiveDefects;
IF OBJECT_ID('dbo.vw_ACM_TopSensorContributors', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_TopSensorContributors;
IF OBJECT_ID('dbo.vw_ACM_RecentEpisodes', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_RecentEpisodes;
IF OBJECT_ID('dbo.vw_ACM_RULSummary', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_RULSummary;
IF OBJECT_ID('dbo.vw_ACM_DetectorScores', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_DetectorScores;
IF OBJECT_ID('dbo.vw_ACM_RegimeAnalysis', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_RegimeAnalysis;
IF OBJECT_ID('dbo.vw_ACM_DriftStatus', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_DriftStatus;
IF OBJECT_ID('dbo.vw_ACM_RunHistory', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_RunHistory;
IF OBJECT_ID('dbo.vw_ACM_EquipmentOverview', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_EquipmentOverview;
IF OBJECT_ID('dbo.vw_ACM_SensorForecasts', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_SensorForecasts;
IF OBJECT_ID('dbo.vw_ACM_HealthForecasts', 'V') IS NOT NULL DROP VIEW dbo.vw_ACM_HealthForecasts;
GO

PRINT 'Old views dropped successfully.';
GO

-- ==============================================================================
-- VIEW 1: vw_ACM_CurrentHealth
-- Purpose: Current health status for all equipment (single row per equipment)
-- Grafana Panel: Stat panels, Gauge panels, Status indicators
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_CurrentHealth
AS
WITH LatestHealth AS (
    SELECT 
        EquipID,
        HealthIndex,
        HealthZone,
        Confidence,
        Timestamp,
        FusedZ,
        ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY Timestamp DESC) AS rn
    FROM ACM_HealthTimeline
)
SELECT 
    e.EquipCode,
    e.EquipName,
    lh.HealthIndex,
    lh.HealthZone,
    lh.Confidence,
    lh.FusedZ AS FusedScore,
    lh.Timestamp AS LastUpdate,
    CASE 
        WHEN lh.HealthZone = 'CRITICAL' THEN 1
        WHEN lh.HealthZone = 'WARNING' THEN 2
        WHEN lh.HealthZone = 'CAUTION' THEN 3
        ELSE 4 
    END AS SeverityRank
FROM LatestHealth lh
INNER JOIN Equipment e ON e.EquipID = lh.EquipID
WHERE lh.rn = 1;
GO

PRINT 'Created vw_ACM_CurrentHealth';
GO

-- ==============================================================================
-- VIEW 2: vw_ACM_HealthHistory
-- Purpose: Health timeline for time series plots
-- Grafana Panel: Time series panel with spanNulls
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_HealthHistory
AS
SELECT 
    ht.Timestamp AS time,
    e.EquipCode,
    e.EquipName,
    ht.HealthIndex,
    ht.HealthZone,
    ht.Confidence,
    ht.FusedZ,
    ht.RunID
FROM ACM_HealthTimeline ht
INNER JOIN Equipment e ON e.EquipID = ht.EquipID;
GO

PRINT 'Created vw_ACM_HealthHistory';
GO

-- ==============================================================================
-- VIEW 3: vw_ACM_ActiveDefects
-- Purpose: Current sensor defects requiring attention
-- Grafana Panel: Table panel with color-coded severity
-- Note: ACM_SensorDefects is by DetectorType, not SensorName
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_ActiveDefects
AS
SELECT 
    e.EquipCode,
    e.EquipName,
    sd.DetectorType,
    sd.DetectorFamily,
    sd.CurrentZ AS ZScore,
    sd.Severity,
    sd.ActiveDefect,
    sd.ViolationPct,
    sd.MaxZ,
    sd.AvgZ,
    sd.RunID
FROM ACM_SensorDefects sd
INNER JOIN Equipment e ON e.EquipID = sd.EquipID
WHERE sd.ActiveDefect = 1;
GO

PRINT 'Created vw_ACM_ActiveDefects';
GO

-- ==============================================================================
-- VIEW 4: vw_ACM_TopSensorContributors
-- Purpose: Top contributing sensors to anomalies
-- Grafana Panel: Bar chart, Table panel
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_TopSensorContributors
AS
SELECT 
    e.EquipCode,
    e.EquipName,
    sh.SensorName,
    sh.FailureContribution AS ContributionPct,
    sh.MaxAbsZ,
    sh.LatestAbsZ,
    sh.AlertCount,
    sh.RunID
FROM ACM_SensorHotspots sh
INNER JOIN Equipment e ON e.EquipID = sh.EquipID;
GO

PRINT 'Created vw_ACM_TopSensorContributors';
GO

-- ==============================================================================
-- VIEW 5: vw_ACM_RecentEpisodes
-- Purpose: Recent anomaly episodes with diagnostics
-- Grafana Panel: Table panel, Timeline panel
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_RecentEpisodes
AS
SELECT 
    ed.EpisodeID,
    e.EquipCode,
    e.EquipName,
    ed.StartTime,
    ed.EndTime,
    ROUND(ed.DurationHours * 60, 0) AS DurationMinutes,
    ed.PeakZ,
    ed.AvgZ,
    ed.Severity,
    ed.TopSensor1,
    ed.TopSensor2,
    ed.TopSensor3,
    ed.RegimeAtStart,
    ed.AlertMode,
    ed.RunID,
    ed.CreatedAt
FROM ACM_EpisodeDiagnostics ed
INNER JOIN Equipment e ON e.EquipID = ed.EquipID;
GO

PRINT 'Created vw_ACM_RecentEpisodes';
GO

-- ==============================================================================
-- VIEW 6: vw_ACM_RULSummary
-- Purpose: Remaining Useful Life summary (latest per equipment)
-- Grafana Panel: Stat panel, Table panel
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_RULSummary
AS
WITH LatestRUL AS (
    SELECT 
        EquipID,
        RUL_Hours,
        P10_LowerBound,
        P50_Median,
        P90_UpperBound,
        Confidence,
        RUL_Status,
        MaturityState,
        Method,
        TopSensor1,
        TopSensor2,
        TopSensor3,
        CreatedAt,
        ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY CreatedAt DESC) AS rn
    FROM ACM_RUL
)
SELECT 
    e.EquipCode,
    e.EquipName,
    lr.RUL_Hours,
    lr.P10_LowerBound,
    lr.P50_Median,
    lr.P90_UpperBound,
    lr.Confidence,
    lr.RUL_Status,
    lr.MaturityState,
    lr.Method,
    lr.TopSensor1,
    lr.TopSensor2,
    lr.TopSensor3,
    lr.CreatedAt AS CalculatedAt,
    CASE 
        WHEN lr.RUL_Hours <= 24 THEN 'CRITICAL'
        WHEN lr.RUL_Hours <= 168 THEN 'WARNING'  -- 7 days
        WHEN lr.RUL_Hours <= 720 THEN 'CAUTION'  -- 30 days
        ELSE 'HEALTHY'
    END AS RULStatusCalc
FROM LatestRUL lr
INNER JOIN Equipment e ON e.EquipID = lr.EquipID
WHERE lr.rn = 1;
GO

PRINT 'Created vw_ACM_RULSummary';
GO

-- ==============================================================================
-- VIEW 7: vw_ACM_DetectorScores
-- Purpose: Detector scores for time series analysis
-- Grafana Panel: Time series panel (multi-line)
-- Note: Uses actual column names from ACM_Scores_Wide
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_DetectorScores
AS
SELECT 
    sw.Timestamp AS time,
    e.EquipCode,
    e.EquipName,
    sw.ar1_z AS AR1,
    sw.pca_spe_z AS PCA_SPE,
    sw.pca_t2_z AS PCA_T2,
    sw.iforest_z AS IForest,
    sw.gmm_z AS GMM,
    sw.cusum_z AS CUSUM,
    sw.fused AS Fused,
    sw.regime_label AS RegimeLabel,
    sw.RunID
FROM ACM_Scores_Wide sw
INNER JOIN Equipment e ON e.EquipID = sw.EquipID;
GO

PRINT 'Created vw_ACM_DetectorScores';
GO

-- ==============================================================================
-- VIEW 8: vw_ACM_RegimeAnalysis
-- Purpose: Operating regime occupancy and transitions
-- Grafana Panel: Pie chart, Bar chart
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_RegimeAnalysis
AS
SELECT 
    e.EquipCode,
    e.EquipName,
    ro.RegimeLabel,
    ro.DwellFraction AS OccupancyPct,
    ROUND(ro.DwellTimeHours * 60, 0) AS TotalMinutes,
    ro.EntryCount AS TransitionCount,
    ro.AvgDwellMinutes,
    ro.RunID,
    ro.CreatedAt
FROM ACM_RegimeOccupancy ro
INNER JOIN Equipment e ON e.EquipID = ro.EquipID;
GO

PRINT 'Created vw_ACM_RegimeAnalysis';
GO

-- ==============================================================================
-- VIEW 9: vw_ACM_DriftStatus
-- Purpose: Drift detection status per equipment
-- Grafana Panel: Status indicators, Time series
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_DriftStatus
AS
WITH LatestDrift AS (
    SELECT 
        EquipID,
        ControllerState,
        LastDriftValue,
        LastDriftTime,
        Threshold,
        Sensitivity,
        CreatedAt,
        ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY CreatedAt DESC) AS rn
    FROM ACM_DriftController
)
SELECT 
    e.EquipCode,
    e.EquipName,
    ld.ControllerState AS DriftState,
    ld.LastDriftValue,
    ld.Threshold,
    ld.Sensitivity,
    ld.LastDriftTime,
    ld.CreatedAt AS LastChecked,
    CASE 
        WHEN ld.ControllerState = 'FAULT' THEN 1
        WHEN ld.ControllerState = 'DRIFTING' THEN 2
        ELSE 3
    END AS SeverityRank
FROM LatestDrift ld
INNER JOIN Equipment e ON e.EquipID = ld.EquipID
WHERE ld.rn = 1;
GO

PRINT 'Created vw_ACM_DriftStatus';
GO

-- ==============================================================================
-- VIEW 10: vw_ACM_RunHistory
-- Purpose: ACM run history with outcomes
-- Grafana Panel: Table panel, Timeline
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_RunHistory
AS
SELECT 
    r.RunID,
    e.EquipCode,
    e.EquipName,
    r.StartedAt,
    r.CompletedAt,
    r.DurationSeconds,
    r.HealthStatus AS Outcome,
    r.ScoreRowCount AS RowsProcessed,
    r.ConfigSignature,
    r.ErrorMessage
FROM ACM_Runs r
INNER JOIN Equipment e ON e.EquipID = r.EquipID;
GO

PRINT 'Created vw_ACM_RunHistory';
GO

-- ==============================================================================
-- VIEW 11: vw_ACM_EquipmentOverview
-- Purpose: Complete equipment overview with health, RUL, and defects
-- Grafana Panel: Table panel (main dashboard summary)
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_EquipmentOverview
AS
WITH LatestHealth AS (
    SELECT EquipID, HealthIndex, HealthZone, Confidence, Timestamp,
           ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY Timestamp DESC) AS rn
    FROM ACM_HealthTimeline
),
LatestRUL AS (
    SELECT EquipID, RUL_Hours, P50_Median, RUL_Status,
           ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY CreatedAt DESC) AS rn
    FROM ACM_RUL
),
ActiveDefectCount AS (
    SELECT EquipID, COUNT(*) AS DefectCount
    FROM ACM_SensorDefects
    WHERE ActiveDefect = 1
    GROUP BY EquipID
),
LatestRun AS (
    SELECT EquipID, RunID, StartedAt, HealthStatus,
           ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY StartedAt DESC) AS rn
    FROM ACM_Runs
)
SELECT 
    e.EquipCode,
    e.EquipName,
    e.Area,
    e.Unit,
    COALESCE(lh.HealthIndex, 0) AS HealthIndex,
    COALESCE(lh.HealthZone, 'UNKNOWN') AS HealthZone,
    lh.Timestamp AS HealthUpdatedAt,
    COALESCE(lr.RUL_Hours, 0) AS RUL_Hours,
    COALESCE(lr.P50_Median, 0) AS RUL_P50,
    COALESCE(lr.RUL_Status, 'UNKNOWN') AS RUL_Reliability,
    COALESCE(adc.DefectCount, 0) AS ActiveDefects,
    lrun.StartedAt AS LastRunAt,
    lrun.HealthStatus AS LastRunOutcome,
    CASE 
        WHEN lh.HealthZone = 'CRITICAL' OR (lr.RUL_Hours IS NOT NULL AND lr.RUL_Hours <= 24) THEN 'CRITICAL'
        WHEN lh.HealthZone = 'WARNING' OR (lr.RUL_Hours IS NOT NULL AND lr.RUL_Hours <= 168) THEN 'WARNING'
        WHEN lh.HealthZone = 'CAUTION' OR (lr.RUL_Hours IS NOT NULL AND lr.RUL_Hours <= 720) THEN 'CAUTION'
        WHEN lh.HealthIndex IS NULL THEN 'NO DATA'
        ELSE 'HEALTHY'
    END AS OverallStatus
FROM Equipment e
LEFT JOIN LatestHealth lh ON lh.EquipID = e.EquipID AND lh.rn = 1
LEFT JOIN LatestRUL lr ON lr.EquipID = e.EquipID AND lr.rn = 1
LEFT JOIN ActiveDefectCount adc ON adc.EquipID = e.EquipID
LEFT JOIN LatestRun lrun ON lrun.EquipID = e.EquipID AND lrun.rn = 1;
GO

PRINT 'Created vw_ACM_EquipmentOverview';
GO

-- ==============================================================================
-- VIEW 12: vw_ACM_SensorForecasts
-- Purpose: Sensor value forecasts for trend analysis
-- Grafana Panel: Time series panel
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_SensorForecasts
AS
SELECT 
    sf.Timestamp AS time,
    e.EquipCode,
    e.EquipName,
    sf.SensorName,
    sf.ForecastValue,
    sf.CiLower AS LowerBound,
    sf.CiUpper AS UpperBound,
    sf.ForecastStd,
    sf.Method,
    sf.RegimeLabel,
    sf.CreatedAt
FROM ACM_SensorForecast sf
INNER JOIN Equipment e ON e.EquipID = sf.EquipID;
GO

PRINT 'Created vw_ACM_SensorForecasts';
GO

-- ==============================================================================
-- VIEW 13: vw_ACM_HealthForecasts
-- Purpose: Health trajectory forecasts
-- Grafana Panel: Time series panel with confidence bands
-- ==============================================================================
CREATE VIEW dbo.vw_ACM_HealthForecasts
AS
SELECT 
    hf.Timestamp AS time,
    e.EquipCode,
    e.EquipName,
    hf.ForecastHealth AS HealthIndex,
    hf.CiLower AS LowerBound,
    hf.CiUpper AS UpperBound,
    hf.ForecastStd,
    hf.Method,
    hf.RegimeLabel,
    hf.CreatedAt
FROM ACM_HealthForecast hf
INNER JOIN Equipment e ON e.EquipID = hf.EquipID;
GO

PRINT 'Created vw_ACM_HealthForecasts';
GO

-- ==============================================================================
-- SUMMARY
-- ==============================================================================
PRINT '';
PRINT '=================================================================';
PRINT 'ACM Dashboard Views Created Successfully (v11.1.5)';
PRINT '=================================================================';
PRINT '';
PRINT 'Views Created:';
PRINT '  1. vw_ACM_CurrentHealth      - Current health per equipment';
PRINT '  2. vw_ACM_HealthHistory      - Health time series';
PRINT '  3. vw_ACM_ActiveDefects      - Active sensor defects';
PRINT '  4. vw_ACM_TopSensorContributors - Top anomaly contributors';
PRINT '  5. vw_ACM_RecentEpisodes     - Recent anomaly episodes';
PRINT '  6. vw_ACM_RULSummary         - Remaining useful life summary';
PRINT '  7. vw_ACM_DetectorScores     - Detector scores time series';
PRINT '  8. vw_ACM_RegimeAnalysis     - Operating regime analysis';
PRINT '  9. vw_ACM_DriftStatus        - Drift detection status';
PRINT ' 10. vw_ACM_RunHistory         - ACM run history';
PRINT ' 11. vw_ACM_EquipmentOverview  - Complete equipment summary';
PRINT ' 12. vw_ACM_SensorForecasts    - Sensor value forecasts';
PRINT ' 13. vw_ACM_HealthForecasts    - Health trajectory forecasts';
PRINT '';
PRINT 'Old view removed: vw_ACM_ResourceSummary';
PRINT '=================================================================';
GO

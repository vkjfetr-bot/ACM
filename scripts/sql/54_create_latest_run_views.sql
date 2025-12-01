-- 54_create_latest_run_views.sql
-- Creates views that automatically filter to the latest RunID for each timestamp
-- This fixes Grafana visualization issues caused by overlapping batch runs

USE ACM;
GO

-- View for RegimeTimeline (fixes wave pattern issue)
CREATE OR ALTER VIEW dbo.ACM_RegimeTimeline_Latest AS
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
GO

-- View for HealthTimeline
CREATE OR ALTER VIEW dbo.ACM_HealthTimeline_Latest AS
SELECT ht.*
FROM dbo.ACM_HealthTimeline ht
INNER JOIN (
    SELECT EquipID, Timestamp, MAX(RunID) AS LatestRunID
    FROM dbo.ACM_HealthTimeline
    GROUP BY EquipID, Timestamp
) latest 
ON ht.EquipID = latest.EquipID 
AND ht.Timestamp = latest.Timestamp 
AND ht.RunID = latest.LatestRunID;
GO

-- View for Scores_Wide
CREATE OR ALTER VIEW dbo.ACM_Scores_Wide_Latest AS
SELECT sw.*
FROM dbo.ACM_Scores_Wide sw
INNER JOIN (
    SELECT EquipID, Timestamp, MAX(RunID) AS LatestRunID
    FROM dbo.ACM_Scores_Wide
    GROUP BY EquipID, Timestamp
) latest 
ON sw.EquipID = latest.EquipID 
AND sw.Timestamp = latest.Timestamp 
AND sw.RunID = latest.LatestRunID;
GO

-- View for ThresholdCrossings (aggregated by Equipment+Run, no Timestamp)
-- This one doesn't need timestamp deduplication but include for consistency
CREATE OR ALTER VIEW dbo.ACM_ThresholdCrossings_Latest AS
SELECT tc.*
FROM dbo.ACM_ThresholdCrossings tc
INNER JOIN (
    SELECT EquipID, MAX(RunID) AS LatestRunID
    FROM dbo.ACM_ThresholdCrossings
    GROUP BY EquipID
) latest 
ON tc.EquipID = latest.EquipID 
AND tc.RunID = latest.LatestRunID;
GO

-- View for DefectSummary (aggregated by Equipment+Run, no Timestamp)
CREATE OR ALTER VIEW dbo.ACM_DefectSummary_Latest AS
SELECT ds.*
FROM dbo.ACM_DefectSummary ds
INNER JOIN (
    SELECT EquipID, MAX(RunID) AS LatestRunID
    FROM dbo.ACM_DefectSummary
    GROUP BY EquipID
) latest 
ON ds.EquipID = latest.EquipID 
AND ds.RunID = latest.LatestRunID;
GO

-- View for Episodes (aggregated by Equipment+Run, summary table)
CREATE OR ALTER VIEW dbo.ACM_Episodes_Latest AS
SELECT e.*
FROM dbo.ACM_Episodes e
INNER JOIN (
    SELECT EquipID, MAX(RunID) AS LatestRunID
    FROM dbo.ACM_Episodes
    GROUP BY EquipID
) latest 
ON e.EquipID = latest.EquipID 
AND e.RunID = latest.LatestRunID;
GO

-- View for SensorHotspots (if any exist)
CREATE OR ALTER VIEW dbo.ACM_SensorHotspots_Latest AS
SELECT sh.*
FROM dbo.ACM_SensorHotspots sh
INNER JOIN (
    SELECT EquipID, SensorName, MAX(RunID) AS LatestRunID
    FROM dbo.ACM_SensorHotspots
    GROUP BY EquipID, SensorName
) latest 
ON sh.EquipID = latest.EquipID 
AND sh.SensorName = latest.SensorName 
AND sh.RunID = latest.LatestRunID;
GO

PRINT 'Successfully created 7 _Latest views for deduplicating batch run overlaps';
PRINT 'Grafana dashboards should query these views instead of base tables';
PRINT '';
PRINT 'View List:';
PRINT '  - ACM_RegimeTimeline_Latest';
PRINT '  - ACM_HealthTimeline_Latest';
PRINT '  - ACM_Scores_Wide_Latest';
PRINT '  - ACM_ThresholdCrossings_Latest';
PRINT '  - ACM_DefectSummary_Latest';
PRINT '  - ACM_Episodes_Latest';
PRINT '  - ACM_SensorHotspots_Latest';
GO

USE [ACM];
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

CREATE OR ALTER VIEW ACM_LatestThresholds AS
SELECT 
    tm.EquipID,
    e.EquipName,
    tm.RegimeID,
    tm.ThresholdType,
    tm.ThresholdValue,
    tm.CalculationMethod,
    tm.SampleCount,
    tm.CreatedAt,
    tm.ConfigSignature
FROM ACM_ThresholdMetadata tm
INNER JOIN Equipment e ON tm.EquipID = e.EquipID
WHERE tm.IsActive = 1
AND tm.ThresholdID IN (
    -- Get latest threshold for each (EquipID, RegimeID, ThresholdType) combination
    SELECT MAX(ThresholdID)
    FROM ACM_ThresholdMetadata
    WHERE IsActive = 1
    GROUP BY EquipID, RegimeID, ThresholdType
);
GO

CREATE OR ALTER VIEW ACM_OMR_SensorContributions AS 
          SELECT * FROM ACM_OMRContributionsLong
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

CREATE OR ALTER VIEW dbo.v_Equip_Anomalies AS
SELECT EquipID, StartEntryDateTime, EndEntryDateTime, Severity, Detector, Score, RunID
FROM dbo.AnomalyEvents;
GO

CREATE OR ALTER VIEW dbo.v_Equip_DriftTS AS
SELECT EquipID, EntryDateTime, Method, DriftZ, RunID
FROM dbo.DriftTS;
GO

CREATE OR ALTER VIEW dbo.v_Equip_SensorTS AS
SELECT EquipID, EntryDateTime, Sensor, Value, RunID
FROM dbo.ScoresTS;
GO

CREATE OR ALTER VIEW dbo.v_PCA_Loadings AS
SELECT RunID, EntryDateTime, ComponentNo, Sensor, Loading
FROM dbo.PCA_Components;
GO

CREATE OR ALTER VIEW dbo.v_PCA_Scree AS
SELECT RunID, EquipID, EntryDateTime, VarExplainedJSON
FROM dbo.PCA_Model;
GO

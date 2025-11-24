/*
ACM Repair Script - Create Missing Tables and Synonyms
======================================================
This script repairs the SQL schema by creating missing tables and synonyms
that are required by the Python codebase but were found to be missing.

Tables covered:
- ScoresTS (ACM_Scores_Long)
- DriftTS (ACM_Drift_TS)
- AnomalyEvents (ACM_Anomaly_Events)
- RegimeEpisodes (ACM_Regime_Episodes)
- PCA_Model (ACM_PCA_Models)
- PCA_Components (ACM_PCA_Loadings)
- PCA_Metrics (ACM_PCA_Metrics)
- RunStats (ACM_Run_Stats)
- ACM_BaselineBuffer
- ACM_EpisodeCulprits
- ACM_EpisodeDiagnostics
*/

USE [ACM];
GO

-- ============================================================================
-- 1. Core Tables (from scripts/sql/10_core_tables.sql)
-- ============================================================================

-- ScoresTS (long format)
IF OBJECT_ID('dbo.ScoresTS','U') IS NULL
BEGIN
    CREATE TABLE dbo.ScoresTS (
        EntryDateTime   datetime2(3) NOT NULL,
        EquipID         int NOT NULL, -- REFERENCES dbo.Equipments(EquipID),
        Sensor          nvarchar(64) NOT NULL,
        Value           float NULL,
        Source          nvarchar(32) NULL,
        RunID           uniqueidentifier NOT NULL
    );
    CREATE INDEX IX_Scores_Equip_Time ON dbo.ScoresTS(EquipID, EntryDateTime);
    CREATE INDEX IX_Scores_RunID ON dbo.ScoresTS(RunID);
    PRINT 'Created table: ScoresTS';
END
GO

-- DriftTS
IF OBJECT_ID('dbo.DriftTS','U') IS NULL
BEGIN
    CREATE TABLE dbo.DriftTS (
        EntryDateTime   datetime2(3) NOT NULL,
        EquipID         int NOT NULL, -- REFERENCES dbo.Equipments(EquipID),
        DriftZ          float NOT NULL,
        Method          nvarchar(32) NOT NULL,
        RunID           uniqueidentifier NOT NULL
    );
    CREATE INDEX IX_Drift_Equip_Time ON dbo.DriftTS(EquipID, EntryDateTime);
    CREATE INDEX IX_Drift_RunID ON dbo.DriftTS(RunID);
    PRINT 'Created table: DriftTS';
END
GO

-- AnomalyEvents
IF OBJECT_ID('dbo.AnomalyEvents','U') IS NULL
BEGIN
    CREATE TABLE dbo.AnomalyEvents (
        EquipID              int NOT NULL, -- REFERENCES dbo.Equipments(EquipID),
        StartEntryDateTime   datetime2(3) NOT NULL,
        EndEntryDateTime     datetime2(3) NOT NULL,
        Severity             nvarchar(16) NOT NULL,
        Detector             nvarchar(32) NOT NULL,
        Score                float NULL,
        ContributorsJSON     nvarchar(max) NULL,
        RunID                uniqueidentifier NOT NULL
    );
    CREATE INDEX IX_Anom_Equip_Start ON dbo.AnomalyEvents(EquipID, StartEntryDateTime);
    CREATE INDEX IX_Anom_RunID ON dbo.AnomalyEvents(RunID);
    PRINT 'Created table: AnomalyEvents';
END
GO

-- RegimeEpisodes
IF OBJECT_ID('dbo.RegimeEpisodes','U') IS NULL
BEGIN
    CREATE TABLE dbo.RegimeEpisodes (
        EquipID              int NOT NULL, -- REFERENCES dbo.Equipments(EquipID),
        StartEntryDateTime   datetime2(3) NOT NULL,
        EndEntryDateTime     datetime2(3) NOT NULL,
        RegimeLabel          nvarchar(32) NOT NULL,
        Confidence           float NULL,
        RunID                uniqueidentifier NOT NULL
    );
    CREATE INDEX IX_Regime_Equip_Start ON dbo.RegimeEpisodes(EquipID, StartEntryDateTime);
    CREATE INDEX IX_Regime_RunID ON dbo.RegimeEpisodes(RunID);
    PRINT 'Created table: RegimeEpisodes';
END
GO

-- PCA_Model
IF OBJECT_ID('dbo.PCA_Model','U') IS NULL
BEGIN
    CREATE TABLE dbo.PCA_Model (
        RunID                  uniqueidentifier NOT NULL PRIMARY KEY,
        EquipID                int NOT NULL, -- REFERENCES dbo.Equipments(EquipID),
        EntryDateTime          datetime2(3) NOT NULL,
        NComponents            int NOT NULL,
        TargetVar              nvarchar(max) NULL,
        VarExplainedJSON       nvarchar(max) NULL,
        ScalingSpecJSON        nvarchar(max) NULL,
        ModelVersion           nvarchar(32) NULL,
        TrainStartEntryDateTime datetime2(3) NULL,
        TrainEndEntryDateTime   datetime2(3) NULL
    );
    PRINT 'Created table: PCA_Model';
END
GO

-- PCA_Components
IF OBJECT_ID('dbo.PCA_Components','U') IS NULL
BEGIN
    CREATE TABLE dbo.PCA_Components (
        RunID          uniqueidentifier NOT NULL,
        EntryDateTime  datetime2(3) NOT NULL,
        ComponentNo    int NOT NULL,
        Sensor         nvarchar(64) NOT NULL,
        Loading        float NOT NULL
    );
    CREATE INDEX IX_PCAComp_Run ON dbo.PCA_Components(RunID);
    PRINT 'Created table: PCA_Components';
END
GO

-- PCA_Metrics
IF OBJECT_ID('dbo.PCA_Metrics','U') IS NULL
BEGIN
    CREATE TABLE dbo.PCA_Metrics (
        RunID          uniqueidentifier NOT NULL,
        EntryDateTime  datetime2(3) NOT NULL,
        Var90_N        int NULL,
        ReconRMSE      float NULL,
        P95_ReconRMSE  float NULL,
        Notes          nvarchar(max) NULL
    );
    CREATE INDEX IX_PCAMet_Run ON dbo.PCA_Metrics(RunID);
    PRINT 'Created table: PCA_Metrics';
END
GO

-- RunStats
IF OBJECT_ID('dbo.RunStats','U') IS NULL
BEGIN
    CREATE TABLE dbo.RunStats (
        RunID                   uniqueidentifier NOT NULL PRIMARY KEY,
        EquipID                 int NOT NULL, -- REFERENCES dbo.Equipments(EquipID),
        WindowStartEntryDateTime datetime2(3) NOT NULL,
        WindowEndEntryDateTime   datetime2(3) NOT NULL,
        SamplesIn               int NULL,
        SamplesKept             int NULL,
        SensorsKept             int NULL,
        CadenceOKPct            float NULL,
        DriftP95                float NULL,
        ReconRMSE               float NULL,
        AnomalyCount            int NULL
    );
    PRINT 'Created table: RunStats';
END
GO

-- ============================================================================
-- 2. Missing Tables (from docs/sql/ACM_SchemaExtensions.sql and inferred)
-- ============================================================================

-- ACM_BaselineBuffer
IF OBJECT_ID('dbo.ACM_BaselineBuffer','U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_BaselineBuffer (
        ID BIGINT IDENTITY(1,1) PRIMARY KEY,
        EquipID INT NOT NULL,
        Timestamp DATETIME2(3) NOT NULL,
        SensorName VARCHAR(200) NOT NULL,
        SensorValue FLOAT NOT NULL,
        DataQuality FLOAT NULL,
        CreatedAt DATETIME2(3) NOT NULL DEFAULT GETUTCDATE()
    );
    CREATE CLUSTERED INDEX IX_BaselineBuffer_EquipID_Timestamp ON dbo.ACM_BaselineBuffer(EquipID, Timestamp DESC);
    CREATE INDEX IX_BaselineBuffer_SensorName ON dbo.ACM_BaselineBuffer(SensorName);
    PRINT 'Created table: ACM_BaselineBuffer';
END
GO

-- ACM_EpisodeCulprits
IF OBJECT_ID('dbo.ACM_EpisodeCulprits','U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_EpisodeCulprits (
        ID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EpisodeID INT NOT NULL,
        DetectorType NVARCHAR(64) NULL,
        SensorName NVARCHAR(200) NULL,
        ContributionPct FLOAT NULL,
        Rank INT NULL,
        CreatedAt DATETIME2(3) NOT NULL DEFAULT GETUTCDATE()
    );
    CREATE INDEX IX_EpisodeCulprits_RunID ON dbo.ACM_EpisodeCulprits(RunID);
    CREATE INDEX IX_EpisodeCulprits_EpisodeID ON dbo.ACM_EpisodeCulprits(EpisodeID);
    PRINT 'Created table: ACM_EpisodeCulprits';
END
GO

-- ACM_EpisodeDiagnostics
IF OBJECT_ID('dbo.ACM_EpisodeDiagnostics','U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_EpisodeDiagnostics (
        ID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        episode_id INT NULL,
        peak_z FLOAT NULL,
        peak_timestamp DATETIME2(3) NULL,
        duration_h FLOAT NULL,
        dominant_sensor NVARCHAR(200) NULL,
        severity NVARCHAR(50) NULL,
        severity_reason NVARCHAR(500) NULL,
        avg_z FLOAT NULL,
        min_health_index FLOAT NULL,
        CreatedAt DATETIME2(3) NOT NULL DEFAULT GETUTCDATE()
    );
    CREATE INDEX IX_EpisodeDiagnostics_RunID ON dbo.ACM_EpisodeDiagnostics(RunID);
    PRINT 'Created table: ACM_EpisodeDiagnostics';
END
GO

-- ============================================================================
-- 3. Synonyms (from scripts/sql/create_acm_synonyms.sql)
-- ============================================================================

IF NOT EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_Scores_Long')
    CREATE SYNONYM dbo.ACM_Scores_Long FOR dbo.ScoresTS;
    
IF NOT EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_Drift_TS')
    CREATE SYNONYM dbo.ACM_Drift_TS FOR dbo.DriftTS;
    
IF NOT EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_Anomaly_Events')
    CREATE SYNONYM dbo.ACM_Anomaly_Events FOR dbo.AnomalyEvents;
    
IF NOT EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_Regime_Episodes')
    CREATE SYNONYM dbo.ACM_Regime_Episodes FOR dbo.RegimeEpisodes;
    
IF NOT EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_PCA_Models')
    CREATE SYNONYM dbo.ACM_PCA_Models FOR dbo.PCA_Model;
    
IF NOT EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_PCA_Loadings')
    CREATE SYNONYM dbo.ACM_PCA_Loadings FOR dbo.PCA_Components;
    
IF NOT EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_PCA_Metrics')
    CREATE SYNONYM dbo.ACM_PCA_Metrics FOR dbo.PCA_Metrics;
    
IF NOT EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_Run_Stats')
    CREATE SYNONYM dbo.ACM_Run_Stats FOR dbo.RunStats;

PRINT 'Synonyms verified/created.';
GO

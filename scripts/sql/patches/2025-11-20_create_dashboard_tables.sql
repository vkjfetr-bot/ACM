USE [ACM];
GO

-- Dashboard artifacts now persisted in SQL to supplant CSV exports.
-- This patch creates tables for analytics that previously only lived as CSV files.

IF OBJECT_ID('dbo.ACM_OMRContributionsLong', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_OMRContributionsLong (
        RunID uniqueidentifier NOT NULL,
        EquipID int NOT NULL,
        Timestamp datetime2(3) NOT NULL,
        SensorName nvarchar(128) NOT NULL,
        ContributionScore float NOT NULL,
        ContributionPct float NOT NULL,
        OMR_Z float NULL,
        CreatedAt datetime2(3) NOT NULL CONSTRAINT DF_ACM_OMRContributionsLong_CreatedAt DEFAULT (SYSUTCDATETIME())
    );
    CREATE CLUSTERED INDEX PK_ACM_OMRContributionsLong ON dbo.ACM_OMRContributionsLong(RunID, EquipID, Timestamp, SensorName);
END
GO

IF OBJECT_ID('dbo.ACM_FusionQualityReport', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_FusionQualityReport (
        RunID uniqueidentifier NOT NULL,
        EquipID int NOT NULL,
        Detector nvarchar(64) NOT NULL,
        Weight float NOT NULL,
        Present bit NOT NULL,
        MeanZ float NULL,
        MaxZ float NULL,
        Points int NULL,
        CreatedAt datetime2(3) NOT NULL CONSTRAINT DF_ACM_FusionQualityReport_CreatedAt DEFAULT (SYSUTCDATETIME())
    );
    CREATE CLUSTERED INDEX PK_ACM_FusionQualityReport ON dbo.ACM_FusionQualityReport(RunID, EquipID, Detector);
END
GO

IF OBJECT_ID('dbo.ACM_OMRTimeline', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_OMRTimeline (
        RunID uniqueidentifier NOT NULL,
        EquipID int NOT NULL,
        Timestamp datetime2(3) NOT NULL,
        OMR_Z float NULL,
        OMR_Weight float NULL,
        CreatedAt datetime2(3) NOT NULL CONSTRAINT DF_ACM_OMRTimeline_CreatedAt DEFAULT (SYSUTCDATETIME())
    );
    CREATE CLUSTERED INDEX PK_ACM_OMRTimeline ON dbo.ACM_OMRTimeline(RunID, EquipID, Timestamp);
END
GO

IF OBJECT_ID('dbo.ACM_RegimeStats', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_RegimeStats (
        RunID uniqueidentifier NOT NULL,
        EquipID int NOT NULL,
        RegimeLabel int NOT NULL,
        OccupancyPct float NULL,
        AvgDwellSeconds float NULL,
        FusedMean float NULL,
        FusedP90 float NULL,
        CreatedAt datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeStats_CreatedAt DEFAULT (SYSUTCDATETIME())
    );
    CREATE CLUSTERED INDEX PK_ACM_RegimeStats ON dbo.ACM_RegimeStats(RunID, EquipID, RegimeLabel);
END
GO

IF OBJECT_ID('dbo.ACM_DailyFusedProfile', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_DailyFusedProfile (
        RunID uniqueidentifier NOT NULL,
        EquipID int NOT NULL,
        DayOfWeek int NOT NULL,
        Hour int NOT NULL,
        FusedMean float NULL,
        FusedP90 float NULL,
        FusedP95 float NULL,
        RecordCount int NULL,
        CreatedAt datetime2(3) NOT NULL CONSTRAINT DF_ACM_DailyFusedProfile_CreatedAt DEFAULT (SYSUTCDATETIME())
    );
    CREATE CLUSTERED INDEX PK_ACM_DailyFusedProfile ON dbo.ACM_DailyFusedProfile(RunID, EquipID, DayOfWeek, Hour);
END
GO

IF OBJECT_ID('dbo.ACM_HealthDistributionOverTime', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_HealthDistributionOverTime (
        RunID uniqueidentifier NOT NULL,
        EquipID int NOT NULL,
        BucketStart datetime2(3) NOT NULL,
        BucketSeconds int NOT NULL,
        FusedP50 float NULL,
        FusedP75 float NULL,
        FusedP90 float NULL,
        FusedP95 float NULL,
        HealthP50 float NULL,
        HealthP10 float NULL,
        BucketCount int NULL,
        CreatedAt datetime2(3) NOT NULL CONSTRAINT DF_ACM_HealthDistributionOverTime_CreatedAt DEFAULT (SYSUTCDATETIME())
    );
    CREATE CLUSTERED INDEX PK_ACM_HealthDistributionOverTime ON dbo.ACM_HealthDistributionOverTime(RunID, EquipID, BucketStart);
END
GO

IF OBJECT_ID('dbo.ACM_SensorNormalized_TS', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_SensorNormalized_TS (
        RunID uniqueidentifier NOT NULL,
        EquipID int NOT NULL,
        Timestamp datetime2(3) NOT NULL,
        SensorName nvarchar(128) NOT NULL,
        NormValue float NULL,
        ZScore float NULL,
        AnomalyLevel nvarchar(16) NOT NULL,
        EpisodeActive bit NOT NULL,
        CreatedAt datetime2(3) NOT NULL CONSTRAINT DF_ACM_SensorNormalized_TS_CreatedAt DEFAULT (SYSUTCDATETIME())
    );
    CREATE CLUSTERED INDEX PK_ACM_SensorNormalized_TS ON dbo.ACM_SensorNormalized_TS(RunID, EquipID, Timestamp, SensorName);
END
GO

IF OBJECT_ID('dbo.ACM_ChartGenerationLog', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_ChartGenerationLog (
        RunID uniqueidentifier NOT NULL,
        EquipID int NOT NULL,
        ChartName nvarchar(256) NOT NULL,
        Status nvarchar(32) NOT NULL,
        Reason nvarchar(256) NULL,
        DurationSeconds float NULL,
        Timestamp datetime2(3) NULL,
        CreatedAt datetime2(3) NOT NULL CONSTRAINT DF_ACM_ChartGenerationLog_CreatedAt DEFAULT (SYSUTCDATETIME())
    );
    CREATE CLUSTERED INDEX PK_ACM_ChartGenerationLog ON dbo.ACM_ChartGenerationLog(RunID, EquipID, ChartName, Timestamp);
END
GO

/*
Create Missing Tables for CSV Counterparts
==========================================
Creates tables that were identified as missing but have CSV counterparts.
*/

USE [ACM];
GO

-- ACM_OMRContributions
IF OBJECT_ID('dbo.ACM_OMRContributions', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_OMRContributions (
        ID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        Timestamp DATETIME2(3) NOT NULL,
        SensorName NVARCHAR(200) NOT NULL,
        Contribution FLOAT NULL,
        CreatedAt DATETIME2(3) DEFAULT GETUTCDATE()
    );
    CREATE INDEX IX_OMRContrib_RunID ON dbo.ACM_OMRContributions(RunID);
    CREATE INDEX IX_OMRContrib_Equip_Time ON dbo.ACM_OMRContributions(EquipID, Timestamp);
    PRINT 'Created table: ACM_OMRContributions';
END
GO

-- ACM_FusionQuality
IF OBJECT_ID('dbo.ACM_FusionQuality', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_FusionQuality (
        ID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        Timestamp DATETIME2(3) NOT NULL,
        MetricName NVARCHAR(100) NOT NULL,
        Value FLOAT NULL,
        CreatedAt DATETIME2(3) DEFAULT GETUTCDATE()
    );
    CREATE INDEX IX_FusionQuality_RunID ON dbo.ACM_FusionQuality(RunID);
    PRINT 'Created table: ACM_FusionQuality';
END
GO

-- ACM_DailyFusedProfile
IF OBJECT_ID('dbo.ACM_DailyFusedProfile', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_DailyFusedProfile (
        ID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        ProfileDate DATE NOT NULL,
        AvgFusedScore FLOAT NULL,
        MaxFusedScore FLOAT NULL,
        MinFusedScore FLOAT NULL,
        SampleCount INT NULL,
        CreatedAt DATETIME2(3) DEFAULT GETUTCDATE()
    );
    CREATE INDEX IX_DailyFused_RunID ON dbo.ACM_DailyFusedProfile(RunID);
    CREATE INDEX IX_DailyFused_Equip_Date ON dbo.ACM_DailyFusedProfile(EquipID, ProfileDate);
    PRINT 'Created table: ACM_DailyFusedProfile';
END
GO

-- ACM_FusionMetrics
IF OBJECT_ID('dbo.ACM_FusionMetrics', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_FusionMetrics (
        ID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        Metric NVARCHAR(100) NOT NULL,
        Value FLOAT NULL,
        Context NVARCHAR(MAX) NULL,
        CreatedAt DATETIME2(3) DEFAULT GETUTCDATE()
    );
    CREATE INDEX IX_FusionMetrics_RunID ON dbo.ACM_FusionMetrics(RunID);
    PRINT 'Created table: ACM_FusionMetrics';
END
GO

-- ACM_ChartGenerationLog
IF OBJECT_ID('dbo.ACM_ChartGenerationLog', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_ChartGenerationLog (
        ID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        ChartName NVARCHAR(200) NOT NULL,
        GeneratedAt DATETIME2(3) NOT NULL,
        Status NVARCHAR(50) NOT NULL,
        FilePath NVARCHAR(500) NULL,
        ErrorMessage NVARCHAR(MAX) NULL,
        CreatedAt DATETIME2(3) DEFAULT GETUTCDATE()
    );
    CREATE INDEX IX_ChartLog_RunID ON dbo.ACM_ChartGenerationLog(RunID);
    PRINT 'Created table: ACM_ChartGenerationLog';
END
GO

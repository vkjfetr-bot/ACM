-- Create missing OMR and other critical tables for Grafana dashboard
-- Date: 2025-11-24
-- Purpose: Support new OMR metrics, gating, and top contributors features

USE ACM;
GO

-- =====================================================
-- ACM_OMR_Metrics: Quality metrics and gating status
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_OMR_Metrics')
BEGIN
    CREATE TABLE ACM_OMR_Metrics (
        MetricID INT IDENTITY(1,1) PRIMARY KEY,
        EquipID INT NOT NULL,
        RunID VARCHAR(100),
        MetricName VARCHAR(100) NOT NULL,
        MetricValue FLOAT NOT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        INDEX IX_ACM_OMR_Metrics_Equip (EquipID, CreatedAt DESC),
        INDEX IX_ACM_OMR_Metrics_Run (RunID)
    );
    PRINT 'Created ACM_OMR_Metrics table';
END
ELSE
    PRINT 'ACM_OMR_Metrics table already exists';
GO

-- =====================================================
-- ACM_OMR_TopContributors: Episode-level top sensors
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_OMR_TopContributors')
BEGIN
    CREATE TABLE ACM_OMR_TopContributors (
        ContribID INT IDENTITY(1,1) PRIMARY KEY,
        EquipID INT NOT NULL,
        RunID VARCHAR(100),
        EpisodeID INT,
        EpisodeStart DATETIME2,
        Rank INT NOT NULL,
        SensorName VARCHAR(200) NOT NULL,
        Contribution FLOAT NOT NULL,
        ContributionPct FLOAT NOT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        INDEX IX_ACM_OMR_TopContrib_Equip (EquipID, EpisodeStart DESC),
        INDEX IX_ACM_OMR_TopContrib_Episode (EpisodeID, Rank)
    );
    PRINT 'Created ACM_OMR_TopContributors table';
END
ELSE
    PRINT 'ACM_OMR_TopContributors table already exists';
GO

-- =====================================================
-- ACM_OMR_SensorContributions: Alias/view for ContributionsLong
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.views WHERE name = 'ACM_OMR_SensorContributions')
BEGIN
    EXEC('CREATE VIEW ACM_OMR_SensorContributions AS 
          SELECT * FROM ACM_OMRContributionsLong');
    PRINT 'Created ACM_OMR_SensorContributions view';
END
ELSE
    PRINT 'ACM_OMR_SensorContributions view already exists';
GO

-- =====================================================
-- ACM_DetectorContributions: Detector breakdown
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_DetectorContributions')
BEGIN
    CREATE TABLE ACM_DetectorContributions (
        ContribID INT IDENTITY(1,1) PRIMARY KEY,
        EquipID INT NOT NULL,
        RunID VARCHAR(100),
        Timestamp DATETIME2 NOT NULL,
        DetectorName VARCHAR(100) NOT NULL,
        ContributionPct FLOAT NOT NULL,
        ZScore FLOAT,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        INDEX IX_ACM_DetectorContrib_Time (EquipID, Timestamp DESC),
        INDEX IX_ACM_DetectorContrib_Run (RunID)
    );
    PRINT 'Created ACM_DetectorContributions table';
END
ELSE
    PRINT 'ACM_DetectorContributions table already exists';
GO

-- =====================================================
-- ACM_ForecastTimeline: Forecasting results timeline
-- Note: This may already exist as ACM_HealthForecast_TS or ACM_SensorForecast_TS
-- Creating a unified view
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.views WHERE name = 'ACM_ForecastTimeline')
BEGIN
    EXEC('CREATE VIEW ACM_ForecastTimeline AS 
          SELECT 
              Timestamp,
              EquipID,
              ''Health'' as ForecastType,
              NULL as SensorName,
              ForecastValue as Value,
              ConfLower,
              ConfUpper,
              HorizonHours
          FROM ACM_HealthForecast_TS
          WHERE ACM_HealthForecast_TS.Timestamp IS NOT NULL');
    PRINT 'Created ACM_ForecastTimeline view';
END
ELSE
    PRINT 'ACM_ForecastTimeline view already exists';
GO

PRINT '';
PRINT '=====================================================';
PRINT 'Missing tables creation complete!';
PRINT '=====================================================';
PRINT '';
PRINT 'Summary:';
PRINT '  - ACM_OMR_Metrics: Quality metrics and gating status';
PRINT '  - ACM_OMR_TopContributors: Episode-level top contributors';
PRINT '  - ACM_OMR_SensorContributions: Alias for ACM_OMRContributionsLong';
PRINT '  - ACM_DetectorContributions: Detector breakdown by time';
PRINT '  - ACM_ForecastTimeline: Unified forecast view';
PRINT '';
PRINT 'Next steps:';
PRINT '  1. Run ACM pipeline to populate tables';
PRINT '  2. Verify data in new tables';
PRINT '  3. Refresh Grafana dashboard';
GO

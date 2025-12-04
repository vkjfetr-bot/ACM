/*
 * Rollback Script: 60_rollback_to_v9.sql
 * Version: v10.0.0 → v9.0.0
 * Purpose: Restore v9 forecast table schema (rollback from v10)
 * 
 * WARNING: This drops all v10 forecast data and restores v9 table structure
 * Use only when rolling back v10.0.0 deployment
 * 
 * Tables Dropped (4):
 *   - ACM_HealthForecast
 *   - ACM_FailureForecast
 *   - ACM_RUL
 *   - ACM_ForecastingState
 * 
 * Tables Restored (from v9):
 *   - ACM_HealthForecast_TS
 *   - ACM_FailureForecast_TS
 *   - ACM_RUL_TS
 *   - ACM_RUL_Summary
 *   - ACM_RUL_Attribution
 *   - ACM_SensorForecast_TS
 *   - ACM_MaintenanceRecommendation
 *   - ACM_EnhancedFailureProbability_TS
 *   - ACM_FailureCausation
 *   - ACM_EnhancedMaintenanceRecommendation
 *   - ACM_RecommendedActions
 *   - ACM_HealthForecast_Continuous
 *   - ACM_FailureHazard_TS
 * 
 * Post-Rollback Steps:
 *   1. git checkout v9.0.0
 *   2. Deploy v9.0.0 code to application server
 *   3. Run ACM to regenerate forecast data
 * 
 * Estimated Time: <5 minutes
 * Author: ACM Development Team
 * Date: 2025-12-04
 */

USE ACM;
GO

PRINT '========================================';
PRINT 'ACM v10.0.0 → v9.0.0 Rollback';
PRINT 'Started: ' + CONVERT(VARCHAR, GETDATE(), 120);
PRINT '========================================';
GO

-- ============================================================================
-- STEP 1: Drop v10 tables
-- ============================================================================

PRINT '';
PRINT 'STEP 1: Dropping v10.0.0 forecast tables...';
GO

IF OBJECT_ID('dbo.ACM_HealthForecast', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_HealthForecast;
    PRINT '  ✓ Dropped ACM_HealthForecast';
END
GO

IF OBJECT_ID('dbo.ACM_FailureForecast', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_FailureForecast;
    PRINT '  ✓ Dropped ACM_FailureForecast';
END
GO

IF OBJECT_ID('dbo.ACM_RUL', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_RUL;
    PRINT '  ✓ Dropped ACM_RUL';
END
GO

IF OBJECT_ID('dbo.ACM_ForecastingState', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_ForecastingState;
    PRINT '  ✓ Dropped ACM_ForecastingState';
END
GO

PRINT '';
PRINT 'v10 tables dropped successfully.';
GO

-- ============================================================================
-- STEP 2: Restore v9 forecast tables (from 57_create_forecast_and_rul_tables.sql)
-- ============================================================================

PRINT '';
PRINT 'STEP 2: Restoring v9.0.0 forecast tables...';
GO

-- ACM_HealthForecast_TS
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_HealthForecast_TS')
BEGIN
    CREATE TABLE dbo.ACM_HealthForecast_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        Timestamp       DATETIME2 NOT NULL,
        ForecastHealth  FLOAT NULL,
        CiLower         FLOAT NULL,
        CiUpper         FLOAT NULL,
        ForecastStd     FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,
        CreatedAt       DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_HealthForecast_TS PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp)
    );
    PRINT '  ✓ Restored ACM_HealthForecast_TS';
END
GO

-- ACM_FailureForecast_TS
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_FailureForecast_TS')
BEGIN
    CREATE TABLE dbo.ACM_FailureForecast_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        Timestamp       DATETIME2 NOT NULL,
        FailureProb     FLOAT NOT NULL,
        ThresholdUsed   FLOAT NOT NULL,
        Method          NVARCHAR(50) NOT NULL,
        CreatedAt       DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_FailureForecast_TS PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp)
    );
    PRINT '  ✓ Restored ACM_FailureForecast_TS';
END
GO

-- ACM_RUL_TS
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RUL_TS')
BEGIN
    CREATE TABLE dbo.ACM_RUL_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        Timestamp       DATETIME2 NOT NULL,
        RUL_Hours       FLOAT NOT NULL,
        LowerBound      FLOAT NULL,
        UpperBound      FLOAT NULL,
        Confidence      FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,
        CreatedAt       DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_RUL_TS PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp)
    );
    PRINT '  ✓ Restored ACM_RUL_TS';
END
GO

-- ACM_RUL_Summary
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RUL_Summary')
BEGIN
    CREATE TABLE dbo.ACM_RUL_Summary (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        RUL_Hours       FLOAT NOT NULL,
        LowerBound      FLOAT NULL,
        UpperBound      FLOAT NULL,
        Confidence      FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,
        LastUpdate      DATETIME2 NOT NULL,
        CreatedAt       DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_RUL_Summary PRIMARY KEY CLUSTERED (RunID, EquipID)
    );
    PRINT '  ✓ Restored ACM_RUL_Summary';
END
GO

-- ACM_RUL_Attribution
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RUL_Attribution')
BEGIN
    CREATE TABLE dbo.ACM_RUL_Attribution (
        RunID               UNIQUEIDENTIFIER NOT NULL,
        EquipID             INT NOT NULL,
        FailureTime         DATETIME2 NOT NULL,
        SensorName          NVARCHAR(255) NOT NULL,
        FailureContribution FLOAT NOT NULL,
        ZScoreAtFailure     FLOAT NULL,
        AlertCount          INT NULL,
        Comment             NVARCHAR(400) NULL,
        CreatedAt           DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_RUL_Attribution PRIMARY KEY CLUSTERED (RunID, EquipID, FailureTime, SensorName)
    );
    PRINT '  ✓ Restored ACM_RUL_Attribution';
END
GO

-- ACM_SensorForecast_TS
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_SensorForecast_TS')
BEGIN
    CREATE TABLE dbo.ACM_SensorForecast_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        SensorName      NVARCHAR(255) NOT NULL,
        Timestamp       DATETIME2 NOT NULL,
        ForecastValue   FLOAT NOT NULL,
        CiLower         FLOAT NULL,
        CiUpper         FLOAT NULL,
        ForecastStd     FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,
        CreatedAt       DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_SensorForecast_TS PRIMARY KEY CLUSTERED (RunID, EquipID, SensorName, Timestamp)
    );
    PRINT '  ✓ Restored ACM_SensorForecast_TS';
END
GO

-- ACM_MaintenanceRecommendation
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_MaintenanceRecommendation')
BEGIN
    CREATE TABLE dbo.ACM_MaintenanceRecommendation (
        RunID                 UNIQUEIDENTIFIER NOT NULL,
        EquipID               INT NOT NULL,
        EarliestMaintenance   DATETIME2 NOT NULL,
        PreferredWindowStart  DATETIME2 NOT NULL,
        PreferredWindowEnd    DATETIME2 NOT NULL,
        FailureProbAtWindowEnd FLOAT NOT NULL,
        Comment               NVARCHAR(400) NULL,
        CreatedAt             DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_MaintenanceRecommendation PRIMARY KEY CLUSTERED (RunID, EquipID)
    );
    PRINT '  ✓ Restored ACM_MaintenanceRecommendation';
END
GO

-- ACM_EnhancedFailureProbability_TS
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_EnhancedFailureProbability_TS')
BEGIN
    CREATE TABLE dbo.ACM_EnhancedFailureProbability_TS (
        RunID                  UNIQUEIDENTIFIER NOT NULL,
        EquipID                INT NOT NULL,
        Timestamp              DATETIME2 NOT NULL,
        ForecastHorizon_Hours  FLOAT NOT NULL,
        ForecastHealth         FLOAT NULL,
        ForecastUncertainty    FLOAT NULL,
        FailureProbability     FLOAT NOT NULL,
        RiskLevel              NVARCHAR(50) NOT NULL,
        Confidence             FLOAT NULL,
        Model                  NVARCHAR(50) NULL,
        CreatedAt              DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_EnhancedFailureProbability_TS PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp, ForecastHorizon_Hours)
    );
    PRINT '  ✓ Restored ACM_EnhancedFailureProbability_TS';
END
GO

-- ACM_FailureCausation
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_FailureCausation')
BEGIN
    CREATE TABLE dbo.ACM_FailureCausation (
        RunID               UNIQUEIDENTIFIER NOT NULL,
        EquipID             INT NOT NULL,
        PredictedFailureTime DATETIME2 NOT NULL,
        FailurePattern      NVARCHAR(200) NULL,
        Detector            NVARCHAR(100) NOT NULL,
        MeanZ               FLOAT NULL,
        MaxZ                FLOAT NULL,
        SpikeCount          INT NULL,
        TrendSlope          FLOAT NULL,
        ContributionWeight  FLOAT NULL,
        ContributionPct     FLOAT NULL,
        CreatedAt           DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_FailureCausation PRIMARY KEY CLUSTERED (RunID, EquipID, Detector)
    );
    PRINT '  ✓ Restored ACM_FailureCausation';
END
GO

-- ACM_EnhancedMaintenanceRecommendation
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_EnhancedMaintenanceRecommendation')
BEGIN
    CREATE TABLE dbo.ACM_EnhancedMaintenanceRecommendation (
        RunID                   UNIQUEIDENTIFIER NOT NULL,
        EquipID                 INT NOT NULL,
        UrgencyScore            FLOAT NOT NULL,
        MaintenanceRequired     BIT NOT NULL,
        EarliestMaintenance     FLOAT NULL,
        PreferredWindowStart    FLOAT NULL,
        PreferredWindowEnd      FLOAT NULL,
        LatestSafeTime          FLOAT NULL,
        FailureProbAtLatest     FLOAT NULL,
        FailurePattern          NVARCHAR(200) NULL,
        Confidence              FLOAT NULL,
        EstimatedDuration_Hours FLOAT NULL,
        CreatedAt               DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_EnhancedMaintenanceRecommendation PRIMARY KEY CLUSTERED (RunID, EquipID)
    );
    PRINT '  ✓ Restored ACM_EnhancedMaintenanceRecommendation';
END
GO

-- ACM_RecommendedActions (incomplete in v9, create minimal structure)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RecommendedActions')
BEGIN
    CREATE TABLE dbo.ACM_RecommendedActions (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        ActionID        INT IDENTITY(1,1) NOT NULL,
        ActionType      NVARCHAR(100) NOT NULL,
        Description     NVARCHAR(500) NULL,
        Priority        NVARCHAR(50) NULL,
        CreatedAt       DATETIME2 NOT NULL DEFAULT GETDATE(),
        CONSTRAINT PK_ACM_RecommendedActions PRIMARY KEY CLUSTERED (ActionID)
    );
    PRINT '  ✓ Restored ACM_RecommendedActions';
END
GO

-- ============================================================================
-- STEP 3: Restore v9 continuous forecast tables (from create_continuous_forecast_tables.sql)
-- ============================================================================

PRINT '';
PRINT 'STEP 3: Restoring v9 continuous forecast tables...';
GO

-- ACM_ForecastState (v9 version without RowVersion)
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_ForecastState')
BEGIN
    CREATE TABLE dbo.ACM_ForecastState (
        EquipID INT NOT NULL,
        StateVersion INT NOT NULL,
        ModelType NVARCHAR(50),
        ModelParamsJson NVARCHAR(MAX),
        ResidualVariance FLOAT,
        LastForecastHorizonJson NVARCHAR(MAX),
        HazardBaseline FLOAT,
        LastRetrainTime DATETIME2,
        TrainingDataHash NVARCHAR(64),
        TrainingWindowHours INT,
        ForecastQualityJson NVARCHAR(MAX),
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT PK_ForecastState PRIMARY KEY (EquipID, StateVersion)
    );
    
    CREATE INDEX IX_ForecastState_Latest ON dbo.ACM_ForecastState(EquipID, StateVersion DESC);
    PRINT '  ✓ Restored ACM_ForecastState (v9)';
END
GO

-- ACM_HealthForecast_Continuous
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_HealthForecast_Continuous')
BEGIN
    CREATE TABLE dbo.ACM_HealthForecast_Continuous (
        Timestamp DATETIME2 NOT NULL,
        ForecastHealth FLOAT NOT NULL,
        CI_Lower FLOAT,
        CI_Upper FLOAT,
        SourceRunID NVARCHAR(50) NOT NULL,
        MergeWeight FLOAT,
        EquipID INT NOT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT PK_HealthForecast_Continuous PRIMARY KEY (EquipID, Timestamp, SourceRunID)
    );
    
    CREATE INDEX IX_HealthForecast_TimeRange ON dbo.ACM_HealthForecast_Continuous(EquipID, Timestamp);
    PRINT '  ✓ Restored ACM_HealthForecast_Continuous';
END
GO

-- ACM_FailureHazard_TS
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_FailureHazard_TS')
BEGIN
    CREATE TABLE dbo.ACM_FailureHazard_TS (
        Timestamp DATETIME2 NOT NULL,
        HazardRaw FLOAT,
        HazardSmooth FLOAT,
        Survival FLOAT,
        FailureProb FLOAT,
        RunID NVARCHAR(50),
        EquipID INT,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT PK_FailureHazard_TS PRIMARY KEY (EquipID, RunID, Timestamp)
    );
    
    CREATE INDEX IX_FailureHazard_Time ON dbo.ACM_FailureHazard_TS(EquipID, Timestamp);
    PRINT '  ✓ Restored ACM_FailureHazard_TS';
END
GO

PRINT '';
PRINT 'v9 tables restored successfully.';
GO

-- ============================================================================
-- STEP 4: Verify rollback
-- ============================================================================

PRINT '';
PRINT 'STEP 4: Verifying rollback...';
GO

DECLARE @V10TablesCount INT = 0;
DECLARE @V9TablesCount INT = 0;

-- Count v10 tables (should be 0)
SELECT @V10TablesCount = COUNT(*)
FROM sys.tables
WHERE name IN ('ACM_HealthForecast', 'ACM_FailureForecast', 'ACM_RUL', 'ACM_ForecastingState');

-- Count v9 tables (should be 13)
SELECT @V9TablesCount = COUNT(*)
FROM sys.tables
WHERE name IN (
    'ACM_HealthForecast_TS', 'ACM_FailureForecast_TS', 'ACM_RUL_TS',
    'ACM_RUL_Summary', 'ACM_RUL_Attribution', 'ACM_SensorForecast_TS',
    'ACM_MaintenanceRecommendation', 'ACM_EnhancedFailureProbability_TS',
    'ACM_FailureCausation', 'ACM_EnhancedMaintenanceRecommendation',
    'ACM_RecommendedActions', 'ACM_HealthForecast_Continuous', 'ACM_FailureHazard_TS'
);

PRINT '  v10 tables remaining: ' + CAST(@V10TablesCount AS VARCHAR);
PRINT '  v9 tables restored: ' + CAST(@V9TablesCount AS VARCHAR);

IF @V10TablesCount = 0 AND @V9TablesCount = 13
BEGIN
    PRINT '  ✓ Rollback verification PASSED';
END
ELSE
BEGIN
    PRINT '  ✗ Rollback verification FAILED';
    PRINT '  Expected: 0 v10 tables, 13 v9 tables';
    PRINT '  Actual: ' + CAST(@V10TablesCount AS VARCHAR) + ' v10, ' + CAST(@V9TablesCount AS VARCHAR) + ' v9';
END
GO

-- ============================================================================
-- STEP 5: Summary
-- ============================================================================

PRINT '';
PRINT '========================================';
PRINT 'Rollback Complete: v10.0.0 → v9.0.0';
PRINT 'Completed: ' + CONVERT(VARCHAR, GETDATE(), 120);
PRINT '========================================';
PRINT '';
PRINT 'NEXT STEPS:';
PRINT '  1. Checkout v9: git checkout v9.0.0';
PRINT '  2. Deploy v9.0.0 code to application server';
PRINT '  3. Regenerate forecasts: python scripts/sql_batch_runner.py --equip FD_FAN --max-batches 10';
PRINT '';
PRINT 'WARNING: All v10 forecast data has been dropped';
PRINT 'Equipment data (FD_FAN_Data, GAS_TURBINE_Data) preserved';
PRINT '';
GO

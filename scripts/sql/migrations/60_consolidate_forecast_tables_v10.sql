/*
 * Migration Script: 60_consolidate_forecast_tables_v10.sql
 * Version: v10.0.0
 * Purpose: Consolidate 12 legacy forecast tables into 4 streamlined tables
 * 
 * BREAKING CHANGE: This migration drops all legacy forecast data
 * Data Preservation: Only FD_FAN and GAS_TURBINE equipment data tables preserved
 * 
 * Tables Dropped (12):
 *   - ACM_HealthForecast_TS
 *   - ACM_FailureForecast_TS
 *   - ACM_RUL_TS
 *   - ACM_RUL_Summary
 *   - ACM_SensorForecast_TS
 *   - ACM_MaintenanceRecommendation
 *   - ACM_EnhancedFailureProbability_TS
 *   - ACM_FailureCausation
 *   - ACM_EnhancedMaintenanceRecommendation
 *   - ACM_RecommendedActions
 *   - ACM_HealthForecast_Continuous
 *   - ACM_FailureHazard_TS
 * 
 * Tables Created (4):
 *   - ACM_HealthForecast
 *   - ACM_FailureForecast
 *   - ACM_RUL
 *   - ACM_ForecastingState
 * 
 * Rollback: Run 60_rollback_to_v9.sql to restore v9 schema
 * Estimated Time: <5 minutes
 * Author: ACM Development Team
 * Date: 2025-12-04
 */

USE ACM;
GO

PRINT '========================================';
PRINT 'ACM v10.0.0 Forecast Table Consolidation';
PRINT 'Started: ' + CONVERT(VARCHAR, GETDATE(), 120);
PRINT '========================================';
GO

-- ============================================================================
-- STEP 1: Drop legacy forecast tables (12 tables)
-- ============================================================================

PRINT '';
PRINT 'STEP 1: Dropping 12 legacy forecast tables...';
GO

IF OBJECT_ID('dbo.ACM_HealthForecast_TS', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_HealthForecast_TS;
    PRINT '  ✓ Dropped ACM_HealthForecast_TS';
END
GO

IF OBJECT_ID('dbo.ACM_FailureForecast_TS', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_FailureForecast_TS;
    PRINT '  ✓ Dropped ACM_FailureForecast_TS';
END
GO

IF OBJECT_ID('dbo.ACM_RUL_TS', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_RUL_TS;
    PRINT '  ✓ Dropped ACM_RUL_TS';
END
GO

IF OBJECT_ID('dbo.ACM_RUL_Summary', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_RUL_Summary;
    PRINT '  ✓ Dropped ACM_RUL_Summary';
END
GO

IF OBJECT_ID('dbo.ACM_SensorForecast_TS', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_SensorForecast_TS;
    PRINT '  ✓ Dropped ACM_SensorForecast_TS';
END
GO

IF OBJECT_ID('dbo.ACM_MaintenanceRecommendation', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_MaintenanceRecommendation;
    PRINT '  ✓ Dropped ACM_MaintenanceRecommendation';
END
GO

IF OBJECT_ID('dbo.ACM_EnhancedFailureProbability_TS', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_EnhancedFailureProbability_TS;
    PRINT '  ✓ Dropped ACM_EnhancedFailureProbability_TS';
END
GO

IF OBJECT_ID('dbo.ACM_FailureCausation', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_FailureCausation;
    PRINT '  ✓ Dropped ACM_FailureCausation';
END
GO

IF OBJECT_ID('dbo.ACM_EnhancedMaintenanceRecommendation', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_EnhancedMaintenanceRecommendation;
    PRINT '  ✓ Dropped ACM_EnhancedMaintenanceRecommendation';
END
GO

IF OBJECT_ID('dbo.ACM_RecommendedActions', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_RecommendedActions;
    PRINT '  ✓ Dropped ACM_RecommendedActions';
END
GO

IF OBJECT_ID('dbo.ACM_HealthForecast_Continuous', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_HealthForecast_Continuous;
    PRINT '  ✓ Dropped ACM_HealthForecast_Continuous';
END
GO

IF OBJECT_ID('dbo.ACM_FailureHazard_TS', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_FailureHazard_TS;
    PRINT '  ✓ Dropped ACM_FailureHazard_TS';
END
GO

-- Also drop ACM_RUL_Attribution if exists (part of legacy)
IF OBJECT_ID('dbo.ACM_RUL_Attribution', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.ACM_RUL_Attribution;
    PRINT '  ✓ Dropped ACM_RUL_Attribution';
END
GO

PRINT '';
PRINT 'Legacy tables dropped successfully.';
GO

-- ============================================================================
-- STEP 2: Create new consolidated tables (4 tables)
-- ============================================================================

PRINT '';
PRINT 'STEP 2: Creating 4 new consolidated forecast tables...';
GO

-- ACM_HealthForecast: Time-series health predictions
IF OBJECT_ID('dbo.ACM_HealthForecast', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_HealthForecast (
        EquipID             INT NOT NULL,
        RunID               UNIQUEIDENTIFIER NOT NULL,
        Timestamp           DATETIME2 NOT NULL,
        ForecastHealth      FLOAT NOT NULL,
        CiLower             FLOAT NULL,
        CiUpper             FLOAT NULL,
        ForecastStd         FLOAT NULL,
        Method              NVARCHAR(50) NOT NULL DEFAULT 'LinearTrend',
        CreatedAt           DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        CONSTRAINT PK_ACM_HealthForecast PRIMARY KEY CLUSTERED (EquipID, RunID, Timestamp)
    );
    
    -- Index for time-range queries
    CREATE NONCLUSTERED INDEX IX_ACM_HealthForecast_Time 
        ON dbo.ACM_HealthForecast(EquipID, Timestamp)
        INCLUDE (ForecastHealth, CiLower, CiUpper);
    
    -- Index for run-based queries
    CREATE NONCLUSTERED INDEX IX_ACM_HealthForecast_Run
        ON dbo.ACM_HealthForecast(RunID)
        INCLUDE (EquipID, Timestamp, ForecastHealth);
    
    PRINT '  ✓ Created ACM_HealthForecast with indexes';
END
GO

-- ACM_FailureForecast: Time-series failure probability predictions
IF OBJECT_ID('dbo.ACM_FailureForecast', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_FailureForecast (
        EquipID             INT NOT NULL,
        RunID               UNIQUEIDENTIFIER NOT NULL,
        Timestamp           DATETIME2 NOT NULL,
        FailureProb         FLOAT NOT NULL,
        SurvivalProb        FLOAT NULL,
        HazardRate          FLOAT NULL,
        ThresholdUsed       FLOAT NOT NULL,
        Method              NVARCHAR(50) NOT NULL DEFAULT 'GaussianCDF',
        CreatedAt           DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        CONSTRAINT PK_ACM_FailureForecast PRIMARY KEY CLUSTERED (EquipID, RunID, Timestamp)
    );
    
    -- Index for time-range queries
    CREATE NONCLUSTERED INDEX IX_ACM_FailureForecast_Time
        ON dbo.ACM_FailureForecast(EquipID, Timestamp)
        INCLUDE (FailureProb, SurvivalProb, HazardRate);
    
    -- Index for high-risk queries (FailureProb > 0.5)
    CREATE NONCLUSTERED INDEX IX_ACM_FailureForecast_HighRisk
        ON dbo.ACM_FailureForecast(EquipID, FailureProb)
        WHERE FailureProb > 0.5;
    
    PRINT '  ✓ Created ACM_FailureForecast with indexes';
END
GO

-- ACM_RUL: Remaining Useful Life summary per run
IF OBJECT_ID('dbo.ACM_RUL', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_RUL (
        EquipID             INT NOT NULL,
        RunID               UNIQUEIDENTIFIER NOT NULL,
        RUL_Hours           FLOAT NOT NULL,
        P10_LowerBound      FLOAT NULL,
        P50_Median          FLOAT NULL,
        P90_UpperBound      FLOAT NULL,
        Confidence          FLOAT NULL,
        FailureTime         DATETIME2 NULL,
        Method              NVARCHAR(50) NOT NULL DEFAULT 'MonteCarlo',
        NumSimulations      INT NULL,
        TopSensor1          NVARCHAR(255) NULL,
        TopSensor2          NVARCHAR(255) NULL,
        TopSensor3          NVARCHAR(255) NULL,
        CreatedAt           DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        CONSTRAINT PK_ACM_RUL PRIMARY KEY CLUSTERED (EquipID, RunID)
    );
    
    -- Index for latest RUL queries
    CREATE NONCLUSTERED INDEX IX_ACM_RUL_Latest
        ON dbo.ACM_RUL(EquipID, CreatedAt DESC)
        INCLUDE (RUL_Hours, Confidence, FailureTime);
    
    -- Index for low-RUL alerts (RUL < 168 hours = 7 days)
    CREATE NONCLUSTERED INDEX IX_ACM_RUL_LowRUL
        ON dbo.ACM_RUL(EquipID, RUL_Hours)
        WHERE RUL_Hours < 168;
    
    PRINT '  ✓ Created ACM_RUL with indexes';
END
GO

-- ACM_ForecastingState: Persistent model state with optimistic locking
IF OBJECT_ID('dbo.ACM_ForecastingState', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_ForecastingState (
        EquipID                 INT NOT NULL,
        StateVersion            INT NOT NULL,
        ModelCoefficientsJson   NVARCHAR(MAX) NULL,
        LastForecastJson        NVARCHAR(MAX) NULL,
        LastRetrainTime         DATETIME2 NULL,
        TrainingDataHash        NVARCHAR(64) NULL,
        DataVolumeAnalyzed      BIGINT NULL,
        RecentMAE               FLOAT NULL,
        RecentRMSE              FLOAT NULL,
        RetriggerReason         NVARCHAR(200) NULL,
        RowVersion              ROWVERSION NOT NULL,
        CreatedAt               DATETIME2 NOT NULL DEFAULT GETDATE(),
        UpdatedAt               DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        CONSTRAINT PK_ACM_ForecastingState PRIMARY KEY CLUSTERED (EquipID, StateVersion)
    );
    
    -- Index for latest state queries (critical for forecast engine startup)
    CREATE NONCLUSTERED INDEX IX_ACM_ForecastingState_Latest
        ON dbo.ACM_ForecastingState(EquipID, StateVersion DESC)
        INCLUDE (ModelCoefficientsJson, LastForecastJson, LastRetrainTime);
    
    -- Unique index per equipment for optimistic locking
    CREATE UNIQUE NONCLUSTERED INDEX UQ_ACM_ForecastingState_Equip
        ON dbo.ACM_ForecastingState(EquipID, RowVersion);
    
    PRINT '  ✓ Created ACM_ForecastingState with optimistic locking (ROWVERSION)';
END
GO

PRINT '';
PRINT 'New consolidated tables created successfully.';
GO

-- ============================================================================
-- STEP 3: Verify migration
-- ============================================================================

PRINT '';
PRINT 'STEP 3: Verifying migration...';
GO

DECLARE @OldTablesCount INT = 0;
DECLARE @NewTablesCount INT = 0;

-- Count legacy tables (should be 0)
SELECT @OldTablesCount = COUNT(*)
FROM sys.tables
WHERE name IN (
    'ACM_HealthForecast_TS', 'ACM_FailureForecast_TS', 'ACM_RUL_TS',
    'ACM_RUL_Summary', 'ACM_SensorForecast_TS', 'ACM_MaintenanceRecommendation',
    'ACM_EnhancedFailureProbability_TS', 'ACM_FailureCausation',
    'ACM_EnhancedMaintenanceRecommendation', 'ACM_RecommendedActions',
    'ACM_HealthForecast_Continuous', 'ACM_FailureHazard_TS', 'ACM_RUL_Attribution'
);

-- Count new tables (should be 4)
SELECT @NewTablesCount = COUNT(*)
FROM sys.tables
WHERE name IN (
    'ACM_HealthForecast', 'ACM_FailureForecast', 'ACM_RUL', 'ACM_ForecastingState'
);

PRINT '  Legacy tables remaining: ' + CAST(@OldTablesCount AS VARCHAR);
PRINT '  New tables created: ' + CAST(@NewTablesCount AS VARCHAR);

IF @OldTablesCount = 0 AND @NewTablesCount = 4
BEGIN
    PRINT '  ✓ Migration verification PASSED';
END
ELSE
BEGIN
    PRINT '  ✗ Migration verification FAILED';
    PRINT '  Expected: 0 legacy tables, 4 new tables';
    PRINT '  Actual: ' + CAST(@OldTablesCount AS VARCHAR) + ' legacy, ' + CAST(@NewTablesCount AS VARCHAR) + ' new';
END
GO

-- ============================================================================
-- STEP 4: Verify equipment data preserved
-- ============================================================================

PRINT '';
PRINT 'STEP 4: Verifying equipment data preservation...';
GO

DECLARE @FD_FAN_Count BIGINT = 0;
DECLARE @GAS_TURBINE_Count BIGINT = 0;

IF OBJECT_ID('dbo.FD_FAN_Data', 'U') IS NOT NULL
BEGIN
    SELECT @FD_FAN_Count = COUNT(*) FROM dbo.FD_FAN_Data;
    PRINT '  ✓ FD_FAN_Data preserved: ' + CAST(@FD_FAN_Count AS VARCHAR) + ' rows';
END
ELSE
BEGIN
    PRINT '  ⚠ FD_FAN_Data table not found';
END

IF OBJECT_ID('dbo.GAS_TURBINE_Data', 'U') IS NOT NULL
BEGIN
    SELECT @GAS_TURBINE_Count = COUNT(*) FROM dbo.GAS_TURBINE_Data;
    PRINT '  ✓ GAS_TURBINE_Data preserved: ' + CAST(@GAS_TURBINE_Count AS VARCHAR) + ' rows';
END
ELSE
BEGIN
    PRINT '  ⚠ GAS_TURBINE_Data table not found';
END
GO

-- ============================================================================
-- STEP 5: Summary
-- ============================================================================

PRINT '';
PRINT '========================================';
PRINT 'Migration Complete: v10.0.0';
PRINT 'Completed: ' + CONVERT(VARCHAR, GETDATE(), 120);
PRINT '========================================';
PRINT '';
PRINT 'NEXT STEPS:';
PRINT '  1. Run: scripts/sql/migrations/61_adaptive_config_v10.sql';
PRINT '  2. Update code: deploy v10.0.0 to application server';
PRINT '  3. Test: python scripts/sql_batch_runner.py --equip FD_FAN --max-batches 10';
PRINT '';
PRINT 'ROLLBACK (if needed):';
PRINT '  1. Run: scripts/sql/migrations/60_rollback_to_v9.sql';
PRINT '  2. Checkout: git checkout v9.0.0';
PRINT '  3. Redeploy: v9.0.0 code to application server';
PRINT '';
GO

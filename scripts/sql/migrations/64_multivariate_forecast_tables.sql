-- Migration 64: Multivariate Forecasting Tables (v10.1.0)
-- Creates tables for VAR-based multivariate sensor forecasting
-- 
-- Tables created:
--   - ACM_SensorCorrelations: Pairwise sensor correlations with lead/lag relationships
--   - ACM_MultivariateForecast: Joint sensor forecasts from VAR model
--
-- Usage: sqlcmd -S "localhost\INSTANCE" -d ACM -E -i "scripts/sql/migrations/64_multivariate_forecast_tables.sql"

SET NOCOUNT ON;
GO

PRINT '=== Migration 64: Multivariate Forecasting Tables ==='
PRINT 'Starting migration at ' + CONVERT(VARCHAR, GETDATE(), 120);
GO

-- ============================================================================
-- Table: ACM_SensorCorrelations
-- Purpose: Store pairwise sensor correlations with lead/lag relationships
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ACM_SensorCorrelations')
BEGIN
    PRINT 'Creating ACM_SensorCorrelations table...';
    
    CREATE TABLE dbo.ACM_SensorCorrelations (
        CorrelationID       INT IDENTITY(1,1) PRIMARY KEY,
        EquipID             INT NOT NULL,
        RunID               UNIQUEIDENTIFIER NOT NULL,
        SensorA             NVARCHAR(128) NOT NULL,
        SensorB             NVARCHAR(128) NOT NULL,
        Correlation         FLOAT NOT NULL,          -- Pearson correlation coefficient (-1 to 1)
        OptimalLag          INT NULL,                -- Lag (hours) where max cross-correlation occurs
        GrangerPValue       FLOAT NULL,              -- Granger causality p-value (null if not computed)
        LeadSensor          NVARCHAR(128) NULL,      -- Which sensor leads (SensorA or SensorB)
        CreatedAt           DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        -- Index for efficient lookups
        INDEX IX_SensorCorrelations_EquipRun (EquipID, RunID),
        INDEX IX_SensorCorrelations_Sensors (SensorA, SensorB)
    );
    
    PRINT 'Created ACM_SensorCorrelations table successfully.';
END
ELSE
BEGIN
    PRINT 'ACM_SensorCorrelations table already exists - skipping creation.';
END
GO

-- ============================================================================
-- Table: ACM_MultivariateForecast
-- Purpose: Store joint sensor forecasts from VAR model
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ACM_MultivariateForecast')
BEGIN
    PRINT 'Creating ACM_MultivariateForecast table...';
    
    CREATE TABLE dbo.ACM_MultivariateForecast (
        ForecastID          INT IDENTITY(1,1) PRIMARY KEY,
        EquipID             INT NOT NULL,
        RunID               UNIQUEIDENTIFIER NOT NULL,
        SensorName          NVARCHAR(128) NOT NULL,
        ForecastTimestamp   DATETIME2 NOT NULL,      -- Future timestamp being forecasted
        ForecastValue       FLOAT NOT NULL,          -- Predicted Z-score value
        CiLower             FLOAT NULL,              -- 95% confidence interval lower bound
        CiUpper             FLOAT NULL,              -- 95% confidence interval upper bound
        ForecastStd         FLOAT NULL,              -- Forecast standard deviation
        Method              NVARCHAR(50) NOT NULL,   -- 'VAR', 'CorrelatedEWM', 'IndependentEWM'
        VarOrder            INT NULL,                -- VAR model lag order (p)
        CreatedAt           DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        -- Indexes for efficient queries
        INDEX IX_MultivariateForecast_EquipRun (EquipID, RunID),
        INDEX IX_MultivariateForecast_Sensor (SensorName),
        INDEX IX_MultivariateForecast_Timestamp (ForecastTimestamp)
    );
    
    PRINT 'Created ACM_MultivariateForecast table successfully.';
END
ELSE
BEGIN
    PRINT 'ACM_MultivariateForecast table already exists - skipping creation.';
END
GO

-- ============================================================================
-- Add foreign key constraints (if Equipment table exists)
-- ============================================================================
IF EXISTS (SELECT 1 FROM sys.tables WHERE name = 'Equipment')
BEGIN
    -- ACM_SensorCorrelations FK
    IF NOT EXISTS (
        SELECT 1 FROM sys.foreign_keys 
        WHERE name = 'FK_SensorCorrelations_Equipment'
    )
    BEGIN
        ALTER TABLE dbo.ACM_SensorCorrelations
        ADD CONSTRAINT FK_SensorCorrelations_Equipment
        FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID);
        PRINT 'Added FK_SensorCorrelations_Equipment constraint.';
    END
    
    -- ACM_MultivariateForecast FK
    IF NOT EXISTS (
        SELECT 1 FROM sys.foreign_keys 
        WHERE name = 'FK_MultivariateForecast_Equipment'
    )
    BEGIN
        ALTER TABLE dbo.ACM_MultivariateForecast
        ADD CONSTRAINT FK_MultivariateForecast_Equipment
        FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID);
        PRINT 'Added FK_MultivariateForecast_Equipment constraint.';
    END
END
GO

-- ============================================================================
-- Summary
-- ============================================================================
PRINT '';
PRINT '=== Migration 64 Complete ==='
PRINT 'Tables created/verified:';
PRINT '  - ACM_SensorCorrelations (pairwise sensor correlations)';
PRINT '  - ACM_MultivariateForecast (VAR model forecasts)';
PRINT '';
PRINT 'Next steps:';
PRINT '  1. Run config sync: python scripts/sql/populate_acm_config.py';
PRINT '  2. Run ACM batch: python scripts/sql_batch_runner.py --equip FD_FAN';
PRINT '';
GO

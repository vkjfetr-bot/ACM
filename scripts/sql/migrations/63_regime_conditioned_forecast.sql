-- Migration 63: Regime-Conditioned Forecasting Tables
-- Version: 10.1.0
-- Date: 2024-12-10
-- Purpose: Add per-regime RUL, hazard summaries, and drift context for forecasting

-- =====================================================
-- 1. ACM_RUL_ByRegime: Per-regime RUL estimates
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RUL_ByRegime')
BEGIN
    CREATE TABLE dbo.ACM_RUL_ByRegime (
        ID INT IDENTITY(1,1) PRIMARY KEY,
        EquipID INT NOT NULL,
        RunID NVARCHAR(64) NOT NULL,
        RegimeLabel INT NOT NULL,
        RUL_Hours FLOAT NOT NULL,
        P10_LowerBound FLOAT NULL,
        P50_Median FLOAT NULL,
        P90_UpperBound FLOAT NULL,
        DegradationRate FLOAT NULL,     -- Health units per hour in this regime
        Confidence FLOAT NULL,
        Method NVARCHAR(64) DEFAULT 'RegimeConditioned',
        SampleCount INT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE()
    );
    CREATE INDEX IX_RUL_ByRegime_Equip_Run ON dbo.ACM_RUL_ByRegime(EquipID, RunID);
    CREATE INDEX IX_RUL_ByRegime_Regime ON dbo.ACM_RUL_ByRegime(RegimeLabel);
    PRINT 'Created ACM_RUL_ByRegime table';
END
GO

-- =====================================================
-- 2. ACM_RegimeHazard: Per-regime hazard time series
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RegimeHazard')
BEGIN
    CREATE TABLE dbo.ACM_RegimeHazard (
        ID INT IDENTITY(1,1) PRIMARY KEY,
        EquipID INT NOT NULL,
        RunID NVARCHAR(64) NOT NULL,
        RegimeLabel INT NOT NULL,
        Timestamp DATETIME2 NOT NULL,
        HazardRate FLOAT NOT NULL,          -- Instantaneous failure rate
        SurvivalProb FLOAT NULL,            -- P(survival) cumulative
        CumulativeHazard FLOAT NULL,        -- Integral of hazard
        FailureProb FLOAT NULL,             -- 1 - SurvivalProb
        HealthAtTime FLOAT NULL,            -- Health index at this time
        DriftAtTime FLOAT NULL,             -- Drift Z at this time
        OMR_Z_AtTime FLOAT NULL,            -- OMR Z at this time
        CreatedAt DATETIME2 DEFAULT GETDATE()
    );
    CREATE INDEX IX_RegimeHazard_Equip_Run ON dbo.ACM_RegimeHazard(EquipID, RunID);
    CREATE INDEX IX_RegimeHazard_Timestamp ON dbo.ACM_RegimeHazard(Timestamp);
    PRINT 'Created ACM_RegimeHazard table';
END
GO

-- =====================================================
-- 3. ACM_ForecastContext: Unified forecast context with OMR/drift/regime
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_ForecastContext')
BEGIN
    CREATE TABLE dbo.ACM_ForecastContext (
        ID INT IDENTITY(1,1) PRIMARY KEY,
        EquipID INT NOT NULL,
        RunID NVARCHAR(64) NOT NULL,
        Timestamp DATETIME2 NOT NULL,       -- Reference timestamp
        ForecastHorizon_Hours FLOAT NULL,
        -- Current state
        CurrentHealth FLOAT NOT NULL,
        CurrentRegime INT NULL,
        RegimeConfidence FLOAT NULL,
        -- OMR context
        CurrentOMR_Z FLOAT NULL,
        OMR_Contribution FLOAT NULL,        -- Weight-adjusted contribution
        -- Drift context
        CurrentDrift_Z FLOAT NULL,
        DriftTrend NVARCHAR(16) NULL,       -- stable, increasing, decreasing
        -- Health trend
        FusedZ FLOAT NULL,
        HealthTrend NVARCHAR(16) NULL,      -- improving, stable, degrading
        -- Quality metrics
        DataQuality FLOAT NULL,             -- 0-1
        ModelConfidence FLOAT NULL,         -- 0-1
        ActiveDefects INT NULL,             -- Count of active sensor defects
        TopContributor NVARCHAR(128) NULL,  -- Sensor/detector driving health
        Notes NVARCHAR(512) NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE()
    );
    CREATE INDEX IX_ForecastContext_Equip_Run ON dbo.ACM_ForecastContext(EquipID, RunID);
    CREATE INDEX IX_ForecastContext_Timestamp ON dbo.ACM_ForecastContext(Timestamp);
    PRINT 'Created ACM_ForecastContext table';
END
GO

-- =====================================================
-- 4. ACM_AdaptiveThresholds_ByRegime: Per-regime adaptive thresholds
-- =====================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_AdaptiveThresholds_ByRegime')
BEGIN
    CREATE TABLE dbo.ACM_AdaptiveThresholds_ByRegime (
        ID INT IDENTITY(1,1) PRIMARY KEY,
        EquipID INT NOT NULL,
        RunID NVARCHAR(64) NOT NULL,
        RegimeLabel INT NOT NULL,
        ThresholdType NVARCHAR(64) NOT NULL,  -- fused_alert_z, fused_warn_z, etc.
        ThresholdValue FLOAT NOT NULL,
        CalculationMethod NVARCHAR(32) NULL,  -- quantile, mad, hybrid
        SampleCount INT NULL,
        RegimeHealthMean FLOAT NULL,          -- Mean health in this regime
        RegimeHealthStd FLOAT NULL,           -- Std health in this regime
        RegimeOccupancy FLOAT NULL,           -- % time spent in this regime
        IsActive BIT DEFAULT 1,
        CreatedAt DATETIME2 DEFAULT GETDATE()
    );
    CREATE INDEX IX_AdaptiveThresh_Regime ON dbo.ACM_AdaptiveThresholds_ByRegime(EquipID, RegimeLabel);
    CREATE INDEX IX_AdaptiveThresh_Type ON dbo.ACM_AdaptiveThresholds_ByRegime(ThresholdType);
    PRINT 'Created ACM_AdaptiveThresholds_ByRegime table';
END
GO

-- =====================================================
-- 5. Add RegimeLabel column to ACM_HealthForecast if missing
-- =====================================================
IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.ACM_HealthForecast') AND name = 'RegimeLabel'
)
BEGIN
    ALTER TABLE dbo.ACM_HealthForecast ADD RegimeLabel INT NULL;
    PRINT 'Added RegimeLabel column to ACM_HealthForecast';
END
GO

-- =====================================================
-- 6. Add context columns to ACM_RUL if missing
-- =====================================================
IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.ACM_RUL') AND name = 'DriftZ'
)
BEGIN
    ALTER TABLE dbo.ACM_RUL ADD DriftZ FLOAT NULL;
    PRINT 'Added DriftZ column to ACM_RUL';
END
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.ACM_RUL') AND name = 'CurrentRegime'
)
BEGIN
    ALTER TABLE dbo.ACM_RUL ADD CurrentRegime INT NULL;
    PRINT 'Added CurrentRegime column to ACM_RUL';
END
GO

IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.ACM_RUL') AND name = 'OMR_Z'
)
BEGIN
    ALTER TABLE dbo.ACM_RUL ADD OMR_Z FLOAT NULL;
    PRINT 'Added OMR_Z column to ACM_RUL';
END
GO

PRINT 'Migration 63 complete: Regime-conditioned forecasting tables created';

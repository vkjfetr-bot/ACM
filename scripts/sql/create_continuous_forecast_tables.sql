-- ============================================================================
-- Continuous Forecasting Tables (FORECAST-STATE-04)
-- Created: 2025-11-20
-- Purpose: Support state persistence and continuous forecasting
-- ============================================================================

USE ACM;
GO

-- ============================================================================
-- ACM_ForecastState: Persistent forecast model state between batches
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_ForecastState')
BEGIN
    CREATE TABLE dbo.ACM_ForecastState (
        EquipID INT NOT NULL,
        StateVersion INT NOT NULL,
        ModelType NVARCHAR(50),
        ModelParamsJson NVARCHAR(MAX),  -- JSON serialized model parameters
        ResidualVariance FLOAT,
        LastForecastHorizonJson NVARCHAR(MAX),  -- JSON array of forecast points
        HazardBaseline FLOAT,  -- EWMA smoothed hazard rate
        LastRetrainTime DATETIME2,
        TrainingDataHash NVARCHAR(64),  -- SHA256 hash for change detection
        TrainingWindowHours INT,
        ForecastQualityJson NVARCHAR(MAX),  -- {rmse, mae, mape}
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT PK_ForecastState PRIMARY KEY (EquipID, StateVersion)
    );
    
    CREATE INDEX IX_ForecastState_Latest ON dbo.ACM_ForecastState(EquipID, StateVersion DESC);
    
    PRINT 'Created table: ACM_ForecastState';
END
ELSE
BEGIN
    PRINT 'Table ACM_ForecastState already exists';
END
GO

-- ============================================================================
-- ACM_HealthForecast_Continuous: Merged forecast horizons with temporal blending
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_HealthForecast_Continuous')
BEGIN
    CREATE TABLE dbo.ACM_HealthForecast_Continuous (
        Timestamp DATETIME2 NOT NULL,
        ForecastHealth FLOAT NOT NULL,
        CI_Lower FLOAT,
        CI_Upper FLOAT,
        SourceRunID NVARCHAR(50) NOT NULL,  -- UUID string to match system-wide RunID format
        MergeWeight FLOAT,  -- Temporal blending weight (0-1)
        EquipID INT NOT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT PK_HealthForecast_Continuous PRIMARY KEY (EquipID, Timestamp, SourceRunID)
    );
    
    CREATE INDEX IX_HealthForecast_TimeRange ON dbo.ACM_HealthForecast_Continuous(EquipID, Timestamp);
    CREATE INDEX IX_HealthForecast_SourceRun ON dbo.ACM_HealthForecast_Continuous(EquipID, SourceRunID);
    
    PRINT 'Created table: ACM_HealthForecast_Continuous';
END
ELSE
BEGIN
    PRINT 'Table ACM_HealthForecast_Continuous already exists';
END
GO

-- ============================================================================
-- ACM_FailureHazard_TS: Smoothed hazard rates and survival probabilities
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_FailureHazard_TS')
BEGIN
    CREATE TABLE dbo.ACM_FailureHazard_TS (
        Timestamp DATETIME2 NOT NULL,
        HazardRaw FLOAT,  -- Raw hazard rate from batch probability
        HazardSmooth FLOAT,  -- EWMA smoothed hazard
        Survival FLOAT,  -- Survival probability S(t)
        FailureProb FLOAT,  -- Failure probability F(t) = 1 - S(t)
        RunID NVARCHAR(50),  -- UUID string to match system-wide RunID format
        EquipID INT,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT PK_FailureHazard_TS PRIMARY KEY (EquipID, RunID, Timestamp)
    );
    
    CREATE INDEX IX_FailureHazard_Time ON dbo.ACM_FailureHazard_TS(EquipID, Timestamp);
    
    PRINT 'Created table: ACM_FailureHazard_TS';
END
ELSE
BEGIN
    PRINT 'Table ACM_FailureHazard_TS already exists';
END
GO

-- ============================================================================
-- Extend ACM_RunMetadata for retraining tracking (FORECAST-STATE-08)
-- ============================================================================
IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.ACM_RunMetadata') 
    AND name = 'RetrainDecision'
)
BEGIN
    ALTER TABLE dbo.ACM_RunMetadata ADD
        RetrainDecision NVARCHAR(50),  -- "FULL_RETRAIN", "INCREMENTAL_UPDATE", "NO_RETRAIN"
        RetrainReason NVARCHAR(500),
        LastRetrainRunID INT,
        ModelAgeInBatches INT,
        ForecastQualityRMSE FLOAT,
        ForecastStateVersion INT;
    
    PRINT 'Extended ACM_RunMetadata with retraining columns';
END
ELSE
BEGIN
    PRINT 'ACM_RunMetadata already has RetrainDecision column';
END
GO

-- ============================================================================
-- Extend ACM_RUL_Summary for multi-path RUL (FORECAST-STATE-04)
-- ============================================================================
IF NOT EXISTS (
    SELECT * FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.ACM_RUL_Summary') 
    AND name = 'RUL_Trajectory_Hours'
)
BEGIN
    ALTER TABLE dbo.ACM_RUL_Summary ADD
        RUL_Trajectory_Hours FLOAT,  -- RUL from health trajectory crossing
        RUL_Hazard_Hours FLOAT,  -- RUL from hazard accumulation
        RUL_Energy_Hours FLOAT,  -- RUL from anomaly energy threshold
        RUL_Final_Hours FLOAT,  -- min(trajectory, hazard, energy)
        ConfidenceBand_Hours FLOAT,  -- CI_Upper - CI_Lower crossing time
        DominantPath NVARCHAR(20);  -- "trajectory" | "hazard" | "energy"
    
    PRINT 'Extended ACM_RUL_Summary with multi-path RUL columns';
END
ELSE
BEGIN
    PRINT 'ACM_RUL_Summary already has RUL_Trajectory_Hours column';
END
GO

PRINT '';
PRINT '============================================================================';
PRINT 'Continuous forecasting tables created/verified successfully!';
PRINT '============================================================================';
PRINT 'Tables created:';
PRINT '  - ACM_ForecastState (state persistence)';
PRINT '  - ACM_HealthForecast_Continuous (merged forecast horizons)';
PRINT '  - ACM_FailureHazard_TS (smoothed hazard/probability)';
PRINT 'Tables extended:';
PRINT '  - ACM_RunMetadata (retrain tracking)';
PRINT '  - ACM_RUL_Summary (multi-path RUL)';
PRINT '============================================================================';
GO

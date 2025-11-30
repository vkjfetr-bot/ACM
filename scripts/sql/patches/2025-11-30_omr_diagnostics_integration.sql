-- ============================================================================
-- OMR Diagnostics Integration - SQL Schema Update
-- Date: 2025-11-30
-- Purpose: Add SQL persistence for new OMR detector diagnostics from upgraded omr.py
-- ============================================================================

USE ACM;
GO

-- ============================================================================
-- ACM_OMR_Diagnostics: Model metadata and calibration status
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_OMR_Diagnostics')
BEGIN
    CREATE TABLE dbo.ACM_OMR_Diagnostics (
        DiagnosticID INT IDENTITY(1,1) PRIMARY KEY,
        RunID VARCHAR(100) NOT NULL,
        EquipID INT NOT NULL,
        
        -- Model architecture
        ModelType VARCHAR(20) NOT NULL,  -- "pls", "linear", "pca"
        NComponents INT NOT NULL,
        TrainSamples INT NOT NULL,
        TrainFeatures INT NOT NULL,
        
        -- Training quality
        TrainResidualStd FLOAT NOT NULL,
        TrainStartTime DATETIME2 NULL,
        TrainEndTime DATETIME2 NULL,
        
        -- Calibration status
        CalibrationStatus VARCHAR(20) NOT NULL,  -- "VALID", "SATURATED", "DISABLED"
        SaturationRate FLOAT NULL,  -- % of z-scores hitting max_z_score
        FusionWeight FLOAT NULL,  -- Current weight in fusion
        
        -- Metadata
        FitTimestamp DATETIME2 NOT NULL DEFAULT GETDATE(),
        CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        INDEX IX_OMR_Diagnostics_Run (RunID, EquipID),
        INDEX IX_OMR_Diagnostics_Equipment (EquipID, FitTimestamp DESC)
    );
    
    PRINT 'Created table: ACM_OMR_Diagnostics';
END
ELSE
BEGIN
    PRINT 'Table ACM_OMR_Diagnostics already exists';
END
GO

-- ============================================================================
-- ACM_Forecast_QualityMetrics: Track forecast accuracy over time
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_Forecast_QualityMetrics')
BEGIN
    CREATE TABLE dbo.ACM_Forecast_QualityMetrics (
        MetricID INT IDENTITY(1,1) PRIMARY KEY,
        RunID VARCHAR(100) NOT NULL,
        EquipID INT NOT NULL,
        
        -- Quality metrics
        RMSE FLOAT NULL,  -- Root Mean Squared Error
        MAE FLOAT NULL,   -- Mean Absolute Error
        MAPE FLOAT NULL,  -- Mean Absolute Percentage Error
        R2Score FLOAT NULL,  -- Coefficient of determination
        
        -- Model state
        DataHash VARCHAR(32) NULL,  -- Hash of training data
        ModelVersion INT NULL,
        RetrainTriggered BIT NOT NULL DEFAULT 0,
        RetrainReason VARCHAR(200) NULL,  -- "drift", "anomaly_spike", "poor_quality", etc.
        
        -- Metadata
        ForecastHorizonHours FLOAT NOT NULL,
        SampleCount INT NULL,
        ComputeTimestamp DATETIME2 NOT NULL DEFAULT GETDATE(),
        CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        INDEX IX_Forecast_Quality_Run (RunID, EquipID),
        INDEX IX_Forecast_Quality_Equipment (EquipID, ComputeTimestamp DESC)
    );
    
    PRINT 'Created table: ACM_Forecast_QualityMetrics';
END
ELSE
BEGIN
    PRINT 'Table ACM_Forecast_QualityMetrics already exists';
END
GO

-- ============================================================================
-- Add OMR Diagnostics to ALLOWED_TABLES in output_manager
-- ============================================================================
PRINT '';
PRINT '============================================================================';
PRINT 'SQL schema updates completed successfully!';
PRINT '============================================================================';
PRINT 'Tables created/verified:';
PRINT '  - ACM_OMR_Diagnostics: OMR model metadata and calibration';
PRINT '  - ACM_Forecast_QualityMetrics: Forecast accuracy tracking';
PRINT '';
PRINT 'Next steps:';
PRINT '  1. Update output_manager.py ALLOWED_TABLES';
PRINT '  2. Update acm_main.py to call omr_detector.get_diagnostics()';
PRINT '  3. Update forecasting.py to compute and write quality metrics';
PRINT '  4. Update Grafana dashboards with new panels';
PRINT '============================================================================';
GO

/*
====================================================================================
ACM Missing Tables Creation Script
====================================================================================
Created: December 25, 2025
Purpose: Create 15 missing tables identified in ACM_IMPLEMENTATION_ROADMAP.md

Missing Tables by Priority:
P1 - Root Cause (5): EpisodeDiagnostics, DetectorCorrelation, DriftSeries, 
                     SensorCorrelations, FeatureDropLog
P2 - Model Quality (3): SensorNormalized_TS, CalibrationSummary, PCA_Metrics
P3 - Advanced Analytics (5): RegimeOccupancy, RegimeTransitions, ContributionTimeline,
                             RegimePromotionLog, DriftController
P4 - V11 Features (5): RegimeDefinitions, ActiveModels, DataContractValidation,
                       SeasonalPatterns, AssetProfiles

Note: Some tables may already exist in 14_complete_schema.sql but not created.
This script uses IF NOT EXISTS pattern for safety.
====================================================================================
*/

USE ACM;
GO
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

-- ============================================================================
-- P1: ROOT CAUSE TABLES (Tier 3)
-- ============================================================================

-- ACM_EpisodeDiagnostics: Per-episode diagnostic details
IF OBJECT_ID('dbo.ACM_EpisodeDiagnostics', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_EpisodeDiagnostics (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        EpisodeID INT NULL,
        StartTime DATETIME2 NOT NULL,
        EndTime DATETIME2 NULL,
        DurationHours FLOAT NULL,
        PeakZ FLOAT NULL,
        AvgZ FLOAT NULL,
        Severity NVARCHAR(20) NULL,  -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        TopSensor1 NVARCHAR(200) NULL,
        TopSensor2 NVARCHAR(200) NULL,
        TopSensor3 NVARCHAR(200) NULL,
        RegimeAtStart NVARCHAR(50) NULL,
        AlertMode NVARCHAR(50) NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_EpisodeDiagnostics PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_EpisodeDiagnostics_Run NONCLUSTERED (RunID, EquipID),
        INDEX IX_ACM_EpisodeDiagnostics_Equip NONCLUSTERED (EquipID, StartTime DESC)
    );
    PRINT 'Created ACM_EpisodeDiagnostics';
END
GO

-- ACM_DetectorCorrelation: Inter-detector correlation matrix
IF OBJECT_ID('dbo.ACM_DetectorCorrelation', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_DetectorCorrelation (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        Detector1 NVARCHAR(50) NOT NULL,  -- 'ar1_z', 'pca_spe_z', etc.
        Detector2 NVARCHAR(50) NOT NULL,
        Correlation FLOAT NOT NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_DetectorCorrelation PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_DetectorCorrelation_Run NONCLUSTERED (RunID, EquipID)
    );
    PRINT 'Created ACM_DetectorCorrelation';
END
GO

-- ACM_DriftSeries: Drift detection time series (CUSUM values)
IF OBJECT_ID('dbo.ACM_DriftSeries', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_DriftSeries (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        Timestamp DATETIME2 NOT NULL,
        DriftValue FLOAT NOT NULL,
        DriftState NVARCHAR(20) NULL,  -- 'STABLE', 'DRIFTING', 'RESET'
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_DriftSeries PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_DriftSeries_Run NONCLUSTERED (RunID, EquipID, Timestamp),
        INDEX IX_ACM_DriftSeries_Equip NONCLUSTERED (EquipID, Timestamp DESC)
    );
    PRINT 'Created ACM_DriftSeries';
END
GO

-- ACM_SensorCorrelations: Sensor-to-sensor correlation matrix
IF OBJECT_ID('dbo.ACM_SensorCorrelations', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_SensorCorrelations (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        Sensor1 NVARCHAR(200) NOT NULL,
        Sensor2 NVARCHAR(200) NOT NULL,
        Correlation FLOAT NOT NULL,
        CorrelationType NVARCHAR(20) NULL,  -- 'pearson', 'spearman'
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_SensorCorrelations PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_SensorCorrelations_Run NONCLUSTERED (RunID, EquipID)
    );
    PRINT 'Created ACM_SensorCorrelations';
END
GO

-- ACM_FeatureDropLog: Log of features dropped during preprocessing
IF OBJECT_ID('dbo.ACM_FeatureDropLog', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_FeatureDropLog (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        FeatureName NVARCHAR(200) NOT NULL,
        DropReason NVARCHAR(100) NOT NULL,  -- 'nan_pct_high', 'zero_variance', 'constant', 'correlation'
        DropValue FLOAT NULL,  -- e.g., NaN percentage
        Threshold FLOAT NULL,  -- threshold that triggered drop
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_FeatureDropLog PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_FeatureDropLog_Run NONCLUSTERED (RunID, EquipID),
        INDEX IX_ACM_FeatureDropLog_Reason NONCLUSTERED (DropReason, EquipID)
    );
    PRINT 'Created ACM_FeatureDropLog';
END
GO

-- ============================================================================
-- P2: MODEL QUALITY TABLES (Tier 4)
-- ============================================================================

-- ACM_SensorNormalized_TS: Normalized sensor values time series
IF OBJECT_ID('dbo.ACM_SensorNormalized_TS', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_SensorNormalized_TS (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        Timestamp DATETIME2 NOT NULL,
        SensorName NVARCHAR(200) NOT NULL,
        RawValue FLOAT NULL,
        NormalizedValue FLOAT NULL,  -- Z-score normalized
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_SensorNormalized_TS PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_SensorNormalized_TS_Run NONCLUSTERED (RunID, EquipID, Timestamp),
        INDEX IX_ACM_SensorNormalized_TS_Sensor NONCLUSTERED (EquipID, SensorName, Timestamp DESC)
    );
    PRINT 'Created ACM_SensorNormalized_TS';
END
GO

-- ACM_CalibrationSummary: Model calibration quality over time
IF OBJECT_ID('dbo.ACM_CalibrationSummary', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_CalibrationSummary (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        DetectorType NVARCHAR(50) NOT NULL,  -- 'ar1', 'pca_spe', 'iforest', etc.
        CalibrationScore FLOAT NULL,
        TrainR2 FLOAT NULL,
        MeanAbsError FLOAT NULL,
        P95Error FLOAT NULL,
        DatapointsUsed INT NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_CalibrationSummary PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_CalibrationSummary_Run NONCLUSTERED (RunID, EquipID),
        INDEX IX_ACM_CalibrationSummary_Equip NONCLUSTERED (EquipID, DetectorType)
    );
    PRINT 'Created ACM_CalibrationSummary';
END
GO

-- ACM_PCA_Metrics: PCA component metrics
IF OBJECT_ID('dbo.ACM_PCA_Metrics', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_PCA_Metrics (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        ComponentIndex INT NOT NULL,
        ExplainedVariance FLOAT NULL,
        CumulativeVariance FLOAT NULL,
        Eigenvalue FLOAT NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_PCA_Metrics PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_PCA_Metrics_Run NONCLUSTERED (RunID, EquipID)
    );
    PRINT 'Created ACM_PCA_Metrics';
END
GO

-- ============================================================================
-- P3: ADVANCED ANALYTICS TABLES (Tier 6)
-- ============================================================================

-- ACM_RegimeOccupancy: Time spent in each operating regime
IF OBJECT_ID('dbo.ACM_RegimeOccupancy', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_RegimeOccupancy (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        RegimeLabel NVARCHAR(50) NOT NULL,
        DwellTimeHours FLOAT NOT NULL,
        DwellFraction FLOAT NOT NULL,  -- 0.0 to 1.0
        EntryCount INT NULL,
        AvgDwellMinutes FLOAT NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_RegimeOccupancy PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_RegimeOccupancy_Run NONCLUSTERED (RunID, EquipID)
    );
    PRINT 'Created ACM_RegimeOccupancy';
END
GO

-- ACM_RegimeTransitions: Regime transition matrix
IF OBJECT_ID('dbo.ACM_RegimeTransitions', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_RegimeTransitions (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        FromRegime NVARCHAR(50) NOT NULL,
        ToRegime NVARCHAR(50) NOT NULL,
        TransitionCount INT NOT NULL,
        TransitionProbability FLOAT NULL,  -- 0.0 to 1.0
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_RegimeTransitions PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_RegimeTransitions_Run NONCLUSTERED (RunID, EquipID)
    );
    PRINT 'Created ACM_RegimeTransitions';
END
GO

-- ACM_ContributionTimeline: Historical sensor contribution to fused score
IF OBJECT_ID('dbo.ACM_ContributionTimeline', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_ContributionTimeline (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        Timestamp DATETIME2 NOT NULL,
        DetectorType NVARCHAR(50) NOT NULL,
        ContributionPct FLOAT NOT NULL,  -- 0-100
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_ContributionTimeline PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_ContributionTimeline_Run NONCLUSTERED (RunID, EquipID, Timestamp)
    );
    PRINT 'Created ACM_ContributionTimeline';
END
GO

-- ACM_RegimePromotionLog: Regime maturity state changes (v11)
IF OBJECT_ID('dbo.ACM_RegimePromotionLog', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_RegimePromotionLog (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        RegimeLabel NVARCHAR(50) NOT NULL,
        FromState NVARCHAR(30) NOT NULL,  -- 'INITIALIZING', 'LEARNING', 'CONVERGED'
        ToState NVARCHAR(30) NOT NULL,
        Reason NVARCHAR(200) NULL,
        DataPointsAtPromotion INT NULL,
        PromotedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_RegimePromotionLog PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_RegimePromotionLog_Equip NONCLUSTERED (EquipID, PromotedAt DESC)
    );
    PRINT 'Created ACM_RegimePromotionLog';
END
GO

-- ACM_DriftController: Drift detection configuration and state
IF OBJECT_ID('dbo.ACM_DriftController', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_DriftController (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        ControllerState NVARCHAR(30) NOT NULL,  -- 'STABLE', 'ALERT', 'RESET'
        Threshold FLOAT NULL,
        Sensitivity FLOAT NULL,
        LastDriftValue FLOAT NULL,
        LastDriftTime DATETIME2 NULL,
        ResetCount INT NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_DriftController PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_DriftController_Equip NONCLUSTERED (EquipID, CreatedAt DESC)
    );
    PRINT 'Created ACM_DriftController';
END
GO

-- ============================================================================
-- P4: V11 FEATURE TABLES (Tier 7)
-- ============================================================================

-- ACM_RegimeDefinitions: Regime centroids and metadata (v11)
IF OBJECT_ID('dbo.ACM_RegimeDefinitions', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_RegimeDefinitions (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        EquipID INT NOT NULL,
        RegimeVersion INT NOT NULL,
        RegimeID INT NOT NULL,
        RegimeName NVARCHAR(100) NOT NULL,
        CentroidJSON NVARCHAR(MAX) NOT NULL,  -- JSON array of centroid values
        FeatureColumns NVARCHAR(MAX) NOT NULL,  -- JSON array of feature names
        DataPointCount INT NOT NULL,
        SilhouetteScore FLOAT NULL,
        MaturityState NVARCHAR(30) NULL,  -- 'INITIALIZING', 'LEARNING', 'CONVERGED'
        CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        CreatedByRunID NVARCHAR(50) NULL,
        
        CONSTRAINT PK_ACM_RegimeDefinitions PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_RegimeDefinitions_Equip NONCLUSTERED (EquipID, RegimeVersion DESC)
    );
    PRINT 'Created ACM_RegimeDefinitions';
END
GO

-- ACM_ActiveModels: Active model versions per equipment (v11)
IF OBJECT_ID('dbo.ACM_ActiveModels', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_ActiveModels (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        EquipID INT NOT NULL,
        ActiveRegimeVersion INT NULL,
        RegimeMaturityState NVARCHAR(30) NULL,
        RegimePromotedAt DATETIME2 NULL,
        ActiveThresholdVersion INT NULL,
        ThresholdPromotedAt DATETIME2 NULL,
        ActiveForecastVersion INT NULL,
        ForecastPromotedAt DATETIME2 NULL,
        LastUpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        LastUpdatedBy NVARCHAR(100) NULL,
        
        CONSTRAINT PK_ACM_ActiveModels PRIMARY KEY CLUSTERED (ID),
        CONSTRAINT UQ_ACM_ActiveModels_Equip UNIQUE (EquipID),
        INDEX IX_ACM_ActiveModels_Updated NONCLUSTERED (LastUpdatedAt DESC)
    );
    PRINT 'Created ACM_ActiveModels';
END
GO

-- ACM_DataContractValidation: Pipeline entry validation (v11)
IF OBJECT_ID('dbo.ACM_DataContractValidation', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_DataContractValidation (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(50) NOT NULL,
        EquipID INT NOT NULL,
        Passed BIT NOT NULL,
        RowsValidated INT NOT NULL,
        ColumnsValidated INT NOT NULL,
        IssuesJSON NVARCHAR(MAX) NULL,  -- JSON array of issues
        WarningsJSON NVARCHAR(MAX) NULL,  -- JSON array of warnings
        ContractSignature NVARCHAR(100) NULL,  -- Hash of expected schema
        ValidatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        CONSTRAINT PK_ACM_DataContractValidation PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_DataContractValidation_Run NONCLUSTERED (RunID, EquipID)
    );
    PRINT 'Created ACM_DataContractValidation';
END
GO

-- ACM_SeasonalPatterns: Detected seasonal patterns (v11)
IF OBJECT_ID('dbo.ACM_SeasonalPatterns', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_SeasonalPatterns (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        EquipID INT NOT NULL,
        SensorName NVARCHAR(200) NOT NULL,
        PatternType NVARCHAR(30) NOT NULL,  -- 'DIURNAL', 'WEEKLY', 'SEASONAL'
        PeriodHours FLOAT NOT NULL,
        Amplitude FLOAT NOT NULL,
        PhaseShift FLOAT NULL,
        Confidence FLOAT NULL,
        DetectedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        DetectedByRunID NVARCHAR(50) NULL,
        
        CONSTRAINT PK_ACM_SeasonalPatterns PRIMARY KEY CLUSTERED (ID),
        INDEX IX_ACM_SeasonalPatterns_Equip NONCLUSTERED (EquipID, SensorName)
    );
    PRINT 'Created ACM_SeasonalPatterns';
END
GO

-- ACM_AssetProfiles: Asset similarity profiles (v11)
IF OBJECT_ID('dbo.ACM_AssetProfiles', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_AssetProfiles (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        EquipID INT NOT NULL,
        EquipType NVARCHAR(100) NOT NULL,
        SensorNamesJSON NVARCHAR(MAX) NOT NULL,  -- JSON array
        SensorMeansJSON NVARCHAR(MAX) NOT NULL,  -- JSON array
        SensorStdsJSON NVARCHAR(MAX) NOT NULL,   -- JSON array
        RegimeCount INT NULL,
        TypicalHealth FLOAT NULL,
        DataHours FLOAT NULL,
        LastUpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        LastUpdatedByRunID NVARCHAR(50) NULL,
        
        CONSTRAINT PK_ACM_AssetProfiles PRIMARY KEY CLUSTERED (ID),
        CONSTRAINT UQ_ACM_AssetProfiles_Equip UNIQUE (EquipID),
        INDEX IX_ACM_AssetProfiles_Type NONCLUSTERED (EquipType)
    );
    PRINT 'Created ACM_AssetProfiles';
END
GO

-- ============================================================================
-- SUMMARY
-- ============================================================================
PRINT '';
PRINT '====================================================================================';
PRINT 'ACM Missing Tables Creation Complete';
PRINT '====================================================================================';
PRINT 'P1 - Root Cause (5): EpisodeDiagnostics, DetectorCorrelation, DriftSeries,';
PRINT '                     SensorCorrelations, FeatureDropLog';
PRINT 'P2 - Model Quality (3): SensorNormalized_TS, CalibrationSummary, PCA_Metrics';
PRINT 'P3 - Advanced Analytics (5): RegimeOccupancy, RegimeTransitions, ContributionTimeline,';
PRINT '                             RegimePromotionLog, DriftController';
PRINT 'P4 - V11 Features (5): RegimeDefinitions, ActiveModels, DataContractValidation,';
PRINT '                       SeasonalPatterns, AssetProfiles';
PRINT '';
PRINT 'Total: 18 tables (includes some overlap with existing DDL)';
PRINT '====================================================================================';
GO

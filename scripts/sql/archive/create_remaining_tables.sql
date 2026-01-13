-- Create remaining 5 missing ACM tables
-- These tables are referenced in OutputManager.ALLOWED_TABLES but were not created
-- Run this against the ACM database

USE ACM;
GO

-- 1. ACM_PCA_Models: Model metadata for PCA subspace detector
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_PCA_Models')
BEGIN
    CREATE TABLE dbo.ACM_PCA_Models (
        RecordID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        EntryDateTime DATETIME2 NOT NULL,
        NComponents INT NULL,
        TargetVar NVARCHAR(MAX) NULL,  -- JSON: {SPE_P95_train, T2_P95_train}
        VarExplainedJSON NVARCHAR(MAX) NULL,  -- JSON array of variance ratios
        ScalingSpecJSON NVARCHAR(MAX) NULL,  -- JSON: {scaler, with_mean, with_std}
        ModelVersion NVARCHAR(50) NULL,
        TrainStartEntryDateTime DATETIME2 NULL,
        TrainEndEntryDateTime DATETIME2 NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT FK_PCAModels_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID)
    );
    
    CREATE NONCLUSTERED INDEX IX_PCAModels_RunID ON dbo.ACM_PCA_Models(RunID);
    CREATE NONCLUSTERED INDEX IX_PCAModels_EquipID ON dbo.ACM_PCA_Models(EquipID);
    
    PRINT 'Created table ACM_PCA_Models';
END
ELSE
BEGIN
    PRINT 'Table ACM_PCA_Models already exists';
END
GO

-- 2. ACM_PCA_Loadings: Component loadings for each sensor
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_PCA_Loadings')
BEGIN
    CREATE TABLE dbo.ACM_PCA_Loadings (
        RecordID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        EntryDateTime DATETIME2 NOT NULL,
        ComponentNo INT NOT NULL,
        ComponentID INT NULL,  -- Alias for ComponentNo
        Sensor NVARCHAR(200) NOT NULL,
        FeatureName NVARCHAR(200) NULL,  -- Alias for Sensor
        Loading FLOAT NOT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT FK_PCALoadings_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID)
    );
    
    CREATE NONCLUSTERED INDEX IX_PCALoadings_RunID ON dbo.ACM_PCA_Loadings(RunID);
    CREATE NONCLUSTERED INDEX IX_PCALoadings_EquipID_Component ON dbo.ACM_PCA_Loadings(EquipID, ComponentNo);
    
    PRINT 'Created table ACM_PCA_Loadings';
END
ELSE
BEGIN
    PRINT 'Table ACM_PCA_Loadings already exists';
END
GO

-- 3. ACM_RegimeSummary: Per-Run Regime Statistics
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RegimeSummary')
BEGIN
    CREATE TABLE dbo.ACM_RegimeSummary (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        Regime INT NOT NULL,
        State VARCHAR(50) NULL,
        DwellSeconds FLOAT NULL,
        DwellFraction FLOAT NULL,
        AvgDwellSeconds FLOAT NULL,
        TransitionCount INT NULL,
        StabilityScore FLOAT NULL,
        MedianFused FLOAT NULL,
        P95AbsFused FLOAT NULL,
        Count INT NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        CONSTRAINT PK_ACM_RegimeSummary PRIMARY KEY CLUSTERED (RunID, EquipID, Regime),
        CONSTRAINT FK_RegimeSummary_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID),
        INDEX IX_ACM_RegimeSummary_Equip NONCLUSTERED (EquipID, RunID)
    );
    
    PRINT 'Created table ACM_RegimeSummary';
END
ELSE
BEGIN
    PRINT 'Table ACM_RegimeSummary already exists';
END
GO

-- 4. ACM_RegimeFeatureImportance: Feature importance for regime clustering
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RegimeFeatureImportance')
BEGIN
    CREATE TABLE dbo.ACM_RegimeFeatureImportance (
        ID BIGINT IDENTITY(1,1) NOT NULL,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        Feature NVARCHAR(200) NOT NULL,
        Importance FLOAT NOT NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        
        CONSTRAINT PK_ACM_RegimeFeatureImportance PRIMARY KEY CLUSTERED (ID),
        CONSTRAINT FK_RegimeFeatureImportance_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID),
        INDEX IX_ACM_RegimeFeatureImportance_Run NONCLUSTERED (RunID, EquipID, Importance DESC)
    );
    
    PRINT 'Created table ACM_RegimeFeatureImportance';
END
ELSE
BEGIN
    PRINT 'Table ACM_RegimeFeatureImportance already exists';
END
GO

-- 5. ACM_Run_Stats: Compact run statistics summary
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_Run_Stats')
BEGIN
    CREATE TABLE dbo.ACM_Run_Stats (
        RecordID BIGINT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        WindowStartEntryDateTime DATETIME2 NULL,
        WindowEndEntryDateTime DATETIME2 NULL,
        SamplesIn INT NULL,
        SamplesKept INT NULL,
        SensorsKept INT NULL,
        CadenceOKPct FLOAT NULL,
        DriftP95 FLOAT NULL,
        ReconRMSE FLOAT NULL,
        AnomalyCount INT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT FK_RunStats_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID)
    );
    
    CREATE NONCLUSTERED INDEX IX_RunStats_RunID ON dbo.ACM_Run_Stats(RunID);
    CREATE NONCLUSTERED INDEX IX_RunStats_EquipID ON dbo.ACM_Run_Stats(EquipID);
    
    PRINT 'Created table ACM_Run_Stats';
END
ELSE
BEGIN
    PRINT 'Table ACM_Run_Stats already exists';
END
GO

PRINT 'All 5 missing tables created successfully.';

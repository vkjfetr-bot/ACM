-- Drop synonyms and create proper tables
USE ACM;
GO

-- Drop existing synonyms
IF EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_PCA_Models')
    DROP SYNONYM dbo.ACM_PCA_Models;
    
IF EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_PCA_Loadings')
    DROP SYNONYM dbo.ACM_PCA_Loadings;
    
IF EXISTS (SELECT * FROM sys.synonyms WHERE name = 'ACM_Run_Stats')
    DROP SYNONYM dbo.ACM_Run_Stats;
GO

-- 1. ACM_PCA_Models
CREATE TABLE dbo.ACM_PCA_Models (
    RecordID BIGINT IDENTITY(1,1) PRIMARY KEY,
    RunID UNIQUEIDENTIFIER NOT NULL,
    EquipID INT NOT NULL,
    EntryDateTime DATETIME2 NOT NULL,
    NComponents INT NULL,
    TargetVar NVARCHAR(MAX) NULL,
    VarExplainedJSON NVARCHAR(MAX) NULL,
    ScalingSpecJSON NVARCHAR(MAX) NULL,
    ModelVersion NVARCHAR(50) NULL,
    TrainStartEntryDateTime DATETIME2 NULL,
    TrainEndEntryDateTime DATETIME2 NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_PCAModels_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID)
);
CREATE NONCLUSTERED INDEX IX_PCAModels_RunID ON dbo.ACM_PCA_Models(RunID);
CREATE NONCLUSTERED INDEX IX_PCAModels_EquipID ON dbo.ACM_PCA_Models(EquipID);
GO

-- 2. ACM_PCA_Loadings
CREATE TABLE dbo.ACM_PCA_Loadings (
    RecordID BIGINT IDENTITY(1,1) PRIMARY KEY,
    RunID UNIQUEIDENTIFIER NOT NULL,
    EquipID INT NOT NULL,
    EntryDateTime DATETIME2 NOT NULL,
    ComponentNo INT NOT NULL,
    ComponentID INT NULL,
    Sensor NVARCHAR(200) NOT NULL,
    FeatureName NVARCHAR(200) NULL,
    Loading FLOAT NOT NULL,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    CONSTRAINT FK_PCALoadings_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID)
);
CREATE NONCLUSTERED INDEX IX_PCALoadings_RunID ON dbo.ACM_PCA_Loadings(RunID);
CREATE NONCLUSTERED INDEX IX_PCALoadings_EquipID_Component ON dbo.ACM_PCA_Loadings(EquipID, ComponentNo);
GO

-- 3. ACM_Run_Stats
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
GO

PRINT 'Dropped synonyms and created proper tables.';

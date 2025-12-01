-- Create missing ACM tables required for SQL-only operation
-- Run this script against the ACM database

USE ACM;
GO

-- 1. ACM_RUL_LearningState: Stores online learning state for RUL ensemble models
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_RUL_LearningState')
BEGIN
    CREATE TABLE dbo.ACM_RUL_LearningState (
        EquipID INT PRIMARY KEY,
        AR1_MAE FLOAT NULL,
        AR1_RMSE FLOAT NULL,
        AR1_Bias FLOAT NULL,
        AR1_RecentErrors NVARCHAR(MAX) NULL,  -- JSON array
        AR1_Weight FLOAT NULL,
        Exp_MAE FLOAT NULL,
        Exp_RMSE FLOAT NULL,
        Exp_Bias FLOAT NULL,
        Exp_RecentErrors NVARCHAR(MAX) NULL,  -- JSON array
        Exp_Weight FLOAT NULL,
        Weibull_MAE FLOAT NULL,
        Weibull_RMSE FLOAT NULL,
        Weibull_Bias FLOAT NULL,
        Weibull_RecentErrors NVARCHAR(MAX) NULL,  -- JSON array
        Weibull_Weight FLOAT NULL,
        CalibrationFactor FLOAT NULL,
        LastUpdated DATETIME2 NULL,
        PredictionHistory NVARCHAR(MAX) NULL,  -- JSON array
        CONSTRAINT FK_RULLearning_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID)
    );
    PRINT 'Created table ACM_RUL_LearningState';
END
ELSE
BEGIN
    PRINT 'Table ACM_RUL_LearningState already exists';
END
GO

-- 2. ACM_EpisodesQC: Run-level episode quality summary
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_EpisodesQC')
BEGIN
    CREATE TABLE dbo.ACM_EpisodesQC (
        RecordID INT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        EpisodeCount INT NULL,
        MedianDurationMinutes FLOAT NULL,
        CoveragePct FLOAT NULL,
        TimeInAlertPct FLOAT NULL,
        MaxFusedZ FLOAT NULL,
        AvgFusedZ FLOAT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        CONSTRAINT FK_EpisodesQC_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID)
    );
    
    CREATE NONCLUSTERED INDEX IX_EpisodesQC_RunID ON dbo.ACM_EpisodesQC(RunID);
    CREATE NONCLUSTERED INDEX IX_EpisodesQC_EquipID ON dbo.ACM_EpisodesQC(EquipID);
    
    PRINT 'Created table ACM_EpisodesQC';
END
ELSE
BEGIN
    PRINT 'Table ACM_EpisodesQC already exists';
END
GO

PRINT 'Missing tables creation script completed.';

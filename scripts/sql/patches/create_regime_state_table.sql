-- ============================================================================
-- Script: create_regime_state_table.sql
-- Purpose: Create ACM_RegimeState table for regime model state persistence
--          Enables batch-to-batch continuity for regime clustering
-- Author: ACM System
-- Date: 2025-01-24
-- ============================================================================

USE ConditionMonitoring;
GO

PRINT 'Creating ACM_RegimeState table...';
GO

IF OBJECT_ID('dbo.ACM_RegimeState', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ACM_RegimeState (
        EquipID                      INT NOT NULL,
        StateVersion                 INT NOT NULL,
        NumClusters                  INT NOT NULL,
        ClusterCentersJson           NVARCHAR(MAX) NULL,
        ScalerMeanJson               NVARCHAR(MAX) NULL,
        ScalerScaleJson              NVARCHAR(MAX) NULL,
        PCAComponentsJson            NVARCHAR(MAX) NULL,
        PCAExplainedVarianceJson     NVARCHAR(MAX) NULL,
        NumPCAComponents             INT NOT NULL DEFAULT 0,
        SilhouetteScore              FLOAT NULL,
        QualityOk                    BIT NOT NULL DEFAULT 0,
        LastTrainedTime              DATETIME2(3) NOT NULL,
        ConfigHash                   NVARCHAR(64) NULL,
        RegimeBasisHash              NVARCHAR(64) NULL,
        CreatedAt                    DATETIME2(3) NOT NULL CONSTRAINT DF_ACM_RegimeState_CreatedAt DEFAULT (SYSUTCDATETIME()),
        
        CONSTRAINT PK_ACM_RegimeState PRIMARY KEY CLUSTERED (EquipID, StateVersion)
    );
    
    -- Index for querying latest state per equipment
    CREATE NONCLUSTERED INDEX IX_RegimeState_EquipID_Version 
        ON dbo.ACM_RegimeState(EquipID, StateVersion DESC);
    
    -- Index for quality filtering
    CREATE NONCLUSTERED INDEX IX_RegimeState_QualityOk 
        ON dbo.ACM_RegimeState(EquipID, QualityOk, StateVersion DESC)
        WHERE QualityOk = 1;
    
    PRINT 'ACM_RegimeState table created successfully.';
END
ELSE
BEGIN
    PRINT 'ACM_RegimeState table already exists. Verifying schema...';
    
    -- Add columns if they don't exist (for incremental updates)
    IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RegimeState') AND name = 'NumPCAComponents')
    BEGIN
        ALTER TABLE dbo.ACM_RegimeState ADD NumPCAComponents INT NOT NULL DEFAULT 0;
        PRINT 'Added NumPCAComponents column.';
    END
    
    IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RegimeState') AND name = 'PCAComponentsJson')
    BEGIN
        ALTER TABLE dbo.ACM_RegimeState ADD PCAComponentsJson NVARCHAR(MAX) NULL;
        PRINT 'Added PCAComponentsJson column.';
    END
    
    IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RegimeState') AND name = 'PCAExplainedVarianceJson')
    BEGIN
        ALTER TABLE dbo.ACM_RegimeState ADD PCAExplainedVarianceJson NVARCHAR(MAX) NULL;
        PRINT 'Added PCAExplainedVarianceJson column.';
    END
    
    IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RegimeState') AND name = 'RegimeBasisHash')
    BEGIN
        ALTER TABLE dbo.ACM_RegimeState ADD RegimeBasisHash NVARCHAR(64) NULL;
        PRINT 'Added RegimeBasisHash column.';
    END
    
    PRINT 'Schema verification complete.';
END
GO

PRINT 'ACM_RegimeState table setup complete!';
GO

-- Sample query to check latest state per equipment
-- SELECT 
--     EquipID, 
--     StateVersion, 
--     NumClusters, 
--     SilhouetteScore, 
--     QualityOk, 
--     LastTrainedTime,
--     LEN(ClusterCentersJson) AS CentersSizeBytes
-- FROM dbo.ACM_RegimeState
-- WHERE StateVersion = (SELECT MAX(StateVersion) FROM dbo.ACM_RegimeState AS sub WHERE sub.EquipID = ACM_RegimeState.EquipID)
-- ORDER BY EquipID;

-- ============================================================================
-- ACM_RegimeDefinitions Table Migration
-- Version: 11.0.0
-- Purpose: Store immutable regime model definitions with versioning
-- Phase: P2.5
-- ============================================================================

-- Create ACM_RegimeDefinitions table
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'dbo.ACM_RegimeDefinitions') AND type = 'U')
BEGIN
    CREATE TABLE dbo.ACM_RegimeDefinitions (
        -- Primary Key
        DefinitionID        INT IDENTITY(1,1) PRIMARY KEY,
        
        -- Equipment + Version (natural key)
        EquipID             INT NOT NULL,
        RegimeVersion       INT NOT NULL,
        
        -- Model summary
        NumRegimes          INT NOT NULL,
        
        -- JSON-serialized model components
        Centroids           NVARCHAR(MAX) NOT NULL,    -- JSON array of centroid objects
        FeatureColumns      NVARCHAR(MAX) NOT NULL,    -- JSON array of column names
        ScalerParams        NVARCHAR(MAX) NOT NULL,    -- JSON object with mean/scale arrays
        TransitionMatrix    NVARCHAR(MAX) NULL,        -- JSON 2D array (optional)
        DiscoveryParams     NVARCHAR(MAX) NULL,        -- JSON discovery config
        
        -- Training metadata
        TrainingRowCount    INT NULL,
        TrainingStartTime   DATETIME NULL,
        TrainingEndTime     DATETIME NULL,
        
        -- Audit columns
        CreatedAt           DATETIME NOT NULL DEFAULT GETDATE(),
        CreatedBy           NVARCHAR(128) NULL,
        
        -- Unique constraint on EquipID + Version
        CONSTRAINT UQ_RegimeDefinitions_EquipVersion UNIQUE (EquipID, RegimeVersion),
        
        -- Foreign key to Equipment
        CONSTRAINT FK_RegimeDefinitions_Equipment FOREIGN KEY (EquipID)
            REFERENCES Equipment(EquipID)
    );
    
    PRINT 'Created table ACM_RegimeDefinitions';
END
ELSE
BEGIN
    PRINT 'Table ACM_RegimeDefinitions already exists';
END
GO

-- Create indexes for common queries
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_RegimeDefinitions_EquipID_Version')
BEGIN
    CREATE NONCLUSTERED INDEX IX_RegimeDefinitions_EquipID_Version
    ON dbo.ACM_RegimeDefinitions (EquipID, RegimeVersion DESC);
    
    PRINT 'Created index IX_RegimeDefinitions_EquipID_Version';
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_RegimeDefinitions_CreatedAt')
BEGIN
    CREATE NONCLUSTERED INDEX IX_RegimeDefinitions_CreatedAt
    ON dbo.ACM_RegimeDefinitions (CreatedAt DESC);
    
    PRINT 'Created index IX_RegimeDefinitions_CreatedAt';
END
GO

-- ============================================================================
-- Helper stored procedure to get latest version for equipment
-- ============================================================================
IF OBJECT_ID('dbo.usp_ACM_GetLatestRegimeVersion', 'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_ACM_GetLatestRegimeVersion;
GO

CREATE PROCEDURE dbo.usp_ACM_GetLatestRegimeVersion
    @EquipID INT
AS
BEGIN
    SET NOCOUNT ON;
    
    SELECT TOP 1
        DefinitionID,
        EquipID,
        RegimeVersion,
        NumRegimes,
        Centroids,
        FeatureColumns,
        ScalerParams,
        TransitionMatrix,
        DiscoveryParams,
        TrainingRowCount,
        TrainingStartTime,
        TrainingEndTime,
        CreatedAt,
        CreatedBy
    FROM dbo.ACM_RegimeDefinitions
    WHERE EquipID = @EquipID
    ORDER BY RegimeVersion DESC;
END
GO

PRINT 'Created stored procedure usp_ACM_GetLatestRegimeVersion';
GO

-- ============================================================================
-- Add RegimeVersion column to ACM_RegimeTimeline (P2.6)
-- ============================================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.ACM_RegimeTimeline') 
    AND name = 'RegimeVersion'
)
BEGIN
    ALTER TABLE dbo.ACM_RegimeTimeline
    ADD RegimeVersion INT NULL;
    
    PRINT 'Added RegimeVersion column to ACM_RegimeTimeline';
END
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.ACM_RegimeTimeline') 
    AND name = 'AssignmentConfidence'
)
BEGIN
    ALTER TABLE dbo.ACM_RegimeTimeline
    ADD AssignmentConfidence FLOAT NULL;
    
    PRINT 'Added AssignmentConfidence column to ACM_RegimeTimeline';
END
GO

-- ============================================================================
-- Summary
-- ============================================================================
PRINT '';
PRINT '=== Migration Summary ===';
PRINT 'Table: ACM_RegimeDefinitions - Created/Verified';
PRINT 'Table: ACM_RegimeTimeline - Added RegimeVersion, AssignmentConfidence';
PRINT 'Stored Procedure: usp_ACM_GetLatestRegimeVersion - Created';
PRINT '=========================';
GO

-- scripts/sql/migrations/v11/010_acm_active_models.sql
-- ACM v11.0.0 - Active Models Pointer Table (P2.1)
--
-- Purpose: Single source of truth for which model versions are active in production.
-- This table ensures that all pipeline stages use consistent model versions.

-- Create ACM_ActiveModels table
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_ActiveModels')
BEGIN
    CREATE TABLE ACM_ActiveModels (
        EquipID INT PRIMARY KEY,
        
        -- Regime model versioning
        ActiveRegimeVersion INT NULL,  -- NULL = cold-start, no regimes
        RegimeMaturityState NVARCHAR(20) DEFAULT 'INITIALIZING',  -- INITIALIZING, LEARNING, CONVERGED, DEPRECATED
        RegimePromotedAt DATETIME2 NULL,
        
        -- Threshold versioning
        ActiveThresholdVersion INT NULL,
        ThresholdPromotedAt DATETIME2 NULL,
        
        -- Forecasting model versioning
        ActiveForecastVersion INT NULL,
        ForecastPromotedAt DATETIME2 NULL,
        
        -- Audit
        LastUpdatedAt DATETIME2 DEFAULT GETDATE(),
        LastUpdatedBy NVARCHAR(100) DEFAULT 'SYSTEM',
        
        CONSTRAINT FK_ActiveModels_Equipment FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
    );
    
    PRINT 'Created ACM_ActiveModels table';
END
ELSE
BEGIN
    PRINT 'ACM_ActiveModels table already exists';
END
GO

-- Add check constraint for valid maturity states
IF NOT EXISTS (SELECT * FROM sys.check_constraints WHERE name = 'CK_ActiveModels_MaturityState')
BEGIN
    ALTER TABLE ACM_ActiveModels
    ADD CONSTRAINT CK_ActiveModels_MaturityState 
    CHECK (RegimeMaturityState IN ('INITIALIZING', 'LEARNING', 'CONVERGED', 'DEPRECATED'));
    
    PRINT 'Added maturity state check constraint';
END
GO

-- Create index for faster lookups
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_ActiveModels_Maturity')
BEGIN
    CREATE INDEX IX_ActiveModels_Maturity 
    ON ACM_ActiveModels(RegimeMaturityState);
    
    PRINT 'Created maturity state index';
END
GO

-- Insert records for existing equipment (initialize as cold-start)
INSERT INTO ACM_ActiveModels (EquipID, RegimeMaturityState, LastUpdatedBy)
SELECT e.EquipID, 'INITIALIZING', 'MIGRATION_V11'
FROM Equipment e
WHERE NOT EXISTS (
    SELECT 1 FROM ACM_ActiveModels am WHERE am.EquipID = e.EquipID
);

PRINT 'Initialized ACM_ActiveModels for existing equipment';
GO

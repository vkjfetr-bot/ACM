-- ============================================================================
-- ACM_RegimePromotionLog Table Migration
-- Version: 11.0.0
-- Purpose: Audit trail for regime model promotion attempts
-- Phase: P2.10
-- ============================================================================

-- Create ACM_RegimePromotionLog table
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'dbo.ACM_RegimePromotionLog') AND type = 'U')
BEGIN
    CREATE TABLE dbo.ACM_RegimePromotionLog (
        -- Primary Key
        LogID               INT IDENTITY(1,1) PRIMARY KEY,
        
        -- Key fields
        EquipID             INT NOT NULL,
        RegimeVersion       INT NOT NULL,
        AttemptTime         DATETIME NOT NULL DEFAULT GETDATE(),
        Success             BIT NOT NULL DEFAULT 0,
        
        -- State transition
        FromState           NVARCHAR(32) NOT NULL,
        ToState             NVARCHAR(32) NOT NULL,
        
        -- Metrics at evaluation time
        Stability           FLOAT NULL,
        NoveltyRate         FLOAT NULL,
        Coverage            FLOAT NULL,
        Balance             FLOAT NULL,
        SampleCount         INT NULL,
        OverallScore        FLOAT NULL,
        
        -- Failure tracking
        FailureReasons      NVARCHAR(MAX) NULL,
        
        -- Audit
        TriggeredBy         NVARCHAR(128) NOT NULL DEFAULT 'SYSTEM',
        Notes               NVARCHAR(MAX) NULL,
        
        -- Foreign key to Equipment
        CONSTRAINT FK_RegimePromotionLog_Equipment FOREIGN KEY (EquipID)
            REFERENCES Equipment(EquipID)
    );
    
    PRINT 'Created table ACM_RegimePromotionLog';
END
ELSE
BEGIN
    PRINT 'Table ACM_RegimePromotionLog already exists';
END
GO

-- Create indexes for common queries
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_RegimePromotionLog_EquipID_AttemptTime')
BEGIN
    CREATE NONCLUSTERED INDEX IX_RegimePromotionLog_EquipID_AttemptTime
    ON dbo.ACM_RegimePromotionLog (EquipID, AttemptTime DESC);
    
    PRINT 'Created index IX_RegimePromotionLog_EquipID_AttemptTime';
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_RegimePromotionLog_Success')
BEGIN
    CREATE NONCLUSTERED INDEX IX_RegimePromotionLog_Success
    ON dbo.ACM_RegimePromotionLog (Success)
    INCLUDE (EquipID, RegimeVersion, AttemptTime);
    
    PRINT 'Created index IX_RegimePromotionLog_Success';
END
GO

-- ============================================================================
-- Summary
-- ============================================================================
PRINT '';
PRINT '=== Migration Summary ===';
PRINT 'Table: ACM_RegimePromotionLog - Created/Verified';
PRINT 'Indexes: IX_RegimePromotionLog_EquipID_AttemptTime, IX_RegimePromotionLog_Success';
PRINT '=========================';
GO

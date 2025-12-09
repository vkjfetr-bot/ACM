-- Migration: Add health quality tracking columns
-- Purpose: Support data quality monitoring and health smoothing validation
-- Related to: Anomaly detection validation branch - health volatility fixes

USE ACM;
GO

-- Add RawHealthIndex column (unsmoothed health for comparison)
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'ACM_HealthTimeline' 
    AND COLUMN_NAME = 'RawHealthIndex'
)
BEGIN
    ALTER TABLE ACM_HealthTimeline
    ADD RawHealthIndex FLOAT NULL;
    
    PRINT 'Added RawHealthIndex column to ACM_HealthTimeline';
END
ELSE
BEGIN
    PRINT 'RawHealthIndex column already exists';
END
GO

-- Add QualityFlag column (data quality indicator)
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'ACM_HealthTimeline' 
    AND COLUMN_NAME = 'QualityFlag'
)
BEGIN
    ALTER TABLE ACM_HealthTimeline
    ADD QualityFlag NVARCHAR(50) NULL;
    
    PRINT 'Added QualityFlag column to ACM_HealthTimeline';
END
ELSE
BEGIN
    PRINT 'QualityFlag column already exists';
END
GO

-- Add check constraint for valid quality flags (separate batch after column creation)
IF NOT EXISTS (
    SELECT 1 FROM sys.check_constraints 
    WHERE name = 'CK_HealthTimeline_QualityFlag'
)
BEGIN
    ALTER TABLE ACM_HealthTimeline
    ADD CONSTRAINT CK_HealthTimeline_QualityFlag 
    CHECK (QualityFlag IN ('NORMAL', 'VOLATILE', 'EXTREME_ANOMALY', 'MISSING_DATA', 'COLDSTART'));
    
    PRINT 'Added quality flag constraint';
END
GO

-- Backfill RawHealthIndex for existing records (set equal to HealthIndex as best estimate)
UPDATE ACM_HealthTimeline
SET RawHealthIndex = HealthIndex
WHERE RawHealthIndex IS NULL;

PRINT 'Backfilled RawHealthIndex for existing records';
GO

-- Create index on QualityFlag for quality monitoring queries
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_ACM_HealthTimeline_QualityFlag' 
    AND object_id = OBJECT_ID('ACM_HealthTimeline')
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_ACM_HealthTimeline_QualityFlag
    ON ACM_HealthTimeline(EquipID, QualityFlag, Timestamp);
    
    PRINT 'Created index on QualityFlag for monitoring queries';
END
GO

-- Update existing records to have NORMAL quality flag
UPDATE ACM_HealthTimeline
SET QualityFlag = 'NORMAL'
WHERE QualityFlag IS NULL;

PRINT 'Migration complete: Health quality tracking columns added';
GO

-- Verification query
SELECT 
    'ACM_HealthTimeline' AS TableName,
    COUNT(*) AS TotalRecords,
    COUNT(RawHealthIndex) AS RecordsWithRawHealth,
    COUNT(QualityFlag) AS RecordsWithQuality,
    COUNT(DISTINCT QualityFlag) AS UniqueQualityFlags
FROM ACM_HealthTimeline;
GO

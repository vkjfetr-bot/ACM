-- V11 Schema Additions
-- Adds new columns for confidence model, reliability status, and maturity state
-- Run after deploying v11.0.0 code
-- Safe to run multiple times (uses IF NOT EXISTS pattern)

USE ACM;
GO

PRINT 'V11 Schema Migration: Adding confidence and reliability columns...';
GO

-- ============================================================================
-- ACM_RUL: Add RUL_Status and MaturityState columns
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RUL') AND name = 'RUL_Status')
BEGIN
    ALTER TABLE dbo.ACM_RUL ADD RUL_Status NVARCHAR(50) NULL;
    PRINT '  Added ACM_RUL.RUL_Status';
END
ELSE
    PRINT '  ACM_RUL.RUL_Status already exists';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RUL') AND name = 'MaturityState')
BEGIN
    ALTER TABLE dbo.ACM_RUL ADD MaturityState NVARCHAR(50) NULL;
    PRINT '  Added ACM_RUL.MaturityState';
END
ELSE
    PRINT '  ACM_RUL.MaturityState already exists';
GO

-- ============================================================================
-- ACM_HealthTimeline: Add Confidence column
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_HealthTimeline') AND name = 'Confidence')
BEGIN
    ALTER TABLE dbo.ACM_HealthTimeline ADD Confidence FLOAT NULL;
    PRINT '  Added ACM_HealthTimeline.Confidence';
END
ELSE
    PRINT '  ACM_HealthTimeline.Confidence already exists';
GO

-- ============================================================================
-- ACM_Anomaly_Events: Add Confidence column
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_Anomaly_Events') AND name = 'Confidence')
BEGIN
    ALTER TABLE dbo.ACM_Anomaly_Events ADD Confidence FLOAT NULL;
    PRINT '  Added ACM_Anomaly_Events.Confidence';
END
ELSE
    PRINT '  ACM_Anomaly_Events.Confidence already exists';
GO

-- ============================================================================
-- ACM_RegimeTimeline: Add AssignmentConfidence column
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RegimeTimeline') AND name = 'AssignmentConfidence')
BEGIN
    ALTER TABLE dbo.ACM_RegimeTimeline ADD AssignmentConfidence FLOAT NULL;
    PRINT '  Added ACM_RegimeTimeline.AssignmentConfidence';
END
ELSE
    PRINT '  ACM_RegimeTimeline.AssignmentConfidence already exists';
GO

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RegimeTimeline') AND name = 'RegimeVersion')
BEGIN
    ALTER TABLE dbo.ACM_RegimeTimeline ADD RegimeVersion INT NULL;
    PRINT '  Added ACM_RegimeTimeline.RegimeVersion';
END
ELSE
    PRINT '  ACM_RegimeTimeline.RegimeVersion already exists';
GO

-- ============================================================================
-- ACM_RegimeState: Add MaturityState column for regime lifecycle
-- ============================================================================
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.ACM_RegimeState') AND name = 'RegimeMaturityState')
BEGIN
    ALTER TABLE dbo.ACM_RegimeState ADD RegimeMaturityState NVARCHAR(50) NULL DEFAULT 'LEARNING';
    PRINT '  Added ACM_RegimeState.RegimeMaturityState';
END
ELSE
    PRINT '  ACM_RegimeState.RegimeMaturityState already exists';
GO

-- ============================================================================
-- Summary
-- ============================================================================
PRINT '';
PRINT 'V11 Schema Migration Complete.';
PRINT 'New columns support:';
PRINT '  - RUL reliability gating (RUL_Status: RELIABLE, NOT_RELIABLE, LEARNING, INSUFFICIENT_DATA)';
PRINT '  - Model maturity tracking (MaturityState: COLDSTART, LEARNING, CONVERGED, DEPRECATED)';
PRINT '  - Confidence scores for health, anomaly, and regime outputs';
PRINT '';
GO

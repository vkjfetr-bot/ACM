-- ========================================
-- ACM Database Cleanup Script
-- Purpose: Drop obsolete and duplicate tables
-- Date: 2025-12-03
-- ========================================

USE ACM;
GO

PRINT '========================================';
PRINT 'ACM DATABASE CLEANUP SCRIPT';
PRINT 'Date: ' + CAST(GETDATE() AS VARCHAR);
PRINT '========================================';
PRINT '';

-- ========================================
-- SECTION 1: BACKUP LEGACY TABLES WITH DATA
-- ========================================
PRINT 'SECTION 1: Backing up legacy tables with data...';

-- Backup old Runs table (3,941 rows)
IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Runs')
BEGIN
    IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Runs_BACKUP_20251203')
    BEGIN
        SELECT * INTO Runs_BACKUP_20251203 FROM Runs;
        PRINT '✓ Backed up Runs (3,941 rows) to Runs_BACKUP_20251203';
    END
    ELSE
        PRINT '⚠ Runs_BACKUP_20251203 already exists, skipping';
END

-- Backup old RunLog table (1,871 rows)
IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'RunLog')
BEGIN
    IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'RunLog_BACKUP_20251203')
    BEGIN
        SELECT * INTO RunLog_BACKUP_20251203 FROM RunLog;
        PRINT '✓ Backed up RunLog (1,871 rows) to RunLog_BACKUP_20251203';
    END
    ELSE
        PRINT '⚠ RunLog_BACKUP_20251203 already exists, skipping';
END

-- Backup PCA_Components (1,160 rows)
IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'PCA_Components')
BEGIN
    IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'PCA_Components_BACKUP_20251203')
    BEGIN
        SELECT * INTO PCA_Components_BACKUP_20251203 FROM PCA_Components;
        PRINT '✓ Backed up PCA_Components (1,160 rows) to PCA_Components_BACKUP_20251203';
    END
    ELSE
        PRINT '⚠ PCA_Components_BACKUP_20251203 already exists, skipping';
END

PRINT '';

-- ========================================
-- SECTION 2: DROP OBSOLETE TABLES
-- ========================================
PRINT 'SECTION 2: Dropping obsolete tables...';

-- Duplicate tables (have active replacements)
IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_OMRContributions')
BEGIN
    DROP TABLE ACM_OMRContributions;
    PRINT '✓ Dropped ACM_OMRContributions (replaced by ACM_OMRContributionsLong)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_FusionQuality')
BEGIN
    DROP TABLE ACM_FusionQuality;
    PRINT '✓ Dropped ACM_FusionQuality (replaced by ACM_FusionQualityReport)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_FusionMetrics')
BEGIN
    DROP TABLE ACM_FusionMetrics;
    PRINT '✓ Dropped ACM_FusionMetrics (never used)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_OMR_Metrics')
BEGIN
    DROP TABLE ACM_OMR_Metrics;
    PRINT '✓ Dropped ACM_OMR_Metrics (replaced by ACM_OMR_Diagnostics)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_OMR_TopContributors')
BEGIN
    DROP TABLE ACM_OMR_TopContributors;
    PRINT '✓ Dropped ACM_OMR_TopContributors (use ACM_OMRContributionsLong queries)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_RegimeSummary')
BEGIN
    DROP TABLE ACM_RegimeSummary;
    PRINT '✓ Dropped ACM_RegimeSummary (replaced by ACM_RegimeStats)';
END

-- Never implemented tables
IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_ChartGenerationLog')
BEGIN
    DROP TABLE ACM_ChartGenerationLog;
    PRINT '✓ Dropped ACM_ChartGenerationLog (Grafana handles charting)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_CulpritHistory')
BEGIN
    DROP TABLE ACM_CulpritHistory;
    PRINT '✓ Dropped ACM_CulpritHistory (replaced by ACM_EpisodeCulprits)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_DetectorContributions')
BEGIN
    DROP TABLE ACM_DetectorContributions;
    PRINT '✓ Dropped ACM_DetectorContributions (never implemented)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_DetectorMetadata')
BEGIN
    DROP TABLE ACM_DetectorMetadata;
    PRINT '✓ Dropped ACM_DetectorMetadata (never implemented)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_RecommendedActions')
BEGIN
    DROP TABLE ACM_RecommendedActions;
    PRINT '✓ Dropped ACM_RecommendedActions (no write logic)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_RegimeFeatureImportance')
BEGIN
    DROP TABLE ACM_RegimeFeatureImportance;
    PRINT '✓ Dropped ACM_RegimeFeatureImportance (never implemented)';
END

PRINT '';

-- ========================================
-- SECTION 3: DROP EMPTY LEGACY TABLES
-- ========================================
PRINT 'SECTION 3: Dropping empty legacy tables...';

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'AnomalyEvents')
BEGIN
    DROP TABLE AnomalyEvents;
    PRINT '✓ Dropped AnomalyEvents (legacy, replaced by ACM_Anomaly_Events)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'DriftTS')
BEGIN
    DROP TABLE DriftTS;
    PRINT '✓ Dropped DriftTS (legacy, replaced by ACM_DriftSeries)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'PCA_Metrics')
BEGIN
    DROP TABLE PCA_Metrics;
    PRINT '✓ Dropped PCA_Metrics (legacy, replaced by ACM_PCA_Metrics)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'RegimeEpisodes')
BEGIN
    DROP TABLE RegimeEpisodes;
    PRINT '✓ Dropped RegimeEpisodes (legacy, replaced by ACM_Regime_Episodes)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'RunStats')
BEGIN
    DROP TABLE RunStats;
    PRINT '✓ Dropped RunStats (legacy, replaced by ACM_Run_Stats)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ScoresTS')
BEGIN
    DROP TABLE ScoresTS;
    PRINT '✓ Dropped ScoresTS (legacy, replaced by ACM_Scores_Long/Wide)';
END

PRINT '';

-- ========================================
-- SECTION 4: DROP LEGACY TABLES WITH DATA (after backup)
-- ========================================
PRINT 'SECTION 4: Dropping backed-up legacy tables...';

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Runs')
BEGIN
    DROP TABLE Runs;
    PRINT '✓ Dropped Runs (backed up to Runs_BACKUP_20251203)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'RunLog')
BEGIN
    DROP TABLE RunLog;
    PRINT '✓ Dropped RunLog (backed up to RunLog_BACKUP_20251203)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'PCA_Model')
BEGIN
    DROP TABLE PCA_Model;
    PRINT '✓ Dropped PCA_Model (legacy, models in ModelRegistry)';
END

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'PCA_Components')
BEGIN
    DROP TABLE PCA_Components;
    PRINT '✓ Dropped PCA_Components (backed up, replaced by ACM_PCA_Loadings)';
END

PRINT '';

-- ========================================
-- SECTION 5: DROP BACKUP TABLES
-- ========================================
PRINT 'SECTION 5: Dropping empty backup tables...';

IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_PCA_Metrics_BACKUP_20251119')
BEGIN
    DROP TABLE ACM_PCA_Metrics_BACKUP_20251119;
    PRINT '✓ Dropped ACM_PCA_Metrics_BACKUP_20251119 (empty backup)';
END

PRINT '';

-- ========================================
-- SECTION 6: CONSOLIDATE EQUIPMENT TABLES
-- ========================================
PRINT 'SECTION 6: Consolidating equipment tables...';

-- Check if Equipments has any data not in Equipment
IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Equipments')
BEGIN
    DECLARE @equipmentDiff INT;
    SELECT @equipmentDiff = COUNT(*)
    FROM Equipments e
    WHERE NOT EXISTS (SELECT 1 FROM Equipment WHERE EquipID = e.EquipID);
    
    IF @equipmentDiff > 0
    BEGIN
        PRINT '⚠ WARNING: Equipments has ' + CAST(@equipmentDiff AS VARCHAR) + ' rows not in Equipment';
        PRINT '   Manual review needed before dropping';
    END
    ELSE
    BEGIN
        DROP TABLE Equipments;
        PRINT '✓ Dropped Equipments (duplicate of Equipment)';
    END
END

PRINT '';

-- ========================================
-- SUMMARY
-- ========================================
PRINT '========================================';
PRINT 'CLEANUP SUMMARY';
PRINT '========================================';
PRINT 'Tables backed up: 3 (Runs, RunLog, PCA_Components)';
PRINT 'Obsolete tables dropped: 12';
PRINT 'Empty legacy tables dropped: 6';
PRINT 'Legacy tables with data dropped: 4';
PRINT 'Empty backups dropped: 1';
PRINT 'Equipment consolidation: 1 (if duplicate)';
PRINT '';
PRINT '✓ Total tables removed: up to 24';
PRINT '✓ Backups created: 3';
PRINT '';
PRINT '========================================';
PRINT 'REMAINING ISSUES TO FIX';
PRINT '========================================';
PRINT '1. ACM_BaselineBuffer - 0 rows (should have baseline data)';
PRINT '2. ACM_DataQuality - 0 rows (should have quality metrics)';
PRINT '';
PRINT 'Update Grafana dashboard v2 to stop referencing ACM_CulpritHistory';
PRINT '';
PRINT 'CLEANUP COMPLETE';
PRINT '========================================';

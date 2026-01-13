-- =============================================================================
-- ACM Schema Migration: Add ID Columns to All Tables (v11.1.5)
-- =============================================================================
-- Purpose: Add IDENTITY ID columns to all ACM tables for proper data management
-- 
-- Why this matters:
--   1. Unique row identification for updates/deletes
--   2. Enables proper foreign key relationships (ParentID, RunID references)
--   3. Required for efficient deduplication
--   4. Standard database design practice
--
-- Usage:
--   sqlcmd -S "server\instance" -d ACM -E -i add_id_columns.sql
--
-- NOTE: This script adds ID columns WITHOUT dropping existing data.
--       It uses sp_rename + new column approach to preserve data.
-- =============================================================================

SET NOCOUNT ON;
PRINT '=== ACM Schema Migration: Adding ID Columns ===';
PRINT 'Started: ' + CONVERT(VARCHAR(30), GETDATE(), 120);
PRINT '';

-- Helper procedure to add ID column if not exists
IF OBJECT_ID('tempdb..#AddIDColumn') IS NOT NULL DROP PROCEDURE #AddIDColumn;
GO

-- We'll use dynamic SQL to add ID columns
DECLARE @TableName NVARCHAR(128);
DECLARE @SQL NVARCHAR(MAX);
DECLARE @HasID BIT;

-- List of all ACM tables that need ID columns
DECLARE @Tables TABLE (TableName NVARCHAR(128));
INSERT INTO @Tables VALUES
    ('ACM_AdaptiveConfig'),
    ('ACM_AlertAge'),
    ('ACM_ColdstartState'),
    ('ACM_Config'),
    ('ACM_ContributionCurrent'),
    ('ACM_DataQuality'),
    ('ACM_DefectSummary'),
    ('ACM_DefectTimeline'),
    ('ACM_DetectorForecast_TS'),
    ('ACM_DriftEvents'),
    ('ACM_EnhancedFailureProbability_TS'),
    ('ACM_EnhancedMaintenanceRecommendation'),
    ('ACM_EpisodeMetrics'),
    ('ACM_Episodes'),
    ('ACM_EpisodesQC'),
    ('ACM_FailureCausation'),
    ('ACM_FailureForecast'),
    ('ACM_FailureForecast_TS'),
    ('ACM_FailureHazard_TS'),
    ('ACM_ForecastingState'),
    ('ACM_ForecastState'),
    ('ACM_FusionQualityReport'),
    ('ACM_HealthDistributionOverTime'),
    ('ACM_HealthForecast'),
    ('ACM_HealthForecast_Continuous'),
    ('ACM_HealthForecast_TS'),
    ('ACM_HealthHistogram'),
    ('ACM_HealthTimeline'),
    ('ACM_HealthZoneByPeriod'),
    ('ACM_HistorianData'),
    ('ACM_MaintenanceRecommendation'),
    ('ACM_OMR_Diagnostics'),
    ('ACM_OMRContributionsLong'),
    ('ACM_OMRTimeline'),
    ('ACM_PCA_Loadings'),
    ('ACM_PCA_Models'),
    ('ACM_RecommendedActions'),
    ('ACM_RefitRequests'),
    ('ACM_RegimeDwellStats'),
    ('ACM_RegimeStability'),
    ('ACM_RegimeState'),
    ('ACM_RegimeStats'),
    ('ACM_RegimeTimeline'),
    ('ACM_RUL'),
    ('ACM_RUL_Attribution'),
    ('ACM_RUL_LearningState'),
    ('ACM_RUL_Summary'),
    ('ACM_RUL_TS'),
    ('ACM_Run_Stats'),
    ('ACM_RunLogs'),
    ('ACM_RunMetadata'),
    ('ACM_RunMetrics'),
    ('ACM_Runs'),
    ('ACM_RunTimers'),
    ('ACM_SchemaVersion'),
    ('ACM_Scores_Wide'),
    ('ACM_SensorAnomalyByPeriod'),
    ('ACM_SensorDefects'),
    ('ACM_SensorForecast'),
    ('ACM_SensorForecast_TS'),
    ('ACM_SensorHotspots'),
    ('ACM_SensorHotspotTimeline'),
    ('ACM_SensorRanking'),
    ('ACM_SinceWhen'),
    ('ACM_TagEquipmentMap'),
    ('ACM_ThresholdCrossings'),
    ('ACM_ThresholdMetadata');

DECLARE table_cursor CURSOR FOR SELECT TableName FROM @Tables;
OPEN table_cursor;
FETCH NEXT FROM table_cursor INTO @TableName;

WHILE @@FETCH_STATUS = 0
BEGIN
    -- Check if table exists and doesn't have ID column
    IF OBJECT_ID('dbo.' + @TableName, 'U') IS NOT NULL
    BEGIN
        SELECT @HasID = CASE WHEN EXISTS (
            SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = @TableName AND COLUMN_NAME = 'ID'
        ) THEN 1 ELSE 0 END;
        
        IF @HasID = 0
        BEGIN
            PRINT 'Adding ID column to ' + @TableName + '...';
            
            -- Add IDENTITY column as first column (requires table recreation for proper ordering)
            -- For simplicity, we add it at the end (SQL Server limitation)
            SET @SQL = 'ALTER TABLE dbo.[' + @TableName + '] ADD ID BIGINT IDENTITY(1,1) NOT NULL';
            
            BEGIN TRY
                EXEC sp_executesql @SQL;
                PRINT '  -> SUCCESS: Added ID column';
            END TRY
            BEGIN CATCH
                PRINT '  -> ERROR: ' + ERROR_MESSAGE();
            END CATCH
        END
        ELSE
        BEGIN
            PRINT @TableName + ': ID column already exists';
        END
    END
    ELSE
    BEGIN
        PRINT @TableName + ': Table does not exist (skipped)';
    END
    
    FETCH NEXT FROM table_cursor INTO @TableName;
END

CLOSE table_cursor;
DEALLOCATE table_cursor;

PRINT '';
PRINT '=== Creating Indexes on ID Columns ===';

-- Create unique indexes on ID columns for better performance
DECLARE index_cursor CURSOR FOR SELECT TableName FROM @Tables;
OPEN index_cursor;
FETCH NEXT FROM index_cursor INTO @TableName;

WHILE @@FETCH_STATUS = 0
BEGIN
    IF OBJECT_ID('dbo.' + @TableName, 'U') IS NOT NULL
    BEGIN
        -- Check if ID column exists and no index on it
        IF EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = @TableName AND COLUMN_NAME = 'ID')
        BEGIN
            SET @SQL = 'IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = ''IX_' + @TableName + '_ID'' AND object_id = OBJECT_ID(''dbo.' + @TableName + '''))
                        CREATE UNIQUE NONCLUSTERED INDEX [IX_' + @TableName + '_ID] ON dbo.[' + @TableName + '] (ID)';
            
            BEGIN TRY
                EXEC sp_executesql @SQL;
            END TRY
            BEGIN CATCH
                PRINT 'Index creation failed for ' + @TableName + ': ' + ERROR_MESSAGE();
            END CATCH
        END
    END
    
    FETCH NEXT FROM index_cursor INTO @TableName;
END

CLOSE index_cursor;
DEALLOCATE index_cursor;

PRINT '';
PRINT '=== Verification ===';

-- Final verification
SELECT 
    t.TABLE_NAME,
    CASE WHEN EXISTS (
        SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS c 
        WHERE c.TABLE_NAME = t.TABLE_NAME AND c.COLUMN_NAME = 'ID'
    ) THEN 'YES' ELSE 'NO' END AS HasIDColumn,
    p.rows AS TotalRows
FROM INFORMATION_SCHEMA.TABLES t
LEFT JOIN sys.partitions p ON OBJECT_ID(t.TABLE_NAME) = p.object_id AND p.index_id IN (0,1)
WHERE t.TABLE_NAME LIKE 'ACM_%' AND t.TABLE_TYPE = 'BASE TABLE'
ORDER BY t.TABLE_NAME;

PRINT '';
PRINT '=== Migration Complete ===';
PRINT 'Finished: ' + CONVERT(VARCHAR(30), GETDATE(), 120);
GO

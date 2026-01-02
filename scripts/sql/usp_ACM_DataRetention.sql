-- =============================================================================
-- ACM Data Retention Stored Procedure
-- =============================================================================
-- Purpose: Clean up old data from ACM tables to prevent unbounded growth
-- 
-- Retention Policies:
--   - ACM_SensorNormalized_TS: 30 days (used for sensor forecasting)
--   - ACM_SensorCorrelations: Latest run only per equipment (handled in code)
--   - ACM_Scores_Wide: 90 days (core analytics)
--   - ACM_HealthTimeline: 90 days (trending dashboards)
--   - ACM_RegimeTimeline: 90 days (regime analysis)
--   - ACM_RunTimers: 30 days (performance metrics)
--   - ACM_RunLogs: 14 days (debugging only)
--   - ACM_PCA_Loadings: Latest 5 runs per equipment
--   - ACM_FeatureDropLog: 30 days
--
-- Usage:
--   EXEC dbo.usp_ACM_DataRetention @DryRun = 1;  -- Preview what would be deleted
--   EXEC dbo.usp_ACM_DataRetention @DryRun = 0;  -- Actually delete
--   EXEC dbo.usp_ACM_DataRetention @DryRun = 0, @DeduplicateOnly = 1;  -- Only remove duplicates
--
-- Parameters:
--   @DryRun             - 1 = preview only, 0 = execute deletions
--   @DeduplicateOnly    - 1 = skip time-based retention, only remove duplicates
--                         (Use this for historical/test data to preserve all data)
--
-- Schedule: Run daily via SQL Agent job (production with live data)
-- =============================================================================

IF OBJECT_ID('dbo.usp_ACM_DataRetention', 'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_ACM_DataRetention;
GO

CREATE PROCEDURE dbo.usp_ACM_DataRetention
    @DryRun BIT = 1,  -- 1 = preview only, 0 = actually delete
    @RetentionDays_TimeSeries INT = 90,  -- For Scores_Wide, HealthTimeline, RegimeTimeline
    @RetentionDays_SensorTS INT = 30,    -- For SensorNormalized_TS
    @RetentionDays_Logs INT = 14,        -- For RunLogs
    @RetentionDays_Metrics INT = 30,     -- For RunTimers, FeatureDropLog
    @MaxRunsPerEquip_PCA INT = 5,        -- Keep latest N runs for PCA_Loadings
    @DeduplicateOnly BIT = 0             -- 1 = ONLY do deduplication, skip time-based retention
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @Cutoff_TimeSeries DATETIME2 = DATEADD(DAY, -@RetentionDays_TimeSeries, GETDATE());
    DECLARE @Cutoff_SensorTS DATETIME2 = DATEADD(DAY, -@RetentionDays_SensorTS, GETDATE());
    DECLARE @Cutoff_Logs DATETIME2 = DATEADD(DAY, -@RetentionDays_Logs, GETDATE());
    DECLARE @Cutoff_Metrics DATETIME2 = DATEADD(DAY, -@RetentionDays_Metrics, GETDATE());
    
    DECLARE @RowsToDelete INT;
    DECLARE @RowsDeleted INT = 0;
    DECLARE @TotalRowsDeleted INT = 0;
    
    PRINT '=== ACM Data Retention Cleanup ===';
    PRINT 'Mode: ' + CASE WHEN @DryRun = 1 THEN 'DRY RUN (preview only)' ELSE 'LIVE DELETE' END;
    IF @DeduplicateOnly = 1 PRINT 'DEDUPLICATE ONLY - skipping time-based retention';
    PRINT '';
    IF @DeduplicateOnly = 0
    BEGIN
        PRINT 'Cutoffs:';
        PRINT '  Time Series (90d): ' + CONVERT(VARCHAR(30), @Cutoff_TimeSeries, 120);
        PRINT '  Sensor TS (30d):   ' + CONVERT(VARCHAR(30), @Cutoff_SensorTS, 120);
        PRINT '  Logs (14d):        ' + CONVERT(VARCHAR(30), @Cutoff_Logs, 120);
        PRINT '  Metrics (30d):     ' + CONVERT(VARCHAR(30), @Cutoff_Metrics, 120);
        PRINT '';
    END
    
    -- =========================================================================
    -- TIME-BASED RETENTION (skip if @DeduplicateOnly = 1)
    -- =========================================================================
    IF @DeduplicateOnly = 0
    BEGIN
    -- =========================================================================
    -- ACM_SensorNormalized_TS (30 days) - HIGHEST PRIORITY
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_SensorNormalized_TS', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = COUNT(*) 
        FROM ACM_SensorNormalized_TS 
        WHERE CreatedAt < @Cutoff_SensorTS;
        
        PRINT 'ACM_SensorNormalized_TS: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            -- Delete in batches to avoid log bloat
            WHILE 1 = 1
            BEGIN
                DELETE TOP (50000) FROM ACM_SensorNormalized_TS 
                WHERE CreatedAt < @Cutoff_SensorTS;
                
                SET @RowsDeleted = @@ROWCOUNT;
                SET @TotalRowsDeleted = @TotalRowsDeleted + @RowsDeleted;
                
                IF @RowsDeleted = 0 BREAK;
                
                -- Checkpoint to release log space
                CHECKPOINT;
            END
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    -- =========================================================================
    -- ACM_SensorCorrelations - Keep only latest run per equipment
    -- (Already handled in code, but cleanup orphans here)
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_SensorCorrelations', 'U') IS NOT NULL
    BEGIN
        -- Delete all but latest run per equipment based on CreatedAt
        ;WITH LatestPerEquip AS (
            SELECT EquipID, MAX(CreatedAt) AS LatestCreatedAt
            FROM ACM_SensorCorrelations
            GROUP BY EquipID
        )
        SELECT @RowsToDelete = COUNT(*)
        FROM ACM_SensorCorrelations sc
        WHERE NOT EXISTS (
            SELECT 1 FROM LatestPerEquip lpe
            WHERE sc.EquipID = lpe.EquipID 
              AND sc.CreatedAt = lpe.LatestCreatedAt
        );
        
        PRINT 'ACM_SensorCorrelations: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' stale rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            SET @TotalRowsDeleted = 0;
            
            ;WITH LatestPerEquip AS (
                SELECT EquipID, MAX(CreatedAt) AS LatestCreatedAt
                FROM ACM_SensorCorrelations
                GROUP BY EquipID
            )
            DELETE FROM ACM_SensorCorrelations
            WHERE NOT EXISTS (
                SELECT 1 FROM LatestPerEquip lpe
                WHERE ACM_SensorCorrelations.EquipID = lpe.EquipID 
                  AND ACM_SensorCorrelations.CreatedAt = lpe.LatestCreatedAt
            );
            
            SET @TotalRowsDeleted = @@ROWCOUNT;
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    -- =========================================================================
    -- ACM_Scores_Wide (90 days) - Uses Timestamp column
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_Scores_Wide', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = COUNT(*) 
        FROM ACM_Scores_Wide 
        WHERE Timestamp < @Cutoff_TimeSeries;
        
        PRINT 'ACM_Scores_Wide: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            SET @TotalRowsDeleted = 0;
            WHILE 1 = 1
            BEGIN
                DELETE TOP (50000) FROM ACM_Scores_Wide 
                WHERE Timestamp < @Cutoff_TimeSeries;
                
                SET @RowsDeleted = @@ROWCOUNT;
                SET @TotalRowsDeleted = @TotalRowsDeleted + @RowsDeleted;
                IF @RowsDeleted = 0 BREAK;
                CHECKPOINT;
            END
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    -- =========================================================================
    -- ACM_HealthTimeline (90 days) - Uses Timestamp column
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_HealthTimeline', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = COUNT(*) 
        FROM ACM_HealthTimeline 
        WHERE Timestamp < @Cutoff_TimeSeries;
        
        PRINT 'ACM_HealthTimeline: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            SET @TotalRowsDeleted = 0;
            WHILE 1 = 1
            BEGIN
                DELETE TOP (50000) FROM ACM_HealthTimeline 
                WHERE Timestamp < @Cutoff_TimeSeries;
                
                SET @RowsDeleted = @@ROWCOUNT;
                SET @TotalRowsDeleted = @TotalRowsDeleted + @RowsDeleted;
                IF @RowsDeleted = 0 BREAK;
                CHECKPOINT;
            END
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    -- =========================================================================
    -- ACM_RegimeTimeline (90 days) - Uses Timestamp column
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_RegimeTimeline', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = COUNT(*) 
        FROM ACM_RegimeTimeline 
        WHERE Timestamp < @Cutoff_TimeSeries;
        
        PRINT 'ACM_RegimeTimeline: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            SET @TotalRowsDeleted = 0;
            WHILE 1 = 1
            BEGIN
                DELETE TOP (50000) FROM ACM_RegimeTimeline 
                WHERE Timestamp < @Cutoff_TimeSeries;
                
                SET @RowsDeleted = @@ROWCOUNT;
                SET @TotalRowsDeleted = @TotalRowsDeleted + @RowsDeleted;
                IF @RowsDeleted = 0 BREAK;
                CHECKPOINT;
            END
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    -- =========================================================================
    -- ACM_RunTimers (30 days)
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_RunTimers', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = COUNT(*) 
        FROM ACM_RunTimers 
        WHERE CreatedAt < @Cutoff_Metrics;
        
        PRINT 'ACM_RunTimers: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            DELETE FROM ACM_RunTimers WHERE CreatedAt < @Cutoff_Metrics;
            SET @TotalRowsDeleted = @@ROWCOUNT;
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    -- =========================================================================
    -- ACM_RunLogs (14 days) - Uses LoggedAt column
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_RunLogs', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = COUNT(*) 
        FROM ACM_RunLogs 
        WHERE LoggedAt < @Cutoff_Logs;
        
        PRINT 'ACM_RunLogs: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            DELETE FROM ACM_RunLogs WHERE LoggedAt < @Cutoff_Logs;
            SET @TotalRowsDeleted = @@ROWCOUNT;
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    -- =========================================================================
    -- ACM_FeatureDropLog (30 days)
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_FeatureDropLog', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = COUNT(*) 
        FROM ACM_FeatureDropLog 
        WHERE CreatedAt < @Cutoff_Metrics;
        
        PRINT 'ACM_FeatureDropLog: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            DELETE FROM ACM_FeatureDropLog WHERE CreatedAt < @Cutoff_Metrics;
            SET @TotalRowsDeleted = @@ROWCOUNT;
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    -- =========================================================================
    -- ACM_PCA_Loadings - Keep only latest N runs per equipment
    -- =========================================================================
    IF OBJECT_ID('dbo.ACM_PCA_Loadings', 'U') IS NOT NULL
    BEGIN
        -- Count rows from runs beyond the keep limit
        ;WITH RankedRuns AS (
            SELECT DISTINCT EquipID, RunID, MIN(CreatedAt) AS RunCreatedAt
            FROM ACM_PCA_Loadings
            GROUP BY EquipID, RunID
        ),
        RankedByEquip AS (
            SELECT EquipID, RunID, 
                   ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY RunCreatedAt DESC) AS RunRank
            FROM RankedRuns
        )
        SELECT @RowsToDelete = COUNT(*) 
        FROM ACM_PCA_Loadings pl
        WHERE EXISTS (
            SELECT 1 FROM RankedByEquip rbe
            WHERE pl.EquipID = rbe.EquipID AND pl.RunID = rbe.RunID AND rbe.RunRank > @MaxRunsPerEquip_PCA
        );
        
        PRINT 'ACM_PCA_Loadings: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete (keeping latest ' + CAST(@MaxRunsPerEquip_PCA AS VARCHAR(5)) + ' runs/equip)';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            ;WITH RankedRuns AS (
                SELECT DISTINCT EquipID, RunID, MIN(CreatedAt) AS RunCreatedAt
                FROM ACM_PCA_Loadings
                GROUP BY EquipID, RunID
            ),
            RankedByEquip AS (
                SELECT EquipID, RunID, 
                       ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY RunCreatedAt DESC) AS RunRank
                FROM RankedRuns
            )
            DELETE FROM ACM_PCA_Loadings
            WHERE EXISTS (
                SELECT 1 FROM RankedByEquip rbe
                WHERE ACM_PCA_Loadings.EquipID = rbe.EquipID 
                  AND ACM_PCA_Loadings.RunID = rbe.RunID 
                  AND rbe.RunRank > @MaxRunsPerEquip_PCA
            );
            
            SET @TotalRowsDeleted = @@ROWCOUNT;
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' rows';
        END
    END
    
    END  -- End of IF @DeduplicateOnly = 0 block
    
    -- =========================================================================
    -- DEDUPLICATION: Remove duplicate timestamps (v11.1.5 fix)
    -- =========================================================================
    -- These duplicates occur when overlapping batch runs write the same timestamps
    -- Keep only the most recent row (by RunID) for each EquipID+Timestamp
    PRINT '';
    PRINT '=== Deduplication (v11.1.5) ===';
    
    -- ACM_Scores_Wide duplicates (has ID column)
    IF OBJECT_ID('dbo.ACM_Scores_Wide', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = SUM(Cnt - 1)
        FROM (
            SELECT Timestamp, EquipID, COUNT(*) AS Cnt 
            FROM ACM_Scores_Wide 
            GROUP BY Timestamp, EquipID 
            HAVING COUNT(*) > 1
        ) x;
        
        SET @RowsToDelete = ISNULL(@RowsToDelete, 0);
        PRINT 'ACM_Scores_Wide duplicates: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            ;WITH Duplicates AS (
                SELECT ID, ROW_NUMBER() OVER (PARTITION BY EquipID, Timestamp ORDER BY ID DESC) AS RowNum
                FROM ACM_Scores_Wide
            )
            DELETE FROM ACM_Scores_Wide WHERE ID IN (SELECT ID FROM Duplicates WHERE RowNum > 1);
            
            SET @TotalRowsDeleted = @@ROWCOUNT;
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' duplicate rows';
        END
    END
    
    -- ACM_HealthTimeline duplicates (has ID column)
    IF OBJECT_ID('dbo.ACM_HealthTimeline', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = SUM(Cnt - 1)
        FROM (
            SELECT Timestamp, EquipID, COUNT(*) AS Cnt 
            FROM ACM_HealthTimeline 
            GROUP BY Timestamp, EquipID 
            HAVING COUNT(*) > 1
        ) x;
        
        SET @RowsToDelete = ISNULL(@RowsToDelete, 0);
        PRINT 'ACM_HealthTimeline duplicates: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            ;WITH Duplicates AS (
                SELECT ID, ROW_NUMBER() OVER (PARTITION BY EquipID, Timestamp ORDER BY ID DESC) AS RowNum
                FROM ACM_HealthTimeline
            )
            DELETE FROM ACM_HealthTimeline WHERE ID IN (SELECT ID FROM Duplicates WHERE RowNum > 1);
            
            SET @TotalRowsDeleted = @@ROWCOUNT;
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' duplicate rows';
        END
    END
    
    -- ACM_RegimeTimeline duplicates (has ID column)
    IF OBJECT_ID('dbo.ACM_RegimeTimeline', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = SUM(Cnt - 1)
        FROM (
            SELECT Timestamp, EquipID, COUNT(*) AS Cnt 
            FROM ACM_RegimeTimeline 
            GROUP BY Timestamp, EquipID 
            HAVING COUNT(*) > 1
        ) x;
        
        SET @RowsToDelete = ISNULL(@RowsToDelete, 0);
        PRINT 'ACM_RegimeTimeline duplicates: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            ;WITH Duplicates AS (
                SELECT ID, ROW_NUMBER() OVER (PARTITION BY EquipID, Timestamp ORDER BY ID DESC) AS RowNum
                FROM ACM_RegimeTimeline
            )
            DELETE FROM ACM_RegimeTimeline WHERE ID IN (SELECT ID FROM Duplicates WHERE RowNum > 1);
            
            SET @TotalRowsDeleted = @@ROWCOUNT;
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' duplicate rows';
        END
    END
    
    -- ACM_SensorNormalized_TS duplicates (has ID column, key: EquipID + Timestamp + SensorName)
    IF OBJECT_ID('dbo.ACM_SensorNormalized_TS', 'U') IS NOT NULL
    BEGIN
        SELECT @RowsToDelete = SUM(Cnt - 1)
        FROM (
            SELECT Timestamp, EquipID, SensorName, COUNT(*) AS Cnt 
            FROM ACM_SensorNormalized_TS 
            GROUP BY Timestamp, EquipID, SensorName
            HAVING COUNT(*) > 1
        ) x;
        
        SET @RowsToDelete = ISNULL(@RowsToDelete, 0);
        PRINT 'ACM_SensorNormalized_TS duplicates: ' + CAST(@RowsToDelete AS VARCHAR(20)) + ' rows to delete';
        
        IF @DryRun = 0 AND @RowsToDelete > 0
        BEGIN
            SET @TotalRowsDeleted = 0;
            -- Delete in batches for large table
            WHILE 1 = 1
            BEGIN
                ;WITH Duplicates AS (
                    SELECT ID, ROW_NUMBER() OVER (PARTITION BY EquipID, Timestamp, SensorName ORDER BY ID DESC) AS RowNum
                    FROM ACM_SensorNormalized_TS
                )
                DELETE TOP (50000) FROM ACM_SensorNormalized_TS 
                WHERE ID IN (SELECT ID FROM Duplicates WHERE RowNum > 1);
                
                SET @RowsDeleted = @@ROWCOUNT;
                SET @TotalRowsDeleted = @TotalRowsDeleted + @RowsDeleted;
                
                IF @RowsDeleted = 0 BREAK;
                CHECKPOINT;
            END
            PRINT '  -> Deleted ' + CAST(@TotalRowsDeleted AS VARCHAR(20)) + ' duplicate rows';
        END
    END
    
    PRINT '';
    PRINT '=== Cleanup Complete ===';
    
    -- Show current table sizes
    PRINT '';
    PRINT 'Current Table Sizes:';
    
    SELECT 
        t.NAME AS TableName,
        p.rows AS TotalRows,
        CAST(ROUND(((SUM(a.total_pages) * 8) / 1024.0), 2) AS DECIMAL(10,2)) AS TotalSpaceMB
    FROM sys.tables t
    INNER JOIN sys.indexes i ON t.OBJECT_ID = i.object_id
    INNER JOIN sys.partitions p ON i.object_id = p.OBJECT_ID AND i.index_id = p.index_id
    INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
    WHERE t.NAME IN (
        'ACM_SensorNormalized_TS', 'ACM_SensorCorrelations', 
        'ACM_Scores_Wide', 'ACM_HealthTimeline', 'ACM_RegimeTimeline',
        'ACM_RunTimers', 'ACM_RunLogs', 'ACM_FeatureDropLog', 'ACM_PCA_Loadings'
    )
    AND i.index_id <= 1
    GROUP BY t.Name, p.Rows
    ORDER BY p.rows DESC;
END
GO

PRINT 'Created stored procedure: dbo.usp_ACM_DataRetention';
PRINT '';
PRINT 'Usage:';
PRINT '  EXEC dbo.usp_ACM_DataRetention @DryRun = 1;  -- Preview deletions';
PRINT '  EXEC dbo.usp_ACM_DataRetention @DryRun = 0;  -- Execute deletions';
GO

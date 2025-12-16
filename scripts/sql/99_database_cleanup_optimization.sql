-- ============================================================================
-- ACM Database Cleanup and Optimization Script
-- Generated: 2025-12-15
-- 
-- CATEGORIES:
--   PHASE 1: Drop empty/deprecated tables (safe)
--   PHASE 2: Add missing indexes for performance
--   PHASE 3: Archive/truncate massive tables (requires decision)
--   PHASE 4: Statistics and maintenance
-- ============================================================================

SET NOCOUNT ON;
GO

PRINT '============================================================';
PRINT 'ACM DATABASE CLEANUP AND OPTIMIZATION';
PRINT '============================================================';
PRINT '';

-- ============================================================================
-- PHASE 1: DROP EMPTY/DEPRECATED TABLES
-- These tables have 0 rows and are either:
--   - Replaced by consolidated tables in v10.0.0
--   - Never-implemented features
-- ============================================================================

PRINT '=== PHASE 1: Dropping empty/deprecated tables ===';
PRINT '';

-- v10.0.0 consolidated these tables - originals are empty and deprecated
-- ACM_HealthForecast_TS -> ACM_HealthForecast
-- ACM_FailureForecast_TS -> ACM_FailureForecast  
-- ACM_RUL_TS -> ACM_RUL
-- ACM_RUL_Summary -> ACM_RUL
-- ACM_RUL_Attribution -> ACM_RUL
-- ACM_SensorForecast_TS -> ACM_SensorForecast
-- ACM_EnhancedFailureProbability_TS -> ACM_FailureForecast

-- Never-implemented features (0 rows, no code writes to them)
-- ACM_MaintenanceRecommendation
-- ACM_EnhancedMaintenanceRecommendation
-- ACM_RecommendedActions
-- ACM_FailureCausation
-- ACM_MultivariateForecast
-- ACM_SensorCorrelations

-- Tables in ALLOWED_TABLES but empty (keep for future use)
-- ACM_AdaptiveThresholds_ByRegime (v10.1.0 - regime-conditioned)
-- ACM_RegimeHazard (v10.1.0 - regime-conditioned)
-- ACM_ForecastContext (v10.1.0 - regime-conditioned)
-- ACM_RUL_ByRegime (v10.1.0 - regime-conditioned)

-- DROP deprecated _TS tables (replaced by consolidated tables)
IF OBJECT_ID('dbo.ACM_HealthForecast_TS', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_HealthForecast_TS (0 rows, replaced by ACM_HealthForecast)';
    DROP TABLE dbo.ACM_HealthForecast_TS;
END

IF OBJECT_ID('dbo.ACM_FailureForecast_TS', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_FailureForecast_TS (0 rows, replaced by ACM_FailureForecast)';
    DROP TABLE dbo.ACM_FailureForecast_TS;
END

IF OBJECT_ID('dbo.ACM_RUL_TS', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_RUL_TS (0 rows, replaced by ACM_RUL)';
    DROP TABLE dbo.ACM_RUL_TS;
END

IF OBJECT_ID('dbo.ACM_RUL_Summary', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_RUL_Summary (0 rows, replaced by ACM_RUL)';
    DROP TABLE dbo.ACM_RUL_Summary;
END

IF OBJECT_ID('dbo.ACM_RUL_Attribution', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_RUL_Attribution (0 rows, replaced by ACM_RUL)';
    DROP TABLE dbo.ACM_RUL_Attribution;
END

IF OBJECT_ID('dbo.ACM_SensorForecast_TS', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_SensorForecast_TS (0 rows, replaced by ACM_SensorForecast)';
    DROP TABLE dbo.ACM_SensorForecast_TS;
END

IF OBJECT_ID('dbo.ACM_EnhancedFailureProbability_TS', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_EnhancedFailureProbability_TS (0 rows, replaced by ACM_FailureForecast)';
    DROP TABLE dbo.ACM_EnhancedFailureProbability_TS;
END

-- DROP never-implemented feature tables
IF OBJECT_ID('dbo.ACM_MaintenanceRecommendation', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_MaintenanceRecommendation (0 rows, never implemented)';
    DROP TABLE dbo.ACM_MaintenanceRecommendation;
END

IF OBJECT_ID('dbo.ACM_EnhancedMaintenanceRecommendation', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_EnhancedMaintenanceRecommendation (0 rows, never implemented)';
    DROP TABLE dbo.ACM_EnhancedMaintenanceRecommendation;
END

IF OBJECT_ID('dbo.ACM_RecommendedActions', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_RecommendedActions (0 rows, never implemented)';
    DROP TABLE dbo.ACM_RecommendedActions;
END

IF OBJECT_ID('dbo.ACM_FailureCausation', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_FailureCausation (0 rows, never implemented)';
    DROP TABLE dbo.ACM_FailureCausation;
END

IF OBJECT_ID('dbo.ACM_MultivariateForecast', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_MultivariateForecast (0 rows, never implemented)';
    DROP TABLE dbo.ACM_MultivariateForecast;
END

IF OBJECT_ID('dbo.ACM_SensorCorrelations', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping ACM_SensorCorrelations (0 rows, never implemented)';
    DROP TABLE dbo.ACM_SensorCorrelations;
END

PRINT '';
PRINT 'Phase 1 complete: Dropped deprecated/empty tables';
PRINT '';

-- ============================================================================
-- PHASE 2: ADD MISSING INDEXES FOR PERFORMANCE
-- Based on common query patterns in Grafana dashboards
-- ============================================================================

PRINT '=== PHASE 2: Adding performance indexes ===';
PRINT '';

-- ACM_OMRContributionsLong - 39.7M rows, needs better indexes
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_OMRContrib_Equip_Ts' AND object_id = OBJECT_ID('ACM_OMRContributionsLong'))
BEGIN
    PRINT 'Creating IX_OMRContrib_Equip_Ts on ACM_OMRContributionsLong...';
    CREATE NONCLUSTERED INDEX IX_OMRContrib_Equip_Ts 
    ON dbo.ACM_OMRContributionsLong (EquipID, Timestamp DESC)
    INCLUDE (SensorName, ContributionPct, OMR_Z);
    PRINT 'Created IX_OMRContrib_Equip_Ts';
END

-- ACM_RunLogs - 598K rows, commonly filtered by RunID and Level
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunLogs_RunID_Level' AND object_id = OBJECT_ID('ACM_RunLogs'))
BEGIN
    PRINT 'Creating IX_RunLogs_RunID_Level on ACM_RunLogs...';
    CREATE NONCLUSTERED INDEX IX_RunLogs_RunID_Level 
    ON dbo.ACM_RunLogs (RunID, Level)
    INCLUDE (Message, LoggedAt);
    PRINT 'Created IX_RunLogs_RunID_Level';
END

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunLogs_EquipID_LoggedAt' AND object_id = OBJECT_ID('ACM_RunLogs'))
BEGIN
    PRINT 'Creating IX_RunLogs_EquipID_LoggedAt on ACM_RunLogs...';
    CREATE NONCLUSTERED INDEX IX_RunLogs_EquipID_LoggedAt 
    ON dbo.ACM_RunLogs (EquipID, LoggedAt DESC);
    PRINT 'Created IX_RunLogs_EquipID_LoggedAt';
END

-- ACM_HealthTimeline - Commonly filtered by EquipID and Timestamp for dashboards
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_HealthTimeline_Equip_Ts' AND object_id = OBJECT_ID('ACM_HealthTimeline'))
BEGIN
    PRINT 'Creating IX_HealthTimeline_Equip_Ts on ACM_HealthTimeline...';
    CREATE NONCLUSTERED INDEX IX_HealthTimeline_Equip_Ts 
    ON dbo.ACM_HealthTimeline (EquipID, Timestamp DESC)
    INCLUDE (HealthIndex, HealthZone, FusedZ);
    PRINT 'Created IX_HealthTimeline_Equip_Ts';
END

-- ACM_Scores_Wide - Time series queries need EquipID + Timestamp
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ScoresWide_Equip_Ts' AND object_id = OBJECT_ID('ACM_Scores_Wide'))
BEGIN
    PRINT 'Creating IX_ScoresWide_Equip_Ts on ACM_Scores_Wide...';
    CREATE NONCLUSTERED INDEX IX_ScoresWide_Equip_Ts 
    ON dbo.ACM_Scores_Wide (EquipID, Timestamp DESC)
    INCLUDE (fused, ar1_z, pca_spe_z, regime_label);
    PRINT 'Created IX_ScoresWide_Equip_Ts';
END

-- ACM_Runs - Dashboard queries for latest run
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Runs_Equip_Created' AND object_id = OBJECT_ID('ACM_Runs'))
BEGIN
    PRINT 'Creating IX_Runs_Equip_Created on ACM_Runs...';
    CREATE NONCLUSTERED INDEX IX_Runs_Equip_Created 
    ON dbo.ACM_Runs (EquipID, CreatedAt DESC)
    INCLUDE (DurationSeconds, HealthStatus, AvgHealthIndex);
    PRINT 'Created IX_Runs_Equip_Created';
END

-- ACM_HealthForecast - Time series forecast queries
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_HealthForecast_Equip_Ts' AND object_id = OBJECT_ID('ACM_HealthForecast'))
BEGIN
    PRINT 'Creating IX_HealthForecast_Equip_Ts on ACM_HealthForecast...';
    CREATE NONCLUSTERED INDEX IX_HealthForecast_Equip_Ts 
    ON dbo.ACM_HealthForecast (EquipID, ForecastTimestamp DESC)
    INCLUDE (PredictedHealth, ConfidenceLower, ConfidenceUpper);
    PRINT 'Created IX_HealthForecast_Equip_Ts';
END

-- ACM_RUL - Latest RUL lookup
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RUL_Equip_Created' AND object_id = OBJECT_ID('ACM_RUL'))
BEGIN
    PRINT 'Creating IX_RUL_Equip_Created on ACM_RUL...';
    CREATE NONCLUSTERED INDEX IX_RUL_Equip_Created 
    ON dbo.ACM_RUL (EquipID, CreatedAt DESC)
    INCLUDE (RUL_Hours, P10_LowerBound, P90_UpperBound, Confidence);
    PRINT 'Created IX_RUL_Equip_Created';
END

-- ACM_EpisodeDiagnostics - Episode lookups
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_EpisodeDiag_Equip_Peak' AND object_id = OBJECT_ID('ACM_EpisodeDiagnostics'))
BEGIN
    PRINT 'Creating IX_EpisodeDiag_Equip_Peak on ACM_EpisodeDiagnostics...';
    CREATE NONCLUSTERED INDEX IX_EpisodeDiag_Equip_Peak 
    ON dbo.ACM_EpisodeDiagnostics (EquipID, peak_timestamp DESC)
    INCLUDE (severity, peak_z, dominant_sensor);
    PRINT 'Created IX_EpisodeDiag_Equip_Peak';
END

-- ACM_Anomaly_Events - Event timeline queries
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_AnomalyEvents_Equip_Start' AND object_id = OBJECT_ID('ACM_Anomaly_Events'))
BEGIN
    PRINT 'Creating IX_AnomalyEvents_Equip_Start on ACM_Anomaly_Events...';
    CREATE NONCLUSTERED INDEX IX_AnomalyEvents_Equip_Start 
    ON dbo.ACM_Anomaly_Events (EquipID, StartTime DESC)
    INCLUDE (EndTime, Severity);
    PRINT 'Created IX_AnomalyEvents_Equip_Start';
END

-- ACM_SensorDefects - Defect lookups
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SensorDefects_Equip_Active' AND object_id = OBJECT_ID('ACM_SensorDefects'))
BEGIN
    PRINT 'Creating IX_SensorDefects_Equip_Active on ACM_SensorDefects...';
    CREATE NONCLUSTERED INDEX IX_SensorDefects_Equip_Active 
    ON dbo.ACM_SensorDefects (EquipID, ActiveDefect DESC)
    INCLUDE (DetectorType, Severity, CurrentZ);
    PRINT 'Created IX_SensorDefects_Equip_Active';
END

-- ACM_RunTimers - Performance dashboard queries
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunTimers_Equip_Section' AND object_id = OBJECT_ID('ACM_RunTimers'))
BEGIN
    PRINT 'Creating IX_RunTimers_Equip_Section on ACM_RunTimers...';
    CREATE NONCLUSTERED INDEX IX_RunTimers_Equip_Section 
    ON dbo.ACM_RunTimers (EquipID, Section)
    INCLUDE (DurationSeconds, CreatedAt);
    PRINT 'Created IX_RunTimers_Equip_Section';
END

PRINT '';
PRINT 'Phase 2 complete: Added performance indexes';
PRINT '';

-- ============================================================================
-- PHASE 3: DATA RETENTION (OPTIONAL - REQUIRES DECISION)
-- ACM_OMRContributionsLong is 7.1 GB (81% of database)
-- Consider archiving old data or implementing retention policy
-- ============================================================================

PRINT '=== PHASE 3: Data retention recommendations ===';
PRINT '';

-- Show data age distribution for largest tables
PRINT 'ACM_OMRContributionsLong data age distribution:';
SELECT 
    YEAR(CreatedAt) AS Year,
    MONTH(CreatedAt) AS Month,
    COUNT(*) AS Records,
    CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ACM_OMRContributionsLong) AS DECIMAL(5,2)) AS PctOfTotal
FROM ACM_OMRContributionsLong
GROUP BY YEAR(CreatedAt), MONTH(CreatedAt)
ORDER BY YEAR(CreatedAt), MONTH(CreatedAt);

PRINT '';
PRINT 'RECOMMENDATION: Consider implementing a retention policy:';
PRINT '  - Keep last 90 days of ACM_OMRContributionsLong (or archive to separate table)';
PRINT '  - Keep last 30 days of ACM_RunLogs';
PRINT '  - Keep last 30 days of ACM_SensorNormalized_TS';
PRINT '';
PRINT 'To implement 90-day retention for ACM_OMRContributionsLong (DESTRUCTIVE):';
PRINT '  DELETE FROM ACM_OMRContributionsLong WHERE CreatedAt < DATEADD(DAY, -90, GETDATE());';
PRINT '';

-- ============================================================================
-- PHASE 4: UPDATE STATISTICS
-- ============================================================================

PRINT '=== PHASE 4: Updating statistics ===';
PRINT '';

-- Update statistics on the most-queried tables
PRINT 'Updating statistics on ACM_HealthTimeline...';
UPDATE STATISTICS dbo.ACM_HealthTimeline;

PRINT 'Updating statistics on ACM_Scores_Wide...';
UPDATE STATISTICS dbo.ACM_Scores_Wide;

PRINT 'Updating statistics on ACM_Runs...';
UPDATE STATISTICS dbo.ACM_Runs;

PRINT 'Updating statistics on ACM_RUL...';
UPDATE STATISTICS dbo.ACM_RUL;

PRINT 'Updating statistics on ACM_EpisodeDiagnostics...';
UPDATE STATISTICS dbo.ACM_EpisodeDiagnostics;

PRINT '';
PRINT 'Phase 4 complete: Statistics updated';
PRINT '';

-- ============================================================================
-- SUMMARY
-- ============================================================================

PRINT '============================================================';
PRINT 'CLEANUP COMPLETE - SUMMARY';
PRINT '============================================================';

-- Show remaining table count and size
SELECT 
    COUNT(*) AS RemainingTables,
    CAST(SUM(a.total_pages) * 8 / 1024.0 AS DECIMAL(10,2)) AS TotalSizeMB
FROM sys.tables t
INNER JOIN sys.indexes i ON t.object_id = i.object_id
INNER JOIN sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
WHERE t.is_ms_shipped = 0 AND t.name LIKE 'ACM_%' AND i.index_id <= 1;

PRINT '';
PRINT 'Next steps:';
PRINT '  1. Review data retention recommendations above';
PRINT '  2. Consider archiving ACM_OMRContributionsLong (7.1 GB)';
PRINT '  3. Set up scheduled maintenance for statistics updates';
PRINT '  4. Monitor query performance in Grafana';
PRINT '';
GO

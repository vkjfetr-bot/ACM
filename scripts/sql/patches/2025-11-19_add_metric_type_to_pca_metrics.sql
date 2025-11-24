-- Patch: Add MetricType column to ACM_PCA_Metrics if missing
-- Date: 2025-11-19
-- Purpose: Resolve Grafana error "Invalid column name 'MetricType'" for PCA Explained Variance panel.
-- Safe approach: Because ACM_PCA_Metrics stores derived PCA metrics (recomputable),
-- we back up the existing table (if column missing), recreate with correct schema.
-- To apply: run in target SQL Server database used by ACM.
-- After patch: re-run ACM pipeline to repopulate PCA metrics.

SET NOCOUNT ON;

IF COL_LENGTH('dbo.ACM_PCA_Metrics', 'MetricType') IS NULL
BEGIN
    PRINT 'MetricType column missing - recreating ACM_PCA_Metrics with correct schema.';

    DECLARE @backupName sysname = 'ACM_PCA_Metrics_BACKUP_' + CONVERT(varchar(8), GETDATE(), 112);

    EXEC sp_rename 'dbo.ACM_PCA_Metrics', @backupName;  -- backup existing structure/data

    CREATE TABLE dbo.ACM_PCA_Metrics (
        RunID          uniqueidentifier NOT NULL,
        EquipID        int           NOT NULL,
        ComponentName  nvarchar(100) NOT NULL,
        MetricType     nvarchar(50)  NOT NULL,  -- 'VarianceRatio', 'CumulativeVariance', 'ComponentCount'
        Value          float         NOT NULL,
        Timestamp      datetime2     NOT NULL DEFAULT (SYSUTCDATETIME())
    );

    ALTER TABLE dbo.ACM_PCA_Metrics
        ADD CONSTRAINT PK_ACM_PCA_Metrics PRIMARY KEY CLUSTERED (RunID, EquipID, ComponentName, MetricType);

    PRINT 'Recreated ACM_PCA_Metrics. Please re-run ACM pipeline to regenerate PCA metrics.';
END
ELSE
BEGIN
    PRINT 'MetricType column already exists - no action needed.';
END

-- SQL Patch: Migrate ALL tables with RunID from bigint to uniqueidentifier
-- Purpose: Fix "Operand type clash" errors across the entire schema.
-- Tables covered: RunLog, ScoresTS, DriftTS, AnomalyEvents, RegimeEpisodes, PCA_Model, PCA_Components, PCA_Metrics, RunStats
-- Strategy:
-- 1. For each table, check if RunID is bigint.
-- 2. If so, add RunID_guid, populate with NEWID() (or map from RunLog if possible, but NEWID is safer for unlinked data), swap columns.
-- 3. Recreate PKs/Indexes as needed.

USE [ACM];
GO

SET NOCOUNT ON;

-------------------------------------------------------------------------------
-- Helper: Drop Constraint if exists
-------------------------------------------------------------------------------
-- (Logic inlined below for simplicity)

-------------------------------------------------------------------------------
-- 1. RunLog (Master)
-------------------------------------------------------------------------------
DECLARE @dt_runlog sysname;
SELECT @dt_runlog = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'RunLog' AND COLUMN_NAME = 'RunID';

IF @dt_runlog = 'bigint'
BEGIN
    PRINT 'Migrating RunLog...';
    ALTER TABLE dbo.RunLog ADD RunID_guid uniqueidentifier NULL;
    EXEC('UPDATE dbo.RunLog SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.RunLog ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;
    
    -- Drop PK
    DECLARE @pk_runlog sysname;
    SELECT @pk_runlog = kc.NAME FROM sys.key_constraints kc JOIN sys.tables t ON kc.parent_object_id = t.object_id WHERE t.name = 'RunLog' AND kc.type = 'PK';
    IF @pk_runlog IS NOT NULL EXEC('ALTER TABLE dbo.RunLog DROP CONSTRAINT ' + @pk_runlog);

    EXEC sp_rename 'dbo.RunLog.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.RunLog.RunID_guid', 'RunID', 'COLUMN';
    ALTER TABLE dbo.RunLog ADD CONSTRAINT PK_RunLog_RunID PRIMARY KEY CLUSTERED (RunID);
    PRINT 'RunLog migrated.';
END

-------------------------------------------------------------------------------
-- 2. ScoresTS
-------------------------------------------------------------------------------
DECLARE @dt_scores sysname;
SELECT @dt_scores = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'ScoresTS' AND COLUMN_NAME = 'RunID';

IF @dt_scores = 'bigint'
BEGIN
    PRINT 'Migrating ScoresTS...';
    ALTER TABLE dbo.ScoresTS ADD RunID_guid uniqueidentifier NULL;
    -- Try to link to RunLog if possible, else NEWID() (orphaned rows get new IDs, acceptable for legacy data)
    -- Actually, if we just migrated RunLog, we lost the link unless we use the backup column.
    -- BUT: ScoresTS is usually huge. Updating it might be slow.
    -- Strategy: If RunLog has backup, join on it.
    IF COL_LENGTH('dbo.RunLog', 'RunID_bigint_backup') IS NOT NULL
    BEGIN
        EXEC('UPDATE s SET s.RunID_guid = r.RunID FROM dbo.ScoresTS s JOIN dbo.RunLog r ON s.RunID = r.RunID_bigint_backup');
    END
    EXEC('UPDATE dbo.ScoresTS SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.ScoresTS ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;

    -- Drop Indexes on RunID
    DECLARE @idx_scores sysname;
    SELECT @idx_scores = name FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.ScoresTS') AND name = 'IX_Scores_RunID';
    IF @idx_scores IS NOT NULL DROP INDEX IX_Scores_RunID ON dbo.ScoresTS;

    EXEC sp_rename 'dbo.ScoresTS.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.ScoresTS.RunID_guid', 'RunID', 'COLUMN';
    CREATE INDEX IX_Scores_RunID ON dbo.ScoresTS(RunID);
    PRINT 'ScoresTS migrated.';
END

-------------------------------------------------------------------------------
-- 3. DriftTS
-------------------------------------------------------------------------------
DECLARE @dt_drift sysname;
SELECT @dt_drift = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'DriftTS' AND COLUMN_NAME = 'RunID';

IF @dt_drift = 'bigint'
BEGIN
    PRINT 'Migrating DriftTS...';
    ALTER TABLE dbo.DriftTS ADD RunID_guid uniqueidentifier NULL;
    IF COL_LENGTH('dbo.RunLog', 'RunID_bigint_backup') IS NOT NULL
    BEGIN
        EXEC('UPDATE s SET s.RunID_guid = r.RunID FROM dbo.DriftTS s JOIN dbo.RunLog r ON s.RunID = r.RunID_bigint_backup');
    END
    EXEC('UPDATE dbo.DriftTS SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.DriftTS ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;

    DECLARE @idx_drift sysname;
    SELECT @idx_drift = name FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.DriftTS') AND name = 'IX_Drift_RunID';
    IF @idx_drift IS NOT NULL DROP INDEX IX_Drift_RunID ON dbo.DriftTS;

    EXEC sp_rename 'dbo.DriftTS.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.DriftTS.RunID_guid', 'RunID', 'COLUMN';
    CREATE INDEX IX_Drift_RunID ON dbo.DriftTS(RunID);
    PRINT 'DriftTS migrated.';
END

-------------------------------------------------------------------------------
-- 4. AnomalyEvents
-------------------------------------------------------------------------------
DECLARE @dt_anom sysname;
SELECT @dt_anom = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'AnomalyEvents' AND COLUMN_NAME = 'RunID';

IF @dt_anom = 'bigint'
BEGIN
    PRINT 'Migrating AnomalyEvents...';
    ALTER TABLE dbo.AnomalyEvents ADD RunID_guid uniqueidentifier NULL;
    IF COL_LENGTH('dbo.RunLog', 'RunID_bigint_backup') IS NOT NULL
    BEGIN
        EXEC('UPDATE s SET s.RunID_guid = r.RunID FROM dbo.AnomalyEvents s JOIN dbo.RunLog r ON s.RunID = r.RunID_bigint_backup');
    END
    EXEC('UPDATE dbo.AnomalyEvents SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.AnomalyEvents ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;

    DECLARE @idx_anom sysname;
    SELECT @idx_anom = name FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.AnomalyEvents') AND name = 'IX_Anom_RunID';
    IF @idx_anom IS NOT NULL DROP INDEX IX_Anom_RunID ON dbo.AnomalyEvents;

    EXEC sp_rename 'dbo.AnomalyEvents.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.AnomalyEvents.RunID_guid', 'RunID', 'COLUMN';
    CREATE INDEX IX_Anom_RunID ON dbo.AnomalyEvents(RunID);
    PRINT 'AnomalyEvents migrated.';
END

-------------------------------------------------------------------------------
-- 5. RegimeEpisodes
-------------------------------------------------------------------------------
DECLARE @dt_reg sysname;
SELECT @dt_reg = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'RegimeEpisodes' AND COLUMN_NAME = 'RunID';

IF @dt_reg = 'bigint'
BEGIN
    PRINT 'Migrating RegimeEpisodes...';
    ALTER TABLE dbo.RegimeEpisodes ADD RunID_guid uniqueidentifier NULL;
    IF COL_LENGTH('dbo.RunLog', 'RunID_bigint_backup') IS NOT NULL
    BEGIN
        EXEC('UPDATE s SET s.RunID_guid = r.RunID FROM dbo.RegimeEpisodes s JOIN dbo.RunLog r ON s.RunID = r.RunID_bigint_backup');
    END
    EXEC('UPDATE dbo.RegimeEpisodes SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.RegimeEpisodes ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;

    DECLARE @idx_reg sysname;
    SELECT @idx_reg = name FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.RegimeEpisodes') AND name = 'IX_Regime_RunID';
    IF @idx_reg IS NOT NULL DROP INDEX IX_Regime_RunID ON dbo.RegimeEpisodes;

    EXEC sp_rename 'dbo.RegimeEpisodes.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.RegimeEpisodes.RunID_guid', 'RunID', 'COLUMN';
    CREATE INDEX IX_Regime_RunID ON dbo.RegimeEpisodes(RunID);
    PRINT 'RegimeEpisodes migrated.';
END

-------------------------------------------------------------------------------
-- 6. PCA_Model
-------------------------------------------------------------------------------
DECLARE @dt_pca sysname;
SELECT @dt_pca = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'PCA_Model' AND COLUMN_NAME = 'RunID';

IF @dt_pca = 'bigint'
BEGIN
    PRINT 'Migrating PCA_Model...';
    ALTER TABLE dbo.PCA_Model ADD RunID_guid uniqueidentifier NULL;
    IF COL_LENGTH('dbo.RunLog', 'RunID_bigint_backup') IS NOT NULL
    BEGIN
        EXEC('UPDATE s SET s.RunID_guid = r.RunID FROM dbo.PCA_Model s JOIN dbo.RunLog r ON s.RunID = r.RunID_bigint_backup');
    END
    EXEC('UPDATE dbo.PCA_Model SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.PCA_Model ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;

    DECLARE @pk_pca sysname;
    SELECT @pk_pca = kc.NAME FROM sys.key_constraints kc JOIN sys.tables t ON kc.parent_object_id = t.object_id WHERE t.name = 'PCA_Model' AND kc.type = 'PK';
    IF @pk_pca IS NOT NULL EXEC('ALTER TABLE dbo.PCA_Model DROP CONSTRAINT ' + @pk_pca);

    EXEC sp_rename 'dbo.PCA_Model.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.PCA_Model.RunID_guid', 'RunID', 'COLUMN';
    ALTER TABLE dbo.PCA_Model ADD CONSTRAINT PK_PCA_Model PRIMARY KEY CLUSTERED (RunID);
    PRINT 'PCA_Model migrated.';
END

-------------------------------------------------------------------------------
-- 7. PCA_Components
-------------------------------------------------------------------------------
DECLARE @dt_pcac sysname;
SELECT @dt_pcac = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'PCA_Components' AND COLUMN_NAME = 'RunID';

IF @dt_pcac = 'bigint'
BEGIN
    PRINT 'Migrating PCA_Components...';
    ALTER TABLE dbo.PCA_Components ADD RunID_guid uniqueidentifier NULL;
    IF COL_LENGTH('dbo.RunLog', 'RunID_bigint_backup') IS NOT NULL
    BEGIN
        EXEC('UPDATE s SET s.RunID_guid = r.RunID FROM dbo.PCA_Components s JOIN dbo.RunLog r ON s.RunID = r.RunID_bigint_backup');
    END
    EXEC('UPDATE dbo.PCA_Components SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.PCA_Components ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;

    DECLARE @idx_pcac sysname;
    SELECT @idx_pcac = name FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.PCA_Components') AND name = 'IX_PCAComp_Run';
    IF @idx_pcac IS NOT NULL DROP INDEX IX_PCAComp_Run ON dbo.PCA_Components;

    EXEC sp_rename 'dbo.PCA_Components.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.PCA_Components.RunID_guid', 'RunID', 'COLUMN';
    CREATE INDEX IX_PCAComp_Run ON dbo.PCA_Components(RunID);
    PRINT 'PCA_Components migrated.';
END

-------------------------------------------------------------------------------
-- 8. PCA_Metrics
-------------------------------------------------------------------------------
DECLARE @dt_pcam sysname;
SELECT @dt_pcam = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'PCA_Metrics' AND COLUMN_NAME = 'RunID';

IF @dt_pcam = 'bigint'
BEGIN
    PRINT 'Migrating PCA_Metrics...';
    ALTER TABLE dbo.PCA_Metrics ADD RunID_guid uniqueidentifier NULL;
    IF COL_LENGTH('dbo.RunLog', 'RunID_bigint_backup') IS NOT NULL
    BEGIN
        EXEC('UPDATE s SET s.RunID_guid = r.RunID FROM dbo.PCA_Metrics s JOIN dbo.RunLog r ON s.RunID = r.RunID_bigint_backup');
    END
    EXEC('UPDATE dbo.PCA_Metrics SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.PCA_Metrics ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;

    DECLARE @idx_pcam sysname;
    SELECT @idx_pcam = name FROM sys.indexes WHERE object_id = OBJECT_ID('dbo.PCA_Metrics') AND name = 'IX_PCAMet_Run';
    IF @idx_pcam IS NOT NULL DROP INDEX IX_PCAMet_Run ON dbo.PCA_Metrics;

    EXEC sp_rename 'dbo.PCA_Metrics.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.PCA_Metrics.RunID_guid', 'RunID', 'COLUMN';
    CREATE INDEX IX_PCAMet_Run ON dbo.PCA_Metrics(RunID);
    PRINT 'PCA_Metrics migrated.';
END

-------------------------------------------------------------------------------
-- 9. RunStats
-------------------------------------------------------------------------------
DECLARE @dt_rs sysname;
SELECT @dt_rs = DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'RunStats' AND COLUMN_NAME = 'RunID';

IF @dt_rs = 'bigint'
BEGIN
    PRINT 'Migrating RunStats...';
    ALTER TABLE dbo.RunStats ADD RunID_guid uniqueidentifier NULL;
    IF COL_LENGTH('dbo.RunLog', 'RunID_bigint_backup') IS NOT NULL
    BEGIN
        EXEC('UPDATE s SET s.RunID_guid = r.RunID FROM dbo.RunStats s JOIN dbo.RunLog r ON s.RunID = r.RunID_bigint_backup');
    END
    EXEC('UPDATE dbo.RunStats SET RunID_guid = NEWID() WHERE RunID_guid IS NULL');
    ALTER TABLE dbo.RunStats ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;

    DECLARE @pk_rs sysname;
    SELECT @pk_rs = kc.NAME FROM sys.key_constraints kc JOIN sys.tables t ON kc.parent_object_id = t.object_id WHERE t.name = 'RunStats' AND kc.type = 'PK';
    IF @pk_rs IS NOT NULL EXEC('ALTER TABLE dbo.RunStats DROP CONSTRAINT ' + @pk_rs);

    EXEC sp_rename 'dbo.RunStats.RunID', 'RunID_bigint_backup', 'COLUMN';
    EXEC sp_rename 'dbo.RunStats.RunID_guid', 'RunID', 'COLUMN';
    ALTER TABLE dbo.RunStats ADD CONSTRAINT PK_RunStats PRIMARY KEY CLUSTERED (RunID);
    PRINT 'RunStats migrated.';
END

PRINT 'Comprehensive RunID migration completed.';
GO

-- ============================================================================
-- Script: fix_regime_runid_schema.sql
-- Purpose: Change RunID from uniqueidentifier to NVARCHAR(50) in regime tables
--          to match new RunID format: EQUIP_YYYYMMDD_HHMMSS
-- Author: ACM System
-- Date: 2025-01-24
-- ============================================================================

USE ConditionMonitoring;
GO

PRINT 'Starting regime table RunID schema migration...';
GO

-- ============================================================================
-- 1. ACM_RegimeTimeline
-- ============================================================================
IF EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ACM_RegimeTimeline')
BEGIN
    PRINT 'Migrating ACM_RegimeTimeline...';
    
    -- Drop existing primary key and indexes
    IF EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ACM_RegimeTimeline')
        ALTER TABLE dbo.ACM_RegimeTimeline DROP CONSTRAINT PK_ACM_RegimeTimeline;
    
    IF EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RegimeTimeline_EquipID_Time')
        DROP INDEX IX_RegimeTimeline_EquipID_Time ON dbo.ACM_RegimeTimeline;
    
    -- Create new table with corrected schema
    CREATE TABLE dbo.ACM_RegimeTimeline_New (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        Timestamp          datetime2(3) NOT NULL,
        RegimeLabel        int NULL,
        RegimeState        nvarchar(32) NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeTimeline_New_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeTimeline_New PRIMARY KEY CLUSTERED (RunID, Timestamp)
    );
    
    -- Copy data (convert uniqueidentifier to string)
    INSERT INTO dbo.ACM_RegimeTimeline_New (RunID, EquipID, Timestamp, RegimeLabel, RegimeState, CreatedAt)
    SELECT 
        CAST(RunID AS NVARCHAR(50)),
        EquipID,
        Timestamp,
        RegimeLabel,
        RegimeState,
        CreatedAt
    FROM dbo.ACM_RegimeTimeline;
    
    -- Drop old table and rename new
    DROP TABLE dbo.ACM_RegimeTimeline;
    EXEC sp_rename 'dbo.ACM_RegimeTimeline_New', 'ACM_RegimeTimeline';
    EXEC sp_rename 'dbo.PK_ACM_RegimeTimeline_New', 'PK_ACM_RegimeTimeline', 'OBJECT';
    EXEC sp_rename 'dbo.DF_ACM_RegimeTimeline_New_CreatedAt', 'DF_ACM_RegimeTimeline_CreatedAt', 'OBJECT';
    
    -- Recreate index
    CREATE NONCLUSTERED INDEX IX_RegimeTimeline_EquipID_Time ON dbo.ACM_RegimeTimeline(EquipID, Timestamp DESC);
    
    PRINT 'ACM_RegimeTimeline migration complete.';
END
ELSE
BEGIN
    PRINT 'ACM_RegimeTimeline does not exist. Creating with new schema...';
    
    CREATE TABLE dbo.ACM_RegimeTimeline (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        Timestamp          datetime2(3) NOT NULL,
        RegimeLabel        int NULL,
        RegimeState        nvarchar(32) NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeTimeline_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeTimeline PRIMARY KEY CLUSTERED (RunID, Timestamp)
    );
    CREATE NONCLUSTERED INDEX IX_RegimeTimeline_EquipID_Time ON dbo.ACM_RegimeTimeline(EquipID, Timestamp DESC);
END
GO

-- ============================================================================
-- 2. ACM_RegimeOccupancy
-- ============================================================================
IF EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ACM_RegimeOccupancy')
BEGIN
    PRINT 'Migrating ACM_RegimeOccupancy...';
    
    IF EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ACM_RegimeOccupancy')
        ALTER TABLE dbo.ACM_RegimeOccupancy DROP CONSTRAINT PK_ACM_RegimeOccupancy;
    
    IF EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RegimeOccupancy_EquipID')
        DROP INDEX IX_RegimeOccupancy_EquipID ON dbo.ACM_RegimeOccupancy;
    
    CREATE TABLE dbo.ACM_RegimeOccupancy_New (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        RegimeLabel        int NOT NULL,
        RecordCount        int NULL,
        Percentage         float NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeOccupancy_New_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeOccupancy_New PRIMARY KEY CLUSTERED (RunID, RegimeLabel)
    );
    
    INSERT INTO dbo.ACM_RegimeOccupancy_New (RunID, EquipID, RegimeLabel, RecordCount, Percentage, CreatedAt)
    SELECT 
        CAST(RunID AS NVARCHAR(50)),
        EquipID,
        RegimeLabel,
        RecordCount,
        Percentage,
        CreatedAt
    FROM dbo.ACM_RegimeOccupancy;
    
    DROP TABLE dbo.ACM_RegimeOccupancy;
    EXEC sp_rename 'dbo.ACM_RegimeOccupancy_New', 'ACM_RegimeOccupancy';
    EXEC sp_rename 'dbo.PK_ACM_RegimeOccupancy_New', 'PK_ACM_RegimeOccupancy', 'OBJECT';
    EXEC sp_rename 'dbo.DF_ACM_RegimeOccupancy_New_CreatedAt', 'DF_ACM_RegimeOccupancy_CreatedAt', 'OBJECT';
    
    CREATE NONCLUSTERED INDEX IX_RegimeOccupancy_EquipID ON dbo.ACM_RegimeOccupancy(EquipID, RunID);
    
    PRINT 'ACM_RegimeOccupancy migration complete.';
END
ELSE
BEGIN
    PRINT 'ACM_RegimeOccupancy does not exist. Creating with new schema...';
    
    CREATE TABLE dbo.ACM_RegimeOccupancy (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        RegimeLabel        int NOT NULL,
        RecordCount        int NULL,
        Percentage         float NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeOccupancy_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeOccupancy PRIMARY KEY CLUSTERED (RunID, RegimeLabel)
    );
    CREATE NONCLUSTERED INDEX IX_RegimeOccupancy_EquipID ON dbo.ACM_RegimeOccupancy(EquipID, RunID);
END
GO

-- ============================================================================
-- 3. ACM_RegimeTransitions
-- ============================================================================
IF EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ACM_RegimeTransitions')
BEGIN
    PRINT 'Migrating ACM_RegimeTransitions...';
    
    IF EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ACM_RegimeTransitions')
        ALTER TABLE dbo.ACM_RegimeTransitions DROP CONSTRAINT PK_ACM_RegimeTransitions;
    
    IF EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RegimeTransitions_EquipID')
        DROP INDEX IX_RegimeTransitions_EquipID ON dbo.ACM_RegimeTransitions;
    
    CREATE TABLE dbo.ACM_RegimeTransitions_New (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        FromLabel          int NOT NULL,
        ToLabel            int NOT NULL,
        TransitionCount    int NULL,
        Probability        float NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeTransitions_New_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeTransitions_New PRIMARY KEY CLUSTERED (RunID, FromLabel, ToLabel)
    );
    
    INSERT INTO dbo.ACM_RegimeTransitions_New (RunID, EquipID, FromLabel, ToLabel, TransitionCount, Probability, CreatedAt)
    SELECT 
        CAST(RunID AS NVARCHAR(50)),
        EquipID,
        FromLabel,
        ToLabel,
        TransitionCount,
        Probability,
        CreatedAt
    FROM dbo.ACM_RegimeTransitions;
    
    DROP TABLE dbo.ACM_RegimeTransitions;
    EXEC sp_rename 'dbo.ACM_RegimeTransitions_New', 'ACM_RegimeTransitions';
    EXEC sp_rename 'dbo.PK_ACM_RegimeTransitions_New', 'PK_ACM_RegimeTransitions', 'OBJECT';
    EXEC sp_rename 'dbo.DF_ACM_RegimeTransitions_New_CreatedAt', 'DF_ACM_RegimeTransitions_CreatedAt', 'OBJECT';
    
    CREATE NONCLUSTERED INDEX IX_RegimeTransitions_EquipID ON dbo.ACM_RegimeTransitions(EquipID, RunID);
    
    PRINT 'ACM_RegimeTransitions migration complete.';
END
ELSE
BEGIN
    PRINT 'ACM_RegimeTransitions does not exist. Creating with new schema...';
    
    CREATE TABLE dbo.ACM_RegimeTransitions (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        FromLabel          int NOT NULL,
        ToLabel            int NOT NULL,
        TransitionCount    int NULL,
        Probability        float NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeTransitions_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeTransitions PRIMARY KEY CLUSTERED (RunID, FromLabel, ToLabel)
    );
    CREATE NONCLUSTERED INDEX IX_RegimeTransitions_EquipID ON dbo.ACM_RegimeTransitions(EquipID, RunID);
END
GO

-- ============================================================================
-- 4. ACM_RegimeDwellStats
-- ============================================================================
IF EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ACM_RegimeDwellStats')
BEGIN
    PRINT 'Migrating ACM_RegimeDwellStats...';
    
    IF EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ACM_RegimeDwellStats')
        ALTER TABLE dbo.ACM_RegimeDwellStats DROP CONSTRAINT PK_ACM_RegimeDwellStats;
    
    IF EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RegimeDwellStats_EquipID')
        DROP INDEX IX_RegimeDwellStats_EquipID ON dbo.ACM_RegimeDwellStats;
    
    CREATE TABLE dbo.ACM_RegimeDwellStats_New (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        RegimeLabel        int NOT NULL,
        Runs               int NULL,
        MeanSeconds        float NULL,
        MedianSeconds      float NULL,
        MinSeconds         float NULL,
        MaxSeconds         float NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeDwellStats_New_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeDwellStats_New PRIMARY KEY CLUSTERED (RunID, RegimeLabel)
    );
    
    INSERT INTO dbo.ACM_RegimeDwellStats_New (RunID, EquipID, RegimeLabel, Runs, MeanSeconds, MedianSeconds, MinSeconds, MaxSeconds, CreatedAt)
    SELECT 
        CAST(RunID AS NVARCHAR(50)),
        EquipID,
        RegimeLabel,
        Runs,
        MeanSeconds,
        MedianSeconds,
        MinSeconds,
        MaxSeconds,
        CreatedAt
    FROM dbo.ACM_RegimeDwellStats;
    
    DROP TABLE dbo.ACM_RegimeDwellStats;
    EXEC sp_rename 'dbo.ACM_RegimeDwellStats_New', 'ACM_RegimeDwellStats';
    EXEC sp_rename 'dbo.PK_ACM_RegimeDwellStats_New', 'PK_ACM_RegimeDwellStats', 'OBJECT';
    EXEC sp_rename 'dbo.DF_ACM_RegimeDwellStats_New_CreatedAt', 'DF_ACM_RegimeDwellStats_CreatedAt', 'OBJECT';
    
    CREATE NONCLUSTERED INDEX IX_RegimeDwellStats_EquipID ON dbo.ACM_RegimeDwellStats(EquipID, RunID);
    
    PRINT 'ACM_RegimeDwellStats migration complete.';
END
ELSE
BEGIN
    PRINT 'ACM_RegimeDwellStats does not exist. Creating with new schema...';
    
    CREATE TABLE dbo.ACM_RegimeDwellStats (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        RegimeLabel        int NOT NULL,
        Runs               int NULL,
        MeanSeconds        float NULL,
        MedianSeconds      float NULL,
        MinSeconds         float NULL,
        MaxSeconds         float NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeDwellStats_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeDwellStats PRIMARY KEY CLUSTERED (RunID, RegimeLabel)
    );
    CREATE NONCLUSTERED INDEX IX_RegimeDwellStats_EquipID ON dbo.ACM_RegimeDwellStats(EquipID, RunID);
END
GO

-- ============================================================================
-- 5. ACM_RegimeStability
-- ============================================================================
IF EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ACM_RegimeStability')
BEGIN
    PRINT 'Migrating ACM_RegimeStability...';
    
    IF EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ACM_RegimeStability')
        ALTER TABLE dbo.ACM_RegimeStability DROP CONSTRAINT PK_ACM_RegimeStability;
    
    IF EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RegimeStability_EquipID')
        DROP INDEX IX_RegimeStability_EquipID ON dbo.ACM_RegimeStability;
    
    CREATE TABLE dbo.ACM_RegimeStability_New (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        MetricName         nvarchar(100) NOT NULL,
        MetricValue        float NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeStability_New_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeStability_New PRIMARY KEY CLUSTERED (RunID, EquipID, MetricName)
    );
    
    INSERT INTO dbo.ACM_RegimeStability_New (RunID, EquipID, MetricName, MetricValue, CreatedAt)
    SELECT 
        CAST(RunID AS NVARCHAR(50)),
        EquipID,
        MetricName,
        MetricValue,
        CreatedAt
    FROM dbo.ACM_RegimeStability;
    
    DROP TABLE dbo.ACM_RegimeStability;
    EXEC sp_rename 'dbo.ACM_RegimeStability_New', 'ACM_RegimeStability';
    EXEC sp_rename 'dbo.PK_ACM_RegimeStability_New', 'PK_ACM_RegimeStability', 'OBJECT';
    EXEC sp_rename 'dbo.DF_ACM_RegimeStability_New_CreatedAt', 'DF_ACM_RegimeStability_CreatedAt', 'OBJECT';
    
    CREATE NONCLUSTERED INDEX IX_RegimeStability_EquipID ON dbo.ACM_RegimeStability(EquipID, RunID);
    
    PRINT 'ACM_RegimeStability migration complete.';
END
ELSE
BEGIN
    PRINT 'ACM_RegimeStability does not exist. Creating with new schema...';
    
    CREATE TABLE dbo.ACM_RegimeStability (
        RunID              NVARCHAR(50) NOT NULL,
        EquipID            int NOT NULL,
        MetricName         nvarchar(100) NOT NULL,
        MetricValue        float NULL,
        CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeStability_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_RegimeStability PRIMARY KEY CLUSTERED (RunID, EquipID, MetricName)
    );
    CREATE NONCLUSTERED INDEX IX_RegimeStability_EquipID ON dbo.ACM_RegimeStability(EquipID, RunID);
END
GO

PRINT 'Regime table RunID schema migration complete!';
GO

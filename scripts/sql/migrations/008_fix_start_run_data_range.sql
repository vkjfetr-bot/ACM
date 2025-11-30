-- Migration: Fix usp_ACM_StartRun to query actual data range instead of hardcoded fallback
-- Problem: When no previous run exists, SP uses fixed "2023-10-15" or "now - 30 min" which may not have data
-- Solution: Query the actual equipment's data table MIN(EntryDateTime) when no previous run exists

USE ACM;
GO

-- Drop existing procedure
IF OBJECT_ID('dbo.usp_ACM_StartRun', 'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_ACM_StartRun;
GO

CREATE PROCEDURE dbo.usp_ACM_StartRun
    @EquipID     int,
    @ConfigHash  nvarchar(64)  = NULL,
    @WindowStartEntryDateTime datetime2(3)  = NULL,
    @WindowEndEntryDateTime   datetime2(3)  = NULL,
    @Stage       nvarchar(32) = N'started',
    @Version     nvarchar(32) = NULL,
    @TriggerReason nvarchar(64) = NULL,
    @TickMinutes int = 30,
    @DefaultStartUtc datetime2(3) = NULL,
    @RunID       uniqueidentifier OUTPUT,
    @EquipIDOut  int OUTPUT
AS
BEGIN
    SET NOCOUNT ON;
    IF @RunID IS NULL SET @RunID = NEWID();
    DECLARE @now datetime2(3) = SYSUTCDATETIME();
    DECLARE @EquipCode VARCHAR(50);
    DECLARE @TableName VARCHAR(100);
    DECLARE @SQL NVARCHAR(MAX);
    DECLARE @MinTimestamp DATETIME2(3);
    
    -- Set EquipIDOut for the caller
    SET @EquipIDOut = @EquipID;

    -- Get equipment code for this EquipID
    SELECT @EquipCode = EquipCode
    FROM dbo.Equipment
    WHERE EquipID = @EquipID;
    
    IF @EquipCode IS NULL
    BEGIN
        RAISERROR('EquipID %d not found in Equipment table', 16, 1, @EquipID);
        RETURN;
    END
    
    -- Construct data table name
    SET @TableName = @EquipCode + '_Data';

    -- 1. Determine Window Start
    -- If not provided, look for the last successful run's end time
    IF @WindowStartEntryDateTime IS NULL
    BEGIN
        SELECT TOP 1 @WindowStartEntryDateTime = WindowEndEntryDateTime
        FROM dbo.RunLog
        WHERE EquipID = @EquipID 
          AND Outcome = 'OK' 
          AND WindowEndEntryDateTime IS NOT NULL
        ORDER BY WindowEndEntryDateTime DESC;
        
        -- If no previous run, query the actual data table's MIN timestamp
        IF @WindowStartEntryDateTime IS NULL
        BEGIN
            -- Verify table exists
            IF OBJECT_ID('dbo.' + @TableName, 'U') IS NOT NULL
            BEGIN
                -- Dynamically query MIN(EntryDateTime) from equipment data table
                SET @SQL = N'SELECT @MinTimestamp = MIN(EntryDateTime) FROM dbo.' + QUOTENAME(@TableName);
                EXEC sp_executesql @SQL, N'@MinTimestamp DATETIME2(3) OUTPUT', @MinTimestamp = @MinTimestamp OUTPUT;
                
                IF @MinTimestamp IS NOT NULL
                BEGIN
                    SET @WindowStartEntryDateTime = @MinTimestamp;
                END
                ELSE
                BEGIN
                    -- Table exists but is empty - use DefaultStartUtc or current time
                    SET @WindowStartEntryDateTime = COALESCE(@DefaultStartUtc, DATEADD(minute, -@TickMinutes, @now));
                END
            END
            ELSE
            BEGIN
                -- Table doesn't exist - use DefaultStartUtc or current time
                SET @WindowStartEntryDateTime = COALESCE(@DefaultStartUtc, DATEADD(minute, -@TickMinutes, @now));
            END
        END
    END

    -- 2. Determine Window End
    -- If not provided, add TickMinutes to Start
    IF @WindowEndEntryDateTime IS NULL
    BEGIN
        SET @WindowEndEntryDateTime = DATEADD(minute, @TickMinutes, @WindowStartEntryDateTime);
    END

    -- 3. Log the run
    INSERT INTO dbo.RunLog(
        RunID, EquipID, Stage, StartEntryDateTime, EndEntryDateTime,
        Outcome, RowsRead, RowsWritten, ErrorJSON, TriggerReason, Version, ConfigHash,
        WindowStartEntryDateTime, WindowEndEntryDateTime
    )
    VALUES(
        @RunID, @EquipID, @Stage, @now, NULL,
        NULL, NULL, NULL, NULL, @TriggerReason, @Version, @ConfigHash,
        @WindowStartEntryDateTime, @WindowEndEntryDateTime
    );

    -- Delete prior partial artifacts for same RunID if any (idempotent re-run)
    DELETE FROM dbo.ACM_Scores_Long WHERE RunID = @RunID;
    DELETE FROM dbo.ACM_Drift_TS WHERE RunID = @RunID;
    DELETE FROM dbo.ACM_Anomaly_Events WHERE RunID = @RunID;
    DELETE FROM dbo.ACM_Regime_Episodes WHERE RunID = @RunID;
    -- Skip ACM_PCA_Metrics - has incorrect RunID type (bigint instead of uniqueidentifier)
    -- DELETE FROM dbo.ACM_PCA_Metrics WHERE RunID = @RunID;
    DELETE FROM dbo.ACM_HealthTimeline WHERE RunID = @RunID;
    DELETE FROM dbo.ACM_DefectTimeline WHERE RunID = @RunID;
END
GO

PRINT '✓ Migration 008: usp_ACM_StartRun now queries actual data range for first run';
PRINT '';
PRINT 'Changes:';
PRINT '  - When no previous run exists, SP queries MIN(EntryDateTime) from equipment''s data table';
PRINT '  - Falls back to DefaultStartUtc or "now - tick" only if table is empty/missing';
PRINT '  - Ensures batch mode processes historical data chronologically from data start';
PRINT '';
GO

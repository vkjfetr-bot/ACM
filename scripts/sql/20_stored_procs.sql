/*
ACM Stored Procedures: run lifecycle + equipment registration
Assumptions:
- Core tables exist from 10_core_tables.sql
- Optional config tables from 15_config_tables.sql
*/
USE [ACM];
GO

/* Safety drop (idempotent deploy) */
IF OBJECT_ID('dbo.usp_ACM_StartRun','P') IS NOT NULL DROP PROCEDURE dbo.usp_ACM_StartRun; 
IF OBJECT_ID('dbo.usp_ACM_FinalizeRun','P') IS NOT NULL DROP PROCEDURE dbo.usp_ACM_FinalizeRun; 
IF OBJECT_ID('dbo.usp_ACM_RegisterEquipment','P') IS NOT NULL DROP PROCEDURE dbo.usp_ACM_RegisterEquipment; 
GO

CREATE PROCEDURE dbo.usp_ACM_RegisterEquipment
    @EquipCode   nvarchar(64),
    @ExternalDb  sysname = NULL,
    @Active      bit = 1,
    @EquipID     int OUTPUT
AS
BEGIN
    SET NOCOUNT ON;
    -- Upsert-like: try find by code
    SELECT @EquipID = EquipID FROM dbo.Equipments WITH (UPDLOCK, HOLDLOCK) WHERE EquipCode = @EquipCode;
    IF @EquipID IS NULL
    BEGIN
        INSERT INTO dbo.Equipments(EquipCode, ExternalDb, Active)
        VALUES(@EquipCode, @ExternalDb, @Active);
        SET @EquipID = SCOPE_IDENTITY();
    END
    ELSE
    BEGIN
        UPDATE dbo.Equipments
        SET ExternalDb = COALESCE(@ExternalDb, ExternalDb),
            Active     = COALESCE(@Active, Active)
        WHERE EquipID = @EquipID;
    END
END
GO

/* Start a run: records metadata and returns RunID */
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
    
    -- Set EquipIDOut for the caller
    SET @EquipIDOut = @EquipID;

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
        
        -- If no previous run, use DefaultStartUtc or fallback to a fixed start (e.g. 2023-10-15 for this dataset)
        IF @WindowStartEntryDateTime IS NULL
        BEGIN
            SET @WindowStartEntryDateTime = COALESCE(@DefaultStartUtc, '2023-10-15T00:00:00.000');
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
    DELETE FROM dbo.ACM_PCA_Metrics WHERE RunID = @RunID;
    -- Also clear new tables
    DELETE FROM dbo.ACM_HealthTimeline WHERE RunID = @RunID;
    DELETE FROM dbo.ACM_DefectTimeline WHERE RunID = @RunID;
END
GO

/* Finalize a run: set status and stats */
CREATE PROCEDURE dbo.usp_ACM_FinalizeRun
    @RunID       uniqueidentifier,
    @Outcome     nvarchar(16),
    @RowsRead    int = NULL,
    @RowsWritten int = NULL,
    @ErrorJSON   nvarchar(max) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE dbo.RunLog
    SET Outcome = @Outcome,
        EndEntryDateTime = SYSUTCDATETIME(),
        RowsRead = COALESCE(@RowsRead, RowsRead),
        RowsWritten = COALESCE(@RowsWritten, RowsWritten),
        ErrorJSON = COALESCE(@ErrorJSON, ErrorJSON)
    WHERE RunID = @RunID;
END
GO

-- Fix usp_ACM_StartRun to use ACM_Runs table with correct schema
-- Date: 2025-12-03

USE ACM;
GO

SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE OR ALTER PROCEDURE dbo.usp_ACM_StartRun
  @EquipCode                 NVARCHAR(100) = NULL,
  @EquipID                   INT = NULL,
  @Stage                     VARCHAR(10),
  @TickMinutes               INT,
  @DefaultStartUtc           DATETIME2(3) = NULL,
  @Version                   VARCHAR(50) = NULL,
  @ConfigHash                VARCHAR(128) = NULL,
  @TriggerReason             VARCHAR(64) = 'timer'
AS
BEGIN
  SET NOCOUNT ON;
  SET QUOTED_IDENTIFIER ON;
  
  BEGIN TRY
    -- Resolve EquipID
    DECLARE @EquipName NVARCHAR(100);
    DECLARE @RunID UNIQUEIDENTIFIER;
    DECLARE @WindowStartEntryDateTime DATETIME2(3);
    DECLARE @WindowEndEntryDateTime DATETIME2(3);
    DECLARE @EquipIDOut INT;
    
    IF @EquipID IS NULL
    BEGIN
      SELECT @EquipID = e.EquipID, @EquipName = e.EquipName
      FROM dbo.Equipment e 
      WHERE e.EquipCode = @EquipCode;
      
      IF @EquipID IS NULL
        THROW 50001, 'Unknown equipment (EquipCode/EquipID).', 1;
    END
    ELSE
    BEGIN
      SELECT @EquipName = e.EquipName
      FROM dbo.Equipment e 
      WHERE e.EquipID = @EquipID;
      
      IF @EquipName IS NULL
        THROW 50002, 'Equipment ID exists but EquipName not found', 1;
    END

    -- Get last run window end time from ACM_Runs
    DECLARE @lastEnd DATETIME2(3);
    
    -- Note: ACM_Runs doesn't have WindowEndEntryDateTime, use CompletedAt as proxy
    SELECT @lastEnd = MAX(r.CompletedAt)
    FROM dbo.ACM_Runs r 
    WHERE r.EquipID = @EquipID 
      AND r.CompletedAt IS NOT NULL;

    IF @lastEnd IS NULL
    BEGIN
      -- First run: Query equipment-specific data table dynamically
      DECLARE @sql NVARCHAR(MAX);
      DECLARE @tableName NVARCHAR(200) = (
        SELECT EquipCode FROM dbo.Equipment WHERE EquipID = @EquipID
      ) + '_Data';
      DECLARE @minDate DATETIME2(3);

      -- Check if equipment data table exists
      IF OBJECT_ID('dbo.' + @tableName, 'U') IS NOT NULL
      BEGIN
        SET @sql = N'SELECT @minDate = MIN(EntryDateTime) FROM dbo.' + QUOTENAME(@tableName);
        EXEC sp_executesql @sql, N'@minDate DATETIME2(3) OUTPUT', @minDate OUTPUT;
        SET @WindowStartEntryDateTime = @minDate;
      END

      -- Fallback to DefaultStartUtc or calculated time if no data
      IF @WindowStartEntryDateTime IS NULL
        SET @WindowStartEntryDateTime = ISNULL(@DefaultStartUtc, DATEADD(MINUTE, -@TickMinutes, SYSUTCDATETIME()));
    END
    ELSE
      SET @WindowStartEntryDateTime = @lastEnd;

    -- Calculate window end time
    SET @WindowEndEntryDateTime = DATEADD(MINUTE, @TickMinutes, @WindowStartEntryDateTime);
    
    -- Generate new RunID
    SET @RunID = NEWID();

    -- Insert minimal run record into ACM_Runs
    -- Note: ACM_Runs has different schema than old Runs table
    -- Many fields will be updated later by run_metadata_writer.py
    INSERT INTO dbo.ACM_Runs (
      RunID,
      EquipID,
      EquipName,
      StartedAt,
      ConfigSignature
    )
    VALUES (
      @RunID,
      @EquipID,
      @EquipName,
      SYSUTCDATETIME(),
      @ConfigHash
    );

    -- Return resolved EquipID
    SET @EquipIDOut = @EquipID;

    -- Return results via SELECT (for pyodbc compatibility)
    SELECT 
      CONVERT(VARCHAR(36), @RunID) AS RunID,
      @WindowStartEntryDateTime AS WindowStart,
      @WindowEndEntryDateTime AS WindowEnd,
      @EquipIDOut AS EquipID;

  END TRY
  BEGIN CATCH
    DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
    DECLARE @ErrorSeverity INT = ERROR_SEVERITY();
    DECLARE @ErrorState INT = ERROR_STATE();
    
    RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
  END CATCH
END
GO

PRINT 'Successfully recreated usp_ACM_StartRun for ACM_Runs schema';
GO

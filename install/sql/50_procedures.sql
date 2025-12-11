USE [ACM];
GO

CREATE OR ALTER PROCEDURE dbo.usp_ACM_CheckColdstartStatus
    @EquipID INT,
    @Stage VARCHAR(20) = 'score',
    @RequiredRows INT = 500,
    @TickMinutes INT = 30,
    @NeedsColdstart BIT OUTPUT,
    @AccumulatedRows INT OUTPUT,
    @AttemptCount INT OUTPUT
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Check if models exist in ModelRegistry
    DECLARE @ModelCount INT;
    SELECT @ModelCount = COUNT(DISTINCT ModelType)
    FROM dbo.ModelRegistry
    WHERE EquipID = @EquipID;
    
    -- If models exist, coldstart is complete
    IF @ModelCount >= 3  -- Expect at least ar1, pca, iforest models
    BEGIN
        -- Update coldstart state to COMPLETE if exists
        UPDATE dbo.ACM_ColdstartState
        SET Status = 'COMPLETE',
            CompletedAt = GETUTCDATE(),
            UpdatedAt = GETUTCDATE()
        WHERE EquipID = @EquipID 
          AND Stage = @Stage
          AND Status != 'COMPLETE';
        
        SET @NeedsColdstart = 0;
        SET @AccumulatedRows = 0;
        SET @AttemptCount = 0;
        RETURN;
    END
    
    -- Models don't exist, check coldstart state
    SELECT 
        @NeedsColdstart = CASE WHEN Status IN ('PENDING', 'IN_PROGRESS') THEN 1 ELSE 0 END,
        @AccumulatedRows = AccumulatedRows,
        @AttemptCount = AttemptCount
    FROM dbo.ACM_ColdstartState
    WHERE EquipID = @EquipID AND Stage = @Stage;
    
    -- If no coldstart record exists, create one
    IF @@ROWCOUNT = 0
    BEGIN
        INSERT INTO dbo.ACM_ColdstartState (
            EquipID, Stage, Status, AttemptCount, 
            AccumulatedRows, RequiredRows, TickMinutes, ColdstartSplitRatio
        )
        VALUES (
            @EquipID, @Stage, 'PENDING', 0,
            0, @RequiredRows, @TickMinutes, 0.6
        );
        
        SET @NeedsColdstart = 1;
        SET @AccumulatedRows = 0;
        SET @AttemptCount = 0;
    END
    ELSE
    BEGIN
        SET @NeedsColdstart = 1;
    END
END
GO

/* Finalize a run: set status and stats */
CREATE OR ALTER PROCEDURE dbo.usp_ACM_FinalizeRun
    @RunID       uniqueidentifier,
    @Outcome     nvarchar(16),
    @RowsRead    int = NULL,
    @RowsWritten int = NULL,
    @ErrorJSON   nvarchar(max) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    SET QUOTED_IDENTIFIER ON;
    
    UPDATE dbo.ACM_Runs
    SET CompletedAt = SYSUTCDATETIME(),
        TrainRowCount = COALESCE(@RowsRead, TrainRowCount),
        ScoreRowCount = COALESCE(@RowsWritten, ScoreRowCount),
        ErrorMessage = COALESCE(@ErrorJSON, ErrorMessage)
    WHERE RunID = @RunID;
END
GO

CREATE OR ALTER PROCEDURE dbo.usp_ACM_GetHistorianData_TEMP
    @StartTime DATETIME2,
    @EndTime DATETIME2,
    @TagNames NVARCHAR(MAX) = NULL,        -- Comma-separated tag names (NULL = all tags)
    @EquipID INT = NULL,                   -- Equipment ID (alternative to EquipmentName)
    @EquipmentName VARCHAR(50) = NULL      -- Equipment name (alternative to EquipID)
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @ErrorMsg NVARCHAR(500);
    DECLARE @SQL NVARCHAR(MAX);
    DECLARE @TableName VARCHAR(100);
    DECLARE @ColumnList NVARCHAR(MAX);
    DECLARE @WhereClause NVARCHAR(500);
    
    -- =====================================================================
    -- Validate inputs and determine equipment
    -- =====================================================================
    IF @EquipID IS NULL AND @EquipmentName IS NULL
    BEGIN
        RAISERROR('Either @EquipID or @EquipmentName must be provided', 16, 1);
        RETURN;
    END
    
    -- Resolve EquipmentName from EquipID if needed
    IF @EquipmentName IS NULL
    BEGIN
        SELECT @EquipmentName = EquipCode
        FROM dbo.Equipment
        WHERE EquipID = @EquipID;
        
        IF @EquipmentName IS NULL
        BEGIN
            SET @ErrorMsg = 'EquipID ' + CAST(@EquipID AS VARCHAR(10)) + ' not found in Equipment table';
            RAISERROR(@ErrorMsg, 16, 1);
            RETURN;
        END
    END
    ELSE
    BEGIN
        -- Resolve EquipID from EquipmentName if needed
        IF @EquipID IS NULL
        BEGIN
            SELECT @EquipID = EquipID
            FROM dbo.Equipment
            WHERE EquipCode = @EquipmentName OR EquipName = @EquipmentName;
            
            IF @EquipID IS NULL
            BEGIN
                SET @ErrorMsg = 'Equipment ''' + @EquipmentName + ''' not found in Equipment table';
                RAISERROR(@ErrorMsg, 16, 1);
                RETURN;
            END
        END
    END
    
    -- Determine data table name based on equipment
    SET @TableName = @EquipmentName + '_Data';
    
    -- Verify table exists
    IF OBJECT_ID('dbo.' + @TableName, 'U') IS NULL
    BEGIN
        SET @ErrorMsg = 'Data table dbo.' + @TableName + ' does not exist';
        RAISERROR(@ErrorMsg, 16, 1);
        RETURN;
    END
    
    -- =====================================================================
    -- Build column list (validate tags if specified)
    -- =====================================================================
    IF @TagNames IS NULL OR LTRIM(RTRIM(@TagNames)) = ''
    BEGIN
        -- Return all sensor columns (exclude EntryDateTime, audit columns)
        SELECT @ColumnList = STRING_AGG(QUOTENAME(TagName), ', ') WITHIN GROUP (ORDER BY TagName)
        FROM dbo.ACM_TagEquipmentMap
        WHERE EquipID = @EquipID AND IsActive = 1;
        
        IF @ColumnList IS NULL
        BEGIN
            SET @ErrorMsg = 'No active tags found for EquipID ' + CAST(@EquipID AS VARCHAR(10));
            RAISERROR(@ErrorMsg, 16, 1);
            RETURN;
        END
    END
    ELSE
    BEGIN
        -- Parse and validate requested tags
        DECLARE @TagList TABLE (TagName VARCHAR(255));
        DECLARE @InvalidTags NVARCHAR(MAX);
        
        -- Split comma-separated tag names
        INSERT INTO @TagList (TagName)
        SELECT LTRIM(RTRIM(value))
        FROM STRING_SPLIT(@TagNames, ',')
        WHERE LTRIM(RTRIM(value)) <> '';
        
        -- Validate all requested tags exist for this equipment
        SELECT @InvalidTags = STRING_AGG(t.TagName, ', ')
        FROM @TagList t
        LEFT JOIN dbo.ACM_TagEquipmentMap m ON t.TagName = m.TagName AND m.EquipID = @EquipID AND m.IsActive = 1
        WHERE m.TagID IS NULL;
        
        IF @InvalidTags IS NOT NULL
        BEGIN
            SET @ErrorMsg = 'Invalid tags for ' + @EquipmentName + ': ' + @InvalidTags;
            RAISERROR(@ErrorMsg, 16, 1);
            RETURN;
        END
        
        -- Build column list from validated tags
        SELECT @ColumnList = STRING_AGG(QUOTENAME(t.TagName), ', ') WITHIN GROUP (ORDER BY t.TagName)
        FROM @TagList t
        INNER JOIN dbo.ACM_TagEquipmentMap m ON t.TagName = m.TagName AND m.EquipID = @EquipID;
    END
    
    -- =====================================================================
    -- Build WHERE clause
    -- =====================================================================
    SET @WhereClause = 'EntryDateTime >= @StartTime AND EntryDateTime <= @EndTime';
    
    -- =====================================================================
    -- Execute dynamic SQL
    -- =====================================================================
    SET @SQL = N'
        SELECT 
            EntryDateTime,
            ' + @ColumnList + N'
        FROM dbo.' + QUOTENAME(@TableName) + N'
        WHERE ' + @WhereClause + N'
        ORDER BY EntryDateTime ASC;
    ';
    
    -- Uncomment for debugging:
    -- PRINT @SQL;
    
    EXEC sp_executesql 
        @SQL,
        N'@StartTime DATETIME2, @EndTime DATETIME2',
        @StartTime = @StartTime,
        @EndTime = @EndTime;
    
END
GO

CREATE OR ALTER PROCEDURE dbo.usp_ACM_RegisterEquipment
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
CREATE OR ALTER PROCEDURE dbo.usp_ACM_StartRun
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
    SET QUOTED_IDENTIFIER ON;
    
    IF @RunID IS NULL SET @RunID = NEWID();
    DECLARE @now datetime2(3) = SYSUTCDATETIME();
    
    -- Set EquipIDOut for the caller
    SET @EquipIDOut = @EquipID;

    -- 1. Determine Window Start
    -- If not provided, look for the last successful run's end time
    IF @WindowStartEntryDateTime IS NULL
    BEGIN
        SELECT TOP 1 @WindowStartEntryDateTime = CompletedAt
        FROM dbo.ACM_Runs
        WHERE EquipID = @EquipID 
          AND HealthStatus IS NOT NULL
          AND CompletedAt IS NOT NULL
        ORDER BY CompletedAt DESC;
        
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
    INSERT INTO dbo.ACM_Runs(
        RunID, EquipID, StartedAt, ConfigSignature
    )
    VALUES(
        @RunID, @EquipID, @now, @ConfigHash
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
    
    -- Return window information via SELECT for Python caller
    SELECT @RunID, @EquipIDOut, @WindowStartEntryDateTime, @WindowEndEntryDateTime;
END
GO

CREATE OR ALTER PROCEDURE dbo.usp_ACM_UpdateColdstartProgress
    @EquipID INT,
    @Stage VARCHAR(20) = 'score',
    @RowsReceived INT,
    @DataStartTime DATETIME2 = NULL,
    @DataEndTime DATETIME2 = NULL,
    @ErrorMessage NVARCHAR(2000) = NULL,
    @Success BIT = 0
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        -- Upsert coldstart state
        IF EXISTS (SELECT 1 FROM dbo.ACM_ColdstartState WHERE EquipID = @EquipID AND Stage = @Stage)
        BEGIN
            UPDATE dbo.ACM_ColdstartState
            SET 
                AttemptCount = AttemptCount + 1,
                LastAttemptAt = GETUTCDATE(),
                AccumulatedRows = AccumulatedRows + @RowsReceived,
                DataStartTime = ISNULL(DataStartTime, @DataStartTime),
                DataEndTime = ISNULL(@DataEndTime, DataEndTime),
                Status = CASE 
                    WHEN @Success = 1 THEN 'COMPLETE'
                    WHEN @ErrorMessage IS NOT NULL THEN 'FAILED'
                    WHEN AccumulatedRows + @RowsReceived >= RequiredRows THEN 'IN_PROGRESS'
                    ELSE 'PENDING'
                END,
                CompletedAt = CASE WHEN @Success = 1 THEN GETUTCDATE() ELSE CompletedAt END,
                LastError = @ErrorMessage,
                ErrorCount = CASE WHEN @ErrorMessage IS NOT NULL THEN ErrorCount + 1 ELSE ErrorCount END,
                UpdatedAt = GETUTCDATE()
            WHERE EquipID = @EquipID AND Stage = @Stage;
        END
        ELSE
        BEGIN
            INSERT INTO dbo.ACM_ColdstartState (
                EquipID, Stage, Status, AttemptCount,
                AccumulatedRows, RequiredRows, TickMinutes,
                DataStartTime, DataEndTime, LastError, ErrorCount
            )
            VALUES (
                @EquipID, @Stage, 
                CASE WHEN @Success = 1 THEN 'COMPLETE' WHEN @ErrorMessage IS NOT NULL THEN 'FAILED' ELSE 'PENDING' END,
                1, @RowsReceived, 500, 30,
                @DataStartTime, @DataEndTime, @ErrorMessage,
                CASE WHEN @ErrorMessage IS NOT NULL THEN 1 ELSE 0 END
            );
        END
    END TRY
    BEGIN CATCH
        THROW;
    END CATCH
END
GO

-- ========================================
-- Baseline retention cleanup procedure
-- ========================================
CREATE OR ALTER PROCEDURE dbo.usp_CleanupBaselineBuffer
    @EquipID INT = NULL,
    @RetentionHours INT = 72,
    @MaxRowsPerEquip INT = 100000
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @CutoffTime DATETIME2(3) = DATEADD(HOUR, -@RetentionHours, GETDATE());
    DECLARE @RowsDeleted INT = 0;
    
    -- Time-based cleanup
    DELETE FROM dbo.ACM_BaselineBuffer
    WHERE (@EquipID IS NULL OR EquipID = @EquipID)
        AND CreatedAt < @CutoffTime;
    
    SET @RowsDeleted = @@ROWCOUNT;
    
    -- Row count limit per equipment
    IF @MaxRowsPerEquip > 0
    BEGIN
        ;WITH RankedRows AS (
            SELECT 
                ID,
                ROW_NUMBER() OVER (PARTITION BY EquipID ORDER BY Timestamp DESC) AS RowNum
            FROM dbo.ACM_BaselineBuffer
            WHERE @EquipID IS NULL OR EquipID = @EquipID
        )
        DELETE FROM RankedRows
        WHERE RowNum > @MaxRowsPerEquip;
        
        SET @RowsDeleted = @RowsDeleted + @@ROWCOUNT;
    END
    
    PRINT 'Deleted ' + CAST(@RowsDeleted AS VARCHAR(20)) + ' rows from ACM_BaselineBuffer';
    
    RETURN @RowsDeleted;
END
GO

CREATE PROC dbo.usp_Write_AnomalyEvents
  @RunID UNIQUEIDENTIFIER,
  @Rows  dbo.TVP_AnomalyEvents READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.AnomalyEvents WHERE RunID = @RunID;
      INSERT INTO dbo.AnomalyEvents
        (EquipID, StartEntryDateTime, EndEntryDateTime, Severity, Detector, Score, ContributorsJSON, RunID)
      SELECT EquipID, StartEntryDateTime, EndEntryDateTime, Severity, Detector, Score, ContributorsJSON, RunID
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK;
    DECLARE @msg nvarchar(4000) = ERROR_MESSAGE();
    THROW 50021, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_AnomalyTopSpikes
  @RunID UNIQUEIDENTIFIER,
  @Rows  dbo.TVP_AnomalyTopSpikes READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.AnomalyTopSpikes WHERE RunID = @RunID;
      INSERT INTO dbo.AnomalyTopSpikes
        (RunID, EntryDateTime, EquipID, Sensor, Score, Rank, WindowStartEntryDateTime, WindowEndEntryDateTime)
      SELECT RunID, EntryDateTime, EquipID, Sensor, Score, Rank, WindowStartEntryDateTime, WindowEndEntryDateTime
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50071, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_ConfigLog
  @Rows dbo.TVP_ConfigLog READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    DECLARE @RunID UNIQUEIDENTIFIER;
    SELECT TOP(1) @RunID = RunID FROM @Rows;

    BEGIN TRAN;
      DELETE FROM dbo.ConfigLog WHERE RunID = @RunID;
      INSERT INTO dbo.ConfigLog (RunID, EquipID, EntryDateTime, ConfigHash, ConfigJSON, Active)
      SELECT RunID, EquipID, EntryDateTime, ConfigHash, ConfigJSON, Active
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50091, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_CPD_Points
  @Rows dbo.TVP_CPD_Points READONLY
AS
BEGIN
  SET NOCOUNT ON;
  INSERT INTO dbo.CPD_Points (EntryDateTime, EquipID, Detector, Score, RunID)
  SELECT EntryDateTime, EquipID, Detector, Score, RunID
  FROM @Rows;
END
GO

CREATE PROC dbo.usp_Write_DataQualityTS
  @Rows dbo.TVP_DataQualityTS READONLY
AS
BEGIN
  SET NOCOUNT ON;
  INSERT INTO dbo.DataQualityTS (EntryDateTime, EquipID, CadenceOK, MissingPct, InterpPct, KeptSensors, RunID)
  SELECT EntryDateTime, EquipID, CadenceOK, MissingPct, InterpPct, KeptSensors, RunID
  FROM @Rows;
END
GO

CREATE PROC dbo.usp_Write_DriftSummary
  @RunID UNIQUEIDENTIFIER,
  @Rows  dbo.TVP_DriftSummary READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.DriftSummary WHERE RunID = @RunID;
      INSERT INTO dbo.DriftSummary
        (RunID, EntryDateTime, Method, P50, P75, P90, P95, P99, Mean, Std)
      SELECT RunID, EntryDateTime, Method, P50, P75, P90, P95, P99, Mean, Std
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50061, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_DriftTS
  @Rows dbo.TVP_DriftTS READONLY
AS
BEGIN
  SET NOCOUNT ON;
  INSERT INTO dbo.DriftTS (EntryDateTime, EquipID, DriftZ, Method, RunID)
  SELECT EntryDateTime, EquipID, DriftZ, Method, RunID
  FROM @Rows;
END
GO

CREATE PROC dbo.usp_Write_FeatureImportance
  @RunID UNIQUEIDENTIFIER,
  @Rows  dbo.TVP_FeatureImportance READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.FeatureImportance WHERE RunID = @RunID;
      INSERT INTO dbo.FeatureImportance
        (RunID, EntryDateTime, ModelName, Feature, Importance, Method, Rank)
      SELECT RunID, EntryDateTime, ModelName, Feature, Importance, Method, Rank
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50051, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_ForecastResidualsTS
  @Rows dbo.TVP_ForecastResidualsTS READONLY
AS
BEGIN
  SET NOCOUNT ON;
  INSERT INTO dbo.ForecastResidualsTS (EntryDateTime, EquipID, Sensor, Residual, ModelName, HorizonSec, RunID)
  SELECT EntryDateTime, EquipID, Sensor, Residual, ModelName, HorizonSec, RunID
  FROM @Rows;
END
GO

CREATE PROC dbo.usp_Write_PCA_Loadings
  @RunID UNIQUEIDENTIFIER,
  @Rows  dbo.TVP_PCA_Loadings READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.PCA_Components WHERE RunID = @RunID;
      INSERT INTO dbo.PCA_Components (RunID, EntryDateTime, ComponentNo, Sensor, Loading)
      SELECT RunID, EntryDateTime, ComponentNo, Sensor, Loading
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50032, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_PCA_Metrics
  @RunID UNIQUEIDENTIFIER,
  @Rows  dbo.TVP_PCA_Metrics READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.PCA_Metrics WHERE RunID = @RunID;
      INSERT INTO dbo.PCA_Metrics (RunID, EntryDateTime, Var90_N, ReconRMSE, P95_ReconRMSE, Notes)
      SELECT RunID, EntryDateTime, Var90_N, ReconRMSE, P95_ReconRMSE, Notes
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50033, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_PCA_Model
  @RunID                   UNIQUEIDENTIFIER,
  @EquipID                 INT,
  @EntryDateTime           DATETIME2(3),
  @NComponents             INT,
  @TargetVar               REAL = NULL,
  @VarExplainedJSON        NVARCHAR(MAX) = NULL,
  @ScalingSpecJSON         NVARCHAR(MAX) = NULL,
  @ModelVersion            INT = NULL,
  @TrainStartEntryDateTime DATETIME2(3) = NULL,
  @TrainEndEntryDateTime   DATETIME2(3) = NULL
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.PCA_Model WHERE RunID = @RunID;
      INSERT INTO dbo.PCA_Model
        (RunID, EquipID, EntryDateTime, NComponents, TargetVar, VarExplainedJSON,
         ScalingSpecJSON, ModelVersion, TrainStartEntryDateTime, TrainEndEntryDateTime)
      VALUES
        (@RunID, @EquipID, @EntryDateTime, @NComponents, @TargetVar, @VarExplainedJSON,
         @ScalingSpecJSON, @ModelVersion, @TrainStartEntryDateTime, @TrainEndEntryDateTime);
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50031, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_PCA_ScoresTS
  @Rows dbo.TVP_PCA_ScoresTS READONLY
AS
BEGIN
  SET NOCOUNT ON;
  INSERT INTO dbo.PCA_ScoresTS (EntryDateTime, EquipID, ComponentNo, ScoreValue, RunID)
  SELECT EntryDateTime, EquipID, ComponentNo, ScoreValue, RunID
  FROM @Rows;
END
GO

CREATE PROC dbo.usp_Write_RegimeEpisodes
  @RunID UNIQUEIDENTIFIER,
  @Rows  dbo.TVP_RegimeEpisodes READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.RegimeEpisodes WHERE RunID = @RunID;
      INSERT INTO dbo.RegimeEpisodes
        (EquipID, StartEntryDateTime, EndEntryDateTime, RegimeLabel, Confidence, RunID)
      SELECT EquipID, StartEntryDateTime, EndEntryDateTime, RegimeLabel, Confidence, RunID
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50022, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_ScoresTS
  @Rows dbo.TVP_ScoresTS READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    INSERT INTO dbo.ScoresTS (EntryDateTime, EquipID, Sensor, Value, Source, RunID)
    SELECT EntryDateTime, EquipID, Sensor, Value, Source, RunID
    FROM @Rows;
  END TRY
  BEGIN CATCH
    DECLARE @msg nvarchar(4000) = ERROR_MESSAGE();
    THROW 50011, @msg, 1;
  END CATCH
END
GO

CREATE PROC dbo.usp_Write_XCorrTopPairs
  @RunID UNIQUEIDENTIFIER,
  @Rows  dbo.TVP_XCorrTopPairs READONLY
AS
BEGIN
  SET NOCOUNT ON;
  BEGIN TRY
    BEGIN TRAN;
      DELETE FROM dbo.XCorrTopPairs WHERE RunID = @RunID;
      INSERT INTO dbo.XCorrTopPairs (RunID, EntryDateTime, SensorA, SensorB, R, LagSec, Rank)
      SELECT RunID, EntryDateTime, SensorA, SensorB, R, LagSec, Rank
      FROM @Rows;
    COMMIT TRAN;
  END TRY
  BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    DECLARE @msg NVARCHAR(4000) = ERROR_MESSAGE();
    THROW 50041, @msg, 1;
  END CATCH
END
GO

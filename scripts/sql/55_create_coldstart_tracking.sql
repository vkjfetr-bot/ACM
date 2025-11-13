-- ============================================================================
-- SQL-55: Coldstart State Tracking Table
-- ============================================================================
-- Tracks coldstart attempts and accumulated data for each equipment
-- Allows ACM to intelligently retry coldstart until sufficient data exists
-- ============================================================================

USE ACM;
GO

SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF OBJECT_ID('dbo.ACM_ColdstartState', 'U') IS NOT NULL 
BEGIN
    PRINT 'ACM_ColdstartState table already exists, dropping...';
    DROP TABLE dbo.ACM_ColdstartState;
END
GO

CREATE TABLE dbo.ACM_ColdstartState (
    EquipID INT NOT NULL,
    Stage VARCHAR(20) NOT NULL DEFAULT 'score',  -- 'train' or 'score'
    
    -- Coldstart status
    Status VARCHAR(20) NOT NULL,  -- 'PENDING', 'IN_PROGRESS', 'COMPLETE', 'FAILED'
    AttemptCount INT NOT NULL DEFAULT 0,
    FirstAttemptAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    LastAttemptAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    CompletedAt DATETIME2 NULL,
    
    -- Data accumulation tracking
    AccumulatedRows INT NOT NULL DEFAULT 0,
    RequiredRows INT NOT NULL DEFAULT 500,  -- Configurable minimum for coldstart
    DataStartTime DATETIME2 NULL,  -- Earliest data timestamp accumulated
    DataEndTime DATETIME2 NULL,    -- Latest data timestamp accumulated
    
    -- Configuration snapshot
    TickMinutes INT NOT NULL,
    ColdstartSplitRatio FLOAT NOT NULL DEFAULT 0.6,
    
    -- Error tracking
    LastError NVARCHAR(2000) NULL,
    ErrorCount INT NOT NULL DEFAULT 0,
    
    -- Audit
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT PK_ACM_ColdstartState PRIMARY KEY CLUSTERED (EquipID, Stage),
    CONSTRAINT FK_ColdstartState_Equipment FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID),
    CONSTRAINT CK_ColdstartState_Status CHECK (Status IN ('PENDING', 'IN_PROGRESS', 'COMPLETE', 'FAILED'))
);
GO

CREATE INDEX IX_ColdstartState_Status 
    ON dbo.ACM_ColdstartState (Status, LastAttemptAt DESC)
    WHERE Status IN ('PENDING', 'IN_PROGRESS');
GO

PRINT 'ACM_ColdstartState table created successfully';
GO

-- ============================================================================
-- Stored Procedure: usp_ACM_CheckColdstartStatus
-- ============================================================================
-- Check if coldstart is needed and return current state
-- ============================================================================

IF OBJECT_ID('dbo.usp_ACM_CheckColdstartStatus', 'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_ACM_CheckColdstartStatus;
GO

CREATE PROCEDURE dbo.usp_ACM_CheckColdstartStatus
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

-- ============================================================================
-- Stored Procedure: usp_ACM_UpdateColdstartProgress
-- ============================================================================
-- Update coldstart progress after each attempt
-- ============================================================================

IF OBJECT_ID('dbo.usp_ACM_UpdateColdstartProgress', 'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_ACM_UpdateColdstartProgress;
GO

CREATE PROCEDURE dbo.usp_ACM_UpdateColdstartProgress
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

PRINT 'Coldstart tracking stored procedures created successfully';
GO

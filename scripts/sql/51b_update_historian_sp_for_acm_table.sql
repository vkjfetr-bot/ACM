/*
 * Script: 51b_update_historian_sp_for_acm_table.sql
 * Purpose: Update historian SP to use ACM_HistorianData table instead of equipment-specific tables
 * Context: Migration from FD_FAN_Data/GAS_TURBINE_Data to unified ACM_HistorianData
 */

USE ACM;
GO

PRINT 'Updating usp_ACM_GetHistorianData_TEMP to use ACM_HistorianData...';
PRINT '';

IF OBJECT_ID('dbo.usp_ACM_GetHistorianData_TEMP', 'P') IS NOT NULL
    DROP PROCEDURE dbo.usp_ACM_GetHistorianData_TEMP;
GO

CREATE PROCEDURE dbo.usp_ACM_GetHistorianData_TEMP
    @StartTime DATETIME2,
    @EndTime DATETIME2,
    @TagNames NVARCHAR(MAX) = NULL,        -- Comma-separated tag names (NULL = all tags)
    @EquipID INT = NULL,                   -- Equipment ID (alternative to EquipmentName)
    @EquipmentName VARCHAR(50) = NULL      -- Equipment name (alternative to EquipID)
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @ErrorMsg NVARCHAR(500);
    
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
    
    -- =====================================================================
    -- Query ACM_HistorianData and pivot to wide format
    -- =====================================================================
    
    DECLARE @PivotCols NVARCHAR(MAX);
    DECLARE @PivotSQL NVARCHAR(MAX);
    
    -- Determine which tags to include
    IF @TagNames IS NOT NULL AND LTRIM(RTRIM(@TagNames)) <> ''
    BEGIN
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
        LEFT JOIN (
            SELECT DISTINCT TagName 
            FROM dbo.ACM_HistorianData 
            WHERE EquipID = @EquipID
        ) h ON t.TagName = h.TagName
        WHERE h.TagName IS NULL;
        
        IF @InvalidTags IS NOT NULL
        BEGIN
            SET @ErrorMsg = 'Invalid tags for ' + @EquipmentName + ': ' + @InvalidTags;
            RAISERROR(@ErrorMsg, 16, 1);
            RETURN;
        END
        
        -- Build pivot column list from validated tags
        SELECT @PivotCols = STRING_AGG(QUOTENAME(TagName), ', ')
        FROM @TagList;
    END
    ELSE
    BEGIN
        -- Get all unique tags for this equipment in the time range
        SELECT @PivotCols = STRING_AGG(QUOTENAME(TagName), ', ')
        FROM (
            SELECT DISTINCT TagName
            FROM dbo.ACM_HistorianData
            WHERE EquipID = @EquipID
                AND Timestamp >= @StartTime 
                AND Timestamp <= @EndTime
        ) AS Tags;
    END
    
    IF @PivotCols IS NULL OR LTRIM(RTRIM(@PivotCols)) = ''
    BEGIN
        SET @ErrorMsg = 'No data found for EquipID ' + CAST(@EquipID AS VARCHAR(10)) + ' in specified time range';
        RAISERROR(@ErrorMsg, 16, 1);
        RETURN;
    END
    
    -- Build and execute dynamic pivot query
    SET @PivotSQL = N'
        SELECT 
            EntryDateTime,
            ' + @PivotCols + N'
        FROM (
            SELECT 
                Timestamp AS EntryDateTime,
                TagName,
                Value
            FROM dbo.ACM_HistorianData
            WHERE 
                EquipID = @EquipID
                AND Timestamp >= @StartTime 
                AND Timestamp <= @EndTime
        ) AS SourceData
        PIVOT (
            MAX(Value)
            FOR TagName IN (' + @PivotCols + N')
        ) AS PivotTable
        ORDER BY EntryDateTime;
    ';
    
    EXEC sp_executesql 
        @PivotSQL,
        N'@EquipID INT, @StartTime DATETIME2, @EndTime DATETIME2',
        @EquipID = @EquipID,
        @StartTime = @StartTime,
        @EndTime = @EndTime;
    
END
GO

PRINT '✓ usp_ACM_GetHistorianData_TEMP updated to use ACM_HistorianData';
PRINT '';
PRINT 'Testing with FD_FAN data...';

-- Test the updated SP
DECLARE @TestStart DATETIME2 = '2012-01-06 00:00:00';
DECLARE @TestEnd DATETIME2 = '2012-01-06 01:00:00';

EXEC usp_ACM_GetHistorianData_TEMP
    @StartTime = @TestStart,
    @EndTime = @TestEnd,
    @EquipID = 1;

PRINT '';
PRINT '========================================';
PRINT 'Stored procedure updated and tested!';
PRINT '========================================';
GO

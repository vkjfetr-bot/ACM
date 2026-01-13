-- Check GAS_TURBINE_Data table schema and data range
USE ACM;
GO

-- Get column info
SELECT 
    COLUMN_NAME, 
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'GAS_TURBINE_Data'
ORDER BY ORDINAL_POSITION;
GO

-- Get data range (try different timestamp column names)
DECLARE @sql NVARCHAR(MAX);
DECLARE @tsCol NVARCHAR(128);

SELECT TOP 1 @tsCol = COLUMN_NAME 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'GAS_TURBINE_Data' 
  AND DATA_TYPE IN ('datetime', 'datetime2', 'datetimeoffset')
ORDER BY ORDINAL_POSITION;

IF @tsCol IS NOT NULL
BEGIN
    SET @sql = 'SELECT MIN(' + QUOTENAME(@tsCol) + ') AS MinTime, MAX(' + QUOTENAME(@tsCol) + ') AS MaxTime, COUNT(*) AS RowCount FROM GAS_TURBINE_Data';
    EXEC sp_executesql @sql;
END
ELSE
BEGIN
    SELECT 'No timestamp column found' AS Error;
END
GO

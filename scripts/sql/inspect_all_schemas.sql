-- =====================================================
-- COMPREHENSIVE SQL SCHEMA INSPECTOR
-- Run this to get complete schema of all ACM tables
-- =====================================================

USE ACM;
GO

PRINT '=====================================================';
PRINT 'ACM DATABASE SCHEMA INSPECTION REPORT';
PRINT 'Generated: ' + CONVERT(VARCHAR, GETDATE(), 120);
PRINT '=====================================================';
PRINT '';

-- =====================================================
-- 1. ALL TABLES WITH ROW COUNTS
-- =====================================================
PRINT '1. ALL TABLES (with row counts)';
PRINT '-----------------------------------------------------';

SELECT 
    t.TABLE_SCHEMA + '.' + t.TABLE_NAME AS [Table],
    CAST(p.rows AS VARCHAR(20)) AS [RowCount],
    CAST(CAST(SUM(a.total_pages) * 8 / 1024.0 AS DECIMAL(10,2)) AS VARCHAR(20)) + ' MB' AS [Size]
FROM INFORMATION_SCHEMA.TABLES t
LEFT JOIN sys.tables st ON t.TABLE_NAME = st.name
LEFT JOIN sys.partitions p ON st.object_id = p.object_id AND p.index_id IN (0,1)
LEFT JOIN sys.allocation_units a ON p.partition_id = a.container_id
WHERE t.TABLE_TYPE = 'BASE TABLE'
AND t.TABLE_SCHEMA IN ('dbo')
AND t.TABLE_NAME LIKE 'ACM%'
GROUP BY t.TABLE_SCHEMA, t.TABLE_NAME, p.rows
ORDER BY p.rows DESC;

PRINT '';
PRINT '=====================================================';
PRINT '';

-- =====================================================
-- 2. DETAILED COLUMN SCHEMAS FOR EACH TABLE
-- =====================================================
PRINT '2. DETAILED COLUMN SCHEMAS';
PRINT '-----------------------------------------------------';

DECLARE @TableName NVARCHAR(128);
DECLARE table_cursor CURSOR FOR
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
    AND TABLE_SCHEMA = 'dbo'
    AND TABLE_NAME LIKE 'ACM%'
    ORDER BY TABLE_NAME;

OPEN table_cursor;
FETCH NEXT FROM table_cursor INTO @TableName;

WHILE @@FETCH_STATUS = 0
BEGIN
    PRINT '';
    PRINT '--- ' + @TableName + ' ---';
    
    SELECT 
        COLUMN_NAME AS [Column],
        DATA_TYPE + 
        CASE 
            WHEN DATA_TYPE IN ('varchar', 'nvarchar', 'char', 'nchar') THEN '(' + CAST(CHARACTER_MAXIMUM_LENGTH AS VARCHAR) + ')'
            WHEN DATA_TYPE IN ('decimal', 'numeric') THEN '(' + CAST(NUMERIC_PRECISION AS VARCHAR) + ',' + CAST(NUMERIC_SCALE AS VARCHAR) + ')'
            ELSE ''
        END AS [DataType],
        CASE WHEN IS_NULLABLE = 'YES' THEN 'NULL' ELSE 'NOT NULL' END AS [Nullable],
        ISNULL(COLUMN_DEFAULT, '') AS [Default]
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = @TableName
    ORDER BY ORDINAL_POSITION;
    
    FETCH NEXT FROM table_cursor INTO @TableName;
END;

CLOSE table_cursor;
DEALLOCATE table_cursor;

PRINT '';
PRINT '=====================================================';
PRINT '';

-- =====================================================
-- 3. ALL INDEXES
-- =====================================================
PRINT '3. INDEXES (Performance Optimization)';
PRINT '-----------------------------------------------------';

SELECT 
    OBJECT_NAME(i.object_id) AS [Table],
    i.name AS [IndexName],
    i.type_desc AS [IndexType],
    STRING_AGG(c.name, ', ') WITHIN GROUP (ORDER BY ic.key_ordinal) AS [Columns]
FROM sys.indexes i
INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
INNER JOIN sys.tables t ON i.object_id = t.object_id
WHERE t.name LIKE 'ACM%'
GROUP BY OBJECT_NAME(i.object_id), i.name, i.type_desc
ORDER BY [Table], [IndexName];

PRINT '';
PRINT '=====================================================';
PRINT '';

-- =====================================================
-- 4. FOREIGN KEY RELATIONSHIPS
-- =====================================================
PRINT '4. FOREIGN KEY RELATIONSHIPS';
PRINT '-----------------------------------------------------';

SELECT 
    OBJECT_NAME(f.parent_object_id) AS [FromTable],
    COL_NAME(fc.parent_object_id, fc.parent_column_id) AS [FromColumn],
    OBJECT_NAME(f.referenced_object_id) AS [ToTable],
    COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS [ToColumn],
    f.name AS [ConstraintName]
FROM sys.foreign_keys AS f
INNER JOIN sys.foreign_key_columns AS fc ON f.object_id = fc.constraint_object_id
WHERE OBJECT_NAME(f.parent_object_id) LIKE 'ACM%'
   OR OBJECT_NAME(f.referenced_object_id) LIKE 'ACM%'
ORDER BY [FromTable], [ToTable];

PRINT '';
PRINT '=====================================================';
PRINT '';

-- =====================================================
-- 5. VIEWS
-- =====================================================
PRINT '5. VIEWS (Virtual Tables)';
PRINT '-----------------------------------------------------';

SELECT 
    TABLE_NAME AS [ViewName],
    VIEW_DEFINITION AS [Definition]
FROM INFORMATION_SCHEMA.VIEWS
WHERE TABLE_NAME LIKE 'ACM%'
ORDER BY TABLE_NAME;

PRINT '';
PRINT '=====================================================';
PRINT '';

-- =====================================================
-- 6. STORED PROCEDURES
-- =====================================================
PRINT '6. STORED PROCEDURES';
PRINT '-----------------------------------------------------';

SELECT 
    ROUTINE_NAME AS [ProcedureName],
    CREATED AS [Created],
    LAST_ALTERED AS [LastModified]
FROM INFORMATION_SCHEMA.ROUTINES
WHERE ROUTINE_TYPE = 'PROCEDURE'
AND ROUTINE_NAME LIKE 'usp_ACM%'
ORDER BY ROUTINE_NAME;

PRINT '';
PRINT '=====================================================';
PRINT '';

-- =====================================================
-- 7. DATA RANGES (TIMESTAMPS)
-- =====================================================
PRINT '7. DATA RANGES (Timestamp Analysis)';
PRINT '-----------------------------------------------------';

-- Helper function to check if column exists
DECLARE @SQL NVARCHAR(MAX);
DECLARE @Results TABLE (TableName NVARCHAR(128), MinTime DATETIME2, MaxTime DATETIME2, RowCount BIGINT);

DECLARE @CheckTable NVARCHAR(128);
DECLARE range_cursor CURSOR FOR
    SELECT DISTINCT TABLE_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME LIKE 'ACM%'
    AND COLUMN_NAME IN ('Timestamp', 'EntryDateTime', 'CreatedAt', 'LastUpdate')
    AND DATA_TYPE IN ('datetime', 'datetime2', 'date');

OPEN range_cursor;
FETCH NEXT FROM range_cursor INTO @CheckTable;

WHILE @@FETCH_STATUS = 0
BEGIN
    -- Dynamically build query for timestamp columns
    SET @SQL = N'
    SELECT 
        ''' + @CheckTable + ''' AS TableName,
        MIN(COALESCE(Timestamp, EntryDateTime, CreatedAt, LastUpdate)) AS MinTime,
        MAX(COALESCE(Timestamp, EntryDateTime, CreatedAt, LastUpdate)) AS MaxTime,
        COUNT(*) AS RowCount
    FROM ' + @CheckTable + '
    WHERE COALESCE(Timestamp, EntryDateTime, CreatedAt, LastUpdate) IS NOT NULL';
    
    BEGIN TRY
        INSERT INTO @Results
        EXEC sp_executesql @SQL;
    END TRY
    BEGIN CATCH
        -- Skip tables with errors
    END CATCH;
    
    FETCH NEXT FROM range_cursor INTO @CheckTable;
END;

CLOSE range_cursor;
DEALLOCATE range_cursor;

SELECT 
    TableName AS [Table],
    MinTime AS [EarliestData],
    MaxTime AS [LatestData],
    RowCount AS [Rows]
FROM @Results
WHERE MinTime IS NOT NULL
ORDER BY MaxTime DESC;

PRINT '';
PRINT '=====================================================';
PRINT 'SCHEMA INSPECTION COMPLETE';
PRINT '=====================================================';
GO

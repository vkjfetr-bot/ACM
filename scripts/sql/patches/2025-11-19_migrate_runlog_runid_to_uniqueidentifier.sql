-- SQL Patch: Migrate RunLog.RunID from bigint (legacy) to uniqueidentifier (current expected schema)
-- Purpose: Resolve Python start-run failure "Operand type clash: uniqueidentifier is incompatible with bigint"
-- Safe re-runnable: Only acts when RunLog.RunID is bigint.
-- Steps:
--  1. Detect column data type
--  2. If bigint: add temporary GUID column, populate, enforce NOT NULL
--  3. Drop PK constraint on old RunID
--  4. Rename columns: RunID -> RunID_bigint_backup; RunID_guid -> RunID
--  5. Recreate PK on new GUID RunID
--  6. (Optional) Keep backup column for audit; user may drop later.
--  7. Print status messages

USE [ACM];
GO

DECLARE @data_type sysname;
SELECT @data_type = DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'RunLog' AND COLUMN_NAME = 'RunID';

IF @data_type = 'bigint'
BEGIN
    PRINT 'RunLog.RunID is bigint - starting migration to uniqueidentifier.';

    -- 1. Add temporary GUID column if missing
    IF COL_LENGTH('dbo.RunLog','RunID_guid') IS NULL
    BEGIN
        ALTER TABLE dbo.RunLog ADD RunID_guid uniqueidentifier NULL;
        PRINT 'Added RunID_guid column.';
    END

    -- 2. Populate GUIDs
    UPDATE dbo.RunLog SET RunID_guid = COALESCE(RunID_guid, NEWID());
    PRINT 'Populated RunID_guid values.';

    -- 3. Enforce NOT NULL
    ALTER TABLE dbo.RunLog ALTER COLUMN RunID_guid uniqueidentifier NOT NULL;
    PRINT 'RunID_guid set NOT NULL.';

    -- 4. Drop existing PK (discover name dynamically)
    DECLARE @pk_name sysname;
    SELECT @pk_name = kc.NAME
    FROM sys.key_constraints kc
    JOIN sys.tables t ON kc.parent_object_id = t.object_id
    WHERE t.name = 'RunLog' AND kc.type = 'PK';

    IF @pk_name IS NOT NULL
    BEGIN
        DECLARE @drop_pk_sql nvarchar(400) = N'ALTER TABLE dbo.RunLog DROP CONSTRAINT ' + QUOTENAME(@pk_name) + ';';
        EXEC sp_executesql @drop_pk_sql;
        PRINT 'Dropped PK constraint ' + @pk_name + '.';
    END

    -- 5. Rename old RunID column
    EXEC sp_rename 'dbo.RunLog.RunID', 'RunID_bigint_backup', 'COLUMN';
    PRINT 'Renamed old RunID to RunID_bigint_backup.';

    -- 6. Rename GUID column to RunID
    EXEC sp_rename 'dbo.RunLog.RunID_guid', 'RunID', 'COLUMN';
    PRINT 'Renamed RunID_guid to RunID.';

    -- 7. Create new PK on GUID RunID
    ALTER TABLE dbo.RunLog ADD CONSTRAINT PK_RunLog_RunID PRIMARY KEY CLUSTERED (RunID);
    PRINT 'Created PK_RunLog_RunID on new GUID RunID.';

    PRINT 'Migration complete.';
END
ELSE IF @data_type = 'uniqueidentifier'
BEGIN
    PRINT 'RunLog.RunID already uniqueidentifier - no migration needed.';
END
ELSE
BEGIN
    PRINT 'RunLog.RunID unexpected type ' + COALESCE(@data_type,'(NULL)') + ' - manual intervention required.';
END
GO

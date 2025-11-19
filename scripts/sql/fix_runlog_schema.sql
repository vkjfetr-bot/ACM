/*
Fix RunLog table schema to match usp_ACM_StartRun expectations
Adds missing columns needed by the stored procedure
*/

USE [ACM];
GO

-- Add missing columns to RunLog table
IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'StartEntryDateTime')
BEGIN
    ALTER TABLE dbo.RunLog ADD StartEntryDateTime DATETIME2(3) NULL;
    PRINT 'Added StartEntryDateTime column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'EndEntryDateTime')
BEGIN
    ALTER TABLE dbo.RunLog ADD EndEntryDateTime DATETIME2(3) NULL;
    PRINT 'Added EndEntryDateTime column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'Outcome')
BEGIN
    ALTER TABLE dbo.RunLog ADD Outcome NVARCHAR(32) NULL;
    PRINT 'Added Outcome column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'RowsRead')
BEGIN
    ALTER TABLE dbo.RunLog ADD RowsRead INT NULL;
    PRINT 'Added RowsRead column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'RowsWritten')
BEGIN
    ALTER TABLE dbo.RunLog ADD RowsWritten INT NULL;
    PRINT 'Added RowsWritten column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'ErrorJSON')
BEGIN
    ALTER TABLE dbo.RunLog ADD ErrorJSON NVARCHAR(MAX) NULL;
    PRINT 'Added ErrorJSON column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'TriggerReason')
BEGIN
    ALTER TABLE dbo.RunLog ADD TriggerReason NVARCHAR(64) NULL;
    PRINT 'Added TriggerReason column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'Version')
BEGIN
    ALTER TABLE dbo.RunLog ADD Version NVARCHAR(32) NULL;
    PRINT 'Added Version column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'ConfigHash')
BEGIN
    ALTER TABLE dbo.RunLog ADD ConfigHash NVARCHAR(64) NULL;
    PRINT 'Added ConfigHash column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'WindowStartEntryDateTime')
BEGIN
    ALTER TABLE dbo.RunLog ADD WindowStartEntryDateTime DATETIME2(3) NULL;
    PRINT 'Added WindowStartEntryDateTime column';
END

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.RunLog') AND name = 'WindowEndEntryDateTime')
BEGIN
    ALTER TABLE dbo.RunLog ADD WindowEndEntryDateTime DATETIME2(3) NULL;
    PRINT 'Added WindowEndEntryDateTime column';
END

PRINT 'RunLog schema fix completed';
GO

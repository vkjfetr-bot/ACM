-- ACM_RefitRequests table for SQL-mode retrain signaling
IF NOT EXISTS (SELECT 1 FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[ACM_RefitRequests]') AND type in (N'U'))
BEGIN
    CREATE TABLE [dbo].[ACM_RefitRequests] (
        [RequestID] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        [EquipID] INT NOT NULL,
        [RequestedAt] DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
        [Reason] NVARCHAR(MAX) NULL,
        [AnomalyRate] FLOAT NULL,
        [DriftScore] FLOAT NULL,
        [ModelAgeHours] FLOAT NULL,
        [RegimeQuality] FLOAT NULL,
        [Acknowledged] BIT NOT NULL DEFAULT 0,
        [AcknowledgedAt] DATETIME2 NULL
    );

    CREATE INDEX [IX_RefitRequests_EquipID_Ack] ON [dbo].[ACM_RefitRequests]([EquipID], [Acknowledged]);
END

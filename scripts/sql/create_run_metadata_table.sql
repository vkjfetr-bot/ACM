-- ACM_RunMetadata table creation script
-- Captures per-run forecasting retrain decisions and quality metrics
IF NOT EXISTS (
    SELECT 1 FROM sys.tables t JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE t.name = 'ACM_RunMetadata' AND s.name = 'dbo'
)
BEGIN
    CREATE TABLE dbo.ACM_RunMetadata (
        RunMetadataID INT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        EquipName NVARCHAR(128) NOT NULL,
        CreatedAt DATETIME2(0) NOT NULL DEFAULT SYSUTCDATETIME(),
        RetrainDecision BIT NOT NULL,
        RetrainReason NVARCHAR(256) NULL,
        ForecastStateVersion INT NULL,
        ModelAgeBatches INT NULL,
        ForecastRMSE FLOAT NULL,
        ForecastMAE FLOAT NULL,
        ForecastMAPE FLOAT NULL
    );

    CREATE INDEX IX_RunMetadata_RunID ON dbo.ACM_RunMetadata (RunID);
    CREATE INDEX IX_RunMetadata_EquipID ON dbo.ACM_RunMetadata (EquipID);
END;

GO
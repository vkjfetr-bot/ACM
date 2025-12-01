-- =============================================
-- ACM Detector Forecast Table
-- =============================================
-- Stores forecast trends for detector Z-scores (PCA, CUSUM, GMM, IForest, etc.)
-- Separate from ACM_SensorForecast_TS which stores physical sensor forecasts

USE ACM;
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_DetectorForecast_TS' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.ACM_DetectorForecast_TS (
        RunID           UNIQUEIDENTIFIER NOT NULL,
        EquipID         INT NOT NULL,
        DetectorName    NVARCHAR(100) NOT NULL,  -- e.g., 'pca_spe', 'cusum', 'gmm', 'iforest'
        Timestamp       DATETIME2 NOT NULL,
        ForecastValue   FLOAT NOT NULL,          -- Projected Z-score
        CiLower         FLOAT NULL,
        CiUpper         FLOAT NULL,
        ForecastStd     FLOAT NULL,
        Method          NVARCHAR(50) NOT NULL,   -- e.g., 'LinearTrend', 'ExponentialDecay'
        CreatedAt       DATETIME2 NOT NULL CONSTRAINT DF_ACM_DetectorForecast_TS_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT PK_ACM_DetectorForecast_TS PRIMARY KEY CLUSTERED (RunID, EquipID, DetectorName, Timestamp)
    );
    
    PRINT 'Created table: ACM_DetectorForecast_TS';
END
ELSE
BEGIN
    PRINT 'Table ACM_DetectorForecast_TS already exists';
END
GO

-- Add index for efficient time-series queries
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_ACM_DetectorForecast_TS_EquipID_Timestamp' AND object_id = OBJECT_ID('dbo.ACM_DetectorForecast_TS'))
BEGIN
    CREATE NONCLUSTERED INDEX IX_ACM_DetectorForecast_TS_EquipID_Timestamp 
    ON dbo.ACM_DetectorForecast_TS (EquipID, Timestamp)
    INCLUDE (DetectorName, ForecastValue, Method);
    
    PRINT 'Created index: IX_ACM_DetectorForecast_TS_EquipID_Timestamp';
END
GO

-- Add foreign key constraint to Equipment table
IF NOT EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_ACM_DetectorForecast_TS_Equipment')
BEGIN
    ALTER TABLE dbo.ACM_DetectorForecast_TS
    ADD CONSTRAINT FK_ACM_DetectorForecast_TS_Equipment
    FOREIGN KEY (EquipID) REFERENCES dbo.Equipment(EquipID);
    
    PRINT 'Created foreign key: FK_ACM_DetectorForecast_TS_Equipment';
END
GO

PRINT 'ACM_DetectorForecast_TS table setup complete';
GO

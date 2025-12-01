-- Create ACM_FeatureDropLog table for tracking dropped features
-- Used by acm_main.py to log features dropped during imputation/variance filtering

IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_FeatureDropLog')
BEGIN
    CREATE TABLE dbo.ACM_FeatureDropLog (
        LogID INT IDENTITY(1,1) PRIMARY KEY,
        RunID UNIQUEIDENTIFIER NOT NULL,
        EquipID INT NOT NULL,
        FeatureName NVARCHAR(500) NOT NULL,
        Reason NVARCHAR(200) NOT NULL,  -- 'all_nan', 'low_variance', etc.
        TrainMedian FLOAT NULL,
        TrainStd FLOAT NULL,
        Timestamp DATETIME NOT NULL DEFAULT GETDATE(),
        
        INDEX IX_ACM_FeatureDropLog_RunID (RunID),
        INDEX IX_ACM_FeatureDropLog_EquipID (EquipID),
        INDEX IX_ACM_FeatureDropLog_Timestamp (Timestamp DESC)
    );
    
    PRINT 'Created table ACM_FeatureDropLog';
END
ELSE
BEGIN
    PRINT 'Table ACM_FeatureDropLog already exists';
END
GO

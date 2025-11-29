-- =====================================================
-- DYNAMIC THRESHOLD METADATA TABLE
-- Purpose: Store adaptive thresholds per equipment/regime
-- =====================================================

USE ACM;
GO

-- Drop table if exists (for clean re-creation during development)
IF EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_ThresholdMetadata')
BEGIN
    DROP TABLE ACM_ThresholdMetadata;
    PRINT 'Dropped existing ACM_ThresholdMetadata table';
END
GO

-- Create table
CREATE TABLE ACM_ThresholdMetadata (
    ThresholdID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    RegimeID INT NULL,  -- NULL = global threshold for equipment
    ThresholdType VARCHAR(50) NOT NULL,  -- 'fused_alert_z', 'fused_warn_z', etc.
    ThresholdValue FLOAT NOT NULL,
    CalculationMethod VARCHAR(100) NOT NULL,  -- 'quantile_99.7', 'mad_3sigma', 'hardcoded', 'hybrid'
    SampleCount INT,  -- Number of training samples used
    TrainStartTime DATETIME2(3),
    TrainEndTime DATETIME2(3),
    CreatedAt DATETIME2(3) DEFAULT GETDATE(),
    ConfigSignature VARCHAR(32),  -- To invalidate on config changes
    IsActive BIT DEFAULT 1,
    Notes VARCHAR(500),  -- Optional explanation
    
    -- Composite index for fast lookup
    INDEX IX_ThresholdMetadata_Lookup (EquipID, RegimeID, ThresholdType, IsActive),
    INDEX IX_ThresholdMetadata_Created (CreatedAt DESC),
    
    -- Ensure Equipment exists
    FOREIGN KEY (EquipID) REFERENCES Equipment(EquipID)
);

PRINT 'Created ACM_ThresholdMetadata table';
PRINT '';

-- Add some example documentation
PRINT '=====================================================';
PRINT 'ACM_ThresholdMetadata Schema';
PRINT '=====================================================';
PRINT '';
PRINT 'Columns:';
PRINT '  ThresholdID       - Auto-increment primary key';
PRINT '  EquipID           - Equipment identifier (FK to Equipment)';
PRINT '  RegimeID          - Operating regime ID (NULL = global)';
PRINT '  ThresholdType     - Type of threshold (e.g., fused_alert_z)';
PRINT '  ThresholdValue    - Calculated threshold value';
PRINT '  CalculationMethod - Algorithm used (quantile, MAD, etc.)';
PRINT '  SampleCount       - Training sample size';
PRINT '  TrainStartTime    - Training data start';
PRINT '  TrainEndTime      - Training data end';
PRINT '  CreatedAt         - When threshold was calculated';
PRINT '  ConfigSignature   - Config hash for invalidation';
PRINT '  IsActive          - Whether threshold is currently active';
PRINT '  Notes             - Optional explanation';
PRINT '';
PRINT 'Usage Example:';
PRINT '  -- Get active alert threshold for Equipment 1, Regime 2';
PRINT '  SELECT ThresholdValue';
PRINT '  FROM ACM_ThresholdMetadata';
PRINT '  WHERE EquipID = 1';
PRINT '    AND RegimeID = 2';
PRINT '    AND ThresholdType = ''fused_alert_z''';
PRINT '    AND IsActive = 1';
PRINT '  ORDER BY CreatedAt DESC;';
PRINT '';
PRINT '  -- Fallback to global if per-regime not found';
PRINT '  SELECT TOP 1 ThresholdValue';
PRINT '  FROM ACM_ThresholdMetadata';
PRINT '  WHERE EquipID = 1';
PRINT '    AND RegimeID IS NULL';
PRINT '    AND ThresholdType = ''fused_alert_z''';
PRINT '    AND IsActive = 1';
PRINT '  ORDER BY CreatedAt DESC;';
PRINT '';
PRINT '=====================================================';
GO

-- Create a view for easy access to latest active thresholds
IF EXISTS (SELECT * FROM sys.views WHERE name = 'ACM_LatestThresholds')
    DROP VIEW ACM_LatestThresholds;
GO

CREATE VIEW ACM_LatestThresholds AS
SELECT 
    tm.EquipID,
    e.EquipName,
    tm.RegimeID,
    tm.ThresholdType,
    tm.ThresholdValue,
    tm.CalculationMethod,
    tm.SampleCount,
    tm.CreatedAt,
    tm.ConfigSignature
FROM ACM_ThresholdMetadata tm
INNER JOIN Equipment e ON tm.EquipID = e.EquipID
WHERE tm.IsActive = 1
AND tm.ThresholdID IN (
    -- Get latest threshold for each (EquipID, RegimeID, ThresholdType) combination
    SELECT MAX(ThresholdID)
    FROM ACM_ThresholdMetadata
    WHERE IsActive = 1
    GROUP BY EquipID, RegimeID, ThresholdType
);
GO

PRINT 'Created ACM_LatestThresholds view';
PRINT '';
PRINT 'View provides easy access to latest active thresholds';
PRINT 'Query: SELECT * FROM ACM_LatestThresholds WHERE EquipID = 1;';
GO

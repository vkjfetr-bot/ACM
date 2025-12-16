-- ACM Resource Metrics Table
-- Stores per-section resource usage for performance analysis
-- Created: 2025-12-16 for feature/resource-monitoring

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ACM_ResourceMetrics')
BEGIN
    CREATE TABLE ACM_ResourceMetrics (
        ID INT IDENTITY(1,1) PRIMARY KEY,
        RunID NVARCHAR(100) NOT NULL,
        EquipID INT NOT NULL,
        SectionName NVARCHAR(200) NOT NULL,
        DurationSeconds FLOAT NOT NULL,
        MemStartMB FLOAT NULL,
        MemEndMB FLOAT NULL,
        MemPeakMB FLOAT NULL,
        MemDeltaMB FLOAT NULL,
        CpuAvgPct FLOAT NULL,
        CpuSampleCount INT NULL,
        SectionDepth INT NULL,
        ThreadCount INT NULL,
        CreatedAt DATETIME2 DEFAULT GETDATE(),
        
        INDEX IX_ResourceMetrics_RunID (RunID),
        INDEX IX_ResourceMetrics_EquipID (EquipID),
        INDEX IX_ResourceMetrics_Section (SectionName),
        INDEX IX_ResourceMetrics_CreatedAt (CreatedAt)
    );
    
    PRINT 'Created ACM_ResourceMetrics table';
END
ELSE
BEGIN
    PRINT 'ACM_ResourceMetrics table already exists';
END
GO

-- View for quick performance analysis
IF EXISTS (SELECT * FROM sys.views WHERE name = 'vw_ACM_ResourceSummary')
    DROP VIEW vw_ACM_ResourceSummary;
GO

CREATE VIEW vw_ACM_ResourceSummary AS
SELECT 
    e.EquipName,
    rm.RunID,
    rm.SectionName,
    ROUND(AVG(rm.DurationSeconds), 3) AS AvgDurationS,
    ROUND(MAX(rm.DurationSeconds), 3) AS MaxDurationS,
    ROUND(AVG(rm.MemDeltaMB), 1) AS AvgMemDeltaMB,
    ROUND(MAX(rm.MemPeakMB), 0) AS MaxPeakMemMB,
    ROUND(AVG(rm.CpuAvgPct), 0) AS AvgCpuPct,
    COUNT(*) AS RunCount,
    MAX(rm.CreatedAt) AS LastRun
FROM ACM_ResourceMetrics rm
INNER JOIN Equipment e ON rm.EquipID = e.EquipID
GROUP BY e.EquipName, rm.RunID, rm.SectionName;
GO

PRINT 'Created vw_ACM_ResourceSummary view';
GO

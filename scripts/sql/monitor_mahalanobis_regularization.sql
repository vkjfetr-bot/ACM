-- ============================================================================
-- ACM Mahalanobis Regularization Monitoring Query
-- ============================================================================
-- Purpose: Monitor and track cases where Mahalanobis detector requires
--          significant regularization due to ill-conditioned covariance matrices.
-- 
-- Task: #14 from FIXES_ACTION_PLAN.md
-- Created: 2025-12-11
-- 
-- Background: High regularization (100x->1000x increases) indicates:
--   1. Near-singular covariance matrix (sensor collinearity)
--   2. Insufficient samples for covariance estimation
--   3. Numerical precision issues with sensor data
-- ============================================================================

USE ACM;
GO

-- ============================================================================
-- Query 1: Recent regularization events from run logs
-- ============================================================================
PRINT '=== Query 1: Recent Mahalanobis Regularization Events ===';

SELECT TOP 100
    rl.RunID,
    r.EquipName,
    r.StartedAt,
    rl.LoggedAt,
    rl.Level,
    rl.Message
FROM ACM_RunLogs rl
JOIN ACM_Runs r ON rl.RunID = r.RunID
WHERE rl.Message LIKE '%regularization%'
   OR rl.Message LIKE '%condition number%'
   OR rl.Message LIKE '%Mahalanobis%covariance%'
   OR rl.Message LIKE '%ill-conditioned%'
ORDER BY rl.LoggedAt DESC;
GO

-- ============================================================================
-- Query 2: Regularization trend by equipment
-- ============================================================================
PRINT '';
PRINT '=== Query 2: Regularization Events by Equipment (Last 30 days) ===';

SELECT 
    r.EquipName,
    COUNT(DISTINCT r.RunID) AS RunsWithRegularization,
    COUNT(*) AS TotalRegularizationEvents,
    MIN(rl.LoggedAt) AS FirstEvent,
    MAX(rl.LoggedAt) AS LastEvent
FROM ACM_RunLogs rl
JOIN ACM_Runs r ON rl.RunID = r.RunID
WHERE (rl.Message LIKE '%regularization%' 
   OR rl.Message LIKE '%condition number%')
  AND rl.LoggedAt >= DATEADD(DAY, -30, GETDATE())
GROUP BY r.EquipName
ORDER BY TotalRegularizationEvents DESC;
GO

-- ============================================================================
-- Query 3: Daily regularization frequency
-- ============================================================================
PRINT '';
PRINT '=== Query 3: Daily Regularization Frequency (Last 14 days) ===';

SELECT 
    CAST(rl.LoggedAt AS DATE) AS EventDate,
    COUNT(*) AS RegularizationEvents,
    COUNT(DISTINCT rl.RunID) AS AffectedRuns
FROM ACM_RunLogs rl
WHERE (rl.Message LIKE '%regularization%' 
   OR rl.Message LIKE '%condition number%')
  AND rl.LoggedAt >= DATEADD(DAY, -14, GETDATE())
GROUP BY CAST(rl.LoggedAt AS DATE)
ORDER BY EventDate DESC;
GO

-- ============================================================================
-- Query 4: Current Mahalanobis regularization config values
-- ============================================================================
PRINT '';
PRINT '=== Query 4: Current Mahalanobis Regularization Config ===';

SELECT 
    EquipID,
    ParamPath,
    ParamValue,
    UpdatedAt,
    UpdatedBy
FROM ACM_Config
WHERE ParamPath LIKE '%mahl%'
   OR ParamPath LIKE '%regularization%'
ORDER BY EquipID, ParamPath;
GO

-- ============================================================================
-- Query 5: High Mahalanobis Z-scores (indicates detector triggering)
-- ============================================================================
PRINT '';
PRINT '=== Query 5: High Mahalanobis Z-Scores (recent runs) ===';

SELECT TOP 50
    sw.RunID,
    r.EquipName,
    sw.Timestamp,
    sw.z_mahl AS Mahalanobis_Z
FROM ACM_Scores_Wide sw
JOIN ACM_Runs r ON sw.RunID = r.RunID
WHERE sw.z_mahl > 3.0  -- Elevated Mahalanobis score
ORDER BY sw.Timestamp DESC;
GO

-- ============================================================================
-- Query 6: Recommended actions based on findings
-- ============================================================================
PRINT '';
PRINT '=== Recommended Actions ===';
PRINT '';
PRINT 'If regularization events are frequent (>10% of runs):';
PRINT '  1. Review sensor selection - remove redundant/collinear sensors';
PRINT '  2. Increase mahl.regularization config value (current default: 0.1)';
PRINT '  3. Ensure minimum training samples (200+) for covariance estimation';
PRINT '';
PRINT 'Config update command (increase regularization):';
PRINT '  UPDATE ACM_Config SET ParamValue = ''1.0'' WHERE ParamPath = ''models.mahl.regularization'' AND EquipID = 0;';
PRINT '';
PRINT 'After config update, sync to SQL:';
PRINT '  python scripts/sql/populate_acm_config.py';
GO

-- ============================================================================
-- Create monitoring view for Grafana dashboard
-- ============================================================================
PRINT '';
PRINT '=== Creating Monitoring View ===';

IF OBJECT_ID('dbo.vw_MahalanobisRegularizationMonitor', 'V') IS NOT NULL
    DROP VIEW dbo.vw_MahalanobisRegularizationMonitor;
GO

CREATE VIEW dbo.vw_MahalanobisRegularizationMonitor AS
SELECT 
    CAST(rl.LoggedAt AS DATE) AS EventDate,
    r.EquipName,
    COUNT(*) AS RegularizationEvents,
    COUNT(DISTINCT rl.RunID) AS AffectedRuns
FROM ACM_RunLogs rl
JOIN ACM_Runs r ON rl.RunID = r.RunID
WHERE rl.Message LIKE '%regularization%' 
   OR rl.Message LIKE '%condition number%'
GROUP BY CAST(rl.LoggedAt AS DATE), r.EquipName;
GO

PRINT 'View created: dbo.vw_MahalanobisRegularizationMonitor';
PRINT '';
PRINT '=== Monitoring setup complete ===';
GO

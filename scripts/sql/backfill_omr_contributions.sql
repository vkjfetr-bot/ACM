-- ============================================================================
-- ACM OMR Contributions Backfill Script
-- ============================================================================
-- Purpose: Backfill historical ACM_OMRContributionsLong records that were lost
--          due to NULL constraint violations on ContributionScore column.
-- 
-- Task: #9 from FIXES_ACTION_PLAN.md
-- Created: 2025-12-11
-- 
-- IMPORTANT: This script identifies affected runs and provides queries to
-- diagnose the impact. Actual recalculation requires re-running ACM pipeline.
-- ============================================================================

USE ACM;
GO

-- ============================================================================
-- STEP 1: Identify affected runs (runs with missing OMR contributions)
-- ============================================================================
PRINT '=== STEP 1: Identifying affected runs ===';

-- Find runs that have ACM_Scores_Wide data but no OMR contributions
SELECT 
    r.RunID,
    r.EquipID,
    r.EquipName,
    r.StartedAt,
    r.CompletedAt,
    r.HealthStatus,
    sw.ScoreCount,
    COALESCE(oc.OMRCount, 0) AS OMRCount,
    CASE 
        WHEN COALESCE(oc.OMRCount, 0) = 0 THEN 'MISSING - needs backfill'
        WHEN oc.OMRCount < sw.ScoreCount * 5 THEN 'PARTIAL - may need review'
        ELSE 'OK'
    END AS OMRStatus
FROM ACM_Runs r
LEFT JOIN (
    SELECT RunID, COUNT(*) AS ScoreCount
    FROM ACM_Scores_Wide
    GROUP BY RunID
) sw ON r.RunID = sw.RunID
LEFT JOIN (
    SELECT RunID, COUNT(*) AS OMRCount
    FROM ACM_OMRContributionsLong
    GROUP BY RunID
) oc ON r.RunID = oc.RunID
WHERE sw.ScoreCount > 0  -- Run has score data
ORDER BY r.StartedAt DESC;
GO

-- ============================================================================
-- STEP 2: Count of affected runs by status
-- ============================================================================
PRINT '';
PRINT '=== STEP 2: Summary of affected runs ===';

SELECT 
    CASE 
        WHEN COALESCE(oc.OMRCount, 0) = 0 THEN 'MISSING'
        WHEN oc.OMRCount < sw.ScoreCount * 5 THEN 'PARTIAL'
        ELSE 'OK'
    END AS OMRStatus,
    COUNT(*) AS RunCount
FROM ACM_Runs r
LEFT JOIN (
    SELECT RunID, COUNT(*) AS ScoreCount
    FROM ACM_Scores_Wide
    GROUP BY RunID
) sw ON r.RunID = sw.RunID
LEFT JOIN (
    SELECT RunID, COUNT(*) AS OMRCount
    FROM ACM_OMRContributionsLong
    GROUP BY RunID
) oc ON r.RunID = oc.RunID
WHERE sw.ScoreCount > 0
GROUP BY 
    CASE 
        WHEN COALESCE(oc.OMRCount, 0) = 0 THEN 'MISSING'
        WHEN oc.OMRCount < sw.ScoreCount * 5 THEN 'PARTIAL'
        ELSE 'OK'
    END
ORDER BY 1;
GO

-- ============================================================================
-- STEP 3: Check for NULL ContributionScore values that slipped through
-- ============================================================================
PRINT '';
PRINT '=== STEP 3: Check for NULL ContributionScore values ===';

SELECT 
    COUNT(*) AS NullScoreCount
FROM ACM_OMRContributionsLong
WHERE ContributionScore IS NULL;
GO

-- ============================================================================
-- STEP 4: Get list of runs that need backfill (for batch re-processing)
-- ============================================================================
PRINT '';
PRINT '=== STEP 4: Runs requiring backfill ===';

-- This query generates the equipment and time ranges for backfill runs
SELECT DISTINCT
    r.EquipName,
    MIN(r.StartedAt) AS EarliestMissing,
    MAX(r.CompletedAt) AS LatestMissing,
    COUNT(DISTINCT r.RunID) AS MissingRunCount
FROM ACM_Runs r
LEFT JOIN (
    SELECT RunID, COUNT(*) AS OMRCount
    FROM ACM_OMRContributionsLong
    GROUP BY RunID
) oc ON r.RunID = oc.RunID
WHERE COALESCE(oc.OMRCount, 0) = 0
  AND EXISTS (
      SELECT 1 FROM ACM_Scores_Wide sw WHERE sw.RunID = r.RunID
  )
GROUP BY r.EquipName
ORDER BY r.EquipName;
GO

-- ============================================================================
-- STEP 5: Sample command to re-run ACM for backfill
-- ============================================================================
PRINT '';
PRINT '=== STEP 5: Backfill Commands (run in PowerShell) ===';
PRINT '';
PRINT '# Example backfill commands for missing OMR data:';
PRINT '# Replace date ranges with values from Step 4 above';
PRINT '';
PRINT '# For FD_FAN:';
PRINT '# python -m core.acm_main --equip FD_FAN --start-time "2024-01-01T00:00:00" --end-time "2024-12-31T23:59:59"';
PRINT '';
PRINT '# Or use batch runner:';
PRINT '# python scripts/sql_batch_runner.py --equip FD_FAN --tick-minutes 1440 --max-workers 2 --start-time "2024-01-01" --end-time "2024-12-31"';
PRINT '';
GO

-- ============================================================================
-- STEP 6: Verify backfill after running ACM
-- ============================================================================
PRINT '';
PRINT '=== STEP 6: Post-backfill verification query ===';
PRINT '';
PRINT '-- Run this after backfill to confirm fix:';
PRINT 'SELECT COUNT(*) AS TotalOMRRecords FROM ACM_OMRContributionsLong;';
PRINT 'SELECT COUNT(*) AS NullScores FROM ACM_OMRContributionsLong WHERE ContributionScore IS NULL;';
GO

-- End of backfill script
PRINT '';
PRINT '=== Backfill diagnosis complete ===';
GO

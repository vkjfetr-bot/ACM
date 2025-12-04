-- Comprehensive SQL validation tests for P0 fixes
-- Run this script to verify all critical fixes are in place

SET NOCOUNT ON;

PRINT '========================================';
PRINT 'ACM P0 Fixes Validation Report';
PRINT 'Date: ' + CAST(GETDATE() AS VARCHAR(30));
PRINT '========================================';
PRINT '';

-- 1. Verify backup tables are deleted
PRINT 'Test 1: Verify backup tables deleted';
PRINT '-------------------------------------';
DECLARE @backup_count INT;
SELECT @backup_count = COUNT(*) FROM sys.tables 
WHERE name LIKE '%BACKUP%' OR name LIKE '%TEMP%' AND name NOT LIKE 'temp%';
IF @backup_count = 0
    PRINT 'PASS: No backup tables found';
ELSE
    PRINT 'FAIL: Found ' + CAST(@backup_count AS VARCHAR(10)) + ' backup/temp tables';
PRINT '';

-- 2. Verify detector labels have full format
PRINT 'Test 2: Verify detector labels are full format';
PRINT '-------------------------------------';
SELECT TOP 10 dominant_sensor 
FROM ACM_EpisodeDiagnostics 
WHERE dominant_sensor IS NOT NULL 
AND dominant_sensor NOT IN ('UNKNOWN', '')
ORDER BY episode_id DESC;
PRINT '';

-- 3. Count valid vs invalid detector labels
DECLARE @valid_labels INT, @total_labels INT, @short_labels INT;
SELECT 
    @total_labels = COUNT(*),
    @valid_labels = COUNT(CASE WHEN dominant_sensor LIKE '%(%' THEN 1 END),
    @short_labels = COUNT(CASE WHEN dominant_sensor NOT LIKE '%(%' AND dominant_sensor NOT IN ('UNKNOWN', '') THEN 1 END)
FROM ACM_EpisodeDiagnostics
WHERE dominant_sensor IS NOT NULL;

PRINT 'Total detector labels: ' + CAST(@total_labels AS VARCHAR(10));
PRINT 'Valid full format: ' + CAST(@valid_labels AS VARCHAR(10));
PRINT 'Invalid short format: ' + CAST(@short_labels AS VARCHAR(10));
IF @short_labels = 0
    PRINT 'PASS: All labels in full format';
ELSE
    PRINT 'FAIL: Found ' + CAST(@short_labels AS VARCHAR(10)) + ' labels in short format';
PRINT '';

-- 4. Verify equipment names standardized
PRINT 'Test 3: Verify equipment names standardized';
PRINT '-------------------------------------';
SELECT r.EquipID, r.EquipName, e.EquipCode, e.EquipName as StandardName
FROM ACM_Runs r
LEFT JOIN Equipment e ON r.EquipID = e.EquipID
WHERE r.EquipName != e.EquipCode;

DECLARE @non_standard_count INT;
SELECT @non_standard_count = COUNT(*)
FROM ACM_Runs r
WHERE r.EquipName NOT IN (SELECT EquipCode FROM Equipment);

IF @non_standard_count = 0
    PRINT 'PASS: All equipment names standardized';
ELSE
    PRINT 'FAIL: Found ' + CAST(@non_standard_count AS VARCHAR(10)) + ' non-standard equipment names';
PRINT '';

-- 5. Verify all runs have completion time
PRINT 'Test 4: Verify all runs have completion time';
PRINT '-------------------------------------';
DECLARE @incomplete_count INT;
SELECT @incomplete_count = COUNT(*) FROM ACM_Runs WHERE CompletedAt IS NULL;

IF @incomplete_count = 0
    PRINT 'PASS: All runs have valid CompletedAt timestamp';
ELSE
    PRINT 'FAIL: Found ' + CAST(@incomplete_count AS VARCHAR(10)) + ' incomplete runs';
    SELECT RunID, EquipName, StartedAt, CompletedAt FROM ACM_Runs WHERE CompletedAt IS NULL;
PRINT '';

-- 6. Verify FinalizeRun procedure exists and is correct
PRINT 'Test 5: Verify FinalizeRun procedure';
PRINT '-------------------------------------';
IF OBJECT_ID('dbo.usp_ACM_FinalizeRun', 'P') IS NOT NULL
    PRINT 'PASS: usp_ACM_FinalizeRun procedure exists';
ELSE
    PRINT 'FAIL: usp_ACM_FinalizeRun procedure not found';

-- Check procedure references correct table
DECLARE @proc_text NVARCHAR(MAX);
SET @proc_text = OBJECT_DEFINITION(OBJECT_ID('dbo.usp_ACM_FinalizeRun'));
IF @proc_text LIKE '%ACM_Runs%'
    PRINT 'PASS: FinalizeRun references ACM_Runs table';
ELSE
    PRINT 'FAIL: FinalizeRun does not reference ACM_Runs table';
PRINT '';

-- 7. Summary statistics
PRINT 'Test 6: Database health summary';
PRINT '-------------------------------------';
DECLARE @table_count INT;
SELECT @table_count = COUNT(*) FROM sys.tables;
PRINT 'Total tables: ' + CAST(@table_count AS VARCHAR(10));

DECLARE @episode_count INT;
SELECT @episode_count = COUNT(*) FROM ACM_Episodes;
PRINT 'Total episodes detected: ' + CAST(@episode_count AS VARCHAR(10));

DECLARE @run_count INT;
SELECT @run_count = COUNT(*) FROM ACM_Runs;
PRINT 'Total runs in database: ' + CAST(@run_count AS VARCHAR(10));

DECLARE @completed_run_count INT;
SELECT @completed_run_count = COUNT(*) FROM ACM_Runs WHERE CompletedAt IS NOT NULL AND DurationSeconds > 0;
PRINT 'Completed runs (non-NOOP): ' + CAST(@completed_run_count AS VARCHAR(10));

DECLARE @noop_count INT;
SELECT @noop_count = COUNT(*) FROM ACM_Runs WHERE DurationSeconds = 0;
PRINT 'NOOP runs: ' + CAST(@noop_count AS VARCHAR(10));

DECLARE @score_count INT;
SELECT @score_count = COUNT(*) FROM ACM_Scores_Wide;
PRINT 'Total scoring records: ' + CAST(@score_count AS VARCHAR(10));

DECLARE @culprit_count INT;
SELECT @culprit_count = COUNT(*) FROM ACM_EpisodeCulprits;
PRINT 'Total culprit records: ' + CAST(@culprit_count AS VARCHAR(10));
PRINT '';

-- 8. Sample detector labels
PRINT 'Test 7: Sample detector labels (should be full format)';
PRINT '-------------------------------------';
SELECT TOP 20 
    episode_id, 
    dominant_sensor, 
    severity, 
    peak_z
FROM ACM_EpisodeDiagnostics
WHERE dominant_sensor IS NOT NULL 
ORDER BY episode_id DESC;
PRINT '';

-- 9. FK integrity check
PRINT 'Test 8: Foreign key integrity';
PRINT '-------------------------------------';
DECLARE @fk_violations INT;
SELECT @fk_violations = COUNT(*) FROM sys.foreign_keys 
WHERE OBJECT_NAME(referenced_object_id) IS NULL;

IF @fk_violations = 0
    PRINT 'PASS: No orphaned foreign keys';
ELSE
    PRINT 'FAIL: Found ' + CAST(@fk_violations AS VARCHAR(10)) + ' orphaned foreign keys';
PRINT '';

-- 10. Final summary
PRINT '========================================';
PRINT 'Validation Summary';
PRINT '========================================';
DECLARE @pass_count INT = 
    CASE WHEN @backup_count = 0 THEN 1 ELSE 0 END +
    CASE WHEN @short_labels = 0 THEN 1 ELSE 0 END +
    CASE WHEN @non_standard_count = 0 THEN 1 ELSE 0 END +
    CASE WHEN @incomplete_count = 0 THEN 1 ELSE 0 END +
    CASE WHEN @proc_text LIKE '%ACM_Runs%' THEN 1 ELSE 0 END +
    CASE WHEN @fk_violations = 0 THEN 1 ELSE 0 END;

PRINT 'Tests Passed: ' + CAST(@pass_count AS VARCHAR(2)) + '/6';
IF @pass_count = 6
    PRINT 'Status: ALL TESTS PASSED - System is ready for production';
ELSE
    PRINT 'Status: SOME TESTS FAILED - Review required';
PRINT '========================================';

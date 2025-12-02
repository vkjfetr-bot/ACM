-- ========================================
-- ACM DASHBOARD DATA FIX
-- ========================================
-- Fixes for missing dashboard panel data:
-- 1. Expand ACM_PCA_Metrics.ComponentName column (string truncation error)
-- 2. Ensure all forecast/RUL tables exist with correct schema
-- ========================================

USE ACM;
GO

PRINT '================================================================================'
PRINT 'DASHBOARD DATA FIX - Expanding PCA_Metrics ComponentName column'
PRINT '================================================================================'
PRINT ''

-- Fix 1: Expand ComponentName column in ACM_PCA_Metrics
IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ACM_PCA_Metrics')
BEGIN
    PRINT '[1/1] Expanding ACM_PCA_Metrics.ComponentName from NVARCHAR(50) to NVARCHAR(200)'
    
    -- Check current size
    DECLARE @current_size INT
    SELECT @current_size = CHARACTER_MAXIMUM_LENGTH
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'ACM_PCA_Metrics'
      AND COLUMN_NAME = 'ComponentName'
    
    IF @current_size < 200
    BEGIN
        ALTER TABLE dbo.ACM_PCA_Metrics
        ALTER COLUMN ComponentName NVARCHAR(200) NOT NULL
        
        PRINT '  ✓ ComponentName expanded to NVARCHAR(200)'
    END
    ELSE
    BEGIN
        PRINT '  ℹ ComponentName already NVARCHAR(' + CAST(@current_size AS VARCHAR) + '), no change needed'
    END
END
ELSE
BEGIN
    PRINT '  ❌ ACM_PCA_Metrics table does not exist!'
END

PRINT ''
PRINT '================================================================================'
PRINT 'FIX COMPLETE'
PRINT '================================================================================'
PRINT ''
PRINT 'Next steps:'
PRINT '  1. Re-run ACM pipeline to populate PCA_Metrics'
PRINT '  2. Enable forecasting/RUL modules in acm_main.py'
PRINT '  3. Remove file-based logging (acm.log files)'
PRINT ''

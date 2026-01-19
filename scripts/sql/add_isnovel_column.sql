-- v11.3.1: Add IsNovel column to ACM_RegimeTimeline
-- This column indicates points that are in sparse/novel regions of feature space.
-- Equipment is ALWAYS in some operating state, but these points have:
-- - Valid regime assignment (nearest cluster)
-- - Lower confidence (sparse region)
-- - IsNovel = 1 (candidate for new regime discovery)
--
-- Run this migration before deploying v11.3.1

-- Check if column already exists before adding
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'ACM_RegimeTimeline' 
    AND COLUMN_NAME = 'IsNovel'
)
BEGIN
    ALTER TABLE dbo.ACM_RegimeTimeline
    ADD IsNovel BIT NOT NULL DEFAULT 0;
    
    PRINT 'Added IsNovel column to ACM_RegimeTimeline';
END
ELSE
BEGIN
    PRINT 'IsNovel column already exists in ACM_RegimeTimeline';
END
GO

-- Add comment/description (only if not already present)
IF NOT EXISTS (
    SELECT 1 FROM sys.extended_properties ep
    INNER JOIN sys.columns c ON ep.major_id = c.object_id AND ep.minor_id = c.column_id
    INNER JOIN sys.tables t ON c.object_id = t.object_id
    WHERE t.name = 'ACM_RegimeTimeline' AND c.name = 'IsNovel' AND ep.name = 'MS_Description'
)
BEGIN
    EXEC sp_addextendedproperty 
        @name = N'MS_Description', 
        @value = N'v11.3.1: True if point is in sparse/novel region. Valid regime assigned but lower confidence. Candidate for new regime discovery.', 
        @level0type = N'SCHEMA', @level0name = N'dbo',
        @level1type = N'TABLE', @level1name = N'ACM_RegimeTimeline',
        @level2type = N'COLUMN', @level2name = N'IsNovel';
    PRINT 'Added extended property for IsNovel column';
END
ELSE
BEGIN
    PRINT 'Extended property for IsNovel already exists';
END
GO

# ACM DATABASE TABLE AUDIT - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

## EMPTY TABLES THAT SHOULD HAVE DATA (BUGS):

### 1. ACM_BaselineBuffer - MISSING DATA
- **Purpose**: Store baseline data for batch runs
- **Expected**: Should contain data from recent batch runs
- **Status**: 0 rows (written to in acm_main.py line 3545)
- **Action**: FIX - Baseline write logic may be failing

### 2. ACM_DataQuality - MISSING DATA  
- **Purpose**: Data quality metrics per sensor
- **Expected**: Quality metrics for each run
- **Status**: 0 rows (code exists in acm_main.py line 1358)
- **Action**: FIX - Data quality module not writing

## EMPTY TABLES THAT ARE PLANNED/OPTIONAL:

### 3. ACM_Drift_TS - Optional
- **Purpose**: Drift timeseries (alternative to ACM_DriftSeries)
- **Status**: 0 rows, ACM_DriftSeries has 18,629 rows
- **Action**: KEEP - May be used if drift module expanded

### 4. ACM_EnhancedFailureProbability_TS - Planned
- **Status**: 0 rows (in ALLOWED_TABLES, schema defined)
- **Action**: KEEP - Future enhancement

### 5. ACM_EnhancedMaintenanceRecommendation - Planned
- **Status**: 0 rows (in ALLOWED_TABLES, schema defined)  
- **Action**: KEEP - Future enhancement

### 6. ACM_FailureCausation - Planned
- **Status**: 0 rows (in ALLOWED_TABLES, schema defined)
- **Action**: KEEP - Future enhancement

### 7. ACM_Forecast_QualityMetrics - Planned
- **Status**: 0 rows (in ALLOWED_TABLES)
- **Action**: KEEP - Future forecast quality tracking

### 8. ACM_HealthForecast_Continuous - Planned
- **Purpose**: Continuous rolling forecast horizon
- **Status**: 0 rows (documented in Task Backlog.md)
- **Action**: KEEP - P1 feature for forecast continuity

## DUPLICATE/OBSOLETE TABLES:

### 9. ACM_OMRContributions - DUPLICATE
- **Status**: 0 rows
- **Replacement**: ACM_OMRContributionsLong (67,530 rows)
- **Action**: DROP - Wide format replaced by long format

### 10. ACM_FusionQuality - DUPLICATE  
- **Status**: 0 rows
- **Replacement**: ACM_FusionQualityReport (270 rows)
- **Action**: DROP - Old fusion quality table

### 11. ACM_FusionMetrics - DUPLICATE
- **Status**: 0 rows (in ALLOWED_TABLES but never used)
- **Action**: DROP - No references in code

### 12. ACM_ChartGenerationLog - OBSOLETE
- **Status**: 0 rows (in ALLOWED_TABLES)
- **Purpose**: Chart generation logging (Grafana handles this)
- **Action**: DROP - Not needed with Grafana

### 13. ACM_CulpritHistory - OBSOLETE
- **Status**: 0 rows
- **Replacement**: ACM_EpisodeCulprits (4,218 rows)
- **Action**: DROP - Replaced by enhanced episode system
- **Note**: Still referenced in old Grafana dashboard v2

### 14. ACM_DetectorContributions - OBSOLETE
- **Status**: 0 rows (never implemented)
- **Action**: DROP - Functionality in ACM_ContributionTimeline

### 15. ACM_DetectorMetadata - OBSOLETE
- **Status**: 0 rows (never implemented)
- **Action**: DROP - Not used

### 16. ACM_OMR_Metrics - OBSOLETE
- **Status**: 0 rows
- **Replacement**: ACM_OMR_Diagnostics (13 rows)
- **Action**: DROP - Consolidated into diagnostics

### 17. ACM_OMR_TopContributors - OBSOLETE
- **Status**: 0 rows
- **Replacement**: ACM_OMRContributionsLong aggregations
- **Action**: DROP - Use queries on long table

### 18. ACM_RecommendedActions - OBSOLETE
- **Status**: 0 rows (in ALLOWED_TABLES but no write logic)
- **Action**: DROP - Maintenance recommendations use different table

### 19. ACM_RegimeFeatureImportance - OBSOLETE
- **Status**: 0 rows (in ALLOWED_TABLES but never implemented)
- **Action**: DROP - Feature importance not tracked

### 20. ACM_RegimeSummary - OBSOLETE
- **Status**: 0 rows
- **Replacement**: ACM_RegimeStats (102 rows)
- **Action**: DROP - Consolidated into RegimeStats

## LEGACY TABLES (OLD SCHEMA):

### 21. AnomalyEvents - LEGACY
- **Status**: 0 rows
- **Replacement**: ACM_Anomaly_Events (89 rows)
- **Action**: DROP - Old naming convention

### 22. DriftTS - LEGACY
- **Status**: 0 rows  
- **Replacement**: ACM_DriftSeries (18,629 rows)
- **Action**: DROP - Old naming convention

### 23. PCA_Metrics - LEGACY
- **Status**: 0 rows
- **Replacement**: ACM_PCA_Metrics (45 rows)
- **Action**: DROP - Old naming convention

### 24. RegimeEpisodes - LEGACY
- **Status**: 0 rows
- **Replacement**: ACM_Regime_Episodes (89 rows)
- **Action**: DROP - Old naming convention

### 25. RunStats - LEGACY
- **Status**: 0 rows
- **Replacement**: ACM_Run_Stats (12 rows)
- **Action**: DROP - Old naming convention

### 26. ScoresTS - LEGACY
- **Status**: 0 rows
- **Replacement**: ACM_Scores_Long (109,170 rows), ACM_Scores_Wide (18,629 rows)
- **Action**: DROP - Old naming convention

### 27. Runs - LEGACY (POPULATED)
- **Status**: 3,941 rows
- **Replacement**: ACM_Runs (29 rows)
- **Note**: Old runs table still has data
- **Action**: ARCHIVE then DROP - Contains historical run data

### 28. RunLog - LEGACY (POPULATED)
- **Status**: 1,871 rows
- **Replacement**: ACM_RunLogs (163,153 rows)
- **Action**: ARCHIVE then DROP - Contains historical logs

### 29. PCA_Model - LEGACY (POPULATED)
- **Status**: 2 rows
- **Replacement**: ACM_PCA_Models (12 rows)
- **Action**: DROP - Minimal data, models in ModelRegistry

### 30. PCA_Components - LEGACY (POPULATED)
- **Status**: 1,160 rows
- **Replacement**: ACM_PCA_Loadings (6,805 rows)
- **Action**: DROP - Superseded by new table

## BACKUP TABLES:

### 31. ACM_PCA_Metrics_BACKUP_20251119
- **Status**: 0 rows (backup from 2025-11-19)
- **Action**: DROP - Empty backup, no longer needed

## EQUIPMENT DATA TABLES:

### 32. Equipments - DUPLICATE
- **Status**: 6 rows
- **Replacement**: Equipment (6 rows)
- **Action**: CONSOLIDATE - Merge into Equipment table

## SUMMARY:

**TOTAL TABLES**: 105
**ACTIVE/HEALTHY**: 75 tables with data
**BUGS (MISSING DATA)**: 2 tables
**PLANNED/FUTURE**: 6 tables  
**OBSOLETE**: 20 tables
**LEGACY**: 9 tables (7 empty, 2 with old data)
**BACKUPS**: 1 empty backup table

## RECOMMENDED ACTIONS:

### IMMEDIATE (BUGS):
1. Fix ACM_BaselineBuffer write logic
2. Fix ACM_DataQuality write logic

### CLEANUP (SAFE TO DROP):
3. Drop 20 obsolete tables
4. Drop 7 empty legacy tables
5. Drop 1 backup table
6. Archive Runs (3,941 rows) and RunLog (1,871 rows) before dropping

### CONSOLIDATION:
7. Merge Equipments into Equipment table
8. Update Grafana dashboard v2 to stop using ACM_CulpritHistory

### TOTAL TABLES TO DROP: 29 tables
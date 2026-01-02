# ACM V11 SQL Table & Dashboard Audit

**Generated**: 2025-12-31  
**Version**: 11.0.3  
**Purpose**: Comprehensive audit of SQL tables, dashboard queries, orphaned write methods, and timestamp standardization
**Status**: FIXES APPLIED - All orphaned write methods have been wired up in acm_main.py

---

## 1. CRITICAL BUG: Orphaned Write Methods

### 1.1 Tables with Write Methods That Are NEVER Called

These tables have fully implemented write methods in `core/output_manager.py` but the methods are **NEVER CALLED** from `acm_main.py` or any other pipeline code. This is a **BUG** - data is supposed to be written but isn't.

| Table | Write Method | Line in output_manager.py | Expected Call Location | Status |
|-------|--------------|---------------------------|------------------------|--------|
| ACM_CalibrationSummary | `write_calibration_summary()` | 2706 | After calibration phase | ORPHANED |
| ACM_RegimeOccupancy | `write_regime_occupancy()` | 2723 | After regime labeling | ORPHANED |
| ACM_RegimeTransitions | `write_regime_transitions()` | 2740 | After regime labeling | ORPHANED |
| ACM_ContributionTimeline | `write_contribution_timeline()` | 2768 | After sensor attribution | ORPHANED |
| ACM_RegimePromotionLog | `write_regime_promotion_log()` | 2785 | After model lifecycle | ORPHANED |
| ACM_DriftController | `write_drift_controller()` | 2802 | After drift detection | ORPHANED |

**NOT Orphaned (verified):**
- ACM_FeatureDropLog (550 rows) - Written via direct SQL in `_log_dropped_features()` at acm_main.py:1816
- ACM_DriftSeries (0 rows) - Called via `write_drift_ts()` alias at acm_main.py:3207, just no drift detected

### 1.2 Fixes Applied in acm_main.py (v11.0.2 -> v11.0.3)

All 6 orphaned write methods have been wired up with calls at appropriate pipeline phases:

| Table | Write Method | Added Location | Line |
|-------|--------------|----------------|------|
| ACM_CalibrationSummary | `write_calibration_summary()` | After calibration/thresholds | ~4955 |
| ACM_RegimeOccupancy | `write_regime_occupancy()` | After regime labeling | ~4530 |
| ACM_RegimeTransitions | `write_regime_transitions()` | After regime labeling | ~4545 |
| ACM_RegimePromotionLog | `write_regime_promotion_log()` | After model promotion | ~4780 |
| ACM_DriftController | `write_drift_controller()` | After drift detection | ~5365 |
| ACM_ContributionTimeline | `write_contribution_timeline()` | Before persist phase | ~5510 |

Each write is wrapped in try/except with appropriate Console.warn() on failure.

---

## 2. Dashboard Query Audit

### 2.1 All 49 Panel Queries - Verified Correct

| Dashboard | Panels | Tables Used | Status |
|-----------|--------|-------------|--------|
| Executive | 12 | ACM_HealthTimeline, ACM_RegimeTimeline, ACM_RUL, ACM_ActiveModels, ACM_Anomaly_Events, ACM_SensorHotspots | All OK |
| Detectors | 10 | ACM_Scores_Wide, ACM_DetectorCorrelation, ACM_RegimeTimeline | All OK |
| Forecasting | 7 | ACM_RUL, ACM_HealthForecast, ACM_FailureForecast | All OK |
| Diagnostics | 8 | ACM_SensorHotspots, ACM_SensorDefects, ACM_EpisodeDiagnostics, ACM_Anomaly_Events, ACM_EpisodeCulprits, ACM_SensorCorrelations | All OK |
| Operations | 12 | ACM_Runs, ACM_ActiveModels, ACM_DataContractValidation, ACM_RunLogs | All OK |

### 2.2 Column Name Clarification

| Concept | Table | Column | Notes |
|---------|-------|--------|-------|
| Fused Score (raw) | ACM_Scores_Wide | `fused` | Raw weighted fusion value |
| Fused Z-Score | ACM_HealthTimeline | `FusedZ` | Standardized z-score |

**These are DIFFERENT columns by design** - ACM_Scores_Wide stores detector outputs, ACM_HealthTimeline stores derived health metrics.

---

## 3. Table Classification

### 3.1 ACTIVE Tables (Written and Working)

| Table | Write Method/Generic | Called From | Rows |
|-------|---------------------|-------------|------|
| ACM_HealthTimeline | `write_table()` | acm_main.py L3500+ | 2,847 |
| ACM_Scores_Wide | `write_scores_wide()` | acm_main.py L2800+ | 713 |
| ACM_RegimeTimeline | `write_table()` | acm_main.py L3100+ | 713 |
| ACM_RUL | `write_table()` | forecast_engine.py | 41 |
| ACM_HealthForecast | `write_table()` | forecast_engine.py | 336 |
| ACM_FailureForecast | `write_table()` | forecast_engine.py | 336 |
| ACM_SensorForecast | `write_table()` | forecast_engine.py | 0 |
| ACM_Anomaly_Events | `write_anomaly_events()` | acm_main.py L3200+ | 16 |
| ACM_SensorHotspots | `write_table()` | acm_main.py L3600+ | 186 |
| ACM_SensorDefects | `write_table()` | acm_main.py L3650+ | 41 |
| ACM_EpisodeDiagnostics | `write_table()` | acm_main.py L3250+ | 16 |
| ACM_EpisodeCulprits | `write_table()` | episode_culprits_writer.py | 80 |
| ACM_DetectorCorrelation | `write_detector_correlation()` | acm_main.py L2900+ | 50 |
| ACM_SensorCorrelations | `write_sensor_correlation()` | acm_main.py L3550+ | 3,245 |
| ACM_Runs | `write_table()` | acm_main.py L4000+ | 41 |
| ACM_RunLogs | `write_table()` | observability.py | 10,437 |
| ACM_Run_Stats | `write_run_stats()` | acm_main.py L3900+ | 52 |
| ACM_ActiveModels | `write_active_models()` | acm_main.py L3150+ | 7 |
| ACM_RegimeDefinitions | `write_regime_definitions()` | acm_main.py L3100+ | 3 |
| ACM_DataContractValidation | `write_data_contract_validation()` | acm_main.py L1500+ | 26 |
| ACM_SeasonalPatterns | `write_seasonal_patterns()` | acm_main.py L2500+ | 0 |
| ACM_AssetProfiles | `write_asset_profile()` | acm_main.py L3700+ | 1 |
| ACM_PCA_Loadings | `write_pca_loadings()` | acm_main.py L2850+ | 170 |
| ACM_PCA_Metrics | `write_pca_metrics()` | acm_main.py L2850+ | 51 |
| ACM_PCA_Models | `write_pca_models()` | acm_main.py L2850+ | 5 |
| ACM_SensorNormalized_TS | `write_sensor_normalized_ts()` | acm_main.py L3500+ | 7,130 |
| ACM_BaselineBuffer | `write_table()` | acm_main.py L2000+ | 1,426 |
| ACM_Config | `write_table()` | populate_acm_config.py | 59 |
| ACM_ConfigHistory | `write_table()` | config_history_writer.py | 80 |
| ACM_ColdstartState | `write_table()` | sql_batch_runner.py | 2 |
| ACM_Thresholds | `write_table()` | acm_main.py L2950+ | 14 |
| ACM_Regime_Episodes | `write_regime_episodes()` | acm_main.py L3250+ | 6 |

### 3.2 ORPHANED Tables (Write Method Exists, NEVER Called)

| Table | Write Method | Rows | Fix Required |
|-------|--------------|------|--------------|
| ACM_CalibrationSummary | `write_calibration_summary()` | 0 | Call after calibration |
| ACM_RegimeOccupancy | `write_regime_occupancy()` | 0 | Call after regime labeling |
| ACM_RegimeTransitions | `write_regime_transitions()` | 0 | Call after regime labeling |
| ACM_ContributionTimeline | `write_contribution_timeline()` | 0 | Call after sensor attribution |
| ACM_RegimePromotionLog | `write_regime_promotion_log()` | 0 | Call after model lifecycle |
| ACM_DriftController | `write_drift_controller()` | 0 | Call after drift detection |

**NOT Orphaned (verified with data):**
- ACM_FeatureDropLog (550 rows) - Written via direct SQL in `_log_dropped_features()`
- ACM_DriftSeries (0 rows) - Called via `write_drift_ts()` at acm_main.py:3207

**Unused Write Methods (can be deleted from output_manager.py):**
- `write_feature_drop_log()` - Superseded by direct SQL in `_log_dropped_features()`

### 3.3 Tables in ALLOWED_TABLES Without Write Methods (Planned)

| Table | Purpose | Status |
|-------|---------|--------|
| ACM_Episodes | Legacy episode format | May be deprecated |
| ACM_DataQuality | Data quality metrics | Use write_table() |
| ACM_ForecastingState | Forecast model state | Use write_table() |
| ACM_AdaptiveConfig | Adaptive parameters | Use write_table() |
| ACM_RefitRequests | Refit queue | Use write_table() |
| ACM_RunMetadata | Extended run info | Use write_table() |
| ACM_RunMetrics | Run timing metrics | Use write_table() |
| ACM_HistorianData | Cache table | Populated on demand |
| ACM_OMR_Diagnostics | OMR debug info | Use write_table() |

### 3.4 OBSOLETE Tables (NOT in ALLOWED_TABLES, 0 Rows)

These tables are truly obsolete - not in ALLOWED_TABLES, no write methods, no data:

| Table | Recommendation |
|-------|----------------|
| ACM_AlertAge | DELETE |
| ACM_ContributionCurrent | DELETE |
| ACM_DailyFusedProfile | DELETE |
| ACM_DefectSummary | DELETE |
| ACM_DefectTimeline | DELETE |
| ACM_DetectorForecast_TS | DELETE |
| ACM_DriftEvents | DELETE |
| ACM_EnhancedFailureProbability_TS | DELETE |
| ACM_EnhancedMaintenanceRecommendation | DELETE |
| ACM_EpisodeMetrics | DELETE |
| ACM_FailureCausation | DELETE |
| ACM_FailureForecast_TS | DELETE |
| ACM_FailureHazard_TS | DELETE |
| ACM_ForecastState | DELETE |
| ACM_FusionQualityReport | DELETE |
| ACM_HealthDistributionOverTime | DELETE |
| ACM_HealthForecast_Continuous | DELETE |
| ACM_HealthForecast_TS | DELETE |
| ACM_HealthHistogram | DELETE |
| ACM_HealthZoneByPeriod | DELETE |
| ACM_MaintenanceRecommendation | DELETE |
| ACM_OMRContributionsLong | DELETE |
| ACM_OMRTimeline | DELETE |
| ACM_RecommendedActions | DELETE |
| ACM_RegimeDwellStats | DELETE |
| ACM_RegimeStability | DELETE |
| ACM_RegimeStats | DELETE |
| ACM_RUL_Attribution | DELETE |
| ACM_RUL_LearningState | DELETE |
| ACM_RUL_Summary | DELETE |
| ACM_RUL_TS | DELETE |
| ACM_Scores_Long | DELETE |
| ACM_SensorAnomalyByPeriod | DELETE |
| ACM_SensorForecast_TS | DELETE |
| ACM_SensorHotspotTimeline | DELETE |
| ACM_SensorRanking | DELETE |
| ACM_SinceWhen | DELETE |
| ACM_ThresholdCrossings | DELETE |
| ACM_ThresholdMetadata | DELETE |

**Total: 39 tables for deletion**

---

## 4. Timestamp Column Standardization

### 4.1 Standard Column Names

| Column | Purpose | Usage |
|--------|---------|-------|
| `CreatedAt` | When record was inserted | All non-time-series tables |
| `ModifiedAt` | When record was last updated | Tables that support updates |
| `Timestamp` | Time of the measurement/event | Time-series tables only |

### 4.2 Current State Audit

| Table | Has CreatedAt | Has ModifiedAt | Current Names | Action |
|-------|---------------|----------------|---------------|--------|
| ACM_ActiveModels | NO | NO | LastUpdatedAt | ADD CreatedAt, RENAME LastUpdatedAt to ModifiedAt |
| ACM_AdaptiveConfig | YES | NO | UpdatedAt | RENAME UpdatedAt to ModifiedAt |
| ACM_Anomaly_Events | NO | NO | StartTime, EndTime | ADD CreatedAt |
| ACM_AssetProfiles | NO | NO | LastUpdatedAt | ADD CreatedAt, RENAME LastUpdatedAt to ModifiedAt |
| ACM_BaselineBuffer | YES | NO | | OK |
| ACM_CalibrationSummary | YES | NO | | OK |
| ACM_ColdstartState | YES | NO | UpdatedAt | RENAME UpdatedAt to ModifiedAt |
| ACM_Config | YES | NO | UpdatedAt | RENAME UpdatedAt to ModifiedAt |
| ACM_ConfigHistory | YES | NO | | OK |
| ACM_DataContractValidation | NO | NO | ValidatedAt | RENAME ValidatedAt to CreatedAt |
| ACM_DataQuality | NO | NO | NONE | ADD CreatedAt |
| ACM_DetectorCorrelation | NO | NO | CalculatedAt | RENAME CalculatedAt to CreatedAt |
| ACM_DriftController | YES | NO | | OK |
| ACM_DriftSeries | YES | NO | Timestamp | OK |
| ACM_EpisodeCulprits | YES | NO | | OK |
| ACM_EpisodeDiagnostics | NO | NO | NONE | ADD CreatedAt |
| ACM_Episodes | NO | NO | StartTime, EndTime | ADD CreatedAt |
| ACM_FailureForecast | NO | NO | Timestamp | ADD CreatedAt |
| ACM_FeatureDropLog | NO | NO | DroppedAt | RENAME DroppedAt to CreatedAt |
| ACM_ForecastingState | YES | NO | UpdatedAt | RENAME UpdatedAt to ModifiedAt |
| ACM_HealthForecast | YES | NO | Timestamp | OK |
| ACM_HealthTimeline | NO | NO | Timestamp | ADD CreatedAt |
| ACM_HistorianData | YES | NO | Timestamp | OK |
| ACM_OMR_Diagnostics | YES | NO | | OK |
| ACM_PCA_Loadings | YES | NO | | OK |
| ACM_PCA_Metrics | YES | NO | | OK |
| ACM_PCA_Models | YES | NO | | OK |
| ACM_RefitRequests | YES | NO | ProcessedAt | OK |
| ACM_RegimeDefinitions | YES | NO | | OK |
| ACM_RegimeTimeline | NO | NO | Timestamp | ADD CreatedAt |
| ACM_Regime_Episodes | NO | NO | StartTime, EndTime | ADD CreatedAt |
| ACM_RUL | YES | NO | | OK |
| ACM_RunLogs | NO | NO | LoggedAt | RENAME LoggedAt to CreatedAt |
| ACM_RunMetrics | NO | NO | NONE | ADD CreatedAt |
| ACM_Runs | YES | NO | StartedAt, CompletedAt | OK |
| ACM_Run_Stats | YES | NO | | OK |
| ACM_Scores_Wide | NO | NO | Timestamp | ADD CreatedAt |
| ACM_SeasonalPatterns | YES | NO | | OK |
| ACM_SensorCorrelations | NO | NO | CalculatedAt | RENAME CalculatedAt to CreatedAt |
| ACM_SensorDefects | NO | NO | DetectedAt | Keep DetectedAt (semantic), ADD CreatedAt |
| ACM_SensorForecast | NO | NO | Timestamp | ADD CreatedAt |
| ACM_SensorHotspots | NO | NO | NONE | ADD CreatedAt |
| ACM_SensorNormalized_TS | YES | NO | Timestamp | OK |
| ACM_Thresholds | YES | NO | | OK |

### 4.3 Summary of Changes Needed

| Action | Count | Tables |
|--------|-------|--------|
| Already OK | 18 | ACM_BaselineBuffer, ACM_CalibrationSummary, etc. |
| ADD CreatedAt | 15 | ACM_Anomaly_Events, ACM_DataQuality, ACM_EpisodeDiagnostics, etc. |
| RENAME to CreatedAt | 5 | ValidatedAt, CalculatedAt, DroppedAt, LoggedAt |
| RENAME to ModifiedAt | 5 | UpdatedAt (4 tables), LastUpdatedAt (1 table) |
| ADD CreatedAt + RENAME | 2 | ACM_ActiveModels, ACM_AssetProfiles |

---

## 5. SQL Migration Scripts

### 5.1 Add CreatedAt Columns

```sql
-- Tables missing CreatedAt (15 tables)
ALTER TABLE dbo.ACM_Anomaly_Events ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_DataQuality ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_EpisodeDiagnostics ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_Episodes ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_FailureForecast ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_HealthTimeline ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_RegimeTimeline ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_Regime_Episodes ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_RunMetrics ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_Scores_Wide ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_SensorDefects ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_SensorForecast ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_SensorHotspots ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_ActiveModels ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
ALTER TABLE dbo.ACM_AssetProfiles ADD CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE();
```

### 5.2 Standardize to CreatedAt (Rename Non-Standard)

```sql
-- Rename non-standard timestamp columns to CreatedAt
EXEC sp_rename 'dbo.ACM_DataContractValidation.ValidatedAt', 'CreatedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_DetectorCorrelation.CalculatedAt', 'CreatedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_FeatureDropLog.DroppedAt', 'CreatedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_RunLogs.LoggedAt', 'CreatedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_SensorCorrelations.CalculatedAt', 'CreatedAt', 'COLUMN';
```

### 5.3 Standardize to ModifiedAt (Rename UpdatedAt)

```sql
-- Rename UpdatedAt to ModifiedAt for consistency
EXEC sp_rename 'dbo.ACM_AdaptiveConfig.UpdatedAt', 'ModifiedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_ColdstartState.UpdatedAt', 'ModifiedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_Config.UpdatedAt', 'ModifiedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_ForecastingState.UpdatedAt', 'ModifiedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_ActiveModels.LastUpdatedAt', 'ModifiedAt', 'COLUMN';
EXEC sp_rename 'dbo.ACM_AssetProfiles.LastUpdatedAt', 'ModifiedAt', 'COLUMN';
```

### 5.4 Drop Obsolete Tables

```sql
-- 39 obsolete tables (NOT in ALLOWED_TABLES, 0 rows, no code references)
DROP TABLE IF EXISTS dbo.ACM_AlertAge;
DROP TABLE IF EXISTS dbo.ACM_ContributionCurrent;
DROP TABLE IF EXISTS dbo.ACM_DailyFusedProfile;
DROP TABLE IF EXISTS dbo.ACM_DefectSummary;
DROP TABLE IF EXISTS dbo.ACM_DefectTimeline;
DROP TABLE IF EXISTS dbo.ACM_DetectorForecast_TS;
DROP TABLE IF EXISTS dbo.ACM_DriftEvents;
DROP TABLE IF EXISTS dbo.ACM_EnhancedFailureProbability_TS;
DROP TABLE IF EXISTS dbo.ACM_EnhancedMaintenanceRecommendation;
DROP TABLE IF EXISTS dbo.ACM_EpisodeMetrics;
DROP TABLE IF EXISTS dbo.ACM_FailureCausation;
DROP TABLE IF EXISTS dbo.ACM_FailureForecast_TS;
DROP TABLE IF EXISTS dbo.ACM_FailureHazard_TS;
DROP TABLE IF EXISTS dbo.ACM_ForecastState;
DROP TABLE IF EXISTS dbo.ACM_FusionQualityReport;
DROP TABLE IF EXISTS dbo.ACM_HealthDistributionOverTime;
DROP TABLE IF EXISTS dbo.ACM_HealthForecast_Continuous;
DROP TABLE IF EXISTS dbo.ACM_HealthForecast_TS;
DROP TABLE IF EXISTS dbo.ACM_HealthHistogram;
DROP TABLE IF EXISTS dbo.ACM_HealthZoneByPeriod;
DROP TABLE IF EXISTS dbo.ACM_MaintenanceRecommendation;
DROP TABLE IF EXISTS dbo.ACM_OMRContributionsLong;
DROP TABLE IF EXISTS dbo.ACM_OMRTimeline;
DROP TABLE IF EXISTS dbo.ACM_RecommendedActions;
DROP TABLE IF EXISTS dbo.ACM_RegimeDwellStats;
DROP TABLE IF EXISTS dbo.ACM_RegimeStability;
DROP TABLE IF EXISTS dbo.ACM_RegimeStats;
DROP TABLE IF EXISTS dbo.ACM_RUL_Attribution;
DROP TABLE IF EXISTS dbo.ACM_RUL_LearningState;
DROP TABLE IF EXISTS dbo.ACM_RUL_Summary;
DROP TABLE IF EXISTS dbo.ACM_RUL_TS;
DROP TABLE IF EXISTS dbo.ACM_Scores_Long;
DROP TABLE IF EXISTS dbo.ACM_SensorAnomalyByPeriod;
DROP TABLE IF EXISTS dbo.ACM_SensorForecast_TS;
DROP TABLE IF EXISTS dbo.ACM_SensorHotspotTimeline;
DROP TABLE IF EXISTS dbo.ACM_SensorRanking;
DROP TABLE IF EXISTS dbo.ACM_SinceWhen;
DROP TABLE IF EXISTS dbo.ACM_ThresholdCrossings;
DROP TABLE IF EXISTS dbo.ACM_ThresholdMetadata;
```

---

## 6. Summary

| Category | Count | Action |
|----------|-------|--------|
| ACTIVE tables (working) | 34 | None (includes ACM_DriftSeries, ACM_FeatureDropLog) |
| ORPHANED tables (bug) | 6 | Fix by calling write methods in acm_main.py |
| PLANNED tables (no write method) | 9 | Use generic write_table() when needed |
| OBSOLETE tables | 39 | DELETE from database |
| Tables needing CreatedAt | 15 | Run ALTER TABLE |
| Columns to rename to CreatedAt | 5 | Run sp_rename |
| Columns to rename to ModifiedAt | 6 | Run sp_rename |

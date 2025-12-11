# ACM Comprehensive Database Schema Reference

_Generated automatically on 2025-12-11 11:01:26_

This document provides detailed information about all tables in the ACM database:
- Schema (columns, data types, nullability, defaults)
- Primary keys
- Row counts and date ranges
- Top 10 and bottom 10 records per table

**Generation Command:**
```bash
python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md
```

---

## Table of Contents

- [dbo.ACM_AdaptiveConfig](#dboacmadaptiveconfig)
- [dbo.ACM_AdaptiveThresholds_ByRegime](#dboacmadaptivethresholdsbyregime)
- [dbo.ACM_AlertAge](#dboacmalertage)
- [dbo.ACM_Anomaly_Events](#dboacmanomalyevents)
- [dbo.ACM_BaselineBuffer](#dboacmbaselinebuffer)
- [dbo.ACM_CalibrationSummary](#dboacmcalibrationsummary)
- [dbo.ACM_ColdstartState](#dboacmcoldstartstate)
- [dbo.ACM_Config](#dboacmconfig)
- [dbo.ACM_ConfigHistory](#dboacmconfighistory)
- [dbo.ACM_ContributionCurrent](#dboacmcontributioncurrent)
- [dbo.ACM_ContributionTimeline](#dboacmcontributiontimeline)
- [dbo.ACM_DailyFusedProfile](#dboacmdailyfusedprofile)
- [dbo.ACM_DataQuality](#dboacmdataquality)
- [dbo.ACM_DefectSummary](#dboacmdefectsummary)
- [dbo.ACM_DefectTimeline](#dboacmdefecttimeline)
- [dbo.ACM_DetectorCorrelation](#dboacmdetectorcorrelation)
- [dbo.ACM_DetectorForecast_TS](#dboacmdetectorforecastts)
- [dbo.ACM_DriftEvents](#dboacmdriftevents)
- [dbo.ACM_DriftSeries](#dboacmdriftseries)
- [dbo.ACM_EnhancedFailureProbability_TS](#dboacmenhancedfailureprobabilityts)
- [dbo.ACM_EnhancedMaintenanceRecommendation](#dboacmenhancedmaintenancerecommendation)
- [dbo.ACM_EpisodeCulprits](#dboacmepisodeculprits)
- [dbo.ACM_EpisodeDiagnostics](#dboacmepisodediagnostics)
- [dbo.ACM_EpisodeMetrics](#dboacmepisodemetrics)
- [dbo.ACM_Episodes](#dboacmepisodes)
- [dbo.ACM_EpisodesQC](#dboacmepisodesqc)
- [dbo.ACM_FailureCausation](#dboacmfailurecausation)
- [dbo.ACM_FailureForecast](#dboacmfailureforecast)
- [dbo.ACM_FailureForecast_TS](#dboacmfailureforecastts)
- [dbo.ACM_FailureHazard_TS](#dboacmfailurehazardts)
- [dbo.ACM_FeatureDropLog](#dboacmfeaturedroplog)
- [dbo.ACM_ForecastContext](#dboacmforecastcontext)
- [dbo.ACM_ForecastState](#dboacmforecaststate)
- [dbo.ACM_ForecastingState](#dboacmforecastingstate)
- [dbo.ACM_FusionQualityReport](#dboacmfusionqualityreport)
- [dbo.ACM_HealthDistributionOverTime](#dboacmhealthdistributionovertime)
- [dbo.ACM_HealthForecast](#dboacmhealthforecast)
- [dbo.ACM_HealthForecast_Continuous](#dboacmhealthforecastcontinuous)
- [dbo.ACM_HealthForecast_TS](#dboacmhealthforecastts)
- [dbo.ACM_HealthHistogram](#dboacmhealthhistogram)
- [dbo.ACM_HealthTimeline](#dboacmhealthtimeline)
- [dbo.ACM_HealthZoneByPeriod](#dboacmhealthzonebyperiod)
- [dbo.ACM_HistorianData](#dboacmhistoriandata)
- [dbo.ACM_MaintenanceRecommendation](#dboacmmaintenancerecommendation)
- [dbo.ACM_OMRContributionsLong](#dboacmomrcontributionslong)
- [dbo.ACM_OMRTimeline](#dboacmomrtimeline)
- [dbo.ACM_OMR_Diagnostics](#dboacmomrdiagnostics)
- [dbo.ACM_PCA_Loadings](#dboacmpcaloadings)
- [dbo.ACM_PCA_Metrics](#dboacmpcametrics)
- [dbo.ACM_PCA_Models](#dboacmpcamodels)
- [dbo.ACM_RUL](#dboacmrul)
- [dbo.ACM_RUL_Attribution](#dboacmrulattribution)
- [dbo.ACM_RUL_ByRegime](#dboacmrulbyregime)
- [dbo.ACM_RUL_LearningState](#dboacmrullearningstate)
- [dbo.ACM_RUL_Summary](#dboacmrulsummary)
- [dbo.ACM_RUL_TS](#dboacmrults)
- [dbo.ACM_RecommendedActions](#dboacmrecommendedactions)
- [dbo.ACM_RefitRequests](#dboacmrefitrequests)
- [dbo.ACM_RegimeDwellStats](#dboacmregimedwellstats)
- [dbo.ACM_RegimeHazard](#dboacmregimehazard)
- [dbo.ACM_RegimeOccupancy](#dboacmregimeoccupancy)
- [dbo.ACM_RegimeStability](#dboacmregimestability)
- [dbo.ACM_RegimeState](#dboacmregimestate)
- [dbo.ACM_RegimeStats](#dboacmregimestats)
- [dbo.ACM_RegimeTimeline](#dboacmregimetimeline)
- [dbo.ACM_RegimeTransitions](#dboacmregimetransitions)
- [dbo.ACM_Regime_Episodes](#dboacmregimeepisodes)
- [dbo.ACM_RunLogs](#dboacmrunlogs)
- [dbo.ACM_RunMetadata](#dboacmrunmetadata)
- [dbo.ACM_RunMetrics](#dboacmrunmetrics)
- [dbo.ACM_Run_Stats](#dboacmrunstats)
- [dbo.ACM_Runs](#dboacmruns)
- [dbo.ACM_SchemaVersion](#dboacmschemaversion)
- [dbo.ACM_Scores_Long](#dboacmscoreslong)
- [dbo.ACM_Scores_Wide](#dboacmscoreswide)
- [dbo.ACM_SensorAnomalyByPeriod](#dboacmsensoranomalybyperiod)
- [dbo.ACM_SensorDefects](#dboacmsensordefects)
- [dbo.ACM_SensorForecast](#dboacmsensorforecast)
- [dbo.ACM_SensorForecast_TS](#dboacmsensorforecastts)
- [dbo.ACM_SensorHotspotTimeline](#dboacmsensorhotspottimeline)
- [dbo.ACM_SensorHotspots](#dboacmsensorhotspots)
- [dbo.ACM_SensorNormalized_TS](#dboacmsensornormalizedts)
- [dbo.ACM_SensorRanking](#dboacmsensorranking)
- [dbo.ACM_SinceWhen](#dboacmsincewhen)
- [dbo.ACM_TagEquipmentMap](#dboacmtagequipmentmap)
- [dbo.ACM_ThresholdCrossings](#dboacmthresholdcrossings)
- [dbo.ACM_ThresholdMetadata](#dboacmthresholdmetadata)
- [dbo.Equipment](#dboequipment)
- [dbo.FD_FAN_Data](#dbofdfandata)
- [dbo.GAS_TURBINE_Data](#dbogasturbinedata)
- [dbo.ModelRegistry](#dbomodelregistry)


## Summary

| Table | Columns | Rows | Primary Key |
| --- | ---: | ---: | --- |
| dbo.ACM_AdaptiveConfig | 13 | 9 | ConfigID |
| dbo.ACM_AdaptiveThresholds_ByRegime | 11 | 0 | ID |
| dbo.ACM_AlertAge | 6 | 15 | — |
| dbo.ACM_Anomaly_Events | 6 | 98 | Id |
| dbo.ACM_BaselineBuffer | 7 | 200,000 | Id |
| dbo.ACM_CalibrationSummary | 10 | 120 | — |
| dbo.ACM_ColdstartState | 17 | 2 | EquipID, Stage |
| dbo.ACM_Config | 7 | 245 | ConfigID |
| dbo.ACM_ConfigHistory | 9 | 10 | ID |
| dbo.ACM_ContributionCurrent | 5 | 120 | — |
| dbo.ACM_ContributionTimeline | 5 | 59,904 | — |
| dbo.ACM_DailyFusedProfile | 9 | 277 | ID |
| dbo.ACM_DataQuality | 24 | 163 | — |
| dbo.ACM_DefectSummary | 12 | 15 | — |
| dbo.ACM_DefectTimeline | 10 | 2,934 | — |
| dbo.ACM_DetectorCorrelation | 7 | 420 | — |
| dbo.ACM_DetectorForecast_TS | 10 | 3,864 | RunID, EquipID, DetectorName, Timestamp |
| dbo.ACM_DriftEvents | 2 | 2 | — |
| dbo.ACM_DriftSeries | 4 | 10,893 | — |
| dbo.ACM_EnhancedFailureProbability_TS | 11 | 0 | RunID, EquipID, Timestamp, ForecastHorizon_Hours |
| dbo.ACM_EnhancedMaintenanceRecommendation | 13 | 0 | RunID, EquipID |
| dbo.ACM_EpisodeCulprits | 9 | 581 | ID |
| dbo.ACM_EpisodeDiagnostics | 13 | 98 | ID |
| dbo.ACM_EpisodeMetrics | 10 | 14 | — |
| dbo.ACM_Episodes | 8 | 14 | — |
| dbo.ACM_EpisodesQC | 10 | 14 | RecordID |
| dbo.ACM_FailureCausation | 12 | 0 | RunID, EquipID, Detector |
| dbo.ACM_FailureForecast | 9 | 672 | EquipID, RunID, Timestamp |
| dbo.ACM_FailureForecast_TS | 7 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_FailureHazard_TS | 8 | 2,352 | EquipID, RunID, Timestamp |
| dbo.ACM_FeatureDropLog | 8 | 14,043 | LogID |
| dbo.ACM_ForecastContext | 26 | 0 | ID |
| dbo.ACM_ForecastState | 12 | 913 | EquipID, StateVersion |
| dbo.ACM_ForecastingState | 13 | 0 | EquipID, StateVersion |
| dbo.ACM_FusionQualityReport | 9 | 150 | — |
| dbo.ACM_HealthDistributionOverTime | 12 | 6,356 | — |
| dbo.ACM_HealthForecast | 10 | 672 | EquipID, RunID, Timestamp |
| dbo.ACM_HealthForecast_Continuous | 8 | 3,988 | EquipID, Timestamp, SourceRunID |
| dbo.ACM_HealthForecast_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_HealthHistogram | 5 | 150 | — |
| dbo.ACM_HealthTimeline | 8 | 10,893 | — |
| dbo.ACM_HealthZoneByPeriod | 9 | 831 | — |
| dbo.ACM_HistorianData | 7 | 204,067 | DataID |
| dbo.ACM_MaintenanceRecommendation | 8 | 0 | RunID, EquipID |
| dbo.ACM_OMRContributionsLong | 8 | 1,105,723 | — |
| dbo.ACM_OMRTimeline | 6 | 10,893 | — |
| dbo.ACM_OMR_Diagnostics | 15 | 15 | DiagnosticID |
| dbo.ACM_PCA_Loadings | 10 | 8,140 | RecordID |
| dbo.ACM_PCA_Metrics | 6 | 45 | RunID, EquipID, ComponentName, MetricType |
| dbo.ACM_PCA_Models | 12 | 15 | RecordID |
| dbo.ACM_RUL | 18 | 4 | EquipID, RunID |
| dbo.ACM_RUL_Attribution | 9 | 0 | RunID, EquipID, FailureTime, SensorName |
| dbo.ACM_RUL_ByRegime | 15 | 0 | ID |
| dbo.ACM_RUL_LearningState | 19 | 2 | EquipID |
| dbo.ACM_RUL_Summary | 15 | 0 | RunID, EquipID |
| dbo.ACM_RUL_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_RecommendedActions | 6 | 0 | RunID, EquipID, Action |
| dbo.ACM_RefitRequests | 10 | 1,371 | RequestID |
| dbo.ACM_RegimeDwellStats | 8 | 66 | — |
| dbo.ACM_RegimeHazard | 11 | 0 | ID |
| dbo.ACM_RegimeOccupancy | 5 | 66 | — |
| dbo.ACM_RegimeStability | 4 | 15 | — |
| dbo.ACM_RegimeState | 15 | 2 | EquipID, StateVersion |
| dbo.ACM_RegimeStats | 8 | 66 | — |
| dbo.ACM_RegimeTimeline | 5 | 10,893 | — |
| dbo.ACM_RegimeTransitions | 6 | 95 | — |
| dbo.ACM_Regime_Episodes | 6 | 98 | Id |
| dbo.ACM_RunLogs | 25 | 338,940 | LogID |
| dbo.ACM_RunMetadata | 12 | 715 | RunMetadataID |
| dbo.ACM_RunMetrics | 5 | 25,305 | RunID, EquipID, MetricName |
| dbo.ACM_Run_Stats | 13 | 15 | RecordID |
| dbo.ACM_Runs | 19 | 17 | RunID |
| dbo.ACM_SchemaVersion | 5 | 2 | VersionID |
| dbo.ACM_Scores_Long | 9 | 65,358 | Id |
| dbo.ACM_Scores_Wide | 15 | 10,893 | — |
| dbo.ACM_SensorAnomalyByPeriod | 11 | 2,216 | — |
| dbo.ACM_SensorDefects | 11 | 120 | — |
| dbo.ACM_SensorForecast | 11 | 6,720 | RunID, EquipID, Timestamp, SensorName |
| dbo.ACM_SensorForecast_TS | 10 | 0 | RunID, EquipID, SensorName, Timestamp |
| dbo.ACM_SensorHotspotTimeline | 9 | 12,807 | — |
| dbo.ACM_SensorHotspots | 18 | 163 | — |
| dbo.ACM_SensorNormalized_TS | 10 | 110,679 | Id |
| dbo.ACM_SensorRanking | 6 | 120 | — |
| dbo.ACM_SinceWhen | 6 | 15 | — |
| dbo.ACM_TagEquipmentMap | 10 | 25 | TagID |
| dbo.ACM_ThresholdCrossings | 7 | 306 | — |
| dbo.ACM_ThresholdMetadata | 13 | 30 | ThresholdID |
| dbo.Equipment | 8 | 3 | EquipID |
| dbo.FD_FAN_Data | 11 | 17,499 | EntryDateTime |
| dbo.GAS_TURBINE_Data | 18 | 2,911 | EntryDateTime |
| dbo.ModelRegistry | 8 | 105 | ModelType, EquipID, Version |

---



## Detailed Table Information



## dbo.ACM_AdaptiveConfig

**Primary Key:** ConfigID  
**Row Count:** 9  
**Date Range:** 2025-12-04 10:46:47 to 2025-12-04 10:46:47  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ConfigID | int | NO | 10 | — |
| EquipID | int | YES | 10 | — |
| ConfigKey | nvarchar | NO | 100 | — |
| ConfigValue | float | NO | 53 | — |
| MinBound | float | NO | 53 | — |
| MaxBound | float | NO | 53 | — |
| IsLearned | bit | NO | — | ((0)) |
| DataVolumeAtTuning | bigint | YES | 19 | — |
| PerformanceMetric | float | YES | 53 | — |
| ResearchReference | nvarchar | YES | 500 | — |
| Source | nvarchar | NO | 50 | — |
| CreatedAt | datetime2 | NO | — | (getdate()) |
| UpdatedAt | datetime2 | NO | — | (getdate()) |

### Top 10 Records

| ConfigID | EquipID | ConfigKey | ConfigValue | MinBound | MaxBound | IsLearned | DataVolumeAtTuning | PerformanceMetric | ResearchReference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NULL | alpha | 0.3 | 0.05 | 0.95 | False | NULL | NULL | Hyndman & Athanasopoulos (2018) - Exponential smoothing level |
| 2 | NULL | beta | 0.1 | 0.01 | 0.3 | False | NULL | NULL | Hyndman & Athanasopoulos (2018) - Exponential smoothing trend |
| 3 | NULL | training_window_hours | 168.0 | 72.0 | 720.0 | False | NULL | NULL | NIST SP 1225 - 3-30 day training window |
| 4 | NULL | failure_threshold | 70.0 | 40.0 | 80.0 | False | NULL | NULL | ISO 13381-1:2015 - Health index threshold |
| 5 | NULL | confidence_min | 0.8 | 0.5 | 0.95 | False | NULL | NULL | Agresti & Coull (1998) - Statistical confidence |
| 6 | NULL | max_forecast_hours | 168.0 | 168.0 | 720.0 | False | NULL | NULL | Industry standard - 7-30 day horizon |
| 7 | NULL | monte_carlo_simulations | 1000.0 | 500.0 | 5000.0 | False | NULL | NULL | Saxena et al. (2008) - RUL simulation count |
| 8 | NULL | blend_tau_hours | 12.0 | 6.0 | 48.0 | False | NULL | NULL | Expert tuning - Warm-start alpha blending |
| 9 | NULL | auto_tune_data_threshold | 10000.0 | 5000.0 | 50000.0 | False | NULL | NULL | Expert tuning - Auto-tuning trigger |

---


## dbo.ACM_AdaptiveThresholds_ByRegime

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| RunID | nvarchar | NO | 64 | — |
| RegimeLabel | int | NO | 10 | — |
| RegimeState | nvarchar | YES | 32 | — |
| DetectorType | nvarchar | NO | 64 | — |
| WarnThreshold | float | NO | 53 | — |
| AlertThreshold | float | NO | 53 | — |
| Method | nvarchar | YES | 32 | ('quantile') |
| SampleCount | int | YES | 10 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

---


## dbo.ACM_AlertAge

**Primary Key:** No primary key  
**Row Count:** 15  
**Date Range:** 2025-12-05 11:37:01 to 2025-12-11 09:25:07  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| AlertZone | nvarchar | NO | 50 | — |
| StartTimestamp | datetime2 | NO | — | — |
| DurationHours | float | NO | 53 | — |
| RecordCount | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| AlertZone | StartTimestamp | DurationHours | RecordCount | RunID | EquipID |
| --- | --- | --- | --- | --- | --- |
| GOOD | 2025-12-05 11:37:01 | 0.0 | 0 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |
| GOOD | 2025-12-05 11:37:36 | 0.0 | 0 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 |
| GOOD | 2025-12-05 11:38:01 | 0.0 | 0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| GOOD | 2025-12-05 11:38:26 | 0.0 | 0 | E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 |
| GOOD | 2025-12-11 09:21:17 | 0.0 | 0 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| GOOD | 2025-12-11 09:21:33 | 0.0 | 0 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| GOOD | 2025-12-11 09:21:49 | 0.0 | 0 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 |
| GOOD | 2025-12-11 09:22:19 | 0.0 | 0 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 |
| GOOD | 2025-12-11 09:22:50 | 0.0 | 0 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 |
| GOOD | 2025-12-11 09:23:22 | 0.0 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |

### Bottom 10 Records

| AlertZone | StartTimestamp | DurationHours | RecordCount | RunID | EquipID |
| --- | --- | --- | --- | --- | --- |
| GOOD | 2025-12-11 09:25:07 | 0.0 | 0 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| GOOD | 2025-12-11 09:24:52 | 0.0 | 0 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 |
| GOOD | 2025-12-11 09:24:37 | 0.0 | 0 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 |
| GOOD | 2025-12-11 09:24:21 | 0.0 | 0 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| GOOD | 2025-12-11 09:23:53 | 0.0 | 0 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| GOOD | 2025-12-11 09:23:22 | 0.0 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| GOOD | 2025-12-11 09:22:50 | 0.0 | 0 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 |
| GOOD | 2025-12-11 09:22:19 | 0.0 | 0 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 |
| GOOD | 2025-12-11 09:21:49 | 0.0 | 0 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 |
| GOOD | 2025-12-11 09:21:33 | 0.0 | 0 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |

---


## dbo.ACM_Anomaly_Events

**Primary Key:** Id  
**Row Count:** 98  
**Date Range:** 2023-10-18 22:00:00 to 2025-09-11 15:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | YES | 10 | — |
| StartTime | datetime2 | YES | — | — |
| EndTime | datetime2 | YES | — | — |
| Severity | nvarchar | YES | 32 | — |

### Top 10 Records

| Id | RunID | EquipID | StartTime | EndTime | Severity |
| --- | --- | --- | --- | --- | --- |
| 3344 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3345 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3346 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3347 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3348 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3349 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3350 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3351 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3352 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |
| 3353 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | info |

### Bottom 10 Records

| Id | RunID | EquipID | StartTime | EndTime | Severity |
| --- | --- | --- | --- | --- | --- |
| 4602 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-11 15:30:00 | 2025-09-11 23:00:00 | info |
| 4601 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-11 01:00:00 | 2025-09-11 03:30:00 | info |
| 4600 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2025-06-12 23:00:00 | 2025-06-13 00:00:00 | info |
| 4599 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2025-06-12 00:00:00 | 2025-06-12 06:30:00 | info |
| 4598 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2025-05-13 01:00:00 | 2025-05-13 07:00:00 | info |
| 4597 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2025-05-12 00:00:00 | 2025-05-12 07:00:00 | info |
| 4596 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 2025-04-14 00:00:00 | 2025-04-14 02:30:00 | info |
| 4595 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 2025-04-11 00:00:00 | 2025-04-12 06:30:00 | info |
| 4594 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 2025-03-15 00:00:00 | 2025-03-15 02:30:00 | info |
| 4593 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 2025-03-12 00:00:00 | 2025-03-13 07:00:00 | info |

---


## dbo.ACM_BaselineBuffer

**Primary Key:** Id  
**Row Count:** 200,000  
**Date Range:** 2024-05-19 20:00:00 to 2025-09-14 23:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime | NO | — | — |
| SensorName | nvarchar | NO | 128 | — |
| SensorValue | float | NO | 53 | — |
| DataQuality | nvarchar | YES | 64 | — |
| CreatedAt | datetime | NO | — | (getdate()) |

### Top 10 Records

| Id | EquipID | Timestamp | SensorName | SensorValue | DataQuality | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 8528556 | 2621 | 2024-05-19 20:00:00 | ACTTBTEMP1 | 164.20663452148438 | NULL | 2025-12-04 10:53:56 |
| 8528557 | 2621 | 2024-05-19 20:00:00 | B1RADVIBX | 2.4915695190429688 | NULL | 2025-12-04 10:53:56 |
| 8528558 | 2621 | 2024-05-19 20:00:00 | B1RADVIBY | 2.5513648986816406 | NULL | 2025-12-04 10:53:56 |
| 8528559 | 2621 | 2024-05-19 20:00:00 | B1TEMP1 | 194.63162231445312 | NULL | 2025-12-04 10:53:56 |
| 8528560 | 2621 | 2024-05-19 20:00:00 | B1VIB1 | 0.15218257904052734 | NULL | 2025-12-04 10:53:56 |
| 8528561 | 2621 | 2024-05-19 20:00:00 | B1VIB2 | 0.15003085136413574 | NULL | 2025-12-04 10:53:56 |
| 8528562 | 2621 | 2024-05-19 20:00:00 | B2RADVIBX | 1.2345314025878906 | NULL | 2025-12-04 10:53:56 |
| 8528563 | 2621 | 2024-05-19 20:00:00 | B2RADVIBY | 1.3340950012207031 | NULL | 2025-12-04 10:53:56 |
| 8528564 | 2621 | 2024-05-19 20:00:00 | B2TEMP1 | 190.1356201171875 | NULL | 2025-12-04 10:53:56 |
| 8528565 | 2621 | 2024-05-19 20:00:00 | B2VIB1 | 0.06996989250183105 | NULL | 2025-12-04 10:53:56 |

### Bottom 10 Records

| Id | EquipID | Timestamp | SensorName | SensorValue | DataQuality | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 13037076 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 397.95001220703125 | NULL | 2025-12-11 09:25:05 |
| 13037075 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | 383.5799865722656 | NULL | 2025-12-11 09:25:05 |
| 13037074 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.06T34_1FD Fan Outlet Termperature | 33.2400016784668 | NULL | 2025-12-11 09:25:05 |
| 13037073 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 55.02000045776367 | NULL | 2025-12-11 09:25:05 |
| 13037072 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | 63.439998626708984 | NULL | 2025-12-11 09:25:05 |
| 13037071 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 48.11000061035156 | NULL | 2025-12-11 09:25:05 |
| 13037070 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.06I03_1FD Fan Motor Current | 45.2400016784668 | NULL | 2025-12-11 09:25:05 |
| 13037069 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1.3300000429153442 | NULL | 2025-12-11 09:25:05 |
| 13037068 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 48.119998931884766 | NULL | 2025-12-11 09:25:05 |
| 13037067 | 1 | 2025-09-14 22:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 393.2699890136719 | NULL | 2025-12-11 09:25:05 |

---


## dbo.ACM_CalibrationSummary

**Primary Key:** No primary key  
**Row Count:** 120  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorType | nvarchar | NO | 50 | — |
| MeanZ | float | NO | 53 | — |
| StdZ | float | NO | 53 | — |
| P95Z | float | NO | 53 | — |
| P99Z | float | NO | 53 | — |
| ClipZ | float | NO | 53 | — |
| SaturationPct | float | NO | 53 | — |
| MahalCondNum | float | YES | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| DetectorType | MeanZ | StdZ | P95Z | P99Z | ClipZ | SaturationPct | MahalCondNum | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time-Series Anomaly (AR1) | 1.0378999710083008 | 1.2263000011444092 | 3.014 | 7.3888 | 100.0 | 0.0 | NULL | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Correlation Break (PCA-SPE) | 2.217400074005127 | 2.8215999603271484 | 10.0 | 10.0 | 100.0 | 0.0 | NULL | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Multivariate Outlier (PCA-T2) | 1.375100016593933 | 1.861199975013733 | 5.2589 | 10.0 | 100.0 | 0.0 | NULL | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Multivariate Distance (Mahalanobis) | 2.552000045776367 | 3.308799982070923 | 10.0 | 10.0 | 100.0 | 0.0 | 19602958498.52 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Rare State (IsolationForest) | 1.2407000064849854 | 1.2539000511169434 | 4.0326 | 5.4005 | 100.0 | 0.0 | NULL | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Density Anomaly (GMM) | 2.0494000911712646 | 2.5023000240325928 | 8.6401 | 10.0 | 100.0 | 0.0 | NULL | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Baseline Consistency (OMR) | 1.5563000440597534 | 1.6759999990463257 | 5.7577 | 6.1271 | 100.0 | 0.0 | NULL | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| cusum_z | 0.7296000123023987 | 0.4966000020503998 | 1.5479 | 1.7272 | 100.0 | 0.0 | NULL | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Time-Series Anomaly (AR1) | 1.2907999753952026 | 2.0439000129699707 | 6.0105 | 10.0 | 100.0 | 0.0 | NULL | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| Correlation Break (PCA-SPE) | 1.5791000127792358 | 2.7023000717163086 | 10.0 | 10.0 | 100.0 | 0.0 | NULL | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |

### Bottom 10 Records

| DetectorType | MeanZ | StdZ | P95Z | P99Z | ClipZ | SaturationPct | MahalCondNum | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time-Series Anomaly (AR1) | 2.353100061416626 | 2.6875 | 8.5294 | 10.0 | 43.2 | 0.0 | NULL | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Correlation Break (PCA-SPE) | 1.3502000570297241 | 1.885200023651123 | 5.3942 | 10.0 | 43.2 | 0.0 | NULL | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Multivariate Outlier (PCA-T²) | 4.861000061035156 | 4.212399959564209 | 10.0 | 10.0 | 43.2 | 0.0 | NULL | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Multivariate Distance (Mahalanobis) | 3.2820000648498535 | 3.921999931335449 | 10.0 | 10.0 | 43.2 | 0.0 | 35187087754.55 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Rare State (IsolationForest) | 0.8465999960899353 | 0.4641000032424927 | 1.6023 | 1.9762 | 43.2 | 0.0 | NULL | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Density Anomaly (GMM) | 1.2755000591278076 | 1.0312999486923218 | 2.4431 | 4.8183 | 43.2 | 0.0 | NULL | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Baseline Consistency (OMR) | 0.9666000008583069 | 1.149999976158142 | 2.844 | 7.9734 | 43.2 | 0.0 | NULL | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| cusum_z | 0.7502999901771545 | 0.38519999384880066 | 1.5023 | 1.8297 | 43.2 | 0.0 | NULL | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Time-Series Anomaly (AR1) | 1.4823999404907227 | 2.2643001079559326 | 8.7046 | 10.0 | 100.0 | 0.0 | NULL | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| Correlation Break (PCA-SPE) | 3.1577000617980957 | 3.358099937438965 | 10.0 | 10.0 | 100.0 | 0.0 | NULL | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |

---


## dbo.ACM_ColdstartState

**Primary Key:** EquipID, Stage  
**Row Count:** 2  
**Date Range:** 2025-12-05 06:06:49 to 2025-12-11 03:51:09  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| Stage | varchar | NO | 20 | ('score') |
| Status | varchar | NO | 20 | — |
| AttemptCount | int | NO | 10 | ((0)) |
| FirstAttemptAt | datetime2 | NO | — | (getutcdate()) |
| LastAttemptAt | datetime2 | NO | — | (getutcdate()) |
| CompletedAt | datetime2 | YES | — | — |
| AccumulatedRows | int | NO | 10 | ((0)) |
| RequiredRows | int | NO | 10 | ((500)) |
| DataStartTime | datetime2 | YES | — | — |
| DataEndTime | datetime2 | YES | — | — |
| TickMinutes | int | NO | 10 | — |
| ColdstartSplitRatio | float | NO | 53 | ((0.6)) |
| LastError | nvarchar | YES | 2000 | — |
| ErrorCount | int | NO | 10 | ((0)) |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |
| UpdatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| EquipID | Stage | Status | AttemptCount | FirstAttemptAt | LastAttemptAt | CompletedAt | AccumulatedRows | RequiredRows | DataStartTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | score | COMPLETE | 1 | 2025-12-11 03:51:09 | 2025-12-11 03:51:09 | 2025-12-11 03:51:09 | 241 | 200 | 2023-10-15 00:00:00 |
| 2621 | score | COMPLETE | 1 | 2025-12-05 06:06:49 | 2025-12-05 06:06:49 | 2025-12-05 06:06:49 | 241 | 200 | 2023-10-15 00:00:00 |

---


## dbo.ACM_Config

**Primary Key:** ConfigID  
**Row Count:** 245  
**Date Range:** 2025-12-09 12:47:06 to 2025-12-11 03:50:40  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ConfigID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| ParamPath | nvarchar | NO | 500 | — |
| ParamValue | nvarchar | NO | -1 | — |
| ValueType | varchar | NO | 50 | — |
| UpdatedAt | datetime2 | NO | — | (getutcdate()) |
| UpdatedBy | nvarchar | YES | 100 | (suser_sname()) |

### Top 10 Records

| ConfigID | EquipID | ParamPath | ParamValue | ValueType | UpdatedAt | UpdatedBy |
| --- | --- | --- | --- | --- | --- | --- |
| 492 | 0 | data.train_csv | data/FD_FAN_BASELINE_DATA.csv | string | 2025-12-11 03:43:34 | B19cl3pc\bhadk |
| 493 | 0 | data.score_csv | data/FD_FAN_BATCH_DATA.csv | string | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 494 | 0 | data.data_dir | data | string | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 495 | 0 | data.timestamp_col | EntryDateTime | string | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 496 | 0 | data.tag_columns | [] | list | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 497 | 0 | data.sampling_secs | 1800 | int | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 498 | 0 | data.max_rows | 100000 | int | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 499 | 0 | features.window | 16 | int | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 500 | 0 | features.fft_bands | [0.0, 0.1, 0.3, 0.5] | list | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 501 | 0 | features.top_k_tags | 5 | int | 2025-12-11 03:43:35 | B19cl3pc\bhadk |

### Bottom 10 Records

| ConfigID | EquipID | ParamPath | ParamValue | ValueType | UpdatedAt | UpdatedBy |
| --- | --- | --- | --- | --- | --- | --- |
| 748 | 0 | forecasting.confidence_k | 1.96 | float | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 747 | 0 | models.ar1.z_cap | 8.0 | float | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 746 | 0 | models.ar1.alpha | 0.05 | float | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 745 | 0 | models.ar1.window | 256 | int | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 742 | 0 | health.extreme_z_threshold | 10.0 | float | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 741 | 0 | health.max_change_per_period | 20.0 | float | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 740 | 0 | health.smoothing_alpha | 0.3 | float | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 734 | 1 | runtime.tick_minutes | 1440 | int | 2025-12-11 03:50:40 | sql_batch_runner |
| 733 | 2621 | data.min_train_samples | 200 | int | 2025-12-11 03:43:35 | B19cl3pc\bhadk |
| 732 | 1 | data.min_train_samples | 200 | int | 2025-12-11 03:43:35 | B19cl3pc\bhadk |

---


## dbo.ACM_ConfigHistory

**Primary Key:** ID  
**Row Count:** 10  
**Date Range:** 2025-12-03 14:40:59 to 2025-12-08 15:37:35  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| Timestamp | datetime2 | NO | — | (sysutcdatetime()) |
| EquipID | int | NO | 10 | — |
| ParameterPath | nvarchar | NO | 256 | — |
| OldValue | nvarchar | YES | -1 | — |
| NewValue | nvarchar | YES | -1 | — |
| ChangedBy | nvarchar | YES | 64 | — |
| ChangeReason | nvarchar | YES | 256 | — |
| RunID | nvarchar | YES | 64 | — |

### Top 10 Records

| ID | Timestamp | EquipID | ParameterPath | OldValue | NewValue | ChangedBy | ChangeReason | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2025-12-03 14:40:59 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.96e+35 exceeds 1e28 (critical instability) | 9FE7E697-6F6E-48A7-BFCB-B08797B61D1C |
| 2 | 2025-12-03 14:45:24 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.96e+35 exceeds 1e28 (critical instability) | 03DAA545-ECB9-406A-AE1D-928C823A63C8 |
| 3 | 2025-12-03 14:55:40 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.96e+35 exceeds 1e28 (critical instability) | E940356F-81F3-4EF8-BF88-C553AA86AE3D |
| 4 | 2025-12-03 15:20:30 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.01e+34 exceeds 1e28 (critical instability) | AFB8AE5E-847B-4407-BBFF-393251387EA3 |
| 5 | 2025-12-03 16:16:53 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.01e+34 exceeds 1e28 (critical instability) | F40EF3EA-BEFF-4AC8-BBE2-9D553F98982F |
| 6 | 2025-12-03 16:20:51 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.01e+34 exceeds 1e28 (critical instability) | DEF498BA-F846-4093-BD45-9D770CE5380B |
| 7 | 2025-12-03 16:43:12 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.01e+34 exceeds 1e28 (critical instability) | 1DB9295F-6A17-42CF-8B9D-8D65C01662A0 |
| 8 | 2025-12-04 11:11:02 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 4.28e+33 exceeds 1e28 (critical instability) | DB338AC8-ADA2-41E7-A72D-D70BC84DAC0E |
| 9 | 2025-12-04 15:09:29 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 4.28e+33 exceeds 1e28 (critical instability) | c7de5f9b-a612-4a10-a853-58545790dadb |
| 10 | 2025-12-08 15:37:35 | 1 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.28e+36 exceeds 1e28 (critical instability) | e56a239d-c9df-4348-b396-d82b47af9ecc |

---


## dbo.ACM_ContributionCurrent

**Primary Key:** No primary key  
**Row Count:** 120  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorType | nvarchar | NO | 50 | — |
| ContributionPct | float | NO | 53 | — |
| ZScore | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| DetectorType | ContributionPct | ZScore | RunID | EquipID |
| --- | --- | --- | --- | --- |
| Time-Series Anomaly (AR1) | 29.8700008392334 | 0.7490874528884888 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Multivariate Outlier (PCA-T2) | 17.170000076293945 | 0.43060800433158875 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Multivariate Distance (Mahalanobis) | 12.819999694824219 | 0.3216104507446289 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Rare State (IsolationForest) | 12.0 | 0.30092769861221313 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Density Anomaly (GMM) | 7.570000171661377 | 0.18974179029464722 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Correlation Break (PCA-SPE) | 7.449999809265137 | 0.18681998550891876 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Baseline Consistency (OMR) | 7.03000020980835 | 0.17620302736759186 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| cusum_z | 6.099999904632568 | 0.15303833782672882 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| Multivariate Distance (Mahalanobis) | 30.799999237060547 | 5.20524787902832 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| Density Anomaly (GMM) | 28.6299991607666 | 4.837921619415283 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |

### Bottom 10 Records

| DetectorType | ContributionPct | ZScore | RunID | EquipID |
| --- | --- | --- | --- | --- |
| Multivariate Distance (Mahalanobis) | 35.540000915527344 | 10.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Multivariate Outlier (PCA-T²) | 35.540000915527344 | 10.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Time-Series Anomaly (AR1) | 13.039999961853027 | 3.6689839363098145 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Density Anomaly (GMM) | 6.389999866485596 | 1.7988011837005615 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Rare State (IsolationForest) | 5.150000095367432 | 1.4499531984329224 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| cusum_z | 3.5799999237060547 | 1.008407711982727 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Correlation Break (PCA-SPE) | 0.4000000059604645 | 0.11277640610933304 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| Baseline Consistency (OMR) | 0.3499999940395355 | 0.09875394403934479 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| cusum_z | 33.47999954223633 | 1.6350982189178467 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| Density Anomaly (GMM) | 32.380001068115234 | 1.5814913511276245 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |

---


## dbo.ACM_ContributionTimeline

**Primary Key:** No primary key  
**Row Count:** 59,904  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| DetectorType | nvarchar | NO | 50 | — |
| ContributionPct | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Timestamp | DetectorType | ContributionPct | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 2023-10-18 00:00:00 | cusum_z | 14.90999984741211 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | Density Anomaly (GMM) | 16.040000915527344 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | Multivariate Outlier (PCA-T2) | 0.0 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | Rare State (IsolationForest) | 11.239999771118164 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | Multivariate Distance (Mahalanobis) | 17.1200008392334 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | Correlation Break (PCA-SPE) | 7.190000057220459 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | Baseline Consistency (OMR) | 6.760000228881836 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | Time-Series Anomaly (AR1) | 26.75 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:30:00 | Rare State (IsolationForest) | 7.289999961853027 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:30:00 | Multivariate Outlier (PCA-T2) | 10.079999923706055 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |

### Bottom 10 Records

| Timestamp | DetectorType | ContributionPct | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | Correlation Break (PCA-SPE) | 2.440000057220459 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | Time-Series Anomaly (AR1) | 7.690000057220459 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | Multivariate Outlier (PCA-T2) | 8.289999961853027 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | Multivariate Distance (Mahalanobis) | 32.61000061035156 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | Rare State (IsolationForest) | 25.219999313354492 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | Density Anomaly (GMM) | 18.479999542236328 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | Baseline Consistency (OMR) | 1.2200000286102295 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | cusum_z | 4.059999942779541 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:30:00 | Time-Series Anomaly (AR1) | 8.479999542236328 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:30:00 | Correlation Break (PCA-SPE) | 0.6499999761581421 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |

---


## dbo.ACM_DailyFusedProfile

**Primary Key:** ID  
**Row Count:** 277  
**Date Range:** 2025-12-05 00:00:00 to 2025-12-11 00:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| ProfileDate | date | NO | — | — |
| AvgFusedScore | float | YES | 53 | — |
| MaxFusedScore | float | YES | 53 | — |
| MinFusedScore | float | YES | 53 | — |
| SampleCount | int | YES | 10 | — |
| CreatedAt | datetime2 | YES | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | ProfileDate | AvgFusedScore | MaxFusedScore | MinFusedScore | SampleCount | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 166648 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:00 |
| 166649 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:00 |
| 166650 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:00 |
| 166651 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:00 |
| 166652 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:00 |
| 166664 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:36 |
| 166665 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:36 |
| 166666 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:36 |
| 166667 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:36 |
| 166668 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2025-12-05 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-05 11:37:36 |

### Bottom 10 Records

| ID | RunID | EquipID | ProfileDate | AvgFusedScore | MaxFusedScore | MinFusedScore | SampleCount | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 170264 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:25:07 |
| 170263 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:25:07 |
| 170262 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:25:07 |
| 170261 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:25:07 |
| 170260 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:25:07 |
| 170259 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:25:07 |
| 170258 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:25:07 |
| 170257 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:25:07 |
| 170256 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:24:52 |
| 170255 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2025-12-11 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-11 09:24:52 |

---


## dbo.ACM_DataQuality

**Primary Key:** No primary key  
**Row Count:** 163  
**Date Range:** 2023-10-15 00:00:00 to 2025-07-11 00:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| sensor | nvarchar | NO | 255 | — |
| train_count | int | YES | 10 | — |
| train_nulls | int | YES | 10 | — |
| train_null_pct | float | YES | 53 | — |
| train_std | float | YES | 53 | — |
| train_longest_gap | int | YES | 10 | — |
| train_flatline_span | int | YES | 10 | — |
| train_min_ts | datetime2 | YES | — | — |
| train_max_ts | datetime2 | YES | — | — |
| score_count | int | YES | 10 | — |
| score_nulls | int | YES | 10 | — |
| score_null_pct | float | YES | 53 | — |
| score_std | float | YES | 53 | — |
| score_longest_gap | int | YES | 10 | — |
| score_flatline_span | int | YES | 10 | — |
| score_min_ts | datetime2 | YES | — | — |
| score_max_ts | datetime2 | YES | — | — |
| interp_method | nvarchar | YES | 50 | — |
| sampling_secs | float | YES | 53 | — |
| notes | nvarchar | YES | -1 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| CheckName | nvarchar | NO | 100 | — |
| CheckResult | nvarchar | NO | 50 | — |

### Top 10 Records

| sensor | train_count | train_nulls | train_null_pct | train_std | train_longest_gap | train_flatline_span | train_min_ts | train_max_ts | score_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEMO.SIM.06G31_1FD Fan Damper Position | 1442 | 0 | 0.0 | 6.5572638511657715 | 0 | 0 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1442 | 0 | 0.0 | 0.25368690490722656 | 0 | 5 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.06I03_1FD Fan Motor Current | 1442 | 0 | 0.0 | 2.095574378967285 | 0 | 1 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1442 | 0 | 0.0 | 6.686100006103516 | 0 | 1 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | 1442 | 0 | 0.0 | 1.8766218423843384 | 0 | 1 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1442 | 0 | 0.0 | 4.897747039794922 | 0 | 1 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.06T34_1FD Fan Outlet Termperature | 1442 | 0 | 0.0 | 5.125950336456299 | 0 | 1 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | 1442 | 0 | 0.0 | 46.07988357543945 | 0 | 0 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 1442 | 0 | 0.0 | 49.41089630126953 | 0 | 2 | 2024-07-21 09:30:00 | 2024-08-25 10:00:00 | 1443 |
| DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1257 | 0 | 0.0 | 11.019021034240723 | 0 | 2 | 2024-12-08 14:30:00 | 2025-01-10 18:30:00 | 1258 |

### Bottom 10 Records

| sensor | train_count | train_nulls | train_null_pct | train_std | train_longest_gap | train_flatline_span | train_min_ts | train_max_ts | score_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACTTBTEMP1 | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B1RADVIBX | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B1RADVIBY | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B1TEMP1 | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B1VIB1 | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B1VIB2 | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B2RADVIBX | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B2RADVIBY | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B2TEMP1 | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |
| B2VIB1 | 300 | 0 | 0.0 | NULL | NULL | NULL | NULL | NULL | 505 |

---


## dbo.ACM_DefectSummary

**Primary Key:** No primary key  
**Row Count:** 15  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Status | nvarchar | NO | 50 | — |
| Severity | nvarchar | NO | 50 | — |
| CurrentHealth | float | NO | 53 | — |
| AvgHealth | float | NO | 53 | — |
| MinHealth | float | NO | 53 | — |
| EpisodeCount | int | NO | 10 | — |
| WorstSensor | nvarchar | YES | 255 | — |
| GoodCount | int | NO | 10 | — |
| WatchCount | int | NO | 10 | — |
| AlertCount | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Status | Severity | CurrentHealth | AvgHealth | MinHealth | EpisodeCount | WorstSensor | GoodCount | WatchCount | AlertCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAUTION | MEDIUM | 71.5999984741211 | 70.69999694824219 | 7.599999904632568 | 14 | ar1 | 445 | 411 | 587 |
| HEALTHY | LOW | 95.5999984741211 | 74.69999694824219 | 9.0 | 11 | mhal | 486 | 429 | 343 |
| HEALTHY | LOW | 99.30000305175781 | 75.69999694824219 | 5.300000190734863 | 5 | mhal | 232 | 106 | 165 |
| ALERT | HIGH | 49.900001525878906 | 73.19999694824219 | 6.900000095367432 | 14 | cusum | 560 | 247 | 547 |
| ALERT | HIGH | 46.900001525878906 | 65.30000305175781 | 19.899999618530273 | 11 | ar1 | 268 | 92 | 482 |
| CAUTION | MEDIUM | 76.4000015258789 | 73.80000305175781 | 11.899999618530273 | 4 | omr | 175 | 160 | 141 |
| HEALTHY | LOW | 95.0 | 77.80000305175781 | 8.5 | 2 | mhal | 213 | 74 | 112 |
| HEALTHY | LOW | 95.30000305175781 | 73.80000305175781 | 11.100000381469727 | 2 | mhal | 155 | 59 | 145 |
| ALERT | HIGH | 31.399999618530273 | 79.4000015258789 | 6.5 | 0 | pca_spe | 51 | 23 | 23 |
| ALERT | HIGH | 53.400001525878906 | 68.80000305175781 | 8.600000381469727 | 4 | mhal | 88 | 86 | 188 |

### Bottom 10 Records

| Status | Severity | CurrentHealth | AvgHealth | MinHealth | EpisodeCount | WorstSensor | GoodCount | WatchCount | AlertCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALERT | HIGH | 69.69999694824219 | 68.9000015258789 | 13.300000190734863 | 7 | pca_t2 | 155 | 83 | 267 |
| ALERT | HIGH | 44.900001525878906 | 75.19999694824219 | 11.100000381469727 | 10 | cusum | 585 | 326 | 435 |
| ALERT | HIGH | 49.900001525878906 | 64.30000305175781 | 31.799999237060547 | 4 | mhal | 88 | 114 | 239 |
| HEALTHY | LOW | 98.69999694824219 | 75.19999694824219 | 5.900000095367432 | 1 | mhal | 41 | 15 | 41 |
| CAUTION | MEDIUM | 80.5 | 75.19999694824219 | 4.599999904632568 | 9 | pca_t2 | 541 | 433 | 437 |
| ALERT | HIGH | 53.400001525878906 | 68.80000305175781 | 8.600000381469727 | 4 | mhal | 88 | 86 | 188 |
| ALERT | HIGH | 31.399999618530273 | 79.4000015258789 | 6.5 | 0 | pca_spe | 51 | 23 | 23 |
| HEALTHY | LOW | 95.30000305175781 | 73.80000305175781 | 11.100000381469727 | 2 | mhal | 155 | 59 | 145 |
| HEALTHY | LOW | 95.0 | 77.80000305175781 | 8.5 | 2 | mhal | 213 | 74 | 112 |
| CAUTION | MEDIUM | 76.4000015258789 | 73.80000305175781 | 11.899999618530273 | 4 | omr | 175 | 160 | 141 |

---


## dbo.ACM_DefectTimeline

**Primary Key:** No primary key  
**Row Count:** 2,934  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 20:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| EventType | nvarchar | NO | 50 | — |
| FromZone | nvarchar | YES | 50 | — |
| ToZone | nvarchar | YES | 50 | — |
| HealthZone | nvarchar | NO | 50 | — |
| HealthAtEvent | float | NO | 53 | — |
| HealthIndex | float | NO | 53 | — |
| FusedZ | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Timestamp | EventType | FromZone | ToZone | HealthZone | HealthAtEvent | HealthIndex | FusedZ | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-18 00:00:00 | ZONE_CHANGE | START | GOOD | GOOD | 86.23999786376953 | 86.23999786376953 | 0.3995 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:30:00 | ZONE_CHANGE | GOOD | ALERT | ALERT | 15.0 | 15.0 | 2.3802 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:30:00 | ZONE_CHANGE | ALERT | GOOD | GOOD | 94.23999786376953 | 94.23999786376953 | 0.2472 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 04:30:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 73.47000122070312 | 73.47000122070312 | -0.6009 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 05:00:00 | ZONE_CHANGE | WATCH | ALERT | ALERT | 56.380001068115234 | 56.380001068115234 | -0.8796 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 09:00:00 | ZONE_CHANGE | ALERT | WATCH | WATCH | 75.12000274658203 | 75.12000274658203 | -0.5755 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 09:30:00 | ZONE_CHANGE | WATCH | ALERT | ALERT | 66.66000366210938 | 66.66000366210938 | -0.7073 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 12:00:00 | ZONE_CHANGE | ALERT | GOOD | GOOD | 99.97000122070312 | 99.97000122070312 | -0.018 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 14:30:00 | ZONE_CHANGE | GOOD | ALERT | ALERT | 66.20999908447266 | 66.20999908447266 | 0.7144 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 15:00:00 | ZONE_CHANGE | ALERT | WATCH | WATCH | 75.7300033569336 | 75.7300033569336 | 0.5661 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |

### Bottom 10 Records

| Timestamp | EventType | FromZone | ToZone | HealthZone | HealthAtEvent | HealthIndex | FusedZ | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 20:30:00 | ZONE_CHANGE | WATCH | GOOD | GOOD | 85.41999816894531 | 85.41999816894531 | -0.4131 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 18:00:00 | ZONE_CHANGE | ALERT | WATCH | WATCH | 73.98999786376953 | 73.98999786376953 | -0.5929 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 15:30:00 | ZONE_CHANGE | WATCH | ALERT | ALERT | 67.83999633789062 | 67.83999633789062 | -0.6886 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 15:00:00 | ZONE_CHANGE | ALERT | WATCH | WATCH | 74.61000061035156 | 74.61000061035156 | -0.5834 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 14:00:00 | ZONE_CHANGE | WATCH | ALERT | ALERT | 59.189998626708984 | 59.189998626708984 | -0.8303 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 13:30:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 75.13999938964844 | 75.13999938964844 | -0.5752 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 07:30:00 | ZONE_CHANGE | ALERT | GOOD | GOOD | 99.33999633789062 | 99.33999633789062 | 0.0817 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 07:00:00 | ZONE_CHANGE | GOOD | ALERT | ALERT | 62.63999938964844 | 62.63999938964844 | 0.7722 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 04:00:00 | ZONE_CHANGE | ALERT | GOOD | GOOD | 99.45999908447266 | 99.45999908447266 | 0.0739 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 03:30:00 | ZONE_CHANGE | GOOD | ALERT | ALERT | 66.2699966430664 | 66.2699966430664 | 0.7134 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |

---


## dbo.ACM_DetectorCorrelation

**Primary Key:** No primary key  
**Row Count:** 420  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorA | nvarchar | NO | 50 | — |
| DetectorB | nvarchar | NO | 50 | — |
| PearsonR | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| PairLabel | nvarchar | YES | 256 | — |
| DisturbanceHint | nvarchar | YES | 256 | — |

### Top 10 Records

| DetectorA | DetectorB | PearsonR | RunID | EquipID | PairLabel | DisturbanceHint |
| --- | --- | --- | --- | --- | --- | --- |
| Time-Series Anomaly (AR1) | Correlation Break (PCA-SPE) | 0.5879 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Time-Series Anomaly (AR1) <-> Correlation Break (PCA-SPE) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Multivariate Outlier (PCA-T2) | 0.5734 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Time-Series Anomaly (AR1) <-> Multivariate Outlier (PCA-T2) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Multivariate Distance (Mahalanobis) | 0.5782 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Time-Series Anomaly (AR1) <-> Multivariate Distance (Mahalanobis) | Temporal spikes seen by both detectors |
| Time-Series Anomaly (AR1) | Rare State (IsolationForest) | 0.575 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Time-Series Anomaly (AR1) <-> Rare State (IsolationForest) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Density Anomaly (GMM) | 0.6044 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Time-Series Anomaly (AR1) <-> Density Anomaly (GMM) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Baseline Consistency (OMR) | 0.5777 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Time-Series Anomaly (AR1) <-> Baseline Consistency (OMR) | Health baseline tracking repeated spikes |
| Time-Series Anomaly (AR1) | cusum_z | 0.0708 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Time-Series Anomaly (AR1) <-> cusum_z | Temporal spikes seen by both detectors |
| Correlation Break (PCA-SPE) | Multivariate Outlier (PCA-T2) | 0.7022 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Correlation Break (PCA-SPE) <-> Multivariate Outlier (PCA-T2) | Detectors reacting together; check shared cause |
| Correlation Break (PCA-SPE) | Multivariate Distance (Mahalanobis) | 0.8847 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Correlation Break (PCA-SPE) <-> Multivariate Distance (Mahalanobis) | Regime/cluster shift across many sensors |
| Correlation Break (PCA-SPE) | Rare State (IsolationForest) | 0.8239 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Correlation Break (PCA-SPE) <-> Rare State (IsolationForest) | Detectors reacting together; check shared cause |

### Bottom 10 Records

| DetectorA | DetectorB | PearsonR | RunID | EquipID | PairLabel | DisturbanceHint |
| --- | --- | --- | --- | --- | --- | --- |
| Rare State (IsolationForest) | Density Anomaly (GMM) | 0.692 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Rare State (IsolationForest) <-> Density Anomaly (GMM) | Detectors reacting together; check shared cause |
| Rare State (IsolationForest) | Baseline Consistency (OMR) | 0.4023 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Rare State (IsolationForest) <-> Baseline Consistency (OMR) | Health baseline moving with a pattern change |
| Rare State (IsolationForest) | cusum_z | 0.0247 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Rare State (IsolationForest) <-> cusum_z | Detectors reacting together; check shared cause |
| Density Anomaly (GMM) | Baseline Consistency (OMR) | 0.7282 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Density Anomaly (GMM) <-> Baseline Consistency (OMR) | Health baseline moving with a pattern change |
| Density Anomaly (GMM) | cusum_z | -0.0286 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Density Anomaly (GMM) <-> cusum_z | Detectors reacting together; check shared cause |
| Baseline Consistency (OMR) | cusum_z | -0.1076 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Baseline Consistency (OMR) <-> cusum_z | Overall health shifting with another detector |
| Time-Series Anomaly (AR1) | Correlation Break (PCA-SPE) | 0.534 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Time-Series Anomaly (AR1) <-> Correlation Break (PCA-SPE) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Multivariate Outlier (PCA-T²) | 0.691 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Time-Series Anomaly (AR1) <-> Multivariate Outlier (PCA-T²) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Multivariate Distance (Mahalanobis) | 0.5677 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Time-Series Anomaly (AR1) <-> Multivariate Distance (Mahalanobis) | Temporal spikes seen by both detectors |
| Time-Series Anomaly (AR1) | Rare State (IsolationForest) | 0.7564 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | Time-Series Anomaly (AR1) <-> Rare State (IsolationForest) | Transient spikes align with pattern change |

---


## dbo.ACM_DetectorForecast_TS

**Primary Key:** RunID, EquipID, DetectorName, Timestamp  
**Row Count:** 3,864  
**Date Range:** 2023-10-25 00:59:00 to 2024-06-23 01:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| DetectorName | nvarchar | NO | 100 | — |
| Timestamp | datetime2 | NO | — | — |
| ForecastValue | float | NO | 53 | — |
| CiLower | float | YES | 53 | — |
| CiUpper | float | YES | 53 | — |
| ForecastStd | float | YES | 53 | — |
| Method | nvarchar | NO | 50 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| RunID | EquipID | DetectorName | Timestamp | ForecastValue | CiLower | CiUpper | ForecastStd | Method | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 02:59:00 | 9.283351885164064 | 5.554204629724673 | 13.012499140603456 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 03:59:00 | 8.642291843532664 | 3.63887127501323 | 13.645712412052099 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 04:59:00 | 8.068847263872346 | 2.2431869831338327 | 13.89450754461086 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 05:59:00 | 7.555886441817434 | 1.1478317791423995 | 13.96394110449247 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 06:59:00 | 7.097029885671912 | 0.25858740253493373 | 13.93547236880889 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 07:59:00 | 6.686570977183984 | -0.4776598598414239 | 13.850801814209392 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 08:59:00 | 6.319405000582638 | -1.0952111341655497 | 13.734021135330824 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 09:59:00 | 5.990965657238272 | -1.6180709322260274 | 13.600002246702571 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 10:59:00 | 5.697168276405239 | -2.0639319060531687 | 13.458268458863646 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ar1 | 2023-12-24 11:59:00 | 5.434359015780673 | -2.4463057934904198 | 13.315023825051767 | 0.0 | DetectorAR1 | 2025-12-05 11:37:39 |

### Bottom 10 Records

| RunID | EquipID | DetectorName | Timestamp | ForecastValue | CiLower | CiUpper | ForecastStd | Method | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-21 02:59:00 | 6.4802127575201 | -1.4543710283745144 | 14.414796543414713 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-21 01:59:00 | 6.480214750837116 | -1.4543690350211307 | 14.414798536695363 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-21 00:59:00 | 6.480216888686802 | -1.4543668971296126 | 14.414800674503216 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-20 23:59:00 | 6.480219181549022 | -1.4543646042192728 | 14.414802967317318 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-20 22:59:00 | 6.480221640663522 | -1.4543621450494228 | 14.414805426376466 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-20 21:59:00 | 6.480224278085025 | -1.4543595075642521 | 14.414808063734302 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-20 20:59:00 | 6.480227106742327 | -1.4543566788337134 | 14.414810892318368 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-20 19:59:00 | 6.480230140501672 | -1.4543536449901282 | 14.414813925993473 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-20 18:59:00 | 6.480233394234726 | -1.4543503911601743 | 14.414817179629626 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2 | 2024-01-20 17:59:00 | 6.48023688389148 | -1.4543469013919585 | 14.414820669174919 | 0.0 | DetectorAR1 | 2025-12-05 11:38:03 |

---


## dbo.ACM_DriftEvents

**Primary Key:** No primary key  
**Row Count:** 2  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| RunID | EquipID |
| --- | --- |
| 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |
| 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 |

---


## dbo.ACM_DriftSeries

**Primary Key:** No primary key  
**Row Count:** 10,893  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| DriftValue | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Timestamp | DriftValue | RunID | EquipID |
| --- | --- | --- | --- |
| 2023-10-18 00:00:00 | -1.5786000490188599 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:30:00 | -1.4704999923706055 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:00:00 | -1.3588000535964966 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:30:00 | -1.2727999687194824 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 02:00:00 | -1.2182999849319458 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 02:30:00 | -1.1996999979019165 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 03:00:00 | -1.2050000429153442 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 03:30:00 | -1.2197999954223633 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 04:00:00 | -1.2223999500274658 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 04:30:00 | -1.2562999725341797 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |

### Bottom 10 Records

| Timestamp | DriftValue | RunID | EquipID |
| --- | --- | --- | --- |
| 2025-09-14 23:00:00 | 0.24310000240802765 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:30:00 | 0.2565999925136566 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:00:00 | 0.2711000144481659 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 21:30:00 | 0.28760001063346863 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 21:00:00 | 0.3061999976634979 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 20:30:00 | 0.3276999890804291 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 20:00:00 | 0.35089999437332153 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 19:30:00 | 0.37599998712539673 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 19:00:00 | 0.4016999900341034 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 18:30:00 | 0.4275999963283539 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |

---


## dbo.ACM_EnhancedFailureProbability_TS

**Primary Key:** RunID, EquipID, Timestamp, ForecastHorizon_Hours  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| ForecastHorizon_Hours | float | NO | 53 | — |
| ForecastHealth | float | YES | 53 | — |
| ForecastUncertainty | float | YES | 53 | — |
| FailureProbability | float | NO | 53 | — |
| RiskLevel | nvarchar | NO | 50 | — |
| Confidence | float | YES | 53 | — |
| Model | nvarchar | YES | 50 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_EnhancedMaintenanceRecommendation

**Primary Key:** RunID, EquipID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| UrgencyScore | float | NO | 53 | — |
| MaintenanceRequired | bit | NO | — | — |
| EarliestMaintenance | float | YES | 53 | — |
| PreferredWindowStart | float | YES | 53 | — |
| PreferredWindowEnd | float | YES | 53 | — |
| LatestSafeTime | float | YES | 53 | — |
| FailureProbAtLatest | float | YES | 53 | — |
| FailurePattern | nvarchar | YES | 200 | — |
| Confidence | float | YES | 53 | — |
| EstimatedDuration_Hours | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_EpisodeCulprits

**Primary Key:** ID  
**Row Count:** 581  
**Date Range:** 2025-12-11 03:51:17 to 2025-12-11 03:55:08  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EpisodeID | int | NO | 10 | — |
| DetectorType | nvarchar | YES | 64 | — |
| SensorName | nvarchar | YES | 200 | — |
| ContributionPct | float | YES | 53 | — |
| Rank | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |
| EquipID | int | NO | 10 | ((1)) |

### Top 10 Records

| ID | RunID | EpisodeID | DetectorType | SensorName | ContributionPct | Rank | CreatedAt | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33102 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | Correlation Break (PCA-SPE) | NULL | 22.951675415039062 | 1 | 2025-12-11 03:51:17 | 1 |
| 33103 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | Multivariate Outlier (PCA-T2) | NULL | 18.488798141479492 | 2 | 2025-12-11 03:51:17 | 1 |
| 33104 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | Baseline Consistency (OMR) | NULL | 16.30406951904297 | 3 | 2025-12-11 03:51:17 | 1 |
| 33105 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | Time-Series Anomaly (AR1) | NULL | 13.520299911499023 | 4 | 2025-12-11 03:51:17 | 1 |
| 33106 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | Rare State (IsolationForest) | NULL | 9.547196388244629 | 5 | 2025-12-11 03:51:17 | 1 |
| 33107 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | Multivariate Distance (Mahalanobis) | NULL | 9.086981773376465 | 6 | 2025-12-11 03:51:17 | 1 |
| 33108 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | Density Anomaly (GMM) | NULL | 5.758297920227051 | 7 | 2025-12-11 03:51:17 | 1 |
| 33109 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | cusum_z | NULL | 4.342683792114258 | 8 | 2025-12-11 03:51:17 | 1 |
| 33110 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | Multivariate Distance (Mahalanobis) | NULL | 29.325246810913086 | 1 | 2025-12-11 03:51:35 | 1 |
| 33111 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | Correlation Break (PCA-SPE) | NULL | 22.192501068115234 | 2 | 2025-12-11 03:51:35 | 1 |

### Bottom 10 Records

| ID | RunID | EpisodeID | DetectorType | SensorName | ContributionPct | Rank | CreatedAt | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33682 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 2 | Time-Series Anomaly (AR1) | NULL | 2.7659318447113037 | 7 | 2025-12-11 03:55:08 | 1 |
| 33681 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 2 | Rare State (IsolationForest) | NULL | 3.680147886276245 | 6 | 2025-12-11 03:55:08 | 1 |
| 33680 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 2 | Multivariate Outlier (PCA-T2) | NULL | 7.447863578796387 | 5 | 2025-12-11 03:55:08 | 1 |
| 33679 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 2 | Density Anomaly (GMM) | NULL | 14.351807594299316 | 4 | 2025-12-11 03:55:08 | 1 |
| 33678 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 2 | Baseline Consistency (OMR) | NULL | 16.639680862426758 | 3 | 2025-12-11 03:55:08 | 1 |
| 33677 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 2 | Correlation Break (PCA-SPE) | NULL | 25.991621017456055 | 2 | 2025-12-11 03:55:08 | 1 |
| 33676 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 2 | Multivariate Distance (Mahalanobis) | NULL | 28.683015823364258 | 1 | 2025-12-11 03:55:08 | 1 |
| 33675 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | Baseline Consistency (OMR) | NULL | 5.129462718963623 | 7 | 2025-12-11 03:55:08 | 1 |
| 33674 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | Rare State (IsolationForest) | NULL | 9.302803039550781 | 6 | 2025-12-11 03:55:08 | 1 |
| 33673 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | Density Anomaly (GMM) | NULL | 10.606640815734863 | 5 | 2025-12-11 03:55:08 | 1 |

---


## dbo.ACM_EpisodeDiagnostics

**Primary Key:** ID  
**Row Count:** 98  
**Date Range:** 2023-10-18 22:00:00 to 2025-09-11 15:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| episode_id | int | YES | 10 | — |
| peak_z | float | YES | 53 | — |
| peak_timestamp | datetime2 | YES | — | — |
| duration_h | float | YES | 53 | — |
| dominant_sensor | nvarchar | YES | 200 | — |
| severity | nvarchar | YES | 50 | — |
| severity_reason | nvarchar | YES | 500 | — |
| avg_z | float | YES | 53 | — |
| min_health_index | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | episode_id | peak_z | peak_timestamp | duration_h | dominant_sensor | severity | severity_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11994 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 1 | 1.7678886172616954 | 2023-11-19 22:59:00 | 15.0 | Multivariate Outlier (PCA-T²) | LOW | UNKNOWN |
| 11995 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2 | 1.8926318063021332 | 2023-11-22 16:59:00 | 7.0 | Multivariate Distance (Mahalanobis) | LOW | UNKNOWN |
| 11996 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 3 | 1.62326336506371 | 2023-11-23 23:59:00 | 8.0 | Multivariate Outlier (PCA-T²) | LOW | UNKNOWN |
| 11997 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 4 | 1.685754267680011 | 2023-11-29 23:59:00 | 8.0 | Multivariate Outlier (PCA-T²) | LOW | UNKNOWN |
| 11998 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 5 | 1.7533280504940343 | 2023-12-03 22:59:00 | 8.0 | Multivariate Outlier (PCA-T²) | LOW | UNKNOWN |
| 11999 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 6 | 1.73752690432199 | 2023-12-07 10:59:00 | 13.0 | Multivariate Distance (Mahalanobis) | LOW | UNKNOWN |
| 12000 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 7 | 1.5627700914379705 | 2023-12-08 21:59:00 | 2.0 | Multivariate Distance (Mahalanobis) | LOW | UNKNOWN |
| 12001 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 8 | 1.895926432154782 | 2023-12-10 19:59:00 | 17.0 | Multivariate Distance (Mahalanobis) | LOW | UNKNOWN |
| 12002 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 9 | 1.9155707193500928 | 2023-12-12 19:59:00 | 40.0 | Multivariate Outlier (PCA-T²) | LOW | UNKNOWN |
| 12003 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 10 | 1.849379497256723 | 2023-12-18 21:59:00 | 13.0 | Multivariate Outlier (PCA-T²) | LOW | UNKNOWN |

### Bottom 10 Records

| ID | RunID | EquipID | episode_id | peak_z | peak_timestamp | duration_h | dominant_sensor | severity | severity_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14520 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2 | 1.8099453765647722 | 2025-09-11 15:30:00 | 7.5 | Multivariate Distance (Mahalanobis) | LOW | UNKNOWN |
| 14519 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 1 | 2.618636853071696 | 2025-09-11 01:00:00 | 2.5 | Multivariate Outlier (PCA-T2) | MEDIUM | UNKNOWN |
| 14516 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 4 | 2.714839787148049 | 2025-06-12 23:00:00 | 1.0 | Density Anomaly (GMM) | MEDIUM | UNKNOWN |
| 14515 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 3 | 2.685987230123123 | 2025-06-12 00:00:00 | 6.5 | Multivariate Outlier (PCA-T2) | MEDIUM | UNKNOWN |
| 14514 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2 | 2.6576966193434868 | 2025-05-13 01:00:00 | 6.0 | Multivariate Outlier (PCA-T2) | MEDIUM | UNKNOWN |
| 14513 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 1 | 2.567368367695458 | 2025-05-12 00:00:00 | 7.0 | Multivariate Outlier (PCA-T2) | MEDIUM | UNKNOWN |
| 14508 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 4 | 1.0684760354282654 | 2025-04-14 00:00:00 | 2.5 | Density Anomaly (GMM) | LOW | UNKNOWN |
| 14507 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 3 | 1.4516717445856173 | 2025-04-11 00:00:00 | 30.5 | Density Anomaly (GMM) | LOW | UNKNOWN |
| 14506 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 2 | 1.0093497297484157 | 2025-03-15 00:00:00 | 2.5 | Density Anomaly (GMM) | LOW | UNKNOWN |
| 14505 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 1 | 1.4646564720373885 | 2025-03-12 00:00:00 | 31.0 | Density Anomaly (GMM) | LOW | UNKNOWN |

---


## dbo.ACM_EpisodeMetrics

**Primary Key:** No primary key  
**Row Count:** 14  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| TotalEpisodes | int | NO | 10 | — |
| TotalDurationHours | float | NO | 53 | — |
| AvgDurationHours | float | NO | 53 | — |
| MedianDurationHours | float | NO | 53 | — |
| MaxDurationHours | float | NO | 53 | — |
| MinDurationHours | float | NO | 53 | — |
| RatePerDay | float | NO | 53 | — |
| MeanInterarrivalHours | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| TotalEpisodes | TotalDurationHours | AvgDurationHours | MedianDurationHours | MaxDurationHours | MinDurationHours | RatePerDay | MeanInterarrivalHours | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 92.5 | 6.61 | 3.75 | 19.0 | 1.0 | 0.0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 11 | 58.5 | 5.32 | 6.0 | 13.5 | 1.0 | 0.0 | 0.0 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| 5 | 11.5 | 2.3 | 1.5 | 4.5 | 1.0 | 0.0 | 0.0 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 14 | 45.0 | 3.21 | 3.0 | 6.0 | 1.0 | 0.0 | 0.0 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 |
| 11 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 |
| 4 | 20.5 | 5.12 | 6.25 | 7.0 | 1.0 | 0.0 | 0.0 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 |
| 2 | 3.0 | 1.5 | 1.5 | 2.0 | 1.0 | 0.0 | 0.0 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 |
| 2 | 10.0 | 5.0 | 5.0 | 7.5 | 2.5 | 0.0 | 0.0 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 |
| 9 | 47.0 | 5.22 | 3.0 | 14.0 | 1.0 | 0.0 | 0.0 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 |

### Bottom 10 Records

| TotalEpisodes | TotalDurationHours | AvgDurationHours | MedianDurationHours | MaxDurationHours | MinDurationHours | RatePerDay | MeanInterarrivalHours | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 10 | 31.0 | 3.1 | 2.5 | 5.5 | 1.5 | 0.0 | 0.0 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| 4 | 66.5 | 16.62 | 16.5 | 31.0 | 2.5 | 0.0 | 0.0 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 |
| 1 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 0.0 | 0.0 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 9 | 47.0 | 5.22 | 3.0 | 14.0 | 1.0 | 0.0 | 0.0 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 |
| 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 |
| 2 | 10.0 | 5.0 | 5.0 | 7.5 | 2.5 | 0.0 | 0.0 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2 | 3.0 | 1.5 | 1.5 | 2.0 | 1.0 | 0.0 | 0.0 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 |
| 4 | 20.5 | 5.12 | 6.25 | 7.0 | 1.0 | 0.0 | 0.0 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 |
| 11 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 |

---


## dbo.ACM_Episodes

**Primary Key:** No primary key  
**Row Count:** 14  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| EpisodeCount | int | YES | 10 | — |
| MedianDurationMinutes | float | YES | 53 | — |
| CoveragePct | float | YES | 53 | — |
| TimeInAlertPct | float | YES | 53 | — |
| MaxFusedZ | float | YES | 53 | — |
| AvgFusedZ | float | YES | 53 | — |

### Top 10 Records

| RunID | EquipID | EpisodeCount | MedianDurationMinutes | CoveragePct | TimeInAlertPct | MaxFusedZ | AvgFusedZ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 14 | 225.0 | NULL | NULL | 3.0610144031389273 | 1.3300895239858785 |
| 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 | 11 | 360.0 | NULL | NULL | 3.155382065069861 | 1.3811883954618758 |
| 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 5 | 90.0 | NULL | NULL | 3.0200201679646046 | 1.3868064145836627 |
| 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 14 | 180.0 | NULL | NULL | 3.0890736518126807 | 1.1594247276529421 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 11 | 780.0 | NULL | NULL | 1.9155707193500928 | 1.2239104486643215 |
| D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 4 | 375.0 | NULL | NULL | 2.714839787148049 | 1.8125611470308292 |
| 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | 2 | 90.0 | NULL | NULL | 1.5712091662287793 | 1.0968430228734005 |
| 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2 | 300.0 | NULL | NULL | 2.618636853071696 | 1.4238169198426824 |
| E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 | 4 | 840.0 | NULL | NULL | 2.429699413491418 | 1.3920619483295007 |
| ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | 9 | 180.0 | NULL | NULL | 2.8622627049474714 | 1.3179738272330825 |

### Bottom 10 Records

| RunID | EquipID | EpisodeCount | MedianDurationMinutes | CoveragePct | TimeInAlertPct | MaxFusedZ | AvgFusedZ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 7 | 720.0 | NULL | NULL | 2.4770509242211958 | 0.9752637030744661 |
| 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | 10 | 150.0 | NULL | NULL | 2.1406981719192535 | 1.2073447257188232 |
| 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 4 | 990.0 | NULL | NULL | 1.4646564720373885 | 0.9719713859708234 |
| 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 1 | 120.0 | NULL | NULL | 3.988077274486474 | 1.336501737451136 |
| ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | 9 | 180.0 | NULL | NULL | 2.8622627049474714 | 1.3179738272330825 |
| E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 | 4 | 840.0 | NULL | NULL | 2.429699413491418 | 1.3920619483295007 |
| 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2 | 300.0 | NULL | NULL | 2.618636853071696 | 1.4238169198426824 |
| 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | 2 | 90.0 | NULL | NULL | 1.5712091662287793 | 1.0968430228734005 |
| D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 4 | 375.0 | NULL | NULL | 2.714839787148049 | 1.8125611470308292 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 11 | 780.0 | NULL | NULL | 1.9155707193500928 | 1.2239104486643215 |

---


## dbo.ACM_EpisodesQC

**Primary Key:** RecordID  
**Row Count:** 14  
**Date Range:** 2025-12-05 11:37:38 to 2025-12-11 09:25:08  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RecordID | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| EpisodeCount | int | YES | 10 | — |
| MedianDurationMinutes | float | YES | 53 | — |
| CoveragePct | float | YES | 53 | — |
| TimeInAlertPct | float | YES | 53 | — |
| MaxFusedZ | float | YES | 53 | — |
| AvgFusedZ | float | YES | 53 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

### Top 10 Records

| RecordID | RunID | EquipID | EpisodeCount | MedianDurationMinutes | CoveragePct | TimeInAlertPct | MaxFusedZ | AvgFusedZ | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 973 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 11 | 780.0 | 18.55 | 0.12 | 2.0032997131347656 | 0.0 | 2025-12-05 11:37:38 |
| 974 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 7 | 720.0 | 18.65 | 1.78 | 2.5572545528411865 | -7.55385620720972e-09 | 2025-12-05 11:38:03 |
| 975 | E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 | 4 | 840.0 | 13.3 | 2.21 | 3.263977289199829 | -1.053783815763154e-08 | 2025-12-05 11:38:27 |
| 1147 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 1 | 120.0 | 4.17 | 3.09 | 3.988077163696289 | 8.986782873421362e-09 | 2025-12-11 09:21:17 |
| 1148 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 5 | 90.0 | 1.47 | 4.97 | 4.233062267303467 | 1.8959727121625747e-09 | 2025-12-11 09:21:34 |
| 1149 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | 2 | 90.0 | 0.7 | 5.01 | 3.2824506759643555 | -4.780322360176115e-09 | 2025-12-11 09:21:50 |
| 1150 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | 9 | 180.0 | 5.7 | 4.75 | 4.579349994659424 | -1.0814166451211804e-08 | 2025-12-11 09:22:21 |
| 1151 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 14 | 180.0 | 5.65 | 3.91 | 3.669748067855835 | 0.0 | 2025-12-11 09:22:53 |
| 1152 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 14 | 225.0 | 11.0 | 4.5 | 3.4808783531188965 | 0.0 | 2025-12-11 09:23:24 |
| 1153 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | 10 | 150.0 | 3.68 | 4.38 | 2.831772804260254 | 0.0 | 2025-12-11 09:23:55 |

### Bottom 10 Records

| RecordID | RunID | EquipID | EpisodeCount | MedianDurationMinutes | CoveragePct | TimeInAlertPct | MaxFusedZ | AvgFusedZ | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1157 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2 | 300.0 | 1.28 | 4.46 | 2.8297431468963623 | 0.0 | 2025-12-11 09:25:08 |
| 1156 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 4 | 375.0 | 2.37 | 5.46 | 2.7148396968841553 | 0.0 | 2025-12-11 09:24:53 |
| 1155 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 4 | 990.0 | 8.1 | 0.0 | 1.4646564722061157 | -4.3250536485572866e-09 | 2025-12-11 09:24:38 |
| 1154 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 | 11 | 360.0 | 7.34 | 5.41 | 3.185734272003174 | 6.064701629782121e-09 | 2025-12-11 09:24:23 |
| 1153 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | 10 | 150.0 | 3.68 | 4.38 | 2.831772804260254 | 0.0 | 2025-12-11 09:23:55 |
| 1152 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 14 | 225.0 | 11.0 | 4.5 | 3.4808783531188965 | 0.0 | 2025-12-11 09:23:24 |
| 1151 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 14 | 180.0 | 5.65 | 3.91 | 3.669748067855835 | 0.0 | 2025-12-11 09:22:53 |
| 1150 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | 9 | 180.0 | 5.7 | 4.75 | 4.579349994659424 | -1.0814166451211804e-08 | 2025-12-11 09:22:21 |
| 1149 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | 2 | 90.0 | 0.7 | 5.01 | 3.2824506759643555 | -4.780322360176115e-09 | 2025-12-11 09:21:50 |
| 1148 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 5 | 90.0 | 1.47 | 4.97 | 4.233062267303467 | 1.8959727121625747e-09 | 2025-12-11 09:21:34 |

---


## dbo.ACM_FailureCausation

**Primary Key:** RunID, EquipID, Detector  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| PredictedFailureTime | datetime2 | NO | — | — |
| FailurePattern | nvarchar | YES | 200 | — |
| Detector | nvarchar | NO | 100 | — |
| MeanZ | float | YES | 53 | — |
| MaxZ | float | YES | 53 | — |
| SpikeCount | int | YES | 10 | — |
| TrendSlope | float | YES | 53 | — |
| ContributionWeight | float | YES | 53 | — |
| ContributionPct | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_FailureForecast

**Primary Key:** EquipID, RunID, Timestamp  
**Row Count:** 672  
**Date Range:** 2023-10-25 00:59:00 to 2024-06-23 01:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| Timestamp | datetime2 | NO | — | — |
| FailureProb | float | NO | 53 | — |
| SurvivalProb | float | YES | 53 | — |
| HazardRate | float | YES | 53 | — |
| ThresholdUsed | float | NO | 53 | — |
| Method | nvarchar | NO | 50 | ('GaussianCDF') |
| CreatedAt | datetime2 | NO | — | (getdate()) |

### Top 10 Records

| EquipID | RunID | Timestamp | FailureProb | SurvivalProb | HazardRate | ThresholdUsed | Method | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 02:59:00 | 0.39245658829056385 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 03:59:00 | 0.5736189861196903 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 04:59:00 | 0.6687290634709507 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 05:59:00 | 0.7233874508054856 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 06:59:00 | 0.7568215579405445 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 07:59:00 | 0.7781097810180873 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 08:59:00 | 0.7919330936959887 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 09:59:00 | 0.8010937610571449 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 10:59:00 | 0.8072651800165334 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 11:59:00 | 0.8114708860460553 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:37:39 |

### Bottom 10 Records

| EquipID | RunID | Timestamp | FailureProb | SurvivalProb | HazardRate | ThresholdUsed | Method | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-21 02:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-21 01:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-21 00:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 23:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 22:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 21:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 20:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 19:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 18:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 17:59:00 | 0.7896123333918845 | NULL | NULL | 50.0 | GaussianTail | 2025-12-05 11:38:03 |

---


## dbo.ACM_FailureForecast_TS

**Primary Key:** RunID, EquipID, Timestamp  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| FailureProb | float | NO | 53 | — |
| ThresholdUsed | float | NO | 53 | — |
| Method | nvarchar | NO | 50 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_FailureHazard_TS

**Primary Key:** EquipID, RunID, Timestamp  
**Row Count:** 2,352  
**Date Range:** 2023-10-20 00:30:00 to 2025-09-18 11:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| HazardRaw | float | YES | 53 | — |
| HazardSmooth | float | YES | 53 | — |
| Survival | float | YES | 53 | — |
| FailureProb | float | YES | 53 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

### Top 10 Records

| Timestamp | HazardRaw | HazardSmooth | Survival | FailureProb | RunID | EquipID | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-20 00:30:00 | 0.7079578009461528 | 0.21241520494257718 | 0.8992379446769597 | 0.10076205532304028 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 00:30:00 | 0.7079578009461528 | 0.21290118540544353 | 0.8990194651859352 | 0.10098053481406477 | dd25af55-769a-4469-81e2-e17330ebca91 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 01:00:00 | 0.001746989758419388 | 0.14921474038732982 | 0.8345897663308537 | 0.1654102336691463 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 01:00:00 | 0.001746989758419388 | 0.14955492671133627 | 0.8342450823545651 | 0.16575491764543493 | dd25af55-769a-4469-81e2-e17330ebca91 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 01:30:00 | 0.03296316868437979 | 0.1143392688764448 | 0.7882148239098503 | 0.21178517609014968 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 01:30:00 | 0.03296316868437979 | 0.11457739930324931 | 0.7877954880738032 | 0.21220451192619683 | dd25af55-769a-4469-81e2-e17330ebca91 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 02:00:00 | 0.0497958130932218 | 0.09497623214147789 | 0.7516588437662725 | 0.24834115623372754 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 02:00:00 | 0.0497958130932218 | 0.09514292344024106 | 0.7511963444127457 | 0.24880365558725426 | dd25af55-769a-4469-81e2-e17330ebca91 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 02:30:00 | 0.05333780457896565 | 0.08248470387272422 | 0.7212892270049297 | 0.2787107729950703 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 02:30:00 | 0.05333780457896565 | 0.08260138778185842 | 0.7208033599211245 | 0.27919664007887546 | dd25af55-769a-4469-81e2-e17330ebca91 | 1 | 2025-12-08 18:31:06 |

### Bottom 10 Records

| Timestamp | HazardRaw | HazardSmooth | Survival | FailureProb | RunID | EquipID | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-18 11:00:00 | 0.0007225369069171444 | 0.0007405305076694454 | 0.5339589527366884 | 0.46604104726331164 | 3b1b1b1b-d9cc-47ad-9f7b-04ad1912c73e | 1 | 2025-12-08 18:27:45 |
| 2025-09-18 11:00:00 | 0.0007162210624672929 | 0.0007340644594252792 | 0.5339609273994637 | 0.46603907260053634 | 453a03f2-a2f9-452d-bf48-4ac77b7387c3 | 1 | 2025-12-08 18:30:09 |
| 2025-09-18 11:00:00 | 3.838918001329742e-05 | 3.980665533048449e-05 | 0.5098016055042732 | 0.4901983944957268 | 8fd5373f-292c-4565-9f68-b00efff3da31 | 1 | 2025-12-08 18:33:24 |
| 2025-09-18 10:30:00 | 0.0007299791818650741 | 0.000748242050849003 | 0.5341566957902926 | 0.46584330420970743 | 3b1b1b1b-d9cc-47ad-9f7b-04ad1912c73e | 1 | 2025-12-08 18:27:45 |
| 2025-09-18 10:30:00 | 0.0007236010917343461 | 0.0007417116295501305 | 0.5341569442392748 | 0.46584305576072516 | 453a03f2-a2f9-452d-bf48-4ac77b7387c3 | 1 | 2025-12-08 18:30:09 |
| 2025-09-18 10:30:00 | 3.896619363304359e-05 | 4.041414475213609e-05 | 0.5098117523536496 | 0.4901882476463504 | 8fd5373f-292c-4565-9f68-b00efff3da31 | 1 | 2025-12-08 18:33:24 |
| 2025-09-18 10:00:00 | 0.0007375314716760365 | 0.0007560689946992584 | 0.5343565724277539 | 0.4656434275722461 | 3b1b1b1b-d9cc-47ad-9f7b-04ad1912c73e | 1 | 2025-12-08 18:27:45 |
| 2025-09-18 10:00:00 | 0.0007310902640843793 | 0.000749473288614038 | 0.5343550761849697 | 0.4656449238150303 | 453a03f2-a2f9-452d-bf48-4ac77b7387c3 | 1 | 2025-12-08 18:30:09 |
| 2025-09-18 10:00:00 | 3.9555420497232835e-05 | 4.103469523174716e-05 | 0.5098220542607129 | 0.4901779457392871 | 8fd5373f-292c-4565-9f68-b00efff3da31 | 1 | 2025-12-08 18:33:24 |
| 2025-09-18 09:30:00 | 0.0007451959465220986 | 0.0007640136474234966 | 0.5345586158332996 | 0.46544138416670044 | 3b1b1b1b-d9cc-47ad-9f7b-04ad1912c73e | 1 | 2025-12-08 18:27:45 |

---


## dbo.ACM_FeatureDropLog

**Primary Key:** LogID  
**Row Count:** 14,043  
**Date Range:** 2025-12-01 10:33:45 to 2025-12-11 09:25:01  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| LogID | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| FeatureName | nvarchar | NO | 500 | — |
| Reason | nvarchar | NO | 200 | — |
| TrainMedian | float | YES | 53 | — |
| TrainStd | float | YES | 53 | — |
| Timestamp | datetime | NO | — | (getdate()) |

### Top 10 Records

| LogID | RunID | EquipID | FeatureName | Reason | TrainMedian | TrainStd | Timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_skew | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 2 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_kurt | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 3 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_skew | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 4 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_skew | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 5 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_kurt | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 6 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_kurt | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 7 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.06I03_1FD Fan Motor Current_skew | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 8 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_kurt | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 9 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.06T31_1FD Fan Inlet Temperature_skew | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |
| 10 | AC26C07F-B9B1-4F07-9A29-3236D0A5792A | 1 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_skew | low_variance | 0.0 | 0.0 | 2025-12-01 10:33:45 |

### Bottom 10 Records

| LogID | RunID | EquipID | FeatureName | Reason | TrainMedian | TrainStd | Timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14043 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14042 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14041 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14040 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.06T31_1FD Fan Inlet Temperature_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14039 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14038 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14037 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14036 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14035 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | DEMO.SIM.06I03_1FD Fan Motor Current_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:25:01 |
| 14034 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_energy_0 | low_variance | 0.0 | 0.0 | 2025-12-11 09:24:45 |

---


## dbo.ACM_ForecastContext

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| RunID | nvarchar | NO | 64 | — |
| CurrentHealth | float | NO | 53 | — |
| CurrentRegime | int | YES | 10 | — |
| CurrentRegimeState | nvarchar | YES | 32 | — |
| DriftZ | float | YES | 53 | — |
| DriftTrend | nvarchar | YES | 16 | — |
| DriftEventCount | int | YES | 10 | — |
| ModelDriftDetected | bit | YES | — | ((0)) |
| OMR_Z | float | YES | 53 | — |
| OMR_Trend | nvarchar | YES | 16 | — |
| TopOMRSensor1 | nvarchar | YES | 128 | — |
| TopOMRSensor1_Contrib | float | YES | 53 | — |
| TopOMRSensor2 | nvarchar | YES | 128 | — |
| TopOMRSensor2_Contrib | float | YES | 53 | — |
| TopOMRSensor3 | nvarchar | YES | 128 | — |
| TopOMRSensor3_Contrib | float | YES | 53 | — |
| RegimeStability | float | YES | 53 | — |
| RecentRegimeTransitions | int | YES | 10 | — |
| DominantRegime | int | YES | 10 | — |
| DegradationRateMultiplier | float | YES | 53 | ((1.0)) |
| ConfidenceAdjustment | float | YES | 53 | ((0.0)) |
| RetrainingRecommended | bit | YES | — | ((0)) |
| RetrainingReason | nvarchar | YES | 256 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

---


## dbo.ACM_ForecastState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 913  
**Date Range:** 2012-01-12 23:30:00 to 2025-09-14 23:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| StateVersion | int | NO | 10 | — |
| ModelType | nvarchar | YES | 50 | — |
| ModelParamsJson | nvarchar | YES | -1 | — |
| ResidualVariance | float | YES | 53 | — |
| LastForecastHorizonJson | nvarchar | YES | -1 | — |
| HazardBaseline | float | YES | 53 | — |
| LastRetrainTime | datetime2 | YES | — | — |
| TrainingDataHash | nvarchar | YES | 64 | — |
| TrainingWindowHours | int | YES | 10 | — |
| ForecastQualityJson | nvarchar | YES | -1 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

### Top 10 Records

| EquipID | StateVersion | ModelType | ModelParamsJson | ResidualVariance | LastForecastHorizonJson | HazardBaseline | LastRetrainTime | TrainingDataHash | TrainingWindowHours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": 0... | 486.4996093433315 | [{"timestamp": "2012-01-13T00:30:00", "health": 80.66468525873267, "horizon_hours": 1}, {"timesta... | 0.0 | 2012-01-12 23:30:00 | f604256a41b9c182 | 72 |
| 1 | 2 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": -... | 398.7796011809328 | [{"timestamp": "2012-01-13T00:30:00", "health": 58.686868298412676, "horizon_hours": 1}, {"timest... | 0.9574096714732573 | 2012-01-12 23:30:00 | a408f517b3122fb9 | 72 |
| 1 | 3 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": 0... | 500.77935317670347 | [{"timestamp": "2023-10-20T01:00:00", "health": 86.6484225058764, "horizon_hours": 1}, {"timestam... | 0.0 | 2023-10-20 00:00:00 | 997180dce3b28459 | 72 |
| 1 | 4 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": -... | 536.2202653505643 | [{"timestamp": "2023-11-22T00:30:00", "health": 75.55050864233007, "horizon_hours": 1}, {"timesta... | 0.3912422630036432 | 2023-11-21 23:30:00 | 2990929a2a118c62 | 72 |
| 1 | 5 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": -... | 365.87708479675916 | [{"timestamp": "2023-11-22T00:30:00", "health": 75.55050864269842, "horizon_hours": 1}, {"timesta... | 0.3912422629658352 | 2023-10-20 00:00:00 | d2f3c060f9bdb06e | 72 |
| 1 | 6 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": -... | 267.1186160666233 | [{"timestamp": "2023-11-22T00:30:00", "health": 70.94030204346264, "horizon_hours": 1}, {"timesta... | 0.6511012390046419 | 2023-11-21 23:30:00 | 58425170ba165d15 | 72 |
| 1 | 7 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": -... | 441.70591903837004 | [{"timestamp": "2024-03-11T00:30:00", "health": 67.09906494873553, "horizon_hours": 1}, {"timesta... | 0.7816995713607833 | 2024-03-10 23:30:00 | 40786a2cf3871e73 | 72 |
| 1 | 8 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": -... | 466.86304093794143 | [{"timestamp": "2024-03-21T01:00:00", "health": 62.73590233212773, "horizon_hours": 1}, {"timesta... | 0.960802119018195 | 2024-03-21 00:00:00 | a0e6de7e43b0ba0a | 72 |
| 1 | 9 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": -... | 397.47829073836965 | [{"timestamp": "2024-03-31T01:00:00", "health": 64.54373952848992, "horizon_hours": 1}, {"timesta... | 0.6615145873242813 | 2024-03-31 00:00:00 | ae1ce5312133f6da | 72 |
| 1 | 10 | ExponentialSmoothing_v2 | {"alpha": 0.3, "beta": 0.1, "failure_threshold": 70.0, "forecast_hours": 24, "estimated_trend": -... | 423.889066823529 | [{"timestamp": "2024-04-10T00:30:00", "health": 76.97899498260298, "horizon_hours": 1}, {"timesta... | 0.0 | 2024-04-09 23:30:00 | 5cd3f4dd15f187df | 72 |

### Bottom 10 Records

| EquipID | StateVersion | ModelType | ModelParamsJson | ResidualVariance | LastForecastHorizonJson | HazardBaseline | LastRetrainTime | TrainingDataHash | TrainingWindowHours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | 93 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.028896158489766698, "beta": 0.48132683658068764, "failure_threshold": 7... | 349.2604321117603 | [{"Timestamp": "2024-06-16T02:59:00", "ForecastHealth": 87.99047972211969, "CI_Lower": 0.03994375... | 1.4767120547224407e-25 | 2024-06-16 01:59:00 |  | 72 |
| 2621 | 92 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.01, "beta": 0.01, "failure_threshold": 70.0, "forecast_hours": 168, "es... | 659.0766987068479 | [{"Timestamp": "2024-01-14T03:59:00", "ForecastHealth": 54.81959831447349, "CI_Lower": 0.94870957... | 4.971001205472685e-26 | 2024-01-14 02:59:00 |  | 72 |
| 2621 | 91 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.0676343785300828, "beta": 0.01640218558215069, "failure_threshold": 70.... | 583.4111333243875 | [{"Timestamp": "2023-12-24T02:59:00", "ForecastHealth": 48.78904688601052, "CI_Lower": 2.43793128... | 7.398626821409059e-27 | 2023-12-24 01:59:00 |  | 72 |
| 2621 | 90 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.25855669585062807, "beta": 0.28023792276867054, "failure_threshold": 70... | 455.12087327891396 | [{"Timestamp": "2023-10-25T00:59:00", "ForecastHealth": 51.27261029195955, "CI_Lower": -0.7326889... | 7.120125104954122e-27 | 2023-10-24 23:59:00 |  | 72 |
| 2621 | 89 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.22007497905267193, "beta": 0.029401083108364676, "failure_threshold": 7... | 396.40128039034795 | [{"Timestamp": "2024-06-16T02:59:00", "ForecastHealth": 50.574125434498406, "CI_Lower": -1.646424... | 7.782539975836995e-27 | 2024-06-16 01:59:00 |  | 72 |
| 2621 | 88 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.06252122129667274, "beta": 0.5022264923117525, "failure_threshold": 70.... | 467.3103334489665 | [{"Timestamp": "2024-01-14T03:59:00", "ForecastHealth": 55.22713736394084, "CI_Lower": 1.01828156... | 5.6739349145918125e-27 | 2024-01-14 02:59:00 |  | 72 |
| 2621 | 87 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.03892160825491426, "beta": 0.02251092372624429, "failure_threshold": 70... | 578.2040014447064 | [{"Timestamp": "2023-12-03T00:59:00", "ForecastHealth": 59.85685902332158, "CI_Lower": 15.9714442... | 4.420728083703644e-27 | 2023-12-02 23:59:00 |  | 72 |
| 2621 | 86 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.25855669585062807, "beta": 0.28023792276867054, "failure_threshold": 70... | 455.12087327891396 | [{"Timestamp": "2023-10-25T00:59:00", "ForecastHealth": 51.27261029195955, "CI_Lower": -0.7326889... | 7.120125104954122e-27 | 2023-10-24 23:59:00 |  | 72 |
| 2621 | 85 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.22007497905267193, "beta": 0.029401083108364676, "failure_threshold": 7... | 396.40128039034795 | [{"Timestamp": "2024-06-16T02:59:00", "ForecastHealth": 50.574125434498406, "CI_Lower": -1.646424... | 7.782539975836995e-27 | 2024-06-16 01:59:00 |  | 72 |
| 2621 | 84 | ExponentialSmoothing_v2 | {"smoothing": {"alpha": 0.06252122129667274, "beta": 0.5022264923117525, "failure_threshold": 70.... | 467.3103334489665 | [{"Timestamp": "2024-01-14T03:59:00", "ForecastHealth": 55.22713736394084, "CI_Lower": 1.01828156... | 5.6739349145918125e-27 | 2024-01-14 02:59:00 |  | 72 |

---


## dbo.ACM_ForecastingState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| StateVersion | int | NO | 10 | — |
| ModelCoefficientsJson | nvarchar | YES | -1 | — |
| LastForecastJson | nvarchar | YES | -1 | — |
| LastRetrainTime | datetime2 | YES | — | — |
| TrainingDataHash | nvarchar | YES | 64 | — |
| DataVolumeAnalyzed | bigint | YES | 19 | — |
| RecentMAE | float | YES | 53 | — |
| RecentRMSE | float | YES | 53 | — |
| RetriggerReason | nvarchar | YES | 200 | — |
| RowVersion | timestamp | NO | — | — |
| CreatedAt | datetime2 | NO | — | (getdate()) |
| UpdatedAt | datetime2 | NO | — | (getdate()) |

---


## dbo.ACM_FusionQualityReport

**Primary Key:** No primary key  
**Row Count:** 150  
**Date Range:** 2025-12-05 11:36:59 to 2025-12-11 09:25:06  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Detector | nvarchar | NO | 64 | — |
| Weight | float | NO | 53 | — |
| Present | bit | NO | — | — |
| MeanZ | float | YES | 53 | — |
| MaxZ | float | YES | 53 | — |
| Points | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| RunID | EquipID | Detector | Weight | Present | MeanZ | MaxZ | Points | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Baseline Consistency (OMR) | 0.1 | True | 1.248983383178711 | 6.127109527587891 | 1443 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Correlation Break (PCA-SPE) | 0.2 | True | 1.9291510581970215 | 10.0 | 1443 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | cusum_z | 0.0 | True | 0.1701298952102661 | 1.7539159059524536 | 1443 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Density Anomaly (GMM) | 0.1 | True | 1.4463495016098022 | 10.0 | 1443 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Fused Multi-Detector | 0.0 | True | 0.0 | 3.4808783531188965 | 1443 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Multivariate Distance (Mahalanobis) | 0.2 | True | 2.268909215927124 | 10.0 | 1443 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Multivariate Outlier (PCA-T2) | 0.0 | True | 0.8361253142356873 | 10.0 | 1443 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Rare State (IsolationForest) | 0.2 | True | 0.8827471137046814 | 6.32260274887085 | 1443 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Streaming Anomaly (River) | 0.0 | False | 0.0 | 0.0 | 0 | 2025-12-11 09:23:20 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | Time-Series Anomaly (AR1) | 0.2 | True | 0.09133588522672653 | 10.0 | 1443 | 2025-12-11 09:23:20 |

### Bottom 10 Records

| RunID | EquipID | Detector | Weight | Present | MeanZ | MaxZ | Points | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | river_hst_z | 0.0 | False | 0.0 | 0.0 | 0 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_t2_z | 0.0 | True | 4.459890365600586 | 10.0 | 505 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | pca_spe_z | 0.2 | True | 0.928608238697052 | 10.0 | 505 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | omr_z | 0.1 | True | 0.50687575340271 | 7.973381042480469 | 505 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | mhal_z | 0.2 | True | 2.9545204639434814 | 10.0 | 505 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | iforest_z | 0.2 | True | 0.4153628349304199 | 2.1375653743743896 | 505 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | gmm_z | 0.1 | True | 0.8712297677993774 | 10.0 | 505 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | fused | 0.0 | True | -7.55385620720972e-09 | 2.5572545528411865 | 505 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | cusum_z | 0.0 | True | 0.06587760150432587 | 1.8989629745483398 | 505 | 2025-12-05 11:37:59 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | ar1_z | 0.2 | True | 1.93893301486969 | 10.0 | 505 | 2025-12-05 11:37:59 |

---


## dbo.ACM_HealthDistributionOverTime

**Primary Key:** No primary key  
**Row Count:** 6,356  
**Date Range:** 2025-12-05 11:36:59 to 2025-12-11 09:25:07  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| BucketStart | datetime2 | NO | — | — |
| BucketSeconds | int | NO | 10 | — |
| FusedP50 | float | YES | 53 | — |
| FusedP75 | float | YES | 53 | — |
| FusedP90 | float | YES | 53 | — |
| FusedP95 | float | YES | 53 | — |
| HealthP50 | float | YES | 53 | — |
| HealthP10 | float | YES | 53 | — |
| BucketCount | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| RunID | EquipID | BucketStart | BucketSeconds | FusedP50 | FusedP75 | FusedP90 | FusedP95 | HealthP50 | HealthP10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:21 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Bottom 10 Records

| RunID | EquipID | BucketStart | BucketSeconds | FusedP50 | FusedP75 | FusedP90 | FusedP95 | HealthP50 | HealthP10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:00 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

---


## dbo.ACM_HealthForecast

**Primary Key:** EquipID, RunID, Timestamp  
**Row Count:** 672  
**Date Range:** 2023-10-25 00:59:00 to 2024-06-23 01:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| Timestamp | datetime2 | NO | — | — |
| ForecastHealth | float | NO | 53 | — |
| CiLower | float | YES | 53 | — |
| CiUpper | float | YES | 53 | — |
| ForecastStd | float | YES | 53 | — |
| Method | nvarchar | NO | 50 | ('LinearTrend') |
| CreatedAt | datetime2 | NO | — | (getdate()) |
| RegimeLabel | int | YES | 10 | — |

### Top 10 Records

| EquipID | RunID | Timestamp | ForecastHealth | CiLower | CiUpper | ForecastStd | Method | CreatedAt | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 02:59:00 | 48.78904688601052 | 8.322316900707131 | 90.37985063803727 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 03:59:00 | 48.40947939663753 | 5.844388170483199 | 98.077533385637 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 04:59:00 | 48.02991190726454 | 0.36141129413125733 | 96.42369817481638 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 05:59:00 | 47.65034441789156 | 1.0140159474131158 | 97.64033352844228 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 06:59:00 | 47.27077692851857 | 1.5432910001051385 | 100.0 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 07:59:00 | 46.891209439145584 | 1.6257469519540675 | 98.87385541385507 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 08:59:00 | 46.511641949772596 | 2.5948179930257758 | 97.40174127978011 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 09:59:00 | 46.13207446039961 | 0.0 | 92.2069912610704 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 10:59:00 | 45.75250697102662 | 0.0 | 94.8553173609554 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2023-12-24 11:59:00 | 45.37293948165363 | 0.0 | 95.86716824893229 | 24.15390513611386 | ExponentialSmoothing | 2025-12-05 11:37:39 | NULL |

### Bottom 10 Records

| EquipID | RunID | Timestamp | ForecastHealth | CiLower | CiUpper | ForecastStd | Method | CreatedAt | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-21 02:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-21 01:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-21 00:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 23:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 22:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 21:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 20:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 19:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 18:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2024-01-20 17:59:00 | 0.0 | 0.0 | 100.0 | 25.672489141235367 | ExponentialSmoothing | 2025-12-05 11:38:03 | NULL |

---


## dbo.ACM_HealthForecast_Continuous

**Primary Key:** EquipID, Timestamp, SourceRunID  
**Row Count:** 3,988  
**Date Range:** 2023-10-20 00:30:00 to 2025-09-18 11:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| ForecastHealth | float | NO | 53 | — |
| CI_Lower | float | YES | 53 | — |
| CI_Upper | float | YES | 53 | — |
| SourceRunID | nvarchar | NO | 50 | — |
| MergeWeight | float | YES | 53 | — |
| EquipID | int | NO | 10 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

### Top 10 Records

| Timestamp | ForecastHealth | CI_Lower | CI_Upper | SourceRunID | MergeWeight | EquipID | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-20 00:30:00 | 84.9768817904151 | -0.48045198811352496 | 2.064029577668977 | dd25af55-769a-4469-81e2-e17330ebca91 | 1.0 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 00:30:00 | 84.9768817904151 | -0.48045198811352496 | 2.064029577668977 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1.0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 01:00:00 | 86.25933528270215 | -0.9778447615764474 | 2.1026223330226874 | dd25af55-769a-4469-81e2-e17330ebca91 | 1.0 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 01:00:00 | 86.25933528270215 | -0.9778447615764474 | 2.1026223330226874 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1.0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 01:30:00 | 87.54178877498921 | -1.2443956503777984 | 2.0560899013969336 | dd25af55-769a-4469-81e2-e17330ebca91 | 1.0 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 01:30:00 | 87.54178877498921 | -1.2443956503777984 | 2.0560899013969336 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1.0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 02:00:00 | 88.82424226727626 | -1.4000174725649281 | 1.9980649021789074 | dd25af55-769a-4469-81e2-e17330ebca91 | 1.0 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 02:00:00 | 88.82424226727626 | -1.4000174725649281 | 1.9980649021789074 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1.0 | 1 | 2025-12-09 17:02:51 |
| 2023-10-20 02:30:00 | 90.10669575956332 | -1.4951649922530508 | 1.9474206746916827 | dd25af55-769a-4469-81e2-e17330ebca91 | 1.0 | 1 | 2025-12-08 18:31:06 |
| 2023-10-20 02:30:00 | 90.10669575956332 | -1.4951649922530508 | 1.9474206746916827 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1.0 | 1 | 2025-12-09 17:02:51 |

### Bottom 10 Records

| Timestamp | ForecastHealth | CI_Lower | CI_Upper | SourceRunID | MergeWeight | EquipID | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-18 11:00:00 | 100.0 | 0.5152415089617066 | 2.7015606060899957 | 2ab00429-5144-41d0-ae47-2027fd52f9b0 | 1.0 | 1 | 2025-12-09 17:02:51 |
| 2025-09-18 11:00:00 | 100.0 | -4.1473511616068155 | 6.952605426773897 | 2fe3b906-69cd-4c7a-b29c-f67975652b2c | 1.0 | 1 | 2025-12-08 18:31:55 |
| 2025-09-18 11:00:00 | 100.0 | -4.147351161606815 | 6.952605426773898 | 3b1b1b1b-d9cc-47ad-9f7b-04ad1912c73e | 1.0 | 1 | 2025-12-08 18:27:45 |
| 2025-09-18 11:00:00 | 100.0 | -4.1473511616068155 | 6.952605426773897 | 453a03f2-a2f9-452d-bf48-4ac77b7387c3 | 1.0 | 1 | 2025-12-08 18:30:10 |
| 2025-09-18 11:00:00 | 100.0 | 0.5152415089617066 | 2.7015606060899957 | 5e804715-ffd9-445e-b56e-d9558903f514 | 1.0 | 1 | 2025-12-09 17:05:14 |
| 2025-09-18 11:00:00 | 100.0 | 0.5152415089617066 | 2.7015606060899957 | 6ff9aa5a-4ed4-4b53-844a-a95715c20f9e | 1.0 | 1 | 2025-12-09 17:04:07 |
| 2025-09-18 11:00:00 | 100.0 | 0.5152415089617066 | 2.7015606060899957 | 80633197-2d93-4cb9-a59e-d534018dc965 | 1.0 | 1 | 2025-12-09 17:03:26 |
| 2025-09-18 11:00:00 | 100.0 | 0.5152415089617066 | 2.7015606060899957 | 8fd5373f-292c-4565-9f68-b00efff3da31 | 1.0 | 1 | 2025-12-08 18:33:24 |
| 2025-09-18 11:00:00 | 100.0 | -4.1473511616068155 | 6.952605426773897 | b3392d12-fc73-4b71-9be9-d349f9137ef8 | 1.0 | 1 | 2025-12-08 18:32:23 |
| 2025-09-18 11:00:00 | 100.0 | 0.5152415089617066 | 2.7015606060899957 | c525a0a2-ebc0-44aa-ac37-48bf2c7ac95b | 1.0 | 1 | 2025-12-09 17:04:38 |

---


## dbo.ACM_HealthForecast_TS

**Primary Key:** RunID, EquipID, Timestamp  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| ForecastHealth | float | YES | 53 | — |
| CiLower | float | YES | 53 | — |
| CiUpper | float | YES | 53 | — |
| ForecastStd | float | YES | 53 | — |
| Method | nvarchar | NO | 50 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_HealthHistogram

**Primary Key:** No primary key  
**Row Count:** 150  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| HealthBin | nvarchar | NO | 50 | — |
| RecordCount | int | NO | 10 | — |
| Percentage | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| HealthBin | RecordCount | Percentage | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0-10 | 0 | 0.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |

### Bottom 10 Records

| HealthBin | RecordCount | Percentage | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0-10 | 0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |

---


## dbo.ACM_HealthTimeline

**Primary Key:** No primary key  
**Row Count:** 10,893  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| HealthIndex | float | NO | 53 | — |
| HealthZone | nvarchar | NO | 50 | — |
| FusedZ | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| RawHealthIndex | float | YES | 53 | — |
| QualityFlag | nvarchar | YES | 50 | — |

### Top 10 Records

| Timestamp | HealthIndex | HealthZone | FusedZ | RunID | EquipID | RawHealthIndex | QualityFlag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-18 00:00:00 | 86.24 | GOOD | 0.3995000123977661 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 86.23999786376953 | NORMAL |
| 2023-10-18 00:30:00 | 64.87 | ALERT | 2.380199909210205 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 15.0 | NORMAL |
| 2023-10-18 01:00:00 | 62.82 | ALERT | 0.8501999974250793 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 58.040000915527344 | NORMAL |
| 2023-10-18 01:30:00 | 72.25 | WATCH | 0.24719999730587006 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 94.23999786376953 | NORMAL |
| 2023-10-18 02:00:00 | 80.52 | WATCH | -0.04149999842047691 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 99.83000183105469 | NORMAL |
| 2023-10-18 02:30:00 | 83.39 | WATCH | -0.3321000039577484 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 90.06999969482422 | NORMAL |
| 2023-10-18 03:00:00 | 85.78 | GOOD | -0.3075000047683716 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 91.36000061035156 | NORMAL |
| 2023-10-18 03:30:00 | 89.36 | GOOD | -0.15299999713897705 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 97.70999908447266 | NORMAL |
| 2023-10-18 04:00:00 | 90.82 | GOOD | 0.24719999730587006 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 94.23999786376953 | NORMAL |
| 2023-10-18 04:30:00 | 85.62 | GOOD | -0.6008999943733215 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 73.47000122070312 | NORMAL |

### Bottom 10 Records

| Timestamp | HealthIndex | HealthZone | FusedZ | RunID | EquipID | RawHealthIndex | QualityFlag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | 91.98 | GOOD | -0.22179999947547913 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 95.30999755859375 | NORMAL |
| 2025-09-14 22:30:00 | 90.55 | GOOD | -0.18199999630451202 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 96.79000091552734 | NORMAL |
| 2025-09-14 22:00:00 | 87.88 | GOOD | -0.22050000727176666 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 95.36000061035156 | NORMAL |
| 2025-09-14 21:30:00 | 84.67 | WATCH | -0.23970000445842743 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 94.56999969482422 | NORMAL |
| 2025-09-14 21:00:00 | 80.43 | WATCH | -0.3824999928474426 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 87.23999786376953 | NORMAL |
| 2025-09-14 20:30:00 | 77.51 | WATCH | -0.413100004196167 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 85.41999816894531 | NORMAL |
| 2025-09-14 20:00:00 | 74.12 | WATCH | -0.5521000027656555 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 76.63999938964844 | NORMAL |
| 2025-09-14 19:30:00 | 73.03 | WATCH | -0.5964999794960022 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 73.75 | NORMAL |
| 2025-09-14 19:00:00 | 72.73 | WATCH | -0.5442000031471252 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 77.1500015258789 | NORMAL |
| 2025-09-14 18:30:00 | 70.83 | WATCH | -0.6365000009536743 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 71.16999816894531 | NORMAL |

---


## dbo.ACM_HealthZoneByPeriod

**Primary Key:** No primary key  
**Row Count:** 831  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 00:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| PeriodStart | datetime2 | NO | — | — |
| PeriodType | nvarchar | NO | 20 | — |
| HealthZone | nvarchar | NO | 50 | — |
| ZonePct | float | NO | 53 | — |
| ZoneCount | int | NO | 10 | — |
| TotalPoints | int | NO | 10 | — |
| Date | date | NO | — | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| PeriodStart | PeriodType | HealthZone | ZonePct | ZoneCount | TotalPoints | Date | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-18 00:00:00 | DAY | GOOD | 52.1 | 25 | 48 | 2023-10-18 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | DAY | WATCH | 6.2 | 3 | 48 | 2023-10-18 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:00:00 | DAY | ALERT | 41.7 | 20 | 48 | 2023-10-18 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-19 00:00:00 | DAY | GOOD | 31.2 | 15 | 48 | 2023-10-19 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-19 00:00:00 | DAY | WATCH | 25.0 | 12 | 48 | 2023-10-19 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-19 00:00:00 | DAY | ALERT | 43.8 | 21 | 48 | 2023-10-19 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-20 00:00:00 | DAY | GOOD | 100.0 | 1 | 1 | 2023-10-20 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-20 00:00:00 | DAY | WATCH | 0.0 | 0 | 1 | 2023-10-20 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-20 00:00:00 | DAY | ALERT | 0.0 | 0 | 1 | 2023-10-20 00:00:00 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-20 00:00:00 | DAY | GOOD | 0.0 | 0 | 1 | 2023-10-20 00:00:00 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |

### Bottom 10 Records

| PeriodStart | PeriodType | HealthZone | ZonePct | ZoneCount | TotalPoints | Date | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 00:00:00 | DAY | GOOD | 63.8 | 30 | 47 | 2025-09-14 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 00:00:00 | DAY | WATCH | 14.9 | 7 | 47 | 2025-09-14 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 00:00:00 | DAY | ALERT | 21.3 | 10 | 47 | 2025-09-14 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-13 00:00:00 | DAY | GOOD | 77.1 | 37 | 48 | 2025-09-13 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-13 00:00:00 | DAY | WATCH | 12.5 | 6 | 48 | 2025-09-13 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-13 00:00:00 | DAY | ALERT | 10.4 | 5 | 48 | 2025-09-13 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-12 00:00:00 | DAY | GOOD | 2.1 | 1 | 48 | 2025-09-12 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-12 00:00:00 | DAY | WATCH | 4.2 | 2 | 48 | 2025-09-12 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-12 00:00:00 | DAY | ALERT | 93.8 | 45 | 48 | 2025-09-12 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-11 00:00:00 | DAY | GOOD | 12.5 | 6 | 48 | 2025-09-11 00:00:00 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |

---


## dbo.ACM_HistorianData

**Primary Key:** DataID  
**Row Count:** 204,067  
**Date Range:** 2012-01-06 00:00:00 to 2020-12-01 23:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DataID | bigint | NO | 19 | — |
| EquipID | int | NO | 10 | — |
| TagName | varchar | NO | 255 | — |
| Timestamp | datetime2 | NO | — | — |
| Value | float | NO | 53 | — |
| Quality | tinyint | YES | 3 | ((192)) |
| CreatedAt | datetime2 | YES | — | (getutcdate()) |

### Top 10 Records

| DataID | EquipID | TagName | Timestamp | Value | Quality | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 204176 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 14:00:00 | 33.0 | 192 | 2025-12-01 06:22:21 |
| 204177 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 14:30:00 | 32.48 | 192 | 2025-12-01 06:22:21 |
| 204178 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 15:00:00 | 33.22 | 192 | 2025-12-01 06:22:21 |
| 204179 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 15:30:00 | 31.44 | 192 | 2025-12-01 06:22:21 |
| 204180 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 16:00:00 | 34.92 | 192 | 2025-12-01 06:22:21 |
| 204181 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 16:30:00 | 31.86 | 192 | 2025-12-01 06:22:21 |
| 204182 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 17:00:00 | 36.75 | 192 | 2025-12-01 06:22:21 |
| 204183 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 17:30:00 | 34.7 | 192 | 2025-12-01 06:22:21 |
| 204184 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 18:00:00 | 34.0 | 192 | 2025-12-01 06:22:21 |
| 204185 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position | 2012-05-21 18:30:00 | 31.72 | 192 | 2025-12-01 06:22:21 |

### Bottom 10 Records

| DataID | EquipID | TagName | Timestamp | Value | Quality | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 408350 | 2621 | LOTEMP1 | 2020-01-31 01:59:00 | 98.53143311 | 192 | 2025-12-01 06:22:44 |
| 408349 | 2621 | LOTEMP1 | 2020-01-31 00:59:00 | 98.5369339 | 192 | 2025-12-01 06:22:44 |
| 408348 | 2621 | LOTEMP1 | 2020-01-30 23:59:00 | 98.68974304 | 192 | 2025-12-01 06:22:44 |
| 408347 | 2621 | LOTEMP1 | 2020-01-30 22:59:00 | 98.48343658 | 192 | 2025-12-01 06:22:44 |
| 408346 | 2621 | LOTEMP1 | 2020-01-30 21:59:00 | 94.38490295 | 192 | 2025-12-01 06:22:44 |
| 408345 | 2621 | LOTEMP1 | 2020-01-30 20:59:00 | 128.8777618 | 192 | 2025-12-01 06:22:44 |
| 408344 | 2621 | LOTEMP1 | 2020-01-30 19:59:00 | 128.6075897 | 192 | 2025-12-01 06:22:44 |
| 408343 | 2621 | LOTEMP1 | 2020-01-30 18:59:00 | 128.4442139 | 192 | 2025-12-01 06:22:44 |
| 408342 | 2621 | LOTEMP1 | 2020-01-30 17:59:00 | 128.4984741 | 192 | 2025-12-01 06:22:44 |
| 408341 | 2621 | LOTEMP1 | 2020-01-30 16:59:00 | 128.4319153 | 192 | 2025-12-01 06:22:44 |

---


## dbo.ACM_MaintenanceRecommendation

**Primary Key:** RunID, EquipID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| EarliestMaintenance | datetime2 | NO | — | — |
| PreferredWindowStart | datetime2 | NO | — | — |
| PreferredWindowEnd | datetime2 | NO | — | — |
| FailureProbAtWindowEnd | float | NO | 53 | — |
| Comment | nvarchar | YES | 400 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_OMRContributionsLong

**Primary Key:** No primary key  
**Row Count:** 1,105,723  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| SensorName | nvarchar | NO | 128 | — |
| ContributionScore | float | NO | 53 | — |
| ContributionPct | float | NO | 53 | — |
| OMR_Z | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| RunID | EquipID | Timestamp | SensorName | ContributionScore | ContributionPct | OMR_Z | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_energy_1 | 1.166429789348353 | 6.956293788043815 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_energy_2 | 0.3229709864146161 | 1.9261177029519505 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_kurt | 0.31516966608553526 | 1.8795925913341796 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_mad | 0.025999562429771833 | 0.15505484879901787 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_mean | 0.019479558316686272 | 0.11617118471220161 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_med | 0.03290157284391242 | 0.19621670235192046 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_rz | 0.16742365648319865 | 0.9984725632018606 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_skew | 0.03414101057110392 | 0.20360839711234502 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_slope | 0.1453613866389481 | 0.8668987367536813 | -0.7553902864456177 | 2025-12-11 09:23:16 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position_std | 0.002008943652857117 | 0.011980834492153434 | -0.7553902864456177 | 2025-12-11 09:23:16 |

### Bottom 10 Records

| RunID | EquipID | Timestamp | SensorName | ContributionScore | ContributionPct | OMR_Z | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_std | 0.17044633479889648 | 0.4960372680163778 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_slope | 0.0013155264029286093 | 0.0038284784690855366 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_skew | 0.00010789697167520713 | 0.00031400451714117325 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_rz | 0.0017086255372970975 | 0.004972485589577812 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_med | 0.22025790532560205 | 0.6410001702038887 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_mean | 0.030887633596548548 | 0.08988997858358186 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_mad | 0.12156761158070233 | 0.35378916184332154 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_kurt | 0.013531764130296895 | 0.03938048488137948 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_energy_2 | 0.003374605663188981 | 0.009820863415900782 | 0.09875394403934479 | 2025-12-05 11:37:56 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | TURBAXDISP2_energy_1 | 0.1843347239067483 | 0.5364556119974057 | 0.09875394403934479 | 2025-12-05 11:37:56 |

---


## dbo.ACM_OMRTimeline

**Primary Key:** No primary key  
**Row Count:** 10,893  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| OMR_Z | float | YES | 53 | — |
| OMR_Weight | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| RunID | EquipID | Timestamp | OMR_Z | OMR_Weight | CreatedAt |
| --- | --- | --- | --- | --- | --- |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 10:30:00 | -0.7554000020027161 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 11:00:00 | 2.3440001010894775 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 11:30:00 | 3.7219998836517334 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 12:00:00 | 6.127099990844727 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 12:30:00 | 6.127099990844727 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 13:00:00 | 6.127099990844727 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 13:30:00 | 5.552499771118164 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 14:00:00 | 6.127099990844727 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 14:30:00 | 6.127099990844727 | 0.1 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-08-25 15:00:00 | 6.127099990844727 | 0.1 | 2025-12-11 09:23:21 |

### Bottom 10 Records

| RunID | EquipID | Timestamp | OMR_Z | OMR_Weight | CreatedAt |
| --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 02:59:00 | 0.09880000352859497 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 01:59:00 | 0.48590001463890076 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-14 00:59:00 | 2.05679988861084 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-13 23:59:00 | 2.9767000675201416 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-13 22:59:00 | 4.160200119018555 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-13 21:59:00 | 0.17919999361038208 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-13 20:59:00 | 0.9797000288963318 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-13 19:59:00 | 2.604300022125244 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-13 18:59:00 | 1.9293999671936035 | 0.1 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-13 17:59:00 | 1.1894999742507935 | 0.1 | 2025-12-05 11:38:00 |

---


## dbo.ACM_OMR_Diagnostics

**Primary Key:** DiagnosticID  
**Row Count:** 15  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DiagnosticID | int | NO | 10 | — |
| RunID | varchar | NO | 100 | — |
| EquipID | int | NO | 10 | — |
| ModelType | varchar | NO | 20 | — |
| NComponents | int | NO | 10 | — |
| TrainSamples | int | NO | 10 | — |
| TrainFeatures | int | NO | 10 | — |
| TrainResidualStd | float | NO | 53 | — |
| TrainStartTime | datetime2 | YES | — | — |
| TrainEndTime | datetime2 | YES | — | — |
| CalibrationStatus | varchar | NO | 20 | — |
| SaturationRate | float | YES | 53 | — |
| FusionWeight | float | YES | 53 | — |
| FitTimestamp | datetime2 | NO | — | (getdate()) |
| CreatedAt | datetime2 | NO | — | (getdate()) |

### Top 10 Records

| DiagnosticID | RunID | EquipID | ModelType | NComponents | TrainSamples | TrainFeatures | TrainResidualStd | TrainStartTime | TrainEndTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 928 | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2621 | pca | 5 | 97 | 149 | 1.1695945553041047 | NULL | NULL |
| 930 | bf973ffa-8011-45b8-b155-7f6aee79667f | 2621 | pls | 5 | 841 | 160 | 1.800617704312548 | NULL | NULL |
| 931 | 17af291b-3a84-456f-bde8-f424528c797d | 2621 | pls | 5 | 300 | 160 | 2.758045052312038 | NULL | NULL |
| 932 | e4cf6f6a-210e-4527-a300-b51e78840353 | 2621 | pls | 5 | 361 | 160 | 1.9473207774521042 | NULL | NULL |
| 1101 | 91c95c14-74b8-43a1-892d-cc11bbebc7cf | 1 | pls | 5 | 97 | 90 | 1.1910469897232634 | NULL | NULL |
| 1102 | 1c1aea19-068c-4140-902e-206c202ec225 | 1 | pls | 5 | 503 | 90 | 1.4801945775713412 | NULL | NULL |
| 1103 | 45da4afc-a1ff-48bb-b696-9855d1d43b59 | 1 | pls | 5 | 399 | 90 | 1.2042133753658206 | NULL | NULL |
| 1104 | abc66545-7031-429f-82bc-c1f25b7850bc | 1 | pls | 5 | 1411 | 90 | 1.4179095471967185 | NULL | NULL |
| 1105 | 7d32992a-c9cf-4f13-a8f4-59ee346fc4e2 | 1 | pls | 5 | 1353 | 90 | 1.3907038142223689 | NULL | NULL |
| 1106 | dd26755a-6ae3-4ffc-a2bc-063332340eb9 | 1 | pls | 5 | 1442 | 90 | 1.3135110704386805 | NULL | NULL |

### Bottom 10 Records

| DiagnosticID | RunID | EquipID | ModelType | NComponents | TrainSamples | TrainFeatures | TrainResidualStd | TrainStartTime | TrainEndTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1111 | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 1 | pls | 5 | 358 | 90 | 1.150854869215602 | NULL | NULL |
| 1110 | d855f3d7-b588-4bf3-9dce-97da7a53c6a5 | 1 | pls | 5 | 300 | 90 | 1.102313001254648 | NULL | NULL |
| 1109 | 7131196b-b651-4cd8-865c-d7ee8f740882 | 1 | pls | 5 | 440 | 90 | 1.1084013253713985 | NULL | NULL |
| 1108 | 4bd3930b-1ca2-41e4-ad72-0c2d772fbced | 1 | pls | 5 | 1257 | 90 | 1.1034498041301413 | NULL | NULL |
| 1107 | 3c73280a-9598-4410-bb08-e1da79fb0675 | 1 | pls | 5 | 1345 | 90 | 1.3673997188723672 | NULL | NULL |
| 1106 | dd26755a-6ae3-4ffc-a2bc-063332340eb9 | 1 | pls | 5 | 1442 | 90 | 1.3135110704386805 | NULL | NULL |
| 1105 | 7d32992a-c9cf-4f13-a8f4-59ee346fc4e2 | 1 | pls | 5 | 1353 | 90 | 1.3907038142223689 | NULL | NULL |
| 1104 | abc66545-7031-429f-82bc-c1f25b7850bc | 1 | pls | 5 | 1411 | 90 | 1.4179095471967185 | NULL | NULL |
| 1103 | 45da4afc-a1ff-48bb-b696-9855d1d43b59 | 1 | pls | 5 | 399 | 90 | 1.2042133753658206 | NULL | NULL |
| 1102 | 1c1aea19-068c-4140-902e-206c202ec225 | 1 | pls | 5 | 503 | 90 | 1.4801945775713412 | NULL | NULL |

---


## dbo.ACM_PCA_Loadings

**Primary Key:** RecordID  
**Row Count:** 8,140  
**Date Range:** 2025-12-05 11:37:03 to 2025-12-11 09:25:08  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RecordID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| EntryDateTime | datetime2 | NO | — | — |
| ComponentNo | int | NO | 10 | — |
| ComponentID | int | YES | 10 | — |
| Sensor | nvarchar | NO | 200 | — |
| FeatureName | nvarchar | YES | 200 | — |
| Loading | float | NO | 53 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

### Top 10 Records

| RecordID | RunID | EquipID | EntryDateTime | ComponentNo | ComponentID | Sensor | FeatureName | Loading | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 117666 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | ACTTBTEMP1_med | ACTTBTEMP1_med | 0.1528612013419846 | 2025-12-05 11:37:03 |
| 117667 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B1RADVIBX_med | B1RADVIBX_med | 0.12019190743751362 | 2025-12-05 11:37:03 |
| 117668 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B1RADVIBY_med | B1RADVIBY_med | 0.0485883730784335 | 2025-12-05 11:37:03 |
| 117669 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B1TEMP1_med | B1TEMP1_med | 0.15384795524198647 | 2025-12-05 11:37:03 |
| 117670 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B1VIB1_med | B1VIB1_med | 0.01690025652379255 | 2025-12-05 11:37:03 |
| 117671 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B1VIB2_med | B1VIB2_med | 0.0359477007271804 | 2025-12-05 11:37:03 |
| 117672 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B2RADVIBX_med | B2RADVIBX_med | 0.023622930395621453 | 2025-12-05 11:37:03 |
| 117673 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B2RADVIBY_med | B2RADVIBY_med | 0.003446477533364396 | 2025-12-05 11:37:03 |
| 117674 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B2TEMP1_med | B2TEMP1_med | 0.15335474197378904 | 2025-12-05 11:37:03 |
| 117675 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 1 | 1 | B2VIB1_med | B2VIB1_med | 0.07572701458877493 | 2025-12-05 11:37:03 |

### Bottom 10 Records

| RecordID | RunID | EquipID | EntryDateTime | ComponentNo | ComponentID | Sensor | FeatureName | Loading | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 202305 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 0.06741481457357686 | 2025-12-11 09:25:08 |
| 202304 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | 0.067031575284901 | 2025-12-11 09:25:08 |
| 202303 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | 0.07320664119385696 | 2025-12-11 09:25:08 |
| 202302 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | 0.06961937382880672 | 2025-12-11 09:25:08 |
| 202301 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_rz | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_rz | 0.11126882687781409 | 2025-12-11 09:25:08 |
| 202300 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.06T31_1FD Fan Inlet Temperature_rz | DEMO.SIM.06T31_1FD Fan Inlet Temperature_rz | 0.08030642165149311 | 2025-12-11 09:25:08 |
| 202299 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.06I03_1FD Fan Motor Current_rz | DEMO.SIM.06I03_1FD Fan Motor Current_rz | 0.07381666054268814 | 2025-12-11 09:25:08 |
| 202298 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_rz | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_rz | 0.05105257525349337 | 2025-12-11 09:25:08 |
| 202297 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.06G31_1FD Fan Damper Position_rz | DEMO.SIM.06G31_1FD Fan Damper Position_rz | 0.07086780323728355 | 2025-12-11 09:25:08 |
| 202296 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | 5 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_energy_2 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_energy_2 | -0.054464451833280254 | 2025-12-11 09:25:08 |

---


## dbo.ACM_PCA_Metrics

**Primary Key:** RunID, EquipID, ComponentName, MetricType  
**Row Count:** 45  
**Date Range:** 2025-12-05 11:36:52 to 2025-12-11 09:25:01  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | nvarchar | NO | 72 | — |
| EquipID | int | NO | 10 | — |
| ComponentName | nvarchar | NO | 200 | — |
| MetricType | nvarchar | NO | 50 | — |
| Value | float | NO | 53 | — |
| Timestamp | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| RunID | EquipID | ComponentName | MetricType | Value | Timestamp |
| --- | --- | --- | --- | --- | --- |
| 17af291b-3a84-456f-bde8-f424528c797d | 2621 | PCA | n_components | 5.0 | 2025-12-05 11:37:51 |
| 17af291b-3a84-456f-bde8-f424528c797d | 2621 | PCA | n_features | 160.0 | 2025-12-05 11:37:51 |
| 17af291b-3a84-456f-bde8-f424528c797d | 2621 | PCA | variance_explained | 0.7297713273176499 | 2025-12-05 11:37:51 |
| 1c1aea19-068c-4140-902e-206c202ec225 | 1 | PCA | n_components | 5.0 | 2025-12-11 09:21:27 |
| 1c1aea19-068c-4140-902e-206c202ec225 | 1 | PCA | n_features | 90.0 | 2025-12-11 09:21:27 |
| 1c1aea19-068c-4140-902e-206c202ec225 | 1 | PCA | variance_explained | 0.6121461283051094 | 2025-12-11 09:21:27 |
| 3c73280a-9598-4410-bb08-e1da79fb0675 | 1 | PCA | n_components | 5.0 | 2025-12-11 09:23:42 |
| 3c73280a-9598-4410-bb08-e1da79fb0675 | 1 | PCA | n_features | 90.0 | 2025-12-11 09:23:42 |
| 3c73280a-9598-4410-bb08-e1da79fb0675 | 1 | PCA | variance_explained | 0.6736403055879454 | 2025-12-11 09:23:42 |
| 45da4afc-a1ff-48bb-b696-9855d1d43b59 | 1 | PCA | n_components | 5.0 | 2025-12-11 09:21:42 |

### Bottom 10 Records

| RunID | EquipID | ComponentName | MetricType | Value | Timestamp |
| --- | --- | --- | --- | --- | --- |
| e4cf6f6a-210e-4527-a300-b51e78840353 | 2621 | PCA | variance_explained | 0.8095708647775336 | 2025-12-05 11:38:18 |
| e4cf6f6a-210e-4527-a300-b51e78840353 | 2621 | PCA | n_features | 160.0 | 2025-12-05 11:38:18 |
| e4cf6f6a-210e-4527-a300-b51e78840353 | 2621 | PCA | n_components | 5.0 | 2025-12-05 11:38:18 |
| dd26755a-6ae3-4ffc-a2bc-063332340eb9 | 1 | PCA | variance_explained | 0.5914009961403585 | 2025-12-11 09:23:11 |
| dd26755a-6ae3-4ffc-a2bc-063332340eb9 | 1 | PCA | n_features | 90.0 | 2025-12-11 09:23:11 |
| dd26755a-6ae3-4ffc-a2bc-063332340eb9 | 1 | PCA | n_components | 5.0 | 2025-12-11 09:23:11 |
| d855f3d7-b588-4bf3-9dce-97da7a53c6a5 | 1 | PCA | variance_explained | 0.7896926933891086 | 2025-12-11 09:24:46 |
| d855f3d7-b588-4bf3-9dce-97da7a53c6a5 | 1 | PCA | n_features | 90.0 | 2025-12-11 09:24:46 |
| d855f3d7-b588-4bf3-9dce-97da7a53c6a5 | 1 | PCA | n_components | 5.0 | 2025-12-11 09:24:46 |
| bf973ffa-8011-45b8-b155-7f6aee79667f | 2621 | PCA | variance_explained | 0.8036143026022342 | 2025-12-05 11:37:23 |

---


## dbo.ACM_PCA_Models

**Primary Key:** RecordID  
**Row Count:** 15  
**Date Range:** 2025-12-05 11:37:03 to 2025-12-11 09:25:08  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RecordID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| EntryDateTime | datetime2 | NO | — | — |
| NComponents | int | YES | 10 | — |
| TargetVar | nvarchar | YES | -1 | — |
| VarExplainedJSON | nvarchar | YES | -1 | — |
| ScalingSpecJSON | nvarchar | YES | -1 | — |
| ModelVersion | nvarchar | YES | 50 | — |
| TrainStartEntryDateTime | datetime2 | YES | — | — |
| TrainEndEntryDateTime | datetime2 | YES | — | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

### Top 10 Records

| RecordID | RunID | EquipID | EntryDateTime | NComponents | TargetVar | VarExplainedJSON | ScalingSpecJSON | ModelVersion | TrainStartEntryDateTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 239 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2025-12-05 11:37:03 | 5 | {"SPE_P95_train": 5.269982814788818, "T2_P95_train": 1.9085338115692139} | [0.222871101735632, 0.1413781846798076, 0.10431687511998361, 0.0787824354517283, 0.05857872020549... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v5.0.0 | 2023-10-20 23:59:00 |
| 241 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2025-12-05 11:37:40 | 5 | {"SPE_P95_train": 4.956071853637695, "T2_P95_train": 10.0} | [0.3616785252426999, 0.1950990807811145, 0.11830795470290596, 0.08124325949967932, 0.047285482375... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v5.0.0 | 2023-10-15 00:00:00 |
| 242 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2025-12-05 11:38:04 | 5 | {"SPE_P95_train": 3.1673319339752197, "T2_P95_train": 10.0} | [0.3199685537685077, 0.17224387997154395, 0.11025780777284727, 0.084562167470981, 0.0427389183337... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v5.0.0 | 2023-12-24 02:59:00 |
| 243 | E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 | 2025-12-05 11:38:28 | 5 | {"SPE_P95_train": 3.9742472171783447, "T2_P95_train": 5.920322895050049} | [0.28356089779813354, 0.2418748134351031, 0.12315339569614002, 0.10601016050827235, 0.05497159733... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v5.0.0 | 2024-05-17 00:00:00 |
| 415 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 2025-12-11 09:21:17 | 5 | {"SPE_P95_train": 3.4016430377960205, "T2_P95_train": 2.116558790206909} | [0.27989694184187136, 0.23426346832848857, 0.10448614396646073, 0.07837094925275088, 0.0711203784... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-18 00:00:00 |
| 416 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 2025-12-11 09:21:35 | 5 | {"SPE_P95_train": 6.432009220123291, "T2_P95_train": 6.628495216369629} | [0.19648372194546906, 0.1392822602032956, 0.1151769829100149, 0.08811291055681245, 0.073090252689... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 417 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | 2025-12-11 09:21:50 | 5 | {"SPE_P95_train": 4.5547966957092285, "T2_P95_train": 6.338170051574707} | [0.21917876968606634, 0.16336473949400046, 0.11099733017459443, 0.0840586480193309, 0.07309745075... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-01-14 00:00:00 |
| 418 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | 2025-12-11 09:22:22 | 5 | {"SPE_P95_train": 4.653730392456055, "T2_P95_train": 5.959028244018555} | [0.2103212304457447, 0.13617566774773296, 0.10502305169124686, 0.0796131033956107, 0.057323346042... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-03-03 05:00:00 |
| 419 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 2025-12-11 09:22:54 | 5 | {"SPE_P95_train": 4.577191352844238, "T2_P95_train": 6.197824954986572} | [0.24883393109296878, 0.15133635805194803, 0.09896181910096219, 0.0893060449216775, 0.05513775114... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-05-16 00:00:00 |
| 420 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:24 | 5 | {"SPE_P95_train": 5.9401044845581055, "T2_P95_train": 3.519364356994629} | [0.1743959478541592, 0.14019479914465502, 0.11907144814200503, 0.08626290960671808, 0.07147589139... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-07-21 09:30:00 |

### Bottom 10 Records

| RecordID | RunID | EquipID | EntryDateTime | NComponents | TargetVar | VarExplainedJSON | ScalingSpecJSON | ModelVersion | TrainStartEntryDateTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 425 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-12-11 09:25:08 | 5 | {"SPE_P95_train": 10.0, "T2_P95_train": 10.0} | [0.2949383153247925, 0.2231758226456243, 0.12287701265121986, 0.0940732544318872, 0.0658200078176... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2025-07-11 00:00:00 |
| 424 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2025-12-11 09:24:54 | 5 | {"SPE_P95_train": 10.0, "T2_P95_train": 10.0} | [0.28564384224036343, 0.2035041296043215, 0.13817488126754873, 0.08999943919055246, 0.07237040108... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2025-05-11 00:00:00 |
| 423 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 2025-12-11 09:24:38 | 5 | {"SPE_P95_train": 5.296046257019043, "T2_P95_train": 3.0513722896575928} | [0.25895994462251287, 0.15694299404248258, 0.09508938897882706, 0.06609787238465455, 0.0547718188... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2025-02-20 00:00:00 |
| 422 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 | 2025-12-11 09:24:23 | 5 | {"SPE_P95_train": 10.0, "T2_P95_train": 10.0} | [0.257098890634532, 0.1537266568480462, 0.1000354433479626, 0.06413416279109337, 0.05533649489393... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-12-08 14:30:00 |
| 421 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | 2025-12-11 09:23:55 | 5 | {"SPE_P95_train": 8.665353775024414, "T2_P95_train": 6.468105792999268} | [0.2661319084562329, 0.20772833440525637, 0.09955019958184394, 0.05542432747664752, 0.04480553566... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-09-29 12:00:00 |
| 420 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2025-12-11 09:23:24 | 5 | {"SPE_P95_train": 5.9401044845581055, "T2_P95_train": 3.519364356994629} | [0.1743959478541592, 0.14019479914465502, 0.11907144814200503, 0.08626290960671808, 0.07147589139... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-07-21 09:30:00 |
| 419 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 2025-12-11 09:22:54 | 5 | {"SPE_P95_train": 4.577191352844238, "T2_P95_train": 6.197824954986572} | [0.24883393109296878, 0.15133635805194803, 0.09896181910096219, 0.0893060449216775, 0.05513775114... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-05-16 00:00:00 |
| 418 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | 2025-12-11 09:22:22 | 5 | {"SPE_P95_train": 4.653730392456055, "T2_P95_train": 5.959028244018555} | [0.2103212304457447, 0.13617566774773296, 0.10502305169124686, 0.0796131033956107, 0.057323346042... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-03-03 05:00:00 |
| 417 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | 2025-12-11 09:21:50 | 5 | {"SPE_P95_train": 4.5547966957092285, "T2_P95_train": 6.338170051574707} | [0.21917876968606634, 0.16336473949400046, 0.11099733017459443, 0.0840586480193309, 0.07309745075... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-01-14 00:00:00 |
| 416 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 2025-12-11 09:21:35 | 5 | {"SPE_P95_train": 6.432009220123291, "T2_P95_train": 6.628495216369629} | [0.19648372194546906, 0.1392822602032956, 0.1151769829100149, 0.08811291055681245, 0.073090252689... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |

---


## dbo.ACM_RUL

**Primary Key:** EquipID, RunID  
**Row Count:** 4  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| RUL_Hours | float | NO | 53 | — |
| P10_LowerBound | float | YES | 53 | — |
| P50_Median | float | YES | 53 | — |
| P90_UpperBound | float | YES | 53 | — |
| Confidence | float | YES | 53 | — |
| FailureTime | datetime2 | YES | — | — |
| Method | nvarchar | NO | 50 | ('MonteCarlo') |
| NumSimulations | int | YES | 10 | — |
| TopSensor1 | nvarchar | YES | 255 | — |
| TopSensor2 | nvarchar | YES | 255 | — |
| TopSensor3 | nvarchar | YES | 255 | — |
| CreatedAt | datetime2 | NO | — | (getdate()) |
| DriftZ | float | YES | 53 | — |
| CurrentRegime | int | YES | 10 | — |
| RegimeState | nvarchar | YES | 32 | — |
| OMR_Z | float | YES | 53 | — |

### Top 10 Records

| EquipID | RunID | RUL_Hours | P10_LowerBound | P50_Median | P90_UpperBound | Confidence | FailureTime | Method | NumSimulations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 1.0 | NULL | NULL | NULL | 0.8 | NULL | ExponentialSmoothingProbabilistic | NULL |
| 2621 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 1.0 | NULL | NULL | NULL | 0.6 | NULL | ExponentialSmoothingProbabilistic | NULL |
| 2621 | E4CF6F6A-210E-4527-A300-B51E78840353 | 3.0 | NULL | NULL | NULL | 0.6 | NULL | ExponentialSmoothingProbabilistic | NULL |
| 2621 | 17AF291B-3A84-456F-BDE8-F424528C797D | 1.0 | NULL | NULL | NULL | 0.6 | NULL | ExponentialSmoothingProbabilistic | NULL |

---


## dbo.ACM_RUL_Attribution

**Primary Key:** RunID, EquipID, FailureTime, SensorName  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| FailureTime | datetime2 | NO | — | — |
| SensorName | nvarchar | NO | 255 | — |
| FailureContribution | float | NO | 53 | — |
| ZScoreAtFailure | float | YES | 53 | — |
| AlertCount | int | YES | 10 | — |
| Comment | nvarchar | YES | 400 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_RUL_ByRegime

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| RunID | nvarchar | NO | 64 | — |
| RegimeLabel | int | NO | 10 | — |
| RegimeState | nvarchar | YES | 32 | — |
| RUL_Hours | float | NO | 53 | — |
| P10_LowerBound | float | YES | 53 | — |
| P50_Median | float | YES | 53 | — |
| P90_UpperBound | float | YES | 53 | — |
| FailureProbability | float | YES | 53 | — |
| DegradationRate | float | YES | 53 | — |
| TimeInRegime_Hours | float | YES | 53 | — |
| Confidence | float | YES | 53 | — |
| Method | nvarchar | YES | 64 | ('RegimeConditioned') |
| CreatedAt | datetime2 | YES | — | (getdate()) |

---


## dbo.ACM_RUL_LearningState

**Primary Key:** EquipID  
**Row Count:** 2  
**Date Range:** 2025-12-02 12:27:29 to 2025-12-03 14:45:15  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| AR1_MAE | float | YES | 53 | — |
| AR1_RMSE | float | YES | 53 | — |
| AR1_Bias | float | YES | 53 | — |
| AR1_RecentErrors | nvarchar | YES | -1 | — |
| AR1_Weight | float | YES | 53 | — |
| Exp_MAE | float | YES | 53 | — |
| Exp_RMSE | float | YES | 53 | — |
| Exp_Bias | float | YES | 53 | — |
| Exp_RecentErrors | nvarchar | YES | -1 | — |
| Exp_Weight | float | YES | 53 | — |
| Weibull_MAE | float | YES | 53 | — |
| Weibull_RMSE | float | YES | 53 | — |
| Weibull_Bias | float | YES | 53 | — |
| Weibull_RecentErrors | nvarchar | YES | -1 | — |
| Weibull_Weight | float | YES | 53 | — |
| CalibrationFactor | float | YES | 53 | — |
| LastUpdated | datetime2 | YES | — | — |
| PredictionHistory | nvarchar | YES | -1 | — |

### Top 10 Records

| EquipID | AR1_MAE | AR1_RMSE | AR1_Bias | AR1_RecentErrors | AR1_Weight | Exp_MAE | Exp_RMSE | Exp_Bias | Exp_RecentErrors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0 | 0.0 | 0.0 | [] | 1.0 | 0.0 | 0.0 | 0.0 | [] |
| 2621 | 0.0 | 0.0 | 0.0 | [] | 1.0 | 0.0 | 0.0 | 0.0 | [] |

---


## dbo.ACM_RUL_Summary

**Primary Key:** RunID, EquipID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| RUL_Hours | float | NO | 53 | — |
| LowerBound | float | YES | 53 | — |
| UpperBound | float | YES | 53 | — |
| Confidence | float | YES | 53 | — |
| Method | nvarchar | NO | 50 | — |
| LastUpdate | datetime2 | NO | — | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |
| RUL_Trajectory_Hours | float | YES | 53 | — |
| RUL_Hazard_Hours | float | YES | 53 | — |
| RUL_Energy_Hours | float | YES | 53 | — |
| RUL_Final_Hours | float | YES | 53 | — |
| ConfidenceBand_Hours | float | YES | 53 | — |
| DominantPath | nvarchar | YES | 20 | — |

---


## dbo.ACM_RUL_TS

**Primary Key:** RunID, EquipID, Timestamp  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| RUL_Hours | float | NO | 53 | — |
| LowerBound | float | YES | 53 | — |
| UpperBound | float | YES | 53 | — |
| Confidence | float | YES | 53 | — |
| Method | nvarchar | NO | 50 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_RecommendedActions

**Primary Key:** RunID, EquipID, Action  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Action | nvarchar | NO | 400 | — |
| Priority | nvarchar | YES | 50 | — |
| EstimatedDuration_Hours | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_RefitRequests

**Primary Key:** RequestID  
**Row Count:** 1,371  
**Date Range:** 2025-12-01 05:03:49 to 2025-12-11 03:55:04  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RequestID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| RequestedAt | datetime2 | NO | — | (sysutcdatetime()) |
| Reason | nvarchar | YES | -1 | — |
| AnomalyRate | float | YES | 53 | — |
| DriftScore | float | YES | 53 | — |
| ModelAgeHours | float | YES | 53 | — |
| RegimeQuality | float | YES | 53 | — |
| Acknowledged | bit | NO | — | ((0)) |
| AcknowledgedAt | datetime2 | YES | — | — |

### Top 10 Records

| RequestID | EquipID | RequestedAt | Reason | AnomalyRate | DriftScore | ModelAgeHours | RegimeQuality | Acknowledged | AcknowledgedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 569 | 1 | 2025-12-01 05:03:49 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:07:55 |
| 570 | 1 | 2025-12-01 05:07:59 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:09:40 |
| 571 | 1 | 2025-12-01 05:09:43 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:14:58 |
| 572 | 1 | 2025-12-01 05:15:02 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:15:06 |
| 573 | 1 | 2025-12-01 05:15:10 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:15:14 |
| 574 | 1 | 2025-12-01 05:15:17 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:15:22 |
| 575 | 1 | 2025-12-01 05:15:25 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:15:29 |
| 576 | 1 | 2025-12-01 05:15:32 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:15:36 |
| 577 | 1 | 2025-12-01 05:15:40 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:15:44 |
| 578 | 1 | 2025-12-01 05:15:47 | Anomaly rate too low; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-01 05:15:51 |

### Bottom 10 Records

| RequestID | EquipID | RequestedAt | Reason | AnomalyRate | DriftScore | ModelAgeHours | RegimeQuality | Acknowledged | AcknowledgedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1939 | 1 | 2025-12-11 03:55:04 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 1938 | 1 | 2025-12-11 03:54:49 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:55:01 |
| 1937 | 1 | 2025-12-11 03:54:34 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:54:46 |
| 1936 | 1 | 2025-12-11 03:54:15 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:54:31 |
| 1935 | 1 | 2025-12-11 03:53:45 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:54:11 |
| 1934 | 1 | 2025-12-11 03:53:14 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:53:42 |
| 1933 | 1 | 2025-12-11 03:52:42 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:53:11 |
| 1932 | 1 | 2025-12-11 03:52:12 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:52:37 |
| 1931 | 1 | 2025-12-11 03:51:46 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:52:08 |
| 1930 | 1 | 2025-12-11 03:51:30 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-11 03:51:42 |

---


## dbo.ACM_RegimeDwellStats

**Primary Key:** No primary key  
**Row Count:** 66  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RegimeLabel | nvarchar | NO | 50 | — |
| Runs | int | NO | 10 | — |
| MeanSeconds | float | NO | 53 | — |
| MedianSeconds | float | NO | 53 | — |
| MinSeconds | float | NO | 53 | — |
| MaxSeconds | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| RegimeLabel | Runs | MeanSeconds | MedianSeconds | MinSeconds | MaxSeconds | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 27 | 86666.66666666667 | 25200.0 | 1800.0 | 736200.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 1 | 26 | 22846.153846153848 | 21600.0 | 1800.0 | 73800.0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0 | 4 | 697050.0 | 145800.0 | 19800.0 | 2476800.0 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| 1 | 3 | 22800.0 | 23400.0 | 21600.0 | 23400.0 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| 0 | 9 | 5200.0 | 5400.0 | 1800.0 | 9000.0 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 1 | 10 | 23580.0 | 18000.0 | 5400.0 | 81000.0 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 2 | 14 | 174214.2857142857 | 25200.0 | 3600.0 | 2044800.0 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 3 | 2 | 12600.0 | 12600.0 | 9000.0 | 16200.0 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 1 | 42 | 27000.0 | 18000.0 | 1800.0 | 462600.0 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 |
| 0 | 41 | 37800.0 | 32400.0 | 3600.0 | 100800.0 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 |

### Bottom 10 Records

| RegimeLabel | Runs | MeanSeconds | MedianSeconds | MinSeconds | MaxSeconds | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 172800.0 | 129600.0 | 7200.0 | 450000.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 1 | 7 | 21600.0 | 21600.0 | 10800.0 | 32400.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 2 | 7 | 24685.714285714286 | 21600.0 | 21600.0 | 32400.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 3 | 9 | 14800.0 | 18000.0 | 0.0 | 25200.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 4 | 6 | 43200.0 | 30600.0 | 7200.0 | 93600.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 5 | 5 | 19440.0 | 18000.0 | 10800.0 | 36000.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0 | 5 | 471960.0 | 356400.0 | 237600.0 | 950400.0 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| 1 | 4 | 12600.0 | 9900.0 | 5400.0 | 25200.0 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| 0 | 7 | 9257.142857142857 | 9000.0 | 1800.0 | 19800.0 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 |
| 1 | 10 | 37980.0 | 38700.0 | 14400.0 | 64800.0 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 |

---


## dbo.ACM_RegimeHazard

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| RunID | nvarchar | NO | 64 | — |
| RegimeLabel | int | NO | 10 | — |
| RegimeState | nvarchar | YES | 32 | — |
| HazardRate | float | NO | 53 | — |
| CumulativeHazard | float | YES | 53 | — |
| SurvivalProbability | float | YES | 53 | — |
| MeanResidenceTime_Hours | float | YES | 53 | — |
| TransitionProbToFailure | float | YES | 53 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

---


## dbo.ACM_RegimeOccupancy

**Primary Key:** No primary key  
**Row Count:** 66  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RegimeLabel | nvarchar | NO | 50 | — |
| RecordCount | int | NO | 10 | — |
| Percentage | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| RegimeLabel | RecordCount | Percentage | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 0 | 1087 | 75.33 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 1 | 356 | 24.67 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0 | 1217 | 96.74 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| 1 | 41 | 3.26 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| 3 | 16 | 3.18 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 2 | 313 | 62.23 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 1 | 139 | 27.63 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 0 | 35 | 6.96 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 0 | 902 | 66.62 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 |
| 1 | 432 | 31.91 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 |

### Bottom 10 Records

| RegimeLabel | RecordCount | Percentage | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 0 | 245 | 48.51 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 1 | 49 | 9.7 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 2 | 55 | 10.89 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 3 | 46 | 9.11 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 4 | 78 | 15.45 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 5 | 32 | 6.34 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0 | 1314 | 97.62 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| 1 | 32 | 2.38 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| 0 | 43 | 9.75 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 |
| 1 | 221 | 50.11 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 |

---


## dbo.ACM_RegimeStability

**Primary Key:** No primary key  
**Row Count:** 15  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| MetricName | nvarchar | NO | 100 | — |
| MetricValue | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| MetricName | MetricValue | RunID | EquipID |
| --- | --- | --- | --- |
| RegimeStability | 96.52173913043478 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| RegimeStability | 99.52531645569621 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| RegimeStability | 93.6685288640596 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| RegimeStability | 94.09312022237664 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 |
| RegimeStability | 95.03386004514672 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 |
| RegimeStability | 97.54098360655739 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 |
| RegimeStability | 95.45454545454545 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 |
| RegimeStability | 98.35616438356165 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| RegimeStability | 95.09803921568627 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |
| RegimeStability | 90.0497512437811 | E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 |

### Bottom 10 Records

| MetricName | MetricValue | RunID | EquipID |
| --- | --- | --- | --- |
| RegimeStability | 93.00184162062615 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| RegimeStability | 99.40915805022156 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| RegimeStability | 93.03797468354429 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 |
| RegimeStability | 92.3809523809524 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| RegimeStability | 93.25842696629213 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 |
| RegimeStability | 90.0497512437811 | E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 |
| RegimeStability | 95.09803921568627 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |
| RegimeStability | 98.35616438356165 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| RegimeStability | 95.45454545454545 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 |
| RegimeStability | 97.54098360655739 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 |

---


## dbo.ACM_RegimeState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 2  
**Date Range:** 2025-12-05 06:08:22 to 2025-12-11 03:55:04  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| StateVersion | int | NO | 10 | — |
| NumClusters | int | NO | 10 | — |
| ClusterCentersJson | nvarchar | YES | -1 | — |
| ScalerMeanJson | nvarchar | YES | -1 | — |
| ScalerScaleJson | nvarchar | YES | -1 | — |
| PCAComponentsJson | nvarchar | YES | -1 | — |
| PCAExplainedVarianceJson | nvarchar | YES | -1 | — |
| NumPCAComponents | int | NO | 10 | ((0)) |
| SilhouetteScore | float | YES | 53 | — |
| QualityOk | bit | NO | — | ((0)) |
| LastTrainedTime | datetime2 | NO | — | — |
| ConfigHash | nvarchar | YES | 64 | — |
| RegimeBasisHash | nvarchar | YES | 64 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| EquipID | StateVersion | NumClusters | ClusterCentersJson | ScalerMeanJson | ScalerScaleJson | PCAComponentsJson | PCAExplainedVarianceJson | NumPCAComponents | SilhouetteScore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 6 | [[-0.4314314126968384, 0.38159599900245667, 0.3209207355976105], [1.0566831827163696, 0.998161971... | [0.21920855120136584, 0.08178160588748691, -0.1191714756560184] | [5.730766484677446, 4.53361937015279, 3.360560794527931] | [] | [] | 3 | 0.737538754940033 |
| 2621 | 1 | 6 | [[0.9438815116882324, -1.0460931062698364, 0.33586952090263367], [0.3532538414001465, 0.828715622... | [-1.9585829837005225e-09, 2.1464276511913523e-09, 4.254157688479014e-09] | [6.7357066205517295, 6.220930007614507, 4.438979977146924] | [] | [] | 3 | 0.5188907384872437 |

---


## dbo.ACM_RegimeStats

**Primary Key:** No primary key  
**Row Count:** 66  
**Date Range:** 2025-12-05 11:37:00 to 2025-12-11 09:25:07  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| RegimeLabel | int | NO | 10 | — |
| OccupancyPct | float | YES | 53 | — |
| AvgDwellSeconds | float | YES | 53 | — |
| FusedMean | float | YES | 53 | — |
| FusedP90 | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| RunID | EquipID | RegimeLabel | OccupancyPct | AvgDwellSeconds | FusedMean | FusedP90 | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:23:21 |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:23:21 |
| 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:24:21 |
| 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:24:21 |
| 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:21:33 |
| 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:21:33 |
| 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:21:33 |
| 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:21:33 |
| 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:22:50 |
| 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:22:50 |

### Bottom 10 Records

| RunID | EquipID | RegimeLabel | OccupancyPct | AvgDwellSeconds | FusedMean | FusedP90 | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:23:52 |
| 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:23:52 |
| 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:24:37 |
| 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-11 09:24:37 |

---


## dbo.ACM_RegimeTimeline

**Primary Key:** No primary key  
**Row Count:** 10,893  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| RegimeLabel | nvarchar | NO | 50 | — |
| RegimeState | nvarchar | NO | 50 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Timestamp | RegimeLabel | RegimeState | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 2023-10-18 00:00:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:30:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:00:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:30:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 02:00:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 02:30:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 03:00:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 03:30:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 04:00:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 04:30:00 | 5 | unknown | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |

### Bottom 10 Records

| Timestamp | RegimeLabel | RegimeState | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:30:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:00:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 21:30:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 21:00:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 20:30:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 20:00:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 19:30:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 19:00:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 18:30:00 | 0 | unknown | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |

---


## dbo.ACM_RegimeTransitions

**Primary Key:** No primary key  
**Row Count:** 95  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| FromLabel | nvarchar | NO | 50 | — |
| ToLabel | nvarchar | NO | 50 | — |
| Count | int | NO | 10 | — |
| Prob | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| FromLabel | ToLabel | Count | Prob | RunID | EquipID |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 1.0 | 26 | 0.5 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 1.0 | 0.0 | 26 | 0.5 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| 0.0 | 1.0 | 3 | 0.5 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| 1.0 | 0.0 | 3 | 0.5 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| 0.0 | 3.0 | 1 | 0.0294 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 0.0 | 2.0 | 4 | 0.1176 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 0.0 | 1.0 | 3 | 0.0882 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 1.0 | 0.0 | 1 | 0.0294 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 1.0 | 2.0 | 8 | 0.2353 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |
| 1.0 | 3.0 | 1 | 0.0294 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 |

### Bottom 10 Records

| FromLabel | ToLabel | Count | Prob | RunID | EquipID |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 3.0 | 5 | 0.1316 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 1.0 | 2.0 | 2 | 0.0526 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 1.0 | 5.0 | 4 | 0.1053 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 2.0 | 1.0 | 1 | 0.0263 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 2.0 | 4.0 | 6 | 0.1579 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 3.0 | 0.0 | 3 | 0.0789 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 3.0 | 2.0 | 5 | 0.1316 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 3.0 | 5.0 | 1 | 0.0263 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 4.0 | 1.0 | 6 | 0.1579 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 5.0 | 0.0 | 1 | 0.0263 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |

---


## dbo.ACM_Regime_Episodes

**Primary Key:** Id  
**Row Count:** 98  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | YES | 10 | — |
| StartTime | datetime2 | YES | — | — |
| EndTime | datetime2 | YES | — | — |
| RegimeID | int | YES | 10 | — |

### Top 10 Records

| Id | RunID | EquipID | StartTime | EndTime | RegimeID |
| --- | --- | --- | --- | --- | --- |
| 3344 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3345 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3346 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3347 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3348 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3349 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3350 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3351 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3352 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |
| 3353 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | NULL | NULL |

### Bottom 10 Records

| Id | RunID | EquipID | StartTime | EndTime | RegimeID |
| --- | --- | --- | --- | --- | --- |
| 4602 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | NULL | NULL | NULL |
| 4601 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | NULL | NULL | NULL |
| 4600 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | NULL | NULL | NULL |
| 4599 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | NULL | NULL | NULL |
| 4598 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | NULL | NULL | NULL |
| 4597 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | NULL | NULL | NULL |
| 4596 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | NULL | NULL | NULL |
| 4595 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | NULL | NULL | NULL |
| 4594 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | NULL | NULL | NULL |
| 4593 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | NULL | NULL | NULL |

---


## dbo.ACM_RunLogs

**Primary Key:** LogID  
**Row Count:** 338,940  
**Date Range:** 2025-12-02 05:59:43 to 2025-12-11 03:55:08  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| LogID | bigint | NO | 19 | — |
| RunID | nvarchar | YES | 64 | — |
| EquipID | int | YES | 10 | — |
| LoggedAt | datetime2 | NO | — | (sysutcdatetime()) |
| Level | nvarchar | NO | 16 | — |
| Module | nvarchar | YES | 128 | — |
| Message | nvarchar | NO | 4000 | — |
| Context | nvarchar | YES | -1 | — |
| LoggedLocal | datetimeoffset | YES | — | — |
| LoggedLocalNaive | datetime2 | YES | — | — |
| EventType | nvarchar | YES | 32 | — |
| Stage | nvarchar | YES | 64 | — |
| StepName | nvarchar | YES | 128 | — |
| DurationMs | float | YES | 53 | — |
| RowCount | int | YES | 10 | — |
| ColCount | int | YES | 10 | — |
| WindowSize | int | YES | 10 | — |
| BatchStart | datetime2 | YES | — | — |
| BatchEnd | datetime2 | YES | — | — |
| BaselineStart | datetime2 | YES | — | — |
| BaselineEnd | datetime2 | YES | — | — |
| DataQualityMetric | nvarchar | YES | 64 | — |
| DataQualityValue | float | YES | 53 | — |
| LeakageFlag | bit | YES | — | — |
| ParamsJson | nvarchar | YES | -1 | — |

---


## dbo.ACM_RunMetadata

**Primary Key:** RunMetadataID  
**Row Count:** 715  
**Date Range:** 2025-12-02 05:59:53 to 2025-12-08 05:09:28  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunMetadataID | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| EquipName | nvarchar | NO | 128 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |
| RetrainDecision | bit | NO | — | — |
| RetrainReason | nvarchar | YES | 256 | — |
| ForecastStateVersion | int | YES | 10 | — |
| ModelAgeBatches | int | YES | 10 | — |
| ForecastRMSE | float | YES | 53 | — |
| ForecastMAE | float | YES | 53 | — |
| ForecastMAPE | float | YES | 53 | — |

### Top 10 Records

| RunMetadataID | RunID | EquipID | EquipName | CreatedAt | RetrainDecision | RetrainReason | ForecastStateVersion | ModelAgeBatches | ForecastRMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 697 | A9FC5331-B292-4515-A146-6DEFDF4DB83B | 2621 | GAS_TURBINE | 2025-12-02 05:59:53 | False | Model stable, incremental update | 32 | NULL | 0.0 |
| 698 | B687BF83-7290-4E4F-9180-8CECAD93BF75 | 1 | FD_FAN | 2025-12-02 05:59:55 | False | Model stable, incremental update | 149 | NULL | 0.0 |
| 699 | 8684168E-DFB2-4220-A6D6-772D3FB513EF | 1 | FD_FAN | 2025-12-02 06:01:47 | False | Model stable, incremental update | 150 | NULL | 0.0 |
| 700 | FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | FD_FAN | 2025-12-02 06:01:48 | False | Model stable, incremental update | 151 | NULL | 0.0 |
| 701 | 60B9EFF0-C9F2-4F64-B95A-1E4566B46493 | 2621 | GAS_TURBINE | 2025-12-02 06:01:48 | False | Model stable, incremental update | 33 | NULL | 0.0 |
| 702 | 236A2227-190A-40F1-81B8-0F3F45E43CFF | 2621 | GAS_TURBINE | 2025-12-02 06:01:48 | False | Model stable, incremental update | 33 | NULL | 0.0 |
| 703 | A98A8F95-BF62-413B-AE50-EAA158A19941 | 1 | FD_FAN | 2025-12-02 06:21:57 | True | Scheduled retrain (9720h since last > 168.0h limit) | 152 | NULL | 0.0 |
| 704 | 2BC2A328-9D69-428C-A45A-6495E7503A61 | 1 | FD_FAN | 2025-12-02 06:22:05 | True | Forecast accuracy degraded (RMSE=47.09 > 2.0) | 153 | NULL | 47.09326712906889 |
| 705 | 790BAAA8-3248-484A-ADD3-4B587D777863 | 1 | FD_FAN | 2025-12-02 06:22:13 | True | Forecast accuracy degraded (RMSE=68.35 > 2.0) | 154 | NULL | 68.35162486866703 |
| 706 | 434F66BA-3426-4397-903D-5A48F15DCCAB | 1 | FD_FAN | 2025-12-02 06:22:19 | False | Model stable, incremental update | 155 | NULL | 0.0 |

### Bottom 10 Records

| RunMetadataID | RunID | EquipID | EquipName | CreatedAt | RetrainDecision | RetrainReason | ForecastStateVersion | ModelAgeBatches | ForecastRMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1411 | 2297A0E1-DC1F-4073-AC0B-A44FF11B0391 | 1 | FD_FAN | 2025-12-08 05:09:28 | True | Anomaly energy spike (Max=2.99 > 1.5x avg) | 803 | NULL | 0.0 |
| 1410 | DE7598F0-E4B9-4A0D-AAC9-100DF39E172F | 1 | FD_FAN | 2025-12-08 05:09:14 | True | Anomaly energy spike (Max=2.39 > 1.5x avg) | 802 | NULL | 0.0 |
| 1409 | B837C8B2-6023-41A6-8C47-293DF05B9916 | 1 | FD_FAN | 2025-12-08 05:08:58 | True | Anomaly energy spike (Max=1.75 > 1.5x avg) | 801 | NULL | 0.0 |
| 1408 | E5EE817F-C89E-48AE-9F12-63E1436D430A | 1 | FD_FAN | 2025-12-08 05:08:40 | True | Anomaly energy spike (Max=7.33 > 1.5x avg) | 800 | NULL | 0.0 |
| 1407 | 303F2A8B-E23E-4155-82B4-BC56A058EB04 | 1 | FD_FAN | 2025-12-08 05:08:11 | True | Anomaly energy spike (Max=1.53 > 1.5x avg) | 799 | NULL | 0.0 |
| 1406 | A659815F-17C2-4F69-A323-A91ED6708777 | 1 | FD_FAN | 2025-12-08 05:07:39 | True | Anomaly energy spike (Max=3.14 > 1.5x avg) | 798 | NULL | 0.0 |
| 1405 | 8E918F6B-B84F-4440-924F-D4D970F490F6 | 1 | FD_FAN | 2025-12-08 05:07:08 | True | Anomaly energy spike (Max=9.59 > 1.5x avg) | 797 | NULL | 0.0 |
| 1404 | 43574183-1F99-4BD8-AB6C-E89D80D9F64C | 1 | FD_FAN | 2025-12-08 05:06:36 | True | Anomaly energy spike (Max=7.40 > 1.5x avg) | 796 | NULL | 0.0 |
| 1403 | B1CF7C65-8699-49B9-8C32-8AA9C0DCA270 | 1 | FD_FAN | 2025-12-08 05:06:04 | True | Anomaly energy spike (Max=1.56 > 1.5x avg) | 795 | NULL | 0.0 |
| 1402 | 454F21D3-5BB3-42E6-890C-1F2BE8062D04 | 1 | FD_FAN | 2025-12-08 05:05:48 | True | Anomaly energy spike (Max=2.93 > 1.5x avg) | 794 | NULL | 0.0 |

---


## dbo.ACM_RunMetrics

**Primary Key:** RunID, EquipID, MetricName  
**Row Count:** 25,305  
**Date Range:** 2025-12-01 17:15:57 to 2025-12-11 09:25:04  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | nvarchar | NO | 72 | — |
| EquipID | int | NO | 10 | — |
| MetricName | nvarchar | NO | 100 | — |
| MetricValue | float | NO | 53 | — |
| Timestamp | datetime2 | NO | — | (getdate()) |

### Top 10 Records

| RunID | EquipID | MetricName | MetricValue | Timestamp |
| --- | --- | --- | --- | --- |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.n_samples.ar1_z | 97.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.n_samples.gmm_z | 97.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.n_samples.iforest_z | 97.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.n_samples.mhal_z | 97.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.n_samples.omr_z | 97.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.n_samples.pca_spe_z | 97.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.n_samples.pca_t2_z | 97.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.quality.ar1_z | 0.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.quality.gmm_z | 0.0 | 2025-12-01 17:15:57 |
| 006C43EA-0C88-46AC-9824-C4EB85C17297 | 1 | fusion.quality.iforest_z | 0.0 | 2025-12-01 17:15:57 |

### Bottom 10 Records

| RunID | EquipID | MetricName | MetricValue | Timestamp |
| --- | --- | --- | --- | --- |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.weight.pca_t2_z | 0.04964539007092199 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.weight.pca_spe_z | 0.18156028368794327 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.weight.omr_z | 0.11205673758865248 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.weight.mhal_z | 0.18156028368794327 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.weight.iforest_z | 0.18156028368794327 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.weight.gmm_z | 0.11205673758865248 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.weight.ar1_z | 0.18156028368794327 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.quality.pca_t2_z | 0.0 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.quality.pca_spe_z | 0.0 | 2025-12-02 11:31:44 |
| FFD41532-B96A-43F0-AAA2-75B0341E6CAE | 1 | fusion.quality.omr_z | 0.0 | 2025-12-02 11:31:44 |

---


## dbo.ACM_Run_Stats

**Primary Key:** RecordID  
**Row Count:** 15  
**Date Range:** 2023-10-15 00:00:00 to 2025-07-06 21:09:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RecordID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| WindowStartEntryDateTime | datetime2 | YES | — | — |
| WindowEndEntryDateTime | datetime2 | YES | — | — |
| SamplesIn | int | YES | 10 | — |
| SamplesKept | int | YES | 10 | — |
| SensorsKept | int | YES | 10 | — |
| CadenceOKPct | float | YES | 53 | — |
| DriftP95 | float | YES | 53 | — |
| ReconRMSE | float | YES | 53 | — |
| AnomalyCount | int | YES | 10 | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |

### Top 10 Records

| RecordID | RunID | EquipID | WindowStartEntryDateTime | WindowEndEntryDateTime | SamplesIn | SamplesKept | SensorsKept | CadenceOKPct | DriftP95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 239 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-15 00:00:00 | 2023-10-15 23:59:59 | 97 | 97 | 16 | 100.0 | NULL |
| 241 | BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-10-15 00:00:00 | 2023-12-24 02:20:59 | 842 | 842 | 16 | 100.0 | NULL |
| 242 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2023-12-24 02:21:00 | 2024-03-03 04:41:59 | 505 | 505 | 16 | 100.0 | NULL |
| 243 | E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 | 2024-05-12 07:03:00 | 2024-06-16 01:59:00 | 362 | 362 | 16 | 100.0 | NULL |
| 415 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | 2023-10-15 00:00:00 | 2023-10-15 23:59:59 | 97 | 97 | 9 | 100.0 | NULL |
| 416 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 2023-10-15 00:00:00 | 2023-12-24 02:20:59 | 503 | 503 | 9 | 100.0 | NULL |
| 417 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | 2023-12-24 02:21:00 | 2024-03-03 04:41:59 | 399 | 399 | 9 | 100.0 | NULL |
| 418 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | 2024-03-03 04:42:00 | 2024-05-12 07:02:59 | 1411 | 1411 | 9 | 100.0 | NULL |
| 419 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 2024-05-12 07:03:00 | 2024-07-21 09:23:59 | 1354 | 1354 | 9 | 100.0 | NULL |
| 420 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-07-21 09:24:00 | 2024-09-29 11:44:59 | 1443 | 1443 | 9 | 100.0 | NULL |

### Bottom 10 Records

| RecordID | RunID | EquipID | WindowStartEntryDateTime | WindowEndEntryDateTime | SamplesIn | SamplesKept | SensorsKept | CadenceOKPct | DriftP95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 425 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-07-06 21:09:00 | 2025-09-14 23:29:59 | 359 | 359 | 9 | 100.0 | NULL |
| 424 | D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | 2025-04-27 18:48:00 | 2025-07-06 21:08:59 | 476 | 476 | 9 | 100.0 | NULL |
| 423 | 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | 2025-02-16 16:27:00 | 2025-04-27 18:47:59 | 441 | 441 | 9 | 100.0 | NULL |
| 422 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 | 2024-12-08 14:06:00 | 2025-02-16 16:26:59 | 1258 | 1258 | 9 | 100.0 | NULL |
| 421 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | 2024-09-29 11:45:00 | 2024-12-08 14:05:59 | 1346 | 1346 | 9 | 100.0 | NULL |
| 420 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | 2024-07-21 09:24:00 | 2024-09-29 11:44:59 | 1443 | 1443 | 9 | 100.0 | NULL |
| 419 | 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | 2024-05-12 07:03:00 | 2024-07-21 09:23:59 | 1354 | 1354 | 9 | 100.0 | NULL |
| 418 | ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | 2024-03-03 04:42:00 | 2024-05-12 07:02:59 | 1411 | 1411 | 9 | 100.0 | NULL |
| 417 | 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | 2023-12-24 02:21:00 | 2024-03-03 04:41:59 | 399 | 399 | 9 | 100.0 | NULL |
| 416 | 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | 2023-10-15 00:00:00 | 2023-12-24 02:20:59 | 503 | 503 | 9 | 100.0 | NULL |

---


## dbo.ACM_Runs

**Primary Key:** RunID  
**Row Count:** 17  
**Date Range:** 2025-12-05 06:06:49 to 2025-12-11 03:54:57  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| EquipName | nvarchar | YES | 200 | — |
| StartedAt | datetime2 | NO | — | — |
| CompletedAt | datetime2 | YES | — | — |
| DurationSeconds | int | YES | 10 | — |
| ConfigSignature | varchar | YES | 64 | — |
| TrainRowCount | int | YES | 10 | — |
| ScoreRowCount | int | YES | 10 | — |
| EpisodeCount | int | YES | 10 | — |
| HealthStatus | varchar | YES | 50 | — |
| AvgHealthIndex | float | YES | 53 | — |
| MinHealthIndex | float | YES | 53 | — |
| MaxFusedZ | float | YES | 53 | — |
| DataQualityScore | float | YES | 53 | — |
| RefitRequested | bit | YES | — | ((0)) |
| ErrorMessage | nvarchar | YES | 1000 | — |
| KeptColumns | nvarchar | YES | -1 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| RunID | EquipID | EquipName | StartedAt | CompletedAt | DurationSeconds | ConfigSignature | TrainRowCount | ScoreRowCount | EpisodeCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ECF1B048-3C9F-432E-8495-047931ED3351 | 2621 | NULL | 2025-12-05 06:08:08 | 2025-12-05 06:08:08 | NULL |  | 0 | 0 | NULL |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | NULL | 2025-12-11 03:52:57 | 2025-12-11 03:53:24 | NULL |  | 1443 | 9137 | NULL |
| 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 | NULL | 2025-12-11 03:53:59 | 2025-12-11 03:54:23 | NULL |  | 1258 | 8021 | NULL |
| 9AE4804E-79C6-45F7-8DC3-17F84319B26B | 2621 | NULL | 2025-12-05 06:30:24 | 2025-12-05 06:30:24 | NULL |  | 0 | 0 | NULL |
| 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | NULL | 2025-12-11 03:51:21 | 2025-12-11 03:51:35 | NULL |  | 503 | 3479 | NULL |
| 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | NULL | 2025-12-11 03:52:25 | 2025-12-11 03:52:54 | NULL |  | 1354 | 8603 | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | NULL | 2025-12-05 06:07:06 | 2025-12-05 06:07:40 | NULL |  | 842 | 5875 | NULL |
| D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | NULL | 2025-12-11 03:54:42 | 2025-12-11 03:54:54 | NULL |  | 476 | 3315 | NULL |
| 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | NULL | 2025-12-11 03:51:38 | 2025-12-11 03:51:50 | NULL |  | 399 | 2849 | NULL |
| 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | NULL | 2025-12-11 03:54:57 | 2025-12-11 03:55:08 | NULL |  | 359 | 2609 | NULL |

### Bottom 10 Records

| RunID | EquipID | EquipName | StartedAt | CompletedAt | DurationSeconds | ConfigSignature | TrainRowCount | ScoreRowCount | EpisodeCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | NULL | 2025-12-05 06:07:43 | 2025-12-05 06:08:04 | NULL |  | 505 | 3845 | NULL |
| 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | NULL | 2025-12-11 03:53:28 | 2025-12-11 03:53:55 | NULL |  | 1346 | 8547 | NULL |
| 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | NULL | 2025-12-11 03:54:26 | 2025-12-11 03:54:39 | NULL |  | 441 | 3105 | NULL |
| 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | NULL | 2025-12-11 03:51:09 | 2025-12-11 03:51:17 | NULL |  | 97 | 1035 | NULL |
| ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | NULL | 2025-12-11 03:51:54 | 2025-12-11 03:52:22 | NULL |  | 1411 | 8935 | NULL |
| E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 | NULL | 2025-12-05 06:08:11 | 2025-12-05 06:08:29 | NULL |  | 362 | 2981 | NULL |
| 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | NULL | 2025-12-05 06:06:49 | 2025-12-05 06:07:03 | NULL |  | 97 | 1373 | NULL |
| 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | NULL | 2025-12-11 03:54:57 | 2025-12-11 03:55:08 | NULL |  | 359 | 2609 | NULL |
| 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | NULL | 2025-12-11 03:51:38 | 2025-12-11 03:51:50 | NULL |  | 399 | 2849 | NULL |
| D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | NULL | 2025-12-11 03:54:42 | 2025-12-11 03:54:54 | NULL |  | 476 | 3315 | NULL |

---


## dbo.ACM_SchemaVersion

**Primary Key:** VersionID  
**Row Count:** 2  
**Date Range:** 2025-12-03 11:06:16 to 2025-12-03 11:06:16  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| VersionID | int | NO | 10 | — |
| VersionNumber | varchar | NO | 20 | — |
| Description | varchar | YES | 500 | — |
| AppliedAt | datetime2 | NO | — | (getutcdate()) |
| AppliedBy | varchar | NO | 100 | (suser_sname()) |

### Top 10 Records

| VersionID | VersionNumber | Description | AppliedAt | AppliedBy |
| --- | --- | --- | --- | --- |
| 1 | 1.0.0 | Initial ACM schema with core tables | 2025-12-03 11:06:16 | SYSTEM |
| 2 | 1.1.0 | Added ACM_SinceWhen and ACM_BaselineBuffer tables | 2025-12-03 11:06:16 | B19cl3pc\bhadk |

---


## dbo.ACM_Scores_Long

**Primary Key:** Id  
**Row Count:** 65,358  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | YES | 10 | — |
| Timestamp | datetime2 | YES | — | — |
| SensorName | nvarchar | YES | 128 | — |
| DetectorName | nvarchar | YES | 64 | — |
| Score | float | YES | 53 | — |
| Threshold | float | YES | 53 | — |
| IsAnomaly | bit | YES | — | — |

### Top 10 Records

| Id | RunID | EquipID | Timestamp | SensorName | DetectorName | Score | Threshold | IsAnomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3131911 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-20 23:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131912 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 00:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131913 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 01:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131914 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 02:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131915 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 03:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131916 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 04:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131917 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 05:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131918 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 06:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131919 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 07:59:00 | NULL | NULL | NULL | NULL | NULL |
| 3131920 | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | 2023-10-21 08:59:00 | NULL | NULL | NULL | NULL | NULL |

### Bottom 10 Records

| Id | RunID | EquipID | Timestamp | SensorName | DetectorName | Score | Threshold | IsAnomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4112646 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 23:00:00 | NULL | NULL | NULL | NULL | NULL |
| 4112645 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 22:30:00 | NULL | NULL | NULL | NULL | NULL |
| 4112644 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 22:00:00 | NULL | NULL | NULL | NULL | NULL |
| 4112643 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 21:30:00 | NULL | NULL | NULL | NULL | NULL |
| 4112642 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 21:00:00 | NULL | NULL | NULL | NULL | NULL |
| 4112641 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 20:30:00 | NULL | NULL | NULL | NULL | NULL |
| 4112640 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 20:00:00 | NULL | NULL | NULL | NULL | NULL |
| 4112639 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 19:30:00 | NULL | NULL | NULL | NULL | NULL |
| 4112638 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 19:00:00 | NULL | NULL | NULL | NULL | NULL |
| 4112637 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | 2025-09-14 18:30:00 | NULL | NULL | NULL | NULL | NULL |

---


## dbo.ACM_Scores_Wide

**Primary Key:** No primary key  
**Row Count:** 10,893  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| ar1_z | float | YES | 53 | — |
| pca_spe_z | float | YES | 53 | — |
| pca_t2_z | float | YES | 53 | — |
| mhal_z | float | YES | 53 | — |
| iforest_z | float | YES | 53 | — |
| gmm_z | float | YES | 53 | — |
| cusum_z | float | YES | 53 | — |
| drift_z | float | YES | 53 | — |
| hst_z | float | YES | 53 | — |
| river_hst_z | float | YES | 53 | — |
| fused | float | YES | 53 | — |
| regime_label | nvarchar | YES | 50 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Timestamp | ar1_z | pca_spe_z | pca_t2_z | mhal_z | iforest_z | gmm_z | cusum_z | drift_z | hst_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-18 00:00:00 | -2.8312058448791504 | 0.7606475949287415 | 0.0 | 1.8121169805526733 | 1.1893054246902466 | 1.697385549545288 | -1.5786223411560059 | NULL | NULL |
| 2023-10-18 00:30:00 | 5.791759967803955 | 4.947620391845703 | 2.2207753658294678 | 1.1282336711883545 | 1.60547935962677 | 1.4426548480987549 | -1.4705451726913452 | NULL | NULL |
| 2023-10-18 01:00:00 | 2.160860538482666 | 0.6616067290306091 | 1.2305405139923096 | 1.4335641860961914 | 0.8909816145896912 | 0.4405704736709595 | -1.3587651252746582 | NULL | NULL |
| 2023-10-18 01:30:00 | 0.674490749835968 | -0.11016302555799484 | 0.3788636028766632 | 0.8532254695892334 | 0.7114445567131042 | 0.0 | -1.2727506160736084 | NULL | NULL |
| 2023-10-18 02:00:00 | 0.7255694270133972 | -0.059708625078201294 | 0.259790301322937 | -0.4578806757926941 | 0.6377455592155457 | -0.229465514421463 | -1.2183494567871094 | NULL | NULL |
| 2023-10-18 02:30:00 | -0.4277987778186798 | 0.0757986381649971 | -0.026880580931901932 | -1.2002801895141602 | 0.8676197528839111 | -0.5965303778648376 | -1.1997430324554443 | NULL | NULL |
| 2023-10-18 03:00:00 | -1.0020948648452759 | 0.5092217922210693 | -0.26506495475769043 | -0.7330832481384277 | 0.449108362197876 | -0.5655981302261353 | -1.2050386667251587 | NULL | NULL |
| 2023-10-18 03:30:00 | -0.5956984162330627 | 0.7133747339248657 | -0.679521918296814 | -0.008700143545866013 | 0.35423150658607483 | -1.1140401363372803 | -1.2197966575622559 | NULL | NULL |
| 2023-10-18 04:00:00 | 0.6916334629058838 | 0.8790250420570374 | -0.7308681607246399 | 0.26368093490600586 | 1.0782363414764404 | -1.158806324005127 | -1.2223577499389648 | NULL | NULL |
| 2023-10-18 04:30:00 | -0.8363309502601624 | -0.07078950107097626 | -0.7215713262557983 | -0.7953579425811768 | 0.13801367580890656 | -1.5127971172332764 | -1.256269097328186 | NULL | NULL |

### Bottom 10 Records

| Timestamp | ar1_z | pca_spe_z | pca_t2_z | mhal_z | iforest_z | gmm_z | cusum_z | drift_z | hst_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | 0.46040481328964233 | -0.14593912661075592 | 0.4965899586677551 | 1.9531084299087524 | 1.510094404220581 | 1.1065171957015991 | 0.24305035173892975 | NULL | NULL |
| 2025-09-14 22:30:00 | 0.5406763553619385 | 0.04148339480161667 | 0.6109948754310608 | 2.1521005630493164 | 1.5021002292633057 | 1.155322790145874 | 0.25662779808044434 | NULL | NULL |
| 2025-09-14 22:00:00 | 0.7433663606643677 | -0.28319427371025085 | 0.6626898646354675 | 3.0498695373535156 | 1.15378737449646 | 0.8512459397315979 | 0.27113020420074463 | NULL | NULL |
| 2025-09-14 21:30:00 | 1.5137767791748047 | -0.024792121723294258 | 0.7170666456222534 | 2.0537002086639404 | 0.687981903553009 | 0.6548027396202087 | 0.28759118914604187 | NULL | NULL |
| 2025-09-14 21:00:00 | 0.7707768082618713 | -0.17121942341327667 | 0.4449594020843506 | 1.3849334716796875 | 0.4332031011581421 | 0.36992523074150085 | 0.30623260140419006 | NULL | NULL |
| 2025-09-14 20:30:00 | 0.6612345576286316 | -0.30413147807121277 | 0.24392221868038177 | 1.6964364051818848 | 0.33080869913101196 | 0.2858586609363556 | 0.3276815712451935 | NULL | NULL |
| 2025-09-14 20:00:00 | 0.3016025125980377 | -0.5251924991607666 | 0.2010573297739029 | 1.2892252206802368 | -0.1915440410375595 | 0.06451921910047531 | 0.3508540987968445 | NULL | NULL |
| 2025-09-14 19:30:00 | 0.29689913988113403 | -0.46577373147010803 | 0.07979092746973038 | 1.0747450590133667 | -0.4705916941165924 | -0.15570619702339172 | 0.3759993314743042 | NULL | NULL |
| 2025-09-14 19:00:00 | 0.7381052374839783 | -0.4735943675041199 | 0.08426221460103989 | 1.1302820444107056 | -0.4146885275840759 | 0.16424739360809326 | 0.40173599123954773 | NULL | NULL |
| 2025-09-14 18:30:00 | 0.3653334081172943 | -0.5496241450309753 | 0.028022831305861473 | 0.5700039267539978 | -0.5838836431503296 | -0.13422787189483643 | 0.4276062548160553 | NULL | NULL |

---


## dbo.ACM_SensorAnomalyByPeriod

**Primary Key:** No primary key  
**Row Count:** 2,216  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 00:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Date | date | NO | — | — |
| PeriodStart | datetime2 | NO | — | — |
| PeriodType | nvarchar | NO | 20 | — |
| PeriodSeconds | float | NO | 53 | — |
| DetectorType | nvarchar | NO | 50 | — |
| AnomalyRatePct | float | NO | 53 | — |
| MaxZ | float | NO | 53 | — |
| AvgZ | float | NO | 53 | — |
| Points | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Date | PeriodStart | PeriodType | PeriodSeconds | DetectorType | AnomalyRatePct | MaxZ | AvgZ | Points | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-18 00:00:00 | 2023-10-18 00:00:00 | DAY | 86400.0 | ar1 | 6.25 | 5.791800022125244 | 0.9652000069618225 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-18 00:00:00 | 2023-10-18 00:00:00 | DAY | 86400.0 | pca_spe | 8.33 | 4.9475998878479 | 0.7569000124931335 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-18 00:00:00 | 2023-10-18 00:00:00 | DAY | 86400.0 | pca_t2 | 4.17 | 2.2207999229431152 | 0.8314999938011169 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-18 00:00:00 | 2023-10-18 00:00:00 | DAY | 86400.0 | mhal | 0.0 | 1.8703999519348145 | 0.670799970626831 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-18 00:00:00 | 2023-10-18 00:00:00 | DAY | 86400.0 | iforest | 0.0 | 1.6054999828338623 | 0.6093000173568726 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-18 00:00:00 | 2023-10-18 00:00:00 | DAY | 86400.0 | gmm | 2.08 | 2.0327000617980957 | 0.980400025844574 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-18 00:00:00 | 2023-10-18 00:00:00 | DAY | 86400.0 | omr | 6.25 | 3.4261999130249023 | 0.6837999820709229 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-18 00:00:00 | 2023-10-18 00:00:00 | DAY | 86400.0 | cusum | 0.0 | 1.5786000490188599 | 0.6761999726295471 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-19 00:00:00 | 2023-10-19 00:00:00 | DAY | 86400.0 | ar1 | 6.25 | 6.743800163269043 | 0.7534999847412109 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |
| 2023-10-19 00:00:00 | 2023-10-19 00:00:00 | DAY | 86400.0 | pca_spe | 12.5 | 6.627099990844727 | 1.1741000413894653 | 48 | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF |

### Bottom 10 Records

| Date | PeriodStart | PeriodType | PeriodSeconds | DetectorType | AnomalyRatePct | MaxZ | AvgZ | Points | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | ar1 | 8.51 | 4.035600185394287 | 0.7642999887466431 | 47 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | pca_spe | 8.51 | 7.460700035095215 | 0.8166999816894531 | 47 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | pca_t2 | 2.13 | 2.161099910736084 | 0.5256999731063843 | 47 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | mhal | 63.83 | 10.0 | 5.161900043487549 | 47 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | iforest | 8.51 | 2.5731000900268555 | 1.055299997329712 | 47 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | gmm | 27.66 | 4.924699783325195 | 1.4084999561309814 | 47 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | omr | 6.38 | 4.2032999992370605 | 0.683899998664856 | 47 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | cusum | 0.0 | 0.7073000073432922 | 0.580299973487854 | 47 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-13 00:00:00 | 2025-09-13 00:00:00 | DAY | 86400.0 | ar1 | 14.58 | 8.993800163269043 | 1.082800030708313 | 48 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |
| 2025-09-13 00:00:00 | 2025-09-13 00:00:00 | DAY | 86400.0 | pca_spe | 10.42 | 9.134400367736816 | 0.9368000030517578 | 48 | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 |

---


## dbo.ACM_SensorDefects

**Primary Key:** No primary key  
**Row Count:** 120  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorType | nvarchar | NO | 50 | — |
| DetectorFamily | nvarchar | NO | 50 | — |
| Severity | nvarchar | NO | 50 | — |
| ViolationCount | int | NO | 10 | — |
| ViolationPct | float | NO | 53 | — |
| MaxZ | float | NO | 53 | — |
| AvgZ | float | NO | 53 | — |
| CurrentZ | float | NO | 53 | — |
| ActiveDefect | nvarchar | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| DetectorType | DetectorFamily | Severity | ViolationCount | ViolationPct | MaxZ | AvgZ | CurrentZ | ActiveDefect | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multivariate Distance (Mahalanobis) | Multivariate | CRITICAL | 484 | 33.54 | 10.0 | 2.552 | 0.3216 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 |
| Correlation Break (PCA-SPE) | Correlation | CRITICAL | 461 | 31.95 | 10.0 | 2.2174 | 0.1868 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 |
| Density Anomaly (GMM) | Density | CRITICAL | 425 | 29.45 | 10.0 | 2.0494 | 0.1897 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 |
| Baseline Consistency (OMR) | Baseline | CRITICAL | 390 | 27.03 | 6.1271 | 1.5563 | 0.1762 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 |
| Rare State (IsolationForest) | Rare | CRITICAL | 302 | 20.93 | 6.3226 | 1.2407 | 0.3009 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 |
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 302 | 20.93 | 10.0 | 1.3751 | 0.4306 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 |
| Time-Series Anomaly (AR1) | Time-Series | MEDIUM | 143 | 9.91 | 10.0 | 1.0379 | 0.7491 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 |
| cusum_z | cusum_z | LOW | 0 | 0.0 | 1.7539 | 0.7296 | 0.153 | 0 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 |
| Multivariate Distance (Mahalanobis) | Multivariate | CRITICAL | 743 | 59.06 | 10.0 | 3.4534 | 5.2052 | 1 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED |
| Density Anomaly (GMM) | Density | CRITICAL | 335 | 26.63 | 10.0 | 1.9358 | 4.8379 | 1 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED |

### Bottom 10 Records

| DetectorType | DetectorFamily | Severity | ViolationCount | ViolationPct | MaxZ | AvgZ | CurrentZ | ActiveDefect | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pca_t2 | pca | CRITICAL | 278 | 55.05 | 10.0 | 4.861 | 10.0 | 1 | 17AF291B-3A84-456F-BDE8-F424528C797D |
| ar1 | ar1 | CRITICAL | 190 | 37.62 | 10.0 | 2.3531 | 3.669 | 1 | 17AF291B-3A84-456F-BDE8-F424528C797D |
| mhal | mhal | CRITICAL | 188 | 37.23 | 10.0 | 3.282 | 10.0 | 1 | 17AF291B-3A84-456F-BDE8-F424528C797D |
| pca_spe | pca | HIGH | 83 | 16.44 | 10.0 | 1.3502 | 0.1128 | 0 | 17AF291B-3A84-456F-BDE8-F424528C797D |
| gmm | gmm | HIGH | 55 | 10.89 | 10.0 | 1.2755 | 1.7988 | 0 | 17AF291B-3A84-456F-BDE8-F424528C797D |
| omr | omr | MEDIUM | 46 | 9.11 | 7.9734 | 0.9666 | 0.0988 | 0 | 17AF291B-3A84-456F-BDE8-F424528C797D |
| iforest | iforest | LOW | 3 | 0.59 | 2.1376 | 0.8466 | 1.45 | 0 | 17AF291B-3A84-456F-BDE8-F424528C797D |
| cusum | cusum | LOW | 0 | 0.0 | 1.899 | 0.7503 | 1.0084 | 0 | 17AF291B-3A84-456F-BDE8-F424528C797D |
| Multivariate Distance (Mahalanobis) | Multivariate | CRITICAL | 1159 | 86.11 | 10.0 | 5.8972 | 0.2203 | 0 | 3C73280A-9598-4410-BB08-E1DA79FB0675 |
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 998 | 74.15 | 10.0 | 2.9355 | 0.7194 | 0 | 3C73280A-9598-4410-BB08-E1DA79FB0675 |

---


## dbo.ACM_SensorForecast

**Primary Key:** RunID, EquipID, Timestamp, SensorName  
**Row Count:** 6,720  
**Date Range:** 2023-10-25 00:59:00 to 2024-06-23 01:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| SensorName | nvarchar | NO | 255 | — |
| ForecastValue | float | NO | 53 | — |
| CiLower | float | YES | 53 | — |
| CiUpper | float | YES | 53 | — |
| ForecastStd | float | YES | 53 | — |
| Method | nvarchar | NO | 50 | — |
| RegimeLabel | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (getdate()) |

### Top 10 Records

| RunID | EquipID | Timestamp | SensorName | ForecastValue | CiLower | CiUpper | ForecastStd | Method | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | ACTTBTEMP1_rz | -8.251665119567933 | -18.923939858205756 | 2.4206096190698876 | 10.672274738637821 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | B1RADVIBX_skew | -1.3591382303308908 | -2.402591027668833 | -0.3156854329929484 | 1.0434527973379424 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | B1RADVIBY_skew | -1.3458651692387305 | -2.444114826998942 | -0.24761551147851857 | 1.0982496577602119 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | B2RADVIBX_skew | -1.3411161231335302 | -2.7351519522975822 | 0.05291970603052176 | 1.394035829164052 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | B2RADVIBY_rz | -29.965630845972797 | -42.14891803542606 | -17.782343656519537 | 12.18328718945326 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | B2VIB1_kurt | -0.4096786019281591 | -2.4636462852670746 | 1.6442890814107562 | 2.0539676833389153 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | B2VIB1_skew | -1.1908835006985543 | -2.0655891139089424 | -0.3161778874881661 | 0.8747056132103882 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | B2VIB2_kurt | -0.45673957579232494 | -2.3059240926722295 | 1.3924449410875797 | 1.8491845168799046 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | B2VIB2_skew | -1.1246328691248935 | -1.9214589195475198 | -0.32780681870226724 | 0.7968260504226262 | LinearTrend | NULL |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | 2023-12-24 02:59:00 | INACTTBTEMP1_skew | -1.3843187393189471 | -2.974463627845629 | 0.20582614920773468 | 1.5901448885266818 | LinearTrend | NULL |

### Bottom 10 Records

| RunID | EquipID | Timestamp | SensorName | ForecastValue | CiLower | CiUpper | ForecastStd | Method | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | TURBAXDISP1_slope | 0.9819457936786957 | -0.7034418142805223 | 2.6673334016379138 | 1.685387607959218 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | LOTEMP1_slope | -0.5336339293574968 | -2.345632754335652 | 1.2783648956206586 | 1.8119988249781553 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | INACTTBTEMP1_slope | -2.0540434986904783 | -4.6239727417685454 | 0.5158857443875893 | 2.5699292430780676 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | B2TEMP1_slope | -3.1088252229422104 | -7.5938175085421165 | 1.3761670626576952 | 4.484992285599906 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | B2RADVIBY_slope | -0.05760751953515236 | -0.1264058447210703 | 0.011190805650765592 | 0.06879832518591796 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | B2RADVIBX_slope | -0.26313321730669803 | -0.32267459209709404 | -0.20359184251630202 | 0.05954137479039601 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | B1VIB2_slope | -0.0019384377836505675 | -0.009434874157130076 | 0.0055579985898289415 | 0.007496436373479509 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | B1VIB1_slope | -0.0021202188598145634 | -0.00970279080040483 | 0.0054623530807757035 | 0.007582571940590267 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | B1TEMP1_slope | -5.242229119654495 | -9.836563736900771 | -0.6478945024082181 | 4.594334617246277 | LinearTrend | NULL |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2024-01-21 02:59:00 | ACTTBTEMP1_slope | 3.1302479452733323 | -0.6961585793342522 | 6.956654469880917 | 3.8264065246075845 | LinearTrend | NULL |

---


## dbo.ACM_SensorForecast_TS

**Primary Key:** RunID, EquipID, SensorName, Timestamp  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| SensorName | nvarchar | NO | 255 | — |
| Timestamp | datetime2 | NO | — | — |
| ForecastValue | float | NO | 53 | — |
| CiLower | float | YES | 53 | — |
| CiUpper | float | YES | 53 | — |
| ForecastStd | float | YES | 53 | — |
| Method | nvarchar | NO | 50 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_SensorHotspotTimeline

**Primary Key:** No primary key  
**Row Count:** 12,807  
**Date Range:** 2023-10-18 00:30:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| SensorName | nvarchar | NO | 255 | — |
| Rank | int | NO | 10 | — |
| AbsZ | float | NO | 53 | — |
| SignedZ | float | NO | 53 | — |
| Value | float | NO | 53 | — |
| Level | nvarchar | NO | 50 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Timestamp | SensorName | Rank | AbsZ | SignedZ | Value | Level | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-18 00:30:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1 | 1.5351 | 1.5351 | 1.4299999475479126 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 2 | 1.521 | 1.521 | 52.47999954223633 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:00:00 | DEMO.SIM.06I03_1FD Fan Motor Current | 1 | 1.6143 | 1.6143 | 45.04999923706055 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 2 | 1.5804 | 1.5804 | 52.93000030517578 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:30:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1 | 1.6566 | 1.6566 | 39.380001068115234 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:30:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 2 | 1.5958 | 1.5958 | 53.529998779296875 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 02:00:00 | DEMO.SIM.06I03_1FD Fan Motor Current | 1 | 2.1809 | 2.1809 | 47.16999816894531 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 02:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 2 | 1.9066 | 1.9066 | 55.400001525878906 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 02:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 3 | 1.7914 | 1.7914 | 1.5199999809265137 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 02:30:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1 | 1.8431 | 1.8431 | 54.560001373291016 | WARN | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |

### Bottom 10 Records

| Timestamp | SensorName | Rank | AbsZ | SignedZ | Value | Level | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1 | 1.4742 | 1.4742 | 48.11000061035156 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 2 | 1.3954 | 1.3954 | 48.119998931884766 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 23:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 3 | 1.3614 | 1.3614 | 1.3300000429153442 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:30:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1 | 1.421 | 1.421 | 47.400001525878906 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1 | 1.6603 | 1.6603 | 1.4900000095367432 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 2 | 1.4454 | 1.4454 | 48.93000030517578 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 22:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 3 | 1.3685 | 1.3685 | 46.70000076293945 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 21:30:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1 | 1.3183 | 1.3183 | 46.029998779296875 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 21:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1 | 1.5108 | 1.5108 | 1.409999966621399 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-14 21:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 2 | 1.319 | 1.319 | 46.040000915527344 | WARN | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |

---


## dbo.ACM_SensorHotspots

**Primary Key:** No primary key  
**Row Count:** 163  
**Date Range:** 2023-10-18 03:00:00 to 2025-09-13 00:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| SensorName | nvarchar | NO | 255 | — |
| MaxTimestamp | datetime2 | NO | — | — |
| LatestTimestamp | datetime2 | NO | — | — |
| MaxAbsZ | float | NO | 53 | — |
| MaxSignedZ | float | NO | 53 | — |
| LatestAbsZ | float | NO | 53 | — |
| LatestSignedZ | float | NO | 53 | — |
| ValueAtPeak | float | NO | 53 | — |
| LatestValue | float | NO | 53 | — |
| TrainMean | float | NO | 53 | — |
| TrainStd | float | NO | 53 | — |
| AboveWarnCount | int | NO | 10 | — |
| AboveAlertCount | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| FailureContribution | float | YES | 53 | — |
| ZScoreAtFailure | float | YES | 53 | — |
| AlertCount | int | YES | 10 | — |

### Top 10 Records

| SensorName | MaxTimestamp | LatestTimestamp | MaxAbsZ | MaxSignedZ | LatestAbsZ | LatestSignedZ | ValueAtPeak | LatestValue | TrainMean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 2023-10-18 03:00:00 | 2023-10-20 00:00:00 | 1.8696 | 1.8696 | 1.7012 | -1.7012 | 54.66999816894531 | 39.79999923706055 | 46.88443374633789 |
| DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | 2023-10-18 04:00:00 | 2023-10-20 00:00:00 | 2.1789 | 2.1789 | 1.2043 | -1.2043 | 65.16999816894531 | 59.7599983215332 | 61.685768127441406 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2023-10-18 05:00:00 | 2023-10-20 00:00:00 | 2.2471 | 2.2471 | 1.2272 | -1.2272 | 1.6799999475479126 | 0.46000000834465027 | 0.8909279108047485 |
| DEMO.SIM.06I03_1FD Fan Motor Current | 2023-10-18 07:30:00 | 2023-10-20 00:00:00 | 2.5337 | 2.5337 | 0.5131 | -0.5131 | 48.4900016784668 | 37.09000015258789 | 39.0099983215332 |
| DEMO.SIM.06G31_1FD Fan Damper Position | 2023-10-18 07:30:00 | 2023-10-20 00:00:00 | 1.9779 | 1.9779 | 0.7487 | -0.7487 | 55.939998626708984 | 35.290000915527344 | 40.96041488647461 |
| DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | 2023-10-18 07:30:00 | 2023-10-20 00:00:00 | 1.9034 | 1.9034 | 0.6684 | -0.6684 | 443.2200012207031 | 289.95001220703125 | 329.7829895019531 |
| DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 2023-10-18 15:30:00 | 2023-10-20 00:00:00 | 1.8186 | -1.8186 | 0.2495 | -0.2495 | 200.16000366210938 | 308.8299865722656 | 326.10980224609375 |
| DEMO.SIM.06T34_1FD Fan Outlet Termperature | 2023-10-19 07:30:00 | 2023-10-20 00:00:00 | 1.8467 | 1.8467 | 0.2004 | 0.2004 | 39.81999969482422 | 29.920000076293945 | 28.71474266052246 |
| DEMO.SIM.06T31_1FD Fan Inlet Temperature | 2023-10-19 21:30:00 | 2023-10-20 00:00:00 | 1.9227 | -1.9227 | 0.3876 | -0.3876 | 25.18000030517578 | 31.270000457763672 | 32.80783462524414 |
| INACTTBTEMP1 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 4.0669 | -4.0669 | 0.1117 | -0.1117 | 89.56571960449219 | 91.55691528320312 | 91.61314392089844 |

### Bottom 10 Records

| SensorName | MaxTimestamp | LatestTimestamp | MaxAbsZ | MaxSignedZ | LatestAbsZ | LatestSignedZ | ValueAtPeak | LatestValue | TrainMean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEMO.SIM.06T34_1FD Fan Outlet Termperature | 2025-09-13 00:30:00 | 2025-09-14 23:00:00 | 1.8647 | 1.8647 | 0.8879 | 0.8879 | 43.25 | 33.2400016784668 | 24.141145706176758 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-09-12 00:00:00 | 2025-09-14 23:00:00 | 3.3416 | 3.3416 | 1.3614 | 1.3614 | 2.390000104904175 | 1.3300000429153442 | 0.6012569665908813 |
| DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 2025-09-11 18:30:00 | 2025-09-14 23:00:00 | 1.6602 | -1.6602 | 1.2233 | 1.2233 | -61.099998474121094 | 397.95001220703125 | 203.19981384277344 |
| DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | 2025-09-11 17:00:00 | 2025-09-14 23:00:00 | 1.6325 | -1.6325 | 1.1755 | 1.1755 | -63.029998779296875 | 383.5799865722656 | 196.6173553466797 |
| DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | 2025-09-11 16:30:00 | 2025-09-14 23:00:00 | 1.7369 | -1.7369 | 0.7682 | 0.7682 | -0.1599999964237213 | 63.439998626708984 | 43.93709182739258 |
| DEMO.SIM.06T31_1FD Fan Inlet Temperature | 2025-09-11 15:00:00 | 2025-09-14 23:00:00 | 1.8315 | -1.8315 | 1.4742 | 1.4742 | 4.019999980926514 | 48.11000061035156 | 28.4476261138916 |
| DEMO.SIM.06I03_1FD Fan Motor Current | 2025-09-11 03:00:00 | 2025-09-14 23:00:00 | 1.629 | -1.629 | 1.0604 | 1.0604 | 0.11999999731779099 | 45.2400016784668 | 27.44955062866211 |
| DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 2025-09-11 00:00:00 | 2025-09-14 23:00:00 | 1.5713 | -1.5713 | 1.2651 | 1.2651 | 9.710000038146973 | 55.02000045776367 | 34.8105583190918 |
| DEMO.SIM.06G31_1FD Fan Damper Position | 2025-08-15 20:30:00 | 2025-09-14 23:00:00 | 1.9613 | 1.9613 | 1.3954 | 1.3954 | 57.290000915527344 | 48.119998931884766 | 25.508378982543945 |
| DEMO.SIM.06I03_1FD Fan Motor Current | 2025-06-12 23:00:00 | 2025-06-15 23:30:00 | 1.9623 | -1.9623 | 0.5277 | 0.5277 | 0.11999999731779099 | 38.720001220703125 | 30.539199829101562 |

---


## dbo.ACM_SensorNormalized_TS

**Primary Key:** Id  
**Row Count:** 110,679  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-14 23:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | bigint | NO | 19 | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime | NO | — | — |
| SensorName | nvarchar | NO | 128 | — |
| NormValue | float | YES | 53 | — |
| ZScore | float | YES | 53 | — |
| AnomalyLevel | nvarchar | YES | 16 | — |
| EpisodeActive | bit | YES | — | — |
| RunID | varchar | YES | 64 | — |
| CreatedAt | datetime | NO | — | (getdate()) |

### Top 10 Records

| Id | EquipID | Timestamp | SensorName | NormValue | ZScore | AnomalyLevel | EpisodeActive | RunID | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11575096 | 2621 | 2023-10-20 23:59:00 | TURBAXDISP1 | 2.2117412090301514 | 2.2117412090301514 | WARN | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575097 | 2621 | 2023-10-21 00:59:00 | TURBAXDISP1 | 1.6102073192596436 | 1.6102073192596436 | WARN | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575098 | 2621 | 2023-10-21 01:59:00 | TURBAXDISP1 | 1.632542610168457 | 1.632542610168457 | WARN | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575099 | 2621 | 2023-10-21 02:59:00 | TURBAXDISP1 | -0.030118007212877274 | -0.030118007212877274 | GOOD | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575100 | 2621 | 2023-10-21 03:59:00 | TURBAXDISP1 | 0.10164161026477814 | 0.10164161026477814 | GOOD | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575101 | 2621 | 2023-10-21 04:59:00 | TURBAXDISP1 | 0.8670921325683594 | 0.8670921325683594 | GOOD | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575102 | 2621 | 2023-10-21 05:59:00 | TURBAXDISP1 | 0.6397144794464111 | 0.6397144794464111 | GOOD | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575103 | 2621 | 2023-10-21 06:59:00 | TURBAXDISP1 | -0.6856111288070679 | -0.6856111288070679 | GOOD | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575104 | 2621 | 2023-10-21 07:59:00 | TURBAXDISP1 | -0.6792943477630615 | -0.6792943477630615 | GOOD | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |
| 11575105 | 2621 | 2023-10-21 08:59:00 | TURBAXDISP1 | -0.5921653509140015 | -0.5921653509140015 | GOOD | False | 9b2207a9-fd0e-48dc-91ba-9941c0ec88fb | 2025-12-05 11:37:01 |

### Bottom 10 Records

| Id | EquipID | Timestamp | SensorName | NormValue | ZScore | AnomalyLevel | EpisodeActive | RunID | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13066959 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1.2651435136795044 | 1.2651435136795044 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066958 | 1 | 2025-09-14 22:30:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1.2300865650177002 | 1.2300865650177002 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066957 | 1 | 2025-09-14 22:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1.1737452745437622 | 1.1737452745437622 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066956 | 1 | 2025-09-14 21:30:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1.1449482440948486 | 1.1449482440948486 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066955 | 1 | 2025-09-14 21:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1.1543385982513428 | 1.1543385982513428 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066954 | 1 | 2025-09-14 20:30:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1.1249158382415771 | 1.1249158382415771 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066953 | 1 | 2025-09-14 20:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1.0322654247283936 | 1.0322654247283936 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066952 | 1 | 2025-09-14 19:30:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 1.0040947198867798 | 1.0040947198867798 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066951 | 1 | 2025-09-14 19:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 0.9634037613868713 | 0.9634037613868713 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |
| 13066950 | 1 | 2025-09-14 18:30:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 0.8832736015319824 | 0.8832736015319824 | GOOD | False | 46cd373a-b3e8-4aa3-bf07-987fefc1d2a1 | 2025-12-11 09:25:07 |

---


## dbo.ACM_SensorRanking

**Primary Key:** No primary key  
**Row Count:** 120  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorType | nvarchar | NO | 50 | — |
| RankPosition | int | NO | 10 | — |
| ContributionPct | float | NO | 53 | — |
| ZScore | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| DetectorType | RankPosition | ContributionPct | ZScore | RunID | EquipID |
| --- | --- | --- | --- | --- | --- |
| ar1_z | 1 | 29.8700008392334 | 0.7490874528884888 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| pca_t2_z | 2 | 17.170000076293945 | 0.43060800433158875 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| mhal_z | 3 | 12.819999694824219 | 0.3216104507446289 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| iforest_z | 4 | 12.0 | 0.30092769861221313 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| gmm_z | 5 | 7.570000171661377 | 0.18974179029464722 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| pca_spe_z | 6 | 7.449999809265137 | 0.18681998550891876 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| omr_z | 7 | 7.03000020980835 | 0.17620302736759186 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| cusum_z | 8 | 6.099999904632568 | 0.15303833782672882 | DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 |
| mhal_z | 1 | 30.799999237060547 | 5.20524787902832 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |
| gmm_z | 2 | 28.6299991607666 | 4.837921619415283 | 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 |

### Bottom 10 Records

| DetectorType | RankPosition | ContributionPct | ZScore | RunID | EquipID |
| --- | --- | --- | --- | --- | --- |
| mhal_z | 1 | 35.540000915527344 | 10.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| pca_t2_z | 2 | 35.540000915527344 | 10.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| ar1_z | 3 | 13.039999961853027 | 3.6689839363098145 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| gmm_z | 4 | 6.389999866485596 | 1.7988011837005615 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| iforest_z | 5 | 5.150000095367432 | 1.4499531984329224 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| cusum_z | 6 | 3.5799999237060547 | 1.008407711982727 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| pca_spe_z | 7 | 0.4000000059604645 | 0.11277640610933304 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| omr_z | 8 | 0.3499999940395355 | 0.09875394403934479 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| cusum_z | 1 | 33.47999954223633 | 1.6350982189178467 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |
| gmm_z | 2 | 32.380001068115234 | 1.5814913511276245 | 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 |

---


## dbo.ACM_SinceWhen

**Primary Key:** No primary key  
**Row Count:** 15  
**Date Range:** 2023-10-18 00:30:00 to 2025-12-11 09:24:37  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| AlertZone | nvarchar | NO | 50 | — |
| DurationHours | float | NO | 53 | — |
| StartTimestamp | datetime2 | NO | — | — |
| RecordCount | int | NO | 10 | — |

### Top 10 Records

| RunID | EquipID | AlertZone | DurationHours | StartTimestamp | RecordCount |
| --- | --- | --- | --- | --- | --- |
| DD26755A-6AE3-4FFC-A2BC-063332340EB9 | 1 | ALERT | 839.5 | 2024-08-25 12:00:00 | 65 |
| 4BD3930B-1CA2-41E4-AD72-0C2D772FBCED | 1 | ALERT | 791.5 | 2025-01-11 00:00:00 | 68 |
| 1C1AEA19-068C-4140-902E-206C202EC225 | 1 | ALERT | 767.5 | 2023-11-19 00:00:00 | 25 |
| 7D32992A-C9CF-4F13-A8F4-59EE346FC4E2 | 1 | ALERT | 795.5 | 2024-06-18 05:30:00 | 53 |
| BF973FFA-8011-45B8-B155-7F6AEE79667F | 2621 | ALERT | 367.0 | 2023-12-08 18:59:00 | 1 |
| D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | ALERT | 836.0 | 2025-05-12 03:30:00 | 26 |
| 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | ALERT | 428.0 | 2024-02-14 08:30:00 | 20 |
| 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | ALERT | 95.0 | 2025-09-11 00:00:00 | 16 |
| 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | ALERT | 96.0 | 2023-10-20 23:59:00 | 5 |
| E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 | ALERT | 360.0 | 2024-06-01 01:59:00 | 8 |

### Bottom 10 Records

| RunID | EquipID | AlertZone | DurationHours | StartTimestamp | RecordCount |
| --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | ALERT | 314.0 | 2024-01-01 00:59:00 | 9 |
| 3C73280A-9598-4410-BB08-E1DA79FB0675 | 1 | ALERT | 802.0 | 2024-11-05 04:00:00 | 59 |
| 7131196B-B651-4CD8-865C-D7EE8F740882 | 1 | GOOD | 0.0 | 2025-12-11 09:24:37 | 0 |
| 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 | ALERT | 47.5 | 2023-10-18 00:30:00 | 3 |
| ABC66545-7031-429F-82BC-C1F25B7850BC | 1 | ALERT | 824.0 | 2024-04-06 15:30:00 | 67 |
| E4CF6F6A-210E-4527-A300-B51E78840353 | 2621 | ALERT | 360.0 | 2024-06-01 01:59:00 | 8 |
| 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 | ALERT | 96.0 | 2023-10-20 23:59:00 | 5 |
| 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 | ALERT | 95.0 | 2025-09-11 00:00:00 | 16 |
| 45DA4AFC-A1FF-48BB-B696-9855D1D43B59 | 1 | ALERT | 428.0 | 2024-02-14 08:30:00 | 20 |
| D855F3D7-B588-4BF3-9DCE-97DA7A53C6A5 | 1 | ALERT | 836.0 | 2025-05-12 03:30:00 | 26 |

---


## dbo.ACM_TagEquipmentMap

**Primary Key:** TagID  
**Row Count:** 25  
**Date Range:** 2025-12-01 04:53:29 to 2025-12-01 04:53:29  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| TagID | int | NO | 10 | — |
| TagName | varchar | NO | 255 | — |
| EquipmentName | varchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| TagDescription | varchar | YES | 500 | — |
| TagUnit | varchar | YES | 50 | — |
| TagType | varchar | YES | 50 | — |
| IsActive | bit | YES | — | ((1)) |
| CreatedAt | datetime2 | YES | — | (getutcdate()) |
| UpdatedAt | datetime2 | YES | — | (getutcdate()) |

### Top 10 Records

| TagID | TagName | EquipmentName | EquipID | TagDescription | TagUnit | TagType | IsActive | CreatedAt | UpdatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 51 | DEMO.SIM.06G31_1FD Fan Damper Position | FD_FAN | 1 | FD Fan Damper Position | % | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 52 | DEMO.SIM.06I03_1FD Fan Motor Current | FD_FAN | 1 | FD Fan Motor Current | A | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 53 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | FD_FAN | 1 | FD Fan Outlet Pressure | inH2O | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 54 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | FD_FAN | 1 | FD Fan Inlet Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 55 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | FD_FAN | 1 | FD Fan Bearing Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 56 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | FD_FAN | 1 | FD Fan Winding Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 57 | DEMO.SIM.06T34_1FD Fan Outlet Termperature | FD_FAN | 1 | FD Fan Outlet Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 58 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | FD_FAN | 1 | FD Fan Left Inlet Flow | KPPH | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 59 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | FD_FAN | 1 | FD Fan Right Inlet Flow | KPPH | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 60 | DWATT | GAS_TURBINE | 2621 | Generator Power Output | MW | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |

### Bottom 10 Records

| TagID | TagName | EquipmentName | EquipID | TagDescription | TagUnit | TagType | IsActive | CreatedAt | UpdatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 75 | LOTEMP1 | GAS_TURBINE | 2621 | Lube Oil Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 74 | INACTTBTEMP1 | GAS_TURBINE | 2621 | Inactive Turbine Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 73 | ACTTBTEMP1 | GAS_TURBINE | 2621 | Active Turbine Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 72 | B2TEMP1 | GAS_TURBINE | 2621 | Bearing 2 Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 71 | B1TEMP1 | GAS_TURBINE | 2621 | Bearing 1 Temperature | F | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 70 | TURBAXDISP2 | GAS_TURBINE | 2621 | Turbine Axial Displacement 2 | mil | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 69 | TURBAXDISP1 | GAS_TURBINE | 2621 | Turbine Axial Displacement 1 | mil | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 68 | B2RADVIBY | GAS_TURBINE | 2621 | Bearing 2 Radial Vibration Y | mil | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 67 | B2RADVIBX | GAS_TURBINE | 2621 | Bearing 2 Radial Vibration X | mil | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |
| 66 | B2VIB2 | GAS_TURBINE | 2621 | Bearing 2 Vibration Sensor 2 | mil | Analog | True | 2025-12-01 04:53:29 | 2025-12-01 04:53:29 |

---


## dbo.ACM_ThresholdCrossings

**Primary Key:** No primary key  
**Row Count:** 306  
**Date Range:** 2023-10-18 00:00:00 to 2025-09-12 07:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| DetectorType | nvarchar | NO | 50 | — |
| Threshold | float | NO | 53 | — |
| ZScore | float | NO | 53 | — |
| Direction | nvarchar | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| Timestamp | DetectorType | Threshold | ZScore | Direction | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- |
| 2023-10-18 00:00:00 | fused | 2.0 | 0.3995 | down | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 00:30:00 | fused | 2.0 | 2.3802 | up | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-18 01:00:00 | fused | 2.0 | 0.8502 | down | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-19 00:00:00 | fused | 2.0 | 3.9881 | up | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-19 00:30:00 | fused | 2.0 | 1.5843 | down | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-19 01:30:00 | fused | 2.0 | 2.5795 | up | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-19 02:00:00 | fused | 2.0 | 0.7102 | down | 91C95C14-74B8-43A1-892D-CC11BBEBC7CF | 1 |
| 2023-10-20 23:59:00 | fused | 2.0 | 2.3333 | up | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |
| 2023-10-21 03:59:00 | fused | 2.0 | 1.9878 | down | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |
| 2023-10-24 22:59:00 | fused | 2.0 | 2.7997 | up | 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |

### Bottom 10 Records

| Timestamp | DetectorType | Threshold | ZScore | Direction | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-09-12 07:30:00 | fused | 2.0 | 0.3211 | down | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-12 06:30:00 | fused | 2.0 | 2.0332 | up | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-12 04:30:00 | fused | 2.0 | 1.784 | down | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-12 00:00:00 | fused | 2.0 | 2.3362 | up | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-11 08:00:00 | fused | 2.0 | 1.6389 | down | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-11 07:00:00 | fused | 2.0 | 2.4161 | up | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-11 04:30:00 | fused | 2.0 | 1.485 | down | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-11 03:30:00 | fused | 2.0 | 2.6186 | up | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-11 00:30:00 | fused | 2.0 | 1.7039 | down | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |
| 2025-09-11 00:00:00 | fused | 2.0 | 2.0972 | up | 46CD373A-B3E8-4AA3-BF07-987FEFC1D2A1 | 1 |

---


## dbo.ACM_ThresholdMetadata

**Primary Key:** ThresholdID  
**Row Count:** 30  
**Date Range:** 2023-10-15 00:00:00 to 2025-07-11 00:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ThresholdID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| RegimeID | int | YES | 10 | — |
| ThresholdType | varchar | NO | 50 | — |
| ThresholdValue | float | NO | 53 | — |
| CalculationMethod | varchar | NO | 100 | — |
| SampleCount | int | YES | 10 | — |
| TrainStartTime | datetime2 | YES | — | — |
| TrainEndTime | datetime2 | YES | — | — |
| CreatedAt | datetime2 | YES | — | (getdate()) |
| ConfigSignature | varchar | YES | 32 | — |
| IsActive | bit | YES | — | ((1)) |
| Notes | varchar | YES | 500 | — |

### Top 10 Records

| ThresholdID | EquipID | RegimeID | ThresholdType | ThresholdValue | CalculationMethod | SampleCount | TrainStartTime | TrainEndTime | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3534 | 2621 | NULL | fused_alert_z | 3.0 | quantile_0.997 | 97 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 2025-12-05 11:36:56 |
| 3535 | 2621 | NULL | fused_warn_z | 1.5 | quantile_0.997 | 97 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 2025-12-05 11:36:56 |
| 3538 | 2621 | NULL | fused_alert_z | 1.8977547883987427 | quantile_0.997 | 842 | 2023-10-15 00:00:00 | 2023-12-24 01:59:00 | 2025-12-05 11:37:28 |
| 3539 | 2621 | NULL | fused_warn_z | 0.9488773941993713 | quantile_0.997 | 842 | 2023-10-15 00:00:00 | 2023-12-24 01:59:00 | 2025-12-05 11:37:28 |
| 3540 | 2621 | NULL | fused_alert_z | 2.4079856872558594 | quantile_0.997 | 505 | 2023-12-24 02:59:00 | 2024-01-14 02:59:00 | 2025-12-05 11:37:55 |
| 3541 | 2621 | NULL | fused_warn_z | 1.2039928436279297 | quantile_0.997 | 505 | 2023-12-24 02:59:00 | 2024-01-14 02:59:00 | 2025-12-05 11:37:55 |
| 3542 | 2621 | NULL | fused_alert_z | 3.054851531982422 | quantile_0.997 | 362 | 2024-05-17 00:00:00 | 2024-06-16 01:59:00 | 2025-12-05 11:38:22 |
| 3543 | 2621 | NULL | fused_warn_z | 1.527425765991211 | quantile_0.997 | 362 | 2024-05-17 00:00:00 | 2024-06-16 01:59:00 | 2025-12-05 11:38:22 |
| 3892 | 1 | NULL | fused_alert_z | 3.0 | quantile_0.997 | 97 | 2023-10-18 00:00:00 | 2023-10-20 00:00:00 | 2025-12-11 09:21:15 |
| 3893 | 1 | NULL | fused_warn_z | 1.5 | quantile_0.997 | 97 | 2023-10-18 00:00:00 | 2023-10-20 00:00:00 | 2025-12-11 09:21:15 |

### Bottom 10 Records

| ThresholdID | EquipID | RegimeID | ThresholdType | ThresholdValue | CalculationMethod | SampleCount | TrainStartTime | TrainEndTime | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3913 | 1 | NULL | fused_warn_z | 1.3020668029785156 | quantile_0.997 | 359 | 2025-07-11 00:00:00 | 2025-09-14 23:00:00 | 2025-12-11 09:25:04 |
| 3912 | 1 | NULL | fused_alert_z | 2.6041336059570312 | quantile_0.997 | 359 | 2025-07-11 00:00:00 | 2025-09-14 23:00:00 | 2025-12-11 09:25:04 |
| 3911 | 1 | NULL | fused_warn_z | 1.3392335176467896 | quantile_0.997 | 476 | 2025-05-11 00:00:00 | 2025-06-15 23:30:00 | 2025-12-11 09:24:49 |
| 3910 | 1 | NULL | fused_alert_z | 2.678467035293579 | quantile_0.997 | 476 | 2025-05-11 00:00:00 | 2025-06-15 23:30:00 | 2025-12-11 09:24:49 |
| 3909 | 1 | NULL | fused_warn_z | 0.7252260446548462 | quantile_0.997 | 441 | 2025-02-20 00:00:00 | 2025-04-14 23:30:00 | 2025-12-11 09:24:34 |
| 3908 | 1 | NULL | fused_alert_z | 1.4504520893096924 | quantile_0.997 | 441 | 2025-02-20 00:00:00 | 2025-04-14 23:30:00 | 2025-12-11 09:24:34 |
| 3907 | 1 | NULL | fused_warn_z | 1.592867136001587 | quantile_0.997 | 1258 | 2024-12-08 14:30:00 | 2025-02-12 23:30:00 | 2025-12-11 09:24:15 |
| 3906 | 1 | NULL | fused_alert_z | 3.185734272003174 | quantile_0.997 | 1258 | 2024-12-08 14:30:00 | 2025-02-12 23:30:00 | 2025-12-11 09:24:15 |
| 3905 | 1 | NULL | fused_warn_z | 1.415886402130127 | quantile_0.997 | 1346 | 2024-09-29 12:00:00 | 2024-12-08 14:00:00 | 2025-12-11 09:23:45 |
| 3904 | 1 | NULL | fused_alert_z | 2.831772804260254 | quantile_0.997 | 1346 | 2024-09-29 12:00:00 | 2024-12-08 14:00:00 | 2025-12-11 09:23:45 |

---


## dbo.Equipment

**Primary Key:** EquipID  
**Row Count:** 3  
**Date Range:** 2025-01-01 00:00:00 to 2025-01-01 00:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| EquipCode | nvarchar | NO | 100 | — |
| EquipName | nvarchar | YES | 200 | — |
| Area | nvarchar | YES | 100 | — |
| Unit | nvarchar | YES | 100 | — |
| Status | tinyint | YES | 3 | — |
| CommissionDate | datetime2 | YES | — | — |
| CreatedAtUTC | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| EquipID | EquipCode | EquipName | Area | Unit | Status | CommissionDate | CreatedAtUTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | * | Default/Wildcard Config | Global | All Plants | 1 | 2025-01-01 00:00:00 | 2025-11-13 09:21:41 |
| 1 | FD_FAN | Forced Draft Fan | Boiler Section | Plant A | 1 | 2025-01-01 00:00:00 | 2025-11-13 07:54:36 |
| 2621 | GAS_TURBINE | Gas Turbine Generator | Power Generation | Plant A | 1 | 2025-01-01 00:00:00 | 2025-11-13 07:54:36 |

---


## dbo.FD_FAN_Data

**Primary Key:** EntryDateTime  
**Row Count:** 17,499  
**Date Range:** 2023-10-15 00:00:00 to 2025-09-14 23:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| DEMO.SIM.06G31_1FD Fan Damper Position | float | YES | 53 | — |
| DEMO.SIM.06I03_1FD Fan Motor Current | float | YES | 53 | — |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | float | YES | 53 | — |
| DEMO.SIM.06T31_1FD Fan Inlet Temperature | float | YES | 53 | — |
| DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | float | YES | 53 | — |
| DEMO.SIM.06T33-1_1FD Fan Winding Temperature | float | YES | 53 | — |
| DEMO.SIM.06T34_1FD Fan Outlet Termperature | float | YES | 53 | — |
| DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | float | YES | 53 | — |
| DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | float | YES | 53 | — |
| LoadedAt | datetime2 | YES | — | (getutcdate()) |

### Top 10 Records

| EntryDateTime | DEMO.SIM.06G31_1FD Fan Damper Position | DEMO.SIM.06I03_1FD Fan Motor Current | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | DEMO.SIM.06T31_1FD Fan Inlet Temperature | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | DEMO.SIM.06T34_1FD Fan Outlet Termperature | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-15 00:00:00 | 32.78 | 35.86 | 0.52 | 36.37 | 61.05 | 44.09 | 22.0 | 294.7 | 301.38 |
| 2023-10-15 00:30:00 | 31.6 | 35.03 | 0.39 | 36.46 | 60.42 | 44.56 | 22.81 | 264.18 | 272.99 |
| 2023-10-15 01:00:00 | 30.18 | 35.1 | 0.57 | 37.24 | 61.17 | 45.0 | 23.01 | 263.76 | 269.35 |
| 2023-10-15 01:30:00 | 27.8 | 34.89 | 0.23 | 36.93 | 61.13 | 45.63 | 24.24 | 245.16 | 251.28 |
| 2023-10-15 02:00:00 | 28.22 | 34.33 | 0.28 | 37.42 | 62.29 | 45.92 | 25.56 | 239.65 | 235.69 |
| 2023-10-15 02:30:00 | 30.71 | 35.08 | 0.6 | 38.89 | 61.96 | 46.62 | 26.04 | 273.33 | 272.12 |
| 2023-10-15 03:00:00 | 31.8 | 36.22 | 0.54 | 39.0 | 62.04 | 47.15 | 27.25 | 290.31 | 287.79 |
| 2023-10-15 03:30:00 | 33.36 | 36.47 | 0.57 | 39.83 | 62.56 | 47.99 | 27.8 | 305.63 | 298.89 |
| 2023-10-15 04:00:00 | 29.94 | 35.16 | 0.65 | 40.24 | 62.82 | 48.34 | 28.83 | 271.35 | 271.9 |
| 2023-10-15 04:30:00 | 28.39 | 34.86 | 0.68 | 40.89 | 63.11 | 48.71 | 28.19 | 261.49 | 261.95 |

### Bottom 10 Records

| EntryDateTime | DEMO.SIM.06G31_1FD Fan Damper Position | DEMO.SIM.06I03_1FD Fan Motor Current | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | DEMO.SIM.06T31_1FD Fan Inlet Temperature | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | DEMO.SIM.06T34_1FD Fan Outlet Termperature | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 23:30:00 | 52.05 | 47.83 | 1.38 | 48.64 | 63.56 | 55.66 | 34.29 | 399.31 | 417.34 |
| 2025-09-14 23:00:00 | 48.12 | 45.24 | 1.33 | 48.11 | 63.44 | 55.02 | 33.24 | 383.58 | 397.95 |
| 2025-09-14 22:30:00 | 46.54 | 44.42 | 1.19 | 47.4 | 63.45 | 54.46 | 32.21 | 373.67 | 393.27 |
| 2025-09-14 22:00:00 | 48.93 | 45.3 | 1.49 | 46.7 | 62.59 | 53.56 | 32.26 | 379.2 | 396.8 |
| 2025-09-14 21:30:00 | 45.28 | 42.38 | 1.12 | 46.03 | 62.87 | 53.1 | 30.98 | 347.54 | 359.95 |
| 2025-09-14 21:00:00 | 45.39 | 42.78 | 1.41 | 46.04 | 62.29 | 53.25 | 30.63 | 365.25 | 378.35 |
| 2025-09-14 20:30:00 | 50.7 | 46.65 | 1.45 | 44.96 | 62.04 | 52.78 | 30.08 | 399.45 | 418.41 |
| 2025-09-14 20:00:00 | 48.6 | 45.26 | 1.45 | 44.28 | 61.37 | 51.3 | 28.84 | 381.99 | 395.47 |
| 2025-09-14 19:30:00 | 44.23 | 41.86 | 1.49 | 43.57 | 60.16 | 50.85 | 28.18 | 358.22 | 373.61 |
| 2025-09-14 19:00:00 | 46.91 | 44.4 | 1.61 | 42.74 | 60.53 | 50.2 | 26.88 | 375.07 | 390.55 |

---


## dbo.GAS_TURBINE_Data

**Primary Key:** EntryDateTime  
**Row Count:** 2,911  
**Date Range:** 2023-10-15 00:00:00 to 2024-06-16 01:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| DWATT | float | YES | 53 | — |
| B1VIB1 | float | YES | 53 | — |
| B1VIB2 | float | YES | 53 | — |
| B1RADVIBX | float | YES | 53 | — |
| B1RADVIBY | float | YES | 53 | — |
| B2VIB1 | float | YES | 53 | — |
| B2VIB2 | float | YES | 53 | — |
| B2RADVIBX | float | YES | 53 | — |
| B2RADVIBY | float | YES | 53 | — |
| TURBAXDISP1 | float | YES | 53 | — |
| TURBAXDISP2 | float | YES | 53 | — |
| B1TEMP1 | float | YES | 53 | — |
| B2TEMP1 | float | YES | 53 | — |
| ACTTBTEMP1 | float | YES | 53 | — |
| INACTTBTEMP1 | float | YES | 53 | — |
| LOTEMP1 | float | YES | 53 | — |
| LoadedAt | datetime2 | YES | — | (getutcdate()) |

### Top 10 Records

| EntryDateTime | DWATT | B1VIB1 | B1VIB2 | B1RADVIBX | B1RADVIBY | B2VIB1 | B2VIB2 | B2RADVIBX | B2RADVIBY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-15 00:00:00 | 0.046386719 | 0.003474951 | 0.003582239 | 0.088310242 | 0.073814392 | 0.00166893 | 0.001722574 | 0.102710724 | 0.080680847 |
| 2023-10-15 01:00:00 | 0.048736572 | 0.003510714 | 0.003516674 | 0.089359283 | 0.078964233 | 0.001817942 | 0.002282858 | 0.092029572 | 0.074386597 |
| 2023-10-15 02:00:00 | 0.058380127 | 0.003516674 | 0.003367662 | 0.064277649 | 0.08058548 | 0.001698732 | 0.001859665 | 0.089359283 | 0.07276535 |
| 2023-10-15 03:00:00 | 0.056243896 | 0.003319979 | 0.003421307 | 0.096035004 | 0.103378296 | 0.001847744 | 0.001585484 | 0.132274628 | 0.107097626 |
| 2023-10-15 04:00:00 | 0.046020508 | 0.003510714 | 0.003546476 | 0.154399872 | 0.11920929 | 0.00193119 | 0.001746416 | 0.091648102 | 0.066566467 |
| 2023-10-15 05:00:00 | 0.05670166 | 0.003254414 | 0.003486872 | 0.121307373 | 0.084114075 | 0.001633167 | 0.001603365 | 0.071239471 | 0.0623703 |
| 2023-10-15 06:00:00 | 0.050323486 | 0.003314018 | 0.00346899 | 0.225830078 | 0.125312805 | 0.002282858 | 0.002169609 | 0.082397461 | 0.074291229 |
| 2023-10-15 07:00:00 | 0.04699707 | 0.003361702 | 0.00321269 | 0.080394745 | 0.089740753 | 0.001698732 | 0.001680851 | 0.078964233 | 0.091552734 |
| 2023-10-15 08:00:00 | 0.037811279 | 0.003302097 | 0.003272295 | 0.083351135 | 0.112819672 | 0.001722574 | 0.001704693 | 0.123596191 | 0.109767914 |
| 2023-10-15 09:00:00 | 0.025543213 | 0.003635883 | 0.003671646 | 0.121593475 | 0.103569031 | 0.001752377 | 0.001722574 | 0.095081329 | 0.086212158 |

### Bottom 10 Records

| EntryDateTime | DWATT | B1VIB1 | B1VIB2 | B1RADVIBX | B1RADVIBY | B2VIB1 | B2VIB2 | B2RADVIBX | B2RADVIBY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-06-16 01:59:00 | 0.032104492 | 0.004428625 | 0.004589558 | 0.102615356 | 0.086212158 | 0.001949072 | 0.002008677 | 0.090408325 | 0.135612488 |
| 2024-06-16 00:59:00 | 0.035583496 | 0.004279613 | 0.004172325 | 0.120735168 | 0.075054169 | 0.002062321 | 0.002026558 | 0.067615509 | 0.069141388 |
| 2024-06-15 23:59:00 | 0.032012939 | 0.004637241 | 0.004476309 | 0.09727478 | 0.063991547 | 0.002026558 | 0.001955032 | 0.098609924 | 0.094413757 |
| 2024-06-15 22:59:00 | 0.033325195 | 0.00398159 | 0.004124641 | 0.101089478 | 0.080966949 | 0.001943111 | 0.001806021 | 0.108337402 | 0.105381012 |
| 2024-06-15 21:59:00 | 0.030883789 | 0.004047155 | 0.004076958 | 0.125312805 | 0.113677979 | 0.002080202 | 0.00205636 | 0.074005127 | 0.077819824 |
| 2024-06-15 20:59:00 | 154.3208008 | 0.207543373 | 0.206416845 | 3.847885132 | 3.883361816 | 0.089865923 | 0.08701086 | 1.396083832 | 1.22385025 |
| 2024-06-15 19:59:00 | 160.8225708 | 0.213122368 | 0.2125144 | 3.946208954 | 3.966903687 | 0.087571144 | 0.085371733 | 1.428318024 | 1.244831085 |
| 2024-06-15 18:59:00 | 159.7585144 | 0.221902132 | 0.220447779 | 4.005908966 | 4.01468277 | 0.088727474 | 0.08649826 | 1.43699646 | 1.271915436 |
| 2024-06-15 17:59:00 | 157.7884216 | 0.211650133 | 0.207263231 | 3.846263885 | 3.898620605 | 0.086086988 | 0.08571744 | 1.389503479 | 1.257705688 |
| 2024-06-15 16:59:00 | 156.4640808 | 0.214719772 | 0.211650133 | 3.853225708 | 3.888607025 | 0.098234415 | 0.089746714 | 1.364040375 | 1.244735718 |

---


## dbo.ModelRegistry

**Primary Key:** ModelType, EquipID, Version  
**Row Count:** 105  
**Date Range:** 2025-12-05 06:06:56 to 2025-12-11 03:55:04  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ModelType | varchar | NO | 16 | — |
| EquipID | int | NO | 10 | — |
| Version | int | NO | 10 | — |
| EntryDateTime | datetime2 | NO | — | (sysutcdatetime()) |
| ParamsJSON | nvarchar | YES | -1 | — |
| StatsJSON | nvarchar | YES | -1 | — |
| RunID | uniqueidentifier | YES | — | — |
| ModelBytes | varbinary | YES | -1 | — |

### Top 10 Records

| ModelType | EquipID | Version | EntryDateTime | ParamsJSON | StatsJSON | RunID | ModelBytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ar1_params | 1 | 1 | 2025-12-11 03:51:14 | {"n_sensors": 90, "mean_autocorr": 2084.939, "mean_residual_std": 1390.7155, "params_count": 180} | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 7167 bytes> |
| feature_medians | 1 | 1 | 2025-12-11 03:51:14 | NULL | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 12288 bytes> |
| gmm_model | 1 | 1 | 2025-12-11 03:51:14 | {"n_components": 3, "covariance_type": "diag", "bic": 9043485744053.15, "aic": 9043485742657.66, ... | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 9718 bytes> |
| iforest_model | 1 | 1 | 2025-12-11 03:51:14 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 693657 bytes> |
| mhal_params | 1 | 1 | 2025-12-11 03:51:14 | NULL | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 65826 bytes> |
| omr_model | 1 | 1 | 2025-12-11 03:51:14 | NULL | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 115328 bytes> |
| pca_model | 1 | 1 | 2025-12-11 03:51:14 | {"n_components": 5, "variance_ratio_sum": 0.7681, "variance_ratio_first_component": 0.2799, "vari... | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 5375 bytes> |
| ar1_params | 1 | 2 | 2025-12-11 03:51:30 | {"n_sensors": 90, "mean_autocorr": 1372.1481, "mean_residual_std": 1086.2972, "params_count": 180} | {"train_rows": 503, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 7167 bytes> |
| feature_medians | 1 | 2 | 2025-12-11 03:51:30 | NULL | {"train_rows": 503, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 12288 bytes> |
| gmm_model | 1 | 2 | 2025-12-11 03:51:30 | {"n_components": 3, "covariance_type": "diag", "bic": 10927079700552.87, "aic": 10927079698265.31... | {"train_rows": 503, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 9870 bytes> |

### Bottom 10 Records

| ModelType | EquipID | Version | EntryDateTime | ParamsJSON | StatsJSON | RunID | ModelBytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pca_model | 2621 | 4 | 2025-12-05 06:08:22 | {"n_components": 5, "variance_ratio_sum": 0.8096, "variance_ratio_first_component": 0.2836, "vari... | {"train_rows": 361, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 8735 bytes> |
| omr_model | 2621 | 4 | 2025-12-05 06:08:22 | NULL | {"train_rows": 361, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 317873 bytes> |
| mhal_params | 2621 | 4 | 2025-12-05 06:08:22 | NULL | {"train_rows": 361, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 206386 bytes> |
| iforest_model | 2621 | 4 | 2025-12-05 06:08:22 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 361, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 1632825 bytes> |
| gmm_model | 2621 | 4 | 2025-12-05 06:08:22 | {"n_components": 3, "covariance_type": "diag", "bic": 48226091727766.22, "aic": 48226091724025.12... | {"train_rows": 361, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 16495 bytes> |
| feature_medians | 2621 | 4 | 2025-12-05 06:08:22 | NULL | {"train_rows": 361, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 9864 bytes> |
| ar1_params | 2621 | 4 | 2025-12-05 06:08:22 | {"n_sensors": 160, "mean_autocorr": 1801.7668, "mean_residual_std": 1031.1214, "params_count": 320} | {"train_rows": 361, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 7744 bytes> |
| pca_model | 2621 | 3 | 2025-12-05 06:07:54 | {"n_components": 5, "variance_ratio_sum": 0.7298, "variance_ratio_first_component": 0.32, "varian... | {"train_rows": 300, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 8735 bytes> |
| omr_model | 2621 | 3 | 2025-12-05 06:07:54 | NULL | {"train_rows": 300, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 308089 bytes> |
| mhal_params | 2621 | 3 | 2025-12-05 06:07:54 | NULL | {"train_rows": 300, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP... | NULL | <binary 206386 bytes> |

---

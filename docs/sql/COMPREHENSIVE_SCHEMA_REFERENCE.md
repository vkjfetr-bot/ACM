# ACM Comprehensive Database Schema Reference

_Generated automatically on 2025-12-13 12:33:17_

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
- [dbo.ACM_MultivariateForecast](#dboacmmultivariateforecast)
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
- [dbo.ACM_SensorCorrelations](#dboacmsensorcorrelations)
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
- [dbo.ELECTRIC_MOTOR_Data](#dboelectricmotordata)
- [dbo.ELECTRIC_MOTOR_Data_RAW](#dboelectricmotordataraw)
- [dbo.Equipment](#dboequipment)
- [dbo.FD_FAN_Data](#dbofdfandata)
- [dbo.GAS_TURBINE_Data](#dbogasturbinedata)
- [dbo.ModelRegistry](#dbomodelregistry)
- [dbo.WFA_TURBINE_0_Data](#dbowfaturbine0data)
- [dbo.WFA_TURBINE_10_Data](#dbowfaturbine10data)
- [dbo.WFA_TURBINE_11_Data](#dbowfaturbine11data)
- [dbo.WFA_TURBINE_13_Data](#dbowfaturbine13data)
- [dbo.WFA_TURBINE_14_Data](#dbowfaturbine14data)
- [dbo.WFA_TURBINE_17_Data](#dbowfaturbine17data)
- [dbo.WFA_TURBINE_21_Data](#dbowfaturbine21data)
- [dbo.WFA_TURBINE_22_Data](#dbowfaturbine22data)
- [dbo.WFA_TURBINE_24_Data](#dbowfaturbine24data)
- [dbo.WFA_TURBINE_25_Data](#dbowfaturbine25data)
- [dbo.WFA_TURBINE_26_Data](#dbowfaturbine26data)
- [dbo.WFA_TURBINE_38_Data](#dbowfaturbine38data)
- [dbo.WFA_TURBINE_3_Data](#dbowfaturbine3data)
- [dbo.WFA_TURBINE_40_Data](#dbowfaturbine40data)
- [dbo.WFA_TURBINE_42_Data](#dbowfaturbine42data)
- [dbo.WFA_TURBINE_45_Data](#dbowfaturbine45data)
- [dbo.WFA_TURBINE_51_Data](#dbowfaturbine51data)
- [dbo.WFA_TURBINE_68_Data](#dbowfaturbine68data)
- [dbo.WFA_TURBINE_69_Data](#dbowfaturbine69data)
- [dbo.WFA_TURBINE_71_Data](#dbowfaturbine71data)
- [dbo.WFA_TURBINE_72_Data](#dbowfaturbine72data)
- [dbo.WFA_TURBINE_73_Data](#dbowfaturbine73data)
- [dbo.WFA_TURBINE_84_Data](#dbowfaturbine84data)
- [dbo.WFA_TURBINE_92_Data](#dbowfaturbine92data)
- [dbo.WIND_TURBINE_Data](#dbowindturbinedata)


## Summary

| Table | Columns | Rows | Primary Key |
| --- | ---: | ---: | --- |
| dbo.ACM_AdaptiveConfig | 13 | 9 | ConfigID |
| dbo.ACM_AdaptiveThresholds_ByRegime | 11 | 0 | ID |
| dbo.ACM_AlertAge | 6 | 90 | — |
| dbo.ACM_Anomaly_Events | 6 | 332 | Id |
| dbo.ACM_BaselineBuffer | 7 | 546,950 | Id |
| dbo.ACM_CalibrationSummary | 10 | 720 | — |
| dbo.ACM_ColdstartState | 17 | 6 | EquipID, Stage |
| dbo.ACM_Config | 7 | 332 | ConfigID |
| dbo.ACM_ConfigHistory | 9 | 56 | ID |
| dbo.ACM_ContributionCurrent | 5 | 720 | — |
| dbo.ACM_ContributionTimeline | 5 | 214,368 | — |
| dbo.ACM_DailyFusedProfile | 9 | 557 | ID |
| dbo.ACM_DataQuality | 24 | 5,150 | — |
| dbo.ACM_DefectSummary | 12 | 90 | — |
| dbo.ACM_DefectTimeline | 10 | 2,450 | — |
| dbo.ACM_DetectorCorrelation | 7 | 2,520 | — |
| dbo.ACM_DetectorForecast_TS | 10 | 3,864 | RunID, EquipID, DetectorName, Timestamp |
| dbo.ACM_DriftEvents | 2 | 21 | — |
| dbo.ACM_DriftSeries | 4 | 30,201 | — |
| dbo.ACM_EnhancedFailureProbability_TS | 11 | 0 | RunID, EquipID, Timestamp, ForecastHorizon_Hours |
| dbo.ACM_EnhancedMaintenanceRecommendation | 13 | 0 | RunID, EquipID |
| dbo.ACM_EpisodeCulprits | 9 | 3,950 | ID |
| dbo.ACM_EpisodeDiagnostics | 13 | 353 | ID |
| dbo.ACM_EpisodeMetrics | 10 | 88 | — |
| dbo.ACM_Episodes | 8 | 88 | — |
| dbo.ACM_EpisodesQC | 10 | 88 | RecordID |
| dbo.ACM_FailureCausation | 12 | 0 | RunID, EquipID, Detector |
| dbo.ACM_FailureForecast | 9 | 117,264 | EquipID, RunID, Timestamp |
| dbo.ACM_FailureForecast_TS | 7 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_FailureHazard_TS | 8 | 2,352 | EquipID, RunID, Timestamp |
| dbo.ACM_FeatureDropLog | 8 | 20,562 | LogID |
| dbo.ACM_ForecastContext | 26 | 0 | ID |
| dbo.ACM_ForecastState | 12 | 913 | EquipID, StateVersion |
| dbo.ACM_ForecastingState | 13 | 5 | EquipID, StateVersion |
| dbo.ACM_FusionQualityReport | 9 | 900 | — |
| dbo.ACM_HealthDistributionOverTime | 12 | 11,444 | — |
| dbo.ACM_HealthForecast | 10 | 117,264 | EquipID, RunID, Timestamp |
| dbo.ACM_HealthForecast_Continuous | 8 | 3,988 | EquipID, Timestamp, SourceRunID |
| dbo.ACM_HealthForecast_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_HealthHistogram | 5 | 900 | — |
| dbo.ACM_HealthTimeline | 8 | 30,201 | — |
| dbo.ACM_HealthZoneByPeriod | 9 | 1,671 | — |
| dbo.ACM_HistorianData | 7 | 204,067 | DataID |
| dbo.ACM_MaintenanceRecommendation | 8 | 0 | RunID, EquipID |
| dbo.ACM_MultivariateForecast | 12 | 0 | ForecastID |
| dbo.ACM_OMRContributionsLong | 8 | 8,029,365 | — |
| dbo.ACM_OMRTimeline | 6 | 30,201 | — |
| dbo.ACM_OMR_Diagnostics | 15 | 90 | DiagnosticID |
| dbo.ACM_PCA_Loadings | 10 | 209,350 | RecordID |
| dbo.ACM_PCA_Metrics | 6 | 273 | RunID, EquipID, ComponentName, MetricType |
| dbo.ACM_PCA_Models | 12 | 82 | RecordID |
| dbo.ACM_RUL | 18 | 77 | EquipID, RunID |
| dbo.ACM_RUL_Attribution | 9 | 0 | RunID, EquipID, FailureTime, SensorName |
| dbo.ACM_RUL_ByRegime | 15 | 0 | ID |
| dbo.ACM_RUL_LearningState | 19 | 2 | EquipID |
| dbo.ACM_RUL_Summary | 15 | 0 | RunID, EquipID |
| dbo.ACM_RUL_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_RecommendedActions | 6 | 0 | RunID, EquipID, Action |
| dbo.ACM_RefitRequests | 10 | 1,499 | RequestID |
| dbo.ACM_RegimeDwellStats | 8 | 276 | — |
| dbo.ACM_RegimeHazard | 11 | 0 | ID |
| dbo.ACM_RegimeOccupancy | 5 | 276 | — |
| dbo.ACM_RegimeStability | 4 | 90 | — |
| dbo.ACM_RegimeState | 15 | 6 | EquipID, StateVersion |
| dbo.ACM_RegimeStats | 8 | 276 | — |
| dbo.ACM_RegimeTimeline | 5 | 30,201 | — |
| dbo.ACM_RegimeTransitions | 6 | 379 | — |
| dbo.ACM_Regime_Episodes | 6 | 332 | Id |
| dbo.ACM_RunLogs | 25 | 403,502 | LogID |
| dbo.ACM_RunMetadata | 12 | 715 | RunMetadataID |
| dbo.ACM_RunMetrics | 5 | 27,993 | RunID, EquipID, MetricName |
| dbo.ACM_Run_Stats | 13 | 82 | RecordID |
| dbo.ACM_Runs | 19 | 232 | RunID |
| dbo.ACM_SchemaVersion | 5 | 2 | VersionID |
| dbo.ACM_Scores_Long | 9 | 169,884 | Id |
| dbo.ACM_Scores_Wide | 15 | 30,201 | — |
| dbo.ACM_SensorAnomalyByPeriod | 11 | 4,456 | — |
| dbo.ACM_SensorCorrelations | 10 | 0 | CorrelationID |
| dbo.ACM_SensorDefects | 11 | 720 | — |
| dbo.ACM_SensorForecast | 11 | 6,720 | RunID, EquipID, Timestamp, SensorName |
| dbo.ACM_SensorForecast_TS | 10 | 0 | RunID, EquipID, SensorName, Timestamp |
| dbo.ACM_SensorHotspotTimeline | 9 | 52,565 | — |
| dbo.ACM_SensorHotspots | 18 | 1,751 | — |
| dbo.ACM_SensorNormalized_TS | 10 | 354,908 | Id |
| dbo.ACM_SensorRanking | 6 | 720 | — |
| dbo.ACM_SinceWhen | 6 | 90 | — |
| dbo.ACM_TagEquipmentMap | 10 | 1,986 | TagID |
| dbo.ACM_ThresholdCrossings | 7 | 573 | — |
| dbo.ACM_ThresholdMetadata | 13 | 180 | ThresholdID |
| dbo.ELECTRIC_MOTOR_Data | 14 | 17,477 | — |
| dbo.ELECTRIC_MOTOR_Data_RAW | 14 | 1,048,575 | — |
| dbo.Equipment | 8 | 29 | EquipID |
| dbo.FD_FAN_Data | 11 | 17,499 | EntryDateTime |
| dbo.GAS_TURBINE_Data | 18 | 2,911 | EntryDateTime |
| dbo.ModelRegistry | 8 | 630 | ModelType, EquipID, Version |
| dbo.WFA_TURBINE_0_Data | 87 | 54,986 | EntryDateTime |
| dbo.WFA_TURBINE_10_Data | 87 | 53,592 | EntryDateTime |
| dbo.WFA_TURBINE_11_Data | 87 | 0 | EntryDateTime |
| dbo.WFA_TURBINE_13_Data | 87 | 54,010 | EntryDateTime |
| dbo.WFA_TURBINE_14_Data | 87 | 54,197 | EntryDateTime |
| dbo.WFA_TURBINE_17_Data | 87 | 55,090 | EntryDateTime |
| dbo.WFA_TURBINE_21_Data | 87 | 0 | EntryDateTime |
| dbo.WFA_TURBINE_22_Data | 87 | 53,036 | EntryDateTime |
| dbo.WFA_TURBINE_24_Data | 87 | 55,003 | EntryDateTime |
| dbo.WFA_TURBINE_25_Data | 87 | 54,712 | EntryDateTime |
| dbo.WFA_TURBINE_26_Data | 87 | 53,702 | EntryDateTime |
| dbo.WFA_TURBINE_38_Data | 87 | 54,835 | EntryDateTime |
| dbo.WFA_TURBINE_3_Data | 87 | 55,487 | EntryDateTime |
| dbo.WFA_TURBINE_40_Data | 87 | 56,158 | EntryDateTime |
| dbo.WFA_TURBINE_42_Data | 87 | 53,886 | EntryDateTime |
| dbo.WFA_TURBINE_45_Data | 87 | 53,739 | EntryDateTime |
| dbo.WFA_TURBINE_51_Data | 87 | 54,436 | EntryDateTime |
| dbo.WFA_TURBINE_68_Data | 87 | 54,358 | EntryDateTime |
| dbo.WFA_TURBINE_69_Data | 87 | 54,813 | EntryDateTime |
| dbo.WFA_TURBINE_71_Data | 87 | 54,744 | EntryDateTime |
| dbo.WFA_TURBINE_72_Data | 87 | 54,082 | EntryDateTime |
| dbo.WFA_TURBINE_73_Data | 87 | 54,042 | EntryDateTime |
| dbo.WFA_TURBINE_84_Data | 87 | 53,772 | EntryDateTime |
| dbo.WFA_TURBINE_92_Data | 87 | 54,067 | EntryDateTime |
| dbo.WIND_TURBINE_Data | 5 | 50,530 | — |

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
**Row Count:** 90  
**Date Range:** 2025-12-05 11:37:01 to 2025-12-13 12:32:57  

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
| GOOD | 2025-12-11 15:24:04 | 0.0 | 0 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 |
| GOOD | 2025-12-11 15:24:20 | 0.0 | 0 | D12F3400-A17C-4DAF-9F69-4E08F5102CD6 | 1 |
| GOOD | 2025-12-11 15:24:36 | 0.0 | 0 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 1 |
| GOOD | 2025-12-11 15:25:08 | 0.0 | 0 | 4BF97DB3-902F-460B-B6C2-B685EE80B6B9 | 1 |
| GOOD | 2025-12-11 15:25:41 | 0.0 | 0 | 04F37A78-E7FA-46CB-9B06-65D0AE7CE180 | 1 |
| GOOD | 2025-12-11 15:26:16 | 0.0 | 0 | FE2888AF-6C3F-4574-B101-31E2D4303BBB | 1 |

### Bottom 10 Records

| AlertZone | StartTimestamp | DurationHours | RecordCount | RunID | EquipID |
| --- | --- | --- | --- | --- | --- |
| GOOD | 2025-12-13 12:32:57 | 0.0 | 0 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 |
| GOOD | 2025-12-13 12:32:08 | 0.0 | 0 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 |
| GOOD | 2025-12-13 12:31:20 | 0.0 | 0 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| GOOD | 2025-12-13 12:30:16 | 0.0 | 0 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 |
| GOOD | 2025-12-13 12:29:28 | 0.0 | 0 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 |
| GOOD | 2025-12-13 12:28:27 | 0.0 | 0 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 |
| GOOD | 2025-12-13 12:27:45 | 0.0 | 0 | 12F14125-8F0B-4969-B6C9-CEB2A4212EDF | 5000 |
| GOOD | 2025-12-13 12:26:58 | 0.0 | 0 | F0A42054-5820-4769-A0BA-49307CAB6CFC | 5000 |
| GOOD | 2025-12-13 12:25:58 | 0.0 | 0 | C8281286-EB94-420E-85F1-A4428AB4A9CD | 5000 |
| GOOD | 2025-12-13 12:25:17 | 0.0 | 0 | 7EB9892B-9A43-41D2-8BE7-5699C70214E8 | 5000 |

---


## dbo.ACM_Anomaly_Events

**Primary Key:** Id  
**Row Count:** 332  
**Date Range:** 2022-04-04 02:40:00 to 2025-09-11 15:30:00  

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
| 5240 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 02:20:00 | 2022-09-02 05:50:00 | info |
| 5239 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-01 22:50:00 | 2022-09-02 00:50:00 | info |
| 5238 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 2022-09-01 01:40:00 | 2022-09-01 05:20:00 | info |
| 5237 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 | 2022-08-31 02:10:00 | 2022-08-31 02:30:00 | info |
| 5236 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 2022-08-30 00:50:00 | 2022-08-30 01:20:00 | info |
| 5235 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 2022-08-29 19:00:00 | 2022-08-29 20:10:00 | info |
| 5234 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 2022-08-29 12:20:00 | 2022-08-29 12:40:00 | info |
| 5233 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | 2022-08-29 00:30:00 | 2022-08-29 05:40:00 | info |
| 5232 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | 2022-08-28 19:00:00 | 2022-08-28 20:30:00 | info |
| 5231 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | 2022-08-28 13:20:00 | 2022-08-28 13:50:00 | info |

---


## dbo.ACM_BaselineBuffer

**Primary Key:** Id  
**Row Count:** 546,950  
**Date Range:** 2018-01-05 00:00:00 to 2025-09-14 23:30:00  

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
| 14728862 | 5000 | 2022-09-03 06:00:00 | wind_speed_4_avg | 3.5999999046325684 | NULL | 2025-12-13 12:32:52 |
| 14728861 | 5000 | 2022-09-03 06:00:00 | wind_speed_3_std | 0.6000000238418579 | NULL | 2025-12-13 12:32:52 |
| 14728860 | 5000 | 2022-09-03 06:00:00 | wind_speed_3_min | 1.600000023841858 | NULL | 2025-12-13 12:32:52 |
| 14728859 | 5000 | 2022-09-03 06:00:00 | wind_speed_3_max | 7.199999809265137 | NULL | 2025-12-13 12:32:52 |
| 14728858 | 5000 | 2022-09-03 06:00:00 | wind_speed_3_avg | 3.5999999046325684 | NULL | 2025-12-13 12:32:52 |
| 14728857 | 5000 | 2022-09-03 06:00:00 | sensor_9_avg | 39.0 | NULL | 2025-12-13 12:32:52 |
| 14728856 | 5000 | 2022-09-03 06:00:00 | sensor_8_avg | 47.0 | NULL | 2025-12-13 12:32:52 |
| 14728855 | 5000 | 2022-09-03 06:00:00 | sensor_7_avg | 40.0 | NULL | 2025-12-13 12:32:52 |
| 14728854 | 5000 | 2022-09-03 06:00:00 | sensor_6_avg | 33.0 | NULL | 2025-12-13 12:32:52 |
| 14728853 | 5000 | 2022-09-03 06:00:00 | sensor_5_std | 0.0 | NULL | 2025-12-13 12:32:52 |

---


## dbo.ACM_CalibrationSummary

**Primary Key:** No primary key  
**Row Count:** 720  

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
| Time-Series Anomaly (AR1) | 0.8596000075340271 | 0.791100025177002 | 2.5653 | 3.4884 | 20.0 | 0.0 | NULL | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Correlation Break (PCA-SPE) | 0.515500009059906 | 1.003499984741211 | 2.6411 | 2.9726 | 20.0 | 0.0 | NULL | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Multivariate Outlier (PCA-T2) | 1.4027999639511108 | 1.174399971961975 | 3.2948 | 3.2948 | 20.0 | 0.0 | NULL | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Multivariate Distance (Mahalanobis) | 0.6378999948501587 | 0.33550000190734863 | 1.2795 | 1.6187 | 20.0 | 0.0 | 1.6164943610945112e+29 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Rare State (IsolationForest) | 0.707099974155426 | 0.4618000090122223 | 1.3856 | 2.2137 | 20.0 | 0.0 | NULL | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Density Anomaly (GMM) | 0.6704000234603882 | 0.48240000009536743 | 1.3954 | 2.2046 | 20.0 | 0.0 | NULL | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Baseline Consistency (OMR) | 0.7753000259399414 | 0.6182000041007996 | 1.6842 | 2.9088 | 20.0 | 0.0 | NULL | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| cusum_z | 1.0170999765396118 | 0.9740999937057495 | 3.2457 | 3.2972 | 20.0 | 0.0 | NULL | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Multivariate Outlier (PCA-T2) | 2.152600049972534 | 2.955899953842163 | 10.0 | 10.0 | 20.0 | 0.0 | NULL | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| Correlation Break (PCA-SPE) | 2.573199987411499 | 3.357100009918213 | 10.0 | 10.0 | 20.0 | 0.0 | NULL | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |

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
| Time-Series Anomaly (AR1) | 1.2835999727249146 | 1.443600058555603 | 4.6626 | 5.8397 | 20.0 | 0.0 | NULL | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| Correlation Break (PCA-SPE) | 1.0398000478744507 | 0.5212000012397766 | 1.617 | 1.617 | 20.0 | 0.0 | NULL | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |

---


## dbo.ACM_ColdstartState

**Primary Key:** EquipID, Stage  
**Row Count:** 6  
**Date Range:** 2025-12-05 06:06:49 to 2025-12-13 06:25:16  

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
| 1 | score | COMPLETE | 1 | 2025-12-11 09:53:57 | 2025-12-11 09:53:57 | 2025-12-11 09:53:57 | 241 | 200 | 2023-10-15 00:00:00 |
| 2621 | score | COMPLETE | 1 | 2025-12-05 06:06:49 | 2025-12-05 06:06:49 | 2025-12-05 06:06:49 | 241 | 200 | 2023-10-15 00:00:00 |
| 5000 | score | COMPLETE | 1 | 2025-12-13 06:25:16 | 2025-12-13 06:25:17 | 2025-12-13 06:25:17 | 241 | 200 | 2022-08-04 06:10:00 |
| 5092 | score | COMPLETE | 1 | 2025-12-13 06:00:34 | 2025-12-13 06:00:34 | 2025-12-13 06:00:34 | 241 | 200 | 2022-04-04 02:30:00 |
| 8632 | score | COMPLETE | 3 | 2025-12-11 12:30:24 | 2025-12-11 12:30:24 | 2025-12-11 12:30:24 | 563 | 200 | 2024-01-01 00:00:00 |
| 8634 | score | COMPLETE | 1 | 2025-12-13 03:08:03 | 2025-12-13 03:08:03 | 2025-12-13 03:08:03 | 241 | 200 | 2024-01-01 00:00:00 |

---


## dbo.ACM_Config

**Primary Key:** ConfigID  
**Row Count:** 332  
**Date Range:** 2025-12-09 12:47:06 to 2025-12-13 06:53:01  

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
| 492 | 0 | data.train_csv | data/FD_FAN_BASELINE_DATA.csv | string | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 493 | 0 | data.score_csv | data/FD_FAN_BATCH_DATA.csv | string | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 494 | 0 | data.data_dir | data | string | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 495 | 0 | data.timestamp_col | EntryDateTime | string | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 496 | 0 | data.tag_columns | [] | list | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 497 | 0 | data.sampling_secs | 1800 | int | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 498 | 0 | data.max_rows | 100000 | int | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 499 | 0 | features.window | 16 | int | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 500 | 0 | features.fft_bands | [0.0, 0.1, 0.3, 0.5] | list | 2025-12-13 06:43:39 | B19cl3pc\bhadk |
| 501 | 0 | features.top_k_tags | 5 | int | 2025-12-13 06:43:39 | B19cl3pc\bhadk |

### Bottom 10 Records

| ConfigID | EquipID | ParamPath | ParamValue | ValueType | UpdatedAt | UpdatedBy |
| --- | --- | --- | --- | --- | --- | --- |
| 835 | 5000 | runtime.tick_minutes | 1440 | int | 2025-12-13 06:53:01 | sql_batch_runner |
| 834 | 5092 | runtime.tick_minutes | 1440 | int | 2025-12-13 06:00:24 | sql_batch_runner |
| 833 | 5092 | data.tag_columns | ["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3... | list | 2025-12-13 06:43:40 | B19cl3pc\bhadk |
| 832 | 5092 | data.sampling_secs | 600 | int | 2025-12-13 06:43:40 | B19cl3pc\bhadk |
| 831 | 5092 | data.timestamp_col | EntryDateTime | string | 2025-12-13 06:43:40 | B19cl3pc\bhadk |
| 830 | 5084 | data.tag_columns | ["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3... | list | 2025-12-13 06:43:40 | B19cl3pc\bhadk |
| 829 | 5084 | data.sampling_secs | 600 | int | 2025-12-13 06:43:40 | B19cl3pc\bhadk |
| 828 | 5084 | data.timestamp_col | EntryDateTime | string | 2025-12-13 06:43:40 | B19cl3pc\bhadk |
| 827 | 5073 | data.tag_columns | ["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3... | list | 2025-12-13 06:43:40 | B19cl3pc\bhadk |
| 826 | 5073 | data.sampling_secs | 600 | int | 2025-12-13 06:43:40 | B19cl3pc\bhadk |

---


## dbo.ACM_ConfigHistory

**Primary Key:** ID  
**Row Count:** 56  
**Date Range:** 2025-12-03 14:40:59 to 2025-12-13 12:32:49  

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

### Bottom 10 Records

| ID | Timestamp | EquipID | ParameterPath | OldValue | NewValue | ChangedBy | ChangeReason | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 56 | 2025-12-13 12:32:49 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 6.72e+56 exceeds 1e28 (critical instability) | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c |
| 55 | 2025-12-13 12:32:00 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 5.16e+74 exceeds 1e28 (critical instability) | 69732a1e-a2c5-44fd-a9b5-93fcedbe4c62 |
| 54 | 2025-12-13 12:31:14 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.56e+54 exceeds 1e28 (critical instability) | 1362b1e7-3406-4352-858b-0b8464efa0e5 |
| 53 | 2025-12-13 12:30:09 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 3.45e+50 exceeds 1e28 (critical instability) | 75e70d35-4ee3-4e09-a7db-26669cef9484 |
| 52 | 2025-12-13 12:29:20 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.68e+60 exceeds 1e28 (critical instability) | 2c963a3b-ece6-4004-b6a4-911defa955e2 |
| 51 | 2025-12-13 12:28:20 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number inf exceeds 1e28 (critical instability) | 4dbe8707-50f4-463e-9e58-7e0e585588cd |
| 50 | 2025-12-13 12:27:38 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number inf exceeds 1e28 (critical instability) | 12f14125-8f0b-4969-b6c9-ceb2a4212edf |
| 49 | 2025-12-13 12:26:49 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 7.69e+72 exceeds 1e28 (critical instability) | f0a42054-5820-4769-a0ba-49307cab6cfc |
| 48 | 2025-12-13 12:25:50 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.18e+71 exceeds 1e28 (critical instability) | c8281286-eb94-420e-85f1-a4428ab4a9cd |
| 47 | 2025-12-13 12:25:10 | 5000 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.34e+40 exceeds 1e28 (critical instability) | 7eb9892b-9a43-41d2-8be7-5699c70214e8 |

---


## dbo.ACM_ContributionCurrent

**Primary Key:** No primary key  
**Row Count:** 720  

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
| Baseline Consistency (OMR) | 23.0 | 1.324679970741272 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Multivariate Distance (Mahalanobis) | 20.510000228881836 | 1.1814050674438477 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Rare State (IsolationForest) | 17.15999984741211 | 0.9883794188499451 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| cusum_z | 14.949999809265137 | 0.8609230518341064 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Multivariate Outlier (PCA-T2) | 11.579999923706055 | 0.6669564247131348 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Density Anomaly (GMM) | 10.65999984741211 | 0.6142964959144592 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Time-Series Anomaly (AR1) | 2.1500000953674316 | 0.12359651178121567 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Correlation Break (PCA-SPE) | 0.0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| Multivariate Distance (Mahalanobis) | 19.329999923706055 | 0.8986645340919495 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| Time-Series Anomaly (AR1) | 19.110000610351562 | 0.8886600136756897 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |

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
| Multivariate Outlier (PCA-T2) | 51.81999969482422 | 10.0 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| Multivariate Distance (Mahalanobis) | 22.110000610351562 | 4.267228603363037 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |

---


## dbo.ACM_ContributionTimeline

**Primary Key:** No primary key  
**Row Count:** 214,368  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| 2022-04-04 02:30:00 | cusum_z | 18.56999969482422 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | Multivariate Distance (Mahalanobis) | 0.0 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | Time-Series Anomaly (AR1) | 18.43000030517578 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | Baseline Consistency (OMR) | 25.290000915527344 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | Rare State (IsolationForest) | 8.920000076293945 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | Multivariate Outlier (PCA-T2) | 4.730000019073486 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | Correlation Break (PCA-SPE) | 17.700000762939453 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | Density Anomaly (GMM) | 6.369999885559082 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:40:00 | Multivariate Distance (Mahalanobis) | 0.0 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:40:00 | cusum_z | 19.600000381469727 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |

### Bottom 10 Records

| Timestamp | DetectorType | ContributionPct | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | Time-Series Anomaly (AR1) | 7.690000057220459 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | cusum_z | 4.059999942779541 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | Correlation Break (PCA-SPE) | 2.440000057220459 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | Multivariate Outlier (PCA-T2) | 8.289999961853027 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | Multivariate Distance (Mahalanobis) | 32.61000061035156 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | Rare State (IsolationForest) | 25.219999313354492 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | Density Anomaly (GMM) | 18.479999542236328 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | Baseline Consistency (OMR) | 1.2200000286102295 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:30:00 | cusum_z | 4.03000020980835 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:30:00 | Time-Series Anomaly (AR1) | 8.479999542236328 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |

---


## dbo.ACM_DailyFusedProfile

**Primary Key:** ID  
**Row Count:** 557  
**Date Range:** 2025-12-05 00:00:00 to 2025-12-13 00:00:00  

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
| 171387 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:32:57 |
| 171386 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:32:57 |
| 171385 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:32:08 |
| 171384 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:32:08 |
| 171383 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:31:20 |
| 171382 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:31:20 |
| 171381 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:30:16 |
| 171380 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:30:16 |
| 171379 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:29:28 |
| 171378 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 2025-12-13 00:00:00 | NULL | NULL | NULL | NULL | 2025-12-13 12:29:28 |

---


## dbo.ACM_DataQuality

**Primary Key:** No primary key  
**Row Count:** 5,150  
**Date Range:** 2022-04-04 02:30:00 to 2025-07-11 00:00:00  

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
| power_29_avg | 144 | 0 | 0.0 | 0.40405070781707764 | 0 | 1 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| power_29_max | 144 | 0 | 0.0 | 0.441830575466156 | 0 | 56 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| power_29_min | 144 | 0 | 0.0 | 0.2658897638320923 | 0 | 20 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| power_29_std | 144 | 0 | 0.0 | 0.0576351173222065 | 0 | 1 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| power_30_avg | 144 | 0 | 0.0 | 0.41087841987609863 | 0 | 1 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| power_30_max | 144 | 0 | 0.0 | 0.47445425391197205 | 0 | 2 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| power_30_min | 144 | 0 | 0.0 | 0.28014156222343445 | 0 | 1 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| power_30_std | 144 | 0 | 0.0 | 0.059888921678066254 | 0 | 1 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| reactive_power_27_avg | 144 | 0 | 0.0 | 0.15484461188316345 | 0 | 3 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |
| reactive_power_27_max | 144 | 0 | 0.0 | 0.1821976751089096 | 0 | 30 | 2022-04-19 02:30:00 | 2022-04-20 02:20:00 | 144 |

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
**Row Count:** 90  

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
| HEALTHY | LOW | 88.80000305175781 | 90.19999694824219 | 60.5 | 2 | omr | 133 | 10 | 1 |
| HEALTHY | LOW | 88.5999984741211 | 88.9000015258789 | 39.79999923706055 | 6 | mhal | 781 | 42 | 56 |
| HEALTHY | LOW | 87.80000305175781 | 89.0 | 66.4000015258789 | 2 | iforest | 121 | 22 | 1 |
| HEALTHY | LOW | 93.30000305175781 | 91.19999694824219 | 63.5 | 1 | ar1 | 63 | 3 | 1 |
| HEALTHY | LOW | 91.5 | 92.19999694824219 | 72.9000015258789 | 3 | mhal | 141 | 3 | 0 |
| HEALTHY | LOW | 93.0 | 91.9000015258789 | 61.79999923706055 | 1 | pca_spe | 138 | 5 | 1 |
| HEALTHY | LOW | 93.19999694824219 | 88.5999984741211 | 74.9000015258789 | 2 | omr | 112 | 32 | 0 |
| HEALTHY | LOW | 88.9000015258789 | 91.19999694824219 | 76.0999984741211 | 12 | pca_spe | 666 | 54 | 0 |
| HEALTHY | LOW | 93.0 | 91.9000015258789 | 61.79999923706055 | 1 | pca_spe | 138 | 5 | 1 |
| HEALTHY | LOW | 94.5999984741211 | 89.19999694824219 | 14.399999618530273 | 1 | mhal | 89 | 5 | 3 |

### Bottom 10 Records

| Status | Severity | CurrentHealth | AvgHealth | MinHealth | EpisodeCount | WorstSensor | GoodCount | WatchCount | AlertCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALERT | HIGH | 69.69999694824219 | 68.9000015258789 | 13.300000190734863 | 7 | pca_t2 | 155 | 83 | 267 |
| HEALTHY | LOW | 91.69999694824219 | 87.80000305175781 | 52.099998474121094 | 2 | pca_t2 | 122 | 16 | 6 |
| HEALTHY | LOW | 92.4000015258789 | 88.4000015258789 | 72.30000305175781 | 1 | cusum | 121 | 23 | 0 |
| HEALTHY | LOW | 88.0 | 89.0 | 68.9000015258789 | 2 | pca_spe | 118 | 22 | 4 |
| HEALTHY | LOW | 90.0999984741211 | 89.5 | 49.099998474121094 | 2 | gmm | 129 | 13 | 2 |
| HEALTHY | LOW | 94.30000305175781 | 89.5 | 19.899999618530273 | 8 | pca_spe | 783 | 67 | 29 |
| HEALTHY | LOW | 90.69999694824219 | 92.19999694824219 | 84.4000015258789 | 13 | gmm | 716 | 4 | 0 |
| HEALTHY | LOW | 90.0999984741211 | 89.0 | 12.199999809265137 | 9 | cusum | 770 | 60 | 49 |
| CAUTION | MEDIUM | 70.19999694824219 | 92.5999984741211 | 70.19999694824219 | 4 | mhal | 137 | 7 | 0 |
| ALERT | HIGH | 60.70000076293945 | 91.80000305175781 | 59.70000076293945 | 1 | gmm | 137 | 5 | 2 |

---


## dbo.ACM_DefectTimeline

**Primary Key:** No primary key  
**Row Count:** 2,450  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 00:30:00  

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
| 2022-04-04 02:30:00 | ZONE_CHANGE | START | GOOD | GOOD | 93.41999816894531 | 93.41999816894531 | -0.2889 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:10:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 78.91000366210938 | 78.91000366210938 | 1.4005 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:20:00 | ZONE_CHANGE | WATCH | GOOD | GOOD | 85.62999725341797 | 85.62999725341797 | 1.0123 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:40:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 83.0199966430664 | 83.0199966430664 | 1.1777 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:50:00 | ZONE_CHANGE | WATCH | GOOD | GOOD | 90.66999816894531 | 90.66999816894531 | 0.6048 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 05:10:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 82.08999633789062 | 82.08999633789062 | 1.2313 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 05:40:00 | ZONE_CHANGE | WATCH | ALERT | ALERT | 60.369998931884766 | 60.369998931884766 | 2.1492 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 05:50:00 | ZONE_CHANGE | ALERT | WATCH | WATCH | 70.19999694824219 | 70.19999694824219 | 1.7859 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 06:00:00 | ZONE_CHANGE | WATCH | GOOD | GOOD | 87.05000305175781 | 87.05000305175781 | 0.912 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 15:40:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 83.7300033569336 | 83.7300033569336 | -1.1349 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |

### Bottom 10 Records

| Timestamp | EventType | FromZone | ToZone | HealthZone | HealthAtEvent | HealthIndex | FusedZ | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 00:30:00 | ZONE_CHANGE | WATCH | GOOD | GOOD | 92.62999725341797 | 92.62999725341797 | 0.3906 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 00:00:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 80.05999755859375 | 80.05999755859375 | 1.3415 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-13 07:30:00 | ZONE_CHANGE | WATCH | GOOD | GOOD | 94.80999755859375 | 94.80999755859375 | 0.0786 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-13 07:00:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 71.58000183105469 | 71.58000183105469 | 1.7304 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-13 00:30:00 | ZONE_CHANGE | WATCH | GOOD | GOOD | 91.36000061035156 | 91.36000061035156 | 0.5347 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-13 00:00:00 | ZONE_CHANGE | GOOD | WATCH | WATCH | 71.5999984741211 | 71.5999984741211 | 1.7294 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 07:30:00 | ZONE_CHANGE | ALERT | GOOD | GOOD | 93.18000030517578 | 93.18000030517578 | 0.3211 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 05:00:00 | ZONE_CHANGE | WATCH | ALERT | ALERT | 67.30999755859375 | 67.30999755859375 | 1.898 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 04:30:00 | ZONE_CHANGE | ALERT | WATCH | WATCH | 70.25 | 70.25 | 1.784 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 00:00:00 | ZONE_CHANGE | GOOD | ALERT | ALERT | 54.900001525878906 | 54.900001525878906 | 2.3362 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |

---


## dbo.ACM_DetectorCorrelation

**Primary Key:** No primary key  
**Row Count:** 2,520  

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
| Time-Series Anomaly (AR1) | Correlation Break (PCA-SPE) | -0.0737 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Time-Series Anomaly (AR1) <-> Correlation Break (PCA-SPE) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Multivariate Outlier (PCA-T2) | 0.1165 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Time-Series Anomaly (AR1) <-> Multivariate Outlier (PCA-T2) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Multivariate Distance (Mahalanobis) | 0.623 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Time-Series Anomaly (AR1) <-> Multivariate Distance (Mahalanobis) | Temporal spikes seen by both detectors |
| Time-Series Anomaly (AR1) | Rare State (IsolationForest) | 0.8117 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Time-Series Anomaly (AR1) <-> Rare State (IsolationForest) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Density Anomaly (GMM) | 0.7177 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Time-Series Anomaly (AR1) <-> Density Anomaly (GMM) | Transient spikes align with pattern change |
| Time-Series Anomaly (AR1) | Baseline Consistency (OMR) | 0.6803 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Time-Series Anomaly (AR1) <-> Baseline Consistency (OMR) | Health baseline tracking repeated spikes |
| Time-Series Anomaly (AR1) | cusum_z | 0.1593 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Time-Series Anomaly (AR1) <-> cusum_z | Temporal spikes seen by both detectors |
| Correlation Break (PCA-SPE) | Multivariate Outlier (PCA-T2) | 0.4606 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Correlation Break (PCA-SPE) <-> Multivariate Outlier (PCA-T2) | Detectors reacting together; check shared cause |
| Correlation Break (PCA-SPE) | Multivariate Distance (Mahalanobis) | -0.06 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Correlation Break (PCA-SPE) <-> Multivariate Distance (Mahalanobis) | Regime/cluster shift across many sensors |
| Correlation Break (PCA-SPE) | Rare State (IsolationForest) | -0.0753 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Correlation Break (PCA-SPE) <-> Rare State (IsolationForest) | Detectors reacting together; check shared cause |

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
**Row Count:** 21  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

### Top 10 Records

| RunID | EquipID |
| --- | --- |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 |
| 3C41A9D9-2E8C-4AB0-9010-2CEDAA2315D4 | 5000 |
| E8535A73-4E4F-4F48-92AA-43EBD5DC7EB4 | 5000 |
| 0813F8C9-A231-494D-BDC8-4F85DDACB9DE | 5000 |
| 19AD36A7-7882-4538-AE66-5CEC6E8C069B | 5092 |
| 200EFE6D-C8B3-4BD2-8D08-69A4D569D23C | 5000 |
| 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 |

### Bottom 10 Records

| RunID | EquipID |
| --- | --- |
| 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 |
| 1C3C911D-2883-459C-845E-D43A1AEE5724 | 5092 |
| 96A0DF64-6A48-400C-AD81-D2F4C1704750 | 8632 |
| 12F14125-8F0B-4969-B6C9-CEB2A4212EDF | 5000 |
| C8281286-EB94-420E-85F1-A4428AB4A9CD | 5000 |
| DC06050E-6596-43E3-A725-A2DAFB28567D | 1 |
| 9B2207A9-FD0E-48DC-91BA-9941C0EC88FB | 2621 |
| 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 |
| 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 |
| BF7E31AE-6091-4EE0-BF79-82B64ABCE703 | 5000 |

---


## dbo.ACM_DriftSeries

**Primary Key:** No primary key  
**Row Count:** 30,201  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| 2022-04-04 02:30:00 | -2.6512999534606934 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:40:00 | -2.6470999717712402 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:50:00 | -2.6326000690460205 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:00:00 | -2.6084001064300537 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:10:00 | -2.5520999431610107 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:20:00 | -2.484999895095825 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:30:00 | -2.4237000942230225 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:40:00 | -2.3482000827789307 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:50:00 | -2.279599905014038 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 04:00:00 | -2.2130000591278076 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |

### Bottom 10 Records

| Timestamp | DriftValue | RunID | EquipID |
| --- | --- | --- | --- |
| 2025-09-14 23:00:00 | 0.24310000240802765 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:30:00 | 0.2565999925136566 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:00:00 | 0.2711000144481659 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 21:30:00 | 0.28760001063346863 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 21:00:00 | 0.3061999976634979 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 20:30:00 | 0.3276999890804291 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 20:00:00 | 0.35089999437332153 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 19:30:00 | 0.37599998712539673 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 19:00:00 | 0.4016999900341034 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 18:30:00 | 0.4275999963283539 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |

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
**Row Count:** 3,950  
**Date Range:** 2025-12-11 09:54:05 to 2025-12-13 07:02:40  

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
| 34376 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | Correlation Break (PCA-SPE) | NULL | 22.951675415039062 | 1 | 2025-12-11 09:54:05 | 1 |
| 34377 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | Multivariate Outlier (PCA-T2) | NULL | 18.488798141479492 | 2 | 2025-12-11 09:54:05 | 1 |
| 34378 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | Baseline Consistency (OMR) | NULL | 16.30406951904297 | 3 | 2025-12-11 09:54:05 | 1 |
| 34379 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | Time-Series Anomaly (AR1) | NULL | 13.520299911499023 | 4 | 2025-12-11 09:54:05 | 1 |
| 34380 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | Rare State (IsolationForest) | NULL | 9.547196388244629 | 5 | 2025-12-11 09:54:05 | 1 |
| 34381 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | Multivariate Distance (Mahalanobis) | NULL | 9.086981773376465 | 6 | 2025-12-11 09:54:05 | 1 |
| 34382 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | Density Anomaly (GMM) | NULL | 5.758297920227051 | 7 | 2025-12-11 09:54:05 | 1 |
| 34383 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | cusum_z | NULL | 4.342683792114258 | 8 | 2025-12-11 09:54:05 | 1 |
| 34384 | D12F3400-A17C-4DAF-9F69-4E08F5102CD6 | 1 | Multivariate Distance (Mahalanobis) | NULL | 29.325246810913086 | 1 | 2025-12-11 09:54:22 | 1 |
| 34385 | D12F3400-A17C-4DAF-9F69-4E08F5102CD6 | 1 | Correlation Break (PCA-SPE) | NULL | 22.192501068115234 | 2 | 2025-12-11 09:54:22 | 1 |

### Bottom 10 Records

| ID | RunID | EpisodeID | DetectorType | SensorName | ContributionPct | Rank | CreatedAt | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 38325 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 2 | Correlation Break (PCA-SPE) | NULL | 1.2297711372375488 | 8 | 2025-12-13 07:02:40 | 1 |
| 38324 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 2 | Multivariate Outlier (PCA-T2) | NULL | 1.229771375656128 | 7 | 2025-12-13 07:02:40 | 1 |
| 38323 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 2 | Rare State (IsolationForest) | NULL | 3.912198543548584 | 6 | 2025-12-13 07:02:40 | 1 |
| 38322 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 2 | Baseline Consistency (OMR) | NULL | 4.848148345947266 | 5 | 2025-12-13 07:02:40 | 1 |
| 38321 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 2 | Density Anomaly (GMM) | NULL | 5.052623748779297 | 4 | 2025-12-13 07:02:40 | 1 |
| 38320 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 2 | cusum_z | NULL | 8.600296020507812 | 3 | 2025-12-13 07:02:40 | 1 |
| 38319 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 2 | Multivariate Distance (Mahalanobis) | NULL | 15.012364387512207 | 2 | 2025-12-13 07:02:40 | 1 |
| 38318 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 2 | Time-Series Anomaly (AR1) | NULL | 60.11481857299805 | 1 | 2025-12-13 07:02:40 | 1 |
| 38317 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 1 | cusum_z | NULL | 4.528357028961182 | 5 | 2025-12-13 07:02:40 | 1 |
| 38316 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 1 | Baseline Consistency (OMR) | NULL | 4.865088939666748 | 4 | 2025-12-13 07:02:40 | 1 |

---


## dbo.ACM_EpisodeDiagnostics

**Primary Key:** ID  
**Row Count:** 353  
**Date Range:** 2022-04-04 02:40:00 to 2025-09-11 15:30:00  

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
| 15838 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | 4 | 0.6017947988158024 | 2022-09-03 02:20:00 | 0.8333333333333334 | Multivariate Distance (Mahalanobis) | LOW | UNKNOWN |
| 15837 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | 3 | 0.44223891289522516 | 2022-09-02 23:50:00 | 0.8333333333333334 | Baseline Consistency (OMR) | LOW | UNKNOWN |
| 15836 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | 2 | 1.6757838990333698 | 2022-09-02 21:30:00 | 0.8333333333333334 | Time-Series Anomaly (AR1) | LOW | UNKNOWN |
| 15835 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | 1 | 1.2290378478956647 | 2022-09-02 14:20:00 | 0.3333333333333333 | Multivariate Distance (Mahalanobis) | LOW | UNKNOWN |
| 15830 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2 | 0.6910197262432975 | 2022-09-02 02:20:00 | 3.5 | Time-Series Anomaly (AR1) | LOW | UNKNOWN |
| 15829 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 1 | 0.8835716323713849 | 2022-09-01 22:50:00 | 2.0 | Time-Series Anomaly (AR1) | LOW | UNKNOWN |
| 15826 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 1 | 0.9157930079990309 | 2022-09-01 01:40:00 | 3.6666666666666665 | Correlation Break (PCA-SPE) | LOW | UNKNOWN |
| 15824 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 | 1 | 0.9119551698925968 | 2022-08-31 02:10:00 | 0.3333333333333333 | Correlation Break (PCA-SPE) | LOW | UNKNOWN |
| 15822 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 3 | 1.1102170560590616 | 2022-08-30 00:50:00 | 0.5 | Correlation Break (PCA-SPE) | LOW | UNKNOWN |
| 15821 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 2 | 0.5477284042557031 | 2022-08-29 19:00:00 | 1.1666666666666667 | Correlation Break (PCA-SPE) | LOW | UNKNOWN |

---


## dbo.ACM_EpisodeMetrics

**Primary Key:** No primary key  
**Row Count:** 88  

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
| 2 | 5.83 | 2.92 | 2.92 | 4.33 | 1.5 | 0.0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 6 | 21.0 | 3.5 | 1.75 | 11.0 | 1.0 | 0.0 | 0.0 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| 2 | 6.83 | 3.42 | 3.42 | 3.83 | 3.0 | 0.0 | 0.0 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 1 | 3.67 | 3.67 | 3.67 | 3.67 | 3.67 | 0.0 | 0.0 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| 3 | 5.33 | 1.78 | 2.17 | 2.83 | 0.33 | 0.0 | 0.0 | 5BA6BC6D-56B6-4719-BE05-0E9BEEDF38C7 | 5000 |
| 1 | 3.67 | 3.67 | 3.67 | 3.67 | 3.67 | 0.0 | 0.0 | FFAD6AD5-E725-45B5-A792-14BF8330500D | 5000 |
| 2 | 6.5 | 3.25 | 3.25 | 4.33 | 2.17 | 0.0 | 0.0 | E1596C1E-4B19-4E7A-A7EF-168EB50EC183 | 5092 |
| 12 | 1.67 | 0.14 | 0.12 | 0.58 | 0.03 | 0.0 | 0.0 | 79E7DD32-441E-43EC-8A52-18707041B651 | 8634 |
| 1 | 3.67 | 3.67 | 3.67 | 3.67 | 3.67 | 0.0 | 0.0 | C4F1062D-682C-45CA-97B3-1995F206A9A3 | 5000 |
| 1 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 0.0 | 0.0 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 |

### Bottom 10 Records

| TotalEpisodes | TotalDurationHours | AvgDurationHours | MedianDurationHours | MaxDurationHours | MinDurationHours | RatePerDay | MeanInterarrivalHours | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 2 | 3.83 | 1.92 | 1.92 | 2.0 | 1.83 | 0.0 | 0.0 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| 1 | 4.33 | 4.33 | 4.33 | 4.33 | 4.33 | 0.0 | 0.0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 |
| 2 | 4.67 | 2.33 | 2.33 | 3.0 | 1.67 | 0.0 | 0.0 | CF9514D8-C683-4A5B-8DE3-EC6A9D6E6A92 | 5000 |
| 2 | 4.67 | 2.33 | 2.33 | 3.33 | 1.33 | 0.0 | 0.0 | 55DB3B5A-ED32-4DFA-970B-E41DD242B1F1 | 5000 |
| 8 | 40.0 | 5.0 | 4.5 | 10.5 | 1.5 | 0.0 | 0.0 | 202974D5-76B2-4D1E-BA3D-E321EEE5D336 | 8632 |
| 13 | 3.35 | 0.26 | 0.22 | 1.0 | 0.07 | 0.0 | 0.0 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 8634 |
| 9 | 55.0 | 6.11 | 5.0 | 17.5 | 2.0 | 0.0 | 0.0 | 682A58D9-4C3C-4B2C-8EC5-DC9852F27CC7 | 8632 |
| 4 | 2.83 | 0.71 | 0.83 | 0.83 | 0.33 | 0.0 | 0.0 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 |
| 1 | 2.67 | 2.67 | 2.67 | 2.67 | 2.67 | 0.0 | 0.0 | AE60C5F3-E31D-4D61-A4D5-D9E6BFDCC217 | 5092 |

---


## dbo.ACM_Episodes

**Primary Key:** No primary key  
**Row Count:** 88  

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
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2 | 175.0 | NULL | NULL | 2.1463578292035006 | 0.7573724334687362 |
| DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 | 6 | 105.0 | NULL | NULL | 2.0972625468055073 | 1.3029439992553395 |
| 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 | 2 | 205.0 | NULL | NULL | 1.7729776493994116 | 0.848264159098515 |
| 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 1 | 220.0 | NULL | NULL | 0.9157930079990309 | 0.4559061397437107 |
| 5BA6BC6D-56B6-4719-BE05-0E9BEEDF38C7 | 5000 | 3 | 130.0 | NULL | NULL | 0.7703306676881329 | 0.4743469905289605 |
| FFAD6AD5-E725-45B5-A792-14BF8330500D | 5000 | 1 | 220.0 | NULL | NULL | 2.0988487506816877 | 0.674075510041189 |
| E1596C1E-4B19-4E7A-A7EF-168EB50EC183 | 5092 | 2 | 195.0 | NULL | NULL | 1.5868823434631993 | 0.9781944567732942 |
| 79E7DD32-441E-43EC-8A52-18707041B651 | 8634 | 12 | 7.0 | NULL | NULL | 1.3026396925277732 | 0.6816917735606497 |
| C4F1062D-682C-45CA-97B3-1995F206A9A3 | 5000 | 1 | 220.0 | NULL | NULL | 2.0988487506816877 | 0.674075510041189 |
| 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | 1 | 120.0 | NULL | NULL | 3.988077274486474 | 1.336501737451136 |

### Bottom 10 Records

| RunID | EquipID | EpisodeCount | MedianDurationMinutes | CoveragePct | TimeInAlertPct | MaxFusedZ | AvgFusedZ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 7 | 720.0 | NULL | NULL | 2.4770509242211958 | 0.9752637030744661 |
| ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 | 2 | 115.0 | NULL | NULL | 1.624213165369021 | 1.1464465128703365 |
| F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 | 1 | 260.0 | NULL | NULL | 1.7012475804898572 | 1.080207519428952 |
| CF9514D8-C683-4A5B-8DE3-EC6A9D6E6A92 | 5000 | 2 | 140.0 | NULL | NULL | 1.8277245804170967 | 1.1131531768922223 |
| 55DB3B5A-ED32-4DFA-970B-E41DD242B1F1 | 5000 | 2 | 140.0 | NULL | NULL | 1.504275492520083 | 0.8241598090446985 |
| 202974D5-76B2-4D1E-BA3D-E321EEE5D336 | 8632 | 8 | 270.0 | NULL | NULL | 1.9047932894581576 | 0.8993395918161661 |
| 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 8634 | 13 | 13.0 | NULL | NULL | 1.0598112873069765 | 0.4975426415152678 |
| 682A58D9-4C3C-4B2C-8EC5-DC9852F27CC7 | 8632 | 9 | 300.0 | NULL | NULL | 2.3280873564786253 | 1.1082421750542462 |
| 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | 4 | 50.0 | NULL | NULL | 1.6757838990333698 | 0.5786891588214801 |
| AE60C5F3-E31D-4D61-A4D5-D9E6BFDCC217 | 5092 | 1 | 160.0 | NULL | NULL | 0.9261133335180507 | 0.5401970212940471 |

---


## dbo.ACM_EpisodesQC

**Primary Key:** RecordID  
**Row Count:** 88  
**Date Range:** 2025-12-05 11:37:38 to 2025-12-13 12:32:57  

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
| 1175 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | 1 | 120.0 | 4.17 | 3.09 | 3.988077163696289 | 8.986782873421362e-09 | 2025-12-11 15:24:05 |
| 1176 | D12F3400-A17C-4DAF-9F69-4E08F5102CD6 | 1 | 5 | 90.0 | 1.47 | 4.97 | 4.233062267303467 | 1.8959727121625747e-09 | 2025-12-11 15:24:21 |
| 1177 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 1 | 2 | 90.0 | 0.7 | 5.01 | 3.2824506759643555 | -4.780322360176115e-09 | 2025-12-11 15:24:37 |
| 1178 | 4BF97DB3-902F-460B-B6C2-B685EE80B6B9 | 1 | 9 | 180.0 | 5.7 | 4.75 | 4.579349994659424 | -1.0814166451211804e-08 | 2025-12-11 15:25:09 |
| 1179 | 04F37A78-E7FA-46CB-9B06-65D0AE7CE180 | 1 | 14 | 180.0 | 5.65 | 3.91 | 3.669748067855835 | 0.0 | 2025-12-11 15:25:43 |
| 1180 | FE2888AF-6C3F-4574-B101-31E2D4303BBB | 1 | 14 | 225.0 | 11.0 | 4.5 | 3.4808783531188965 | 0.0 | 2025-12-11 15:26:18 |
| 1181 | AEEF77AD-7C2C-4102-98E9-30F912E96DFF | 1 | 10 | 150.0 | 3.68 | 4.38 | 2.831772804260254 | 0.0 | 2025-12-11 15:26:52 |

### Bottom 10 Records

| RecordID | RunID | EquipID | EpisodeCount | MedianDurationMinutes | CoveragePct | TimeInAlertPct | MaxFusedZ | AvgFusedZ | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1282 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | 4 | 50.0 | 11.89 | 0.0 | 1.7866246700286865 | 0.0 | 2025-12-13 12:32:57 |
| 1281 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2 | 165.0 | 23.08 | 0.0 | 1.3008109331130981 | 0.0 | 2025-12-13 12:32:09 |
| 1280 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 1 | 220.0 | 15.38 | 1.49 | 2.0369691848754883 | -2.668864684807204e-09 | 2025-12-13 12:31:20 |
| 1279 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 | 1 | 20.0 | 1.4 | 1.39 | 2.1131763458251953 | -6.622738357719982e-09 | 2025-12-13 12:30:17 |
| 1278 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 3 | 30.0 | 8.39 | 0.0 | 1.6510183811187744 | 0.0 | 2025-12-13 12:29:29 |
| 1277 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | 4 | 60.0 | 31.47 | 0.0 | 1.327412724494934 | 0.0 | 2025-12-13 12:28:28 |
| 1276 | 12F14125-8F0B-4969-B6C9-CEB2A4212EDF | 5000 | 2 | 120.0 | 16.78 | 0.0 | 1.6908671855926514 | 0.0 | 2025-12-13 12:27:46 |
| 1275 | F0A42054-5820-4769-A0BA-49307CAB6CFC | 5000 | 2 | 195.0 | 27.27 | 0.0 | 1.0746101140975952 | -6.622738357719982e-09 | 2025-12-13 12:26:59 |
| 1274 | C8281286-EB94-420E-85F1-A4428AB4A9CD | 5000 | 1 | 360.0 | 25.17 | 0.0 | 1.146001935005188 | 0.0 | 2025-12-13 12:25:58 |
| 1273 | 7EB9892B-9A43-41D2-8BE7-5699C70214E8 | 5000 | 3 | 80.0 | 20.98 | 1.39 | 2.6529228687286377 | -1.3245476715439963e-08 | 2025-12-13 12:25:18 |

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
**Row Count:** 117,264  
**Date Range:** 2022-04-05 18:40:00 to 2025-09-21 23:00:00  

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
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 05:00:00 | 3.219266685479905e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 05:30:00 | 3.0282553711216685e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 06:00:00 | 2.848439732771641e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 06:30:00 | 2.6791718513213385e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 07:00:00 | 2.5198408110883774e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 07:30:00 | 2.3698706154985867e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 08:00:00 | 2.2287182184904475e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 08:30:00 | 2.095871665311122e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 09:00:00 | 1.97084833671845e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 09:30:00 | 1.853193290921899e-18 | 1.0 | 0.0 | 50.0 | GaussianTail | 2025-12-11 15:24:38 |

### Bottom 10 Records

| EquipID | RunID | Timestamp | FailureProb | SurvivalProb | HazardRate | ThresholdUsed | Method | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:59:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:58:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:57:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:56:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:55:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:54:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:53:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:52:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:51:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:50:00 | 1.0 | 0.0 | 0.0 | 50.0 | GaussianTail | 2025-12-13 08:39:29 |

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
**Row Count:** 20,562  
**Date Range:** 2025-12-01 10:33:45 to 2025-12-13 12:32:45  

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
| 20562 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_26_avg_med | low_variance | 50.0 | 0.0 | 2025-12-13 12:32:45 |
| 20561 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_49_med | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |
| 20560 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_46_rz | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |
| 20559 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_46_kurt | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |
| 20558 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_49_slope | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |
| 20557 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_46_mad | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |
| 20556 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_49_std | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |
| 20555 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_49_mad | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |
| 20554 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_46_med | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |
| 20553 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | sensor_46_std | low_variance | 0.0 | 0.0 | 2025-12-13 12:32:45 |

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
**Row Count:** 5  

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

### Top 10 Records

| EquipID | StateVersion | ModelCoefficientsJson | LastForecastJson | LastRetrainTime | TrainingDataHash | DataVolumeAnalyzed | RecentMAE | RecentRMSE | RetriggerReason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | {"alpha": 0.95, "beta": 0.3, "level": 93.43001279454205, "trend": 0.35352802876491185, "std_error... | {"forecast_mean": 99.8279569143469, "forecast_std": 0.8385297982085383, "forecast_range": 6.21645... | NULL |  | 14286 | 0.8385297982085383 | NULL | NULL |
| 5000 | 1 | {"alpha": 0.43571428571428567, "beta": 0.01, "level": 92.11957778495608, "trend": -0.003422203970... | {"forecast_mean": 90.39307588207197, "forecast_std": 0.9958079431624226, "forecast_range": 3.4461... | NULL |  | 75883 | 0.9958079431624226 | NULL | NULL |
| 5092 | 1 | {"alpha": 0.05, "beta": 0.01, "level": 92.64558701073078, "trend": 0.03150434016807175, "std_erro... | {"forecast_mean": 99.15204522899016, "forecast_std": 1.8518969071118483, "forecast_range": 7.3229... | NULL |  | 35301 | 1.8518969071118483 | NULL | NULL |
| 8632 | 1 | {"alpha": 0.43571428571428567, "beta": 0.01, "level": 93.34374049492133, "trend": 0.0417209147289... | {"forecast_mean": 98.42959754179881, "forecast_std": 2.116742613423934, "forecast_range": 6.61453... | NULL |  | 15720 | 2.116742613423934 | NULL | NULL |
| 8634 | 1 | {"alpha": 0.05, "beta": 0.01, "level": 91.24078966652314, "trend": -0.00039374875742422493, "std_... | {"forecast_mean": 89.25609905472633, "forecast_std": 1.1457479878652943, "forecast_range": 3.9685... | NULL |  | 10537 | 1.1457479878652943 | NULL | NULL |

---


## dbo.ACM_FusionQualityReport

**Primary Key:** No primary key  
**Row Count:** 900  
**Date Range:** 2025-12-05 11:36:59 to 2025-12-13 12:32:56  

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
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Baseline Consistency (OMR) | 0.1 | True | 0.05630768835544586 | 4.117178440093994 | 144 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Correlation Break (PCA-SPE) | 0.2 | True | -0.5155112147331238 | 0.0 | 144 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | cusum_z | 0.0 | True | -0.5107228755950928 | 0.9662541747093201 | 144 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Density Anomaly (GMM) | 0.1 | True | -0.0741300955414772 | 2.4915473461151123 | 144 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Fused Multi-Detector | 0.0 | True | 0.0 | 2.146357774734497 | 144 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Multivariate Distance (Mahalanobis) | 0.2 | True | 0.012796216644346714 | 1.781790018081665 | 144 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Multivariate Outlier (PCA-T2) | 0.0 | True | 0.7952334880828857 | 3.2948145866394043 | 144 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Rare State (IsolationForest) | 0.2 | True | 0.008366107940673828 | 2.315645933151245 | 144 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Streaming Anomaly (River) | 0.0 | False | 0.0 | 0.0 | 0 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | Time-Series Anomaly (AR1) | 0.2 | True | 0.22974474728107452 | 4.58666467666626 | 144 | 2025-12-13 11:41:57 |

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
**Row Count:** 11,444  
**Date Range:** 2025-12-05 11:36:59 to 2025-12-13 12:32:56  

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
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2025-12-13 11:41:57 | 3600 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

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
**Row Count:** 117,264  
**Date Range:** 2022-04-05 18:40:00 to 2025-09-21 23:00:00  

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
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 05:00:00 | 93.98373936383322 | 90.75476364722986 | 97.21271508043658 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 05:30:00 | 94.00319931600562 | 90.08087946902288 | 97.92551916298837 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 06:00:00 | 94.022659268178 | 89.51126624142978 | 98.53405229492623 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 06:30:00 | 94.0421192203504 | 89.00917710289193 | 99.07506133780888 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 07:00:00 | 94.06157917252278 | 88.5553907513262 | 99.56776759371937 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 07:30:00 | 94.08103912469518 | 88.13835616513151 | 100.0 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 08:00:00 | 94.10049907686756 | 87.75048995544148 | 100.0 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 08:30:00 | 94.11995902903996 | 87.38649660747423 | 100.0 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 09:00:00 | 94.13941898121234 | 87.04250544227034 | 100.0 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 2024-03-03 09:30:00 | 94.15887893338473 | 86.71558580252284 | 100.0 | 1.614487858301679 | ExponentialSmoothing | 2025-12-11 15:24:38 | NULL |

### Bottom 10 Records

| EquipID | RunID | Timestamp | ForecastHealth | CiLower | CiUpper | ForecastStd | Method | CreatedAt | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:59:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:58:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:57:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:56:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:55:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:54:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:53:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:52:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:51:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 2024-01-08 23:50:00 | 0.0 | 0.0 | 100.0 | 0.7050051385193825 | ExponentialSmoothing | 2025-12-13 08:39:29 | NULL |

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
**Row Count:** 900  

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
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0-10 | 0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |

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
**Row Count:** 30,201  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| 2022-04-04 02:30:00 | 93.42 | GOOD | -0.2888999879360199 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 93.41999816894531 | NORMAL |
| 2022-04-04 02:40:00 | 93.09 | GOOD | 0.42899999022483826 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 92.30999755859375 | NORMAL |
| 2022-04-04 02:50:00 | 92.76 | GOOD | 0.4652000069618225 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 92.0 | NORMAL |
| 2022-04-04 03:00:00 | 92.29 | GOOD | 0.5508000254631042 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 91.20999908447266 | NORMAL |
| 2022-04-04 03:10:00 | 88.28 | GOOD | 1.4005000591278076 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 78.91000366210938 | NORMAL |
| 2022-04-04 03:20:00 | 87.49 | GOOD | 1.0123000144958496 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 85.62999725341797 | NORMAL |
| 2022-04-04 03:30:00 | 88.58 | GOOD | 0.5569000244140625 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 91.1500015258789 | NORMAL |
| 2022-04-04 03:40:00 | 86.91 | GOOD | 1.1777000427246094 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 83.0199966430664 | NORMAL |
| 2022-04-04 03:50:00 | 88.04 | GOOD | 0.6047999858856201 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 90.66999816894531 | NORMAL |
| 2022-04-04 04:00:00 | 88.53 | GOOD | 0.6998999714851379 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 89.66000366210938 | NORMAL |

### Bottom 10 Records

| Timestamp | HealthIndex | HealthZone | FusedZ | RunID | EquipID | RawHealthIndex | QualityFlag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | 93.42 | GOOD | -0.22179999947547913 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 93.9000015258789 | NORMAL |
| 2025-09-14 22:30:00 | 93.21 | GOOD | -0.18199999630451202 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 94.16999816894531 | NORMAL |
| 2025-09-14 22:00:00 | 92.8 | GOOD | -0.22050000727176666 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 93.91000366210938 | NORMAL |
| 2025-09-14 21:30:00 | 92.33 | GOOD | -0.23970000445842743 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 93.77999877929688 | NORMAL |
| 2025-09-14 21:00:00 | 91.71 | GOOD | -0.3824999928474426 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 92.69999694824219 | NORMAL |
| 2025-09-14 20:30:00 | 91.29 | GOOD | -0.413100004196167 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 92.44000244140625 | NORMAL |
| 2025-09-14 20:00:00 | 90.8 | GOOD | -0.5521000027656555 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 91.19000244140625 | NORMAL |
| 2025-09-14 19:30:00 | 90.63 | GOOD | -0.5964999794960022 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 90.76000213623047 | NORMAL |
| 2025-09-14 19:00:00 | 90.57 | GOOD | -0.5442000031471252 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 91.2699966430664 | NORMAL |
| 2025-09-14 18:30:00 | 90.27 | GOOD | -0.6365000009536743 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 | 90.3499984741211 | NORMAL |

---


## dbo.ACM_HealthZoneByPeriod

**Primary Key:** No primary key  
**Row Count:** 1,671  
**Date Range:** 2022-04-04 00:00:00 to 2025-09-14 00:00:00  

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
| 2022-04-04 00:00:00 | DAY | GOOD | 89.9 | 116 | 129 | 2022-04-04 00:00:00 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 00:00:00 | DAY | WATCH | 9.3 | 12 | 129 | 2022-04-04 00:00:00 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 00:00:00 | DAY | ALERT | 0.8 | 1 | 129 | 2022-04-04 00:00:00 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-05 00:00:00 | DAY | GOOD | 93.3 | 14 | 15 | 2022-04-05 00:00:00 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-05 00:00:00 | DAY | WATCH | 6.7 | 1 | 15 | 2022-04-05 00:00:00 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-05 00:00:00 | DAY | ALERT | 0.0 | 0 | 15 | 2022-04-05 00:00:00 | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-05 00:00:00 | DAY | GOOD | 97.7 | 126 | 129 | 2022-04-05 00:00:00 | 88401CDD-1BC5-4ECB-A340-BC98624CDC94 | 5092 |
| 2022-04-05 00:00:00 | DAY | WATCH | 0.8 | 1 | 129 | 2022-04-05 00:00:00 | 88401CDD-1BC5-4ECB-A340-BC98624CDC94 | 5092 |
| 2022-04-05 00:00:00 | DAY | ALERT | 1.6 | 2 | 129 | 2022-04-05 00:00:00 | 88401CDD-1BC5-4ECB-A340-BC98624CDC94 | 5092 |
| 2022-04-05 00:00:00 | DAY | GOOD | 94.8 | 92 | 97 | 2022-04-05 00:00:00 | C51C247D-6B86-4CF4-93C4-9547CD9E4069 | 5092 |

### Bottom 10 Records

| PeriodStart | PeriodType | HealthZone | ZonePct | ZoneCount | TotalPoints | Date | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 00:00:00 | DAY | GOOD | 97.9 | 46 | 47 | 2025-09-14 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 00:00:00 | DAY | WATCH | 2.1 | 1 | 47 | 2025-09-14 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 00:00:00 | DAY | ALERT | 0.0 | 0 | 47 | 2025-09-14 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-13 00:00:00 | DAY | GOOD | 95.8 | 46 | 48 | 2025-09-13 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-13 00:00:00 | DAY | WATCH | 4.2 | 2 | 48 | 2025-09-13 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-13 00:00:00 | DAY | ALERT | 0.0 | 0 | 48 | 2025-09-13 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 00:00:00 | DAY | GOOD | 68.8 | 33 | 48 | 2025-09-12 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 00:00:00 | DAY | WATCH | 2.1 | 1 | 48 | 2025-09-12 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 00:00:00 | DAY | ALERT | 29.2 | 14 | 48 | 2025-09-12 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-11 00:00:00 | DAY | GOOD | 41.7 | 20 | 48 | 2025-09-11 00:00:00 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |

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


## dbo.ACM_MultivariateForecast

**Primary Key:** ForecastID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ForecastID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| SensorName | nvarchar | NO | 128 | — |
| ForecastTimestamp | datetime2 | NO | — | — |
| ForecastValue | float | NO | 53 | — |
| CiLower | float | YES | 53 | — |
| CiUpper | float | YES | 53 | — |
| ForecastStd | float | YES | 53 | — |
| Method | nvarchar | NO | 50 | — |
| VarOrder | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (getdate()) |

---


## dbo.ACM_OMRContributionsLong

**Primary Key:** No primary key  
**Row Count:** 8,029,365  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_energy_1 | 0.7614771951934958 | 0.00862988115966095 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_energy_2 | 0.052951811632267747 | 0.0006001070609331968 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_kurt | 0.039372295116836174 | 0.0004462093283766111 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_mad | 1.671102326381367 | 0.018938734571873263 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_mean | 4.139474414371678 | 0.04691299028384852 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_med | 17.990738552238838 | 0.20389046009565648 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_rz | 0.04183578018082719 | 0.00047412819905984937 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_skew | 0.7090224113260095 | 0.008035406953618473 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_slope | 0.23095448729438747 | 0.0026174254346967383 | 0.9236524701118469 | 2025-12-13 11:41:54 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | power_29_avg_std | 47.33952585879031 | 0.5365025832615328 | 0.9236524701118469 | 2025-12-13 11:41:54 |

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
**Row Count:** 30,201  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:30:00 | 0.9236999750137329 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:40:00 | 4.117199897766113 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 02:50:00 | 3.174099922180176 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 03:00:00 | 2.396699905395508 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 03:10:00 | 0.8689000010490417 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 03:20:00 | 0.9409999847412109 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 03:30:00 | 1.5472999811172485 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 03:40:00 | 0.1582999974489212 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 03:50:00 | -0.048500001430511475 | 0.1 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 2022-04-19 04:00:00 | -0.3928000032901764 | 0.1 | 2025-12-13 11:41:57 |

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
**Row Count:** 90  

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
| 1129 | 90edc87d-0f32-4d59-b32f-1ee7c714a926 | 1 | pls | 5 | 97 | 90 | 1.1910469897232634 | NULL | NULL |
| 1130 | d12f3400-a17c-4daf-9f69-4e08f5102cd6 | 1 | pls | 5 | 503 | 90 | 1.4801945775713412 | NULL | NULL |
| 1131 | 162f38f2-76ad-45b3-8954-22618b3c26be | 1 | pls | 5 | 399 | 90 | 1.2042133753658206 | NULL | NULL |
| 1132 | 4bf97db3-902f-460b-b6c2-b685ee80b6b9 | 1 | pls | 5 | 1411 | 90 | 1.4179095471967185 | NULL | NULL |
| 1133 | 04f37a78-e7fa-46cb-9b06-65d0ae7ce180 | 1 | pls | 5 | 1353 | 90 | 1.3907038142223689 | NULL | NULL |
| 1134 | fe2888af-6c3f-4574-b101-31e2d4303bbb | 1 | pls | 5 | 1442 | 90 | 1.3135110704386805 | NULL | NULL |

### Bottom 10 Records

| DiagnosticID | RunID | EquipID | ModelType | NComponents | TrainSamples | TrainFeatures | TrainResidualStd | TrainStartTime | TrainEndTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1239 | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 5000 | pca | 5 | 144 | 630 | 3.4043159952609816 | NULL | NULL |
| 1238 | 69732a1e-a2c5-44fd-a9b5-93fcedbe4c62 | 5000 | pca | 5 | 144 | 630 | 2.8047480001422467 | NULL | NULL |
| 1237 | 1362b1e7-3406-4352-858b-0b8464efa0e5 | 5000 | pca | 5 | 67 | 630 | 2.8975276232071856 | NULL | NULL |
| 1236 | 75e70d35-4ee3-4e09-a7db-26669cef9484 | 5000 | pca | 5 | 72 | 624 | 2.620989755919279 | NULL | NULL |
| 1235 | 2c963a3b-ece6-4004-b6a4-911defa955e2 | 5000 | pca | 5 | 144 | 630 | 1.975888383178358 | NULL | NULL |
| 1234 | 4dbe8707-50f4-463e-9e58-7e0e585588cd | 5000 | pca | 5 | 144 | 630 | 2.2808052424990914 | NULL | NULL |
| 1233 | 12f14125-8f0b-4969-b6c9-ceb2a4212edf | 5000 | pca | 5 | 144 | 624 | 2.707667077214721 | NULL | NULL |
| 1232 | f0a42054-5820-4769-a0ba-49307cab6cfc | 5000 | pca | 5 | 144 | 630 | 2.7625866800445427 | NULL | NULL |
| 1231 | c8281286-eb94-420e-85f1-a4428ab4a9cd | 5000 | pca | 5 | 144 | 628 | 2.227397410998237 | NULL | NULL |
| 1230 | 7eb9892b-9a43-41d2-8be7-5699c70214e8 | 5000 | pca | 5 | 144 | 622 | 2.3512912184805557 | NULL | NULL |

---


## dbo.ACM_PCA_Loadings

**Primary Key:** RecordID  
**Row Count:** 209,350  
**Date Range:** 2025-12-05 11:37:03 to 2025-12-13 12:32:40  

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
| 457265 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | wind_speed_4_avg_rz | wind_speed_4_avg_rz | 0.0274604240765112 | 2025-12-13 12:32:40 |
| 457264 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | wind_speed_3_std_rz | wind_speed_3_std_rz | 0.005728943968735776 | 2025-12-13 12:32:40 |
| 457263 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | wind_speed_3_min_rz | wind_speed_3_min_rz | 0.026461145543397825 | 2025-12-13 12:32:40 |
| 457262 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | wind_speed_3_max_rz | wind_speed_3_max_rz | -0.021023522448342363 | 2025-12-13 12:32:40 |
| 457261 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | wind_speed_3_avg_rz | wind_speed_3_avg_rz | 0.03216490485112909 | 2025-12-13 12:32:40 |
| 457260 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | sensor_9_avg_rz | sensor_9_avg_rz | 0.0036096280911734726 | 2025-12-13 12:32:40 |
| 457259 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | sensor_8_avg_rz | sensor_8_avg_rz | -0.03957409266945698 | 2025-12-13 12:32:40 |
| 457258 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | sensor_7_avg_rz | sensor_7_avg_rz | -0.015456924154539461 | 2025-12-13 12:32:40 |
| 457257 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | sensor_6_avg_rz | sensor_6_avg_rz | -0.04180296259806525 | 2025-12-13 12:32:40 |
| 457256 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | 5 | sensor_5_std_rz | sensor_5_std_rz | 0.030385293238638554 | 2025-12-13 12:32:40 |

---


## dbo.ACM_PCA_Metrics

**Primary Key:** RunID, EquipID, ComponentName, MetricType  
**Row Count:** 273  
**Date Range:** 2025-12-05 11:36:52 to 2025-12-13 12:32:46  

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
| 04f37a78-e7fa-46cb-9b06-65d0ae7ce180 | 1 | PCA | n_components | 5.0 | 2025-12-11 15:25:29 |
| 04f37a78-e7fa-46cb-9b06-65d0ae7ce180 | 1 | PCA | n_features | 90.0 | 2025-12-11 15:25:29 |
| 04f37a78-e7fa-46cb-9b06-65d0ae7ce180 | 1 | PCA | variance_explained | 0.6435759043114693 | 2025-12-11 15:25:29 |
| 06678ab8-a0a8-4bb8-87e9-33663eb309bd | 8632 | PCA | n_components | 5.0 | 2025-12-11 18:02:52 |
| 06678ab8-a0a8-4bb8-87e9-33663eb309bd | 8632 | PCA | n_features | 40.0 | 2025-12-11 18:02:52 |
| 06678ab8-a0a8-4bb8-87e9-33663eb309bd | 8632 | PCA | variance_explained | 0.6577009725085278 | 2025-12-11 18:02:52 |
| 0813f8c9-a231-494d-bdc8-4f85ddacb9de | 5000 | PCA | n_components | 5.0 | 2025-12-13 11:59:26 |
| 0813f8c9-a231-494d-bdc8-4f85ddacb9de | 5000 | PCA | n_features | 790.0 | 2025-12-13 11:59:26 |
| 0813f8c9-a231-494d-bdc8-4f85ddacb9de | 5000 | PCA | variance_explained | 0.6194415264414262 | 2025-12-13 11:59:26 |
| 0b61b75e-e221-49b0-8b08-bc26c0acc3d9 | 5092 | PCA | n_components | 5.0 | 2025-12-13 11:34:53 |

### Bottom 10 Records

| RunID | EquipID | ComponentName | MetricType | Value | Timestamp |
| --- | --- | --- | --- | --- | --- |
| ffad6ad5-e725-45b5-a792-14bf8330500d | 5000 | PCA | variance_explained | 0.6125961149610554 | 2025-12-13 12:13:55 |
| ffad6ad5-e725-45b5-a792-14bf8330500d | 5000 | PCA | n_features | 630.0 | 2025-12-13 12:13:55 |
| ffad6ad5-e725-45b5-a792-14bf8330500d | 5000 | PCA | n_components | 5.0 | 2025-12-13 12:13:55 |
| fed2c8d0-d011-4cbf-95c0-3c47eb8d7b29 | 5000 | PCA | variance_explained | 0.6125961149610554 | 2025-12-13 12:21:25 |
| fed2c8d0-d011-4cbf-95c0-3c47eb8d7b29 | 5000 | PCA | n_features | 630.0 | 2025-12-13 12:21:25 |
| fed2c8d0-d011-4cbf-95c0-3c47eb8d7b29 | 5000 | PCA | n_components | 5.0 | 2025-12-13 12:21:25 |
| fe2888af-6c3f-4574-b101-31e2d4303bbb | 1 | PCA | variance_explained | 0.5914009961403585 | 2025-12-11 15:26:05 |
| fe2888af-6c3f-4574-b101-31e2d4303bbb | 1 | PCA | n_features | 90.0 | 2025-12-11 15:26:05 |
| fe2888af-6c3f-4574-b101-31e2d4303bbb | 1 | PCA | n_components | 5.0 | 2025-12-11 15:26:05 |
| fc89c156-1a12-48b0-914f-a2b3c705d1e9 | 1 | PCA | variance_explained | 0.7896926933891086 | 2025-12-11 15:27:53 |

---


## dbo.ACM_PCA_Models

**Primary Key:** RecordID  
**Row Count:** 82  
**Date Range:** 2025-12-05 11:37:03 to 2025-12-13 12:32:40  

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
| 443 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | 2025-12-11 15:24:05 | 5 | {"SPE_P95_train": 3.4016430377960205, "T2_P95_train": 2.116558790206909} | [0.27989694184187136, 0.23426346832848857, 0.10448614396646073, 0.07837094925275088, 0.0711203784... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-18 00:00:00 |
| 444 | D12F3400-A17C-4DAF-9F69-4E08F5102CD6 | 1 | 2025-12-11 15:24:22 | 5 | {"SPE_P95_train": 6.432009220123291, "T2_P95_train": 6.628495216369629} | [0.19648372194546906, 0.1392822602032956, 0.1151769829100149, 0.08811291055681245, 0.073090252689... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 445 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 1 | 2025-12-11 15:24:39 | 5 | {"SPE_P95_train": 4.5547966957092285, "T2_P95_train": 6.338170051574707} | [0.21917876968606634, 0.16336473949400046, 0.11099733017459443, 0.0840586480193309, 0.07309745075... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-01-14 00:00:00 |
| 446 | 4BF97DB3-902F-460B-B6C2-B685EE80B6B9 | 1 | 2025-12-11 15:25:13 | 5 | {"SPE_P95_train": 4.653730392456055, "T2_P95_train": 5.959028244018555} | [0.2103212304457447, 0.13617566774773296, 0.10502305169124686, 0.0796131033956107, 0.057323346042... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-03-03 05:00:00 |
| 447 | 04F37A78-E7FA-46CB-9B06-65D0AE7CE180 | 1 | 2025-12-11 15:25:47 | 5 | {"SPE_P95_train": 4.577191352844238, "T2_P95_train": 6.197824954986572} | [0.24883393109296878, 0.15133635805194803, 0.09896181910096219, 0.0893060449216775, 0.05513775114... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-05-16 00:00:00 |
| 448 | FE2888AF-6C3F-4574-B101-31E2D4303BBB | 1 | 2025-12-11 15:26:22 | 5 | {"SPE_P95_train": 5.9401044845581055, "T2_P95_train": 3.519364356994629} | [0.1743959478541592, 0.14019479914465502, 0.11907144814200503, 0.08626290960671808, 0.07147589139... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2024-07-21 09:30:00 |

### Bottom 10 Records

| RecordID | RunID | EquipID | EntryDateTime | NComponents | TargetVar | VarExplainedJSON | ScalingSpecJSON | ModelVersion | TrainStartEntryDateTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 543 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2025-12-13 12:32:40 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.2639908861965811, 0.17781156908304227, 0.093904688439134, 0.0543830405781972, 0.04849108349829... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-09-01 06:10:00 |
| 542 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 2025-12-13 12:31:52 | 5 | {"SPE_P95_train": 10.0, "T2_P95_train": 10.0} | [0.2685698676209199, 0.1879437768741901, 0.14157691012893295, 0.08743533660418615, 0.043160849854... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-31 06:10:00 |
| 541 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 | 2025-12-13 12:31:03 | 5 | {"SPE_P95_train": 10.0, "T2_P95_train": 10.0} | [0.2808360649892274, 0.14155723168408996, 0.11234932589924672, 0.08656809252639226, 0.05414302521... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-30 06:10:00 |
| 540 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 2025-12-13 12:29:59 | 5 | {"SPE_P95_train": 10.0, "T2_P95_train": 10.0} | [0.2949021674253958, 0.12394137130715417, 0.09573625857362564, 0.06086572010810422, 0.04634678049... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-29 06:10:00 |
| 539 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | 2025-12-13 12:29:05 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.29146535377091026, 0.1279260874596147, 0.10790215390779175, 0.05988179063178455, 0.04258375974... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-28 06:10:00 |
| 538 | 12F14125-8F0B-4969-B6C9-CEB2A4212EDF | 5000 | 2025-12-13 12:28:13 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.26572625040040715, 0.12070657366366468, 0.09157921959601623, 0.06877491624597776, 0.0612180569... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-27 06:10:00 |
| 537 | F0A42054-5820-4769-A0BA-49307CAB6CFC | 5000 | 2025-12-13 12:27:28 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.21974765645987782, 0.1487525266115244, 0.1383346756526739, 0.09036132262625145, 0.054427532036... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-26 06:10:00 |
| 536 | C8281286-EB94-420E-85F1-A4428AB4A9CD | 5000 | 2025-12-13 12:26:38 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.2511383108532881, 0.15055828827796253, 0.12597783150830213, 0.07181159705257965, 0.05059793625... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-25 06:10:00 |
| 535 | 7EB9892B-9A43-41D2-8BE7-5699C70214E8 | 5000 | 2025-12-13 12:25:41 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.25902804666035834, 0.14718574193013856, 0.12939024776217162, 0.06386826733469098, 0.0489352825... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-24 06:10:00 |
| 534 | 5BA6BC6D-56B6-4719-BE05-0E9BEEDF38C7 | 5000 | 2025-12-13 12:25:02 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.24284406547357165, 0.1596674377186631, 0.14534250572177243, 0.0655364654757473, 0.049444771876... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-23 06:10:00 |

---


## dbo.ACM_RUL

**Primary Key:** EquipID, RunID  
**Row Count:** 77  
**Date Range:** 2025-12-12 04:31:12 to 2025-12-20 12:32:38  

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
| 1 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-18 15:24:38 | Multipath | 1000 |
| 1 | 32D469E6-5161-4535-87E9-2D5FCBEDBEFE | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-18 15:27:26 | Multipath | 1000 |
| 1 | AEEF77AD-7C2C-4102-98E9-30F912E96DFF | 33.0 | 26.640832777868553 | 33.0 | 37.998843364071455 | 0.5612813786308118 | 2025-12-13 00:26:54 | Multipath | 1000 |
| 1 | FE2888AF-6C3F-4574-B101-31E2D4303BBB | 37.0 | 31.524985453811126 | 37.0 | 41.54540207805145 | 0.5699385628178081 | 2025-12-13 04:26:21 | Multipath | 1000 |
| 1 | D12F3400-A17C-4DAF-9F69-4E08F5102CD6 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-18 15:24:22 | Multipath | 1000 |
| 1 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-18 15:28:17 | Multipath | 1000 |
| 1 | 04F37A78-E7FA-46CB-9B06-65D0AE7CE180 | 16.5 | 14.307113899225705 | 16.5 | 18.746096059608583 | 0.570823110467152 | 2025-12-12 07:55:47 | Multipath | 1000 |
| 1 | FC89C156-1A12-48B0-914F-A2B3C705D1E9 | 59.0 | 51.30827053515425 | 59.0 | 65.86466183105719 | 0.5734433703520754 | 2025-12-14 02:28:02 | Multipath | 1000 |
| 1 | DC06050E-6596-43E3-A725-A2DAFB28567D | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-18 15:27:46 | Multipath | 1000 |
| 1 | 4BF97DB3-902F-460B-B6C2-B685EE80B6B9 | 56.5 | 47.36148049398854 | 56.5 | 63.33140560678576 | 0.5688416502279786 | 2025-12-13 23:55:12 | Multipath | 1000 |

### Bottom 10 Records

| EquipID | RunID | RUL_Hours | P10_LowerBound | P50_Median | P90_UpperBound | Confidence | FailureTime | Method | NumSimulations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8634 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 7.691666666666666 | 7.069687661238081 | 7.691666666666666 | 8.241526916296385 | 0.5850434998154206 | 2025-12-13 16:21:00 | Multipath | 1000 |
| 8634 | 5526AB2D-34AE-421F-8069-6B6BDB120CBE | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 08:41:14 | Multipath | 1000 |
| 8634 | 52828146-2917-47C6-A32D-54A2D2C2B906 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 08:50:19 | Multipath | 1000 |
| 8634 | 4CC12D3B-2F8A-471A-B48F-41FA2C387680 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 08:53:28 | Multipath | 1000 |
| 8634 | B3E09FDE-A100-4C91-B853-2AD51F8FB52E | 36.0 | 32.44425863423264 | 36.0 | 38.45482948444031 | 0.5829156156915735 | 2025-12-14 20:47:01 | Multipath | 1000 |
| 8634 | 79E7DD32-441E-43EC-8A52-18707041B651 | 3.6333333333333333 | 3.3054366594762836 | 3.6333333333333333 | 3.9012145853780025 | 0.5836395068270722 | 2025-12-13 12:21:49 | Multipath | 1000 |
| 8632 | 202974D5-76B2-4D1E-BA3D-E321EEE5D336 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-18 18:02:44 | Multipath | 1000 |
| 8632 | 682A58D9-4C3C-4B2C-8EC5-DC9852F27CC7 | 10.5 | 1.9733950205828559 | 10.5 | 35.00960101943117 | 0.36 | 2025-12-12 04:31:12 | Multipath | 1000 |
| 8632 | 96A0DF64-6A48-400C-AD81-D2F4C1704750 | 168.0 | 17.26720643009999 | 168.0 | 170.2348182710401 | 0.46706641095574686 | 2025-12-18 18:01:41 | Multipath | 1000 |
| 8632 | 06678AB8-A0A8-4BB8-87E9-33663EB309BD | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.5971691236664811 | 2025-12-18 18:03:15 | Multipath | 1000 |

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
**Row Count:** 1,499  
**Date Range:** 2025-12-01 05:03:49 to 2025-12-13 07:02:52  

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
| 2067 | 5000 | 2025-12-13 07:02:52 | Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 2066 | 5000 | 2025-12-13 07:02:03 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 07:02:45 |
| 2065 | 5000 | 2025-12-13 07:01:17 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 07:01:57 |
| 2064 | 5000 | 2025-12-13 07:00:12 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 07:01:10 |
| 2063 | 5000 | 2025-12-13 06:59:23 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 07:00:04 |
| 2062 | 5000 | 2025-12-13 06:58:23 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 06:59:16 |
| 2061 | 5000 | 2025-12-13 06:57:41 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 06:58:17 |
| 2060 | 5000 | 2025-12-13 06:56:52 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 06:57:33 |
| 2059 | 5000 | 2025-12-13 06:55:53 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 06:56:44 |
| 2058 | 5000 | 2025-12-13 06:55:13 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-13 06:55:46 |

---


## dbo.ACM_RegimeDwellStats

**Primary Key:** No primary key  
**Row Count:** 276  

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
| 0 | 2 | 21600.0 | 21600.0 | 600.0 | 42600.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 1 | 2 | 20400.0 | 20400.0 | 2400.0 | 38400.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0 | 16 | 26325.0 | 23400.0 | 1800.0 | 55800.0 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| 1 | 16 | 68962.5 | 28800.0 | 1800.0 | 230400.0 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| 0 | 2 | 23700.0 | 23700.0 | 22200.0 | 25200.0 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 1 | 1 | 17400.0 | 17400.0 | 17400.0 | 17400.0 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 2 | 2 | 9300.0 | 9300.0 | 6000.0 | 12600.0 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 0 | 2 | 300.0 | 300.0 | 0.0 | 600.0 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| 1 | 3 | 3600.0 | 600.0 | 600.0 | 9600.0 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| 2 | 1 | 55800.0 | 55800.0 | 55800.0 | 55800.0 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |

### Bottom 10 Records

| RegimeLabel | Runs | MeanSeconds | MedianSeconds | MinSeconds | MaxSeconds | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 172800.0 | 129600.0 | 7200.0 | 450000.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 1 | 7 | 21600.0 | 21600.0 | 10800.0 | 32400.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 2 | 7 | 24685.714285714286 | 21600.0 | 21600.0 | 32400.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 3 | 9 | 14800.0 | 18000.0 | 0.0 | 25200.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 4 | 6 | 43200.0 | 30600.0 | 7200.0 | 93600.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 5 | 5 | 19440.0 | 18000.0 | 10800.0 | 36000.0 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0 | 4 | 10500.0 | 6000.0 | 600.0 | 29400.0 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| 1 | 1 | 27000.0 | 27000.0 | 27000.0 | 27000.0 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| 2 | 3 | 4200.0 | 2400.0 | 1200.0 | 9000.0 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| 0 | 4 | 14400.0 | 7800.0 | 3000.0 | 39000.0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 |

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
**Row Count:** 276  

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
| 0 | 74 | 51.39 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 1 | 70 | 48.61 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0 | 250 | 28.44 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| 1 | 629 | 71.56 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| 0 | 81 | 56.25 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 1 | 30 | 20.83 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 2 | 33 | 22.92 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 0 | 3 | 4.48 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| 1 | 21 | 31.34 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| 2 | 17 | 25.37 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |

### Bottom 10 Records

| RegimeLabel | RecordCount | Percentage | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 0 | 245 | 48.51 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 1 | 49 | 9.7 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 2 | 55 | 10.89 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 3 | 46 | 9.11 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 4 | 78 | 15.45 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 5 | 32 | 6.34 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| 0 | 74 | 51.39 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| 1 | 46 | 31.94 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| 2 | 24 | 16.67 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| 0 | 100 | 69.44 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 |

---


## dbo.ACM_RegimeStability

**Primary Key:** No primary key  
**Row Count:** 90  

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
| RegimeStability | 97.9591836734694 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| RegimeStability | 96.5934065934066 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| RegimeStability | 97.2972972972973 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| RegimeStability | 88.15789473684211 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| RegimeStability | 98.63013698630138 | 5BA6BC6D-56B6-4719-BE05-0E9BEEDF38C7 | 5000 |
| RegimeStability | 91.71974522292994 | FFAD6AD5-E725-45B5-A792-14BF8330500D | 5000 |
| RegimeStability | 96.0 | E1596C1E-4B19-4E7A-A7EF-168EB50EC183 | 5092 |
| RegimeStability | 97.03504043126685 | 79E7DD32-441E-43EC-8A52-18707041B651 | 8634 |
| RegimeStability | 91.71974522292994 | C4F1062D-682C-45CA-97B3-1995F206A9A3 | 5000 |
| RegimeStability | 95.09803921568627 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 |

### Bottom 10 Records

| MetricName | MetricValue | RunID | EquipID |
| --- | --- | --- | --- |
| RegimeStability | 93.00184162062615 | 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 |
| RegimeStability | 95.36423841059602 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| RegimeStability | 94.73684210526315 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 |
| RegimeStability | 98.63013698630138 | CF9514D8-C683-4A5B-8DE3-EC6A9D6E6A92 | 5000 |
| RegimeStability | 91.71974522292994 | 55DB3B5A-ED32-4DFA-970B-E41DD242B1F1 | 5000 |
| RegimeStability | 95.43973941368078 | 202974D5-76B2-4D1E-BA3D-E321EEE5D336 | 8632 |
| RegimeStability | 93.14359637774903 | 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 8634 |
| RegimeStability | 97.12707182320442 | 682A58D9-4C3C-4B2C-8EC5-DC9852F27CC7 | 8632 |
| RegimeStability | 92.30769230769232 | 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 |
| RegimeStability | 96.0 | AE60C5F3-E31D-4D61-A4D5-D9E6BFDCC217 | 5092 |

---


## dbo.ACM_RegimeState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 6  
**Date Range:** 2025-12-05 06:08:22 to 2025-12-13 07:02:50  

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
| 1 | 1 | 2 | [[0.4095362044250003, 0.599265353066298, 0.571150233428146, 0.6018965218294224, 0.582693272831768... | [] | [] | [] | [] | 0 | 0.8049672447757448 |
| 2621 | 1 | 6 | [[0.9438815116882324, -1.0460931062698364, 0.33586952090263367], [0.3532538414001465, 0.828715622... | [-1.9585829837005225e-09, 2.1464276511913523e-09, 4.254157688479014e-09] | [6.7357066205517295, 6.220930007614507, 4.438979977146924] | [] | [] | 3 | 0.5188907384872437 |
| 5000 | 1 | 5 | [[0.8311897859650423, 0.9609392805572362, 0.1564433139827207, 1.1030944154780602, -0.728272153890... | [] | [] | [] | [] | 0 | 0.44026781403151655 |
| 5092 | 1 | 2 | [[-0.45438787880016485, -0.4528699836702049, -0.43297153329443333, -0.40982296695685727, -0.45516... | [] | [] | [] | [] | 0 | 0.6061100682115345 |
| 8632 | 1 | 2 | [[0.9969194272095111, 1.0084716166025738, 0.9446738676541959], [-0.8613621319158566, -0.871489332... | [] | [] | [] | [] | 0 | 0.7097052096345993 |
| 8634 | 1 | 3 | [[-1.9899488317455225, 1.3363767656666485, 1.1044912909885563], [0.17542983617897057, -0.73885968... | [] | [] | [] | [] | 0 | 0.6634951182444943 |

---


## dbo.ACM_RegimeStats

**Primary Key:** No primary key  
**Row Count:** 276  
**Date Range:** 2025-12-05 11:37:00 to 2025-12-13 12:32:56  

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
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 0 | 51.39 | 22200.0 | 0.4214 | 1.3841 | 2025-12-13 11:41:57 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | 1 | 48.61 | 42000.0 | -0.4454 | 0.1477 | 2025-12-13 11:41:57 |
| DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 | 0 | 28.44 | 28125.0 | 0.2044 | 2.0274 | 2025-12-11 18:01:56 |
| DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 | 1 | 71.56 | 75480.0 | -0.0813 | 0.8647 | 2025-12-11 18:01:56 |
| 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 | 0 | 56.25 | 48600.0 | -0.4467 | 0.7034 | 2025-12-13 12:05:03 |
| 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 | 1 | 20.83 | 18000.0 | 0.3224 | 1.0074 | 2025-12-13 12:05:03 |
| 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 | 2 | 22.92 | 9900.0 | 0.8032 | 1.1767 | 2025-12-13 12:05:03 |
| 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 0 | 4.48 | 900.0 | 0.1752 | 0.2859 | 2025-12-13 12:31:19 |
| 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 1 | 31.34 | 4200.0 | 0.3312 | 0.644 | 2025-12-13 12:31:19 |
| 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 2 | 25.37 | 10200.0 | -0.1278 | 0.4927 | 2025-12-13 12:31:19 |

### Bottom 10 Records

| RunID | EquipID | RegimeLabel | OccupancyPct | AvgDwellSeconds | FusedMean | FusedP90 | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 2025-12-05 11:38:00 |
| ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 | 2 | 16.67 | 4800.0 | 1.0807 | 1.6211 | 2025-12-13 12:02:09 |
| ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 | 1 | 31.94 | 27600.0 | 0.0385 | 1.4583 | 2025-12-13 12:02:09 |
| ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 | 0 | 51.39 | 14800.0 | -0.3745 | 0.3571 | 2025-12-13 12:02:09 |
| F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 | 2 | 6.94 | 6000.0 | 1.2481 | 1.5053 | 2025-12-13 11:44:24 |

---


## dbo.ACM_RegimeTimeline

**Primary Key:** No primary key  
**Row Count:** 30,201  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| 2022-04-04 02:30:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:40:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:50:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:00:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:10:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:20:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:30:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:40:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:50:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 04:00:00 | 1 | unknown | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |

### Bottom 10 Records

| Timestamp | RegimeLabel | RegimeState | RunID | EquipID |
| --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:30:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:00:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 21:30:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 21:00:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 20:30:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 20:00:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 19:30:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 19:00:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 18:30:00 | 0 | unknown | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |

---


## dbo.ACM_RegimeTransitions

**Primary Key:** No primary key  
**Row Count:** 379  

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
| 0.0 | 1.0 | 1 | 0.3333 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 1.0 | 0.0 | 2 | 0.6667 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| 0.0 | 1.0 | 15 | 0.4839 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| 1.0 | 0.0 | 16 | 0.5161 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| 0.0 | 2.0 | 1 | 0.25 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 1.0 | 2.0 | 1 | 0.25 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 2.0 | 0.0 | 1 | 0.25 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 2.0 | 1.0 | 1 | 0.25 | 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 |
| 0.0 | 1.0 | 1 | 0.1111 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |
| 0.0 | 4.0 | 1 | 0.1111 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 |

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
**Row Count:** 332  

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
| 5240 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | NULL | NULL | NULL |
| 5239 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | NULL | NULL | NULL |
| 5238 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | NULL | NULL | NULL |
| 5237 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 | NULL | NULL | NULL |
| 5236 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | NULL | NULL | NULL |
| 5235 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | NULL | NULL | NULL |
| 5234 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | NULL | NULL | NULL |
| 5233 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | NULL | NULL | NULL |
| 5232 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | NULL | NULL | NULL |
| 5231 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | NULL | NULL | NULL |

---


## dbo.ACM_RunLogs

**Primary Key:** LogID  
**Row Count:** 403,502  
**Date Range:** 2025-12-02 05:59:43 to 2025-12-13 07:02:57  

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
**Row Count:** 27,993  
**Date Range:** 2025-12-01 17:15:57 to 2025-12-13 12:32:51  

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
**Row Count:** 82  
**Date Range:** 2022-04-04 02:30:00 to 2025-07-06 21:09:00  

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
| 443 | 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | 2023-10-15 00:00:00 | 2023-10-15 23:59:59 | 97 | 97 | 9 | 100.0 | NULL |
| 444 | D12F3400-A17C-4DAF-9F69-4E08F5102CD6 | 1 | 2023-10-15 00:00:00 | 2023-12-24 02:20:59 | 503 | 503 | 9 | 100.0 | NULL |
| 445 | 162F38F2-76AD-45B3-8954-22618B3C26BE | 1 | 2023-12-24 02:21:00 | 2024-03-03 04:41:59 | 399 | 399 | 9 | 100.0 | NULL |
| 446 | 4BF97DB3-902F-460B-B6C2-B685EE80B6B9 | 1 | 2024-03-03 04:42:00 | 2024-05-12 07:02:59 | 1411 | 1411 | 9 | 100.0 | NULL |
| 447 | 04F37A78-E7FA-46CB-9B06-65D0AE7CE180 | 1 | 2024-05-12 07:03:00 | 2024-07-21 09:23:59 | 1354 | 1354 | 9 | 100.0 | NULL |
| 448 | FE2888AF-6C3F-4574-B101-31E2D4303BBB | 1 | 2024-07-21 09:24:00 | 2024-09-29 11:44:59 | 1443 | 1443 | 9 | 100.0 | NULL |

### Bottom 10 Records

| RecordID | RunID | EquipID | WindowStartEntryDateTime | WindowEndEntryDateTime | SamplesIn | SamplesKept | SensorsKept | CadenceOKPct | DriftP95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 543 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-01 06:09:58 | 2022-09-02 06:09:57 | 144 | 144 | 81 | 100.0 | NULL |
| 542 | 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | 2022-08-31 06:09:58 | 2022-09-01 06:09:57 | 67 | 67 | 81 | 100.0 | NULL |
| 541 | 75E70D35-4EE3-4E09-A7DB-26669CEF9484 | 5000 | 2022-08-30 06:09:58 | 2022-08-31 06:09:57 | 72 | 72 | 81 | 100.0 | NULL |
| 540 | 2C963A3B-ECE6-4004-B6A4-911DEFA955E2 | 5000 | 2022-08-29 06:09:58 | 2022-08-30 06:09:57 | 144 | 144 | 81 | 100.0 | NULL |
| 539 | 4DBE8707-50F4-463E-9E58-7E0E585588CD | 5000 | 2022-08-28 06:09:58 | 2022-08-29 06:09:57 | 144 | 144 | 81 | 100.0 | NULL |
| 538 | 12F14125-8F0B-4969-B6C9-CEB2A4212EDF | 5000 | 2022-08-27 06:09:58 | 2022-08-28 06:09:57 | 144 | 144 | 81 | 100.0 | NULL |
| 537 | F0A42054-5820-4769-A0BA-49307CAB6CFC | 5000 | 2022-08-26 06:09:58 | 2022-08-27 06:09:57 | 144 | 144 | 81 | 100.0 | NULL |
| 536 | C8281286-EB94-420E-85F1-A4428AB4A9CD | 5000 | 2022-08-25 06:09:58 | 2022-08-26 06:09:57 | 144 | 144 | 81 | 100.0 | NULL |
| 535 | 7EB9892B-9A43-41D2-8BE7-5699C70214E8 | 5000 | 2022-08-24 06:09:58 | 2022-08-25 06:09:57 | 144 | 144 | 81 | 100.0 | NULL |
| 534 | 5BA6BC6D-56B6-4719-BE05-0E9BEEDF38C7 | 5000 | 2022-08-23 06:09:58 | 2022-08-24 06:09:57 | 144 | 144 | 81 | 100.0 | NULL |

---


## dbo.ACM_Runs

**Primary Key:** RunID  
**Row Count:** 232  
**Date Range:** 2025-12-05 06:06:49 to 2025-12-13 07:02:43  

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
| 29877AB1-CD05-4A11-BBD2-010F14E71B62 | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:21:19 | 2025-12-13 03:21:19 | 0 |  | 0 | 0 | 0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | WFA_TURBINE_92 | 2025-12-13 06:11:31 | 2025-12-13 06:12:14 | 42 |  | 144 | 4819 | 2 |
| 495FC058-DD31-4E1B-8D56-023B69A8123F | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:18:45 | 2025-12-13 03:18:46 | 0 |  | 0 | 0 | 0 |
| DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 | WIND_TURBINE | 2025-12-11 12:31:46 | 2025-12-11 12:32:13 | 27 |  | 879 | 5487 | 6 |
| 943FB14A-3322-4563-ADCF-02EBDA06D6A2 | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:15:11 | 2025-12-13 03:15:11 | 0 |  | 0 | 0 | 0 |
| 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 | WFA_TURBINE_0 | 2025-12-13 06:34:36 | 2025-12-13 06:35:20 | 43 |  | 144 | 0 | 2 |
| ECF1B048-3C9F-432E-8495-047931ED3351 | 2621 | NULL | 2025-12-05 06:08:08 | 2025-12-05 06:08:08 | NULL |  | 0 | 0 | NULL |
| 9D740610-A7FA-445D-BA32-0629B40ADDEC | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:17:37 | 2025-12-13 03:17:37 | 0 |  | 0 | 0 | 0 |
| CA34A20C-D111-41BF-BDDD-06BFC475F47F | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:21:16 | 2025-12-13 03:21:16 | 0 |  | 0 | 0 | 0 |
| 92803781-4EC3-48FA-8D7F-072918C480C7 | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:24:26 | 2025-12-13 03:24:27 | 0 |  | 0 | 0 | 0 |

### Bottom 10 Records

| RunID | EquipID | EquipName | StartedAt | CompletedAt | DurationSeconds | ConfigSignature | TrainRowCount | ScoreRowCount | EpisodeCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29DB0F1F-1265-41F2-BA63-FD8248613D60 | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:09:55 | 2025-12-13 03:09:55 | 0 |  | 0 | 0 | 0 |
| 419D52D9-9467-43FD-8393-FD10867A84E8 | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:23:41 | 2025-12-13 03:23:41 | 0 |  | 0 | 0 | 0 |
| 9C3103E1-095D-4B07-9BDC-FBF72DC634CB | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:21:33 | 2025-12-13 03:21:34 | 0 |  | 0 | 0 | 0 |
| BA05B1EA-B53C-4ABB-9B88-FB2CEEBD2925 | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:21:02 | 2025-12-13 03:21:02 | 0 |  | 0 | 0 | 0 |
| 1707E493-721F-478C-85BE-F779A1A20CB9 | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:23:52 | 2025-12-13 03:23:52 | 0 |  | 0 | 0 | 0 |
| A17CB11C-3F80-4B8B-BDB2-F676A902237D | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:14:21 | 2025-12-13 03:14:21 | 0 |  | 0 | 0 | 0 |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | NULL | 2025-12-05 06:07:43 | 2025-12-05 06:08:04 | NULL |  | 505 | 3845 | NULL |
| ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 | WFA_TURBINE_0 | 2025-12-13 06:31:44 | 2025-12-13 06:32:21 | 36 |  | 144 | 4819 | 2 |
| F8B82D84-7F18-4EF0-A794-F13DAC33FAE5 | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:09:58 | 2025-12-13 03:09:58 | 0 |  | 0 | 0 | 0 |
| 23F180C7-7F66-4CF4-AD90-EFC3795FE7EE | 8634 | ELECTRIC_MOTOR | 2025-12-13 03:11:19 | 2025-12-13 03:11:19 | 0 |  | 0 | 0 | 0 |

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
**Row Count:** 169,884  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| 4464270 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 06:00:00 | NULL | NULL | NULL | NULL | NULL |
| 4464269 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 05:50:00 | NULL | NULL | NULL | NULL | NULL |
| 4464268 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 05:40:00 | NULL | NULL | NULL | NULL | NULL |
| 4464267 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 05:30:00 | NULL | NULL | NULL | NULL | NULL |
| 4464266 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 05:20:00 | NULL | NULL | NULL | NULL | NULL |
| 4464265 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 05:10:00 | NULL | NULL | NULL | NULL | NULL |
| 4464264 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 05:00:00 | NULL | NULL | NULL | NULL | NULL |
| 4464263 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 04:50:00 | NULL | NULL | NULL | NULL | NULL |
| 4464262 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 04:40:00 | NULL | NULL | NULL | NULL | NULL |
| 4464261 | 69732A1E-A2C5-44FD-A9B5-93FCEDBE4C62 | 5000 | 2022-09-02 04:30:00 | NULL | NULL | NULL | NULL | NULL |

---


## dbo.ACM_Scores_Wide

**Primary Key:** No primary key  
**Row Count:** 30,201  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| 2022-04-04 02:30:00 | -2.6316685676574707 | -2.5272278785705566 | -0.6749735474586487 | 6.235900946194306e-05 | 1.27415931224823 | 0.9093510508537292 | -2.6512813568115234 | NULL | NULL |
| 2022-04-04 02:40:00 | 1.603296160697937 | -2.5272271633148193 | -0.6749735474586487 | -0.00023293188132811338 | 1.631801962852478 | 0.8110426068305969 | -2.647142171859741 | NULL | NULL |
| 2022-04-04 02:50:00 | 1.634392261505127 | -2.5272274017333984 | -0.6749735474586487 | -6.226979894563556e-05 | 1.7581665515899658 | 0.8740924000740051 | -2.632643461227417 | NULL | NULL |
| 2022-04-04 03:00:00 | 2.0919880867004395 | -2.5272271633148193 | -0.6749735474586487 | 0.005716688930988312 | 1.755974531173706 | 0.9705751538276672 | -2.6083662509918213 | NULL | NULL |
| 2022-04-04 03:10:00 | 6.0496649742126465 | -2.5272257328033447 | -0.6749734878540039 | -0.002921238774433732 | 2.7715322971343994 | 1.2988232374191284 | -2.5521414279937744 | NULL | NULL |
| 2022-04-04 03:20:00 | 4.25379753112793 | -2.5272269248962402 | -0.6749734878540039 | -0.027915693819522858 | 2.5713114738464355 | 0.8131898045539856 | -2.4850211143493652 | NULL | NULL |
| 2022-04-04 03:30:00 | 1.769059658050537 | -2.5272278785705566 | -0.6749735474586487 | -0.03191104158759117 | 2.4616143703460693 | 0.49141180515289307 | -2.4237284660339355 | NULL | NULL |
| 2022-04-04 03:40:00 | 3.3048226833343506 | -2.4652116298675537 | -0.6748861074447632 | 3.797358751296997 | 2.10090708732605 | 0.1553295999765396 | -2.3481740951538086 | NULL | NULL |
| 2022-04-04 03:50:00 | 1.2151697874069214 | -2.418144941329956 | -0.6748138666152954 | 2.853015422821045 | 2.0927846431732178 | 0.04534095153212547 | -2.2795603275299072 | NULL | NULL |
| 2022-04-04 04:00:00 | 1.2971237897872925 | -2.3760218620300293 | -0.6747872233390808 | 2.8596231937408447 | 1.8063793182373047 | 0.09757348895072937 | -2.212995767593384 | NULL | NULL |

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
**Row Count:** 4,456  
**Date Range:** 2022-04-04 00:00:00 to 2025-09-14 00:00:00  

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
| 2022-04-04 00:00:00 | 2022-04-04 00:00:00 | DAY | 86400.0 | ar1 | 7.75 | 6.049699783325195 | 0.8845000267028809 | 129 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-04 00:00:00 | 2022-04-04 00:00:00 | DAY | 86400.0 | pca_spe | 23.26 | 2.5271999835968018 | 0.8208000063896179 | 129 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-04 00:00:00 | 2022-04-04 00:00:00 | DAY | 86400.0 | pca_t2 | 20.16 | 3.734499931335449 | 1.1904000043869019 | 129 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-04 00:00:00 | 2022-04-04 00:00:00 | DAY | 86400.0 | mhal | 14.73 | 5.318600177764893 | 1.0054999589920044 | 129 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-04 00:00:00 | 2022-04-04 00:00:00 | DAY | 86400.0 | iforest | 7.75 | 2.7715001106262207 | 0.9024999737739563 | 129 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-04 00:00:00 | 2022-04-04 00:00:00 | DAY | 86400.0 | gmm | 3.1 | 3.098299980163574 | 0.7172999978065491 | 129 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-04 00:00:00 | 2022-04-04 00:00:00 | DAY | 86400.0 | omr | 12.4 | 3.6106998920440674 | 0.9761999845504761 | 129 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-04 00:00:00 | 2022-04-04 00:00:00 | DAY | 86400.0 | cusum | 11.63 | 2.6512999534606934 | 0.8240000009536743 | 129 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-05 00:00:00 | 2022-04-05 00:00:00 | DAY | 86400.0 | ar1 | 0.0 | 1.1455999612808228 | 0.4327000081539154 | 15 | D51FE222-B378-420B-8A64-854D5A0F645B |
| 2022-04-05 00:00:00 | 2022-04-05 00:00:00 | DAY | 86400.0 | pca_spe | 0.0 | 0.0 | 0.0 | 15 | D51FE222-B378-420B-8A64-854D5A0F645B |

### Bottom 10 Records

| Date | PeriodStart | PeriodType | PeriodSeconds | DetectorType | AnomalyRatePct | MaxZ | AvgZ | Points | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | ar1 | 8.51 | 4.035600185394287 | 0.7642999887466431 | 47 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | pca_spe | 8.51 | 7.460700035095215 | 0.8166999816894531 | 47 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | pca_t2 | 2.13 | 2.161099910736084 | 0.5256999731063843 | 47 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | mhal | 63.83 | 10.0 | 5.161900043487549 | 47 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | iforest | 8.51 | 2.5731000900268555 | 1.055299997329712 | 47 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | gmm | 27.66 | 4.924699783325195 | 1.4084999561309814 | 47 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | omr | 6.38 | 4.2032999992370605 | 0.683899998664856 | 47 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-14 00:00:00 | 2025-09-14 00:00:00 | DAY | 86400.0 | cusum | 0.0 | 0.7073000073432922 | 0.580299973487854 | 47 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-13 00:00:00 | 2025-09-13 00:00:00 | DAY | 86400.0 | iforest | 6.25 | 3.0081000328063965 | 0.894599974155426 | 48 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |
| 2025-09-13 00:00:00 | 2025-09-13 00:00:00 | DAY | 86400.0 | gmm | 37.5 | 8.011099815368652 | 1.77839994430542 | 48 | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 |

---


## dbo.ACM_SensorCorrelations

**Primary Key:** CorrelationID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| CorrelationID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| SensorA | nvarchar | NO | 128 | — |
| SensorB | nvarchar | NO | 128 | — |
| Correlation | float | NO | 53 | — |
| OptimalLag | int | YES | 10 | — |
| GrangerPValue | float | YES | 53 | — |
| LeadSensor | nvarchar | YES | 128 | — |
| CreatedAt | datetime2 | NO | — | (getdate()) |

---


## dbo.ACM_SensorDefects

**Primary Key:** No primary key  
**Row Count:** 720  

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
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 42 | 29.17 | 3.2948 | 1.4028 | 0.667 | 0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 |
| Correlation Break (PCA-SPE) | Correlation | HIGH | 26 | 18.06 | 2.9726 | 0.5155 | 0.0 | 0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 |
| cusum_z | cusum_z | HIGH | 26 | 18.06 | 3.3046 | 1.0171 | 0.8609 | 0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 |
| Time-Series Anomaly (AR1) | Time-Series | MEDIUM | 13 | 9.03 | 4.5867 | 0.8596 | 0.1236 | 0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 |
| Baseline Consistency (OMR) | Baseline | LOW | 6 | 4.17 | 4.1172 | 0.7753 | 1.3247 | 0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 |
| Rare State (IsolationForest) | Rare | LOW | 4 | 2.78 | 2.3156 | 0.7071 | 0.9884 | 0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 |
| Density Anomaly (GMM) | Density | LOW | 3 | 2.08 | 2.4915 | 0.6704 | 0.6143 | 0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 |
| Multivariate Distance (Mahalanobis) | Multivariate | LOW | 0 | 0.0 | 1.7818 | 0.6379 | 1.1814 | 0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 |
| Multivariate Distance (Mahalanobis) | Multivariate | CRITICAL | 381 | 43.34 | 10.0 | 2.813 | 0.8987 | 0 | DC5F36D3-753B-4944-AAF8-02532A64D28B |
| Correlation Break (PCA-SPE) | Correlation | CRITICAL | 270 | 30.72 | 10.0 | 2.5732 | 0.6427 | 0 | DC5F36D3-753B-4944-AAF8-02532A64D28B |

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
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 62 | 43.06 | 10.0 | 4.6643 | 10.0 | 1 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F |
| Multivariate Distance (Mahalanobis) | Multivariate | CRITICAL | 39 | 27.08 | 8.4408 | 1.7565 | 4.2672 | 1 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F |

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
**Row Count:** 52,565  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| 2022-04-04 02:30:00 | sensor_5_min | 1 | 6.009 | 6.009 | 24.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | reactive_power_28_min | 2 | 5.4612 | 5.4612 | 0.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:30:00 | sensor_47 | 3 | 5.3239 | -5.3239 | -375.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:40:00 | sensor_5_min | 1 | 6.009 | 6.009 | 24.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:40:00 | reactive_power_28_min | 2 | 5.4612 | 5.4612 | 0.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:40:00 | sensor_52_max | 3 | 5.218 | -5.218 | 2.200000047683716 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:50:00 | sensor_5_min | 1 | 6.009 | 6.009 | 24.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:50:00 | reactive_power_28_min | 2 | 5.4612 | 5.4612 | 0.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 02:50:00 | sensor_44 | 3 | 5.2763 | -5.2763 | -884.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 03:00:00 | reactive_power_28_min | 1 | 5.4612 | 5.4612 | 0.0 | ALERT | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |

### Bottom 10 Records

| Timestamp | SensorName | Rank | AbsZ | SignedZ | Value | Level | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-14 23:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1 | 1.4742 | 1.4742 | 48.11000061035156 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 2 | 1.3954 | 1.3954 | 48.119998931884766 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 23:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 3 | 1.3614 | 1.3614 | 1.3300000429153442 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:30:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1 | 1.421 | 1.421 | 47.400001525878906 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1 | 1.6603 | 1.6603 | 1.4900000095367432 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 2 | 1.4454 | 1.4454 | 48.93000030517578 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 22:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 3 | 1.3685 | 1.3685 | 46.70000076293945 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 21:30:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 1 | 1.3183 | 1.3183 | 46.029998779296875 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 21:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1 | 1.5108 | 1.5108 | 1.409999966621399 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-14 21:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 2 | 1.319 | 1.319 | 46.040000915527344 | WARN | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |

---


## dbo.ACM_SensorHotspots

**Primary Key:** No primary key  
**Row Count:** 1,751  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-13 00:30:00  

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
| sensor_5_min | 2022-04-04 02:30:00 | 2022-04-05 02:20:00 | 6.009 | 6.009 | 0.2253 | -0.2253 | 24.0 | -2.4000000953674316 | -1.4458333253860474 |
| reactive_power_28_min | 2022-04-04 02:30:00 | 2022-04-05 02:20:00 | 5.4612 | 5.4612 | 0.2441 | -0.2441 | 0.0 | -0.4878048896789551 | -0.4669305980205536 |
| sensor_47 | 2022-04-04 02:30:00 | 2022-04-05 02:20:00 | 5.3239 | -5.3239 | 0.2723 | 0.2723 | -375.0 | 0.0 | -18.25 |
| sensor_44 | 2022-04-04 02:30:00 | 2022-04-05 02:20:00 | 5.2825 | -5.2825 | 0.241 | 0.241 | -885.0 | 0.0 | -38.61805725097656 |
| sensor_52_max | 2022-04-04 02:30:00 | 2022-04-05 02:20:00 | 5.218 | -5.218 | 0.346 | 0.346 | 2.200000047683716 | 15.5 | 14.672917366027832 |
| reactive_power_27_max | 2022-04-04 02:30:00 | 2022-04-05 02:20:00 | 5.1846 | -5.1846 | 0.3563 | 0.3563 | 0.0 | 0.4878048896789551 | 0.45643532276153564 |
| sensor_5_avg | 2022-04-04 02:30:00 | 2022-04-05 02:20:00 | 4.3454 | 4.3454 | 0.3761 | -0.3761 | 24.0 | 0.20000000298023224 | 2.0958333015441895 |
| sensor_18_max | 2022-04-04 02:40:00 | 2022-04-05 02:20:00 | 5.1997 | -5.1997 | 0.3362 | 0.3362 | 247.60000610351562 | 1754.0999755859375 | 1662.612548828125 |
| sensor_18_avg | 2022-04-04 02:40:00 | 2022-04-05 02:20:00 | 4.95 | -4.95 | 0.3852 | 0.3852 | 209.0 | 1665.699951171875 | 1560.5284423828125 |
| sensor_52_avg | 2022-04-04 02:40:00 | 2022-04-05 02:20:00 | 4.9346 | -4.9346 | 0.4028 | 0.4028 | 1.899999976158142 | 14.800000190734863 | 13.82638931274414 |

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
**Row Count:** 354,908  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-14 23:00:00  

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
| 13632523 | 5000 | 2022-09-03 06:00:00 | sensor_44 | -0.4229404330253601 | -0.4229404330253601 | GOOD | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632522 | 5000 | 2022-09-03 05:50:00 | sensor_44 | -0.4853969216346741 | -0.4853969216346741 | GOOD | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632521 | 5000 | 2022-09-03 05:40:00 | sensor_44 | -0.517386794090271 | -0.517386794090271 | GOOD | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632520 | 5000 | 2022-09-03 05:30:00 | sensor_44 | -0.5737499594688416 | -0.5737499594688416 | GOOD | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632519 | 5000 | 2022-09-03 05:20:00 | sensor_44 | -0.4549303352832794 | -0.4549303352832794 | GOOD | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632518 | 5000 | 2022-09-03 05:10:00 | sensor_44 | -0.43817374110221863 | -0.43817374110221863 | GOOD | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632517 | 5000 | 2022-09-03 05:00:00 | sensor_44 | -0.45036035776138306 | -0.45036035776138306 | GOOD | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632516 | 5000 | 2022-09-03 04:50:00 | sensor_44 | -0.42979541420936584 | -0.42979541420936584 | GOOD | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632515 | 5000 | 2022-09-03 04:40:00 | sensor_44 | -1.1191015243530273 | -1.1191015243530273 | WARN | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |
| 13632514 | 5000 | 2022-09-03 04:30:00 | sensor_44 | -1.1533764600753784 | -1.1533764600753784 | WARN | False | 0da11bfe-83c5-4d42-b99e-da21d21c0b6c | 2025-12-13 12:32:57 |

---


## dbo.ACM_SensorRanking

**Primary Key:** No primary key  
**Row Count:** 720  

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
| gmm_z | 6 | 10.65999984741211 | 0.6142964959144592 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| ar1_z | 7 | 2.1500000953674316 | 0.12359651178121567 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| pca_spe_z | 8 | 0.0 | 0.0 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| pca_t2_z | 5 | 11.579999923706055 | 0.6669564247131348 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| omr_z | 1 | 23.0 | 1.324679970741272 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| mhal_z | 2 | 20.510000228881836 | 1.1814050674438477 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| iforest_z | 3 | 17.15999984741211 | 0.9883794188499451 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| cusum_z | 4 | 14.949999809265137 | 0.8609230518341064 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 |
| omr_z | 3 | 14.800000190734863 | 0.6879103779792786 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |
| ar1_z | 2 | 19.110000610351562 | 0.8886600136756897 | DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 |

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
| pca_t2_z | 1 | 51.81999969482422 | 10.0 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |
| mhal_z | 2 | 22.110000610351562 | 4.267228603363037 | ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 |

---


## dbo.ACM_SinceWhen

**Primary Key:** No primary key  
**Row Count:** 90  
**Date Range:** 2022-04-04 05:40:00 to 2025-12-13 12:32:56  

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
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | ALERT | 11.67 | 2022-04-19 14:40:00 | 1 |
| DC5F36D3-753B-4944-AAF8-02532A64D28B | 8632 | ALERT | 426.5 | 2024-04-02 00:20:00 | 45 |
| 7E657C21-6F0C-4537-86CF-03BC533CEE17 | 5000 | GOOD | 0.0 | 2025-12-13 12:05:04 | 0 |
| 1362B1E7-3406-4352-858B-0B8464EFA0E5 | 5000 | ALERT | 6.33 | 2022-08-31 23:40:00 | 1 |
| 5BA6BC6D-56B6-4719-BE05-0E9BEEDF38C7 | 5000 | GOOD | 0.0 | 2025-12-13 12:24:39 | 0 |
| FFAD6AD5-E725-45B5-A792-14BF8330500D | 5000 | ALERT | 10.17 | 2022-08-21 19:50:00 | 1 |
| E1596C1E-4B19-4E7A-A7EF-168EB50EC183 | 5092 | GOOD | 0.0 | 2025-12-13 11:45:10 | 0 |
| 79E7DD32-441E-43EC-8A52-18707041B651 | 8634 | GOOD | 0.0 | 2025-12-13 08:42:45 | 0 |
| C4F1062D-682C-45CA-97B3-1995F206A9A3 | 5000 | ALERT | 10.17 | 2022-08-21 19:50:00 | 1 |
| 90EDC87D-0F32-4D59-B32F-1EE7C714A926 | 1 | ALERT | 47.5 | 2023-10-18 00:30:00 | 3 |

### Bottom 10 Records

| RunID | EquipID | AlertZone | DurationHours | StartTimestamp | RecordCount |
| --- | --- | --- | --- | --- | --- |
| 17AF291B-3A84-456F-BDE8-F424528C797D | 2621 | ALERT | 314.0 | 2024-01-01 00:59:00 | 9 |
| ECC1BBDE-5D8D-4E4C-9E7F-F2898AE0571F | 5000 | ALERT | 15.67 | 2022-08-14 14:20:00 | 4 |
| F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 | GOOD | 0.0 | 2025-12-13 11:44:24 | 0 |
| CF9514D8-C683-4A5B-8DE3-EC6A9D6E6A92 | 5000 | GOOD | 0.0 | 2025-12-13 12:00:14 | 0 |
| 55DB3B5A-ED32-4DFA-970B-E41DD242B1F1 | 5000 | ALERT | 10.17 | 2022-08-21 19:50:00 | 1 |
| 202974D5-76B2-4D1E-BA3D-E321EEE5D336 | 8632 | ALERT | 417.0 | 2024-05-09 00:10:00 | 20 |
| 503A734B-F1C0-4F20-AFF2-E0A3A115C7C5 | 8634 | GOOD | 0.0 | 2025-12-13 08:38:38 | 0 |
| 682A58D9-4C3C-4B2C-8EC5-DC9852F27CC7 | 8632 | ALERT | 397.0 | 2024-01-21 01:00:00 | 28 |
| 0DA11BFE-83C5-4D42-B99E-DA21D21C0B6C | 5000 | GOOD | 0.0 | 2025-12-13 12:32:56 | 0 |
| AE60C5F3-E31D-4D61-A4D5-D9E6BFDCC217 | 5092 | ALERT | 15.5 | 2022-04-12 10:50:00 | 2 |

---


## dbo.ACM_TagEquipmentMap

**Primary Key:** TagID  
**Row Count:** 1,986  
**Date Range:** 2025-12-01 04:53:29 to 2025-12-13 10:44:18  

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
| 2036 | sensor_53_avg | Wind Farm A Turbine 92 | 5092 | sensor_53_avg for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2035 | sensor_52_std | Wind Farm A Turbine 92 | 5092 | sensor_52_std for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2034 | sensor_52_min | Wind Farm A Turbine 92 | 5092 | sensor_52_min for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2033 | sensor_52_max | Wind Farm A Turbine 92 | 5092 | sensor_52_max for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2032 | sensor_52_avg | Wind Farm A Turbine 92 | 5092 | sensor_52_avg for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2031 | sensor_51 | Wind Farm A Turbine 92 | 5092 | sensor_51 for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2030 | sensor_50 | Wind Farm A Turbine 92 | 5092 | sensor_50 for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2029 | sensor_49 | Wind Farm A Turbine 92 | 5092 | sensor_49 for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2028 | sensor_48 | Wind Farm A Turbine 92 | 5092 | sensor_48 for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |
| 2027 | sensor_47 | Wind Farm A Turbine 92 | 5092 | sensor_47 for WFA_TURBINE_92 | units | SENSOR | True | 2025-12-13 10:44:18 | 2025-12-13 10:44:18 |

---


## dbo.ACM_ThresholdCrossings

**Primary Key:** No primary key  
**Row Count:** 573  
**Date Range:** 2022-04-04 02:30:00 to 2025-09-12 07:30:00  

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
| 2022-04-04 02:30:00 | fused | 2.0 | -0.2889 | down | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 05:40:00 | fused | 2.0 | 2.1492 | up | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-04 05:50:00 | fused | 2.0 | 1.7859 | down | D51FE222-B378-420B-8A64-854D5A0F645B | 5092 |
| 2022-04-05 02:30:00 | fused | 2.0 | -0.6159 | down | C51C247D-6B86-4CF4-93C4-9547CD9E4069 | 5092 |
| 2022-04-05 02:30:00 | fused | 2.0 | -0.5271 | down | 88401CDD-1BC5-4ECB-A340-BC98624CDC94 | 5092 |
| 2022-04-05 23:30:00 | fused | 2.0 | 2.4851 | up | 88401CDD-1BC5-4ECB-A340-BC98624CDC94 | 5092 |
| 2022-04-05 23:50:00 | fused | 2.0 | 1.5645 | down | 88401CDD-1BC5-4ECB-A340-BC98624CDC94 | 5092 |
| 2022-04-06 02:30:00 | fused | 2.0 | -1.1195 | down | B938DABA-38C1-4680-AF86-3B31F85FF50F | 5092 |
| 2022-04-06 11:00:00 | fused | 2.0 | 2.1648 | up | B938DABA-38C1-4680-AF86-3B31F85FF50F | 5092 |
| 2022-04-06 11:10:00 | fused | 2.0 | 1.9106 | down | B938DABA-38C1-4680-AF86-3B31F85FF50F | 5092 |

### Bottom 10 Records

| Timestamp | DetectorType | Threshold | ZScore | Direction | RunID | EquipID |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-09-12 07:30:00 | fused | 2.0 | 0.3211 | down | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 06:30:00 | fused | 2.0 | 2.0332 | up | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 04:30:00 | fused | 2.0 | 1.784 | down | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-12 00:00:00 | fused | 2.0 | 2.3362 | up | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-11 08:00:00 | fused | 2.0 | 1.6389 | down | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-11 07:00:00 | fused | 2.0 | 2.4161 | up | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-11 04:30:00 | fused | 2.0 | 1.485 | down | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-11 03:30:00 | fused | 2.0 | 2.6186 | up | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-11 00:30:00 | fused | 2.0 | 1.7039 | down | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |
| 2025-09-11 00:00:00 | fused | 2.0 | 2.0972 | up | 284D10F7-5A0B-45D1-8707-5A1BD9790D11 | 1 |

---


## dbo.ACM_ThresholdMetadata

**Primary Key:** ThresholdID  
**Row Count:** 180  
**Date Range:** 2022-04-04 02:30:00 to 2025-07-11 00:00:00  

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
| 3948 | 1 | NULL | fused_alert_z | 3.0 | quantile_0.997 | 97 | 2023-10-18 00:00:00 | 2023-10-20 00:00:00 | 2025-12-11 15:24:02 |
| 3949 | 1 | NULL | fused_warn_z | 1.5 | quantile_0.997 | 97 | 2023-10-18 00:00:00 | 2023-10-20 00:00:00 | 2025-12-11 15:24:02 |

### Bottom 10 Records

| ThresholdID | EquipID | RegimeID | ThresholdType | ThresholdValue | CalculationMethod | SampleCount | TrainStartTime | TrainEndTime | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4169 | 5000 | NULL | fused_warn_z | 0.8695369362831116 | quantile_0.997 | 144 | 2022-09-02 06:10:00 | 2022-09-03 06:00:00 | 2025-12-13 12:32:51 |
| 4168 | 5000 | NULL | fused_alert_z | 1.7390738725662231 | quantile_0.997 | 144 | 2022-09-02 06:10:00 | 2022-09-03 06:00:00 | 2025-12-13 12:32:51 |
| 4167 | 5000 | NULL | fused_warn_z | 0.6448281407356262 | quantile_0.997 | 144 | 2022-09-01 06:10:00 | 2022-09-02 06:00:00 | 2025-12-13 12:32:02 |
| 4166 | 5000 | NULL | fused_alert_z | 1.2896562814712524 | quantile_0.997 | 144 | 2022-09-01 06:10:00 | 2022-09-02 06:00:00 | 2025-12-13 12:32:02 |
| 4165 | 5000 | NULL | fused_warn_z | 1.5 | quantile_0.997 | 67 | 2022-08-31 06:10:00 | 2022-09-01 06:00:00 | 2025-12-13 12:31:15 |
| 4164 | 5000 | NULL | fused_alert_z | 3.0 | quantile_0.997 | 67 | 2022-08-31 06:10:00 | 2022-09-01 06:00:00 | 2025-12-13 12:31:15 |
| 4163 | 5000 | NULL | fused_warn_z | 1.5 | quantile_0.997 | 72 | 2022-08-30 06:10:00 | 2022-08-31 06:00:00 | 2025-12-13 12:30:10 |
| 4162 | 5000 | NULL | fused_alert_z | 3.0 | quantile_0.997 | 72 | 2022-08-30 06:10:00 | 2022-08-31 06:00:00 | 2025-12-13 12:30:10 |
| 4161 | 5000 | NULL | fused_warn_z | 0.7906934022903442 | quantile_0.997 | 144 | 2022-08-29 06:10:00 | 2022-08-30 06:00:00 | 2025-12-13 12:29:22 |
| 4160 | 5000 | NULL | fused_alert_z | 1.5813868045806885 | quantile_0.997 | 144 | 2022-08-29 06:10:00 | 2022-08-30 06:00:00 | 2025-12-13 12:29:22 |

---


## dbo.ELECTRIC_MOTOR_Data

**Primary Key:** No primary key  
**Row Count:** 17,477  
**Date Range:** 2024-01-01 00:00:00 to 2024-12-01 23:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| u_q | float | YES | 53 | — |
| coolant | float | YES | 53 | — |
| stator_winding | float | YES | 53 | — |
| u_d | float | YES | 53 | — |
| stator_tooth | float | YES | 53 | — |
| motor_speed | float | YES | 53 | — |
| i_d | float | YES | 53 | — |
| i_q | float | YES | 53 | — |
| pm | float | YES | 53 | — |
| stator_yoke | float | YES | 53 | — |
| ambient | float | YES | 53 | — |
| torque | float | YES | 53 | — |
| profile_id | bigint | YES | 19 | — |

### Top 10 Records

| EntryDateTime | u_q | coolant | stator_winding | u_d | stator_tooth | motor_speed | i_d | i_q | pm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01-01 00:00:00 | 48.378386286166666 | 18.900812435 | 19.361286926000002 | -43.17679374463333 | 18.451467990333338 | 2311.6027119875334 | -63.28096596180001 | 29.90897307461666 | 24.585809103833338 |
| 2024-01-01 00:01:00 | 91.34201901700003 | 18.897907416333336 | 25.64166495033333 | -92.01435012816667 | 21.45641950016667 | 4999.943725566666 | -142.82388814666666 | 52.83300215416664 | 26.36999613433333 |
| 2024-01-01 00:02:00 | 91.1145528155 | 18.91490360850001 | 32.93693752266667 | -92.23500722199998 | 26.474174785333332 | 4999.956746416667 | -142.35288645666665 | 52.887767536833344 | 28.736558691500004 |
| 2024-01-01 00:03:00 | 90.86030057283334 | 18.994687843666668 | 38.51075560316666 | -92.53561503099999 | 30.74385830500001 | 4999.9565837499995 | -141.90893859666667 | 52.92798016816666 | 30.93217407866666 |
| 2024-01-01 00:04:00 | 90.70566151933333 | 19.10296223966667 | 43.49067993133335 | -92.70875523816667 | 34.10806763916666 | 4999.956079150001 | -141.6960528066667 | 52.94287637083333 | 33.0365564985 |
| 2024-01-01 00:05:00 | 90.47746200599998 | 19.161996460333334 | 47.097871653333314 | -92.94932467099999 | 37.19874865233333 | 4999.955940799998 | -141.4490150466667 | 52.98018754316669 | 35.018084462333334 |
| 2024-01-01 00:06:00 | 90.30193456049997 | 19.209312978666667 | 51.04982903816666 | -93.19797770199999 | 39.2428047815 | 4999.955851216666 | -141.24095255833333 | 53.027876091166675 | 36.90033461233334 |
| 2024-01-01 00:07:00 | 90.17929255133335 | 19.242657598 | 53.6280256903333 | -93.34241714483332 | 41.46424528800001 | 4999.956014016667 | -141.08919804833334 | 53.058543268833326 | 38.65695915233334 |
| 2024-01-01 00:08:00 | 89.95980161050001 | 19.236234633000006 | 56.21492614716668 | -93.55297711766667 | 43.65091775233334 | 4999.957389366666 | -140.8591430666667 | 53.06756922349999 | 40.33262329116666 |
| 2024-01-01 00:09:00 | 89.84458618133333 | 19.1955741885 | 58.23025868683329 | -93.72211799616666 | 45.23047421716665 | 4999.956494133333 | -140.74482015666666 | 53.09469763416668 | 41.95414447783334 |

### Bottom 10 Records

| EntryDateTime | u_q | coolant | stator_winding | u_d | stator_tooth | motor_speed | i_d | i_q | pm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-12-01 23:59:00 | 1.1721619141333333 | 28.186712892666662 | 28.08523399083333 | 0.3652682746333333 | 27.701225878499997 | 0.0009566072166666668 | -2.0005455226499995 | 1.0970678800000002 | 35.299491783666674 |
| 2024-12-01 23:58:00 | 1.164350336366667 | 28.172085320333327 | 28.104769323166668 | 0.3524907381000001 | 27.70104749266667 | 9.413821666666666e-05 | -2.00056689405 | 1.0972193864166664 | 35.37197066249999 |
| 2024-12-01 23:57:00 | 1.1713139229833331 | 28.159938034 | 28.107086791333337 | 0.35682004373333337 | 27.663690461499996 | 0.0001793159333333334 | -2.0006962553166665 | 1.0970444282833336 | 35.4505278965 |
| 2024-12-01 23:56:00 | 1.1680835715000002 | 28.18664423383333 | 28.102322523166663 | 0.3647678287666667 | 27.669372358500002 | 0.0001553014166666668 | -2.0003284368000003 | 1.0971704809333336 | 35.52268177966666 |
| 2024-12-01 23:55:00 | 1.1703049006166668 | 28.178678517333335 | 28.090460950500002 | 0.3640781494499999 | 27.7156685265 | -1.569796666666673e-05 | -2.0004054199166665 | 1.0969923302666666 | 35.59825376333333 |
| 2024-12-01 23:54:00 | 1.1704963194 | 28.172214498000002 | 28.131948800166672 | 0.3631390080166667 | 27.72754888733333 | 0.0001639094333333333 | -2.0005341474166665 | 1.0970136549166665 | 35.67296063933335 |
| 2024-12-01 23:53:00 | 1.1699569248666668 | 28.177647645999993 | 28.099683289 | 0.35316880525000005 | 27.693439247499995 | -0.00018186781666666666 | -2.0005412548166666 | 1.0968460292666664 | 35.756014847833335 |
| 2024-12-01 23:52:00 | 1.1608535779 | 28.170571701499988 | 28.136193458166677 | 0.3571568778166668 | 27.540225634500008 | 0.0008304430666666668 | -2.0004174596500004 | 1.0972237057333334 | 35.83020165100001 |
| 2024-12-01 23:51:00 | 1.1752969764499999 | 28.176263269 | 28.14836788033332 | 0.3677545190166666 | 27.682216454166667 | -0.0007401398500000002 | -2.0005444712 | 1.0970086890166664 | 35.913733745333325 |
| 2024-12-01 23:50:00 | 1.1749393239333332 | 28.176702847999998 | 28.24717568383334 | 0.36633154763333325 | 27.717695598166657 | -0.00020579585000000004 | -2.0004473565333334 | 1.0970381099000002 | 35.99464184800001 |

---


## dbo.ELECTRIC_MOTOR_Data_RAW

**Primary Key:** No primary key  
**Row Count:** 1,048,575  
**Date Range:** 2024-01-01 00:00:00 to 2024-12-01 23:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| u_q | float | YES | 53 | — |
| coolant | float | YES | 53 | — |
| stator_winding | float | YES | 53 | — |
| u_d | float | YES | 53 | — |
| stator_tooth | float | YES | 53 | — |
| motor_speed | float | YES | 53 | — |
| i_d | float | YES | 53 | — |
| i_q | float | YES | 53 | — |
| pm | float | YES | 53 | — |
| stator_yoke | float | YES | 53 | — |
| ambient | float | YES | 53 | — |
| torque | float | YES | 53 | — |
| profile_id | bigint | YES | 19 | — |

### Top 10 Records

| EntryDateTime | u_q | coolant | stator_winding | u_d | stator_tooth | motor_speed | i_d | i_q | pm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-01-01 00:00:00 | -0.450681508 | 18.80517197 | 19.08666992 | -0.350054592 | 18.29321861 | 0.002865568 | 0.004419137 | 0.000328102 | 24.55421448 |
| 2024-01-01 00:00:00 | -0.325737 | 18.81857109 | 19.09239006 | -0.305803001 | 18.29480743 | 0.000256782 | 0.000605872 | -0.000785353 | 24.53807831 |
| 2024-01-01 00:00:00 | -0.440864027 | 18.82876968 | 19.08938026 | -0.372502625 | 18.29409409 | 0.002354971 | 0.001289587 | 0.000386468 | 24.54469299 |
| 2024-01-01 00:00:00 | -0.327025682 | 18.83556747 | 19.0830307 | -0.316198707 | 18.2925415 | 0.006104666 | 2.56e-05 | 0.002045661 | 24.55401802 |
| 2024-01-01 00:00:00 | -0.47115013 | 18.85703278 | 19.08252525 | -0.332272142 | 18.29142761 | 0.003132823 | -0.064316779 | 0.037183776 | 24.56539726 |
| 2024-01-01 00:00:00 | -0.538972616 | 18.90154839 | 19.07710838 | 0.009147473 | 18.29062843 | 0.009636124 | -0.613635242 | 0.336747348 | 24.57360077 |
| 2024-01-01 00:00:00 | -0.653148472 | 18.94171143 | 19.07458305 | 0.238889694 | 18.29252434 | 0.001337012 | -1.005647302 | 0.554211259 | 24.57657814 |
| 2024-01-01 00:00:00 | -0.758391559 | 18.96086121 | 19.08249855 | 0.395099252 | 18.29404068 | 0.001421958 | -1.288383722 | 0.706369996 | 24.57494926 |
| 2024-01-01 00:00:00 | -0.727128446 | 18.97354507 | 19.08553314 | 0.546622515 | 18.29196358 | 0.000576553 | -1.490530491 | 0.81733948 | 24.56707954 |
| 2024-01-01 00:00:00 | -0.874307454 | 18.98781204 | 19.07602501 | 0.578943968 | 18.28723335 | -0.00124788 | -1.634463549 | 0.898012877 | 24.55324173 |

### Bottom 10 Records

| EntryDateTime | u_q | coolant | stator_winding | u_d | stator_tooth | motor_speed | i_d | i_q | pm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-12-01 23:59:00 | 1.137114022 | 28.13785325 | 28.09149068 | 0.347978841 | 27.70218009 | 0.007293692 | -2.001524275 | 1.097026618 | 35.25571142 |
| 2024-12-01 23:59:00 | 1.212985191 | 28.15191758 | 28.08352885 | 0.325080895 | 27.69210776 | -0.00248922 | -2.000248251 | 1.096491804 | 35.2537029 |
| 2024-12-01 23:59:00 | 1.131010044 | 28.1875456 | 28.09089738 | 0.380257415 | 27.67168962 | -0.006319263 | -2.001364778 | 1.097398743 | 35.25990047 |
| 2024-12-01 23:59:00 | 1.21948531 | 28.22228993 | 28.08972825 | 0.340979721 | 27.64104971 | -0.001454695 | -2.00143121 | 1.096344035 | 35.26434707 |
| 2024-12-01 23:59:00 | 1.137752435 | 28.22954785 | 28.11811227 | 0.367792107 | 27.60556824 | 0.002287588 | -2.001507436 | 1.097547149 | 35.27352702 |
| 2024-12-01 23:59:00 | 1.23980985 | 28.24267235 | 28.1445875 | 0.297356804 | 27.64451354 | 0.008019502 | -2.001208195 | 1.097788805 | 35.28422604 |
| 2024-12-01 23:59:00 | 1.128303649 | 28.25044137 | 28.12256643 | 0.357707544 | 27.64212944 | 0.002806887 | -1.999326862 | 1.098612936 | 35.27906876 |
| 2024-12-01 23:59:00 | 1.239920079 | 28.25659208 | 28.11138549 | 0.293788543 | 27.62084437 | 0.003575936 | -2.001055169 | 1.097157751 | 35.27650155 |
| 2024-12-01 23:59:00 | 1.136833857 | 28.24703663 | 28.11280315 | 0.369323379 | 27.69073599 | 0.005486384 | -2.001329434 | 1.09705893 | 35.28160367 |
| 2024-12-01 23:59:00 | 1.232331681 | 28.23297177 | 28.10448049 | 0.310092131 | 27.71307039 | 0.005195291 | -2.000172126 | 1.097474372 | 35.2798471 |

---


## dbo.Equipment

**Primary Key:** EquipID  
**Row Count:** 29  
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
| 5000 | WFA_TURBINE_0 | Wind Farm A Turbine 0 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5003 | WFA_TURBINE_3 | Wind Farm A Turbine 3 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5010 | WFA_TURBINE_10 | Wind Farm A Turbine 10 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5011 | WFA_TURBINE_11 | Wind Farm A Turbine 11 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5013 | WFA_TURBINE_13 | Wind Farm A Turbine 13 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5014 | WFA_TURBINE_14 | Wind Farm A Turbine 14 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5017 | WFA_TURBINE_17 | Wind Farm A Turbine 17 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |

### Bottom 10 Records

| EquipID | EquipCode | EquipName | Area | Unit | Status | CommissionDate | CreatedAtUTC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8634 | ELECTRIC_MOTOR | Electric Motor | NULL | NULL | 1 | NULL | 2025-12-11 12:57:37 |
| 8632 | WIND_TURBINE | Wind Turbine SCADA | Renewable Energy | Wind Farm | 1 | NULL | 2025-12-11 11:34:08 |
| 5092 | WFA_TURBINE_92 | Wind Farm A Turbine 92 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5084 | WFA_TURBINE_84 | Wind Farm A Turbine 84 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5073 | WFA_TURBINE_73 | Wind Farm A Turbine 73 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5072 | WFA_TURBINE_72 | Wind Farm A Turbine 72 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5071 | WFA_TURBINE_71 | Wind Farm A Turbine 71 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5069 | WFA_TURBINE_69 | Wind Farm A Turbine 69 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5068 | WFA_TURBINE_68 | Wind Farm A Turbine 68 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |
| 5051 | WFA_TURBINE_51 | Wind Farm A Turbine 51 | Wind Farm A | Turbine | 1 | 2025-01-01 00:00:00 | 2025-12-13 04:26:58 |

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
**Row Count:** 630  
**Date Range:** 2025-12-05 06:06:56 to 2025-12-13 07:02:50  

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
| ar1_params | 1 | 1 | 2025-12-11 09:54:02 | {"n_sensors": 90, "mean_autocorr": 2084.939, "mean_residual_std": 1390.7155, "params_count": 180} | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 7167 bytes> |
| feature_medians | 1 | 1 | 2025-12-11 09:54:02 | NULL | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 12288 bytes> |
| gmm_model | 1 | 1 | 2025-12-11 09:54:02 | {"n_components": 3, "covariance_type": "diag", "bic": 9043485744053.15, "aic": 9043485742657.66, ... | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 9718 bytes> |
| iforest_model | 1 | 1 | 2025-12-11 09:54:02 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 693657 bytes> |
| mhal_params | 1 | 1 | 2025-12-11 09:54:02 | NULL | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 65826 bytes> |
| omr_model | 1 | 1 | 2025-12-11 09:54:02 | NULL | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 115328 bytes> |
| pca_model | 1 | 1 | 2025-12-11 09:54:02 | {"n_components": 5, "variance_ratio_sum": 0.7681, "variance_ratio_first_component": 0.2799, "vari... | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 5375 bytes> |
| ar1_params | 1 | 2 | 2025-12-11 09:54:16 | {"n_sensors": 90, "mean_autocorr": 1372.1481, "mean_residual_std": 1086.2972, "params_count": 180} | {"train_rows": 503, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 7167 bytes> |
| feature_medians | 1 | 2 | 2025-12-11 09:54:16 | NULL | {"train_rows": 503, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 12288 bytes> |
| gmm_model | 1 | 2 | 2025-12-11 09:54:16 | {"n_components": 3, "covariance_type": "diag", "bic": 10927079700552.87, "aic": 10927079698265.31... | {"train_rows": 503, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 9870 bytes> |

### Bottom 10 Records

| ModelType | EquipID | Version | EntryDateTime | ParamsJSON | StatsJSON | RunID | ModelBytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pca_model | 8634 | 7 | 2025-12-13 03:22:18 | {"n_components": 5, "variance_ratio_sum": 0.567, "variance_ratio_first_component": 0.2334, "varia... | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 7295 bytes> |
| omr_model | 8634 | 7 | 2025-12-13 03:22:19 | NULL | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 295637 bytes> |
| mhal_params | 8634 | 7 | 2025-12-13 03:22:19 | NULL | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 136546 bytes> |
| iforest_model | 8634 | 7 | 2025-12-13 03:22:19 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 1623081 bytes> |
| gmm_model | 8634 | 7 | 2025-12-13 03:22:19 | {"n_components": 3, "covariance_type": "diag", "bic": 2.3871463216416103e+27, "aic": 2.3871463216... | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 13634 bytes> |
| feature_medians | 8634 | 7 | 2025-12-13 03:22:19 | NULL | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 7920 bytes> |
| ar1_params | 8634 | 7 | 2025-12-13 03:22:18 | {"n_sensors": 130, "mean_autocorr": -313111011.0363, "mean_residual_std": 22268736505.7601, "para... | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 6092 bytes> |
| pca_model | 8634 | 6 | 2025-12-13 03:19:08 | {"n_components": 5, "variance_ratio_sum": 0.5524, "variance_ratio_first_component": 0.229, "varia... | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 7295 bytes> |
| omr_model | 8634 | 6 | 2025-12-13 03:19:08 | NULL | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 295637 bytes> |
| mhal_params | 8634 | 6 | 2025-12-13 03:19:08 | NULL | {"train_rows": 720, "train_sensors": ["ambient_med", "coolant_med", "i_d_med", "i_q_med", "motor_... | NULL | <binary 136546 bytes> |

---


## dbo.WFA_TURBINE_0_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,986  
**Date Range:** 2022-08-04 06:10:00 to 2023-08-24 06:10:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-08-04 06:10:00 | 0 | 0 | train | 0 | 22.0 | 302.9 | 129.4 | 1.7000000000000002 | 1.7000000000000002 |
| 2022-08-04 06:20:00 | 0 | 1 | train | 0 | 22.0 | 307.1 | 133.6 | 1.7000000000000002 | 1.7000000000000002 |
| 2022-08-04 06:30:00 | 0 | 2 | train | 0 | 22.0 | 340.6 | 167.1 | 0.9 | 0.9 |
| 2022-08-04 06:40:00 | 0 | 3 | train | 0 | 22.0 | 124.4 | -49.1 | 1.5 | 1.5 |
| 2022-08-04 06:50:00 | 0 | 4 | train | 0 | 22.0 | 66.2 | -107.3 | 1.0 | 1.0 |
| 2022-08-04 07:00:00 | 0 | 5 | train | 0 | 22.0 | 92.0 | -81.4 | 1.1 | 1.1 |
| 2022-08-04 07:10:00 | 0 | 6 | train | 0 | 22.0 | 286.9 | 113.4 | 0.7000000000000001 | 0.7000000000000001 |
| 2022-08-04 07:20:00 | 0 | 7 | train | 0 | 22.0 | 154.4 | -19.1 | 1.5 | 1.5 |
| 2022-08-04 07:30:00 | 0 | 8 | train | 0 | 22.0 | 128.7 | -44.8 | 1.7000000000000002 | 1.7000000000000002 |
| 2022-08-04 07:40:00 | 0 | 9 | train | 0 | 22.0 | 126.6 | -46.9 | 1.8 | 1.8 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-08-24 06:10:00 | 0 | 54985 | prediction | 3 | 25.0 | 101.8 | 5.4 | 3.4 | 0.0 |
| 2023-08-24 06:00:00 | 0 | 54984 | prediction | 3 | 25.0 | 104.4 | 8.0 | 3.1 | 0.0 |
| 2023-08-24 05:50:00 | 0 | 54983 | prediction | 3 | 25.0 | 114.5 | 18.2 | 2.7 | 0.0 |
| 2023-08-24 05:40:00 | 0 | 54982 | prediction | 3 | 25.0 | 117.2 | 20.9 | 2.8 | 0.0 |
| 2023-08-24 05:30:00 | 0 | 54981 | prediction | 3 | 25.0 | 121.0 | 24.6 | 3.3 | 0.0 |
| 2023-08-24 05:20:00 | 0 | 54980 | prediction | 3 | 25.0 | 120.9 | 24.6 | 4.4 | 0.0 |
| 2023-08-24 05:10:00 | 0 | 54979 | prediction | 3 | 25.0 | 117.8 | 21.4 | 5.2 | 0.0 |
| 2023-08-24 05:00:00 | 0 | 54978 | prediction | 3 | 25.0 | 110.8 | 14.5 | 6.0 | 0.0 |
| 2023-08-24 04:50:00 | 0 | 54977 | prediction | 3 | 25.0 | 106.0 | 9.6 | 6.3 | 0.0 |
| 2023-08-24 04:40:00 | 0 | 54976 | prediction | 3 | 25.0 | 101.0 | 4.6 | 5.8 | 0.0 |

---


## dbo.WFA_TURBINE_10_Data

**Primary Key:** EntryDateTime  
**Row Count:** 53,592  
**Date Range:** 2022-10-09 08:40:00 to 2023-10-18 08:40:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-10-09 08:40:00 | 10 | 0 | train | 4 | 20.0 | 265.7 | 19.4 | 2.6 | 2.6 |
| 2022-10-09 08:50:00 | 10 | 1 | train | 4 | 20.0 | 244.9 | -11.8 | 2.6 | 2.6 |
| 2022-10-09 09:00:00 | 10 | 2 | train | 4 | 20.0 | 299.5 | 42.7 | 2.5 | 2.5 |
| 2022-10-09 09:10:00 | 10 | 3 | train | 4 | 20.0 | 280.2 | 23.5 | 2.5 | 2.5 |
| 2022-10-09 09:20:00 | 10 | 4 | train | 4 | 20.0 | 281.1 | 24.3 | 2.7 | 2.7 |
| 2022-10-09 09:30:00 | 10 | 5 | train | 4 | 20.0 | 251.5 | -5.2 | 3.0 | 3.0 |
| 2022-10-09 09:40:00 | 10 | 6 | train | 4 | 20.0 | 246.2 | -25.6 | 2.9 | 2.9 |
| 2022-10-09 09:50:00 | 10 | 7 | train | 4 | 21.0 | 294.0 | 9.2 | 2.5 | 2.5 |
| 2022-10-09 10:00:00 | 10 | 8 | train | 4 | 20.0 | 301.6 | 16.8 | 2.4 | 2.4 |
| 2022-10-09 10:10:00 | 10 | 9 | train | 4 | 21.0 | 285.7 | 0.8 | 2.2 | 2.2 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-18 08:40:00 | 10 | 53591 | prediction | 3 | 19.0 | 272.4 | 6.0 | 3.5 | 3.5 |
| 2023-10-18 08:30:00 | 10 | 53590 | prediction | 4 | 18.0 | 279.1 | -6.3 | 7.6 | 6.9 |
| 2023-10-18 08:20:00 | 10 | 53589 | prediction | 4 | 19.0 | 285.9 | 5.9 | 9.2 | 9.0 |
| 2023-10-18 08:10:00 | 10 | 53588 | prediction | 4 | 19.0 | 260.8 | 2.8 | 9.7 | 9.6 |
| 2023-10-18 08:00:00 | 10 | 53587 | prediction | 4 | 20.0 | 261.6 | -3.8 | 7.9 | 7.7 |
| 2023-10-18 07:50:00 | 10 | 53586 | prediction | 4 | 21.0 | 269.2 | -1.9 | 7.8 | 7.7 |
| 2023-10-18 07:40:00 | 10 | 53585 | prediction | 4 | 21.0 | 271.1 | 0.0 | 7.4 | 7.4 |
| 2023-10-18 07:30:00 | 10 | 53584 | prediction | 4 | 21.0 | 268.7 | 0.0 | 8.1 | 8.1 |
| 2023-10-18 07:20:00 | 10 | 53583 | prediction | 4 | 21.0 | 263.5 | 0.4 | 7.0 | 7.2 |
| 2023-10-18 07:10:00 | 10 | 53582 | prediction | 4 | 21.0 | 262.6 | 2.3 | 6.2 | 6.2 |

---


## dbo.WFA_TURBINE_11_Data

**Primary Key:** EntryDateTime  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

---


## dbo.WFA_TURBINE_13_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,010  
**Date Range:** 2022-04-30 13:20:00 to 2023-05-25 10:20:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-04-30 13:20:00 | 21 | 0 | train | 0 | 22.0 | 57.4 | -75.5 | 2.5 | 2.5 |
| 2022-04-30 13:30:00 | 21 | 1 | train | 0 | 23.0 | 93.0 | 31.4 | 2.2 | 2.2 |
| 2022-04-30 13:40:00 | 21 | 2 | train | 0 | 24.0 | 119.9 | 58.3 | 1.7000000000000002 | 1.7000000000000002 |
| 2022-04-30 13:50:00 | 21 | 3 | train | 0 | 24.0 | 65.2 | 3.6 | 1.2 | 1.2 |
| 2022-04-30 14:00:00 | 21 | 4 | train | 0 | 25.0 | 53.7 | -7.8 | 1.5 | 1.5 |
| 2022-04-30 14:10:00 | 21 | 5 | train | 0 | 22.0 | 36.5 | -24.8 | 5.7 | 5.6 |
| 2022-04-30 14:20:00 | 21 | 6 | train | 0 | 22.0 | 274.8 | -11.8 | 5.0 | 5.3 |
| 2022-04-30 14:30:00 | 21 | 7 | train | 0 | 22.0 | 247.2 | -17.3 | 5.4 | 5.5 |
| 2022-04-30 14:40:00 | 21 | 8 | train | 0 | 22.0 | 264.5 | -3.1 | 5.6 | 5.8 |
| 2022-04-30 14:50:00 | 21 | 9 | train | 0 | 22.0 | 249.5 | -20.2 | 4.8 | 4.9 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-05-25 10:20:00 | 21 | 54009 | prediction | 0 | 23.0 | 87.5 | -14.3 | 5.3 | 5.5 |
| 2023-05-25 10:10:00 | 21 | 54008 | prediction | 0 | 23.0 | 100.2 | -8.4 | 6.0 | 6.2 |
| 2023-05-25 10:00:00 | 21 | 54007 | prediction | 0 | 23.0 | 122.3 | 13.7 | 6.1 | 6.3 |
| 2023-05-25 09:50:00 | 21 | 54006 | prediction | 0 | 23.0 | 94.2 | -5.6 | 5.9 | 6.1 |
| 2023-05-25 09:40:00 | 21 | 54005 | prediction | 0 | 22.0 | 104.9 | 5.1 | 5.5 | 5.7 |
| 2023-05-25 09:30:00 | 21 | 54004 | prediction | 0 | 22.0 | 94.9 | 2.5 | 5.6 | 5.8 |
| 2023-05-25 09:20:00 | 21 | 54003 | prediction | 0 | 22.0 | 91.6 | -8.5 | 6.3 | 6.4 |
| 2023-05-25 09:10:00 | 21 | 54002 | prediction | 0 | 22.0 | 100.0 | -6.4 | 7.5 | 7.5 |
| 2023-05-25 09:00:00 | 21 | 54001 | prediction | 0 | 22.0 | 92.2 | 2.9 | 6.5 | 6.6 |
| 2023-05-25 08:50:00 | 21 | 54000 | prediction | 0 | 21.0 | 91.0 | 1.3 | 6.5 | 6.7 |

---


## dbo.WFA_TURBINE_14_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,197  
**Date Range:** 2022-03-03 14:00:00 to 2023-03-16 18:40:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-03-03 14:00:00 | 13 | 0 | train | 4 | 14.0 | 273.7 | -46.4 | 6.0 | 6.0 |
| 2022-03-03 14:10:00 | 13 | 1 | train | 4 | 14.0 | 259.5 | -60.6 | 6.0 | 0.0 |
| 2022-03-03 14:40:00 | 13 | 2 | train | 4 | 14.0 | 251.8 | -68.3 | 5.5 | 0.3 |
| 2022-03-03 14:50:00 | 13 | 3 | train | 4 | 14.0 | 263.3 | -56.7 | 5.4 | 5.4 |
| 2022-03-03 15:00:00 | 13 | 4 | train | 4 | 14.0 | 271.5 | 24.2 | 4.6 | 4.6 |
| 2022-03-03 15:10:00 | 13 | 5 | train | 4 | 14.0 | 230.1 | -17.1 | 4.7 | 4.7 |
| 2022-03-03 15:20:00 | 13 | 6 | train | 4 | 14.0 | 229.0 | -18.2 | 5.1 | 5.1 |
| 2022-03-03 15:30:00 | 13 | 7 | train | 4 | 14.0 | 268.9 | 21.7 | 4.3 | 4.3 |
| 2022-03-03 15:40:00 | 13 | 8 | train | 4 | 15.0 | 281.4 | 34.1 | 3.5 | 3.5 |
| 2022-03-03 15:50:00 | 13 | 9 | train | 4 | 14.0 | 245.5 | -1.7000000000000002 | 3.7 | 3.7 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-03-16 18:40:00 | 13 | 54196 | prediction | 0 | 16.0 | 108.0 | -1.8 | 10.0 | 10.3 |
| 2023-03-16 18:30:00 | 13 | 54195 | prediction | 0 | 16.0 | 111.7 | 1.8 | 11.1 | 11.2 |
| 2023-03-16 18:20:00 | 13 | 54194 | prediction | 0 | 16.0 | 105.6 | -4.3 | 9.6 | 9.9 |
| 2023-03-16 18:10:00 | 13 | 54193 | prediction | 0 | 16.0 | 130.7 | 14.5 | 9.3 | 9.5 |
| 2023-03-16 18:00:00 | 13 | 54192 | prediction | 0 | 16.0 | 118.1 | 3.7 | 8.9 | 9.1 |
| 2023-03-16 17:50:00 | 13 | 54191 | prediction | 0 | 16.0 | 104.1 | -5.4 | 8.5 | 8.7 |
| 2023-03-16 17:40:00 | 13 | 54190 | prediction | 0 | 16.0 | 129.1 | 19.6 | 8.5 | 8.7 |
| 2023-03-16 17:30:00 | 13 | 54189 | prediction | 0 | 16.0 | 95.3 | -14.1 | 8.9 | 9.2 |
| 2023-03-16 17:20:00 | 13 | 54188 | prediction | 0 | 16.0 | 122.1 | 13.3 | 9.9 | 10.1 |
| 2023-03-16 17:10:00 | 13 | 54187 | prediction | 0 | 16.0 | 76.0 | -32.7 | 8.5 | 8.6 |

---


## dbo.WFA_TURBINE_17_Data

**Primary Key:** EntryDateTime  
**Row Count:** 55,090  
**Date Range:** 2022-10-31 15:20:00 to 2023-11-20 00:40:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-10-31 15:20:00 | 10 | 0 | train | 0 | 23.0 | 91.8 | -23.0 | 11.9 | 11.0 |
| 2022-10-31 15:30:00 | 10 | 1 | train | 0 | 23.0 | 113.6 | 6.8 | 11.3 | 10.7 |
| 2022-10-31 15:40:00 | 10 | 2 | train | 0 | 23.0 | 120.8 | 14.0 | 12.0 | 11.5 |
| 2022-10-31 15:50:00 | 10 | 3 | train | 0 | 23.0 | 103.1 | -3.6 | 11.1 | 10.6 |
| 2022-10-31 16:00:00 | 10 | 4 | train | 0 | 23.0 | 106.3 | -0.4 | 11.1 | 10.6 |
| 2022-10-31 16:10:00 | 10 | 5 | train | 0 | 23.0 | 111.5 | 4.8 | 10.7 | 10.1 |
| 2022-10-31 16:20:00 | 10 | 6 | train | 0 | 23.0 | 89.7 | -14.5 | 11.3 | 10.7 |
| 2022-10-31 16:30:00 | 10 | 7 | train | 0 | 23.0 | 99.3 | -12.7 | 12.2 | 11.4 |
| 2022-10-31 16:40:00 | 10 | 8 | train | 0 | 23.0 | 100.0 | -12.0 | 9.6 | 9.3 |
| 2022-10-31 16:50:00 | 10 | 9 | train | 0 | 23.0 | 102.2 | -9.8 | 9.6 | 9.2 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-11-20 00:40:00 | 10 | 55089 | prediction | 0 | 17.0 | 98.2 | -3.0 | 9.5 | 9.1 |
| 2023-11-20 00:30:00 | 10 | 55088 | prediction | 0 | 17.0 | 100.2 | 4.1 | 10.1 | 9.7 |
| 2023-11-20 00:20:00 | 10 | 55087 | prediction | 0 | 17.0 | 97.6 | 4.5 | 9.3 | 9.4 |
| 2023-11-20 00:10:00 | 10 | 55086 | prediction | 0 | 17.0 | 97.4 | 4.3 | 10.1 | 10.0 |
| 2023-11-20 00:00:00 | 10 | 55085 | prediction | 0 | 17.0 | 93.4 | 0.4 | 10.5 | 10.3 |
| 2023-11-19 23:50:00 | 10 | 55084 | prediction | 0 | 17.0 | 93.3 | 0.2 | 10.8 | 10.9 |
| 2023-11-19 23:40:00 | 10 | 55083 | prediction | 0 | 17.0 | 96.1 | 3.0 | 11.1 | 11.0 |
| 2023-11-19 23:30:00 | 10 | 55082 | prediction | 0 | 17.0 | 90.9 | -2.1 | 10.9 | 10.8 |
| 2023-11-19 23:20:00 | 10 | 55081 | prediction | 0 | 17.0 | 90.8 | -2.0 | 10.6 | 10.4 |
| 2023-11-19 23:10:00 | 10 | 55080 | prediction | 0 | 17.0 | 88.6 | -4.1 | 10.4 | 10.1 |

---


## dbo.WFA_TURBINE_21_Data

**Primary Key:** EntryDateTime  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

---


## dbo.WFA_TURBINE_22_Data

**Primary Key:** EntryDateTime  
**Row Count:** 53,036  
**Date Range:** 2022-08-12 09:50:00 to 2023-08-20 09:50:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-08-12 09:50:00 | 21 | 0 | train | 0 | 25.0 | 105.9 | 7.5 | 18.4 | 17.2 |
| 2022-08-12 10:00:00 | 21 | 1 | train | 0 | 25.0 | 107.9 | 9.5 | 18.6 | 17.5 |
| 2022-08-12 10:10:00 | 21 | 2 | train | 0 | 25.0 | 124.8 | 26.4 | 18.3 | 17.2 |
| 2022-08-12 10:20:00 | 21 | 3 | train | 0 | 25.0 | 114.1 | 15.8 | 18.8 | 17.6 |
| 2022-08-12 10:30:00 | 21 | 4 | train | 0 | 26.0 | 100.9 | 2.5 | 18.7 | 17.5 |
| 2022-08-12 10:40:00 | 21 | 5 | train | 0 | 26.0 | 85.8 | -12.9 | 17.6 | 16.5 |
| 2022-08-12 10:50:00 | 21 | 6 | train | 0 | 26.0 | 91.9 | -6.5 | 18.3 | 17.1 |
| 2022-08-12 11:00:00 | 21 | 7 | train | 0 | 26.0 | 99.5 | -5.9 | 18.7 | 17.8 |
| 2022-08-12 11:10:00 | 21 | 8 | train | 0 | 26.0 | 103.2 | -1.5 | 19.3 | 18.1 |
| 2022-08-12 11:20:00 | 21 | 9 | train | 0 | 26.0 | 109.4 | 12.4 | 18.0 | 17.0 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-08-20 09:50:00 | 21 | 53035 | prediction | 3 | 27.0 | 89.2 | -1.1 | 10.7 | 10.4 |
| 2023-08-20 09:40:00 | 21 | 53034 | prediction | 3 | 27.0 | 86.5 | -3.8 | 10.6 | 10.4 |
| 2023-08-20 09:30:00 | 21 | 53033 | prediction | 3 | 26.0 | 87.3 | -3.0 | 11.8 | 11.4 |
| 2023-08-20 09:20:00 | 21 | 53032 | prediction | 3 | 26.0 | 89.0 | -1.3 | 10.7 | 10.4 |
| 2023-08-20 09:10:00 | 21 | 53031 | prediction | 3 | 26.0 | 90.4 | 0.1 | 10.4 | 10.1 |
| 2023-08-20 09:00:00 | 21 | 53030 | prediction | 3 | 26.0 | 89.2 | -1.1 | 11.3 | 10.9 |
| 2023-08-20 08:50:00 | 21 | 53029 | prediction | 3 | 26.0 | 88.9 | -1.4 | 11.2 | 10.8 |
| 2023-08-20 08:40:00 | 21 | 53028 | prediction | 3 | 26.0 | 88.2 | -2.1 | 10.4 | 10.2 |
| 2023-08-20 08:30:00 | 21 | 53027 | prediction | 3 | 26.0 | 88.9 | -1.4 | 9.6 | 9.4 |
| 2023-08-20 08:20:00 | 21 | 53026 | prediction | 3 | 26.0 | 89.7 | -0.6000000000000001 | 9.4 | 9.3 |

---


## dbo.WFA_TURBINE_24_Data

**Primary Key:** EntryDateTime  
**Row Count:** 55,003  
**Date Range:** 2022-04-24 15:00:00 to 2023-05-13 11:20:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-04-24 15:00:00 | 0 | 0 | train | 4 | 25.0 | 317.8 | -1.9 | 3.2 | 3.2 |
| 2022-04-24 15:10:00 | 0 | 1 | train | 4 | 25.0 | 2.5 | 15.5 | 4.0 | 3.9 |
| 2022-04-24 15:20:00 | 0 | 2 | train | 4 | 26.0 | 17.3 | 13.9 | 3.1 | 2.9 |
| 2022-04-24 15:30:00 | 0 | 3 | train | 4 | 26.0 | 47.1 | 67.8 | 4.2 | 4.3 |
| 2022-04-24 15:40:00 | 0 | 4 | train | 4 | 26.0 | 329.3 | -9.6 | 3.1 | 3.0 |
| 2022-04-24 15:50:00 | 0 | 5 | train | 4 | 26.0 | 11.1 | 28.3 | 4.6 | 4.5 |
| 2022-04-24 16:00:00 | 0 | 6 | train | 4 | 25.0 | 329.3 | 6.6 | 5.1 | 4.9 |
| 2022-04-24 16:10:00 | 0 | 7 | train | 4 | 25.0 | 281.2 | -11.8 | 6.2 | 6.0 |
| 2022-04-24 16:20:00 | 0 | 8 | train | 4 | 24.0 | 286.7 | -27.6 | 7.5 | 7.3 |
| 2022-04-24 16:30:00 | 0 | 9 | train | 4 | 24.0 | 288.7 | 2.4 | 9.1 | 9.0 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-05-13 11:20:00 | 0 | 55002 | prediction | 0 | 20.0 | 204.7 | -3.5 | 4.8 | 5.0 |
| 2023-05-13 11:10:00 | 0 | 55001 | prediction | 0 | 20.0 | 208.2 | 4.2 | 4.5 | 4.6 |
| 2023-05-13 11:00:00 | 0 | 55000 | prediction | 0 | 20.0 | 203.1 | -16.3 | 5.8 | 5.8 |
| 2023-05-13 10:50:00 | 0 | 54999 | prediction | 0 | 20.0 | 202.0 | -4.1 | 4.5 | 4.7 |
| 2023-05-13 10:40:00 | 0 | 54998 | prediction | 0 | 20.0 | 241.7 | 22.3 | 3.3 | 3.4 |
| 2023-05-13 10:30:00 | 0 | 54997 | prediction | 0 | 19.0 | 256.9 | 42.4 | 2.6 | 2.6 |
| 2023-05-13 10:20:00 | 0 | 54996 | prediction | 0 | 19.0 | 221.7 | 7.2 | 2.3 | 2.3 |
| 2023-05-13 10:10:00 | 0 | 54995 | prediction | 0 | 19.0 | 247.1 | 16.5 | 2.4 | 2.4 |
| 2023-05-13 10:00:00 | 0 | 54994 | prediction | 0 | 18.0 | 233.1 | 22.8 | 2.8 | 2.8 |
| 2023-05-13 09:50:00 | 0 | 54993 | prediction | 0 | 18.0 | 197.7 | 1.5 | 2.8 | 2.8 |

---


## dbo.WFA_TURBINE_25_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,712  
**Date Range:** 2022-05-23 06:50:00 to 2023-06-09 02:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-05-23 06:50:00 | 11 | 0 | train | 0 | 18.0 | 161.3 | 156.2 | 2.4 | 2.4 |
| 2022-05-23 07:00:00 | 11 | 1 | train | 0 | 18.0 | 172.3 | 167.1 | 2.8 | 2.8 |
| 2022-05-23 07:10:00 | 11 | 2 | train | 0 | 18.0 | 173.0 | 41.0 | 3.1 | 3.1 |
| 2022-05-23 07:20:00 | 11 | 3 | train | 0 | 18.0 | 168.4 | 5.6 | 2.4 | 2.4 |
| 2022-05-23 07:30:00 | 11 | 4 | train | 0 | 18.0 | 155.4 | -18.6 | 2.3 | 2.3 |
| 2022-05-23 07:40:00 | 11 | 5 | train | 0 | 18.0 | 190.8 | 16.8 | 2.4 | 2.4 |
| 2022-05-23 07:50:00 | 11 | 6 | train | 0 | 17.0 | 135.2 | -38.7 | 2.8 | 2.8 |
| 2022-05-23 08:00:00 | 11 | 7 | train | 0 | 16.0 | 149.3 | -2.6 | 2.9 | 2.9 |
| 2022-05-23 08:10:00 | 11 | 8 | train | 0 | 16.0 | 133.2 | -16.2 | 2.6 | 2.6 |
| 2022-05-23 08:20:00 | 11 | 9 | train | 0 | 16.0 | 144.0 | -6.8 | 2.7 | 2.7 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-06-09 02:30:00 | 11 | 54711 | prediction | 0 | 23.0 | 103.5 | -8.8 | 4.3 | 4.1 |
| 2023-06-09 02:20:00 | 11 | 54710 | prediction | 0 | 23.0 | 108.7 | -1.5 | 5.0 | 4.8 |
| 2023-06-09 02:10:00 | 11 | 54709 | prediction | 0 | 23.0 | 112.0 | 1.8 | 6.5 | 6.3 |
| 2023-06-09 02:00:00 | 11 | 54708 | prediction | 0 | 23.0 | 123.1 | 12.9 | 5.9 | 5.8 |
| 2023-06-09 01:50:00 | 11 | 54707 | prediction | 0 | 23.0 | 97.8 | -12.3 | 7.0 | 6.8 |
| 2023-06-09 01:40:00 | 11 | 54706 | prediction | 0 | 23.0 | 107.9 | 5.8 | 8.1 | 7.9 |
| 2023-06-09 01:30:00 | 11 | 54705 | prediction | 0 | 23.0 | 104.8 | 2.7 | 8.9 | 8.6 |
| 2023-06-09 01:20:00 | 11 | 54704 | prediction | 0 | 23.0 | 104.1 | 2.0 | 8.8 | 8.7 |
| 2023-06-09 01:10:00 | 11 | 54703 | prediction | 0 | 23.0 | 96.2 | -5.9 | 9.0 | 8.7 |
| 2023-06-09 01:00:00 | 11 | 54702 | prediction | 0 | 23.0 | 105.8 | 3.7 | 9.2 | 9.0 |

---


## dbo.WFA_TURBINE_26_Data

**Primary Key:** EntryDateTime  
**Row Count:** 53,702  
**Date Range:** 2022-10-12 10:20:00 to 2023-10-22 10:20:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-10-12 10:20:00 | 0 | 0 | train | 0 | 17.0 | 184.0 | -10.8 | 13.2 | 13.4 |
| 2022-10-12 10:30:00 | 0 | 1 | train | 0 | 17.0 | 188.3 | -6.2 | 13.6 | 14.0 |
| 2022-10-12 10:40:00 | 0 | 2 | train | 0 | 17.0 | 186.0 | -8.4 | 12.7 | 13.2 |
| 2022-10-12 10:50:00 | 0 | 3 | train | 0 | 18.0 | 190.8 | -3.7 | 13.4 | 13.8 |
| 2022-10-12 11:00:00 | 0 | 4 | train | 0 | 17.0 | 205.0 | 10.5 | 12.7 | 12.9 |
| 2022-10-12 11:10:00 | 0 | 5 | train | 0 | 17.0 | 192.7 | -1.8 | 12.4 | 12.9 |
| 2022-10-12 11:20:00 | 0 | 6 | train | 0 | 17.0 | 177.5 | -17.3 | 12.3 | 12.8 |
| 2022-10-12 11:30:00 | 0 | 7 | train | 0 | 17.0 | 177.1 | -17.7 | 13.1 | 13.3 |
| 2022-10-12 11:40:00 | 0 | 8 | train | 0 | 16.0 | 198.1 | 3.2 | 13.2 | 13.4 |
| 2022-10-12 11:50:00 | 0 | 9 | train | 0 | 17.0 | 221.3 | 32.8 | 12.6 | 13.1 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-22 10:20:00 | 0 | 53701 | prediction | 0 | 20.0 | 59.6 | 7.4 | 1.4 | 1.4 |
| 2023-10-22 10:10:00 | 0 | 53700 | prediction | 3 | 19.0 | 34.8 | -17.3 | 1.7000000000000002 | 1.7000000000000002 |
| 2023-10-22 10:00:00 | 0 | 53699 | prediction | 3 | 20.0 | 40.0 | -12.2 | 2.0 | 2.0 |
| 2023-10-22 09:50:00 | 0 | 53698 | prediction | 3 | 20.0 | 88.5 | 36.3 | 1.6 | 1.6 |
| 2023-10-22 09:40:00 | 0 | 53697 | prediction | 3 | 21.0 | 117.8 | 65.6 | 1.4 | 1.4 |
| 2023-10-22 09:30:00 | 0 | 53696 | prediction | 3 | 21.0 | 110.5 | 58.3 | 0.9 | 0.9 |
| 2023-10-22 09:20:00 | 0 | 53695 | prediction | 3 | 21.0 | 261.4 | -150.7 | 0.7000000000000001 | 0.7000000000000001 |
| 2023-10-22 09:10:00 | 0 | 53694 | prediction | 3 | 20.0 | 283.8 | -128.3 | 1.0 | 1.0 |
| 2023-10-22 09:00:00 | 0 | 53693 | prediction | 3 | 22.0 | 200.4 | 148.2 | 1.0 | 1.0 |
| 2023-10-22 08:50:00 | 0 | 53692 | prediction | 3 | 20.0 | 190.9 | 138.7 | 1.0 | 1.0 |

---


## dbo.WFA_TURBINE_38_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,835  
**Date Range:** 2022-06-28 15:40:00 to 2023-07-17 07:40:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-06-28 15:40:00 | 13 | 0 | train | 0 | 28.0 | 95.7 | -21.4 | 5.7 | 5.9 |
| 2022-06-28 15:50:00 | 13 | 1 | train | 0 | 28.0 | 95.1 | -15.1 | 7.2 | 7.4 |
| 2022-06-28 16:00:00 | 13 | 2 | train | 0 | 28.0 | 118.3 | 7.7 | 6.2 | 6.3 |
| 2022-06-28 16:10:00 | 13 | 3 | train | 0 | 28.0 | 124.7 | 3.3 | 6.6 | 6.4 |
| 2022-06-28 16:20:00 | 13 | 4 | train | 0 | 27.0 | 104.4 | -0.8 | 7.3 | 7.2 |
| 2022-06-28 16:30:00 | 13 | 5 | train | 0 | 27.0 | 82.4 | -13.4 | 7.1 | 7.1 |
| 2022-06-28 16:40:00 | 13 | 6 | train | 0 | 27.0 | 101.4 | 5.6 | 7.1 | 7.0 |
| 2022-06-28 16:50:00 | 13 | 7 | train | 0 | 28.0 | 93.4 | -2.4 | 6.3 | 6.1 |
| 2022-06-28 17:00:00 | 13 | 8 | train | 0 | 28.0 | 91.6 | 3.4 | 6.4 | 6.6 |
| 2022-06-28 17:10:00 | 13 | 9 | train | 0 | 28.0 | 90.5 | -4.9 | 6.9 | 6.9 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-07-17 07:40:00 | 13 | 54834 | prediction | 0 | 25.0 | 105.9 | -0.7000000000000001 | 7.3 | 7.3 |
| 2023-07-17 07:30:00 | 13 | 54833 | prediction | 0 | 25.0 | 94.6 | 0.6000000000000001 | 3.9 | 3.9 |
| 2023-07-17 07:20:00 | 13 | 54832 | prediction | 0 | 25.0 | 101.6 | 7.6 | 9.1 | 9.0 |
| 2023-07-17 07:10:00 | 13 | 54831 | prediction | 0 | 25.0 | 92.6 | -1.4 | 8.4 | 8.3 |
| 2023-07-17 07:00:00 | 13 | 54830 | prediction | 0 | 25.0 | 92.5 | -8.8 | 7.5 | 7.1 |
| 2023-07-17 06:50:00 | 13 | 54829 | prediction | 0 | 25.0 | 100.8 | 6.4 | 6.2 | 6.0 |
| 2023-07-17 06:40:00 | 13 | 54828 | prediction | 0 | 25.0 | 92.0 | -2.4 | 6.3 | 6.1 |
| 2023-07-17 06:30:00 | 13 | 54827 | prediction | 0 | 25.0 | 99.4 | -1.6 | 6.3 | 5.9 |
| 2023-07-17 06:20:00 | 13 | 54826 | prediction | 0 | 25.0 | 95.1 | 3.2 | 5.4 | 5.3 |
| 2023-07-17 06:10:00 | 13 | 54825 | prediction | 0 | 25.0 | 106.5 | 14.6 | 5.5 | 5.4 |

---


## dbo.WFA_TURBINE_3_Data

**Primary Key:** EntryDateTime  
**Row Count:** 55,487  
**Date Range:** 2022-04-27 03:00:00 to 2023-05-20 01:10:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-04-27 03:00:00 | 10 | 0 | train | 0 | 18.0 | 75.0 | 89.9 | 1.0 | 1.0 |
| 2022-04-27 03:10:00 | 10 | 1 | train | 0 | 18.0 | 154.6 | 169.5 | 1.5 | 1.5 |
| 2022-04-27 03:20:00 | 10 | 2 | train | 0 | 18.0 | 248.8 | -96.3 | 1.3 | 1.3 |
| 2022-04-27 03:30:00 | 10 | 3 | train | 0 | 18.0 | 234.7 | -110.4 | 1.6 | 1.6 |
| 2022-04-27 03:40:00 | 10 | 4 | train | 0 | 18.0 | 334.6 | -10.5 | 2.0 | 2.0 |
| 2022-04-27 03:50:00 | 10 | 5 | train | 0 | 18.0 | 233.0 | -112.1 | 2.3 | 2.3 |
| 2022-04-27 04:00:00 | 10 | 6 | train | 0 | 18.0 | 270.8 | -74.2 | 2.7 | 2.7 |
| 2022-04-27 04:10:00 | 10 | 7 | train | 0 | 18.0 | 299.1 | 22.3 | 2.5 | 2.5 |
| 2022-04-27 04:20:00 | 10 | 8 | train | 0 | 18.0 | 291.4 | 14.7 | 2.3 | 2.3 |
| 2022-04-27 04:30:00 | 10 | 9 | train | 0 | 17.0 | 245.5 | -31.2 | 2.0 | 2.0 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-05-20 01:10:00 | 10 | 55486 | prediction | 0 | 19.0 | 96.7 | 0.2 | 8.4 | 8.3 |
| 2023-05-20 01:00:00 | 10 | 55485 | prediction | 0 | 19.0 | 111.2 | 14.6 | 8.3 | 8.2 |
| 2023-05-20 00:50:00 | 10 | 55484 | prediction | 0 | 19.0 | 83.3 | -13.2 | 8.3 | 8.1 |
| 2023-05-20 00:40:00 | 10 | 55483 | prediction | 0 | 19.0 | 91.5 | -5.0 | 9.3 | 8.9 |
| 2023-05-20 00:30:00 | 10 | 55482 | prediction | 0 | 19.0 | 97.3 | 0.7000000000000001 | 8.4 | 8.2 |
| 2023-05-20 00:20:00 | 10 | 55481 | prediction | 0 | 19.0 | 95.1 | -1.4 | 8.2 | 7.9 |
| 2023-05-20 00:10:00 | 10 | 55480 | prediction | 0 | 19.0 | 91.5 | -13.1 | 5.9 | 5.8 |
| 2023-05-20 00:00:00 | 10 | 55479 | prediction | 0 | 19.0 | 102.3 | -1.3 | 5.3 | 5.2 |
| 2023-05-19 23:50:00 | 10 | 55478 | prediction | 0 | 19.0 | 101.8 | -1.8 | 5.4 | 5.2 |
| 2023-05-19 23:40:00 | 10 | 55477 | prediction | 0 | 19.0 | 110.2 | 6.6 | 5.8 | 5.6 |

---


## dbo.WFA_TURBINE_40_Data

**Primary Key:** EntryDateTime  
**Row Count:** 56,158  
**Date Range:** 2022-01-01 00:00:00 to 2023-01-28 13:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-01-01 00:00:00 | 10 | 0 | train | 0 | 18.0 | 220.1 | 6.1 | 5.2 | 5.1 |
| 2022-01-01 00:10:00 | 10 | 1 | train | 0 | 18.0 | 218.7 | 4.7 | 5.7 | 5.3 |
| 2022-01-01 00:20:00 | 10 | 2 | train | 0 | 18.0 | 216.7 | 2.7 | 6.2 | 5.8 |
| 2022-01-01 00:30:00 | 10 | 3 | train | 0 | 18.0 | 197.9 | -16.1 | 6.3 | 6.2 |
| 2022-01-01 00:40:00 | 10 | 4 | train | 0 | 18.0 | 217.2 | 3.2 | 6.6 | 6.1 |
| 2022-01-01 00:50:00 | 10 | 5 | train | 0 | 18.0 | 215.1 | 1.1 | 6.7 | 6.5 |
| 2022-01-01 01:00:00 | 10 | 6 | train | 0 | 18.0 | 219.6 | 5.6 | 5.5 | 5.3 |
| 2022-01-01 01:10:00 | 10 | 7 | train | 0 | 18.0 | 212.8 | -1.2 | 4.2 | 4.4 |
| 2022-01-01 01:20:00 | 10 | 8 | train | 0 | 18.0 | 230.9 | 11.3 | 5.2 | 5.1 |
| 2022-01-01 01:30:00 | 10 | 9 | train | 0 | 18.0 | 228.2 | 15.6 | 4.6 | 4.6 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-01-28 13:00:00 | 10 | 56157 | prediction | 3 | 15.0 | 318.3 | 38.4 | 10.3 | 10.3 |
| 2023-01-28 12:50:00 | 10 | 56156 | prediction | 3 | 14.0 | 317.3 | 37.4 | 6.9 | 6.9 |
| 2023-01-28 12:40:00 | 10 | 56155 | prediction | 3 | 13.0 | 318.5 | 38.6 | 6.0 | 6.0 |
| 2023-01-28 12:30:00 | 10 | 56154 | prediction | 3 | 13.0 | 325.6 | 45.7 | 7.9 | 7.9 |
| 2023-01-28 12:20:00 | 10 | 56153 | prediction | 3 | 12.0 | 337.5 | 57.5 | 9.3 | 9.3 |
| 2023-01-28 12:10:00 | 10 | 56152 | prediction | 3 | 13.0 | 332.0 | 52.1 | 8.9 | 8.9 |
| 2023-01-28 12:00:00 | 10 | 56151 | prediction | 3 | 15.0 | 282.8 | 2.9 | 7.9 | 7.9 |
| 2023-01-28 11:50:00 | 10 | 56150 | prediction | 3 | 15.0 | 292.8 | 12.8 | 9.8 | 9.8 |
| 2023-01-28 11:40:00 | 10 | 56149 | prediction | 3 | 15.0 | 286.5 | 6.5 | 10.9 | 10.9 |
| 2023-01-28 11:30:00 | 10 | 56148 | prediction | 3 | 16.0 | 283.5 | 3.6 | 10.1 | 10.1 |

---


## dbo.WFA_TURBINE_42_Data

**Primary Key:** EntryDateTime  
**Row Count:** 53,886  
**Date Range:** 2022-09-09 15:50:00 to 2023-09-20 15:50:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-09-09 15:50:00 | 10 | 0 | train | 0 | 28.0 | 298.4 | 21.7 | 5.4 | 5.1 |
| 2022-09-09 16:00:00 | 10 | 1 | train | 0 | 28.0 | 261.6 | -11.9 | 4.2 | 4.1 |
| 2022-09-09 16:10:00 | 10 | 2 | train | 0 | 28.0 | 278.1 | 23.3 | 5.1 | 5.1 |
| 2022-09-09 16:20:00 | 10 | 3 | train | 0 | 28.0 | 260.2 | -19.0 | 5.4 | 5.1 |
| 2022-09-09 16:30:00 | 10 | 4 | train | 0 | 27.0 | 243.5 | -38.5 | 5.1 | 4.9 |
| 2022-09-09 16:40:00 | 10 | 5 | train | 0 | 27.0 | 267.4 | -16.7 | 5.7 | 5.5 |
| 2022-09-09 16:50:00 | 10 | 6 | train | 0 | 27.0 | 287.3 | 3.2 | 5.2 | 5.1 |
| 2022-09-09 17:00:00 | 10 | 7 | train | 0 | 26.0 | 270.7 | -4.7 | 5.7 | 5.5 |
| 2022-09-09 17:10:00 | 10 | 8 | train | 0 | 26.0 | 310.9 | 28.5 | 5.7 | 5.4 |
| 2022-09-09 17:20:00 | 10 | 9 | train | 0 | 26.0 | 278.8 | -18.2 | 5.1 | 5.0 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-09-20 15:50:00 | 10 | 53885 | prediction | 0 | 29.0 | 128.3 | 3.1 | 5.9 | 5.8 |
| 2023-09-20 15:40:00 | 10 | 53884 | prediction | 0 | 29.0 | 124.4 | -6.9 | 6.3 | 5.9 |
| 2023-09-20 15:30:00 | 10 | 53883 | prediction | 0 | 29.0 | 137.0 | 1.4 | 7.0 | 6.9 |
| 2023-09-20 15:20:00 | 10 | 53882 | prediction | 0 | 29.0 | 130.1 | -1.2 | 7.0 | 6.7 |
| 2023-09-20 15:10:00 | 10 | 53881 | prediction | 0 | 29.0 | 130.8 | 0.0 | 7.5 | 7.1 |
| 2023-09-20 15:00:00 | 10 | 53880 | prediction | 0 | 29.0 | 134.1 | 3.8 | 7.9 | 7.7 |
| 2023-09-20 14:50:00 | 10 | 53879 | prediction | 0 | 29.0 | 125.7 | -1.0 | 7.1 | 6.8 |
| 2023-09-20 14:40:00 | 10 | 53878 | prediction | 0 | 29.0 | 141.5 | -1.1 | 7.1 | 6.8 |
| 2023-09-20 14:30:00 | 10 | 53877 | prediction | 0 | 29.0 | 140.1 | 3.0 | 6.6 | 6.3 |
| 2023-09-20 14:20:00 | 10 | 53876 | prediction | 0 | 29.0 | 129.9 | -2.6 | 6.7 | 6.4 |

---


## dbo.WFA_TURBINE_45_Data

**Primary Key:** EntryDateTime  
**Row Count:** 53,739  
**Date Range:** 2022-04-16 18:10:00 to 2023-04-26 18:10:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-04-16 18:10:00 | 13 | 0 | train | 0 | 18.0 | 216.1 | -15.7 | 10.0 | 10.2 |
| 2022-04-16 18:20:00 | 13 | 1 | train | 0 | 18.0 | 240.3 | 7.4 | 9.8 | 9.9 |
| 2022-04-16 18:30:00 | 13 | 2 | train | 0 | 18.0 | 223.8 | -9.1 | 10.5 | 10.6 |
| 2022-04-16 18:40:00 | 13 | 3 | train | 0 | 18.0 | 216.8 | -9.1 | 9.4 | 9.6 |
| 2022-04-16 18:50:00 | 13 | 4 | train | 0 | 18.0 | 239.8 | 5.1 | 8.8 | 9.1 |
| 2022-04-16 19:00:00 | 13 | 5 | train | 0 | 18.0 | 222.1 | -10.4 | 8.8 | 9.1 |
| 2022-04-16 19:10:00 | 13 | 6 | train | 0 | 18.0 | 248.2 | 15.7 | 9.5 | 9.7 |
| 2022-04-16 19:20:00 | 13 | 7 | train | 0 | 18.0 | 257.0 | 24.5 | 9.3 | 9.7 |
| 2022-04-16 19:30:00 | 13 | 8 | train | 0 | 18.0 | 235.3 | 2.8 | 9.4 | 9.6 |
| 2022-04-16 19:40:00 | 13 | 9 | train | 0 | 18.0 | 232.3 | -0.2 | 9.1 | 9.4 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-04-26 18:10:00 | 13 | 53738 | prediction | 3 | 19.0 | 287.4 | 32.8 | 5.5 | 5.5 |
| 2023-04-26 18:00:00 | 13 | 53737 | prediction | 4 | 19.0 | 264.9 | 10.3 | 5.8 | 5.8 |
| 2023-04-26 17:50:00 | 13 | 53736 | prediction | 4 | 19.0 | 255.9 | 1.3 | 6.0 | 6.0 |
| 2023-04-26 17:40:00 | 13 | 53735 | prediction | 4 | 18.0 | 267.1 | 12.4 | 6.0 | 6.0 |
| 2023-04-26 17:30:00 | 13 | 53734 | prediction | 4 | 19.0 | 235.7 | -18.9 | 5.4 | 5.4 |
| 2023-04-26 17:20:00 | 13 | 53733 | prediction | 4 | 19.0 | 248.5 | -6.0 | 5.5 | 5.5 |
| 2023-04-26 17:10:00 | 13 | 53732 | prediction | 4 | 19.0 | 262.6 | 8.0 | 6.1 | 6.1 |
| 2023-04-26 17:00:00 | 13 | 53731 | prediction | 4 | 19.0 | 238.6 | -16.0 | 6.4 | 6.4 |
| 2023-04-26 16:50:00 | 13 | 53730 | prediction | 4 | 19.0 | 261.9 | 7.3 | 5.9 | 5.9 |
| 2023-04-26 16:40:00 | 13 | 53729 | prediction | 4 | 19.0 | 242.8 | -11.8 | 6.4 | 6.4 |

---


## dbo.WFA_TURBINE_51_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,436  
**Date Range:** 2022-10-04 01:30:00 to 2023-10-20 16:10:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-10-04 01:30:00 | 21 | 0 | train | 3 | 23.0 | 113.8 | 11.6 | 5.0 | 5.2 |
| 2022-10-04 01:40:00 | 21 | 1 | train | 3 | 23.0 | 106.5 | 4.2 | 4.7 | 4.7 |
| 2022-10-04 01:50:00 | 21 | 2 | train | 3 | 23.0 | 104.9 | -4.0 | 5.9 | 5.9 |
| 2022-10-04 02:00:00 | 21 | 3 | train | 3 | 23.0 | 109.2 | 15.7 | 5.8 | 5.9 |
| 2022-10-04 02:10:00 | 21 | 4 | train | 3 | 23.0 | 87.4 | 2.0 | 6.1 | 6.1 |
| 2022-10-04 02:20:00 | 21 | 5 | train | 3 | 23.0 | 95.0 | 9.6 | 5.9 | 6.0 |
| 2022-10-04 02:30:00 | 21 | 6 | train | 3 | 23.0 | 78.6 | -6.8 | 5.9 | 6.0 |
| 2022-10-04 02:40:00 | 21 | 7 | train | 3 | 23.0 | 85.1 | -0.3 | 6.4 | 6.6 |
| 2022-10-04 02:50:00 | 21 | 8 | train | 3 | 23.0 | 101.1 | 8.3 | 7.1 | 7.3 |
| 2022-10-04 03:00:00 | 21 | 9 | train | 3 | 23.0 | 84.4 | -8.4 | 7.5 | 7.7 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-20 16:10:00 | 21 | 54435 | prediction | 0 | 24.0 | 292.4 | -3.6 | 4.8 | 5.0 |
| 2023-10-20 16:00:00 | 21 | 54434 | prediction | 0 | 24.0 | 289.2 | 3.6 | 4.6 | 4.8 |
| 2023-10-20 15:50:00 | 21 | 54433 | prediction | 0 | 24.0 | 287.4 | -2.4 | 3.6 | 3.8 |
| 2023-10-20 15:40:00 | 21 | 54432 | prediction | 0 | 24.0 | 286.5 | 0.3 | 4.0 | 4.2 |
| 2023-10-20 15:30:00 | 21 | 54431 | prediction | 0 | 24.0 | 296.0 | 1.1 | 4.5 | 4.6 |
| 2023-10-20 15:20:00 | 21 | 54430 | prediction | 0 | 24.0 | 289.5 | -2.7 | 3.1 | 3.1 |
| 2023-10-20 15:10:00 | 21 | 54429 | prediction | 0 | 24.0 | 288.2 | 5.1 | 3.5 | 3.6 |
| 2023-10-20 15:00:00 | 21 | 54428 | prediction | 0 | 24.0 | 292.1 | -1.1 | 3.9 | 4.0 |
| 2023-10-20 14:50:00 | 21 | 54427 | prediction | 0 | 24.0 | 288.0 | -4.7 | 3.2 | 3.2 |
| 2023-10-20 14:40:00 | 21 | 54426 | prediction | 0 | 24.0 | 301.6 | 2.5 | 3.6 | 3.7 |

---


## dbo.WFA_TURBINE_68_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,358  
**Date Range:** 2022-07-28 13:20:00 to 2023-08-13 13:20:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-07-28 13:20:00 | 11 | 0 | train | 0 | 31.0 | 152.0 | 48.7 | 3.9 | 3.9 |
| 2022-07-28 13:30:00 | 11 | 1 | train | 0 | 31.0 | 86.1 | 150.9 | 6.0 | 6.0 |
| 2022-07-28 13:40:00 | 11 | 2 | train | 0 | 31.0 | 115.2 | 69.6 | 6.3 | 6.3 |
| 2022-07-28 13:50:00 | 11 | 3 | train | 0 | 32.0 | 129.3 | -29.1 | 6.0 | 5.9 |
| 2022-07-28 14:00:00 | 11 | 4 | train | 0 | 32.0 | 137.7 | 26.4 | 7.1 | 6.9 |
| 2022-07-28 14:10:00 | 11 | 5 | train | 0 | 32.0 | 123.7 | 1.6 | 8.1 | 7.8 |
| 2022-07-28 14:20:00 | 11 | 6 | train | 0 | 32.0 | 114.2 | -11.4 | 7.5 | 7.2 |
| 2022-07-28 14:30:00 | 11 | 7 | train | 0 | 32.0 | 137.2 | 3.2 | 8.0 | 7.7 |
| 2022-07-28 14:40:00 | 11 | 8 | train | 0 | 32.0 | 132.9 | -0.4 | 8.4 | 8.1 |
| 2022-07-28 14:50:00 | 11 | 9 | train | 0 | 32.0 | 129.4 | -11.3 | 9.3 | 8.8 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-08-13 13:20:00 | 11 | 54357 | prediction | 3 | 28.0 | 118.3 | 21.4 | 14.2 | 13.4 |
| 2023-08-13 13:10:00 | 11 | 54356 | prediction | 3 | 28.0 | 105.0 | -6.2 | 15.2 | 14.1 |
| 2023-08-13 13:00:00 | 11 | 54355 | prediction | 3 | 28.0 | 98.2 | -6.0 | 14.6 | 13.5 |
| 2023-08-13 12:50:00 | 11 | 54354 | prediction | 3 | 27.0 | 99.8 | -10.7 | 15.7 | 14.2 |
| 2023-08-13 12:40:00 | 11 | 54353 | prediction | 3 | 27.0 | 94.7 | -9.5 | 16.2 | 14.8 |
| 2023-08-13 12:30:00 | 11 | 54352 | prediction | 3 | 27.0 | 111.4 | -0.5 | 17.4 | 16.1 |
| 2023-08-13 12:20:00 | 11 | 54351 | prediction | 3 | 27.0 | 107.3 | 4.5 | 17.1 | 15.7 |
| 2023-08-13 12:10:00 | 11 | 54350 | prediction | 3 | 27.0 | 101.0 | -8.4 | 16.9 | 15.5 |
| 2023-08-13 12:00:00 | 11 | 54349 | prediction | 3 | 27.0 | 113.4 | 10.9 | 15.3 | 14.3 |
| 2023-08-13 11:50:00 | 11 | 54348 | prediction | 3 | 27.0 | 101.0 | -1.5 | 15.3 | 14.0 |

---


## dbo.WFA_TURBINE_69_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,813  
**Date Range:** 2022-09-03 00:50:00 to 2023-09-21 00:50:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-09-03 00:50:00 | 11 | 0 | train | 0 | 26.0 | 111.8 | -8.1 | 5.4 | 5.4 |
| 2022-09-03 01:00:00 | 11 | 1 | train | 0 | 26.0 | 114.0 | -6.0 | 6.0 | 5.6 |
| 2022-09-03 01:10:00 | 11 | 2 | train | 0 | 26.0 | 98.8 | -13.5 | 6.6 | 6.2 |
| 2022-09-03 01:20:00 | 11 | 3 | train | 0 | 26.0 | 112.9 | 0.6000000000000001 | 6.4 | 6.2 |
| 2022-09-03 01:30:00 | 11 | 4 | train | 0 | 26.0 | 114.6 | 2.3 | 6.5 | 6.2 |
| 2022-09-03 01:40:00 | 11 | 5 | train | 0 | 26.0 | 107.9 | -3.3 | 7.3 | 7.0 |
| 2022-09-03 01:50:00 | 11 | 6 | train | 0 | 26.0 | 106.8 | -4.4 | 5.6 | 5.6 |
| 2022-09-03 02:00:00 | 11 | 7 | train | 0 | 26.0 | 100.8 | -17.1 | 5.5 | 5.6 |
| 2022-09-03 02:10:00 | 11 | 8 | train | 0 | 26.0 | 121.9 | 4.0 | 6.6 | 6.2 |
| 2022-09-03 02:20:00 | 11 | 9 | train | 0 | 26.0 | 111.5 | 0.6000000000000001 | 7.7 | 7.2 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-09-21 00:50:00 | 11 | 54812 | prediction | 0 | 22.0 | 102.0 | -1.1 | 6.1 | 5.9 |
| 2023-09-21 00:40:00 | 11 | 54811 | prediction | 0 | 22.0 | 103.0 | -0.2 | 6.9 | 6.7 |
| 2023-09-21 00:30:00 | 11 | 54810 | prediction | 0 | 22.0 | 104.4 | 1.2 | 6.5 | 6.4 |
| 2023-09-21 00:20:00 | 11 | 54809 | prediction | 0 | 23.0 | 104.1 | 0.9 | 6.4 | 6.2 |
| 2023-09-21 00:10:00 | 11 | 54808 | prediction | 0 | 23.0 | 100.6 | -2.5 | 5.5 | 5.3 |
| 2023-09-21 00:00:00 | 11 | 54807 | prediction | 0 | 23.0 | 99.6 | -3.7 | 5.7 | 5.6 |
| 2023-09-20 23:50:00 | 11 | 54806 | prediction | 0 | 23.0 | 100.9 | -2.6 | 5.5 | 5.6 |
| 2023-09-20 23:40:00 | 11 | 54805 | prediction | 0 | 23.0 | 102.3 | -1.7000000000000002 | 6.7 | 6.6 |
| 2023-09-20 23:30:00 | 11 | 54804 | prediction | 0 | 23.0 | 105.4 | -5.8 | 6.8 | 6.3 |
| 2023-09-20 23:20:00 | 11 | 54803 | prediction | 0 | 23.0 | 110.3 | -0.9 | 5.6 | 5.4 |

---


## dbo.WFA_TURBINE_71_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,744  
**Date Range:** 2022-01-01 00:00:00 to 2023-01-18 00:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-01-01 00:00:00 | 0 | 0 | train | 0 | 18.0 | 178.7 | -18.6 | 4.1 | 4.4 |
| 2022-01-01 00:10:00 | 0 | 1 | train | 0 | 18.0 | 191.8 | -12.2 | 4.1 | 4.3 |
| 2022-01-01 00:20:00 | 0 | 2 | train | 0 | 18.0 | 213.8 | 16.8 | 4.1 | 4.4 |
| 2022-01-01 00:30:00 | 0 | 3 | train | 0 | 18.0 | 199.3 | -4.6 | 4.4 | 4.6 |
| 2022-01-01 00:40:00 | 0 | 4 | train | 0 | 18.0 | 199.9 | -4.0 | 5.5 | 5.7 |
| 2022-01-01 00:50:00 | 0 | 5 | train | 0 | 18.0 | 203.6 | 6.7 | 5.4 | 5.7 |
| 2022-01-01 01:00:00 | 0 | 6 | train | 0 | 18.0 | 193.4 | -9.8 | 4.9 | 5.2 |
| 2022-01-01 01:10:00 | 0 | 7 | train | 0 | 18.0 | 215.1 | 11.8 | 4.0 | 4.2 |
| 2022-01-01 01:20:00 | 0 | 8 | train | 0 | 18.0 | 227.0 | 17.8 | 4.4 | 4.6 |
| 2022-01-01 01:30:00 | 0 | 9 | train | 0 | 18.0 | 205.5 | 2.5 | 4.4 | 4.6 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-01-18 00:00:00 | 0 | 54743 | prediction | 0 | 12.0 | 342.7 | -2.5 | 2.9 | 2.9 |
| 2023-01-17 23:50:00 | 0 | 54742 | prediction | 0 | 13.0 | 331.5 | -13.7 | 3.5 | 3.6 |
| 2023-01-17 23:40:00 | 0 | 54741 | prediction | 0 | 13.0 | 343.1 | -2.1 | 4.1 | 4.0 |
| 2023-01-17 23:30:00 | 0 | 54740 | prediction | 0 | 12.0 | 345.5 | 7.6 | 4.0 | 4.0 |
| 2023-01-17 23:20:00 | 0 | 54739 | prediction | 0 | 12.0 | 314.6 | -24.0 | 3.4 | 3.5 |
| 2023-01-17 23:10:00 | 0 | 54738 | prediction | 0 | 12.0 | 328.6 | -3.2 | 4.4 | 4.1 |
| 2023-01-17 23:00:00 | 0 | 54737 | prediction | 0 | 12.0 | 348.5 | 9.2 | 4.9 | 4.7 |
| 2023-01-17 22:50:00 | 0 | 54736 | prediction | 0 | 12.0 | 324.4 | -14.8 | 4.6 | 4.6 |
| 2023-01-17 22:40:00 | 0 | 54735 | prediction | 0 | 12.0 | 331.9 | -7.4 | 4.3 | 4.3 |
| 2023-01-17 22:30:00 | 0 | 54734 | prediction | 0 | 12.0 | 323.6 | 0.4 | 5.1 | 5.1 |

---


## dbo.WFA_TURBINE_72_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,082  
**Date Range:** 2022-10-07 08:40:00 to 2023-10-21 08:40:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-10-07 08:40:00 | 21 | 0 | train | 0 | 18.0 | 171.6 | 61.3 | 2.1 | 2.1 |
| 2022-10-07 08:50:00 | 21 | 1 | train | 0 | 18.0 | 166.0 | 55.7 | 1.6 | 1.6 |
| 2022-10-07 09:00:00 | 21 | 2 | train | 0 | 18.0 | 179.5 | 69.2 | 1.7000000000000002 | 1.7000000000000002 |
| 2022-10-07 09:10:00 | 21 | 3 | train | 0 | 18.0 | 168.3 | 58.0 | 1.8 | 1.8 |
| 2022-10-07 09:20:00 | 21 | 4 | train | 0 | 19.0 | 173.6 | 63.3 | 1.7000000000000002 | 1.7000000000000002 |
| 2022-10-07 09:30:00 | 21 | 5 | train | 0 | 19.0 | 166.1 | 55.8 | 1.1 | 1.1 |
| 2022-10-07 09:40:00 | 21 | 6 | train | 0 | 19.0 | 209.6 | 99.3 | 1.3 | 1.3 |
| 2022-10-07 09:50:00 | 21 | 7 | train | 0 | 19.0 | 212.8 | 102.5 | 1.5 | 1.5 |
| 2022-10-07 10:00:00 | 21 | 8 | train | 0 | 19.0 | 129.6 | 19.2 | 0.9 | 0.9 |
| 2022-10-07 10:10:00 | 21 | 9 | train | 0 | 19.0 | 172.5 | 62.2 | 0.9 | 0.9 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-21 08:40:00 | 21 | 54081 | prediction | 0 | 20.0 | 346.6 | -0.7000000000000001 | 1.4 | 1.4 |
| 2023-10-21 08:30:00 | 21 | 54080 | prediction | 0 | 20.0 | 356.8 | 9.5 | 1.4 | 1.4 |
| 2023-10-21 08:20:00 | 21 | 54079 | prediction | 0 | 20.0 | 348.8 | 1.5 | 1.8 | 1.8 |
| 2023-10-21 08:10:00 | 21 | 54078 | prediction | 0 | 20.0 | 339.7 | -7.6 | 1.6 | 1.6 |
| 2023-10-21 08:00:00 | 21 | 54077 | prediction | 0 | 19.0 | 352.3 | 5.0 | 2.0 | 2.0 |
| 2023-10-21 07:50:00 | 21 | 54076 | prediction | 0 | 19.0 | 356.9 | 9.6 | 2.3 | 2.3 |
| 2023-10-21 07:40:00 | 21 | 54075 | prediction | 0 | 19.0 | 344.1 | -2.8 | 2.6 | 2.6 |
| 2023-10-21 07:30:00 | 21 | 54074 | prediction | 0 | 20.0 | 329.9 | -35.2 | 2.6 | 2.6 |
| 2023-10-21 07:20:00 | 21 | 54073 | prediction | 0 | 19.0 | 301.2 | -63.9 | 1.8 | 1.8 |
| 2023-10-21 07:10:00 | 21 | 54072 | prediction | 0 | 19.0 | 270.5 | -94.6 | 1.4 | 1.4 |

---


## dbo.WFA_TURBINE_73_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,042  
**Date Range:** 2022-06-07 11:40:00 to 2023-06-19 11:40:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-06-07 11:40:00 | 0 | 0 | train | 0 | 26.0 | 79.6 | -7.2 | 9.4 | 9.3 |
| 2022-06-07 11:50:00 | 0 | 1 | train | 0 | 26.0 | 72.7 | -14.2 | 9.8 | 9.8 |
| 2022-06-07 12:00:00 | 0 | 2 | train | 0 | 26.0 | 75.4 | -11.4 | 10.3 | 10.0 |
| 2022-06-07 12:10:00 | 0 | 3 | train | 0 | 26.0 | 97.7 | 10.4 | 10.9 | 10.6 |
| 2022-06-07 12:20:00 | 0 | 4 | train | 0 | 26.0 | 96.8 | 16.9 | 10.1 | 10.0 |
| 2022-06-07 12:30:00 | 0 | 5 | train | 0 | 27.0 | 85.4 | 5.5 | 10.3 | 10.1 |
| 2022-06-07 12:40:00 | 0 | 6 | train | 0 | 27.0 | 66.2 | -14.0 | 10.4 | 10.3 |
| 2022-06-07 12:50:00 | 0 | 7 | train | 0 | 28.0 | 111.7 | 23.7 | 9.8 | 9.5 |
| 2022-06-07 13:00:00 | 0 | 8 | train | 0 | 28.0 | 78.8 | -11.2 | 10.6 | 10.3 |
| 2022-06-07 13:10:00 | 0 | 9 | train | 0 | 27.0 | 95.1 | 5.0 | 11.7 | 11.4 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-06-19 11:40:00 | 0 | 54041 | prediction | 3 | 29.0 | 123.2 | 2.9 | 11.3 | 10.8 |
| 2023-06-19 11:30:00 | 0 | 54040 | prediction | 3 | 28.0 | 106.7 | -19.4 | 11.5 | 10.9 |
| 2023-06-19 11:20:00 | 0 | 54039 | prediction | 3 | 29.0 | 160.4 | 31.8 | 10.8 | 10.3 |
| 2023-06-19 11:10:00 | 0 | 54038 | prediction | 3 | 28.0 | 127.0 | 4.4 | 10.6 | 10.0 |
| 2023-06-19 11:00:00 | 0 | 54037 | prediction | 3 | 28.0 | 114.8 | -7.8 | 9.8 | 9.4 |
| 2023-06-19 10:50:00 | 0 | 54036 | prediction | 3 | 28.0 | 103.3 | -19.4 | 10.3 | 9.9 |
| 2023-06-19 10:40:00 | 0 | 54035 | prediction | 3 | 28.0 | 95.8 | -12.1 | 9.2 | 9.0 |
| 2023-06-19 10:30:00 | 0 | 54034 | prediction | 3 | 28.0 | 121.6 | 12.6 | 10.1 | 9.7 |
| 2023-06-19 10:20:00 | 0 | 54033 | prediction | 3 | 28.0 | 136.3 | 20.3 | 11.1 | 10.5 |
| 2023-06-19 10:10:00 | 0 | 54032 | prediction | 3 | 28.0 | 120.0 | 11.4 | 11.2 | 10.9 |

---


## dbo.WFA_TURBINE_84_Data

**Primary Key:** EntryDateTime  
**Row Count:** 53,772  
**Date Range:** 2022-09-03 15:30:00 to 2023-09-13 15:30:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-09-03 15:30:00 | 13 | 0 | train | 0 | 31.0 | 81.6 | -2.2 | 13.5 | 13.4 |
| 2022-09-03 15:40:00 | 13 | 1 | train | 0 | 31.0 | 89.8 | 13.3 | 13.1 | 12.8 |
| 2022-09-03 15:50:00 | 13 | 2 | train | 0 | 31.0 | 68.7 | -6.7 | 13.3 | 13.2 |
| 2022-09-03 16:00:00 | 13 | 3 | train | 0 | 31.0 | 76.2 | -6.6 | 13.2 | 13.0 |
| 2022-09-03 16:10:00 | 13 | 4 | train | 0 | 31.0 | 70.5 | -5.7 | 12.4 | 12.3 |
| 2022-09-03 16:20:00 | 13 | 5 | train | 0 | 32.0 | 99.0 | 16.9 | 11.2 | 11.0 |
| 2022-09-03 16:30:00 | 13 | 6 | train | 0 | 32.0 | 93.7 | 10.9 | 12.3 | 12.2 |
| 2022-09-03 16:40:00 | 13 | 7 | train | 0 | 32.0 | 84.8 | 2.0 | 12.4 | 11.9 |
| 2022-09-03 16:50:00 | 13 | 8 | train | 0 | 32.0 | 69.7 | -13.1 | 11.4 | 11.2 |
| 2022-09-03 17:00:00 | 13 | 9 | train | 0 | 32.0 | 69.4 | -13.5 | 10.6 | 10.3 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-09-13 15:30:00 | 13 | 53771 | prediction | 3 | 30.0 | 96.7 | 1.3 | 8.1 | 8.2 |
| 2023-09-13 15:20:00 | 13 | 53770 | prediction | 3 | 30.0 | 92.4 | -3.6 | 9.1 | 8.8 |
| 2023-09-13 15:10:00 | 13 | 53769 | prediction | 3 | 30.0 | 89.1 | -0.3 | 8.9 | 8.9 |
| 2023-09-13 15:00:00 | 13 | 53768 | prediction | 3 | 30.0 | 89.1 | -1.7000000000000002 | 10.3 | 9.9 |
| 2023-09-13 14:50:00 | 13 | 53767 | prediction | 3 | 30.0 | 85.2 | -1.5 | 9.4 | 9.3 |
| 2023-09-13 14:40:00 | 13 | 53766 | prediction | 3 | 30.0 | 89.7 | 3.9 | 8.5 | 8.6 |
| 2023-09-13 14:30:00 | 13 | 53765 | prediction | 3 | 29.0 | 84.0 | 0.1 | 10.3 | 10.2 |
| 2023-09-13 14:20:00 | 13 | 53764 | prediction | 3 | 29.0 | 87.5 | 2.5 | 10.0 | 9.8 |
| 2023-09-13 14:10:00 | 13 | 53763 | prediction | 3 | 29.0 | 87.0 | -3.8 | 10.1 | 9.9 |
| 2023-09-13 14:00:00 | 13 | 53762 | prediction | 3 | 29.0 | 88.0 | 2.4 | 10.1 | 10.0 |

---


## dbo.WFA_TURBINE_92_Data

**Primary Key:** EntryDateTime  
**Row Count:** 54,067  
**Date Range:** 2022-04-04 02:30:00 to 2023-04-16 10:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| asset_id | int | NO | 10 | — |
| id | int | NO | 10 | — |
| train_test | varchar | YES | 16 | — |
| status_type_id | int | YES | 10 | — |
| sensor_0_avg | float | YES | 53 | — |
| sensor_1_avg | float | YES | 53 | — |
| sensor_2_avg | float | YES | 53 | — |
| wind_speed_3_avg | float | YES | 53 | — |
| wind_speed_4_avg | float | YES | 53 | — |
| wind_speed_3_max | float | YES | 53 | — |
| wind_speed_3_min | float | YES | 53 | — |
| wind_speed_3_std | float | YES | 53 | — |
| sensor_5_avg | float | YES | 53 | — |
| sensor_5_max | float | YES | 53 | — |
| sensor_5_min | float | YES | 53 | — |
| sensor_5_std | float | YES | 53 | — |
| sensor_6_avg | float | YES | 53 | — |
| sensor_7_avg | float | YES | 53 | — |
| sensor_8_avg | float | YES | 53 | — |
| sensor_9_avg | float | YES | 53 | — |
| sensor_10_avg | float | YES | 53 | — |
| sensor_11_avg | float | YES | 53 | — |
| sensor_12_avg | float | YES | 53 | — |
| sensor_13_avg | float | YES | 53 | — |
| sensor_14_avg | float | YES | 53 | — |
| sensor_15_avg | float | YES | 53 | — |
| sensor_16_avg | float | YES | 53 | — |
| sensor_17_avg | float | YES | 53 | — |
| sensor_18_avg | float | YES | 53 | — |
| sensor_18_max | float | YES | 53 | — |
| sensor_18_min | float | YES | 53 | — |
| sensor_18_std | float | YES | 53 | — |
| sensor_19_avg | float | YES | 53 | — |
| sensor_20_avg | float | YES | 53 | — |
| sensor_21_avg | float | YES | 53 | — |
| sensor_22_avg | float | YES | 53 | — |
| sensor_23_avg | float | YES | 53 | — |
| sensor_24_avg | float | YES | 53 | — |
| sensor_25_avg | float | YES | 53 | — |
| sensor_26_avg | float | YES | 53 | — |
| reactive_power_27_avg | float | YES | 53 | — |
| reactive_power_27_max | float | YES | 53 | — |
| reactive_power_27_min | float | YES | 53 | — |
| reactive_power_27_std | float | YES | 53 | — |
| reactive_power_28_avg | float | YES | 53 | — |
| reactive_power_28_max | float | YES | 53 | — |
| reactive_power_28_min | float | YES | 53 | — |
| reactive_power_28_std | float | YES | 53 | — |
| power_29_avg | float | YES | 53 | — |
| power_29_max | float | YES | 53 | — |
| power_29_min | float | YES | 53 | — |
| power_29_std | float | YES | 53 | — |
| power_30_avg | float | YES | 53 | — |
| power_30_max | float | YES | 53 | — |
| power_30_min | float | YES | 53 | — |
| power_30_std | float | YES | 53 | — |
| sensor_31_avg | float | YES | 53 | — |
| sensor_31_max | float | YES | 53 | — |
| sensor_31_min | float | YES | 53 | — |
| sensor_31_std | float | YES | 53 | — |
| sensor_32_avg | float | YES | 53 | — |
| sensor_33_avg | float | YES | 53 | — |
| sensor_34_avg | float | YES | 53 | — |
| sensor_35_avg | float | YES | 53 | — |
| sensor_36_avg | float | YES | 53 | — |
| sensor_37_avg | float | YES | 53 | — |
| sensor_38_avg | float | YES | 53 | — |
| sensor_39_avg | float | YES | 53 | — |
| sensor_40_avg | float | YES | 53 | — |
| sensor_41_avg | float | YES | 53 | — |
| sensor_42_avg | float | YES | 53 | — |
| sensor_43_avg | float | YES | 53 | — |
| sensor_44 | float | YES | 53 | — |
| sensor_45 | float | YES | 53 | — |
| sensor_46 | float | YES | 53 | — |
| sensor_47 | float | YES | 53 | — |
| sensor_48 | float | YES | 53 | — |
| sensor_49 | float | YES | 53 | — |
| sensor_50 | float | YES | 53 | — |
| sensor_51 | float | YES | 53 | — |
| sensor_52_avg | float | YES | 53 | — |
| sensor_52_max | float | YES | 53 | — |
| sensor_52_min | float | YES | 53 | — |
| sensor_52_std | float | YES | 53 | — |
| sensor_53_avg | float | YES | 53 | — |
| QualityFlag | int | NO | 10 | ((0)) |

### Top 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-04-04 02:30:00 | 11 | 0 | train | 0 | 15.0 | 257.0 | -1.8 | 2.7 | 2.7 |
| 2022-04-04 02:40:00 | 11 | 1 | train | 0 | 15.0 | 267.3 | 13.4 | 2.6 | 2.6 |
| 2022-04-04 02:50:00 | 11 | 2 | train | 0 | 15.0 | 280.9 | 27.0 | 3.0 | 3.0 |
| 2022-04-04 03:00:00 | 11 | 3 | train | 0 | 15.0 | 256.5 | 6.8 | 2.8 | 2.8 |
| 2022-04-04 03:10:00 | 11 | 4 | train | 0 | 15.0 | 260.2 | 7.4 | 3.4 | 3.4 |
| 2022-04-04 03:20:00 | 11 | 5 | train | 0 | 15.0 | 240.2 | -9.5 | 3.8 | 3.9 |
| 2022-04-04 03:30:00 | 11 | 6 | train | 0 | 15.0 | 249.7 | 0.0 | 3.7 | 3.7 |
| 2022-04-04 03:40:00 | 11 | 7 | train | 0 | 15.0 | 228.7 | -18.9 | 4.1 | 4.0 |
| 2022-04-04 03:50:00 | 11 | 8 | train | 0 | 15.0 | 243.3 | -7.0 | 3.7 | 3.6 |
| 2022-04-04 04:00:00 | 11 | 9 | train | 0 | 15.0 | 254.1 | 18.7 | 3.8 | 3.8 |

### Bottom 10 Records

| EntryDateTime | asset_id | id | train_test | status_type_id | sensor_0_avg | sensor_1_avg | sensor_2_avg | wind_speed_3_avg | wind_speed_4_avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-04-16 10:00:00 | 11 | 54066 | prediction | 0 | 16.0 | 91.9 | 17.8 | 3.1 | 3.1 |
| 2023-04-16 09:50:00 | 11 | 54065 | prediction | 0 | 16.0 | 63.5 | -10.6 | 2.4 | 2.4 |
| 2023-04-16 09:40:00 | 11 | 54064 | prediction | 0 | 16.0 | 69.8 | -16.9 | 2.8 | 2.8 |
| 2023-04-16 09:30:00 | 11 | 54063 | prediction | 0 | 16.0 | 69.7 | -32.0 | 2.3 | 2.3 |
| 2023-04-16 09:20:00 | 11 | 54062 | prediction | 0 | 15.0 | 117.8 | 4.8 | 2.7 | 2.7 |
| 2023-04-16 09:10:00 | 11 | 54061 | prediction | 0 | 15.0 | 98.2 | 0.2 | 2.8 | 2.8 |
| 2023-04-16 09:00:00 | 11 | 54060 | prediction | 0 | 15.0 | 79.9 | -21.2 | 3.3 | 3.3 |
| 2023-04-16 08:50:00 | 11 | 54059 | prediction | 0 | 15.0 | 106.3 | -149.1 | 3.7 | 3.7 |
| 2023-04-16 08:40:00 | 11 | 54058 | prediction | 0 | 15.0 | 82.5 | -176.3 | 2.2 | 2.2 |
| 2023-04-16 08:30:00 | 11 | 54057 | prediction | 0 | 15.0 | 98.7 | -160.1 | 1.7000000000000002 | 1.7000000000000002 |

---


## dbo.WIND_TURBINE_Data

**Primary Key:** No primary key  
**Row Count:** 50,530  
**Date Range:** 2024-01-01 00:00:00 to 2024-12-31 23:50:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EntryDateTime | datetime2 | NO | — | — |
| LV_ActivePower | float | YES | 53 | — |
| Wind_Speed | float | YES | 53 | — |
| Theoretical_Power_Curve | float | YES | 53 | — |
| Wind_Direction | float | YES | 53 | — |

### Top 10 Records

| EntryDateTime | LV_ActivePower | Wind_Speed | Theoretical_Power_Curve | Wind_Direction |
| --- | --- | --- | --- | --- |
| 2024-01-01 00:00:00 | 380.0477905 | 5.31133604 | 416.3289078 | 259.9949036 |
| 2024-01-01 00:10:00 | 453.7691956 | 5.672166824 | 519.9175111 | 268.6411133 |
| 2024-01-01 00:20:00 | 306.3765869 | 5.216036797 | 390.9000158 | 272.5647888 |
| 2024-01-01 00:30:00 | 419.6459045 | 5.659674168 | 516.127569 | 271.2580872 |
| 2024-01-01 00:40:00 | 380.6506958 | 5.577940941 | 491.702972 | 265.6742859 |
| 2024-01-01 00:50:00 | 402.3919983 | 5.604052067 | 499.436385 | 264.5786133 |
| 2024-01-01 01:00:00 | 447.6057129 | 5.793007851 | 557.3723633 | 266.1636047 |
| 2024-01-01 01:10:00 | 387.2421875 | 5.306049824 | 414.8981788 | 257.9494934 |
| 2024-01-01 01:20:00 | 463.6512146 | 5.584629059 | 493.6776521 | 253.4806976 |
| 2024-01-01 01:30:00 | 439.725708 | 5.523228168 | 475.7067828 | 258.7237854 |

### Bottom 10 Records

| EntryDateTime | LV_ActivePower | Wind_Speed | Theoretical_Power_Curve | Wind_Direction |
| --- | --- | --- | --- | --- |
| 2024-12-31 23:50:00 | 2820.466064 | 9.97933197 | 2779.184096 | 82.27462006 |
| 2024-12-31 23:40:00 | 2515.694092 | 9.421365738 | 2418.382503 | 84.2979126 |
| 2024-12-31 23:30:00 | 2201.106934 | 8.435358047 | 1788.284755 | 84.74250031 |
| 2024-12-31 23:20:00 | 1684.353027 | 7.3326478 | 1173.055771 | 84.06259918 |
| 2024-12-31 23:10:00 | 2963.980957 | 11.40402985 | 3397.190793 | 80.50272369 |
| 2024-12-31 23:00:00 | 3514.269043 | 12.55916977 | 3583.288363 | 80.49526215 |
| 2024-12-31 22:50:00 | 3429.021973 | 12.49250984 | 3578.567804 | 82.11186981 |
| 2024-12-31 22:40:00 | 3455.282959 | 12.19565964 | 3549.150371 | 82.21061707 |
| 2024-12-31 22:30:00 | 3333.819092 | 12.06766033 | 3532.081496 | 81.98590088 |
| 2024-12-31 22:20:00 | 2771.110107 | 10.1545496 | 2884.512812 | 82.33519745 |

---

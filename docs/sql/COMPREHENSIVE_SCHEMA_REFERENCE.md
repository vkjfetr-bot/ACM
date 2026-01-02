# ACM Comprehensive Database Schema Reference

_Generated automatically on 2025-12-31 10:00:09_

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

- [dbo.ACM_ActiveModels](#dboacmactivemodels)
- [dbo.ACM_AdaptiveConfig](#dboacmadaptiveconfig)
- [dbo.ACM_AlertAge](#dboacmalertage)
- [dbo.ACM_Anomaly_Events](#dboacmanomalyevents)
- [dbo.ACM_AssetProfiles](#dboacmassetprofiles)
- [dbo.ACM_BaselineBuffer](#dboacmbaselinebuffer)
- [dbo.ACM_CalibrationSummary](#dboacmcalibrationsummary)
- [dbo.ACM_ColdstartState](#dboacmcoldstartstate)
- [dbo.ACM_Config](#dboacmconfig)
- [dbo.ACM_ConfigHistory](#dboacmconfighistory)
- [dbo.ACM_ContributionCurrent](#dboacmcontributioncurrent)
- [dbo.ACM_ContributionTimeline](#dboacmcontributiontimeline)
- [dbo.ACM_DailyFusedProfile](#dboacmdailyfusedprofile)
- [dbo.ACM_DataContractValidation](#dboacmdatacontractvalidation)
- [dbo.ACM_DataQuality](#dboacmdataquality)
- [dbo.ACM_DefectSummary](#dboacmdefectsummary)
- [dbo.ACM_DefectTimeline](#dboacmdefecttimeline)
- [dbo.ACM_DetectorCorrelation](#dboacmdetectorcorrelation)
- [dbo.ACM_DetectorForecast_TS](#dboacmdetectorforecastts)
- [dbo.ACM_DriftController](#dboacmdriftcontroller)
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
- [dbo.ACM_RUL_LearningState](#dboacmrullearningstate)
- [dbo.ACM_RUL_Summary](#dboacmrulsummary)
- [dbo.ACM_RUL_TS](#dboacmrults)
- [dbo.ACM_RecommendedActions](#dboacmrecommendedactions)
- [dbo.ACM_RefitRequests](#dboacmrefitrequests)
- [dbo.ACM_RegimeDefinitions](#dboacmregimedefinitions)
- [dbo.ACM_RegimeDwellStats](#dboacmregimedwellstats)
- [dbo.ACM_RegimeOccupancy](#dboacmregimeoccupancy)
- [dbo.ACM_RegimePromotionLog](#dboacmregimepromotionlog)
- [dbo.ACM_RegimeStability](#dboacmregimestability)
- [dbo.ACM_RegimeState](#dboacmregimestate)
- [dbo.ACM_RegimeStats](#dboacmregimestats)
- [dbo.ACM_RegimeTimeline](#dboacmregimetimeline)
- [dbo.ACM_RegimeTransitions](#dboacmregimetransitions)
- [dbo.ACM_Regime_Episodes](#dboacmregimeepisodes)
- [dbo.ACM_RunLogs](#dboacmrunlogs)
- [dbo.ACM_RunMetadata](#dboacmrunmetadata)
- [dbo.ACM_RunMetrics](#dboacmrunmetrics)
- [dbo.ACM_RunTimers](#dboacmruntimers)
- [dbo.ACM_Run_Stats](#dboacmrunstats)
- [dbo.ACM_Runs](#dboacmruns)
- [dbo.ACM_SchemaVersion](#dboacmschemaversion)
- [dbo.ACM_Scores_Long](#dboacmscoreslong)
- [dbo.ACM_Scores_Wide](#dboacmscoreswide)
- [dbo.ACM_SeasonalPatterns](#dboacmseasonalpatterns)
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
| dbo.ACM_ActiveModels | 11 | 7 | ID |
| dbo.ACM_AdaptiveConfig | 13 | 23 | ConfigID |
| dbo.ACM_AlertAge | 6 | 0 | — |
| dbo.ACM_Anomaly_Events | 7 | 6 | Id |
| dbo.ACM_AssetProfiles | 11 | 6 | ID |
| dbo.ACM_BaselineBuffer | 7 | 96,328 | Id |
| dbo.ACM_CalibrationSummary | 10 | 0 | ID |
| dbo.ACM_ColdstartState | 17 | 14 | EquipID, Stage |
| dbo.ACM_Config | 7 | 336 | ConfigID |
| dbo.ACM_ConfigHistory | 9 | 10 | ID |
| dbo.ACM_ContributionCurrent | 5 | 0 | — |
| dbo.ACM_ContributionTimeline | 7 | 0 | ID |
| dbo.ACM_DailyFusedProfile | 9 | 0 | ID |
| dbo.ACM_DataContractValidation | 10 | 26 | ID |
| dbo.ACM_DataQuality | 24 | 101 | — |
| dbo.ACM_DefectSummary | 12 | 0 | — |
| dbo.ACM_DefectTimeline | 10 | 0 | — |
| dbo.ACM_DetectorCorrelation | 7 | 343 | ID |
| dbo.ACM_DetectorForecast_TS | 10 | 0 | RunID, EquipID, DetectorName, Timestamp |
| dbo.ACM_DriftController | 10 | 0 | ID |
| dbo.ACM_DriftEvents | 2 | 0 | — |
| dbo.ACM_DriftSeries | 7 | 0 | ID |
| dbo.ACM_EnhancedFailureProbability_TS | 11 | 0 | RunID, EquipID, Timestamp, ForecastHorizon_Hours |
| dbo.ACM_EnhancedMaintenanceRecommendation | 13 | 0 | RunID, EquipID |
| dbo.ACM_EpisodeCulprits | 9 | 31 | ID |
| dbo.ACM_EpisodeDiagnostics | 16 | 7 | ID |
| dbo.ACM_EpisodeMetrics | 10 | 0 | — |
| dbo.ACM_Episodes | 8 | 5 | — |
| dbo.ACM_EpisodesQC | 10 | 0 | RecordID |
| dbo.ACM_FailureCausation | 12 | 0 | RunID, EquipID, Detector |
| dbo.ACM_FailureForecast | 9 | 336 | EquipID, RunID, Timestamp |
| dbo.ACM_FailureForecast_TS | 7 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_FailureHazard_TS | 8 | 0 | EquipID, RunID, Timestamp |
| dbo.ACM_FeatureDropLog | 8 | 534 | ID |
| dbo.ACM_ForecastState | 12 | 0 | EquipID, StateVersion |
| dbo.ACM_ForecastingState | 13 | 1 | EquipID, StateVersion |
| dbo.ACM_FusionQualityReport | 9 | 0 | — |
| dbo.ACM_HealthDistributionOverTime | 12 | 0 | — |
| dbo.ACM_HealthForecast | 10 | 336 | EquipID, RunID, Timestamp |
| dbo.ACM_HealthForecast_Continuous | 8 | 0 | EquipID, Timestamp, SourceRunID |
| dbo.ACM_HealthForecast_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_HealthHistogram | 5 | 0 | — |
| dbo.ACM_HealthTimeline | 10 | 664 | — |
| dbo.ACM_HealthZoneByPeriod | 9 | 0 | — |
| dbo.ACM_HistorianData | 7 | 0 | DataID |
| dbo.ACM_MaintenanceRecommendation | 8 | 0 | RunID, EquipID |
| dbo.ACM_OMRContributionsLong | 8 | 0 | — |
| dbo.ACM_OMRTimeline | 6 | 0 | — |
| dbo.ACM_OMR_Diagnostics | 15 | 5 | DiagnosticID |
| dbo.ACM_PCA_Loadings | 10 | 16,800 | RecordID |
| dbo.ACM_PCA_Metrics | 8 | 7 | ID |
| dbo.ACM_PCA_Models | 12 | 6 | RecordID |
| dbo.ACM_RUL | 32 | 1 | EquipID, RunID |
| dbo.ACM_RUL_Attribution | 9 | 0 | RunID, EquipID, FailureTime, SensorName |
| dbo.ACM_RUL_LearningState | 19 | 0 | EquipID |
| dbo.ACM_RUL_Summary | 15 | 0 | RunID, EquipID |
| dbo.ACM_RUL_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_RecommendedActions | 6 | 0 | RunID, EquipID, Action |
| dbo.ACM_RefitRequests | 10 | 7 | RequestID |
| dbo.ACM_RegimeDefinitions | 12 | 18 | ID |
| dbo.ACM_RegimeDwellStats | 8 | 0 | — |
| dbo.ACM_RegimeOccupancy | 9 | 0 | ID |
| dbo.ACM_RegimePromotionLog | 9 | 0 | ID |
| dbo.ACM_RegimeStability | 4 | 0 | — |
| dbo.ACM_RegimeState | 15 | 6 | EquipID, StateVersion |
| dbo.ACM_RegimeStats | 8 | 0 | — |
| dbo.ACM_RegimeTimeline | 7 | 664 | — |
| dbo.ACM_RegimeTransitions | 8 | 0 | ID |
| dbo.ACM_Regime_Episodes | 6 | 6 | Id |
| dbo.ACM_RunLogs | 25 | 10,437 | LogID |
| dbo.ACM_RunMetadata | 12 | 0 | RunMetadataID |
| dbo.ACM_RunMetrics | 5 | 588 | RunID, EquipID, MetricName |
| dbo.ACM_RunTimers | 7 | 205,327 | TimerID |
| dbo.ACM_Run_Stats | 13 | 6 | RecordID |
| dbo.ACM_Runs | 19 | 41 | RunID |
| dbo.ACM_SchemaVersion | 5 | 2 | VersionID |
| dbo.ACM_Scores_Long | 9 | 0 | Id |
| dbo.ACM_Scores_Wide | 15 | 713 | — |
| dbo.ACM_SeasonalPatterns | 10 | 7 | ID |
| dbo.ACM_SensorAnomalyByPeriod | 11 | 0 | — |
| dbo.ACM_SensorCorrelations | 8 | 1,256,079 | ID |
| dbo.ACM_SensorDefects | 11 | 42 | — |
| dbo.ACM_SensorForecast | 11 | 1,512 | RunID, EquipID, Timestamp, SensorName |
| dbo.ACM_SensorForecast_TS | 10 | 0 | RunID, EquipID, SensorName, Timestamp |
| dbo.ACM_SensorHotspotTimeline | 9 | 0 | — |
| dbo.ACM_SensorHotspots | 18 | 125 | — |
| dbo.ACM_SensorNormalized_TS | 8 | 43,951 | ID |
| dbo.ACM_SensorRanking | 6 | 0 | — |
| dbo.ACM_SinceWhen | 6 | 0 | — |
| dbo.ACM_TagEquipmentMap | 10 | 1,986 | TagID |
| dbo.ACM_ThresholdCrossings | 7 | 0 | — |
| dbo.ACM_ThresholdMetadata | 13 | 0 | ThresholdID |
| dbo.ELECTRIC_MOTOR_Data | 14 | 17,477 | — |
| dbo.ELECTRIC_MOTOR_Data_RAW | 14 | 1,048,575 | — |
| dbo.Equipment | 8 | 29 | EquipID |
| dbo.FD_FAN_Data | 11 | 17,499 | EntryDateTime |
| dbo.GAS_TURBINE_Data | 18 | 2,911 | EntryDateTime |
| dbo.ModelRegistry | 8 | 33 | ModelType, EquipID, Version |
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



## dbo.ACM_ActiveModels

**Primary Key:** ID  
**Row Count:** 7  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| EquipID | int | NO | 10 | — |
| ActiveRegimeVersion | int | YES | 10 | — |
| RegimeMaturityState | nvarchar | YES | 30 | — |
| RegimePromotedAt | datetime2 | YES | — | — |
| ActiveThresholdVersion | int | YES | 10 | — |
| ThresholdPromotedAt | datetime2 | YES | — | — |
| ActiveForecastVersion | int | YES | 10 | — |
| ForecastPromotedAt | datetime2 | YES | — | — |
| LastUpdatedAt | datetime2 | NO | — | (getutcdate()) |
| LastUpdatedBy | nvarchar | YES | 100 | — |

### Top 10 Records

| ID | EquipID | ActiveRegimeVersion | RegimeMaturityState | RegimePromotedAt | ActiveThresholdVersion | ThresholdPromotedAt | ActiveForecastVersion | ForecastPromotedAt | LastUpdatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 5000 | 1 | LEARNING | NULL | NULL | NULL | NULL | NULL | 2025-12-27 11:56:23 |
| 5 | 5003 | 1 | LEARNING | NULL | NULL | NULL | NULL | NULL | 2025-12-27 11:56:23 |
| 6 | 5013 | 1 | LEARNING | NULL | NULL | NULL | NULL | NULL | 2025-12-27 11:56:26 |
| 7 | 5010 | 1 | LEARNING | NULL | NULL | NULL | NULL | NULL | 2025-12-27 11:56:52 |
| 9 | 2621 | 1 | LEARNING | NULL | NULL | NULL | NULL | NULL | 2025-12-29 15:27:23 |
| 26 | 1 | 1 | LEARNING | NULL | NULL | NULL | NULL | NULL | 2025-12-30 17:23:55 |
| 27 | 5092 | 0 | LEARNING | NULL | NULL | NULL | NULL | NULL | 2025-12-31 09:54:25 |

---


## dbo.ACM_AdaptiveConfig

**Primary Key:** ConfigID  
**Row Count:** 23  
**Date Range:** 2025-12-04 10:46:47 to 2025-12-31 09:54:45  

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
| 16 | 5000 | fused_alert_z | 1.4745798110961914 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated from 129 accumulated samples |

### Bottom 10 Records

| ConfigID | EquipID | ConfigKey | ConfigValue | MinBound | MaxBound | IsLearned | DataVolumeAtTuning | PerformanceMetric | ResearchReference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 55 | 5092 | fused_warn_z | 1.5 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 54 | 5092 | fused_alert_z | 3.0 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated from 49 accumulated samples |
| 53 | 1 | fused_warn_z | 1.5 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 52 | 1 | fused_alert_z | 3.0 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated from 49 accumulated samples |
| 27 | 2621 | fused_warn_z | 1.5 | 0.0 | 999999.0 | True | 97 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 26 | 2621 | fused_alert_z | 3.0 | 0.0 | 999999.0 | True | 97 | 0.0 | quantile_0.997: Auto-calculated from 97 accumulated samples |
| 23 | 5010 | fused_warn_z | 0.7621059417724609 | 0.0 | 999999.0 | True | 131 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 22 | 5010 | fused_alert_z | 1.5242118835449219 | 0.0 | 999999.0 | True | 131 | 0.0 | quantile_0.997: Auto-calculated from 131 accumulated samples |
| 21 | 5013 | fused_warn_z | 0.5874592661857605 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 20 | 5013 | fused_alert_z | 1.174918532371521 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated from 129 accumulated samples |

---


## dbo.ACM_AlertAge

**Primary Key:** No primary key  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| AlertZone | nvarchar | NO | 50 | — |
| StartTimestamp | datetime2 | NO | — | — |
| DurationHours | float | NO | 53 | — |
| RecordCount | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

---


## dbo.ACM_Anomaly_Events

**Primary Key:** Id  
**Row Count:** 6  
**Date Range:** 2022-05-03 13:30:00 to 2022-10-13 13:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | YES | 10 | — |
| StartTime | datetime2 | YES | — | — |
| EndTime | datetime2 | YES | — | — |
| Severity | nvarchar | YES | 32 | — |
| Confidence | float | YES | 53 | — |

### Top 10 Records

| Id | RunID | EquipID | StartTime | EndTime | Severity | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| 381 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2022-05-05 19:20:00 | 2022-05-06 01:50:00 | info | NULL |
| 382 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2022-05-06 21:50:00 | 2022-05-07 03:50:00 | info | NULL |
| 383 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 2022-05-03 13:30:00 | 2022-05-03 17:00:00 | info | NULL |
| 384 | 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | 2022-08-09 01:10:00 | 2022-08-09 02:10:00 | info | NULL |
| 385 | 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | 2022-08-09 21:40:00 | 2022-08-10 14:10:00 | info | NULL |
| 386 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2022-10-13 13:00:00 | 2022-10-13 20:00:00 | info | NULL |

---


## dbo.ACM_AssetProfiles

**Primary Key:** ID  
**Row Count:** 6  
**Date Range:** 2025-12-27 12:24:04 to 2025-12-30 17:24:44  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| EquipID | int | NO | 10 | — |
| EquipType | nvarchar | NO | 100 | — |
| SensorNamesJSON | nvarchar | NO | -1 | — |
| SensorMeansJSON | nvarchar | NO | -1 | — |
| SensorStdsJSON | nvarchar | NO | -1 | — |
| RegimeCount | int | YES | 10 | — |
| TypicalHealth | float | YES | 53 | — |
| DataHours | float | YES | 53 | — |
| LastUpdatedAt | datetime2 | NO | — | (getutcdate()) |
| LastUpdatedByRunID | nvarchar | YES | 50 | — |

### Top 10 Records

| ID | EquipID | EquipType | SensorNamesJSON | SensorMeansJSON | SensorStdsJSON | RegimeCount | TypicalHealth | DataHours | LastUpdatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 5013 | WFA_TURBINE_13 | ["power_29_avg", "power_29_max", "power_29_min", "power_29_std", "power_30_avg", "power_30_max", ... | {"power_29_avg": 0.39128342270851135, "power_29_max": 0.539457380771637, "power_29_min": 0.235427... | {"power_29_avg": 0.34754589200019836, "power_29_max": 0.3684498071670532, "power_29_min": 0.28379... | 5 | 85.0 | 64.0 | 2025-12-27 12:24:04 |
| 4 | 5003 | WFA_TURBINE_3 | ["power_29_avg", "power_29_max", "power_29_min", "power_29_std", "power_30_avg", "power_30_max", ... | {"power_29_avg": 0.7298295497894287, "power_29_max": 0.8526586890220642, "power_29_min": 0.441692... | {"power_29_avg": 0.27196452021598816, "power_29_max": 0.24192680418491364, "power_29_min": 0.2381... | 2 | 85.0 | 64.0 | 2025-12-27 12:24:27 |
| 5 | 5000 | WFA_TURBINE_0 | ["power_29_avg", "power_29_max", "power_29_min", "power_29_std", "power_30_avg", "power_30_max", ... | {"power_29_avg": 0.6802162528038025, "power_29_max": 0.7749495506286621, "power_29_min": 0.477090... | {"power_29_avg": 0.34482336044311523, "power_29_max": 0.34186238050460815, "power_29_min": 0.3117... | 2 | 85.0 | 64.0 | 2025-12-27 12:24:28 |
| 6 | 5010 | WFA_TURBINE_10 | ["power_29_avg", "power_29_max", "power_29_min", "power_29_std", "power_30_avg", "power_30_max", ... | {"power_29_avg": 0.069669708609581, "power_29_max": 0.13419324159622192, "power_29_min": 0.023929... | {"power_29_avg": 0.10068515688180923, "power_29_max": 0.16620099544525146, "power_29_min": 0.0436... | 3 | 85.0 | 65.0 | 2025-12-27 12:24:55 |
| 8 | 2621 | GAS_TURBINE | ["ACTTBTEMP1", "B1RADVIBX", "B1RADVIBY", "B1TEMP1", "B1VIB1", "B1VIB2", "B2RADVIBX", "B2RADVIBY",... | {"ACTTBTEMP1": 92.13705444335938, "B1RADVIBX": 0.11145060509443283, "B1RADVIBY": 0.10637401044368... | {"ACTTBTEMP1": 0.48425886034965515, "B1RADVIBX": 0.04079153388738632, "B1RADVIBY": 0.030386300757... | 2 | 85.0 | 96.0 | 2025-12-29 15:29:09 |
| 22 | 1 | FD_FAN | ["DEMO.SIM.06G31_1FD Fan Damper Position", "DEMO.SIM.06GP34_1FD Fan Outlet Pressure", "DEMO.SIM.0... | {"DEMO.SIM.06G31_1FD Fan Damper Position": 34.10795974731445, "DEMO.SIM.06GP34_1FD Fan Outlet Pre... | {"DEMO.SIM.06G31_1FD Fan Damper Position": 2.9000372886657715, "DEMO.SIM.06GP34_1FD Fan Outlet Pr... | 4 | 85.0 | 24.0 | 2025-12-30 17:24:44 |

---


## dbo.ACM_BaselineBuffer

**Primary Key:** Id  
**Row Count:** 96,328  
**Date Range:** 2018-01-05 00:00:00 to 2024-10-19 18:50:00  

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
| 13215340 | 8632 | 2018-01-05 00:00:00 | LV_ActivePower | 432.54400634765625 | NULL | 2025-12-11 17:29:21 |
| 13215341 | 8632 | 2018-01-05 00:00:00 | Theoretical_Power_Curve | 572.2332763671875 | NULL | 2025-12-11 17:29:21 |
| 13215342 | 8632 | 2018-01-05 00:00:00 | Wind_Direction | 303.6416931152344 | NULL | 2025-12-11 17:29:21 |
| 13215343 | 8632 | 2018-01-05 00:00:00 | Wind_Speed | 5.839684963226318 | NULL | 2025-12-11 17:29:21 |
| 13215344 | 8632 | 2018-01-05 00:30:00 | LV_ActivePower | 176.84620666503906 | NULL | 2025-12-11 17:29:21 |
| 13215345 | 8632 | 2018-01-05 00:30:00 | Theoretical_Power_Curve | 313.94146728515625 | NULL | 2025-12-11 17:29:21 |
| 13215346 | 8632 | 2018-01-05 00:30:00 | Wind_Direction | 296.4529113769531 | NULL | 2025-12-11 17:29:21 |
| 13215347 | 8632 | 2018-01-05 00:30:00 | Wind_Speed | 4.908945083618164 | NULL | 2025-12-11 17:29:21 |
| 13215348 | 8632 | 2018-01-05 01:00:00 | LV_ActivePower | 141.7480926513672 | NULL | 2025-12-11 17:29:21 |
| 13215349 | 8632 | 2018-01-05 01:00:00 | Theoretical_Power_Curve | 243.74163818359375 | NULL | 2025-12-11 17:29:21 |

### Bottom 10 Records

| Id | EquipID | Timestamp | SensorName | SensorValue | DataQuality | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 13311667 | 8632 | 2024-10-19 18:50:00 | Wind_Speed | 2.921562910079956 | NULL | 2025-12-11 18:04:42 |
| 13311666 | 8632 | 2024-10-19 18:50:00 | Wind_Direction | 62.68605041503906 | NULL | 2025-12-11 18:04:42 |
| 13311665 | 8632 | 2024-10-19 18:50:00 | Theoretical_Power_Curve | 0.0 | NULL | 2025-12-11 18:04:42 |
| 13311664 | 8632 | 2024-10-19 18:50:00 | LV_ActivePower | 0.0 | NULL | 2025-12-11 18:04:42 |
| 13311663 | 8632 | 2024-10-19 18:20:00 | Wind_Speed | 2.765307903289795 | NULL | 2025-12-11 18:04:42 |
| 13311662 | 8632 | 2024-10-19 18:20:00 | Wind_Direction | 65.24392700195312 | NULL | 2025-12-11 18:04:42 |
| 13311661 | 8632 | 2024-10-19 18:20:00 | Theoretical_Power_Curve | 0.0 | NULL | 2025-12-11 18:04:42 |
| 13311660 | 8632 | 2024-10-19 18:20:00 | LV_ActivePower | 0.0 | NULL | 2025-12-11 18:04:42 |
| 13311659 | 8632 | 2024-10-19 17:50:00 | Wind_Speed | 3.3385400772094727 | NULL | 2025-12-11 18:04:42 |
| 13311658 | 8632 | 2024-10-19 17:50:00 | Wind_Direction | 52.429908752441406 | NULL | 2025-12-11 18:04:42 |

---


## dbo.ACM_CalibrationSummary

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| DetectorType | nvarchar | NO | 50 | — |
| CalibrationScore | float | YES | 53 | — |
| TrainR2 | float | YES | 53 | — |
| MeanAbsError | float | YES | 53 | — |
| P95Error | float | YES | 53 | — |
| DatapointsUsed | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

---


## dbo.ACM_ColdstartState

**Primary Key:** EquipID, Stage  
**Row Count:** 14  
**Date Range:** 2025-12-20 04:25:52 to 2025-12-31 04:19:00  

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
| 1 | score | COMPLETE | 1 | 2025-12-30 11:52:40 | 2025-12-30 11:52:41 | 2025-12-30 11:52:41 | 121 | 100 | 2023-10-15 00:00:00 |
| 2621 | score | COMPLETE | 2 | 2025-12-29 09:48:11 | 2025-12-29 09:51:26 | 2025-12-29 09:51:26 | 482 | 200 | 2023-10-15 00:00:00 |
| 5000 | score | COMPLETE | 3 | 2025-12-27 05:49:00 | 2025-12-27 05:49:09 | 2025-12-27 05:49:09 | 563 | 200 | 2022-08-04 06:10:00 |
| 5003 | score | COMPLETE | 3 | 2025-12-27 05:49:00 | 2025-12-27 05:49:09 | 2025-12-27 05:49:09 | 563 | 200 | 2022-04-27 03:00:00 |
| 5010 | score | COMPLETE | 3 | 2025-12-27 05:49:00 | 2025-12-27 05:49:12 | 2025-12-27 05:49:12 | 563 | 200 | 2022-10-09 08:40:00 |
| 5013 | score | COMPLETE | 3 | 2025-12-27 05:49:00 | 2025-12-27 05:49:09 | 2025-12-27 05:49:09 | 563 | 200 | 2022-04-30 13:20:00 |
| 5014 | score | COMPLETE | 6 | 2025-12-27 06:56:20 | 2025-12-27 06:57:11 | 2025-12-27 06:57:11 | 1126 | 200 | 2022-03-03 14:00:00 |
| 5017 | score | COMPLETE | 6 | 2025-12-27 06:56:37 | 2025-12-27 06:57:24 | 2025-12-27 06:57:24 | 1126 | 200 | 2022-10-31 15:20:00 |
| 5022 | score | COMPLETE | 6 | 2025-12-27 06:56:40 | 2025-12-27 06:57:23 | 2025-12-27 06:57:23 | 1126 | 200 | 2022-08-12 09:50:00 |
| 5024 | score | COMPLETE | 6 | 2025-12-27 06:57:06 | 2025-12-27 06:57:49 | 2025-12-27 06:57:49 | 1126 | 200 | 2022-04-24 15:00:00 |

### Bottom 10 Records

| EquipID | Stage | Status | AttemptCount | FirstAttemptAt | LastAttemptAt | CompletedAt | AccumulatedRows | RequiredRows | DataStartTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8632 | score | IN_PROGRESS | 15 | 2025-12-20 04:25:52 | 2025-12-20 04:26:22 | 2025-12-20 04:25:52 | 1719 | 200 | 2024-01-01 00:00:00 |
| 5092 | score | COMPLETE | 1 | 2025-12-31 04:19:00 | 2025-12-31 04:19:02 | 2025-12-31 04:19:02 | 121 | 100 | 2022-04-04 02:30:00 |
| 5026 | score | COMPLETE | 6 | 2025-12-27 06:57:52 | 2025-12-27 06:58:39 | 2025-12-27 06:58:39 | 1126 | 200 | 2022-10-12 10:20:00 |
| 5025 | score | COMPLETE | 6 | 2025-12-27 06:57:36 | 2025-12-27 06:58:24 | 2025-12-27 06:58:24 | 1126 | 200 | 2022-05-23 06:50:00 |
| 5024 | score | COMPLETE | 6 | 2025-12-27 06:57:06 | 2025-12-27 06:57:49 | 2025-12-27 06:57:49 | 1126 | 200 | 2022-04-24 15:00:00 |
| 5022 | score | COMPLETE | 6 | 2025-12-27 06:56:40 | 2025-12-27 06:57:23 | 2025-12-27 06:57:23 | 1126 | 200 | 2022-08-12 09:50:00 |
| 5017 | score | COMPLETE | 6 | 2025-12-27 06:56:37 | 2025-12-27 06:57:24 | 2025-12-27 06:57:24 | 1126 | 200 | 2022-10-31 15:20:00 |
| 5014 | score | COMPLETE | 6 | 2025-12-27 06:56:20 | 2025-12-27 06:57:11 | 2025-12-27 06:57:11 | 1126 | 200 | 2022-03-03 14:00:00 |
| 5013 | score | COMPLETE | 3 | 2025-12-27 05:49:00 | 2025-12-27 05:49:09 | 2025-12-27 05:49:09 | 563 | 200 | 2022-04-30 13:20:00 |
| 5010 | score | COMPLETE | 3 | 2025-12-27 05:49:00 | 2025-12-27 05:49:12 | 2025-12-27 05:49:12 | 563 | 200 | 2022-10-09 08:40:00 |

---


## dbo.ACM_Config

**Primary Key:** ConfigID  
**Row Count:** 336  
**Date Range:** 2025-12-09 12:47:06 to 2025-12-31 04:18:52  

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
| 492 | 0 | data.train_csv | data/FD_FAN_BASELINE_DATA.csv | string | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 493 | 0 | data.score_csv | data/FD_FAN_BATCH_DATA.csv | string | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 494 | 0 | data.data_dir | data | string | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 495 | 0 | data.timestamp_col | EntryDateTime | string | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 496 | 0 | data.tag_columns | [] | list | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 497 | 0 | data.sampling_secs | 60 | int | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 498 | 0 | data.max_rows | 100000 | int | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 499 | 0 | features.window | 16 | int | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 500 | 0 | features.fft_bands | [0.0, 0.1, 0.3, 0.5] | list | 2025-12-30 12:20:44 | B19cl3pc\bhadk |
| 501 | 0 | features.top_k_tags | 5 | int | 2025-12-30 12:20:44 | B19cl3pc\bhadk |

### Bottom 10 Records

| ConfigID | EquipID | ParamPath | ParamValue | ValueType | UpdatedAt | UpdatedBy |
| --- | --- | --- | --- | --- | --- | --- |
| 928 | 5092 | runtime.tick_minutes | 54333 | int | 2025-12-31 04:18:52 | sql_batch_runner |
| 927 | 5026 | data.tag_columns | ["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3... | list | 2025-12-30 12:20:45 | B19cl3pc\bhadk |
| 926 | 5026 | data.sampling_secs | 600 | int | 2025-12-30 12:20:45 | B19cl3pc\bhadk |
| 925 | 5026 | data.timestamp_col | EntryDateTime | string | 2025-12-30 12:20:45 | B19cl3pc\bhadk |
| 924 | 5025 | data.tag_columns | ["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3... | list | 2025-12-30 12:20:45 | B19cl3pc\bhadk |
| 923 | 5025 | data.sampling_secs | 600 | int | 2025-12-30 12:20:45 | B19cl3pc\bhadk |
| 922 | 5025 | data.timestamp_col | EntryDateTime | string | 2025-12-30 12:20:45 | B19cl3pc\bhadk |
| 921 | 5024 | data.tag_columns | ["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3... | list | 2025-12-30 12:20:45 | B19cl3pc\bhadk |
| 920 | 5024 | data.sampling_secs | 600 | int | 2025-12-30 12:20:45 | B19cl3pc\bhadk |
| 919 | 5024 | data.timestamp_col | EntryDateTime | string | 2025-12-30 12:20:45 | B19cl3pc\bhadk |

---


## dbo.ACM_ConfigHistory

**Primary Key:** ID  
**Row Count:** 10  
**Date Range:** 2025-12-11 17:29:31 to 2025-12-31 09:57:18  

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
| 11 | 2025-12-11 17:29:31 | 8632 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 3.04e+38 exceeds 1e28 (critical instability) | 979ae5c4-a559-4f23-bda9-04d800d60ac0 |
| 13492 | 2025-12-27 12:00:48 | 5000 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 03097745-09ca-4250-b0c2-1d64c948247f |
| 13493 | 2025-12-27 12:00:48 | 5003 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 61a6978e-47c5-494f-a88a-e4e71e9989a1 |
| 13494 | 2025-12-27 12:00:49 | 5013 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | b1795ae8-0a34-4097-ac71-0c3cbff9592d |
| 13495 | 2025-12-27 12:01:16 | 5010 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | d648127b-adde-4248-8a81-75034636efe8 |
| 13497 | 2025-12-29 15:28:13 | 2621 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 3879d1b4-a397-4926-84af-46dfd2dece43 |
| 13525 | 2025-12-30 17:24:22 | 1 | episodes.cpd.k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa |
| 13526 | 2025-12-30 17:24:22 | 1 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa |
| 13527 | 2025-12-31 09:57:18 | 5092 | episodes.cpd.k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | b434fc8c-641b-4922-9a00-4d9cd738daeb |
| 13528 | 2025-12-31 09:57:18 | 5092 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | b434fc8c-641b-4922-9a00-4d9cd738daeb |

---


## dbo.ACM_ContributionCurrent

**Primary Key:** No primary key  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorType | nvarchar | NO | 50 | — |
| ContributionPct | float | NO | 53 | — |
| ZScore | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

---


## dbo.ACM_ContributionTimeline

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| DetectorType | nvarchar | NO | 50 | — |
| ContributionPct | float | NO | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

---


## dbo.ACM_DailyFusedProfile

**Primary Key:** ID  
**Row Count:** 0  

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

---


## dbo.ACM_DataContractValidation

**Primary Key:** ID  
**Row Count:** 26  
**Date Range:** 2025-12-27 11:19:09 to 2025-12-31 09:49:02  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| Passed | bit | NO | — | — |
| RowsValidated | int | NO | 10 | — |
| ColumnsValidated | int | NO | 10 | — |
| IssuesJSON | nvarchar | YES | -1 | — |
| WarningsJSON | nvarchar | YES | -1 | — |
| ContractSignature | nvarchar | YES | 100 | — |
| ValidatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | Passed | RowsValidated | ColumnsValidated | IssuesJSON | WarningsJSON | ContractSignature | ValidatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 71d074cc5cf1 | 2025-12-27 11:19:09 |
| 5 | 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 2abf45df7787 | 2025-12-27 11:19:09 |
| 6 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | f74e3d03db67 | 2025-12-27 11:19:10 |
| 7 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | False | 131 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | d5dda024e44d | 2025-12-27 11:19:12 |
| 8 | 211fc759-412b-4d44-9feb-7685e19039e2 | 5013 | False | 749 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 71d074cc5cf1 | 2025-12-27 12:25:48 |
| 9 | c5f3cca9-adfa-41f8-aa2f-45aa569d80ce | 5003 | False | 749 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | f74e3d03db67 | 2025-12-27 12:26:08 |
| 10 | e8008d4a-b739-4894-aa3a-f62215ae4135 | 5000 | False | 749 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 2abf45df7787 | 2025-12-27 12:26:11 |
| 11 | a7156ce9-07eb-40f4-8158-2cb83f0d0db0 | 5014 | False | 128 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 22d389599bb4 | 2025-12-27 12:26:31 |
| 12 | cf404cbc-d3df-4178-b5e9-6dbd10d0d1a5 | 5010 | False | 727 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | d5dda024e44d | 2025-12-27 12:26:34 |
| 13 | c8ea4c4a-0b58-40c8-b284-01b64d26d884 | 5017 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 432f58c3355a | 2025-12-27 12:26:46 |

### Bottom 10 Records

| ID | RunID | EquipID | Passed | RowsValidated | ColumnsValidated | IssuesJSON | WarningsJSON | ContractSignature | ValidatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 71 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | True | 49 | 81 | NULL | NULL | ee73ded9e2a4 | 2025-12-31 09:49:02 |
| 70 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | True | 49 | 9 | NULL | NULL | ed632fb386d6 | 2025-12-30 17:22:42 |
| 43 | 3879d1b4-a397-4926-84af-46dfd2dece43 | 2621 | True | 97 | 16 | NULL | NULL | 1598146da8ea | 2025-12-29 15:21:26 |
| 42 | 97e7f9ee-c00d-460f-baab-a671d2b1528a | 2621 | True | 97 | 16 | NULL | NULL | 1598146da8ea | 2025-12-29 15:18:13 |
| 37 | 7b7ccff1-0a54-43a3-ac2b-a6497524b591 | 2621 | False | 97 | 16 | ["Missing timestamp column: EntryDateTime", "Insufficient rows: 97 < 100"] | NULL | 7a211a9e7e50 | 2025-12-29 14:49:40 |
| 36 | f585e91b-d893-447d-bba7-09a0bc0e4655 | 2621 | False | 97 | 16 | ["Missing timestamp column: EntryDateTime", "Insufficient rows: 97 < 100"] | NULL | 7a211a9e7e50 | 2025-12-29 14:49:16 |
| 25 | dc5383bf-4484-4982-8f3a-e127ba879a11 | 5026 | False | 127 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | f4e463b52b45 | 2025-12-27 12:28:39 |
| 23 | 7842171d-f59a-4e6d-84fa-b506b12b32fa | 5025 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 3c776b413cc5 | 2025-12-27 12:28:24 |
| 21 | 2e79c7a6-8dab-4567-b187-e0d9ab4c54d2 | 5026 | False | 127 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | f4e463b52b45 | 2025-12-27 12:28:03 |
| 20 | c6f10086-6226-4886-8877-a412b38f79d9 | 5024 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 814cdbd30915 | 2025-12-27 12:27:49 |

---


## dbo.ACM_DataQuality

**Primary Key:** No primary key  
**Row Count:** 101  
**Date Range:** 2022-04-04 14:30:00 to 2024-01-05 00:00:00  

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
| sensor_24_avg | 49 | 0 | 0.0 | 140.97512817382812 | 0 | 0 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_25_avg | 49 | 0 | 0.0 | 140.8668670654297 | 0 | 0 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_26_avg | 49 | 0 | 0.0 | 0.0 | 0 | 48 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_2_avg | 49 | 0 | 0.0 | 18.01894760131836 | 0 | 0 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_31_avg | 49 | 0 | 0.0 | 0.002772704930976033 | 0 | 0 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_31_max | 49 | 0 | 0.0 | 7.635170936584473 | 0 | 1 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_31_min | 49 | 0 | 0.0 | 5.194819927215576 | 0 | 1 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_31_std | 49 | 0 | 0.0 | 1.0864697694778442 | 0 | 2 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_32_avg | 49 | 0 | 0.0 | 1.4606834650039673 | 0 | 1 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |
| sensor_33_avg | 49 | 0 | 0.0 | 1.485682487487793 | 0 | 0 | 2022-04-04 14:30:00 | 2022-04-04 22:30:00 | 49 |

### Bottom 10 Records

| sensor | train_count | train_nulls | train_null_pct | train_std | train_longest_gap | train_flatline_span | train_min_ts | train_max_ts | score_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2RADVIBX | 97 | 0 | 0.0 | 0.01760849356651306 | 0 | 0 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| B1VIB2 | 97 | 0 | 0.0 | 0.00018338656809646636 | 0 | 0 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| B1VIB1 | 97 | 0 | 0.0 | 0.0001708765485091135 | 0 | 1 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| B1TEMP1 | 97 | 0 | 0.0 | 0.45875027775764465 | 0 | 0 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| B1RADVIBY | 97 | 0 | 0.0 | 0.03038630075752735 | 0 | 1 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| B1RADVIBX | 97 | 0 | 0.0 | 0.04079153388738632 | 0 | 0 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| ACTTBTEMP1 | 97 | 0 | 0.0 | 0.48425886034965515 | 0 | 0 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| B2RADVIBY | 97 | 0 | 0.0 | 0.028653090819716454 | 0 | 0 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| B2TEMP1 | 97 | 0 | 0.0 | 0.4339117407798767 | 0 | 0 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |
| B2VIB1 | 97 | 0 | 0.0 | 0.00036073365481570363 | 0 | 0 | 2023-10-20 23:59:00 | 2023-10-24 23:59:00 | 97 |

---


## dbo.ACM_DefectSummary

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_DefectTimeline

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_DetectorCorrelation

**Primary Key:** ID  
**Row Count:** 343  
**Date Range:** 2025-12-27 06:31:05 to 2025-12-31 04:27:32  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| Detector1 | nvarchar | NO | 50 | — |
| Detector2 | nvarchar | NO | 50 | — |
| Correlation | float | NO | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | Detector1 | Detector2 | Correlation | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 3725 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | ar1_z | ar1_z | 1.0 | 2025-12-27 06:31:05 |
| 3726 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | ar1_z | pca_spe_z | -0.24868604227785962 | 2025-12-27 06:31:05 |
| 3727 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | ar1_z | pca_t2_z | -0.2852374977110462 | 2025-12-27 06:31:05 |
| 3728 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | ar1_z | iforest_z | 0.7868171705039417 | 2025-12-27 06:31:05 |
| 3729 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | ar1_z | gmm_z | 0.61193463789505 | 2025-12-27 06:31:05 |
| 3730 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | ar1_z | omr_z | 0.582251203022299 | 2025-12-27 06:31:05 |
| 3731 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | ar1_z | cusum_z | 0.40394543757007867 | 2025-12-27 06:31:05 |
| 3732 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | pca_spe_z | ar1_z | -0.24868604227785962 | 2025-12-27 06:31:05 |
| 3733 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | pca_spe_z | pca_spe_z | 1.0 | 2025-12-27 06:31:05 |
| 3734 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | pca_spe_z | pca_t2_z | 0.9922914105928333 | 2025-12-27 06:31:05 |

### Bottom 10 Records

| ID | RunID | EquipID | Detector1 | Detector2 | Correlation | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 4753 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | cusum_z | cusum_z | 1.0 | 2025-12-31 04:27:32 |
| 4752 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | cusum_z | omr_z | 0.0 | 2025-12-31 04:27:32 |
| 4751 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | cusum_z | gmm_z | 0.7108452848514479 | 2025-12-31 04:27:32 |
| 4750 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | cusum_z | iforest_z | 0.47687232914845445 | 2025-12-31 04:27:32 |
| 4749 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | cusum_z | pca_t2_z | 0.23001474611842193 | 2025-12-31 04:27:32 |
| 4748 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | cusum_z | pca_spe_z | 0.22096144845861806 | 2025-12-31 04:27:32 |
| 4747 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | cusum_z | ar1_z | 0.1483513750676419 | 2025-12-31 04:27:32 |
| 4746 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | omr_z | cusum_z | 0.0 | 2025-12-31 04:27:32 |
| 4745 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | omr_z | omr_z | 0.0 | 2025-12-31 04:27:32 |
| 4744 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | omr_z | gmm_z | 0.0 | 2025-12-31 04:27:32 |

---


## dbo.ACM_DetectorForecast_TS

**Primary Key:** RunID, EquipID, DetectorName, Timestamp  
**Row Count:** 0  

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

---


## dbo.ACM_DriftController

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| ControllerState | nvarchar | NO | 30 | — |
| Threshold | float | YES | 53 | — |
| Sensitivity | float | YES | 53 | — |
| LastDriftValue | float | YES | 53 | — |
| LastDriftTime | datetime2 | YES | — | — |
| ResetCount | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

---


## dbo.ACM_DriftEvents

**Primary Key:** No primary key  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

---


## dbo.ACM_DriftSeries

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| DriftValue | float | NO | 53 | — |
| DriftState | nvarchar | YES | 20 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

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
**Row Count:** 31  
**Date Range:** 2025-12-27 06:54:38 to 2025-12-27 06:55:29  

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
| 150564 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 1 | Density Anomaly (GMM) | NULL | 31.398883819580078 | 1 | 2025-12-27 06:54:38 | 5013 |
| 150565 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 1 | Time-Series Anomaly (AR1) | NULL | 25.082210540771484 | 2 | 2025-12-27 06:54:38 | 5013 |
| 150566 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 1 | Rare State (IsolationForest) | NULL | 18.64297866821289 | 3 | 2025-12-27 06:54:38 | 5013 |
| 150567 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 1 | Baseline Consistency (OMR) | NULL | 18.352582931518555 | 4 | 2025-12-27 06:54:38 | 5013 |
| 150568 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 1 | cusum_z | NULL | 6.523347854614258 | 5 | 2025-12-27 06:54:38 | 5013 |
| 150569 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 2 | Rare State (IsolationForest) | NULL | 34.761024475097656 | 1 | 2025-12-27 06:54:38 | 5013 |
| 150570 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 2 | Time-Series Anomaly (AR1) | NULL | 31.293701171875 | 2 | 2025-12-27 06:54:38 | 5013 |
| 150571 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 2 | cusum_z | NULL | 15.905296325683594 | 3 | 2025-12-27 06:54:38 | 5013 |
| 150572 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 2 | Density Anomaly (GMM) | NULL | 9.13425350189209 | 4 | 2025-12-27 06:54:38 | 5013 |
| 150573 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 2 | Baseline Consistency (OMR) | NULL | 8.905723571777344 | 5 | 2025-12-27 06:54:38 | 5013 |

### Bottom 10 Records

| ID | RunID | EpisodeID | DetectorType | SensorName | ContributionPct | Rank | CreatedAt | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 150594 | D648127B-ADDE-4248-8A81-75034636EFE8 | 1 | Multivariate Outlier (PCA-T2) | NULL | 1.2696409225463867 | 6 | 2025-12-27 06:55:29 | 5010 |
| 150593 | D648127B-ADDE-4248-8A81-75034636EFE8 | 1 | Baseline Consistency (OMR) | NULL | 11.446576118469238 | 5 | 2025-12-27 06:55:29 | 5010 |
| 150592 | D648127B-ADDE-4248-8A81-75034636EFE8 | 1 | Time-Series Anomaly (AR1) | NULL | 19.971088409423828 | 4 | 2025-12-27 06:55:29 | 5010 |
| 150591 | D648127B-ADDE-4248-8A81-75034636EFE8 | 1 | cusum_z | NULL | 22.111421585083008 | 3 | 2025-12-27 06:55:29 | 5010 |
| 150590 | D648127B-ADDE-4248-8A81-75034636EFE8 | 1 | Rare State (IsolationForest) | NULL | 22.269657135009766 | 2 | 2025-12-27 06:55:29 | 5010 |
| 150589 | D648127B-ADDE-4248-8A81-75034636EFE8 | 1 | Density Anomaly (GMM) | NULL | 22.931621551513672 | 1 | 2025-12-27 06:55:29 | 5010 |
| 150588 | 03097745-09CA-4250-B0C2-1D64C948247F | 2 | Baseline Consistency (OMR) | NULL | 10.163461685180664 | 5 | 2025-12-27 06:55:03 | 5000 |
| 150587 | 03097745-09CA-4250-B0C2-1D64C948247F | 2 | Time-Series Anomaly (AR1) | NULL | 18.30774688720703 | 4 | 2025-12-27 06:55:03 | 5000 |
| 150586 | 03097745-09CA-4250-B0C2-1D64C948247F | 2 | Rare State (IsolationForest) | NULL | 20.13909149169922 | 3 | 2025-12-27 06:55:03 | 5000 |
| 150585 | 03097745-09CA-4250-B0C2-1D64C948247F | 2 | cusum_z | NULL | 24.558935165405273 | 2 | 2025-12-27 06:55:03 | 5000 |

---


## dbo.ACM_EpisodeDiagnostics

**Primary Key:** ID  
**Row Count:** 7  
**Date Range:** 2022-04-04 18:40:00 to 2022-10-13 13:00:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| EpisodeID | int | YES | 10 | — |
| StartTime | datetime2 | NO | — | — |
| EndTime | datetime2 | YES | — | — |
| DurationHours | float | YES | 53 | — |
| PeakZ | float | YES | 53 | — |
| AvgZ | float | YES | 53 | — |
| Severity | nvarchar | YES | 20 | — |
| TopSensor1 | nvarchar | YES | 200 | — |
| TopSensor2 | nvarchar | YES | 200 | — |
| TopSensor3 | nvarchar | YES | 200 | — |
| RegimeAtStart | nvarchar | YES | 50 | — |
| AlertMode | nvarchar | YES | 50 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | EpisodeID | StartTime | EndTime | DurationHours | PeakZ | AvgZ | Severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 575 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | 1 | 2022-05-03 13:30:00 | 2022-05-03 17:00:00 | 3.5 | 1.3405905304603716 | 0.7888532335090899 | LOW |
| 576 | 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | 1 | 2022-08-09 01:10:00 | 2022-08-09 02:10:00 | 1.0 | 1.1679315361915623 | 0.6428725300299372 | LOW |
| 577 | 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | 2 | 2022-08-09 21:40:00 | 2022-08-10 14:10:00 | 16.5 | 0.8849055986861347 | 0.5153642647715159 | LOW |
| 578 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 1 | 2022-05-05 19:20:00 | 2022-05-06 01:50:00 | 6.5 | 1.1149577005805225 | 0.6497194487681559 | LOW |
| 579 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2 | 2022-05-06 21:50:00 | 2022-05-07 03:50:00 | 6.0 | 0.7903578147671737 | 0.5064891643713243 | LOW |
| 580 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 1 | 2022-10-13 13:00:00 | 2022-10-13 20:00:00 | 7.0 | 1.5744499986548597 | 1.0450690668091742 | LOW |
| 1213 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | 1 | 2022-04-04 18:40:00 | 2022-04-04 20:30:00 | 1.8333333333333333 | 1.2276367723269928 | 0.906276384649631 | LOW |

---


## dbo.ACM_EpisodeMetrics

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_Episodes

**Primary Key:** No primary key  
**Row Count:** 5  

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
| B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2 | 375.0 | NULL | NULL | 1.1149577005805225 | 0.5781043065697401 |
| 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | 2 | 525.0 | NULL | NULL | 1.1679315361915623 | 0.5791183974007266 |
| B434FC8C-641B-4922-9A00-4D9CD738DAEB | 5092 | 1 | 110.0 | NULL | NULL | 1.2276367723269928 | 0.906276384649631 |
| D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 1 | 420.0 | NULL | NULL | 1.5744499986548597 | 1.0450690668091742 |
| 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 1 | 210.0 | NULL | NULL | 1.3405905304603716 | 0.7888532335090899 |

---


## dbo.ACM_EpisodesQC

**Primary Key:** RecordID  
**Row Count:** 0  

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
**Row Count:** 336  
**Date Range:** 2023-10-17 12:30:00 to 2023-10-24 12:00:00  

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
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 12:30:00 | 1.0866587646621502e-08 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 13:00:00 | 1.0102321216001997e-08 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 13:30:00 | 9.390348474025462e-09 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 14:00:00 | 8.727197161958785e-09 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 14:30:00 | 8.109618072026863e-09 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 15:00:00 | 7.53457123797892e-09 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 15:30:00 | 6.999212945674716e-09 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 16:00:00 | 6.50088351728363e-09 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 16:30:00 | 6.037095825172962e-09 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 17:00:00 | 5.605524493915512e-09 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |

### Bottom 10 Records

| EquipID | RunID | Timestamp | FailureProb | SurvivalProb | HazardRate | ThresholdUsed | Method | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 12:00:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 11:30:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 11:00:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 10:30:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 10:00:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 09:30:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 09:00:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 08:30:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 08:00:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 07:30:00 | 7.914548857487e-13 | 0.9999999891334124 | 0.0 | 50.0 | GaussianTail | 2025-12-30 17:25:05 |

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
**Row Count:** 0  

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

---


## dbo.ACM_FeatureDropLog

**Primary Key:** ID  
**Row Count:** 534  
**Date Range:** 2025-12-27 06:21:53 to 2025-12-31 04:21:33  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| FeatureName | nvarchar | NO | 200 | — |
| DropReason | nvarchar | NO | 100 | — |
| DropValue | float | YES | 53 | — |
| Threshold | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | FeatureName | DropReason | DropValue | Threshold | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_7_avg_energy_0 | low_variance | 1.1455798979978607e-27 | NULL | 2025-12-27 06:21:53 |
| 11 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_49_energy_2 | low_variance | 0.0 | NULL | 2025-12-27 06:21:53 |
| 12 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_41_avg_energy_0 | low_variance | 1.9084039671002483e-28 | NULL | 2025-12-27 06:21:53 |
| 13 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_46_std | low_variance | 0.0 | NULL | 2025-12-27 06:21:53 |
| 14 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_49_energy_1 | low_variance | 0.0 | NULL | 2025-12-27 06:21:53 |
| 15 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_43_avg_energy_0 | low_variance | 3.760478588369181e-29 | NULL | 2025-12-27 06:21:53 |
| 16 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_46_med | low_variance | 0.0 | NULL | 2025-12-27 06:21:53 |
| 17 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_17_avg_energy_0 | low_variance | 1.2323868146266205e-27 | NULL | 2025-12-27 06:21:53 |
| 18 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_51_energy_0 | low_variance | 1.6704663340960258e-21 | NULL | 2025-12-27 06:21:53 |
| 19 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | sensor_50_energy_0 | low_variance | 1.104756364502596e-19 | NULL | 2025-12-27 06:21:53 |

### Bottom 10 Records

| ID | RunID | EquipID | FeatureName | DropReason | DropValue | Threshold | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 590 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | reactive_power_27_min_skew | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 589 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | sensor_22_avg_std | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 588 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | power_29_max_skew | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 587 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | sensor_47_skew | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 586 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | sensor_49_std | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 585 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | sensor_46_med | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 584 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | sensor_49_mad | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 583 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | reactive_power_28_min_slope | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 582 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | sensor_47_mad | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |
| 581 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | sensor_6_avg_med | low_variance | 0.0 | NULL | 2025-12-31 04:21:33 |

---


## dbo.ACM_ForecastState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 0  

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

---


## dbo.ACM_ForecastingState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 1  

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
| 1 | 1 | {"alpha": 0.1, "beta": 0.15, "level": 93.70858841155022, "trend": 0.053630680385821555, "std_erro... | {"forecast_mean": 98.91106378230567, "forecast_std": 1.8335694716875912, "forecast_range": 6.2377... | NULL |  | 49 | 1.8335694716875912 | NULL | NULL |

---


## dbo.ACM_FusionQualityReport

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_HealthDistributionOverTime

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_HealthForecast

**Primary Key:** EquipID, RunID, Timestamp  
**Row Count:** 336  
**Date Range:** 2023-10-17 12:30:00 to 2023-10-24 12:00:00  

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
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 12:30:00 | 93.76221909193605 | 90.18947968922403 | 97.33495849464806 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 13:00:00 | 93.81584977232187 | 90.21956322733607 | 97.41213631730767 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 13:30:00 | 93.86948045270769 | 90.24332596024095 | 97.49563494517443 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 14:00:00 | 93.92311113309351 | 90.26013838974131 | 97.58608387644571 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 14:30:00 | 93.97674181347934 | 90.26943286093204 | 97.68405076602663 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 15:00:00 | 94.03037249386516 | 90.27071140985734 | 97.79003357787298 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 15:30:00 | 94.08400317425098 | 90.26355179784065 | 97.90445455066131 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 16:00:00 | 94.1376338546368 | 90.24761130477239 | 98.02765640450122 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 16:30:00 | 94.19126453502261 | 90.22262803951251 | 98.15990103053271 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-17 17:00:00 | 94.24489521540843 | 90.18841973604009 | 98.30137069477678 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |

### Bottom 10 Records

| EquipID | RunID | Timestamp | ForecastHealth | CiLower | CiUpper | ForecastStd | Method | CreatedAt | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 12:00:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 11:30:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 11:00:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 10:30:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 10:00:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 09:30:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 09:00:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 08:30:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 08:00:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 2023-10-24 07:30:00 | 100.0 | 0.0 | 100.0 | 1.7863697013560085 | ExponentialSmoothing | 2025-12-30 17:25:02 | NULL |

---


## dbo.ACM_HealthForecast_Continuous

**Primary Key:** EquipID, Timestamp, SourceRunID  
**Row Count:** 0  

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
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| HealthBin | nvarchar | NO | 50 | — |
| RecordCount | int | NO | 10 | — |
| Percentage | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

---


## dbo.ACM_HealthTimeline

**Primary Key:** No primary key  
**Row Count:** 664  
**Date Range:** 2022-05-01 03:00:00 to 2023-10-24 23:59:00  

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
| Confidence | float | YES | 53 | — |
| ConfidenceFactors | nvarchar | YES | 200 | — |

### Top 10 Records

| Timestamp | HealthIndex | HealthZone | FusedZ | RunID | EquipID | RawHealthIndex | QualityFlag | Confidence | ConfidenceFactors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-05-01 03:00:00 | 75.91 | WATCH | -1.5434000492095947 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 75.91000366210938 | NORMAL | NULL | NULL |
| 2022-05-01 03:30:00 | 80.02 | WATCH | -0.7046999931335449 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 89.61000061035156 | NORMAL | NULL | NULL |
| 2022-05-01 04:00:00 | 83.24 | WATCH | -0.5954999923706055 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 90.7699966430664 | NORMAL | NULL | NULL |
| 2022-05-01 04:30:00 | 83.3 | WATCH | -1.1533000469207764 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 83.43000030517578 | NORMAL | NULL | NULL |
| 2022-05-01 05:00:00 | 83.01 | WATCH | -1.2170000076293945 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 82.33999633789062 | NORMAL | NULL | NULL |
| 2022-05-01 05:30:00 | 81.8 | WATCH | -1.3976999521255493 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 78.95999908447266 | NORMAL | NULL | NULL |
| 2022-05-01 06:00:00 | 83.28 | WATCH | -0.9348999857902527 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 86.73999786376953 | NORMAL | NULL | NULL |
| 2022-05-01 06:30:00 | 86.39 | GOOD | 0.2581000030040741 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 93.63999938964844 | NORMAL | NULL | NULL |
| 2022-05-01 07:00:00 | 88.81 | GOOD | 0.1370999962091446 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 94.45999908447266 | NORMAL | NULL | NULL |
| 2022-05-01 07:30:00 | 90.26 | GOOD | 0.2590999901294708 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 93.63999938964844 | NORMAL | NULL | NULL |

### Bottom 10 Records

| Timestamp | HealthIndex | HealthZone | FusedZ | RunID | EquipID | RawHealthIndex | QualityFlag | Confidence | ConfidenceFactors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-24 23:59:00 | 70.91 | WATCH | 1.7269999980926514 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 71.66000366210938 | NORMAL | 0.473 | NULL |
| 2023-10-24 22:59:00 | 70.59 | WATCH | 3.708899974822998 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 18.989999771118164 | NORMAL | 0.473 | NULL |
| 2023-10-24 21:59:00 | 92.7 | GOOD | -0.34369999170303345 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 93.01000213623047 | NORMAL | 0.473 | NULL |
| 2023-10-24 20:59:00 | 92.57 | GOOD | -0.33799999952316284 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 93.05000305175781 | NORMAL | 0.473 | NULL |
| 2023-10-24 19:59:00 | 92.37 | GOOD | -0.18619999289512634 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 94.13999938964844 | NORMAL | 0.473 | NULL |
| 2023-10-24 18:59:00 | 91.61 | GOOD | 0.21240000426769257 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 93.95999908447266 | NORMAL | 0.473 | NULL |
| 2023-10-24 17:59:00 | 90.6 | GOOD | 0.9790999889373779 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 86.12000274658203 | NORMAL | 0.473 | NULL |
| 2023-10-24 16:59:00 | 92.52 | GOOD | 0.5314000248908997 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 91.38999938964844 | NORMAL | 0.473 | NULL |
| 2023-10-24 15:59:00 | 93.0 | GOOD | 0.3905999958515167 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 92.62999725341797 | NORMAL | 0.473 | NULL |
| 2023-10-24 14:59:00 | 93.16 | GOOD | 0.5394999980926514 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 91.30999755859375 | NORMAL | 0.473 | NULL |

---


## dbo.ACM_HealthZoneByPeriod

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_HistorianData

**Primary Key:** DataID  
**Row Count:** 0  

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
**Row Count:** 0  

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

---


## dbo.ACM_OMRTimeline

**Primary Key:** No primary key  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| OMR_Z | float | YES | 53 | — |
| OMR_Weight | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_OMR_Diagnostics

**Primary Key:** DiagnosticID  
**Row Count:** 5  

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
| 17 | 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | pca | 5 | 129 | 790 | 2.1736166741523597 | NULL | NULL |
| 18 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | pca | 5 | 129 | 790 | 2.5312222532300086 | NULL | NULL |
| 19 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | pca | 5 | 129 | 783 | 3.1322536090858555 | NULL | NULL |
| 20 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | pca | 5 | 131 | 790 | 3.052277817462197 | NULL | NULL |
| 22 | 3879d1b4-a397-4926-84af-46dfd2dece43 | 2621 | pca | 5 | 97 | 135 | 1.1439257643908496 | NULL | NULL |

---


## dbo.ACM_PCA_Loadings

**Primary Key:** RecordID  
**Row Count:** 16,800  
**Date Range:** 2025-12-27 12:24:27 to 2025-12-30 17:25:37  

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
| 6391 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | power_29_avg_med | power_29_avg_med | 0.06831576237583573 | 2025-12-27 12:24:35 |
| 6392 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | power_29_max_med | power_29_max_med | 0.06858443869676183 | 2025-12-27 12:24:35 |
| 6393 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | power_29_min_med | power_29_min_med | 0.062356150753892184 | 2025-12-27 12:24:35 |
| 6394 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | power_29_std_med | power_29_std_med | 0.028957640547677185 | 2025-12-27 12:24:35 |
| 6395 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | power_30_avg_med | power_30_avg_med | 0.06837116839296829 | 2025-12-27 12:24:35 |
| 6396 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | power_30_max_med | power_30_max_med | 0.06852783423113995 | 2025-12-27 12:24:35 |
| 6397 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | power_30_min_med | power_30_min_med | 0.06282140873221853 | 2025-12-27 12:24:35 |
| 6398 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | power_30_std_med | power_30_std_med | 0.030036866889153916 | 2025-12-27 12:24:35 |
| 6399 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | reactive_power_27_avg_med | reactive_power_27_avg_med | 0.012448600651993196 | 2025-12-27 12:24:35 |
| 6400 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 1 | 1 | reactive_power_27_max_med | reactive_power_27_max_med | 0.008175612047554359 | 2025-12-27 12:24:35 |

### Bottom 10 Records

| RecordID | RunID | EquipID | EntryDateTime | ComponentNo | ComponentID | Sensor | FeatureName | Loading | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28045 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | -0.025844238321615465 | 2025-12-30 17:25:40 |
| 28044 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | -0.04293502996132773 | 2025-12-30 17:25:40 |
| 28043 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | 0.12637994550874987 | 2025-12-30 17:25:40 |
| 28042 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | 0.09063256148035317 | 2025-12-30 17:25:40 |
| 28041 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_rz | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_rz | 0.06519638397703639 | 2025-12-30 17:25:40 |
| 28040 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.06T31_1FD Fan Inlet Temperature_rz | DEMO.SIM.06T31_1FD Fan Inlet Temperature_rz | 0.1516849414565426 | 2025-12-30 17:25:40 |
| 28039 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.06I03_1FD Fan Motor Current_rz | DEMO.SIM.06I03_1FD Fan Motor Current_rz | -0.014192596369373678 | 2025-12-30 17:25:40 |
| 28038 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_rz | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_rz | -0.0113936312963658 | 2025-12-30 17:25:40 |
| 28037 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.06G31_1FD Fan Damper Position_rz | DEMO.SIM.06G31_1FD Fan Damper Position_rz | -0.006950985796384126 | 2025-12-30 17:25:40 |
| 28036 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | 5 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_kurt | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_kurt | 0.09094167091741338 | 2025-12-30 17:25:40 |

---


## dbo.ACM_PCA_Metrics

**Primary Key:** ID  
**Row Count:** 7  
**Date Range:** 2025-12-27 06:23:37 to 2025-12-31 04:22:47  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| ComponentIndex | int | NO | 10 | — |
| ExplainedVariance | float | YES | 53 | — |
| CumulativeVariance | float | YES | 53 | — |
| Eigenvalue | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | ComponentIndex | ExplainedVariance | CumulativeVariance | Eigenvalue | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 21 | 61a6978e-47c5-494f-a88a-e4e71e9989a1 | 5003 | 0 | 0.6257169506655602 | NULL | 790.0 | 2025-12-27 06:23:37 |
| 22 | 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | 0 | 0.6512559664854273 | NULL | 790.0 | 2025-12-27 06:23:37 |
| 23 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 0 | 0.6137611389342381 | NULL | 783.0 | 2025-12-27 06:23:41 |
| 24 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 0 | 0.6192699319519162 | NULL | 790.0 | 2025-12-27 06:24:05 |
| 26 | 3879d1b4-a397-4926-84af-46dfd2dece43 | 2621 | 0 | 0.620571841873642 | NULL | 135.0 | 2025-12-29 09:56:24 |
| 45 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 0 | 0.8142418399863025 | NULL | 72.0 | 2025-12-30 11:53:16 |
| 46 | b434fc8c-641b-4922-9a00-4d9cd738daeb | 5092 | 0 | 0.7634928196221591 | NULL | 566.0 | 2025-12-31 04:22:47 |

---


## dbo.ACM_PCA_Models

**Primary Key:** RecordID  
**Row Count:** 6  
**Date Range:** 2025-12-27 12:24:27 to 2025-12-30 17:25:37  

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
| 19 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2025-12-27 12:24:27 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.25799938190307775, 0.14664933955056267, 0.09290923805461906, 0.06274860737035344, 0.0534545720... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-05-04 13:20:00 |
| 20 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 2025-12-27 12:24:50 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.2295510693116072, 0.18291527391123294, 0.09950270530972463, 0.060852496195278584, 0.0528954059... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-05-01 03:00:00 |
| 21 | 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | 2025-12-27 12:24:53 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.2442275170160447, 0.1536438321746414, 0.12445376366607512, 0.0791424872106899, 0.0497883664179... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-08-08 06:10:00 |
| 22 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | {"SPE_P95_train": 0.0, "T2_P95_train": 0.0} | [0.2621816258131944, 0.1682218852236984, 0.08260738571662195, 0.05916325324800247, 0.047095781950... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2022-10-13 07:30:00 |
| 24 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 2025-12-29 15:29:23 | 5 | {"SPE_P95_train": 4.361407279968262, "T2_P95_train": 2.9234392642974854} | [0.23090401380606393, 0.147278324942666, 0.10871588813773764, 0.07383341888381037, 0.059840196103... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-20 23:59:00 |
| 37 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2025-12-30 17:25:37 | 5 | {"SPE_P95_train": 7.335094451904297, "T2_P95_train": 5.0390543937683105} | [0.2929481030226121, 0.19991159410877093, 0.17927615193848404, 0.08261513369686664, 0.05949085721... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-16 12:00:00 |

---


## dbo.ACM_RUL

**Primary Key:** EquipID, RunID  
**Row Count:** 1  
**Date Range:** 2026-01-06 17:25:07 to 2026-01-06 17:25:07  

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
| RUL_Status | nvarchar | YES | 50 | — |
| MaturityState | nvarchar | YES | 50 | — |
| MeanRUL | float | YES | 53 | — |
| StdRUL | float | YES | 53 | — |
| MTTF_Hours | float | YES | 53 | — |
| FailureProbability | float | YES | 53 | — |
| CurrentHealth | float | YES | 53 | — |
| HealthLevel | nvarchar | YES | 50 | — |
| TrendSlope | float | YES | 53 | — |
| DataQuality | nvarchar | YES | 50 | — |
| ForecastStd | float | YES | 53 | — |
| TopSensor1Contribution | float | YES | 53 | — |
| TopSensor2Contribution | float | YES | 53 | — |
| TopSensor3Contribution | float | YES | 53 | — |

### Top 10 Records

| EquipID | RunID | RUL_Hours | P10_LowerBound | P50_Median | P90_UpperBound | Confidence | FailureTime | Method | NumSimulations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.3 | 2026-01-06 17:25:07 | Multipath | 1000 |

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


## dbo.ACM_RUL_LearningState

**Primary Key:** EquipID  
**Row Count:** 0  

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
**Row Count:** 7  
**Date Range:** 2025-12-27 06:30:48 to 2025-12-31 04:27:18  

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
| 80 | 5000 | 2025-12-27 06:30:48 | Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 81 | 5003 | 2025-12-27 06:30:48 | Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 82 | 5013 | 2025-12-27 06:30:49 | Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 83 | 5010 | 2025-12-27 06:31:16 | Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 85 | 2621 | 2025-12-29 09:58:13 | Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 101 | 1 | 2025-12-30 11:54:22 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 102 | 5092 | 2025-12-31 04:27:18 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |

---


## dbo.ACM_RegimeDefinitions

**Primary Key:** ID  
**Row Count:** 18  
**Date Range:** 2025-12-27 06:26:10 to 2025-12-30 11:53:49  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| EquipID | int | NO | 10 | — |
| RegimeVersion | int | NO | 10 | — |
| RegimeID | int | NO | 10 | — |
| RegimeName | nvarchar | NO | 100 | — |
| CentroidJSON | nvarchar | NO | -1 | — |
| FeatureColumns | nvarchar | NO | -1 | — |
| DataPointCount | int | NO | 10 | — |
| SilhouetteScore | float | YES | 53 | — |
| MaturityState | nvarchar | YES | 30 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |
| CreatedByRunID | nvarchar | YES | 50 | — |

### Top 10 Records

| ID | EquipID | RegimeVersion | RegimeID | RegimeName | CentroidJSON | FeatureColumns | DataPointCount | SilhouetteScore | MaturityState |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 5000 | 1 | 0 | Regime_0 | [0.2917396499704652, 0.32645193580037013, 0.23177174208669057, 0.15406001063294203, 0.31166137728... | [] | 110 | NULL | LEARNING |
| 8 | 5000 | 1 | 1 | Regime_1 | [-1.7084235273699726, -1.8895139457615273, -1.3986691433282301, -0.8540498652417241, -1.845860685... | [] | 19 | NULL | LEARNING |
| 9 | 5003 | 1 | 0 | Regime_0 | [-0.11307527014937278, -0.09937626516279638, -0.008573813168239295, -0.0499809403618687, 0.242925... | [] | 113 | NULL | LEARNING |
| 10 | 5003 | 1 | 1 | Regime_1 | [0.7855003223251116, 0.5101972222328186, 0.1256801780536965, 0.029965903055574502, -1.76903894177... | [] | 16 | NULL | LEARNING |
| 11 | 5013 | 1 | 0 | Regime_0 | [1.1013239100160759, 1.1718048602619064, 0.8186670980728983, 1.0979557428811375, 1.10139900694751... | [] | 34 | NULL | LEARNING |
| 12 | 5013 | 1 | 1 | Regime_1 | [-0.9571371021435846, -0.9844831097757256, -0.8117585231961642, -0.6222996969694032, -0.958811484... | [] | 31 | NULL | LEARNING |
| 13 | 5013 | 1 | 2 | Regime_2 | [-0.375968825384818, -0.20761686028435133, -0.48440716797667466, 0.20114904745430245, -0.37044659... | [] | 40 | NULL | LEARNING |
| 14 | 5013 | 1 | 3 | Regime_3 | [-1.1050834447308313, -1.3831270757268688, -0.8264039728810878, -1.2487332983746555, -1.116041944... | [] | 12 | NULL | LEARNING |
| 15 | 5013 | 1 | 4 | Regime_4 | [1.6785599012737689, 1.1883648633956911, 2.3031346247248035, -1.1685142986800359, 1.6729365891736... | [] | 12 | NULL | LEARNING |
| 16 | 5010 | 1 | 0 | Regime_0 | [0.07395956263262563, 0.1925636953961924, -0.16600490057603617, 0.29864275364500104, 0.1032667960... | [] | 40 | NULL | LEARNING |

### Bottom 10 Records

| ID | EquipID | RegimeVersion | RegimeID | RegimeName | CentroidJSON | FeatureColumns | DataPointCount | SilhouetteScore | MaturityState |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 85 | 1 | 1 | 3 | Regime_3 | [0.9514788412130797, 1.7539466444689493, 0.8030543934840422, 1.9671966043802407, 0.48471171795748... | [] | 3 | NULL | LEARNING |
| 84 | 1 | 1 | 2 | Regime_2 | [0.5559887934672206, -0.033120644719977124, 1.658762097358703, 0.7775367761913099, 2.280816731954... | [] | 5 | NULL | LEARNING |
| 83 | 1 | 1 | 1 | Regime_1 | [-0.4983442430226308, -0.9738212022016633, 0.7319680051983527, 0.7464334538522759, -0.02090235954... | [] | 14 | NULL | LEARNING |
| 82 | 1 | 1 | 0 | Regime_0 | [0.02172093175827189, 0.3489755274331633, -0.8088183971795629, -0.7363048372512803, -0.5125311545... | [] | 27 | NULL | LEARNING |
| 22 | 2621 | 1 | 1 | Regime_1 | [-1.4995371645354496, -1.4851165219989357, -1.5217098194573606, -1.508949532724765, -1.4733449468... | [] | 19 | NULL | LEARNING |
| 21 | 2621 | 1 | 0 | Regime_0 | [0.3660379364530714, 0.35871560537181846, 0.3779250016881423, 0.38112820480673537, 0.362687013776... | [] | 78 | NULL | LEARNING |
| 18 | 5010 | 1 | 2 | Regime_2 | [1.7732375475106303, 1.67223554036834, 1.9386688279254096, 1.365046832009745, 1.772301350611371, ... | [] | 23 | NULL | LEARNING |
| 17 | 5010 | 1 | 1 | Regime_1 | [-0.6510032323638281, -0.6971255694616888, -0.5508004426956177, -0.6790673614924555, -0.666808662... | [] | 68 | NULL | LEARNING |
| 16 | 5010 | 1 | 0 | Regime_0 | [0.07395956263262563, 0.1925636953961924, -0.16600490057603617, 0.29864275364500104, 0.1032667960... | [] | 40 | NULL | LEARNING |
| 15 | 5013 | 1 | 4 | Regime_4 | [1.6785599012737689, 1.1883648633956911, 2.3031346247248035, -1.1685142986800359, 1.6729365891736... | [] | 12 | NULL | LEARNING |

---


## dbo.ACM_RegimeDwellStats

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_RegimeOccupancy

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| RegimeLabel | nvarchar | NO | 50 | — |
| DwellTimeHours | float | NO | 53 | — |
| DwellFraction | float | NO | 53 | — |
| EntryCount | int | YES | 10 | — |
| AvgDwellMinutes | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

---


## dbo.ACM_RegimePromotionLog

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| RegimeLabel | nvarchar | NO | 50 | — |
| FromState | nvarchar | NO | 30 | — |
| ToState | nvarchar | NO | 30 | — |
| Reason | nvarchar | YES | 200 | — |
| DataPointsAtPromotion | int | YES | 10 | — |
| PromotedAt | datetime2 | NO | — | (getutcdate()) |

---


## dbo.ACM_RegimeStability

**Primary Key:** No primary key  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| MetricName | nvarchar | NO | 100 | — |
| MetricValue | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

---


## dbo.ACM_RegimeState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 6  
**Date Range:** 2025-12-27 06:26:08 to 2025-12-30 11:53:47  

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
| 1 | 1 | 4 | [[0.02172093175827189, 0.3489755274331633, -0.8088183971795629, -0.7363048372512803, -0.512531154... | [] | [] | [] | [] | 0 | 0.3630909957253809 |
| 2621 | 1 | 2 | [[0.3660379364530714, 0.35871560537181846, 0.3779250016881423, 0.38112820480673537, 0.36268701377... | [] | [] | [] | [] | 0 | 0.5865870150681485 |
| 5000 | 1 | 2 | [[0.2917396499704652, 0.32645193580037013, 0.23177174208669057, 0.15406001063294203, 0.3116613772... | [] | [] | [] | [] | 0 | 0.48468588826125614 |
| 5003 | 1 | 2 | [[-0.11307527014937278, -0.09937626516279638, -0.008573813168239295, -0.0499809403618687, 0.24292... | [] | [] | [] | [] | 0 | 0.4821212740575696 |
| 5010 | 1 | 3 | [[0.07395956263262563, 0.1925636953961924, -0.16600490057603617, 0.29864275364500104, 0.103266796... | [] | [] | [] | [] | 0 | 0.559156687397022 |
| 5013 | 1 | 5 | [[1.1013239100160759, 1.1718048602619064, 0.8186670980728983, 1.0979557428811375, 1.1013990069475... | [] | [] | [] | [] | 0 | 0.5042883062311074 |

---


## dbo.ACM_RegimeStats

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_RegimeTimeline

**Primary Key:** No primary key  
**Row Count:** 664  
**Date Range:** 2022-05-01 03:00:00 to 2023-10-24 23:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| RegimeLabel | nvarchar | NO | 50 | — |
| RegimeState | nvarchar | NO | 50 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| AssignmentConfidence | float | YES | 53 | — |
| RegimeVersion | int | YES | 10 | — |

### Top 10 Records

| Timestamp | RegimeLabel | RegimeState | RunID | EquipID | AssignmentConfidence | RegimeVersion |
| --- | --- | --- | --- | --- | --- | --- |
| 2022-05-01 03:00:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 03:30:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 04:00:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 04:30:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 05:00:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 05:30:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 06:00:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 06:30:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 07:00:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |
| 2022-05-01 07:30:00 | 0 | unknown | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL |

### Bottom 10 Records

| Timestamp | RegimeLabel | RegimeState | RunID | EquipID | AssignmentConfidence | RegimeVersion |
| --- | --- | --- | --- | --- | --- | --- |
| 2023-10-24 23:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.69 | NULL |
| 2023-10-24 22:59:00 | -1 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.0 | NULL |
| 2023-10-24 21:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.736 | NULL |
| 2023-10-24 20:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.384 | NULL |
| 2023-10-24 19:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.695 | NULL |
| 2023-10-24 18:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.691 | NULL |
| 2023-10-24 17:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.818 | NULL |
| 2023-10-24 16:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.405 | NULL |
| 2023-10-24 15:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.946 | NULL |
| 2023-10-24 14:59:00 | 0 | unknown | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 0.479 | NULL |

---


## dbo.ACM_RegimeTransitions

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| FromRegime | nvarchar | NO | 50 | — |
| ToRegime | nvarchar | NO | 50 | — |
| TransitionCount | int | NO | 10 | — |
| TransitionProbability | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

---


## dbo.ACM_Regime_Episodes

**Primary Key:** Id  
**Row Count:** 6  

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
| 381 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | NULL | NULL | NULL |
| 382 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | NULL | NULL | NULL |
| 383 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | NULL | NULL | NULL |
| 384 | 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | NULL | NULL | NULL |
| 385 | 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | NULL | NULL | NULL |
| 386 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | NULL | NULL | NULL |

---


## dbo.ACM_RunLogs

**Primary Key:** LogID  
**Row Count:** 10,437  
**Date Range:** 2025-12-11 11:55:49 to 2025-12-11 12:34:46  

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
**Row Count:** 0  

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

---


## dbo.ACM_RunMetrics

**Primary Key:** RunID, EquipID, MetricName  
**Row Count:** 588  
**Date Range:** 2025-12-11 17:29:21 to 2025-12-31 09:54:43  

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
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.n_samples.ar1_z | 129.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.n_samples.gmm_z | 129.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.n_samples.iforest_z | 129.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.n_samples.omr_z | 129.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.n_samples.pca_spe_z | 129.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.n_samples.pca_t2_z | 129.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.quality.ar1_z | 0.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.quality.gmm_z | 0.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.quality.iforest_z | 0.0 | 2025-12-27 11:57:22 |
| 03097745-09ca-4250-b0c2-1d64c948247f | 5000 | fusion.quality.omr_z | 0.0 | 2025-12-27 11:57:22 |

### Bottom 10 Records

| RunID | EquipID | MetricName | MetricValue | Timestamp |
| --- | --- | --- | --- | --- |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.weight.pca_t2_z | 0.04964539007092199 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.weight.pca_spe_z | 0.18156028368794325 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.weight.omr_z | 0.11205673758865248 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.weight.mhal_z | 0.18156028368794325 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.weight.iforest_z | 0.18156028368794325 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.weight.gmm_z | 0.11205673758865248 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.weight.ar1_z | 0.18156028368794325 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.quality.pca_t2_z | 0.0 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.quality.pca_spe_z | 0.0 | 2025-12-11 17:37:27 |
| fc7c8606-e4e1-4edc-afc2-6d635ac06552 | 8632 | fusion.quality.omr_z | 0.0 | 2025-12-11 17:37:27 |

---


## dbo.ACM_RunTimers

**Primary Key:** TimerID  
**Row Count:** 205,327  
**Date Range:** 2025-12-15 12:26:06 to 2025-12-25 11:36:57  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| TimerID | int | NO | 10 | — |
| RunID | varchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| BatchNum | int | YES | 10 | ((0)) |
| Section | varchar | NO | 100 | — |
| DurationSeconds | float | NO | 53 | — |
| CreatedAt | datetime | YES | — | (getutcdate()) |

### Top 10 Records

| TimerID | RunID | EquipID | BatchNum | Section | DurationSeconds | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | startup | 0.08555479999631643 | 2025-12-15 12:26:06 |
| 2 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | load_data | 0.3112183999910485 | 2025-12-15 12:26:06 |
| 3 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | baseline.seed | 0.33316979999653995 | 2025-12-15 12:26:06 |
| 4 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | data.guardrails.data_quality | 0.21355120002408512 | 2025-12-15 12:26:06 |
| 5 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | data.guardrails | 0.22407970001222566 | 2025-12-15 12:26:06 |
| 6 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | features.compute_fill_values | 0.00831909998669289 | 2025-12-15 12:26:06 |
| 7 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | features.polars_convert | 0.031281900010071695 | 2025-12-15 12:26:06 |
| 8 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | features.compute_train | 0.1149751000048127 | 2025-12-15 12:26:06 |
| 9 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | features.compute_score | 0.10255989999859594 | 2025-12-15 12:26:06 |
| 10 | 4c33d3c5-6632-45ed-a550-1d4c44cde5ad | 5010 | 7 | features.normalize | 0.1632044999860227 | 2025-12-15 12:26:06 |

### Bottom 10 Records

| TimerID | RunID | EquipID | BatchNum | Section | DurationSeconds | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 205327 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | sql.culprits | 0.027524499979335815 | 2025-12-25 11:36:57 |
| 205326 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | sql.run_stats | 0.1674388999817893 | 2025-12-25 11:36:57 |
| 205325 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | sql.pca | 0.05573459999868646 | 2025-12-25 11:36:57 |
| 205324 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | sql.regimes | 0.02631319995271042 | 2025-12-25 11:36:57 |
| 205323 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | sql.events | 0.026703699957579374 | 2025-12-25 11:36:57 |
| 205322 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | sql.drift | 0.026692899991758168 | 2025-12-25 11:36:57 |
| 205321 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | persist | 170.6630495999707 | 2025-12-25 11:36:57 |
| 205320 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | outputs.forecasting | 145.17328049999196 | 2025-12-25 11:36:57 |
| 205319 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | outputs.comprehensive_analytics | 8.797092400025576 | 2025-12-25 11:36:57 |
| 205318 | cb4188a3-3038-41e3-91d2-7cbd1d15077e | 1 | 65 | persist.sensor_correlation | 11.758547600009479 | 2025-12-25 11:36:57 |

---


## dbo.ACM_Run_Stats

**Primary Key:** RecordID  
**Row Count:** 6  
**Date Range:** 2022-04-27 03:00:00 to 2023-10-15 00:00:00  

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
| 19 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2022-04-30 13:20:00 | 2022-05-16 03:36:59 | 129 | 129 | 81 | 100.0 | NULL |
| 20 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 2022-04-27 03:00:00 | 2022-05-12 17:16:59 | 129 | 129 | 81 | 100.0 | NULL |
| 21 | 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | 2022-08-04 06:10:00 | 2022-08-19 20:26:59 | 129 | 129 | 81 | 100.0 | NULL |
| 22 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2022-10-09 08:40:00 | 2022-10-24 22:56:59 | 131 | 131 | 81 | 100.0 | NULL |
| 24 | 3879D1B4-A397-4926-84AF-46DFD2DECE43 | 2621 | 2023-10-15 00:00:00 | 2023-10-15 23:59:59 | 97 | 97 | 16 | 0.0 | NULL |
| 37 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 49 | 49 | 9 | 0.0 | NULL |

---


## dbo.ACM_Runs

**Primary Key:** RunID  
**Row Count:** 41  
**Date Range:** 2025-12-20 04:25:52 to 2025-12-31 04:19:00  

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
| 9AA20F0F-D9E1-49C3-B8B9-01A7ED9F33E6 | 8634 | ELECTRIC_MOTOR | 2025-12-29 09:23:32 | 2025-12-29 09:23:38 | 5 |  | 0 | 0 | 0 |
| C8EA4C4A-0B58-40C8-B284-01B64D26D884 | 5017 | WFA_TURBINE_17 | 2025-12-27 06:56:37 | 2025-12-27 06:56:49 | 12 |  | 0 | 0 | 0 |
| 61B2669A-EDD8-4C3B-9CE4-04888D086216 | 8632 | WIND_TURBINE | 2025-12-20 04:26:22 | 2025-12-20 04:26:22 | 0 |  | 0 | 0 | 0 |
| F585E91B-D893-447D-BBA7-09A0BC0E4655 | 2621 | GAS_TURBINE | 2025-12-29 09:19:14 | 2025-12-29 09:19:19 | 4 |  | 0 | 0 | 0 |
| B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | WFA_TURBINE_13 | 2025-12-27 05:49:00 | 2025-12-27 06:54:39 | 3938 |  | 129 | 3920 | 2 |
| 485841A7-C79A-4389-BD70-1240D29D6990 | 8632 | WIND_TURBINE | 2025-12-20 04:26:16 | 2025-12-20 04:26:16 | 0 |  | 0 | 0 | 0 |
| 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | WFA_TURBINE_0 | 2025-12-27 05:49:00 | 2025-12-27 06:55:05 | 3964 |  | 129 | 3955 | 2 |
| E8AA0E82-6593-478D-93F3-229B9CC8B00C | 8634 | ELECTRIC_MOTOR | 2025-12-29 09:20:26 | 2025-12-29 09:20:32 | 5 |  | 0 | 0 | 0 |
| 4D28948A-7100-4FD1-B757-2B6A6E412D8E | 8632 | WIND_TURBINE | 2025-12-20 04:26:02 | 2025-12-20 04:26:02 | 0 |  | 0 | 0 | 0 |
| A7156CE9-07EB-40F4-8158-2CB83F0D0DB0 | 5014 | WFA_TURBINE_14 | 2025-12-27 06:56:20 | 2025-12-27 06:56:34 | 13 |  | 0 | 0 | 0 |

### Bottom 10 Records

| RunID | EquipID | EquipName | StartedAt | CompletedAt | DurationSeconds | ConfigSignature | TrainRowCount | ScoreRowCount | EpisodeCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3E488317-AA29-4081-A473-FBE267A091ED | 5025 | WFA_TURBINE_25 | 2025-12-27 06:57:36 | 2025-12-27 06:57:48 | 12 |  | 0 | 0 | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | FD_FAN | 2025-12-30 11:52:40 | 2025-12-30 11:55:43 | 182 |  | 49 | 361 | 0 |
| E8008D4A-B739-4894-AA3A-F62215AE4135 | 5000 | WFA_TURBINE_0 | 2025-12-27 06:56:02 | 2025-12-27 06:56:13 | 10 |  | 0 | 0 | 0 |
| 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | WFA_TURBINE_3 | 2025-12-27 05:49:00 | 2025-12-27 06:55:03 | 3962 |  | 129 | 3953 | 1 |
| B70A0738-778C-4D95-8B23-E4659705F948 | 8634 | ELECTRIC_MOTOR | 2025-12-29 09:23:09 | 2025-12-29 09:23:14 | 5 |  | 0 | 0 | 0 |
| 84B7FE0C-C270-43BE-B356-E19D55235709 | 8634 | ELECTRIC_MOTOR | 2025-12-29 09:21:12 | 2025-12-29 09:21:18 | 5 |  | 0 | 0 | 0 |
| DC5383BF-4484-4982-8F3A-E127BA879A11 | 5026 | WFA_TURBINE_26 | 2025-12-27 06:58:28 | 2025-12-27 06:58:41 | 13 |  | 0 | 0 | 0 |
| 2E79C7A6-8DAB-4567-B187-E0D9AB4C54D2 | 5026 | WFA_TURBINE_26 | 2025-12-27 06:57:51 | 2025-12-27 06:58:06 | 14 |  | 0 | 0 | 0 |
| 7C486880-C614-43BF-A888-CCD3CA946950 | 8634 | ELECTRIC_MOTOR | 2025-12-29 09:22:22 | 2025-12-29 09:22:28 | 5 |  | 0 | 0 | 0 |
| 817832A1-CD7B-41D0-B6EC-C9A1109CCD97 | 8632 | WIND_TURBINE | 2025-12-20 04:26:09 | 2025-12-20 04:26:10 | 0 |  | 0 | 0 | 0 |

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
**Row Count:** 0  

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

---


## dbo.ACM_Scores_Wide

**Primary Key:** No primary key  
**Row Count:** 713  
**Date Range:** 2022-04-04 14:30:00 to 2023-10-24 23:59:00  

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
| 2022-04-04 14:30:00 | -2.413301944732666 | 2.0664875507354736 | 0.4877432584762573 | NULL | 0.419490247964859 | -0.640396237373352 | -3.0342392921447754 | NULL | NULL |
| 2022-04-04 14:40:00 | 2.9462926387786865 | 0.04240603372454643 | 6.014198303222656 | NULL | 0.5109601616859436 | -1.3246656656265259 | -3.001863956451416 | NULL | NULL |
| 2022-04-04 14:50:00 | 1.0582184791564941 | 2.089611053466797 | 0.05122394487261772 | NULL | -0.2443850338459015 | -2.7268149852752686 | -2.9789443016052246 | NULL | NULL |
| 2022-04-04 15:00:00 | 0.2972220182418823 | 0.04182754456996918 | -0.4534777104854584 | NULL | -1.1747092008590698 | -3.011247396469116 | -3.0108444690704346 | NULL | NULL |
| 2022-04-04 15:10:00 | -0.017323821783065796 | 0.19453704357147217 | -0.7022605538368225 | NULL | -2.0146162509918213 | -3.404099225997925 | -2.9723474979400635 | NULL | NULL |
| 2022-04-04 15:20:00 | -0.29460084438323975 | -0.6528130769729614 | -0.4216887056827545 | NULL | -1.5631227493286133 | -3.6008846759796143 | -2.8821616172790527 | NULL | NULL |
| 2022-04-04 15:30:00 | -0.7246575951576233 | -0.5466448068618774 | -0.292610228061676 | NULL | -1.4785016775131226 | -3.4564030170440674 | -2.7478280067443848 | NULL | NULL |
| 2022-04-04 15:40:00 | -0.8044577240943909 | -0.5591035485267639 | -0.4418603479862213 | NULL | -2.028472661972046 | -3.5824882984161377 | -2.568763494491577 | NULL | NULL |
| 2022-04-04 15:50:00 | -0.3712281286716461 | 0.054440587759017944 | -0.8104121685028076 | NULL | -1.7923343181610107 | -3.2086567878723145 | -2.3778817653656006 | NULL | NULL |
| 2022-04-04 16:00:00 | -0.674490749835968 | -0.19072821736335754 | -0.7860829830169678 | NULL | -2.2372334003448486 | -3.5251948833465576 | -2.159472703933716 | NULL | NULL |

### Bottom 10 Records

| Timestamp | ar1_z | pca_spe_z | pca_t2_z | mhal_z | iforest_z | gmm_z | cusum_z | drift_z | hst_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10-24 23:59:00 | -0.23500417172908783 | 5.563698768615723 | 3.5257716178894043 | NULL | 1.5593429803848267 | 2.5691657066345215 | -0.05972636118531227 | NULL | NULL |
| 2023-10-24 22:59:00 | 8.574509620666504 | 8.298726081848145 | 7.780318737030029 | NULL | 2.360262632369995 | 4.552582740783691 | 0.268197238445282 | NULL | NULL |
| 2023-10-24 21:59:00 | -0.9049322605133057 | -0.4378678798675537 | 0.46806710958480835 | NULL | -0.31458336114883423 | 0.6521518230438232 | 0.5488665699958801 | NULL | NULL |
| 2023-10-24 20:59:00 | -0.6937774419784546 | -0.7798349261283875 | 0.8994808793067932 | NULL | -0.20543353259563446 | 0.674490749835968 | 0.5561730861663818 | NULL | NULL |
| 2023-10-24 19:59:00 | -0.6597496271133423 | -0.33801692724227905 | 0.8171771764755249 | NULL | 0.044906966388225555 | 0.8844259977340698 | 0.5939002633094788 | NULL | NULL |
| 2023-10-24 18:59:00 | 0.5398170351982117 | 1.0199830532073975 | 0.6364930868148804 | NULL | 0.022665956988930702 | 1.0307503938674927 | 0.674490749835968 | NULL | NULL |
| 2023-10-24 17:59:00 | 1.1793360710144043 | 3.164802074432373 | 1.2235976457595825 | NULL | 0.2667878568172455 | 2.9557011127471924 | 0.8005485534667969 | NULL | NULL |
| 2023-10-24 16:59:00 | 1.9758613109588623 | 2.398425817489624 | -0.27437955141067505 | NULL | 0.05168084800243378 | 1.0594416856765747 | 0.9501566886901855 | NULL | NULL |
| 2023-10-24 15:59:00 | 0.49610164761543274 | 1.4501100778579712 | 0.39431512355804443 | NULL | 0.8380511999130249 | 1.0594176054000854 | 1.0537689924240112 | NULL | NULL |
| 2023-10-24 14:59:00 | 2.372408866882324 | 0.8543003797531128 | 0.9252175688743591 | NULL | 0.8405818939208984 | 1.5324875116348267 | 1.1381797790527344 | NULL | NULL |

---


## dbo.ACM_SeasonalPatterns

**Primary Key:** ID  
**Row Count:** 7  
**Date Range:** 2025-12-29 15:29:07 to 2025-12-29 15:29:07  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| EquipID | int | NO | 10 | — |
| SensorName | nvarchar | NO | 200 | — |
| PatternType | nvarchar | NO | 30 | — |
| PeriodHours | float | NO | 53 | — |
| Amplitude | float | NO | 53 | — |
| PhaseShift | float | YES | 53 | — |
| Confidence | float | YES | 53 | — |
| DetectedAt | datetime2 | NO | — | (getutcdate()) |
| DetectedByRunID | nvarchar | YES | 50 | — |

### Top 10 Records

| ID | EquipID | SensorName | PatternType | PeriodHours | Amplitude | PhaseShift | Confidence | DetectedAt | DetectedByRunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2621 | ACTTBTEMP1 | DAILY | 24.0 | 0.461 | 20.0 | 0.2513 | 2025-12-29 15:29:07 | 3879d1b4-a397-4926-84af-46dfd2dece43 |
| 9 | 2621 | B1RADVIBY | DAILY | 24.0 | 0.0277 | 13.0 | 0.1748 | 2025-12-29 15:29:07 | 3879d1b4-a397-4926-84af-46dfd2dece43 |
| 10 | 2621 | B1TEMP1 | DAILY | 24.0 | 0.415 | 18.0 | 0.2013 | 2025-12-29 15:29:07 | 3879d1b4-a397-4926-84af-46dfd2dece43 |
| 11 | 2621 | B2RADVIBX | DAILY | 24.0 | 0.0148 | 12.0 | 0.2311 | 2025-12-29 15:29:07 | 3879d1b4-a397-4926-84af-46dfd2dece43 |
| 12 | 2621 | B2TEMP1 | DAILY | 24.0 | 0.4152 | 18.0 | 0.2805 | 2025-12-29 15:29:07 | 3879d1b4-a397-4926-84af-46dfd2dece43 |
| 13 | 2621 | INACTTBTEMP1 | DAILY | 24.0 | 0.4683 | 18.0 | 0.2939 | 2025-12-29 15:29:07 | 3879d1b4-a397-4926-84af-46dfd2dece43 |
| 14 | 2621 | LOTEMP1 | DAILY | 24.0 | 0.4475 | 18.0 | 0.2961 | 2025-12-29 15:29:07 | 3879d1b4-a397-4926-84af-46dfd2dece43 |

---


## dbo.ACM_SensorAnomalyByPeriod

**Primary Key:** No primary key  
**Row Count:** 0  

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

---


## dbo.ACM_SensorCorrelations

**Primary Key:** ID  
**Row Count:** 1,256,079  
**Date Range:** 2025-12-27 06:51:41 to 2025-12-30 11:54:42  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| Sensor1 | nvarchar | NO | 200 | — |
| Sensor2 | nvarchar | NO | 200 | — |
| Correlation | float | NO | 53 | — |
| CorrelationType | nvarchar | YES | 20 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | Sensor1 | Sensor2 | Correlation | CorrelationType | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 204130 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | power_29_avg_med | 1.0 | pearson | 2025-12-27 06:51:41 |
| 204131 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | power_29_max_med | 0.9674391717984256 | pearson | 2025-12-27 06:51:41 |
| 204132 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | power_29_min_med | 0.9497565801895512 | pearson | 2025-12-27 06:51:41 |
| 204133 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | power_29_std_med | 0.29966259609352025 | pearson | 2025-12-27 06:51:41 |
| 204134 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | power_30_avg_med | 0.9999863741395671 | pearson | 2025-12-27 06:51:41 |
| 204135 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | power_30_max_med | 0.9678224624585376 | pearson | 2025-12-27 06:51:41 |
| 204136 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | power_30_min_med | 0.9537013495900214 | pearson | 2025-12-27 06:51:41 |
| 204137 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | power_30_std_med | 0.31402673067084613 | pearson | 2025-12-27 06:51:41 |
| 204138 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | reactive_power_27_avg_med | 0.014227662754664094 | pearson | 2025-12-27 06:51:41 |
| 204139 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | power_29_avg_med | reactive_power_27_max_med | -0.052589802773888285 | pearson | 2025-12-27 06:51:41 |

### Bottom 10 Records

| ID | RunID | EquipID | Sensor1 | Sensor2 | Correlation | CorrelationType | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1499862 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 1.0 | pearson | 2025-12-30 11:54:42 |
| 1499861 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 0.9735456822433837 | pearson | 2025-12-30 11:54:42 |
| 1499860 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | 1.0 | pearson | 2025-12-30 11:54:42 |
| 1499859 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 0.4034144263361732 | pearson | 2025-12-30 11:54:42 |
| 1499858 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | 0.3703260056545505 | pearson | 2025-12-30 11:54:42 |
| 1499857 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | 1.0 | pearson | 2025-12-30 11:54:42 |
| 1499856 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 0.5281243822401246 | pearson | 2025-12-30 11:54:42 |
| 1499855 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | 0.5421085756330508 | pearson | 2025-12-30 11:54:42 |
| 1499854 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | 0.7869908057569098 | pearson | 2025-12-30 11:54:42 |
| 1499853 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | 1.0 | pearson | 2025-12-30 11:54:42 |

---


## dbo.ACM_SensorDefects

**Primary Key:** No primary key  
**Row Count:** 42  

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
| Density Anomaly (GMM) | Density | HIGH | 25 | 19.38 | 5.8758 | 1.1387 | 2.4642 | 1 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |
| Time-Series Anomaly (AR1) | Time-Series | HIGH | 15 | 11.63 | 3.2005 | 0.8874 | 0.4137 | 0 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |
| Multivariate Outlier (PCA-T2) | Multivariate | MEDIUM | 10 | 7.75 | 4.1524 | 0.2907 | 2.4166 | 1 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |
| Baseline Consistency (OMR) | Baseline | MEDIUM | 10 | 7.75 | 4.9189 | 0.8705 | 1.3146 | 0 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |
| Rare State (IsolationForest) | Rare | MEDIUM | 9 | 6.98 | 3.0753 | 0.8783 | 3.0753 | 1 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |
| Correlation Break (PCA-SPE) | Correlation | MEDIUM | 7 | 5.43 | 4.4153 | 0.2395 | 0.0 | 0 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |
| cusum_z | cusum_z | LOW | 3 | 2.33 | 2.3213 | 0.7381 | 0.4114 | 0 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |
| cusum_z | cusum_z | CRITICAL | 37 | 28.68 | 5.8957 | 1.3 | 1.808 | 0 | 03097745-09CA-4250-B0C2-1D64C948247F |
| Density Anomaly (GMM) | Density | CRITICAL | 26 | 20.16 | 5.2244 | 1.1091 | 0.2832 | 0 | 03097745-09CA-4250-B0C2-1D64C948247F |
| Time-Series Anomaly (AR1) | Time-Series | MEDIUM | 11 | 8.53 | 6.1021 | 0.8901 | 0.3206 | 0 | 03097745-09CA-4250-B0C2-1D64C948247F |

### Bottom 10 Records

| DetectorType | DetectorFamily | Severity | ViolationCount | ViolationPct | MaxZ | AvgZ | CurrentZ | ActiveDefect | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rare State (IsolationForest) | Rare | HIGH | 5 | 10.2 | 4.1969 | 0.9509 | 0.313 | 0 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA |
| Multivariate Outlier (PCA-T2) | Multivariate | HIGH | 5 | 10.2 | 10.0 | 1.1861 | 0.447 | 0 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA |
| Correlation Break (PCA-SPE) | Correlation | HIGH | 7 | 14.29 | 9.8541 | 1.4417 | 0.7502 | 0 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA |
| Time-Series Anomaly (AR1) | Time-Series | HIGH | 8 | 16.33 | 7.287 | 1.2087 | 0.1729 | 0 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA |
| cusum_z | cusum_z | MEDIUM | 4 | 8.16 | 2.475 | 0.8785 | 1.2029 | 0 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA |
| Density Anomaly (GMM) | Density | LOW | 0 | 0.0 | 1.621 | 0.7835 | 0.6972 | 0 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA |
| Baseline Consistency (OMR) | Baseline | LOW | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA |
| Density Anomaly (GMM) | Density | CRITICAL | 36 | 27.91 | 6.3973 | 1.207 | 4.5797 | 1 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 |
| Time-Series Anomaly (AR1) | Time-Series | HIGH | 24 | 18.6 | 10.0 | 1.436 | 10.0 | 1 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 |
| Baseline Consistency (OMR) | Baseline | HIGH | 20 | 15.5 | 3.7355 | 1.0283 | 3.7355 | 1 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 |

---


## dbo.ACM_SensorForecast

**Primary Key:** RunID, EquipID, Timestamp, SensorName  
**Row Count:** 1,512  
**Date Range:** 2023-10-17 13:00:00 to 2023-10-24 12:00:00  

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
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 33.51924010129852 | 27.67619903157607 | 39.362281171020975 | 2.981143402919619 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 0.6047289278158209 | 0.30876688564085863 | 0.9006909699907831 | 0.15100104192600114 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.06I03_1FD Fan Motor Current | 35.924343062159984 | 34.12762677124949 | 37.72105935307048 | 0.9166919851584148 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 35.754961484621646 | 31.8113015140542 | 39.69862145518909 | 2.0120714135548194 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | 60.17718740985005 | 57.99636631362736 | 62.358008506072736 | 1.1126638246034102 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 48.69209398851978 | 47.43829773638725 | 49.94589024065231 | 0.6396919653737395 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.06T34_1FD Fan Outlet Termperature | 25.34647414366318 | 22.3832316868635 | 28.30971660046286 | 1.5118583963263674 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | 288.30027538642776 | 242.93094067616522 | 333.6696100966903 | 23.147619750133952 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 13:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 292.2381573876215 | 245.303851652739 | 339.17246312250404 | 23.9460743545319 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-17 14:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 33.44255658088258 | 27.599515511160128 | 39.285597650605034 | 2.981143402919619 | ExponentialSmoothing | 0 |

### Bottom 10 Records

| RunID | EquipID | Timestamp | SensorName | ForecastValue | CiLower | CiUpper | ForecastStd | Method | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 214.30494280646548 | 167.37063707158296 | 261.239248541348 | 23.9460743545319 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | 251.2464246336294 | 205.87708992336687 | 296.61575934389197 | 23.147619750133952 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.06T34_1FD Fan Outlet Termperature | -4.132845660176724 | -7.096088116976404 | -1.1696032033770436 | 1.5118583963263674 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 40.49600762385402 | 39.24221137172149 | 41.74980387598655 | 0.6396919653737395 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | 57.89493377777709 | 55.714112681554404 | 60.07575487399978 | 1.1126638246034102 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 17.506307942772214 | 13.562647972204768 | 21.44996791333966 | 2.0120714135548194 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.06I03_1FD Fan Motor Current | 34.13564507144364 | 32.338928780533145 | 35.932361362354136 | 0.9166919851584148 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1.567989285899884 | 1.2720272437249218 | 1.8639513280748463 | 0.15100104192600114 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 12:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 20.713092191837056 | 14.870051122114603 | 26.55613326155951 | 2.981143402919619 | ExponentialSmoothing | 0 |
| 3EAA50E5-84D4-4A1B-9E3D-F69E2A51F7AA | 1 | 2023-10-24 11:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 214.77160876204124 | 167.83730302715873 | 261.70591449692375 | 23.9460743545319 | ExponentialSmoothing | 0 |

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
**Row Count:** 0  

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

---


## dbo.ACM_SensorHotspots

**Primary Key:** No primary key  
**Row Count:** 125  
**Date Range:** 2022-05-01 03:00:00 to 2023-10-24 17:59:00  

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
| sensor_31_std | 2022-05-01 03:00:00 | 2022-05-03 19:00:00 | 9.9492 | 9.9492 | 0.2769 | -0.2769 | 190.39999389648438 | 0.800000011920929 | 5.933332920074463 |
| sensor_31_min | 2022-05-01 03:00:00 | 2022-05-03 19:00:00 | 6.0836 | -6.0836 | 2.0713 | 2.0713 | -652.9000244140625 | -18.200000762939453 | -179.41006469726562 |
| sensor_31_max | 2022-05-01 03:00:00 | 2022-05-03 19:00:00 | 6.0301 | 6.0301 | 1.7289 | 1.7289 | 333.5 | -13.600000381469727 | -153.1186065673828 |
| sensor_9_avg | 2022-05-01 03:00:00 | 2022-05-03 19:00:00 | 3.4371 | -3.4371 | 1.6898 | -1.6898 | 35.0 | 39.0 | 42.86821746826172 |
| power_29_max | 2022-05-01 03:30:00 | 2022-05-03 19:00:00 | 3.192 | -3.192 | 0.5082 | 0.5082 | 0.0804390236735344 | 0.9756097793579102 | 0.8526586890220642 |
| sensor_42_avg | 2022-05-01 15:30:00 | 2022-05-03 19:00:00 | 2.9842 | 2.9842 | 0.6525 | 0.6525 | 126.69999694824219 | 106.69999694824219 | 101.10310363769531 |
| sensor_26_avg | 2022-05-01 22:30:00 | 2022-05-03 19:00:00 | 11.2699 | 11.2699 | 0.088 | -0.088 | 50.099998474121094 | 50.0 | 50.00077438354492 |
| sensor_18_std | 2022-05-02 18:00:00 | 2022-05-03 19:00:00 | 9.2059 | 9.2059 | 0.4146 | -0.4146 | 736.5999755859375 | 8.100000381469727 | 39.49380111694336 |
| sensor_52_std | 2022-05-02 18:00:00 | 2022-05-03 19:00:00 | 9.1174 | 9.1174 | 0.4826 | -0.4826 | 6.599999904632568 | 0.0 | 0.3317829668521881 |
| sensor_5_std | 2022-05-02 18:00:00 | 2022-05-03 19:00:00 | 8.3672 | 8.3672 | 0.4746 | -0.4746 | 35.5 | 0.0 | 1.9054263830184937 |

### Bottom 10 Records

| SensorName | MaxTimestamp | LatestTimestamp | MaxAbsZ | MaxSignedZ | LatestAbsZ | LatestSignedZ | ValueAtPeak | LatestValue | TrainMean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DWATT | 2023-10-24 17:59:00 | 2023-10-24 23:59:00 | 3.1681 | 3.1681 | 0.1483 | -0.1483 | 0.111175537109375 | 0.077606201171875 | 0.07910722494125366 |
| TURBAXDISP1 | 2023-10-24 00:59:00 | 2023-10-24 23:59:00 | 7.1592 | -7.1592 | 0.5266 | -0.5266 | -25.919198989868164 | -24.969924926757812 | -24.89455223083496 |
| B2VIB2 | 2023-10-23 22:59:00 | 2023-10-24 23:59:00 | 4.4798 | 4.4798 | 0.4399 | -0.4399 | 0.0033319001086056232 | 0.0016957520274445415 | 0.0018420600099489093 |
| B1RADVIBY | 2023-10-23 12:59:00 | 2023-10-24 23:59:00 | 4.2169 | 4.2169 | 0.4224 | 0.4224 | 0.23450851440429688 | 0.11920928955078125 | 0.10637401044368744 |
| TURBAXDISP2 | 2023-10-23 08:59:00 | 2023-10-24 23:59:00 | 4.6009 | -4.6009 | 0.4155 | -0.4155 | -25.384187698364258 | -24.448131561279297 | -24.355199813842773 |
| B1VIB1 | 2023-10-23 08:59:00 | 2023-10-24 23:59:00 | 2.2999 | -2.2999 | 0.4732 | 0.4732 | 0.0029444689862430096 | 0.003418325912207365 | 0.0033374608028680086 |
| B2VIB1 | 2023-10-22 05:59:00 | 2023-10-24 23:59:00 | 4.1003 | 4.1003 | 0.4817 | 0.4817 | 0.003415345912799239 | 0.002110003959387541 | 0.0019362291786819696 |
| B2RADVIBX | 2023-10-22 03:59:00 | 2023-10-24 23:59:00 | 2.9861 | 2.9861 | 0.1184 | 0.1184 | 0.14190673828125 | 0.09140968322753906 | 0.08932536095380783 |
| B1RADVIBX | 2023-10-21 15:59:00 | 2023-10-24 23:59:00 | 5.0788 | 5.0788 | 0.496 | -0.496 | 0.3186225891113281 | 0.09121894836425781 | 0.11145060509443283 |
| B2RADVIBY | 2023-10-21 10:59:00 | 2023-10-24 23:59:00 | 5.0117 | 5.0117 | 0.0242 | 0.0242 | 0.2307891845703125 | 0.08788108825683594 | 0.08718893676996231 |

---


## dbo.ACM_SensorNormalized_TS

**Primary Key:** ID  
**Row Count:** 43,951  
**Date Range:** 2022-05-01 03:00:00 to 2023-10-24 23:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | nvarchar | NO | 50 | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| SensorName | nvarchar | NO | 200 | — |
| RawValue | float | YES | 53 | — |
| NormalizedValue | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | Timestamp | SensorName | RawValue | NormalizedValue | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 502609 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 13:20:00 | power_29_avg | NULL | 0.9634146094322205 | 2025-12-27 06:54:02 |
| 502610 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 13:50:00 | power_29_avg | NULL | 0.9756097793579102 | 2025-12-27 06:54:02 |
| 502611 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 14:20:00 | power_29_avg | NULL | 0.9756097793579102 | 2025-12-27 06:54:02 |
| 502612 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 14:50:00 | power_29_avg | NULL | 0.9756097793579102 | 2025-12-27 06:54:02 |
| 502613 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 15:20:00 | power_29_avg | NULL | 0.9754633903503418 | 2025-12-27 06:54:02 |
| 502614 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 15:50:00 | power_29_avg | NULL | 0.9756097793579102 | 2025-12-27 06:54:02 |
| 502615 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 16:20:00 | power_29_avg | NULL | 0.9755122065544128 | 2025-12-27 06:54:02 |
| 502616 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 16:50:00 | power_29_avg | NULL | 0.8078048825263977 | 2025-12-27 06:54:02 |
| 502617 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 17:20:00 | power_29_avg | NULL | 0.8672682642936707 | 2025-12-27 06:54:02 |
| 502618 | b1795ae8-0a34-4097-ac71-0c3cbff9592d | 5013 | 2022-05-04 17:50:00 | power_29_avg | NULL | 0.9620487689971924 | 2025-12-27 06:54:02 |

### Bottom 10 Records

| ID | RunID | EquipID | Timestamp | SensorName | RawValue | NormalizedValue | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 556774 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 12:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 300.19000244140625 | 2025-12-30 11:54:44 |
| 556773 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 11:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 271.1199951171875 | 2025-12-30 11:54:44 |
| 556772 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 11:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 298.4700012207031 | 2025-12-30 11:54:44 |
| 556771 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 10:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 301.1700134277344 | 2025-12-30 11:54:44 |
| 556770 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 10:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 299.6499938964844 | 2025-12-30 11:54:44 |
| 556769 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 09:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 292.8900146484375 | 2025-12-30 11:54:44 |
| 556768 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 09:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 289.1300048828125 | 2025-12-30 11:54:44 |
| 556767 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 08:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 297.7799987792969 | 2025-12-30 11:54:44 |
| 556766 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 08:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 314.9700012207031 | 2025-12-30 11:54:44 |
| 556765 | 3eaa50e5-84d4-4a1b-9e3d-f69e2a51f7aa | 1 | 2023-10-17 07:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 299.260009765625 | 2025-12-30 11:54:44 |

---


## dbo.ACM_SensorRanking

**Primary Key:** No primary key  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorType | nvarchar | NO | 50 | — |
| RankPosition | int | NO | 10 | — |
| ContributionPct | float | NO | 53 | — |
| ZScore | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

---


## dbo.ACM_SinceWhen

**Primary Key:** No primary key  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| AlertZone | nvarchar | NO | 50 | — |
| DurationHours | float | NO | 53 | — |
| StartTimestamp | datetime2 | NO | — | — |
| RecordCount | int | NO | 10 | — |

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
**Row Count:** 0  

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

---


## dbo.ACM_ThresholdMetadata

**Primary Key:** ThresholdID  
**Row Count:** 0  

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
**Row Count:** 33  
**Date Range:** 2025-12-27 06:26:15 to 2025-12-31 04:24:25  

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
| gmm_model | 1 | 1 | 2025-12-30 11:53:55 | {"n_components": 3, "covariance_type": "diag", "bic": 17037772.56, "aic": 17036951.51, "lower_bou... | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 7952 bytes> |
| ar1_params | 1 | 1 | 2025-12-30 11:53:50 | {"n_sensors": 72, "mean_autocorr": 12.2818, "mean_residual_std": 0.838, "params_count": 144} | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 5677 bytes> |
| iforest_model | 1 | 1 | 2025-12-30 11:53:55 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 457321 bytes> |
| pca_model | 1 | 1 | 2025-12-30 11:53:50 | {"n_components": 5, "variance_ratio_sum": 0.8142, "variance_ratio_first_component": 0.2929, "vari... | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 4511 bytes> |
| gmm_model | 2621 | 1 | 2025-12-29 09:57:23 | {"n_components": 3, "covariance_type": "diag", "bic": 20184426.23, "aic": 20182335.57, "lower_bou... | {"train_rows": 97, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP1... | NULL | <binary 14084 bytes> |
| ar1_params | 2621 | 1 | 2025-12-29 09:57:17 | {"n_sensors": 135, "mean_autocorr": 3.9754, "mean_residual_std": 0.5605, "params_count": 270} | {"train_rows": 97, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP1... | NULL | <binary 6518 bytes> |
| iforest_model | 2621 | 1 | 2025-12-29 09:57:22 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 97, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP1... | NULL | <binary 707401 bytes> |
| omr_model | 2621 | 1 | 2025-12-29 09:57:23 | NULL | {"train_rows": 97, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP1... | NULL | <binary 15213 bytes> |
| pca_model | 2621 | 1 | 2025-12-29 09:57:17 | {"n_components": 5, "variance_ratio_sum": 0.6206, "variance_ratio_first_component": 0.2309, "vari... | {"train_rows": 97, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEMP1... | NULL | <binary 7519 bytes> |
| ar1_params | 5000 | 1 | 2025-12-27 06:26:15 | {"n_sensors": 790, "mean_autocorr": 655489383.9208, "mean_residual_std": 448915724.5588, "params_... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 43780 bytes> |

### Bottom 10 Records

| ModelType | EquipID | Version | EntryDateTime | ParamsJSON | StatsJSON | RunID | ModelBytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ar1_params | 5092 | 1 | 2025-12-31 04:24:19 | {"n_sensors": 566, "mean_autocorr": 1459464706535.6265, "mean_residual_std": 2032131359443.5737, ... | {"train_rows": 49, "train_sensors": ["power_29_avg_med", "power_29_min_med", "power_29_std_med", ... | NULL | <binary 30678 bytes> |
| gmm_model | 5092 | 1 | 2025-12-31 04:24:25 | {"n_components": 3, "covariance_type": "diag", "bic": 3.8181564656701496e+32, "aic": 3.8181564656... | {"train_rows": 49, "train_sensors": ["power_29_avg_med", "power_29_min_med", "power_29_std_med", ... | NULL | <binary 55376 bytes> |
| iforest_model | 5092 | 1 | 2025-12-31 04:24:25 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 49, "train_sensors": ["power_29_avg_med", "power_29_min_med", "power_29_std_med", ... | NULL | <binary 839145 bytes> |
| pca_model | 5092 | 1 | 2025-12-31 04:24:20 | {"n_components": 5, "variance_ratio_sum": 0.7635, "variance_ratio_first_component": 0.4062, "vari... | {"train_rows": 49, "train_sensors": ["power_29_avg_med", "power_29_min_med", "power_29_std_med", ... | NULL | <binary 28223 bytes> |
| ar1_params | 5013 | 1 | 2025-12-27 06:26:18 | {"n_sensors": 783, "mean_autocorr": 603917848.7389, "mean_residual_std": 432069368.1003, "params_... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 43389 bytes> |
| gmm_model | 5013 | 1 | 2025-12-27 06:26:23 | {"n_components": 3, "covariance_type": "diag", "bic": 5.398619277253911e+25, "aic": 5.39861927725... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 76235 bytes> |
| iforest_model | 5013 | 1 | 2025-12-27 06:26:23 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 1445625 bytes> |
| omr_model | 5013 | 1 | 2025-12-27 06:26:26 | NULL | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 83205 bytes> |
| pca_model | 5013 | 1 | 2025-12-27 06:26:18 | {"n_components": 5, "variance_ratio_sum": 0.6138, "variance_ratio_first_component": 0.258, "varia... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 38639 bytes> |
| ar1_params | 5010 | 1 | 2025-12-27 06:26:43 | {"n_sensors": 790, "mean_autocorr": 88959372.0627, "mean_residual_std": 63261084.5647, "params_co... | {"train_rows": 131, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 43780 bytes> |

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

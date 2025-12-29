# ACM Comprehensive Database Schema Reference

_Generated automatically on 2025-12-29 11:23:01_

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
| dbo.ACM_ActiveModels | 11 | 4 | ID |
| dbo.ACM_AdaptiveConfig | 13 | 17 | ConfigID |
| dbo.ACM_AlertAge | 6 | 0 | — |
| dbo.ACM_Anomaly_Events | 6 | 6 | Id |
| dbo.ACM_AssetProfiles | 11 | 4 | ID |
| dbo.ACM_BaselineBuffer | 7 | 246,950 | Id |
| dbo.ACM_CalibrationSummary | 10 | 0 | ID |
| dbo.ACM_ColdstartState | 17 | 15 | EquipID, Stage |
| dbo.ACM_Config | 7 | 292 | ConfigID |
| dbo.ACM_ConfigHistory | 9 | 19 | ID |
| dbo.ACM_ContributionCurrent | 5 | 0 | — |
| dbo.ACM_ContributionTimeline | 7 | 0 | ID |
| dbo.ACM_DailyFusedProfile | 9 | 0 | ID |
| dbo.ACM_DataContractValidation | 10 | 24 | ID |
| dbo.ACM_DataQuality | 24 | 1,880 | — |
| dbo.ACM_DefectSummary | 12 | 0 | — |
| dbo.ACM_DefectTimeline | 10 | 0 | — |
| dbo.ACM_DetectorCorrelation | 7 | 196 | ID |
| dbo.ACM_DetectorForecast_TS | 10 | 0 | RunID, EquipID, DetectorName, Timestamp |
| dbo.ACM_DriftController | 10 | 0 | ID |
| dbo.ACM_DriftEvents | 2 | 0 | — |
| dbo.ACM_DriftSeries | 7 | 0 | ID |
| dbo.ACM_EnhancedFailureProbability_TS | 11 | 0 | RunID, EquipID, Timestamp, ForecastHorizon_Hours |
| dbo.ACM_EnhancedMaintenanceRecommendation | 13 | 0 | RunID, EquipID |
| dbo.ACM_EpisodeCulprits | 9 | 31 | ID |
| dbo.ACM_EpisodeDiagnostics | 16 | 6 | ID |
| dbo.ACM_EpisodeMetrics | 10 | 0 | — |
| dbo.ACM_Episodes | 8 | 26 | — |
| dbo.ACM_EpisodesQC | 10 | 0 | RecordID |
| dbo.ACM_FailureCausation | 12 | 0 | RunID, EquipID, Detector |
| dbo.ACM_FailureForecast | 9 | 21,168 | EquipID, RunID, Timestamp |
| dbo.ACM_FailureForecast_TS | 7 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_FailureHazard_TS | 8 | 0 | EquipID, RunID, Timestamp |
| dbo.ACM_FeatureDropLog | 8 | 411 | ID |
| dbo.ACM_ForecastState | 12 | 0 | EquipID, StateVersion |
| dbo.ACM_ForecastingState | 13 | 1 | EquipID, StateVersion |
| dbo.ACM_FusionQualityReport | 9 | 0 | — |
| dbo.ACM_HealthDistributionOverTime | 12 | 0 | — |
| dbo.ACM_HealthForecast | 10 | 21,168 | EquipID, RunID, Timestamp |
| dbo.ACM_HealthForecast_Continuous | 8 | 0 | EquipID, Timestamp, SourceRunID |
| dbo.ACM_HealthForecast_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_HealthHistogram | 5 | 0 | — |
| dbo.ACM_HealthTimeline | 8 | 3,639 | — |
| dbo.ACM_HealthZoneByPeriod | 9 | 0 | — |
| dbo.ACM_HistorianData | 7 | 0 | DataID |
| dbo.ACM_MaintenanceRecommendation | 8 | 0 | RunID, EquipID |
| dbo.ACM_OMRContributionsLong | 8 | 0 | — |
| dbo.ACM_OMRTimeline | 6 | 0 | — |
| dbo.ACM_OMR_Diagnostics | 15 | 4 | DiagnosticID |
| dbo.ACM_PCA_Loadings | 10 | 15,765 | RecordID |
| dbo.ACM_PCA_Metrics | 8 | 4 | ID |
| dbo.ACM_PCA_Models | 12 | 4 | RecordID |
| dbo.ACM_RUL | 18 | 21 | EquipID, RunID |
| dbo.ACM_RUL_Attribution | 9 | 0 | RunID, EquipID, FailureTime, SensorName |
| dbo.ACM_RUL_LearningState | 19 | 0 | EquipID |
| dbo.ACM_RUL_Summary | 15 | 0 | RunID, EquipID |
| dbo.ACM_RUL_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_RecommendedActions | 6 | 0 | RunID, EquipID, Action |
| dbo.ACM_RefitRequests | 10 | 4 | RequestID |
| dbo.ACM_RegimeDefinitions | 12 | 12 | ID |
| dbo.ACM_RegimeDwellStats | 8 | 0 | — |
| dbo.ACM_RegimeOccupancy | 9 | 0 | ID |
| dbo.ACM_RegimePromotionLog | 9 | 0 | ID |
| dbo.ACM_RegimeStability | 4 | 0 | — |
| dbo.ACM_RegimeState | 15 | 5 | EquipID, StateVersion |
| dbo.ACM_RegimeStats | 8 | 0 | — |
| dbo.ACM_RegimeTimeline | 5 | 3,639 | — |
| dbo.ACM_RegimeTransitions | 8 | 0 | ID |
| dbo.ACM_Regime_Episodes | 6 | 6 | Id |
| dbo.ACM_RunLogs | 25 | 31,438 | LogID |
| dbo.ACM_RunMetadata | 12 | 0 | RunMetadataID |
| dbo.ACM_RunMetrics | 5 | 1,311 | RunID, EquipID, MetricName |
| dbo.ACM_RunTimers | 7 | 205,327 | TimerID |
| dbo.ACM_Run_Stats | 13 | 4 | RecordID |
| dbo.ACM_Runs | 19 | 57 | RunID |
| dbo.ACM_SchemaVersion | 5 | 2 | VersionID |
| dbo.ACM_Scores_Long | 9 | 0 | Id |
| dbo.ACM_Scores_Wide | 15 | 3,639 | — |
| dbo.ACM_SeasonalPatterns | 10 | 0 | ID |
| dbo.ACM_SensorAnomalyByPeriod | 11 | 0 | — |
| dbo.ACM_SensorCorrelations | 8 | 1,244,271 | ID |
| dbo.ACM_SensorDefects | 11 | 204 | — |
| dbo.ACM_SensorForecast | 11 | 0 | RunID, EquipID, Timestamp, SensorName |
| dbo.ACM_SensorForecast_TS | 10 | 0 | RunID, EquipID, SensorName, Timestamp |
| dbo.ACM_SensorHotspotTimeline | 9 | 0 | — |
| dbo.ACM_SensorHotspots | 18 | 650 | — |
| dbo.ACM_SensorNormalized_TS | 8 | 41,958 | ID |
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
| dbo.ModelRegistry | 8 | 174 | ModelType, EquipID, Version |
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
**Row Count:** 4  

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

---


## dbo.ACM_AdaptiveConfig

**Primary Key:** ConfigID  
**Row Count:** 17  
**Date Range:** 2025-12-04 10:46:47 to 2025-12-27 11:57:52  

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
| 23 | 5010 | fused_warn_z | 0.7621059417724609 | 0.0 | 999999.0 | True | 131 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 22 | 5010 | fused_alert_z | 1.5242118835449219 | 0.0 | 999999.0 | True | 131 | 0.0 | quantile_0.997: Auto-calculated from 131 accumulated samples |
| 21 | 5013 | fused_warn_z | 0.5874592661857605 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 20 | 5013 | fused_alert_z | 1.174918532371521 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated from 129 accumulated samples |
| 19 | 5003 | fused_warn_z | 0.7344083189964294 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 18 | 5003 | fused_alert_z | 1.4688166379928589 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated from 129 accumulated samples |
| 17 | 5000 | fused_warn_z | 0.7372899055480957 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 16 | 5000 | fused_alert_z | 1.4745798110961914 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated from 129 accumulated samples |
| 9 | NULL | auto_tune_data_threshold | 10000.0 | 5000.0 | 50000.0 | False | NULL | NULL | Expert tuning - Auto-tuning trigger |
| 8 | NULL | blend_tau_hours | 12.0 | 6.0 | 48.0 | False | NULL | NULL | Expert tuning - Warm-start alpha blending |

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

### Top 10 Records

| Id | RunID | EquipID | StartTime | EndTime | Severity |
| --- | --- | --- | --- | --- | --- |
| 381 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2022-05-05 19:20:00 | 2022-05-06 01:50:00 | info |
| 382 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2022-05-06 21:50:00 | 2022-05-07 03:50:00 | info |
| 383 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 2022-05-03 13:30:00 | 2022-05-03 17:00:00 | info |
| 384 | 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | 2022-08-09 01:10:00 | 2022-08-09 02:10:00 | info |
| 385 | 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | 2022-08-09 21:40:00 | 2022-08-10 14:10:00 | info |
| 386 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2022-10-13 13:00:00 | 2022-10-13 20:00:00 | info |

---


## dbo.ACM_AssetProfiles

**Primary Key:** ID  
**Row Count:** 4  
**Date Range:** 2025-12-27 12:24:04 to 2025-12-27 12:24:55  

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

---


## dbo.ACM_BaselineBuffer

**Primary Key:** Id  
**Row Count:** 246,950  
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
| 14017520 | 5092 | 2022-05-14 00:00:00 | wind_speed_4_avg | 4.300000190734863 | NULL | 2025-12-13 11:27:58 |
| 14017519 | 5092 | 2022-05-14 00:00:00 | wind_speed_3_std | 0.6000000238418579 | NULL | 2025-12-13 11:27:58 |
| 14017518 | 5092 | 2022-05-14 00:00:00 | wind_speed_3_min | 1.2999999523162842 | NULL | 2025-12-13 11:27:58 |
| 14017517 | 5092 | 2022-05-14 00:00:00 | wind_speed_3_max | 7.5 | NULL | 2025-12-13 11:27:58 |
| 14017516 | 5092 | 2022-05-14 00:00:00 | wind_speed_3_avg | 4.699999809265137 | NULL | 2025-12-13 11:27:58 |
| 14017515 | 5092 | 2022-05-14 00:00:00 | sensor_9_avg | 42.0 | NULL | 2025-12-13 11:27:58 |
| 14017514 | 5092 | 2022-05-14 00:00:00 | sensor_8_avg | 97.0 | NULL | 2025-12-13 11:27:58 |
| 14017513 | 5092 | 2022-05-14 00:00:00 | sensor_7_avg | 40.0 | NULL | 2025-12-13 11:27:58 |
| 14017512 | 5092 | 2022-05-14 00:00:00 | sensor_6_avg | 26.0 | NULL | 2025-12-13 11:27:58 |
| 14017511 | 5092 | 2022-05-14 00:00:00 | sensor_5_std | 0.10000000149011612 | NULL | 2025-12-13 11:27:58 |

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
**Row Count:** 15  
**Date Range:** 2025-12-13 06:00:34 to 2025-12-29 05:27:03  

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
| 1 | score | COMPLETE | 2 | 2025-12-29 05:27:03 | 2025-12-29 05:27:59 | 2025-12-29 05:27:59 | 482 | 200 | 2023-10-15 00:00:00 |
| 2621 | score | COMPLETE | 2 | 2025-12-27 06:58:20 | 2025-12-27 06:58:45 | 2025-12-27 06:58:45 | 482 | 200 | 2023-10-15 00:00:00 |
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
| 8634 | score | FAILED | 13 | 2025-12-20 04:25:53 | 2025-12-20 04:26:24 | 2025-12-20 04:25:53 | 241 | 200 | 2024-01-01 00:00:00 |
| 8632 | score | IN_PROGRESS | 15 | 2025-12-20 04:25:52 | 2025-12-20 04:26:22 | 2025-12-20 04:25:52 | 1719 | 200 | 2024-01-01 00:00:00 |
| 5092 | score | COMPLETE | 1 | 2025-12-13 06:00:34 | 2025-12-13 06:00:34 | 2025-12-13 06:00:34 | 241 | 200 | 2022-04-04 02:30:00 |
| 5026 | score | COMPLETE | 6 | 2025-12-27 06:57:52 | 2025-12-27 06:58:39 | 2025-12-27 06:58:39 | 1126 | 200 | 2022-10-12 10:20:00 |
| 5025 | score | COMPLETE | 6 | 2025-12-27 06:57:36 | 2025-12-27 06:58:24 | 2025-12-27 06:58:24 | 1126 | 200 | 2022-05-23 06:50:00 |
| 5024 | score | COMPLETE | 6 | 2025-12-27 06:57:06 | 2025-12-27 06:57:49 | 2025-12-27 06:57:49 | 1126 | 200 | 2022-04-24 15:00:00 |
| 5022 | score | COMPLETE | 6 | 2025-12-27 06:56:40 | 2025-12-27 06:57:23 | 2025-12-27 06:57:23 | 1126 | 200 | 2022-08-12 09:50:00 |
| 5017 | score | COMPLETE | 6 | 2025-12-27 06:56:37 | 2025-12-27 06:57:24 | 2025-12-27 06:57:24 | 1126 | 200 | 2022-10-31 15:20:00 |
| 5014 | score | COMPLETE | 6 | 2025-12-27 06:56:20 | 2025-12-27 06:57:11 | 2025-12-27 06:57:11 | 1126 | 200 | 2022-03-03 14:00:00 |
| 5013 | score | COMPLETE | 3 | 2025-12-27 05:49:00 | 2025-12-27 05:49:09 | 2025-12-27 05:49:09 | 563 | 200 | 2022-04-30 13:20:00 |

---


## dbo.ACM_Config

**Primary Key:** ConfigID  
**Row Count:** 292  
**Date Range:** 2025-12-09 12:47:06 to 2025-12-29 05:26:56  

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
| 492 | 0 | data.train_csv | data/FD_FAN_BASELINE_DATA.csv | string | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 493 | 0 | data.score_csv | data/FD_FAN_BATCH_DATA.csv | string | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 494 | 0 | data.data_dir | data | string | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 495 | 0 | data.timestamp_col | EntryDateTime | string | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 496 | 0 | data.tag_columns | [] | list | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 497 | 0 | data.sampling_secs | 1800 | int | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 498 | 0 | data.max_rows | 100000 | int | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 499 | 0 | features.window | 16 | int | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 500 | 0 | features.fft_bands | [0.0, 0.1, 0.3, 0.5] | list | 2025-12-17 04:35:46 | B19cl3pc\bhadk |
| 501 | 0 | features.top_k_tags | 5 | int | 2025-12-17 04:35:46 | B19cl3pc\bhadk |

### Bottom 10 Records

| ConfigID | EquipID | ParamPath | ParamValue | ValueType | UpdatedAt | UpdatedBy |
| --- | --- | --- | --- | --- | --- | --- |
| 856 | 1 | runtime.tick_minutes | 1009410 | int | 2025-12-29 05:26:56 | sql_batch_runner |
| 854 | 5026 | runtime.tick_minutes | 21600 | int | 2025-12-27 06:57:44 | sql_batch_runner |
| 853 | 5025 | runtime.tick_minutes | 21993 | int | 2025-12-27 06:57:30 | sql_batch_runner |
| 852 | 5024 | runtime.tick_minutes | 22110 | int | 2025-12-27 06:57:00 | sql_batch_runner |
| 851 | 5017 | runtime.tick_minutes | 22141 | int | 2025-12-27 06:56:31 | sql_batch_runner |
| 850 | 5014 | runtime.tick_minutes | 21784 | int | 2025-12-27 06:56:14 | sql_batch_runner |
| 849 | 5003 | runtime.tick_minutes | 22345 | int | 2025-12-27 05:48:40 | sql_batch_runner |
| 848 | 5013 | runtime.tick_minutes | 22457 | int | 2025-12-27 05:48:40 | sql_batch_runner |
| 839 | 0 | runtime.baseline.refresh_interval_batches | 10 | int | 2025-12-17 04:35:47 | B19cl3pc\bhadk |
| 838 | 0 | runtime.baseline.max_points | 100000 | int | 2025-12-17 04:35:47 | B19cl3pc\bhadk |

---


## dbo.ACM_ConfigHistory

**Primary Key:** ID  
**Row Count:** 19  
**Date Range:** 2025-12-11 17:29:31 to 2025-12-27 12:01:16  

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
| 12 | 2025-12-13 10:48:49 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.05e+30 exceeds 1e28 (critical instability) | c296502f-7b4f-453d-8d54-042f53be1c48 |
| 13 | 2025-12-13 10:49:15 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 4.24e+42 exceeds 1e28 (critical instability) | 0b511082-7139-4f9a-a0b9-f02ea7e86848 |
| 14 | 2025-12-13 11:07:00 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.05e+30 exceeds 1e28 (critical instability) | 89a2af2a-7231-445e-95ff-21240b4014e2 |
| 15 | 2025-12-13 11:07:22 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 3.63e+42 exceeds 1e28 (critical instability) | 690134bf-21fa-4890-b397-0870707812ac |
| 16 | 2025-12-13 11:27:45 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.40e+44 exceeds 1e28 (critical instability) | 452e6b7c-eb60-4714-8280-2ff1d920d350 |
| 17 | 2025-12-13 11:30:49 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.05e+30 exceeds 1e28 (critical instability) | c51c247d-6b86-4cf4-93c4-9547cd9e4069 |
| 18 | 2025-12-13 11:31:19 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 4.42e+28 exceeds 1e28 (critical instability) | d51fe222-b378-420b-8a64-854d5a0f645b |
| 19 | 2025-12-13 11:31:57 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.27e+29 exceeds 1e28 (critical instability) | 88401cdd-1bc5-4ecb-a340-bc98624cdc94 |
| 20 | 2025-12-13 11:34:57 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.01e+29 exceeds 1e28 (critical instability) | 0b61b75e-e221-49b0-8b08-bc26c0acc3d9 |

### Bottom 10 Records

| ID | Timestamp | EquipID | ParameterPath | OldValue | NewValue | ChangedBy | ChangeReason | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13495 | 2025-12-27 12:01:16 | 5010 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | d648127b-adde-4248-8a81-75034636efe8 |
| 13494 | 2025-12-27 12:00:49 | 5013 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | b1795ae8-0a34-4097-ac71-0c3cbff9592d |
| 13493 | 2025-12-27 12:00:48 | 5003 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 61a6978e-47c5-494f-a88a-e4e71e9989a1 |
| 13492 | 2025-12-27 12:00:48 | 5000 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 03097745-09ca-4250-b0c2-1d64c948247f |
| 25 | 2025-12-13 11:42:37 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.69e+28 exceeds 1e28 (critical instability) | a16676b4-828e-44e2-9fe9-ccb5dc264d42 |
| 24 | 2025-12-13 11:41:50 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.62e+29 exceeds 1e28 (critical instability) | e17435c3-d5f4-437e-b2f9-01365fe163d1 |
| 23 | 2025-12-13 11:40:38 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 3.28e+28 exceeds 1e28 (critical instability) | ceeaecf9-6414-4ed6-a467-5199b491715c |
| 22 | 2025-12-13 11:36:13 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 3.79e+29 exceeds 1e28 (critical instability) | ae60c5f3-e31d-4d61-a4d5-d9e6bfdcc217 |
| 21 | 2025-12-13 11:35:35 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 1.58e+28 exceeds 1e28 (critical instability) | d38f2d50-db9e-4a2d-a7d8-b630b8e3174f |
| 20 | 2025-12-13 11:34:57 | 5092 | models.mahl.regularization | 1.0 | 10.0 | ADAPTIVE_TUNING | Condition number 2.01e+29 exceeds 1e28 (critical instability) | 0b61b75e-e221-49b0-8b08-bc26c0acc3d9 |

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
**Row Count:** 24  
**Date Range:** 2025-12-27 11:19:09 to 2025-12-29 10:57:59  

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
| 33 | cf9794af-bad7-4af5-9051-6a363540ef56 | 1 | False | 97 | 9 | ["Missing timestamp column: EntryDateTime", "Insufficient rows: 97 < 100"] | NULL | b44d1ef97088 | 2025-12-29 10:57:59 |
| 32 | 4b84eb86-d803-46ce-9303-0f6fd65b78ca | 1 | False | 97 | 9 | ["Missing timestamp column: EntryDateTime", "Insufficient rows: 97 < 100"] | NULL | b44d1ef97088 | 2025-12-29 10:57:05 |
| 26 | 5e01e5d5-bb0a-42fa-97c6-14041794e22e | 2621 | False | 97 | 16 | ["Missing timestamp column: EntryDateTime", "Insufficient rows: 97 < 100"] | NULL | 7a211a9e7e50 | 2025-12-27 12:28:45 |
| 25 | dc5383bf-4484-4982-8f3a-e127ba879a11 | 5026 | False | 127 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | f4e463b52b45 | 2025-12-27 12:28:39 |
| 23 | 7842171d-f59a-4e6d-84fa-b506b12b32fa | 5025 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 3c776b413cc5 | 2025-12-27 12:28:24 |
| 22 | e8f29a1a-ab7d-489a-ac58-eab2613a56c8 | 2621 | False | 97 | 16 | ["Missing timestamp column: EntryDateTime", "Insufficient rows: 97 < 100"] | NULL | 7a211a9e7e50 | 2025-12-27 12:28:22 |
| 21 | 2e79c7a6-8dab-4567-b187-e0d9ab4c54d2 | 5026 | False | 127 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | f4e463b52b45 | 2025-12-27 12:28:03 |
| 20 | c6f10086-6226-4886-8877-a412b38f79d9 | 5024 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 814cdbd30915 | 2025-12-27 12:27:49 |
| 19 | 3e488317-aa29-4081-a473-fbe267a091ed | 5025 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 3c776b413cc5 | 2025-12-27 12:27:45 |
| 18 | 3689924b-18e4-4e03-9284-43d874372c74 | 5017 | False | 129 | 81 | ["Missing timestamp column: EntryDateTime"] | NULL | 432f58c3355a | 2025-12-27 12:27:25 |

---


## dbo.ACM_DataQuality

**Primary Key:** No primary key  
**Row Count:** 1,880  
**Date Range:** 2022-04-04 02:30:00 to 2024-01-05 00:00:00  

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
| ambient | 97 | 0 | 0.0 | 0.4284217655658722 | 0 | 0 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| coolant | 97 | 0 | 0.0 | 0.4119116961956024 | 0 | 0 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| i_d | 97 | 0 | 0.0 | 0.6761859655380249 | 0 | 0 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| i_q | 97 | 0 | 0.0 | 0.01414421759545803 | 0 | 0 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| motor_speed | 97 | 0 | 0.0 | 0.0009838500991463661 | 0 | 12 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| pm | 97 | 0 | 0.0 | 3.802807092666626 | 0 | 0 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| profile_id | 97 | 0 | 0.0 | 0.0 | 0 | 96 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| stator_tooth | 97 | 0 | 0.0 | 2.375089645385742 | 0 | 1 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| stator_winding | 97 | 0 | 0.0 | 3.0065267086029053 | 0 | 0 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |
| stator_yoke | 97 | 0 | 0.0 | 1.4982963800430298 | 0 | 0 | 2024-01-01 02:24:00 | 2024-01-01 04:00:00 | 97 |

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
**Row Count:** 196  
**Date Range:** 2025-12-27 06:31:05 to 2025-12-27 06:31:31  

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
| 3920 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | cusum_z | cusum_z | 1.0 | 2025-12-27 06:31:31 |
| 3919 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | cusum_z | omr_z | -0.3490801951489417 | 2025-12-27 06:31:31 |
| 3918 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | cusum_z | gmm_z | 0.13054986011106023 | 2025-12-27 06:31:31 |
| 3917 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | cusum_z | iforest_z | -0.23040830646312513 | 2025-12-27 06:31:31 |
| 3916 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | cusum_z | pca_t2_z | 0.08062480516794114 | 2025-12-27 06:31:31 |
| 3915 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | cusum_z | pca_spe_z | 0.31891721903755055 | 2025-12-27 06:31:31 |
| 3914 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | cusum_z | ar1_z | -0.19298157914631084 | 2025-12-27 06:31:31 |
| 3913 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | omr_z | cusum_z | -0.3490801951489417 | 2025-12-27 06:31:31 |
| 3912 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | omr_z | omr_z | 1.0 | 2025-12-27 06:31:31 |
| 3911 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | omr_z | gmm_z | 0.13961614599906466 | 2025-12-27 06:31:31 |

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
**Row Count:** 6  
**Date Range:** 2022-05-03 13:30:00 to 2022-10-13 13:00:00  

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
**Row Count:** 26  

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
| B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | 2 | 375.0 | NULL | NULL | 1.1149577005805225 | 0.5781043065697401 |
| E1596C1E-4B19-4E7A-A7EF-168EB50EC183 | 5092 | 2 | 195.0 | NULL | NULL | 1.5868823434631993 | 0.9781944567732942 |
| 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | 2 | 525.0 | NULL | NULL | 1.1679315361915623 | 0.5791183974007266 |
| B425F409-811B-4B39-B026-3198F34D5336 | 5092 | 1 | 340.0 | NULL | NULL | 1.7627930715382036 | 0.689486096200057 |
| 601EDF93-FC4D-448C-9E2C-374ED1582373 | 5092 | 3 | 120.0 | NULL | NULL | 1.739453144252883 | 0.9867868016616163 |
| B938DABA-38C1-4680-AF86-3B31F85FF50F | 5092 | 2 | 210.0 | NULL | NULL | 1.5995743038010275 | 0.7721095755236788 |
| CEEAECF9-6414-4ED6-A467-5199B491715C | 5092 | 2 | 75.0 | NULL | NULL | 1.5428271673589493 | 0.8318848460286552 |
| 19AD36A7-7882-4538-AE66-5CEC6E8C069B | 5092 | 4 | 25.0 | NULL | NULL | 1.5849892125448568 | 0.9274378191447452 |
| B3B39E65-B546-413C-A1FF-67EC80C7AF7D | 5092 | 1 | 560.0 | NULL | NULL | 1.6615149047239914 | 1.0041955327730945 |

### Bottom 10 Records

| RunID | EquipID | EpisodeCount | MedianDurationMinutes | CoveragePct | TimeInAlertPct | MaxFusedZ | AvgFusedZ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 | 1 | 260.0 | NULL | NULL | 1.7012475804898572 | 1.080207519428952 |
| 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | 1 | 210.0 | NULL | NULL | 1.3405905304603716 | 0.7888532335090899 |
| AE60C5F3-E31D-4D61-A4D5-D9E6BFDCC217 | 5092 | 1 | 160.0 | NULL | NULL | 0.9261133335180507 | 0.5401970212940471 |
| 1C3C911D-2883-459C-845E-D43A1AEE5724 | 5092 | 1 | 20.0 | NULL | NULL | 1.7665934918317086 | 1.3682368878795295 |
| A16676B4-828E-44E2-9FE9-CCB5DC264D42 | 5092 | 2 | 70.0 | NULL | NULL | 1.1063869301636333 | 0.7670993925435602 |
| 88401CDD-1BC5-4ECB-A340-BC98624CDC94 | 5092 | 2 | 65.0 | NULL | NULL | 1.8423836028067593 | 0.872676615538867 |
| 0B61B75E-E221-49B0-8B08-BC26C0ACC3D9 | 5092 | 5 | 20.0 | NULL | NULL | 1.935941871856427 | 0.886069710139758 |
| D38F2D50-DB9E-4A2D-A7D8-B630B8E3174F | 5092 | 2 | 150.0 | NULL | NULL | 1.7619586331919688 | 0.7826185980282756 |
| C51C247D-6B86-4CF4-93C4-9547CD9E4069 | 5092 | 1 | 280.0 | NULL | NULL | 1.3736364944286321 | 0.4684065646851135 |
| D51FE222-B378-420B-8A64-854D5A0F645B | 5092 | 2 | 160.0 | NULL | NULL | 2.1491588758222804 | 0.8090813724990242 |

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
**Row Count:** 21,168  
**Date Range:** 2022-04-05 18:40:00 to 2022-05-02 02:20:00  

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
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 02:30:00 | 5.66556203717848e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 02:40:00 | 5.486203156935715e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 02:50:00 | 5.312384840675239e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:00:00 | 5.143940400263454e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:10:00 | 4.980708020513876e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:20:00 | 4.822530621138808e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:30:00 | 4.669255722476889e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:40:00 | 4.5207353149000626e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:50:00 | 4.376825731799679e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 04:00:00 | 4.237387526061357e-10 | 0.9999999994334438 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:42:13 |

### Bottom 10 Records

| EquipID | RunID | Timestamp | FailureProb | SurvivalProb | HazardRate | ThresholdUsed | Method | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 02:20:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 02:10:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 02:00:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:50:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:40:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:30:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:20:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:10:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:00:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 00:50:00 | 3.5752262629154693e-17 | 0.9999999993437955 | 0.0 | 50.0 | GaussianTail | 2025-12-13 11:44:41 |

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
**Row Count:** 411  
**Date Range:** 2025-12-27 06:21:53 to 2025-12-27 06:22:20  

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
| 420 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_12_avg_energy_0 | low_variance | 4.085587598093954e-28 | NULL | 2025-12-27 06:22:20 |
| 419 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_49_energy_2 | low_variance | 0.0 | NULL | 2025-12-27 06:22:20 |
| 418 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_31_std_energy_0 | low_variance | 1.9568969367435736e-26 | NULL | 2025-12-27 06:22:20 |
| 417 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_15_avg_energy_0 | low_variance | 2.7726905013302353e-27 | NULL | 2025-12-27 06:22:20 |
| 416 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_51_energy_0 | low_variance | 1.2510750933511437e-21 | NULL | 2025-12-27 06:22:20 |
| 415 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_31_max_energy_0 | low_variance | 2.0790867458829796e-26 | NULL | 2025-12-27 06:22:20 |
| 414 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | power_30_avg_energy_0 | low_variance | 1.7230729878270477e-32 | NULL | 2025-12-27 06:22:20 |
| 413 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_50_energy_0 | low_variance | 4.873275977121772e-21 | NULL | 2025-12-27 06:22:20 |
| 412 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_5_std_energy_0 | low_variance | 2.773522285126173e-27 | NULL | 2025-12-27 06:22:20 |
| 411 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | sensor_5_avg_energy_0 | low_variance | 8.351192996132956e-27 | NULL | 2025-12-27 06:22:20 |

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
| 5092 | 1 | {"alpha": 0.05, "beta": 0.01, "level": 92.64558701073078, "trend": 0.03150434016807175, "std_erro... | {"forecast_mean": 99.15204522899016, "forecast_std": 1.8518969071118483, "forecast_range": 7.3229... | NULL |  | 35301 | 1.8518969071118483 | NULL | NULL |

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
**Row Count:** 21,168  
**Date Range:** 2022-04-05 18:40:00 to 2022-05-02 02:20:00  

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
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 02:30:00 | 92.55784776674905 | 89.44012506187434 | 95.67557047162376 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 02:40:00 | 92.57692014330884 | 89.45522448366415 | 95.69861580295353 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 02:50:00 | 92.59599251986864 | 89.47025004514572 | 95.72173499459157 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:00:00 | 92.61506489642844 | 89.4852012564089 | 95.74492853644797 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:10:00 | 92.63413727298824 | 89.5000776353888 | 95.76819691058768 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:20:00 | 92.65320964954803 | 89.51487870792872 | 95.79154059116735 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:30:00 | 92.67228202610784 | 89.52960400784069 | 95.81496004437498 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:40:00 | 92.69135440266763 | 89.54425307696323 | 95.83845572837203 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 03:50:00 | 92.71042677922743 | 89.55882546521653 | 95.86202809323834 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 2022-04-20 04:00:00 | 92.72949915578722 | 89.57332073065477 | 95.88567758091968 | 1.5588613524373542 | ExponentialSmoothing | 2025-12-13 11:42:13 | NULL |

### Bottom 10 Records

| EquipID | RunID | Timestamp | ForecastHealth | CiLower | CiUpper | ForecastStd | Method | CreatedAt | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 02:20:00 | 100.0 | 97.05794986266811 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 02:10:00 | 100.0 | 97.06405504443742 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 02:00:00 | 100.0 | 97.07014013837704 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:50:00 | 100.0 | 97.07620513434733 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:40:00 | 100.0 | 97.08225002219166 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:30:00 | 100.0 | 97.08827479173638 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:20:00 | 100.0 | 97.09427943279054 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:10:00 | 100.0 | 97.10026393514617 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 01:00:00 | 100.0 | 97.10622828857792 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 2022-04-30 00:50:00 | 100.0 | 97.11217248284319 | 100.0 | 1.512887566209102 | ExponentialSmoothing | 2025-12-13 11:44:41 | NULL |

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
**Row Count:** 3,639  
**Date Range:** 2022-04-04 02:30:00 to 2022-10-16 00:30:00  

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
| 2022-10-16 00:30:00 | 92.27 | GOOD | 0.37470000982284546 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 92.76000213623047 | NORMAL |
| 2022-10-16 00:00:00 | 92.06 | GOOD | 0.4458000063896179 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 92.16999816894531 | NORMAL |
| 2022-10-15 23:30:00 | 92.01 | GOOD | 0.40119999647140503 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 92.54000091552734 | NORMAL |
| 2022-10-15 23:00:00 | 91.78 | GOOD | 0.44600000977516174 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 92.16000366210938 | NORMAL |
| 2022-10-15 22:30:00 | 91.62 | GOOD | 0.44339999556541443 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 92.19000244140625 | NORMAL |
| 2022-10-15 22:00:00 | 91.37 | GOOD | 0.4178999960422516 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 92.4000015258789 | NORMAL |
| 2022-10-15 21:30:00 | 90.93 | GOOD | 0.4593000113964081 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 92.05000305175781 | NORMAL |
| 2022-10-15 21:00:00 | 90.45 | GOOD | 0.4422000050544739 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 92.19999694824219 | NORMAL |
| 2022-10-15 20:30:00 | 89.7 | GOOD | 0.6851999759674072 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 89.81999969482422 | NORMAL |
| 2022-10-15 20:00:00 | 89.64 | GOOD | 1.0436999797821045 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 85.16000366210938 | NORMAL |

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
**Row Count:** 4  

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

---


## dbo.ACM_PCA_Loadings

**Primary Key:** RecordID  
**Row Count:** 15,765  
**Date Range:** 2025-12-27 12:24:27 to 2025-12-27 12:25:18  

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
| 22155 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | wind_speed_4_avg_rz | wind_speed_4_avg_rz | 0.04493549084603461 | 2025-12-27 12:25:26 |
| 22154 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | wind_speed_3_std_rz | wind_speed_3_std_rz | 0.030551704685874325 | 2025-12-27 12:25:26 |
| 22153 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | wind_speed_3_min_rz | wind_speed_3_min_rz | 0.02208703402405643 | 2025-12-27 12:25:26 |
| 22152 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | wind_speed_3_max_rz | wind_speed_3_max_rz | 0.04662940687965499 | 2025-12-27 12:25:26 |
| 22151 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | wind_speed_3_avg_rz | wind_speed_3_avg_rz | 0.05552718829925102 | 2025-12-27 12:25:26 |
| 22150 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | sensor_9_avg_rz | sensor_9_avg_rz | -0.002992295549064732 | 2025-12-27 12:25:26 |
| 22149 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | sensor_8_avg_rz | sensor_8_avg_rz | 0.002173348183830534 | 2025-12-27 12:25:26 |
| 22148 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | sensor_7_avg_rz | sensor_7_avg_rz | 0.002696482205405415 | 2025-12-27 12:25:26 |
| 22147 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | sensor_6_avg_rz | sensor_6_avg_rz | -0.01946690305740125 | 2025-12-27 12:25:26 |
| 22146 | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 | 2025-12-27 12:25:18 | 5 | 5 | sensor_5_std_rz | sensor_5_std_rz | 0.02211051373580976 | 2025-12-27 12:25:26 |

---


## dbo.ACM_PCA_Metrics

**Primary Key:** ID  
**Row Count:** 4  
**Date Range:** 2025-12-27 06:23:37 to 2025-12-27 06:24:05  

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

---


## dbo.ACM_PCA_Models

**Primary Key:** RecordID  
**Row Count:** 4  
**Date Range:** 2025-12-27 12:24:27 to 2025-12-27 12:25:18  

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

---


## dbo.ACM_RUL

**Primary Key:** EquipID, RunID  
**Row Count:** 21  
**Date Range:** 2025-12-14 03:55:51 to 2025-12-20 11:46:20  

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
| 5092 | E17435C3-D5F4-437E-B2F9-01365FE163D1 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:42:13 | Multipath | 1000 |
| 5092 | E1596C1E-4B19-4E7A-A7EF-168EB50EC183 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:45:30 | Multipath | 1000 |
| 5092 | B425F409-811B-4B39-B026-3198F34D5336 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:39:26 | Multipath | 1000 |
| 5092 | 601EDF93-FC4D-448C-9E2C-374ED1582373 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:37:59 | Multipath | 1000 |
| 5092 | B938DABA-38C1-4680-AF86-3B31F85FF50F | 17.166666666666664 | 14.307113899225705 | 17.166666666666664 | 19.59051480103239 | 0.5652893623846694 | 2025-12-14 04:42:44 | Multipath | 1000 |
| 5092 | CEEAECF9-6414-4ED6-A467-5199B491715C | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:41:01 | Multipath | 1000 |
| 5092 | 19AD36A7-7882-4538-AE66-5CEC6E8C069B | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:46:20 | Multipath | 1000 |
| 5092 | B3B39E65-B546-413C-A1FF-67EC80C7AF7D | 143.0 | 106.33310169240622 | 143.0 | 170.2348182710401 | 0.5489622840025496 | 2025-12-19 10:43:52 | Multipath | 1000 |
| 5092 | 251FC2E9-C235-447A-9D43-6AD22DDD1D0A | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:40:11 | Multipath | 1000 |
| 5092 | 58CCC45C-0441-4D90-98A2-6AE5973FE05E | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:33:56 | Multipath | 1000 |

### Bottom 10 Records

| EquipID | RunID | RUL_Hours | P10_LowerBound | P50_Median | P90_UpperBound | Confidence | FailureTime | Method | NumSimulations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5092 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:44:41 | Multipath | 1000 |
| 5092 | AE60C5F3-E31D-4D61-A4D5-D9E6BFDCC217 | 168.0 | 135.96691691815874 | 168.0 | 170.2348182710401 | 0.5670697962783527 | 2025-12-20 11:36:31 | Multipath | 1000 |
| 5092 | 1C3C911D-2883-459C-845E-D43A1AEE5724 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:38:43 | Multipath | 1000 |
| 5092 | A16676B4-828E-44E2-9FE9-CCB5DC264D42 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:43:02 | Multipath | 1000 |
| 5092 | 88401CDD-1BC5-4ECB-A340-BC98624CDC94 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:32:10 | Multipath | 1000 |
| 5092 | 0B61B75E-E221-49B0-8B08-BC26C0ACC3D9 | 40.5 | 36.01445912563712 | 40.5 | 44.078658302322886 | 0.5787239934782364 | 2025-12-15 04:05:13 | Multipath | 1000 |
| 5092 | D38F2D50-DB9E-4A2D-A7D8-B630B8E3174F | 16.333333333333332 | 13.81376514407999 | 16.333333333333332 | 18.408328563039056 | 0.5704420327677934 | 2025-12-14 03:55:51 | Multipath | 1000 |
| 5092 | D51FE222-B378-420B-8A64-854D5A0F645B | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2025-12-20 11:31:33 | Multipath | 1000 |
| 5092 | 0E56EE15-040F-44FD-A4B4-821E8F63BB06 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.590678079171218 | 2025-12-20 11:34:35 | Multipath | 1000 |
| 5092 | 7AE2BB4D-F115-4001-88C2-808ED3699330 | 168.0 | 164.5646997581053 | 168.0 | 170.2348182710401 | 0.5814895690686156 | 2025-12-20 11:37:13 | Multipath | 1000 |

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
**Row Count:** 4  
**Date Range:** 2025-12-27 06:30:48 to 2025-12-27 06:31:16  

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

---


## dbo.ACM_RegimeDefinitions

**Primary Key:** ID  
**Row Count:** 12  
**Date Range:** 2025-12-27 06:26:10 to 2025-12-27 06:26:37  

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
| 18 | 5010 | 1 | 2 | Regime_2 | [1.7732375475106303, 1.67223554036834, 1.9386688279254096, 1.365046832009745, 1.772301350611371, ... | [] | 23 | NULL | LEARNING |
| 17 | 5010 | 1 | 1 | Regime_1 | [-0.6510032323638281, -0.6971255694616888, -0.5508004426956177, -0.6790673614924555, -0.666808662... | [] | 68 | NULL | LEARNING |
| 16 | 5010 | 1 | 0 | Regime_0 | [0.07395956263262563, 0.1925636953961924, -0.16600490057603617, 0.29864275364500104, 0.1032667960... | [] | 40 | NULL | LEARNING |
| 15 | 5013 | 1 | 4 | Regime_4 | [1.6785599012737689, 1.1883648633956911, 2.3031346247248035, -1.1685142986800359, 1.6729365891736... | [] | 12 | NULL | LEARNING |
| 14 | 5013 | 1 | 3 | Regime_3 | [-1.1050834447308313, -1.3831270757268688, -0.8264039728810878, -1.2487332983746555, -1.116041944... | [] | 12 | NULL | LEARNING |
| 13 | 5013 | 1 | 2 | Regime_2 | [-0.375968825384818, -0.20761686028435133, -0.48440716797667466, 0.20114904745430245, -0.37044659... | [] | 40 | NULL | LEARNING |
| 12 | 5013 | 1 | 1 | Regime_1 | [-0.9571371021435846, -0.9844831097757256, -0.8117585231961642, -0.6222996969694032, -0.958811484... | [] | 31 | NULL | LEARNING |
| 11 | 5013 | 1 | 0 | Regime_0 | [1.1013239100160759, 1.1718048602619064, 0.8186670980728983, 1.0979557428811375, 1.10139900694751... | [] | 34 | NULL | LEARNING |
| 10 | 5003 | 1 | 1 | Regime_1 | [0.7855003223251116, 0.5101972222328186, 0.1256801780536965, 0.029965903055574502, -1.76903894177... | [] | 16 | NULL | LEARNING |
| 9 | 5003 | 1 | 0 | Regime_0 | [-0.11307527014937278, -0.09937626516279638, -0.008573813168239295, -0.0499809403618687, 0.242925... | [] | 113 | NULL | LEARNING |

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
**Row Count:** 5  
**Date Range:** 2025-12-26 11:42:12 to 2025-12-27 06:26:36  

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
| 1 | 1 | 2 | [[-0.48160468919582583, -0.47764426459677967, -0.19194448437272843, -0.3768078673995784, -0.33756... | [] | [] | [] | [] | 0 | 0.4377001021963198 |
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
**Row Count:** 3,639  
**Date Range:** 2022-04-04 02:30:00 to 2022-10-16 00:30:00  

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
| 2022-10-16 00:30:00 | 1 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-16 00:00:00 | 1 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-15 23:30:00 | 1 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-15 23:00:00 | 1 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-15 22:30:00 | 0 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-15 22:00:00 | 2 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-15 21:30:00 | 2 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-15 21:00:00 | 2 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-15 20:30:00 | 2 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |
| 2022-10-15 20:00:00 | 2 | unknown | D648127B-ADDE-4248-8A81-75034636EFE8 | 5010 |

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
**Row Count:** 31,438  
**Date Range:** 2025-12-11 11:55:49 to 2025-12-13 06:16:40  

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
**Row Count:** 1,311  
**Date Range:** 2025-12-11 17:29:21 to 2025-12-27 11:57:50  

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
**Row Count:** 4  
**Date Range:** 2022-04-27 03:00:00 to 2022-10-09 08:40:00  

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

---


## dbo.ACM_Runs

**Primary Key:** RunID  
**Row Count:** 57  
**Date Range:** 2025-12-13 06:00:34 to 2025-12-29 05:27:57  

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
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | WFA_TURBINE_92 | 2025-12-13 06:11:31 | 2025-12-13 06:12:14 | 42 |  | 144 | 4819 | 2 |
| C8EA4C4A-0B58-40C8-B284-01B64D26D884 | 5017 | WFA_TURBINE_17 | 2025-12-27 06:56:37 | 2025-12-27 06:56:49 | 12 |  | 0 | 0 | 0 |
| 61B2669A-EDD8-4C3B-9CE4-04888D086216 | 8632 | WIND_TURBINE | 2025-12-20 04:26:22 | 2025-12-20 04:26:22 | 0 |  | 0 | 0 | 0 |
| B1795AE8-0A34-4097-AC71-0C3CBFF9592D | 5013 | WFA_TURBINE_13 | 2025-12-27 05:49:00 | 2025-12-27 06:54:39 | 3938 |  | 129 | 3920 | 2 |
| 4B84EB86-D803-46CE-9303-0F6FD65B78CA | 1 | FD_FAN | 2025-12-29 05:27:03 | 2025-12-29 05:27:08 | 4 |  | 0 | 0 | 0 |
| 485841A7-C79A-4389-BD70-1240D29D6990 | 8632 | WIND_TURBINE | 2025-12-20 04:26:16 | 2025-12-20 04:26:16 | 0 |  | 0 | 0 | 0 |
| 5E01E5D5-BB0A-42FA-97C6-14041794E22E | 2621 | GAS_TURBINE | 2025-12-27 06:58:43 | 2025-12-27 06:58:48 | 4 |  | 0 | 0 | 0 |
| E1596C1E-4B19-4E7A-A7EF-168EB50EC183 | 5092 | WFA_TURBINE_92 | 2025-12-13 06:14:45 | 2025-12-13 06:15:31 | 45 |  | 144 | 4819 | 2 |
| 03097745-09CA-4250-B0C2-1D64C948247F | 5000 | WFA_TURBINE_0 | 2025-12-27 05:49:00 | 2025-12-27 06:55:05 | 3964 |  | 129 | 3955 | 2 |
| 4D28948A-7100-4FD1-B757-2B6A6E412D8E | 8632 | WIND_TURBINE | 2025-12-20 04:26:02 | 2025-12-20 04:26:02 | 0 |  | 0 | 0 | 0 |

### Bottom 10 Records

| RunID | EquipID | EquipName | StartedAt | CompletedAt | DurationSeconds | ConfigSignature | TrainRowCount | ScoreRowCount | EpisodeCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3E488317-AA29-4081-A473-FBE267A091ED | 5025 | WFA_TURBINE_25 | 2025-12-27 06:57:36 | 2025-12-27 06:57:48 | 12 |  | 0 | 0 | 0 |
| E750E06F-E624-45FD-B683-FB4815EEF360 | 8634 | ELECTRIC_MOTOR | 2025-12-20 04:26:11 | 2025-12-20 04:26:12 | 0 |  | 0 | 0 | 0 |
| 81A35D0D-2214-42A1-8DC8-F6D14783FCC5 | 8634 | ELECTRIC_MOTOR | 2025-12-20 04:25:52 | 2025-12-20 04:25:59 | 5 |  | 0 | 0 | 0 |
| E8008D4A-B739-4894-AA3A-F62215AE4135 | 5000 | WFA_TURBINE_0 | 2025-12-27 06:56:02 | 2025-12-27 06:56:13 | 10 |  | 0 | 0 | 0 |
| F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E | 5092 | WFA_TURBINE_92 | 2025-12-13 06:13:57 | 2025-12-13 06:14:42 | 44 |  | 144 | 4817 | 1 |
| E8F29A1A-AB7D-489A-AC58-EAB2613A56C8 | 2621 | GAS_TURBINE | 2025-12-27 06:58:20 | 2025-12-27 06:58:24 | 3 |  | 0 | 0 | 0 |
| 61A6978E-47C5-494F-A88A-E4E71E9989A1 | 5003 | WFA_TURBINE_3 | 2025-12-27 05:49:00 | 2025-12-27 06:55:03 | 3962 |  | 129 | 3953 | 1 |
| DC5383BF-4484-4982-8F3A-E127BA879A11 | 5026 | WFA_TURBINE_26 | 2025-12-27 06:58:28 | 2025-12-27 06:58:41 | 13 |  | 0 | 0 | 0 |
| 2E79C7A6-8DAB-4567-B187-E0D9AB4C54D2 | 5026 | WFA_TURBINE_26 | 2025-12-27 06:57:51 | 2025-12-27 06:58:06 | 14 |  | 0 | 0 | 0 |
| AE60C5F3-E31D-4D61-A4D5-D9E6BFDCC217 | 5092 | WFA_TURBINE_92 | 2025-12-13 06:05:55 | 2025-12-13 06:06:32 | 37 |  | 144 | 4782 | 1 |

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
**Row Count:** 3,639  
**Date Range:** 2022-04-04 02:30:00 to 2022-10-16 00:30:00  

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
| 2022-10-16 00:30:00 | -0.5980498194694519 | 0.0 | 0.0 | NULL | 0.5538449287414551 | 1.2035874128341675 | -0.013208406046032906 | NULL | NULL |
| 2022-10-16 00:00:00 | 0.008513766340911388 | 0.0 | 0.0 | NULL | 0.3183366358280182 | 1.1309788227081299 | 0.09224414825439453 | NULL | NULL |
| 2022-10-15 23:30:00 | -0.2781026065349579 | 0.0 | 0.0 | NULL | 0.3150147795677185 | 1.3235409259796143 | 0.201992005109787 | NULL | NULL |
| 2022-10-15 23:00:00 | -0.3151337504386902 | 0.0 | 0.0 | NULL | 0.5984691381454468 | 1.6201395988464355 | 0.3114207983016968 | NULL | NULL |
| 2022-10-15 22:30:00 | 0.06586960703134537 | 0.0 | 0.0 | NULL | 0.1831870824098587 | 2.2528746128082275 | 0.42444688081741333 | NULL | NULL |
| 2022-10-15 22:00:00 | 0.0 | 0.0 | 0.0 | NULL | 0.29738569259643555 | 1.7554396390914917 | 0.5385384559631348 | NULL | NULL |
| 2022-10-15 21:30:00 | 0.48289385437965393 | 0.0 | 0.0 | NULL | 0.325592577457428 | 0.8631320595741272 | 0.6543880105018616 | NULL | NULL |
| 2022-10-15 21:00:00 | 0.3472713828086853 | 0.0 | 0.0 | NULL | 0.4262773394584656 | 0.306374728679657 | 0.7750669121742249 | NULL | NULL |
| 2022-10-15 20:30:00 | 1.1250247955322266 | 0.0 | 0.0 | NULL | 1.0041459798812866 | -0.01778220757842064 | 0.8988869190216064 | NULL | NULL |
| 2022-10-15 20:00:00 | 2.6166844367980957 | 0.0 | 0.0 | NULL | 1.1738219261169434 | -0.1392204463481903 | 1.0287480354309082 | NULL | NULL |

---


## dbo.ACM_SeasonalPatterns

**Primary Key:** ID  
**Row Count:** 0  

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
**Row Count:** 1,244,271  
**Date Range:** 2025-12-27 06:51:41 to 2025-12-27 06:54:33  

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
| 1448400 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_4_avg_rz | wind_speed_4_avg_rz | 1.0 | pearson | 2025-12-27 06:54:33 |
| 1448399 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_std_rz | wind_speed_4_avg_rz | 0.4570344879492993 | pearson | 2025-12-27 06:54:33 |
| 1448398 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_std_rz | wind_speed_3_std_rz | 1.0 | pearson | 2025-12-27 06:54:33 |
| 1448397 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_min_rz | wind_speed_4_avg_rz | 0.3562723602364797 | pearson | 2025-12-27 06:54:33 |
| 1448396 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_min_rz | wind_speed_3_std_rz | 0.10630135180079728 | pearson | 2025-12-27 06:54:33 |
| 1448395 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_min_rz | wind_speed_3_min_rz | 1.0 | pearson | 2025-12-27 06:54:33 |
| 1448394 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_max_rz | wind_speed_4_avg_rz | 0.5922121050655708 | pearson | 2025-12-27 06:54:33 |
| 1448393 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_max_rz | wind_speed_3_std_rz | 0.7052258639937359 | pearson | 2025-12-27 06:54:33 |
| 1448392 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_max_rz | wind_speed_3_min_rz | 0.07107836785210722 | pearson | 2025-12-27 06:54:33 |
| 1448391 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | wind_speed_3_max_rz | wind_speed_3_max_rz | 1.0 | pearson | 2025-12-27 06:54:33 |

---


## dbo.ACM_SensorDefects

**Primary Key:** No primary key  
**Row Count:** 204  

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
| Density Anomaly (GMM) | Density | HIGH | 25 | 19.38 | 5.8758 | 1.1387 | 2.4642 | 1 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |
| Time-Series Anomaly (AR1) | Time-Series | HIGH | 15 | 11.63 | 3.2005 | 0.8874 | 0.4137 | 0 | B1795AE8-0A34-4097-AC71-0C3CBFF9592D |

### Bottom 10 Records

| DetectorType | DetectorFamily | Severity | ViolationCount | ViolationPct | MaxZ | AvgZ | CurrentZ | ActiveDefect | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Correlation Break (PCA-SPE) | Correlation | CRITICAL | 70 | 48.61 | 10.0 | 5.1942 | 0.594 | 0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E |
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 70 | 48.61 | 10.0 | 5.1923 | 0.6094 | 0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E |
| Time-Series Anomaly (AR1) | Time-Series | HIGH | 27 | 18.75 | 5.471 | 1.1007 | 0.0109 | 0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E |
| cusum_z | cusum_z | HIGH | 16 | 11.11 | 2.722 | 0.8493 | 0.9726 | 0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E |
| Baseline Consistency (OMR) | Baseline | MEDIUM | 14 | 9.72 | 3.2693 | 0.8891 | 0.7851 | 0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E |
| Multivariate Distance (Mahalanobis) | Multivariate | LOW | 5 | 3.47 | 2.3151 | 0.8641 | 0.2235 | 0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E |
| Rare State (IsolationForest) | Rare | LOW | 4 | 2.78 | 2.1744 | 0.7393 | 0.1282 | 0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E |
| Density Anomaly (GMM) | Density | LOW | 0 | 0.0 | 1.2272 | 0.5727 | 0.6617 | 0 | F8ADFB6C-950A-4C2F-B059-EC7EF3BD096E |
| Density Anomaly (GMM) | Density | CRITICAL | 36 | 27.91 | 6.3973 | 1.207 | 4.5797 | 1 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 |
| Time-Series Anomaly (AR1) | Time-Series | HIGH | 24 | 18.6 | 10.0 | 1.436 | 10.0 | 1 | 61A6978E-47C5-494F-A88A-E4E71E9989A1 |

---


## dbo.ACM_SensorForecast

**Primary Key:** RunID, EquipID, Timestamp, SensorName  
**Row Count:** 0  

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
**Row Count:** 650  
**Date Range:** 2022-04-04 02:30:00 to 2022-10-15 20:00:00  

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
| wind_speed_3_min | 2022-10-15 20:00:00 | 2022-10-16 00:30:00 | 3.6004 | 3.6004 | 0.8879 | -0.8879 | 3.9000000953674316 | 0.4000000059604645 | 1.092366337776184 |
| sensor_31_std | 2022-10-14 20:00:00 | 2022-10-16 00:30:00 | 7.2107 | 7.2107 | 0.3789 | -0.3789 | 217.10000610351562 | 0.8999999761581421 | 11.694656372070312 |
| sensor_5_min | 2022-10-14 08:30:00 | 2022-10-16 00:30:00 | 5.2771 | 5.2771 | 0.9512 | 0.9512 | 90.0 | 24.0 | 9.487404823303223 |
| sensor_5_avg | 2022-10-14 08:30:00 | 2022-10-16 00:30:00 | 3.5473 | 3.5473 | 0.4195 | 0.4195 | 90.0 | 24.0 | 15.147709846496582 |
| sensor_5_std | 2022-10-14 07:30:00 | 2022-10-16 00:30:00 | 5.8448 | 5.8448 | 0.3911 | -0.3911 | 39.400001525878906 | 0.0 | 2.4709925651550293 |
| sensor_18_std | 2022-10-14 07:30:00 | 2022-10-16 00:30:00 | 3.8367 | 3.8367 | 0.3003 | -0.3003 | 576.9000244140625 | 35.5 | 74.80496978759766 |
| sensor_52_std | 2022-10-14 07:30:00 | 2022-10-16 00:30:00 | 3.7917 | 3.7917 | 0.016 | 0.016 | 5.199999809265137 | 0.699999988079071 | 0.6809160709381104 |
| sensor_44 | 2022-10-14 06:00:00 | 2022-10-16 00:30:00 | 4.7637 | -4.7637 | 1.8938 | -1.8938 | -3614.0 | -1770.0 | -553.217529296875 |
| sensor_47 | 2022-10-14 06:00:00 | 2022-10-16 00:30:00 | 3.6809 | -3.6809 | 2.7096 | -2.7096 | -3238.0 | -2512.0 | -486.8091735839844 |
| sensor_26_avg | 2022-10-13 21:00:00 | 2022-10-16 00:30:00 | 11.3582 | 11.3582 | 0.0873 | -0.0873 | 50.099998474121094 | 50.0 | 50.000762939453125 |

---


## dbo.ACM_SensorNormalized_TS

**Primary Key:** ID  
**Row Count:** 41,958  
**Date Range:** 2022-05-01 03:00:00 to 2022-10-16 00:30:00  

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
| 544566 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-16 00:30:00 | wind_speed_4_avg | NULL | 2.5 | 2025-12-27 06:54:55 |
| 544565 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-16 00:00:00 | wind_speed_4_avg | NULL | 2.4000000953674316 | 2025-12-27 06:54:55 |
| 544564 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-15 23:30:00 | wind_speed_4_avg | NULL | 2.0999999046325684 | 2025-12-27 06:54:55 |
| 544563 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-15 23:00:00 | wind_speed_4_avg | NULL | 2.5999999046325684 | 2025-12-27 06:54:55 |
| 544562 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-15 22:30:00 | wind_speed_4_avg | NULL | 4.199999809265137 | 2025-12-27 06:54:55 |
| 544561 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-15 22:00:00 | wind_speed_4_avg | NULL | 5.300000190734863 | 2025-12-27 06:54:55 |
| 544560 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-15 21:30:00 | wind_speed_4_avg | NULL | 6.099999904632568 | 2025-12-27 06:54:55 |
| 544559 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-15 21:00:00 | wind_speed_4_avg | NULL | 5.599999904632568 | 2025-12-27 06:54:55 |
| 544558 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-15 20:30:00 | wind_speed_4_avg | NULL | 6.199999809265137 | 2025-12-27 06:54:55 |
| 544557 | d648127b-adde-4248-8a81-75034636efe8 | 5010 | 2022-10-15 20:00:00 | wind_speed_4_avg | NULL | 7.0 | 2025-12-27 06:54:55 |

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
**Row Count:** 174  
**Date Range:** 2025-12-13 06:00:50 to 2025-12-27 06:26:52  

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
| ar1_params | 5000 | 1 | 2025-12-27 06:26:15 | {"n_sensors": 790, "mean_autocorr": 655489383.9208, "mean_residual_std": 448915724.5588, "params_... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 43780 bytes> |
| gmm_model | 5000 | 1 | 2025-12-27 06:26:20 | {"n_components": 3, "covariance_type": "diag", "bic": 6.829492608516723e+25, "aic": 6.82949260851... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 76880 bytes> |
| iforest_model | 5000 | 1 | 2025-12-27 06:26:20 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 1422089 bytes> |
| omr_model | 5000 | 1 | 2025-12-27 06:26:23 | NULL | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 83948 bytes> |
| pca_model | 5000 | 1 | 2025-12-27 06:26:16 | {"n_components": 5, "variance_ratio_sum": 0.6513, "variance_ratio_first_component": 0.2442, "vari... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 38975 bytes> |
| ar1_params | 5003 | 1 | 2025-12-27 06:26:16 | {"n_sensors": 790, "mean_autocorr": 784001549.624, "mean_residual_std": 650872304.488, "params_co... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 43780 bytes> |
| gmm_model | 5003 | 1 | 2025-12-27 06:26:20 | {"n_components": 3, "covariance_type": "diag", "bic": 8.300190678260456e+25, "aic": 8.30019067826... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 76937 bytes> |
| iforest_model | 5003 | 1 | 2025-12-27 06:26:20 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 1418921 bytes> |
| omr_model | 5003 | 1 | 2025-12-27 06:26:23 | NULL | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 83948 bytes> |
| pca_model | 5003 | 1 | 2025-12-27 06:26:16 | {"n_components": 5, "variance_ratio_sum": 0.6257, "variance_ratio_first_component": 0.2296, "vari... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 38975 bytes> |

### Bottom 10 Records

| ModelType | EquipID | Version | EntryDateTime | ParamsJSON | StatsJSON | RunID | ModelBytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pca_model | 5092 | 22 | 2025-12-13 06:15:54 | {"n_components": 5, "variance_ratio_sum": 0.6144, "variance_ratio_first_component": 0.2577, "vari... | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 38975 bytes> |
| omr_model | 5092 | 22 | 2025-12-13 06:15:54 | NULL | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 83948 bytes> |
| mhal_params | 5092 | 22 | 2025-12-13 06:15:54 | NULL | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 4999426 bytes> |
| iforest_model | 5092 | 22 | 2025-12-13 06:15:54 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 1495657 bytes> |
| gmm_model | 5092 | 22 | 2025-12-13 06:15:54 | {"n_components": 3, "covariance_type": "diag", "bic": 1.2503175226128738e+25, "aic": 1.2503175226... | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 76937 bytes> |
| feature_medians | 5092 | 22 | 2025-12-13 06:15:54 | NULL | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 54705 bytes> |
| ar1_params | 5092 | 22 | 2025-12-13 06:15:54 | {"n_sensors": 790, "mean_autocorr": 167364310.7437, "mean_residual_std": 191252786.2074, "params_... | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 43780 bytes> |
| pca_model | 5092 | 21 | 2025-12-13 06:15:03 | {"n_components": 5, "variance_ratio_sum": 0.6526, "variance_ratio_first_component": 0.2903, "vari... | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 38975 bytes> |
| omr_model | 5092 | 21 | 2025-12-13 06:15:03 | NULL | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 83948 bytes> |
| mhal_params | 5092 | 21 | 2025-12-13 06:15:03 | NULL | {"train_rows": 144, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 4999426 bytes> |

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

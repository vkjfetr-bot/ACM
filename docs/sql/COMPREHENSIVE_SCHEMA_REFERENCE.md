# ACM Comprehensive Database Schema Reference

_Generated automatically on 2025-12-26 16:14:32_

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
| dbo.ACM_ActiveModels | 11 | 0 | ID |
| dbo.ACM_AdaptiveConfig | 13 | 11 | ConfigID |
| dbo.ACM_AlertAge | 6 | 0 | — |
| dbo.ACM_Anomaly_Events | 6 | 375 | Id |
| dbo.ACM_AssetProfiles | 11 | 0 | ID |
| dbo.ACM_BaselineBuffer | 7 | 1,162,924 | Id |
| dbo.ACM_CalibrationSummary | 10 | 0 | ID |
| dbo.ACM_ColdstartState | 17 | 15 | EquipID, Stage |
| dbo.ACM_Config | 7 | 342 | ConfigID |
| dbo.ACM_ConfigHistory | 9 | 3,495 | ID |
| dbo.ACM_ContributionCurrent | 5 | 0 | — |
| dbo.ACM_ContributionTimeline | 7 | 0 | ID |
| dbo.ACM_DailyFusedProfile | 9 | 0 | ID |
| dbo.ACM_DataContractValidation | 10 | 0 | ID |
| dbo.ACM_DataQuality | 24 | 7,195 | — |
| dbo.ACM_DefectSummary | 12 | 0 | — |
| dbo.ACM_DefectTimeline | 10 | 0 | — |
| dbo.ACM_DetectorCorrelation | 7 | 3,577 | ID |
| dbo.ACM_DetectorForecast_TS | 10 | 0 | RunID, EquipID, DetectorName, Timestamp |
| dbo.ACM_DriftController | 10 | 0 | ID |
| dbo.ACM_DriftEvents | 2 | 0 | — |
| dbo.ACM_DriftSeries | 7 | 0 | ID |
| dbo.ACM_EnhancedFailureProbability_TS | 11 | 0 | RunID, EquipID, Timestamp, ForecastHorizon_Hours |
| dbo.ACM_EnhancedMaintenanceRecommendation | 13 | 0 | RunID, EquipID |
| dbo.ACM_EpisodeCulprits | 9 | 5,108 | ID |
| dbo.ACM_EpisodeDiagnostics | 16 | 425 | ID |
| dbo.ACM_EpisodeMetrics | 10 | 0 | — |
| dbo.ACM_Episodes | 8 | 174 | — |
| dbo.ACM_EpisodesQC | 10 | 0 | RecordID |
| dbo.ACM_FailureCausation | 12 | 0 | RunID, EquipID, Detector |
| dbo.ACM_FailureForecast | 9 | 113,064 | EquipID, RunID, Timestamp |
| dbo.ACM_FailureForecast_TS | 7 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_FailureHazard_TS | 8 | 0 | EquipID, RunID, Timestamp |
| dbo.ACM_FeatureDropLog | 8 | 0 | ID |
| dbo.ACM_ForecastState | 12 | 0 | EquipID, StateVersion |
| dbo.ACM_ForecastingState | 13 | 7 | EquipID, StateVersion |
| dbo.ACM_FusionQualityReport | 9 | 0 | — |
| dbo.ACM_HealthDistributionOverTime | 12 | 0 | — |
| dbo.ACM_HealthForecast | 10 | 113,064 | EquipID, RunID, Timestamp |
| dbo.ACM_HealthForecast_Continuous | 8 | 0 | EquipID, Timestamp, SourceRunID |
| dbo.ACM_HealthForecast_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_HealthHistogram | 5 | 0 | — |
| dbo.ACM_HealthTimeline | 8 | 190,982 | — |
| dbo.ACM_HealthZoneByPeriod | 9 | 0 | — |
| dbo.ACM_HistorianData | 7 | 204,067 | DataID |
| dbo.ACM_MaintenanceRecommendation | 8 | 0 | RunID, EquipID |
| dbo.ACM_OMRContributionsLong | 8 | 0 | — |
| dbo.ACM_OMRTimeline | 6 | 0 | — |
| dbo.ACM_OMR_Diagnostics | 15 | 13 | DiagnosticID |
| dbo.ACM_PCA_Loadings | 10 | 5,040 | RecordID |
| dbo.ACM_PCA_Metrics | 8 | 17 | ID |
| dbo.ACM_PCA_Models | 12 | 15 | RecordID |
| dbo.ACM_RUL | 18 | 430 | EquipID, RunID |
| dbo.ACM_RUL_Attribution | 9 | 0 | RunID, EquipID, FailureTime, SensorName |
| dbo.ACM_RUL_LearningState | 19 | 0 | EquipID |
| dbo.ACM_RUL_Summary | 15 | 0 | RunID, EquipID |
| dbo.ACM_RUL_TS | 9 | 0 | RunID, EquipID, Timestamp |
| dbo.ACM_RecommendedActions | 6 | 0 | RunID, EquipID, Action |
| dbo.ACM_RefitRequests | 10 | 76 | RequestID |
| dbo.ACM_RegimeDefinitions | 12 | 0 | ID |
| dbo.ACM_RegimeDwellStats | 8 | 0 | — |
| dbo.ACM_RegimeOccupancy | 9 | 0 | ID |
| dbo.ACM_RegimePromotionLog | 9 | 0 | ID |
| dbo.ACM_RegimeStability | 4 | 0 | — |
| dbo.ACM_RegimeState | 15 | 1 | EquipID, StateVersion |
| dbo.ACM_RegimeStats | 8 | 0 | — |
| dbo.ACM_RegimeTimeline | 5 | 296,615 | — |
| dbo.ACM_RegimeTransitions | 8 | 0 | ID |
| dbo.ACM_Regime_Episodes | 6 | 375 | Id |
| dbo.ACM_RunLogs | 25 | 656,958 | LogID |
| dbo.ACM_RunMetadata | 12 | 715 | RunMetadataID |
| dbo.ACM_RunMetrics | 5 | 105,396 | RunID, EquipID, MetricName |
| dbo.ACM_RunTimers | 7 | 205,327 | TimerID |
| dbo.ACM_Run_Stats | 13 | 15 | RecordID |
| dbo.ACM_Runs | 19 | 802 | RunID |
| dbo.ACM_SchemaVersion | 5 | 2 | VersionID |
| dbo.ACM_Scores_Long | 9 | 0 | Id |
| dbo.ACM_Scores_Wide | 15 | 298,298 | — |
| dbo.ACM_SeasonalPatterns | 10 | 0 | ID |
| dbo.ACM_SensorAnomalyByPeriod | 11 | 0 | — |
| dbo.ACM_SensorCorrelations | 8 | 191,844 | ID |
| dbo.ACM_SensorDefects | 11 | 3,165 | — |
| dbo.ACM_SensorForecast | 11 | 1,512 | RunID, EquipID, Timestamp, SensorName |
| dbo.ACM_SensorForecast_TS | 10 | 0 | RunID, EquipID, SensorName, Timestamp |
| dbo.ACM_SensorHotspotTimeline | 9 | 0 | — |
| dbo.ACM_SensorHotspots | 18 | 8,465 | — |
| dbo.ACM_SensorNormalized_TS | 8 | 472,446 | ID |
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
| dbo.ModelRegistry | 8 | 3,820 | ModelType, EquipID, Version |
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
**Row Count:** 0  

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

---


## dbo.ACM_AdaptiveConfig

**Primary Key:** ConfigID  
**Row Count:** 11  
**Date Range:** 2025-12-04 10:46:47 to 2025-12-25 17:52:16  

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
| 12 | 1 | fused_alert_z | 1.7893608808517456 | 0.0 | 999999.0 | True | 8749 | 0.0 | quantile_0.997: Auto-calculated from 8749 accumulated samples |

### Bottom 10 Records

| ConfigID | EquipID | ConfigKey | ConfigValue | MinBound | MaxBound | IsLearned | DataVolumeAtTuning | PerformanceMetric | ResearchReference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 1 | fused_warn_z | 0.8946804404258728 | 0.0 | 999999.0 | True | 8749 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 12 | 1 | fused_alert_z | 1.7893608808517456 | 0.0 | 999999.0 | True | 8749 | 0.0 | quantile_0.997: Auto-calculated from 8749 accumulated samples |
| 9 | NULL | auto_tune_data_threshold | 10000.0 | 5000.0 | 50000.0 | False | NULL | NULL | Expert tuning - Auto-tuning trigger |
| 8 | NULL | blend_tau_hours | 12.0 | 6.0 | 48.0 | False | NULL | NULL | Expert tuning - Warm-start alpha blending |
| 7 | NULL | monte_carlo_simulations | 1000.0 | 500.0 | 5000.0 | False | NULL | NULL | Saxena et al. (2008) - RUL simulation count |
| 6 | NULL | max_forecast_hours | 168.0 | 168.0 | 720.0 | False | NULL | NULL | Industry standard - 7-30 day horizon |
| 5 | NULL | confidence_min | 0.8 | 0.5 | 0.95 | False | NULL | NULL | Agresti & Coull (1998) - Statistical confidence |
| 4 | NULL | failure_threshold | 70.0 | 40.0 | 80.0 | False | NULL | NULL | ISO 13381-1:2015 - Health index threshold |
| 3 | NULL | training_window_hours | 168.0 | 72.0 | 720.0 | False | NULL | NULL | NIST SP 1225 - 3-30 day training window |
| 2 | NULL | beta | 0.1 | 0.01 | 0.3 | False | NULL | NULL | Hyndman & Athanasopoulos (2018) - Exponential smoothing trend |

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
**Row Count:** 375  
**Date Range:** 2024-10-03 19:00:00 to 2025-09-11 00:00:00  

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
| 1 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-10-03 19:00:00 | 2024-10-04 01:30:00 | info |
| 2 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-10-04 06:30:00 | 2024-10-04 08:00:00 | info |
| 3 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-10-05 09:30:00 | 2024-10-06 00:00:00 | info |
| 4 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-10-07 00:00:00 | 2024-10-07 02:30:00 | info |
| 5 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-10-07 09:30:00 | 2024-10-07 12:00:00 | info |
| 6 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-10-08 11:30:00 | 2024-10-08 14:00:00 | info |
| 7 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-10-09 22:30:00 | 2024-10-10 03:00:00 | info |
| 8 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-10-10 12:30:00 | 2024-10-13 06:30:00 | info |
| 9 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-11-01 07:00:00 | 2024-11-13 07:30:00 | info |
| 10 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2024-11-23 00:00:00 | 2024-12-04 12:30:00 | info |

### Bottom 10 Records

| Id | RunID | EquipID | StartTime | EndTime | Severity |
| --- | --- | --- | --- | --- | --- |
| 375 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-11 00:00:00 | 2025-09-12 07:00:00 | info |
| 374 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-08-12 00:00:00 | 2025-08-13 07:00:00 | info |
| 373 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-07-12 00:00:00 | 2025-07-13 10:00:00 | info |
| 372 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-06-12 00:00:00 | 2025-06-13 07:00:00 | info |
| 371 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-05-12 00:00:00 | 2025-05-13 07:30:00 | info |
| 370 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-04-11 00:00:00 | 2025-04-12 06:30:00 | info |
| 369 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-03-15 00:00:00 | 2025-03-15 04:00:00 | info |
| 368 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-03-14 03:00:00 | 2025-03-14 10:00:00 | info |
| 367 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-03-12 00:00:00 | 2025-03-13 06:30:00 | info |
| 366 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-02-23 22:00:00 | 2025-02-24 02:30:00 | info |

---


## dbo.ACM_AssetProfiles

**Primary Key:** ID  
**Row Count:** 0  

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

---


## dbo.ACM_BaselineBuffer

**Primary Key:** Id  
**Row Count:** 1,162,924  
**Date Range:** 2018-01-05 00:00:00 to 2025-09-14 23:00:00  

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
| 42448640 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 397.95001220703125 | NULL | 2025-12-26 10:14:28 |
| 42448639 | 1 | 2025-09-14 22:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 393.2699890136719 | NULL | 2025-12-26 10:14:28 |
| 42448638 | 1 | 2025-09-14 22:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 396.79998779296875 | NULL | 2025-12-26 10:14:28 |
| 42448637 | 1 | 2025-09-14 21:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 359.95001220703125 | NULL | 2025-12-26 10:14:28 |
| 42448636 | 1 | 2025-09-14 21:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 378.3500061035156 | NULL | 2025-12-26 10:14:28 |
| 42448635 | 1 | 2025-09-14 20:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 418.4100036621094 | NULL | 2025-12-26 10:14:28 |
| 42448634 | 1 | 2025-09-14 20:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 395.4700012207031 | NULL | 2025-12-26 10:14:28 |
| 42448633 | 1 | 2025-09-14 19:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 373.6099853515625 | NULL | 2025-12-26 10:14:28 |
| 42448632 | 1 | 2025-09-14 19:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 390.54998779296875 | NULL | 2025-12-26 10:14:28 |
| 42448631 | 1 | 2025-09-14 18:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 367.2799987792969 | NULL | 2025-12-26 10:14:28 |

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
**Date Range:** 2025-12-13 06:00:34 to 2025-12-25 05:36:52  

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
| 1 | score | COMPLETE | 1 | 2025-12-25 05:36:52 | 2025-12-25 05:36:53 | 2025-12-25 05:36:53 | 241 | 200 | 2023-10-15 00:00:00 |
| 2621 | score | COMPLETE | 1 | 2025-12-22 16:02:23 | 2025-12-22 16:02:24 | 2025-12-22 16:02:24 | 241 | 200 | 2023-10-15 00:00:00 |
| 5000 | score | COMPLETE | 1 | 2025-12-22 08:43:38 | 2025-12-22 08:43:40 | 2025-12-22 08:43:40 | 241 | 200 | 2022-08-04 06:10:00 |
| 5003 | score | COMPLETE | 1 | 2025-12-22 08:43:38 | 2025-12-22 08:43:39 | 2025-12-22 08:43:39 | 241 | 200 | 2022-04-27 03:00:00 |
| 5010 | score | COMPLETE | 1 | 2025-12-22 08:43:39 | 2025-12-22 08:43:41 | 2025-12-22 08:43:41 | 240 | 200 | 2022-10-09 08:40:00 |
| 5013 | score | COMPLETE | 1 | 2025-12-22 08:43:34 | 2025-12-22 08:43:36 | 2025-12-22 08:43:36 | 241 | 200 | 2022-04-30 13:20:00 |
| 5014 | score | COMPLETE | 1 | 2025-12-20 05:11:39 | 2025-12-20 05:11:40 | 2025-12-20 05:11:40 | 238 | 200 | 2022-03-03 14:00:00 |
| 5017 | score | COMPLETE | 1 | 2025-12-20 05:11:39 | 2025-12-20 05:11:40 | 2025-12-20 05:11:40 | 241 | 200 | 2022-10-31 15:20:00 |
| 5022 | score | COMPLETE | 1 | 2025-12-20 06:26:08 | 2025-12-20 06:26:08 | 2025-12-20 06:26:08 | 241 | 200 | 2022-08-12 09:50:00 |
| 5024 | score | COMPLETE | 1 | 2025-12-20 06:26:49 | 2025-12-20 06:26:49 | 2025-12-20 06:26:49 | 241 | 200 | 2022-04-24 15:00:00 |

### Bottom 10 Records

| EquipID | Stage | Status | AttemptCount | FirstAttemptAt | LastAttemptAt | CompletedAt | AccumulatedRows | RequiredRows | DataStartTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8634 | score | FAILED | 13 | 2025-12-20 04:25:53 | 2025-12-20 04:26:24 | 2025-12-20 04:25:53 | 241 | 200 | 2024-01-01 00:00:00 |
| 8632 | score | IN_PROGRESS | 15 | 2025-12-20 04:25:52 | 2025-12-20 04:26:22 | 2025-12-20 04:25:52 | 1719 | 200 | 2024-01-01 00:00:00 |
| 5092 | score | COMPLETE | 1 | 2025-12-13 06:00:34 | 2025-12-13 06:00:34 | 2025-12-13 06:00:34 | 241 | 200 | 2022-04-04 02:30:00 |
| 5026 | score | COMPLETE | 1 | 2025-12-20 07:58:45 | 2025-12-20 07:58:46 | 2025-12-20 07:58:46 | 230 | 200 | 2022-10-12 10:20:00 |
| 5025 | score | COMPLETE | 1 | 2025-12-20 07:41:35 | 2025-12-20 07:41:37 | 2025-12-20 07:41:37 | 241 | 200 | 2022-05-23 06:50:00 |
| 5024 | score | COMPLETE | 1 | 2025-12-20 06:26:49 | 2025-12-20 06:26:49 | 2025-12-20 06:26:49 | 241 | 200 | 2022-04-24 15:00:00 |
| 5022 | score | COMPLETE | 1 | 2025-12-20 06:26:08 | 2025-12-20 06:26:08 | 2025-12-20 06:26:08 | 241 | 200 | 2022-08-12 09:50:00 |
| 5017 | score | COMPLETE | 1 | 2025-12-20 05:11:39 | 2025-12-20 05:11:40 | 2025-12-20 05:11:40 | 241 | 200 | 2022-10-31 15:20:00 |
| 5014 | score | COMPLETE | 1 | 2025-12-20 05:11:39 | 2025-12-20 05:11:40 | 2025-12-20 05:11:40 | 238 | 200 | 2022-03-03 14:00:00 |
| 5013 | score | COMPLETE | 1 | 2025-12-22 08:43:34 | 2025-12-22 08:43:36 | 2025-12-22 08:43:36 | 241 | 200 | 2022-04-30 13:20:00 |

---


## dbo.ACM_Config

**Primary Key:** ConfigID  
**Row Count:** 342  
**Date Range:** 2025-12-09 12:47:06 to 2025-12-26 06:07:09  

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
| 847 | 5026 | runtime.tick_minutes | 5400 | int | 2025-12-20 07:58:37 | sql_batch_runner |
| 846 | 5025 | runtime.tick_minutes | 5499 | int | 2025-12-20 07:41:28 | sql_batch_runner |
| 845 | 5024 | runtime.tick_minutes | 1440 | int | 2025-12-22 11:18:44 | sql_batch_runner |
| 844 | 5022 | runtime.tick_minutes | 1440 | int | 2025-12-22 11:18:40 | sql_batch_runner |
| 843 | 5017 | runtime.tick_minutes | 1440 | int | 2025-12-22 11:18:40 | sql_batch_runner |
| 842 | 5014 | runtime.tick_minutes | 1440 | int | 2025-12-22 11:18:40 | sql_batch_runner |
| 841 | 5003 | runtime.tick_minutes | 1440 | int | 2025-12-22 08:43:17 | sql_batch_runner |
| 840 | 5013 | runtime.tick_minutes | 5615 | int | 2025-12-22 08:43:28 | sql_batch_runner |
| 839 | 0 | runtime.baseline.refresh_interval_batches | 10 | int | 2025-12-17 04:35:47 | B19cl3pc\bhadk |
| 838 | 0 | runtime.baseline.max_points | 100000 | int | 2025-12-17 04:35:47 | B19cl3pc\bhadk |

---


## dbo.ACM_ConfigHistory

**Primary Key:** ID  
**Row Count:** 3,495  
**Date Range:** 2025-12-03 14:40:59 to 2025-12-22 16:46:54  

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
| 13491 | 2025-12-22 16:46:54 | 5003 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 8742e7f7-41c9-42a0-ba93-b57fa7bc1d3c |
| 13490 | 2025-12-22 16:46:08 | 5000 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | fe7e5408-7ae9-46c9-b161-7f5af7366f38 |
| 13489 | 2025-12-22 16:46:07 | 5010 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 4da18918-f7ad-4be1-9322-9f5a1517bab0 |
| 13488 | 2025-12-22 16:43:52 | 5013 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | f1cfb725-ae18-4367-bb47-78140905a6b7 |
| 13487 | 2025-12-22 16:29:48 | 5010 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | fd4a83e2-5e8d-43af-9d02-fd1912bc4233 |
| 13486 | 2025-12-22 16:28:11 | 5003 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | afce4a51-d757-483c-82d8-2bedf83ca937 |
| 13485 | 2025-12-22 16:27:16 | 5000 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 40f66cb6-5298-4f23-8698-2225104c0e15 |
| 13484 | 2025-12-22 16:26:27 | 5013 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | ea5be767-e70a-454b-b0ed-e23bfb923915 |
| 13483 | 2025-12-22 16:12:10 | 5003 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 16ecdf24-6dc2-49c0-9c0a-8bae11e77646 |
| 13482 | 2025-12-22 16:10:16 | 5000 | regimes.auto_k.k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 2c1094ef-2e24-4a10-83b0-3f14dad1c90c |

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
**Row Count:** 0  

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

---


## dbo.ACM_DataQuality

**Primary Key:** No primary key  
**Row Count:** 7,195  
**Date Range:** 2022-04-04 02:30:00 to 2024-06-14 12:59:00  

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
| ACTTBTEMP1 | 47 | 0 | 0.0 | 1.6677963733673096 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B1RADVIBX | 47 | 0 | 0.0 | 0.0767005905508995 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B1RADVIBY | 47 | 0 | 0.0 | 0.0793059840798378 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B1TEMP1 | 47 | 0 | 0.0 | 2.9028818607330322 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B1VIB1 | 47 | 0 | 0.0 | 0.00032087290310300887 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B1VIB2 | 47 | 0 | 0.0 | 0.00036095670657232404 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B2RADVIBX | 47 | 0 | 0.0 | 0.12735570967197418 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B2RADVIBY | 47 | 0 | 0.0 | 0.08861265331506729 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B2TEMP1 | 47 | 0 | 0.0 | 2.840440273284912 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |
| B2VIB1 | 47 | 0 | 0.0 | 0.000871750176884234 | 0 | 0 | 2023-10-17 01:00:00 | 2023-10-18 00:00:00 | 47 |

### Bottom 10 Records

| sensor | train_count | train_nulls | train_null_pct | train_std | train_longest_gap | train_flatline_span | train_min_ts | train_max_ts | score_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| power_29_avg | 300 | 0 | 0.0 | 0.28070151805877686 | 0 | 39 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| power_29_max | 300 | 0 | 0.0 | 0.3581143915653229 | 0 | 32 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| power_29_min | 300 | 0 | 0.0 | 0.15604470670223236 | 0 | 62 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| power_29_std | 300 | 0 | 0.0 | 0.054747290909290314 | 0 | 32 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| power_30_avg | 300 | 0 | 0.0 | 0.28275102376937866 | 0 | 7 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| power_30_max | 300 | 0 | 0.0 | 0.3741777539253235 | 0 | 19 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| power_30_min | 300 | 0 | 0.0 | 0.1649143099784851 | 0 | 2 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| power_30_std | 300 | 0 | 0.0 | 0.05534970760345459 | 0 | 2 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| reactive_power_27_avg | 300 | 0 | 0.0 | 0.21538427472114563 | 0 | 55 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |
| reactive_power_27_max | 300 | 0 | 0.0 | 0.23632940649986267 | 0 | 55 | 2022-05-27 02:30:00 | 2022-05-29 04:20:00 | 550 |

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
**Row Count:** 3,577  
**Date Range:** 2025-12-25 06:56:18 to 2025-12-26 06:12:03  

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
| 1 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | ar1_z | ar1_z | 1.0 | 2025-12-25 06:56:18 |
| 2 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | ar1_z | pca_spe_z | 0.46192756595472817 | 2025-12-25 06:56:18 |
| 3 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | ar1_z | pca_t2_z | 0.6678743160683138 | 2025-12-25 06:56:18 |
| 4 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | ar1_z | iforest_z | 0.6526334232682297 | 2025-12-25 06:56:18 |
| 5 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | ar1_z | gmm_z | 0.5525943985960821 | 2025-12-25 06:56:18 |
| 6 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | ar1_z | omr_z | 0.4625937053180276 | 2025-12-25 06:56:18 |
| 7 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | ar1_z | cusum_z | 0.07756488312734423 | 2025-12-25 06:56:18 |
| 8 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | pca_spe_z | ar1_z | 0.46192756595472817 | 2025-12-25 06:56:18 |
| 9 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | pca_spe_z | pca_spe_z | 1.0 | 2025-12-25 06:56:18 |
| 10 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | pca_spe_z | pca_t2_z | 0.46977194322346805 | 2025-12-25 06:56:18 |

### Bottom 10 Records

| ID | RunID | EquipID | Detector1 | Detector2 | Correlation | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 3577 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | cusum_z | cusum_z | 1.0 | 2025-12-26 06:12:03 |
| 3576 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | cusum_z | omr_z | 0.14527505049431036 | 2025-12-26 06:12:03 |
| 3575 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | cusum_z | gmm_z | 0.08986819686214732 | 2025-12-26 06:12:03 |
| 3574 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | cusum_z | iforest_z | 0.08644924274712142 | 2025-12-26 06:12:03 |
| 3573 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | cusum_z | pca_t2_z | 0.052878279625317895 | 2025-12-26 06:12:03 |
| 3572 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | cusum_z | pca_spe_z | 0.12413327692833613 | 2025-12-26 06:12:03 |
| 3571 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | cusum_z | ar1_z | 0.07116125066304928 | 2025-12-26 06:12:03 |
| 3570 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | omr_z | cusum_z | 0.14527505049431036 | 2025-12-26 06:12:03 |
| 3569 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | omr_z | omr_z | 1.0 | 2025-12-26 06:12:03 |
| 3568 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | omr_z | gmm_z | 0.9034896743694014 | 2025-12-26 06:12:03 |

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
**Row Count:** 5,108  
**Date Range:** 2025-12-20 07:51:55 to 2025-12-26 06:16:15  

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
| 141818 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 1 | Time-Series Anomaly (AR1) | NULL | 23.085338592529297 | 1 | 2025-12-20 07:51:55 | 5025 |
| 141819 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 1 | Correlation Break (PCA-SPE) | NULL | 22.35803985595703 | 2 | 2025-12-20 07:51:55 | 5025 |
| 141820 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 1 | Density Anomaly (GMM) | NULL | 19.251317977905273 | 3 | 2025-12-20 07:51:55 | 5025 |
| 141821 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 1 | Baseline Consistency (OMR) | NULL | 18.43108367919922 | 4 | 2025-12-20 07:51:55 | 5025 |
| 141822 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 1 | cusum_z | NULL | 6.5168867111206055 | 5 | 2025-12-20 07:51:55 | 5025 |
| 141823 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 1 | Rare State (IsolationForest) | NULL | 6.237335681915283 | 6 | 2025-12-20 07:51:55 | 5025 |
| 141824 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 1 | Multivariate Outlier (PCA-T2) | NULL | 4.119987964630127 | 7 | 2025-12-20 07:51:55 | 5025 |
| 141825 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 2 | Correlation Break (PCA-SPE) | NULL | 40.368324279785156 | 1 | 2025-12-20 07:51:55 | 5025 |
| 141826 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 2 | Multivariate Outlier (PCA-T2) | NULL | 40.368324279785156 | 2 | 2025-12-20 07:51:55 | 5025 |
| 141827 | 76FEF25C-88A8-4FE2-85EC-E15BE10FF09B | 2 | cusum_z | NULL | 7.585345268249512 | 3 | 2025-12-20 07:51:55 | 5025 |

### Bottom 10 Records

| ID | RunID | EpisodeID | DetectorType | SensorName | ContributionPct | Rank | CreatedAt | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 150530 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 25 | Baseline Consistency (OMR) | NULL | 11.331165313720703 | 6 | 2025-12-26 06:16:15 | 1 |
| 150529 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 25 | Rare State (IsolationForest) | NULL | 15.979448318481445 | 5 | 2025-12-26 06:16:15 | 1 |
| 150528 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 25 | Time-Series Anomaly (AR1) | NULL | 16.207496643066406 | 4 | 2025-12-26 06:16:15 | 1 |
| 150527 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 25 | Density Anomaly (GMM) | NULL | 18.53544044494629 | 3 | 2025-12-26 06:16:15 | 1 |
| 150526 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 25 | Multivariate Outlier (PCA-T2) | NULL | 18.53544044494629 | 2 | 2025-12-26 06:16:15 | 1 |
| 150525 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 25 | Correlation Break (PCA-SPE) | NULL | 18.53544044494629 | 1 | 2025-12-26 06:16:15 | 1 |
| 150524 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 24 | cusum_z | NULL | 1.0491502285003662 | 7 | 2025-12-26 06:16:15 | 1 |
| 150523 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 24 | Baseline Consistency (OMR) | NULL | 11.070684432983398 | 6 | 2025-12-26 06:16:15 | 1 |
| 150522 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 24 | Rare State (IsolationForest) | NULL | 15.853971481323242 | 5 | 2025-12-26 06:16:15 | 1 |
| 150521 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 24 | Time-Series Anomaly (AR1) | NULL | 17.698152542114258 | 4 | 2025-12-26 06:16:15 | 1 |

---


## dbo.ACM_EpisodeDiagnostics

**Primary Key:** ID  
**Row Count:** 425  
**Date Range:** 2024-10-03 19:00:00 to 2025-09-11 00:00:00  

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
| 145 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 1 | 2024-10-03 19:00:00 | 2024-10-04 01:30:00 | 6.5 | 1.5907792385206019 | 1.4495876546087534 | LOW |
| 146 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 2 | 2024-10-04 06:30:00 | 2024-10-04 08:00:00 | 1.5 | 1.5887340159306884 | 1.3582928523405462 | LOW |
| 147 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 3 | 2024-10-05 09:30:00 | 2024-10-06 00:00:00 | 14.5 | 1.566529533573589 | 1.5256429323628322 | LOW |
| 148 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 4 | 2024-10-07 00:00:00 | 2024-10-07 02:30:00 | 2.5 | 1.5724494641057625 | 1.5484483946261693 | LOW |
| 149 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 5 | 2024-10-07 09:30:00 | 2024-10-07 12:00:00 | 2.5 | 1.3123133995571967 | 1.1146128498824421 | LOW |
| 150 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 6 | 2024-10-08 11:30:00 | 2024-10-08 14:00:00 | 2.5 | 1.5658636622087676 | 1.550270296996495 | LOW |
| 151 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 7 | 2024-10-09 22:30:00 | 2024-10-10 03:00:00 | 4.5 | 1.7384634776280046 | 1.5052473641151547 | LOW |
| 152 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 8 | 2024-10-10 12:30:00 | 2024-10-13 06:30:00 | 66.0 | 1.7893609166449744 | 1.6001595073687644 | LOW |
| 153 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 9 | 2024-11-01 07:00:00 | 2024-11-13 07:30:00 | 288.5 | 1.7893609166449744 | 1.3844467670631948 | LOW |
| 154 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 10 | 2024-11-23 00:00:00 | 2024-12-04 12:30:00 | 276.5 | 1.7893609166449744 | 1.550449453888558 | LOW |

### Bottom 10 Records

| ID | RunID | EquipID | EpisodeID | StartTime | EndTime | DurationHours | PeakZ | AvgZ | Severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 569 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 25 | 2025-09-11 00:00:00 | 2025-09-12 07:00:00 | 31.0 | 1.7893609166449744 | 1.6569926293136485 | LOW |
| 568 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 24 | 2025-08-12 00:00:00 | 2025-08-13 07:00:00 | 31.0 | 1.7893609166449744 | 1.7153269468726637 | LOW |
| 567 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 23 | 2025-07-12 00:00:00 | 2025-07-13 10:00:00 | 34.0 | 1.7893609166449744 | 1.6787804454862407 | LOW |
| 566 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 22 | 2025-06-12 00:00:00 | 2025-06-13 07:00:00 | 31.0 | 1.7893609166449744 | 1.7121320047349415 | LOW |
| 565 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 21 | 2025-05-12 00:00:00 | 2025-05-13 07:30:00 | 31.5 | 1.7893609166449744 | 1.6830956134444195 | LOW |
| 564 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 20 | 2025-04-11 00:00:00 | 2025-04-12 06:30:00 | 30.5 | 1.7893609166449744 | 1.6624835209110491 | LOW |
| 563 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 19 | 2025-03-15 00:00:00 | 2025-03-15 04:00:00 | 4.0 | 1.502840949988757 | 1.289215237184902 | LOW |
| 562 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 18 | 2025-03-14 03:00:00 | 2025-03-14 10:00:00 | 7.0 | 1.5843207038607716 | 1.0835809471090847 | LOW |
| 561 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 17 | 2025-03-12 00:00:00 | 2025-03-13 06:30:00 | 30.5 | 1.7893609166449744 | 1.5739801434271905 | LOW |
| 560 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 16 | 2025-02-23 22:00:00 | 2025-02-24 02:30:00 | 4.5 | 1.3695584829504464 | 0.9716414932999035 | LOW |

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
**Row Count:** 174  

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
| 01434C28-7E92-4F95-921C-03AB03F5169B | 2621 | 1 | 150.0 | NULL | NULL | 0.860022160478252 | 0.5727634360701214 |
| 9FCD79AB-B6BD-479A-B248-08310A5EA8BF | 2621 | 1 | 210.0 | NULL | NULL | 1.333583709690176 | 0.5429477777430093 |
| 97F1E50A-C859-4497-A6F8-0898A7CBDAD9 | 5010 | 7 | 90.0 | NULL | NULL | 1.2009846449762915 | 0.7725041408682278 |
| D33BC3BB-2918-493E-9D83-08D4AF44CDB0 | 2621 | 2 | 60.0 | NULL | NULL | 1.859461901724742 | 1.0487426178069652 |
| 7777B310-0063-41C4-B5F4-0953ABC2D450 | 2621 | 2 | 165.0 | NULL | NULL | 2.0795061759810296 | 0.9279856553113313 |
| E2E92EB7-FEC0-466A-A792-0955C714592F | 5013 | 5 | 250.0 | NULL | NULL | 0.9754589677032719 | 0.4974348366870015 |
| 119A0E03-7732-4BA4-A344-0D27EBA12FDF | 1 | 25 | 1830.0 | NULL | NULL | 1.7893609166449744 | 1.4524301320859399 |
| 6084CDE3-D2C8-4107-BF43-0F91EBC48DE5 | 2621 | 1 | 240.0 | NULL | NULL | 2.413417968782031 | 1.3645043677281243 |
| 42EF9AC2-0934-464F-A546-126C13073D71 | 2621 | 1 | 180.0 | NULL | NULL | 1.2780286494437278 | 0.8250214785791739 |

### Bottom 10 Records

| RunID | EquipID | EpisodeCount | MedianDurationMinutes | CoveragePct | TimeInAlertPct | MaxFusedZ | AvgFusedZ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F8F21D38-676D-497F-B9C9-FEF2CAE2B0E7 | 5000 | 6 | 175.0 | NULL | NULL | 1.1061586061484665 | 0.6945518471096924 |
| FD4A83E2-5E8D-43AF-9D02-FD1912BC4233 | 5010 | 9 | 60.0 | NULL | NULL | 1.015234384477372 | 0.6685333788472316 |
| 952ED5C7-71D9-472A-8C40-FD0A3D3967EA | 5010 | 5 | 550.0 | NULL | NULL | 0.9685877385701352 | 0.5815487172809679 |
| C550A327-2035-436D-89E3-FC73503BCFF9 | 2621 | 2 | 75.0 | NULL | NULL | 1.660910998530984 | 0.8943816759125716 |
| C8605AC2-4416-4FA2-A551-FA89B8A0FECB | 1 | 25 | 1830.0 | NULL | NULL | 1.7893609166449744 | 1.4524301320859399 |
| C0F2873D-E05B-475B-80B3-FA4DF917FFC6 | 1 | 25 | 1830.0 | NULL | NULL | 1.7893609166449744 | 1.4524301320859399 |
| 4D6E3870-9B7A-44ED-A5FB-F9CC475B3DBB | 2621 | 1 | 60.0 | NULL | NULL | 1.5479631583900786 | 0.7802746526835082 |
| 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 25 | 1830.0 | NULL | NULL | 1.7893609166449744 | 1.4524301320859399 |
| 28037F72-019C-4E3C-8FE4-F88DC498A9D1 | 2621 | 2 | 90.0 | NULL | NULL | 1.1213726360081764 | 0.8341633593702674 |
| 5A7CF713-C390-4883-AB54-F768C96C2BB2 | 5003 | 3 | 640.0 | NULL | NULL | 1.1938944009963324 | 0.8693554059645727 |

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
**Row Count:** 113,064  
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
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 00:00:00 | 4.6194817205810365e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 00:30:00 | 4.562407810126338e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 01:00:00 | 4.506019627119422e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 01:30:00 | 4.450309179246367e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 02:00:00 | 4.3952685642919516e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 02:30:00 | 4.340889969160759e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 03:00:00 | 4.287165668908412e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 03:30:00 | 4.234088025783484e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 04:00:00 | 4.181649488278369e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 04:30:00 | 4.129842590191048e-09 | 0.9999999953805183 | 0.0 | 50.0 | GaussianTail | 2025-12-25 15:45:05 |

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
**Row Count:** 0  

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
**Row Count:** 7  

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
| 1 | 1 | {"alpha": 0.95, "beta": 0.01, "level": 92.02134109523143, "trend": 0.03166752238887965, "std_erro... | {"forecast_mean": 97.02045737492202, "forecast_std": 2.634267518827345, "forecast_range": 7.94699... | NULL |  | 219466 | 2.634267518827345 | NULL | NULL |
| 2621 | 1 | {"alpha": 0.4, "beta": 0.01, "level": 88.5946945323379, "trend": -0.023843931001453628, "std_erro... | {"forecast_mean": 84.57699215859296, "forecast_std": 2.3127281542089166, "forecast_range": 7.9877... | NULL |  | 218581 | 2.3127281542089166 | NULL | NULL |
| 5000 | 1 | {"alpha": 0.95, "beta": 0.15, "level": 91.91003625934573, "trend": 0.22183455193897195, "std_erro... | {"forecast_mean": 99.85764198107917, "forecast_std": 0.8583952949455335, "forecast_range": 7.8681... | NULL |  | 20571 | 0.8583952949455335 | NULL | NULL |
| 5003 | 1 | {"alpha": 0.95, "beta": 0.01, "level": 94.33245174585916, "trend": 0.04066454624832299, "std_erro... | {"forecast_mean": 99.61098891633989, "forecast_std": 1.1459518614576236, "forecast_range": 5.6268... | NULL |  | 20811 | 1.1459518614576236 | NULL | NULL |
| 5010 | 1 | {"alpha": 0.05, "beta": 0.01, "level": 89.75279124465042, "trend": 0.0013834515872053412, "std_er... | {"forecast_mean": 90.45074257039552, "forecast_std": 0.40256281962775325, "forecast_range": 1.393... | NULL |  | 20876 | 0.40256281962775325 | NULL | NULL |
| 5013 | 1 | {"alpha": 0.2, "beta": 0.03, "level": 93.34754564032983, "trend": 0.06747191223493719, "std_error... | {"forecast_mean": 99.6779421815516, "forecast_std": 1.1477441244609181, "forecast_range": 6.58498... | NULL |  | 20887 | 1.1477441244609181 | NULL | NULL |
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
**Row Count:** 113,064  
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
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 00:00:00 | 91.99776992025515 | 88.77469526117261 | 95.2208445793377 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 00:30:00 | 92.00582566839746 | 88.77864380317348 | 95.23300753362145 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 01:00:00 | 92.01388141653977 | 88.78251598902904 | 95.24524684405051 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 01:30:00 | 92.02193716468209 | 88.78631131227446 | 95.25756301708971 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 02:00:00 | 92.0299929128244 | 88.79002927455525 | 95.26995655109354 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 02:30:00 | 92.03804866096671 | 88.79366938569234 | 95.28242793624108 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 03:00:00 | 92.04610440910903 | 88.79723116374463 | 95.29497765447344 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 03:30:00 | 92.05416015725135 | 88.80071413506876 | 95.30760617943393 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 04:00:00 | 92.06221590539366 | 88.80411783437614 | 95.32031397641117 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 2024-10-06 04:30:00 | 92.07027165353597 | 88.80744180478705 | 95.33310150228489 | 1.611537329541271 | ExponentialSmoothing | 2025-12-25 15:45:03 | NULL |

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
**Row Count:** 190,982  
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
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | C8605AC2-4416-4FA2-A551-FA89B8A0FECB | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | 93F076B5-577F-4036-8AAC-7E6653AFD094 | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | 375CF0E4-D46E-486D-AF42-811C0A5580F1 | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | F2170C61-A77C-49BC-98B0-71AA76951952 | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | 119A0E03-7732-4BA4-A344-0D27EBA12FDF | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | 9BB8853B-A058-4772-B52E-C943EC04E277 | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | D18C67FB-DFCA-41B0-B309-9A5F86B8FBCD | 1 | 92.02999877929688 | NORMAL |
| 2025-09-14 23:00:00 | 92.02 | GOOD | -0.4618000090122223 | E210DE1F-F6FB-44C9-A3F9-A50D0B7FA1AA | 1 | 92.02999877929688 | NORMAL |

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
**Row Count:** 13  

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
| 1 | d18c67fb-dfca-41b0-b309-9a5f86b8fbcd | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 2 | 375cf0e4-d46e-486d-af42-811c0a5580f1 | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 3 | 56255fa1-61a5-4556-b8bd-e3a19569661e | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 4 | 53e5f267-0e88-4f56-a07e-a396727c1150 | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 5 | a53903df-560f-4076-a5a6-d34f64e9e51f | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 6 | f2170c61-a77c-49bc-98b0-71aa76951952 | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 7 | e210de1f-f6fb-44c9-a3f9-a50d0b7fa1aa | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 8 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 9 | f5752054-9219-4dfe-a9d0-165f32994eda | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 10 | c8605ac2-4416-4fa2-a551-fa89b8a0fecb | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |

### Bottom 10 Records

| DiagnosticID | RunID | EquipID | ModelType | NComponents | TrainSamples | TrainFeatures | TrainResidualStd | TrainStartTime | TrainEndTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 12 | 9bb8853b-a058-4772-b52e-c943ec04e277 | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 11 | c0f2873d-e05b-475b-80b3-fa4df917ffc6 | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 10 | c8605ac2-4416-4fa2-a551-fa89b8a0fecb | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 9 | f5752054-9219-4dfe-a9d0-165f32994eda | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 8 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 7 | e210de1f-f6fb-44c9-a3f9-a50d0b7fa1aa | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 6 | f2170c61-a77c-49bc-98b0-71aa76951952 | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 5 | a53903df-560f-4076-a5a6-d34f64e9e51f | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |
| 4 | 53e5f267-0e88-4f56-a07e-a396727c1150 | 1 | pls | 5 | 8749 | 72 | 1.1448313171751956 | NULL | NULL |

---


## dbo.ACM_PCA_Loadings

**Primary Key:** RecordID  
**Row Count:** 5,040  
**Date Range:** 2025-12-25 17:55:37 to 2025-12-26 11:46:02  

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
| 1 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06G31_1FD Fan Damper Position_med | 0.23725233195914316 | 2025-12-25 17:55:40 |
| 2 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_med | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_med | 0.23063721110930158 | 2025-12-25 17:55:40 |
| 3 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.06I03_1FD Fan Motor Current_med | DEMO.SIM.06I03_1FD Fan Motor Current_med | 0.23328489413889641 | 2025-12-25 17:55:40 |
| 4 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.06T31_1FD Fan Inlet Temperature_med | DEMO.SIM.06T31_1FD Fan Inlet Temperature_med | 0.005894088145149204 | 2025-12-25 17:55:40 |
| 5 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_med | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_med | 0.09224670510839353 | 2025-12-25 17:55:40 |
| 6 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_med | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_med | 0.0713896679672681 | 2025-12-25 17:55:40 |
| 7 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_med | DEMO.SIM.06T34_1FD Fan Outlet Termperature_med | 0.08248458652293224 | 2025-12-25 17:55:40 |
| 8 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_med | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_med | 0.24125680576391623 | 2025-12-25 17:55:40 |
| 9 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_med | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_med | 0.23960697528288746 | 2025-12-25 17:55:40 |
| 10 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 1 | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_mad | DEMO.SIM.06G31_1FD Fan Damper Position_mad | 0.1032750343350841 | 2025-12-25 17:55:40 |

### Bottom 10 Records

| RecordID | RunID | EquipID | EntryDateTime | ComponentNo | ComponentID | Sensor | FeatureName | Loading | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5040 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 0.17609101759334406 | 2025-12-26 11:46:05 |
| 5039 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | 0.17993886791299363 | 2025-12-26 11:46:05 |
| 5038 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | -0.039855281596838404 | 2025-12-26 11:46:05 |
| 5037 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | -0.1175839137596772 | 2025-12-26 11:46:05 |
| 5036 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_rz | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_rz | -0.11066922867173945 | 2025-12-26 11:46:05 |
| 5035 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.06T31_1FD Fan Inlet Temperature_rz | DEMO.SIM.06T31_1FD Fan Inlet Temperature_rz | -0.12075107012805397 | 2025-12-26 11:46:05 |
| 5034 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.06I03_1FD Fan Motor Current_rz | DEMO.SIM.06I03_1FD Fan Motor Current_rz | 0.19803509613756995 | 2025-12-26 11:46:05 |
| 5033 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_rz | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_rz | 0.11775297643204817 | 2025-12-26 11:46:05 |
| 5032 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.06G31_1FD Fan Damper Position_rz | DEMO.SIM.06G31_1FD Fan Damper Position_rz | 0.169515920877398 | 2025-12-26 11:46:05 |
| 5031 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | 5 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_kurt | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_kurt | 0.14152414395880888 | 2025-12-26 11:46:05 |

---


## dbo.ACM_PCA_Metrics

**Primary Key:** ID  
**Row Count:** 17  
**Date Range:** 2025-12-25 11:54:25 to 2025-12-26 06:07:57  

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
| 1 | fb830dae-9960-416d-b17b-ca50b1238466 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 11:54:25 |
| 2 | 542592bf-a9c7-4644-af53-f94e70198a1d | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 12:05:59 |
| 3 | ced978ec-673b-4014-a5c8-950617fd87a4 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 12:18:56 |
| 4 | 93f076b5-577f-4036-8aac-7e6653afd094 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 12:27:36 |
| 5 | d18c67fb-dfca-41b0-b309-9a5f86b8fbcd | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 12:39:19 |
| 6 | 375cf0e4-d46e-486d-af42-811c0a5580f1 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 12:49:46 |
| 7 | 56255fa1-61a5-4556-b8bd-e3a19569661e | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 13:01:15 |
| 8 | 53e5f267-0e88-4f56-a07e-a396727c1150 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 13:11:48 |
| 9 | a53903df-560f-4076-a5a6-d34f64e9e51f | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 04:17:57 |
| 10 | f2170c61-a77c-49bc-98b0-71aa76951952 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 04:25:17 |

### Bottom 10 Records

| ID | RunID | EquipID | ComponentIndex | ExplainedVariance | CumulativeVariance | Eigenvalue | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 06:07:57 |
| 16 | 9bb8853b-a058-4772-b52e-c943ec04e277 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 05:50:24 |
| 15 | c0f2873d-e05b-475b-80b3-fa4df917ffc6 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 05:38:19 |
| 14 | c8605ac2-4416-4fa2-a551-fa89b8a0fecb | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 05:28:07 |
| 13 | f5752054-9219-4dfe-a9d0-165f32994eda | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 05:12:23 |
| 12 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 04:55:56 |
| 11 | e210de1f-f6fb-44c9-a3f9-a50d0b7fa1aa | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 04:39:47 |
| 10 | f2170c61-a77c-49bc-98b0-71aa76951952 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 04:25:17 |
| 9 | a53903df-560f-4076-a5a6-d34f64e9e51f | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-26 04:17:57 |
| 8 | 53e5f267-0e88-4f56-a07e-a396727c1150 | 1 | 0 | 0.5847637327423546 | NULL | 72.0 | 2025-12-25 13:11:48 |

---


## dbo.ACM_PCA_Models

**Primary Key:** RecordID  
**Row Count:** 15  
**Date Range:** 2025-12-25 17:42:07 to 2025-12-26 11:46:02  

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
| 1 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2025-12-25 17:42:07 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 2 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2025-12-25 17:55:37 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 3 | 93F076B5-577F-4036-8AAC-7E6653AFD094 | 1 | 2025-12-25 18:04:44 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 4 | D18C67FB-DFCA-41B0-B309-9A5F86B8FBCD | 1 | 2025-12-25 18:16:49 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 5 | 375CF0E4-D46E-486D-AF42-811C0A5580F1 | 1 | 2025-12-25 18:27:58 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 6 | 56255FA1-61A5-4556-B8BD-E3A19569661E | 1 | 2025-12-25 18:39:45 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 7 | 53E5F267-0E88-4F56-A07E-A396727C1150 | 1 | 2025-12-25 18:50:12 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 8 | F2170C61-A77C-49BC-98B0-71AA76951952 | 1 | 2025-12-26 10:05:29 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 9 | E210DE1F-F6FB-44C9-A3F9-A50D0B7FA1AA | 1 | 2025-12-26 10:21:49 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 10 | 119A0E03-7732-4BA4-A344-0D27EBA12FDF | 1 | 2025-12-26 10:38:32 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |

### Bottom 10 Records

| RecordID | RunID | EquipID | EntryDateTime | NComponents | TargetVar | VarExplainedJSON | ScalingSpecJSON | ModelVersion | TrainStartEntryDateTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-12-26 11:46:02 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 14 | 9BB8853B-A058-4772-B52E-C943EC04E277 | 1 | 2025-12-26 11:28:06 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 13 | C0F2873D-E05B-475B-80B3-FA4DF917FFC6 | 1 | 2025-12-26 11:15:46 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 12 | C8605AC2-4416-4FA2-A551-FA89B8A0FECB | 1 | 2025-12-26 11:05:11 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 11 | F5752054-9219-4DFE-A9D0-165F32994EDA | 1 | 2025-12-26 10:55:23 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 10 | 119A0E03-7732-4BA4-A344-0D27EBA12FDF | 1 | 2025-12-26 10:38:32 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 9 | E210DE1F-F6FB-44C9-A3F9-A50D0B7FA1AA | 1 | 2025-12-26 10:21:49 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 8 | F2170C61-A77C-49BC-98B0-71AA76951952 | 1 | 2025-12-26 10:05:29 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 7 | 53E5F267-0E88-4F56-A07E-A396727C1150 | 1 | 2025-12-25 18:50:12 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |
| 6 | 56255FA1-61A5-4556-B8BD-E3A19569661E | 1 | 2025-12-25 18:39:45 | 5 | {"SPE_P95_train": 5.1218438148498535, "T2_P95_train": 4.32567024230957} | [0.19102631497606973, 0.1368361474759876, 0.10592943138675423, 0.0837418413033619, 0.067229997600... | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} | v10.1.0 | 2023-10-15 00:00:00 |

---


## dbo.ACM_RUL

**Primary Key:** EquipID, RunID  
**Row Count:** 430  
**Date Range:** 2025-12-14 03:55:51 to 2026-01-02 11:45:09  

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
| 1 | 198EC38B-24B0-48F7-A901-003DF53343A1 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2026-01-01 15:45:07 | Multipath | 1000 |
| 1 | C0508133-8393-44C3-8F0B-077AD1F60518 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.5825018304475389 | 2026-01-01 15:50:50 | Multipath | 1000 |
| 1 | 119A0E03-7732-4BA4-A344-0D27EBA12FDF | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2026-01-02 10:38:25 | Multipath | 1000 |
| 1 | 91B3C5CC-5D51-4EF3-AA75-11DC4918CBF4 | 4.5 | 0.9866975102914279 | 4.5 | 8.613071162522862 | 0.42131843549701437 | 2025-12-25 18:57:58 | Multipath | 1000 |
| 1 | F5752054-9219-4DFE-A9D0-165F32994EDA | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2026-01-02 10:55:16 | Multipath | 1000 |
| 1 | D623D9FB-F56C-4D19-B180-19AD2284C09F | 168.0 | 80.51451683978054 | 168.0 | 170.2348182710401 | 0.5165273455449774 | 2026-01-01 16:56:11 | Multipath | 1000 |
| 1 | CE7FF921-F158-42E5-BA1D-229D40168006 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2026-01-01 13:55:05 | Multipath | 1000 |
| 1 | 99D99403-6215-4BE5-85D7-23CD30635A9F | 66.5 | 10.853672613205708 | 66.5 | 170.2348182710401 | 0.3882518577174355 | 2025-12-28 07:36:27 | Multipath | 1000 |
| 1 | 2F886D42-DE74-4ABB-AB82-2D907B9814D4 | 168.0 | 165.7651817289599 | 168.0 | 170.2348182710401 | 0.6 | 2026-01-01 13:44:34 | Multipath | 1000 |
| 1 | D9E32D97-9B0B-4A57-A39C-30536D9576B5 | 122.5 | 97.68305351885137 | 122.5 | 142.87565104890865 | 0.5589916723911872 | 2025-12-30 18:42:58 | Multipath | 1000 |

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
**Row Count:** 76  
**Date Range:** 2025-12-25 05:21:58 to 2025-12-26 06:11:41  

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
| 1 | 1 | 2025-12-25 05:21:58 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 05:29:01 |
| 2 | 1 | 2025-12-25 05:30:25 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 05:37:13 |
| 3 | 1 | 2025-12-25 05:38:35 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 06:54:25 |
| 4 | 1 | 2025-12-25 06:56:08 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 06:59:42 |
| 5 | 1 | 2025-12-25 07:01:05 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 07:03:25 |
| 6 | 1 | 2025-12-25 07:04:46 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 07:07:42 |
| 7 | 1 | 2025-12-25 07:09:01 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 07:11:38 |
| 8 | 1 | 2025-12-25 07:13:02 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 07:16:29 |
| 9 | 1 | 2025-12-25 07:17:57 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 07:21:17 |
| 10 | 1 | 2025-12-25 07:22:45 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-25 07:25:24 |

### Bottom 10 Records

| RequestID | EquipID | RequestedAt | Reason | AnomalyRate | DriftScore | ModelAgeHours | RegimeQuality | Acknowledged | AcknowledgedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 76 | 1 | 2025-12-26 06:11:41 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |
| 75 | 1 | 2025-12-26 05:54:08 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 06:07:46 |
| 74 | 1 | 2025-12-26 05:42:00 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 05:50:14 |
| 73 | 1 | 2025-12-26 05:31:42 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 05:38:07 |
| 72 | 1 | 2025-12-26 05:16:08 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 05:27:56 |
| 71 | 1 | 2025-12-26 04:59:34 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 05:12:13 |
| 70 | 1 | 2025-12-26 04:43:29 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 04:55:46 |
| 69 | 1 | 2025-12-26 04:29:07 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 04:39:36 |
| 68 | 1 | 2025-12-26 04:21:42 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 04:25:07 |
| 67 | 1 | 2025-12-25 13:15:29 | Anomaly rate too high; Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2025-12-26 04:17:45 |

---


## dbo.ACM_RegimeDefinitions

**Primary Key:** ID  
**Row Count:** 0  

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
**Row Count:** 1  
**Date Range:** 2025-12-26 06:10:41 to 2025-12-26 06:10:41  

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
| 1 | 1 | 3 | [[-0.3959299016130773, -0.3949126718764154, -1.0419757840109494, -0.9975409258739362, -1.06814680... | [] | [] | [] | [] | 0 | 0.3233703364570749 |

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
**Row Count:** 296,615  
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
| 2025-09-14 23:00:00 | 2 | unknown | 9BB8853B-A058-4772-B52E-C943EC04E277 | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | C0F2873D-E05B-475B-80B3-FA4DF917FFC6 | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | FB830DAE-9960-416D-B17B-CA50B1238466 | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | C8605AC2-4416-4FA2-A551-FA89B8A0FECB | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | A53903DF-560F-4076-A5A6-D34F64E9E51F | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | F2170C61-A77C-49BC-98B0-71AA76951952 | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | E210DE1F-F6FB-44C9-A3F9-A50D0B7FA1AA | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | 93F076B5-577F-4036-8AAC-7E6653AFD094 | 1 |
| 2025-09-14 23:00:00 | 2 | unknown | 56255FA1-61A5-4556-B8BD-E3A19569661E | 1 |

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
**Row Count:** 375  

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
| 1 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 2 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 3 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 4 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 5 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 6 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 7 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 8 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 9 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |
| 10 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | NULL | NULL | NULL |

### Bottom 10 Records

| Id | RunID | EquipID | StartTime | EndTime | RegimeID |
| --- | --- | --- | --- | --- | --- |
| 375 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 374 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 373 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 372 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 371 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 370 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 369 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 368 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 367 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |
| 366 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | NULL | NULL | NULL |

---


## dbo.ACM_RunLogs

**Primary Key:** LogID  
**Row Count:** 656,958  
**Date Range:** 2025-12-02 05:59:43 to 2025-12-17 16:27:35  

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
**Row Count:** 105,396  
**Date Range:** 2025-12-01 17:15:57 to 2025-12-26 11:40:58  

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
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.n_samples.ar1_z | 540.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.n_samples.gmm_z | 540.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.n_samples.iforest_z | 540.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.n_samples.omr_z | 540.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.n_samples.pca_spe_z | 540.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.n_samples.pca_t2_z | 540.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.quality.ar1_z | 0.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.quality.gmm_z | 0.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.quality.iforest_z | 0.0 | 2025-12-19 14:24:24 |
| 000128aa-6556-406d-b599-168a13de7880 | 5026 | fusion.quality.omr_z | 0.0 | 2025-12-19 14:24:24 |

### Bottom 10 Records

| RunID | EquipID | MetricName | MetricValue | Timestamp |
| --- | --- | --- | --- | --- |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.weight.pca_t2_z | 0.19 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.weight.pca_spe_z | 0.26000000000000006 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.weight.omr_z | 0.12000000000000001 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.weight.iforest_z | 0.15500000000000003 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.weight.gmm_z | 0.085 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.weight.ar1_z | 0.19 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.quality.pca_t2_z | 0.0 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.quality.pca_spe_z | 0.0 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.quality.omr_z | 0.0 | 2025-12-19 15:36:31 |
| fff350ae-6407-47bf-8ebc-5916d9fadac8 | 5010 | fusion.quality.iforest_z | 0.0 | 2025-12-19 15:36:31 |

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
**Row Count:** 15  
**Date Range:** 2023-10-15 00:00:00 to 2023-10-15 00:00:00  

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
| 1 | 542592BF-A9C7-4644-AF53-F94E70198A1D | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 2 | CED978EC-673B-4014-A5C8-950617FD87A4 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 3 | 93F076B5-577F-4036-8AAC-7E6653AFD094 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 4 | D18C67FB-DFCA-41B0-B309-9A5F86B8FBCD | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 5 | 375CF0E4-D46E-486D-AF42-811C0A5580F1 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 6 | 56255FA1-61A5-4556-B8BD-E3A19569661E | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 7 | 53E5F267-0E88-4F56-A07E-A396727C1150 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 8 | F2170C61-A77C-49BC-98B0-71AA76951952 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 9 | E210DE1F-F6FB-44C9-A3F9-A50D0B7FA1AA | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 10 | 119A0E03-7732-4BA4-A344-0D27EBA12FDF | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |

### Bottom 10 Records

| RecordID | RunID | EquipID | WindowStartEntryDateTime | WindowEndEntryDateTime | SamplesIn | SamplesKept | SensorsKept | CadenceOKPct | DriftP95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 14 | 9BB8853B-A058-4772-B52E-C943EC04E277 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 13 | C0F2873D-E05B-475B-80B3-FA4DF917FFC6 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 12 | C8605AC2-4416-4FA2-A551-FA89B8A0FECB | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 11 | F5752054-9219-4DFE-A9D0-165F32994EDA | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 10 | 119A0E03-7732-4BA4-A344-0D27EBA12FDF | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 9 | E210DE1F-F6FB-44C9-A3F9-A50D0B7FA1AA | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 8 | F2170C61-A77C-49BC-98B0-71AA76951952 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 7 | 53E5F267-0E88-4F56-A07E-A396727C1150 | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |
| 6 | 56255FA1-61A5-4556-B8BD-E3A19569661E | 1 | 2023-10-15 00:00:00 | 2025-09-14 23:29:59 | 8749 | 8749 | 9 | 100.0 | NULL |

---


## dbo.ACM_Runs

**Primary Key:** RunID  
**Row Count:** 802  
**Date Range:** 2025-12-13 06:00:34 to 2025-12-26 06:07:19  

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
| 07555857-36ED-4278-8AB4-0018B10E35EE | 2621 | GAS_TURBINE | 2025-12-23 00:23:16 | 2025-12-23 00:23:17 | 0 |  | 0 | 0 | 0 |
| 198EC38B-24B0-48F7-A901-003DF53343A1 | 1 | FD_FAN | 2025-12-25 10:10:30 | 2025-12-25 10:15:13 | 281 |  | 336 | 0 | 2 |
| B5110E61-7861-4B27-A02B-004CFE24A761 | 5017 | WFA_TURBINE_17 | 2025-12-20 06:10:53 | 2025-12-20 06:11:31 | 38 |  | 554 | 5945 | 7 |
| E4034AD7-1CCD-4C45-B522-00EBB5065B8B | 2621 | GAS_TURBINE | 2025-12-22 16:10:50 | 2025-12-22 16:13:34 | 163 |  | 47 | 796 | 0 |
| E17435C3-D5F4-437E-B2F9-01365FE163D1 | 5092 | WFA_TURBINE_92 | 2025-12-13 06:11:31 | 2025-12-13 06:12:14 | 42 |  | 144 | 4819 | 2 |
| 49FAF19F-2FE6-4148-873B-017B7CFC830E | 2621 | GAS_TURBINE | 2025-12-23 00:13:02 | 2025-12-23 00:13:03 | 0 |  | 0 | 0 | 0 |
| 06E6233C-E6CB-4ECA-964F-019CCE7B6B36 | 5024 | WFA_TURBINE_24 | 2025-12-20 07:06:28 | 2025-12-20 07:07:03 | 34 |  | 553 | 5924 | 4 |
| CD9F1BB3-E997-42D3-B124-01F1FC3E0DDA | 5022 | WFA_TURBINE_22 | 2025-12-20 06:34:52 | 2025-12-20 06:35:30 | 37 |  | 553 | 5922 | 3 |
| 720DF3BF-9903-418C-9915-0219240DA724 | 5022 | WFA_TURBINE_22 | 2025-12-20 07:13:55 | 2025-12-20 07:14:29 | 34 |  | 553 | 5924 | 4 |
| 907BB29F-756F-4730-8379-024CAAAA4F90 | 5024 | WFA_TURBINE_24 | 2025-12-20 06:44:05 | 2025-12-20 06:44:44 | 39 |  | 553 | 5922 | 3 |

### Bottom 10 Records

| RunID | EquipID | EquipName | StartedAt | CompletedAt | DurationSeconds | ConfigSignature | TrainRowCount | ScoreRowCount | EpisodeCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A3E5AD7C-06F3-426D-B093-FFB033319777 | 5025 | WFA_TURBINE_25 | 2025-12-20 07:52:36 | 2025-12-20 07:57:37 | 299 |  | 0 | 0 | 0 |
| 4E373307-844C-479E-8FE7-FF9FB7D8C070 | 5024 | WFA_TURBINE_24 | 2025-12-20 06:40:57 | 2025-12-20 06:41:39 | 41 |  | 552 | 5917 | 3 |
| 1FB1A67B-0173-4879-9B43-FF09F8EFE845 | 5022 | WFA_TURBINE_22 | 2025-12-20 06:47:48 | 2025-12-20 06:48:29 | 41 |  | 553 | 5926 | 5 |
| F8F21D38-676D-497F-B9C9-FEF2CAE2B0E7 | 5000 | WFA_TURBINE_0 | 2025-12-22 09:40:53 | 2025-12-22 09:56:35 | 941 |  | 562 | 5973 | 6 |
| BF6A836E-5A92-4230-92B5-FEDE05E009E0 | 5014 | WFA_TURBINE_14 | 2025-12-20 05:29:06 | 2025-12-20 05:29:42 | 35 |  | 554 | 5937 | 8 |
| B1B899AE-601A-48AA-B3A1-FE482624FE1D | 5022 | WFA_TURBINE_22 | 2025-12-20 06:41:01 | 2025-12-20 06:41:36 | 34 |  | 489 | 5065 | 2 |
| C6ABEE59-793D-4E87-B917-FD544E69334D | 2621 | GAS_TURBINE | 2025-12-22 17:46:20 | 2025-12-22 17:49:50 | 209 |  | 47 | 776 | 0 |
| FD4A83E2-5E8D-43AF-9D02-FD1912BC4233 | 5010 | WFA_TURBINE_10 | 2025-12-22 10:51:55 | 2025-12-22 11:06:51 | 895 |  | 562 | 5979 | 9 |
| 952ED5C7-71D9-472A-8C40-FD0A3D3967EA | 5010 | WFA_TURBINE_10 | 2025-12-22 08:55:28 | 2025-12-22 09:09:32 | 843 |  | 548 | 5901 | 5 |
| 07AC845F-6F01-4ECA-B548-FCF90C03BE15 | 5024 | WFA_TURBINE_24 | 2025-12-20 07:35:49 | 2025-12-20 07:36:35 | 45 |  | 553 | 5810 | 7 |

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
**Row Count:** 298,298  
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
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |
| 2025-09-14 23:00:00 | 1.0671190023422241 | 0.20852130651474 | 2.395613431930542 | NULL | 2.701765775680542 | 0.7915700674057007 | -0.4324653744697571 | NULL | NULL |

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
**Row Count:** 191,844  
**Date Range:** 2025-12-25 06:56:31 to 2025-12-26 06:12:16  

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
| 1 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06G31_1FD Fan Damper Position_med | 1.0 | pearson | 2025-12-25 06:56:31 |
| 2 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06GP34_1FD Fan Outlet Pressure_med | 0.9544004584475696 | pearson | 2025-12-25 06:56:31 |
| 3 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06I03_1FD Fan Motor Current_med | 0.9454283456869451 | pearson | 2025-12-25 06:56:31 |
| 4 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06T31_1FD Fan Inlet Temperature_med | 0.1142947341871645 | pearson | 2025-12-25 06:56:31 |
| 5 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature_med | 0.5062771207393317 | pearson | 2025-12-25 06:56:31 |
| 6 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_med | 0.4177737846762664 | pearson | 2025-12-25 06:56:31 |
| 7 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06T34_1FD Fan Outlet Termperature_med | 0.4300696403587483 | pearson | 2025-12-25 06:56:31 |
| 8 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_med | 0.9730210379348073 | pearson | 2025-12-25 06:56:31 |
| 9 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_med | 0.9048549456201679 | pearson | 2025-12-25 06:56:31 |
| 10 | e17f92fb-b548-4ee1-83e0-b12f4dda2f1c | 1 | DEMO.SIM.06G31_1FD Fan Damper Position_med | DEMO.SIM.06G31_1FD Fan Damper Position_mad | 0.5021781222034186 | pearson | 2025-12-25 06:56:31 |

### Bottom 10 Records

| ID | RunID | EquipID | Sensor1 | Sensor2 | Correlation | CorrelationType | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 191844 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 1.0 | pearson | 2025-12-26 06:12:16 |
| 191843 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 0.9593028100751169 | pearson | 2025-12-26 06:12:16 |
| 191842 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | 1.0 | pearson | 2025-12-26 06:12:16 |
| 191841 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 0.1963024035851471 | pearson | 2025-12-26 06:12:16 |
| 191840 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | 0.18286373902666567 | pearson | 2025-12-26 06:12:16 |
| 191839 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | 1.0 | pearson | 2025-12-26 06:12:16 |
| 191838 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow_rz | 0.15520005113530588 | pearson | 2025-12-26 06:12:16 |
| 191837 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow_rz | 0.1515257545718971 | pearson | 2025-12-26 06:12:16 |
| 191836 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.06T34_1FD Fan Outlet Termperature_rz | 0.2905775827364145 | pearson | 2025-12-26 06:12:16 |
| 191835 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | DEMO.SIM.06T33-1_1FD Fan Winding Temperature_rz | 1.0 | pearson | 2025-12-26 06:12:16 |

---


## dbo.ACM_SensorDefects

**Primary Key:** No primary key  
**Row Count:** 3,165  

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
| Correlation Break (PCA-SPE) | Correlation | HIGH | 40 | 11.9 | 10.0 | 1.0443 | 1.454 | 0 | 198EC38B-24B0-48F7-A901-003DF53343A1 |
| Baseline Consistency (OMR) | Baseline | MEDIUM | 31 | 9.23 | 5.2572 | 0.8882 | 1.2511 | 0 | 198EC38B-24B0-48F7-A901-003DF53343A1 |
| cusum_z | cusum_z | MEDIUM | 30 | 8.93 | 2.8922 | 0.876 | 0.7881 | 0 | 198EC38B-24B0-48F7-A901-003DF53343A1 |
| Time-Series Anomaly (AR1) | Time-Series | MEDIUM | 25 | 7.44 | 10.0 | 0.9187 | 0.5154 | 0 | 198EC38B-24B0-48F7-A901-003DF53343A1 |
| Rare State (IsolationForest) | Rare | MEDIUM | 20 | 5.95 | 4.1999 | 0.7892 | 0.0103 | 0 | 198EC38B-24B0-48F7-A901-003DF53343A1 |
| Multivariate Outlier (PCA-T2) | Multivariate | MEDIUM | 19 | 5.65 | 10.0 | 0.8282 | 0.8719 | 0 | 198EC38B-24B0-48F7-A901-003DF53343A1 |
| Density Anomaly (GMM) | Density | MEDIUM | 19 | 5.65 | 10.0 | 0.928 | 0.5732 | 0 | 198EC38B-24B0-48F7-A901-003DF53343A1 |
| Density Anomaly (GMM) | Density | CRITICAL | 13 | 27.66 | 3.1319 | 1.0607 | 0.6089 | 0 | E4034AD7-1CCD-4C45-B522-00EBB5065B8B |
| Time-Series Anomaly (AR1) | Time-Series | HIGH | 6 | 12.77 | 10.0 | 1.1883 | 10.0 | 1 | E4034AD7-1CCD-4C45-B522-00EBB5065B8B |
| Correlation Break (PCA-SPE) | Correlation | HIGH | 5 | 10.64 | 9.9799 | 1.1161 | 2.7373 | 1 | E4034AD7-1CCD-4C45-B522-00EBB5065B8B |

### Bottom 10 Records

| DetectorType | DetectorFamily | Severity | ViolationCount | ViolationPct | MaxZ | AvgZ | CurrentZ | ActiveDefect | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Density Anomaly (GMM) | Density | CRITICAL | 314 | 56.88 | 10.0 | 4.4925 | 10.0 | 1 | 4E373307-844C-479E-8FE7-FF9FB7D8C070 |
| Correlation Break (PCA-SPE) | Correlation | CRITICAL | 299 | 54.17 | 10.0 | 5.466 | 10.0 | 1 | 4E373307-844C-479E-8FE7-FF9FB7D8C070 |
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 278 | 50.36 | 10.0 | 5.1445 | 10.0 | 1 | 4E373307-844C-479E-8FE7-FF9FB7D8C070 |
| Baseline Consistency (OMR) | Baseline | CRITICAL | 251 | 45.47 | 3.8965 | 2.0307 | 3.8965 | 1 | 4E373307-844C-479E-8FE7-FF9FB7D8C070 |
| Rare State (IsolationForest) | Rare | CRITICAL | 205 | 37.14 | 8.4542 | 1.964 | 3.3594 | 1 | 4E373307-844C-479E-8FE7-FF9FB7D8C070 |
| Time-Series Anomaly (AR1) | Time-Series | LOW | 27 | 4.89 | 10.0 | 0.4891 | 0.0 | 0 | 4E373307-844C-479E-8FE7-FF9FB7D8C070 |
| cusum_z | cusum_z | LOW | 0 | 0.0 | 1.6472 | 0.6675 | 1.5014 | 0 | 4E373307-844C-479E-8FE7-FF9FB7D8C070 |
| Density Anomaly (GMM) | Density | CRITICAL | 288 | 52.08 | 10.0 | 3.8776 | 6.6978 | 1 | 1FB1A67B-0173-4879-9B43-FF09F8EFE845 |
| Baseline Consistency (OMR) | Baseline | CRITICAL | 271 | 49.01 | 4.1486 | 2.1407 | 3.7765 | 1 | 1FB1A67B-0173-4879-9B43-FF09F8EFE845 |
| Time-Series Anomaly (AR1) | Time-Series | CRITICAL | 132 | 23.87 | 10.0 | 2.387 | 10.0 | 1 | 1FB1A67B-0173-4879-9B43-FF09F8EFE845 |

---


## dbo.ACM_SensorForecast

**Primary Key:** RunID, EquipID, Timestamp, SensorName  
**Row Count:** 1,512  
**Date Range:** 2025-09-15 00:00:00 to 2025-09-21 23:00:00  

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
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 48.12678485717701 | 42.163383324219076 | 54.09018639013494 | 3.0425518025295566 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1.3137478207230482 | 1.0301776614649123 | 1.597317979981184 | 0.14467865268272234 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.06I03_1FD Fan Motor Current | 45.37447400259745 | 40.352224855240266 | 50.39672314995464 | 2.562372013957749 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 48.36847776886253 | 45.2162894584921 | 51.52066607923297 | 1.608259342025731 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | 63.58911051626387 | 57.92584392564117 | 69.25237710688658 | 2.8894217299095444 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 55.26790572467295 | 51.364490574560065 | 59.171320874785835 | 1.991538341894327 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.06T34_1FD Fan Outlet Termperature | 33.46724673606946 | 30.75095977099539 | 36.18353370114353 | 1.3858606964663636 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | 385.33832621245307 | 335.4701265758386 | 435.20652584906753 | 25.442958998272676 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 00:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 400.0457737896118 | 348.64859073217826 | 451.44295684704537 | 26.223052580323248 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-15 01:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 48.194414633227794 | 42.23101310026986 | 54.157816166185725 | 3.0425518025295566 | ExponentialSmoothing | 0 |

### Bottom 10 Records

| RunID | EquipID | Timestamp | SensorName | ForecastValue | CiLower | CiUpper | ForecastStd | Method | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 519.252444365917 | 467.8552613084834 | 570.6496274233506 | 26.223052580323248 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.FSAA_1FD Fan Left Inlet Flow | 536.9387413739273 | 487.07054173731285 | 586.8069410105418 | 25.442958998272676 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.06T34_1FD Fan Outlet Termperature | 50.7952740016389 | 48.078987036564826 | 53.51156096671298 | 1.3858606964663636 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.06T33-1_1FD Fan Winding Temperature | 70.70773715248757 | 66.80432200237469 | 74.61115230260044 | 1.991538341894327 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.06T32-1_1FD Fan Bearing Temperature | 70.07466994266613 | 64.41140335204342 | 75.73793653328885 | 2.8894217299095444 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.06T31_1FD Fan Inlet Temperature | 65.16598947272284 | 62.01380116235241 | 68.31817778309328 | 1.608259342025731 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.06I03_1FD Fan Motor Current | 56.70130454800951 | 51.679055400652324 | 61.7235536953667 | 2.562372013957749 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 1.582990346754514 | 1.2994201874963784 | 1.8665605060126498 | 0.14467865268272234 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 23:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | 59.42095745765765 | 53.45755592469972 | 65.38435899061558 | 3.0425518025295566 | ExponentialSmoothing | 0 |
| DFFED1D3-93F6-414F-ABE4-AD82F40C48D0 | 1 | 2025-09-21 22:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | 518.5386319672564 | 467.1414489098228 | 569.93581502469 | 26.223052580323248 | ExponentialSmoothing | 0 |

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
**Row Count:** 8,465  
**Date Range:** 2022-04-04 02:30:00 to 2025-05-13 17:00:00  

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
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |
| DEMO.SIM.06GP34_1FD Fan Outlet Pressure | 2025-05-13 17:00:00 | 2025-09-14 23:00:00 | 6.5488 | 6.5488 | 2.2873 | 2.2873 | 2.569999933242798 | 1.3300000429153442 | 0.6644644737243652 |

---


## dbo.ACM_SensorNormalized_TS

**Primary Key:** ID  
**Row Count:** 472,446  
**Date Range:** 2024-08-24 21:30:00 to 2025-09-14 23:00:00  

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
| 26248 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-24 21:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 40.7400016784668 | 2025-12-26 05:01:29 |
| 26249 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-24 22:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 43.91999816894531 | 2025-12-26 05:01:29 |
| 26250 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-24 22:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 35.36000061035156 | 2025-12-26 05:01:29 |
| 26251 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-24 23:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 36.84000015258789 | 2025-12-26 05:01:29 |
| 26252 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-24 23:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 37.45000076293945 | 2025-12-26 05:01:29 |
| 26253 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-25 00:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 37.33000183105469 | 2025-12-26 05:01:29 |
| 26254 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-25 00:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 37.959999084472656 | 2025-12-26 05:01:29 |
| 26255 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-25 01:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 28.81999969482422 | 2025-12-26 05:01:29 |
| 26256 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-25 01:30:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 37.18000030517578 | 2025-12-26 05:01:29 |
| 26257 | 119a0e03-7732-4ba4-a344-0d27eba12fdf | 1 | 2024-08-25 02:00:00 | DEMO.SIM.06G31_1FD Fan Damper Position | NULL | 35.86000061035156 | 2025-12-26 05:01:29 |

### Bottom 10 Records

| ID | RunID | EquipID | Timestamp | SensorName | RawValue | NormalizedValue | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 498693 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 23:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 397.95001220703125 | 2025-12-26 06:13:59 |
| 498692 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 22:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 393.2699890136719 | 2025-12-26 06:13:59 |
| 498691 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 22:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 396.79998779296875 | 2025-12-26 06:13:59 |
| 498690 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 21:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 359.95001220703125 | 2025-12-26 06:13:59 |
| 498689 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 21:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 378.3500061035156 | 2025-12-26 06:13:59 |
| 498688 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 20:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 418.4100036621094 | 2025-12-26 06:13:59 |
| 498687 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 20:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 395.4700012207031 | 2025-12-26 06:13:59 |
| 498686 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 19:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 373.6099853515625 | 2025-12-26 06:13:59 |
| 498685 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 19:00:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 390.54998779296875 | 2025-12-26 06:13:59 |
| 498684 | dffed1d3-93f6-414f-abe4-ad82f40c48d0 | 1 | 2025-09-14 18:30:00 | DEMO.SIM.FSAB_1FD Fan Right Inlet Flow | NULL | 367.2799987792969 | 2025-12-26 06:13:59 |

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
**Row Count:** 3,820  
**Date Range:** 2025-12-13 06:00:50 to 2025-12-26 06:10:47  

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
| ar1_params | 1 | 1 | 2025-12-25 05:38:01 | {"n_sensors": 72, "mean_autocorr": 14.3704, "mean_residual_std": 1.5281, "params_count": 144} | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 5677 bytes> |
| feature_medians | 1 | 1 | 2025-12-25 05:38:06 | NULL | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 9064 bytes> |
| gmm_model | 1 | 1 | 2025-12-25 05:38:06 | {"n_components": 3, "covariance_type": "diag", "bic": 152026373.84, "aic": 152025256.42, "lower_b... | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 7990 bytes> |
| iforest_model | 1 | 1 | 2025-12-25 05:38:05 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 689641 bytes> |
| omr_model | 1 | 1 | 2025-12-25 05:38:06 | NULL | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 85416 bytes> |
| pca_model | 1 | 1 | 2025-12-25 05:38:01 | {"n_components": 5, "variance_ratio_sum": 0.7769, "variance_ratio_first_component": 0.2932, "vari... | {"train_rows": 97, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 4511 bytes> |
| ar1_params | 1 | 2 | 2025-12-25 06:55:34 | {"n_sensors": 72, "mean_autocorr": 13.0591, "mean_residual_std": 1.2951, "params_count": 144} | {"train_rows": 300, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 5677 bytes> |
| feature_medians | 1 | 2 | 2025-12-25 06:55:39 | NULL | {"train_rows": 300, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 9064 bytes> |
| gmm_model | 1 | 2 | 2025-12-25 06:55:39 | {"n_components": 3, "covariance_type": "diag", "bic": 156070828.7, "aic": 156069221.26, "lower_bo... | {"train_rows": 300, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 8218 bytes> |
| iforest_model | 1 | 2 | 2025-12-25 06:55:38 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 300, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06G... | NULL | <binary 1316729 bytes> |

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

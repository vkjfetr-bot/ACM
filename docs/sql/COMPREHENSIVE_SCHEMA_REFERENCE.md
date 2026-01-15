# ACM Comprehensive Database Schema Reference

_Generated automatically on 2026-01-15 11:03:13_

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
- [dbo.ACM_Anomaly_Events](#dboacmanomalyevents)
- [dbo.ACM_AssetProfiles](#dboacmassetprofiles)
- [dbo.ACM_BaselineBuffer](#dboacmbaselinebuffer)
- [dbo.ACM_CalibrationSummary](#dboacmcalibrationsummary)
- [dbo.ACM_ColdstartState](#dboacmcoldstartstate)
- [dbo.ACM_Config](#dboacmconfig)
- [dbo.ACM_ConfigHistory](#dboacmconfighistory)
- [dbo.ACM_ContributionTimeline](#dboacmcontributiontimeline)
- [dbo.ACM_DataContractValidation](#dboacmdatacontractvalidation)
- [dbo.ACM_DataQuality](#dboacmdataquality)
- [dbo.ACM_DetectorCorrelation](#dboacmdetectorcorrelation)
- [dbo.ACM_DriftController](#dboacmdriftcontroller)
- [dbo.ACM_DriftSeries](#dboacmdriftseries)
- [dbo.ACM_EpisodeCulprits](#dboacmepisodeculprits)
- [dbo.ACM_EpisodeDiagnostics](#dboacmepisodediagnostics)
- [dbo.ACM_Episodes](#dboacmepisodes)
- [dbo.ACM_FailureForecast](#dboacmfailureforecast)
- [dbo.ACM_FeatureDropLog](#dboacmfeaturedroplog)
- [dbo.ACM_ForecastState](#dboacmforecaststate)
- [dbo.ACM_Forecast_QualityMetrics](#dboacmforecastqualitymetrics)
- [dbo.ACM_ForecastingState](#dboacmforecastingstate)
- [dbo.ACM_HealthForecast](#dboacmhealthforecast)
- [dbo.ACM_HealthTimeline](#dboacmhealthtimeline)
- [dbo.ACM_HistorianData](#dboacmhistoriandata)
- [dbo.ACM_MultivariateForecast](#dboacmmultivariateforecast)
- [dbo.ACM_OMR_Diagnostics](#dboacmomrdiagnostics)
- [dbo.ACM_PCA_Loadings](#dboacmpcaloadings)
- [dbo.ACM_PCA_Metrics](#dboacmpcametrics)
- [dbo.ACM_PCA_Models](#dboacmpcamodels)
- [dbo.ACM_RUL](#dboacmrul)
- [dbo.ACM_RefitRequests](#dboacmrefitrequests)
- [dbo.ACM_RegimeDefinitions](#dboacmregimedefinitions)
- [dbo.ACM_RegimeOccupancy](#dboacmregimeoccupancy)
- [dbo.ACM_RegimePromotionLog](#dboacmregimepromotionlog)
- [dbo.ACM_RegimeState](#dboacmregimestate)
- [dbo.ACM_RegimeTimeline](#dboacmregimetimeline)
- [dbo.ACM_RegimeTransitions](#dboacmregimetransitions)
- [dbo.ACM_Regime_Episodes](#dboacmregimeepisodes)
- [dbo.ACM_RunLogs](#dboacmrunlogs)
- [dbo.ACM_RunMetadata](#dboacmrunmetadata)
- [dbo.ACM_RunMetrics](#dboacmrunmetrics)
- [dbo.ACM_Run_Stats](#dboacmrunstats)
- [dbo.ACM_Runs](#dboacmruns)
- [dbo.ACM_SchemaVersion](#dboacmschemaversion)
- [dbo.ACM_Scores_Wide](#dboacmscoreswide)
- [dbo.ACM_SeasonalPatterns](#dboacmseasonalpatterns)
- [dbo.ACM_SensorCorrelations](#dboacmsensorcorrelations)
- [dbo.ACM_SensorDefects](#dboacmsensordefects)
- [dbo.ACM_SensorForecast](#dboacmsensorforecast)
- [dbo.ACM_SensorHotspots](#dboacmsensorhotspots)
- [dbo.ACM_SensorNormalized_TS](#dboacmsensornormalizedts)
- [dbo.ACM_TagEquipmentMap](#dboacmtagequipmentmap)
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
| dbo.ACM_Anomaly_Events | 7 | 0 | Id |
| dbo.ACM_AssetProfiles | 11 | 1 | ID |
| dbo.ACM_BaselineBuffer | 7 | 91,854 | Id |
| dbo.ACM_CalibrationSummary | 10 | 48 | ID |
| dbo.ACM_ColdstartState | 18 | 4 | EquipID, Stage |
| dbo.ACM_Config | 7 | 339 | ConfigID |
| dbo.ACM_ConfigHistory | 9 | 13 | ID |
| dbo.ACM_ContributionTimeline | 7 | 0 | ID |
| dbo.ACM_DataContractValidation | 10 | 12 | ID |
| dbo.ACM_DataQuality | 25 | 324 | — |
| dbo.ACM_DetectorCorrelation | 7 | 278 | ID |
| dbo.ACM_DriftController | 10 | 7 | ID |
| dbo.ACM_DriftSeries | 7 | 0 | ID |
| dbo.ACM_EpisodeCulprits | 9 | 4,994 | ID |
| dbo.ACM_EpisodeDiagnostics | 16 | 875 | ID |
| dbo.ACM_Episodes | 15 | 875 | ID |
| dbo.ACM_FailureForecast | 10 | 3,192 | EquipID, RunID, Timestamp |
| dbo.ACM_FeatureDropLog | 8 | 146 | ID |
| dbo.ACM_ForecastState | 13 | 0 | EquipID, StateVersion |
| dbo.ACM_Forecast_QualityMetrics | 15 | 0 | MetricID |
| dbo.ACM_ForecastingState | 13 | 4 | EquipID, StateVersion |
| dbo.ACM_HealthForecast | 11 | 5,208 | EquipID, RunID, Timestamp |
| dbo.ACM_HealthTimeline | 11 | 74,541 | — |
| dbo.ACM_HistorianData | 6 | 0 | ID |
| dbo.ACM_MultivariateForecast | 10 | 0 | ID |
| dbo.ACM_OMR_Diagnostics | 15 | 10 | DiagnosticID |
| dbo.ACM_PCA_Loadings | 8 | 16,430 | ID |
| dbo.ACM_PCA_Metrics | 10 | 10 | ID |
| dbo.ACM_PCA_Models | 12 | 5 | ID |
| dbo.ACM_RUL | 33 | 7 | EquipID, RunID |
| dbo.ACM_RefitRequests | 10 | 7 | RequestID |
| dbo.ACM_RegimeDefinitions | 12 | 27 | ID |
| dbo.ACM_RegimeOccupancy | 9 | 32 | ID |
| dbo.ACM_RegimePromotionLog | 9 | 0 | ID |
| dbo.ACM_RegimeState | 15 | 4 | EquipID, StateVersion |
| dbo.ACM_RegimeTimeline | 8 | 74,541 | — |
| dbo.ACM_RegimeTransitions | 8 | 79 | ID |
| dbo.ACM_Regime_Episodes | 8 | 0 | ID |
| dbo.ACM_RunLogs | 8 | 0 | ID |
| dbo.ACM_RunMetadata | 11 | 0 | ID |
| dbo.ACM_RunMetrics | 7 | 54 | ID |
| dbo.ACM_Run_Stats | 13 | 7 | RecordID |
| dbo.ACM_Runs | 20 | 15 | RunID |
| dbo.ACM_SchemaVersion | 5 | 2 | VersionID |
| dbo.ACM_Scores_Wide | 16 | 74,541 | — |
| dbo.ACM_SeasonalPatterns | 10 | 511 | ID |
| dbo.ACM_SensorCorrelations | 8 | 9,616 | ID |
| dbo.ACM_SensorDefects | 12 | 49 | — |
| dbo.ACM_SensorForecast | 12 | 0 | RunID, EquipID, Timestamp, SensorName |
| dbo.ACM_SensorHotspots | 19 | 166 | — |
| dbo.ACM_SensorNormalized_TS | 8 | 69,006 | ID |
| dbo.ACM_TagEquipmentMap | 10 | 1,986 | TagID |
| dbo.ELECTRIC_MOTOR_Data | 14 | 17,477 | — |
| dbo.ELECTRIC_MOTOR_Data_RAW | 14 | 1,048,575 | — |
| dbo.Equipment | 8 | 29 | EquipID |
| dbo.FD_FAN_Data | 11 | 17,499 | EntryDateTime |
| dbo.GAS_TURBINE_Data | 18 | 2,911 | EntryDateTime |
| dbo.ModelRegistry | 8 | 96 | ModelType, EquipID, Version |
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
| 28 | 5000 | 1 | LEARNING | NULL | 1 | NULL | 1 | NULL | 2026-01-12 13:29:52 |
| 39 | 5013 | 1 | LEARNING | NULL | 1 | NULL | 1 | NULL | 2026-01-13 09:51:54 |
| 40 | 2621 | 1 | LEARNING | NULL | 1 | NULL | 1 | NULL | 2026-01-13 09:59:14 |
| 41 | 5010 | 1 | LEARNING | NULL | 1 | NULL | 1 | NULL | 2026-01-13 12:25:35 |

---


## dbo.ACM_AdaptiveConfig

**Primary Key:** ConfigID  
**Row Count:** 17  
**Date Range:** 2025-12-04 10:46:47 to 2026-01-01 12:04:51  

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
| 18 | 5003 | fused_alert_z | 1.4688166379928589 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated from 129 accumulated samples |

### Bottom 10 Records

| ConfigID | EquipID | ConfigKey | ConfigValue | MinBound | MaxBound | IsLearned | DataVolumeAtTuning | PerformanceMetric | ResearchReference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 85 | 5040 | fused_warn_z | 1.5 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 84 | 5040 | fused_alert_z | 3.0 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated from 49 accumulated samples |
| 77 | 1 | fused_warn_z | 1.5 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 76 | 1 | fused_alert_z | 3.0 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated from 49 accumulated samples |
| 55 | 5092 | fused_warn_z | 0.5271811485290527 | 0.0 | 999999.0 | True | 2717 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 54 | 5092 | fused_alert_z | 1.0543622970581055 | 0.0 | 999999.0 | True | 2717 | 0.0 | quantile_0.997: Auto-calculated from 2717 accumulated samples |
| 19 | 5003 | fused_warn_z | 0.7344083189964294 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 18 | 5003 | fused_alert_z | 1.4688166379928589 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated from 129 accumulated samples |
| 9 | NULL | auto_tune_data_threshold | 10000.0 | 5000.0 | 50000.0 | False | NULL | NULL | Expert tuning - Auto-tuning trigger |
| 8 | NULL | blend_tau_hours | 12.0 | 6.0 | 48.0 | False | NULL | NULL | Expert tuning - Warm-start alpha blending |

---


## dbo.ACM_Anomaly_Events

**Primary Key:** Id  
**Row Count:** 0  

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

---


## dbo.ACM_AssetProfiles

**Primary Key:** ID  
**Row Count:** 1  
**Date Range:** 2026-01-02 22:43:05 to 2026-01-02 22:43:05  

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
| RunID | nvarchar | YES | 50 | — |

### Top 10 Records

| ID | EquipID | EquipType | SensorNamesJSON | SensorMeansJSON | SensorStdsJSON | RegimeCount | TypicalHealth | DataHours | LastUpdatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 5010 | WFA_TURBINE_10 | ["power_29_avg", "power_29_max", "power_29_min", "power_29_std", "power_30_avg", "power_30_max", ... | {"power_29_avg": 0.27795690212598545, "power_29_max": 0.4237643052903167, "power_29_min": 0.04104... | {"power_29_avg": 0.35138767809240373, "power_29_max": 0.39163233970897193, "power_29_min": 0.0886... | 1 | 85.0 | 539.3333333333334 | 2026-01-02 22:43:05 |

---


## dbo.ACM_BaselineBuffer

**Primary Key:** Id  
**Row Count:** 91,854  
**Date Range:** 2022-08-12 03:10:00 to 2022-08-20 00:00:00  

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
| 1 | 5000 | 2022-08-12 03:10:00 | power_29_avg | 1.0791017365736264 | NULL | 2026-01-04 16:15:22 |
| 2 | 5000 | 2022-08-12 03:20:00 | power_29_avg | 1.0840107494273612 | NULL | 2026-01-04 16:15:22 |
| 3 | 5000 | 2022-08-12 03:30:00 | power_29_avg | 1.0816890074866805 | NULL | 2026-01-04 16:15:22 |
| 4 | 5000 | 2022-08-12 03:40:00 | power_29_avg | 1.082039971876368 | NULL | 2026-01-04 16:15:22 |
| 5 | 5000 | 2022-08-12 03:50:00 | power_29_avg | 1.0736991268750575 | NULL | 2026-01-04 16:15:22 |
| 6 | 5000 | 2022-08-12 04:00:00 | power_29_avg | 1.0542779701037395 | NULL | 2026-01-04 16:15:22 |
| 7 | 5000 | 2022-08-12 04:10:00 | power_29_avg | 1.0469007218161104 | NULL | 2026-01-04 16:15:22 |
| 8 | 5000 | 2022-08-12 04:20:00 | power_29_avg | 1.072789283654519 | NULL | 2026-01-04 16:15:22 |
| 9 | 5000 | 2022-08-12 04:30:00 | power_29_avg | 1.075263675864118 | NULL | 2026-01-04 16:15:22 |
| 10 | 5000 | 2022-08-12 04:40:00 | power_29_avg | 1.0717906791668672 | NULL | 2026-01-04 16:15:22 |

### Bottom 10 Records

| Id | EquipID | Timestamp | SensorName | SensorValue | DataQuality | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 91854 | 5000 | 2022-08-20 00:00:00 | wind_speed_4_avg | 4.217885432840985 | NULL | 2026-01-04 16:15:46 |
| 91853 | 5000 | 2022-08-19 23:50:00 | wind_speed_4_avg | 3.9677605976270716 | NULL | 2026-01-04 16:15:46 |
| 91852 | 5000 | 2022-08-19 23:40:00 | wind_speed_4_avg | 4.314080516828613 | NULL | 2026-01-04 16:15:46 |
| 91851 | 5000 | 2022-08-19 23:30:00 | wind_speed_4_avg | 4.456947182977061 | NULL | 2026-01-04 16:15:46 |
| 91850 | 5000 | 2022-08-19 23:20:00 | wind_speed_4_avg | 4.496469257343981 | NULL | 2026-01-04 16:15:46 |
| 91849 | 5000 | 2022-08-19 23:10:00 | wind_speed_4_avg | 4.8327621967026335 | NULL | 2026-01-04 16:15:46 |
| 91848 | 5000 | 2022-08-19 23:00:00 | wind_speed_4_avg | 4.665946842091915 | NULL | 2026-01-04 16:15:46 |
| 91847 | 5000 | 2022-08-19 22:50:00 | wind_speed_4_avg | 4.996150856901715 | NULL | 2026-01-04 16:15:46 |
| 91846 | 5000 | 2022-08-19 22:40:00 | wind_speed_4_avg | 5.4235067681523175 | NULL | 2026-01-04 16:15:46 |
| 91845 | 5000 | 2022-08-19 22:30:00 | wind_speed_4_avg | 5.448152857995472 | NULL | 2026-01-04 16:15:46 |

---


## dbo.ACM_CalibrationSummary

**Primary Key:** ID  
**Row Count:** 48  
**Date Range:** 2026-01-04 08:15:12 to 2026-01-13 06:56:34  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | NO | 10 | — |
| DetectorType | nvarchar | NO | 50 | — |
| CalibrationScore | float | YES | 53 | — |
| TrainR2 | float | YES | 53 | — |
| MeanAbsError | float | YES | 53 | — |
| P95Error | float | YES | 53 | — |
| DatapointsUsed | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | DetectorType | CalibrationScore | TrainR2 | MeanAbsError | P95Error | DatapointsUsed | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 55 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | ar1_z | 5.447473289727075 | NULL | NULL | NULL | NULL | 2026-01-04 08:15:12 |
| 56 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | pca_spe_z | -20.0 | NULL | NULL | NULL | NULL | 2026-01-04 08:15:12 |
| 57 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | pca_t2_z | 20.0 | NULL | NULL | NULL | NULL | 2026-01-04 08:15:12 |
| 58 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | iforest_z | 7.951392955515674 | NULL | NULL | NULL | NULL | 2026-01-04 08:15:12 |
| 59 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | gmm_z | 0.13755265519987922 | NULL | NULL | NULL | NULL | 2026-01-04 08:15:12 |
| 60 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | omr_z | 5.9388913517680875 | NULL | NULL | NULL | NULL | 2026-01-04 08:15:12 |
| 67 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | ar1_z | 13.001892733214328 | NULL | NULL | NULL | NULL | 2026-01-04 09:53:13 |
| 68 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | pca_spe_z | 0.28447227933347896 | NULL | NULL | NULL | NULL | 2026-01-04 09:53:13 |
| 69 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | pca_t2_z | 20.0 | NULL | NULL | NULL | NULL | 2026-01-04 09:53:13 |
| 70 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | iforest_z | 5.60450249619977 | NULL | NULL | NULL | NULL | 2026-01-04 09:53:13 |

### Bottom 10 Records

| ID | RunID | EquipID | DetectorType | CalibrationScore | TrainR2 | MeanAbsError | P95Error | DatapointsUsed | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 210 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | omr_z | 5.772561665931953 | NULL | NULL | NULL | NULL | 2026-01-13 06:56:34 |
| 209 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | gmm_z | -1.0603150154174938 | NULL | NULL | NULL | NULL | 2026-01-13 06:56:34 |
| 208 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | iforest_z | 5.97386613829903 | NULL | NULL | NULL | NULL | 2026-01-13 06:56:34 |
| 207 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | pca_t2_z | 0.0034755570743520143 | NULL | NULL | NULL | NULL | 2026-01-13 06:56:34 |
| 206 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | pca_spe_z | -20.0 | NULL | NULL | NULL | NULL | 2026-01-13 06:56:34 |
| 205 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | ar1_z | 20.0 | NULL | NULL | NULL | NULL | 2026-01-13 06:56:33 |
| 204 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | omr_z | 4.121639787457539 | NULL | NULL | NULL | NULL | 2026-01-13 04:29:28 |
| 203 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | gmm_z | 2.550052119956899 | NULL | NULL | NULL | NULL | 2026-01-13 04:29:28 |
| 202 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | iforest_z | 3.3488986301577808 | NULL | NULL | NULL | NULL | 2026-01-13 04:29:28 |
| 201 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | pca_t2_z | 20.0 | NULL | NULL | NULL | NULL | 2026-01-13 04:29:28 |

---


## dbo.ACM_ColdstartState

**Primary Key:** EquipID, Stage  
**Row Count:** 4  
**Date Range:** 2026-01-04 09:37:37 to 2026-01-13 06:44:04  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| EquipID | Stage | Status | AttemptCount | FirstAttemptAt | LastAttemptAt | CompletedAt | AccumulatedRows | RequiredRows | DataStartTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | score | COMPLETE | 1 | 2026-01-13 04:27:10 | 2026-01-13 04:27:12 | 2026-01-13 04:27:12 | 2910 | 500 | 2023-10-15 00:00:00 |
| 5000 | score | COMPLETE | 2 | 2026-01-04 09:37:37 | 2026-01-04 09:46:36 | 2026-01-04 09:46:36 | 12934 | 500 | 2022-08-04 06:10:00 |
| 5010 | score | COMPLETE | 1 | 2026-01-13 06:44:04 | 2026-01-13 06:45:22 | 2026-01-13 06:45:22 | 53591 | 500 | 2022-10-09 08:40:00 |
| 5013 | score | COMPLETE | 1 | 2026-01-13 04:09:57 | 2026-01-13 04:11:14 | 2026-01-13 04:11:14 | 54009 | 500 | 2022-04-30 13:20:00 |

---


## dbo.ACM_Config

**Primary Key:** ConfigID  
**Row Count:** 339  
**Date Range:** 2025-12-09 12:47:06 to 2026-01-13 07:01:09  

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
| 492 | 0 | data.train_csv | data/FD_FAN_BASELINE_DATA.csv | string | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 493 | 0 | data.score_csv | data/FD_FAN_BATCH_DATA.csv | string | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 494 | 0 | data.data_dir | data | string | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 495 | 0 | data.timestamp_col | EntryDateTime | string | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 496 | 0 | data.tag_columns | [] | list | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 497 | 0 | data.sampling_secs | 60 | int | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 498 | 0 | data.max_rows | 100000 | int | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 499 | 0 | features.window | 16 | int | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 500 | 0 | features.fft_bands | [0.0, 0.1, 0.3, 0.5] | list | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 501 | 0 | features.top_k_tags | 5 | int | 2026-01-12 11:02:25 | B19cl3pc\bhadk |

### Bottom 10 Records

| ConfigID | EquipID | ParamPath | ParamValue | ValueType | UpdatedAt | UpdatedBy |
| --- | --- | --- | --- | --- | --- | --- |
| 1043 | 5010 | runtime.tick_minutes | 538560 | int | 2026-01-13 07:01:09 | sql_batch_runner |
| 1042 | 2621 | runtime.tick_minutes | 352919 | int | 2026-01-13 04:27:05 | sql_batch_runner |
| 1041 | 5013 | runtime.tick_minutes | 561420 | int | 2026-01-13 04:09:51 | sql_batch_runner |
| 1028 | 0 | regimes.clustering.fallback_method | gmm | string | 2026-01-12 11:02:25 | B19cl3pc\bhadk |
| 1026 | 5040 | data.tag_columns | ["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3... | list | 2026-01-12 11:02:26 | B19cl3pc\bhadk |
| 1025 | 5040 | data.sampling_secs | 600 | int | 2026-01-12 11:02:26 | B19cl3pc\bhadk |
| 1024 | 5040 | data.timestamp_col | EntryDateTime | string | 2026-01-12 11:02:26 | B19cl3pc\bhadk |
| 1020 | 5000 | data.tag_columns | ["sensor_0_avg","sensor_1_avg","sensor_2_avg","wind_speed_3_avg","wind_speed_4_avg","wind_speed_3... | list | 2026-01-12 11:02:26 | B19cl3pc\bhadk |
| 1010 | 5000 | data.timestamp_col | EntryDateTime | string | 2026-01-12 11:02:26 | COPILOT |
| 1009 | 5000 | data.sampling_secs | 600 | int | 2026-01-12 11:02:26 | COPILOT |

---


## dbo.ACM_ConfigHistory

**Primary Key:** ID  
**Row Count:** 13  
**Date Range:** 2026-01-04 15:37:26 to 2026-01-13 12:27:15  

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
| 15 | 2026-01-04 15:37:26 | 5000 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 16 | 2026-01-04 15:37:26 | 5000 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 17 | 2026-01-04 15:59:55 | 5000 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | c82be7b6-357e-4541-85a6-d02a4460c7d2 |
| 18 | 2026-01-04 15:59:55 | 5000 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | c82be7b6-357e-4541-85a6-d02a4460c7d2 |
| 19 | 2026-01-04 16:14:45 | 5000 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | 66ab154c-8a81-4450-9180-8d859017d3b7 |
| 20 | 2026-01-04 16:14:45 | 5000 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 66ab154c-8a81-4450-9180-8d859017d3b7 |
| 32 | 2026-01-12 15:15:24 | 5000 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | f7849950-ae3e-42e4-8ab2-b90a562008da |
| 33 | 2026-01-12 15:15:24 | 5000 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | f7849950-ae3e-42e4-8ab2-b90a562008da |
| 46 | 2026-01-13 09:53:32 | 5013 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc |
| 47 | 2026-01-13 09:53:32 | 5013 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc |

### Bottom 10 Records

| ID | Timestamp | EquipID | ParameterPath | OldValue | NewValue | ChangedBy | ChangeReason | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 2026-01-13 12:27:15 | 5010 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | fc487ad0-dd33-43e6-9833-d737537c178f |
| 49 | 2026-01-13 09:59:42 | 2621 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | aef8fe81-a23e-4529-a281-a8a0c3047a9d |
| 48 | 2026-01-13 09:59:42 | 2621 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | aef8fe81-a23e-4529-a281-a8a0c3047a9d |
| 47 | 2026-01-13 09:53:32 | 5013 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc |
| 46 | 2026-01-13 09:53:32 | 5013 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc |
| 33 | 2026-01-12 15:15:24 | 5000 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | f7849950-ae3e-42e4-8ab2-b90a562008da |
| 32 | 2026-01-12 15:15:24 | 5000 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | f7849950-ae3e-42e4-8ab2-b90a562008da |
| 20 | 2026-01-04 16:14:45 | 5000 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | 66ab154c-8a81-4450-9180-8d859017d3b7 |
| 19 | 2026-01-04 16:14:45 | 5000 | k_sigma | 2.0 | 2.2 | AUTO_TUNE | Auto-tuning based on quality assessment | 66ab154c-8a81-4450-9180-8d859017d3b7 |
| 18 | 2026-01-04 15:59:55 | 5000 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | c82be7b6-357e-4541-85a6-d02a4460c7d2 |

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


## dbo.ACM_DataContractValidation

**Primary Key:** ID  
**Row Count:** 12  
**Date Range:** 2026-01-04 13:29:19 to 2026-01-13 13:39:27  

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
| 37 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | True | 21994 | 81 | NULL | NULL | 66d0c2b0a100 | 2026-01-04 13:29:19 |
| 39 | ccdd4d7f-3acc-4cf4-9987-d729fd313c6f | 5000 | True | 21994 | 81 | NULL | NULL | 66d0c2b0a100 | 2026-01-04 14:43:13 |
| 40 | 60af54b0-9239-4e85-8bc9-be6d65998ba1 | 5000 | True | 21994 | 81 | NULL | NULL | 66d0c2b0a100 | 2026-01-04 14:59:41 |
| 41 | d6995220-b292-4c02-a0d5-d76d80301788 | 5000 | True | 3448 | 81 | NULL | NULL | 66d0c2b0a100 | 2026-01-04 15:07:52 |
| 42 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | True | 1727 | 81 | NULL | NULL | 66d0c2b0a100 | 2026-01-04 15:16:36 |
| 43 | c82be7b6-357e-4541-85a6-d02a4460c7d2 | 5000 | True | 2268 | 81 | NULL | NULL | 2abf45df7787 | 2026-01-04 15:51:50 |
| 44 | 66ab154c-8a81-4450-9180-8d859017d3b7 | 5000 | True | 2268 | 81 | NULL | NULL | 2abf45df7787 | 2026-01-04 16:06:43 |
| 52 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | True | 54949 | 81 | NULL | NULL | 2abf45df7787 | 2026-01-12 13:19:47 |
| 65 | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc | 5013 | True | 21604 | 81 | NULL | NULL | b575e5dcc188 | 2026-01-13 09:41:14 |
| 66 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | True | 1164 | 16 | NULL | NULL | 1598146da8ea | 2026-01-13 09:57:13 |

### Bottom 10 Records

| ID | RunID | EquipID | Passed | RowsValidated | ColumnsValidated | IssuesJSON | WarningsJSON | ContractSignature | ValidatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 68 | f8415fd2-68dd-4542-b08d-4885023c198d | 5010 | True | 511631 | 81 | NULL | NULL | d5dda024e44d | 2026-01-13 13:39:27 |
| 67 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | True | 21437 | 81 | NULL | NULL | 40548782ca90 | 2026-01-13 12:15:22 |
| 66 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | True | 1164 | 16 | NULL | NULL | 1598146da8ea | 2026-01-13 09:57:13 |
| 65 | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc | 5013 | True | 21604 | 81 | NULL | NULL | b575e5dcc188 | 2026-01-13 09:41:14 |
| 52 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | True | 54949 | 81 | NULL | NULL | 2abf45df7787 | 2026-01-12 13:19:47 |
| 44 | 66ab154c-8a81-4450-9180-8d859017d3b7 | 5000 | True | 2268 | 81 | NULL | NULL | 2abf45df7787 | 2026-01-04 16:06:43 |
| 43 | c82be7b6-357e-4541-85a6-d02a4460c7d2 | 5000 | True | 2268 | 81 | NULL | NULL | 2abf45df7787 | 2026-01-04 15:51:50 |
| 42 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | True | 1727 | 81 | NULL | NULL | 66d0c2b0a100 | 2026-01-04 15:16:36 |
| 41 | d6995220-b292-4c02-a0d5-d76d80301788 | 5000 | True | 3448 | 81 | NULL | NULL | 66d0c2b0a100 | 2026-01-04 15:07:52 |
| 40 | 60af54b0-9239-4e85-8bc9-be6d65998ba1 | 5000 | True | 21994 | 81 | NULL | NULL | 66d0c2b0a100 | 2026-01-04 14:59:41 |

---


## dbo.ACM_DataQuality

**Primary Key:** No primary key  
**Row Count:** 324  
**Date Range:** 2022-08-04 06:10:00 to 2022-10-28 01:30:00  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| sensor | train_count | train_nulls | train_null_pct | train_std | train_longest_gap | train_flatline_span | train_min_ts | train_max_ts | score_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| power_29_avg | 255815 | 77178 | 30.16945839767019 | 0.32729235758454733 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| power_29_max | 255815 | 77178 | 30.16945839767019 | 0.381173836195736 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| power_29_min | 255815 | 77178 | 30.16945839767019 | 0.22938250940776359 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| power_29_std | 255815 | 77178 | 30.16945839767019 | 0.05561500853616361 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| power_30_avg | 255815 | 77178 | 30.16945839767019 | 0.33072166447111134 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| power_30_max | 255815 | 77178 | 30.16945839767019 | 0.40082496719517646 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| power_30_min | 255815 | 77178 | 30.16945839767019 | 0.24118098560006523 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| power_30_std | 255815 | 77178 | 30.16945839767019 | 0.05592902186037983 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| reactive_power_27_avg | 255815 | 77178 | 30.16945839767019 | 0.21449756978333107 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |
| reactive_power_27_max | 255815 | 77178 | 30.16945839767019 | 0.23636881306714472 | 113 | 0 | 2022-10-28 01:30:00 | 2023-04-23 17:04:00 | 255816 |

### Bottom 10 Records

| sensor | train_count | train_nulls | train_null_pct | train_std | train_longest_gap | train_flatline_span | train_min_ts | train_max_ts | score_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| power_29_avg | 5170 | 0 | 0.0 | 0.40346681925776 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| power_29_max | 5170 | 0 | 0.0 | 0.41068368128537286 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| power_29_min | 5170 | 0 | 0.0 | 0.33953750061461707 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| power_29_std | 5170 | 0 | 0.0 | 0.04862856023062583 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| power_30_avg | 5170 | 0 | 0.0 | 0.39478082126859587 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| power_30_max | 5170 | 0 | 0.0 | 0.43661848531625425 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| power_30_min | 5170 | 0 | 0.0 | 0.3315238030756567 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| power_30_std | 5170 | 0 | 0.0 | 0.05160622001975781 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| reactive_power_27_avg | 5170 | 0 | 0.0 | 0.187861668204575 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |
| reactive_power_27_max | 5170 | 0 | 0.0 | 0.22100151489368985 | 0 | 0 | 2022-08-04 06:10:00 | 2022-09-10 07:30:00 | 3448 |

---


## dbo.ACM_DetectorCorrelation

**Primary Key:** ID  
**Row Count:** 278  
**Date Range:** 2026-01-04 10:07:43 to 2026-01-13 06:57:59  

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
| 331 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | ar1_z | ar1_z | 1.0 | 2026-01-04 10:07:43 |
| 332 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | ar1_z | pca_spe_z | 0.735186412405734 | 2026-01-04 10:07:43 |
| 333 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | ar1_z | pca_t2_z | 0.5475717643738854 | 2026-01-04 10:07:43 |
| 334 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | ar1_z | iforest_z | 0.7585925341518248 | 2026-01-04 10:07:43 |
| 335 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | ar1_z | gmm_z | 0.6784408469139491 | 2026-01-04 10:07:43 |
| 336 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | ar1_z | omr_z | 0.7099348528070111 | 2026-01-04 10:07:43 |
| 337 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | ar1_z | cusum_z | -0.008728643594403881 | 2026-01-04 10:07:43 |
| 338 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | pca_spe_z | ar1_z | 0.735186412405734 | 2026-01-04 10:07:43 |
| 339 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | pca_spe_z | pca_spe_z | 1.0 | 2026-01-04 10:07:43 |
| 340 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | pca_spe_z | pca_t2_z | 0.6709298307885977 | 2026-01-04 10:07:43 |

### Bottom 10 Records

| ID | RunID | EquipID | Detector1 | Detector2 | Correlation | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 1166 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | cusum_z | cusum_z | 1.0 | 2026-01-13 06:57:59 |
| 1165 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | cusum_z | omr_z | -0.01751616098162938 | 2026-01-13 06:57:59 |
| 1164 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | cusum_z | gmm_z | -0.04774646569368668 | 2026-01-13 06:57:59 |
| 1163 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | cusum_z | iforest_z | -0.07029039262229068 | 2026-01-13 06:57:59 |
| 1162 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | cusum_z | pca_t2_z | 0.13775971316555155 | 2026-01-13 06:57:59 |
| 1161 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | cusum_z | ar1_z | 0.12497446662865695 | 2026-01-13 06:57:59 |
| 1160 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | omr_z | cusum_z | -0.01751616098162938 | 2026-01-13 06:57:59 |
| 1159 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | omr_z | omr_z | 1.0 | 2026-01-13 06:57:59 |
| 1158 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | omr_z | gmm_z | 0.8627134553295833 | 2026-01-13 06:57:59 |
| 1157 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | omr_z | iforest_z | 0.7657909788874632 | 2026-01-13 06:57:59 |

---


## dbo.ACM_DriftController

**Primary Key:** ID  
**Row Count:** 7  

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

### Top 10 Records

| ID | RunID | EquipID | ControllerState | Threshold | Sensitivity | LastDriftValue | LastDriftTime | ResetCount | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | STABLE | 3.0 | 1.0 | NULL | NULL | NULL | 2026-01-04 10:07:28 |
| 9 | c82be7b6-357e-4541-85a6-d02a4460c7d2 | 5000 | STABLE | 3.0 | 1.0 | NULL | NULL | NULL | 2026-01-04 10:29:56 |
| 10 | 66ab154c-8a81-4450-9180-8d859017d3b7 | 5000 | STABLE | 3.0 | 1.0 | NULL | NULL | NULL | 2026-01-04 10:44:46 |
| 18 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | STABLE | 3.0 | 1.0 | NULL | NULL | NULL | 2026-01-12 09:45:32 |
| 28 | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc | 5013 | STABLE | 3.0 | 1.0 | NULL | NULL | NULL | 2026-01-13 04:23:39 |
| 29 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | STABLE | 3.0 | 1.0 | NULL | NULL | NULL | 2026-01-13 04:29:44 |
| 30 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | STABLE | 3.0 | 1.0 | NULL | NULL | NULL | 2026-01-13 06:57:22 |

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


## dbo.ACM_EpisodeCulprits

**Primary Key:** ID  
**Row Count:** 4,994  
**Date Range:** 2026-01-04 10:09:01 to 2026-01-13 07:00:24  

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
| 744 | 37F68854-5FA4-456A-8575-542F067F7E01 | 1 | Multivariate Outlier (PCA-T2) | NULL | 26.010852813720703 | 1 | 2026-01-04 10:09:01 | 5000 |
| 745 | 37F68854-5FA4-456A-8575-542F067F7E01 | 1 | Correlation Break (PCA-SPE) | NULL | 22.519622802734375 | 2 | 2026-01-04 10:09:01 | 5000 |
| 746 | 37F68854-5FA4-456A-8575-542F067F7E01 | 1 | Time-Series Anomaly (AR1) | NULL | 16.338924407958984 | 3 | 2026-01-04 10:09:01 | 5000 |
| 747 | 37F68854-5FA4-456A-8575-542F067F7E01 | 1 | Rare State (IsolationForest) | NULL | 14.575786590576172 | 4 | 2026-01-04 10:09:01 | 5000 |
| 748 | 37F68854-5FA4-456A-8575-542F067F7E01 | 1 | Density Anomaly (GMM) | NULL | 10.894606590270996 | 5 | 2026-01-04 10:09:01 | 5000 |
| 749 | 37F68854-5FA4-456A-8575-542F067F7E01 | 1 | Baseline Consistency (OMR) | NULL | 5.72343111038208 | 6 | 2026-01-04 10:09:01 | 5000 |
| 750 | 37F68854-5FA4-456A-8575-542F067F7E01 | 1 | cusum_z | NULL | 3.9367730617523193 | 7 | 2026-01-04 10:09:01 | 5000 |
| 751 | 37F68854-5FA4-456A-8575-542F067F7E01 | 2 | Correlation Break (PCA-SPE) | NULL | 28.163358688354492 | 1 | 2026-01-04 10:09:01 | 5000 |
| 752 | 37F68854-5FA4-456A-8575-542F067F7E01 | 2 | Multivariate Outlier (PCA-T2) | NULL | 19.64913558959961 | 2 | 2026-01-04 10:09:01 | 5000 |
| 753 | 37F68854-5FA4-456A-8575-542F067F7E01 | 2 | Time-Series Anomaly (AR1) | NULL | 14.208327293395996 | 3 | 2026-01-04 10:09:01 | 5000 |

### Bottom 10 Records

| ID | RunID | EpisodeID | DetectorType | SensorName | ContributionPct | Rank | CreatedAt | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14852 | FC487AD0-DD33-43E6-9833-D737537C178F | 209 | Time-Series Anomaly (AR1) | NULL | 8.563364028930664 | 5 | 2026-01-13 07:00:24 | 5010 |
| 14851 | FC487AD0-DD33-43E6-9833-D737537C178F | 209 | cusum_z | NULL | 16.370264053344727 | 4 | 2026-01-13 07:00:24 | 5010 |
| 14850 | FC487AD0-DD33-43E6-9833-D737537C178F | 209 | Rare State (IsolationForest) | NULL | 16.8203125 | 3 | 2026-01-13 07:00:24 | 5010 |
| 14849 | FC487AD0-DD33-43E6-9833-D737537C178F | 209 | Baseline Consistency (OMR) | NULL | 26.50141716003418 | 2 | 2026-01-13 07:00:24 | 5010 |
| 14848 | FC487AD0-DD33-43E6-9833-D737537C178F | 209 | Density Anomaly (GMM) | NULL | 31.74464988708496 | 1 | 2026-01-13 07:00:24 | 5010 |
| 14847 | FC487AD0-DD33-43E6-9833-D737537C178F | 208 | Time-Series Anomaly (AR1) | NULL | 9.438705444335938 | 5 | 2026-01-13 07:00:24 | 5010 |
| 14846 | FC487AD0-DD33-43E6-9833-D737537C178F | 208 | cusum_z | NULL | 12.893841743469238 | 4 | 2026-01-13 07:00:24 | 5010 |
| 14845 | FC487AD0-DD33-43E6-9833-D737537C178F | 208 | Rare State (IsolationForest) | NULL | 16.14639663696289 | 3 | 2026-01-13 07:00:24 | 5010 |
| 14844 | FC487AD0-DD33-43E6-9833-D737537C178F | 208 | Baseline Consistency (OMR) | NULL | 24.107288360595703 | 2 | 2026-01-13 07:00:24 | 5010 |
| 14843 | FC487AD0-DD33-43E6-9833-D737537C178F | 208 | Density Anomaly (GMM) | NULL | 37.41375732421875 | 1 | 2026-01-13 07:00:24 | 5010 |

---


## dbo.ACM_EpisodeDiagnostics

**Primary Key:** ID  
**Row Count:** 875  
**Date Range:** 2022-08-12 17:00:00 to 2024-06-15 21:59:00  

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
| 122 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 1 | 2022-08-22 13:40:00 | 2022-08-22 16:20:00 | 2.6666666666666665 | 8.331068407790504 | 5.309108704630257 | CRITICAL |
| 123 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 2 | 2022-08-23 00:30:00 | 2022-08-23 06:30:00 | 6.0 | 8.465417541388035 | 5.668386165820579 | CRITICAL |
| 124 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 3 | 2022-08-23 12:50:00 | 2022-08-23 15:30:00 | 2.6666666666666665 | 8.516799186656753 | 5.49055098886961 | CRITICAL |
| 125 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 4 | 2022-08-24 00:00:00 | 2022-08-24 02:40:00 | 2.6666666666666665 | 8.245831213885173 | 5.583750963599811 | CRITICAL |
| 126 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 5 | 2022-08-24 11:50:00 | 2022-08-24 14:40:00 | 2.8333333333333335 | 8.822206348047812 | 5.419071526563553 | CRITICAL |
| 127 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 6 | 2022-08-25 01:10:00 | 2022-08-25 03:50:00 | 2.6666666666666665 | 9.020777505700126 | 5.612196749912451 | CRITICAL |
| 128 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 7 | 2022-08-25 11:50:00 | 2022-08-25 14:40:00 | 2.8333333333333335 | 8.625531293383336 | 5.418249036869019 | CRITICAL |
| 129 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 8 | 2022-08-25 21:00:00 | 2022-08-26 02:30:00 | 5.5 | 8.372887006019685 | 4.360487497262375 | CRITICAL |
| 130 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 9 | 2022-08-26 14:50:00 | 2022-08-26 17:30:00 | 2.6666666666666665 | 8.236208523829367 | 5.354827186577198 | CRITICAL |
| 131 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 10 | 2022-08-26 23:00:00 | 2022-08-27 01:40:00 | 2.6666666666666665 | 8.593787210522253 | 5.716410202509885 | CRITICAL |

### Bottom 10 Records

| ID | RunID | EquipID | EpisodeID | StartTime | EndTime | DurationHours | PeakZ | AvgZ | Severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2653 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 209 | 2023-10-18 03:40:00 | 2023-10-18 04:40:00 | 1.0 | 1.7944104921589814 | 1.031290323074823 | LOW |
| 2652 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 208 | 2023-10-17 04:20:00 | 2023-10-17 05:30:00 | 1.1666666666666667 | 2.1996770743768455 | 1.3535285756863882 | MEDIUM |
| 2651 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 207 | 2023-10-15 09:40:00 | 2023-10-15 16:20:00 | 6.666666666666667 | 4.151242125134404 | 1.6417102839532682 | HIGH |
| 2650 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 206 | 2023-10-12 14:10:00 | 2023-10-12 16:00:00 | 1.8333333333333333 | 1.60482795157558 | 1.0352386175110886 | LOW |
| 2649 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 205 | 2023-10-12 00:50:00 | 2023-10-12 10:00:00 | 9.166666666666666 | 4.004022084198785 | 3.0283045608370522 | HIGH |
| 2648 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 204 | 2023-10-11 21:00:00 | 2023-10-11 23:40:00 | 2.6666666666666665 | 2.052013642512483 | 1.4603779299410438 | MEDIUM |
| 2647 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 203 | 2023-10-10 08:20:00 | 2023-10-10 13:20:00 | 5.0 | 1.862735083882085 | 0.9908065363322334 | LOW |
| 2646 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 202 | 2023-10-10 00:10:00 | 2023-10-10 01:50:00 | 1.6666666666666667 | 2.301390860692317 | 1.215574071978101 | MEDIUM |
| 2645 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 201 | 2023-10-09 15:10:00 | 2023-10-09 16:40:00 | 1.5 | 2.399135916466908 | 1.7092452303397478 | MEDIUM |
| 2644 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 200 | 2023-10-09 09:40:00 | 2023-10-09 12:30:00 | 2.8333333333333335 | 1.040968364129126 | 0.6477698314982758 | LOW |

---


## dbo.ACM_Episodes

**Primary Key:** ID  
**Row Count:** 875  
**Date Range:** 2022-08-12 17:00:00 to 2024-06-15 21:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| EpisodeID | int | NO | 10 | — |
| StartTime | datetime2 | NO | — | — |
| EndTime | datetime2 | YES | — | — |
| DurationSeconds | float | YES | 53 | — |
| DurationHours | float | YES | 53 | — |
| RecordCount | int | YES | 10 | — |
| Culprits | nvarchar | YES | 512 | — |
| PrimaryDetector | nvarchar | YES | 64 | — |
| Severity | nvarchar | YES | 16 | — |
| RegimeLabel | int | YES | 10 | — |
| RegimeState | nvarchar | YES | 32 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| ID | RunID | EquipID | EpisodeID | StartTime | EndTime | DurationSeconds | DurationHours | RecordCount | Culprits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 1 | 2022-08-22 13:40:00 | 2022-08-22 16:20:00 | 9600.0 | 2.6666666666666665 | 1 | Multivariate Outlier (PCA-T2) -> sensor |
| 6 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2 | 2022-08-23 00:30:00 | 2022-08-23 06:30:00 | 21600.0 | 6.0 | 1 | Correlation Break (PCA-SPE) -> sensor |
| 7 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 3 | 2022-08-23 12:50:00 | 2022-08-23 15:30:00 | 9600.0 | 2.6666666666666665 | 1 | Multivariate Outlier (PCA-T2) -> sensor |
| 8 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 4 | 2022-08-24 00:00:00 | 2022-08-24 02:40:00 | 9600.0 | 2.6666666666666665 | 1 | Multivariate Outlier (PCA-T2) -> sensor |
| 9 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 5 | 2022-08-24 11:50:00 | 2022-08-24 14:40:00 | 10200.0 | 2.8333333333333335 | 1 | Multivariate Outlier (PCA-T2) -> sensor |
| 10 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 6 | 2022-08-25 01:10:00 | 2022-08-25 03:50:00 | 9600.0 | 2.6666666666666665 | 1 | Correlation Break (PCA-SPE) -> sensor |
| 11 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 7 | 2022-08-25 11:50:00 | 2022-08-25 14:40:00 | 10200.0 | 2.8333333333333335 | 1 | Multivariate Outlier (PCA-T2) -> sensor |
| 12 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 8 | 2022-08-25 21:00:00 | 2022-08-26 02:30:00 | 19800.0 | 5.5 | 1 | Multivariate Outlier (PCA-T2) -> sensor |
| 13 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 9 | 2022-08-26 14:50:00 | 2022-08-26 17:30:00 | 9600.0 | 2.6666666666666665 | 1 | Multivariate Outlier (PCA-T2) -> sensor |
| 14 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 10 | 2022-08-26 23:00:00 | 2022-08-27 01:40:00 | 9600.0 | 2.6666666666666665 | 1 | Multivariate Outlier (PCA-T2) -> power |

### Bottom 10 Records

| ID | RunID | EquipID | EpisodeID | StartTime | EndTime | DurationSeconds | DurationHours | RecordCount | Culprits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2536 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 209 | 2023-10-18 03:40:00 | 2023-10-18 04:40:00 | 3600.0 | 1.0 | 1 | Density Anomaly (GMM) |
| 2535 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 208 | 2023-10-17 04:20:00 | 2023-10-17 05:30:00 | 4200.0 | 1.1666666666666667 | 1 | Density Anomaly (GMM) |
| 2534 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 207 | 2023-10-15 09:40:00 | 2023-10-15 16:20:00 | 24000.0 | 6.666666666666667 | 1 | Time-Series Anomaly (AR1) |
| 2533 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 206 | 2023-10-12 14:10:00 | 2023-10-12 16:00:00 | 6600.0 | 1.8333333333333333 | 1 | Density Anomaly (GMM) |
| 2532 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 205 | 2023-10-12 00:50:00 | 2023-10-12 10:00:00 | 33000.0 | 9.166666666666666 | 1 | Time-Series Anomaly (AR1) |
| 2531 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 204 | 2023-10-11 21:00:00 | 2023-10-11 23:40:00 | 9600.0 | 2.6666666666666665 | 1 | Density Anomaly (GMM) |
| 2530 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 203 | 2023-10-10 08:20:00 | 2023-10-10 13:20:00 | 18000.0 | 5.0 | 1 | Baseline Consistency (OMR) |
| 2529 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 202 | 2023-10-10 00:10:00 | 2023-10-10 01:50:00 | 6000.0 | 1.6666666666666667 | 1 | Time-Series Anomaly (AR1) |
| 2528 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 201 | 2023-10-09 15:10:00 | 2023-10-09 16:40:00 | 5400.0 | 1.5 | 1 | Density Anomaly (GMM) |
| 2527 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 200 | 2023-10-09 09:40:00 | 2023-10-09 12:30:00 | 10200.0 | 2.8333333333333335 | 1 | Time-Series Anomaly (AR1) |

---


## dbo.ACM_FailureForecast

**Primary Key:** EquipID, RunID, Timestamp  
**Row Count:** 3,192  
**Date Range:** 2022-09-04 06:20:00 to 2024-06-23 00:59:00  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| EquipID | RunID | Timestamp | FailureProb | SurvivalProb | HazardRate | ThresholdUsed | Method | CreatedAt | ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 01:59:00 | 0.0019099101649621127 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32929 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 02:59:00 | 0.0017839339718525714 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32930 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 03:59:00 | 0.0016655622499029753 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32931 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 04:59:00 | 0.0015543867288166358 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32932 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 05:59:00 | 0.0014500175887198368 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32933 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 06:59:00 | 0.001352082851992361 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32934 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 07:59:00 | 0.001260227780761153 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32935 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 08:59:00 | 0.0011741142810410116 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32936 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 09:59:00 | 0.0010934203144285727 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32937 |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 10:59:00 | 0.001017839318179159 | 0.9980900898350379 | 0.0 | 70.0 | HoltWinters | 2026-01-13 10:00:45 | 32938 |

### Bottom 10 Records

| EquipID | RunID | Timestamp | FailureProb | SurvivalProb | HazardRate | ThresholdUsed | Method | CreatedAt | ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 10:00:00 | 2.6700982892776507e-258 | 1.0 | 1.5260023880708735e-258 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32928 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 09:45:00 | 2.2885976922599324e-258 | 1.0 | 1.3081269122675606e-258 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32927 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 09:30:00 | 1.9615659641930422e-258 | 1.0 | 1.1213360767667122e-258 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32926 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 09:15:00 | 1.6812319450013642e-258 | 1.0 | 9.611982167854021e-259 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32925 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 09:00:00 | 1.4409323908050136e-258 | 1.0 | 8.239130182979007e-259 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32924 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 08:45:00 | 1.2349541362305384e-258 | 1.0 | 7.062216394059388e-259 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32923 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 08:30:00 | 1.0583987263790537e-258 | 1.0 | 6.053296173913482e-259 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32922 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 08:15:00 | 9.070663220312167e-259 | 1.0 | 5.1884074389369185e-259 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32921 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 08:00:00 | 7.773561360577937e-259 | 1.0 | 4.447003490228409e-259 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32920 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 07:45:00 | 6.661810488020835e-259 | 1.0 | 3.811466564707953e-259 | 70.0 | HoltWinters | 2026-01-13 09:55:28 | 32919 |

---


## dbo.ACM_FeatureDropLog

**Primary Key:** ID  
**Row Count:** 146  
**Date Range:** 2026-01-04 08:02:07 to 2026-01-13 08:13:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | NO | 10 | — |
| FeatureName | nvarchar | NO | 200 | — |
| DropReason | nvarchar | NO | 100 | — |
| DropValue | float | YES | 53 | — |
| Threshold | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | FeatureName | DropReason | DropValue | Threshold | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 610 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_49_med | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 611 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_46_mad | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 612 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_49_slope | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 613 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_49_std | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 614 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_46_skew | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 615 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_46_med | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 616 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_49_kurt | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 617 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_46_kurt | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 618 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_49_mad | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |
| 619 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | sensor_46_slope | low_variance | 0.0 | NULL | 2026-01-04 08:02:07 |

### Bottom 10 Records

| ID | RunID | EquipID | FeatureName | DropReason | DropValue | Threshold | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1051 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_46_mad | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1050 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_49_rz | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1049 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_46_skew | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1048 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_46_slope | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1047 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_49_mad | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1046 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_49_kurt | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1045 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_49_skew | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1044 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_49_mean | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1043 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_46_kurt | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |
| 1042 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | sensor_46_std | low_variance | 0.0 | NULL | 2026-01-13 08:13:00 |

---


## dbo.ACM_ForecastState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
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


## dbo.ACM_Forecast_QualityMetrics

**Primary Key:** MetricID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| MetricID | int | NO | 10 | — |
| RunID | varchar | NO | 100 | — |
| EquipID | int | NO | 10 | — |
| RMSE | float | YES | 53 | — |
| MAE | float | YES | 53 | — |
| MAPE | float | YES | 53 | — |
| R2Score | float | YES | 53 | — |
| DataHash | varchar | YES | 32 | — |
| ModelVersion | int | YES | 10 | — |
| RetrainTriggered | bit | NO | — | ((0)) |
| RetrainReason | varchar | YES | 200 | — |
| ForecastHorizonHours | float | NO | 53 | — |
| SampleCount | int | YES | 10 | — |
| ComputeTimestamp | datetime2 | NO | — | (getdate()) |
| CreatedAt | datetime2 | NO | — | (getdate()) |

---


## dbo.ACM_ForecastingState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 4  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
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
| CreatedAt | datetime2 | NO | — | (getdate()) |
| UpdatedAt | datetime2 | NO | — | (getdate()) |

### Top 10 Records

| ID | EquipID | StateVersion | ModelCoefficientsJson | LastForecastJson | LastRetrainTime | TrainingDataHash | DataVolumeAnalyzed | RecentMAE | RecentRMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 5000 | 1 | {"alpha": 0.7, "beta": 0.01, "level": 83.34365479593721, "trend": -0.06256213907819105, "std_erro... | {"forecast_mean": 62.29149499612591, "forecast_std": 12.136396555918049, "forecast_range": 41.979... | NULL |  | 15954 | 12.136396555918049 | NULL |
| 14 | 5013 | 1 | {"alpha": 0.05, "beta": 0.01, "level": 93.59188413209672, "trend": -0.002837340826633356, "std_er... | {"forecast_mean": 92.6371189439346, "forecast_std": 0.5504142592260355, "forecast_range": 1.90385... | NULL |  | 8636 | 0.5504142592260355 | NULL |
| 15 | 2621 | 1 | {"alpha": 0.1, "beta": 0.13, "level": 87.1175468034547, "trend": 0.1273838924572157, "std_error":... | {"forecast_mean": 96.16087295950027, "forecast_std": 4.250800177934946, "forecast_range": 12.7550... | NULL |  | 722 | 4.250800177934946 | NULL |
| 16 | 5010 | 1 | {"alpha": 0.05, "beta": 0.01, "level": 93.51456884249477, "trend": -0.0006207120376706277, "std_e... | {"forecast_mean": 93.3056992418186, "forecast_std": 0.12041160272329507, "forecast_range": 0.4164... | NULL |  | 8621 | 0.12041160272329507 | NULL |

---


## dbo.ACM_HealthForecast

**Primary Key:** EquipID, RunID, Timestamp  
**Row Count:** 5,208  
**Date Range:** 2022-09-04 06:20:00 to 2024-06-23 00:59:00  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| EquipID | RunID | Timestamp | ForecastHealth | CiLower | CiUpper | ForecastStd | Method | CreatedAt | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 01:59:00 | 87.24493069591192 | 82.22752521229812 | 92.26233617952572 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 02:59:00 | 87.37231458836914 | 82.32297709107506 | 92.42165208566321 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 03:59:00 | 87.49969848082635 | 82.41093858613625 | 92.58845837551645 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 04:59:00 | 87.62708237328357 | 82.49075381753396 | 92.76341092903318 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 05:59:00 | 87.75446626574077 | 82.56182730323627 | 92.94710522824528 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 06:59:00 | 87.88185015819799 | 82.6236307556178 | 93.14006956077819 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 07:59:00 | 88.00923405065521 | 82.67570842141201 | 93.3427596798984 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 08:59:00 | 88.13661794311243 | 82.71768065022727 | 93.55555523599759 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 09:59:00 | 88.26400183556964 | 82.74924549844239 | 93.7787581726969 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2024-06-16 10:59:00 | 88.39138572802686 | 82.77017831398163 | 94.01259314207209 | 2.5087027418069 | ExponentialSmoothing | 2026-01-13 10:00:43 | NULL |

### Bottom 10 Records

| EquipID | RunID | Timestamp | ForecastHealth | CiLower | CiUpper | ForecastStd | Method | CreatedAt | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 10:00:00 | 91.68519109659911 | 88.35560357747522 | 95.014778615723 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 09:45:00 | 91.68802843742574 | 88.36476108468628 | 95.0112957901652 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 09:30:00 | 91.69086577825237 | 88.37391420353008 | 95.00781735297467 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 09:15:00 | 91.693703119079 | 88.3830629302569 | 95.00434330790111 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 09:00:00 | 91.69654045990563 | 88.39220726110592 | 95.00087365870534 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 08:45:00 | 91.69937780073228 | 88.40134719230508 | 94.99740840915948 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 08:30:00 | 91.70221514155891 | 88.41048272007106 | 94.99394756304676 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 08:15:00 | 91.70505248238554 | 88.41961384060924 | 94.99049112416184 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 08:00:00 | 91.70788982321217 | 88.42874055011362 | 94.98703909631072 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 2023-06-01 07:45:00 | 91.71072716403881 | 88.4378628447668 | 94.98359148331083 | 0.2659585169840568 | ExponentialSmoothing | 2026-01-13 09:55:26 | NULL |

---


## dbo.ACM_HealthTimeline

**Primary Key:** No primary key  
**Row Count:** 74,541  
**Date Range:** 2022-08-12 03:10:00 to 2024-06-16 00:59:00  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| Timestamp | HealthIndex | HealthZone | FusedZ | RunID | EquipID | RawHealthIndex | QualityFlag | Confidence | ConfidenceFactors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-08-12 03:10:00 | 90.89 | GOOD | -0.5831000208854675 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 90.88999938964844 | NORMAL | 0.329 | NULL |
| 2022-08-12 03:20:00 | 92.08 | GOOD | 0.0697999969124794 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 94.86000061035156 | NORMAL | 0.304 | NULL |
| 2022-08-12 03:30:00 | 92.71 | GOOD | -0.18160000443458557 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 94.16999816894531 | NORMAL | 0.31 | NULL |
| 2022-08-12 03:40:00 | 92.31 | GOOD | -0.5342000126838684 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 91.36000061035156 | NORMAL | 0.328 | NULL |
| 2022-08-12 03:50:00 | 93.18 | GOOD | 0.006099999882280827 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 95.22000122070312 | NORMAL | 0.301 | NULL |
| 2022-08-12 04:00:00 | 93.15 | GOOD | -0.3330000042915344 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 93.08999633789062 | NORMAL | 0.318 | NULL |
| 2022-08-12 04:10:00 | 92.54 | GOOD | -0.5612000226974487 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 91.11000061035156 | NORMAL | 0.329 | NULL |
| 2022-08-12 04:20:00 | 91.76 | GOOD | -0.6747000217437744 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 89.94000244140625 | NORMAL | 0.335 | NULL |
| 2022-08-12 04:30:00 | 91.32 | GOOD | -0.6402999758720398 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 90.30999755859375 | NORMAL | 0.334 | NULL |
| 2022-08-12 04:40:00 | 91.45 | GOOD | -0.49480000138282776 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 91.7300033569336 | NORMAL | 0.327 | NULL |

### Bottom 10 Records

| Timestamp | HealthIndex | HealthZone | FusedZ | RunID | EquipID | RawHealthIndex | QualityFlag | Confidence | ConfidenceFactors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-06-16 00:59:00 | 53.14 | ALERT | 2.4951000213623047 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 50.150001525878906 | NORMAL | 0.625 | NULL |
| 2024-06-15 23:59:00 | 54.42 | ALERT | 2.654900074005127 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 45.369998931884766 | NORMAL | 0.633 | NULL |
| 2024-06-15 22:59:00 | 58.3 | ALERT | 3.0487000942230225 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 34.11000061035156 | NORMAL | 0.652 | NULL |
| 2024-06-15 21:59:00 | 68.67 | ALERT | 3.3915998935699463 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 25.540000915527344 | NORMAL | 0.67 | NULL |
| 2024-06-15 20:59:00 | 87.16 | GOOD | -0.3878999948501587 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 92.6500015258789 | NORMAL | 0.519 | NULL |
| 2024-06-15 19:59:00 | 84.8 | WATCH | -0.24539999663829803 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 93.73999786376953 | NORMAL | 0.512 | NULL |
| 2024-06-15 18:59:00 | 80.98 | WATCH | 0.9308000206947327 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 86.80000305175781 | NORMAL | 0.547 | NULL |
| 2024-06-15 17:59:00 | 78.48 | WATCH | 2.242500066757202 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 57.65999984741211 | NORMAL | 0.612 | NULL |
| 2024-06-15 16:59:00 | 87.4 | GOOD | 1.5025999546051025 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 76.80000305175781 | NORMAL | 0.575 | NULL |
| 2024-06-15 15:59:00 | 91.95 | GOOD | 0.6710000038146973 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 89.9800033569336 | NORMAL | 0.534 | NULL |

---


## dbo.ACM_HistorianData

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| EquipID | int | NO | 10 | — |
| EntryDateTime | datetime2 | NO | — | — |
| SensorName | nvarchar | NO | 128 | — |
| SensorValue | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_MultivariateForecast

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| ForecastTime | datetime2 | NO | — | — |
| SensorName | nvarchar | NO | 128 | — |
| ForecastValue | float | YES | 53 | — |
| CI_Lower | float | YES | 53 | — |
| CI_Upper | float | YES | 53 | — |
| CorrelationGroup | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_OMR_Diagnostics

**Primary Key:** DiagnosticID  
**Row Count:** 10  

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
| 12 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | pls | 5 | 32991 | 632 | 4.269288813967508 | NULL | NULL |
| 14 | ccdd4d7f-3acc-4cf4-9987-d729fd313c6f | 5000 | pls | 5 | 32991 | 632 | 4.269288813967508 | NULL | NULL |
| 15 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | pls | 5 | 2589 | 632 | 4.700902675279064 | NULL | NULL |
| 16 | c82be7b6-357e-4541-85a6-d02a4460c7d2 | 5000 | pls | 5 | 1134 | 632 | 5.120508594845961 | NULL | NULL |
| 17 | 66ab154c-8a81-4450-9180-8d859017d3b7 | 5000 | pls | 5 | 1134 | 632 | 5.120508594845961 | NULL | NULL |
| 25 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | pls | 5 | 27474 | 632 | 4.265885334101474 | NULL | NULL |
| 37 | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc | 5013 | pls | 5 | 32405 | 630 | 4.330518683848428 | NULL | NULL |
| 38 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | pls | 5 | 1746 | 128 | 2.3124458607966063 | NULL | NULL |
| 39 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | pls | 5 | 32154 | 632 | 4.128491124782325 | NULL | NULL |
| 40 | f8415fd2-68dd-4542-b08d-4885023c198d | 5010 | pls | 5 | 255815 | 632 | 3.827027089539472 | NULL | NULL |

---


## dbo.ACM_PCA_Loadings

**Primary Key:** ID  
**Row Count:** 16,430  
**Date Range:** 2026-01-04 16:01:22 to 2026-01-13 12:29:45  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| ComponentIndex | int | YES | 10 | — |
| SensorName | nvarchar | YES | 100 | — |
| Loading | float | NO | 53 | — |
| AbsLoading | float | NO | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| ID | RunID | EquipID | ComponentIndex | SensorName | Loading | AbsLoading | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16816 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | power_29_avg_med | -4.2813195516459995e-07 | 4.2813195516459995e-07 | 2026-01-04 16:01:22 |
| 16817 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | power_29_max_med | -5.007032640724773e-07 | 5.007032640724773e-07 | 2026-01-04 16:01:22 |
| 16818 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | power_29_min_med | -3.355974848295862e-07 | 3.355974848295862e-07 | 2026-01-04 16:01:22 |
| 16819 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | power_29_std_med | -6.061275659961785e-07 | 6.061275659961785e-07 | 2026-01-04 16:01:22 |
| 16820 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | power_30_avg_med | -5.405751546364197e-07 | 5.405751546364197e-07 | 2026-01-04 16:01:22 |
| 16821 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | power_30_max_med | -5.872465929606909e-07 | 5.872465929606909e-07 | 2026-01-04 16:01:22 |
| 16822 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | power_30_min_med | -4.403284106235069e-07 | 4.403284106235069e-07 | 2026-01-04 16:01:22 |
| 16823 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | power_30_std_med | -5.850246781840352e-07 | 5.850246781840352e-07 | 2026-01-04 16:01:22 |
| 16824 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | reactive_power_27_avg_med | -8.108253439215521e-07 | 8.108253439215521e-07 | 2026-01-04 16:01:22 |
| 16825 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 1 | reactive_power_27_max_med | -9.063464237139544e-07 | 9.063464237139544e-07 | 2026-01-04 16:01:22 |

### Bottom 10 Records

| ID | RunID | EquipID | ComponentIndex | SensorName | Loading | AbsLoading | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 81255 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | wind_speed_4_avg_rz | -3.955433614519692e-07 | 3.955433614519692e-07 | 2026-01-13 12:29:45 |
| 81254 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | wind_speed_3_std_rz | -9.000479985891717e-07 | 9.000479985891717e-07 | 2026-01-13 12:29:45 |
| 81253 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | wind_speed_3_min_rz | -3.45216924587101e-07 | 3.45216924587101e-07 | 2026-01-13 12:29:45 |
| 81252 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | wind_speed_3_max_rz | -3.02650137311857e-07 | 3.02650137311857e-07 | 2026-01-13 12:29:45 |
| 81251 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | wind_speed_3_avg_rz | -4.223071688024805e-07 | 4.223071688024805e-07 | 2026-01-13 12:29:45 |
| 81250 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | sensor_9_avg_rz | 4.3200346408848685e-07 | 4.3200346408848685e-07 | 2026-01-13 12:29:45 |
| 81249 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | sensor_8_avg_rz | 2.9259588581085613e-07 | 2.9259588581085613e-07 | 2026-01-13 12:29:45 |
| 81248 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | sensor_7_avg_rz | 3.4902542201145343e-07 | 3.4902542201145343e-07 | 2026-01-13 12:29:45 |
| 81247 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | sensor_6_avg_rz | -3.8499578582254754e-08 | 3.8499578582254754e-08 | 2026-01-13 12:29:45 |
| 81246 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | sensor_5_std_rz | -1.520052108142878e-06 | 1.520052108142878e-06 | 2026-01-13 12:29:45 |

---


## dbo.ACM_PCA_Metrics

**Primary Key:** ID  
**Row Count:** 10  
**Date Range:** 2026-01-04 08:02:35 to 2026-01-13 08:14:08  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| NComponents | int | NO | 10 | — |
| ExplainedVariance | float | YES | 53 | — |
| ComponentsJson | nvarchar | YES | -1 | — |
| MetricType | nvarchar | YES | 50 | — |
| TrainSamples | int | YES | 10 | — |
| TrainFeatures | int | YES | 10 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| ID | RunID | EquipID | NComponents | ExplainedVariance | ComponentsJson | MetricType | TrainSamples | TrainFeatures | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-04 08:02:35 |
| 15 | CCDD4D7F-3ACC-4CF4-9987-D729FD313C6F | 5000 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-04 09:16:43 |
| 16 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-04 09:49:20 |
| 17 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-04 10:24:39 |
| 18 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-04 10:39:25 |
| 26 | F7849950-AE3E-42E4-8AB2-B90A562008DA | 5000 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-12 07:53:03 |
| 38 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 5013 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-13 04:14:32 |
| 39 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-13 04:27:56 |
| 40 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-13 06:48:42 |
| 41 | F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-13 08:14:08 |

---


## dbo.ACM_PCA_Models

**Primary Key:** ID  
**Row Count:** 5  
**Date Range:** 2026-01-04 16:17:03 to 2026-01-13 12:29:42  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| ModelVersion | int | NO | 10 | ((1)) |
| NComponents | int | NO | 10 | — |
| ExplainedVarianceRatio | float | YES | 53 | — |
| TrainSamples | int | YES | 10 | — |
| TrainFeatures | int | YES | 10 | — |
| ScalerMeanJson | nvarchar | YES | -1 | — |
| ScalerScaleJson | nvarchar | YES | -1 | — |
| ComponentsJson | nvarchar | YES | -1 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| ID | RunID | EquipID | ModelVersion | NComponents | ExplainedVarianceRatio | TrainSamples | TrainFeatures | ScalerMeanJson | ScalerScaleJson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 10 | 5 | 0.999990793801205 | NULL | NULL | NULL | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} |
| 9 | F7849950-AE3E-42E4-8AB2-B90A562008DA | 5000 | 10 | 5 | 0.9999993022784295 | NULL | NULL | NULL | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} |
| 19 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 5013 | 10 | 5 | 0.9999999958935263 | NULL | NULL | NULL | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} |
| 20 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 10 | 5 | 0.8629530206328949 | NULL | NULL | NULL | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} |
| 21 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 10 | 5 | 0.9961597879999842 | NULL | NULL | NULL | {"scaler": "RobustStandardScaler", "with_mean": true, "with_std": true} |

---


## dbo.ACM_RUL

**Primary Key:** EquipID, RunID  
**Row Count:** 7  
**Date Range:** 2026-01-11 15:38:23 to 2026-01-20 12:29:18  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| EquipID | RunID | RUL_Hours | P10_LowerBound | P50_Median | P90_UpperBound | Confidence | FailureTime | Method | NumSimulations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2621 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 2026-01-13 10:00:47 | Multipath | 1000 |
| 5000 | 37F68854-5FA4-456A-8575-542F067F7E01 | 168.0 | 163.00985373523335 | 168.0 | 172.99014626476665 | 0.3 | 2026-01-11 15:38:23 | Multipath | 1000 |
| 5000 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 168.0 | 163.00985373523335 | 168.0 | 172.99014626476665 | 0.3 | 2026-01-11 16:16:40 | Multipath | 1000 |
| 5000 | F7849950-AE3E-42E4-8AB2-B90A562008DA | 46.125 | 40.94652278349314 | 46.125 | 50.19803351432961 | 0.3 | 2026-01-14 13:30:05 | Multipath | 1000 |
| 5000 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 168.0 | 163.00985373523335 | 168.0 | 172.99014626476665 | 0.3 | 2026-01-11 16:00:50 | Multipath | 1000 |
| 5010 | FC487AD0-DD33-43E6-9833-D737537C178F | 168.0 | 163.00985373523335 | 168.0 | 172.99014626476665 | 0.3 | 2026-01-20 12:29:18 | Multipath | 1000 |
| 5013 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 168.0 | 163.00985373523335 | 168.0 | 172.99014626476665 | 0.3 | 2026-01-20 09:55:30 | Multipath | 1000 |

---


## dbo.ACM_RefitRequests

**Primary Key:** RequestID  
**Row Count:** 7  
**Date Range:** 2026-01-04 10:07:26 to 2026-01-13 06:57:15  

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
| 9 | 5000 | 2026-01-04 10:07:26 | Anomaly rate too high; Silhouette score too low; anomaly_rate=40.01% > 25.00% | 0.40011580775911987 | NULL | NULL | 0.0 | True | 2026-01-04 10:24:18 |
| 10 | 5000 | 2026-01-04 10:29:55 | Anomaly rate too high; Silhouette score too low; anomaly_rate=42.42% > 25.00% | 0.42416225749559083 | NULL | NULL | 0.0 | True | 2026-01-04 10:39:05 |
| 11 | 5000 | 2026-01-04 10:44:45 | Anomaly rate too high; Silhouette score too low; anomaly_rate=42.42% > 25.00% | 0.42416225749559083 | NULL | NULL | 0.0 | True | 2026-01-12 07:52:39 |
| 19 | 5000 | 2026-01-12 09:45:24 | Anomaly rate too high; Silhouette score too low; anomaly_rate=31.63% > 25.00% | 0.316287534121929 | NULL | NULL | 0.0 | False | NULL |
| 29 | 5013 | 2026-01-13 04:23:33 | Anomaly rate too high; Silhouette score too low; anomaly_rate=32.15% > 25.00% | 0.3214682466209961 | NULL | NULL | 0.0 | False | NULL |
| 30 | 2621 | 2026-01-13 04:29:42 | Anomaly rate too high; Silhouette score too low; anomaly_rate=26.63% > 25.00% | 0.2663230240549828 | NULL | NULL | 0.0 | False | NULL |
| 31 | 5010 | 2026-01-13 06:57:15 | Silhouette score too low | NULL | NULL | NULL | 0.0 | True | 2026-01-13 08:13:06 |

---


## dbo.ACM_RegimeDefinitions

**Primary Key:** ID  
**Row Count:** 27  
**Date Range:** 2026-01-04 08:13:56 to 2026-01-13 06:55:19  

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
| RunID | nvarchar | YES | 50 | — |

### Top 10 Records

| ID | EquipID | RegimeVersion | RegimeID | RegimeName | CentroidJSON | FeatureColumns | DataPointCount | SilhouetteScore | MaturityState |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 45 | 5000 | 1 | 0 | Regime_0 | [-0.8002188151331562, -0.8267412153825474, -0.7070072024206104, -0.5336105050401869, -0.727736896... | [] | 0 | NULL | LEARNING |
| 46 | 5000 | 1 | 1 | Regime_1 | [0.23383433594011355, 0.508982023968742, 0.059802965459089565, 0.7214684926862098, 0.329189050735... | [] | 0 | NULL | LEARNING |
| 47 | 5000 | 1 | 2 | Regime_2 | [1.5816446096428562, 1.4769240261764751, 1.5232702051002687, 0.7325254757128681, 1.71412101962411... | [] | 0 | NULL | LEARNING |
| 48 | 5000 | 1 | 3 | Regime_3 | [1.3003296009311964, 1.2217669073327406, 1.1117880267943385, 0.62126524625748, -0.881239984686063... | [] | 0 | NULL | LEARNING |
| 49 | 5000 | 1 | 4 | Regime_4 | [-0.8059369556479116, -0.932843109386338, -0.6641545752765021, -0.7792635448575108, -0.7613633281... | [] | 0 | NULL | LEARNING |
| 50 | 5000 | 1 | 5 | Regime_5 | [-0.5061378329711791, -0.41593011273473834, -0.5226838707734022, -0.07571344262063875, -0.4190003... | [] | 0 | NULL | LEARNING |
| 57 | 5000 | 1 | 0 | Regime_0 | [0.8708339228487767, 0.8088461302959167, 0.7358532915403647, 0.10203207723653733, -0.945015522273... | [] | 222 | NULL | LEARNING |
| 58 | 5000 | 1 | 1 | Regime_1 | [1.0218655542522908, 0.9709802445122624, 1.0185545027452756, 0.15048461301252514, 1.1737639857124... | [] | 871 | NULL | LEARNING |
| 59 | 5000 | 1 | 2 | Regime_2 | [-1.0898489743749673, -1.2283963379458847, -0.9295799399090704, -0.7260148872699693, -1.023931856... | [] | 535 | NULL | LEARNING |
| 60 | 5000 | 1 | 3 | Regime_3 | [-0.9256531492071717, -0.8229820689921709, -0.901430076125421, -0.10843789797248686, -0.577801561... | [] | 380 | NULL | LEARNING |

### Bottom 10 Records

| ID | EquipID | RegimeVersion | RegimeID | RegimeName | CentroidJSON | FeatureColumns | DataPointCount | SilhouetteScore | MaturityState |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 109 | 5010 | 1 | 0 | Regime_0 | [-0.4098015262568227, -0.4323212007847782, -0.36100262337337324, -0.3247227911663611, -0.37859163... | [] | 5507 | NULL | LEARNING |
| 108 | 2621 | 1 | 5 | Regime_5 | [-0.49987971829798644, -0.09704091570434585, 0.105777767663088] | [] | 1168 | NULL | LEARNING |
| 107 | 2621 | 1 | 4 | Regime_4 | [0.6266929092614547, -0.9378862380981445, -0.7028816665212313] | [] | 69 | NULL | LEARNING |
| 106 | 2621 | 1 | 3 | Regime_3 | [1.4665723785440972, -1.1319260559183486, 0.3023604788480604] | [] | 47 | NULL | LEARNING |
| 105 | 2621 | 1 | 2 | Regime_2 | [-0.3635497375650973, -0.031198405309534463, -2.325201292506984] | [] | 61 | NULL | LEARNING |
| 104 | 2621 | 1 | 1 | Regime_1 | [2.065992832183838, 1.0538343687852223, -0.16802042339824969] | [] | 36 | NULL | LEARNING |
| 103 | 2621 | 1 | 0 | Regime_0 | [0.3491467829730551, -0.5305677985198306, 2.446327318579464] | [] | 59 | NULL | LEARNING |
| 102 | 5013 | 1 | 0 | Regime_0 | [-0.1923758939653729, -0.2236474172340617, -0.13588056736397025, -0.22365760659664466, -0.1364152... | [] | 6490 | NULL | LEARNING |
| 81 | 5000 | 1 | 0 | Regime_0 | [-0.05694204906785319, -0.07489770019920762, -0.03739896948013891, -0.09178737522160596, -0.02357... | [] | 7359 | NULL | LEARNING |
| 68 | 5000 | 1 | 3 | Regime_3 | [-1.5130350332304905, -1.6178347580309764, -1.277733282863814, -0.7924504044045969, -1.3584026576... | [] | 213 | NULL | LEARNING |

---


## dbo.ACM_RegimeOccupancy

**Primary Key:** ID  
**Row Count:** 32  
**Date Range:** 2026-01-04 08:13:57 to 2026-01-13 06:55:20  

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

### Top 10 Records

| ID | RunID | EquipID | RegimeLabel | DwellTimeHours | DwellFraction | EntryCount | AvgDwellMinutes | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 62 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 4 | 4875.0 | 0.22165135946167136 | NULL | NULL | 2026-01-04 08:13:57 |
| 63 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 2 | 4366.0 | 0.19850868418659634 | NULL | NULL | 2026-01-04 08:13:57 |
| 64 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 0 | 3819.0 | 0.17363826498135856 | NULL | NULL | 2026-01-04 08:13:57 |
| 65 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 1 | 3599.0 | 0.1636355369646267 | NULL | NULL | 2026-01-04 08:13:57 |
| 66 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 5 | 3021.0 | 0.13735564244794035 | NULL | NULL | 2026-01-04 08:13:57 |
| 67 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | -1 | 1693.0 | 0.07697553878330454 | NULL | NULL | 2026-01-04 08:13:57 |
| 68 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 3 | 621.0 | 0.02823497317450214 | NULL | NULL | 2026-01-04 08:13:57 |
| 76 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | -1 | 550.0 | 0.3184713375796178 | NULL | NULL | 2026-01-04 09:52:07 |
| 77 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 0 | 437.0 | 0.25303995367689636 | NULL | NULL | 2026-01-04 09:52:07 |
| 78 | 37f68854-5fa4-456a-8575-542f067f7e01 | 5000 | 1 | 373.0 | 0.21598147075854082 | NULL | NULL | 2026-01-04 09:52:07 |

### Bottom 10 Records

| ID | RunID | EquipID | RegimeLabel | DwellTimeHours | DwellFraction | EntryCount | AvgDwellMinutes | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 137 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | 0 | 21437.0 | 1.0 | NULL | NULL | 2026-01-13 06:55:20 |
| 136 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 0 | 34.0 | 0.029209621993127148 | NULL | NULL | 2026-01-13 04:29:06 |
| 135 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 1 | 43.0 | 0.036941580756013746 | NULL | NULL | 2026-01-13 04:29:06 |
| 134 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 2 | 86.0 | 0.07388316151202749 | NULL | NULL | 2026-01-13 04:29:06 |
| 133 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 3 | 106.0 | 0.09106529209621993 | NULL | NULL | 2026-01-13 04:29:06 |
| 132 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 4 | 135.0 | 0.11597938144329897 | NULL | NULL | 2026-01-13 04:29:06 |
| 131 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 5 | 760.0 | 0.6529209621993127 | NULL | NULL | 2026-01-13 04:29:06 |
| 130 | 42d92d4c-a6f2-4fe6-9eef-301aae0517fc | 5013 | 0 | 21604.0 | 1.0 | NULL | NULL | 2026-01-13 04:21:39 |
| 108 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | 0 | 200.0 | 0.007279344858962694 | NULL | NULL | 2026-01-12 07:59:36 |
| 107 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | -1 | 27275.0 | 0.9927206551410374 | NULL | NULL | 2026-01-12 07:59:36 |

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


## dbo.ACM_RegimeState

**Primary Key:** EquipID, StateVersion  
**Row Count:** 4  
**Date Range:** 2026-01-12 07:59:34 to 2026-01-13 06:55:18  

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
| 2621 | 1 | 6 | [[0.3491467829730551, -0.5305677985198306, 2.446327318579464], [2.065992832183838, 1.053834368785... | [] | [] | [] | [] | 3 | 0.07234602846940957 |
| 5000 | 1 | 1 | [[-0.05694204906785319, -0.07489770019920762, -0.03739896948013891, -0.09178737522160596, -0.0235... | [] | [] | [] | [] | 0 | 0.3126020382392627 |
| 5010 | 1 | 1 | [[-0.4098015262568227, -0.4323212007847782, -0.36100262337337324, -0.3247227911663611, -0.3785916... | [] | [] | [] | [] | 0 | 0.42195783629161954 |
| 5013 | 1 | 1 | [[-0.1923758939653729, -0.2236474172340617, -0.13588056736397025, -0.22365760659664466, -0.136415... | [] | [] | [] | [] | 0 | 0.3675654672051222 |

---


## dbo.ACM_RegimeTimeline

**Primary Key:** No primary key  
**Row Count:** 74,541  
**Date Range:** 2022-08-12 03:10:00 to 2024-06-16 00:59:00  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| Timestamp | RegimeLabel | RegimeState | RunID | EquipID | AssignmentConfidence | RegimeVersion | ID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-08-12 03:10:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.556 | NULL | 15076 |
| 2022-08-12 03:20:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.93 | NULL | 15077 |
| 2022-08-12 03:30:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.76 | NULL | 15078 |
| 2022-08-12 03:40:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.817 | NULL | 15079 |
| 2022-08-12 03:50:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.955 | NULL | 15080 |
| 2022-08-12 04:00:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.965 | NULL | 15081 |
| 2022-08-12 04:10:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.805 | NULL | 15082 |
| 2022-08-12 04:20:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.904 | NULL | 15083 |
| 2022-08-12 04:30:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 1.0 | NULL | 15084 |
| 2022-08-12 04:40:00 | 1 | unknown | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 0.903 | NULL | 15085 |

### Bottom 10 Records

| Timestamp | RegimeLabel | RegimeState | RunID | EquipID | AssignmentConfidence | RegimeVersion | ID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-06-16 00:59:00 | 3 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.8 | NULL | 325511 |
| 2024-06-15 23:59:00 | 3 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.8 | NULL | 325510 |
| 2024-06-15 22:59:00 | 3 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.8 | NULL | 325509 |
| 2024-06-15 21:59:00 | 1 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.8 | NULL | 325508 |
| 2024-06-15 20:59:00 | 1 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.712 | NULL | 325507 |
| 2024-06-15 19:59:00 | 5 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.256 | NULL | 325506 |
| 2024-06-15 18:59:00 | 5 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.641 | NULL | 325505 |
| 2024-06-15 17:59:00 | 4 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.8 | NULL | 325504 |
| 2024-06-15 16:59:00 | 4 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.8 | NULL | 325503 |
| 2024-06-15 15:59:00 | 5 | unknown | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 0.555 | NULL | 325502 |

---


## dbo.ACM_RegimeTransitions

**Primary Key:** ID  
**Row Count:** 79  
**Date Range:** 2026-01-04 08:14:01 to 2026-01-13 04:29:07  

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

### Top 10 Records

| ID | RunID | EquipID | FromRegime | ToRegime | TransitionCount | TransitionProbability | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 165 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 5 | 1 | 194 | 0.375968992248062 | 2026-01-04 08:14:01 |
| 166 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 5 | 0 | 263 | 0.5096899224806202 | 2026-01-04 08:14:01 |
| 167 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 5 | -1 | 58 | 0.1124031007751938 | 2026-01-04 08:14:01 |
| 168 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 5 | 4 | 1 | 0.001937984496124031 | 2026-01-04 08:14:01 |
| 169 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 1 | 5 | 208 | 0.45714285714285713 | 2026-01-04 08:14:01 |
| 170 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 1 | -1 | 78 | 0.17142857142857143 | 2026-01-04 08:14:01 |
| 171 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 1 | 2 | 151 | 0.33186813186813185 | 2026-01-04 08:14:01 |
| 172 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 1 | 0 | 17 | 0.03736263736263736 | 2026-01-04 08:14:01 |
| 173 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 1 | 4 | 1 | 0.002197802197802198 | 2026-01-04 08:14:01 |
| 174 | 5fb87cc0-cbae-4506-a3bf-8e65b12290ee | 5000 | 0 | 4 | 273 | 0.4674657534246575 | 2026-01-04 08:14:01 |

### Bottom 10 Records

| ID | RunID | EquipID | FromRegime | ToRegime | TransitionCount | TransitionProbability | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 340 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 3 | 5 | 11 | 0.3548387096774194 | 2026-01-13 04:29:07 |
| 339 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 3 | 4 | 2 | 0.06451612903225806 | 2026-01-13 04:29:07 |
| 338 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 3 | 1 | 5 | 0.16129032258064516 | 2026-01-13 04:29:07 |
| 337 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 3 | 2 | 13 | 0.41935483870967744 | 2026-01-13 04:29:07 |
| 336 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 2 | 5 | 15 | 0.7894736842105263 | 2026-01-13 04:29:07 |
| 335 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 2 | 4 | 4 | 0.21052631578947367 | 2026-01-13 04:29:07 |
| 334 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 1 | 3 | 4 | 0.17391304347826086 | 2026-01-13 04:29:07 |
| 333 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 1 | 5 | 9 | 0.391304347826087 | 2026-01-13 04:29:07 |
| 332 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 1 | 4 | 10 | 0.43478260869565216 | 2026-01-13 04:29:07 |
| 331 | aef8fe81-a23e-4529-a281-a8a0c3047a9d | 2621 | 0 | 3 | 3 | 0.6 | 2026-01-13 04:29:07 |

---


## dbo.ACM_Regime_Episodes

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| RegimeLabel | int | NO | 10 | — |
| EpisodeStart | datetime2 | NO | — | — |
| EpisodeEnd | datetime2 | YES | — | — |
| DurationMinutes | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_RunLogs

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | YES | 10 | — |
| LoggedAt | datetime2 | NO | — | (sysutcdatetime()) |
| Level | nvarchar | NO | 16 | — |
| Component | nvarchar | YES | 64 | — |
| Message | nvarchar | NO | -1 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_RunMetadata

**Primary Key:** ID  
**Row Count:** 0  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| BatchNumber | int | YES | 10 | — |
| WindowStart | datetime2 | YES | — | — |
| WindowEnd | datetime2 | YES | — | — |
| RowsIn | int | YES | 10 | — |
| RowsOut | int | YES | 10 | — |
| ConfigSignature | nvarchar | YES | 64 | — |
| PipelineMode | nvarchar | YES | 32 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

---


## dbo.ACM_RunMetrics

**Primary Key:** ID  
**Row Count:** 54  
**Date Range:** 2026-01-04 15:30:26 to 2026-01-04 16:13:50  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| MetricName | nvarchar | NO | 128 | — |
| MetricValue | float | YES | 53 | — |
| MetricUnit | nvarchar | YES | 32 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

### Top 10 Records

| ID | RunID | EquipID | MetricName | MetricValue | MetricUnit | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 19 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.weight.ar1_z | 0.19 | NULL | 2026-01-04 15:30:26 |
| 20 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.weight.gmm_z | 0.085 | NULL | 2026-01-04 15:30:26 |
| 21 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.weight.iforest_z | 0.15500000000000003 | NULL | 2026-01-04 15:30:26 |
| 22 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.weight.omr_z | 0.12000000000000001 | NULL | 2026-01-04 15:30:26 |
| 23 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.weight.pca_spe_z | 0.26000000000000006 | NULL | 2026-01-04 15:30:26 |
| 24 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.weight.pca_t2_z | 0.19 | NULL | 2026-01-04 15:30:26 |
| 25 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.quality.ar1_z | 0.0 | NULL | 2026-01-04 15:30:26 |
| 26 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.quality.gmm_z | 0.0 | NULL | 2026-01-04 15:30:26 |
| 27 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.quality.iforest_z | 0.0 | NULL | 2026-01-04 15:30:26 |
| 28 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | fusion.quality.omr_z | 0.0 | NULL | 2026-01-04 15:30:26 |

### Bottom 10 Records

| ID | RunID | EquipID | MetricName | MetricValue | MetricUnit | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 72 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.n_samples.pca_t2_z | 1134.0 | NULL | 2026-01-04 16:13:50 |
| 71 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.n_samples.pca_spe_z | 1134.0 | NULL | 2026-01-04 16:13:50 |
| 70 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.n_samples.omr_z | 1134.0 | NULL | 2026-01-04 16:13:50 |
| 69 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.n_samples.iforest_z | 1134.0 | NULL | 2026-01-04 16:13:50 |
| 68 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.n_samples.gmm_z | 1134.0 | NULL | 2026-01-04 16:13:50 |
| 67 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.n_samples.ar1_z | 1134.0 | NULL | 2026-01-04 16:13:50 |
| 66 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.quality.pca_t2_z | 0.0 | NULL | 2026-01-04 16:13:50 |
| 65 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.quality.pca_spe_z | 0.0 | NULL | 2026-01-04 16:13:50 |
| 64 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.quality.omr_z | 0.0 | NULL | 2026-01-04 16:13:50 |
| 63 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | fusion.quality.iforest_z | 0.0 | NULL | 2026-01-04 16:13:50 |

---


## dbo.ACM_Run_Stats

**Primary Key:** RecordID  
**Row Count:** 7  
**Date Range:** 2022-04-30 13:20:00 to 2023-10-15 00:00:00  

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
| 8 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-04 06:10:00 | 2022-09-04 06:10:00 | 1727 | 1727 | 81 | 0.0 | NULL |
| 9 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | 2022-08-04 00:00:00 | 2022-08-20 00:00:00 | 1134 | 1134 | 81 | 100.0 | NULL |
| 10 | 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | 2022-08-04 00:00:00 | 2022-08-20 00:00:00 | 1134 | 1134 | 81 | 100.0 | NULL |
| 18 | F7849950-AE3E-42E4-8AB2-B90A562008DA | 5000 | 2022-08-04 00:00:00 | 2023-08-24 00:00:00 | 27475 | 27475 | 81 | 100.0 | NULL |
| 28 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 5013 | 2022-04-30 13:20:00 | 2023-05-25 10:19:59 | 21604 | 21604 | 81 | 0.0 | NULL |
| 29 | AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | 2023-10-15 00:00:00 | 2024-06-16 01:58:59 | 1164 | 1164 | 16 | 0.0 | NULL |
| 30 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2022-10-09 08:40:00 | 2023-10-18 08:39:59 | 21437 | 21437 | 81 | 0.0 | NULL |

---


## dbo.ACM_Runs

**Primary Key:** RunID  
**Row Count:** 15  
**Date Range:** 2026-01-04 07:56:29 to 2026-01-13 07:01:14  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| RunID | EquipID | EquipName | StartedAt | CompletedAt | DurationSeconds | ConfigSignature | TrainRowCount | ScoreRowCount | EpisodeCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC | 5013 | WFA_TURBINE_13 | 2026-01-13 04:09:56 | 2026-01-13 04:26:32 | 995 |  | 21604 | 3151 | 176 |
| F8415FD2-68DD-4542-B08D-4885023C198D | 5010 | NULL | 2026-01-13 07:01:14 | NULL | NULL |  | NULL | NULL | NULL |
| 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | WFA_TURBINE_0 | 2026-01-04 09:46:28 | 2026-01-04 10:09:02 | 1352 |  | 1727 | 0 | 28 |
| F456F1C4-445E-46F9-8894-6CDB8026E174 | 5000 | WFA_TURBINE_0 | 2026-01-04 09:07:09 | 2026-01-04 09:09:56 | 167 |  | 0 | 0 | 0 |
| 66AB154C-8A81-4450-9180-8D859017D3B7 | 5000 | WFA_TURBINE_0 | 2026-01-04 10:36:37 | 2026-01-04 10:47:12 | 634 |  | 1134 | 3161 | 13 |
| 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | NULL | 2026-01-04 07:57:59 | NULL | NULL |  | NULL | NULL | NULL |
| AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | GAS_TURBINE | 2026-01-13 04:27:10 | 2026-01-13 04:31:20 | 249 |  | 1164 | 641 | 15 |
| F7849950-AE3E-42E4-8AB2-B90A562008DA | 5000 | WFA_TURBINE_0 | 2026-01-12 07:48:28 | 2026-01-12 09:54:02 | 7533 |  | 27475 | 3161 | 421 |
| 55DD07D8-DF44-455C-8FFF-B9991F491937 | 5000 | NULL | 2026-01-04 07:56:29 | NULL | NULL |  | NULL | NULL | NULL |
| 60AF54B0-9239-4E85-8BC9-BE6D65998BA1 | 5000 | WFA_TURBINE_0 | 2026-01-04 09:29:31 | 2026-01-04 09:29:50 | 19 |  | 0 | 0 | 0 |

### Bottom 10 Records

| RunID | EquipID | EquipName | StartedAt | CompletedAt | DurationSeconds | ConfigSignature | TrainRowCount | ScoreRowCount | EpisodeCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BC1F72AE-732E-4258-9A57-ECEA3A04CBAE | 5000 | WFA_TURBINE_0 | 2026-01-04 10:14:17 | 2026-01-04 10:14:23 | 5 |  | 0 | 0 | 0 |
| D6995220-B292-4C02-A0D5-D76D80301788 | 5000 | WFA_TURBINE_0 | 2026-01-04 09:37:37 | 2026-01-04 09:39:31 | 113 |  | 0 | 0 | 0 |
| FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | WFA_TURBINE_10 | 2026-01-13 06:44:04 | 2026-01-13 07:00:26 | 980 |  | 21437 | 3161 | 209 |
| CCDD4D7F-3ACC-4CF4-9987-D729FD313C6F | 5000 | WFA_TURBINE_0 | 2026-01-04 09:11:50 | 2026-01-04 09:27:04 | 913 |  | 0 | 0 | 0 |
| C82BE7B6-357E-4541-85A6-D02A4460C7D2 | 5000 | WFA_TURBINE_0 | 2026-01-04 10:21:44 | 2026-01-04 10:31:28 | 583 |  | 1134 | 3160 | 13 |
| 60AF54B0-9239-4E85-8BC9-BE6D65998BA1 | 5000 | WFA_TURBINE_0 | 2026-01-04 09:29:31 | 2026-01-04 09:29:50 | 19 |  | 0 | 0 | 0 |
| 55DD07D8-DF44-455C-8FFF-B9991F491937 | 5000 | NULL | 2026-01-04 07:56:29 | NULL | NULL |  | NULL | NULL | NULL |
| F7849950-AE3E-42E4-8AB2-B90A562008DA | 5000 | WFA_TURBINE_0 | 2026-01-12 07:48:28 | 2026-01-12 09:54:02 | 7533 |  | 27475 | 3161 | 421 |
| AEF8FE81-A23E-4529-A281-A8A0C3047A9D | 2621 | GAS_TURBINE | 2026-01-13 04:27:10 | 2026-01-13 04:31:20 | 249 |  | 1164 | 641 | 15 |
| 5FB87CC0-CBAE-4506-A3BF-8E65B12290EE | 5000 | NULL | 2026-01-04 07:57:59 | NULL | NULL |  | NULL | NULL | NULL |

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


## dbo.ACM_Scores_Wide

**Primary Key:** No primary key  
**Row Count:** 74,541  
**Date Range:** 2022-08-12 03:10:00 to 2024-06-16 00:59:00  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| Timestamp | ar1_z | pca_spe_z | pca_t2_z | mhal_z | iforest_z | gmm_z | cusum_z | drift_z | hst_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-08-12 03:10:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | -0.1973414570093155 | 0.12787827849388123 | -5.555525302886963 | NULL | NULL |
| 2022-08-12 03:20:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | 0.2998310625553131 | 10.0 | -5.554003715515137 | NULL | NULL |
| 2022-08-12 03:30:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | 0.19781142473220825 | 4.744383335113525 | -5.551417350769043 | NULL | NULL |
| 2022-08-12 03:40:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | -0.15877103805541992 | 0.49035167694091797 | -5.5480852127075195 | NULL | NULL |
| 2022-08-12 03:50:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | -0.04016982764005661 | 10.0 | -5.544231414794922 | NULL | NULL |
| 2022-08-12 04:00:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | 0.07655788213014603 | 2.5143163204193115 | -5.540012359619141 | NULL | NULL |
| 2022-08-12 04:10:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | -0.15626879036426544 | 0.1439698189496994 | -5.535537242889404 | NULL | NULL |
| 2022-08-12 04:20:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | -0.5159972906112671 | -0.32509955763816833 | -5.530883312225342 | NULL | NULL |
| 2022-08-12 04:30:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | -0.31598612666130066 | -0.34790828824043274 | -5.52610445022583 | NULL | NULL |
| 2022-08-12 04:40:00 | -0.674490749835968 | 0.0 | -0.674490749835968 | NULL | -0.5115694403648376 | 1.666170597076416 | -5.521237373352051 | NULL | NULL |

### Bottom 10 Records

| Timestamp | ar1_z | pca_spe_z | pca_t2_z | mhal_z | iforest_z | gmm_z | cusum_z | drift_z | hst_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-06-16 00:59:00 | 5.321649551391602 | 10.0 | 10.0 | NULL | 2.7155375480651855 | 3.025763511657715 | 3.6051278114318848 | NULL | NULL |
| 2024-06-15 23:59:00 | 6.200459957122803 | 10.0 | 10.0 | NULL | 3.0169520378112793 | 3.1252658367156982 | 3.5793232917785645 | NULL | NULL |
| 2024-06-15 22:59:00 | 10.0 | 10.0 | 10.0 | NULL | 3.130505084991455 | 3.3944098949432373 | 3.555100202560425 | NULL | NULL |
| 2024-06-15 21:59:00 | 10.0 | 10.0 | 10.0 | NULL | 4.416773319244385 | 4.862733840942383 | 3.533977746963501 | NULL | NULL |
| 2024-06-15 20:59:00 | -0.34741759300231934 | 0.506199300289154 | 3.99658203125 | NULL | 0.9240269064903259 | 0.9177649617195129 | 3.5193567276000977 | NULL | NULL |
| 2024-06-15 19:59:00 | 1.1246637105941772 | 0.5554882287979126 | 4.5014448165893555 | NULL | 0.7940686941146851 | 0.9603965878486633 | 3.5158281326293945 | NULL | NULL |
| 2024-06-15 18:59:00 | 10.0 | 1.7921950817108154 | 6.142233371734619 | NULL | 1.2774474620819092 | 1.7209326028823853 | 3.50825834274292 | NULL | NULL |
| 2024-06-15 17:59:00 | 10.0 | 7.250339031219482 | 10.0 | NULL | 3.185835599899292 | 2.1584742069244385 | 3.4956657886505127 | NULL | NULL |
| 2024-06-15 16:59:00 | 8.441933631896973 | 5.272065162658691 | 7.962900161743164 | NULL | 2.155742883682251 | 1.6308239698410034 | 3.482086420059204 | NULL | NULL |
| 2024-06-15 15:59:00 | 5.223424434661865 | 1.8472689390182495 | 5.595080852508545 | NULL | 2.123090982437134 | 1.5303877592086792 | 3.4739997386932373 | NULL | NULL |

---


## dbo.ACM_SeasonalPatterns

**Primary Key:** ID  
**Row Count:** 511  
**Date Range:** 2026-01-04 15:37:56 to 2026-01-13 12:28:13  

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
| RunID | nvarchar | YES | 50 | — |

### Top 10 Records

| ID | EquipID | SensorName | PatternType | PeriodHours | Amplitude | PhaseShift | Confidence | DetectedAt | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 293 | 5000 | power_29_avg | DAILY | 24.0 | 0.0992 | 10.0 | 0.5812 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 294 | 5000 | power_29_max | DAILY | 24.0 | 0.1131 | 10.0 | 0.5853 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 295 | 5000 | power_29_min | DAILY | 24.0 | 0.0764 | 9.0 | 0.5321 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 296 | 5000 | power_29_std | DAILY | 24.0 | 0.0129 | 14.0 | 0.2002 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 297 | 5000 | power_30_avg | DAILY | 24.0 | 0.0962 | 5.0 | 0.4572 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 298 | 5000 | power_30_max | DAILY | 24.0 | 0.1027 | 6.0 | 0.4464 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 299 | 5000 | power_30_min | DAILY | 24.0 | 0.1082 | 6.0 | 0.4419 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 300 | 5000 | power_30_std | DAILY | 24.0 | 0.015 | 12.0 | 0.1998 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 301 | 5000 | reactive_power_27_avg | DAILY | 24.0 | 0.0625 | 10.0 | 0.2968 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |
| 302 | 5000 | reactive_power_27_max | DAILY | 24.0 | 0.0643 | 7.0 | 0.4183 | 2026-01-04 15:37:56 | 37f68854-5fa4-456a-8575-542f067f7e01 |

### Bottom 10 Records

| ID | EquipID | SensorName | PatternType | PeriodHours | Amplitude | PhaseShift | Confidence | DetectedAt | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1852 | 5010 | wind_speed_4_avg | DAILY | 24.0 | 0.9276 | 7.0 | 0.4296 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1851 | 5010 | wind_speed_3_std | DAILY | 24.0 | 0.2711 | 6.0 | 0.4516 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1850 | 5010 | wind_speed_3_min | DAILY | 24.0 | 0.2215 | 15.0 | 0.4011 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1849 | 5010 | wind_speed_3_max | DAILY | 24.0 | 2.0848 | 7.0 | 0.4382 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1848 | 5010 | wind_speed_3_avg | DAILY | 24.0 | 0.9428 | 7.0 | 0.4391 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1847 | 5010 | sensor_9_avg | WEEKLY | 168.0 | 3.0286 | 57.0 | 0.2683 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1846 | 5010 | sensor_9_avg | DAILY | 24.0 | 2.2552 | 9.0 | 0.6372 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1845 | 5010 | sensor_8_avg | WEEKLY | 168.0 | 14.7969 | 129.0 | 0.1524 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1844 | 5010 | sensor_8_avg | DAILY | 24.0 | 8.2867 | 9.0 | 0.4798 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |
| 1843 | 5010 | sensor_7_avg | WEEKLY | 168.0 | 2.9062 | 81.0 | 0.2207 | 2026-01-13 12:28:13 | fc487ad0-dd33-43e6-9833-d737537c178f |

---


## dbo.ACM_SensorCorrelations

**Primary Key:** ID  
**Row Count:** 9,616  
**Date Range:** 2026-01-12 09:46:22 to 2026-01-13 06:58:03  

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
| 48051 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | power_29_avg | 1.0 | pearson | 2026-01-12 09:46:22 |
| 48052 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | power_29_max | 0.9592361248889012 | pearson | 2026-01-12 09:46:22 |
| 48053 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | power_29_min | 0.9463412076203062 | pearson | 2026-01-12 09:46:22 |
| 48054 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | power_29_std | 0.3467896870596915 | pearson | 2026-01-12 09:46:22 |
| 48055 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | power_30_avg | 0.9505474090977465 | pearson | 2026-01-12 09:46:22 |
| 48056 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | power_30_max | 0.9106741596695218 | pearson | 2026-01-12 09:46:22 |
| 48057 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | power_30_min | 0.9152218234810313 | pearson | 2026-01-12 09:46:22 |
| 48058 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | power_30_std | 0.32569634344498405 | pearson | 2026-01-12 09:46:22 |
| 48059 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | reactive_power_27_avg | 0.24744830651433505 | pearson | 2026-01-12 09:46:22 |
| 48060 | f7849950-ae3e-42e4-8ab2-b90a562008da | 5000 | power_29_avg | reactive_power_27_max | 0.27713762220125676 | pearson | 2026-01-12 09:46:22 |

### Bottom 10 Records

| ID | RunID | EquipID | Sensor1 | Sensor2 | Correlation | CorrelationType | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 83082 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_4_avg | wind_speed_4_avg | 1.0 | pearson | 2026-01-13 06:58:03 |
| 83081 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_std | wind_speed_4_avg | 0.9023026129730616 | pearson | 2026-01-13 06:58:03 |
| 83080 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_std | wind_speed_3_std | 1.0 | pearson | 2026-01-13 06:58:03 |
| 83079 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_min | wind_speed_4_avg | 0.8034291688890138 | pearson | 2026-01-13 06:58:03 |
| 83078 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_min | wind_speed_3_std | 0.5691262868607075 | pearson | 2026-01-13 06:58:03 |
| 83077 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_min | wind_speed_3_min | 1.0 | pearson | 2026-01-13 06:58:03 |
| 83076 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_max | wind_speed_4_avg | 0.9589328309098262 | pearson | 2026-01-13 06:58:03 |
| 83075 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_max | wind_speed_3_std | 0.9397185393612528 | pearson | 2026-01-13 06:58:03 |
| 83074 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_max | wind_speed_3_min | 0.7151721839885149 | pearson | 2026-01-13 06:58:03 |
| 83073 | fc487ad0-dd33-43e6-9833-d737537c178f | 5010 | wind_speed_3_max | wind_speed_3_max | 1.0 | pearson | 2026-01-13 06:58:03 |

---


## dbo.ACM_SensorDefects

**Primary Key:** No primary key  
**Row Count:** 49  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| DetectorType | DetectorFamily | Severity | ViolationCount | ViolationPct | MaxZ | AvgZ | CurrentZ | ActiveDefect | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time-Series Anomaly (AR1) | Time-Series | CRITICAL | 6508 | 30.12 | 10.0 | 3.3435 | 0.5789 | 0 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC |
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 4329 | 20.04 | 2.7797 | 0.557 | 0.0 | 0 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC |
| cusum_z | cusum_z | HIGH | 3347 | 15.49 | 2.9971 | 0.9165 | 0.9211 | 0 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC |
| Baseline Consistency (OMR) | Baseline | MEDIUM | 1525 | 7.06 | 6.1722 | 0.8391 | 0.219 | 0 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC |
| Rare State (IsolationForest) | Rare | MEDIUM | 1393 | 6.45 | 9.2286 | 0.8235 | 0.6047 | 0 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC |
| Density Anomaly (GMM) | Density | LOW | 397 | 1.84 | 9.2325 | 0.728 | 0.268 | 0 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC |
| Correlation Break (PCA-SPE) | Correlation | LOW | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 42D92D4C-A6F2-4FE6-9EEF-301AAE0517FC |
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 647 | 37.46 | 10.0 | 3.5615 | 0.3988 | 0 | 37F68854-5FA4-456A-8575-542F067F7E01 |
| Correlation Break (PCA-SPE) | Correlation | CRITICAL | 434 | 25.13 | 10.0 | 2.0657 | 1.1198 | 0 | 37F68854-5FA4-456A-8575-542F067F7E01 |
| Rare State (IsolationForest) | Rare | CRITICAL | 401 | 23.22 | 6.7378 | 1.348 | 2.0443 | 1 | 37F68854-5FA4-456A-8575-542F067F7E01 |

### Bottom 10 Records

| DetectorType | DetectorFamily | Severity | ViolationCount | ViolationPct | MaxZ | AvgZ | CurrentZ | ActiveDefect | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Density Anomaly (GMM) | Density | HIGH | 2180 | 10.17 | 10.0 | 0.9696 | 0.8554 | 0 | FC487AD0-DD33-43E6-9833-D737537C178F |
| Time-Series Anomaly (AR1) | Time-Series | MEDIUM | 2088 | 9.74 | 10.0 | 1.0432 | 0.3224 | 0 | FC487AD0-DD33-43E6-9833-D737537C178F |
| Baseline Consistency (OMR) | Baseline | MEDIUM | 1947 | 9.08 | 5.7726 | 0.875 | 0.4546 | 0 | FC487AD0-DD33-43E6-9833-D737537C178F |
| Rare State (IsolationForest) | Rare | MEDIUM | 1922 | 8.97 | 9.3878 | 0.9041 | 0.3784 | 0 | FC487AD0-DD33-43E6-9833-D737537C178F |
| cusum_z | cusum_z | MEDIUM | 1407 | 6.56 | 2.2888 | 0.8569 | 2.2372 | 1 | FC487AD0-DD33-43E6-9833-D737537C178F |
| Multivariate Outlier (PCA-T2) | Multivariate | LOW | 520 | 2.43 | 3.4759 | 0.0843 | 0.0 | 0 | FC487AD0-DD33-43E6-9833-D737537C178F |
| Correlation Break (PCA-SPE) | Correlation | LOW | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | FC487AD0-DD33-43E6-9833-D737537C178F |
| Density Anomaly (GMM) | Density | CRITICAL | 621 | 54.76 | 10.0 | 4.2548 | 1.1497 | 0 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 |
| Multivariate Outlier (PCA-T2) | Multivariate | CRITICAL | 593 | 52.29 | 10.0 | 4.9105 | 0.6745 | 0 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 |
| Baseline Consistency (OMR) | Baseline | CRITICAL | 477 | 42.06 | 6.7361 | 2.7949 | 0.4942 | 0 | C82BE7B6-357E-4541-85A6-D02A4460C7D2 |

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
| ID | bigint | NO | 19 | — |

---


## dbo.ACM_SensorHotspots

**Primary Key:** No primary key  
**Row Count:** 166  
**Date Range:** 2022-08-12 14:30:00 to 2024-06-06 00:59:00  

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
| ID | bigint | NO | 19 | — |

### Top 10 Records

| SensorName | MaxTimestamp | LatestTimestamp | MaxAbsZ | MaxSignedZ | LatestAbsZ | LatestSignedZ | ValueAtPeak | LatestValue | TrainMean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sensor_5_avg | 2022-08-12 14:30:00 | 2022-08-20 00:00:00 | 11.3287 | 11.3287 | 2.7943 | 2.7943 | 86.0 | 24.0 | 3.700000047683716 |
| sensor_5_min | 2022-08-12 14:30:00 | 2022-08-20 00:00:00 | 84.5041 | 84.5041 | 21.7764 | 21.7764 | 85.9000015258789 | 20.799999237060547 | -1.7999999523162842 |
| sensor_5_min | 2022-08-12 14:30:00 | 2022-08-20 00:00:00 | 84.5041 | 84.5041 | 21.7764 | 21.7764 | 85.9000015258789 | 20.799999237060547 | -1.7999999523162842 |
| sensor_5_avg | 2022-08-12 14:30:00 | 2022-08-20 00:00:00 | 11.3287 | 11.3287 | 2.7943 | 2.7943 | 86.0 | 24.0 | 3.700000047683716 |
| sensor_52_max | 2022-08-12 19:00:00 | 2022-08-20 00:00:00 | 5.5947 | -5.5947 | 5.1595 | -5.1595 | -1.6739583015441895 | -0.43325228914710395 | 14.276459457208606 |
| sensor_52_avg | 2022-08-12 19:00:00 | 2022-08-20 00:00:00 | 4.7593 | -4.7593 | 4.3849 | -4.3849 | -1.6166658401489258 | -0.41842390899720877 | 13.617412374113387 |
| sensor_52_max | 2022-08-12 19:00:00 | 2022-08-20 00:00:00 | 5.5947 | -5.5947 | 5.1595 | -5.1595 | -1.6739583015441895 | -0.43325228914710395 | 14.276459457208606 |
| sensor_52_avg | 2022-08-12 19:00:00 | 2022-08-20 00:00:00 | 4.7593 | -4.7593 | 4.3849 | -4.3849 | -1.6166658401489258 | -0.41842390899720877 | 13.617412374113387 |
| sensor_5_max | 2022-08-12 21:00:00 | 2022-08-20 00:00:00 | 6.9376 | 6.9376 | 1.165 | 1.165 | 89.9000015258789 | 24.0 | 10.699999809265137 |
| sensor_5_max | 2022-08-12 21:00:00 | 2022-08-20 00:00:00 | 6.9376 | 6.9376 | 1.165 | 1.165 | 89.9000015258789 | 24.0 | 10.699999809265137 |

### Bottom 10 Records

| SensorName | MaxTimestamp | LatestTimestamp | MaxAbsZ | MaxSignedZ | LatestAbsZ | LatestSignedZ | ValueAtPeak | LatestValue | TrainMean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2RADVIBY | 2024-06-06 00:59:00 | 2024-06-16 00:59:00 | 5.4865 | 5.4865 | 0.0007 | -0.0007 | 1.5356162452047926 | 0.16518625252901134 | 0.16535926277200647 |
| LOTEMP1 | 2024-06-06 00:59:00 | 2024-06-16 00:59:00 | 3.9202 | 3.9202 | 0.876 | 0.876 | 132.76067486362047 | 102.40339794711656 | 93.66803757551064 |
| B1RADVIBY | 2024-05-27 03:59:00 | 2024-06-16 00:59:00 | 11.7007 | 11.7007 | 0.0085 | -0.0085 | 5.885454120720506 | 0.2611349407275736 | 0.2652371259419301 |
| B1RADVIBX | 2024-05-27 03:59:00 | 2024-06-16 00:59:00 | 9.9035 | 9.9035 | 0.0763 | 0.0763 | 4.984700847386789 | 0.30332252696239603 | 0.26699665847531495 |
| B1VIB1 | 2024-05-27 03:59:00 | 2024-06-16 00:59:00 | 8.9302 | 8.9302 | 0.0637 | 0.0637 | 0.34612354021531183 | 0.018328837319184346 | 0.01597258222413709 |
| B1VIB2 | 2024-05-27 03:59:00 | 2024-06-16 00:59:00 | 8.896 | 8.896 | 0.0594 | 0.0594 | 0.3506372342453951 | 0.018430614865191507 | 0.016198951194013708 |
| INACTTBTEMP1 | 2024-01-10 01:59:00 | 2024-06-16 00:59:00 | 4.8539 | 4.8539 | 0.6798 | 0.6798 | 163.29366764760672 | 105.51074156591326 | 96.10068167513852 |
| B2RADVIBX | 2024-01-09 22:59:00 | 2024-06-16 00:59:00 | 5.0773 | 5.0773 | 0.0239 | -0.0239 | 1.9008213664422162 | 0.19767206021689973 | 0.2056552328407073 |
| B1TEMP1 | 2024-01-05 23:59:00 | 2024-06-16 00:59:00 | 4.4588 | 4.4588 | 0.4286 | 0.4286 | 208.88181257135756 | 123.34862453721757 | 114.25335532319255 |
| B2TEMP1 | 2023-12-30 23:59:00 | 2024-06-16 00:59:00 | 4.0499 | 4.0499 | 0.4061 | 0.4061 | 202.6374682064297 | 122.33339370471066 | 113.38261899653084 |

---


## dbo.ACM_SensorNormalized_TS

**Primary Key:** ID  
**Row Count:** 69,006  
**Date Range:** 2022-08-12 03:10:00 to 2024-06-16 00:59:00  

### Schema

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ID | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| SensorName | nvarchar | NO | 200 | — |
| RawValue | float | YES | 53 | — |
| NormalizedValue | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (getutcdate()) |

### Top 10 Records

| ID | RunID | EquipID | Timestamp | SensorName | RawValue | NormalizedValue | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 46544 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-22 05:40:00 | power_29_avg | NULL | 0.7627162671782372 | 2026-01-04 10:07:54 |
| 46545 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-22 08:00:00 | power_29_avg | NULL | 0.7504676878452301 | 2026-01-04 10:07:54 |
| 46546 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-22 10:20:00 | power_29_avg | NULL | 0.9663803541891459 | 2026-01-04 10:07:54 |
| 46547 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-22 12:40:00 | power_29_avg | NULL | 0.9118585365116152 | 2026-01-04 10:07:54 |
| 46548 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-22 15:00:00 | power_29_avg | NULL | 0.8726879253130452 | 2026-01-04 10:07:54 |
| 46549 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-22 17:20:00 | power_29_avg | NULL | 0.8824116971829787 | 2026-01-04 10:07:54 |
| 46550 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-22 19:40:00 | power_29_avg | NULL | 0.9146252732279521 | 2026-01-04 10:07:54 |
| 46551 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-22 22:00:00 | power_29_avg | NULL | 0.9151707291603088 | 2026-01-04 10:07:54 |
| 46552 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-23 00:20:00 | power_29_avg | NULL | 0.4412771601674338 | 2026-01-04 10:07:54 |
| 46553 | 37F68854-5FA4-456A-8575-542F067F7E01 | 5000 | 2022-08-23 02:40:00 | power_29_avg | NULL | 1.064515135016118 | 2026-01-04 10:07:54 |

### Bottom 10 Records

| ID | RunID | EquipID | Timestamp | SensorName | RawValue | NormalizedValue | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 273979 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-18 02:50:00 | wind_speed_4_avg | NULL | 8.12280806393838 | 2026-01-13 06:58:13 |
| 273978 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-16 21:50:00 | wind_speed_4_avg | NULL | 11.726690519958344 | 2026-01-13 06:58:13 |
| 273977 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-15 16:50:00 | wind_speed_4_avg | NULL | 12.301591003968927 | 2026-01-13 06:58:13 |
| 273976 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-14 07:10:00 | wind_speed_4_avg | NULL | 5.95953783876583 | 2026-01-13 06:58:13 |
| 273975 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-13 02:10:00 | wind_speed_4_avg | NULL | 7.184685908801648 | 2026-01-13 06:58:13 |
| 273974 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-11 21:10:00 | wind_speed_4_avg | NULL | 7.198408996031072 | 2026-01-13 06:58:13 |
| 273973 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-10 16:10:00 | wind_speed_4_avg | NULL | 6.073309670776519 | 2026-01-13 06:58:13 |
| 273972 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-09 11:10:00 | wind_speed_4_avg | NULL | 2.0771922221639145 | 2026-01-13 06:58:13 |
| 273971 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-08 06:10:00 | wind_speed_4_avg | NULL | 8.700773442397876 | 2026-01-13 06:58:13 |
| 273970 | FC487AD0-DD33-43E6-9833-D737537C178F | 5010 | 2023-10-07 01:10:00 | wind_speed_4_avg | NULL | 2.9267358544902464 | 2026-01-13 06:58:13 |

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
**Row Count:** 96  
**Date Range:** 2025-12-27 06:26:16 to 2026-01-13 06:55:35  

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
| ar1_params | 1 | 1 | 2025-12-31 14:58:55 | {"n_sensors": 72, "mean_autocorr": 12.2818, "mean_residual_std": 0.838, "params_count": 144} | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 5677 bytes> |
| gmm_model | 1 | 1 | 2025-12-31 14:59:00 | {"n_components": 3, "covariance_type": "diag", "bic": 17037772.56, "aic": 17036951.51, "lower_bou... | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 7952 bytes> |
| iforest_model | 1 | 1 | 2025-12-31 14:59:00 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 457321 bytes> |
| pca_model | 1 | 1 | 2025-12-31 14:58:56 | {"n_components": 5, "variance_ratio_sum": 0.8179, "variance_ratio_first_component": 0.281, "varia... | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 4511 bytes> |
| regime_model | 1 | 1 | 2025-12-31 14:59:01 | NULL | {"train_rows": 49, "train_sensors": ["DEMO.SIM.06G31_1FD Fan Damper Position_med", "DEMO.SIM.06GP... | NULL | <binary 19291 bytes> |
| ar1_params | 2621 | 1 | 2026-01-13 04:29:09 | {"n_sensors": 128, "mean_autocorr": 4.6219, "mean_residual_std": 0.3463, "params_count": 256} | {"train_rows": 1746, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEM... | NULL | <binary 6012 bytes> |
| gmm_model | 2621 | 1 | 2026-01-13 04:29:13 | {"n_components": 3, "covariance_type": "diag", "bic": 1313405125.56, "aic": 1313400917.44, "lower... | {"train_rows": 1746, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEM... | NULL | <binary 13537 bytes> |
| iforest_model | 2621 | 1 | 2026-01-13 04:29:13 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 1746, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEM... | NULL | <binary 4049225 bytes> |
| omr_model | 2621 | 1 | 2026-01-13 04:29:13 | NULL | {"train_rows": 1746, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEM... | NULL | <binary 455029 bytes> |
| pca_model | 2621 | 1 | 2026-01-13 04:29:09 | {"n_components": 5, "variance_ratio_sum": 0.863, "variance_ratio_first_component": 0.301, "varian... | {"train_rows": 1746, "train_sensors": ["ACTTBTEMP1_med", "B1RADVIBX_med", "B1RADVIBY_med", "B1TEM... | NULL | <binary 7199 bytes> |

### Bottom 10 Records

| ModelType | EquipID | Version | EntryDateTime | ParamsJSON | StatsJSON | RunID | ModelBytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ar1_params | 5092 | 1 | 2025-12-31 04:24:19 | {"n_sensors": 566, "mean_autocorr": 1459464706535.6265, "mean_residual_std": 2032131359443.5737, ... | {"train_rows": 49, "train_sensors": ["power_29_avg_med", "power_29_min_med", "power_29_std_med", ... | NULL | <binary 30678 bytes> |
| ar1_params | 5092 | 2 | 2025-12-31 04:59:28 | {"n_sensors": 632, "mean_autocorr": -1.4690724257372932e+20, "mean_residual_std": Infinity, "para... | {"train_rows": 27160, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 34392 bytes> |
| ar1_params | 5092 | 3 | 2025-12-31 06:07:00 | {"n_sensors": 632, "mean_autocorr": -2.44824132505186e+19, "mean_residual_std": Infinity, "params... | {"train_rows": 27160, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 34392 bytes> |
| ar1_params | 5092 | 4 | 2025-12-31 07:17:38 | {"n_sensors": 632, "mean_autocorr": -9.494212814703844e+19, "mean_residual_std": Infinity, "param... | {"train_rows": 27165, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 34392 bytes> |
| ar1_params | 5092 | 5 | 2025-12-31 08:34:38 | {"n_sensors": 632, "mean_autocorr": -2.609685739840889e+20, "mean_residual_std": Infinity, "param... | {"train_rows": 27160, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 34392 bytes> |
| ar1_params | 5092 | 6 | 2025-12-31 09:10:19 | {"n_sensors": 630, "mean_autocorr": -2.4608676238385395e+17, "mean_residual_std": Infinity, "para... | {"train_rows": 2710, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med"... | NULL | <binary 34284 bytes> |
| ar1_params | 5092 | 7 | 2025-12-31 09:34:34 | {"n_sensors": 630, "mean_autocorr": 3.2359288601957066e+20, "mean_residual_std": Infinity, "param... | {"train_rows": 2717, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med"... | NULL | <binary 34284 bytes> |
| ar1_params | 5092 | 8 | 2025-12-31 10:00:49 | {"n_sensors": 632, "mean_autocorr": 7222294508196930.0, "mean_residual_std": Infinity, "params_co... | {"train_rows": 2716, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med"... | NULL | <binary 34392 bytes> |
| gmm_model | 5092 | 1 | 2025-12-31 04:24:25 | {"n_components": 3, "covariance_type": "diag", "bic": 3.8181564656701496e+32, "aic": 3.8181564656... | {"train_rows": 49, "train_sensors": ["power_29_avg_med", "power_29_min_med", "power_29_std_med", ... | NULL | <binary 55376 bytes> |
| gmm_model | 5092 | 2 | 2025-12-31 04:59:33 | {"n_components": 3, "covariance_type": "diag", "bic": 2.4543645508588758e+51, "aic": 2.4543645508... | {"train_rows": 27160, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 61959 bytes> |

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

# ACM Comprehensive Database Schema Reference

_Generated automatically on 2026-01-03 09:04:38_

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
| dbo.ACM_ActiveModels | 11 | 1 | ID |
| dbo.ACM_AdaptiveConfig | 13 | 23 | ConfigID |
| dbo.ACM_Anomaly_Events | 7 | 8 | Id |
| dbo.ACM_AssetProfiles | 11 | 1 | ID |
| dbo.ACM_BaselineBuffer | 7 | 0 | Id |
| dbo.ACM_CalibrationSummary | 10 | 6 | ID |
| dbo.ACM_ColdstartState | 18 | 2 | EquipID, Stage |
| dbo.ACM_Config | 7 | 337 | ConfigID |
| dbo.ACM_ConfigHistory | 9 | 1 | ID |
| dbo.ACM_ContributionTimeline | 7 | 0 | ID |
| dbo.ACM_DataContractValidation | 10 | 2 | ID |
| dbo.ACM_DataQuality | 25 | 16 | — |
| dbo.ACM_DetectorCorrelation | 7 | 49 | ID |
| dbo.ACM_DriftController | 10 | 1 | ID |
| dbo.ACM_DriftSeries | 7 | 0 | ID |
| dbo.ACM_EpisodeCulprits | 9 | 40 | ID |
| dbo.ACM_EpisodeDiagnostics | 16 | 8 | ID |
| dbo.ACM_Episodes | 15 | 0 | ID |
| dbo.ACM_FailureForecast | 10 | 1,008 | EquipID, RunID, Timestamp |
| dbo.ACM_FeatureDropLog | 8 | 16 | ID |
| dbo.ACM_ForecastState | 13 | 0 | EquipID, StateVersion |
| dbo.ACM_Forecast_QualityMetrics | 15 | 0 | MetricID |
| dbo.ACM_ForecastingState | 13 | 1 | EquipID, StateVersion |
| dbo.ACM_HealthForecast | 11 | 1,008 | EquipID, RunID, Timestamp |
| dbo.ACM_HealthTimeline | 11 | 2,149 | — |
| dbo.ACM_HistorianData | 6 | 0 | ID |
| dbo.ACM_MultivariateForecast | 10 | 0 | ID |
| dbo.ACM_OMR_Diagnostics | 15 | 1 | DiagnosticID |
| dbo.ACM_PCA_Loadings | 8 | 0 | ID |
| dbo.ACM_PCA_Metrics | 10 | 1 | ID |
| dbo.ACM_PCA_Models | 12 | 0 | ID |
| dbo.ACM_RUL | 33 | 1 | EquipID, RunID |
| dbo.ACM_RefitRequests | 10 | 1 | RequestID |
| dbo.ACM_RegimeDefinitions | 12 | 1 | ID |
| dbo.ACM_RegimeOccupancy | 9 | 2 | ID |
| dbo.ACM_RegimePromotionLog | 9 | 0 | ID |
| dbo.ACM_RegimeState | 15 | 1 | EquipID, StateVersion |
| dbo.ACM_RegimeTimeline | 8 | 2,149 | — |
| dbo.ACM_RegimeTransitions | 8 | 2 | ID |
| dbo.ACM_Regime_Episodes | 8 | 0 | ID |
| dbo.ACM_RunLogs | 8 | 0 | ID |
| dbo.ACM_RunMetadata | 11 | 0 | ID |
| dbo.ACM_RunMetrics | 7 | 0 | ID |
| dbo.ACM_Run_Stats | 13 | 1 | RecordID |
| dbo.ACM_Runs | 20 | 3 | RunID |
| dbo.ACM_SchemaVersion | 5 | 2 | VersionID |
| dbo.ACM_Scores_Wide | 16 | 2,149 | — |
| dbo.ACM_SeasonalPatterns | 10 | 121 | ID |
| dbo.ACM_SensorCorrelations | 8 | 3,160 | ID |
| dbo.ACM_SensorDefects | 12 | 7 | — |
| dbo.ACM_SensorForecast | 12 | 1,680 | RunID, EquipID, Timestamp, SensorName |
| dbo.ACM_SensorHotspots | 19 | 25 | — |
| dbo.ACM_SensorNormalized_TS | 8 | 10,287 | ID |
| dbo.ACM_TagEquipmentMap | 10 | 1,986 | TagID |
| dbo.ELECTRIC_MOTOR_Data | 14 | 17,477 | — |
| dbo.ELECTRIC_MOTOR_Data_RAW | 14 | 1,048,575 | — |
| dbo.Equipment | 8 | 29 | EquipID |
| dbo.FD_FAN_Data | 11 | 17,499 | EntryDateTime |
| dbo.GAS_TURBINE_Data | 18 | 2,911 | EntryDateTime |
| dbo.ModelRegistry | 8 | 70 | ModelType, EquipID, Version |
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
**Row Count:** 1  

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
| 5 | 5010 | 1 | LEARNING | NULL | 1 | NULL | 1 | NULL | 2026-01-02 22:41:18 |

---


## dbo.ACM_AdaptiveConfig

**Primary Key:** ConfigID  
**Row Count:** 23  
**Date Range:** 2025-12-04 10:46:47 to 2026-01-02 22:42:21  

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
| 95 | 5010 | fused_warn_z | 1.1534738540649414 | 0.0 | 999999.0 | True | 2149 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 94 | 5010 | fused_alert_z | 2.306947708129883 | 0.0 | 999999.0 | True | 2149 | 0.0 | quantile_0.997: Auto-calculated from 2149 accumulated samples |
| 85 | 5040 | fused_warn_z | 1.5 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 84 | 5040 | fused_alert_z | 3.0 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated from 49 accumulated samples |
| 77 | 1 | fused_warn_z | 1.5 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 76 | 1 | fused_alert_z | 3.0 | 0.0 | 999999.0 | True | 49 | 0.0 | quantile_0.997: Auto-calculated from 49 accumulated samples |
| 55 | 5092 | fused_warn_z | 0.5271811485290527 | 0.0 | 999999.0 | True | 2717 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 54 | 5092 | fused_alert_z | 1.0543622970581055 | 0.0 | 999999.0 | True | 2717 | 0.0 | quantile_0.997: Auto-calculated from 2717 accumulated samples |
| 21 | 5013 | fused_warn_z | 0.5874592661857605 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated warning threshold (50% of alert) |
| 20 | 5013 | fused_alert_z | 1.174918532371521 | 0.0 | 999999.0 | True | 129 | 0.0 | quantile_0.997: Auto-calculated from 129 accumulated samples |

---


## dbo.ACM_Anomaly_Events

**Primary Key:** Id  
**Row Count:** 8  
**Date Range:** 2022-10-31 20:20:00 to 2022-11-12 05:10:00  

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
| 61 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-10-31 20:20:00 | 2022-10-31 20:50:00 | info | 0.512 |
| 62 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-02 20:20:00 | 2022-11-02 20:40:00 | info | 0.479 |
| 63 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-03 03:40:00 | 2022-11-03 05:00:00 | info | 0.612 |
| 64 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-04 14:20:00 | 2022-11-04 16:00:00 | info | 0.612 |
| 65 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-08 03:00:00 | 2022-11-08 05:30:00 | info | 0.612 |
| 66 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-08 11:30:00 | 2022-11-08 12:50:00 | info | 0.612 |
| 67 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-11 19:10:00 | 2022-11-11 20:10:00 | info | 0.612 |
| 68 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-12 05:10:00 | 2022-11-12 07:50:00 | info | 0.612 |

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
**Row Count:** 0  

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

---


## dbo.ACM_CalibrationSummary

**Primary Key:** ID  
**Row Count:** 6  
**Date Range:** 2026-01-02 17:12:15 to 2026-01-02 17:12:15  

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
| 25 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | ar1_z | 20.0 | NULL | NULL | NULL | NULL | 2026-01-02 17:12:15 |
| 26 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | pca_spe_z | -20.0 | NULL | NULL | NULL | NULL | 2026-01-02 17:12:15 |
| 27 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | pca_t2_z | 0.002378418782594019 | NULL | NULL | NULL | NULL | 2026-01-02 17:12:15 |
| 28 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | iforest_z | 4.491896388011225 | NULL | NULL | NULL | NULL | 2026-01-02 17:12:15 |
| 29 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | gmm_z | -0.24986814661535747 | NULL | NULL | NULL | NULL | 2026-01-02 17:12:15 |
| 30 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | omr_z | 5.518891424282705 | NULL | NULL | NULL | NULL | 2026-01-02 17:12:15 |

---


## dbo.ACM_ColdstartState

**Primary Key:** EquipID, Stage  
**Row Count:** 2  
**Date Range:** 2026-01-02 17:04:31 to 2026-01-02 17:30:31  

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
| 2621 | score | COMPLETE | 1 | 2026-01-02 17:30:31 | 2026-01-02 17:30:33 | 2026-01-02 17:30:33 | 2188 | 500 | 2023-10-15 00:00:00 |
| 5010 | score | COMPLETE | 1 | 2026-01-02 17:04:31 | 2026-01-02 17:04:40 | 2026-01-02 17:04:40 | 5372 | 500 | 2022-10-09 08:40:00 |

---


## dbo.ACM_Config

**Primary Key:** ConfigID  
**Row Count:** 337  
**Date Range:** 2025-12-09 12:47:06 to 2026-01-02 17:30:25  

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
| 492 | 0 | data.train_csv | data/FD_FAN_BASELINE_DATA.csv | string | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 493 | 0 | data.score_csv | data/FD_FAN_BATCH_DATA.csv | string | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 494 | 0 | data.data_dir | data | string | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 495 | 0 | data.timestamp_col | EntryDateTime | string | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 496 | 0 | data.tag_columns | [] | list | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 497 | 0 | data.sampling_secs | 60 | int | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 498 | 0 | data.max_rows | 100000 | int | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 499 | 0 | features.window | 16 | int | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 500 | 0 | features.fft_bands | [0.0, 0.1, 0.3, 0.5] | list | 2026-01-01 06:10:59 | B19cl3pc\bhadk |
| 501 | 0 | features.top_k_tags | 5 | int | 2026-01-01 06:10:59 | B19cl3pc\bhadk |

### Bottom 10 Records

| ConfigID | EquipID | ParamPath | ParamValue | ValueType | UpdatedAt | UpdatedBy |
| --- | --- | --- | --- | --- | --- | --- |
| 994 | 2621 | runtime.tick_minutes | 176460 | int | 2026-01-02 17:30:25 | sql_batch_runner |
| 992 | 5010 | runtime.tick_minutes | 53856 | int | 2026-01-02 17:04:23 | sql_batch_runner |
| 978 | 1 | runtime.storage_backend | sql | string | 2026-01-01 06:11:00 | B19cl3pc\bhadk |
| 977 | 1 | episodes.cpd.k_sigma | 4.0 | float | 2026-01-01 06:11:00 | B19cl3pc\bhadk |
| 976 | 1 | thresholds.self_tune.clip_z | 100.0 | float | 2026-01-01 06:11:00 | B19cl3pc\bhadk |
| 975 | 1 | models.mahl.regularization | 1.0 | float | 2026-01-01 06:11:00 | B19cl3pc\bhadk |
| 974 | 1 | data.min_train_samples | 100 | int | 2026-01-01 06:11:00 | B19cl3pc\bhadk |
| 973 | 0 | forecasting.monte_carlo_simulations | 200 | int | 2026-01-01 06:11:00 | B19cl3pc\bhadk |
| 972 | 0 | episodes.cpd.auto_tune.h_max | 8.0 | float | 2026-01-01 10:29:48 | fix_cusum |
| 971 | 0 | episodes.cpd.auto_tune.k_max | 2.0 | float | 2026-01-01 10:29:48 | fix_cusum |

---


## dbo.ACM_ConfigHistory

**Primary Key:** ID  
**Row Count:** 1  
**Date Range:** 2026-01-02 22:42:25 to 2026-01-02 22:42:25  

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
| 9 | 2026-01-02 22:42:25 | 5010 | k_max | 6.0 | 8.0 | AUTO_TUNE | Auto-tuning based on quality assessment | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |

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
**Row Count:** 2  
**Date Range:** 2026-01-02 22:34:41 to 2026-01-02 23:00:34  

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
| 15 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | True | 2149 | 81 | NULL | NULL | 40548782ca90 | 2026-01-02 22:34:41 |
| 18 | 5c87cbc0-470c-4973-be78-793100f4ec2a | 2621 | True | 876 | 16 | NULL | NULL | 1598146da8ea | 2026-01-02 23:00:34 |

---


## dbo.ACM_DataQuality

**Primary Key:** No primary key  
**Row Count:** 16  
**Date Range:** 2023-10-15 00:00:00 to 2023-10-15 00:00:00  

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
| B2VIB2 | 1312 | 0 | 0.0 | 0.03943781851748799 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B2VIB1 | 1312 | 0 | 0.0 | 0.04149460641978327 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B2TEMP1 | 1312 | 0 | 0.0 | 41.68804530598887 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B2RADVIBY | 1312 | 0 | 0.0 | 0.5155898011779884 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B2RADVIBX | 1312 | 0 | 0.0 | 0.6694469243007635 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1VIB2 | 1312 | 0 | 0.0 | 0.0665001850709071 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1VIB1 | 1312 | 0 | 0.0 | 0.065961799234887 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1TEMP1 | 1312 | 0 | 0.0 | 40.20665572529121 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1RADVIBY | 1312 | 0 | 0.0 | 0.836202738818104 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1RADVIBX | 1312 | 0 | 0.0 | 0.8261767310709012 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |

### Bottom 10 Records

| sensor | train_count | train_nulls | train_null_pct | train_std | train_longest_gap | train_flatline_span | train_min_ts | train_max_ts | score_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2VIB2 | 1312 | 0 | 0.0 | 0.03943781851748799 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B2VIB1 | 1312 | 0 | 0.0 | 0.04149460641978327 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B2TEMP1 | 1312 | 0 | 0.0 | 41.68804530598887 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B2RADVIBY | 1312 | 0 | 0.0 | 0.5155898011779884 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B2RADVIBX | 1312 | 0 | 0.0 | 0.6694469243007635 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1VIB2 | 1312 | 0 | 0.0 | 0.0665001850709071 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1VIB1 | 1312 | 0 | 0.0 | 0.065961799234887 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1TEMP1 | 1312 | 0 | 0.0 | 40.20665572529121 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1RADVIBY | 1312 | 0 | 0.0 | 0.836202738818104 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |
| B1RADVIBX | 1312 | 0 | 0.0 | 0.8261767310709012 | 0 | 0 | 2023-10-15 00:00:00 | 2023-12-08 14:59:00 | 876 |

---


## dbo.ACM_DetectorCorrelation

**Primary Key:** ID  
**Row Count:** 49  
**Date Range:** 2026-01-02 17:12:49 to 2026-01-02 17:12:49  

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
| 197 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | ar1_z | ar1_z | 1.0 | 2026-01-02 17:12:49 |
| 198 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | ar1_z | pca_spe_z | 0.0 | 2026-01-02 17:12:49 |
| 199 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | ar1_z | pca_t2_z | -0.06674452342483617 | 2026-01-02 17:12:49 |
| 200 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | ar1_z | iforest_z | 0.5557699806755195 | 2026-01-02 17:12:49 |
| 201 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | ar1_z | gmm_z | 0.5056714813428221 | 2026-01-02 17:12:49 |
| 202 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | ar1_z | omr_z | 0.5590696112906886 | 2026-01-02 17:12:49 |
| 203 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | ar1_z | cusum_z | -0.06974113971203451 | 2026-01-02 17:12:49 |
| 204 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | pca_spe_z | ar1_z | 0.0 | 2026-01-02 17:12:49 |
| 205 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | pca_spe_z | pca_spe_z | 0.0 | 2026-01-02 17:12:49 |
| 206 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | pca_spe_z | pca_t2_z | 0.0 | 2026-01-02 17:12:49 |

### Bottom 10 Records

| ID | RunID | EquipID | Detector1 | Detector2 | Correlation | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- |
| 245 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | cusum_z | cusum_z | 1.0 | 2026-01-02 17:12:49 |
| 244 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | cusum_z | omr_z | -0.09999348786308415 | 2026-01-02 17:12:49 |
| 243 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | cusum_z | gmm_z | -0.0963812281313332 | 2026-01-02 17:12:49 |
| 242 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | cusum_z | iforest_z | -0.02231253274602093 | 2026-01-02 17:12:49 |
| 241 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | cusum_z | pca_t2_z | 0.000475988808099413 | 2026-01-02 17:12:49 |
| 240 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | cusum_z | pca_spe_z | 0.0 | 2026-01-02 17:12:49 |
| 239 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | cusum_z | ar1_z | -0.06974113971203451 | 2026-01-02 17:12:49 |
| 238 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | omr_z | cusum_z | -0.09999348786308415 | 2026-01-02 17:12:49 |
| 237 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | omr_z | omr_z | 1.0 | 2026-01-02 17:12:49 |
| 236 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | omr_z | gmm_z | 0.6782845188163705 | 2026-01-02 17:12:49 |

---


## dbo.ACM_DriftController

**Primary Key:** ID  
**Row Count:** 1  

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
| 5 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | STABLE | 3.0 | 1.0 | NULL | NULL | NULL | 2026-01-02 17:12:27 |

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
**Row Count:** 40  
**Date Range:** 2026-01-02 17:15:16 to 2026-01-02 17:15:16  

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
| 414 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 1 | Density Anomaly (GMM) | NULL | 36.43388366699219 | 1 | 2026-01-02 17:15:16 | 5010 |
| 415 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 1 | Baseline Consistency (OMR) | NULL | 29.768653869628906 | 2 | 2026-01-02 17:15:16 | 5010 |
| 416 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 1 | Time-Series Anomaly (AR1) | NULL | 13.317184448242188 | 3 | 2026-01-02 17:15:16 | 5010 |
| 417 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 1 | cusum_z | NULL | 12.628277778625488 | 4 | 2026-01-02 17:15:16 | 5010 |
| 418 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 1 | Rare State (IsolationForest) | NULL | 7.851996898651123 | 5 | 2026-01-02 17:15:16 | 5010 |
| 419 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2 | Baseline Consistency (OMR) | NULL | 30.561452865600586 | 1 | 2026-01-02 17:15:16 | 5010 |
| 420 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2 | Rare State (IsolationForest) | NULL | 27.455705642700195 | 2 | 2026-01-02 17:15:16 | 5010 |
| 421 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2 | Density Anomaly (GMM) | NULL | 19.390357971191406 | 3 | 2026-01-02 17:15:16 | 5010 |
| 422 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2 | Time-Series Anomaly (AR1) | NULL | 17.016849517822266 | 4 | 2026-01-02 17:15:16 | 5010 |
| 423 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2 | cusum_z | NULL | 5.575625419616699 | 5 | 2026-01-02 17:15:16 | 5010 |

### Bottom 10 Records

| ID | RunID | EpisodeID | DetectorType | SensorName | ContributionPct | Rank | CreatedAt | EquipID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 453 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 8 | cusum_z | NULL | 3.189293622970581 | 5 | 2026-01-02 17:15:16 | 5010 |
| 452 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 8 | Time-Series Anomaly (AR1) | NULL | 6.5879621505737305 | 4 | 2026-01-02 17:15:16 | 5010 |
| 451 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 8 | Density Anomaly (GMM) | NULL | 21.021928787231445 | 3 | 2026-01-02 17:15:16 | 5010 |
| 450 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 8 | Rare State (IsolationForest) | NULL | 31.593915939331055 | 2 | 2026-01-02 17:15:16 | 5010 |
| 449 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 8 | Baseline Consistency (OMR) | NULL | 37.60689926147461 | 1 | 2026-01-02 17:15:16 | 5010 |
| 448 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 7 | cusum_z | NULL | 6.873376846313477 | 5 | 2026-01-02 17:15:16 | 5010 |
| 447 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 7 | Time-Series Anomaly (AR1) | NULL | 11.233541488647461 | 4 | 2026-01-02 17:15:16 | 5010 |
| 446 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 7 | Density Anomaly (GMM) | NULL | 21.752981185913086 | 3 | 2026-01-02 17:15:16 | 5010 |
| 445 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 7 | Baseline Consistency (OMR) | NULL | 22.593164443969727 | 2 | 2026-01-02 17:15:16 | 5010 |
| 444 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 7 | Rare State (IsolationForest) | NULL | 37.54693603515625 | 1 | 2026-01-02 17:15:16 | 5010 |

---


## dbo.ACM_EpisodeDiagnostics

**Primary Key:** ID  
**Row Count:** 8  
**Date Range:** 2022-10-31 20:20:00 to 2022-11-12 05:10:00  

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
| 61 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 1 | 2022-10-31 20:20:00 | 2022-10-31 20:50:00 | 0.5 | 4.1074810243408 | 2.3318023670028802 | HIGH |
| 62 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 2 | 2022-11-02 20:20:00 | 2022-11-02 20:40:00 | 0.3333333333333333 | 3.137569420572317 | 1.859411131853853 | MEDIUM |
| 63 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 3 | 2022-11-03 03:40:00 | 2022-11-03 05:00:00 | 1.3333333333333333 | 2.2107284288773648 | 1.4885865348332024 | MEDIUM |
| 64 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 4 | 2022-11-04 14:20:00 | 2022-11-04 16:00:00 | 1.6666666666666667 | 2.9695369248411425 | 1.467365661445401 | MEDIUM |
| 65 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 5 | 2022-11-08 03:00:00 | 2022-11-08 05:30:00 | 2.5 | 1.790912652801862 | 0.9725816510388272 | LOW |
| 66 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 6 | 2022-11-08 11:30:00 | 2022-11-08 12:50:00 | 1.3333333333333333 | 1.8598739703640184 | 1.1928410809905272 | LOW |
| 67 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 7 | 2022-11-11 19:10:00 | 2022-11-11 20:10:00 | 1.0 | 1.2316154378793 | 0.8126324993672412 | LOW |
| 68 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 8 | 2022-11-12 05:10:00 | 2022-11-12 07:50:00 | 2.6666666666666665 | 2.358004855496864 | 0.9195119895152263 | MEDIUM |

---


## dbo.ACM_Episodes

**Primary Key:** ID  
**Row Count:** 0  

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

---


## dbo.ACM_FailureForecast

**Primary Key:** EquipID, RunID, Timestamp  
**Row Count:** 1,008  
**Date Range:** 2022-11-15 18:20:00 to 2022-11-22 18:10:00  

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
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 18:20:00 | 4.814197894470332e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4033 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 18:30:00 | 4.611386337715224e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4034 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 18:40:00 | 4.417084207540395e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4035 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 18:50:00 | 4.230935934635067e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4036 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:00:00 | 4.0526007455957746e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4037 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:10:00 | 3.881752049825318e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4038 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:20:00 | 3.7180768517382235e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4039 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:30:00 | 3.5612751872145874e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4040 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:40:00 | 3.411059583322565e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4041 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:50:00 | 3.267154540337075e-53 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 4042 |

### Bottom 10 Records

| EquipID | RunID | Timestamp | FailureProb | SurvivalProb | HazardRate | ThresholdUsed | Method | CreatedAt | ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 18:10:00 | 1.3691560079335314e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5040 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 18:00:00 | 1.4406804829599143e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5039 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:50:00 | 1.5159295033054993e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5038 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:40:00 | 1.595096397562465e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5037 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:30:00 | 1.6783844968635703e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5036 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:20:00 | 1.766007650706049e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5035 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:10:00 | 1.858190769292398e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5034 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:00:00 | 1.9551703937374786e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5033 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 16:50:00 | 2.0571952955802414e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5032 |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 16:40:00 | 2.164527107087978e-73 | 1.0 | 0.0 | 50.0 | GaussianTail | 2026-01-02 22:44:37 | 5031 |

---


## dbo.ACM_FeatureDropLog

**Primary Key:** ID  
**Row Count:** 16  
**Date Range:** 2026-01-02 17:07:32 to 2026-01-02 17:07:32  

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
| 339 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_mean | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 340 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_kurt | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 341 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_std | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 342 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_mean | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 343 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_mad | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 344 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_slope | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 345 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_slope | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 346 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_skew | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 347 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_mad | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 348 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_rz | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |

### Bottom 10 Records

| ID | RunID | EquipID | FeatureName | DropReason | DropValue | Threshold | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 354 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_med | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 353 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_std | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 352 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_med | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 351 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_rz | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 350 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_skew | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 349 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_kurt | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 348 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_rz | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 347 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_49_mad | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 346 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_skew | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |
| 345 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | sensor_46_slope | low_variance | 0.0 | NULL | 2026-01-02 17:07:32 |

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
**Row Count:** 1  

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
| 3 | 5010 | 1 | {"alpha": 0.05, "beta": 0.01, "level": 93.31947322015088, "trend": 0.004278296459112131, "std_err... | {"forecast_mean": 95.47787378377295, "forecast_std": 1.2449174960019613, "forecast_range": 4.3082... | NULL |  | 2149 | 1.2449174960019613 | NULL |

---


## dbo.ACM_HealthForecast

**Primary Key:** EquipID, RunID, Timestamp  
**Row Count:** 1,008  
**Date Range:** 2022-11-15 18:20:00 to 2022-11-22 18:10:00  

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
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 18:20:00 | 93.32375151660999 | 92.03950010980009 | 94.60800292341989 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 18:30:00 | 93.32802981306911 | 92.04214186791519 | 94.61391775822302 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 18:40:00 | 93.33230810952821 | 92.04475320151413 | 94.6198630175423 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 18:50:00 | 93.33658640598733 | 92.0473339087933 | 94.62583890318136 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:00:00 | 93.34086470244644 | 92.04988379118065 | 94.63184561371223 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:10:00 | 93.34514299890556 | 92.05240265336171 | 94.6378833444494 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:20:00 | 93.34942129536466 | 92.05489030330445 | 94.64395228742488 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:30:00 | 93.35369959182378 | 92.05734655228315 | 94.65005263136442 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:40:00 | 93.35797788828289 | 92.05977121490105 | 94.65618456166473 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-15 19:50:00 | 93.36225618474201 | 92.06216410911199 | 94.66234826037203 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |

### Bottom 10 Records

| EquipID | RunID | Timestamp | ForecastHealth | CiLower | CiUpper | ForecastStd | Method | CreatedAt | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 18:10:00 | 97.63199605093591 | 83.91295011728043 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 18:00:00 | 97.6277177544768 | 83.92709960939322 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:50:00 | 97.62343945801769 | 83.94124057548458 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:40:00 | 97.61916116155858 | 83.95537301125091 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:30:00 | 97.61488286509946 | 83.96949691238135 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:20:00 | 97.61060456864035 | 83.9836122745578 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:10:00 | 97.60632627218124 | 83.99771909345488 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 17:00:00 | 97.60204797572213 | 84.01181736473993 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 16:50:00 | 97.59776967926301 | 84.02590708407293 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 2022-11-22 16:40:00 | 97.5934913828039 | 84.03998824710654 | 100.0 | 0.6421257034049503 | ExponentialSmoothing | 2026-01-02 22:44:34 | NULL |

---


## dbo.ACM_HealthTimeline

**Primary Key:** No primary key  
**Row Count:** 2,149  
**Date Range:** 2022-10-31 20:10:00 to 2022-11-15 18:10:00  

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
| 2022-10-31 20:10:00 | 92.92 | GOOD | 0.35409998893737793 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 92.91999816894531 | NORMAL | 0.318 | NULL |
| 2022-10-31 20:20:00 | 68.85 | ALERT | 4.107500076293945 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 12.6899995803833 | NORMAL | 0.506 | NULL |
| 2022-10-31 20:30:00 | 59.14 | ALERT | 2.962100028991699 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 36.47999954223633 | NORMAL | 0.449 | NULL |
| 2022-10-31 20:40:00 | 65.67 | ALERT | 1.2965999841690063 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 80.91000366210938 | NORMAL | 0.366 | NULL |
| 2022-10-31 20:50:00 | 71.88 | WATCH | 0.9610000252723694 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 86.37000274658203 | NORMAL | 0.349 | NULL |
| 2022-10-31 21:00:00 | 78.2 | WATCH | 0.35120001435279846 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 92.94999694824219 | NORMAL | 0.319 | NULL |
| 2022-10-31 21:10:00 | 83.14 | WATCH | -0.10199999809265137 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 94.66999816894531 | NORMAL | 0.306 | NULL |
| 2022-10-31 21:20:00 | 86.05 | GOOD | -0.36559998989105225 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 92.83000183105469 | NORMAL | 0.32 | NULL |
| 2022-10-31 21:30:00 | 88.71 | GOOD | 0.05900000035762787 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 94.93000030517578 | NORMAL | 0.305 | NULL |
| 2022-10-31 21:40:00 | 90.54 | GOOD | 0.07760000228881836 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 94.81999969482422 | NORMAL | 0.306 | NULL |

### Bottom 10 Records

| Timestamp | HealthIndex | HealthZone | FusedZ | RunID | EquipID | RawHealthIndex | QualityFlag | Confidence | ConfidenceFactors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-11-15 18:10:00 | 94.31 | GOOD | 0.03400000184774399 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 95.06999969482422 | NORMAL | 0.502 | NULL |
| 2022-11-15 18:00:00 | 93.99 | GOOD | -0.020800000056624413 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 95.13999938964844 | NORMAL | 0.501 | NULL |
| 2022-11-15 17:50:00 | 93.49 | GOOD | 0.1378999948501587 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 94.44999694824219 | NORMAL | 0.507 | NULL |
| 2022-11-15 17:40:00 | 93.08 | GOOD | 0.42309999465942383 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 92.36000061035156 | NORMAL | 0.521 | NULL |
| 2022-11-15 17:30:00 | 93.39 | GOOD | -0.34380000829696655 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 93.0 | NORMAL | 0.517 | NULL |
| 2022-11-15 17:20:00 | 93.56 | GOOD | -0.2515000104904175 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 93.69000244140625 | NORMAL | 0.513 | NULL |
| 2022-11-15 17:10:00 | 93.5 | GOOD | 0.3682999908924103 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 92.80999755859375 | NORMAL | 0.518 | NULL |
| 2022-11-15 17:00:00 | 93.8 | GOOD | -0.19750000536441803 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 94.05999755859375 | NORMAL | 0.51 | NULL |
| 2022-11-15 16:50:00 | 93.68 | GOOD | -0.20010000467300415 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 94.05000305175781 | NORMAL | 0.51 | NULL |
| 2022-11-15 16:40:00 | 93.53 | GOOD | 0.10939999669790268 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 94.62999725341797 | NORMAL | 0.505 | NULL |

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
**Row Count:** 1  

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
| 4 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | pls | 5 | 3223 | 632 | 3.9457956972097152 | NULL | NULL |

---


## dbo.ACM_PCA_Loadings

**Primary Key:** ID  
**Row Count:** 0  

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

---


## dbo.ACM_PCA_Metrics

**Primary Key:** ID  
**Row Count:** 1  
**Date Range:** 2026-01-02 17:07:56 to 2026-01-02 17:07:56  

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
| 1 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 5 | NULL | [{"name": "PCA", "type": "n_components", "value": 5.0}, {"name": "PCA", "type": "variance_explain... | pca_fit | NULL | NULL | 2026-01-02 17:07:56 |

---


## dbo.ACM_PCA_Models

**Primary Key:** ID  
**Row Count:** 0  

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

---


## dbo.ACM_RUL

**Primary Key:** EquipID, RunID  
**Row Count:** 1  
**Date Range:** 2026-01-09 22:44:40 to 2026-01-09 22:44:40  

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
| 5010 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 168.0 | 163.00985373523335 | 168.0 | 172.99014626476665 | 0.3 | 2026-01-09 22:44:40 | Multipath | 1000 |

---


## dbo.ACM_RefitRequests

**Primary Key:** RequestID  
**Row Count:** 1  
**Date Range:** 2026-01-02 17:12:25 to 2026-01-02 17:12:25  

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
| 5 | 5010 | 2026-01-02 17:12:25 | Silhouette score too low | NULL | NULL | NULL | 0.0 | False | NULL |

---


## dbo.ACM_RegimeDefinitions

**Primary Key:** ID  
**Row Count:** 1  
**Date Range:** 2026-01-02 17:11:05 to 2026-01-02 17:11:05  

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
| 10 | 5010 | 1 | 0 | Regime_0 | [-0.2760356093580652, -0.28124455666729115, -0.45333225981604824, -0.4580290178115268, -0.2730347... | [] | 1548 | NULL | LEARNING |

---


## dbo.ACM_RegimeOccupancy

**Primary Key:** ID  
**Row Count:** 2  
**Date Range:** 2026-01-02 17:11:06 to 2026-01-02 17:11:06  

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
| 14 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | -1 | 1691.0 | 0.78687761749651 | NULL | NULL | 2026-01-02 17:11:06 |
| 15 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 0 | 458.0 | 0.21312238250349 | NULL | NULL | 2026-01-02 17:11:06 |

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
**Row Count:** 1  
**Date Range:** 2026-01-02 17:11:04 to 2026-01-02 17:11:04  

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
| 5010 | 1 | 1 | [[-0.2760356093580652, -0.28124455666729115, -0.45333225981604824, -0.4580290178115268, -0.273034... | [] | [] | [] | [] | 0 | 0.5657388277818803 |

---


## dbo.ACM_RegimeTimeline

**Primary Key:** No primary key  
**Row Count:** 2,149  
**Date Range:** 2022-10-31 20:10:00 to 2022-11-15 18:10:00  

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
| 2022-10-31 20:10:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4449 |
| 2022-10-31 20:20:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4450 |
| 2022-10-31 20:30:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4451 |
| 2022-10-31 20:40:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4452 |
| 2022-10-31 20:50:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4453 |
| 2022-10-31 21:00:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4454 |
| 2022-10-31 21:10:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4455 |
| 2022-10-31 21:20:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4456 |
| 2022-10-31 21:30:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4457 |
| 2022-10-31 21:40:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 4458 |

### Bottom 10 Records

| Timestamp | RegimeLabel | RegimeState | RunID | EquipID | AssignmentConfidence | RegimeVersion | ID |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-11-15 18:10:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6597 |
| 2022-11-15 18:00:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6596 |
| 2022-11-15 17:50:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6595 |
| 2022-11-15 17:40:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6594 |
| 2022-11-15 17:30:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6593 |
| 2022-11-15 17:20:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6592 |
| 2022-11-15 17:10:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6591 |
| 2022-11-15 17:00:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6590 |
| 2022-11-15 16:50:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6589 |
| 2022-11-15 16:40:00 | -1 | unknown | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 0.0 | NULL | 6588 |

---


## dbo.ACM_RegimeTransitions

**Primary Key:** ID  
**Row Count:** 2  
**Date Range:** 2026-01-02 17:11:07 to 2026-01-02 17:11:07  

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
| 37 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | -1 | 0 | 109 | 1.0 | 2026-01-02 17:11:07 |
| 38 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | 0 | -1 | 109 | 1.0 | 2026-01-02 17:11:07 |

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
**Row Count:** 0  

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

---


## dbo.ACM_Run_Stats

**Primary Key:** RecordID  
**Row Count:** 1  
**Date Range:** 2022-10-09 08:40:00 to 2022-10-09 08:40:00  

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
| 5 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-10-09 08:40:00 | 2022-11-15 18:15:59 | 2149 | 2149 | 81 | 0.0 | NULL |

---


## dbo.ACM_Runs

**Primary Key:** RunID  
**Row Count:** 3  
**Date Range:** 2026-01-02 17:04:31 to 2026-01-02 17:30:31  

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
| 5C87CBC0-470C-4973-BE78-793100F4EC2A | 2621 | NULL | 2026-01-02 17:30:31 | NULL | NULL |  | NULL | NULL | NULL |
| 4709AAC2-5D24-43FE-8210-BDB80662F964 | 5010 | NULL | 2026-01-02 17:15:58 | NULL | NULL |  | NULL | NULL | NULL |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | WFA_TURBINE_10 | 2026-01-02 17:04:31 | 2026-01-02 17:15:17 | 645 |  | 2149 | 8 | 8 |

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
**Row Count:** 2,149  
**Date Range:** 2022-10-31 20:10:00 to 2022-11-15 18:10:00  

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
| 2022-10-31 20:10:00 | -1.2951147556304932 | 0.0 | 2.020492930951412e-12 | NULL | 0.4528471827507019 | 1.4132510423660278 | -2.43926739692688 | NULL | NULL |
| 2022-10-31 20:20:00 | 6.143909931182861 | 0.0 | 2.020492930951412e-12 | NULL | 2.0821330547332764 | 10.0 | -2.379836082458496 | NULL | NULL |
| 2022-10-31 20:30:00 | 2.7078609466552734 | 0.0 | 2.020492930951412e-12 | NULL | 2.073974847793579 | 10.0 | -2.296025037765503 | NULL | NULL |
| 2022-10-31 20:40:00 | 0.6348956823348999 | 0.0 | 2.020492930951412e-12 | NULL | 0.4355580806732178 | 3.5966439247131348 | -2.2201921939849854 | NULL | NULL |
| 2022-10-31 20:50:00 | -0.058136194944381714 | 0.0 | 2.020492930951412e-12 | NULL | 1.0360815525054932 | 2.5165507793426514 | -2.1549899578094482 | NULL | NULL |
| 2022-10-31 21:00:00 | 0.06387928128242493 | 0.0 | 2.020492930951412e-12 | NULL | 0.33902686834335327 | 0.742008626461029 | -2.1063976287841797 | NULL | NULL |
| 2022-10-31 21:10:00 | -0.6099005341529846 | 0.0 | 2.020492930951412e-12 | NULL | -0.008800684474408627 | 0.1089731901884079 | -2.076247215270996 | NULL | NULL |
| 2022-10-31 21:20:00 | -0.7597023248672485 | 0.0 | 2.020492930951412e-12 | NULL | -0.12018129974603653 | -0.6063557267189026 | -2.06296968460083 | NULL | NULL |
| 2022-10-31 21:30:00 | -0.1842242032289505 | 0.0 | 2.020492930951412e-12 | NULL | -0.08213785290718079 | 0.5840126872062683 | -2.0551187992095947 | NULL | NULL |
| 2022-10-31 21:40:00 | -0.2435627430677414 | 0.0 | 2.020492930951412e-12 | NULL | 0.17898470163345337 | 0.9488375186920166 | -2.0507876873016357 | NULL | NULL |

### Bottom 10 Records

| Timestamp | ar1_z | pca_spe_z | pca_t2_z | mhal_z | iforest_z | gmm_z | cusum_z | drift_z | hst_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-11-15 18:10:00 | -0.30330219864845276 | 0.0 | 3.8991965038703835e-12 | NULL | 0.7458319067955017 | 0.7507633566856384 | 2.8428921699523926 | NULL | NULL |
| 2022-11-15 18:00:00 | -0.48202526569366455 | 0.0 | 3.5801715324995653e-12 | NULL | 0.7075054049491882 | 0.8079203963279724 | 2.8410444259643555 | NULL | NULL |
| 2022-11-15 17:50:00 | -0.2898358404636383 | 0.0 | 8.578232742195713e-12 | NULL | 1.0802242755889893 | 0.9174686670303345 | 2.8390865325927734 | NULL | NULL |
| 2022-11-15 17:40:00 | 0.07934647053480148 | 0.0 | 8.790915911882635e-12 | NULL | 1.361511468887329 | 1.3005131483078003 | 2.8381481170654297 | NULL | NULL |
| 2022-11-15 17:30:00 | -0.603990375995636 | 0.0 | 1.7723620964957276e-12 | NULL | 0.4477172791957855 | -0.9523867964744568 | 2.8352572917938232 | NULL | NULL |
| 2022-11-15 17:20:00 | -0.521775484085083 | 0.0 | 1.3824424461086893e-12 | NULL | 0.7333997488021851 | -0.9545819163322449 | 2.8234493732452393 | NULL | NULL |
| 2022-11-15 17:10:00 | 0.07528027147054672 | 0.0 | -1.7723620964957276e-13 | NULL | 1.002386212348938 | 1.2872459888458252 | 2.8153774738311768 | NULL | NULL |
| 2022-11-15 17:00:00 | -0.336079478263855 | 0.0 | 1.985045591443302e-12 | NULL | 0.6320621371269226 | -0.828921914100647 | 2.810659408569336 | NULL | NULL |
| 2022-11-15 16:50:00 | -0.4348493814468384 | 0.0 | 1.3469952150207964e-12 | NULL | 0.8851815462112427 | -0.9686383008956909 | 2.797419548034668 | NULL | NULL |
| 2022-11-15 16:40:00 | -0.3639293611049652 | 0.0 | 4.253669085799855e-13 | NULL | 0.9482881426811218 | 1.0311338901519775 | 2.784158945083618 | NULL | NULL |

---


## dbo.ACM_SeasonalPatterns

**Primary Key:** ID  
**Row Count:** 121  
**Date Range:** 2026-01-02 22:43:04 to 2026-01-02 22:43:04  

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
| 80 | 5010 | power_29_avg | DAILY | 24.0 | 0.0672 | 5.0 | 0.4407 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 81 | 5010 | power_29_avg | WEEKLY | 168.0 | 0.3147 | 76.0 | 0.1108 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 82 | 5010 | power_29_max | DAILY | 24.0 | 0.0802 | 5.0 | 0.4552 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 83 | 5010 | power_29_max | WEEKLY | 168.0 | 0.3982 | 76.0 | 0.1569 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 84 | 5010 | power_29_min | DAILY | 24.0 | 0.042 | 16.0 | 0.3901 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 85 | 5010 | power_29_std | DAILY | 24.0 | 0.0228 | 8.0 | 0.1593 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 86 | 5010 | power_30_avg | DAILY | 24.0 | 0.0627 | 6.0 | 0.4302 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 87 | 5010 | power_30_avg | WEEKLY | 168.0 | 0.318 | 76.0 | 0.122 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 88 | 5010 | power_30_max | DAILY | 24.0 | 0.0749 | 6.0 | 0.4414 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 89 | 5010 | power_30_max | WEEKLY | 168.0 | 0.4118 | 76.0 | 0.1799 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |

### Bottom 10 Records

| ID | EquipID | SensorName | PatternType | PeriodHours | Amplitude | PhaseShift | Confidence | DetectedAt | RunID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 5010 | wind_speed_4_avg | WEEKLY | 168.0 | 3.7139 | 77.0 | 0.1337 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 199 | 5010 | wind_speed_4_avg | DAILY | 24.0 | 0.8928 | 5.0 | 0.4233 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 198 | 5010 | wind_speed_3_std | WEEKLY | 168.0 | 0.7889 | 77.0 | 0.1017 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 197 | 5010 | wind_speed_3_std | DAILY | 24.0 | 0.2933 | 6.0 | 0.4064 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 196 | 5010 | wind_speed_3_min | WEEKLY | 168.0 | 1.0778 | 69.0 | 0.1875 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 195 | 5010 | wind_speed_3_min | DAILY | 24.0 | 0.3445 | 16.0 | 0.4156 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 194 | 5010 | wind_speed_3_max | WEEKLY | 168.0 | 7.8528 | 76.0 | 0.1341 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 193 | 5010 | wind_speed_3_max | DAILY | 24.0 | 2.3722 | 6.0 | 0.4145 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 192 | 5010 | wind_speed_3_avg | WEEKLY | 168.0 | 3.9167 | 77.0 | 0.1225 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |
| 191 | 5010 | wind_speed_3_avg | DAILY | 24.0 | 0.8609 | 5.0 | 0.423 | 2026-01-02 22:43:04 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 |

---


## dbo.ACM_SensorCorrelations

**Primary Key:** ID  
**Row Count:** 3,160  
**Date Range:** 2026-01-02 17:12:53 to 2026-01-02 17:12:53  

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
| 9995 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | power_29_avg | 1.0 | pearson | 2026-01-02 17:12:53 |
| 9996 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | power_29_max | 0.9874375107282752 | pearson | 2026-01-02 17:12:53 |
| 9997 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | power_29_min | 0.798104957610822 | pearson | 2026-01-02 17:12:53 |
| 9998 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | power_29_std | 0.6713792821288759 | pearson | 2026-01-02 17:12:53 |
| 9999 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | power_30_avg | 0.9994265693095876 | pearson | 2026-01-02 17:12:53 |
| 10000 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | power_30_max | 0.9895554794008585 | pearson | 2026-01-02 17:12:53 |
| 10001 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | power_30_min | 0.8042379314279924 | pearson | 2026-01-02 17:12:53 |
| 10002 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | power_30_std | 0.7258579223932722 | pearson | 2026-01-02 17:12:53 |
| 10003 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | reactive_power_27_avg | 0.7556450302344758 | pearson | 2026-01-02 17:12:53 |
| 10004 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | power_29_avg | reactive_power_27_max | -0.13594368388080028 | pearson | 2026-01-02 17:12:53 |

### Bottom 10 Records

| ID | RunID | EquipID | Sensor1 | Sensor2 | Correlation | CorrelationType | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 13154 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_4_avg | wind_speed_4_avg | 1.0 | pearson | 2026-01-02 17:12:53 |
| 13153 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_std | wind_speed_4_avg | 0.9648256087543936 | pearson | 2026-01-02 17:12:53 |
| 13152 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_std | wind_speed_3_std | 1.0 | pearson | 2026-01-02 17:12:53 |
| 13151 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_min | wind_speed_4_avg | 0.8380790782362301 | pearson | 2026-01-02 17:12:53 |
| 13150 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_min | wind_speed_3_std | 0.7374637657959692 | pearson | 2026-01-02 17:12:53 |
| 13149 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_min | wind_speed_3_min | 1.0 | pearson | 2026-01-02 17:12:53 |
| 13148 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_max | wind_speed_4_avg | 0.9518321018105127 | pearson | 2026-01-02 17:12:53 |
| 13147 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_max | wind_speed_3_std | 0.9615896654754789 | pearson | 2026-01-02 17:12:53 |
| 13146 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_max | wind_speed_3_min | 0.7677948665130644 | pearson | 2026-01-02 17:12:53 |
| 13145 | d3dd89eb-0374-46cd-93dc-e3210a6dbb98 | 5010 | wind_speed_3_max | wind_speed_3_max | 1.0 | pearson | 2026-01-02 17:12:53 |

---


## dbo.ACM_SensorDefects

**Primary Key:** No primary key  
**Row Count:** 7  

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
| Correlation Break (PCA-SPE) | Correlation | LOW | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 |
| Multivariate Outlier (PCA-T2) | Multivariate | LOW | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 |
| Time-Series Anomaly (AR1) | Time-Series | LOW | 20 | 0.93 | 6.1439 | 0.5072 | 0.3033 | 0 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 |
| Density Anomaly (GMM) | Density | LOW | 74 | 3.44 | 10.0 | 0.7338 | 0.7508 | 0 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 |
| cusum_z | cusum_z | MEDIUM | 113 | 5.26 | 2.8429 | 0.8063 | 2.8429 | 1 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 |
| Rare State (IsolationForest) | Rare | MEDIUM | 168 | 7.82 | 4.6945 | 0.8487 | 0.7458 | 0 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 |
| Baseline Consistency (OMR) | Baseline | HIGH | 223 | 10.38 | 5.5189 | 0.9471 | 0.6573 | 0 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 |

---


## dbo.ACM_SensorForecast

**Primary Key:** RunID, EquipID, Timestamp, SensorName  
**Row Count:** 1,680  
**Date Range:** 2022-11-15 18:10:00 to 2022-11-22 17:10:00  

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

### Top 10 Records

| RunID | EquipID | Timestamp | SensorName | ForecastValue | CiLower | CiUpper | ForecastStd | Method | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | power_29_min | 0.5483825538535219 | 0.41820787846266844 | 0.6785572292443753 | 0.06641565070961909 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | power_30_min | 0.5658897160372565 | 0.421116299930858 | 0.710663132143655 | 0.07386398780938697 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | sensor_18_std | 30.174311199870104 | -99.92340186038832 | 160.27202426012855 | 66.37638421441757 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | sensor_2_avg | 2.126176808145429 | -35.839835852625015 | 40.092189468915876 | 19.370414622842063 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | sensor_26_avg | 50.030828442692005 | 50.01678961698126 | 50.044867268402754 | 0.00716266617895482 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | sensor_31_std | 3.1319461397915993 | -19.354278451842674 | 25.61817073142587 | 11.472563567160343 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | sensor_44 | -54.28863446543081 | -1257.275598591218 | 1148.6983296603566 | 613.7688592478507 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | sensor_47 | -53.54863519174322 | -1523.7223484133594 | 1416.625078029873 | 750.0886291947022 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | sensor_5_std | 2.2335804419059753 | -0.2771317179167676 | 4.744292601728718 | 1.2809755917462975 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 18:10:00 | sensor_52_std | 0.2563635927059314 | -1.0053544695528462 | 1.518081654964709 | 0.6437337052340703 | ExponentialSmoothing | 0 |

### Bottom 10 Records

| RunID | EquipID | Timestamp | SensorName | ForecastValue | CiLower | CiUpper | ForecastStd | Method | RegimeLabel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | sensor_52_std | -2.0846795713161868 | -3.3463976335749646 | -0.8229615090574092 | 0.6437337052340703 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | sensor_5_std | 21.403543137091546 | 18.892830977268805 | 23.914255296914288 | 1.2809755917462975 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | sensor_47 | 598.2795645591311 | -871.8941486624851 | 2068.453277780747 | 750.0886291947022 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | sensor_44 | 190.6216836814389 | -1012.3652804443485 | 1393.6086478072264 | 613.7688592478507 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | sensor_31_std | -80.29963737065758 | -102.78586196229185 | -57.81341277902331 | 11.472563567160343 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | sensor_26_avg | 50.49517529316666 | 50.48113646745591 | 50.50921411887741 | 0.00716266617895482 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | sensor_2_avg | 21.4207546951289 | -16.545257965641547 | 59.386767355899345 | 19.370414622842063 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | sensor_18_std | -219.13356048353552 | -349.2312735437939 | -89.03584742327709 | 66.37638421441757 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | power_30_min | 5.229294194134021 | 5.084520778027623 | 5.374067610240419 | 0.07386398780938697 | ExponentialSmoothing | 0 |
| D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-22 17:10:00 | power_29_min | 5.034502239606962 | 4.904327564216108 | 5.164676914997815 | 0.06641565070961909 | ExponentialSmoothing | 0 |

---


## dbo.ACM_SensorHotspots

**Primary Key:** No primary key  
**Row Count:** 25  
**Date Range:** 2022-11-01 03:10:00 to 2022-11-15 14:20:00  

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
| sensor_34_avg | 2022-11-01 03:10:00 | 2022-11-15 18:10:00 | 3.3176 | 3.3176 | 0.2329 | -0.2329 | 405.8436617750205 | 398.0091677928005 | 398.5231570167478 |
| sensor_44 | 2022-11-01 22:20:00 | 2022-11-15 18:10:00 | 9.6678 | -9.6678 | 0.3751 | 0.3751 | -4423.121704338643 | 9.619876303703625 | -155.94624728568039 |
| sensor_31_max | 2022-11-02 10:00:00 | 2022-11-15 18:10:00 | 7.0097 | 7.0097 | 1.0749 | -1.0749 | 646.3545630313783 | -181.10112117285917 | -71.08361515723345 |
| sensor_31_std | 2022-11-03 06:10:00 | 2022-11-15 18:10:00 | 114.1309 | 114.1309 | 0.0355 | -0.0355 | 325.29998779296875 | 3.700000047683716 | 3.799999952316284 |
| sensor_31_min | 2022-11-03 06:30:00 | 2022-11-15 18:10:00 | 3.8798 | -3.8798 | 0.1038 | 0.1038 | -867.1453431584976 | -121.3284267880012 | -140.7577807409416 |
| sensor_33_avg | 2022-11-03 06:40:00 | 2022-11-15 18:10:00 | 4.0469 | -4.0469 | 0.9198 | -0.9198 | 391.70001220703125 | 396.79998779296875 | 398.29998779296875 |
| sensor_32_avg | 2022-11-03 06:40:00 | 2022-11-15 18:10:00 | 3.8016 | -3.8016 | 1.2263 | -1.2263 | 394.20001220703125 | 398.3999938964844 | 400.3999938964844 |
| sensor_47 | 2022-11-03 18:10:00 | 2022-11-15 18:10:00 | 9.4925 | -9.4925 | 0.4266 | 0.4266 | -4067.269545633542 | 11.730454366457982 | -163.7127111216936 |
| sensor_22_avg | 2022-11-04 10:20:00 | 2022-11-15 18:10:00 | 4.0992 | -4.0992 | 0.4081 | 0.4081 | 0.2783870021923561 | 0.9826007892975996 | 0.9188405424356461 |
| sensor_52_std | 2022-11-04 14:10:00 | 2022-11-15 18:10:00 | 11.6912 | 11.6912 | 0.4497 | -0.4497 | 5.5 | 0.10000000149011612 | 0.30000001192092896 |

### Bottom 10 Records

| SensorName | MaxTimestamp | LatestTimestamp | MaxAbsZ | MaxSignedZ | LatestAbsZ | LatestSignedZ | ValueAtPeak | LatestValue | TrainMean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| power_30_min | 2022-11-15 14:20:00 | 2022-11-15 18:10:00 | 8.7629 | 8.7629 | 3.8378 | 3.8378 | 0.865847764289015 | 0.3975836623535441 | 0.03270372598260145 |
| power_29_min | 2022-11-15 14:20:00 | 2022-11-15 18:10:00 | 8.9885 | 8.9885 | 3.9243 | 3.9243 | 0.8383034496538968 | 0.3891192815102922 | 0.041043402051674116 |
| wind_speed_3_max | 2022-11-15 06:10:00 | 2022-11-15 18:10:00 | 4.7142 | 4.7142 | 1.5601 | 1.5601 | 46.4687365806644 | 23.99280272725148 | 12.875402537162774 |
| wind_speed_3_std | 2022-11-15 01:20:00 | 2022-11-15 18:10:00 | 3.6866 | 3.6866 | 1.3587 | 1.3587 | 4.0135008700347505 | 2.273514018823494 | 1.2579737086603164 |
| sensor_5_std | 2022-11-14 16:10:00 | 2022-11-15 18:10:00 | 66.7746 | 66.7746 | 0.3372 | 0.3372 | 40.0 | 0.6000000238418579 | 0.4000000059604645 |
| sensor_41_avg | 2022-11-14 09:50:00 | 2022-11-15 18:10:00 | 6.7449 | -6.7449 | 0.0 | 0.0 | 28.0 | 48.0 | 48.0 |
| sensor_26_avg | 2022-11-13 15:50:00 | 2022-11-15 18:10:00 | 18.8815 | -18.8815 | 0.7467 | -0.7467 | 49.89481656987967 | 49.995622886421096 | 49.999773619932135 |
| sensor_6_avg | 2022-11-11 09:30:00 | 2022-11-15 18:10:00 | 3.4998 | -3.4998 | 0.2976 | -0.2976 | 16.242472475707874 | 29.731321742687676 | 30.984696636810387 |
| sensor_0_avg | 2022-11-11 08:40:00 | 2022-11-15 18:10:00 | 3.9912 | -3.9912 | 0.784 | -0.784 | 6.1339245073532425 | 18.458783116338026 | 21.471642828385026 |
| sensor_53_avg | 2022-11-11 08:40:00 | 2022-11-15 18:10:00 | 3.2547 | -3.2547 | 0.6854 | -0.6854 | 8.288019670034622 | 20.804917850457592 | 24.143862406570257 |

---


## dbo.ACM_SensorNormalized_TS

**Primary Key:** ID  
**Row Count:** 10,287  
**Date Range:** 2022-10-31 20:10:00 to 2022-11-15 17:10:00  

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
| 22195 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-10-31 20:10:00 | power_29_avg | NULL | 0.8939787222817317 | 2026-01-02 17:13:02 |
| 22196 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-10-31 23:00:00 | power_29_avg | NULL | 1.135937435507024 | 2026-01-02 17:13:02 |
| 22197 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-01 01:50:00 | power_29_avg | NULL | 1.2419674003351886 | 2026-01-02 17:13:02 |
| 22198 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-01 04:40:00 | power_29_avg | NULL | 1.1004380220674863 | 2026-01-02 17:13:02 |
| 22199 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-01 07:30:00 | power_29_avg | NULL | 1.1367104797506136 | 2026-01-02 17:13:02 |
| 22200 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-01 10:20:00 | power_29_avg | NULL | 1.0887782612277699 | 2026-01-02 17:13:02 |
| 22201 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-01 13:10:00 | power_29_avg | NULL | 1.0939107173218616 | 2026-01-02 17:13:02 |
| 22202 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-01 16:00:00 | power_29_avg | NULL | 0.6890389313753614 | 2026-01-02 17:13:02 |
| 22203 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-01 18:50:00 | power_29_avg | NULL | 0.8796244098547918 | 2026-01-02 17:13:02 |
| 22204 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-01 21:40:00 | power_29_avg | NULL | 1.1325092164438653 | 2026-01-02 17:13:02 |

### Bottom 10 Records

| ID | RunID | EquipID | Timestamp | SensorName | RawValue | NormalizedValue | CreatedAt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32481 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 17:10:00 | wind_speed_4_avg | NULL | 12.954494204766261 | 2026-01-02 17:13:03 |
| 32480 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 14:20:00 | wind_speed_4_avg | NULL | 16.011215179364353 | 2026-01-02 17:13:03 |
| 32479 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 11:30:00 | wind_speed_4_avg | NULL | 13.828123937955471 | 2026-01-02 17:13:03 |
| 32478 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 08:40:00 | wind_speed_4_avg | NULL | 12.168455643300744 | 2026-01-02 17:13:03 |
| 32477 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 05:50:00 | wind_speed_4_avg | NULL | 12.851540736377588 | 2026-01-02 17:13:03 |
| 32476 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 03:00:00 | wind_speed_4_avg | NULL | 16.695268172412433 | 2026-01-02 17:13:03 |
| 32475 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-15 00:10:00 | wind_speed_4_avg | NULL | 13.864636577307193 | 2026-01-02 17:13:03 |
| 32474 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-14 21:20:00 | wind_speed_4_avg | NULL | 11.948331389053378 | 2026-01-02 17:13:03 |
| 32473 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-14 18:30:00 | wind_speed_4_avg | NULL | 7.170544701418322 | 2026-01-02 17:13:03 |
| 32472 | D3DD89EB-0374-46CD-93DC-E3210A6DBB98 | 5010 | 2022-11-14 15:40:00 | wind_speed_4_avg | NULL | 7.079264533008439 | 2026-01-02 17:13:03 |

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
**Row Count:** 70  
**Date Range:** 2025-12-27 06:26:15 to 2026-01-02 17:11:18  

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
| ar1_params | 5000 | 1 | 2025-12-27 06:26:15 | {"n_sensors": 790, "mean_autocorr": 655489383.9208, "mean_residual_std": 448915724.5588, "params_... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 43780 bytes> |
| gmm_model | 5000 | 1 | 2025-12-27 06:26:20 | {"n_components": 3, "covariance_type": "diag", "bic": 6.829492608516723e+25, "aic": 6.82949260851... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 76880 bytes> |
| iforest_model | 5000 | 1 | 2025-12-27 06:26:20 | {"n_estimators": 100, "contamination": 0.01, "max_features": 1.0, "max_samples": 2048} | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 1422089 bytes> |
| omr_model | 5000 | 1 | 2025-12-27 06:26:23 | NULL | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 83948 bytes> |
| pca_model | 5000 | 1 | 2025-12-27 06:26:16 | {"n_components": 5, "variance_ratio_sum": 0.6513, "variance_ratio_first_component": 0.2442, "vari... | {"train_rows": 129, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med",... | NULL | <binary 38975 bytes> |

### Bottom 10 Records

| ModelType | EquipID | Version | EntryDateTime | ParamsJSON | StatsJSON | RunID | ModelBytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ar1_params | 5092 | 5 | 2025-12-31 08:34:38 | {"n_sensors": 632, "mean_autocorr": -2.609685739840889e+20, "mean_residual_std": Infinity, "param... | {"train_rows": 27160, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 34392 bytes> |
| ar1_params | 5092 | 4 | 2025-12-31 07:17:38 | {"n_sensors": 632, "mean_autocorr": -9.494212814703844e+19, "mean_residual_std": Infinity, "param... | {"train_rows": 27165, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 34392 bytes> |
| ar1_params | 5092 | 3 | 2025-12-31 06:07:00 | {"n_sensors": 632, "mean_autocorr": -2.44824132505186e+19, "mean_residual_std": Infinity, "params... | {"train_rows": 27160, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 34392 bytes> |
| ar1_params | 5092 | 2 | 2025-12-31 04:59:28 | {"n_sensors": 632, "mean_autocorr": -1.4690724257372932e+20, "mean_residual_std": Infinity, "para... | {"train_rows": 27160, "train_sensors": ["power_29_avg_med", "power_29_max_med", "power_29_min_med... | NULL | <binary 34392 bytes> |
| ar1_params | 5092 | 1 | 2025-12-31 04:24:19 | {"n_sensors": 566, "mean_autocorr": 1459464706535.6265, "mean_residual_std": 2032131359443.5737, ... | {"train_rows": 49, "train_sensors": ["power_29_avg_med", "power_29_min_med", "power_29_std_med", ... | NULL | <binary 30678 bytes> |
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

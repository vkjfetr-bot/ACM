# ACM SQL Schema Reference (v10)
_Generated automatically on 2025-12-05 10:26:59_

This document is produced by `python scripts/sql/export_schema_doc.py` and reflects the live structure
of the `ACM` database. Re-run the script whenever tables change.

## CRITICAL: Column & Table Naming for Dashboard Development

**⚠️ Common Dashboard Mistakes to Avoid:**

| Mistake | Correct Column/Table | Used In |
|---------|----------------------|---------|
| ❌ `LatestHealthScore` | ✓ `HealthIndex` | ACM_HealthTimeline |
| ✓ `ACM_HealthForecast` | Column: `ForecastHealth` | V10 Forecasting |
| ✓ `ACM_FailureForecast` | Column: `FailureProb` | V10 Failure Probability |
| ✓ `ACM_SensorForecast` | Column: `ForecastValue` | V10 Physical Sensor Predictions |
| ✓ `ACM_RUL` | Column: `RUL_Hours` | V10 RUL Reporting |
| ❌ `DriftScore` | ✓ `DriftValue` | ACM_DriftSeries |
| ⚠️  `ACM_OMRContributionsLong` | ✓ `ACM_ContributionTimeline` | Detector Fusion (use DetectorType instead of SensorName) |

**Best Practice:** Always validate queries against V2 dashboard patterns before implementing new dashboards.

| Table | Column Count | Primary Key |
| --- | ---: | --- |
| dbo.ACM_AdaptiveConfig | 13 | ConfigID |
| dbo.ACM_AlertAge | 6 | — |
| dbo.ACM_Anomaly_Events | 6 | Id |
| dbo.ACM_BaselineBuffer | 7 | Id |
| dbo.ACM_CalibrationSummary | 10 | — |
| dbo.ACM_ColdstartState | 17 | EquipID, Stage |
| dbo.ACM_Config | 7 | ConfigID |
| dbo.ACM_ConfigHistory | 9 | ID |
| dbo.ACM_ContributionCurrent | 5 | — |
| dbo.ACM_ContributionTimeline | 5 | — |
| dbo.ACM_DailyFusedProfile | 9 | ID |
| dbo.ACM_DataQuality | 24 | — |
| dbo.ACM_DefectSummary | 12 | — |
| dbo.ACM_DefectTimeline | 10 | — |
| dbo.ACM_DetectorCorrelation | 7 | — |
| dbo.ACM_DetectorForecast_TS | 10 | RunID, EquipID, DetectorName, Timestamp |
| dbo.ACM_DriftEvents | 2 | — |
| dbo.ACM_DriftSeries | 4 | — |
| dbo.ACM_EpisodeCulprits | 8 | ID |
| dbo.ACM_EpisodeDiagnostics | 13 | ID |
| dbo.ACM_EpisodeMetrics | 10 | — |
| dbo.ACM_Episodes | 8 | — |
| dbo.ACM_EpisodesQC | 10 | RecordID |
| dbo.ACM_FailureForecast | 9 | EquipID, RunID, Timestamp |
| dbo.ACM_FeatureDropLog | 8 | LogID |
| dbo.ACM_ForecastingState | 13 | EquipID, StateVersion |
| dbo.ACM_ForecastState | 12 | EquipID, StateVersion |
| dbo.ACM_FusionQualityReport | 9 | — |
| dbo.ACM_HealthDistributionOverTime | 12 | — |
| dbo.ACM_HealthForecast | 9 | EquipID, RunID, Timestamp |
| dbo.ACM_HealthHistogram | 5 | — |
| dbo.ACM_HealthTimeline | 6 | — |
| dbo.ACM_HealthZoneByPeriod | 9 | — |
| dbo.ACM_HistorianData | 7 | DataID |
| dbo.ACM_OMR_Diagnostics | 15 | DiagnosticID |
| dbo.ACM_OMRContributionsLong | 8 | — |
| dbo.ACM_OMRTimeline | 6 | — |
| dbo.ACM_PCA_Loadings | 10 | RecordID |
| dbo.ACM_PCA_Metrics | 6 | RunID, EquipID, ComponentName, MetricType |
| dbo.ACM_PCA_Models | 12 | RecordID |
| dbo.ACM_RefitRequests | 10 | RequestID |
| dbo.ACM_Regime_Episodes | 6 | Id |
| dbo.ACM_RegimeDwellStats | 8 | — |
| dbo.ACM_RegimeOccupancy | 5 | — |
| dbo.ACM_RegimeStability | 4 | — |
| dbo.ACM_RegimeState | 15 | EquipID, StateVersion |
| dbo.ACM_RegimeStats | 8 | — |
| dbo.ACM_RegimeTimeline | 5 | — |
| dbo.ACM_RegimeTransitions | 6 | — |
| dbo.ACM_RUL | 14 | EquipID, RunID |
| dbo.ACM_RUL_LearningState | 19 | EquipID |
| dbo.ACM_Run_Stats | 13 | RecordID |
| dbo.ACM_RunLogs | 25 | LogID |
| dbo.ACM_RunMetadata | 12 | RunMetadataID |
| dbo.ACM_RunMetrics | 5 | RunID, EquipID, MetricName |
| dbo.ACM_Runs | 19 | RunID |
| dbo.ACM_SchemaVersion | 5 | VersionID |
| dbo.ACM_Scores_Long | 9 | Id |
| dbo.ACM_Scores_Wide | 15 | — |
| dbo.ACM_SensorAnomalyByPeriod | 11 | — |
| dbo.ACM_SensorDefects | 11 | — |
| dbo.ACM_SensorHotspots | 18 | — |
| dbo.ACM_SensorHotspotTimeline | 9 | — |
| dbo.ACM_SensorForecast | 11 | RunID, EquipID, Timestamp, SensorName |
| dbo.ACM_SensorNormalized_TS | 10 | Id |
| dbo.ACM_SensorRanking | 6 | — |
| dbo.ACM_SinceWhen | 6 | — |
| dbo.ACM_TagEquipmentMap | 10 | TagID |
| dbo.ACM_ThresholdCrossings | 7 | — |
| dbo.ACM_ThresholdMetadata | 13 | ThresholdID |
| dbo.Equipment | 8 | EquipID |
| dbo.FD_FAN_Data | 11 | EntryDateTime |
| dbo.GAS_TURBINE_Data | 18 | EntryDateTime |
| dbo.ModelRegistry | 8 | ModelType, EquipID, Version |

## dbo.ACM_AdaptiveConfig
- **Primary Key:** ConfigID

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

## dbo.ACM_AlertAge
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| AlertZone | nvarchar | NO | 50 | — |
| StartTimestamp | datetime2 | NO | — | — |
| DurationHours | float | NO | 53 | — |
| RecordCount | int | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_Anomaly_Events
- **Primary Key:** Id

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | YES | 10 | — |
| StartTime | datetime2 | YES | — | — |
| EndTime | datetime2 | YES | — | — |
| Severity | nvarchar | YES | 32 | — |

## dbo.ACM_BaselineBuffer
- **Primary Key:** Id

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime | NO | — | — |
| SensorName | nvarchar | NO | 128 | — |
| SensorValue | float | NO | 53 | — |
| DataQuality | nvarchar | YES | 64 | — |
| CreatedAt | datetime | NO | — | (getdate()) |

## dbo.ACM_CalibrationSummary
- **Primary Key:** —

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

## dbo.ACM_ColdstartState
- **Primary Key:** EquipID, Stage

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

## dbo.ACM_Config
- **Primary Key:** ConfigID

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| ConfigID | int | NO | 10 | — |
| EquipID | int | NO | 10 | — |
| ParamPath | nvarchar | NO | 500 | — |
| ParamValue | nvarchar | NO | -1 | — |
| ValueType | varchar | NO | 50 | — |
| UpdatedAt | datetime2 | NO | — | (getutcdate()) |
| UpdatedBy | nvarchar | YES | 100 | (suser_sname()) |

## dbo.ACM_ConfigHistory
- **Primary Key:** ID

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

## dbo.ACM_ContributionCurrent
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorType | nvarchar | NO | 50 | — |
| ContributionPct | float | NO | 53 | — |
| ZScore | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_ContributionTimeline
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| DetectorType | nvarchar | NO | 50 | — |
| ContributionPct | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_DailyFusedProfile
- **Primary Key:** ID

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

## dbo.ACM_DataQuality
- **Primary Key:** —

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

## dbo.ACM_DefectSummary
- **Primary Key:** —

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

## dbo.ACM_DefectTimeline
- **Primary Key:** —

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

## dbo.ACM_DetectorCorrelation
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorA | nvarchar | NO | 50 | — |
| DetectorB | nvarchar | NO | 50 | — |
| PearsonR | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| PairLabel | nvarchar | YES | 256 | — |
| DisturbanceHint | nvarchar | YES | 256 | — |

## dbo.ACM_DetectorForecast_TS
- **Primary Key:** RunID, EquipID, DetectorName, Timestamp

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

## dbo.ACM_DriftEvents
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_DriftSeries
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| DriftValue | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_EpisodeCulprits
- **Primary Key:** ID

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

## dbo.ACM_EpisodeDiagnostics
- **Primary Key:** ID

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

## dbo.ACM_EpisodeMetrics
- **Primary Key:** —

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

## dbo.ACM_Episodes
- **Primary Key:** —

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

## dbo.ACM_EpisodesQC
- **Primary Key:** RecordID

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

## dbo.ACM_FailureForecast
- **Primary Key:** EquipID, RunID, Timestamp

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

## dbo.ACM_FeatureDropLog
- **Primary Key:** LogID

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

## dbo.ACM_ForecastingState
- **Primary Key:** EquipID, StateVersion

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

## dbo.ACM_ForecastState
- **Primary Key:** EquipID, StateVersion

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

## dbo.ACM_FusionQualityReport
- **Primary Key:** —

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

## dbo.ACM_HealthDistributionOverTime
- **Primary Key:** —

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

## dbo.ACM_HealthForecast
- **Primary Key:** EquipID, RunID, Timestamp

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

## dbo.ACM_HealthHistogram
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| HealthBin | nvarchar | NO | 50 | — |
| RecordCount | int | NO | 10 | — |
| Percentage | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_HealthTimeline
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| HealthIndex | float | NO | 53 | — |
| HealthZone | nvarchar | NO | 50 | — |
| FusedZ | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_HealthZoneByPeriod
- **Primary Key:** —

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

## dbo.ACM_HistorianData
- **Primary Key:** DataID

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DataID | bigint | NO | 19 | — |
| EquipID | int | NO | 10 | — |
| TagName | varchar | NO | 255 | — |
| Timestamp | datetime2 | NO | — | — |
| Value | float | NO | 53 | — |
| Quality | tinyint | YES | 3 | ((192)) |
| CreatedAt | datetime2 | YES | — | (getutcdate()) |

## dbo.ACM_OMR_Diagnostics
- **Primary Key:** DiagnosticID

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

## dbo.ACM_OMRContributionsLong
- **Primary Key:** —

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

## dbo.ACM_OMRTimeline
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| Timestamp | datetime2 | NO | — | — |
| OMR_Z | float | YES | 53 | — |
| OMR_Weight | float | YES | 53 | — |
| CreatedAt | datetime2 | NO | — | (sysutcdatetime()) |

## dbo.ACM_PCA_Loadings
- **Primary Key:** RecordID

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

## dbo.ACM_PCA_Metrics
- **Primary Key:** RunID, EquipID, ComponentName, MetricType

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | nvarchar | NO | 72 | — |
| EquipID | int | NO | 10 | — |
| ComponentName | nvarchar | NO | 200 | — |
| MetricType | nvarchar | NO | 50 | — |
| Value | float | NO | 53 | — |
| Timestamp | datetime2 | NO | — | (sysutcdatetime()) |

## dbo.ACM_PCA_Models
- **Primary Key:** RecordID

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

## dbo.ACM_RefitRequests
- **Primary Key:** RequestID

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

## dbo.ACM_Regime_Episodes
- **Primary Key:** Id

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Id | bigint | NO | 19 | — |
| RunID | uniqueidentifier | YES | — | — |
| EquipID | int | YES | 10 | — |
| StartTime | datetime2 | YES | — | — |
| EndTime | datetime2 | YES | — | — |
| RegimeID | int | YES | 10 | — |

## dbo.ACM_RegimeDwellStats
- **Primary Key:** —

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

## dbo.ACM_RegimeOccupancy
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RegimeLabel | nvarchar | NO | 50 | — |
| RecordCount | int | NO | 10 | — |
| Percentage | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_RegimeStability
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| MetricName | nvarchar | NO | 100 | — |
| MetricValue | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_RegimeState
- **Primary Key:** EquipID, StateVersion

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

## dbo.ACM_RegimeStats
- **Primary Key:** —

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

## dbo.ACM_RegimeTimeline
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| RegimeLabel | nvarchar | NO | 50 | — |
| RegimeState | nvarchar | NO | 50 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_RegimeTransitions
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| FromLabel | nvarchar | NO | 50 | — |
| ToLabel | nvarchar | NO | 50 | — |
| Count | int | NO | 10 | — |
| Prob | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_RUL
- **Primary Key:** EquipID, RunID

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

## dbo.ACM_RUL_LearningState
- **Primary Key:** EquipID

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| EquipID | int | NO | 10 | — |
| AR1_MAE | float | YES | 53 | — |

## dbo.ACM_SensorForecast
- **Primary Key:** RunID, EquipID, Timestamp, SensorName
- **Purpose:** Physical sensor value forecasts (Motor Current, Bearing Temperature, Pressure, etc.)
- **Forecast Methods:** LinearTrend (default), VAR (Vector AutoRegression for multivariate)
- **Forecast Horizon:** Configurable (default 168 hours / 7 days)

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

## dbo.ACM_Run_Stats
- **Primary Key:** RecordID

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

## dbo.ACM_RunLogs
- **Primary Key:** LogID

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

## dbo.ACM_RunMetadata
- **Primary Key:** RunMetadataID

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

## dbo.ACM_RunMetrics
- **Primary Key:** RunID, EquipID, MetricName

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | nvarchar | NO | 72 | — |
| EquipID | int | NO | 10 | — |
| MetricName | nvarchar | NO | 100 | — |
| MetricValue | float | NO | 53 | — |
| Timestamp | datetime2 | NO | — | (getdate()) |

## dbo.ACM_Runs
- **Primary Key:** RunID

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

## dbo.ACM_SchemaVersion
- **Primary Key:** VersionID

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| VersionID | int | NO | 10 | — |
| VersionNumber | varchar | NO | 20 | — |
| Description | varchar | YES | 500 | — |
| AppliedAt | datetime2 | NO | — | (getutcdate()) |
| AppliedBy | varchar | NO | 100 | (suser_sname()) |

## dbo.ACM_Scores_Long
- **Primary Key:** Id

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

## dbo.ACM_Scores_Wide
- **Primary Key:** —

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

## dbo.ACM_SensorAnomalyByPeriod
- **Primary Key:** —

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

## dbo.ACM_SensorDefects
- **Primary Key:** —

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

## dbo.ACM_SensorHotspots
- **Primary Key:** —

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

## dbo.ACM_SensorHotspotTimeline
- **Primary Key:** —

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

## dbo.ACM_SensorNormalized_TS
- **Primary Key:** Id

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

## dbo.ACM_SensorRanking
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| DetectorType | nvarchar | NO | 50 | — |
| RankPosition | int | NO | 10 | — |
| ContributionPct | float | NO | 53 | — |
| ZScore | float | NO | 53 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_SinceWhen
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |
| AlertZone | nvarchar | NO | 50 | — |
| DurationHours | float | NO | 53 | — |
| StartTimestamp | datetime2 | NO | — | — |
| RecordCount | int | NO | 10 | — |

## dbo.ACM_TagEquipmentMap
- **Primary Key:** TagID

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

## dbo.ACM_ThresholdCrossings
- **Primary Key:** —

| Column | Data Type | Nullable | Length/Precision | Default |
| --- | --- | --- | --- | --- |
| Timestamp | datetime2 | NO | — | — |
| DetectorType | nvarchar | NO | 50 | — |
| Threshold | float | NO | 53 | — |
| ZScore | float | NO | 53 | — |
| Direction | nvarchar | NO | 10 | — |
| RunID | uniqueidentifier | NO | — | — |
| EquipID | int | NO | 10 | — |

## dbo.ACM_ThresholdMetadata
- **Primary Key:** ThresholdID

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

## dbo.Equipment
- **Primary Key:** EquipID

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

## dbo.FD_FAN_Data
- **Primary Key:** EntryDateTime

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

## dbo.GAS_TURBINE_Data
- **Primary Key:** EntryDateTime

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

## dbo.ModelRegistry
- **Primary Key:** ModelType, EquipID, Version

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
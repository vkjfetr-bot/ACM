USE [ACM];
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_AdaptiveConfig_Equip' AND object_id = OBJECT_ID('dbo.[ACM_AdaptiveConfig]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_AdaptiveConfig_Equip] ON dbo.[ACM_AdaptiveConfig] ([EquipID], [ConfigKey]) INCLUDE ([ConfigValue], [MinBound], [MaxBound], [IsLearned]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_AdaptiveConfig_Learned' AND object_id = OBJECT_ID('dbo.[ACM_AdaptiveConfig]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_AdaptiveConfig_Learned] ON dbo.[ACM_AdaptiveConfig] ([IsLearned], [ConfigKey]) WHERE ([IsLearned]=(1));
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_BaselineBuffer_Equip_Sensor_Timestamp' AND object_id = OBJECT_ID('dbo.[ACM_BaselineBuffer]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_BaselineBuffer_Equip_Sensor_Timestamp] ON dbo.[ACM_BaselineBuffer] ([EquipID], [SensorName], [Timestamp]) INCLUDE ([SensorValue], [DataQuality]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_BaselineBuffer_Equip_Timestamp' AND object_id = OBJECT_ID('dbo.[ACM_BaselineBuffer]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_BaselineBuffer_Equip_Timestamp] ON dbo.[ACM_BaselineBuffer] ([EquipID], [Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ColdstartState_Status' AND object_id = OBJECT_ID('dbo.[ACM_ColdstartState]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ColdstartState_Status] ON dbo.[ACM_ColdstartState] ([Status], [LastAttemptAt]) WHERE ([Status] IN ('PENDING', 'IN_PROGRESS'));
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_Config_ParamPath' AND object_id = OBJECT_ID('dbo.[ACM_Config]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_Config_ParamPath] ON dbo.[ACM_Config] ([ParamPath]) INCLUDE ([ParamValue], [ValueType]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_DailyFused_Equip_Date' AND object_id = OBJECT_ID('dbo.[ACM_DailyFusedProfile]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_DailyFused_Equip_Date] ON dbo.[ACM_DailyFusedProfile] ([EquipID], [ProfileDate]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_DailyFused_RunID' AND object_id = OBJECT_ID('dbo.[ACM_DailyFusedProfile]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_DailyFused_RunID] ON dbo.[ACM_DailyFusedProfile] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_DetectorForecast_TS_Equip_Det_Ts_CreatedAt' AND object_id = OBJECT_ID('dbo.[ACM_DetectorForecast_TS]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_DetectorForecast_TS_Equip_Det_Ts_CreatedAt] ON dbo.[ACM_DetectorForecast_TS] ([EquipID], [DetectorName], [Timestamp], [CreatedAt]) INCLUDE ([RunID], [ForecastValue]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_DetectorForecast_TS_EquipID_Timestamp' AND object_id = OBJECT_ID('dbo.[ACM_DetectorForecast_TS]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_DetectorForecast_TS_EquipID_Timestamp] ON dbo.[ACM_DetectorForecast_TS] ([EquipID], [Timestamp]) INCLUDE ([DetectorName], [ForecastValue], [Method]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_EpisodeCulprits_EpisodeID' AND object_id = OBJECT_ID('dbo.[ACM_EpisodeCulprits]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_EpisodeCulprits_EpisodeID] ON dbo.[ACM_EpisodeCulprits] ([EpisodeID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_EpisodeCulprits_RunID' AND object_id = OBJECT_ID('dbo.[ACM_EpisodeCulprits]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_EpisodeCulprits_RunID] ON dbo.[ACM_EpisodeCulprits] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_EpisodeDiagnostics_RunID' AND object_id = OBJECT_ID('dbo.[ACM_EpisodeDiagnostics]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_EpisodeDiagnostics_RunID] ON dbo.[ACM_EpisodeDiagnostics] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_EpisodesQC_EquipID' AND object_id = OBJECT_ID('dbo.[ACM_EpisodesQC]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_EpisodesQC_EquipID] ON dbo.[ACM_EpisodesQC] ([EquipID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_EpisodesQC_RunID' AND object_id = OBJECT_ID('dbo.[ACM_EpisodesQC]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_EpisodesQC_RunID] ON dbo.[ACM_EpisodesQC] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_FailureForecast_Time' AND object_id = OBJECT_ID('dbo.[ACM_FailureForecast]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_FailureForecast_Time] ON dbo.[ACM_FailureForecast] ([EquipID], [Timestamp]) INCLUDE ([FailureProb], [SurvivalProb], [HazardRate]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_FailureHazard_Time' AND object_id = OBJECT_ID('dbo.[ACM_FailureHazard_TS]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_FailureHazard_Time] ON dbo.[ACM_FailureHazard_TS] ([EquipID], [Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_FeatureDropLog_EquipID' AND object_id = OBJECT_ID('dbo.[ACM_FeatureDropLog]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_FeatureDropLog_EquipID] ON dbo.[ACM_FeatureDropLog] ([EquipID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_FeatureDropLog_RunID' AND object_id = OBJECT_ID('dbo.[ACM_FeatureDropLog]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_FeatureDropLog_RunID] ON dbo.[ACM_FeatureDropLog] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_FeatureDropLog_Timestamp' AND object_id = OBJECT_ID('dbo.[ACM_FeatureDropLog]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_FeatureDropLog_Timestamp] ON dbo.[ACM_FeatureDropLog] ([Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_ForecastingState_Latest' AND object_id = OBJECT_ID('dbo.[ACM_ForecastingState]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_ForecastingState_Latest] ON dbo.[ACM_ForecastingState] ([EquipID], [StateVersion]) INCLUDE ([ModelCoefficientsJson], [LastForecastJson], [LastRetrainTime]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_ACM_ForecastingState_Equip' AND object_id = OBJECT_ID('dbo.[ACM_ForecastingState]'))
BEGIN
    CREATE UNIQUE NONCLUSTERED INDEX [UQ_ACM_ForecastingState_Equip] ON dbo.[ACM_ForecastingState] ([EquipID], [RowVersion]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ForecastState_Latest' AND object_id = OBJECT_ID('dbo.[ACM_ForecastState]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ForecastState_Latest] ON dbo.[ACM_ForecastState] ([EquipID], [StateVersion]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'PK_ACM_FusionQualityReport' AND object_id = OBJECT_ID('dbo.[ACM_FusionQualityReport]'))
BEGIN
    CREATE CLUSTERED INDEX [PK_ACM_FusionQualityReport] ON dbo.[ACM_FusionQualityReport] ([RunID], [EquipID], [Detector]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'PK_ACM_HealthDistributionOverTime' AND object_id = OBJECT_ID('dbo.[ACM_HealthDistributionOverTime]'))
BEGIN
    CREATE CLUSTERED INDEX [PK_ACM_HealthDistributionOverTime] ON dbo.[ACM_HealthDistributionOverTime] ([RunID], [EquipID], [BucketStart]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_HealthForecast_Run' AND object_id = OBJECT_ID('dbo.[ACM_HealthForecast]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_HealthForecast_Run] ON dbo.[ACM_HealthForecast] ([RunID]) INCLUDE ([EquipID], [Timestamp], [ForecastHealth]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_HealthForecast_Time' AND object_id = OBJECT_ID('dbo.[ACM_HealthForecast]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_HealthForecast_Time] ON dbo.[ACM_HealthForecast] ([EquipID], [Timestamp]) INCLUDE ([ForecastHealth], [CiLower], [CiUpper]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_HealthForecast_SourceRun' AND object_id = OBJECT_ID('dbo.[ACM_HealthForecast_Continuous]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_HealthForecast_SourceRun] ON dbo.[ACM_HealthForecast_Continuous] ([EquipID], [SourceRunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_HealthForecast_TimeRange' AND object_id = OBJECT_ID('dbo.[ACM_HealthForecast_Continuous]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_HealthForecast_TimeRange] ON dbo.[ACM_HealthForecast_Continuous] ([EquipID], [Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_HealthTimeline_QualityFlag' AND object_id = OBJECT_ID('dbo.[ACM_HealthTimeline]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_HealthTimeline_QualityFlag] ON dbo.[ACM_HealthTimeline] ([EquipID], [QualityFlag], [Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_HistorianData_EquipTag' AND object_id = OBJECT_ID('dbo.[ACM_HistorianData]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_HistorianData_EquipTag] ON dbo.[ACM_HistorianData] ([EquipID], [TagName], [Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_HistorianData_Timestamp' AND object_id = OBJECT_ID('dbo.[ACM_HistorianData]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_HistorianData_Timestamp] ON dbo.[ACM_HistorianData] ([Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_OMR_Diagnostics_Equipment' AND object_id = OBJECT_ID('dbo.[ACM_OMR_Diagnostics]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_OMR_Diagnostics_Equipment] ON dbo.[ACM_OMR_Diagnostics] ([EquipID], [FitTimestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_OMR_Diagnostics_Run' AND object_id = OBJECT_ID('dbo.[ACM_OMR_Diagnostics]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_OMR_Diagnostics_Run] ON dbo.[ACM_OMR_Diagnostics] ([RunID], [EquipID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'PK_ACM_OMRContributionsLong' AND object_id = OBJECT_ID('dbo.[ACM_OMRContributionsLong]'))
BEGIN
    CREATE CLUSTERED INDEX [PK_ACM_OMRContributionsLong] ON dbo.[ACM_OMRContributionsLong] ([RunID], [EquipID], [Timestamp], [SensorName]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'PK_ACM_OMRTimeline' AND object_id = OBJECT_ID('dbo.[ACM_OMRTimeline]'))
BEGIN
    CREATE CLUSTERED INDEX [PK_ACM_OMRTimeline] ON dbo.[ACM_OMRTimeline] ([RunID], [EquipID], [Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_PCALoadings_EquipID_Component' AND object_id = OBJECT_ID('dbo.[ACM_PCA_Loadings]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_PCALoadings_EquipID_Component] ON dbo.[ACM_PCA_Loadings] ([EquipID], [ComponentNo]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_PCALoadings_RunID' AND object_id = OBJECT_ID('dbo.[ACM_PCA_Loadings]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_PCALoadings_RunID] ON dbo.[ACM_PCA_Loadings] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_PCAModels_EquipID' AND object_id = OBJECT_ID('dbo.[ACM_PCA_Models]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_PCAModels_EquipID] ON dbo.[ACM_PCA_Models] ([EquipID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_PCAModels_RunID' AND object_id = OBJECT_ID('dbo.[ACM_PCA_Models]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_PCAModels_RunID] ON dbo.[ACM_PCA_Models] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RefitRequests_EquipID_Ack' AND object_id = OBJECT_ID('dbo.[ACM_RefitRequests]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_RefitRequests_EquipID_Ack] ON dbo.[ACM_RefitRequests] ([EquipID], [Acknowledged]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RegimeState_EquipID_Version' AND object_id = OBJECT_ID('dbo.[ACM_RegimeState]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_RegimeState_EquipID_Version] ON dbo.[ACM_RegimeState] ([EquipID], [StateVersion]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'PK_ACM_RegimeStats' AND object_id = OBJECT_ID('dbo.[ACM_RegimeStats]'))
BEGIN
    CREATE CLUSTERED INDEX [PK_ACM_RegimeStats] ON dbo.[ACM_RegimeStats] ([RunID], [EquipID], [RegimeLabel]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_RUL_Latest' AND object_id = OBJECT_ID('dbo.[ACM_RUL]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_RUL_Latest] ON dbo.[ACM_RUL] ([EquipID], [CreatedAt]) INCLUDE ([RUL_Hours], [Confidence], [FailureTime]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunStats_EquipID' AND object_id = OBJECT_ID('dbo.[ACM_Run_Stats]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_RunStats_EquipID] ON dbo.[ACM_Run_Stats] ([EquipID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunStats_RunID' AND object_id = OBJECT_ID('dbo.[ACM_Run_Stats]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_RunStats_RunID] ON dbo.[ACM_Run_Stats] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunMetadata_EquipID' AND object_id = OBJECT_ID('dbo.[ACM_RunMetadata]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_RunMetadata_EquipID] ON dbo.[ACM_RunMetadata] ([EquipID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_RunMetadata_RunID' AND object_id = OBJECT_ID('dbo.[ACM_RunMetadata]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_RunMetadata_RunID] ON dbo.[ACM_RunMetadata] ([RunID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_Runs_EquipStarted' AND object_id = OBJECT_ID('dbo.[ACM_Runs]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_Runs_EquipStarted] ON dbo.[ACM_Runs] ([EquipID], [StartedAt]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_Runs_Status' AND object_id = OBJECT_ID('dbo.[ACM_Runs]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_Runs_Status] ON dbo.[ACM_Runs] ([EquipID], [HealthStatus]) WHERE ([HealthStatus] IN ('CAUTION', 'ALERT'));
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SchemaVersion_VersionNumber' AND object_id = OBJECT_ID('dbo.[ACM_SchemaVersion]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_SchemaVersion_VersionNumber] ON dbo.[ACM_SchemaVersion] ([VersionNumber]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_SensorForecast_EquipID_Timestamp' AND object_id = OBJECT_ID('dbo.[ACM_SensorForecast]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_SensorForecast_EquipID_Timestamp] ON dbo.[ACM_SensorForecast] ([EquipID], [Timestamp]) INCLUDE ([SensorName], [ForecastValue], [CiLower], [CiUpper]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ACM_SensorForecast_SensorName' AND object_id = OBJECT_ID('dbo.[ACM_SensorForecast]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ACM_SensorForecast_SensorName] ON dbo.[ACM_SensorForecast] ([SensorName], [EquipID], [Timestamp]) INCLUDE ([ForecastValue], [Method]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ASN_Equip_Sensor_Ts' AND object_id = OBJECT_ID('dbo.[ACM_SensorNormalized_TS]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ASN_Equip_Sensor_Ts] ON dbo.[ACM_SensorNormalized_TS] ([EquipID], [SensorName], [Timestamp]) INCLUDE ([ZScore], [NormValue], [AnomalyLevel], [EpisodeActive]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ASN_Equip_Ts' AND object_id = OBJECT_ID('dbo.[ACM_SensorNormalized_TS]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ASN_Equip_Ts] ON dbo.[ACM_SensorNormalized_TS] ([EquipID], [Timestamp]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_TagEquipmentMap_Equipment' AND object_id = OBJECT_ID('dbo.[ACM_TagEquipmentMap]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_TagEquipmentMap_Equipment] ON dbo.[ACM_TagEquipmentMap] ([EquipID], [IsActive]) INCLUDE ([TagName], [EquipmentName]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_TagEquipmentMap_TagName' AND object_id = OBJECT_ID('dbo.[ACM_TagEquipmentMap]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_TagEquipmentMap_TagName] ON dbo.[ACM_TagEquipmentMap] ([TagName], [IsActive]) INCLUDE ([EquipmentName], [EquipID]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ThresholdMetadata_Created' AND object_id = OBJECT_ID('dbo.[ACM_ThresholdMetadata]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ThresholdMetadata_Created] ON dbo.[ACM_ThresholdMetadata] ([CreatedAt]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ThresholdMetadata_Lookup' AND object_id = OBJECT_ID('dbo.[ACM_ThresholdMetadata]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ThresholdMetadata_Lookup] ON dbo.[ACM_ThresholdMetadata] ([EquipID], [RegimeID], [ThresholdType], [IsActive]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_FD_FAN_Data_TimeRange' AND object_id = OBJECT_ID('dbo.[FD_FAN_Data]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_FD_FAN_Data_TimeRange] ON dbo.[FD_FAN_Data] ([EntryDateTime]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_GAS_TURBINE_Data_TimeRange' AND object_id = OBJECT_ID('dbo.[GAS_TURBINE_Data]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_GAS_TURBINE_Data_TimeRange] ON dbo.[GAS_TURBINE_Data] ([EntryDateTime]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ModelRegistry_EquipID_Version' AND object_id = OBJECT_ID('dbo.[ModelRegistry]'))
BEGIN
    CREATE NONCLUSTERED INDEX [IX_ModelRegistry_EquipID_Version] ON dbo.[ModelRegistry] ([EquipID], [Version]) INCLUDE ([ModelType], [EntryDateTime], [ModelBytes]);
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UX_ModelRegistry_Current' AND object_id = OBJECT_ID('dbo.[ModelRegistry]'))
BEGIN
    CREATE UNIQUE NONCLUSTERED INDEX [UX_ModelRegistry_Current] ON dbo.[ModelRegistry] ([ModelType], [EquipID], [EntryDateTime]);
END
GO

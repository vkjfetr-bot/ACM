USE [ACM];
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_AlertAge_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_AlertAge]'))
BEGIN
    ALTER TABLE dbo.[ACM_AlertAge] WITH CHECK ADD CONSTRAINT [FK_ACM_AlertAge_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_AlertAge] CHECK CONSTRAINT [FK_ACM_AlertAge_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_CalibrationSummary_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_CalibrationSummary]'))
BEGIN
    ALTER TABLE dbo.[ACM_CalibrationSummary] WITH CHECK ADD CONSTRAINT [FK_ACM_CalibrationSummary_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_CalibrationSummary] CHECK CONSTRAINT [FK_ACM_CalibrationSummary_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_Config_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_Config]'))
BEGIN
    ALTER TABLE dbo.[ACM_Config] WITH CHECK ADD CONSTRAINT [FK_ACM_Config_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_Config] CHECK CONSTRAINT [FK_ACM_Config_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_ContributionCurrent_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_ContributionCurrent]'))
BEGIN
    ALTER TABLE dbo.[ACM_ContributionCurrent] WITH CHECK ADD CONSTRAINT [FK_ACM_ContributionCurrent_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_ContributionCurrent] CHECK CONSTRAINT [FK_ACM_ContributionCurrent_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_ContributionTimeline_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_ContributionTimeline]'))
BEGIN
    ALTER TABLE dbo.[ACM_ContributionTimeline] WITH CHECK ADD CONSTRAINT [FK_ACM_ContributionTimeline_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_ContributionTimeline] CHECK CONSTRAINT [FK_ACM_ContributionTimeline_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_DataQuality_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_DataQuality]'))
BEGIN
    ALTER TABLE dbo.[ACM_DataQuality] WITH CHECK ADD CONSTRAINT [FK_ACM_DataQuality_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_DataQuality] CHECK CONSTRAINT [FK_ACM_DataQuality_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_DefectSummary_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_DefectSummary]'))
BEGIN
    ALTER TABLE dbo.[ACM_DefectSummary] WITH CHECK ADD CONSTRAINT [FK_ACM_DefectSummary_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_DefectSummary] CHECK CONSTRAINT [FK_ACM_DefectSummary_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_DefectTimeline_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_DefectTimeline]'))
BEGIN
    ALTER TABLE dbo.[ACM_DefectTimeline] WITH CHECK ADD CONSTRAINT [FK_ACM_DefectTimeline_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_DefectTimeline] CHECK CONSTRAINT [FK_ACM_DefectTimeline_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_DetectorCorrelation_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_DetectorCorrelation]'))
BEGIN
    ALTER TABLE dbo.[ACM_DetectorCorrelation] WITH CHECK ADD CONSTRAINT [FK_ACM_DetectorCorrelation_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_DetectorCorrelation] CHECK CONSTRAINT [FK_ACM_DetectorCorrelation_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_DetectorForecast_TS_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_DetectorForecast_TS]'))
BEGIN
    ALTER TABLE dbo.[ACM_DetectorForecast_TS] WITH CHECK ADD CONSTRAINT [FK_ACM_DetectorForecast_TS_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_DetectorForecast_TS] CHECK CONSTRAINT [FK_ACM_DetectorForecast_TS_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_DriftEvents_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_DriftEvents]'))
BEGIN
    ALTER TABLE dbo.[ACM_DriftEvents] WITH CHECK ADD CONSTRAINT [FK_ACM_DriftEvents_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_DriftEvents] CHECK CONSTRAINT [FK_ACM_DriftEvents_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_DriftSeries_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_DriftSeries]'))
BEGIN
    ALTER TABLE dbo.[ACM_DriftSeries] WITH CHECK ADD CONSTRAINT [FK_ACM_DriftSeries_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_DriftSeries] CHECK CONSTRAINT [FK_ACM_DriftSeries_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_EpisodeMetrics_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_EpisodeMetrics]'))
BEGIN
    ALTER TABLE dbo.[ACM_EpisodeMetrics] WITH CHECK ADD CONSTRAINT [FK_ACM_EpisodeMetrics_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_EpisodeMetrics] CHECK CONSTRAINT [FK_ACM_EpisodeMetrics_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_Episodes_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_Episodes]'))
BEGIN
    ALTER TABLE dbo.[ACM_Episodes] WITH CHECK ADD CONSTRAINT [FK_ACM_Episodes_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_Episodes] CHECK CONSTRAINT [FK_ACM_Episodes_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_HealthHistogram_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_HealthHistogram]'))
BEGIN
    ALTER TABLE dbo.[ACM_HealthHistogram] WITH CHECK ADD CONSTRAINT [FK_ACM_HealthHistogram_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_HealthHistogram] CHECK CONSTRAINT [FK_ACM_HealthHistogram_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_HealthTimeline_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_HealthTimeline]'))
BEGIN
    ALTER TABLE dbo.[ACM_HealthTimeline] WITH CHECK ADD CONSTRAINT [FK_ACM_HealthTimeline_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_HealthTimeline] CHECK CONSTRAINT [FK_ACM_HealthTimeline_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_HealthZoneByPeriod_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_HealthZoneByPeriod]'))
BEGIN
    ALTER TABLE dbo.[ACM_HealthZoneByPeriod] WITH CHECK ADD CONSTRAINT [FK_ACM_HealthZoneByPeriod_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_HealthZoneByPeriod] CHECK CONSTRAINT [FK_ACM_HealthZoneByPeriod_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_RegimeDwellStats_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_RegimeDwellStats]'))
BEGIN
    ALTER TABLE dbo.[ACM_RegimeDwellStats] WITH CHECK ADD CONSTRAINT [FK_ACM_RegimeDwellStats_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_RegimeDwellStats] CHECK CONSTRAINT [FK_ACM_RegimeDwellStats_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_RegimeOccupancy_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_RegimeOccupancy]'))
BEGIN
    ALTER TABLE dbo.[ACM_RegimeOccupancy] WITH CHECK ADD CONSTRAINT [FK_ACM_RegimeOccupancy_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_RegimeOccupancy] CHECK CONSTRAINT [FK_ACM_RegimeOccupancy_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_RegimeStability_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_RegimeStability]'))
BEGIN
    ALTER TABLE dbo.[ACM_RegimeStability] WITH CHECK ADD CONSTRAINT [FK_ACM_RegimeStability_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_RegimeStability] CHECK CONSTRAINT [FK_ACM_RegimeStability_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_RegimeTimeline_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_RegimeTimeline]'))
BEGIN
    ALTER TABLE dbo.[ACM_RegimeTimeline] WITH CHECK ADD CONSTRAINT [FK_ACM_RegimeTimeline_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_RegimeTimeline] CHECK CONSTRAINT [FK_ACM_RegimeTimeline_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_RegimeTransitions_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_RegimeTransitions]'))
BEGIN
    ALTER TABLE dbo.[ACM_RegimeTransitions] WITH CHECK ADD CONSTRAINT [FK_ACM_RegimeTransitions_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_RegimeTransitions] CHECK CONSTRAINT [FK_ACM_RegimeTransitions_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_RunMetrics_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_RunMetrics]'))
BEGIN
    ALTER TABLE dbo.[ACM_RunMetrics] WITH CHECK ADD CONSTRAINT [FK_ACM_RunMetrics_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_RunMetrics] CHECK CONSTRAINT [FK_ACM_RunMetrics_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_Runs_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_Runs]'))
BEGIN
    ALTER TABLE dbo.[ACM_Runs] WITH CHECK ADD CONSTRAINT [FK_ACM_Runs_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_Runs] CHECK CONSTRAINT [FK_ACM_Runs_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_Scores_Wide_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_Scores_Wide]'))
BEGIN
    ALTER TABLE dbo.[ACM_Scores_Wide] WITH CHECK ADD CONSTRAINT [FK_ACM_Scores_Wide_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_Scores_Wide] CHECK CONSTRAINT [FK_ACM_Scores_Wide_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_SensorAnomalyByPeriod_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_SensorAnomalyByPeriod]'))
BEGIN
    ALTER TABLE dbo.[ACM_SensorAnomalyByPeriod] WITH CHECK ADD CONSTRAINT [FK_ACM_SensorAnomalyByPeriod_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_SensorAnomalyByPeriod] CHECK CONSTRAINT [FK_ACM_SensorAnomalyByPeriod_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_SensorDefects_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_SensorDefects]'))
BEGIN
    ALTER TABLE dbo.[ACM_SensorDefects] WITH CHECK ADD CONSTRAINT [FK_ACM_SensorDefects_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_SensorDefects] CHECK CONSTRAINT [FK_ACM_SensorDefects_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_SensorForecast_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_SensorForecast]'))
BEGIN
    ALTER TABLE dbo.[ACM_SensorForecast] WITH CHECK ADD CONSTRAINT [FK_ACM_SensorForecast_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_SensorForecast] CHECK CONSTRAINT [FK_ACM_SensorForecast_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_SensorForecast_Runs' AND parent_object_id = OBJECT_ID('dbo.[ACM_SensorForecast]'))
BEGIN
    ALTER TABLE dbo.[ACM_SensorForecast] WITH CHECK ADD CONSTRAINT [FK_ACM_SensorForecast_Runs] FOREIGN KEY ([RunID]) REFERENCES dbo.[ACM_Runs] ([RunID]) ON DELETE CASCADE;
    ALTER TABLE dbo.[ACM_SensorForecast] CHECK CONSTRAINT [FK_ACM_SensorForecast_Runs];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_SensorHotspotTimeline_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_SensorHotspotTimeline]'))
BEGIN
    ALTER TABLE dbo.[ACM_SensorHotspotTimeline] WITH CHECK ADD CONSTRAINT [FK_ACM_SensorHotspotTimeline_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_SensorHotspotTimeline] CHECK CONSTRAINT [FK_ACM_SensorHotspotTimeline_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_SensorHotspots_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_SensorHotspots]'))
BEGIN
    ALTER TABLE dbo.[ACM_SensorHotspots] WITH CHECK ADD CONSTRAINT [FK_ACM_SensorHotspots_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_SensorHotspots] CHECK CONSTRAINT [FK_ACM_SensorHotspots_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_SensorRanking_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_SensorRanking]'))
BEGIN
    ALTER TABLE dbo.[ACM_SensorRanking] WITH CHECK ADD CONSTRAINT [FK_ACM_SensorRanking_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_SensorRanking] CHECK CONSTRAINT [FK_ACM_SensorRanking_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_SinceWhen_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_SinceWhen]'))
BEGIN
    ALTER TABLE dbo.[ACM_SinceWhen] WITH CHECK ADD CONSTRAINT [FK_ACM_SinceWhen_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_SinceWhen] CHECK CONSTRAINT [FK_ACM_SinceWhen_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ACM_ThresholdCrossings_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_ThresholdCrossings]'))
BEGIN
    ALTER TABLE dbo.[ACM_ThresholdCrossings] WITH CHECK ADD CONSTRAINT [FK_ACM_ThresholdCrossings_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_ThresholdCrossings] CHECK CONSTRAINT [FK_ACM_ThresholdCrossings_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ColdstartState_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_ColdstartState]'))
BEGIN
    ALTER TABLE dbo.[ACM_ColdstartState] WITH CHECK ADD CONSTRAINT [FK_ColdstartState_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_ColdstartState] CHECK CONSTRAINT [FK_ColdstartState_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_EpisodesQC_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_EpisodesQC]'))
BEGIN
    ALTER TABLE dbo.[ACM_EpisodesQC] WITH CHECK ADD CONSTRAINT [FK_EpisodesQC_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_EpisodesQC] CHECK CONSTRAINT [FK_EpisodesQC_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ModelRegistry_Equip' AND parent_object_id = OBJECT_ID('dbo.[ModelRegistry]'))
BEGIN
    ALTER TABLE dbo.[ModelRegistry] WITH CHECK ADD CONSTRAINT [FK_ModelRegistry_Equip] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ModelRegistry] CHECK CONSTRAINT [FK_ModelRegistry_Equip];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PCALoadings_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_PCA_Loadings]'))
BEGIN
    ALTER TABLE dbo.[ACM_PCA_Loadings] WITH CHECK ADD CONSTRAINT [FK_PCALoadings_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_PCA_Loadings] CHECK CONSTRAINT [FK_PCALoadings_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PCAModels_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_PCA_Models]'))
BEGIN
    ALTER TABLE dbo.[ACM_PCA_Models] WITH CHECK ADD CONSTRAINT [FK_PCAModels_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_PCA_Models] CHECK CONSTRAINT [FK_PCAModels_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_RULLearning_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_RUL_LearningState]'))
BEGIN
    ALTER TABLE dbo.[ACM_RUL_LearningState] WITH CHECK ADD CONSTRAINT [FK_RULLearning_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_RUL_LearningState] CHECK CONSTRAINT [FK_RULLearning_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_RunStats_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_Run_Stats]'))
BEGIN
    ALTER TABLE dbo.[ACM_Run_Stats] WITH CHECK ADD CONSTRAINT [FK_RunStats_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_Run_Stats] CHECK CONSTRAINT [FK_RunStats_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_TagEquipmentMap_Equipment' AND parent_object_id = OBJECT_ID('dbo.[ACM_TagEquipmentMap]'))
BEGIN
    ALTER TABLE dbo.[ACM_TagEquipmentMap] WITH CHECK ADD CONSTRAINT [FK_TagEquipmentMap_Equipment] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_TagEquipmentMap] CHECK CONSTRAINT [FK_TagEquipmentMap_Equipment];
END
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK__ACM_Thres__Equip__10AB74EC' AND parent_object_id = OBJECT_ID('dbo.[ACM_ThresholdMetadata]'))
BEGIN
    ALTER TABLE dbo.[ACM_ThresholdMetadata] WITH CHECK ADD CONSTRAINT [FK__ACM_Thres__Equip__10AB74EC] FOREIGN KEY ([EquipID]) REFERENCES dbo.[Equipment] ([EquipID]);
    ALTER TABLE dbo.[ACM_ThresholdMetadata] CHECK CONSTRAINT [FK__ACM_Thres__Equip__10AB74EC];
END
GO

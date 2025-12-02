-- =============================================
-- ACM Analytics Tables (Phase 2 - All Output Tables)
-- =============================================
-- Comprehensive tables for all ACM analytics outputs
-- Designed for multi-instance concurrent runs
-- Each table has RunID + EquipID for filtering
-- =============================================

USE [ACM];
GO

-- 1. Health Timeline (time-series health index)
IF OBJECT_ID('dbo.ACM_HealthTimeline','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_HealthTimeline (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Timestamp          datetime2(3) NOT NULL,
    FusedZ             float NULL,
    HealthIndex        float NULL,
    Zone               nvarchar(16) NULL,  -- GOOD, WATCH, ALERT
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_HealthTimeline_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_HealthTimeline PRIMARY KEY CLUSTERED (RunID, Timestamp)
);
CREATE NONCLUSTERED INDEX IX_HealthTimeline_EquipID_Time ON dbo.ACM_HealthTimeline(EquipID, Timestamp DESC);
END
GO

-- 2. Regime Timeline
IF OBJECT_ID('dbo.ACM_RegimeTimeline','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_RegimeTimeline (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Timestamp          datetime2(3) NOT NULL,
    RegimeLabel        int NULL,
    RegimeState        nvarchar(32) NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeTimeline_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_RegimeTimeline PRIMARY KEY CLUSTERED (RunID, Timestamp)
);
CREATE NONCLUSTERED INDEX IX_RegimeTimeline_EquipID_Time ON dbo.ACM_RegimeTimeline(EquipID, Timestamp DESC);
END
GO

-- 3. Detector Correlation
IF OBJECT_ID('dbo.ACM_DetectorCorrelation','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_DetectorCorrelation (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    DetectorA          nvarchar(64) NOT NULL,
    DetectorB          nvarchar(64) NOT NULL,
    PearsonR           float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_DetectorCorrelation_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_DetectorCorrelation PRIMARY KEY CLUSTERED (RunID, DetectorA, DetectorB)
);
CREATE NONCLUSTERED INDEX IX_DetectorCorrelation_EquipID ON dbo.ACM_DetectorCorrelation(EquipID, RunID);
END
GO

-- 4. Calibration Summary
IF OBJECT_ID('dbo.ACM_CalibrationSummary','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_CalibrationSummary (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Detector           nvarchar(64) NOT NULL,
    Mean               float NULL,
    Std                float NULL,
    P95                float NULL,
    P99                float NULL,
    ClipZ              float NULL,
    ClipPct            float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_CalibrationSummary_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_CalibrationSummary PRIMARY KEY CLUSTERED (RunID, Detector)
);
CREATE NONCLUSTERED INDEX IX_CalibrationSummary_EquipID ON dbo.ACM_CalibrationSummary(EquipID, RunID);
END
GO

-- 5. Regime Transition Matrix
IF OBJECT_ID('dbo.ACM_RegimeTransitions','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_RegimeTransitions (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    FromLabel          int NOT NULL,
    ToLabel            int NOT NULL,
    TransitionCount    int NULL,
    Probability        float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeTransitions_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_RegimeTransitions PRIMARY KEY CLUSTERED (RunID, FromLabel, ToLabel)
);
CREATE NONCLUSTERED INDEX IX_RegimeTransitions_EquipID ON dbo.ACM_RegimeTransitions(EquipID, RunID);
END
GO

-- 6. Regime Dwell Stats
IF OBJECT_ID('dbo.ACM_RegimeDwellStats','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_RegimeDwellStats (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    RegimeLabel        int NOT NULL,
    Runs               int NULL,
    MeanSeconds        float NULL,
    MedianSeconds      float NULL,
    MinSeconds         float NULL,
    MaxSeconds         float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeDwellStats_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_RegimeDwellStats PRIMARY KEY CLUSTERED (RunID, RegimeLabel)
);
CREATE NONCLUSTERED INDEX IX_RegimeDwellStats_EquipID ON dbo.ACM_RegimeDwellStats(EquipID, RunID);
END
GO

-- 7. Drift Events
IF OBJECT_ID('dbo.ACM_DriftEvents','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_DriftEvents (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Timestamp          datetime2(3) NOT NULL,
    DriftValue         float NULL,
    SegmentStart       datetime2(3) NULL,
    SegmentEnd         datetime2(3) NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_DriftEvents_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_DriftEvents PRIMARY KEY CLUSTERED (RunID, Timestamp)
);
CREATE NONCLUSTERED INDEX IX_DriftEvents_EquipID_Time ON dbo.ACM_DriftEvents(EquipID, Timestamp DESC);
END
GO

-- 8. Threshold Crossings
IF OBJECT_ID('dbo.ACM_ThresholdCrossings','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_ThresholdCrossings (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Timestamp          datetime2(3) NOT NULL,
    Detector           nvarchar(64) NOT NULL,
    ZScore             float NULL,
    Threshold          float NULL,
    Direction          nvarchar(16) NULL,  -- UP, DOWN
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_ThresholdCrossings_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_ThresholdCrossings PRIMARY KEY CLUSTERED (RunID, Timestamp, Detector)
);
CREATE NONCLUSTERED INDEX IX_ThresholdCrossings_EquipID_Time ON dbo.ACM_ThresholdCrossings(EquipID, Timestamp DESC);
END
GO

-- 9. Regime Occupancy
IF OBJECT_ID('dbo.ACM_RegimeOccupancy','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_RegimeOccupancy (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    RegimeLabel        int NOT NULL,
    RecordCount        int NULL,
    Percentage         float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeOccupancy_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_RegimeOccupancy PRIMARY KEY CLUSTERED (RunID, RegimeLabel)
);
CREATE NONCLUSTERED INDEX IX_RegimeOccupancy_EquipID ON dbo.ACM_RegimeOccupancy(EquipID, RunID);
END
GO

-- 10. Episode Metrics Summary (single row per run)
IF OBJECT_ID('dbo.ACM_EpisodeMetrics','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_EpisodeMetrics (
    RunID                    uniqueidentifier NOT NULL PRIMARY KEY,
    EquipID                  int NOT NULL,
    TotalEpisodes            int NULL,
    TotalDurationHours       float NULL,
    AvgDurationHours         float NULL,
    MedianDurationHours      float NULL,
    MaxDurationHours         float NULL,
    MinDurationHours         float NULL,
    RatePerDay               float NULL,
    MeanInterarrivalHours    float NULL,
    CreatedAt                datetime2(3) NOT NULL CONSTRAINT DF_ACM_EpisodeMetrics_CreatedAt DEFAULT (SYSUTCDATETIME())
);
CREATE NONCLUSTERED INDEX IX_EpisodeMetrics_EquipID ON dbo.ACM_EpisodeMetrics(EquipID, RunID);
END
GO

-- 11. Culprit History (detector frequency across episodes)
IF OBJECT_ID('dbo.ACM_CulpritHistory','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_CulpritHistory (
    RunID                uniqueidentifier NOT NULL,
    EquipID              int NOT NULL,
    StartTimestamp       datetime2(3) NOT NULL,
    EndTimestamp         datetime2(3) NULL,
    DurationHours        float NULL,
    PrimaryDetector      nvarchar(256) NOT NULL,
    WeightedContribution float NULL,
    LeadMeanZ            float NULL,
    DuringMeanZ          float NULL,
    LagMeanZ             float NULL,
    CreatedAt            datetime2(3) NOT NULL CONSTRAINT DF_ACM_CulpritHistory_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_CulpritHistory PRIMARY KEY CLUSTERED (RunID, EquipID, StartTimestamp)
);
CREATE NONCLUSTERED INDEX IX_CulpritHistory_EquipID ON dbo.ACM_CulpritHistory(EquipID, RunID);
END
GO

PRINT 'ACM Analytics tables created successfully';
GO

-- 12. Contribution Timeline
IF OBJECT_ID('dbo.ACM_ContributionTimeline','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_ContributionTimeline (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Timestamp          datetime2(3) NOT NULL,
    DetectorType       nvarchar(50) NOT NULL,
    ContributionPct    float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_ContributionTimeline_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_ContributionTimeline PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp, DetectorType)
);
CREATE NONCLUSTERED INDEX IX_ContributionTimeline_EquipID_Time ON dbo.ACM_ContributionTimeline(EquipID, Timestamp DESC);
END
GO

-- 13. Drift Series
IF OBJECT_ID('dbo.ACM_DriftSeries','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_DriftSeries (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Timestamp          datetime2(3) NOT NULL,
    DriftValue         float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_DriftSeries_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_DriftSeries PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp)
);
CREATE NONCLUSTERED INDEX IX_DriftSeries_EquipID_Time ON dbo.ACM_DriftSeries(EquipID, Timestamp DESC);
END
GO

-- 14. Defect Timeline
IF OBJECT_ID('dbo.ACM_DefectTimeline','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_DefectTimeline (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Timestamp          datetime2(3) NOT NULL,
    EventType          nvarchar(50) NOT NULL,
    FromZone           nvarchar(50) NULL,
    ToZone             nvarchar(50) NULL,
    HealthZone         nvarchar(50) NOT NULL,
    HealthAtEvent      float NULL,
    HealthIndex        float NULL,
    FusedZ             float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_DefectTimeline_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_DefectTimeline PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp)
);
CREATE NONCLUSTERED INDEX IX_DefectTimeline_EquipID_Time ON dbo.ACM_DefectTimeline(EquipID, Timestamp DESC);
END
GO

-- 15. PCA Metrics (Variance Explained, Component Count)
IF OBJECT_ID('dbo.ACM_PCA_Metrics','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_PCA_Metrics (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    ComponentName      nvarchar(50) NOT NULL,  -- 'PC1', 'PC2', ..., 'Total'
    MetricType         nvarchar(50) NOT NULL,  -- 'VarianceRatio', 'CumulativeVariance', 'ComponentCount'
    Value              float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_PCA_Metrics_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_PCA_Metrics PRIMARY KEY CLUSTERED (RunID, EquipID, ComponentName, MetricType)
);
CREATE NONCLUSTERED INDEX IX_PCA_Metrics_EquipID ON dbo.ACM_PCA_Metrics(EquipID, RunID);
END
GO

-- 16. Sensor Hotspot Timeline
IF OBJECT_ID('dbo.ACM_SensorHotspotTimeline','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_SensorHotspotTimeline (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Timestamp          datetime2(3) NOT NULL,
    SensorName         nvarchar(255) NOT NULL,
    Rank               int NOT NULL,
    AbsZ               float NULL,
    SignedZ            float NULL,
    Value              float NULL,
    Level              nvarchar(50) NOT NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_SensorHotspotTimeline_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_SensorHotspotTimeline PRIMARY KEY CLUSTERED (RunID, EquipID, Timestamp, SensorName)
);
CREATE NONCLUSTERED INDEX IX_SensorHotspotTimeline_EquipID_Time ON dbo.ACM_SensorHotspotTimeline(EquipID, Timestamp DESC);
END
GO

-- 16. Contribution Current
IF OBJECT_ID('dbo.ACM_ContributionCurrent','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_ContributionCurrent (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    DetectorType       nvarchar(50) NOT NULL,
    ContributionPct    float NULL,
    ZScore             float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_ContributionCurrent_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_ContributionCurrent PRIMARY KEY CLUSTERED (RunID, EquipID, DetectorType)
);
CREATE NONCLUSTERED INDEX IX_ContributionCurrent_EquipID ON dbo.ACM_ContributionCurrent(EquipID, RunID);
END
GO

-- 17. Sensor Ranking
IF OBJECT_ID('dbo.ACM_SensorRanking','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_SensorRanking (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    RankPosition       int NOT NULL,
    DetectorType       nvarchar(50) NOT NULL,
    ContributionPct    float NULL,
    ZScore             float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_SensorRanking_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_SensorRanking PRIMARY KEY CLUSTERED (RunID, EquipID, RankPosition)
);
CREATE NONCLUSTERED INDEX IX_SensorRanking_EquipID ON dbo.ACM_SensorRanking(EquipID, RunID);
END
GO

-- 18. Sensor Hotspots
IF OBJECT_ID('dbo.ACM_SensorHotspots','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_SensorHotspots (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    SensorName         nvarchar(255) NOT NULL,
    MaxTimestamp       datetime2(3) NOT NULL,
    LatestTimestamp    datetime2(3) NOT NULL,
    MaxAbsZ            float NULL,
    MaxSignedZ         float NULL,
    LatestAbsZ         float NULL,
    LatestSignedZ      float NULL,
    ValueAtPeak        float NULL,
    LatestValue        float NULL,
    TrainMean          float NULL,
    TrainStd           float NULL,
    AboveWarnCount     int NULL,
    AboveAlertCount    int NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_SensorHotspots_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_SensorHotspots PRIMARY KEY CLUSTERED (RunID, EquipID, SensorName)
);
CREATE NONCLUSTERED INDEX IX_SensorHotspots_EquipID ON dbo.ACM_SensorHotspots(EquipID, RunID);
END
GO

-- 19. Since When
IF OBJECT_ID('dbo.ACM_SinceWhen','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_SinceWhen (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    AlertZone          nvarchar(50) NOT NULL,
    DurationHours      float NULL,
    StartTimestamp     datetime2(3) NOT NULL,
    RecordCount        int NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_SinceWhen_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_SinceWhen PRIMARY KEY CLUSTERED (RunID, EquipID, AlertZone)
);
CREATE NONCLUSTERED INDEX IX_SinceWhen_EquipID ON dbo.ACM_SinceWhen(EquipID, RunID);
END
GO

-- 20. Episodes
IF OBJECT_ID('dbo.ACM_Episodes','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_Episodes (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    EpisodeCount       int NULL,
    MedianDurationMinutes float NULL,
    CoveragePct        float NULL,
    TimeInAlertPct     float NULL,
    MaxFusedZ          float NULL,
    AvgFusedZ          float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_Episodes_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_Episodes PRIMARY KEY CLUSTERED (RunID, EquipID)
);
CREATE NONCLUSTERED INDEX IX_Episodes_EquipID ON dbo.ACM_Episodes(EquipID, RunID);
END
GO

-- 21. Alert Age
IF OBJECT_ID('dbo.ACM_AlertAge','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_AlertAge (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    AlertZone          nvarchar(50) NOT NULL,
    StartTimestamp     datetime2(3) NOT NULL,
    DurationHours      float NULL,
    RecordCount        int NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_AlertAge_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_AlertAge PRIMARY KEY CLUSTERED (RunID, EquipID, AlertZone)
);
CREATE NONCLUSTERED INDEX IX_AlertAge_EquipID ON dbo.ACM_AlertAge(EquipID, RunID);
END
GO

-- 22. Regime Stability
IF OBJECT_ID('dbo.ACM_RegimeStability','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_RegimeStability (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    MetricName         nvarchar(100) NOT NULL,
    MetricValue        float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_RegimeStability_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_RegimeStability PRIMARY KEY CLUSTERED (RunID, EquipID, MetricName)
);
CREATE NONCLUSTERED INDEX IX_RegimeStability_EquipID ON dbo.ACM_RegimeStability(EquipID, RunID);
END
GO

-- 23. Defect Summary
IF OBJECT_ID('dbo.ACM_DefectSummary','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_DefectSummary (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Status             nvarchar(50) NOT NULL,
    Severity           nvarchar(50) NOT NULL,
    CurrentHealth      float NULL,
    AvgHealth          float NULL,
    MinHealth          float NULL,
    EpisodeCount       int NULL,
    WorstSensor        nvarchar(255) NULL,
    GoodCount          int NULL,
    WatchCount         int NULL,
    AlertCount         int NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_DefectSummary_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_DefectSummary PRIMARY KEY CLUSTERED (RunID, EquipID)
);
CREATE NONCLUSTERED INDEX IX_DefectSummary_EquipID ON dbo.ACM_DefectSummary(EquipID, RunID);
END
GO

-- 24. Sensor Defects
IF OBJECT_ID('dbo.ACM_SensorDefects','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_SensorDefects (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    DetectorType       nvarchar(50) NOT NULL,
    DetectorFamily     nvarchar(50) NULL,
    Severity           nvarchar(50) NULL,
    ViolationCount     int NULL,
    ViolationPct       float NULL,
    MaxZ               float NULL,
    AvgZ               float NULL,
    CurrentZ           float NULL,
    ActiveDefect       nvarchar(10) NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_SensorDefects_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_SensorDefects PRIMARY KEY CLUSTERED (RunID, EquipID, DetectorType)
);
CREATE NONCLUSTERED INDEX IX_SensorDefects_EquipID ON dbo.ACM_SensorDefects(EquipID, RunID);
END
GO

-- 25. Health Zone By Period
IF OBJECT_ID('dbo.ACM_HealthZoneByPeriod','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_HealthZoneByPeriod (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    PeriodStart        datetime2(3) NOT NULL,
    PeriodType         nvarchar(20) NOT NULL,
    HealthZone         nvarchar(50) NOT NULL,
    ZonePct            float NULL,
    ZoneCount          int NULL,
    TotalPoints        int NULL,
    Date               date NOT NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_HealthZoneByPeriod_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_HealthZoneByPeriod PRIMARY KEY CLUSTERED (RunID, EquipID, PeriodStart, HealthZone)
);
CREATE NONCLUSTERED INDEX IX_HealthZoneByPeriod_EquipID ON dbo.ACM_HealthZoneByPeriod(EquipID, RunID);
END
GO

-- 26. Sensor Anomaly By Period
IF OBJECT_ID('dbo.ACM_SensorAnomalyByPeriod','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_SensorAnomalyByPeriod (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    Date               date NOT NULL,
    PeriodStart        datetime2(3) NOT NULL,
    PeriodType         nvarchar(20) NOT NULL,
    PeriodSeconds      float NULL,
    DetectorType       nvarchar(50) NOT NULL,
    AnomalyRatePct     float NULL,
    MaxZ               float NULL,
    AvgZ               float NULL,
    Points             int NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_SensorAnomalyByPeriod_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_SensorAnomalyByPeriod PRIMARY KEY CLUSTERED (RunID, EquipID, Date, DetectorType)
);
CREATE NONCLUSTERED INDEX IX_SensorAnomalyByPeriod_EquipID ON dbo.ACM_SensorAnomalyByPeriod(EquipID, RunID);
END
GO

-- 27. Health Histogram
IF OBJECT_ID('dbo.ACM_HealthHistogram','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_HealthHistogram (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    HealthBin          nvarchar(50) NOT NULL,
    RecordCount        int NULL,
    Percentage         float NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_HealthHistogram_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_HealthHistogram PRIMARY KEY CLUSTERED (RunID, EquipID, HealthBin)
);
CREATE NONCLUSTERED INDEX IX_HealthHistogram_EquipID ON dbo.ACM_HealthHistogram(EquipID, RunID);
END
GO

-- 28. Data Quality
IF OBJECT_ID('dbo.ACM_DataQuality','U') IS NULL
BEGIN
CREATE TABLE dbo.ACM_DataQuality (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    CheckName          nvarchar(100) NOT NULL,
    sensor             nvarchar(255) NOT NULL,
    CheckResult        nvarchar(50) NOT NULL,
    train_count        int NULL,
    train_nulls        int NULL,
    train_null_pct     float NULL,
    train_std          float NULL,
    train_longest_gap  int NULL,
    train_flatline_span int NULL,
    train_min_ts       datetime2(3) NULL,
    train_max_ts       datetime2(3) NULL,
    score_count        int NULL,
    score_nulls        int NULL,
    score_null_pct     float NULL,
    score_std          float NULL,
    score_longest_gap  int NULL,
    score_flatline_span int NULL,
    score_min_ts       datetime2(3) NULL,
    score_max_ts       datetime2(3) NULL,
    interp_method      nvarchar(50) NULL,
    sampling_secs      float NULL,
    notes              nvarchar(max) NULL,
    CreatedAt          datetime2(3) NOT NULL CONSTRAINT DF_ACM_DataQuality_CreatedAt DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_DataQuality PRIMARY KEY CLUSTERED (RunID, EquipID, CheckName, sensor)
);
CREATE NONCLUSTERED INDEX IX_DataQuality_EquipID ON dbo.ACM_DataQuality(EquipID, RunID);
END
GO

-- Verification
SELECT 
    t.name AS TableName,
    (SELECT COUNT(*) FROM sys.columns c WHERE c.object_id = t.object_id) AS ColumnCount
FROM sys.tables t
WHERE t.name LIKE 'ACM[_]%'
ORDER BY t.name;
GO

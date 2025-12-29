# ACM Output Requirements - Functionality-Based Analysis

**Date:** December 25, 2024  
**Approach:** Design tables based on ACM's core functionality and user needs, NOT based on existing dashboards

---

## Executive Summary

This document defines what ACM should output based on **what ACM does** and **what users need to see**, not based on what existing dashboards happen to query. Dashboards will be designed later to visualize this data.

### Design Principles

1. **Functionality-First**: Tables support ACM's core capabilities
2. **User-Centric**: Enable meaningful questions from operations, maintenance, and engineering teams
3. **Completeness**: Provide full visibility into ACM's analytical processes
4. **Actionability**: Every table enables specific decisions or actions
5. **Efficiency**: Avoid redundancy while maintaining analytical depth

---

## Part 1: What ACM Does

ACM is a **multi-detector predictive maintenance system** with these core capabilities:

### 1. Health Monitoring
- 6 analytical detectors (AR1, PCA-SPE, PCA-T², IForest, GMM, OMR)
- Each answers "what's wrong?" from different perspectives
- Fusion combines detectors into overall health score
- Health zones (GOOD, CAUTION, ALERT, CRITICAL)

### 2. Anomaly Detection
- Episode detection (continuous anomaly periods)
- Change-point detection
- Severity assessment
- Sensor attribution (which sensors caused it)

### 3. Failure Prediction
- Remaining Useful Life (RUL) estimation
- Health trajectory forecasting
- Failure probability curves
- Hazard rate analysis
- Monte Carlo confidence bounds

### 4. Operating Context
- Regime detection (normal operating modes)
- Regime transitions
- Behavior pattern recognition

### 5. Drift Detection
- Concept drift monitoring
- Behavior change events
- CUSUM-based tracking

### 6. Root Cause Analysis
- Sensor contribution tracking
- Top culprit identification
- Per-episode diagnostics
- Feature importance

### 7. Adaptive Learning
- Auto-tuning thresholds
- Model retraining triggers
- Configuration adaptation
- Continuous learning

### 8. Data & Model Quality
- Data quality assessment
- Model performance tracking
- Validation metrics
- Quality gates

---

## Part 2: What Users Need to See

### Operations Team (Real-Time Monitoring)
**Questions they ask:**
- Is my equipment healthy right now?
- What's currently wrong?
- Which assets need attention?
- Are there active alerts?
- What's the trend - improving or degrading?

**Enables:**
- Real-time monitoring
- Alert response
- Operational decisions

### Maintenance Team (Planning & Execution)
**Questions they ask:**
- When will this fail?
- Which components should I inspect?
- What's the failure risk this month?
- Has this failure mode occurred before?
- Which sensors are consistently problematic?

**Enables:**
- Maintenance scheduling
- Parts inventory
- Inspection planning
- Preventive actions

### Engineering Team (System Administration)
**Questions they ask:**
- Is ACM working correctly?
- Are models accurate?
- Is data quality good?
- What's configured?
- What did auto-tuning change?
- Do models need retraining?

**Enables:**
- System health monitoring
- Model management
- Configuration control
- Performance optimization

### Analytics Team (Deep Analysis)
**Questions they ask:**
- How do detectors correlate?
- Which features matter most?
- What are normal operating patterns?
- How does behavior change over time?
- What causes specific anomaly types?

**Enables:**
- Model improvement
- Pattern discovery
- Causal analysis
- Research & development

---

## Part 3: Required Output Tables

### TIER 1: Real-Time State (Current Status) - 4 tables

#### 1.1 ACM_HealthTimeline
**Purpose**: Continuous health monitoring over time  
**Updates**: Every ACM run  
**Retention**: Full history  
**Columns**:
- RunID, EquipID, Timestamp
- Health (0-100), HealthZone (GOOD/CAUTION/ALERT/CRITICAL)
- RegimeLabel (current operating mode)
- TrendDirection (improving/stable/degrading)

**Enables**:
- "Show equipment health over last week/month/year"
- Trend visualization
- Fleet-wide health comparisons

#### 1.2 ACM_Scores_Wide
**Purpose**: Detector-level scores for diagnostic depth  
**Updates**: Every ACM run  
**Retention**: Full history  
**Columns**:
- RunID, EquipID, Timestamp
- ar1_z, pca_spe_z, pca_t2_z, iforest_z, gmm_z, omr_z
- FusedScore, FusedHealth

**Enables**:
- "Which detector flagged this anomaly?"
- Detector performance analysis
- Multi-detector pattern analysis

#### 1.3 ACM_Episodes
**Purpose**: Anomaly event tracking  
**Updates**: When episodes detected  
**Retention**: Full history  
**Columns**:
- EpisodeID, RunID, EquipID
- StartTime, EndTime, Duration
- Severity, Status (ACTIVE/RESOLVED/ACKNOWLEDGED)
- PrimarySensor, AffectedRegime

**Enables**:
- "What anomalies occurred and when?"
- Active alert management
- Historical pattern analysis

#### 1.4 ACM_RegimeTimeline
**Purpose**: Operating mode context  
**Updates**: Every ACM run  
**Retention**: Full history  
**Columns**:
- RunID, EquipID, Timestamp
- RegimeID, RegimeLabel
- TransitionFlag (new regime vs continuing)

**Enables**:
- "What operating mode was equipment in?"
- Regime-aware analysis
- Mode transition tracking

---

### TIER 2: Predictive Intelligence (Future State) - 4 tables

#### 2.1 ACM_RUL
**Purpose**: Remaining Useful Life predictions  
**Updates**: Every successful ACM run  
**Retention**: Latest + history for trending  
**Columns**:
- RunID, EquipID, PredictedAt
- RUL_Hours, P10_LowerBound, P50_Median, P90_UpperBound
- Confidence, Method
- FailureTime (predicted timestamp)
- TopSensor1, TopSensor2, TopSensor3 (culprits)

**Enables**:
- "When will this equipment fail?"
- Maintenance scheduling
- Risk assessment

#### 2.2 ACM_HealthForecast
**Purpose**: Projected health trajectory  
**Updates**: Every successful forecast run  
**Retention**: Latest continuous forecast  
**Columns**:
- RunID, EquipID, ForecastTime
- PredictedHealth, LowerBound, UpperBound
- Method, Confidence

**Enables**:
- "How will health evolve?"
- Trend projection
- Intervention planning

#### 2.3 ACM_FailureForecast
**Purpose**: Failure probability over time  
**Updates**: Every successful forecast run  
**Retention**: Latest forecast  
**Columns**:
- RunID, EquipID, ForecastTime
- FailureProbability, HazardRate
- ThresholdUsed, Method

**Enables**:
- "What's failure risk in next 30/60/90 days?"
- Risk-based decisions
- Insurance/warranty analysis

#### 2.4 ACM_SensorForecast
**Purpose**: Physical sensor value predictions  
**Updates**: Every successful forecast run  
**Retention**: Latest forecast  
**Columns**:
- RunID, EquipID, ForecastTime
- SensorName, PredictedValue, LowerBound, UpperBound
- Method

**Enables**:
- "What will temperatures/pressures be?"
- Process optimization
- Anomaly pre-detection

---

### TIER 3: Root Cause & Diagnostics (Why?) - 5 tables

#### 3.1 ACM_SensorDefects
**Purpose**: Sensor-level anomaly flagging  
**Updates**: Every ACM run  
**Retention**: Latest + recent history  
**Columns**:
- RunID, EquipID, Timestamp
- SensorName, DefectFlag (0/1)
- AnomalyScore, DetectorSource
- Status (ACTIVE/RESOLVED)

**Enables**:
- "Which sensors are problematic?"
- Sensor health tracking
- Maintenance targeting

#### 3.2 ACM_SensorHotspots
**Purpose**: Top anomalous sensors  
**Updates**: Every ACM run with anomalies  
**Retention**: Recent history  
**Columns**:
- RunID, EquipID, Timestamp
- SensorName, Rank (1-N)
- ContributionScore, AnomalyMagnitude

**Enables**:
- "Which sensors contribute most to current anomaly?"
- Priority ranking
- Inspection focus

#### 3.3 ACM_EpisodeCulprits
**Purpose**: Per-episode sensor attribution  
**Updates**: When episodes detected  
**Retention**: Full history  
**Columns**:
- EpisodeID, RunID, EquipID
- SensorName, Rank
- ContributionScore, DetectorSource

**Enables**:
- "What caused this specific episode?"
- Root cause documentation
- Pattern analysis

#### 3.4 ACM_EpisodeDiagnostics
**Purpose**: Detailed episode analysis  
**Updates**: When episodes detected  
**Retention**: Full history  
**Columns**:
- EpisodeID, RunID, EquipID
- Duration_h, PeakSeverity, AverageSeverity
- AffectedRegimes, SensorCount
- Resolution, Notes

**Enables**:
- "How severe was this episode?"
- Episode comparison
- Diagnostic details

#### 3.5 ACM_DetectorCorrelation
**Purpose**: Inter-detector relationships  
**Updates**: Periodic (weekly/monthly)  
**Retention**: Trend history  
**Columns**:
- RunID, EquipID, AnalysisDate
- Detector1, Detector2
- Correlation, Significance

**Enables**:
- "Do detectors agree or detect different things?"
- Model quality assessment
- Detector redundancy analysis

---

### TIER 4: System Operations (Is ACM Working?) - 5 tables

#### 4.1 ACM_Runs
**Purpose**: Execution tracking  
**Updates**: Every ACM run (start + end)  
**Retention**: Full history  
**Columns**:
- RunID, EquipID
- StartedAt, CompletedAt, Duration
- Status (SUCCESS/FAILED/NOOP/PARTIAL)
- DataStartTime, DataEndTime
- RowsProcessed, ErrorMessage

**Enables**:
- "Did ACM run successfully?"
- Execution history
- Failure diagnosis

#### 4.2 ACM_DataQuality
**Purpose**: Input data health  
**Updates**: Every ACM run  
**Retention**: Recent history (90 days)  
**Columns**:
- RunID, EquipID
- SensorName
- MissingPct, OutlierPct, FlatlinePct
- QualityScore (0-100)
- Status (GOOD/MARGINAL/POOR)

**Enables**:
- "Is incoming data reliable?"
- Data issue identification
- Sensor maintenance needs

#### 4.3 ACM_ForecastingState
**Purpose**: Forecast model persistence  
**Updates**: When forecasts generated  
**Retention**: Latest state + version history  
**Columns**:
- StateID, RunID, EquipID
- StateVersion, StateData (JSON)
- CreatedAt, ExpiresAt
- LockStatus

**Enables**:
- Continuous forecast evolution
- State recovery
- Version tracking

#### 4.4 ACM_AdaptiveConfig
**Purpose**: Auto-tuned configuration  
**Updates**: When auto-tuning triggers  
**Retention**: Latest + change history  
**Columns**:
- ConfigID, RunID, EquipID
- ParameterPath, Value
- UpdatedAt, UpdatedBy (AUTO/MANUAL)
- Reason, MetricBefore, MetricAfter

**Enables**:
- "What did auto-tuning change?"
- Configuration tracking
- Tuning effectiveness

#### 4.5 ACM_RunTimers
**Purpose**: Performance profiling  
**Updates**: Every ACM run  
**Retention**: Recent history (30 days)  
**Columns**:
- RunID, EquipID
- Stage (DataLoad/Features/Detectors/Fusion/Forecast)
- DurationSeconds
- MemoryMB, CPUPercent

**Enables**:
- "Where is ACM slow?"
- Performance optimization
- Bottleneck identification

---

### TIER 5: Configuration & Audit (How is ACM Configured?) - 3 tables

#### 5.1 ACM_Config
**Purpose**: Current configuration  
**Updates**: Manual changes or sync from CSV  
**Retention**: Latest state  
**Columns**:
- ConfigID, EquipID
- ParameterPath, Value, DataType
- UpdatedAt, Source (CSV/SQL/AUTO)

**Enables**:
- "What are current settings?"
- Configuration management
- Equipment-specific overrides

#### 5.2 ACM_ConfigHistory
**Purpose**: Configuration change audit  
**Updates**: On any config change  
**Retention**: Full history  
**Columns**:
- HistoryID, RunID, EquipID
- ParameterPath, OldValue, NewValue
- ChangedAt, ChangedBy, Reason

**Enables**:
- "What changed and when?"
- Audit compliance
- Rollback capability

#### 5.3 ACM_RunLogs
**Purpose**: Detailed execution logs  
**Updates**: Throughout ACM execution  
**Retention**: 30-90 days  
**Columns**:
- LogID, RunID, EquipID
- LogTime, Level (INFO/WARN/ERROR)
- Component, Message, Details (JSON)

**Enables**:
- "What happened during this run?"
- Error diagnosis
- Troubleshooting

---

### TIER 6: Advanced Analytics (Deep Insights) - 4 tables

#### 6.1 ACM_DriftSeries
**Purpose**: Behavior change tracking  
**Updates**: Every ACM run  
**Retention**: Full history  
**Columns**:
- RunID, EquipID, Timestamp
- DriftScore, CUSUMValue
- ThresholdUsed, DriftFlag (0/1)

**Enables**:
- "When did equipment behavior change?"
- Drift visualization
- Change-point analysis

#### 6.2 ACM_RegimeOccupancy
**Purpose**: Operating mode statistics  
**Updates**: Periodic aggregation  
**Retention**: Trend history  
**Columns**:
- EquipID, RegimeID, RegimeLabel
- Period (daily/weekly/monthly)
- OccupancyPct, TotalHours
- AverageHealth, EpisodeCount

**Enables**:
- "How much time in each mode?"
- Mode utilization
- Mode-specific health

#### 6.3 ACM_RegimeTransitions
**Purpose**: Mode switching patterns  
**Updates**: Periodic aggregation  
**Retention**: Trend history  
**Columns**:
- EquipID, FromRegime, ToRegime
- Period, TransitionCount
- AverageTransitionTime

**Enables**:
- "How do modes transition?"
- Transition patterns
- Unstable operation detection

#### 6.4 ACM_CalibrationSummary
**Purpose**: Model quality metrics  
**Updates**: After model fitting  
**Retention**: Recent history  
**Columns**:
- RunID, EquipID
- DetectorName, ModelVersion
- TrainAccuracy, ValidationAccuracy
- CalibrationDate, NeedsRefit

**Enables**:
- "Are models well-calibrated?"
- Model quality tracking
- Refit scheduling

---

## Part 4: Recommended ALLOWED_TABLES

### Final Table Set: 25 Core Tables

**TIER 1: Real-Time State (4 tables)**
1. ACM_HealthTimeline
2. ACM_Scores_Wide
3. ACM_Episodes
4. ACM_RegimeTimeline

**TIER 2: Predictive Intelligence (4 tables)**
5. ACM_RUL
6. ACM_HealthForecast
7. ACM_FailureForecast
8. ACM_SensorForecast

**TIER 3: Root Cause & Diagnostics (5 tables)**
9. ACM_SensorDefects
10. ACM_SensorHotspots
11. ACM_EpisodeCulprits
12. ACM_EpisodeDiagnostics
13. ACM_DetectorCorrelation

**TIER 4: System Operations (5 tables)**
14. ACM_Runs
15. ACM_DataQuality
16. ACM_ForecastingState
17. ACM_AdaptiveConfig
18. ACM_RunTimers

**TIER 5: Configuration & Audit (3 tables)**
19. ACM_Config
20. ACM_ConfigHistory
21. ACM_RunLogs

**TIER 6: Advanced Analytics (4 tables)**
22. ACM_DriftSeries
23. ACM_RegimeOccupancy
24. ACM_RegimeTransitions
25. ACM_CalibrationSummary

---

## Part 5: Tables NOT Recommended

These are in the database but NOT recommended based on functionality analysis:

### Redundant / Derivable
- **ACM_Scores_Long** - Redundant with Scores_Wide (just long format)
- **ACM_HealthHistogram** - Derivable from HealthTimeline
- **ACM_HealthZoneByPeriod** - Derivable from HealthTimeline aggregation
- **ACM_SensorAnomalyByPeriod** - Derivable from SensorDefects aggregation
- **ACM_DefectTimeline** - Redundant with SensorDefects over time
- **ACM_DefectSummary** - Derivable from SensorDefects aggregation

### Overly Specific / Niche
- **ACM_HealthForecast_Continuous** - Merged into HealthForecast
- **ACM_FailureHazard_TS** - Merged into FailureForecast (HazardRate column)
- **ACM_DetectorForecast_TS** - Too detailed, not actionable
- **ACM_ContributionCurrent** - Snapshot of SensorHotspots (latest query suffices)
- **ACM_ContributionTimeline** - Redundant with SensorHotspots over time
- **ACM_SensorHotspotTimeline** - Redundant with SensorHotspots over time
- **ACM_Anomaly_Events** - Redundant with Episodes

### Operational Tables (Outside Core ACM)
- **ACM_ColdstartState** - Internal state, not user-facing
- **ACM_RefitRequests** - Internal orchestration
- **ACM_BaselineBuffer** - Internal buffer
- **ACM_HistorianData** - Raw data cache
- **ACM_SensorNormalized_TS** - Intermediate processing

### Model Internals (Too Technical)
- **ACM_PCA_Loadings** - Model internals
- **ACM_PCA_Metrics** - Model internals
- **ACM_PCA_Models** - Model internals
- **ACM_OMRTimeline** - Individual detector timeline (use Scores_Wide)
- **ACM_OMR_Diagnostics** - Model internals

### Unimplemented / Deprecated
- **ACM_RegimeDefinitions** - New v11 feature, not yet implemented
- **ACM_ActiveModels** - New v11 feature, not yet implemented
- **ACM_ThresholdCrossings** - Not implemented
- **ACM_AlertAge** - Derivable from Episodes
- **ACM_SinceWhen** - Derivable from Episodes
- **ACM_ThresholdMetadata** - Part of Config

---

## Part 6: Design Rationale

### Why 25 Tables is Optimal

**Compared to 17 (Too Few)**:
- 17 tables lacked operational visibility (no Runs, RunLogs, RunTimers)
- Missing root cause analysis (no SensorHotspots, EpisodeCulprits)
- No advanced analytics (no DriftSeries, RegimeOccupancy)

**Compared to 42 (Dashboard-Based Approach)**:
- Removed 17 redundant/derivable tables
- Removed 5 overly specific tables
- Removed 8 internal/operational tables
- Removed 5 unimplemented tables

**Compared to 73 (Full Database)**:
- Removed 48 legacy/unused tables
- Focused on user-facing outputs
- Eliminated model internals

### Coverage by User Type

| User Type | Tables Used | Coverage |
|-----------|------------|----------|
| **Operations** | 8 tables | ✅ Full |
| **Maintenance** | 10 tables | ✅ Full |
| **Engineering** | 12 tables | ✅ Full |
| **Analytics** | 13 tables | ✅ Full |

### Capability Coverage

| ACM Capability | Tables Supporting | Coverage |
|----------------|-------------------|----------|
| **Health Monitoring** | HealthTimeline, Scores_Wide | ✅ Complete |
| **Anomaly Detection** | Episodes, EpisodeDiagnostics | ✅ Complete |
| **Failure Prediction** | RUL, HealthForecast, FailureForecast | ✅ Complete |
| **Operating Context** | RegimeTimeline, RegimeOccupancy | ✅ Complete |
| **Drift Detection** | DriftSeries | ✅ Complete |
| **Root Cause** | SensorDefects, SensorHotspots, EpisodeCulprits | ✅ Complete |
| **Adaptive Learning** | AdaptiveConfig, ConfigHistory | ✅ Complete |
| **Quality Tracking** | DataQuality, CalibrationSummary | ✅ Complete |

---

## Part 7: Implementation Notes

### Tables Already Implemented
Based on code analysis, these are already being written:
- ACM_HealthTimeline
- ACM_RegimeTimeline
- ACM_Scores_Wide (likely via fuse.py)
- ACM_Episodes (likely via fuse.py)
- ACM_RUL
- ACM_HealthForecast
- ACM_FailureForecast
- ACM_SensorForecast
- ACM_DataQuality
- ACM_ForecastingState
- ACM_Runs
- ACM_RunTimers
- ACM_EpisodeCulprits
- ACM_Config (via sync script)

### Tables Needing Implementation (~11 tables)
- ACM_SensorDefects
- ACM_SensorHotspots
- ACM_EpisodeDiagnostics
- ACM_DetectorCorrelation
- ACM_AdaptiveConfig
- ACM_ConfigHistory (partially implemented)
- ACM_RunLogs
- ACM_DriftSeries
- ACM_RegimeOccupancy
- ACM_RegimeTransitions
- ACM_CalibrationSummary

---

## Conclusion

This functionality-based approach yields **25 core tables** that provide:

✅ **Complete visibility** into all ACM capabilities  
✅ **Actionable information** for all user types  
✅ **Balanced scope** - not too few, not too many  
✅ **No dashboard dependency** - tables chosen for intrinsic value  
✅ **Future-proof** - supports dashboards not yet designed  

The 25-table set represents the **minimum complete** output for a production predictive maintenance system, with each table serving a clear, non-redundant purpose.

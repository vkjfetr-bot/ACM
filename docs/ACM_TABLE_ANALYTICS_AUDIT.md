# ACM Table and Analytics Audit Report

**Date:** 2024-12-25  
**Version:** 11.0.0  
**Status:** 🔴 CRITICAL GAPS IDENTIFIED

---

## Executive Summary

This audit reveals **significant gaps** between ACM's table inventory and what's needed for full system functionality:

### Key Findings

| Metric | Count | Status |
|--------|-------|--------|
| **Total tables in database** | 73 | ℹ️ Full inventory |
| **Tables actively written (ALLOWED_TABLES)** | 17 | ⚠️ Only 23% of inventory |
| **Tables required by Grafana dashboards** | 26 | 📊 Visualization needs |
| **Dashboard coverage gap** | 15 tables (58%) | 🔴 **CRITICAL** |
| **Orphaned tables (in DB, not written)** | 58 tables | 🧹 Cleanup candidate |
| **Missing schema definitions** | 2 tables | 🔧 Need creation |

### Critical Issue

**Only 42% of dashboard requirements are met** (11 out of 26 tables). Dashboards are showing incomplete or stale data, severely limiting ACM visibility and operational effectiveness.

---

## 1. Current State: ALLOWED_TABLES (17 tables)

These are the ONLY tables actively written by the ACM pipeline (defined in `core/output_manager.py`):

### TIER 1: Core Pipeline Output (Essential)
1. **ACM_Scores_Wide** - Per-timestamp detector z-scores (PRIMARY OUTPUT)
2. **ACM_HealthTimeline** - Health % over time (REQUIRED for RUL forecasting)
3. **ACM_Episodes** - Detected anomaly episodes with diagnostics
4. **ACM_RegimeTimeline** - Operating regime assignments

### TIER 2: Forecasting (Predictions)
5. **ACM_RUL** - Remaining Useful Life with confidence bounds
6. **ACM_HealthForecast** - Projected health trajectory
7. **ACM_FailureForecast** - Failure probability over time
8. **ACM_SensorForecast** - Physical sensor value forecasts

### TIER 3: Operational (Run tracking, state management)
9. **ACM_DataQuality** - Data quality per sensor
10. **ACM_ForecastingState** - Persistent forecasting model state
11. **ACM_AdaptiveConfig** - Dynamic per-equipment configuration

### TIER 4: Diagnostics (Sensor attribution, model metadata)
12. **ACM_SensorDefects** - Sensor-level anomaly flags
13. **ACM_SensorHotspots** - Top anomalous sensors (used for RUL attribution)
14. **ACM_EpisodeCulprits** - Per-episode sensor culprits
15. **ACM_EpisodeDiagnostics** - Per-episode diagnostic details (duration_h, etc.)
16. **ACM_RegimeDefinitions** - Regime centroids and metadata (v11) ⚠️ *Missing schema*
17. **ACM_ActiveModels** - Active model versions per equipment (v11) ⚠️ *Missing schema*

---

## 2. Dashboard Requirements (26 tables)

Analysis of Grafana dashboards reveals these table dependencies:

### By Dashboard

#### acm_asset_story.json (9 tables)
- ✅ ACM_HealthTimeline
- ✅ ACM_RUL
- ✅ ACM_RegimeTimeline
- ✅ ACM_Scores_Wide
- ✅ ACM_SensorDefects
- 🔴 ACM_Anomaly_Events
- 🔴 ACM_ContributionCurrent
- 🔴 ACM_HealthForecast_Continuous
- 🔴 ACM_SensorHotspotTimeline

#### acm_behavior.json (12 tables)
- ✅ ACM_EpisodeDiagnostics
- ✅ ACM_Episodes
- ✅ ACM_HealthTimeline
- ✅ ACM_RUL
- ✅ ACM_RegimeTimeline
- ✅ ACM_Scores_Wide
- 🔴 ACM_Anomaly_Events
- 🔴 ACM_ContributionCurrent
- 🔴 ACM_ContributionTimeline
- 🔴 ACM_DriftSeries
- 🔴 ACM_RegimeOccupancy
- 🔴 ACM_Runs

#### acm_fleet_overview.json (6 tables)
- ✅ ACM_DataQuality
- ✅ ACM_Episodes
- ✅ ACM_HealthTimeline
- ✅ ACM_RUL
- 🔴 ACM_HealthZoneByPeriod
- 🔴 ACM_Runs

#### acm_forecasting.json (8 tables)
- ✅ ACM_FailureForecast
- ✅ ACM_HealthForecast
- ✅ ACM_RUL
- ✅ ACM_SensorForecast
- 🔴 ACM_DetectorForecast_TS
- 🔴 ACM_FailureHazard_TS
- 🔴 ACM_HealthForecast_Continuous
- 🔴 ACM_Runs

#### acm_operations_monitor.json (5 tables)
- 🔴 ACM_ColdstartState
- 🔴 ACM_RefitRequests
- 🔴 ACM_RunLogs
- 🔴 ACM_RunTimers
- 🔴 ACM_Runs

#### acm_performance_monitor.json (2 tables)
- 🔴 ACM_RunTimers
- 🔴 ACM_Runs

**Legend:**
- ✅ = In ALLOWED_TABLES (actively written)
- 🔴 = NOT in ALLOWED_TABLES (dashboards showing stale/empty data)

---

## 3. Critical Gaps: Dashboard Tables NOT in ALLOWED_TABLES

**15 tables are needed by dashboards but NOT being written** - these dashboards are broken or showing stale data:

| Table | Used By | Impact |
|-------|---------|--------|
| **ACM_Anomaly_Events** | acm_asset_story, acm_behavior | 🔴 Episode visualization broken |
| **ACM_Runs** | acm_behavior, acm_fleet_overview, acm_forecasting, acm_operations_monitor, acm_performance_monitor | 🔴 Run tracking completely broken (5 dashboards) |
| **ACM_RunLogs** | acm_operations_monitor | 🔴 Cannot view pipeline logs |
| **ACM_RunTimers** | acm_operations_monitor, acm_performance_monitor | 🔴 Performance monitoring broken |
| **ACM_ContributionCurrent** | acm_asset_story, acm_behavior | ⚠️ Current sensor contributions missing |
| **ACM_ContributionTimeline** | acm_behavior | ⚠️ Historical contributions missing |
| **ACM_DriftSeries** | acm_behavior | ⚠️ Drift visualization broken |
| **ACM_RegimeOccupancy** | acm_behavior | ⚠️ Regime statistics missing |
| **ACM_HealthZoneByPeriod** | acm_fleet_overview | ⚠️ Health aggregates missing |
| **ACM_ColdstartState** | acm_operations_monitor | ⚠️ Coldstart tracking missing |
| **ACM_RefitRequests** | acm_operations_monitor | ⚠️ Refit status missing |
| **ACM_SensorHotspotTimeline** | acm_asset_story | ⚠️ Sensor trends missing |
| **ACM_DetectorForecast_TS** | acm_forecasting | ⚠️ Detector forecasts missing |
| **ACM_FailureHazard_TS** | acm_forecasting | ⚠️ Hazard rates missing |
| **ACM_HealthForecast_Continuous** | acm_asset_story, acm_forecasting | ⚠️ Continuous forecasts missing |

---

## 4. What ACM Lost in Terms of Features and Visibility

### 4.1 Lost Operational Visibility (High Impact)

#### Run Management & Tracking
- **ACM_Runs** - Core run metadata (start/end times, status, equipment)
  - **Impact:** Cannot track pipeline execution history
  - **Dashboards affected:** 5 (most critical)
  - **Functionality lost:** Run status, execution timeline, error tracking

- **ACM_RunLogs** - Detailed pipeline logs per run
  - **Impact:** Cannot diagnose pipeline issues from UI
  - **Dashboards affected:** acm_operations_monitor
  - **Functionality lost:** Error investigation, debug capability

- **ACM_RunTimers** - Performance metrics per pipeline stage
  - **Impact:** Cannot identify performance bottlenecks
  - **Dashboards affected:** acm_operations_monitor, acm_performance_monitor
  - **Functionality lost:** Performance profiling, optimization insights

#### State Management
- **ACM_ColdstartState** - Coldstart progression tracking
  - **Impact:** No visibility into data accumulation for new equipment
  - **Functionality lost:** Coldstart monitoring, readiness assessment

- **ACM_RefitRequests** - Model refit request tracking
  - **Impact:** Cannot see when/why models need retraining
  - **Functionality lost:** Model lifecycle management

### 4.2 Lost Analytical Depth (Medium Impact)

#### Sensor Attribution & Contribution Analysis
- **ACM_ContributionCurrent** - Current sensor contribution scores
  - **Impact:** Cannot see which sensors are driving current anomalies
  - **Functionality lost:** Real-time root cause analysis

- **ACM_ContributionTimeline** - Historical sensor contributions
  - **Impact:** Cannot analyze how sensor importance changes over time
  - **Functionality lost:** Trend analysis for sensor behavior

- **ACM_SensorHotspotTimeline** - Sensor anomaly trends over time
  - **Impact:** Cannot identify persistent problematic sensors
  - **Functionality lost:** Long-term sensor health tracking

#### Drift & Concept Changes
- **ACM_DriftSeries** - Drift detection time series
  - **Impact:** Cannot visualize when equipment behavior changes
  - **Functionality lost:** Concept drift monitoring, behavior change detection

#### Regime Analytics
- **ACM_RegimeOccupancy** - Time spent in each operating regime
  - **Impact:** Cannot analyze operating mode distribution
  - **Functionality lost:** Regime utilization analysis, operating pattern insights

- **ACM_HealthZoneByPeriod** - Aggregated health by time period
  - **Impact:** Cannot see fleet-wide health trends
  - **Functionality lost:** Fleet health analytics, comparative analysis

#### Episode Details
- **ACM_Anomaly_Events** - Structured anomaly event records
  - **Impact:** Cannot track discrete anomaly events vs episodes
  - **Functionality lost:** Event-based alerting, event timeline

### 4.3 Lost Forecasting Insights (Medium Impact)

- **ACM_DetectorForecast_TS** - Per-detector forecast time series
  - **Impact:** Cannot see individual detector predictions
  - **Functionality lost:** Detector-level forecast validation

- **ACM_FailureHazard_TS** - Failure hazard rate over time
  - **Impact:** Cannot visualize failure probability evolution
  - **Functionality lost:** Risk curve visualization

- **ACM_HealthForecast_Continuous** - Continuous health forecast (vs discrete)
  - **Impact:** Gaps in forecast visualization
  - **Functionality lost:** Smooth forecast interpolation

### 4.4 Lost Diagnostic Capabilities (Low Impact - Deep Analysis)

These tables exist in DB but are not in ALLOWED_TABLES. While not critical for dashboards, they provide deep diagnostic value:

#### Model Diagnostics
- **ACM_PCA_Loadings** - PCA component loadings
- **ACM_PCA_Metrics** - PCA model quality metrics
- **ACM_PCA_Models** - PCA model metadata
- **ACM_OMRTimeline** - Overall Model Residual trends
- **ACM_OMR_Diagnostics** - OMR diagnostic details
- **ACM_OMRContributionsLong** - OMR sensor contributions

#### Detector Quality
- **ACM_DetectorCorrelation** - Cross-detector correlation analysis
- **ACM_FusionQualityReport** - Fusion algorithm quality metrics
- **ACM_CalibrationSummary** - Detector calibration status

#### Regime Analysis
- **ACM_RegimeTransitions** - Regime switching patterns
- **ACM_RegimeDwellStats** - Regime duration statistics
- **ACM_RegimeStability** - Regime stability metrics
- **ACM_RegimeStats** - Regime-specific statistics
- **ACM_RegimeState** - Current regime state

#### Episode Analytics
- **ACM_EpisodeMetrics** - Episode-level aggregate metrics
- **ACM_EpisodesQC** - Episode quality control flags
- **ACM_Regime_Episodes** - Episodes by regime

#### Defect Tracking
- **ACM_DefectSummary** - Aggregated defect statistics
- **ACM_DefectTimeline** - Defect counts over time
- **ACM_ThresholdCrossings** - Threshold violation events
- **ACM_ThresholdMetadata** - Threshold configuration
- **ACM_AlertAge** - How long alerts have been active
- **ACM_SinceWhen** - Alert start timestamps

#### Feature Management
- **ACM_FeatureDropLog** - Features removed during pipeline
- **ACM_SensorRanking** - Sensor importance ranking
- **ACM_SensorAnomalyByPeriod** - Sensor anomalies aggregated by time

#### Data Quality
- **ACM_HistorianData** - Raw historian data cache
- **ACM_BaselineBuffer** - Baseline data accumulation
- **ACM_SensorNormalized_TS** - Normalized sensor time series

#### Configuration & Metadata
- **ACM_Config** - Runtime configuration (current)
- **ACM_ConfigHistory** - Configuration change history
- **ACM_TagEquipmentMap** - Sensor tag to equipment mapping
- **ACM_SchemaVersion** - Database schema version tracking
- **ACM_Run_Stats** - Run-level statistics
- **ACM_RunMetadata** - Extended run metadata
- **ACM_RunMetrics** - Run performance metrics

#### Specialized Analytics
- **ACM_HealthHistogram** - Health score distribution
- **ACM_HealthDistributionOverTime** - Health distribution trends
- **ACM_DailyFusedProfile** - Daily health profile
- **ACM_DriftEvents** - Discrete drift event markers
- **ACM_ForecastState** - Forecast algorithm state
- **ACM_RUL_LearningState** - RUL model learning state
- **ACM_Scores_Long** - Scores in long format (alternative to wide)

---

## 5. Orphaned Tables (58 tables in DB, not in ALLOWED_TABLES)

These tables exist in the database but are NOT actively written by the current pipeline. They may contain stale data or be completely unused.

### Category Breakdown

#### 🔴 **HIGH PRIORITY - Needed by Dashboards (15 tables)**
Already listed in Section 3

#### ⚠️ **MEDIUM PRIORITY - Valuable Diagnostics (23 tables)**
- Model diagnostics: PCA_Loadings, PCA_Metrics, PCA_Models, OMRTimeline, OMR_Diagnostics, OMRContributionsLong
- Detector quality: DetectorCorrelation, FusionQualityReport, CalibrationSummary
- Regime analysis: RegimeTransitions, RegimeDwellStats, RegimeStability, RegimeStats, RegimeState, Regime_Episodes
- Episode analytics: EpisodeMetrics, EpisodesQC
- Defect tracking: DefectSummary, DefectTimeline, ThresholdCrossings, AlertAge, SinceWhen, ThresholdMetadata

#### 📊 **LOW PRIORITY - Specialized Analytics (20 tables)**
- Feature management: FeatureDropLog, SensorRanking, SensorAnomalyByPeriod
- Data quality: HistorianData, BaselineBuffer, SensorNormalized_TS
- Configuration: Config, ConfigHistory, TagEquipmentMap, SchemaVersion
- Run metadata: Run_Stats, RunMetadata, RunMetrics
- Health analytics: HealthHistogram, HealthDistributionOverTime, DailyFusedProfile
- Drift: DriftEvents
- State: ForecastState, RUL_LearningState
- Alternative formats: Scores_Long

---

## 6. Missing Schema Definitions (2 tables)

These tables are in ALLOWED_TABLES but have NO schema definition in the database:

1. **ACM_RegimeDefinitions** (v11.0.0 feature)
   - Purpose: Regime centroids and metadata
   - Status: Schema needs to be created

2. **ACM_ActiveModels** (v11.0.0 feature)
   - Purpose: Active model versions per equipment
   - Status: Schema needs to be created

---

## 7. Recommendations

### 7.1 Immediate Actions (Fix Critical Dashboard Gaps)

**Priority 1: Restore Run Management (Affects 5 dashboards)**
```python
# Add to ALLOWED_TABLES in core/output_manager.py
ALLOWED_TABLES = {
    # ... existing tables ...
    
    # RUN MANAGEMENT (Critical for operations)
    'ACM_Runs',              # Run metadata and status
    'ACM_RunLogs',           # Pipeline logs
    'ACM_RunTimers',         # Performance metrics
}
```

**Priority 2: Restore Sensor Analytics (Affects 2 dashboards)**
```python
    # SENSOR ANALYTICS
    'ACM_ContributionCurrent',    # Current sensor contributions
    'ACM_ContributionTimeline',   # Historical contributions
    'ACM_SensorHotspotTimeline',  # Sensor trend analysis
```

**Priority 3: Restore Regime & Drift Analytics (Affects 2 dashboards)**
```python
    # REGIME & DRIFT
    'ACM_DriftSeries',       # Drift detection
    'ACM_RegimeOccupancy',   # Regime statistics
    'ACM_HealthZoneByPeriod', # Health aggregates
```

**Priority 4: Restore Episode Tracking (Affects 2 dashboards)**
```python
    # EPISODE MANAGEMENT
    'ACM_Anomaly_Events',    # Event-based anomaly tracking
```

**Priority 5: Restore Operations Monitoring (Affects 1 dashboard)**
```python
    # OPERATIONS
    'ACM_ColdstartState',    # Coldstart progress
    'ACM_RefitRequests',     # Refit tracking
```

**Priority 6: Restore Forecast Details (Affects 1 dashboard)**
```python
    # FORECASTING DETAILS
    'ACM_DetectorForecast_TS',      # Per-detector forecasts
    'ACM_FailureHazard_TS',         # Hazard rates
    'ACM_HealthForecast_Continuous', # Continuous health forecast
```

### 7.2 Medium-Term Actions (Restore Diagnostic Depth)

**Add valuable diagnostic tables back:**
```python
    # MODEL DIAGNOSTICS
    'ACM_PCA_Loadings',
    'ACM_PCA_Metrics',
    'ACM_OMRTimeline',
    'ACM_OMR_Diagnostics',
    
    # DETECTOR QUALITY
    'ACM_DetectorCorrelation',
    'ACM_CalibrationSummary',
    
    # REGIME ANALYTICS
    'ACM_RegimeTransitions',
    'ACM_RegimeDwellStats',
    'ACM_RegimeStability',
    
    # EPISODE ANALYTICS
    'ACM_EpisodeMetrics',
    
    # DEFECT TRACKING
    'ACM_DefectSummary',
    'ACM_ThresholdCrossings',
    'ACM_AlertAge',
```

### 7.3 Long-Term Actions (Complete Table Set)

**Create schema for new v11.0.0 tables:**
1. Create `ACM_RegimeDefinitions` table schema
2. Create `ACM_ActiveModels` table schema

**Deprecate truly unused tables:**
Review these low-value tables for removal:
- ACM_Scores_Long (redundant with Scores_Wide)
- ACM_ForecastState (if superseded by ForecastingState)
- ACM_DailyFusedProfile (if not used)
- ACM_HealthDistributionOverTime (if not used)

### 7.4 Optimal Table Set Recommendation

**Recommended ALLOWED_TABLES (42 tables total):**

This represents a balanced approach - not too many (73), not too few (17):

**TIER 1: CORE OUTPUT (4 tables)** ✅ Already included
- ACM_Scores_Wide
- ACM_HealthTimeline
- ACM_Episodes
- ACM_RegimeTimeline

**TIER 2: FORECASTING (7 tables)** +3 additions
- ACM_RUL ✅
- ACM_HealthForecast ✅
- ACM_FailureForecast ✅
- ACM_SensorForecast ✅
- ➕ ACM_HealthForecast_Continuous
- ➕ ACM_DetectorForecast_TS
- ➕ ACM_FailureHazard_TS

**TIER 3: OPERATIONAL (8 tables)** +5 additions
- ACM_DataQuality ✅
- ACM_ForecastingState ✅
- ACM_AdaptiveConfig ✅
- ➕ ACM_Runs
- ➕ ACM_RunLogs
- ➕ ACM_RunTimers
- ➕ ACM_ColdstartState
- ➕ ACM_RefitRequests

**TIER 4: DIAGNOSTICS (14 tables)** +8 additions
- ACM_SensorDefects ✅
- ACM_SensorHotspots ✅
- ACM_EpisodeCulprits ✅
- ACM_EpisodeDiagnostics ✅
- ACM_RegimeDefinitions ✅
- ACM_ActiveModels ✅
- ➕ ACM_Anomaly_Events
- ➕ ACM_SensorHotspotTimeline
- ➕ ACM_ContributionCurrent
- ➕ ACM_ContributionTimeline
- ➕ ACM_EpisodeMetrics
- ➕ ACM_DefectSummary
- ➕ ACM_ThresholdCrossings
- ➕ ACM_AlertAge

**TIER 5: ANALYTICS (9 tables)** New tier
- ➕ ACM_DriftSeries
- ➕ ACM_RegimeOccupancy
- ➕ ACM_RegimeTransitions
- ➕ ACM_RegimeDwellStats
- ➕ ACM_RegimeStability
- ➕ ACM_HealthZoneByPeriod
- ➕ ACM_DetectorCorrelation
- ➕ ACM_CalibrationSummary
- ➕ ACM_FeatureDropLog

**Total: 42 tables (58% of full DB, 147% increase from current 17)**

This restores:
- ✅ 100% dashboard coverage (all 26 tables)
- ✅ Full operational visibility
- ✅ Complete diagnostic capabilities
- ✅ Valuable analytics depth

---

## 8. Impact Assessment

### Current State (17 tables)
- ❌ Dashboard coverage: 42% (11/26 tables)
- ❌ Operational visibility: POOR (no run tracking, logs, or timers)
- ❌ Diagnostic depth: LIMITED (basic episode/sensor attribution only)
- ❌ Analytics: MINIMAL (no drift, regime stats, or trends)

### After Immediate Actions (32 tables, +15)
- ✅ Dashboard coverage: 100% (26/26 tables)
- ✅ Operational visibility: EXCELLENT (full run management)
- ⚠️ Diagnostic depth: GOOD (episodes, sensors, basic trends)
- ⚠️ Analytics: MODERATE (drift, regime basics)

### After Medium-Term Actions (42 tables, +25)
- ✅ Dashboard coverage: 100% (26/26 tables)
- ✅ Operational visibility: EXCELLENT
- ✅ Diagnostic depth: EXCELLENT (full model, detector, regime diagnostics)
- ✅ Analytics: COMPREHENSIVE (all trends, correlations, quality metrics)

---

## 9. Conclusion

ACM's table reduction from 73 to 17 active tables (77% reduction) went **too far**, eliminating critical operational visibility and diagnostic capabilities:

### What Was Gained
- ✅ Cleaner codebase (fewer write operations)
- ✅ Reduced SQL write overhead
- ✅ Simpler maintenance surface

### What Was Lost
- ❌ 58% of dashboard functionality (15/26 tables missing)
- ❌ All run management and operational tracking
- ❌ All performance monitoring capability
- ❌ Sensor contribution analysis
- ❌ Drift detection visualization
- ❌ Regime analytics
- ❌ Model diagnostics
- ❌ Detector quality metrics

### Recommended Path Forward

**Immediate (Week 1):** Add 15 dashboard-critical tables → Restore full dashboard functionality

**Short-term (Month 1):** Add 10 diagnostic tables → Restore deep analysis capabilities

**Result:** 42 total tables (58% of original 73) - **optimal balance**

This provides:
- ✅ Full dashboard coverage
- ✅ Complete operational visibility
- ✅ Comprehensive diagnostics
- ✅ Streamlined vs. original 73 (31 tables removed)
- ✅ Manageable codebase

**The optimal table set is ~40-45 tables, not 17.**

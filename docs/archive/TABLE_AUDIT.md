# ACM Database Table Audit

**Last Updated:** 2024-12-25  
**Version:** 11.0.0  
**Approach:** Functionality-based + Code Analysis  
**Status:** ✅ COMPREHENSIVE AUDIT COMPLETE

## Summary
**Design Principle**: Tables chosen based on **4 fundamental questions ACM must answer** + **actual code behavior analysis**

**Critical Discovery**: Initial analysis missed 13 actively-written tables

**Complete Table Set**: 42 core tables organized in 7 functional tiers  
**Previous Version**: 29 tables  
**Increase**: +45% (13 tables added after comprehensive code review)

**What Was Missing:**
- 7 operational/diagnostic tables actively written by code
- 5 V11 feature tables (new architecture)
- 1 additional analytics table

---

## The Crux of ACM

ACM exists to answer 4 questions:

### 1. **What is current health?**
→ TIER 1 tables (6): HealthTimeline, Scores_Wide, Episodes, RegimeTimeline, SensorDefects, SensorHotspots  
→ **Plus:** RunMetrics (system quality tracking)

### 2. **If not healthy, what's the reason?**
→ TIER 3 tables (6): EpisodeCulprits, EpisodeDiagnostics, DetectorCorrelation, DriftSeries, SensorCorrelations, FeatureDropLog

### 3. **What will future health look like?**
→ TIER 2 tables (4): RUL, HealthForecast, FailureForecast, SensorForecast  
→ **Plus:** SeasonalPatterns (improves forecasting)

### 4. **What will cause future degradation?**
→ TIER 2 + TIER 3 tables (combined)  
→ **Plus:** RefitRequests (model freshness), RegimePromotionLog (regime stability)

---

## ALLOWED_TABLES (42 tables) - Comprehensive Set

See `docs/ACM_TABLES_COMPLETE_AUDIT.md` for detailed analysis of all tables.

### TIER 1: Current State (6 tables)
**Answers:** "What is current health?"

| Table | Purpose | Key Question |
|-------|---------|--------------|
| **ACM_HealthTimeline** | Health history + current | What's the health trend? |
| **ACM_Scores_Wide** | All 6 detector scores | Which detectors are firing? |
| **ACM_Episodes** | Active/historical anomalies | What anomalies are active? |
| **ACM_RegimeTimeline** | Operating mode context | What mode is equipment in? |
| **ACM_SensorDefects** | Problematic sensors NOW | Which sensors are bad? |
| **ACM_SensorHotspots** | Top culprit sensors | What are worst sensors? |

### TIER 2: Future State (4 tables)
**Answers:** "What will future health look like?"

| Table | Purpose | Key Question |
|-------|---------|--------------|
| **ACM_RUL** | Remaining Useful Life | When will it fail? |
| **ACM_HealthForecast** | Projected health trajectory | How will health evolve? |
| **ACM_FailureForecast** | Failure probability | What's the failure risk? |
| **ACM_SensorForecast** | Physical sensor predictions | Which sensors will degrade? |

### TIER 3: Root Cause (6 tables - was 4)
**Answers:** "Why is this happening?" (current + future)

| Table | Purpose | Key Question | Status |
|-------|---------|--------------|--------|
| **ACM_EpisodeCulprits** | What caused each episode | Why did this anomaly occur? | ✅ |
| **ACM_EpisodeDiagnostics** | Episode details/severity | How bad was this episode? | ✅ |
| **ACM_DetectorCorrelation** | Inter-detector relationships | Are models working together? | ✅ |
| **ACM_DriftSeries** | Behavior changes | When did behavior change? | ✅ |
| **ACM_SensorCorrelations** | Multivariate relationships | How do sensors co-vary? | **NEW** |
| **ACM_FeatureDropLog** | Why features dropped | Why was sensor excluded? | **NEW** |

### TIER 4: Data & Model Management (10 tables - was 7)
**Purpose:** Long-term storage, progressive learning

| Table | Purpose | Why Critical | Status |
|-------|---------|--------------|--------|
| **ACM_BaselineBuffer** | Raw sensor data accumulation | Coldstart, progressive training | ✅ |
| **ACM_HistorianData** | Cached historian data | Performance, reduces DB load | ✅ |
| **ACM_SensorNormalized_TS** | Normalized sensor values | Pattern analysis | ✅ |
| **ACM_DataQuality** | Input data health | Data reliability | ✅ |
| **ACM_ForecastingState** | Forecast model state | Continuous forecast evolution | ✅ |
| **ACM_CalibrationSummary** | Model quality over time | Performance tracking | ✅ |
| **ACM_AdaptiveConfig** | Auto-tuned configuration | Track what ACM learned | ✅ |
| **ACM_RefitRequests** | Model retraining requests | Model freshness tracking | **NEW** |
| **ACM_PCA_Metrics** | PCA component metrics | Model quality assessment | **NEW** |
| **ACM_RunMetadata** | Detailed run context | Batch info, data ranges | **NEW** |

### TIER 5: Operations & Audit (6 tables - was 5)
**Purpose:** Is ACM working? What changed?

| Table | Purpose | Key Question | Status |
|-------|---------|--------------|--------|
| **ACM_Runs** | Execution tracking | Did ACM run successfully? | ✅ |
| **ACM_RunLogs** | Detailed logs | Why did ACM fail/NOOP? | ✅ |
| **ACM_RunTimers** | Performance profiling | Where is ACM slow? | ✅ |
| **ACM_Config** | Current configuration | What's configured? | ✅ |
| **ACM_ConfigHistory** | Configuration changes | What changed and when? | ✅ |
| **ACM_RunMetrics** | Fusion quality metrics | How well are detectors working together? | **NEW** |

### TIER 6: Advanced Analytics (5 tables - was 3)
**Purpose:** Deep insights and patterns

| Table | Purpose | Key Question | Status |
|-------|---------|--------------|--------|
| **ACM_RegimeOccupancy** | Operating mode utilization | How much time in each mode? | ✅ |
| **ACM_RegimeTransitions** | Mode switching patterns | How do modes transition? | ✅ |
| **ACM_ContributionTimeline** | Historical sensor attribution | Which sensors historically cause issues? | ✅ |
| **ACM_RegimePromotionLog** | Regime maturity evolution | How stable are operating modes? | **NEW** |
| **ACM_DriftController** | Drift detection control | How sensitive is drift detection? | **NEW** |

### TIER 7: V11 NEW FEATURES (5 tables - NEW)
**Purpose:** Advanced v11.0.0 capabilities

| Table | Purpose | Key Question | Status |
|-------|---------|--------------|--------|
| **ACM_RegimeDefinitions** | Regime centroids and metadata | What does each operating mode represent? | **CRITICAL** |
| **ACM_ActiveModels** | Active model versions | Which model version is active? | **NEW** |
| **ACM_DataContractValidation** | Pipeline entry validation | Did input data pass validation? | **NEW** |
| **ACM_SeasonalPatterns** | Seasonal pattern detection | Does equipment have seasonal behavior? | **NEW** |
| **ACM_AssetProfiles** | Asset similarity profiles | Which equipment is similar? | **NEW** |

---

## Critical Additions from Code Analysis

### Why These Were Missing Initially:

**ACM_RegimeDefinitions** - **CRITICAL OVERSIGHT**
- Written by `core/regime_definitions.py`
- Referenced by `core/regime_manager.py`, `core/regime_evaluation.py`
- **Impact if missing:** Regime detection broken, no regime context available
- **Status:** Now in TIER 7

**ACM_RunMetadata, ACM_RunMetrics** - Operational Visibility
- Written by `core/run_metadata_writer.py`, `core/acm_main.py`
- Tracks run context and fusion quality
- **Impact if missing:** Cannot reconstruct runs, no quality tracking
- **Status:** Now in TIER 4 and TIER 5

**ACM_RefitRequests** - Model Lifecycle
- Written by `core/acm_main.py`, `core/config_history_writer.py`
- Tracks when models need retraining
- **Impact if missing:** Model freshness unknown, stale models possible
- **Status:** Now in TIER 4

**ACM_SensorCorrelations** - Multivariate Analysis
- Written by `core/multivariate_forecast.py`
- Correlation matrix for sensor relationships
- **Impact if missing:** Multivariate patterns unavailable
- **Status:** Now in TIER 3

**ACM_FeatureDropLog** - Data Quality
- Written by `core/acm_main.py`
- Logs dropped features and reasons
- **Impact if missing:** Data quality issues hidden
- **Status:** Now in TIER 3

**V11 Feature Tables** - Architecture Support
- Written by various v11 modules
- Support typed contracts, maturity lifecycle, seasonality
- **Impact if missing:** V11 features disabled
- **Status:** All 5 now in TIER 7

---

## Tables NOT Recommended

These exist in database but are NOT included in ALLOWED_TABLES based on functionality analysis:

### Redundant / Derivable (6 tables)
- **ACM_Scores_Long** - Redundant with Scores_Wide (just long format)
- **ACM_HealthHistogram** - Derivable from HealthTimeline aggregation
- **ACM_HealthZoneByPeriod** - Derivable from HealthTimeline aggregation
- **ACM_SensorAnomalyByPeriod** - Derivable from SensorDefects aggregation
- **ACM_DefectTimeline** - Redundant with SensorDefects over time
- **ACM_DefectSummary** - Derivable from SensorDefects aggregation

### Overly Specific / Niche (7 tables)
- **ACM_HealthForecast_Continuous** - Merged into HealthForecast
- **ACM_FailureHazard_TS** - Merged into FailureForecast (HazardRate column)
- **ACM_DetectorForecast_TS** - Too detailed, not actionable
- **ACM_ContributionCurrent** - Snapshot of SensorHotspots (latest query suffices)
- **ACM_ContributionTimeline** - Redundant with SensorHotspots over time
- **ACM_SensorHotspotTimeline** - Redundant with SensorHotspots over time
- **ACM_Anomaly_Events** - Redundant with Episodes

### Operational/Internal (8 tables)
- **ACM_ColdstartState** - Internal state, not user-facing
- **ACM_RefitRequests** - Internal orchestration
- **ACM_BaselineBuffer** - Internal buffer
- **ACM_HistorianData** - Raw data cache
- **ACM_SensorNormalized_TS** - Intermediate processing
- **ACM_PCA_Loadings** - Model internals
- **ACM_PCA_Metrics** - Model internals
- **ACM_PCA_Models** - Model internals

### Unimplemented / Deprecated (5 tables)
- **ACM_RegimeDefinitions** - New v11 feature, not yet implemented
- **ACM_ActiveModels** - New v11 feature, not yet implemented
- **ACM_ThresholdCrossings** - Not implemented
- **ACM_AlertAge** - Derivable from Episodes
- **ACM_SinceWhen** - Derivable from Episodes

---

## Coverage Analysis

### By User Type
| User Type | Tables Available | Coverage |
|-----------|-----------------|----------|
| **Operations Teams** | 8 tables | ✅ Complete |
| **Maintenance Teams** | 10 tables | ✅ Complete |
| **Engineering Teams** | 12 tables | ✅ Complete |
| **Analytics Teams** | 13 tables | ✅ Complete |

### By ACM Capability
| Capability | Supporting Tables | Coverage |
|------------|------------------|----------|
| **Health Monitoring** | HealthTimeline, Scores_Wide | ✅ Complete |
| **Anomaly Detection** | Episodes, EpisodeDiagnostics | ✅ Complete |
| **Failure Prediction** | RUL, HealthForecast, FailureForecast | ✅ Complete |
| **Operating Context** | RegimeTimeline, RegimeOccupancy | ✅ Complete |
| **Drift Detection** | DriftSeries | ✅ Complete |
| **Root Cause** | SensorDefects, SensorHotspots, EpisodeCulprits | ✅ Complete |
| **Adaptive Learning** | AdaptiveConfig, ConfigHistory | ✅ Complete |
| **Quality Tracking** | DataQuality, CalibrationSummary | ✅ Complete |

---

## Implementation Status

### Already Implemented (~14 tables)
Based on code analysis, these are already being written:
- ACM_HealthTimeline
- ACM_RegimeTimeline
- ACM_Scores_Wide
- ACM_Episodes
- ACM_RUL
- ACM_HealthForecast
- ACM_FailureForecast
- ACM_SensorForecast
- ACM_DataQuality
- ACM_ForecastingState
- ACM_Runs
- ACM_RunTimers
- ACM_EpisodeCulprits
- ACM_Config

### Need Implementation (~11 tables)
- ACM_SensorDefects
- ACM_SensorHotspots
- ACM_EpisodeDiagnostics
- ACM_DetectorCorrelation
- ACM_AdaptiveConfig
- ACM_ConfigHistory
- ACM_RunLogs
- ACM_DriftSeries
- ACM_RegimeOccupancy
- ACM_RegimeTransitions
- ACM_CalibrationSummary

---

## Comparison to Previous Approaches

| Approach | Table Count | Principle | Pros | Cons |
|----------|------------|-----------|------|------|
| **Original** | 73 | Everything | Complete history | Too many, redundant |
| **Initial Cleanup** | 17 | Minimalist | Simple | Missing key outputs |
| **Dashboard-Based** | 42 | Dashboard coverage | 100% dashboard | Redundant, niche tables |
| **Functionality-Based** | 25 | User needs | Purpose-driven, complete | Requires new dashboards |

---

## References

**Full Requirements Document:**
- `docs/ACM_OUTPUT_REQUIREMENTS.md` - Complete functionality analysis

**ALLOWED_TABLES Definition:**
- `core/output_manager.py` lines 56-106

**SQL Schema:**
- `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md` - Full database schema

---

## Next Steps

1. ✅ Define required tables based on ACM functionality
2. ✅ Update ALLOWED_TABLES in output_manager.py
3. ⏭️ Implement remaining 11 tables
4. ⏭️ Design new dashboards based on these 25 tables
5. ⏭️ Deprecate unused tables from database

**Status**: Requirements defined, ready for implementation

---

## Proposed Database Views for Dashboarding

Views provide dashboard-friendly interfaces without additional storage.

**See `docs/ACM_OUTPUT_TABLES_REFINED.md` for complete view definitions and usage patterns.**

Five views proposed:
1. **ACM_CurrentHealth_View** - Latest health snapshot per equipment
2. **ACM_ActiveAnomalies_View** - Currently active anomalies with primary sensors
3. **ACM_LatestRUL_View** - Most recent RUL prediction per equipment  
4. **ACM_ProblematicSensors_View** - Combines defects + quality
5. **ACM_FleetSummary_View** - Fleet-wide health summary

---

## Updated Summary (Post-Feedback Refinement)

**Final Table Count:** 29 tables (up from 25)

**Tables Added Based on Feedback:**
- ACM_BaselineBuffer (progressive learning, coldstart)
- ACM_HistorianData (performance, caching)
- ACM_SensorNormalized_TS (time-series analysis)
- ACM_ContributionTimeline (historical attribution patterns)

**Database Views:** 5 proposed for dashboarding efficiency

**Status:** ✅ Refined based on crux analysis - ready for implementation

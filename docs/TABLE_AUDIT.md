# ACM Database Table Audit

**Last Updated:** 2024-12-25  
**Version:** 11.0.0  
**Approach:** Functionality-based, focused on ACM's core mission  
**Status:** ✅ REFINED BASED ON CRUX ANALYSIS

## Summary
**Design Principle**: Tables chosen based on **4 fundamental questions ACM must answer**:
1. What is current health?
2. If not healthy, what's the reason?
3. What will future health look like?
4. What will cause future degradation?

**Recommended Table Set**: 29 core tables organized in 6 functional tiers  
**Previous Version**: 25 tables  
**Increase**: +16% (4 new tables added based on feedback)

**Key Refinements**:
- Re-added ACM_BaselineBuffer (critical for progressive learning)
- Re-added ACM_HistorianData (performance, caching)
- Re-added ACM_SensorNormalized_TS (analysis, patterns)
- Re-added ACM_ContributionTimeline (historical attribution)
- All additions support long-term storage and operational needs

---

## The Crux of ACM

ACM exists to answer 4 questions:

### 1. **What is current health?**
→ TIER 1 tables (6): HealthTimeline, Scores_Wide, Episodes, RegimeTimeline, SensorDefects, SensorHotspots

### 2. **If not healthy, what's the reason?**
→ TIER 3 tables (4): EpisodeCulprits, EpisodeDiagnostics, DetectorCorrelation, DriftSeries

### 3. **What will future health look like?**
→ TIER 2 tables (4): RUL, HealthForecast, FailureForecast, SensorForecast

### 4. **What will cause future degradation?**
→ TIER 2 + TIER 3 tables (combined): SensorForecast, DriftSeries, EpisodeCulprits

---

## ALLOWED_TABLES (29 tables) - Crux-Focused

See `docs/ACM_OUTPUT_TABLES_REFINED.md` for complete analysis and rationale.

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

### TIER 3: Root Cause (4 tables)
**Answers:** "Why is this happening?" (current + future)

| Table | Purpose | Key Question |
|-------|---------|--------------|
| **ACM_EpisodeCulprits** | What caused each episode | Why did this anomaly occur? |
| **ACM_EpisodeDiagnostics** | Episode details/severity | How bad was this episode? |
| **ACM_DetectorCorrelation** | Inter-detector relationships | Are models working together? |
| **ACM_DriftSeries** | Behavior changes | When did behavior change? |

### TIER 4: Data & Model Management (7 tables)
**Purpose:** Long-term storage, progressive learning

| Table | Purpose | Why Critical |
|-------|---------|--------------|
| **ACM_BaselineBuffer** | Raw sensor data accumulation | Enables coldstart, progressive training |
| **ACM_HistorianData** | Cached historian data | Performance, reduces DB load |
| **ACM_SensorNormalized_TS** | Normalized sensor values | Analysis, pattern detection |
| **ACM_DataQuality** | Input data health | Data reliability tracking |
| **ACM_ForecastingState** | Forecast model state | Continuous forecast evolution |
| **ACM_CalibrationSummary** | Model quality over time | Model performance tracking |
| **ACM_AdaptiveConfig** | Auto-tuned configuration | Track what ACM learned |

### TIER 5: Operations & Audit (5 tables)
**Purpose:** Is ACM working? What changed?

| Table | Purpose | Key Question |
|-------|---------|--------------|
| **ACM_Runs** | Execution tracking | Did ACM run successfully? |
| **ACM_RunLogs** | Detailed logs | Why did ACM fail/NOOP? |
| **ACM_RunTimers** | Performance profiling | Where is ACM slow? |
| **ACM_Config** | Current configuration | What's configured? |
| **ACM_ConfigHistory** | Configuration changes | What changed and when? |

### TIER 6: Advanced Analytics (3 tables)
**Purpose:** Deep insights and patterns

| Table | Purpose | Key Question |
|-------|---------|--------------|
| **ACM_RegimeOccupancy** | Operating mode utilization | How much time in each mode? |
| **ACM_RegimeTransitions** | Mode switching patterns | How do modes transition? |
| **ACM_ContributionTimeline** | Historical sensor attribution | Which sensors historically cause issues? |

---

## Key Changes from Previous Version

### Added Back (4 tables):
1. **ACM_BaselineBuffer** - Critical for progressive model building and coldstart
2. **ACM_HistorianData** - Performance optimization through data caching
3. **ACM_SensorNormalized_TS** - Processed sensor data for time-series analysis
4. **ACM_ContributionTimeline** - Historical attribution patterns (vs current snapshot)

### Why These Were Re-Added:

**ACM_BaselineBuffer**
- **Removal reason:** "Internal operational"
- **Why critical:** Enables progressive model building, coldstart support, data accumulation across runs
- **User question:** "How much training data has ACM collected?"
- **Long-term storage:** YES

**ACM_HistorianData**
- **Removal reason:** "Raw data cache"
- **Why critical:** Reduces historian load, improves performance, enables offline analysis
- **User question:** "Is ACM using cached or fresh data?"
- **Long-term storage:** YES

**ACM_SensorNormalized_TS**
- **Removal reason:** "Intermediate processing"
- **Why critical:** Normalized values needed for correlations, pattern analysis, trend detection
- **User question:** "What are the normalized sensor trends?"
- **Long-term storage:** YES

**ACM_ContributionTimeline**
- **Removal reason:** "Redundant with SensorHotspots"
- **Why critical:** Historical patterns (timeline) vs current state (hotspots) - different purposes
- **User question:** "Which sensors consistently cause problems over time?"
- **Long-term storage:** YES

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

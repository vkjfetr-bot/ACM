# ACM Database Table Audit

**Last Updated:** 2024-12-25  
**Version:** 11.0.0  
**Approach:** Functionality-based (NOT dashboard-based)  
**Status:** ✅ REQUIREMENTS DEFINED

## Summary
**Design Principle**: Tables chosen based on **what ACM does** and **what users need to see**, not based on existing dashboards.

**Recommended Table Set**: 25 core tables organized in 6 functional tiers  
**Current ALLOWED_TABLES**: 25 tables (expanded from 17)  
**Increase**: +47% (8 new tables added)

**Key Changes from Dashboard-Based Approach**:
- Removed redundant/derivable tables (17 tables)
- Removed overly specific tables (5 tables)  
- Removed internal operational tables (8 tables)
- Focus on user-facing, actionable outputs
- Each table serves clear, non-redundant purpose

---

## Design Philosophy

### What Changed
**Previous Approach (Dashboard-Based)**:
- Started with existing dashboards
- Identified 26 tables needed by 6 dashboards
- Expanded to 42 tables to achieve 100% coverage
- Result: Many redundant/niche tables

**New Approach (Functionality-Based)**:
- Analyzed ACM's 8 core capabilities
- Identified what operations/maintenance/engineering teams need
- Designed 25 tables to provide complete visibility
- Dashboards will be built later to visualize this data

### Why This is Better
✅ **Purpose-driven**: Every table has clear user value  
✅ **Complete**: Covers all ACM functionality  
✅ **Non-redundant**: No duplicate information  
✅ **Actionable**: Enables specific decisions  
✅ **Future-proof**: Not tied to current dashboard limitations  

---

## ALLOWED_TABLES (25 tables) - Functionality-Based

See `docs/ACM_OUTPUT_REQUIREMENTS.md` for complete rationale and detailed specifications.

### TIER 1: Real-Time State (4 tables)
**Purpose**: Current equipment status

| Table | Purpose | Updates | Key Users |
|-------|---------|---------|-----------|
| **ACM_HealthTimeline** | Continuous health monitoring | Every run | Operations, Maintenance |
| **ACM_Scores_Wide** | All 6 detector scores | Every run | Engineering, Analytics |
| **ACM_Episodes** | Anomaly event tracking | When detected | Operations |
| **ACM_RegimeTimeline** | Operating mode context | Every run | Analytics |

### TIER 2: Predictive Intelligence (4 tables)
**Purpose**: Future state predictions

| Table | Purpose | Updates | Key Users |
|-------|---------|---------|-----------|
| **ACM_RUL** | Remaining Useful Life | Every successful run | Maintenance |
| **ACM_HealthForecast** | Projected health trajectory | Every forecast run | Operations, Maintenance |
| **ACM_FailureForecast** | Failure probability curves | Every forecast run | Maintenance, Management |
| **ACM_SensorForecast** | Physical sensor predictions | Every forecast run | Process engineers |

### TIER 3: Root Cause & Diagnostics (5 tables)
**Purpose**: Why problems occur

| Table | Purpose | Updates | Key Users |
|-------|---------|---------|-----------|
| **ACM_SensorDefects** | Sensor-level anomaly flags | Every run | Maintenance |
| **ACM_SensorHotspots** | Top culprit sensors | When anomalies | Maintenance |
| **ACM_EpisodeCulprits** | Per-episode attribution | When episodes | Analytics |
| **ACM_EpisodeDiagnostics** | Episode details | When episodes | Engineering |
| **ACM_DetectorCorrelation** | Inter-detector relationships | Periodic | Analytics, Engineering |

### TIER 4: System Operations (5 tables)
**Purpose**: Is ACM working correctly?

| Table | Purpose | Updates | Key Users |
|-------|---------|---------|-----------|
| **ACM_Runs** | Execution tracking | Every run | Engineering |
| **ACM_DataQuality** | Input data health | Every run | Engineering |
| **ACM_ForecastingState** | Forecast model persistence | Forecast runs | Internal |
| **ACM_AdaptiveConfig** | Auto-tuned settings | When tuned | Engineering |
| **ACM_RunTimers** | Performance profiling | Every run | Engineering |

### TIER 5: Configuration & Audit (3 tables)
**Purpose**: How is ACM configured?

| Table | Purpose | Updates | Key Users |
|-------|---------|---------|-----------|
| **ACM_Config** | Current configuration | Manual/sync | Engineering |
| **ACM_ConfigHistory** | Configuration changes | On change | Compliance, Engineering |
| **ACM_RunLogs** | Detailed execution logs | During run | Engineering |

### TIER 6: Advanced Analytics (4 tables)
**Purpose**: Deep analytical insights

| Table | Purpose | Updates | Key Users |
|-------|---------|---------|-----------|
| **ACM_DriftSeries** | Behavior change tracking | Every run | Analytics |
| **ACM_RegimeOccupancy** | Operating mode stats | Periodic | Analytics |
| **ACM_RegimeTransitions** | Mode switching patterns | Periodic | Analytics |
| **ACM_CalibrationSummary** | Model quality metrics | After fitting | Engineering |

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

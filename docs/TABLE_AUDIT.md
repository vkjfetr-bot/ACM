# ACM Database Table Audit

**Last Updated:** 2024-12-25  
**Version:** 11.0.0  
**Status:** ✅ AUDIT COMPLETE - TABLES EXPANDED

## Summary
**Original Database**: 73 ACM tables  
**After Initial Cleanup**: 17 tables in ALLOWED_TABLES  
**Post-Audit (Current)**: 42 tables in ALLOWED_TABLES ✅  

**Audit Results**:
- **Dashboard Coverage**: 100% (26/26 tables) ✅ (was 42%)
- **Operational Visibility**: EXCELLENT ✅ (was POOR)
- **Diagnostic Depth**: COMPREHENSIVE ✅ (was LIMITED)
- **Tables Restored**: +25 critical tables

**Key Finding**: Initial cleanup went too far (77% reduction), eliminating critical operational visibility and breaking 58% of dashboard functionality.

## ALLOWED_TABLES (42 tables) - Post-Audit 2024-12-25

These tables are actively written by the ACM pipeline (defined in `core/output_manager.py`):

### TIER 1: Core Pipeline Output (4 tables)
| Table Name | Purpose | Dashboard Usage |
|------------|---------|-----------------|
| **ACM_Scores_Wide** | Per-timestamp detector z-scores (PRIMARY OUTPUT) | acm_asset_story, acm_behavior |
| **ACM_HealthTimeline** | Health % over time (REQUIRED for RUL forecasting) | acm_asset_story, acm_behavior, acm_fleet_overview |
| **ACM_Episodes** | Detected anomaly episodes with diagnostics | acm_behavior, acm_fleet_overview |
| **ACM_RegimeTimeline** | Operating regime assignments | acm_asset_story, acm_behavior |

### TIER 2: Forecasting (7 tables)
| Table Name | Purpose | Dashboard Usage |
|------------|---------|-----------------|
| **ACM_RUL** | Remaining Useful Life with confidence bounds | acm_asset_story, acm_behavior, acm_fleet_overview, acm_forecasting |
| **ACM_HealthForecast** | Projected health trajectory (discrete) | acm_forecasting |
| **ACM_HealthForecast_Continuous** | Continuous health forecast | acm_asset_story, acm_forecasting |
| **ACM_FailureForecast** | Failure probability over time | acm_forecasting |
| **ACM_FailureHazard_TS** | Failure hazard rate time series | acm_forecasting |
| **ACM_SensorForecast** | Physical sensor value forecasts | acm_forecasting |
| **ACM_DetectorForecast_TS** | Per-detector forecast time series | acm_forecasting |

### TIER 3: Operational (8 tables)
| Table Name | Purpose | Dashboard Usage |
|------------|---------|-----------------|
| **ACM_Runs** | Run metadata and status | acm_behavior, acm_fleet_overview, acm_forecasting, acm_operations_monitor, acm_performance_monitor |
| **ACM_RunLogs** | Pipeline logs for debugging | acm_operations_monitor |
| **ACM_RunTimers** | Performance metrics per pipeline stage | acm_operations_monitor, acm_performance_monitor |
| **ACM_DataQuality** | Data quality per sensor | acm_fleet_overview |
| **ACM_ForecastingState** | Persistent forecasting model state | - |
| **ACM_AdaptiveConfig** | Dynamic per-equipment configuration | - |
| **ACM_ColdstartState** | Coldstart progression tracking | acm_operations_monitor |
| **ACM_RefitRequests** | Model refit request tracking | acm_operations_monitor |

### TIER 4: Diagnostics (14 tables)
| Table Name | Purpose | Dashboard Usage |
|------------|---------|-----------------|
| **ACM_SensorDefects** | Sensor-level anomaly flags | acm_asset_story |
| **ACM_SensorHotspots** | Top anomalous sensors (for RUL attribution) | - |
| **ACM_SensorHotspotTimeline** | Sensor anomaly trends over time | acm_asset_story |
| **ACM_EpisodeCulprits** | Per-episode sensor culprits | - |
| **ACM_EpisodeDiagnostics** | Per-episode diagnostic details | acm_behavior |
| **ACM_EpisodeMetrics** | Episode-level aggregate metrics | - |
| **ACM_Anomaly_Events** | Structured anomaly event records | acm_asset_story, acm_behavior |
| **ACM_ContributionCurrent** | Current sensor contribution scores | acm_asset_story, acm_behavior |
| **ACM_ContributionTimeline** | Historical sensor contributions | acm_behavior |
| **ACM_DefectSummary** | Aggregated defect statistics | - |
| **ACM_ThresholdCrossings** | Threshold violation events | - |
| **ACM_AlertAge** | How long alerts have been active | - |
| **ACM_RegimeDefinitions** | Regime centroids and metadata (v11) | - |
| **ACM_ActiveModels** | Active model versions per equipment (v11) | - |

### TIER 5: Analytics (9 tables)
| Table Name | Purpose | Dashboard Usage |
|------------|---------|-----------------|
| **ACM_DriftSeries** | Drift detection time series | acm_behavior |
| **ACM_RegimeOccupancy** | Time spent in each operating regime | acm_behavior |
| **ACM_RegimeTransitions** | Regime switching patterns | - |
| **ACM_RegimeDwellStats** | Regime duration statistics | - |
| **ACM_RegimeStability** | Regime stability metrics | - |
| **ACM_HealthZoneByPeriod** | Aggregated health by time period | acm_fleet_overview |
| **ACM_DetectorCorrelation** | Cross-detector correlation analysis | - |
| **ACM_CalibrationSummary** | Detector calibration status | - |
| **ACM_FeatureDropLog** | Features removed during pipeline | - |

---

## Orphaned Tables (31 tables in DB, not in ALLOWED_TABLES)

These tables exist in the database but are NOT actively written by the current pipeline. They may contain stale data or be candidates for removal.

### Low-Value / Specialized Tables (Consider for deprecation)
| Table Name | Status | Notes |
|------------|--------|-------|
| **ACM_BaselineBuffer** | Low priority | Buffer for baseline accumulation |
| **ACM_Config** | Keep | Runtime configuration (static) |
| **ACM_ConfigHistory** | Keep | Configuration change tracking |
| **ACM_DailyFusedProfile** | Low priority | Daily health profiles |
| **ACM_DefectTimeline** | Low priority | Defect trends (can derive from Scores_Wide) |
| **ACM_DriftEvents** | Low priority | Event markers (have DriftSeries) |
| **ACM_EpisodesQC** | Low priority | Quality control flags |
| **ACM_ForecastState** | Low priority | Superseded by ForecastingState? |
| **ACM_FusionQualityReport** | Medium priority | Fusion algorithm metrics |
| **ACM_HealthDistributionOverTime** | Low priority | Distribution trends |
| **ACM_HealthHistogram** | Low priority | Distribution snapshots |
| **ACM_HistorianData** | Keep | Raw historian cache |
| **ACM_OMRContributionsLong** | Medium priority | OMR sensor contributions |
| **ACM_OMRTimeline** | Medium priority | OMR trends |
| **ACM_OMR_Diagnostics** | Medium priority | OMR diagnostics |
| **ACM_PCA_Loadings** | Medium priority | PCA component loadings |
| **ACM_PCA_Metrics** | Medium priority | PCA model quality |
| **ACM_PCA_Models** | Medium priority | PCA model metadata |
| **ACM_RegimeState** | Low priority | Current regime (can query RegimeTimeline) |
| **ACM_RegimeStats** | Medium priority | Regime-specific statistics |
| **ACM_Regime_Episodes** | Low priority | Episodes by regime |
| **ACM_RUL_LearningState** | Medium priority | RUL model state |
| **ACM_RunMetadata** | Keep | Extended run metadata |
| **ACM_RunMetrics** | Keep | Run performance metrics |
| **ACM_Run_Stats** | Keep | Run-level statistics |
| **ACM_SchemaVersion** | Keep | Schema versioning |
| **ACM_Scores_Long** | Remove | Redundant with Scores_Wide |
| **ACM_SensorAnomalyByPeriod** | Low priority | Aggregated anomalies |
| **ACM_SensorNormalized_TS** | Keep | Normalized sensor time series |
| **ACM_SensorRanking** | Low priority | Sensor importance |
| **ACM_SinceWhen** | Low priority | Alert timestamps |
| **ACM_TagEquipmentMap** | Keep | Sensor tag mapping |
| **ACM_ThresholdMetadata** | Keep | Threshold configuration |

### Core Infrastructure (KEEP)
| Table Name | Purpose |
|------------|---------|
| **Equipment** | Equipment registry |
| **ModelRegistry** | SQL-based model persistence |

---

## Key Findings from 2024-12-25 Audit

### What Was Lost in Over-Aggressive Cleanup (17 tables → too few)

**Critical Operational Visibility Lost:**
- ❌ ACM_Runs - No run tracking (affected 5 dashboards)
- ❌ ACM_RunLogs - No pipeline diagnostics
- ❌ ACM_RunTimers - No performance monitoring
- ❌ ACM_ColdstartState - No coldstart visibility
- ❌ ACM_RefitRequests - No refit tracking

**Sensor Analytics Lost:**
- ❌ ACM_ContributionCurrent/Timeline - No root cause analysis
- ❌ ACM_SensorHotspotTimeline - No sensor trend analysis

**Regime & Drift Analytics Lost:**
- ❌ ACM_DriftSeries - No drift visualization
- ❌ ACM_RegimeOccupancy - No regime statistics
- ❌ ACM_RegimeTransitions/DwellStats/Stability - No regime analytics
- ❌ ACM_HealthZoneByPeriod - No fleet health aggregates

**Episode & Defect Tracking Lost:**
- ❌ ACM_Anomaly_Events - No event-based tracking
- ❌ ACM_EpisodeMetrics - No episode aggregates
- ❌ ACM_DefectSummary - No defect statistics
- ❌ ACM_ThresholdCrossings - No violation events
- ❌ ACM_AlertAge - No alert aging

**Forecasting Details Lost:**
- ❌ ACM_DetectorForecast_TS - No detector-level forecasts
- ❌ ACM_FailureHazard_TS - No hazard rate visualization
- ❌ ACM_HealthForecast_Continuous - Gaps in forecasts

**Model Diagnostics Lost:**
- ❌ ACM_DetectorCorrelation - No quality metrics
- ❌ ACM_CalibrationSummary - No calibration tracking
- ❌ ACM_FeatureDropLog - No feature engineering visibility

### Audit Results

**Before Audit:**
- 17 tables in ALLOWED_TABLES
- 42% dashboard coverage (11/26 tables)
- POOR operational visibility
- LIMITED diagnostic capabilities

**After Audit (Current):**
- 42 tables in ALLOWED_TABLES (+147% increase)
- 100% dashboard coverage (26/26 tables) ✅
- EXCELLENT operational visibility ✅
- COMPREHENSIVE diagnostic capabilities ✅

**Optimal Balance:** ~40-45 tables (not 17, not 73)

---

## References

**Full Audit Report:**
- `docs/ACM_TABLE_ANALYTICS_AUDIT.md` - Comprehensive analysis

**ALLOWED_TABLES Definition:**
- `core/output_manager.py` lines 56-108 (expanded from 17 to 42 tables)

**Grafana Dashboards:**
- 6 active dashboards requiring 26 unique tables (now all supported)

**Model Persistence:**
- `core/model_persistence.py` uses ModelRegistry table

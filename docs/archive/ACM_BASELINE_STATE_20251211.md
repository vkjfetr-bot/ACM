# ACM Baseline State Snapshot – 2025-12-11

**Date**: 2025-12-11 09:23:24 UTC (Latest successful run)  
**ACM Version**: 10.0.0 (released 2025-12-08)  
**Purpose**: Document the complete baseline state of ACM before any major changes or improvements.

---

## Executive Summary

ACM is currently operational and successfully executing end-to-end analytics pipelines. The most recent run (RunID: `dd26755a-6ae3-4ffc-a2bc-063332340eb9`) completed successfully with:

- **Data Window**: 2024-07-21 09:24:00 to 2024-09-29 11:44:59 (71 days)
- **Samples Processed**: 1,443 records
- **Tables Written**: 35 comprehensive analytics tables
- **Episodes Detected**: 14 anomaly episodes
- **Status**: SUCCESS (all SQL writes completed, no errors)

---

## System Architecture

### Core Version & Release

| Aspect | Value |
|--------|-------|
| **Current Version** | 10.0.0 |
| **Release Date** | 2025-12-08 |
| **Versioning Model** | Semantic (MAJOR.MINOR.PATCH) |
| **Git Branching** | feature/fix/refactor/docs branches → main |
| **Release Tags** | Annotated git tags (v10.0.0, etc.) |

### Key Architectural Components

| Component | Status | Details |
|-----------|--------|---------|
| **Orchestrator** | Active | `core/acm_main.py` - Entry point for all runs |
| **Output Manager** | Active | `core/output_manager.py` - All CSV/PNG/SQL writes |
| **Detectors** | Active | 8 detectors: AR1, PCA-SPE, PCA-T2, Mahalanobis, IForest, GMM, OMR, CUSUM |
| **Fusion Engine** | Active | `core/fuse.py` - Combines 7 detectors (river_hst_z missing) |
| **Regime Analysis** | Active | `core/regimes.py` - Identifies operational states |
| **Drift Detection** | Active | `core/drift.py` - Monitors model degradation |
| **Episode Detection** | Active | `core/episode_culprits_writer.py` - 14 episodes found in last run |
| **SQL Bridge** | Active | `core/sql_client.py` - All data persisted to SQL |
| **Fast Features** | Active | `core/fast_features.py` - Vectorized pandas calculations |

### Operational Modes

| Mode | Status | Entry Point |
|------|--------|-------------|
| **File Mode** | Available | `scripts/run_file_mode.ps1` + `data/*.csv` |
| **SQL Mode** | **CURRENT** | `core/sql_client.SQLClient` via `configs/sql_connection.ini` |
| **Batch Mode** | Available | `python scripts/sql_batch_runner.py` (sets `ACM_BATCH_MODE` env var) |

---

## Database State

### Tables Overview

**Total Tables**: 91 (schema exported 2025-12-11 09:56:42)

### Tier-A Core Analytics Tables (Comprehensive)

| Category | Tables | Status | Last Updated |
|----------|--------|--------|---------------|
| **Scores** | ACM_Scores_Wide, ACM_Scores_Long | 1,443-8,658 rows | 2025-12-11 |
| **Health** | ACM_HealthTimeline, ACM_HealthDistribution OverTime, ACM_HealthZoneByPeriod | 1,443-722 rows | 2025-12-11 |
| **Regimes** | ACM_RegimeTimeline, ACM_RegimeOccupancy, ACM_RegimeTransitions, ACM_RegimeDwellStats | 1,443-2 rows | 2025-12-11 |
| **Episodes** | ACM_EpisodeDiagnostics, ACM_EpisodeMetrics, ACM_EpisodesQC | 14-1 rows | 2025-12-11 |
| **Detectors** | ACM_CalibrationSummary, ACM_DetectorCorrelation, ACM_FusionQualityReport | 8-28 rows | 2025-12-11 |
| **OMR** | ACM_OMRContributionsLong, ACM_OMRTimeline | 129,870-1,443 rows | 2025-12-11 |

### Tier-C Time Series Tables (Per-Detector)

| Category | Tables | Status | Last Updated |
|----------|--------|--------|---------------|
| **Sensor Data** | ACM_SensorNormalized_TS, ACM_SensorHotspots, ACM_SensorHotspotTimeline, ACM_SensorAnomalyByPeriod | 12,987-1,690 rows | 2025-12-11 |
| **Defects** | ACM_SensorDefects, ACM_DefectSummary, ACM_DefectTimeline | 8-409 rows | 2025-12-11 |
| **Anomalies** | ACM_Anomaly_Events, ACM_ThresholdCrossings | 14-41 rows | 2025-12-11 |
| **Drift** | ACM_DriftSeries | 1,443 rows | 2025-12-11 |

### Tier-B Forecasting Tables (Currently NOOP)

| Table | Status | Reason |
|-------|--------|--------|
| ACM_HealthForecast_TS | **SKIPPED** | Warning: No health data in last 2160 hours |
| ACM_FailureForecast_TS | **SKIPPED** | Forecast engine skipped (no health data) |
| ACM_RUL | **NOT POPULATED** | Requires health forecast to generate |
| ACM_SensorForecast_TS | **NOT POPULATED** | Requires valid forecasts |

**Root Cause**: The run only processed 71 days of historical data (2024-07-21 to 2024-09-29). The health timeline data exists in the SQL table but may not be loaded correctly into the forecasting pipeline.

### Model Registry

| Component | Status | Details |
|-----------|--------|---------|
| **PCA Model** | Trained | 5 components, v10.1.0, 90 feature loadings written |
| **ModelRegistry** | Active | Tracks detector weights and versions |
| **Coldstart State** | Tracked | ACM_ColdstartState records enabled |

---

## Configuration State

### Primary Config File

**File**: `configs/config_table.csv`

**Global Settings** (rows with `*` equipment):
- `fusion.weights.omr_z`: 0.1 (OMR detector weight in fused score)
- `fusion.weights.ar1_z`: 0.2
- `fusion.weights.pca_spe_z`, `pca_t2_z`, `mhal_z`, `iforest_z`, `gmm_z`: 0.1-0.15 each
- `river_hst_z`: Missing (not in fusion stream)

**Equipment-Specific Overrides**: Per-equipment rows for GAS_TURBINE, FD_FAN override globals

### SQL Connection

**File**: `configs/sql_connection.ini`

```ini
[acm]
server = localhost\B19CL3PCQLSERVER
database = ACM
trusted_connection = yes
driver = ODBC Driver 18 for SQL Server
TrustServerCertificate = yes
```

**Status**: Connected and operational (latest run successful)

---

## Last Successful Run Details

### Run Metadata

| Field | Value |
|-------|-------|
| **RunID** | dd26755a-6ae3-4ffc-a2bc-063332340eb9 |
| **EquipID** | 1 (FD_FAN) |
| **Started** | 2025-12-11 03:53:14 |
| **Completed** | 2025-12-11 03:53:24 |
| **Duration** | 10.37 seconds |
| **Status** | SUCCESS |

### Data Ingestion

| Metric | Value |
|--------|-------|
| **Window Start** | 2024-07-21 09:24:00 |
| **Window End** | 2024-09-29 11:44:59 |
| **Duration** | 71 days (1,715 hours) |
| **Samples** | 1,443 records (30-min cadence) |
| **Sensors** | 9 active sensors |

### Processing Results

| Stage | Records | Status |
|-------|---------|--------|
| **Loaded** | 1,443 | All cadence OK (100%) |
| **Kept** | 1,443 | No NaN drops |
| **PCA Features** | 90 | 5 components × 18 features |
| **Detectors** | 1,443 rows × 8 detectors | All computed |
| **Fusion** | 1,443 fused scores | Dynamic normalization applied |
| **Regimes** | 2 clusters detected | Transient: trip=1443 |
| **Episodes** | 14 episodes | Peak Z=3.06, Avg Z=1.33 |

### Detector Performance

| Detector | Mean Z | Max Z | P95 | Present | Weight |
|----------|--------|-------|-----|---------|--------|
| AR1 (Time Series) | 1.038 | 10.00 | 3.014 | Yes | 0.20 |
| PCA-SPE | -0.162 | 2.564 | 0.701 | Yes | 0.15 |
| PCA-T2 | 0.298 | 3.896 | 0.957 | Yes | 0.15 |
| Mahalanobis | 0.904 | 10.00 | 2.401 | Yes | 0.15 |
| IForest | 0.563 | 6.206 | 1.428 | Yes | 0.10 |
| GMM | 0.727 | 5.918 | 1.793 | Yes | 0.10 |
| OMR (Residual) | 0.028 | 1.976 | 0.204 | Yes | 0.10 |
| CUSUM | 0.094 | 10.00 | 0.387 | Yes | 0.05 |

**Note**: river_hst_z missing from stream; dynamic re-weighting applied.

### Episode Summary

| Metric | Value |
|--------|-------|
| **Total Episodes** | 14 |
| **Peak Fused Z** | 3.06 |
| **Avg Fused Z** | 1.33 |
| **Max Duration** | 19 hours |
| **Median Duration** | 3.75 hours |
| **Dominant Detector** | Mahalanobis (multivariate distance) |

### Health Status (Latest State)

| Zone | Count | % |
|------|-------|-----|
| **GOOD** | 445 | 30.8% |
| **WATCH** | 411 | 28.5% |
| **ALERT** | 587 | 40.7% |

**Latest Health Index**: 71.6 (CAUTION zone - yellow alert)

### Adaptive Thresholds (Global)

| Threshold | Value | Method | Confidence |
|-----------|-------|--------|------------|
| **Alert** | 3.236 | quantile | q=0.997 |
| **Warn** | 1.618 | quantile | q=0.997 |
| **Samples Used** | 1,443 | Full accumulated | N/A |

---

## Known Issues & Limitations

### 1. Forecasting Disabled (Critical)

**Status**: Non-critical for analytical runs, but impacts predictive capabilities

**Issue**:
```
[FORECAST] Warning: No health timeline found for EquipID=1 in last 2160 hours
[FORECAST] Skipped: No health data available (quality=NONE)
```

**Root Cause**: Health data is in SQL tables but not loading into forecasting pipeline correctly.

**Impact**:
- No RUL predictions generated
- No failure forecasts computed
- No sensor forecasts available
- Grafana RUL/forecast panels show no data

**Fix Required**: Debug health data loader in `core/forecasting.py` - line that fetches ACM_HealthTimeline

### 2. Missing river_hst_z Stream

**Status**: Expected (river package optional)

**Impact**: Fusion engine dynamically re-weights remaining 7 detectors; quality reduced slightly but acceptable.

### 3. Per-Regime Thresholds Disabled

**Issue**: 
```
[REGIME] Clustering quality below threshold; per-regime thresholds disabled.
[AUTO-TUNE] Quality degradation detected: Anomaly rate too high, Silhouette score too low
```

**Cause**: Silhouette score too low (data too noisy or regimes poorly separated)

**Impact**: Uses global thresholds (3.236 alert, 1.618 warn) instead of per-regime. Acceptable for current data.

### 4. Coldstart Mode Active

**Status**: Expected for new equipment

- First 200 samples (min_train_samples) reserved for training
- Remaining samples scored for analytics
- PCA models refit on each new run (no persistence across runs)

---

## Performance Metrics

### Execution Timings

| Stage | Duration | Notes |
|-------|----------|-------|
| **Data Load** | 0.009s | SQL historian SP call |
| **Feature Engineering** | Fast | Vectorized pandas |
| **PCA Fit + Score** | 0.03s | 5 components, 90 features |
| **Detector Pipeline** | 0.04s | 8 detectors × 1,443 samples |
| **Fusion** | 0.089s | Dynamic weighting + CUSUM |
| **Regimes** | 0.157s | KMeans + transient detection |
| **Episodes** | 0.063s | CUSUM + culprit attribution |
| **Comprehensive Analytics** | 8.808s | 35 tables, 1.4M+ rows written |
| **Total** | ~10.37s | End-to-end pipeline |

**Throughput**: ~139 samples/second (1,443 records in 10.37s)

---

## Code Quality & Testing

### Lint & Type Status

- **Linter**: ruff (configured, checked)
- **Type Checker**: mypy (available)
- **Test Coverage**: 
  - `tests/test_fast_features.py` - Vectorization tests
  - `tests/test_dual_write.py` - SQL + CSV writes
  - `tests/test_progress_tracking.py` - Batch monitoring

### Documentation

| Document | Status | Last Updated |
|----------|--------|---------------|
| README.md | Current | High-level overview |
| docs/ACM_SYSTEM_OVERVIEW.md | Current | Architecture walkthrough |
| docs/SOURCE_CONTROL_PRACTICES.md | Current | Git workflow guide |
| docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md | **Regenerated** | 2025-12-11 09:56:42 |
| docs/OMR_DETECTOR.md | Current | OMR algorithm details |
| docs/COLDSTART_MODE.md | Current | Warmup strategy |
| Grafana dashboards | Current | 5 main dashboards + drill-downs |

---

## Infrastructure & Dependencies

### Python Environment

| Component | Version | Status |
|-----------|---------|--------|
| **Python** | 3.11 | Current |
| **pandas** | Latest | Core data manipulation |
| **NumPy** | Latest | Vectorization |
| **scikit-learn** | Latest | PCA, KMeans, IForest |
| **pyodbc** | Latest | SQL Server connection |
| **Optional: Polars** | Available | Fast features backward compat |

### SQL Server

| Component | Value | Status |
|-----------|-------|--------|
| **Server** | localhost\B19CL3PCQLSERVER | Running |
| **Database** | ACM | 91 tables |
| **Driver** | ODBC 18 for SQL Server | Installed |
| **Auth** | Trusted (Windows) | Configured |

### Git & Version Control

| Aspect | Status | Details |
|--------|--------|---------|
| **Repository** | Ready | Main branch + feature branches |
| **Tags** | v10.0.0 | Latest production release |
| **Staging Area** | Clean | No uncommitted changes expected |

---

## Grafana Dashboard Status

### Main Dashboards

| Dashboard | Status | Key Panels | Last Data |
|-----------|--------|-----------|-----------|
| **Overview** | Ready | Health, Episodes, Regime | 2025-12-11 |
| **Detector Analysis** | Ready | Detector Z-scores, Correlations | 2025-12-11 |
| **Anomalies** | Ready | Episodes, Culprits, Timeline | 2025-12-11 |
| **Health Zones** | Ready | Zone distribution, Transitions | 2025-12-11 |
| **RUL Prediction** | **NO DATA** | Requires forecasting fix | N/A |

**Note**: RUL panel will populate once health forecast pipeline is fixed.

---

## Next Steps for Improvements

### High Priority

1. **Fix Health Forecasting Pipeline**
   - Debug health data loader in `core/forecasting.py`
   - Verify ACM_HealthTimeline is being read correctly
   - Test with recent data window
   - Enable RUL predictions & Grafana panels

2. **Enable Per-Regime Thresholds**
   - Improve regime clustering quality (reduce anomaly rate)
   - Auto-tune silhouette threshold
   - Per-regime alert/warn thresholds

3. **Add river_hst_z Stream (Optional)**
   - Integrate river anomaly detector if needed
   - Improve fusion ensemble quality

### Medium Priority

4. **Performance Optimization**
   - Profile comprehensive analytics (8.8s mostly SQL writes)
   - Batch operations better (currently auto-flush at 1,450 rows)
   - Consider Polars for 10,000+ row datasets

5. **Model Persistence**
   - Save/load PCA models across runs (avoid retraining)
   - Persist detector calibration data
   - Track model drift over time

### Low Priority

6. **Data Quality Improvements**
   - Increase coldstart threshold if more training data available
   - Fine-tune detector weights based on episode detection accuracy
   - Add sensor-specific anomaly contexts

---

## Appendices

### A. Schema Statistics

- **Total Tables**: 91
- **Total Columns**: ~1,200+ (varies by table)
- **Largest Tables**: 
  - ACM_SensorNormalized_TS: 12,987 rows
  - ACM_OMRContributionsLong: 129,870 rows
  - ACM_Scores_Long: 8,658 rows
- **Primary Keys**: Most tables (auto-enforced)
- **Indexes**: Key performance indexes on timestamp, RunID, EquipID

### B. Configuration Quick Reference

```bash
# Run ACM on FD_FAN equipment
python -m core.acm_main --equip FD_FAN

# Run batch from start
python scripts/sql_batch_runner.py --equip FD_FAN GAS_TURBINE --tick-minutes 1440 --max-workers 2 --start-from-beginning

# Export schema
python scripts/sql/export_comprehensive_schema.py --output docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md

# Verify SQL connection
python scripts/sql/verify_acm_connection.py
```

### C. Useful SQL Queries

```sql
-- Check latest run
SELECT TOP 1 RunID, EquipID, CreatedAt, Status 
FROM ACM_Runs 
ORDER BY CreatedAt DESC

-- Health timeline for latest run
SELECT TOP 100 Timestamp, HealthIndex, HealthZone, FusedZ 
FROM ACM_HealthTimeline 
WHERE EquipID=1 
ORDER BY Timestamp DESC

-- Recent episodes
SELECT episode_id, peak_timestamp, duration_h, severity, dominant_sensor 
FROM ACM_EpisodeDiagnostics 
WHERE EquipID=1 
ORDER BY peak_timestamp DESC

-- Detector summary
SELECT DetectorType, MeanZ, P95Z, SaturationPct 
FROM ACM_CalibrationSummary 
ORDER BY MeanZ DESC
```

---

## Document Control

| Field | Value |
|-------|-------|
| **Created** | 2025-12-11 |
| **Version** | 1.0 |
| **Baseline ACM Version** | 10.0.0 |
| **Purpose** | System state documentation before improvements |
| **Next Review** | After major changes (forecasting, thresholds, etc.) |

---

**End of Baseline Snapshot**

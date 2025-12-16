# ACM Grafana Dashboards

This folder contains production-ready Grafana dashboards for the ACM (Autonomous Condition Monitoring) system.

## Architecture: Two Dashboard Strategy

ACM uses a streamlined two-dashboard approach:

1. **User-Facing Story Dashboard** - Visual storytelling for asset health and failure prediction
2. **Operations Monitor** - Technical performance monitoring for ACM administrators

All legacy dashboards have been archived to `archive/` folder.

---

## Available Dashboards

### 1. ACM Asset Story (`acm_asset_story.json`)

**Purpose**: User-facing dashboard that tells the visual story of asset health, failure prediction, and fault diagnostics.

**Target Audience**: Plant operators, reliability engineers, asset managers

**Key Sections**:

| Section | Description |
|---------|-------------|
| **EXECUTIVE SUMMARY** | Health gauge (0-100), RUL countdown, failure prediction date, confidence score, status indicator |
| **FAULT DIAGNOSTICS** | Detector signals with fault-type mapping, sensor contribution table |
| **HEALTH FORECAST** | Health trend timeline with forecast overlay, RUL prediction details |
| **DETECTOR DEEP DIVE** | 6-detector Z-score timeseries (AR1, PCA-SPE, PCA-T2, IForest, GMM, OMR) |
| **OPERATING REGIMES** | Regime timeline, anomaly episodes table |
| **SENSOR HOTSPOTS** | Top contributing sensors over time |

**Fault-Type Mapping**:
The dashboard translates detector signals into operator-friendly fault types:

| Detector | Fault Type | Typical Cause |
|----------|-----------|---------------|
| AR1 | Sensor degradation | Drift, spikes, control loop issues |
| PCA-SPE | Mechanical coupling loss | Structural fatigue, decoupling |
| PCA-T2 | Process upset | Load imbalance, abnormal state |
| IForest | Novel failure mode | Rare transients, unknown events |
| GMM | Regime transition | Mode confusion, state instability |
| OMR | Baseline consistency | Fouling, wear, misalignment |

### 2. ACM Operations Monitor (`acm_operations_monitor.json`)

**Purpose**: Technical dashboard for monitoring ACM system performance, run statistics, and errors.

**Target Audience**: ACM administrators, DevOps, system engineers

**Key Sections**:

| Section | Description |
|---------|-------------|
| **SYSTEM OVERVIEW** | Total runs, success/error counts, avg duration, equipment count, data processed |
| **RUN PERFORMANCE** | Duration over time (bar chart), equipment statistics table |
| **EXECUTION BREAKDOWN** | Average time by section (horizontal bar), section timing over time (stacked) |
| **LOGS & ERRORS** | Recent error/warning logs with color-coded severity |
| **DATA QUALITY** | Model refit requests, coldstart status table |
| **RECENT RUNS** | Last 50 runs with status, duration, row counts |

---

## Prerequisites

- Grafana 12.0.0+
- SQL Server database with ACM tables (see `docs/sql/COMPREHENSIVE_SCHEMA_REFERENCE.md`)
- MSSQL data source configured in Grafana

## Key SQL Tables Used

### Asset Story Dashboard
- `ACM_HealthTimeline` - Health index and fused Z-score over time
- `ACM_RUL` - Remaining useful life predictions with confidence bounds
- `ACM_SensorDefects` - Per-sensor detector signals and severity
- `ACM_ContributionCurrent` - Current sensor contribution percentages
- `ACM_Scores_Wide` - All detector Z-scores in wide format
- `ACM_RegimeTimeline` - Operating regime labels
- `ACM_Anomaly_Events` - Anomaly episode events
- `ACM_SensorHotspotTimeline` - Sensor contribution timeline
- `ACM_HealthForecast_Continuous` - Forecasted health values
- `Equipment` - Equipment metadata

### Operations Monitor
- `ACM_Runs` - Run metadata (duration, status, rows processed)
- `ACM_RunTimers` - Section-level execution timing
- `ACM_RunLogs` - Error and warning logs
- `ACM_RefitRequests` - Model refit request history
- `ACM_ColdstartState` - Coldstart progress tracking
- `Equipment` - Equipment metadata

## Importing Dashboards

1. In Grafana, go to **Dashboards > Import**
2. Click **Upload dashboard JSON file**
3. Select `acm_asset_story.json` or `acm_operations_monitor.json`
4. Select your MSSQL data source
5. Click **Import**

## Variables

Both dashboards use these template variables:

| Variable | Description |
|----------|-------------|
| `datasource` | MSSQL data source selector |
| `equipment` | Equipment filter (includes "All Equipment" option) |

## Archive

Legacy dashboards are preserved in `archive/` for reference. These are no longer actively maintained:

- `acm_operator_dashboard.json` - Old operator view (replaced by Asset Story)
- `acm_forecasting_dashboard.json` - Standalone forecasting (merged into Asset Story)
- `acm_performance_monitor.json` - Old perf monitor (replaced by Operations Monitor)
- Various broken/backup versions

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 10.2.0 | 2025 | Two-dashboard architecture: Asset Story + Operations Monitor |
| 10.0.0 | 2025 | MHAL detector removed, 6-detector architecture |
| 9.x | 2024 | Multiple dashboard iterations, schema stabilization |


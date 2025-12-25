# ACM Grafana Dashboards

This folder contains production-ready Grafana dashboards for the ACM (Autonomous Condition Monitoring) system.

## Architecture: Comprehensive Dashboard Suite

ACM provides a multi-tiered dashboard suite for different user personas and use cases:

1. **Main Dashboard** (`acm_main_dashboard.json`) - ⭐ PRIMARY - Executive overview with comprehensive monitoring
2. **Sensor Deep-Dive** (`acm_sensor_deepdive.json`) - ⭐ NEW - Detailed sensor-level diagnostics
3. **Asset Story Dashboard** (`acm_asset_story.json`) - Visual storytelling for asset health
4. **Operations Monitor** (`acm_operations_monitor.json`) - ACM system performance monitoring

**Recommended Navigation Flow**:
```
Main Dashboard (Executive View)
    ↓ Drill-Down for Sensor Issues
Sensor Deep-Dive (Diagnostic View)
    ↓ Drill-Down for Full Story
Asset Story (Detailed Narrative)
    ↓ Technical Performance
Operations Monitor (System Health)
```

All legacy dashboards have been archived to `archive/` folder.

---

## Available Dashboards

### 1. ACM Main Dashboard (`acm_main_dashboard.json`) ⭐ NEW - PRIMARY DASHBOARD

**Purpose**: Comprehensive executive dashboard serving as the main entry point to ACM. Provides at-a-glance asset health monitoring with drill-down navigation to detailed dashboards.

**Target Audience**: All users - Executives, operators, reliability engineers, maintenance planners

**Design Philosophy**:
- **Minimal Cognitive Friction**: Information hierarchy from executive summary to detailed diagnostics
- **Industry Best Practices**: Consistent color palette, proper labeling, semantic colors
- **Drill-Down Navigation**: Links to specialized dashboards for deep-dive analysis
- **Real-Time Monitoring**: 30-second auto-refresh with live anomaly annotations

**Color Palette** (Standardized across ACM):

| Purpose | Color | Hex Code | Usage |
|---------|-------|----------|-------|
| Critical/Failure | Red | `#C4162A` | Health < 50%, RUL < 24h, Z-score > 5 |
| Warning | Orange | `#FF9830` | Health 50-70%, RUL 24-72h, Z-score 3-5 |
| Caution | Yellow | `#FADE2A` | Health 70-85%, RUL 72-168h, Z-score 2-3 |
| Healthy | Green | `#73BF69` | Health 85-95%, RUL > 168h, Z-score < 2 |
| Excellent | Blue | `#5794F2` | Health > 95%, forecasts, info |

**Detector-Specific Colors**:

| Detector | Color | Hex Code | Fault Type |
|----------|-------|----------|------------|
| AR1 | Red-Pink | `#E02F44` | Sensor degradation |
| PCA-SPE | Orange | `#FF9830` | Mechanical coupling loss |
| PCA-T² | Yellow | `#FADE2A` | Process upset |
| IForest | Purple | `#B877D9` | Novel failure modes |
| GMM | Blue | `#5794F2` | Regime confusion |
| OMR | Dark Green | `#37872D` | Baseline drift |

**Dashboard Sections**:

| Section | Panel Types | Purpose |
|---------|-------------|---------|
| **⚡ EXECUTIVE OVERVIEW** | Gauge, Stats, Bar Gauge | At-a-glance health, RUL, confidence, detector status |
| **📈 HEALTH & PREDICTION TRENDS** | Time Series, Table | Historical health with forecast horizon, RUL confidence bounds (P10/P50/P90) |
| **🔬 DETECTOR DEEP-DIVE** | Time Series | All 6 detector signals with color-coded fault types |
| **🎯 SENSOR DIAGNOSTICS** | Bar Chart, Table | Top sensor contributors, active defects with severity |
| **⚙️ OPERATING CONTEXT** | State Timeline | Operating regime transitions |
| **⚠️ ANOMALY EPISODES** | Table | Recent anomaly episodes with duration, severity, dominant detector |

**Key Features**:
- **Smart Thresholds**: Color-coded health states (Critical/Warning/Caution/Healthy/Excellent)
- **RUL Predictions**: Pessimistic (P10), Median (P50), Optimistic (P90) confidence bounds
- **Anomaly Annotations**: Automatic markers for anomaly episodes on time series
- **Detector Matrix**: Visual grid showing all 6 detector Z-scores with gradient backgrounds
- **Sensor Hotspots**: Horizontal bar chart of top contributing sensors
- **Live Updates**: 30-second refresh interval with smooth transitions

**Panel Highlights**:
1. **Asset Health Gauge**: 0-100% with 5-level color coding (red→orange→yellow→green→blue)
2. **RUL Stat**: Hours until intervention with context-aware colors (< 24h = red, > 7 days = green)
3. **Failure Date**: Predicted failure timestamp based on degradation trajectory
4. **Detector Status Matrix**: Horizontal bar gauge showing all 6 detectors with gradient fill
5. **Health Timeline**: Smooth line chart with forecast overlay and confidence bands
6. **Detector Signals**: 6-line time series with detector-specific colors and threshold lines
7. **Sensor Hotspots**: Bar chart sorted by contribution percentage
8. **Regime Timeline**: State timeline showing operating mode transitions

**Variables**:
- `$datasource`: MSSQL data source selector
- `$equipment`: Equipment filter (EquipID from Equipment table)

**Default Time Range**: Last 5 years (`now-5y` to `now`) to show full historical context

**Navigation**:
- Dropdown menu linking to all ACM dashboards (Asset Story, Operations Monitor, etc.)
- Uses `$equipment` and time range variables for consistent drill-down experience

---

### 2. ACM Sensor Deep-Dive (`acm_sensor_deepdive.json`) ⭐ NEW - DIAGNOSTIC DASHBOARD

**Purpose**: Detailed sensor-level diagnostic dashboard for deep-dive troubleshooting. Provides sensor contribution analysis, detector breakdown, OMR analysis, and sensor value forecasting.

**Target Audience**: Maintenance engineers, reliability engineers, diagnostics specialists

**Dashboard Sections**:

| Section | Panel Types | Purpose |
|---------|-------------|---------|
| **📊 SENSOR CONTRIBUTION ANALYSIS** | Stacked Area Chart | Timeline showing top 5 sensors' contribution to anomaly score |
| **🔍 DETECTOR BREAKDOWN BY SENSOR** | Heatmap Table | Matrix showing which detectors are firing for each sensor |
| **📈 SENSOR DEFECT STATISTICS** | Table | Comprehensive sensor ranking with defect frequency, max Z-score, last occurrence |
| **🎯 SENSOR VALUES & FORECASTS** | Time Series | Actual vs forecasted values for top 3 contributing sensors |
| **🔗 OMR ANALYSIS** | Table + Time Series | Sensor-to-sensor prediction contributions and residual timelines |

**Key Features**:
- **Stacked Area Chart**: Shows how sensor contributions evolve over time with 100% stacking
- **Detector Heatmap**: Color-coded matrix reveals which detectors identify issues with each sensor
- **Sensor Ranking**: Sortable table with defect count, max/avg Z-scores, dominant detector
- **Value Forecasting**: Overlay of actual sensor values with 7-day forecasts (dashed lines)
- **OMR Deep-Dive**: Reveals sensor-to-sensor prediction residuals (baseline consistency detector)
- **Smart Filtering**: Focuses on top contributing sensors for clarity

**Panel Details**:
1. **Sensor Contribution Timeline**: Stacked area showing top 5 sensors' contribution % over time
2. **Detector Signals Heatmap**: Pivot table with sensors as rows, detectors as columns, Z-scores as values
3. **Sensor Defect Statistics**: Full ranking with defect count, max/avg Z, dominant detector, last seen
4. **Sensor Values & Forecasts**: Line chart with actual (solid) and forecasted (dashed) sensor values
5. **OMR Contributions Table**: Sensor-to-sensor prediction residuals sorted by magnitude
6. **OMR Timeline**: Time series of residuals for top 5 sensors

**Variables**:
- `$datasource`: MSSQL data source selector
- `$equipment`: Equipment filter (EquipID from Equipment table)

**Default Time Range**: Last 7 days (`now-7d` to `now`) - focused on recent diagnostics

**Refresh**: 1 minute (more frequent for active troubleshooting)

**Navigation**: Links to all ACM dashboards via dropdown menu

---

### 3. ACM Asset Story (`acm_asset_story.json`)

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

### 4. ACM Operations Monitor (`acm_operations_monitor.json`)

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


# ACM Distilled - Analytics-Only Pipeline

## Overview

`acm_distilled.py` is a simplified, analytics-focused version of the full ACM pipeline. It focuses **ONLY** on answering the fundamental analytical questions about equipment health without the product engineering overhead.

## Purpose

This script is designed for:
- **Quick analytical investigations** - Run on-demand analysis for specific time periods
- **Manual inspection** - Investigate specific equipment behavior
- **Analytical insights** - Get comprehensive health reports without writing everything to SQL
- **Prototyping** - Test analytical approaches before integrating into the full pipeline

## What It Does

The script runs the complete ACM analytical pipeline:

1. **Data Loading** - Loads historian data from SQL for specified equipment and time range
2. **Feature Engineering** - Builds statistical features from raw sensor data
3. **Detector Fitting** - Fits 6 analytical detectors (AR1, PCA, IForest, GMM, OMR, drift)
4. **Detector Scoring** - Scores data with all detectors to produce anomaly Z-scores
5. **Regime Detection** - Identifies operating regimes using unsupervised clustering
6. **Score Calibration** - Calibrates raw detector scores to normalized Z-scores
7. **Fusion** - Combines detector scores with adaptive weights
8. **Episode Detection** - Identifies anomaly episodes using CUSUM change-point detection
9. **Drift Analysis** - Detects behavioral drift from baseline
10. **Health Scoring** - Computes equipment health index (0-100%)
11. **RUL Forecasting** - Predicts Remaining Useful Life with confidence intervals
12. **Culprit Attribution** - Identifies sensors contributing most to anomalies

## Core Questions Answered

| Question | ACM Answer |
|----------|------------|
| **1. What is wrong?** | Multi-detector anomaly scores (AR1, PCA, IForest, GMM, OMR) |
| **2. When did it start?** | Episode detection with precise timestamps |
| **3. Which sensors?** | Ranked list of contributing sensors (culprits) |
| **4. Which operating mode?** | Regime identification with confidence scores |
| **5. What will happen?** | RUL forecast with P10/P50/P90 confidence intervals |
| **6. How severe?** | Health index (0-100%) and trend analysis |

## Usage

### Basic Usage

```bash
python acm_distilled.py --equip FD_FAN \
    --start-time "2024-01-01T00:00:00" \
    --end-time "2024-01-31T23:59:59"
```

### Save Report to File

```bash
python acm_distilled.py --equip GAS_TURBINE \
    --start-time "2024-10-01T00:00:00" \
    --end-time "2024-10-31T23:59:59" \
    --output /tmp/gas_turbine_report.txt
```

### Command Line Options

- `--equip EQUIP` (required) - Equipment code from Equipment table (e.g., FD_FAN, GAS_TURBINE)
- `--start-time TIME` (required) - Analysis start time in ISO format (YYYY-MM-DDTHH:MM:SS)
- `--end-time TIME` (required) - Analysis end time in ISO format
- `--output FILE` (optional) - Output file path (default: print to console)

## Output Report Format

The script generates a comprehensive text report with 9 sections:

### 1. Data Summary
- Number of training/scoring rows
- Number of sensors analyzed
- Time periods covered

### 2. Detector Scores
- Statistics for each detector (AR1, PCA-SPE, PCA-T², IForest, GMM, OMR)
- Mean, Max, and P95 Z-scores

### 3. Anomaly Episodes
- Total number of episodes detected
- Top 5 episodes with:
  - Start time and duration
  - Maximum Z-score
  - Severity classification

### 4. Operating Regimes
- Number of regimes detected
- Current regime
- Quality score (silhouette)
- Time spent in each regime (%)

### 5. Equipment Health
- Current health index (0-100%)
- Average health over period
- Trend (IMPROVING, STABLE, DEGRADING)
- Status (GOOD, WARNING, CRITICAL)

### 6. Drift Analysis
- Drift status (STABLE, DRIFTING)
- Drift Z-score
- Multi-feature drift assessment

### 7. RUL Forecast
- P10, P50, P90 RUL estimates (hours)
- Confidence score
- Reliability status

### 8. Top Contributing Sensors
- Ranked list of sensors contributing to anomalies
- Contribution scores for each

### 9. Recommendations
- Actionable recommendations based on findings
- Prioritized by urgency

## Differences from Full ACM Pipeline

### What's INCLUDED:
- ✅ Complete analytical pipeline (detectors, fusion, episodes, forecasting)
- ✅ Operating regime detection
- ✅ Health scoring and RUL forecasting
- ✅ Comprehensive text report generation

### What's EXCLUDED:
- ❌ Extensive SQL writes (only minimal data loading, no result persistence)
- ❌ Model persistence and caching
- ❌ Continuous learning and model lifecycle management
- ❌ Grafana dashboard generation
- ❌ OpenTelemetry observability integration
- ❌ Batch processing and scheduling
- ❌ Configuration history tracking
- ❌ Adaptive threshold tuning history

## Requirements

### Python Version
- Python 3.11 or higher

### Dependencies
All dependencies from the main ACM project:
- numpy
- pandas
- scikit-learn
- matplotlib
- pyyaml
- joblib
- statsmodels
- scipy
- pyodbc

### Database
- Access to ACM SQL Server database
- Valid `configs/sql_connection.ini` file
- Equipment must exist in Equipment table

## Example Output

```
================================================================================
ACM ANALYTICS REPORT - FD_FAN
================================================================================
Analysis Period: 2024-01-01 00:00:00 to 2024-01-31 23:59:59

1. DATA SUMMARY
--------------------------------------------------------------------------------
  Train Rows: 1440
  Score Rows: 960
  Sensors: 12
  Train Period: 2024-01-01 00:00:00 to 2024-01-19 09:36:00
  Score Period: 2024-01-19 09:36:00 to 2024-01-31 23:59:59

2. DETECTOR SCORES (Z-Scores)
--------------------------------------------------------------------------------
  ar1_z          - Mean:   1.23 | Max:   8.45 | P95:   4.12
  pca_spe_z      - Mean:   0.98 | Max:   6.78 | P95:   3.45
  pca_t2_z       - Mean:   1.05 | Max:   7.23 | P95:   3.89
  iforest_z      - Mean:   0.87 | Max:   5.34 | P95:   2.98
  gmm_z          - Mean:   0.92 | Max:   6.12 | P95:   3.21
  omr_z          - Mean:   1.15 | Max:   7.89 | P95:   4.05

3. ANOMALY EPISODES
--------------------------------------------------------------------------------
  Total Episodes: 3
  Episode 1:
    Start: 2024-01-15 14:30:00
    Duration: 2.5 hours
    Max Z-Score: 8.45
    Severity: CRITICAL

...
```

## Best Practices

1. **Analysis Window** - Use 3-7 day windows for meaningful patterns (too short = noisy, too long = slow)
2. **Equipment Selection** - Ensure equipment has sufficient historian data in the time period
3. **Interpretation** - Focus on trends and patterns, not single anomalies
4. **Validation** - Cross-check findings with operational logs and maintenance records

## Troubleshooting

### "Equipment not found"
- Verify equipment code exists in Equipment table: `SELECT EquipCode FROM Equipment`

### "Insufficient data"
- Ensure historian data exists for time period: `SELECT COUNT(*) FROM vw_HistorianData WHERE EquipID = ? AND EntryDateTime BETWEEN ? AND ?`

### "RUL forecast failed"
- RUL requires health history - ensure analysis window is long enough (> 3 days recommended)

## See Also

- `core/acm_main.py` - Full ACM pipeline with product engineering
- `scripts/sql_batch_runner.py` - Automated batch processing
- `docs/ACM_SYSTEM_OVERVIEW.md` - Complete system documentation

# Critical Issues Found in ACM Dashboard and Analytics

## Date: November 19, 2025
## Equipment: FD_FAN (EquipID=1)

---

## SUMMARY OF PROBLEMS

### 1. **PCA Explained Variance Chart - EMPTY TABLE**
- **Issue**: `ACM_PCA_Metrics` table has 0 rows
- **Root Cause**: Method signature mismatch in line 3349 of `acm_main.py`
  - Calling `write_pca_metrics(df_metrics, run_id)` with DataFrame
  - But method expects `write_pca_metrics(pca_detector, tables_dir, enable_sql)`
- **Impact**: PCA Explained Variance dashboard panel shows no data

### 2. **Sensor Anomaly Rate - NOT WORKING**
- **Issue**: No sensor-level anomaly rate calculations
- **Related Tables**: `ACM_SensorAnomalyByPeriod`, `ACM_SensorHotspots`
- **Root Cause**: Tables exist but may have wrong column names or no data
- **Impact**: Cannot see which sensors are most anomalous

### 3. **Health Forecast - DATA MISMATCH**
- **Issue**: Forecasts extend 7 days beyond actual health data
  - Health data ends: **Nov 10, 2024 13:30**
  - Forecasts go to: **Nov 17, 2024 13:30**
- **Root Cause**: Forecasting is extrapolating into the future without new base data
- **Impact**: Misleading forecast charts showing predictions without recent health updates

### 4. **Failure Probability - LIMITED DATA**
- **Issue**: Only 75 rows in `ACM_EnhancedFailureProbability_TS` vs 336 in basic forecasts
- **Date Range**: Oct 20, 2023 to Nov 10, 2024
- **Impact**: Sparse failure probability predictions

### 5. **Drift Detection - WORKING CORRECTLY (User Misunderstanding)**
- **Issue**: User believes system is "clearly drifting" but metrics show otherwise
- **ACTUAL Evidence**:
  - CUSUM_Z P95=1.051 (threshold=3.0) - BELOW threshold ✓
  - DRIFT_Z column: **EMPTY** (not being written)
  - FUSED P95=0.490 - very low, indicates healthy system ✓
  - Health Index: 99-100% - excellent condition ✓
- **Root Cause**: 
  1. User confusing "DriftValue" column (0.99) with drift z-score
  2. DRIFT_Z column not being populated (but CUSUM_Z is working)
- **Reality**: System is CORRECTLY staying in WATCH phase - equipment is healthy!
- **Impact**: No issue - system working as designed. Need to explain metrics to user.

### 6. **Regime Identification Beyond Health Data**
- **Issue**: Regime timeline matches health timeline (both end Nov 10)
- **Problem**: Dashboard query may be requesting regime data for dates with no health
- **Impact**: Regime occupancy charts may show empty periods

### 7. **Detector Correlation Matrix - UNKNOWN STATUS**
- **Issue**: Need to verify if `ACM_DetectorCorrelation` table has data
- **Impact**: Cannot see which detectors agree/disagree

---

## DATA RANGE ANALYSIS

| Table | Rows | Start Date | End Date |
|-------|------|------------|----------|
| ACM_HealthTimeline | 11,999 | 2023-10-15 00:00 | 2024-11-10 13:30 |
| ACM_RegimeTimeline | 11,999 | 2023-10-15 00:00 | 2024-11-10 13:30 |
| ACM_DriftSeries | 11,999 | 2023-10-15 00:00 | 2024-11-10 13:30 |
| ACM_HealthForecast_TS | 336 | 2024-10-27 14:00 | 2024-11-17 13:30 |
| ACM_FailureForecast_TS | 336 | 2024-10-27 14:00 | 2024-11-17 13:30 |
| ACM_EnhancedFailureProbability | 75 | 2023-10-20 00:00 | 2024-11-10 13:30 |
| ACM_PCA_Metrics | 0 | N/A | N/A |

**Critical Gap**: Health data stopped being updated after Nov 10, 2024 (9 days ago)

---

## DRIFT STATISTICS

```
Recent 100 drift values:
  Min:  0.6266
  Max:  0.9943
  Mean: 0.8201
  P95:  0.9908
  P99:  0.9943
```

**Analysis**: 
- Values near 1.0 indicate maximum drift
- P95 of 0.9908 is extremely high
- System should have transitioned to DRIFT/ALERT mode
- Likely stuck in WATCH phase due to threshold misconfiguration

---

## REQUIRED FIXES

### Priority 1: Fix PCA Metrics Writing

**File**: `core/acm_main.py` line 3349

**Current (BROKEN)**:
```python
rows_pca_metrics = output_manager.write_pca_metrics(df_metrics, run_id or "")
```

**Should Be**:
```python
rows_pca_metrics = output_manager.write_pca_metrics(
    pca_detector=pca_detector,
    tables_dir=tables_dir,
    enable_sql=(sql_client is not None)
)
```

### Priority 2: Fix Drift Threshold Logic

**File**: `core/acm_main.py` lines 2457-2539

**Issue**: Multi-feature drift detection not triggering properly

**Check**:
1. Verify drift threshold in `configs/config_table.csv`
2. Check if `multi_feature.enabled` is True
3. Verify hysteresis thresholds are appropriate:
   - `hysteresis_on: 3.0` (turn ON drift alert)
   - `hysteresis_off: 1.5` (turn OFF drift alert)
4. Current P95=0.9908 should exceed hysteresis_on threshold

### Priority 3: Fix Stage Determination

**File**: `core/acm_main.py` line 515

**Current**: Stage hardcoded to "score"
```python
stage = cfg.get("runtime", {}).get("stage", "score")
```

**Need**: Dynamic stage based on drift/health
- WATCH: Drift detected but not critical
- ALERT: High drift or health degradation
- NORMAL: No issues

### Priority 4: Update Batch Runner to Continue Processing

**Issue**: Batch runner likely stopped at Nov 10 due to data gap (Oct 23-24)

**Solution**: 
- Run batch processor with `--start-from-beginning` to fill all gaps
- Or adjust batch runner to skip empty windows gracefully

### Priority 5: Verify Chart Table Structures

**Tables to Check**:
- `ACM_SensorAnomalyByPeriod` - verify column names
- `ACM_RegimeOccupancy` - verify has Timestamp column
- `ACM_DetectorCorrelation` - verify has data

---

## NEXT STEPS

1. **Immediate**: Fix PCA metrics writing bug (1-line change)
2. **Critical**: Investigate why drift detection not transitioning to ALERT
3. **Important**: Run full batch processing to update all data through current date
4. **Validation**: Verify all dashboard panels load after fixes

---

## NOTES

- Latest batch run: RunID `95E49F31-AC7B-4017-A6A9-C9A76945DC70`
- All data ends at Nov 10, 2024 13:30
- System has been in WATCH phase too long given drift levels
- Forecasts are extrapolating without fresh input data

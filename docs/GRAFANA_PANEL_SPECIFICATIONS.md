# Grafana Dashboard Panel Specifications
**For Integration:** OMR Diagnostics, Forecast Quality Metrics, RUL Multipath  
**Target Dashboard:** `operator_dashboard.json`  
**Date:** 2025-11-30

---

## Panel Addition Summary

Add **8 new panels** to visualize the newly integrated SQL tables:
- **3 panels** for OMR Diagnostics (`ACM_OMR_Diagnostics`)
- **3 panels** for Forecast Quality (`ACM_Forecast_QualityMetrics`)
- **2 panels** for enhanced RUL Multipath visualization

---

## 1. OMR Diagnostics Section

### Panel 1.1: OMR Model Configuration
**Type:** Stat Panel (Grid)  
**Position:** Suggest placing near existing OMR panels (around line 1018)  
**Datasource:** MSSQL  

**SQL Query:**
```sql
SELECT TOP 1
  ModelType AS [Model Type],
  NComponents AS [Components],
  TrainSamples AS [Training Samples],
  TrainFeatures AS [Features Used]
FROM ACM_OMR_Diagnostics
WHERE EquipID = $equipment
ORDER BY FitTimestamp DESC
```

**Configuration:**
- **Title:** "OMR Model Configuration"
- **Description:** "Current OMR detector model settings from latest training run"
- **Field Config:**
  - No thresholds
  - Text mode: value_and_name
  - Orientation: Horizontal
  - Color mode: None

**Value Ranges:**
- ModelType: "pls", "linear", or "pca"
- Components: 1-20 (typical)
- Training Samples: 100-10000+
- Features Used: 5-100+

---

### Panel 1.2: OMR Training Quality
**Type:** Stat Panel  
**Position:** Next to Panel 1.1  

**SQL Query:**
```sql
SELECT TOP 1
  TrainResidualStd AS value,
  'Training Quality' AS metric
FROM ACM_OMR_Diagnostics
WHERE EquipID = $equipment
ORDER BY FitTimestamp DESC
```

**Configuration:**
- **Title:** "OMR Training Quality (Residual Std)"
- **Description:** "Lower residual std indicates better model fit. Values >3.0 may indicate poor calibration."
- **Field Config:**
  - Unit: None (dimensionless)
  - Decimals: 3
  - Thresholds:
    - Green: 0.0 - 1.5 (excellent fit)
    - Yellow: 1.5 - 3.0 (acceptable)
    - Red: >3.0 (poor fit)
  - Color mode: Background
  - Graph mode: Area

**Interpretation:**
- <1.5: Excellent model calibration
- 1.5-3.0: Acceptable, monitor for drift
- >3.0: Poor fit, consider recalibration

---

### Panel 1.3: OMR Calibration Status
**Type:** Stat Panel  
**Position:** Next to Panel 1.2  

**SQL Query:**
```sql
SELECT TOP 1
  CalibrationStatus AS [Status],
  DATEDIFF(HOUR, FitTimestamp, GETDATE()) AS [Hours Since Calibration]
FROM ACM_OMR_Diagnostics
WHERE EquipID = $equipment
ORDER BY FitTimestamp DESC
```

**Configuration:**
- **Title:** "OMR Calibration Status"
- **Description:** "Model calibration health. SATURATED indicates detector is hitting limits and may need retraining."
- **Field Config:**
  - Mappings:
    - "VALID" → "✓ VALID" (Green)
    - "SATURATED" → "⚠ SATURATED" (Red)
    - "DISABLED" → "○ DISABLED" (Gray)
  - Color mode: Background
  - Text mode: Value only

**Alert Thresholds:**
- Status = "SATURATED" → Critical alert (model hitting limits)
- Hours Since Calibration >168 (7 days) → Warning (consider recalibration)

---

## 2. Forecast Quality Section

### Panel 2.1: Current Forecast Accuracy
**Type:** Stat Panel (Grid)  
**Position:** Near existing forecast/RUL panels  

**SQL Query:**
```sql
SELECT TOP 1
  RMSE AS [RMSE],
  MAE AS [MAE],
  MAPE AS [MAPE %]
FROM ACM_Forecast_QualityMetrics
WHERE EquipID = $equipment
ORDER BY ComputeTimestamp DESC
```

**Configuration:**
- **Title:** "Current Forecast Accuracy"
- **Description:** "Latest forecast quality metrics. Lower values = better predictions."
- **Field Config:**
  - RMSE thresholds:
    - Green: 0-5
    - Yellow: 5-10
    - Red: >10
  - MAE thresholds: (same as RMSE)
  - MAPE thresholds:
    - Green: 0-10%
    - Yellow: 10-20%
    - Red: >20%
  - Decimals: 2
  - Orientation: Horizontal

**Interpretation:**
- RMSE/MAE: Absolute prediction error (health units)
- MAPE: Percentage error (easier to interpret)
- Target: MAPE <10%, RMSE <5

---

### Panel 2.2: Forecast Quality Trend
**Type:** Time Series  
**Position:** Below Panel 2.1  

**SQL Query:**
```sql
SELECT
  ComputeTimestamp AS time,
  RMSE,
  MAE,
  MAPE AS [MAPE (%)]
FROM ACM_Forecast_QualityMetrics
WHERE EquipID = $equipment
  AND $__timeFilter(ComputeTimestamp)
ORDER BY time
```

**Configuration:**
- **Title:** "Forecast Quality Over Time"
- **Description:** "Track forecast accuracy trends. Rising RMSE/MAE indicates model drift—retrain recommended."
- **Field Config:**
  - Legend: Show (bottom)
  - Tooltip: All series
  - Y-axis: Dual axis
    - Left: RMSE, MAE (0-20 scale)
    - Right: MAPE (0-50% scale)
  - Line width: 2
  - Point size: 5

**Alert Rules:**
- RMSE increasing >50% over 24 hours → Warning
- MAPE >20% for 3+ consecutive runs → Critical

---

### Panel 2.3: Model Retrain Events
**Type:** State Timeline  
**Position:** Below Panel 2.2  

**SQL Query:**
```sql
SELECT
  ComputeTimestamp AS time,
  CASE 
    WHEN RetrainTriggered = 1 THEN 'RETRAIN'
    ELSE 'OK'
  END AS [Status],
  COALESCE(RetrainReason, 'No retrain needed') AS [Reason]
FROM ACM_Forecast_QualityMetrics
WHERE EquipID = $equipment
  AND $__timeFilter(ComputeTimestamp)
ORDER BY time
```

**Configuration:**
- **Title:** "Forecast Model Retrain Events"
- **Description:** "Timeline showing when model retraining was triggered and why"
- **Field Config:**
  - Mappings:
    - "OK" → Green
    - "RETRAIN" → Yellow
  - Show legend: Yes
  - Show values: On hover

**Value Mappings:**
- RetrainReason examples: "Quality degradation", "Concept drift", "Scheduled retrain"

---

## 3. Enhanced RUL Multipath Section

### Panel 3.1: RUL Multipath Comparison (Enhanced)
**Type:** Bar Gauge (Horizontal)  
**Position:** Replace or augment existing RUL table panel  

**SQL Query:**
```sql
SELECT TOP 1
  RUL_Trajectory_Hours AS [Trajectory Path],
  RUL_Hazard_Hours AS [Hazard Path],
  RUL_Energy_Hours AS [Energy Path],
  RUL_Final_Hours AS [Consensus (Final)]
FROM ACM_RUL_Summary
WHERE EquipID = $equipment
  AND $__timeFilter(LastUpdate)
ORDER BY LastUpdate DESC
```

**Configuration:**
- **Title:** "RUL Multi-Path Comparison"
- **Description:** "Compare RUL estimates from different failure models. Consensus = weighted average. Large divergence indicates uncertainty."
- **Field Config:**
  - Orientation: Horizontal
  - Display mode: Gradient
  - Show unfilled: Yes
  - Max value: Auto
  - Thresholds:
    - Red: 0-48 (critical, <2 days)
    - Yellow: 48-168 (warning, 2-7 days)
    - Green: >168 (healthy, >7 days)
  - Unit: hours

**Interpretation:**
- Paths agree → High confidence
- Trajectory < others → Rapid degradation trend
- Hazard < others → High failure probability
- Energy < others → Accelerating damage accumulation

---

### Panel 3.2: RUL Multipath Timeline
**Type:** Time Series (Multi-line)  
**Position:** Below Panel 3.1  

**SQL Query:**
```sql
SELECT
  LastUpdate AS time,
  RUL_Trajectory_Hours AS [Trajectory-based RUL],
  RUL_Hazard_Hours AS [Hazard-based RUL],
  RUL_Energy_Hours AS [Energy-based RUL],
  RUL_Final_Hours AS [Consensus RUL]
FROM ACM_RUL_Summary
WHERE EquipID = $equipment
  AND $__timeFilter(LastUpdate)
ORDER BY time
```

**Configuration:**
- **Title:** "RUL Multi-Path Timeline"
- **Description:** "Historical RUL estimates from all failure models. Diverging paths indicate model disagreement—check which sensors are driving each estimate."
- **Field Config:**
  - Legend: Show (right side)
  - Tooltip: All series
  - Line styles:
    - Trajectory: Solid blue
    - Hazard: Dashed orange
    - Energy: Dotted red
    - Consensus: Bold green (width 3)
  - Fill opacity: 0 (transparent)
  - Point size: Auto
  - Y-axis: Hours (0-auto)

**Alert Rules:**
- Any path drops below 48 hours → Critical
- Path divergence >100 hours → Warning (high uncertainty)

---

## 4. Implementation Guide

### Step 1: Backup Current Dashboard
```powershell
Copy-Item "grafana_dashboards/operator_dashboard.json" `
          "grafana_dashboards/operator_dashboard.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
```

### Step 2: Add Panels via Grafana UI

**Recommended Approach (Manual via UI):**
1. Open Grafana → Dashboards → "ACM Operator Dashboard"
2. Click "Add" → "Visualization"
3. Copy SQL query from specs above
4. Configure panel settings as specified
5. Position panels in logical groups:
   - **Row 1 (OMR Diagnostics):** Panels 1.1, 1.2, 1.3
   - **Row 2 (Forecast Quality):** Panels 2.1, 2.2, 2.3
   - **Row 3 (RUL Multipath):** Panels 3.1, 3.2
6. Save dashboard

**Alternative (JSON Edit):**
- See section "JSON Structure Reference" below for panel object template
- Insert new panel objects into `panels` array
- Adjust `gridPos` coordinates to avoid overlaps

### Step 3: Test Panels

**Verification Checklist:**
- [ ] All panels load without errors
- [ ] Queries return data (run ACM pipeline first if empty)
- [ ] Thresholds display correct colors
- [ ] Time range filter works ($__timeFilter)
- [ ] Equipment variable filter works ($equipment)
- [ ] Legends are readable
- [ ] Tooltips show correct values

**Test Queries Directly:**
```sql
-- Should return 1 row with OMR diagnostics
SELECT TOP 1 * FROM ACM_OMR_Diagnostics ORDER BY FitTimestamp DESC;

-- Should return forecast quality metrics
SELECT TOP 1 * FROM ACM_Forecast_QualityMetrics ORDER BY ComputeTimestamp DESC;

-- Should show multipath columns populated
SELECT TOP 1 
  RUL_Final_Hours, RUL_Trajectory_Hours, RUL_Hazard_Hours, RUL_Energy_Hours, DominantPath 
FROM ACM_RUL_Summary 
ORDER BY LastUpdate DESC;
```

### Step 4: Export Updated Dashboard

After adding panels via UI:
```bash
# In Grafana UI:
# 1. Dashboard Settings → JSON Model → Copy to clipboard
# 2. Save to file:
```

```powershell
# Paste JSON into file
Set-Content -Path "grafana_dashboards/operator_dashboard.json" -Value $clipboardJson
```

---

## 5. Panel Layout Recommendations

```
┌─────────────────────────────────────────────────────────────────┐
│  Row: Current Health (Existing panels remain unchanged)        │
├─────────────────────────────────────────────────────────────────┤
│  Row: OMR Diagnostics (NEW)                                    │
│  ┌──────────────┬──────────────┬──────────────┐               │
│  │  Model       │  Training    │  Calibration │               │
│  │  Config      │  Quality     │  Status      │               │
│  └──────────────┴──────────────┴──────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│  Row: Forecast Quality (NEW)                                   │
│  ┌──────────────────────────────────────────┐                 │
│  │  Current Forecast Accuracy (RMSE/MAE)    │                 │
│  ├──────────────────────────────────────────┤                 │
│  │  Forecast Quality Trend (Time Series)    │                 │
│  ├──────────────────────────────────────────┤                 │
│  │  Model Retrain Events (Timeline)         │                 │
│  └──────────────────────────────────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│  Row: RUL Multi-Path Analysis (ENHANCED)                       │
│  ┌──────────────────────────────────────────┐                 │
│  │  RUL Multipath Comparison (Bar Gauge)    │                 │
│  ├──────────────────────────────────────────┤                 │
│  │  RUL Multipath Timeline (4-line chart)   │                 │
│  └──────────────────────────────────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│  Existing panels: Sensor trends, anomalies, etc. (unchanged)   │
└─────────────────────────────────────────────────────────────────┘
```

**Grid Positioning (if editing JSON):**
- Use 24-column grid
- Standard panel heights: 4-8 units
- Row height: ~8 units per row
- Leave 1-unit gaps between panels

---

## 6. Alert Configuration (Optional)

### OMR Diagnostics Alerts

**Alert: OMR Model Saturated**
```sql
-- Condition: CalibrationStatus = 'SATURATED'
SELECT COUNT(*) AS saturated_count
FROM ACM_OMR_Diagnostics
WHERE EquipID = $equipment
  AND CalibrationStatus = 'SATURATED'
  AND FitTimestamp > DATEADD(hour, -1, GETDATE())
HAVING COUNT(*) > 0
```
- Severity: Critical
- Action: Email alert + dashboard annotation
- Message: "OMR detector saturated—model hitting detection limits. Recalibration recommended."

### Forecast Quality Alerts

**Alert: Forecast Quality Degraded**
```sql
-- Condition: MAPE > 20% for 3+ consecutive runs
SELECT COUNT(*) AS poor_quality_count
FROM (
  SELECT TOP 3 MAPE
  FROM ACM_Forecast_QualityMetrics
  WHERE EquipID = $equipment
  ORDER BY ComputeTimestamp DESC
) AS recent
WHERE MAPE > 20
HAVING COUNT(*) >= 3
```
- Severity: Warning
- Action: Notification
- Message: "Forecast accuracy degraded (MAPE >20%). Model retrain may be needed."

---

## 7. Data Validation Queries

Before finalizing dashboards, verify data exists:

```sql
-- Check OMR diagnostics coverage
SELECT 
  EquipID,
  COUNT(*) AS diagnostic_records,
  MIN(FitTimestamp) AS first_diagnostic,
  MAX(FitTimestamp) AS latest_diagnostic,
  AVG(TrainResidualStd) AS avg_residual_std
FROM ACM_OMR_Diagnostics
GROUP BY EquipID;

-- Check forecast quality coverage
SELECT 
  EquipID,
  COUNT(*) AS quality_records,
  AVG(RMSE) AS avg_rmse,
  AVG(MAPE) AS avg_mape,
  SUM(CASE WHEN RetrainTriggered=1 THEN 1 ELSE 0 END) AS retrain_count
FROM ACM_Forecast_QualityMetrics
GROUP BY EquipID;

-- Verify multipath columns populated
SELECT 
  EquipID,
  COUNT(*) AS rul_records,
  COUNT(RUL_Trajectory_Hours) AS trajectory_count,
  COUNT(RUL_Hazard_Hours) AS hazard_count,
  COUNT(RUL_Energy_Hours) AS energy_count,
  COUNT(DominantPath) AS dominant_path_count
FROM ACM_RUL_Summary
GROUP BY EquipID;
```

---

## 8. Known Limitations & Future Enhancements

### Current Limitations
1. **R2Score not computed** - Currently NULL in `ACM_Forecast_QualityMetrics`
2. **Saturation rate not dynamic** - CalibrationStatus hardcoded to "VALID" at fit time
3. **No time-to-retrain indicator** - Could add predictive "days until retrain needed"

### Future Panel Ideas
1. **OMR Saturation Heatmap** - Show which sensors are saturating over time
2. **Forecast Horizon Accuracy** - Break down RMSE by forecast horizon (1hr, 6hr, 24hr)
3. **RUL Path Confidence** - Show uncertainty bands around each RUL path
4. **Model Version Timeline** - Track model version changes and their impact on quality

---

## 9. Documentation References

- **OMR Detector:** `docs/OMR_DETECTOR.md`
- **Forecasting:** `docs/Analytics Backbone.md` (Section: Health Forecasting)
- **RUL Engine:** `core/rul_engine.py` docstrings
- **SQL Schema:** `scripts/sql/patches/2025-11-30_omr_diagnostics_integration.sql`
- **Integration Status:** `docs/OMR_UPGRADE_INTEGRATION_STATUS.md`

---

## 10. Quick Reference: SQL Table Summary

| Table | Key Columns | Update Frequency | Panel Usage |
|-------|-------------|------------------|-------------|
| `ACM_OMR_Diagnostics` | ModelType, NComponents, TrainResidualStd, CalibrationStatus | Per model fit (~1x/day) | Panels 1.1, 1.2, 1.3 |
| `ACM_Forecast_QualityMetrics` | RMSE, MAE, MAPE, RetrainTriggered | Per forecast run (~hourly) | Panels 2.1, 2.2, 2.3 |
| `ACM_RUL_Summary` | RUL_Trajectory_Hours, RUL_Hazard_Hours, RUL_Energy_Hours, DominantPath, RUL_Final_Hours | Per RUL computation (~hourly) | Panels 3.1, 3.2 |

---

**End of Specifications**

**Next Actions:**
1. ✅ Review this specification document
2. ⏳ Add panels to Grafana via UI (manual, ~30-60 min)
3. ⏳ Test with live data from pipeline run
4. ⏳ Export updated dashboard JSON
5. ⏳ Commit dashboard changes to Git

**Estimated Time:** 1-2 hours for full dashboard integration and testing.

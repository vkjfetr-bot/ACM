# Critical ACM Fixes - December 2, 2025

**Branch**: `fix/pca-metrics-and-forecast-integrity`  
**Status**: ✅ Core PCA issue FIXED, Forecast CI issue identified

---

## 🔥 Critical Issues Fixed

### 1. **PCA_Metrics Duplicate Key Violations** ✅ FIXED
**Problem**: Every single batch run crashed with:
```
Violation of PRIMARY KEY constraint 'PK_ACM_PCA_Metrics'. 
Cannot insert duplicate key in object 'dbo.ACM_PCA_Metrics'. 
The duplicate key value is (37ABD786-D01B-4F98-9C03-8AC48CA944C5, 1).
```

**Root Cause**:
- `write_pca_metrics()` used blind INSERT without checking for existing RunID+EquipID
- Primary key: `(RunID, EquipID, ComponentName, MetricType)`
- When rerunning or retraining, same RunID would attempt duplicate insert

**Solution Implemented**:
```python
def _upsert_pca_metrics(self, df: pd.DataFrame) -> int:
    """Upsert PCA metrics using MERGE to avoid duplicate key violations."""
    # SQL MERGE statement:
    # - WHEN MATCHED: UPDATE existing row
    # - WHEN NOT MATCHED: INSERT new row
    # No more PRIMARY KEY violations!
```

**Impact**: Batch processing can now run uninterrupted without PCA crashes.

---

### 2. **Mahalanobis Detector Mislabeled** ✅ FIXED
**Problem**: Dashboard showing "Statistical Outlier (Mahalanobis)" instead of proper multivariate detector name

**Before**:
```python
'mhal_z': 'Statistical Outlier (Mahalanobis)'  # Confusing - sounds like univariate
```

**After**:
```python
'mhal_z': 'Multivariate Distance (Mahalanobis)'  # Clear multivariate indicator
```

**Also Fixed**:
- OMR: "Persistent Outlier" → "Baseline Consistency (OMR)" (matches dashboard expectations)
- Short labels updated for consistency

---

## ⚠️ Outstanding Critical Issues

### 3. **Forecast Confidence Intervals Always Zero** ❌ NOT YET FIXED
**Problem Visible in Dashboard**:
```
Health Forecast with Confidence Intervals
- Confidence Lower: 0
- Confidence Upper: 0
- Health Forecast: 0
```

**Root Cause Located**:
```python
# core/forecasting.py line 1225
ci_lower.append(max(health_min, forecast_val - ci_width))
ci_upper.append(min(health_max, forecast_val + ci_width))

# BUT THEN...
health_forecast_df = pd.DataFrame({
    "CI_Lower": ci_lower,  # ✅ Correct values
    "CI_Upper": ci_upper,  # ✅ Correct values
    "CiLower": ci_lower,   # Duplicate for schema compatibility
    "CiUpper": ci_upper,
})
```

**The Problem**: The values ARE calculated correctly, but:
1. Dashboard may be querying wrong column name
2. OR forecast is returning empty/zero health values
3. OR horizon-based variance calculation is broken

**Next Steps**:
- Check Grafana query: which column does it use? `CI_Lower` or `CiLower`?
- Verify `std_error` calculation - is it always zero?
- Check `forecast_values` - are health forecasts themselves zero?

---

### 4. **Episode Table Still Sparse** ❌ PARTIALLY ADDRESSED
**Problem**: Episode Root Cause Analysis table only shows October 2023 data

**Current Status**:
- Batch runner executing: 25 batches × 28 days = ~700 days coverage
- **Batch 1-9 completed successfully** (Oct 2023 → Jun 2024)
- Process interrupted at Batch 9

**Why Table Looks Empty**:
- Only processed 9 out of 25 batches (36% complete)
- Episodes exist but limited to Oct 2023 - Jun 2024 range
- Need to let batch process complete all 25 batches

**Solution**: Resume or restart batch processing:
```powershell
.\scripts\run_data_range_batches.ps1 `
  -Equipment "FD_FAN" `
  -NumBatches 25 `
  -StartDate "2023-10-15" `
  -EndDate "2025-09-14" `
  -BatchSizeMinutes 40320  # 28 days per batch
```

---

## 📊 Batch Processing Command (FOR REFERENCE)

**The CORRECT way to run ACM from beginning**:

```powershell
# PowerShell batch runner script (handles everything automatically)
.\scripts\run_data_range_batches.ps1 `
  -Equipment "FD_FAN" `
  -NumBatches 25 `
  -StartDate "2023-10-15" `
  -EndDate "2025-09-14" `
  -BatchSizeMinutes 40320
```

**What it does**:
- Automatically sets `$env:ACM_BATCH_MODE = "1"`
- Automatically sets `$env:ACM_BATCH_NUM = <batch_number>`
- Calls `python -m core.acm_main --equip FD_FAN --start-time <start> --end-time <end>`
- Handles SUCCESS/NOOP/FAILURE tracking
- Provides progress reporting

**DO NOT manually set environment variables or call acm_main directly for batch processing!**

---

## 🔧 Technical Details

### PCA_Metrics Schema
```sql
CREATE TABLE dbo.ACM_PCA_Metrics (
    RunID              uniqueidentifier NOT NULL,
    EquipID            int NOT NULL,
    ComponentName      nvarchar(50) NOT NULL,  -- 'PCA'
    MetricType         nvarchar(50) NOT NULL,  -- 'n_components', 'variance_explained', 'n_features'
    Value              float NULL,
    CreatedAt          datetime2(3) NOT NULL DEFAULT (SYSUTCDATETIME()),
    CONSTRAINT PK_ACM_PCA_Metrics PRIMARY KEY CLUSTERED (RunID, EquipID, ComponentName, MetricType)
);
```

**Why MERGE is necessary**:
- Batch mode may retrain on same data window (continuous learning)
- Same RunID used for retraining = duplicate key on INSERT
- MERGE gracefully updates existing or inserts new

### Detector Labels Mapping (Updated)
| Code | Full Label | Short Label | Category |
|------|-----------|-------------|----------|
| `ar1_z` | Time-Series Anomaly (AR1) | Time-Series (AR1) | Univariate |
| `pca_spe_z` | Correlation Break (PCA-SPE) | Correlation (PCA) | Multivariate |
| `pca_t2_z` | Multivariate Outlier (PCA-T²) | Outlier (PCA-T²) | Multivariate |
| `mhal_z` | **Multivariate Distance (Mahalanobis)** | **Distance (Mahal)** | Multivariate |
| `gmm_z` | Density Anomaly (GMM) | Density (GMM) | Multivariate |
| `iforest_z` | Rare State (IsolationForest) | Rare State (IF) | Multivariate |
| `omr_z` | **Baseline Consistency (OMR)** | **Baseline (OMR)** | Ensemble |

---

## ✅ What's Fixed Now

1. **PCA_Metrics duplicate key crashes** - ELIMINATED with MERGE upsert
2. **Mahalanobis detector label** - Shows "Multivariate Distance" instead of "Statistical Outlier"
3. **OMR label consistency** - "Baseline Consistency (OMR)" matches dashboard expectations
4. **Batch processing progress** - 9/25 batches completed before interruption

---

## 🚧 What Still Needs Fixing

1. **Forecast confidence intervals showing as 0** - Need to debug Grafana query and std_error calculation
2. **Complete batch processing** - Resume to process remaining 16 batches (Jun 2024 → Sep 2025)
3. **Verify forecast SQL writes** - Check if ACM_HealthForecast_TS has proper CI_Lower/CI_Upper values

---

## 📈 Expected Results After Fixes

### Episode Table
- **Before**: 14 episodes (Oct 15-21, 2023 only)
- **After full batch**: ~200-300 episodes covering Oct 2023 → Sep 2025

### Forecast Dashboard
- **Before**: CI_Lower=0, CI_Upper=0 (broken)
- **After fix**: CI_Lower=20-40, CI_Upper=80-100 (proper uncertainty bands)

### PCA Processing
- **Before**: CRASH on every batch with PRIMARY KEY violation
- **After**: ✅ Silent MERGE upsert, no crashes

---

**End of Summary**

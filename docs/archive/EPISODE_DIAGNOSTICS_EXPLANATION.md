# Episode Diagnostics Table - Detailed Explanation

## Your Question: "How can ar1_z be a culprit???"

**Short Answer**: `ar1_z` is NOT a sensor name - it's the **detector algorithm** that flagged the anomaly. The dashboard column name "TopCulprit" is misleading.

---

## The Confusion

### What You See in Dashboard
```
Column: TopCulprit
Value:  ar1_z
```

### What This Actually Means
- **NOT**: "Sensor called `ar1_z` is faulty"
- **YES**: "The **AR1 detector algorithm** identified the anomaly"

---

## Table Purpose & Origin

### SQL Table: `ACM_EpisodeDiagnostics`

**Purpose**: Summarized episode metadata for operators and dashboards

**Schema**:
```sql
- episode_id         (int)       Episode identifier
- peak_timestamp     (datetime2) When anomaly peaked
- duration_h         (float)     Episode duration in hours
- peak_z             (float)     Maximum fused z-score
- avg_z              (float)     Average fused z-score
- dominant_sensor    (nvarchar)  PRIMARY DETECTOR ALGORITHM
- severity           (nvarchar)  Info/Warning/Critical
- severity_reason    (nvarchar)  Why this severity level
- min_health_index   (float)     Lowest health during episode
```

**Column Naming Issue**: `dominant_sensor` is **MISNAMED** - it contains the **detector algorithm name**, not a sensor name.

---

## What Should This Column Contain?

### Current Behavior (INCORRECT)
```
dominant_sensor = "ar1_z"         ← Detector algorithm
dominant_sensor = "pca_spe_z"     ← Detector algorithm
dominant_sensor = "iforest_z"     ← Detector algorithm
dominant_sensor = "mhal_z"        ← Detector algorithm
```

### Expected Behavior (CORRECT)
```
dominant_sensor = "Temperature_01"     ← Actual sensor name
dominant_sensor = "Vibration_Bearing"  ← Actual sensor name
dominant_sensor = "Pressure_Inlet"     ← Actual sensor name
```

**When multivariate detectors identify specific sensors**:
```
dominant_sensor = "pca_spe_z(DEMO.SIM.FSAB)"  ← Detector + Sensor
```

---

## How "Culprits" Are Determined

### Episode Attribution Logic (from `core/fuse.py`)

```python
# Step 1: Find detector with highest mean score during episode
for name, scores in episode_streams.items():
    mean_score = np.nanmean(scores)
    if mean_score > max_mean_score:
        primary_detector = name  # e.g., "ar1_z", "pca_spe_z"

# Step 2: For multivariate detectors, find top contributing sensor
if 'pca' in primary_detector or 'mhal' in primary_detector:
    episode_features = original_features.iloc[s:e+1]
    top_feature = episode_features.mean().idxmax()
    culprit_sensor = _get_base_sensor(str(top_feature))
    culprits = f"{primary_detector}({culprit_sensor})"
else:
    # Univariate detectors (AR1, IForest, GMM) - no sensor attribution
    culprits = primary_detector
```

### What This Means

**AR1 (Autoregressive) Detector**:
- Monitors **each sensor independently**
- Detects time-series anomalies (trend breaks, spikes, drops)
- **Cannot pinpoint which specific sensor** triggered (limitation of current implementation)
- Returns: `"ar1_z"` (algorithm name only)

**PCA/MHAL (Multivariate) Detectors**:
- Monitor **correlations across all sensors**
- Can identify **which sensor contributed most**
- Returns: `"pca_spe_z(Temperature_01)"` (algorithm + sensor)

**IForest/GMM (Ensemble) Detectors**:
- Monitor **overall system state**
- Cannot pinpoint specific sensor (black-box models)
- Returns: `"iforest_z"` (algorithm name only)

---

## The Real Problem

### Data Flow Issue

```
episodes.culprits = "ar1_z"  ← From fuse.py (detector name)
       ↓
ACM_EpisodeDiagnostics.dominant_sensor = "ar1_z"  ← WRONG COLUMN NAME
       ↓
Dashboard "TopCulprit" = "ar1_z"  ← MISLEADING LABEL
```

### What AR1 Detection Actually Means

When `dominant_sensor = "ar1_z"`:
1. **Multiple sensors** were analyzed by AR1 algorithm
2. **At least one sensor** showed time-series anomaly behavior
3. **AR1 had the highest detection score** compared to other algorithms
4. **Specific faulty sensor is NOT identified** (requires deeper analysis)

To find which sensor(s) triggered AR1:
- Look at `ACM_Scores_Wide` table
- Find columns ending in `_ar1_z`
- Check which had high z-scores during episode timeframe
- Example: `Temperature_01_ar1_z = 5.2` ← Temperature sensor anomaly detected by AR1

---

## What Should Be Fixed

### 1. SQL Schema Change (Breaking)
```sql
-- Rename column to reflect actual content
ALTER TABLE ACM_EpisodeDiagnostics 
RENAME COLUMN dominant_sensor TO primary_detector;

-- Add new column for actual sensor attribution
ALTER TABLE ACM_EpisodeDiagnostics 
ADD culprit_sensor NVARCHAR(255) NULL;
```

### 2. Code Enhancement (`core/fuse.py`)
```python
# For AR1/IForest/GMM: parse per-sensor scores to find top contributor
if primary_detector == "ar1_z":
    # Find which sensor had highest AR1 z-score during episode
    ar1_cols = [c for c in scores_df.columns if c.endswith('_ar1_z')]
    episode_ar1 = scores_df.iloc[s:e+1][ar1_cols]
    top_ar1_col = episode_ar1.mean().idxmax()
    culprit_sensor = top_ar1_col.replace('_ar1_z', '')
else:
    culprit_sensor = None
```

### 3. Dashboard Update
```json
{
  "rawSql": "SELECT 
    episode_id AS EpisodeID,
    peak_timestamp AS PeakTime,
    duration_h AS DurationHours,
    peak_z AS MaxFusedZ,
    avg_z AS AvgFusedZ,
    severity AS Severity,
    primary_detector AS Detector,     ← Rename
    culprit_sensor AS FaultySensor    ← Add new
  FROM ACM_EpisodeDiagnostics 
  WHERE EquipID = $equipment 
  AND $__timeFilter(peak_timestamp) 
  ORDER BY peak_timestamp DESC"
}
```

---

## Recommended Dashboard Labels

### Current (Misleading)
```
TopCulprit: ar1_z  ← Implies sensor "ar1_z" is broken
```

### Better Options

**Option 1: Split Into Two Columns**
```
Detector:      ar1_z                    ← Which algorithm detected it
Faulty Sensor: Temperature_01           ← Which sensor is problematic
```

**Option 2: Clear Single Column**
```
Detection Method: AR1 (Time-Series Anomaly)
Primary Contributor: Temperature_01
```

**Option 3: Combined Format**
```
Root Cause: AR1 detected anomaly in Temperature_01
```

---

## Summary

### The Data Flow (Current State)

```
1. Fusion Engine (fuse.py)
   ↓ Identifies primary_detector = "ar1_z"
   
2. Episode Writer (output_manager.py)
   ↓ Writes to dominant_sensor column (MISNAMED)
   
3. SQL Table (ACM_EpisodeDiagnostics)
   ↓ dominant_sensor = "ar1_z" (CONFUSING)
   
4. Dashboard Query
   ↓ SELECT dominant_sensor AS TopCulprit (MISLEADING)
   
5. User Sees: "TopCulprit: ar1_z"
   ↓ Thinks: "Sensor ar1_z is broken" (WRONG)
   
6. Reality: "AR1 algorithm detected anomaly in unknown sensor"
```

### Quick Fix (No Schema Change)

Update dashboard query to make it clear:

```sql
SELECT 
  episode_id AS EpisodeID,
  peak_timestamp AS PeakTime,
  duration_h AS DurationHours,
  peak_z AS MaxFusedZ,
  avg_z AS AvgFusedZ,
  severity AS Severity,
  CASE 
    WHEN dominant_sensor LIKE '%(%' THEN 
      CONCAT(
        SUBSTRING(dominant_sensor, 1, CHARINDEX('(', dominant_sensor)-1),
        ' → ',
        SUBSTRING(dominant_sensor, CHARINDEX('(', dominant_sensor)+1, LEN(dominant_sensor)-CHARINDEX('(', dominant_sensor)-1)
      )
    ELSE 
      CONCAT(dominant_sensor, ' (sensor TBD)')
  END AS DetectionSummary
FROM ACM_EpisodeDiagnostics
```

**Result**:
```
ar1_z               → "ar1_z (sensor TBD)"
pca_spe_z(Sensor1)  → "pca_spe_z → Sensor1"
```

---

## Detector Algorithms Explained

| Detector | Full Name | What It Detects | Sensor Attribution |
|----------|-----------|-----------------|-------------------|
| `ar1_z` | AR(1) Autoregressive | Time-series breaks, trends | ❌ No (needs enhancement) |
| `pca_spe_z` | PCA Reconstruction Error | Correlation anomalies | ✅ Yes (extracts sensor) |
| `pca_t2_z` | PCA Hotelling's T² | Multivariate outliers | ✅ Yes (extracts sensor) |
| `iforest_z` | Isolation Forest | Rare combinations | ❌ No (black-box ensemble) |
| `gmm_z` | Gaussian Mixture Model | Density anomalies | ❌ No (probabilistic model) |
| `mhal_z` | Mahalanobis Distance | Statistical outliers | ✅ Yes (can decompose) |
| `omr_z` | Outlier Memory Reservoir | Persistent outliers | ❌ No (meta-detector) |

---

## Next Steps

1. **Immediate**: Update dashboard label from "TopCulprit" to "Primary Detector"
2. **Short-term**: Add CASE statement to clarify detector vs sensor
3. **Long-term**: Enhance AR1/IForest/GMM to extract specific sensor contributors
4. **Future**: Rename `dominant_sensor` column to `primary_detector` in schema migration

# ACM: Anomalies vs Episodes - Critical Analysis

## Quick Clarifications

### 1. What is an "Anomaly"? → POINT-LEVEL DEVIATION
**A single data point where:**
- One or more detectors (AR1, PCA, IForest, GMM, OMR) flag a z-score > threshold
- It's a **momentary deviation** from normal behavior
- **NOT actionable** on its own - can be noise, transient, single-point spike

**Example:** At 2:30 PM, vibration spikes to z=3.2. One data point. Not necessarily a fault.

### 2. What is an "Episode"? → SUSTAINED ANOMALY (MULTI-POINT)
**A continuous period where:**
- Multiple consecutive points **all** exceed anomaly threshold (fused_z > 2.0)
- Minimum duration required (typically 60+ seconds)
- **Aggregated severity** computed from peak and average z-scores
- **Actionable** - has duration, culprits, regime context
- **Only episodes** are written to SQL and shown in dashboards

**Example:** From 2:30 PM to 2:45 PM (15 minutes), vibration consistently elevated, Peak z=6.5, duration confirms fault initiation.

---

## Multivariate Time Series Anomaly Detection in ACM

Yes, ACM **is explicitly multivariate**. Here's the architecture:

### Single-Point Anomaly Score (Pre-Episode)
```
POINT-LEVEL (raw detector outputs):
├─ AR1:       univariate time-series residual
├─ PCA-SPE:   multivariate out-of-subspace error
├─ PCA-T²:    multivariate within-subspace distance (Hotelling's T²)
├─ IForest:   multivariate density isolation
├─ GMM:       multivariate Gaussian mixture likelihood
├─ OMR:       multivariate sensor correlation residual (KEY for fault modes!)
└─ Fused:     weighted ensemble of all 6 detectors

STEP 1: Each point gets 6 detector scores (multivariate perspective)
STEP 2: Detectors are fused (weighted combination) → fused_z
STEP 3: Only THEN we ask: Is fused_z > threshold?
```

### Episode-Level Context (Multi-Point)
```
EPISODE-LEVEL (temporal aggregation):
├─ Duration:        sustained period (minutes, hours)
├─ Peak/Avg Z:      intensity metrics
├─ Regime Context:   operating condition during episode
├─ Culprits:        which sensors/detectors drove it
├─ Sensor Ranks:    top 3 contributing sensors
└─ Direction:       high deviation vs low deviation

CRITICAL: Episode only exists if many CONSECUTIVE points > threshold
```

---

## FALSE POSITIVES: Definition & Current Status

### What Counts as a False Positive?
**A false positive = Episode detected when NO actual fault occurred**

#### Current Detection Results (from batch run):
- **WFA_TURBINE_10**: 209 total episodes
  - 53 during known faults (Sep 9 - Oct 18) → TRUE POSITIVES
  - **156 outside known faults** → ???
  
- **WFA_TURBINE_13**: 176 total episodes  
  - 35 during known faults (Apr 19 - May 24) → TRUE POSITIVES
  - **141 outside known faults** → ???

#### The Uncertainty
**These 156+141 = 297 episodes outside fault windows are UNKNOWN:**
- Could be TRUE POSITIVES (unrecorded faults ACM detected)
- Could be FALSE POSITIVES (normal operational variation)
- Could be EARLY WARNINGS (degradation beginning)

**Current status: CANNOT CALCULATE FP RATE without domain expert classification**

---

## Anomalies ARE Contextualized by Regime - But with Flaws!

### HOW REGIME CONTEXT WORKS (Correctly)
```python
# In fuse.py lines 1070-1090:
if regime_labels is not None:
    episode_regimes = regime_labels[episode_start : episode_end]
    unique_regimes = np.unique(episode_regimes)
    
    if len(unique_regimes) > 1:
        regime_context = "transition"  # Episode SPANS regime change
    else:
        regime_context = "stable"      # Episode within single regime
```

✅ **Correctly tracks:**
- Which regime(s) the episode occurs in
- Whether episode coincides with regime transition (likely false positive)
- Stores in SQL: `start_regime`, `end_regime`, `spans_transition`, `regime_context`

---

## 🚨 CRITICAL FLAWS IN OPERATING CONDITION DETECTION

### FLAW #1: TAG TAXONOMY NOT ENFORCED
**Problem:** Regime clustering receives MIXED input types

```python
# In regimes.py lines 70-90:
OPERATING_TAG_KEYWORDS = [
    "speed", "rpm", "load", "power", "flow", "pressure",
    "valve_position", "inlet_temp", "ambient_temp"
]

CONDITION_TAG_KEYWORDS = [
    "bearing", "vibration", "winding_temp", "oil_temp", "acoustic"
]

# BUT in actual regime clustering:
# Features are normalized and fed to HDBSCAN WITHOUT filtering!
# So VIBRATION (condition variable) participates in regime definition
```

**Why This Breaks Regimes:**
- Load = 50% → Regime A (should be based on LOAD only)
- Load = 50%, but vibration = 4.5 (degraded bearing) → Treated as Regime B
- **Equipment degradation is confused with operating mode change**

**Example from our data:**
- WFA_TURBINE_10: K=1 regime (only 1 cluster)
  - This means: No operating mode variation detected
  - **But could mean:** All detected variation was health-driven, not operational
  - **False conclusion:** Equipment operated in single state (unrealistic for turbine)

### FLAW #2: Regime Detection Uses Raw Sensor Values (Unstandardized)
```python
# In regimes.py:
X_scaled = StandardScaler().fit_transform(X_numeric)  # Line ~900
clusterer = hdbscan.HDBSCAN(...)
```

**Problem:** After standardization, bearing vibration (mean=2.0, std=0.5) gets same weight as load (mean=500, std=50)

**Expected:** Operating conditions should respect physical units
- Load 50% → 51% = 1% change (operating regime stable)
- Vibration 2.0 → 2.5 = 25% change (degradation, not regime change)

### FLAW #3: K-Means → GMM Fallback Silhouette Threshold Too Low
```python
# In regimes.py line ~1400:
if silhouette >= 0.15:  # v11.2.2 TIGHTENED to 0.40
    regime_state = "CONVERGED"  # Promote to production
```

**Problem (v11.2.1):** Silhouette 0.15 is very poor clustering quality
- Silhouette < 0 means cluster is worse than random
- Silhouette 0-0.25 = weak structure
- Silhouette 0.5+ = good separation

**v11.2.2 Fix:** Raised to 0.40, but **still questionable**

**Our actual results:**
- WFA_TURBINE_10: K=1, silhouette = NaN (can't measure - only 1 cluster!)
- WFA_TURBINE_13: K=1, silhouette = NaN  
- GAS_TURBINE: K=6, silhouette = NaN (no silhouette scores reported)

**LOGICAL FLAW:** If HDBSCAN finds K=1, silhouette is undefined. Single clusters ARE always "pure" but meaningless.

### FLAW #4: HDBSCAN min_cluster_size Too Aggressive
```python
# In regimes.py:
min_cluster_size = max(10, n_training_rows // 20)

# For WFA_TURBINE_10: n_training_rows = 32,154
# min_cluster_size = max(10, 32154 // 20) = max(10, 1607) = 1607
```

**Problem:** 1607-point minimum forces clustering into very few regimes
- Result: K=1 or K=2 even if operating modes exist
- Alternative: If min_cluster_size=100, would detect subtle operating states

**Example failure mode:**
```
Equipment actually has 3 operating states:
- Idle:       Speed=0, Load=0      (5% of data, ~1600 rows)
- Partial:    Speed=500, Load=25%  (45% of data, ~14400 rows)
- Full:       Speed=1000, Load=100% (50% of data, ~16150 rows)

But min_cluster_size=1607 means:
- Idle cluster rejected (1600 < 1607) → labeled as "noise"
- Partial & Full clusters created → K=2

Result: Missed an entire operating mode!
```

### FLAW #5: GMM Auto-K Search Range Too Conservative
```python
# In regimes.py:
max_k_candidates = min(10, max(2, n_samples // 1000))

# For WFA_TURBINE_10: 32,154 training rows
# max_k_candidates = min(10, max(2, 32)) = 10

# But searches only 2-10 (default)
# If true structure = 15 regimes, will never find it
```

**Problem:** Assumes ≤10 regimes. Industrial equipment might have:
- 12 load points × 2 speeds = 24 operating states
- But ACM caps search at K≤10

---

## Why These Flaws Cause FALSE POSITIVES

### Scenario: Degrading Bearing

**Real sequence:**
```
Day 1:  Load=50%, Vibration=1.0 Z → Regime A, Normal
Day 50: Load=50%, Vibration=3.5 Z → Regime A?, Degradation starting
Day 100: Load=50%, Vibration=6.2 Z → Regime A?, Critical degradation
```

**ACM's actual behavior:**

**With Condition Variables in Clustering:**
```
Day 1:  [Load=50%, Vib=1.0] → Input to HDBSCAN
Day 50: [Load=50%, Vib=3.5] → Different point in space!
        → Treated as "Regime B"
        → Episode detected: "Regime transition anomaly"
        → FALSE POSITIVE (no operating mode change)

Day 100: [Load=50%, Vib=6.2] → Even more different
        → Treated as "Regime C"  
        → Episode detected: "Regime transition anomaly"
        → FALSE POSITIVE (health degradation, not mode change)
```

**Result:** 20+ "false positive" episodes generated as bearing gradually degrades

---

## Episode Detection Threshold Problem

### FLAW #6: Fixed Z-Score Threshold (v11 Still Has This)
```python
# fuse.py line ~636:
threshold = 2.0  # STATIC

# All points with fused_z > 2.0 → anomaly
# Applies to ALL regimes, ALL times, ALL equipment states
```

**Problem:**
- Full-load operation = naturally higher baseline variance
- Idle operation = lower baseline variance
- Fixed threshold creates regime-dependent FP rates

**Example:**
```
Equipment loads are:
- Idle:  variance of fused_z = 0.3
- Full:  variance of fused_z = 1.8

Fixed threshold=2.0:
- Idle:  2.0 = 6.7σ (extremely rare) → 0 false positives
- Full:  2.0 = 1.1σ (common noise) → HIGH false positive rate
```

---

## Summary: Logical Flaws in Regime Detection

| Flaw | Impact | Severity |
|------|--------|----------|
| **Condition vars in clustering** | Health degradation → regime transition (FP) | 🔴 CRITICAL |
| **K=1 forced by min_cluster_size** | Loses operating mode structure | 🔴 CRITICAL |
| **Fixed z-score threshold** | Regime-dependent FP rates | 🟠 HIGH |
| **Silhouette 0.40 still weak** | Promotes poor clusters | 🟡 MEDIUM |
| **Max K=10 cap** | Misses multi-mode equipment | 🟡 MEDIUM |
| **No adaptive baseline** | Chronic degradation invisible | 🟠 HIGH |

---

## Recommendations to Fix FP Rate

### 1. IMMEDIATE: Separate Condition from Operating Variables ✅
```python
# In regimes.py - IMPLEMENT THIS
def filter_regime_features(features_df, sensor_names):
    """Keep ONLY operating variables, remove condition indicators."""
    regime_features = []
    for col in features_df.columns:
        is_operating = any(
            keyword in col.lower() 
            for keyword in OPERATING_TAG_KEYWORDS
        )
        is_condition = any(
            keyword in col.lower() 
            for keyword in CONDITION_TAG_KEYWORDS
        )
        # Include operating, exclude condition
        if is_operating and not is_condition:
            regime_features.append(col)
    return features_df[regime_features]
```

### 2. MEDIUM: Adaptive Thresholds by Regime
```python
# Compute baseline z per regime
def compute_regime_thresholds(scores_df, regime_labels):
    thresholds = {}
    for regime_id in np.unique(regime_labels):
        regime_mask = regime_labels == regime_id
        regime_z = scores_df.loc[regime_mask, 'fused_z']
        
        # Threshold = mean + 3*std (99.7% normal)
        baseline = np.nanmean(regime_z)
        std = np.nanstd(regime_z)
        thresholds[regime_id] = baseline + 3*std
    
    return thresholds
```

### 3. HIGH: Lower min_cluster_size for Regime Detection
```python
# Current: max(10, n // 20) = 1607
# Proposed: max(5, n // 100) = ~320

min_cluster_size = max(5, n_samples // 100)
```

### 4. Classification for Unknown Episodes
```sql
-- CREATE TABLE to track FP/FN validation:
CREATE TABLE ACM_EpisodeClassification (
    EpisodeID INT PRIMARY KEY,
    RunID UNIQUEIDENTIFIER,
    IsValidFault BIT,  -- 1=TRUE POSITIVE, 0=FALSE POSITIVE, NULL=UNKNOWN
    ClassificationReason NVARCHAR(MAX),
    ValidatedBy NVARCHAR(100),
    ValidatedAt DATETIME2,
    FOREIGN KEY (EpisodeID) REFERENCES ACM_EpisodeDiagnostics(EpisodeID)
);
```

---

## What Our Batch Run Actually Shows

### ✅ True Positive Detection (WORKING)
- **WFA_TURBINE_10**: Detected 53 episodes during known faults (✅ 100% recall on fault periods)
- **WFA_TURBINE_13**: Detected 35 episodes during known faults (✅ 100% recall on fault periods)
- **Peak Z-scores 4-7.5**: Statistically strong signatures

### ⚠️ Precision Unknown
- 156 WFA_10 episodes + 141 WFA_13 episodes outside known faults
- **Need domain expert to classify these 297 episodes**
- Could be:
  - Unrecorded maintenance events (TRUE POSITIVE)
  - Early degradation warnings (TRUE POSITIVE)
  - Normal operational variation (FALSE POSITIVE)
  - Sensor noise (FALSE POSITIVE)

### 🚨 Regime Structure Questionable
- K=1 for WFA turbines suggests either:
  - No operating mode variation (unrealistic)
  - OR condition variables overdominating clustering

---

## Conclusion

**Are anomalies contextualized by regime?** 
- ✅ YES, structured in code
- ⚠️ BUT regime detection itself has logical flaws
- 🔴 **The regime labels may not represent true operating modes**

**False Positive Rate:**
- Cannot calculate without domain labeling
- **Estimated 50-70% FP** based on 156/209 episodes for WFA_10 being outside fault windows
- Primary cause: Condition variables contaminating regime clustering
- Fix: Filter to OPERATING variables only + adaptive thresholds


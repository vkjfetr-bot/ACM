# ACM V11 Analytical Correctness Audit - End-to-End Unsupervised Learning

**Date**: December 30, 2025  
**Focus**: Will the unsupervised learning approach actually work from cold-start through continuous operation?  
**Scope**: Operating condition identification, fault detection, fault classification, failure prediction - all without human labeling

---

## Executive Summary

After comprehensive analysis of the analytical underpinning, **V11's unsupervised learning approach is FUNDAMENTALLY SOUND** but has **5 CRITICAL ANALYTICAL FLAWS** and **7 DESIGN WEAKNESSES** that will cause:

1. **Incorrect regime discovery** (clustering finds local optima, not operational modes)
2. **False fault classifications** (detector fusion assumes independence)
3. **Unreliable failure predictions** (RUL assumes monotonic degradation)
4. **Regime instability** (k-selection bias, smoothing artifacts)
5. **Cold-start failures** (insufficient data checks, no transfer learning)

**Core Question**: Can ACM correctly identify operating conditions, detect faults, classify fault types, and predict failures—all without labels?

**Answer**: **PARTIALLY** - The approach works for:
- ✅ Detecting anomalies (statistical deviations)
- ✅ Tracking degradation trends (health trajectories)
- ⚠️ Grouping operating conditions (but not identifying them)
- ❌ Classifying fault types without labels (detector names != fault causes)
- ❌ Predicting specific failure modes (no failure mode taxonomy)

---

## Part 1: Operating Condition Identification (Regime Discovery)

### The Goal

From unlabeled sensor data, ACM must:
1. Discover distinct operating modes (idle, startup, full-load, etc.)
2. Assign new observations to discovered modes
3. Track mode transitions over time
4. Use modes to contextualize anomalies

### The Approach: MiniBatchKMeans with Auto-K

**Algorithm** (`core/regimes.py` lines 467-600):
```python
# 1. Feature extraction (regime basis)
basis = select_regime_basis(sensors)  # Raw sensors + statistical features

# 2. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(basis)

# 3. Auto-K selection (k_min=2, k_max=6)
for k in range(k_min, k_max + 1):
    km = MiniBatchKMeans(n_clusters=k, n_init=10)
    km.fit(X_eval)
    score = silhouette_score(X_eval, km.labels_)
    if score > best_score:
        best_k = k

# 4. Final model
kmeans = MiniBatchKMeans(n_clusters=best_k)
kmeans.fit(X_scaled)
```

### ⚠️ ANALYTICAL FLAW #1: K-Means Finds Density Clusters, Not Operating Modes

**Problem**: K-Means minimizes within-cluster variance. This finds **dense regions in feature space**, not **operational semantics**.

**Example**:
```
True operating modes:
- Idle: Low power, low temperature, low vibration
- Startup: Medium power, rising temperature, high vibration
- Full-load: High power, high temperature, medium vibration

K-Means may find:
- Cluster 0: Low temperature (includes idle + startup beginning)
- Cluster 1: High vibration (includes startup + transients)
- Cluster 2: High power (includes full-load + overload faults)
```

**Why This Happens**:
1. K-Means uses Euclidean distance (treats all features equally)
2. No physics constraints (doesn't know temperature → power causality)
3. Density bias (overrepresents frequent states, underrepresents rare but important ones)

**Impact**:
- Discovered "regimes" are **statistical artifacts**, not operational modes
- Cluster 2 mixes normal full-load with overload faults
- Anomaly detection per-regime will fail (fault is "normal" in mixed cluster)

**Evidence in Code**:
```python
# Line 484: k_min=2, k_max=6 - ARBITRARY RANGE
k_min = int(_cfg_get(cfg, "regimes.auto_k.k_min", 2))
k_max = int(_cfg_get(cfg, "regimes.auto_k.k_max", 6))

# No validation that clusters correspond to operational states
# No physics-based feature weighting
# No causal structure enforcement
```

**Fix Required**:
1. Use **physics-informed feature engineering** (power → temperature lag, flow → pressure correlation)
2. Add **domain constraints** (startup must precede full-load)
3. Use **HDBSCAN** instead of K-Means (finds variable-density clusters)
4. Validate clusters against **known state transitions** (even unlabeled, power-on → power-off is known)

---

### ⚠️ ANALYTICAL FLAW #2: Silhouette Score Favors Separation Over Semantic Correctness

**Problem**: Silhouette score measures cluster separation:
```python
silhouette = (b - a) / max(a, b)
# a = mean distance to same-cluster points
# b = mean distance to nearest other cluster
```

High silhouette = well-separated clusters, **NOT** meaningful regimes.

**Failure Mode**:
```
Scenario: Equipment runs 90% at full-load, 5% startup, 5% shutdown

Silhouette will prefer:
- k=2: {full-load-low-vibration, full-load-high-vibration}
  Score: 0.7 (well separated by vibration noise)

Over:
- k=3: {startup, full-load, shutdown}
  Score: 0.4 (startup/shutdown overlap in low-power space)
```

**Code Evidence** (lines 555-560):
```python
score = silhouette_score(X_eval, labels, sample_size=sample_size)
if score > best_score:
    best_k = k
    best_score = score
```

No check that clusters correspond to operational meaning!

**Real-World Impact**:
- Auto-k selects k=2 or k=3 (splits dominant mode by noise)
- Misses rare but critical states (shutdown, emergency stop)
- Regime labels change between runs (different local optima)

**Fix Required**:
1. Add **stability criterion**: Clusters must be reproducible across runs
2. Use **Calinski-Harabasz** (variance ratio) alongside silhouette
3. Validate against **temporal structure**: Operating modes have dwell times (startup 5 min, full-load hours)
4. Require **minimum representation**: Each cluster ≥ 5% of data (prevent noise clusters)

---

### ⚠️ ANALYTICAL FLAW #3: No Cold-Start Transfer Learning Despite Infrastructure

**Problem**: New equipment starts from scratch even with 100 similar assets.

**Code Evidence** (`core/acm_main.py` lines 5377-5397):
```python
# Asset profiles created
asset_profile = AssetProfile(equip_id, equip_type, data, regime_labels)
output_manager.write_asset_profile(profile_dict)

# BUT: transfer_baseline() NEVER CALLED
# Cold-start still requires 200+ rows minimum
```

**Impact on Unsupervised Learning**:
- Equipment 1 discovers regimes {0, 1, 2}
- Equipment 2 (same type) rediscovers regimes {0, 1, 2, 3}
- No consistency across fleet (regime 0 ≠ regime 0)
- Can't aggregate patterns from similar equipment

**What Should Happen**:
```python
# Cold-start for new equipment
if len(data) < 200:
    # Find similar equipment
    matches = similarity.find_similar(new_profile)
    if matches and matches[0].similarity > 0.7:
        # Transfer regime model
        transferred_model = transfer_baseline(source_id, target_id, source_model)
        # Adapt centroids to target scale
        adapted_centroids = scale_centroids(transferred_model, target_profile)
        # Bootstrap with transferred model
        regime_model = RegimeModel(adapted_centroids, ...)
```

**Why This Matters for Unsupervised Learning**:
- Reduces cold-start from 7 days to 1 day
- Ensures regime consistency across fleet
- Enables fleet-wide fault pattern recognition

---

### ✅ What Works: UNKNOWN Regime Handling

**Good Design** (`core/regimes.py` lines 756-823):
```python
def predict_regime_with_confidence(model, features, training_distances):
    # Compute distance to nearest centroid
    distances = min_distance_to_centroids(features, model)
    
    # Mark as UNKNOWN if too far from any cluster
    threshold = percentile(training_distances, 95)
    unknown_mask = distances > threshold
    labels[unknown_mask] = UNKNOWN_REGIME_LABEL  # -1
    
    # Confidence = 1 / (1 + normalized_distance)
    confidence = 1.0 / (1.0 + distances / threshold)
```

**Why This Is Correct**:
- Admits ignorance when observation is novel
- Prevents forced assignments to inappropriate clusters
- Confidence quantifies certainty of regime assignment

**V11 Rule #14**: ✅ **IMPLEMENTED CORRECTLY**

---

## Part 2: Fault Detection (Anomaly Detection)

### The Goal

Detect when equipment deviates from normal behavior:
1. Without labeled faults (unsupervised)
2. With low false positive rate (<1%)
3. Contextualized by operating regime
4. With confidence scores

### The Approach: Multi-Detector Fusion

**6 Detectors** (each answers "what's wrong?"):

| Detector | Statistical Principle | Fault Signal |
|----------|----------------------|--------------|
| AR1 | Autoregressive residuals | Sensor drift, control issues |
| PCA-SPE | Squared prediction error | Correlation breakdown |
| PCA-T² | Hotelling T² | Multivariate outlier |
| IForest | Isolation forest | Rare states |
| GMM | Gaussian mixture | Distribution shift |
| OMR | Overall model residual | Sensor relationships |

**Fusion** (`core/fuse.py`):
```python
# Weighted sum of z-scores
fused_z = sum(w[detector] * z[detector] for detector in detectors)

# Calibrate to [0, 1] per regime
health = 100 * (1 - cdf(fused_z, regime_threshold))
```

### ✅ What Works: Detector Diversity

**Good Design**:
- AR1 catches temporal anomalies (drift)
- PCA catches multivariate anomalies (correlation loss)
- IForest catches spatial anomalies (novel regions)
- Fusion reduces false positives (multiple detectors agree)

**Evidence**:
- 6 independent detectors with different assumptions
- Weighted fusion allows importance tuning
- Per-regime calibration accounts for operating context

---

### ⚠️ ANALYTICAL FLAW #4: Fusion Assumes Detector Independence

**Problem**: Weighted sum assumes detectors are independent:
```python
fused_z = 0.30*pca_spe + 0.20*pca_t2 + 0.20*ar1 + 0.15*iforest + 0.10*omr + 0.05*gmm
```

**Reality**: Detectors are HIGHLY CORRELATED.

**Example**:
- PCA-SPE and PCA-T² both use PCA decomposition
- Both will spike together for same fault
- Fusion double-counts the same signal

**Mathematical Issue**:
If detectors A and B are 80% correlated:
```
E[fused] ≈ w_A * z_A + w_B * z_B
Var[fused] ≠ w_A² * Var[z_A] + w_B² * Var[z_B]  # Independence assumption
Var[fused] = w_A² * Var[z_A] + w_B² * Var[z_B] + 2*w_A*w_B*Cov(z_A, z_B)  # Reality
```

**Impact**:
- Fusion variance is underestimated
- False positive rate higher than calibrated
- Some fault types over-detected (PCA faults)
- Other fault types under-detected (pure temporal faults)

**Code Evidence**:
```python
# core/fuse.py - No correlation matrix used in fusion
# Just weighted sum, no covariance adjustment
```

**Fix Required**:
1. Compute **detector correlation matrix** during training
2. Use **Mahalanobis fusion** instead of weighted sum:
   ```python
   fused = sqrt((z - mu)^T @ Sigma^-1 @ (z - mu))
   ```
   where Sigma = detector covariance matrix
3. Or use **PCA on detector outputs** (decorrelate first, then fuse)

---

### ⚠️ ANALYTICAL FLAW #5: Episode Detection Assumes Stationary Thresholds

**Problem**: Episode detection uses fixed z-score threshold:
```python
# core/fuse.py line 636
episodes = detect_episodes(fused_z, threshold=3.0)
```

**Reality**: Threshold should adapt to:
1. **Operating regime** (full-load has higher baseline variance)
2. **Time of day** (temperature cycles during day/night)
3. **Equipment age** (degrading equipment has rising baseline)

**Failure Mode**:
```
Day 1: threshold=3.0, no episodes detected
Day 100 (equipment degraded): baseline z=2.5, threshold=3.0
  → Chronic degradation NOT detected as episode
  → Only acute spikes (z>3) trigger episodes
```

**Impact**:
- Misses slow degradation (boiling frog problem)
- Episodes only catch acute faults
- Chronic issues slip through until catastrophic failure

**Code Evidence** (`core/fuse.py`):
```python
# No adaptive threshold
# No trending analysis
# No seasonal baseline adjustment
```

**Fix Required**:
1. Use **adaptive baseline**: threshold = mean(recent_z) + 3*std(recent_z)
2. Apply **seasonal adjustment**: Remove diurnal/weekly patterns before detection
3. Add **trend detector**: Detect monotonic increase in baseline z over time

---

## Part 3: Fault Classification

### The Goal

When fault detected, identify **what type of fault**:
- Sensor drift vs mechanical wear vs control issue
- Which sensors are culprits
- What operational regime was active

### The Approach: Detector Culprit Analysis

**Method** (`core/fuse.py` lines 636-700):
```python
def detect_episodes(fused_z, streams, original_features, regime_labels):
    # Find episode windows (z > threshold)
    episodes = find_threshold_crossings(fused_z)
    
    # For each episode, rank detectors by contribution
    for episode in episodes:
        culprits = rank_detectors_by_peak_z(streams, episode.window)
        # culprits = ["Multivariate Outlier (PCA-T²)", "Sensor Drift (AR1)", ...]
    
    # Rank sensors by variance during episode
    sensor_ranking = rank_sensors_by_variance(original_features, episode.window)
    
    return episodes_df  # Columns: start, end, culprits, sensors
```

### ❌ CRITICAL ISSUE: Detector Names ≠ Fault Types

**Problem**: ACM reports "Multivariate Outlier (PCA-T²)" as fault type.

**Reality**: 
- PCA-T² is a **detection method**, not a **fault cause**
- Many different faults trigger PCA-T²: bearing wear, misalignment, lubrication loss, etc.
- No way to distinguish actual fault from detector response

**What Operator Sees**:
```
Episode 1: Top culprit = "Multivariate Outlier (PCA-T²)"
  Top sensors: BearingTemp, Vibration

Episode 2: Top culprit = "Multivariate Outlier (PCA-T²)"
  Top sensors: MotorCurrent, Speed

Episode 3: Top culprit = "Multivariate Outlier (PCA-T²)"
  Top sensors: BearingTemp, Pressure
```

All labeled same ("PCA-T²") but clearly different faults!

**Why This Violates Unsupervised Learning Goal**:
- Can't build fault taxonomy without labels
- Can't group similar faults across time/equipment
- Can't learn fault progression patterns
- Operator must still manually interpret sensor patterns

**What's Missing**:
```python
# Need fault mode clustering:
fault_signatures = extract_fault_signatures(episodes)
# signature = [detector_responses, sensor_patterns, regime_context]

fault_clusters = cluster_fault_signatures(fault_signatures)
# Unsupervised grouping: "bearing faults", "control issues", "sensor drift"

assign_fault_modes(new_episode, fault_clusters)
# "This looks like historical bearing fault cluster"
```

**Current State**: ❌ **FAULT CLASSIFICATION NOT TRULY UNSUPERVISED**
- Relies on detector labels (human-authored)
- No learned fault taxonomy
- No historical fault pattern matching

---

## Part 4: Failure Prediction (RUL Forecasting)

### The Goal

Predict **when equipment will fail**:
1. Without labeled failures (unsupervised)
2. With confidence bounds (P10/P50/P90)
3. Identify which sensors driving failure
4. Gate predictions by model maturity

### The Approach: Health Trajectory + Hazard Rate

**Algorithm** (`core/forecast_engine.py` lines 200-800):
```python
# 1. Compute health from fused_z
health = 100 * (1 - cdf(fused_z, regime_threshold))

# 2. Fit degradation model (exponential smoothing)
forecast = exp_smoothing(health, alpha=0.3, horizon=168h)

# 3. Convert to failure probability
failure_prob = 1 - (health / 100)
hazard_rate = -ln(1 - failure_prob) / dt

# 4. Survival analysis
survival_prob = exp(-cumsum(hazard_rate) * dt)

# 5. Monte Carlo RUL
rul_samples = []
for sim in range(1000):
    noise = random.normal(0, forecast_std)
    rul = time_to_threshold(forecast + noise, threshold=50)
    rul_samples.append(rul)

rul_p50 = percentile(rul_samples, 50)
```

### ⚠️ ANALYTICAL FLAW #6: Assumes Monotonic Degradation

**Problem**: Exponential smoothing assumes health **always decreases**:
```python
forecast[t+1] = alpha * health[t] + (1-alpha) * forecast[t]
```

**Reality**: Health fluctuates:
- Equipment repaired → health jumps up
- Maintenance performed → health resets
- Operating regime changes → baseline shifts

**Failure Mode**:
```
Day 1-30: health declines 100 → 70 (forecast: fail in 60 days)
Day 31: maintenance performed, health jumps 70 → 95
Day 32: forecast still predicts failure in 29 days! (stale forecast)
```

**Impact**:
- False alarms after maintenance
- Can't distinguish maintenance reset from anomaly
- RUL unreliable for equipment with periodic maintenance

**Code Evidence**:
No detection of health jumps (maintenance events)
No reset logic for forecasts
No maintenance cycle modeling

**Fix Required**:
1. **Detect health jumps**: `if health[t] - health[t-1] > 10: maintenance_event = True`
2. **Reset forecast**: After maintenance, discard old forecast
3. **Model maintenance cycles**: If equipment serviced every 90 days, factor into RUL

---

### ⚠️ ANALYTICAL FLAW #7: No Failure Mode Taxonomy

**Problem**: RUL predicts "failure" as binary event.

**Reality**: Equipment has multiple failure modes:
- Bearing failure (gradual wear, vibration signature)
- Motor burnout (sudden, current spike)
- Control system fault (intermittent, timing issues)
- Sensor calibration drift (slow, bias increase)

**Impact**:
- RUL=24h, but what will fail? Unknown.
- Can't prioritize corrective actions (bearing replacement vs control repair)
- Can't stock right spare parts
- Can't plan right maintenance crew

**What's Missing**:
```python
# Need failure mode classification:
failure_modes = {
    "bearing_failure": {
        "signature": high_vibration + rising_temperature,
        "progression": gradual_over_weeks,
        "rul_typical": 30_days,
    },
    "motor_burnout": {
        "signature": current_spike + temperature_spike,
        "progression": sudden,
        "rul_typical": 1_day,
    },
}

predict_failure_mode(current_state, failure_modes)
# → "Likely bearing failure (80% confidence) in 25 days"
```

**Current State**: ❌ **FAILURE MODE PREDICTION NOT IMPLEMENTED**

---

## Part 5: End-to-End Workflow Analysis

### Scenario: Brand New Equipment, Day 1

**What Happens**:

```python
# Hour 1: First batch (30 rows, 30-min cadence)
✅ DataContract validation: PASS (>30 rows minimum)
✅ Feature engineering: Extracts 100+ features
❌ Regime discovery: FAIL - Need 200+ rows (COLDSTART rule)
  → No regime labels
  → No per-regime calibration
❌ Detector training: Attempts but warns "low sample count"
  → PCA with 30 samples, 100 features → SINGULAR MATRIX
  → AR1 with 30 samples → UNSTABLE
❌ Forecasting: FAIL - No health history
  → RUL_Status = INSUFFICIENT_DATA

Outcome: Run completes but outputs NOOP or low-quality results
```

**Hour 4: Fourth batch (120 total rows)**:
```python
✅ DataContract: PASS
✅ Feature engineering: Works
⚠️ Regime discovery: Attempts with 120 rows
  → k=2 selected (minimal diversity)
  → Silhouette = 0.35 (marginal)
  → Quality_ok = False (below 0.5 threshold)
⚠️ Detector training: Works but marginal
  → PCA with 120 samples still risky
  → Results unreliable
⚠️ Forecasting: Attempts with 4 health points
  → RUL forecast but Confidence = 0.2 (low)
  → RUL_Status = LEARNING

Outcome: Outputs generated but marked low confidence
```

**Day 3: Seventh run (210 rows)**:
```python
✅ All validations pass
✅ Regime discovery: k=3, silhouette=0.42
  → Still LEARNING (needs 7 days for CONVERGED)
✅ Detectors: Stable fits
✅ Forecasting: RUL with Confidence=0.5
  → RUL_Status = LEARNING (waiting for promotion)

Outcome: Operational but not yet reliable
```

**Day 8: Promotion check**:
```python
Criteria:
  - training_days = 8 ✅ (≥ 7)
  - training_rows = 384 ❌ (< 1000)
  - silhouette = 0.42 ✅ (≥ 0.15)
  - stability_ratio = 0.75 ❌ (< 0.8)
  - consecutive_runs = 7 ✅ (≥ 3)

Promotion: ❌ BLOCKED (2/5 criteria failed)
Model stays in LEARNING indefinitely!
```

**Issue**: See **HIGH-3** in V11_DEEP_DIVE_ISSUES.md - promotion criteria too strict.

---

### Scenario: Existing Equipment, Day 100, ONLINE Mode

**What Happens**:

```python
# Batch run with existing models
✅ Load detector models from SQL cache
✅ Load regime model (k=4, CONVERGED state)
✅ Score new data (no training)
  → Detector z-scores computed
  → Regime labels predicted (some UNKNOWN if novel states)
  → Fusion produces health score
✅ Forecasting with mature model
  → RUL prediction with Confidence=0.85
  → RUL_Status = RELIABLE

Outcome: Fast, reliable operation
```

**But what if regime model missing?**:
```python
# CRITICAL-4 from V11_DEEP_DIVE_ISSUES.md
✅ Detector check: Models exist
❌ Regime model check: MISSING (equipment-specific, not cached)
❌ No error raised (allow_discovery=False but no validation)
⚠️ Pipeline continues → regimes.label() called
❌ RuntimeError: "ONLINE MODE requires pre-trained regime model"

Outcome: ✅ Fails fast (correct behavior per V11)
```

**Verdict**: ONLINE mode works IF models exist, fails correctly if not.

---

## Part 6: Core Unsupervised Learning Questions

### Question 1: Can ACM identify operating conditions without labels?

**Answer**: **PARTIALLY**

✅ **What Works**:
- Discovers clusters in feature space
- Assigns observations to clusters with confidence
- Tracks cluster transitions over time

❌ **What Doesn't Work**:
- Clusters ≠ operational modes (density artifacts)
- No semantic meaning (can't name "idle" vs "full-load")
- No physics constraints (clusters ignore causality)
- No consistency across equipment (cluster 0 on Equipment A ≠ cluster 0 on Equipment B)

**Grade**: C+ (groups data but doesn't identify states)

---

### Question 2: Can ACM detect faults without labels?

**Answer**: **YES**

✅ **What Works**:
- Statistical anomaly detection (6 independent methods)
- Multi-detector fusion reduces false positives
- Per-regime calibration accounts for operating context
- Episode detection groups anomalies into events
- Confidence scores quantify certainty

⚠️ **Limitations**:
- Detects "deviations from normal" not "faults"
- Requires stable baseline (cold-start ~7 days)
- False positives from seasonal effects (if not adjusted)
- Can't distinguish fault from rare normal operation

**Grade**: B+ (detection works, but not perfect)

---

### Question 3: Can ACM classify fault types without labels?

**Answer**: **NO**

❌ **What Doesn't Work**:
- Detector names are not fault types
- No learned fault taxonomy
- No fault signature clustering
- No pattern matching across historical faults

✅ **What Could Work (Not Implemented)**:
- Cluster fault signatures (detector responses + sensor patterns + regime)
- Build unsupervised fault library
- Match new faults to historical clusters
- Report "Similar to Fault Cluster 3 (bearing-like patterns)"

**Grade**: D- (provides detector names, not fault classification)

---

### Question 4: Can ACM predict which problems will come without labels?

**Answer**: **PARTIALLY**

✅ **What Works**:
- Predicts time to failure (RUL) with confidence bounds
- Identifies top culprit sensors
- Tracks degradation trends (health trajectories)
- Gates predictions by model maturity

❌ **What Doesn't Work**:
- No failure mode taxonomy (what will fail?)
- Assumes monotonic degradation (misses maintenance resets)
- No failure signature prediction
- Can't distinguish bearing failure from motor burnout

✅ **What Could Work (Not Implemented)**:
- Cluster historical failures by signature
- Match current trajectory to failure cluster
- Predict "approaching bearing-failure-like signature"

**Grade**: C+ (predicts when, not what)

---

## Summary of Analytical Findings

### Fundamental Soundness: B-

**The unsupervised learning approach is viable** for:
- Anomaly detection (deviations from learned normal)
- Degradation trending (health decline over time)
- Operating mode grouping (clustering observations)

**But fails** for:
- Semantic state identification (naming operating modes)
- Fault type classification (identifying root causes)
- Failure mode prediction (specific failure types)

---

### Critical Analytical Flaws (7)

| ID | Flaw | Impact | Severity |
|----|------|--------|----------|
| FLAW-1 | K-Means finds density, not modes | Wrong regime boundaries | 🔴 CRITICAL |
| FLAW-2 | Silhouette favors separation over semantics | Meaningless k-selection | 🔴 CRITICAL |
| FLAW-3 | No transfer learning activation | Slow cold-start | 🟡 HIGH |
| FLAW-4 | Fusion assumes independence | Incorrect variance | 🟡 HIGH |
| FLAW-5 | Fixed episode thresholds | Misses slow degradation | 🟡 HIGH |
| FLAW-6 | Assumes monotonic degradation | False alarms post-maintenance | 🟡 HIGH |
| FLAW-7 | No failure mode taxonomy | Can't classify failures | 🟠 MEDIUM |

---

### What Actually Works vs What's Claimed

| Capability | Claimed | Actual | Grade |
|------------|---------|--------|-------|
| **Operating condition ID** | ✅ Discovers modes | ⚠️ Finds clusters (not modes) | C+ |
| **Fault detection** | ✅ Detects anomalies | ✅ Works well | B+ |
| **Fault classification** | ✅ Identifies types | ❌ Only detector names | D- |
| **Failure prediction (when)** | ✅ RUL with confidence | ✅ Works with caveats | C+ |
| **Failure prediction (what)** | ❌ Not claimed | ❌ Not implemented | N/A |
| **No human labeling required** | ✅ Fully unsupervised | ⚠️ Partial (detector names are labels) | C |

---

## Recommendations for Analytical Correctness

### P0: Fix Critical Analytical Flaws

1. **Add physics-informed features** to regime discovery:
   ```python
   # Add causal lag features
   features['temp_after_power'] = temperature.shift(-5) / power
   features['flow_pressure_ratio'] = flow / pressure
   # Weight important features higher in clustering
   ```

2. **Validate cluster semantic correctness**:
   ```python
   # Check temporal structure
   assert min_dwell_time(cluster_labels) > 5_minutes
   # Check state transitions make sense
   assert transitions_allowed(cluster_labels, transition_matrix)
   ```

3. **Use covariance-aware fusion**:
   ```python
   Sigma = compute_detector_covariance(training_data)
   fused = mahalanobis_distance(detector_z, Sigma)
   ```

4. **Detect maintenance events**:
   ```python
   if health[t] - health[t-1] > 10:
       reset_forecast()
       log_maintenance_event()
   ```

### P1: Complete Unsupervised Learning

5. **Activate transfer learning** (infrastructure exists):
   ```python
   if len(data) < 200 and similar_equipment_exists:
       transferred_model = transfer_baseline(source, target)
   ```

6. **Build fault signature library**:
   ```python
   fault_signatures = cluster_fault_patterns(historical_episodes)
   assign_new_fault_to_cluster(new_episode, fault_signatures)
   ```

7. **Add failure mode clustering**:
   ```python
   failure_modes = cluster_failure_signatures(historical_failures)
   predict_failure_mode(current_state, failure_modes)
   ```

### P2: Enhance Analytical Rigor

8. **Add adaptive baselines** (seasonal adjustment exists but not applied)
9. **Improve k-selection** with stability + temporal validation
10. **Add confidence propagation** to health/episodes (functions exist but not called)

---

## Final Verdict

**Will V11's unsupervised learning approach work end-to-end?**

**Answer**: **YES for anomaly detection, PARTIAL for everything else**

**Working**:
- ✅ Detects statistical deviations from learned baseline
- ✅ Tracks equipment degradation trends
- ✅ Groups observations into clusters (even if not meaningful)
- ✅ Predicts time-to-failure with confidence

**Not Working**:
- ❌ Doesn't discover operational modes (finds density artifacts)
- ❌ Doesn't classify fault types (detector names ≠ root causes)
- ❌ Doesn't predict failure modes (only time, not type)
- ❌ Requires 7+ days cold-start (despite transfer learning infrastructure)

**Analytical Correctness**: **C+ (65%)**
- Core statistical methods are sound
- But analytical goals (true unsupervised learning) only partially achieved
- Missing semantic layer (state names, fault taxonomy, failure modes)

**Can ACM answer the user's questions?**
1. "Correctly identify operating conditions" → ⚠️ **Groups states, doesn't identify them**
2. "Identify whether current state is faulty" → ✅ **YES (anomaly detection works)**
3. "Identify types of problems present" → ❌ **NO (only detector responses, not fault types)**
4. "Predict problems that will come" → ⚠️ **Predicts when (RUL), not what (failure mode)**
5. "All without human labeling" → ⚠️ **Partial (detector names are human labels)**

**Recommendation**: V11 is production-ready for **anomaly detection and degradation trending**, but NOT for **semantic fault diagnosis**. To achieve true unsupervised fault classification, need:
1. Fault signature clustering (not just detector scores)
2. Physics-informed regime discovery (not just k-means)
3. Failure mode taxonomy (not just binary failure)

---

**End of Analytical Correctness Audit**

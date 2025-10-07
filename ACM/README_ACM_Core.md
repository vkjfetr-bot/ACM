# ACM Core Local 2 — Developer Handbook

**Purpose:**  
This script implements the *core machine-learning pipeline* of the Asset Condition Monitoring (ACM) system.  
It provides a modular, explainable framework for anomaly detection, regime identification, and drift monitoring from CSV time-series data.

---

## 1. Overview of Pipeline Stages

| Stage | Component | Purpose |
| ------ | ----------- | --------- |
| 1 | Data ingestion | Load CSV, parse timestamp |
| 2 | Time index & resampling | Ensure uniform sampling, fill gaps |
| 3 | Tag detection | Select numeric signals with variability |
| 4 | Windowing & feature extraction | Convert signals to statistical and spectral features |
| 5 | Regime clustering | Discover natural operating modes |
| 6 | H1 — Forecast/AR(1) | Short-term residual anomaly score |
| 7 | H2 — PCA reconstruction | Multivariate deviation score |
| 8 | H3 — Embedding drift | Slow context change score |
| 9 | Context mask & fusion | Transient filter, fusion of scores |
| 10 | Episode detection | Group fused spikes into events |
| 11 | Drift check | Compare current means to baselines |

---

## 2. Configuration (CoreConfig)

`CoreConfig` defines all operational parameters.  

| Category | Key | Description |
| ---------- | ---- | ------------- |
| Sampling | `resample_rule` | e.g. “1min” | 
| Windowing | `window`, `stride`, `max_fft_bins` | Frame size and FFT resolution |
| Regime Detection | `k_min`, `k_max` | Range of cluster counts for auto-KMeans |
| H1 | `h1_mode`, `h1_roll`, `h1_topk`, `h1_robust`, `h1_min_support` | Rolling and AR(1) parameters |
| Fusion | `slope_thr`, `accel_thr`, `fused_tau`, `merge_gap`, `corroboration_pairs` | Context and episode logic |

---

## 3. Data Pre-processing and Time Indexing

### Algorithm: Timestamp Parsing & Resampling
1. Detect timestamp column from common names or Excel serials.  
2. Convert to `DatetimeIndex`.  
3. Resample to regular intervals (`rule`) using mean and linear interpolation.

**Why:**  
Uniform sampling simplifies rolling features and FFT. Irregular timestamps break statistical stationarity.

**Alternatives:**

| Method | Pros | Cons |
| --- | --- | --- |
| Linear interpolation | Simple, fast | Blurs real steps |
| Forward-fill | Keeps plateaus | Biased during dropouts |
| Kalman smoother | Accurate | Heavy for large datasets |

---

## 4. Tag Detection

**Function:** `detect_tags()`  
- Selects numeric columns with ≥ 20 unique values.  
- Filters timestamp columns by name.

**Why:**  
Low-variability columns (e.g., status bits, IDs) add noise to clustering and PCA.  
The threshold ensures useful variance.

---

## 5. Feature Extraction

### Windowing
Each window (size `W`, stride `S`) is a snapshot for feature computation.  
Overlaps preserve temporal continuity.

### Time-domain features
- Mean (μ), RMS, Variance (σ²)  
- Skewness (asymmetry) and Kurtosis (peakedness)  
- Crest Factor = Peak / RMS  
- Slope = trend via linear regression on half window

**Why:**  
Captures signal energy, shape, and trend — sensitive to faults like imbalance, vibration spikes.

### Frequency-domain features
- FFT magnitude spectrum (up to `max_fft_bins`)  
- Spectral Centroid (frequency center)  
- Spectral Flatness (tonal vs. noisy content)

**Why:**  
Many mechanical faults manifest as frequency signatures (e.g., bearing tones).  
FFT adds “what frequency energy shifted” context missing in time domain.

**Alternatives:**
| Approach | Pros | Cons |
| Short-Time FFT | Fast | Fixed resolution |
| Wavelet | Good for non-stationary data | More complex |
| Autoregressive spectra | Compact | Sensitive to noise |

---

## 6. Feature Scaling & Normalization

`RobustScaler` (uses median & IQR).  

**Why:** Outlier-resistant; ensures each feature contributes equally to KMeans and PCA.  
**Alternative:** `StandardScaler` (Z-score) — faster but outlier-sensitive.

---

## 7. Regime Detection (K-Means Clustering)

- Try k = `k_min..k_max`, select best via Silhouette Score \(s=\frac{b−a}{\max(a,b)}\).  
- Stores cluster centroids → defines operating modes.

**Why KMeans:**  
Industrial signals often form compact clusters (steady-states). Fast, explainable.  
**Alternatives:** GMM (probabilistic soft assignments), DBSCAN (no k needed but needs eps).

---

## 8. H1 — Forecast Lite + AR(1)

### Concept
Models each tag as first-order auto-regressive:
\[
x_t = φx_{t−1} + ε_t
\]
Residual \(r_t = x_t − φx_{t−1}\) should be white noise if normal. Deviations → anomalies.

### Steps
1. Estimate φ per tag (correlation of lag 1).  
2. Compute rolling median baseline (`h1_roll`).  
3. Z-score residuals using MAD → aggregate across tags.

**Why:** Captures short-term predictability; detects sudden departures without complex forecast models.

**Alternatives:** ARIMA, Kalman filter, LSTM forecasting (heavier, needs more data).

---

## 9. H2 — PCA Reconstruction Error

### Theory
Principal Component Analysis projects features onto orthogonal axes maximizing variance.  
Reconstruction error:
\[
e_i = \|x_i − \hat{x}_i\|^2
\]
where \(\hat{x}_i = PCA^{-1}(PCA(x_i))\).

**Why:** Captures multivariate structure — changes in relationships between tags trigger higher errors.  
Ideal for detecting correlated faults (spanning multiple tags).

**Alternatives:** Autoencoder NN (similar concept, nonlinear), Robust PCA (outlier-tolerant).

---

## 10. H3 — Embedding Drift (Contrast Score)

### Concept
In PCA space, take cosine similarity to rolling mean:
\[
s_i = 1 − \frac{x_i·μ_{i−50:i}}{\|x_i\|\|μ_{i−50:i}\|}
\]
Higher \(s_i\) → state diverged from recent context.

**Why:** Detects slow, systematic drifts (e.g., aging, control offsets) that don’t spike residuals.

---

## 11. Context Mask & Corroboration

### Transient Mask
Detect high median |Δ| and |Δ²| across tags → mark transients.  
Reduces false alarms during start-ups and load changes.

### Corroboration Boost
Top N highly correlated tag pairs; when both deviate → boost confidence.

**Why:** Correlated tags (e.g., flow & pressure) should fail together; isolated deviation may be sensor noise.

---

## 12. Change-Point Signal (CPD)

Rolling mean & std (60 pts), sum of their absolute derivatives → detect step changes.  
**Why:** Provides simple, low-latency change-point proxy without offline algorithms (PELT, BOCPD).

---

## 13. Fusion and Episode Detection

Fused score:
\[
F = \min\!\big(1,\;0.45 H1 + 0.35 H2 + 0.35 H3 + 0.15 Corr + 0.10 CPD\big)
\]
then \(F × 0.7\) where masked.

Episodes = continuous segments where \(F ≥ τ\) (`fused_tau`), merged if gap < `merge_gap`.

**Why:** Combines fast, multivariate, and drift detectors into one interpretable timeline.

---

## 14. Drift Check

Compare current means vs. stored baselines:  
\[
Z = \frac{|μ_{current} − μ_{train}|}{σ_{train}}
\]
Sort descending → tags with largest drift.

**Why:** Quantifies slow shifts beyond normal variance.  
Critical for sensor bias or process creep.

---

## 15. Command-Line Usage

```powershell
python acm_core_local_2.py train --csv "Train.csv"
python acm_core_local_2.py score --csv "Test.csv"
python acm_core_local_2.py drift --csv "Recent.csv"
````

Artifacts and logs are written to `acm_artifacts\`.

---

## 16. Performance and Tuning Notes

| Parameter      | Effect              | Recommendation               |
| -------------- | ------------------- | ---------------------------- |
| `window`       | Feature granularity | Smaller = faster but noisier |
| `stride`       | Overlap density     | Higher = more samples        |
| `max_fft_bins` | Spectral resolution | 32–64 adequate               |
| `h1_mode`      | Computation load    | “lite_ar1” balanced          |
| `fused_tau`    | Sensitivity         | Lower → more events          |
| `k_min/k_max`  | Clustering scope    | Narrow for speed             |

---

## 17. Why These Algorithms Fit ACM

| Task                      | Algorithm      | Why Chosen                | Key Alternatives   |
| ------------------------- | -------------- | ------------------------- | ------------------ |
| Operating mode clustering | KMeans         | Fast, interpretable       | GMM, HDBSCAN       |
| Multivariate compression  | PCA            | Linear, deterministic     | Autoencoder        |
| Short-term forecast       | AR(1)          | Minimal data, instant fit | ARIMA, LSTM        |
| Anomaly fusion            | Weighted sum   | Transparent logic         | Ensemble learners  |
| Drift                     | Mean/σ Z-shift | Explainable               | KL-divergence, MMD |

---

## 18. Extensibility

* Plug other feature functions in `build_feature_matrix`.
* Swap `PCA` with `KernelPCA` or `AE`.
* Connect to SQL views or Historian for real-time windows.
* Wrap the pipeline inside your MES/n8n workflow.

---

## 19. Developer Checklist

* ✅ Code organized into pure functions.
* ✅ Each stage timed & logged (JSONL).
* ✅ Artifacts portable between runs.
* ✅ Safe for parallel equipment execution.
* ✅ No external dependencies beyond pandas/numpy/sklearn.

---
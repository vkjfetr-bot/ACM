# ACM Machine Learning Theory & Practice Guide

This guide explains the machine-learning logic that powers **ACM Core Local 2**.  
It is intended both for developers maintaining the code and for engineers learning how data science concepts apply to equipment health monitoring.

---

## 1. Philosophy of ACM

An industrial asset (motor, pump, fan, furnace) continuously produces numerical signals — pressures, flows, temperatures, currents.  
The goal of ACM is to **learn normal behavior from data** and **quantify deviations** that indicate faults, degradation, or abnormal operation.

Core principles:

1. **Context matters** — behavior changes across operating modes.  
2. **Anomalies are multivariate** — one tag may look normal alone.  
3. **Models must be fast, explainable, and re-trainable** on-prem.  
4. **Outputs must be human-interpretable** — tables, charts, events.

---

## 2. Time-Series Foundations

### 2.1  Sampling and Stationarity
Most ML assumes equally spaced data.  Therefore:
- We **resample** signals to a fixed rate (e.g., 1 minute).  
- Missing points are interpolated.  
This creates a *stationary frame* where statistics and spectra are comparable over time.

**Mathematically**
\[
x_r(t) = \text{Interp}\big(x(\tau)\big),\;\tau\in[t-\Delta,t+\Delta]
\]

**Industrial meaning:** converting raw historian logs with uneven timestamps into a consistent timeline.

---

### 2.2  Tags and Variability
Tags with almost constant values contain no information.  
Variance is a rough proxy for information content.  
Hence the rule: keep only numeric tags with ≥ 20 unique values.

---

## 3. Windowing — Making Time Local

To analyse dynamic systems, we slide a *window* of length W across time with step S:

```

|------W------|
|------W------|

```

Each window becomes one sample in feature space.

- **W (256 points)**: captures short-term behavior.  
- **S (64 points)**: controls overlap and temporal resolution.

This allows algorithms designed for static data (KMeans, PCA) to operate on streaming signals.

---

## 4. Feature Engineering

### 4.1  Why Features
Raw time-series are high-dimensional and correlated.  
Features condense each window into a vector summarising shape, energy, and frequency.

### 4.2  Time-Domain Features
For each tag:
| Feature | Meaning | Formula / Intuition |
| -------- | -------- | ------------------ |
| Mean (μ) | Level | average operating point |
| RMS | Energy | \(\sqrt{E[x^2]}\) |
| Variance | Stability | \(E[(x-μ)^2]\) |
| Skewness | Asymmetry | positive → tail to right |
| Kurtosis | Peakedness | >3 → spiky |
| Crest factor | Shock indicator | peak/RMS |
| Slope | Local trend | regression line in last half window |

**Why:** these cover both *steady-state* and *transient* characteristics.

### 4.3  Frequency-Domain Features
Compute FFT magnitude spectrum up to `max_fft_bins`.

Add:
- **Spectral centroid** — weighted average frequency  
- **Spectral flatness** — noise vs tonal measure

**Why:** many mechanical faults (imbalance, bearing defects) cause specific frequency tones; power and acoustic signatures reveal them.

### 4.4  Scaling
All features are normalized using **RobustScaler** (median + IQR).  
This avoids distortion from spikes or unit differences.

---

## 5. Discovering Operating Regimes

### 5.1  Concept
Different machine states (Idle, Loaded, Cleaning) form clusters in feature space.  
We detect them using **K-Means** clustering.

### 5.2  Algorithm
1. Try K = 2 … 6.  
2. Compute *silhouette score* for each K.  
3. Choose the K with highest mean silhouette.

\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

- \(a(i)\): intra-cluster distance  
- \(b(i)\): nearest-cluster distance

### 5.3  Why KMeans
- Fast (O(n k d))  
- Deterministic with fixed seed  
- Produces crisp “mode labels” for regime diagnostics.

**Alternatives:**  
Gaussian Mixture Models (GMM) – probabilistic, slower;  
DBSCAN – automatic K but requires ε.

---

## 6. H1 — Forecast-Lite + AR(1)

### 6.1  Forecast intuition
If a variable can be predicted from its past, residual errors reflect abnormality.

**AR(1) model**
\[
x_t = φx_{t-1} + ε_t
\]
\(φ\) ≈ correlation at lag 1.

Residual: \(r_t = x_t − φx_{t−1}\)

### 6.2  Two Components
1. **Rolling baseline residual** — compares current value to median of last N samples.  
2. **AR(1) residual** — deviation from one-step forecast.

Both are z-scored (using Median Absolute Deviation) and averaged across tags.

### 6.3  Why AR(1)
- Captures short-term correlation common in process signals.  
- Extremely fast (no model fitting loop).  
- Works on short histories.  
- Avoids overfitting compared to ARIMA or LSTM.

### 6.4  Industrial interpretation
Detects sudden step, spike, or oscillation that breaks usual temporal correlation.

---

## 7. H2 — PCA Reconstruction

### 7.1  Idea
PCA finds orthogonal axes capturing maximal variance.  
Low-variance directions correspond to noise or rare behavior.

Reconstruction error:
\[
e_i = \|x_i - \hat{x}_i\|^2
\]
High \(e_i\) ⇒ sample cannot be well represented by normal subspace.

### 7.2  Why PCA
- Learns correlation structure between tags.  
- Highlights when relationships (e.g., pressure ∝ flow) change.  
- Linear, explainable, easily serialized (`joblib`).

**Alternative:** Autoencoder — nonlinear, but heavier and less transparent.

---

## 8. H3 — Embedding Drift Detection

### 8.1  Motivation
Not all degradation is sudden. Some drifts slowly alter the *direction* of system behavior in feature space.

### 8.2  Method
Transform data into PCA space \(z_i = PCA(x_i)\).  
Compute cosine similarity with rolling mean of previous 50 embeddings:

\[
d_i = 1 - \frac{z_i·μ_{i-50:i}}{\|z_i\|\|μ_{i-50:i}\|}
\]

Higher \(d_i\) = greater drift.

### 8.3  Why cosine drift
- Scale-invariant  
- Detects shape/orientation change rather than magnitude  
- Suitable for slow equipment aging or recalibration effects.

---

## 9. Context Masking and Corroboration

### 9.1  Transient Detection
Transient = rapid global change (start/stop).  
Compute median slope and acceleration z-scores; mark high values.

Purpose: suppress false anomalies during expected transitions.

### 9.2  Corroboration Boost
Compute correlation matrix R.  
Select top-N correlated tag pairs.  
If both tags spike simultaneously → increase confidence.

Reasoning:  
Multi-tag concurrence is more reliable than isolated sensor noise.

---

## 10. Change-Point Signal (CPD proxy)

For each tag:
1. Rolling mean μ and std σ over 60 points.  
2. Compute |Δμ| + |Δσ| across tags, median combine.  
3. Normalize to [0,1].

Represents structural change in process mean/variance.

**Why simple:**  
Real-time, no heavy segmentation like Bayesian Online CPD.  
Good enough for trend shifts.

---

## 11. Fusion of Evidence

We combine all signals into a unified **Fused Score F(t)**:

\[
F(t)=\min\!\big[1,\;
0.45H1 + 0.35H2 + 0.35H3 + 0.15Corr + 0.10CPD\big]
\]
and multiply by 0.7 where transient mask = 1.

### Rationale
- H1 → fast spikes  
- H2 → cross-tag structure  
- H3 → slow drift  
- Corr + CPD → confidence & change  
Weighted fusion balances sensitivity and stability.

---

## 12. Episode Formation

Threshold \(τ=0.7\).  
Whenever \(F(t)≥τ\), start an episode; end when below τ.  
Merge events separated by < `merge_gap` minutes.

Outputs: `Start`, `End`, `PeakScore`.

This converts noisy anomaly series into clean events for reporting.

---

## 13. Drift Monitoring

### 13.1  Baseline
During training, compute for each tag:
\[
μ_{train},\,σ_{train}
\]

### 13.2  Runtime
For new data:
\[
Z = \frac{|μ_{new}-μ_{train}|}{σ_{train}}
\]

Tags with Z > 3 → significant drift.

### 13.3  Why
Detects slow bias (sensor drift, fouling, wear) beyond transient anomalies.

---

## 14. Putting It All Together

### Training Phase
1. Clean → Resample → Tag Select  
2. Feature Build → Scale → Cluster (KMeans)  
3. Fit PCA → Save models + baselines  

### Scoring Phase
1. Load models → Build features from new data  
2. Compute H1–H3, Corr, CPD  
3. Fuse → Mask → Episode → Export  

### Drift Phase
Compare new vs. baseline statistics.

All stages log timing into `run_*.jsonl` for performance tracking.

---

## 15. Interpreting Outputs

| File | What it Tells You |
| ---- | ----------------- |
| `*_scored_window.csv` | Per-window scores (plot as trend) |
| `*_events.csv` | Start/end/peak of detected events |
| `*_drift.csv` | Tags with long-term shifts |
| `*_train_diagnostics.csv` | Feature set + regime labels |
| `*_pca.joblib` | Basis of normal correlation |
| `*_regimes.joblib` | Discovered modes (clusters) |

---

## 16. Design Philosophy — Why These Choices

| Problem | Selected Technique | Reason |
| -------- | ------------------ | ------- |
| Unsupervised detection | KMeans + PCA | No labels required, fast, interpretable |
| High-dimensional features | PCA | Linear compression, explainable variance |
| Short-term prediction | AR(1) | Minimal data need, quick retrain |
| Noise & missing data | Robust Scaler, resample | Outlier-tolerant |
| Model transparency | Simple math | Easier debugging vs deep nets |
| Industrial deployability | CSV-based, no cloud deps | On-prem edge friendly |

---

## 17. Extensions for Advanced Users

| Enhancement | Description |
| ------------ | ----------- |
| **Isolation Forest / One-Class SVM** | Replace fusion with learned anomaly boundary |
| **Wavelet features** | Multi-scale temporal frequency |
| **Autoencoder drift** | Non-linear reconstruction error |
| **RLS/Kalman** | Online parameter update for H1 |
| **Explainability** | SHAP values on PCA/IF scores |
| **Real-time streaming** | Implement incremental windowing with Kafka or n8n |

---

## 18. Practical Tuning Tips

| Symptom | Likely Cause | Adjustment |
| -------- | ------------- | ----------- |
| Too many events | Lower H1 weight / raise τ |
| Missed anomalies | Increase H1 weight or reduce τ |
| High drift count | Sensor bias / bad baseline |
| Cluster flip-flop | Reduce k_max or smooth features |
| Long runtime | Lower FFT bins or stride up |

---

## 19. Conceptual Summary

The ACM pipeline is a **layered unsupervised learning system**:

1. **Statistical** → mean, variance, skew.  
2. **Spectral** → FFT energy distribution.  
3. **Structural** → PCA correlation.  
4. **Temporal** → AR(1) residuals & drift.  
5. **Contextual** → transient mask & corroboration.  
6. **Fusion** → unified health indicator.  
7. **Eventization** → actionable episodes.  
8. **Drift** → long-term trend health.

Together these cover the full spectrum from noise spikes to slow degradation.

---

### Final Thought
The strength of ACM Core 2 lies not in a single model but in the **ensemble of simple, explainable mechanisms** — each grounded in classical statistics — that together form a robust, real-time view of asset health.

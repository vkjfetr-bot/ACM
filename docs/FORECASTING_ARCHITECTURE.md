# ACM Forecasting Engine Architecture (v10.0.0)

**Last Updated**: December 8, 2025  
**Author**: ACM Development Team  
**Purpose**: Comprehensive explanation of forecasting system, detector integration, and predictive maintenance workflow

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Detector Heads (7 Algorithms)](#detector-heads)
4. [Fusion & Health Calculation](#fusion--health-calculation)
5. [Regime Identification](#regime-identification)
6. [Forecasting Engine](#forecasting-engine)
7. [RUL Estimation](#rul-estimation)
8. [Integration Flow](#integration-flow)
9. [SQL Schema Reference](#sql-schema-reference)

---

## System Overview

The **ACM (Advanced Condition Monitoring)** system is a **real-time predictive maintenance platform** that:

1. **Detects anomalies** across multiple dimensions (time-series, density, correlation, outliers)
2. **Fuses detector signals** into unified health scores
3. **Identifies operating regimes** to provide operational context
4. **Forecasts health trajectories** using exponential smoothing
5. **Predicts failure probability** via Gaussian tail modeling
6. **Estimates RUL (Remaining Useful Life)** with Monte Carlo simulation
7. **Forecasts sensor values** using multivariate VAR models

**Key Innovation**: Multi-algorithm fusion with context-aware thresholds and predictive forecasting.

---

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAW SENSOR DATA                               │
│    9 sensors @ 30-min cadence: Temp, Pressure, Flow, Current, etc.   │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                               │
│  • Rolling statistics (mean, std, skew, kurtosis) @ 16-point window  │
│  • FFT energy bands (0-5 harmonics)                                  │
│  • Lag features (t-1, t-2), derivatives (Δt)                         │
│  → Result: 9 sensors × 10 features = ~90 engineered features         │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    7 DETECTOR HEADS (Parallel)                       │
├──────────────────────────────────────────────────────────────────────┤
│ **1. AR1 (Time-Series Anomaly)**                                     │
│    - Algorithm: Autoregressive AR(1) residual analysis               │
│    - Detects: Sudden jumps, trend breaks, autocorrelation shifts     │
│    - Output: ar1_z (Z-score of residuals)                            │
│                                                                      │
│ **2. PCA-SPE (Correlation Break)**                                   │
│    - Algorithm: Principal Component Analysis - Squared Prediction    │
│    - Detects: Changes in sensor correlation patterns                 │
│    - Output: pca_spe_z (reconstruction error Z-score)                │
│                                                                      │
│ **3. PCA-T² (Multivariate Outlier)**                                 │
│    - Algorithm: Hotelling's T² statistic on PC space                 │
│    - Detects: Extreme operating points                               │
│    - Output: pca_t2_z (Mahalanobis distance Z-score)                 │
│                                                                      │
│ **4. MHAL (Density Anomaly - Mahalanobis)**                          │
│    - Algorithm: Regularized covariance-based distance                │
│    - Detects: Deviation from multivariate normal baseline            │
│    - Output: mhal_z (Mahalanobis Z-score)                            │
│                                                                      │
│ **5. IForest (Isolation Anomaly)**                                   │
│    - Algorithm: Isolation Forest (random partitioning)               │
│    - Detects: Rare/isolated operating conditions                     │
│    - Output: iforest_z (isolation score Z-score)                     │
│                                                                      │
│ **6. GMM (Density Anomaly - Gaussian Mixture)**                      │
│    - Algorithm: Gaussian Mixture Model (k=2-5 components)            │
│    - Detects: Multi-modal distribution deviations                    │
│    - Output: gmm_z (log-likelihood Z-score)                          │
│                                                                      │
│ **7. OMR (Baseline Consistency - Overall Model Residual)**           │
│    - Algorithm: PLS regression + residual analysis                   │
│    - Detects: Overall drift from baseline operating envelope         │
│    - Output: omr_z (reconstruction Z-score)                          │
│    - Bonus: Provides sensor contribution scores (root cause)         │
├──────────────────────────────────────────────────────────────────────┤
│ **Why 7 Detectors?**                                                 │
│  - Different anomaly types require different algorithms              │
│  - AR1 catches time-series breaks; PCA catches correlation shifts    │
│  - Fusion reduces false positives via majority voting                │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FUSION ENGINE (core/fuse.py)                       │
├──────────────────────────────────────────────────────────────────────┤
│ **Weighted Fusion**:                                                 │
│   FusedZ = Σ(weight_i × detector_i_z)                                │
│                                                                      │
│ **Auto-Tuning Weights** (episode_separability metric):               │
│   - Detectors with better episode discrimination get higher weight   │
│   - Weights renormalized to sum = 1.0                                │
│   - Example: ar1_z=0.20, gmm_z=0.15, omr_z=0.12, etc.                │
│                                                                      │
│ **CUSUM Drift Detection**:                                           │
│   - Tracks cumulative sum of (FusedZ - k_sigma)                      │
│   - Triggers alert when CUSUM > h_sigma                              │
│   - Auto-tunes k (drift threshold) and h (alarm threshold)           │
│                                                                      │
│ **Alert Zones** (threshold mapping):                                 │
│   - FusedZ < 1.0  → GOOD     (green)                                 │
│   - FusedZ 1-2    → WATCH    (yellow)                                │
│   - FusedZ 2-3    → ALERT    (orange)                                │
│   - FusedZ > 3.0  → CRITICAL (red)                                   │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│              HEALTH INDEX CALCULATION                                │
│                                                                      │
│   Health = 100 - (FusedZ / 4.0) * 100                                │
│                                                                      │
│   Examples:                                                          │
│   • FusedZ = -2.0 → Health = 150 (capped at 100) → GOOD              │
│   • FusedZ =  0.0 → Health = 100              → GOOD                 │
│   • FusedZ =  2.0 → Health = 50               → WATCH                │
│   • FusedZ =  3.0 → Health = 25               → ALERT                │
│   • FusedZ =  4.0 → Health = 0                → CRITICAL             │
│                                                                      │
│   Health Zones:                                                      │
│   • 80-100: GOOD     (Normal operation)                              │
│   • 60-80:  WATCH    (Early warning)                                 │
│   • 40-60:  ALERT    (Degradation detected)                          │
│   • 0-40:   CRITICAL (Failure imminent)                              │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│              REGIME IDENTIFICATION (core/regimes.py)                 │
├──────────────────────────────────────────────────────────────────────┤
│ **Purpose**: Identify distinct operating states                      │
│                                                                      │
│ **Algorithm**: K-Means clustering on detector Z-scores               │
│   - Features: [ar1_z, pca_spe_z, pca_t2_z, mhal_z, iforest_z, ...]   │
│   - Auto-selects K (2-6) using silhouette score                      │
│   - Example regimes: "Normal", "High Load", "Startup", "Transient"   │
│                                                                      │
│ **Why Regimes?**                                                     │
│   - "Normal" looks different during startup vs steady-state          │
│   - Per-regime baselines reduce false positives                      │
│   - Enables context-aware alerting                                   │
│                                                                      │
│ **Outputs**:                                                         │
│   - regime_label: Integer label (0, 1, 2, ...)                       │
│   - regime_state: Human-readable ("Normal", "High Load", etc.)       │
│   - ACM_RegimeTimeline: Time-series of regime transitions            │
│   - ACM_RegimeTransitions: Markov chain transition matrix            │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│          EPISODE DETECTION & DIAGNOSTICS                             │
├──────────────────────────────────────────────────────────────────────┤
│ **Definition**: Episode = continuous period where FusedZ > threshold │
│                                                                      │
│ **Detection Logic**:                                                 │
│   - Start: FusedZ crosses above alert threshold (default 2.0)        │
│   - End: FusedZ drops below threshold for 3+ consecutive points      │
│   - Merge: Episodes separated by <30min are merged                   │
│                                                                      │
│ **Culprit Identification** (via OMR contributions):                  │
│   - OMR provides per-sensor contribution to anomaly score            │
│   - Top 3 sensors with highest contributions = culprits              │
│   - Example: "Multivariate Outlier (PCA-T2) → DEMO.SIM.06GP34"       │
│                                                                      │
│ **Severity Scoring**:                                                │
│   - INFO:     peak_z < 2.5 AND duration < 2h                         │
│   - MEDIUM:   peak_z < 4.0 OR duration < 12h                         │
│   - CRITICAL: peak_z ≥ 4.0 OR duration ≥ 12h                         │
│                                                                      │
│ **Outputs**:                                                         │
│   - ACM_EpisodeDiagnostics: Per-episode metrics                      │
│   - ACM_EpisodeCulprits: Root cause sensors with scores              │
│   - ACM_EpisodeMetrics: Aggregate statistics (count, duration, etc.) │
└─────────────────────────┬────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│            FORECASTING ENGINE (core/forecasting.py)                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ╔══════════════════════════════════════════════════════════════════╗ │
│ ║  1. HEALTH FORECAST (Exponential Smoothing - Holt's Linear)      ║   
│ ╚══════════════════════════════════════════════════════════════════╝ │
│                                                                      │
│ **Algorithm**: Holt's Linear Trend (Double Exponential Smoothing)    │
│                                                                      │
│   Level[t] = α * Health[t] + (1-α) * (Level[t-1] + Trend[t-1])       │
│   Trend[t] = β * (Level[t] - Level[t-1]) + (1-β) * Trend[t-1]        │
│   Forecast[t+h] = Level[t] + h * Trend[t]                            │
│                                                                      │
│ **Parameters**:                                                      │
│   - α (level smoothing): 0.1-0.3 (adaptive via grid search)          │
│   - β (trend smoothing): 0.01-0.1 (adaptive)                         │
│   - Forecast horizon: 168 hours (7 days)                             │
│   - Data cadence: 30 minutes (dt_hours = 0.5)                        │
│                                                                      │
│ **Confidence Intervals**:                                            │
│   - Method 1: Analytic (Hyndman & Athanasopoulos formula)            │
│     CI_width[h] = 1.96 × σ_error × √variance_multiplier[h]           │
│     variance_mult[h] = 1 + (h-1) * [α² + αβh + β²h(h+1)/2]           │
│                                                                      │
│   - Method 2: Bootstrap (default, 500 replicates)                    │
│     - Generates 500 noisy trajectories                               │
│     - Reports P2.5 and P97.5 percentiles as CI bounds                │
│     - More robust to non-Gaussian errors                             │
│                                                                      │
│ **Adaptive Smoothing** (FORECAST-P2-3.1):                            │
│   - Grid search over α ∈ [0.1, 0.3], β ∈ [0.01, 0.1]                 │
│   - Minimizes one-step-ahead forecast RMSE                           │
│   - Runs every retrain (when drift/anomaly energy exceeds limit)     │
│                                                                      │
│ **Model Diagnostics** (FORECAST-P3-4.4):                             │
│   - Normality test: Shapiro-Wilk on residuals                        │
│   - Autocorrelation: Ljung-Box test                                  │
│   - Theil's U: Forecast accuracy vs naive baseline                   │
│   - MAPE: Mean Absolute Percentage Error                             │
│   - Variance ratio: σ²(residuals) / σ²(actuals)                      │
│                                                                      │
│ **Continuous State** (FORECAST-STATE-02):                            │
│   - Loads previous [level, trend, α, β] from ForecastState           │
│   - Uses 72-hour sliding window instead of single batch              │
│   - Temporal continuity prevents discontinuous jumps                 │
│   - State persisted to ACM_ForecastState after each run              │
│                                                                      │
│ **Retrain Triggers**:                                                │
│   1. Anomaly energy spike: max(FusedZ) > 1.5 × avg(FusedZ)           │
│   2. Drift detected: CUSUM exceeds threshold                         │
│   3. Regime shift: K-Means labels changed significantly              │
│   4. Scheduled: Every 7 days (configurable)                          │
│                                                                      │
│ **Output Table**: `ACM_HealthForecast`                               │
│   Columns: EquipID, RunID, Timestamp, ForecastHealth, CiLower,       │
│            CiUpper, ForecastStd, Method, CreatedAt                   │
│                                                                      │
│ ╔══════════════════════════════════════════════════════════════════╗ │
│ ║  2. FAILURE PROBABILITY (Gaussian Tail Model)                    ║ │
│ ╚══════════════════════════════════════════════════════════════════╝ │
│                                                                      │
│ **Algorithm**: Gaussian tail probability on health trajectory        │
│                                                                      │
│   P(Failure at t+h) = P(Health[t+h] < threshold)                     │
│                     = Φ((threshold - μ[t+h]) / σ[t+h])               │
│                                                                      │
│   where:                                                             │
│   - Φ = Standard normal CDF (cumulative distribution function)       │
│   - μ[t+h] = ForecastHealth[t+h]                                     │
│   - σ[t+h] = ForecastStd[t+h] (from bootstrap or analytic CI)        │
│   - threshold = 70.0 (configurable failure threshold)                │
│                                                                      │
│ **Hazard Rate Smoothing** (exponential blend, τ=12h):                │
│   - Prevents discontinuous probability jumps between runs            │
│   - Blends previous hazard with new hazard using time constant       │
│   - Ensures smooth probability curves in Grafana                     │
│                                                                      │
│ **Output Tables**:                                                   │
│   - `ACM_FailureForecast`: Time-series of failure probability        │
│     Columns: EquipID, RunID, Timestamp, FailureProb, ThresholdUsed,  │
│              Method, CreatedAt                                       │
│                                                                      │
│ ╔══════════════════════════════════════════════════════════════════╗ │
│ ║  3. RUL ESTIMATION (Monte Carlo Simulation)                      ║ │
│ ╚══════════════════════════════════════════════════════════════════╝ │
│                                                                      │
│ **Algorithm**: Stochastic trajectory simulation                      │
│                                                                      │
│ **Steps**:                                                           │
│   1. Generate 1000 noisy health trajectories:                        │
│      trajectory[i][h] = Forecast[h] + N(0, ForecastStd[h])           │
│                                                                      │
│   2. For each trajectory, find first crossing:                       │
│      RUL[i] = min{h : trajectory[i][h] < threshold}                  │
│                                                                      │
│   3. Compute percentiles:                                            │
│      P10_LowerBound = 10th percentile of RUL[1..1000]                │
│      P50_Median     = 50th percentile (median)                       │
│      P90_UpperBound = 90th percentile                                │
│                                                                      │
│ **Confidence Score**:                                                │
│   - Based on CI width and trajectory consistency                     │
│   - Confidence = 1 - (P90 - P10) / (2 × P50)                         │
│   - Range: [0, 1] where 1 = high confidence, 0 = high uncertainty    │
│                                                                      │
│ **Culprit Sensors** (via OMR contributions):                         │
│   - OMR provides sensor contribution scores at forecast time         │
│   - Top 3 sensors with highest scores → TopSensor1/2/3               │
│   - Helps prioritize maintenance actions                             │
│                                                                      │
│ **Output Table**: `ACM_RUL`                                          │
│   Columns: EquipID, RunID, RUL_Hours, P10_LowerBound, P50_Median,    │
│            P90_UpperBound, Confidence, Method, FailureTime,          │
│            TopSensor1, TopSensor2, TopSensor3, NumSimulations,       │
│            CreatedAt                                                 │
│                                                                      │
│ ╔══════════════════════════════════════════════════════════════════╗ │
│ ║  4. DETECTOR FORECAST (Linear Extrapolation)                     ║ │
│ ╚══════════════════════════════════════════════════════════════════╝ │
│                                                                      │
│ **Purpose**: Project individual detector Z-scores forward            │
│                                                                      │
│ **Algorithm**: Simple linear trend extrapolation                     │
│   - Computes slope via least-squares on last 10 points               │
│   - Projects forward 168 hours                                       │
│   - Only forecasts top 3 most active detectors (highest |Z|)         │
│                                                                      │
│ **Output Table**: `ACM_DetectorForecast_TS`                          │
│   Columns: EquipID, RunID, Timestamp, DetectorType, ForecastZScore,  │
│            Method, CreatedAt                                         │
│                                                                      │
│ ╔══════════════════════════════════════════════════════════════════╗ │
│ ║  5. SENSOR FORECAST (VAR - Vector Autoregression) **NEW v10!**  ║  │
│ ╚══════════════════════════════════════════════════════════════════╝ │
│                                                                      │
│ **Purpose**: Multivariate forecasting of physical sensor values      │
│                                                                      │
│ **Algorithm**: VAR(p) - Vector Autoregression                        │
│                                                                      │
│   X[t] = Φ₁·X[t-1] + Φ₂·X[t-2] + ... + Φₚ·X[t-p] + ε[t]              │
│                                                                      │
│   where:                                                             │
│   - X[t] = [sensor1[t], sensor2[t], ..., sensor9[t]]ᵀ                │
│   - Φₚ = Lag coefficient matrices (9×9 each)                          │
│   - p = Lag order (selected via AIC, typically 2-6)                  │
│   - ε[t] = White noise vector                                        │
│                                                                      │
│ **Key Feature**: Captures sensor interdependencies                   │
│   - Example: Temp ↑ → Flow ↓ (physical coupling)                     │
│   - More accurate than independent univariate forecasts              │
│                                                                      │
│ **Implementation** (statsmodels.tsa.vector_ar.var_model.VAR):        │
│   1. Select top 10 sensors by variance (most dynamic)                │
│   2. Remove NaNs, require ≥50 data points                            │
│   3. Fit VAR model with AIC lag selection (maxlags=10)               │
│   4. Generate 168-hour forecast                                      │
│                                                                      │
│ **Output Table**: `ACM_SensorForecast`                               │
│   Columns: EquipID, RunID, Timestamp, SensorName, ForecastValue,     │
│            CiLower, CiUpper, ForecastStd, Method, RegimeLabel,       │
│            CreatedAt                                                 │
│                                                                      │
│   Method values:                                                     │
│   - "VAR(p)" - Vector Autoregression with p lags (multivariate)      │
│   - "LinearTrend" - Fallback univariate linear extrapolation         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Integration Flow: How Everything Connects

### **Phase 1: Real-Time Detection**
```
Sensors → Features → Detectors → Fusion → Health
                                      ↓
                                  Regimes
                                      ↓
                                  Episodes
```

### **Phase 2: Forecasting & Prediction**
```
Health Timeline → Exponential Smoothing → Health Forecast
                                              ↓
                                     Gaussian Tail Model
                                              ↓
                                      Failure Probability
                                              ↓
                                    Monte Carlo Simulation
                                              ↓
                                          RUL (P10/P50/P90)

Sensor Data → VAR Model → Sensor Forecast
```

### **Phase 3: Root Cause Analysis**
```
OMR Contributions → Episode Culprits → TopSensor1/2/3
                                              ↓
                                     Maintenance Actions
```

### **Complete Pipeline Execution**
```python
# Simplified pseudo-code
def run_acm_pipeline(sensors, config):
    # 1. Feature engineering
    features = build_features(sensors, window=16)
    
    # 2. Detector scoring
    ar1_z = ar1_detector.score(features)
    pca_spe_z = pca_detector.score(features)
    # ... (5 more detectors)
    
    # 3. Fusion
    fused_z = fusion_engine.fuse([ar1_z, pca_spe_z, ...], weights)
    health = 100 - (fused_z / 4.0) * 100
    
    # 4. Regime identification
    regime_label = kmeans.predict(features)
    
    # 5. Episode detection
    episodes = detect_episodes(fused_z, threshold=2.0)
    culprits = identify_culprits(episodes, omr_contributions)
    
    # 6. Forecasting
    forecast = exponential_smoothing(health, alpha, beta, horizon=168)
    failure_prob = gaussian_tail_model(forecast, threshold=70)
    rul = monte_carlo_simulation(forecast, threshold=70, n_sims=1000)
    sensor_forecast = var_model(sensors, horizon=168)
    
    # 7. Write to SQL
    write_to_sql(health, fused_z, regime_label, episodes, 
                 forecast, failure_prob, rul, sensor_forecast)
```

---

## SQL Schema Reference

### **Core Tables**

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `ACM_HealthTimeline` | Health time-series | Timestamp, HealthIndex, HealthZone, FusedZ |
| `ACM_Scores_Wide` | Detector Z-scores | ar1_z, pca_spe_z, ..., fused |
| `ACM_RegimeTimeline` | Regime labels | RegimeLabel, RegimeState |
| `ACM_EpisodeDiagnostics` | Anomaly periods | episode_id, duration_h, severity, peak_z |

### **Forecasting Tables (v10.0.0)**

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `ACM_HealthForecast` | 7-day health projection | ForecastHealth, CiLower, CiUpper, Method |
| `ACM_FailureForecast` | Failure probability curve | FailureProb, ThresholdUsed, Method |
| `ACM_RUL` | Remaining useful life | RUL_Hours, P10/P50/P90, Confidence, TopSensor1-3 |
| `ACM_DetectorForecast_TS` | Detector trajectories | DetectorType, ForecastZScore |
| `ACM_SensorForecast` | Physical sensor forecasts | SensorName, ForecastValue, Method (VAR/Linear) |

---

## Key Configuration Parameters

```python
# Forecasting (configs/config_table.csv)
forecasting.enabled = True
forecasting.forecast_hours = 168  # 7 days
forecasting.health_threshold = 70.0  # Failure threshold
forecasting.alpha = 0.2  # Level smoothing (adaptive)
forecasting.beta = 0.05  # Trend smoothing (adaptive)
forecasting.enable_bootstrap_ci = True
forecasting.bootstrap_n = 500
forecasting.enable_adaptive_smoothing = True

# Fusion weights (auto-tuned)
fusion.weights.ar1_z = 0.20
fusion.weights.pca_spe_z = 0.20
fusion.weights.omr_z = 0.10
# ...

# Episode detection
episodes.alert_threshold = 2.0  # FusedZ
episodes.min_duration_minutes = 30
episodes.merge_gap_minutes = 30
```

---

## Troubleshooting Guide

### **Issue**: Health forecast shows flat line (0.0 values)
**Root Cause**: ForecastEngine skeleton code with missing dependencies  
**Solution**: Use `forecasting.py` (working implementation) instead of `forecast_engine.py`

### **Issue**: Sensor forecasts show only LinearTrend method
**Root Cause**: VAR code path not executing due to missing `output_manager` parameter  
**Solution**: Store VAR forecasts in `tables` dict, let wrapper function persist

### **Issue**: RUL showing imminent failure (<24h) when Health > 80%
**Root Cause**: Query uses `ORDER BY RUL_Hours ASC` (worst-case historical)  
**Solution**: Use `ORDER BY CreatedAt DESC` (most recent prediction)

---

## Performance Characteristics

- **Typical runtime**: 8-12 seconds per equipment per 30-day window
- **Feature engineering**: 2.3s (23% of total)
- **GMM fitting**: 2.0s (20% of total)
- **Analytics generation**: 2.3s (23% of total)
- **Forecasting**: 1.0-1.5s (10-15% of total)

---

## Future Enhancements

1. **Deep Learning Forecasting**: LSTM/GRU for non-linear health trajectories
2. **Multi-Equipment Correlations**: Fleet-wide anomaly detection
3. **Prescriptive Maintenance**: Action recommendations beyond diagnosis
4. **Real-Time Streaming**: Sub-minute latency (currently 30-min batch)

---

**Document Version**: 1.0.0  
**Last Updated**: December 8, 2025

# RUL Methodology & Failure Condition Definition

**Version:** 2.0  
**Date:** 2025-11-20  
**Status:** Production

---

## Overview

ACM's Remaining Useful Life (RUL) estimation uses a **multi-path approach** that combines:
1. **Health trajectory analysis** - Extrapolation of HealthIndex trends
2. **Hazard accumulation** - Probabilistic failure modeling
3. **Anomaly energy thresholds** - Cumulative stress indicators

The final RUL is the **minimum** of all three paths, representing the most conservative (safest) estimate.

---

## Unified Failure Condition

A **failure event** is detected when **ANY** of the following conditions occur:

### 1. Sustained Low Health (Gradual Degradation)
```
HealthIndex < 75.0 for >= 4 consecutive hours
```
**Rationale:**
- Captures slow degradation trends
- 75% threshold provides early warning before critical failure
- 4-hour sustain period filters transient dips
- Typical for bearing wear, insulation degradation, efficiency loss

**Detection Logic:**
```python
health_failures = health_timeline[health_timeline['HealthIndex'] < 75.0]
sustained = health_failures.groupby((health_failures.index.to_series().diff() > pd.Timedelta(hours=2)).cumsum())
failure_events = sustained[sustained.size >= 4].first()
```

---

### 2. Critical Episode (Known Fault Pattern)
```
Episode with Severity = 'CRITICAL' logged in ACM_CulpritHistory
```
**Rationale:**
- Leverages domain knowledge from episode detection
- Critical episodes indicate known dangerous patterns
- Examples: Bearing fault signatures, stator winding hotspots, rotor imbalance
- Instant failure flag (no sustain period required)

**Detection Logic:**
```python
critical_episodes = df_episodes[
    (df_episodes['Severity'] == 'CRITICAL') &
    (df_episodes['EpisodeEndTime'].notna())
]
failure_times = critical_episodes['EpisodeEndTime']
```

---

### 3. Acute Anomaly Spike (Sudden Failure)
```
FusedZ >= 3.0 for >= 2 consecutive hours
```
**Rationale:**
- Detects sudden catastrophic events
- FusedZ combines all detector outputs (weighted ensemble)
- 3σ threshold = 99.7% confidence (rare event)
- 2-hour sustain filters sensor noise/transients
- Typical for thermal runaway, electrical faults, mechanical shock

**Detection Logic:**
```python
acute_anomalies = scores_timeline[scores_timeline['FusedZ'] >= 3.0]
sustained = acute_anomalies.groupby((acute_anomalies.index.to_series().diff() > pd.Timedelta(hours=1.5)).cumsum())
failure_events = sustained[sustained.size >= 2].first()
```

---

### Optional: Pre-Failure Warning (Early Indicator)
```
DriftValue > 1.5 AND anomaly_energy_slope > threshold
```
**Purpose:**
- Early warning before any failure condition triggers
- Indicates *trend toward failure* but not yet failed
- Used for maintenance scheduling, not RUL calculation
- Alerts operations team to monitor equipment closely

---

## Multi-Path RUL Derivation

### Path 1: Health Trajectory Crossing
**Method:** Linear extrapolation of recent HealthIndex trend

```
1. Fit linear model: HealthIndex(t) = β₀ + β₁·t
2. Solve for crossing: β₀ + β₁·t_fail = 75.0
3. RUL_trajectory = t_fail - t_now
```

**Advantages:**
- Simple, interpretable
- Works well for gradual degradation
- Confidence intervals from regression uncertainty

**Limitations:**
- Assumes linear decline (may miss non-linear patterns)
- Sensitive to recent noise

**Implementation:**
```python
from sklearn.linear_model import LinearRegression

X = (health_timeline.index - health_timeline.index[0]).total_seconds().values.reshape(-1, 1)
y = health_timeline['HealthIndex'].values

model = LinearRegression().fit(X, y)
slope = model.coef_[0]

if slope < 0:  # Declining health
    rul_hours = (75.0 - model.predict([[X[-1][0]]])[0]) / abs(slope) * (1/3600)
else:
    rul_hours = np.inf  # Improving health, no failure predicted
```

---

### Path 2: Hazard Accumulation
**Method:** Probabilistic failure via hazard rate modeling

```
1. Convert failure probability to hazard rate:
   λ(t) = -ln(1 - p(t)) / Δt

2. Apply EWMA smoothing:
   λ_smooth(t) = α·λ(t) + (1-α)·λ_smooth(t-1)

3. Compute survival probability:
   S(t) = exp(-∫₀ᵗ λ_smooth(τ) dτ)

4. Failure probability:
   F(t) = 1 - S(t)

5. RUL when F(t) >= 0.5 (50% chance of failure)
```

**Advantages:**
- Probabilistic interpretation
- Smooth curves via EWMA (no batch steps)
- Incorporates uncertainty naturally

**Limitations:**
- Requires accurate failure probability estimates
- Sensitive to α parameter (smoothing factor)

**Configuration:**
- `forecasting.hazard_smoothing_alpha = 0.3` (default)
- Higher α = more reactive, Lower α = smoother

---

### Path 3: Anomaly Energy Threshold
**Method:** Cumulative stress accumulation

```
1. Compute anomaly energy at each timestep:
   E(t) = Σ (FusedZ(t))²

2. Cumulative energy:
   E_cumulative(t) = ∫₀ᵗ E(τ) dτ

3. Failure when E_cumulative(t) >= E_fail_threshold

4. RUL = t_fail - t_now
```

**Advantages:**
- Captures cumulative damage (fatigue-like behavior)
- Sensitive to both magnitude and duration of anomalies
- Good for vibration/thermal cycling failures

**Limitations:**
- E_fail_threshold must be calibrated per equipment type
- May lag sudden failures

**Configuration:**
- `forecasting.energy_fail_threshold = 1000.0` (default, adjust per equipment)

---

## Final RUL Calculation

```python
RUL_final = min(RUL_trajectory, RUL_hazard, RUL_energy)

if RUL_trajectory == RUL_final:
    dominant_path = "trajectory"  # Gradual degradation
elif RUL_hazard == RUL_final:
    dominant_path = "hazard"  # Probabilistic failure
else:
    dominant_path = "energy"  # Cumulative damage
```

**Interpretation:**
- **Trajectory-dominated**: Linear health decline, predictable maintenance window
- **Hazard-dominated**: High failure probability, schedule immediate inspection
- **Energy-dominated**: Accumulated stress, reduce operating load or schedule downtime

---

## Confidence Bands

Uncertainty in RUL estimate quantified by confidence bands:

```
1. Trajectory method: Use CI from linear regression
   CI_lower = t_fail using upper confidence bound of slope
   CI_upper = t_fail using lower confidence bound of slope

2. Confidence band width:
   ΔT = |t_upper - t_lower|

3. Report RUL as:
   RUL_final ± ΔT/2
```

**Example:**
```
RUL = 72 hours ± 18 hours
Means: Failure expected between 54h and 90h from now
```

---

## Integration with ACM Pipeline

### Data Flow:
```
1. Health Timeline (ACM_HealthTimeline)
   ↓
2. Forecast Generation (run_enhanced_forecasting_sql)
   ↓
3. Horizon Merging (merge_forecast_horizons)
   ↓
4. Hazard Smoothing (smooth_failure_probability_hazard)
   ↓
5. Multi-Path RUL (compute_rul_multipath)
   ↓
6. Write Results (ACM_RUL_Summary, ACM_FailureHazard_TS)
```

### SQL Schema:
```sql
-- RUL Summary with multipath breakdown
SELECT 
    RUL_Trajectory_Hours,
    RUL_Hazard_Hours,
    RUL_Energy_Hours,
    RUL_Final_Hours,
    ConfidenceBand_Hours,
    DominantPath
FROM ACM_RUL_Summary
WHERE EquipID = ? AND RunID = (SELECT MAX(RunID) FROM ACM_Runs WHERE EquipID = ?)
```

---

## Calibration & Validation

### Backtest Procedure:
1. Identify historical failure events using unified condition
2. For each event, measure RUL prediction accuracy at T-24h, T-48h, T-72h
3. Compute metrics:
   - **MAE** (Mean Absolute Error): avg(|predicted_RUL - actual_RUL|)
   - **MAPE** (Mean Absolute Percentage Error): avg(|error| / actual_RUL)
   - **Hit Rate**: % of predictions within ±20% of actual

### Calibration Adjustments:
- **Trajectory slope**: Adjust if systematic bias detected
- **Hazard α**: Tune for optimal smoothness vs reactivity
- **Energy threshold**: Calibrate per equipment type from failure history

### Script:
```bash
python scripts/evaluate_rul_backtest.py \
    --equip 1 \
    --health-alert-threshold 75 \
    --sustain-points 4 \
    --tolerance-frac 0.2
```

---

## Configuration Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `forecasting.failure_threshold` | 75.0 | 50-90 | HealthIndex failure threshold |
| `forecasting.health_sustain_hours` | 4 | 2-12 | Sustain period for condition 1 |
| `forecasting.fused_z_threshold` | 3.0 | 2.0-5.0 | FusedZ threshold for condition 3 |
| `forecasting.fused_z_sustain_hours` | 2 | 1-6 | Sustain period for condition 3 |
| `forecasting.hazard_failure_prob` | 0.5 | 0.3-0.7 | Failure prob for hazard path |
| `forecasting.energy_fail_threshold` | 1000.0 | 500-5000 | Cumulative energy threshold |
| `forecasting.max_forecast_hours` | 168.0 | 24-720 | Maximum forecast horizon |

---

## Operational Guidelines

### When RUL < 24 hours:
- **Action**: Schedule immediate maintenance
- **Risk**: High probability of failure within 1 day
- **Recommendation**: Reduce load or shut down if critical

### When RUL 24-72 hours:
- **Action**: Plan maintenance within 3 days
- **Risk**: Moderate risk, window for scheduled downtime
- **Recommendation**: Monitor closely, prepare spare parts

### When RUL > 72 hours:
- **Action**: Routine monitoring
- **Risk**: Low immediate risk
- **Recommendation**: Continue normal operations, review trending

### Dominant Path Interpretation:
- **Trajectory**: Order replacement parts (gradual wear)
- **Hazard**: Inspect for hidden faults (probabilistic concern)
- **Energy**: Reduce operating stress (cumulative damage)

---

## References

1. ISO 13381-1:2015 - Condition monitoring and diagnostics of machines - Prognostics
2. NIST Handbook - Remaining Useful Life Estimation
3. PHM Society - Prognostics and Health Management Best Practices
4. ACM Internal Documentation: Analytics Backbone, COLDSTART_MODE, OMR_DETECTOR

---

**Document Owner:** ACM Development Team  
**Review Cycle:** Quarterly  
**Last Reviewed:** 2025-11-20

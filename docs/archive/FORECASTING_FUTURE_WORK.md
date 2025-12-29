# ACM Forecasting & RUL - Future Development Roadmap

**Status**: DISABLED (as of v10.0.0)  
**Reason**: Focusing on anomaly detection analytical integrity before addressing forecasting architectural issues  
**Revisit**: After anomaly detection feature set is validated and production-ready

---

## Executive Summary

ACM's forecasting and RUL prediction capabilities have been temporarily disabled to focus development efforts on:
1. **Anomaly Detection Core**: OMR (Overall Model Residual), correlation analysis, outlier detection, regime identification
2. **Analytical Integrity**: Ensuring unsupervised learning approach evolves correctly with incoming batch data
3. **Foundation Before Prediction**: Establishing robust multi-detector fusion before building forecasting on top

This document captures the comprehensive forecasting audit findings from December 2025, architectural recommendations, and future implementation priorities.

---

## Critical Issues Discovered (December 2025 Audit)

### 1. Data Gap Problem (ROOT CAUSE)

**Symptom**: Dashboard showing constant 100% health forecasts with invalid confidence intervals

**Root Cause**: Batch processing windows extending beyond available historian data
- ACM_HealthTimeline contains data through **2025-04-14 23:30:00**
- sql_batch_runner.py scheduling batches from **2025-04-27 18:48:00** to **2025-07-06 21:08:59**
- **13-day gap** between last available data and batch window start

**Query Failure**:
```python
# Sliding window query (line ~2106 in forecasting.py)
cutoff_time = current_batch_time - timedelta(hours=72)  # April 24
cur.execute("""
    SELECT Timestamp, HealthIndex, FusedZ
    FROM dbo.ACM_HealthTimeline
    WHERE EquipID = ? AND Timestamp >= ?  -- Returns 0 rows (data ends April 14)
    ORDER BY Timestamp
""", (equip_id, cutoff_time))
```

**Impact**:
- Forecasting fails silently with warning: "[WARNING] No health timeline available from SQL; skipping"
- Most recent run (020000B3-8FF3-4919-94A5-ECA5C900E7A8) wrote ZERO forecasts to ACM_HealthForecast_Continuous
- Dashboard displays stale forecasts from 2023-2024 runs (all ~100% health projecting to September 2025)

### 2. Silent Failure - No Error Handling

**Problem**: When health timeline query returns 0 rows, forecasting returns empty result without triggering alerts

**Consequence**:
- No forecast quality flags written
- No dashboard indicators of prediction reliability
- Old/stale forecasts remain visible indefinitely

**Required Fix**:
```python
# Add graceful degradation
if df_health is None or df_health.empty:
    # Try bootstrap query as fallback
    df_health = query_all_available_history(equip_id)
    if df_health is not None:
        Console.warn("[FORECAST] Using fallback: all available history (data gap detected)")
        forecast.quality_flag = "EXTRAPOLATED_DATA_GAP"
    else:
        # Generate low-confidence extrapolation from state
        return generate_degraded_forecast(prev_state, quality="INSUFFICIENT_DATA")
```

### 3. Batch vs. Streaming Paradigm Mismatch

**Current Architecture** (Batch-Oriented):
```
For each batch window:
  1. Query health history (sliding window or bootstrap)
  2. Fit exponential smoothing model from scratch
  3. Generate forecast horizon (168 hours)
  4. Serialize entire forecast as JSON blob → ForecastState
  5. Write forecast points → ACM_HealthForecast_Continuous
  6. Discard fitted model (except serialized params)
```

**Problems**:
- **O(n) complexity**: Every batch recomputes from historical data (not truly "continuous")
- **State confusion**: ForecastState stores entire forecast horizon (output) rather than just model coefficients (state)
- **No incremental learning**: System blends discrete batch forecasts rather than updating model state
- **Data dependency**: Requires historical query for every prediction (brittle)

**Ideal Architecture** (Streaming State-Space):
```
Initialization (once):
  1. Bootstrap model from historical data
  2. Save model state: {level, trend, variance, regime}
  
For each new data point:
  1. Load model state from DB (O(1))
  2. Update state incrementally: level += alpha * error, trend += beta * alpha * error
  3. Save updated state (O(1))
  4. Generate forecast on-demand from current state (no historical query)
  5. Widen confidence intervals if data gap detected
```

---

## Architectural Analysis: What's Right, What's Wrong

### ✅ What ACM Does Well

1. **Multi-Head Detector System**: OMR, correlation, outliers, PCA, IForest, GMM, AR1 provide diverse anomaly signals
2. **Regime-Aware Processing**: Drift detection and regime identification capture operational mode shifts
3. **Fusion Logic**: Combines detector outputs into unified health score with FusedZ
4. **Hazard-Based RUL**: estimate_rul_monte_carlo() with P10/P50/P90 quantiles is solid conceptual approach
5. **Uncertainty Quantification**: Bootstrap CI and analytic CI options for confidence bounds

### ❌ Critical Weaknesses

1. **No True Continuous Learning**
   - "Continuous forecast" is misnomer - system blends discrete batch predictions
   - Model re-fitted from scratch each batch (not incremental state update)
   - Forecasting disabled = 100% data loss (no fallback)

2. **Data Gap Brittle**
   - Assumes historian data always available for query window
   - No fallback to "best effort" forecast when data missing
   - Silent failures hide prediction unavailability

3. **State Management Confusion**
   - ForecastState stores entire forecast horizon (336 JSON values)
   - Mixes model parameters (level, trend) with forecast output (horizon predictions)
   - Should store ONLY model coefficients, generate forecasts on-demand

4. **No Forecast Verification**
   - No tracking of "24h-ago forecast vs. actual" accuracy over time
   - Cannot measure prediction drift or model degradation
   - Forecast quality metrics (JSON blob) never validated

5. **RUL Oversimplification**
   - Single threshold crossing (Health < 20%) defines "failure"
   - No physics-based degradation models
   - Ignores regime transitions, maintenance interventions, sensor drift

---

## Ideal Production Forecasting Architecture

### Core Principles

1. **Online Learning (O(1) Updates)**
   - State-space model (Kalman filter, exponential smoothing, ARIMA)
   - Model state stored in DB: {level, trend, seasonal, variance, regime}
   - New data point → update state → save (no historical query)

2. **Multi-Horizon Forecasting**
   - Separate models for short (24h), medium (7d), long (30d) horizons
   - Forecast reconciliation to ensure coherence across timescales
   - Horizon-specific uncertainty quantification

3. **Regime-Aware Prediction**
   - Separate model states per operational regime
   - Regime transition probabilities for multi-step forecasts
   - Adjust forecast when regime shift detected

4. **Forecast Verification Loop**
   - Store forecast snapshots separately from model state
   - Daily comparison: "24h-ago forecast vs. actual"
   - Track MAPE, RMSE, calibration score over time
   - Auto-retrain trigger when accuracy degrades

5. **Graceful Degradation**
   - ALWAYS return forecast (even if low quality)
   - Widen confidence intervals when:
     - Data gap detected (missing historian points)
     - Regime shift uncertainty high
     - Model staleness exceeds threshold
   - Quality flags: NORMAL, EXTRAPOLATED, STALE, DEGRADED

### Proposed OnlineHealthForecaster Class

```python
class OnlineHealthForecaster:
    """Streaming state-space forecaster with O(1) updates"""
    
    def __init__(self, equip_id: int, sql_client: SqlClient):
        self.equip_id = equip_id
        self.sql = sql_client
        self.state = self._load_or_initialize_state()
    
    def _load_or_initialize_state(self) -> ForecastState:
        """Load from DB or bootstrap from history"""
        state = self.sql.load_forecast_state(self.equip_id)
        if state is None:
            # Bootstrap: query last 1000 health points, fit initial model
            history = self.sql.query_health_history(self.equip_id, limit=1000)
            state = self._bootstrap_state(history)
        return state
    
    def update(self, health_value: float, timestamp: datetime, regime: int):
        """O(1) incremental state update - no historical query"""
        # Exponential smoothing update
        error = health_value - (self.state.level + self.state.trend)
        self.state.level += self.state.alpha * error
        self.state.trend += self.state.beta * self.state.alpha * error
        
        # Variance update (EWMA)
        self.state.variance = 0.9 * self.state.variance + 0.1 * error**2
        
        # Update timestamp and regime
        self.state.last_update = timestamp
        self.state.regime = regime
        
        # Persist state (O(1) write)
        self.sql.save_forecast_state(self.state)
    
    def forecast(self, horizon_hours: int = 168) -> ForecastResult:
        """Generate forecast from current state - always available"""
        now = datetime.now()
        age_hours = (now - self.state.last_update).total_seconds() / 3600
        
        # Base forecast from state
        timestamps = [self.state.last_update + timedelta(hours=h) for h in range(1, horizon_hours+1)]
        base_forecast = [self.state.level + h * self.state.trend for h in range(1, horizon_hours+1)]
        
        # Uncertainty scaling (wider if state stale)
        staleness_factor = 1.0 + (age_hours / 72.0)  # +1% per hour beyond 3 days
        horizon_std = [np.sqrt(self.state.variance * h * staleness_factor) for h in range(1, horizon_hours+1)]
        
        # Confidence intervals (95%)
        ci_lower = [f - 1.96*s for f, s in zip(base_forecast, horizon_std)]
        ci_upper = [f + 1.96*s for f, s in zip(base_forecast, horizon_std)]
        
        # Quality assessment
        quality = self._assess_quality(age_hours)
        
        return ForecastResult(
            timestamps=timestamps,
            values=base_forecast,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            quality=quality,
            model_state_age_hours=age_hours
        )
    
    def _assess_quality(self, age_hours: float) -> str:
        """Assign quality flag based on state freshness"""
        if age_hours < 2:
            return "NORMAL"
        elif age_hours < 24:
            return "RECENT"
        elif age_hours < 72:
            return "STALE"
        else:
            return "DEGRADED"
    
    def retrain(self, force: bool = False):
        """Periodic re-bootstrap from full history"""
        if force or self._should_retrain():
            history = self.sql.query_health_history(self.equip_id, limit=2000)
            self.state = self._bootstrap_state(history)
            self.sql.save_forecast_state(self.state)
    
    def _should_retrain(self) -> bool:
        """Check if model needs re-bootstrap"""
        age_days = (datetime.now() - self.state.last_retrain).days
        return age_days > 7  # Retrain weekly
```

### Usage in ACM Pipeline

```python
# In acm_main.py - SQL mode
forecaster = OnlineHealthForecaster(equip_id, sql_client)

# Update model state with new health value (O(1))
forecaster.update(health_value=final_health_index, timestamp=win_end, regime=current_regime)

# Generate forecast on-demand (no historical query)
forecast = forecaster.forecast(horizon_hours=168)

# Write forecast to continuous table
output_manager.write_continuous_health_forecast(
    timestamps=forecast.timestamps,
    values=forecast.values,
    ci_lower=forecast.ci_lower,
    ci_upper=forecast.ci_upper,
    quality=forecast.quality
)

# Periodic retraining (weekly or when accuracy degrades)
if should_retrain_today:
    forecaster.retrain(force=False)
```

---

## Immediate Fixes (Before Re-Enabling)

### Priority 1: Data Gap Handling

**File**: `core/forecasting.py`, lines 2120-2135

**Change**: Remove time filter from bootstrap query
```python
# BEFORE (brittle - fails if batch window beyond data)
cur.execute("""
    SELECT Timestamp, HealthIndex, FusedZ
    FROM dbo.ACM_HealthTimeline
    WHERE EquipID = ? AND Timestamp <= ?  -- Excludes future batches
    ORDER BY Timestamp DESC
""", (equip_id, current_batch_time))

# AFTER (robust - queries all available history)
cur.execute("""
    SELECT TOP 2000 Timestamp, HealthIndex, FusedZ
    FROM dbo.ACM_HealthTimeline
    WHERE EquipID = ?
    ORDER BY Timestamp DESC
""", (equip_id,))
```

**Impact**: Forecasting will always load available history, even when batch window is in "future"

### Priority 2: Graceful Degradation

**Add fallback logic** when sliding window returns 0 rows:
```python
# After sliding window query (line ~2115)
if df_health is None or df_health.empty:
    Console.warn("[FORECAST] Sliding window returned 0 rows; trying bootstrap query")
    df_health = _query_bootstrap_health(sql_client, equip_id)
    if df_health is not None and not df_health.empty:
        Console.info(f"[FORECAST] Bootstrap found {len(df_health)} points; continuing")
        forecast_quality = "EXTRAPOLATED_DATA_GAP"
    else:
        Console.error("[FORECAST] No health data available; generating degraded forecast")
        return _generate_degraded_forecast(prev_state, equip_id, current_batch_time)
```

### Priority 3: Forecast Quality Tracking

**Add verification table**:
```sql
CREATE TABLE ACM_ForecastVerification (
    VerificationID INT IDENTITY(1,1) PRIMARY KEY,
    EquipID INT NOT NULL,
    ForecastTimestamp DATETIME2 NOT NULL,     -- When forecast was made
    TargetTimestamp DATETIME2 NOT NULL,       -- What time was forecasted
    ForecastedHealth FLOAT,                   -- Predicted value
    ActualHealth FLOAT,                       -- Observed value (filled later)
    AbsoluteError FLOAT,                      -- |actual - forecast|
    WithinCI BIT,                             -- Was actual within confidence interval?
    ForecastHorizonHours INT,                 -- How many hours ahead
    ModelVersion VARCHAR(20),
    CreatedAt DATETIME2 DEFAULT GETDATE()
);
```

**Add daily verification job**:
```python
def verify_forecasts_against_actuals(sql_client: SqlClient, equip_id: int):
    """Compare yesterday's forecasts to actual observed values"""
    yesterday = datetime.now() - timedelta(days=1)
    
    # Get forecasts made 24h ago for yesterday's timepoints
    forecasts = sql_client.query("""
        SELECT Timestamp, ForecastHealth, CI_Lower, CI_Upper
        FROM ACM_HealthForecast_Continuous
        WHERE EquipID = ? AND CreatedAt BETWEEN ? AND ?
    """, (equip_id, yesterday, yesterday + timedelta(hours=1)))
    
    # Get actual health values for those timestamps
    actuals = sql_client.query("""
        SELECT Timestamp, HealthIndex
        FROM ACM_HealthTimeline
        WHERE EquipID = ? AND Timestamp BETWEEN ? AND ?
    """, (equip_id, yesterday, yesterday + timedelta(hours=1)))
    
    # Compare and log errors
    for forecast, actual in zip(forecasts, actuals):
        error = abs(actual.HealthIndex - forecast.ForecastHealth)
        within_ci = (forecast.CI_Lower <= actual.HealthIndex <= forecast.CI_Upper)
        
        sql_client.insert_forecast_verification(
            equip_id=equip_id,
            forecast_timestamp=forecast.CreatedAt,
            target_timestamp=forecast.Timestamp,
            forecasted_health=forecast.ForecastHealth,
            actual_health=actual.HealthIndex,
            absolute_error=error,
            within_ci=within_ci
        )
```

---

## RUL Prediction Improvements

### Current Issues

1. **Single Threshold Approach**: Defines "failure" as Health < 20%, ignoring:
   - Regime-specific thresholds (startup vs. steady-state)
   - Rate of change (slow degradation vs. rapid fault)
   - Sensor-level defects (one critical sensor failing)

2. **No Physics Integration**: RUL based purely on statistical extrapolation, not:
   - Equipment-specific degradation models
   - Maintenance history (last overhaul, parts replaced)
   - Operating conditions (load, temperature, cycles)

3. **Overly Simplistic Uncertainty**: Monte Carlo samples random noise, but doesn't model:
   - Regime transition uncertainty
   - Measurement error propagation
   - Model misspecification

### Enhanced RUL Architecture

```python
class EnhancedRULEstimator:
    """Multi-model RUL with physics-informed constraints"""
    
    def estimate_rul(self, health_forecast: ForecastResult, 
                     sensor_defects: pd.DataFrame,
                     current_regime: int,
                     maintenance_history: pd.DataFrame) -> RULResult:
        
        # Model 1: Statistical (current approach)
        rul_statistical = self._monte_carlo_threshold_crossing(health_forecast)
        
        # Model 2: Sensor-specific (critical component failure)
        rul_sensors = self._sensor_degradation_model(sensor_defects)
        
        # Model 3: Physics-informed (wear model)
        rul_physics = self._physics_based_degradation(
            current_regime, maintenance_history
        )
        
        # Ensemble: take minimum (most pessimistic) with confidence weighting
        rul_ensemble = self._weighted_ensemble([
            (rul_statistical, 0.4),
            (rul_sensors, 0.3),
            (rul_physics, 0.3)
        ])
        
        return RULResult(
            p10_hours=rul_ensemble.p10,
            p50_hours=rul_ensemble.p50,
            p90_hours=rul_ensemble.p90,
            confidence=rul_ensemble.confidence,
            primary_model=rul_ensemble.primary_model,
            failure_mode=rul_ensemble.predicted_failure_mode
        )
```

---

## Development Priorities (When Re-Enabling)

### Phase 1: Foundation (2-3 weeks)
1. Implement OnlineHealthForecaster with O(1) updates
2. Add data gap handling and graceful degradation
3. Create ACM_ForecastVerification table and daily verification job
4. Add forecast quality flags (NORMAL, STALE, DEGRADED, EXTRAPOLATED)

### Phase 2: Multi-Horizon (2 weeks)
1. Separate short-term (24h), medium-term (7d), long-term (30d) models
2. Implement forecast reconciliation across horizons
3. Add regime-specific model states

### Phase 3: RUL Enhancement (3 weeks)
1. Develop sensor-specific degradation models
2. Integrate maintenance history into RUL calculation
3. Add physics-informed wear models (equipment-specific)
4. Implement multi-model RUL ensemble

### Phase 4: Validation & Monitoring (2 weeks)
1. Build forecast accuracy dashboard (MAPE, RMSE, calibration)
2. Add auto-retrain triggers based on accuracy degradation
3. Implement forecast explainability (top contributing sensors)
4. Production testing with live data

---

## References

**Current Implementation**:
- `core/forecasting.py` (3350 lines) - Enhanced forecasting engine
- `core/rul_engine.py` (500+ lines) - RUL estimation with Monte Carlo
- `ACM_ForecastState` table - Model state persistence
- `ACM_HealthForecast_Continuous` table - Forecast output storage

**Architectural Documents**:
- `README.md` - ACM product overview
- `docs/ACM_SYSTEM_OVERVIEW.md` - Complete architecture walkthrough
- `docs/CONTINUOUS_LEARNING_ROBUSTNESS.md` - Current learning approach

**Related Issues**:
- Data gap problem (batch windows beyond available historian data)
- Silent failures (forecasting returns empty when query fails)
- State confusion (ForecastState stores output, not just model coefficients)

---

**Document Version**: 1.0  
**Created**: December 2025  
**Author**: ACM Development Team  
**Next Review**: When anomaly detection feature set validated and ready for forecasting re-enablement

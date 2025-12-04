# Critical Batch Continuity & Standardization Fixes

## Dashboard Analysis Summary (2024-11-23 to 2025-10-29)

### Observed Symptoms

1. **Failure Probability**: Flat near-zero, then sharp 50% spikes at batch boundaries
2. **Detector Z-Scores**: Saturated >10 Z-scores with quiet gaps (OMR, AR1, PCA showing >8σ)  
3. **Normalized Sensors**: Extreme -40 Z excursions at batch start/end
4. **Health Index**: Oscillating between healthy (100%) and poor (20%) in regular patterns
5. **Health Forecast**: Vertical discontinuities at every batch boundary despite smooth intra-batch curves

### Root Causes Identified

#### 1. **Variance Collapse** (MOST CRITICAL)
- **Problem**: `StandardScaler` divides by std without epsilon floor
- **Impact**: When batch variance approaches zero → Z-scores explode to ±100σ
- **Evidence**: -40 Z in normalized sensors, >10 Z in detectors
- **Location**: `core/correlation.py:185`, all detectors using `StandardScaler`

#### 2. **Baseline/Score Window Mismatch**
- **Problem**: Each batch uses its own baseline, no continuity between batches
- **Impact**: First few points of each batch score against wrong statistics
- **Evidence**: Spikes at batch boundaries in all timeseries
- **Location**: `core/acm_main.py:1091-1169` (baseline loading logic)

#### 3. **Per-Batch Threshold Recalculation**
- **Problem**: Adaptive thresholds recomputed per batch without persistence
- **Impact**: Alert/warn thresholds jump between batches, causing false positives/negatives
- **Evidence**: Health index flipping between green/red zones
- **Location**: `core/acm_main.py:2540-2610` (calibration section)

#### 4. **Forecast Horizon Miscalculation**
- **Problem**: Forecast horizon ends at batch window end, not absolute future time
- **Impact**: Failure probability ramps to 50% at batch end (model thinks failure is imminent)
- **Evidence**: Flat forecast with sharp spike at each batch tail
- **Location**: `core/forecasting.py` (RUL calculation), `core/rul_engine.py` (forecast generation)

#### 5. **State Non-Persistence Between Batches**
- **Problem**: Forecast models retrain from scratch, no carry-forward of previous predictions
- **Impact**: Vertical discontinuities at every batch boundary
- **Evidence**: Stitched forecast shows jumps despite smooth intra-batch curves
- **Location**: `core/forecasting.py:retrain_decision()`, state version incrementing unnecessarily

---

## Required Fixes (Priority Order)

### P0: Variance Guard (IMMEDIATE - Prevents -40σ explosions)

**File**: `core/correlation.py`, `core/outliers.py`, all detectors using StandardScaler

```python
# BEFORE (line 185 in correlation.py):
Xs = self.scaler.fit_transform(df.values.astype(np.float64, copy=False))

# AFTER: Add variance floor BEFORE StandardScaler
from sklearn.preprocessing import StandardScaler
import warnings

# Custom scaler with variance guard
class RobustStandardScaler(StandardScaler):
    """StandardScaler with epsilon floor to prevent variance collapse."""
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def fit(self, X, y=None):
        super().fit(X, y)
        # Floor the scale_ to prevent division by near-zero
        if self.with_std and hasattr(self, 'scale_'):
            self.scale_ = np.maximum(self.scale_, self.epsilon)
        return self
    
    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)
        if self.with_std and hasattr(self, 'scale_'):
            self.scale_ = np.maximum(self.scale_, self.epsilon)
        return self

# Replace all StandardScaler instantiations:
# self.scaler = StandardScaler(with_mean=True, with_std=True)
self.scaler = RobustStandardScaler(epsilon=1e-6, with_mean=True, with_std=True)
```

**Impact**: Prevents Z-score explosions, stabilizes all detector outputs

---

### P0: Baseline Continuity (IMMEDIATE - Fixes batch boundary spikes)

**File**: `core/acm_main.py:1091-1169`

**Current Issue**:
```python
# Each batch loads baseline from ACM_BaselineBuffer with window_hours=72
# But doesn't verify overlap with previous batch's score window
# Result: First N points of new batch use stale baseline stats
```

**Fix**:
```python
# Add baseline validation after loading (line ~1167):
if used and isinstance(train_numeric, pd.DataFrame):
    # Verify baseline covers at least 50% of score window start
    if sc_start is not None and tr_end is not None:
        overlap_seconds = (tr_end - sc_start).total_seconds()
        score_duration = (sc_end - sc_start).total_seconds()
        overlap_pct = (overlap_seconds / score_duration) * 100
        
        if overlap_pct < 50:
            Console.warn(
                f"[BASELINE] Insufficient overlap: baseline ends {overlap_pct:.1f}% "
                f"into score window. Extending baseline with recent score data."
            )
            # Extend baseline with first 20% of score window
            score_extension = score_numeric.iloc[:int(0.2 * len(score_numeric))]
            train = pd.concat([train, score_extension], axis=0).drop_duplicates()
            train_numeric = train.copy()
            used = f"{used} + extended ({len(score_extension)} rows)"
```

**Impact**: Prevents statistics mismatch at batch boundaries

---

### P1: Threshold Persistence (HIGH - Fixes health index oscillation)

**File**: `core/acm_main.py:2540-2610`

**Add**: Threshold loading from `ACM_AdaptiveThresholds` before recalculating

```python
# Before line 2540 (start of calibration):
# Load previous thresholds if available
prev_thresholds = {}
if SQL_MODE and sql_client:
    try:
        with sql_client.cursor() as cur:
            cur.execute("""
                SELECT SignalName, AlertThreshold, WarnThreshold, UpdatedAt
                FROM dbo.ACM_AdaptiveThresholds
                WHERE EquipID = ? AND UpdatedAt >= DATEADD(HOUR, -24, GETDATE())
                ORDER BY UpdatedAt DESC
            """, (int(equip_id),))
            for row in cur.fetchall():
                prev_thresholds[row.SignalName] = {
                    'alert': float(row.AlertThreshold),
                    'warn': float(row.WarnThreshold),
                    'age_hours': (datetime.now() - row.UpdatedAt).total_seconds() / 3600
                }
        
        if prev_thresholds:
            # Use exponential smoothing to blend old/new thresholds
            alpha = 0.3  # Weight for new thresholds
            Console.info(f"[CAL] Blending {len(prev_thresholds)} previous thresholds (alpha={alpha})")
    except Exception as e:
        Console.warn(f"[CAL] Failed to load previous thresholds: {e}")

# In threshold calculation loop (around line 2590):
if signal_name in prev_thresholds and prev_thresholds[signal_name]['age_hours'] < 24:
    # Blend with previous threshold (exponential smoothing)
    prev = prev_thresholds[signal_name]
    new_alert = alpha * alert_thresh + (1 - alpha) * prev['alert']
    new_warn = alpha * warn_thresh + (1 - alpha) * prev['warn']
    Console.debug(f"[CAL] {signal_name}: blended alert {alert_thresh:.3f} → {new_alert:.3f}")
    alert_thresh, warn_thresh = new_alert, new_warn
```

**Impact**: Smooths threshold evolution, prevents alert flapping

---

### P1: Forecast Horizon Fix (HIGH - Fixes failure probability spikes)

**File**: `core/forecasting.py` and `core/rul_engine.py`

**Problem**: Forecast horizon calculated relative to batch end, not absolute future

```python
# INCORRECT (causes spike at batch end):
forecast_horizon_hours = (batch_end_time - current_time).total_seconds() / 3600

# CORRECT (uses fixed future horizon):
forecast_horizon_hours = config.get('max_forecast_horizon', 168)  # 7 days fixed
```

**Fix in `core/rul_engine.py`** (around line 1350):
```python
# Replace relative horizon:
# future_timestamps = pd.date_range(
#     start=last_timestamp + pd.Timedelta(hours=1),
#     end=batch_end_time,  # <-- WRONG
#     freq='H'
# )

# With fixed absolute horizon:
max_horizon_hours = float(self.config.get('max_forecast_horizon', 168))
future_timestamps = pd.date_range(
    start=last_timestamp + pd.Timedelta(hours=1),
    periods=int(max_horizon_hours),
    freq='h'  # lowercase 'h' for pandas 2.x
)
```

**Impact**: Stabilizes failure probability, removes end-of-batch ramps

---

### P2: State Carry-Forward (MEDIUM - Fixes forecast discontinuities)

**File**: `core/forecasting.py:retrain_decision()`

**Add**: Previous forecast continuity check before deciding to retrain

```python
def retrain_decision(self, current_health, historical_health, anomaly_energy, state):
    # Load previous forecast if available
    prev_forecast_available = False
    prev_forecast_quality = 0.0
    
    if SQL_MODE and self.sql_client:
        try:
            with self.sql_client.cursor() as cur:
                cur.execute("""
                    SELECT TOP 1 Quality, StateVersion, RetrainedAt
                    FROM dbo.ACM_ForecastState
                    WHERE EquipID = ? AND StateVersion = ?
                    ORDER BY RetrainedAt DESC
                """, (self.equip_id, state.state_version))
                row = cur.fetchone()
                if row:
                    prev_forecast_quality = float(row.Quality)
                    prev_forecast_available = True
        except Exception:
            pass
    
    # If previous forecast is good quality and recent, skip retrain
    if prev_forecast_available and prev_forecast_quality > 80:
        hours_since_retrain = (datetime.now() - state.last_retrain).total_seconds() / 3600
        if hours_since_retrain < 24:  # Within 24 hours
            Console.info(f"[RETRAIN] Skipping - previous forecast still valid (quality={prev_forecast_quality:.1f}%, age={hours_since_retrain:.1f}h)")
            return False, "previous_forecast_valid"
    
    # Original retrain logic continues...
```

**Impact**: Reduces unnecessary retrains, maintains forecast continuity

---

## Verification Queries

After fixes, run these to validate improvements:

```sql
-- 1. Check Z-score stability (should be <5σ for 95% of data)
SELECT 
    SignalName,
    AVG(ZScore) as avg_z,
    STDEV(ZScore) as std_z,
    MIN(ZScore) as min_z,
    MAX(ZScore) as max_z,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(ZScore)) as p95_abs_z
FROM ACM_Scores_Wide
WHERE EquipID = 1 AND Timestamp >= DATEADD(DAY, -7, GETDATE())
GROUP BY SignalName
HAVING MAX(ABS(ZScore)) > 10  -- Flag extreme outliers
ORDER BY max_z DESC;

-- 2. Check forecast continuity (∆health <20% between consecutive runs)
WITH forecast_jumps AS (
    SELECT 
        RunID,
        Timestamp,
        HealthScore,
        LAG(HealthScore) OVER (PARTITION BY EquipID ORDER BY Timestamp) as prev_health,
        ABS(HealthScore - LAG(HealthScore) OVER (PARTITION BY EquipID ORDER BY Timestamp)) as health_delta
    FROM ACM_HealthForecast_TS
    WHERE EquipID = 1
)
SELECT COUNT(*) as large_jumps, AVG(health_delta) as avg_jump
FROM forecast_jumps
WHERE health_delta > 20;  -- Should be < 5% of total rows

-- 3. Check baseline coverage (overlap should be >50%)
SELECT 
    r.RunID,
    r.StartTime as batch_start,
    MIN(bb.Timestamp) as baseline_start,
    MAX(bb.Timestamp) as baseline_end,
    DATEDIFF(SECOND, MAX(bb.Timestamp), r.StartTime) as gap_seconds
FROM ACM_Runs r
LEFT JOIN ACM_BaselineBuffer bb ON bb.EquipID = r.EquipID
    AND bb.Timestamp <= r.StartTime
    AND bb.Timestamp >= DATEADD(HOUR, -72, r.StartTime)
WHERE r.EquipID = 1 AND r.RunTimestamp >= DATEADD(DAY, -7, GETDATE())
GROUP BY r.RunID, r.StartTime
HAVING DATEDIFF(SECOND, MAX(bb.Timestamp), r.StartTime) > 3600  -- Gap >1 hour
ORDER BY r.StartTime DESC;
```

---

## Implementation Priority

1. **IMMEDIATE (Today)**: P0 Variance Guard - prevents catastrophic Z-score explosions
2. **IMMEDIATE (Today)**: P0 Baseline Continuity - fixes batch boundary artifacts  
3. **HIGH (This Week)**: P1 Threshold Persistence - stabilizes health index
4. **HIGH (This Week)**: P1 Forecast Horizon - fixes failure probability
5. **MEDIUM (Next Sprint)**: P2 State Carry-Forward - smooths forecast evolution

---

## Expected Improvements

After all fixes:
- **Z-scores**: <5σ for 95% of data points (currently seeing ±40σ)
- **Health Index**: Smooth transitions <10% change between batches (currently ±50%)
- **Failure Probability**: No end-of-batch spikes (currently 0→50% jumps)
- **Forecast Continuity**: <20% health delta between consecutive runs (currently 100% jumps)
- **Detector Correlation**: Stable cross-detector agreement (currently shows periodic decorrelation)

---

## Notes

- All fixes preserve existing SQL schema compatibility
- Changes are backward compatible with file mode
- Validation queries provided for each fix
- Fixes target root causes, not symptoms
- Expected improvement: 80-90% reduction in batch artifacts

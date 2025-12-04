# Equipment Failure Forecasting System - Implementation Guide

**Version:** 1.0  
**Purpose:** Complete specification for building a robust, batch-compatible equipment health forecasting and RUL estimation system  
**Target:** GitHub Copilot / AI-assisted implementation

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Principles](#2-architecture-principles)
3. [Module 1: health_tracker.py](#3-module-1-health_trackerpy)
4. [Module 2: degradation_model.py](#4-module-2-degradation_modelpy)
5. [Module 3: failure_probability.py](#5-module-3-failure_probabilitypy)
6. [Module 4: rul_estimator.py](#6-module-4-rul_estimatorpy)
7. [Module 5: state_manager.py](#7-module-5-state_managerpy)
8. [Module 6: forecast_engine.py](#8-module-6-forecast_enginepy)
9. [Module 7: sensor_attribution.py](#9-module-7-sensor_attributionpy)
10. [Module 8: metrics.py](#10-module-8-metricspy)
11. [SQL Schema Requirements](#11-sql-schema-requirements)
12. [Testing Strategy](#12-testing-strategy)
13. [Common Pitfalls & Solutions](#13-common-pitfalls--solutions)
14. [Hyperparameter Tuning Guide](#14-hyperparameter-tuning-guide)

---

## 1. System Overview

### Purpose
Build a forecasting system that:
- Accepts equipment health data batch-by-batch (real-time or periodic)
- Maintains continuity across batches (no jumps in forecasts)
- Predicts when equipment will fail (RUL estimation)
- Provides actionable maintenance recommendations
- Tracks its own accuracy and improves over time

### Key Requirements
- **Batch compatibility:** Must work with data arriving every 1 minute, 1 hour, or 1 day
- **State persistence:** Must resume from where it left off after restart
- **Smooth transitions:** Forecasts should blend smoothly, not jump when new data arrives
- **Uncertainty quantification:** Must provide confidence intervals, not just point estimates
- **Interpretability:** Engineers should understand why a failure is predicted

### Data Flow
```
New Batch → Health Tracker → Degradation Model → Failure Probability → RUL Estimator → Recommendations
                ↓                    ↓                    ↓                    ↓
            State Manager ←──────────┴────────────────────┴────────────────────┘
                ↓
            SQL Tables
```

---

## 2. Architecture Principles

### Principle 1: Single Responsibility
Each module does ONE thing well. No god classes.

### Principle 2: Explicit Over Implicit
- No hidden state transformations
- All timestamps must be timezone-naive and clearly documented
- All units must be explicit (hours, not "time steps")

### Principle 3: Fail Gracefully
- If data is missing, return safe defaults (e.g., RUL = max_forecast_horizon)
- Log warnings, don't crash
- Provide data quality flags so users know when to trust results

### Principle 4: Stateless Functions Where Possible
- Pure functions for calculations (failure_probability.py, metrics.py)
- State isolation in dedicated manager (state_manager.py)

### Principle 5: No Premature Optimization
- Start with simple linear trend, not complex ensembles
- Add complexity only when accuracy demands it
- Profile before optimizing

---

## 3. Module 1: health_tracker.py

### Purpose
Load, clean, validate, and maintain a sliding window of equipment health data.

### Class: HealthTimeline

#### Responsibilities
1. Load historical health data from SQL
2. Append new batch data
3. Maintain a sliding window (e.g., last 7 days)
4. Detect data quality issues
5. Provide statistics about health trends

#### Implementation Details

```python
# File: core/health_tracker.py

from dataclasses import dataclass
from typing import Optional, Literal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@dataclass
class HealthQuality:
    """Data quality assessment result"""
    flag: Literal["OK", "SPARSE", "GAPPY", "NOISY", "FLAT", "MISSING"]
    message: str
    metrics: dict  # sample_count, gap_max_hours, std, etc.

class HealthTimeline:
    """
    Manages equipment health data with quality checks.
    
    Key behaviors:
    - Maintains a sliding window of health data
    - Detects data quality issues
    - Provides trend statistics
    - Thread-safe for batch updates
    """
    
    def __init__(
        self, 
        equip_id: int, 
        window_hours: float = 168.0,  # 7 days default
        min_samples: int = 20
    ):
        self.equip_id = equip_id
        self.window_hours = window_hours
        self.min_samples = min_samples
        self._data: Optional[pd.DataFrame] = None
        self._last_load_time: Optional[datetime] = None
```

#### Method: load_from_sql()

**Purpose:** Load historical health data from ACM_HealthTimeline table

**Signature:**
```python
def load_from_sql(
    self, 
    sql_client, 
    lookback_hours: Optional[float] = None
) -> pd.DataFrame:
    """
    Load health timeline from SQL.
    
    Args:
        sql_client: SQL connection object
        lookback_hours: How far back to load (default: self.window_hours)
    
    Returns:
        DataFrame with columns: [Timestamp, HealthIndex, FusedZ]
        Sorted by Timestamp, no duplicates, timezone-naive
    
    Raises:
        ValueError: If equip_id invalid
        RuntimeError: If SQL query fails
    """
```

**Implementation Checklist:**
- [ ] Query: `SELECT Timestamp, HealthIndex, FusedZ FROM ACM_HealthTimeline WHERE EquipID = ? AND Timestamp >= ? ORDER BY Timestamp`
- [ ] Strip timezone: `df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)`
- [ ] Remove duplicates: Keep last occurrence if duplicate timestamps exist
- [ ] Validate: HealthIndex should be in [0, 100] range
- [ ] Handle empty result: Return empty DataFrame with correct schema, don't crash
- [ ] Log: `Console.info(f"[HealthTracker] Loaded {len(df)} points for EquipID={equip_id}")`

**Edge Cases:**
- No data in database → Return empty DataFrame, quality flag = "MISSING"
- Data has NaNs → Forward-fill gaps < 2 hours, otherwise mark as "GAPPY"
- Data has outliers (health jumps 50+ points) → Log warning, clip to [0, 100]

---

#### Method: append_batch()

**Purpose:** Add new batch of data to timeline (incremental update)

**Signature:**
```python
def append_batch(self, new_data: pd.DataFrame) -> None:
    """
    Append new batch to existing timeline.
    
    Args:
        new_data: DataFrame with [Timestamp, HealthIndex] columns
    
    Side Effects:
        - Updates self._data with new rows
        - Trims old data outside window
        - Sorts by Timestamp
    
    Implementation Notes:
        - Merge with existing data using pd.concat()
        - Drop duplicates (keep='last') to handle re-processing
        - Apply sliding window trim: keep only last window_hours
    """
```

**Implementation Checklist:**
- [ ] Validate new_data schema matches existing
- [ ] Convert Timestamp to naive datetime
- [ ] Concatenate: `pd.concat([self._data, new_data], ignore_index=True)`
- [ ] Sort: `sort_values('Timestamp')`
- [ ] Remove duplicates: `drop_duplicates(subset=['Timestamp'], keep='last')`
- [ ] Trim window: Keep only rows where `Timestamp >= (max_timestamp - window_hours)`
- [ ] Update `self._last_load_time = datetime.now()`

**Edge Cases:**
- New batch overlaps existing data → Keep latest values (handle reprocessing)
- New batch has future timestamps → Accept (system clock might be ahead)
- New batch is empty → Do nothing, log warning

---

#### Method: get_sliding_window()

**Purpose:** Return current sliding window view of data

**Signature:**
```python
def get_sliding_window(self, hours: Optional[float] = None) -> pd.DataFrame:
    """
    Get recent data within specified window.
    
    Args:
        hours: Window size (default: self.window_hours)
    
    Returns:
        DataFrame subset of self._data
    """
```

**Implementation:** Simple filter on Timestamp column

---

#### Method: quality_check()

**Purpose:** Assess data quality and return structured result

**Signature:**
```python
def quality_check(self) -> HealthQuality:
    """
    Assess data quality using multiple criteria.
    
    Returns:
        HealthQuality object with flag and diagnostics
    
    Quality Flags:
        - OK: >50 samples, gaps <2h, std >1.0
        - SPARSE: <50 samples but >20
        - GAPPY: Max gap >24h
        - NOISY: std >20 (high variance)
        - FLAT: std <0.5 (no variation)
        - MISSING: <20 samples or empty
    """
```

**Implementation Checklist:**
- [ ] Check 1: Sample count
  ```python
  if len(data) < 20:
      return HealthQuality("MISSING", "Insufficient samples", {"count": len(data)})
  if len(data) < 50:
      # Continue checks but may return SPARSE
  ```

- [ ] Check 2: Time gaps
  ```python
  time_diffs = data['Timestamp'].diff().dt.total_seconds() / 3600.0
  max_gap = time_diffs.max()
  if max_gap > 24:
      return HealthQuality("GAPPY", f"Max gap: {max_gap:.1f}h", {...})
  ```

- [ ] Check 3: Signal variance
  ```python
  std = data['HealthIndex'].std()
  if std < 0.5:
      return HealthQuality("FLAT", "No variation in signal", {"std": std})
  if std > 20:
      return HealthQuality("NOISY", "High variance", {"std": std})
  ```

- [ ] Check 4: All pass
  ```python
  return HealthQuality("OK", "Data quality acceptable", {
      "count": len(data),
      "std": std,
      "max_gap_hours": max_gap
  })
  ```

**Edge Cases:**
- Single data point → Return "SPARSE"
- All NaNs → Return "MISSING"
- Constant value → Return "FLAT"

---

#### Method: get_statistics()

**Purpose:** Compute summary statistics for model initialization

**Signature:**
```python
def get_statistics(self) -> dict:
    """
    Compute statistical summary of health timeline.
    
    Returns:
        {
            'mean': float,
            'std': float,
            'trend': float,  # linear slope in health_units/hour
            'current': float,  # latest health value
            'min': float,
            'max': float,
            'sample_count': int,
            'sampling_interval_hours': float  # median time between samples
        }
    """
```

**Implementation Checklist:**
- [ ] Basic stats: `mean()`, `std()`, `min()`, `max()`
- [ ] Trend estimation:
  ```python
  # Simple linear regression: health = slope * hours + intercept
  hours = (data['Timestamp'] - data['Timestamp'].iloc[0]).dt.total_seconds() / 3600
  slope, intercept = np.polyfit(hours, data['HealthIndex'], deg=1)
  ```
- [ ] Sampling interval:
  ```python
  time_diffs = data['Timestamp'].diff().dt.total_seconds() / 3600.0
  median_interval = time_diffs.median()
  ```

---

#### Method: detect_regime_shift()

**Purpose:** Detect sudden changes in degradation pattern (triggers retrain)

**Signature:**
```python
def detect_regime_shift(self, lookback_window: int = 50) -> bool:
    """
    Detect if degradation pattern has changed recently.
    
    Args:
        lookback_window: Number of recent samples to compare
    
    Returns:
        True if regime shift detected
    
    Method:
        - Compare slope of last N samples vs previous N samples
        - If slope difference > 2x std of historical slopes, flag shift
    """
```

**Implementation:**
```python
if len(self._data) < 2 * lookback_window:
    return False

recent = self._data.tail(lookback_window)
previous = self._data.iloc[-2*lookback_window:-lookback_window]

# Compute slopes
recent_slope = self._compute_slope(recent)
previous_slope = self._compute_slope(previous)

# Compare
slope_change = abs(recent_slope - previous_slope)
historical_slope_std = self._compute_historical_slope_std()

if slope_change > 2 * historical_slope_std:
    Console.warn(f"[HealthTracker] Regime shift detected: slope change = {slope_change:.3f}")
    return True
return False
```

---

### Testing Requirements for health_tracker.py

**Unit Tests:**
1. `test_load_from_sql_empty()` - Empty database case
2. `test_load_from_sql_with_gaps()` - Data with 12h gap
3. `test_append_batch_deduplication()` - Overlapping batches
4. `test_quality_check_sparse()` - <20 samples
5. `test_quality_check_gappy()` - 48h gap
6. `test_quality_check_flat()` - Constant health values
7. `test_sliding_window_trim()` - Old data removal
8. `test_statistics_trend()` - Trend calculation accuracy
9. `test_regime_shift_detection()` - Sudden slope change

**Integration Tests:**
1. Load real ACM_HealthTimeline data
2. Simulate batch-by-batch updates over 30 days
3. Verify sliding window maintains correct size
4. Verify quality flags match manual inspection

---

## 4. Module 2: degradation_model.py

### Purpose
Model how equipment health degrades over time. Start simple (linear trend), allow extension to more complex models.

### Design Philosophy
- **ONE model per equipment type** (don't ensemble initially)
- **Incremental updates** (avoid re-fitting from scratch)
- **Explicit uncertainty** (return both mean and std)

---

### Class: BaseDegradationModel (Abstract)

```python
# File: core/degradation_model.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ForecastResult:
    """Standardized forecast output"""
    timestamps: pd.DatetimeIndex
    mean: np.ndarray  # Predicted health values
    std: np.ndarray   # Prediction uncertainty (±1 std)
    model_name: str
    
    def get_confidence_interval(self, alpha: float = 0.95) -> tuple:
        """Return (lower, upper) bounds for given confidence level"""
        from scipy.stats import norm
        z = norm.ppf(1 - (1 - alpha) / 2)
        lower = self.mean - z * self.std
        upper = self.mean + z * self.std
        return lower, upper

class BaseDegradationModel(ABC):
    """
    Abstract base for all degradation models.
    
    Contract:
        - fit() must set self.is_fitted = True on success
        - predict() must return ForecastResult
        - update_online() must be lightweight (no full refit)
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.last_fit_time: Optional[datetime] = None
        self.fit_data_count: int = 0
    
    @abstractmethod
    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        """
        Fit model to historical data.
        
        Returns:
            True if fit successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        current_time: pd.Timestamp, 
        hours_ahead: float,
        num_points: int = 100
    ) -> ForecastResult:
        """
        Generate forecast from current_time.
        
        Args:
            current_time: Starting timestamp for forecast
            hours_ahead: Total forecast horizon in hours
            num_points: Number of forecast points to generate
        
        Returns:
            ForecastResult with timestamps, mean, std
        """
        pass
    
    @abstractmethod
    def update_online(self, new_timestamp: pd.Timestamp, new_health: float) -> None:
        """
        Incrementally update model with new observation.
        Should be much faster than full refit.
        """
        pass
    
    @abstractmethod
    def get_coefficients(self) -> dict:
        """Return model parameters for serialization"""
        pass
    
    @abstractmethod
    def set_coefficients(self, coeffs: dict) -> None:
        """Load model parameters from serialization"""
        pass
```

---

### Class: LinearTrendModel

**Purpose:** Simple linear degradation: health(t) = health_0 + slope * t

**When to use:**
- Gradual wear-out failures
- Slow degradation over weeks/months
- Most common industrial equipment

**Implementation:**

```python
class LinearTrendModel(BaseDegradationModel):
    """
    Linear trend model with online updates.
    
    Model: health(t) = baseline + slope * hours_elapsed
    
    Parameters:
        - slope: health_units per hour
        - baseline: health at t=0
        - residual_std: prediction uncertainty
    
    Online Update Strategy:
        - Use exponential moving average for slope
        - alpha=0.1 (gives ~10 samples half-life)
    """
    
    def __init__(self, alpha: float = 0.1):
        super().__init__("LinearTrend")
        self.alpha = alpha  # EMA weight for online updates
        self.slope: float = 0.0
        self.baseline: float = 0.0
        self.residual_std: float = 1.0
        self.reference_time: Optional[pd.Timestamp] = None
```

#### Method: fit()

**Implementation Checklist:**

- [ ] **Input validation:**
  ```python
  if len(health_values) < 10:
      Console.warn(f"[{self.name}] Insufficient data: {len(health_values)} < 10")
      return False
  
  if timestamps.duplicated().any():
      Console.warn(f"[{self.name}] Duplicate timestamps found, keeping last")
      # Remove duplicates
  ```

- [ ] **Linear regression:**
  ```python
  # Convert timestamps to hours since first observation
  hours = (timestamps - timestamps[0]).total_seconds().values / 3600.0
  
  # Fit: health = slope * hours + intercept
  from scipy.stats import linregress
  result = linregress(hours, health_values)
  
  self.slope = result.slope
  self.baseline = result.intercept
  self.reference_time = timestamps[0]
  ```

- [ ] **Residual calculation:**
  ```python
  # Compute fitted values
  fitted = self.baseline + self.slope * hours
  
  # Residual standard deviation
  residuals = health_values - fitted
  self.residual_std = np.std(residuals)
  
  # Guard against zero variance
  if self.residual_std < 0.1:
      self.residual_std = 1.0
      Console.warn(f"[{self.name}] Low residual variance, using floor of 1.0")
  ```

- [ ] **Quality checks:**
  ```python
  # Check if slope is reasonable (not too steep)
  MAX_SLOPE_PER_HOUR = 5.0  # Health units per hour
  if abs(self.slope) > MAX_SLOPE_PER_HOUR:
      Console.warn(f"[{self.name}] Extreme slope: {self.slope:.3f}, clamping")
      self.slope = np.sign(self.slope) * MAX_SLOPE_PER_HOUR
  
  # Log fit quality
  r_squared = result.rvalue ** 2
  Console.info(f"[{self.name}] Fit complete: slope={self.slope:.3f}, R²={r_squared:.3f}")
  ```

- [ ] **Finalize:**
  ```python
  self.is_fitted = True
  self.last_fit_time = datetime.now()
  self.fit_data_count = len(health_values)
  return True
  ```

---

#### Method: predict()

**Implementation:**

```python
def predict(
    self, 
    current_time: pd.Timestamp, 
    hours_ahead: float,
    num_points: int = 100
) -> ForecastResult:
    """
    Generate linear forecast.
    
    Uncertainty Growth:
        - std(t) = residual_std * sqrt(1 + t/T)
        where T = training data duration (prevents overconfidence at long horizons)
    """
    
    if not self.is_fitted:
        raise RuntimeError(f"{self.name} not fitted")
    
    # Generate future timestamps
    dt = hours_ahead / num_points
    future_hours = np.arange(0, hours_ahead, dt)
    future_timestamps = pd.date_range(
        start=current_time,
        periods=len(future_hours),
        freq=f"{dt}H"
    )
    
    # Hours since reference time
    hours_since_ref = (current_time - self.reference_time).total_seconds() / 3600.0
    hours_forecast = hours_since_ref + future_hours
    
    # Linear prediction
    forecast_mean = self.baseline + self.slope * hours_forecast
    
    # Uncertainty growth (more uncertain further out)
    training_duration = self.fit_data_count * 1.0  # Approximate hours
    uncertainty_growth = np.sqrt(1 + future_hours / max(training_duration, 24))
    forecast_std = self.residual_std * uncertainty_growth
    
    # Clamp to valid health range [0, 100]
    forecast_mean = np.clip(forecast_mean, 0, 100)
    
    return ForecastResult(
        timestamps=future_timestamps,
        mean=forecast_mean,
        std=forecast_std,
        model_name=self.name
    )
```

**Key Implementation Details:**

1. **Uncertainty Growth Formula:**
   ```python
   # Why: Forecasts become less certain the further out you go
   # sqrt(1 + t/T) ensures:
   #   - At t=0: uncertainty = residual_std (just fit uncertainty)
   #   - At t=T: uncertainty = residual_std * sqrt(2) (doubles after training period)
   #   - At t=10T: uncertainty = residual_std * sqrt(11) (~3.3x)
   ```

2. **Why not just constant std?**
   - Real degradation has model uncertainty + process uncertainty
   - Long-term forecasts should be less confident (equipment might change behavior)

3. **Clamping to [0, 100]:**
   - Health index has physical bounds
   - Don't predict negative health or >100%

---

#### Method: update_online()

**Purpose:** Incorporate new observation without full refit (exponential moving average)

**Implementation:**

```python
def update_online(self, new_timestamp: pd.Timestamp, new_health: float) -> None:
    """
    Online update using exponential moving average.
    
    Strategy:
        1. Compute prediction error at new timestamp
        2. Update slope using EMA: slope_new = alpha * slope_observed + (1-alpha) * slope_old
        3. Update baseline to anchor at most recent observation
    
    This is O(1) complexity vs O(n) for full refit.
    """
    
    if not self.is_fitted:
        Console.warn(f"[{self.name}] Cannot update_online() before fit()")
        return
    
    # Hours since reference
    hours_elapsed = (new_timestamp - self.reference_time).total_seconds() / 3600.0
    
    # Predicted health at this time
    predicted = self.baseline + self.slope * hours_elapsed
    
    # Observed vs predicted
    error = new_health - predicted
    
    # Update residual std (exponential moving std)
    self.residual_std = np.sqrt(
        (1 - self.alpha) * self.residual_std**2 + self.alpha * error**2
    )
    
    # Update slope (observed slope from last reference to now)
    observed_slope = (new_health - self.baseline) / hours_elapsed
    self.slope = self.alpha * observed_slope + (1 - self.alpha) * self.slope
    
    # Re-anchor baseline to latest observation for numerical stability
    self.baseline = new_health - self.slope * hours_elapsed
    
    Console.debug(f"[{self.name}] Online update: error={error:.2f}, new_slope={self.slope:.4f}")
```

**Implementation Notes:**

- **Why re-anchor baseline?**
  - Prevents accumulation of numerical drift
  - Keeps predictions anchored to most recent observation
  
- **When to refit vs update online?**
  - Use online updates for small corrections
  - Trigger full refit if:
    - Regime shift detected
    - Residual std increases by 2x
    - More than 100 online updates since last fit

---

#### Method: get_coefficients() / set_coefficients()

**Purpose:** Serialize/deserialize for state persistence

```python
def get_coefficients(self) -> dict:
    """Return all parameters for state saving"""
    return {
        'slope': float(self.slope),
        'baseline': float(self.baseline),
        'residual_std': float(self.residual_std),
        'reference_time': self.reference_time.isoformat() if self.reference_time else None,
        'alpha': float(self.alpha),
        'is_fitted': bool(self.is_fitted),
        'last_fit_time': self.last_fit_time.isoformat() if self.last_fit_time else None,
        'fit_data_count': int(self.fit_data_count)
    }

def set_coefficients(self, coeffs: dict) -> None:
    """Load parameters from saved state"""
    self.slope = float(coeffs['slope'])
    self.baseline = float(coeffs['baseline'])
    self.residual_std = float(coeffs['residual_std'])
    self.reference_time = pd.Timestamp(coeffs['reference_time']) if coeffs['reference_time'] else None
    self.alpha = float(coeffs.get('alpha', 0.1))
    self.is_fitted = bool(coeffs['is_fitted'])
    self.last_fit_time = pd.Timestamp(coeffs['last_fit_time']) if coeffs.get('last_fit_time') else None
    self.fit_data_count = int(coeffs.get('fit_data_count', 0))
```

---

### Alternative Model: ExponentialDecayModel (Optional)

**When to use:**
- Battery degradation
- Chemical processes
- Systems with exponential wear-out

**Model:** `health(t) = h0 * exp(-λt) + offset`

**Implementation sketch:**

```python
class ExponentialDecayModel(BaseDegradationModel):
    """
    Exponential decay: health(t) = h0 * exp(-lambda * t) + offset
    
    Suitable for:
        - Battery State of Health
        - Capacitor aging
        - Chemical catalyst degradation
    """
    
    def fit(self, timestamps, health_values):
        # Transform to log-linear: log(health - offset) = log(h0) - lambda * t
        # Estimate offset as 5th percentile of recent values
        offset = np.percentile(health_values[-20:], 5)
        
        # Log transform
        y_shifted = health_values - offset
        if np.any(y_shifted <= 0):
            Console.warn("Non-positive shifted values, adding floor")
            y_shifted = np.maximum(y_shifted, 0.1)
        
        log_y = np.log(y_shifted)
        
        # Linear regression on log scale
        hours = (timestamps - timestamps[0]).total_seconds().values / 3600.0
        result = linregress(hours, log_y)
        
        self.lambda_ = -result.slope  # Decay rate
        self.h0 = np.exp(result.intercept)
        self.offset = offset
        
        # ...residual calculation...
        
    def predict(self, current_time, hours_ahead, num_points=100):
        # Exponential projection
        hours_since_ref = (current_time - self.reference_time).total_seconds() / 3600.0
        future_hours = hours_since_ref + np.linspace(0, hours_ahead, num_points)
        
        forecast_mean = self.h0 * np.exp(-self.lambda_ * future_hours) + self.offset
        
        # ...uncertainty calculation...
```

---

### Testing Requirements for degradation_model.py

**Unit Tests:**
1. `test_linear_fit_simple()` - Fit to synthetic linear data
2. `test_linear_predict_horizon()` - Forecast extends correctly
3. `test_linear_uncertainty_growth()` - std increases with time
4. `test_online_update_convergence()` - EMA updates converge to new slope
5. `test_coefficients_roundtrip()` - Save/load preserves state
6. `test_fit_insufficient_data()` - Handles <10 samples gracefully
7. `test_extreme_slope_clamping()` - Slope clamping works

**Integration Tests:**
1. Fit to real equipment data with known failure
2. Verify RUL predicted within ±20% of actual
3. Compare online updates vs full refit accuracy (should be ~95% similar)

---

## 5. Module 3: failure_probability.py

### Purpose
Convert health forecast into failure probability using survival analysis concepts.

### Key Concepts

**Failure Threshold:** Health value below which equipment is considered "failed" (e.g., 70%)

**Cumulative Distribution Function (CDF):** F(t) = P(failure by time t)

**Survival Function:** S(t) = 1 - F(t) = P(survival past time t)

**Hazard Rate:** λ(t) = f(t)/S(t) = instantaneous failure rate given survival to t

---

### Function: health_to_failure_probability()

**Signature:**
```python
def health_to_failure_probability(
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    failure_threshold: float = 70.0
) -> np.ndarray:
    """
    Convert health forecast to cumulative failure probability.
    
    Uses Gaussian assumption:
        P(failure at t) = P(health(t) < threshold)
                        = Φ((threshold - mean(t)) / std(t))
    
    where Φ is the standard normal CDF.
    
    Args:
        forecast_mean: Array of predicted health values
        forecast_std: Array of prediction uncertainties (±1 std)
        failure_threshold: Health level considered as failure
    
    Returns:
        Array of cumulative failure probabilities in [0, 1]
    
    Implementation Notes:
        - Use scipy.stats.norm.cdf() for Gaussian CDF
        - Clip std to min of 0.1 to avoid division by zero
        - Result should be monotonically increasing (add check)
    """
```
**Implementation:**

```python
from scipy.stats import norm

def health_to_failure_probability(
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    failure_threshold: float = 70.0
) -> np.ndarray:
    
    # Guard against zero/negative std
    forecast_std = np.maximum(forecast_std, 0.1)
    
    # Standardized distance to threshold
    z_scores = (failure_threshold - forecast_mean) / forecast_std
    
    # Cumulative probability of being below threshold
    failure_probs = norm.cdf(z_scores)
    
    # Clip to valid probability range
    failure_probs = np.clip(failure_probs, 0.0, 1.0)
    
    # Enforce monotonicity (probability can't decrease over time)
    failure_probs = np.maximum.accumulate(failure_probs)
    
    return failure_probs
```

**Why Gaussian assumption?**
- Simple, interpretable
- Works well for gradual degradation
- Can be replaced with empirical distributions later if needed

**When assumption breaks:**
- Sudden failures (use change-point detection)
- Multi-modal degradation (rare in practice)
- Solution: Add warning if residuals fail normality test

---

### Function: compute_survival_curve()

**Signature:**
```python
def compute_survival_curve(failure_probs: np.ndarray) -> np.ndarray:
    """
    Convert failure CDF to survival function.
    
    S(t) = 1 - F(t)
    
    Args:
        failure_probs: Cumulative failure probabilities
    
    Returns:
        Survival probabilities (1.0 → 0.0 over time)
    """
    return 1.0 - failure_probs
```

---

### Function: compute_hazard_rate()

**Signature:**
```python
def compute_hazard_rate(
    failure_probs: np.ndarray,
    dt_hours: float
) -> np.ndarray:
    """
    Compute discrete hazard rate from failure CDF.
    
    Hazard rate λ(t) represents the instantaneous failure rate
    given that the equipment has survived up to time t.
    
    Discrete approximation:
        λ[i] = (F[i] - F[i-1]) / ((1 - F[i-1]) * dt)
    
    Args:
        failure_probs: Cumulative failure probabilities F(t)
        dt_hours: Time step size in hours
    
    Returns:
        Hazard rates (failures per hour)
    
    Physical Interpretation:
        - λ = 0.01/hour means 1% chance of failure in next hour (given survival so far)
        - Increasing λ → wear-out phase
        - Constant λ → random failures
        - Decreasing λ → infant mortality
    """
```

**Implementation:**

```python
def compute_hazard_rate(
    failure_probs: np.ndarray,
    dt_hours: float
) -> np.ndarray:
    
    n = len(failure_probs)
    if n == 0:
        return np.array([])
    
    # Ensure monotonicity
    F = np.maximum.accumulate(failure_probs)
    
    # Initialize hazard array
    hazard = np.zeros(n)
    
    # First point: hazard over first interval
    if F[0] < 1.0:
        # λ[0] = -log(1 - F[0]) / dt (derived from exponential assumption)
        hazard[0] = -np.log(max(1e-9, 1.0 - F[0])) / dt_hours
    else:
        hazard[0] = 10.0  # Sentinel for certain failure
    
    # Subsequent points: incremental hazard
    for i in range(1, n):
        dF = F[i] - F[i-1]  # Probability increment
        S_prev = 1.0 - F[i-1]  # Survival to previous time
        
        if S_prev > 1e-9 and dt_hours > 1e-6:
            hazard[i] = dF / (S_prev * dt_hours)
        else:
            hazard[i] = 10.0  # Saturated (almost certain failure)
    
    # Clip to reasonable range
    hazard = np.clip(hazard, 0.0, 10.0)
    
    return hazard
```

**Common Mistakes to Avoid:**

1. **Don't use:** `hazard = dF / dt` ← This ignores survival probability
2. **Don't use:** `hazard = F / (1 - F)` ← This is odds ratio, not hazard
3. **Do use:** `hazard = dF / (S_prev * dt)` ← Correct conditional probability

---

### Function: mean_time_to_failure()

**Signature:**
```python
def mean_time_to_failure(
    survival_probs: np.ndarray,
    dt_hours: float
) -> float:
    """
    Compute Mean Time To Failure (MTTF) from survival curve.
    
    MTTF = integral of S(t) dt from 0 to infinity
         ≈ sum(S[i] * dt) for discrete approximation
    
    Args:
        survival_probs: S(t) values
        dt_hours: Time step
    
    Returns:
        MTTF in hours
    
    Note: This is the expected remaining life if failure follows
          the predicted distribution.
    """
    
    # Trapezoidal integration
    mttf = np.trapz(survival_probs, dx=dt_hours)
    return float(mttf)
```

---

### Testing Requirements for failure_probability.py

**Unit Tests:**
1. `test_health_to_failure_prob_threshold_crossing()` - Probability increases as health drops below threshold
2. `test_monotonicity_enforcement()` - Output is always non-decreasing
3. `test_hazard_rate_calculation()` - Hazard computed correctly
4. `test_survival_curve_bounds()` - S(t) in [0, 1] and decreasing
5. `test_mttf_calculation()` - MTTF matches analytical solution for exponential

**Edge Cases:**
1. Constant health (no degradation) → failure_prob stays near 0
2. Rapid degradation → failure_prob jumps to 1 quickly
3. High uncertainty → failure_prob increases smoothly (less sharp)

---

## 6. Module 4: rul_estimator.py

### Purpose
Estimate Remaining Useful Life (RUL) from health forecast and failure probability.

### Key Concept
RUL = Time until equipment health crosses failure threshold

### Class: RULEstimator

```python
# File: core/rul_estimator.py

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import pandas as pd

@dataclass
class RULResult:
    """Structured RUL estimation result"""
    median_hours: float
    mean_hours: float
    p10_hours: float  # Optimistic (90% chance failure later than this)
    p90_hours: float  # Pessimistic (90% chance failure earlier than this)
    confidence: float  # [0, 1] - how confident in this estimate
    method: str  # "deterministic" or "monte_carlo"
    
    def get_recommendation(self) -> str:
        """Return maintenance urgency level"""
        if self.median_hours < 12:
            return "URGENT"
        elif self.median_hours < 24:
            return "HIGH"
        elif self.median_hours < 72:
            return "MEDIUM"
        elif self.median_hours < 168:
            return "LOW"
        else:
            return "NORMAL"

class RULEstimator:
    """
    Estimate Remaining Useful Life from forecast.
    
    Supports two modes:
        1. Deterministic: Simple threshold crossing
        2. Probabilistic: Monte Carlo uncertainty quantification
    """
```

---

### Method: compute_rul_deterministic()

**Purpose:** Find when mean forecast crosses failure threshold (fast, simple)

**Signature:**
```python
def compute_rul_deterministic(
    self,
    forecast_timestamps: pd.DatetimeIndex,
    forecast_mean: np.ndarray,
    failure_threshold: float = 70.0,
    current_time: Optional[pd.Timestamp] = None
) -> float:
    """
    Compute RUL as first crossing of failure threshold.
    
    Args:
        forecast_timestamps: Timestamps of forecast
        forecast_mean: Predicted health values
        failure_threshold: Health level considered failure
        current_time: Reference time (default: first timestamp)
    
    Returns:
        RUL in hours (or inf if no crossing within forecast horizon)
    
    Method:
        Find first index where forecast_mean < failure_threshold
        Return time difference from current_time
    """
```

**Implementation:**

```python
def compute_rul_deterministic(
    self,
    forecast_timestamps: pd.DatetimeIndex,
    forecast_mean: np.ndarray,
    failure_threshold: float = 70.0,
    current_time: Optional[pd.Timestamp] = None
) -> float:
    
    if current_time is None:
        current_time = forecast_timestamps[0]
    
    # Find first crossing
    crossing_indices = np.where(forecast_mean < failure_threshold)[0]
    
    if len(crossing_indices) == 0:
        # No crossing within forecast horizon
        Console.info("[RUL] No threshold crossing detected within forecast horizon")
        return float('inf')
    
    # First crossing time
    crossing_idx = crossing_indices[0]
    crossing_time = forecast_timestamps[crossing_idx]
    
    # RUL in hours
    rul_hours = (crossing_time - current_time).total_seconds() / 3600.0
    
    return float(rul_hours)
```

**Pros:**
- Fast (O(n) search)
- Deterministic (reproducible)
- Easy to explain

**Cons:**
- Ignores uncertainty
- Single point estimate

---

### Method: compute_rul_probabilistic()

**Purpose:** Use Monte Carlo to get full RUL distribution (more accurate)

**Signature:**
```python
def compute_rul_probabilistic(
    self,
    forecast_timestamps: pd.DatetimeIndex,
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    failure_threshold: float = 70.0,
    current_time: Optional[pd.Timestamp] = None,
    n_simulations: int = 1000
) -> RULResult:
    """
    Compute RUL distribution using Monte Carlo simulation.
    
    Method:
        1. Sample N possible future trajectories from (mean, std)
        2. For each trajectory, find threshold crossing time
        3. Compute statistics: median, P10, P90, mean
    
    Args:
        forecast_timestamps: Timestamps of forecast
        forecast_mean: Predicted health values
        forecast_std: Prediction uncertainties
        failure_threshold: Failure health level
        current_time: Reference time
        n_simulations: Number of Monte Carlo samples
    
    Returns:
        RULResult with full distribution statistics
    
    Advantages over deterministic:
        - Quantifies uncertainty (P10/P90 bounds)
        - Accounts for forecast uncertainty
        - More realistic for decision-making
    """
```

**Implementation:**

```python
def compute_rul_probabilistic(
    self,
    forecast_timestamps: pd.DatetimeIndex,
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    failure_threshold: float = 70.0,
    current_time: Optional[pd.Timestamp] = None,
    n_simulations: int = 1000
) -> RULResult:
    
    if current_time is None:
        current_time = forecast_timestamps[0]
    
    n_steps = len(forecast_mean)
    rul_samples = []
    
    # Monte Carlo simulation
    for _ in range(n_simulations):
        # Sample one possible trajectory
        trajectory = np.random.normal(forecast_mean, forecast_std)
        
        # Find first crossing
        crossings = np.where(trajectory < failure_threshold)[0]
        
        if len(crossings) > 0:
            crossing_idx = crossings[0]
            crossing_time = forecast_timestamps[crossing_idx]
            rul_hours = (crossing_time - current_time).total_seconds() / 3600.0
        else:
            # Censored (no crossing in forecast horizon)
            # Use horizon + buffer as conservative estimate
            rul_hours = (forecast_timestamps[-1] - current_time).total_seconds() / 3600.0 + 24
        
        rul_samples.append(rul_hours)
    
    # Compute statistics
    rul_array = np.array(rul_samples)
    
    result = RULResult(
        median_hours=float(np.median(rul_array)),
        mean_hours=float(np.mean(rul_array)),
        p10_hours=float(np.percentile(rul_array, 10)),
        p90_hours=float(np.percentile(rul_array, 90)),
        confidence=self._compute_confidence(rul_array, forecast_std),
        method="monte_carlo"
    )
    
    Console.info(
        f"[RUL] Probabilistic: "
        f"median={result.median_hours:.1f}h, "
        f"P10={result.p10_hours:.1f}h, "
        f"P90={result.p90_hours:.1f}h"
    )
    
    return result
```

---

### Method: _compute_confidence()

**Purpose:** Quantify confidence in RUL estimate

**Factors:**
1. **Narrow RUL distribution** → High confidence (P90 - P10 is small)
2. **Low forecast uncertainty** → High confidence (std is small)
3. **Long RUL** → Lower confidence (more time for things to change)

**Implementation:**

```python
def _compute_confidence(
    self,
    rul_samples: np.ndarray,
    forecast_std: np.ndarray
) -> float:
    """
    Compute confidence score [0, 1] for RUL estimate.
    
    Factors:
        - Width of RUL distribution (P90 - P10)
        - Average forecast uncertainty
        - RUL magnitude (longer = less confident)
    
    Returns:
        Confidence in [0.3, 1.0] range
    """
    
    # Factor 1: RUL distribution width
    p10 = np.percentile(rul_samples, 10)
    p90 = np.percentile(rul_samples, 90)
    median = np.median(rul_samples)
    
    if median > 0:
        width_ratio = (p90 - p10) / median
        width_score = 1.0 / (1.0 + width_ratio)  # Narrower → higher score
    else:
        width_score = 0.5
    
    # Factor 2: Average forecast uncertainty
    avg_std = np.mean(forecast_std)
    uncertainty_score = 1.0 / (1.0 + avg_std / 10.0)  # Lower std → higher score
    
    # Factor 3: RUL magnitude penalty
    if median < 24:
        rul_score = 1.0  # High confidence in near-term failures
    elif median < 168:
        rul_score = 0.8
    else:
        rul_score = 0.6  # Lower confidence in long-term predictions
    
    # Weighted combination
    confidence = 0.4 * width_score + 0.3 * uncertainty_score + 0.3 * rul_score
    
    # Clamp to [0.3, 1.0] - never completely unconfident
    confidence = np.clip(confidence, 0.3, 1.0)
    
    return float(confidence)
```

---

### Testing Requirements for rul_estimator.py

**Unit Tests:**
1. `test_deterministic_rul_crossing()` - Finds correct crossing time
2. `test_deterministic_rul_no_crossing()` - Returns inf when no crossing
3. `test_probabilistic_rul_distribution()` - P10 < median < P90
4. `test_confidence_narrow_distribution()` - High confidence when tight bounds
5. `test_recommendation_urgency()` - Correct urgency levels

**Integration Tests:**
1. Compare deterministic vs probabilistic on real data (should be within 20%)
2. Verify confidence correlates with actual accuracy (using historical failures)

---

## 7. Module 5: state_manager.py

### Purpose
Persist forecasting state between batches to enable incremental updates.

### Why State Management Matters
- Avoid re-fitting models from scratch every batch (expensive)
- Enable smooth forecast transitions (blend old/new forecasts)
- Track model performance over time
- Support recovery after system restart

---

### Class: ForecastingState

```python
# File: core/state_manager.py

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime
import json
import pandas as pd

@dataclass
class ForecastingState:
    """
    Complete forecasting state for one equipment.
    
    Persisted in SQL table: ACM_ForecastingState
    """
    
    # Identity
    equip_id: int
    
    # Model state
    model_type: str  # "LinearTrend", "ExponentialDecay", etc.
    model_coefficients: Dict[str, Any]  # From degradation_model.get_coefficients()
    
    # Last forecast (for blending)
    last_forecast_timestamps: Optional[str] = None  # JSON array of ISO timestamps
    last_forecast_mean: Optional[str] = None  # JSON array of floats
    last_forecast_std: Optional[str] = None  # JSON array of floats
    
    # Performance tracking
    recent_mae: float = 0.0  # Mean Absolute Error on last 10 predictions
    recent_rmse: float = 0.0  # Root Mean Squared Error
    prediction_count: int = 0  # How many forecasts made
    
    # Retraining logic
    last_retrain_time: Optional[datetime] = None
    retrain_reason: str = ""  # "initial", "scheduled", "drift_detected", "performance_drop"
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
```

---

### Function: save_state()

**Signature:**
```python
def save_state(
    state: ForecastingState,
    sql_client
) -> None:
    """
    Save forecasting state to SQL.
    
    Table: ACM_ForecastingState
    Schema:
        EquipID INT PRIMARY KEY,
        ModelType VARCHAR(50),
        ModelCoefficients NVARCHAR(MAX),  -- JSON
        LastForecastTimestamps NVARCHAR(MAX),  -- JSON
        LastForecastMean NVARCHAR(MAX),  -- JSON
        LastForecastStd NVARCHAR(MAX),  -- JSON
        RecentMAE FLOAT,
        RecentRMSE FLOAT,
        PredictionCount INT,
        LastRetrainTime DATETIME,
        RetrainReason VARCHAR(100),
        CreatedAt DATETIME,
        UpdatedAt DATETIME
    
    Uses MERGE (upsert) pattern to handle create/update.
    """
```

**Implementation:**

```python
def save_state(state: ForecastingState, sql_client) -> None:
    
    # Serialize model coefficients
    model_coeffs_json = json.dumps(state.model_coefficients)
    
    # Update timestamp
    state.updated_at = datetime.now()
    
    # Prepare data dict
    data = {
        'EquipID': state.equip_id,
        'ModelType': state.model_type,
        'ModelCoefficients': model_coeffs_json,
        'LastForecastTimestamps': state.last_forecast_timestamps,
        'LastForecastMean': state.last_forecast_mean,
        'LastForecastStd': state.last_forecast_std,
        'RecentMAE': state.recent_mae,
        'RecentRMSE': state.recent_rmse,
        'PredictionCount': state.prediction_count,
        'LastRetrainTime': state.last_retrain_time,
        'RetrainReason': state.retrain_reason,
        'CreatedAt': state.created_at,
        'UpdatedAt': state.updated_at
    }
    
    # Execute MERGE query
    cursor = sql_client.cursor()
    try:
        cursor.execute("""
            MERGE dbo.ACM_ForecastingState AS target
            USING (SELECT ? AS EquipID) AS source
            ON target.EquipID = source.EquipID
            WHEN MATCHED THEN
                UPDATE SET
                    ModelType = ?,
                    ModelCoefficients = ?,
                    LastForecastTimestamps = ?,
                    LastForecastMean = ?,
                    LastForecastStd = ?,
                    RecentMAE = ?,
                    RecentRMSE = ?,
                    PredictionCount = ?,
                    LastRetrainTime = ?,
                    RetrainReason = ?,
                    UpdatedAt = ?
            WHEN NOT MATCHED THEN
                INSERT (EquipID, ModelType, ModelCoefficients, 
                       LastForecastTimestamps, LastForecastMean, LastForecastStd,
                       RecentMAE, RecentRMSE, PredictionCount,
                       LastRetrainTime, RetrainReason, CreatedAt, UpdatedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            # USING clause
            data['EquipID'],
            # UPDATE clause
            data['ModelType'], data['ModelCoefficients'],
            data['LastForecastTimestamps'], data['LastForecastMean'], data['LastForecastStd'],
            data['RecentMAE'], data['RecentRMSE'], data['PredictionCount'],
            data['LastRetrainTime'], data['RetrainReason'], data['UpdatedAt'],
            # INSERT clause
            data['EquipID'], data['ModelType'], data['ModelCoefficients'],
            data['LastForecastTimestamps'], data['LastForecastMean'], data['LastForecastStd'],
            data['RecentMAE'], data['RecentRMSE'], data['PredictionCount'],
            data['LastRetrainTime'], data['RetrainReason'], 
            data['CreatedAt'], data['UpdatedAt']
        ))
        
        if not sql_client.conn.autocommit:
            sql_client.conn.commit()
        
        Console.info(f"[StateManager] Saved state for EquipID={state.equip_id}")
        
    except Exception as e:
        Console.error(f"[StateManager] Failed to save state: {e}")
        raise
    finally:
        cursor.close()
```

---

### Function: load_state()

**Signature:**
```python
def load_state(
    equip_id: int,
    sql_client
) -> Optional[ForecastingState]:
    """
    Load forecasting state from SQL.
    
    Returns:
        ForecastingState if found, None otherwise
    """
```

**Implementation:**

```python
def load_state(equip_id: int, sql_client) -> Optional[ForecastingState]:
    
    cursor = sql_client.cursor()
    try:
        cursor.execute("""
            SELECT ModelType, ModelCoefficients,
                   LastForecastTimestamps, LastForecastMean, LastForecastStd,
                   RecentMAE, RecentRMSE, PredictionCount,
                   LastRetrainTime, RetrainReason,
                   CreatedAt, UpdatedAt
            FROM dbo.ACM_ForecastingState
            WHERE EquipID = ?
        """, (equip_id,))
        
        row = cursor.fetchone()
        
        if row is None:
            Console.info(f"[StateManager] No state found for EquipID={equip_id}")
            return None
        
        # Deserialize
        model_coeffs = json.loads(row.ModelCoefficients) if row.ModelCoefficients else {}
        
        state = ForecastingState(
            equip_id=equip_id,
            model_type=row.ModelType,
            model_coefficients=model_coeffs,
            last_forecast_timestamps=row.LastForecastTimestamps,
            last_forecast_mean=row.LastForecastMean,
            last_forecast_std=row.LastForecastStd,
            recent_mae=float(row.RecentMAE) if row.RecentMAE else 0.0,
            recent_rmse=float(row.RecentRMSE) if row.RecentRMSE else 0.0,
            prediction_count=int(row.PredictionCount) if row.PredictionCount else 0,
            last_retrain_time=row.LastRetrainTime,
            retrain_reason=row.RetrainReason or "",
            created_at=row.CreatedAt,
            updated_at=row.UpdatedAt
        )
        
        Console.info(f"[StateManager] Loaded state for EquipID={equip_id}, last_update={state.updated_at}")
        return state
        
    except Exception as e:
        Console.error(f"[StateManager] Failed to load state: {e}")
        return None
    finally:
        cursor.close()
```

---

### Function: should_retrain()

**Purpose:** Decide if model needs full retraining vs incremental update

**Signature:**
```python
def should_retrain(
    state: Optional[ForecastingState],
    health_timeline: 'HealthTimeline',
    max_hours_since_retrain: float = 168.0,  # 1 week
    performance_threshold: float = 2.0  # MAE multiplier
) -> tuple[bool, str]:
    """
    Decide if model should be retrained.
    
    Triggers:
        1. No previous state (initial training)
        2. Time since last retrain > max_hours_since_retrain
        3. Recent MAE > performance_threshold * historical MAE
        4. Regime shift detected in health data
    
    Returns:
        (should_retrain, reason)
    """
```

**Implementation:**

```python
def should_retrain(
    state: Optional[ForecastingState],
    health_timeline: 'HealthTimeline',
    max_hours_since_retrain: float = 168.0,
    performance_threshold: float = 2.0
) -> tuple[bool, str]:
    
    # Trigger 1: No previous state
    if state is None:
        return True, "initial_training"
    
    # Trigger 2: Scheduled retrain (time-based)
    if state.last_retrain_time is not None:
        hours_since_retrain = (datetime.now() - state.last_retrain_time).total_seconds() / 3600.0
        if hours_since_retrain > max_hours_since_retrain:
            return True, f"scheduled (last retrain {hours_since_retrain:.0f}h ago)"
    
    # Trigger 3: Performance degradation
    if state.prediction_count > 10 and state.recent_mae > 0:
        # Compare recent MAE to baseline (use first 10 predictions as baseline)
        baseline_mae = state.recent_mae / max(1.0, state.prediction_count / 10)
        if state.recent_mae > performance_threshold * baseline_mae:
            return True, f"performance_drop (MAE {state.recent_mae:.1f} > {performance_threshold}x baseline)"
    
    # Trigger 4: Regime shift
    if health_timeline.detect_regime_shift():
        return True, "regime_shift_detected"
    
    # No retrain needed
    return False, "model_stable"
```

---

### Function: blend_forecasts()

**Purpose:** Smooth transition between old and new forecasts (prevents jumps)

**Signature:**
```python
def blend_forecasts(
    old_forecast_mean: np.ndarray,
    old_forecast_std: np.ndarray,
    new_forecast_mean: np.ndarray,
    new_forecast_std: np.ndarray,
    alpha: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exponentially blend old and new forecasts.
    
    Formula:
        blended = alpha * new + (1 - alpha) * old
    
    Args:
        old_forecast_mean: Previous forecast
        old_forecast_std: Previous uncertainty
        new_forecast_mean: Current forecast
        new_forecast_std: Current uncertainty
        alpha: Blending weight (0=keep old, 1=use new)
    
    Returns:
        (blended_mean, blended_std)
    
    Why blend?
        - Prevents forecast "jumps" when new data arrives
        - Smoother user experience
        - Reduces false alarms from noise
    
    When NOT to blend?
        - After regime shift (alpha=1.0, full trust new forecast)
        - Initial forecast (no old forecast exists)
    """
    
    # Align lengths (use shorter)
    n = min(len(old_forecast_mean), len(new_forecast_mean))
    old_mean = old_forecast_mean[:n]
    old_std = old_forecast_std[:n]
    new_mean = new_forecast_mean[:n]
    new_std = new_forecast_std[:n]
    
    # Blend
    blended_mean = alpha * new_mean + (1 - alpha) * old_mean
    blended_std = alpha * new_std + (1 - alpha) * old_std
    
    return blended_mean, blended_std
```

---

### Testing Requirements for state_manager.py

**Unit Tests:**
1. `test_save_load_roundtrip()` - State survives save/load
2. `test_should_retrain_initial()` - Triggers on first run
3. `test_should_retrain_scheduled()` - Triggers after time threshold
4. `test_should_retrain_performance()` - Triggers on MAE increase
5. `test_blend_forecasts_alpha()` - Blending weights work correctly

**Integration Tests:**
1. Simulate 100 batches with state persistence
2. Verify forecasts remain continuous (no jumps)
3. Verify retrains triggered at correct times

---

## 8. Module 6: forecast_engine.py

### Purpose
Main orchestrator that ties all modules together for batch-by-batch forecasting.

### Class: ForecastEngine

```python
# File: core/forecast_engine.py

from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from core.health_tracker import HealthTimeline, HealthQuality
from core.degradation_model import LinearTrendModel, ForecastResult
from core.failure_probability import (
    health_to_failure_probability,
    compute_survival_curve,
    compute_hazard_rate
)
from core.rul_estimator import RULEstimator, RULResult
from core.state_manager import (
    ForecastingState,
    save_state,
    load_state,
    should_retrain,
    blend_forecasts
    )

from utils.logger import Console
class ForecastEngine:
"""
Main forecasting engine coordinating all components.
Usage:
    engine = ForecastEngine(sql_client, equip_id)
    result = engine.run_forecast(new_batch_data)
"""

def __init__(
    self,
    sql_client,
    equip_id: int,
    config: Optional[Dict[str, Any]] = None
):
    self.sql_client = sql_client
    self.equip_id = equip_id
    
    # Load configuration
    self.config = config or {}
    self.failure_threshold = float(self.config.get('failure_threshold', 70.0))
    self.forecast_hours = float(self.config.get('forecast_hours', 168.0))
    self.window_hours = float(self.config.get('window_hours', 168.0))
    self.blend_alpha = float(self.config.get('blend_alpha', 0.3))
    
    # Initialize components
    self.health_tracker = HealthTimeline(equip_id, window_hours=self.window_hours)
    self.rul_estimator = RULEstimator()
    
    Console.info(f"[ForecastEngine] Initialized for EquipID={equip_id}")

---

### Method: run_forecast()

**Purpose:** Main entry point for batch-by-batch forecasting

**Signature:**
```python
def run_forecast(
    self,
    run_id: str,
    new_batch_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Run complete forecasting pipeline for one batch.
    
    Args:
        run_id: Unique identifier for this forecast run
        new_batch_data: Optional new data to append (if None, just reforecast existing)
    
    Returns:
        {
            'health_forecast': pd.DataFrame,  # Timestamp, HealthIndex, CI_Lower, CI_Upper
            'failure_forecast': pd.DataFrame,  # Timestamp, FailureProb, Survival, Hazard
            'rul_summary': pd.DataFrame,  # Single row with RUL statistics
            'metrics': dict,  # Performance metrics
            'data_quality': str  # Quality flag
        }
    
    Pipeline Steps:
        1. Load previous state
        2. Update health timeline
        3. Quality check
        4. Decide: retrain or update?
        5. Generate forecast
        6. Blend with previous (if applicable)
        7. Compute failure probability
        8. Estimate RUL
        9. Save outputs to SQL
        10. Update and save state
    """
```

**Implementation (Step-by-Step):**
```python
def run_forecast(
    self,
    run_id: str,
    new_batch_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    
    Console.info("=" * 80)
    Console.info(f"[ForecastEngine] Starting forecast for EquipID={self.equip_id}, RunID={run_id}")
    Console.info("=" * 80)
    
    # ========================================================================
    # STEP 1: Load Previous State
    # ========================================================================
    previous_state = load_state(self.equip_id, self.sql_client)
    
    if previous_state:
        Console.info(f"[ForecastEngine] Loaded previous state (last update: {previous_state.updated_at})")
    else:
        Console.info("[ForecastEngine] No previous state found (initial run)")
    
    # ========================================================================
    # STEP 2: Update Health Timeline
    # ========================================================================
    # Load historical data
    self.health_tracker.load_from_sql(
        self.sql_client,
        lookback_hours=self.window_hours
    )
    
    # Append new batch if provided
    if new_batch_data is not None and not new_batch_data.empty:
        Console.info(f"[ForecastEngine] Appending {len(new_batch_data)} new observations")
        self.health_tracker.append_batch(new_batch_data)
    
    # Get current data
    health_data = self.health_tracker.get_sliding_window()
    
    if health_data.empty or len(health_data) < 20:
        Console.error("[ForecastEngine] Insufficient health data")
        return self._return_default_forecast(run_id, "MISSING")
    
    # ========================================================================
    # STEP 3: Quality Check
    # ========================================================================
    quality_result = self.health_tracker.quality_check()
    Console.info(f"[ForecastEngine] Data quality: {quality_result.flag} - {quality_result.message}")
    
    if quality_result.flag in ["MISSING", "FLAT"]:
        Console.warn("[ForecastEngine] Data quality insufficient for forecasting")
        return self._return_default_forecast(run_id, quality_result.flag)
    
    # ========================================================================
    # STEP 4: Decide - Retrain or Update?
    # ========================================================================
    needs_retrain, retrain_reason = should_retrain(
        previous_state,
        self.health_tracker,
        max_hours_since_retrain=168.0,  # 1 week
        performance_threshold=2.0
    )
    
    Console.info(f"[ForecastEngine] Retrain decision: {needs_retrain} ({retrain_reason})")
    
    # ========================================================================
    # STEP 5: Generate Forecast
    # ========================================================================
    current_time = health_data['Timestamp'].iloc[-1]
    
    if needs_retrain or previous_state is None:
        # Full model retraining
        Console.info("[ForecastEngine] Retraining model from scratch...")
        
        model = LinearTrendModel(alpha=0.1)
        success = model.fit(
            timestamps=health_data['Timestamp'],
            health_values=health_data['HealthIndex'].values
        )
        
        if not success:
            Console.error("[ForecastEngine] Model fit failed")
            return self._return_default_forecast(run_id, quality_result.flag)
        
        last_retrain_time = datetime.now()
        
    else:
        # Incremental update
        Console.info("[ForecastEngine] Updating model incrementally...")
        
        model = LinearTrendModel()
        model.set_coefficients(previous_state.model_coefficients)
        
        # Online update with latest observations (last 10)
        recent_data = health_data.tail(10)
        for idx, row in recent_data.iterrows():
            model.update_online(row['Timestamp'], row['HealthIndex'])
        
        last_retrain_time = previous_state.last_retrain_time
    
    # Generate forecast
    forecast = model.predict(
        current_time=current_time,
        hours_ahead=self.forecast_hours,
        num_points=100
    )
    
    Console.info(f"[ForecastEngine] Generated {len(forecast.timestamps)} forecast points")
    
    # ========================================================================
    # STEP 6: Blend with Previous Forecast (if applicable)
    # ========================================================================
    if previous_state and previous_state.last_forecast_mean and not needs_retrain:
        Console.info(f"[ForecastEngine] Blending with previous forecast (alpha={self.blend_alpha})")
        
        # Deserialize previous forecast
        old_mean = np.array(json.loads(previous_state.last_forecast_mean))
        old_std = np.array(json.loads(previous_state.last_forecast_std))
        
        # Blend
        forecast.mean, forecast.std = blend_forecasts(
            old_forecast_mean=old_mean,
            old_forecast_std=old_std,
            new_forecast_mean=forecast.mean,
            new_forecast_std=forecast.std,
            alpha=self.blend_alpha if quality_result.flag == "OK" else 1.0  # Full trust if data quality poor
        )
        
        Console.info("[ForecastEngine] Forecast blending complete")
    
    # ========================================================================
    # STEP 7: Compute Failure Probability
    # ========================================================================
    failure_probs = health_to_failure_probability(
        forecast_mean=forecast.mean,
        forecast_std=forecast.std,
        failure_threshold=self.failure_threshold
    )
    
    survival_probs = compute_survival_curve(failure_probs)
    
    # Compute hazard rate
    dt_hours = self.forecast_hours / len(forecast.timestamps)
    hazard_rates = compute_hazard_rate(failure_probs, dt_hours)
    
    Console.info(f"[ForecastEngine] Max failure probability: {failure_probs.max()*100:.1f}%")
    
    # ========================================================================
    # STEP 8: Estimate RUL
    # ========================================================================
    rul_result = self.rul_estimator.compute_rul_probabilistic(
        forecast_timestamps=forecast.timestamps,
        forecast_mean=forecast.mean,
        forecast_std=forecast.std,
        failure_threshold=self.failure_threshold,
        current_time=current_time,
        n_simulations=1000
    )
    
    Console.info(
        f"[ForecastEngine] RUL: {rul_result.median_hours:.1f}h "
        f"(P10={rul_result.p10_hours:.1f}, P90={rul_result.p90_hours:.1f}), "
        f"Recommendation: {rul_result.get_recommendation()}"
    )
    
    # ========================================================================
    # STEP 9: Build Output DataFrames
    # ========================================================================
    
    # Health forecast table
    health_forecast_df = pd.DataFrame({
        'RunID': run_id,
        'EquipID': self.equip_id,
        'Timestamp': forecast.timestamps,
        'HealthIndex': forecast.mean,
        'CI_Lower': forecast.mean - 1.96 * forecast.std,
        'CI_Upper': forecast.mean + 1.96 * forecast.std,
        'ForecastStd': forecast.std
    })
    
    # Failure forecast table
    failure_forecast_df = pd.DataFrame({
        'RunID': run_id,
        'EquipID': self.equip_id,
        'Timestamp': forecast.timestamps,
        'FailureProb': failure_probs,
        'Survival': survival_probs,
        'HazardRate': hazard_rates,
        'ThresholdUsed': self.failure_threshold
    })
    
    # RUL summary table (single row)
    rul_summary_df = pd.DataFrame([{
        'RunID': run_id,
        'EquipID': self.equip_id,
        'RUL_Median_Hours': rul_result.median_hours,
        'RUL_Mean_Hours': rul_result.mean_hours,
        'RUL_P10_Hours': rul_result.p10_hours,
        'RUL_P90_Hours': rul_result.p90_hours,
        'Confidence': rul_result.confidence,
        'Recommendation': rul_result.get_recommendation(),
        'Method': rul_result.method,
        'DataQuality': quality_result.flag,
        'LastUpdate': datetime.now()
    }])
    
    # ========================================================================
    # STEP 10: Write to SQL
    # ========================================================================
    self._write_to_sql(health_forecast_df, 'ACM_HealthForecast_TS')
    self._write_to_sql(failure_forecast_df, 'ACM_FailureForecast_TS')
    self._write_to_sql(rul_summary_df, 'ACM_RUL_Summary')
    
    # ========================================================================
    # STEP 11: Update and Save State
    # ========================================================================
    new_state = ForecastingState(
        equip_id=self.equip_id,
        model_type=model.name,
        model_coefficients=model.get_coefficients(),
        last_forecast_timestamps=json.dumps([ts.isoformat() for ts in forecast.timestamps]),
        last_forecast_mean=json.dumps(forecast.mean.tolist()),
        last_forecast_std=json.dumps(forecast.std.tolist()),
        recent_mae=0.0,  # TODO: Compute from metrics module
        recent_rmse=0.0,
        prediction_count=(previous_state.prediction_count + 1) if previous_state else 1,
        last_retrain_time=last_retrain_time,
        retrain_reason=retrain_reason if needs_retrain else previous_state.retrain_reason if previous_state else ""
    )
    
    save_state(new_state, self.sql_client)
    Console.info("[ForecastEngine] State saved successfully")
    
    # ========================================================================
    # STEP 12: Return Results
    # ========================================================================
    Console.info("=" * 80)
    Console.info("[ForecastEngine] Forecast complete")
    Console.info("=" * 80)
    
    return {
        'health_forecast': health_forecast_df,
        'failure_forecast': failure_forecast_df,
        'rul_summary': rul_summary_df,
        'metrics': {
            'rul_hours': rul_result.median_hours,
            'failure_prob_24h': float(failure_probs[min(24, len(failure_probs)-1)]),
            'confidence': rul_result.confidence,
            'recommendation': rul_result.get_recommendation()
        },
        'data_quality': quality_result.flag
    }
```

---

### Helper Methods
```python
def _return_default_forecast(self, run_id: str, quality_flag: str) -> Dict[str, Any]:
    """Return safe default forecast when data insufficient"""
    
    # Generate flat forecast at threshold
    timestamps = pd.date_range(
        start=datetime.now(),
        periods=10,
        freq='1H'
    )
    
    health_forecast_df = pd.DataFrame({
        'RunID': run_id,
        'EquipID': self.equip_id,
        'Timestamp': timestamps,
        'HealthIndex': [self.failure_threshold] * len(timestamps),
        'CI_Lower': [self.failure_threshold - 10] * len(timestamps),
        'CI_Upper': [self.failure_threshold + 10] * len(timestamps),
        'ForecastStd': [10.0] * len(timestamps)
    })
    
    failure_forecast_df = pd.DataFrame({
        'RunID': run_id,
        'EquipID': self.equip_id,
        'Timestamp': timestamps,
        'FailureProb': [0.0] * len(timestamps),
        'Survival': [1.0] * len(timestamps),
        'HazardRate': [0.0] * len(timestamps),
        'ThresholdUsed': self.failure_threshold
    })
    
    rul_summary_df = pd.DataFrame([{
        'RunID': run_id,
        'EquipID': self.equip_id,
        'RUL_Median_Hours': self.forecast_hours,
        'RUL_Mean_Hours': self.forecast_hours,
        'RUL_P10_Hours': self.forecast_hours * 0.7,
        'RUL_P90_Hours': self.forecast_hours * 1.3,
        'Confidence': 0.3,
        'Recommendation': 'INSUFFICIENT_DATA',
        'Method': 'default',
        'DataQuality': quality_flag,
        'LastUpdate': datetime.now()
    }])
    
    return {
        'health_forecast': health_forecast_df,
        'failure_forecast': failure_forecast_df,
        'rul_summary': rul_summary_df,
        'metrics': {'rul_hours': self.forecast_hours, 'confidence': 0.3},
        'data_quality': quality_flag
    }

def _write_to_sql(self, df: pd.DataFrame, table_name: str) -> None:
    """Write DataFrame to SQL table"""
    try:
        # Add CreatedAt if not present
        if 'CreatedAt' not in df.columns:
            df['CreatedAt'] = datetime.now()
        
        # Bulk insert (implementation depends on SQL client)
        # For pyodbc: use fast_executemany or BULK INSERT
        # For sqlalchemy: use to_sql with method='multi'
        
        Console.info(f"[ForecastEngine] Wrote {len(df)} rows to {table_name}")
    except Exception as e:
        Console.error(f"[ForecastEngine] Failed to write to {table_name}: {e}")
        raise
```

---

## 9. Module 7: sensor_attribution.py

### Purpose
Identify which sensors are driving equipment failure (root cause analysis).

---

### Function: rank_sensors_by_correlation()

**Signature:**
```python
def rank_sensors_by_correlation(
    sensor_data: pd.DataFrame,
    health_index: pd.Series,
    min_correlation: float = 0.3
) -> pd.DataFrame:
    """
    Rank sensors by correlation with health decline.
    
    Args:
        sensor_data: DataFrame with sensor columns (numeric)
        health_index: Health timeline
        min_correlation: Minimum |correlation| to include
    
    Returns:
        DataFrame with columns:
            - SensorName
            - Correlation (negative = sensor increases as health decreases)
            - AbsCorrelation (for ranking)
    
    Sorted by AbsCorrelation descending.
    """
    
    # Align sensor data with health index (common timestamps)
    merged = sensor_data.merge(
        health_index.to_frame('HealthIndex'),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    # Compute correlations
    correlations = []
    for col in sensor_data.columns:
        if col in merged.columns:
            corr = merged[col].corr(merged['HealthIndex'])
            if abs(corr) >= min_correlation:
                correlations.append({
                    'SensorName': col,
                    'Correlation': corr,
                    'AbsCorrelation': abs(corr)
                })
    
    # Sort by absolute correlation
    df = pd.DataFrame(correlations)
    df = df.sort_values('AbsCorrelation', ascending=False)
    
    return df
```

---

### Function: compute_sensor_contributions()

**Signature:**
```python
def compute_sensor_contributions(
    sensor_hotspots: pd.DataFrame,
    rul_hours: float
) -> pd.DataFrame:
    """
    Compute each sensor's contribution to predicted failure.
    
    Args:
        sensor_hotspots: From ACM_SensorHotspots (SensorName, MaxAbsZ, AlertCount)
        rul_hours: Predicted RUL
    
    Returns:
        DataFrame with:
            - SensorName
            - FailureContribution (fraction of total failure signal)
            - ZScoreAtFailure
            - AlertCount
            - FirstAlertHours (how long ago first alert)
    
    Method:
        Contribution = (sensor_z_score / sum_all_z_scores) * alert_count_weight
    """
    
    if sensor_hotspots.empty:
        return pd.DataFrame()
    
    # Normalize Z-scores to contributions
    total_z = sensor_hotspots['MaxAbsZ'].sum()
    if total_z > 0:
        sensor_hotspots['FailureContribution'] = sensor_hotspots['MaxAbsZ'] / total_z
    else:
        sensor_hotspots['FailureContribution'] = 0.0
    
    # Weight by alert count (sensors with sustained high z-scores)
    sensor_hotspots['FailureContribution'] *= (1 + np.log1p(sensor_hotspots['AlertCount']))
    
    # Re-normalize
    total_contrib = sensor_hotspots['FailureContribution'].sum()
    if total_contrib > 0:
        sensor_hotspots['FailureContribution'] /= total_contrib
    
    # Sort by contribution
    sensor_hotspots = sensor_hotspots.sort_values('FailureContribution', ascending=False)
    
    return sensor_hotspots
```

---

## 10. Module 8: metrics.py

### Purpose
Track forecast accuracy over time to enable continuous improvement.

---

### Function: compute_forecast_error()

**Signature:**
```python
def compute_forecast_error(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    horizons: List[int] = [1, 24, 168]
) -> Dict[str, Dict[str, float]]:
    """
    Compute forecast accuracy metrics at different horizons.
    
    Args:
        predicted: Forecast DataFrame (Timestamp, HealthIndex)
        actual: Actual health values (Timestamp, HealthIndex)
        horizons: Forecast horizons to evaluate (hours)
    
    Returns:
        {
            '1h': {'mae': ..., 'rmse': ..., 'mape': ...},
            '24h': {...},
            '168h': {...}
        }
    
    Method:
        - Merge predicted/actual on Timestamp
        - Filter to specific horizons
        - Compute MAE, RMSE, MAPE
    """
    
    # Merge on timestamp
    merged = predicted.merge(
        actual,
        on='Timestamp',
        suffixes=('_pred', '_actual'),
        how='inner'
    )
    
    if merged.empty:
        return {}
    
    results = {}
    
    for horizon_hours in horizons:
        # Filter to this horizon (±1 hour tolerance)
        forecast_start = merged['Timestamp'].min()
        target_time = forecast_start + pd.Timedelta(hours=horizon_hours)
        
        horizon_data = merged[
            (merged['Timestamp'] >= target_time - pd.Timedelta(hours=1)) &
            (merged['Timestamp'] <= target_time + pd.Timedelta(hours=1))
        ]
        
        if horizon_data.empty:
            continue
        
        # Compute metrics
        y_true = horizon_data['HealthIndex_actual'].values
        y_pred = horizon_data['HealthIndex_pred'].values
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        results[f'{horizon_hours}h'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'n_samples': len(horizon_data)
        }
    
    return results
```

---

## 11. SQL Schema Requirements

### Table: ACM_ForecastingState
```sql
CREATE TABLE dbo.ACM_ForecastingState (
    EquipID INT PRIMARY KEY,
    ModelType VARCHAR(50) NOT NULL,
    ModelCoefficients NVARCHAR(MAX),  -- JSON
    LastForecastTimestamps NVARCHAR(MAX),  -- JSON array of ISO timestamps
    LastForecastMean NVARCHAR(MAX),  -- JSON array of floats
    LastForecastStd NVARCHAR(MAX),  -- JSON array of floats
    RecentMAE FLOAT DEFAULT 0.0,
    RecentRMSE FLOAT DEFAULT 0.0,
    PredictionCount INT DEFAULT 0,
    LastRetrainTime DATETIME,
    RetrainReason VARCHAR(100),
    CreatedAt DATETIME DEFAULT GETDATE(),
    UpdatedAt DATETIME DEFAULT GETDATE()
);

CREATE INDEX IX_ForecastingState_UpdatedAt ON dbo.ACM_ForecastingState(UpdatedAt);
```

---

### Table: ACM_HealthForecast_TS
```sql
CREATE TABLE dbo.ACM_HealthForecast_TS (
    RunID VARCHAR(50) NOT NULL,
    EquipID INT NOT NULL,
    Timestamp DATETIME NOT NULL,
    HealthIndex FLOAT NOT NULL,
    CI_Lower FLOAT,
    CI_Upper FLOAT,
    ForecastStd FLOAT,
    CreatedAt DATETIME DEFAULT GETDATE(),
    PRIMARY KEY (RunID, EquipID, Timestamp)
);

CREATE INDEX IX_HealthForecast_EquipID_Timestamp ON dbo.ACM_HealthForecast_TS(EquipID, Timestamp);
```

---

### Table: ACM_FailureForecast_TS
```sql
CREATE TABLE dbo.ACM_FailureForecast_TS (
    RunID VARCHAR(50) NOT NULL,
    EquipID INT NOT NULL,
    Timestamp DATETIME NOT NULL,
    FailureProb FLOAT NOT NULL,
    Survival FLOAT,
    HazardRate FLOAT,
    ThresholdUsed FLOAT,
    CreatedAt DATETIME DEFAULT GETDATE(),
    PRIMARY KEY (RunID, EquipID, Timestamp)
);

CREATE INDEX IX_FailureForecast_EquipID_Timestamp ON dbo.ACM_FailureForecast_TS(EquipID, Timestamp);
```

---

### Table: ACM_RUL_Summary
```sql
CREATE TABLE dbo.ACM_RUL_Summary (
    RunID VARCHAR(50) NOT NULL,
    EquipID INT NOT NULL,
    RUL_Median_Hours FLOAT NOT NULL,
    RUL_Mean_Hours FLOAT,
    RUL_P10_Hours FLOAT,
    RUL_P90_Hours FLOAT,
    Confidence FLOAT,
    Recommendation VARCHAR(50),  -- URGENT, HIGH, MEDIUM, LOW, NORMAL
    Method VARCHAR(50),
    DataQuality VARCHAR(20),
    LastUpdate DATETIME DEFAULT GETDATE(),
    PRIMARY KEY (RunID, EquipID)
);

CREATE INDEX IX_RUL_Summary_EquipID_LastUpdate ON dbo.ACM_RUL_Summary(EquipID, LastUpdate);
```

---

## 12. Testing Strategy

### Phase 1: Unit Tests (Per Module)

**Coverage target:** >80% line coverage

**Tools:** pytest, unittest

**Structure:**
tests/
├── test_health_tracker.py
├── test_degradation_model.py
├── test_failure_probability.py
├── test_rul_estimator.py
├── test_state_manager.py
├── test_forecast_engine.py
├── test_sensor_attribution.py
└── test_metrics.py
---

### Phase 2: Integration Tests

**Test scenario 1: Batch-by-batch continuity**
```python
def test_forecast_continuity_over_100_batches():
    """
    Simulate 100 consecutive batches.
    Verify:
        1. Forecasts don't jump (max change < 10% per batch)
        2. State persists correctly
        3. Retrains trigger at expected times
    """
```

**Test scenario 2: Data quality degradation**
```python
def test_forecast_with_gappy_data():
    """
    Introduce 24h data gap.
    Verify:
        1. Quality flag = "GAPPY"
        2. Forecast confidence drops
        3. System doesn't crash
    """
```

**Test scenario 3: Regime shift detection**
```python
def test_regime_shift_retrain():
    """
    Simulate sudden degradation acceleration.
    Verify:
        1. Regime shift detected
        2. Model retrained
        3. RUL updated appropriately
    """
```

---

### Phase 3: Accuracy Validation

**Use historical failure data:**

1. Load equipment that failed at known time T_fail
2. Run forecasting starting at T_fail - 30 days
3. Track RUL predictions over time
4. Measure:
   - RUL error at T_fail - 7 days
   - RUL error at T_fail - 1 day
   - False alarm rate (predicted failure but didn't happen)

**Acceptance criteria:**
- RUL within ±20% of actual at T_fail - 7 days
- RUL within ±10% of actual at T_fail - 1 day
- False alarm rate < 10%

---

## 13. Common Pitfalls & Solutions

### Pitfall 1: Forecast Jumps When New Data Arrives

**Symptom:** Health forecast changes drastically between batches

**Cause:** No blending of old/new forecasts

**Solution:** Always use `blend_forecasts()` with alpha=0.3 unless retrain triggered

---

### Pitfall 2: Model Overconfidence at Long Horizons

**Symptom:** Narrow confidence intervals 168 hours out

**Cause:** Constant forecast std (doesn't grow with time)

**Solution:** Use uncertainty growth formula:
```python
forecast_std = residual_std * sqrt(1 + t/T)
```

---

### Pitfall 3: Retrain Too Frequently

**Symptom:** Model refits every batch, slow performance

**Cause:** Overly aggressive retrain triggers

**Solution:** Use conservative thresholds:
- Time-based: max_hours_since_retrain = 168 (1 week)
- Performance-based: threshold = 2.0 (MAE doubles)

---

### Pitfall 4: State Corruption After Crash

**Symptom:** Forecasts break after system restart

**Cause:** Incomplete state writes or deserialization errors

**Solution:**
- Use transactions (BEGIN/COMMIT) for state writes
- Add schema version to state
- Graceful fallback to initial training if state load fails

---

### Pitfall 5: Hazard Rate Calculation Errors

**Symptom:** Negative or inf hazard rates

**Cause:** Division by zero when S(t) â†' 0

**Solution:** Add guards:
```python
if S_prev > 1e-9 and dt_hours > 1e-6:
    hazard[i] = dF / (S_prev * dt_hours)
else:
    hazard[i] = 10.0  # Sentinel for saturation
```

---

## 14. Hyperparameter Tuning Guide

### Parameter: window_hours (Sliding Window Size)

**Default:** 168 hours (7 days)

**Tuning:**
- **Shorter (<72h):** More reactive, less stable
- **Longer (>336h):** More stable, less reactive
- **Guideline:** Use 2-4x typical degradation timescale

---

### Parameter: blend_alpha (Forecast Blending Weight)

**Default:** 0.3

**Tuning:**
- **Lower (<0.2):** Smoother transitions, slower adaptation
- **Higher (>0.5):** Faster adaptation, more jumps
- **Guideline:** 0.3 for stable equipment, 0.5 for fast-changing

---

### Parameter: failure_threshold

**Default:** 70.0 (health units)

**Tuning:**
- Equipment-specific based on historical failures
- Method: Find health value at which 90% of failures occurred

---

### Parameter: forecast_hours (Forecast Horizon)

**Default:** 168 hours (7 days)

**Tuning:**
- Maintenance planning window + 50%
- Longer horizons = more uncertainty
- Typical: 168h (weekly planning), 720h (monthly planning)

---

### Parameter: n_simulations (Monte Carlo)

**Default:** 1000
Tuning:

Fewer (<500): Faster, noisier RUL estimates
More (>2000): Slower, stabler RUL estimates
Guideline: 1000 is good balance


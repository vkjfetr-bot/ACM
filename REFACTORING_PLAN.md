# ACM RUL Engine Refactoring Plan

**Objective**: Consolidate `rul_estimator.py` and `enhanced_rul_estimator.py` into single unified `rul_engine.py` with SQL-only I/O

**Status**: 1/30 tasks complete (3.3%)  
**Created**: 2025-01-01  
**Last Updated**: 2025-01-01 after commit ea427f4

---

## Executive Summary

This document tracks consolidation of two RUL estimator implementations (~1900 lines of code) into a single unified engine. The refactoring eliminates CSV fallbacks, implements SQL-backed learning persistence, and combines the best features of both estimators into one maintainable module.

**Key Requirements**:
- NO CSV file I/O anywhere
- SQL-backed learning state (no JSON files)
- Single unified engine (no "simple vs enhanced" split)
- Ensemble models always active (AR1 + Exponential + Weibull)
- Multipath RUL calculation (trajectory + hazard + energy)
- All existing SQL table schemas preserved

**Current Files**:
- `core/rul_estimator.py` - Simple AR1-based RUL (~720 lines)
- `core/enhanced_rul_estimator.py` - Ensemble with learning (~1175 lines)
- `core/rul_common.py` - Shared utilities
- `core/rul_engine.py` - NEW unified engine (456 lines foundation complete)

---

## Progress Summary

**Total Tasks**: 30  
**Completed**: 1 ✅  
**In Progress**: 0  
**Not Started**: 29  
**Overall Progress**: 3.3%

### Phase Breakdown
- **Phase 1 - Foundation**: 1/1 complete (100%) ✅
- **Phase 2 - I/O Layer**: 3/3 complete (100%) ✅ (integrated in Phase 1)
- **Phase 3 - Model Layer**: 0/5 complete (0%) ⏳ NEXT PRIORITY
- **Phase 4 - Core Engine**: 0/4 complete (0%)
- **Phase 5 - Output Builders**: 0/3 complete (0%)
- **Phase 6 - Public API**: 1/2 complete (50%) ✅
- **Phase 7 - Integration**: 0/3 complete (0%)

---

## Phase 1: Foundation ✅ COMPLETE

### ✅ RUL-REF-01: Create unified rul_engine.py
**Status**: COMPLETED  
**Commit**: ea427f4  
**Date**: 2025-01-01  
**Lines**: 456

**What Was Built**:
- Created `core/rul_engine.py` with SQL-only architecture
- Unified `RULConfig` dataclass consolidating both estimators
- SQL-backed `LearningState` class (replaces JSON file persistence)
- Data quality assessment with flags (OK/SPARSE/GAPPY/FLAT/MISSING)
- Forecast cleanup with `ACM_FORECAST_RUNS_RETAIN` env var
- Type normalization utilities (RunID→str, EquipID→int)

**Key Functions Implemented**:
- `cleanup_old_forecasts()`: Keeps N most recent RunIDs
- `load_health_timeline()`: Cache > SQL priority, NO CSV fallback
- `load_sensor_hotspots()`: SQL-only with column mapping fixes
- `load_learning_state()` / `save_learning_state()`: SQL upsert pattern
- `_normalize_health_timeline()`: Timestamp standardization
- `_apply_row_limit()`: Downsampling for large datasets
- `_assess_data_quality()`: Returns quality flags

**Code Organization**:
- Lines 1-30: Module docstring, imports
- Lines 35-58: RULConfig dataclass
- Lines 61-70: Utility functions
- Lines 75-95: cleanup_old_forecasts
- Lines 98-180: load_health_timeline
- Lines 183-220: Helper functions
- Lines 223-275: load_sensor_hotspots
- Lines 281-342: LearningState dataclasses
- Lines 345-410: Learning state persistence
- Lines 415-456: Placeholders with TODO comments

---

## Phase 2: I/O Layer ✅ COMPLETE

All Phase 2 tasks were integrated into Phase 1 (RUL-REF-01).

### ✅ RUL-REF-04: SQL-only health timeline loader
**Status**: COMPLETED (integrated in RUL-REF-01)
- Priority: Cache > SQL, NO CSV
- Row limit enforcement (max 100k rows)
- Data quality assessment
- Timezone-naive datetime handling

### ✅ RUL-REF-05: SQL-only sensor hotspots
**Status**: COMPLETED (integrated in RUL-REF-01)
- Fixed column name mapping (AlertCount vs AboveAlertCount)
- Fixed Z vs ZScoreAtFailure inconsistency
- NO CSV fallback

### ✅ RUL-REF-06: SQL learning state persistence
**Status**: COMPLETED (integrated in RUL-REF-01)
- SQL table: `ACM_RUL_LearningState`
- Upsert pattern (UPDATE if exists, INSERT if not)
- JSON columns for error arrays
- NO JSON file persistence

---

## Phase 3: Model Layer ⏳ NEXT PRIORITY

### RUL-REF-11: Base degradation model
**Status**: NOT STARTED  
**Priority**: HIGH  
**Est. Time**: 30 min  
**Dependencies**: None

**Requirements**:
- Copy `DegradationModel` base class from `enhanced_rul_estimator.py`
- Abstract methods: `fit(t, h) -> bool`, `predict(t_future) -> (mean, std)`
- Keep purely numerical (no file I/O)
- Add fit_failed flag for ensemble weighting

**Files to Modify**:
- `core/rul_engine.py` (add after line 456)

**Code to Extract** (from enhanced_rul_estimator.py lines ~60-90):
```python
class DegradationModel:
    """Base class for health degradation models."""
    
    def fit(self, t: np.ndarray, h: np.ndarray) -> bool:
        """
        Train model on time series.
        Returns True if fit succeeded, False if insufficient data.
        """
        raise NotImplementedError
    
    def predict(self, t_future: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future health values.
        Returns (mean_prediction, std_prediction).
        """
        raise NotImplementedError
```

---

### RUL-REF-12: AR1 model
**Status**: NOT STARTED  
**Priority**: HIGH  
**Est. Time**: 45 min  
**Dependencies**: RUL-REF-11

**Requirements**:
- Copy `AR1Model` from `enhanced_rul_estimator.py` (lines ~93-150)
- Remove simple `_simple_ar1_forecast` from old `rul_estimator.py`
- Configurable recent-window training (last N points or hours)
- Return fit_failed=True if insufficient data (<min_points)
- Use recent_window_points from config

**Files to Modify**:
- `core/rul_engine.py`

**Key Features**:
- First-order autoregressive model
- Residual variance estimation
- Confidence intervals from residual std
- Recent data windowing (default: last 1000 points)

---

### RUL-REF-13: Exponential & Weibull models
**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Est. Time**: 1 hour  
**Dependencies**: RUL-REF-11

**Requirements**:

**ExponentialDegradationModel**:
- Copy from enhanced_rul_estimator (lines ~153-200)
- Fit exponential decay: `h(t) = a * exp(-b * t) + c`
- Use scipy.optimize.curve_fit with robust bounds
- Return fit_failed if curve_fit fails or parameters out of range

**WeibullInspiredModel**:
- Copy from enhanced_rul_estimator (lines ~203-260)
- Fit Weibull-like degradation curve
- Handle short/noisy time series gracefully
- Return fit_failed for extreme parameters

**Files to Modify**:
- `core/rul_engine.py`

**Error Handling**:
- Both models must handle RuntimeError, ValueError from scipy
- Set fit_failed=True on any fitting exception
- Log warnings (not errors) when fit fails

---

### RUL-REF-14: Ensemble orchestration (RULModel wrapper)
**Status**: NOT STARTED  
**Priority**: HIGH  
**Est. Time**: 1.5 hours  
**Dependencies**: RUL-REF-11, RUL-REF-12, RUL-REF-13

**Requirements**:
- Create `RULModel` wrapper class that manages ensemble
- Instantiate AR1 + Exp + Weibull models
- `fit()` trains all 3, tracks which succeeded
- `forecast()` combines predictions using weights from LearningState
- Normalize weights with `min_model_weight` floor
- Return per-model predictions + ensemble mean/std

**Files to Modify**:
- `core/rul_engine.py`

**Design**:
```python
class RULModel:
    """Ensemble wrapper for multiple degradation models."""
    
    def __init__(self, cfg: RULConfig, learning_state: LearningState):
        self.ar1 = AR1Model()
        self.exp = ExponentialDegradationModel()
        self.weibull = WeibullInspiredModel()
        self.cfg = cfg
        self.learning_state = learning_state
        self.fit_succeeded = {'ar1': False, 'exp': False, 'weibull': False}
    
    def fit(self, t: np.ndarray, h: np.ndarray) -> Dict[str, bool]:
        """Fit all models, return success status for each."""
        self.fit_succeeded['ar1'] = self.ar1.fit(t, h)
        self.fit_succeeded['exp'] = self.exp.fit(t, h)
        self.fit_succeeded['weibull'] = self.weibull.fit(t, h)
        return self.fit_succeeded
    
    def forecast(self, t_future: np.ndarray) -> Dict[str, Any]:
        """
        Combine predictions using learned weights.
        
        Returns:
            mean: Ensemble mean forecast
            std: Ensemble std forecast
            per_model_forecasts: Dict with ar1/exp/weibull predictions
            weights_used: Normalized weights
        """
        # Get predictions from each model that fit succeeded
        # Normalize weights from learning_state
        # Combine with weighted average
        # Return ensemble + diagnostics
```

**Weight Normalization**:
- Start with weights from `learning_state.ar1.weight`, etc.
- Zero out weights where `fit_succeeded == False`
- Apply `min_model_weight` floor (default 0.1)
- Normalize to sum to 1.0
- Log final weights used

---

### RUL-REF-15: Failure distribution
**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Est. Time**: 45 min  
**Dependencies**: RUL-REF-14

**Requirements**:
- Create `compute_failure_distribution()` helper
- Use ensemble mean + std with threshold crossing
- Compute failure probability at each future timestamp
- Use Gaussian CDF approximation via `norm_cdf`
- Return DataFrame: Timestamp, FailureProb

**Files to Modify**:
- `core/rul_engine.py`

**Algorithm**:
```python
def compute_failure_distribution(
    t_future: np.ndarray,
    health_mean: np.ndarray,
    health_std: np.ndarray,
    threshold: float = 70.0
) -> pd.DataFrame:
    """
    Compute probability of failure at each future time.
    
    P(failure at t) = P(health(t) < threshold)
                    = Φ((threshold - mean(t)) / std(t))
    
    where Φ is standard normal CDF.
    """
    # Use norm_cdf utility
    # Return DataFrame with Timestamp, FailureProb
```

---

## Phase 4: Core Engine

### RUL-REF-18: Single compute_rul engine
**Status**: NOT STARTED  
**Priority**: CRITICAL  
**Est. Time**: 2 hours  
**Dependencies**: RUL-REF-14, RUL-REF-15

**Requirements**:
- Implement main `compute_rul()` function
- Data conditioning: sort, deduplicate, handle gaps
- Detect sampling interval, build future index
- Instantiate RULModel and fit on recent window
- Forecast health mean/std over horizon
- Call failure distribution helper
- Call multipath RUL selection
- Return comprehensive dict with forecasts + diagnostics

**Files to Modify**:
- `core/rul_engine.py`

**Function Signature**:
```python
def compute_rul(
    health_df: pd.DataFrame,
    cfg: RULConfig,
    learning_state: LearningState,
    data_quality_flag: str
) -> Dict[str, Any]:
    """
    Core RUL computation engine.
    
    Args:
        health_df: Health timeline (Timestamp, HealthIndex)
        cfg: Configuration
        learning_state: Current learning state
        data_quality_flag: Data quality assessment
    
    Returns:
        Dict with keys:
        - health_forecast: DataFrame (Timestamp, HealthIndex, CI_Lower, CI_Upper)
        - failure_curve: DataFrame (Timestamp, FailureProb)
        - rul_ts: DataFrame (Timestamp, RUL_Hours, LowerBound, UpperBound)
        - rul_summary: Dict (RUL_Selected, method, confidence, bounds)
        - model_diagnostics: Dict (weights, fit_status, per_model_forecasts)
        - data_quality: str flag
    """
```

**Steps**:
1. Validate and condition health_df
2. Detect sampling interval (median of time deltas)
3. Build future time index (cfg.max_forecast_hours)
4. Apply recent window if needed
5. Instantiate and fit RULModel
6. Generate ensemble forecast
7. Compute failure distribution
8. Call compute_rul_multipath
9. Package all outputs

---

### RUL-REF-19: Multipath RUL calculation
**Status**: NOT STARTED  
**Priority**: HIGH  
**Est. Time**: 1.5 hours  
**Dependencies**: RUL-REF-18

**Requirements**:
- Copy `compute_rul_multipath` from enhanced_rul_estimator (lines ~263-350)
- Adapt for ensemble outputs
- Three paths:
  - **Trajectory**: Mean forecast crosses threshold
  - **Hazard**: Failure probability exceeds threshold (e.g., 50%)
  - **Energy**: Optional area-under-curve approach
- Select dominant RUL and log rationale
- Return: RUL_Selected (hours), method, LowerBound, UpperBound

**Files to Modify**:
- `core/rul_engine.py`

**Selection Logic**:
- If trajectory path valid, use it (most reliable)
- If hazard path significantly different, log warning
- If only hazard available, use it
- If neither available, return None with warning

---

### RUL-REF-17: Learning update
**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Est. Time**: 1 hour  
**Dependencies**: Phase 3 complete

**Requirements**:
- Implement `update_learning_state()` function
- Update per-model MAE, RMSE, bias
- Update model weights based on recent performance
- Update calibration_factor
- Trim prediction_history to `calibration_window`
- Clamp calibration_factor to [0.5, 2.0]

**Files to Modify**:
- `core/rul_engine.py`

**Function Signature**:
```python
def update_learning_state(
    learning_state: LearningState,
    actual_outcomes: Dict[str, Any],
    predictions: Dict[str, Any],
    cfg: RULConfig
) -> LearningState:
    """
    Update learning state with new observations.
    
    Args:
        learning_state: Current state
        actual_outcomes: Ground truth values
        predictions: Model predictions
        cfg: Configuration
    
    Returns:
        Updated LearningState
    """
```

**Update Logic**:
- Compute errors for each model
- Append to recent_errors list
- Keep last `calibration_window` errors
- Recompute MAE, RMSE, bias
- Adjust weights (lower error → higher weight)
- Update calibration factor (systematic bias correction)

---

### RUL-REF-20: Confidence computation
**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Est. Time**: 45 min  
**Dependencies**: RUL-REF-18, RUL-REF-19

**Requirements**:
- Compute confidence score for RUL estimate
- Factors:
  - CI width (narrower → higher confidence)
  - Model agreement (all models close → higher confidence)
  - Calibration stability (stable factor → higher confidence)
  - Data quality (OK → high, GAPPY → lower)
- Return confidence in [0, 1]

**Files to Modify**:
- `core/rul_engine.py`

**Function Signature**:
```python
def compute_confidence(
    rul_bounds: Tuple[float, float],
    model_diagnostics: Dict[str, Any],
    learning_state: LearningState,
    data_quality_flag: str
) -> float:
    """
    Compute confidence score for RUL estimate.
    
    Returns confidence in [0.0, 1.0].
    """
```

---

## Phase 5: Output Builders

### RUL-REF-24: Output DataFrame shaping
**Status**: NOT STARTED  
**Priority**: HIGH  
**Est. Time**: 1 hour  
**Dependencies**: RUL-REF-18

**Requirements**:
- Create helper functions to build output DataFrames
- Match existing SQL table schemas exactly
- Fix CI naming inconsistencies (use CI_Lower/CI_Upper consistently)

**Files to Modify**:
- `core/rul_engine.py`

**Functions to Implement**:

```python
def make_health_forecast_df(
    timestamps: np.ndarray,
    health_mean: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray
) -> pd.DataFrame:
    """
    Build ACM_HealthForecast_TS compatible DataFrame.
    
    Columns: Timestamp, HealthIndex, CI_Lower, CI_Upper
    """

def make_failure_curve_df(
    timestamps: np.ndarray,
    failure_probs: np.ndarray,
    threshold: float
) -> pd.DataFrame:
    """
    Build ACM_FailureForecast_TS compatible DataFrame.
    
    Columns: Timestamp, FailureProb, ThresholdUsed
    """

def make_rul_ts_df(
    timestamps: np.ndarray,
    rul_hours: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    confidence: float
) -> pd.DataFrame:
    """
    Build ACM_RUL_TS compatible DataFrame.
    
    Columns: Timestamp, RUL_Hours, LowerBound, UpperBound, Confidence
    """

def make_rul_summary_df(
    rul_selected: float,
    lower_bound: float,
    upper_bound: float,
    confidence: float,
    method: str,
    model_weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Build ACM_RUL_Summary compatible DataFrame (single row).
    
    Columns: RUL_Hours, LowerBound, UpperBound, Confidence, Method,
             AR1_Weight, Exp_Weight, Weibull_Weight
    """
```

---

### RUL-REF-22: Sensor attribution
**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Est. Time**: 45 min  
**Dependencies**: RUL-REF-18

**Requirements**:
- Build sensor attribution table from hotspots
- Use SQL-loaded hotspots (no CSV)
- Match existing `ACM_RUL_Attribution` schema

**Files to Modify**:
- `core/rul_engine.py`

**Function Signature**:
```python
def build_sensor_attribution(
    sensor_hotspots_df: Optional[pd.DataFrame],
    rul_result: Dict[str, Any],
    run_id: str,
    equip_id: int
) -> pd.DataFrame:
    """
    Build ACM_RUL_Attribution compatible DataFrame.
    
    Columns: RunID, EquipID, FailureTime, SensorName,
             FailureContribution, ZScoreAtFailure, AlertCount
    """
```

**Logic**:
- Use sensor hotspots loaded from SQL
- Add RunID, EquipID columns
- Compute FailureTime from current time + RUL
- Use existing FailureContribution, ZScoreAtFailure, AlertCount

---

### RUL-REF-23: Maintenance recommendations
**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Est. Time**: 45 min  
**Dependencies**: RUL-REF-18, RUL-REF-20

**Requirements**:
- Generate maintenance recommendations based on RUL + confidence + data quality
- Use maintenance bands from config
- Match existing `ACM_MaintenanceRecommendation` schema

**Files to Modify**:
- `core/rul_engine.py`

**Function Signature**:
```python
def build_maintenance_recommendation(
    rul_result: Dict[str, Any],
    bands: Dict[str, float],
    data_quality_flag: str,
    run_id: str,
    equip_id: int
) -> pd.DataFrame:
    """
    Build ACM_MaintenanceRecommendation compatible DataFrame.
    
    Columns: RunID, EquipID, Action, Urgency, RUL_Hours,
             Confidence, DataQuality
    """
```

**Bands** (from RULConfig):
- **Normal**: RUL > 168h (7 days) → "Continue monitoring"
- **Watch**: 72h < RUL ≤ 168h → "Increase monitoring frequency"
- **Plan**: 24h < RUL ≤ 72h → "Schedule maintenance"
- **Urgent**: RUL ≤ 24h → "Immediate action required"

**Urgency Adjustment**:
- Low confidence → increase urgency
- GAPPY/SPARSE data → increase urgency
- High confidence + good data → use band as-is

---

## Phase 6: Public API

### ✅ RUL-REF-26: Type normalization
**Status**: COMPLETED (integrated in RUL-REF-01)

Utilities already implemented:
- `ensure_runid_str()`: Normalizes RunID to string
- `ensure_equipid_int()`: Normalizes EquipID to positive integer

---

### RUL-REF-25: Unified run_rul() API
**Status**: NOT STARTED  
**Priority**: CRITICAL  
**Est. Time**: 2 hours  
**Dependencies**: All previous phases

**Requirements**:
- Single public entry point: `run_rul()`
- Orchestrates entire RUL pipeline
- Loads config from SQL/defaults
- Cleans up old forecasts
- Loads health timeline + learning state
- Calls compute_rul
- Builds attribution + maintenance reco
- Writes all outputs to SQL via output_manager
- Saves updated learning state
- Returns dict of DataFrames for all 6 tables

**Files to Modify**:
- `core/rul_engine.py`

**Function Signature**:
```python
def run_rul(
    sql_client: Any,
    equip_id: int,
    run_id: str,
    output_manager: Optional[Any] = None,
    config_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Unified RUL estimation entry point.
    
    This is the single public API for RUL estimation. Replaces both
    estimate_rul_and_failure() from rul_estimator.py and
    enhanced_rul_estimator.py.
    
    Args:
        sql_client: SQL client for data access
        equip_id: Equipment ID (normalized to int)
        run_id: Run ID (normalized to str)
        output_manager: Optional OutputManager for dual-write
        config_row: Optional config overrides from config_table.csv
    
    Returns:
        Dict with keys:
        - ACM_HealthForecast_TS: Health forecast time series
        - ACM_FailureForecast_TS: Failure probability curve
        - ACM_RUL_TS: RUL time series
        - ACM_RUL_Summary: Single-row summary
        - ACM_RUL_Attribution: Sensor attribution
        - ACM_MaintenanceRecommendation: Maintenance actions
    
    Raises:
        ValueError: Invalid inputs
        RuntimeError: SQL connection issues or data unavailable
    """
    # 1. Normalize types
    equip_id = ensure_equipid_int(equip_id)
    run_id = ensure_runid_str(run_id)
    
    # 2. Build config
    cfg = RULConfig.from_config_row(config_row) if config_row else RULConfig()
    
    # 3. Cleanup old forecasts
    cleanup_old_forecasts(sql_client, equip_id, cfg)
    
    # 4. Load health timeline
    health_df, data_quality = load_health_timeline(
        sql_client, equip_id, run_id, output_manager, cfg
    )
    
    # 5. Load learning state
    learning_state = load_learning_state(sql_client, equip_id)
    
    # 6. Compute RUL
    rul_result = compute_rul(health_df, cfg, learning_state, data_quality)
    
    # 7. Load sensor hotspots
    hotspots_df = load_sensor_hotspots(sql_client, equip_id, run_id)
    
    # 8. Build output DataFrames
    tables = {}
    tables['ACM_HealthForecast_TS'] = make_health_forecast_df(...)
    tables['ACM_FailureForecast_TS'] = make_failure_curve_df(...)
    tables['ACM_RUL_TS'] = make_rul_ts_df(...)
    tables['ACM_RUL_Summary'] = make_rul_summary_df(...)
    tables['ACM_RUL_Attribution'] = build_sensor_attribution(...)
    tables['ACM_MaintenanceRecommendation'] = build_maintenance_recommendation(...)
    
    # 9. Add RunID and EquipID columns to all tables
    for table_name, df in tables.items():
        df['RunID'] = run_id
        df['EquipID'] = equip_id
    
    # 10. Write to SQL (if output_manager provided)
    if output_manager:
        for table_name, df in tables.items():
            output_manager.write_table(table_name, df)
    
    # 11. Save updated learning state
    # (Only if online learning enabled and actual outcomes available)
    # save_learning_state(sql_client, equip_id, learning_state)
    
    # 12. Return all tables
    return tables
```

---

## Phase 7: Integration

### RUL-REF-29: Remove old estimators
**Status**: NOT STARTED  
**Priority**: HIGH  
**Est. Time**: 30 min  
**Dependencies**: RUL-REF-25, Integration complete

**Requirements**:
- Delete or deprecate old RUL modules
- Options:
  1. **Delete entirely**: `rul_estimator.py`, `enhanced_rul_estimator.py`
  2. **Keep thin wrappers**: Add deprecation warnings, call `rul_engine.run_rul()`

**Recommended Approach**: Thin wrappers for backward compatibility

**Files to Modify**:
- `core/rul_estimator.py` (reduce to wrapper or delete)
- `core/enhanced_rul_estimator.py` (reduce to wrapper or delete)
- `core/rul_common.py` (may reduce to re-exports or delete)

**Wrapper Example**:
```python
# core/rul_estimator.py
import warnings
from core.rul_engine import run_rul

warnings.warn(
    "rul_estimator.py is deprecated. Use rul_engine.run_rul() instead.",
    DeprecationWarning,
    stacklevel=2
)

def estimate_rul_and_failure(*args, **kwargs):
    """DEPRECATED: Use rul_engine.run_rul() instead."""
    return run_rul(*args, **kwargs)
```

---

### Update acm_main integration
**Status**: NOT STARTED  
**Priority**: CRITICAL  
**Est. Time**: 1 hour  
**Dependencies**: RUL-REF-25

**Requirements**:
- Update `forecasting.estimate_rul()` to call `rul_engine.run_rul()`
- Remove dispatch between simple/enhanced modes
- Test end-to-end in file mode and SQL mode
- Verify all outputs match previous behavior

**Files to Modify**:
- `core/forecasting.py` (lines ~470-550)
- `core/acm_main.py` (verify integration at lines ~3820+)

**Changes in forecasting.py**:
```python
# OLD
from core.enhanced_rul_estimator import estimate_rul_and_failure

# NEW
from core.rul_engine import run_rul

# Update estimate_rul() function
def estimate_rul(...):
    """Wrapper for backward compatibility."""
    return run_rul(
        sql_client=sql_client,
        equip_id=equip_id,
        run_id=run_id,
        output_manager=output_manager,
        config_row=cfg_row
    )
```

---

### RUL-REF-30: Validation harness
**Status**: NOT STARTED  
**Priority**: MEDIUM  
**Est. Time**: 1.5 hours  
**Dependencies**: Integration complete

**Requirements**:
- Create end-to-end validation script or notebook
- Run on historical data for 2-3 assets
- Verify SQL table schemas match expectations
- Log RUL summaries for manual review
- NO CSV dependencies anywhere

**Files to Create**:
- `scripts/validate_rul_engine.py`

**Script Structure**:
```python
"""
Validate unified RUL engine against historical data.

Usage:
    python scripts/validate_rul_engine.py --equip FD_FAN --runs 5
"""

import argparse
from core.rul_engine import run_rul
from core.sql_client import SQLClient
from utils.logger import Console

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--equip', required=True)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    
    # Connect to SQL
    sql_client = SQLClient.from_config()
    
    # Get recent RunIDs for equipment
    run_ids = get_recent_runs(sql_client, args.equip, args.runs)
    
    # Run RUL engine for each RunID
    for run_id in run_ids:
        Console.info(f"\n{'='*60}")
        Console.info(f"Validating: {args.equip} / {run_id}")
        
        try:
            result = run_rul(sql_client, args.equip, run_id)
            validate_result(result)
            log_summary(result)
        except Exception as e:
            Console.error(f"Failed: {e}")
    
    Console.info("\n✅ Validation complete")

def validate_result(result):
    """Check result structure and schemas."""
    required_tables = [
        'ACM_HealthForecast_TS',
        'ACM_FailureForecast_TS',
        'ACM_RUL_TS',
        'ACM_RUL_Summary',
        'ACM_RUL_Attribution',
        'ACM_MaintenanceRecommendation'
    ]
    
    for table in required_tables:
        assert table in result, f"Missing table: {table}"
        assert not result[table].empty, f"Empty table: {table}"
    
    # Validate schemas
    # ...

def log_summary(result):
    """Log key metrics for manual review."""
    summary = result['ACM_RUL_Summary'].iloc[0]
    Console.info(f"  RUL: {summary['RUL_Hours']:.1f} hours")
    Console.info(f"  Confidence: {summary['Confidence']:.2f}")
    Console.info(f"  Method: {summary['Method']}")
    # ...

if __name__ == '__main__':
    main()
```

---

## SQL Schema Requirements

### ACM_RUL_LearningState (NEW TABLE)

**Must be created before testing SQL learning state persistence**.

**SQL Script** (add to `scripts/sql/14_complete_schema.sql`):

```sql
-- RUL Learning State (replaces JSON file persistence)
CREATE TABLE dbo.ACM_RUL_LearningState (
    EquipID INT PRIMARY KEY,
    
    -- AR1 Model Metrics
    AR1_MAE FLOAT DEFAULT 0.0,
    AR1_RMSE FLOAT DEFAULT 0.0,
    AR1_Bias FLOAT DEFAULT 0.0,
    AR1_RecentErrors NVARCHAR(MAX),  -- JSON array: [e1, e2, ...]
    AR1_Weight FLOAT DEFAULT 1.0,
    
    -- Exponential Model Metrics
    Exp_MAE FLOAT DEFAULT 0.0,
    Exp_RMSE FLOAT DEFAULT 0.0,
    Exp_Bias FLOAT DEFAULT 0.0,
    Exp_RecentErrors NVARCHAR(MAX),
    Exp_Weight FLOAT DEFAULT 1.0,
    
    -- Weibull Model Metrics
    Weibull_MAE FLOAT DEFAULT 0.0,
    Weibull_RMSE FLOAT DEFAULT 0.0,
    Weibull_Bias FLOAT DEFAULT 0.0,
    Weibull_RecentErrors NVARCHAR(MAX),
    Weibull_Weight FLOAT DEFAULT 1.0,
    
    -- Overall Metrics
    CalibrationFactor FLOAT DEFAULT 1.0,
    PredictionHistory NVARCHAR(MAX),  -- JSON array of recent predictions
    
    -- Metadata
    LastUpdated DATETIME2,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    
    -- Constraints
    CONSTRAINT CK_LearningState_WeightsPositive CHECK (
        AR1_Weight >= 0 AND Exp_Weight >= 0 AND Weibull_Weight >= 0
    ),
    CONSTRAINT CK_LearningState_CalibrationRange CHECK (
        CalibrationFactor BETWEEN 0.1 AND 10.0
    )
);

CREATE INDEX IX_ACM_RUL_LearningState_LastUpdated 
    ON dbo.ACM_RUL_LearningState(LastUpdated);
```

---

## Recommended Execution Order

### Week 1: Models (Phase 3)
1. RUL-REF-11: Base model class (30 min)
2. RUL-REF-12: AR1 model (45 min)
3. RUL-REF-13: Exp + Weibull models (1 hr)
4. RUL-REF-14: Ensemble wrapper (1.5 hr)
5. RUL-REF-15: Failure distribution (45 min)

**Total**: ~4.5 hours

### Week 2: Core Engine (Phase 4)
6. RUL-REF-18: compute_rul function (2 hr)
7. RUL-REF-19: Multipath RUL (1.5 hr)
8. RUL-REF-17: Learning update (1 hr)
9. RUL-REF-20: Confidence computation (45 min)

**Total**: ~5.25 hours

### Week 3: Outputs & API (Phases 5-6)
10. RUL-REF-24: Output DataFrames (1 hr)
11. RUL-REF-22: Sensor attribution (45 min)
12. RUL-REF-23: Maintenance reco (45 min)
13. RUL-REF-25: Public run_rul() API (2 hr)

**Total**: ~4.5 hours

### Week 4: Integration (Phase 7)
14. Create ACM_RUL_LearningState table (15 min)
15. Update forecasting.py (1 hr)
16. Update acm_main.py if needed (30 min)
17. RUL-REF-30: Validation harness (1.5 hr)
18. RUL-REF-29: Remove old estimators (30 min)

**Total**: ~3.5 hours

**Grand Total**: ~18 hours of focused work

---

## Success Criteria

- ✅ Single `rul_engine.py` module (~1500-2000 lines)
- ✅ NO CSV file I/O anywhere
- ✅ NO JSON learning state files
- ✅ SQL-backed learning persistence (ACM_RUL_LearningState)
- ✅ Ensemble always active (AR1 + Exp + Weibull)
- ✅ Multipath RUL calculation
- ✅ All existing SQL table schemas preserved
- ✅ `acm_main.py` uses only new engine
- ✅ Validation harness passes
- ✅ Old estimators deprecated or deleted

---

## Notes & Decisions

### Breaking Changes
- Old `estimate_rul_and_failure()` signature changes
- No more "simple vs enhanced" mode selection
- Config parameters consolidated

### Migration Path
- Keep thin wrappers in old files temporarily
- Add deprecation warnings
- Give users 1-2 releases to migrate

### Testing Strategy
- Each phase commits independently
- Run validation harness after each phase
- Git tags recommended after major phases
- Keep old code until validation passes

### Rollback Plan
- Git revert to last working commit
- Old estimators remain until validation complete
- Can run old and new in parallel during transition

---

## Quick Reference

### Key Files
- `core/rul_engine.py` - NEW unified engine
- `core/rul_estimator.py` - OLD simple estimator (to deprecate)
- `core/enhanced_rul_estimator.py` - OLD enhanced estimator (to deprecate)
- `core/rul_common.py` - Shared utilities (may consolidate)
- `core/forecasting.py` - Calls RUL estimators
- `core/acm_main.py` - Main pipeline
- `RUL Audit Updated.md` - Source requirements document

### SQL Tables
- **Inputs**: ACM_HealthTimeline, ACM_SensorHotspots
- **Outputs**: ACM_HealthForecast_TS, ACM_FailureForecast_TS, ACM_RUL_TS, ACM_RUL_Summary, ACM_RUL_Attribution, ACM_MaintenanceRecommendation
- **NEW**: ACM_RUL_LearningState

### Git Workflow
- Branch: `refactor/rul-engine-consolidation`
- Commit after each task: `RUL-REF-XX: <description>`
- Tag major phases: `rul-engine-phase-3-complete`
- Squash merge to main when complete

### Progress Tracking
Update this document after each task:
- Change status to COMPLETED
- Add commit hash
- Update progress percentages
- Mark blocking tasks

---

**Last Updated**: 2025-01-01 after commit ea427f4 (RUL-REF-01 complete)

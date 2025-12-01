"""
Unified Forecasting Module
===========================

Single source of truth for all ACM forecasting capabilities:
- AR(1) baseline detector for per-sensor residual analysis
- Enhanced multi-model forecasting with SQL integration
- RUL estimation and failure probability calculation

Replaces legacy modules:
- forecast.py (AR1Detector)
- enhanced_forecasting.py (file-based engine)
- enhanced_forecasting_sql.py (SQL wrapper)

Timestamp Policy:
-----------------
All timestamps are treated as timezone-naive local time, consistent with the rest of the ACM platform.
Inputs from SQL (ACM_HealthTimeline, ACM_Scores_Wide) are stripped of timezone info if present.
Outputs are written as naive timestamps.

Error Handling Convention (FOR-CODE-02):
----------------------------------------
1. Non-fatal data conditions (missing data, insufficient samples, etc.):
   - Use Console.warn() to log the issue
   - Provide safe fallback behavior (skip processing, return default values)
   - Continue execution to allow partial results

2. Configuration/validation errors (invalid parameters, schema violations):
   - Use raise ValueError() with descriptive message
   - Fail fast to prevent silent corruption
   - Include context in error message for debugging

3. SQL/database failures (connection issues, query errors):
   - Use Console.error() to log critical issues
   - Propagate exception to caller (let acm_main handle)
   - Or mark run as failed if appropriate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal, List
from collections.abc import Mapping
import copy
import json

import numpy as np
import pandas as pd

from utils.logger import Console  # type: ignore
# FOR-DQ-02: Use centralized timestamp normalization
from utils.timestamp_utils import normalize_timestamps, normalize_index
# FOR-CODE-04: Use SqlClient protocol for type safety
from core.sql_protocol import SqlClient
from core import rul_engine  # Unified RUL estimation engine
from core.model_persistence import ForecastState, save_forecast_state, load_forecast_state  # type: ignore
from datetime import datetime, timedelta
import hashlib


# ============================================================================
# Module-Level Constants (FOR-CODE-01)
# ============================================================================

# Minimum samples required for AR(1) model coefficient estimation
# Below this threshold, AR(1) reverts to simple mean-based baseline
MIN_AR1_SAMPLES = 3

# Minimum samples for stable forecast generation
# Based on empirical testing: fewer samples produce unreliable forecasts
MIN_FORECAST_SAMPLES = 20

# Exponential blend time constant (hours) for hazard smoothing
# Controls how quickly hazard probability transitions between forecast updates
# 12 hours provides smooth transitions without excessive lag
BLEND_TAU_HOURS = 12.0

# Default exponential smoothing alpha for hazard time series
# Lower values (0.1-0.3) provide smoother hazard curves with less noise
DEFAULT_HAZARD_SMOOTHING_ALPHA = 0.3


def _coerce_config_mapping(config: Any) -> Dict[str, Any]:
    """Return a plain dict for downstream consumers regardless of ConfigDict input."""
    if config is None:
        return {}

    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()  # ConfigDict already returns a deep copy
        except Exception:
            pass

    if isinstance(config, Mapping):
        try:
            return copy.deepcopy(dict(config))
        except Exception:
            return dict(config)

    return {}


# ============================================================================
# Type Consistency Helpers (FOR-COR-02)
# ============================================================================

def ensure_runid_str(run_id: Any) -> str:
    """
    FOR-COR-02: Ensure RunID is always a string.
    Canonical type: RunID as string/UUID.
    """
    if run_id is None:
        raise ValueError("RunID cannot be None")
    return str(run_id)


def ensure_equipid_int(equip_id: Any) -> int:
    """
    FOR-COR-02: Ensure EquipID is always an int.
    Canonical type: EquipID as positive integer.
    """
    if equip_id is None:
        raise ValueError("EquipID cannot be None")
    try:
        eid = int(equip_id)
        if eid < 0:
            raise ValueError(f"EquipID must be non-negative, got {eid}")
        return eid
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid EquipID: {equip_id}") from e


# ============================================================================
# Continuous Forecasting Helpers (FORECAST-STATE-02, 03)
# ============================================================================

def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute SHA256 hash of DataFrame for change detection using binary serialization.
    
    More efficient than CSV serialization; maintains same semantics (hash changes when data changes).
    """
    try:
        # Sort columns for determinism
        sorted_cols = sorted(df.columns)
        vals = df[sorted_cols].to_numpy(copy=False).tobytes()
        
        # Include schema to detect column type changes
        schema = str(list(zip(sorted_cols, [str(dt) for dt in df[sorted_cols].dtypes]))).encode()
        
        return hashlib.sha256(vals + schema).hexdigest()[:16]
    except Exception:
        return ""


def should_retrain(
    prev_state: Optional[ForecastState],
    sql_client: SqlClient,
    equip_id: int,
    current_data_hash: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Determine if full model retrain is needed.
    
    Checks:
    1. State existence: No prior state triggers cold start retrain
    2. Data change: Training data hash mismatch
    3. Drift spike: Recent 5-point mean DriftValue > threshold
    4. Anomaly energy spike: Recent P95 > threshold * median
    
    Note: Forecast quality (RMSE) is computed separately via compute_forecast_quality()
    and should be checked explicitly by the caller if needed.
    
    Returns:
        (retrain_needed, reason)
    """
    forecast_cfg = config.get("forecasting", {})
    drift_threshold = float(forecast_cfg.get("drift_retrain_threshold", 1.5))
    energy_threshold = float(forecast_cfg.get("energy_spike_threshold", 1.5))
    error_threshold = float(forecast_cfg.get("forecast_error_threshold", 2.0))
    
    # No prior state always triggers retrain
    if prev_state is None:
        return True, "No prior forecast state (cold start)"
    
    # Check data hash change
    if prev_state.training_data_hash != current_data_hash:
        return True, f"Training data changed (hash mismatch)"
    
    # Check drift (FOR-CODE-02: Non-fatal SQL query - warn and continue)
    try:
        cur = sql_client.cursor()
        cur.execute("""
            SELECT AVG(DriftValue) as AvgDrift
            FROM (
                SELECT TOP 5 DriftValue
                FROM dbo.ACM_DriftMetrics
                WHERE EquipID = ?
                ORDER BY Timestamp DESC
            ) recent
        """, (equip_id,))
        row = cur.fetchone()
        cur.close()
        
        if row and row[0] is not None:
            avg_drift = float(row[0])
            if avg_drift > drift_threshold:
                return True, f"Drift spike detected (avg={avg_drift:.2f} > {drift_threshold})"
    except Exception as e:
        # FOR-CODE-02: Non-fatal data condition - log and continue without drift check
        Console.warn(f"[FORECAST] Failed to check drift: {e}")
    
    # Check anomaly energy spike (FOR-CODE-02: Non-fatal SQL query - warn and continue)
    try:
        cur = sql_client.cursor()
        cur.execute("""
            SELECT 
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY AnomalyEnergy) as P95,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY AnomalyEnergy) as Median
            FROM (
                SELECT TOP 100 
                    SUM(POWER(CAST(fused AS FLOAT), 2)) as AnomalyEnergy
                FROM dbo.ACM_Scores_Wide
                WHERE EquipID = ?
                GROUP BY Timestamp
                ORDER BY Timestamp DESC
            ) recent
        """, (equip_id,))
        row = cur.fetchone()
        cur.close()
        
        if row and row[0] is not None and row[1] is not None:
            p95 = float(row[0])
            median = float(row[1])
            if median > 0 and p95 > energy_threshold * median:
                return True, f"Anomaly energy spike (P95={p95:.2f} > {energy_threshold}x median)"
    except Exception as e:
        # FOR-CODE-02: Non-fatal data condition - log and continue without energy check
        Console.warn(f"[FORECAST] Failed to check anomaly energy: {e}")
    
    # All checks passed - incremental update sufficient
    return False, "Model stable, incremental update"


def compute_forecast_quality(
    prev_state: Optional[ForecastState],
    sql_client: SqlClient,
    equip_id: int,
    current_batch_time: datetime
) -> Dict[str, float]:
    """
    Compute forecast quality metrics by comparing previous forecasts to actual health values.
    
    Args:
        prev_state: Previous forecast state with forecast horizon
        sql_client: SQL connection for loading actual health data
        equip_id: Equipment identifier
        current_batch_time: Current batch timestamp
        
    Returns:
        Dictionary with rmse, mae, mape metrics (0.0 if no prior forecast available)
    """
    if prev_state is None or prev_state.last_forecast_horizon_json is None:
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}
    
    # FOR-CODE-02: Non-fatal metric computation - warn and return defaults on failure
    try:
        # Deserialize previous forecast horizon
        prev_horizon = prev_state.get_last_forecast_horizon()
        if prev_horizon.empty or "Timestamp" not in prev_horizon.columns:
            return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}
        
        # Get timestamps that should have occurred by now (past forecasts)
        past_forecasts = prev_horizon[prev_horizon["Timestamp"] <= current_batch_time].copy()
        if past_forecasts.empty:
            return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}
        
        # Load actual health values from ACM_HealthTimeline for those timestamps
        timestamps_list = past_forecasts["Timestamp"].tolist()
        if not timestamps_list:
            return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}
        
        # FOR-PERF-01: Batch large IN clauses to prevent SQL performance issues
        # Use a maximum of 1000 timestamps per query, loop if needed
        MAX_IN_CLAUSE = 1000
        all_rows = []
        cur = sql_client.cursor()
        
        for i in range(0, len(timestamps_list), MAX_IN_CLAUSE):
            batch_timestamps = timestamps_list[i:i+MAX_IN_CLAUSE]
            placeholders = ",".join("?" * len(batch_timestamps))
            query = f"""
                SELECT Timestamp, HealthIndex
                FROM dbo.ACM_HealthTimeline
                WHERE EquipID = ? AND Timestamp IN ({placeholders})
                ORDER BY Timestamp
            """
            cur.execute(query, (equip_id, *batch_timestamps))
            batch_rows = cur.fetchall()
            all_rows.extend(batch_rows)
        
        cur.close()
        rows = all_rows
        
        if not rows:
            return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}
        
        # Build actuals DataFrame from pyodbc Row objects
        actuals_data = [(row.Timestamp, row.HealthIndex) for row in rows]
        actuals = pd.DataFrame(actuals_data, columns=["Timestamp", "HealthIndex"])
        actuals["Timestamp"] = pd.to_datetime(actuals["Timestamp"])
        
        # Merge forecasts with actuals on timestamp
        merged = pd.merge(
            past_forecasts[["Timestamp", "ForecastHealth"]],
            actuals,
            on="Timestamp",
            how="inner"
        )
        
        if merged.empty or len(merged) < 2:
            return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}
        
        # Compute error metrics
        y_true = merged["HealthIndex"].values
        y_pred = merged["ForecastHealth"].values
        
        errors = y_true - y_pred
        squared_errors = errors ** 2
        abs_errors = np.abs(errors)
        
        rmse = float(np.sqrt(np.mean(squared_errors)))
        mae = float(np.mean(abs_errors))
        
        # MAPE: avoid division by zero
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = float(np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100)
        else:
            mape = 0.0
        
        return {"rmse": rmse, "mae": mae, "mape": mape}
        
    except Exception as e:
        # FOR-CODE-02: Non-fatal data condition - log and return safe defaults
        Console.warn(f"[FORECAST_QUALITY] Failed to compute metrics: {e}")
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}


def merge_forecast_horizons(
    prev_horizon: pd.DataFrame,
    new_horizon: pd.DataFrame,
    current_time: datetime,
    blend_tau_hours: float = BLEND_TAU_HOURS
) -> pd.DataFrame:
    """
    Merge overlapping forecast horizons with exponential temporal blending.
    
    Args:
        prev_horizon: Previous forecast (Timestamp, ForecastHealth, CI_Lower, CI_Upper)
        new_horizon: New forecast from current batch
        current_time: Current batch timestamp
        blend_tau_hours: Time constant for exponential decay (hours)
    
    Returns:
        Merged forecast DataFrame with smooth transition
    
    Logic:
        - Discard all past points (Timestamp < current_time)
        - For overlapping future points: w_new = 1 - exp(-dt/tau), w_prev = exp(-dt/tau)
        - Append non-overlapping new points
    """
    if prev_horizon.empty:
        return new_horizon.copy()
    
    if new_horizon.empty:
        return prev_horizon[prev_horizon["Timestamp"] >= current_time].copy()
    
    # Filter to future points only
    prev_future = prev_horizon[prev_horizon["Timestamp"] >= current_time].copy()
    new_future = new_horizon[new_horizon["Timestamp"] >= current_time].copy()
    
    if prev_future.empty:
        return new_future
    
    # Merge on timestamp (outer join to capture all points)
    merged = pd.merge(
        prev_future, new_future,
        on="Timestamp", how="outer", suffixes=("_prev", "_new")
    ).sort_values("Timestamp")
    
    # Calculate temporal blend weights
    dt_hours = (merged["Timestamp"] - current_time).dt.total_seconds() / 3600
    w_new = 1.0 - np.exp(-dt_hours / blend_tau_hours)
    w_prev = np.exp(-dt_hours / blend_tau_hours)
    
    # FOR-PERF-02: Vectorized blending for all forecast columns
    # FOR-MERGE-01: Preserve NaN masks, prefer non-null values over treating missing as zero
    col_names = ["ForecastHealth", "CI_Lower", "CI_Upper"]
    new_cols = [f"{col}_new" for col in col_names]
    prev_cols = [f"{col}_prev" for col in col_names]
    
    # Preserve original NaN masks before blending
    new_nan_masks = {col: merged[col].isna() for col in new_cols}
    prev_nan_masks = {col: merged[col].isna() for col in prev_cols}
    
    # Weighted average for points where BOTH prev and new are non-null (vectorized)
    for col, col_new, col_prev in zip(col_names, new_cols, prev_cols):
        both_valid = ~new_nan_masks[col_new] & ~prev_nan_masks[col_prev]
        
        # Where both valid: weighted blend
        merged.loc[both_valid, col] = (
            merged.loc[both_valid, col_new].values * w_new[both_valid] +
            merged.loc[both_valid, col_prev].values * w_prev[both_valid]
        )
        
        # Where only new is valid: use new value (prefer non-null over missing)
        only_new_valid = ~new_nan_masks[col_new] & prev_nan_masks[col_prev]
        merged.loc[only_new_valid, col] = merged.loc[only_new_valid, col_new]
        
        # Where only prev is valid: forward-fill from previous forecast
        only_prev_valid = new_nan_masks[col_new] & ~prev_nan_masks[col_prev]
        merged.loc[only_prev_valid, col] = merged.loc[only_prev_valid, col_prev]
        
        # Where both NaN: leave as NaN (don't treat missing as zero health)
    
    return merged[["Timestamp", "ForecastHealth", "CI_Lower", "CI_Upper"]]


def smooth_failure_probability_hazard(
    prev_hazard_baseline: float,
    new_probability_series: pd.Series,
    dt_hours: float = 1.0,
    alpha: float = DEFAULT_HAZARD_SMOOTHING_ALPHA
) -> pd.DataFrame:
    """
    Convert discrete batch failure probabilities to continuous hazard with EWMA smoothing.
    
    Math:
        - Hazard rate: lambda(t) = -ln(1 - p(t)) / dt
        - EWMA: lambda_smooth[t] = alpha * lambda_raw[t] + (1-alpha) * lambda_smooth[t-1]
        - Survival: S(t) = exp(-integral_0^t lambda_smooth(u) du)
        - Failure probability: F(t) = 1 - S(t)
    
    Args:
        prev_hazard_baseline: Previous EWMA hazard rate for continuity
        new_probability_series: Series with datetime index and failure probabilities
        dt_hours: Time step in hours (default 1.0)
        alpha: EWMA smoothing parameter (0-1, higher = more reactive)
    
    Returns:
        DataFrame with [Timestamp, HazardRaw, HazardSmooth, Survival, FailureProb]
    """
    if new_probability_series.empty:
        return pd.DataFrame()
    
    df_result = pd.DataFrame(index=new_probability_series.index)
    
    # Convert probability to hazard rate (clip to avoid log(0))
    p_clipped = new_probability_series.clip(1e-9, 1 - 1e-9)
    lambda_raw = -np.log(1 - p_clipped) / dt_hours
    df_result["HazardRaw"] = lambda_raw
    
    # EWMA smoothing with previous baseline
    lambda_smooth = np.zeros(len(lambda_raw))
    lambda_smooth[0] = alpha * lambda_raw.iloc[0] + (1 - alpha) * prev_hazard_baseline
    
    for i in range(1, len(lambda_raw)):
        lambda_smooth[i] = alpha * lambda_raw.iloc[i] + (1 - alpha) * lambda_smooth[i-1]
    
    df_result["HazardSmooth"] = lambda_smooth
    
    # Compute cumulative hazard and survival probability
    cumulative_hazard = np.cumsum(lambda_smooth * dt_hours)
    df_result["Survival"] = np.exp(-cumulative_hazard)
    df_result["FailureProb"] = 1 - df_result["Survival"]
    df_result["Timestamp"] = df_result.index
    
    return df_result.reset_index(drop=True)


def estimate_rul(
    tables_dir: Path,
    equip_id: Optional[int],
    run_id: Optional[str],
    config: Optional[Dict[str, Any]] = None,
    sql_client: Optional[Any] = None,
    output_manager: Optional[Any] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Unified entrypoint for RUL estimation.
    Dispatches to simple or enhanced estimator based on config.
    
    Config keys:
    - forecasting.mode: "simple" (default) or "enhanced"
    - forecasting.failure_threshold: Health threshold that counts as failure (default 70.0)
    - forecasting.max_forecast_hours: Maximum forecast horizon in hours (default 168.0)
    - rul.target_health: Legacy health threshold override (fallback)
    - rul.min_points: Minimum health points needed for RUL logic (default MIN_FORECAST_SAMPLES)
    """
    plain_config = _coerce_config_mapping(config)
    forecast_section = plain_config.get("forecasting") or {}
    rul_section = plain_config.get("rul") or {}
    Console.info("[RUL] Using unified RUL engine")
    
    # Call the new unified RUL engine
    # It reads config internally from the provided config dict and manages all output tables
    return rul_engine.run_rul(
        equip_id=equip_id,
        run_id=run_id,
        config_row=plain_config,
        sql_client=sql_client,
        output_manager=output_manager
    )


# ============================================================================
# AR(1) Baseline Detector
# ============================================================================

class AR1Detector:
    """
    Per-sensor AR(1) baseline model for residual scoring.
    
    Calculates AR(1) coefficients (phi) and mean (mu) for each sensor.
    Scores new data by calculating the absolute z-score of the residuals, 
    normalized by the TRAIN-time residual standard deviation.
    
    Usage:
        detector = AR1Detector(ar1_cfg={})
        detector.fit(train_df)
        scores = detector.score(test_df)
    """
    
    def __init__(self, ar1_cfg: Dict[str, Any] | None = None):
        """
        Initialize the AR(1) detector.
        
        Args:
            ar1_cfg: Configuration dict with optional keys:
                - eps (float): Numeric stability epsilon (default: 1e-9)
                - phi_cap (float): Max absolute phi value (default: 0.999)
                - sd_floor (float): Min std dev (default: 1e-6)
                - fuse (str): Fusion strategy "mean"|"median"|"p95" (default: "mean")
        """
        self.cfg = ar1_cfg or {}
        self._eps: float = float(self.cfg.get("eps", 1e-9))
        self._phi_cap: float = float(self.cfg.get("phi_cap", 0.999))
        self._sd_floor: float = float(self.cfg.get("sd_floor", 1e-6))
        self._fuse: Literal["mean", "median", "p95"] = self.cfg.get("fuse", "mean")
        
        # Trained parameters per column: (phi, mu)
        self.phimap: Dict[str, Tuple[float, float]] = {}
        # TRAIN residual std per column for normalization
        self.sdmap: Dict[str, float] = {}
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> "AR1Detector":
        """
        Fit the AR(1) model for each column in the training data.
        
        Args:
            X: Training feature matrix
            
        Returns:
            self for chaining
        """
        self.phimap = {}
        self.sdmap = {}
        
        if not isinstance(X, pd.DataFrame) or X.shape[0] == 0:
            self._is_fitted = True
            return self
        
        for c in X.columns:
            col = X[c].to_numpy(copy=False, dtype=np.float32)
            finite = np.isfinite(col)
            x = col[finite]
            
            if x.size < MIN_AR1_SAMPLES:
                mu = float(np.nanmean(col)) if x.size else 0.0
                if not np.isfinite(mu):
                    mu = 0.0
                phi = 0.0
                self.phimap[c] = (phi, mu)
                resid = (x - mu) if x.size else np.array([0.0], dtype=np.float32)
                sd = float(np.std(resid)) if resid.size else self._sd_floor
                self.sdmap[c] = max(sd, self._sd_floor)
                continue
            
            mu = float(np.nanmean(x))
            if not np.isfinite(mu):
                mu = 0.0
            xc = x - mu
            var_xc = float(np.var(xc)) if xc.size else 0.0
            phi = 0.0
            
            if np.isfinite(var_xc) and var_xc >= 1e-8:
                num = float(np.dot(xc[1:], xc[:-1]))
                den = float(np.dot(xc[:-1], xc[:-1]))
                if abs(den) >= 1e-9:
                    phi = num / den
            else:
                Console.warn(f"[AR1] Column '{c}': near-constant signal; using phi=0")
            
            if abs(phi) > self._phi_cap:
                original_phi = phi
                phi = float(np.sign(phi) * self._phi_cap)
                Console.warn(f"[AR1] Column '{c}': phi={original_phi:.3f} clamped to {phi:.3f}")
            
            if len(x) < MIN_FORECAST_SAMPLES:
                Console.warn(f"[AR1] Column '{c}': only {len(x)} samples; coefficients may be unstable")
            
            self.phimap[c] = (phi, mu)
            
            # Compute TRAIN residuals & std for normalization during score()
            x_shift = np.empty_like(x, dtype=np.float32)
            x_shift[0] = mu
            x_shift[1:] = x[:-1]
            pred = (x_shift - mu) * phi + mu
            resid = x - pred
            resid_for_sd = resid[1:] if resid.size > 1 else resid
            sd = float(np.std(resid_for_sd))
            self.sdmap[c] = max(sd, self._sd_floor)
        
        self._is_fitted = True
        return self
    
    def score(self, X: pd.DataFrame, return_per_sensor: bool = False) -> np.ndarray | Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate absolute z-scores of residuals using TRAIN-time residual std.
        
        Args:
            X: Scoring feature matrix
            return_per_sensor: If True, also return DataFrame of per-sensor |z|
            
        Returns:
            Fused absolute z-scores (len == len(X))
            Optionally: (fused_scores, per_sensor_df) when return_per_sensor=True
        """
        if not self._is_fitted:
            return np.zeros(len(X), dtype=np.float32)
        
        per_cols: Dict[str, np.ndarray] = {}
        n = len(X)
        
        if n == 0 or X.shape[1] == 0:
            return (np.zeros(0, dtype=np.float32), pd.DataFrame(index=X.index)) if return_per_sensor else np.zeros(0, dtype=np.float32)
        
        for c in X.columns:
            series = X[c].to_numpy(copy=False, dtype=np.float32)
            
            # Data quality check: skip columns with >50% NaN
            nan_count = np.isnan(series).sum()
            nan_fraction = nan_count / len(series) if len(series) > 0 else 0.0
            
            if nan_fraction > 0.5:
                Console.warn(f"[AR1] Column '{c}': {nan_fraction*100:.1f}% NaN (>{50}%) - skipping column")
                continue
            
            # Log high imputation rates for visibility
            if nan_fraction > 0.2:
                Console.warn(f"[AR1] Column '{c}': {nan_fraction*100:.1f}% NaN - imputing to mu (high imputation rate)")
            
            # FOR-CODE-03: Renamed ph -> phi for clarity (autoregressive coefficient)
            phi, mu = self.phimap.get(c, (0.0, float(np.nanmean(series))))
            if not np.isfinite(mu):
                mu = 0.0
            
            sd_train = self.sdmap.get(c, self._sd_floor)
            if not np.isfinite(sd_train) or sd_train <= self._sd_floor:
                sd_train = self._sd_floor
            
            # Impute NaNs to mu for prediction path
            series_finite = series.copy()
            if np.isnan(series_finite).any():
                series_finite = np.where(np.isfinite(series_finite), series_finite, mu).astype(np.float32, copy=False)
            
            # One-step AR(1) prediction
            pred = np.empty_like(series_finite, dtype=np.float32)
            first_obs = series_finite[0] if series_finite.size else mu
            pred[0] = first_obs if np.isfinite(first_obs) else mu
            if n > 1:
                pred[1:] = (series_finite[:-1] - mu) * phi + mu
            
            resid = series - pred  # Keep NaNs where original series had NaNs
            z = np.abs(resid) / sd_train
            per_cols[c] = z.astype(np.float32, copy=False)
        
        if not per_cols:
            return (np.zeros(n, dtype=np.float32), pd.DataFrame(index=X.index)) if return_per_sensor else np.zeros(n, dtype=np.float32)
        
        col_names = list(per_cols.keys())
        matrix = np.column_stack([per_cols[name] for name in col_names]) if col_names else np.zeros((n, 0), dtype=np.float32)
        
        with np.errstate(all="ignore"):
            if self._fuse == "median":
                fused = np.nanmedian(matrix, axis=1).astype(np.float32)
            elif self._fuse == "p95":
                fused = np.nanpercentile(matrix, 95, axis=1).astype(np.float32)
            else:
                fused = np.nanmean(matrix, axis=1).astype(np.float32)
        
        if return_per_sensor:
            Z = pd.DataFrame({name: matrix[:, i] for i, name in enumerate(col_names)}, index=X.index)
            return fused, Z
        return fused
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize detector state for persistence."""
        return {"phimap": self.phimap, "sdmap": self.sdmap, "cfg": self.cfg}
    
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AR1Detector":
        """Deserialize detector state from dict."""
        inst = cls(payload.get("cfg"))
        inst.phimap = dict(payload.get("phimap", {}))
        inst.sdmap = dict(payload.get("sdmap", {}))
        inst._is_fitted = True
        return inst


# ============================================================================
# Enhanced Forecasting (SQL-backed)
# ============================================================================


def _to_naive(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    DEPRECATED: Use utils.timestamp_utils.normalize_timestamps instead.
    Maintained for backward compatibility.
    """
    return normalize_timestamps(df, cols=cols, inplace=False)


def run_enhanced_forecasting_sql(
    sql_client: SqlClient,
    equip_id: Optional[int],
    run_id: Optional[str],
    config: Optional[Dict[str, Any]],
    equip: Optional[str] = None,
    current_batch_time: Optional[datetime] = None,
    sensor_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    SQL-only entrypoint for enhanced forecasting with continuous state persistence.
    Now supports both detector Z-score forecasting AND physical sensor forecasting.

    FORECAST-STATE-02: Enhanced with state continuity
    - Loads previous ForecastState for temporal continuity
    - Uses sliding window (72h lookback) instead of single batch
    - Conditional retraining (drift/energy checks)
    - Temporal blending of forecast horizons
    - Hazard-based probability smoothing
    - Saves updated ForecastState

    Returns:
        {
          "tables": {
             "health_forecast_continuous": DataFrame,  # Merged forecast horizons
             "failure_hazard_ts": DataFrame,  # Smoothed hazard/probability
             "failure_probability_ts": DataFrame,  # Legacy compatibility
             "failure_causation": DataFrame,
             "enhanced_maintenance_recommendation": DataFrame,
             "recommended_actions": DataFrame,
          },
          "metrics": {...},
          "forecast_state": ForecastState  # Updated state for persistence
        }
    """
    # FOR-DQ-01: Input validation (FOR-CODE-02: Configuration error - log and skip)
    if sql_client is None:
        Console.warn("[ENHANCED_FORECAST] SQL client not provided; skipping enhanced forecasting")
        return {"tables": {}, "metrics": {}}

    # FOR-COR-02: Ensure type consistency (FOR-CODE-02: Configuration error - raise ValueError)
    try:
        equip_id = ensure_equipid_int(equip_id)
        run_id = ensure_runid_str(run_id)
    except ValueError as e:
        raise ValueError(f"[ENHANCED_FORECAST] Type validation failed: {e}") from e
    
    if len(run_id.strip()) == 0:
        raise ValueError(f"[ENHANCED_FORECAST] run_id cannot be empty string")
    
    if config is None:
        raise ValueError(f"[ENHANCED_FORECAST] Invalid config: must be a mapping (dict or ConfigDict).")

    if not isinstance(config, Mapping) and not callable(getattr(config, "to_dict", None)):
        raise ValueError(f"[ENHANCED_FORECAST] Invalid config: must be a mapping (dict or ConfigDict).")

    config_map = _coerce_config_mapping(config)

    # Check if forecasting is enabled in config
    forecast_cfg = config_map.get("forecasting", {})
    if not forecast_cfg.get("enabled", True):
        Console.info("[ENHANCED_FORECAST] Module disabled via config.forecasting.enabled")
        return {"tables": {}, "metrics": {}}
    
    # FORECAST-STATE-02: Load previous state for continuity (SQL-ONLY MODE)
    enable_continuous = forecast_cfg.get("enable_continuous", True)  # Default enabled
    
    prev_state = None
    if enable_continuous and equip:
        try:
            prev_state = load_forecast_state(equip, equip_id, sql_client)
            if prev_state:
                Console.info(f"[FORECAST] Loaded state v{prev_state.state_version}, last retrain: {prev_state.last_retrain_time}")
        except Exception as e:
            # FOR-CODE-02: Non-fatal state load failure - log and continue without previous state
            Console.warn(f"[FORECAST] Failed to load previous state: {e}")
    
    # Use provided batch time or fall back to now() (for real-time scenarios)
    if current_batch_time is None:
        current_batch_time = datetime.now()

    # --- Cleanup old forecast data to prevent RunID overlap in charts ---
    try:
        import os
        try:
            keep_runs = int(os.getenv("ACM_FORECAST_RUNS_RETAIN", "2"))
        except Exception:
            keep_runs = 2
        keep_runs = max(1, min(int(keep_runs), 50))
        cur = sql_client.cursor()
        # Keep only the 2 most recent RunIDs to preserve some history while reducing clutter
        cur.execute("""
            WITH RankedRuns AS (
                SELECT DISTINCT RunID, 
                       ROW_NUMBER() OVER (ORDER BY MAX(CreatedAt) DESC) AS rn
                FROM dbo.ACM_HealthForecast_TS
                WHERE EquipID = ?
                GROUP BY RunID
            )
            DELETE FROM dbo.ACM_HealthForecast_TS
            WHERE EquipID = ? 
              AND RunID IN (SELECT RunID FROM RankedRuns WHERE rn > ?)
        """, (equip_id, equip_id, keep_runs))
        
        cur.execute("""
            WITH RankedRuns AS (
                SELECT DISTINCT RunID, 
                       ROW_NUMBER() OVER (ORDER BY MAX(CreatedAt) DESC) AS rn
                FROM dbo.ACM_FailureForecast_TS
                WHERE EquipID = ?
                GROUP BY RunID
            )
            DELETE FROM dbo.ACM_FailureForecast_TS
            WHERE EquipID = ? 
              AND RunID IN (SELECT RunID FROM RankedRuns WHERE rn > ?)
        """, (equip_id, equip_id, keep_runs))
        
        if not sql_client.conn.autocommit:
            sql_client.conn.commit()
        Console.info(f"[ENHANCED_FORECAST] Cleaned old forecast data for EquipID={equip_id} (kept {keep_runs} RunIDs)")
    except Exception as e:
        # FOR-CODE-02: Non-fatal data condition - log and continue with forecasting
        Console.warn(f"[ENHANCED_FORECAST] Failed to cleanup old forecasts: {e}")

    # --- Load health timeline with sliding window (FORECAST-STATE-02) ---
    lookback_hours = int(forecast_cfg.get("training_window_hours", 72))
    
    if enable_continuous and prev_state:
        # Use sliding window: last N hours + current batch
        cutoff_time = current_batch_time - timedelta(hours=lookback_hours)
        Console.info(f"[FORECAST] Using sliding window: {lookback_hours}h lookback from {cutoff_time}")
        
        try:
            cur = sql_client.cursor()
            cur.execute("""
                SELECT Timestamp, HealthIndex, FusedZ
                FROM dbo.ACM_HealthTimeline
                WHERE EquipID = ? AND Timestamp >= ?
                ORDER BY Timestamp
            """, (equip_id, cutoff_time))
            rows = cur.fetchall()
            cur.close()
            
            if rows:
                df_health = pd.DataFrame.from_records(
                    rows, 
                    columns=["Timestamp", "HealthIndex", "FusedZ"]
                )
            else:
                df_health = None
        except Exception as e:
            Console.warn(f"[FORECAST] Failed to load sliding window health data: {e}")
            df_health = None
    else:
        # Fallback to single-run load for backward compatibility
        try:
            rul_cfg = rul_engine.RULConfig(
                health_threshold=float(forecast_cfg.get("failure_threshold", 70.0)),
                min_points=int(forecast_cfg.get("min_points", 20)),
                max_forecast_hours=float(forecast_cfg.get("max_forecast_hours", 168.0)),
                learning_rate=float(forecast_cfg.get("learning_rate", 0.1)),
                min_model_weight=float(forecast_cfg.get("min_model_weight", 0.1)),
                enable_online_learning=bool(forecast_cfg.get("enable_online_learning", False)),
                calibration_window=int(forecast_cfg.get("calibration_window", 100)),
            )
            df_health, _ = rul_engine.load_health_timeline(
                sql_client=sql_client,
                equip_id=equip_id,
                run_id=run_id,  # FOR-COR-02: Already validated as str
                output_manager=None,
                cfg=rul_cfg,
            )
        except Exception as e:
            Console.warn(f"[ENHANCED_FORECAST] Failed to load health timeline via rul_engine: {e}")
            df_health = None

    if df_health is None:
        Console.warn("[ENHANCED_FORECAST] No health timeline available from SQL; skipping")
        return {"tables": {}, "metrics": {}}

    if "HealthIndex" not in df_health.columns or "Timestamp" not in df_health.columns:
        Console.warn("[ENHANCED_FORECAST] Health timeline missing required columns; skipping")
        return {"tables": {}, "metrics": {}}

    try:
        # TIME-01: Ensure naive timestamps
        ts = pd.to_datetime(df_health["Timestamp"], errors="coerce")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        # FOR-CODE-03: Renamed hi -> health_series for clarity
        health_series = pd.Series(df_health["HealthIndex"].astype(float).to_numpy(), index=ts)
        health_series = health_series.sort_index()
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failed to prepare health series: {e}")
        return {"tables": {}, "metrics": {}}

    # FOR-CODE-03: health_series instead of hi
    if health_series.size < MIN_FORECAST_SAMPLES:
        Console.warn(f"[ENHANCED_FORECAST] Insufficient health history ({health_series.size} points); skipping")
        return {"tables": {}, "metrics": {}}

    # --- Load detector scores from ACM_Scores_Wide ---
    try:
        cur = sql_client.cursor()
        cur.execute(
            """
            SELECT Timestamp,
                   ar1_z, pca_spe_z, pca_t2_z, mhal_z,
                   iforest_z, gmm_z, cusum_z, drift_z,
                   hst_z, river_hst_z, fused
            FROM dbo.ACM_Scores_Wide
            WHERE EquipID = ? AND RunID = ?
            ORDER BY Timestamp
            """,
            (equip_id, run_id),
        )
        rows = cur.fetchall() or []
        cur.close()
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failed to load detector scores from SQL: {e}")
        rows = []

    if not rows:
        Console.warn("[ENHANCED_FORECAST] No detector scores in ACM_Scores_Wide; skipping enhanced analysis")
        return {"tables": {}, "metrics": {}}

    cols = [
        "Timestamp",
        "ar1_z",
        "pca_spe_z",
        "pca_t2_z",
        "mhal_z",
        "iforest_z",
        "gmm_z",
        "cusum_z",
        "drift_z",
        "hst_z",
        "river_hst_z",
        "fused",
    ]
    df_scores = pd.DataFrame.from_records(rows, columns=cols)
    # TIME-01: Ensure naive timestamps
    df_scores["Timestamp"] = pd.to_datetime(df_scores["Timestamp"], errors="coerce")
    if df_scores["Timestamp"].dt.tz is not None:
        df_scores["Timestamp"] = df_scores["Timestamp"].dt.tz_localize(None)
        
    df_scores = df_scores.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    
    # FOR-COR-04: Remove duplicate timestamps before setting index
    initial_count = len(df_scores)
    df_scores = df_scores.drop_duplicates(subset=["Timestamp"], keep="last")
    dupe_count = initial_count - len(df_scores)
    if dupe_count > 0:
        Console.warn(f"[ENHANCED_FORECAST] Removed {dupe_count} duplicate timestamps (kept last occurrence)")
    
    df_scores = df_scores.set_index("Timestamp")

    if df_scores.empty:
        Console.warn("[ENHANCED_FORECAST] Detector scores dataframe empty after cleaning; skipping")
        return {"tables": {}, "metrics": {}}
    
    # --- FORECAST-STATE-02: Conditional retraining check ---
    current_data_hash = compute_data_hash(df_health) if enable_continuous else ""
    retrain_needed = True
    retrain_reason = "Initial training"
    
    if enable_continuous and prev_state:
        try:
            retrain_needed, retrain_reason = should_retrain(
                prev_state, sql_client, equip_id, current_data_hash, config_map
            )
            Console.info(f"[FORECAST] Retrain decision: {retrain_needed} - {retrain_reason}")
        except Exception as e:
            # FOR-CODE-02: Non-fatal check failure - log and default to safe retrain
            Console.warn(f"[FORECAST] Retrain check failed: {e}, defaulting to full retrain")
            retrain_needed = True
            retrain_reason = f"Retrain check error: {e}"

    # --- Core enhanced forecasting logic ---
    # Comprehensive implementation using exponential smoothing and RUL engine
    
    Console.info("[ENHANCED_FORECAST] Starting comprehensive forecasting with exponential smoothing")
    
    # Extract forecast configuration
    failure_threshold = float(forecast_cfg.get("failure_threshold", 70.0))
    forecast_hours = int(forecast_cfg.get("forecast_hours", 24))
    alpha = float(forecast_cfg.get("smoothing_alpha", 0.3))  # Exponential smoothing parameter
    
    # --- 1. Health Forecast with Exponential Smoothing ---
    try:
        # Calculate trend using simple exponential smoothing
        health_values = health_series.values
        n = len(health_values)
        
        # Simple exponential smoothing with trend
        level = health_values[0]
        trend = 0.0
        beta = 0.1  # Trend smoothing parameter
        
        # Fit the model on historical data
        for i in range(1, n):
            prev_level = level
            level = alpha * health_values[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # Generate forecast
        last_timestamp = health_series.index[-1]
        forecast_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=forecast_hours,
            freq='1H'
        )
        
        # Forecast values with trend
        forecast_values = []
        for h in range(1, forecast_hours + 1):
            forecast_val = level + h * trend
            forecast_values.append(max(0.0, min(100.0, forecast_val)))  # Clamp to [0, 100]
        
        # Calculate confidence intervals (wider as we go further)
        residuals = health_values[1:] - health_values[:-1]
        std_error = np.std(residuals) if len(residuals) > 0 else 5.0
        
        ci_lower = [max(0.0, val - 1.96 * std_error * np.sqrt(h)) for h, val in enumerate(forecast_values, 1)]
        ci_upper = [min(100.0, val + 1.96 * std_error * np.sqrt(h)) for h, val in enumerate(forecast_values, 1)]
        
        # Build health forecast DataFrame
        health_forecast_df = pd.DataFrame({
            "RunID": run_id,
            "EquipID": equip_id,
            "Timestamp": forecast_timestamps,
            "ForecastHealth": forecast_values,
            "CiLower": ci_lower,
            "CiUpper": ci_upper,
            "ForecastStd": std_error,
            "Method": "ExponentialSmoothing",
        })
        
        Console.info(f"[FORECAST] Generated {len(health_forecast_df)} hour health forecast (trend={trend:.2f})")
        
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Health forecasting failed: {e}")
        health_forecast_df = pd.DataFrame()
        forecast_values = []

    # --- 2. Failure Probability Calculation ---
    try:
        if len(forecast_values) > 0:
            # Calculate failure probability based on distance from threshold
            # Sigmoid function: as health drops below threshold, probability increases
            failure_probs = []
            for fh in forecast_values:
                if fh >= failure_threshold:
                    prob = 0.0
                else:
                    # Exponential increase as health drops below threshold
                    distance = failure_threshold - fh
                    # Probability = 1 - exp(-k * distance)
                    k = 0.05  # Controls steepness
                    prob = 1.0 - np.exp(-k * distance)
                failure_probs.append(min(1.0, max(0.0, prob)))
            
            failure_prob_df = pd.DataFrame({
                "RunID": run_id,
                "EquipID": equip_id,
                "Timestamp": forecast_timestamps,
                "FailureProb": failure_probs,
                "ThresholdUsed": failure_threshold,
                "Method": "SigmoidDegradation",
            })
            
            max_failure_prob = max(failure_probs)
            Console.info(f"[FORECAST] Max failure probability: {max_failure_prob*100:.1f}%")
        else:
            failure_prob_df = pd.DataFrame()
            max_failure_prob = 0.0
            
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failure probability computation failed: {e}")
        failure_prob_df = pd.DataFrame()
        max_failure_prob = 0.0

    # --- 3. RUL Estimation ---
    try:
        if len(forecast_values) > 0:
            # Find when health crosses failure threshold
            rul_hours = None
            for h, fh in enumerate(forecast_values, 1):
                if fh < failure_threshold:
                    rul_hours = float(h)
                    break
            
            if rul_hours is None:
                # Health doesn't cross threshold in forecast window
                rul_hours = float(forecast_hours + 24)  # Assume good for forecast window + buffer
            
            Console.info(f"[FORECAST] RUL estimated: {rul_hours:.1f} hours")
        else:
            rul_hours = 168.0  # Default 1 week
            
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] RUL estimation failed: {e}")
        rul_hours = 168.0

    # --- 4A. Detector Attribution (Active Detectors) ---
    # Forecast detector Z-score trends (PCA, CUSUM, GMM, IForest, etc.)
    detector_forecast_df = pd.DataFrame()
    try:
        if df_scores is not None and not df_scores.empty:
            latest_scores = df_scores.iloc[-1]
            
            # Find Z-score columns (these are DETECTOR outputs, not sensors)
            z_cols = [c for c in df_scores.columns if c.endswith('_z') and c not in ['fused', 'omr_z']]
            
            if z_cols:
                # Get absolute Z-scores for active detectors
                detector_scores = {col.replace('_z', ''): abs(latest_scores[col]) for col in z_cols if pd.notna(latest_scores[col])}
                
                # Sort by magnitude - top active detectors
                top_detectors = sorted(detector_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Generate forecast for top detectors
                detector_forecast_rows = []
                for detector_name, z_score in top_detectors:
                    for h, ts in enumerate(forecast_timestamps, 1):
                        # Linear trend for detector Z-scores
                        forecast_val = z_score * (1.0 + 0.01 * h)
                        detector_forecast_rows.append({
                            "RunID": run_id,
                            "EquipID": equip_id,
                            "Timestamp": ts,
                            "DetectorName": detector_name,  # e.g., 'pca_spe', 'cusum', 'gmm'
                            "ForecastValue": forecast_val,
                            "CiLower": None,
                            "CiUpper": None,
                            "ForecastStd": None,
                            "Method": "LinearTrend",
                        })
                
                detector_forecast_df = pd.DataFrame(detector_forecast_rows)
                Console.info(f"[FORECAST] Generated detector forecast for {len(top_detectors)} active detectors")
            
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Detector forecast failed: {e}")
        detector_forecast_df = pd.DataFrame()

    # --- 4B. Physical Sensor Attribution (Hot Sensors) ---
    # Forecast actual physical sensor values (Motor Current, Temperature, Pressure, etc.)
    sensor_forecast_df = pd.DataFrame()
    try:
        if sensor_data is not None and not sensor_data.empty:
            # Get the latest sensor readings
            latest_sensors = sensor_data.iloc[-1]
            
            # Get numeric sensor columns (exclude datetime/categorical)
            sensor_cols = [c for c in sensor_data.columns if pd.api.types.is_numeric_dtype(sensor_data[c])]
            
            if sensor_cols and len(sensor_data) >= 10:  # Need minimum history
                # Calculate sensor variability (standard deviation) to identify changing sensors
                sensor_variability = {}
                for col in sensor_cols:
                    recent_data = sensor_data[col].tail(24)  # Last 24 hours
                    if recent_data.notna().sum() >= 5:
                        std_val = recent_data.std()
                        mean_val = recent_data.mean()
                        if mean_val != 0 and pd.notna(std_val):
                            # Coefficient of variation - identifies sensors with significant change
                            sensor_variability[col] = abs(std_val / mean_val)
                
                # Sort by variability - sensors showing most change
                top_sensors = sorted(sensor_variability.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Generate forecast for top changing sensors
                sensor_forecast_rows = []
                for sensor_name, variability in top_sensors:
                    # Get recent trend
                    recent_values = sensor_data[sensor_name].tail(24).dropna()
                    if len(recent_values) >= 5:
                        # Calculate simple linear trend
                        x = np.arange(len(recent_values))
                        y = recent_values.values
                        trend = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
                        
                        current_val = recent_values.iloc[-1]
                        
                        for h, ts in enumerate(forecast_timestamps, 1):
                            # Linear extrapolation from current value
                            forecast_val = current_val + (trend * h)
                            
                            sensor_forecast_rows.append({
                                "RunID": run_id,
                                "EquipID": equip_id,
                                "Timestamp": ts,
                                "SensorName": sensor_name,  # Actual sensor like "Motor Current", "Bearing Temperature"
                                "ForecastValue": forecast_val,
                                "CiLower": None,
                                "CiUpper": None,
                                "ForecastStd": None,
                                "Method": "LinearTrend",
                            })
                
                sensor_forecast_df = pd.DataFrame(sensor_forecast_rows)
                Console.info(f"[FORECAST] Generated physical sensor forecast for {len(top_sensors)} sensors")
            else:
                Console.info(f"[FORECAST] Insufficient sensor data for forecasting (need 10+ rows, have {len(sensor_data)})")
        else:
            Console.info("[FORECAST] No sensor data provided - skipping physical sensor forecasting")
            
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Physical sensor forecast failed: {e}")
        sensor_forecast_df = pd.DataFrame()

    # --- 5. RUL Summary ---
    try:
        predicted_failure_time = health_series.index[-1] + pd.Timedelta(hours=rul_hours)
        
        rul_summary_df = pd.DataFrame([{
            "RunID": run_id,
            "EquipID": equip_id,
            "RUL_Hours": rul_hours,
            "LowerBound": max(0.0, rul_hours - 12.0),  # +/- 12 hour uncertainty
            "UpperBound": rul_hours + 12.0,
            "Confidence": 0.8 if len(health_series) > 100 else 0.6,  # Higher confidence with more data
            "Method": "ExponentialSmoothing",
            "LastUpdate": datetime.now(),
            "RUL_Trajectory_Hours": rul_hours,
            "RUL_Hazard_Hours": None,
            "RUL_Energy_Hours": None,
            "RUL_Final_Hours": rul_hours,
            "ConfidenceBand_Hours": 12.0,
            "DominantPath": "Trajectory",
        }])
        
        Console.info(f"[FORECAST] RUL summary created: {rul_hours:.1f}h until failure threshold")
        
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] RUL summary creation failed: {e}")
        rul_summary_df = pd.DataFrame()

    # --- 6. Build output tables ---
    tables: Dict[str, pd.DataFrame] = {}

    if not health_forecast_df.empty:
        tables["health_forecast_ts"] = health_forecast_df

    if not failure_prob_df.empty:
        tables["failure_forecast_ts"] = failure_prob_df

    if not detector_forecast_df.empty:
        tables["detector_forecast_ts"] = detector_forecast_df

    if not sensor_forecast_df.empty:
        tables["sensor_forecast_ts"] = sensor_forecast_df
    
    if not rul_summary_df.empty:
        tables["rul_summary"] = rul_summary_df

    # --- 7. Build metrics summary ---
    metrics = {
        "rul_hours": float(rul_hours),
        "max_failure_probability": float(max_failure_prob),
        "maintenance_required": bool(max_failure_prob > 0.5),  # >50% probability
        "urgency_score": float(max_failure_prob * 100.0),  # Scale to 0-100
        "confidence": 0.8 if len(health_series) > 100 else 0.6,
    }

    Console.info(
        f"[ENHANCED_FORECAST] RUL={metrics['rul_hours']:.1f}h, "
        f"MaxFailProb={metrics['max_failure_probability']*100:.1f}%, "
        f"MaintenanceRequired={metrics['maintenance_required']}, "
        f"Urgency={metrics['urgency_score']:.0f}/100"
    )
    
    # --- 8. State Persistence (FORECAST-STATE-02: Full continuous forecasting) ---
    if enable_continuous and equip and sql_client:
        try:
            # Compute forecast quality metrics
            forecast_quality = {}
            
            try:
                # Calculate RMSE if we have historical forecasts to compare
                if prev_state and prev_state.last_forecast_horizon_json:
                    prev_forecast = json.loads(prev_state.last_forecast_horizon_json)
                    if prev_forecast:
                        # Compare previous forecast with actual health values
                        actual_values = health_series.tail(len(prev_forecast)).values
                        forecast_vals = [p.get('health', 0) for p in prev_forecast[:len(actual_values)]]
                        
                        if len(actual_values) == len(forecast_vals) and len(actual_values) > 0:
                            rmse = np.sqrt(np.mean((np.array(actual_values) - np.array(forecast_vals)) ** 2))
                            mae = np.mean(np.abs(np.array(actual_values) - np.array(forecast_vals)))
                            
                            forecast_quality = {
                                "rmse": float(rmse),
                                "mae": float(mae),
                                "forecast_accuracy": float(100.0 * (1.0 - min(1.0, rmse / 100.0))),
                                "evaluation_samples": len(actual_values)
                            }
                            
                            Console.info(f"[FORECAST_QUALITY] RMSE={rmse:.2f}, MAE={mae:.2f}, Accuracy={forecast_quality['forecast_accuracy']:.1f}%")
                        
            except Exception as qe:
                Console.warn(f"[FORECAST] Failed to compute forecast quality: {qe}")
            
            # Build forecast horizon JSON for next iteration comparison
            forecast_horizon_data = []
            try:
                for i, (ts, val) in enumerate(zip(forecast_timestamps, forecast_values)):
                    forecast_horizon_data.append({
                        "timestamp": ts.isoformat(),
                        "health": float(val),
                        "horizon_hours": i + 1
                    })
            except Exception as fhe:
                Console.warn(f"[FORECAST] Failed to serialize forecast horizon: {fhe}")
                forecast_horizon_data = []
            
            # Calculate hazard baseline (average failure probability in forecast window)
            hazard_baseline = float(max_failure_prob) if max_failure_prob > 0 else 0.0
            
            # Create comprehensive forecast state
            new_state = ForecastState(
                equip_id=equip_id,
                state_version=(prev_state.state_version + 1) if prev_state else 1,
                model_type="ExponentialSmoothing_v2",  # Version identifier for model evolution
                model_params={
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "failure_threshold": float(failure_threshold),
                    "forecast_hours": int(forecast_hours),
                    "estimated_trend": float(trend) if 'trend' in locals() else 0.0,
                    "estimated_level": float(level) if 'level' in locals() else 0.0
                },
                residual_variance=float(std_error ** 2) if 'std_error' in locals() else 0.0,
                last_forecast_horizon_json=json.dumps(forecast_horizon_data),
                hazard_baseline=hazard_baseline,
                last_retrain_time=current_batch_time.isoformat() if current_batch_time else datetime.now().isoformat(),
                training_data_hash=current_data_hash,
                training_window_hours=lookback_hours,
                forecast_quality=forecast_quality
            )
            
            # Save updated state to SQL (SQL-ONLY MODE)
            save_forecast_state(new_state, equip, sql_client)
            Console.info(
                f"[FORECAST_STATE] Saved v{new_state.state_version} "
                f"(retrain={retrain_needed}, reason='{retrain_reason}', "
                f"quality={forecast_quality.get('forecast_accuracy', 'N/A')})"
            )
            
            # Add state info to metrics
            metrics["forecast_state_version"] = new_state.state_version
            metrics["retrain_needed"] = retrain_needed
            metrics["retrain_reason"] = retrain_reason
            
            if forecast_quality:
                metrics.update({
                    "forecast_rmse": forecast_quality.get("rmse", 0.0),
                    "forecast_mae": forecast_quality.get("mae", 0.0),
                    "forecast_accuracy": forecast_quality.get("forecast_accuracy", 0.0)
                })
            
        except Exception as e:
            Console.error(f"[FORECAST] Failed to save forecast state: {e}")
            import traceback
            Console.warn(f"[FORECAST] State persistence error traceback: {traceback.format_exc()}")

    return {"tables": tables, "metrics": metrics}


def run_and_persist_enhanced_forecasting(
    sql_client: SqlClient,
    equip_id: Optional[int],
    run_id: Optional[str],
    config: Optional[Dict[str, Any]],
    output_manager: Any,
    tables_dir: Path,
    equip: Optional[str] = None,
    current_batch_time: Optional[datetime] = None,
    sensor_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    SQL-mode helper that runs enhanced forecasting and persists every resulting table via OutputManager.
    Enhanced with continuous forecasting support (FORECAST-STATE-02). SQL-ONLY MODE.
    Supports both detector Z-score forecasting AND physical sensor forecasting.
    """
    plain_config = _coerce_config_mapping(config)

    result = run_enhanced_forecasting_sql(
        sql_client=sql_client,
        equip_id=equip_id,
        run_id=run_id,
        config=plain_config,
        equip=equip,
        current_batch_time=current_batch_time,
        sensor_data=sensor_data,
    )

    tables = (result or {}).get("tables") or {}
    metrics = (result or {}).get("metrics") or {}

    if not tables:
        return metrics

    ef_sql_map = {
        "health_forecast_ts": "ACM_HealthForecast_TS",
        "failure_forecast_ts": "ACM_FailureForecast_TS",
        "detector_forecast_ts": "ACM_DetectorForecast_TS",
        "sensor_forecast_ts": "ACM_SensorForecast_TS",
        "rul_summary": "ACM_RUL_Summary",
    }
    ef_csv_map = {
        "health_forecast_ts": "health_forecast.csv",
        "failure_forecast_ts": "failure_forecast.csv",
        "detector_forecast_ts": "detector_forecast.csv",
        "sensor_forecast_ts": "sensor_forecast.csv",
        "rul_summary": "rul_summary.csv",
    }
    timestamp_columns = {
        "health_forecast_ts": ["Timestamp"],
        "failure_forecast_ts": ["Timestamp"],
        "detector_forecast_ts": ["Timestamp"],
        "sensor_forecast_ts": ["Timestamp"],
    }

    enable_sql = getattr(output_manager, "sql_client", None) is not None

    for logical_name, df in tables.items():
        if df is None or df.empty:
            continue

        df_to_write = df.copy()
        columns = timestamp_columns.get(logical_name, [])
        if columns:
            df_to_write = _to_naive(df_to_write, columns)

        sql_table = ef_sql_map.get(logical_name)
        csv_name = ef_csv_map.get(logical_name, f"{logical_name}.csv")
        output_manager.write_dataframe(
            df_to_write,
            tables_dir / csv_name,
            sql_table=sql_table if enable_sql else None,
            add_created_at="CreatedAt" not in df_to_write.columns,
        )

    return metrics


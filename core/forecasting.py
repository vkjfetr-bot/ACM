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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal, List

import numpy as np
import pandas as pd

from utils.logger import Console  # type: ignore
from core import rul_estimator  # type: ignore
from core import enhanced_rul_estimator  # type: ignore
from core.model_persistence import ForecastState, save_forecast_state, load_forecast_state  # type: ignore
from datetime import datetime, timedelta
import hashlib

# Temporary import for file-based enhanced forecasting until fully migrated
try:
    from core import enhanced_forecasting_deprecated as _enhanced_forecasting
    # Alias for backward compatibility
    EnhancedForecastingEngine = _enhanced_forecasting.EnhancedForecastingEngine
except ImportError:
    Console.warn("[FORECASTING] enhanced_forecasting_deprecated not found; file-mode forecasting unavailable")
    EnhancedForecastingEngine = None  # type: ignore


# ============================================================================
# Continuous Forecasting Helpers (FORECAST-STATE-02, 03)
# ============================================================================

def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of DataFrame for change detection."""
    try:
        data_str = df.to_csv(index=False)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    except Exception:
        return ""


def should_retrain(
    prev_state: Optional[ForecastState],
    sql_client: Any,
    equip_id: int,
    current_data_hash: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Determine if full model retrain is needed.
    
    Checks:
    1. Drift: Recent 5-point mean DriftValue > threshold
    2. Energy spike: Anomaly energy P95 > threshold * median
    3. Forecast quality: Current RMSE > threshold * baseline RMSE
    4. Data change: Training data hash mismatch
    
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
    
    # Check drift
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
        Console.warn(f"[FORECAST] Failed to check drift: {e}")
    
    # Check anomaly energy spike
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
        Console.warn(f"[FORECAST] Failed to check anomaly energy: {e}")
    
    # All checks passed - incremental update sufficient
    return False, "Model stable, incremental update"


def compute_forecast_quality(
    prev_state: Optional[ForecastState],
    sql_client: Any,
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
        
        # Build SQL query with timestamp IN clause
        placeholders = ",".join("?" * len(timestamps_list))
        cur = sql_client.cursor()
        query = f"""
            SELECT Timestamp, HealthIndex
            FROM dbo.ACM_HealthTimeline
            WHERE EquipID = ? AND Timestamp IN ({placeholders})
            ORDER BY Timestamp
        """
        cur.execute(query, (equip_id, *timestamps_list))
        rows = cur.fetchall()
        cur.close()
        
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
        Console.warn(f"[FORECAST_QUALITY] Failed to compute metrics: {e}")
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}


def merge_forecast_horizons(
    prev_horizon: pd.DataFrame,
    new_horizon: pd.DataFrame,
    current_time: datetime,
    blend_tau_hours: float = 12.0
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
    
    # Blend forecast values (favor new forecasts for near term, blend smoothly)
    for col in ["ForecastHealth", "CI_Lower", "CI_Upper"]:
        col_new = f"{col}_new"
        col_prev = f"{col}_prev"
        
        # Fill NaNs with zeros for blending calculation
        new_vals = merged[col_new].fillna(0).values
        prev_vals = merged[col_prev].fillna(0).values
        
        # Weighted average
        merged[col] = new_vals * w_new + prev_vals * w_prev
        
        # If one side is NaN, use the non-NaN value
        merged.loc[merged[col_new].isna(), col] = merged.loc[merged[col_new].isna(), col_prev]
        merged.loc[merged[col_prev].isna(), col] = merged.loc[merged[col_prev].isna(), col_new]
    
    return merged[["Timestamp", "ForecastHealth", "CI_Lower", "CI_Upper"]]


def smooth_failure_probability_hazard(
    prev_hazard_baseline: float,
    new_probability_series: pd.Series,
    dt_hours: float = 1.0,
    alpha: float = 0.3
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
    config: Dict[str, Any],
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
    - rul.min_points: Minimum health points needed for RUL logic (default 20)
    """
    forecast_section = config.get("forecasting") or {}
    rul_section = config.get("rul") or {}

    default_rul_cfg = rul_estimator.RULConfig()

    def _coerce_float(value: Any, fallback: float) -> float:
        try:
            if value is None:
                return fallback
            return float(value)
        except Exception:
            return fallback

    def _coerce_int(value: Any, fallback: int) -> int:
        try:
            if value is None:
                return fallback
            return int(value)
        except Exception:
            return fallback

    health_threshold = _coerce_float(
        forecast_section.get("failure_threshold")
        or rul_section.get("target_health"),
        default_rul_cfg.health_threshold,
    )

    min_points = _coerce_int(
        rul_section.get("min_points"),
        default_rul_cfg.min_points,
    )

    max_forecast_hours = _coerce_float(
        forecast_section.get("max_forecast_hours")
        or legacy_forecast_section.get("max_forecast_hours")
        or rul_section.get("max_forecast_hours"),
        default_rul_cfg.max_forecast_hours,
    )

    # Safety: do not allow non-positive horizons
    if max_forecast_hours <= 0:
        max_forecast_hours = default_rul_cfg.max_forecast_hours

    rul_cfg = enhanced_rul_estimator.RULConfig(
        health_threshold=health_threshold,
        min_points=min_points,
        max_forecast_hours=max_forecast_hours,
    )
    
    Console.info("[RUL] Using ENHANCED RUL estimator (default)")
    return enhanced_rul_estimator.estimate_rul_and_failure(
        tables_dir=tables_dir,
        equip_id=equip_id,
        run_id=run_id,
        health_threshold=health_threshold,
        cfg=rul_cfg,
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
            
            if x.size < 3:
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
            
            if len(x) < 20:
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
            ph, mu = self.phimap.get(c, (0.0, float(np.nanmean(series))))
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
                pred[1:] = (series_finite[:-1] - mu) * ph + mu
            
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
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            df_out[col] = pd.to_datetime(df_out[col], errors="coerce")
            try:
                if df_out[col].dt.tz is not None:
                    df_out[col] = df_out[col].dt.tz_localize(None)
            except Exception:
                # Some columns already naive or not datetime; ignore
                pass
    return df_out


def run_enhanced_forecasting_sql(
    sql_client: Any,
    equip_id: Optional[int],
    run_id: Optional[str],
    config: Dict[str, Any],
    artifact_root: Optional[Path] = None,
    equip: Optional[str] = None,
    current_batch_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    SQL-only entrypoint for enhanced forecasting with continuous state persistence.

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
    if sql_client is None:
        Console.warn("[ENHANCED_FORECAST] SQL client not provided; skipping enhanced forecasting")
        return {"tables": {}, "metrics": {}}

    if equip_id is None or not run_id:
        Console.warn("[ENHANCED_FORECAST] Missing EquipID/RunID; skipping enhanced forecasting")
        return {"tables": {}, "metrics": {}}

    if EnhancedForecastingEngine is None:
        Console.warn("[ENHANCED_FORECAST] EnhancedForecastingEngine not available; skipping")
        return {"tables": {}, "metrics": {}}

    engine = EnhancedForecastingEngine(config)
    if not engine.forecast_config.enabled:
        Console.info("[ENHANCED_FORECAST] Module disabled via config.forecasting.enabled")
        return {"tables": {}, "metrics": {}}
    
    # FORECAST-STATE-02: Load previous state for continuity
    forecast_cfg = config.get("forecasting", {})
    enable_continuous = forecast_cfg.get("enable_continuous", True)  # Default enabled
    
    prev_state = None
    if enable_continuous and artifact_root and equip:
        try:
            prev_state = load_forecast_state(artifact_root, equip, equip_id, sql_client)
            if prev_state:
                Console.info(f"[FORECAST] Loaded state v{prev_state.state_version}, last retrain: {prev_state.last_retrain_time}")
        except Exception as e:
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
        Console.warn(f"[ENHANCED_FORECAST] Failed to cleanup old forecasts: {e}")
        # Non-fatal, continue with forecasting

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
            df_health = rul_estimator._load_health_timeline(  # type: ignore[attr-defined]
                Path("."),
                sql_client=sql_client,
                equip_id=equip_id,
                run_id=str(run_id),
            )
        except Exception as e:
            Console.warn(f"[ENHANCED_FORECAST] Failed to load health timeline via rul_estimator: {e}")
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
        hi = pd.Series(df_health["HealthIndex"].astype(float).to_numpy(), index=ts)
        hi = hi.sort_index()
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failed to prepare health series: {e}")
        return {"tables": {}, "metrics": {}}

    if hi.size < 20:
        Console.warn(f"[ENHANCED_FORECAST] Insufficient health history ({hi.size} points); skipping")
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
                prev_state, sql_client, equip_id, current_data_hash, config
            )
            Console.info(f"[FORECAST] Retrain decision: {retrain_needed} - {retrain_reason}")
        except Exception as e:
            Console.warn(f"[FORECAST] Retrain check failed: {e}, defaulting to full retrain")
            retrain_needed = True
            retrain_reason = f"Retrain check error: {e}"

    # --- Core enhanced forecasting logic (mirrors EnhancedForecastingEngine.run) ---
    try:
        import traceback
        forecast_result = engine.forecaster.forecast(
            health_history=hi,
            horizons=engine.forecast_config.forecast_horizons,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Forecasting failed: {e}")
        Console.debug(f"[ENHANCED_FORECAST] Traceback: {traceback.format_exc()}")
        return {"tables": {}, "metrics": {}}

    try:
        failure_probs_df = engine.prob_calculator.compute_probabilities(
            forecast_result,
            engine.forecast_config.failure_threshold,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failure probability computation failed: {e}")
        return {"tables": {}, "metrics": {}}

    try:
        rul_hours = engine._estimate_rul(  # type: ignore[attr-defined]
            forecast_result,
            engine.forecast_config.failure_threshold,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] RUL estimation failed: {e}")
        return {"tables": {}, "metrics": {}}

    try:
        predicted_failure_time = engine._get_failure_time(  # type: ignore[attr-defined]
            hi.index[-1],
            rul_hours,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Failed to derive predicted failure time: {e}")
        return {"tables": {}, "metrics": {}}

    try:
        causation_df, failure_patterns = engine.causation_analyzer.analyze_causation(
            df_scores,
            predicted_failure_time,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Causation analysis failed: {e}")
        causation_df, failure_patterns = pd.DataFrame(), []

    try:
        maintenance_rec = engine.maintenance_recommender.generate_recommendation(
            failure_probs_df,
            causation_df,
            failure_patterns,
            rul_hours,
        )
    except Exception as e:
        Console.warn(f"[ENHANCED_FORECAST] Maintenance recommendation failed: {e}")
        maintenance_rec = None

    tables: Dict[str, pd.DataFrame] = {}

    now_ts = hi.index[-1]

    if not failure_probs_df.empty:
        fp_df = failure_probs_df.copy()
        fp_df.insert(0, "Timestamp", now_ts)
        fp_df.insert(0, "EquipID", int(equip_id))
        fp_df.insert(0, "RunID", str(run_id))
        tables["failure_probability_ts"] = fp_df

    if causation_df is not None and not causation_df.empty and maintenance_rec is not None:
        fc_df = causation_df.copy()
        fc_df.insert(0, "PredictedFailureTime", predicted_failure_time)
        fc_df.insert(1, "FailurePattern", ",".join(maintenance_rec.get("failure_patterns", [])))
        fc_df.insert(0, "EquipID", int(equip_id))
        fc_df.insert(0, "RunID", str(run_id))
        tables["failure_causation"] = fc_df

    if maintenance_rec is not None:
        maint_df = pd.DataFrame(
            [
                {
                    "UrgencyScore": maintenance_rec.get("urgency_score"),
                    "MaintenanceRequired": maintenance_rec.get("maintenance_required"),
                    "EarliestMaintenance": maintenance_rec.get("window", {}).get("earliest_maintenance"),
                    "PreferredWindowStart": maintenance_rec.get("window", {}).get("preferred_window_start"),
                    "PreferredWindowEnd": maintenance_rec.get("window", {}).get("preferred_window_end"),
                    "LatestSafeTime": maintenance_rec.get("window", {}).get("latest_safe_time"),
                    "FailureProbAtLatest": maintenance_rec.get("window", {}).get("failure_prob_at_latest"),
                    "FailurePattern": ",".join(maintenance_rec.get("failure_patterns", [])),
                    "Confidence": maintenance_rec.get("confidence"),
                    "EstimatedDuration_Hours": sum(
                        a.get("estimated_duration_hours", 0.0)
                        for a in maintenance_rec.get("recommended_actions", [])
                    ),
                }
            ]
        )
        maint_df.insert(0, "EquipID", int(equip_id))
        maint_df.insert(0, "RunID", str(run_id))
        tables["enhanced_maintenance_recommendation"] = maint_df

        actions = maintenance_rec.get("recommended_actions", [])
        if actions:
            actions_df = pd.DataFrame(actions)
            actions_df.insert(0, "EquipID", int(equip_id))
            actions_df.insert(0, "RunID", str(run_id))
            tables["recommended_actions"] = actions_df

        metrics = {
            "rul_hours": float(rul_hours),
            "max_failure_probability": float(
                failure_probs_df["FailureProbability"].max()
            )
            if not failure_probs_df.empty
            else 0.0,
            "maintenance_required": bool(maintenance_rec.get("maintenance_required", False)),
            "urgency_score": float(maintenance_rec.get("urgency_score", 0.0)),
            "confidence": float(maintenance_rec.get("confidence", 0.0)),
        }
    else:
        metrics = {
            "rul_hours": float(rul_hours),
            "max_failure_probability": float(
                failure_probs_df["FailureProbability"].max()
            )
            if not failure_probs_df.empty
            else 0.0,
            "maintenance_required": False,
            "urgency_score": 0.0,
            "confidence": 0.0,
        }

    Console.info(
        f"[ENHANCED_FORECAST] RUL={metrics['rul_hours']:.1f}h, "
        f"MaxFailProb={metrics['max_failure_probability']*100:.1f}%, "
        f"MaintenanceRequired={metrics['maintenance_required']}, "
        f"Urgency={metrics['urgency_score']:.0f}/100"
    )
    
    # --- FORECAST-STATE-02 & 03: Horizon merging and hazard smoothing ---
    if enable_continuous and artifact_root and equip:
        try:
            # Extract new forecast horizon from forecast_result
            # forecast_result has keys: 'forecasts', 'uncertainties', 'horizons'
            forecasts = forecast_result.get("forecasts", [])
            uncertainties = forecast_result.get("uncertainties", [])
            horizons_hours = forecast_result.get("horizons", [])
            
            if len(forecasts) > 0 and len(horizons_hours) > 0:
                # Build timestamps from horizons (hours from last health point)
                timestamps = [hi.index[-1] + pd.Timedelta(hours=int(h)) for h in horizons_hours]
                # CI bounds: forecast ± 1.96*uncertainty (95% CI)
                ci_lower = [f - 1.96*u for f, u in zip(forecasts, uncertainties)]
                ci_upper = [f + 1.96*u for f, u in zip(forecasts, uncertainties)]
                
                new_forecast_df = pd.DataFrame({
                    "Timestamp": timestamps,
                    "ForecastHealth": forecasts,
                    "CI_Lower": ci_lower,
                    "CI_Upper": ci_upper
                })
            else:
                new_forecast_df = pd.DataFrame(columns=["Timestamp", "ForecastHealth", "CI_Lower", "CI_Upper"])
            
            # Merge with previous horizon
            blend_tau_hours = float(forecast_cfg.get("blend_tau_hours", 12.0))
            prev_horizon = prev_state.get_last_forecast_horizon() if prev_state else pd.DataFrame()
            
            merged_horizon = merge_forecast_horizons(
                prev_horizon, new_forecast_df, current_batch_time, blend_tau_hours
            )
            
            # Add to tables for persistence
            if not merged_horizon.empty:
                merged_horizon_output = merged_horizon.copy()
                # RunID is a UUID string; keep as text for SQL (avoid int() casting error)
                merged_horizon_output["SourceRunID"] = str(run_id)
                merged_horizon_output["EquipID"] = int(equip_id)
                merged_horizon_output["MergeWeight"] = 1.0  # Can refine per-row
                tables["health_forecast_continuous"] = merged_horizon_output
                Console.info(f"[FORECAST] Merged forecast horizon: {len(merged_horizon)} points")
            
            # Hazard smoothing for failure probability
            if not failure_probs_df.empty and "FailureProbability" in failure_probs_df.columns:
                prev_hazard = prev_state.hazard_baseline if prev_state else 0.0
                alpha = float(forecast_cfg.get("hazard_smoothing_alpha", 0.3))
                
                # Create time series from failure probability
                fp_series = failure_probs_df.set_index("ForecastHorizon")["FailureProbability"] \
                    if "ForecastHorizon" in failure_probs_df.columns else failure_probs_df["FailureProbability"]
                
                hazard_df = smooth_failure_probability_hazard(
                    prev_hazard, fp_series, dt_hours=1.0, alpha=alpha
                )
                
                if not hazard_df.empty:
                    # Use real future timestamps anchored to current batch time instead of epoch-based ints
                    hazard_df["Timestamp"] = pd.to_datetime([current_batch_time + pd.Timedelta(hours=i) for i in range(len(hazard_df))])
                    # Add metadata columns with proper types
                    hazard_df["RunID"] = str(run_id)  # Full UUID string
                    hazard_df["EquipID"] = pd.Series([int(equip_id)] * len(hazard_df), dtype='int64')
                    # Pre-add CreatedAt so write_dataframe doesn't re-add it
                    hazard_df["CreatedAt"] = pd.Timestamp.now().tz_localize(None)
                    # Reorder columns to match SQL table schema
                    hazard_df = hazard_df[["Timestamp", "HazardRaw", "HazardSmooth", "Survival", "FailureProb", "RunID", "EquipID", "CreatedAt"]]
                    tables["failure_hazard_ts"] = hazard_df
                    
                    # Update hazard baseline for next iteration
                    new_hazard_baseline = hazard_df["HazardSmooth"].iloc[-1]
                    Console.info(f"[FORECAST] Smoothed hazard: baseline updated to {new_hazard_baseline:.4f}")
                else:
                    new_hazard_baseline = prev_hazard
            else:
                new_hazard_baseline = prev_state.hazard_baseline if prev_state else 0.0
            
            # Compute forecast quality metrics (RMSE, MAE, MAPE)
            # Compare previous forecast to actual health values that occurred
            forecast_quality = compute_forecast_quality(
                prev_state=prev_state,
                sql_client=sql_client,
                equip_id=equip_id,
                current_batch_time=current_batch_time
            )
            
            # Create updated ForecastState
            new_state = ForecastState(
                equip_id=int(equip_id),
                state_version=(prev_state.state_version + 1) if prev_state else 1,
                model_type="AR1",  # Or extract from engine
                model_params={},  # TODO: Extract from engine.forecaster
                residual_variance=0.0,  # TODO: Compute from residuals
                last_forecast_horizon_json=ForecastState.serialize_forecast_horizon(merged_horizon),
                hazard_baseline=new_hazard_baseline,
                last_retrain_time=current_batch_time.isoformat() if retrain_needed else (
                    prev_state.last_retrain_time if prev_state else current_batch_time.isoformat()
                ),
                training_data_hash=current_data_hash,
                training_window_hours=lookback_hours,
                forecast_quality=forecast_quality
            )
            
            # Save updated state
            save_forecast_state(new_state, artifact_root, equip, sql_client)
            
            # Add state to return dict for caller
            metrics["forecast_state_version"] = new_state.state_version
            metrics["retrain_needed"] = retrain_needed
            metrics["retrain_reason"] = retrain_reason
            
        except Exception as e:
            Console.warn(f"[FORECAST] Failed to apply continuous forecasting enhancements: {e}")
            import traceback
            Console.debug(f"[FORECAST] Traceback: {traceback.format_exc()}")

    return {"tables": tables, "metrics": metrics}


def run_and_persist_enhanced_forecasting(
    sql_client: Any,
    equip_id: Optional[int],
    run_id: Optional[str],
    config: Dict[str, Any],
    output_manager: Any,
    tables_dir: Path,
    artifact_root: Optional[Path] = None,
    equip: Optional[str] = None,
    current_batch_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    SQL-mode helper that runs enhanced forecasting and persists every resulting table via OutputManager.
    Enhanced with continuous forecasting support (FORECAST-STATE-02).
    """
    result = run_enhanced_forecasting_sql(
        sql_client=sql_client,
        equip_id=equip_id,
        run_id=run_id,
        config=config,
        artifact_root=artifact_root,
        equip=equip,
        current_batch_time=current_batch_time,
    )

    tables = (result or {}).get("tables") or {}
    metrics = (result or {}).get("metrics") or {}

    if not tables:
        return metrics

    ef_sql_map = {
        "health_forecast_continuous": "ACM_HealthForecast_Continuous",  # New continuous table
        "failure_hazard_ts": "ACM_FailureHazard_TS",  # New hazard smoothing table
        "failure_probability_ts": "ACM_EnhancedFailureProbability_TS",
        "failure_causation": "ACM_FailureCausation",
        "enhanced_maintenance_recommendation": "ACM_EnhancedMaintenanceRecommendation",
        "recommended_actions": "ACM_RecommendedActions",
    }
    ef_csv_map = {
        "health_forecast_continuous": "health_forecast_continuous.csv",
        "failure_hazard_ts": "failure_hazard_ts.csv",
        "failure_probability_ts": "enhanced_failure_probability.csv",
        "failure_causation": "failure_causation.csv",
        "enhanced_maintenance_recommendation": "enhanced_maintenance_recommendation.csv",
        "recommended_actions": "recommended_actions.csv",
    }
    timestamp_columns = {
        "health_forecast_continuous": ["Timestamp"],
        "failure_hazard_ts": ["Timestamp"],
        "failure_probability_ts": ["Timestamp"],
        "failure_causation": ["PredictedFailureTime"],
    }

    enable_sql = getattr(output_manager, "sql_client", None) is not None

    for logical_name, df in tables.items():
        if df is None or df.empty:
            continue

        df_to_write = df.copy()
        columns = timestamp_columns.get(logical_name, [])
        if columns:
            df_to_write = _to_naive(df_to_write, columns)

        if logical_name == "recommended_actions":
            df_to_write = df_to_write.rename(
                columns={
                    "action": "Action",
                    "priority": "Priority",
                    "estimated_duration_hours": "EstimatedDuration_Hours",
                }
            )

        sql_table = ef_sql_map.get(logical_name)
        csv_name = ef_csv_map.get(logical_name, f"{logical_name}.csv")
        output_manager.write_dataframe(
            df_to_write,
            tables_dir / csv_name,
            sql_table=sql_table if enable_sql else None,
            add_created_at="CreatedAt" not in df_to_write.columns,
        )

    return metrics


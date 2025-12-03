"""
Unified RUL Engine
==================

Single RUL estimation module combining:
- Ensemble modeling (AR1 + Exponential + Weibull)
- Online learning with SQL-backed state persistence
- Multipath RUL calculation (trajectory + hazard + energy)
- SQL-only inputs/outputs (no CSV fallbacks)

Replaces both rul_estimator.py and enhanced_rul_estimator.py.

Timestamp Policy:
-----------------
All timestamps are timezone-naive local time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.logger import Console


# ============================================================================
# Configuration & Utilities
# ============================================================================

@dataclass
class RULConfig:
    """Unified RUL configuration"""
    health_threshold: float = 70.0
    min_points: int = 20
    max_forecast_hours: float = 168.0
    # Ensemble settings
    learning_rate: float = 0.1
    min_model_weight: float = 0.05
    enable_online_learning: bool = True
    calibration_window: int = 10
    # Maintenance bands (hours)
    band_normal: float = 168.0  # > 1 week
    band_watch: float = 72.0    # 3-7 days
    band_plan: float = 24.0     # 1-3 days
    band_urgent: float = 12.0   # < 12 hours
    # Row limit for health timeline
    max_health_timeline_rows: int = 100000
    health_downsample_freq: str = "1min"


def norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF using exact scipy implementation"""
    return norm.cdf(z)


def ensure_runid_str(run_id: Any) -> str:
    """Normalize RunID to string"""
    if run_id is None:
        return ""
    return str(run_id)


def ensure_equipid_int(equip_id: Any) -> int:
    """Normalize EquipID to positive integer"""
    if equip_id is None:
        raise ValueError("EquipID cannot be None")
    val = int(equip_id)
    if val <= 0:
        raise ValueError(f"EquipID must be positive, got {val}")
    return val


# ============================================================================
# I/O Layer - SQL Only
# ============================================================================

def cleanup_old_forecasts(sql_client: Any, equip_id: int, keep_runs: int = 2) -> None:
    """
    Clean old forecast data to prevent RunID overlap in charts.
    Keeps N most recent RunIDs based on MAX(CreatedAt).
    """
    try:
        import os
        keep_runs = max(1, min(int(os.getenv("ACM_FORECAST_RUNS_RETAIN", str(keep_runs))), 50))
        
        cur = sql_client.cursor()
        for table in ["ACM_HealthForecast_TS", "ACM_FailureForecast_TS"]:
            cur.execute(
                f"""
                WITH RankedRuns AS (
                    SELECT DISTINCT RunID, 
                           ROW_NUMBER() OVER (ORDER BY MAX(CreatedAt) DESC) AS rn
                    FROM dbo.{table}
                    WHERE EquipID = ?
                    GROUP BY RunID
                )
                DELETE FROM dbo.{table}
                WHERE EquipID = ? 
                  AND RunID IN (SELECT RunID FROM RankedRuns WHERE rn > ?)
                """,
                (equip_id, equip_id, keep_runs),
            )
        
        if not sql_client.conn.autocommit:
            sql_client.conn.commit()
        Console.info(f"[RUL] Cleaned old forecast data for EquipID={equip_id} (kept {keep_runs} RunIDs)")
    except Exception as e:
        Console.warn(f"[RUL] Failed to cleanup old forecasts: {e}")


def load_health_timeline(
    sql_client: Optional[Any],
    equip_id: int,
    run_id: str,
    output_manager: Optional[Any],
    cfg: RULConfig,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load health timeline with SQL-first priority.
    Returns: (DataFrame, data_quality_flag)
    
    Priority:
    1. OutputManager cache (in-memory artifact)
    2. SQL query (ACM_HealthTimeline)
    NO CSV FALLBACK.
    
    Data quality flags: 'OK', 'SPARSE', 'GAPPY', 'FLAT', 'MISSING'
    """
    # Try cache first
    if output_manager is not None:
        df = output_manager.get_cached_table("health_timeline.csv")
        if df is not None:
            Console.info(f"[RUL] Using cached health_timeline ({len(df)} rows)")
            df = _normalize_health_timeline(df)
            return _apply_row_limit(df, cfg), _assess_data_quality(df)
    
    # Try SQL
    if sql_client is None:
        Console.warn("[RUL] No SQL client provided; cannot load health timeline")
        return None, "MISSING"
    
    try:
        cur = sql_client.cursor()
        cur.execute(
            """
            SELECT Timestamp, HealthIndex, FusedZ
            FROM dbo.ACM_HealthTimeline
            WHERE EquipID = ? AND RunID = ?
            ORDER BY Timestamp
            """,
            (equip_id, run_id),
        )
        rows = cur.fetchall()
        cur.close()
        
        if not rows:
            Console.warn(f"[RUL] No health timeline found for EquipID={equip_id}, RunID={run_id}")
            return None, "MISSING"
        
        df = pd.DataFrame.from_records(rows, columns=["Timestamp", "HealthIndex", "FusedZ"])
        Console.info(f"[RUL] Loaded {len(df)} health points from SQL")
        df = _normalize_health_timeline(df)
        return _apply_row_limit(df, cfg), _assess_data_quality(df)
        
    except Exception as e:
        Console.warn(f"[RUL] Failed to load health timeline from SQL: {e}")
        return None, "MISSING"


def _normalize_health_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and timestamps"""
    # Normalize timestamp column
    ts_col = None
    for col in ["Timestamp", "timestamp", "ts"]:
        if col in df.columns:
            ts_col = col
            break
    if ts_col is None:
        ts_col = df.columns[0]
    
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].dt.tz is not None:
        df[ts_col] = df[ts_col].dt.tz_localize(None)
    
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.rename(columns={ts_col: "Timestamp"})
    
    return df


def _apply_row_limit(df: pd.DataFrame, cfg: RULConfig) -> pd.DataFrame:
    """Apply row limit and downsample if needed"""
    if df is None or len(df) <= cfg.max_health_timeline_rows:
        return df
    
    Console.warn(
        f"[RUL] Health timeline has {len(df)} rows (limit={cfg.max_health_timeline_rows}); "
        f"downsampling to {cfg.health_downsample_freq}"
    )
    
    df = df.set_index("Timestamp").resample(cfg.health_downsample_freq).mean().dropna().reset_index()
    return df


def _assess_data_quality(df: pd.DataFrame) -> str:
    """Assess data quality and return flag"""
    if df is None or df.empty:
        return "MISSING"
    
    if len(df) < 20:
        return "SPARSE"
    
    # Check for large time gaps
    if "Timestamp" in df.columns:
        time_diffs = df["Timestamp"].diff().dt.total_seconds() / 3600.0  # hours
        if time_diffs.max() > 24:  # Gap > 24 hours
            return "GAPPY"
    
    # Check for flat signal
    if "HealthIndex" in df.columns:
        std = df["HealthIndex"].std()
        if std < 0.1:
            return "FLAT"
    
    return "OK"


def load_sensor_hotspots(
    sql_client: Optional[Any],
    equip_id: int,
    run_id: str,
) -> pd.DataFrame:
    """
    Load sensor hotspots from SQL only.
    Returns DataFrame with columns: SensorName, FailureContribution, ZScoreAtFailure, AlertCount
    """
    if sql_client is None:
        Console.warn("[RUL] No SQL client; cannot load sensor hotspots")
        return pd.DataFrame()

    def _run_query(sql: str) -> pd.DataFrame:
        cur = sql_client.cursor()
        try:
            cur.execute(sql, (equip_id, run_id))
            rows = cur.fetchall()
            columns = [col[0] for col in (cur.description or [])]
            return pd.DataFrame.from_records(rows, columns=columns)
        finally:
            cur.close()

    base_query = """
        SELECT SensorName,
               FailureContribution,
               ZScoreAtFailure,
               AlertCount,
               MaxAbsZ,
               MaxSignedZ,
               LatestSignedZ,
               AboveAlertCount
        FROM dbo.ACM_SensorHotspots
        WHERE EquipID = ? AND RunID = ?
        ORDER BY FailureContribution DESC
    """

    fallback_query = """
        SELECT SensorName,
               MaxAbsZ,
               MaxSignedZ,
               LatestSignedZ,
               AboveAlertCount
        FROM dbo.ACM_SensorHotspots
        WHERE EquipID = ? AND RunID = ?
        ORDER BY MaxAbsZ DESC
    """

    df: pd.DataFrame
    try:
        df = _run_query(base_query)
    except Exception as primary_err:
        Console.warn(
            f"[RUL] SensorHotspots schema missing attribution columns; deriving contributions instead ({primary_err})"
        )
        try:
            df = _run_query(fallback_query)
        except Exception as fallback_err:
            Console.warn(f"[RUL] Failed to load sensor hotspots: {fallback_err}")
            return pd.DataFrame()

    if df.empty:
        Console.warn(f"[RUL] No sensor hotspots found for EquipID={equip_id}, RunID={run_id}")
        return pd.DataFrame()

    # Derive required columns when legacy schema omits them OR when columns have NULL values
    if "FailureContribution" not in df.columns or df["FailureContribution"].isna().any():
        abs_vals = pd.to_numeric(df.get("MaxAbsZ"), errors="coerce").abs().fillna(0.0)
        total = abs_vals.sum()
        if total > 0:
            df["FailureContribution"] = (abs_vals / total).clip(lower=0.0)
        elif abs_vals.max() > 0:
            df["FailureContribution"] = (abs_vals / abs_vals.max()).clip(lower=0.0)
        else:
            df["FailureContribution"] = 0.0

    if "ZScoreAtFailure" not in df.columns or df["ZScoreAtFailure"].isna().any():
        z_source = df.get("MaxSignedZ")
        if z_source is None or (hasattr(z_source, "isna") and z_source.isna().all()):
            z_source = df.get("LatestSignedZ")
        if z_source is None:
            z_source = pd.Series([0.0] * len(df))
        df["ZScoreAtFailure"] = pd.to_numeric(z_source, errors="coerce").fillna(0.0)

    if "AlertCount" not in df.columns or df["AlertCount"].isna().any():
        alerts = df.get("AboveAlertCount")
        if alerts is None:
            alerts = pd.Series([0] * len(df))
        df["AlertCount"] = pd.to_numeric(alerts, errors="coerce").fillna(0).astype(int)

    result = df[["SensorName", "FailureContribution", "ZScoreAtFailure", "AlertCount"]].copy()
    result = result.sort_values("FailureContribution", ascending=False).reset_index(drop=True)
    Console.info(f"[RUL] Loaded {len(result)} sensor hotspots from SQL")
    return result


# ============================================================================
# Learning State - SQL Backed
# ============================================================================

@dataclass
class ModelPerformanceMetrics:
    """Per-model performance tracking"""
    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    recent_errors: List[float] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class LearningState:
    """
    Online learning state for RUL ensemble.
    Persisted in SQL (ACM_RUL_LearningState), no JSON files.
    """
    equip_id: int
    ar1_metrics: ModelPerformanceMetrics = field(default_factory=ModelPerformanceMetrics)
    exp_metrics: ModelPerformanceMetrics = field(default_factory=ModelPerformanceMetrics)
    weibull_metrics: ModelPerformanceMetrics = field(default_factory=ModelPerformanceMetrics)
    calibration_factor: float = 1.0
    last_updated: Optional[datetime] = None
    prediction_history: List[Dict[str, float]] = field(default_factory=list)
    
    def to_sql_dict(self) -> Dict[str, Any]:
        """Convert to SQL-friendly dict"""
        return {
            "EquipID": self.equip_id,
            "AR1_MAE": self.ar1_metrics.mae,
            "AR1_RMSE": self.ar1_metrics.rmse,
            "AR1_Bias": self.ar1_metrics.bias,
            "AR1_RecentErrors": json.dumps(self.ar1_metrics.recent_errors[-10:]),
            "AR1_Weight": self.ar1_metrics.weight,
            "Exp_MAE": self.exp_metrics.mae,
            "Exp_RMSE": self.exp_metrics.rmse,
            "Exp_Bias": self.exp_metrics.bias,
            "Exp_RecentErrors": json.dumps(self.exp_metrics.recent_errors[-10:]),
            "Exp_Weight": self.exp_metrics.weight,
            "Weibull_MAE": self.weibull_metrics.mae,
            "Weibull_RMSE": self.weibull_metrics.rmse,
            "Weibull_Bias": self.weibull_metrics.bias,
            "Weibull_RecentErrors": json.dumps(self.weibull_metrics.recent_errors[-10:]),
            "Weibull_Weight": self.weibull_metrics.weight,
            "CalibrationFactor": self.calibration_factor,
            "LastUpdated": self.last_updated or datetime.now(),
            "PredictionHistory": json.dumps(self.prediction_history[-10:]),
        }
    
    @classmethod
    def from_sql_dict(cls, equip_id: int, row: Dict[str, Any]) -> LearningState:
        """Load from SQL row"""
        ar1 = ModelPerformanceMetrics(
            mae=float(row.get("AR1_MAE", 0.0)),
            rmse=float(row.get("AR1_RMSE", 0.0)),
            bias=float(row.get("AR1_Bias", 0.0)),
            recent_errors=json.loads(row.get("AR1_RecentErrors", "[]")),
            weight=float(row.get("AR1_Weight", 1.0)),
        )
        exp = ModelPerformanceMetrics(
            mae=float(row.get("Exp_MAE", 0.0)),
            rmse=float(row.get("Exp_RMSE", 0.0)),
            bias=float(row.get("Exp_Bias", 0.0)),
            recent_errors=json.loads(row.get("Exp_RecentErrors", "[]")),
            weight=float(row.get("Exp_Weight", 1.0)),
        )
        weibull = ModelPerformanceMetrics(
            mae=float(row.get("Weibull_MAE", 0.0)),
            rmse=float(row.get("Weibull_RMSE", 0.0)),
            bias=float(row.get("Weibull_Bias", 0.0)),
            recent_errors=json.loads(row.get("Weibull_RecentErrors", "[]")),
            weight=float(row.get("Weibull_Weight", 1.0)),
        )
        
        return cls(
            equip_id=equip_id,
            ar1_metrics=ar1,
            exp_metrics=exp,
            weibull_metrics=weibull,
            calibration_factor=float(row.get("CalibrationFactor", 1.0)),
            last_updated=row.get("LastUpdated"),
            prediction_history=json.loads(row.get("PredictionHistory", "[]")),
        )


def load_learning_state(sql_client: Optional[Any], equip_id: int) -> LearningState:
    """Load learning state from SQL or create default"""
    if sql_client is None:
        return LearningState(equip_id=equip_id)
    
    try:
        cur = sql_client.cursor()
        cur.execute(
            "SELECT * FROM dbo.ACM_RUL_LearningState WHERE EquipID = ?",
            (equip_id,)
        )
        row = cur.fetchone()
        
        if row:
            # Read column names BEFORE closing cursor
            col_names = [desc[0] for desc in cur.description]
            cur.close()
            row_dict = dict(zip(col_names, row))
            state = LearningState.from_sql_dict(equip_id, row_dict)
            Console.info(f"[RUL-Learn] Loaded learning state for EquipID={equip_id}")
            return state
        else:
            cur.close()
            Console.info(f"[RUL-Learn] No learning state found for EquipID={equip_id}; using defaults")
            return LearningState(equip_id=equip_id)
            
    except Exception as e:
        Console.warn(f"[RUL-Learn] Failed to load learning state: {e}; using defaults")
        return LearningState(equip_id=equip_id)


def save_learning_state(sql_client: Optional[Any], state: LearningState) -> None:
    """Save learning state to SQL"""
    if sql_client is None:
        return
    
    try:
        data = state.to_sql_dict()
        cur = sql_client.cursor()
        
        # Upsert pattern
        cur.execute(
            """
            IF EXISTS (SELECT 1 FROM dbo.ACM_RUL_LearningState WHERE EquipID = ?)
                UPDATE dbo.ACM_RUL_LearningState
                SET AR1_MAE=?, AR1_RMSE=?, AR1_Bias=?, AR1_RecentErrors=?, AR1_Weight=?,
                    Exp_MAE=?, Exp_RMSE=?, Exp_Bias=?, Exp_RecentErrors=?, Exp_Weight=?,
                    Weibull_MAE=?, Weibull_RMSE=?, Weibull_Bias=?, Weibull_RecentErrors=?, Weibull_Weight=?,
                    CalibrationFactor=?, LastUpdated=?, PredictionHistory=?
                WHERE EquipID = ?
            ELSE
                INSERT INTO dbo.ACM_RUL_LearningState
                (EquipID, AR1_MAE, AR1_RMSE, AR1_Bias, AR1_RecentErrors, AR1_Weight,
                 Exp_MAE, Exp_RMSE, Exp_Bias, Exp_RecentErrors, Exp_Weight,
                 Weibull_MAE, Weibull_RMSE, Weibull_Bias, Weibull_RecentErrors, Weibull_Weight,
                 CalibrationFactor, LastUpdated, PredictionHistory)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["EquipID"],
                data["AR1_MAE"], data["AR1_RMSE"], data["AR1_Bias"], data["AR1_RecentErrors"], data["AR1_Weight"],
                data["Exp_MAE"], data["Exp_RMSE"], data["Exp_Bias"], data["Exp_RecentErrors"], data["Exp_Weight"],
                data["Weibull_MAE"], data["Weibull_RMSE"], data["Weibull_Bias"], data["Weibull_RecentErrors"], data["Weibull_Weight"],
                data["CalibrationFactor"], data["LastUpdated"], data["PredictionHistory"],
                data["EquipID"],  # For UPDATE WHERE clause
                # For INSERT
                data["EquipID"],
                data["AR1_MAE"], data["AR1_RMSE"], data["AR1_Bias"], data["AR1_RecentErrors"], data["AR1_Weight"],
                data["Exp_MAE"], data["Exp_RMSE"], data["Exp_Bias"], data["Exp_RecentErrors"], data["Exp_Weight"],
                data["Weibull_MAE"], data["Weibull_RMSE"], data["Weibull_Bias"], data["Weibull_RecentErrors"], data["Weibull_Weight"],
                data["CalibrationFactor"], data["LastUpdated"], data["PredictionHistory"],
            )
        )
        
        if not sql_client.conn.autocommit:
            sql_client.conn.commit()
        
        Console.info(f"[RUL-Learn] Saved learning state for EquipID={state.equip_id}")
        
    except Exception as e:
        Console.warn(f"[RUL-Learn] Failed to save learning state: {e}")


# ============================================================================
# Ensemble Models (RUL-REF-11 through RUL-REF-15)
# ============================================================================


class DegradationModel:
    """
    Base class for health degradation models.
    
    All models must implement fit() and predict() methods.
    fit() returns True if successful, False if insufficient data.
    """

    def __init__(self, name: str):
        self.name = name
        self.params = {}
        self.fit_succeeded = False

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        """
        Fit model to historical data.
        
        Args:
            timestamps: Historical timestamp index
            health_values: Historical health index values
            
        Returns:
            True if fit succeeded, False otherwise
        """
        raise NotImplementedError

    def predict(self, future_timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast future health values.
        
        Args:
            future_timestamps: Future timestamp index for prediction
            
        Returns:
            Tuple of (forecast_mean, forecast_std)
        """
        raise NotImplementedError


class AR1Model(DegradationModel):
    """
    First-order autoregressive model with drift correction.
    
    Models health as: h(t) = μ + φ*(h(t-1) - μ) + drift*t + ε
    where φ is AR coefficient, μ is mean level, and drift captures systematic trend.
    """

    def __init__(self):
        super().__init__("AR1")
        self.mu = 0.0
        self.phi = 0.0
        self.sigma = 1.0
        self.drift = 0.0
        self.last_value = 0.0
        self.last_time = None
        self.step_sec = 60.0

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        y = health_values
        if len(y) < 10:
            Console.warn(f"[{self.name}] Insufficient data points: {len(y)} < 10")
            return False

        # Center data
        self.mu = float(np.nanmean(y))
        yc = y - self.mu

        # Check variance
        var_yc = float(np.var(yc))
        if var_yc < 1e-8:
            Console.warn(f"[{self.name}] Insufficient variance: {var_yc}")
            return False

        # Estimate AR(1) coefficient with proper normalization
        n = len(yc) - 1
        cov = float(np.sum(yc[1:] * yc[:-1]) / n)  # Normalized covariance
        var = float(np.var(yc[:-1]))  # Proper variance calculation
        self.phi = np.clip(cov / (var + 1e-9), -0.99, 0.99)

        # Residual variance
        y_shift = np.concatenate([[self.mu], y[:-1]])
        pred = (y_shift - self.mu) * self.phi + self.mu
        resid = y - pred
        self.sigma = float(np.std(resid[1:])) if len(resid) > 1 else 1.0
        self.sigma = max(self.sigma, 0.1)

        # Detect systematic drift (recent trend)
        recent_window = min(len(y), 30)
        t = np.arange(recent_window)
        y_recent = y[-recent_window:]
        if len(y_recent) > 3:
            drift = np.polyfit(t, y_recent, 1)[0]
            self.drift = float(drift)
        else:
            self.drift = 0.0

        self.last_value = float(y[-1])
        self.last_time = timestamps[-1]

        # Infer sampling cadence
        deltas = np.diff(timestamps.values.astype("int64")) / 1e9
        self.step_sec = float(np.median(deltas))
        if self.step_sec <= 0:
            self.step_sec = 60.0

        self.fit_succeeded = True
        return True

    def predict(self, future_timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fit_succeeded:
            raise RuntimeError(f"{self.name} not fitted successfully")

        steps = len(future_timestamps)
        h = np.arange(1, steps + 1)

        # AR(1) forecast component
        if abs(self.phi) > 1e-9:
            phi_h = self.phi**h
            ar_component = phi_h * (self.last_value - self.mu)
        else:
            ar_component = np.zeros(steps)

        # Linear drift component
        drift_component = self.drift * h * (self.step_sec / 3600.0)

        forecast = self.mu + ar_component + drift_component

        # Growing uncertainty over time (AR component + drift uncertainty)
        if abs(self.phi) < 0.999:
            var_ar = (1 - self.phi ** (2 * h)) / (1 - self.phi**2 + 1e-9)
        else:
            var_ar = h
        
        # Drift adds quadratic uncertainty growth
        time_hours = h * (self.step_sec / 3600.0)
        var_drift = (time_hours ** 2) * 0.1  # Drift uncertainty grows quadratically
        var_mult = var_ar + var_drift

        var_mult = np.clip(var_mult, 1.0, 100.0)
        forecast_std = self.sigma * np.sqrt(var_mult)

        return forecast, forecast_std


class ExponentialDegradationModel(DegradationModel):
    """
    Exponential degradation model: h(t) = h0 * exp(-λ*t) + offset
    
    Suitable for systems with exponential decay patterns.
    """

    def __init__(self):
        super().__init__("Exponential")
        self.lambda_ = 0.0
        self.h0 = 0.0
        self.offset = 0.0
        self.sigma = 1.0
        self.last_time = None

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        y = health_values
        if len(y) < 10:
            Console.warn(f"[{self.name}] Insufficient data points: {len(y)} < 10")
            return False

        # Time in hours from start
        t = (timestamps - timestamps[0]).total_seconds().values / 3600.0

        # Fit exponential decay via log-linear regression
        # Use 5th percentile of recent values as asymptotic level estimate
        recent_window = min(len(y), 20)
        offset = float(np.percentile(y[-recent_window:], 5))  # Statistically justified offset
        y_shifted = y - offset

        if np.any(y_shifted <= 0):
            Console.warn(f"[{self.name}] Non-positive shifted values")
            return False

        log_y = np.log(y_shifted)

        # Linear regression on log scale
        A = np.vstack([t, np.ones(len(t))]).T
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
            self.lambda_ = float(-coeffs[0])  # Decay rate
            self.h0 = float(np.exp(coeffs[1]))
            self.offset = offset

            # Residual std in original scale
            pred_log = A @ coeffs
            pred = np.exp(pred_log) + offset
            self.sigma = float(np.std(y - pred))
            self.sigma = max(self.sigma, 0.1)

            self.last_time = timestamps[-1]
            self.fit_succeeded = True
            return True
        except Exception as e:
            Console.warn(f"[{self.name}] Fit failed: {e}")
            return False

    def predict(self, future_timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fit_succeeded:
            raise RuntimeError(f"{self.name} not fitted successfully")

        t = (future_timestamps - self.last_time).total_seconds().values / 3600.0

        # Exponential projection
        forecast = self.h0 * np.exp(-self.lambda_ * t) + self.offset

        # Uncertainty grows with time
        forecast_std = self.sigma * np.sqrt(1 + 0.1 * t)

        return forecast, forecast_std


class WeibullInspiredModel(DegradationModel):
    """
    Weibull-inspired power-law degradation: h(t) = h0 - k * t^β
    
    Suitable for non-linear failure acceleration patterns.
    """

    def __init__(self):
        super().__init__("Weibull")
        self.beta = 1.0
        self.k = 0.0
        self.h0 = 0.0
        self.sigma = 1.0
        self.last_time = None
        self.t_base = None

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> bool:
        from scipy.optimize import minimize_scalar

        y = health_values
        if len(y) < 15:
            Console.warn(f"[{self.name}] Insufficient data points: {len(y)} < 15")
            return False

        # Time in hours from start
        t = (timestamps - timestamps[0]).total_seconds().values / 3600.0

        # Fit power-law degradation via shape parameter optimization
        def objective(beta):
            if beta <= 0:
                return 1e10
            try:
                t_beta = t**beta
                A = np.vstack([t_beta, np.ones(len(t))]).T
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                pred = A @ coeffs
                return np.sum((y - pred) ** 2)
            except Exception:
                return 1e10

        # Optimize beta (shape parameter)
        result = minimize_scalar(objective, bounds=(0.5, 3.0), method="bounded")

        if not result.success:
            Console.warn(f"[{self.name}] Optimization failed")
            return False

        self.beta = float(result.x)
        t_beta = t**self.beta
        A = np.vstack([t_beta, np.ones(len(t))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        self.k = float(-coeffs[0])  # Degradation rate (positive)
        self.h0 = float(coeffs[1])

        pred = A @ coeffs
        self.sigma = float(np.std(y - pred))
        self.sigma = max(self.sigma, 0.1)

        self.last_time = timestamps[-1]
        self.t_base = timestamps[0]
        self.fit_succeeded = True
        return True

    def predict(self, future_timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fit_succeeded:
            raise RuntimeError(f"{self.name} not fitted successfully")

        t = (future_timestamps - self.t_base).total_seconds().values / 3600.0

        # Add lower bound to prevent negative health predictions
        forecast = np.maximum(self.h0 - self.k * (t**self.beta), 0.0)

        # Uncertainty grows non-linearly
        t_rel = (future_timestamps - self.last_time).total_seconds().values / 3600.0
        forecast_std = self.sigma * np.sqrt(1 + 0.15 * t_rel)

        return forecast, forecast_std


class RULModel:
    """
    Ensemble wrapper that orchestrates AR1 + Exponential + Weibull models.
    
    Combines predictions using learned weights from LearningState.
    """

    def __init__(self, cfg: RULConfig, learning_state: LearningState):
        self.cfg = cfg
        self.learning_state = learning_state

        # Instantiate models
        self.ar1 = AR1Model()
        self.exp = ExponentialDegradationModel()
        self.weibull = WeibullInspiredModel()

        self.models = [self.ar1, self.exp, self.weibull]
        self.fit_status = {m.name: False for m in self.models}

    def fit(self, timestamps: pd.DatetimeIndex, health_values: np.ndarray) -> Dict[str, bool]:
        """
        Fit all ensemble models.
        
        Returns:
            Dict mapping model name to fit success status
        """
        Console.info("[RUL-Ensemble] Fitting all models...")

        for model in self.models:
            try:
                success = model.fit(timestamps, health_values)
                self.fit_status[model.name] = success
                if success:
                    Console.info(f"[RUL-Ensemble] ✓ {model.name} fitted successfully")
                else:
                    Console.warn(f"[RUL-Ensemble] ✗ {model.name} failed to fit")
            except Exception as e:
                Console.warn(f"[RUL-Ensemble] ✗ {model.name} exception: {e}")
                self.fit_status[model.name] = False

        fitted_count = sum(self.fit_status.values())
        Console.info(f"[RUL-Ensemble] {fitted_count}/{len(self.models)} models fitted successfully")

        return self.fit_status

    def forecast(self, future_timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Generate ensemble forecast with adaptive weighting.
        
        Returns:
            Dict with keys:
            - mean: Ensemble mean forecast
            - std: Ensemble std forecast
            - per_model: Dict of per-model (mean, std) predictions
            - weights: Normalized weights used
            - fit_status: Which models succeeded
        """
        # Collect predictions from successfully fitted models
        predictions = []
        stds = []
        model_names = []

        for model in self.models:
            if not self.fit_status[model.name]:
                continue

            try:
                pred, std = model.predict(future_timestamps)
                predictions.append(pred)
                stds.append(std)
                model_names.append(model.name)
            except Exception as e:
                Console.warn(f"[RUL-Ensemble] {model.name} prediction failed: {e}")

        if not predictions:
            Console.warn("[RUL-Ensemble] No models available for prediction")
            # Return flat forecast
            n = len(future_timestamps)
            return {
                "mean": np.full(n, 70.0),
                "std": np.full(n, 10.0),
                "per_model": {},
                "weights": {},
                "fit_status": self.fit_status,
            }

        predictions = np.array(predictions)
        stds = np.array(stds)

        # Get weights from learning state
        raw_weights = np.ones(len(model_names))
        for i, name in enumerate(model_names):
            model_key = name.lower()
            if model_key == "ar1":
                raw_weights[i] = self.learning_state.ar1.weight
            elif model_key == "exponential":
                raw_weights[i] = self.learning_state.exp.weight
            elif model_key == "weibull":
                raw_weights[i] = self.learning_state.weibull.weight

        # Apply minimum weight floor
        min_weight = self.cfg.min_model_weight
        raw_weights = np.maximum(raw_weights, min_weight)

        # Normalize to sum to 1.0
        weights = raw_weights / (np.sum(raw_weights) + 1e-9)

        # Weighted ensemble
        ensemble_mean = np.sum(weights[:, None] * predictions, axis=0)
        
        # Ensemble variance combines prediction variance + disagreement variance
        # Prediction variance (weighted)
        ensemble_var_pred = np.sum(weights[:, None] * (stds**2), axis=0)
        
        # Disagreement variance (spread between models)
        disagreement = predictions - ensemble_mean[None, :]
        ensemble_var_disagreement = np.sum(weights[:, None] * (disagreement**2), axis=0)
        
        # Total ensemble variance
        ensemble_var = ensemble_var_pred + ensemble_var_disagreement
        ensemble_std = np.sqrt(ensemble_var)

        # Package per-model predictions
        per_model = {
            name: {"mean": predictions[i], "std": stds[i]}
            for i, name in enumerate(model_names)
        }

        # Package weights
        weights_dict = {name: float(weights[i]) for i, name in enumerate(model_names)}

        Console.info(f"[RUL-Ensemble] Weights: {weights_dict}")

        return {
            "mean": ensemble_mean,
            "std": ensemble_std,
            "per_model": per_model,
            "weights": weights_dict,
            "fit_status": self.fit_status,
        }


def compute_failure_distribution(
    t_future: np.ndarray,
    health_mean: np.ndarray,
    health_std: np.ndarray,
    threshold: float = 70.0,
) -> np.ndarray:
    """
    Compute probability of failure at each future time.
    
    P(failure at t) = P(health(t) < threshold)
                    = Φ((threshold - mean(t)) / std(t))
    
    where Φ is standard normal CDF.
    
    Args:
        t_future: Future time array (hours from now)
        health_mean: Forecast mean health values
        health_std: Forecast std health values
        threshold: Failure threshold
        
    Returns:
        Array of failure probabilities at each time
    """
    # Standardized distance to threshold
    z = (threshold - health_mean) / (health_std + 1e-9)
    
    # Probability that health < threshold
    failure_prob = norm_cdf(z)
    
    # Clip to valid probability range
    failure_prob = np.clip(failure_prob, 0.0, 1.0)
    
    return failure_prob


# ============================================================================
# Core RUL Engine (RUL-REF-18, RUL-REF-19)
# ============================================================================


def compute_rul(
    health_df: pd.DataFrame,
    cfg: RULConfig,
    learning_state: LearningState,
    data_quality_flag: str,
    current_time: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Core RUL computation engine.
    
    Args:
        health_df: Health timeline DataFrame (Timestamp, HealthIndex)
        cfg: Configuration
        learning_state: Current learning state
        data_quality_flag: Data quality assessment (OK/SPARSE/GAPPY/FLAT/MISSING)
        current_time: Current time (defaults to last timestamp in health_df)
    
    Returns:
        Dict with keys:
        - health_forecast: DataFrame (Timestamp, HealthIndex, CI_Lower, CI_Upper)
        - failure_curve: DataFrame (Timestamp, FailureProb)
        - rul_multipath: Dict (trajectory/hazard/energy RUL estimates)
        - model_diagnostics: Dict (weights, fit_status, per_model_forecasts)
        - data_quality: str flag
        - current_time: Timestamp
    """
    Console.info("[RUL] Starting RUL computation...")
    
    # Validate input
    if health_df is None or health_df.empty:
        Console.warn("[RUL] Empty health timeline, returning default forecast")
        return _default_rul_result(cfg, data_quality_flag, current_time)
    
    # Ensure Timestamp column exists
    if "Timestamp" not in health_df.columns:
        Console.error("[RUL] Health DataFrame missing 'Timestamp' column")
        return _default_rul_result(cfg, data_quality_flag, current_time)
    
    # Sort and deduplicate
    health_df = health_df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])
    
    # Set current time
    if current_time is None:
        current_time = health_df["Timestamp"].iloc[-1]
    
    # Convert to DatetimeIndex
    health_df = health_df.set_index("Timestamp")
    timestamps = health_df.index
    health_values = health_df["HealthIndex"].values
    
    # Check for sufficient data
    if len(health_values) < cfg.min_points:
        Console.warn(f"[RUL] Insufficient data points: {len(health_values)} < {cfg.min_points}")
        return _default_rul_result(cfg, data_quality_flag, current_time)
    
    # Check current health for recovery detection
    current_health = float(health_values[-1])
    healthy_threshold = 90.0  # Consider equipment healthy above 90%
    Console.info(f"[RUL] Current health: {current_health:.1f}%")
    
    # Recovery detection: If health is well above failure threshold, equipment is healthy
    if current_health >= healthy_threshold:
        Console.info(f"[RUL] Equipment is HEALTHY (health={current_health:.1f}% >= {healthy_threshold}%)")
        Console.info(f"[RUL] Setting RUL to maximum forecast horizon: {cfg.max_forecast_hours:.1f}h")
        # Return healthy state result with max RUL
        return _healthy_rul_result(health_df, cfg, data_quality_flag, current_time, current_health)
    
    # Detect sampling interval
    deltas = np.diff(timestamps.values.astype("int64")) / 1e9 / 3600.0  # hours
    sampling_interval_hours = float(np.median(deltas))
    Console.info(f"[RUL] Detected sampling interval: {sampling_interval_hours:.2f} hours")
    
    # Build future time index
    n_future_steps = int(cfg.max_forecast_hours / sampling_interval_hours)
    n_future_steps = max(n_future_steps, 10)  # At least 10 steps
    
    future_timestamps = pd.date_range(
        start=current_time,
        periods=n_future_steps + 1,
        freq=f"{sampling_interval_hours}h"
    )[1:]  # Exclude current time
    
    Console.info(f"[RUL] Forecasting {n_future_steps} steps ({cfg.max_forecast_hours:.1f} hours)")
    
    # Instantiate and fit ensemble model
    rul_model = RULModel(cfg, learning_state)
    fit_status = rul_model.fit(timestamps, health_values)
    
    if not any(fit_status.values()):
        Console.warn("[RUL] All models failed to fit")
        return _default_rul_result(cfg, data_quality_flag, current_time)
    
    # Generate ensemble forecast
    forecast_result = rul_model.forecast(future_timestamps)
    
    # Build health forecast DataFrame
    health_forecast_df = pd.DataFrame({
        "Timestamp": future_timestamps,
        "ForecastHealth": forecast_result["mean"],
        "CI_Lower": forecast_result["mean"] - 1.96 * forecast_result["std"],
        "CI_Upper": forecast_result["mean"] + 1.96 * forecast_result["std"],
    })
    
    # Compute failure distribution
    failure_probs = compute_failure_distribution(
        t_future=np.arange(len(future_timestamps)),
        health_mean=forecast_result["mean"],
        health_std=forecast_result["std"],
        threshold=cfg.health_threshold,
    )
    
    failure_curve_df = pd.DataFrame({
        "Timestamp": future_timestamps,
        "FailureProb": failure_probs,
        "ThresholdUsed": cfg.health_threshold,
    })
    
    # Compute multipath RUL
    rul_multipath = compute_rul_multipath(
        health_forecast=health_forecast_df,
        failure_curve=failure_curve_df,
        current_time=current_time,
        cfg=cfg,
    )
    
    # Package model diagnostics
    model_diagnostics = {
        "weights": forecast_result["weights"],
        "fit_status": forecast_result["fit_status"],
        "per_model": forecast_result["per_model"],
        "sampling_interval_hours": sampling_interval_hours,
    }
    
    Console.info(f"[RUL] Computation complete. RUL={rul_multipath['rul_final_hours']:.1f}h")
    
    return {
        "health_forecast": health_forecast_df,
        "failure_curve": failure_curve_df,
        "rul_multipath": rul_multipath,
        "model_diagnostics": model_diagnostics,
        "data_quality": data_quality_flag,
        "current_time": current_time,
    }


def compute_rul_multipath(
    health_forecast: pd.DataFrame,
    failure_curve: pd.DataFrame,
    current_time: pd.Timestamp,
    cfg: RULConfig,
) -> Dict[str, Any]:
    """
    Compute RUL via three independent paths and select dominant.
    
    Path 1 (Expected): Mean health forecast crosses failure threshold (50th percentile)
    Path 2 (Conservative): CI_Lower crosses threshold (2.5th percentile)
    Path 3 (Optimistic): CI_Upper crosses threshold (97.5th percentile)
    
    Args:
        health_forecast: DataFrame with [Timestamp, ForecastHealth, CI_Lower, CI_Upper]
        failure_curve: DataFrame with [Timestamp, FailureProb, ThresholdUsed]
        current_time: Current timestamp
        cfg: Configuration
    
    Returns:
        Dict with:
        - rul_trajectory_hours: RUL from expected/mean path (or None)
        - rul_hazard_hours: RUL from conservative path (or None, 2.5th percentile)
        - rul_energy_hours: RUL from optimistic path (or None, 97.5th percentile)
        - rul_final_hours: Selected RUL
        - lower_bound_hours: Lower confidence bound
        - upper_bound_hours: Upper confidence bound
        - dominant_path: Which path was used ("trajectory", "conservative", or "optimistic")
    """
    Console.info("[RUL-Multipath] Computing RUL via multiple paths...")

    def _hours_delta(ts_val, current):
        """Return hours between two timestamps, robust to pandas Series/arrays."""
        ts_scalar = pd.to_datetime(ts_val)
        current_scalar = pd.to_datetime(current)
        delta = ts_scalar - current_scalar
        # If delta is a Series/array, take first element
        if hasattr(delta, "__len__") and not isinstance(delta, (pd.Timestamp, datetime)):
            delta = delta.iloc[0] if hasattr(delta, "iloc") else delta[0]
        return delta.total_seconds() / 3600.0
    
    # Path 1: Trajectory crossing (mean forecast < threshold) - Expected RUL (50th percentile)
    rul_trajectory = None
    if health_forecast is not None and not health_forecast.empty:
        trajectory_crossing = health_forecast[
            health_forecast["ForecastHealth"] <= cfg.health_threshold
        ]
        if not trajectory_crossing.empty:
            t1 = trajectory_crossing.iloc[0]["Timestamp"]
            rul_trajectory = _hours_delta(t1, current_time)
            Console.info(f"[RUL-Multipath] Trajectory crossing (mean) at {t1}, RUL={rul_trajectory:.1f}h")
        else:
            Console.info("[RUL-Multipath] No trajectory crossing within forecast horizon")
    
    # Path 2: Conservative (2.5th percentile) - CI_Lower crossing (early warning)
    rul_conservative = None
    if health_forecast is not None and not health_forecast.empty:
        conservative_crossing = health_forecast[
            health_forecast["CI_Lower"] <= cfg.health_threshold
        ]
        if not conservative_crossing.empty:
            t2 = conservative_crossing.iloc[0]["Timestamp"]
            rul_conservative = _hours_delta(t2, current_time)
            Console.info(f"[RUL-Multipath] Conservative crossing (CI_Lower) at {t2}, RUL={rul_conservative:.1f}h")
        else:
            Console.info("[RUL-Multipath] No conservative crossing within forecast horizon")
    
    # Path 3: Optimistic (97.5th percentile) - CI_Upper crossing (late warning)
    rul_optimistic = None
    if health_forecast is not None and not health_forecast.empty:
        optimistic_crossing = health_forecast[
            health_forecast["CI_Upper"] <= cfg.health_threshold
        ]
        if not optimistic_crossing.empty:
            t3 = optimistic_crossing.iloc[0]["Timestamp"]
            rul_optimistic = _hours_delta(t3, current_time)
            Console.info(f"[RUL-Multipath] Optimistic crossing (CI_Upper) at {t3}, RUL={rul_optimistic:.1f}h")
        else:
            Console.info("[RUL-Multipath] No optimistic crossing within forecast horizon")
    
    # Select dominant RUL (use conservative estimate for safety-critical decisions)
    available_ruls = [r for r in [rul_trajectory, rul_conservative, rul_optimistic] if r is not None]
    
    if available_ruls:
        # Use conservative path (minimum RUL) for safety
        rul_final = min(available_ruls)
        
        # Determine dominant path
        if rul_final == rul_trajectory:
            dominant_path = "trajectory"
        elif rul_final == rul_conservative:
            dominant_path = "conservative"
        else:
            dominant_path = "optimistic"
    else:
        # No crossing detected, use max forecast horizon
        rul_final = cfg.max_forecast_hours
        dominant_path = "none"
        Console.info(f"[RUL-Multipath] No crossing detected, using max horizon: {rul_final:.1f}h")
    
    # Confidence bounds: reuse percentile crossings; fallback to +/-30%
    lower_bound = rul_conservative if rul_conservative is not None else max(0.0, rul_final * 0.7)
    upper_bound = rul_optimistic if rul_optimistic is not None else rul_final * 1.3
    
    result = {
        "rul_trajectory_hours": rul_trajectory,
        "rul_hazard_hours": rul_conservative,  # Conservative estimate
        "rul_energy_hours": rul_optimistic,  # Optimistic estimate
        "rul_final_hours": float(rul_final),
        "lower_bound_hours": float(lower_bound),
        "upper_bound_hours": float(upper_bound),
        "dominant_path": dominant_path,
    }
    
    Console.info(
        f"[RUL-Multipath] Final RUL={rul_final:.1f}h "
        f"(trajectory={rul_trajectory}, conservative={rul_conservative}, optimistic={rul_optimistic}, dominant={dominant_path})"
    )
    
    return result


def _default_rul_result(
    cfg: RULConfig,
    data_quality_flag: str,
    current_time: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    """
    Return default RUL result when computation fails.
    
    Used when:
    - Empty health timeline
    - Insufficient data points
    - All models fail to fit
    """
    if current_time is None:
        current_time = pd.Timestamp.now()
    
    # Generate empty forecast
    future_timestamps = pd.date_range(
        start=current_time,
        periods=2,
        freq="1H"
    )[1:]
    
    health_forecast_df = pd.DataFrame({
        "Timestamp": future_timestamps,
        "ForecastHealth": [cfg.health_threshold] * len(future_timestamps),
        "CI_Lower": [cfg.health_threshold - 10] * len(future_timestamps),
        "CI_Upper": [cfg.health_threshold + 10] * len(future_timestamps),
    })
    
    failure_curve_df = pd.DataFrame({
        "Timestamp": future_timestamps,
        "FailureProb": [0.0] * len(future_timestamps),
        "ThresholdUsed": [cfg.health_threshold] * len(future_timestamps),
    })
    
    rul_multipath = {
        "rul_trajectory_hours": None,
        "rul_hazard_hours": None,
        "rul_energy_hours": None,
        "rul_final_hours": cfg.max_forecast_hours,
        "lower_bound_hours": cfg.max_forecast_hours * 0.7,
        "upper_bound_hours": cfg.max_forecast_hours * 1.3,
        "dominant_path": "default",
    }
    
    model_diagnostics = {
        "weights": {},
        "fit_status": {},
        "per_model": {},
        "sampling_interval_hours": 1.0,
    }
    
    return {
        "health_forecast": health_forecast_df,
        "failure_curve": failure_curve_df,
        "rul_multipath": rul_multipath,
        "model_diagnostics": model_diagnostics,
        "data_quality": data_quality_flag,
        "current_time": current_time,
    }


def _healthy_rul_result(
    health_df: pd.DataFrame,
    cfg: RULConfig,
    data_quality_flag: str,
    current_time: pd.Timestamp,
    current_health: float,
) -> Dict[str, Any]:
    """
    Return RUL result for healthy equipment (health > 90%).
    
    When equipment has recovered or is in healthy state, we return:
    - Flat health forecast at current level
    - Zero failure probability
    - RUL set to max forecast horizon
    
    Args:
        health_df: Health timeline DataFrame (indexed by Timestamp)
        cfg: Configuration
        data_quality_flag: Data quality assessment
        current_time: Current timestamp
        current_health: Current health percentage
    
    Returns:
        Dict with same structure as compute_rul()
    """
    Console.info(f"[RUL-Healthy] Equipment recovered/healthy (health={current_health:.1f}%)")
    
    # Detect sampling interval from health_df
    timestamps = health_df.index
    deltas = np.diff(timestamps.values.astype("int64")) / 1e9 / 3600.0  # hours
    sampling_interval_hours = float(np.median(deltas)) if len(deltas) > 0 else 1.0
    
    # Build future time index
    n_future_steps = int(cfg.max_forecast_hours / sampling_interval_hours)
    n_future_steps = max(n_future_steps, 10)
    
    future_timestamps = pd.date_range(
        start=current_time,
        periods=n_future_steps + 1,
        freq=f"{sampling_interval_hours}H"
    )[1:]
    
    # Flat forecast at current health level (assuming stability)
    health_forecast_df = pd.DataFrame({
        "Timestamp": future_timestamps,
        "ForecastHealth": [current_health] * len(future_timestamps),
        "CI_Lower": [max(current_health - 5.0, 0.0)] * len(future_timestamps),
        "CI_Upper": [min(current_health + 5.0, 100.0)] * len(future_timestamps),
    })
    
    # Zero failure probability (equipment is healthy)
    failure_curve_df = pd.DataFrame({
        "Timestamp": future_timestamps,
        "FailureProb": [0.0] * len(future_timestamps),
        "ThresholdUsed": [cfg.health_threshold] * len(future_timestamps),
    })
    
    # Set RUL to maximum forecast horizon
    rul_multipath = {
        "rul_trajectory_hours": None,  # No crossing
        "rul_hazard_hours": None,       # No crossing
        "rul_energy_hours": None,       # No crossing
        "rul_final_hours": cfg.max_forecast_hours,
        "lower_bound_hours": cfg.max_forecast_hours,
        "upper_bound_hours": cfg.max_forecast_hours,
        "dominant_path": "healthy",
    }
    
    # Minimal diagnostics for healthy state
    model_diagnostics = {
        "weights": {"healthy_state": 1.0},
        "fit_status": {"healthy_state": True},
        "per_model": {},
        "sampling_interval_hours": sampling_interval_hours,
    }
    
    Console.info(f"[RUL-Healthy] RUL set to {cfg.max_forecast_hours:.1f}h (healthy state)")
    
    return {
        "health_forecast": health_forecast_df,
        "failure_curve": failure_curve_df,
        "rul_multipath": rul_multipath,
        "model_diagnostics": model_diagnostics,
        "data_quality": data_quality_flag,
        "current_time": current_time,
    }


# ============================================================================
# Output Builders (RUL-REF-22, RUL-REF-23, RUL-REF-24)
# ============================================================================


def make_health_forecast_ts(
    health_forecast: pd.DataFrame,
    run_id: str,
    equip_id: int,
) -> pd.DataFrame:
    """
    Build ACM_HealthForecast_TS compatible DataFrame.
    
    Args:
        health_forecast: DataFrame from compute_rul (Timestamp, ForecastHealth, CI_Lower, CI_Upper)
        run_id: Run ID
        equip_id: Equipment ID
    
    Returns:
        DataFrame with columns: RunID, EquipID, Timestamp, HealthIndex, CI_Lower, CI_Upper
    """
    df = health_forecast.copy()
    df["RunID"] = run_id
    df["EquipID"] = equip_id
    df["HealthIndex"] = df["ForecastHealth"]
    
    # Reorder columns
    df = df[["RunID", "EquipID", "Timestamp", "HealthIndex", "CI_Lower", "CI_Upper"]]
    
    return df


def make_failure_forecast_ts(
    failure_curve: pd.DataFrame,
    run_id: str,
    equip_id: int,
) -> pd.DataFrame:
    """
    Build ACM_FailureForecast_TS compatible DataFrame.
    
    Args:
        failure_curve: DataFrame from compute_rul (Timestamp, FailureProb, ThresholdUsed)
        run_id: Run ID
        equip_id: Equipment ID
    
    Returns:
        DataFrame with columns: RunID, EquipID, Timestamp, FailureProb, ThresholdUsed
    """
    df = failure_curve.copy()
    df["RunID"] = run_id
    df["EquipID"] = equip_id
    
    # Reorder columns
    df = df[["RunID", "EquipID", "Timestamp", "FailureProb", "ThresholdUsed"]]
    
    return df


def make_rul_ts(
    health_forecast: pd.DataFrame,
    rul_multipath: Dict[str, Any],
    current_time: pd.Timestamp,
    run_id: str,
    equip_id: int,
    confidence: float,
) -> pd.DataFrame:
    """
    Build ACM_RUL_TS compatible DataFrame (RUL time series).
    
    Args:
        health_forecast: DataFrame with forecast
        rul_multipath: Multipath RUL result
        current_time: Current timestamp
        run_id: Run ID
        equip_id: Equipment ID
        confidence: Confidence score
    
    Returns:
        DataFrame with columns: RunID, EquipID, Timestamp, RUL_Hours, LowerBound, UpperBound, Confidence
    """
    # Compute RUL at each forecast step
    timestamps = health_forecast["Timestamp"].values
    rul_hours = []
    
    rul_final = rul_multipath["rul_final_hours"]
    lower_bound = rul_multipath["lower_bound_hours"]
    upper_bound = rul_multipath["upper_bound_hours"]
    
    for ts in timestamps:
        elapsed_hours = (pd.Timestamp(ts) - current_time).total_seconds() / 3600
        remaining_rul = max(0.0, rul_final - elapsed_hours)
        rul_hours.append(remaining_rul)
    
    df = pd.DataFrame({
        "RunID": run_id,
        "EquipID": equip_id,
        "Timestamp": timestamps,
        "RUL_Hours": rul_hours,
        "LowerBound": [max(0.0, lower_bound - (pd.Timestamp(ts) - current_time).total_seconds() / 3600) for ts in timestamps],
        "UpperBound": [upper_bound - (pd.Timestamp(ts) - current_time).total_seconds() / 3600 for ts in timestamps],
        "Confidence": confidence,
        "Method": rul_multipath.get("dominant_path", "Multipath"),
    })
    
    return df


def make_rul_summary(
    rul_multipath: Dict[str, Any],
    model_diagnostics: Dict[str, Any],
    data_quality: str,
    run_id: str,
    equip_id: int,
    confidence: float,
) -> pd.DataFrame:
    """
    Build ACM_RUL_Summary compatible DataFrame (single row summary).
    
    Args:
        rul_multipath: Multipath RUL result
        model_diagnostics: Model diagnostics from compute_rul
        data_quality: Data quality flag
        run_id: Run ID
        equip_id: Equipment ID
        confidence: Confidence score
    
    Returns:
        DataFrame with single row containing RUL summary
    """
    weights = model_diagnostics.get("weights", {})
    
    summary = {
        "RunID": run_id,
        "EquipID": equip_id,
        "RUL_Hours": rul_multipath["rul_final_hours"],
        "LowerBound": rul_multipath["lower_bound_hours"],
        "UpperBound": rul_multipath["upper_bound_hours"],
        "Confidence": confidence,
        "Method": rul_multipath.get("dominant_path", "Multipath"),
        "LastUpdate": pd.Timestamp.now(),
        "AR1_Weight": weights.get("AR1", 0.0),
        "Exp_Weight": weights.get("Exponential", 0.0),
        "Weibull_Weight": weights.get("Weibull", 0.0),
        "DataQuality": data_quality,
        # RUL-MULTIPATH: Match SQL column names from create_continuous_forecast_tables.sql
        "RUL_Trajectory_Hours": rul_multipath.get("rul_trajectory_hours"),
        "RUL_Hazard_Hours": rul_multipath.get("rul_hazard_hours"),
        "RUL_Energy_Hours": rul_multipath.get("rul_energy_hours"),
        "RUL_Final_Hours": rul_multipath["rul_final_hours"],  # Duplicate for explicit multipath tracking
        "DominantPath": rul_multipath.get("dominant_path", "Multipath"),  # Add dominant path indicator
    }
    
    df = pd.DataFrame([summary])
    return df


def build_sensor_attribution(
    sensor_hotspots_df: Optional[pd.DataFrame],
    rul_multipath: Dict[str, Any],
    current_time: pd.Timestamp,
    run_id: str,
    equip_id: int,
) -> pd.DataFrame:
    """
    Build ACM_RUL_Attribution compatible DataFrame from sensor hotspots.
    
    Args:
        sensor_hotspots_df: Sensor hotspots from SQL (or None)
        rul_multipath: Multipath RUL result
        current_time: Current timestamp
        run_id: Run ID
        equip_id: Equipment ID
    
    Returns:
        DataFrame with columns: RunID, EquipID, FailureTime, SensorName,
                                FailureContribution, ZScoreAtFailure, AlertCount
    """
    if sensor_hotspots_df is None or sensor_hotspots_df.empty:
        Console.warn("[RUL-Attribution] No sensor hotspots available")
        return pd.DataFrame(columns=[
            "RunID", "EquipID", "FailureTime", "SensorName",
            "FailureContribution", "ZScoreAtFailure", "AlertCount"
        ])
    
    # Compute projected failure time
    rul_hours = rul_multipath["rul_final_hours"]
    failure_time = current_time + pd.Timedelta(hours=rul_hours)
    
    # Build attribution table
    df = sensor_hotspots_df.copy()
    df["RunID"] = run_id
    df["EquipID"] = equip_id
    df["FailureTime"] = failure_time
    
    # Ensure required columns exist
    required_cols = ["SensorName", "FailureContribution", "ZScoreAtFailure", "AlertCount"]
    for col in required_cols:
        if col not in df.columns:
            Console.warn(f"[RUL-Attribution] Missing column: {col}, filling with defaults")
            df[col] = 0.0 if col != "SensorName" else "Unknown"
    
    # Reorder columns
    df = df[["RunID", "EquipID", "FailureTime", "SensorName",
             "FailureContribution", "ZScoreAtFailure", "AlertCount"]]
    
    return df


def build_maintenance_recommendation(
    rul_multipath: Dict[str, Any],
    data_quality: str,
    confidence: float,
    cfg: RULConfig,
    run_id: str,
    equip_id: int,
) -> pd.DataFrame:
    """
    Build ACM_MaintenanceRecommendation compatible DataFrame.
    
    Generates actionable recommendations based on RUL, confidence, and data quality.
    
    Args:
        rul_multipath: Multipath RUL result
        data_quality: Data quality flag
        confidence: Confidence score
        cfg: Configuration with maintenance bands
        run_id: Run ID
        equip_id: Equipment ID
    
    Returns:
        DataFrame with columns: RunID, EquipID, Action, Urgency, RUL_Hours, Confidence, DataQuality
    """
    rul_hours = rul_multipath["rul_final_hours"]
    
    # Determine base urgency from RUL bands
    if rul_hours <= cfg.band_urgent:
        action = "Immediate action required"
        base_urgency = "URGENT"
    elif rul_hours <= cfg.band_plan:
        action = "Schedule maintenance soon"
        base_urgency = "HIGH"
    elif rul_hours <= cfg.band_watch:
        action = "Increase monitoring frequency"
        base_urgency = "MEDIUM"
    elif rul_hours <= cfg.band_normal:
        action = "Continue monitoring"
        base_urgency = "LOW"
    else:
        action = "Normal operation"
        base_urgency = "NONE"
    
    # Adjust urgency based on confidence and data quality
    urgency = base_urgency
    
    if confidence < 0.5:
        # Low confidence - increase urgency
        if base_urgency == "LOW":
            urgency = "MEDIUM"
        elif base_urgency == "MEDIUM":
            urgency = "HIGH"
        action += " (low confidence - verify with additional diagnostics)"
    
    if data_quality in ["GAPPY", "SPARSE"]:
        # Poor data quality - increase urgency
        if base_urgency == "NONE":
            urgency = "LOW"
        elif base_urgency == "LOW":
            urgency = "MEDIUM"
        action += " (poor data quality - improve sensor coverage)"
    
    # Calculate maintenance windows based on RUL
    from datetime import datetime, timedelta
    now = datetime.now()
    earliest_maintenance = now + timedelta(hours=max(0, rul_hours - 48))  # 48h before expected failure
    preferred_window_start = now + timedelta(hours=max(0, rul_hours - 72))  # 72h before
    preferred_window_end = now + timedelta(hours=rul_hours)  # At expected failure time
    failure_prob_at_window_end = min(1.0, max(0.0, 1.0 - (rul_hours / max(rul_hours, 168))))  # Higher prob as RUL decreases
    
    recommendation = {
        "RunID": run_id,
        "EquipID": equip_id,
        "Action": action,
        "Urgency": urgency,
        "RUL_Hours": rul_hours,
        "Confidence": confidence,
        "DataQuality": data_quality,
        "EarliestMaintenance": earliest_maintenance,
        "PreferredWindowStart": preferred_window_start,
        "PreferredWindowEnd": preferred_window_end,
        "FailureProbAtWindowEnd": failure_prob_at_window_end,
        "Comment": None,  # Optional field
    }
    
    df = pd.DataFrame([recommendation])
    return df


def compute_confidence(
    rul_multipath: Dict[str, Any],
    model_diagnostics: Dict[str, Any],
    learning_state: LearningState,
    data_quality: str,
) -> float:
    """
    Compute confidence score for RUL estimate.
    
    Factors considered:
    - CI width (narrower → higher confidence)
    - Model agreement (all models close → higher confidence)
    - Calibration stability (stable factor → higher confidence)
    - Data quality (OK → high, GAPPY → lower)
    
    Args:
        rul_multipath: Multipath RUL result
        model_diagnostics: Model diagnostics
        learning_state: Learning state
        data_quality: Data quality flag
    
    Returns:
        Confidence score in [0.0, 1.0]
    """
    # Use principled weighted sum instead of multiplicative factors
    # Weights: 40% CI confidence, 30% model agreement, 20% calibration, 10% data quality
    
    # Factor 1: CI width (relative to RUL) - narrower is better
    rul = rul_multipath["rul_final_hours"]
    lower = rul_multipath["lower_bound_hours"]
    upper = rul_multipath["upper_bound_hours"]
    
    ci_confidence = 0.5  # Default
    if rul > 0:
        ci_width_fraction = (upper - lower) / rul
        ci_confidence = np.clip(1.0 - ci_width_fraction / 2.0, 0.0, 1.0)
    
    # Factor 2: Model agreement - similar weights indicate agreement
    agreement_score = 0.5  # Default
    weights = model_diagnostics.get("weights", {})
    if len(weights) > 1:
        weight_values = list(weights.values())
        weight_std = np.std(weight_values)
        weight_mean = np.mean(weight_values)
        if weight_mean > 0:
            agreement_score = 1.0 - min(weight_std / weight_mean, 1.0)
    
    # Factor 3: Calibration stability - well-calibrated models are reliable
    cal_factor = learning_state.calibration_factor
    if 0.8 <= cal_factor <= 1.2:
        calibration_score = 1.0  # Well-calibrated
    else:
        calibration_score = max(0.3, 1.0 - abs(cal_factor - 1.0))  # Degrade smoothly
    
    # Factor 4: Data quality
    quality_scores = {
        "OK": 1.0,
        "SPARSE": 0.8,
        "GAPPY": 0.6,
        "FLAT": 0.5,
        "MISSING": 0.3,
    }
    quality_score = quality_scores.get(data_quality, 0.7)
    
    # Weighted sum (normalized to [0, 1])
    confidence = (
        0.4 * ci_confidence +
        0.3 * agreement_score +
        0.2 * calibration_score +
        0.1 * quality_score
    )
    
    # Clamp to [0, 1]
    confidence = np.clip(confidence, 0.0, 1.0)
    
    return float(confidence)


# ============================================================================
# Public API (RUL-REF-25)
# ============================================================================


def run_rul(
    sql_client: Any,
    equip_id: int,
    run_id: str,
    output_manager: Optional[Any] = None,
    config_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Unified RUL estimation entry point.
    
    This is the single public API for RUL estimation, replacing both
    estimate_rul_and_failure() from rul_estimator.py and
    enhanced_rul_estimator.py.
    
    Pipeline:
    1. Normalize inputs and build config
    2. Cleanup old forecast runs
    3. Load health timeline (cache > SQL)
    4. Load learning state from SQL
    5. Compute RUL via ensemble models and multipath
    6. Load sensor hotspots
    7. Build all output DataFrames
    8. Write outputs to SQL (if output_manager provided)
    9. Save updated learning state (future: after learning update)
    10. Return all tables
    
    Args:
        sql_client: SQL client for data access
        equip_id: Equipment ID (will be normalized to int)
        run_id: Run ID (will be normalized to str)
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
        RuntimeError: SQL connection issues or critical errors
    """
    Console.info("=" * 80)
    Console.info("[RUL] Starting unified RUL estimation")
    Console.info("=" * 80)
    
    # Step 1: Normalize inputs
    equip_id = ensure_equipid_int(equip_id)
    run_id = ensure_runid_str(run_id)
    
    Console.info(f"[RUL] EquipID={equip_id}, RunID={run_id}")
    
    # Step 2: Build configuration
    if config_row is not None:
        # Extract RUL-specific config from config_row
        forecasting_cfg = config_row.get("forecasting", {})
        cfg = RULConfig(
            health_threshold=float(forecasting_cfg.get("failure_threshold", 70.0)),
            min_points=int(forecasting_cfg.get("min_points", 20)),
            max_forecast_hours=float(forecasting_cfg.get("max_forecast_hours", 168.0)),
            learning_rate=float(forecasting_cfg.get("learning_rate", 0.1)),
            min_model_weight=float(forecasting_cfg.get("min_model_weight", 0.1)),
            enable_online_learning=bool(forecasting_cfg.get("enable_online_learning", False)),
            calibration_window=int(forecasting_cfg.get("calibration_window", 100)),
        )
    else:
        cfg = RULConfig()
    
    Console.info(f"[RUL] Config: threshold={cfg.health_threshold}, max_forecast={cfg.max_forecast_hours}h")
    
    # Step 3: Cleanup old forecasts
    try:
        keep_runs = int(os.getenv("ACM_FORECAST_RUNS_RETAIN", "2"))
        cleanup_old_forecasts(sql_client, equip_id, keep_runs)
    except Exception as e:
        Console.warn(f"[RUL] Forecast cleanup failed: {e}")
    
    # Step 4: Load health timeline
    health_df, data_quality = load_health_timeline(
        sql_client=sql_client,
        equip_id=equip_id,
        run_id=run_id,
        output_manager=output_manager,
        cfg=cfg,
    )
    
    if health_df is None or health_df.empty:
        Console.error("[RUL] Cannot proceed without health timeline")
        raise RuntimeError("Health timeline unavailable")
    
    Console.info(f"[RUL] Loaded health timeline: {len(health_df)} points, quality={data_quality}")
    
    # Step 5: Load learning state
    learning_state = load_learning_state(sql_client, equip_id)
    Console.info(f"[RUL] Loaded learning state: cal_factor={learning_state.calibration_factor:.2f}")
    
    # Step 6: Compute RUL
    rul_result = compute_rul(
        health_df=health_df,
        cfg=cfg,
        learning_state=learning_state,
        data_quality_flag=data_quality,
    )
    
    # Step 7: Compute confidence
    confidence = compute_confidence(
        rul_multipath=rul_result["rul_multipath"],
        model_diagnostics=rul_result["model_diagnostics"],
        learning_state=learning_state,
        data_quality=data_quality,
    )
    
    Console.info(f"[RUL] Confidence score: {confidence:.2f}")
    
    # Step 8: Load sensor hotspots
    sensor_hotspots_df = load_sensor_hotspots(sql_client, equip_id, run_id)
    
    # Step 9: Build all output DataFrames
    Console.info("[RUL] Building output DataFrames...")
    
    tables = {}
    
    tables["ACM_HealthForecast_TS"] = make_health_forecast_ts(
        health_forecast=rul_result["health_forecast"],
        run_id=run_id,
        equip_id=equip_id,
    )
    
    tables["ACM_FailureForecast_TS"] = make_failure_forecast_ts(
        failure_curve=rul_result["failure_curve"],
        run_id=run_id,
        equip_id=equip_id,
    )
    
    tables["ACM_RUL_TS"] = make_rul_ts(
        health_forecast=rul_result["health_forecast"],
        rul_multipath=rul_result["rul_multipath"],
        current_time=rul_result["current_time"],
        run_id=run_id,
        equip_id=equip_id,
        confidence=confidence,
    )
    
    tables["ACM_RUL_Summary"] = make_rul_summary(
        rul_multipath=rul_result["rul_multipath"],
        model_diagnostics=rul_result["model_diagnostics"],
        data_quality=data_quality,
        run_id=run_id,
        equip_id=equip_id,
        confidence=confidence,
    )
    
    tables["ACM_RUL_Attribution"] = build_sensor_attribution(
        sensor_hotspots_df=sensor_hotspots_df,
        rul_multipath=rul_result["rul_multipath"],
        current_time=rul_result["current_time"],
        run_id=run_id,
        equip_id=equip_id,
    )
    
    tables["ACM_MaintenanceRecommendation"] = build_maintenance_recommendation(
        rul_multipath=rul_result["rul_multipath"],
        data_quality=data_quality,
        confidence=confidence,
        cfg=cfg,
        run_id=run_id,
        equip_id=equip_id,
    )
    
    # Step 10: Write to SQL (if output_manager provided)
    if output_manager is not None:
        Console.info("[RUL] Writing outputs to SQL via OutputManager...")
        for table_name, df in tables.items():
            try:
                # Prefer OutputManager.write_table if available
                if hasattr(output_manager, 'write_table'):
                    output_manager.write_table(table_name, df)
                else:
                    # Fallback: inject metadata and required fields then bulk insert
                    sql_df = df.copy()
                    if 'RunID' not in sql_df.columns:
                        sql_df['RunID'] = run_id
                    if 'EquipID' not in sql_df.columns:
                        sql_df['EquipID'] = equip_id
                    now = pd.Timestamp.now()
                    if 'Method' in sql_df.columns:
                        sql_df['Method'] = sql_df['Method'].fillna('default')
                    if 'LastUpdate' in sql_df.columns:
                        sql_df['LastUpdate'] = sql_df['LastUpdate'].fillna(now)
                    if 'EarliestMaintenance' in sql_df.columns:
                        sql_df['EarliestMaintenance'] = sql_df['EarliestMaintenance'].fillna(now)
                    output_manager._bulk_insert_sql(table_name, sql_df)
                Console.info(f"[RUL] ✓ Wrote {len(df)} rows to {table_name}")
            except Exception as e:
                Console.warn(f"[RUL] ✗ Failed to write {table_name}: {e}")
    
    # Step 11: Save learning state (future: after learning update)
    # For now, save unchanged state to ensure table exists
    try:
        # save_learning_state expects (sql_client, state)
        save_learning_state(sql_client, learning_state)
    except Exception as e:
        Console.warn(f"[RUL] Failed to save learning state: {e}")
    
    # Step 12: Log summary
    rul_hours = rul_result["rul_multipath"]["rul_final_hours"]
    method = rul_result["rul_multipath"]["dominant_path"]
    
    Console.info("=" * 80)
    Console.info(f"[RUL] ✓ Estimation complete")
    Console.info(f"[RUL] RUL: {rul_hours:.1f} hours ({rul_hours/24:.1f} days)")
    Console.info(f"[RUL] Method: {method}")
    Console.info(f"[RUL] Confidence: {confidence:.2f}")
    Console.info(f"[RUL] Data Quality: {data_quality}")
    Console.info("=" * 80)
    
    return tables

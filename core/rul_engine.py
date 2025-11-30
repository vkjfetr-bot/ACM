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

import numpy as np
import pandas as pd

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
    """Standard normal CDF approximation"""
    return 0.5 * (1.0 + np.tanh(z * np.sqrt(2.0 / np.pi)))


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
    
    try:
        cur = sql_client.cursor()
        cur.execute(
            """
            SELECT SensorName, FailureContribution, 
                   COALESCE(Z, ZScoreAtFailure) as ZScoreAtFailure,
                   COALESCE(AlertCount, AboveAlertCount) as AlertCount
            FROM dbo.ACM_SensorHotspots
            WHERE EquipID = ? AND RunID = ?
            ORDER BY FailureContribution DESC
            """,
            (equip_id, run_id),
        )
        rows = cur.fetchall()
        cur.close()
        
        if rows:
            df = pd.DataFrame.from_records(
                rows,
                columns=["SensorName", "FailureContribution", "ZScoreAtFailure", "AlertCount"]
            )
            Console.info(f"[RUL] Loaded {len(df)} sensor hotspots from SQL")
            return df
        else:
            Console.warn(f"[RUL] No sensor hotspots found for EquipID={equip_id}, RunID={run_id}")
            return pd.DataFrame()
            
    except Exception as e:
        Console.warn(f"[RUL] Failed to load sensor hotspots: {e}")
        return pd.DataFrame()


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
        cur.close()
        
        if row:
            col_names = [desc[0] for desc in cur.description]
            row_dict = dict(zip(col_names, row))
            state = LearningState.from_sql_dict(equip_id, row_dict)
            Console.info(f"[RUL-Learn] Loaded learning state for EquipID={equip_id}")
            return state
        else:
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

        # Estimate AR(1) coefficient
        cov = float(np.dot(yc[1:], yc[:-1]))
        var = float(np.dot(yc[:-1], yc[:-1]))
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

        # Growing uncertainty over time
        if abs(self.phi) < 0.999:
            var_mult = (1 - self.phi ** (2 * h)) / (1 - self.phi**2 + 1e-9)
        else:
            var_mult = h

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
        offset = float(np.min(y)) - 1.0  # Stabilize for log
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

        forecast = self.h0 - self.k * (t**self.beta)

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
        freq=f"{sampling_interval_hours}H"
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
    
    Path 1 (Trajectory): Mean health forecast crosses failure threshold
    Path 2 (Hazard): Failure probability exceeds threshold (e.g., 50%)
    Path 3 (Energy): Reserved for future anomaly energy integration
    
    Args:
        health_forecast: DataFrame with [Timestamp, ForecastHealth, CI_Lower, CI_Upper]
        failure_curve: DataFrame with [Timestamp, FailureProb, ThresholdUsed]
        current_time: Current timestamp
        cfg: Configuration
    
    Returns:
        Dict with:
        - rul_trajectory_hours: RUL from trajectory path (or None)
        - rul_hazard_hours: RUL from hazard path (or None)
        - rul_energy_hours: RUL from energy path (or None)
        - rul_final_hours: Selected RUL
        - lower_bound_hours: Lower confidence bound
        - upper_bound_hours: Upper confidence bound
        - dominant_path: Which path was used ("trajectory", "hazard", or "energy")
    """
    Console.info("[RUL-Multipath] Computing RUL via multiple paths...")
    
    # Path 1: Trajectory crossing (mean forecast < threshold)
    rul_trajectory = None
    if health_forecast is not None and not health_forecast.empty:
        trajectory_crossing = health_forecast[
            health_forecast["ForecastHealth"] <= cfg.health_threshold
        ]
        if not trajectory_crossing.empty:
            t1 = trajectory_crossing.iloc[0]["Timestamp"]
            rul_trajectory = (t1 - current_time).total_seconds() / 3600
            Console.info(f"[RUL-Multipath] Trajectory crossing at {t1}, RUL={rul_trajectory:.1f}h")
        else:
            Console.info("[RUL-Multipath] No trajectory crossing within forecast horizon")
    
    # Path 2: Hazard accumulation (failure probability >= 50%)
    rul_hazard = None
    hazard_prob_threshold = 0.5  # 50% probability
    if failure_curve is not None and not failure_curve.empty:
        hazard_crossing = failure_curve[failure_curve["FailureProb"] >= hazard_prob_threshold]
        if not hazard_crossing.empty:
            t2 = hazard_crossing.iloc[0]["Timestamp"]
            rul_hazard = (t2 - current_time).total_seconds() / 3600
            Console.info(f"[RUL-Multipath] Hazard crossing at {t2}, RUL={rul_hazard:.1f}h")
        else:
            Console.info("[RUL-Multipath] No hazard crossing within forecast horizon")
    
    # Path 3: Energy (reserved for future integration)
    rul_energy = None
    # TODO: Integrate anomaly energy if available
    
    # Select dominant RUL (minimum of available paths)
    available_ruls = [r for r in [rul_trajectory, rul_hazard, rul_energy] if r is not None]
    
    if available_ruls:
        rul_final = min(available_ruls)
        
        # Determine dominant path
        if rul_final == rul_trajectory:
            dominant_path = "trajectory"
        elif rul_final == rul_hazard:
            dominant_path = "hazard"
        else:
            dominant_path = "energy"
    else:
        # No crossing detected, use max forecast horizon
        rul_final = cfg.max_forecast_hours
        dominant_path = "none"
        Console.info(f"[RUL-Multipath] No crossing detected, using max horizon: {rul_final:.1f}h")
    
    # Compute confidence bounds from CI crossings
    lower_bound = None
    upper_bound = None
    
    if health_forecast is not None and not health_forecast.empty:
        # Lower bound: CI_Lower crosses threshold
        ci_lower_crossing = health_forecast[
            health_forecast["CI_Lower"] <= cfg.health_threshold
        ]
        if not ci_lower_crossing.empty:
            t_lower = ci_lower_crossing.iloc[0]["Timestamp"]
            lower_bound = (t_lower - current_time).total_seconds() / 3600
        
        # Upper bound: CI_Upper crosses threshold
        ci_upper_crossing = health_forecast[
            health_forecast["CI_Upper"] <= cfg.health_threshold
        ]
        if not ci_upper_crossing.empty:
            t_upper = ci_upper_crossing.iloc[0]["Timestamp"]
            upper_bound = (t_upper - current_time).total_seconds() / 3600
    
    # If bounds not available, use ±30% of RUL
    if lower_bound is None:
        lower_bound = max(0.0, rul_final * 0.7)
    if upper_bound is None:
        upper_bound = rul_final * 1.3
    
    result = {
        "rul_trajectory_hours": rul_trajectory,
        "rul_hazard_hours": rul_hazard,
        "rul_energy_hours": rul_energy,
        "rul_final_hours": float(rul_final),
        "lower_bound_hours": float(lower_bound),
        "upper_bound_hours": float(upper_bound),
        "dominant_path": dominant_path,
    }
    
    Console.info(
        f"[RUL-Multipath] Final RUL={rul_final:.1f}h "
        f"(trajectory={rul_trajectory}, hazard={rul_hazard}, dominant={dominant_path})"
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


# ============================================================================
# TODO: Output Builders (RUL-REF-22, RUL-REF-23, RUL-REF-24)
# ============================================================================
# - make_health_forecast_df
# - make_failure_curve_df
# - make_rul_ts_df
# - make_rul_summary_df
# - build_sensor_attribution
# - build_maintenance_recommendation


# ============================================================================
# TODO: Public API (RUL-REF-25)
# ============================================================================
# - run_rul function (single entry point)

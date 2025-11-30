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
# Placeholder for Model Layer (to be continued in next part)
# ============================================================================

# TODO: Implement ensemble models (AR1, Exponential, Weibull)
# TODO: Implement RULModel wrapper
# TODO: Implement compute_rul and multipath logic
# TODO: Implement output builders
# TODO: Implement run_rul public API

"""
Common helpers for RUL estimation modules.

RUL-COR-01: Consolidated shared functions from rul_estimator.py and enhanced_rul_estimator.py
to eliminate code duplication and ensure consistent behavior across both modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np
from math import erf, sqrt
import pandas as pd

# Import Console for logging
try:
    from utils.logger import Console
except ImportError:
    class Console:
        @staticmethod
        def info(msg): print(f"[INFO] {msg}")
        @staticmethod
        def warn(msg): print(f"[WARN] {msg}")

def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF that accepts scalars or numpy arrays."""
    arr = np.asarray(x, dtype=float)
    scaled = arr / sqrt(2.0)
    # math.erf only supports scalars; vectorize to handle numpy arrays.
    erf_vec = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(scaled))

@dataclass
class RULConfig:
    health_threshold: float = 70.0
    min_points: int = 20
    max_forecast_hours: float = 24.0
    maintenance_risk_low: float = 0.2
    maintenance_risk_high: float = 0.5
    
    # Enhanced specific (optional here, but good to have base)
    learning_rate: float = 0.1
    min_model_weight: float = 0.05
    enable_online_learning: bool = True
    calibration_window: int = 50


# ============================================================================
# RUL-COR-01: Shared Data Loading Functions
# ============================================================================

def apply_health_timeline_row_limit(df: pd.DataFrame, config: Optional[Any] = None) -> pd.DataFrame:
    """
    Apply row limit guard to health timeline data.
    
    RUL-COR-01: Consolidated from _apply_health_timeline_row_limit in both RUL modules.
    FOR-PERF-04: Prevents unbounded memory usage from large health timelines.
    
    If dataframe exceeds max_health_timeline_rows config, downsamples to health_downsample_freq.
    
    Args:
        df: DataFrame with Timestamp column and HealthIndex data
        config: Configuration dictionary with forecasting.max_health_timeline_rows and 
                forecasting.health_downsample_freq settings
    
    Returns:
        Original or downsampled DataFrame
    """
    if df is None or df.empty:
        return df
    
    # Default conservative limits
    max_rows = 100000
    downsample_freq = "1min"
    
    if config is not None:
        max_rows = config.get("forecasting", {}).get("max_health_timeline_rows", max_rows)
        downsample_freq = config.get("forecasting", {}).get("health_downsample_freq", downsample_freq)
    
    if len(df) <= max_rows:
        return df
    
    # Downsample: resample to frequency with mean aggregation
    Console.warn(
        f"[RUL] Health timeline has {len(df)} rows (max={max_rows}). "
        f"Downsampling to {downsample_freq} frequency."
    )
    
    # Ensure Timestamp is index for resample
    df_resampled = df.set_index("Timestamp").resample(downsample_freq).mean()
    df_resampled = df_resampled.dropna().reset_index()
    
    Console.info(f"[RUL] Downsampled to {len(df_resampled)} rows")
    return df_resampled


def load_health_timeline(
    tables_dir: Path,
    sql_client: Optional[Any] = None,
    equip_id: Optional[int] = None,
    run_id: Optional[str] = None,
    output_manager: Optional[Any] = None,
    config: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """
    Load health timeline for RUL estimation with SQL-first priority.
    
    RUL-COR-01: Consolidated from _load_health_timeline in both RUL modules.
    RUL-CSV-01, RUL-CSV-02: SQL-first loading strategy.
    FOR-PERF-04: Applies row limit guard to prevent unbounded memory usage.
    
    Priority order:
    1. Artifact cache (output_manager.get_cached_table) - SQL-only mode
    2. SQL query (ACM_HealthTimeline) - when sql_client, equip_id, run_id available
    3. CSV file fallback (health_timeline.csv) - legacy file mode for dev/testing
    
    Args:
        tables_dir: Directory containing CSV files (fallback path)
        sql_client: SQL client for querying ACM_HealthTimeline
        equip_id: Equipment ID filter for SQL query
        run_id: Run ID filter for SQL query
        output_manager: OutputManager with cached table access
        config: Configuration for row limits and downsampling
    
    Returns:
        DataFrame with Timestamp and HealthIndex columns, or None if no data found
    """
    # Priority 1: Try artifact cache first (SQL-only mode support)
    if output_manager is not None:
        df = output_manager.get_cached_table("health_timeline.csv")
        if df is not None:
            Console.info(f"[RUL] Using cached health_timeline.csv from OutputManager ({len(df)} rows)")
            # Normalize column names
            if "Timestamp" in df.columns:
                ts_col = "Timestamp"
            elif "timestamp" in df.columns:
                ts_col = "timestamp"
            else:
                ts_col = df.columns[0]
            # TIME-01: Ensure naive timestamps
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            if df[ts_col].dt.tz is not None:
                df[ts_col] = df[ts_col].dt.tz_localize(None)
            
            df = df.dropna(subset=[ts_col]).sort_values(ts_col)
            df = df.rename(columns={ts_col: "Timestamp"})
            # FOR-PERF-04: Apply max row guard
            df = apply_health_timeline_row_limit(df, config)
            return df
    
    # Priority 2: Try SQL when possible (SQL-only mode without cache)
    if sql_client is not None and equip_id is not None and run_id:
        try:
            cur = sql_client.cursor()
            cur.execute(
                """
                SELECT Timestamp, HealthIndex
                FROM dbo.ACM_HealthTimeline
                WHERE EquipID = ? AND RunID = ?
                ORDER BY Timestamp
                """,
                (equip_id, run_id),
            )
            rows = cur.fetchall()
            cur.close()
            if rows:
                df = pd.DataFrame.from_records(rows, columns=["Timestamp", "HealthIndex"])
                # TIME-01: Ensure naive timestamps
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                if df["Timestamp"].dt.tz is not None:
                    df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)
                
                df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
                Console.info(
                    f"[RUL] Loaded {len(df)} health points from SQL for EquipID={equip_id}, RunID={run_id}"
                )
                # FOR-PERF-04: Apply max row guard
                df = apply_health_timeline_row_limit(df, config)
                return df
            else:
                Console.warn(
                    f"[RUL] No rows in ACM_HealthTimeline for EquipID={equip_id}, RunID={run_id}"
                )
        except Exception as e:
            Console.warn(f"[RUL] Failed to load health timeline from SQL: {e}")

    # Priority 3: Fallback to legacy CSV path (file mode for dev/testing)
    p = tables_dir / "health_timeline.csv"
    if not p.exists():
        Console.warn(f"[RUL] health_timeline.csv not found in {tables_dir} (no cache, no SQL, no file)")
        return None
    df = pd.read_csv(p)
    Console.info(f"[RUL] Loaded health_timeline.csv from file ({len(df)} rows)")
    if "Timestamp" in df.columns:
        ts_col = "Timestamp"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        ts_col = df.columns[0]
    # TIME-01: Ensure naive timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].dt.tz is not None:
        df[ts_col] = df[ts_col].dt.tz_localize(None)
        
    df = df.sort_values(ts_col).dropna(subset=[ts_col])
    df = df.rename(columns={ts_col: "Timestamp"})
    # FOR-PERF-04: Apply max row guard
    df = apply_health_timeline_row_limit(df, config)
    return df

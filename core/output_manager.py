"""
Unified Output Manager for ACM
==============================

Consolidates all scattered output generation into a single, efficient system:
- Batched file writes with intelligent buffering
- Smart SQL/file dual-write coordination with caching
- Single point of control for all CSV, JSON, and model outputs
- Performance optimizations: vectorized operations, reduced I/O
- Unified error handling and logging

This replaces scattered to_csv() calls throughout the codebase and provides
consistent behavior for all output operations.
"""

from __future__ import annotations
# pyright: reportGeneralTypeIssues=false

import json
import time
import math
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Union, Tuple, TypeVar, Callable, cast
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import warnings 
from datetime import datetime, timezone

# FOR-DQ-02: Use centralized timestamp normalization
from utils.timestamp_utils import (
    normalize_timestamp_scalar,
    normalize_timestamp_series,
    normalize_timestamps,
    normalize_index
)

from utils.detector_labels import get_detector_label, format_culprit_label

from core.observability import Console

# V11: Confidence model for health and episode confidence
try:
    from core.confidence import compute_health_confidence, compute_episode_confidence
    _CONFIDENCE_AVAILABLE = True
except ImportError:
    _CONFIDENCE_AVAILABLE = False
    compute_health_confidence = None
    compute_episode_confidence = None

# V11: Model lifecycle for maturity state
try:
    from core.model_lifecycle import ModelLifecycle
    _LIFECYCLE_AVAILABLE = True
except ImportError:
    _LIFECYCLE_AVAILABLE = False
    ModelLifecycle = None

# Optional observability integration (P0 SQL ops tracking)
try:
    from core.observability import record_sql_op, Span
    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    _OBSERVABILITY_AVAILABLE = False
    record_sql_op = None
    Span = None

# =============================================================================
# ALLOWED_TABLES - v11.0.0 Functionality-Based Table Set
# =============================================================================
# DESIGN PRINCIPLE: Tables chosen based on ACM's core mission:
#   1. What is current health? (HealthTimeline, Scores, Episodes)
#   2. If not healthy, what's the reason? (SensorDefects, Hotspots, Culprits)
#   3. What will future health look like? (RUL, Forecasts)
#   4. What will cause future degradation? (SensorForecast, Drift, Contributions)
#
# Additional tables support: data persistence, model evolution, diagnostics
#
# See docs/ACM_OUTPUT_TABLES_REFINED.md for complete rationale.
#
# TIER 1 - CURRENT STATE (What's happening NOW?)
# TIER 2 - FUTURE STATE (What will happen?)
# TIER 3 - ROOT CAUSE (WHY is this happening/will happen?)
# TIER 4 - DATA & MODEL MANAGEMENT (Long-term storage, model building)
# TIER 5 - OPERATIONS & AUDIT (Is ACM working? What changed?)
# TIER 6 - ADVANCED ANALYTICS (Deep patterns and trends)
# TIER 7 - V11 NEW FEATURES (Typed contracts, maturity lifecycle, seasonality)
# =============================================================================

ALLOWED_TABLES = {
    # TIER 1: CURRENT STATE (6 tables) - Answers "What is current health?"
    'ACM_HealthTimeline',        # Health history + current state
    'ACM_Scores_Wide',           # All 6 detector scores per timestamp
    'ACM_Episodes',              # Active and historical anomaly events
    'ACM_RegimeTimeline',        # Operating mode context
    'ACM_SensorDefects',         # Which sensors are problematic NOW
    'ACM_SensorHotspots',        # Top culprit sensors ranked
    
    # TIER 2: FUTURE STATE (4 tables) - Answers "What will future health look like?"
    'ACM_RUL',                   # When will it fail? (Remaining Useful Life)
    'ACM_HealthForecast',        # Projected health trajectory with confidence bounds
    'ACM_FailureForecast',       # Failure probability over time
    'ACM_SensorForecast',        # Physical sensor value predictions
    
    # TIER 3: ROOT CAUSE (6 tables) - Answers "Why?" (current + future)
    'ACM_EpisodeCulprits',       # What caused each episode
    'ACM_EpisodeDiagnostics',    # Episode details and severity
    'ACM_DetectorCorrelation',   # How detectors relate (model quality)
    'ACM_DriftSeries',           # Behavior changes that lead to degradation
    'ACM_SensorCorrelations',    # Multivariate sensor relationships (correlation matrix)
    'ACM_FeatureDropLog',        # Why features were dropped (quality issues)
    'ACM_OMR_Diagnostics',       # OMR detector diagnostics
    
    # TIER 4: DATA & MODEL MANAGEMENT (10 tables) - Long-term storage, enables progressive learning
    'ACM_BaselineBuffer',        # Raw sensor data accumulation for training
    'ACM_HistorianData',         # Cached historian data for efficiency
    'ACM_SensorNormalized_TS',   # Normalized sensor time series
    'ACM_DataQuality',           # Input data health metrics
    'ACM_ForecastingState',      # Forecast model state persistence
    'ACM_CalibrationSummary',    # Model quality tracking over time
    'ACM_AdaptiveConfig',        # Auto-tuned configuration
    'ACM_RefitRequests',         # Model retraining requests and acknowledgements
    'ACM_PCA_Metrics',           # PCA component metrics and explained variance
    'ACM_RunMetadata',           # Detailed run context (batch info, data ranges)
    
    # TIER 5: OPERATIONS & AUDIT (6 tables) - Is ACM working? What changed?
    'ACM_Runs',                  # Execution tracking and status
    'ACM_RunLogs',               # Detailed logs for troubleshooting
    # 'ACM_RunTimers' DEPRECATED - observability stack (Tempo/Prometheus/Loki) handles timing
    'ACM_RunMetrics',            # Fusion quality metrics (EAV format)
    'ACM_Run_Stats',             # Run-level statistics
    'ACM_Config',                # Current configuration
    'ACM_ConfigHistory',         # Configuration change audit trail
    
    # TIER 6: ADVANCED ANALYTICS (5 tables) - Deep insights and patterns
    'ACM_RegimeOccupancy',       # Operating mode utilization
    'ACM_RegimeTransitions',     # Mode switching patterns
    'ACM_Regime_Episodes',       # Regime episode tracking
    'ACM_RegimePromotionLog',    # Regime maturity evolution tracking
    'ACM_ContributionTimeline',  # Historical sensor attribution for pattern analysis
    'ACM_DriftController',       # Drift detection control and thresholds
    'ACM_PCA_Models',            # PCA model metadata
    'ACM_PCA_Loadings',          # PCA component loadings per sensor
    'ACM_Anomaly_Events',        # Anomaly event records
    
    # TIER 7: V11 NEW FEATURES (5 tables) - Advanced capabilities from v11.0.0
    'ACM_RegimeDefinitions',     # Regime centroids and metadata (MaturityState lifecycle)
    'ACM_ActiveModels',          # Active model versions per equipment
    'ACM_DataContractValidation',# Data quality validation at pipeline entry
    'ACM_SeasonalPatterns',      # Detected seasonal patterns (diurnal, weekly)
    'ACM_AssetProfiles',         # Asset similarity for cold-start transfer learning
}

def _table_exists(cursor_factory: Callable[[], Any], name: str) -> bool:
    cur = None
    try:
        cur = cursor_factory()
        cur.execute(f"SELECT TOP 0 * FROM dbo.[{name}]")
        return True
    except Exception:
        return False
    finally:
        try:
            if cur is not None:
                cur.close()
        except Exception:
            pass

def _get_table_columns(cursor_factory: Callable[[], Any], name: str) -> List[str]:
    """Return the list of column names for a table by probing TOP 0."""
    cur = cursor_factory()
    try:
        cur.execute(f"SELECT TOP 0 * FROM dbo.[{name}]")
        return [d[0] for d in (cur.description or [])]
    finally:
        try:
            cur.close()
        except Exception:
            pass

def _get_insertable_columns(cursor_factory: Callable[[], Any], name: str) -> List[str]:
    """Return columns excluding identity columns for safe INSERT.
    Uses sys.columns joined via OBJECT_ID('schema.table') without square brackets.
    """
    cur = cursor_factory()
    try:
        # OBJECT_ID expects a schema-qualified name without brackets
        cur.execute(
            "SELECT c.name, c.is_identity FROM sys.columns c WHERE c.object_id = OBJECT_ID(?)",
            (f"dbo.{name}",)
        )
        rows = cur.fetchall() or []
        return [r[0] for r in rows if not getattr(r, 'is_identity', r[1])]
    finally:
        try:
            cur.close()
        except Exception:
            pass

# constants for analytics & guardrails
class AnalyticsConstants:
    DRIFT_EVENT_THRESHOLD = 3.0
    DEFAULT_CLIP_Z = 30.0
    TARGET_SAMPLING_POINTS = 500
    HEALTH_ALERT_THRESHOLD = 70.0
    HEALTH_CAUTION_THRESHOLD = 85.0  # Previously WATCH
    HEALTH_WATCH_THRESHOLD = HEALTH_CAUTION_THRESHOLD  # kept for backward compatibility

    @staticmethod
    def anomaly_level(abs_z: float, warn: float, alert: float) -> str:
        try:
            if abs_z >= float(alert):
                return "ALERT"
            if abs_z >= float(warn):
                return "CAUTION"
        except Exception:
            pass
        return "GOOD"

# CHART-12: Centralized severity color palette
SEVERITY_COLORS = {
    "CRITICAL": "#dc2626",  # Red - immediate action required
    "HIGH": "#f97316",      # Orange - urgent attention
    "MEDIUM": "#f59e0b",    # Amber - monitor closely
    "LOW": "#10b981",       # Green - informational
    "INFO": "#10b981",      # Alias for LOW
    "WARNING": "#f59e0b",   # Alias for MEDIUM
}

# Enhanced config getter with typing
T = TypeVar("T")

def _cfg_get(cfg: Dict[str, Any], path: str, default: T) -> T:
    """Get config value by dot path with type preservation."""
    keys = path.split('.')
    current: Any = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current  # type: ignore[return-value]


def _future_cutoff_ts(cfg: Dict[str, Any]) -> pd.Timestamp:
    """Return timestamp cutoff that optionally allows future data via config."""
    raw_value = _cfg_get(cfg, "runtime.future_grace_minutes", 0) or 0
    try:
        minutes = int(raw_value)
    except (TypeError, ValueError):
        minutes = 0
    minutes = max(0, minutes)
    return pd.Timestamp.now() + pd.Timedelta(minutes=minutes)

# ==================== DATA LOADING SUPPORT ====================
@dataclass
class DataMeta:
    """Metadata about loaded dataset."""
    timestamp_col: str
    cadence_ok: bool
    kept_cols: List[str]
    dropped_cols: List[str]
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    n_rows: int
    sampling_seconds: float
    tz_stripped: int = 0
    future_rows_dropped: int = 0
    dup_timestamps_removed: int = 0

# (duplicate removed; use the typed _cfg_get defined above)

def _parse_ts_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Parse timestamp column and set as index."""
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found")
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.set_index(ts_col).sort_index()
    return df


def _coerce_local_and_filter_future(df: pd.DataFrame, label: str, now_cutoff: pd.Timestamp) -> Tuple[pd.DataFrame, int, int]:
    """Convert timestamp index to naive local time and drop future rows.

    Returns the sanitized DataFrame along with counts for timezone stripping and
    future-dated rows that were removed.
    """
    tz_stripped = 0
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    else:
        try:
            if df.index.tz is not None:
                tz_stripped = len(df)
                df.index = df.index.tz_localize(None)
        except Exception:
            df.index = pd.to_datetime(df.index, errors="coerce")

    # Drop NaT entries created during coercion
    before_drop = len(df)
    df = df[~df.index.isna()]
    if before_drop and len(df) != before_drop:
        Console.warn(f"Dropped {before_drop - len(df)} rows with invalid timestamps from {label}", component="DATA", label=label, rows_dropped=before_drop - len(df), rows_remaining=len(df))

    future_mask = df.index > now_cutoff
    future_rows = int(future_mask.sum())
    if future_rows:
        Console.warn(f"Dropping {future_rows} future timestamp row(s) from {label} (cutoff={now_cutoff:%Y-%m-%d %H:%M:%S})", component="DATA", label=label, future_rows=future_rows, cutoff=str(now_cutoff))
        df = df[~future_mask]

    return df, tz_stripped, future_rows

def _infer_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def _native_cadence_secs(idx: pd.DatetimeIndex) -> float:
    """Estimate native cadence in seconds."""
    if len(idx) < 2:
        return float('inf')
    diffs = idx.to_series().diff().dropna()
    # Handle pandas Timedelta median vs numeric
    med = diffs.median()
    try:
        # Timedelta has total_seconds()
        return float(getattr(med, "total_seconds", lambda: float(med))())
    except Exception:
        try:
            import numpy as np
            return float(np.median(diffs))
        except Exception:
            return float('inf')

def _check_cadence(idx: pd.DatetimeIndex, sampling_secs: Optional[int], jitter_ratio: float = 0.05) -> bool:
    """Check if timestamps have regular cadence."""
    if sampling_secs is None or len(idx) < 2:
        return True
    diffs = idx.to_series().diff().dropna()
    expected = pd.Timedelta(seconds=sampling_secs)
    tolerance = expected * jitter_ratio
    return ((diffs - expected).abs() <= tolerance).mean() >= 0.9

def _resample(df: pd.DataFrame, sampling_secs: int, interp_method: str = "linear", strict: bool = False, max_gap_secs: int = 300, max_fill_ratio: float = 0.2) -> pd.DataFrame:
    """Resample DataFrame to regular intervals."""
    if df.empty:
        return df
    if df.index.min() == df.index.max():
        return df  # single-point; nothing to resample
    freq = f"{sampling_secs}s"
    start = df.index.min()
    end = df.index.max()
    regular_idx = pd.date_range(start=start, end=end, freq=freq)
    df_resampled = df.reindex(regular_idx)
    if interp_method != "none":
        max_gap_periods = max_gap_secs // sampling_secs
        # Cast method to Any to satisfy type-checkers across pandas versions
        df_resampled = df_resampled.interpolate(method=cast(Any, interp_method), limit=max_gap_periods, limit_direction='both')
    if strict:
        fill_ratio = df_resampled.isnull().sum().sum() / (len(df_resampled) * len(df_resampled.columns))
        if fill_ratio > max_fill_ratio:
            raise ValueError(f"Too much missing data after resample: {fill_ratio:.1%} > {max_fill_ratio:.1%}")
    return df_resampled


def _health_index(fused_z, z_threshold: float = 5.0, steepness: float = 1.5):
    """
    Calculate health index from fused z-score using a softer sigmoid mapping.
    
    v10.1.0: Replaced overly aggressive 100/(1+Z^2) formula.
    
    OLD formula issues:
    - Z=2.5 gave Health=14% (too harsh for moderate anomaly)
    - Z=3.0 gave Health=10% (equipment in crisis for minor deviation)
    
    NEW sigmoid formula:
    - Z=0: Health = 100% (perfectly normal)
    - Z=z_threshold/2 (2.5): Health = 50% (moderate concern)
    - Z=z_threshold (5.0): Health   15% (serious anomaly)
    - Z>z_threshold: Health approaches 0% asymptotically
    
    Args:
        fused_z: Fused z-score (scalar, array, or Series)
        z_threshold: Z-score at which health should be very low (default 5.0)
        steepness: Controls sigmoid slope (default 1.5, higher=sharper transition)
    
    Returns:
        Health index 0-100 (same type as input)
    """
    import numpy as np
    
    # Handle various input types
    abs_z = np.abs(fused_z)
    
    # Sigmoid centered at z_threshold/2, with steepness controlling transition sharpness
    # At z=0: normalized << 0, sigmoid 0, health  100
    # At z=z_threshold/2: normalized = 0, sigmoid = 0.5, health = 50
    # At z=z_threshold: normalized > 0, sigmoid   0.85, health   15
    normalized = (abs_z - z_threshold / 2) / (z_threshold / 4)
    sigmoid = 1 / (1 + np.exp(-normalized * steepness))
    
    health = 100.0 * (1 - sigmoid)
    
    # Ensure bounds
    return np.clip(health, 0.0, 100.0)


# ==================== MAIN OUTPUT MANAGER CLASS ====================


@dataclass
class OutputBatch:
    """Represents a batch of outputs to be written together."""
    sql_operations: List[Tuple[str, pd.DataFrame, Dict[str, Any]]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # OUT-18: Batch tracking for flush triggers
    created_at: float = field(default_factory=time.time)
    total_rows: int = 0


class OutputManager:
    """
    Unified output manager that consolidates all scattered output generation.
    
    Features:
    - Batched writes for improved I/O performance
    - Smart SQL health checking with caching
    - Automatic file/SQL dual-write coordination
    - Thread-safe operations with connection pooling
    - Intelligent error handling and fallback strategies
    """
    
    def __init__(self, 
                 sql_client=None, 
                 run_id: Optional[str] = None,
                 equip_id: Optional[int] = None,
                 batch_size: int = 5000,
                 enable_batching: bool = True,
                 sql_health_cache_seconds: float = 60.0,
                 max_io_workers: int = 8,
                 batch_flush_rows: int = 1000,
                 batch_flush_seconds: float = 30.0,
                 max_in_flight_futures: int = 50):
        self.sql_client = sql_client
        self.run_id = run_id
        self.equip_id = equip_id
        self.batch_size = batch_size
        self._batched_transaction_active = False
        self.enable_batching = enable_batching
        self.max_io_workers = max_io_workers
        
        # OUT-18: Batch flush triggers and backpressure
        self.batch_flush_rows = batch_flush_rows  # Flush after N rows
        self.batch_flush_seconds = batch_flush_seconds  # Flush after N seconds
        self.max_in_flight_futures = max_in_flight_futures  # Max concurrent operations

        self.stats = {
            'sql_writes': 0,
            'total_rows': 0,
            'sql_health_checks': 0,
            'sql_failures': 0,
            'write_time': 0.0
        }

        self._sql_health_cache: Tuple[float, bool] = (0.0, False)
        self._sql_health_cache_duration = sql_health_cache_seconds

        self._current_batch = OutputBatch()
        self._batch_lock = threading.Lock()
        
        # OUT-18: Track in-flight operations for backpressure
        self._in_flight_futures: List[Any] = []  # List of active futures
        self._futures_lock = threading.Lock()

        # lightweight caches for table probes
        self._table_exists_cache: Dict[str, bool] = {}
        self._table_columns_cache: Dict[str, set] = {}
        self._table_insertable_cache: Dict[str, set] = {}
        
        # PERF-OPT: Track tables that have been bulk pre-deleted (skip individual DELETE)
        self._bulk_predeleted_tables: set = set()
        
        # FCST-15: Artifact cache for SQL-only mode
        # Stores DataFrames written to files/SQL so they can be consumed by downstream modules
        # without file system dependencies
        self._artifact_cache: Dict[str, pd.DataFrame] = {}

        # Minimal per-table NOT NULL requirements with safe defaults
        # Only for tables in ALLOWED_TABLES that need repair defaults.
        # 'ts' indicates we should use a sentinel timestamp (1900-01-01 00:00:00)
        self._sql_required_defaults: Dict[str, Dict[str, Any]] = {
            # TIER 1: Core pipeline output
            'ACM_HealthTimeline': {
                'Timestamp': 'ts', 'HealthIndex': 0.0, 'HealthZone': 'GOOD', 'FusedZ': 0.0
            },
            'ACM_RegimeTimeline': {
                'Timestamp': 'ts', 'RegimeLabel': -1, 'RegimeState': 'unknown'
            },
            # TIER 2: Forecasting
            'ACM_HealthForecast': {
                'Timestamp': 'ts', 'ForecastHealth': 0.0, 'CI_Lower': 0.0, 'CI_Upper': 0.0,
                'Method': 'ExponentialSmoothing'
            },
            'ACM_FailureForecast': {
                'Timestamp': 'ts', 'FailureProb': 0.0, 'ThresholdUsed': 50.0,
                'Method': 'GaussianTail'
            },
            'ACM_SensorForecast': {
                'Timestamp': 'ts', 'SensorName': 'UNKNOWN', 'ForecastValue': 0.0,
                'CI_Lower': 0.0, 'CI_Upper': 0.0, 'Method': 'LinearTrend'
            },
            'ACM_RUL': {
                'RUL_Hours': 0.0, 'Method': 'Multipath'
            },
            # TIER 4: Diagnostics
            'ACM_SensorDefects': {
                'DetectorType': 'UNKNOWN', 'Severity': 'LOW', 'ViolationCount': 0, 'ViolationPct': 0.0,
                'MaxZ': 0.0, 'AvgZ': 0.0, 'CurrentZ': 0.0, 'ActiveDefect': 0
            },
            'ACM_SensorHotspots': {
                'SensorName': 'UNKNOWN', 'MaxTimestamp': 'ts', 'LatestTimestamp': 'ts',
                'MaxAbsZ': 0.0, 'MaxSignedZ': 0.0, 'LatestAbsZ': 0.0, 'LatestSignedZ': 0.0,
                'ValueAtPeak': 0.0, 'LatestValue': 0.0, 'TrainMean': 0.0, 'TrainStd': 0.0,
                'AboveWarnCount': 0, 'AboveAlertCount': 0
            },
            'ACM_EpisodeDiagnostics': {
                'EpisodeID': 0, 'StartTime': 'ts', 'EndTime': 'ts', 'PeakZ': 0.0,
                'DurationHours': 0.0, 'TopSensor1': 'UNKNOWN', 'Severity': 'UNKNOWN',
                'severity_reason': 'UNKNOWN', 'AvgZ': 0.0, 'min_health_index': 0.0
            },
        }

        Console.info("" + f"Manager initialized (batch_size={batch_size}, batching={'ON' if enable_batching else 'OFF'}, sql_cache={sql_health_cache_seconds}s, io_workers={max_io_workers}, flush={batch_flush_rows} rows/{batch_flush_seconds}s, max_futures={max_in_flight_futures})", component="OUTPUT")
    
    @contextmanager
    def batched_transaction(self):
        """
        Context manager for writing multiple tables in a single transaction.
        Improves performance by reducing commit overhead when writing many tables.
        
        Usage:
            with output_mgr.batched_transaction():
                output_mgr.write_table("Table1", df1, "sql")
                output_mgr.write_table("Table2", df2, "sql")
                # Single commit after all writes
        """
        if self._batched_transaction_active:
            # Nested transaction - just pass through
            yield
            return
        
        if self.sql_client is None:
            # No SQL client - just pass through
            yield
            return
        
        self._batched_transaction_active = True
        start_time = time.time()
        
        try:
            Console.info("" + "Starting batched transaction", component="OUTPUT")
            yield
            # Commit at end of transaction
            try:
                if hasattr(self.sql_client, "commit"):
                    self.sql_client.commit()
                    Console.info("" + "Called sql_client.commit()", component="OUTPUT")
                elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                    if not getattr(self.sql_client.conn, "autocommit", True):
                        self.sql_client.conn.commit()
                        Console.info("" + "Called sql_client.conn.commit()", component="OUTPUT")
                    else:
                        Console.warn("Autocommit is ON - no explicit commit needed", component="OUTPUT", equip_id=self.equip_id, run_id=self.run_id)
                elapsed = time.time() - start_time
                Console.info("" + f"Batched transaction committed ({elapsed:.2f}s)", component="OUTPUT")
            except Exception as e:
                Console.error(f"Batched transaction commit failed: {e}", component="OUTPUT", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                raise
        except Exception as e:
            # Rollback on error
            try:
                if hasattr(self.sql_client, "rollback"):
                    self.sql_client.rollback()
                elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "rollback"):
                    self.sql_client.conn.rollback()
                Console.error(f"Batched transaction rolled back: {e}", component="OUTPUT", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
            except:
                pass
            raise
        finally:
            self._batched_transaction_active = False
    
    # ==================== DATA LOADING METHOD ====================
    
    def load_data(self, cfg: Dict[str, Any], start_utc: Optional[pd.Timestamp] = None, end_utc: Optional[pd.Timestamp] = None, equipment_name: Optional[str] = None):
        """
        Load training and scoring data from SQL historian.
        
        Args:
            cfg: Configuration dictionary
            start_utc: Start time for SQL window queries
            end_utc: End time for SQL window queries
            equipment_name: Equipment name for SQL historian queries (e.g., 'FD_FAN', 'GAS_TURBINE')
        """
        if not self.sql_client:
            raise ValueError("[DATA] SQL client required but not available")
        if not equipment_name:
            raise ValueError("[DATA] equipment_name parameter required")
        return self._load_data_from_sql(cfg, equipment_name, start_utc, end_utc)
    
    def _load_data_from_sql(self, cfg: Dict[str, Any], equipment_name: str, start_utc: Optional[pd.Timestamp], end_utc: Optional[pd.Timestamp], is_coldstart: bool = False):
        """
        Load training and scoring data from SQL historian using stored procedure.
        
        Args:
            cfg: Configuration dictionary
            equipment_name: Equipment name (e.g., 'FD_FAN', 'GAS_TURBINE')
            start_utc: Start time for query window
            end_utc: End time for query window
            is_coldstart: If True, split data for coldstart training. If False, use all data for scoring.
        
        Returns:
            Tuple of (train_df, score_df, DataMeta)
        """
        data_cfg = cfg.get("data", {})
        ts_col = _cfg_get(data_cfg, "timestamp_col", "EntryDateTime")
        min_train_samples = int(_cfg_get(data_cfg, "min_train_samples", 10))
        
        # SQL mode requires explicit time windows
        if not start_utc or not end_utc:
            raise ValueError("[DATA] SQL mode requires start_utc and end_utc parameters")
        
        # COLD-02: Configurable cold-start split ratio (default 0.6 = 60% train, 40% score)
        # Only used during coldstart - regular batch mode uses ALL data for scoring
        cold_start_split_ratio = float(_cfg_get(data_cfg, "cold_start_split_ratio", 0.6))
        if not (0.1 <= cold_start_split_ratio <= 0.9):
            Console.warn(f"Invalid cold_start_split_ratio={cold_start_split_ratio}, using default 0.6", component="DATA", invalid_value=cold_start_split_ratio, equipment=equipment_name)
            cold_start_split_ratio = 0.6
        
        min_train_samples = int(_cfg_get(data_cfg, "min_train_samples", 500))
        
        Console.info("" + f"Loading from SQL historian: {equipment_name}", component="DATA")
        Console.info("" + f"Time range: {start_utc} to {end_utc}", component="DATA")
        
        # Call stored procedure to get all data for time range
        # Pass EquipmentName directly - SP will resolve to correct data table (e.g., FD_FAN_Data)
        try:
            if self.sql_client is None:
                raise ValueError("[DATA] SQL mode requested but no SQL client available")
            cur = cast(Any, self.sql_client).cursor()
            # Pass EquipmentName to stored procedure (SP resolves to {EquipmentName}_Data table)
            cur.execute(
                "EXEC dbo.usp_ACM_GetHistorianData_TEMP @StartTime=?, @EndTime=?, @EquipmentName=?",
                (start_utc, end_utc, equipment_name)
            )
            
            # Fetch all rows
            rows = cur.fetchall()
            if not rows:
                raise ValueError(f"[DATA] No data returned from SQL historian for {equipment_name} in time range")
            
            # Get column names from cursor description
            columns = [desc[0] for desc in cur.description]
            
            # Convert to DataFrame
            df_all = pd.DataFrame.from_records(rows, columns=columns)
            
            Console.info("" + f"Retrieved {len(df_all)} rows from SQL historian", component="DATA")
            
        except Exception as e:
            Console.error(f"Failed to load from SQL historian: {e}", component="DATA", equipment=equipment_name, error_type=type(e).__name__, error=str(e)[:200])
            raise
        finally:
            try:
                cur.close()
            except:
                pass
        
        # Validate sufficient data
        # For coldstart, enforce minimum. For incremental scoring, allow smaller batches.
        required_minimum = min_train_samples if is_coldstart else max(10, min_train_samples // 10)
        if len(df_all) < required_minimum:
            raise ValueError(f"[DATA] Insufficient data from SQL historian: {len(df_all)} rows (minimum {required_minimum} required)")

        # Robust timestamp handling for SQL historian: if configured column is missing
        # but the standard EntryDateTime column is present, fall back to it.
        if ts_col not in df_all.columns and "EntryDateTime" in df_all.columns:
            Console.warn(
                f"[DATA] Timestamp column '{ts_col}' not found in SQL historian results; "
                "falling back to 'EntryDateTime'.", component="DATA", configured_col=ts_col, fallback_col="EntryDateTime", equipment=equipment_name
            )
            ts_col = "EntryDateTime"
        
        # Split into train/score based on mode
        if is_coldstart:
            # COLDSTART MODE: Split data for initial model training
            split_idx = int(len(df_all) * cold_start_split_ratio)
            train_raw = df_all.iloc[:split_idx].copy()
            score_raw = df_all.iloc[split_idx:].copy()
            
            # Warn if training samples below minimum
            if len(train_raw) < min_train_samples:
                Console.warn(f"Training data ({len(train_raw)} rows) is below recommended minimum ({min_train_samples} rows)", component="DATA", actual_rows=len(train_raw), min_required=min_train_samples, equipment=equipment_name)
                Console.warn(f"Model quality may be degraded. Consider: wider time window, higher split_ratio (current: {cold_start_split_ratio:.2f})", component="DATA", split_ratio=cold_start_split_ratio, equipment=equipment_name)
            
            Console.info("" + f"COLDSTART Split ({cold_start_split_ratio:.1%}): {len(train_raw)} train rows, {len(score_raw)} score rows", component="DATA")
        else:
            # REGULAR BATCH MODE: Use ALL data for scoring, load baseline from cache
            train_raw = pd.DataFrame()  # Empty train, will be loaded from baseline_buffer
            score_raw = df_all.copy()
            Console.info("" + f"BATCH MODE: All {len(score_raw)} rows allocated to scoring (baseline from cache)", component="DATA")
        
        # Parse timestamps / index
        # Handle empty train in batch mode
        if len(train_raw) == 0 and not is_coldstart:
            # Create empty DataFrame with DatetimeIndex matching score columns
            train = pd.DataFrame(columns=train_raw.columns)
            train.index = pd.DatetimeIndex([], name=ts_col)
        else:
            train = _parse_ts_index(train_raw, ts_col)
        
        score = _parse_ts_index(score_raw, ts_col)
        
        # Filter future timestamps
        now_cutoff = _future_cutoff_ts(cfg)
        train, tz_stripped_train, future_train = _coerce_local_and_filter_future(train, "TRAIN", now_cutoff)
        score, tz_stripped_score, future_score = _coerce_local_and_filter_future(score, "SCORE", now_cutoff)
        tz_stripped_total = tz_stripped_train + tz_stripped_score
        future_rows_total = future_train + future_score
        
        # Validate training sample count (skip in batch mode - train comes from baseline_buffer)
        if len(train) < min_train_samples and is_coldstart:
            Console.warn(f"Training data ({len(train)} rows) is below recommended minimum ({min_train_samples} rows)", component="DATA", actual_rows=len(train), min_required=min_train_samples, equipment=equipment_name, mode="coldstart")
        
        # Keep numeric only (same set across train/score)
        if len(train) == 0 and not is_coldstart:
            # BATCH MODE: Train is empty, use all score columns
            # Train will be loaded from baseline_buffer later in acm_main.py
            score_num = _infer_numeric_cols(score)
            kept = sorted(score_num)
            dropped = [c for c in score.columns if c not in kept]
            train = pd.DataFrame(columns=kept)  # Empty train with correct columns
            score = score[kept]
            score = score.astype(np.float32)
            Console.info("" + f"BATCH MODE: Train empty (will load from baseline_buffer), using all {len(kept)} score columns", component="DATA")
        else:
            # COLDSTART MODE or TRAIN EXISTS: Use intersection of train/score columns
            train_num = _infer_numeric_cols(train)
            score_num = _infer_numeric_cols(score)
            kept = sorted(list(set(train_num).intersection(score_num)))
            dropped = [c for c in train.columns if c not in kept]
            train = train[kept]
            score = score[kept]
            train = train.astype(np.float32)
            score = score.astype(np.float32)
        
        Console.info("" + f"Kept {len(kept)} numeric columns, dropped {len(dropped)} non-numeric", component="DATA")
        
        # Cadence check + resampling (same logic as CSV mode)
        _sampling = data_cfg.get("sampling_secs", 1)
        # Treat empty/invalid values as "auto" (let cadence be inferred)
        try:
            if _sampling in (None, "", "auto", "null"):
                sampling_secs: Optional[int] = None
            else:
                sampling_secs = int(_sampling)
        except (TypeError, ValueError):
            sampling_secs = None
        
        allow_resample = bool(_cfg_get(data_cfg, "allow_resample", True))
        resample_strict = bool(_cfg_get(data_cfg, "resample_strict", False))
        interp_method = str(_cfg_get(data_cfg, "interp_method", "linear"))
        max_fill_ratio = float(_cfg_get(data_cfg, "max_fill_ratio", _cfg_get(cfg, "runtime.max_fill_ratio", 0.20)))
        
        cad_ok_train = _check_cadence(cast(pd.DatetimeIndex, train.index), sampling_secs)
        cad_ok_score = _check_cadence(cast(pd.DatetimeIndex, score.index), sampling_secs)
        cadence_ok = bool(cad_ok_train and cad_ok_score)
        
        native_train = _native_cadence_secs(cast(pd.DatetimeIndex, train.index))
        if sampling_secs and math.isfinite(native_train) and sampling_secs < native_train:
            Console.warn(f"Requested resample ({sampling_secs}s) < native cadence ({native_train:.1f}s) - skipping to avoid upsample.", component="DATA", requested_secs=sampling_secs, native_secs=native_train, equipment=equipment_name)
            sampling_secs = None

        if sampling_secs is not None:
            base_secs = float(sampling_secs)
        else:
            base_secs = native_train if math.isfinite(native_train) else 1.0
        max_gap_secs = int(_cfg_get(data_cfg, "max_gap_secs", base_secs * 3))
        
        explode_guard_factor = float(_cfg_get(data_cfg, "explode_guard_factor", 2.0))
        will_resample = allow_resample and (not cadence_ok) and (sampling_secs is not None)
        if will_resample:
            span_secs = (train.index[-1].value - train.index[0].value) / 1e9 if len(train.index) else 0.0
            safe_sampling = float(sampling_secs) if sampling_secs is not None else 1.0
            approx_rows = int(span_secs / max(1.0, safe_sampling)) + 1
            if len(train) and approx_rows > explode_guard_factor * len(train):
                Console.warn(f"Resample would expand rows from {len(train)} -> ~{approx_rows} (>x{explode_guard_factor:.1f}). Skipping resample.", component="DATA")
                will_resample = False

        if will_resample:
            assert sampling_secs is not None
            train = _resample(train, int(sampling_secs), interp_method, resample_strict, max_gap_secs, max_fill_ratio)
            score = _resample(score, int(sampling_secs), interp_method, resample_strict, max_gap_secs, max_fill_ratio)
            train = train.astype(np.float32)
            score = score.astype(np.float32)
            cadence_ok = True

        meta = DataMeta(
            timestamp_col=ts_col,
            cadence_ok=cadence_ok,
            kept_cols=kept,
            dropped_cols=dropped,
            start_ts=train.index.min() if len(train) else pd.Timestamp.now(),
            end_ts=score.index.max() if len(score) else pd.Timestamp.now(),
            n_rows=len(train) + len(score),
            sampling_seconds=sampling_secs or native_train,
            tz_stripped=tz_stripped_total,
            future_rows_dropped=future_rows_total,
            dup_timestamps_removed=0
        )
        
        Console.info("" + f"SQL historian load complete: {len(train)} train + {len(score)} score = {len(train) + len(score)} total rows", component="DATA")
        return train, score, meta
    
    def _check_sql_health(self) -> bool:
        """Check SQL availability with caching for performance."""
        if self.sql_client is None:
            return False
        
        now = time.time()
        last_check, last_result = self._sql_health_cache
        
        # Use cached result if fresh
        if now - last_check < self._sql_health_cache_duration:
            return last_result
        
        # Perform health check
        try:
            cur = self.sql_client.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            self._sql_health_cache = (now, True)
            self.stats['sql_health_checks'] += 1
            return True
        except Exception as e:
            self._sql_health_cache = (now, False)
            self.stats['sql_health_checks'] += 1
            self.stats['sql_failures'] += 1
            Console.error(f"SQL health check failed: {e}", component="OUTPUT", equip_id=self.equip_id, error_type=type(e).__name__, error=str(e)[:200])
            return False
        finally:
            try:
                if 'cur' in locals():
                    cur.close()
            except Exception:
                pass
    
    def _prepare_dataframe_for_sql(self, df: pd.DataFrame, non_numeric_cols: Optional[set] = None) -> pd.DataFrame:
        """Prepare DataFrame for SQL insertion with robust type coercion (SQL Server safe)."""
        if df.empty:
            return df

        out = df.copy()
        non_numeric_cols = set(non_numeric_cols or set())

        # Helper: name-based timestamp detection to handle object/string timestamp columns too
        def _looks_like_ts(col_name: str) -> bool:
            c = (col_name or "").lower()
            return c in {"timestamp", "time", "ts", "datetime"} or c.endswith("_ts") or c.endswith("_time") or c.endswith("_timestamp")

        # 1) Normalize datetime columns (dtype-based) + timestamp-like object columns (name-based)
        for col in out.columns:
            if col in non_numeric_cols:
                # Still allow datetime normalization for non-numeric columns if they are timestamp-like
                pass

            is_dt = pd.api.types.is_datetime64_any_dtype(out[col])
            is_obj_ts = (not is_dt) and _looks_like_ts(col) and out[col].dtype == object

            if is_dt or is_obj_ts:
                ts_series = pd.to_datetime(out[col], errors="coerce")
                # drop timezone if present, keep UTC-naive convention
                try:
                    ts_series = ts_series.dt.tz_localize(None)
                except Exception:
                    pass
                ts_series = ts_series.dt.floor("s")

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*to_pydatetime.*", category=FutureWarning)
                    out[col] = np.array(ts_series.dt.to_pydatetime())

        # 2) Replace Inf/-Inf and NaNs to None (SQL-friendly)
        num_only = out.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(num_only.values).sum()) if not num_only.empty else 0
        if inf_count > 0:
            Console.warn(
                f"Replaced {inf_count} Inf/-Inf values with None for SQL compatibility",
                component="OUTPUT",
                inf_count=inf_count,
                columns=len(num_only.columns),
            )

        out = out.replace({np.inf: None, -np.inf: None})
        # Note: do NOT blanket-coerce everything to object early; keep types until after numeric handling

        # 3) Preserve integer types for known ID-like columns (case-insensitive)
        integer_columns_ci = {c.lower() for c in {"EquipID", "equip_id", "episode_id", "EpisodeID", "RegimeLabel", "regime_label"}}

        for col in out.columns:
            col_l = col.lower()

            # Skip caller-marked non-numeric columns for numeric coercion only
            # (but do NOT skip boolean normalization; booleans must become int for SQL)
            if pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype(object).where(pd.isna(out[col]), out[col].astype(int))
                continue

            if col_l in integer_columns_ci and pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].astype("Int64")  # nullable integer
                continue

            if col in non_numeric_cols:
                continue

            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].astype(float)

        # 4) Finally, normalize NaN/NaT to None for pyodbc binding
        out = out.where(pd.notnull(out), None)

        return out
    
    
    def _should_auto_flush(self) -> bool:
        """OUT-18: Check if batch should be automatically flushed based on triggers."""
        with self._batch_lock:
            # Size-based trigger
            if self._current_batch.total_rows >= self.batch_flush_rows:
                return True
            
            # Time-based trigger
            batch_age = time.time() - self._current_batch.created_at
            if batch_age >= self.batch_flush_seconds:
                return True
            
            return False
    
    def _wait_for_futures_capacity(self) -> None:
        """OUT-18: Block if too many in-flight operations (backpressure)."""
        while True:
            with self._futures_lock:
                # Clean up completed futures
                self._in_flight_futures = [f for f in self._in_flight_futures if not f.done()]
                
                # If we have capacity, proceed
                if len(self._in_flight_futures) < self.max_in_flight_futures:
                    break
            
            # Wait a bit before checking again
            time.sleep(0.1)
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        artifact_name: str,
        sql_table: Optional[str] = None,
        sql_columns: Optional[Dict[str, str]] = None,
        non_numeric_cols: Optional[set] = None,
        add_created_at: bool = False,
        allow_repair: bool = True,
        required: bool = False,
    ) -> Dict[str, Any]:
        """
        Write DataFrame to SQL (SQL-only; file output removed).

        Args:
            df: DataFrame to write
            artifact_name: Logical name for the artifact (used for caching/logging)
            sql_table: Optional SQL table name. If None, no SQL write is attempted.
            sql_columns: Optional column mapping for SQL (df_col -> sql_col)
            non_numeric_cols: Set of columns to treat as non-numeric for SQL preparation
            add_created_at: Whether to add CreatedAt timestamp column for SQL
            allow_repair: If False, block SQL write when required fields missing instead of auto-repairing
            required: If True, raise on write failure; if False, log warning and continue (default False for backwards-compat)

        Returns:
            Dict with SQL write results and metadata.
        """
        start_time = time.time()

        result: Dict[str, Any] = {
            "sql_written": False,
            "rows": int(len(df)),
            "inserted": 0,
            "error": None,
            "sql_table": sql_table,
            "artifact": artifact_name,
        }

        # OUT-18: auto-flush before write if needed
        if self._should_auto_flush():
            Console.info(
                f"Auto-flushing batch (rows={self._current_batch.total_rows}, age={time.time() - self._current_batch.created_at:.1f}s)",
                component="OUTPUT",
            )
            self.flush()

        # OUT-18: backpressure
        self._wait_for_futures_capacity()

        # Track rows in current batch regardless of whether we write to SQL
        with self._batch_lock:
            self._current_batch.total_rows += len(df)

        sql_df: Optional[pd.DataFrame] = None

        try:
            # If no sql_table requested, skip SQL quietly
            if not sql_table:
                return result

            # Guard: Only write to ALLOWED_TABLES (contract enforcement)
            if sql_table not in ALLOWED_TABLES:
                Console.warn(
                    f"Table '{sql_table}' not in ALLOWED_TABLES; skipping write",
                    component="OUTPUT",
                    table=sql_table,
                    rows=len(df),
                )
                result["error"] = f"Table '{sql_table}' not in ALLOWED_TABLES"
                return result

            # Skip empty DataFrames (no-op)
            if df.empty:
                return result

            # If SQL unhealthy, skip with an error string (do not throw)
            if not self._check_sql_health():
                msg = f"SQL health check failed; skipped write to {sql_table}"
                Console.warn(
                    msg,
                    component="OUTPUT",
                    table=sql_table,
                    rows=len(df),
                    equip_id=self.equip_id,
                    run_id=self.run_id,
                )
                result["error"] = msg
                return result

            # Prepare data for SQL
            sql_df = self._prepare_dataframe_for_sql(df, non_numeric_cols or set())

            # Apply column mapping (df_col -> sql_col)
            if sql_columns:
                mapped_source_cols = [c for c in sql_columns.keys() if c in sql_df.columns]
                sql_df = sql_df[mapped_source_cols].rename(columns=sql_columns)

            # Required metadata columns (all SQL tables)
            if "RunID" not in sql_df.columns:
                sql_df["RunID"] = self.run_id
            if "EquipID" not in sql_df.columns:
                sql_df["EquipID"] = int(self.equip_id or 0)

            # OUT-17: apply required defaults w/ repair policy
            sql_df, repair_info = self._apply_sql_required_defaults(sql_table, sql_df, allow_repair)

            if repair_info.get("repairs_needed"):
                Console.info(
                    f"Applied defaults to {sql_table}: {repair_info.get('missing_fields')}",
                    component="SCHEMA",
                )

            if (not allow_repair) and repair_info.get("repairs_needed"):
                raise ValueError(
                    f"Required fields missing and allow_repair=False: {repair_info.get('missing_fields')}"
                )

            # CreatedAt is optional and explicitly requested
            if add_created_at and "CreatedAt" not in sql_df.columns:
                sql_df["CreatedAt"] = pd.Timestamp.now().tz_localize(None)

            # Special-case: avoid PK collisions for maintenance recommendation (keep as-is)
            if sql_table == "ACM_MaintenanceRecommendation" and self.sql_client is not None:
                try:
                    with self.sql_client.cursor() as cur:
                        cur.execute(
                            "DELETE FROM dbo.[ACM_MaintenanceRecommendation] WHERE RunID = ? AND EquipID = ?",
                            (self.run_id, int(self.equip_id or 0)),
                        )
                    # Commit semantics: support both patterns
                    if hasattr(self.sql_client, "commit"):
                        self.sql_client.commit()
                    elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                        if not getattr(self.sql_client.conn, "autocommit", True):
                            self.sql_client.conn.commit()
                except Exception as del_ex:
                    Console.warn(
                        f"Pre-delete failed for {sql_table}: {del_ex}",
                        component="OUTPUT",
                        table=sql_table,
                        equip_id=self.equip_id,
                        run_id=self.run_id,
                        error_type=type(del_ex).__name__,
                    )

            # Route forecast tables to upsert methods, else bulk insert
            if sql_table == "ACM_HealthForecast":
                inserted = int(self._upsert_health_forecast(sql_df))
            elif sql_table == "ACM_FailureForecast":
                inserted = int(self._upsert_failure_forecast(sql_df))
            elif sql_table == "ACM_DetectorForecast_TS":
                inserted = int(self._upsert_detector_forecast_ts(sql_df))
            elif sql_table == "ACM_SensorForecast":
                inserted = int(self._upsert_sensor_forecast(sql_df))
            else:
                inserted = int(self._bulk_insert_sql(sql_table, sql_df))

            result["inserted"] = inserted
            result["sql_written"] = inserted > 0

            if result["sql_written"]:
                self.stats["sql_writes"] += 1

            return result

        except Exception as e:
            # Ensure we never reference sql_df if it wasn't built
            rows_for_log = len(sql_df) if isinstance(sql_df, pd.DataFrame) else len(df)
            
            error_msg = f"SQL write failed for {sql_table}: {str(e)[:500]}"
            
            if required:
                # For required tables, escalate to error and re-raise
                Console.error(
                    f"[CRITICAL] Required table {sql_table} write failed: {error_msg}",
                    component="OUTPUT",
                    table=sql_table,
                    rows=rows_for_log,
                    equip_id=self.equip_id,
                    run_id=self.run_id,
                    error_type=type(e).__name__,
                )
                result["error"] = error_msg
                raise RuntimeError(f"Required table write failed: {error_msg}") from e
            else:
                # For optional tables, warn and continue (backwards-compatible behavior)
                Console.warn(
                    f"SQL write failed for {sql_table}: {error_msg}",
                    component="OUTPUT",
                    table=sql_table,
                    rows=rows_for_log,
                    equip_id=self.equip_id,
                    run_id=self.run_id,
                    error_type=type(e).__name__,
                )
                result["error"] = error_msg
            result["error"] = str(e)
            self.stats["sql_failures"] += 1
            return result

        finally:
            elapsed = time.time() - start_time
            self.stats["write_time"] += elapsed
            # FCST-15: cache for downstream modules
            self._artifact_cache[artifact_name] = df

    def write_table(self, table_name: str, df: pd.DataFrame, delete_existing: bool = False) -> int:
        """Generic SQL table writer with RunID/EquipID injection, defaults, and upsert routing."""
        # Start span for this write operation (v10.3.0 tracing enhancement)
        span_context = None
        if _OBSERVABILITY_AVAILABLE and Span is not None:
            span_context = Span(
                f"persist.write",
                table=table_name,
                delete_existing=delete_existing,
            )
            span_context.__enter__()
        
        try:
            if not self._check_sql_health() or df is None or df.empty:
                if span_context:
                    span_context._span.set_attribute("acm.rows_written", 0)
                return 0
            try:
                sql_df = df.copy()
                now = pd.Timestamp.now().tz_localize(None)

                # Inject metadata
                if 'RunID' not in sql_df.columns:
                    sql_df['RunID'] = self.run_id
                if 'EquipID' not in sql_df.columns:
                    sql_df['EquipID'] = self.equip_id or 0

                # Apply required defaults/repairs
                sql_df, _ = self._apply_sql_required_defaults(table_name, sql_df, allow_repair=True)

                # Fill common fields when present
                if 'Method' in sql_df.columns:
                    sql_df['Method'] = sql_df['Method'].fillna('default')
                if 'LastUpdate' in sql_df.columns:
                    sql_df['LastUpdate'] = pd.to_datetime(sql_df['LastUpdate']).dt.tz_localize(None).fillna(now)
                if 'EarliestMaintenance' in sql_df.columns:
                    sql_df['EarliestMaintenance'] = pd.to_datetime(sql_df['EarliestMaintenance']).dt.tz_localize(None).fillna(now)
                if 'PreferredWindowStart' in sql_df.columns:
                    sql_df['PreferredWindowStart'] = pd.to_datetime(sql_df['PreferredWindowStart']).dt.tz_localize(None).fillna(now)
                if 'PreferredWindowEnd' in sql_df.columns:
                    sql_df['PreferredWindowEnd'] = pd.to_datetime(sql_df['PreferredWindowEnd']).dt.tz_localize(None).fillna(now)

                # Normalize types/nulls for SQL
                sql_df = self._prepare_dataframe_for_sql(sql_df)

                # Optional delete-existing by RunID (+EquipID when available)
                if delete_existing and self.sql_client is not None and self.run_id:
                    try:
                        with self.sql_client.cursor() as cur:
                            if 'EquipID' in sql_df.columns and self.equip_id is not None:
                                cur.execute(f"DELETE FROM dbo.[{table_name}] WHERE RunID = ? AND EquipID = ?", (self.run_id, int(self.equip_id or 0)))
                            else:
                                cur.execute(f"DELETE FROM dbo.[{table_name}] WHERE RunID = ?", (self.run_id,))
                            if hasattr(self.sql_client, "commit"):
                                self.sql_client.commit()
                            elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                                if not getattr(self.sql_client.conn, "autocommit", True):
                                    self.sql_client.conn.commit()
                    except Exception as del_ex:
                        Console.warn(f"delete_existing failed for {table_name}: {del_ex}", component="OUTPUT", table=table_name, equip_id=self.equip_id, run_id=self.run_id, error_type=type(del_ex).__name__)

                # Route known upsert tables (v10 schema)
                if table_name == "ACM_HealthForecast":
                    rows_written = self._upsert_health_forecast(sql_df)
                elif table_name == "ACM_FailureForecast":
                    rows_written = self._upsert_failure_forecast(sql_df)
                elif table_name == "ACM_SensorForecast":
                    rows_written = self._upsert_sensor_forecast(sql_df)
                elif table_name == "ACM_PCA_Metrics":
                    rows_written = self._upsert_pca_metrics(sql_df)
                else:
                    # Default: bulk insert
                    rows_written = self._bulk_insert_sql(table_name, sql_df)
                
                # Record rows written in span
                if span_context:
                    span_context._span.set_attribute("acm.rows_written", rows_written)
                
                return rows_written
            except Exception as e:
                Console.warn(f"write_table failed for {table_name}: {e}", component="OUTPUT", table=table_name, rows=len(df) if df is not None else 0, equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                if span_context:
                    span_context._span.set_attribute("acm.error", True)
                return 0
        finally:
            if span_context:
                span_context.__exit__(None, None, None)

    def _apply_sql_required_defaults(self, table_name: str, df: pd.DataFrame, allow_repair: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fill required columns with safe defaults to satisfy NOT NULL constraints.
        Only applies to known analytics tables; unknown tables pass through.
        
        OUT-17: Now returns tuple of (repaired_df, repair_info) with audit columns added.
        
        Args:
            table_name: SQL table name
            df: DataFrame to repair
            allow_repair: If False, don't apply defaults but still report missing fields
            
        Returns:
            Tuple of (repaired DataFrame, repair_info dict with keys:
                      'repairs_needed', 'missing_fields', 'repaired_count')
        """
        repair_info = {
            'repairs_needed': False,
            'missing_fields': [],
            'repaired_count': 0
        }
        
        if df is None or df.empty:
            return df, repair_info
            
        req = self._sql_required_defaults.get(table_name)
        if not req:
            return df, repair_info
            
        out = df.copy()
        # CRIT-04: Use current local-naive timestamp instead of 1900-01-01
        sentinel_ts = pd.Timestamp.now().tz_localize(None)
        sentinel_date = sentinel_ts.date()
        filled = {}
        missing_fields = []
        
        for col, default in req.items():
            if default == 'ts':
                val = sentinel_ts
            elif default == 'date':
                val = sentinel_date
            else:
                val = default
                
            needs_repair = False
            
            if col not in out.columns:
                missing_fields.append(col)
                needs_repair = True
                if allow_repair:
                    out[col] = val
                    filled[col] = 'added'
            else:
                # Check for nulls
                try:
                    null_mask = out[col].isna()
                except Exception:
                    null_mask = None
                    
                if null_mask is not None and getattr(null_mask, 'any', lambda: False)():
                    count = int(null_mask.sum())
                    missing_fields.append(f"{col}({count} nulls)")
                    needs_repair = True
                    if allow_repair:
                        out.loc[null_mask, col] = val
                        filled[col] = count
                        
            if needs_repair:
                repair_info['repairs_needed'] = True
                repair_info['repaired_count'] += 1
        
        repair_info['missing_fields'] = missing_fields
        
        if filled:
            # Debug-level only - applying defaults is expected behavior, not a warning
            pass  # Console.debug(f"{table_name}: applied defaults {filled}", component="SCHEMA")
        if not allow_repair and repair_info['repairs_needed']:
            Console.warn(f"{table_name}: repairs blocked (allow_repair=False), missing: {missing_fields}", component="SCHEMA", table=table_name, missing_fields=missing_fields)
            
        return out, repair_info

    def _bulk_insert_sql(self, table_name: str, df: pd.DataFrame) -> int:
        """Perform bulk SQL insert with optimized batching and robust commit."""
        _sql_start_time = time.perf_counter()  # Track for observability
        
        if df.empty:
            return 0
        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table_name}")
        if self.sql_client is None:
            return 0

        inserted = 0
        if self.sql_client is None:
            return 0
        cursor_factory = lambda: cast(Any, self.sql_client).cursor()

        # Optional: skip if the table doesn't exist (avoids noisy logs on dev DBs)
        exists = self._table_exists_cache.get(table_name)
        if exists is None:
            exists = _table_exists(cursor_factory, table_name)
            self._table_exists_cache[table_name] = bool(exists)
        if not exists:
            Console.warn(f"Skipping write: table dbo.[{table_name}] not found", component="OUTPUT", table=table_name, equip_id=self.equip_id, run_id=self.run_id)
            return 0

        cur = cursor_factory()
        try:
            try:
                cur.fast_executemany = True
            except Exception:
                pass

            # discover table columns and scope delete by RunID only when present
            # determine insertable columns (exclude identity)
            table_cols: set
            try:
                if table_name in self._table_insertable_cache:
                    table_cols = self._table_insertable_cache[table_name]
                else:
                    cols = set(_get_insertable_columns(cursor_factory, table_name))
                    if not cols:
                        cols = set(_get_table_columns(cursor_factory, table_name))
                    self._table_insertable_cache[table_name] = cols
                    table_cols = cols
            except Exception:
                if table_name in self._table_columns_cache:
                    table_cols = self._table_columns_cache[table_name]
                else:
                    cols_all = set(_get_table_columns(cursor_factory, table_name))
                    self._table_columns_cache[table_name] = cols_all
                    table_cols = cols_all
            # CRIT-01/HIGH-02: Standardize DELETE-before-INSERT for tables keyed by RunID
            # Apply for all tables with RunID in schema, scoped by EquipID when available.
            # PERF-OPT: Skip if table was already bulk pre-deleted
            if table_name in self._bulk_predeleted_tables:
                pass  # Already deleted in bulk operation
            else:
                try:
                    if "RunID" in table_cols and self.run_id:
                        if "EquipID" in table_cols and self.equip_id is not None:
                            rows_deleted = cur.execute(
                                f"DELETE FROM dbo.[{table_name}] WHERE RunID = ? AND EquipID = ?",
                                (self.run_id, int(self.equip_id or 0))
                            ).rowcount
                        else:
                            rows_deleted = cur.execute(
                                f"DELETE FROM dbo.[{table_name}] WHERE RunID = ?",
                                (self.run_id,)
                            ).rowcount
                        if rows_deleted and rows_deleted > 0:
                            Console.info("" + f"Deleted {rows_deleted} existing rows from {table_name} for RunID={self.run_id}, EquipID={self.equip_id}", component="OUTPUT")
                except Exception as del_ex:
                    Console.warn(f"Standard pre-delete for {table_name} failed: {del_ex}", component="OUTPUT", table=table_name, equip_id=self.equip_id, run_id=self.run_id, error_type=type(del_ex).__name__)

            # Note: RUL tables are already covered by standardized pre-delete above.

            # only insert columns that actually exist in the table
            columns = [c for c in df.columns if c in table_cols]
            cols_str = ", ".join(f"[{c}]" for c in columns)
            placeholders = ", ".join(["?"] * len(columns))
            insert_sql = f"INSERT INTO dbo.[{table_name}] ({cols_str}) VALUES ({placeholders})"

            # Clean NaN/Inf values for SQL Server compatibility (pyodbc cannot handle these)
            import numpy as np
            import warnings
            df_clean = df[columns].copy()
            
            # Replace 'N/A' strings with NaN (common in CSV data)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting.*')
                df_clean = df_clean.replace(['N/A', 'n/a', 'NA', 'na', '#N/A'], np.nan)
            
            # Convert timestamp columns to datetime objects FIRST
            for col in df_clean.columns:
                if 'timestamp' in col.lower() or col in ['Date', 'date']:
                    try:
                        # Try standard format first, then let pandas infer
                        df_clean[col] = pd.to_datetime(df_clean[col], format='mixed', errors='coerce')
                        try:
                            # Drop timezone info if present
                            df_clean[col] = df_clean[col].dt.tz_localize(None)
                        except Exception:
                            pass
                        # Log if any conversions failed
                        null_count = df_clean[col].isna().sum()
                        if null_count > 0:
                            Console.warn(f"{null_count} timestamps failed to parse in column {col}", component="OUTPUT", table=table_name, column=col, failed_count=null_count)
                    except Exception as ex:
                        Console.warn(f"Timestamp conversion failed for {col}: {ex}", component="OUTPUT", table=table_name, column=col, error_type=type(ex).__name__)
                        pass  # Not a timestamp column, skip conversion
            
            # Replace extreme float values BEFORE replacing NaN (so we can use .abs())
            import numpy as np
            for col in df_clean.columns:
                if df_clean[col].dtype in [np.float64, np.float32]:
                    # Check for extreme values that will cause SQL Server errors
                    # Use pandas notnull() to avoid NaN before checking absolute value
                    valid_mask = pd.notnull(df_clean[col])
                    if valid_mask.any():
                        # CRIT-06: Lower extreme threshold to 1e38 for SQL safety
                        extreme_mask = valid_mask & (df_clean[col].abs() > 1e38)
                        if extreme_mask.any():
                            Console.warn(f"Replacing {extreme_mask.sum()} extreme float values in {table_name}.{col}", component="OUTPUT", table=table_name, column=col, extreme_count=int(extreme_mask.sum()))
                            df_clean.loc[extreme_mask, col] = None
            
            # Replace Inf with None, then replace all NaN with None for SQL NULL
            df_clean = df_clean.replace([np.inf, -np.inf], None)
            # Convert to object dtype to allow None values, then replace NaN with None
            df_clean = df_clean.astype(object).where(pd.notnull(df_clean), None)
            
            records = [tuple(row) for row in df_clean.itertuples(index=False, name=None)]
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i+self.batch_size]
                try:
                    cur.executemany(insert_sql, batch)
                    inserted += len(batch)
                except Exception as batch_error:
                    # Log failed batch with first few records for debugging
                    sample = batch[:3] if len(batch) > 3 else batch
                    Console.error(f"Batch insert failed for {table_name} (sample: {sample}): {batch_error}", component="OUTPUT", table=table_name, batch_size=len(batch), equip_id=self.equip_id, run_id=self.run_id, error_type=type(batch_error).__name__, error=str(batch_error)[:200])
                    raise
        except Exception as e:
            Console.error(f"SQL insert failed for {table_name}: {e}", component="OUTPUT", table=table_name, rows=len(df), equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

        # Only commit if not in batched transaction mode
        if not self._batched_transaction_active:
            try:
                if hasattr(self.sql_client, "commit"):
                    self.sql_client.commit()
                elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                    # some wrappers expose .conn
                    if not getattr(self.sql_client.conn, "autocommit", True):
                        self.sql_client.conn.commit()
            except Exception as e:
                Console.error(f"SQL commit failed for {table_name}: {e}", component="OUTPUT", table=table_name, equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                raise

        Console.info("" + f"SQL insert to {table_name}: {inserted} rows", component="OUTPUT")
        
        # P1: Track SQL ops for observability metrics
        if _OBSERVABILITY_AVAILABLE and record_sql_op:
            try:
                duration_ms = (time.perf_counter() - _sql_start_time) * 1000
                record_sql_op(
                    equipment=getattr(self, 'equipment', ''),
                    table=table_name,
                    operation='insert',
                    rows=inserted,
                    duration_ms=duration_ms,
                )
            except Exception:
                pass  # Non-critical tracking
        
        return inserted

    
    # ==================== ARTIFACT CACHE METHODS (FCST-15) ====================
    
    def get_cached_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a cached DataFrame from the artifact cache.
        
        This enables SQL-only mode by allowing downstream modules (forecast, RUL)
        to access previously written tables without file system dependencies.
        
        Args:
            table_name: Name of the table/file to retrieve (e.g., "scores.csv")
            
        Returns:
            DataFrame if found in cache, None otherwise
            
        Example:
            >>> scores = output_manager.get_cached_table("scores.csv")
            >>> if scores is not None:
            ...     # Use scores for forecasting
        """
        cached = self._artifact_cache.get(table_name)
        if cached is not None:
            Console.info("" + f"Retrieved {table_name} from artifact cache ({len(cached)} rows)", component="OUTPUT")
            return cached.copy()  # Return copy to prevent mutation
        else:
            Console.warn(f"Table {table_name} not found in artifact cache", component="OUTPUT", table=table_name, available_tables=list(self._artifact_cache.keys())[:5])
            return None
    
    def clear_artifact_cache(self) -> None:
        """Clear the artifact cache to free memory."""
        count = len(self._artifact_cache)
        self._artifact_cache.clear()
        Console.info("" + f"Cleared artifact cache ({count} tables)", component="OUTPUT")
    
    def list_cached_tables(self) -> List[str]:
        """Return list of tables currently in the artifact cache."""
        return list(self._artifact_cache.keys())

    def write_pca_metrics(self, pca_detector=None, df=None, run_id=None) -> int:
        """Write PCA metrics to ACM_PCA_Metrics table (SQL-only).
        
        Args:
            pca_detector: PCASubspaceDetector instance (new style)
            df: Pre-built DataFrame (legacy style, optional)
            run_id: Run ID for legacy style
        """
        if not self._check_sql_health():
            return 0
            
        try:
            # New style: extract metrics from PCA detector
            if pca_detector is not None:
                # PCA may be None if insufficient samples (< 2) during fit - this is expected
                if not hasattr(pca_detector, 'pca') or pca_detector.pca is None:
                    # Silently skip - not an error, just insufficient training data
                    return 0
                    
                # Build metrics in long format for SQL schema:
                # RunID, EquipID, ComponentName, MetricType, Value, Timestamp
                metrics_rows = []
                timestamp = datetime.now()
                run_id_val = run_id or self.run_id
                equip_id_val = self.equip_id or 0
                
                metrics_rows.append({
                    'RunID': run_id_val,
                    'EquipID': equip_id_val,
                    'ComponentName': 'PCA',
                    'MetricType': 'n_components',
                    'Value': float(pca_detector.pca.n_components_),
                    'Timestamp': timestamp
                })
                metrics_rows.append({
                    'RunID': run_id_val,
                    'EquipID': equip_id_val,
                    'ComponentName': 'PCA',
                    'MetricType': 'variance_explained',
                    'Value': float(pca_detector.pca.explained_variance_ratio_.sum()),
                    'Timestamp': timestamp
                })
                metrics_rows.append({
                    'RunID': run_id_val,
                    'EquipID': equip_id_val,
                    'ComponentName': 'PCA',
                    'MetricType': 'n_features',
                    'Value': float(len(pca_detector.keep_cols)),
                    'Timestamp': timestamp
                })
                
                sql_df = pd.DataFrame(metrics_rows)
            # Legacy style: use provided dataframe
            elif df is not None:
                # Legacy path: DataFrame must have ComponentName and MetricType columns
                if 'ComponentName' not in df.columns or 'MetricType' not in df.columns:
                    Console.warn("write_pca_metrics legacy path requires ComponentName and MetricType columns. "
                                 + "Provided DataFrame has columns: " + str(list(df.columns)), component="OUTPUT", columns=list(df.columns))
                    return 0
                sql_df = df.copy()
            else:
                Console.warn("write_pca_metrics called without pca_detector or df", component="OUTPUT", equip_id=self.equip_id, run_id=self.run_id)
                return 0
            
            # Use MERGE upsert to handle duplicate keys gracefully
            return self._upsert_pca_metrics(sql_df)
            
        except Exception as e:
            Console.warn(f"write_pca_metrics failed: {e}", component="OUTPUT", equip_id=self.equip_id, error_type=type(e).__name__, error=str(e)[:200])
            return 0

    def write_pca_loadings(self, df: pd.DataFrame, run_id: str = None) -> int:
        """Write PCA loadings to ACM_PCA_Loadings table.
        
        Schema: RunID, EquipID, EntryDateTime, ComponentNo, ComponentID, 
                Sensor, FeatureName, Loading, CreatedAt
        
        Args:
            df: DataFrame with columns: RunID, EntryDateTime, ComponentNo, Sensor, Loading
            run_id: Run ID (optional, can come from df)
        
        Returns:
            Number of rows written
        """
        if df is None or df.empty:
            return 0
        if not self._check_sql_health():
            return 0
        
        try:
            sql_df = df.copy()
            
            # Ensure required columns
            if 'EquipID' not in sql_df.columns:
                sql_df['EquipID'] = self.equip_id or 0
            if 'RunID' not in sql_df.columns:
                sql_df['RunID'] = run_id or self.run_id or ''
            if 'EntryDateTime' not in sql_df.columns:
                sql_df['EntryDateTime'] = datetime.now()
            if 'ComponentID' not in sql_df.columns:
                sql_df['ComponentID'] = sql_df.get('ComponentNo', 0)
            if 'FeatureName' not in sql_df.columns:
                sql_df['FeatureName'] = sql_df.get('Sensor', '')
            
            # Select only the columns the table expects
            keep_cols = ['RunID', 'EquipID', 'EntryDateTime', 'ComponentNo', 'ComponentID', 
                         'Sensor', 'FeatureName', 'Loading']
            keep_cols = [c for c in keep_cols if c in sql_df.columns]
            sql_df = sql_df[keep_cols]
            
            return self._bulk_insert_sql('ACM_PCA_Loadings', sql_df)
            
        except Exception as e:
            Console.warn(f"write_pca_loadings failed: {e}", component="OUTPUT",
                        equip_id=self.equip_id, run_id=run_id, error=str(e)[:200])
            return 0

    def _upsert_pca_metrics(self, df: pd.DataFrame) -> int:
        """Upsert PCA metrics using DELETE + INSERT pattern.
        
        Actual table schema:
        - ID (identity)
        - RunID (required)
        - EquipID (required) 
        - ComponentIndex (required)
        - ExplainedVariance (nullable)
        - CumulativeVariance (nullable)
        - Eigenvalue (nullable)
        - CreatedAt (default)
        
        Handles both:
        - New format: ComponentIndex, ExplainedVariance, etc.
        - Legacy format: ComponentName, MetricType, Value (convert to new format)
        """
        if df.empty or self.sql_client is None:
            return 0
        
        try:
            conn = self.sql_client.conn
            cursor = conn.cursor()
            
            # Check if this is legacy format (ComponentName, MetricType, Value)
            is_legacy_format = 'MetricType' in df.columns and 'ComponentName' in df.columns
            
            if is_legacy_format:
                # Convert legacy format to new format
                # Pivot MetricType values into columns
                pivot_records = {}  # (RunID, EquipID, ComponentIdx) -> {metric: value}
                
                for _, row in df.iterrows():
                    run_id = row['RunID']
                    equip_id = row['EquipID']
                    metric_type = row.get('MetricType', '')
                    value = row.get('Value', 0.0)
                    
                    # Extract component index from ComponentName (e.g., "PC1" -> 0, "PC2" -> 1)
                    comp_name = str(row.get('ComponentName', 'PC1'))
                    if comp_name.startswith('PC'):
                        try:
                            comp_idx = int(comp_name[2:]) - 1
                        except ValueError:
                            comp_idx = 0
                    else:
                        comp_idx = 0
                    
                    key = (run_id, equip_id, comp_idx)
                    if key not in pivot_records:
                        pivot_records[key] = {}
                    
                    # Map metric types
                    if 'variance' in metric_type.lower() and 'cumulative' not in metric_type.lower():
                        pivot_records[key]['ExplainedVariance'] = float(value) if value is not None else None
                    elif 'cumulative' in metric_type.lower():
                        pivot_records[key]['CumulativeVariance'] = float(value) if value is not None else None
                    elif 'eigenvalue' in metric_type.lower() or metric_type.startswith('n_'):
                        pivot_records[key]['Eigenvalue'] = float(value) if value is not None else None
                
                # Build new-format rows
                new_rows = []
                for (run_id, equip_id, comp_idx), metrics in pivot_records.items():
                    new_rows.append({
                        'RunID': run_id,
                        'EquipID': equip_id,
                        'ComponentIndex': comp_idx,
                        **metrics
                    })
                
                if not new_rows:
                    Console.debug("No PCA metrics to write after legacy conversion", component="OUTPUT")
                    return 0
                    
                df = pd.DataFrame(new_rows)
            
            # Validate required columns for new format
            required_cols = ['RunID', 'EquipID', 'ComponentIndex']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                Console.warn(f"_upsert_pca_metrics missing required columns: {missing_cols}", component="OUTPUT", missing_columns=missing_cols, available_columns=list(df.columns))
                return 0
            
            # DELETE existing rows for this RunID+EquipID
            run_equip_pairs = df[['RunID', 'EquipID']].drop_duplicates()
            deleted_count = 0
            for _, row in run_equip_pairs.iterrows():
                try:
                    cursor.execute("""
                        DELETE FROM ACM_PCA_Metrics 
                        WHERE RunID = ? AND EquipID = ?
                        """,
                        (str(row['RunID']), int(row['EquipID']))
                    )
                    deleted_count += cursor.rowcount
                except Exception as del_err:
                    Console.debug(f"DELETE failed for PCA metrics: {del_err}", component="OUTPUT")
            
            # Prepare bulk insert
            insert_sql = """
            INSERT INTO ACM_PCA_Metrics (RunID, EquipID, ComponentIndex, ExplainedVariance, CumulativeVariance, Eigenvalue)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            rows_to_insert = []
            for _, row in df.iterrows():
                rows_to_insert.append((
                    str(row['RunID']),
                    int(row['EquipID']),
                    int(row.get('ComponentIndex', 0)),
                    float(row['ExplainedVariance']) if pd.notna(row.get('ExplainedVariance')) else None,
                    float(row['CumulativeVariance']) if pd.notna(row.get('CumulativeVariance')) else None,
                    float(row['Eigenvalue']) if pd.notna(row.get('Eigenvalue')) else None
                ))
            
            if rows_to_insert:
                cursor.executemany(insert_sql, rows_to_insert)
            
            conn.commit()
            return len(rows_to_insert)
            
        except Exception as e:
            Console.warn(f"_upsert_pca_metrics failed: {e}", component="OUTPUT", table="ACM_PCA_Metrics", rows=len(df), equip_id=self.equip_id, error_type=type(e).__name__, error=str(e)[:200])
            if self.sql_client and self.sql_client.conn:
                try:
                    self.sql_client.conn.rollback()
                except:
                    pass
            return 0
    
    def _upsert_health_forecast(self, df: pd.DataFrame) -> int:
        """
        FORECAST-WRITE-01: Write health forecast using bulk insert.
        v10 schema: ACM_HealthForecast has (RunID, EquipID, Timestamp, ForecastHealth, CI_Lower, CI_Upper, Method, CreatedAt)
        """
        if df.empty or self.sql_client is None:
            return 0
        
        try:
            # Bulk insert is standard for forecast tables in v10
            return self._bulk_insert_sql('ACM_HealthForecast', df)
        except Exception as e:
            Console.warn(f"_upsert_health_forecast failed: {e}", component="OUTPUT", table="ACM_HealthForecast", rows=len(df), equip_id=self.equip_id, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def _upsert_failure_forecast(self, df: pd.DataFrame) -> int:
        """
        FORECAST-WRITE-02: Write failure forecast using bulk insert.
        v10 schema: ACM_FailureForecast has (RunID, EquipID, Timestamp, FailureProb, ThresholdUsed, Method, CreatedAt)
        """
        if df.empty or self.sql_client is None:
            return 0
        
        try:
            return self._bulk_insert_sql('ACM_FailureForecast', df)
        except Exception as e:
            Console.warn(f"_upsert_failure_forecast failed: {e}", component="OUTPUT", table="ACM_FailureForecast", rows=len(df), equip_id=self.equip_id, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def _upsert_detector_forecast_ts(self, df: pd.DataFrame) -> int:
        """
        FORECAST-UPSERT-03: Upsert detector forecast time series using MERGE.
        
        Primary key is (RunID, EquipID, DetectorName, Timestamp), so update if exists.
        Schema: RunID, EquipID, DetectorName, Timestamp, ForecastValue, CiLower, CiUpper, ForecastStd, Method, CreatedAt
        """
        if df.empty or self.sql_client is None:
            return 0
        
        # Ensure all required columns exist with defaults before upsert
        df = df.copy()
        if 'Method' not in df.columns:
            df['Method'] = 'AR1'
        if 'ForecastStd' not in df.columns:
            df['ForecastStd'] = 0.0
        
        # Fill any NaN values in critical columns
        df['Method'] = df['Method'].fillna('AR1')
        # Avoid FutureWarning for silent downcasting by ensuring float dtype before fillna
        df['ForecastStd'] = pd.to_numeric(df['ForecastStd'], errors='coerce').astype(float).fillna(0.0)
        if 'DetectorName' in df.columns:
            df['DetectorName'] = df['DetectorName'].fillna('UNKNOWN')
        
        try:
            conn = self.sql_client.conn
            cursor = conn.cursor()
            row_count = 0
            
            for _, row in df.iterrows():
                run_id = row['RunID']
                equip_id = row['EquipID']
                detector_name = row['DetectorName']
                timestamp = row['Timestamp']
                forecast_value = row.get('ForecastValue', 0.0)
                ci_lower = row.get('CiLower', 0.0)
                ci_upper = row.get('CiUpper', 0.0)
                forecast_std = row.get('ForecastStd', 0.0)
                method = row.get('Method', 'AR1')
                
                # Ensure no NaN values are passed as None to SQL
                if pd.isna(forecast_value):
                    forecast_value = 0.0
                if pd.isna(ci_lower):
                    ci_lower = 0.0
                if pd.isna(ci_upper):
                    ci_upper = 0.0
                if pd.isna(forecast_std):
                    forecast_std = 0.0
                if pd.isna(method):
                    method = 'AR1'
                if pd.isna(detector_name):
                    detector_name = 'UNKNOWN'
                    
                created_at = row.get('CreatedAt', datetime.now())
                
                merge_sql = """
                MERGE INTO ACM_DetectorForecast_TS AS target
                USING (SELECT ? AS RunID, ? AS EquipID, ? AS DetectorName, ? AS Timestamp) AS source
                ON (target.RunID = source.RunID AND target.EquipID = source.EquipID AND target.DetectorName = source.DetectorName AND target.Timestamp = source.Timestamp)
                WHEN MATCHED THEN
                    UPDATE SET ForecastValue = ?, CiLower = ?, CiUpper = ?, ForecastStd = ?, Method = ?
                WHEN NOT MATCHED THEN
                    INSERT (RunID, EquipID, DetectorName, Timestamp, ForecastValue, CiLower, CiUpper, ForecastStd, Method, CreatedAt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """
                cursor.execute(merge_sql, (
                    # ON clause
                    run_id, equip_id, detector_name, timestamp,
                    # UPDATE
                    forecast_value, ci_lower, ci_upper, forecast_std, method,
                    # INSERT
                    run_id, equip_id, detector_name, timestamp, forecast_value, ci_lower, ci_upper, forecast_std, method, created_at
                ))
                row_count += cursor.rowcount
            
            conn.commit()
            return row_count
            
        except Exception as e:
            Console.warn(f"_upsert_detector_forecast_ts failed: {e}", component="OUTPUT", table="ACM_DetectorForecast_TS", rows=len(df), equip_id=self.equip_id, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def _upsert_sensor_forecast(self, df: pd.DataFrame) -> int:
        """
        FORECAST-WRITE-04: Write sensor forecast using bulk insert.
        v10 schema: ACM_SensorForecast has (RunID, EquipID, SensorName, Timestamp, ForecastValue, CiLower, CiUpper, ForecastStd, Method, RegimeLabel, CreatedAt)
        """
        if df.empty or self.sql_client is None:
            return 0
        
        try:
            return self._bulk_insert_sql('ACM_SensorForecast', df)
        except Exception as e:
            Console.warn(f"_upsert_sensor_forecast failed: {e}", component="OUTPUT", table="ACM_SensorForecast", rows=len(df), equip_id=self.equip_id, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def write_run_stats(self, stats_data: Dict[str, Any]) -> int:
        """Write run statistics to ACM_Run_Stats table."""
        if not self._check_sql_health():
            return 0
        try:
            row = dict(stats_data)
            # Normalize key variants
            if 'StartTime' not in row and 'WindowStartEntryDateTime' in row:
                row['StartTime'] = normalize_timestamp_scalar(row.get('WindowStartEntryDateTime'))
            if 'EndTime' not in row and 'WindowEndEntryDateTime' in row:
                row['EndTime'] = normalize_timestamp_scalar(row.get('WindowEndEntryDateTime'))
            row.setdefault('RunID', self.run_id)
            row.setdefault('EquipID', self.equip_id or 0)
            sql_df = pd.DataFrame([row])
            return self._bulk_insert_sql('ACM_Run_Stats', sql_df)
        except Exception as e:
            Console.error(f"write_run_stats failed: {e}", component="OUTPUT", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def write_scores(self, scores_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Write scores (SQL-only) to dbo.ACM_Scores_Wide.

        Normalizes index to tz-naive seconds and writes a Timestamp column.
        Timestamp normalization is deterministic and always applied.
        """
        scores_for_output = scores_df.copy()
        scores_for_output.index.name = "timestamp"

        if len(scores_for_output.index):
            ts = pd.to_datetime(scores_for_output.index, errors="coerce")
            # Deterministic normalization: always tz-naive, floored to seconds
            # This ensures consistent SQL Server insertion regardless of input timezone
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            ts = ts.floor("s")
            scores_for_output.index = ts

        score_columns = {
            "timestamp": "Timestamp",
            "ar1_z": "ar1_z",
            "pca_spe_z": "pca_spe_z",
            "pca_t2_z": "pca_t2_z",
            "iforest_z": "iforest_z",
            "gmm_z": "gmm_z",
            "cusum_z": "cusum_z",
            "drift_z": "drift_z",
            "hst_z": "hst_z",
            "fused": "fused",
            "regime_label": "regime_label",
            "transient_state": "transient_state",
        }

        # non_numeric_cols must refer to *pre-mapping* column names because prepare() runs first.
        return self.write_dataframe(
            df=scores_for_output.reset_index(),
            artifact_name="scores",
            sql_table="ACM_Scores_Wide",
            sql_columns=score_columns,
            non_numeric_cols={"timestamp", "regime_label", "transient_state"},
            add_created_at=False,
            allow_repair=True,
        )
    
    def write_episodes(self, episodes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Write episodes to SQL (SQL-only).
        
        Individual episodes go to ACM_EpisodeDiagnostics.
        Run-level summary goes to ACM_Episodes.
        """
        if episodes_df.empty:
            return {"sql_written": False, "rows": 0, "inserted": 0, "error": None}
        
        # Prepare episodes for output
        episodes_for_output = episodes_df.copy().reset_index(drop=True)
        
        # Track which repairs are applied for explicit schema drift reporting
        repairs_applied = []
        
        episode_columns = {
            'episode_id': 'EpisodeID',
            'start_ts': 'StartTime',
            'end_ts': 'EndTime',
            'peak_fused_z': 'PeakZ',
            'peak_timestamp': 'peak_timestamp', 
            'duration_hours': 'DurationHours',
            'dominant_sensor': 'TopSensor1',
            'severity': 'Severity',
            'avg_fused_z': 'AvgZ',
            'min_health_index': 'min_health_index'
        }
        
        # Add episode_id if missing (sequential)
        if 'episode_id' not in episodes_for_output.columns:
            episodes_for_output['episode_id'] = range(1, len(episodes_for_output) + 1)
            repairs_applied.append("episode_id_added")
        
        # Calculate duration_hours from duration_s if needed
        if 'duration_hours' not in episodes_for_output.columns and 'duration_s' in episodes_for_output.columns:
            episodes_for_output['duration_hours'] = episodes_for_output['duration_s'] / 3600.0
            repairs_applied.append("duration_hours_derived")
        
        # Extract peak_timestamp from start_ts if missing
        if 'peak_timestamp' not in episodes_for_output.columns and 'start_ts' in episodes_for_output.columns:
            episodes_for_output['peak_timestamp'] = episodes_for_output['start_ts']
            repairs_applied.append("peak_timestamp_fallback_used")
        
        # Map regime_label to MaxRegimeLabel
        if 'regime_label' in episodes_for_output.columns:
            episodes_for_output['MaxRegimeLabel'] = episodes_for_output['regime_label']
            repairs_applied.append("regime_label_mapped")
        elif 'regime' in episodes_for_output.columns:
            episodes_for_output['MaxRegimeLabel'] = episodes_for_output['regime']
            repairs_applied.append("regime_mapped_fallback")
        
        # Extract dominant sensor from culprits field
        # culprits format: "Detector (Sensor Name)" or "Detector"
        if 'culprits' in episodes_for_output.columns:
            def extract_dominant_sensor(culprit_str):
                if pd.isna(culprit_str) or culprit_str == '':
                    return 'UNKNOWN'
                # culprits field already contains formatted label from format_culprit_label()
                # e.g., "Multivariate Outlier (PCA-TÂ²)" or "Multivariate Outlier (PCA-TÂ²) â†’ SensorName"
                # Extract just the detector label (before " â†’ " if sensor attribution exists)
                if ' â†’ ' in str(culprit_str):
                    return str(culprit_str).split(' â†’ ')[0].strip()
                else:
                    return str(culprit_str).strip()
            episodes_for_output['dominant_sensor'] = episodes_for_output['culprits'].apply(extract_dominant_sensor)
            repairs_applied.append("dominant_sensor_extracted")
        else:
            episodes_for_output['dominant_sensor'] = 'UNKNOWN'
            repairs_applied.append("dominant_sensor_defaulted")
        
        # Calculate severity from peak_fused_z
        if 'peak_fused_z' in episodes_for_output.columns:
            def calculate_severity(peak_z):
                if pd.isna(peak_z):
                    return 'UNKNOWN'
                if peak_z >= 6:
                    return 'CRITICAL'
                elif peak_z >= 4:
                    return 'HIGH'
                elif peak_z >= 2:
                    return 'MEDIUM'
                return 'LOW'
            episodes_for_output['severity'] = episodes_for_output['peak_fused_z'].apply(calculate_severity)
            repairs_applied.append("severity_calculated")
        elif 'severity' not in episodes_for_output.columns:
            episodes_for_output['severity'] = 'UNKNOWN'
            repairs_applied.append("severity_defaulted")
        
        # Add status default
        if 'status' not in episodes_for_output.columns:
            episodes_for_output['status'] = 'CLOSED'
            repairs_applied.append("status_defaulted")
        
        # Log all repairs for explicit schema drift tracking
        if repairs_applied:
            Console.info(f"Applied {len(repairs_applied)} schema repairs to episodes: {', '.join(repairs_applied)}", 
                        component="EPISODES", equip_id=self.equip_id, episode_count=len(episodes_for_output))
        
        result = self.write_dataframe(
            episodes_for_output,
            "episodes",
            sql_table="ACM_EpisodeDiagnostics",
            sql_columns=episode_columns,
            non_numeric_cols={
                "RunID", "EquipID", "episode_id", "peak_timestamp",
                "dominant_sensor", "severity", "min_health_index"
            },
            required=True
        )
        
        # Also write run-level summary to ACM_Episodes
        if not episodes_df.empty:
            try:
                summary = pd.DataFrame([{
                    'RunID': self.run_id,
                    'EquipID': self.equip_id or 0,
                    'EpisodeCount': len(episodes_df),
                    'MedianDurationMinutes': episodes_df.get('duration_s', pd.Series([0])).median() / 60.0 if 'duration_s' in episodes_df.columns else 0.0,
                    'MaxFusedZ': episodes_df.get('peak_fused_z', pd.Series([0])).max() if 'peak_fused_z' in episodes_df.columns else 0.0,
                    'AvgFusedZ': episodes_df.get('avg_fused_z', pd.Series([0])).mean() if 'avg_fused_z' in episodes_df.columns else 0.0
                }])
                self._bulk_insert_sql('ACM_Episodes', summary)
            except Exception as summary_err:
                Console.warn(f"Failed to write summary to ACM_Episodes: {summary_err}", component="EPISODES", equip_id=self.equip_id, run_id=self.run_id, episode_count=len(episodes_df), error_type=type(summary_err).__name__)
        
        return result
    
    def write_threshold_metadata(
        self,
        equip_id: int,
        threshold_type: str,
        threshold_value: float,
        calculation_method: str,
        sample_count: int,
        train_start: Optional[datetime] = None,
        train_end: Optional[datetime] = None,
        config_signature: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """Write adaptive threshold metadata to ACM_AdaptiveConfig.
        
        Maps to actual table schema:
        - ConfigKey: threshold_type (e.g., 'fused_alert_z')
        - ConfigValue: threshold_value as string
        - MinBound/MaxBound: 0.0/infinity for thresholds
        - IsLearned: True (since computed from data)
        - DataVolumeAtTuning: sample_count
        - PerformanceMetric: calculation_method
        - Source: 'adaptive_threshold_calculator'
        - ResearchReference: notes
        
        Args:
            equip_id: Equipment ID
            threshold_type: Type of threshold (e.g., 'fused_alert_z', 'fused_warn_z')
            threshold_value: Calculated threshold value
            calculation_method: Method used (e.g., 'quantile_0.997')
            sample_count: Number of samples used in calculation
            train_start: Start of training window (unused - table doesn't have this)
            train_end: End of training window (unused - table doesn't have this)
            config_signature: Hash of config used (unused - table doesn't have this)
            notes: Optional notes about calculation
        
        Returns:
            Number of rows written (1 on success, 0 on failure)
        """
        if not self._check_sql_health():
            return 0
        try:
            # Handle dict values (per-regime thresholds) - store first value only
            if isinstance(threshold_value, dict):
                # Take first value for storage, or average
                threshold_float = float(list(threshold_value.values())[0]) if threshold_value else 0.0
            else:
                threshold_float = float(threshold_value)
            
            row = {
                'EquipID': int(equip_id),
                'ConfigKey': threshold_type,
                'ConfigValue': threshold_float,  # Float column
                'MinBound': 0.0,
                'MaxBound': 999999.0,  # Effectively no upper bound
                'IsLearned': 1,  # BIT column
                'DataVolumeAtTuning': int(sample_count),
                'PerformanceMetric': 0.0,  # Float column - store 0 for now
                'ResearchReference': f"{calculation_method}: {notes}" if notes else calculation_method,
                'Source': 'adaptive_threshold_calculator',
            }
            # Use MERGE to handle existing records (UNIQUE constraint on EquipID+ConfigKey)
            return self._upsert_adaptive_config(row)
        except Exception as e:
            Console.warn(f"write_threshold_metadata failed: {e}", component="THRESHOLD", error=str(e)[:200])
            return 0
    
    def _upsert_adaptive_config(self, row: dict) -> int:
        """Upsert single row into ACM_AdaptiveConfig using MERGE.
        
        Table has UNIQUE constraint on (EquipID, ConfigKey).
        """
        if self.sql_client is None:
            return 0
        try:
            conn = self.sql_client.conn
            cursor = conn.cursor()
            merge_sql = """
            MERGE INTO ACM_AdaptiveConfig AS target
            USING (SELECT ? AS EquipID, ? AS ConfigKey) AS source (EquipID, ConfigKey)
            ON target.EquipID = source.EquipID AND target.ConfigKey = source.ConfigKey
            WHEN MATCHED THEN
                UPDATE SET 
                    ConfigValue = ?,
                    MinBound = ?,
                    MaxBound = ?,
                    IsLearned = ?,
                    DataVolumeAtTuning = ?,
                    PerformanceMetric = ?,
                    ResearchReference = ?,
                    Source = ?,
                    UpdatedAt = GETDATE()
            WHEN NOT MATCHED THEN
                INSERT (EquipID, ConfigKey, ConfigValue, MinBound, MaxBound, IsLearned, 
                        DataVolumeAtTuning, PerformanceMetric, ResearchReference, Source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(merge_sql, (
                # Match keys
                row['EquipID'], row['ConfigKey'],
                # Update values
                row['ConfigValue'], row['MinBound'], row['MaxBound'], row['IsLearned'],
                row['DataVolumeAtTuning'], row['PerformanceMetric'], row['ResearchReference'], row['Source'],
                # Insert values
                row['EquipID'], row['ConfigKey'], row['ConfigValue'], row['MinBound'], row['MaxBound'],
                row['IsLearned'], row['DataVolumeAtTuning'], row['PerformanceMetric'], 
                row['ResearchReference'], row['Source']
            ))
            conn.commit()
            return 1
        except Exception as e:
            Console.warn(f"_upsert_adaptive_config failed: {e}", component="OUTPUT",
                        equip_id=row.get('EquipID'), config_key=row.get('ConfigKey'), error=str(e)[:200])
            if self.sql_client and self.sql_client.conn:
                try:
                    self.sql_client.conn.rollback()
                except:
                    pass
            return 0
    
    def load_omr_drift_context(self, equip_id: int, lookback_hours: int = 24) -> dict:
        """Load OMR and drift context from recent data for forecasting.
        
        Queries ACM_Scores_Wide for recent OMR/drift values and ACM_DriftSeries for trend.
        
        Args:
            equip_id: Equipment ID
            lookback_hours: How many hours of history to consider
            
        Returns:
            Dict with keys:
                - omr_z: Most recent OMR z-score (or None)
                - omr_trend: 'increasing', 'decreasing', 'stable', or 'unknown'
                - top_contributors: List of top contributing sensors
                - drift_z: Most recent drift z-score (or None)
                - drift_trend: 'increasing', 'decreasing', 'stable', or 'unknown'
        """
        result = {
            'omr_z': None,
            'omr_trend': 'unknown',
            'top_contributors': [],
            'drift_z': None,
            'drift_trend': 'unknown'
        }
        
        if not self._check_sql_health():
            return result
        
        try:
            conn = self.sql_client.conn
            cursor = conn.cursor()
            
            # Get most recent fused score (Overall Model Residual is computed from fused scores)
            # ACM_Scores_Wide has 'fused' column, not 'omr_z'
            cursor.execute("""
                SELECT TOP 1 fused
                FROM ACM_Scores_Wide
                WHERE EquipID = ? AND fused IS NOT NULL
                ORDER BY Timestamp DESC
            """, (equip_id,))
            row = cursor.fetchone()
            if row:
                result['omr_z'] = float(row[0]) if row[0] is not None else None
            
            # Get OMR/fused trend from recent values
            cursor.execute("""
                SELECT fused
                FROM (
                    SELECT TOP 10 fused, Timestamp
                    FROM ACM_Scores_Wide
                    WHERE EquipID = ? AND fused IS NOT NULL
                    ORDER BY Timestamp DESC
                ) sub
                ORDER BY Timestamp ASC
            """, (equip_id,))
            omr_values = [float(r[0]) for r in cursor.fetchall() if r[0] is not None]
            if len(omr_values) >= 3:
                # Simple trend: compare first half avg to second half avg
                mid = len(omr_values) // 2
                first_avg = sum(omr_values[:mid]) / mid
                second_avg = sum(omr_values[mid:]) / (len(omr_values) - mid)
                if second_avg > first_avg * 1.1:
                    result['omr_trend'] = 'increasing'
                elif second_avg < first_avg * 0.9:
                    result['omr_trend'] = 'decreasing'
                else:
                    result['omr_trend'] = 'stable'
            
            # Get top contributors from SensorHotspots (most recent by max z-score)
            cursor.execute("""
                SELECT TOP 3 SensorName
                FROM ACM_SensorHotspots
                WHERE EquipID = ?
                ORDER BY MaxAbsZ DESC
            """, (equip_id,))
            result['top_contributors'] = [r[0] for r in cursor.fetchall() if r[0]]
            
            # Get drift from DriftSeries
            cursor.execute("""
                SELECT TOP 1 DriftValue
                FROM ACM_DriftSeries
                WHERE EquipID = ?
                ORDER BY Timestamp DESC
            """, (equip_id,))
            row = cursor.fetchone()
            if row:
                result['drift_z'] = float(row[0]) if row[0] is not None else None
            
            # Get drift trend
            cursor.execute("""
                SELECT DriftValue
                FROM (
                    SELECT TOP 10 DriftValue, Timestamp
                    FROM ACM_DriftSeries
                    WHERE EquipID = ?
                    ORDER BY Timestamp DESC
                ) sub
                ORDER BY Timestamp ASC
            """, (equip_id,))
            drift_values = [float(r[0]) for r in cursor.fetchall() if r[0] is not None]
            if len(drift_values) >= 3:
                mid = len(drift_values) // 2
                first_avg = sum(drift_values[:mid]) / mid
                second_avg = sum(drift_values[mid:]) / (len(drift_values) - mid)
                if second_avg > first_avg * 1.1:
                    result['drift_trend'] = 'increasing'
                elif second_avg < first_avg * 0.9:
                    result['drift_trend'] = 'decreasing'
                else:
                    result['drift_trend'] = 'stable'
            
            cursor.close()
        except Exception as e:
            Console.debug(f"load_omr_drift_context failed: {e}", component="OUTPUT", error=str(e)[:200])
        
        return result
    
    def write_anomaly_events(self, df_events: pd.DataFrame, run_id: str) -> int:
        """Write anomaly events to ACM_Anomaly_Events table.
        
        V11: Adds Confidence column based on episode duration and maturity state.
        
        Args:
            df_events: DataFrame with event data (EquipID, start_ts, end_ts, severity, etc.)
            run_id: Current run ID
        
        Returns:
            Number of rows written
        """
        if not self._check_sql_health() or df_events is None or df_events.empty:
            return 0
        try:
            df = df_events.copy()
            df['RunID'] = run_id
            # Rename columns to match table schema
            col_map = {
                'start_ts': 'StartTime',
                'end_ts': 'EndTime',
                'severity': 'Severity',
                'Detector': 'DetectorType',
                'Score': 'PeakScore',
                'ContributorsJSON': 'ContributorsJSON'
            }
            for old, new in col_map.items():
                if old in df.columns and new not in df.columns:
                    df[new] = df[old]
            
            # V11: Add confidence for each episode
            if _CONFIDENCE_AVAILABLE and _LIFECYCLE_AVAILABLE and compute_episode_confidence is not None:
                try:
                    # Get current maturity state from model lifecycle
                    maturity_state = 'COLDSTART'  # Default
                    if hasattr(self, 'sql_client') and self.sql_client is not None:
                        try:
                            lifecycle = ModelLifecycle(self.sql_client, self.equip_id)
                            state_info = lifecycle.get_state()
                            maturity_state = state_info.get('maturity_state', 'COLDSTART')
                        except Exception:
                            pass  # Use default COLDSTART
                    
                    confidence_values = []
                    for _, row in df.iterrows():
                        # Calculate episode duration in minutes
                        duration_minutes = 60  # Default 1 hour if can't calculate
                        start_col = 'StartTime' if 'StartTime' in df.columns else 'start_ts'
                        end_col = 'EndTime' if 'EndTime' in df.columns else 'end_ts'
                        if start_col in df.columns and end_col in df.columns:
                            try:
                                start_t = pd.to_datetime(row[start_col])
                                end_t = pd.to_datetime(row[end_col])
                                duration_minutes = (end_t - start_t).total_seconds() / 60
                            except Exception:
                                pass
                        
                        # Get peak score if available
                        peak_z = row.get('PeakScore', row.get('Score', 3.0))
                        
                        conf = compute_episode_confidence(
                            maturity_state=maturity_state,
                            episode_duration_minutes=duration_minutes,
                            peak_z=peak_z if peak_z is not None else 3.0
                        )
                        confidence_values.append(round(conf, 3))
                    
                    df['Confidence'] = confidence_values
                except Exception as e:
                    Console.warn(f"Failed to compute episode confidence: {e}", component="EPISODES")
            
            return self.write_table('ACM_Anomaly_Events', df, delete_existing=True)
        except Exception as e:
            Console.warn(f"write_anomaly_events failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_regime_episodes(self, df_reg: pd.DataFrame, run_id: str) -> int:
        """Write regime episodes to ACM_RegimeEpisodes table.
        
        Args:
            df_reg: DataFrame with regime episode data
            run_id: Current run ID
            
        Returns:
            Number of rows written
        """
        if not self._check_sql_health() or df_reg is None or df_reg.empty:
            return 0
        try:
            df = df_reg.copy()
            df['RunID'] = run_id
            return self.write_table('ACM_Regime_Episodes', df, delete_existing=True)
        except Exception as e:
            Console.warn(f"write_regime_episodes failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_pca_model(self, model_row: Dict[str, Any]) -> int:
        """Write PCA model metadata to ACM_PCA_Model table.
        
        Args:
            model_row: Dict with model metadata
            
        Returns:
            Number of rows written
        """
        if not self._check_sql_health() or not model_row:
            return 0
        try:
            row = dict(model_row)
            row['RunID'] = self.run_id
            row['EquipID'] = self.equip_id or 0
            return self.write_table('ACM_PCA_Models', pd.DataFrame([row]), delete_existing=True)
        except Exception as e:
            Console.warn(f"write_pca_model failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_run_stats(self, stats: Dict[str, Any]) -> int:
        """Write run statistics to ACM_RunStats table.
        
        Args:
            stats: Dict with run statistics
            
        Returns:
            Number of rows written
        """
        if not self._check_sql_health() or not stats:
            return 0
        try:
            return self.write_table('ACM_Run_Stats', pd.DataFrame([stats]), delete_existing=False)
        except Exception as e:
            Console.warn(f"write_run_stats failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    # =========================================================================
    # NEW TABLE WRITE METHODS (Dec 25, 2025 - v11 completion)
    # =========================================================================
    
    def write_detector_correlation(self, detector_correlations: Dict[str, Dict[str, float]]) -> int:
        """Write detector correlation matrix to ACM_DetectorCorrelation.
        
        Args:
            detector_correlations: Nested dict {detector1: {detector2: correlation}}
        """
        if not self._check_sql_health() or not detector_correlations:
            return 0
        try:
            rows = []
            for d1, correlations in detector_correlations.items():
                for d2, corr in correlations.items():
                    rows.append({
                        'RunID': self.run_id,
                        'EquipID': self.equip_id or 0,
                        'Detector1': d1,
                        'Detector2': d2,
                        'Correlation': float(corr) if not pd.isna(corr) else 0.0
                    })
            if not rows:
                return 0
            return self.write_table('ACM_DetectorCorrelation', pd.DataFrame(rows), delete_existing=True)
        except Exception as e:
            Console.warn(f"write_detector_correlation failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_drift_series(self, drift_df: pd.DataFrame) -> int:
        """Write drift detection time series to ACM_DriftSeries.
        
        Args:
            drift_df: DataFrame with Timestamp, DriftValue, optionally DriftState
        """
        if not self._check_sql_health() or drift_df is None or drift_df.empty:
            return 0
        try:
            df = drift_df.copy()
            df['RunID'] = self.run_id
            df['EquipID'] = self.equip_id or 0
            # Map column names if needed
            if 'DriftZ' in df.columns and 'DriftValue' not in df.columns:
                df['DriftValue'] = df['DriftZ']
            return self.write_table('ACM_DriftSeries', df, delete_existing=True)
        except Exception as e:
            Console.warn(f"write_drift_series failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    # Alias for backward compatibility
    def write_drift_ts(self, drift_df: pd.DataFrame, run_id: str = None) -> int:
        """Alias for write_drift_series() for backward compatibility."""
        return self.write_drift_series(drift_df)
    
    def write_sensor_normalized_ts(self, scores_df: pd.DataFrame, sensor_cols: List[str] = None) -> int:
        """Write normalized sensor z-scores to ACM_SensorNormalized_TS.
        
        Transforms wide-format scores DataFrame (one column per sensor) to long format
        (one row per timestamp/sensor pair) for time-series analysis.
        
        Schema: ID, RunID, EquipID, Timestamp, SensorName, RawValue, NormalizedValue, CreatedAt
        
        Args:
            scores_df: DataFrame with Timestamp index/column and sensor z-score columns
            sensor_cols: List of sensor column names to extract (if None, uses all float columns)
            
        Returns:
            Number of rows written
        """
        if not self._check_sql_health() or scores_df is None or scores_df.empty:
            return 0
        
        try:
            df = scores_df.copy()
            
            # Ensure Timestamp is a column, not index
            # Check for 'Timestamp' column or index first
            if 'Timestamp' not in df.columns:
                # Check if index is datetime-like (common pattern: index is EntryDateTime or Timestamp)
                if isinstance(df.index, pd.DatetimeIndex) or df.index.name in ('Timestamp', 'EntryDateTime'):
                    df = df.reset_index()
                    # Rename EntryDateTime to Timestamp for consistency
                    if 'EntryDateTime' in df.columns:
                        df['Timestamp'] = df['EntryDateTime']
                    elif df.columns[0] != 'Timestamp' and pd.api.types.is_datetime64_any_dtype(df[df.columns[0]]):
                        df['Timestamp'] = df.iloc[:, 0]
                elif 'EntryDateTime' in df.columns:
                    df['Timestamp'] = df['EntryDateTime']
                else:
                    # Try to find a datetime column
                    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                    if dt_cols:
                        df['Timestamp'] = df[dt_cols[0]]
                    else:
                        Console.warn("write_sensor_normalized_ts: No Timestamp column found", component="OUTPUT")
                        return 0
            
            # Determine sensor columns
            if sensor_cols is None:
                # Use all numeric columns except known non-sensor columns
                exclude = {'Timestamp', 'RunID', 'EquipID', 'regime_label', 'fused', 'health', 
                          'ar1_z', 'pca_spe_z', 'pca_t2_z', 'iforest_z', 'gmm_z', 'omr_z',
                          'mhal_z', 'cusum_z', 'drift_z', 'hst_z', 'river_hst_z'}
                sensor_cols = [c for c in df.columns if c not in exclude 
                              and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
                              and not c.endswith('_z')]
            
            if not sensor_cols:
                Console.debug("write_sensor_normalized_ts: No sensor columns found", component="OUTPUT")
                return 0
            
            # Melt to long format
            long_rows = []
            timestamps = df['Timestamp'].tolist()
            
            for col in sensor_cols:
                if col not in df.columns:
                    continue
                values = df[col].tolist()
                for i, (ts, val) in enumerate(zip(timestamps, values)):
                    if pd.notna(val):
                        long_rows.append({
                            'RunID': self.run_id or '',
                            'EquipID': self.equip_id or 0,
                            'Timestamp': ts,
                            'SensorName': str(col),
                            'RawValue': None,  # Original raw value (not available in z-score frame)
                            'NormalizedValue': float(val)
                        })
            
            if not long_rows:
                return 0
            
            # Batch insert - this can be large, so use chunked writes
            long_df = pd.DataFrame(long_rows)
            
            # Delete existing data for this run to avoid duplicates
            return self.write_table('ACM_SensorNormalized_TS', long_df, delete_existing=True)
            
        except Exception as e:
            Console.warn(f"write_sensor_normalized_ts failed: {e}", component="OUTPUT", 
                        error=str(e)[:200], sensor_count=len(sensor_cols) if sensor_cols else 0)
            return 0
    
    def write_sensor_correlations(self, corr_matrix: pd.DataFrame, corr_type: str = 'pearson') -> int:
        """Write sensor correlation matrix to ACM_SensorCorrelations.
        
        Args:
            corr_matrix: Pandas correlation matrix (sensors x sensors)
            corr_type: 'pearson' or 'spearman'
        """
        if not self._check_sql_health() or corr_matrix is None or corr_matrix.empty:
            return 0
        try:
            rows = []
            sensors = list(corr_matrix.columns)
            for i, s1 in enumerate(sensors):
                for j, s2 in enumerate(sensors):
                    if i <= j:  # Upper triangle only to avoid duplicates
                        corr = corr_matrix.loc[s1, s2]
                        if not pd.isna(corr):
                            rows.append({
                                'RunID': self.run_id,
                                'EquipID': self.equip_id or 0,
                                'Sensor1': str(s1),
                                'Sensor2': str(s2),
                                'Correlation': float(corr),
                                'CorrelationType': corr_type
                            })
            if not rows:
                return 0
            return self.write_table('ACM_SensorCorrelations', pd.DataFrame(rows), delete_existing=True)
        except Exception as e:
            Console.warn(f"write_sensor_correlations failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_feature_drop_log(self, dropped_features: List[Dict[str, Any]]) -> int:
        """Write dropped features log to ACM_FeatureDropLog.
        
        Args:
            dropped_features: List of dicts with keys: FeatureName, DropReason, DropValue, Threshold
        """
        if not self._check_sql_health() or not dropped_features:
            return 0
        try:
            df = pd.DataFrame(dropped_features)
            df['RunID'] = self.run_id
            df['EquipID'] = self.equip_id or 0
            return self.write_table('ACM_FeatureDropLog', df, delete_existing=True)
        except Exception as e:
            Console.warn(f"write_feature_drop_log failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_calibration_summary(self, calibration_data: List[Dict[str, Any]]) -> int:
        """Write detector calibration summary to ACM_CalibrationSummary.
        
        Args:
            calibration_data: List of dicts with DetectorType, CalibrationScore, etc.
        """
        if not self._check_sql_health() or not calibration_data:
            return 0
        try:
            df = pd.DataFrame(calibration_data)
            df['RunID'] = self.run_id
            df['EquipID'] = self.equip_id or 0
            return self.write_table('ACM_CalibrationSummary', df, delete_existing=True)
        except Exception as e:
            Console.warn(f"write_calibration_summary failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_regime_occupancy(self, occupancy_data: List[Dict[str, Any]]) -> int:
        """Write regime occupancy stats to ACM_RegimeOccupancy.
        
        Args:
            occupancy_data: List of dicts with RegimeLabel, DwellTimeHours, DwellFraction, etc.
        """
        if not self._check_sql_health() or not occupancy_data:
            return 0
        try:
            df = pd.DataFrame(occupancy_data)
            df['RunID'] = self.run_id
            df['EquipID'] = self.equip_id or 0
            return self.write_table('ACM_RegimeOccupancy', df, delete_existing=True)
        except Exception as e:
            Console.warn(f"write_regime_occupancy failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_regime_transitions(self, transition_matrix: Dict[str, Dict[str, int]]) -> int:
        """Write regime transition matrix to ACM_RegimeTransitions.
        
        Args:
            transition_matrix: Nested dict {from_regime: {to_regime: count}}
        """
        if not self._check_sql_health() or not transition_matrix:
            return 0
        try:
            rows = []
            for from_r, transitions in transition_matrix.items():
                total = sum(transitions.values())
                for to_r, count in transitions.items():
                    rows.append({
                        'RunID': self.run_id,
                        'EquipID': self.equip_id or 0,
                        'FromRegime': str(from_r),
                        'ToRegime': str(to_r),
                        'TransitionCount': int(count),
                        'TransitionProbability': float(count) / total if total > 0 else 0.0
                    })
            if not rows:
                return 0
            return self.write_table('ACM_RegimeTransitions', pd.DataFrame(rows), delete_existing=True)
        except Exception as e:
            Console.warn(f"write_regime_transitions failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_contribution_timeline(self, contributions_df: pd.DataFrame) -> int:
        """Write detector contribution timeline to ACM_ContributionTimeline.
        
        Args:
            contributions_df: DataFrame with Timestamp, DetectorType, ContributionPct
        """
        if not self._check_sql_health() or contributions_df is None or contributions_df.empty:
            return 0
        try:
            df = contributions_df.copy()
            df['RunID'] = self.run_id
            df['EquipID'] = self.equip_id or 0
            return self.write_table('ACM_ContributionTimeline', df, delete_existing=True)
        except Exception as e:
            Console.warn(f"write_contribution_timeline failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_regime_promotion_log(self, promotions: List[Dict[str, Any]]) -> int:
        """Write regime maturity promotions to ACM_RegimePromotionLog.
        
        Args:
            promotions: List of dicts with RegimeLabel, FromState, ToState, Reason, etc.
        """
        if not self._check_sql_health() or not promotions:
            return 0
        try:
            df = pd.DataFrame(promotions)
            df['RunID'] = self.run_id
            df['EquipID'] = self.equip_id or 0
            return self.write_table('ACM_RegimePromotionLog', df, delete_existing=False)  # Append, don't delete
        except Exception as e:
            Console.warn(f"write_regime_promotion_log failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_drift_controller(self, controller_state: Dict[str, Any]) -> int:
        """Write drift controller state to ACM_DriftController.
        
        Args:
            controller_state: Dict with ControllerState, Threshold, Sensitivity, etc.
        """
        if not self._check_sql_health() or not controller_state:
            return 0
        try:
            row = dict(controller_state)
            row['RunID'] = self.run_id
            row['EquipID'] = self.equip_id or 0
            return self.write_table('ACM_DriftController', pd.DataFrame([row]), delete_existing=True)
        except Exception as e:
            Console.warn(f"write_drift_controller failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_regime_definitions(self, regime_defs: List[Dict[str, Any]], version: int) -> int:
        """Write regime definitions to ACM_RegimeDefinitions (v11).
        
        Args:
            regime_defs: List of dicts with RegimeID, RegimeName, CentroidJSON, etc.
            version: Regime model version number
        """
        if not self._check_sql_health() or not regime_defs:
            return 0
        try:
            df = pd.DataFrame(regime_defs)
            df['EquipID'] = self.equip_id or 0
            df['RegimeVersion'] = version
            df['CreatedByRunID'] = self.run_id
            return self.write_table('ACM_RegimeDefinitions', df, delete_existing=False)  # Keep history
        except Exception as e:
            Console.warn(f"write_regime_definitions failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_active_models(self, model_state: Dict[str, Any]) -> int:
        """Write/update active model versions to ACM_ActiveModels (v11).
        
        Args:
            model_state: Dict with ActiveRegimeVersion, RegimeMaturityState, etc.
        
        Note: ACM_ActiveModels has EquipID only (no RunID), so we delete by EquipID before insert.
        """
        if not self._check_sql_health() or not model_state:
            return 0
        try:
            row = dict(model_state)
            row['EquipID'] = self.equip_id or 0
            row['LastUpdatedAt'] = datetime.now()
            row['LastUpdatedBy'] = self.run_id
            
            # Manual delete by EquipID (table has no RunID column)
            try:
                with self.sql_client.cursor() as cur:
                    cur.execute("DELETE FROM dbo.[ACM_ActiveModels] WHERE EquipID = ?", (int(self.equip_id or 0),))
                    if hasattr(self.sql_client, "commit"):
                        self.sql_client.commit()
            except Exception as del_ex:
                Console.debug(f"ACM_ActiveModels delete skipped: {del_ex}", component="OUTPUT")
            
            # Insert new row (delete_existing=False since we manually deleted)
            return self.write_table('ACM_ActiveModels', pd.DataFrame([row]), delete_existing=False)
        except Exception as e:
            Console.warn(f"write_active_models failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_data_contract_validation(self, validation_result: Dict[str, Any]) -> int:
        """Write data contract validation result to ACM_DataContractValidation (v11).
        
        Args:
            validation_result: Dict with Passed, RowsValidated, ColumnsValidated, IssuesJSON, etc.
        """
        if not self._check_sql_health() or not validation_result:
            return 0
        try:
            row = dict(validation_result)
            row['RunID'] = self.run_id
            row['EquipID'] = self.equip_id or 0
            row['ValidatedAt'] = datetime.now()
            return self.write_table('ACM_DataContractValidation', pd.DataFrame([row]), delete_existing=True)
        except Exception as e:
            Console.warn(f"write_data_contract_validation failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_seasonal_patterns(self, patterns: List[Dict[str, Any]]) -> int:
        """Write detected seasonal patterns to ACM_SeasonalPatterns (v11).
        
        Args:
            patterns: List of dicts with SensorName, PatternType, PeriodHours, Amplitude, etc.
        """
        if not self._check_sql_health() or not patterns:
            return 0
        try:
            df = pd.DataFrame(patterns)
            df['EquipID'] = self.equip_id or 0
            df['DetectedAt'] = datetime.now()
            df['DetectedByRunID'] = self.run_id
            return self.write_table('ACM_SeasonalPatterns', df, delete_existing=True)
        except Exception as e:
            Console.warn(f"write_seasonal_patterns failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    def write_asset_profile(self, profile: Dict[str, Any]) -> int:
        """Write asset profile to ACM_AssetProfiles (v11).
        
        Args:
            profile: Dict with EquipType, SensorNamesJSON, SensorMeansJSON, SensorStdsJSON, etc.
        
        Note: ACM_AssetProfiles has EquipID only (no RunID), so we delete by EquipID before insert.
        """
        if not self._check_sql_health() or not profile:
            return 0
        try:
            row = dict(profile)
            row['EquipID'] = self.equip_id or 0
            row['LastUpdatedAt'] = datetime.now()
            row['LastUpdatedByRunID'] = self.run_id
            
            # Manual delete by EquipID (table has no RunID column for standard delete)
            try:
                with self.sql_client.cursor() as cur:
                    cur.execute("DELETE FROM dbo.[ACM_AssetProfiles] WHERE EquipID = ?", (int(self.equip_id or 0),))
                    if hasattr(self.sql_client, "commit"):
                        self.sql_client.commit()
            except Exception as del_ex:
                Console.debug(f"ACM_AssetProfiles delete skipped: {del_ex}", component="OUTPUT")
            
            # Insert new row (delete_existing=False since we manually deleted)
            return self.write_table('ACM_AssetProfiles', pd.DataFrame([row]), delete_existing=False)
        except Exception as e:
            Console.warn(f"write_asset_profile failed: {e}", component="OUTPUT", error=str(e)[:200])
            return 0
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.stats,
            'avg_write_time': self.stats['write_time'] / max(1, self.stats['sql_writes']),
            'sql_success_rate': 1.0 - (self.stats['sql_failures'] / max(1, self.stats['sql_health_checks']))
        }
    
    def flush(self) -> None:
        """OUT-18: Flush current batch without finalizing (for auto-flush triggers)."""
        with self._batch_lock:
            # Reset batch for next accumulation
            self._current_batch = OutputBatch()
    
    def flush_and_finalize(self) -> Dict[str, Any]:
        """Flush any pending operations and return final statistics."""
        self.flush()  # OUT-18: Use flush() for DRY
        
        stats = self.get_stats()
        Console.info(f"[OUTPUT] Finalized: {stats['sql_writes']} SQL ops, "
                f"{stats['total_rows']} total rows, "
                f"{stats['avg_write_time']:.3f}s avg write time")
        
        return stats

    def close(self) -> None:
        """Gracefully finalize outstanding work. Compatible with acm_main finally block."""
        try:
            self.flush_and_finalize()
        except Exception:
            pass

    # ==================== BULK DELETE OPTIMIZATION ====================

    def _bulk_delete_analytics_tables(self, tables: List[str]) -> int:
        """
        PERF-OPT v11: Delete existing rows for current RunID/EquipID from multiple tables in ONE SQL batch.

        - Filters to ALLOWED_TABLES only.
        - Skips non-existent tables (cached).
        - Chooses predicate based on presence of RunID / EquipID columns.
        - Executes a single SQL batch to avoid N round-trips.

        Returns:
            Number of tables for which a DELETE statement was included in the batch.
            (Rowcount not tracked in this batched approach.)
        """
        if not self.sql_client or not self.run_id:
            return 0

        start_time = time.perf_counter()
        tables_targeted = 0

        # Local cursor factory for helper functions that expect a callable
        cursor_factory = lambda: cast(Any, self.sql_client).cursor()

        try:
            # Normalize candidate tables: allowed + unique, preserve input order
            seen = set()
            candidate_tables: List[str] = []
            for t in tables:
                if not t or t in seen:
                    continue
                seen.add(t)
                if t in ALLOWED_TABLES:
                    candidate_tables.append(t)

            if not candidate_tables:
                return 0

            delete_statements: List[str] = []

            for table_name in candidate_tables:
                # Table existence (cached)
                exists = self._table_exists_cache.get(table_name)
                if exists is None:
                    try:
                        exists = bool(_table_exists(cursor_factory, table_name))
                    except Exception:
                        exists = False
                    self._table_exists_cache[table_name] = exists

                if not exists:
                    continue

                # Column presence (prefer cached insertable/columns; avoid repeated metadata calls)
                table_cols = (
                    self._table_insertable_cache.get(table_name)
                    or self._table_columns_cache.get(table_name)
                )

                if table_cols is None:
                    try:
                        cols = set(_get_insertable_columns(cursor_factory, table_name) or [])
                        if not cols:
                            cols = set(_get_table_columns(cursor_factory, table_name) or [])
                        table_cols = cols
                        # Cache in insertable cache since we use it as "known columns" anyway
                        self._table_insertable_cache[table_name] = table_cols
                    except Exception:
                        # If we cannot introspect columns, skip (safe default)
                        continue

                # Build the WHERE predicate
                has_runid = "RunID" in table_cols
                has_equipid = "EquipID" in table_cols and (self.equip_id is not None)

                if not has_runid:
                    continue  # no RunID => cannot safely scope delete to current run

                if has_equipid:
                    delete_statements.append(
                        f"DELETE FROM dbo.[{table_name}] WHERE RunID = @RunID AND EquipID = @EquipID"
                    )
                else:
                    delete_statements.append(
                        f"DELETE FROM dbo.[{table_name}] WHERE RunID = @RunID"
                    )

                # Mark as pre-deleted so _bulk_insert_sql can skip its own per-table pre-delete logic
                self._bulk_predeleted_tables.add(table_name)
                tables_targeted += 1

            if not delete_statements:
                return 0

            # One network round-trip
            batch_sql = ";\n".join(delete_statements)

            cur = cursor_factory()
            try:
                # Parameter binding via pyodbc placeholders within DECLARE assignment is OK.
                # We declare parameters once and reuse them in all statements.
                sql = f"""
DECLARE @RunID NVARCHAR(36) = ?;
DECLARE @EquipID INT = ?;
{batch_sql};
"""
                cur.execute(sql, (self.run_id, int(self.equip_id or 0)))

                # Commit behavior: follow your existing pattern (depends on autocommit / wrapper)
                if hasattr(self.sql_client, "commit"):
                    self.sql_client.commit()
                elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                    if not getattr(self.sql_client.conn, "autocommit", True):
                        self.sql_client.conn.commit()

            finally:
                try:
                    cur.close()
                except Exception:
                    pass

            elapsed = time.perf_counter() - start_time
            Console.info(
                f"Bulk pre-delete: {tables_targeted} tables targeted, {len(delete_statements)} DELETE statements in {elapsed:.2f}s (batched)",
                component="OUTPUT",
                tables_targeted=tables_targeted,
                delete_count=len(delete_statements)
            )
            return tables_targeted

        except Exception as e:
            Console.warn(
                f"Bulk pre-delete failed (non-fatal): {e}",
                component="OUTPUT",
                tables=len(tables),
                equip_id=self.equip_id,
                run_id=self.run_id,
                error_type=type(e).__name__,
            )
            return 0

    # ==================== COMPREHENSIVE ANALYTICS TABLES ====================

    def generate_all_analytics_tables(
        self,
        scores_df: pd.DataFrame,
        cfg: Dict[str, Any],
        sensor_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Generate essential analytics tables (v11 - SQL-only).

        Writes only the tables in ALLOWED_TABLES:
          - ACM_HealthTimeline: Health % over time (required for RUL forecasting)
          - ACM_RegimeTimeline: Operating regime assignments
          - ACM_SensorDefects: Sensor-level anomaly flags
          - ACM_SensorHotspots: Top anomalous sensors (RUL attribution)
          - ACM_DataQuality: Data quality per sensor (generated in-memory, SQL write)
        """
        Console.info("Generating analytics tables (v11 SQL-only)...", component="ANALYTICS")

        sql_count = 0

        # SQL is mandatory: if not ready, skip (non-fatal)
        if (self.sql_client is None) or (not self.run_id):
            Console.warn(
                "Analytics table generation skipped (SQL not ready). Missing sql_client and/or run_id.",
                component="ANALYTICS",
                equip_id=getattr(self, "equip_id", None),
                run_id=getattr(self, "run_id", None),
            )
            return {"sql_tables": 0}

        has_fused = "fused" in scores_df.columns
        has_regimes = "regime_label" in scores_df.columns

        # Extract sensor context
        sensor_values = None
        sensor_zscores = None
        sensor_train_mean = None
        sensor_train_std = None
        data_quality_df = None

        if sensor_context:
            v = sensor_context.get("values")
            z = sensor_context.get("z_scores")

            if isinstance(v, pd.DataFrame) and len(v.columns):
                sensor_values = v.reindex(scores_df.index)

            if isinstance(z, pd.DataFrame) and len(z.columns):
                sensor_zscores = z.reindex(scores_df.index)

            m = sensor_context.get("train_mean")
            s = sensor_context.get("train_std")

            if isinstance(m, pd.Series):
                sensor_train_mean = m
            if isinstance(s, pd.Series):
                sensor_train_std = s

            dq = sensor_context.get("data_quality_df")
            if isinstance(dq, pd.DataFrame) and len(dq.columns):
                data_quality_df = dq.copy()

        analytics_tables = [
            "ACM_HealthTimeline",
            "ACM_RegimeTimeline",
            "ACM_SensorDefects",
            "ACM_SensorHotspots",
            "ACM_DataQuality",
        ]

        with self.batched_transaction():
            try:
                # Delete only the tables we write (for this run/equipment scope)
                self._bulk_delete_analytics_tables(analytics_tables)

                # 1) ACM_HealthTimeline
                if has_fused:
                    health_df = self._generate_health_timeline(scores_df, cfg)
                    result = self.write_dataframe(
                        health_df,
                        "health_timeline",
                        sql_table="ACM_HealthTimeline",
                        add_created_at=True,
                    )
                    if result.get("sql_written"):
                        sql_count += 1

                # 2) ACM_RegimeTimeline
                if has_regimes:
                    regime_df = self._generate_regime_timeline(scores_df)
                    result = self.write_dataframe(
                        regime_df,
                        "regime_timeline",
                        sql_table="ACM_RegimeTimeline",
                        add_created_at=True,
                    )
                    if result.get("sql_written"):
                        sql_count += 1

                # 3) ACM_SensorDefects
                if has_fused:
                    sensor_defects_df = self._generate_sensor_defects(scores_df)
                    result = self.write_dataframe(
                        sensor_defects_df,
                        "sensor_defects",
                        sql_table="ACM_SensorDefects",
                        add_created_at=True,
                    )
                    if result.get("sql_written"):
                        sql_count += 1

                # 4) ACM_SensorHotspots
                sensor_ready = (sensor_zscores is not None) and (sensor_values is not None)
                if sensor_ready:
                    warn_threshold = float(
                        ((cfg.get("regimes", {}) or {}).get("health", {}) or {}).get("fused_warn_z", 1.5) or 1.5
                    )
                    alert_threshold = float(
                        ((cfg.get("regimes", {}) or {}).get("health", {}) or {}).get("fused_alert_z", 3.0) or 3.0
                    )
                    top_n = int((cfg.get("output", {}) or {}).get("sensor_hotspot_top_n", 25))

                    sensor_hotspots_df = self._generate_sensor_hotspots_table(
                        sensor_zscores=sensor_zscores,
                        sensor_values=sensor_values,
                        train_mean=sensor_train_mean,
                        train_std=sensor_train_std,
                        warn_z=warn_threshold,
                        alert_z=alert_threshold,
                        top_n=top_n,
                    )

                    result = self.write_dataframe(
                        sensor_hotspots_df,
                        "sensor_hotspots",
                        sql_table="ACM_SensorHotspots",
                        non_numeric_cols={"SensorName"},
                        add_created_at=True,
                    )
                    if result.get("sql_written"):
                        sql_count += 1

                # 5) ACM_DataQuality (SQL-only)
                if isinstance(data_quality_df, pd.DataFrame) and len(data_quality_df.columns):
                    dq_df = data_quality_df.copy()

                    if "CheckName" not in dq_df.columns:
                        dq_df["CheckName"] = "NullsBySensor"

                    if "CheckResult" not in dq_df.columns:

                        def _derive_result(row):
                            try:
                                # Support both legacy lowercase and new PascalCase
                                notes = str(row.get("Notes", row.get("notes", "")) or "").lower()
                                tr_pct = float(row.get("TrainNullPct", row.get("train_null_pct", 0)) or 0)
                                sc_pct = float(row.get("ScoreNullPct", row.get("score_null_pct", 0)) or 0)
                                if "all_nulls_train" in notes or "all_nulls_score" in notes:
                                    return "FAIL"
                                if max(tr_pct, sc_pct) >= 80:
                                    return "FAIL"
                                if max(tr_pct, sc_pct) >= 10:
                                    return "CAUTION"
                                if "low_variance_train" in notes:
                                    return "CAUTION"
                                return "OK"
                            except Exception:
                                return "OK"

                        dq_df["CheckResult"] = dq_df.apply(_derive_result, axis=1)

                    if "RunID" not in dq_df.columns:
                        dq_df["RunID"] = self.run_id
                    if "EquipID" not in dq_df.columns:
                        dq_df["EquipID"] = self.equip_id

                    # Standardize column naming to PascalCase for SQL
                    col_mapping = {
                        "sensor": "Sensor",
                        "train_count": "TrainCount",
                        "train_nulls": "TrainNulls",
                        "train_null_pct": "TrainNullPct",
                        "train_std": "TrainStd",
                        "train_longest_gap": "TrainLongestGap",
                        "train_flatline_span": "TrainFlatlineSpan",
                        "train_min_ts": "TrainMinTs",
                        "train_max_ts": "TrainMaxTs",
                        "score_count": "ScoreCount",
                        "score_nulls": "ScoreNulls",
                        "score_null_pct": "ScoreNullPct",
                        "score_std": "ScoreStd",
                        "score_longest_gap": "ScoreLongestGap",
                        "score_flatline_span": "ScoreFlatlineSpan",
                        "score_min_ts": "ScoreMinTs",
                        "score_max_ts": "ScoreMaxTs",
                        "interp_method": "InterpMethod",
                        "sampling_secs": "SamplingSecs",
                        "notes": "Notes",
                    }
                    dq_df = dq_df.rename(columns={k: v for k, v in col_mapping.items() if k in dq_df.columns})

                    expected_cols = [
                        "Sensor", "TrainCount", "TrainNulls", "TrainNullPct", "TrainStd",
                        "TrainLongestGap", "TrainFlatlineSpan", "TrainMinTs", "TrainMaxTs",
                        "ScoreCount", "ScoreNulls", "ScoreNullPct", "ScoreStd",
                        "ScoreLongestGap", "ScoreFlatlineSpan", "ScoreMinTs", "ScoreMaxTs",
                        "InterpMethod", "SamplingSecs", "Notes",
                        "RunID", "EquipID", "CheckName", "CheckResult",
                    ]
                    cols_to_keep = [c for c in expected_cols if c in dq_df.columns]
                    dq_df = dq_df[cols_to_keep]

                    result = self.write_dataframe(
                        dq_df,
                        "data_quality",
                        sql_table="ACM_DataQuality",
                        add_created_at=True,
                    )
                    if result.get("sql_written"):
                        sql_count += 1

                Console.info(
                    f"Generated analytics tables (SQL written: {sql_count})",
                    component="ANALYTICS",
                    equip_id=self.equip_id,
                    run_id=self.run_id,
                )
                return {"sql_tables": sql_count}

            except Exception as e:
                Console.warn(
                    f"Analytics table generation failed: {e}",
                    component="ANALYTICS",
                    equip_id=self.equip_id,
                    run_id=self.run_id,
                    error_type=type(e).__name__,
                    error=str(e)[:200],
                )
                import traceback
                Console.warn(
                    f"Traceback: {traceback.format_exc()}",
                    component="ANALYTICS",
                    equip_id=self.equip_id,
                    run_id=self.run_id,
                )
                return {"sql_tables": sql_count}

    def _generate_health_timeline(self, scores_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate enhanced health timeline with smoothing, quality flags, and V11 confidence.
        
        Implements EMA smoothing + rate limiting to prevent unrealistic health jumps
        caused by sensor noise, missing data, or coldstart artifacts.
        
        V11: Adds Confidence column based on maturity state and data quality.
        
        Args:
            scores_df: DataFrame with fused Z-scores and timestamp index
            cfg: Configuration dictionary with health smoothing parameters
        """
        # Calculate raw health index (unsmoothed)
        raw_health = _health_index(scores_df['fused'])
        
        # Load config parameters (with defaults)
        health_cfg = cfg.get('health', {})
        smoothing_alpha = health_cfg.get('smoothing_alpha', 0.3)
        max_change_per_period = health_cfg.get('max_change_rate_per_hour', 20.0)
        extreme_volatility_threshold = health_cfg.get('extreme_volatility_threshold', 30.0)
        extreme_anomaly_z_threshold = health_cfg.get('extreme_anomaly_z_threshold', 10.0)
        
        # Apply EMA smoothing (pandas ewm handles initialization automatically)
        smoothed_health = raw_health.ewm(alpha=smoothing_alpha, adjust=False).mean()
        
        # Calculate rate of change for quality flagging
        health_change = smoothed_health.diff().abs()
        
        # Initialize quality flags as NORMAL
        quality_flag = pd.Series(['NORMAL'] * len(scores_df), index=scores_df.index)
        
        # Flag extreme volatility (large health jumps)
        volatile_mask = health_change > extreme_volatility_threshold
        quality_flag[volatile_mask] = 'EXTREME_VOLATILITY'
        
        # Flag extreme anomalies (sensor faults, broken thermocouples, etc.)
        extreme_mask = scores_df['fused'].abs() > extreme_anomaly_z_threshold
        quality_flag[extreme_mask] = 'EXTREME_ANOMALY'
        
        # First point has no previous value, always NORMAL
        quality_flag.iloc[0] = 'NORMAL'
        
        # Log quality issues for operator awareness
        volatile_count = (quality_flag == 'EXTREME_VOLATILITY').sum()
        extreme_count = (quality_flag == 'EXTREME_ANOMALY').sum()
        if volatile_count > 0:
            Console.warn(f"{volatile_count} volatile health transitions detected (>{extreme_volatility_threshold}% change)", component="HEALTH", equip_id=self.equip_id, run_id=self.run_id, volatile_count=volatile_count, threshold=extreme_volatility_threshold)
        if extreme_count > 0:
            Console.warn(f"{extreme_count} extreme anomaly scores detected (|Z| > {extreme_anomaly_z_threshold})", component="HEALTH", equip_id=self.equip_id, run_id=self.run_id, extreme_count=extreme_count, threshold=extreme_anomaly_z_threshold)
        
        # Calculate health zones based on SMOOTHED health
        zones = pd.cut(
            smoothed_health,
            bins=[0, AnalyticsConstants.HEALTH_ALERT_THRESHOLD, AnalyticsConstants.HEALTH_WATCH_THRESHOLD, 100],
            labels=['ALERT', 'WATCH', 'GOOD']
        )
        
        # V11: Compute confidence for each health reading
        confidence_values = []
        if _CONFIDENCE_AVAILABLE and _LIFECYCLE_AVAILABLE and compute_health_confidence is not None:
            try:
                # Get current maturity state from model lifecycle
                maturity_state = 'COLDSTART'  # Default
                if hasattr(self, 'sql_client') and self.sql_client is not None:
                    try:
                        lifecycle = ModelLifecycle(self.sql_client, self.equip_id)
                        state_info = lifecycle.get_state()
                        maturity_state = state_info.get('maturity_state', 'COLDSTART')
                    except Exception:
                        pass  # Use default COLDSTART
                
                # Compute confidence for each timestamp based on available data
                for i, (ts, fused_z, qflag) in enumerate(zip(scores_df.index, scores_df['fused'], quality_flag)):
                    # Compute how many samples we have up to this point
                    sample_count = i + 1
                    # Is this a quality issue point?
                    is_quality_issue = qflag in ('EXTREME_VOLATILITY', 'EXTREME_ANOMALY')
                    
                    conf = compute_health_confidence(
                        maturity_state=maturity_state,
                        sample_count=sample_count,
                        is_extrapolated=False,  # Health timeline is always from actual data
                        has_quality_issues=is_quality_issue
                    )
                    confidence_values.append(round(conf, 3))
            except Exception as e:
                Console.warn(f"Failed to compute health confidence: {e}", component="HEALTH")
                # Fill with None if confidence computation fails
                confidence_values = [None] * len(scores_df)
        else:
            # No confidence module available
            confidence_values = [None] * len(scores_df)
        
        ts_values = normalize_timestamp_series(scores_df.index).to_list()
        return pd.DataFrame({
            'Timestamp': ts_values,
            'HealthIndex': smoothed_health.round(2).to_list(),
            'RawHealthIndex': raw_health.round(2).to_list(),
            'QualityFlag': quality_flag.astype(str).to_list(),
            'HealthZone': zones.astype(str).to_list(),
            'FusedZ': scores_df['fused'].round(4).to_list(),
            'Confidence': confidence_values  # V11: Health confidence
        })
    
    def _generate_regime_timeline(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate regime timeline with confidence.
        
        V11: Adds AssignmentConfidence from regime_confidence column (computed in Phase 2).
        """
        regimes = pd.to_numeric(scores_df['regime_label'], errors='coerce').astype('Int64')
        ts_values = normalize_timestamp_series(scores_df.index).to_list()
        
        # V11: Get assignment confidence if available (computed in regimes.py predict_regime_with_confidence)
        assignment_confidence = None
        if 'regime_confidence' in scores_df.columns:
            assignment_confidence = scores_df['regime_confidence'].round(3).to_list()
        
        # V11: Get regime version if available
        regime_version = None
        if 'regime_version' in scores_df.columns:
            regime_version = scores_df['regime_version'].to_list()
        
        result = pd.DataFrame({
            'Timestamp': ts_values,
            'RegimeLabel': regimes.to_list(),
            'RegimeState': (scores_df['regime_state'].astype(str).to_list() if 'regime_state' in scores_df.columns else [str('unknown')] * len(scores_df))
        })
        
        # Add V11 columns if available
        if assignment_confidence is not None:
            result['AssignmentConfidence'] = assignment_confidence
        if regime_version is not None:
            result['RegimeVersion'] = regime_version
            
        return result
    
    def _generate_sensor_defects(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate per-sensor defect analysis."""
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        defect_data = []
        
        for detector in detector_cols:
            # Defensive: validate detector column name
            if detector is None or (isinstance(detector, float) and pd.isna(detector)):
                Console.warn("Skipping NULL detector column name", component="DEFECTS", equip_id=self.equip_id, run_id=self.run_id)
                continue
            detector_col = str(detector)
            
            # Use SQL-safe human-readable label instead of raw code
            detector_label = get_detector_label(detector_col, sql_safe=True)
            
            # Determine family from label (first word before space/paren)
            family_parts = detector_label.split(' ')[0] if ' ' in detector_label else detector_label.split('(')[0]
            family = family_parts.strip()

            # Safely access values; skip if column missing unexpectedly
            if detector not in scores_df.columns:
                Console.warn(f"Missing detector column: {detector}", component="DEFECTS", detector=detector, equip_id=self.equip_id, run_id=self.run_id)
                continue
            values = pd.to_numeric(scores_df[detector], errors='coerce').abs()
            violations = values > 2.0
            violation_count = int(violations.sum())
            total_points = int(len(values)) if len(values) else 0
            violation_pct = (violation_count / total_points * 100) if total_points > 0 else 0.0
            
            # Determine severity
            if violation_pct > 20:
                severity = "CRITICAL"
            elif violation_pct > 10:
                severity = "HIGH"
            elif violation_pct > 5:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            defect_data.append({
                'DetectorType': detector_label,
                'DetectorFamily': family,
                'Severity': severity,
                'ViolationCount': violation_count,
                'ViolationPct': round(violation_pct, 2),
                'MaxZ': round(float(values.max()) if len(values) else 0.0, 4),
                'AvgZ': round(float(values.mean()) if len(values) else 0.0, 4),
                'CurrentZ': round(float(values.iloc[-1]) if len(values) else 0.0, 4),
                'ActiveDefect': bool(violations.iloc[-1]) if len(violations) else False
            })
        
        return pd.DataFrame(defect_data).sort_values('ViolationPct', ascending=False)
    
    def _generate_sensor_hotspots_table(
        self,
        sensor_zscores: pd.DataFrame,
        sensor_values: pd.DataFrame,
        train_mean: Optional[pd.Series],
        train_std: Optional[pd.Series],
        warn_z: float,
        alert_z: float,
        top_n: int
    ) -> pd.DataFrame:
        """Summarize top sensors by peak z-score deviation."""
        empty_schema = {
            'SensorName': [], 'MaxTimestamp': [], 'LatestTimestamp': [], 'MaxAbsZ': [],
            'MaxSignedZ': [], 'LatestAbsZ': [], 'LatestSignedZ': [], 'ValueAtPeak': [],
            'LatestValue': [], 'TrainMean': [], 'TrainStd': [], 'AboveWarnCount': [],
            'AboveAlertCount': []
        }

        if sensor_zscores is None or sensor_zscores.empty:
            return pd.DataFrame(empty_schema)

        records: List[Dict[str, Any]] = []
        for sensor in sensor_zscores.columns:
            series = sensor_zscores[sensor].dropna()
            if series.empty:
                continue
            abs_series = series.abs()
            max_idx = abs_series.idxmax()
            max_abs = float(abs_series.loc[max_idx])
            max_signed = float(series.loc[max_idx])
            latest_ts = series.index[-1]
            latest_signed = float(series.iloc[-1])
            latest_abs = abs(latest_signed)
            above_warn = int((abs_series >= warn_z).sum())
            above_alert = int((abs_series >= alert_z).sum()) if alert_z > 0 else 0

            value_at_peak = None
            latest_value = None
            if sensor in sensor_values.columns:
                try:
                    value_at_peak = sensor_values.loc[max_idx, sensor]
                except Exception:
                    value_at_peak = sensor_values[sensor].reindex([max_idx]).iloc[-1]
                try:
                    latest_value = sensor_values.loc[latest_ts, sensor]
                except Exception:
                    latest_value = sensor_values[sensor].iloc[-1]

            train_mean_val = (float(cast(Any, train_mean.get(sensor))) if isinstance(train_mean, pd.Series) and sensor in train_mean.index and pd.notna(train_mean.get(sensor)) else None)
            train_std_val = (float(cast(Any, train_std.get(sensor))) if isinstance(train_std, pd.Series) and sensor in train_std.index and pd.notna(train_std.get(sensor)) else None)

            records.append({
                'SensorName': sensor,
                'MaxTimestamp': normalize_timestamp_scalar(max_idx),
                'LatestTimestamp': normalize_timestamp_scalar(latest_ts),
                'MaxAbsZ': round(max_abs, 4),
                'MaxSignedZ': round(max_signed, 4),
                'LatestAbsZ': round(latest_abs, 4),
                'LatestSignedZ': round(latest_signed, 4),
                'ValueAtPeak': (float(cast(Any, value_at_peak)) if value_at_peak is not None and pd.notna(value_at_peak) else None),
                'LatestValue': (float(cast(Any, latest_value)) if latest_value is not None and pd.notna(latest_value) else None),
                'TrainMean': train_mean_val,
                'TrainStd': train_std_val,
                'AboveWarnCount': above_warn,
                'AboveAlertCount': above_alert
            })

        if not records:
            return pd.DataFrame(empty_schema)

        df = pd.DataFrame(records)
        df = df[df['MaxAbsZ'] >= warn_z]
        if df.empty:
            return pd.DataFrame(empty_schema)
        df = df.sort_values('MaxAbsZ', ascending=False)
        if top_n > 0:
            df = df.head(top_n)
        return df.reset_index(drop=True)

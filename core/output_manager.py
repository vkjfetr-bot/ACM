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
from core.observability import Console, Heartbeat

# Optional observability integration (P0 SQL ops tracking)
try:
    from core.observability import record_sql_op, Span
    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    _OBSERVABILITY_AVAILABLE = False
    record_sql_op = None
    Span = None

# whitelist of SQL tables we will write to (defined early so class methods can use it)
ALLOWED_TABLES = {
    'ACM_Scores_Wide','ACM_Episodes','ACM_EpisodesQC',
    'ACM_HealthTimeline','ACM_RegimeTimeline',
    'ACM_RegimeTransitions','ACM_ContributionCurrent','ACM_ContributionTimeline',
    'ACM_DriftSeries','ACM_ThresholdCrossings',
    'ACM_AlertAge','ACM_SensorRanking','ACM_RegimeOccupancy',
    'ACM_HealthHistogram','ACM_RegimeStability',
    'ACM_DefectSummary','ACM_DefectTimeline','ACM_SensorDefects',
    'ACM_HealthZoneByPeriod','ACM_SensorAnomalyByPeriod',
    'ACM_DetectorCorrelation','ACM_CalibrationSummary',
    'ACM_RegimeDwellStats','ACM_DriftEvents','ACM_EpisodeMetrics',
    'ACM_EpisodeDiagnostics','ACM_EpisodeCulprits',
    'ACM_DataQuality',
    # 'ACM_Scores_Long', -- DEPRECATED v11.0.0: Redundant with ACM_Scores_Wide
    'ACM_DriftSeries',
    'ACM_Anomaly_Events','ACM_Regime_Episodes',
    'ACM_PCA_Models','ACM_PCA_Loadings','ACM_PCA_Metrics',
    'ACM_Run_Stats','ACM_SinceWhen',
    'ACM_SensorHotspots','ACM_SensorHotspotTimeline',
    # v10.0.0 Forecasting & RUL tables (consolidated from 12â†’4 tables)
    'ACM_HealthForecast',        # Replaces ACM_HealthForecast_TS, ACM_HealthForecast_Continuous
    'ACM_FailureForecast',       # Replaces ACM_FailureForecast_TS, ACM_FailureHazard_TS, ACM_EnhancedFailureProbability_TS
    'ACM_SensorForecast',        # Physical sensor value forecasts (Motor Current, Temperature, etc.)
    'ACM_RUL',                   # Replaces ACM_RUL_TS, ACM_RUL_Summary, ACM_RUL_Attribution
    'ACM_ForecastingState',      # New: persistent model state with optimistic locking
    # Adaptive configuration (v10.0.0)
    'ACM_AdaptiveConfig',        # New: per-equipment and global config with auto-tuning
    # Other detector-level tables (unchanged)
    'ACM_DetectorForecast_TS','ACM_SensorNormalized_TS',
    'ACM_OMRContributionsLong','ACM_FusionQualityReport',
    'ACM_OMRTimeline','ACM_RegimeStats','ACM_DailyFusedProfile',
    'ACM_OMR_Diagnostics','ACM_Forecast_QualityMetrics',
    'ACM_HealthDistributionOverTime',
    # Adaptive threshold metadata
    'ACM_ThresholdMetadata',
    # v10.1.0 Regime-conditioned forecasting tables (migration 63)
    'ACM_RUL_ByRegime',           # Per-regime RUL estimates with degradation rates
    'ACM_RegimeHazard',           # Per-regime hazard rates and survival probabilities
    'ACM_ForecastContext',        # Unified forecast context with OMR/drift/regime state
    'ACM_AdaptiveThresholds_ByRegime',  # Per-regime adaptive thresholds
    # v10.2.0 Resource monitoring
    'ACM_ResourceMetrics',        # Per-section CPU/memory/time metrics
    # v11.0.0 New tables
    'ACM_ActiveModels',           # Active model versions per equipment
    'ACM_RegimeDefinitions',      # Regime centroids and metadata
    'ACM_DataContractValidation', # Data contract validation history
    'ACM_SeasonalPatterns',       # Detected seasonal patterns
    'ACM_AssetProfiles',          # Asset similarity profiles
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

def _read_csv_with_peek(path: Union[str, Path], ts_col_hint: Optional[str], engine: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Read CSV and auto-detect timestamp column."""
    path = Path(path)
    try:
        # Ensure engine is one of accepted options for pandas
        _engine = engine if engine in {None, 'c', 'python', 'pyarrow', 'python-fwf'} else None
        df = pd.read_csv(path, engine=cast(Any, _engine))
    except Exception as e:
        raise ValueError(f"Failed to read CSV {path}: {e}")
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    
    # Auto-detect timestamp column if not provided
    ts_col = ts_col_hint
    if not ts_col:
        candidates = ['timestamp', 'time', 'datetime', 'date']
        for candidate in candidates:
            if candidate in df.columns:
                ts_col = candidate
                break
        if not ts_col:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].iloc[:10], errors='raise')
                        ts_col = col
                        break
                    except:
                        continue
    if not ts_col:
        raise ValueError(f"Could not find timestamp column in {path}")
    return df, ts_col


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
    - Z=z_threshold (5.0): Health â‰ˆ 15% (serious anomaly)
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
    # At z=0: normalized << 0, sigmoid â‰ˆ 0, health â‰ˆ 100
    # At z=z_threshold/2: normalized = 0, sigmoid = 0.5, health = 50
    # At z=z_threshold: normalized > 0, sigmoid â‰ˆ 0.85, health â‰ˆ 15
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
            'files_written': 0,
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
        # 'ts' indicates we should use a sentinel timestamp (1900-01-01 00:00:00)
        self._sql_required_defaults: Dict[str, Dict[str, Any]] = {
            'ACM_DefectTimeline': {
                'Timestamp': 'ts', 'FusedZ': 0.0, 'HealthIndex': 0.0, 'HealthZone': 'UNKNOWN',
                'EventType': 'ZONE_CHANGE', 'FromZone': 'START', 'ToZone': 'GOOD'
            },
            'ACM_HealthTimeline': {
                'Timestamp': 'ts', 'HealthIndex': 0.0, 'HealthZone': 'GOOD', 'FusedZ': 0.0
            },
            'ACM_ThresholdCrossings': {
                'Timestamp': 'ts', 'DetectorType': 'fused', 'ZScore': 0.0, 'Threshold': 0.0, 'Direction': 'up'
            },
            'ACM_HealthZoneByPeriod': {
                'PeriodStart': 'ts', 'PeriodType': 'DAY', 'HealthZone': 'GOOD', 'ZonePct': 0.0,
                'ZoneCount': 0, 'TotalPoints': 0
            },
            'ACM_SensorAnomalyByPeriod': {
                'PeriodStart': 'ts', 'PeriodType': 'DAY', 'DetectorType': 'UNKNOWN', 'AnomalyRatePct': 0.0
            },
            'ACM_ContributionCurrent': {
                'DetectorType': 'UNKNOWN', 'ContributionPct': 0.0, 'ZScore': 0.0
            },
            'ACM_ContributionTimeline': {
                'Timestamp': 'ts', 'DetectorType': 'UNKNOWN', 'ContributionPct': 0.0
            },
            'ACM_SensorDefects': {
                'DetectorType': 'UNKNOWN', 'Severity': 'LOW', 'ViolationCount': 0, 'ViolationPct': 0.0,
                'MaxZ': 0.0, 'AvgZ': 0.0, 'CurrentZ': 0.0, 'ActiveDefect': 0
            },
            'ACM_DriftSeries': {
                'Timestamp': 'ts', 'DriftValue': 0.0
            },
            'ACM_DriftEvents': {
                'Timestamp': 'ts', 'SegmentStart': 'ts', 'SegmentEnd': 'ts', 'Value': 0.0
            },
            'ACM_RegimeTransitions': {
                'FromLabel': -1, 'ToLabel': -1, 'Count': 0, 'Prob': 0.0
            },
            'ACM_RegimeDwellStats': {
                'RegimeLabel': -1, 'Runs': 0, 'MeanSeconds': 0.0, 'MedianSeconds': 0.0, 'MinSeconds': 0.0, 'MaxSeconds': 0.0
            },
            'ACM_RegimeTimeline': {
                'Timestamp': 'ts', 'RegimeLabel': -1, 'RegimeState': 'unknown'
            },
            'ACM_RegimeOccupancy': {
                'RegimeLabel': -1, 'RecordCount': 0, 'Percentage': 0.0
            },
            'ACM_HealthHistogram': {
                'HealthBin': '0-10', 'RecordCount': 0, 'Percentage': 0.0
            },
            'ACM_CalibrationSummary': {
                'DetectorType': 'UNKNOWN', 'ClipZ': AnalyticsConstants.DEFAULT_CLIP_Z
            },
            'ACM_CulpritHistory': {
                'StartTimestamp': 'ts',
                'EndTimestamp': 'ts',
                'DurationHours': 0.0,
                'PrimaryDetector': 'unknown'
            },
            'ACM_EpisodeMetrics': {
                'TotalEpisodes': 0, 'TotalDurationHours': 0.0, 'AvgDurationHours': 0.0, 'MedianDurationHours': 0.0,
                'MaxDurationHours': 0.0, 'MinDurationHours': 0.0, 'RatePerDay': 0.0, 'MeanInterarrivalHours': 0.0
            },
            'ACM_SensorHotspots': {
                'SensorName': 'UNKNOWN', 'MaxTimestamp': 'ts', 'LatestTimestamp': 'ts',
                'MaxAbsZ': 0.0, 'MaxSignedZ': 0.0, 'LatestAbsZ': 0.0, 'LatestSignedZ': 0.0,
                'ValueAtPeak': 0.0, 'LatestValue': 0.0, 'TrainMean': 0.0, 'TrainStd': 0.0,
                'AboveWarnCount': 0, 'AboveAlertCount': 0
            },
            'ACM_EnhancedFailureProbability_TS': {
                'Timestamp': 'ts', 'ForecastHorizon_Hours': 0.0,
                'FailureProbability': 0.0, 'RiskLevel': 'UNKNOWN'
            },
            'ACM_FailureCausation': {
                'PredictedFailureTime': 'ts', 'Detector': 'unknown',
                'FailurePattern': 'unknown'
            },
            'ACM_EnhancedMaintenanceRecommendation': {
                'UrgencyScore': 0.0, 'MaintenanceRequired': 0
            },
            'ACM_MaintenanceRecommendation': {
                'EarliestMaintenance': 'ts',
                'PreferredWindowStart': 'ts', 
                'PreferredWindowEnd': 'ts',
                'FailureProbAtWindowEnd': 0.0,
                'Comment': ''
            },
            'ACM_RecommendedActions': {
                'Action': 'unspecified'
            },
            'ACM_SensorHotspotTimeline': {
                'Timestamp': 'ts', 'SensorName': 'UNKNOWN', 'Rank': 0, 'AbsZ': 0.0,
                'SignedZ': 0.0, 'Value': 0.0, 'Level': 'GOOD'
            },
            'ACM_SinceWhen': {
                'AlertZone': 'GOOD', 'DurationHours': 0.0, 'StartTimestamp': 'ts', 'RecordCount': 0
            },
            'ACM_SensorNormalized_TS': {
                'Timestamp': 'ts', 'SensorName': 'UNKNOWN', 'NormValue': 0.0,
                'ZScore': 0.0, 'AnomalyLevel': 'GOOD', 'EpisodeActive': 0
            },
            'ACM_OMRContributionsLong': {
                'Timestamp': 'ts', 'SensorName': 'UNKNOWN', 'ContributionScore': 0.0,
                'ContributionPct': 0.0, 'OMR_Z': 0.0
            },
            'ACM_FusionQualityReport': {
                'Detector': 'UNKNOWN', 'Weight': 0.0, 'Present': 0,
                'MeanZ': 0.0, 'MaxZ': 0.0, 'Points': 0
            },
            'ACM_OMRTimeline': {
                'Timestamp': 'ts', 'OMR_Z': 0.0, 'OMR_Weight': 0.0
            },
            'ACM_RegimeStats': {
                'RegimeLabel': -1, 'OccupancyPct': 0.0, 'AvgDwellSeconds': 0.0,
                'FusedMean': 0.0, 'FusedP90': 0.0
            },
            'ACM_DailyFusedProfile': {
                'ProfileDate': 'date', 'DayOfWeek': 0, 'Hour': 0, 'FusedMean': 0.0, 'FusedP90': 0.0,
                'FusedP95': 0.0, 'RecordCount': 0
            },
            'ACM_HealthForecast_Continuous': {
                'Timestamp': 'ts', 'ForecastHealth': 0.0, 'CI_Lower': 0.0, 'CI_Upper': 0.0,
                'SourceRunID': 'UNKNOWN', 'EquipID': 0, 'MergeWeight': 0.0
            },
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
            'ACM_RUL_TS': {
                'Timestamp': 'ts', 'RUL_Hours': 0.0, 'Method': 'Multipath'
            },
            'ACM_FailureHazard_TS': {
                'Timestamp': 'ts', 'HazardRaw': 0.0, 'HazardSmooth': 0.0, 'Survival': 0.0,
                'FailureProb': 0.0, 'RunID': 'UNKNOWN', 'EquipID': 0
            },
            'ACM_EpisodeDiagnostics': {
                'episode_id': 0, 'peak_z': 0.0, 'peak_timestamp': 'ts', 'duration_h': 0.0,
                'dominant_sensor': 'UNKNOWN', 'severity': 'UNKNOWN', 'severity_reason': 'UNKNOWN',
                'avg_z': 0.0, 'min_health_index': 0.0
            },
            'ACM_HealthDistributionOverTime': {
                'BucketStart': 'ts', 'BucketSeconds': 3600, 'FusedP50': 0.0,
                'FusedP75': 0.0, 'FusedP90': 0.0, 'FusedP95': 0.0,
                'HealthP50': 0.0, 'HealthP10': 0.0, 'BucketCount': 0
            },
            'ACM_ChartGenerationLog': {
                'ChartName': 'unknown', 'Status': 'unknown', 'Reason': '',
                'DurationSeconds': 0.0, 'Timestamp': 'ts'
            }
            ,
            'ACM_AlertAge': {
                'AlertZone': 'GOOD', 'DurationHours': 0.0, 'StartTimestamp': 'ts', 'RecordCount': 0
            }
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
    
    def load_data(self, cfg: Dict[str, Any], start_utc: Optional[pd.Timestamp] = None, end_utc: Optional[pd.Timestamp] = None, equipment_name: Optional[str] = None, sql_mode: bool = False):
        """
        Load training and scoring data from CSV files or SQL historian.
        
        Consolidates data loading into OutputManager since it's part of the I/O pipeline.
        
        Args:
            cfg: Configuration dictionary
            start_utc: Optional start time for SQL window queries
            end_utc: Optional end time for SQL window queries
            equipment_name: Equipment name for SQL historian queries (e.g., 'FD_FAN', 'GAS_TURBINE')
            sql_mode: If True, load from SQL historian; if False, load from CSV files
        """
        # Use consistent config access with fallback
        data_cfg = cfg.get("data", {})
        train_path = data_cfg.get("train_csv")
        score_path = data_cfg.get("score_csv")
        
        # SQL mode: Load from historian SP instead of CSV
        if sql_mode:
            if not self.sql_client:
                raise ValueError("[DATA] SQL mode requested but no SQL client available")
            if not equipment_name:
                raise ValueError("[DATA] SQL mode requires equipment_name parameter")
            return self._load_data_from_sql(cfg, equipment_name, start_utc, end_utc)
        
        # ACM is SQL-only mode - CSV operations removed
        
        # CSV mode: Cold-start mode: If no training data, use first N% of score data for training
        cold_start_mode = False
        if not train_path and score_path:
            Console.info("" + "Cold-start mode: No training data provided, will split score data", component="DATA")
            cold_start_mode = True
        elif not train_path or not score_path:
            raise ValueError("[DATA] Please set data.train_csv and data.score_csv in config.")

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
        ts_col_cfg = _cfg_get(data_cfg, "timestamp_col", None)
        io_engine = _cfg_get(data_cfg, "io_engine", None)
        # Allow override in data.max_fill_ratio (preferred), fallback to runtime.max_fill_ratio
        max_fill_ratio = float(_cfg_get(data_cfg, "max_fill_ratio", _cfg_get(cfg, "runtime.max_fill_ratio", 0.20)))
        
        # COLD-02: Configurable cold-start split ratio (default 0.6 = 60% train, 40% score)
        cold_start_split_ratio = float(_cfg_get(data_cfg, "cold_start_split_ratio", 0.6))
        if not (0.1 <= cold_start_split_ratio <= 0.9):
            Console.warn(f"Invalid cold_start_split_ratio={cold_start_split_ratio}, using default 0.6", component="DATA", invalid_value=cold_start_split_ratio)
            cold_start_split_ratio = 0.6
        
        # COLD-03: Minimum samples validation (default 500)
        min_train_samples = int(_cfg_get(data_cfg, "min_train_samples", 500))

        # Read CSVs (with heartbeat)
        if cold_start_mode:
            hb = Heartbeat("Reading CSV (cold-start: will split data)", next_hint="parse timestamps", eta_hint=10).start()
            score_raw, ts_score = _read_csv_with_peek(score_path, ts_col_cfg, engine=io_engine)
            
            # COLD-03: Validate sufficient data for cold-start
            if len(score_raw) < 10:
                raise ValueError(f"[DATA] Cold-start requires at least 10 rows, got {len(score_raw)}")
            
            ts_col = ts_col_cfg or ts_score
            split_idx = int(len(score_raw) * cold_start_split_ratio)
            train_raw = score_raw.iloc[:split_idx].copy()
            score_raw = score_raw.iloc[split_idx:].copy()
            ts_train = ts_score
            
            # COLD-03: Warn if training samples below recommended minimum
            if len(train_raw) < min_train_samples:
                Console.warn(f"Cold-start training data ({len(train_raw)} rows) is below recommended minimum ({min_train_samples} rows)", component="DATA", actual_rows=len(train_raw), min_required=min_train_samples)
                Console.warn(f"Model quality may be degraded. Consider: more data, higher split_ratio (current: {cold_start_split_ratio:.2f})", component="DATA", split_ratio=cold_start_split_ratio)
            
            Console.info("" + f"Cold-start split ({cold_start_split_ratio:.1%}): {len(train_raw)} train rows, {len(score_raw)} score rows", component="DATA")
            hb.stop()
        else:
            hb = Heartbeat("Reading CSVs (train & score)", next_hint="parse timestamps", eta_hint=10).start()
            train_raw, ts_train = _read_csv_with_peek(train_path, ts_col_cfg, engine=io_engine)
            score_raw, ts_score = _read_csv_with_peek(score_path, ts_col_cfg, engine=io_engine)
            hb.stop()
            ts_col = ts_col_cfg or ts_train or ts_score

        # Apply window constraints if provided (for SQL mode)
        if start_utc:
            train_raw = train_raw[pd.to_datetime(train_raw[ts_col], errors='coerce') >= start_utc]
        if end_utc:
            score_raw = score_raw[pd.to_datetime(score_raw[ts_col], errors='coerce') < end_utc]

        # Parse timestamps / index
        hb = Heartbeat("Parsing timestamps & indexing", next_hint="numeric pruning", eta_hint=8).start()
        train = _parse_ts_index(train_raw, ts_col)
        score = _parse_ts_index(score_raw, ts_col)
        hb.stop()

        now_cutoff = _future_cutoff_ts(cfg)
        train, tz_stripped_train, future_train = _coerce_local_and_filter_future(train, "TRAIN", now_cutoff)
        score, tz_stripped_score, future_score = _coerce_local_and_filter_future(score, "SCORE", now_cutoff)
        tz_stripped_total = tz_stripped_train + tz_stripped_score
        future_rows_total = future_train + future_score
        
        # COLD-03: Validate training sample count (both cold-start and normal modes)
        if len(train) < min_train_samples:
            if cold_start_mode:
                # Already warned during split
                pass
            else:
                Console.warn(f"Training data ({len(train)} rows) is below recommended minimum ({min_train_samples} rows)", component="DATA", actual_rows=len(train), min_required=min_train_samples)
                Console.warn("Model quality may be degraded. Consider providing more training data.", component="DATA", actual_rows=len(train), min_required=min_train_samples)

        # Keep numeric only (same set across train/score)
        hb = Heartbeat("Selecting numeric sensor columns", next_hint="cadence check", eta_hint=4).start()
        train_num = _infer_numeric_cols(train)
        score_num = _infer_numeric_cols(score)
        kept = sorted(list(set(train_num).intersection(score_num)))
        dropped = [c for c in train.columns if c not in kept]
        train = train[kept]
        score = score[kept]
        train = train.astype(np.float32)
        score = score.astype(np.float32)
        hb.stop()

        # Cadence check + guardrails
        hb = Heartbeat("Cadence check / resample / fill small gaps", next_hint="finalize", eta_hint=15).start()
        # Ensure type stability for indexes
        train.index = pd.DatetimeIndex(train.index)
        score.index = pd.DatetimeIndex(score.index)
        train.index = pd.DatetimeIndex(train.index)
        score.index = pd.DatetimeIndex(score.index)
        cad_ok_train = _check_cadence(cast(pd.DatetimeIndex, train.index), sampling_secs)
        cad_ok_score = _check_cadence(cast(pd.DatetimeIndex, score.index), sampling_secs)
        cadence_ok = bool(cad_ok_train and cad_ok_score)

        native_train = _native_cadence_secs(cast(pd.DatetimeIndex, train.index))
        if sampling_secs and math.isfinite(native_train) and sampling_secs < native_train:
            Console.warn(f"Requested resample ({sampling_secs}s) < native cadence ({native_train:.1f}s) - skipping to avoid upsample.", component="DATA", requested_secs=sampling_secs, native_secs=native_train)
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
        hb.stop()

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
        return train, score, meta
    
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
        hb = Heartbeat("Calling usp_ACM_GetHistorianData_TEMP", next_hint="parse results", eta_hint=5).start()
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
            hb.stop()
        
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
        hb = Heartbeat("Splitting train/score data", next_hint="parse timestamps", eta_hint=3).start()
        
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
        
        hb.stop()
        
        # Parse timestamps / index
        hb = Heartbeat("Parsing timestamps & indexing", next_hint="numeric pruning", eta_hint=4).start()
        
        # Handle empty train in batch mode
        if len(train_raw) == 0 and not is_coldstart:
            # Create empty DataFrame with DatetimeIndex matching score columns
            train = pd.DataFrame(columns=train_raw.columns)
            train.index = pd.DatetimeIndex([], name=ts_col)
        else:
            train = _parse_ts_index(train_raw, ts_col)
        
        score = _parse_ts_index(score_raw, ts_col)
        hb.stop()
        
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
        hb = Heartbeat("Selecting numeric sensor columns", next_hint="cadence check", eta_hint=2).start()
        
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
        
        hb.stop()
        
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
        
        hb = Heartbeat("Cadence check / resample / fill small gaps", next_hint="finalize", eta_hint=8).start()
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
        hb.stop()

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
        """Prepare DataFrame for SQL insertion with proper type coercion."""
        if df.empty:
            return df
            
        out = df.copy()
        non_numeric_cols = non_numeric_cols or set()

        # Timestamps â†’ UTC naive and strip microseconds (floor to seconds) for SQL
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                ts_series = pd.to_datetime(out[col], errors='coerce')
                ts_series = ts_series.dt.tz_localize(None)
                # Floor to whole seconds using lowercase 's' (Pandas FutureWarning fix)
                ts_series = ts_series.dt.floor('s')
                # Convert to native Python datetime objects to avoid SQL Server microsecond overflow
                # Suppress FutureWarning - we're intentionally using np.array() wrapper
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*to_pydatetime.*', category=FutureWarning)
                    out[col] = np.array(ts_series.dt.to_pydatetime())

        # Count infs before replace
        num_only = out.select_dtypes(include=[np.number])
        inf_count = np.isinf(num_only.values).sum() if not num_only.empty else 0
        if inf_count > 0:
            Console.warn(f"Replaced {int(inf_count)} Inf/-Inf values with None for SQL compatibility", component="OUTPUT", inf_count=int(inf_count), columns=len(num_only.columns))

        # Replace in one pass
        out = out.replace({np.inf: None, -np.inf: None})
        out = out.where(pd.notnull(out), None)

        # Convert numpy scalars to Python types
        # Preserve integer types for ID columns to avoid SQL cast errors
        integer_columns = {'EquipID', 'RegimeLabel', 'episode_id'}
        for col in out.columns:
            if col in non_numeric_cols:
                continue
            if pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype(object).where(pd.isna(out[col]), out[col].astype(int))
            elif col in integer_columns and pd.api.types.is_numeric_dtype(out[col]):
                # Keep as integer for SQL INT columns
                out[col] = out[col].astype('Int64')  # nullable integer
            elif pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].astype(float)

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
    
    def write_dataframe(self, 
                       df: pd.DataFrame, 
                       artifact_name: str,
                       sql_table: Optional[str] = None,
                       sql_columns: Optional[Dict[str, str]] = None,
                       non_numeric_cols: Optional[set] = None,
                       add_created_at: bool = False,
                       allow_repair: bool = True) -> Dict[str, Any]:
        """
        Write DataFrame to SQL (file output disabled).
        
        Args:
            df: DataFrame to write
            artifact_name: Logical name for the artifact (used for caching/logging)
            sql_table: Optional SQL table name
            sql_columns: Optional column mapping for SQL (df_col -> sql_col)
            non_numeric_cols: Set of columns to treat as non-numeric for SQL
            add_created_at: Whether to add CreatedAt timestamp column for SQL
            allow_repair: OUT-17: If False, block SQL write when required fields missing instead of auto-repairing
            
        Returns:
            Dictionary with write results and metadata
        """
        start_time = time.time()
        
        result = {
            'file_written': False,
            'sql_written': False,
            'rows': len(df),
            'error': None
        }
        
        # OUT-18: Check if auto-flush needed before write
        if self._should_auto_flush():
            Console.info("" + f"Auto-flushing batch (rows={self._current_batch.total_rows}, age={time.time() - self._current_batch.created_at:.1f}s)", component="OUTPUT")
            self.flush()
        
        # OUT-18: Apply backpressure if too many in-flight operations
        self._wait_for_futures_capacity()
        
        try:
            # OUT-18: Update batch row tracking
            with self._batch_lock:
                self._current_batch.total_rows += len(df)
            
            # Attempt SQL write if configured
            if sql_table and self._check_sql_health():
                try:
                    # Prepare data for SQL
                    sql_df = self._prepare_dataframe_for_sql(df, non_numeric_cols or set())
                    
                    # Apply column mapping if provided
                    if sql_columns:
                        # First, select only columns that are in the mapping (source columns)
                        mapped_source_cols = [col for col in sql_columns.keys() if col in sql_df.columns]
                        sql_df = sql_df[mapped_source_cols]
                        # Then rename to target SQL column names
                        sql_df = sql_df.rename(columns=sql_columns)
                    
                    # Add metadata columns (required for all SQL tables)
                    # Preserve existing RunID if already present (e.g., hazard table with truncated ID)
                    if "RunID" not in sql_df.columns:
                        sql_df["RunID"] = self.run_id
                    if "EquipID" not in sql_df.columns:
                        sql_df["EquipID"] = self.equip_id or 0
                    # OUT-17: Apply per-table required NOT NULL defaults with repair policy
                    sql_df, repair_info = self._apply_sql_required_defaults(sql_table, sql_df, allow_repair)
                    # MED-02: Log what was repaired for observability
                    if repair_info.get('repairs_needed'):
                        Console.info(f"Applied defaults to {sql_table}: {repair_info.get('missing_fields')}", component="SCHEMA")
                    
                    # OUT-17: Block write if repairs needed but not allowed
                    if not allow_repair and repair_info['repairs_needed']:
                        raise ValueError(f"Required fields missing and allow_repair=False: {repair_info['missing_fields']}")
                    
                    # Add CreatedAt timestamp only if requested
                    if add_created_at:
                        sql_df["CreatedAt"] = pd.Timestamp.now().tz_localize(None)
                    
                    # Special-case: avoid PK collisions for maintenance recommendation by deleting existing
                    if sql_table == "ACM_MaintenanceRecommendation" and self.sql_client is not None:
                        try:
                            with self.sql_client.cursor() as cur:
                                cur.execute(
                                    "DELETE FROM dbo.[ACM_MaintenanceRecommendation] WHERE RunID = ? AND EquipID = ?",
                                    (self.run_id, int(self.equip_id or 0))
                                )
                                if hasattr(self.sql_client, "commit"):
                                    self.sql_client.commit()
                                elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                                    if not getattr(self.sql_client.conn, "autocommit", True):
                                        self.sql_client.conn.commit()
                        except Exception as del_ex:
                            Console.warn(f"Pre-delete failed for {sql_table}: {del_ex}", component="OUTPUT", table=sql_table, equip_id=self.equip_id, run_id=self.run_id, error_type=type(del_ex).__name__)

                    # FORECAST-UPSERT-05: Route forecast tables to MERGE upsert methods (v10 schema)
                    if sql_table == "ACM_HealthForecast":
                        inserted = self._upsert_health_forecast(sql_df)
                    elif sql_table == "ACM_FailureForecast":
                        inserted = self._upsert_failure_forecast(sql_df)
                    elif sql_table == "ACM_DetectorForecast_TS":
                        inserted = self._upsert_detector_forecast_ts(sql_df)
                    elif sql_table == "ACM_SensorForecast":
                        inserted = self._upsert_sensor_forecast(sql_df)
                    elif sql_table == "ACM_RUL":
                        inserted = self._bulk_insert_sql(sql_table, sql_df)  # RUL uses standard insert
                    else:
                        # Bulk insert with batching for all other tables
                        inserted = self._bulk_insert_sql(sql_table, sql_df)
                    
                    result['sql_written'] = inserted > 0
                    if result['sql_written']: 
                        self.stats['sql_writes'] += 1
                    
                except Exception as e:
                    Console.warn(f"SQL write failed for {sql_table}: {e}", component="OUTPUT", table=sql_table, rows=len(sql_df), equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                    result['error'] = str(e)
                    self.stats['sql_failures'] += 1
            elif not sql_table:
                # No target; skip quietly
                return result
            
        except Exception as e:
            Console.error(f"Failed to process artifact {artifact_name}: {e}", component="OUTPUT", artifact=artifact_name, equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
            result['error'] = str(e)
            raise
        
        finally:
            elapsed = time.time() - start_time
            self.stats['write_time'] += elapsed
            
            # FCST-15: Cache DataFrame for downstream modules (lightweight)
            # Use artifact_name directly as key
            self._artifact_cache[artifact_name] = df
        
        return result

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
    
    # ==================== DATA TRANSFORMATION METHODS ====================
    
    def melt_scores_long(self, df_wide: pd.DataFrame, equip_id: int, run_id: str, source: str = "ACM") -> pd.DataFrame:
        """
        Convert a wide sensor frame (index = timestamps, columns = sensors) to
        the canonical long format for dbo.ScoresTS.
        """
        if not isinstance(df_wide.index, pd.DatetimeIndex):
            raise ValueError("melt_scores_long expects a DatetimeIndex.")
        out = df_wide.copy()
        out.index = pd.to_datetime(out.index)
        out = out.reset_index().rename(columns={"index": "EntryDateTime"})
        long = out.melt(id_vars=["EntryDateTime"], var_name="Sensor", value_name="Value")
        long["EquipID"] = int(equip_id)
        long["Source"] = source
        long["RunID"] = run_id
        return long

    # ==================== MISSING ANALYTICS GENERATORS (MED-03..MED-09) ====================

    def _generate_culprit_history(self, scores_df: pd.DataFrame, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Extract culprit sensors per episode with basic scoring.
        Returns DataFrame suitable for ACM_CulpritHistory: RunID, EquipID, EpisodeID, Sensor, Contribution.
        """
        if episodes_df is None or len(episodes_df) == 0:
            return pd.DataFrame(columns=["RunID","EquipID","EpisodeID","Sensor","Contribution"])  
        rows = []
        for i, ep in episodes_df.iterrows():
            ep_id = ep.get("EpisodeID", i)
            sensors = ep.get("CulpritSensors") or ep.get("Sensors") or []
            if isinstance(sensors, str):
                try:
                    sensors = [s.strip() for s in sensors.split(',') if s.strip()]
                except Exception:
                    sensors = []
            for s in sensors:
                rows.append({
                    "RunID": self.run_id,
                    "EquipID": int(self.equip_id or 0),
                    "EpisodeID": int(ep_id),
                    "Sensor": str(s),
                    "Contribution": float(ep.get("PeakZ", 0.0))
                })
        return pd.DataFrame(rows)

    def _generate_episode_metrics(self, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Compute RUN-LEVEL episode summary statistics for ACM_EpisodeMetrics table.
        Returns a single-row DataFrame with aggregate metrics across all episodes in this run.
        """
        if episodes_df is None or len(episodes_df) == 0:
            return pd.DataFrame([{
                "RunID": self.run_id,
                "EquipID": int(self.equip_id or 0),
                "TotalEpisodes": 0,
                "TotalDurationHours": 0.0,
                "AvgDurationHours": 0.0,
                "MedianDurationHours": 0.0,
                "MaxDurationHours": 0.0,
                "MinDurationHours": 0.0,
                "RatePerDay": 0.0,
                "MeanInterarrivalHours": 0.0
            }])
        
        # Get durations directly from duration_hours or duration_h column (already calculated)
        durations = []
        if 'duration_hours' in episodes_df.columns:
            durations = episodes_df['duration_hours'].dropna().tolist()
        elif 'duration_h' in episodes_df.columns:
            durations = episodes_df['duration_h'].dropna().tolist()
        
        # Calculate interarrival times using peak_timestamp
        timestamps = []
        timestamp_col = 'peak_timestamp' if 'peak_timestamp' in episodes_df.columns else 'PeakTimestamp'
        if timestamp_col in episodes_df.columns:
            for i, ep in episodes_df.iterrows():
                ts_val = ep.get(timestamp_col)
                ts = pd.to_datetime(ts_val) if ts_val is not None else pd.NaT
                if isinstance(ts, pd.Timestamp):
                    timestamps.append(ts)
        
        # Calculate interarrival times (time between consecutive episodes)
        interarrivals = []
        if len(timestamps) > 1:
            timestamps_sorted = sorted(timestamps)
            for i in range(1, len(timestamps_sorted)):
                interarrival_hours = (timestamps_sorted[i] - timestamps_sorted[i-1]).total_seconds() / 3600.0
                interarrivals.append(interarrival_hours)
        
        # Compute statistics
        total_eps = len(episodes_df)
        total_dur_h = sum(durations) if durations else 0.0
        avg_dur_h = np.mean(durations) if durations else 0.0
        median_dur_h = np.median(durations) if durations else 0.0
        max_dur_h = max(durations) if durations else 0.0
        min_dur_h = min(durations) if durations else 0.0
        mean_interarrival_h = np.mean(interarrivals) if interarrivals else 0.0
        
        # Calculate rate per day (episodes per 24 hours)
        rate_per_day = 0.0
        if len(timestamps) >= 2:
            time_span_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
            if time_span_hours > 0:
                rate_per_day = (total_eps / time_span_hours) * 24.0
        
        return pd.DataFrame([{
            "RunID": self.run_id,
            "EquipID": int(self.equip_id or 0),
            "TotalEpisodes": int(total_eps),
            "TotalDurationHours": round(total_dur_h, 2),
            "AvgDurationHours": round(avg_dur_h, 2),
            "MedianDurationHours": round(median_dur_h, 2),
            "MaxDurationHours": round(max_dur_h, 2),
            "MinDurationHours": round(min_dur_h, 2),
            "RatePerDay": round(rate_per_day, 3),
            "MeanInterarrivalHours": round(mean_interarrival_h, 2)
        }])

    def _generate_episode_diagnostics(self, episodes_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Create per-episode diagnostics rows with timestamps and severity label.
        Returns DataFrame for ACM_EpisodeDiagnostics.
        
        NOTE: This is a DUPLICATE write that happens AFTER write_episodes().
        We must use the EXACT same column names and extract the SAME values to avoid overwriting good data with UNKNOWN.
        """
        # Use lowercase column names to match write_episodes output
        cols = ["episode_id", "peak_timestamp", "dominant_sensor", "duration_h", "severity", "avg_z", "peak_z"]
        if episodes_df is None or len(episodes_df) == 0:
            return pd.DataFrame(columns=cols)
        
        rows = []
        for i, ep in episodes_df.iterrows():
            # Extract timestamp
            _ts_val = ep.get("peak_timestamp") or ep.get("PeakTimestamp") or ep.get("start_ts")
            ts = pd.to_datetime(_ts_val) if _ts_val is not None else pd.NaT
            
            # Extract peak_z (try multiple column names)
            peak_z = float(ep.get("peak_z", ep.get("peak_fused_z", ep.get("PeakZ", 0.0))))
            
            # Calculate severity from peak_z
            if pd.isna(peak_z):
                severity = "UNKNOWN"
            elif peak_z >= 6:
                severity = "CRITICAL"
            elif peak_z >= 4:
                severity = "HIGH"
            elif peak_z >= 2:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            # Extract dominant_sensor from culprits
            dominant_sensor = ep.get("dominant_sensor", "UNKNOWN")
            if dominant_sensor == "UNKNOWN" and "culprits" in ep:
                culprit_str = ep["culprits"]
                if pd.notna(culprit_str) and culprit_str != '':
                    # culprits field already contains formatted label from format_culprit_label()
                    # e.g., "Multivariate Outlier (PCA-TÂ²)" or "Multivariate Outlier (PCA-TÂ²) â†’ SensorName"
                    # Extract just the detector label (before " â†’ " if sensor attribution exists)
                    if ' â†’ ' in str(culprit_str):
                        dominant_sensor = str(culprit_str).split(' â†’ ')[0].strip()
                    else:
                        dominant_sensor = str(culprit_str).strip()
            
            # Extract avg_z
            avg_z = float(ep.get("avg_z", ep.get("avg_fused_z", 0.0)))
            
            # Calculate duration_h from duration_s if needed
            duration_h = ep.get("duration_h", ep.get("duration_hours", 0.0))
            if duration_h == 0.0 and "duration_s" in ep:
                duration_h = float(ep["duration_s"]) / 3600.0
            
            rows.append({
                "episode_id": int(ep.get("episode_id", ep.get("EpisodeID", i))),
                "peak_timestamp": ts,
                "dominant_sensor": dominant_sensor,
                "duration_h": duration_h,
                "severity": severity,
                "avg_z": avg_z,
                "peak_z": peak_z,
            })
        return pd.DataFrame(rows)

    def _generate_omr_contributions_long(self, scores_df: pd.DataFrame, omr_contributions: pd.DataFrame, 
                                          top_n: int = 5, downsample_factor: int = 10) -> pd.DataFrame:
        """Melt OMR contributions wide DataFrame to long format, keeping only TOP N contributors per timestamp.
        
        This optimization reduces storage by ~95% for equipment with many sensors (e.g., 792 sensors â†’ 15 per timestamp).
        Expected input columns end with '_contrib'; outputs columns: Timestamp, SensorName, ContributionScore, ContributionPct.
        
        Args:
            scores_df: Scores DataFrame with omr_z column
            omr_contributions: Wide-format OMR contributions (columns = sensors)
            top_n: Number of top contributors to keep per timestamp (default 5, reduced from 15 for P6.2)
            downsample_factor: Keep every Nth timestamp (default 10 for P6.2)
        """
        if omr_contributions is None or len(omr_contributions) == 0:
            return pd.DataFrame(columns=["Timestamp","SensorName","ContributionScore","ContributionPct","OMR_Z","RunID","EquipID"])
        df = omr_contributions.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        
        # P6.2: Apply timestamp downsampling before melting (reduces timestamps by factor of N)
        if downsample_factor > 1:
            df = df.iloc[::downsample_factor]
        
        # Reset index - preserve original index name or use 'index'
        df = df.reset_index()
        # Rename first column (the old index) to 'Timestamp' regardless of its original name
        if df.columns[0] != 'Timestamp':
            df = df.rename(columns={df.columns[0]: 'Timestamp'})
        value_cols = [c for c in df.columns if c.endswith("_contrib")] or [c for c in df.columns if c not in ["Timestamp"]]
        long = df.melt(id_vars=["Timestamp"], var_name="SensorName", value_name="ContributionScore")
        long["SensorName"] = long["SensorName"].astype(str).str.replace("_contrib","", regex=False)
        
        # TASK-1-FIX: Ensure ContributionScore is never NULL to prevent SQL constraint violations
        # Replace NaN/NULL with 0.0 before any downstream processing
        long["ContributionScore"] = pd.to_numeric(long["ContributionScore"], errors="coerce").fillna(0.0)
        
        # PERF-FIX: Keep only TOP N contributors per timestamp to reduce storage by ~95%
        # Sort by absolute contribution score (descending) and keep top N per timestamp
        long["AbsContrib"] = long["ContributionScore"].abs()
        long = long.sort_values(["Timestamp", "AbsContrib"], ascending=[True, False])
        long = long.groupby("Timestamp").head(top_n).drop(columns=["AbsContrib"])
        
        # Calculate ContributionPct: percentage of total contribution at each timestamp
        long["ContributionPct"] = 0.0
        if not long.empty and "ContributionScore" in long.columns:
            # Group by timestamp and calculate percentage (now only for top N)
            long["ContributionPct"] = long.groupby("Timestamp")["ContributionScore"].transform(
                lambda x: (x / x.sum() * 100) if x.sum() != 0 else 0.0
            )
        
        # Add OMR_Z from scores_df if available
        long["OMR_Z"] = 0.0
        if scores_df is not None and "omr_z" in scores_df.columns:
            # Merge OMR_Z values by timestamp
            omr_z_map = scores_df.set_index(scores_df.index)["omr_z"].to_dict()
            long["OMR_Z"] = long["Timestamp"].map(omr_z_map).fillna(0.0)
        
        long["RunID"] = self.run_id
        long["EquipID"] = int(self.equip_id or 0)
        return long

    def _generate_regime_stats(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Compute regime statistics: occupancy, dwell time, fused score stats.
        Returns DataFrame for ACM_RegimeStats with correct column names.
        """
        if scores_df is None or len(scores_df) == 0 or "regime_label" not in scores_df.columns:
            return pd.DataFrame(columns=["RunID", "EquipID", "RegimeLabel", "OccupancyPct", "AvgDwellSeconds", "FusedMean", "FusedP90"])
        
        regimes = pd.to_numeric(scores_df["regime_label"], errors="coerce")
        fused = pd.to_numeric(scores_df.get("fused", pd.Series(dtype=float)), errors="coerce")
        regimes = regimes.dropna().astype(int)
        
        # Total points for occupancy calculation
        total_points = len(regimes)
        if total_points == 0:
            return pd.DataFrame(columns=["RunID", "EquipID", "RegimeLabel", "OccupancyPct", "AvgDwellSeconds", "FusedMean", "FusedP90"])
        
        # Estimate dt_hours from timestamp index
        if hasattr(scores_df.index, 'to_series'):
            ts = scores_df.index.to_series()
            if len(ts) > 1:
                dt_hours = float((ts.diff().median().total_seconds()) / 3600.0)
            else:
                dt_hours = 0.5  # default 30 min
        else:
            dt_hours = 0.5
        
        out = []
        for label in sorted(regimes.unique()):
            mask = regimes == label
            count = int(mask.sum())
            occupancy_pct = float(count / total_points * 100.0)
            
            # Estimate average dwell time (consecutive points in same regime)
            # Simple approximation: total time in regime / number of transitions into regime
            regime_transitions = (mask.astype(int).diff().fillna(0) == 1).sum()
            if regime_transitions > 0:
                avg_dwell_seconds = float((count * dt_hours * 3600) / regime_transitions)
            else:
                avg_dwell_seconds = float(count * dt_hours * 3600)  # All in one segment
            
            # Fused score stats for this regime
            fused_in_regime = fused[mask].dropna()
            if len(fused_in_regime) > 0:
                fused_mean = float(fused_in_regime.mean())
                fused_p90 = float(fused_in_regime.quantile(0.90))
            else:
                fused_mean = 0.0
                fused_p90 = 0.0
            
            out.append({
                "RunID": self.run_id,
                "EquipID": int(self.equip_id or 0),
                "RegimeLabel": int(label),
                "OccupancyPct": round(occupancy_pct, 2),
                "AvgDwellSeconds": round(avg_dwell_seconds, 1),
                "FusedMean": round(fused_mean, 4),
                "FusedP90": round(fused_p90, 4)
            })
        return pd.DataFrame(out)

    def _generate_daily_fused_profile(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate fused scores by date for daily profiling.
        Returns DataFrame for ACM_DailyFusedProfile: ProfileDate, AvgFusedScore, MaxFusedScore, MinFusedScore, SampleCount.
        """
        if scores_df is None or len(scores_df) == 0 or "fused" not in scores_df.columns:
            return pd.DataFrame(columns=["RunID","EquipID","ProfileDate","AvgFusedScore","MaxFusedScore","MinFusedScore","SampleCount"])
        
        idx = pd.to_datetime(scores_df.index, errors="coerce")
        fused = pd.to_numeric(scores_df["fused"], errors="coerce")
        
        # Group by date
        df = pd.DataFrame({"Date": idx.date, "FusedZ": fused})
        grp = df.groupby("Date", dropna=True)["FusedZ"].agg(
            AvgFusedScore='mean',
            MaxFusedScore='max',
            MinFusedScore='min',
            SampleCount='count'
        ).reset_index().rename(columns={"Date": "ProfileDate"})
        
        grp["RunID"] = self.run_id
        grp["EquipID"] = int(self.equip_id or 0)
        return grp[["RunID","EquipID","ProfileDate","AvgFusedScore","MaxFusedScore","MinFusedScore","SampleCount"]]

    def _generate_health_histogram(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Bin health index into 10-point buckets (0-100).
        Returns DataFrame for ACM_HealthHistogram: BinStart, BinEnd, Count.
        """
        if scores_df is None or len(scores_df) == 0:
            return pd.DataFrame(columns=["RunID","EquipID","BinStart","BinEnd","Count"])
        fused = pd.to_numeric(scores_df["fused"], errors="coerce")
        # v10.1.0: Use centralized _health_index function with softer sigmoid
        health = _health_index(fused) if fused is not None else pd.Series([], dtype=float)
        bins = list(range(0, 101, 10))
        cats = pd.cut(health, bins=bins, include_lowest=True, right=False)
        counts = cats.value_counts().sort_index()
        rows = []
        for interval, cnt in counts.items():
            try:
                start = int(getattr(interval, 'left', 0))
                end = int(getattr(interval, 'right', 0))
            except Exception:
                start, end = 0, 0
            rows.append({"RunID": self.run_id, "EquipID": int(self.equip_id or 0), "BinStart": start, "BinEnd": end, "Count": int(cnt)})
        return pd.DataFrame(rows)

    def _generate_alert_age(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Compute time since first alert crossing based on fused z threshold (default 3.0).
        Returns DataFrame for ACM_AlertAge with a single row per run.
        """
        threshold = 3.0
        cols = ["RunID","EquipID","FirstAlert","LastAlert","AlertAgeHours"]
        if scores_df is None or len(scores_df) == 0 or "fused" not in scores_df.columns:
            return pd.DataFrame([{c: None for c in cols}])
        idx = pd.to_datetime(scores_df.index, errors="coerce")
        fused = pd.to_numeric(scores_df["fused"], errors="coerce")
        alert_mask = fused >= threshold
        if not alert_mask.any():
            return pd.DataFrame([{"RunID": self.run_id, "EquipID": int(self.equip_id or 0), "FirstAlert": None, "LastAlert": None, "AlertAgeHours": 0.0}])
        first_ts = idx[alert_mask].min()
        last_ts = idx[alert_mask].max()
        age_hours = float((last_ts - first_ts).total_seconds()/3600.0) if (isinstance(first_ts, pd.Timestamp) and isinstance(last_ts, pd.Timestamp)) else 0.0
        return pd.DataFrame([{"RunID": self.run_id, "EquipID": int(self.equip_id or 0), "FirstAlert": first_ts, "LastAlert": last_ts, "AlertAgeHours": age_hours}])

    # ==================== SPECIALIZED SQL WRITE FUNCTIONS ====================
    # These replace all the scattered write functions from data_io.py, storage.py, 
    # and sql_analytics_writer.py with a unified interface
    
    def write_scores_ts(self, df: pd.DataFrame, run_id: str) -> int:
        """Write scores timeseries to ACM_Scores_Long table.
        
        DEPRECATED (v11.0.0): This table is redundant with ACM_Scores_Wide.
        ACM_Scores_Long stores melted detector scores (one row per timestamp+detector),
        while ACM_Scores_Wide stores the same data in wide format (one row per timestamp).
        
        Wide format is:
        - More efficient for time-series queries (fewer rows, less I/O)
        - Used by all Grafana dashboards
        - The authoritative scores table
        
        This function now returns 0 immediately without writing.
        The table will be dropped in a future version.
        
        SQL Schema for ACM_Scores_Long (for reference):
            - Id (BIGINT, identity)
            - RunID (UNIQUEIDENTIFIER)
            - EquipID (INT)
            - Timestamp (DATETIME2)
            - SensorName (NVARCHAR(128)) - sensor being scored, NULL for detector-level
            - DetectorName (NVARCHAR(64)) - detector name (ar1, pca_spe, etc.)
            - Score (FLOAT) - z-score value
            - Threshold (FLOAT) - threshold used for this detector
            - IsAnomaly (BIT) - whether score exceeds threshold
        """
        # v11.0.0: Skip writing to deprecated ACM_Scores_Long table
        # Scores are already written to ACM_Scores_Wide by write_scores()
        return 0
        
        # --- ORIGINAL CODE BELOW (preserved for reference) ---
        if not self._check_sql_health():
            return 0
            
        try:
            sql_df = df.copy()

            if sql_df.empty:
                return 0

            # Case A: already long from melt_scores_long (legacy format)
            if {'EntryDateTime', 'Sensor', 'Value'}.issubset(set(sql_df.columns)):
                sql_df = sql_df.rename(columns={
                    'EntryDateTime': 'Timestamp',
                    'Sensor': 'DetectorName',  # Map to correct SQL column
                    'Value': 'Score'  # Map to correct SQL column
                })
                # Ensure required metadata exists
                if 'RunID' not in sql_df.columns:
                    sql_df['RunID'] = run_id
                if 'EquipID' not in sql_df.columns:
                    sql_df['EquipID'] = self.equip_id or 0
                # Add missing columns with defaults
                if 'SensorName' not in sql_df.columns:
                    sql_df['SensorName'] = None  # Detector-level scores, no specific sensor
                if 'Threshold' not in sql_df.columns:
                    sql_df['Threshold'] = 3.0  # Default threshold
                if 'IsAnomaly' not in sql_df.columns:
                    sql_df['IsAnomaly'] = (sql_df['Score'].abs() >= 3.0).astype(int)
                sql_df['Timestamp'] = pd.to_datetime(sql_df['Timestamp']).dt.tz_localize(None)
                sql_df = sql_df.dropna(subset=['Score'])
                return self._bulk_insert_sql('ACM_Scores_Long', sql_df)

            # Case B: wide - melt to long
            if sql_df.index.name == 'timestamp' or isinstance(sql_df.index, pd.DatetimeIndex):
                sql_df = sql_df.reset_index().rename(columns={sql_df.columns[0]: 'Timestamp'})
            detector_cols = [c for c in sql_df.columns if c.endswith('_z') or c.startswith('ACM_')]
            if not detector_cols:
                return 0
            long_df = sql_df.melt(
                id_vars=['Timestamp'],
                value_vars=detector_cols,
                var_name='DetectorName',  # Map to correct SQL column
                value_name='Score'  # Map to correct SQL column
            )
            # Normalize detector names (remove _z suffix)
            long_df['DetectorName'] = long_df['DetectorName'].str.replace('_z', '', regex=False)
            long_df['Timestamp'] = pd.to_datetime(long_df['Timestamp']).dt.tz_localize(None)
            long_df['RunID'] = run_id
            long_df['EquipID'] = self.equip_id or 0
            # Add missing columns with defaults
            long_df['SensorName'] = None  # Detector-level scores, no specific sensor
            long_df['Threshold'] = 3.0  # Default threshold
            long_df['IsAnomaly'] = (long_df['Score'].abs() >= 3.0).astype(int)
            long_df = long_df.dropna(subset=['Score'])
            return self._bulk_insert_sql('ACM_Scores_Long', long_df)
            
        except Exception as e:
            Console.warn(f"write_scores_ts failed: {e}", component="OUTPUT", equip_id=self.equip_id, run_id=run_id, rows=len(df) if df is not None else 0, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def write_drift_ts(self, df: pd.DataFrame, run_id: str) -> int:
        """Write drift timeseries to ACM_DriftSeries table."""
        if not self._check_sql_health():
            return 0
            
        try:
            sql_df = df.copy()
            
            # Expected columns: Timestamp, DriftValue, RunID, EquipID
            if 'drift_z' in sql_df.columns:
                sql_df['DriftValue'] = sql_df['drift_z']
            elif 'cusum_z' in sql_df.columns:
                sql_df['DriftValue'] = sql_df['cusum_z']
            elif 'value' in sql_df.columns:
                sql_df['DriftValue'] = sql_df['value']
            else:
                return 0
            
            # Prepare timestamp
            if sql_df.index.name == 'timestamp' or isinstance(sql_df.index, pd.DatetimeIndex):
                sql_df = sql_df.reset_index()
                timestamp_col = sql_df.columns[0]
                sql_df = sql_df.rename(columns={timestamp_col: 'Timestamp'})
            
            # Add metadata
            sql_df['RunID'] = run_id
            sql_df['EquipID'] = self.equip_id or 0
            
            # Select final columns
            sql_df = sql_df[['RunID', 'EquipID', 'Timestamp', 'DriftValue']].copy()
            sql_df['Timestamp'] = pd.to_datetime(sql_df['Timestamp']).dt.tz_localize(None)
            sql_df = sql_df.dropna(subset=['DriftValue'])
            
            return self._bulk_insert_sql('ACM_DriftSeries', sql_df)
            
        except Exception as e:
            Console.warn(f"write_drift_ts failed: {e}", component="OUTPUT", equip_id=self.equip_id, run_id=run_id, rows=len(df) if df is not None else 0, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def write_anomaly_events(self, df: pd.DataFrame, run_id: str) -> int:
        """Write anomaly events to ACM_Anomaly_Events table."""
        if not self._check_sql_health() or df.empty:
            return 0
            
        try:
            sql_df = df.copy()
            
            # Map columns for ACM_Anomaly_Events (table uses StartTime/EndTime, not StartTs/EndTs)
            column_map = {
                'episode_id': 'EpisodeID',
                'start_ts': 'StartTime',
                'end_ts': 'EndTime',
                'peak_fused_z': 'PeakScore',
                'severity': 'Severity',
                'status': 'Status'
            }
            
            for old_col, new_col in column_map.items():
                if old_col in sql_df.columns:
                    sql_df[new_col] = sql_df[old_col]
            
            # Add metadata
            sql_df['RunID'] = run_id
            sql_df['EquipID'] = self.equip_id or 0
            
            # Add defaults
            if 'Severity' not in sql_df.columns:
                sql_df['Severity'] = 'info'
            if 'Status' not in sql_df.columns:
                sql_df['Status'] = 'OPEN'
            
            # Handle timestamps
            for ts_col in ['StartTime', 'EndTime']:
                if ts_col in sql_df.columns:
                    sql_df[ts_col] = pd.to_datetime(sql_df[ts_col]).dt.tz_localize(None)
            
            # Select final columns
            final_cols = ['RunID', 'EquipID', 'StartTime', 'EndTime', 'Severity']
            sql_df = sql_df[[c for c in final_cols if c in sql_df.columns]]
            
            return self._bulk_insert_sql('ACM_Anomaly_Events', sql_df)
            
        except Exception as e:
            Console.warn(f"write_anomaly_events failed: {e}", component="OUTPUT", equip_id=self.equip_id, run_id=run_id, rows=len(df) if df is not None else 0, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def write_regime_episodes(self, df: pd.DataFrame, run_id: str) -> int:
        """Write regime episodes to ACM_Regime_Episodes table."""
        if not self._check_sql_health() or df.empty:
            return 0
            
        try:
            sql_df = df.copy()
            
            # Map columns for ACM_Regime_Episodes
            column_map = {
                'start_ts': 'StartTs',
                'end_ts': 'EndTs',
                'regime_label': 'RegimeLabel',
                'duration_s': 'DurationSeconds',
                'stability_score': 'StabilityScore'
            }
            
            for old_col, new_col in column_map.items():
                if old_col in sql_df.columns:
                    sql_df[new_col] = sql_df[old_col]
            
            # Add metadata
            sql_df['RunID'] = run_id
            sql_df['EquipID'] = self.equip_id or 0
            
            # Handle timestamps
            for ts_col in ['StartTs', 'EndTs']:
                if ts_col in sql_df.columns:
                    sql_df[ts_col] = pd.to_datetime(sql_df[ts_col]).dt.tz_localize(None)
            
            # Select final columns
            final_cols = ['RunID', 'EquipID', 'StartTs', 'EndTs', 'RegimeLabel', 'DurationSeconds', 'StabilityScore']
            sql_df = sql_df[[c for c in final_cols if c in sql_df.columns]]
            
            return self._bulk_insert_sql('ACM_Regime_Episodes', sql_df)
            
        except Exception as e:
            Console.warn(f"write_regime_episodes failed: {e}", component="OUTPUT", equip_id=self.equip_id, run_id=run_id, rows=len(df) if df is not None else 0, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def write_pca_model(self, model_data: Dict[str, Any]) -> int:
        """Write PCA model metadata to ACM_PCA_Models table."""
        if not self._check_sql_health():
            return 0
            
        try:
            # Accept caller-provided schema and add common metadata when missing
            row = dict(model_data)
            row.setdefault('RunID', self.run_id)
            row.setdefault('EquipID', self.equip_id or 0)
            # Provide CreatedAt if not present
            if 'CreatedAt' not in row:
                row['CreatedAt'] = pd.Timestamp.now().tz_localize(None)
            sql_df = pd.DataFrame([row])
            
            return self._bulk_insert_sql('ACM_PCA_Models', sql_df)
            
        except Exception as e:
            Console.warn(f"write_pca_model failed: {e}", component="OUTPUT", equip_id=self.equip_id, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def write_pca_loadings(self, df: pd.DataFrame, run_id: str) -> int:
        """Write PCA loadings to ACM_PCA_Loadings table."""
        if not self._check_sql_health() or df.empty:
            return 0
            
        try:
            sql_df = df.copy()
            
            # Normalize possible caller column names
            sql_df['RunID'] = sql_df.get('RunID', run_id)
            sql_df['EquipID'] = sql_df.get('EquipID', self.equip_id or 0)
            if 'ComponentID' not in sql_df.columns and 'ComponentNo' in sql_df.columns:
                sql_df['ComponentID'] = sql_df['ComponentNo']
            if 'FeatureName' not in sql_df.columns and 'Sensor' in sql_df.columns:
                sql_df['FeatureName'] = sql_df['Sensor']
            # Keep common columns; _bulk_insert_sql will filter to table cols
            
            return self._bulk_insert_sql('ACM_PCA_Loadings', sql_df)
            
        except Exception as e:
            Console.warn(f"write_pca_loadings failed: {e}", component="OUTPUT", equip_id=self.equip_id, run_id=run_id, rows=len(df) if df is not None else 0, error_type=type(e).__name__, error=str(e)[:200])
            return 0
    
    def write_pca_metrics(self, pca_detector=None, tables_dir=None, enable_sql=True, df=None, run_id=None) -> int:
        """Write PCA metrics to ACM_PCA_Metrics table.
        
        Args:
            pca_detector: PCASubspaceDetector instance (new style)
            tables_dir: Path for file output (ignored in SQL-only mode)
            enable_sql: Whether SQL write is enabled
            df: Pre-built DataFrame (legacy style, optional)
            run_id: Run ID for legacy style
        """
        if not self._check_sql_health() or not enable_sql:
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

    def _upsert_pca_metrics(self, df: pd.DataFrame) -> int:
        """Upsert PCA metrics using DELETE by full PK scope to prevent data loss.
        
        Primary key is (RunID, EquipID, ComponentName, MetricType).
        CRITICAL: Delete only specific metric types being updated to prevent data loss.
        This method may be called multiple times per run with different metric types.
        """
        if df.empty or self.sql_client is None:
            return 0
        
        # Validate required columns for PK matching
        required_cols = ['RunID', 'EquipID', 'ComponentName', 'MetricType']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            Console.warn(f"_upsert_pca_metrics missing required columns: {missing_cols}", component="OUTPUT", missing_columns=missing_cols, available_columns=list(df.columns))
            return 0
        
        try:
            conn = self.sql_client.conn
            cursor = conn.cursor()
            
            # Get unique RunID+EquipID pairs
            # Get unique (RunID, EquipID, ComponentName, MetricType) tuples to delete
            pk_tuples = df[['RunID', 'EquipID', 'ComponentName', 'MetricType']].drop_duplicates()
            
            # DELETE existing rows by full PK scope (prevents data loss + PK collisions)
            deleted_count = 0
            for _, row in pk_tuples.iterrows():
                try:
                    cursor.execute("""
                        DELETE FROM ACM_PCA_Metrics 
                        WHERE RunID = ? AND EquipID = ? AND ComponentName = ? AND MetricType = ?
                        """,
                        (row['RunID'], row['EquipID'], row['ComponentName'], row['MetricType'])
                    )
                    deleted_count += cursor.rowcount
                except Exception as del_err:
                    Console.warn(f"DELETE failed for {row['ComponentName']}/{row['MetricType']}: {del_err}", component="OUTPUT", table="ACM_PCA_Metrics", component_name=row['ComponentName'], metric_type=row['MetricType'], error_type=type(del_err).__name__)
            
            if deleted_count > 0:
                Console.info(f"Deleted {deleted_count} existing PCA metric rows before upsert", component="OUTPUT")
            
            # Prepare bulk insert data
            insert_sql = """
            INSERT INTO ACM_PCA_Metrics (RunID, EquipID, ComponentName, MetricType, Value, Timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            # Build list of tuples for bulk insert
            rows_to_insert = []
            for _, row in df.iterrows():
                rows_to_insert.append((
                    row['RunID'],
                    row['EquipID'],
                    row.get('ComponentName', 'PCA'),
                    row['MetricType'],
                    row['Value'],
                    row.get('Timestamp', datetime.now())
                ))
            
            # Bulk insert all rows in one transaction
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
    
    def write_scores(self, 
                    scores_df: pd.DataFrame, 
                    run_dir: Path,
                    enable_sql: bool = True) -> Dict[str, Any]:
        """
        Write scores with optimized dual-write coordination.
        
        Replaces storage.write_scores_csv() with better performance.
        """
        # Prepare scores for output
        scores_for_output = scores_df.copy()
        scores_for_output.index.name = "timestamp"
        
        # Convert index to UTC naive timestamps
        if len(scores_for_output.index):
            try:
                scores_for_output.index = pd.to_datetime(scores_for_output.index).tz_localize(None)
            except Exception:
                scores_for_output.index = pd.to_datetime(scores_for_output.index).tz_localize(None)
        
        # Write with SQL dual-write if enabled
        sql_table = "ACM_Scores_Wide" if enable_sql else None
        score_columns = {
            "timestamp": "Timestamp",
            "ar1_z": "ar1_z", "pca_spe_z": "pca_spe_z", "pca_t2_z": "pca_t2_z",
            "iforest_z": "iforest_z", "gmm_z": "gmm_z",
            "cusum_z": "cusum_z", "drift_z": "drift_z", "hst_z": "hst_z", "fused": "fused",
            "regime_label": "regime_label", "transient_state": "transient_state"
        }
        
        # CHART-04: Use uniform timestamp format without 'T' or 'Z' suffixes
        return self.write_dataframe(
            scores_for_output.reset_index(),
            run_dir / "scores.csv",
            sql_table=sql_table,
            sql_columns=score_columns,
            non_numeric_cols={"RunID", "EquipID", "Timestamp", "regime_label", "transient_state"}
        )
    
    def write_episodes(self, 
                      episodes_df: pd.DataFrame, 
                      run_dir: Path,
                      enable_sql: bool = True) -> Dict[str, Any]:
        """
        Write episodes with optimized dual-write coordination.
        
        Replaces storage.write_episodes_csv() with better performance.
        
        NOTE: Individual episodes go to CSV only. Summary QC goes to ACM_Episodes SQL table.
        """
        if episodes_df.empty:
            # Write empty file
            empty_df = pd.DataFrame(columns=['episode_id', 'start_ts', 'end_ts', 'duration_s'])
            return self.write_dataframe(empty_df, run_dir / "episodes.csv")
        
        # Prepare episodes for output
        episodes_for_output = episodes_df.copy().reset_index(drop=True)
        
        # SCHEMA-FIX: Individual episodes go to ACM_EpisodeDiagnostics (not ACM_Episodes which is run-level summary)
        sql_table = "ACM_EpisodeDiagnostics" if enable_sql else None
        episode_columns = {
            'episode_id': 'episode_id',
            'peak_fused_z': 'peak_z',
            'peak_timestamp': 'peak_timestamp', 
            'duration_hours': 'duration_h',
            'dominant_sensor': 'dominant_sensor',  # Already extracted from culprits
            'severity': 'severity',
            'avg_fused_z': 'avg_z',
            'min_health_index': 'min_health_index'
        }
        
        # Add episode_id if missing (sequential)
        if 'episode_id' not in episodes_for_output.columns:
            episodes_for_output['episode_id'] = range(1, len(episodes_for_output) + 1)
        
        # Calculate duration_hours from duration_s if needed
        if 'duration_hours' not in episodes_for_output.columns and 'duration_s' in episodes_for_output.columns:
            episodes_for_output['duration_hours'] = episodes_for_output['duration_s'] / 3600.0
        
        # Extract peak_timestamp from start_ts if missing
        if 'peak_timestamp' not in episodes_for_output.columns and 'start_ts' in episodes_for_output.columns:
            episodes_for_output['peak_timestamp'] = episodes_for_output['start_ts']
        
        # Map regime_label to MaxRegimeLabel
        if 'regime_label' in episodes_for_output.columns:
            episodes_for_output['MaxRegimeLabel'] = episodes_for_output['regime_label']
        elif 'regime' in episodes_for_output.columns:
            episodes_for_output['MaxRegimeLabel'] = episodes_for_output['regime']
        
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
        else:
            episodes_for_output['dominant_sensor'] = 'UNKNOWN'
        
        # ALWAYS recalculate severity from peak_fused_z (overrides any pre-existing severity like 'info')
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
                else:
                    return 'LOW'
            episodes_for_output['severity'] = episodes_for_output['peak_fused_z'].apply(calculate_severity)
        elif 'severity' not in episodes_for_output.columns:
            episodes_for_output['severity'] = 'UNKNOWN'
        
        # Add defaults for SQL
        if enable_sql:
            if 'status' not in episodes_for_output.columns:
                episodes_for_output['status'] = 'CLOSED'
        
        # DEBUG: Show what we're writing  
        # Columns ready for write to ACM_EpisodeDiagnostics
        if not episodes_for_output.empty:
            Console.info(f"DEBUG: First row before write: severity={episodes_for_output.iloc[0].get('severity')}, peak_z={episodes_for_output.iloc[0].get('peak_fused_z')}, dominant={episodes_for_output.iloc[0].get('dominant_sensor')}", component="EPISODES")
        
        result = self.write_dataframe(
            episodes_for_output,
            run_dir / "episodes.csv",
            sql_table=sql_table,
            sql_columns=episode_columns,
            non_numeric_cols={
                "RunID", "EquipID", "episode_id", "peak_timestamp",
                "dominant_sensor", "severity", "min_health_index"
            }
        )
        
        # Also write run-level summary to ACM_Episodes
        if enable_sql and not episodes_df.empty:
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
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.stats,
            'avg_write_time': self.stats['write_time'] / max(1, self.stats['files_written']),
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
        Console.info(f"[OUTPUT] Finalized: {stats['files_written']} files, "
                f"{stats['sql_writes']} SQL ops, "
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
        PERF-OPT v11: Delete existing rows for RunID/EquipID from multiple tables in a SINGLE SQL batch.
        
        Optimization: Instead of 26+ individual DELETE round-trips, builds ONE batched SQL statement
        that deletes from all tables at once. This eliminates network round-trip overhead.
        
        Typical speedup: 2-3 seconds saved on comprehensive analytics.
        
        Args:
            tables: List of table names to clear for current RunID/EquipID
            
        Returns:
            Total tables processed (rows deleted not trackable with batched approach)
        """
        if not self.sql_client or not self.run_id:
            return 0
        
        start_time = time.perf_counter()
        tables_processed = 0
        
        try:
            cursor_factory = lambda: cast(Any, self.sql_client).cursor()
            
            # Phase 1: Build list of valid tables and their DELETE statements
            delete_statements = []
            for table_name in tables:
                if table_name not in ALLOWED_TABLES:
                    continue
                
                # Check if table exists (use cache)
                exists = self._table_exists_cache.get(table_name)
                if exists is None:
                    exists = _table_exists(cursor_factory, table_name)
                    self._table_exists_cache[table_name] = bool(exists)
                if not exists:
                    continue
                
                # Get table columns to check for RunID/EquipID
                if table_name in self._table_insertable_cache:
                    table_cols = self._table_insertable_cache[table_name]
                elif table_name in self._table_columns_cache:
                    table_cols = self._table_columns_cache[table_name]
                else:
                    try:
                        cols = set(_get_insertable_columns(cursor_factory, table_name))
                        if not cols:
                            cols = set(_get_table_columns(cursor_factory, table_name))
                        self._table_insertable_cache[table_name] = cols
                        table_cols = cols
                    except Exception:
                        continue
                
                # Build DELETE statement for this table
                if "RunID" in table_cols and "EquipID" in table_cols and self.equip_id is not None:
                    delete_statements.append(
                        f"DELETE FROM dbo.[{table_name}] WHERE RunID = @RunID AND EquipID = @EquipID"
                    )
                elif "RunID" in table_cols:
                    delete_statements.append(
                        f"DELETE FROM dbo.[{table_name}] WHERE RunID = @RunID"
                    )
                
                # Mark as pre-deleted regardless of execution (skip in _bulk_insert_sql)
                self._bulk_predeleted_tables.add(table_name)
                tables_processed += 1
            
            # Phase 2: Execute ALL deletes in a single batch (one network round-trip)
            if delete_statements:
                batch_sql = ";\n".join(delete_statements)
                cur = cursor_factory()
                try:
                    # Use sp_executesql for parameterized batch
                    param_sql = f"""
                    DECLARE @RunID NVARCHAR(36) = ?;
                    DECLARE @EquipID INT = ?;
                    {batch_sql}
                    """
                    cur.execute(param_sql, (self.run_id, int(self.equip_id or 0)))
                finally:
                    try:
                        cur.close()
                    except Exception:
                        pass
            
            elapsed = time.perf_counter() - start_time
            if tables_processed > 0:
                Console.info(f"Bulk pre-delete: {tables_processed} tables in {elapsed:.2f}s (batched)", component="OUTPUT")
            
        except Exception as e:
            Console.warn(f"Bulk pre-delete failed (non-fatal): {e}", component="OUTPUT", tables=len(tables), equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__)
        
        return tables_processed

    # ==================== COMPREHENSIVE ANALYTICS TABLES ====================
    
    def generate_all_analytics_tables(self, 
                                     scores_df: pd.DataFrame, 
                                     episodes_df: pd.DataFrame, 
                                     cfg: Dict[str, Any],
                                     tables_dir: Path,
                                     enable_sql: bool = True,
                                     sensor_context: Optional[Dict[str, Any]] = None) -> Dict[str, int]:  # Default to TRUE for SQL mode
        """
        Generate all 26 comprehensive analytics tables for complete analytical coverage.
        
        MANDATES that ALL data is written to SQL database - no exceptions!
        This provides the rock-solid analytical basis required by operators and engineers.
        
        PERFORMANCE: Uses batched transaction for all SQL writes to minimize commit overhead.
        """
        Console.info("Generating comprehensive analytics tables...", component="ANALYTICS")
        table_count = 0
        sql_count = 0
        
        # FORCE SQL WRITES - this is mandatory in SQL mode
        force_sql = enable_sql or (self.sql_client is not None)

        sensor_values = None
        sensor_zscores = None
        sensor_train_mean = None
        sensor_train_std = None
        omr_contributions = None
        if sensor_context:
            values_candidate = sensor_context.get('values')
            zscores_candidate = sensor_context.get('z_scores')
            if isinstance(values_candidate, pd.DataFrame) and len(values_candidate.columns):
                sensor_values = values_candidate.reindex(scores_df.index)
            if isinstance(zscores_candidate, pd.DataFrame) and len(zscores_candidate.columns):
                sensor_zscores = zscores_candidate.reindex(scores_df.index)
            mean_candidate = sensor_context.get('train_mean')
            std_candidate = sensor_context.get('train_std')
            if isinstance(mean_candidate, pd.Series):
                sensor_train_mean = mean_candidate
            if isinstance(std_candidate, pd.Series):
                sensor_train_std = std_candidate
            omni = sensor_context.get('omr_contributions') if isinstance(sensor_context, dict) else None
            if isinstance(omni, pd.DataFrame) and len(omni.index):
                try:
                    omr_contributions = omni.reindex(scores_df.index)
                except Exception:
                    omr_contributions = omni
        
        # ANA-12: Tiered analytics - check fused availability
        has_fused = 'fused' in scores_df.columns
        has_regimes = 'regime_label' in scores_df.columns
        
        # PERF-OPT: Define all analytics tables for bulk pre-delete
        # This eliminates 26+ individual DELETE round-trips (saves ~2-3s)
        analytics_tables = [
            "ACM_DetectorCorrelation", "ACM_CalibrationSummary", "ACM_OMRContributionsLong",
            "ACM_FusionQualityReport", "ACM_RegimeOccupancy", "ACM_RegimeTransitions",
            "ACM_RegimeDwellStats", "ACM_HealthTimeline", "ACM_HealthDistributionOverTime",
            "ACM_RegimeTimeline", "ACM_OMRTimeline", "ACM_RegimeStats", "ACM_ContributionCurrent",
            "ACM_ContributionTimeline", "ACM_DriftSeries", "ACM_ThresholdCrossings",
            "ACM_SinceWhen", "ACM_SensorRanking", "ACM_HealthHistogram", "ACM_DailyFusedProfile",
            "ACM_AlertAge", "ACM_RegimeStability", "ACM_DefectSummary", "ACM_DefectTimeline",
            "ACM_SensorDefects", "ACM_HealthZoneByPeriod", "ACM_SensorAnomalyByPeriod",
            "ACM_SensorHotspots", "ACM_SensorNormalized_TS", "ACM_SensorHotspotTimeline"
        ]
        
        # Use batched transaction for all SQL writes
        with self.batched_transaction():
            try:
                # PERF-OPT: Bulk delete all analytics tables upfront
                if force_sql and self.sql_client and self.run_id:
                    self._bulk_delete_analytics_tables(analytics_tables)
                
                # TIER-A & TIER-B: Write detector-level and regime tables
                # These provide diagnostic value independent of fused scores
                
                # TIER-B: Detector correlation (no fused dependency)
                detector_corr_df = self._generate_detector_correlation(scores_df)
                result = self.write_dataframe(detector_corr_df, tables_dir / "detector_correlation.csv",
                                            sql_table="ACM_DetectorCorrelation" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                # TIER-B: Calibration summary (no fused dependency)
                calibration_df = self._generate_calibration_summary(scores_df, cfg)
                result = self.write_dataframe(calibration_df, tables_dir / "calibration_summary.csv",
                                              sql_table="ACM_CalibrationSummary" if force_sql else None,
                                              add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1

                # Write OMR per-sensor contributions (long format) when available
                if omr_contributions is not None and not omr_contributions.empty:
                    try:
                        omr_long = self._generate_omr_contributions_long(scores_df, omr_contributions)
                        # OMR-FIX: Ensure no NULL scores (causes SQL constraint violation)
                        if "ContributionScore" in omr_long.columns:
                            omr_long["ContributionScore"] = omr_long["ContributionScore"].fillna(0.0)
                        
                        result = self.write_dataframe(
                            omr_long,
                            tables_dir / "omr_contributions_long.csv",
                            sql_table="ACM_OMRContributionsLong" if force_sql else None,
                            add_created_at=True,
                            non_numeric_cols={"SensorName"}
                        )
                        table_count += 1
                        if result.get('sql_written'): sql_count += 1
                    except Exception as e:
                        Console.warn(f"Failed to write omr_contributions_long.csv: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])

                # Fusion quality snapshot (weights + basic stats)
                try:
                    fusion_q_df = self._generate_fusion_quality_report(scores_df, cfg)
                    if not fusion_q_df.empty:
                        result = self.write_dataframe(
                            fusion_q_df,
                            tables_dir / "fusion_quality_report.csv",
                            sql_table="ACM_FusionQualityReport" if force_sql else None,
                            add_created_at=True
                        )
                        table_count += 1
                        if result.get('sql_written'): sql_count += 1
                    else:
                        table_count += 1
                except Exception as e:
                    Console.warn(f"Failed to write fusion_quality_report.csv: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                
                # TIER-A: Regime tables (if available, no fused dependency)
                if has_regimes:
                    Console.info("Writing Tier-A regime tables...", component="ANALYTICS")
                    regime_occ_df = self._generate_regime_occupancy(scores_df)
                    result = self.write_dataframe(regime_occ_df, tables_dir / "regime_occupancy.csv",
                                                sql_table="ACM_RegimeOccupancy" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1
                    
                    regime_trans_df = self._generate_regime_transition_matrix(scores_df)
                    result = self.write_dataframe(regime_trans_df, tables_dir / "regime_transition_matrix.csv",
                                                sql_table="ACM_RegimeTransitions" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1
                    
                    regime_dwell_df = self._generate_regime_dwell_stats(scores_df)
                    result = self.write_dataframe(regime_dwell_df, tables_dir / "regime_dwell_stats.csv",
                                                sql_table="ACM_RegimeDwellStats" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1
                
                # ANA-12: If fused missing, return after Tier-A/B tables
                if not has_fused:
                    Console.warn("No fused scores - Tier-C (health, defects, episodes) tables skipped", component="ANALYTICS")
                    Console.info(f"Wrote {table_count} Tier-A/B tables ({sql_count} to SQL)", component="ANALYTICS")
                    return {"csv_tables": table_count, "sql_tables": sql_count}
                
                # TIER-C: Fused-dependent tables (all below require 'fused' column)
                Console.info("Writing Tier-C fused-dependent tables...", component="ANALYTICS")
                
                # 1. Health Timeline (enhanced with smoothing and quality flags)
                health_df = self._generate_health_timeline(scores_df, cfg)
                result = self.write_dataframe(health_df, tables_dir / "health_timeline.csv", 
                                            sql_table="ACM_HealthTimeline" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1

                # 1b. Health distribution over time (hourly buckets)
                try:
                    hdist_df = self._generate_health_distribution_over_time(scores_df)
                    result = self.write_dataframe(
                        hdist_df,
                        tables_dir / "health_distribution_over_time.csv",
                        sql_table="ACM_HealthDistributionOverTime" if force_sql else None,
                        sql_columns={"Count": "BucketCount"},
                        add_created_at=True
                    )
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1
                except Exception as e:
                    Console.warn(f"Failed to write health_distribution_over_time.csv: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                
                # 2. Regime Timeline (enhanced)  
                if 'regime_label' in scores_df.columns:
                    regime_df = self._generate_regime_timeline(scores_df)
                    result = self.write_dataframe(regime_df, tables_dir / "regime_timeline.csv",
                                                sql_table="ACM_RegimeTimeline" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1

                # 2b. OMR timeline (if available)
                if 'omr_z' in scores_df.columns:
                    try:
                        omr_tl_df = self._generate_omr_timeline(scores_df, cfg)
                        result = self.write_dataframe(
                            omr_tl_df,
                            tables_dir / "omr_timeline.csv",
                            sql_table="ACM_OMRTimeline" if force_sql else None,
                            add_created_at=True
                        )
                        table_count += 1
                        if result.get('sql_written'): sql_count += 1
                    except Exception as e:
                        Console.warn(f"Failed to write omr_timeline.csv: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])

                    # Compact regime stats for dashboards
                    try:
                        regime_stats_df = self._generate_regime_stats(scores_df)
                        result = self.write_dataframe(
                            regime_stats_df,
                            tables_dir / "regime_stats.csv",
                            sql_table="ACM_RegimeStats" if force_sql else None,
                            add_created_at=True
                        )
                        table_count += 1
                        if result.get('sql_written'): sql_count += 1
                    except Exception as e:
                        Console.warn(f"Failed to write regime_stats.csv: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                
                # 3. Contribution Now (current sensor importance)
                contrib_now_df = self._generate_contrib_now(scores_df)
                result = self.write_dataframe(contrib_now_df, tables_dir / "contrib_now.csv",
                                            sql_table="ACM_ContributionCurrent" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                # 4. Contribution Timeline (historical contributions)
                contrib_timeline_df = self._generate_contrib_timeline(scores_df)
                result = self.write_dataframe(contrib_timeline_df, tables_dir / "contrib_timeline.csv",
                                            sql_table="ACM_ContributionTimeline" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                # 5. Drift Series (CUSUM/drift detection)
                if 'drift_z' in scores_df.columns or 'cusum_z' in scores_df.columns:
                    drift_df = self._generate_drift_series(scores_df)
                    result = self.write_dataframe(drift_df, tables_dir / "drift_series.csv",
                                                sql_table="ACM_DriftSeries" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1
                
                # 6. Threshold Crossings (anomaly events)
                threshold_df = self._generate_threshold_crossings(scores_df)
                result = self.write_dataframe(threshold_df, tables_dir / "threshold_crossings.csv",
                                            sql_table="ACM_ThresholdCrossings" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                # 7. Since When (alert start times)
                since_when_df = self._generate_since_when(scores_df)
                result = self.write_dataframe(since_when_df, tables_dir / "since_when.csv",
                                            sql_table="ACM_SinceWhen" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                # 8. Sensor Rank Now (current importance ranking)
                sensor_rank_df = self._generate_sensor_rank_now(scores_df)
                result = self.write_dataframe(sensor_rank_df, tables_dir / "sensor_rank_now.csv",
                                            sql_table="ACM_SensorRanking" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                # 9. Health Histogram (distribution)
                health_hist_df = self._generate_health_histogram(scores_df)
                result = self.write_dataframe(health_hist_df, tables_dir / "health_hist.csv",
                                            sql_table="ACM_HealthHistogram" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1

                # Daily fused profile (hour x weekday) for simplified dashboards
                try:
                    daily_profile_df = self._generate_daily_fused_profile(scores_df)
                    if not daily_profile_df.empty:
                        result = self.write_dataframe(
                            daily_profile_df,
                            tables_dir / "daily_fused_profile.csv",
                            sql_table="ACM_DailyFusedProfile" if force_sql else None,
                            sql_columns={"Count": "RecordCount"},
                            add_created_at=True
                        )
                        table_count += 1
                        if result.get('sql_written'): sql_count += 1
                except Exception as e:
                    Console.warn(f"Failed to write daily_fused_profile.csv: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                
                # 11. Alert Age (duration tracking)
                alert_age_df = self._generate_alert_age(scores_df)
                result = self.write_dataframe(alert_age_df, tables_dir / "alert_age.csv",
                                            sql_table="ACM_AlertAge" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                # 12. Regime Stability (churn metrics)
                if 'regime_label' in scores_df.columns:
                    regime_stab_df = self._generate_regime_stability(scores_df)
                    result = self.write_dataframe(regime_stab_df, tables_dir / "regime_stability.csv",
                                                sql_table="ACM_RegimeStability" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1
                
                # 13-15. Defect-Focused Tables (operator-friendly)
                defect_summary_df = self._generate_defect_summary(scores_df, episodes_df)
                result = self.write_dataframe(defect_summary_df, tables_dir / "defect_summary.csv",
                                            sql_table="ACM_DefectSummary" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                defect_timeline_df = self._generate_defect_timeline(scores_df)
                result = self.write_dataframe(defect_timeline_df, tables_dir / "defect_timeline.csv",
                                            sql_table="ACM_DefectTimeline" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                sensor_defects_df = self._generate_sensor_defects(scores_df)
                result = self.write_dataframe(sensor_defects_df, tables_dir / "sensor_defects.csv",
                                            sql_table="ACM_SensorDefects" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                # 16-17. Health Zone Analysis
                health_zone_df = self._generate_health_zone_by_period(scores_df)
                result = self.write_dataframe(health_zone_df, tables_dir / "health_zone_by_period.csv",
                                            sql_table="ACM_HealthZoneByPeriod" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1
                
                sensor_anomaly_df = self._generate_sensor_anomaly_by_period(scores_df)
                result = self.write_dataframe(sensor_anomaly_df, tables_dir / "sensor_anomaly_by_period.csv",
                                            sql_table="ACM_SensorAnomalyByPeriod" if force_sql else None,
                                            add_created_at=True)
                table_count += 1
                if result.get('sql_written'): sql_count += 1

                sensor_ready = sensor_zscores is not None and sensor_values is not None
                if sensor_ready:
                    warn_threshold = float((cfg.get('regimes', {}) or {}).get('health', {}).get('fused_warn_z', 1.5) or 1.5)
                    alert_threshold = float((cfg.get('regimes', {}) or {}).get('health', {}).get('fused_alert_z', 3.0) or 3.0)
                    top_n = int((cfg.get('output', {}) or {}).get('sensor_hotspot_top_n', 25))

                    sensor_hotspots_df = self._generate_sensor_hotspots_table(
                        sensor_zscores if sensor_zscores is not None else pd.DataFrame(),
                        sensor_values if sensor_values is not None else pd.DataFrame(),
                        sensor_train_mean,
                        sensor_train_std,
                        warn_threshold,
                        alert_threshold,
                        top_n
                    )
                    result = self.write_dataframe(
                        sensor_hotspots_df,
                        tables_dir / "sensor_hotspots.csv",
                        sql_table="ACM_SensorHotspots" if force_sql else None,
                        non_numeric_cols={"SensorName"},
                        add_created_at=True
                    )
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1

                    # Normalized raw sensor timeline with anomaly flags and episode overlays
                    # P6.2: Reduced defaults for SQL performance (5 sensors, every 10th timestamp)
                    try:
                        norm_top_n = int((cfg.get('output', {}) or {}).get('sensor_normalized_top_n', 5) or 5)
                        norm_downsample = int((cfg.get('output', {}) or {}).get('sensor_normalized_downsample', 10) or 10)
                        sensor_norm_df = self._generate_sensor_normalized_ts(
                            sensor_values=sensor_values if sensor_values is not None else pd.DataFrame(),
                            sensor_train_mean=sensor_train_mean,
                            sensor_train_std=sensor_train_std,
                            sensor_zscores=sensor_zscores if sensor_zscores is not None else pd.DataFrame(),
                            episodes_df=episodes_df,
                            warn_z=warn_threshold,
                            alert_z=alert_threshold,
                            top_n=norm_top_n,
                            downsample_factor=norm_downsample
                        )
                        result = self.write_dataframe(
                            sensor_norm_df,
                            tables_dir / "sensor_normalized_ts.csv",
                            sql_table="ACM_SensorNormalized_TS" if force_sql else None,
                            non_numeric_cols={"SensorName", "AnomalyLevel"},
                            add_created_at=True
                        )
                        table_count += 1
                        if result.get('sql_written'): sql_count += 1
                    except Exception as e:
                        Console.warn(f"Sensor normalized timeline skipped: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])

                    sensor_timeline_df = self._generate_sensor_hotspot_timeline(
                        sensor_zscores if sensor_zscores is not None else pd.DataFrame(),
                        sensor_values,
                        warn_threshold,
                        alert_threshold,
                        top_k=3
                    )
                    result = self.write_dataframe(
                        sensor_timeline_df,
                        tables_dir / "sensor_hotspot_timeline.csv",
                        sql_table="ACM_SensorHotspotTimeline" if force_sql else None,
                        non_numeric_cols={"SensorName", "Level"},
                        add_created_at=True
                    )
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1
                
                # 18. Already generated: data_quality.csv
                
                # 19. Drift Events
                if 'drift_z' in scores_df.columns or 'cusum_z' in scores_df.columns:
                    drift_events_df = self._generate_drift_events(scores_df)
                    result = self.write_dataframe(drift_events_df, tables_dir / "drift_events.csv",
                                                sql_table="ACM_DriftEvents" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1
                
                # 24. Culprit History (episode-based sensor culprit tracking)
                if episodes_df is not None and not episodes_df.empty:
                    culprit_hist_df = self._generate_culprit_history(scores_df, episodes_df)
                    result = self.write_dataframe(culprit_hist_df, tables_dir / "culprit_history.csv",
                                                sql_table="ACM_CulpritHistory" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1

                    # 25. Episode Metrics (episode statistical summary)
                    episode_metrics_df = self._generate_episode_metrics(episodes_df)
                    result = self.write_dataframe(episode_metrics_df, tables_dir / "episode_metrics.csv",
                                                sql_table="ACM_EpisodeMetrics" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1

                    # 25b. Episode Diagnostics (per-episode troubleshooting metrics)
                    episode_diagnostics_df = self._generate_episode_diagnostics(episodes_df, scores_df)
                    result = self.write_dataframe(episode_diagnostics_df, tables_dir / "episode_diagnostics.csv",
                                                sql_table="ACM_EpisodeDiagnostics" if force_sql else None,
                                                add_created_at=True)
                    table_count += 1
                    if result.get('sql_written'): sql_count += 1

                    # 25c. Episodes QC (run-level quality summary)
                    try:
                        episodes_qc_df = self._generate_episodes_qc(scores_df, episodes_df)
                        # SCHEMA-FIX: Write to ACM_EpisodesQC (run-level summary), not ACM_Episodes (individual episodes)
                        result = self.write_dataframe(episodes_qc_df, tables_dir / "episodes_qc.csv",
                                             sql_table="ACM_EpisodesQC" if force_sql else None,
                                             add_created_at=False)
                        table_count += 1
                        if result.get('sql_written'): sql_count += 1
                    except Exception as e:
                        Console.warn(f"Failed to write episodes_qc.csv: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                
                # 26. Data Quality (already written separately, but ensure SQL write)
                if force_sql:
                    try:
                        data_quality_path = tables_dir / "data_quality.csv"
                        if data_quality_path.exists():
                            dq_df = pd.read_csv(data_quality_path)
                            # Ensure required SQL columns exist with safe defaults
                            if 'CheckName' not in dq_df.columns:
                                dq_df['CheckName'] = 'NullsBySensor'
                            # Provide a likely-cased Sensor column if only lowercase exists
                            if 'sensor' in dq_df.columns and 'Sensor' not in dq_df.columns:
                                dq_df['Sensor'] = dq_df['sensor']
                            # Derive a CheckResult when missing based on null percentages and notes
                            if 'CheckResult' not in dq_df.columns:
                                def _derive_result(row):
                                    try:
                                        notes = str(row.get('notes', '') or '').lower()
                                        tr_pct = float(row.get('train_null_pct', 0) or 0)
                                        sc_pct = float(row.get('score_null_pct', 0) or 0)
                                        if 'all_nulls_train' in notes or 'all_nulls_score' in notes:
                                            return 'FAIL'
                                        if max(tr_pct, sc_pct) >= 80:
                                            return 'FAIL'
                                        if max(tr_pct, sc_pct) >= 10:
                                            return 'CAUTION'
                                        if 'low_variance_train' in notes:
                                            return 'CAUTION'
                                        return 'OK'
                                    except Exception:
                                        return 'OK'
                                dq_df['CheckResult'] = dq_df.apply(_derive_result, axis=1)
                            
                            # Add required SQL columns that may be missing
                            if 'RunID' not in dq_df.columns:
                                dq_df['RunID'] = self.current_run_id
                            if 'EquipID' not in dq_df.columns:
                                dq_df['EquipID'] = self.equip_id
                            
                            # Rename 'sensor' to 'sensor' (already correct in table schema)
                            # Select only the columns the table expects
                            expected_cols = [
                                'sensor', 'train_count', 'train_nulls', 'train_null_pct', 'train_std',
                                'train_longest_gap', 'train_flatline_span', 'train_min_ts', 'train_max_ts',
                                'score_count', 'score_nulls', 'score_null_pct', 'score_std',
                                'score_longest_gap', 'score_flatline_span', 'score_min_ts', 'score_max_ts',
                                'interp_method', 'sampling_secs', 'notes', 'RunID', 'EquipID', 'CheckName', 'CheckResult'
                            ]
                            # Keep only columns that exist in both dq_df and expected_cols
                            cols_to_keep = [c for c in expected_cols if c in dq_df.columns]
                            dq_df = dq_df[cols_to_keep]
                            
                            result = self.write_dataframe(
                                dq_df,
                                data_quality_path,
                                sql_table="ACM_DataQuality",
                                add_created_at=True
                            )
                            if result.get('sql_written'): sql_count += 1
                    except Exception as e:
                        Console.warn(f"Failed to write data_quality to SQL: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                
                Console.info(f"Generated {table_count} comprehensive analytics tables", component="ANALYTICS")
                Console.info(f"Written {sql_count} tables to SQL database", component="ANALYTICS")
                return {"csv_tables": table_count, "sql_tables": sql_count}
                
            except Exception as e:
                Console.warn(f"Comprehensive table generation failed: {e}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id, error_type=type(e).__name__, error=str(e)[:200])
                import traceback
                Console.warn(f"Error traceback: {traceback.format_exc()}", component="ANALYTICS", equip_id=self.equip_id, run_id=self.run_id)
                return {"csv_tables": table_count, "sql_tables": sql_count}

    def _generate_regime_stability(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime stability metrics (transition frequency, dwell time).
        
        Returns DataFrame with stability indicators.
        """
        if scores_df.empty or 'regime_label' not in scores_df.columns:
            return pd.DataFrame(columns=['RunID', 'EquipID', 'Period', 'TransitionCount', 'AvgDwellTime', 'StabilityScore'])
        
        # Ensure timestamp
        if 'Timestamp' in scores_df.columns:
            ts_col = pd.to_datetime(scores_df['Timestamp'], errors='coerce')
        else:
            try:
                ts_col = pd.to_datetime(scores_df.index, errors='coerce')
            except Exception:
                return pd.DataFrame(columns=['RunID', 'EquipID', 'Period', 'TransitionCount', 'AvgDwellTime', 'StabilityScore'])
        
        df = scores_df.copy()
        df['Timestamp'] = ts_col
        df['regime'] = df['regime_label'].fillna(-1)
        
        # Count transitions
        transitions = (df['regime'] != df['regime'].shift()).sum() - 1  # -1 to exclude first row
        
        # Calculate average dwell time (hours)
        if len(df) > 1:
            total_time = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 3600.0
            avg_dwell = total_time / max(1, transitions)
        else:
            avg_dwell = 0.0
        
        # Stability score (inverse of transition frequency)
        stability = 100.0 / (1.0 + transitions / max(1, len(df)))
        
        # Include MetricName to satisfy NOT NULL column requirement in ACM_RegimeStability
        result = pd.DataFrame({
            'RunID': [self.run_id],
            'EquipID': [int(self.equip_id or 0)],
            'MetricName': ['RegimeStability'],
            # Use StabilityScore as MetricValue to satisfy NOT NULL schema and be meaningful
            'MetricValue': [float(stability)],
            'Period': ['full_window'],
            'TransitionCount': [transitions],
            'AvgDwellTime': [avg_dwell],
            'StabilityScore': [stability]
        })
        
        return result
    
    
    def _generate_episode_severity_mapping(self, episodes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate episodes severity mapping JSON with counts and color palette.
        
        OUT-28: Provides severity level metadata for downstream consumers
        to understand severity distribution and rendering.
        
        Returns:
            Dict with severity levels, color mapping, and counts
        """
        # CHART-12: Use centralized severity color palette
        severity_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        color_map = SEVERITY_COLORS  # Centralized constant from top of file
        
        # Count episodes by severity
        counts = {}
        total_episodes = len(episodes_df)
        
        if 'severity' in episodes_df.columns:
            severity_counts = episodes_df['severity'].str.upper().value_counts().to_dict()
            for level in severity_levels:
                counts[level] = severity_counts.get(level, 0)
        else:
            # No severity column - all defaults
            for level in severity_levels:
                counts[level] = 0
        
        # Validate counts sum
        count_sum = sum(counts.values())
        validation_status = "VALID" if count_sum == total_episodes else "MISMATCH"
        
        return {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "severity_levels": severity_levels,
            "color_map": color_map,
            "counts": counts,
            "total_episodes": total_episodes,
            "validation_status": validation_status,
            "count_sum": count_sum
        }

    def _generate_health_timeline(self, scores_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate enhanced health timeline with smoothing and quality flags.
        
        Implements EMA smoothing + rate limiting to prevent unrealistic health jumps
        caused by sensor noise, missing data, or coldstart artifacts.
        
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
        
        ts_values = normalize_timestamp_series(scores_df.index).to_list()
        return pd.DataFrame({
            'Timestamp': ts_values,
            'HealthIndex': smoothed_health.round(2).to_list(),
            'RawHealthIndex': raw_health.round(2).to_list(),
            'QualityFlag': quality_flag.astype(str).to_list(),
            'HealthZone': zones.astype(str).to_list(),
            'FusedZ': scores_df['fused'].round(4).to_list()
        })
    
    def _generate_regime_timeline(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate regime timeline with confidence."""
        regimes = pd.to_numeric(scores_df['regime_label'], errors='coerce').astype('Int64')
        ts_values = normalize_timestamp_series(scores_df.index).to_list()
        return pd.DataFrame({
            'Timestamp': ts_values,
            'RegimeLabel': regimes.to_list(),
            'RegimeState': (scores_df['regime_state'].astype(str).to_list() if 'regime_state' in scores_df.columns else [str('unknown')] * len(scores_df))
        })
    
    def _generate_contrib_now(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate current sensor contributions."""
        # Get latest z-scores for detectors
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        if not detector_cols:
            return pd.DataFrame({'DetectorType': ['No detectors'], 'ContributionPct': [0.0], 'ZScore': [0.0]})
        
        latest_scores = scores_df[detector_cols].iloc[-1].abs()
        total = latest_scores.sum()
        if total == 0:
            contributions = pd.Series([100.0 / len(detector_cols)] * len(detector_cols), index=detector_cols)
        else:
            contributions = (latest_scores / total * 100).round(2)
        
        df = pd.DataFrame({
            'DetectorType': contributions.index,
            'ContributionPct': contributions.values,
            'ZScore': latest_scores.values
        }).sort_values('ContributionPct', ascending=False)
        # SQL-safe human-readable detector labels for dashboards
        df['DetectorType'] = df['DetectorType'].apply(lambda c: get_detector_label(str(c), sql_safe=True))
        return df
    
    def _generate_contrib_timeline(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate historical sensor contributions over time."""
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c not in ('fused', 'fused_z')]
        if not detector_cols:
            return pd.DataFrame({'Timestamp': [], 'DetectorType': [], 'ContributionPct': []})

        # dynamic sampling to ~target points
        n = len(scores_df)
        step = max(1, n // AnalyticsConstants.TARGET_SAMPLING_POINTS)
        sampled = scores_df[detector_cols].iloc[::step].abs()

        totals = sampled.sum(axis=1)
        valid = totals > 0
        normalized = sampled[valid].div(totals[valid], axis=0) * 100

        tmp = normalized.reset_index()
        idx_col = tmp.columns[0]
        long_df = tmp.melt(id_vars=idx_col, var_name='DetectorType', value_name='ContributionPct')
        long_df.rename(columns={idx_col: 'Timestamp'}, inplace=True)
        long_df['Timestamp'] = normalize_timestamp_series(long_df['Timestamp'])
        long_df['ContributionPct'] = long_df['ContributionPct'].round(2)
        # SQL-safe human-readable detector labels
        long_df['DetectorType'] = long_df['DetectorType'].apply(lambda c: get_detector_label(str(c), sql_safe=True))
        return long_df[['Timestamp', 'DetectorType', 'ContributionPct']]
    
    def _generate_defect_summary(self, scores_df: pd.DataFrame, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Generate executive defect summary."""
        health_index = _health_index(scores_df['fused'])
        current_health = health_index.iloc[-1]
        avg_health = health_index.mean()
        min_health = health_index.min()
        
        # Determine status
        if current_health >= 85:
            status = "HEALTHY"
            severity = "LOW"
        elif current_health >= 70:
            status = "CAUTION"
            severity = "MEDIUM"
        else:
            status = "ALERT"
            severity = "HIGH"
        
        # Count episodes
        episode_count = len(episodes_df) if episodes_df is not None and not episodes_df.empty else 0
        
        # Find worst sensor
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        if detector_cols:
            latest_scores = scores_df[detector_cols].iloc[-1].abs()
            worst_sensor = str(latest_scores.idxmax()).replace('_z', '')
        else:
            worst_sensor = "Unknown"
        
        # Count health zones
        zones = pd.cut(health_index, bins=[0, 70, 85, 100], labels=['ALERT', 'WATCH', 'GOOD'])
        zone_counts = zones.value_counts()
        
        return pd.DataFrame({
            'Status': [status],
            'Severity': [severity], 
            'CurrentHealth': [round(current_health, 1)],
            'AvgHealth': [round(avg_health, 1)],
            'MinHealth': [round(min_health, 1)],
            'EpisodeCount': [episode_count],
            'WorstSensor': [worst_sensor],
            'GoodCount': [zone_counts.get('GOOD', 0)],
            'WatchCount': [zone_counts.get('WATCH', 0)],
            'AlertCount': [zone_counts.get('ALERT', 0)]
        })
    
    def _generate_defect_timeline(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate defect event timeline."""
        health_index = _health_index(scores_df['fused'])
        zones = pd.cut(health_index, bins=[0, 70, 85, 100], labels=['ALERT', 'WATCH', 'GOOD'])
        
        # Find peaks above threshold
        peaks = zones != zones.shift()
        peak_events = []
        
        zones_shifted = zones.shift()
        
        for idx in scores_df[peaks].index:
            from_zone_val = zones_shifted.loc[idx]
            # Handle scalar value properly
            try:
                from_zone = str(from_zone_val) if not pd.isna(from_zone_val) else 'START'
            except (ValueError, TypeError):
                from_zone = 'START'

            # Use tz-naive UTC datetime for SQL compatibility
            ts_naive = normalize_timestamp_scalar(idx)
            to_zone = str(zones.loc[idx])
            # FusedZ at this timestamp if available
            fused_val = None
            try:
                if 'fused' in scores_df.columns:
                    val = scores_df.loc[idx, 'fused']
                    fused_val = round(float(cast(Any, val)), 4) if (val is not None and pd.notna(val)) else 0.0
            except Exception:
                fused_val = 0.0

            peak_events.append({
                'Timestamp': ts_naive,
                'EventType': 'ZONE_CHANGE',
                'FromZone': from_zone,
                'ToZone': to_zone,
                'HealthZone': to_zone,
                'HealthAtEvent': round(health_index.loc[idx], 2),
                'HealthIndex': round(health_index.loc[idx], 2),
                'FusedZ': fused_val if fused_val is not None else 0.0
            })
        
        return pd.DataFrame(peak_events)
    
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
    
    def _generate_health_zone_by_period(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate health zone distribution by time period (long format).
        Emits one row per period per zone with required HealthZone.
        """
        health_index = _health_index(scores_df['fused'])
        zones = pd.cut(health_index, bins=[0, 70, 85, 100], labels=['ALERT', 'WATCH', 'GOOD'])

        idx_dates = pd.to_datetime(scores_df.index, errors='coerce')
        daily_zones = pd.DataFrame({'date': idx_dates.to_series().dt.date, 'zone': zones})
        zone_summary = daily_zones.groupby(['date', 'zone'], observed=False).size().unstack(fill_value=0)
        totals = zone_summary.sum(axis=1)

        rows = []
        for date, counts in zone_summary.iterrows():
            period_start = pd.Timestamp(str(date)).to_pydatetime()
            total_points = int(cast(Any, totals.get(date, 0)))
            for hz in ['GOOD', 'WATCH', 'ALERT']:
                cnt = int(counts.get(hz, 0))
                pct = (cnt / total_points * 100.0) if total_points > 0 else 0.0
                rows.append({
                    'PeriodStart': period_start,
                    'PeriodType': 'DAY',
                    'HealthZone': hz,
                    'ZonePct': round(pct, 1),
                    'ZoneCount': cnt,
                    'TotalPoints': total_points,
                    'Date': str(date),  # CSV readability; SQL will ignore if no column
                })
        return pd.DataFrame(rows)
    
    def _generate_sensor_anomaly_by_period(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate per-sensor anomaly rates by time period."""
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        results = []
        
        # Group by day
        daily_data = scores_df.groupby(pd.to_datetime(scores_df.index, errors='coerce').to_series().dt.date)
        
        for date, day_data in daily_data:
            for detector in detector_cols:
                values = day_data[detector].abs()
                anomaly_rate = (values > 2.0).mean() * 100
                # PeriodStart: start of the day (tz-naive) to satisfy SQL NOT NULL
                period_start = pd.Timestamp(str(date)).to_pydatetime()
                
                results.append({
                    'Date': str(date),  # For CSV readability
                    'PeriodStart': period_start,
                    'PeriodType': 'DAY',
                    'PeriodSeconds': 86400,
                    'DetectorType': detector.replace('_z', ''),
                    'AnomalyRatePct': round(anomaly_rate, 2),
                    'MaxZ': round(values.max(), 4),
                    'AvgZ': round(values.mean(), 4),
                    'Points': len(values)
                })
        
        return pd.DataFrame(results)

    def _generate_sensor_normalized_ts(
        self,
        sensor_values: pd.DataFrame,
        sensor_train_mean: Optional[pd.Series],
        sensor_train_std: Optional[pd.Series],
        sensor_zscores: pd.DataFrame,
        episodes_df: Optional[pd.DataFrame],
        warn_z: float,
        alert_z: float,
        top_n: int = 5,  # P6.2: Reduced from 20 to 5 for 75% row reduction
        downsample_factor: int = 1,  # P6.2: Keep every Nth timestamp (1=all, 10=10%)
    ) -> pd.DataFrame:
        """Build a long-form normalized sensor timeline with anomalies and episode overlays.

        PERFORMANCE OPTIMIZATION (v11 P6.2):
        - Default top_n reduced from 20 to 5 (75% row reduction)
        - Optional downsample_factor to keep every Nth timestamp
        - Combined: 5 sensors × 56 timestamps = 280 rows instead of 11,240

        Columns: Timestamp, SensorName, NormValue, ZScore, AnomalyLevel, EpisodeActive
        """
        if sensor_values is None or sensor_values.empty:
            return pd.DataFrame({
                'Timestamp': [], 'SensorName': [], 'NormValue': [], 'ZScore': [],
                'AnomalyLevel': [], 'EpisodeActive': []
            })

        # Determine sensors to include based on peak |z|
        try:
            abs_z = sensor_zscores.abs() if sensor_zscores is not None else None
            if abs_z is not None and not abs_z.empty:
                ranked = abs_z.max().sort_values(ascending=False)
                sensors = ranked.index[:max(1, int(top_n))].tolist()
            else:
                sensors = list(sensor_values.columns)
        except Exception:
            sensors = list(sensor_values.columns)

        values = sensor_values[sensors].copy()
        
        # P6.2: Apply timestamp downsampling if requested (keeps every Nth row)
        if downsample_factor > 1:
            values = values.iloc[::downsample_factor]

        # Compute normalized values using training mean/std when present; otherwise fallback to zscores or per-column std
        if isinstance(sensor_train_mean, pd.Series) and isinstance(sensor_train_std, pd.Series):
            mean = sensor_train_mean.reindex(values.columns)
            std = sensor_train_std.reindex(values.columns).replace(0.0, np.nan)
            norm = (values - mean) / std
        elif sensor_zscores is not None and not sensor_zscores.empty:
            norm = sensor_zscores[sensors].copy()
        else:
            col_mean = values.mean(axis=0)
            col_std = values.std(axis=0).replace(0.0, np.nan)
            norm = (values - col_mean) / col_std

        norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ZScore source for anomaly marking
        if sensor_zscores is not None and not sensor_zscores.empty:
            z_for_mark = sensor_zscores[sensors].copy()
        else:
            z_for_mark = norm.copy()

        # Build episode active mask per timestamp
        episode_active = pd.Series(0, index=values.index)
        if episodes_df is not None and not episodes_df.empty and {'start_ts', 'end_ts'}.issubset(set(episodes_df.columns)):
            for _, ep in episodes_df.iterrows():
                try:
                    s = pd.to_datetime(ep['start_ts'], errors='coerce')
                    e = pd.to_datetime(ep['end_ts'], errors='coerce')
                    if pd.notna(s) and pd.notna(e) and e >= s:
                        mask = (values.index >= s) & (values.index <= e)
                        if mask.any():
                            episode_active.loc[mask] = 1
                except Exception:
                    continue

        # Melt into long format
        df_norm = norm.copy()
        df_norm.index.name = 'Timestamp'
        long_norm = df_norm.reset_index().melt(id_vars=['Timestamp'], var_name='SensorName', value_name='NormValue')

        df_z = z_for_mark.copy()
        df_z.index.name = 'Timestamp'
        long_z = df_z.reset_index().melt(id_vars=['Timestamp'], var_name='SensorName', value_name='ZScore')

        out = long_norm.merge(long_z, on=['Timestamp', 'SensorName'], how='left')

        # Map anomaly level
        abs_z_vals = out['ZScore'].abs()
        out['AnomalyLevel'] = np.where(abs_z_vals >= float(alert_z), 'ALERT',
                                np.where(abs_z_vals >= float(warn_z), 'WARN', 'GOOD'))

        # Attach episode flag
        ep_flag = episode_active.reset_index()
        ep_flag.columns = ['Timestamp', 'EpisodeActive']
        out = out.merge(ep_flag, on='Timestamp', how='left')
        out['EpisodeActive'] = out['EpisodeActive'].fillna(0).astype(int)

        # Convert timestamps to naive local policy
        out['Timestamp'] = out['Timestamp'].apply(normalize_timestamp_scalar)

        # Order columns
        out = out[['Timestamp', 'SensorName', 'NormValue', 'ZScore', 'AnomalyLevel', 'EpisodeActive']]
        return out

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

    def _generate_sensor_hotspot_timeline(
        self,
        sensor_zscores: pd.DataFrame,
        sensor_values: Optional[pd.DataFrame],
        warn_z: float,
        alert_z: float,
        top_k: int = 3
    ) -> pd.DataFrame:
        """Produce timestamped rows for the strongest sensor deviations."""
        if sensor_zscores is None or sensor_zscores.empty:
            return pd.DataFrame({
                'Timestamp': [], 'SensorName': [], 'Rank': [], 'AbsZ': [],
                'SignedZ': [], 'Value': [], 'Level': []
            })

        records: List[Dict[str, Any]] = []
        abs_df = sensor_zscores.abs()
        for ts, row in abs_df.iterrows():
            top = row.nlargest(top_k)
            for rank, (sensor, abs_val) in enumerate(top.items(), start=1):
                if pd.isna(abs_val) or abs_val < warn_z:
                    continue
                ts_key = (pd.Timestamp(str(cast(Any, ts))) if not isinstance(ts, pd.Timestamp) else ts)
                sensor_key = str(sensor)
                signed_z = sensor_zscores.at[ts_key, sensor_key]
                value = None
                if sensor_values is not None and sensor in sensor_values.columns:
                    try:
                        value = sensor_values.at[ts_key, sensor_key]
                    except Exception:
                        value = sensor_values[sensor].reindex([ts]).iloc[-1]
                level = 'ALERT' if abs_val >= alert_z else 'WARN'
                records.append({
                    'Timestamp': normalize_timestamp_scalar(ts),
                    'SensorName': sensor,
                    'Rank': rank,
                    'AbsZ': round(float(abs_val), 4),
                    'SignedZ': (round(float(cast(Any, signed_z)), 4) if signed_z is not None and pd.notna(signed_z) else None),
                    'Value': (float(cast(Any, value)) if value is not None and pd.notna(value) else None),
                    'Level': level
                })

        if not records:
            return pd.DataFrame({
                'Timestamp': [], 'SensorName': [], 'Rank': [], 'AbsZ': [],
                'SignedZ': [], 'Value': [], 'Level': []
            })

        df = pd.DataFrame(records).sort_values(['Timestamp', 'Rank']).reset_index(drop=True)
        return df
    
    def _generate_detector_correlation(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate detector correlation matrix with SQL schema columns."""
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        if len(detector_cols) < 2:
            return pd.DataFrame({'DetectorA': [], 'DetectorB': [], 'PearsonR': []})

        def _pair_hint(label_a: str, label_b: str) -> str:
            labels = (label_a + " " + label_b).lower()
            has_baseline = "baseline" in labels or "omr" in labels
            has_corr = "correlation" in labels or "pca" in labels or "density" in labels or "isolationforest" in labels
            has_spike = "time-series" in labels or "ar1" in labels
            has_distance = "distance" in labels or "mahal" in labels
            if has_baseline and has_corr:
                return "Health baseline moving with a pattern change"
            if has_baseline and has_spike:
                return "Health baseline tracking repeated spikes"
            if has_corr and has_distance:
                return "Regime/cluster shift across many sensors"
            if has_corr and has_spike:
                return "Transient spikes align with pattern change"
            if has_spike:
                return "Temporal spikes seen by both detectors"
            if has_baseline:
                return "Overall health shifting with another detector"
            return "Detectors reacting together; check shared cause"
        
        correlations = []
        for i, det_a in enumerate(detector_cols):
            for det_b in detector_cols[i+1:]:
                sa = pd.to_numeric(scores_df[det_a], errors='coerce')
                sb = pd.to_numeric(scores_df[det_b], errors='coerce')
                std_a = float(sa.std(skipna=True))
                std_b = float(sb.std(skipna=True))
                if std_a == 0.0 or std_b == 0.0:
                    r = 0.0
                else:
                    r = sa.corr(sb)
                    r = 0.0 if pd.isna(r) else float(round(r, 4))
                correlations.append({
                    'DetectorA': get_detector_label(str(det_a), sql_safe=True),
                    'DetectorB': get_detector_label(str(det_b), sql_safe=True),
                    'PearsonR': r,
                    'PairLabel': f"{get_detector_label(str(det_a), sql_safe=True)} <-> {get_detector_label(str(det_b), sql_safe=True)}",
                    'DisturbanceHint': _pair_hint(get_detector_label(str(det_a), sql_safe=True), get_detector_label(str(det_b), sql_safe=True))
                })
        
        return pd.DataFrame(correlations)
    
    def _generate_calibration_summary(self, scores_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """Generate detector calibration and saturation summary with SQL schema columns."""
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        calibration_data = []
        
        clip_z = cfg.get('thresholds', {}).get('self_tune', {}).get('clip_z', 30.0)
        # ANA-07: Get mhal condition number from diagnostics
        mhal_cond_num = cfg.get("_diagnostics", {}).get("mhal_cond_num", None)
        
        for detector in detector_cols:
            values = scores_df[detector].abs()
            saturation_pct = (values >= clip_z).mean() * 100
            
            row = {
                'DetectorType': get_detector_label(str(detector), sql_safe=True),
                'MeanZ': round(values.mean(), 4),
                'StdZ': round(values.std(), 4),
                'P95Z': round(values.quantile(0.95), 4),
                'P99Z': round(values.quantile(0.99), 4),
                'ClipZ': clip_z,
                'SaturationPct': round(saturation_pct, 2)
            }
            
            # ANA-07: Add mhal_cond_num column (only populated for mhal_z row)
            # Ensure extreme values don't get written (already handled by output_manager SQL cleaning)
            if detector == 'mhal_z' and mhal_cond_num is not None:
                # Check for extreme values before rounding
                if abs(mhal_cond_num) > 1e100:
                    row['MahalCondNum'] = None  # Extreme value, set to NULL
                else:
                    row['MahalCondNum'] = round(mhal_cond_num, 2)
            else:
                row['MahalCondNum'] = None
            
            calibration_data.append(row)
        
        return pd.DataFrame(calibration_data)
    
    def _generate_regime_transition_matrix(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate regime transition counts and probabilities."""
        regime_col = scores_df['regime_label']
        transitions = []
        
        for i in range(1, len(regime_col)):
            from_regime = regime_col.iloc[i-1]
            to_regime = regime_col.iloc[i]
            if from_regime != to_regime:  # Exclude self-transitions
                transitions.append({'FromLabel': from_regime, 'ToLabel': to_regime})
        
        if not transitions:
            return pd.DataFrame({'FromLabel': [], 'ToLabel': [], 'Count': [], 'Prob': []})
        
        # Count transitions
        transition_df = pd.DataFrame(transitions)
        transition_counts = transition_df.groupby(['FromLabel', 'ToLabel']).size().reset_index(name='Count')
        
        # Calculate probabilities
        total_transitions = transition_counts['Count'].sum()
        transition_counts['Prob'] = (transition_counts['Count'] / total_transitions).round(4)
        
        return transition_counts
    
    def _generate_regime_dwell_stats(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate regime dwell time statistics using real timestamps."""
        rl = pd.Series(scores_df['regime_label'])
        ts = pd.to_datetime(scores_df.index)
        group = rl.ne(rl.shift()).cumsum()
        runs = (pd.DataFrame({'regime': rl, 'ts': ts})
                  .groupby(group)
                  .agg(regime=('regime','first'), start=('ts','first'), end=('ts','last')))
        runs['seconds'] = (runs['end'] - runs['start']).dt.total_seconds().clip(lower=0)
        out = (runs.groupby('regime')['seconds']
                  .agg(runs='count',
                       mean_seconds='mean',
                       median_seconds='median',
                       min_seconds='min',
                       max_seconds='max')
                  .reset_index())
        
        # Rename columns for SQL compatibility
        out = out.rename(columns={
            'regime': 'RegimeLabel',
            'runs': 'Runs',
            'mean_seconds': 'MeanSeconds',
            'median_seconds': 'MedianSeconds',
            'min_seconds': 'MinSeconds',
            'max_seconds': 'MaxSeconds'
        })
        
        return out
    
    def _generate_drift_events(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate significant drift events from CUSUM series."""
        drift_col = 'cusum_z' if 'cusum_z' in scores_df.columns else 'drift_z'
        if drift_col not in scores_df.columns:
            return pd.DataFrame({'Timestamp': [], 'Value': [], 'SegmentStart': [], 'SegmentEnd': []})
        
        drift_values = scores_df[drift_col].abs()
        threshold = 3.0  # Threshold for significant drift events
        
        # Find peaks above threshold
        peaks = drift_values > threshold
        peak_events = []
        
        if peaks.any():
            # Find continuous segments above threshold
            in_segment = False
            segment_start = None
            
            for idx in peaks.index:
                is_peak = peaks.loc[idx]
                if is_peak and not in_segment:
                    segment_start = idx
                    in_segment = True
                elif not is_peak and in_segment:
                    peak_events.append({
                        'Timestamp': normalize_timestamp_scalar(idx),
                        'Value': round(drift_values.loc[segment_start:idx].max(), 4),
                        'SegmentStart': normalize_timestamp_scalar(segment_start),
                        'SegmentEnd': normalize_timestamp_scalar(idx)
                    })
                    in_segment = False
            
            # Handle final segment if it ends with a peak
            if in_segment:
                final_idx = scores_df.index[-1]
                peak_events.append({
                    'Timestamp': normalize_timestamp_scalar(final_idx),
                    'Value': round(drift_values.loc[segment_start:final_idx].max(), 4),
                    'SegmentStart': normalize_timestamp_scalar(segment_start),
                    'SegmentEnd': normalize_timestamp_scalar(final_idx)
                })
        
        return pd.DataFrame(peak_events)
    
    def _generate_drift_series(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate drift series timeline."""
        drift_col = 'cusum_z' if 'cusum_z' in scores_df.columns else 'drift_z'
        if drift_col not in scores_df.columns:
            return pd.DataFrame({'Timestamp': [], 'DriftValue': []})
        ts_values = normalize_timestamp_series(scores_df.index).to_list()
        drift_values = scores_df[drift_col].round(4).to_list()
        return pd.DataFrame({
            'Timestamp': ts_values,
            'DriftValue': drift_values
        })
    
    def _generate_threshold_crossings(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate threshold crossing events."""
        threshold = 2.0  # Standard threshold
        crossings = []
        
        # Check fused score crossings
        fused_high = scores_df['fused'] > threshold
        crossing_points = fused_high != fused_high.shift()
        
        for idx in scores_df[crossing_points].index:
            crossings.append({
                'Timestamp': normalize_timestamp_scalar(idx),
                'DetectorType': 'fused',
                'Threshold': threshold,
                'ZScore': round(float(cast(Any, scores_df.loc[idx, 'fused'])), 4),
                'Direction': 'up' if float(cast(Any, scores_df.loc[idx, 'fused'])) > threshold else 'down'
            })
        
        return pd.DataFrame(crossings)
    
    def _generate_since_when(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate alert start times summary."""
        first_alert = scores_df[scores_df['fused'] > 2.0].index.min() if len(scores_df[scores_df['fused'] > 2.0]) > 0 else None
        
        if first_alert:
            duration_hours = (scores_df.index.max() - first_alert).total_seconds() / 3600
            return pd.DataFrame({
                'RunID': [self.run_id],
                'EquipID': [int(self.equip_id) if self.equip_id else 0],
                'AlertZone': ['ALERT'],
                'DurationHours': [round(duration_hours, 2)],
                'StartTimestamp': [normalize_timestamp_scalar(first_alert)],
                'RecordCount': [len(scores_df[scores_df['fused'] > 2.0])]
            })
        else:
            return pd.DataFrame({
                'RunID': [self.run_id],
                'EquipID': [int(self.equip_id) if self.equip_id else 0],
                'AlertZone': ['GOOD'],
                'DurationHours': [0.0],
                'StartTimestamp': [None],
                'RecordCount': [0]
            })

    def _generate_episodes_qc(self, scores_df: pd.DataFrame, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize episode quality for batch-to-batch tracking.

        Columns: RunID, EquipID, EpisodeCount, MedianDurationMinutes, CoveragePct,
                 TimeInAlertPct, MaxFusedZ, AvgFusedZ
        """
        if episodes_df is None or episodes_df.empty:
            return pd.DataFrame({
                'RunID': [self.run_id],
                'EquipID': [int(self.equip_id) if self.equip_id else 0],
                'EpisodeCount': [0],
                'MedianDurationMinutes': [0.0],
                'CoveragePct': [0.0],
                'TimeInAlertPct': [0.0],
                'MaxFusedZ': [float('nan')],
                'AvgFusedZ': [float('nan')],
            })

        ep_count = int(len(episodes_df))
        durations_h = (pd.to_numeric(episodes_df['duration_hours'], errors='coerce') if 'duration_hours' in episodes_df.columns else pd.Series([], dtype=float))
        median_minutes = float(np.nanmedian(durations_h) * 60.0) if len(durations_h) else 0.0

        # Coverage: fraction of timeline covered by episodes (approx using start/end)
        if {'start_ts','end_ts'}.issubset(set(episodes_df.columns)) and len(scores_df.index):
            t0 = scores_df.index.min()
            t1 = scores_df.index.max()
            total_secs = max(1.0, (t1 - t0).total_seconds())
            covered = 0.0
            for _, ep in episodes_df.iterrows():
                try:
                    s = pd.to_datetime(ep['start_ts'], errors='coerce')
                    e = pd.to_datetime(ep['end_ts'], errors='coerce')
                    if pd.notna(s) and pd.notna(e):
                        s = max(s, t0); e = min(e, t1)
                        if e > s:
                            covered += (e - s).total_seconds()
                except Exception:
                    pass
            coverage_pct = float(covered / total_secs * 100.0)
        else:
            coverage_pct = 0.0

        fused = (pd.to_numeric(scores_df['fused'], errors='coerce') if 'fused' in scores_df.columns else pd.Series(dtype=float))
        if len(fused):
            time_in_alert_pct = float((fused > 2.0).mean() * 100.0)
            max_fused = float(np.nanmax(fused))
            avg_fused = float(np.nanmean(fused))
        else:
            time_in_alert_pct = 0.0
            max_fused = float('nan')
            avg_fused = float('nan')

        return pd.DataFrame([{
            'RunID': self.run_id,
            'EquipID': int(self.equip_id) if self.equip_id else 0,
            'EpisodeCount': ep_count,
            'MedianDurationMinutes': round(median_minutes, 2),
            'CoveragePct': round(coverage_pct, 2),
            'TimeInAlertPct': round(time_in_alert_pct, 2),
            'MaxFusedZ': max_fused,
            'AvgFusedZ': avg_fused,
        }])
    def _generate_sensor_rank_now(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate current sensor importance ranking."""
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        if not detector_cols:
            return pd.DataFrame({'DetectorType': [], 'RankPosition': [], 'ContributionPct': [], 'ZScore': []})
        
        # Use latest values for ranking
        latest_scores = scores_df[detector_cols].iloc[-1].abs()
        ranked = latest_scores.sort_values(ascending=False)
        
        # Calculate contributions as percentages
        total = ranked.sum()
        contributions = (ranked / total * 100).round(2) if total > 0 else pd.Series([0] * len(ranked), index=ranked.index)
        
        return pd.DataFrame({
            'DetectorType': ranked.index,
            'RankPosition': range(1, len(ranked) + 1),
            'ContributionPct': contributions.values,
            'ZScore': ranked.values
        })
    
    def _generate_regime_occupancy(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate regime time occupancy statistics with consistent label typing.

        Ensures labels are numeric (Int64) to align with timeline outputs and
        includes all observed labels present in the frame.
        """
        if 'regime_label' not in scores_df.columns or len(scores_df) == 0:
            return pd.DataFrame({'RegimeLabel': [], 'RecordCount': [], 'Percentage': []})

        regimes = pd.to_numeric(scores_df['regime_label'], errors='coerce').dropna().astype('Int64')
        if regimes.empty:
            return pd.DataFrame({'RegimeLabel': [], 'RecordCount': [], 'Percentage': []})

        counts = regimes.value_counts().sort_index()
        total = counts.sum() if len(counts) else 0
        # Dwell approximation
        runs = []
        cur = None
        start = None
        for ts, lab in zip(scores_df.index, regimes):
            if pd.isna(lab):
                continue
            if cur is None:
                cur, start = lab, ts
                continue
            if lab != cur:
                runs.append((int(cur), (ts - start).total_seconds()))
                cur, start = lab, ts
        if cur is not None and start is not None and len(scores_df.index):
            runs.append((int(cur), (scores_df.index[-1] - start).total_seconds()))
        dwell = pd.DataFrame(runs, columns=['RegimeLabel', 'DwellSeconds']) if runs else pd.DataFrame(columns=['RegimeLabel', 'DwellSeconds'])
        stats = []
        for lab, cnt in counts.items():
            try:
                lab_int = int(str(lab))
            except Exception:
                continue
            sel = scores_df['regime_label'] == lab
            fvals = pd.to_numeric(scores_df.loc[sel, 'fused'], errors='coerce') if 'fused' in scores_df.columns else pd.Series(dtype=float)
            fused_mean = float(fvals.mean()) if len(fvals) else float('nan')
            fused_p90 = float(np.nanpercentile(fvals, 90)) if len(fvals) else float('nan')
            avg_dwell = float(dwell[dwell['RegimeLabel'] == lab_int]['DwellSeconds'].mean()) if not dwell.empty else float('nan')
            occ = float(cnt / total * 100.0) if total > 0 else 0.0
            stats.append({
                'RegimeLabel': lab_int,
                'RecordCount': int(cnt),
                'Percentage': round(occ, 2),
                'AvgDwellSeconds': round(avg_dwell, 1) if np.isfinite(avg_dwell) else None,
                'FusedMean': fused_mean,
                'FusedP90': fused_p90,
            })
        return pd.DataFrame(stats).sort_values('RegimeLabel')


    def _generate_fusion_quality_report(self, scores_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """Produce a simple fusion quality snapshot.

        Columns: Detector, Weight, Present, MeanZ, MaxZ, Points
        """
        rows: List[Dict[str, Any]] = []
        weights = ((cfg.get('fusion', {}) or {}).get('weights', {}) or {})
        # Detector columns typically end with _z (including fused)
        zcols = [c for c in scores_df.columns if c.endswith('_z') or c == 'fused']
        stats = {}
        for c in zcols:
            s = pd.to_numeric(scores_df[c], errors='coerce')
            stats[c] = {
                'MeanZ': float(np.nanmean(s)) if len(s) else float('nan'),
                'MaxZ': float(np.nanmax(s)) if len(s) else float('nan'),
                'Points': int(len(s))
            }
        # Normalize common keys (e.g., omr_z vs omr)
        def _norm_key(k: str) -> str:
            return k if k in zcols else (k + '_z' if (k + '_z') in zcols else k)
        for det, w in weights.items():
            key = _norm_key(str(det))
            present = key in zcols
            det_stats = stats.get(key, {'MeanZ': float('nan'), 'MaxZ': float('nan'), 'Points': 0})
            # Use SQL-safe human-readable labels
            detector_label = get_detector_label(key, sql_safe=True)
            rows.append({
                'Detector': detector_label,
                'Weight': float(w) if w is not None else 0.0,
                'Present': bool(present),
                **det_stats
            })
        # Add any present detectors not listed in weights for visibility
        for c in zcols:
            detector_label = get_detector_label(c, sql_safe=True)
            if detector_label not in [r['Detector'] for r in rows]:
                rows.append({'Detector': detector_label, 'Weight': 0.0, 'Present': True, **stats.get(c, {})})
        return pd.DataFrame(rows)

    def _generate_health_distribution_over_time(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Hourly health distribution percentiles to align with heatmaps.

        Columns: BucketStart, BucketSeconds, FusedP50, FusedP75, FusedP90, FusedP95,
                 HealthP50, HealthP10, Count
        """
        if 'fused' not in scores_df.columns or scores_df.empty:
            return pd.DataFrame(columns=['BucketStart','BucketSeconds','FusedP50','FusedP75','FusedP90','FusedP95','HealthP50','HealthP10','Count'])
        idx = pd.DatetimeIndex(scores_df.index)
        fused = pd.to_numeric(scores_df['fused'], errors='coerce')
        health = _health_index(fused)
        df = pd.DataFrame({'fused': fused, 'health': health}, index=idx)
        df = df[~df.index.isna()]
        if df.empty:
            return pd.DataFrame(columns=['BucketStart','BucketSeconds','FusedP50','FusedP75','FusedP90','FusedP95','HealthP50','HealthP10','Count'])
        hourly = df.groupby([idx.year, idx.month, idx.day, idx.hour])
        rows = []
        for (y, m, d, h), grp in hourly:
            bucket_start = pd.Timestamp(year=int(y), month=int(m), day=int(d), hour=int(h))
            fvals = pd.to_numeric(grp['fused'], errors='coerce').dropna()
            hvals = pd.to_numeric(grp['health'], errors='coerce').dropna()
            rows.append({
                'BucketStart': bucket_start,
                'BucketSeconds': 3600,
                'FusedP50': float(np.nanpercentile(fvals, 50)) if len(fvals) else float('nan'),
                'FusedP75': float(np.nanpercentile(fvals, 75)) if len(fvals) else float('nan'),
                'FusedP90': float(np.nanpercentile(fvals, 90)) if len(fvals) else float('nan'),
                'FusedP95': float(np.nanpercentile(fvals, 95)) if len(fvals) else float('nan'),
                'HealthP50': float(np.nanpercentile(hvals, 50)) if len(hvals) else float('nan'),
                'HealthP10': float(np.nanpercentile(hvals, 10)) if len(hvals) else float('nan'),
                'Count': int(len(grp))
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out['BucketStart'] = normalize_timestamp_series(out['BucketStart'])
        return out

    def _generate_omr_timeline(self, scores_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        """Generate OMR timeline with weight annotation when available."""
        ts_values = normalize_timestamp_series(scores_df.index).to_list()
        omr_series = (pd.to_numeric(scores_df['omr_z'], errors='coerce') if 'omr_z' in scores_df.columns else pd.Series(dtype=float))
        try:
            weights = (cfg.get('fusion', {}) or {}).get('weights', {})
            omr_weight = float(weights.get('omr_z', weights.get('omr', 0.0)))
        except Exception:
            omr_weight = 0.0
        return pd.DataFrame({
            'Timestamp': ts_values,
            'OMR_Z': omr_series.round(4).to_list(),
            'OMR_Weight': [omr_weight] * len(ts_values)
        })

    def write_threshold_metadata(
        self,
        equip_id: int,
        threshold_type: str,
        threshold_value: Union[float, Dict[int, float]],
        calculation_method: str,
        sample_count: Optional[int] = None,
        train_start: Optional[datetime] = None,
        train_end: Optional[datetime] = None,
        config_signature: Optional[str] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Persist adaptive threshold metadata to ACM_ThresholdMetadata table.
        
        Args:
            equip_id: Equipment identifier
            threshold_type: Type of threshold ('fused_alert_z', 'fused_warn_z', etc.)
            threshold_value: Threshold value (float for global, dict for per-regime)
            calculation_method: Method used ('quantile_99.7', 'mad_3sigma', 'hybrid', etc.)
            sample_count: Number of training samples used
            train_start: Training data start time
            train_end: Training data end time
            config_signature: Config hash for invalidation
            notes: Optional explanation
            
        Example:
            # Global threshold
            output_manager.write_threshold_metadata(
                equip_id=1,
                threshold_type='fused_alert_z',
                threshold_value=3.2,
                calculation_method='quantile_99.7',
                sample_count=5000
            )
            
            # Per-regime thresholds
            output_manager.write_threshold_metadata(
                equip_id=1,
                threshold_type='fused_alert_z',
                threshold_value={0: 2.8, 1: 3.5, 2: 2.3},
                calculation_method='mad_3sigma',
                sample_count=5000
            )
        """
        if self.sql_client is None:
            Console.warn("SQL client not available - skipping threshold metadata write", component="THRESHOLD", equip_id=equip_id, threshold_type=threshold_type)
            return
            
        try:
            # Mark old thresholds as inactive for this equipment/threshold_type
            with self.sql_client.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ACM_ThresholdMetadata 
                    SET IsActive = 0 
                    WHERE EquipID = ? AND ThresholdType = ?
                    """,
                    (equip_id, threshold_type)
                )
            
            # Prepare rows to insert
            rows = []
            if isinstance(threshold_value, dict):
                # Per-regime thresholds
                for regime_id, thresh_val in threshold_value.items():
                    rows.append({
                        'EquipID': equip_id,
                        'RegimeID': regime_id if regime_id >= 0 else None,
                        'ThresholdType': threshold_type,
                        'ThresholdValue': float(thresh_val),
                        'CalculationMethod': calculation_method,
                        'SampleCount': sample_count,
                        'TrainStartTime': train_start,
                        'TrainEndTime': train_end,
                        'ConfigSignature': config_signature,
                        'IsActive': 1,
                        'Notes': notes
                    })
            else:
                # Global threshold
                rows.append({
                    'EquipID': equip_id,
                    'RegimeID': None,
                    'ThresholdType': threshold_type,
                    'ThresholdValue': float(threshold_value),
                    'CalculationMethod': calculation_method,
                    'SampleCount': sample_count,
                    'TrainStartTime': train_start,
                    'TrainEndTime': train_end,
                    'ConfigSignature': config_signature,
                    'IsActive': 1,
                    'Notes': notes
                })
            
            # Insert new thresholds directly via cursor for reliability
            # Ensure CreatedAt and RunID present
            created_at = pd.Timestamp.now().tz_localize(None)
            for r in rows:
                r['CreatedAt'] = created_at
                # Ensure EquipID set
                r['EquipID'] = equip_id
            
            insert_sql = (
                "INSERT INTO ACM_ThresholdMetadata (EquipID, RegimeID, ThresholdType, ThresholdValue, "
                "CalculationMethod, SampleCount, TrainStartTime, TrainEndTime, CreatedAt, ConfigSignature, "
                "IsActive, Notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            params = [
                (
                    r.get('EquipID'),
                    r.get('RegimeID'),
                    r.get('ThresholdType'),
                    r.get('ThresholdValue'),
                    r.get('CalculationMethod'),
                    r.get('SampleCount'),
                    r.get('TrainStartTime'),
                    r.get('TrainEndTime'),
                    r.get('CreatedAt'),
                    r.get('ConfigSignature'),
                    r.get('IsActive'),
                    r.get('Notes')
                ) for r in rows
            ]
            with self.sql_client.cursor() as cur:
                cur.executemany(insert_sql, params)
            
            Console.info(f"Threshold metadata written: {threshold_type} = "
                        f"{threshold_value if not isinstance(threshold_value, dict) else f'{len(threshold_value)} regimes'} "
                        f"({calculation_method})",
                        component="THRESHOLD")
            
        except Exception as e:
            Console.error(f"Failed to write threshold metadata: {e}", component="THRESHOLD", equip_id=equip_id, threshold_type=threshold_type, error_type=type(e).__name__, error=str(e)[:200])
    
    def read_threshold_metadata(
        self,
        equip_id: int,
        threshold_type: str,
        regime_id: Optional[int] = None
    ) -> Optional[float]:
        """
        Read latest active threshold from ACM_ThresholdMetadata.
        
        Args:
            equip_id: Equipment identifier
            threshold_type: Type of threshold ('fused_alert_z', 'fused_warn_z', etc.)
            regime_id: Optional regime ID (None for global threshold)
            
        Returns:
            Threshold value if found, None otherwise
            
        Example:
            # Read global threshold
            threshold = output_manager.read_threshold_metadata(
                equip_id=1,
                threshold_type='fused_alert_z'
            )
            
            # Read regime-specific threshold
            threshold = output_manager.read_threshold_metadata(
                equip_id=1,
                threshold_type='fused_alert_z',
                regime_id=2
            )
        """
        if self.sql_client is None:
            Console.warn("SQL client not available - cannot read threshold metadata", component="THRESHOLD", equip_id=equip_id, threshold_type=threshold_type)
            return None
            
        try:
            with self.sql_client.get_cursor() as cur:
                if regime_id is not None:
                    # Regime-specific lookup
                    cur.execute(
                        """
                        SELECT TOP 1 ThresholdValue
                        FROM ACM_ThresholdMetadata
                        WHERE EquipID = ? 
                          AND ThresholdType = ?
                          AND RegimeID = ?
                          AND IsActive = 1
                        ORDER BY CreatedAt DESC
                        """,
                        (equip_id, threshold_type, regime_id)
                    )
                else:
                    # Global lookup (RegimeID IS NULL)
                    cur.execute(
                        """
                        SELECT TOP 1 ThresholdValue
                        FROM ACM_ThresholdMetadata
                        WHERE EquipID = ? 
                          AND ThresholdType = ?
                          AND RegimeID IS NULL
                          AND IsActive = 1
                        ORDER BY CreatedAt DESC
                        """,
                        (equip_id, threshold_type)
                    )
                
                row = cur.fetchone()
                if row:
                    return float(row[0])
                return None
                
        except Exception as e:
            Console.error(f"Failed to read threshold metadata: {e}", component="THRESHOLD", equip_id=equip_id, threshold_type=threshold_type, error_type=type(e).__name__, error=str(e)[:200])
            return None

    # ========================================================================
    # v10.1.0 Regime-Conditioned Forecasting Tables
    # ========================================================================
    
    def write_rul_by_regime(
        self,
        equip_id: int,
        run_id: str,
        rul_by_regime: pd.DataFrame
    ) -> None:
        """
        Persist per-regime RUL estimates to ACM_RUL_ByRegime.
        
        Args:
            equip_id: Equipment identifier
            run_id: Run identifier
            rul_by_regime: DataFrame with columns:
                - RegimeLabel: int
                - RUL_Hours: float
                - P10_LowerBound: float
                - P50_Median: float  
                - P90_UpperBound: float
                - DegradationRate: float (health units per hour)
                - Confidence: float (0-1)
                - Method: str
                - SampleCount: int
        """
        if rul_by_regime is None or rul_by_regime.empty:
            return
            
        df = rul_by_regime.copy()
        df['EquipID'] = equip_id
        df['RunID'] = run_id
        df['CreatedAt'] = pd.Timestamp.now().tz_localize(None)
        
        self._ensure_dataframe_columns(df, 'ACM_RUL_ByRegime')
        self.write_dataframe(df, 'ACM_RUL_ByRegime')
        Console.info(f"RUL by regime written: {len(df)} regimes", component="RUL")

    def write_regime_hazard(
        self,
        equip_id: int,
        run_id: str,
        regime_hazard: pd.DataFrame
    ) -> None:
        """
        Persist per-regime hazard statistics to ACM_RegimeHazard.
        
        Args:
            equip_id: Equipment identifier
            run_id: Run identifier
            regime_hazard: DataFrame with columns:
                - RegimeLabel: int
                - Timestamp: datetime
                - HazardRate: float (instantaneous failure rate)
                - SurvivalProb: float (cumulative survival probability)
                - CumulativeHazard: float
                - FailureProb: float (probability of failure by this time)
                - HealthAtTime: float
                - DriftAtTime: float (optional)
                - OMR_Z_AtTime: float (optional)
        """
        if regime_hazard is None or regime_hazard.empty:
            return
            
        df = regime_hazard.copy()
        df['EquipID'] = equip_id
        df['RunID'] = run_id
        df['CreatedAt'] = pd.Timestamp.now().tz_localize(None)
        
        # Normalize timestamps
        if 'Timestamp' in df.columns:
            df['Timestamp'] = normalize_timestamp_series(df['Timestamp'])
        
        self._ensure_dataframe_columns(df, 'ACM_RegimeHazard')
        self.write_dataframe(df, 'ACM_RegimeHazard')
        Console.info(f"Regime hazard written: {len(df)} records", component="RUL")

    def write_forecast_context(
        self,
        equip_id: int,
        run_id: str,
        forecast_context: pd.DataFrame
    ) -> None:
        """
        Persist unified forecast context with OMR/drift/regime state to ACM_ForecastContext.
        
        This table provides a comprehensive snapshot of all factors influencing the forecast.
        
        Args:
            equip_id: Equipment identifier
            run_id: Run identifier
            forecast_context: DataFrame with columns:
                - Timestamp: datetime (forecast reference time)
                - ForecastHorizon_Hours: float
                - CurrentHealth: float
                - CurrentRegime: int
                - RegimeConfidence: float
                - CurrentOMR_Z: float
                - OMR_Contribution: float (weight-adjusted contribution)
                - CurrentDrift_Z: float
                - DriftTrend: str ('stable', 'increasing', 'decreasing')
                - FusedZ: float
                - HealthTrend: str ('improving', 'stable', 'degrading')
                - DataQuality: float (0-1)
                - ModelConfidence: float (0-1, forecast model confidence)
                - ActiveDefects: int (count of active sensor defects)
                - TopContributor: str (sensor/detector driving health)
                - Notes: str (optional context)
        """
        if forecast_context is None or forecast_context.empty:
            return
            
        df = forecast_context.copy()
        df['EquipID'] = equip_id
        df['RunID'] = run_id
        df['CreatedAt'] = pd.Timestamp.now().tz_localize(None)
        
        # Normalize timestamps
        if 'Timestamp' in df.columns:
            df['Timestamp'] = normalize_timestamp_series(df['Timestamp'])
        
        self._ensure_dataframe_columns(df, 'ACM_ForecastContext')
        self.write_dataframe(df, 'ACM_ForecastContext')
        Console.info(f"Forecast context written: {len(df)} records", component="RUL")

    def write_adaptive_thresholds_by_regime(
        self,
        equip_id: int,
        run_id: str,
        thresholds_by_regime: pd.DataFrame
    ) -> None:
        """
        Persist per-regime adaptive thresholds to ACM_AdaptiveThresholds_ByRegime.
        
        Args:
            equip_id: Equipment identifier
            run_id: Run identifier  
            thresholds_by_regime: DataFrame with columns:
                - RegimeLabel: int
                - ThresholdType: str ('fused_alert_z', 'fused_warn_z', etc.)
                - ThresholdValue: float
                - CalculationMethod: str
                - SampleCount: int
                - RegimeHealthMean: float (mean health in this regime)
                - RegimeHealthStd: float (std health in this regime)
                - RegimeOccupancy: float (% time in regime)
        """
        if thresholds_by_regime is None or thresholds_by_regime.empty:
            return
            
        df = thresholds_by_regime.copy()
        df['EquipID'] = equip_id
        df['RunID'] = run_id
        df['IsActive'] = 1
        df['CreatedAt'] = pd.Timestamp.now().tz_localize(None)
        
        self._ensure_dataframe_columns(df, 'ACM_AdaptiveThresholds_ByRegime')
        self.write_dataframe(df, 'ACM_AdaptiveThresholds_ByRegime')
        Console.info(f"Adaptive thresholds by regime written: {len(df)} records", component="RUL")

    def load_regime_hazard_stats(
        self,
        equip_id: int,
        lookback_days: int = 90
    ) -> Optional[pd.DataFrame]:
        """
        Load historical regime hazard statistics for forecasting.
        
        Args:
            equip_id: Equipment identifier
            lookback_days: Days of history to load
            
        Returns:
            DataFrame with per-regime statistics or None
        """
        if self.sql_client is None:
            return None
            
        try:
            query = """
                SELECT RegimeLabel, 
                       AVG(HazardRate) as AvgHazardRate,
                       AVG(DegradationRate) as AvgDegradationRate,
                       AVG(HealthAtTime) as AvgHealth,
                       STDEV(HealthAtTime) as StdHealth,
                       COUNT(*) as SampleCount,
                       AVG(FailureProb) as AvgFailureProb
                FROM (
                    SELECT r.RegimeLabel, h.HazardRate, 
                           COALESCE(rul.DegradationRate, 0) as DegradationRate,
                           h.HealthAtTime, h.FailureProb
                    FROM ACM_RegimeHazard h
                    JOIN ACM_RegimeTimeline r 
                        ON h.EquipID = r.EquipID AND h.Timestamp = r.Timestamp
                    LEFT JOIN ACM_RUL_ByRegime rul
                        ON h.EquipID = rul.EquipID AND h.RunID = rul.RunID 
                        AND r.RegimeLabel = rul.RegimeLabel
                    WHERE h.EquipID = ?
                      AND h.Timestamp >= DATEADD(day, -?, GETDATE())
                ) sub
                GROUP BY RegimeLabel
                ORDER BY RegimeLabel
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(query, (equip_id, lookback_days))
                rows = cur.fetchall()
                if not rows:
                    return None
                cols = ['RegimeLabel', 'AvgHazardRate', 'AvgDegradationRate', 
                        'AvgHealth', 'StdHealth', 'SampleCount', 'AvgFailureProb']
                return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            Console.warn(f"Failed to load regime hazard stats: {e}", component="RUL", equip_id=equip_id, lookback_days=lookback_days, error_type=type(e).__name__)
            return None

    def load_omr_drift_context(
        self,
        equip_id: int,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Load recent OMR and drift context for forecasting.
        
        Args:
            equip_id: Equipment identifier
            lookback_hours: Hours of recent data to analyze
            
        Returns:
            Dict with keys: omr_z, omr_trend, drift_z, drift_trend, top_contributors
        """
        context: Dict[str, Any] = {
            'omr_z': None,
            'omr_trend': 'unknown',
            'drift_z': None,
            'drift_trend': 'unknown',
            'top_contributors': []
        }
        
        if self.sql_client is None:
            return context
            
        try:
            # Get recent OMR values
            omr_query = """
                SELECT TOP 10 OMR_Z 
                FROM ACM_OMRTimeline 
                WHERE EquipID = ? 
                  AND Timestamp >= DATEADD(hour, -?, GETDATE())
                ORDER BY Timestamp DESC
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(omr_query, (equip_id, lookback_hours))
                omr_rows = cur.fetchall()
                if omr_rows:
                    omr_values = [float(r[0]) for r in omr_rows if r[0] is not None]
                    if omr_values:
                        context['omr_z'] = omr_values[0]  # Most recent
                        if len(omr_values) >= 3:
                            # Trend: compare first half vs second half
                            mid = len(omr_values) // 2
                            recent_avg = np.mean(omr_values[:mid])
                            older_avg = np.mean(omr_values[mid:])
                            if recent_avg > older_avg + 0.5:
                                context['omr_trend'] = 'increasing'
                            elif recent_avg < older_avg - 0.5:
                                context['omr_trend'] = 'decreasing'
                            else:
                                context['omr_trend'] = 'stable'
            
            # Get recent drift values
            drift_query = """
                SELECT TOP 10 DriftValue 
                FROM ACM_DriftSeries 
                WHERE EquipID = ?
                  AND Timestamp >= DATEADD(hour, -?, GETDATE())
                ORDER BY Timestamp DESC
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(drift_query, (equip_id, lookback_hours))
                drift_rows = cur.fetchall()
                if drift_rows:
                    drift_values = [float(r[0]) for r in drift_rows if r[0] is not None]
                    if drift_values:
                        context['drift_z'] = drift_values[0]
                        if len(drift_values) >= 3:
                            mid = len(drift_values) // 2
                            recent_avg = np.mean(drift_values[:mid])
                            older_avg = np.mean(drift_values[mid:])
                            if recent_avg > older_avg + 0.3:
                                context['drift_trend'] = 'increasing'
                            elif recent_avg < older_avg - 0.3:
                                context['drift_trend'] = 'decreasing'
                            else:
                                context['drift_trend'] = 'stable'
            
            # Get top OMR contributors
            contrib_query = """
                SELECT TOP 5 SensorName, AVG(ContributionScore) as AvgContrib
                FROM ACM_OMRContributionsLong
                WHERE EquipID = ?
                  AND Timestamp >= DATEADD(hour, -?, GETDATE())
                GROUP BY SensorName
                ORDER BY AVG(ContributionScore) DESC
            """
            with self.sql_client.get_cursor() as cur:
                cur.execute(contrib_query, (equip_id, lookback_hours))
                contrib_rows = cur.fetchall()
                if contrib_rows:
                    context['top_contributors'] = [
                        {'sensor': r[0], 'contribution': float(r[1])} 
                        for r in contrib_rows if r[0] is not None
                    ]
                    
        except Exception as e:
            Console.warn(f"Failed to load OMR/drift context: {e}", component="RUL", equip_id=equip_id, lookback_hours=lookback_hours, error_type=type(e).__name__)
            
        return context



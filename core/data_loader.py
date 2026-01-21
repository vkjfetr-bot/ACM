"""
Data Loader for ACM
====================

Handles all data loading from SQL Server historian tables.
Extracted from output_manager.py as part of Phase 2 debloating.

Key responsibilities:
- Load historian data via stored procedure
- Parse timestamps and set as index
- Filter future timestamps
- Infer numeric columns
- Check and resample cadence
- Cold-start train/score splitting

Usage:
    from core.data_loader import DataLoader, DataMeta
    
    loader = DataLoader(sql_client)
    train, score, meta = loader.load_from_sql(cfg, "FD_FAN", start, end, is_coldstart=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from core.observability import Console


# ============================================================================
# DATA METADATA
# ============================================================================
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    """Get config value by dot path with type preservation."""
    keys = path.split('.')
    current: Any = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _future_cutoff_ts(cfg: Dict[str, Any]) -> pd.Timestamp:
    """Return timestamp cutoff that optionally allows future data via config."""
    raw_value = _cfg_get(cfg, "runtime.future_grace_minutes", 0) or 0
    try:
        minutes = int(raw_value)
    except (TypeError, ValueError):
        minutes = 0
    minutes = max(0, minutes)
    return pd.Timestamp.now() + pd.Timedelta(minutes=minutes)


def parse_ts_index(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Parse timestamp column and set as index."""
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found")
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.set_index(ts_col).sort_index()
    return df


def coerce_local_and_filter_future(
    df: pd.DataFrame, label: str, now_cutoff: pd.Timestamp
) -> Tuple[pd.DataFrame, int, int]:
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
        Console.warn(
            f"Dropped {before_drop - len(df)} rows with invalid timestamps from {label}",
            component="DATA", label=label, rows_dropped=before_drop - len(df), rows_remaining=len(df)
        )

    future_mask = df.index > now_cutoff
    future_rows = int(future_mask.sum())
    if future_rows:
        Console.warn(
            f"Dropping {future_rows} future timestamp row(s) from {label} (cutoff={now_cutoff:%Y-%m-%d %H:%M:%S})",
            component="DATA", label=label, future_rows=future_rows, cutoff=str(now_cutoff)
        )
        df = df[~future_mask]

    return df, tz_stripped, future_rows


def infer_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def native_cadence_secs(idx: pd.DatetimeIndex) -> float:
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
            return float(np.median(diffs))
        except Exception:
            return float('inf')


def check_cadence(idx: pd.DatetimeIndex, sampling_secs: Optional[int], jitter_ratio: float = 0.05) -> bool:
    """Check if timestamps have regular cadence."""
    if sampling_secs is None or len(idx) < 2:
        return True
    diffs = idx.to_series().diff().dropna()
    expected = pd.Timedelta(seconds=sampling_secs)
    tolerance = expected * jitter_ratio
    return ((diffs - expected).abs() <= tolerance).mean() >= 0.9


def resample_df(
    df: pd.DataFrame,
    sampling_secs: int,
    interp_method: str = "linear",
    strict: bool = False,
    max_gap_secs: int = 300,
    max_fill_ratio: float = 0.2
) -> pd.DataFrame:
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
        df_resampled = df_resampled.interpolate(
            method=cast(Any, interp_method), limit=max_gap_periods, limit_direction='both'
        )
    if strict:
        fill_ratio = df_resampled.isnull().sum().sum() / (len(df_resampled) * len(df_resampled.columns))
        if fill_ratio > max_fill_ratio:
            raise ValueError(f"Too much missing data after resample: {fill_ratio:.1%} > {max_fill_ratio:.1%}")
    return df_resampled


# ============================================================================
# DATA LOADER CLASS
# ============================================================================
class DataLoader:
    """
    Handles data loading from SQL Server historian tables.
    
    This class encapsulates all data loading logic previously embedded in OutputManager.
    """
    
    def __init__(self, sql_client: Any):
        """
        Initialize the DataLoader.
        
        Args:
            sql_client: SQL client instance (core.sql_client.SQLClient)
        """
        self.sql_client = sql_client
    
    def load_from_sql(
        self,
        cfg: Dict[str, Any],
        equipment_name: str,
        start_utc: Optional[pd.Timestamp],
        end_utc: Optional[pd.Timestamp],
        is_coldstart: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
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
            raise ValueError("SQL mode requires start_utc and end_utc parameters")
        
        # COLD-02: Configurable cold-start split ratio (default 0.6 = 60% train, 40% score)
        # Only used during coldstart - regular batch mode uses ALL data for scoring
        cold_start_split_ratio = float(_cfg_get(data_cfg, "cold_start_split_ratio", 0.6))
        if not (0.1 <= cold_start_split_ratio <= 0.9):
            Console.warn(
                f"Invalid cold_start_split_ratio={cold_start_split_ratio}, using default 0.6",
                component="DATA", invalid_value=cold_start_split_ratio, equipment=equipment_name
            )
            cold_start_split_ratio = 0.6
        
        min_train_samples = int(_cfg_get(data_cfg, "min_train_samples", 500))
        
        Console.info(f"Loading from SQL historian: {equipment_name}", component="DATA")
        Console.info(f"Time range: {start_utc} to {end_utc}", component="DATA")
        
        # Call stored procedure to get all data for time range
        # Pass EquipmentName directly - SP will resolve to correct data table (e.g., FD_FAN_Data)
        cur = None
        try:
            if self.sql_client is None:
                raise ValueError("SQL mode requested but no SQL client available")
            cur = cast(Any, self.sql_client).cursor()
            # Pass EquipmentName to stored procedure (SP resolves to {EquipmentName}_Data table)
            cur.execute(
                "EXEC dbo.usp_ACM_GetHistorianData_TEMP @StartTime=?, @EndTime=?, @EquipmentName=?",
                (start_utc, end_utc, equipment_name)
            )
            
            # Fetch all rows
            rows = cur.fetchall()
            if not rows:
                raise ValueError(f"No data returned from SQL historian for {equipment_name} in time range")
            
            # Get column names from cursor description
            columns = [desc[0] for desc in cur.description]
            
            # Convert to DataFrame
            df_all = pd.DataFrame.from_records(rows, columns=columns)
            
            Console.info(f"Retrieved {len(df_all)} rows from SQL historian", component="DATA")
            
        except Exception as e:
            Console.error(
                f"Failed to load from SQL historian: {e}",
                component="DATA", equipment=equipment_name, error_type=type(e).__name__, error=str(e)[:200]
            )
            raise
        finally:
            try:
                if cur is not None:
                    cur.close()
            except Exception:
                pass
        
        # Validate sufficient data
        # For coldstart, enforce minimum. For incremental scoring, allow smaller batches.
        required_minimum = min_train_samples if is_coldstart else max(10, min_train_samples // 10)
        if len(df_all) < required_minimum:
            raise ValueError(f"Insufficient data from SQL historian: {len(df_all)} rows (minimum {required_minimum} required)")

        # Robust timestamp handling for SQL historian: if configured column is missing
        # but the standard EntryDateTime column is present, fall back to it.
        if ts_col not in df_all.columns and "EntryDateTime" in df_all.columns:
            Console.warn(
                f"Timestamp column '{ts_col}' not found in SQL historian results; "
                "falling back to 'EntryDateTime'.",
                component="DATA", configured_col=ts_col, fallback_col="EntryDateTime", equipment=equipment_name
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
                Console.warn(
                    f"Training data ({len(train_raw)} rows) is below recommended minimum ({min_train_samples} rows)",
                    component="DATA", actual_rows=len(train_raw), min_required=min_train_samples, equipment=equipment_name
                )
                Console.warn(
                    f"Model quality may be degraded. Consider: wider time window, higher split_ratio (current: {cold_start_split_ratio:.2f})",
                    component="DATA", split_ratio=cold_start_split_ratio, equipment=equipment_name
                )
            
            Console.info(
                f"COLDSTART Split ({cold_start_split_ratio:.1%}): {len(train_raw)} train rows, {len(score_raw)} score rows",
                component="DATA"
            )
        else:
            # REGULAR BATCH MODE: Use ALL data for scoring, load baseline from cache
            train_raw = pd.DataFrame()  # Empty train, will be loaded from baseline_buffer
            score_raw = df_all.copy()
            Console.info(
                f"BATCH MODE: All {len(score_raw)} rows allocated to scoring (baseline from cache)",
                component="DATA"
            )
        
        # Parse timestamps / index
        # Handle empty train in batch mode
        if len(train_raw) == 0 and not is_coldstart:
            # Create empty DataFrame with DatetimeIndex matching score columns
            train = pd.DataFrame(columns=train_raw.columns)
            train.index = pd.DatetimeIndex([], name=ts_col)
        else:
            train = parse_ts_index(train_raw, ts_col)
        
        score = parse_ts_index(score_raw, ts_col)
        
        # Filter future timestamps
        now_cutoff = _future_cutoff_ts(cfg)
        train, tz_stripped_train, future_train = coerce_local_and_filter_future(train, "TRAIN", now_cutoff)
        score, tz_stripped_score, future_score = coerce_local_and_filter_future(score, "SCORE", now_cutoff)
        tz_stripped_total = tz_stripped_train + tz_stripped_score
        future_rows_total = future_train + future_score
        
        # Validate training sample count (skip in batch mode - train comes from baseline_buffer)
        if len(train) < min_train_samples and is_coldstart:
            Console.warn(
                f"Training data ({len(train)} rows) is below recommended minimum ({min_train_samples} rows)",
                component="DATA", actual_rows=len(train), min_required=min_train_samples, equipment=equipment_name, mode="coldstart"
            )
        
        # Keep numeric only (same set across train/score)
        if len(train) == 0 and not is_coldstart:
            # BATCH MODE: Train is empty, use all score columns
            # Train will be loaded from baseline_buffer later in acm_main.py
            score_num = infer_numeric_cols(score)
            kept = sorted(score_num)
            dropped = [c for c in score.columns if c not in kept]
            train = pd.DataFrame(columns=kept)  # Empty train with correct columns
            score = score[kept]
            score = score.astype(np.float32)
            Console.info(
                f"BATCH MODE: Train empty (will load from baseline_buffer), using all {len(kept)} score columns",
                component="DATA"
            )
        else:
            # COLDSTART MODE or TRAIN EXISTS: Use intersection of train/score columns
            train_num = infer_numeric_cols(train)
            score_num = infer_numeric_cols(score)
            kept = sorted(list(set(train_num).intersection(score_num)))
            dropped = [c for c in train.columns if c not in kept]
            train = train[kept]
            score = score[kept]
            train = train.astype(np.float32)
            score = score.astype(np.float32)
        
        Console.info(f"Kept {len(kept)} numeric columns, dropped {len(dropped)} non-numeric", component="DATA")
        
        Console.status(f"Checking cadence and resampling for {len(score)} score rows...")
        
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
        
        Console.status("  Checking train cadence...")
        cad_ok_train = check_cadence(cast(pd.DatetimeIndex, train.index), sampling_secs)
        Console.status("  Checking score cadence...")
        cad_ok_score = check_cadence(cast(pd.DatetimeIndex, score.index), sampling_secs)
        cadence_ok = bool(cad_ok_train and cad_ok_score)
        Console.status(f"  Cadence check complete: train={cad_ok_train}, score={cad_ok_score}")
        
        # v11.5.0: CRITICAL ANTI-UPSAMPLE GUARD
        # Upsampling (creating more rows than exist natively) is NEVER allowed.
        # It creates fake data via interpolation, inflates row counts 10x, and
        # corrupts all downstream calibration and anomaly detection.
        native_train = native_cadence_secs(cast(pd.DatetimeIndex, train.index))
        native_score = native_cadence_secs(cast(pd.DatetimeIndex, score.index))
        native_cadence = min(
            native_train if math.isfinite(native_train) else float('inf'),
            native_score if math.isfinite(native_score) else float('inf')
        )
        
        if sampling_secs is not None and math.isfinite(native_cadence):
            if sampling_secs < native_cadence * 0.9:  # 10% tolerance
                # NEVER upsample - use native cadence instead
                Console.warn(
                    f"ANTI-UPSAMPLE: Requested resample ({sampling_secs}s) < native cadence ({native_cadence:.1f}s). "
                    f"Using native cadence to prevent data inflation.",
                    component="DATA", requested_secs=sampling_secs, native_secs=native_cadence, equipment=equipment_name
                )
                # Set to None to skip resampling entirely (preserve native data)
                sampling_secs = None
                cadence_ok = True  # Native data is always considered valid cadence
        
        Console.info(
            f"Cadence: native={native_cadence:.1f}s, requested={sampling_secs or 'auto'}, will_resample={sampling_secs is not None and not cadence_ok}",
            component="DATA", native_cadence=native_cadence, equipment=equipment_name
        )

        if sampling_secs is not None:
            base_secs = float(sampling_secs)
        else:
            base_secs = native_cadence if math.isfinite(native_cadence) else 1.0
        max_gap_secs = int(_cfg_get(data_cfg, "max_gap_secs", base_secs * 3))
        
        explode_guard_factor = float(_cfg_get(data_cfg, "explode_guard_factor", 2.0))
        will_resample = allow_resample and (not cadence_ok) and (sampling_secs is not None)
        if will_resample:
            span_secs = (train.index[-1].value - train.index[0].value) / 1e9 if len(train.index) else 0.0
            safe_sampling = float(sampling_secs) if sampling_secs is not None else 1.0
            approx_rows = int(span_secs / max(1.0, safe_sampling)) + 1
            if len(train) and approx_rows > explode_guard_factor * len(train):
                Console.warn(
                    f"Resample would expand rows from {len(train)} -> ~{approx_rows} (>x{explode_guard_factor:.1f}). Skipping resample.",
                    component="DATA"
                )
                will_resample = False

        if will_resample:
            assert sampling_secs is not None
            train = resample_df(train, int(sampling_secs), interp_method, resample_strict, max_gap_secs, max_fill_ratio)
            score = resample_df(score, int(sampling_secs), interp_method, resample_strict, max_gap_secs, max_fill_ratio)
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
        
        Console.info(
            f"SQL historian load complete: {len(train)} train + {len(score)} score = {len(train) + len(score)} total rows",
            component="DATA"
        )
        return train, score, meta

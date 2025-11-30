"""
FOR-DQ-02: Centralized timestamp normalization utilities.

All ACM timestamps are stored as timezone-naive local time (not UTC).
This module provides consistent helpers to strip timezone info from
various pandas timestamp structures.

Policy: "All timestamps are stored as naive local time"
- No UTC conversions
- Strip timezone info before SQL writes
- Apply consistently across all modules
"""

from typing import Optional, Union, List
import pandas as pd


def normalize_timestamp_scalar(ts) -> Optional[pd.Timestamp]:
    """
    Convert a scalar timestamp to timezone-naive local time.
    
    Args:
        ts: Timestamp-like value (pd.Timestamp, datetime, string, or None)
    
    Returns:
        Timezone-naive pd.Timestamp or None if invalid
    
    Examples:
        >>> normalize_timestamp_scalar("2024-01-01 12:00:00+00:00")
        Timestamp('2024-01-01 12:00:00')
        >>> normalize_timestamp_scalar(None)
        None
    """
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        result = pd.to_datetime(ts, errors='coerce')
        if pd.isna(result):
            return None
        # Strip timezone info if present
        if hasattr(result, 'tz') and result.tz is not None:
            return result.tz_localize(None)
        return result
    except Exception:
        return None


def normalize_timestamp_series(
    idx_or_series: Union[pd.Index, pd.Series]
) -> pd.Series:
    """
    Convert an Index or Series to timezone-naive local timestamps.
    
    Args:
        idx_or_series: pandas Index or Series with datetime values
    
    Returns:
        Series of timezone-naive timestamps with original index
    
    Examples:
        >>> idx = pd.DatetimeIndex(['2024-01-01+00:00'], tz='UTC')
        >>> normalize_timestamp_series(idx)
        0   2024-01-01
        dtype: datetime64[ns]
    """
    if isinstance(idx_or_series, pd.Index):
        series = pd.Series(idx_or_series, index=idx_or_series)
    else:
        series = idx_or_series.copy()
    
    # Convert to datetime if not already
    series = pd.to_datetime(series, errors='coerce')
    
    # Strip timezone if present
    if hasattr(series.dt, 'tz') and series.dt.tz is not None:
        series = series.dt.tz_localize(None)
    
    return series


def normalize_timestamps(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Convert specified DataFrame columns to timezone-naive local time.
    
    Args:
        df: DataFrame with timestamp columns
        cols: List of column names to normalize. If None, normalizes all datetime columns.
        inplace: If True, modifies df in-place. If False, returns a copy.
    
    Returns:
        DataFrame with normalized timestamp columns
    
    Examples:
        >>> df = pd.DataFrame({'Timestamp': ['2024-01-01+00:00']})
        >>> normalize_timestamps(df, cols=['Timestamp'])
        # Returns df with Timestamp column as naive datetime
    """
    df_out = df if inplace else df.copy()
    
    # If no columns specified, find all datetime columns
    if cols is None:
        cols = [col for col in df_out.columns if pd.api.types.is_datetime64_any_dtype(df_out[col])]
    
    for col in cols:
        if col not in df_out.columns:
            continue
        
        # Convert to datetime if not already
        df_out[col] = pd.to_datetime(df_out[col], errors="coerce")
        
        # Strip timezone if present
        try:
            if df_out[col].dt.tz is not None:
                df_out[col] = df_out[col].dt.tz_localize(None)
        except (AttributeError, TypeError):
            # Column is not datetime or already naive; skip
            pass
    
    return df_out


def normalize_index(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Normalize DataFrame index to timezone-naive local time if it's a DatetimeIndex.
    
    Args:
        df: DataFrame with DatetimeIndex
        inplace: If True, modifies df in-place. If False, returns a copy.
    
    Returns:
        DataFrame with normalized index
    
    Examples:
        >>> df = pd.DataFrame({'value': [1]}, index=pd.DatetimeIndex(['2024-01-01+00:00']))
        >>> normalize_index(df)
        # Returns df with naive datetime index
    """
    df_out = df if inplace else df.copy()
    
    if isinstance(df_out.index, pd.DatetimeIndex):
        if df_out.index.tz is not None:
            df_out.index = df_out.index.tz_localize(None)
    
    return df_out

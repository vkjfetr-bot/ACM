"""
Lean SQL-Only Output Manager for ACM
====================================

Pure SQL output with no filesystem dependencies.
Optimized for bulk inserts, transaction batching, and clean error handling.
"""

from __future__ import annotations

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

import pandas as pd
import numpy as np

from core.observability import Console, Heartbeat


# SQL table whitelist
ALLOWED_TABLES = {
    'ACM_Scores_Wide', 'ACM_Episodes',
    'ACM_HealthTimeline', 'ACM_RegimeTimeline',
    'ACM_ContributionCurrent', 'ACM_ContributionTimeline',
    'ACM_DriftSeries', 'ACM_ThresholdCrossings',
    'ACM_SensorRanking', 'ACM_RegimeOccupancy',
    'ACM_HealthHistogram', 'ACM_RegimeStability',
    'ACM_DefectSummary', 'ACM_DefectTimeline', 'ACM_SensorDefects',
    'ACM_HealthZoneByPeriod', 'ACM_SensorAnomalyByPeriod',
    'ACM_DetectorCorrelation', 'ACM_CalibrationSummary',
    'ACM_RegimeTransitions', 'ACM_RegimeDwellStats',
    'ACM_DriftEvents', 'ACM_CulpritHistory', 'ACM_EpisodeMetrics',
    'ACM_DataQuality', 'ACM_Scores_Long', 'ACM_DriftSeries',
    'ACM_PCA_Models', 'ACM_PCA_Loadings', 'ACM_PCA_Metrics',
    'ACM_Run_Stats', 'ACM_SinceWhen',
    'ACM_SensorHotspots', 'ACM_SensorHotspotTimeline',
    'ACM_HealthForecast_TS', 'ACM_FailureForecast_TS',
    'ACM_RUL_TS', 'ACM_RUL_Summary', 'ACM_RUL_Attribution',
    'ACM_SensorForecast_TS', 'ACM_MaintenanceRecommendation',
    'ACM_EnhancedFailureProbability_TS', 'ACM_FailureCausation',
    'ACM_EnhancedMaintenanceRecommendation', 'ACM_RecommendedActions',
    'ACM_SensorNormalized_TS', 'ACM_EpisodeDiagnostics',
}


# Health calculation helper
def _health_index(fused_z, z_threshold: float = 5.0, steepness: float = 1.5):
    """
    Calculate health index from fused z-score using softer sigmoid formula.
    
    v10.1.0: Replaced overly aggressive 100/(1+Z^2) formula.
    
    Args:
        fused_z: Fused z-score (scalar, array, or Series)
        z_threshold: Z-score at which health should be very low (default 5.0)
        steepness: Controls sigmoid slope (default 1.5)
    
    Returns:
        Health index 0-100
    """
    import numpy as np
    abs_z = np.abs(fused_z)
    normalized = (abs_z - z_threshold / 2) / (z_threshold / 4)
    sigmoid = 1 / (1 + np.exp(-normalized * steepness))
    return np.clip(100.0 * (1 - sigmoid), 0.0, 100.0)


# Timestamp helpers
def _to_naive(ts) -> Optional[pd.Timestamp]:
    """Convert to timezone-naive timestamp."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    try:
        result = pd.to_datetime(ts, errors='coerce')
        if hasattr(result, 'tz') and result.tz is not None:
            return result.tz_localize(None)
        return result
    except Exception:
        return None


def _to_naive_series(series) -> pd.Series:
    """Convert Series/Index to timezone-naive timestamps."""
    s = pd.to_datetime(series, errors='coerce')
    if isinstance(s, pd.Series):
        if hasattr(s.dt, 'tz') and s.dt.tz is not None:
            return s.dt.tz_localize(None)
        return s
    else:  # DatetimeIndex
        if hasattr(s, 'tz') and s.tz is not None:
            idx = s.tz_localize(None)
        else:
            idx = s
        return pd.Series(idx, index=idx)


@dataclass
class SQLWriteStats:
    """Track SQL write statistics."""
    writes: int = 0
    rows: int = 0
    failures: int = 0
    write_time: float = 0.0


class SQLOutputManager:
    """
    Lean SQL-only output manager.
    
    Features:
    - Pure SQL writes (no filesystem dependencies)
    - Batched transactions for performance
    - Automatic schema repair with NOT NULL defaults
    - Query-based data retrieval (replaces artifact cache)
    """
    
    def __init__(self,
                 sql_client,
                 run_id: str,
                 equip_id: int,
                 batch_size: int = 10000,
                 max_retries: int = 3):
        """
        Initialize SQL output manager.
        
        Args:
            sql_client: Database connection (pyodbc or similar)
            run_id: Unique run identifier
            equip_id: Equipment ID
            batch_size: Rows per batch for bulk insert
            max_retries: Retry attempts for failed writes
        """
        self.sql_client = sql_client
        self.run_id = run_id
        self.equip_id = equip_id
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        self._in_transaction = False
        self.stats = SQLWriteStats()
        
        # Schema defaults for NOT NULL columns
        self._required_defaults: Dict[str, Dict[str, Any]] = {
            'ACM_HealthTimeline': {
                'Timestamp': 'ts', 'HealthIndex': 0.0, 'HealthZone': 'GOOD', 'FusedZ': 0.0
            },
            'ACM_DefectTimeline': {
                'Timestamp': 'ts', 'FusedZ': 0.0, 'HealthIndex': 0.0, 'HealthZone': 'UNKNOWN',
                'EventType': 'ZONE_CHANGE', 'FromZone': 'START', 'ToZone': 'GOOD'
            },
            'ACM_ThresholdCrossings': {
                'Timestamp': 'ts', 'DetectorType': 'fused', 'ZScore': 0.0, 
                'Threshold': 0.0, 'Direction': 'up'
            },
            # Add more as needed...
        }
        
        Console.info(f"[SQL] OutputManager initialized (batch_size={batch_size}, "
                    f"RunID={run_id}, EquipID={equip_id})")
    
    @contextmanager
    def transaction(self):
        """Context manager for batched SQL transaction."""
        if self._in_transaction:
            yield  # Nested - just pass through
            return
        
        self._in_transaction = True
        start = time.time()
        
        try:
            Console.info("Starting transaction", component="SQL")
            yield
            self._commit()
            elapsed = time.time() - start
            Console.info(f"Transaction committed ({elapsed:.2f}s)", component="SQL")
        except Exception as e:
            self._rollback()
            Console.error(f"Transaction rolled back: {e}", component="SQL")
            raise
        finally:
            self._in_transaction = False
    
    def _commit(self):
        """Commit current transaction."""
        try:
            if hasattr(self.sql_client, "commit"):
                self.sql_client.commit()
            elif hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                if not getattr(self.sql_client.conn, "autocommit", True):
                    self.sql_client.conn.commit()
        except Exception as e:
            Console.error(f"Commit failed: {e}", component="SQL")
            raise
    
    def _rollback(self):
        """Rollback current transaction."""
        try:
            if hasattr(self.sql_client, "rollback"):
                self.sql_client.rollback()
            elif hasattr(self.sql_client, "conn"):
                self.sql_client.conn.rollback()
        except Exception:
            pass
    
    def write_table(self, 
                   table_name: str, 
                   df: pd.DataFrame,
                   add_metadata: bool = True) -> int:
        """
        Write DataFrame to SQL table.
        
        Args:
            table_name: SQL table name (must be in ALLOWED_TABLES)
            df: DataFrame to write
            add_metadata: Add RunID, EquipID, CreatedAt columns
            
        Returns:
            Number of rows written
        """
        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"Table '{table_name}' not in whitelist")
        
        if df.empty:
            Console.warn(f"Empty DataFrame for {table_name}, skipping", component="SQL")
            return 0
        
        start = time.time()
        
        try:
            # Prepare DataFrame
            sql_df = self._prepare_dataframe(df, table_name, add_metadata)
            
            # Bulk insert with retry
            rows_written = self._bulk_insert_with_retry(table_name, sql_df)
            
            # Update stats
            elapsed = time.time() - start
            self.stats.writes += 1
            self.stats.rows += rows_written
            self.stats.write_time += elapsed
            
            Console.info(f"Wrote {rows_written} rows to {table_name} ({elapsed:.2f}s)", component="SQL")
            return rows_written
            
        except Exception as e:
            self.stats.failures += 1
            Console.error(f"Failed to write {table_name}: {e}", component="SQL")
            raise
    
    def _prepare_dataframe(self, 
                          df: pd.DataFrame,
                          table_name: str,
                          add_metadata: bool) -> pd.DataFrame:
        """Prepare DataFrame for SQL insertion."""
        out = df.copy()
        
        # Add metadata
        if add_metadata:
            out["RunID"] = self.run_id
            out["EquipID"] = self.equip_id
            out["CreatedAt"] = pd.Timestamp.now().tz_localize(None)
        
        # Apply schema defaults
        out = self._apply_defaults(table_name, out)
        
        # Clean timestamps
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col]).dt.tz_localize(None)
        
        # Replace inf/NaN
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], None)
        out = out.where(pd.notnull(out), None)
        
        return out
    
    def _apply_defaults(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NOT NULL defaults to required columns."""
        defaults = self._required_defaults.get(table_name, {})
        if not defaults:
            return df
        
        out = df.copy()
        sentinel_ts = pd.Timestamp(year=1900, month=1, day=1)
        
        for col, default in defaults.items():
            if col not in out.columns:
                # Add missing column
                val = sentinel_ts if default == 'ts' else default
                out[col] = val
            else:
                # Fill NULLs
                if default == 'ts':
                    out[col] = out[col].fillna(sentinel_ts)
                else:
                    out[col] = out[col].fillna(default)
        
        return out
    
    def _bulk_insert_with_retry(self, table_name: str, df: pd.DataFrame) -> int:
        """Bulk insert with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                return self._bulk_insert(table_name, df)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                Console.warn(f"[SQL] Insert failed (attempt {attempt + 1}/{self.max_retries}), "
                           f"retrying in {wait}s: {e}")
                time.sleep(wait)
        return 0
    
    def _bulk_insert(self, table_name: str, df: pd.DataFrame) -> int:
        """Execute bulk SQL insert."""
        cursor = self.sql_client.cursor()
        inserted = 0
        
        try:
            # Enable fast executemany if available (pyodbc)
            try:
                cursor.fast_executemany = True
            except AttributeError:
                pass
            
            # Get column list
            columns = df.columns.tolist()
            col_str = ", ".join(f"[{c}]" for c in columns)
            placeholders = ", ".join(["?"] * len(columns))
            insert_sql = f"INSERT INTO dbo.[{table_name}] ({col_str}) VALUES ({placeholders})"
            
            # Convert to records
            records = [tuple(row) for row in df.itertuples(index=False, name=None)]
            
            # Batch insert
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                cursor.executemany(insert_sql, batch)
                inserted += len(batch)
            
            # Commit if not in transaction
            if not self._in_transaction:
                self._commit()
            
            return inserted
            
        finally:
            cursor.close()
    
    def read_table(self, 
                  table_name: str,
                  filters: Optional[Dict[str, Any]] = None,
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read DataFrame from SQL table.
        
        Args:
            table_name: SQL table name
            filters: Optional WHERE clause filters (e.g., {'RunID': run_id})
            columns: Optional column list (default: SELECT *)
            
        Returns:
            DataFrame with query results
        """
        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"Table '{table_name}' not in whitelist")
        
        # Build query
        col_str = "*" if not columns else ", ".join(f"[{c}]" for c in columns)
        query = f"SELECT {col_str} FROM dbo.[{table_name}]"
        
        # Add filters
        params = []
        if filters:
            where_clauses = []
            for col, val in filters.items():
                where_clauses.append(f"[{col}] = ?")
                params.append(val)
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Execute query
        try:
            return pd.read_sql_query(query, self.sql_client, params=params or None)
        except Exception as e:
            Console.error(f"Failed to read {table_name}: {e}", component="SQL")
            raise
    
    def get_run_data(self, table_name: str) -> pd.DataFrame:
        """Convenience method to get data for current run."""
        return self.read_table(
            table_name,
            filters={'RunID': self.run_id, 'EquipID': self.equip_id}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'writes': self.stats.writes,
            'rows': self.stats.rows,
            'failures': self.stats.failures,
            'write_time': self.stats.write_time,
            'avg_time_per_write': self.stats.write_time / max(1, self.stats.writes),
            'success_rate': 1.0 - (self.stats.failures / max(1, self.stats.writes))
        }
    
    # ==================== HIGH-LEVEL WRITE METHODS ====================
    
    def write_scores(self, scores_df: pd.DataFrame) -> int:
        """Write scores to ACM_Scores_Wide table."""
        out = scores_df.copy()
        out.index.name = "Timestamp"
        out = out.reset_index()
        out['Timestamp'] = _to_naive_series(out['Timestamp'])
        return self.write_table("ACM_Scores_Wide", out)
    
    def write_episodes(self, episodes_df: pd.DataFrame) -> int:
        """Write episodes summary to ACM_Episodes table."""
        if episodes_df.empty:
            return 0
        
        out = episodes_df.copy()
        
        # Rename columns to match SQL schema
        col_map = {
            'start_ts': 'StartTs',
            'end_ts': 'EndTs',
            'duration_hours': 'DurationHours',
            'duration_s': 'DurationSeconds',
            'len': 'RecordCount',
            'peak_fused_z': 'PeakFusedZ',
            'avg_fused_z': 'AvgFusedZ',
            'min_health_index': 'MinHealthIndex',
            'culprits': 'Culprits',
            'severity': 'Severity',
            'status': 'Status'
        }
        out = out.rename(columns=col_map)
        
        # Add defaults
        if 'Severity' not in out.columns:
            out['Severity'] = 'MEDIUM'
        if 'Status' not in out.columns:
            out['Status'] = 'CLOSED'
        
        # Convert DurationSeconds to DurationHours if missing
        if 'DurationHours' not in out.columns and 'DurationSeconds' in out.columns:
            out['DurationHours'] = out['DurationSeconds'] / 3600.0
        
        return self.write_table("ACM_Episodes", out)
    
    def write_health_timeline(self, scores_df: pd.DataFrame) -> int:
        """Generate and write health timeline."""
        health_index = _health_index(scores_df['fused'])
        zones = pd.cut(
            health_index,
            bins=[0, 70, 85, 100],
            labels=['ALERT', 'WATCH', 'GOOD']
        )
        
        df = pd.DataFrame({
            'Timestamp': _to_naive_series(scores_df.index),
            'HealthIndex': health_index.round(2),
            'HealthZone': zones.astype(str),
            'FusedZ': scores_df['fused'].round(4)
        })
        
        return self.write_table("ACM_HealthTimeline", df)
    
    def write_regime_timeline(self, scores_df: pd.DataFrame) -> int:
        """Generate and write regime timeline."""
        if 'regime_label' not in scores_df.columns:
            return 0
        
        df = pd.DataFrame({
            'Timestamp': _to_naive_series(scores_df.index),
            'RegimeLabel': pd.to_numeric(scores_df['regime_label'], errors='coerce').astype('Int64'),
            'RegimeState': scores_df.get('regime_state', 'unknown').astype(str)
        })
        
        return self.write_table("ACM_RegimeTimeline", df)
    
    def write_all_analytics(self, 
                           scores_df: pd.DataFrame,
                           episodes_df: pd.DataFrame) -> Dict[str, int]:
        """
        Write all analytics tables in a single transaction.
        
        Returns:
            Dictionary of {table_name: rows_written}
        """
        results = {}
        
        with self.transaction():
            # Core tables
            results['ACM_Scores_Wide'] = self.write_scores(scores_df)
            results['ACM_Episodes'] = self.write_episodes(episodes_df)
            results['ACM_HealthTimeline'] = self.write_health_timeline(scores_df)
            
            # Regime tables (if available)
            if 'regime_label' in scores_df.columns:
                results['ACM_RegimeTimeline'] = self.write_regime_timeline(scores_df)
                results['ACM_RegimeOccupancy'] = self._write_regime_occupancy(scores_df)
            
            # Detector analysis
            results['ACM_DetectorCorrelation'] = self._write_detector_correlation(scores_df)
            results['ACM_CalibrationSummary'] = self._write_calibration_summary(scores_df)
            
            # Episode analysis
            if not episodes_df.empty:
                results['ACM_EpisodeMetrics'] = self._write_episode_metrics(episodes_df)
        
        total_rows = sum(results.values())
        Console.info(f"Wrote {len(results)} tables, {total_rows} total rows", component="SQL")
        return results
    
    # ==================== ANALYTICS TABLE GENERATORS ====================
    
    def _write_regime_occupancy(self, scores_df: pd.DataFrame) -> int:
        """Generate and write regime occupancy stats."""
        regimes = pd.to_numeric(scores_df['regime_label'], errors='coerce').dropna().astype('Int64')
        if regimes.empty:
            return 0
        
        counts = regimes.value_counts().sort_index()
        total = len(regimes)
        
        df = pd.DataFrame({
            'RegimeLabel': counts.index,
            'RecordCount': counts.values,
            'Percentage': (counts.values / total * 100).round(2)
        })
        
        return self.write_table("ACM_RegimeOccupancy", df)
    
    def _write_detector_correlation(self, scores_df: pd.DataFrame) -> int:
        """Generate and write detector correlation matrix."""
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        if len(detector_cols) < 2:
            return 0
        
        correlations = []
        for i, det_a in enumerate(detector_cols):
            for det_b in detector_cols[i+1:]:
                r = scores_df[det_a].corr(scores_df[det_b])
                correlations.append({
                    'DetectorA': det_a,
                    'DetectorB': det_b,
                    'PearsonR': round(r, 4) if not pd.isna(r) else 0.0
                })
        
        if not correlations:
            return 0
        
        return self.write_table("ACM_DetectorCorrelation", pd.DataFrame(correlations))
    
    def _write_calibration_summary(self, scores_df: pd.DataFrame) -> int:
        """Generate and write detector calibration summary."""
        detector_cols = [c for c in scores_df.columns if c.endswith('_z') and c != 'fused_z']
        if not detector_cols:
            return 0
        
        calibration = []
        for detector in detector_cols:
            values = scores_df[detector].abs()
            calibration.append({
                'DetectorType': detector,
                'MeanZ': round(values.mean(), 4),
                'StdZ': round(values.std(), 4),
                'P95Z': round(values.quantile(0.95), 4),
                'P99Z': round(values.quantile(0.99), 4),
                'ClipZ': 30.0,  # Standard clip threshold
                'SaturationPct': round((values >= 30.0).mean() * 100, 2)
            })
        
        return self.write_table("ACM_CalibrationSummary", pd.DataFrame(calibration))
    
    def _write_episode_metrics(self, episodes_df: pd.DataFrame) -> int:
        """Generate and write episode statistical metrics."""
        if episodes_df.empty:
            return 0
        
        # Calculate durations
        durations = []
        for _, ep in episodes_df.iterrows():
            duration_hours = ep.get('duration_hours', 0.0)
            if duration_hours == 0 and 'duration_s' in ep:
                duration_hours = ep['duration_s'] / 3600.0
            if duration_hours > 0:
                durations.append(duration_hours)
        
        if not durations:
            return 0
        
        total_duration = sum(durations)
        observation_days = max(30, total_duration / 24)
        
        metrics = pd.DataFrame([{
            'TotalEpisodes': len(episodes_df),
            'TotalDurationHours': round(total_duration, 1),
            'AvgDurationHours': round(np.mean(durations), 1),
            'MedianDurationHours': round(np.median(durations), 1),
            'MaxDurationHours': round(max(durations), 1),
            'MinDurationHours': round(min(durations), 1),
            'RatePerDay': round(len(episodes_df) / observation_days, 3),
            'MeanInterarrivalHours': round(total_duration / (len(episodes_df) - 1), 1) if len(episodes_df) > 1 else 0
        }])
        
        return self.write_table("ACM_EpisodeMetrics", metrics)


# Factory function
def create_output_manager(sql_client, run_id: str, equip_id: int, **kwargs) -> SQLOutputManager:
    """Create SQL output manager instance."""
    return SQLOutputManager(
        sql_client=sql_client,
        run_id=run_id,
        equip_id=equip_id,
        **kwargs
    )
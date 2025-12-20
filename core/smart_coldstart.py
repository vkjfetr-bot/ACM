"""
ACM Smart Coldstart Module

Implements intelligent coldstart retry logic that:
1. Detects insufficient data without failing
2. Accumulates data over multiple job runs
3. Auto-detects data cadence and calculates required lookback
4. Retries until sufficient data exists for model training
5. Never falls back to file mode

Author: Copilot
Date: November 13, 2025
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import pandas as pd
from core.observability import Console, Heartbeat


class ColdstartState:
    """Represents the current coldstart state for an equipment."""
    
    def __init__(self, equip_id: int, stage: str = 'score'):
        self.equip_id = equip_id
        self.stage = stage
        self.needs_coldstart = True
        self.attempt_count = 0
        self.accumulated_rows = 0
        self.required_rows = 500
        self.data_start_time: Optional[datetime] = None
        self.data_end_time: Optional[datetime] = None
        self.last_error: Optional[str] = None
        
    def is_ready(self) -> bool:
        """Check if sufficient data has been accumulated for coldstart."""
        return self.accumulated_rows >= self.required_rows
    
    def __repr__(self):
        return (f"ColdstartState(equip={self.equip_id}, attempts={self.attempt_count}, "
                f"rows={self.accumulated_rows}/{self.required_rows}, ready={self.is_ready()})")


class SmartColdstart:
    """
    Smart coldstart manager that handles data accumulation and retry logic.
    
    Key Features:
    - Auto-detects data cadence from histogram
    - Calculates optimal lookback window
    - Tracks progress across multiple job runs
    - Retries with exponential window expansion
    - Never fails - always defers to next run
    """
    
    def __init__(self, sql_client, equip_id: int, equip_name: str, stage: str = 'score'):
        self.sql_client = sql_client
        self.equip_id = equip_id
        self.equip_name = equip_name
        self.stage = stage
        self.state: Optional[ColdstartState] = None
        
    def check_status(self, required_rows: int = 500, tick_minutes: Optional[int] = None) -> ColdstartState:
        """
        Check current coldstart status from database.
        
        Args:
            required_rows: Minimum rows needed to complete coldstart
            tick_minutes: Current job frequency in minutes (auto-detected if None)
            
        Returns:
            ColdstartState with current status
        """
        try:
            # Auto-detect tick_minutes from data cadence if not provided
            if tick_minutes is None:
                table_name = f"{self.equip_name}_Data"
                data_cadence_seconds = self.detect_data_cadence(table_name)
                if data_cadence_seconds:
                    tick_minutes = int(data_cadence_seconds / 60)
                    Console.info(f"Auto-detected tick_minutes from data cadence: {tick_minutes} minutes", component="COLDSTART")
                else:
                    tick_minutes = 30  # Default fallback
                    Console.warn(f"Could not detect cadence, using default tick_minutes: {tick_minutes}", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, default_tick_minutes=tick_minutes)
            
            cur = self.sql_client.cursor()
            
            # Call stored procedure to check status
            needs_coldstart = cur.execute(
                "DECLARE @NeedsColdstart BIT, @AccumulatedRows INT, @AttemptCount INT; "
                "EXEC dbo.usp_ACM_CheckColdstartStatus @EquipID=?, @Stage=?, @RequiredRows=?, @TickMinutes=?, "
                "@NeedsColdstart=@NeedsColdstart OUTPUT, @AccumulatedRows=@AccumulatedRows OUTPUT, @AttemptCount=@AttemptCount OUTPUT; "
                "SELECT @NeedsColdstart, @AccumulatedRows, @AttemptCount",
                (self.equip_id, self.stage, required_rows, tick_minutes)
            ).fetchone()
            
            self.sql_client.conn.commit()
            
            if needs_coldstart:
                needs, accumulated, attempts = needs_coldstart
                self.state = ColdstartState(self.equip_id, self.stage)
                self.state.needs_coldstart = bool(needs)
                self.state.accumulated_rows = accumulated or 0
                self.state.attempt_count = attempts or 0
                self.state.required_rows = required_rows
            else:
                # Coldstart complete
                self.state = ColdstartState(self.equip_id, self.stage)
                self.state.needs_coldstart = False
                self.state.accumulated_rows = required_rows  # Mark as complete
                
            return self.state
            
        except Exception as e:
            Console.error(f"Failed to check status: {e}", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, stage=self.stage, error_type=type(e).__name__, error=str(e)[:200])
            # Default to needing coldstart on error
            self.state = ColdstartState(self.equip_id, self.stage)
            self.state.required_rows = required_rows
            return self.state
        finally:
            try:
                cur.close()
            except:
                pass
    
    def detect_data_cadence(self, table_name: str, sample_hours: int = 24) -> Optional[int]:
        """
        Auto-detect data cadence by analyzing timestamp intervals.
        
        Args:
            table_name: Equipment data table name (e.g., 'FD_FAN_Data')
            sample_hours: Hours of data to sample for cadence detection
            
        Returns:
            Detected cadence in seconds, or None if detection failed
        """
        try:
            cur = self.sql_client.cursor()
            
            # Get sample of timestamps to detect cadence
            query = f"""
            SELECT TOP 100 EntryDateTime
            FROM dbo.{table_name}
            ORDER BY EntryDateTime
            """
            
            cur.execute(query)
            rows = cur.fetchall()
            
            if len(rows) < 10:
                Console.warn(f"Insufficient data for cadence detection: {len(rows)} rows", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, table=table_name, rows_found=len(rows), min_required=10)
                return None
            
            # Calculate intervals between consecutive timestamps
            timestamps = [row[0] for row in rows]
            intervals = []
            for i in range(1, len(timestamps)):
                delta = (timestamps[i] - timestamps[i-1]).total_seconds()
                if delta > 0:  # Skip duplicates
                    intervals.append(delta)
            
            if not intervals:
                return None
            
            # Find most common interval (mode)
            from collections import Counter
            interval_counts = Counter(intervals)
            most_common_interval = interval_counts.most_common(1)[0][0]
            
            Console.info(f"Detected data cadence: {most_common_interval} seconds ({most_common_interval/60:.1f} minutes)", component="COLDSTART")
            
            return int(most_common_interval)
            
        except Exception as e:
            Console.error(f"Failed to detect cadence: {e}", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, table=table_name, sample_hours=sample_hours, error_type=type(e).__name__, error=str(e)[:200])
            return None
        finally:
            try:
                cur.close()
            except:
                pass
    
    def calculate_optimal_window(self, 
                                  current_window_end: datetime,
                                  required_rows: int = 500,
                                  data_cadence_seconds: Optional[int] = None) -> Tuple[datetime, datetime]:
        """
        Calculate optimal lookback window to get required_rows of data.
        For coldstart, we want to load the EARLIEST available data, not recent data.
        
        Args:
            current_window_end: End time for the data window (batch end time) - not used for coldstart
            required_rows: Target number of rows needed
            data_cadence_seconds: Detected data cadence, or None to auto-detect
            
        Returns:
            Tuple of (start_time, end_time) for expanded window
        """
        # Auto-detect cadence if not provided
        if data_cadence_seconds is None:
            table_name = f"{self.equip_name}_Data"
            data_cadence_seconds = self.detect_data_cadence(table_name)
            
            if data_cadence_seconds is None:
                # Fallback: assume 1 minute cadence
                data_cadence_seconds = 60
                Console.warn(f"Could not detect cadence, assuming {data_cadence_seconds}s", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, table=table_name, default_cadence_s=data_cadence_seconds)
        
        # Calculate how many minutes needed to get required_rows
        cadence_minutes = data_cadence_seconds / 60
        required_minutes = required_rows * cadence_minutes
        
        # Add 20% buffer for safety
        required_minutes = int(required_minutes * 1.2)
        
        # For coldstart, get the EARLIEST data available
        # Query the database for the earliest timestamp
        table_name = f"{self.equip_name}_Data"
        cur = None
        try:
            cur = self.sql_client.conn.cursor()
            query = f"SELECT MIN(EntryDateTime) FROM {table_name}"
            cur.execute(query)
            row = cur.fetchone()
            
            if row and row[0]:
                start_time = row[0]
                # Add required minutes to get end time
                end_time = start_time + timedelta(minutes=required_minutes)
                
                Console.info(f"Loading from EARLIEST data: {start_time}", component="COLDSTART")
                Console.info(f"Calculated optimal window: {required_minutes} minutes ({required_minutes/60:.1f} hours)", component="COLDSTART")
                Console.info(f"Expected rows: ~{int(required_minutes / cadence_minutes)} (target: {required_rows})", component="COLDSTART")
                
                return start_time, end_time
            else:
                # Fallback: use lookback from current time if no data found
                Console.warn(f"No data found in {table_name}, using lookback from current batch", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, table=table_name, required_rows=required_rows, lookback_minutes=required_minutes)
                end_time = current_window_end
                start_time = end_time - timedelta(minutes=required_minutes)
                return start_time, end_time
                
        except Exception as e:
            Console.error(f"Error querying earliest timestamp: {e}", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, table=table_name, required_rows=required_rows, error_type=type(e).__name__, error=str(e)[:200])
            # Fallback: lookback from current time
            end_time = current_window_end
            start_time = end_time - timedelta(minutes=required_minutes)
            return start_time, end_time
        finally:
            if cur:
                try:
                    cur.close()
                except:
                    pass
    
    def load_with_retry(self, 
                       output_manager,
                       cfg: Dict[str, Any],
                       initial_start: datetime,
                       initial_end: datetime,
                       max_attempts: int = 3,
                       historical_replay: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Any], bool]:
        """
        Attempt to load data with intelligent retry and window expansion.
        
        Args:
            output_manager: OutputManager instance for data loading
            cfg: Configuration dictionary
            initial_start: Initial window start time
            initial_end: Initial window end time
            max_attempts: Maximum retry attempts per job run
            historical_replay: If True, expand window forward in time instead of backward
            
        Returns:
            Tuple of (train_df, score_df, meta, coldstart_complete)
            If coldstart not ready, returns (None, None, None, False)
        """
        min_rows = cfg.get("data", {}).get("min_train_samples", 500)
        
        # Detect data cadence for tick_minutes calculation
        table_name = f"{self.equip_name}_Data"
        data_cadence_seconds = self.detect_data_cadence(table_name)
        detected_tick_minutes = None
        if data_cadence_seconds:
            detected_tick_minutes = int(data_cadence_seconds / 60)
        
        # Check coldstart status with detected tick_minutes
        state = self.check_status(required_rows=min_rows, tick_minutes=detected_tick_minutes)
        
        if not state.needs_coldstart:
            Console.info(f"Models exist for {self.equip_name}, coldstart not needed", component="COLDSTART")
            # Load with original window for incremental scoring (ALL data goes to score)
            # NOTE: If data insufficient for this batch window, gracefully return NOOP
            try:
                train, score, meta, complete = self._load_data_window(output_manager, cfg, initial_start, initial_end, coldstart_complete=True, is_coldstart=False)
                if train is None and score is None:
                    # Insufficient data in this window - not an error, just skip this batch
                    window_hours = (initial_end - initial_start).total_seconds() / 3600
                    Console.warn(f"Insufficient data in {window_hours:.1f}h window - batch will NOOP (models exist but no new data)", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, window_hours=round(window_hours, 1), start_time=str(initial_start), end_time=str(initial_end))
                    return None, None, None, False
                return train, score, meta, complete
            except ValueError as e:
                # Data insufficient - expected when batch windows are smaller than min samples
                window_hours = (initial_end - initial_start).total_seconds() / 3600
                Console.warn(f"Insufficient data in {window_hours:.1f}h window: {e}", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, window_hours=round(window_hours, 1), error=str(e)[:200])
                Console.info(f"Batch will NOOP (models exist but no new data to score)", component="COLDSTART")
                return None, None, None, False
        
        Console.info(f"Status: {state}", component="COLDSTART")
        
        # Always calculate optimal window to ensure we load enough data
        # The optimal window calculates from EARLIEST available data forward
        if state.attempt_count == 0:
            Console.info(f"First coldstart attempt - calculating optimal window", component="COLDSTART")
        else:
            Console.info(f"Resuming coldstart (previous attempts: {state.attempt_count})", component="COLDSTART")
        
        # Calculate optimal window that should contain enough data
        optimal_start, optimal_end = self.calculate_optimal_window(
            current_window_end=initial_end,
            required_rows=min_rows,
            data_cadence_seconds=data_cadence_seconds
        )
        attempt_start = optimal_start
        attempt_end = optimal_end
        
        for attempt in range(1, max_attempts + 1):
            try:
                window_hours = (attempt_end - attempt_start).total_seconds() / 3600
                Console.info(f"Attempt {attempt}/{max_attempts}: Loading {window_hours:.1f} hours [{attempt_start} to {attempt_end}]", component="COLDSTART")
                
                # Try to load data WITH COLDSTART SPLIT
                train, score, meta = output_manager._load_data_from_sql(
                    cfg, self.equip_name, attempt_start, attempt_end, is_coldstart=True
                )
                
                rows_loaded = len(train) + len(score) if train is not None and score is not None else 0
                
                # Update progress in database
                self._update_progress(
                    rows_received=rows_loaded,
                    data_start=attempt_start,
                    data_end=attempt_end,
                    success=(rows_loaded >= min_rows)
                )
                
                if rows_loaded >= min_rows:
                    Console.info(f"SUCCESS! Loaded {rows_loaded} rows (required: {min_rows})", component="COLDSTART")
                    return train, score, meta, True
                
                else:
                    Console.warn(f"Insufficient data: {rows_loaded} rows (required: {min_rows})", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, rows_loaded=rows_loaded, min_required=min_rows, attempt=attempt, max_attempts=max_attempts)
                    
                    # Expand window for next attempt
                    if attempt < max_attempts:
                        # Exponential expansion: double the window
                        window_size = (attempt_end - attempt_start).total_seconds() / 60
                        new_window_size = window_size * 2
                        if historical_replay:
                            # Historical replay: expand forward in time
                            attempt_end = attempt_start + timedelta(minutes=new_window_size)
                        else:
                            # Live mode: expand backward in time
                            attempt_start = attempt_end - timedelta(minutes=new_window_size)
                        Console.info(f"Expanding window to {new_window_size:.0f} minutes ({new_window_size/60:.1f} hours) for retry", component="COLDSTART")
                
            except ValueError as e:
                # Data insufficient error - expected during coldstart
                error_msg = str(e)
                Console.warn(f"Attempt {attempt} failed: {error_msg}", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, attempt=attempt, max_attempts=max_attempts, error_type="ValueError", error=error_msg[:200])
                
                self._update_progress(
                    rows_received=0,
                    data_start=attempt_start,
                    data_end=attempt_end,
                    error_message=error_msg
                )
                
                # Expand window for next attempt
                if attempt < max_attempts:
                    window_size = (attempt_end - attempt_start).total_seconds() / 60
                    new_window_size = window_size * 2
                    if historical_replay:
                        # Historical replay: expand forward in time
                        attempt_end = attempt_start + timedelta(minutes=new_window_size)
                    else:
                        # Live mode: expand backward in time
                        attempt_start = attempt_end - timedelta(minutes=new_window_size)
                
            except Exception as e:
                # Unexpected error
                Console.error(f"Unexpected error on attempt {attempt}: {e}", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, attempt=attempt, max_attempts=max_attempts, error_type=type(e).__name__, error=str(e)[:200])
                self._update_progress(
                    rows_received=0,
                    data_start=attempt_start,
                    data_end=attempt_end,
                    error_message=str(e)
                )
                break
        
        # All attempts failed - defer to next job run
        Console.info(f"Deferred - will retry on next job run (attempt {state.attempt_count + 1})", component="COLDSTART")
        Console.info(f"Progress: {state.accumulated_rows}/{state.required_rows} rows accumulated", component="COLDSTART")
        
        return None, None, None, False
    
    def _load_data_window(self, output_manager, cfg, start, end, coldstart_complete=False, is_coldstart=False):
        """Helper to load data for a specific window."""
        try:
            train, score, meta = output_manager._load_data_from_sql(cfg, self.equip_name, start, end, is_coldstart=is_coldstart)
            return train, score, meta, coldstart_complete
        except Exception as e:
            Console.error(f"Failed to load data window: {e}", component="COLDSTART", equip_id=self.equip_id, equip_name=self.equip_name, start_time=str(start), end_time=str(end), is_coldstart=is_coldstart, error_type=type(e).__name__, error=str(e)[:200])
            return None, None, None, False
    
    def _update_progress(self, 
                        rows_received: int,
                        data_start: datetime,
                        data_end: datetime,
                        error_message: Optional[str] = None,
                        success: bool = False):
        """Update coldstart progress in database."""
        try:
            cur = self.sql_client.cursor()
            cur.execute(
                "EXEC dbo.usp_ACM_UpdateColdstartProgress @EquipID=?, @Stage=?, @RowsReceived=?, "
                "@DataStartTime=?, @DataEndTime=?, @ErrorMessage=?, @Success=?",
                (self.equip_id, self.stage, rows_received, data_start, data_end, error_message, success)
            )
            self.sql_client.conn.commit()
        except Exception as e:
            Console.error(f"Failed to update progress: {e}", component="COLDSTART", equip_id=self.equip_id, stage=self.stage, rows_received=rows_received, error_type=type(e).__name__, error=str(e)[:200])
        finally:
            try:
                cur.close()
            except:
                pass

"""High-Performance SQL Log Sink with Batched Writes.

This module provides a high-performance logging sink that:
1. Accepts structured log data directly (no regex parsing)
2. Buffers logs in memory
3. Writes to SQL in batches (configurable batch size and flush interval)
4. Uses background thread for non-blocking writes
5. Provides graceful shutdown with final flush

Usage:
    from core.sql_logger_v2 import BatchedSqlLogSink
    
    sink = BatchedSqlLogSink(sql_client, run_id, equip_id, batch_size=100, flush_interval_ms=1000)
    sink.log(level="INFO", message="Processing started", stage="init", step="load_data")
    sink.log(level="INFO", message="Loaded 500 rows", row_count=500, duration_ms=150.5)
    sink.close()  # Flushes remaining logs
"""
from __future__ import annotations

import atexit
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.logger import Console


@dataclass
class LogRecord:
    """Structured log record - no regex needed, data is typed."""
    timestamp: datetime
    level: str
    message: str
    module: Optional[str] = None
    event_type: Optional[str] = None
    stage: Optional[str] = None
    step_name: Optional[str] = None
    duration_ms: Optional[float] = None
    row_count: Optional[int] = None
    col_count: Optional[int] = None
    window_size: Optional[int] = None
    batch_start: Optional[datetime] = None
    batch_end: Optional[datetime] = None
    baseline_start: Optional[datetime] = None
    baseline_end: Optional[datetime] = None
    data_quality_metric: Optional[str] = None
    data_quality_value: Optional[float] = None
    leakage_flag: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class BatchedSqlLogSink:
    """High-performance SQL log sink with batched, non-blocking writes.
    
    Features:
    - Accepts structured data directly (no regex parsing overhead)
    - Buffers logs in memory queue
    - Background thread writes batches to SQL
    - Configurable batch size and flush interval
    - Thread-safe for concurrent logging
    - Graceful shutdown with final flush
    
    Performance characteristics:
    - Log calls are non-blocking (just queue append)
    - SQL writes happen in batches (e.g., 100 records at once)
    - Uses parameterized bulk INSERT for efficiency
    """
    
    def __init__(
        self,
        sql_client,
        run_id: Optional[str],
        equip_id: Optional[int],
        table: str = "ACM_RunLogs",
        batch_size: int = 100,
        flush_interval_ms: int = 2000,
        max_queue_size: int = 10000,
    ):
        """Initialize the batched SQL log sink.
        
        Args:
            sql_client: A DEDICATED SQLClient instance (not shared with main operations).
            run_id: The current run ID to tag all log entries.
            equip_id: The equipment ID to tag all log entries.
            table: The SQL table name for logs (default: ACM_RunLogs).
            batch_size: Number of records to batch before writing (default: 100).
            flush_interval_ms: Max time between flushes in ms (default: 2000).
            max_queue_size: Max queue size before dropping logs (default: 10000).
        """
        self._sql_client = sql_client
        self.run_id = run_id
        self.equip_id = equip_id
        self.table = table
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_ms / 1000.0
        self.max_queue_size = max_queue_size
        
        self._queue: queue.Queue[Optional[LogRecord]] = queue.Queue(maxsize=max_queue_size)
        self._closed = False
        self._shutdown_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None
        self._dropped_count = 0
        self._write_count = 0
        
        # Ensure table exists
        self._ensure_table()
        
        # Start background flush thread
        self._start_flush_thread()
        
        # Register cleanup on exit
        atexit.register(self.close)
    
    @property
    def sql_client(self):
        """Return the SQL client for logging operations."""
        return self._sql_client
    
    def _ensure_table(self) -> None:
        """Create the log table if it doesn't exist."""
        try:
            cur = self._sql_client.cursor()
            cur.execute(f"""
                IF NOT EXISTS (
                    SELECT 1 FROM sys.objects 
                    WHERE object_id = OBJECT_ID(N'dbo.{self.table}') AND type = N'U'
                )
                BEGIN
                    CREATE TABLE dbo.{self.table} (
                        LogID BIGINT IDENTITY(1,1) PRIMARY KEY,
                        RunID NVARCHAR(64) NULL,
                        EquipID INT NULL,
                        LoggedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
                        LoggedLocal DATETIMEOFFSET NULL,
                        LoggedLocalNaive DATETIME2 NULL,
                        Level NVARCHAR(16) NOT NULL,
                        Module NVARCHAR(128) NULL,
                        EventType NVARCHAR(32) NULL,
                        Stage NVARCHAR(64) NULL,
                        StepName NVARCHAR(128) NULL,
                        DurationMs FLOAT NULL,
                        [RowCount] INT NULL,
                        [ColCount] INT NULL,
                        [WindowSize] INT NULL,
                        BatchStart DATETIME2 NULL,
                        BatchEnd DATETIME2 NULL,
                        BaselineStart DATETIME2 NULL,
                        BaselineEnd DATETIME2 NULL,
                        DataQualityMetric NVARCHAR(64) NULL,
                        DataQualityValue FLOAT NULL,
                        LeakageFlag BIT NULL,
                        ParamsJson NVARCHAR(MAX) NULL,
                        Message NVARCHAR(4000) NOT NULL,
                        Context NVARCHAR(MAX) NULL
                    );
                    CREATE NONCLUSTERED INDEX IX_RunLogs_RunID ON dbo.{self.table}(RunID);
                    CREATE NONCLUSTERED INDEX IX_RunLogs_EquipID_LoggedAt ON dbo.{self.table}(EquipID, LoggedAt DESC);
                END
            """)
            cur.close()
            try:
                if hasattr(self._sql_client, "conn"):
                    self._sql_client.conn.commit()
            except Exception:
                pass
        except Exception as exc:
            print(f"[SQL_LOG] Warning: Failed to ensure log table: {exc}", file=sys.stderr)
    
    def _start_flush_thread(self) -> None:
        """Start the background flush thread."""
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True, name="SqlLogFlush")
        self._flush_thread.start()
    
    def _flush_loop(self) -> None:
        """Background thread that periodically flushes the log queue to SQL."""
        batch: List[LogRecord] = []
        last_flush = time.time()
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for next record with timeout
                try:
                    record = self._queue.get(timeout=0.1)
                    if record is None:
                        # Shutdown signal
                        break
                    batch.append(record)
                except queue.Empty:
                    pass
                
                # Flush if batch is full or interval elapsed
                now = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and (now - last_flush) >= self.flush_interval_s)
                )
                
                if should_flush:
                    self._write_batch(batch)
                    batch = []
                    last_flush = now
                    
            except Exception as exc:
                print(f"[SQL_LOG] Flush loop error: {exc}", file=sys.stderr)
        
        # Final flush on shutdown
        if batch:
            self._write_batch(batch)
    
    def _write_batch(self, batch: List[LogRecord]) -> None:
        """Write a batch of log records to SQL in a single transaction."""
        if not batch:
            return
            
        try:
            cur = self._sql_client.cursor()
            
            # Build bulk INSERT with parameterized values
            # Using individual INSERTs in a transaction is more reliable than table-valued params
            for record in batch:
                logged_local = datetime.now().astimezone()
                logged_local_naive = logged_local.replace(tzinfo=None)
                params_json = json.dumps(record.params, default=str) if record.params else None
                context_json = json.dumps(record.context, default=str) if record.context else None
                
                cur.execute(f"""
                    INSERT INTO dbo.{self.table}
                    (RunID, EquipID, LoggedAt, LoggedLocal, LoggedLocalNaive, Level, Module, 
                     EventType, Stage, StepName, DurationMs, [RowCount], [ColCount], WindowSize,
                     BatchStart, BatchEnd, BaselineStart, BaselineEnd, 
                     DataQualityMetric, DataQualityValue, LeakageFlag, ParamsJson, Message, Context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.run_id,
                    self.equip_id,
                    record.timestamp,
                    logged_local,
                    logged_local_naive,
                    record.level,
                    record.module,
                    record.event_type,
                    record.stage,
                    record.step_name,
                    record.duration_ms,
                    record.row_count,
                    record.col_count,
                    record.window_size,
                    record.batch_start,
                    record.batch_end,
                    record.baseline_start,
                    record.baseline_end,
                    record.data_quality_metric,
                    record.data_quality_value,
                    record.leakage_flag,
                    params_json,
                    record.message[:4000] if record.message else "",
                    context_json,
                ))
            
            cur.close()
            
            # Single commit for entire batch
            try:
                if hasattr(self._sql_client, "conn"):
                    self._sql_client.conn.commit()
            except Exception:
                pass
            
            self._write_count += len(batch)
            
        except Exception as exc:
            print(f"[SQL_LOG] Batch write error ({len(batch)} records): {exc}", file=sys.stderr)
    
    def log(
        self,
        level: str,
        message: str,
        *,
        module: Optional[str] = None,
        event_type: Optional[str] = None,
        stage: Optional[str] = None,
        step_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
        row_count: Optional[int] = None,
        col_count: Optional[int] = None,
        window_size: Optional[int] = None,
        batch_start: Optional[datetime] = None,
        batch_end: Optional[datetime] = None,
        baseline_start: Optional[datetime] = None,
        baseline_end: Optional[datetime] = None,
        data_quality_metric: Optional[str] = None,
        data_quality_value: Optional[float] = None,
        leakage_flag: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        **extra_context,
    ) -> None:
        """Log a structured record (non-blocking, queued for batch write).
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            module: Source module name
            event_type: Event type tag (e.g., TIMER, DATA, DETECTOR)
            stage: Pipeline stage (e.g., load, train, score, output)
            step_name: Specific step within stage
            duration_ms: Duration in milliseconds
            row_count: Number of rows processed
            col_count: Number of columns
            window_size: Window size for rolling operations
            batch_start: Batch window start time
            batch_end: Batch window end time
            baseline_start: Baseline window start time
            baseline_end: Baseline window end time
            data_quality_metric: Data quality metric name
            data_quality_value: Data quality metric value
            leakage_flag: Whether data leakage was detected
            params: Additional parameters as dict
            context: Additional context as dict
            **extra_context: Additional key-value pairs merged into context
        """
        if self._closed:
            return
        
        # Merge extra context
        if extra_context:
            context = {**(context or {}), **extra_context}
        
        record = LogRecord(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            module=module,
            event_type=event_type,
            stage=stage,
            step_name=step_name,
            duration_ms=duration_ms,
            row_count=row_count,
            col_count=col_count,
            window_size=window_size,
            batch_start=batch_start,
            batch_end=batch_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            data_quality_metric=data_quality_metric,
            data_quality_value=data_quality_value,
            leakage_flag=leakage_flag,
            params=params,
            context=context,
        )
        
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_count += 1
            if self._dropped_count % 100 == 1:
                print(f"[SQL_LOG] Warning: Log queue full, dropped {self._dropped_count} records", file=sys.stderr)
    
    def __call__(self, record: Dict[str, Any]) -> None:
        """Legacy sink interface for Console.add_sink() compatibility.
        
        Accepts old-style record dict and converts to structured log.
        Minimal parsing - just extract what's already in the record.
        """
        if self._closed:
            return
        
        # Skip ephemeral/progress messages
        context = record.get("context") or {}
        if isinstance(context, dict) and context.get("skip_sql"):
            return
        
        # Extract basic fields (no regex - just use what's provided)
        message = str(record.get("message", ""))
        level = record.get("level", "INFO")
        
        # Skip unstructured sub-logs (indented messages) unless warning+
        if message.startswith("  ") and level not in ("WARNING", "ERROR", "CRITICAL"):
            return
        
        # Extract structured fields from context (if caller provided them)
        self.log(
            level=level,
            message=message,
            module=record.get("module") or context.get("module"),
            event_type=context.get("event_type"),
            stage=context.get("stage") or context.get("pipeline_stage"),
            step_name=context.get("step") or context.get("step_name"),
            duration_ms=context.get("duration_ms"),
            row_count=context.get("row_count") or context.get("rows"),
            col_count=context.get("col_count") or context.get("cols"),
            window_size=context.get("window") or context.get("window_size"),
            batch_start=self._parse_dt(context.get("batch_start")),
            batch_end=self._parse_dt(context.get("batch_end")),
            baseline_start=self._parse_dt(context.get("baseline_start")),
            baseline_end=self._parse_dt(context.get("baseline_end")),
            data_quality_metric=context.get("data_quality_metric"),
            data_quality_value=context.get("data_quality_value"),
            leakage_flag=context.get("leakage_flag"),
            context=context,
        )
    
    def _parse_dt(self, value: Any) -> Optional[datetime]:
        """Best-effort parse of datetime value."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None
    
    def flush(self) -> None:
        """Force flush any pending logs (blocking)."""
        # Drain queue and write immediately
        batch: List[LogRecord] = []
        while True:
            try:
                record = self._queue.get_nowait()
                if record is not None:
                    batch.append(record)
            except queue.Empty:
                break
        
        if batch:
            self._write_batch(batch)
    
    def close(self) -> None:
        """Gracefully shutdown: flush remaining logs and stop background thread."""
        if self._closed:
            return
        self._closed = True
        
        # Signal shutdown
        self._shutdown_event.set()
        self._queue.put(None)  # Wake up flush thread
        
        # Wait for flush thread to finish
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        
        # Final stats
        if self._dropped_count > 0:
            print(f"[SQL_LOG] Warning: Dropped {self._dropped_count} log records due to queue overflow", file=sys.stderr)
        
        # Unregister atexit
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, int]:
        """Return logging statistics."""
        return {
            "written": self._write_count,
            "dropped": self._dropped_count,
            "queued": self._queue.qsize(),
        }

"""SQL log streaming helpers."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Optional

from utils.logger import Console


class SqlLogSink:
    """Structured log sink that writes Console output to dbo.ACM_RunLogs."""

    def __init__(self, sql_client, run_id: Optional[str], equip_id: Optional[int], table: str = "ACM_RunLogs"):
        self.sql_client = sql_client
        self.run_id = run_id
        self.equip_id = equip_id
        self.table = table
        self._ensure_table()

    def _ensure_column(self, cur, column: str, definition: str) -> None:
        """Add a column to the log table if it is missing."""
        cur.execute(
            f"""
            IF NOT EXISTS (
                SELECT 1 FROM sys.columns
                WHERE object_id = OBJECT_ID(N'dbo.{self.table}') AND name = N'{column}'
            )
            BEGIN
                ALTER TABLE dbo.{self.table} ADD [{column}] {definition};
            END
            """
        )

    def _ensure_table(self) -> None:
        """Create the run logs table if it does not exist."""
        try:
            cur = self.sql_client.cursor()
            cur.execute(
                f"""
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
                END
                """
            )
            # Backfill any missing columns on existing tables
            self._ensure_column(cur, "EventType", "NVARCHAR(32) NULL")
            self._ensure_column(cur, "Stage", "NVARCHAR(64) NULL")
            self._ensure_column(cur, "StepName", "NVARCHAR(128) NULL")
            self._ensure_column(cur, "DurationMs", "FLOAT NULL")
            self._ensure_column(cur, "RowCount", "INT NULL")
            self._ensure_column(cur, "ColCount", "INT NULL")
            self._ensure_column(cur, "WindowSize", "INT NULL")
            self._ensure_column(cur, "BatchStart", "DATETIME2 NULL")
            self._ensure_column(cur, "BatchEnd", "DATETIME2 NULL")
            self._ensure_column(cur, "BaselineStart", "DATETIME2 NULL")
            self._ensure_column(cur, "BaselineEnd", "DATETIME2 NULL")
            self._ensure_column(cur, "DataQualityMetric", "NVARCHAR(64) NULL")
            self._ensure_column(cur, "DataQualityValue", "FLOAT NULL")
            self._ensure_column(cur, "LeakageFlag", "BIT NULL")
            self._ensure_column(cur, "ParamsJson", "NVARCHAR(MAX) NULL")
            self._ensure_column(cur, "LoggedLocal", "DATETIMEOFFSET NULL")
            self._ensure_column(cur, "LoggedLocalNaive", "DATETIME2 NULL")
            cur.close()
            try:
                if hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                    self.sql_client.conn.commit()
            except Exception:
                pass
        except Exception as exc:
            Console.warn(f"[LOG] Failed to ensure ACM_RunLogs table: {exc}")

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Best-effort parse of a datetime-like value."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def _extract_structured_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Derive structured fields (event type, stage, step, duration, etc.) from the record."""
        message = str(record.get("message", ""))
        context = record.get("context") or {}
        ctx = context if isinstance(context, dict) else {}

        event_type = ctx.get("event_type")
        stage = ctx.get("stage") or ctx.get("pipeline_stage") or ctx.get("component")
        step_name = ctx.get("step") or ctx.get("step_name") or ctx.get("operation")

        duration_ms = ctx.get("duration_ms")
        if duration_ms is None and "duration_s" in ctx:
            try:
                duration_ms = float(ctx["duration_s"]) * 1000.0
            except Exception:
                duration_ms = None

        window_size = ctx.get("window") or ctx.get("window_size")
        row_count = ctx.get("row_count") or ctx.get("rows") or ctx.get("n_rows")
        col_count = ctx.get("col_count") or ctx.get("cols") or ctx.get("n_cols")

        batch_start = self._parse_datetime(ctx.get("batch_start"))
        batch_end = self._parse_datetime(ctx.get("batch_end"))
        baseline_start = self._parse_datetime(ctx.get("baseline_start"))
        baseline_end = self._parse_datetime(ctx.get("baseline_end"))

        data_quality_metric = ctx.get("data_quality_metric") or ctx.get("dq_metric")
        data_quality_value = ctx.get("data_quality_value") or ctx.get("dq_value") or ctx.get("data_quality_score")

        leakage_flag = ctx.get("leakage_flag")
        if leakage_flag is None and "leakage" in ctx:
            leakage_flag = bool(ctx.get("leakage"))

        # Parse leading bracket tag for event type and step (e.g., "[TIMER] fill_missing 0.002s")
        tag_match = re.match(r"^\[(?P<tag>[A-Z0-9_]+)\]\s*(?P<body>.*)$", message)
        message_body = message
        if tag_match:
            message_body = tag_match.group("body")
            if not event_type:
                event_type = tag_match.group("tag")
            if not step_name:
                body_tokens = message_body.strip().split()
                if body_tokens:
                    step_name = body_tokens[0]
            
            # Derive stage from step_name if it contains dots (e.g., "outputs.comprehensive_analytics" -> "outputs")
            if not stage and step_name and "." in step_name:
                stage = step_name.split(".")[0]
            # Fallback: use event_type as stage if we still don't have one
            elif not stage and event_type:
                stage = event_type.lower()

        # Simple duration parse from message suffix (e.g., "0.123s")
        if duration_ms is None:
            duration_match = re.search(r"(?P<val>[0-9]*\.?[0-9]+)\s*s\b", message_body)
            if duration_match:
                try:
                    duration_ms = float(duration_match.group("val")) * 1000.0
                except Exception:
                    duration_ms = None

        # Window / shape hints from message (window=16 n_cols=9 n_rows=454)
        if window_size is None:
            win_match = re.search(r"window=(?P<win>[0-9]+)", message_body)
            if win_match:
                try:
                    window_size = int(win_match.group("win"))
                except Exception:
                    window_size = None

        if col_count is None:
            col_match = re.search(r"n_cols=(?P<cols>[0-9]+)", message_body)
            if col_match:
                try:
                    col_count = int(col_match.group("cols"))
                except Exception:
                    col_count = None

        if row_count is None:
            row_match = re.search(r"n_rows=(?P<rows>[0-9]+)", message_body)
            if row_match:
                try:
                    row_count = int(row_match.group("rows"))
                except Exception:
                    row_count = None

        # Extract simple key=value pairs from the message body into params
        params: Dict[str, Any] = {}
        for kv_match in re.finditer(r"(?P<key>[A-Za-z0-9_]+)=(?P<val>[^\s]+)", message_body):
            key = kv_match.group("key")
            val = kv_match.group("val")
            # Normalize common datetime-like tokens
            parsed_dt = self._parse_datetime(val)
            if parsed_dt:
                params[key] = parsed_dt.isoformat()
            else:
                params[key] = val

        # Merge context extras that aren't already mapped into dedicated columns
        reserved_keys = {
            "module",
            "event_type",
            "stage",
            "pipeline_stage",
            "component",
            "step",
            "step_name",
            "operation",
            "duration_ms",
            "duration_s",
            "window",
            "window_size",
            "row_count",
            "rows",
            "n_rows",
            "col_count",
            "cols",
            "n_cols",
            "batch_start",
            "batch_end",
            "baseline_start",
            "baseline_end",
            "data_quality_metric",
            "dq_metric",
            "data_quality_value",
            "data_quality_score",
            "dq_value",
            "leakage_flag",
            "leakage",
        }
        for k, v in ctx.items():
            if k in reserved_keys:
                continue
            params[k] = v

        params_json = json.dumps(params, ensure_ascii=True, default=str) if params else None

        return {
            "event_type": event_type,
            "stage": stage,
            "step_name": step_name,
            "duration_ms": duration_ms,
            "row_count": row_count,
            "col_count": col_count,
            "window_size": window_size,
            "batch_start": batch_start,
            "batch_end": batch_end,
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "data_quality_metric": data_quality_metric,
            "data_quality_value": data_quality_value,
            "leakage_flag": leakage_flag,
            "params_json": params_json,
        }

    def __call__(self, record: Dict[str, Any]) -> None:
        """Insert a log record into SQL (best-effort, no exceptions raised)."""
        try:
            # Skip ephemeral progress messages (heartbeats, spinners)
            context = record.get("context") or {}
            if isinstance(context, dict) and context.get("skip_sql"):
                return
            
            # Skip unstructured narrative messages (indented sub-logs, parameter dumps)
            message = str(record.get("message", ""))
            if message.startswith("  ") or (not message.startswith("[") and message.strip()):
                # Skip indented sub-logs or messages without tags that aren't warnings/errors
                level = record.get("level", "INFO")
                if level not in ("WARNING", "ERROR", "CRITICAL"):
                    return
            
            logged_at = record.get("timestamp")
            if isinstance(logged_at, str):
                try:
                    logged_at = datetime.fromisoformat(logged_at)
                except Exception:
                    logged_at = None
            if logged_at is None:
                logged_at = datetime.utcnow()
            logged_local = datetime.now().astimezone()
            logged_local_naive = logged_local.replace(tzinfo=None)
            context_json = json.dumps(context, ensure_ascii=True, default=str) if context else None
            message = str(record.get("message", ""))[:4000]
            structured = self._extract_structured_fields(record)
            module = record.get("module")
            # Fallback: if module inference failed (shows utils.logger), use parsed tag/stage/step
            if (module is None or module == "utils.logger"):
                module = structured.get("event_type") or structured.get("stage") or structured.get("step_name") or module
            cur = self.sql_client.cursor()
            cur.execute(
                f"""
                INSERT INTO dbo.{self.table}
                (RunID, EquipID, LoggedAt, LoggedLocal, LoggedLocalNaive, Level, Module, EventType, Stage, StepName, DurationMs, [RowCount], [ColCount], WindowSize, BatchStart, BatchEnd, BaselineStart, BaselineEnd, DataQualityMetric, DataQualityValue, LeakageFlag, ParamsJson, Message, Context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    self.equip_id,
                    logged_at,
                    logged_local,
                    logged_local_naive,
                    record.get("level"),
                    module,
                    structured["event_type"],
                    structured["stage"],
                    structured["step_name"],
                    structured["duration_ms"],
                    structured["row_count"],
                    structured["col_count"],
                    structured["window_size"],
                    structured["batch_start"],
                    structured["batch_end"],
                    structured["baseline_start"],
                    structured["baseline_end"],
                    structured["data_quality_metric"],
                    structured["data_quality_value"],
                    structured["leakage_flag"],
                    structured["params_json"],
                    message,
                    context_json,
                ),
            )
            cur.close()
            try:
                if hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                    if not getattr(self.sql_client.conn, "autocommit", True):
                        self.sql_client.conn.commit()
            except Exception:
                pass
        except Exception as exc:
            # Never raise from a log sink, but print to stderr for diagnostics
            import sys
            print(f"[SQL_LOG_SINK_ERROR] Failed to log to SQL: {exc}", file=sys.stderr)

    def close(self) -> None:
        """Compatibility shim; nothing to close."""
        return

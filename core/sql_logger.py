"""SQL log streaming helpers."""
from __future__ import annotations

import json
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

    def _ensure_table(self) -> None:
        """Create the run logs table if it does not exist."""
        try:
            cur = self.sql_client.cursor()
            cur.execute(
                """
                IF NOT EXISTS (
                    SELECT 1 FROM sys.objects 
                    WHERE object_id = OBJECT_ID(N'dbo.ACM_RunLogs') AND type = N'U'
                )
                BEGIN
                    CREATE TABLE dbo.ACM_RunLogs (
                        LogID BIGINT IDENTITY(1,1) PRIMARY KEY,
                        RunID NVARCHAR(64) NULL,
                        EquipID INT NULL,
                        LoggedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
                        Level NVARCHAR(16) NOT NULL,
                        Module NVARCHAR(128) NULL,
                        Message NVARCHAR(4000) NOT NULL,
                        Context NVARCHAR(MAX) NULL
                    );
                END
                """
            )
            cur.close()
            try:
                if hasattr(self.sql_client, "conn") and hasattr(self.sql_client.conn, "commit"):
                    self.sql_client.conn.commit()
            except Exception:
                pass
        except Exception as exc:
            Console.warn(f"[LOG] Failed to ensure ACM_RunLogs table: {exc}")

    def __call__(self, record: Dict[str, Any]) -> None:
        """Insert a log record into SQL (best-effort, no exceptions raised)."""
        try:
            logged_at = record.get("timestamp")
            if isinstance(logged_at, str):
                try:
                    logged_at = datetime.fromisoformat(logged_at)
                except Exception:
                    logged_at = None
            if logged_at is None:
                logged_at = datetime.utcnow()
            context = record.get("context") or {}
            context_json = json.dumps(context, ensure_ascii=True) if context else None
            message = str(record.get("message", ""))[:4000]
            module = record.get("module")
            cur = self.sql_client.cursor()
            cur.execute(
                f"""
                INSERT INTO dbo.{self.table}
                (RunID, EquipID, LoggedAt, Level, Module, Message, Context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    self.equip_id,
                    logged_at,
                    record.get("level"),
                    module,
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
        except Exception:
            # Never raise from a log sink
            pass

    def close(self) -> None:
        """Compatibility shim; nothing to close."""
        return

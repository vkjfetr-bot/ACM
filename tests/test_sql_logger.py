"""Tests for SQL log sink (BatchedSqlLogSink from sql_logger_v2)."""
from __future__ import annotations

import time
from datetime import datetime

from core.sql_logger_v2 import BatchedSqlLogSink


class DummyConn:
    def __init__(self):
        self.autocommit = False
        self.commit_calls = 0

    def commit(self):
        self.commit_calls += 1


class DummyCursor:
    def __init__(self, log):
        self.log = log
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.log.append(("execute", sql.strip(), params))

    def close(self):
        self.log.append(("close", None, None))


class DummySQLClient:
    def __init__(self):
        self.conn = DummyConn()
        self.log = []

    def cursor(self):
        return DummyCursor(self.log)


def test_batched_sql_log_sink_writes_records():
    """Test that BatchedSqlLogSink queues and writes log records."""
    client = DummySQLClient()
    sink = BatchedSqlLogSink(
        sql_client=client, 
        run_id="run-123", 
        equip_id=42,
        batch_size=1,  # Flush after each record for testing
        flush_interval_ms=100,
    )
    
    # Use the structured log() method
    sink.log(
        level="INFO",
        message="Hello SQL sink",
        module="tests.test_sql_logger",
        event_type="TEST",
        step_name="hello_step",
        duration_ms=12.5,
        context={"foo": "bar"},
    )
    
    # Allow time for background thread to flush
    time.sleep(0.3)
    
    # Force flush and close
    sink.flush()
    sink.close()
    
    # Check INSERT was executed
    insert_calls = [
        entry for entry in client.log 
        if entry[0] == "execute" and "INSERT INTO" in str(entry[1])
    ]
    assert insert_calls, "Expected INSERT statement to be executed"
    
    # Verify record was written
    stats = sink.get_stats()
    assert stats["written"] >= 1, f"Expected at least 1 written, got {stats}"


def test_batched_sql_log_sink_legacy_interface():
    """Test that BatchedSqlLogSink works with legacy __call__ interface."""
    client = DummySQLClient()
    sink = BatchedSqlLogSink(
        sql_client=client, 
        run_id=None, 
        equip_id=None,
        batch_size=1,
        flush_interval_ms=100,
    )

    # Use legacy dict-based interface (for Console.add_sink compatibility)
    sink({
        "timestamp": "2025-11-15T12:34:56",
        "level": "ERROR",
        "module": "tests.test_sql_logger",
        "message": "Legacy interface test",
        "context": {"event_type": "TEST"},
    })
    
    # Allow time for background thread to flush
    time.sleep(0.3)
    sink.flush()
    sink.close()
    
    # Check INSERT was executed
    insert_calls = [
        entry for entry in client.log 
        if entry[0] == "execute" and "INSERT INTO" in str(entry[1])
    ]
    assert insert_calls, "Expected INSERT statement to be executed"


def test_batched_sql_log_sink_skips_ephemeral():
    """Test that skip_sql context flag prevents SQL writes."""
    client = DummySQLClient()
    sink = BatchedSqlLogSink(
        sql_client=client, 
        run_id="run-456", 
        equip_id=1,
        batch_size=1,
        flush_interval_ms=100,
    )

    # This should be skipped
    sink({
        "level": "INFO",
        "message": "Ephemeral message",
        "context": {"skip_sql": True},
    })
    
    time.sleep(0.3)
    sink.flush()
    sink.close()
    
    # Should have no inserts for this message
    stats = sink.get_stats()
    assert stats["written"] == 0, f"Expected 0 written for skip_sql, got {stats}"

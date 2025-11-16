"""Tests for SQL log sink."""
from __future__ import annotations

from datetime import datetime

from core.sql_logger import SqlLogSink


class DummyConn:
    def __init__(self):
        self.autocommit = False
        self.commit_calls = 0

    def commit(self):
        self.commit_calls += 1


class DummyCursor:
    def __init__(self, log):
        self.log = log

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


def test_sql_log_sink_writes_records():
    client = DummySQLClient()
    sink = SqlLogSink(client, run_id="run-123", equip_id=42)

    record = {
        "timestamp": "2025-11-15T12:34:56",
        "level": "INFO",
        "module": "tests.test_sql_logger",
        "message": "Hello SQL sink",
        "context": {"foo": "bar"},
    }
    sink(record)

    # Last execute call should be the INSERT
    insert_calls = [entry for entry in client.log if entry[0] == "execute" and "INSERT INTO" in entry[1]]
    assert insert_calls, "Expected INSERT statement to be executed"
    _, _, params = insert_calls[-1]
    assert params[0] == "run-123"
    assert params[1] == 42
    assert params[3] == "INFO"
    assert params[4] == "tests.test_sql_logger"
    assert params[5] == "Hello SQL sink"
    assert params[6] == '{"foo": "bar"}'
    assert client.conn.commit_calls >= 1


def test_sql_log_sink_handles_missing_timestamp():
    client = DummySQLClient()
    sink = SqlLogSink(client, run_id=None, equip_id=None)

    sink({
        "level": "ERROR",
        "message": "Missing timestamp",
        "context": {},
    })

    insert_calls = [entry for entry in client.log if entry[0] == "execute" and "INSERT INTO" in entry[1]]
    _, _, params = insert_calls[-1]
    # LoggedAt should be auto-generated datetime
    assert isinstance(params[2], datetime)

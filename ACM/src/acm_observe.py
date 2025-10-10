# acm_observe.py
# Operational instrumentation utilities for ACMnxt.
#
# Responsibilities:
# - Append structured run summaries to CSV
# - Log guardrail events (JSONL) and compute aggregate state
# - Emit run-health snapshots for operators

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

RUN_SUMMARY_FILENAME = "run_summary.csv"
GUARDRAIL_LOG_FILENAME = "guardrail_log.jsonl"
RUN_HEALTH_FILENAME = "run_health.json"

RUN_SUMMARY_COLUMNS: List[str] = [
    "run_id",
    "ts_utc",
    "equip",
    "cmd",
    "rows_in",
    "tags",
    "feat_rows",
    "regimes",
    "events",
    "data_span_min",
    "phase",
    "k_selected",
    "theta_p95",
    "drift_flag",
    "guardrail_state",
    "theta_step_pct",
    "latency_s",
    "artifacts_age_min",
    "status",
    "err_msg",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _csv_path(art_dir: str) -> str:
    return os.path.join(art_dir, RUN_SUMMARY_FILENAME)


def _guardrail_log_path(art_dir: str) -> str:
    return os.path.join(art_dir, GUARDRAIL_LOG_FILENAME)


def _run_health_path(art_dir: str) -> str:
    return os.path.join(art_dir, RUN_HEALTH_FILENAME)


def _coerce_row(row: Dict) -> Dict:
    coerced = {k: row.get(k) for k in RUN_SUMMARY_COLUMNS}
    for k in RUN_SUMMARY_COLUMNS:
        v = coerced.get(k)
        if v is None:
            coerced[k] = ""
        else:
            coerced[k] = v
    return coerced


def write_run_summary(art_dir: str, row: Dict) -> None:
    """Append a run summary row (ordered per RUN_SUMMARY_COLUMNS)."""
    _ensure_dir(art_dir)
    path = _csv_path(art_dir)
    needs_header = not os.path.exists(path)

    data = _coerce_row(row)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_SUMMARY_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(data)


def load_guardrail_events(art_dir: str, limit: Optional[int] = None) -> List[Dict]:
    path = _guardrail_log_path(art_dir)
    if not os.path.exists(path):
        return []
    events: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        return events[-limit:]
    return events


def _default_guardrail_event(g_type: str, level: str, message: Optional[str], extra: Optional[Dict]) -> Dict:
    evt = {
        "ts": _now_iso(),
        "type": g_type,
        "level": level,
        "message": message or "",
        "extra": extra or {},
        "acked": False,
    }
    return evt


def append_guardrail_event(
    art_dir: str,
    g_type: str,
    *,
    level: str = "warn",
    message: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Append a guardrail event to the JSONL log and return the event."""
    _ensure_dir(art_dir)
    evt = _default_guardrail_event(g_type, level, message, extra)
    path = _guardrail_log_path(art_dir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(evt) + "\n")
    return evt


def ack_guardrail(art_dir: str, g_type: str, *, first_unacked_only: bool = True) -> bool:
    """Mark guardrail events of the given type as acknowledged."""
    path = _guardrail_log_path(art_dir)
    if not os.path.exists(path):
        return False
    updated = False
    events = load_guardrail_events(art_dir)
    for evt in events:
        if evt.get("type") != g_type or evt.get("acked"):
            continue
        evt["acked"] = True
        evt["acked_ts"] = _now_iso()
        updated = True
        if first_unacked_only:
            break
    if updated:
        with open(path, "w", encoding="utf-8") as f:
            for evt in events:
                f.write(json.dumps(evt) + "\n")
    return updated


def guardrail_state(events: Iterable[Dict]) -> str:
    state = "ok"
    for evt in events:
        if evt.get("acked"):
            continue
        level = (evt.get("level") or "").lower()
        if level == "alert":
            return "alert"
        if level == "warn" and state != "alert":
            state = "warn"
    return state


def write_run_health(art_dir: str, run_row: Optional[Dict] = None, limit: int = 20) -> Dict:
    """Generate a run health snapshot JSON file."""
    events = load_guardrail_events(art_dir, limit=None)
    state = guardrail_state(events)
    recent = events[-limit:]
    open_events = [evt for evt in events if not evt.get("acked")]
    snapshot = {
        "generated_at": _now_iso(),
        "state": state,
        "recent_guardrails": recent,
        "open_guardrails": open_events,
        "latest_run": run_row or {},
    }
    path = _run_health_path(art_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot


__all__ = [
    "RUN_SUMMARY_COLUMNS",
    "append_guardrail_event",
    "guardrail_state",
    "write_run_summary",
    "load_guardrail_events",
    "write_run_health",
    "ack_guardrail",
]

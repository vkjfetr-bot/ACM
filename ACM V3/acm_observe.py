# acm_observe.py
# Operational instrumentation utilities for ACMnxt.
#
# Responsibilities:
# - Append structured run summaries to CSV
# - Log guardrail events (JSONL) and compute aggregate state
# - Emit run-health snapshots for operators
# - Build payload JSON exports for dashboards
# - Enforce artifact retention policy (minimal/full)

from __future__ import annotations

import csv
import fnmatch
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

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

DEFAULT_MINIMAL_KEEP_FILES: Set[str] = {
    RUN_SUMMARY_FILENAME,
    GUARDRAIL_LOG_FILENAME,
    RUN_HEALTH_FILENAME,
    "scores.csv",
    "events.csv",
    "dq.csv",
    "thresholds.csv",
}

DEFAULT_MINIMAL_KEEP_PATTERNS: List[str] = [
    "report_*.html",
    "*.json", # Keep all payloads and manifests
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
    return {
        "ts": _now_iso(),
        "type": g_type,
        "level": level,
        "message": message or "",
        "extra": extra or {},
        "acked": False,
    }


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


# -------------------------- Payload and Artifact Helpers --------------------------

def _build_payload(payload_type: str, version: int, data_key: str, data: Optional[list]) -> Dict:
    return {"type": payload_type, "version": version, data_key: data or []}


def write_payloads(artifacts_dir: str, limit: int = 1000) -> None:
    """Reads latest artifacts and generates structured JSON payloads for dashboards."""
    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    # Timeline Payload
    timeline_points = []
    if (scores_path := art / "scores.csv").exists():
        df = pd.read_csv(scores_path, parse_dates=["Ts"])
        subset = df.tail(limit).to_dict(orient="records")
        timeline_points = [
            {
                "ts": r["Ts"].isoformat(),
                "fused": r.get("FusedScore"),
                "theta": r.get("Theta"),
            }
            for r in subset
        ]

    # Events Payload
    events_records = []
    if (events_path := art / "events_timeline.json").exists():
        try:
            events_records = json.loads(events_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    elif (events_csv_path := art / "events.csv").exists():
        df = pd.read_csv(events_csv_path, parse_dates=["Start", "End"])
        events_records = df.tail(limit).to_dict(orient="records")

    # DQ Payload
    dq_rows = []
    if (dq_path := art / "dq.csv").exists():
        dq_rows = pd.read_csv(dq_path).to_dict(orient="records")

    payloads = {
        "timeline.json": _build_payload("timeline", 1, "points", timeline_points),
        "events.json": _build_payload("events", 1, "events", events_records),
        "dq.json": _build_payload("data_quality", 1, "rows", dq_rows),
    }
    for name, payload in payloads.items():
        (art / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def enforce_artifact_policy(art_dir: str, mode: Optional[str]) -> None:
    """Applies retention rules ('full' or 'minimal'/'temp')."""
    mode_norm = (mode or "full").lower()
    if mode_norm == "full":
        return

    art = Path(art_dir)
    if not art.is_dir():
        return

    keep: Set[str] = set(DEFAULT_MINIMAL_KEEP_FILES)
    model_dirs = {"models"}

    for path in art.iterdir():
        if path.is_dir():
            if path.name not in model_dirs:
                # In future, could recursively clean non-model dirs
                pass
            continue

        # It's a file, check if we should keep it
        if path.name in keep:
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in DEFAULT_MINIMAL_KEEP_PATTERNS):
            continue

        # If we reach here, the file is a candidate for deletion
        try:
            path.unlink()
        except OSError as exc:
            print(f"[ARTIFACT][WARN] Failed to remove {path.name}: {exc}")


__all__ = [
    "RUN_SUMMARY_COLUMNS",
    "append_guardrail_event",
    "guardrail_state",
    "write_run_summary",
    "load_guardrail_events",
    "write_run_health",
    "write_payloads",
    "enforce_artifact_policy",
]
# acm_observe.py
# Operational instrumentation utilities for ACMnxt.
#
# Responsibilities:
# - Append structured run summaries to CSV
# - Log guardrail events (JSONL) and compute aggregate state
# - Emit run-health snapshots for operators
# - Build payload JSON exports
# - Enforce artifact retention policy (minimal/full)

from __future__ import annotations

import csv
import json
import os
import fnmatch
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


# -------------------------- Payload helpers --------------------------

def build_timeline_payload(scores: Optional[List[Dict]] = None) -> Dict:
    return {
        "type": "timeline",
        "version": 1,
        "points": scores or [],
    }


def build_events_payload(events: Optional[List[Dict]] = None) -> Dict:
    return {
        "type": "events",
        "version": 1,
        "events": events or [],
    }


def build_dq_payload(dq_rows: Optional[List[Dict]] = None) -> Dict:
    return {
        "type": "data_quality",
        "version": 1,
        "rows": dq_rows or [],
    }


def write_placeholder_payloads(output_dir: str) -> None:
    """Emit stub payload JSON files; to be populated in future phases."""
    os.makedirs(output_dir, exist_ok=True)
    payloads = {
        "timeline.json": build_timeline_payload(),
        "events.json": build_events_payload(),
        "dq.json": build_dq_payload(),
    }
    for name, payload in payloads.items():
        path = os.path.join(output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def write_payloads(artifacts_dir: str, limit: int = 500) -> None:
    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    timeline_points: List[Dict] = []
    scores_path = art / "scores.csv"
    if scores_path.exists():
        scores_df = pd.read_csv(scores_path, parse_dates=["Ts"])
        subset = scores_df.tail(limit)
        timeline_points = [
            {
                "ts": row.Ts.isoformat(),
                "fused": float(row.FusedScore),
                "theta": float(row.Theta) if "Theta" in subset.columns else None,
                "dominant_head": row.DominantHead if "DominantHead" in subset.columns else None,
                "persistent": bool(row.PersistentEvent) if "PersistentEvent" in subset.columns else False,
            }
            for row in subset.itertuples()
        ]

    events_records: List[Dict] = []
    timeline_json = art / "events_timeline.json"
    if timeline_json.exists():
        try:
            events_records = json.loads(timeline_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            events_records = []
    elif (art / "events.csv").exists():
        events_df = pd.read_csv(art / "events.csv", parse_dates=["Start", "End"])
        events_records = [
            {
                "start": row.Start.isoformat(),
                "end": row.End.isoformat(),
                "peak": float(row.PeakScore),
                "duration_min": float(row.DurationMin) if "DurationMin" in events_df.columns else None,
                "persistence": row.Persistence if "Persistence" in events_df.columns else None,
                "top_tags": row.TopTags.split(",") if "TopTags" in events_df.columns and isinstance(row.TopTags, str) else [],
                "heads": row.ContributingHeads.split(",") if "ContributingHeads" in events_df.columns and isinstance(row.ContributingHeads, str) else [],
                "tag_contrib": json.loads(row.TagContrib) if "TagContrib" in events_df.columns and isinstance(row.TagContrib, str) else {},
            }
            for row in events_df.itertuples()
        ]

    dq_rows: List[Dict] = []
    dq_path = art / "dq.csv"
    if dq_path.exists():
        dq_df = pd.read_csv(dq_path)
        dq_rows = dq_df.to_dict(orient="records")

    payloads = {
        "timeline.json": build_timeline_payload(timeline_points),
        "events.json": build_events_payload(events_records),
        "dq.json": build_dq_payload(dq_rows),
    }
    for name, payload in payloads.items():
        (art / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def enforce_artifact_policy(
    art_dir: str,
    mode: Optional[str],
    *,
    prefix: Optional[str] = None,
    keep_files: Optional[Iterable[str]] = None,
    keep_patterns: Optional[Iterable[str]] = None,
) -> None:
    """
    Apply retention rules to the artifact directory.

    - mode="full" leaves everything as-is.
    - mode="minimal" removes duplicate prefixed files (e.g. acm_scores.csv) while
      preserving canonical CSVs, guardrail logs, models, and reports.
    - mode="temp" behaves like "minimal" for now (placeholder for future cleanup).
    """

    mode_norm = (mode or "full").lower()
    if mode_norm == "full":
        return

    art = Path(art_dir)
    art.mkdir(parents=True, exist_ok=True)

    keep: Set[str] = set(DEFAULT_MINIMAL_KEEP_FILES)
    if keep_files:
        keep.update(keep_files)
    pattern_list: List[str] = list(DEFAULT_MINIMAL_KEEP_PATTERNS)
    if keep_patterns:
        pattern_list.extend(keep_patterns)

    # Always keep model directories and nested content.
    model_dirs = {"models"}

    def _should_keep(file_path: Path) -> bool:
        name = file_path.name
        if name in keep:
            return True
        for pattern in pattern_list:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    # Remove prefixed duplicates for the requested prefix (usually "acm").
    if prefix:
        for path in art.glob(f"{prefix}_*"):
            if not path.is_file():
                continue
            if _should_keep(path):
                continue
            try:
                path.unlink()
            except OSError as exc:
                print(f"[ARTIFACT][WARN] Failed to remove {path.name}: {exc}")

    # Prune temporary diagnostic files at the top level, keeping canonical outputs.
    for path in art.iterdir():
        if path.is_dir():
            if path.name in model_dirs:
                continue
            # Skip directories; future modes can manage them explicitly.
            continue
        if _should_keep(path):
            continue
        if path.name.startswith("run_") and path.suffix == ".jsonl":
            # Allow retention of recent run logs in minimal mode; keep the latest only.
            continue
        if mode_norm in {"minimal", "temp"} and path.name.endswith("_diagnostics.csv"):
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
    "ack_guardrail",
    "build_timeline_payload",
    "build_events_payload",
    "build_dq_payload",
    "write_placeholder_payloads",
    "write_payloads",
    "enforce_artifact_policy",
]

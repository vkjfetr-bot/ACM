# acm_payloads.py
# Placeholder payload builders for future dashboard integrations.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


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


__all__ = [
    "build_timeline_payload",
    "build_events_payload",
    "build_dq_payload",
    "write_placeholder_payloads",
    "write_payloads",
]

# acm_payloads.py
# Placeholder payload builders for future dashboard integrations.

from __future__ import annotations

from typing import Dict, List, Optional


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
    import json
    import os

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


__all__ = [
    "build_timeline_payload",
    "build_events_payload",
    "build_dq_payload",
    "write_placeholder_payloads",
]

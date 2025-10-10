"""Operator brief generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class BriefInputs:
    equip: str
    window: Dict[str, str]
    regimes: Dict[int, float]
    detectors: Dict[int, str]
    thresholds: Dict[int, float]
    events: pd.DataFrame
    drift: List[Dict[str, object]]


def build_brief(inputs: BriefInputs) -> Dict[str, object]:
    events_payload = []
    for row in inputs.events.itertuples():
        events_payload.append(
            {
                "start": row.Start.isoformat(),
                "end": row.End.isoformat(),
                "regime": int(row.Regime) if "Regime" in inputs.events.columns else None,
                "peak_tau": float(row.PeakScore),
                "family": getattr(row, "Family", None),
                "top_tags": row.TopTags.split(",") if isinstance(row.TopTags, str) else [],
                "persistence": row.Persistence,
            }
        )
    return {
        "equip": inputs.equip,
        "window": inputs.window,
        "regimes": [{"id": int(k), "share": float(v)} for k, v in inputs.regimes.items()],
        "detectors": inputs.detectors,
        "thresholds": inputs.thresholds,
        "events": events_payload,
        "drift": inputs.drift,
    }


def brief_markdown(brief: Dict[str, object]) -> str:
    lines = [
        f"# ACM Brief — {brief['equip']}",
        "",
        f"**Window:** {brief['window'].get('t0')} → {brief['window'].get('t1')}",
        "",
        "## Regime distribution",
    ]
    for regime in brief.get("regimes", []):
        lines.append(f"- Regime {regime['id']}: {regime['share']:.1%}")
    lines.append("")
    lines.append("## Events")
    if not brief.get("events"):
        lines.append("_No anomalies detected._")
    else:
        for event in brief["events"]:
            tags = ", ".join(event.get("top_tags", []))
            lines.append(
                f"- {event['start']} → {event['end']} | Regime {event.get('regime')} | Peak τ {event['peak_tau']:.3f} | Tags: {tags}"
            )
    lines.append("")
    lines.append("## Drift monitors")
    if not brief.get("drift"):
        lines.append("_No drift alerts._")
    else:
        for d in brief["drift"]:
            lines.append(f"- {d['metric']}: {d['value']:.4f} (threshold {d['threshold']}) → {d['level']}")
    return "\n".join(lines)

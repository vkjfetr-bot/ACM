from __future__ import annotations

from typing import List, Tuple, Dict
import pandas as pd


def summarize_overview(kpis: Dict) -> List[str]:
    eps = kpis.get("episodes", 0)
    peak = kpis.get("max_score", 0.0)
    avg = kpis.get("avg_daily_anomaly_minutes", 0)
    return [
        f"Detected {eps} anomaly episode(s) over the selected period.",
        f"Peak fused score reached {peak:.2f}.",
        f"Average daily anomaly duration ≈ {avg} minutes.",
    ]


def summarize_episode(ep: Dict, contrib: List[Tuple[str, float]] | None = None) -> str:
    start = str(ep.get("start", "?"))
    end = str(ep.get("end", "?"))
    peak = float(ep.get("peak", 0.0))
    lead = ", ".join(t for t, _ in (contrib or [])[:2]) or "multiple tags"
    return f"Episode from {start} to {end}; peak score {peak:.2f}. Likely drivers: {lead}."


def summarize_dq(dq_df: pd.DataFrame) -> List[str]:
    if dq_df is None or dq_df.empty:
        return ["Data quality looks acceptable; no major issues detected."]
    worst = dq_df.sort_values(["coverage", "dropout"], ascending=[True, False]).head(3)
    worst_tags = ", ".join(worst["tag"].astype(str).tolist())
    return [
        f"Tags needing attention (coverage/dropout): {worst_tags}.",
        "Low coverage can hide real issues; review sensor uptime and gaps.",
    ]


"""Generate a lightweight HTML report for ACM_V2 artifacts."""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _load_guardrails(path: Path, limit: int = 20) -> List[Dict]:
    if not path.exists():
        return []
    events: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events[-limit:]


def _encode_plot_to_data_uri(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_report(artifacts_dir: str, equip: str) -> None:
    art_dir = Path(artifacts_dir)
    if not art_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    scores_path = art_dir / "scores.csv"
    events_path = art_dir / "events.csv"
    dq_path = art_dir / "dq.csv"
    run_summary_path = art_dir / "run_summary.csv"
    guardrail_path = art_dir / "guardrail_log.jsonl"

    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores.csv in {artifacts_dir}")

    scores = pd.read_csv(scores_path, parse_dates=["Ts"])
    fig, ax = plt.subplots(figsize=(10, 4))
    if "FusedScore" in scores.columns:
        ax.plot(scores["Ts"], scores["FusedScore"], label="Fused Score", color="#1f77b4")
    if "Theta" in scores.columns:
        ax.plot(scores["Ts"], scores["Theta"], label="Theta(t)", color="#ff7f0e")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Score")
    ax.set_title(f"{equip} - Fused Score vs Theta")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plot_uri = _encode_plot_to_data_uri(fig)

    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    dq = pd.read_csv(dq_path) if dq_path.exists() else pd.DataFrame()
    guardrails = _load_guardrails(guardrail_path)
    run_summary = pd.read_csv(run_summary_path) if run_summary_path.exists() else pd.DataFrame()
    latest_summary = run_summary.tail(2) if not run_summary.empty else pd.DataFrame()

    if not events.empty:
        display_events = events[["Start", "End", "PeakScore"]].copy()
        if "DurationMin" in events.columns:
            display_events["DurationMin"] = events["DurationMin"]
        if "Persistence" in events.columns:
            display_events["Persistence"] = events["Persistence"]
        if "TopTags" in events.columns:
            display_events["TopTags"] = events["TopTags"]
        if "ContributingHeads" in events.columns:
            display_events["Heads"] = events["ContributingHeads"]
        html_events = display_events.head(20).to_html(index=False, classes="table table-sm")
    else:
        html_events = "<em>No events recorded.</em>"
    html_dq = (
        dq.sort_values("flatline_pct", ascending=False)
        .head(20)
        .to_html(index=False, classes="table table-sm")
        if not dq.empty
        else "<em>No data-quality metrics.</em>"
    )
    html_runs = (
        latest_summary.to_html(index=False, classes="table table-sm")
        if not latest_summary.empty
        else "<em>No run summary yet.</em>"
    )
    if guardrails:
        guardrail_rows = "".join(
            f"<tr><td>{g.get('ts','')}</td><td>{g.get('type','')}</td><td>{g.get('level','')}</td>"
            f"<td>{g.get('message','')}</td></tr>"
            for g in guardrails
        )
        html_guardrails = (
            "<table class='table table-sm'>"
            "<thead><tr><th>Timestamp</th><th>Type</th><th>Level</th><th>Message</th></tr></thead>"
            f"<tbody>{guardrail_rows}</tbody></table>"
        )
    else:
        html_guardrails = "<em>No guardrail entries.</em>"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ACMnxt Report - {equip}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background-color: #f7f9fb; color: #1d1d1f; }}
    h1, h2 {{ color: #003b73; }}
    .section {{ margin-bottom: 2rem; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ccc; padding: 0.5rem; background: #fff; }}
    .table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    .table th {{ background-color: #003b73; color: #fff; }}
    em {{ color: #555; }}
  </style>
</head>
<body>
  <h1>ACMnxt Report &mdash; {equip}</h1>
  <div class="section">
    <h2>Timeline: Fused Score vs Theta</h2>
    <img src="{plot_uri}" alt="Fused Score vs Theta" />
  </div>
  <div class="section">
    <h2>Recent Guardrail Events</h2>
    {html_guardrails}
  </div>
  <div class="section">
    <h2>Latest Run Summary</h2>
    {html_runs}
  </div>
  <div class="section">
    <h2>Recent Events (Top 20)</h2>
    {html_events}
  </div>
  <div class="section">
    <h2>Data Quality Snapshot (Top 20 by flatline %)</h2>
    {html_dq}
  </div>
</body>
</html>
"""

    out_path = art_dir / f"report_{equip}.html"
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"[REPORT] Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser("acm_report_basic")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--equip", default="equipment")
    args = parser.parse_args()
    build_report(args.artifacts, args.equip)


if __name__ == "__main__":
    main()

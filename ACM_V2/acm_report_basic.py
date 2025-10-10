"""Generate a chart-rich HTML report for ACM_V2 artifacts."""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


ACCENT_BLUE = "#005a9c"
ACCENT_ORANGE = "#ff9248"
ACCENT_GREEN = "#2ca58d"
BACKGROUND = "#f7f9fb"
TEXT_COLOR = "#1d1d1f"
GRID_STYLE = {"linestyle": "--", "alpha": 0.3}


def _encode_fig(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_guardrails(path: Path, limit: int = 10) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def _summarise_guardrails(rows: List[Dict]) -> str:
    if not rows:
        return "<em>No guardrails raised.</em>"
    items = []
    for row in rows[::-1]:
        stamp = row.get("ts", "")
        level = row.get("level", "").upper()
        gtype = row.get("type", "")
        msg = row.get("message", "")
        items.append(f"<li><strong>{level}</strong> &mdash; {gtype}: {msg} <span class='stamp'>({stamp})</span></li>")
    return "<ul>" + "".join(items) + "</ul>"


def _latest_run_summary(path: Path) -> str:
    if not path.exists():
        return "<em>No runs recorded yet.</em>"
    df = pd.read_csv(path)
    if df.empty:
        return "<em>No runs recorded yet.</em>"
    latest = df.tail(1).iloc[0]
    bits = [
        f"<li><strong>Command:</strong> {latest['cmd']}</li>",
        f"<li><strong>Rows scanned:</strong> {int(latest['rows_in'])} | <strong>Events:</strong> {int(latest['events'])}</li>",
        f"<li><strong>Phase:</strong> {latest['phase']} | <strong>Theta p95:</strong> {float(latest['theta_p95']):0.3f}</li>",
        f"<li><strong>Guardrail state:</strong> {latest['guardrail_state']} | <strong>Latency:</strong> {float(latest['latency_s']):0.2f} s</li>",
    ]
    return "<ul>" + "".join(bits) + "</ul>"


def _plot_health(scores: pd.DataFrame, events: Optional[pd.DataFrame]) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(scores["Ts"], scores["FusedScore"], label="Fused score", color=ACCENT_BLUE, linewidth=1.6)
    if "Theta" in scores.columns:
        ax.plot(scores["Ts"], scores["Theta"], label="Theta(t)", color=ACCENT_ORANGE, linewidth=1.2)
    if events is not None and not events.empty:
        for _, event in events.iterrows():
            if event.get("Persistence", "").lower() == "persistent":
                ax.axvspan(event["Start"], event["End"], color=ACCENT_ORANGE, alpha=0.12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.set_ylabel("Score")
    ax.set_title("Health timeline")
    ax.grid(**GRID_STYLE)
    ax.legend(frameon=False)
    return _encode_fig(fig)


def _select_top_event(events: pd.DataFrame) -> Optional[pd.Series]:
    if events is None or events.empty:
        return None
    events = events.copy()
    events["PersistenceScore"] = events["Persistence"].str.lower().eq("persistent").astype(int)
    events = events.sort_values(["PersistenceScore", "PeakScore"], ascending=[False, False])
    return events.iloc[0]


def _plot_blame(top_event: pd.Series) -> str:
    contrib = top_event.get("TagContrib", {})
    if isinstance(contrib, str):
        try:
            contrib = json.loads(contrib)
        except json.JSONDecodeError:
            contrib = {}
    if not contrib:
        return ""
    tags, values = zip(*sorted(contrib.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(tags, values, color=ACCENT_BLUE)
    ax.set_xlabel("Average |z-score|")
    ax.set_title("Blame chart &mdash; top drivers")
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:0.2f}", va="center", fontsize=9, color=TEXT_COLOR)
    ax.grid(**GRID_STYLE, axis="x")
    return _encode_fig(fig)


def _plot_head_mix(scores: pd.DataFrame) -> str:
    if "DominantHead" not in scores.columns:
        return ""
    counts = scores["DominantHead"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.values, color=ACCENT_GREEN)
    ax.set_ylabel("Samples")
    ax.set_title("Which hypothesis fired most")
    ax.grid(**GRID_STYLE, axis="y")
    return _encode_fig(fig)


def _plot_dq(dq: pd.DataFrame) -> str:
    if dq.empty:
        return ""
    metrics = [
        ("flatline_pct", "Flatline %"),
        ("dropout_pct", "Dropout %"),
        ("spikes_pct", "Spike %"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, metrics):
        if col not in dq.columns:
            ax.axis("off")
            continue
        top = dq.nlargest(5, col)
        ax.barh(top["Tag"], top[col], color=ACCENT_BLUE)
        ax.set_xlabel(label)
        ax.invert_yaxis()
        ax.grid(**GRID_STYLE, axis="x")
    fig.suptitle("Data-quality hot spots (top 5)")
    plt.tight_layout()
    return _encode_fig(fig)


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
    events = pd.read_csv(events_path, parse_dates=["Start", "End"]) if events_path.exists() else pd.DataFrame()
    dq = pd.read_csv(dq_path) if dq_path.exists() else pd.DataFrame()

    guardrails = _load_guardrails(guardrail_path)
    guardrail_html = _summarise_guardrails(guardrails)
    run_summary_html = _latest_run_summary(run_summary_path)

    health_img = _plot_health(scores, events if not events.empty else None)
    top_event = _select_top_event(events) if not events.empty else None
    blame_img = _plot_blame(top_event) if top_event is not None else ""
    head_img = _plot_head_mix(scores)
    dq_img = _plot_dq(dq)

    if top_event is not None:
        contrib = json.loads(top_event.get("TagContrib", "{}")) if isinstance(top_event.get("TagContrib"), str) else top_event.get("TagContrib", {})
        top_tags = ", ".join(top_event.get("TopTags", "").split(",")) or "N/A"
        blame_text = f"<ul><li><strong>Window:</strong> {top_event['Start']} ? {top_event['End']}</li>" \
            f"<li><strong>Peak score:</strong> {float(top_event['PeakScore']):0.2f}</li>" \
            f"<li><strong>Persistence:</strong> {top_event['Persistence']}</li>" \
            f"<li><strong>Primary drivers:</strong> {top_tags}</li></ul>"
    else:
        blame_text = "<em>No significant events detected.</em>"

    dq_text = "" if dq_img else "<em>Data-quality metrics were clean for this run.</em>"

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>ACMnxt Report - {equip}</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; background: {BACKGROUND}; color: {TEXT_COLOR}; margin: 2rem; }}
    h1 {{ color: {ACCENT_BLUE}; }}
    h2 {{ color: {ACCENT_BLUE}; margin-top: 2rem; }}
    .section {{ margin-bottom: 2.5rem; background: #fff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 6px 14px rgba(0,0,0,0.06); }}
    img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
    ul {{ margin: 0.5rem 0 0 1.2rem; }}
    li {{ margin-bottom: 0.35rem; }}
    .stamp {{ color: #646c76; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>ACMnxt Situation Report — {equip}</h1>
  <div class='section'>
    <h2>How healthy are we?</h2>
    <img src='{health_img}' alt='Fused score vs theta chart'>
  </div>
  <div class='section'>
    <h2>Who is to blame?</h2>
    {blame_text}
    {f"<img src='{blame_img}' alt='Top tag contributions'>" if blame_img else ""}
    {f"<img src='{head_img}' alt='Dominant hypothesis mix'>" if head_img else ""}
  </div>
  <div class='section'>
    <h2>Guardrails & Run Snapshot</h2>
    <h3>Latest run</h3>
    {run_summary_html}
    <h3>Guardrail events</h3>
    {guardrail_html}
  </div>
  <div class='section'>
    <h2>Sensor data quality</h2>
    {f"<img src='{dq_img}' alt='Data quality issues'>" if dq_img else dq_text}
  </div>
</body>
</html>
"""

    out_path = art_dir / f"report_{equip}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[REPORT] Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser("acm_report_basic")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--equip", default="equipment")
    args = parser.parse_args()
    build_report(args.artifacts, args.equip)


if __name__ == "__main__":
    main()

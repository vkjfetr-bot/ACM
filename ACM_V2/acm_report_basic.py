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

ACCENT_BLUE = "#087f8c"
ACCENT_ORANGE = "#f76f8e"
ACCENT_GREEN = "#43aa8b"
BACKGROUND = "#f5f8ff"
TEXT_COLOR = "#212738"
GRID_STYLE = {"linestyle": "--", "alpha": 0.25}
VIVID_PALETTE = ["#087f8c", "#f76f8e", "#ffb400", "#5a189a", "#43aa8b", "#f94144", "#4cc9f0"]


def _safe_palette(n: int) -> List[str]:
    if n <= len(VIVID_PALETTE):
        return VIVID_PALETTE[:n]
    reps = (n // len(VIVID_PALETTE)) + 1
    return (VIVID_PALETTE * reps)[:n]


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
        items.append(
            f"<li><strong>{level}</strong> &mdash; {gtype}: {msg} <span class='stamp'>({stamp})</span></li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def _latest_runs(path: Path, take: int = 6) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df.tail(take)


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
    ax.barh(tags, values, color=_safe_palette(len(tags)))
    ax.set_xlabel("Average |z-score|")
    ax.set_title("Blame chart - top drivers")
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:0.2f}", va="center", fontsize=9, color=TEXT_COLOR)
    ax.grid(**GRID_STYLE, axis="x")
    return _encode_fig(fig)


def _plot_head_mix(scores: pd.DataFrame) -> str:
    if "DominantHead" not in scores.columns:
        return ""
    counts = scores["DominantHead"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.values, color=_safe_palette(len(counts)))
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
        ax.barh(top["Tag"], top[col], color=_safe_palette(1)[0])
        ax.set_xlabel(label)
        ax.invert_yaxis()
        ax.grid(**GRID_STYLE, axis="x")
    fig.suptitle("Data-quality hot spots (top 5)")
    plt.tight_layout()
    return _encode_fig(fig)


def _plot_tag_overlay(resampled_path: Path, events: Optional[pd.DataFrame], top_event: Optional[pd.Series]) -> str:
    if top_event is None or not resampled_path.exists():
        return ""
    df = pd.read_csv(resampled_path)
    if df.empty:
        return ""
    ts_col = None
    for name in ["Ts", "ts", "timestamp", df.columns[0]]:
        if name in df.columns:
            ts_col = name
            break
    if ts_col is None:
        return ""
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.isna().all():
        return ""
    df = df.assign(Ts=ts).dropna(subset=["Ts"]).set_index("Ts").sort_index()
    raw_tags = top_event.get("TopTags", "")
    tag_list = [t.strip() for t in raw_tags.split(",") if t.strip()]
    contrib = top_event.get("TagContrib", {})
    if isinstance(contrib, str):
        try:
            contrib = json.loads(contrib)
        except json.JSONDecodeError:
            contrib = {}
    if not tag_list and contrib:
        tag_list = list(sorted(contrib, key=contrib.get, reverse=True)[:3])
    if not tag_list:
        tag_list = list(df.columns[:3])
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, "#7c5295"]
    for idx, tag in enumerate(tag_list):
        if tag not in df.columns:
            continue
        series = pd.to_numeric(df[tag], errors="coerce")
        if series.isna().all():
            continue
        z = (series - series.mean()) / (series.std(ddof=0) + 1e-9)
        ax.plot(df.index, z, label=tag, color=colors[idx % len(colors)], linewidth=1.4)
    if events is not None and not events.empty:
        for _, event in events.iterrows():
            alpha = 0.1 if event.get("Persistence", "").lower() == "persistent" else 0.05
            ax.axvspan(event["Start"], event["End"], color="#ed6a5a", alpha=alpha)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.set_ylabel("Normalised value (z)")
    ax.set_title("Tag behaviour overlay")
    ax.grid(**GRID_STYLE)
    ax.legend(frameon=False)
    return _encode_fig(fig)


def _plot_head_timeseries(scores: pd.DataFrame) -> str:
    columns = [c for c in scores.columns if c.startswith("H")]
    if not columns:
        return ""
    palette = _safe_palette(len(columns))
    fig, ax = plt.subplots(figsize=(11, 4))
    for idx, col in enumerate(columns):
        ax.plot(scores["Ts"], scores[col], label=col, color=palette[idx], linewidth=1.1)
    ax.set_title("Hypothesis score footprints")
    ax.set_ylabel("Score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.grid(**GRID_STYLE)
    ax.legend(frameon=False, ncol=3)
    return _encode_fig(fig)


def _plot_event_timeline(events: pd.DataFrame) -> str:
    if events.empty:
        return ""
    events = events.copy()
    events["Date"] = events["Start"].dt.date
    agg = events.groupby(["Date", "Persistence"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    agg.plot(kind="bar", stacked=True, ax=ax, color=["#ff8fab", "#7b2cbf"])
    ax.set_ylabel("Events")
    ax.set_title("Daily anomaly cadence")
    ax.grid(**GRID_STYLE, axis="y")
    ax.legend(["Transient", "Persistent"], frameon=False)
    return _encode_fig(fig)


def _plot_event_scatter(events: pd.DataFrame) -> str:
    if events.empty:
        return ""
    fig, ax = plt.subplots(figsize=(6, 4))
    colours = events["Persistence"].str.lower().map({"persistent": "#ff595e", "transient": "#1982c4"}).fillna("#adb5bd")
    bubble = events["DurationMin"].fillna(5) + 5
    ax.scatter(events["Start"], events["PeakScore"], s=bubble, c=colours, alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    ax.set_ylabel("Peak score")
    ax.set_title("Every anomaly bubble: duration vs severity")
    ax.grid(**GRID_STYLE)
    return _encode_fig(fig)


def _plot_correlation_heatmap(resampled_path: Path, top_event: Optional[pd.Series]) -> str:
    if top_event is None or not resampled_path.exists():
        return ""
    df = pd.read_csv(resampled_path)
    if df.empty:
        return ""
    ts_col = None
    for name in ["Ts", "ts", "timestamp", df.columns[0]]:
        if name in df.columns:
            ts_col = name
            break
    if ts_col:
        df["Ts"] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.drop(columns=[ts_col])
    if "Ts" in df.columns:
        df = df.drop(columns=["Ts"])
    tags = [t.strip() for t in (top_event.get("TopTags", "") or "").split(",") if t.strip()]
    contrib = top_event.get("TagContrib", {})
    if isinstance(contrib, str):
        try:
            contrib = json.loads(contrib)
        except json.JSONDecodeError:
            contrib = {}
    if contrib:
        tags = list({*tags, *sorted(contrib, key=contrib.get, reverse=True)[:4]})
    tags = [t for t in tags if t in df.columns]
    if len(tags) < 2:
        return ""
    corr = df[tags].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Correlation")
    ax.set_title("Tag correlation inside alert window")
    return _encode_fig(fig)


def _plot_split_summary(run_tail: pd.DataFrame) -> str:
    if run_tail.empty:
        return ""
    score_rows = run_tail[run_tail["cmd"].str.contains("score", na=False)]
    if score_rows.empty:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(score_rows["cmd"], score_rows["theta_p95"], color=_safe_palette(len(score_rows)))
    ax.set_ylabel("Theta p95")
    ax.set_title("Multiple score runs (prefix comparison)")
    ax.set_xticklabels(score_rows["cmd"], rotation=45, ha="right")
    ax.grid(**GRID_STYLE, axis="y")
    return _encode_fig(fig)


def build_report(artifacts_dir: str, equip: str) -> None:
    art_dir = Path(artifacts_dir)
    if not art_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    scores_path = art_dir / "scores.csv"
    events_path = art_dir / "events.csv"
    dq_path = art_dir / "dq.csv"
    resampled_path = art_dir / "acm_resampled.csv"
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
    overlay_img = _plot_tag_overlay(resampled_path, events if not events.empty else None, top_event)

    if top_event is not None:
        top_tags = ", ".join([t.strip() for t in top_event.get("TopTags", "").split(",") if t.strip()]) or "N/A"
        blame_text = (
            f"<ul>"
            f"<li><strong>Window:</strong> {top_event['Start']} → {top_event['End']}</li>"
            f"<li><strong>Peak score:</strong> {float(top_event['PeakScore']):0.2f}</li>"
            f"<li><strong>Persistence:</strong> {top_event['Persistence']}</li>"
            f"<li><strong>Primary drivers:</strong> {top_tags}</li>"
            f"</ul>"
        )
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
    <h2>How is the asset behaving?</h2>
    <img src='{health_img}' alt='Fused score vs theta timeline'>
    <p class='caption'>Shaded bands highlight anomaly windows; persistent alerts use a bolder tint so the timeline reads like a story.</p>
  </div>
  <div class='section'>
    <h2>Who caused the biggest alert?</h2>
    {blame_text}
    {f"<img src='{blame_img}' alt='Top contributing tags'>" if blame_img else ""}
    {f"<p class='caption'>Average z-scores per tag inside the highest alert window. Higher bars imply stronger blame.</p>" if blame_img else ""}
    {f"<img src='{overlay_img}' alt='Tag behaviour overlay'>" if overlay_img else ""}
    {f"<p class='caption'>Normalised tag traces for the culprits across time; the red bands show exactly when they pushed the score over threshold.</p>" if overlay_img else ""}
    {f"<img src='{corr_img}' alt='Tag correlation heatmap'>" if corr_img else ""}
    {f"<p class='caption'>Correlation of the same tags during the alert window, showing how joint movements escalate the anomaly.</p>" if corr_img else ""}
  </div>
  <div class='section'>
    <h2>Anomaly cadence</h2>
    {f"<img src='{event_timeline_img}' alt='Daily anomaly cadence'>" if event_timeline_img else "<em>No anomalies recorded in the selected window.</em>"}
    {f"<p class='caption'>Persistent events stack on top of transient ones; busy days stand out immediately.</p>" if event_timeline_img else ""}
    {f"<img src='{event_scatter_img}' alt='Anomaly scatter'>" if event_scatter_img else ""}
    {f"<p class='caption'>Each bubble represents one anomaly — the radius mirrors duration and the height reflects peak severity.</p>" if event_scatter_img else ""}
  </div>
  <div class='section'>
    <h2>Which hypothesis is firing?</h2>
    {f"<img src='{head_mix_img}' alt='Head dominance'>" if head_mix_img else ""}
    {f"<p class='caption'>Counts of windows dominated by each hypothesis (H1 spikes, H2 multivariate structure, H3 drift).</p>" if head_mix_img else ""}
    {f"<img src='{head_ts_img}' alt='Hypothesis score footprints'>" if head_ts_img else ""}
    {f"<p class='caption'>Score footprints per hypothesis reveal when the model switches from spike detection to structural drift.</p>" if head_ts_img else ""}
  </div>
  <div class='section'>
    <h2>Run history & guardrails</h2>
    {guardrail_html}
    {run_table_html}
    {f"<img src='{split_img}' alt='Score prefix summary'>" if split_img else ""}
    {f"<p class='caption'>Multiple score runs in the same execution help compare operating regimes or synthetic splits.</p>" if split_img else ""}
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

"""Native Report Builder with Real Charts

Implements a clear, data-rich report using existing ACMnxt artifacts:
- Header Summary with health grade and key KPIs
- Health Overview: timeline with threshold and event shading
- Data Quality: table + heatmap
- Tag Trends: sampled raw top-variance tags, regime strip
- Regimes & Drift: regime distribution and drift distance
- Event Summary: table of events (top 10)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


DERIVED = {"FusedScore", "H1_Forecast", "H2_Recon", "H3_Drift", "Regime", "ContextMask"}


def _read_scores(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    if "Ts" in df.columns:
        df["Ts"] = pd.to_datetime(df["Ts"], errors="coerce", utc=True)
        df = df.set_index("Ts").sort_index()
    return df


def _pick_tags(df: pd.DataFrame, top_n: int = 6) -> List[str]:
    numcols = [c for c in df.columns if c not in DERIVED and pd.api.types.is_numeric_dtype(df[c])]
    if not numcols:
        return []
    # Prefer tags with enough availability
    avail = {c: float(pd.to_numeric(df[c], errors="coerce").notna().mean()) for c in numcols}
    good = [c for c in numcols if avail.get(c, 0.0) >= 0.5]
    pool = good if good else numcols
    var = df[pool].var(numeric_only=True).sort_values(ascending=False)
    return var.index.tolist()[:top_n]


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_timeline(scores: pd.DataFrame, events: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    fs = scores.get("FusedScore")
    if fs is not None:
        # Ensure timezone-naive for matplotlib
        if isinstance(fs.index, pd.DatetimeIndex) and fs.index.tz is not None:
            fs = fs.tz_convert(None)
        fs.plot(ax=ax, color="black", lw=1.5, label="Fused")
        thr = float(pd.to_numeric(fs, errors="coerce").quantile(0.95))
        ax.axhline(thr, color="#ef4444", lw=1.0, ls="--", alpha=0.8, label="threshold")
        # Shade event windows
        for _, r in events.iterrows():
            if pd.isna(r.get("Start")) or pd.isna(r.get("End")):
                continue
            s = r["Start"]
            e = r["End"]
            if isinstance(s, pd.Timestamp) and s.tzinfo is not None:
                s = s.tz_convert(None)
            if isinstance(e, pd.Timestamp) and e.tzinfo is not None:
                e = e.tz_convert(None)
            ax.axvspan(s, e, color="#ef4444", alpha=0.08)
    if "H1_Forecast" in scores:
        h1 = scores["H1_Forecast"]
        if isinstance(h1.index, pd.DatetimeIndex) and h1.index.tz is not None:
            h1 = h1.tz_convert(None)
        h1.plot(ax=ax, alpha=0.5, label="H1")
    if "H2_Recon" in scores:
        h2 = scores["H2_Recon"]
        if isinstance(h2.index, pd.DatetimeIndex) and h2.index.tz is not None:
            h2 = h2.tz_convert(None)
        h2.plot(ax=ax, alpha=0.5, label="H2")
    ax.set_title("Anomaly Scores Over Time")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    # Tighten x-limits if datetime
    if fs is not None and isinstance(fs.index, pd.DatetimeIndex):
        ax.set_xlim(fs.index.min(), fs.index.max())
    _save_fig(fig, out_path)


def _plot_sampled_tags(df: pd.DataFrame, tags: List[str], out_path: Path) -> None:
    if not tags:
        # Create blank placeholder
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No numeric tags found", ha="center", va="center")
        ax.axis("off")
        _save_fig(fig, out_path)
        return
    # Downsample for readability: resample to 5-minute median if time-indexed
    sub = df[tags[:4]].copy()  # limit to top-4 for clarity
    if isinstance(sub.index, pd.DatetimeIndex):
        sub = sub.resample("5min").median()
    # Fill small gaps for visualization only
    sub = sub.interpolate(limit=3).ffill(limit=3).bfill(limit=3)
    # Plot overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    # Robust z: center by median, scale by MAD
    med = sub.median(numeric_only=True)
    mad = (sub - med).abs().median(numeric_only=True).replace(0.0, 1.0)
    z = (sub - med) / (1.4826 * mad)
    z.plot(ax=ax, lw=0.9)
    ax.set_title("Sampled Raw Tags (std-normalized, 5-min)")
    ax.set_ylabel("z-units")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    _save_fig(fig, out_path)


def _plot_regime_strip(scores: pd.DataFrame, out_path: Path) -> None:
    if "Regime" not in scores:
        return
    regs = pd.to_numeric(scores["Regime"], errors="coerce").fillna(-1).astype(int).values
    uniq = np.unique(regs)
    cmap = plt.get_cmap('tab10')
    colors = {r: cmap(i % 10) for i, r in enumerate(uniq)}
    x = np.arange(len(regs))
    fig, ax = plt.subplots(figsize=(12, 1.1))
    ax.scatter(x, np.zeros_like(x), c=[colors[r] for r in regs], s=8, marker='s')
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title('Operating States (Regimes)')
    _save_fig(fig, out_path)


def _plot_regime_distribution(scores: pd.DataFrame, out_path: Path) -> None:
    if "Regime" not in scores:
        return
    s = pd.to_numeric(scores["Regime"], errors="coerce").dropna().astype(int)
    if s.empty:
        return
    pct = (s.value_counts().sort_index() / len(s) * 100.0)
    fig, ax = plt.subplots(figsize=(6, 3))
    pct.plot(kind="bar", ax=ax, color="#60a5fa")
    ax.set_title("Regime Distribution (% time)")
    ax.set_xlabel("Regime")
    ax.set_ylabel("%")
    ax.grid(True, axis="y", alpha=0.2)
    _save_fig(fig, out_path)


def _compute_drift_line(root: Path, scores: pd.DataFrame) -> Optional[pd.Series]:
    try:
        from joblib import load
        scaler = load(root / "acm_scaler.joblib")
        pca = load(root / "acm_pca.joblib")
    except Exception:
        return None
    num = scores.drop(columns=[c for c in scores.columns if c in DERIVED], errors="ignore")
    num = num.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    xs = scaler.transform(num.values)
    zp = pca.transform(xs)
    d = np.linalg.norm(zp, axis=1)
    return pd.Series(d, index=scores.index, name="DriftDistance")


def _plot_line(series: pd.Series, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    pd.to_numeric(series, errors="coerce").plot(ax=ax)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    _save_fig(fig, out_path)


def _events_from_jsonl(path: Path, scores: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["event_id","t0","t1","Start","End","Duration_s","Peak"])
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["event_id","t0","t1","Start","End","Duration_s","Peak"])
    ev = pd.DataFrame(rows)
    ev = ev.sort_values("t0").reset_index(drop=True)
    # Map to timestamps
    idx = scores.index
    def to_ts(pos: int) -> Optional[pd.Timestamp]:
        if pos < 0 or pos >= len(idx):
            return None
        return idx[pos]
    ev["Start"] = ev["t0"].map(lambda i: to_ts(int(i)))
    ev["End"] = ev["t1"].map(lambda i: to_ts(int(i)))
    fs = scores.get("FusedScore")
    if fs is not None:
        peaks = []
        for _, r in ev.iterrows():
            s = fs.iloc[int(r["t0"]): int(r["t1"])+1]
            peaks.append(float(pd.to_numeric(s, errors="coerce").max()) if len(s) else 0.0)
        ev["Peak"] = peaks
    ev["Duration_s"] = (pd.to_datetime(ev["End"]) - pd.to_datetime(ev["Start"])) .dt.total_seconds().fillna(0.0)
    return ev


def _health_grade(pct_healthy: float) -> str:
    if pct_healthy >= 95: return "A"
    if pct_healthy >= 90: return "B"
    if pct_healthy >= 80: return "C"
    if pct_healthy >= 70: return "D"
    return "E"


def _img_tag(rel_path: str, alt: str) -> str:
    return f"<img class='img-wide' src='{rel_path}' alt='{alt}'/>"


def _dq_table_html(root: Path) -> str:
    p = root / "dq.csv"
    if not p.exists():
        return "<div>No DQ table found.</div>"
    try:
        dq = pd.read_csv(p)
    except Exception:
        return "<div>Failed to read dq.csv</div>"
    cols = ["tag","nan_pct","flatline_ratio","spike_ratio","dropout_runs","dq_flag"]
    lower = {c.lower(): c for c in dq.columns}
    rows_html = []
    # thresholds
    for _, r in dq.iterrows():
        tag = r.get(lower.get("tag", "tag"), "?")
        nanp = float(r.get(lower.get("nan_pct", "nan_pct"), 0.0))
        flat = float(r.get(lower.get("flatline_ratio", "flatline_ratio"), 0.0))
        spike = float(r.get(lower.get("spike_ratio", "spike_ratio"), 0.0))
        note = []
        cls = ""
        if nanp >= 0.2 or flat > 0.95:
            cls = " style='background:#7f1d1d'"
        elif spike > 0.10:
            cls = " style='background:#78350f'"
        rows_html.append(
            f"<tr{cls}><td>{tag}</td><td>{nanp*100:.1f}%</td><td>{flat*100:.1f}%</td><td>{spike*100:.1f}%</td><td></td></tr>"
        )
    thead = "<tr><th>Tag</th><th>Dropout%</th><th>Flatline%</th><th>Spikes%</th><th>Comment</th></tr>"
    return f"<table><thead>{thead}</thead><tbody>{''.join(rows_html[:50])}</tbody></table>"


def _plot_event_miniplot(scores: pd.DataFrame, tags: List[str], t0: int, t1: int, out_path: Path) -> None:
    idx = scores.index
    n = len(idx)
    t0 = max(0, min(int(t0), n-1))
    t1 = max(0, min(int(t1), n-1))
    # Add context: +/- 60 samples around the event
    ctx = 60
    s_pos = max(0, t0 - ctx)
    e_pos = min(n - 1, t1 + ctx)
    win = scores.iloc[s_pos:e_pos+1].copy()
    # Ensure datetime x-axis, tz-naive
    if isinstance(win.index, pd.DatetimeIndex) and win.index.tz is not None:
        win.index = win.index.tz_convert(None)
    # Resample to 1-min for clarity if datetime
    if isinstance(win.index, pd.DatetimeIndex):
        win = win.resample("1min").median()
    # Choose top 3 available tags
    base = scores.drop(columns=[c for c in scores.columns if c in DERIVED], errors="ignore")
    chosen = [c for c in tags if c in base.columns][:3]
    if not chosen:
        var = base.var(numeric_only=True).sort_values(ascending=False)
        chosen = var.index.tolist()[:3]
    sub = win[chosen].apply(pd.to_numeric, errors="coerce")
    # Fill small gaps for viz
    sub = sub.interpolate(limit=3).ffill(limit=3).bfill(limit=3)
    # Robust z for overlay
    med = sub.median(numeric_only=True)
    mad = (sub - med).abs().median(numeric_only=True).replace(0.0, 1.0)
    z = (sub - med) / (1.4826 * mad)
    fig, ax = plt.subplots(figsize=(6, 3))
    z.plot(ax=ax, lw=1.0)
    # Highlight actual event core region
    if isinstance(win.index, pd.DatetimeIndex):
        core_start = scores.index[t0]
        core_end = scores.index[t1]
        if core_start.tz is not None: core_start = core_start.tz_convert(None)
        if core_end.tz is not None: core_end = core_end.tz_convert(None)
        ax.axvspan(core_start, core_end, color="#ef4444", alpha=0.08)
    ax.set_title(f"Event window ({e_pos - s_pos + 1} pts)")
    ax.grid(True, alpha=0.2)
    _save_fig(fig, out_path)


def build_native_report(art_dir: str | Path, equip: str, top_tags: int = 6) -> Path:
    equip_s = "".join(ch if ch.isalnum() else "_" for ch in equip)
    root = Path(art_dir) / equip_s
    images = root / "images_native"
    images.mkdir(parents=True, exist_ok=True)

    scores_csv = root / "scores.csv"
    scores = _read_scores(scores_csv)
    tags = _pick_tags(scores, top_n=top_tags)

    # Events and metrics
    events_path = root / "events.jsonl"
    events = _events_from_jsonl(events_path, scores)

    # Charts
    timeline_png = images / "timeline.png"
    _plot_timeline(scores, events, timeline_png)

    sampled_png = images / "sampled_raw.png"
    _plot_sampled_tags(scores, tags, sampled_png)

    regime_strip_png = images / "regimes.png"
    _plot_regime_strip(scores, regime_strip_png)
    regime_dist_png = images / "regime_dist.png"
    _plot_regime_distribution(scores, regime_dist_png)

    drift_series = _compute_drift_line(root, scores)
    drift_png = None
    if drift_series is not None:
        drift_png = images / "drift.png"
        _plot_line(drift_series, "Embedding Drift Distance", "distance", drift_png)

    # DQ disabled for now (section removed)

    # Header metrics
    fs = pd.to_numeric(scores.get("FusedScore"), errors="coerce") if "FusedScore" in scores else None
    if fs is not None:
        thr = float(fs.quantile(0.95))
        healthy_mask = fs < thr
        pct_healthy = float(healthy_mask.mean() * 100.0)
    else:
        pct_healthy = 100.0
    grade = _health_grade(pct_healthy)
    n_events = int(len(events))
    med_dur = float(events["Duration_s"].median()) if n_events else 0.0
    max_dur = float(events["Duration_s"].max()) if n_events else 0.0
    # MTTA/MTTR/freq
    if n_events >= 2:
        ev_starts = pd.to_datetime(events["Start"]).sort_values()
        mtta = float((ev_starts.diff().dt.total_seconds().dropna().median()))
    else:
        mtta = 0.0
    mttr = med_dur
    days = max(1.0, (scores.index[-1] - scores.index[0]).total_seconds() / 86400.0) if len(scores) >= 2 else 1.0
    freq_per_day = float(n_events / days)

    # Top contributing tags by |corr(tag, FusedScore)|
    top3 = []
    if fs is not None:
        num = scores.drop(columns=[c for c in scores.columns if c in DERIVED], errors="ignore")
        corr = {}
        for c in num.columns:
            s = pd.to_numeric(num[c], errors="coerce")
            if s.std() <= 1e-9: continue
            cc = s.corr(fs)
            if pd.notna(cc):
                corr[c] = abs(float(cc))
        top3 = sorted(corr, key=corr.get, reverse=True)[:3]

    # HTML layout
    # Generate event mini-plots (top 3)
    event_imgs_html = ""
    if not events.empty:
        ev_imgs = []
        for i, r in events.head(3).iterrows():
            img_path = images / f"event_{int(r['event_id']) if 'event_id' in r else i+1}.png"
            _plot_event_miniplot(scores, tags, int(r.get('t0', 0)), int(r.get('t1', 0)), img_path)
            ev_imgs.append(_img_tag(str(img_path.relative_to(root)), 'event'))
        event_imgs_html = "<div style='display:flex; flex-wrap:wrap; gap:10px; margin-top:10px;'>" + "".join(
            f"<div style='flex:1 1 300px'>{img}</div>" for img in ev_imgs
        ) + "</div>"

    html = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'/>
  <title>ACMnxt Report - {equip_s}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#0b1220; color:#e5e7eb; margin:0; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 16px; }}
    .section {{ background:#0f1621; border:1px solid #1f2a37; border-radius:10px; padding:14px 16px; margin:12px 0; }}
    .img-wide {{ max-width: 100%; height: auto; display: block; border:1px solid #334155; border-radius:8px; }}
    a {{ color:#93c5fd; }}
    h2 {{ margin-top:0; }}
  </style>
  </head>
  <body>
    <main>
      <div class='section'>
        <h2>Header Summary</h2>
        <div>Equipment: <b>{equip}</b></div>
        <div>Time window: {scores.index.min()} → {scores.index.max()} &nbsp; | &nbsp; Generated: {pd.Timestamp.utcnow()}</div>
        <div>Health grade: <b>{grade}</b> &nbsp; | &nbsp; Healthy: {pct_healthy:.1f}% &nbsp; | &nbsp; Events: {n_events} &nbsp; | &nbsp; MTTA: {mtta/60:.1f} min &nbsp; | &nbsp; MTTR: {mttr/60:.1f} min &nbsp; | &nbsp; Freq/day: {freq_per_day:.2f}</div>
        <div>Top tags: {', '.join(top3) if top3 else 'N/A'}</div>
      </div>
      <div class='section'>
        <h2>Health Overview</h2>
        {_img_tag(str(timeline_png.relative_to(root)), 'timeline')}
      </div>
      <!-- Data Quality section temporarily removed -->
      <div class='section'>
        <h2>Tag Trends (sampled)</h2>
        {_img_tag(str(sampled_png.relative_to(root)), 'sampled raw')}
        {_img_tag(str(regime_strip_png.relative_to(root)), 'regimes')}
      </div>
      <div class='section'>
        <h2>Regimes & Drift</h2>
        {_img_tag(str(regime_dist_png.relative_to(root)), 'regime distribution')}
        {( _img_tag(str(drift_png.relative_to(root)), 'drift') if drift_png else '<div>No drift plot (no artifacts found).</div>' )}
      </div>
      <div class='section'>
        <h2>Event Summary</h2>
        <div>Events: {n_events} | Median duration: {med_dur/60:.1f} min | Longest: {max_dur/60:.1f} min</div>
        <table><thead><tr><th>Event #</th><th>Start</th><th>End</th><th>Duration (min)</th><th>Peak</th></tr></thead><tbody>
        {''.join(f"<tr><td>{int(r.event_id) if 'event_id' in r else i+1}</td><td>{r.Start}</td><td>{r.End}</td><td>{(r.Duration_s or 0)/60:.1f}</td><td>{getattr(r,'Peak',0):.2f}</td></tr>" for i, r in events.head(5).iterrows()) if not events.empty else '<tr><td colspan=5>No events</td></tr>'}
        </tbody></table>
        {event_imgs_html}
      </div>
    </main>
  </body>
</html>
"""
    out_html = root / "report_native.html"
    out_html.write_text(html, encoding="utf-8")
    return out_html

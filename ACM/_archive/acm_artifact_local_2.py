# acm_artifact_local.py
# Builds HTML report with summary, plots for fused/anomaly heads/regimes,
# and tables for Top Drift, Latest Events, Context Masks coverage.
import os, base64, io, datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

ART_DIR = r"C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def _save_plot(series_or_df, title, fname):
    plt.figure(figsize=(11,4))
    ax = plt.gca()
    if isinstance(series_or_df, pd.Series):
        series_or_df.plot(ax=ax)
    else:
        series_or_df.plot(ax=ax)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(ART_DIR, fname)
    plt.savefig(path, dpi=120)
    plt.close()
    return path

def _b64(path):
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode("ascii")

def build_report(scored_csv=None, drift_csv=None, events_csv=None, masks_csv=None, title="Asset Condition Monitor – Report"):
    if scored_csv is None:
        scored_csv = os.path.join(ART_DIR, "acm_scored_window.csv")
    df = pd.read_csv(scored_csv, index_col=0, parse_dates=True)

    # basic plots
    p1 = _save_plot(df["FusedScore"], "Fused Score", "fused.png")
    cols = [c for c in ["H1_Forecast","H2_Recon","H3_Contrast"] if c in df.columns]
    p2 = _save_plot(df[cols], "Head Scores", "heads.png") if cols else None
    p3 = _save_plot(df["Regime"], "Regime (cluster id)", "regime.png") if "Regime" in df.columns else None

    # drift
    drift_html = ""
    if drift_csv is None:
        dpath = os.path.join(ART_DIR, "acm_drift.csv")
        drift_csv = dpath if os.path.exists(dpath) else None
    if drift_csv and os.path.exists(drift_csv):
        drift = pd.read_csv(drift_csv).sort_values("DriftZ", ascending=False)
        top = drift.head(15)
        rows = "\n".join(f"<tr><td>{r.Tag}</td><td style='text-align:right'>{r.DriftZ:.2f}</td></tr>" for r in top.itertuples())
        drift_html = f"""
        <h2>Top Drifted Tags</h2>
        <table border="1" cellpadding="6" cellspacing="0">
          <thead><tr><th>Tag</th><th>Drift Z</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    # events
    if events_csv is None:
        epath = os.path.join(ART_DIR, "acm_events.csv")
        events_csv = epath if os.path.exists(epath) else None
    events_html = ""
    if events_csv and os.path.exists(events_csv):
        ev = pd.read_csv(events_csv, parse_dates=["Start","End"])
        last = ev.tail(10)
        rows = "\n".join(f"<tr><td>{r.Start}</td><td>{r.End}</td><td style='text-align:right'>{r.PeakScore:.2f}</td></tr>"
                         for r in last.itertuples())
        events_html = f"""
        <h2>Latest Events</h2>
        <table border="1" cellpadding="6" cellspacing="0">
          <thead><tr><th>Start</th><th>End</th><th>Peak</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    # masks coverage
    if masks_csv is None:
        mpath = os.path.join(ART_DIR, "acm_context_masks.csv")
        masks_csv = mpath if os.path.exists(mpath) else None
    masks_html = ""
    if masks_csv and os.path.exists(masks_csv):
        mk = pd.read_csv(masks_csv)
        cov = 100.0 * (mk["Mask"].astype(int).sum() / max(1, len(mk)))
        masks_html = f"<h2>Transient Context</h2><div>Mask coverage: <b>{cov:.2f}%</b> of scored rows</div>"

    imgs = [f'<h3>Fused</h3><img src="data:image/png;base64,{_b64(p1)}" style="width:100%;max-width:1100px;border:1px solid #333;border-radius:8px;margin:6px 0;" />']
    if p2: imgs.append(f'<h3>Heads</h3><img src="data:image/png;base64,{_b64(p2)}" style="width:100%;max-width:1100px;border:1px solid #333;border-radius:8px;margin:6px 0;" />')
    if p3: imgs.append(f'<h3>Regime</h3><img src="data:image/png;base64,{_b64(p3)}" style="width:100%;max-width:1100px;border:1px solid #333;border-radius:8px;margin:6px 0;" />')

    anom_rate = 100.0 * (df["FusedScore"] >= 0.7).mean()
    regimes_seen = int(df["Regime"].nunique()) if "Regime" in df.columns else 0
    t0, t1 = df.index.min(), df.index.max()

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/><title>{title}</title>
<style>
body {{ background:#0b0f14; color:#eef1f4; font-family:Segoe UI,Roboto,Arial; }}
h1,h2,h3 {{ color:#e7eaf0; }}
.card {{ background:#111827; padding:12px 16px; border-radius:10px; border:1px solid #1f2937; max-width:1100px; }}
table {{ border-collapse:collapse; width:100%; max-width:1100px; }}
th,td {{ border:1px solid #444; }}
.small {{ color:#aab3bf; }}
</style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <div class="small">Generated: {dt.datetime.now().isoformat()}</div>
    <h2>Summary</h2>
    <ul>
      <li><b>Window:</b> {t0} → {t1}</li>
      <li><b>Events (fused ≥ 0.70):</b> {int((df["FusedScore"]>=0.7).sum())}</li>
      <li><b>Event rate:</b> {anom_rate:.2f}%</li>
      <li><b>Regimes observed:</b> {regimes_seen}</li>
      <li><b>Rows:</b> {len(df):,}</li>
    </ul>
    <h2>Visuals</h2>
    {''.join(imgs)}
    {drift_html}
    {events_html}
    {masks_html}
  </div>
</body>
</html>"""
    out = os.path.join(ART_DIR, "acm_report.html")
    with open(out, "w", encoding="utf-8") as f: f.write(html)
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("ACM Artifact")
    p.add_argument("--scored_csv", default=os.path.join(ART_DIR,"acm_scored_window.csv"))
    p.add_argument("--drift_csv",  default=os.path.join(ART_DIR,"acm_drift.csv"))
    p.add_argument("--events_csv", default=os.path.join(ART_DIR,"acm_events.csv"))
    p.add_argument("--masks_csv",  default=os.path.join(ART_DIR,"acm_context_masks.csv"))
    a = p.parse_args()
    path = build_report(a.scored_csv, a.drift_csv if os.path.exists(a.drift_csv) else None,
                        a.events_csv if os.path.exists(a.events_csv) else None,
                        a.masks_csv if os.path.exists(a.masks_csv) else None)
    print(f"Report: {path}")

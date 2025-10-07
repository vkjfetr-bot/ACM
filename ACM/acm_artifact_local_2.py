# acm_artifact_local.py
# Creates visual & tabular artifacts from core outputs (no external servers).

import os, base64, io, datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = r"C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts"
os.makedirs(ART_DIR, exist_ok=True)
print(f"[ACM] Using static ART_DIR: {ART_DIR}")

def _save_plot(df: pd.DataFrame, title: str, ycols, fname: str):
    plt.figure(figsize=(11,4))
    df[ycols].plot(ax=plt.gca())
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(ART_DIR, fname)
    plt.savefig(path, dpi=120)
    plt.close()
    return path

def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def build_report(scored_csv: str,
                 drift_csv: str = None,
                 max_tags_plot: int = 4,
                 title: str = "Asset Condition Monitor – Report") -> str:
    df = pd.read_csv(scored_csv, parse_dates=True, index_col=0)
    if "Anomaly" not in df.columns or "Regime" not in df.columns:
        raise ValueError("Scored CSV must contain 'Anomaly' and 'Regime' columns.")

    # Pick some signals to plot
    tag_cols = [c for c in df.columns if not any(x in c for x in ["_ma_","_std_","_slope_","Anomaly","Regime"])]
    tag_cols = tag_cols[:max_tags_plot] if tag_cols else []
    plots = []
    if tag_cols:
        plots.append(_save_plot(df, "Signals (sample)", tag_cols, "signals.png"))
    plots.append(_save_plot(df, "Anomaly (0/1)", ["Anomaly"], "anomaly.png"))
    plots.append(_save_plot(df, "Regime (cluster id)", ["Regime"], "regime.png"))

    drift_tbl = None
    if drift_csv and os.path.exists(drift_csv):
        drift_tbl = pd.read_csv(drift_csv).sort_values("DriftZ", ascending=False)

    # Build simple HTML (single file)
    imgs_html = "".join(
        f'<h3>{os.path.splitext(os.path.basename(p))[0].title()}</h3>'
        f'<img src="data:image/png;base64,{_img_to_b64(p)}" style="width:100%;max-width:1100px;border:1px solid #333;border-radius:8px;margin:6px 0;" />'
        for p in plots
    )

    # Small summary
    anom_rate = 100.0 * df["Anomaly"].mean()
    regimes_seen = int(df["Regime"].nunique())
    t0, t1 = df.index.min(), df.index.max()

    summary_html = f"""
    <ul>
      <li><b>Window:</b> {t0} → {t1}</li>
      <li><b>Anomaly rate:</b> {anom_rate:.2f}%</li>
      <li><b>Regimes observed:</b> {regimes_seen}</li>
      <li><b>Rows:</b> {len(df):,}</li>
    </ul>
    """

    drift_html = ""
    if drift_tbl is not None and not drift_tbl.empty:
        drift_tbl.to_csv(os.path.join(ART_DIR, "drift_top.csv"), index=False)
        top = drift_tbl.head(15)
        rows = "\n".join(f"<tr><td>{r.Tag}</td><td style='text-align:right'>{r.DriftZ:.2f}</td></tr>"
                         for r in top.itertuples())
        drift_html = f"""
        <h2>Top Drifted Tags</h2>
        <table border="1" cellpadding="6" cellspacing="0">
          <thead><tr><th>Tag</th><th>Drift Z</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ background:#0b0f14; color:#eef1f4; font-family:Segoe UI,Roboto,Arial; }}
h1,h2,h3 {{ color:#e7eaf0; }}
table {{ border-collapse:collapse; width:100%; max-width:1100px; }}
th,td {{ border:1px solid #444; }}
.small {{ color:#aab3bf; }}
.card {{ background:#111827; padding:12px 16px; border-radius:10px; border:1px solid #1f2937; max-width:1100px; }}
</style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <div class="small">Generated: {dt.datetime.now().isoformat()}</div>
    <h2>Summary</h2>
    {summary_html}
    <h2>Visuals</h2>
    {imgs_html}
    {drift_html}
  </div>
</body>
</html>
"""
    out_html = os.path.join(ART_DIR, "acm_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser("ACM Artifact (local)")
    p.add_argument("--scored_csv", default=os.path.join(ART_DIR, "acm_scored_window.csv"))
    p.add_argument("--drift_csv",  default=os.path.join(ART_DIR, "acm_drift.csv"))
    args = p.parse_args()
    path = build_report(args.scored_csv, args.drift_csv if os.path.exists(args.drift_csv) else None)
    print(f"Report: {path}")

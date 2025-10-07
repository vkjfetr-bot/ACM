# report_main.py
# Basic ACM report builder (charts + tables; no cards, no JS).
# Reads CSVs from ART_DIR (override via env ACM_ART_DIR) and writes acm_report_basic.html.

import os
from typing import Optional, List
import numpy as np
import pandas as pd

from report_html import wrap_html, section, table, chart, kpi_grid
from report_charts import timeline, sampled_tags_with_marks, drift_bars, FUSED_TAU

# ---------- Settings ----------
ART_DIR = os.environ.get("ACM_ART_DIR", r"C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts")
TITLE   = "Asset Condition Monitor — Basic Report"
SPARKS_N = 18  # how many tags to show on sampled plots

# ---------- IO ----------
def _safe_read_csv(path, **kw):
    return pd.read_csv(path, **kw) if (path and os.path.exists(path)) else None

# ---------- Logic ----------
def _pick_key_tags(scored: pd.DataFrame, drift: Optional[pd.DataFrame]) -> List[str]:
    """Prefer top drifted tags; fallback to top-variance non-derived columns."""
    derived = {"Regime","FusedScore","H1_Forecast","H2_Recon","H3_Contrast","CorrBoost","CPD","ContextMask"}
    if drift is not None and {"Tag","DriftZ"}.issubset(drift.columns):
        cand = (drift.dropna(subset=["DriftZ"])
                     .sort_values("DriftZ", ascending=False)["Tag"].tolist())
        cand = [c for c in cand if c not in derived]
        if cand:
            return cand[:SPARKS_N]
    # variance fallback
    numcols = [c for c in scored.columns if c not in derived]
    if not numcols:
        return []
    v = scored[numcols].apply(pd.to_numeric, errors="coerce").var().sort_values(ascending=False)
    return [c for c in v.index.tolist() if c not in derived][:SPARKS_N]

def compute_dq(raw_df: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
    """Simple DQ: %flatline, %dropout, spike count (IQR-of-diff rule)."""
    rows = []
    for t in tags or []:
        if t not in raw_df.columns:
            continue
        s = pd.to_numeric(raw_df[t], errors="coerce").astype(float)
        n = len(s)
        if n == 0:
            continue
        flat = (s.diff().abs() < 1e-12).sum() / n * 100.0
        drop = s.isna().sum() / n * 100.0
        d = s.diff().dropna().abs()
        iqr = (d.quantile(0.75) - d.quantile(0.25)) + 1e-9
        spikes = int((d > 5 * iqr).sum())
        rows.append({"Tag": t, "Flatline%": flat, "Dropout%": drop, "Spikes": spikes})
    if not rows:
        return pd.DataFrame(columns=["Tag","Flatline%","Dropout%","Spikes"])
    return pd.DataFrame(rows).sort_values(["Flatline%","Dropout%","Spikes"], ascending=[False, False, False])

def explain_block() -> str:
    """Glossary of terms shown in charts/tables."""
    hdrs = ["Term", "Meaning"]
    rows = [
        ("FusedScore",
         "Final anomaly score combining: H1 (forecast error), H2 (PCA reconstruction error), H3 (embedding drift). Higher = more anomalous."),
        (f"τ (tau = {FUSED_TAU})",
         f"Anomaly threshold. Points with FusedScore ≥ {FUSED_TAU} are highlighted as anomalies."),
        ("Regime",
         "Discrete operating mode (cluster) inferred from data; helps compare behavior within the same state."),
        ("Mask",
         "Transient periods (e.g., start-up/maintenance) down-weighted in fusion; shown as a thin red band on charts."),
        ("DriftZ",
         "Z-scored magnitude of a tag’s distribution shift vs baseline; higher = stronger distribution change."),
    ]
    return table(hdrs, rows)

# ---------- Build ----------
def build_basic_report():
    # Load artifacts
    scored = _safe_read_csv(os.path.join(ART_DIR, "acm_scored_window.csv"), index_col=0, parse_dates=True)
    if scored is None or scored.empty:
        raise FileNotFoundError(f"Missing or empty: {os.path.join(ART_DIR, 'acm_scored_window.csv')}")
    events = _safe_read_csv(os.path.join(ART_DIR, "acm_events.csv"), parse_dates=["Start","End"])
    drift  = _safe_read_csv(os.path.join(ART_DIR, "acm_drift.csv"))
    masks  = _safe_read_csv(os.path.join(ART_DIR, "acm_context_masks.csv"))

    # KPIs
    t0, t1 = str(scored.index.min()), str(scored.index.max())
    events_n = int((scored["FusedScore"] >= FUSED_TAU).sum())
    regimes_seen = int(scored["Regime"].nunique()) if "Regime" in scored.columns else 0
    mask_cov = None
    if masks is not None and not masks.empty and "Mask" in masks.columns:
        mask_cov = 100.0 * (masks["Mask"].astype(int).sum() / max(1, len(masks)))

    kpis = [
        ("Window", f"{t0} → {t1}"),
        ("Rows", f"{len(scored):,}"),
        ("Regimes", regimes_seen),
        ("Events ≥ τ", events_n),
        ("Mask %", f"{mask_cov:.2f}%" if mask_cov is not None else "—"),
    ]

    # Assemble body
    body = ""
    body += kpi_grid(kpis)

    # Timeline chart with event & mask overlays
    body += section("Timeline (Fused + Heads + Regime)",
                    chart(timeline(scored, events, masks)))

    # Sampled data plots (z-normalized) with anomaly markers + event shading
    key_tags = _pick_key_tags(scored, drift)
    body += section("Sampled Data (z-normalized) with Anomalies & Events",
                    chart(sampled_tags_with_marks(scored, key_tags, events, scored["FusedScore"])))

    # Latest Events (recency-first, top 30)
    if events is not None and not events.empty:
        ev = events.copy().sort_values("End", ascending=False).head(30)
        hdrs = ["Start", "End", "PeakScore", "Duration (hh:mm:ss)"]
        rows = []
        for r in ev.itertuples():
            dur = pd.to_datetime(r.End) - pd.to_datetime(r.Start)
            peak = getattr(r, "PeakScore", np.nan)
            rows.append([str(r.Start), str(r.End), (f"{float(peak):.2f}" if pd.notna(peak) else "—"), str(dur)])
        body += section("Latest Events", table(hdrs, rows))
    else:
        body += section("Latest Events", "<div class='small'>No events file found.</div>")

    # Drift (chart + table)
    if drift is not None and not drift.empty and {"Tag","DriftZ"}.issubset(drift.columns):
        body += section("Drift (Top 20)", chart(drift_bars(drift, 20)))
        top = drift.dropna(subset=["DriftZ"]).sort_values("DriftZ", ascending=False).head(30)[["Tag","DriftZ"]]
        body += section("Drift — Table", table(["Tag","DriftZ"], [[t, f"{z:.2f}"] for t, z in top.itertuples(index=False)]))
    else:
        body += section("Drift", "<div class='small'>No drift file found.</div>")

    # Data Quality
    dq = compute_dq(scored, key_tags)
    if not dq.empty:
        body += section("Data Quality",
                        table(["Tag","Flatline %","Dropout %","Spikes"],
                              [[r.Tag, f"{r._2:.2f}", f"{r._3:.2f}", r.Spikes] for r in dq.itertuples()]))
    else:
        body += section("Data Quality", "<div class='small'>No DQ issues computed.</div>")

    # Glossary / explanations
    body += section("Glossary (Model/Analysis Terms)", explain_block())

    # Wrap & write
    html = wrap_html(TITLE, body)
    out = os.path.join(ART_DIR, "acm_report_basic.html")
    os.makedirs(ART_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("Report:", out)
    return out

if __name__ == "__main__":
    build_basic_report()

# acm_artifact_local.py
# Report 2.0 — clear timeline story, event drill-downs, tag health, drift/stability,
# data-quality, model cards, and timings. Static HTML, self-contained images (PNG -> base64).
#
# Inputs (defaults to acm_artifacts/*):
#   - acm_scored_window.csv         (required)
#   - acm_events.csv                (optional)
#   - acm_drift.csv                 (optional)
#   - acm_context_masks.csv         (optional)
#   - run_*.jsonl                   (optional; timings)
#   - acm_manifest.json             (optional; model/config metadata)
#
# Output:
#   - acm_report.html (in ART_DIR)
#
# Notes:
#   - Uses matplotlib only (no seaborn). No external JS/CSS; pure HTML+PNG data URIs.

import os, io, base64, glob, json, datetime as dt
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------- Settings ----------
ART_DIR = r"C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts"
TITLE   = "Asset Condition Monitor — Report"
FUSED_TAU = 0.70           # threshold line on fused timeline
EVENT_CARDS_N = 6          # how many latest events to show
SPARKS_N = 20              # how many key tags to plot as sparklines
DRIFT_TOP = 20             # how many drift bars
FIG_DPI = 130

os.makedirs(ART_DIR, exist_ok=True)

# ---------- Helpers ----------
def _embed_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _minmax_index(idx):
    try:
        return idx.min(), idx.max()
    except Exception:
        return None, None

def _safe_read_csv(path, **kw):
    return pd.read_csv(path, **kw) if (path and os.path.exists(path)) else None

def _read_latest_runlog() -> Optional[pd.DataFrame]:
    logs = sorted(glob.glob(os.path.join(ART_DIR, "run_*.jsonl")))
    if not logs: return None
    df = pd.read_json(logs[-1], lines=True)
    return df

def _human_dur(sec: float) -> str:
    if sec < 1: return f"{sec*1000:.0f} ms"
    return f"{sec:.2f} s"

def _pick_key_tags(scored: pd.DataFrame, drift: Optional[pd.DataFrame]) -> List[str]:
    derived = {"Regime","FusedScore","H1_Forecast","H2_Recon","H3_Contrast","CorrBoost","CPD","ContextMask"}
    if drift is not None and "Tag" in drift.columns and "DriftZ" in drift.columns:
        cand = drift.dropna(subset=["DriftZ"]).sort_values("DriftZ", ascending=False)["Tag"].tolist()
        if cand:
            return cand[:SPARKS_N]
    # variance fallback over numeric, non-derived columns
    numcols = []
    for c in scored.columns:
        if c in derived: 
            continue
        s = pd.to_numeric(scored[c], errors="coerce")
        if s.notna().sum() > 0:
            numcols.append(c)
    if numcols:
        v = scored[numcols].apply(pd.to_numeric, errors="coerce").var().sort_values(ascending=False)
        return [c for c in v.index.tolist() if c not in derived][:SPARKS_N]
    return []


# ---------- Plots ----------
def plot_timeline(scored: pd.DataFrame, events: Optional[pd.DataFrame], masks: Optional[pd.DataFrame]) -> str:
    """Fused timeline with event shading + mask overlay + tau; heads mini; regime ribbon; corr/cpd mini."""
    # 4-row vertical layout: Fused, Heads, Ribbon, Corr/CPD
    has_heads = all(c in scored.columns for c in ["H1_Forecast","H2_Recon","H3_Contrast"])
    has_reg   = "Regime" in scored.columns
    has_corr  = "CorrBoost" in scored.columns
    has_cpd   = "CPD" in scored.columns
    ts = pd.to_datetime(scored.index)

    rows = 1 + (1 if has_heads else 0) + (1 if has_reg else 0) + (1 if (has_corr or has_cpd) else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(13, 2.5*rows), sharex=True)
    if rows == 1: axes = [axes]
    axi = 0

    # Row 1: fused timeline
    ax = axes[axi]; axi += 1
    ax.plot(ts, scored["FusedScore"].values, linewidth=1.5)
    ax.axhline(FUSED_TAU, color="gray", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Fused")
    ax.grid(alpha=0.25)

    # Shaded events
    if events is not None and not events.empty:
        for r in events.itertuples():
            ax.axvspan(r.Start, r.End, color="#f59e0b", alpha=0.25)
            if hasattr(r, "PeakScore"):
                ax.text(r.Start, min(0.98, getattr(r, "PeakScore", 0)+0.02), f"{getattr(r, 'PeakScore', 0):.2f}", fontsize=8, color="#f59e0b")

    # Mask overlay (red band at the bottom)
    if masks is not None and "Ts" in masks.columns and "Mask" in masks.columns:
        mk = masks.copy()
        mk["Ts"] = pd.to_datetime(mk["Ts"])
        mk = mk.set_index("Ts").reindex(ts, method="nearest")
        mask_y = np.where(mk["Mask"].fillna(0).values > 0, 0.03, np.nan)
        ax.plot(ts, mask_y, drawstyle="steps-mid", linewidth=2.5, alpha=0.9, color="#ef4444")

    # Row 2: heads mini
    if has_heads:
        ax = axes[axi]; axi += 1
        ax.plot(ts, scored["H1_Forecast"].values, linewidth=1.0, label="H1")
        ax.plot(ts, scored["H2_Recon"].values, linewidth=1.0, label="H2")
        ax.plot(ts, scored["H3_Contrast"].values, linewidth=1.0, label="H3")
        ax.set_ylabel("Heads")
        ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=8)

    # Row 3: regime ribbon (FIXED pcolormesh shapes)
    if has_reg:
        ax = axes[axi]; axi += 1
        reg = scored["Regime"].astype(int)

        # Use date numbers + pcolormesh (avoid int overflow from ns timestamps)
        x = mdates.date2num(ts)
        if len(x) >= 2:
            dx = np.diff(x)
            last_step = dx[-1] if len(dx) else (x[-1] - x[-2])
            if not np.isfinite(last_step) or last_step == 0:
                last_step = 1/1440  # +1 minute
            edges = np.concatenate([x, [x[-1] + last_step]])
        else:
            # single-point fallback: create a tiny 1-minute span
            edges = np.array([x[0], x[0] + 1/1440])

        # Make Z single-row so Y needs exactly 2 edges → avoids mismatch
        Z = reg.values[np.newaxis, :]           # shape (1, N)
        y_edges = np.array([0, 1])              # length 2 = M+1 when M=1

        # Safety check: X edges must be N+1
        if edges.shape[0] != Z.shape[1] + 1:
            raise ValueError(f"Regime ribbon: edges length {edges.shape[0]} must be one more than Z columns {Z.shape[1]}")

        ax.pcolormesh(edges, y_edges, Z, cmap="tab20", shading="auto")
        ax.set_yticks([]); ax.set_ylabel("Regime"); ax.grid(False)

    # Row 4: corr/cpd
    if has_corr or has_cpd:
        ax = axes[axi]; axi += 1
        if has_corr: ax.plot(ts, scored["CorrBoost"].values, linewidth=1.0, label="CorrBoost")
        if has_cpd:  ax.plot(ts, scored["CPD"].values, linewidth=1.0, label="CPD")
        ax.set_ylabel("Extras"); ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Time")
    fig.tight_layout()
    return _embed_png(fig)

def _clip_time(df: pd.DataFrame, start, end, pad="10min") -> pd.DataFrame:
    s = pd.to_datetime(start) - pd.to_timedelta(pad)
    e = pd.to_datetime(end) + pd.to_timedelta(pad)
    return df.loc[(df.index >= s) & (df.index <= e)]

def plot_event_time(scored: pd.DataFrame, ev: pd.Series) -> str:
    seg = _clip_time(scored, ev["Start"], ev["End"])
    ts = pd.to_datetime(seg.index)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(ts, seg["FusedScore"].values, linewidth=1.5, label="Fused")
    for c in ["H1_Forecast","H2_Recon","H3_Contrast"]:
        if c in seg.columns:
            ax.plot(ts, seg[c].values, linewidth=1.0, label=c)
    ax.axhline(FUSED_TAU, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title(f"Event {ev['Start']} → {ev['End']}")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=8)
    return _embed_png(fig)

def plot_event_spectrum(scored: pd.DataFrame, raw_df: pd.DataFrame, ev: pd.Series, tags: List[str]) -> str:
    # Compare spectrum in-event vs. pre-event baseline (same duration); aggregate by median
    try:
        import numpy.fft as nfft
    except Exception:
        return ""  # no FFT
    start, end = pd.to_datetime(ev["Start"]), pd.to_datetime(ev["End"])
    dur = end - start
    pre_s, pre_e = start - dur, start
    use_cols = [c for c in tags if c in raw_df.columns]
    if not use_cols: return ""
    seg_ev  = raw_df.loc[(raw_df.index >= start) & (raw_df.index <= end), use_cols].astype(float)
    seg_pre = raw_df.loc[(raw_df.index >= pre_s) & (raw_df.index <= pre_e), use_cols].astype(float)
    if len(seg_ev) < 8 or len(seg_pre) < 8: return ""
    # windowing to equal length
    n = min(len(seg_ev), len(seg_pre))
    Ev  = seg_ev.tail(n).values
    Pre = seg_pre.tail(n).values
    # rfft per column then median across tags
    def _med_spec(X):
        mags = []
        for i in range(X.shape[1]):
            spec = np.abs(nfft.rfft(X[:, i]))
            mags.append(spec)
        # pad to same length
        L = max(len(s) for s in mags)
        mags = [np.pad(s, (0, L-len(s))) for s in mags]
        return np.median(np.vstack(mags), axis=0)
    sp_ev  = _med_spec(Ev)
    sp_pre = _med_spec(Pre)
    k = np.arange(len(sp_ev))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(k, sp_ev, linewidth=1.0, label="Event")
    ax.plot(k, sp_pre, linewidth=1.0, label="Baseline")
    diff = sp_ev - sp_pre
    # highlight bins above +kσ of diff
    sd = np.std(diff) + 1e-9
    hi = np.where(diff > 2.5*sd)[0]
    if len(hi):
        ax.scatter(hi, sp_ev[hi], s=10)
    ax.set_title("Event vs. Baseline Spectrum (median over key tags)")
    ax.set_xlabel("FFT bin"); ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=8)
    return _embed_png(fig)

def plot_sparklines(scored: pd.DataFrame, tags: List[str], fused: Optional[pd.Series]) -> str:
    # Small multiples grid: each tag normalized z with anomaly dots
    rows = int(np.ceil(len(tags)/5)) or 1
    fig, axes = plt.subplots(rows, 5, figsize=(13, 2.0*rows), sharex=True)
    axes = np.atleast_2d(axes)
    ts = pd.to_datetime(scored.index)
    for i, t in enumerate(tags):
        r, c = divmod(i, 5)
        ax = axes[r, c]
        if t in scored.columns:
            s = scored[t].astype(float)
        else:
            ax.axis("off"); continue
        # normalize (z)
        z = (s - s.mean()) / (s.std() + 1e-9)
        ax.plot(ts, z.values, linewidth=0.8)
        if fused is not None:
            # mark points where fused >= tau
            mark = np.where(fused.values >= FUSED_TAU, z.values, np.nan)
            ax.scatter(ts, mark, s=6)
        ax.set_title(t, fontsize=8)
        ax.grid(alpha=0.15)
    # hide empties
    for j in range(i+1, rows*5):
        r, c = divmod(j, 5); axes[r, c].axis("off")
    fig.tight_layout()
    return _embed_png(fig)

def plot_drift_bars(drift: pd.DataFrame) -> str:
    top = drift.sort_values("DriftZ", ascending=False).head(DRIFT_TOP)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Tag"][::-1], top["DriftZ"][::-1])
    ax.set_xlabel("Drift Z"); ax.set_title(f"Top {len(top)} Drifted Tags")
    fig.tight_layout()
    return _embed_png(fig)

# ---------- DQ & Timings ----------
def compute_dq(raw_df: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
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

    # Guard: if nothing to report, return empty frame with expected columns
    if not rows:
        return pd.DataFrame(columns=["Tag", "Flatline%", "Dropout%", "Spikes"])

    df = pd.DataFrame(rows)
    # Ensure all expected cols exist before sort (paranoia)
    for c in ["Flatline%", "Dropout%", "Spikes"]:
        if c not in df.columns:
            df[c] = np.nan
    return df.sort_values(["Flatline%", "Dropout%", "Spikes"], ascending=[False, False, False])


def timings_table(runlog: Optional[pd.DataFrame]) -> Tuple[str, List[Dict]]:
    if runlog is None or runlog.empty:
        return "", []
    # compute duration by block (end entries already have duration_s)
    ends = runlog[runlog["event"]=="end"].copy()
    ends = ends.sort_values(["ts"])
    totals = ends.groupby("block")["duration_s"].sum().reset_index().sort_values("duration_s", ascending=False)
    rows = []
    for r in totals.itertuples():
        rows.append({"Block": r.block, "Duration": _human_dur(r.duration_s)})
    # HTML table
    tr = "\n".join(f"<tr><td>{x['Block']}</td><td style='text-align:right'>{x['Duration']}</td></tr>" for x in rows)
    html = f"""
    <h2>Performance (Timings)</h2>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:1100px">
      <thead><tr><th>Block</th><th>Duration</th></tr></thead>
      <tbody>{tr}</tbody>
    </table>"""
    return html, rows

# ---------- Report Builder ----------
def build_report(scored_csv=None, drift_csv=None, events_csv=None, masks_csv=None, title=TITLE):
    # Paths
    if scored_csv is None:
        scored_csv = os.path.join(ART_DIR, "acm_scored_window.csv")
    if drift_csv is None and os.path.exists(os.path.join(ART_DIR, "acm_drift.csv")):
        drift_csv = os.path.join(ART_DIR, "acm_drift.csv")
    if events_csv is None and os.path.exists(os.path.join(ART_DIR, "acm_events.csv")):
        events_csv = os.path.join(ART_DIR, "acm_events.csv")
    if masks_csv is None and os.path.exists(os.path.join(ART_DIR, "acm_context_masks.csv")):
        masks_csv = os.path.join(ART_DIR, "acm_context_masks.csv")

    # Load
    scored = _safe_read_csv(scored_csv, index_col=0, parse_dates=True)
    if scored is None or scored.empty:
        raise FileNotFoundError(f"Missing or empty scored csv: {scored_csv}")
    events = _safe_read_csv(events_csv, parse_dates=["Start","End"])
    drift  = _safe_read_csv(drift_csv)
    masks  = _safe_read_csv(masks_csv)

    # Try to load raw df for event spectra (approximate with scored columns)
    raw_df = scored.copy()

    # Metadata: manifest + runlog
    manifest_path = os.path.join(ART_DIR, "acm_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    runlog = _read_latest_runlog()

    # KPIs
    t0, t1 = _minmax_index(scored.index)
    eq_score_path = os.path.join(ART_DIR, "acm_equipment_score.csv")
    eq_score = None
    if os.path.exists(eq_score_path):
        try:
            eq_score = float(pd.read_csv(eq_score_path)["EquipmentScore"].iloc[0])
        except Exception:
            pass
    events_n = int((scored["FusedScore"] >= FUSED_TAU).sum())
    regimes_seen = int(scored["Regime"].nunique()) if "Regime" in scored.columns else 0
    mask_cov = None
    if masks is not None and not masks.empty:
        if "Mask" in masks.columns:
            mask_cov = 100.0 * (masks["Mask"].astype(int).sum() / max(1, len(masks)))

    # Timeline story
    img_timeline = plot_timeline(scored, events, masks)

    # Tag health grid
    key_tags = _pick_key_tags(scored, drift)
    img_sparks = plot_sparklines(scored, key_tags, scored["FusedScore"])

    # Drift bars
    drift_html = ""
    if drift is not None and not drift.empty:
        img_drift = plot_drift_bars(drift)
        drift_html = f'<h2>Drift</h2><img src="{img_drift}" style="width:100%;max-width:1100px;border:1px solid #333;border-radius:8px;margin:6px 0;" />'

    # Event cards (latest N)
    event_cards = ""
    if events is not None and not events.empty:
        last = events.tail(EVENT_CARDS_N)
        cards = []
        for r in last.itertuples():
            ev = pd.Series({"Start": r.Start, "End": r.End, "PeakScore": getattr(r, "PeakScore", np.nan)})
            img_ev_time = plot_event_time(scored, ev)
            # Top tags proxy: biggest |z| change inside event
            use_cols = [c for c in raw_df.columns if c not in
                        {"Regime","FusedScore","H1_Forecast","H2_Recon","H3_Contrast","CorrBoost","CPD","ContextMask"}]
            clip_ev = raw_df.loc[(raw_df.index >= r.Start) & (raw_df.index <= r.End), use_cols]
            clip_pre = raw_df.loc[(raw_df.index >= r.Start - (r.End - r.Start)) & (raw_df.index <= r.Start), use_cols]
            top_tags = []
            if not clip_ev.empty and not clip_pre.empty and len(use_cols):
                zv = ((clip_ev - clip_ev.mean()) / (clip_ev.std() + 1e-9)).abs().mean().sort_values(ascending=False)
                top_tags = zv.head(5).index.tolist()
            img_ev_spec = plot_event_spectrum(scored, raw_df, ev, top_tags[:5])

            meta = f"""
            <div class="meta">
              <b>Start</b> {r.Start} &nbsp; <b>End</b> {r.End} &nbsp;
              <b>Duration</b> {(pd.to_datetime(r.End)-pd.to_datetime(r.Start))} &nbsp;
              <b>Peak</b> {getattr(r, "PeakScore", float('nan')):.2f}
            </div>"""
            top = ""
            if top_tags:
                chips = " ".join([f"<span class='chip'>{t}</span>" for t in top_tags])
                top = f"<div class='meta'><b>Top tags</b> {chips}</div>"
            card = f"""
            <div class="card">
              <h3>Event</h3>
              {meta}
              {top}
              <div><img src="{img_ev_time}" style="width:100%;border:1px solid #333;border-radius:8px;margin:6px 0;" /></div>
              {'<div><img src="'+img_ev_spec+'" style="width:100%;border:1px solid #333;border-radius:8px;margin:6px 0;" /></div>' if img_ev_spec else ''}
            </div>"""
            cards.append(card)
        event_cards = "<h2>Latest Events</h2>" + "\n".join(cards)

    # Data Quality
    dq_html = ""
    dq = compute_dq(raw_df, key_tags)

    if not dq.empty:
        trs = "\n".join(
            f"<tr><td>{r.Tag}</td>"
            f"<td style='text-align:right'>{r.Flatline:.2f}</td>"
            f"<td style='text-align:right'>{r.Dropout:.2f}</td>"
            f"<td style='text-align:right'>{r.Spikes}</td></tr>"
            for r in dq.rename(columns={"Flatline%":"Flatline","Dropout%":"Dropout"}).itertuples()
        )
        dq_html = f"""
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:1100px">
          <thead><tr><th>Tag</th><th>Flatline %</th><th>Dropout %</th><th>Spikes</th></tr></thead>
          <tbody>{trs}</tbody>
        </table>
        """

    # Model cards
    h1_mode = manifest.get("h1_mode", "lite_ar1")
    pca_info = ""
    pca_path = os.path.join(ART_DIR, "acm_pca.joblib")
    if os.path.exists(pca_path):
        try:
            from joblib import load
            pca = load(pca_path)
            if hasattr(pca, "explained_variance_ratio_"):
                evr = pca.explained_variance_ratio_.sum()*100.0
                pca_info = f" (explained variance ≈ {evr:.1f}%)"
        except Exception:
            pass
    h1_card = f"""
      <div class="card">
        <h3>H1 — Forecast (lite+AR1)</h3>
        <div>Mode: <b>{h1_mode}</b>; Roll: <b>{manifest.get('h1_roll',9)}</b>;
             RobustZ: <b>{manifest.get('h1_robust',True)}</b>;
             TopK: <b>{manifest.get('h1_topk',0)}</b>;
             MinSupport: <b>{manifest.get('h1_min_support',200)}</b></div>
      </div>"""
    h2_card = f"""
      <div class="card">
        <h3>H2 — Reconstruction (PCA)</h3>
        <div>n_components: <b>0.9 (variance)</b>{pca_info}</div>
      </div>"""
    h3_card = f"""
      <div class="card">
        <h3>H3 — Embedding Drift</h3>
        <div>Similarity: <b>cosine</b>; Baseline window: <b>50</b></div>
      </div>"""
    regimes_card = ""
    if "k_min" in manifest and "k_max" in manifest:
        regimes_card = f"""
        <div class="card">
          <h3>Regimes</h3>
          <div>Auto-k range: <b>{manifest['k_min']}–{manifest['k_max']}</b>;
               Observed: <b>{regimes_seen}</b></div>
        </div>"""
    fusion_card = f"""
      <div class="card">
        <h3>Fusion</h3>
        <div>Weights (approx): H1 0.45, H2 0.35, H3 0.35; Boosts: Corr 0.15, CPD 0.10; Mask reduces ×0.7</div>
        <div>Threshold τ: <b>{manifest.get('fused_tau', FUSED_TAU):.2f}</b></div>
      </div>"""
    config_card = f"""
      <div class="card">
        <h3>Config</h3>
        <div>Resample: <b>{manifest.get('resample_rule','1min')}</b>;
             Window/Stride: <b>{manifest.get('window',256)}/{manifest.get('stride',64)}</b>;
             FFT bins: <b>{manifest.get('max_fft_bins',64)}</b></div>
      </div>"""

    # Timings
    timings_html, _ = timings_table(runlog)

    # Header KPIs
    eq_html = f"{eq_score:.1f}" if eq_score is not None else "—"
    header_html = f"""
    <div class="kpis">
      <div class="kpi"><div class="kpi-title">Equipment Score</div><div class="kpi-value">{eq_html}</div></div>
      <div class="kpi"><div class="kpi-title">Window</div><div class="kpi-value">{t0} → {t1}</div></div>
      <div class="kpi"><div class="kpi-title">Rows</div><div class="kpi-value">{len(scored):,}</div></div>
      <div class="kpi"><div class="kpi-title">Regimes</div><div class="kpi-value">{regimes_seen}</div></div>
      <div class="kpi"><div class="kpi-title">Events ≥ τ</div><div class="kpi-value">{events_n}</div></div>
      <div class="kpi"><div class="kpi-title">Mask %</div><div class="kpi-value">{mask_cov:.2f}%</div></div>
    </div>""" if mask_cov is not None else f"""
    <div class="kpis">
      <div class="kpi"><div class="kpi-title">Equipment Score</div><div class="kpi-value">{eq_html}</div></div>
      <div class="kpi"><div class="kpi-title">Window</div><div class="kpi-value">{t0} → {t1}</div></div>
      <div class="kpi"><div class="kpi-title">Rows</div><div class="kpi-value">{len(scored):,}</div></div>
      <div class="kpi"><div class="kpi-title">Regimes</div><div class="kpi-value">{regimes_seen}</div></div>
      <div class="kpi"><div class="kpi-title">Events ≥ τ</div><div class="kpi-value">{events_n}</div></div>
    </div>"""

    # HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ background:#0b0f14; color:#e8edf2; font-family:Segoe UI,Roboto,Arial,sans-serif; margin:0; }}
main {{ max-width:1200px; margin:20px auto; padding:0 16px; }}
h1,h2,h3 {{ color:#eef2f7; }}
.card {{ background:#0f1621; padding:14px 16px; border-radius:10px; border:1px solid #1f2a37; margin:12px 0; }}
.kpis {{ display:grid; grid-template-columns: repeat(6, 1fr); gap:12px; }}
.kpi {{ background:#0f1621; border:1px solid #1f2a37; border-radius:10px; padding:12px; }}
.kpi-title {{ font-size:12px; color:#a8b3c0; }}
.kpi-value {{ font-size:20px; font-weight:600; margin-top:4px; }}
.chip {{ background:#1b2636; border:1px solid #334155; border-radius:999px; padding:2px 8px; margin:2px; display:inline-block; font-size:12px; }}
.small {{ color:#a8b3c0; font-size:12px; }}
hr.sep {{ border:0; height:1px; background:#1f2937; margin:18px 0; }}
a, a:visited {{ color:#93c5fd; text-decoration:none; }}
</style>
</head>
<body>
  <main>
    <div class="card">
      <h1>{title}</h1>
      <div class="small">Generated: {dt.datetime.now().isoformat()} &nbsp; | &nbsp; Artifacts: {ART_DIR}</div>
      {header_html}
    </div>

    <div class="card">
      <h2>Timeline Story</h2>
      <div><img src="{img_timeline}" style="width:100%;border:1px solid #334155;border-radius:8px;" /></div>
      <div class="small">Shaded = events; red band = transient mask; dashed = τ</div>
    </div>

    <div class="card">
      <h2>Tag Health (Sparklines)</h2>
      <div><img src="{img_sparks}" style="width:100%;border:1px solid #334155;border-radius:8px;" /></div>
    </div>

    {drift_html}

    {event_cards if event_cards else ''}

    <div class="card">
      <h2>Data Quality</h2>
      {dq_html if dq_html else "<div class='small'>No DQ issues computed.</div>"}
    </div>

    <div class="card">
      <h2>Model & Config</h2>
      {h1_card}
      {h2_card}
      {h3_card}
      {regimes_card}
      {fusion_card}
      {config_card}
    </div>

    <div class="card">
      {timings_html if timings_html else "<h2>Performance (Timings)</h2><div class='small'>No run log found.</div>"}
    </div>

    <div class="small">Run complete.</div>
  </main>
</body>
</html>
"""
    out = os.path.join(ART_DIR, "acm_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report: {out}")
    return out

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("ACM Artifact Report 2.0")
    p.add_argument("--scored_csv", default=os.path.join(ART_DIR,"acm_scored_window.csv"))
    p.add_argument("--drift_csv",  default=os.path.join(ART_DIR,"acm_drift.csv"))
    p.add_argument("--events_csv", default=os.path.join(ART_DIR,"acm_events.csv"))
    p.add_argument("--masks_csv",  default=os.path.join(ART_DIR,"acm_context_masks.csv"))
    a = p.parse_args()
    build_report(
        a.scored_csv if os.path.exists(a.scored_csv) else None,
        a.drift_csv if os.path.exists(a.drift_csv) else None,
        a.events_csv if os.path.exists(a.events_csv) else None,
        a.masks_csv if os.path.exists(a.masks_csv) else None,
        title=TITLE
    )

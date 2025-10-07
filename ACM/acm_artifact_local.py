# acm_artifact_local.py
# Build a simple HTML report from artifacts (no cards; tables + charts only).
# Assumes report_charts.py is in the same folder and that core/score steps
# produced scored.csv, events.json (optional), masks.csv (optional), etc.

import os, io, json, argparse, math, warnings, datetime as dt
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from report_charts import timeline, sampled_tags_with_marks, drift_bars, FUSED_TAU

warnings.filterwarnings("ignore")

# ---- Static paths (match your existing setup) ----
ROOT_DIR = r"C:\Users\bhadk\Documents\CPCL\ACM"
ART_DIR  = os.path.join(ROOT_DIR, "acm_artifacts")
os.makedirs(ART_DIR, exist_ok=True)

# ---- Default artifact filenames ----
SCORED_CSV   = os.path.join(ART_DIR, "scored.csv")
EVENTS_JSON  = os.path.join(ART_DIR, "events.json")     # optional
MASKS_CSV    = os.path.join(ART_DIR, "masks.csv")       # optional
DRIFT_CSV    = os.path.join(ART_DIR, "drift.csv")       # optional
REPORT_HTML  = os.path.join(ART_DIR, "acm_report_basic.html")

# ---- Helpers ----

def _read_csv_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    # Try parsing index as time if column Ts present
    df = pd.read_csv(path)
    if "Ts" in df.columns:
        df["Ts"] = pd.to_datetime(df["Ts"], errors="coerce", utc=False)
        df = df.set_index("Ts", drop=True)
        df = df.sort_index()
    else:
        # try to coerce index if it looks like time
        try:
            df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
            df = df.sort_index()
        except Exception:
            pass
    return df

def _read_events(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Start","End","PeakScore","Kind"])
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept either list[dict] or dict with "events"
        if isinstance(data, dict) and "events" in data:
            data = data["events"]
        ev = pd.DataFrame(data)
        # Normalize time columns
        for c in ["Start","End","Ts","PeakTs"]:
            if c in ev.columns:
                ev[c] = pd.to_datetime(ev[c], errors="coerce", utc=False)
        # Ensure Start/End exist if only Ts given
        if "Start" not in ev.columns and "Ts" in ev.columns:
            ev["Start"] = ev["Ts"]
        if "End" not in ev.columns and "Start" in ev.columns:
            ev["End"] = ev["Start"]
        if "PeakScore" not in ev.columns:
            ev["PeakScore"] = np.nan
        if "Kind" not in ev.columns:
            ev["Kind"] = ""
        # Keep only necessary cols
        keep = ["Start","End","PeakScore","Kind"]
        ev = ev[[c for c in keep if c in ev.columns]].dropna(subset=["Start"])
        return ev
    except Exception:
        # If JSON malformed, return empty
        return pd.DataFrame(columns=["Start","End","PeakScore","Kind"])

def _read_optional_csv(path: str, expect_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=expect_cols or [])
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame(columns=expect_cols or [])

def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "Ts" in df.columns:
        df["Ts"] = pd.to_datetime(df["Ts"], errors="coerce", utc=False)
        df = df.set_index("Ts", drop=True)
        return df.sort_index()
    # Try best effort
    try:
        df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
        return df.sort_index()
    except Exception:
        return df

def _infer_tag_columns(raw_df: pd.DataFrame, scored: pd.DataFrame) -> List[str]:
    score_cols = {"FusedScore","H1_Forecast","H2_Recon","H3_Contrast","Regime"}
    # anything numeric in raw_df not in score cols
    tags = []
    for c in raw_df.columns:
        if c in score_cols:
            continue
        if pd.api.types.is_numeric_dtype(raw_df[c]):
            tags.append(c)
    # If nothing found, try from scored common cols
    if not tags:
        for c in scored.columns:
            if c in score_cols: 
                continue
            if c in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[c]):
                tags.append(c)
    return tags

def _fmt_int(n: Optional[int]) -> str:
    try:
        return f"{int(n):,d}"
    except Exception:
        return "-"

def _count_events(ev: pd.DataFrame) -> int:
    if ev is None or ev.empty: 
        return 0
    return len(ev.dropna(subset=["Start"]))

def _count_anom_points(scored: pd.DataFrame) -> int:
    if "FusedScore" not in scored.columns:
        return 0
    try:
        return int((scored["FusedScore"] >= FUSED_TAU).sum())
    except Exception:
        return 0

# ---- Report builder ----

def build_report(equip: str,
                 test_csv: Optional[str] = None,
                 scored_csv: Optional[str] = None,
                 events_json: Optional[str] = None,
                 masks_csv: Optional[str] = None,
                 drift_csv: Optional[str] = None,
                 out_html: Optional[str] = None) -> str:
    """
    Build a minimal HTML report with timeline + sampled tag trends.
    - equip: display name
    - test_csv: optional path to the test data used for scoring (for raw tag trends)
    - scored_csv: defaults to ART_DIR/scored.csv
    - events_json/masks_csv/drift_csv: optional artifacts
    - out_html: defaults to ART_DIR/acm_report_basic.html
    """
    scored_path = scored_csv or SCORED_CSV
    events_path = events_json or EVENTS_JSON
    masks_path  = masks_csv  or MASKS_CSV
    drift_path  = drift_csv  or DRIFT_CSV
    out_path    = out_html   or REPORT_HTML

    # Load artifacts
    if not os.path.exists(scored_path):
        raise FileNotFoundError(f"Missing scored.csv at: {scored_path}")
    scored = _read_csv_any(scored_path)
    scored = _ensure_time_index(scored)

    # Prefer test CSV for raw trends; else try to reconstruct from scored if raw cols are present
    if test_csv and os.path.exists(test_csv):
        raw_df = _read_csv_any(test_csv)
    else:
        # fallback: if scored already contains raw numeric tag columns, use it
        raw_df = scored.copy()

    # Optional inputs
    events_df = _read_events(events_path)
    masks_df = _read_optional_csv(masks_path, expect_cols=["Ts","Mask"])
    drift_df = _read_optional_csv(drift_path)

    # Ensure indices/types
    raw_df   = _ensure_time_index(raw_df)
    scored   = _ensure_time_index(scored)
    if not events_df.empty:
        for c in ["Start","End"]:
            if c in events_df.columns:
                events_df[c] = pd.to_datetime(events_df[c], errors="coerce", utc=False)
    if not masks_df.empty and "Ts" in masks_df.columns:
        masks_df["Ts"] = pd.to_datetime(masks_df["Ts"], errors="coerce", utc=False)

    # Tag selection
    tag_cols = _infer_tag_columns(raw_df, scored)

    # Charts
    img_timeline = timeline(scored, events_df, masks_df)
    img_sampled  = sampled_tags_with_marks(
        df=raw_df,
        tags=tag_cols,
        events=events_df if not events_df.empty else None,
        fused=scored["FusedScore"] if "FusedScore" in scored.columns else None
    )

    # Optional drift bars if present
    img_drift = ""
    if not drift_df.empty and {"Tag","DriftZ"}.issubset(drift_df.columns):
        img_drift = drift_bars(drift_df, top=20)

    # Summaries
    t0 = scored.index.min()
    t1 = scored.index.max()
    n_rows = len(scored)
    n_events = _count_events(events_df)
    n_anom_pts = _count_anom_points(scored)
    n_tags = len(tag_cols)

    # Minimal CSS
    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;margin:16px;color:#111}
    h1,h2{margin:6px 0 8px 0}
    h1{font-size:22px}
    h2{font-size:18px;border-bottom:1px solid #ddd;padding-bottom:4px}
    table{border-collapse:collapse;width:100%;margin:8px 0}
    th,td{border:1px solid #ccc;padding:6px 8px;font-size:13px;text-align:left}
    small{color:#555}
    .meta{margin:6px 0 12px 0}
    img{display:block}
    """

    # Header
    title = f"ACM Report — {equip}"
    gen_ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Summary table
    sum_rows = [
        ("Equipment", equip),
        ("Time Range", f"{t0} → {t1}"),
        ("Rows (scored)", _fmt_int(n_rows)),
        ("Tags (plotted)", _fmt_int(n_tags)),
        (f"Anomaly threshold τ", f"{FUSED_TAU:.2f}"),
        ("Events (episodes)", _fmt_int(n_events)),
        ("Anomalous points", _fmt_int(n_anom_pts)),
    ]
    sum_tbl = ["<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"]
    for k, v in sum_rows:
        # avoid f-string backslash issues by formatting outside the braces where needed
        row = "<tr><td>{}</td><td>{}</td></tr>".format(k, v)
        sum_tbl.append(row)
    sum_tbl.append("</tbody></table>")
    sum_html = "\n".join(sum_tbl)

    # Sections
    sec_timeline = (
        "<section id='timeline'>"
        "<h2>Timeline</h2>"
        "<p><small>FusedScore with τ = {:.2f} (dashed). Shaded = events; thin red band = masked context. Heads and regime ribbon shown below.</small></p>"
        "<img src='{}' style='width:100%;border:1px solid #333;margin:8px 0;' />"
        "</section>".format(FUSED_TAU, img_timeline)
    )

    sec_sampled = ""
    if img_sampled:
        sec_sampled = (
            "<section id='sampled'>"
            "<h2>Sampled Tag Trends</h2>"
            "<p><small>z-normalized; orange dots mark points where FusedScore ≥ τ = {:.2f}. Event spans lightly shaded.</small></p>"
            "<img src='{}' style='width:100%;border:1px solid #333;margin:8px 0;' />"
            "</section>".format(FUSED_TAU, img_sampled)
        )

    sec_drift = ""
    if img_drift:
        sec_drift = (
            "<section id='drift'>"
            "<h2>Top Drifted Tags</h2>"
            "<img src='{}' style='width:100%;border:1px solid #333;margin:8px 0;' />"
            "</section>".format(img_drift)
        )

    # Final HTML
    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>{}</title><style>{}</style></head><body>".format(title, css),
        "<h1>{}</h1>".format(title),
        "<div class='meta'><small>Generated: {}</small></div>".format(gen_ts),
        sum_html,
        sec_timeline,
        sec_sampled,
        sec_drift,
        "</body></html>"
    ]
    html = "\n".join(html_parts)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[BUILD_REPORT] Wrote {out_path}")
    return out_path

# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="Build basic ACM HTML report (tables + charts).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build the report")
    p_build.add_argument("--equip", required=True, help="Equipment name to display (e.g., 'FD FAN')")
    p_build.add_argument("--test", default="", help="Path to test CSV (used for raw tag trends)")
    p_build.add_argument("--scored", default="", help="Path to scored.csv (defaults to artifacts)")
    p_build.add_argument("--events", default="", help="Path to events.json (optional)")
    p_build.add_argument("--masks", default="", help="Path to masks.csv (optional)")
    p_build.add_argument("--drift", default="", help="Path to drift.csv (optional)")
    p_build.add_argument("--out", default="", help="Path to output HTML (defaults to artifacts)")

    args = parser.parse_args()

    if args.cmd == "build":
        test_csv = args.test if args.test else None
        scored_csv = args.scored if args.scored else None
        events_json = args.events if args.events else None
        masks_csv = args.masks if args.masks else None
        drift_csv = args.drift if args.drift else None
        out_html = args.out if args.out else None

        build_report(
            equip=args.equip,
            test_csv=test_csv,
            scored_csv=scored_csv,
            events_json=events_json,
            masks_csv=masks_csv,
            drift_csv=drift_csv,
            out_html=out_html
        )

if __name__ == "__main__":
    main()

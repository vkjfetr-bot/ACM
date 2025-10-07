#!/usr/bin/env python3
"""
ACM (Asset Condition Monitoring)
- Pulls historian window from SQL Server
- Builds robust features per signal (rolling z, MAD-z, EWMA)
- IsolationForest + simple threshold rules → anomaly events
- Writes events & window summary to SQL
- Additionally exports CSVs, PNGs (optional), and an HTML report

Env vars (examples):
  MSSQL_CNX="DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=plantdb;Trusted_Connection=yes;"
  EQUIPMENT_ID="EAF-1"
  START_ISO="2025-10-05T00:00:00"
  END_ISO="2025-10-06T00:00:00"
  SP_FETCH="dbo.usp_GetEquipmentWindow"
  TABLE_EVENTS="dbo.acm_equipment_event"
  TABLE_SUM="dbo.acm_window_summary"
  PRIMARY_TAGS="Power_kW,Temp_C,Pressure_bar"
  REPORT_DIR="./acm_reports"
  WRITE_CSV="1"
  WRITE_HTML="1"
"""
import os, sys, math, json, datetime as dt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pyodbc

# ---- Optional charts (script still works without matplotlib) ----
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ---- CONFIG ----
CNX_STR = os.getenv("MSSQL_CNX", "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=plantdb;Trusted_Connection=yes;")
EQUIPMENT_ID = os.getenv("EQUIPMENT_ID", "EQUIP-1")
START_ISO = os.getenv("START_ISO")
END_ISO   = os.getenv("END_ISO")
SP_FETCH  = os.getenv("SP_FETCH", "dbo.usp_GetEquipmentWindow")   # (@EquipmentId, @StartTime, @EndTime)
TABLE_EVENTS = os.getenv("TABLE_EVENTS", "dbo.acm_equipment_event")
TABLE_SUM    = os.getenv("TABLE_SUM",    "dbo.acm_window_summary")
PRIMARY_TAGS = [s.strip() for s in os.getenv("PRIMARY_TAGS", "Power_kW,Temp_C,Pressure_bar").split(",") if s.strip()]

# ---- REPORT/EXPORT SETTINGS ----
REPORT_DIR = os.getenv("REPORT_DIR", "./acm_reports")
WRITE_CSV  = os.getenv("WRITE_CSV", "1") == "1"
WRITE_HTML = os.getenv("WRITE_HTML", "1") == "1"

# ---- Guard: time window required ----
if not START_ISO or not END_ISO:
    print("ERROR: set START_ISO and END_ISO (ISO8601).", file=sys.stderr)
    sys.exit(2)

start_ts = dt.datetime.fromisoformat(START_ISO)
end_ts   = dt.datetime.fromisoformat(END_ISO)

os.makedirs(REPORT_DIR, exist_ok=True)
_prefix = f"{EQUIPMENT_ID}_{start_ts.strftime('%Y%m%dT%H%M%S')}_{end_ts.strftime('%Y%m%dT%H%M%S')}"

# ---- Helpers ----
def sql_read_df(cnx, equipment_id, start_ts, end_ts):
    q = f"EXEC {SP_FETCH} @EquipmentId=?, @StartTime=?, @EndTime=?"
    df = pd.read_sql(q, cnx, params=[equipment_id, start_ts, end_ts])
    if "Ts" not in df.columns:
        raise RuntimeError("Expected 'Ts' column (datetime) in historian result.")
    df["Ts"] = pd.to_datetime(df["Ts"])
    return df

def zscore(s, win=60):
    r = s.rolling(win, min_periods=max(3, win//5))
    mu = r.mean()
    sd = r.std(ddof=0).replace(0, np.nan)
    return (s - mu) / sd

def robust_mad_score(s, win=60):
    r = s.rolling(win, min_periods=max(3, win//5))
    med = r.median()
    mad = r.apply(lambda x: np.median(np.abs(x - np.median(x))) if len(x) else np.nan, raw=True)
    mad = mad.replace(0, np.nan)
    return 0.6745 * (s - med) / mad

def ewma(s, alpha=0.1):
    return s.ewm(alpha=alpha, adjust=False).mean()

def fit_isoforest(df, cols, contamination=0.02, seed=42):
    X = df[cols].dropna()
    if len(X) < 100:
        return None, pd.Series(index=df.index, data=np.nan, dtype="float64")
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    iso.fit(X)
    score = -iso.score_samples(df[cols].fillna(method="ffill").fillna(method="bfill"))
    return iso, pd.Series(score, index=df.index)

def upsert_events(cnx, events_df):
    if events_df.empty: return
    sql = f"""
    INSERT INTO {TABLE_EVENTS}
      (EquipmentId, StartTime, EndTime, Signal, Score, Rule, Severity, MetaJson)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cur = cnx.cursor()
    for _, r in events_df.iterrows():
        cur.execute(sql,
            r["EquipmentId"], r["StartTime"], r["EndTime"], r["Signal"],
            float(r["Score"]), r["Rule"], r["Severity"],
            json.dumps(r["MetaJson"], ensure_ascii=False))
    cur.commit()

def upsert_summary(cnx, summary_row: dict):
    sql = f"""
    INSERT INTO {TABLE_SUM}
      (EquipmentId, WindowStart, WindowEnd, NRows, AnomalyFrac, TopSignalsJson)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    cur = cnx.cursor()
    cur.execute(sql,
        summary_row["EquipmentId"], summary_row["WindowStart"], summary_row["WindowEnd"],
        int(summary_row["NRows"]), float(summary_row["AnomalyFrac"]),
        json.dumps(summary_row["TopSignalsJson"], ensure_ascii=False))
    cur.commit()

def _spans(hit_series: pd.Series):
    hit = hit_series.fillna(False).astype(bool)
    if not hit.any(): return []
    idx = hit.index
    spans = []
    in_span = False
    s0 = None
    prev_t = idx[0]
    for t, v in hit.items():
        if v and not in_span:
            in_span = True; s0 = t
        elif not v and in_span:
            spans.append((s0, prev_t)); in_span = False
        prev_t = t
    if in_span:
        spans.append((s0, idx[-1]))
    return spans

# ---- Main ----
def main():
    # Read window
    with pyodbc.connect(CNX_STR) as cnx:
        df = sql_read_df(cnx, EQUIPMENT_ID, start_ts, end_ts)

    if df.empty:
        print("No data in window."); return

    df = df.sort_values("Ts").set_index("Ts")

    feats = []
    rule_hits = []

    # Build features for available primary tags
    available_tags = [c for c in PRIMARY_TAGS if c in df.columns]
    for tag in available_tags:
        s = pd.to_numeric(df[tag], errors="coerce")
        f = pd.DataFrame(index=df.index)
        f[f"{tag}_z60"]   = zscore(s, 60)
        f[f"{tag}_mad60"] = robust_mad_score(s, 60)
        f[f"{tag}_ewm"]   = ewma(s)
        feats.append(f)

        # Simple z-threshold rule
        thr_z = 3.0
        hit = f[f"{tag}_z60"].abs() >= thr_z
        if hit.any():
            for (t0, t1) in _spans(hit):
                rule_hits.append(dict(
                    EquipmentId=EQUIPMENT_ID,
                    StartTime=t0.to_pydatetime(),
                    EndTime=t1.to_pydatetime(),
                    Signal=tag,
                    Score=float(f.loc[t0:t1, f"{tag}_z60"].abs().max()),
                    Rule=f"|z60|>={thr_z}",
                    Severity="HIGH",
                    MetaJson={"reason": "zscore60", "n_points": int(hit.loc[t0:t1].sum())}
                ))

    if feats:
        F = pd.concat(feats, axis=1)
    else:
        # No primary tags present → create empty frame with proper index
        F = pd.DataFrame(index=df.index)

    # IsolationForest
    iso_cols = [c for c in F.columns if c.endswith("_z60") or c.endswith("_mad60")]
    _, iso_score = fit_isoforest(F, iso_cols, contamination=0.02)
    F["iso_score"] = iso_score

    # Aggregate IF anomalies (top 2% quantile)
    q = F["iso_score"].quantile(0.98) if F["iso_score"].notna().any() else np.nan
    if not (isinstance(q, float) and math.isnan(q)):
        iso_hit = F["iso_score"] >= q
        for (t0, t1) in _spans(iso_hit):
            rule_hits.append(dict(
                EquipmentId=EQUIPMENT_ID,
                StartTime=t0.to_pydatetime(),
                EndTime=t1.to_pydatetime(),
                Signal="MULTI",
                Score=float(F.loc[t0:t1, "iso_score"].max()),
                Rule="IsolationForest top 2%",
                Severity="MED",
                MetaJson={"top_feats": iso_cols[:10]}
            ))

    events_df = pd.DataFrame(rule_hits)
    anomaly_frac = float((F["iso_score"] >= q).mean()) if not (isinstance(q, float) and math.isnan(q)) else 0.0

    # ---- Write to SQL ----
    with pyodbc.connect(CNX_STR) as cnx:
        if not events_df.empty:
            upsert_events(cnx, events_df)
        upsert_summary(cnx, dict(
            EquipmentId=EQUIPMENT_ID,
            WindowStart=start_ts, WindowEnd=end_ts,
            NRows=int(len(df)),
            AnomalyFrac=anomaly_frac,
            TopSignalsJson={"features": iso_cols[:10]}
        ))

    # ---- Exports: CSV + JSON ----
    events_csv = None
    scores_csv = None
    summary_json = os.path.join(REPORT_DIR, f"{_prefix}_summary.json")

    if WRITE_CSV:
        if not events_df.empty:
            events_csv = os.path.join(REPORT_DIR, f"{_prefix}_events.csv")
            events_df.to_csv(events_csv, index=False)
        scores_csv = os.path.join(REPORT_DIR, f"{_prefix}_scores.csv")
        F_reset = F.reset_index()
        # Ensure 'Ts' column exists after reset_index
        if "Ts" not in F_reset.columns:
            F_reset = F_reset.rename(columns={F_reset.columns[0]: "Ts"})
        F_reset[["Ts", "iso_score"]].to_csv(scores_csv, index=False)

        with open(summary_json, "w", encoding="utf-8") as fsum:
            json.dump({
                "EquipmentId": EQUIPMENT_ID,
                "WindowStart": START_ISO, "WindowEnd": END_ISO,
                "NRows": int(len(df)),
                "AnomalyFrac": anomaly_frac,
                "TopFeatures": iso_cols[:10],
                "EventsCSV": os.path.basename(events_csv) if events_csv else None,
                "ScoresCSV": os.path.basename(scores_csv) if scores_csv else None
            }, fsum, ensure_ascii=False)

    # ---- Charts (PNG) ----
    iso_png = None
    bar_png = None
    if _HAS_MPL:
        try:
            fig1 = plt.figure(figsize=(9, 3.2))
            F["iso_score"].plot()
            plt.title(f"Isolation Score over Time — {EQUIPMENT_ID}")
            plt.xlabel("Ts"); plt.ylabel("iso_score")
            iso_png = os.path.join(REPORT_DIR, f"{_prefix}_iso_score.png")
            plt.tight_layout(); fig1.savefig(iso_png, dpi=120); plt.close(fig1)
        except Exception:
            iso_png = None
        try:
            if not events_df.empty:
                cnt = events_df.groupby("Signal").size().sort_values(ascending=False)
                fig2 = plt.figure(figsize=(8, 3.2))
                cnt.plot(kind="bar")
                plt.title("Events by Signal"); plt.xlabel("Signal"); plt.ylabel("Count")
                bar_png = os.path.join(REPORT_DIR, f"{_prefix}_events_by_signal.png")
                plt.tight_layout(); fig2.savefig(bar_png, dpi=120); plt.close(fig2)
        except Exception:
            bar_png = None

    # ---- HTML Report ----
    if WRITE_HTML:
        html_path = os.path.join(REPORT_DIR, f"{_prefix}_report.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ACM Report — {EQUIPMENT_ID}</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;margin:16px;}}
 .kpi{{display:flex;gap:16px;margin:8px 0 16px 0;flex-wrap:wrap;}}
 .card{{border:1px solid #ddd;border-radius:8px;padding:12px;min-width:160px}}
 .muted{{color:#666;font-size:12px}}
 img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:6px}}
 a{{text-decoration:none;color:#0b5fff}}
 table{{border-collapse:collapse;width:100%;}}
 th,td{{border:1px solid #eee;padding:6px 8px;text-align:left}}
 th{{background:#fafafa}}
</style></head><body>
  <h2>ACM Report — {EQUIPMENT_ID}</h2>
  <div class="muted">Window: {START_ISO} → {END_ISO}</div>
  <div class="kpi">
    <div class="card"><div>Total rows</div><h3>{len(df)}</h3></div>
    <div class="card"><div>Events</div><h3>{0 if events_df.empty else len(events_df)}</h3></div>
    <div class="card"><div>Anomaly fraction</div><h3>{anomaly_frac:.3f}</h3></div>
  </div>
  <div class="card">
    <h3>Downloads</h3>
    <ul>
      {"<li>Scores CSV: <a href=\"" + os.path.basename(scores_csv) + "\">" + os.path.basename(scores_csv) + "</a></li>" if scores_csv else ""}
      {("<li>Events CSV: <a href=\"" + os.path.basename(events_csv) + "\">" + os.path.basename(events_csv) + "</a></li>") if events_csv else ""}
      <li>Summary JSON: <a href="{os.path.basename(summary_json)}">{os.path.basename(summary_json)}</a></li>
    </ul>
  </div>
  <div class="card">
    <h3>Isolation score timeline</h3>
    {"<img src=\"" + os.path.basename(iso_png) + "\" />" if iso_png else "<div class='muted'>Chart unavailable.</div>"}
  </div>
  <div class="card">
    <h3>Events by signal</h3>
    {"<img src=\"" + os.path.basename(bar_png) + "\" />" if bar_png else "<div class='muted'>No events or chart unavailable.</div>"}
  </div>
  <div class="card">
    <h3>Top features</h3>
    <table><tr><th>#</th><th>Feature</th></tr>
      {"".join(f"<tr><td>{i+1}</td><td>{f}</td></tr>" for i,f in enumerate(iso_cols[:10]))}
    </table>
  </div>
</body></html>""")
        print(f"Report: {html_path}")

    print(f"ACM done: {len(events_df)} events, anomaly_frac={anomaly_frac:.3f}")

if __name__ == "__main__":
    main()

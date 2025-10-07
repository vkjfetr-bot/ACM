# acm_score_local.py
# Computes TagScore and EquipmentScore using fused score burden, regime stability,
# and drift penalties. Reads the new events file to weight recent issues.

import os, json
import pandas as pd
import numpy as np

ART_DIR = r"C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def compute_scores(scored_csv=None, drift_csv=None, events_csv=None, weights=None):
    if weights is None:
        weights = {"anom": 0.55, "regime": 0.20, "drift": 0.25}

    if scored_csv is None:
        scored_csv = os.path.join(ART_DIR, "acm_scored_window.csv")
    df = pd.read_csv(scored_csv, index_col=0, parse_dates=True)

    # anomaly burden via fused score
    anom_burden = (df["FusedScore"]).mean()  # 0..1

    # regime stability penalty
    reg_pen = 0.0
    if "Regime" in df.columns and len(df) > 1:
        switches = int((df["Regime"].diff().fillna(0) != 0).sum())
        reg_pen = min(1.0, switches / max(1, len(df)//10))

    # drift
    drift_pen = 0.0
    drift = None
    if drift_csv is None:
        dpath = os.path.join(ART_DIR, "acm_drift.csv")
        drift_csv = dpath if os.path.exists(dpath) else None
    if drift_csv and os.path.exists(drift_csv):
        drift = pd.read_csv(drift_csv)
        drift_pen = float(np.tanh((drift["DriftZ"].fillna(0).median())/3.0))

    # events (recent severity)
    if events_csv is None:
        epath = os.path.join(ART_DIR, "acm_events.csv")
        events_csv = epath if os.path.exists(epath) else None
    event_boost = 0.0
    if events_csv and os.path.exists(events_csv):
        ev = pd.read_csv(events_csv, parse_dates=["Start","End"])
        if not ev.empty:
            # weighted by duration and peak; normalized
            dur = (ev["End"] - ev["Start"]).dt.total_seconds().clip(lower=0)
            score = (dur.fillna(0)/60.0) * ev["PeakScore"].fillna(0)
            norm = score.quantile(0.9) or 1.0
            event_boost = float(np.clip(score.sum()/ (norm*10), 0, 0.3))  # cap 0.3

    # overall penalties
    penalty = (weights["anom"]*anom_burden) + (weights["regime"]*reg_pen) + (weights["drift"]*drift_pen)
    penalty = float(np.clip(penalty + event_boost, 0, 1))
    equipment_score = 100.0 * (1.0 - penalty)

    # Tag scores (uniform since fused is aggregated; extend later with per-tag heads if needed)
    tag_cols = [c for c in df.columns if not any(x in c for x in ["_ma_","_std_","_slope_","Regime","FusedScore","H1_","H2_","H3_","CorrBoost","CPD","ContextMask"])]
    # If no clean tag list available, derive from manifest if present
    manifest_path = os.path.join(ART_DIR, "acm_manifest.json")
    tags = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as f: tags = json.load(f).get("tags")
    if not tags:
        tags = tag_cols

    rows = []
    for t in tags:
        # proxy per-tag: use fused burden + drift (if available)
        tb = float(anom_burden)
        dp = 0.0
        if drift is not None and t in set(drift["Tag"]):
            z = float(drift.loc[drift["Tag"]==t, "DriftZ"].values[0])
            dp = float(np.tanh(z/3.0))
        tp = np.clip((weights["anom"]*tb) + (weights["regime"]*reg_pen) + (weights["drift"]*dp) + event_boost, 0, 1)
        rows.append({"Tag": t, "TagScore": 100.0 * (1.0 - tp)})

    tag_df = pd.DataFrame(rows).sort_values("TagScore", ascending=True)
    tag_df.to_csv(os.path.join(ART_DIR, "acm_tag_scores.csv"), index=False)
    pd.DataFrame([{"EquipmentScore": equipment_score}]).to_csv(os.path.join(ART_DIR, "acm_equipment_score.csv"), index=False)
    return {"equipment_score": equipment_score, "tags": len(tag_df)}

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser("ACM Score (final)")
    p.add_argument("--scored_csv", default=os.path.join(ART_DIR,"acm_scored_window.csv"))
    p.add_argument("--drift_csv",  default=os.path.join(ART_DIR,"acm_drift.csv"))
    p.add_argument("--events_csv", default=os.path.join(ART_DIR,"acm_events.csv"))
    a = p.parse_args()
    out = compute_scores(a.scored_csv,
                         a.drift_csv if os.path.exists(a.drift_csv) else None,
                         a.events_csv if os.path.exists(a.events_csv) else None)
    print(json.dumps(out, indent=2))

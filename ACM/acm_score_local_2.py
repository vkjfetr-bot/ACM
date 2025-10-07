# acm_score_local.py
# Computes tag-level and equipment-level scores from diagnostics; writes CSVs.

import os, json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = r"C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts"
os.makedirs(ART_DIR, exist_ok=True)
print(f"[ACM] Using static ART_DIR: {ART_DIR}")


def _safe_pct(x): 
    return 100.0 * (float(x) if x is not None else 0.0)

def compute_scores(scored_csv: str,
                   drift_csv: str = None,
                   weights = None) -> pd.DataFrame:
    """
    Inputs:
      - scored_csv: from acm_core_local.score_window() → must include Anomaly, Regime
      - drift_csv: optional per-tag DriftZ table
    Returns:
      - DataFrame with TagScore (0-100), and an EquipmentScore row.
    """
    if weights is None:
        weights = {"anom": 0.6, "regime": 0.2, "drift": 0.2}

    df = pd.read_csv(scored_csv, index_col=0, parse_dates=True)
    tag_cols = [c for c in df.columns if not any(x in c for x in ["_ma_","_std_","_slope_","Anomaly","Regime"])]

    # Anomaly burden per tag via proxy (use window anomaly rate overall & per-tag volatility)
    anom_rate = df["Anomaly"].mean()  # 0..1 overall
    # Regime stability: more switches → worse
    regime_switches = (df["Regime"].diff().fillna(0) != 0).sum()
    regime_penalty = min(1.0, regime_switches / max(1, len(df)//10))  # heuristic 0..1

    drift = None
    if drift_csv and os.path.exists(drift_csv):
        drift = pd.read_csv(drift_csv)

    rows = []
    for t in tag_cols:
        # Volatility proxy -> higher std means lower score
        std_norm = float(df[t].std() / (abs(df[t].mean()) + 1e-6))
        std_penalty = np.tanh(std_norm)  # 0..1

        drift_penalty = 0.0
        if drift is not None and t in set(drift["Tag"]):
            z = float(drift.loc[drift["Tag"]==t, "DriftZ"].values[0])
            drift_penalty = np.tanh(z/3.0)  # soften high z

        # Combine
        penalty = (weights["anom"] * anom_rate) + (weights["regime"] * regime_penalty) + (weights["drift"] * drift_penalty)
        penalty = max(0.0, min(1.0, penalty))
        tag_score = 100.0 * (1.0 - 0.5*std_penalty) * (1.0 - penalty)  # 0..100

        rows.append({"Tag": t,
                     "StdPenalty": float(std_penalty),
                     "AnomRate": float(anom_rate),
                     "RegimePenalty": float(regime_penalty),
                     "DriftPenalty": float(drift_penalty),
                     "TagScore": float(tag_score)})

    res = pd.DataFrame(rows).sort_values("TagScore", ascending=True)
    eq_score = float(res["TagScore"].mean()) if not res.empty else 100.0
    res.to_csv(os.path.join(ART_DIR, "acm_tag_scores.csv"), index=False)
    pd.DataFrame([{"EquipmentScore": eq_score}]).to_csv(os.path.join(ART_DIR, "acm_equipment_score.csv"), index=False)
    return res

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("ACM Score (local)")
    p.add_argument("--scored_csv", default=os.path.join(ART_DIR, "acm_scored_window.csv"))
    p.add_argument("--drift_csv",  default=os.path.join(ART_DIR, "acm_drift.csv"))
    args = p.parse_args()
    df = compute_scores(args.scored_csv, args.drift_csv if os.path.exists(args.drift_csv) else None)
    print(df.head(20).to_string(index=False))
    print(f"Wrote: {os.path.join(ART_DIR, 'acm_tag_scores.csv')} and acm_equipment_score.csv")

# acm_core_local.py
# Changelog
# 2025-10-07: Added auto tag detection (detect_tags) for CSV-only flows; no CLI tags required.
# 2025-10-07: Local core: contextual baselines, regimes, IF ensemble, drift, spans; CSV "table" helpers (no SQL).

import os, json, math, joblib, warnings
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ART_DIR = r"C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts"
os.makedirs(ART_DIR, exist_ok=True)
print(f"[ACM] Using static ART_DIR: {ART_DIR}")


# ------------------------- I/O & helpers -------------------------
def detect_tags(df: pd.DataFrame,
                min_unique: int = 20,
                exclude_cols=("Ts","ts","timestamp","Timestamp")) -> List[str]:
    """Auto-pick numeric signal columns for ACM."""
    cands = []
    for c in df.columns:
        if c in exclude_cols: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) >= min_unique:
            cands.append(c)
    return cands

def ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a DateTimeIndex from a timestamp column while tolerating mixed/invalid formats.
    - Accepts: TS/Ts/timestamp/time/datetime (any case).
    - Uses pandas 'mixed' parser + dayfirst fallback.
    - Also supports Excel serials (numbers).
    - Drops only the bad rows (warns), never raises.
    """
    import pandas as pd
    import numpy as np

    # 1) find a timestamp column (case-insensitive)
    cand_keys = {"ts", "timestamp", "time", "datetime"}
    ts_col = None
    for c in df.columns:
        if str(c).strip().lower() in cand_keys:
            ts_col = c
            break
    if ts_col is None:
        # last attempt: common exact names
        for c in ["Ts", "ts", "TS", "Timestamp", "timestamp", "Time", "time", "Datetime", "datetime"]:
            if c in df.columns:
                ts_col = c
                break
    if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if ts_col is None:
        print("[ACM][WARN] No timestamp column found. Expected TS/Ts/Timestamp/Time.")
        # do NOT crash—return as-is; later steps may handle or you can bail gracefully
        return df

    s = df[ts_col]

    # 2) try robust parsing
    # pass 1: mixed
    dt1 = pd.to_datetime(s, errors="coerce", format="mixed", utc=False)
    # pass 2: mixed + dayfirst=True (helps with DD-MM-YYYY)
    dt2 = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True, utc=False)

    # 3) Excel serials (numbers)
    num = pd.to_numeric(s, errors="coerce")
    excel_dt = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")

    # 4) combine best effort
    ts_parsed = dt1.copy()
    ts_parsed = ts_parsed.fillna(dt2)
    ts_parsed = ts_parsed.fillna(excel_dt.where(num.notna(), pd.NaT))

    # 5) drop bad rows, warn only
    bad = ts_parsed.isna().sum()
    if bad > 0:
        print(f"[ACM][WARN] Skipping {bad} row(s) with invalid timestamp format.")

    good_mask = ts_parsed.notna()
    if good_mask.any():
        out = df.loc[good_mask].copy()
        out.index = ts_parsed.loc[good_mask]
        return out.sort_index()
    else:
        print("[ACM][WARN] All timestamps invalid; continuing with original index (no resampling possible).")
        return df  # keep going without a time index


def resample_fill(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    # If we don't have a time index or nothing to do, just return
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return df
    # Coerce everything to numeric where possible; non-numeric -> NaN
    num = df.apply(pd.to_numeric, errors="coerce")
    # Resample only numeric content
    g = num.resample(rule).mean()
    # Fill small gaps
    return g.interpolate(limit_direction="both")



def rolling_features(df: pd.DataFrame, cols: List[str],
                     w_short=5, w_mid=15, w_long=60) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = df[c]
        out[f"{c}_ma_s"] = s.rolling(w_short, min_periods=1).mean()
        out[f"{c}_ma_m"] = s.rolling(w_mid,   min_periods=1).mean()
        out[f"{c}_ma_l"] = s.rolling(w_long,  min_periods=1).mean()
        out[f"{c}_std_m"] = s.rolling(w_mid,  min_periods=1).std()
        out[f"{c}_slope_m"] = s.diff(w_mid) / max(w_mid,1)
    return out

# ------------------------- Models & states -------------------------
@dataclass
class CoreConfig:
    resample_rule: str = "1min"
    k_regimes: int = 3
    iforest_estimators: int = 200
    iforest_contamination: float = 0.02
    min_unique: int = 20

def fit_regimes(X: np.ndarray, k: int=3, random_state: int=17) -> KMeans:
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    km.fit(X)
    return km

def fit_iforest(X: np.ndarray, n_estimators=200, contamination=0.02, random_state=17) -> IsolationForest:
    ifo = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          random_state=random_state, n_jobs=-1)
    ifo.fit(X)
    return ifo

def simple_drift_score(series: pd.Series, baseline_mu: float, baseline_sigma: float) -> float:
    """Population shift z-score using current mean vs baseline mean."""
    if baseline_sigma <= 1e-8: 
        return 0.0
    z = abs(series.mean() - baseline_mu) / baseline_sigma
    return float(z)

# ------------------------- Core pipeline -------------------------
def train_core(csv_path: str,
               config: Optional[CoreConfig]=None,
               tags: Optional[List[str]]=None,
               save_prefix: str="acm") -> Dict:
    """
    Train baseline scaler, regimes (KMeans), anomaly (IsolationForest), and per-tag baselines.
    Saves artifacts into ./acm_artifacts and returns a small manifest.
    """
    if config is None:
        config = CoreConfig()

    df = pd.read_csv(csv_path)
    df = ensure_time_index(df)
    df = resample_fill(df, config.resample_rule)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    
    if tags is None:
        tags = detect_tags(df, min_unique=config.min_unique)

    if not tags:
        raise ValueError("No usable numeric tags found. Provide tags or check CSV.")

    # Feature engineering
    fdf = rolling_features(df[tags], tags)
    fdf = fdf.dropna()

    # Feature matrix (keep robust set)
    feat_cols = [c for c in fdf.columns if any(s in c for s in ["_ma_", "_std_", "_slope_"])]
    X = fdf[feat_cols].values

    scaler = StandardScaler().fit(X)
    Xn = scaler.transform(X)

    regimes = fit_regimes(Xn, k=config.k_regimes)
    regime_id = regimes.predict(Xn)

    iforest = fit_iforest(Xn, n_estimators=config.iforest_estimators,
                          contamination=config.iforest_contamination)
    anom = (iforest.predict(Xn) == -1).astype(int)

    # Per-tag baselines for drift
    baselines = {}
    for t in tags:
        s = df[t].dropna()
        baselines[t] = {"mu": float(s.mean()), "sigma": float(s.std(ddof=1) or 0.0)}

    # Persist artifacts
    manifest = {
        "feat_cols": feat_cols,
        "tags": tags,
        "resample_rule": config.resample_rule,
        "k_regimes": config.k_regimes,
        "iforest_contamination": config.iforest_contamination,
    }
    joblib.dump(scaler,   os.path.join(ART_DIR, f"{save_prefix}_scaler.joblib"))
    joblib.dump(regimes,  os.path.join(ART_DIR, f"{save_prefix}_regimes.joblib"))
    joblib.dump(iforest,  os.path.join(ART_DIR, f"{save_prefix}_iforest.joblib"))
    with open(os.path.join(ART_DIR, f"{save_prefix}_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Save training diagnostics table (time-aligned)
    diag = fdf.copy()
    diag["Regime"] = regime_id
    diag["Anomaly"] = anom
    diag.to_csv(os.path.join(ART_DIR, f"{save_prefix}_train_diagnostics.csv"))

    # Save baseline stats
    pd.DataFrame(baselines).T.to_csv(os.path.join(ART_DIR, f"{save_prefix}_tag_baselines.csv"))

    return {"ok": True, "artifacts": list(os.listdir(ART_DIR)), "tags": tags, "feat_cols": feat_cols}

def score_window(csv_path: str, save_prefix: str="acm") -> pd.DataFrame:
    """
    Apply trained artifacts to new CSV window; returns per-row diagnostics with Anomaly flag & Regime.
    """
    scaler  = joblib.load(os.path.join(ART_DIR, f"{save_prefix}_scaler.joblib"))
    regimes = joblib.load(os.path.join(ART_DIR, f"{save_prefix}_regimes.joblib"))
    iforest = joblib.load(os.path.join(ART_DIR, f"{save_prefix}_iforest.joblib"))
    with open(os.path.join(ART_DIR, f"{save_prefix}_manifest.json")) as f:
        manifest = json.load(f)
    tags = manifest["tags"]
    feat_cols = manifest["feat_cols"]

    df = pd.read_csv(csv_path)
    df = ensure_time_index(df)
    df = resample_fill(df, manifest["resample_rule"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


    fdf = rolling_features(df[tags], tags).dropna()
    # Backfill any missing engineered columns due to schema drift
    for c in feat_cols:
        if c not in fdf.columns:
            fdf[c] = 0.0
    Xn = scaler.transform(fdf[feat_cols].values)

    regime_id = regimes.predict(Xn)
    anom = (iforest.predict(Xn) == -1).astype(int)

    out = fdf.copy()
    out["Regime"] = regime_id
    out["Anomaly"] = anom
    return out

def drift_check(csv_path: str, save_prefix: str="acm") -> pd.DataFrame:
    """Compute simple drift z-scores vs per-tag baselines saved during training."""
    base = pd.read_csv(os.path.join(ART_DIR, f"{save_prefix}_tag_baselines.csv"), index_col=0)
    df = pd.read_csv(csv_path)
    df = ensure_time_index(df)
    df = resample_fill(df)

    rows = []
    for t in base.index:
        if t not in df.columns: 
            continue
        mu, sigma = float(base.loc[t,"mu"]), float(base.loc[t,"sigma"])
        z = simple_drift_score(df[t].dropna(), mu, sigma)
        rows.append({"Tag": t, "DriftZ": z})
    res = pd.DataFrame(rows).sort_values("DriftZ", ascending=False)
    res.to_csv(os.path.join(ART_DIR, f"{save_prefix}_drift.csv"), index=False)
    return res

# ------------------------- Simple CLIs -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("ACM Core (local)")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train"); tr.add_argument("--csv", required=True)
    sw = sub.add_parser("score"); sw.add_argument("--csv", required=True)
    dr = sub.add_parser("drift"); dr.add_argument("--csv", required=True)

    args = p.parse_args()
    if args.cmd == "train":
        info = train_core(args.csv)
        print(json.dumps(info, indent=2))
    elif args.cmd == "score":
        df = score_window(args.csv)
        outp = os.path.join(ART_DIR, "acm_scored_window.csv")
        df.to_csv(outp)
        print(f"Wrote: {outp}")
    elif args.cmd == "drift":
        df = drift_check(args.csv)
        print(df.head(20).to_string(index=False))

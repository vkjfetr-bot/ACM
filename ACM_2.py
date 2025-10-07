#!/usr/bin/env python3
# acm_rotary.py — Unsupervised, dynamic ACM for rotary equipment (CSV-based, config-less)
# - No SQL, no tag config. Auto-discovers numeric tags.
# - Uses history (from same CSV or a separate one) for baselines, regimes, drift.
# - Outputs machine-readable JSON to stdout (no files/plots).
#
# Usage examples:
#   python acm_rotary.py --csv live.csv --history_csv history.csv --context hour --contam 0.02
#   python acm_rotary.py --csv rotary.csv --start 2025-10-05T00:00:00 --end 2025-10-06T00:00:00

import argparse, json, math, numpy as np, pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# -------------------- Utilities --------------------
def ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Ts" in df.columns:
        df = df.copy(); df["Ts"] = pd.to_datetime(df["Ts"])
        return df.sort_values("Ts").set_index("Ts")
    if isinstance(df.index, pd.DatetimeIndex): return df.sort_index()
    raise SystemExit("CSV must have a 'Ts' column (ISO datetime) or a DatetimeIndex.")

def slice_time(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start: df = df[df.index >= pd.to_datetime(start)]
    if end:   df = df[df.index <= pd.to_datetime(end)]
    if df.empty: raise SystemExit("No data after time filtering.")
    return df

def autodetect_tags(df: pd.DataFrame) -> List[str]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    # drop near-constant columns
    keep = []
    for c in num:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.dropna().nunique() >= 5:
            keep.append(c)
    if not keep: raise SystemExit("No suitable numeric tags found.")
    return keep

def resample_guard(df: pd.DataFrame, max_ffill_mult: float = 5.0) -> pd.DataFrame:
    # infer dominant cadence; resample; cap ffill to max_ffill_mult * cadence
    if len(df.index) < 3: return df
    dt = (df.index.to_series().diff().dropna()).dt.total_seconds()
    if dt.empty: return df
    mode_step = float(dt.mode().iloc[0]) if not dt.mode().empty else float(dt.median())
    if mode_step <= 0: return df
    rule = f"{int(round(mode_step))}S"
    df2 = df.resample(rule).mean()
    limit = int(max(1, round(max_ffill_mult)))
    df2 = df2.ffill(limit=limit).bfill(limit=1)
    return df2

# -------------------- Feature fabric --------------------
def zscore_roll(s: pd.Series, win=60):
    r = s.rolling(win, min_periods=max(3, win//5))
    mu = r.mean(); sd = r.std(ddof=0).replace(0, np.nan)
    return (s - mu) / sd

def robust_mad_score_roll(s: pd.Series, win=60):
    r = s.rolling(win, min_periods=max(3, win//5))
    med = r.median()
    mad = r.apply(lambda x: np.median(np.abs(x - np.median(x))) if len(x) else np.nan, raw=True)
    mad = mad.replace(0, np.nan)
    return 0.6745 * (s - med) / mad

def ewma(s: pd.Series, alpha=0.1):
    return s.ewm(alpha=alpha, adjust=False).mean()

def build_features(df: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
    feats = []
    for t in tags:
        s = pd.to_numeric(df[t], errors="coerce")
        feats.append(pd.DataFrame({
            f"{t}_z60":   zscore_roll(s, 60),
            f"{t}_mad60": robust_mad_score_roll(s, 60),
            f"{t}_ewm":   ewma(s, 0.1)
        }, index=df.index))
    return pd.concat(feats, axis=1)

# -------------------- Baselines (history-aware) --------------------
class Baseline:
    def __init__(self, kind: str, med: Dict[str, any], mad: Dict[str, any], p01: Dict[str, any], p99: Dict[str, any]):
        self.kind, self.med, self.mad, self.p01, self.p99 = kind, med, mad, p01, p99

def build_baseline(df_hist: pd.DataFrame, tags: List[str], kind="hour") -> Baseline:
    dfh = ensure_time_index(df_hist).copy()
    if kind == "global":
        med = {}; mad = {}; p01 = {}; p99 = {}
        for t in tags:
            s = pd.to_numeric(dfh[t], errors="coerce")
            m = float(np.nanmedian(s)); mads = float(np.nanmedian(np.abs(s - m)))
            med[t], mad[t] = m, (mads if mads else np.nan)
            q = s.quantile([0.01,0.99])
            p01[t], p99[t] = float(q.iloc[0]), float(q.iloc[1])
        return Baseline("global", med, mad, p01, p99)

    if kind == "hour":
        dfh["ctx"] = dfh.index.hour
    elif kind == "weekday":
        dfh["ctx"] = dfh.index.weekday
    else:
        raise SystemExit("context must be global|hour|weekday")

    med = {}; mad = {}; p01 = {}; p99 = {}
    for t in tags:
        g = dfh.groupby("ctx")[t]
        med[t] = g.median()
        mad[t] = g.apply(lambda x: float(np.nanmedian(np.abs(x - np.nanmedian(x)))) or np.nan)
        p01[t] = g.quantile(0.01); p99[t] = g.quantile(0.99)
    return Baseline(kind, med, mad, p01, p99)

def z_hist(s: pd.Series, base: Baseline, tag: str, idx: pd.DatetimeIndex) -> pd.Series:
    if base.kind == "global":
        m, M = base.med[tag], base.mad[tag]
        M = M if M not in (0, None) else np.nan
        return (s - m) / M
    ctx = (idx.hour if base.kind == "hour" else idx.weekday)
    med = pd.Series(ctx, index=idx).map(base.med[tag])
    mad = pd.Series(ctx, index=idx).map(base.mad[tag]).replace(0, np.nan)
    return (s - med) / mad

# -------------------- Regimes / states --------------------
class RegimeModel:
    def __init__(self, feat_cols: List[str], kmeans: KMeans):
        self.feat_cols, self.kmeans = feat_cols, kmeans

def train_regimes(df_hist: pd.DataFrame, tags: List[str], k: int = 4, seed=42) -> RegimeModel:
    Fh = build_features(df_hist, tags).dropna()
    if len(Fh) < 200:
        km = KMeans(n_clusters=1, n_init="auto", random_state=seed).fit(Fh)
    else:
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(Fh)
    return RegimeModel(list(Fh.columns), km)

def assign_state(F: pd.DataFrame, model: RegimeModel) -> pd.Series:
    X = F[model.feat_cols].fillna(method="ffill").fillna(method="bfill")
    return pd.Series(model.kmeans.predict(X), index=F.index, name="state")

def state_distance(F: pd.DataFrame, model: RegimeModel) -> pd.Series:
    X = F[model.feat_cols].fillna(method="ffill").fillna(method="bfill").to_numpy()
    centers = model.kmeans.cluster_centers_
    labels = model.kmeans.predict(X)
    d = np.linalg.norm(X - centers[labels], axis=1)
    return pd.Series(d, index=F.index, name="state_dist")

# -------------------- Drift --------------------
def psi(ref: pd.Series, cur: pd.Series, bins=20) -> float:
    r = pd.cut(ref.dropna(), bins=bins, duplicates="drop").value_counts(normalize=True).sort_index()
    c = pd.cut(cur.dropna(), bins=r.index.categories, include_lowest=True).value_counts(normalize=True).sort_index()
    r, c = r.align(c, join="outer", fill_value=0.0)
    r = r.replace(0, 1e-6); c = c.replace(0, 1e-6)
    return float(((r - c) * np.log(r / c)).sum())

def page_hinkley(x: pd.Series, delta=0.005, lamb=50.0, alpha=0.999) -> bool:
    m_t = 0.0; PH = 0.0; M_t = 0.0
    for v in x.dropna():
        m_t = alpha * m_t + (1 - alpha) * v
        PH = max(0.0, PH + v - m_t - delta)
        M_t = max(M_t, PH)
        if (M_t - PH) > lamb: return True
    return False

# -------------------- Scoring --------------------
class IFModels:
    def __init__(self, iso: Optional[IsolationForest], feat_cols: List[str]):
        self.iso, self.feat_cols = iso, feat_cols

def train_isoforest(F: pd.DataFrame, contamination=0.02, seed=42) -> IFModels:
    cols = [c for c in F.columns if c.endswith("_z60") or c.endswith("_mad60")]
    X = F[cols].dropna()
    if len(X) < 100: return IFModels(None, cols)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    return IFModels(iso, cols)

def iso_score(F: pd.DataFrame, models: IFModels) -> pd.Series:
    if models.iso is None: return pd.Series(np.nan, index=F.index, name="iso_score")
    X = F[models.feat_cols].fillna(method="ffill").fillna(method="bfill")
    return pd.Series(-models.iso.score_samples(X), index=F.index, name="iso_score")

def ensemble_score(df: pd.DataFrame, tags: List[str], base: Baseline,
                   F: pd.DataFrame, iso_s: pd.Series, state_d: pd.Series) -> pd.Series:
    parts = []
    for t in tags:
        s = pd.to_numeric(df[t], errors="coerce")
        parts.append(z_hist(s, base, t, df.index).abs().rename(f"{t}_|z_hist|"))
    parts.append(iso_s); parts.append(state_d)
    S = pd.concat(parts, axis=1)

    def rscale(col: pd.Series):
        q1, q99 = col.quantile(0.01), col.quantile(0.99)
        return (col - q1) / (q99 - q1 + 1e-9)

    S = S.apply(rscale)
    return S.mean(axis=1).rename("anom_score")

def quantile_cut(series: pd.Series, q: float):
    return series.quantile(q) if series.notna().any() else np.nan

# -------------------- Intervals --------------------
def build_spans(hit: pd.Series, min_len_s=60, merge_gap_s=30) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    h = hit.fillna(False).astype(bool)
    if not h.any(): return []
    idx = h.index
    spans, active, s0, prev = [], False, None, idx[0]
    for t, v in h.items():
        if v and not active: active, s0 = True, t
        elif not v and active:
            spans.append((s0, prev)); active = False
        prev = t
    if active: spans.append((s0, idx[-1]))
    # merge near gaps
    merged = []
    for s, e in spans:
        if merged and (s - merged[-1][1]).total_seconds() <= merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s).total_seconds() >= min_len_s]

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Unsupervised Dynamic ACM for Rotary Equipment (CSV)")
    ap.add_argument("--csv", required=True, help="Input CSV with Ts + sensor columns")
    ap.add_argument("--history_csv", help="Optional historical CSV for baselines/regimes")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--context", choices=["global","hour","weekday"], default="hour")
    ap.add_argument("--contam", type=float, default=0.02, help="Anomaly fraction target")
    ap.add_argument("--k", type=int, default=4, help="Regime clusters")
    ap.add_argument("--min_span", type=int, default=60)
    ap.add_argument("--merge_gap", type=int, default=30)
    args = ap.parse_args()

    # Load & prepare
    df = ensure_time_index(pd.read_csv(args.csv))
    df = slice_time(df, args.start, args.end)
    df = resample_guard(df)
    tags = autodetect_tags(df)

    if args.history_csv:
        hist = resample_guard(ensure_time_index(pd.read_csv(args.history_csv)))
    else:
        # Use earlier portion of df as history if not provided
        split_idx = int(len(df)*0.6)
        hist = df.iloc[:split_idx]
        df   = df.iloc[split_idx:]

    # Baseline from history
    baseline = build_baseline(hist, tags, kind=args.context)

    # Feature space
    F_hist = build_features(hist, tags)
    F = build_features(df, tags)

    # Regimes from history → states/dist on current
    regimes = train_regimes(hist, tags, k=args.k)
    states  = assign_state(F, regimes)
    sdist   = state_distance(F, regimes)

    # Drift checks (basic, optional)
    drift = {}
    try:
        for t in tags[:5]:  # cap for speed
            drift[f"psi_{t}"] = psi(hist[t], df[t])
        drift["page_hinkley_state_dist"] = page_hinkley(sdist)
    except Exception:
        drift = {}

    # Anomaly ensemble
    ifmods = train_isoforest(F_hist.append(F), contamination=args.contam)
    iso_s  = iso_score(F, ifmods)
    score  = ensemble_score(df, tags, baseline, F, iso_s, sdist)

    # Threshold + spans
    q = quantile_cut(score, 1 - args.contam)
    hit = score >= q if isinstance(q, float) and not math.isnan(q) else pd.Series(False, index=score.index)
    spans = build_spans(hit, min_len_s=args.min_span, merge_gap_s=args.merge_gap)

    # Output (JSON only; no files/plots)
    out = {
        "rows": int(len(df)),
        "tags_count": int(len(tags)),
        "states_present": int(states.nunique()),
        "anomaly_fraction": float(hit.mean()),
        "threshold": float(q) if q==q else None,
        "events": [
            {"start": str(s), "end": str(e),
             "peak_score": float(score.loc[s:e].max()),
             "mode_state": int(states.loc[s:e].mode().iloc[0]) if len(states.loc[s:e]) else None}
            for (s, e) in spans
        ],
        "drift": drift
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()

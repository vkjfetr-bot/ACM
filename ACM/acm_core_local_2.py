# acm_core_local_2.py
# Final plan implemented: robust TS parsing, numeric resample, windowed features (time+freq),
# three-head detection (H1 forecast ridge, H2 reconstruction PCA, H3 contrastive embedding),
# auto-k regimes, drift, context masking, episodes, corroboration, change-point heuristic,
# fused scoring, block-by-block timings and JSONL run log.

import os, json, math, joblib, warnings, time, uuid, datetime as dt
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from numpy.fft import rfft, rfftfreq

warnings.filterwarnings("ignore")

# ---- Static path per your request ----
ART_DIR = r"C:\Users\bhadk\Documents\CPCL\ACM\acm_artifacts"
os.makedirs(ART_DIR, exist_ok=True)
print(f"[ACM] Using static ART_DIR: {ART_DIR}")

RUN_ID = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
RUN_LOG = os.path.join(ART_DIR, f"run_{RUN_ID}.jsonl")

def _log_jsonl(obj: Dict):
    obj = {"ts": dt.datetime.utcnow().isoformat(), "run_id": RUN_ID, **obj}
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

class Timer:
    def __init__(self, name, notes=None):
        self.name = name
        self.notes = notes or {}
    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"[{self.name}] START")
        _log_jsonl({"block": self.name, "event": "start", "notes": self.notes})
        return self
    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.perf_counter()
        dur = self.t1 - self.t0
        print(f"[{self.name}] END   {dur:.3f}s")
        _log_jsonl({"block": self.name, "event": "end", "duration_s": round(dur, 6), "notes": self.notes})

# ------------------------- Config -------------------------
@dataclass
class CoreConfig:
    resample_rule: str = "1min"
    window: int = 256            # samples per window (after resample)
    stride: int = 64             # hop
    k_min: int = 2
    k_max: int = 6
    if_contamination: float = 0.02  # optional IF (kept in case you want it later)
    slope_thr: float = 0.75      # context mask slope threshold (perc of IQR per min)
    accel_thr: float = 1.25      # context mask accel threshold
    fused_tau: float = 0.7       # episode threshold on fused score [0..1]
    merge_gap: int = 3           # windows gap allowed for episode merge
    max_fft_bins: int = 64       # keep first N rfft bins as features
    forecast_lags: int = 1       # H1 lags
    corroboration_pairs: int = 20  # top pairs to monitor for corr break

# ------------------------- IO & time handling -------------------------
def ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    cand_keys = {"ts","timestamp","time","datetime"}
    ts_col = None
    for c in df.columns:
        if str(c).strip().lower() in cand_keys:
            ts_col = c; break
    if ts_col is None:
        for c in ["Ts","ts","TS","Timestamp","timestamp","Time","time","Datetime","datetime"]:
            if c in df.columns:
                ts_col = c; break
    if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    if ts_col is None:
        print("[ACM][WARN] No timestamp column found; continuing without resample.")
        return df

    s = df[ts_col]
    dt1 = pd.to_datetime(s, errors="coerce", format="mixed", utc=False)
    dt2 = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True, utc=False)
    num = pd.to_numeric(s, errors="coerce")
    excel_dt = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")

    ts_parsed = dt1.fillna(dt2).fillna(excel_dt.where(num.notna(), pd.NaT))
    bad = ts_parsed.isna().sum()
    if bad > 0:
        print(f"[ACM][WARN] Skipping {bad} row(s) with invalid timestamp format.")
    good = ts_parsed.notna()
    if good.any():
        out = df.loc[good].copy()
        out.index = ts_parsed.loc[good]
        return out.sort_index()
    print("[ACM][WARN] All timestamps invalid; continuing without resample.")
    return df

def resample_numeric(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return df
    num = df.apply(pd.to_numeric, errors="coerce")
    g = num.resample(rule).mean()
    return g.interpolate(limit_direction="both")

def detect_tags(df: pd.DataFrame, min_unique=20, exclude=("Ts","ts","TS","Timestamp","timestamp","time","Time")) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude: continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) >= min_unique:
            cols.append(c)
    return cols

# ------------------------- Windowing & features -------------------------
def sliding_windows(df: pd.DataFrame, cols: List[str], W: int, S: int):
    idx = df.index
    X = df[cols].values
    n = len(df)
    for start in range(0, max(0, n - W + 1), S):
        end = start + W
        yield (idx[end-1], X[start:end, :])  # align to window end time

def feats_time(window: np.ndarray) -> np.ndarray:
    # compute per-column stats then concatenate
    # stats: mean, rms, var, kurt, skew, crest, slope
    eps = 1e-9
    mu = window.mean(axis=0)
    rms = np.sqrt((window**2).mean(axis=0))
    var = window.var(axis=0)
    # moment helpers
    c = window - mu
    m2 = (c**2).mean(axis=0) + eps
    m3 = (c**3).mean(axis=0)
    m4 = (c**4).mean(axis=0)
    skew = m3 / np.power(m2, 1.5)
    kurt = m4 / (m2**2)
    peak = np.max(np.abs(window), axis=0) + eps
    crest = peak / (rms + eps)
    # slope via simple linear regression on last half
    half = window[window.shape[0]//2:, :]
    t = np.arange(half.shape[0])[:, None]
    # slope = cov(t,x)/var(t)
    vt = (t - t.mean()).T @ (t - t.mean())
    vt = vt[0,0] if vt.size else 1.0
    cov = (t - t.mean()).T @ (half - half.mean(axis=0))
    slope = (cov / (vt + eps)).flatten()
    return np.concatenate([mu, rms, var, kurt, skew, crest, slope], axis=0)

def feats_freq(window: np.ndarray, max_bins: int) -> np.ndarray:
    # apply rfft per column; keep first max_bins magnitudes & basic spectral stats
    W, D = window.shape
    fft_feats = []
    for d in range(D):
        spec = np.abs(rfft(window[:, d]))
        if len(spec) < max_bins:
            pad = np.zeros(max_bins - len(spec))
            spec = np.concatenate([spec, pad])
        else:
            spec = spec[:max_bins]
        # spectral centroid & flatness
        idx = np.arange(len(spec)) + 1
        centroid = (spec * idx).sum() / (spec.sum() + 1e-9)
        geo = np.exp(np.log(spec + 1e-9).mean())
        arith = spec.mean() + 1e-9
        flatness = geo / arith
        fft_feats.append(np.concatenate([spec, [centroid, flatness]]))
    return np.concatenate(fft_feats, axis=0)

def build_feature_matrix(df: pd.DataFrame, tags: List[str], W: int, S: int, max_bins: int):
    rows, times = [], []
    for t_end, w in sliding_windows(df, tags, W, S):
        x = np.nan_to_num(w, copy=False)
        f_t = feats_time(x)
        f_f = feats_freq(x, max_bins)
        rows.append(np.concatenate([f_t, f_f], axis=0))
        times.append(t_end)
    X = np.vstack(rows) if rows else np.empty((0,0))
    return pd.DataFrame(X, index=pd.to_datetime(times))

# ------------------------- Heads & models -------------------------
def auto_kmeans(Xn: np.ndarray, kmin=2, kmax=6, random_state=17) -> KMeans:
    km_best, sil_best = None, -1.0
    for k in range(kmin, kmax+1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(Xn)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(Xn, labels, metric="euclidean")
        if sil > sil_best:
            sil_best, km_best = sil, km
    return km_best if km_best is not None else KMeans(n_clusters=kmin, n_init="auto", random_state=17).fit(Xn)

def fit_h1_forecast(df: pd.DataFrame, tags: List[str], lags: int=5):
    models = {}
    for t in tags:
        s = df[t].astype(float).values
        if len(s) <= lags + 10: continue
        X, y = [], []
        for i in range(lags, len(s)):
            X.append(s[i-lags:i])
            y.append(s[i])
        X, y = np.asarray(X), np.asarray(y)
        sc = StandardScaler().fit(X)
        Xn = sc.transform(X)
        mdl = Ridge(alpha=1.0).fit(Xn, y)
        models[t] = {"scaler": sc, "ridge": mdl, "lags": lags}
    return models

def h1_score(df: pd.DataFrame, models: Dict, tags: List[str]) -> pd.Series:
    zs = []
    for t in tags:
        if t not in models: continue
        s = df[t].astype(float).values
        l = models[t]["lags"]
        sc = models[t]["scaler"]; mdl = models[t]["ridge"]
        preds, errs = [], []
        for i in range(l, len(s)):
            x = s[i-l:i]
            xn = sc.transform([x])
            p = mdl.predict(xn)[0]
            e = abs(s[i] - p)
            preds.append(p); errs.append(e)
        if len(errs) == 0:
            zs.append(pd.Series(index=df.index, data=np.zeros(len(df))))
            continue
        err = pd.Series([np.nan]*l + errs, index=df.index)
        mu, sd = err.mean(skipna=True), err.std(skipna=True) + 1e-9
        z = (err - mu) / sd
        zs.append(z.fillna(0.0))
    if not zs:
        return pd.Series(index=df.index, data=np.zeros(len(df)))
    zmean = pd.concat(zs, axis=1).mean(axis=1).clip(lower=0)
    # normalize to [0,1]
    p95 = zmean.quantile(0.95) or 1.0
    return (zmean / (p95 + 1e-9)).clip(0, 1)

def fit_h2_recon(Xn: np.ndarray, keep_ratio=0.9):
    pca = PCA(n_components=keep_ratio, svd_solver="full").fit(Xn)
    return pca

def h2_score(Xn: np.ndarray, pca: PCA, index) -> pd.Series:
    Xr = pca.inverse_transform(pca.transform(Xn))
    err = ((Xn - Xr)**2).sum(axis=1)
    s = pd.Series(err, index=index)
    p95 = s.quantile(0.95) or 1.0
    return (s / (p95 + 1e-9)).clip(0, 1)

def h3_score(Emb: np.ndarray, index, ref_win=50) -> pd.Series:
    # rolling baseline embedding mean; score = cosine distance to rolling mean
    from numpy.linalg import norm
    emb = Emb
    scores = np.zeros(len(emb))
    for i in range(len(emb)):
        j0 = max(0, i - ref_win); ref = emb[j0:i]
        if len(ref) < 10:
            scores[i] = 0.0; continue
        mu = ref.mean(axis=0)
        num = (emb[i] * mu).sum()
        den = (norm(emb[i]) * norm(mu) + 1e-9)
        cos = num / den
        scores[i] = max(0.0, 1.0 - cos)  # 0..1
    return pd.Series(scores, index=index)

# ------------------------- Context masking & episodes -------------------------
def detect_transients(df: pd.DataFrame, tags: List[str], slope_thr: float, accel_thr: float) -> pd.Series:
    # simple slope/accel mask on z-scored signals
    z = (df[tags] - df[tags].mean()) / (df[tags].std() + 1e-9)
    slope = z.diff().abs().median(axis=1).fillna(0)
    accel = z.diff(2).abs().median(axis=1).fillna(0)
    mask = ((slope > slope_thr) | (accel > accel_thr)).astype(int)
    return mask

def build_episodes(score: pd.Series, tau: float, merge_gap: int) -> List[Dict]:
    events, active = [], None
    for t, v in score.items():
        if v >= tau:
            if active is None:
                active = {"start": t, "end": t, "peak": v}
            else:
                active["end"] = t
                if v > active["peak"]: active["peak"] = v
        else:
            if active is not None:
                events.append(active); active = None
    if active is not None: events.append(active)
    # merge close events
    merged = []
    for e in events:
        if not merged: merged.append(e); continue
        gap = (e["start"] - merged[-1]["end"]).total_seconds()
        # convert merge_gap (windows) to seconds ~ assume minute resample
        if gap <= merge_gap * 60:
            merged[-1]["end"] = e["end"]
            merged[-1]["peak"] = max(merged[-1]["peak"], e["peak"])
        else:
            merged.append(e)
    return merged

def corroboration_boost(df: pd.DataFrame, tags: List[str], pairs: int=20) -> pd.Series:
    # detect co-deviation via instantaneous z |z| sum over top-correlated pairs
    if len(tags) < 2:
        return pd.Series(index=df.index, data=np.zeros(len(df)))
    corr = df[tags].corr().abs()
    iu = np.triu_indices_from(corr, k=1)
    pairs_sorted = sorted([(corr.values[iu][k], iu[0][k], iu[1][k]) for k in range(len(iu[0]))], reverse=True)
    pairs_sorted = pairs_sorted[:min(pairs, len(pairs_sorted))]
    z = (df[tags] - df[tags].mean()) / (df[tags].std() + 1e-9)
    s = pd.Series(0.0, index=df.index)
    for _, i, j in pairs_sorted:
        si = z.iloc[:, i].abs(); sj = z.iloc[:, j].abs()
        s = s + (si.add(sj, fill_value=0))/2.0
    s = s / (s.quantile(0.95) + 1e-9)
    return s.clip(0,1)

def change_point_signal(df: pd.DataFrame, tags: List[str], W: int=60) -> pd.Series:
    # simple mean/var change heuristic: rolling mean/std difference
    mu = df[tags].rolling(W, min_periods=5).mean()
    sd = df[tags].rolling(W, min_periods=5).std()
    dmu = mu.diff().abs().median(axis=1).fillna(0)
    dsd = sd.diff().abs().median(axis=1).fillna(0)
    s = dmu + dsd
    return (s / (s.quantile(0.95) + 1e-9)).clip(0,1)

# ------------------------- Pipeline -------------------------
def train_core(csv_path: str, cfg: Optional[CoreConfig]=None, save_prefix="acm") -> Dict:
    cfg = cfg or CoreConfig()

    with Timer("LOAD_DATA", {"csv": csv_path}):
        df = pd.read_csv(csv_path)

    with Timer("CLEAN_TIME"):
        df = ensure_time_index(df)

    with Timer("RESAMPLE", {"rule": cfg.resample_rule}):
        df = resample_numeric(df, cfg.resample_rule)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    with Timer("TAG_SELECT"):
        tags = detect_tags(df)
        if not tags:
            raise ValueError("No numeric tags with enough variability.")
        _log_jsonl({"block":"TAG_SELECT","event":"tags","tags":tags[:10]+(["..."] if len(tags)>10 else [])})

    with Timer("FEATURES_BUILD", {"W": cfg.window, "S": cfg.stride, "fft_bins": cfg.max_fft_bins}):
        F = build_feature_matrix(df, tags, cfg.window, cfg.stride, cfg.max_fft_bins)
        scaler = RobustScaler().fit(F.values)
        Xn = scaler.transform(F.values)

    with Timer("REGIMES_AUTO_K", {"kmin": cfg.k_min, "kmax": cfg.k_max}):
        km = auto_kmeans(Xn, cfg.k_min, cfg.k_max)
        regimes = km.predict(Xn)
        regime_switches = int(np.sum(regimes[1:] != regimes[:-1]))

    with Timer("H1_FIT_FORECAST", {"lags": cfg.forecast_lags}):
        h1_models = fit_h1_forecast(df, tags, cfg.forecast_lags)

    with Timer("H2_FIT_PCA"):
        pca = fit_h2_recon(Xn, keep_ratio=0.9)

    # Baselines for drift
    with Timer("DRIFT_BASELINES"):
        baselines = {}
        for t in tags:
            s = df[t].dropna()
            baselines[t] = {"mu": float(s.mean()), "sigma": float(s.std(ddof=1) or 0.0)}
        pd.DataFrame(baselines).T.to_csv(os.path.join(ART_DIR, f"{save_prefix}_tag_baselines.csv"))

    # Persist
    with Timer("EXPORT_ARTIFACTS"):
        joblib.dump(scaler,  os.path.join(ART_DIR, f"{save_prefix}_scaler.joblib"))
        joblib.dump(km,      os.path.join(ART_DIR, f"{save_prefix}_regimes.joblib"))
        joblib.dump(pca,     os.path.join(ART_DIR, f"{save_prefix}_pca.joblib"))
        joblib.dump(h1_models, os.path.join(ART_DIR, f"{save_prefix}_h1_models.joblib"))
        manifest = {
            "tags": tags, "resample_rule": cfg.resample_rule, "window": cfg.window, "stride": cfg.stride,
            "k_min": cfg.k_min, "k_max": cfg.k_max, "max_fft_bins": cfg.max_fft_bins,
            "forecast_lags": cfg.forecast_lags, "fused_tau": cfg.fused_tau, "merge_gap": cfg.merge_gap
        }
        with open(os.path.join(ART_DIR, f"{save_prefix}_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        # training diagnostics
        diag = F.copy()
        diag["Regime"] = regimes
        diag.to_csv(os.path.join(ART_DIR, f"{save_prefix}_train_diagnostics.csv"))

    with Timer("SUMMARY"):
        print(f"Tags: {len(tags)} | Feature rows: {len(F)} | Regimes: {km.n_clusters} | Switches: {regime_switches}")

    return {"ok": True, "tags": tags, "rows": len(F), "regimes": int(km.n_clusters)}

def score_window(csv_path: str, save_prefix="acm") -> Dict:
    with Timer("LOAD_DATA", {"csv": csv_path}):
        df = pd.read_csv(csv_path)

    with Timer("CLEAN_TIME"):
        df = ensure_time_index(df)

    with open(os.path.join(ART_DIR, f"{save_prefix}_manifest.json")) as f:
        man = json.load(f)

    with Timer("RESAMPLE", {"rule": man["resample_rule"]}):
        df = resample_numeric(df, man["resample_rule"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    tags = man["tags"]
    W, S, max_fft = man["window"], man["stride"], man["max_fft_bins"]

    with Timer("FEATURES_BUILD", {"W": W, "S": S, "fft_bins": max_fft}):
        F = build_feature_matrix(df, tags, W, S, max_fft)
        scaler = joblib.load(os.path.join(ART_DIR, f"{save_prefix}_scaler.joblib"))
        Xn = scaler.transform(F.values)

    with Timer("REGIMES"):
        km = joblib.load(os.path.join(ART_DIR, f"{save_prefix}_regimes.joblib"))
        regimes = km.predict(Xn)

    with Timer("H1_SCORE"):
        h1_models = joblib.load(os.path.join(ART_DIR, f"{save_prefix}_h1_models.joblib"))
        s_h1 = h1_score(df, h1_models, tags)
        s_h1 = s_h1.reindex(F.index).fillna(method="ffill").fillna(0)

    with Timer("H2_SCORE"):
        pca = joblib.load(os.path.join(ART_DIR, f"{save_prefix}_pca.joblib"))
        s_h2 = h2_score(Xn, pca, F.index)

    with Timer("H3_SCORE"):
        # use reduced PCA projection as embedding
        Emb = pca.transform(Xn)
        s_h3 = h3_score(Emb, F.index)

    with Timer("CONTEXT_MASKS", {"slope_thr": man.get("slope_thr", 0.75), "accel_thr": man.get("accel_thr", 1.25)}):
        mask = detect_transients(df, tags, slope_thr=0.75, accel_thr=1.25)
        mask = mask.reindex(F.index).fillna(0)

    with Timer("CORROBORATION"):
        corr_boost = corroboration_boost(df.reindex(F.index, method="nearest"), tags, pairs=20)

    with Timer("CHANGEPOINT"):
        cpd = change_point_signal(df.reindex(F.index, method="nearest"), tags, W=60)

    with Timer("FUSION"):
        # base fusion: weighted max; context reduces score by 30% where mask=1; add boosts
        s_h1v, s_h2v, s_h3v = s_h1.values, s_h2.values, s_h3.values
        base = np.maximum.reduce([0.45*s_h1v, 0.35*s_h2v, 0.35*s_h3v])  # overlaps okay; scale <=1 with clipping
        boost = 0.15*corr_boost.values + 0.10*cpd.values
        fused = np.clip(base + boost, 0, 1)
        fused = np.where(mask.values>0, fused*0.7, fused)  # context masking

    with Timer("EPISODES", {"tau": man.get("fused_tau", 0.7), "merge_gap": man.get("merge_gap", 3)}):
        tau = man.get("fused_tau", 0.7)
        episodes = build_episodes(pd.Series(fused, index=F.index), tau, man.get("merge_gap",3))

    with Timer("EXPORT"):
        out = F.copy()
        out["Regime"] = regimes
        out["H1_Forecast"] = s_h1.reindex(F.index).values
        out["H2_Recon"] = s_h2.values
        out["H3_Contrast"] = s_h3.values
        out["CorrBoost"] = corr_boost.values
        out["CPD"] = cpd.values
        out["ContextMask"] = mask.values
        out["FusedScore"] = fused
        out.to_csv(os.path.join(ART_DIR, f"{save_prefix}_scored_window.csv"))

        # episodes
        rows = []
        for e in episodes:
            rows.append({"Start": e["start"], "End": e["end"], "PeakScore": e["peak"]})
        pd.DataFrame(rows).to_csv(os.path.join(ART_DIR, f"{save_prefix}_events.csv"), index=False)

        # context masks
        pd.DataFrame({"Ts": F.index, "Mask": mask.values}).to_csv(os.path.join(ART_DIR, f"{save_prefix}_context_masks.csv"), index=False)

    with Timer("SUMMARY"):
        print(f"Rows: {len(F):,} | Regimes seen: {len(np.unique(regimes))} | Events: {len(episodes)}")
    return {"ok": True, "rows": int(len(F)), "events": int(len(episodes))}

def drift_check(csv_path: str, save_prefix="acm") -> pd.DataFrame:
    base = pd.read_csv(os.path.join(ART_DIR, f"{save_prefix}_tag_baselines.csv"), index_col=0)
    with Timer("LOAD_DATA", {"csv": csv_path}):
        df = pd.read_csv(csv_path)
    with Timer("CLEAN_TIME"):
        df = ensure_time_index(df)
    with Timer("RESAMPLE"):
        df = resample_numeric(df, "1min")
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    rows = []
    for t in base.index:
        if t not in df.columns: 
            continue
        mu, sigma = float(base.loc[t,"mu"]), float(base.loc[t,"sigma"])
        if sigma <= 1e-9: z = 0.0
        else: z = float(abs(df[t].mean() - mu) / sigma)
        rows.append({"Tag": t, "DriftZ": z})
    res = pd.DataFrame(rows).sort_values("DriftZ", ascending=False)
    res.to_csv(os.path.join(ART_DIR, f"{save_prefix}_drift.csv"), index=False)
    print(res.head(15).to_string(index=False))
    return res

# ------------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("ACM Core (final)")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train"); tr.add_argument("--csv", required=True)
    sc = sub.add_parser("score"); sc.add_argument("--csv", required=True)
    dr = sub.add_parser("drift"); dr.add_argument("--csv", required=True)
    args = p.parse_args()
    with Timer("START"):
        pass
    try:
        if args.cmd == "train":
            info = train_core(args.csv)
            print(json.dumps(info, indent=2))
        elif args.cmd == "score":
            info = score_window(args.csv)
            print(json.dumps(info, indent=2))
        elif args.cmd == "drift":
            drift_check(args.csv)
    finally:
        with Timer("END"):
            pass

# acm_core_local_2.py
# Core ACM pipeline (final): robust TS parsing, numeric resample, windowed time+freq features,
# H1 = Forecast-lite + AR(1) (fast), H2 = PCA reconstruction, H3 = embedding drift,
# regimes (auto-k), drift, context masking, corroboration, CPD, episodes, fused score,
# block-by-block timings and JSONL run log.

import os, json, math, warnings, time, uuid, datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from numpy.fft import rfft
from sklearn.ensemble import VotingClassifier
from sklearn.mixture import GaussianMixture
from scipy.stats import mode
import acm_observe

warnings.filterwarnings("ignore")

# ---- Artifacts directory (env or local default) ----
ART_DIR = os.environ.get("ACM_ART_DIR")
if not ART_DIR:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ART_DIR = os.path.join(root, "acm_artifacts")
os.makedirs(ART_DIR, exist_ok=True)
print(f"[ACM] Using ART_DIR: {ART_DIR}")

# Lightweight mode (testing): set env ACM_SKIP_DIAG=1 to skip heavy CSVs
SKIP_DIAG = os.environ.get("ACM_SKIP_DIAG", "0") == "1"

RUN_ID = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
RUN_LOG = os.path.join(ART_DIR, f"run_{RUN_ID}.jsonl")

def _log_jsonl(obj: Dict):
    obj = {"ts": dt.datetime.utcnow().isoformat(), "run_id": RUN_ID, **obj}
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

class Timer:
    def __init__(self, name, notes=None):
        self.name = name; self.notes = notes or {}
    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"[{self.name}] START")
        _log_jsonl({"block": self.name, "event": "start", "notes": self.notes})
        return self
    def __exit__(self, exc_type, exc, tb):
        dur = time.perf_counter() - self.t0
        print(f"[{self.name}] END   {dur:.3f}s")
        _log_jsonl({"block": self.name, "event": "end", "duration_s": round(dur, 6), "notes": self.notes})

# ------------------------- Guardrail helpers -------------------------
GUARDRAIL_WARN_STEP_PCT = 20.0
GUARDRAIL_ALERT_STEP_PCT = 40.0


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _equip_name(csv_path: str, provided: Optional[str] = None) -> str:
    if provided:
        return provided
    env = os.environ.get("ACM_EQUIP")
    if env:
        return env
    try:
        return Path(csv_path).stem
    except Exception:
        return "equipment"


def _data_span_minutes(idx: pd.Index) -> float:
    if not isinstance(idx, pd.DatetimeIndex) or idx.size == 0:
        return 0.0
    span = idx.max() - idx.min()
    return span.total_seconds() / 60.0


def compute_dq_metrics(df: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
    rows = []
    for tag in tags:
        s = pd.to_numeric(df.get(tag), errors="coerce")
        total = len(s)
        if total == 0:
            rows.append({"Tag": tag, "flatline_pct": 0.0, "dropout_pct": 0.0,
                         "nan_pct": 0.0, "presence_ratio": 0.0, "spikes_pct": 0.0})
            continue
        nan_mask = s.isna()
        presence_ratio = float((~nan_mask).sum()) / float(total)
        nan_pct = float(nan_mask.sum()) / float(total) * 100.0
        clean = s.fillna(method="ffill").fillna(method="bfill")
        diffs = clean.diff().abs()
        flatline_pct = float((diffs <= 1e-9).sum()) / max(total - 1, 1) * 100.0
        dropout_pct = float((clean == 0).sum()) / float(total) * 100.0
        if clean.std(ddof=0) > 1e-9:
            z = (clean - clean.mean()) / clean.std(ddof=0)
            spikes_pct = float((z.abs() > 4).sum()) / float(total) * 100.0
        else:
            spikes_pct = 0.0
        rows.append({
            "Tag": tag,
            "flatline_pct": round(flatline_pct, 3),
            "dropout_pct": round(dropout_pct, 3),
            "nan_pct": round(nan_pct, 3),
            "presence_ratio": round(presence_ratio, 3),
            "spikes_pct": round(spikes_pct, 3),
        })
    return pd.DataFrame(rows)


PHASE_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _dq_path(save_prefix: str) -> Path:
    return Path(ART_DIR) / f"{save_prefix}_dq.csv"


def write_dq_metrics(df: pd.DataFrame, tags: List[str], save_prefix: str) -> None:
    if not tags:
        return
    dq = compute_dq_metrics(df, tags)
    path_pref = _dq_path(save_prefix)
    path_plain = Path(ART_DIR) / "dq.csv"
    dq.to_csv(path_pref, index=False)
    dq.to_csv(path_plain, index=False)


def dynamic_threshold(series: pd.Series, q: float = 0.95, alpha: float = 0.2) -> pd.Series:
    if series.empty:
        return series.copy()
    quant = series.expanding(min_periods=1).quantile(q)
    theta = quant.ewm(alpha=alpha, adjust=False).mean()
    return theta.clip(lower=0.0)


def _threshold_path(save_prefix: str) -> Path:
    return Path(ART_DIR) / f"{save_prefix}_thresholds.csv"


def load_previous_theta(save_prefix: str) -> Optional[float]:
    path = _threshold_path(save_prefix)
    if not path.exists():
        return None
    try:
        df_prev = pd.read_csv(path)
        if "Theta" in df_prev.columns and not df_prev.empty:
            return float(df_prev["Theta"].iloc[-1])
    except Exception:
        return None
    return None


def save_thresholds(theta: pd.Series, save_prefix: str) -> None:
    out = pd.DataFrame({"Ts": theta.index, "Theta": theta.values})
    out.to_csv(_threshold_path(save_prefix), index=False)
    out.to_csv(Path(ART_DIR) / "thresholds.csv", index=False)


def determine_phase(rows: int, tags: int, span_min: float, feature_rank: int) -> Dict[str, Any]:
    reasons = []
    if rows < 1000 or feature_rank < 5:
        reasons.append("Insufficient samples (<1k) or low feature rank")
        phase = "P0"
        cfg = {"k_min": 1, "k_max": 1, "enable_pca": False, "enable_h3": False}
    elif rows < 5000:
        reasons.append("Limited history (<5k samples)")
        phase = "P1"
        cfg = {"k_min": 1, "k_max": 1, "enable_pca": False, "enable_h3": False}
    elif rows < 20000:
        reasons.append("Moderate history (<20k samples)")
        phase = "P2"
        cfg = {"k_min": 1, "k_max": 3, "enable_pca": True, "enable_h3": True}
    else:
        phase = "P3"
        cfg = {"k_min": 2, "k_max": 6, "enable_pca": True, "enable_h3": True}
    reason = "; ".join(reasons) if reasons else "Full history"
    cfg.update(
        {
            "phase": phase,
            "reason": reason,
            "rows": rows,
            "tags": tags,
            "span_min": span_min,
            "feature_rank": feature_rank,
            "min_cluster_frac": 0.10,
            "silhouette_floor": 0.15,
        }
    )
    return cfg


def phase_to_guardrail(save_prefix: str, phase_info: Dict[str, Any]) -> None:
    phase = phase_info.get("phase", "P3")
    if phase in ("P0", "P1"):
        _append_guardrail(
            save_prefix,
            "phase_limited",
            "warn",
            f"Phase {phase}: {phase_info.get('reason','limited history')}",
            {
                "phase": phase,
                "rows": phase_info.get("rows", 0),
                "feature_rank": phase_info.get("feature_rank", 0),
            },
        )


def theta_step_pct(prev_theta: Optional[float], curr_theta: Optional[float]) -> float:
    if prev_theta is None or curr_theta is None:
        return 0.0
    if abs(prev_theta) < 1e-9:
        return 0.0 if abs(curr_theta) < 1e-9 else 100.0
    return ((curr_theta - prev_theta) / abs(prev_theta)) * 100.0


def artifact_age_minutes(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        mtime = dt.datetime.utcfromtimestamp(path.stat().st_mtime)
        delta = dt.datetime.utcnow() - mtime
        return max(delta.total_seconds() / 60.0, 0.0)
    except Exception:
        return 0.0


def _append_guardrail(save_prefix: str, g_type: str, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        acm_observe.append_guardrail_event(ART_DIR, g_type, level=level, message=message, extra={"save_prefix": save_prefix, **(extra or {})})
    except Exception as exc:
        print(f"[GUARDRAIL][WARN] Failed to append guardrail event: {exc}")


def _write_run_summary(row: Dict[str, Any]) -> None:
    try:
        acm_observe.write_run_summary(ART_DIR, row)
        acm_observe.write_run_health(ART_DIR, row)
    except Exception as exc:
        print(f"[RUN_SUMMARY][WARN] Failed to persist run summary: {exc}")

# ------------------------- Config -------------------------
@dataclass
class CoreConfig:
    # sampling / windows
    resample_rule: str = "1min"
    window: int = 256
    stride: int = 64
    max_fft_bins: int = 64
    # regimes
    k_min: int = 2
    k_max: int = 6
    # H1 fast forecast
    h1_mode: str = "lite_ar1"  # "off" | "lite" | "lite_ar1"
    h1_roll: int = 9           # rolling window for lite baseline (samples)
    h1_centered: bool = True   # centered during training (scoring uses trailing)
    h1_robust: bool = True     # MAD z if True else std z
    h1_min_support: int = 200  # min valid samples per tag for AR1
    h1_topk: int = 0           # 0=all tags; else top-K by variance for H1
    # context / fusion
    slope_thr: float = 0.75
    accel_thr: float = 1.25
    fused_tau: float = 0.7
    merge_gap: int = 3
    corroboration_pairs: int = 20

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
    idx = df.index; X = df[cols].values; n = len(df)
    for start in range(0, max(0, n - W + 1), S):
        end = start + W
        yield (idx[end-1], X[start:end, :])

def feats_time(window: np.ndarray) -> np.ndarray:
    eps = 1e-9
    mu = window.mean(axis=0)
    rms = np.sqrt((window**2).mean(axis=0))
    var = window.var(axis=0)
    c = window - mu
    m2 = (c**2).mean(axis=0) + eps
    m3 = (c**3).mean(axis=0)
    m4 = (c**4).mean(axis=0)
    skew = m3 / np.power(m2, 1.5)
    kurt = m4 / (m2**2)
    peak = np.max(np.abs(window), axis=0) + eps
    crest = peak / (rms + eps)
    # slope via regression on last half
    half = window[window.shape[0]//2:, :]
    t = np.arange(half.shape[0])[:, None]
    vt = (t - t.mean()).T @ (t - t.mean())
    vt = vt[0,0] if vt.size else 1.0
    cov = (t - t.mean()).T @ (half - half.mean(axis=0))
    slope = (cov / (vt + eps)).flatten()
    return np.concatenate([mu, rms, var, kurt, skew, crest, slope], axis=0)

def feats_freq(window: np.ndarray, max_bins: int) -> np.ndarray:
    fft_feats = []
    for d in range(window.shape[1]):
        spec = np.abs(rfft(window[:, d]))
        if len(spec) < max_bins:
            spec = np.pad(spec, (0, max_bins - len(spec)))
        else:
            spec = spec[:max_bins]
        idx = np.arange(len(spec)) + 1
        centroid = (spec * idx).sum() / (spec.sum() + 1e-9)
        geo = np.exp(np.log(spec + 1e-9).mean())
        flatness = geo / (spec.mean() + 1e-9)
        fft_feats.append(np.concatenate([spec, [centroid, flatness]]))
    return np.concatenate(fft_feats, axis=0)

def build_feature_matrix(df: pd.DataFrame, tags: List[str], W: int, S: int, max_bins: int):
    rows, times = [], []
    for t_end, w in sliding_windows(df, tags, W, S):
        x = np.nan_to_num(w, copy=False)
        rows.append(np.concatenate([feats_time(x), feats_freq(x, max_bins)], axis=0))
        times.append(t_end)
    X = np.vstack(rows) if rows else np.empty((0,0))
    return pd.DataFrame(X, index=pd.to_datetime(times))

# ------------------------- Regimes -------------------------

class EnsembleRegimes:
    def __init__(self, kmeans: KMeans, gmm: GaussianMixture, gmm_to_km_map: np.ndarray):
        self.kmeans = kmeans
        self.gmm = gmm
        self.gmm_to_km_map = gmm_to_km_map  # array mapping gmm_label -> km_label
        self.n_clusters = int(getattr(kmeans, 'n_clusters', len(np.unique(kmeans.labels_))))

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict using both models
        y_km = self.kmeans.predict(X)
        y_gm = self.gmm.predict(X)
        # Map GMM labels into KMeans label space
        y_gm_mapped = np.array([self.gmm_to_km_map[l] if l < len(self.gmm_to_km_map) else l for l in y_gm])
        # Two-voter majority with tie-breaker preferring KMeans
        agree = (y_km == y_gm_mapped)
        y = y_km.copy()
        y[agree] = y_km[agree]
        y[~agree] = y_km[~agree]
        return y

def _align_labels_hungarian(ref_labels: np.ndarray, tgt_labels: np.ndarray, n_ref: int) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    # Build confusion matrix of shape (n_ref, n_tgt)
    ref_unique = np.arange(n_ref)
    tgt_unique = np.unique(tgt_labels)
    n_tgt = int(tgt_unique.max()) + 1 if len(tgt_unique) else n_ref
    C = np.zeros((n_ref, n_tgt))
    for i in ref_unique:
        for j in range(n_tgt):
            C[i, j] = np.sum((ref_labels == i) & (tgt_labels == j))
    ri, cj = linear_sum_assignment(-C)
    mapping = np.arange(n_tgt)
    for i, j in zip(ri, cj):
        mapping[j] = i
    return mapping

def auto_ensemble_regimes(
    Xn: np.ndarray,
    kmin=2,
    kmax=6,
    random_state=17,
    min_cluster_frac: float = 0.0,
    silhouette_floor: float = 0.0,
) -> EnsembleRegimes:
    """Train an ensemble (KMeans + GMM) at a single selected K and store label alignment."""
    n_samples = Xn.shape[0]
    ks = [k for k in range(max(1, kmin), max(1, kmax) + 1) if k <= n_samples]
    if not ks:
        ks = [1]
    sil_best, k_best, km_best = -1.0, ks[0], None
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(Xn)
        if k < 2 or len(np.unique(labels)) < 2:
            continue
        try:
            sil = silhouette_score(Xn, labels, metric="euclidean")
        except Exception:
            continue
        if silhouette_floor > 0.0 and sil < silhouette_floor:
            continue
        if min_cluster_frac > 0.0:
            counts = np.bincount(labels)
            if counts.size == 0 or counts.min() / float(len(labels)) < min_cluster_frac:
                continue
        if sil > sil_best:
            sil_best, k_best, km_best = sil, k, km
    if km_best is None:
        k_best = max(1, min(kmax, kmin))
        km_best = KMeans(n_clusters=k_best, n_init="auto", random_state=random_state).fit(Xn)
    else:
        km_best.fit(Xn)

    gmm = GaussianMixture(n_components=k_best, covariance_type='full', random_state=random_state, n_init=10)
    gmm.fit(Xn)

    y_km = km_best.predict(Xn)
    y_gm = gmm.predict(Xn)
    mapping = _align_labels_hungarian(y_km, y_gm, n_ref=int(k_best))

    return EnsembleRegimes(km_best, gmm, mapping)


# ------------------------- H1: Forecast-lite + AR(1) -------------------------
def _mad(x: pd.Series) -> float:
    med = x.median()
    return 1.4826 * (x - med).abs().median()

def _zscore(res: pd.Series, robust=True) -> pd.Series:
    if robust:
        m = res.median(); s = _mad(res)
        if not np.isfinite(s) or s < 1e-9:
            s = res.std(ddof=0)
    else:
        m = res.mean(); s = res.std(ddof=0)
    if not np.isfinite(s) or s < 1e-9:
        return pd.Series(0.0, index=res.index)
    return ((res - m).abs() / (s + 1e-9)).clip(lower=0)

def h1_fit_ar1(df: pd.DataFrame, tags: List[str], min_support: int, topk: int=0) -> Dict[str, Dict]:
    # choose tags by variance if topk > 0
    cand = df[tags].var().sort_values(ascending=False).index.tolist() if topk>0 else tags
    if topk>0: cand = cand[:min(topk, len(cand))]
    coeffs = {}
    for t in cand:
        s = df[t].dropna()
        if len(s) < min_support: continue
        x = s.values
        x0, x1 = x[:-1], x[1:]
        if len(x0) < 5: continue
        # phi = corr(x_t, x_{t-1})
        v0 = x0 - x0.mean(); v1 = x1 - x1.mean()
        num = (v0*v1).sum(); den = (np.sqrt((v0*v0).sum()) * np.sqrt((v1*v1).sum()) + 1e-9)
        phi = float(num / den)
        phi = float(np.clip(phi, -0.999, 0.999))
        coeffs[t] = {"phi": phi, "support": int(len(s))}
    return coeffs

def h1_score_fast(df: pd.DataFrame, tags: List[str], mode: str, roll: int, robust: bool, ar1_coeffs: Dict[str, Dict]) -> pd.Series:
    if mode == "off" or len(tags)==0:
        return pd.Series(0.0, index=df.index)

    # 1) rolling baseline residual (trailing; scoring-time safe)
    z_lite_all = []
    for t in tags:
        s = df[t].astype(float)
        if roll < 3: rb = s.rolling(3, min_periods=1).median()
        else:        rb = s.rolling(roll, min_periods=max(3, roll//2)).median()
        r1 = (s - rb).abs()
        z1 = _zscore(r1.fillna(0), robust=robust)
        z_lite_all.append(z1)
    z_lite = pd.concat(z_lite_all, axis=1).mean(axis=1) if z_lite_all else pd.Series(0.0, index=df.index)

    if mode == "lite":
        # normalize to [0,1] by p95
        p95 = z_lite.quantile(0.95) or 1.0
        return (z_lite / (p95 + 1e-9)).clip(0,1)

    # 2) AR(1) residual
    z_ar_all = []
    for t in tags:
        if t not in ar1_coeffs:
            z_ar_all.append(pd.Series(0.0, index=df.index)); continue
        phi = ar1_coeffs[t]["phi"]
        s = df[t].astype(float)
        yhat = s.shift(1) * phi
        r2 = (s - yhat).abs()
        z2 = _zscore(r2.fillna(0), robust=robust)
        z_ar_all.append(z2)
    z_ar = pd.concat(z_ar_all, axis=1).mean(axis=1) if z_ar_all else pd.Series(0.0, index=df.index)

    z = pd.concat([z_lite, z_ar], axis=1).max(axis=1)
    p95 = z.quantile(0.95) or 1.0
    return (z / (p95 + 1e-9)).clip(0,1)

# ------------------------- Context / episodes / extras -------------------------
def detect_transients(df: pd.DataFrame, tags: List[str], slope_thr: float, accel_thr: float) -> pd.Series:
    z = (df[tags] - df[tags].mean()) / (df[tags].std() + 1e-9)
    slope = z.diff().abs().median(axis=1).fillna(0)
    accel = z.diff(2).abs().median(axis=1).fillna(0)
    return ((slope > slope_thr) | (accel > accel_thr)).astype(int)

def build_episodes(score: pd.Series, tau, merge_gap: int) -> List[Dict]:
    events, active = [], None
    for t, v in score.items():
        thr = tau.loc[t] if isinstance(tau, pd.Series) else tau
        if v >= thr:
            if active is None: active = {"start": t, "end": t, "peak": v}
            else:
                active["end"] = t
                if v > active["peak"]: active["peak"] = v
        else:
            if active is not None: events.append(active); active = None
    if active is not None: events.append(active)
    # merge close events (assume 1min spacing)
    merged = []
    for e in events:
        if not merged: merged.append(e); continue
        gap = (e["start"] - merged[-1]["end"]).total_seconds()
        if gap <= merge_gap * 60:
            merged[-1]["end"] = e["end"]
            merged[-1]["peak"] = max(merged[-1]["peak"], e["peak"])
        else:
            merged.append(e)
    return merged

def corroboration_boost(df: pd.DataFrame, tags: List[str], pairs: int=20) -> pd.Series:
    if len(tags) < 2:
        return pd.Series(index=df.index, data=np.zeros(len(df)))
    corr = df[tags].corr().abs()
    iu = np.triu_indices_from(corr, k=1)
    triples = [(corr.values[iu][k], iu[0][k], iu[1][k]) for k in range(len(iu[0]))]
    triples.sort(reverse=True)
    triples = triples[:min(pairs, len(triples))]
    z = (df[tags] - df[tags].mean()) / (df[tags].std() + 1e-9)
    s = pd.Series(0.0, index=df.index)
    for _, i, j in triples:
        si = z.iloc[:, i].abs(); sj = z.iloc[:, j].abs()
        s = s + (si.add(sj, fill_value=0))/2.0
    s = s / (s.quantile(0.95) + 1e-9)
    return s.clip(0,1)

def change_point_signal(df: pd.DataFrame, tags: List[str], W: int=60) -> pd.Series:
    mu = df[tags].rolling(W, min_periods=5).mean()
    sd = df[tags].rolling(W, min_periods=5).std()
    dmu = mu.diff().abs().median(axis=1).fillna(0)
    dsd = sd.diff().abs().median(axis=1).fillna(0)
    s = dmu + dsd
    return (s / (s.quantile(0.95) + 1e-9)).clip(0,1)

# ------------------------- Pipeline -------------------------
def train_core(csv_path: str, cfg: Optional[CoreConfig]=None, save_prefix="acm", equip: Optional[str]=None) -> Dict:
    cfg = cfg or CoreConfig()
    equip_name = _equip_name(csv_path, equip)
    started = time.perf_counter()
    status = "ok"
    err_msg = ""
    rows_in = 0
    tags: List[str] = []
    feat_rows = 0
    regime_count = 0
    data_span_min = 0.0
    feature_rank = 0
    phase_info: Dict[str, Any] = {"phase": "P3"}
    run_info: Dict[str, Any] = {"ok": False}

    try:
        with Timer("LOAD_DATA", {"csv": csv_path}):
            df = pd.read_csv(csv_path)

        with Timer("CLEAN_TIME"):
            df = ensure_time_index(df)

        with Timer("RESAMPLE", {"rule": cfg.resample_rule}):
            df = resample_numeric(df, cfg.resample_rule)
            df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        rows_in = len(df)
        data_span_min = _data_span_minutes(df.index)

        with Timer("TAG_SELECT"):
            tags = detect_tags(df)
            if not tags:
                raise ValueError("No numeric tags with enough variability.")
            _log_jsonl({"block": "TAG_SELECT", "event": "tags", "tags": tags[:10] + (["..."] if len(tags) > 10 else [])})

        write_dq_metrics(df, tags, save_prefix)

        with Timer("FEATURES_BUILD", {"W": cfg.window, "S": cfg.stride, "fft_bins": cfg.max_fft_bins}):
            F = build_feature_matrix(df, tags, cfg.window, cfg.stride, cfg.max_fft_bins)
            feat_rows = len(F)
            scaler = RobustScaler().fit(F.values)
            Xn = scaler.transform(F.values)
            feature_rank = int(np.linalg.matrix_rank(np.nan_to_num(F.values))) if feat_rows else 0

        phase_info = determine_phase(rows_in, len(tags), data_span_min, feature_rank)
        phase_to_guardrail(save_prefix, phase_info)
        k_min = phase_info["k_min"]
        k_max = phase_info["k_max"]

        with Timer("REGIMES_AUTO_K", {"kmin": k_min, "kmax": k_max}):
            km = auto_ensemble_regimes(
                Xn,
                k_min,
                k_max,
                min_cluster_frac=phase_info.get("min_cluster_frac", 0.0),
                silhouette_floor=phase_info.get("silhouette_floor", 0.0),
            )
            regimes = km.predict(Xn)
            regime_switches = int(np.sum(regimes[1:] != regimes[:-1]))
            regime_count = int(getattr(km, "n_clusters", len(np.unique(regimes))))

        h1_enabled = cfg.h1_mode.lower() != "off"
        ar1_coeffs = {}
        with Timer("H1_FIT", {"mode": cfg.h1_mode, "roll": cfg.h1_roll, "topk": cfg.h1_topk}):
            if h1_enabled and cfg.h1_mode.lower() in ("lite_ar1",):
                ar1_coeffs = h1_fit_ar1(df, tags, cfg.h1_min_support, cfg.h1_topk)
                with open(os.path.join(ART_DIR, f"{save_prefix}_h1_ar1.json"), "w") as f:
                    json.dump(ar1_coeffs, f)
                print(f"[H1] AR1 tags: {len(ar1_coeffs)}/{len(tags)}")
            else:
                print("[H1] SKIPPED (mode off or lite-only)")

        with Timer("DRIFT_BASELINES"):
            baselines = {t: {"mu": float(df[t].mean()), "sigma": float(df[t].std(ddof=1) or 0.0)} for t in tags}
            pd.DataFrame(baselines).T.to_csv(os.path.join(ART_DIR, f"{save_prefix}_tag_baselines.csv"))

        with Timer("EXPORT_ARTIFACTS"):
            from joblib import dump
            dump(scaler, os.path.join(ART_DIR, f"{save_prefix}_scaler.joblib"))
            dump(km, os.path.join(ART_DIR, f"{save_prefix}_regimes.joblib"))
            manifest = {
                "equip": equip_name,
                "tags": tags,
                "resample_rule": cfg.resample_rule,
                "window": cfg.window,
                "stride": cfg.stride,
                "max_fft_bins": cfg.max_fft_bins,
                "k_min": k_min,
                "k_max": k_max,
                "h1_mode": cfg.h1_mode,
                "h1_roll": cfg.h1_roll,
                "h1_centered": cfg.h1_centered,
                "h1_robust": cfg.h1_robust,
                "h1_topk": cfg.h1_topk,
                "h1_min_support": cfg.h1_min_support,
                "fused_tau": cfg.fused_tau,
                "merge_gap": cfg.merge_gap,
                "phase_info": phase_info,
            }
            if phase_info.get("enable_pca", True):
                pca = PCA(n_components=0.9, svd_solver="full").fit(Xn)
                dump(pca, os.path.join(ART_DIR, f"{save_prefix}_pca.joblib"))
                manifest["phase_info"]["enable_pca"] = True
            else:
                manifest["phase_info"]["enable_pca"] = False
            with open(os.path.join(ART_DIR, f"{save_prefix}_manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            diag = F.copy(); diag["Regime"] = regimes
            if not SKIP_DIAG:
                diag.to_csv(os.path.join(ART_DIR, f"{save_prefix}_train_diagnostics.csv"))

        with Timer("SUMMARY"):
            print(f"Tags: {len(tags)} | Feature rows: {len(F)} | Regimes: {regime_count} | Switches: {regime_switches}")

        run_info = {"ok": True, "tags": tags, "rows": len(F), "regimes": regime_count, "phase": phase_info.get("phase", "P3")}
        return run_info
    except Exception as exc:
        status = "error"
        err_msg = str(exc)
        raise
    finally:
        latency_s = time.perf_counter() - started
        guard_events = acm_observe.load_guardrail_events(ART_DIR)
        guard_state = acm_observe.guardrail_state(guard_events)
        run_summary = {
            "run_id": RUN_ID,
            "ts_utc": _now_iso(),
            "equip": equip_name,
            "cmd": "train",
            "rows_in": rows_in,
            "tags": len(tags),
            "feat_rows": feat_rows,
            "regimes": regime_count,
            "events": 0,
            "data_span_min": round(data_span_min, 3),
            "phase": phase_info.get("phase", ""),
            "k_selected": regime_count,
            "theta_p95": 0.0,
            "drift_flag": 0,
            "guardrail_state": guard_state,
            "theta_step_pct": 0.0,
            "latency_s": round(latency_s, 3),
            "artifacts_age_min": 0.0,
            "status": status,
            "err_msg": err_msg,
        }
        _write_run_summary(run_summary)

def score_window(csv_path: str, save_prefix="acm", equip: Optional[str]=None) -> Dict:
    equip_name = _equip_name(csv_path, equip)
    started = time.perf_counter()
    status = "ok"
    err_msg = ""
    rows_in = 0
    feat_rows = 0
    events_count = 0
    regime_count = 0
    theta_latest = None
    theta_prev = None
    theta_step = 0.0
    theta_p95 = 0.0
    data_span_min = 0.0
    feature_rank = 0
    run_info: Dict[str, Any] = {"ok": False}
    scored_path = Path(ART_DIR) / f"{save_prefix}_scored_window.csv"
    artifact_age = artifact_age_minutes(scored_path)

    phase_manifest: Dict[str, Any] = {"phase": "P3"}
    phase_effective: Dict[str, Any] = phase_manifest

    try:
        with Timer("LOAD_DATA", {"csv": csv_path}):
            df = pd.read_csv(csv_path)
        with Timer("CLEAN_TIME"):
            df = ensure_time_index(df)

        with open(os.path.join(ART_DIR, f"{save_prefix}_manifest.json")) as f:
            man = json.load(f)
            phase_manifest = man.get("phase_info", {"phase": "P3", "enable_pca": True, "enable_h3": True})

        with Timer("RESAMPLE", {"rule": man["resample_rule"]}):
            df = resample_numeric(df, man["resample_rule"])
            df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not SKIP_DIAG:
                resampled_out = os.path.join(ART_DIR, f"{save_prefix}_resampled.csv")
                df[numeric_cols].to_csv(resampled_out, index=True)
                print(f"[SAVE] Resampled numeric data -> {resampled_out}")
        rows_in = len(df)
        data_span_min = _data_span_minutes(df.index)

        tags = man["tags"]
        write_dq_metrics(df, tags, save_prefix)
        W, S, max_fft = man["window"], man["stride"], man["max_fft_bins"]

        with Timer("FEATURES_BUILD", {"W": W, "S": S, "fft_bins": max_fft}):
            F = build_feature_matrix(df, tags, W, S, max_fft)
            feat_rows = len(F)
            from joblib import load
            scaler = load(os.path.join(ART_DIR, f"{save_prefix}_scaler.joblib"))
            Xn = scaler.transform(F.values)
            feature_rank = int(np.linalg.matrix_rank(np.nan_to_num(F.values))) if feat_rows else 0

        phase_runtime = determine_phase(rows_in, len(tags), data_span_min, feature_rank)
        if PHASE_ORDER.get(phase_runtime["phase"], 0) < PHASE_ORDER.get(phase_manifest.get("phase", "P3"), 0):
            _append_guardrail(
                save_prefix,
                "phase_downgrade",
                "warn",
                f"Runtime phase {phase_runtime['phase']} below manifest {phase_manifest.get('phase','P3')}",
                {
                    "runtime_rows": phase_runtime.get("rows", rows_in),
                    "runtime_reason": phase_runtime.get("reason", ""),
                },
            )
            phase_to_guardrail(save_prefix, phase_runtime)
            phase_effective = phase_runtime
        else:
            phase_effective = phase_manifest

        enable_pca = phase_effective.get("enable_pca", True)
        enable_h3 = phase_effective.get("enable_h3", True) and enable_pca

        with Timer("REGIMES"):
            from joblib import load
            km = load(os.path.join(ART_DIR, f"{save_prefix}_regimes.joblib"))
            regimes = km.predict(Xn)
            regime_count = int(getattr(km, "n_clusters", len(np.unique(regimes))))

        with Timer("H1_SCORE", {"mode": man.get("h1_mode", "lite_ar1"), "roll": man.get("h1_roll", 9)}):
            mode = man.get("h1_mode", "lite_ar1").lower()
            try:
                with open(os.path.join(ART_DIR, f"{save_prefix}_h1_ar1.json")) as f:
                    ar1_coeffs = json.load(f)
            except FileNotFoundError:
                ar1_coeffs = {}
            s_h1 = h1_score_fast(df, tags, mode, man.get("h1_roll", 9), man.get("h1_robust", True), ar1_coeffs)
            s_h1 = s_h1.reindex(F.index).fillna(method="ffill").fillna(0)

        if enable_pca and Path(ART_DIR, f"{save_prefix}_pca.joblib").exists():
            with Timer("H2_SCORE"):
                from joblib import load
                pca = load(os.path.join(ART_DIR, f"{save_prefix}_pca.joblib"))
                Xr = pca.inverse_transform(pca.transform(Xn))
                err = ((Xn - Xr)**2).sum(axis=1)
                s_h2 = pd.Series(err, index=F.index)
                p95 = s_h2.quantile(0.95) or 1.0
                s_h2 = (s_h2 / (p95 + 1e-9)).clip(0, 1)
        else:
            pca = None
            s_h2 = pd.Series(0.0, index=F.index)

        if enable_h3 and pca is not None:
            with Timer("H3_SCORE"):
                Emb = pca.transform(Xn)
                from numpy.linalg import norm
                scores = np.zeros(len(Emb))
                for i in range(len(Emb)):
                    j0 = max(0, i - 50)
                    ref = Emb[j0:i]
                    if len(ref) < 10:
                        scores[i] = 0.0
                        continue
                    mu = ref.mean(axis=0)
                    cos = float((Emb[i] * mu).sum() / (norm(Emb[i]) * norm(mu) + 1e-9))
                    scores[i] = max(0.0, 1.0 - cos)
                s_h3 = pd.Series(scores, index=F.index)
        else:
            s_h3 = pd.Series(0.0, index=F.index)

        with Timer("CONTEXT_MASKS", {"slope_thr": man.get("slope_thr", 0.75), "accel_thr": man.get("accel_thr", 1.25)}):
            mask = detect_transients(df, tags, slope_thr=man.get("slope_thr", 0.75), accel_thr=man.get("accel_thr", 1.25))
            mask = mask.reindex(F.index).fillna(0)

        with Timer("CORROBORATION"):
            corrb = corroboration_boost(df.reindex(F.index, method="nearest"), tags, pairs=man.get("corroboration_pairs", 20))

        with Timer("CHANGEPOINT"):
            cpd = change_point_signal(df.reindex(F.index, method="nearest"), tags, W=60)

        with Timer("FUSION"):
            components = [0.45 * s_h1.values]
            components.append(0.35 * s_h2.values if enable_pca else 0.0)
            components.append(0.35 * s_h3.values if enable_h3 else 0.0)
            base = np.maximum.reduce(components)
            boost = 0.15 * corrb.values + 0.10 * cpd.values
            fused = np.clip(base + boost, 0, 1)
            fused = np.where(mask.values > 0, fused * 0.7, fused)
        fused_series = pd.Series(fused, index=F.index, name="FusedScore")

        theta_prev = load_previous_theta(save_prefix)
        theta_series = dynamic_threshold(fused_series)
        if not theta_series.empty:
            theta_latest = float(theta_series.iloc[-1])
            theta_p95 = float(theta_series.quantile(0.95))
            theta_step = theta_step_pct(theta_prev, theta_latest)
            if abs(theta_step) >= GUARDRAIL_WARN_STEP_PCT:
                level = "alert" if abs(theta_step) >= GUARDRAIL_ALERT_STEP_PCT else "warn"
                message = f"theta shift {theta_step:.1f}% (prev={theta_prev}, curr={theta_latest})"
                _append_guardrail(save_prefix, "theta_step", level, message, {"theta_prev": theta_prev, "theta_curr": theta_latest})
        save_thresholds(theta_series, save_prefix)

        with Timer("EPISODES", {"merge_gap": man.get("merge_gap", 3)}):
            episodes = build_episodes(fused_series, theta_series, man.get("merge_gap", 3))
            events_count = len(episodes)

        with Timer("EXPORT"):
            out = F.copy()
            out["Regime"] = regimes
            out["H1_Forecast"] = s_h1.values
            out["H2_Recon"] = s_h2.values
            out["H3_Contrast"] = s_h3.values
            out["CorrBoost"] = corrb.values
            out["CPD"] = cpd.values
            out["ContextMask"] = mask.values
            out["FusedScore"] = fused
            theta_aligned = theta_series.reindex(F.index)
            if theta_aligned.isna().any():
                if not theta_series.empty:
                    theta_aligned = theta_aligned.fillna(method="ffill").fillna(method="bfill")
                else:
                    theta_aligned = pd.Series(np.zeros(len(F)), index=F.index)
            out["Theta"] = theta_aligned.values
            out.to_csv(scored_path)

            events_df = pd.DataFrame([{"Start": e["start"], "End": e["end"], "PeakScore": e["peak"]} for e in episodes])
            events_df.to_csv(os.path.join(ART_DIR, f"{save_prefix}_events.csv"), index=False)
            events_df.to_csv(os.path.join(ART_DIR, "events.csv"), index=False)
            pd.DataFrame({"Ts": F.index, "Mask": mask.values}).to_csv(
                os.path.join(ART_DIR, f"{save_prefix}_context_masks.csv"), index=False)
            fused_export = fused_series.reset_index().rename(columns={"index": "Ts"})
            fused_export.to_csv(os.path.join(ART_DIR, f"{save_prefix}_scores.csv"), index=False)
            fused_export.to_csv(os.path.join(ART_DIR, "scores.csv"), index=False)

        with Timer("SUMMARY"):
            print(f"Rows: {len(F):,} | Regimes seen: {len(np.unique(regimes))} | Events: {events_count}")

        run_info = {
            "ok": True,
            "rows": int(len(F)),
            "events": events_count,
            "theta_p95": theta_p95,
            "phase": phase_effective.get("phase", "P3"),
        }
        return run_info
    except Exception as exc:
        status = "error"
        err_msg = str(exc)
        raise
    finally:
        latency_s = time.perf_counter() - started
        guard_events = acm_observe.load_guardrail_events(ART_DIR)
        guard_state = acm_observe.guardrail_state(guard_events)
        tag_count = len(man.get("tags", [])) if "man" in locals() else 0
        run_summary = {
            "run_id": RUN_ID,
            "ts_utc": _now_iso(),
            "equip": equip_name,
            "cmd": "score",
            "rows_in": rows_in,
            "tags": tag_count,
            "feat_rows": feat_rows,
            "regimes": regime_count,
            "events": events_count,
            "data_span_min": round(data_span_min, 3),
            "phase": phase_effective.get("phase", ""),
            "k_selected": regime_count,
            "theta_p95": round(theta_p95, 6) if theta_p95 else 0.0,
            "drift_flag": 0,
            "guardrail_state": guard_state,
            "theta_step_pct": round(theta_step, 3),
            "latency_s": round(latency_s, 3),
            "artifacts_age_min": round(artifact_age, 3),
            "status": status,
            "err_msg": err_msg,
        }
        _write_run_summary(run_summary)

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
        if t not in df.columns: continue
        mu, sigma = float(base.loc[t,"mu"]), float(base.loc[t,"sigma"])
        z = 0.0 if sigma <= 1e-9 else float(abs(df[t].mean() - mu) / sigma)
        rows.append({"Tag": t, "DriftZ": z})
    res = pd.DataFrame(rows).sort_values("DriftZ", ascending=False)
    res.to_csv(os.path.join(ART_DIR, f"{save_prefix}_drift.csv"), index=False)
    print(res.head(15).to_string(index=False))
    return res

# ------------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("ACM Core (H1: lite+AR1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train"); tr.add_argument("--csv", required=True); tr.add_argument("--equip")
    sc = sub.add_parser("score"); sc.add_argument("--csv", required=True); sc.add_argument("--equip")
    dr = sub.add_parser("drift"); dr.add_argument("--csv", required=True)
    args = p.parse_args()
    with Timer("START"): pass
    try:
        if args.cmd == "train":
            info = train_core(args.csv, equip=getattr(args, "equip", None))
            print(json.dumps(info, indent=2))
        elif args.cmd == "score":
            info = score_window(args.csv, equip=getattr(args, "equip", None))
            print(json.dumps(info, indent=2))
        elif args.cmd == "drift":
            drift_check(args.csv)
    finally:
        with Timer("END"): pass

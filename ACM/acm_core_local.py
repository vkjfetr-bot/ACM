#!/usr/bin/env python3
# acm_core_local.py
# Changelog
# 2025-10-07: Local-files core: contextual baselines, regimes, IF ensemble, drift, spans; CSV "table" helpers (no SQL).

import os, math, json, numpy as np, pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# ---------- Paths ----------
OUT_DIR = os.getenv("ACM_OUT_DIR", "./acm_out")
os.makedirs(OUT_DIR, exist_ok=True)
TBL_ARTIFACT     = os.path.join(OUT_DIR, "acm_artifact.csv")
TBL_RUN_SUMMARY  = os.path.join(OUT_DIR, "acm_run_summary.csv")
TBL_EVENT        = os.path.join(OUT_DIR, "acm_event.csv")
TBL_ASSET_HEALTH = os.path.join(OUT_DIR, "acm_asset_health.csv")

# ---------- Data guards ----------
def ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Ts" in df.columns:
        df = df.copy(); df["Ts"] = pd.to_datetime(df["Ts"], errors="coerce")
        df = df.dropna(subset=["Ts"]).sort_values("Ts").set_index("Ts")
        return df
    if isinstance(df.index, pd.DatetimeIndex): return df.sort_index()
    raise ValueError("Data must have 'Ts' column or a DatetimeIndex.")

def slice_time(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start: df = df[df.index >= pd.to_datetime(start)]
    if end:   df = df[df.index <= pd.to_datetime(end)]
    if df.empty: raise ValueError("No data after time filtering.")
    return df

def resample_guard(df: pd.DataFrame, max_ffill_mult: float = 5.0) -> pd.DataFrame:
    if len(df.index) < 3: return df
    dt = (df.index.to_series().diff().dropna()).dt.total_seconds()
    if dt.empty: return df
    step = float(dt.mode().iloc[0]) if not dt.mode().empty else float(np.nanmedian(dt))
    if not np.isfinite(step) or step <= 0: return df
    rule = f"{int(round(step))}S"
    df2 = df.resample(rule).mean()
    df2 = df2.ffill(limit=int(max(1, round(max_ffill_mult)))).bfill(limit=1)
    return df2

def enforce_tags(df: pd.DataFrame, tags: List[str]) -> List[str]:
    missing = [t for t in tags if t not in df.columns]
    if missing: raise ValueError(f"Missing tags in data: {missing}")
    return tags

# ---------- Rolling features ----------
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

# ---------- Baselines ----------
class Baseline:
    def __init__(self, kind: str, med: Dict[str, any], mad: Dict[str, any], p01: Dict[str, any], p99: Dict[str, any]):
        self.kind, self.med, self.mad, self.p01, self.p99 = kind, med, mad, p01, p99

def _mad(x: pd.Series):
    m = float(np.nanmedian(x)); mads = float(np.nanmedian(np.abs(x - m)))
    return m, (mads if mads else np.nan)

def build_baseline(df_hist: pd.DataFrame, tags: List[str], kind="hour") -> Baseline:
    dfh = ensure_time_index(df_hist).copy()
    if kind == "global":
        med = {}; mad = {}; p01 = {}; p99 = {}
        for t in tags:
            s = pd.to_numeric(dfh[t], errors="coerce")
            m, mads = _mad(s); med[t], mad[t] = m, mads
            q = s.quantile([0.01,0.99]); p01[t], p99[t] = float(q.iloc[0]), float(q.iloc[1])
        return Baseline("global", med, mad, p01, p99)
    if kind == "hour":   dfh["ctx"] = dfh.index.hour
    elif kind == "weekday": dfh["ctx"] = dfh.index.weekday
    else: raise ValueError("baseline kind must be global|hour|weekday")
    med = {}; mad = {}; p01 = {}; p99 = {}
    for t in tags:
        g = dfh.groupby("ctx")[t]
        med[t] = g.median()
        mad[t] = g.apply(lambda x: float(np.nanmedian(np.abs(x - np.nanmedian(x)))) or np.nan)
        p01[t] = g.quantile(0.01); p99[t] = g.quantile(0.99)
    return Baseline(kind, med, mad, p01, p99)

def z_hist_from_base(s: pd.Series, base: Baseline, tag: str, idx: pd.DatetimeIndex) -> pd.Series:
    if base.kind == "global":
        M = base.mad[tag] if base.mad[tag] not in (0, None) else np.nan
        return (s - base.med[tag]) / M
    ctx = (idx.hour if base.kind == "hour" else idx.weekday)
    med = pd.Series(ctx, index=idx).map(base.med[tag])
    mad = pd.Series(ctx, index=idx).map(base.mad[tag]).replace(0, np.nan)
    return (s - med) / mad

# ---------- Regimes ----------
class RegimeModel:
    def __init__(self, feat_cols: List[str], centers: np.ndarray):
        self.feat_cols, self.centers = feat_cols, centers

def train_regimes(df_hist: pd.DataFrame, tags: List[str], k: int = 4, seed=42) -> RegimeModel:
    Fh = build_features(df_hist, tags).dropna()
    k_eff = 1 if len(Fh) < 200 else k
    km = KMeans(n_clusters=k_eff, n_init="auto", random_state=seed).fit(Fh)
    return RegimeModel(list(Fh.columns), km.cluster_centers_)

def assign_state(F: pd.DataFrame, model: RegimeModel) -> pd.Series:
    X = F[model.feat_cols].fillna(method="ffill").fillna(method="bfill").to_numpy()
    d = np.linalg.norm(X[:, None, :] - model.centers[None, :, :], axis=2)
    lab = np.argmin(d, axis=1)
    return pd.Series(lab, index=F.index, name="state")

def state_distance(F: pd.DataFrame, model: RegimeModel) -> pd.Series:
    X = F[model.feat_cols].fillna(method="ffill").fillna(method="bfill").to_numpy()
    d = np.linalg.norm(X[:, None, :] - model.centers[None, :, :], axis=2)
    mind = np.min(d, axis=1)
    return pd.Series(mind, index=F.index, name="state_dist")

# ---------- Drift ----------
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
        PH = max(0.0, PH + v - m_t - delta); M_t = max(M_t, PH)
        if (M_t - PH) > lamb: return True
    return False

# ---------- IF + Ensemble ----------
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
        parts.append(z_hist_from_base(s, base, t, df.index).abs().rename(f"{t}_|z_hist|"))
    parts.append(iso_s); parts.append(state_d)
    S = pd.concat(parts, axis=1)
    def rscale(col: pd.Series):
        q1, q99 = col.quantile(0.01), col.quantile(0.99)
        return (col - q1) / (q99 - q1 + 1e-9)
    S = S.apply(rscale)
    return S.mean(axis=1).rename("anom_score")

def quantile_cut(series: pd.Series, q: float):
    return series.quantile(q) if series.notna().any() else np.nan

# ---------- Intervals & health ----------
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
    merged = []
    for s, e in spans:
        if merged and (s - merged[-1][1]).total_seconds() <= merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s).total_seconds() >= min_len_s]

def health_index(score: pd.Series, spans: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    if score.empty: return 100.0
    q95 = float(score.quantile(0.95)) if score.notna().any() else 0.0
    density = len(spans) / max(1, len(score) / 60.0)
    raw = 0.6*q95 + 0.4*density
    hi = 100.0 - 100.0 * (raw / (raw + 1.0))
    return max(0.0, min(100.0, hi))

def health_label(hi: float, prev_label: Optional[str] = None) -> str:
    if hi >= 80: return "Good" if prev_label != "Alert" else ("Watch" if hi < 85 else "Good")
    if hi >= 60: return "Watch" if prev_label != "Good" else ("Good" if hi > 75 else "Watch")
    return "Alert"

# ---------- CSV "table" helpers ----------
def _read_table(path: str, dtypes: dict = None) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path, dtype=dtypes)

def _write_table(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def _append_table(row_dict: dict, path: str):
    df = _read_table(path)
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    _write_table(df, path)

def artifact_insert(equipment_id: str, kind: str, version: int, trained_from, trained_to, params: dict, active=True):
    df = _read_table(TBL_ARTIFACT)
    if active:
        if not df.empty:
            mask = (df["EquipmentId"]==equipment_id) & (df["ArtifactType"]==kind)
            df.loc[mask, "Active"] = 0
    new_row = {
        "EquipmentId": equipment_id,
        "ArtifactType": kind,
        "Version": version,
        "TrainedFrom": pd.to_datetime(trained_from),
        "TrainedTo": pd.to_datetime(trained_to),
        "ParamsJSON": json.dumps(params, ensure_ascii=False),
        "Active": 1 if active else 0,
        "CreatedAt": pd.Timestamp.utcnow()
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _write_table(df, TBL_ARTIFACT)

def artifacts_get_active(equipment_id: str) -> Dict[str, dict]:
    df = _read_table(TBL_ARTIFACT)
    out = {}
    if df.empty: return out
    mask = (df["EquipmentId"]==equipment_id) & (df["Active"]==1)
    for _, r in df.loc[mask].iterrows():
        out[r["ArtifactType"]] = {"version": int(r["Version"]), "params": json.loads(r["ParamsJSON"])}
    return out

def run_summary_insert(equipment_id: str, window_start, window_end, rows_proc: int, tags_used: int,
                       anomaly_frac: float, hi: float, hlabel: str, drift_json: dict,
                       baseline_ver: int, regime_ver: int) -> int:
    df = _read_table(TBL_RUN_SUMMARY)
    run_id = (int(df["RunId"].max()) + 1) if ("RunId" in df.columns and not df.empty) else 1
    row = {
        "RunId": run_id,
        "EquipmentId": equipment_id,
        "WindowStart": pd.to_datetime(window_start),
        "WindowEnd": pd.to_datetime(window_end),
        "RowsProcessed": rows_proc,
        "TagsUsed": tags_used,
        "AnomalyFraction": anomaly_frac,
        "HealthIndex": hi,
        "HealthLabel": hlabel,
        "DriftJSON": json.dumps(drift_json, ensure_ascii=False),
        "BaselineVersion": baseline_ver,
        "RegimeVersion": regime_ver,
        "CreatedAt": pd.Timestamp.utcnow()
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write_table(df, TBL_RUN_SUMMARY)
    return run_id

def events_insert(run_id: int, equipment_id: str, spans: List[Tuple[pd.Timestamp, pd.Timestamp]],
                  score: pd.Series, states: pd.Series, top_tags: List[str]):
    if not spans: return
    df = _read_table(TBL_EVENT)
    next_id = (int(df["EventId"].max()) + 1) if ("EventId" in df.columns and not df.empty) else 1
    rows = []
    for (s, e) in spans:
        peak = float(score.loc[s:e].max())
        mode_state = int(states.loc[s:e].mode().iloc[0]) if len(states.loc[s:e]) else None
        severity = "HIGH" if peak > 0.9 else ("MED" if peak > 0.7 else "LOW")
        rows.append({
            "EventId": next_id, "RunId": run_id, "EquipmentId": equipment_id,
            "StartTime": pd.to_datetime(s), "EndTime": pd.to_datetime(e),
            "PeakScore": peak, "ModeState": mode_state,
            "ContribTopTags": ",".join(top_tags[:3]),
            "ContribNotes": "", "Severity": severity, "Status": "Open"
        })
        next_id += 1
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    _write_table(df, TBL_EVENT)

def asset_health_upsert(equipment_id: str, window_start, window_end, hi: float, hlabel: str,
                        anomaly_frac: float, baseline_ver: int, regime_ver: int, last_event_peak: float = None, drift_flag: bool = False):
    df = _read_table(TBL_ASSET_HEALTH)
    row = {
        "EquipmentId": equipment_id,
        "AsOfTime": pd.to_datetime(window_end),
        "WindowStart": pd.to_datetime(window_start),
        "WindowEnd": pd.to_datetime(window_end),
        "HealthIndex": hi,
        "HealthLabel": hlabel,
        "AnomalyFraction": anomaly_frac,
        "EventCount": None,
        "DominantState": None,
        "DriftFlag": 1 if drift_flag else 0,
        "DataQualityScore": None,
        "BaselineVersion": baseline_ver,
        "RegimeVersion": regime_ver,
        "LastEventPeak": last_event_peak,
        "UpdatedAt": pd.Timestamp.utcnow()
    }
    # upsert by EquipmentId (keep last)
    df = df[df["EquipmentId"] != equipment_id] if not df.empty else df
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write_table(df, TBL_ASSET_HEALTH)

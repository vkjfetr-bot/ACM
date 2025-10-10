"""Consolidated ACM pipeline core utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import acm_observe

try:  # Optional dependencies
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    import ruptures as rpt  # type: ignore
except Exception:  # pragma: no cover
    rpt = None

try:
    from hdbscan import HDBSCAN  # type: ignore
except Exception:  # pragma: no cover
    HDBSCAN = None

try:
    from scipy.stats import genpareto  # type: ignore
except Exception:  # pragma: no cover
    genpareto = None


MAJOR_VERSION = 3
MINOR_VERSION = 0
PATCH_VERSION = 0
__version__ = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "sampling": {"period": "10s"},
    "features": {"window": "60s", "step": "10s", "spectral_bands": [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]]},
    "pca": {"variance": 0.95},
    "segmentation": {"model": "rbf", "min_duration_s": 60},
    "clustering": {"algo": "kmeans_auto", "k_range": [2, 8], "min_cluster_minutes": 3},
    "hmm": {
        "min_state_seconds": 60,
        "allowed_transitions": [["Idle", "Start"], ["Start", "Run"], ["Run", "Stop"], ["Stop", "Idle"]],
    },
    "detectors": {"per_regime": "iforest", "fusion_weights": [0.5, 0.3, 0.2]},
    "thresholds": {"method": "EVT", "alpha": 0.001, "fallback_quantile": 0.995},
    "eventization": {"min_len": "30s", "merge_gap": "120s"},
    "drift": {"psi_regime_warn": 0.25, "psi_regime_alert": 0.5, "adwin_delta": 0.002},
    "report": {"key_tags": []},
}


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "acm_config.yaml"


@dataclass
class PipelineConfig:
    raw: Dict[str, Any]
    source_path: Optional[Path] = None

    def get(self, dotted: str, default: Optional[Any] = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    cfg_path = path or default_config_path()
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        text = cfg_path.read_text(encoding="utf-8")
        if yaml is not None:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged.update(data)
    return PipelineConfig(raw=merged, source_path=cfg_path if cfg_path.exists() else None)


# ---------------------------------------------------------------------------
# Artifact management
# ---------------------------------------------------------------------------


@dataclass
class ArtifactPaths:
    root: Path
    equip_root: Path
    run_dir: Path
    model_dir: Path
    images_dir: Path
    run_id: str


@dataclass
class ArtifactManager:
    art_root: Path
    equip: str
    run_type: str
    run_paths: Optional[ArtifactPaths] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.art_root.mkdir(parents=True, exist_ok=True)
        equip_root = self.art_root / self.equip
        equip_root.mkdir(parents=True, exist_ok=True)
        model_dir = equip_root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self._equip_root = equip_root
        self._model_dir = model_dir

    def start_run(self, timestamp: Optional[datetime] = None) -> ArtifactPaths:
        ts = (timestamp or datetime.utcnow()).strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.run_type}_{ts}"
        run_dir = self._equip_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        images_dir = run_dir / "imgs"
        images_dir.mkdir(parents=True, exist_ok=True)
        self.run_paths = ArtifactPaths(
            root=self.art_root,
            equip_root=self._equip_root,
            run_dir=run_dir,
            model_dir=self._model_dir,
            images_dir=images_dir,
            run_id=run_id,
        )
        return self.run_paths

    def write_json(self, data: Dict[str, Any], name: str) -> Path:
        path = self._ensure_run_dir() / name
        path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return path

    def write_table(self, df: pd.DataFrame, name: str, *, prefer_parquet: bool = True) -> Path:
        path = self._ensure_run_dir() / name
        if prefer_parquet and name.endswith(".parquet"):
            try:
                df.to_parquet(path, index=True)
                return path
            except Exception:
                csv_path = path.with_suffix(".csv")
                df.to_csv(csv_path, index=True)
                return csv_path
        df.to_csv(path, index=True)
        return path

    def write_payloads(self) -> None:
        acm_observe.write_payloads(str(self._equip_root))

    def emit_run_summary(self, row: Dict[str, Any]) -> None:
        acm_observe.write_run_summary(str(self._equip_root), row)
        acm_observe.write_run_health(str(self._equip_root), row)
        # TODO: Replace with usp_WriteRunSummary when SQL persistence is wired.

    def mark_success(self, extra: Optional[Dict[str, Any]] = None) -> None:
        self.metadata["status"] = "ok"
        if extra:
            self.metadata.update(extra)
        self._write_metadata()

    def mark_failure(self, err: Exception) -> None:
        self.metadata["status"] = "error"
        self.metadata["error"] = str(err)
        self._write_metadata()

    def latest_model_path(self, name: str) -> Path:
        return self._model_dir / name

    def archive_model(self, source: Path, name: str) -> Path:
        target = self._model_dir / name
        if source != target:
            target.write_bytes(source.read_bytes())
        return target

    def _ensure_run_dir(self) -> Path:
        if not self.run_paths:
            raise RuntimeError("Run directory not initialised; call start_run() first.")
        return self.run_paths.run_dir

    def _write_metadata(self) -> None:
        if not self.run_paths:
            return
        meta = {**self.metadata, "run_id": self.run_paths.run_id, "equip": self.equip, "run_type": self.run_type}
        (self.run_paths.run_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_raw_data(
    source: Path,
    *,
    equip: str,
    t0: Optional[datetime] = None,
    t1: Optional[datetime] = None,
) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"CSV source not found: {source}")
    df = pd.read_csv(source)
    # TODO: Replace with usp_GetEquipmentWindow(@equip, @t0, @t1) when SQL wiring is ready.
    if "Ts" in df.columns:
        df["Ts"] = pd.to_datetime(df["Ts"])
        df = df.set_index("Ts")
    else:
        df.index = pd.to_datetime(df.index)
    if t0:
        df = df[df.index >= t0]
    if t1:
        df = df[df.index <= t1]
    return df.sort_index()


def resample_and_clean(df: pd.DataFrame, *, resample_rule: str, clamp_sigma: float = 6.0) -> Tuple[pd.DataFrame, List[str]]:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.resample(resample_rule).mean()
    numeric_df = numeric_df.interpolate(method="time").ffill().bfill()
    mu = numeric_df.rolling(window=15, min_periods=1).mean()
    sigma = numeric_df.rolling(window=15, min_periods=1).std().fillna(0)
    upper = mu + clamp_sigma * sigma
    lower = mu - clamp_sigma * sigma
    numeric_df = numeric_df.clip(lower=lower, upper=upper)
    return numeric_df, numeric_df.columns.tolist()


def compute_dq_metrics(df: pd.DataFrame, tags: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for tag in tags:
        s = pd.to_numeric(df.get(tag), errors="coerce")
        total = len(s)
        if total == 0:
            rows.append({"Tag": tag, "flatline_pct": 0.0, "dropout_pct": 0.0, "nan_pct": 0.0, "presence_ratio": 0.0, "spikes_pct": 0.0})
            continue
        nan_mask = s.isna()
        presence_ratio = float((~nan_mask).sum()) / float(total) if total else 0.0
        nan_pct = float(nan_mask.sum()) / float(total) * 100.0 if total else 0.0
        clean = s.fillna(method="ffill").fillna(method="bfill")
        diffs = clean.diff().abs()
        flatline_pct = float((diffs <= 1e-9).sum()) / max(total - 1, 1) * 100.0
        dropout_pct = float((clean == 0).sum()) / float(total) * 100.0 if total else 0.0
        if clean.std(ddof=0) > 1e-9:
            z = (clean - clean.mean()) / clean.std(ddof=0)
            spikes_pct = float((z.abs() > 4).sum()) / float(total) * 100.0
        else:
            spikes_pct = 0.0
        rows.append(
            {
                "Tag": tag,
                "flatline_pct": round(flatline_pct, 3),
                "dropout_pct": round(dropout_pct, 3),
                "nan_pct": round(nan_pct, 3),
                "presence_ratio": round(presence_ratio, 3),
                "spikes_pct": round(spikes_pct, 3),
            }
        )
    return pd.DataFrame(rows)


def prepare_window(
    source: Path,
    *,
    equip: str,
    resample_rule: str,
    clamp_sigma: float = 6.0,
    t0: Optional[datetime] = None,
    t1: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    raw = load_raw_data(source, equip=equip, t0=t0, t1=t1)
    clean, tags = resample_and_clean(raw, resample_rule=resample_rule, clamp_sigma=clamp_sigma)
    dq = compute_dq_metrics(clean, tags)
    return raw, clean, dq, tags


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    period_seconds: int,
    window_seconds: int,
    step_seconds: int,
    spectral_bands: Iterable[Sequence[float]],
    key_tags: Iterable[str],
) -> pd.DataFrame:
    window = max(int(window_seconds / period_seconds), 1)
    step = max(int(step_seconds / period_seconds), 1)
    if window <= 1:
        raise ValueError("Feature window too small relative to sampling period.")

    indices: List[datetime] = []
    rows: List[Dict[str, float]] = []
    values = df.to_numpy(dtype=float)
    columns = df.columns.tolist()
    times = df.index.to_numpy()

    bands = np.asarray([tuple(band) for band in spectral_bands], dtype=float)
    n_bands = len(bands)
    fs = 1.0 / period_seconds if period_seconds else 1.0

    for start in range(0, len(df) - window + 1, step):
        end = start + window
        window_slice = values[start:end]
        if np.isnan(window_slice).all():
            continue
        indices.append(times[end - 1])
        stats = _window_stats(window_slice)
        trends = _window_trends(window_slice)
        iqrs = _window_iqr(window_slice)
        percentiles = _window_percentiles(window_slice)
        spectral = _window_spectrum(window_slice, fs, bands, n_bands)
        row: Dict[str, float] = {}
        for col_idx, tag in enumerate(columns):
            prefix = f"{tag}"
            row[f"{prefix}_mean"] = stats["mean"][col_idx]
            row[f"{prefix}_std"] = stats["std"][col_idx]
            row[f"{prefix}_min"] = stats["min"][col_idx]
            row[f"{prefix}_max"] = stats["max"][col_idx]
            row[f"{prefix}_slope"] = trends["slope"][col_idx]
            row[f"{prefix}_diff"] = trends["diff"][col_idx]
            row[f"{prefix}_iqr"] = iqrs[col_idx]
            row[f"{prefix}_p10"] = percentiles["p10"][col_idx]
            row[f"{prefix}_p50"] = percentiles["p50"][col_idx]
            row[f"{prefix}_p90"] = percentiles["p90"][col_idx]
            for band_idx in range(n_bands):
                row[f"{prefix}_spec_b{band_idx}"] = spectral[band_idx, col_idx]
        key_indices = [columns.index(tag) for tag in key_tags if tag in columns]
        for idx in key_indices:
            for jdx in range(len(columns)):
                if idx == jdx:
                    continue
                corr = np.corrcoef(window_slice[:, idx], window_slice[:, jdx])[0, 1]
                if np.isfinite(corr):
                    row[f"corr_{columns[jdx]}_to_{columns[idx]}"] = corr
        rows.append(row)

    feature_df = pd.DataFrame(rows, index=pd.to_datetime(indices))
    return feature_df.sort_index()


def _window_stats(window_slice: np.ndarray) -> Dict[str, np.ndarray]:
    mu = np.nanmean(window_slice, axis=0)
    sigma = np.nanstd(window_slice, axis=0)
    mn = np.nanmin(window_slice, axis=0)
    mx = np.nanmax(window_slice, axis=0)
    return {"mean": mu, "std": sigma, "min": mn, "max": mx}


def _window_trends(window_slice: np.ndarray) -> Dict[str, np.ndarray]:
    diffs = np.nanmean(np.diff(window_slice, axis=0), axis=0)
    idx = np.arange(window_slice.shape[0])
    slopes = []
    for col in range(window_slice.shape[1]):
        y = window_slice[:, col]
        if np.all(np.isnan(y)):
            slopes.append(0.0)
            continue
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            slopes.append(0.0)
            continue
        coeffs = np.polyfit(idx[mask], y[mask], 1)
        slopes.append(coeffs[0])
    return {"diff": np.asarray(diffs), "slope": np.asarray(slopes)}


def _window_iqr(window_slice: np.ndarray) -> np.ndarray:
    q75 = np.nanpercentile(window_slice, 75, axis=0)
    q25 = np.nanpercentile(window_slice, 25, axis=0)
    return q75 - q25


def _window_percentiles(window_slice: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "p10": np.nanpercentile(window_slice, 10, axis=0),
        "p50": np.nanpercentile(window_slice, 50, axis=0),
        "p90": np.nanpercentile(window_slice, 90, axis=0),
    }


def _window_spectrum(
    window_slice: np.ndarray,
    fs: float,
    bands: np.ndarray,
    n_bands: int,
) -> np.ndarray:
    if window_slice.size == 0:
        return np.zeros((n_bands, window_slice.shape[1]))
    fft_vals = np.fft.rfft(window_slice - np.nanmean(window_slice, axis=0), axis=0)
    freqs = np.fft.rfftfreq(window_slice.shape[0], d=1 / fs)
    power = np.abs(fft_vals) ** 2
    band_energy = np.zeros((n_bands, window_slice.shape[1]))
    for idx, (low, high) in enumerate(bands):
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            band_energy[idx] = power[mask].sum(axis=0)
    col_max = band_energy.max(axis=0)
    col_max[col_max == 0] = 1.0
    band_energy /= col_max
    return band_energy


# ---------------------------------------------------------------------------
# Regime discovery
# ---------------------------------------------------------------------------


def fit_pca(X: np.ndarray, variance: float) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=variance, svd_solver="full")
    latent = pca.fit_transform(X)
    return pca, latent


def segment_latent(latent: np.ndarray, min_size: int) -> List[Tuple[int, int]]:
    n = latent.shape[0]
    if n == 0:
        return []
    if rpt is None or n < min_size * 2:
        segments = []
        for start in range(0, n, min_size):
            end = min(start + min_size, n)
            segments.append((start, end))
        return segments
    model = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
    algo = model.fit(latent)
    bkps = algo.predict(pen=3)
    segments = []
    prev = 0
    for bkp in bkps:
        segments.append((prev, bkp))
        prev = bkp
    return segments


def cluster_segments(
    latent: np.ndarray,
    segments: List[Tuple[int, int]],
    *,
    algo: str,
    k_range: Tuple[int, int],
    min_cluster_size: int,
) -> Tuple[np.ndarray, object, str]:
    if not segments:
        raise ValueError("No segments provided for clustering.")
    centroids = []
    for start, end in segments:
        seg = latent[start:end]
        if len(seg) == 0:
            continue
        centroids.append(seg.mean(axis=0))
    centroid_arr = np.vstack(centroids)
    if algo == "hdbscan" and HDBSCAN is not None:
        clusterer = HDBSCAN(min_cluster_size=max(min_cluster_size, 2))
        labels = clusterer.fit_predict(centroid_arr)
        return labels, clusterer, "hdbscan"
    k_min, k_max = k_range
    best_score = -1.0
    best_labels = None
    best_model: Optional[KMeans] = None
    for k in range(max(2, k_min), min(k_max, len(centroid_arr)) + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(centroid_arr)
        if len(set(labels)) <= 1:
            continue
        try:
            score = silhouette_score(centroid_arr, labels, metric="euclidean")
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_labels = labels
            best_model = km
    if best_labels is None or best_model is None:
        k = max(2, min(k_range[1], len(centroid_arr)))
        best_model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        best_labels = best_model.fit_predict(centroid_arr)
    return best_labels, best_model, "kmeans"


def expand_labels_to_samples(segments: List[Tuple[int, int]], labels: np.ndarray, n_samples: int) -> np.ndarray:
    out = np.zeros(n_samples, dtype=int)
    for (start, end), label in zip(segments, labels):
        out[start:end] = label
    return out


def smooth_labels(labels: np.ndarray, min_duration: int) -> np.ndarray:
    if len(labels) == 0:
        return labels
    smoothed = labels.copy()
    start = 0
    current = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label != current:
            length = idx - start
            if length < min_duration:
                smoothed[start:idx] = current
            start = idx
            current = label
    return smoothed


# ---------------------------------------------------------------------------
# Detection & scoring
# ---------------------------------------------------------------------------


def train_regime_detectors(
    X: np.ndarray,
    regimes: np.ndarray,
) -> Dict[int, IsolationForest]:
    models: Dict[int, IsolationForest] = {}
    for regime in np.unique(regimes):
        mask = regimes == regime
        if mask.sum() < 25:
            continue
        subset = X[mask]
        model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=42,
        )
        model.fit(subset)
        models[int(regime)] = model
    return models


def score_regime_detectors(
    X: np.ndarray,
    regimes: np.ndarray,
    models: Dict[int, IsolationForest],
) -> np.ndarray:
    scores = np.ones(X.shape[0])
    for regime, model in models.items():
        mask = regimes == regime
        if not mask.any():
            continue
        subset = X[mask]
        preds = -model.score_samples(subset)
        scores[mask] = _normalise(preds)
    return scores


def sequence_score(latent: np.ndarray, window: int = 30) -> np.ndarray:
    if latent.shape[0] < window * 2:
        return np.zeros(latent.shape[0])
    errs = np.zeros(latent.shape[0])
    for idx in range(window, latent.shape[0]):
        history = latent[idx - window : idx]
        target = latent[idx]
        mu = history.mean(axis=0)
        errs[idx] = np.linalg.norm(target - mu)
    return _normalise(errs)


def window_score(series: np.ndarray, window: int = 30) -> np.ndarray:
    if len(series) == 0:
        return series
    win = pd.Series(series).rolling(window=window, min_periods=1).mean().values
    return _normalise(win)


def fuse_scores(
    pt_score: np.ndarray,
    win_score: np.ndarray,
    seq_score: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    w1, w2, w3 = weights
    fused = w1 * pt_score + w2 * win_score + w3 * seq_score
    return np.clip(fused, 0.0, 1.0)


def calibrate_thresholds(
    fused: np.ndarray,
    regimes: np.ndarray,
    *,
    alpha: float,
    fallback_quantile: float,
) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    series = pd.Series(fused)
    for regime in np.unique(regimes):
        mask = regimes == regime
        data = series[mask]
        if len(data) < 20:
            thresholds[int(regime)] = float(data.quantile(fallback_quantile)) if len(data) else 0.95
            continue
        tail = data.nlargest(max(int(len(data) * 0.05), 10))
        if genpareto is not None and len(tail) > 10:
            thresholds[int(regime)] = _gpd_threshold(tail.values, alpha) or float(data.quantile(fallback_quantile))
        else:
            thresholds[int(regime)] = float(data.quantile(fallback_quantile))
    return thresholds


def _gpd_threshold(samples: np.ndarray, alpha: float) -> float:
    try:
        params = genpareto.fit(samples - samples.min())
        c, loc, scale = params
        q = genpareto.ppf(1 - alpha, c, loc=loc, scale=scale)
        return float(samples.min() + q)
    except Exception:
        return float(np.quantile(samples, 0.99))


def _normalise(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=np.nanmax(data[np.isfinite(data)]) if np.isfinite(data).any() else 1.0)
    mn = np.nanmin(data)
    mx = np.nanmax(data)
    if mx - mn < 1e-9:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Event labelling
# ---------------------------------------------------------------------------


def label_events(
    fused: pd.Series,
    thresholds: Dict[int, float],
    regimes: np.ndarray,
    clean_df: pd.DataFrame,
    *,
    min_len_seconds: int,
    merge_gap_seconds: int,
) -> pd.DataFrame:
    alerts: List[Tuple[int, int, int]] = []
    run_start = None
    last_idx = None
    times = fused.index
    for idx, score in enumerate(fused.values):
        regime = int(regimes[idx]) if idx < len(regimes) else 0
        tau = thresholds.get(regime, 0.95)
        if score >= tau:
            if run_start is None:
                run_start = idx
            last_idx = idx
        else:
            if run_start is not None and last_idx is not None:
                alerts.append((run_start, last_idx, regime))
                run_start = None
    if run_start is not None and last_idx is not None:
        alerts.append((run_start, last_idx, int(regimes[last_idx])))

    events: List[Dict[str, Any]] = []
    prev_end: Optional[pd.Timestamp] = None
    for start_idx, end_idx, regime in alerts:
        start_ts = times[start_idx]
        end_ts = times[end_idx]
        duration = (end_ts - start_ts).total_seconds()
        if duration < min_len_seconds:
            continue
        if prev_end and (start_ts - prev_end).total_seconds() < merge_gap_seconds and events:
            events[-1]["End"] = end_ts
            events[-1]["PeakScore"] = max(events[-1]["PeakScore"], float(fused.iloc[start_idx : end_idx + 1].max()))
            events[-1]["DurationMin"] = (events[-1]["End"] - events[-1]["Start"]).total_seconds() / 60.0
            continue
        window_scores = fused.iloc[start_idx : end_idx + 1]
        clean_slice = clean_df.loc[window_scores.index]
        z_scores = (clean_slice - clean_slice.mean()) / (clean_slice.std(ddof=0) + 1e-6)
        top_tags = (
            z_scores.abs().mean().sort_values(ascending=False).head(3).index.tolist()
            if not z_scores.empty
            else []
        )
        peak = float(window_scores.max())
        persistent = duration >= 1800 or peak >= 0.98
        events.append(
            {
                "Start": start_ts,
                "End": end_ts,
                "PeakScore": peak,
                "DurationMin": round(duration / 60.0, 3),
                "Persistence": "persistent" if persistent else "transient",
                "Regime": int(regime),
                "TopTags": ",".join(top_tags),
            }
        )
        prev_end = end_ts
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------


def regime_population_stability(current: Sequence[float], reference: Sequence[float]) -> float:
    current = np.asarray(current, dtype=float)
    reference = np.asarray(reference, dtype=float)
    current = current / current.sum() if current.sum() else current
    reference = reference / reference.sum() if reference.sum() else reference
    psi = np.nansum((current - reference) * np.log((current + 1e-9) / (reference + 1e-9)))
    return float(psi)


def detect_regime_drift(
    regimes: np.ndarray,
    reference_hist: Optional[Dict[int, float]],
    warn: float,
    alert: float,
) -> Dict[str, Any]:
    counts = pd.Series(regimes).value_counts(normalize=True)
    if not counts.empty and reference_hist:
        aligned_ref = np.array([reference_hist.get(int(k), 1e-6) for k in counts.index])
    else:
        aligned_ref = np.ones_like(counts.values) / max(len(counts), 1)
    psi = regime_population_stability(counts.values, aligned_ref)
    level = "ok"
    if psi >= alert:
        level = "alert"
    elif psi >= warn:
        level = "warn"
    return {"metric": "psi_regime", "value": psi, "threshold": alert, "level": level}


# ---------------------------------------------------------------------------
# Brief payload
# ---------------------------------------------------------------------------


def build_brief_payload(
    equip: str,
    window: Tuple[Optional[datetime], Optional[datetime]],
    regimes: Dict[int, float],
    thresholds: Dict[int, float],
    events: pd.DataFrame,
    drift_info: Dict[str, Any],
) -> Dict[str, Any]:
    events_payload = []
    for row in events.itertuples():
        events_payload.append(
            {
                "start": row.Start.isoformat(),
                "end": row.End.isoformat(),
                "regime": int(row.Regime) if "Regime" in events.columns else None,
                "peak_tau": float(row.PeakScore),
                "persistence": row.Persistence,
                "top_tags": row.TopTags.split(",") if isinstance(row.TopTags, str) else [],
            }
        )
    return {
        "equip": equip,
        "window": {"t0": window[0].isoformat() if window[0] else None, "t1": window[1].isoformat() if window[1] else None},
        "regimes": [{"id": int(k), "share": float(v)} for k, v in regimes.items()],
        "thresholds": thresholds,
        "events": events_payload,
        "drift": drift_info,
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    run_id: str
    features_path: Path
    regimes_path: Path
    model_paths: Dict[str, Path]
    metadata: Dict[str, Any]


@dataclass
class ScoreResult:
    run_id: str
    scores_path: Path
    events_path: Path
    drift: Dict[str, Any]
    metadata: Dict[str, Any]


class PipelineRunner:
    def __init__(self, equip: str, art_root: Path, config: PipelineConfig):
        self.equip = equip
        self.art_root = art_root
        self.config = config

    def train(
        self,
        csv_source: Path,
        *,
        t0: Optional[datetime] = None,
        t1: Optional[datetime] = None,
    ) -> TrainingResult:
        artifact_manager = ArtifactManager(self.art_root, self.equip, run_type="train")
        art_paths = artifact_manager.start_run()

        sampling = self.config.get("sampling.period", "10s")
        window_seconds = int(pd.to_timedelta(self.config.get("features.window", "60s")).total_seconds())
        step_seconds = int(pd.to_timedelta(self.config.get("features.step", "10s")).total_seconds())
        spectral_bands = self.config.get("features.spectral_bands", [[0.0, 0.1]])

        raw, clean, dq, tags = prepare_window(
            csv_source,
            equip=self.equip,
            resample_rule=sampling,
            clamp_sigma=6.0,
            t0=t0,
            t1=t1,
        )
        artifact_manager.write_table(raw, "raw.parquet")
        artifact_manager.write_table(clean, "clean.parquet")
        artifact_manager.write_json(dq.to_dict(orient="records"), "dq.json")

        period_seconds = int(pd.to_timedelta(sampling).total_seconds())
        feature_df = build_feature_matrix(
            clean,
            period_seconds=period_seconds,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            spectral_bands=spectral_bands,
            key_tags=self.config.get("report.key_tags", []),
        )
        features_path = artifact_manager.write_table(feature_df, "features.parquet")

        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_df.fillna(method="ffill").fillna(0.0).values)
        pca, latent = fit_pca(feature_matrix, self.config.get("pca.variance", 0.95))

        min_duration_s = self.config.get("segmentation.min_duration_s", 60)
        segment_size = max(int(min_duration_s / step_seconds), 5)
        segments = segment_latent(latent, min_size=segment_size)
        cluster_labels, cluster_model, cluster_type = cluster_segments(
            latent,
            segments,
            algo=self.config.get("clustering.algo", "kmeans_auto"),
            k_range=tuple(self.config.get("clustering.k_range", [2, 6])),
            min_cluster_size=max(int((self.config.get("clustering.min_cluster_minutes", 3) * 60) / step_seconds), 2),
        )
        sample_labels = expand_labels_to_samples(segments, cluster_labels, len(feature_df))
        min_state_seconds = self.config.get("hmm.min_state_seconds", 60)
        min_state_samples = max(int(min_state_seconds / step_seconds), 1)
        regimes = smooth_labels(sample_labels, min_state_samples)

        regimes_df = pd.DataFrame({"Ts": feature_df.index, "Regime": regimes})
        regimes_path = artifact_manager.write_json(regimes_df.to_dict(orient="records"), "regimes.json")

        detectors = train_regime_detectors(feature_matrix, regimes)
        detector_path = artifact_manager.latest_model_path("detectors.joblib")
        dump(detectors, detector_path)

        pt_scores = score_regime_detectors(feature_matrix, regimes, detectors)
        seq_scores = sequence_score(latent, window=max(10, min_state_samples))
        win_scores = window_score(pt_scores, window=max(10, min_state_samples))
        fused = fuse_scores(
            pt_scores,
            win_scores,
            seq_scores,
            self.config.get("detectors.fusion_weights", [0.5, 0.3, 0.2]),
        )
        thresholds = calibrate_thresholds(
            fused,
            regimes,
            alpha=self.config.get("thresholds.alpha", 0.001),
            fallback_quantile=self.config.get("thresholds.fallback_quantile", 0.995),
        )
        artifact_manager.write_json({int(k): float(v) for k, v in thresholds.items()}, "thresholds.json")

        scaler_path = artifact_manager.latest_model_path("scaler.joblib")
        dump(scaler, scaler_path)
        pca_path = artifact_manager.latest_model_path("pca.joblib")
        dump(pca, pca_path)
        cluster_path = artifact_manager.latest_model_path("cluster.joblib")
        dump(cluster_model, cluster_path)
        meta = {
            "k": int(len(np.unique(regimes))),
            "segments": len(segments),
            "cluster_type": cluster_type,
            "period_seconds": period_seconds,
            "feature_window_seconds": window_seconds,
            "feature_step_seconds": step_seconds,
            "reference_hist": {int(k): float(v) for k, v in pd.Series(regimes).value_counts(normalize=True).items()},
        }
        meta_path = artifact_manager.latest_model_path("regime_meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # TODO: Push regime model to SQL via usp_WriteRegimeModels once available.

        artifact_manager.mark_success({"stage": "train", "k": meta["k"]})
        artifact_manager.emit_run_summary(
            {
                "run_id": art_paths.run_id,
                "ts_utc": datetime.utcnow().isoformat() + "Z",
                "equip": self.equip,
                "cmd": "train",
                "rows_in": len(clean),
                "tags": len(tags),
                "feat_rows": len(feature_df),
                "regimes": meta["k"],
                "events": 0,
                "data_span_min": (feature_df.index[-1] - feature_df.index[0]).total_seconds() / 60.0 if not feature_df.empty else 0.0,
                "phase": "train",
                "k_selected": meta["k"],
                "theta_p95": float(np.quantile(fused, 0.95)) if len(fused) else 0.0,
                "drift_flag": 0,
                "guardrail_state": "ok",
                "theta_step_pct": 0.0,
                "latency_s": 0.0,
                "artifacts_age_min": 0.0,
                "status": "ok",
                "err_msg": "",
            }
        )

        return TrainingResult(
            run_id=art_paths.run_id,
            features_path=features_path,
            regimes_path=regimes_path,
            model_paths={
                "scaler": scaler_path,
                "pca": pca_path,
                "cluster": cluster_path,
                "detectors": detector_path,
                "thresholds": artifact_manager._ensure_run_dir() / "thresholds.json",
                "meta": meta_path,
            },
            metadata=meta,
        )

    def score(
        self,
        csv_source: Path,
        *,
        t0: Optional[datetime] = None,
        t1: Optional[datetime] = None,
    ) -> ScoreResult:
        artifact_manager = ArtifactManager(self.art_root, self.equip, run_type="score")
        art_paths = artifact_manager.start_run()

        sampling = self.config.get("sampling.period", "10s")
        window_seconds = int(pd.to_timedelta(self.config.get("features.window", "60s")).total_seconds())
        step_seconds = int(pd.to_timedelta(self.config.get("features.step", "10s")).total_seconds())
        spectral_bands = self.config.get("features.spectral_bands", [[0.0, 0.1]])

        raw, clean, dq, _ = prepare_window(
            csv_source,
            equip=self.equip,
            resample_rule=sampling,
            clamp_sigma=6.0,
            t0=t0,
            t1=t1,
        )
        artifact_manager.write_table(raw, "raw.parquet")
        artifact_manager.write_table(clean, "clean.parquet")
        artifact_manager.write_json(dq.to_dict(orient="records"), "dq.json")

        period_seconds = int(pd.to_timedelta(sampling).total_seconds())
        feature_df = build_feature_matrix(
            clean,
            period_seconds=period_seconds,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            spectral_bands=spectral_bands,
            key_tags=self.config.get("report.key_tags", []),
        )
        artifact_manager.write_table(feature_df, "features.parquet")

        scaler = load(artifact_manager.latest_model_path("scaler.joblib"))
        pca = load(artifact_manager.latest_model_path("pca.joblib"))
        cluster_model = load(artifact_manager.latest_model_path("cluster.joblib"))
        detectors = load(artifact_manager.latest_model_path("detectors.joblib"))
        thresholds_path = artifact_manager.latest_model_path("thresholds.json")
        thresholds = (
            {int(k): float(v) for k, v in json.loads(thresholds_path.read_text(encoding="utf-8")).items()}
            if thresholds_path.exists()
            else {}
        )
        meta_path = artifact_manager.latest_model_path("regime_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        feature_matrix = feature_df.fillna(method="ffill").fillna(0.0).values
        X = scaler.transform(feature_matrix)
        latent = pca.transform(X)
        if hasattr(cluster_model, "predict"):
            regimes = cluster_model.predict(latent)
        else:  # pragma: no cover
            raise NotImplementedError("Cluster model does not support predict(); extend for HDBSCAN when needed.")

        pt_scores = score_regime_detectors(X, regimes, detectors)
        seq_scores = sequence_score(latent, window=max(10, int(meta.get("feature_step_seconds", step_seconds))))
        win_scores = window_score(pt_scores, window=max(10, int(meta.get("feature_step_seconds", step_seconds))))
        fused = fuse_scores(
            pt_scores,
            win_scores,
            seq_scores,
            self.config.get("detectors.fusion_weights", [0.5, 0.3, 0.2]),
        )
        if not thresholds:
            thresholds = calibrate_thresholds(
                fused,
                regimes,
                alpha=self.config.get("thresholds.alpha", 0.001),
                fallback_quantile=self.config.get("thresholds.fallback_quantile", 0.995),
            )
            thresholds_path.write_text(json.dumps({int(k): float(v) for k, v in thresholds.items()}, indent=2), encoding="utf-8")

        fused_series = pd.Series(fused, index=feature_df.index, name="FusedScore")
        scores_df = pd.DataFrame(
            {
                "Ts": feature_df.index,
                "Regime": regimes,
                "FusedScore": fused,
                "PtScore": pt_scores,
                "WinScore": win_scores,
                "SeqScore": seq_scores,
            }
        ).set_index("Ts")
        scores_path = artifact_manager.write_table(scores_df, "scores.csv", prefer_parquet=False)

        events_df = label_events(
            fused_series,
            thresholds,
            regimes,
            clean,
            min_len_seconds=int(pd.to_timedelta(self.config.get("eventization.min_len", "30s")).total_seconds()),
            merge_gap_seconds=int(pd.to_timedelta(self.config.get("eventization.merge_gap", "120s")).total_seconds()),
        )
        events_path = artifact_manager.write_table(events_df.set_index("Start"), "anomalies.csv", prefer_parquet=False)

        drift_info = detect_regime_drift(
            regimes,
            meta.get("reference_hist"),
            warn=self.config.get("drift.psi_regime_warn", 0.25),
            alert=self.config.get("drift.psi_regime_alert", 0.5),
        )

        window_tuple = (
            feature_df.index.min().to_pydatetime() if not feature_df.empty else None,
            feature_df.index.max().to_pydatetime() if not feature_df.empty else None,
        )
        brief_payload = build_brief_payload(
            equip=self.equip,
            window=window_tuple,
            regimes={int(k): float(v) for k, v in pd.Series(regimes).value_counts(normalize=True).items()},
            thresholds=thresholds,
            events=events_df,
            drift_info=drift_info,
        )
        artifact_manager.write_json(brief_payload, "brief.json")
        artifact_manager.write_payloads()

        artifact_manager.mark_success({"stage": "score", "events": len(events_df)})
        artifact_manager.emit_run_summary(
            {
                "run_id": art_paths.run_id,
                "ts_utc": datetime.utcnow().isoformat() + "Z",
                "equip": self.equip,
                "cmd": "score",
                "rows_in": len(clean),
                "tags": clean.shape[1],
                "feat_rows": len(feature_df),
                "regimes": int(len(np.unique(regimes))),
                "events": len(events_df),
                "data_span_min": (feature_df.index[-1] - feature_df.index[0]).total_seconds() / 60.0 if not feature_df.empty else 0.0,
                "phase": "score",
                "k_selected": int(len(np.unique(regimes))),
                "theta_p95": float(np.quantile(fused, 0.95)) if len(fused) else 0.0,
                "drift_flag": 1 if drift_info.get("level") in {"warn", "alert"} else 0,
                "guardrail_state": drift_info.get("level", "ok"),
                "theta_step_pct": 0.0,
                "latency_s": 0.0,
                "artifacts_age_min": 0.0,
                "status": "ok",
                "err_msg": "",
            }
        )

        return ScoreResult(
            run_id=art_paths.run_id,
            scores_path=scores_path,
            events_path=events_path,
            drift=drift_info,
            metadata={"thresholds": thresholds},
        )

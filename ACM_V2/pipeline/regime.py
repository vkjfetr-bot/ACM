"""Regime discovery utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import ruptures as rpt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rpt = None

try:
    from hdbscan import HDBSCAN  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    HDBSCAN = None


@dataclass
class RegimeModel:
    pca: PCA
    cluster: object
    cluster_type: str
    metadata: Dict[str, float]


def fit_pca(X: np.ndarray, variance: float) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=variance, svd_solver="full")
    latent = pca.fit_transform(X)
    return pca, latent


def segment_latent(latent: np.ndarray, min_size: int) -> List[Tuple[int, int]]:
    """Detect change-points using ruptures if available, else fixed windows."""
    n = latent.shape[0]
    if n == 0:
        return []
    if rpt is None or n < min_size * 2:
        # Fallback: split into equal segments of min_size.
        segments = []
        for start in range(0, n, min_size):
            end = min(start + min_size, n)
            segments.append((start, end))
        return segments
    model = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
    algo = model.fit(latent)
    bkps = algo.predict(pen=3)  # simple penalty; tune later
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
    """Cluster latent segment centroids."""
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
    # Default to auto-kmeans search.
    k_min, k_max = k_range
    best_score = -1
    best_labels = None
    best_model = None
    for k in range(k_min, k_max + 1):
        if k <= 1 or k > len(centroid_arr):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(centroid_arr)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(centroid_arr, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_model = km
    if best_labels is None:
        km = KMeans(n_clusters=min(k_range[1], len(centroid_arr)), random_state=42, n_init="auto")
        best_labels = km.fit_predict(centroid_arr)
        best_model = km
    return best_labels, best_model, "kmeans"


def expand_labels_to_samples(segments: List[Tuple[int, int]], labels: np.ndarray, n_samples: int) -> np.ndarray:
    out = np.zeros(n_samples, dtype=int)
    for (start, end), label in zip(segments, labels):
        out[start:end] = label
    return out


def smooth_labels(labels: np.ndarray, min_duration: int) -> np.ndarray:
    """Enforce minimum dwell time by collapsing short bursts."""
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

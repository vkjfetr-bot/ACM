"""Feature engineering for ACM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    window_seconds: int
    step_seconds: int
    spectral_bands: List[Tuple[float, float]]


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    period_seconds: int,
    config: FeatureConfig,
    key_tags: Iterable[str],
) -> pd.DataFrame:
    """Construct rolling statistical, trend, and spectral features."""
    window = max(int(config.window_seconds / period_seconds), 1)
    step = max(int(config.step_seconds / period_seconds), 1)
    if window <= 1:
        raise ValueError("Feature window too small relative to sampling period.")

    indices = []
    rows: List[Dict[str, float]] = []
    values = df.to_numpy(dtype=float)
    columns = df.columns.tolist()
    times = df.index.to_numpy()

    spectral_rows = len(config.spectral_bands)
    band_edges = np.asarray(config.spectral_bands)
    fs = 1.0 / period_seconds if period_seconds else 1.0

    for start in range(0, len(df) - window + 1, step):
        end = start + window
        window_slice = values[start:end]
        if np.isnan(window_slice).all():
            continue
        current_index = times[end - 1]
        indices.append(current_index)
        stats = _window_stats(window_slice)
        trends = _window_trends(window_slice)
        iqrs = _window_iqr(window_slice)
        percentiles = _window_percentiles(window_slice)
        spectral = _window_spectrum(window_slice, fs, band_edges, spectral_rows)
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
            for band_idx in range(spectral_rows):
                row[f"{prefix}_spec_b{band_idx}"] = spectral[band_idx, col_idx]
        # Cross-tag correlations with key drivers.
        if key_tags:
            key_indices = [columns.index(tag) for tag in key_tags if tag in columns]
        else:
            key_indices = []
        for idx in key_indices:
            for jdx in range(len(columns)):
                if idx == jdx:
                    continue
                corr = np.corrcoef(window_slice[:, idx], window_slice[:, jdx])[0, 1]
                if np.isfinite(corr):
                    row[f"corr_{columns[jdx]}_to_{columns[idx]}"] = corr
        rows.append(row)

    feature_df = pd.DataFrame(rows, index=pd.to_datetime(indices))
    feature_df = feature_df.sort_index()
    return feature_df


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
    # Normalise each column to [0,1].
    col_max = band_energy.max(axis=0)
    col_max[col_max == 0] = 1.0
    band_energy /= col_max
    return band_energy

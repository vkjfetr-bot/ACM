"""
viz_core.py

Low-level plotting helpers and data sanitization for the next-gen ACM report.
All functions are deterministic by default and tolerate NaNs/Infs.
"""

from __future__ import annotations

import os
import math
import json
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib

# Windows-friendly, no GUI backends
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------- Determinism & Paths ----------------------

def set_seed(seed: int = 42) -> None:
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------- Sanitizers ----------------------

def sanitize_array(a: np.ndarray, replace: float = 0.0, clip_p: float = 99.9) -> np.ndarray:
    """Replace NaN/Inf and clip extremes for robust plotting.
    Returns a copy.
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    b = a.astype(float).copy()
    b[~np.isfinite(b)] = replace
    if b.size == 0:
        return b
    # Clip both tails to reduce colorbar skew
    lo, hi = np.nanpercentile(b, [100 - clip_p, clip_p]) if np.isfinite(b).any() else (None, None)
    if lo is not None and hi is not None and hi > lo:
        b = np.clip(b, lo, hi)
    return b


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def save_fig(fig: plt.Figure, out_path: str, dpi: int = 120) -> str:
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------- Plots ----------------------

def plot_trend(ts: pd.Series, title: str, out_path: str,
               anomalies: Optional[Sequence[int]] = None,
               mask_idx: Optional[Sequence[int]] = None) -> str:
    """Simple trend line; marks ▲ for anomalies, × for masked."""
    s = pd.to_numeric(ts, errors="coerce")
    x = np.arange(len(s))
    y = sanitize_array(s.values)

    fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
    ax.plot(x, y, color="#60a5fa", lw=1.2)
    if anomalies:
        ax.scatter(list(anomalies), y[list(anomalies)], marker='^', color="#ef4444", s=30, label="anomaly")
    if mask_idx:
        ax.scatter(list(mask_idx), y[list(mask_idx)], marker='x', color="#f59e0b", s=16, label="masked")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    return save_fig(fig, out_path)


def plot_heatmap(mat: np.ndarray, title: str, out_path: str,
                 x_label: str = "time", y_label: str = "tags",
                 cmap: str = "magma") -> str:
    m = sanitize_array(np.asarray(mat))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    im = ax.imshow(m, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.ax.tick_params(labelsize=8)
    return save_fig(fig, out_path)


def plot_threshold_trace(score: pd.Series, threshold: pd.Series, out_path: str,
                         title: str = "Score vs Threshold") -> str:
    s = pd.to_numeric(score, errors="coerce").fillna(0.0).values
    t = pd.to_numeric(threshold, errors="coerce").fillna(0.0).values
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
    ax.plot(x, s, label="score", color="#60a5fa", lw=1.2)
    ax.plot(x, t, label="threshold", color="#34d399", lw=1.2)
    ax.fill_between(x, t, s, where=s > t, color="#ef4444", alpha=0.15)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    return save_fig(fig, out_path)


def compute_dq(raw_df: pd.DataFrame, tags: Sequence[str]) -> pd.DataFrame:
    rows = []
    for t in tags:
        if t not in raw_df.columns:
            continue
        s = pd.to_numeric(raw_df[t], errors="coerce").astype(float)
        n = int(s.shape[0])
        if n == 0:
            continue
        flat = float((s.diff().abs() < 1e-12).sum()) / max(n, 1) * 100.0
        drop = float(s.isna().sum()) / max(n, 1) * 100.0
        d = s.diff().dropna().abs()
        iqr = (d.quantile(0.75) - d.quantile(0.25)) + 1e-9
        spikes = int((d > 5 * iqr).sum())
        rows.append({"Tag": t, "Flatline%": flat, "Dropout%": drop, "Spikes": spikes})
    return pd.DataFrame(rows)


def plot_tag_health(dq_df: pd.DataFrame, out_path: str, top_k: int = 20) -> str:
    if dq_df.empty:
        # Create an empty placeholder figure
        fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, "No DQ data", ha="center", va="center")
        return save_fig(fig, out_path)

    df = dq_df.copy()
    df["dq_score"] = df[["Flatline%", "Dropout%", "Spikes"]].apply(
        lambda r: r[0] * 0.5 + r[1] * 0.4 + min(r[2], 100) * 0.1, axis=1
    )
    df = df.sort_values("dq_score", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.barh(df["Tag"], df["dq_score"], color="#93c5fd")
    ax.invert_yaxis()
    ax.set_title("Tag Health (higher = worse)")
    ax.set_xlabel("score")
    return save_fig(fig, out_path)



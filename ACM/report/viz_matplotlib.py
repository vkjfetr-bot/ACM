from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(p: str) -> str:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def plot_overview_timeline(scored: pd.DataFrame, regimes: List[dict], outpath: str) -> str:
    s = pd.to_numeric(scored.get("FusedScore", pd.Series(index=scored.index, data=0.0)), errors="coerce").fillna(0.0)
    x = np.arange(len(s))
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
    axs[0].plot(x, s.values, lw=1.2, color="#60a5fa", label="FusedScore")
    axs[0].set_title("Fused Anomaly Score")
    axs[0].grid(True, alpha=0.2)
    # Regime band (if available)
    if "Regime" in scored.columns:
        axs[1].plot(x, pd.to_numeric(scored["Regime"], errors="coerce").fillna(0.0).values, color="#93c5fd")
        axs[1].set_title("Regime")
    else:
        axs[1].axis("off")
    for ax in axs:
        ax.set_xlabel("time")
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_tag_trend(df: pd.DataFrame, tag: str, anom_mask: pd.Series, outpath: str) -> str:
    s = pd.to_numeric(df.get(tag, pd.Series(index=df.index, data=np.nan)), errors="coerce")
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.plot(x, s.values, color="#34d399", lw=1.0)
    if anom_mask is not None and not anom_mask.empty:
        idx = np.where(anom_mask.values)[0]
        ax.scatter(idx, s.values[idx], s=12, color="#ef4444", label="anom")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(tag)
    ax.grid(True, alpha=0.2)
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_drift(df: pd.DataFrame, tag: str, baseline_win: int, outpath: str) -> str:
    s = pd.to_numeric(df.get(tag, pd.Series(index=df.index, data=np.nan)), errors="coerce").fillna(method="ffill").fillna(method="bfill")
    mu = s.rolling(baseline_win, min_periods=max(3, baseline_win // 4)).mean()
    std = s.rolling(baseline_win, min_periods=max(3, baseline_win // 4)).std().fillna(1.0)
    z = (s - mu) / (std + 1e-9)
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.plot(x, z.values, color="#fbbf24", lw=1.0)
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title(f"Drift z-score — {tag}")
    ax.grid(True, alpha=0.2)
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_dq_heatmap(dq_df: pd.DataFrame, outpath: str) -> str:
    if dq_df is None or dq_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, "No DQ data", ha="center", va="center")
        _ensure_dir(outpath)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return outpath
    M = dq_df[["coverage", "dropout", "flatline", "spikes"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(dq_df))), constrained_layout=True)
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(np.arange(len(dq_df)))
    ax.set_yticklabels(dq_df["tag"].tolist(), fontsize=8)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["coverage", "dropout", "flatline", "spikes"], rotation=30, ha="right")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_corrs(df_normal: pd.DataFrame, df_anom: pd.DataFrame, tags: List[str], outpath_prefix: str) -> tuple[str, str]:
    def _corr(df):
        return df[tags].corr(method="spearman").to_numpy()
    Cn = _corr(df_normal) if not df_normal.empty else np.zeros((len(tags), len(tags)))
    Ca = _corr(df_anom) if not df_anom.empty else np.zeros((len(tags), len(tags)))
    for mat, suffix, title in [(Cn, "normal", "Correlation (normal)"), (Ca, "anom", "Correlation (anomalous)")]:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        p = f"{outpath_prefix}_{suffix}.png"
        _ensure_dir(p)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return f"{outpath_prefix}_normal.png", f"{outpath_prefix}_anom.png"


def plot_episode_strip(scored: pd.DataFrame, ep: dict, tags: List[str], outpath: str) -> str:
    s = pd.to_numeric(scored.get("FusedScore", pd.Series(index=scored.index, data=0.0)), errors="coerce").fillna(0.0)
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 2.5), constrained_layout=True)
    ax.plot(x, s.values, lw=1.0, color="#60a5fa")
    ax.set_title(f"Episode {ep.get('id', '')}")
    ax.grid(True, alpha=0.2)
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


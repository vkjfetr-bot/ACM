"""Regime (Operating State) Visuals.

Plots a color strip of regime labels over time and computes a summary table
with counts and duration per regime.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_regime_strip(labels: pd.Series, out_path: str | Path) -> None:
    if labels.empty:
        return
    # Map regimes to color indices
    regs = labels.astype(int).to_numpy()
    uniq = np.unique(regs)
    cmap = plt.get_cmap('tab10')
    colors = {r: cmap(i % 10) for i, r in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(12, 1.2))
    x = np.arange(len(regs))
    ax.scatter(x, np.zeros_like(x), c=[colors[r] for r in regs], s=10, marker='s')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Operating States (Regimes)')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def regime_summary(labels: pd.Series) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(columns=["Regime","Count","Duration_h"])
    # Approximate duration from timestamps if DateTimeIndex
    if isinstance(labels.index, pd.DatetimeIndex) and len(labels) > 1:
        dt = (labels.index[1] - labels.index[0]).total_seconds() / 3600.0
    else:
        dt = 0.0
    counts = labels.value_counts().sort_index()
    df = pd.DataFrame({
        "Regime": counts.index.astype(int),
        "Count": counts.values,
        "Duration_h": (counts.values * dt) if dt > 0 else np.nan,
    })
    return df


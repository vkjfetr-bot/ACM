"""Timeline Visualization

Plots fused score with H1/H2/H3 bands, regimes, and masks as color strips.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_timeline(scores: pd.DataFrame, masks: pd.DataFrame | None = None, regimes: pd.Series | None = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    if "fused" in scores:
        scores["fused"].plot(ax=ax, color="black", lw=1.5, label="fused")
    for name, color in [("H1_AR1", "#1f77b4"), ("H2_PCA", "#ff7f0e"), ("H3_Drift", "#2ca02c")]:
        if name in scores:
            scores[name].plot(ax=ax, alpha=0.5, label=name)
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("ACMnxt Timeline")
    ax.grid(True, alpha=0.2)
    # Masks/regimes overlays can be added in subsequent iterations
    return fig


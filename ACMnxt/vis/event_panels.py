"""Event Panels Visualization

Renders per-event window, spectra snapshot, and tag trend grid.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_event_panel(df: pd.DataFrame, score: pd.Series, start: pd.Timestamp, end: pd.Timestamp):
    sub = df.loc[start:end]
    ssub = score.loc[start:end]
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
    sub.plot(ax=ax1, lw=0.8)
    ax1.set_title("Tag trends")
    ssub.plot(ax=ax2, color="black")
    ax2.set_title("Fused score")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.2)
    return fig


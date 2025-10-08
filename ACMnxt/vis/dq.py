"""DQ Visualization — Heatmap of NaNs per tag over time.

Creates a simple binary heatmap (NaN=1, OK=0) for each tag over time.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dq_heatmap(df: pd.DataFrame, out_path: str | Path) -> None:
    num = df.select_dtypes(include=[float, int])
    if num.empty:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return
    # Downsample along time and show availability (1-good, 0-bad)
    avail = num.notna().astype(float)
    if isinstance(num.index, pd.DatetimeIndex):
        data = avail.resample("15min").mean()
    else:
        # stride by 10 then mean within simple windows
        data = avail.iloc[::10]
    mat = data.values.T  # tags x time, values in [0,1]
    fig, ax = plt.subplots(figsize=(10, max(2, 0.25 * mat.shape[0])))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="YlGn", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(num.columns)))
    ax.set_yticklabels(list(num.columns), fontsize=7)
    ax.set_title("DQ Availability (1=good, 0=missing)")
    ax.set_xlabel("time buckets")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

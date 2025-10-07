# report_charts.py
# Builds static matplotlib charts (PNGâ†’base64) for HTML embedding.
# Focus: clear, interpretable plots (no interactive JS).

import io, base64
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FIG_DPI = 130
FUSED_TAU = 0.70  # threshold line for anomaly marking

# ---------- Utility helpers ----------

def _embed(fig) -> str:
    """Convert matplotlib figure â†’ base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _sample(df: pd.DataFrame, max_points: int = 1800) -> pd.DataFrame:
    """Downsample long series by rolling mean + stride."""
    if len(df) <= max_points:
        return df
    k = max(1, len(df)//max_points)
    roll = df.rolling(k, min_periods=1).mean()
    return roll.iloc[::k]

# ---------- Charts ----------

def timeline(scored: pd.DataFrame,
             events: Optional[pd.DataFrame],
             masks: Optional[pd.DataFrame]) -> str:
    """
    Fused + heads + regime timeline with event shading & mask overlay.
    """
    ts = pd.to_datetime(scored.index)
    cols = ["FusedScore"] + [c for c in ["H1_Forecast","H2_Recon","H3_Contrast"] if c in scored.columns]
    use = _sample(scored[cols], 1800)
    ts2 = pd.to_datetime(use.index)

    rows = 1 + (1 if len(cols)>1 else 0) + (1 if "Regime" in scored.columns else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(13, 2.3*rows), sharex=True)
    if rows == 1:
        axes = [axes]
    i = 0

    # Row 1 â€“ Fused
    ax = axes[i]; i += 1
    ax.plot(ts2, use["FusedScore"].values, lw=1.4, label="FusedScore")
    ax.axhline(FUSED_TAU, color="gray", ls="--", lw=1.0)
    ax.set_ylabel("Fused"); ax.grid(alpha=0.25)

    # Events shading
    if events is not None and not events.empty:
        for r in events.itertuples():
            ax.axvspan(r.Start, r.End, color="#f59e0b", alpha=0.25)
            if hasattr(r, "PeakScore"):
                ax.scatter([r.End], [min(1.0, float(getattr(r,"PeakScore", 0)))],
                           s=12, color="#f59e0b", zorder=5)

    # Mask overlay (thin red band)
    if masks is not None and set(["Ts","Mask"]).issubset(masks.columns):
        mk = masks.copy()
        mk["Ts"] = pd.to_datetime(mk["Ts"])
        mk = mk.set_index("Ts").reindex(ts2, method="nearest")
        y = np.where(mk["Mask"].fillna(0).values>0, 0.02, np.nan)
        ax.plot(ts2, y, drawstyle="steps-mid", lw=2, color="#ef4444")

    # Row 2 â€“ Heads
    if len(cols)>1:
        ax = axes[i]; i += 1
        for c in cols[1:]:
            ax.plot(ts2, use[c].values, lw=1.0, label=c)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("Heads"); ax.grid(alpha=0.25)

    # Row 3 â€“ Regime ribbon
    if "Regime" in scored.columns:
        ax = axes[i]; i += 1
        reg = scored["Regime"].astype(int)
        x = mdates.date2num(ts)
        if len(x) >= 2:
            dx = np.diff(x)
            last_step = dx[-1] if len(dx) else (x[-1]-x[-2])
            if not np.isfinite(last_step) or last_step == 0:
                last_step = 1/1440
            edges = np.concatenate([x, [x[-1]+last_step]])
        else:
            edges = np.array([x[0], x[0]+1/1440])
        Z = reg.values[np.newaxis, :]
        y_edges = np.array([0,1])
        ax.pcolormesh(edges, y_edges, Z, cmap="tab20", shading="auto")
        ax.set_yticks([]); ax.set_ylabel("Regime")

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return _embed(fig)

def sampled_tags_with_marks(df: pd.DataFrame,
                            tags: List[str],
                            events: Optional[pd.DataFrame],
                            fused: Optional[pd.Series]) -> str:
    """
    Plot sampled raw tags (z-normalized) with:
      - anomalies (FusedScore â‰¥ Ï„) marked as dots
      - event spans shaded
    """
    use_cols = [t for t in tags if t in df.columns]
    if not use_cols:
        return ""
    z = df[use_cols].apply(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    use = _sample(z, 1800)
    ts = pd.to_datetime(use.index)

    rows = int(np.ceil(len(use_cols)/3)) or 1
    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.1*rows), sharex=True)
    axes = np.atleast_2d(axes)
    for i, col in enumerate(use_cols):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        ax.plot(ts, use[col].values, lw=0.9, label=col)
        if fused is not None:
            f2 = fused.reindex(df.index, method="nearest").reindex(use.index, method="nearest")
            mark = np.where(f2.values >= FUSED_TAU, use[col].values, np.nan)
            ax.scatter(ts, mark, s=6, color="#f97316")
        if events is not None and not events.empty:
            for ev in events.itertuples():
                ax.axvspan(ev.Start, ev.End, color="#f59e0b", alpha=0.12)
        ax.set_title(col, fontsize=9)
        ax.grid(alpha=0.15)
    for j in range(i+1, rows*3):
        r, c = divmod(j, 3)
        axes[r, c].axis("off")
    axes[-1,0].set_xlabel("Time")
    fig.tight_layout()
    return _embed(fig)

def drift_bars(drift: pd.DataFrame, top: int = 20) -> str:
    """Horizontal bar chart for top drifted tags."""
    if drift is None or drift.empty or not {"Tag","DriftZ"}.issubset(drift.columns):
        return ""
    d = drift.dropna(subset=["DriftZ"]).sort_values("DriftZ", ascending=False).head(top)
    if d.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(d["Tag"][::-1], d["DriftZ"][::-1], color="#60a5fa")
    ax.set_xlabel("Drift Z"); ax.set_title(f"Top {len(d)} Drifted Tags")
    fig.tight_layout()
    return _embed(fig)


def regime_share(scored: pd.DataFrame) -> str:
    """Pie chart of time share per regime."""
    if "Regime" not in scored.columns or scored.empty:
        return ""
    counts = scored["Regime"].astype(int).value_counts().sort_index()
    labels = [f"R{r} ({c/len(scored):.0%})" for r, c in counts.items()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(counts.values, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Regime Time Share")
    fig.tight_layout()
    return _embed(fig)


def fused_histogram(scored: pd.DataFrame) -> str:
    """Histogram of fused scores with τ line and exceedance fraction."""
    if "FusedScore" not in scored.columns or scored.empty:
        return ""
    s = pd.to_numeric(scored["FusedScore"], errors="coerce").dropna()
    if s.empty:
        return ""
    frac = (s >= FUSED_TAU).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s.values, bins=40, color="#60a5fa", alpha=0.9)
    ax.axvline(FUSED_TAU, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("FusedScore"); ax.set_ylabel("Count")
    ax.set_title(f"Fused Score Distribution (≥ τ: {frac:.0%})")
    fig.tight_layout()
    return _embed(fig)


def hourly_burden_heatmap(scored: pd.DataFrame) -> str:
    """Hour-of-day vs date heatmap of mean FusedScore."""
    if "FusedScore" not in scored.columns or scored.empty:
        return ""
    df = scored.copy()
    idx = pd.to_datetime(df.index)
    df["date"] = idx.date
    df["hour"] = idx.hour
    pivot = df.pivot_table(index="date", columns="hour", values="FusedScore", aggfunc="mean")
    if pivot.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, max(3, 0.2*len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(d) for d in pivot.index])
    ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([str(h) for h in range(0,24,3)])
    ax.set_xlabel("Hour of Day"); ax.set_title("Anomaly Burden by Hour (mean FusedScore)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    return _embed(fig)


def contribution_breakdown(scored: pd.DataFrame, n: int = 600) -> str:
    """Stacked preview of fusion components over the last n points."""
    cols_need = ["H1_Forecast","H2_Recon","H3_Contrast","CorrBoost","CPD","FusedScore"]
    if not all(c in scored.columns for c in cols_need):
        return ""
    use = scored[cols_need].tail(n)
    h1 = 0.45*use["H1_Forecast"].values
    h2 = 0.35*use["H2_Recon"].values
    h3 = 0.35*use["H3_Contrast"].values
    boost = 0.15*use["CorrBoost"].values + 0.10*use["CPD"].values
    x = range(len(use))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.stackplot(x, h1, h2, h3, boost, labels=["0.45*H1","0.35*H2","0.35*H3","Boost"], alpha=0.8)
    ax.plot(x, use["FusedScore"].values, color="black", lw=1.2, label="Fused")
    ax.set_title("Fusion Components (last window)"); ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Samples")
    fig.tight_layout()
    return _embed(fig)

# report_charts.py
# Builds static matplotlib charts (PNG→base64) for HTML embedding.
# Focus: clear, interpretable plots with optimized performance and aesthetics.

import io
import base64
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Configuration
FIG_DPI = 130
FUSED_TAU = 0.70  # threshold line for anomaly marking
COLORS = {
    'primary': '#2563eb',
    'secondary': '#60a5fa',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'accent': '#f97316',
    'success': '#10b981',
}

# Set default matplotlib style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.autolayout': False,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ---------- Utility helpers ----------

def _embed(fig) -> str:
    """Convert matplotlib figure → base64 data URI with optimization."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _sample(df: pd.DataFrame, max_points: int = 1800) -> pd.DataFrame:
    """Downsample long series efficiently using vectorized operations."""
    n = len(df)
    if n <= max_points:
        return df
    
    # Calculate stride to achieve target length
    stride = max(1, n // max_points)
    window = min(stride, 10)  # Cap rolling window for performance
    
    # Use efficient downsampling
    if window > 1:
        result = df.rolling(window, min_periods=1, center=False).mean().iloc[::stride]
    else:
        result = df.iloc[::stride]
    
    return result


def _add_event_spans(ax, events: Optional[pd.DataFrame], color: str = None, alpha: float = 0.25):
    """Add event shading spans to axis efficiently."""
    if events is None or events.empty:
        return
    
    color = color or COLORS['warning']
    for _, row in events.iterrows():
        ax.axvspan(row['Start'], row['End'], color=color, alpha=alpha, zorder=1)
        if 'PeakScore' in row and pd.notna(row['PeakScore']):
            ax.scatter([row['End']], [min(1.0, float(row['PeakScore']))],
                      s=15, color=color, zorder=5, marker='o', edgecolors='white', linewidths=0.5)


# ---------- Charts ----------

def timeline(scored: pd.DataFrame,
             events: Optional[pd.DataFrame],
             masks: Optional[pd.DataFrame]) -> str:
    """
    Fused + heads + regime timeline with event shading & mask overlay.
    Optimized for visual clarity and performance.
    """
    ts = pd.to_datetime(scored.index)
    cols = ["FusedScore"] + [c for c in ["H1_Forecast", "H2_Recon", "H3_Contrast"] 
                              if c in scored.columns]
    use = _sample(scored[cols], 1800)
    ts2 = pd.to_datetime(use.index)

    # Calculate layout
    n_rows = 1 + (1 if len(cols) > 1 else 0) + (1 if "Regime" in scored.columns else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)
    
    idx = 0

    # Row 1 – Fused Score
    ax = axes[idx]
    ax.plot(ts2, use["FusedScore"].values, lw=1.5, label="FusedScore", 
            color=COLORS['primary'], alpha=0.9)
    ax.axhline(FUSED_TAU, color='#6b7280', ls='--', lw=1.2, alpha=0.7, label=f'τ={FUSED_TAU}')
    ax.set_ylabel("Fused Score", fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)
    
    _add_event_spans(ax, events)
    
    # Mask overlay (thin red band at bottom)
    if masks is not None and {'Ts', 'Mask'}.issubset(masks.columns):
        mk = masks.copy()
        mk['Ts'] = pd.to_datetime(mk['Ts'])
        mk = mk.set_index('Ts').reindex(ts2, method='nearest')
        mask_vals = mk['Mask'].fillna(0).values > 0
        if mask_vals.any():
            ax.fill_between(ts2, 0, 0.03, where=mask_vals, 
                           color=COLORS['danger'], alpha=0.8, step='mid', zorder=3)
    
    idx += 1

    # Row 2 – Head Scores
    if len(cols) > 1:
        ax = axes[idx]
        head_colors = ['#8b5cf6', '#ec4899', '#14b8a6']
        for i, col in enumerate(cols[1:]):
            ax.plot(ts2, use[col].values, lw=1.2, label=col, 
                   color=head_colors[i % len(head_colors)], alpha=0.85)
        ax.legend(loc='upper right', framealpha=0.9, ncol=min(3, len(cols)-1))
        ax.set_ylabel("Head Scores", fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        idx += 1

    # Row 3 – Regime ribbon
    if "Regime" in scored.columns:
        ax = axes[idx]
        reg = scored["Regime"].astype(int)
        ts_full = pd.to_datetime(scored.index)
        x = mdates.date2num(ts_full)
        
        if len(x) >= 2:
            dx = np.median(np.diff(x))
            edges = np.concatenate([x, [x[-1] + dx]])
        else:
            edges = np.array([x[0], x[0] + 1/1440])
        
        Z = reg.values[np.newaxis, :]
        y_edges = np.array([0, 1])
        
        mesh = ax.pcolormesh(edges, y_edges, Z, cmap='tab20', shading='flat', 
                            rasterized=True)
        ax.set_yticks([])
        ax.set_ylabel("Regime", fontweight='bold')
        ax.set_ylim(0, 1)

    # Format x-axis
    axes[-1].set_xlabel("Time", fontweight='bold')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.tight_layout()
    return _embed(fig)


def sampled_tags_with_marks(df: pd.DataFrame,
                            tags: List[str],
                            events: Optional[pd.DataFrame],
                            fused: Optional[pd.Series]) -> str:
    """
    Plot sampled raw tags (z-normalized) with anomalies and events.
    Optimized grid layout and visual hierarchy.
    """
    use_cols = [t for t in tags if t in df.columns]
    if not use_cols:
        return ""
    
    # Z-normalize efficiently
    z = df[use_cols].apply(lambda s: (s - s.mean()) / (s.std() + 1e-9), axis=0)
    use = _sample(z, 1800)
    ts = pd.to_datetime(use.index)

    # Optimize grid layout
    n_cols = min(3, len(use_cols))
    n_rows = int(np.ceil(len(use_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 2.2 * n_rows), 
                            sharex=True, squeeze=False)
    
    for i, col in enumerate(use_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        
        # Plot line
        ax.plot(ts, use[col].values, lw=1.0, color=COLORS['primary'], alpha=0.8)
        
        # Mark anomalies
        if fused is not None:
            f_aligned = fused.reindex(df.index, method='nearest').reindex(use.index, method='nearest')
            anomaly_mask = f_aligned.values >= FUSED_TAU
            if anomaly_mask.any():
                ax.scatter(ts[anomaly_mask], use[col].values[anomaly_mask], 
                          s=8, color=COLORS['accent'], alpha=0.9, zorder=4)
        
        # Add events
        _add_event_spans(ax, events, alpha=0.15)
        
        ax.set_title(col, fontsize=9, fontweight='bold', pad=4)
        ax.set_ylabel('Z-score', fontsize=8)
    
    # Hide unused subplots
    for j in range(len(use_cols), n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis('off')
    
    axes[-1, 0].set_xlabel("Time", fontweight='bold')
    fig.tight_layout()
    return _embed(fig)


def drift_bars(drift: pd.DataFrame, top: int = 20) -> str:
    """Horizontal bar chart for top drifted tags with enhanced styling."""
    if drift is None or drift.empty or not {'Tag', 'DriftZ'}.issubset(drift.columns):
        return ""
    
    d = drift.dropna(subset=['DriftZ']).sort_values('DriftZ', ascending=False).head(top)
    if d.empty:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(d))))
    
    # Create color gradient
    colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(d)))
    
    bars = ax.barh(range(len(d)), d['DriftZ'].values, color=colors, 
                   edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d['Tag'].values, fontsize=8)
    ax.set_xlabel("Drift Z-Score", fontweight='bold')
    ax.set_title(f"Top {len(d)} Drifted Tags", fontweight='bold', pad=10)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(d.iterrows()):
        ax.text(row['DriftZ'] + 0.05, i, f"{row['DriftZ']:.2f}", 
               va='center', fontsize=7, color='#374151')
    
    fig.tight_layout()
    return _embed(fig)


def regime_share(scored: pd.DataFrame) -> str:
    """Pie chart of time share per regime with improved aesthetics."""
    if "Regime" not in scored.columns or scored.empty:
        return ""
    
    counts = scored["Regime"].astype(int).value_counts().sort_index()
    labels = [f"Regime {r}" for r in counts.index]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Use distinct colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    
    wedges, texts, autotexts = ax.pie(counts.values, labels=labels, autopct='%1.1f%%',
                                       startangle=90, colors=colors,
                                       explode=[0.05] * len(counts),
                                       textprops={'fontsize': 9})
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("Regime Time Share", fontweight='bold', pad=15, fontsize=11)
    fig.tight_layout()
    return _embed(fig)


def fused_histogram(scored: pd.DataFrame) -> str:
    """Histogram of fused scores with enhanced visual design."""
    if "FusedScore" not in scored.columns or scored.empty:
        return ""
    
    s = pd.to_numeric(scored["FusedScore"], errors='coerce').dropna()
    if s.empty:
        return ""
    
    frac = (s >= FUSED_TAU).mean()
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    # Create histogram with color coding
    n, bins, patches = ax.hist(s.values, bins=50, edgecolor='white', linewidth=0.5)
    
    # Color bars based on threshold
    for i, patch in enumerate(patches):
        if bins[i] >= FUSED_TAU:
            patch.set_facecolor(COLORS['warning'])
            patch.set_alpha(0.9)
        else:
            patch.set_facecolor(COLORS['secondary'])
            patch.set_alpha(0.7)
    
    # Add threshold line
    ax.axvline(FUSED_TAU, color=COLORS['danger'], ls='--', lw=2, 
              label=f'τ={FUSED_TAU}', alpha=0.8)
    
    ax.set_xlabel("Fused Score", fontweight='bold')
    ax.set_ylabel("Frequency", fontweight='bold')
    ax.set_title(f"Fused Score Distribution (≥ τ: {frac:.1%})", 
                fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    
    fig.tight_layout()
    return _embed(fig)


def hourly_burden_heatmap(scored: pd.DataFrame) -> str:
    """Hour-of-day vs date heatmap with improved color scheme."""
    if "FusedScore" not in scored.columns or scored.empty:
        return ""
    
    df = scored.copy()
    idx = pd.to_datetime(df.index)
    df["date"] = idx.date
    df["hour"] = idx.hour
    
    pivot = df.pivot_table(index="date", columns="hour", values="FusedScore", aggfunc="mean")
    if pivot.empty:
        return ""
    
    fig, ax = plt.subplots(figsize=(12, max(4, 0.25 * len(pivot))))
    
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", 
                  vmin=0, vmax=1, interpolation='nearest')
    
    # Format axes
    ax.set_yticks(range(0, len(pivot.index), max(1, len(pivot.index) // 20)))
    ax.set_yticklabels([str(pivot.index[i]) for i in range(0, len(pivot.index), 
                                                            max(1, len(pivot.index) // 20))])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    
    ax.set_xlabel("Hour of Day", fontweight='bold')
    ax.set_ylabel("Date", fontweight='bold')
    ax.set_title("Anomaly Burden by Hour (mean FusedScore)", 
                fontweight='bold', pad=10)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Mean Fused Score", fontweight='bold')
    
    fig.tight_layout()
    return _embed(fig)


def contribution_breakdown(scored: pd.DataFrame, n: int = 600) -> str:
    """Stacked area chart of fusion components with improved styling."""
    cols_need = ["H1_Forecast", "H2_Recon", "H3_Contrast", "CorrBoost", "CPD", "FusedScore"]
    if not all(c in scored.columns for c in cols_need):
        return ""
    
    use = scored[cols_need].tail(n)
    if use.empty:
        return ""
    
    # Calculate components
    h1 = 0.45 * use["H1_Forecast"].values
    h2 = 0.35 * use["H2_Recon"].values
    h3 = 0.35 * use["H3_Contrast"].values
    boost = 0.15 * use["CorrBoost"].values + 0.10 * use["CPD"].values
    
    x = np.arange(len(use))
    
    fig, ax = plt.subplots(figsize=(13, 4.5))
    
    # Stack plot with custom colors
    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b']
    ax.stackplot(x, h1, h2, h3, boost, 
                labels=["0.45×H1", "0.35×H2", "0.35×H3", "Boost"],
                colors=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Overlay fused score
    ax.plot(x, use["FusedScore"].values, color='#1f2937', lw=2, 
           label="Fused Score", zorder=5, alpha=0.9)
    
    ax.set_xlabel("Samples", fontweight='bold')
    ax.set_ylabel("Score Contribution", fontweight='bold')
    ax.set_title("Fusion Components (last window)", fontweight='bold', pad=10)
    ax.legend(loc="upper left", framealpha=0.95, ncol=5, fontsize=8)
    ax.set_ylim(0, None)
    
    fig.tight_layout()
    return _embed(fig)


def raw_tag_panels(raw: pd.DataFrame, tags: List[str], 
                  events: Optional[pd.DataFrame] = None, 
                  max_points: int = 1800) -> str:
    """Multi-panel plot of raw tags with optimized layout."""
    use_cols = [t for t in tags if t in raw.columns]
    if not use_cols:
        return "" 

    data = _sample(raw[use_cols].apply(pd.to_numeric, errors='coerce'), max_points)
    ts = pd.to_datetime(data.index)
    
    n_rows = len(use_cols)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, max(2.2 * n_rows, 5)), sharex=True)
    axes = np.atleast_1d(axes)
    
    for i, col in enumerate(use_cols):
        ax = axes[i]
        y = data[col].astype(float)
        
        ax.plot(ts, y.values, lw=1.1, color=COLORS['primary'], 
               alpha=0.85, label=col)
        
        _add_event_spans(ax, events, alpha=0.2)
        
        ax.set_ylabel(col, fontweight='bold', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
    
    axes[-1].set_xlabel("Time", fontweight='bold')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.tight_layout()
    return _embed(fig)
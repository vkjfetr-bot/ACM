"""
report_main.py

Optimized ACM report builder with improved performance and code structure.
"""

import os
from typing import Optional, List, Tuple
from functools import lru_cache
import numpy as np
import pandas as pd

from report_html import wrap_html, section, table, chart, kpi_grid
from report_charts import (
    timeline, sampled_tags_with_marks, drift_bars, regime_share, 
    fused_histogram, hourly_burden_heatmap, contribution_breakdown, 
    raw_tag_panels, FUSED_TAU
)

# ---------- Settings ----------
ART_DIR = os.environ.get("ACM_ART_DIR")
if not ART_DIR:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ART_DIR = os.path.join(root, "acm_artifacts")
FAST = os.environ.get("ACM_FAST_REPORT", "0") == "1"
TITLE = "Asset Condition Monitor — Enhanced Report"
SPARKS_N = 18
DERIVED_TAGS = frozenset({
    "Regime", "FusedScore", "H1_Forecast", "H2_Recon", "H3_Contrast",
    "CorrBoost", "CPD", "ContextMask"
})

# ---------- IO ----------
@lru_cache(maxsize=10)
def _safe_read_csv(path: str, use_index: bool = False, parse_dates_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Cached CSV reader with optimized parameters."""
    if not path or not os.path.exists(path):
        return None
    
    kwargs = {
        'engine': 'c',  # Faster C parser
        'low_memory': False
    }
    
    if use_index:
        kwargs['index_col'] = 0
        kwargs['parse_dates'] = True
    elif parse_dates_col:
        kwargs['parse_dates'] = [parse_dates_col]
    
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}")
        return None

# ---------- Logic ----------
def _downsample_scored(scored: pd.DataFrame) -> pd.DataFrame:
    """Efficiently downsample scored data for faster processing."""
    if not FAST:
        return scored
    
    if pd.api.types.is_datetime64_any_dtype(scored.index):
        return scored.resample("5T").median()
    
    # Numeric index: take every 5th row
    return scored.iloc[::5].copy()

def _pick_key_tags(scored: pd.DataFrame, drift: Optional[pd.DataFrame]) -> List[str]:
    """Select key tags prioritizing drift, with optimized variance calculation."""
    # Filter out derived tags once
    available_tags = [c for c in scored.columns if c not in DERIVED_TAGS]
    
    if drift is not None and {"Tag", "DriftZ"}.issubset(drift.columns):
        # Use drift rankings
        candidates = (
            drift.dropna(subset=["DriftZ"])
            .nlargest(SPARKS_N * 2, "DriftZ")["Tag"]  # Get more candidates for filtering
            .tolist()
        )
        filtered = [c for c in candidates if c in available_tags]
        if filtered:
            return filtered[:SPARKS_N]
    
    # Variance fallback - vectorized operation
    if not available_tags:
        return []
    
    numeric_df = scored[available_tags].apply(pd.to_numeric, errors='coerce')
    # Use numpy for faster variance calculation
    variances = pd.Series(
        np.nanvar(numeric_df.values, axis=0),
        index=numeric_df.columns
    ).nlargest(SPARKS_N)
    
    return variances.index.tolist()

def compute_dq(raw_df: pd.DataFrame, tags: List[str]) -> pd.DataFrame:
    """Optimized data quality computation with vectorized operations."""
    if not tags:
        return pd.DataFrame(columns=["Tag", "Flatline%", "Dropout%", "Spikes"])
    
    # Filter valid tags upfront
    valid_tags = [t for t in tags if t in raw_df.columns]
    if not valid_tags:
        return pd.DataFrame(columns=["Tag", "Flatline%", "Dropout%", "Spikes"])
    
    # Vectorized computation
    numeric_df = raw_df[valid_tags].apply(pd.to_numeric, errors='coerce').astype(float)
    n = len(numeric_df)
    
    results = []
    for tag in valid_tags:
        series = numeric_df[tag]
        
        # Flatline percentage
        flat_pct = (series.diff().abs() < 1e-12).sum() / n * 100
        
        # Dropout percentage
        drop_pct = series.isna().sum() / n * 100
        
        # Spike detection using IQR
        diffs = series.diff().dropna().abs()
        if len(diffs) > 0:
            q25, q75 = diffs.quantile([0.25, 0.75])
            iqr = (q75 - q25) + 1e-9
            spikes = int((diffs > 5 * iqr).sum())
        else:
            spikes = 0
        
        results.append({
            "Tag": tag,
            "Flatline%": flat_pct,
            "Dropout%": drop_pct,
            "Spikes": spikes
        })
    
    return (pd.DataFrame(results)
            .sort_values(["Flatline%", "Dropout%", "Spikes"], 
                        ascending=[False, False, False]))

def _get_top_variance_tags(df: pd.DataFrame, n: int = 6) -> List[str]:
    """Extract top variance tags efficiently."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return []
    
    # Vectorized variance calculation
    return (numeric.var()
            .nlargest(min(n, len(numeric.columns)))
            .index.tolist())

def _format_kpis(scored: pd.DataFrame, masks: Optional[pd.DataFrame], 
                 equip_score: Optional[float]) -> List[Tuple[str, str]]:
    """Format KPI data."""
    kpis = []
    
    if equip_score is not None:
        kpis.append(("Equipment Score", f"{equip_score:.1f}"))
    
    kpis.extend([
        ("Window", f"{scored.index.min()} → {scored.index.max()}"),
        ("Rows", f"{len(scored):,}"),
        ("Regimes", int(scored.get("Regime", pd.Series()).nunique())),
        ("Events", int((scored.get("FusedScore", pd.Series()) >= FUSED_TAU).sum())),
    ])
    
    # Mask coverage
    if masks is not None and not masks.empty and "Mask" in masks.columns:
        mask_pct = 100.0 * masks["Mask"].astype(int).sum() / len(masks)
        kpis.append(("Mask %", f"{mask_pct:.2f}%"))
    else:
        kpis.append(("Mask %", "—"))
    
    return kpis

def explain_block() -> str:
    """Glossary of terms - extracted as constant for clarity."""
    definitions = [
        ("FusedScore", 
         "Final anomaly score combining: H1 (forecast error), H2 (PCA reconstruction error), "
         "H3 (embedding drift). Higher = more anomalous."),
        (f"τ (tau = {FUSED_TAU})", 
         f"Anomaly threshold. Points with FusedScore ≥ {FUSED_TAU} are highlighted as anomalies."),
        ("Regime", 
         "Discrete operating mode (cluster) inferred from data; helps compare behavior within the same state."),
        ("Mask", 
         "Transient periods (e.g., start-up/maintenance) down-weighted in fusion; shown as a thin red band on charts."),
        ("DriftZ", 
         "Z-scored magnitude of a tag distribution shift vs baseline; higher = stronger distribution change."),
    ]
    return table(["Term", "Meaning"], definitions)

# ---------- Report Sections ----------
def _build_simple_trends_section(resampled: Optional[pd.DataFrame], 
                                  events: Optional[pd.DataFrame]) -> str:
    """Build simple trends section."""
    if resampled is None or resampled.empty:
        return ""
    
    simple_tags = _get_top_variance_tags(resampled, n=6)
    if not simple_tags:
        return ""
    
    return section(
        "Simple Trends (Raw Data: Top-Variance Tags)",
        chart(raw_tag_panels(resampled, simple_tags, events))
    )

def _build_events_section(events: Optional[pd.DataFrame]) -> str:
    """Build events table section."""
    if events is None or events.empty:
        return section("Latest Events", "<div class='small'>No events file found.</div>")
    
    recent_events = events.sort_values("End", ascending=False).head(30)
    
    rows = []
    for _, row in recent_events.iterrows():
        duration = pd.to_datetime(row["End"]) - pd.to_datetime(row["Start"])
        peak = f"{float(row.get('PeakScore', np.nan)):.2f}" if pd.notna(row.get('PeakScore')) else "—"
        rows.append([str(row["Start"]), str(row["End"]), peak, str(duration)])
    
    return section(
        "Latest Events",
        table(["Start", "End", "PeakScore", "Duration (hh:mm:ss)"], rows)
    )

def _build_drift_section(drift: Optional[pd.DataFrame]) -> str:
    """Build drift analysis section."""
    if drift is None or drift.empty or not {"Tag", "DriftZ"}.issubset(drift.columns):
        return section("Drift", "<div class='small'>No drift file found.</div>")
    # Only keep the chart; remove tabular section for compactness
    return section("Drift (Top 20)", chart(drift_bars(drift, 20)))

# ---------- Main Build ----------
def build_basic_report() -> str:
    """Build and save the HTML report."""
    # Load artifacts efficiently
    scored = _safe_read_csv(
        os.path.join(ART_DIR, "acm_scored_window.csv"),
        use_index=True
    )
    if scored is None or scored.empty:
        raise FileNotFoundError(
            f"Missing or empty: {os.path.join(ART_DIR, 'acm_scored_window.csv')}"
        )
    
    scored = _downsample_scored(scored)
    
    # Load other artifacts
    events = _safe_read_csv(
        os.path.join(ART_DIR, "acm_events.csv"),
        parse_dates_col="Start"
    )
    if FAST and events is not None:
        events = events.head(5)
    
    drift = _safe_read_csv(os.path.join(ART_DIR, "acm_drift.csv"))
    masks = _safe_read_csv(os.path.join(ART_DIR, "acm_context_masks.csv"))
    resampled = _safe_read_csv(
        os.path.join(ART_DIR, "acm_resampled.csv"),
        use_index=True
    )
    
    # Equipment score
    equip_score = None
    esc = _safe_read_csv(os.path.join(ART_DIR, "acm_equipment_score.csv"))
    if esc is not None and "EquipmentScore" in esc.columns:
        try:
            equip_score = float(esc["EquipmentScore"].iloc[0])
        except (ValueError, IndexError):
            pass
    
    # Build report body
    body = kpi_grid(_format_kpis(scored, masks, equip_score))
    
    # Add sections
    body += _build_simple_trends_section(resampled, events)
    
    body += section(
        "Timeline (Fused + Heads + Regime)",
        chart(timeline(scored, events, masks))
    )
    
    key_tags = _pick_key_tags(scored, drift)
    body += section(
        "Sampled Data (z-normalized) with Anomalies & Events",
        chart(sampled_tags_with_marks(scored, key_tags, events, scored.get("FusedScore")))
    )
    
    # Remove events table section; charts above already indicate anomaly timing
    body += _build_drift_section(drift)
    
    # Advanced visualizations
    body += section("Regime Time Share", chart(regime_share(scored)))
    body += section("Fused Score Distribution", chart(fused_histogram(scored)))
    body += section("When Are Anomalies Higher?", chart(hourly_burden_heatmap(scored)))
    body += section("What Drives the Fused Score?", chart(contribution_breakdown(scored)))
    body += section("Glossary (Model/Analysis Terms)", explain_block())
    
    # Write output
    html = wrap_html(TITLE, body)
    output_path = os.path.join(ART_DIR, "acm_report.html")
    os.makedirs(ART_DIR, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Report: {output_path}")
    return output_path

if __name__ == "__main__":
    build_basic_report()

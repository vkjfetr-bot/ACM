"""Event detection, clustering, and briefing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .detectors import DetectorConfig


@dataclass
class EventConfig:
    min_len_seconds: int
    merge_gap_seconds: int


def fuse_scores(
    pt_score: pd.Series,
    win_score: pd.Series,
    seq_score: pd.Series,
    config: DetectorConfig,
) -> pd.Series:
    w1, w2, w3 = config.fusion_weights
    fused = w1 * pt_score + w2 * win_score + w3 * seq_score
    return fused.clip(0.0, 1.0)


def label_events(
    fused: pd.Series,
    thresholds: Dict[int, float],
    regimes: np.ndarray,
    clean_df: pd.DataFrame,
    *,
    event_cfg: EventConfig,
    head_scores: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """Turn fused scores into anomaly events."""
    alerts = []
    run_start = None
    last_idx = None
    regime_labels = regimes
    min_len = event_cfg.min_len_seconds
    merge_gap = event_cfg.merge_gap_seconds
    times = fused.index.to_pydatetime()
    for idx, (ts, score) in enumerate(fused.items()):
        regime = int(regime_labels[idx]) if idx < len(regime_labels) else 0
        tau = thresholds.get(regime, 0.95)
        if score >= tau:
            if run_start is None:
                run_start = idx
            last_idx = idx
        else:
            if run_start is not None:
                alerts.append((run_start, last_idx, regime))
                run_start = None
    if run_start is not None:
        alerts.append((run_start, last_idx or len(fused) - 1, int(regime_labels[last_idx or -1])))

    events: List[Dict[str, object]] = []
    merged_events: List[Dict[str, object]] = []
    prev_end_time: Optional[pd.Timestamp] = None
    for start_idx, end_idx, regime in alerts:
        start_ts = fused.index[start_idx]
        end_ts = fused.index[end_idx]
        duration = (end_ts - start_ts).total_seconds()
        if duration < min_len:
            continue
        if prev_end_time and (start_ts - prev_end_time).total_seconds() < merge_gap:
            # Merge with previous event
            merged_events[-1]["End"] = end_ts
            merged_events[-1]["PeakScore"] = max(merged_events[-1]["PeakScore"], float(fused.iloc[start_idx:end_idx + 1].max()))
            merged_events[-1]["DurationMin"] = (merged_events[-1]["End"] - merged_events[-1]["Start"]).total_seconds() / 60.0
            continue
        window_scores = fused.iloc[start_idx:end_idx + 1]
        clean_slice = clean_df.loc[window_scores.index]
        z_scores = (clean_slice - clean_slice.mean()) / (clean_slice.std(ddof=0) + 1e-6)
        top_tags = (
            z_scores.abs().mean().sort_values(ascending=False).head(3).index.tolist()
            if not z_scores.empty
            else []
        )
        head_contrib = {}
        if head_scores:
            for head, series in head_scores.items():
                head_contrib[head] = float(series.iloc[start_idx:end_idx + 1].max())
        peak = float(window_scores.max())
        persistent = duration >= 1800 or peak >= 0.98
        merged_events.append(
            {
                "Start": start_ts,
                "End": end_ts,
                "PeakScore": peak,
                "DurationMin": round((end_ts - start_ts).total_seconds() / 60.0, 3),
                "Persistence": "persistent" if persistent else "transient",
                "Regime": int(regime),
                "TopTags": ",".join(top_tags),
                "ContributingHeads": ",".join([k for k, v in head_contrib.items() if v >= 0.6]),
                "TagContrib": json.dumps({tag: round(val, 3) for tag, val in z_scores.abs().mean().head(5).items()}),
                "HeadPeaks": json.dumps(head_contrib),
            }
        )
        prev_end_time = end_ts

    return pd.DataFrame(merged_events)


def assign_event_families(events: pd.DataFrame, latent: np.ndarray, fused: pd.Series) -> pd.DataFrame:
    if events.empty:
        events["Family"] = []
        return events
    vectors = []
    for row in events.itertuples():
        mask = (fused.index >= row.Start) & (fused.index <= row.End)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            vectors.append(np.zeros(latent.shape[1]))
        else:
            vectors.append(latent[idx].mean(axis=0))
    vectors_arr = np.vstack(vectors)
    n_clusters = min(max(2, int(np.sqrt(len(events)))), len(events))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(vectors_arr)
    events = events.copy()
    events["Family"] = [f"F{label}" for label in labels]
    return events

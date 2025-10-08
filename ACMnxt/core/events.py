"""Events — Thresholding, Peak Detection, Episode Merging, and Stats.

Implements MAD-based thresholding, finds peaks, merges close peaks into episodes,
and computes per-event stats (duration, peak score).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


@dataclass
class Event:
    id: int
    start: pd.Timestamp
    end: pd.Timestamp
    duration_s: float
    peak: float


def _mad_threshold(s: pd.Series, z: float = 3.0) -> float:
    med = s.median()
    mad = (s.sub(med)).abs().median() + 1e-9
    return float(med + z * 1.4826 * mad)


def build_events(score: pd.Series, min_gap: int = 5, z: float = 3.0) -> pd.DataFrame:
    """Detect episodes from a score series using MAD threshold.

    Args:
        score: pd.Series indexed by time, values in [0,1].
        min_gap: minimum gap in samples to separate episodes.
        z: robust z multiplier for MAD threshold.

    Returns:
        DataFrame of events with columns [id, start, end, duration_s, peak].
    """
    thr = _mad_threshold(score, z=z)
    above = score >= thr
    # Identify contiguous regions
    episodes: List[Event] = []
    in_run = False
    run_start = None
    last_idx = None
    for ts, flag in above.items():
        if flag and not in_run:
            in_run = True
            run_start = ts
        if in_run and (not flag):
            # close run
            run_end = last_idx or ts
            seg = score.loc[run_start:run_end]
            episodes.append(Event(
                id=len(episodes) + 1,
                start=run_start,
                end=run_end,
                duration_s=float((run_end - run_start).total_seconds()),
                peak=float(seg.max()),
            ))
            in_run = False
        last_idx = ts
    # Tail
    if in_run and run_start is not None and last_idx is not None:
        seg = score.loc[run_start:last_idx]
        episodes.append(Event(
            id=len(episodes) + 1,
            start=run_start,
            end=last_idx,
            duration_s=float((last_idx - run_start).total_seconds()),
            peak=float(seg.max()),
        ))
    # Merge episodes that are too close
    merged: List[Event] = []
    for ev in episodes:
        if not merged:
            merged.append(ev)
            continue
        gap = (ev.start - merged[-1].end).total_seconds()
        if gap <= min_gap:
            # merge
            prev = merged.pop()
            merged.append(Event(
                id=prev.id,
                start=prev.start,
                end=max(prev.end, ev.end),
                duration_s=float((max(prev.end, ev.end) - prev.start).total_seconds()),
                peak=max(prev.peak, ev.peak),
            ))
        else:
            merged.append(ev)
    df = pd.DataFrame([e.__dict__ for e in merged])
    if not df.empty:
        df = df.sort_values(["start"]).reset_index(drop=True)
        df["id"] = np.arange(1, len(df) + 1)
    return df


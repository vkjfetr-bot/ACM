from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd


def _mad_threshold(s: pd.Series, z: float = 3.0) -> float:
    med = float(s.median())
    mad = float((s - med).abs().median() + 1e-9)
    return med + z * 1.4826 * mad


def build_events(score: pd.Series, min_gap: int = 5, z: float = 3.0) -> pd.DataFrame:
    thr = _mad_threshold(pd.to_numeric(score, errors="coerce"), z=z)
    above = score >= thr
    events: List[dict] = []
    in_run = False
    run_start = None
    last_ts = None
    for ts, flag in above.items():
        if flag and not in_run:
            in_run = True; run_start = ts
        if in_run and not flag:
            run_end = last_ts or ts
            seg = score.loc[run_start:run_end]
            events.append({
                "id": len(events)+1,
                "Start": run_start,
                "End": run_end,
                "Duration_s": float((run_end - run_start).total_seconds()) if isinstance(score.index, pd.DatetimeIndex) else 0.0,
                "Peak": float(pd.to_numeric(seg, errors="coerce").max())
            })
            in_run = False
        last_ts = ts
    if in_run and run_start is not None and last_ts is not None:
        seg = score.loc[run_start:last_ts]
        events.append({
            "id": len(events)+1,
            "Start": run_start,
            "End": last_ts,
            "Duration_s": float((last_ts - run_start).total_seconds()) if isinstance(score.index, pd.DatetimeIndex) else 0.0,
            "Peak": float(pd.to_numeric(seg, errors="coerce").max())
        })
    df = pd.DataFrame(events)
    return df


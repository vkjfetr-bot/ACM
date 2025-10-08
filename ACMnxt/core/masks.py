"""Masks — Maintenance, Startup/Shutdown, and DQ-bad Windows.

Combines known maintenance windows, startup/shutdown heuristics, and
data-quality-derived gaps into a single boolean mask series per category.
"""
from __future__ import annotations

import pandas as pd


def make_masks(df: pd.DataFrame, dq: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute masks as boolean columns indexed like df.

    Returns columns: maintenance, startup, dq_bad, any_mask
    """
    idx = df.index
    m_maint = pd.Series(False, index=idx, name="maintenance")
    m_start = pd.Series(False, index=idx, name="startup")
    m_dq = pd.Series(False, index=idx, name="dq_bad")
    if dq is not None and "dq_bad" in dq:
        m_dq = dq["dq_bad"].reindex(idx, fill_value=False)
    any_mask = (m_maint | m_start | m_dq).rename("any_mask")
    return pd.concat([m_maint, m_start, m_dq, any_mask], axis=1)


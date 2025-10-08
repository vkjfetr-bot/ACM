import pandas as pd
import numpy as np

from acmnxt.core.dq import clean_time, resample_numeric, compute_dq


def _mk_df():
    idx = pd.date_range('2024-01-01', periods=10, freq='T', tz='UTC')
    a = pd.Series([1,1,1,1,1, 2,2,2,2,2], index=idx)  # flat then step
    b = pd.Series(np.arange(10, dtype=float), index=idx)
    b.iloc[3] = np.nan
    b.iloc[4] = np.nan
    c = pd.Series(np.random.randn(10), index=idx)
    return pd.DataFrame({'A': a, 'B': b, 'C': c})


def test_clean_time_sorts_and_dedups():
    df = _mk_df()
    df2 = pd.concat([df, df.iloc[[5]]])
    df2.index = list(df.index) + [df.index[5]]
    out = clean_time(df2)
    assert out.index.is_monotonic_increasing
    assert out.index.duplicated().sum() == 0


def test_resample_numeric_rule():
    df = _mk_df()
    out = resample_numeric(df, '2T')
    # 10 minutes -> 5 bins
    assert len(out) == 5
    assert set(out.columns) == {'A','B','C'}


def test_compute_dq_metrics_and_mask():
    df = _mk_df()
    metrics, dq_bad = compute_dq(df)
    assert {'nan_pct','flatline_ratio','spike_ratio','dropout_runs','dq_flag'}.issubset(metrics.columns)
    # Expect some NaNs in B
    assert metrics.loc['B','nan_pct'] > 0
    # dq_bad mask should be a boolean series aligned to index
    assert dq_bad.index.equals(df.index)
    assert dq_bad.dtype == bool


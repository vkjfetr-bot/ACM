import numpy as np
import pandas as pd

from core import regimes


def test_smoothing_order_preserves_length_and_labels():
    labels = np.array([0, 1, 1, 2, 2, 2, 1, 0], dtype=int)
    ts = pd.date_range("2024-01-01", periods=len(labels), freq="H")

    smoothed = regimes.smooth_labels(labels, passes=1, window=3)
    transitioned = regimes.smooth_transitions(
        smoothed,
        timestamps=ts,
        min_dwell_samples=2,
        min_dwell_seconds=None,
    )

    assert len(transitioned) == len(labels)
    assert set(np.unique(transitioned)).issubset(set(np.unique(labels)))


def test_smoothing_commutes_with_dwell_seconds():
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 1], dtype=int)
    ts = pd.date_range("2024-01-01", periods=len(labels), freq="H")

    first = regimes.smooth_transitions(
        regimes.smooth_labels(labels, passes=1, window=3),
        timestamps=ts,
        min_dwell_samples=0,
        min_dwell_seconds=3600 * 2,
    )
    second = regimes.smooth_labels(
        regimes.smooth_transitions(
            labels,
            timestamps=ts,
            min_dwell_samples=0,
            min_dwell_seconds=3600 * 2,
        ),
        passes=1,
        window=3,
    )

    assert len(first) == len(second) == len(labels)
    assert np.array_equal(first, second)

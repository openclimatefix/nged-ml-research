"""Tests of the pure table functions of the A1 diagnostic."""

from __future__ import annotations

import e00738_a1_diagnostic as diag
import numpy as np
import pytest


def test_auc_separates_and_ties() -> None:
    """A perfect split scores 1, identical scores score 0.5, and an empty side is NaN."""
    assert diag.auc(positive=np.array([2.0, 3.0]), negative=np.array([0.0, 1.0])) == 1.0
    assert diag.auc(positive=np.ones(3), negative=np.ones(4)) == pytest.approx(0.5)
    assert np.isnan(diag.auc(positive=np.array([]), negative=np.ones(2)))


def test_band_mask_is_half_open() -> None:
    """A value on the upper bound belongs to the next band."""
    v = np.array([0.5, 1.0, 2.0])
    assert diag.band_mask(values=v, lo=1.0, hi=2.0).tolist() == [False, True, False]


def test_depth_law_shares_sum_to_one() -> None:
    """Every shift falls in exactly one depth band."""
    out = diag.depth_law(depth_share=np.array([0.01, 0.05, 0.2, 0.5]), std=np.ones(4), noise=0.25)
    assert sum(out["share_by_band"].values()) == pytest.approx(1.0)
    assert out["n_shifts"] == 4


def test_tables_see_a_deep_recent_shift() -> None:
    """A step of 10 noise units in the newest 5 rows is seen at any z of the grid up to 8."""
    rng = np.random.default_rng(0)
    n = 40
    power = rng.normal(0.0, 0.01, (n, 14, 48))
    power[: n // 2, :5] += 2.5
    visible = np.ones_like(power, dtype=bool)
    true_age = np.where(np.arange(n) < n // 2, 6.0, -1.0)
    depth = np.where(np.arange(n) < n // 2, 10.0, 0.0)
    out = diag.tables(
        inj=(power, visible),
        real=(rng.normal(0.0, 0.01, (n, 14, 48)), visible),
        true_age=true_age,
        depth_noise=depth,
        noise=0.25,
        z_main=3.5,
        grid=(2.0, 8.0),
    )
    assert out["miss_by_age_and_depth"]["6-10"]["over 4"]["miss"] == 0.0
    assert out["curve"][0]["false_alarm_real"] == 0.0

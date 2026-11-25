# ruff: noqa: E501
"""Known-answer tests of the scoring and the two-stage resampler on synthetic arrays."""

import datetime

import e00721_core as core
import e00721_score as score
import numpy as np
import polars as pl
import pytest


def test_crps_of_a_point_mass_is_the_absolute_error() -> None:
    q = np.tile(np.array([[0.5]]), (1, len(score.TAUS)))
    assert score.crps_points(q=q, actual=np.array([0.7]))[0] == pytest.approx(20.0, abs=1e-6)
    assert score.crps_points(q=q, actual=np.array([0.5]))[0] == pytest.approx(0.0, abs=1e-6)


def _frame(
    *, seed_values: dict[int, float], n_days: int = 40, truth: float = -0.02
) -> pl.DataFrame:
    rows = []
    for seed, shift in seed_values.items():
        for fold in ("A", "B"):
            for d in range(n_days):
                rows.append({  # noqa: PERF401
                    "time_series_id": 1, "power_fcst_init_time": d, "valid_time": datetime.datetime(2026, 1, 1) + datetime.timedelta(days=d),
                    "fold": fold, "seed": seed, "actual": truth, "crps": 1.0 + shift, "mae": 1.0 + shift,
                })  # fmt: skip
    return pl.DataFrame(rows).with_columns(pl.col("valid_time").cast(pl.Datetime("us")))


def test_pair_arms_refuses_different_row_sets_and_truths() -> None:
    a = _frame(seed_values={0: 0.0})
    with pytest.raises(ValueError, match="different row sets"):
        score.pair_arms(a=a, b=a.head(10))
    with pytest.raises(ValueError, match="different truths"):
        score.pair_arms(a=a, b=a.with_columns(actual=pl.col("actual") + 0.5))


def test_a_constant_gain_gives_that_point_and_a_narrow_interval() -> None:
    a = _frame(seed_values={0: -0.5, 1: -0.5})
    b = _frame(seed_values={0: 0.0, 1: 0.0})
    got = score.contrast(a=a, b=b, seeds=(0, 1), n_resamples=2000)
    assert got.point == pytest.approx(-0.5)
    assert got.upper < 0.0
    assert got.n_units == 8
    assert got.units_below_minimum


def test_block_ids_count_ten_dates_from_each_folds_first_date() -> None:
    dates = np.arange("2026-01-01", "2026-01-31", dtype="datetime64[D]")
    ids = score.block_ids(valid_date=dates, fold=np.array(["A"] * 30))
    assert ids.tolist() == [0] * 10 + [1] * 10 + [2] * 10


def test_a_null_contrast_covers_zero_about_as_often_as_the_level() -> None:
    rng = np.random.default_rng(1)
    hits = 0
    for trial in range(60):
        a_vals = {s: float(rng.normal(0, 0.1)) for s in (0, 1)}
        a = _frame(seed_values={0: 0.0, 1: 0.0}).with_columns(
            crps=pl.col("crps") + pl.Series(rng.normal(0, 0.3, 160))
        )
        b = a.with_columns(crps=pl.col("crps") + pl.Series(rng.normal(0, 0.3, 160)))
        got = score.contrast(a=a, b=b, seeds=(0, 1), n_resamples=400)
        hits += int(got.lower <= 0.0 <= got.upper)
        del a_vals, trial
    assert hits >= 55


def test_the_widening_applies_to_the_reading() -> None:
    a = _frame(seed_values={0: -0.1}).with_columns(
        crps=pl.col("crps") + pl.Series(np.random.default_rng(3).normal(0, 0.3, 80))
    )
    b = _frame(seed_values={0: 0.0})
    got = score.contrast(a=a, b=b, seeds=(0,), n_resamples=1000)
    assert got.lower <= got.raw_lower
    assert got.upper >= got.raw_upper


def test_auc_and_the_noise_check() -> None:
    assert score.auc(
        score=np.array([0.9, 0.8, 0.1, 0.2]), positive=np.array([True, True, False, False])
    ) == pytest.approx(1.0)
    assert score.auc(
        score=np.array([0.5, 0.5, 0.5, 0.5]), positive=np.array([True, True, False, False])
    ) == pytest.approx(0.5)
    se = score.hanley_mcneil_se(area=0.9, n_pos=50, n_neg=50)
    assert 0.0 < se < 0.1
    assert score.hanley_mcneil_se(area=0.9, n_pos=200, n_neg=200) < se


def test_sign_accuracy_and_false_call_rate() -> None:
    rho = np.array([0.9, 0.8, 0.1, 0.6])
    reverse = np.array([True, True, False, False])
    assert score.sign_accuracy(rho=rho, reverse=reverse) == pytest.approx(0.75)
    assert score.false_call_rate(rho=rho[~reverse]) == pytest.approx(0.5)


def test_share_of_gap_and_the_g3_assignments() -> None:
    assert score.share_of_gap(p0=2.0, s=1.0, arm=1.2) == pytest.approx(0.8)
    assigns = score.null_assignments()
    assert len(assigns) == 60
    assert score.g3_joint_pass_rate(plain=[True, True], open_=[True, False]) in (0.0, 0.5, 1.0)
    assert pytest.approx(0.80) == core.G3_POWER_TARGET

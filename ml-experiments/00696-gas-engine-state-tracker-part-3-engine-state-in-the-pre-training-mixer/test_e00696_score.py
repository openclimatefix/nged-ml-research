# ruff: noqa: E501, PERF401
"""Experiment 696's contrast machinery on synthetic losses with a known difference."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from e00696_score import decision_contrast, interval, month_bootstrap
from studies.experiment_store import losses_for_bootstrap

MONTHS = [datetime(2025, 11, 1, tzinfo=UTC) + timedelta(days=31 * i) for i in range(8)]


def _frame(*, shift: float, seed_sd: float = 0.05) -> pl.DataFrame:
    """Return two arms' losses over 8 months, 5 origins a month, 4 valid times each, and 10 seeds.

    Arm `T` equals arm `R` plus `shift` and independent noise per seed and row.
    """
    rng = np.random.default_rng(0)
    rows = []
    for month in MONTHS:
        for origin_day in range(5):
            origin = month + timedelta(days=origin_day, hours=12)
            for step in range(4):
                valid = origin + timedelta(hours=24 + step)
                base = rng.normal(5.0, 0.5)
                for seed in range(10):
                    rows.append((origin, valid, seed, base + rng.normal(0, seed_sd), shift))
    frame = pl.DataFrame(
        rows,
        schema=["power_fcst_init_time", "valid_time", "seed", "crps", "shift"],
        orient="row",
    ).with_columns(time_series_id=pl.lit(1, dtype=pl.Int64), seed=pl.col("seed").cast(pl.Int64))
    reference = frame.drop("shift")
    treatment = frame.with_columns(
        crps=pl.col("crps") + pl.col("shift") + pl.Series(rng.normal(0, seed_sd, frame.height))
    ).drop("shift")
    return losses_for_bootstrap(arm_losses={"R": reference, "T": treatment})


def test_a_clear_gain_passes_all_three_decision_intervals() -> None:
    """A shift of -0.3 against noise of 0.05 gives intervals below zero and an estimate near -0.3."""
    result = decision_contrast(
        frame=_frame(shift=-0.3), treatment="T", reference="R", metric="crps", level=98.75,
        resamples=4_000,
    )  # fmt: skip
    assert result["passes"]
    assert not result["all_above_zero"]
    assert abs(result["estimate"] + 0.3) < 0.05
    assert result["n_months"] == 8
    assert result["n_seeds"] == 10
    for low, high in (result["t_month"], result["bootstrap_month"], result["t_seed"]):
        assert low < high < 0.0


def test_no_difference_does_not_pass() -> None:
    """A shift of zero leaves at least one interval across zero."""
    result = decision_contrast(
        frame=_frame(shift=0.0), treatment="T", reference="R", metric="crps", level=98.75,
        resamples=4_000, library=False,
    )  # fmt: skip
    assert not result["passes"]
    assert not result["all_above_zero"]


def test_a_clear_loss_is_above_zero_in_all_three_intervals() -> None:
    """A shift of +0.3 puts every interval above zero."""
    result = decision_contrast(
        frame=_frame(shift=0.3), treatment="T", reference="R", metric="crps", level=98.75,
        resamples=4_000, library=False,
    )  # fmt: skip
    assert result["all_above_zero"]
    assert not result["passes"]


def test_month_bootstrap_covers_the_mean_and_widens_with_level() -> None:
    """The percentile interval of whole-month resamples contains the row-weighted mean, wider at a higher level."""
    sums = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 2.0])
    rows = np.full(8, 10.0)
    narrow = month_bootstrap(month_sums=sums, month_rows=rows, level=90.0, resamples=20_000)
    wide = month_bootstrap(month_sums=sums, month_rows=rows, level=98.75, resamples=20_000)
    mean = sums.sum() / rows.sum()
    assert narrow[0] < mean < narrow[1]
    assert wide[0] <= narrow[0]
    assert wide[1] >= narrow[1]


def test_interval_is_the_paired_t_interval() -> None:
    """Eight values of mean 1 and sample standard deviation 2 give mean plus or minus t(0.975, 7) * 2 / sqrt(8)."""
    values = np.array([-1.0, 3.0, -1.0, 3.0, -1.0, 3.0, -1.0, 3.0])
    low, high = interval(values=values, level=95.0)
    half = 2.364624 * values.std(ddof=1) / np.sqrt(8)
    assert abs(low - (1.0 - half)) < 1e-4
    assert abs(high - (1.0 + half)) < 1e-4

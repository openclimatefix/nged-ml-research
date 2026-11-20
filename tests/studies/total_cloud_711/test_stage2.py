# ruff: noqa: ANN202
"""Tests of the stage-2 arms, job list, row builders, and scorer of experiment 711 (no torch)."""

import re
from datetime import UTC, datetime

import e00711_arms as arms
import e00711_jobs as jobs
import e00711_rowcloud as rowcloud
import e00711_score as score
import numpy as np
import polars as pl
import pytest

US_PER_DAY = 86_400_000_000


# --- arms and jobs ----------------------------------------------------------------------------


def test_job_list_matches_the_plan_totals():
    planned = jobs.plan_jobs()
    jobs.check_counts(jobs=planned)
    assert len(planned) == 946


def test_set_names_follow_the_pattern_and_are_unique():
    planned = jobs.plan_jobs()
    names = [j.set_name for j in planned]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z0-9]+-\d{4}-\d{2}-s\d", n) for n in names)
    assert arms.set_name(arm="D1p", month="2026-06", seed=0) == "d1p-2026-06-s0"


def test_arm_table_is_self_consistent():
    arms.check_table()


def test_wall_clock_estimate_is_two_streams():
    planned = jobs.plan_jobs()
    low, high = jobs.stream_hours(jobs=planned)
    wall_low, wall_high = jobs.wall_hours(jobs=planned)
    assert wall_low == pytest.approx(low / 2)
    assert wall_high == pytest.approx(high / 2)


def test_pending_drops_finished_sets():
    planned = jobs.plan_jobs()
    done = {planned[0].set_name}
    assert len(jobs.pending(jobs=planned, finished=done)) == len(planned) - 1


# --- row builders -----------------------------------------------------------------------------


def _run_day(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1e6) // US_PER_DAY


def test_row_steps_and_run_days():
    steps = rowcloud.row_steps(lead_hours=np.array([0.5, 1.0, 351.0]))
    assert steps.tolist() == [0, 1, 701]
    with pytest.raises(ValueError, match="half-hour"):
        rowcloud.row_steps(lead_hours=np.array([0.7]))
    with pytest.raises(ValueError, match="00:00"):
        rowcloud.row_run_days(nwp_init=np.array([3_600_000_000]))


def test_missing_flag_follows_the_run_date_alone():
    days = np.array([_run_day(2024, 3, 31), _run_day(2024, 4, 1), _run_day(2025, 11, 20)])
    days = np.append(days, _run_day(2025, 11, 21))
    flag = rowcloud.missing_flag(run_days=days)
    assert flag.tolist() == [False, True, True, False]
    with pytest.raises(ValueError, match="outside"):
        rowcloud.missing_flag(run_days=np.array([_run_day(2026, 7, 5)]))


def test_drop_gap_rows_keeps_only_runs_with_cloud():
    init = np.array([_run_day(2024, 3, 31), _run_day(2024, 5, 1)]) * US_PER_DAY
    assert rowcloud.drop_gap_rows(nwp_init=init).tolist() == [True, False]


def test_daylight_mean_uses_only_daylight_of_the_same_local_day():
    cloud = np.array([0.0, 1.0, 0.5, 0.2, 0.8])
    day = np.array([0, 0, 0, 1, 1])
    daylight = np.array([False, True, True, False, False])
    out = rowcloud.daylight_mean_by_step(cloud_mean=cloud, local_day=day, daylight=daylight)
    assert out[:3] == pytest.approx([0.75, 0.75, 0.75])
    assert out[3:] == pytest.approx([0.2, 0.8])
    nan_out = rowcloud.daylight_mean_by_step(
        cloud_mean=np.full(5, np.nan), local_day=day, daylight=daylight
    )
    assert np.isnan(nan_out).all()


# --- the decision rule ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (0.01, 0.05, "refuted"),
        (-0.2, -0.05, "supported"),
        (-0.2, -0.01, "small gain"),
        (-0.01, 0.01, "not supported"),
        (-0.02, 0.0, "not supported"),
        (-0.1, 0.01, "inconclusive"),
        (0.0, 0.01, "not supported"),
    ],
)
def test_verdict_in_the_plan_order(lower: float, upper: float, expected: str):
    assert score.verdict(lower=lower, upper=upper) == expected


def test_group_verdict_modifiers():
    base = score.GroupReading(
        c1="inconclusive", c2="not supported", c3="supported", c3_lower=-0.2, control_passed=True
    )
    assert score.group_verdict(reading=base)[0] == "partly supported"
    audited = score.GroupReading(
        c1="supported", c2="small gain", c3="refuted", c3_lower=0.1, control_passed=True
    )
    verdict, fired = score.group_verdict(reading=audited)
    assert verdict == "inconclusive"
    assert fired
    failed = score.GroupReading(
        c1="supported", c2="supported", c3="supported", c3_lower=-1.0, control_passed=False
    )
    assert score.group_verdict(reading=failed)[0] == "inconclusive"
    plain = score.GroupReading(
        c1="small gain", c2="supported", c3=None, c3_lower=None, control_passed=True
    )
    assert score.group_verdict(reading=plain)[0] == "supported"


def test_front_matter_verdict():
    f = score.front_matter_verdict
    assert f(six="supported", eight="supported", solar="supported") == "supported"
    assert f(six="supported", eight="not supported", solar="refuted") == "partly supported"
    assert f(six="refuted", eight="not supported", solar="inconclusive") == "refuted"
    assert f(six="not supported", eight="not supported", solar="not supported") == "not supported"
    assert f(six="inconclusive", eight="not supported", solar="inconclusive") == "inconclusive"


def test_planned_level_is_corrected_for_nine_readings():
    assert pytest.approx(100 * (1 - 0.05 / 9)) == score.PLANNED_LEVEL


# --- losses and contrasts on synthetic frames -------------------------------------------------


def _forecast(*, n_months: int, shift: float, seeds: tuple[int, ...], rows_per_month: int = 20):
    rng = np.random.default_rng(1)
    pieces = []
    for seed in seeds:
        for month in range(n_months):
            init = datetime(2025, 7 + month % 6, 1, tzinfo=UTC)
            valid = [
                datetime(2025, 7 + month % 6, 1, 9 + i // 2, 30 * (i % 2), tzinfo=UTC)
                for i in range(rows_per_month)
            ]
            base = rng.normal(0.5, 0.01, rows_per_month)
            data = {c: base + (k - 5) * 0.03 + shift for k, c in enumerate(score.LEVEL_COLUMNS)}
            pieces.append(
                pl.DataFrame(
                    {
                        "time_series_id": [1] * rows_per_month,
                        "power_fcst_init_time": [init] * rows_per_month,
                        "valid_time": valid,
                        "seed": [seed] * rows_per_month,
                        **data,
                    }
                ).with_columns(pl.col("power_fcst_init_time").dt.cast_time_unit("us"))
            )
    return pl.concat(pieces)


def test_widen_moves_only_the_outer_levels():
    frame = _forecast(n_months=1, shift=0.0, seeds=(0,))
    wide = score.widen(frame=frame)
    assert (wide["q50"] == frame["q50"]).all()
    assert (wide["q10"] == frame["q10"]).all()
    spread = (frame["q99"] - frame["q50"]).to_numpy()
    assert (wide["q99"] - wide["q50"]).to_numpy() == pytest.approx(1.15 * spread)


def test_row_losses_for_a_median_that_hits_the_actual():
    frame = _forecast(n_months=1, shift=0.0, seeds=(0,))
    actual = frame["q50"].to_numpy()
    losses = score.row_losses(frame=frame, actual=actual)
    assert (losses["mae_median"] == 0).all()
    assert (losses["crps"] > 0).all()
    assert losses["below_q99"].sum() == losses.height


def test_regime_expression_labels_half_hours():
    frame = pl.DataFrame(
        {"csi": [0.9, 0.2, 0.6, 0.9, 0.9], "elevation_deg": [30.0, 30.0, 30.0, 5.0, -3.0]}
    )
    out = frame.with_columns(
        r=score.regime_expression(csi=pl.col("csi"), elevation=pl.col("elevation_deg"))
    )
    assert out["r"].to_list() == ["sunny", "overcast", "between", "low sun", "night"]

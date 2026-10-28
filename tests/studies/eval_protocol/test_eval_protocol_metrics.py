"""Known-answer tests of the evaluation protocol's metrics, scoring table, bootstrap, and cost.

A forecast whose 11 quantiles are all one value `f` has a quantile score of exactly `|actual - f|`,
because the levels average to 0.5 and the pinball loss weights the two sides by `tau` and `1 - tau`.
Most expected values below follow from that identity and from hand-written sums.
"""

from datetime import UTC, datetime, timedelta
from typing import TypedDict

import numpy as np
import polars as pl
import pytest
from studies.eval_protocol.metrics import (
    HEADLINE_LEVEL_PERCENT,
    HEADLINE_RESAMPLES,
    HOLDOUT_START,
    LEAD_BAND_SLICES,
    LEVELS,
    METRIC_NAMES,
    QUANTILE_COLUMNS,
    REPORTED_LEVEL_PERCENT,
    REPORTED_RESAMPLES,
    STATUS_NOT_FITTED,
    STATUS_SHORT_HISTORY,
    STATUS_UNSIGNED_METER,
    THIN_HALF_HOURS,
    avoidable_curtailment_cost,
    headline_month_bootstrap_contrast,
    month_bootstrap_contrast,
    primaries_won,
    quantile_score,
    reported_contrast_by_primary,
    restrict_to_slice_table,
    score_forecast,
    scored_rows,
    shared_slice_table,
    tail_weighted_score,
    twcrps_thresholds,
)


class ScoringInputs(TypedDict):
    """The frames that `score_forecast` takes beside the thresholds."""

    forecast: pl.DataFrame
    actuals: pl.DataFrame
    series: pl.DataFrame
    slice_table: pl.DataFrame


class CurtailmentInputs(TypedDict):
    """The inputs of `avoidable_curtailment_cost` other than the arms."""

    forecast: pl.DataFrame
    flows: pl.DataFrame
    series: pl.DataFrame
    scored_start: datetime


def utc(*args: int) -> datetime:
    """Return a UTC datetime."""
    return datetime(*args, tzinfo=UTC)


def constant_forecast(
    *, series: int, init: datetime, hours: list[float], actual: list[float], offset: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return a forecast whose quantiles all equal `actual + offset`, and the actuals."""
    valid = [init + timedelta(hours=h) for h in hours]
    forecast = pl.DataFrame(
        {
            "time_series_id": [series] * len(hours),
            "power_fcst_init_time": [init] * len(hours),
            "valid_time": valid,
            **{q: [a + offset for a in actual] for q in QUANTILE_COLUMNS},
        }
    )
    actuals = pl.DataFrame(
        {"time_series_id": [series] * len(hours), "valid_time": valid, "actual": actual}
    )
    return forecast, actuals


def context_for(*, forecast: pl.DataFrame) -> pl.DataFrame:
    """Return a context table of mild daylight conditions for every row of a forecast."""
    return forecast.select(
        "time_series_id",
        "valid_time",
        temperature_c=pl.lit(10.0),
        elevation_deg=pl.lit(30.0),
        pv_unit_shape=pl.lit(0.5),
    )


def thresholds_of(values: dict[int, float | None]) -> pl.DataFrame:
    """Return a threshold table in the shape of `twcrps_thresholds`."""
    return pl.DataFrame(
        {
            "time_series_id": list(values),
            "threshold_points": list(values.values()),
            "status": ["ok" if v is not None else STATUS_SHORT_HISTORY for v in values.values()],
        },
        schema={"time_series_id": pl.Int64, "threshold_points": pl.Float64, "status": pl.String},
    )


def slice_table_of(*, forecasts: dict[str, pl.DataFrame], actuals: pl.DataFrame) -> pl.DataFrame:
    """Return the shared slice table of arms whose context is mild daylight."""
    context = context_for(forecast=pl.concat(list(forecasts.values())))
    return shared_slice_table(forecasts=forecasts, actuals=actuals, context=context)


def test_quantile_score_of_a_constant_forecast_is_the_absolute_error() -> None:
    """The 11 levels average to 0.5, so a constant forecast scores `|actual - f|` on either side."""
    assert LEVELS.mean() == pytest.approx(0.5)
    actual = np.array([10.0, 0.0, 10.0, 7.0])
    quantiles = np.array([[0.0] * 11, [10.0] * 11, [4.0] * 11, [7.0] * 11])
    np.testing.assert_allclose(quantile_score(actual=actual, quantiles=quantiles), [10, 10, 6, 0])


def test_quantile_score_of_a_symmetric_spread_matches_a_hand_sum() -> None:
    """Quantiles minus 5 to 5 around an actual of 0 sum to 2.9 of pinball loss over 11 levels."""
    quantiles = np.arange(-5.0, 6.0)[None, :]
    score = quantile_score(actual=np.array([0.0]), quantiles=quantiles)
    assert score[0] == pytest.approx(2 * 2.9 / 11)


def test_tail_weighted_score_scores_only_what_happens_above_the_threshold() -> None:
    """Below the threshold everything is raised to it and the score is zero; above it is scored."""
    quantiles = np.array([[10.0] * 11, [10.0] * 11, [6.0] * 11])
    actual = np.array([5.0, 12.0, 5.0])
    threshold = np.array([8.0, 8.0, 8.0])
    # Row 1: 8 against 10 is an error of 2. Row 2: 12 against 10 is 2. Row 3: 8 against 8 is 0.
    np.testing.assert_allclose(
        tail_weighted_score(actual=actual, quantiles=quantiles, threshold=threshold), [2, 2, 0]
    )


def test_thresholds_are_the_95th_percentile_of_earlier_power_in_points() -> None:
    """Readings from the cut onward, such as one huge value, never move the threshold."""
    times = [utc(2024, 1, 1) + timedelta(hours=k) for k in range(101)]
    history = pl.DataFrame(
        {
            "time_series_id": [3] * 102,
            "valid_time": [*times, utc(2025, 8, 1)],
            "power_mw": [float(k) for k in range(101)] + [1000.0],
        }
    )
    series = pl.DataFrame({"time_series_id": [3], "label": ["D5"], "capacity_mw": [50.0]})
    thresholds = twcrps_thresholds(history=history, series=series, before=utc(2025, 7, 1))
    assert thresholds["threshold_points"].to_list() == [pytest.approx(95.0 / 50.0 * 100.0)]
    assert thresholds["status"].to_list() == ["ok"]
    ninety_nine = twcrps_thresholds(
        history=history, series=series, before=utc(2025, 7, 1), quantile=0.99
    )
    assert ninety_nine["threshold_points"].to_list() == [pytest.approx(99.0 / 50.0 * 100.0)]


def test_threshold_needs_12_months_of_earlier_readings_and_is_null_with_a_status_otherwise() -> (
    None
):
    """A first reading exactly 12 months before the cut is enough; one day later is not."""
    series = pl.DataFrame({"time_series_id": [3], "label": ["D5"], "capacity_mw": [50.0]})
    before = utc(2025, 7, 1)

    def table(first: datetime) -> pl.DataFrame:
        history = pl.DataFrame(
            {
                "time_series_id": [3, 3],
                "valid_time": [first, utc(2025, 6, 1)],
                "power_mw": [1.0, 2.0],
            }
        )
        return twcrps_thresholds(history=history, series=series, before=before)

    enough = table(utc(2024, 7, 1))
    assert enough["status"].to_list() == ["ok"]
    short = table(utc(2024, 7, 2))
    assert short["threshold_points"].to_list() == [None]
    assert short["status"].to_list() == [STATUS_SHORT_HISTORY]
    absent = twcrps_thresholds(
        history=pl.DataFrame(
            schema={
                "time_series_id": pl.Int64,
                "valid_time": pl.Datetime("us", "UTC"),
                "power_mw": pl.Float64,
            }
        ),
        series=series,
        before=before,
    )
    assert absent["status"].to_list() == [STATUS_SHORT_HISTORY]


def test_a_null_threshold_gives_a_null_twcrps_with_a_status_and_scores_the_rest() -> None:
    """The series whose threshold is undefined keeps its other metrics."""
    inputs = build_two_series()
    table = score_forecast(**inputs, thresholds=thresholds_of({1: None, 2: 1000.0}), slices=["all"])
    twcrps = {
        r["label"]: r for r in table.filter(pl.col("metric") == "twcrps").iter_rows(named=True)
    }
    assert twcrps["D5"]["value"] is None
    assert twcrps["D5"]["status"] == STATUS_SHORT_HISTORY
    assert twcrps["D2"]["status"] == "scored"
    crps = table.filter(pl.col("metric") == "crps", pl.col("label") == "D5")
    assert crps["status"].to_list() == ["scored"]
    assert crps["value"].to_list() == [pytest.approx(10.0)]


def build_two_series() -> ScoringInputs:
    """Return a forecast of a signed-MW series (150 rows) and a MVA-only series (50 rows)."""
    init = utc(2026, 1, 5)
    hours_a = [0.5 * k for k in range(1, 151)]
    hours_b = [0.5 * k for k in range(1, 51)]
    forecast_a, actuals_a = constant_forecast(
        series=1, init=init, hours=hours_a, actual=[20.0] * 150, offset=5.0
    )
    forecast_b, actuals_b = constant_forecast(
        series=2, init=init, hours=hours_b, actual=[10.0] * 50, offset=-2.0
    )
    forecast = pl.concat([forecast_a, forecast_b])
    actuals = pl.concat([actuals_a, actuals_b])
    return ScoringInputs(
        forecast=forecast,
        actuals=actuals,
        series=pl.DataFrame(
            {"time_series_id": [1, 2], "label": ["D5", "D2"], "capacity_mw": [50.0, 40.0]}
        ),
        slice_table=slice_table_of(forecasts={"arm": forecast}, actuals=actuals),
    )


def test_score_forecast_returns_the_known_scores_in_points_with_megawatts_beside() -> None:
    """A forecast 5 MW off at a 50 MW series is 10 points of CRPS; 2 MW off at 40 MW is 5."""
    inputs = build_two_series()
    table = score_forecast(
        **inputs, thresholds=thresholds_of({1: 1000.0, 2: 1000.0}), slices=["all"]
    )
    rows = {(r["metric"], r["label"]): r for r in table.iter_rows(named=True)}
    crps_d5 = rows["crps", "D5"]
    assert (crps_d5["value"], crps_d5["unit"], crps_d5["value_mw"]) == (
        pytest.approx(10.0),
        "points",
        pytest.approx(5.0),
    )
    crps_d2 = rows["crps", "D2"]
    assert (crps_d2["value"], crps_d2["value_mw"]) == (pytest.approx(5.0), pytest.approx(2.0))
    assert rows["mae_median", "D5"]["value"] == pytest.approx(10.0)
    # The forecast sits above the actual at D5, so every quantile covers it, and below at D2.
    assert rows["coverage_q01", "D5"]["value"] == 1.0
    assert rows["coverage_q99", "D2"]["value"] == 0.0
    assert rows["coverage_q99", "D2"]["value_mw"] is None
    assert rows["coverage_q99", "D2"]["unit"] == "fraction"
    # The pinball loss of q99 at D5: the actual is 5 MW under the forecast, so 0.01 * 5 MW.
    assert rows["pinball_q99", "D5"]["value"] == pytest.approx(0.01 * 5.0 / 50.0 * 100.0)
    # At D2 the actual is 2 MW over the forecast: 0.99 * 2 MW = 4.95 points.
    assert rows["pinball_q99", "D2"]["value"] == pytest.approx(0.99 * 2.0 / 40.0 * 100.0)
    # Thresholds of 1,000 points lift everything to the same value, so twCRPS is zero.
    assert rows["twcrps", "D5"]["value"] == 0.0


def test_score_forecast_labels_groups_and_thin_slices() -> None:
    """The group, its protocol note, the half-hour count, and the thin flag are on every row."""
    inputs = build_two_series()
    table = score_forecast(
        **inputs, thresholds=thresholds_of({1: 1000.0, 2: 1000.0}), slices=["all"]
    )
    first = {r["label"]: r for r in table.filter(pl.col("metric") == "crps").iter_rows(named=True)}
    assert first["D5"]["group"] == "signed-MW"
    assert first["D5"]["group_note"] is None
    assert first["D2"]["group"] == "MVA-only"
    assert first["D2"]["group_note"] == "reactive power assumed zero"
    assert (first["D5"]["n_half_hours"], first["D5"]["thin"]) == (150, False)
    assert (first["D2"]["n_half_hours"], first["D2"]["thin"]) == (50, True)
    assert THIN_HALF_HOURS == 100
    assert table["metric"].unique().sort().to_list() == sorted(METRIC_NAMES)


def test_score_forecast_counts_each_slice_by_its_own_half_hours() -> None:
    """The lead bands count the rows of each series that fall inside them."""
    inputs = build_two_series()
    table = score_forecast(
        **inputs,
        thresholds=thresholds_of({1: 1000.0, 2: 1000.0}),
        slices=["lead_day_1", "lead_day_2", "lead_days_3_4"],
    ).filter(pl.col("metric") == "crps")
    counts = {(r["slice"], r["label"]): r["n_half_hours"] for r in table.iter_rows(named=True)}
    # Series 1 has leads 0.5 to 75 hours: 48 in day 1, 48 in day 2, 54 in days 3 to 4 (its rows).
    # Series 2 has leads 0.5 to 25: 48 in day 1, 2 in day 2, none in days 3 to 4.
    assert counts == {
        ("lead_day_1", "D5"): 48,
        ("lead_day_1", "D2"): 48,
        ("lead_day_2", "D5"): 48,
        ("lead_day_2", "D2"): 2,
        ("lead_days_3_4", "D5"): 54,
    }


def test_score_forecast_leaves_out_the_holdout_and_non_finite_actuals() -> None:
    """Valid times from 2026-07-01 and non-finite actuals are never scored."""
    init = utc(2026, 6, 30)
    forecast, actuals = constant_forecast(
        series=1,
        init=init,
        hours=[1.0, 2.0, 23.5, 24.0, 24.5, 30.0],
        actual=[10.0, float("nan"), 10.0, 10.0, 10.0, 10.0],
        offset=1.0,
    )
    # Valid times: 01:00, 02:00 (NaN), 23:30 on the 30th, then 00:00, 00:30, and 06:00 on 1 July.
    assert utc(2026, 7, 1) == HOLDOUT_START
    series = pl.DataFrame({"time_series_id": [1], "label": ["D5"], "capacity_mw": [100.0]})
    table = score_forecast(
        forecast=forecast,
        actuals=actuals,
        series=series,
        slice_table=slice_table_of(forecasts={"arm": forecast}, actuals=actuals),
        thresholds=thresholds_of({1: 1000.0}),
        slices=["all"],
    )
    assert table.filter(pl.col("metric") == "crps")["n_half_hours"].to_list() == [2]


def test_score_forecast_refuses_a_slice_whose_input_was_not_passed() -> None:
    """Asking for the event slices without events is an error, not a silently empty row."""
    inputs = build_two_series()
    with pytest.raises(ValueError, match="event_rows"):
        score_forecast(
            **inputs, thresholds=thresholds_of({1: 1000.0, 2: 1000.0}), slices=["event_rows"]
        )


def test_score_forecast_requires_a_threshold_and_a_context_row_for_every_series() -> None:
    """A missing twCRPS threshold or missing context is an error."""
    inputs = build_two_series()
    with pytest.raises(ValueError, match="threshold"):
        score_forecast(**inputs, thresholds=thresholds_of({1: 1000.0}))
    with pytest.raises(ValueError, match="context"):
        shared_slice_table(
            forecasts={"arm": inputs["forecast"]},
            actuals=inputs["actuals"],
            context=context_for(forecast=inputs["forecast"]).head(10),
        )


def test_per_series_by_lead_band_table_has_every_metric_for_every_series_and_band() -> None:
    """`LEAD_BAND_SLICES` gives every metric for each series and each band holding its rows."""
    inputs = build_two_series()
    table = score_forecast(
        **inputs, thresholds=thresholds_of({1: 1000.0, 2: 1000.0}), slices=LEAD_BAND_SLICES
    )
    assert set(table["slice"].unique()) == {
        "lead_day_1",
        "lead_day_2",
        "lead_days_3_4",
        "lead_days_1_10",
    }
    cells = table.group_by("slice", "label").agg(metrics=pl.col("metric").sort())
    assert len(cells) == 7  # D5 in four bands, D2 in three (its rows reach 25 hours).
    for metrics in cells["metrics"]:
        assert metrics.to_list() == sorted(METRIC_NAMES)


def two_arms_on_different_rows() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return arms A (all 100 half-hours) and B (the last 60), with a unit shape rising each row."""
    forecast_a, actuals = constant_forecast(
        series=1,
        init=utc(2026, 1, 5),
        hours=[0.5 * k for k in range(1, 101)],
        actual=[20.0] * 100,
        offset=1.0,
    )
    forecast_b = forecast_a.tail(60)
    return forecast_a, forecast_b, actuals


def test_two_arms_on_different_rows_are_scored_on_the_same_sunny_half_hours() -> None:
    """The slice table comes from the shared 60 rows, so both arms see one sunny set."""
    forecast_a, forecast_b, actuals = two_arms_on_different_rows()
    context = forecast_a.select(
        "time_series_id",
        "valid_time",
        temperature_c=pl.lit(10.0),
        elevation_deg=pl.lit(30.0),
        pv_unit_shape=pl.int_range(pl.len()).cast(pl.Float64) / 100.0 + 0.1,
    )
    table = shared_slice_table(
        forecasts={"a": forecast_a, "b": forecast_b}, actuals=actuals, context=context
    )
    assert table.height == 60
    # Of the 60 shared rows, the sunny edge is the 2/3 quantile of their own unit shape.
    assert table["sunny_unit_shape"].sum() == 20
    series = pl.DataFrame({"time_series_id": [1], "label": ["D5"], "capacity_mw": [50.0]})
    thresholds = thresholds_of({1: 1000.0})
    scored = {
        name: score_forecast(
            forecast=restrict_to_slice_table(forecast=frame, slice_table=table),
            actuals=actuals,
            series=series,
            slice_table=table,
            thresholds=thresholds,
            slices=["sunny_unit_shape"],
        )
        for name, frame in {"a": forecast_a, "b": forecast_b}.items()
    }
    counts = {
        name: out.filter(pl.col("metric") == "crps")["n_half_hours"].to_list()
        for name, out in scored.items()
    }
    assert counts == {"a": [20], "b": [20]}
    # Scored alone on its own 100 rows, arm A would have had a different sunny set.
    alone = shared_slice_table(forecasts={"a": forecast_a}, actuals=actuals, context=context)
    assert alone["sunny_unit_shape"].sum() != 20


def test_an_arm_scored_on_rows_that_differ_from_the_slice_table_is_an_error() -> None:
    """Scoring arm A on 100 rows against a table of 60 shared rows raises, naming the counts."""
    forecast_a, forecast_b, actuals = two_arms_on_different_rows()
    context = context_for(forecast=forecast_a)
    table = shared_slice_table(
        forecasts={"a": forecast_a, "b": forecast_b}, actuals=actuals, context=context
    )
    series = pl.DataFrame({"time_series_id": [1], "label": ["D5"], "capacity_mw": [50.0]})
    with pytest.raises(ValueError, match="40 scored half-hours are not in the table"):
        score_forecast(
            forecast=forecast_a,
            actuals=actuals,
            series=series,
            slice_table=table,
            thresholds=thresholds_of({1: 1000.0}),
        )
    dropped = restrict_to_slice_table(forecast=forecast_a, slice_table=table)
    assert forecast_a.height - dropped.height == 40
    with pytest.raises(ValueError, match="60 table half-hours are not scored"):
        score_forecast(
            forecast=forecast_b.head(0),
            actuals=actuals,
            series=series,
            slice_table=table,
            thresholds=thresholds_of({1: 1000.0}),
        )


def test_a_series_absent_from_a_forecast_gets_not_fitted_rows_and_the_tally_counts_it() -> None:
    """D2 has no forecast in arm B: arm B reports it not fitted, and the tally excludes it."""
    inputs = build_two_series()
    only_d5 = inputs["forecast"].filter(pl.col("time_series_id") == 1)
    context = context_for(forecast=inputs["forecast"])
    table_b = shared_slice_table(
        forecasts={"b": only_d5}, actuals=inputs["actuals"], context=context
    )
    thresholds = thresholds_of({1: 1000.0, 2: 1000.0})
    arm_a = score_forecast(**inputs, thresholds=thresholds, slices=["all"])
    arm_b = score_forecast(
        forecast=only_d5,
        actuals=inputs["actuals"],
        series=inputs["series"],
        slice_table=table_b,
        thresholds=thresholds,
        slices=["all"],
    )
    missing = arm_b.filter(pl.col("label") == "D2")
    assert missing.height == len(METRIC_NAMES)
    assert set(missing["status"]) == {STATUS_NOT_FITTED}
    assert missing["value"].null_count() == missing.height
    assert missing["n_half_hours"].to_list() == [0] * missing.height
    assert set(arm_b.filter(pl.col("label") == "D5")["status"]) == {"scored"}
    tally = primaries_won(treatment=arm_b, reference=arm_a, metric="crps")
    assert tally["scored_for_both"] == 1
    assert (tally["not_fitted_treatment"], tally["not_fitted_reference"]) == (1, 0)
    assert tally["not_fitted_in_either"] == 1
    assert tally["won"] + tally["lost"] + tally["tied"] == 1
    assert tally["other_unscored"] == 0


def test_primaries_won_counts_lower_scores_over_the_primaries_scored_for_both() -> None:
    """Arm B is one point worse at D5 and one point better at D2."""
    inputs = build_two_series()
    thresholds = thresholds_of({1: 1000.0, 2: 1000.0})
    base = score_forecast(**inputs, thresholds=thresholds, slices=["all"])
    shifted = base.with_columns(
        value=pl.when(pl.col("label") == "D5")
        .then(pl.col("value") + 1)
        .otherwise(pl.col("value") - 1)
    )
    tally = primaries_won(treatment=shifted, reference=base, metric="crps")
    assert (tally["won"], tally["lost"], tally["scored_for_both"]) == (1, 1, 2)
    with pytest.raises(ValueError, match="coverage"):
        primaries_won(treatment=shifted, reference=base, metric="coverage_q95")


def paired_rows(
    *, month_levels: list[float], difference: list[float]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return `scored_rows` of two forecasts over consecutive months, 48 rows a month.

    The first forecast's error in month m is `month_levels[m]` MW at a 100 MW series, so its CRPS is
    that many points. The second's error is `difference[m]` MW larger, so its CRPS is that much
    higher.
    """
    series = pl.DataFrame({"time_series_id": [1], "label": ["D5"], "capacity_mw": [100.0]})

    def build(*, extra: list[float]) -> pl.DataFrame:
        pieces_f, pieces_a = [], []
        for m, level in enumerate(month_levels):
            f, a = constant_forecast(
                series=1,
                init=utc(2025, 7 + m, 1),
                hours=[0.5 * k for k in range(1, 49)],
                actual=[50.0] * 48,
                offset=level + extra[m],
            )
            pieces_f.append(f)
            pieces_a.append(a)
        forecast = pl.concat(pieces_f)
        actuals = pl.concat(pieces_a)
        return scored_rows(
            forecast=forecast,
            actuals=actuals,
            series=series,
            slice_table=slice_table_of(forecasts={"arm": forecast}, actuals=actuals),
            thresholds=thresholds_of({1: 1000.0}),
        )

    return build(extra=difference), build(extra=[0.0] * len(month_levels))


def test_bootstrap_of_a_constant_paired_difference_has_no_width() -> None:
    """Whole months resampled for both forecasts keep the pairing, so a constant gap has no spread.

    The level of the CRPS changes by a factor of 13 between months. An interval that resampled the
    two forecasts separately would be wide, and this one is a single point.
    """
    treatment, reference = paired_rows(
        month_levels=[1.0, 5.0, 9.0, 13.0], difference=[1.0, 1.0, 1.0, 1.0]
    )
    result = month_bootstrap_contrast(treatment=treatment, reference=reference, metric="crps")
    assert result["difference"] == pytest.approx(1.0)
    assert (result["lower"], result["upper"]) == (pytest.approx(1.0), pytest.approx(1.0))
    assert (result["n_half_hours"], result["n_months"]) == (4 * 48, 4)


def test_bootstrap_interval_straddles_zero_when_the_months_disagree() -> None:
    """Three months worse by 1 point and one better by 1 give a mean of 0.5 and span zero."""
    treatment, reference = paired_rows(
        month_levels=[2.0, 2.0, 2.0, 2.0], difference=[1.0, 1.0, 1.0, -1.0]
    )
    result = month_bootstrap_contrast(treatment=treatment, reference=reference, metric="crps")
    assert result["difference"] == pytest.approx(0.5)
    assert result["lower"] < 0.0 < result["upper"]
    assert result["upper"] == pytest.approx(1.0)
    wide = month_bootstrap_contrast(
        treatment=treatment, reference=reference, metric="crps", level_percent=99.0
    )
    assert result["level_percent"] == REPORTED_LEVEL_PERCENT
    assert wide["lower"] <= result["lower"]
    assert wide["upper"] >= result["upper"]


def test_headline_contrast_uses_99_percent_and_20000_resamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline wrapper passes the headline pair and the reported one passes the 95% pair."""
    assert (HEADLINE_LEVEL_PERCENT, HEADLINE_RESAMPLES) == (99.0, 20_000)
    assert (REPORTED_LEVEL_PERCENT, REPORTED_RESAMPLES) == (95.0, 2000)
    seen: list[tuple[float, int]] = []

    def spy(**kwargs: object) -> tuple[float, float]:
        seen.append((kwargs["level"], kwargs["n_resamples"]))  # ty: ignore[invalid-argument-type]
        return (0.0, 1.0)

    monkeypatch.setattr("studies.eval_protocol.metrics.bootstrap_difference_at_level", spy)
    treatment, reference = paired_rows(month_levels=[2.0, 2.0], difference=[1.0, 1.0])
    headline = headline_month_bootstrap_contrast(
        treatment=treatment, reference=reference, metric="crps"
    )
    per_primary = reported_contrast_by_primary(
        treatment=treatment, reference=reference, metric="crps"
    )
    month_bootstrap_contrast(treatment=treatment, reference=reference, metric="crps")
    assert seen == [(99.0, 20_000), (95.0, 2000), (95.0, 2000)]
    assert headline["level_percent"] == 99.0
    assert per_primary["label"].to_list() == ["D5"]
    assert per_primary["level_percent"].to_list() == [95.0]


def test_bootstrap_takes_slice_columns_from_the_slice_table() -> None:
    """Frames without slice columns can be filtered on a slice by passing the table."""
    treatment, reference = paired_rows(month_levels=[2.0, 2.0], difference=[1.0, 1.0])
    table = treatment.select(*treatment.columns[:3], "lead_day_1")
    bare = treatment.select(*treatment.columns[:3], "crps")
    result = month_bootstrap_contrast(
        treatment=bare,
        reference=reference.select(*reference.columns[:3], "crps"),
        metric="crps",
        where=pl.col("lead_day_1"),
        slice_table=table,
    )
    assert result["n_half_hours"] == 2 * 48
    with pytest.raises(ValueError, match="not in the slice table"):
        month_bootstrap_contrast(
            treatment=bare, reference=reference, metric="crps", slice_table=table.head(5)
        )


def test_bootstrap_filter_keeps_only_the_rows_it_selects() -> None:
    """A filter on a slice column restricts the contrast to those rows."""
    treatment, reference = paired_rows(
        month_levels=[2.0, 2.0, 2.0, 2.0], difference=[1.0, 1.0, 1.0, 1.0]
    )
    result = month_bootstrap_contrast(
        treatment=treatment, reference=reference, metric="crps", where=pl.col("lead_day_1")
    )
    assert result["n_half_hours"] == 4 * 48
    only_day_two = month_bootstrap_contrast(
        treatment=treatment.filter(pl.col("lead_hours") > 12.0),
        reference=reference,
        metric="crps",
    )
    assert only_day_two["n_half_hours"] == 4 * 24
    with pytest.raises(ValueError, match="share no half-hour"):
        month_bootstrap_contrast(
            treatment=treatment, reference=reference, metric="crps", where=pl.col("lead_day_2")
        )


def node_inputs(*, actual_mw: float, quantile_mw: float) -> CurtailmentInputs:
    """Return curtailment inputs: a 10 MW earlier maximum and four breach half-hours."""
    cutoff = utc(2025, 7, 2)
    history_times = [cutoff - timedelta(hours=0.5 * k) for k in range(1, 97)]
    flows = pl.DataFrame(
        {
            "time_series_id": [7] * 97,
            "valid_time": [*history_times, utc(2024, 6, 1)],
            "flow_mw": [10.0] + [4.0] * 96,
        }
    )
    hours = [12.5, 13.0, 13.5, 14.0]
    forecast, _ = constant_forecast(
        series=7, init=cutoff, hours=hours, actual=[actual_mw] * 4, offset=0.0
    )
    forecast = forecast.with_columns(
        actual=pl.lit(actual_mw), **{q: pl.lit(quantile_mw) for q in QUANTILE_COLUMNS}
    )
    return CurtailmentInputs(
        forecast=forecast,
        flows=flows,
        series=pl.DataFrame({"time_series_id": [7], "label": ["D5"], "capacity_mw": [20.0]}),
        scored_start=cutoff,
    )


def test_avoidable_curtailment_cost_matches_the_hand_worked_energy_proxy() -> None:
    """A limit of 5 MW (half the earlier 10 MW maximum), an actual of 9 MW, and 4 half-hours.

    A forecast of 0 MW plans no curtailment, so the 4 MW breach of each half-hour (2 MWh) is priced
    at 20 times 2,000 pounds. The perfect forecast plans exactly the 2 MWh. An over-forecast of
    20 MW plans 7.5 MWh per half-hour and has no breach left.
    """
    flows_and_series = node_inputs(actual_mw=9.0, quantile_mw=0.0)
    forecasts = {
        "low": flows_and_series["forecast"],
        "right": flows_and_series["forecast"].with_columns(
            **{q: pl.lit(9.0) for q in QUANTILE_COLUMNS}
        ),
        "high": flows_and_series["forecast"].with_columns(
            **{q: pl.lit(20.0) for q in QUANTILE_COLUMNS}
        ),
    }
    out = avoidable_curtailment_cost(
        forecasts=forecasts,
        series=flows_and_series["series"],
        flows=flows_and_series["flows"],
        scored_start=flows_and_series["scored_start"],
    )
    cost = dict(zip(out["arm"], out["avoidable_cost_gbp"], strict=True))
    perfect = 2000.0 * 2.0 * 4
    assert cost["right"] == pytest.approx(0.0)
    assert cost["low"] == pytest.approx(2000.0 * 20.0 * 2.0 * 4 - perfect)
    assert cost["high"] == pytest.approx(2000.0 * 7.5 * 4 - perfect)
    assert set(out["n_node_half_hours"].to_list()) == {4}


def test_curtailment_cost_is_null_for_the_unsigned_group_and_for_short_history() -> None:
    """An MVA-only series and a signed series with under 12 months of flows get null costs."""
    inputs = node_inputs(actual_mw=9.0, quantile_mw=0.0)
    forecast = inputs["forecast"]
    series = pl.DataFrame(
        {"time_series_id": [7, 8, 9], "label": ["D5", "D2", "D9"], "capacity_mw": [20.0] * 3}
    )
    forecast_all = pl.concat([forecast.with_columns(time_series_id=pl.lit(i)) for i in (7, 8, 9)])
    flows = pl.concat(
        [
            inputs["flows"].with_columns(time_series_id=pl.lit(7)),
            inputs["flows"].with_columns(time_series_id=pl.lit(8)),
            inputs["flows"]
            .filter(pl.col("valid_time") > utc(2025, 1, 1))
            .with_columns(time_series_id=pl.lit(9)),
        ]
    )
    out = avoidable_curtailment_cost(
        forecasts={"arm": forecast_all},
        series=series,
        flows=flows,
        scored_start=inputs["scored_start"],
    )
    rows = {r["label"]: r for r in out.iter_rows(named=True)}
    assert rows["D5"]["status"] == "ok"
    assert rows["D5"]["avoidable_cost_gbp"] is not None
    assert (rows["D2"]["avoidable_cost_gbp"], rows["D2"]["status"]) == (None, STATUS_UNSIGNED_METER)
    assert STATUS_UNSIGNED_METER == (
        "reactive power assumed zero; cost not defined for unsigned meters"
    )
    assert (rows["D9"]["avoidable_cost_gbp"], rows["D9"]["status"]) == (None, STATUS_SHORT_HISTORY)
    assert rows["D2"]["group"] == "MVA-only"

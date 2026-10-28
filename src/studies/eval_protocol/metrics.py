"""The metrics of the demand evaluation protocol, and the one function that scores a forecast.

Every experiment on Primary demand imports its metric definitions from this module, so a change to a
definition reaches every experiment. `score_forecast` returns one table of metric by slice by
series, and `month_bootstrap_contrast` gives the paired month-resampled contrast of two forecasts.

**Units.** The inputs are in MW. Every metric is computed in *points of effective capacity*, where a
point is 1% of the series' effective capacity, and the table carries the value in megawatts beside
it (`value_mw`). The coverage of a quantile has no unit and no megawatt value. The avoidable
curtailment cost is in pounds (`avoidable_curtailment_cost`).

**Metric definitions** (each one is a row-wise loss averaged over a slice):

- `mae_median`: the absolute error of the median forecast `q50`.
- `crps`: the continuous ranked probability score estimated from the 11 forecast quantiles, which is
  twice the pinball loss averaged over the levels 0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9,
  0.95, and 0.99 (the "quantile score" of every card and experiment). It is not the ensemble CRPS of
  `studies.ensemble_scores`, which is a different estimator.
- `twcrps`: the same quantile score after raising the actual value and every forecast quantile to a
  threshold from below, `max(x, threshold)`, so that only the upper tail is scored (experiments
  277 and 311, columns `tw95`). **The threshold is fixed per series: the 95th percentile of the
  series' own power, in points, over the readings before the scored period** (`twcrps_thresholds`),
  which needs `MIN_HISTORY_MONTHS` (12) months of those readings. With fewer, the threshold is null
  with a status, and so is the series' `twcrps`. Experiment 277 also reported a 99th percentile
  threshold, which `twcrps_thresholds` takes as an argument.
- `pinball_q01`, `pinball_q05`, `pinball_q95`, `pinball_q99`: the pinball loss of that quantile.
- `coverage_q01`, `coverage_q05`, `coverage_q95`, `coverage_q99`: the share of rows where the actual
  is at or below that quantile.

**The avoidable curtailment cost** wraps `studies.node_curtailment.node_curtailment`, the function
experiment 277 scored the Primaries with, in its node-only mode, and `avoidable_cost`, which
subtracts the cost of a perfect forecast. Each series is its own node, a positive demand reading is
the flow that can breach, the limit is `HEADLINE_LIMIT_FRACTION` (0.5) of the series' maximum before
the scored period, the price is 2,000 pounds per MWh, and the lead window is 12 to 36 hours, all as
experiment 277 set them. The limit needs `MIN_HISTORY_MONTHS` months of flows before the scored
period, and the cost is null with a status where it has fewer. **The cost is defined only for the
signed-MW group of series.** The MVA-only Primaries read unsigned apparent power, so a reverse flow
is hidden and the cost would be wrong in kind there. Their cost is null, labelled
"reactive power assumed zero; cost not defined for unsigned meters".

**Every arm is scored against one slice table.** `shared_slice_table` computes the
slice columns once on the half-hours that every arm shares, because a slice such as "sunny by unit
shape" takes a per-series quantile of the rows it is given. `score_forecast` raises
if an arm's scored half-hours differ from the slice table's keys, and `restrict_to_slice_table`
drops the rows of an arm that fall outside the table.

**A series with no forecast is reported, not dropped.** `score_forecast` emits one row per metric
and slice with a null value and `status` "not fitted" for each series in the series table that has
no scored half-hour, and `primaries_won` tallies the Primaries won against the Primaries scored for
both arms and the Primaries not fitted.

**Two bootstrap contrasts.** The protocol's gain condition is a 99% interval, so the headline
contrast (`headline_month_bootstrap_contrast`) uses `HEADLINE_LEVEL_PERCENT` (99) and
`HEADLINE_RESAMPLES` (20,000). The reported contrast, per Primary
(`reported_contrast_by_primary`) and by default in `month_bootstrap_contrast`, uses
`REPORTED_LEVEL_PERCENT` (95) and `REPORTED_RESAMPLES` (2,000).

**The temperature and clear-sky inputs of the slices** are named in `studies.eval_protocol.slices`,
which also states that the protected-slice tolerance reads the sunny-by-clear-sky-index slice and
that the caller passes `clear_sky_index` as a realised value.

**Imports from `studies`.** Three: `studies.bootstrap` for the month-resampled paired bootstrap,
`studies.node_curtailment` for the curtailment function, and `studies.dp_calendar` (through the
slices) for the local calendar. The maintainer who moves this package takes those three with it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Final, Literal, TypedDict

import numpy as np
import polars as pl

from studies.bootstrap import bootstrap_difference_at_level
from studies.eval_protocol.slices import (
    GROUP_NOTES,
    KEY_COLUMNS,
    LEAD_BANDS,
    SERIES_GROUPS,
    SLICE_NAMES,
    build_slice_table,
    lead_hours_expression,
)
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    avoidable_cost,
    node_curtailment,
    scaled_limits,
    table_with_scaled_limits,
)

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
"""The 11 quantile levels of the quantile score."""

QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
"""The forecast frame's quantile columns, one per entry of `LEVELS`."""

MEDIAN_INDEX: Final[int] = 5
TAIL_COLUMNS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
"""The quantiles whose pinball loss and coverage are reported."""

THIN_HALF_HOURS: Final[int] = 100
"""A slice with fewer half-hours than this in a series is labelled thin (`e00250_score.py`)."""

HOLDOUT_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No experiment scores a valid time on or after this instant (the protocol's honour rule)."""

TW_THRESHOLD_QUANTILE: Final[float] = 0.95
"""The quantile of a series' earlier power that is the twCRPS threshold."""

HEADLINE_LIMIT_FRACTION: Final[float] = 0.5
"""The curtailment limit as a fraction of a series' earlier maximum (experiment 277's headline)."""

HEADLINE_PRICE_GBP_PER_MWH: Final[float] = 2000.0
CURTAILMENT_LEAD_HOURS: Final[tuple[float, float]] = (12.0, 36.0)
PERFECT_ARM: Final[str] = "perfect"
HEADLINE_LEVEL_PERCENT: Final[float] = 99.0
HEADLINE_RESAMPLES: Final[int] = 20_000
"""The interval of the protocol's gain condition: 99%, which needs far more than 2,000 resamples."""

REPORTED_LEVEL_PERCENT: Final[float] = 95.0
REPORTED_RESAMPLES: Final[int] = 2000
"""The interval reported per Primary beside the headline contrast."""

MIN_HISTORY_MONTHS: Final[int] = 12
"""The months of readings before the scored period that the twCRPS threshold and the curtailment
limit need."""

STATUS_SCORED: Final[str] = "scored"
STATUS_NOT_FITTED: Final[str] = "not fitted"
STATUS_OK: Final[str] = "ok"
STATUS_SHORT_HISTORY: Final[str] = f"under {MIN_HISTORY_MONTHS} months of earlier readings"
STATUS_UNSIGNED_METER: Final[str] = (
    "reactive power assumed zero; cost not defined for unsigned meters"
)
STATUS_NOT_SIGNED_GROUP: Final[str] = "cost defined only for the signed-MW group"

LEAD_BAND_SLICES: Final[tuple[str, ...]] = tuple(LEAD_BANDS)
"""Pass these as `slices` to `score_forecast` for the per-series by lead-band table."""

SCALE_METRICS: Final[tuple[str, ...]] = (
    "mae_median",
    "crps",
    "twcrps",
    *(f"pinball_{name}" for name in TAIL_COLUMNS),
)
COVERAGE_METRICS: Final[tuple[str, ...]] = tuple(f"coverage_{name}" for name in TAIL_COLUMNS)
METRIC_NAMES: Final[tuple[str, ...]] = (*SCALE_METRICS, *COVERAGE_METRICS)

MonthOfType = Literal["power_fcst_init_time", "valid_time"]


class ContrastInterval(TypedDict):
    """A paired contrast of two forecasts and the rows it rests on."""

    difference: float
    lower: float
    upper: float
    level_percent: float
    n_half_hours: int
    n_months: int


def _pinball(*, actual: np.ndarray, quantiles: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Return the pinball loss per row and level, shape (rows, levels)."""
    error = actual[:, None] - quantiles
    return np.maximum(levels[None, :] * error, (levels[None, :] - 1.0) * error)


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's quantile score: twice the pinball loss averaged over `LEVELS`.

    Args:
        actual: The observed values, shape (rows,).
        quantiles: The forecast quantiles at `LEVELS`, shape (rows, 11), in the units of `actual`.

    Returns:
        The score of each row, in the units of `actual`.
    """
    return 2.0 * _pinball(actual=actual, quantiles=quantiles, levels=LEVELS).mean(axis=1)


def tail_weighted_score(
    *, actual: np.ndarray, quantiles: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    """Return each row's twCRPS: the quantile score after taking `max(x, threshold)` of everything.

    Args:
        actual: The observed values, shape (rows,).
        quantiles: The forecast quantiles at `LEVELS`, shape (rows, 11).
        threshold: Each row's threshold, shape (rows,), in the units of `actual`.

    Returns:
        The tail-weighted score of each row.
    """
    return quantile_score(
        actual=np.maximum(actual, threshold), quantiles=np.maximum(quantiles, threshold[:, None])
    )


def _months_before(*, instant: datetime, months: int) -> datetime:
    """Return `instant` less a number of calendar months."""
    return pl.select(pl.lit(instant).dt.offset_by(f"-{months}mo")).item()


def twcrps_thresholds(
    *,
    history: pl.DataFrame,
    series: pl.DataFrame,
    before: datetime,
    quantile: float = TW_THRESHOLD_QUANTILE,
) -> pl.DataFrame:
    """Return each series' twCRPS threshold: a quantile of its earlier power, in points.

    The threshold needs `MIN_HISTORY_MONTHS` months of readings before `before`: the series' first
    reading must be at least that long before it. With fewer, the threshold is null and the status
    says so.

    Args:
        history: Power readings with `time_series_id`, `valid_time` (UTC), and `power_mw`.
        series: The series table of `score_forecast`, for each series' effective capacity.
        before: A UTC datetime. Only readings strictly before it count, so the threshold never
            sees the scored period.
        quantile: The quantile of the earlier readings, 0.95 by default (experiment 277 also used
            0.99).

    Returns:
        One row per series of `series`: `time_series_id`, `threshold_points` (the threshold in
        points of effective capacity, null if the history is too short), and `status` (`ok`, or the
        reason the threshold is null).
    """
    needed_from = _months_before(instant=before, months=MIN_HISTORY_MONTHS)
    earlier = history.filter(pl.col("valid_time") < before, pl.col("power_mw").is_finite())
    rows = []
    for series_id, capacity in zip(series["time_series_id"], series["capacity_mw"], strict=True):
        group = earlier.filter(pl.col("time_series_id") == series_id)
        first = group["valid_time"].min()
        if group.is_empty() or first > needed_from:  # ty: ignore[unsupported-operator]
            rows.append((series_id, None, STATUS_SHORT_HISTORY))
            continue
        threshold = float(np.quantile(group["power_mw"].to_numpy() / capacity * 100.0, quantile))
        rows.append((series_id, threshold, STATUS_OK))
    return pl.DataFrame(
        rows,
        schema={
            "time_series_id": series.schema["time_series_id"],
            "threshold_points": pl.Float64,
            "status": pl.String,
        },
        orient="row",
    )


def row_losses(
    *,
    forecast: pl.DataFrame,
    actuals: pl.DataFrame,
    series: pl.DataFrame,
    thresholds: pl.DataFrame,
    holdout_start: datetime = HOLDOUT_START,
) -> pl.DataFrame:
    """Return every scored half-hour with its losses, in points of effective capacity.

    A half-hour is scored when its actual value is finite and its valid time is before
    `holdout_start`. The output carries the keys, `lead_hours`, `label`, `group`, `capacity_mw`,
    `actual_points`, and one column per entry of `METRIC_NAMES`, in points (the coverage columns
    hold 0 or 1). `twcrps` is null for a series whose threshold is null.

    Args:
        forecast: One row per series, forecast origin, and valid time, with `KEY_COLUMNS` and the 11
            `QUANTILE_COLUMNS` in MW.
        actuals: `time_series_id`, `valid_time`, and `actual` in MW.
        series: One row per series: `time_series_id`, `label` (such as `D5`), and `capacity_mw`, the
            effective capacity.
        thresholds: Each series' twCRPS threshold in points, from `twcrps_thresholds`.
        holdout_start: Valid times at or after this are not scored.

    Returns:
        The per-row table, in the order of `forecast`.

    Raises:
        ValueError: If a forecast row has no series entry or no twCRPS threshold row.
    """
    labelled = series.select(
        "time_series_id",
        "label",
        "capacity_mw",
        group=pl.col("label").replace_strict(SERIES_GROUPS, default=None, return_dtype=pl.String),
    )
    rows = (
        forecast.join(
            actuals.select("time_series_id", "valid_time", "actual"),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="m:1",
            maintain_order="left",
        )
        .join(labelled, on="time_series_id", how="left", validate="m:1", maintain_order="left")
        .filter(pl.col("actual").is_finite(), pl.col("valid_time") < holdout_start)
        .with_columns(lead_hours=lead_hours_expression())
    )
    if rows["capacity_mw"].null_count():
        msg = "a forecast row belongs to a series that is not in the series table"
        raise ValueError(msg)
    threshold_of = dict(
        zip(thresholds["time_series_id"], thresholds["threshold_points"], strict=True)
    )
    unknown = set(rows["time_series_id"].unique().to_list()) - set(threshold_of)
    if unknown:
        msg = f"no twCRPS threshold for the series {sorted(unknown)}"
        raise ValueError(msg)
    scale = 100.0 / rows["capacity_mw"].to_numpy().astype(np.float64)
    actual = rows["actual"].to_numpy().astype(np.float64) * scale
    quantiles = rows.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64) * scale[:, None]
    threshold = np.array(
        [np.nan if threshold_of[s] is None else threshold_of[s] for s in rows["time_series_id"]],
        dtype=np.float64,
    )
    pinball = _pinball(actual=actual, quantiles=quantiles, levels=LEVELS)
    columns: dict[str, np.ndarray] = {
        "actual_points": actual,
        "mae_median": np.abs(actual - quantiles[:, MEDIAN_INDEX]),
        "crps": 2.0 * pinball.mean(axis=1),
        "twcrps": tail_weighted_score(actual=actual, quantiles=quantiles, threshold=threshold),
    }
    for name in TAIL_COLUMNS:
        position = QUANTILE_COLUMNS.index(name)
        columns[f"pinball_{name}"] = pinball[:, position]
        columns[f"coverage_{name}"] = (actual <= quantiles[:, position]).astype(np.float64)
    return rows.select(*KEY_COLUMNS, "lead_hours", "label", "group", "capacity_mw").with_columns(
        *[pl.Series(name, values, nan_to_null=True) for name, values in columns.items()]
    )


def _scored_keys(
    *, forecast: pl.DataFrame, actuals: pl.DataFrame, holdout_start: datetime
) -> pl.DataFrame:
    """Return the keys of the half-hours of one forecast that `row_losses` scores."""
    return (
        forecast.select(*KEY_COLUMNS)
        .join(
            actuals.select("time_series_id", "valid_time", "actual"),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="m:1",
            maintain_order="left",
        )
        .filter(pl.col("actual").is_finite(), pl.col("valid_time") < holdout_start)
        .select(*KEY_COLUMNS)
    )


def shared_slice_table(
    *,
    forecasts: Mapping[str, pl.DataFrame],
    actuals: pl.DataFrame,
    context: pl.DataFrame,
    events: pl.DataFrame | None = None,
    first_readings: Mapping[int, datetime] | None = None,
    holdout_start: datetime = HOLDOUT_START,
) -> pl.DataFrame:
    """Return the slice table of the half-hours that every arm is scored on.

    The shared half-hours are those that every forecast in `forecasts` holds and that
    `row_losses` scores (finite actual, valid time before `holdout_start`). The slice columns,
    including the per-series quantile of "sunny by unit shape", are computed on that set once.
    Restrict each arm to the table with `restrict_to_slice_table` before scoring it.

    Args:
        forecasts: Each arm's forecast, as in `row_losses`.
        actuals: As in `row_losses`.
        context: One row per series and valid time: `time_series_id`, `valid_time`, `temperature_c`,
            `elevation_deg`, `pv_unit_shape`, and optionally `clear_sky_index`, which the caller
            passes as a realised value. The columns are described in `studies.eval_protocol.slices`.
        events: Hand-labelled events, as in `add_slice_columns`.
        first_readings: First power reading of each series, as in `add_slice_columns`.
        holdout_start: Valid times at or after this are not scored.

    Returns:
        `KEY_COLUMNS` and one boolean column per available slice.

    Raises:
        ValueError: If no arm is given, or a shared half-hour has no row in `context`.
    """
    if not forecasts:
        msg = "at least one forecast is needed to build a slice table"
        raise ValueError(msg)
    keys = [
        _scored_keys(forecast=frame, actuals=actuals, holdout_start=holdout_start)
        for frame in forecasts.values()
    ]
    shared = keys[0]
    for other in keys[1:]:
        shared = shared.join(other, on=list(KEY_COLUMNS), how="inner", maintain_order="left")
    joined = shared.join(
        context.with_columns(_has_context=pl.lit(value=True)),
        on=["time_series_id", "valid_time"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if joined["_has_context"].null_count():
        msg = "a shared half-hour has no row in the context table"
        raise ValueError(msg)
    return build_slice_table(
        rows=joined.drop("_has_context"), events=events, first_readings=first_readings
    )


def restrict_to_slice_table(*, forecast: pl.DataFrame, slice_table: pl.DataFrame) -> pl.DataFrame:
    """Return the rows of a forecast whose key is in the slice table, in the forecast's order.

    Args:
        forecast: A forecast with `KEY_COLUMNS`.
        slice_table: The table from `shared_slice_table`.

    Returns:
        The forecast without the rows that fall outside the slice table. Count them as
        `forecast.height - result.height`.
    """
    return forecast.join(
        slice_table.select(*KEY_COLUMNS), on=list(KEY_COLUMNS), how="semi", maintain_order="left"
    )


def scored_rows(
    *,
    forecast: pl.DataFrame,
    actuals: pl.DataFrame,
    series: pl.DataFrame,
    thresholds: pl.DataFrame,
    slice_table: pl.DataFrame,
) -> pl.DataFrame:
    """Return `row_losses` joined to the slice columns of the shared slice table.

    Args:
        forecast: As in `row_losses`.
        actuals: As in `row_losses`.
        series: As in `row_losses`.
        thresholds: As in `row_losses`.
        slice_table: The table from `shared_slice_table`, one boolean column per slice.

    Returns:
        The per-row losses with one boolean column per slice of the table.

    Raises:
        ValueError: If the scored half-hours are not exactly the keys of the slice table.
    """
    losses = row_losses(forecast=forecast, actuals=actuals, series=series, thresholds=thresholds)
    joined = losses.join(
        slice_table.with_columns(_in_table=pl.lit(value=True)),
        on=list(KEY_COLUMNS),
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    outside = joined["_in_table"].null_count()
    unscored = slice_table.height - (losses.height - outside)
    if outside or unscored:
        msg = (
            f"the forecast's scored half-hours differ from the slice table's keys: {outside} "
            f"scored half-hours are not in the table and {unscored} table half-hours are not "
            "scored. Score every arm on the shared rows (restrict_to_slice_table)"
        )
        raise ValueError(msg)
    return joined.drop("_in_table")


def score_forecast(
    *,
    forecast: pl.DataFrame,
    actuals: pl.DataFrame,
    series: pl.DataFrame,
    slice_table: pl.DataFrame,
    thresholds: pl.DataFrame,
    slices: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Score a forecast: every metric, over every slice, for every series.

    The per-series by lead-band table is the output for `slices=LEAD_BAND_SLICES`: one row per
    metric, lead band, and series.

    Args:
        forecast: One row per series, forecast origin, and valid time, with `time_series_id`,
            `power_fcst_init_time`, `valid_time`, and the 11 quantile columns `q01` to `q99`, in MW.
        actuals: `time_series_id`, `valid_time`, and `actual` in MW.
        series: One row per series expected in the output: `time_series_id`, `label` (such as
            `D5`), and `capacity_mw`.
        slice_table: The slice columns of the half-hours shared by every arm (`shared_slice_table`).
        thresholds: Each series' twCRPS threshold in points (`twcrps_thresholds`).
        slices: The slices to report. All the slices that the table holds if not given.

    Returns:
        One row per metric, slice, and series that holds at least one half-hour, and one row per
        metric and slice for every series of `series` with no scored half-hour at all, with
        `metric`, `slice`, `time_series_id`, `label`, `group`, `group_note` (the label the protocol
        attaches to the group, such as "reactive power assumed zero"), `n_half_hours`, `thin`
        (true when `n_half_hours` is under `THIN_HALF_HOURS`, null for a series not fitted),
        `value`, `unit` (`points` or `fraction`), `value_mw` (null for a coverage), and `status`.
        The status is `scored`, `not fitted` (the series has no scored half-hour, so `value` is
        null), or the threshold's status for a `twcrps` row whose threshold is null.

    Raises:
        ValueError: If a requested slice is not in the slice table, or the forecast's scored
            half-hours are not the table's keys.
    """
    rows = scored_rows(
        forecast=forecast,
        actuals=actuals,
        series=series,
        thresholds=thresholds,
        slice_table=slice_table,
    )
    available = [name for name in SLICE_NAMES if name in rows.columns]
    chosen = list(available if slices is None else slices)
    missing = [name for name in chosen if name not in available]
    if missing:
        msg = f"the slices {missing} are not in the slice table"
        raise ValueError(msg)
    index = ["slice", "time_series_id", "label", "group", "capacity_mw", "n_half_hours"]
    pieces = []
    for slice_name in chosen:
        part = rows.filter(pl.col(slice_name))
        if part.is_empty():
            continue
        means = part.group_by("time_series_id", "label", "group", "capacity_mw").agg(
            n_half_hours=pl.len().cast(pl.Int64),
            **{name: pl.col(name).mean() for name in METRIC_NAMES},
        )
        pieces.append(means.with_columns(slice=pl.lit(slice_name)))
    scored = pl.concat(pieces).unpivot(
        on=list(METRIC_NAMES), index=index, variable_name="metric", value_name="value"
    )
    not_fitted = (
        series.join(rows.select("time_series_id").unique(), on="time_series_id", how="anti")
        .select(
            "time_series_id",
            "label",
            "capacity_mw",
            group=pl.col("label").replace_strict(
                SERIES_GROUPS, default=None, return_dtype=pl.String
            ),
            n_half_hours=pl.lit(0, dtype=pl.Int64),
            value=pl.lit(None, dtype=pl.Float64),
        )
        .join(pl.DataFrame({"slice": chosen}), how="cross")
        .join(pl.DataFrame({"metric": list(METRIC_NAMES)}), how="cross")
        .select(*index, "metric", "value")
    )
    threshold_status = thresholds.select("time_series_id", threshold_status=pl.col("status"))
    long = (
        pl.concat(
            [
                scored.with_columns(status=pl.lit(STATUS_SCORED)),
                not_fitted.with_columns(status=pl.lit(STATUS_NOT_FITTED)),
            ],
            how="vertical_relaxed",
        )
        .join(threshold_status, on="time_series_id", how="left", maintain_order="left")
        .with_columns(
            status=pl.when(
                pl.col("status") == STATUS_SCORED,
                pl.col("metric") == "twcrps",
                pl.col("threshold_status") != STATUS_OK,
            )
            .then(pl.col("threshold_status"))
            .otherwise(pl.col("status"))
        )
        .with_columns(
            thin=pl.when(pl.col("status") == STATUS_NOT_FITTED)
            .then(None)
            .otherwise(pl.col("n_half_hours") < THIN_HALF_HOURS),
            group_note=pl.col("group").replace_strict(
                GROUP_NOTES, default=None, return_dtype=pl.String
            ),
            unit=pl.when(pl.col("metric").is_in(COVERAGE_METRICS))
            .then(pl.lit("fraction"))
            .otherwise(pl.lit("points")),
        )
        .with_columns(
            value_mw=pl.when(pl.col("unit") == "points")
            .then(pl.col("value") * pl.col("capacity_mw") / 100.0)
            .otherwise(None)
        )
    )
    order = {name: i for i, name in enumerate(METRIC_NAMES)}
    slice_order = {name: i for i, name in enumerate(chosen)}
    return (
        long.with_columns(
            _m=pl.col("metric").replace_strict(order, return_dtype=pl.Int64),
            _s=pl.col("slice").replace_strict(slice_order, return_dtype=pl.Int64),
        )
        .sort("_m", "_s", "label")
        .select(
            "metric",
            "slice",
            "time_series_id",
            "label",
            "group",
            "group_note",
            "n_half_hours",
            "thin",
            "value",
            "unit",
            "value_mw",
            "status",
        )
    )


class PrimariesTally(TypedDict):
    """The Primaries won by one arm against another, with the denominator and the exclusions."""

    won: int
    lost: int
    tied: int
    scored_for_both: int
    not_fitted_treatment: int
    not_fitted_reference: int
    not_fitted_in_either: int
    other_unscored: int


def primaries_won(
    *,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    metric: str = "crps",
    slice_name: str = "all",
) -> PrimariesTally:
    """Return how many Primaries the treatment wins on a metric, with the denominator.

    A Primary is won when the treatment's value is lower than the reference's. The denominator,
    `scored_for_both`, counts the Primaries with a value in both tables. A Primary with status
    `not fitted` in a table is counted in `not_fitted_treatment` or `not_fitted_reference` and in
    `not_fitted_in_either`. A Primary with no value for another reason (no row, or a null twCRPS
    threshold) is in `other_unscored`, so that `scored_for_both + not_fitted_in_either +
    other_unscored` is the number of Primaries in either table.

    Args:
        treatment: `score_forecast` of the forecast under test.
        reference: `score_forecast` of the forecast it is compared with.
        metric: A loss metric of `SCALE_METRICS`.
        slice_name: The slice to compare on.

    Returns:
        The tally.

    Raises:
        ValueError: If the metric is a coverage, which has no better or worse direction.
    """
    if metric in COVERAGE_METRICS:
        msg = f"{metric} is a coverage; a lower value is not a win"
        raise ValueError(msg)

    def pick(table: pl.DataFrame) -> pl.DataFrame:
        return table.filter(pl.col("metric") == metric, pl.col("slice") == slice_name).select(
            "time_series_id", "value", "status"
        )

    joined = pick(treatment).join(
        pick(reference), on="time_series_id", how="full", coalesce=True, suffix="_reference"
    )
    both = joined.filter(pl.col("value").is_not_null(), pl.col("value_reference").is_not_null())
    not_fitted_t = pl.col("status") == STATUS_NOT_FITTED
    not_fitted_r = pl.col("status_reference") == STATUS_NOT_FITTED
    tally = joined.select(
        t=not_fitted_t.sum(), r=not_fitted_r.sum(), either=(not_fitted_t | not_fitted_r).sum()
    ).row(0, named=True)
    return {
        "won": both.filter(pl.col("value") < pl.col("value_reference")).height,
        "lost": both.filter(pl.col("value") > pl.col("value_reference")).height,
        "tied": both.filter(pl.col("value") == pl.col("value_reference")).height,
        "scored_for_both": both.height,
        "not_fitted_treatment": tally["t"],
        "not_fitted_reference": tally["r"],
        "not_fitted_in_either": tally["either"],
        "other_unscored": joined.height - both.height - tally["either"],
    }


def _with_slice_columns(*, rows: pl.DataFrame, slice_table: pl.DataFrame) -> pl.DataFrame:
    """Return `rows` with the slice-table columns it lacks, raising if a key is not in the table."""
    new = [name for name in slice_table.columns if name not in rows.columns]
    joined = rows.join(
        slice_table.select(*KEY_COLUMNS, *new).with_columns(_in_table=pl.lit(value=True)),
        on=list(KEY_COLUMNS),
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if joined["_in_table"].null_count():
        msg = "a half-hour of an arm is not in the slice table"
        raise ValueError(msg)
    return joined.drop("_in_table")


def month_bootstrap_contrast(
    *,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    metric: str,
    where: pl.Expr | None = None,
    level_percent: float = REPORTED_LEVEL_PERCENT,
    n_resamples: int = REPORTED_RESAMPLES,
    month_of: MonthOfType = "power_fcst_init_time",
    slice_table: pl.DataFrame | None = None,
) -> ContrastInterval:
    """Return the paired contrast of two forecasts, with an interval from resampling whole months.

    The contrast is the mean over the shared half-hours of `treatment - reference` in `metric`. It
    reuses `studies.bootstrap.bootstrap_difference_at_level`, which resamples whole calendar months
    and the same months for both forecasts, so the pairing survives. The forecasts have no fitting
    seed here, so every row carries seed 0.

    The defaults are the reported contrast, a 95% interval from 2,000 resamples
    (`REPORTED_LEVEL_PERCENT`, `REPORTED_RESAMPLES`). The protocol's gain condition is a 99%
    interval, which needs the headline settings (`HEADLINE_LEVEL_PERCENT`, `HEADLINE_RESAMPLES`):
    call `headline_month_bootstrap_contrast`.

    Args:
        treatment: `scored_rows` of the forecast under test.
        reference: `scored_rows` of the forecast it is compared with.
        metric: A column of `METRIC_NAMES`.
        where: A filter on the shared rows, such as `pl.col("cold_nights")` or
            `pl.col("group") == "signed-MW"`. Every row if not given.
        level_percent: The interval's coverage in percent.
        n_resamples: How many resamples to draw.
        month_of: The time whose UTC calendar month labels a row: the forecast origin (the fold's
            scored month, the default) or the valid time.
        slice_table: The shared slice table. When given, its slice columns are joined to both
            frames, so `where` may name a slice on frames that carry none.

    Returns:
        The point estimate, the interval's bounds, and the number of half-hours and months.

    Raises:
        ValueError: If the forecasts share no row in the filter, or a half-hour of either is not
            in `slice_table`.
    """
    keys = list(KEY_COLUMNS)
    if slice_table is not None:
        treatment = _with_slice_columns(rows=treatment, slice_table=slice_table)
        reference = _with_slice_columns(rows=reference, slice_table=slice_table)
    filtered = treatment if where is None else treatment.filter(where)
    shared = filtered.select(*dict.fromkeys([*keys, metric])).join(
        reference.select(*keys, reference=pl.col(metric)), on=keys, how="inner", validate="1:1"
    )
    if shared.is_empty():
        msg = "the two forecasts share no half-hour in the filter"
        raise ValueError(msg)
    month = shared[month_of].dt.strftime("%Y-%m")
    key = pl.concat_str(
        pl.col("time_series_id").cast(pl.String),
        pl.col("power_fcst_init_time").dt.to_string(),
        pl.col("valid_time").dt.to_string(),
        separator="|",
    )
    losses = pl.concat(
        [
            shared.select(
                arm=pl.lit("treatment"), site=key, time=pl.col("valid_time"), value=pl.col(metric)
            ),
            shared.select(
                arm=pl.lit("reference"),
                site=key,
                time=pl.col("valid_time"),
                value=pl.col("reference"),
            ),
        ]
    ).with_columns(
        month=pl.concat([month, month]),
        seed=pl.lit(0, dtype=pl.Int64),
    )
    lower, upper = bootstrap_difference_at_level(
        losses=losses,
        treatment="treatment",
        reference="reference",
        metric="value",
        level=level_percent,
        n_resamples=n_resamples,
    )
    difference = float((shared[metric] - shared["reference"]).mean())  # ty: ignore[invalid-argument-type]
    return {
        "difference": difference,
        "lower": lower,
        "upper": upper,
        "level_percent": level_percent,
        "n_half_hours": shared.height,
        "n_months": month.n_unique(),
    }


def headline_month_bootstrap_contrast(
    *,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    metric: str,
    where: pl.Expr | None = None,
    month_of: MonthOfType = "power_fcst_init_time",
    slice_table: pl.DataFrame | None = None,
) -> ContrastInterval:
    """Return the headline contrast: a 99% interval from 20,000 resamples, for the gain condition.

    This is `month_bootstrap_contrast` with `HEADLINE_LEVEL_PERCENT` and `HEADLINE_RESAMPLES`.

    Args:
        treatment: As in `month_bootstrap_contrast`.
        reference: As in `month_bootstrap_contrast`.
        metric: As in `month_bootstrap_contrast`.
        where: As in `month_bootstrap_contrast`.
        month_of: As in `month_bootstrap_contrast`.
        slice_table: As in `month_bootstrap_contrast`.

    Returns:
        The point estimate, the interval's bounds, and the number of half-hours and months.
    """
    return month_bootstrap_contrast(
        treatment=treatment,
        reference=reference,
        metric=metric,
        where=where,
        level_percent=HEADLINE_LEVEL_PERCENT,
        n_resamples=HEADLINE_RESAMPLES,
        month_of=month_of,
        slice_table=slice_table,
    )


def reported_contrast_by_primary(
    *,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    metric: str,
    where: pl.Expr | None = None,
    month_of: MonthOfType = "power_fcst_init_time",
    slice_table: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return the reported contrast of each Primary: a 95% interval from 2,000 resamples.

    Each series is contrasted on its own half-hours, with `REPORTED_LEVEL_PERCENT` and
    `REPORTED_RESAMPLES`. A series with no shared half-hour in the filter has no row.

    Args:
        treatment: As in `month_bootstrap_contrast`.
        reference: As in `month_bootstrap_contrast`.
        metric: As in `month_bootstrap_contrast`.
        where: As in `month_bootstrap_contrast`, applied in addition to the series.
        month_of: As in `month_bootstrap_contrast`.
        slice_table: As in `month_bootstrap_contrast`.

    Returns:
        One row per series: `time_series_id`, `label`, and the `ContrastInterval` fields.
    """
    pieces = []
    for series_id, label in treatment.select("time_series_id", "label").unique().iter_rows():
        in_series = pl.col("time_series_id") == series_id
        try:
            result = month_bootstrap_contrast(
                treatment=treatment,
                reference=reference,
                metric=metric,
                where=in_series if where is None else in_series & where,
                level_percent=REPORTED_LEVEL_PERCENT,
                n_resamples=REPORTED_RESAMPLES,
                month_of=month_of,
                slice_table=slice_table,
            )
        except ValueError as error:
            if "share no half-hour" in str(error):
                continue
            raise
        pieces.append({"time_series_id": series_id, "label": label, **result})
    return pl.DataFrame(pieces).sort("label") if pieces else pl.DataFrame()


def avoidable_curtailment_cost(
    *,
    forecasts: Mapping[str, pl.DataFrame],
    series: pl.DataFrame,
    flows: pl.DataFrame,
    scored_start: datetime,
    fraction: float = HEADLINE_LIMIT_FRACTION,
    price_gbp_per_mwh: float = HEADLINE_PRICE_GBP_PER_MWH,
    lead_hours: tuple[float, float] = CURTAILMENT_LEAD_HOURS,
) -> pl.DataFrame:
    """Return each forecast's avoidable curtailment cost per series, in pounds.

    The avoidable cost is the cost of the forecast minus the cost of a perfect forecast on the same
    half-hours, from `studies.node_curtailment.avoidable_cost`. Each series is scored as its own
    node, in node-only mode, as experiment 277 scored the Primaries.

    **The cost is defined only for the signed-MW group.** The MVA-only Primaries read unsigned
    apparent power, so a reverse flow is hidden and the cost is wrong in kind. Their cost is null
    with the status `STATUS_UNSIGNED_METER`. A series in neither group is also null (status
    `STATUS_NOT_SIGNED_GROUP`). The limit also needs `MIN_HISTORY_MONTHS` months of flows before
    `scored_start`: with fewer, the cost is null with status `STATUS_SHORT_HISTORY`.

    Args:
        forecasts: Each arm's forecast: `time_series_id`, `power_fcst_init_time`, `valid_time`,
            `actual`, and the 11 quantile columns, all in MW.
        series: The series table of `score_forecast`.
        flows: Metered power in MW before the holdout: `time_series_id`, `valid_time`, `flow_mw`.
            The limit is `fraction` times the series' maximum before `scored_start`.
        scored_start: A UTC datetime. Flows from it onward never change a limit.
        fraction: The limit as a fraction of the earlier maximum.
        price_gbp_per_mwh: The price of curtailed energy.
        lead_hours: The lead window scored, `[low, high)` in hours.

    Returns:
        One row per series and arm with `arm`, `time_series_id`, `label`, `group`,
        `n_node_half_hours`, `avoidable_cost_gbp` (the energy-proxy cost minus the perfect
        forecast's, null where undefined), and `status` (`ok` or the reason the cost is null).
    """
    pieces = []
    needed_from = _months_before(instant=scored_start, months=MIN_HISTORY_MONTHS)
    for series_id, label in zip(series["time_series_id"], series["label"], strict=True):
        group = SERIES_GROUPS.get(str(label))
        earlier = flows.filter(
            pl.col("time_series_id") == series_id, pl.col("valid_time") < scored_start
        )
        first = earlier["valid_time"].min()
        undefined = None
        if group == "MVA-only":
            undefined = STATUS_UNSIGNED_METER
        elif group != "signed-MW":
            undefined = STATUS_NOT_SIGNED_GROUP
        elif earlier.is_empty() or first > needed_from:  # ty: ignore[unsupported-operator]
            undefined = STATUS_SHORT_HISTORY
        if undefined is not None:
            pieces.append(
                pl.DataFrame(
                    {
                        "arm": list(forecasts),
                        "time_series_id": [series_id] * len(forecasts),
                        "label": [label] * len(forecasts),
                        "group": [group] * len(forecasts),
                        "n_node_half_hours": [None] * len(forecasts),
                        "avoidable_cost_gbp": [None] * len(forecasts),
                        "status": [undefined] * len(forecasts),
                    },
                    schema={
                        "arm": pl.String,
                        "time_series_id": pl.Int64,
                        "label": pl.String,
                        "group": pl.String,
                        "n_node_half_hours": pl.Int64,
                        "avoidable_cost_gbp": pl.Float64,
                        "status": pl.String,
                    },
                )
            )
            continue
        arms = {
            name: frame.filter(pl.col("time_series_id") == series_id).with_columns(
                pl.col("time_series_id").cast(pl.Int64),
                seed=pl.lit(0, dtype=pl.Int64),
            )
            for name, frame in forecasts.items()
        }
        reference = next(iter(arms.values()))
        arms[PERFECT_ARM] = reference.with_columns(
            *[pl.col("actual").alias(name) for name in QUANTILE_COLUMNS]
        )
        table = _single_node_table(series_id=int(series_id), label=str(label))
        scaled = scaled_limits(
            table=table,
            flows=flows.filter(pl.col("time_series_id") == series_id).with_columns(
                pl.col("time_series_id").cast(pl.Int64)
            ),
            cutoff=scored_start,
            fraction=fraction,
        )
        result = node_curtailment(
            table=table_with_scaled_limits(table=table, scaled=scaled, fraction=fraction),
            node_forecasts=arms,
            generator_forecasts={},
            assumptions=replace(
                NodeCurtailmentAssumptions(use_generators=False),
                lead_min_hours=lead_hours[0],
                lead_max_hours=lead_hours[1],
                headline_price_gbp_per_mwh=price_gbp_per_mwh,
                prices_gbp_per_mwh=(price_gbp_per_mwh,),
            ),
        )
        avoidable = avoidable_cost(per_arm=result.per_arm, perfect_arm=PERFECT_ARM)
        pieces.append(
            avoidable.filter(pl.col("is_headline_price"), pl.col("arm") != PERFECT_ARM).select(
                "arm",
                time_series_id=pl.lit(series_id, dtype=pl.Int64),
                label=pl.lit(label),
                group=pl.lit(group),
                n_node_half_hours=pl.col("n_node_half_hours").cast(pl.Int64),
                avoidable_cost_gbp=pl.col("energy_proxy_gbp").cast(pl.Float64),
                status=pl.lit(STATUS_OK),
            )
        )
    return pl.concat(pieces, how="vertical_relaxed").sort("arm", "time_series_id")


def _single_node_table(*, series_id: int, label: str) -> NodeTable:
    """Return a node table with one node and no generators, a positive reading being the flow."""
    nodes = pl.DataFrame(
        {
            "node_id": [label],
            "time_series_id": [series_id],
            "node_type": ["BSP"],
            "parent_node_id": [None],
            "export_sign": [1],
            "sign_source": ["assumed: a positive demand reading is the flow that can breach"],
        },
        schema_overrides={"parent_node_id": pl.String, "export_sign": pl.Int64},
    )
    limits = pl.DataFrame(
        schema={
            "node_id": pl.String,
            "season": pl.String,
            "export_limit_mw": pl.Float64,
            "import_limit_mw": pl.Float64,
            "limit_source": pl.String,
        }
    )
    generators = pl.DataFrame(
        schema={
            "generator_id": pl.Int64,
            "node_id": pl.String,
            "capacity_mw": pl.Float64,
            "connection_date": pl.Date,
            "is_forecast": pl.Boolean,
        }
    )
    return NodeTable(nodes=nodes, limits=limits, generators=generators)

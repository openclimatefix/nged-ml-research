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
  series' own power, in points, over the readings before the scored period** (`twcrps_thresholds`).
  Experiment 277 also reported a 99th percentile threshold, which `twcrps_thresholds` takes as an
  argument.
- `pinball_q01`, `pinball_q05`, `pinball_q95`, `pinball_q99`: the pinball loss of that quantile.
- `coverage_q01`, `coverage_q05`, `coverage_q95`, `coverage_q99`: the share of rows where the actual
  is at or below that quantile.

**The avoidable curtailment cost** wraps `studies.node_curtailment.node_curtailment`, the function
experiment 277 scored the Primaries with, in its node-only mode, and `avoidable_cost`, which
subtracts the cost of a perfect forecast. Each series is its own node, a positive demand reading is
the flow that can breach, the limit is `HEADLINE_LIMIT_FRACTION` (0.5) of the series' maximum before
the scored period, the price is 2,000 pounds per MWh, and the lead window is 12 to 36 hours, all as
experiment 277 set them.

**The temperature and clear-sky inputs of the slices** are named in `studies.eval_protocol.slices`.

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
    SERIES_GROUPS,
    SLICE_NAMES,
    add_slice_columns,
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
BOOTSTRAP_LEVEL_PERCENT: Final[float] = 95.0
BOOTSTRAP_RESAMPLES: Final[int] = 2000
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")

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


def twcrps_thresholds(
    *,
    history: pl.DataFrame,
    series: pl.DataFrame,
    before: datetime,
    quantile: float = TW_THRESHOLD_QUANTILE,
) -> dict[int, float]:
    """Return each series' twCRPS threshold: a quantile of its earlier power, in points.

    Args:
        history: Power readings with `time_series_id`, `valid_time` (UTC), and `power_mw`.
        series: The series table of `score_forecast`, for each series' effective capacity.
        before: A UTC datetime. Only readings strictly before it count, so the threshold never
            sees the scored period.
        quantile: The quantile of the earlier readings, 0.95 by default (experiment 277 also used
            0.99).

    Returns:
        A map from `time_series_id` to the threshold in points of effective capacity.
    """
    capacity = dict(zip(series["time_series_id"], series["capacity_mw"], strict=True))
    earlier = history.filter(pl.col("valid_time") < before, pl.col("power_mw").is_finite())
    return {
        int(series_id): float(
            np.quantile(group["power_mw"].to_numpy() / capacity[series_id] * 100.0, quantile)
        )
        for (series_id,), group in earlier.group_by("time_series_id")
        if series_id in capacity
    }


def row_losses(
    *,
    forecast: pl.DataFrame,
    actuals: pl.DataFrame,
    series: pl.DataFrame,
    thresholds: Mapping[int, float],
    holdout_start: datetime = HOLDOUT_START,
) -> pl.DataFrame:
    """Return every scored half-hour with its losses, in points of effective capacity.

    A half-hour is scored when its actual value is finite and its valid time is before
    `holdout_start`. The output carries the keys, `lead_hours`, `label`, `group`, `capacity_mw`,
    `actual_points`, and one column per entry of `METRIC_NAMES`, in points (the coverage columns
    hold 0 or 1).

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
        ValueError: If a forecast row has no series entry or no twCRPS threshold.
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
    unknown = set(rows["time_series_id"].unique().to_list()) - set(thresholds)
    if unknown:
        msg = f"no twCRPS threshold for the series {sorted(unknown)}"
        raise ValueError(msg)
    scale = 100.0 / rows["capacity_mw"].to_numpy().astype(np.float64)
    actual = rows["actual"].to_numpy().astype(np.float64) * scale
    quantiles = rows.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64) * scale[:, None]
    threshold = np.array([thresholds[int(s)] for s in rows["time_series_id"]], dtype=np.float64)
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
        *[pl.Series(name, values) for name, values in columns.items()]
    )


def scored_rows(
    *,
    forecast: pl.DataFrame,
    actuals: pl.DataFrame,
    series: pl.DataFrame,
    context: pl.DataFrame,
    thresholds: Mapping[int, float],
    events: pl.DataFrame | None = None,
    first_readings: Mapping[int, datetime] | None = None,
) -> pl.DataFrame:
    """Return `row_losses` joined to the slice columns of `add_slice_columns`.

    Args:
        forecast: As in `row_losses`.
        actuals: As in `row_losses`.
        series: As in `row_losses`.
        context: One row per series and valid time: `time_series_id`, `valid_time`, `temperature_c`,
            `elevation_deg`, `pv_unit_shape`, and optionally `clear_sky_index`. The input columns
            are described in `studies.eval_protocol.slices`.
        thresholds: As in `row_losses`.
        events: Hand-labelled events, as in `add_slice_columns`.
        first_readings: First power reading of each series, as in `add_slice_columns`.

    Returns:
        The per-row losses with one boolean column per available slice.

    Raises:
        ValueError: If a scored row has no row in `context`.
    """
    losses = row_losses(forecast=forecast, actuals=actuals, series=series, thresholds=thresholds)
    joined = losses.join(
        context.with_columns(_has_context=pl.lit(value=True)),
        on=["time_series_id", "valid_time"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if joined["_has_context"].null_count():
        msg = "a scored half-hour has no row in the context table"
        raise ValueError(msg)
    joined = joined.drop("_has_context")
    return add_slice_columns(rows=joined, events=events, first_readings=first_readings)


def score_forecast(
    *,
    forecast: pl.DataFrame,
    actuals: pl.DataFrame,
    series: pl.DataFrame,
    context: pl.DataFrame,
    thresholds: Mapping[int, float],
    slices: Sequence[str] | None = None,
    events: pl.DataFrame | None = None,
    first_readings: Mapping[int, datetime] | None = None,
) -> pl.DataFrame:
    """Score a forecast: every metric, over every slice, for every series.

    Args:
        forecast: One row per series, forecast origin, and valid time, with `time_series_id`,
            `power_fcst_init_time`, `valid_time`, and the 11 quantile columns `q01` to `q99`, in MW.
        actuals: `time_series_id`, `valid_time`, and `actual` in MW.
        series: One row per series: `time_series_id`, `label` (such as `D5`), and `capacity_mw`.
        context: The weather and solar inputs of the slices, as `scored_rows` describes.
        thresholds: Each series' twCRPS threshold in points (`twcrps_thresholds`).
        slices: The slices to report. All the slices that the inputs allow if not given.
        events: Hand-labelled events (adds the two event slices).
        first_readings: First power reading of each series (adds the `cold_start_months` slice).

    Returns:
        One row per metric, slice, and series that holds at least one half-hour, with `metric`,
        `slice`, `time_series_id`, `label`, `group`, `group_note` (the label the protocol attaches
        to the group, such as "reactive power assumed zero"), `n_half_hours`, `thin` (true when
        `n_half_hours` is under `THIN_HALF_HOURS`), `value`, `unit` (`points` or `fraction`), and
        `value_mw` (null for a coverage).

    Raises:
        ValueError: If a requested slice is not available from the inputs.
    """
    rows = scored_rows(
        forecast=forecast,
        actuals=actuals,
        series=series,
        context=context,
        thresholds=thresholds,
        events=events,
        first_readings=first_readings,
    )
    available = [name for name in SLICE_NAMES if name in rows.columns]
    chosen = list(available if slices is None else slices)
    missing = [name for name in chosen if name not in available]
    if missing:
        msg = f"the slices {missing} need an input that was not passed"
        raise ValueError(msg)
    pieces = []
    for slice_name in chosen:
        part = rows.filter(pl.col(slice_name))
        if part.is_empty():
            continue
        means = part.group_by("time_series_id", "label", "group", "capacity_mw").agg(
            n_half_hours=pl.len(), **{name: pl.col(name).mean() for name in METRIC_NAMES}
        )
        pieces.append(means.with_columns(slice=pl.lit(slice_name)))
    long = (
        pl.concat(pieces)
        .unpivot(
            on=list(METRIC_NAMES),
            index=["slice", "time_series_id", "label", "group", "capacity_mw", "n_half_hours"],
            variable_name="metric",
            value_name="value",
        )
        .with_columns(
            thin=pl.col("n_half_hours") < THIN_HALF_HOURS,
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
        )
    )


def month_bootstrap_contrast(
    *,
    treatment: pl.DataFrame,
    reference: pl.DataFrame,
    metric: str,
    where: pl.Expr | None = None,
    level_percent: float = BOOTSTRAP_LEVEL_PERCENT,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    month_of: MonthOfType = "power_fcst_init_time",
) -> ContrastInterval:
    """Return the paired contrast of two forecasts, with an interval from resampling whole months.

    The contrast is the mean over the shared half-hours of `treatment - reference` in `metric`. It
    reuses `studies.bootstrap.bootstrap_difference_at_level`, which resamples whole calendar months
    and the same months for both forecasts, so the pairing survives. The forecasts have no fitting
    seed here, so every row carries seed 0.

    Args:
        treatment: `scored_rows` of the forecast under test.
        reference: `scored_rows` of the forecast it is compared with.
        metric: A column of `METRIC_NAMES`.
        where: A filter on the shared rows, such as `pl.col("cold_nights")` or
            `pl.col("group") == "signed-MW"`. Every row if not given.
        level_percent: The interval's coverage in percent, such as 95.0 or 99.0.
        n_resamples: How many resamples to draw. A 99% interval needs more than 2,000.
        month_of: The time whose UTC calendar month labels a row: the forecast origin (the fold's
            scored month, the default) or the valid time.

    Returns:
        The point estimate, the interval's bounds, and the number of half-hours and months.

    Raises:
        ValueError: If the forecasts share no row in the filter.
    """
    keys = list(KEY_COLUMNS)
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
        One row per series and arm with `arm`, `time_series_id`, `label`, `n_node_half_hours`, and
        `avoidable_cost_gbp`, which is the energy-proxy cost minus the perfect forecast's.
    """
    pieces = []
    for series_id, label in zip(series["time_series_id"], series["label"], strict=True):
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
                n_node_half_hours=pl.col("n_node_half_hours"),
                avoidable_cost_gbp=pl.col("energy_proxy_gbp"),
            )
        )
    return pl.concat(pieces).sort("arm", "time_series_id")


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

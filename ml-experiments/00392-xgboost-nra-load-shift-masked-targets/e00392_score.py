# ruff: noqa: E501, PLR0912, PLR0915
r"""Score experiment 392's few-fold screen under demand-v1 and print every number quoted.

Arms (forecasts stored as fractions of effective capacity, one file per Primary and refit):

- `r0_lagged`: experiment 381's monthly `control`; `r0_lagless`, `r1_*`, `rc1_*` (and `r2_*`,
  `rc2_*` once they exist): this experiment's `work/arm_preds/<arm>_<form>/`.
- `r5_lagged`: experiment 391's arm `n1` (`work/arm_preds/monthly/n1/`); skipped, with a message, if
  it holds no complete fold.

An arm-and-fold is scored only when the files of every requested Primary exist; otherwise it is
reported and skipped. Within a fold every arm that is present is scored on the same half-hours
(valid time before 2026-07-01 with a finite actual, lead up to 336 hours). Pooled numbers are taken
over the folds in which both arms of a contrast are present.

**Folds.** The four screen refits 2025-10-01, 2026-01-01, 2026-03-01, and 2026-06-01, each over the
origins from the refit to the next refit (2026-06 to 2026-06-30).

**Actuals (`--actuals`).** `cache`: experiment 377's `actuals_and_context` (the headline).
`observed`: experiment 87's `observed.parquet` (power in MW); a half-hour whose reading is null
there but finite in the cache is dropped from every arm, and the count is printed per fold and per
Primary.

**Row sets.** Event windows are the unpadded runs recovered from the feature cache's padded mask
(`e00392_fit.unpadded_runs`). `outside` = not inside an event and not within 48 hours after its end;
`during` = inside an event; `after7d` = within 7 days after an event's end (not inside another
event); `all` = every row. The headline band is lead 0 to 240 hours (days 1 to 10); the four lead
bands are 0-24, 24-72, 72-168, and 168-240 hours. Groups: signed-MW six, MVA-only eight, D15 and D18
(report only); placebos D2 and D4 are shown as a further view.

**Check 2, the exact definition.** Each unpadded event of a Primary is matched to a row of
`briefs/sw_block_detector_label_table.csv` when the Primary is named in the row's `pair` text and
the row's onset is within `MATCH_TOLERANCE_HOURS` of the event's first half-hour; the nearest
onset wins. The label's depth `|depth_mw|` is the event's depth; rows with confidence `low` are
reported separately. For an unmatched event the depth is estimated as the absolute mean change of
the actual (see below) and the event is reported separately as "estimated depth", which is not used
by the screen rule. For every arm, every forecast origin, and every valid time `t` in the window
`[start, end + 48 hours)` at leads up to 240 hours, the movement is `q50(origin, t) - q50(origin,
t - 7 days)`: the arm's own median for the same half-hour a week earlier from the same origin,
which exists whenever that week-earlier half-hour is itself in the forecast. Rows whose
week-earlier half-hour lies inside an event or within 48 hours after one are dropped. The actual
change is `actual(t) - actual(t - 7 days)` on the same rows. Per event and arm the movement and the
actual change are averaged over the rows, and the **share of the labelled depth followed** is

    share = mean(movement) * sign(mean actual change) / depth

(the movement is oriented to the direction in which the actual moved, so a forecast that stays at
the normal running arrangement level scores 0 and a forecast that follows the whole transfer scores
1). The window is `both` (inside the event plus the 48 hours after it, used by the screen rule),
or `inside` (`[start, end)`) or `after` (`[end, end + 48 hours)`) separately. An unpadded run that
contains more than one label-table onset of its Primary (an onset within `MATCH_TOLERANCE_HOURS`
before the run's start, or inside it) is a merged event: it is listed separately with its count and
excluded from every matched-depth share. Two further variants are printed: the unsigned share
`|mean(movement)| / depth` and the share with the sign taken from the label table (`depth_mw` as
written for the Primary named first in `pair`, the opposite sign for the others). A fold's value for
an arm is the mean of the shares of its events. The median is unchanged by widening, so check 2
does not depend on the widening factor.

Everything is scored raw and with each quantile's distance from the median multiplied by 1.15.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets \
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00392-*/e00392_score.py \
    --actuals cache
```
"""

import argparse
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import e00087_common as common  # ty: ignore[unresolved-import]
import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00379_score as ref379  # ty: ignore[unresolved-import]
import e00383_fit as base  # ty: ignore[unresolved-import]
import e00392_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import KEY_COLUMNS, MVA_ONLY_LABELS, SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ActualsSource = Literal["cache", "observed"]

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = fit.WORK_DIR
ARM_DIR: Final[Path] = fit.ARM_DIR
OUT: Final[Path] = WORK / "scores"
CONTROL_381_DIR: Final[Path] = (
    DATA
    / "00381-simple-weekly-xgboost-demand-against-gam-stack"
    / "work"
    / "arm_preds"
    / "monthly"
    / "control"
)
N1_391_DIR: Final[Path] = (
    DATA / "00391-xgboost-lag-surprise-second-stage" / "work" / "arm_preds" / "monthly" / "n1"
)
LABEL_TABLE: Final[Path] = DATA / "briefs" / "sw_block_detector_label_table.csv"
SOURCES: Final[dict[str, Path]] = {
    "r0_lagged": CONTROL_381_DIR,
    **{
        f"{arm}_{form}": ARM_DIR / f"{arm}_{form}"
        for arm in ("r0", "r1", "r2", "rc1", "rc2")
        for form in ("lagged", "lagless")
        if (arm, form) != ("r0", "lagged")
    },
    "r5_lagged": N1_391_DIR,
}
FOLD_STARTS: Final[tuple[datetime, ...]] = (
    datetime(2025, 10, 1, tzinfo=UTC),
    datetime(2026, 1, 1, tzinfo=UTC),
    datetime(2026, 3, 1, tzinfo=UTC),
    datetime(2026, 6, 1, tzinfo=UTC),
)
FOLD_ENDS: Final[tuple[datetime, ...]] = (
    datetime(2025, 11, 1, tzinfo=UTC),
    datetime(2026, 2, 1, tzinfo=UTC),
    datetime(2026, 4, 1, tzinfo=UTC),
    datetime(2026, 7, 1, tzinfo=UTC),
)
FOLD_NAMES: Final[tuple[str, ...]] = tuple(f"{d:%Y-%m}" for d in FOLD_STARTS)
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
MAX_LEAD_HOURS: Final[float] = 336.0
HEADLINE_LEAD_HOURS: Final[float] = 240.0
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("24-72 h", 24.0, 72.0),
    ("72-168 h", 72.0, 168.0),
    ("168-240 h", 168.0, 240.0),
)
WIDENINGS: Final[tuple[float, ...]] = (1.0, 1.15)
QUANTILES: Final[tuple[str, ...]] = metrics.QUANTILE_COLUMNS
TAILS: Final[tuple[str, ...]] = metrics.TAIL_COLUMNS
STEP_48_HOURS: Final[int] = 96
STEP_7_DAYS: Final[int] = 336
MATCH_TOLERANCE_HOURS: Final[float] = 12.0
MIN_ROWS_PER_EVENT: Final[int] = 48
CRPS_TOLERANCE: Final[float] = 0.02
FOLDS_NEEDED: Final[int] = 3
PLACEBOS: Final[tuple[str, ...]] = ("D2", "D4")
GROUP_VIEWS: Final[dict[str, pl.Expr]] = {
    "all 16": pl.lit(value=True),
    "signed-MW six": pl.col("label").is_in(list(SIGNED_MW_LABELS)),
    "MVA-only eight": pl.col("label").is_in(list(MVA_ONLY_LABELS)),
    "D15, D18 (report only)": pl.col("group") == "short-history",
    "placebos D2, D4": pl.col("label").is_in(list(PLACEBOS)),
}
ROW_SETS: Final[dict[str, pl.Expr]] = {
    "outside": ~pl.col("in_event") & ~pl.col("post48"),
    "during": pl.col("in_event"),
    "after7d": pl.col("post7d"),
    "all": pl.lit(value=True),
}
HEADLINE: Final[pl.Expr] = pl.col("lead_hours") <= HEADLINE_LEAD_HOURS
LOSS_COLUMNS: Final[tuple[str, ...]] = (
    "crps",
    *(f"pinball_{q}" for q in TAILS),
    *(f"coverage_{q}" for q in TAILS),
)
COST_FRACTION: Final[float] = 0.9
SHARE_COLUMN: Final[str] = "share_oriented"


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def lead_expression() -> pl.Expr:
    """Return the lead time in hours of a forecast row."""
    return (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0


def widen(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Scale each quantile's distance from the median by `factor`."""
    if factor == 1.0:
        return frame
    median = pl.col("q50")
    return frame.with_columns((median + (pl.col(c) - median) * factor).alias(c) for c in QUANTILES)


def read_fold(*, directory: Path, sites: list[str], fold: int) -> pl.DataFrame | None:
    """Return one arm's stored forecasts for a fold, or None if any Primary's file is missing."""
    refit = FOLD_STARTS[fold]
    paths = [directory / f"{site}_s0_{refit:%Y%m%dT%H%M}.parquet" for site in sites]
    if not paths or not all(p.exists() for p in paths):
        return None
    return pl.concat(
        [pl.read_parquet(p, columns=[*KEY_COLUMNS, *QUANTILES]) for p in paths]
    ).filter(pl.col("power_fcst_init_time") < FOLD_ENDS[fold])


def load_actuals(
    *, source: ActualsSource, series: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, object]:
    """Return the scoring actuals (MW), the cache's actuals (MW), and the feature cache."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=ref379.CACHE_DIR)
    cache_actuals, _ = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )
    cache_actuals = cache_actuals.filter(pl.col("actual").is_finite())
    if source == "cache":
        return cache_actuals, cache_actuals, cache
    observed = (
        pl.read_parquet(common.OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
        .filter(pl.col("time_series_id").is_in(series["time_series_id"].to_list()))
        .with_columns(pl.col("time_series_id").cast(series.schema["time_series_id"]))
        .select("time_series_id", "valid_time", actual=pl.col("power").cast(pl.Float64))
    )
    return observed.filter(pl.col("actual").is_finite()), cache_actuals, cache


def print_dropped(
    *, cache_actuals: pl.DataFrame, actuals: pl.DataFrame, series: pl.DataFrame
) -> None:
    """Print the half-hours that the cache scores and the observed file has flagged, per fold and Primary."""
    dropped = (
        cache_actuals.filter(pl.col("actual").is_finite())
        .join(
            actuals.select("time_series_id", "valid_time", observed=pl.col("actual")),
            on=["time_series_id", "valid_time"],
            how="left",
        )
        .filter(pl.col("observed").is_null())
        .join(series.select("time_series_id", "label"), on="time_series_id")
    )
    print(
        "\n== Half-hours dropped from every arm (finite in the cache, null in observed.parquet) =="
    )
    rows = []
    for fold, (start, end) in enumerate(zip(FOLD_STARTS, FOLD_ENDS, strict=True)):
        part = dropped.filter(pl.col("valid_time") >= start, pl.col("valid_time") < end)
        rows.append(part.group_by("label").agg(pl.len().alias(FOLD_NAMES[fold])))
        print(f"fold {FOLD_NAMES[fold]}: {part.height:,} dropped half-hours")
    table = rows[0]
    for other in rows[1:]:
        table = table.join(other, on="label", how="full", coalesce=True)
    table = table.fill_null(0).with_columns(total=pl.sum_horizontal(pl.exclude("label")))
    print(table.sort(pl.col("label").str.slice(1).cast(pl.Int32)))
    print(f"total dropped in the four folds: {int(table['total'].sum()):,}")


def event_tables(*, series: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the event-flag rows and the event table with matched label depths.

    The flag table holds, for each Primary and valid time that is inside an event, within 48 hours
    after one, or within 7 days after one, the booleans `in_event`, `post48`, and `post7d` (the
    last two exclude half-hours inside an event). The event table holds one row per unpadded event.
    """
    masks = base.event_masks()
    grid = common.grid_times()
    size = grid.len()
    label_of = dict(zip(series["time_series_id"], series["label"], strict=True))
    flag_pieces, event_rows = [], []
    for series_id, label in label_of.items():
        runs = fit.unpadded_runs(padded=masks[label])
        inside = np.zeros(size, dtype=bool)
        post48 = np.zeros(size, dtype=bool)
        post7d = np.zeros(size, dtype=bool)
        for start, end in runs:
            inside[start:end] = True
            post48[end : end + STEP_48_HOURS] = True
            post7d[end : end + STEP_7_DAYS] = True
            event_rows.append(
                (series_id, label, grid[start], grid[end] if end < size else grid[-1])
            )
        post48 &= ~inside
        post7d &= ~inside
        keep = np.flatnonzero(inside | post7d)
        flag_pieces.append(
            pl.DataFrame(
                {
                    "time_series_id": pl.Series(
                        [series_id] * keep.size, dtype=series.schema["time_series_id"]
                    ),
                    "valid_time": grid.gather(keep),
                    "in_event": inside[keep],
                    "post48": post48[keep],
                    "post7d": post7d[keep],
                }
            )
        )
    events = pl.DataFrame(
        event_rows,
        schema={
            "time_series_id": series.schema["time_series_id"],
            "label": pl.String,
            "start": pl.Datetime("us", "UTC"),
            "end": pl.Datetime("us", "UTC"),
        },
        orient="row",
    ).sort("label", "start")
    return pl.concat(flag_pieces), match_labels(events=events)


def match_labels(*, events: pl.DataFrame) -> pl.DataFrame:
    """Attach the label table's depth to each event (`depth_abs`, `convention_sign`, `kind`).

    `kind` is `labelled`, `labelled low-confidence`, `estimated depth` (no matching label), or
    `merged event` (the run holds more than one label-table onset; no depth is matched).
    """
    table = pl.read_csv(LABEL_TABLE).with_columns(
        onset=pl.col("onset_utc").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC", time_unit="us")
    )
    labels = table.to_dicts()
    out = []
    for event in events.iter_rows(named=True):
        onsets = [
            row["onset"]
            for row in labels
            if event["label"] in re.findall(r"D\d+", row["pair"])
            and event["start"] - timedelta(hours=MATCH_TOLERANCE_HOURS)
            <= row["onset"]
            < event["end"]
        ]
        if len(onsets) > 1:
            out.append((None, 0.0, "merged event", None))
            continue
        best, best_gap = None, MATCH_TOLERANCE_HOURS
        for row in labels:
            names = re.findall(r"D\d+", row["pair"])
            if event["label"] not in names:
                continue
            gap = abs((row["onset"] - event["start"]).total_seconds()) / 3600.0
            if gap <= best_gap:
                best, best_gap = row, gap
        if best is None:
            out.append((None, 0.0, "estimated depth", None))
            continue
        names = re.findall(r"D\d+", best["pair"])
        sign = 1.0 if names[0] == event["label"] else -1.0
        kind = "labelled low-confidence" if best["confidence"] == "low" else "labelled"
        depth = float(best["depth_mw"])
        out.append((abs(depth), sign * (1.0 if depth >= 0 else -1.0), kind, best["confidence"]))
    return events.with_row_index("event").with_columns(
        depth_abs=pl.Series([o[0] for o in out], dtype=pl.Float64),
        convention_sign=pl.Series([o[1] for o in out], dtype=pl.Float64),
        kind=pl.Series([o[2] for o in out], dtype=pl.String),
        confidence=pl.Series([o[3] for o in out], dtype=pl.String),
    )


# ---------------------------------------------------------------------------------------------
# Scoring one arm in one fold
# ---------------------------------------------------------------------------------------------


def score_arm(
    *,
    forecast: pl.DataFrame,
    actuals: pl.DataFrame,
    series: pl.DataFrame,
    thresholds: pl.DataFrame,
    flags: pl.DataFrame,
) -> pl.DataFrame:
    """Return the per-row losses of one arm (float32) with the event flags and the lead band."""
    losses = metrics.row_losses(
        forecast=forecast, actuals=actuals, series=series, thresholds=thresholds
    )
    band = pl.lit("240-336 h")
    for name, low, high in reversed(LEAD_BANDS):
        band = (
            pl.when((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))
            .then(pl.lit(name))
            .otherwise(band)
        )
    return (
        losses.select(
            *KEY_COLUMNS,
            "lead_hours",
            "label",
            "group",
            *[pl.col(c).cast(pl.Float32) for c in LOSS_COLUMNS],
        )
        .join(flags, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(
            pl.col("in_event", "post48", "post7d").fill_null(value=False),
            band=band,
        )
    )


def event_moves(
    *, forecast: pl.DataFrame, actuals: pl.DataFrame, flags: pl.DataFrame, events: pl.DataFrame
) -> pl.DataFrame:
    """Return, per event, the mean movement and mean actual change in each window (check 2)."""
    week = pl.duration(days=7)
    after_window = pl.duration(hours=48)
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    current = forecast.filter(lead_expression() <= HEADLINE_LEAD_HOURS).select(*keys, "q50")
    previous = current.select(
        "time_series_id",
        "power_fcst_init_time",
        valid_time=pl.col("valid_time") + week,
        q50_previous=pl.col("q50"),
    )
    actual_now = actuals.select("time_series_id", "valid_time", "actual")
    actual_before = actuals.select(
        "time_series_id", valid_time=pl.col("valid_time") + week, actual_previous=pl.col("actual")
    )
    bad_before = (
        flags.filter(pl.col("in_event") | pl.col("post48"))
        .select("time_series_id", valid_time=pl.col("valid_time") + week)
        .with_columns(previous_bad=pl.lit(value=True))
    )
    moves = (
        current.join(previous, on=keys)
        .join(actual_now, on=["time_series_id", "valid_time"])
        .join(actual_before, on=["time_series_id", "valid_time"])
        .join(bad_before, on=["time_series_id", "valid_time"], how="left")
        .filter(pl.col("previous_bad").is_null())
        .with_columns(
            move=pl.col("q50") - pl.col("q50_previous"),
            change=pl.col("actual") - pl.col("actual_previous"),
        )
        .join(events.select("time_series_id", "event", "start", "end"), on="time_series_id")
        .filter(
            pl.col("valid_time") >= pl.col("start"),
            pl.col("valid_time") < pl.col("end") + after_window,
        )
        .with_columns(inside=pl.col("valid_time") < pl.col("end"))
    )
    windows = {"both": pl.lit(value=True), "inside": pl.col("inside"), "after": ~pl.col("inside")}
    pieces = [
        moves.filter(cond)
        .group_by("event")
        .agg(n=pl.len(), move=pl.col("move").mean(), change=pl.col("change").mean())
        .with_columns(window=pl.lit(name))
        for name, cond in windows.items()
    ]
    return pl.concat(pieces)


def event_shares(*, moves: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Return the shares of the labelled depth followed, per event and window."""
    joined = moves.join(
        events.select("event", "label", "kind", "confidence", "depth_abs", "convention_sign"),
        on="event",
    )
    depth = (
        pl.when(pl.col("kind") == "estimated depth")
        .then(pl.col("change").abs())
        .otherwise(pl.col("depth_abs"))
    )
    return joined.filter(
        pl.col("n") >= MIN_ROWS_PER_EVENT, pl.col("kind") != "merged event"
    ).with_columns(
        share_oriented=pl.col("move") * pl.col("change").sign() / depth,
        share_abs=pl.col("move").abs() / depth,
        share_conv=pl.when(pl.col("kind") == "estimated depth")
        .then(None)
        .otherwise(pl.col("move") * pl.col("convention_sign") / depth),
    )


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------


def means(
    *, rows: pl.DataFrame, where: pl.Expr, by: list[str], columns: tuple[str, ...] = ("crps",)
) -> pl.DataFrame:
    """Return the row count and the mean of `columns` per `by`."""
    return (
        rows.filter(where)
        .group_by(by)
        .agg(pl.len().alias("n"), *[pl.col(c).mean() for c in columns])
        .sort(by)
    )


def contrast_table(
    *,
    rows: pl.DataFrame,
    arm: str,
    control: str,
    where: pl.Expr,
    by: list[str],
    metric: str = "crps",
) -> pl.DataFrame:
    """Return `arm - control` in `metric` per `by` and fold, with the pooled value over common folds."""
    keep = rows.filter(where, pl.col("arm").is_in([arm, control]))
    aggregated = keep.group_by([*by, "fold", "arm"]).agg(n=pl.len(), total=pl.col(metric).sum())
    left = aggregated.filter(pl.col("arm") == arm).drop("arm")
    right = aggregated.filter(pl.col("arm") == control).drop("arm")
    both = left.join(right, on=[*by, "fold"], suffix="_control")
    per_fold = both.select(
        *by,
        "fold",
        diff=pl.col("total") / pl.col("n") - pl.col("total_control") / pl.col("n_control"),
    )
    pooled = (
        both.group_by(by)
        .agg(
            diff=(pl.col("total").sum() - pl.col("total_control").sum()) / pl.col("n").sum(),
            n=pl.col("n").sum(),
        )
        .with_columns(fold=pl.lit(-1, dtype=pl.Int64))
    )
    wide = pl.concat(
        [per_fold.with_columns(pl.col("fold").cast(pl.Int64)), pooled.select(*by, "fold", "diff")]
    ).with_columns(
        column=pl.col("fold").map_elements(
            lambda f: "pooled" if f < 0 else FOLD_NAMES[f], return_dtype=pl.String
        )
    )
    out = (
        wide.pivot(on="column", index=by, values="diff")
        if by
        else wide.select("column", "diff").transpose(column_names="column")
    )
    return ordered(frame=out.sort(by) if by else out, index=by)


def ordered(*, frame: pl.DataFrame, index: list[str]) -> pl.DataFrame:
    """Return `frame` with the index columns first, then the fold columns in time order, then `pooled`."""
    wanted = [c for c in (*FOLD_NAMES, "pooled") if c in frame.columns]
    return frame.select(*index, *wanted)


def control_of(arm: str) -> str:
    """Return the R0 arm of the same form as `arm`."""
    return f"r0_{arm.split('_')[1]}"


def folds_of(*, rows: pl.DataFrame, arm: str) -> set[int]:
    """Return the folds in which `arm` has rows."""
    return set(rows.filter(pl.col("arm") == arm)["fold"].unique().to_list())


def fold_diff(*, rows: pl.DataFrame, arm: str, control: str, where: pl.Expr) -> dict[int, float]:
    """Return `arm - control` mean CRPS per fold for the folds both arms hold."""
    table = contrast_table(rows=rows, arm=arm, control=control, where=where, by=[])
    return {
        i: float(table[FOLD_NAMES[i]][0])
        for i in range(len(FOLD_NAMES))
        if FOLD_NAMES[i] in table.columns
    }


def fold_shares(*, shares: pl.DataFrame, arm: str, kinds: tuple[str, ...]) -> dict[int, float]:
    """Return the mean share (`both` window) of the events of `kinds`, per fold, for `arm`."""
    part = shares.filter(
        pl.col("arm") == arm, pl.col("window") == "both", pl.col("kind").is_in(list(kinds))
    )
    grouped = part.group_by("fold").agg(pl.col(SHARE_COLUMN).mean())
    return {int(f): float(s) for f, s in zip(grouped["fold"], grouped[SHARE_COLUMN], strict=True)}


def print_check_1(*, rows: pl.DataFrame, arms: list[str]) -> None:
    """Print check 1: arm minus R0 CRPS outside events and the 48 hours after (points)."""
    outside = ROW_SETS["outside"]
    for arm in arms:
        if arm.startswith("r0_"):
            continue
        control = control_of(arm)
        if control not in arms:
            print(f"\n== Check 1 {arm}: control {control} not scored ==")
            continue
        print(
            f"\n== Check 1: {arm} minus {control}, CRPS in points of effective capacity, outside events and 48 h after =="
        )
        for view, cond in GROUP_VIEWS.items():
            print(f"-- {view}, headline band (days 1 to 10)")
            print(
                contrast_table(
                    rows=rows, arm=arm, control=control, where=outside & HEADLINE & cond, by=[]
                )
            )
        print("-- all 16, by lead band")
        print(
            contrast_table(
                rows=rows,
                arm=arm,
                control=control,
                where=outside & (pl.col("band") != "240-336 h"),
                by=["band"],
            )
        )
        print("-- by Primary, headline band, pooled over folds")
        by_label = contrast_table(
            rows=rows, arm=arm, control=control, where=outside & HEADLINE, by=["label"]
        )
        print(by_label.select("label", "pooled").sort(pl.col("label").str.slice(1).cast(pl.Int32)))


def print_check_4(*, rows: pl.DataFrame, arms: list[str]) -> None:
    """Print check 4: coverage and pinball at the tail quantiles, per row set, pooled over each arm's folds."""
    common_folds = set.intersection(*[folds_of(rows=rows, arm=a) for a in arms])
    print(
        f"\n== Check 4: tails, headline band, pooled over the folds all arms hold: {sorted(FOLD_NAMES[i] for i in common_folds)} =="
    )
    in_common = pl.col("fold").is_in(sorted(common_folds))
    for name, cond in ROW_SETS.items():
        print(f"-- {name}")
        print(
            means(
                rows=rows,
                where=cond & HEADLINE & in_common,
                by=["arm"],
                columns=("crps", *LOSS_COLUMNS[1:]),
            )
        )
    print("-- 5% to 95% band coverage is coverage_q95 - coverage_q05; nominal 0.90")


def _word(outcome: bool | None) -> str:
    """Return `pass`, `fail`, or `not evaluable` for a rule outcome."""
    return "not evaluable" if outcome is None else "pass" if outcome else "fail"


def print_screen(*, rows: pl.DataFrame, shares: pl.DataFrame, arms: list[str]) -> None:
    """Print the advance rule of the plan for R1 and RC1."""
    print(
        "\n== Screen rule (a): CRPS outside events and 48 h after no worse than +0.02 points vs control in >= 3 of 4 folds =="
    )
    outcome_a: dict[str, bool | None] = {}
    for arm in ("r1", "rc1", "r2", "rc2"):
        passes: list[bool | None] = []
        for form in ("lagged", "lagless"):
            name = f"{arm}_{form}"
            if name not in arms or control_of(name) not in arms:
                continue
            diffs = fold_diff(
                rows=rows, arm=name, control=control_of(name), where=ROW_SETS["outside"] & HEADLINE
            )
            good = sum(d <= CRPS_TOLERANCE for d in diffs.values())
            ok = good >= FOLDS_NEEDED and len(diffs) == len(FOLD_NAMES)
            if len(diffs) < len(FOLD_NAMES):
                print(
                    f"{name}: only {len(diffs)} of {len(FOLD_NAMES)} folds present; not evaluable"
                )
                passes.append(None)
                continue
            passes.append(ok)
            print(
                f"{name}: per fold { ({FOLD_NAMES[i]: round(d, 4) for i, d in diffs.items()}) }; {good} of {len(diffs)} folds within +{CRPS_TOLERANCE}; {'pass' if ok else 'fail'}"
            )
        outcome_a[arm] = None if not passes or None in passes else all(passes)
    print(
        "\n== Screen rule (b): lagless form follows less of the labelled depth than R0 lagless in >= 3 of 4 folds =="
    )
    print(
        f"(share = {SHARE_COLUMN}, window both, events with a matching label incl. low confidence)"
    )
    kinds = ("labelled", "labelled low-confidence")
    outcome_b: dict[str, bool | None] = {}
    r0 = fold_shares(shares=shares, arm="r0_lagless", kinds=kinds)
    for arm in ("r1", "rc1", "r2", "rc2"):
        name = f"{arm}_lagless"
        if name not in arms or "r0_lagless" not in arms:
            outcome_b[arm] = None
            continue
        mine = fold_shares(shares=shares, arm=name, kinds=kinds)
        folds = sorted(set(mine) & set(r0))
        less = sum(mine[f] < r0[f] for f in folds)
        if len(folds) < len(FOLD_NAMES):
            outcome_b[arm] = None
            print(
                f"{name}: only {len(folds)} of {len(FOLD_NAMES)} folds with events; rule not evaluable"
            )
            continue
        outcome_b[arm] = less >= FOLDS_NEEDED
        print(
            f"{name}: share per fold { ({FOLD_NAMES[f]: round(mine[f], 3) for f in folds}) } vs r0_lagless { ({FOLD_NAMES[f]: round(r0[f], 3) for f in folds}) }; less in {less} of {len(folds)} folds; {'pass' if outcome_b[arm] else 'fail'}"
        )
    beats: dict[str, bool | None] = {}
    for treatment, control in (("r1", "rc1"), ("r2", "rc2")):
        beats[treatment] = None
        a, b = f"{treatment}_lagless", f"{control}_lagless"
        if a in arms and b in arms:
            mine, other = (
                fold_shares(shares=shares, arm=a, kinds=kinds),
                fold_shares(shares=shares, arm=b, kinds=kinds),
            )
            folds = sorted(set(mine) & set(other))
            less = sum(mine[f] < other[f] for f in folds)
            if len(folds) < len(FOLD_NAMES):
                print(
                    f"{treatment} beats {control} on (b): only {len(folds)} of {len(FOLD_NAMES)} folds present; not evaluable"
                )
                continue
            beats[treatment] = less >= FOLDS_NEEDED
            print(
                f"{treatment} beats {control} on (b) in {less} of {len(folds)} folds: {'yes' if beats[treatment] else 'no'}"
            )
    for arm in ("r1", "r2"):
        parts = (outcome_a.get(arm), outcome_b.get(arm), beats.get(arm))
        if arm == "r2" and all(p is None for p in parts):
            continue
        label = f"(a) {_word(parts[0])}, (b) vs R0 {_word(parts[1])}, beats {'RC1' if arm == 'r1' else 'RC2'} {_word(parts[2])}"
        if None in parts:
            print(f"{arm}: {label} -> advance rule not evaluable")
        else:
            print(f"{arm}: {label} -> {'ADVANCES' if all(parts) else 'does not advance'}")
    for arm in ("rc1", "rc2"):
        if outcome_a.get(arm) is None and outcome_b.get(arm) is None:
            continue
        print(
            f"{arm} (no verdict, comparison arm): (a) {_word(outcome_a.get(arm))}, (b) vs R0 {_word(outcome_b.get(arm))}"
        )


def print_check_2(*, shares: pl.DataFrame, events: pl.DataFrame) -> None:
    """Print check 2: the share of the labelled depth followed, per arm, fold, and event kind."""
    pl.Config.set_tbl_rows(400)
    print("\n== Check 2: events (unpadded) and their matched label depth ==")
    print(
        events.filter(pl.col("end") >= FOLD_STARTS[0]).select(
            "event", "label", "start", "end", "kind", "confidence", "depth_abs"
        )
    )
    merged = events.filter(pl.col("kind") == "merged event", pl.col("end") >= FOLD_STARTS[0])
    print(f"\n== Check 2: merged events (excluded from the shares): {merged.height} ==")
    if not merged.is_empty():
        print(merged.select("event", "label", "start", "end"))
    if shares.is_empty():
        print("no event has enough rows in the scored folds")
        return
    for column in ("share_oriented", "share_abs", "share_conv"):
        print(f"\n== Check 2: mean {column} over events, per arm and fold (window = both) ==")
        for kind in ("labelled", "labelled low-confidence", "estimated depth"):
            part = shares.filter(pl.col("window") == "both", pl.col("kind") == kind)
            if part.is_empty():
                continue
            table = part.group_by("arm", "fold").agg(pl.col(column).mean(), n_events=pl.len())
            wide = table.with_columns(
                fold_name=pl.col("fold").map_elements(
                    lambda f: FOLD_NAMES[f], return_dtype=pl.String
                )
            ).pivot(on="fold_name", index="arm", values=[column, "n_events"])
            pooled = part.group_by("arm").agg(
                pooled_mean=pl.col(column).mean(),
                pooled_median=pl.col(column).median(),
                n_event_folds=pl.len(),
            )
            print(f"-- {kind}")
            print(wide.join(pooled, on="arm").sort("arm"))
    for window in ("inside", "after"):
        part = shares.filter(pl.col("window") == window, pl.col("kind") != "estimated depth")
        print(f"\n== Check 2: mean share_oriented, window {window}, matched events, by arm ==")
        print(part.group_by("arm").agg(pl.col(SHARE_COLUMN).mean(), n=pl.len()).sort("arm"))
    print("\n== Check 2: per event (share_oriented, window both) ==")
    per_event = shares.filter(pl.col("window") == "both").with_columns(
        event_fold=pl.concat_str(
            "label", pl.col("event").cast(pl.String), pl.col("fold").cast(pl.String), separator="/"
        )
    )
    print(
        per_event.pivot(
            on="arm",
            index=["event_fold", "kind", "n"],
            values=SHARE_COLUMN,
            aggregate_function="first",
        ).sort("event_fold")
    )


def cost_report(
    *, frames: dict[str, list[pl.DataFrame]], series: pl.DataFrame, flows: pl.DataFrame
) -> None:
    """Print the avoidable curtailment cost at the signed-MW six in three row sets (383's function)."""
    if not frames:
        return
    present = {arm: {int(f["fold"][0]) for f in parts} for arm, parts in frames.items()}
    common_folds = set.intersection(*present.values())
    print(
        f"\n== Avoidable curtailment cost (GBP, signed-MW six, limit fraction {COST_FRACTION}); folds all arms hold: {sorted(FOLD_NAMES[i] for i in common_folds)} =="
    )
    if not common_folds:
        print("no fold is held by every arm; cost skipped")
        return
    signed = series.filter(pl.col("label").is_in(list(SIGNED_MW_LABELS)))
    sets = {
        "during events": pl.col("in_event"),
        "7 days after events": pl.col("post7d"),
        "all rows": pl.lit(value=True),
    }
    for name, cond in sets.items():
        forecasts = {
            arm: pl.concat([p for p in parts if int(p["fold"][0]) in common_folds])
            .filter(cond)
            .drop("in_event", "post7d", "fold")
            for arm, parts in frames.items()
        }
        parts_out = []
        for label in SIGNED_MW_LABELS:
            one = signed.filter(pl.col("label") == label)
            try:
                parts_out.append(
                    metrics.avoidable_curtailment_cost(
                        forecasts=forecasts,
                        series=one,
                        flows=flows,
                        scored_start=SCORED_FIRST,
                        fraction=COST_FRACTION,
                    )
                )
            except ValueError as error:
                print(f"{label}: {name}: no cost ({error})")
        if not parts_out:
            continue
        cost = pl.concat(parts_out)
        print(f"-- {name}")
        wide = cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label")
        print(wide)
        print(wide.select(pl.exclude("label").sum()))


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main() -> None:
    """Score the screen folds and print the checks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--actuals", choices=["cache", "observed"], default="cache")
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    args = parser.parse_args()
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(260)
    OUT.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table(
        labels=tuple(args.sites) if args.sites else adapter.PRIMARY_LABELS
    )
    sites = series["label"].to_list()
    actuals, cache_actuals, _ = load_actuals(source=args.actuals, series=series)
    print(f"actuals source: {args.actuals}; Primaries: {sites}")
    if args.actuals == "observed":
        print_dropped(cache_actuals=cache_actuals, actuals=actuals, series=series)
    thresholds = metrics.twcrps_thresholds(
        history=cache_actuals.rename({"actual": "power_mw"}), series=series, before=SCORED_FIRST
    )
    flags, events = event_tables(series=series)
    flows = actuals.rename({"actual": "flow_mw"})
    rows_by_width: dict[float, list[pl.DataFrame]] = {w: [] for w in WIDENINGS}
    cost_by_width: dict[float, dict[str, list[pl.DataFrame]]] = {w: {} for w in WIDENINGS}
    shares_parts: list[pl.DataFrame] = []
    available: dict[str, list[int]] = {}
    for fold, name in enumerate(FOLD_NAMES):
        stored: dict[str, pl.DataFrame] = {}
        for arm, directory in SOURCES.items():
            frame = read_fold(directory=directory, sites=sites, fold=fold)
            if frame is None:
                print(f"fold {name}: arm {arm} skipped (files missing in {directory})")
                continue
            stored[arm] = adapter.forecast_frame(
                stored=frame, series=series, unit="fraction"
            ).filter(lead_expression() <= MAX_LEAD_HOURS)
            available.setdefault(arm, []).append(fold)
        if not stored:
            continue
        finite = actuals.filter(pl.col("valid_time") < metrics.HOLDOUT_START)
        shared = None
        for frame in stored.values():
            keys = frame.select(*KEY_COLUMNS).join(
                finite.select("time_series_id", "valid_time"), on=["time_series_id", "valid_time"]
            )
            shared = keys if shared is None else shared.join(keys, on=list(KEY_COLUMNS))
        assert shared is not None
        shared = shared.sort(*KEY_COLUMNS)
        print(f"fold {name}: arms {list(stored)}; shared half-hours {shared.height:,}")
        for arm, frame in stored.items():
            stored[arm] = frame.join(shared, on=list(KEY_COLUMNS), how="semi").sort(*KEY_COLUMNS)
            moves = event_moves(forecast=stored[arm], actuals=actuals, flags=flags, events=events)
            shares_parts.append(
                event_shares(moves=moves, events=events).with_columns(
                    arm=pl.lit(arm), fold=pl.lit(fold, dtype=pl.Int64)
                )
            )
        for width in WIDENINGS:
            for arm, frame in stored.items():
                widened = widen(frame=frame, factor=width)
                scored = score_arm(
                    forecast=widened,
                    actuals=actuals,
                    series=series,
                    thresholds=thresholds,
                    flags=flags,
                )
                rows_by_width[width].append(
                    scored.with_columns(arm=pl.lit(arm), fold=pl.lit(fold, dtype=pl.Int64))
                )
                cost_frame = (
                    widened.filter(
                        (lead_expression() >= metrics.CURTAILMENT_LEAD_HOURS[0])
                        & (lead_expression() < metrics.CURTAILMENT_LEAD_HOURS[1])
                    )
                    .join(actuals, on=["time_series_id", "valid_time"])
                    .join(
                        flags.select("time_series_id", "valid_time", "in_event", "post7d"),
                        on=["time_series_id", "valid_time"],
                        how="left",
                    )
                    .with_columns(
                        pl.col("in_event", "post7d").fill_null(value=False),
                        fold=pl.lit(fold, dtype=pl.Int64),
                    )
                )
                cost_by_width[width].setdefault(arm, []).append(cost_frame)
    print("\n== Arms and folds scored ==")
    for arm in SOURCES:
        print(f"{arm}: {[FOLD_NAMES[f] for f in available.get(arm, [])] or 'none (skipped)'}")
    if "r5_lagged" not in available:
        print("R5 (experiment 391's N1) skipped: no complete fold in its arm_preds/monthly/n1/")
    if not shares_parts:
        print("nothing scored")
        return
    shares = pl.concat(shares_parts)
    print_check_2(shares=shares, events=events)
    suffix = "" if args.actuals == "cache" else "_observed"
    written = []
    for width in WIDENINGS:
        rows = pl.concat(rows_by_width[width])
        arms = sorted(rows["arm"].unique().to_list())
        written.append(rows.with_columns(widening=pl.lit(width)))
        print(f"\n{'=' * 100}\nWIDENING {width} (actuals: {args.actuals})\n{'=' * 100}")
        print(
            "\n== CRPS, headline band, by arm and fold, per row set (points of effective capacity) =="
        )
        for set_name, cond in ROW_SETS.items():
            table = means(rows=rows, where=cond & HEADLINE, by=["arm", "fold"])
            print(f"-- {set_name}")
            print(
                table.with_columns(
                    fold_name=pl.col("fold").map_elements(
                        lambda f: FOLD_NAMES[f], return_dtype=pl.String
                    )
                )
                .pivot(on="fold_name", index="arm", values="crps")
                .sort("arm")
                .pipe(lambda f: ordered(frame=f, index=["arm"]))
            )
        print("\n== Rows per set, headline band, all arms of the first listed fold ==")
        print(
            means(rows=rows.filter(pl.col("arm") == arms[0]), where=HEADLINE, by=["fold", "group"])
        )
        print_check_1(rows=rows, arms=arms)
        print_check_4(rows=rows, arms=arms)
        print_screen(rows=rows, shares=shares, arms=arms)
        cost_report(
            frames=dict(cost_by_width[width]),
            series=series,
            flows=flows,
        )
    pl.concat(written).write_parquet(OUT / f"rows{suffix}.parquet")
    shares.write_parquet(OUT / f"event_shares{suffix}.parquet")
    print(f"\nwrote {OUT / f'rows{suffix}.parquet'}")


if __name__ == "__main__":
    main()

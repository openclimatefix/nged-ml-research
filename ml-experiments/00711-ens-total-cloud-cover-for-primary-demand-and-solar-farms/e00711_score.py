r"""Score experiment 711: every planned contrast, the controls, and the decision rule of section 8.

The scorer reads the predictions that `e00711_fit.py` wrote to the experiment store (one set per
arm,
fold, and seed), turns each row into a loss, and prints, in this order: the arms found and each
arm's input columns, the minimum detectable effect (before any contrast), the six planned contrasts
at 1 - 0.05/9 = 99.44%, the controls, the verdicts, and the exploratory readings.

**Losses.** CRPS is twice the mean pinball loss over the 11 quantile levels, in points of effective
capacity (100 times the fraction of effective capacity), with the absolute error of the median
beside it. Each Primary demand forecast is widened by 1.15 about the median at q01, q05, q95, and
q99, as experiment 707 widens `R_F`. Each solar farm forecast is not widened. A farm's error is
divided by that farm's own capacity before any mean.

**Intervals.** Per-row losses of each seed go through `experiment_store.losses_for_bootstrap`, which
raises when the arms of a contrast hold different seeds or rows, and then through
`studies.bootstrap.bootstrap_difference_at_level`, which resamples whole calendar months paired
across arms and draws a fitting seed, with 50,000 resamples. The bootstrap restarts its random
stream at every call, so every arm pair meets the same resampled months and seeds. The decisive
interval is the wider of the bootstrap interval and the t interval on the monthly differences.
Each contrast is read on the seeds that its two arms share.

**Decision rule** (`verdict`): the first test that holds decides. Lower above 0: refuted. Upper
below
-0.02: supported. Upper below 0: small gain. Lower at or above -0.02 and upper at or above 0: not
supported. Otherwise inconclusive. Negative favours real cloud.

**Anonymity.** Solar farms are labelled A to F. The scorer prints no farm identifier, no coordinate,
and no calendar date of a farm's data.

```bash
# environment as in e00711_common.py; PYTHONPATH holds the folders of experiments 707 and 711
uv run --no-sync python ml-experiments/00711-*/e00711_score.py sky-index   # once
uv run --no-sync python ml-experiments/00711-*/e00711_score.py gate
uv run --no-sync python ml-experiments/00711-*/e00711_score.py score --resamples 50000
uv run --no-sync python ml-experiments/00711-*/e00711_score.py score --stage B1
```

**Every run also saves its full output** to `work/scores/<command>[-stage]_<UTC time>.txt` under the
experiment's data folder.
"""

import argparse
import contextlib
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Final, Literal, TextIO

import e00711_arms as arms
import e00711_common as common
import e00711_gate as gate_marker
import e00711_jobs as jobs
import numpy as np
import polars as pl
from scipy import stats
from studies.era5_ladder import CLEAR_SKY_INDEX_THRESHOLDS
from studies.eval_protocol import metrics

from studies import bootstrap, experiment_store

StageType = jobs.StageType
VerdictType = Literal["refuted", "supported", "small gain", "not supported", "inconclusive"]
TargetVerdictType = Literal[
    "supported",
    "partly supported",
    "small gain",
    "inconclusive",
    "not supported",
    "refuted",
]

N_PLANNED_READINGS: Final[int] = 9
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_PLANNED_READINGS)
"""99.44%: each of the nine planned readings (C1, C2, and C3 at two groups, S1, S2, S3)."""
N_CHECKS: Final[int] = 8
CHECK_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_CHECKS)
"""99.375%: the eight checks (`D1p` and `D1t` minus `DF` and gap-scored `D1` minus `D1p`, each at
the six and at the eight, and `P1p` and `P1t` minus `PF`). The positive control is read at the same
level."""
SLICE_LEVEL: Final[float] = 95.0
N_RESAMPLES: Final[int] = 50_000
N_SLICE_RESAMPLES: Final[int] = 2_000
GAIN_THRESHOLD: Final[float] = -0.02
"""Points of CRPS: the smallest gain worth porting."""
OUTER_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
WIDEN_FACTOR: Final[float] = 1.15
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
KEYS: Final[tuple[str, str, str]] = ("time_series_id", "power_fcst_init_time", "valid_time")
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
FIVE: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""The six without D9, whose first reading in the cleaned power table is 2023-11-21."""
EIGHT: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
EIGHT_WITHOUT_D8: Final[tuple[str, ...]] = tuple(p for p in EIGHT if p != "D8")
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
PRIMARY_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "the six": SIX,
    "the eight": EIGHT,
}
REPLICATE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "the five": FIVE,
    "the eight": EIGHT,
}
PRIMARY_HEADLINE_HOURS: Final[float] = 240.0
"""Days 1 to 10 for Primary demand: the lead from the 09:00 UTC origin is at most 10 days."""
SOLAR_HEADLINE_HOURS: Final[float] = 240.0
"""Days 1 to 10 for solar: 12 to under 252 hours from the run is under 240 from the 12:00 origin."""
SOLAR_RUN_OFFSET_HOURS: Final[float] = 12.0
PRIMARY_BANDS: Final[dict[str, tuple[float, float]]] = {
    "day 1": (0.0, 24.0),
    "days 2 to 3": (24.0, 72.0),
    "days 4 to 6": (72.0, 144.0),
    "days 7 to 10": (144.0, 240.0),
}
"""Lead from the origin in hours: a half-hour is in a band if the lead is above the first number
and at most the second."""
SOLAR_BANDS: Final[dict[str, tuple[float, float]]] = {
    "day 1": (12.0, 36.0),
    "days 2 to 3": (36.0, 84.0),
    "days 4 to 6": (84.0, 156.0),
    "days 7 to 10": (156.0, 252.0),
    "days 11 to 14": (252.0, 348.0),
}
"""Lead from the run in hours: a half-hour is in a band from the first number up to the second."""
SOLAR_S3_BAND: Final[tuple[float, float]] = (12.0, 36.0)
REGIMES: Final[tuple[str, ...]] = ("sunny", "between", "overcast", "low sun", "night")
SUN_MIN_ELEVATION_DEG: Final[float] = 10.0
CLEAR_SKY_MIN_W_M2: Final[float] = 20.0
SKY_INDEX_PATH: Final[Path] = common.WORK_DIR / "work" / "sky_index.parquet"
"""Realised clear-sky index and sun elevation, by series and valid time; written by `sky-index`."""
RANK: Final[tuple[TargetVerdictType, ...]] = (
    "supported",
    "partly supported",
    "small gain",
    "inconclusive",
    "not supported",
    "refuted",
)
"""The better verdict comes first."""
SEED_HALVES: Final[dict[str, tuple[tuple[int, ...], tuple[int, ...]]]] = {
    "interim": ((0,), (1,)),
    "primary": ((0, 1, 2), (3, 4, 5)),
    "solar": ((0, 1), (2, 3)),
}
"""The two seed sets of the minimum-detectable-effect null contrast."""


# --- the decision rule ------------------------------------------------------------------------


def verdict(*, lower: float, upper: float) -> VerdictType:
    """Return the decision rule's verdict from an interval of real minus placebo, in points of CRPS.

    The first test that holds decides. Lower above 0: refuted. Upper below -0.02: supported. Upper
    below 0: small gain. Lower at or above -0.02 and upper at or above 0: not supported (a gain of
    0.02 is excluded). Otherwise inconclusive.
    """
    if lower > 0.0:
        return "refuted"
    if upper < GAIN_THRESHOLD:
        return "supported"
    if upper < 0.0:
        return "small gain"
    if lower >= GAIN_THRESHOLD:
        return "not supported"
    return "inconclusive"


def better(*, first: TargetVerdictType, second: TargetVerdictType) -> TargetVerdictType:
    """Return the better of two verdicts, in the order of `RANK`."""
    return first if RANK.index(first) <= RANK.index(second) else second


@dataclass(frozen=True)
class GroupReading:
    """The inputs of a Primary demand group's verdict.

    Attributes:
        c1: The verdict of C1 at the group.
        c2: The verdict of C2 at the group, or `None` in Stage A, which does not fit C2's arms.
        c3: The verdict of C3 at the group, or `None` if the replicate has not been scored.
        c3_lower: The lower limit of C3's interval, for audit trigger 2.
        control_passed: Whether the positive control passed (it makes both groups `inconclusive`
            when it fails), or `None` if it has not been run (Stage A).
    """

    c1: VerdictType
    c2: VerdictType | None
    c3: VerdictType | None
    c3_lower: float | None
    control_passed: bool | None


def group_verdict(*, reading: GroupReading) -> tuple[TargetVerdictType, list[str]]:
    """Return a Primary demand group's verdict and the audit triggers that the readings fire.

    The group reads the better of C1 and C2. A failed positive control makes it `inconclusive`. If
    C1 or C2 is `supported` and C3's lower limit is above 0, audit trigger 2 fires and the group
    reads `inconclusive`. If C1 and C2 are `inconclusive` or `not supported` and C3 is `supported`,
    the group reads `partly supported`.

    Args:
        reading: The verdicts of C1, C2, and C3, and the control.

    Returns:
        The group's verdict and the fired triggers (text).
    """
    fired: list[str] = []
    seen = [v for v in (reading.c1, reading.c2) if v is not None]
    base = better(first=seen[0], second=seen[-1])
    if reading.control_passed is False:
        fired.append("positive control failed: both groups inconclusive")
        return "inconclusive", fired
    if (
        "supported" in (reading.c1, reading.c2)
        and reading.c3_lower is not None
        and reading.c3_lower > 0.0
    ):
        fired.append("audit trigger 2: C1 or C2 supported and the replicate's lower limit above 0")
        return "inconclusive", fired
    if all(v in ("inconclusive", "not supported") for v in seen) and reading.c3 == "supported":
        return "partly supported", fired
    return base, fired


def front_matter_verdict(
    *, six: TargetVerdictType, eight: TargetVerdictType, solar: TargetVerdictType
) -> TargetVerdictType:
    """Return the front-matter verdict from the three targets' verdicts.

    `supported` if all three are `supported`. `partly supported` if any is `supported`, `partly
    supported`, or `small gain`. `refuted` if any is `refuted` and none is `supported`. `not
    supported` if all three are `not supported`. Otherwise `inconclusive`.
    """
    parts = (six, eight, solar)
    if all(p == "supported" for p in parts):
        return "supported"
    if any(p in ("supported", "partly supported", "small gain") for p in parts):
        return "partly supported"
    if any(p == "refuted" for p in parts):
        return "refuted"
    if all(p == "not supported" for p in parts):
        return "not supported"
    return "inconclusive"


# --- losses -----------------------------------------------------------------------------------


def widen(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame with q01, q05, q95, and q99 widened by 1.15 about the median."""
    return frame.with_columns(
        *[
            (pl.col("q50") + WIDEN_FACTOR * (pl.col(c) - pl.col("q50"))).alias(c)
            for c in OUTER_LEVELS
        ]
    )


def row_losses(*, frame: pl.DataFrame, actual: np.ndarray) -> pl.DataFrame:
    """Return per-row CRPS, MAE of the median, and the outer-level coverage, in points.

    Args:
        frame: Predictions with the 11 quantile columns as fractions of effective capacity.
        actual: The observed power as a fraction of effective capacity, one per row.

    Returns:
        The frame's key columns, `seed` if present, and the losses. `crps` and `mae_median` are in
        points of effective capacity (percent).
    """
    quantiles = np.sort(frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    crps = metrics.quantile_score(actual=actual, quantiles=quantiles) * 100.0
    median = quantiles[:, 5]
    out = frame.select([c for c in (*KEYS, "seed") if c in frame.columns])
    columns = {
        "crps": crps,
        "mae_median": np.abs(median - actual) * 100.0,
        **{
            f"below_{c}": (actual < quantiles[:, LEVEL_COLUMNS.index(c)]).astype(float)
            for c in OUTER_LEVELS
        },
    }
    return out.with_columns(*[pl.Series(k, v) for k, v in columns.items()])


def seed_average(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the forecast averaged over the seeds in quantile space (experiment 707's convention).

    The averaged quantiles are sorted, and `seed` is set to -1.
    """
    mean = frame.group_by(list(KEYS)).agg(*[pl.col(c).mean() for c in LEVEL_COLUMNS])
    values = np.sort(mean.select(LEVEL_COLUMNS).to_numpy(), axis=1)
    return mean.with_columns(
        *[pl.Series(c, values[:, i]) for i, c in enumerate(LEVEL_COLUMNS)],
        seed=pl.lit(-1).cast(pl.Int64),
    )


def regime_expression(*, csi: pl.Expr, elevation: pl.Expr) -> pl.Expr:
    """Return the realised sky regime of a half-hour from its clear-sky index and sun elevation.

    Sunny is an index of 0.8 or more and overcast is 0.4 or less on half-hours with the sun at least
    10 degrees up. The half-hours between are `between`; a half-hour with the sun up but under 10
    degrees is `low sun`; and a half-hour with the sun down is `night`. The regime reads what
    happened, never an arm's input.
    """
    overcast_up_to, sunny_from = CLEAR_SKY_INDEX_THRESHOLDS
    return (
        pl.when(elevation <= 0.0)
        .then(pl.lit("night"))
        .when(elevation < SUN_MIN_ELEVATION_DEG)
        .then(pl.lit("low sun"))
        .when(csi >= sunny_from)
        .then(pl.lit("sunny"))
        .when(csi <= overcast_up_to)
        .then(pl.lit("overcast"))
        .otherwise(pl.lit("between"))
    )


def sky_index_table(
    *, sites: pl.DataFrame, cams: pl.DataFrame, valid_times: pl.Series
) -> pl.DataFrame:
    """Return the realised clear-sky index and sun elevation of each series at each valid time.

    The index is the CAMS global horizontal irradiance of the hour that contains the half-hour (the
    hour ending at the half-hour's label or the next whole hour) over the Haurwitz clear-sky
    irradiance of that hour, and 0 where the clear-sky irradiance is under 20 W/m2.

    Args:
        sites: `time_series_id`, `site`, `latitude`, and `longitude`.
        cams: `site`, `time` (an hour's end), and `ghi_w_m2`.
        valid_times: The half-hours to cover, as UTC datetimes.

    Returns:
        `time_series_id`, `valid_time`, `csi`, and `elevation_deg` (at the half-hour's midpoint).
    """
    from studies.baselines import hourly_clear_sky
    from studies.solar import zenith

    unique = valid_times.unique().sort()
    hour_end = unique.dt.offset_by("-1s").dt.truncate("1h").dt.offset_by("1h")
    hours = hourly_clear_sky(sites=sites, first=hour_end.min(), last=hour_end.max())  # ty: ignore[invalid-argument-type]
    parts = []
    for row in sites.iter_rows(named=True):
        elevation = 90.0 - zenith(
            stamps=unique.dt.offset_by("-15m"), latitude=row["latitude"], longitude=row["longitude"]
        )
        mapping = pl.DataFrame({"valid_time": unique, "time": hour_end})
        site_hours = hours.filter(pl.col("site") == row["site"]).join(
            cams.filter(pl.col("site") == row["site"]).select("time", "ghi_w_m2"),
            on="time",
            how="left",
        )
        part = (
            mapping.join(
                site_hours.select("time", "clear_sky_w_m2", "ghi_w_m2"), on="time", how="left"
            )
            .with_columns(
                csi=pl.when(pl.col("clear_sky_w_m2") >= CLEAR_SKY_MIN_W_M2)
                .then(pl.col("ghi_w_m2") / pl.col("clear_sky_w_m2"))
                .otherwise(0.0),
                elevation_deg=pl.Series(elevation),
                time_series_id=pl.lit(row["time_series_id"]).cast(pl.Int64),
            )
            .select("time_series_id", "valid_time", "csi", "elevation_deg")
        )
        parts.append(part)
    return pl.concat(parts)


def add_regime(*, losses: pl.DataFrame, sky: pl.DataFrame) -> pl.DataFrame:
    """Return the loss frame with a `regime` column (`unknown` where the sky value is missing)."""
    joined = losses.join(sky, on=["time_series_id", "valid_time"], how="left")
    return joined.with_columns(
        regime=pl.when(pl.col("csi").is_null() | pl.col("elevation_deg").is_null())
        .then(pl.lit("unknown"))
        .otherwise(regime_expression(csi=pl.col("csi"), elevation=pl.col("elevation_deg")))
    ).drop("csi", "elevation_deg")


# --- reading one contrast ---------------------------------------------------------------------


@dataclass(frozen=True)
class Incomplete:
    """A reading that cannot be made.

    The reason is an absent arm, a missing fold or seed, or rows that differ between the two arms.

    Attributes:
        key: The reading's key, such as `C1 the six`.
        reason: Why the reading is incomplete.
    """

    key: str
    reason: str

    def line(self) -> str:
        """Return the reading as one printable line."""
        return f"{self.key}: INCOMPLETE, {self.reason}"


@dataclass(frozen=True)
class Reading:
    """One contrast on one set of rows.

    Attributes:
        key: The reading's key, such as `C1 the six`.
        label: The contrast's name, such as `C1 D1 minus D1p`.
        scope: The rows, such as `the six, days 1 to 10`.
        difference: The mean per-row difference, treatment minus reference, in points of CRPS.
        lower: The lower limit of the decisive interval (the wider of the two below).
        upper: The upper limit of the decisive interval.
        boot: The month-block bootstrap interval, or `None` if it was not computed.
        t_interval: The t interval on the monthly differences, or `None` with fewer than 2 months.
        averaged: The difference of the seed-averaged forecasts (descriptive), or `None`.
        n_half_hours: The number of rows of one seed.
        n_months: The number of calendar months.
        seeds: The seeds that both arms hold.
        level: The interval's coverage in percent.
        label_rows: The number of rows of one seed per Primary or farm label.
    """

    key: str
    label: str
    scope: str
    difference: float
    lower: float
    upper: float
    boot: tuple[float, float] | None
    t_interval: tuple[float, float] | None
    averaged: float | None
    n_half_hours: int
    n_months: int
    seeds: tuple[int, ...]
    level: float
    label_rows: dict[str, int]

    @property
    def verdict(self) -> VerdictType:
        """Return the decision rule's verdict on the decisive interval."""
        return verdict(lower=self.lower, upper=self.upper)

    def line(self) -> str:
        """Return the reading as one printable line."""
        boot = "" if self.boot is None else f" boot {self.boot[0]:+.3f} to {self.boot[1]:+.3f}"
        t_text = (
            ""
            if self.t_interval is None
            else f" t {self.t_interval[0]:+.3f} to {self.t_interval[1]:+.3f}"
        )
        averaged = "" if self.averaged is None else f" seed-averaged {self.averaged:+.3f}"
        return (
            f"{self.label:34s} {self.scope:34s} {self.difference:+.3f} "
            f"[{self.lower:+.3f}, {self.upper:+.3f}] level {self.level:.2f}% "
            f"seeds {list(self.seeds)} n={self.n_half_hours:,} months {self.n_months}"
            f"{boot}{t_text}{averaged} -> {self.verdict}"
        )

    def exploratory_line(self) -> str:
        """Return the reading as one line with no verdict word.

        An exploratory reading is descriptive. With one month, the interval holds only the spread of
        the per-seed contrasts, so the line says `range of N seeds` instead of an interval.
        """
        if self.n_months < 2:
            spread = f"range of {len(self.seeds)} seeds [{self.lower:+.3f}, {self.upper:+.3f}]"
        else:
            spread = f"[{self.lower:+.3f}, {self.upper:+.3f}] level {self.level:.2f}%"
        averaged = "" if self.averaged is None else f" seed-averaged {self.averaged:+.3f}"
        return (
            f"{self.label:34s} {self.scope:34s} {self.difference:+.3f} {spread} "
            f"seeds {list(self.seeds)} n={self.n_half_hours:,} months {self.n_months}"
            f"{averaged} -> exploratory"
        )

    def rows_line(self) -> str:
        """Return the rows of one seed per label, which are the weights of the pooled mean."""
        rows = ", ".join(f"{k} {v:,}" for k, v in sorted(self.label_rows.items()))
        return f"    rows per label (one seed): {rows}"


Outcome = Reading | Incomplete


@dataclass(frozen=True)
class Paired:
    """Per-seed, per-month totals of a paired difference.

    Attributes:
        sums: The sum of the per-row difference, shape (seeds, months).
        counts: The number of rows in each month, shape (months,), the same for every seed.
        months: The month labels, sorted.
        seeds: The seeds, sorted.
    """

    sums: np.ndarray
    counts: np.ndarray
    months: tuple[str, ...]
    seeds: tuple[int, ...]

    @property
    def mean(self) -> float:
        """Return the mean per-row difference over all seeds and months."""
        return float(self.sums.sum() / (self.sums.shape[0] * self.counts.sum()))


def paired_month_sums(*, a: pl.DataFrame, b: pl.DataFrame, metric: str = "crps") -> Paired:
    """Return the month totals of the difference `a` minus `b`, paired row by row.

    The two frames must hold the same seeds and the same (series, run, valid time) keys. Totalling
    each month once lets the bootstrap resample a few numbers instead of millions of rows.

    Args:
        a: The treatment's per-seed losses (columns `KEYS`, `seed`, `fold`, and the metric).
        b: The reference's per-seed losses.
        metric: The loss column.

    Returns:
        The totals.

    Raises:
        ValueError: If the seeds differ, the keys differ, or the months hold different counts in
            two seeds.
    """
    order = [*KEYS, "seed"]
    a_sorted, b_sorted = a.sort(order), b.sort(order)
    if a_sorted.height != b_sorted.height or not a_sorted.select(order).equals(
        b_sorted.select(order)
    ):
        msg = f"the two arms do not hold the same rows ({a.height:,} and {b.height:,})"
        raise ValueError(msg)
    if a_sorted.height == 0:
        msg = "the selection holds no rows"
        raise ValueError(msg)
    frame = pl.DataFrame(
        {
            "seed": a_sorted["seed"],
            "fold": a_sorted["fold"].cast(pl.String),
            "diff": a_sorted[metric].cast(pl.Float64) - b_sorted[metric].cast(pl.Float64),
        }
    )
    grouped = (
        frame.group_by("seed", "fold")
        .agg(total=pl.col("diff").sum(), n=pl.len())
        .sort("seed", "fold")
    )
    seeds = tuple(sorted(grouped["seed"].unique().to_list()))
    months = tuple(sorted(grouped["fold"].unique().to_list()))
    sums = np.zeros((len(seeds), len(months)))
    counts = np.zeros((len(seeds), len(months)), dtype=np.int64)
    for seed, month, total, n in grouped.iter_rows():
        sums[seeds.index(seed), months.index(month)] = total
        counts[seeds.index(seed), months.index(month)] = n
    if not (counts == counts[0]).all():
        msg = "two seeds hold different numbers of rows in a month"
        raise ValueError(msg)
    return Paired(sums=sums, counts=counts[0], months=months, seeds=seeds)


def month_total_interval(*, paired: Paired, level: float, n_resamples: int) -> tuple[float, float]:
    """Return the month-block bootstrap interval of the mean from month totals.

    This draws a fitting seed and then whole months in the order of
    `studies.bootstrap.bootstrap_difference_at_level`, from the same random seed, so that its
    interval equals that function's for the same rows (the tests check the equality). It takes
    month totals, so a contrast of millions of rows costs the number of months per resample.

    Args:
        paired: The month totals.
        level: The coverage in percent, above 1 and below 100.
        n_resamples: The number of resamples.

    Returns:
        The lower and upper limits.

    Raises:
        ValueError: If `level` is not above 1 and below 100.
    """
    if not 1.0 < level < 100.0:
        msg = f"level is a coverage in percent, above 1 and below 100: {level}"
        raise ValueError(msg)
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    n_seeds, n_months = paired.sums.shape
    resampled = np.empty(n_resamples)
    for resample in range(n_resamples):
        seed_index = generator.integers(0, n_seeds)
        drawn = generator.integers(0, n_months, size=n_months)
        resampled[resample] = paired.sums[seed_index, drawn].sum() / paired.counts[drawn].sum()
    tail = (100.0 - level) / 2.0
    return float(np.percentile(resampled, tail)), float(np.percentile(resampled, 100.0 - tail))


def t_interval_of_months(*, paired: Paired, level: float) -> tuple[float, float] | None:
    """Return the t interval on the monthly mean differences, or `None` with fewer than 2 months.

    Args:
        paired: The month totals.
        level: The coverage in percent.
    """
    if len(paired.months) < 2:
        return None
    monthly = paired.sums.sum(axis=0) / (paired.sums.shape[0] * paired.counts)
    half = float(
        stats.t.ppf(1.0 - (1.0 - level / 100.0) / 2.0, df=len(monthly) - 1)
        * monthly.std(ddof=1)
        / np.sqrt(len(monthly))
    )
    return float(monthly.mean()) - half, float(monthly.mean()) + half


def wider(
    *, boot: tuple[float, float] | None, t_int: tuple[float, float] | None
) -> tuple[float, float]:
    """Return the wider of the bootstrap interval and the t interval (the decisive interval).

    The wider interval is the one with the larger upper minus lower. It is not their union.

    Raises:
        ValueError: If neither interval exists.
    """
    candidates = [i for i in (boot, t_int) if i is not None]
    if not candidates:
        msg = "no interval can be computed"
        raise ValueError(msg)
    return max(candidates, key=lambda i: i[1] - i[0])


# --- the arms' losses -------------------------------------------------------------------------


@dataclass
class ArmLosses:
    """One arm's per-row losses.

    Attributes:
        losses: Per-seed losses with `KEYS`, `seed`, `crps`, `mae_median`, `label`, `fold`,
            `lead_hours`, and `regime`.
        averaged: The same for the seed-averaged forecast (seed -1).
        coverage: For each outer level, the share of actuals below that quantile.
        cells: The (fold, seed) pairs the arm holds.
    """

    losses: pl.DataFrame
    averaged: pl.DataFrame
    coverage: dict[str, float]
    cells: frozenset[tuple[str, int]]

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return the seeds that the arm holds in at least one fold."""
        return tuple(sorted({s for _, s in self.cells}))


class LossSource:
    """Hands out arms' losses one at a time and forgets the oldest arms, to bound memory.

    Args:
        load: Returns an arm's losses, or `None` if the arm has no predictions.
        keep: How many arms to hold at once.
    """

    def __init__(self, *, load: Callable[[str], ArmLosses | None], keep: int = 3) -> None:
        """Hold the loader and an empty cache."""
        self._load = load
        self._keep = keep
        self._cache: dict[str, ArmLosses | None] = {}
        self.errors: dict[str, str] = {}

    def get(self, name: str) -> ArmLosses | None:
        """Return the arm's losses, or `None` if it has none or could not be read.

        A stored arm whose file is absent gives `None`. Any other `ValueError` from the reader is
        kept in `errors` and also gives `None`, so one unreadable arm cannot stop the scorer.
        """
        if name in self._cache:
            self._cache[name] = self._cache.pop(name)
            return self._cache[name]
        try:
            loaded = self._load(name)
        except FileNotFoundError:
            loaded = None
        except ValueError as error:
            self.errors[name] = str(error)
            loaded = None
        self._cache[name] = loaded
        while len(self._cache) > self._keep:
            self._cache.pop(next(iter(self._cache)))
        return loaded


def planned_seeds(*, arm: str, stage: StageType) -> tuple[int, ...]:
    """Return the seeds that the arm needs when `stage` is complete."""
    seeds = {s for _, s in jobs.expected_cells(arm=arm, stage=stage)}
    return tuple(sorted(seeds))


def missing_for(
    *, source: LossSource, arm: str, months: tuple[str, ...], stage: StageType
) -> list[str]:
    """Return why an arm cannot serve a reading on `months`, or an empty list if it can.

    A reading needs the arm's every planned seed on each of `months`. A stored arm is checked
    against its stored seeds.
    """
    held = source.get(arm)
    if held is None:
        return [f"{arm} has no predictions{_error_text(source=source, arm=arm)}"]
    needed = {(m, s) for m in months for s in planned_seeds(arm=arm, stage=stage)}
    absent = sorted(needed - held.cells)
    if absent:
        return [f"{arm} lacks {len(absent)} of {len(needed)} (fold, seed) cells, e.g. {absent[:2]}"]
    return []


def _error_text(*, source: LossSource, arm: str) -> str:
    """Return the reader's error for an arm as text to append to a message, or an empty string."""
    return f" ({source.errors[arm]})" if arm in source.errors else ""


def read_contrast(
    *,
    source: LossSource,
    treatment: str,
    reference: str,
    label: str,
    scope: str,
    where: pl.Expr | None,
    level: float,
    n_resamples: int,
    key: str = "",
    metric: str = "crps",
    seeds: tuple[int, ...] | None = None,
    t_only: bool = False,
) -> Reading:
    """Return one contrast: its difference, the bootstrap and t intervals, and the decisive one.

    The contrast reads the seeds that both arms hold (or `seeds`), totals the difference by seed
    and month, and resamples the totals.

    Args:
        source: The arms' losses.
        treatment: The treatment arm.
        reference: The reference arm.
        label: The contrast's name.
        scope: The rows' description.
        where: A filter on the loss frames' columns.
        level: The interval's coverage in percent.
        n_resamples: The number of bootstrap resamples.
        key: The reading's key; the label if not given.
        metric: The loss column.
        seeds: The seeds, or `None` for the seeds that both arms hold.
        t_only: Read only the t interval (for a contrast on a few folds, where the bootstrap has
            few distinct resamples).

    Returns:
        The reading.

    Raises:
        ValueError: If an arm is absent, the arms share no seed, or the arms' rows differ.
    """
    a, b = source.get(treatment), source.get(reference)
    for name, value in ((treatment, a), (reference, b)):
        if value is None:
            msg = f"{name} has no predictions{_error_text(source=source, arm=name)}"
            raise ValueError(msg)
    if a is None or b is None:
        raise ValueError
    shared = tuple(sorted(set(a.seeds) & set(b.seeds))) if seeds is None else seeds
    if not shared:
        msg = f"{treatment} and {reference} share no seed"
        raise ValueError(msg)
    frames = []
    for value in (a, b):
        frame = value.losses.filter(pl.col("seed").is_in(list(shared)))
        frames.append(frame if where is None else frame.filter(where))
    paired = paired_month_sums(a=frames[0], b=frames[1], metric=metric)
    boot = (
        None
        if t_only
        else month_total_interval(paired=paired, level=level, n_resamples=n_resamples)
    )
    t_int = t_interval_of_months(paired=paired, level=level)
    lower, upper = wider(boot=boot, t_int=t_int)
    n_seeds = len(paired.seeds)
    label_rows = {
        str(k): int(n // n_seeds) for k, n in frames[0].group_by("label").len().iter_rows()
    }
    return Reading(
        key=key or label,
        label=label,
        scope=scope,
        difference=paired.mean,
        lower=lower,
        upper=upper,
        boot=boot,
        t_interval=t_int,
        averaged=_averaged_difference(a=a, b=b, where=where, metric=metric),
        n_half_hours=int(paired.counts.sum()),
        n_months=len(paired.months),
        seeds=paired.seeds,
        level=level,
        label_rows=label_rows,
    )


def _averaged_difference(
    *, a: ArmLosses, b: ArmLosses, where: pl.Expr | None, metric: str
) -> float | None:
    """Return the difference of two arms' seed-averaged losses on their shared rows, or `None`."""
    frames = [v.averaged if where is None else v.averaged.filter(where) for v in (a, b)]
    keys = list(KEYS)
    joined = (
        frames[0]
        .select(*keys, a=pl.col(metric))
        .join(frames[1].select(*keys, b=pl.col(metric)), on=keys, how="inner")
    )
    if joined.is_empty():
        return None
    return float((joined["a"].cast(pl.Float64) - joined["b"].cast(pl.Float64)).mean())  # ty: ignore[invalid-argument-type]


def minimum_detectable_effect(
    *,
    source: LossSource,
    arm: str,
    target: Literal["primary", "solar", "interim"],
    where: pl.Expr | None,
    level: float,
    n_resamples: int,
) -> float | None:
    """Return the half-width at `level` of a null contrast between two halves of an arm's seeds.

    The first half's seeds are `SEED_HALVES[target][0]`, and the second half's seeds are relabelled
    to the first half's before pairing. Because the bootstrap draws one fitting seed per resample, a
    half-split has the noise structure of the real contrast. Returns `None` if the arm does not
    hold all of both halves' seeds.
    """
    first, second = SEED_HALVES[target]
    held = source.get(arm)
    if held is None or not (set(first) <= set(held.seeds) and set(second) <= set(held.seeds)):
        return None
    shift = dict(zip(second, first, strict=True))
    base = held.losses if where is None else held.losses.filter(where)
    one = base.filter(pl.col("seed").is_in(list(first)))
    two = base.filter(pl.col("seed").is_in(list(second))).with_columns(
        seed=pl.col("seed").replace_strict(shift, return_dtype=base.schema["seed"])
    )
    paired = paired_month_sums(a=one, b=two)
    lower, upper = month_total_interval(paired=paired, level=level, n_resamples=n_resamples)
    return (upper - lower) / 2.0


# --- reading the store ------------------------------------------------------------------------


@cache
def _head() -> str:
    """Return the checked-out commit, read once."""
    return gate_marker.head_commit()


REJECTED_SETS: Final[dict[str, str]] = {}
"""The prediction sets that the manifest check rejected, by set name, with the reason."""


def available(*, arm: str, month: str, seed: int) -> bool:
    """Return whether the store holds a usable set for the arm, fold, and seed.

    A set whose manifest is `dirty`, whose commit is not an ancestor of `HEAD`, or whose commit is
    older than the floor commit is not used. It is recorded in `REJECTED_SETS`, and the arm's cell
    then counts as absent.
    """
    name = arms.set_name(arm=arm, month=month, seed=seed)
    folder = common.WORK_DIR / "predictions" / name
    if not folder.exists():
        return False
    try:
        gate_marker.check_existing_set(folder=folder, head=_head())
    except (ValueError, FileNotFoundError) as error:
        REJECTED_SETS[name] = str(error)
        return False
    return True


def stored_d0_path(*, month: str, seed: int) -> Path:
    """Return the stored forecast of `D0`: experiment 707's `R_F` raw quantiles."""
    import e00707_arms as arms707  # ty: ignore[unresolved-import]

    return arms707.WORK_DIR / "work" / "seed_preds" / "R_F" / f"{month}_s{seed}.parquet"


def stored_p0_files(*, seed: int) -> list[Path]:
    """Return experiment 666's solar PLP files of one seed (one file per quarterly refit)."""
    from e00666_common import FIT_DIR  # ty: ignore[unresolved-import]

    return sorted((FIT_DIR / "solar").glob(f"refit=*_seed={seed}.parquet"))


def cell_exists(*, arm: arms.Arm, month: str, seed: int) -> bool:
    """Return whether the arm has predictions for the fold and seed."""
    if not arm.stored:
        return available(arm=arm.name, month=month, seed=seed)
    if arm.name == "D0":
        return stored_d0_path(month=month, seed=seed).exists()
    return bool(stored_p0_files(seed=seed))


def month_bounds(*, month: str) -> tuple[datetime, datetime]:
    """Return the first instant of a month `YYYY-MM` and of the next month, in UTC."""
    year, number = (int(p) for p in month.split("-"))
    start = datetime(year, number, 1, tzinfo=UTC)
    end = datetime(year + number // 12, number % 12 + 1, 1, tzinfo=UTC)
    return start, end


def read_stored_solar(*, month: str, seed: int) -> pl.DataFrame:
    """Return experiment 666's PLP forecast of the runs of one month, as 711's solar predictions.

    The files hold megawatts of the refit's capacity. This divides by each farm's stored effective
    capacity, which is the unit of `actual`, and sets the origin to 12:00 UTC on the run's day. It
    drops the half-hour labelled at the origin (a 711 arm does not forecast it), the valid times
    from 2026-07-01 on, and every run outside the month.

    Raises:
        ValueError: If no file covers the month.
    """
    from e00666_common import QUANTILES  # ty: ignore[unresolved-import]
    from studies.pv_dataset import pv_sites

    start, end = month_bounds(month=month)
    capacity = pv_sites().select(pl.col("time_series_id").cast(pl.Int64), "effective_capacity_mw")
    pieces = [
        pl.read_parquet(path).filter((pl.col("init_time") >= start) & (pl.col("init_time") < end))
        for path in stored_p0_files(seed=seed)
    ]
    frame = pl.concat([p for p in pieces if not p.is_empty()])
    if frame.is_empty():
        msg = f"no stored PLP row covers {month}"
        raise ValueError(msg)
    origin = pl.col("init_time") + pl.duration(hours=12)
    return (
        frame.with_columns(pl.col("time_series_id").cast(pl.Int64))
        .join(capacity, on="time_series_id", how="left")
        .select(
            "time_series_id",
            power_fcst_init_time=origin,
            valid_time=pl.col("time"),
            **{
                level: pl.col(quantile) / pl.col("effective_capacity_mw")
                for level, quantile in zip(LEVEL_COLUMNS, QUANTILES, strict=True)
            },
        )
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (pl.col("valid_time") < experiment_store.HOLDOUT_START)
        )
        .with_columns(seed=pl.lit(seed).cast(pl.Int64))
    )


def read_arm(*, arm: arms.Arm, months: tuple[str, ...], seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return an arm's predictions over folds and seeds, with the `seed` and `fold` columns.

    The stored `D0` reads experiment 707's stored `R_F` raw quantiles and keeps the valid times
    before 2026-07-01. The stored `P0` reads experiment 666's solar files.
    """
    pieces = []
    for month in months:
        for seed in seeds:
            if arm.stored and arm.name == "D0":
                frame = pl.read_parquet(stored_d0_path(month=month, seed=seed)).with_columns(
                    seed=pl.lit(seed).cast(pl.Int64)
                )
                frame = frame.filter(pl.col("valid_time") < experiment_store.HOLDOUT_START)
            elif arm.stored:
                frame = read_stored_solar(month=month, seed=seed)
            else:
                frame = experiment_store.read_predictions(
                    experiment_id=arms.EXPERIMENT_ID,
                    name=arms.set_name(arm=arm.name, month=month, seed=seed),
                    row_set_id=arm.row_set_id,
                )
            pieces.append(frame.with_columns(fold=pl.lit(month)))
    return pl.concat(pieces)


def primary_actuals(*, frame: pl.DataFrame) -> np.ndarray:
    """Return each Primary row's observed power as a fraction of effective capacity.

    The actual is read from experiment 707's dense half-hourly arrays.
    """
    import e00707_common as base  # ty: ignore[unresolved-import]

    series, arrays = base.series_and_arrays()
    site_of = dict(zip(series["time_series_id"].to_list(), series["site"].to_list(), strict=True))
    actual = np.full(frame.height, np.nan)
    ids = frame["time_series_id"].to_numpy()
    for tsid in np.unique(ids):
        rows = np.flatnonzero(ids == tsid)
        index = np.asarray(base.grid_index(times=frame["valid_time"].gather(rows)))
        actual[rows] = base.take(values=arrays[site_of[int(tsid)]].power, index=index)
    return actual


def solar_actuals(*, frame: pl.DataFrame) -> np.ndarray:
    """Return each solar row's observed power as a fraction of effective capacity.

    The value is the `actual` column of the farm's table in `solar_inputs/`, which `e00711_solar`
    reads (null where unobserved, which becomes NaN).
    """
    import e00711_solar as solar

    farms = pl.concat(
        [
            solar.load_farm(site=site)
            .select("time_series_id", "time", "actual")
            .unique(subset=["time_series_id", "time"])
            for site in farm_sites()
        ]
    ).with_columns(pl.col("time_series_id").cast(pl.Int64))
    joined = frame.select("time_series_id", "valid_time").join(
        farms,
        left_on=["time_series_id", "valid_time"],
        right_on=["time_series_id", "time"],
        how="left",
    )
    return joined["actual"].cast(pl.Float64).fill_null(float("nan")).to_numpy()


def farm_sites() -> list[str]:
    """Return the farms' site keys, which are the letters A to F in the farm table."""
    from studies.pv_dataset import pv_sites

    return pv_sites()["site"].to_list()


def drop_flagged_and_missing(
    *, frame: pl.DataFrame, actual: np.ndarray, target: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the rows that have an observed actual, without the flagged exact-zero half-hours.

    Args:
        frame: The predictions.
        actual: The actual of each row (NaN where unobserved).
        target: `primary` or `solar`; only Primary demand has a list of flagged half-hours.
    """
    keep = np.isfinite(actual)
    out = frame.filter(pl.Series(keep))
    kept_actual = actual[keep]
    if target == "primary":
        import e00707_arms as arms707  # ty: ignore[unresolved-import]

        flagged = pl.read_parquet(arms707.WORK_DIR / "work" / "flagged_keys.parquet").select(
            "time_series_id", "valid_time"
        )
        marked = out.join(
            flagged.with_columns(_flagged=pl.lit(True)),
            on=["time_series_id", "valid_time"],
            how="left",
        )
        ok = marked["_flagged"].is_null().to_numpy()
        out, kept_actual = out.filter(pl.Series(ok)), kept_actual[ok]
    return out, kept_actual


def compact(
    *, frame: pl.DataFrame, labels: dict[int, str], sky: pl.DataFrame | None
) -> pl.DataFrame:
    """Return a loss frame with small dtypes and the `label`, `lead_hours`, and `regime` columns."""
    out = frame.with_columns(
        label=pl.col("time_series_id")
        .replace_strict(labels, default="?", return_dtype=pl.String)
        .cast(pl.Categorical),
        lead_hours=(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
        ).cast(pl.Float32),
        fold=pl.col("power_fcst_init_time").dt.strftime("%Y-%m").cast(pl.Categorical),
        seed=pl.col("seed").cast(pl.Int16),
        crps=pl.col("crps").cast(pl.Float32),
        mae_median=pl.col("mae_median").cast(pl.Float32),
    )
    if sky is not None:
        out = add_regime(losses=out, sky=sky)
    else:
        out = out.with_columns(regime=pl.lit("unknown"))
    return out.with_columns(pl.col("regime").cast(pl.Categorical)).select(
        *KEYS, "seed", "crps", "mae_median", "label", "fold", "lead_hours", "regime"
    )


def prepare(*, arm: arms.Arm, labels: dict[int, str], sky: pl.DataFrame | None) -> ArmLosses | None:
    """Return an arm's per-seed and seed-averaged losses on rows with an observed actual.

    The arm is read one fold at a time, so that only one fold's quantiles are in memory. The rows
    without an observed actual, and the flagged exact-zero half-hours of Primary demand, are
    dropped. Primary demand forecasts are widened by 1.15 about the median.

    Returns:
        The losses, or `None` if the arm has no predictions.
    """
    cells = [(m, s) for m in arm.months for s in arm.seeds if cell_exists(arm=arm, month=m, seed=s)]
    if not cells:
        return None
    seed_chunks, average_chunks = [], []
    below = dict.fromkeys(OUTER_LEVELS, 0.0)
    total = 0
    for month in arm.months:
        seeds = tuple(s for m, s in cells if m == month)
        if not seeds:
            continue
        frame = read_arm(arm=arm, months=(month,), seeds=seeds)
        frame = frame.with_columns(pl.col("time_series_id").cast(pl.Int64))
        if arm.target == "primary":
            frame = widen(frame=frame)
        actual = (
            primary_actuals(frame=frame) if arm.target == "primary" else solar_actuals(frame=frame)
        )
        frame, actual = drop_flagged_and_missing(frame=frame, actual=actual, target=arm.target)
        averaged = seed_average(frame=frame)
        lookup = (
            frame.select(*KEYS).with_columns(actual=pl.Series(actual)).unique(subset=list(KEYS))
        )
        avg_actual = averaged.select(*KEYS).join(lookup, on=list(KEYS), how="left")["actual"]
        for kind, source_frame, values in (
            ("seed", frame, actual),
            ("avg", averaged, avg_actual.to_numpy()),
        ):
            loss = row_losses(frame=source_frame, actual=values)
            if kind == "seed":
                total += loss.height
                for c in OUTER_LEVELS:
                    below[c] += float(loss[f"below_{c}"].sum())
            chunk = compact(
                frame=loss.drop(*[f"below_{c}" for c in OUTER_LEVELS]), labels=labels, sky=sky
            )
            (seed_chunks if kind == "seed" else average_chunks).append(chunk)
    if not seed_chunks:
        return None
    return ArmLosses(
        losses=pl.concat(seed_chunks),
        averaged=pl.concat(average_chunks),
        coverage={c: below[c] / total for c in OUTER_LEVELS},
        cells=frozenset(cells),
    )


# --- scope expressions ------------------------------------------------------------------------


def primary_scope(
    *,
    members: tuple[str, ...],
    months: tuple[str, ...],
    band: tuple[float, float] | None = None,
) -> pl.Expr:
    """Return the rows of Primaries `members` on folds `months` at the headline or a lead band."""
    low, high = (0.0, PRIMARY_HEADLINE_HOURS) if band is None else band
    return (
        pl.col("label").cast(pl.String).is_in(list(members))
        & pl.col("fold").cast(pl.String).is_in(list(months))
        & (pl.col("lead_hours") > low)
        & (pl.col("lead_hours") <= high)
    )


def solar_scope(
    *,
    months: tuple[str, ...],
    band: tuple[float, float] | None = None,
    farms: tuple[str, ...] = FARMS,
) -> pl.Expr:
    """Return the solar rows with a lead from the run in `band` (days 1 to 10 by default)."""
    low, high = (12.0, 12.0 + SOLAR_HEADLINE_HOURS) if band is None else band
    from_run = pl.col("lead_hours") + SOLAR_RUN_OFFSET_HOURS
    return (
        pl.col("label").cast(pl.String).is_in(list(farms))
        & pl.col("fold").cast(pl.String).is_in(list(months))
        & (from_run >= low)
        & (from_run < high)
    )


# --- the scoring run --------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """One planned reading or check: the arms, the folds, the rows, and the level.

    Attributes:
        key: The reading's name.
        treatment: The treatment arm.
        reference: The reference arm.
        scope: The rows' description.
        months: The folds that the reading needs.
        where: The row filter.
        level: The coverage in percent.
        t_only: Whether only the t interval is read (four folds cannot support a bootstrap).
    """

    key: str
    treatment: str
    reference: str
    scope: str
    months: tuple[str, ...]
    where: pl.Expr
    level: float
    t_only: bool = False


def run_spec(*, source: LossSource, spec: Spec, stage: StageType, n_resamples: int) -> Outcome:
    """Return a reading, or `Incomplete` if an arm lacks a planned fold or seed, or the rows differ.

    A reading never runs on the seeds that happen to be complete: it needs every planned cell of
    both arms on the reading's folds in `stage`.
    """
    reasons = [
        r
        for name in (spec.treatment, spec.reference)
        for r in missing_for(source=source, arm=name, months=spec.months, stage=stage)
    ]
    if reasons:
        return Incomplete(key=spec.key, reason="; ".join(reasons))
    seeds = tuple(
        sorted(
            set(planned_seeds(arm=spec.treatment, stage=stage))
            & set(planned_seeds(arm=spec.reference, stage=stage))
        )
    )
    try:
        return read_contrast(
            source=source,
            treatment=spec.treatment,
            reference=spec.reference,
            key=spec.key,
            label=f"{spec.key} {spec.treatment} minus {spec.reference}",
            scope=spec.scope,
            where=spec.where,
            level=spec.level,
            n_resamples=n_resamples,
            seeds=seeds,
            t_only=spec.t_only,
        )
    except ValueError as error:
        return Incomplete(key=spec.key, reason=str(error))


def _spec(
    *,
    key: str,
    pair: tuple[str, str],
    scope: str,
    months: tuple[str, ...],
    where: pl.Expr,
    level: float,
    t_only: bool = False,
) -> Spec:
    """Return a spec of the arms `pair` (treatment, reference)."""
    return Spec(
        key=key,
        treatment=pair[0],
        reference=pair[1],
        scope=scope,
        months=months,
        where=where,
        level=level,
        t_only=t_only,
    )


def _primary_specs(
    *,
    prefix: str,
    pair: tuple[str, str],
    months: tuple[str, ...],
    level: float,
    t_only: bool = False,
) -> list[Spec]:
    """Return one spec of a Primary demand pair at the six and at the eight, never pooled."""
    return [
        _spec(
            key=f"{prefix} {group}",
            pair=pair,
            scope=f"{group}, days 1 to 10, {len(months)} folds",
            months=months,
            where=primary_scope(members=members, months=months),
            level=level,
            t_only=t_only,
        )
        for group, members in PRIMARY_GROUPS.items()
    ]


def planned_specs(*, stage: StageType) -> list[Spec]:
    """Return the planned readings of section 8 at 99.44% for `stage`.

    Stage A reads C1 at the six and at the eight, and S1. Stage B (the whole plan) adds C2, C3, S2,
    and S3, and reads all nine.
    """
    if stage == "B1":
        return b1_specs()
    cloud, solar_months = arms.CLOUD_SCORED_MONTHS, arms.SOLAR_MONTHS
    solar_all = solar_scope(months=solar_months)
    specs = _primary_specs(prefix="C1", pair=("D1", "D1p"), months=cloud, level=PLANNED_LEVEL)
    specs.append(
        _spec(
            key="S1",
            pair=("P1", "P1p"),
            scope="six farms, days 1 to 10",
            months=solar_months,
            where=solar_all,
            level=PLANNED_LEVEL,
        )
    )
    if stage == "A":
        return specs
    specs += _primary_specs(prefix="C2", pair=("D2", "D2p"), months=cloud, level=PLANNED_LEVEL)
    specs += [
        _spec(
            key=f"C3 {group}",
            pair=("R1", "R1p"),
            scope=f"{group}, days 1 to 10, replicate",
            months=arms.REPLICATE_MONTHS,
            where=primary_scope(members=members, months=arms.REPLICATE_MONTHS),
            level=PLANNED_LEVEL,
        )
        for group, members in REPLICATE_GROUPS.items()
    ]
    specs += [
        _spec(
            key="S2",
            pair=("P2", "P2p"),
            scope="six farms, days 1 to 10",
            months=solar_months,
            where=solar_all,
            level=PLANNED_LEVEL,
        ),
        _spec(
            key="S3",
            pair=("P1", "P1p"),
            scope="six farms, 12 to 36 hours",
            months=solar_months,
            where=solar_scope(months=solar_months, band=SOLAR_S3_BAND),
            level=PLANNED_LEVEL,
        ),
    ]
    return specs


def b1_specs() -> list[Spec]:
    """Return Stage B1's planned readings at 99.44%: C3 (the five, the eight) and G1 (six, eight).

    C3 is the replicate `R1` minus `R1p`, with no gap and no flag, on the 12 replicate folds. G1 is
    `D1g` minus `D0g`, the flag-free, gap-free pair on the 7 cloud-scored folds.
    """
    cloud = arms.CLOUD_SCORED_MONTHS
    c3 = [
        _spec(
            key=f"C3 {group}",
            pair=("R1", "R1p"),
            scope=f"{group}, days 1 to 10, replicate",
            months=arms.REPLICATE_MONTHS,
            where=primary_scope(members=members, months=arms.REPLICATE_MONTHS),
            level=PLANNED_LEVEL,
        )
        for group, members in REPLICATE_GROUPS.items()
    ]
    return c3 + _primary_specs(prefix="G1", pair=("D1g", "D0g"), months=cloud, level=PLANNED_LEVEL)


def b1_bound_specs() -> list[Spec]:
    """Return `R1p` minus `R0` and `R1` minus `R0` (the five, the eight) at 99.375%.

    With no flag and no gap, `R1p` minus `R0` shows what the placebo's own shift is, and `R1` minus
    `R0` shows the total effect of the cloud columns. Both are descriptive bounds beside C3.
    """
    out: list[Spec] = []
    for pair in (("R1p", "R0"), ("R1", "R0")):
        out += [
            _spec(
                key=f"{pair[0]} minus {pair[1]} {group}",
                pair=pair,
                scope=f"{group}, days 1 to 10, replicate",
                months=arms.REPLICATE_MONTHS,
                where=primary_scope(members=members, months=arms.REPLICATE_MONTHS),
                level=CHECK_LEVEL,
            )
            for group, members in REPLICATE_GROUPS.items()
        ]
    return out


def check_specs(*, stage: StageType) -> list[Spec]:
    """Return the checks at 99.375%: placebo carries nothing, and the gap-scored folds.

    Every Primary demand check is read at the six and at the eight separately. Stage A reads `D1p`
    minus `DF` and `P1p` minus `PF`. Stage B adds the time-shifted placebos and the gap-scored
    check.
    """
    if stage == "B1":
        return []
    cloud, solar_months = arms.CLOUD_SCORED_MONTHS, arms.SOLAR_MONTHS
    primary_pairs = [("D1p", "DF")] + ([("D1t", "DF")] if stage != "A" else [])
    solar_pairs = [("P1p", "PF")] + ([("P1t", "PF")] if stage != "A" else [])
    specs: list[Spec] = []
    for pair in primary_pairs:
        specs += _primary_specs(
            prefix=f"{pair[0]} minus {pair[1]}", pair=pair, months=cloud, level=CHECK_LEVEL
        )
    specs += [
        _spec(
            key=f"{pair[0]} minus {pair[1]}",
            pair=pair,
            scope="six farms, days 1 to 10",
            months=solar_months,
            where=solar_scope(months=solar_months),
            level=CHECK_LEVEL,
        )
        for pair in solar_pairs
    ]
    if stage != "A":
        specs += _primary_specs(
            prefix="gap-scored D1 minus D1p",
            pair=("D1", "D1p"),
            months=arms.GAP_SCORED_MONTHS,
            level=CHECK_LEVEL,
            t_only=True,
        )
    return specs


def bound_specs() -> list[Spec]:
    """Return `D1` minus `DF` (at the six and at the eight) and `P1` minus `PF`.

    They are printed beside C1 and S1 as a bound: they show how much of a C1 or S1 reading the
    placebo's own shift from no cloud explains.
    """
    solar_months = arms.SOLAR_MONTHS
    return [
        *_primary_specs(
            prefix="D1 minus DF",
            pair=("D1", "DF"),
            months=arms.CLOUD_SCORED_MONTHS,
            level=CHECK_LEVEL,
        ),
        _spec(
            key="P1 minus PF",
            pair=("P1", "PF"),
            scope="six farms, days 1 to 10",
            months=solar_months,
            where=solar_scope(months=solar_months),
            level=CHECK_LEVEL,
        ),
    ]


def control_specs() -> list[Spec]:
    """Return the positive control `K1` minus `K1p` at the six and at the eight (Stage B only)."""
    return _primary_specs(
        prefix="K1", pair=("K1", "K1p"), months=arms.CLOUD_SCORED_MONTHS, level=CHECK_LEVEL
    )


def print_arms(*, source: LossSource, stage: StageType) -> None:
    """Print each arm's folds, seeds, and outer-level coverage, and the input columns it logged."""
    print("== arms found in the store ==")
    for arm in arms.ARM_LIST:
        held = source.get(arm.name)
        if held is None:
            note = _error_text(source=source, arm=arm.name)
            print(f"{arm.name:4s} none{note}")
            continue
        needed = jobs.expected_cells(arm=arm.name, stage=stage) if not arm.stored else held.cells
        coverage = " ".join(f"{k} {v:.3f}" for k, v in held.coverage.items())
        print(
            f"{arm.name:4s} {len(held.cells)} of {len(needed)} planned (fold, seed) cells; "
            f"actual below: {coverage}"
        )
    for name, reason in sorted(REJECTED_SETS.items()):
        print(f"set {name} not used: {reason}")
    print_input_columns(names=[a.name for a in arms.fitted_arms()])


def print_input_columns(*, names: list[str]) -> None:
    """Print each fitted arm's input columns from its first fit log."""
    import json

    print("\n== input columns of each arm (from the first fit log) ==")
    for name in names:
        arm = arms.ARMS[name]
        for month in arm.months:
            path = common.WORK_DIR / "work" / "logs" / name / f"{month}_s{arm.seeds[0]}.json"
            if path.exists():
                columns = json.loads(path.read_text())["input_columns"]
                extras = columns[-len(arm.extra_columns) :] if arm.extra_columns else []
                print(f"{name:4s} {len(columns)} columns, last {len(extras)}: {extras}")
                break


def print_interim_mde(*, source: LossSource, n_resamples: int) -> None:
    """Print Stage A's interim minimum detectable effect: seed 0 of the placebo against seed 1.

    The bootstrap draws one fitting seed per resample, so a split of seed 0 against seed 1 of the
    placebo arm has the noise structure of the 3-against-3 split that Stage B can make. Without this
    number, a Stage A `inconclusive` cannot be told from an underpowered contrast.
    """
    print("\n== interim minimum detectable effect (seed 0 against seed 1; half-width at 99.44%) ==")
    cloud = arms.CLOUD_SCORED_MONTHS
    cases = [
        ("D1p", f"{group}, 7 folds", primary_scope(members=members, months=cloud))
        for group, members in PRIMARY_GROUPS.items()
    ] + [("P1p", "six farms, days 1 to 10", solar_scope(months=arms.SOLAR_MONTHS))]
    for arm, scope, where in cases:
        mde = minimum_detectable_effect(
            source=source,
            arm=arm,
            target="interim",
            where=where,
            level=PLANNED_LEVEL,
            n_resamples=n_resamples,
        )
        text = "not computed: the arm lacks seeds 0 and 1" if mde is None else f"{mde:.3f} points"
        print(f"{arm} {scope}: {text}")


def print_mde(*, source: LossSource, n_resamples: int) -> None:
    """Print the minimum detectable effect of each planned reading's placebo (Stage B only)."""
    print("\n== minimum detectable effect (before any contrast; half-width at 99.44%) ==")
    cloud = arms.CLOUD_SCORED_MONTHS
    cases = (
        [
            (n, "primary", f"{group}, 7 folds", primary_scope(members=members, months=cloud))
            for n in ("D1p", "D2p")
            for group, members in PRIMARY_GROUPS.items()
        ]
        + [
            (
                "R1p",
                "primary",
                f"{group}, replicate",
                primary_scope(members=members, months=arms.REPLICATE_MONTHS),
            )
            for group, members in REPLICATE_GROUPS.items()
        ]
        + [
            (n, "solar", "six farms, days 1 to 10", solar_scope(months=arms.SOLAR_MONTHS))
            for n in ("P1p", "P2p")
        ]
    )
    for arm, target, scope, where in cases:
        mde = minimum_detectable_effect(
            source=source,
            arm=arm,
            target=target,  # ty: ignore[invalid-argument-type]
            where=where,
            level=PLANNED_LEVEL,
            n_resamples=n_resamples,
        )
        text = (
            "not computed: the arm lacks both halves' seeds" if mde is None else f"{mde:.3f} points"
        )
        print(f"{arm} {scope}: {text}")


def print_outcomes(*, outcomes: list[Outcome], rows: bool = False) -> None:
    """Print each reading's line, and with `rows` the rows per label beside it."""
    for outcome in outcomes:
        print(outcome.line(), flush=True)
        if rows and isinstance(outcome, Reading):
            print(outcome.rows_line())


def check_results(*, outcomes: list[Outcome]) -> dict[str, bool | None]:
    """Return which checks passed: the interval contains zero. An incomplete check gives `None`."""
    results: dict[str, bool | None] = {}
    for outcome in outcomes:
        if isinstance(outcome, Incomplete):
            results[outcome.key] = None
            continue
        results[outcome.key] = outcome.lower <= 0.0 <= outcome.upper
    return results


def control_results(*, outcomes: list[Outcome]) -> dict[str, bool | None]:
    """Return which positive controls passed: the interval lies wholly below zero."""
    return {o.key: (None if isinstance(o, Incomplete) else o.upper < 0.0) for o in outcomes}


def _solar_verdict(
    *, readings: Mapping[str, Outcome], checks: Mapping[str, bool | None] | None, full: bool
) -> TargetVerdictType | None:
    """Print and return the solar farms' verdict, or `None` if a reading or the check is missing."""
    s1 = readings.get("S1")
    s2 = readings.get("S2") if full else None
    if not isinstance(s1, Reading) or (full and not isinstance(s2, Reading)):
        print("solar farms: verdict withheld, a planned reading is incomplete")
        return None
    state = (checks or {}).get("P1p minus PF", "absent")
    if state in (None, "absent"):
        print("solar farms: verdict withheld, the `P1p` minus `PF` check is incomplete")
        return None
    result: TargetVerdictType = s1.verdict
    if isinstance(s2, Reading):
        result = better(first=s1.verdict, second=s2.verdict)
    if state is False:
        result = "inconclusive"
    print(f"solar farms: S1 {s1.verdict} -> {result}")
    return result


def verdicts(
    *,
    readings: Mapping[str, Outcome],
    controls: dict[str, bool | None],
    stage: StageType,
    checks: Mapping[str, bool | None] | None = None,
) -> dict[str, str]:
    """Print the verdict of each Primary demand group, the solar farms, and the front matter.

    Stage A reads C1 and S1 only: its verdict is decided by those contrasts, and it does not use
    the positive control (Stage B's). Stage B reads the whole plan, and withholds a verdict if a
    planned reading or the control is incomplete.

    **A failed placebo check overrides C1 and S1.** If `D1p` minus `DF` fails at a group (its
    interval does not contain zero), that group's C1 verdict reads `inconclusive`. If the check is
    incomplete, the verdict is withheld. The same holds for S1 and `P1p` minus `PF`.

    Args:
        readings: The planned readings by key.
        controls: The positive control by key (Stage B).
        stage: The stage.
        checks: Which checks passed, by key; `None` or an absent key means the check was not read.

    Returns:
        The printed verdicts by name (for the tests).
    """
    print(f"\n== verdicts (decision rule, section 8; stage {stage}) ==")
    out: dict[str, str] = {}
    full = stage != "A"
    group_results: dict[str, TargetVerdictType] = {}
    for group in PRIMARY_GROUPS:
        c1 = readings.get(f"C1 {group}")
        c2 = readings.get(f"C2 {group}") if full else None
        c3 = readings.get(f"C3 {'the five' if group == 'the six' else group}") if full else None
        needed = [c1, c2] if full else [c1]
        if any(not isinstance(o, Reading) for o in needed) or (
            full and not isinstance(c3, Reading)
        ):
            print(f"{group}: verdict withheld, a planned reading is incomplete")
            continue
        control = controls.get(f"K1 {group}") if full else None
        if full and control is None:
            print(f"{group}: verdict withheld, the positive control has not run")
            continue
        assert isinstance(c1, Reading)
        placebo_state = (checks or {}).get(f"D1p minus DF {group}", "absent")
        if placebo_state in (None, "absent"):
            print(f"{group}: verdict withheld, the `D1p` minus `DF` check is incomplete")
            continue
        result, fired = group_verdict(
            reading=GroupReading(
                c1=c1.verdict,
                c2=c2.verdict if isinstance(c2, Reading) else None,
                c3=c3.verdict if isinstance(c3, Reading) else None,
                c3_lower=c3.lower if isinstance(c3, Reading) else None,
                control_passed=control,
            )
        )
        if placebo_state is False:
            result = "inconclusive"
            fired.append("`D1p` minus `DF` failed: C1 reads inconclusive until the audit reports")
        group_results[group] = result
        out[group] = result
        c2_text = c2.verdict if isinstance(c2, Reading) else "not in this stage"
        c3_text = c3.verdict if isinstance(c3, Reading) else "not in this stage"
        print(f"{group}: C1 {c1.verdict}, C2 {c2_text}, C3 {c3_text} -> {result}; {fired}")
    solar_result = _solar_verdict(readings=readings, checks=checks, full=full)
    if solar_result is not None:
        out["solar"] = solar_result
    if len(group_results) == len(PRIMARY_GROUPS) and solar_result is not None:
        front = front_matter_verdict(
            six=group_results["the six"], eight=group_results["the eight"], solar=solar_result
        )
        out["front matter"] = front
        label = "Stage A verdict" if stage == "A" else "front-matter verdict"
        print(f"{label}: {front}")
    return out


def verdicts_b1(*, readings: dict[str, Outcome]) -> dict[str, str]:
    """Print and return Stage B1's verdicts: C3 and G1, each by the decision rule.

    The verdicts are the decision rule applied to the decisive interval. Stage B1 gives no
    front-matter verdict, because it holds no positive control and no solar arm. Section 12 of the
    page says what each outcome means for the flag and date-proxy question.
    """
    print(
        "\n== Stage B1 verdicts (decision rule applied to each reading; no front-matter verdict) =="
    )
    out: dict[str, str] = {}
    for key in (*(f"C3 {g}" for g in REPLICATE_GROUPS), *(f"G1 {g}" for g in PRIMARY_GROUPS)):
        reading = readings.get(key)
        if not isinstance(reading, Reading):
            print(f"{key}: verdict withheld, the reading is incomplete")
            continue
        out[key] = reading.verdict
        print(
            f"{key}: {reading.verdict} ({reading.difference:+.3f}, "
            f"[{reading.lower:+.3f}, {reading.upper:+.3f}])"
        )
    return out


def score_b1(*, n_resamples: int) -> None:
    """Print Stage B1's readings: C3 and G1 at 99.44%, and the `R0` bounds at 99.375%."""
    source = make_source(with_sky=False)
    print_arms(source=source, stage="B1")
    print(
        f"\n== Stage B1 planned readings, level {PLANNED_LEVEL:.2f}%, {n_resamples:,} resamples =="
    )
    readings = [
        run_spec(source=source, spec=s, stage="B1", n_resamples=n_resamples) for s in b1_specs()
    ]
    print_outcomes(outcomes=readings, rows=True)
    print(f"\n== bounds beside C3: placebo and real arm against no cloud ({CHECK_LEVEL:.3f}%) ==")
    print_outcomes(
        outcomes=[
            run_spec(source=source, spec=b, stage="B1", n_resamples=n_resamples)
            for b in b1_bound_specs()
        ]
    )
    verdicts_b1(readings={r.key: r for r in readings})


def exploratory(*, source: LossSource) -> None:
    """Print the exploratory readings at 95%: lead bands, sky regimes, labels, folds, totals."""
    print("\n== exploratory readings (95% interval; each with its half-hour count) ==")
    cloud = arms.CLOUD_SCORED_MONTHS
    for treatment, reference, target in (("D1", "D1p", "primary"), ("P1", "P1p", "solar")):
        if source.get(treatment) is None or source.get(reference) is None:
            continue
        bands = PRIMARY_BANDS if target == "primary" else SOLAR_BANDS
        groups = PRIMARY_GROUPS if target == "primary" else {"six farms": FARMS}

        def scope(
            *,
            labels: tuple[str, ...],
            months: tuple[str, ...] = cloud,
            band: tuple[float, float] | None = None,
            extra: pl.Expr | None = None,
            target: str = target,
        ) -> pl.Expr:
            base = (
                primary_scope(members=labels, months=months, band=band)
                if target == "primary"
                else solar_scope(months=months, band=band, farms=labels)
            )
            return base if extra is None else base & extra

        for group, labels in groups.items():
            for band_name, band in bands.items():
                _print_slice(
                    source=source,
                    treatment=treatment,
                    reference=reference,
                    name=f"{group}, {band_name}",
                    where=scope(labels=labels, band=band),
                )
            for regime in REGIMES:
                _print_slice(
                    source=source,
                    treatment=treatment,
                    reference=reference,
                    name=f"{group}, {regime} half-hours",
                    where=scope(labels=labels, extra=pl.col("regime").cast(pl.String) == regime),
                )
            for label in labels:
                _print_slice(
                    source=source,
                    treatment=treatment,
                    reference=reference,
                    name=f"label {label}",
                    where=scope(labels=(label,)),
                )
                _print_slice(
                    source=source,
                    treatment=treatment,
                    reference=reference,
                    name=f"{group} without {label}",
                    where=scope(labels=tuple(m for m in labels if m != label)),
                )
            for month in cloud:
                _print_slice(
                    source=source,
                    treatment=treatment,
                    reference=reference,
                    name=f"{group}, fold {month} ({cloud.index(month) + 1} months of recent cloud)",
                    where=scope(labels=labels, months=(month,)),
                )
    print("\n-- total effect and date-proxy effect (95%; six and eight separately) --")
    for group, members in PRIMARY_GROUPS.items():
        where = primary_scope(members=members, months=cloud)
        for treatment, reference in (("D1", "D0"), ("DF", "D0"), ("D1g", "D0g"), ("D3", "D1p")):
            _print_slice(
                source=source,
                treatment=treatment,
                reference=reference,
                name=f"{group}, all in-scope rows",
                where=where,
            )
    for treatment, reference in (("P1", "P0L"), ("PF", "P0L"), ("P1g", "P0g"), ("P1", "P0")):
        _print_slice(
            source=source,
            treatment=treatment,
            reference=reference,
            name="six farms, all in-scope rows",
            where=solar_scope(months=arms.SOLAR_MONTHS),
        )


def _print_slice(
    *, source: LossSource, treatment: str, reference: str, name: str, where: pl.Expr
) -> None:
    """Print one exploratory reading, or say why it is unavailable."""
    if source.get(treatment) is None or source.get(reference) is None:
        return
    try:
        reading = read_contrast(
            source=source,
            treatment=treatment,
            reference=reference,
            label=f"{treatment} minus {reference}",
            scope=name,
            where=where,
            level=SLICE_LEVEL,
            n_resamples=N_SLICE_RESAMPLES,
        )
    except ValueError as error:
        print(f"{treatment} minus {reference} {name}: unavailable ({error})")
        return
    print(reading.exploratory_line())


def gate(*, source: LossSource, record: bool) -> bool:
    """Print and return the fit gate: `D0n` matches stored `R_F` on the gate folds.

    On each gate fold and each group, the two arms must hold the same rows, and their headline CRPS
    (days 1 to 10) must differ by at most 2 standard errors of the seed-to-seed spread. With
    `record`, each fold's result goes to the gate marker.
    """
    print(
        "\n== fit gate: D0n against stored R_F (D0), within 2 standard errors of the seed spread =="
    )
    new, stored = source.get("D0n"), source.get("D0")
    if new is None or stored is None:
        print("not run: D0n or D0 is not in the store")
        return False
    ok_all = True
    for month in arms.GATE_MONTHS:
        month_ok = True
        details = []
        for group, members in PRIMARY_GROUPS.items():
            where = primary_scope(members=members, months=(month,))
            frames = {"D0n": new.losses.filter(where), "D0": stored.losses.filter(where)}
            first = {
                name: frame.filter(pl.col("seed") == frame["seed"].min()).select(*KEYS).sort(*KEYS)
                for name, frame in frames.items()
            }
            if not first["D0n"].equals(first["D0"]):
                text = f"{month} {group}: FAIL, D0n and D0 hold different rows"
                print(text)
                details.append(text)
                month_ok = False
                continue
            means = {}
            for name, frame in frames.items():
                per_seed = frame.group_by("seed").agg(pl.col("crps").mean())["crps"].to_numpy()
                means[name] = (float(per_seed.mean()), float(per_seed.std(ddof=1)), len(per_seed))
            difference = means["D0n"][0] - means["D0"][0]
            error = float(
                np.sqrt(
                    means["D0n"][1] ** 2 / means["D0n"][2] + means["D0"][1] ** 2 / means["D0"][2]
                )
            )
            passes = abs(difference) <= 2.0 * error
            month_ok &= passes
            text = (
                f"{month} {group}: D0n {means['D0n'][0]:.3f}, D0 {means['D0'][0]:.3f}, "
                f"difference {difference:+.3f}, 2 SE {2 * error:.3f} -> "
                f"{'PASS' if passes else 'FAIL'}"
            )
            print(text)
            details.append(text)
        ok_all &= month_ok
        if record:
            gate_marker.record(
                kind="fit",
                month=month,
                passed=month_ok,
                detail="; ".join(details),
                commit=gate_marker.head_commit(),
            )
    return ok_all


# --- commands ---------------------------------------------------------------------------------


def labels_and_sky(*, with_sky: bool) -> tuple[dict[int, str], pl.DataFrame | None]:
    """Return the label of every series (Primary `D` numbers, farm letters) and the sky table."""
    import e00707_common as base  # ty: ignore[unresolved-import]
    from studies.pv_dataset import pv_sites

    primaries = base.series_table().filter(
        pl.col("substation_type").is_in(["Primary", "BSP", "GSP"])
    )
    labels = dict(
        zip(primaries["time_series_id"].to_list(), primaries["site"].to_list(), strict=True)
    )
    farms = pv_sites()
    labels |= dict(zip(farms["time_series_id"].to_list(), farms["site"].to_list(), strict=True))
    sky = pl.read_parquet(SKY_INDEX_PATH) if with_sky and SKY_INDEX_PATH.exists() else None
    if with_sky and sky is None:
        print(f"no sky index at {SKY_INDEX_PATH}: regime slices are 'unknown'; run `sky-index`")
    return labels, sky


def build_sky_index() -> None:
    """Write the realised clear-sky index and sun elevation of the Primaries and the farms.

    The valid times are every half-hour from 2025-07-01 to 2026-07-01 and, for the replicate, from
    2023-04-01 to 2024-04-11.
    """
    import e00707_common as base  # ty: ignore[unresolved-import]
    import e00707_prep as prep  # ty: ignore[unresolved-import]
    from studies.pv_dataset import pv_sites, read_cams

    spans = (
        (datetime(2023, 4, 1, tzinfo=UTC), datetime(2024, 4, 11, tzinfo=UTC)),
        (datetime(2025, 7, 1, tzinfo=UTC), datetime(2026, 7, 1, tzinfo=UTC)),
    )
    times = pl.concat(
        [
            pl.datetime_range(a, b, interval="30m", time_zone="UTC", eager=True, closed="left")
            for a, b in spans
        ]
    )
    series = base.series_table().filter(pl.col("substation_type") == "Primary")
    cams_primary = pl.read_parquet(prep.CAMS_PATH).select("site", time="time", ghi_w_m2="ghi_w_m2")
    primary = sky_index_table(
        sites=series.select("time_series_id", "site", "latitude", "longitude"),
        cams=cams_primary,
        valid_times=times,
    )
    farms = pv_sites()
    solar = sky_index_table(
        sites=farms.select("time_series_id", "site", "latitude", "longitude"),
        cams=read_cams(min_reliability=0.0),
        valid_times=times,
    )
    SKY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SKY_INDEX_PATH.with_suffix(".tmp")
    pl.concat([primary, solar]).write_parquet(temporary)
    temporary.rename(SKY_INDEX_PATH)
    print(f"wrote {SKY_INDEX_PATH.name}")


def make_source(*, with_sky: bool) -> LossSource:
    """Return a loss source over the experiment store."""
    labels, sky = labels_and_sky(with_sky=with_sky)

    def load(name: str) -> ArmLosses | None:
        return prepare(arm=arms.ARMS[name], labels=labels, sky=sky)

    return LossSource(load=load)


def score(*, n_resamples: int, stage: StageType) -> None:
    """Print every number the page quotes for `stage`, in the order of the module docstring."""
    if stage == "B1":
        score_b1(n_resamples=n_resamples)
        return
    source = make_source(with_sky=True)
    print_arms(source=source, stage=stage)
    if stage == "A":
        print_interim_mde(source=source, n_resamples=n_resamples)
    else:
        print_mde(source=source, n_resamples=n_resamples)
    print(f"\n== planned readings, level {PLANNED_LEVEL:.2f}%, {n_resamples:,} resamples ==")
    readings = [
        run_spec(source=source, spec=s, stage=stage, n_resamples=n_resamples)
        for s in planned_specs(stage=stage)
    ]
    print_outcomes(outcomes=readings, rows=True)
    print(f"\n== checks, level {CHECK_LEVEL:.3f}% ==")
    checks = [
        run_spec(source=source, spec=s, stage=stage, n_resamples=n_resamples)
        for s in check_specs(stage=stage)
    ]
    print_outcomes(outcomes=checks)
    for key, passed in check_results(outcomes=checks).items():
        text = "incomplete" if passed is None else ("PASS" if passed else "FAIL (audit trigger 5)")
        print(f"  placebo or gap check {key}: {text}")
    controls: dict[str, bool | None] = {}
    if stage != "A":
        control_outcomes = [
            run_spec(source=source, spec=s, stage=stage, n_resamples=n_resamples)
            for s in control_specs()
        ]
        print_outcomes(outcomes=control_outcomes)
        controls = control_results(outcomes=control_outcomes)
    print("\n== bounds beside C1 and S1: the real arm against no cloud (99.375%) ==")
    print_outcomes(
        outcomes=[
            run_spec(source=source, spec=b, stage=stage, n_resamples=n_resamples)
            for b in bound_specs()
        ]
    )
    verdicts(
        readings={r.key: r for r in readings},
        controls=controls,
        stage=stage,
        checks=check_results(outcomes=checks),
    )
    exploratory(source=source)


class _Tee:
    """Write each string to the real output and to a file."""

    def __init__(self, *, first: TextIO, second: TextIO) -> None:
        self._streams = (first, second)

    def write(self, text: str) -> int:
        """Write `text` to both streams."""
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        """Flush both streams."""
        for stream in self._streams:
            stream.flush()


def scores_path(*, name: str, now: datetime | None = None) -> Path:
    """Return `work/scores/<name>_<UTC timestamp>.txt` under the experiment's data folder."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return common.WORK_DIR / "work" / "scores" / f"{name}_{stamp}.txt"


def run_saved(*, name: str, action: Callable[[], None]) -> Path:
    """Run `action`, print its output as usual, and save the same output to `scores_path`."""
    path = scores_path(name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        tee = _Tee(first=sys.stdout, second=handle)
        with contextlib.redirect_stdout(tee):  # ty: ignore[invalid-argument-type]
            action()
    print(f"saved the output to {path}")
    return path


def main() -> None:
    """Run a scorer command."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("score", "gate", "sky-index"))
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--stage", choices=("A", "B1", "B", "all"), default="A")
    args = parser.parse_args()
    arms.check_table()
    if args.command == "sky-index":
        run_saved(name="sky-index", action=build_sky_index)
    elif args.command == "gate":

        def run_gate() -> None:
            passed = gate(source=make_source(with_sky=False), record=True)
            print("gate:", "PASS" if passed else "FAIL")

        run_saved(name="gate", action=run_gate)
    else:
        run_saved(
            name=f"score-{args.stage}",
            action=lambda: score(n_resamples=args.resamples, stage=args.stage),
        )
    _ = timedelta


if __name__ == "__main__":
    main()

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
```
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import e00711_arms as arms
import e00711_common as common
import numpy as np
import polars as pl
from scipy import stats
from studies.era5_ladder import CLEAR_SKY_INDEX_THRESHOLDS
from studies.eval_protocol import metrics

from studies import bootstrap, experiment_store

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
CHECK_LEVEL: Final[float] = 99.0
"""The common level of the five checks that the plan reads at 95%, corrected to 99%."""
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
        c2: The verdict of C2 at the group.
        c3: The verdict of C3 at the group, or `None` if the replicate has not been scored.
        c3_lower: The lower limit of C3's interval, for audit trigger 2.
        control_passed: Whether the positive control passed (it makes both groups `inconclusive`
            when it fails).
    """

    c1: VerdictType
    c2: VerdictType
    c3: VerdictType | None
    c3_lower: float | None
    control_passed: bool


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
    base = better(first=reading.c1, second=reading.c2)
    if not reading.control_passed:
        fired.append("positive control failed: both groups inconclusive")
        return "inconclusive", fired
    if (
        "supported" in (reading.c1, reading.c2)
        and reading.c3_lower is not None
        and reading.c3_lower > 0.0
    ):
        fired.append("audit trigger 2: C1 or C2 supported and the replicate's lower limit above 0")
        return "inconclusive", fired
    if (
        reading.c1 in ("inconclusive", "not supported")
        and reading.c2 in ("inconclusive", "not supported")
        and reading.c3 == "supported"
    ):
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
class Reading:
    """One contrast on one set of rows.

    Attributes:
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
    """

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


def t_interval_of_months(
    *, differences: np.ndarray, months: np.ndarray, level: float
) -> tuple[float, float] | None:
    """Return the t interval on the monthly mean differences, or `None` with fewer than 2 months.

    Args:
        differences: Per-seed, per-row differences, shape (seeds, rows).
        months: The month label of each row.
        level: The coverage in percent.
    """
    labels = np.unique(months)
    if len(labels) < 2:
        return None
    monthly = np.array([differences[:, months == m].mean() for m in labels])
    half = float(
        stats.t.ppf(1.0 - (1.0 - level / 100.0) / 2.0, df=len(monthly) - 1)
        * monthly.std(ddof=1)
        / np.sqrt(len(monthly))
    )
    return float(monthly.mean()) - half, float(monthly.mean()) + half


def contrast_losses(
    *,
    losses: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    where: pl.Expr | None,
    seeds: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """Return `losses_for_bootstrap`'s frame for two arms on the rows and seeds they share.

    Args:
        losses: Each arm's per-row loss frame with `fold`, `label`, `lead_hours`, and `regime`.
        treatment: The treatment arm's name.
        reference: The reference arm's name.
        where: A filter on the loss frames' columns, or `None`.
        seeds: The seeds to use; the seeds that both arms hold if not given.

    Returns:
        The frame of both arms, built by `experiment_store.losses_for_bootstrap`, which raises if
        the arms hold different rows or seeds.

    Raises:
        ValueError: If the two arms share no seed.
    """
    pair = {}
    held = [set(losses[a]["seed"].unique().to_list()) for a in (treatment, reference)]
    shared = tuple(sorted(held[0] & held[1])) if seeds is None else seeds
    if not shared:
        msg = f"{treatment} and {reference} share no seed"
        raise ValueError(msg)
    for name in (treatment, reference):
        frame = losses[name].filter(pl.col("seed").is_in(list(shared)))
        pair[name] = frame if where is None else frame.filter(where)
    return experiment_store.losses_for_bootstrap(arm_losses=pair)


def read_contrast(
    *,
    losses: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    label: str,
    scope: str,
    where: pl.Expr | None,
    level: float,
    n_resamples: int,
    metric: str = "crps",
    averaged: dict[str, pl.DataFrame] | None = None,
    seeds: tuple[int, ...] | None = None,
    t_only: bool = False,
) -> Reading:
    """Return one contrast: its difference, the bootstrap and t intervals, and the decisive one.

    Args:
        losses: Each arm's per-row losses.
        treatment: The treatment arm.
        reference: The reference arm.
        label: The contrast's name.
        scope: The rows' description.
        where: A filter on the loss frames' columns.
        level: The interval's coverage in percent.
        n_resamples: The number of bootstrap resamples.
        metric: The loss column.
        averaged: Each arm's seed-averaged losses, for the descriptive reading.
        seeds: The seeds, or `None` for the seeds that both arms hold.
        t_only: Read only the t interval (for a contrast on a few folds, where the bootstrap has
            few distinct resamples).

    Returns:
        The reading.
    """
    frame = contrast_losses(
        losses=losses, treatment=treatment, reference=reference, where=where, seeds=seeds
    )
    differences, months = bootstrap.paired_differences(
        losses=frame, treatment=treatment, reference=reference, metric=metric
    )
    boot = (
        None
        if t_only
        else bootstrap.bootstrap_difference_at_level(
            losses=frame,
            treatment=treatment,
            reference=reference,
            metric=metric,
            level=level,
            n_resamples=n_resamples,
        )
    )
    t_int = t_interval_of_months(differences=differences, months=months, level=level)
    candidates = [i for i in (boot, t_int) if i is not None]
    if not candidates:
        msg = f"{label} {scope}: no interval can be computed"
        raise ValueError(msg)
    lower = min(i[0] for i in candidates)
    upper = max(i[1] for i in candidates)
    seed_averaged = None
    if averaged is not None:
        seed_averaged = _averaged_difference(
            averaged=averaged, treatment=treatment, reference=reference, where=where, metric=metric
        )
    return Reading(
        label=label,
        scope=scope,
        difference=float(differences.mean()),
        lower=lower,
        upper=upper,
        boot=boot,
        t_interval=t_int,
        averaged=seed_averaged,
        n_half_hours=int(differences.shape[1]),
        n_months=len(np.unique(months)),
        seeds=tuple(sorted(frame["seed"].unique().to_list())),
        level=level,
    )


def _averaged_difference(
    *,
    averaged: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    where: pl.Expr | None,
    metric: str,
) -> float | None:
    """Return the difference of two arms' seed-averaged losses on their shared rows, or `None`."""
    pair = {}
    for name in (treatment, reference):
        frame = averaged[name]
        pair[name] = frame if where is None else frame.filter(where)
    keys = list(KEYS)
    joined = (
        pair[treatment]
        .select(*keys, a=pl.col(metric))
        .join(pair[reference].select(*keys, b=pl.col(metric)), on=keys, how="inner")
    )
    return None if joined.is_empty() else float((joined["a"] - joined["b"]).mean())  # ty: ignore[invalid-argument-type]


def minimum_detectable_effect(
    *,
    losses: dict[str, pl.DataFrame],
    arm: str,
    target: Literal["primary", "solar"],
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
    held = set(losses[arm]["seed"].unique().to_list())
    if not (set(first) <= held and set(second) <= held):
        return None
    shift = dict(zip(second, first, strict=True))
    base = losses[arm] if where is None else losses[arm].filter(where)
    one = base.filter(pl.col("seed").is_in(list(first)))
    two = base.filter(pl.col("seed").is_in(list(second))).with_columns(
        seed=pl.col("seed").replace_strict(shift, return_dtype=pl.Int64)
    )
    frame = experiment_store.losses_for_bootstrap(arm_losses={"half_a": one, "half_b": two})
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=frame,
        treatment="half_a",
        reference="half_b",
        metric="crps",
        level=level,
        n_resamples=n_resamples,
    )
    return (upper - lower) / 2.0


# --- reading the store ------------------------------------------------------------------------


def available(*, arm: str, month: str, seed: int) -> bool:
    """Return whether the store holds the arm's set for the fold and seed."""
    folder = common.WORK_DIR / "predictions" / arms.set_name(arm=arm, month=month, seed=seed)
    return folder.exists()


def complete_seeds(*, arm: arms.Arm, months: tuple[str, ...]) -> tuple[int, ...]:
    """Return the arm's seeds whose set exists for every one of `months`."""
    return tuple(
        s for s in arm.seeds if all(available(arm=arm.name, month=m, seed=s) for m in months)
    )


def read_arm(*, arm: arms.Arm, months: tuple[str, ...], seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return an arm's predictions over folds and seeds, with the `seed` and `fold` columns.

    The stored `D0` reads experiment 707's stored `R_F` raw quantiles.
    """
    pieces = []
    for month in months:
        for seed in seeds:
            if arm.stored:
                frame = pl.read_parquet(_stored_path(arm=arm, month=month, seed=seed)).with_columns(
                    seed=pl.lit(seed).cast(pl.Int64)
                )
            else:
                frame = experiment_store.read_predictions(
                    experiment_id=arms.EXPERIMENT_ID,
                    name=arms.set_name(arm=arm.name, month=month, seed=seed),
                    row_set_id=arm.row_set_id,
                )
            pieces.append(frame.with_columns(fold=pl.lit(month)))
    return pl.concat(pieces)


def _stored_path(*, arm: arms.Arm, month: str, seed: int) -> Path:
    """Return the stored forecast of a stored arm: experiment 707's `R_F`."""
    if arm.name != "D0":
        msg = (
            f"only D0 (experiment 707's R_F) is a stored arm that the scorer reads, not {arm.name}"
        )
        raise ValueError(msg)
    import e00707_arms as arms707  # ty: ignore[unresolved-import]

    return arms707.WORK_DIR / "work" / "seed_preds" / "R_F" / f"{month}_s{seed}.parquet"


def primary_actuals(*, frame: pl.DataFrame) -> np.ndarray:
    """Return each Primary row's observed power as a fraction of effective capacity.

    The actual is read from experiment 707's dense half-hourly arrays. Rows whose actual is a
    flagged exact-zero half-hour are marked NaN (and are dropped by the caller).
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
            for site in FARMS
        ]
    ).with_columns(pl.col("time_series_id").cast(pl.Int64))
    joined = frame.select("time_series_id", "valid_time").join(
        farms,
        left_on=["time_series_id", "valid_time"],
        right_on=["time_series_id", "time"],
        how="left",
    )
    return joined["actual"].cast(pl.Float64).fill_null(float("nan")).to_numpy()


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


def prepare(
    *,
    arm: arms.Arm,
    months: tuple[str, ...],
    seeds: tuple[int, ...],
    labels: dict[int, str],
    sky: pl.DataFrame | None,
    actual_cache: dict[tuple[str, tuple[str, ...]], tuple[pl.DataFrame, np.ndarray]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return an arm's per-seed losses and seed-averaged losses on rows with an observed actual.

    The loss frames carry `fold` (the month of the run), `label` (D-number or farm letter),
    `lead_hours` (from the origin), `lead_from_run_hours` (solar), and `regime`.
    """
    frame = read_arm(arm=arm, months=months, seeds=seeds)
    frame = frame.with_columns(
        pl.col("time_series_id").cast(pl.Int64),
        fold=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
    )
    if arm.target == "primary":
        frame = widen(frame=frame)
    actual = primary_actuals(frame=frame) if arm.target == "primary" else solar_actuals(frame=frame)
    frame, actual = drop_flagged_and_missing(frame=frame, actual=actual, target=arm.target)
    averaged_frame = seed_average(frame=frame)
    out = []
    for kind, source in (("seed", frame), ("avg", averaged_frame)):
        values = (
            actual if kind == "seed" else _actuals_for(frame=source, reference=frame, actual=actual)
        )
        loss = row_losses(frame=source, actual=values)
        extra = loss.with_columns(
            fold=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            label=pl.col("time_series_id").replace_strict(
                labels, default="?", return_dtype=pl.String
            ),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
        ).with_columns(lead_from_run_hours=pl.col("lead_hours") + SOLAR_RUN_OFFSET_HOURS)
        out.append(extra if sky is None else add_regime(losses=extra, sky=sky))
    _ = actual_cache
    return out[0], out[1]


def _actuals_for(*, frame: pl.DataFrame, reference: pl.DataFrame, actual: np.ndarray) -> np.ndarray:
    """Return the actual of each row of the seed-averaged `frame`, looked up from `reference`."""
    table = reference.select(*KEYS).with_columns(actual=pl.Series(actual)).unique(subset=list(KEYS))
    return frame.select(*KEYS).join(table, on=list(KEYS), how="left")["actual"].to_numpy()


# --- scope expressions ------------------------------------------------------------------------


def primary_scope(
    *, members: tuple[str, ...], months: tuple[str, ...], band: tuple[float, float] | None = None
) -> pl.Expr:
    """Return the rows of Primaries `members` on folds `months` at the headline or a lead band."""
    low, high = (0.0, PRIMARY_HEADLINE_HOURS) if band is None else band
    return (
        pl.col("label").is_in(list(members))
        & pl.col("fold").is_in(list(months))
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
    return (
        pl.col("label").is_in(list(farms))
        & pl.col("fold").is_in(list(months))
        & (pl.col("lead_from_run_hours") >= low)
        & (pl.col("lead_from_run_hours") < high)
    )


# --- the scoring run --------------------------------------------------------------------------


@dataclass
class Store:
    """The losses of every arm that the store holds, per seed and seed-averaged."""

    losses: dict[str, pl.DataFrame]
    averaged: dict[str, pl.DataFrame]


def load_store(*, labels: dict[int, str], sky: pl.DataFrame | None) -> Store:
    """Read every arm that has at least one complete seed on its folds."""
    losses: dict[str, pl.DataFrame] = {}
    averaged: dict[str, pl.DataFrame] = {}
    for arm in arms.ARM_LIST:
        seeds = complete_seeds(arm=arm, months=arm.months) if not arm.stored else arm.seeds
        if not seeds:
            continue
        if arm.stored:
            try:
                losses[arm.name], averaged[arm.name] = prepare(
                    arm=arm, months=arm.months, seeds=seeds, labels=labels, sky=sky, actual_cache={}
                )
            except FileNotFoundError:
                continue
        else:
            losses[arm.name], averaged[arm.name] = prepare(
                arm=arm, months=arm.months, seeds=seeds, labels=labels, sky=sky, actual_cache={}
            )
        print(f"{arm.name:4s} seeds {list(seeds)} folds {len(arm.months)}")
    return Store(losses=losses, averaged=averaged)


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


def planned_readings(*, store: Store, n_resamples: int) -> dict[str, Reading]:
    """Compute the nine planned readings of section 8 at 99.44%.

    Returns:
        The readings keyed `C1 the six`, `C1 the eight`, `C2 ...`, `C3 the five`, `C3 the eight`,
        `S1`, `S2`, and `S3`. A reading whose arms are not all in the store is absent.
    """
    losses, averaged = store.losses, store.averaged
    cloud = arms.CLOUD_SCORED_MONTHS
    out: dict[str, Reading] = {}

    def add(key: str, treatment: str, reference: str, scope: str, where: pl.Expr) -> None:
        if treatment not in losses or reference not in losses:
            print(f"{key}: skipped, {treatment} or {reference} is not in the store")
            return
        reading = read_contrast(
            losses=losses,
            treatment=treatment,
            reference=reference,
            label=f"{key} {treatment} minus {reference}",
            scope=scope,
            where=where,
            level=PLANNED_LEVEL,
            n_resamples=n_resamples,
            averaged=averaged,
        )
        out[key] = reading
        print(reading.line(), flush=True)

    for name, (real, fake) in {"C1": ("D1", "D1p"), "C2": ("D2", "D2p")}.items():
        for group, members in PRIMARY_GROUPS.items():
            add(
                f"{name} {group}",
                real,
                fake,
                f"{group}, days 1 to 10, 7 folds",
                primary_scope(members=members, months=cloud),
            )
    for group, members in REPLICATE_GROUPS.items():
        add(
            f"C3 {group}",
            "R1",
            "R1p",
            f"{group}, days 1 to 10, replicate",
            primary_scope(members=members, months=arms.REPLICATE_MONTHS),
        )
    add("S1", "P1", "P1p", "six farms, days 1 to 10", solar_scope(months=arms.SOLAR_MONTHS))
    add("S2", "P2", "P2p", "six farms, days 1 to 10", solar_scope(months=arms.SOLAR_MONTHS))
    add(
        "S3",
        "P1",
        "P1p",
        "six farms, 12 to 36 hours",
        solar_scope(months=arms.SOLAR_MONTHS, band=SOLAR_S3_BAND),
    )
    return out


def print_mde(*, store: Store, n_resamples: int) -> dict[str, float | None]:
    """Print the minimum detectable effect of each planned reading's group before any contrast."""
    print("\n== minimum detectable effect (before any contrast; half-width at 99.44%) ==")
    cloud = arms.CLOUD_SCORED_MONTHS
    result: dict[str, float | None] = {}
    cases = (
        [
            ("D1p", "primary", f"{group}, 7 folds", primary_scope(members=members, months=cloud))
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
        + [("P1p", "solar", "six farms, days 1 to 10", solar_scope(months=arms.SOLAR_MONTHS))]
    )
    for arm, target, scope, where in cases:
        if arm not in store.losses:
            print(f"{arm} {scope}: not computed, the arm is not in the store")
            continue
        mde = minimum_detectable_effect(
            losses=store.losses,
            arm=arm,
            target=target,  # ty: ignore[invalid-argument-type]
            where=where,
            level=PLANNED_LEVEL,
            n_resamples=n_resamples,
        )
        result[f"{arm} {scope}"] = mde
        text = (
            "not computed: the arm lacks both halves' seeds" if mde is None else f"{mde:.3f} points"
        )
        print(f"{arm} {scope}: {text}")
    return result


def checks(*, store: Store, n_resamples: int) -> dict[str, bool]:
    """Print the controls and return which passed (an interval that contains zero, or below it).

    Placebo carries nothing: `D1p` minus `DF`, `D1t` minus `DF`, `P1p` minus `PF`, `P1t` minus `PF`
    each have a 99% interval that contains zero. Gap-scored folds: `D1` minus `D1p` on folds
    2025-07 to 2025-10, read with the t interval over the four folds. Positive control: `K1` minus
    `K1p` is wholly below zero at the six and at the eight.
    """
    print("\n== controls (99%) ==")
    losses = store.losses
    passed: dict[str, bool] = {}
    cloud = arms.CLOUD_SCORED_MONTHS
    pooled = (*SIX, *EIGHT)
    for key, treatment, reference in (
        ("D1p minus DF", "D1p", "DF"),
        ("D1t minus DF", "D1t", "DF"),
        ("P1p minus PF", "P1p", "PF"),
        ("P1t minus PF", "P1t", "PF"),
    ):
        if treatment not in losses or reference not in losses:
            print(f"{key}: skipped")
            continue
        solar_pair = treatment.startswith("P")
        where = (
            solar_scope(months=arms.SOLAR_MONTHS)
            if solar_pair
            else primary_scope(members=pooled, months=cloud)
        )
        reading = read_contrast(
            losses=losses,
            treatment=treatment,
            reference=reference,
            label=key,
            scope="solar farms" if solar_pair else "the six and the eight",
            where=where,
            level=CHECK_LEVEL,
            n_resamples=n_resamples,
        )
        passed[key] = reading.lower <= 0.0 <= reading.upper
        print(
            f"{reading.line()} placebo carries nothing: "
            f"{'PASS' if passed[key] else 'FAIL (audit trigger 5)'}"
        )
    if "D1" in losses and "D1p" in losses:
        reading = read_contrast(
            losses=losses,
            treatment="D1",
            reference="D1p",
            label="gap-scored D1 minus D1p",
            scope="the six and the eight, 4 folds",
            where=primary_scope(members=pooled, months=arms.GAP_SCORED_MONTHS),
            level=CHECK_LEVEL,
            n_resamples=n_resamples,
            t_only=True,
        )
        passed["gap-scored"] = reading.lower <= 0.0 <= reading.upper
        print(
            f"{reading.line()} gap-scored: "
            f"{'PASS' if passed['gap-scored'] else 'FAIL (audit trigger 5)'}"
        )
    for group, members in PRIMARY_GROUPS.items():
        if "K1" in losses and "K1p" in losses:
            reading = read_contrast(
                losses=losses,
                treatment="K1",
                reference="K1p",
                label="positive control K1 minus K1p",
                scope=f"{group}, 7 folds",
                where=primary_scope(members=members, months=cloud),
                level=CHECK_LEVEL,
                n_resamples=n_resamples,
            )
            passed[f"K1 {group}"] = reading.upper < 0.0
            print(f"{reading.line()} wholly below 0: {'PASS' if passed[f'K1 {group}'] else 'FAIL'}")
    return passed


def verdicts(*, readings: dict[str, Reading], passed: dict[str, bool]) -> None:
    """Print the verdict of each Primary demand group, the solar farms, and the front matter."""
    print("\n== verdicts (decision rule, section 8) ==")
    control_ok = (
        all(v for k, v in passed.items() if k.startswith("K1"))
        if any(k.startswith("K1") for k in passed)
        else False
    )
    group_results: dict[str, TargetVerdictType] = {}
    for group in PRIMARY_GROUPS:
        c1, c2 = readings.get(f"C1 {group}"), readings.get(f"C2 {group}")
        c3_group = "the five" if group == "the six" else group
        c3 = readings.get(f"C3 {c3_group}")
        if c1 is None or c2 is None:
            print(f"{group}: not read, C1 or C2 is missing")
            continue
        result, fired = group_verdict(
            reading=GroupReading(
                c1=c1.verdict,
                c2=c2.verdict,
                c3=None if c3 is None else c3.verdict,
                c3_lower=None if c3 is None else c3.lower,
                control_passed=control_ok,
            )
        )
        group_results[group] = result
        print(
            f"{group}: C1 {c1.verdict}, C2 {c2.verdict}, "
            f"C3 {None if c3 is None else c3.verdict} -> {result}; {fired}"
        )
    s1, s2 = readings.get("S1"), readings.get("S2")
    solar_result: TargetVerdictType | None = None
    if s1 is not None and s2 is not None:
        solar_result = better(first=s1.verdict, second=s2.verdict)
        print(f"solar farms: S1 {s1.verdict}, S2 {s2.verdict} -> {solar_result}")
    if len(group_results) == len(PRIMARY_GROUPS) and solar_result is not None:
        print(
            "front-matter verdict:",
            front_matter_verdict(
                six=group_results["the six"], eight=group_results["the eight"], solar=solar_result
            ),
        )


def exploratory(*, store: Store) -> None:
    """Print the exploratory readings at 95%: lead bands, sky regimes, labels, folds, totals."""
    print("\n== exploratory readings (95% interval; each with its half-hour count) ==")
    losses, averaged = store.losses, store.averaged
    cloud = arms.CLOUD_SCORED_MONTHS
    pairs = (("D1", "D1p", "primary"), ("P1", "P1p", "solar"))
    for treatment, reference, target in pairs:
        if treatment not in losses or reference not in losses:
            continue
        bands = PRIMARY_BANDS if target == "primary" else SOLAR_BANDS
        members = SIX if target == "primary" else FARMS

        def scope(
            *,
            extra: pl.Expr | None = None,
            band: tuple[float, float] | None = None,
            labels: tuple[str, ...] = members,
            months: tuple[str, ...] = cloud,
            target: str = target,
        ) -> pl.Expr:
            base = (
                primary_scope(members=labels, months=months, band=band)
                if target == "primary"
                else solar_scope(months=arms.SOLAR_MONTHS, band=band, farms=labels)
            )
            return base if extra is None else base & extra

        groups = PRIMARY_GROUPS if target == "primary" else {"six farms": FARMS}
        for group, labels in groups.items():
            for band_name, band in bands.items():
                _print_slice(
                    losses=losses,
                    averaged=averaged,
                    treatment=treatment,
                    reference=reference,
                    name=f"{group}, {band_name}",
                    where=scope(labels=labels, band=band),
                )
            for regime in REGIMES:
                if "regime" in losses[treatment].columns:
                    _print_slice(
                        losses=losses,
                        averaged=averaged,
                        treatment=treatment,
                        reference=reference,
                        name=f"{group}, {regime} half-hours",
                        where=scope(labels=labels, extra=pl.col("regime") == regime),
                    )
        for label in (*SIX, *EIGHT, *SHORT_HISTORY) if target == "primary" else FARMS:
            _print_slice(
                losses=losses,
                averaged=averaged,
                treatment=treatment,
                reference=reference,
                name=f"label {label}",
                where=scope(labels=(label,)),
            )
        for month in cloud:
            _print_slice(
                losses=losses,
                averaged=averaged,
                treatment=treatment,
                reference=reference,
                name=f"fold {month} (months of recent cloud {cloud.index(month) + 1})",
                where=scope(months=(month,)),
            )
        for label in members:
            _print_slice(
                losses=losses,
                averaged=averaged,
                treatment=treatment,
                reference=reference,
                name=f"leave out {label}",
                where=scope(labels=tuple(m for m in members if m != label)),
            )
    print("\n-- total effect and date-proxy effect (95%) --")
    for treatment, reference, where in (
        ("D1", "D0", primary_scope(members=(*SIX, *EIGHT), months=cloud)),
        ("DF", "D0", primary_scope(members=(*SIX, *EIGHT), months=cloud)),
        ("P1", "P0L", solar_scope(months=arms.SOLAR_MONTHS)),
        ("PF", "P0L", solar_scope(months=arms.SOLAR_MONTHS)),
        ("D1g", "D0g", primary_scope(members=(*SIX, *EIGHT), months=cloud)),
        ("P1g", "P0g", solar_scope(months=arms.SOLAR_MONTHS)),
        ("D3", "D1p", primary_scope(members=(*SIX, *EIGHT), months=cloud)),
    ):
        if treatment in losses and reference in losses:
            _print_slice(
                losses=losses,
                averaged=averaged,
                treatment=treatment,
                reference=reference,
                name="all in-scope rows",
                where=where,
            )


def _print_slice(
    *,
    losses: dict[str, pl.DataFrame],
    averaged: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    name: str,
    where: pl.Expr,
) -> None:
    """Print one exploratory reading, or say why it is unavailable."""
    try:
        reading = read_contrast(
            losses=losses,
            treatment=treatment,
            reference=reference,
            label=f"{treatment} minus {reference}",
            scope=name,
            where=where,
            level=SLICE_LEVEL,
            n_resamples=N_SLICE_RESAMPLES,
            averaged=averaged,
        )
    except ValueError as error:
        print(f"{treatment} minus {reference} {name}: unavailable ({error})")
        return
    print(reading.line())


def gate(*, store: Store) -> bool:
    """Print and return the gate: `D0n` matches stored `R_F` on folds 2026-01 and 2026-06.

    The two arms' headline CRPS at the six and the eight (days 1 to 10, equal rows) must differ by
    at most 2 standard errors of the seed-to-seed spread, per fold.
    """
    print("\n== gate: D0n against stored R_F (D0), within 2 standard errors of the seed spread ==")
    losses = store.losses
    if "D0n" not in losses or "D0" not in losses:
        print("not run: D0n or D0 is not in the store")
        return False
    ok = True
    for month in arms.GATE_MONTHS:
        for group, members in PRIMARY_GROUPS.items():
            where = primary_scope(members=(*members,), months=(month,))
            means = {}
            for name in ("D0n", "D0"):
                frame = losses[name].filter(where)
                keys = frame.select(*KEYS).unique().height
                per_seed = frame.group_by("seed").agg(pl.col("crps").mean())["crps"].to_numpy()
                means[name] = (
                    float(per_seed.mean()),
                    float(per_seed.std(ddof=1)),
                    len(per_seed),
                    keys,
                )
            difference = means["D0n"][0] - means["D0"][0]
            error = float(
                np.sqrt(
                    means["D0n"][1] ** 2 / means["D0n"][2] + means["D0"][1] ** 2 / means["D0"][2]
                )
            )
            passes = abs(difference) <= 2.0 * error
            ok &= passes
            print(
                f"{month} {group}: D0n {means['D0n'][0]:.3f}, D0 {means['D0'][0]:.3f}, "
                f"difference {difference:+.3f}, 2 SE {2 * error:.3f} -> "
                f"{'PASS' if passes else 'FAIL'}"
            )
    return ok


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
    2023-04-01 to 2024-04-01.
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


def score(*, n_resamples: int) -> None:
    """Print every number the page quotes, in the order of the module docstring."""
    labels, sky = labels_and_sky(with_sky=True)
    print("== arms found in the store ==")
    store = load_store(labels=labels, sky=sky)
    print_input_columns(names=[n for n in store.losses if not arms.ARMS[n].stored])
    print_mde(store=store, n_resamples=n_resamples)
    print(f"\n== planned readings, level {PLANNED_LEVEL:.2f}%, {n_resamples:,} resamples ==")
    readings = planned_readings(store=store, n_resamples=n_resamples)
    passed = checks(store=store, n_resamples=n_resamples)
    verdicts(readings=readings, passed=passed)
    exploratory(store=store)


def main() -> None:
    """Run a scorer command."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("score", "gate", "sky-index"))
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    args = parser.parse_args()
    arms.check_table()
    if args.command == "sky-index":
        build_sky_index()
    elif args.command == "gate":
        labels, sky = labels_and_sky(with_sky=False)
        print("gate:", "PASS" if gate(store=load_store(labels=labels, sky=sky)) else "FAIL")
    else:
        score(n_resamples=args.resamples)
    _ = timedelta


if __name__ == "__main__":
    main()

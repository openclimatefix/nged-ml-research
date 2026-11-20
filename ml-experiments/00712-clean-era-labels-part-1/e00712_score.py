r"""Experiment 712, scorer: every planned contrast, veto, gate, and check of the result page.

    source ml-experiments/00524*/env.sh     # puts every experiment folder on PYTHONPATH
    uv run --no-sync python ml-experiments/00712-*/e00712_score.py [--resamples N]
    uv run --no-sync python ml-experiments/00712-*/e00712_score.py --self-test

**What the scorer reads.** The real-row forecasts are the prediction sets `<arm>-<month>-s<seed>`
under the experiment's predictions folder, read with `studies.experiment_store.read_predictions`.
The five new arms (`det`, `llm`, `ecr`, `llm-ecr`, `ceiling`) are read from experiment 712's folder
and the eight stored arms (`none`, `oracle`, `placebo`, `all`, `soft`, `rule`, `stat`, `hmm`) from
experiment 709's folder. The injected-bank forecasts are the folders
`bank/<arm>_<fold>_s<seed>/passNN.parquet` (with `passNN_truth.parquet`) of the same two folders,
in the layout that `e00709_bank_score.py` reads. Two further bank arms are optional: `ecr-adv30`
(arm `ecr` re-scored with each register row visible 30 days early, contrast K6b) and `a0prev` (the
prevailing output of `none`, contrasts K8 and K9). A contrast whose arm is absent is skipped and
listed. The simulated register rows of the bank are read from `register/sim_register.parquet`
(columns `fold`, `pass`, `event`, `row_kind` of `true` or `decoy`, `decoy_at` of `donor` or
`recipient`); without the file, K5, K6, K6b, and the decoy tables are skipped.

**What the scorer computes.** Every unit and bootstrap follows experiment 709, so a new number sits
beside a stored number on the same rows.

- Real rows: the quantile score (CRPS) in points of effective capacity on the rows of experiment
  677, with the paired difference interval from a bootstrap that resamples whole origin months and
  a seed, as `studies.bootstrap` does. The month totals are resampled directly, which draws the
  same random numbers in the same order as `bootstrap_difference_at_level` and runs in seconds.
- Injected bank: one unit is one injected event, whose loss is the mean over seeds of the mean CRPS
  of the event's rows, and the bootstrap resamples whole passes (`fold/passNN`), as
  `e00709_bank_score.py` does.
- The interval level is 99.58% (1 - 0.05 / 12, twelve planned contrasts).

**Beside every contrast** the table carries the paired difference in the normalised mean absolute
error of the median and in the absolute error and the signed error of the mean forecast. The mean
forecast is the integral of the piecewise-linear quantile function through the 11 stored quantiles,
flat beyond the first and last. `median_minus_mean_by_site` lists the median minus the mean forecast
for each Primary and arm.

**Every output file has `INTERIM` in its name** and goes to
`<experiment 712 data folder>/scores/`.

**`--self-test` runs no new fit and makes no call.** It treats stored experiment 709 arms as the new
arms (`llm` reads `rule`, `det` reads `hmm`, `ecr` reads `stat`, `llm-ecr` reads `soft`, `ceiling`
reads `oracle`, `ecr-adv30` reads `all`, and `a0prev` reads `placebo`), invents a seeded register,
and checks that the engine reproduces experiment 709's stored tables:
`interim_phase_b_real_rows.csv` for the real rows (at 99.44% and 2,000 resamples, the settings
that wrote that file) and `interim_bank_rows.csv` for the bank (at 99.44% and 10,000 resamples, in
the same random-number order as `e00709_bank_score.py`).

Experiment 709's `e00578_common` (imported for the cleaned power and the site map) imports `torch`
through a chain of earlier experiments, so the scorer runs where `uv sync --group gpu` ran.
"""

import argparse
import pickle
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00709_features as features  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.bootstrap import BOOTSTRAP_SEED, bootstrap_difference_at_level
from studies.eval_protocol.metrics import LEVELS, quantile_score
from studies.experiment_store import read_predictions
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 712
SLUG: Final[str] = "clean-era-labels-part-1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
"""Private folder of experiment 712: predictions, bank forecasts, register, and scores."""
STORED_EXPERIMENT_ID: Final[int] = features.EXPERIMENT_ID
STORED_DIR: Final[Path] = features.DATA_DIR
SCORES_DIR: Final[Path] = DATA_DIR / "scores"
"""Every table is written here, with `INTERIM` in the file name."""
REGISTER_PATH: Final[Path] = DATA_DIR / "register" / "sim_register.parquet"

ROW_SET_ID: Final[str] = "experiment_677_monthly_refit_rows_valid_2025_11_to_2026_06"
E_ROW_SET_ID: Final[str] = "experiment_709_phase_e_monthly_refit_rows_valid_2025_04_to_2025_06"
"""Phase E arms are named `e-<arm>`; their sets carry this row set id, and they are scored apart."""
TEST_MONTHS: Final[tuple[str, ...]] = ("2025-11", "2026-01", "2026-04", "2026-06")
PHASE_E_MONTHS: Final[tuple[str, ...]] = ("2025-04", "2025-05", "2025-06")
SEEDS: Final[tuple[int, ...]] = (0, 1)
QCOLS: Final[tuple[str, ...]] = (
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
TAIL_INDEX: Final[Mapping[str, int]] = {"q01": 0, "q05": 1, "q95": 9, "q99": 10}
MEDIAN_INDEX: Final[int] = 5
D8_LABEL: Final[str] = "D8"
HALF_HOURS_PER_DAY: Final[int] = features.HALF_HOURS_PER_DAY

STORED_ARMS: Final[tuple[str, ...]] = (
    "none", "oracle", "placebo", "rule", "soft", "all", "stat", "hmm",
)  # fmt: skip
"""Experiment 709's arms, read from its folder."""
NEW_ARMS: Final[tuple[str, ...]] = ("det", "llm", "ecr", "llm-ecr", "ceiling")
OPTIONAL_BANK_ARMS: Final[tuple[str, ...]] = ("ecr-adv30", "a0prev")
SELF_TEST_ALIASES: Final[Mapping[str, str]] = {
    "llm": "rule",
    "det": "hmm",
    "ecr": "stat",
    "llm-ecr": "soft",
    "ceiling": "oracle",
    "ecr-adv30": "all",
    "a0prev": "placebo",
}
"""In `--self-test` each new arm reads the stored 709 arm named here."""

NUMBER_OF_CONTRASTS: Final[int] = 12
LEVEL_PERCENT: Final[float] = 100.0 * (1.0 - 0.05 / NUMBER_OF_CONTRASTS)
"""99.58%: each of the twelve planned contrasts has a two-sided interval at this level."""
DEFAULT_RESAMPLES: Final[int] = 20_000
"""Draws behind each interval. The tails of a 99.58% interval hold 0.21% each, about 42 draws."""
BANK_SEED: Final[int] = 0
"""Seed of the bank bootstrap, as `e00709_bank_score.py` (a fresh stream for each interval)."""

K1_THRESHOLD: Final[float] = -1.5
K2_POINT_LIMIT: Final[float] = 0.1
K2_UPPER_LIMIT: Final[float] = 0.2
K3_UPPER_LIMIT: Final[float] = 0.5
K4_POINT_LIMIT: Final[float] = 0.03
PLACEBO_GATE: Final[float] = 0.06
REPRODUCTION_TOLERANCE_MW: Final[float] = 1e-5
STABLE_DAYS: Final[int] = 90
POST_EVENT_DAYS: Final[int] = 7

BANK_DONOR_SLICE: Final[str] = "switch to end+7 d"
BANK_RECIPIENT_SLICE: Final[str] = "switch recipient to end+7 d"
BANK_ERA_SLICES: Final[tuple[str, ...]] = ("era 0-7 d", "era 7-28 d", "era 0-28 d")
BANK_METRICS: Final[tuple[str, ...]] = ("crps", "mae_median", "mean_ae", "mean_bias")
REAL_METRICS: Final[tuple[str, ...]] = (
    "crps", "mae_median", "pin_q01", "pin_q05", "pin_q95", "pin_q99",
    "cov_q01", "cov_q05", "cov_q95", "cov_q99", "mean_ae", "mean_bias", "median_minus_mean",
)  # fmt: skip
"""Per-row metrics of the real rows, in points of capacity (coverage is a 0-or-1 share)."""

ArmKindType = Literal["real", "bank"]
"""Which table a contrast reads."""
VerdictType = Literal[
    "supported",
    "small gain",
    "partly supported",
    "not supported",
    "not supported for adoption",
    "inconclusive pending an audit",
]
"""The decision rule's outcomes."""


class ArmSource(NamedTuple):
    """Where one arm's files live."""

    experiment_id: int
    stored_name: str


def source_for(*, arm: str, self_test: bool) -> ArmSource:
    """Return the experiment and the stored name that hold an arm's files.

    Args:
        arm: The arm's name as the result page uses it.
        self_test: Whether the new arms read the stored arms of `SELF_TEST_ALIASES`.

    Returns:
        Experiment 709 for a stored arm (and for every arm in the self-test), experiment 712
        otherwise.
    """
    if self_test:
        return ArmSource(STORED_EXPERIMENT_ID, SELF_TEST_ALIASES.get(arm, arm))
    phase_e_base = arm.removeprefix("e-")
    if phase_e_base in STORED_ARMS:
        return ArmSource(STORED_EXPERIMENT_ID, arm)
    return ArmSource(EXPERIMENT_ID, arm)


def data_dir_of(*, experiment_id: int) -> Path:
    """Return the private data folder of experiment 709 or 712."""
    return STORED_DIR if experiment_id == STORED_EXPERIMENT_ID else DATA_DIR


def prediction_name(*, source: ArmSource, month: str, seed: int) -> str:
    """Return the name of a real-row prediction set."""
    return f"{source.stored_name}-{month}-s{seed}"


def bank_folder(*, source: ArmSource, month: str, seed: int) -> Path:
    """Return the bank folder of one arm, fold, and seed (`-` becomes `_`, as in experiment 709)."""
    name = f"{source.stored_name.replace('-', '_')}_{month}_s{seed}"
    return data_dir_of(experiment_id=source.experiment_id) / "bank" / name


# --------------------------------------------------------------------------------------------
# Real rows
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RealContext:
    """Everything a real-row score needs besides the forecasts."""

    truth: np.ndarray
    capacity_mw: np.ndarray
    to_site: Mapping[int, int]
    event_windows: Sequence[tuple[int, int, int]]
    alarm_days: Sequence[tuple[int, int]]


def series_to_site() -> dict[int, int]:
    """Return the Primary index of each `time_series_id` (the site map's `primary` column)."""
    sites = k578.sites()
    assert tuple(sites["label"].to_list()) == features.PRIMARIES, "site map and panel differ"
    assert sites["primary"].to_list() == list(range(len(features.PRIMARIES))), "primary != index"
    return {int(t): int(p) for t, p in zip(sites["time_series_id"], sites["primary"], strict=True)}


def _epoch_us(*, moment: datetime) -> int:
    """Return a datetime as microseconds since the Unix epoch."""
    return int(moment.timestamp() * 1_000_000)


def load_event_windows() -> list[tuple[int, int, int]]:
    """Return the labelled events as (Primary index, onset, end) in epoch microseconds.

    The events are the verified events of experiment 580 and the labelled set of experiment 41, in
    experiment 709's `known_events.parquet`. An event is listed at both of its Primaries.

    Returns:
        One tuple per event and Primary.
    """
    events = pl.read_parquet(STORED_DIR / "known_events.parquet")
    index = {label: i for i, label in enumerate(features.PRIMARIES)}
    windows: list[tuple[int, int, int]] = []
    for row in events.iter_rows(named=True):
        onset = _epoch_us(moment=row["onset_utc"])
        end = _epoch_us(moment=row["end_utc"])
        windows.extend(
            (index[label], onset, end)
            for label in (row["site"], row["other_site"])
            if label in index
        )
    return windows


def load_alarm_days() -> list[tuple[int, int]]:
    """Return every alarm of either detector as (Primary index, day since the grid start).

    The detectors are experiment 709's statistics detector (`stat`) and change-point detector
    (`all`).

    Returns:
        One tuple per alarm.
    """
    alarms: list[tuple[int, int]] = []
    for name in ("stat_episodes", "all_episodes"):
        episodes = pl.read_parquet(STORED_DIR / "signals" / f"{name}.parquet")
        alarms.extend(
            (int(site), int(alarm))
            for site, alarm in zip(episodes["site"], episodes["alarm"], strict=True)
        )
    return alarms


def load_real_context() -> RealContext:
    """Return the cleaned power, capacities, site map, labelled events, and detector alarms."""
    panel = features.load_panel()
    sites = k578.sites()
    assert np.allclose(sites["capacity_mw"].to_numpy(), panel.capacity_mw, rtol=1e-4), (
        "capacities differ"
    )
    truth = k578.clean()[:, : panel.power.shape[1]].astype(np.float64)
    return RealContext(
        truth=truth,
        capacity_mw=np.asarray(panel.capacity_mw, dtype=np.float64),
        to_site=series_to_site(),
        event_windows=load_event_windows(),
        alarm_days=load_alarm_days(),
    )


def mean_from_quantiles(*, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's mean forecast: the integral of the piecewise-linear quantile function.

    Args:
        quantiles: The forecast quantiles at `LEVELS`, shape (rows, 11).

    Returns:
        The integral over levels 0 to 1, with the first and last quantile held flat beyond the
        first and last level, shape (rows,).
    """
    middle = ((quantiles[:, 1:] + quantiles[:, :-1]) / 2.0 * np.diff(LEVELS)).sum(axis=1)
    return LEVELS[0] * quantiles[:, 0] + middle + (1.0 - LEVELS[-1]) * quantiles[:, -1]


def forecast_metrics(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return the per-row metrics of finite rows, in points of capacity.

    Args:
        quantiles: Forecast quantiles in points of capacity, shape (rows, 11).
        actual: Observed values in points of capacity, shape (rows,).

    Returns:
        A dictionary of arrays named as `REAL_METRICS`, bar `median_minus_mean`'s sign convention:
        `mean_bias` is the mean forecast minus the actual, and `median_minus_mean` is the median
        forecast minus the mean forecast.
    """
    mean_forecast = mean_from_quantiles(quantiles=quantiles)
    out: dict[str, np.ndarray] = {
        "crps": quantile_score(actual=actual, quantiles=quantiles),
        "mae_median": np.abs(quantiles[:, MEDIAN_INDEX] - actual),
        "mean_ae": np.abs(mean_forecast - actual),
        "mean_bias": mean_forecast - actual,
        "median_minus_mean": quantiles[:, MEDIAN_INDEX] - mean_forecast,
    }
    for name, position in TAIL_INDEX.items():
        error = actual - quantiles[:, position]
        level = float(LEVELS[position])
        out[f"pin_{name}"] = np.maximum(level * error, (level - 1.0) * error)
        out[f"cov_{name}"] = (error <= 0.0).astype(np.float64)
    return out


def _flag_rows(*, primary: np.ndarray, origin_us: np.ndarray, context: RealContext) -> pl.DataFrame:
    """Return the event and stability flags of each row, from the row's Primary and origin.

    Args:
        primary: The Primary index of each row.
        origin_us: The forecast origin in epoch microseconds.
        context: The labelled events and detector alarms.

    Returns:
        Boolean columns `in_event` (origin inside a labelled event), `post_event` (origin up to 7
        days after an event ends), and `stable` (origin outside and more than 7 days after every
        labelled event, and at least 90 days after every alarm of either detector).
    """
    day_us = 86_400 * 1_000_000
    start_us = _epoch_us(moment=features.GRID_START)
    origin_day = (origin_us - start_us) // day_us
    in_event = np.zeros(len(primary), dtype=bool)
    post_event = np.zeros(len(primary), dtype=bool)
    for site, onset, end in context.event_windows:
        on_site = primary == site
        in_event |= on_site & (origin_us >= onset) & (origin_us <= end)
        post_event |= on_site & (origin_us > end) & (origin_us <= end + POST_EVENT_DAYS * day_us)
    after_alarm = np.zeros(len(primary), dtype=bool)
    for site, alarm in context.alarm_days:
        gap = origin_day - alarm
        after_alarm |= (primary == site) & (gap >= 0) & (gap < STABLE_DAYS)
    return pl.DataFrame(
        {
            "in_event": in_event,
            "post_event": post_event,
            "stable": ~(in_event | post_event | after_alarm),
        }
    )


def row_table(*, forecast: pl.DataFrame, context: RealContext) -> pl.DataFrame:
    """Return the per-row metrics of one forecast, on the rows with a finite reading.

    Args:
        forecast: A prediction set's frame (quantiles in MW).
        context: The cleaned power and the capacities.

    Returns:
        The columns `time_series_id`, `power_fcst_init_time`, `valid_time`, `primary`, `month`,
        the flags of `_flag_rows`, `key_hash`, and `REAL_METRICS`.
    """
    site = np.array([context.to_site[int(t)] for t in forecast["time_series_id"].to_list()])
    start = np.datetime64(features.GRID_START.replace(tzinfo=None), "us")
    valid = forecast["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    index = ((valid - start) // np.timedelta64(1800, "s")).astype(np.int64)
    inside = (index >= 0) & (index < context.truth.shape[1])
    index = np.clip(index, 0, context.truth.shape[1] - 1)
    actual = np.where(inside, context.truth[site, index], np.nan) * 100.0
    capacity = context.capacity_mw[site]
    quantiles = (
        forecast.select(list(QCOLS)).to_numpy().astype(np.float64) / capacity[:, None] * 100.0
    )
    finite = np.isfinite(actual)
    metrics = forecast_metrics(quantiles=quantiles[finite], actual=actual[finite])
    keys = (
        forecast.select("time_series_id", "power_fcst_init_time", "valid_time")
        .filter(pl.Series(finite))
        .with_columns(
            primary=pl.Series(site[finite]),
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            key_hash=pl.struct("time_series_id", "power_fcst_init_time", "valid_time").hash()
            % 2**31,
        )
    )
    origin_us = keys["power_fcst_init_time"].dt.epoch("us").to_numpy()
    flags = _flag_rows(primary=site[finite], origin_us=origin_us, context=context)
    return pl.concat(
        [keys, flags, pl.DataFrame(dict(metrics))],
        how="horizontal",
    )


def real_scopes(*, d8: int) -> dict[str, pl.Expr]:
    """Return the real-row scopes: the screen's three and the four load-shift scenarios.

    Args:
        d8: The Primary index of D8.

    Returns:
        Scope names mapped to boolean expressions over `primary`, `in_event`, `post_event`, and
        `stable`.
    """
    return {
        "all rows": pl.lit(value=True),
        "D8": pl.col("primary") == d8,
        "other Primaries": pl.col("primary") != d8,
        "scenario: rows outside events and 7+ days after one": ~(
            pl.col("in_event") | pl.col("post_event")
        ),
        "scenario: rows inside events": pl.col("in_event"),
        "scenario: the 7 days after an event ends": pl.col("post_event"),
        "stable rows (90+ days after every alarm)": pl.col("stable"),
    }


def arm_totals(*, rows: pl.DataFrame, scopes: Mapping[str, pl.Expr]) -> pl.DataFrame:
    """Return an arm's sums by scope, seed, and month.

    Args:
        rows: The arm's per-row table over every fold and seed (column `seed`).
        scopes: The scopes of `real_scopes`.

    Returns:
        The columns `scope`, `seed`, `month`, `n`, `key_hash` (the sum of the row-key hashes, which
        pairs arms row for row), and `sum_<metric>` for each of `REAL_METRICS`.
    """
    frames = [
        rows.filter(expr)
        .group_by("seed", "month")
        .agg(
            pl.len().alias("n"),
            pl.col("key_hash").sum(),
            *[pl.col(metric).sum().alias(f"sum_{metric}") for metric in REAL_METRICS],
        )
        .with_columns(scope=pl.lit(name))
        for name, expr in scopes.items()
    ]
    return pl.concat(frames).sort("scope", "seed", "month")


def read_real_arm(
    *,
    arm: str,
    months: Sequence[str],
    seeds: Sequence[int],
    context: RealContext,
    self_test: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read one arm's real-row forecasts and return its sums and its median-minus-mean by site.

    Args:
        arm: The arm's name.
        months: Fold months.
        seeds: Seeds.
        context: The cleaned power, capacities, events, and alarms.
        self_test: Whether the new arms read stored arms.

    Returns:
        The `arm_totals` frame and a frame of the mean median-minus-mean forecast, in points of
        capacity, by Primary label.
    """
    source = source_for(arm=arm, self_test=self_test)
    row_set = E_ROW_SET_ID if arm.startswith("e-") else ROW_SET_ID
    parts = []
    for month in months:
        for seed in seeds:
            forecast = read_predictions(
                experiment_id=source.experiment_id,
                name=prediction_name(source=source, month=month, seed=seed),
                row_set_id=row_set,
            )
            parts.append(
                row_table(forecast=forecast, context=context).with_columns(seed=pl.lit(seed))
            )
    rows = pl.concat(parts)
    totals = arm_totals(rows=rows, scopes=real_scopes(d8=features.PRIMARIES.index(D8_LABEL)))
    by_site = (
        rows.group_by("primary")
        .agg(pl.col("median_minus_mean").mean())
        .sort("primary")
        .with_columns(
            arm=pl.lit(arm),
            primary_label=pl.col("primary").map_elements(
                lambda i: features.PRIMARIES[i], return_dtype=pl.String
            ),
        )
    )
    return totals, by_site


def missing_real_sets(
    *, arms: Iterable[str], months: Sequence[str], seeds: Sequence[int], self_test: bool
) -> dict[str, list[str]]:
    """Return, for each arm with an absent prediction set, the names of the absent sets.

    Args:
        arms: Arms wanted.
        months: Fold months.
        seeds: Seeds.
        self_test: Whether the new arms read stored arms.

    Returns:
        A mapping from arm to absent set names, holding only arms with a set missing.
    """
    missing: dict[str, list[str]] = {}
    for arm in arms:
        source = source_for(arm=arm, self_test=self_test)
        folder = data_dir_of(experiment_id=source.experiment_id) / "predictions"
        names = [
            prediction_name(source=source, month=month, seed=seed)
            for month in months
            for seed in seeds
        ]
        absent = [name for name in names if not (folder / name).exists()]
        if absent:
            missing[arm] = absent
    return missing


def _bootstrap_percentiles(*, means: np.ndarray, level: float) -> tuple[float, float]:
    """Return the two tail percentiles of the resampled means."""
    tail = (100.0 - level) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def bootstrap_month_totals_at_level(
    *, month_sums: np.ndarray, month_counts: np.ndarray, level: float, n_resamples: int
) -> tuple[float, float]:
    """Bound a paired mean from each month's totals, drawing as `studies.bootstrap` does.

    Each resample draws one seed, then draws the months with replacement, in the order and from the
    stream of `studies.bootstrap._resample_bounds`. The bounds equal
    `bootstrap_difference_at_level` up to floating-point summation order, at a fraction of the time.

    Args:
        month_sums: Each seed's sum of the paired differences in each month, (seeds, months).
        month_counts: The rows in each month, (months,).
        level: Coverage in percent, above 1 and below 100.
        n_resamples: Number of resamples.

    Returns:
        The lower and upper bounds of the resampled mean difference.
    """
    if not 1.0 < level < 100.0:
        msg = f"level is a coverage in percent, above 1 and below 100: {level}"
        raise ValueError(msg)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    n_seeds, n_months = month_sums.shape
    resampled = np.empty(n_resamples)
    for resample in range(n_resamples):
        seed_index = generator.integers(0, n_seeds)
        drawn = generator.integers(0, n_months, size=n_months)
        resampled[resample] = month_sums[seed_index, drawn].sum() / month_counts[drawn].sum()
    return _bootstrap_percentiles(means=resampled, level=level)


@dataclass(frozen=True)
class ContrastResult:
    """One paired difference (treatment minus reference) with its interval and companions."""

    contrast: str
    table: str
    treatment: str
    reference: str
    rows: str
    subset: str
    n_units: int
    n_clusters: int
    treatment_mean: float
    reference_mean: float
    diff: float
    lower: float
    upper: float
    level: float
    n_resamples: int
    mae_median_diff: float
    mean_ae_diff: float
    mean_bias_diff: float


def real_contrast(
    *,
    totals: Mapping[str, pl.DataFrame],
    treatment: str,
    reference: str,
    scope: str,
    contrast: str,
    level: float,
    n_resamples: int,
) -> ContrastResult:
    """Return the paired real-row difference in CRPS with its bootstrap interval.

    Args:
        totals: Each arm's `arm_totals`.
        treatment: The arm whose score is compared.
        reference: The arm it is compared against.
        scope: One of the scopes of `real_scopes`.
        contrast: The contrast's name.
        level: Interval coverage in percent.
        n_resamples: Number of resamples.

    Returns:
        The result. The rows of the two arms must be identical in every seed and month.
    """
    columns = ["seed", "month", "n", "key_hash", *(f"sum_{m}" for m in REAL_METRICS)]
    treat = totals[treatment].filter(pl.col("scope") == scope).select(columns)
    ref = totals[reference].filter(pl.col("scope") == scope).select(columns)
    keys = ["seed", "month"]
    if not (treat.select(*keys, "n", "key_hash").equals(ref.select(*keys, "n", "key_hash"))):
        msg = f"{treatment!r} and {reference!r} do not hold the same rows in scope {scope!r}"
        raise ValueError(msg)
    treat = treat.sort(keys)
    ref = ref.sort(keys)
    n_seeds = treat["seed"].n_unique()
    n_months = treat["month"].n_unique()
    if treat.height != n_seeds * n_months:
        msg = f"{treatment!r} lacks a seed-month cell in scope {scope!r}"
        raise ValueError(msg)

    def grid(frame: pl.DataFrame, metric: str) -> np.ndarray:
        return frame[f"sum_{metric}"].to_numpy().reshape(n_seeds, n_months)

    counts = treat["n"].to_numpy().reshape(n_seeds, n_months)[0]
    total_rows = float(counts.sum() * n_seeds)
    difference = grid(treat, "crps") - grid(ref, "crps")
    low, high = bootstrap_month_totals_at_level(
        month_sums=difference, month_counts=counts, level=level, n_resamples=n_resamples
    )

    def mean_of(frame: pl.DataFrame, metric: str) -> float:
        return float(grid(frame, metric).sum() / total_rows)

    return ContrastResult(
        contrast=contrast,
        table="real",
        treatment=treatment,
        reference=reference,
        rows=scope,
        subset="",
        n_units=int(total_rows),
        n_clusters=n_months,
        treatment_mean=mean_of(treat, "crps"),
        reference_mean=mean_of(ref, "crps"),
        diff=float(difference.sum() / total_rows),
        lower=low,
        upper=high,
        level=level,
        n_resamples=n_resamples,
        mae_median_diff=mean_of(treat, "mae_median") - mean_of(ref, "mae_median"),
        mean_ae_diff=mean_of(treat, "mean_ae") - mean_of(ref, "mean_ae"),
        mean_bias_diff=mean_of(treat, "mean_bias") - mean_of(ref, "mean_bias"),
    )


def absolute_metrics(*, totals: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Return each arm's mean metrics by scope: CRPS, MAE of the median, tails, and mean forecast.

    Args:
        totals: Each arm's `arm_totals`.

    Returns:
        One row per arm and scope, with `n` and the mean of each of `REAL_METRICS`.
    """
    frames = [
        frame.group_by("scope")
        .agg(
            pl.col("n").sum(),
            *[(pl.col(f"sum_{m}").sum() / pl.col("n").sum()).alias(m) for m in REAL_METRICS],
        )
        .with_columns(arm=pl.lit(arm))
        for arm, frame in totals.items()
    ]
    return pl.concat(frames).sort("scope", "arm")


# --------------------------------------------------------------------------------------------
# Injected bank
# --------------------------------------------------------------------------------------------


class _StandIn:
    """A stand-in for a class of an earlier experiment that is not on the import path."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickle a cached world even where the classes of earlier experiments cannot be imported."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ImportError, AttributeError:
            return _StandIn


def infer_recipients(*, world: Mapping[str, Any]) -> dict[int, int | None]:
    """Return the metered recipient of each injected switching event of a cached world.

    The bank's truth table names the donor and no recipient. The recipient is the other Primary
    whose injected change is positive over the event's window and whose first and last changed
    half-hours best match the event's onset and end. A hidden-mirror event has no recipient.

    Args:
        world: A cached world of experiment 709 (`worlds/bank_<fold>_p<k>.pkl`).

    Returns:
        The recipient's Primary index, or None, for each event index of the truth table.
    """
    table = world["truth_table"]
    delta = np.asarray(world["delta"], dtype=np.float64) - np.asarray(
        world["era_delta"], dtype=np.float64
    )
    recipients: dict[int, int | None] = {}
    for index, event in enumerate(table.iter_rows(named=True)):
        recipients[index] = None
        if event["truth"] != "switch" or event["hidden_mirror"]:
            continue
        start = int(event["onset_day"]) * HALF_HOURS_PER_DAY
        stop = (int(event["end_day"]) + 1) * HALF_HOURS_PER_DAY
        pad = 5 * HALF_HOURS_PER_DAY
        best: tuple[int, int] | None = None
        for site in range(delta.shape[0]):
            if site == int(event["host"]) or delta[site, start:stop].mean() <= 0.0:
                continue
            window = np.flatnonzero(delta[site, max(start - pad, 0) : stop + pad] > 0.0)
            if len(window) == 0:
                continue
            first = window[0] + max(start - pad, 0)
            last = window[-1] + 1 + max(start - pad, 0)
            mismatch = abs(int(first) - start) + abs(int(last) - stop)
            if best is None or mismatch < best[0]:
                best = (mismatch, site)
        recipients[index] = None if best is None else best[1]
    return recipients


def load_recipients(
    *, folds: Sequence[str], passes: Sequence[str]
) -> dict[tuple[str, str], dict[int, int | None]]:
    """Return the recipients of every switching event, by (fold, pass stem).

    Args:
        folds: Fold months.
        passes: Pass file stems such as `pass00`.

    Returns:
        A mapping from (fold, pass stem) to `infer_recipients`' output, empty where a world is
        missing.
    """
    out: dict[tuple[str, str], dict[int, int | None]] = {}
    for fold in folds:
        for stem in passes:
            path = STORED_DIR / "worlds" / f"bank_{fold}_p{int(stem.removeprefix('pass'))}.pkl"
            if not path.exists():
                continue
            with path.open("rb") as handle:
                world = _TolerantUnpickler(handle).load()
            out[(fold, stem)] = infer_recipients(world=world)
    return out


def bank_event_losses(
    *,
    source: ArmSource,
    month: str,
    seed: int,
    ids: np.ndarray,
    recipients: Mapping[tuple[str, str], Mapping[int, int | None]],
) -> pl.DataFrame:
    """Return one row per (pass, event, slice) with the mean metrics of the arm's rows in the slice.

    Args:
        source: Where the arm's bank folders live.
        month: Fold month.
        seed: Seed.
        ids: The `time_series_id` of each Primary in panel order; an event's `host` indexes it.
        recipients: The recipients of `load_recipients`.

    Returns:
        Columns `fold`, `pass`, `event`, `slice`, `seed`, `kind`, `truth`, `n_rows`, and
        `BANK_METRICS`.
    """
    rows: list[dict[str, Any]] = []
    folder = bank_folder(source=source, month=month, seed=seed)
    start_day = features.GRID_START.replace(tzinfo=None)
    for forecast_path in sorted(folder.glob("pass[0-9][0-9].parquet")):
        events = pl.read_parquet(forecast_path.with_name(forecast_path.stem + "_truth.parquet"))
        forecast = pl.read_parquet(forecast_path)
        origin_day = (
            (
                forecast["power_fcst_init_time"].dt.replace_time_zone(None)
                - pl.Series([start_day] * forecast.height)
            )
            .dt.total_days()
            .to_numpy()
        )
        site = forecast["time_series_id"].to_numpy()
        quantiles = forecast.select(list(QCOLS)).to_numpy().astype(np.float64) * 100.0
        for index, event in enumerate(events.iter_rows(named=True)):
            onset = event["onset_day"]
            if event["truth"] == "era":
                windows = {
                    "era 0-7 d": (onset, onset + 7),
                    "era 7-28 d": (onset + 7, onset + 28),
                    "era 0-28 d": (onset, onset + 28),
                }
                truth_column = "truth_share"
                hosts = {name: int(event["host"]) for name in windows}
            else:
                windows = {BANK_DONOR_SLICE: (onset, event["end_day"] + 7)}
                hosts = {BANK_DONOR_SLICE: int(event["host"])}
                truth_column = "clean_share"
                recipient = recipients.get((month, forecast_path.stem), {}).get(index)
                if recipient is not None:
                    windows[BANK_RECIPIENT_SLICE] = (onset, event["end_day"] + 7)
                    hosts[BANK_RECIPIENT_SLICE] = recipient
            truth = forecast[truth_column].to_numpy().astype(np.float64)
            for name, (low, high) in windows.items():
                selected = (
                    (site == ids[hosts[name]])
                    & (origin_day >= low)
                    & (origin_day < high)
                    & np.isfinite(truth)
                )
                if not selected.any():
                    continue
                metrics = forecast_metrics(
                    quantiles=quantiles[selected], actual=truth[selected] * 100.0
                )
                rows.append(
                    {
                        "fold": month,
                        "pass": forecast_path.stem,
                        "event": index,
                        "slice": name,
                        "seed": seed,
                        "kind": event["kind"],
                        "truth": event["truth"],
                        "n_rows": int(selected.sum()),
                        **{m: float(np.mean(metrics[m])) for m in BANK_METRICS},
                    }
                )
    assert rows, f"no bank rows for {folder}"
    return pl.DataFrame(rows)


def read_bank_arm(
    *,
    arm: str,
    folds: Sequence[str],
    seeds: Sequence[int],
    ids: np.ndarray,
    recipients: Mapping[tuple[str, str], Mapping[int, int | None]],
    self_test: bool,
) -> pl.DataFrame:
    """Return one arm's event losses averaged over the seeds, one row per (event, slice).

    Args:
        arm: The arm's name.
        folds: Fold months.
        seeds: Seeds.
        ids: The `time_series_id` of each Primary in panel order.
        recipients: The recipients of `load_recipients`.
        self_test: Whether the new arms read stored arms.

    Returns:
        Columns `fold`, `pass`, `event`, `slice`, `kind`, `truth`, `n_rows`, and `BANK_METRICS`.
    """
    source = source_for(arm=arm, self_test=self_test)
    frames = [
        bank_event_losses(source=source, month=month, seed=seed, ids=ids, recipients=recipients)
        for month in folds
        for seed in seeds
    ]
    losses = (
        pl.concat(frames)
        .group_by("fold", "pass", "event", "slice")
        .agg(
            pl.col("kind").first(),
            pl.col("truth").first(),
            pl.col("n_rows").first(),
            *[pl.col(m).mean() for m in BANK_METRICS],
            pl.col("seed").n_unique().alias("n_seeds"),
        )
    )
    assert losses["n_seeds"].min() == len(seeds), f"an event of {arm!r} is missing a seed"
    return losses.drop("n_seeds")


def missing_bank_arms(
    *, arms: Iterable[str], folds: Sequence[str], seeds: Sequence[int], self_test: bool
) -> list[str]:
    """Return the arms with a missing bank folder in any fold or seed."""
    return [
        arm
        for arm in arms
        if not all(
            bank_folder(source=source_for(arm=arm, self_test=self_test), month=m, seed=s).exists()
            for m in folds
            for s in seeds
        )
    ]


def bank_bootstrap(
    *,
    diff: np.ndarray,
    clusters: np.ndarray,
    level: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bound the mean paired difference by resampling whole passes, as `e00709_bank_score.py`.

    Args:
        diff: The paired difference of each event.
        clusters: The pass (`fold/passNN`) of each event.
        level: Coverage in percent.
        n_resamples: Number of resamples.
        rng: The random generator, advanced by this call.

    Returns:
        The lower and upper percentile bounds.
    """
    names, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=diff, minlength=len(names))
    counts = np.bincount(inverse, minlength=len(names))
    draws = rng.integers(0, len(names), size=(n_resamples, len(names)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return _bootstrap_percentiles(means=means, level=level)


def register_subset(*, register: pl.DataFrame | None, subset: str) -> pl.DataFrame | None:
    """Return the (fold, pass, event) keys of a register subset.

    Args:
        register: The simulated register, or None.
        subset: `register_true` (eras with a true row), `register_decoy` (switching events with a
            decoy row), `decoy_donor`, or `decoy_recipient` (decoys by the Primary they sit at).

    Returns:
        The keys, or None when no register is available.
    """
    if register is None:
        return None
    if subset == "register_true":
        chosen = register.filter(pl.col("row_kind") == "true")
    elif subset == "register_decoy":
        chosen = register.filter(pl.col("row_kind") == "decoy")
    elif subset == "decoy_donor":
        chosen = register.filter((pl.col("row_kind") == "decoy") & (pl.col("decoy_at") == "donor"))
    elif subset == "decoy_recipient":
        chosen = register.filter(
            (pl.col("row_kind") == "decoy") & (pl.col("decoy_at") == "recipient")
        )
    else:
        msg = f"unknown register subset {subset!r}"
        raise ValueError(msg)
    return chosen.select("fold", "pass", "event")


def bank_contrast(
    *,
    units: Mapping[str, pl.DataFrame],
    treatment: str,
    reference: str,
    slice_name: str,
    contrast: str,
    level: float,
    n_resamples: int,
    rng: np.random.Generator,
    keys: pl.DataFrame | None = None,
    subset: str = "",
) -> ContrastResult:
    """Return the paired bank difference in CRPS with its pass-resampling interval.

    Args:
        units: Each arm's `read_bank_arm` frame.
        treatment: The arm whose score is compared.
        reference: The arm it is compared against.
        slice_name: One of the bank slices.
        contrast: The contrast's name.
        level: Interval coverage in percent.
        n_resamples: Number of resamples.
        rng: The random generator, advanced by this call.
        keys: If given, only the events with these (fold, pass, event) keys.
        subset: The subset's name, recorded in the result.

    Returns:
        The result, over the events that both arms hold.
    """
    index = ["fold", "pass", "event"]
    columns = [*index, *BANK_METRICS]
    treat = units[treatment].filter(pl.col("slice") == slice_name).select(columns)
    ref = units[reference].filter(pl.col("slice") == slice_name).select(columns)
    if keys is not None:
        treat = treat.join(keys, on=index, how="semi")
        ref = ref.join(keys, on=index, how="semi")
    wide = ref.join(treat, on=index, how="inner", suffix="_t").sort(index)
    if wide.is_empty():
        msg = f"no events for {contrast} in slice {slice_name!r}"
        raise ValueError(msg)
    diff = (wide["crps_t"] - wide["crps"]).to_numpy()
    clusters = (wide["fold"] + "/" + wide["pass"]).to_numpy()
    low, high = bank_bootstrap(
        diff=diff, clusters=clusters, level=level, n_resamples=n_resamples, rng=rng
    )
    companions = {m: float((wide[f"{m}_t"] - wide[m]).mean()) for m in BANK_METRICS}  # ty: ignore[invalid-argument-type]
    return ContrastResult(
        contrast=contrast,
        table="bank",
        treatment=treatment,
        reference=reference,
        rows=slice_name,
        subset=subset,
        n_units=wide.height,
        n_clusters=len(np.unique(clusters)),
        treatment_mean=float(wide["crps_t"].mean()),  # ty: ignore[invalid-argument-type]
        reference_mean=float(wide["crps"].mean()),  # ty: ignore[invalid-argument-type]
        diff=float(diff.mean()),
        lower=low,
        upper=high,
        level=level,
        n_resamples=n_resamples,
        mae_median_diff=companions["mae_median"],
        mean_ae_diff=companions["mean_ae"],
        mean_bias_diff=companions["mean_bias"],
    )


# --------------------------------------------------------------------------------------------
# Contrasts, vetoes, gates, and the decision rule
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContrastSpec:
    """One planned contrast (or a companion of one)."""

    name: str
    treatment: str
    reference: str
    table: ArmKindType
    rows: str
    subset: str = ""


def planned_contrasts() -> list[ContrastSpec]:
    """Return the twelve planned contrasts K1 to K9, then their companion tables.

    Returns:
        The specifications. The twelve planned contrasts come first; the companions follow: the
        recipient reading of each K2, the decoy subsets, and K4 for every arm.
    """
    era = "era 7-28 d"
    donor = BANK_DONOR_SLICE
    recipient = BANK_RECIPIENT_SLICE
    planned = [
        ContrastSpec("K1", "llm", "none", "bank", era),
        ContrastSpec("K2", "llm", "none", "bank", donor),
        ContrastSpec("K2-ecr", "ecr", "none", "bank", donor),
        ContrastSpec("K2-llm-ecr", "llm-ecr", "none", "bank", donor),
        ContrastSpec("K3", "llm", "soft", "bank", era),
        ContrastSpec("K4", "llm", "none", "real", "all rows"),
        ContrastSpec("K5", "llm-ecr", "llm", "bank", "era 0-28 d", "register_true"),
        ContrastSpec("K6", "ecr", "none", "bank", era, "register_true"),
        ContrastSpec("K6b", "ecr-adv30", "none", "bank", "era 0-28 d", "register_true"),
        ContrastSpec("K7", "llm", "det", "bank", donor),
        ContrastSpec("K8", "llm", "a0prev", "bank", era),
        ContrastSpec("K9", "llm", "a0prev", "bank", donor),
    ]
    assert len(planned) == NUMBER_OF_CONTRASTS
    companions = [ContrastSpec("K4-placebo", "placebo", "none", "real", "all rows")]
    for name, arm in (("K2", "llm"), ("K2-ecr", "ecr"), ("K2-llm-ecr", "llm-ecr")):
        companions.append(ContrastSpec(f"{name} recipient", arm, "none", "bank", recipient))
    for name, arm in (("K2-ecr", "ecr"), ("K2-llm-ecr", "llm-ecr")):
        for subset in ("register_decoy", "decoy_donor", "decoy_recipient"):
            companions.append(ContrastSpec(f"{name} {subset}", arm, "none", "bank", donor, subset))
            companions.append(
                ContrastSpec(f"{name} {subset} recipient", arm, "none", "bank", recipient, subset)
            )
    companions.extend(
        ContrastSpec(f"K4 {arm}", arm, "none", "real", "all rows")
        for arm in (*NEW_ARMS, "soft", "rule")
        if arm != "llm"
    )
    return planned + companions


def veto_k2(*, result: ContrastResult) -> bool:
    """Return whether a donor contrast passes K2 (point at most +0.1, upper end at most +0.2)."""
    return result.diff <= K2_POINT_LIMIT and result.upper <= K2_UPPER_LIMIT


def veto_k4(*, result: ContrastResult, placebo: ContrastResult) -> bool:
    """Return whether an all-rows contrast passes K4.

    Args:
        result: The arm minus `none` on all real rows.
        placebo: The stored placebo minus `none` on the same rows at the same level.

    Returns:
        True when the point estimate is at most +0.03 and the upper end is no higher than the
        placebo's.
    """
    return result.diff <= K4_POINT_LIMIT and result.upper <= placebo.upper


def decide(*, results: Mapping[str, ContrastResult], gates_passed: bool) -> tuple[VerdictType, str]:
    """Apply the decision rule of the result page.

    Args:
        results: The results by contrast name; K1, K2, K3, and K4 (with K4-placebo) are needed.
        gates_passed: Whether the `none` reproduction gate and the placebo gate passed.

    Returns:
        The verdict and a sentence stating the reason.
    """
    if not gates_passed:
        return "inconclusive pending an audit", "the reproduction gate or the placebo gate failed"
    k1, k2, k3 = results["K1"], results["K2"], results["K3"]
    k4_ok = veto_k4(result=results["K4"], placebo=results["K4-placebo"])
    k2_ok = veto_k2(result=k2)
    if not (k2_ok and k4_ok):
        return "not supported for adoption", f"K2 passes: {k2_ok}; K4 passes: {k4_ok}"
    if k1.upper >= 0.0:
        return "not supported", "K1's interval does not lie wholly below zero"
    if k1.diff > K1_THRESHOLD:
        return "small gain", f"K1 {k1.diff:+.3f} is above {K1_THRESHOLD}"
    if k3.upper <= K3_UPPER_LIMIT:
        return "supported", f"K1 {k1.diff:+.3f}, K3 upper {k3.upper:+.3f}"
    return "partly supported", f"K1 {k1.diff:+.3f} but K3 upper {k3.upper:+.3f} exceeds +0.5"


VERDICT_GATE_NAMES: Final[frozenset[str]] = frozenset(
    {"none reproduction", "placebo within 0.06 of none"}
)
"""The gates whose failure makes the verdict `inconclusive pending an audit`."""


class Gate(NamedTuple):
    """One gate or check: its name, whether it passed, and the figures behind it."""

    name: str
    passed: bool | None
    detail: str


def reproduction_gate(*, new_name: str, stored_name: str) -> Gate:
    """Check that a new run reproduces the stored `none` median to within 1e-5 MW.

    Args:
        new_name: Prediction set of the new script's `none` run (in experiment 712's folder).
        stored_name: The stored set (in experiment 709's folder).

    Returns:
        The gate: passed when the median agrees on every row; None when the new set is absent.
    """
    new_path = DATA_DIR / "predictions" / new_name
    if not new_path.exists():
        return Gate("none reproduction", None, f"{new_name} is not yet written")
    new = read_predictions(experiment_id=EXPERIMENT_ID, name=new_name, row_set_id=ROW_SET_ID)
    old = read_predictions(
        experiment_id=STORED_EXPERIMENT_ID, name=stored_name, row_set_id=ROW_SET_ID
    )
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = new.select(*keys, new_q50="q50").join(
        old.select(*keys, old_q50="q50"), on=keys, how="full", coalesce=True
    )
    if joined["new_q50"].null_count() or joined["old_q50"].null_count():
        return Gate("none reproduction", False, "the two sets hold different rows")
    worst = float((joined["new_q50"] - joined["old_q50"]).abs().max())  # ty: ignore[invalid-argument-type]
    return Gate(
        "none reproduction",
        worst <= REPRODUCTION_TOLERANCE_MW,
        f"max |q50 difference| {worst:.2e} MW",
    )


def placebo_gate(*, results: Iterable[ContrastResult]) -> Gate:
    """Check that the stored placebo sits within 0.06 points of `none` on every bank slice.

    Args:
        results: Placebo minus `none` on each bank slice.

    Returns:
        The gate.
    """
    parts = {r.rows: r.diff for r in results}
    passed = all(abs(v) <= PLACEBO_GATE for v in parts.values())
    return Gate(
        "placebo within 0.06 of none", passed, "; ".join(f"{k} {v:+.4f}" for k, v in parts.items())
    )


def detector_gate(*, det: ContrastResult, ceiling: ContrastResult | None) -> Gate:
    """Return the stage 1 gate: `det` minus `none` on eras at 7 to 28 days must reach -1.5.

    Args:
        det: `det` minus `none` on `era 7-28 d`.
        ceiling: `ceiling` minus `none` on the same rows, or None.

    Returns:
        The gate, passed when the point estimate is at or below -1.5.
    """
    beside = "" if ceiling is None else f"; ceiling {ceiling.diff:+.3f}"
    return Gate("stage 1 detector gate", det.diff <= K1_THRESHOLD, f"det {det.diff:+.3f}{beside}")


def invent_register(*, units: pl.DataFrame) -> pl.DataFrame:
    """Return a seeded stand-in register for the self-test (half the eras, a quarter of switches).

    Args:
        units: One arm's `read_bank_arm` frame.

    Returns:
        The columns of `sim_register.parquet`.
    """
    rng = np.random.default_rng(0)
    events = (
        units.group_by("fold", "pass", "event")
        .agg(pl.col("truth").first())
        .sort("fold", "pass", "event")
    )
    rows = []
    for row in events.iter_rows(named=True):
        is_era = row["truth"] == "era"
        draw = rng.random()
        if is_era and draw < 0.5:
            rows.append((row["fold"], row["pass"], row["event"], "true", None))
        elif not is_era and draw < 0.25:
            where = "donor" if rng.random() < 0.5 else "recipient"
            rows.append((row["fold"], row["pass"], row["event"], "decoy", where))
    return pl.DataFrame(
        rows, schema=["fold", "pass", "event", "row_kind", "decoy_at"], orient="row"
    )


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def write_table(*, frame: pl.DataFrame, name: str, prefix: str) -> Path:
    """Write a table to `SCORES_DIR` as `<prefix>INTERIM_712_<name>.csv` and return its path."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    path = SCORES_DIR / f"{prefix}INTERIM_712_{name}.csv"
    frame.write_csv(path)
    return path


def print_result(*, result: ContrastResult) -> None:
    """Print one contrast on one line."""
    print(
        f"{result.contrast:30s} {result.treatment:10s}-{result.reference:8s} "
        f"{result.rows:26s} {result.subset:16s} n {result.n_units:8d}  "
        f"diff {result.diff:+.4f} [{result.lower:+.4f}, {result.upper:+.4f}]  "
        f"meanAE {result.mean_ae_diff:+.4f} MAE50 {result.mae_median_diff:+.4f}"
    )


def run_bank(
    *,
    arms: Sequence[str],
    folds: Sequence[str],
    seeds: Sequence[int],
    self_test: bool,
) -> dict[str, pl.DataFrame]:
    """Read every available bank arm and return its event frame.

    Args:
        arms: Arms wanted; an arm with a missing folder is skipped with a message.
        folds: Fold months.
        seeds: Seeds.
        self_test: Whether the new arms read stored arms.

    Returns:
        The frame of each arm that exists.
    """
    absent = missing_bank_arms(arms=arms, folds=folds, seeds=seeds, self_test=self_test)
    if absent:
        print(f"bank arms not found, their contrasts are skipped: {absent}")
    present = [a for a in arms if a not in absent]
    ids = k578.sites()["time_series_id"].to_numpy()
    first = bank_folder(
        source=source_for(arm=present[0], self_test=self_test), month=folds[0], seed=seeds[0]
    )
    stems = [p.stem for p in sorted(first.glob("pass[0-9][0-9].parquet"))]
    recipients = load_recipients(folds=folds, passes=stems)
    if not recipients:
        print("no cached worlds found: recipient slices are skipped")
    return {
        arm: read_bank_arm(
            arm=arm, folds=folds, seeds=seeds, ids=ids, recipients=recipients, self_test=self_test
        )
        for arm in present
    }


def score_contrasts(
    *,
    specs: Sequence[ContrastSpec],
    real_totals: Mapping[str, pl.DataFrame],
    bank_units: Mapping[str, pl.DataFrame],
    register: pl.DataFrame | None,
    level: float,
    n_resamples: int,
) -> tuple[list[ContrastResult], list[str]]:
    """Score every specification whose arms and rows exist.

    Args:
        specs: The contrasts.
        real_totals: Each real-row arm's totals.
        bank_units: Each bank arm's event frame.
        register: The simulated register, or None.
        level: Interval coverage in percent.
        n_resamples: Number of resamples.

    Returns:
        The results and the names of the contrasts that were skipped.
    """
    results: list[ContrastResult] = []
    skipped: list[str] = []
    for spec in specs:
        pool: Mapping[str, Any] = real_totals if spec.table == "real" else bank_units
        if spec.treatment not in pool or spec.reference not in pool:
            skipped.append(f"{spec.name} (arm missing)")
            continue
        if spec.table == "real":
            results.append(
                real_contrast(
                    totals=real_totals,
                    treatment=spec.treatment,
                    reference=spec.reference,
                    scope=spec.rows,
                    contrast=spec.name,
                    level=level,
                    n_resamples=n_resamples,
                )
            )
            continue
        keys = None
        if spec.subset:
            keys = register_subset(register=register, subset=spec.subset)
            if keys is None:
                skipped.append(f"{spec.name} (no simulated register)")
                continue
        if spec.rows not in set(bank_units[spec.treatment]["slice"].unique().to_list()):
            skipped.append(f"{spec.name} (no rows for {spec.rows})")
            continue
        results.append(
            bank_contrast(
                units=bank_units,
                treatment=spec.treatment,
                reference=spec.reference,
                slice_name=spec.rows,
                contrast=spec.name,
                level=level,
                n_resamples=n_resamples,
                rng=np.random.default_rng(BANK_SEED),
                keys=keys,
                subset=spec.subset,
            )
        )
    return results, skipped


def every_arm_tables(
    *,
    arms: Sequence[str],
    reference: str,
    real_totals: Mapping[str, pl.DataFrame],
    bank_units: Mapping[str, pl.DataFrame],
    level: float,
    n_resamples: int,
) -> list[ContrastResult]:
    """Return each arm minus the reference on every real scope and every bank slice.

    Args:
        arms: The arms to compare (the reference itself is skipped).
        reference: The reference arm.
        real_totals: Each real-row arm's totals.
        bank_units: Each bank arm's event frame.
        level: Interval coverage in percent.
        n_resamples: Number of resamples.

    Returns:
        One result per arm, table, and scope or slice.
    """
    out: list[ContrastResult] = []
    for arm in arms:
        if arm == reference:
            continue
        if arm in real_totals and reference in real_totals:
            scopes = real_totals[arm]["scope"].unique(maintain_order=True).to_list()
            out.extend(
                real_contrast(
                    totals=real_totals,
                    treatment=arm,
                    reference=reference,
                    scope=scope,
                    contrast="arm minus reference",
                    level=level,
                    n_resamples=n_resamples,
                )
                for scope in scopes
            )
        if arm in bank_units and reference in bank_units:
            slices = sorted(bank_units[arm]["slice"].unique().to_list())
            out.extend(
                bank_contrast(
                    units=bank_units,
                    treatment=arm,
                    reference=reference,
                    slice_name=slice_name,
                    contrast="arm minus reference",
                    level=level,
                    n_resamples=n_resamples,
                    rng=np.random.default_rng(BANK_SEED),
                )
                for slice_name in slices
            )
    return out


def results_frame(*, results: Iterable[ContrastResult]) -> pl.DataFrame:
    """Return the results as a table."""
    return pl.DataFrame([asdict(r) for r in results])


def report_verdict(
    *, by_name: Mapping[str, ContrastResult], gates: Sequence[Gate], prefix: str
) -> None:
    """Apply the decision rule, print the verdict, the vetoes, and K8, and write the verdict table.

    Args:
        by_name: The results by contrast name.
        gates: The gates evaluated; the stage 1 detector gate is reported and does not enter the
            verdict, because it stops the run before the bank is labelled.
        prefix: The file-name prefix.
    """
    needed = {"K1", "K2", "K3", "K4", "K4-placebo"}
    if not needed <= set(by_name):
        print(f"\nno verdict: the contrasts {sorted(needed - set(by_name))} are missing")
        return
    state = {g.name: g.passed for g in gates if g.name in VERDICT_GATE_NAMES}
    not_evaluated = sorted(n for n in VERDICT_GATE_NAMES if state.get(n) is None)
    if not_evaluated:
        print(f"\nno verdict: the gates {not_evaluated} are NOT EVALUATED")
        return
    verdict, reason = decide(results=by_name, gates_passed=all(state.values()))
    k2s = {n: veto_k2(result=by_name[n]) for n in ("K2", "K2-ecr", "K2-llm-ecr") if n in by_name}
    k4s = {
        n: veto_k4(result=r, placebo=by_name["K4-placebo"])
        for n, r in by_name.items()
        if n.startswith("K4") and n != "K4-placebo"
    }
    print(f"\nverdict (INTERIM): {verdict} ({reason})")
    print(f"K2 vetoes pass: {k2s}\nK4 vetoes pass: {k4s}")
    if "K8" in by_name:
        k8 = by_name["K8"]
        print(f"K8 interval [{k8.lower:+.3f}, {k8.upper:+.3f}]")
    rows: list[tuple[str, str, str]] = [("verdict", verdict, reason)]
    rows += [(f"{n} veto passes", str(v), "") for n, v in {**k2s, **k4s}.items()]
    write_table(
        frame=pl.DataFrame(rows, schema=["item", "value", "detail"], orient="row"),
        name="verdict",
        prefix=prefix,
    )


def run_scoring(args: argparse.Namespace) -> None:
    """Score the real rows and the bank, then write the INTERIM tables and print the verdict."""
    self_test = bool(args.self_test)
    months = list(args.folds)
    seeds = list(args.seeds)
    arms = [*STORED_ARMS, *NEW_ARMS]
    absent = missing_real_sets(arms=arms, months=months, seeds=seeds, self_test=self_test)
    if absent:
        print(f"real-row sets not found, those arms are skipped: {sorted(absent)}")
    real_arms = [a for a in arms if a not in absent]
    context = load_real_context()
    real_totals: dict[str, pl.DataFrame] = {}
    by_site: list[pl.DataFrame] = []
    for arm in real_arms:
        real_totals[arm], site_frame = read_real_arm(
            arm=arm, months=months, seeds=seeds, context=context, self_test=self_test
        )
        by_site.append(site_frame)
        print(f"read real rows of {arm}")
    bank_wanted = [*arms, *OPTIONAL_BANK_ARMS, *(("oracle-enc",) if self_test else ())]
    bank_units = run_bank(arms=bank_wanted, folds=months, seeds=seeds, self_test=self_test)
    register = None
    if self_test and "none" in bank_units:
        register = invent_register(units=bank_units["none"])
    elif REGISTER_PATH.exists():
        register = pl.read_parquet(REGISTER_PATH)
    else:
        print(f"no simulated register at {REGISTER_PATH}: K5, K6, K6b, and decoy rows skipped")
    prefix = "SELFTEST_" if self_test else ""
    results, skipped = score_contrasts(
        specs=planned_contrasts(),
        real_totals=real_totals,
        bank_units=bank_units,
        register=register,
        level=args.level,
        n_resamples=args.resamples,
    )
    print(
        f"\nINTERIM: seeds {seeds}, folds {months}; CRPS in points of capacity; difference = "
        f"treatment minus reference; level {args.level:.2f}%, {args.resamples} resamples\n"
    )
    for result in results:
        print_result(result=result)
    if skipped:
        print(f"\nskipped contrasts: {skipped}")
    by_name = {r.contrast: r for r in results}
    write_table(frame=results_frame(results=results), name="contrasts", prefix=prefix)
    write_table(
        frame=every_arm_frame(
            arms=list(dict.fromkeys([*real_arms, *bank_units])),
            real_totals=real_totals,
            bank_units=bank_units,
            level=args.level,
            n_resamples=args.resamples,
        ),
        name="every_arm_vs_none",
        prefix=prefix,
    )
    write_table(frame=absolute_metrics(totals=real_totals), name="real_arm_metrics", prefix=prefix)
    write_table(frame=pl.concat(by_site), name="median_minus_mean_by_site", prefix=prefix)
    gates = build_gates(by_name=by_name, bank_units=bank_units, args=args, level=args.level)
    for gate in gates:
        shown = "NOT EVALUATED" if gate.passed is None else str(gate.passed)
        print(f"gate {gate.name}: {shown} ({gate.detail})")
    write_table(
        frame=pl.DataFrame(
            [g._asdict() for g in gates],
            schema={"name": pl.String, "passed": pl.Boolean, "detail": pl.String},
        ),
        name="gates",
        prefix=prefix,
    )
    report_verdict(by_name=by_name, gates=gates, prefix=prefix)
    if self_test:
        check_against_709(real_totals=real_totals, bank_units=bank_units)


def every_arm_frame(
    *,
    arms: Sequence[str],
    real_totals: Mapping[str, pl.DataFrame],
    bank_units: Mapping[str, pl.DataFrame],
    level: float,
    n_resamples: int,
) -> pl.DataFrame:
    """Return the arm-minus-`none` table over every real scope and bank slice."""
    return results_frame(
        results=every_arm_tables(
            arms=arms,
            reference="none",
            real_totals=real_totals,
            bank_units=bank_units,
            level=level,
            n_resamples=n_resamples,
        )
    )


def build_gates(
    *,
    by_name: Mapping[str, ContrastResult],
    bank_units: Mapping[str, pl.DataFrame],
    args: argparse.Namespace,
    level: float,
) -> list[Gate]:
    """Return the gates that can be evaluated from the files present.

    Args:
        by_name: The planned results by contrast name.
        bank_units: Each bank arm's event frame.
        args: The command-line arguments (`resamples`, `self_test`, `repro_name`).
        level: Interval coverage in percent.

    Returns:
        The reproduction gate, the placebo gate, and the stage 1 detector gate where computable.
        The pilot gates and the label-level checks need the label tables, which this script does
        not read.
    """
    gates = [
        reproduction_gate(new_name=args.repro_name, stored_name="none-2025-11-s0")
        if not args.self_test
        else Gate("none reproduction", None, "not applicable to the self-test")
    ]
    if "placebo" in bank_units and "none" in bank_units:
        placebo = [
            bank_contrast(
                units=bank_units,
                treatment="placebo",
                reference="none",
                slice_name=s,
                contrast="placebo gate",
                level=level,
                n_resamples=args.resamples,
                rng=np.random.default_rng(BANK_SEED),
            )
            for s in (*BANK_ERA_SLICES[:2], BANK_DONOR_SLICE)
        ]
        gates.append(placebo_gate(results=placebo))
    if "det" in bank_units and "none" in bank_units:
        det = bank_contrast(
            units=bank_units,
            treatment="det",
            reference="none",
            slice_name="era 7-28 d",
            contrast="stage 1 gate",
            level=level,
            n_resamples=args.resamples,
            rng=np.random.default_rng(BANK_SEED),
        )
        ceiling = None
        if "ceiling" in bank_units:
            ceiling = bank_contrast(
                units=bank_units,
                treatment="ceiling",
                reference="none",
                slice_name="era 7-28 d",
                contrast="stage 1 gate",
                level=level,
                n_resamples=args.resamples,
                rng=np.random.default_rng(BANK_SEED),
            )
        gates.append(detector_gate(det=det, ceiling=ceiling))
    return gates


# --------------------------------------------------------------------------------------------
# Self-test against experiment 709's stored tables
# --------------------------------------------------------------------------------------------

REAL_REFERENCE_LEVEL: Final[float] = 99.44
REAL_REFERENCE_RESAMPLES: Final[int] = 2000
BANK_REFERENCE_RESAMPLES: Final[int] = 10_000
PRINTED_PRECISION: Final[float] = 5e-7
"""A stored value matches when it agrees to the 5e-7 tolerance that this check applies."""
BANK_ARM_ORDER: Final[tuple[str, ...]] = (
    "oracle", "placebo", "oracle-enc", "all", "rule", "soft", "hmm", "stat",
)  # fmt: skip
"""The order in which `e00709_bank_score.py` drew its intervals from one random stream."""


LIBRARY_TOLERANCE: Final[float] = 1e-9


def fast_bootstrap_gap_to_library() -> float:
    """Return the largest bound difference between the month-total bootstrap and the library's.

    A seeded synthetic frame of two arms, 2 seeds, 4 months, and 200 rows per month is bootstrapped
    with `bootstrap_difference_at_level` and with `bootstrap_month_totals_at_level` at 99.58%.

    Returns:
        The largest absolute difference between the two lower and the two upper bounds.
    """
    rng = np.random.default_rng(1)
    months = ["2026-01", "2026-02", "2026-03", "2026-04"]
    keys = pl.concat(
        [
            pl.DataFrame({"time": [f"{m}|{t:04d}" for t in range(200)]}).with_columns(
                month=pl.lit(m), seed=pl.lit(seed)
            )
            for m in months
            for seed in (0, 1)
        ]
    )
    losses = pl.concat(
        [
            keys.with_columns(
                arm=pl.lit(arm), site=pl.lit(7), crps=pl.Series(rng.normal(size=keys.height))
            )
            for arm in ("a", "b")
        ]
    )
    library = bootstrap_difference_at_level(
        losses=losses, treatment="a", reference="b", metric="crps", level=LEVEL_PERCENT,
        n_resamples=3000,
    )  # fmt: skip
    totals = losses.group_by("arm", "seed", "month").agg(pl.col("crps").sum()).sort("seed", "month")
    first = totals.filter(pl.col("arm") == "a")["crps"].to_numpy().reshape(2, 4)
    second = totals.filter(pl.col("arm") == "b")["crps"].to_numpy().reshape(2, 4)
    mine = bootstrap_month_totals_at_level(
        month_sums=first - second, month_counts=np.full(4, 200), level=LEVEL_PERCENT,
        n_resamples=3000,
    )  # fmt: skip
    return max(abs(library[0] - mine[0]), abs(library[1] - mine[1]))


def check_against_709(
    *, real_totals: Mapping[str, pl.DataFrame], bank_units: Mapping[str, pl.DataFrame]
) -> None:
    """Compare the engine with experiment 709's stored real-row and bank tables and print the match.

    Real rows: `rule` minus `none` (and the other aliased arms) at 99.44% and 2,000 resamples
    against `interim_phase_b_real_rows.csv`. Bank: every stored arm minus `none` against
    `interim_bank_rows.csv`, in the order and from the one random stream of `e00709_bank_score.py`.

    Args:
        real_totals: Each real-row arm's totals.
        bank_units: Each bank arm's event frame.
    """
    failures = 0
    deviations: list[str] = []
    stored_real = pl.read_csv(STORED_DIR / "scores" / "interim_phase_b_real_rows.csv")
    print(
        "\nSELF-TEST: real rows against interim_phase_b_real_rows.csv "
        f"({REAL_REFERENCE_LEVEL}%, {REAL_REFERENCE_RESAMPLES} resamples)"
    )
    for alias, stored_arm in (
        ("llm", "rule"),
        ("det", "hmm"),
        ("ecr", "stat"),
        ("llm-ecr", "soft"),
        ("ceiling", "oracle"),
    ):
        for scope in ("all rows", "D8", "other Primaries"):
            if alias not in real_totals:
                continue
            mine = real_contrast(
                totals=real_totals,
                treatment=alias,
                reference="none",
                scope=scope,
                contrast="self-test",
                level=REAL_REFERENCE_LEVEL,
                n_resamples=REAL_REFERENCE_RESAMPLES,
            )
            row = stored_real.filter((pl.col("scope") == scope) & (pl.col("arm") == stored_arm))
            expected = row.row(0, named=True)
            point_delta = max(
                abs(mine.treatment_mean - expected["crps"]), abs(mine.diff - expected["diff"])
            )
            interval_delta = max(
                abs(mine.lower - expected["lower"]), abs(mine.upper - expected["upper"])
            )
            point_ok = point_delta <= PRINTED_PRECISION
            interval_ok = interval_delta <= PRINTED_PRECISION
            failures += not point_ok
            if point_ok and not interval_ok:
                deviations.append(f"{alias}(= {stored_arm}) / {scope}")
            verdict = (
                "MATCH"
                if point_ok and interval_ok
                else (
                    "POINT MATCH, interval differs from the stored file" if point_ok else "MISMATCH"
                )
            )
            print(
                f"  {alias:8s}(= {stored_arm:7s}) {scope:16s} mine diff {mine.diff:+.6f} "
                f"[{mine.lower:+.6f}, {mine.upper:+.6f}] stored {expected['diff']:+.6f} "
                f"[{expected['lower']:+.6f}, {expected['upper']:+.6f}] {verdict}"
            )
    stored_bank = pl.read_csv(STORED_DIR / "scores" / "interim_bank_rows.csv")
    print(
        "\nSELF-TEST: bank against interim_bank_rows.csv "
        f"(99.44%, {BANK_REFERENCE_RESAMPLES} resamples, one random stream in 709's order)"
    )
    rng = np.random.default_rng(0)
    for slice_name in sorted(stored_bank["slice"].unique().to_list()):
        for arm in BANK_ARM_ORDER:
            if arm not in bank_units or "none" not in bank_units:
                continue
            mine = bank_contrast(
                units=bank_units,
                treatment=arm,
                reference="none",
                slice_name=slice_name,
                contrast="self-test",
                level=REAL_REFERENCE_LEVEL,
                n_resamples=BANK_REFERENCE_RESAMPLES,
                rng=rng,
            )
            expected = stored_bank.filter(
                (pl.col("slice") == slice_name) & (pl.col("arm") == arm)
            ).row(0, named=True)
            deltas = [
                abs(mine.treatment_mean - expected["crps"]),
                abs(mine.diff - expected["diff"]),
                abs(mine.lower - expected["lower"]),
                abs(mine.upper - expected["upper"]),
            ]
            ok = max(deltas) <= PRINTED_PRECISION and mine.n_units == expected["events"]
            failures += not ok
            print(
                f"  {slice_name:18s} {arm:11s} mine {mine.diff:+.6f} "
                f"[{mine.lower:+.6f}, {mine.upper:+.6f}] stored {expected['diff']:+.6f} "
                f"[{expected['lower']:+.6f}, {expected['upper']:+.6f}] "
                f"max |delta| {max(deltas):.2e} {'MATCH' if ok else 'MISMATCH'}"
            )
    library_gap = fast_bootstrap_gap_to_library()
    print(
        "\nfast month-total bootstrap against bootstrap_difference_at_level on a synthetic "
        f"frame: max |bound difference| {library_gap:.2e}"
    )
    failures += library_gap > LIBRARY_TOLERANCE
    if deviations:
        print(
            f"WARNING: {len(deviations)} real-row interval cells differ from the stored file "
            f"while the point estimates match: {deviations}. The stored 709 file records no "
            "level, resample count, or seed, so 99.44% and 2,000 resamples are inferred."
        )
    print(f"\nSELF-TEST {'PASSED' if failures == 0 else f'FAILED ({failures} mismatches)'}")
    if failures:
        sys.exit(1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--folds", nargs="+", default=list(TEST_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--level", type=float, default=LEVEL_PERCENT)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument(
        "--repro-name",
        default="none-repro-2025-11-s0",
        help="prediction set of the new script's `none` run, for the reproduction gate",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="treat stored 709 arms as the new arms and compare with 709's stored tables",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Score the arms and write the INTERIM tables."""
    args = parse_args()
    run_scoring(args)


if __name__ == "__main__":
    main()

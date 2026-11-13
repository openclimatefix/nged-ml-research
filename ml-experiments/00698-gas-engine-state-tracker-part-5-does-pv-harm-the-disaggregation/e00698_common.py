"""Paths, constants, solar curves, host rows, and paired week-block intervals for experiment 698.

D8 and D3 are substations and may be named. The engine record is called "Gas A" and the solar farm
is called "the farm" everywhere that is printed or drawn, and both appear only inside scores pooled
over a whole run, never with a date and never as a series.

**Lineage.** This module imports experiment 694's injection and trackers, experiment 695's grid
index, and experiment 697's local-time helpers, week-block intervals, span bounds, seed offsets, and
`track_span`. `farm_norm_series` is copied from `e00405_prep.py` (experiment 405), which experiments
649 and 662 used to inject their farms, because importing that module imports `torch` through
experiment 397. The copy changes nothing but its docstring and the use of one constant for the
frame's path.

**Time conventions.** A power stamp marks the end of its half-hour, and so do the stamps of the
farm series and of CAMS irradiance (the end of the hour). Panel-grid index `i` has the end stamp
`GRID_START + i * 30 min`. Local fields come from the half-hour start, as in experiment 697.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

import numpy as np
import polars as pl
from e00694_common import (  # ty: ignore[unresolved-import]
    GRID_START,
    PANEL_PATH,
    Q_FLOOR,
    inject_folded,
)
from e00697_common import (  # ty: ignore[unresolved-import]
    DAY,
    E694_DIR,
    add_local_fields,
    group_within,
    half_hour_starts,
    percentile_interval,
    ratio,
)
from e00697_common import FRAMES_DIR as E697_FRAMES_DIR  # ty: ignore[unresolved-import]
from e00697_controls import (  # ty: ignore[unresolved-import]
    CONTROL_SEED_OFFSET,
    SERIES_SEED_OFFSET,
    span_bounds,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 698
SLUG: Final[str] = "gas-engine-state-tracker-part-5-does-pv-harm-the-disaggregation"
FOLDER_NAME: Final[str] = f"{EXPERIMENT_ID:05d}-{SLUG}"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / FOLDER_NAME
"""The experiment's private data folder: audit tables, one file per run, scores, verdict."""
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / FOLDER_NAME
)
"""Where the published figures and the page are written."""
PRINTED_DIR: Final[Path] = Path(__file__).resolve().parent / "printed"
"""Printed reports, committed with the scripts. They hold no date and no Gas A or farm series."""
JOBS_DIR: Final[Path] = DATA_DIR / "jobs"
AUDIT_PATH: Final[Path] = DATA_DIR / "audit.json"
GATE_PATH: Final[Path] = DATA_DIR / "gate.json"
SCORES_PATH: Final[Path] = DATA_DIR / "scores.json"
VERDICT_PATH: Final[Path] = DATA_DIR / "verdict.json"
FARM_FRAME: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
"""Experiment 39's frame of the six solar farms (experiments 405, 649, and 662 read it)."""
E697_FRAME_D8: Final[Path] = E697_FRAMES_DIR / "frame_d8.parquet"
E697_FRAME_C1_NULL: Final[Path] = E697_FRAMES_DIR / "frame_c1_null.parquet"
E697_FRAME_C1S_SOLAR: Final[Path] = E697_FRAMES_DIR / "frame_c1s_solar_null.parquet"
D8_STATES_PATH: Final[Path] = E694_DIR / "d8.pkl"

# --------------------------------------------------------------------------- the plan's numbers

HOST_KEYS: Final[tuple[str, ...]] = ("d8pre", "d3solar")
HOST_SERIES: Final[dict[str, str]] = {"d8pre": "D8", "d3solar": "D3"}
HOST_SPAN: Final[dict[str, str]] = {"d8pre": "pre", "d3solar": "solar"}
HOST_CONTROL: Final[dict[str, str]] = {"d8pre": "c1", "d3solar": "c2"}
"""The control of experiment 697 whose seed recipe each host's engine draw follows."""
CAPACITIES_MW: Final[tuple[float, ...]] = (7.0, 8.9)
"""7.0 MW is the lower value of the private review and 8.9 MW the register figure that
experiment 673 published. 2.6 MW (a linear probe, biased low by the fold) is not run."""
SHAPES: Final[tuple[str, ...]] = ("farm", "cams", "smooth", "smooth_farm")
"""`farm` and `cams` carry weather. `smooth` is the CAMS cell mean (the twin of `cams`, and the base
of the level contrast). `smooth_farm` is the farm's own cell mean (the twin of `farm`)."""
PV_FREE: Final[str] = "free"
"""The shape name of the solar-free twin."""
FARM_LETTERS: Final[str] = "ABCF"
"""Experiment 662's four farms."""
N_SEEDS: Final[int] = 3
STEP4_SEEDS: Final[int] = 5
STEP4_ERAS: Final[tuple[str, ...]] = ("net", "solar")
STEP4_SEED_OFFSET: Final[dict[str, int]] = {"net": 400, "solar": 500}
"""Offsets of the seed of step 4's day draw, apart from every host job's offset."""
STEP4_MIN_SPELL: Final[int] = 48
"""Step 4 injects into whole local days inside labelled off-spells of this many half-hours."""
STEP4_WHOLE_DAYS_ONLY: Final[bool] = True
"""The plan injects only whole local days. With False, every half-hour of an off-spell of
`STEP4_MIN_SPELL` or more takes part, which gives many more half-hours in the solar era."""
N_WORKERS: Final[int] = 8

MIDDAY_FIRST: Final[int] = 20
MIDDAY_LAST: Final[int] = 29
"""Local half-hours 20 to 29 start at 10:00 to 14:30, so they cover 10:00 to 15:00."""
EVENING_FIRST: Final[int] = 32
EVENING_LAST: Final[int] = 37
"""Local half-hours 32 to 37 cover 16:00 to 19:00."""
NIGHT_LAST: Final[int] = 9
"""Local half-hours 0 to 9 cover 00:00 to 05:00."""
SUMMER_MONTHS: Final[tuple[int, ...]] = (5, 6, 7, 8, 9)
GHI_MIN: Final[float] = 50.0
"""A half-hour has a bright-or-dull third only above this CAMS irradiance (W/m2)."""
UNIT_THRESHOLD_MW: Final[float] = 1.97 / 2
"""An injected output of this size or more counts as 1 or more units running (`truth1`)."""
FOLD_NEAR_ZERO_MW: Final[float] = 1.0
"""Fold states: signed real power above this is forward flow, below minus this is reverse flow."""

GATE_TOLERANCE_MVA: Final[float] = 1.0
GATE_FIFTHS: Final[int] = 4
"""The curve gate reads the first four fifths of solar output."""
N_FIFTHS: Final[int] = 5
MIN_ROWS_PER_FIFTH: Final[int] = 20
COVERAGE_MIN: Final[float] = 0.90
"""Branch 1 fires when the chosen farm's finite share of the daylight half-hours of a host that runs
is under this. Daylight half-hours are those with a positive CAMS shape at the host. Farm gaps are
filled with the farm's cell mean and masked out of every score."""
TWIN_F1_EXPECTED: Final[float] = 0.767
TWIN_F1_TOLERANCE: Final[float] = 0.02
HALF_WIDTH_MAX: Final[float] = 0.05
"""Branch 1: the paired half-width of the F1 contrast at D8 may not exceed this."""
STEP4_HALF_WIDTH_MAX: Final[float] = 0.10

CriterionType = Literal["f1", "false_flag", "bmd"]
CRITERIA: Final[tuple[CriterionType, ...]] = ("f1", "false_flag", "bmd")
THRESHOLDS: Final[dict[str, float]] = {"f1": 0.05, "false_flag": 0.02, "bmd": 0.05}
"""Harm thresholds at 10:00 to 15:00: F1 lower by 0.05, false-flag share higher by 0.02, and the
bright-minus-dull flagged share higher by 0.05."""
STRONG_BMD: Final[float] = 0.125
"""Half of D8's swing between the eras, for "weather explains D8's contrast" at D8's reading."""

N_RESAMPLES: Final[int] = 2000
LABEL_ONLY_RESAMPLES: Final[int] = 1000
BOOTSTRAP_SEED: Final[int] = 698

ShapeType = Literal["farm", "cams", "smooth", "smooth_farm", "free"]
KindType = Literal["host", "step4"]
LabelType = Literal["smooth", "filtered"]
LABELS: Final[tuple[LabelType, ...]] = ("smooth", "filtered")
SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "tp",
    "fp",
    "fn",
    "ff",
    "zero",
    "flag_bright",
    "bright",
    "flag_dull",
    "dull",
)
"""Row-level quantities whose sums give F1, the false-flag share, and the flagged shares."""
FOLD_STATES: Final[tuple[str, ...]] = ("forward", "near_zero", "reverse")


class Interval(TypedDict):
    """A point estimate with a percentile interval."""

    point: float
    lo: float
    hi: float


# --------------------------------------------------------------------------- jobs


@dataclass(frozen=True)
class Job:
    """One tracker run on an injected reading.

    A host job has a host, a capacity, a solar shape, and a seed (None for the run with no engine).
    A step 4 job has an era of D8 and a seed. The solar-free twin has no capacity and the shape
    `free`.
    """

    name: str
    kind: KindType
    host: str
    capacity_mw: float | None
    shape: str
    seed: int | None
    era: str | None = None


def capacity_key(*, capacity_mw: float | None) -> str:
    """Return the short key of a capacity, such as `c70`, or `none` for the solar-free twin."""
    return "none" if capacity_mw is None else f"c{round(capacity_mw * 10):02d}"


def host_job_name(*, host: str, capacity_mw: float | None, shape: str, seed: int | None) -> str:
    """Return the name of a host job, which names its stored file."""
    tag = "n" if seed is None else str(seed)
    return f"{host}_{capacity_key(capacity_mw=capacity_mw)}_{shape}_s{tag}"


def host_jobs(*, gate: dict | None) -> list[Job]:
    """Return the host jobs that the gate lets run, with the solar-free twins.

    Args:
        gate: The parsed `gate.json` (its `run` table), or None to list every job the plan allows
            (56 at most).

    Returns:
        For each host: the solar-free twin (3 seeds and no engine) when any shape passes at any
        capacity, and each passing shape at each capacity (3 seeds and no engine).
    """
    out: list[Job] = []
    seeds: list[int | None] = [*range(N_SEEDS), None]
    for host in HOST_KEYS:
        passing = [
            (cap, shape)
            for cap in CAPACITIES_MW
            for shape in SHAPES
            if gate is None or gate["run"][host][capacity_key(capacity_mw=cap)][shape]
        ]
        if not passing:
            continue
        combos: list[tuple[float | None, str]] = [(None, PV_FREE), *passing]
        out.extend(
            Job(
                name=host_job_name(host=host, capacity_mw=cap, shape=shape, seed=seed),
                kind="host",
                host=host,
                capacity_mw=cap,
                shape=shape,
                seed=seed,
            )
            for cap, shape in combos
            for seed in seeds
        )
    return out


def step4_jobs() -> list[Job]:
    """Return the 10 jobs of step 4: Gas A injected into D8's own off-spells, 5 seeds per era."""
    return [
        Job(
            name=f"step4_{era}_s{seed}",
            kind="step4",
            host="d8own",
            capacity_mw=None,
            shape="none",
            seed=seed,
            era=era,
        )
        for era in STEP4_ERAS
        for seed in range(STEP4_SEEDS)
    ]


def all_jobs(*, gate: dict | None) -> list[Job]:
    """Return the host jobs followed by the step 4 jobs."""
    return [*host_jobs(gate=gate), *step4_jobs()]


def engine_seed(*, host: str, seed: int) -> int:
    """Return the seed of a host job's day draw, by experiment 697's recipe.

    The seed is the job's seed plus the series offset and the control offset of experiment 697, so
    the solar-free twin at D8 draws the days of experiment 697's control C1 and the twin at D3
    draws those of its control C2 for seeds 0 and 1. The draw seed offset `DRAW_SEED` is 0.
    """
    from e00697_engine import DRAW_SEED  # ty: ignore[unresolved-import]

    return (
        DRAW_SEED
        + seed
        + SERIES_SEED_OFFSET[HOST_SERIES[host]]
        + CONTROL_SEED_OFFSET[HOST_CONTROL[host]]
    )


def host_bounds(*, host: str) -> tuple[int, int]:
    """Return the first panel index of a host's span and one past its last."""
    return span_bounds(span=HOST_SPAN[host])


# --------------------------------------------------------------------------- stored files


def read_json(*, path: Path) -> dict:
    """Return a stored JSON file, or raise with the script that writes it."""
    if not path.exists():
        msg = f"{path} is missing; run the script that writes it first"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def write_json(*, path: Path, payload: dict) -> None:
    """Write a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_json_default))
    tmp.replace(path)


def _json_default(value: object) -> object:
    """Make numpy scalars and arrays serialisable."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    msg = f"cannot serialise {type(value)}"
    raise TypeError(msg)


AUDIT_CODE: Final[tuple[str, ...]] = ("e00698_common.py", "e00698_audit.py", "e00698_labels.py")
"""The scripts whose change makes an earlier audit stale."""


def require_passing_audit() -> dict:
    """Return the stored audit, or stop if it failed, is missing, or is older than its inputs.

    The audit's inputs are the scripts in `AUDIT_CODE` and the stored files that it reads.
    `gate.json` must exist too, because a failed audit does not write one.

    Returns:
        The parsed `audit.json`.
    """
    audit = read_json(path=AUDIT_PATH)
    here = Path(__file__).resolve().parent
    inputs = [
        *(here / name for name in AUDIT_CODE),
        PANEL_PATH,
        FARM_FRAME,
        E697_FRAME_D8,
        E697_FRAME_C1_NULL,
        E697_FRAME_C1S_SOLAR,
    ]
    problems = []
    if audit["failed"]:
        problems.append(f"the audit failed these checks: {audit['failed']}")
    if not GATE_PATH.exists():
        problems.append(f"{GATE_PATH.name} is missing")
    newer = [p.name for p in inputs if p.stat().st_mtime > AUDIT_PATH.stat().st_mtime]
    if newer:
        problems.append(f"the audit is older than its inputs {newer}")
    if problems:
        msg = "rerun e00698_audit.py before this script: " + "; ".join(problems)
        raise SystemExit(msg)
    return audit


# --------------------------------------------------------------------------- the panel


@dataclass(frozen=True)
class GridCalendar:
    """The local fields of every half-hour of the panel grid, as numpy arrays."""

    month: np.ndarray
    half_hour: np.ndarray
    weekday: np.ndarray
    weekend: np.ndarray
    summer_half: np.ndarray
    clock_change_day: np.ndarray
    week: np.ndarray
    local_date: np.ndarray
    """The local date as days since 1970-01-01."""


def grid_calendar_fields(*, n_grid: int) -> GridCalendar:
    """Return the local fields of grid indices 0 to `n_grid - 1`."""
    frame = add_local_fields(
        frame=pl.DataFrame({"start": half_hour_starts(index=np.arange(n_grid))})
    )
    return GridCalendar(
        month=frame["month"].to_numpy().astype(np.int64),
        half_hour=frame["local_half_hour"].to_numpy().astype(np.int64),
        weekday=frame["weekday"].to_numpy().astype(np.int64),
        weekend=frame["weekend"].to_numpy(),
        summer_half=frame["summer_half"].to_numpy(),
        clock_change_day=frame["clock_change_day"].to_numpy(),
        week=frame["week"].to_numpy().astype(np.int64),
        local_date=frame["local_date"].dt.epoch("d").to_numpy().astype(np.int64),
    )


def panel_series(*, panel: dict[int, dict], label: str, key: str) -> np.ndarray:
    """Return one array of a site in experiment 196's panel: `power`, `ghi`, or `pv_shape`."""
    entry = next(v for v in panel.values() if v["site"] == label)
    return np.asarray(entry[key], dtype=float)


# --------------------------------------------------------------------------- the farm


FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI: Final[float] = 100.0
US_PER_HALF_HOUR: Final[int] = 1_800_000_000
US_PER_HOUR: Final[int] = 3_600_000_000


def farm_norm_series(*, farm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a farm's sorted times (microseconds) and power over its 99th percentile.

    Copied from `farm_norm_series` of `e00405_prep.py` (experiment 405). The percentile is over the
    farm's readings with the sun under 75 degrees of zenith, as in experiment 129. A reading at or
    below zero while the irradiance exceeds 100 W/m2 is a dropout and returns NaN. Other negative
    readings are set to zero.

    Args:
        farm: The farm's letter.

    Returns:
        The times as microseconds since the epoch, and the normalised power.
    """
    everything = pl.read_parquet(FARM_FRAME).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    daylight = everything.filter(pl.col("zenith") < FARM_MAX_ZENITH_DEG)
    p99 = float(
        np.nanpercentile(daylight.filter(pl.col("power").is_not_null())["power"].to_numpy(), 99)
    )
    dropout = (pl.col("power") <= 0.0) & (pl.col("ghi_u") > FARM_DROPOUT_GHI)
    frame = everything.select(
        "time",
        norm=pl.when(pl.col("zenith") >= FARM_MAX_ZENITH_DEG)
        .then(0.0)
        .when(pl.col("power").is_null() | dropout)
        .then(float("nan"))
        .otherwise(pl.col("power").clip(lower_bound=0.0) / p99),
    ).sort("time")
    return frame["time"].dt.epoch("us").to_numpy(), frame["norm"].to_numpy().astype(np.float64)


def farm_on_grid(*, series: tuple[np.ndarray, np.ndarray], n_grid: int) -> np.ndarray:
    """Return a farm's normalised power on the panel grid, NaN in a gap or a dropout.

    A grid half-hour that the farm's file lacks is 0 at night (UTC hours before 05:00 and after
    20:00, as `farm_at` of experiment 405 decides) and NaN otherwise.

    Args:
        series: The output of `farm_norm_series`.
        n_grid: The length of the panel grid.

    Returns:
        An array of length `n_grid`. Index `i` is the farm's value at the end stamp
        `GRID_START + i * 30 min`.
    """
    times, values = series
    origin = int(np.datetime64(GRID_START.replace(tzinfo=None), "us").astype("int64"))
    ends = origin + np.arange(n_grid, dtype=np.int64) * US_PER_HALF_HOUR
    pos = np.clip(np.searchsorted(times, ends), 0, len(times) - 1)
    found = times[pos] == ends
    out = np.where(found, values[pos], np.nan)
    hour = (ends // US_PER_HOUR) % 24
    night = (hour < 5) | (hour > 20)
    return np.where(~found & night, 0.0, out)


def farm_finite_share(*, norm: np.ndarray, first: int, last: int) -> float:
    """Return the share of grid indices `[first, last)` where the farm series is finite."""
    return float(np.isfinite(norm[first:last]).mean())


def farm_daylight_share(*, norm: np.ndarray, pv_shape: np.ndarray, first: int, last: int) -> float:
    """Return the farm's finite share of the daylight half-hours of `[first, last)`.

    A half-hour is daylight where the host's CAMS shape is positive. Night half-hours, where the
    farm series is zero by construction, do not count.
    """
    day = np.nan_to_num(pv_shape[first:last]) > 0
    return float(np.isfinite(norm[first:last][day]).mean())


def longest_gap(*, finite: np.ndarray) -> int:
    """Return the length in half-hours of the longest run of False in `finite`."""
    best = run = 0
    for ok in finite:
        run = 0 if ok else run + 1
        best = max(best, run)
    return best


# --------------------------------------------------------------------------- solar curves


def cell_mean(*, value: np.ndarray, calendar: GridCalendar, first: int, last: int) -> np.ndarray:
    """Return, for every grid index, the mean of `value` in its cell of month and local half-hour.

    The mean runs over `[first, last)` only, so a run's smooth twin uses data the run holds. A cell
    with no half-hour in the range gets the mean of the whole range.
    """
    key = calendar.month * DAY + calendar.half_hour
    n_key = 13 * DAY
    window = slice(first, last)
    sums = np.bincount(key[window], weights=value[window], minlength=n_key)
    counts = np.bincount(key[window], minlength=n_key)
    overall = float(np.mean(value[window]))
    means = np.where(counts > 0, sums / np.maximum(counts, 1), overall)
    return means[key]


def finite_cell_mean(
    *, value: np.ndarray, calendar: GridCalendar, first: int, last: int
) -> np.ndarray:
    """Return `cell_mean` of the finite values of `value` only (NaN gaps are left out of the mean).

    A cell with no finite half-hour in the range gets the mean of every finite half-hour of the
    range.
    """
    key = calendar.month * DAY + calendar.half_hour
    window = slice(first, last)
    finite = np.isfinite(value[window])
    sums = np.bincount(key[window][finite], weights=value[window][finite], minlength=13 * DAY)
    counts = np.bincount(key[window][finite], minlength=13 * DAY)
    overall = float(np.mean(value[window][finite]))
    means = np.where(counts > 0, sums / np.maximum(counts, 1), overall)
    return means[key]


@dataclass(frozen=True)
class SolarCurve:
    """The injected solar output (MW) on the panel grid, and where the farm is missing."""

    pv_mw: np.ndarray
    farm_ok: np.ndarray


def solar_curve(
    *,
    shape: str,
    capacity_mw: float,
    pv_shape: np.ndarray,
    farm_norm: np.ndarray | None,
    calendar: GridCalendar,
    first: int,
    last: int,
) -> SolarCurve:
    """Return one solar curve, zero outside `[first, last)`.

    Args:
        shape: `farm` (the farm's output per MW times the capacity), `cams` (the CAMS shape of the
            host's own location times the capacity), `smooth` (the capacity times the mean of the
            CAMS shape in each cell of month and local half-hour), or `smooth_farm` (the capacity
            times the mean of the farm's normalised power in each such cell, gaps left out).
        capacity_mw: The capacity in MW of AC capacity.
        pv_shape: The host's CAMS shape per MW of AC capacity on the whole grid.
        farm_norm: The farm's normalised power on the grid (NaN in a gap), or None.
        calendar: The local fields of the grid.
        first: The first grid index that carries the curve.
        last: One past the last.

    Returns:
        The curve, and `farm_ok`, True where the farm series is finite. **A farm gap is filled with
        the farm's cell mean so the reading stays continuous, and every score leaves these
        half-hours out for every run of the host, so every shape is scored on the same rows.**
        The farm run and its twin `smooth_farm` have the same mean output per cell, so their
        contrast holds weather and nothing of the farm's level or daily shape.
    """
    smooth = cell_mean(value=np.nan_to_num(pv_shape), calendar=calendar, first=first, last=last)
    if shape == "cams":
        unit = np.nan_to_num(pv_shape)
        ok = np.ones(len(pv_shape), dtype=bool)
    elif shape == "smooth":
        unit = smooth
        ok = np.ones(len(pv_shape), dtype=bool)
    elif shape in ("farm", "smooth_farm"):
        if farm_norm is None:
            msg = f"the {shape} shape needs the farm series"
            raise ValueError(msg)
        farm_smooth = finite_cell_mean(value=farm_norm, calendar=calendar, first=first, last=last)
        if shape == "smooth_farm":
            unit = farm_smooth
            ok = np.ones(len(pv_shape), dtype=bool)
        else:
            ok = np.isfinite(farm_norm)
            unit = np.where(ok, farm_norm, farm_smooth)
    else:
        msg = f"unknown shape {shape!r}"
        raise ValueError(msg)
    pv = np.zeros(len(pv_shape))
    pv[first:last] = capacity_mw * unit[first:last]
    return SolarCurve(pv_mw=pv, farm_ok=ok)


def fold_reading(
    *, y: np.ndarray, total_mw: np.ndarray, first: int, last: int
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract solar and engine output together from the real power behind a reading.

    **The fold rule is applied once, on the sum.** The real power behind the reading is
    `sqrt(y^2 - q^2)` with `q` = 0.14 MVA. The summed output is subtracted from it, and the reading
    is `sqrt((real - total)^2 + q^2)` (experiment 694's `inject_folded`). The signed real power
    `real - total` is returned, because the reading itself cannot show reverse flow.

    Args:
        y: The whole reading (MVA, NaN where dropped).
        total_mw: Solar plus engine output on the whole grid.
        first: The first grid index of the injection.
        last: One past the last.

    Returns:
        The reading (untouched outside `[first, last)`) and the signed real power on the whole grid
        (NaN outside the span and where the reading is NaN).
    """
    reading = y.copy()
    reading[first:last] = inject_folded(y=y[first:last], g=total_mw[first:last], q=Q_FLOOR)
    signed = np.full(len(y), np.nan)
    real = np.sqrt(np.maximum(y[first:last] ** 2 - Q_FLOOR**2, 0.0))
    signed[first:last] = real - total_mw[first:last]
    return reading, signed


def fold_state(*, signed_mw: np.ndarray) -> np.ndarray:
    """Return 0 for forward flow, 1 within 1 MW of zero, 2 for reverse flow, and -1 for NaN."""
    out = np.where(
        signed_mw > FOLD_NEAR_ZERO_MW,
        0,
        np.where(signed_mw < -FOLD_NEAR_ZERO_MW, 2, 1),
    )
    return np.where(np.isfinite(signed_mw), out, -1)


# --------------------------------------------------------------------------- curves of reading


def fifth_edges(*, values: np.ndarray) -> np.ndarray:
    """Return the four edges (20%, 40%, 60%, 80% quantiles) of the finite `values`."""
    finite = values[np.isfinite(values)]
    return np.quantile(finite, np.arange(1, N_FIFTHS) / N_FIFTHS)


def curve_medians(
    *, reading: np.ndarray, shape_value: np.ndarray, edges: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the median reading and the row count in each fifth of `shape_value`.

    Args:
        reading: The reading (MVA).
        shape_value: The host's CAMS shape, which the fifths are taken of.
        edges: The four fifth edges.
        rows: A boolean mask of the rows that count.

    Returns:
        Two arrays of length 5, NaN median where a fifth holds no row.
    """
    keep = rows & np.isfinite(reading) & np.isfinite(shape_value)
    fifth = np.searchsorted(edges, shape_value, side="right")
    medians = np.full(N_FIFTHS, np.nan)
    counts = np.zeros(N_FIFTHS, dtype=int)
    for k in range(N_FIFTHS):
        sel = keep & (fifth == k)
        counts[k] = int(sel.sum())
        if counts[k]:
            medians[k] = float(np.median(reading[sel]))
    return medians, counts


def gate_passes(*, curve: np.ndarray, reference: np.ndarray, counts: np.ndarray) -> bool:
    """Return whether a curve lies within 1 MVA of the reference in each of the first four fifths.

    A fifth with a NaN median or fewer than `MIN_ROWS_PER_FIFTH` rows fails the gate.
    """
    first_four = slice(0, GATE_FIFTHS)
    ok_rows = bool(np.all(counts[first_four] >= MIN_ROWS_PER_FIFTH))
    diff = np.abs(curve[first_four] - reference[first_four])
    return bool(ok_rows and np.all(np.isfinite(diff)) and np.all(diff <= GATE_TOLERANCE_MVA))


def summer_midday_rows(*, calendar: GridCalendar) -> np.ndarray:
    """Return grid half-hours of May to September, 10:00 to 15:00 local, off clock-change days."""
    return (
        np.isin(calendar.month, SUMMER_MONTHS)
        & (calendar.half_hour >= MIDDAY_FIRST)
        & (calendar.half_hour <= MIDDAY_LAST)
        & ~calendar.clock_change_day
    )


# --------------------------------------------------------------------------- thirds


def bright_dull_thirds(
    *, ghi: np.ndarray, value: np.ndarray, calendar: GridCalendar, rows: np.ndarray
) -> np.ndarray:
    """Return 0 (dullest third), 1, or 2 (brightest third) of each row, and -1 for the rest.

    Thirds are taken of `value` within a cell of local half-hour, weekday class, and month, among
    the `rows` whose CAMS irradiance `ghi` exceeds 50 W/m2. Ties take their average rank, as in
    experiment 697's `group_within`.

    Args:
        ghi: CAMS irradiance on the whole grid (W/m2).
        value: The quantity the thirds are taken of (`ghi` itself or the CAMS shape).
        calendar: The local fields of the grid.
        rows: A boolean mask of the rows that may take part (observed, not on a clock change).

    Returns:
        An integer array on the whole grid.
    """
    eligible = rows & (ghi > GHI_MIN) & np.isfinite(value)
    index = np.flatnonzero(eligible)
    frame = pl.DataFrame(
        {
            "index": index,
            "value": value[index],
            "half_hour": calendar.half_hour[index],
            "weekend": calendar.weekend[index],
            "month": calendar.month[index],
        }
    ).with_columns(
        third=group_within(value="value", n_groups=3, within=("half_hour", "weekend", "month"))
    )
    out = np.full(len(ghi), -1, dtype=np.int64)
    out[frame["index"].to_numpy()] = frame["third"].to_numpy().astype(np.int64)
    return out


# --------------------------------------------------------------------------- score statistics


def score_columns(
    *,
    truth1: np.ndarray,
    flag: np.ndarray,
    zero: np.ndarray,
    third: np.ndarray,
) -> np.ndarray:
    """Return the nine row-level columns of `SCORE_COLUMNS`, shape (rows, 9), as floats.

    Args:
        truth1: Whether 1 or more units run in the injected truth.
        flag: Whether the tracker's label says 1 or more units.
        zero: Whether the injected output is exactly zero (`truth_mw` = 0).
        third: The bright-or-dull third of each row (2 bright, 0 dull, -1 neither).
    """
    t = np.asarray(truth1, dtype=bool)
    f = np.asarray(flag, dtype=bool)
    z = np.asarray(zero, dtype=bool)
    bright = np.asarray(third) == 2
    dull = np.asarray(third) == 0
    return np.stack(
        [t & f, ~t & f, t & ~f, z & f, z, f & bright, bright, f & dull, dull], axis=1
    ).astype(float)


def criterion_value(*, sums: np.ndarray, criterion: str) -> np.ndarray:
    """Return F1, the false-flag share, or the bright-minus-dull flagged share from column sums.

    Args:
        sums: Sums of `SCORE_COLUMNS` with the columns on the last axis (any leading axes).
        criterion: `f1`, `false_flag`, or `bmd`.
    """
    if criterion == "f1":
        tp, fp, fn = sums[..., 0], sums[..., 1], sums[..., 2]
        return ratio(numerator=2 * tp, denominator=2 * tp + fp + fn)
    if criterion == "false_flag":
        return ratio(numerator=sums[..., 3], denominator=sums[..., 4])
    if criterion == "bmd":
        return ratio(numerator=sums[..., 5], denominator=sums[..., 6]) - ratio(
            numerator=sums[..., 7], denominator=sums[..., 8]
        )
    msg = f"unknown criterion {criterion!r}"
    raise ValueError(msg)


def harm_sign(*, criterion: str) -> float:
    """Return +1 where a rise in the criterion's value is harm and -1 where a fall is harm."""
    return -1.0 if criterion == "f1" else 1.0


# --------------------------------------------------------------------------- paired intervals


def week_table(*, week: np.ndarray, columns: np.ndarray, weeks: np.ndarray) -> np.ndarray:
    """Return the per-week column sums on the fixed list of `weeks`, shape (len(weeks), columns).

    Args:
        week: The week number of each row.
        columns: Row-level quantities, shape (rows, columns).
        weeks: The sorted unique week numbers of the host's scored rows. A week with no row in this
            run gets zeros, so every run of a host shares one table layout.
    """
    position = np.searchsorted(weeks, week)
    position = np.clip(position, 0, len(weeks) - 1)
    if not np.all(weeks[position] == week):
        msg = "a row has a week that is not in the host's list of weeks"
        raise ValueError(msg)
    return np.stack(
        [
            np.bincount(position, weights=columns[:, j], minlength=len(weeks))
            for j in range(columns.shape[1])
        ],
        axis=1,
    )


def week_counts(*, n_weeks: int, seed: int, n_resamples: int = N_RESAMPLES) -> np.ndarray:
    """Return the number of times each whole week is drawn, shape (n_resamples, n_weeks).

    **One draw of weeks serves every run of a host**: the counts are generated once from the seed
    and the week list, and every run's per-week table is multiplied by the same counts.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_weeks, np.full(n_weeks, 1.0 / n_weeks), size=n_resamples)


def paired_contrast(
    *,
    criterion: str,
    job_weeks: np.ndarray,
    base_weeks: np.ndarray,
    counts: np.ndarray,
) -> Interval:
    """Return the harm of a job against its base run, with a paired week-block interval.

    Harm is `value(base) - value(job)` for F1 and `value(job) - value(base)` for the false-flag
    share and the bright-minus-dull flagged share, so a positive harm is always worse.

    Args:
        criterion: `f1`, `false_flag`, or `bmd`.
        job_weeks: The job's per-week sums (seeds pooled), shape (weeks, 9).
        base_weeks: The base run's per-week sums on the same weeks and seeds.
        counts: The output of `week_counts`.

    Returns:
        The harm's point estimate and 95% percentile interval.
    """
    sign = harm_sign(criterion=criterion)
    point = sign * (
        criterion_value(sums=job_weeks.sum(axis=0), criterion=criterion)
        - criterion_value(sums=base_weeks.sum(axis=0), criterion=criterion)
    )
    draws = sign * (
        criterion_value(sums=counts @ job_weeks, criterion=criterion)
        - criterion_value(sums=counts @ base_weeks, criterion=criterion)
    )
    lo, hi = percentile_interval(draws=np.asarray(draws, dtype=float))
    return {"point": float(point), "lo": lo, "hi": hi}


def meets_criterion(*, interval: Interval | None, criterion: str) -> bool:
    """Return whether a harm interval reaches the criterion's threshold and excludes zero.

    The point estimate must reach the threshold, and the whole interval must lie above zero, so an
    interval that excludes zero on the harmless side never counts.
    """
    if interval is None or not np.isfinite(interval["lo"]):
        return False
    return bool(interval["point"] >= THRESHOLDS[criterion] and interval["lo"] > 0.0)


def inside_threshold(*, interval: Interval | None, criterion: str) -> bool:
    """Return whether a harm interval lies wholly below the criterion's threshold."""
    if interval is None or not np.isfinite(interval["hi"]):
        return False
    return bool(interval["hi"] < THRESHOLDS[criterion])


def half_width(*, interval: Interval) -> float:
    """Return half the width of an interval."""
    return (interval["hi"] - interval["lo"]) / 2.0


def off_spell_rows(*, flag: np.ndarray, min_length: int) -> np.ndarray:
    """Return the rows inside runs of False in `flag` that are at least `min_length` long."""
    flag = np.asarray(flag, dtype=bool)
    out = np.zeros(len(flag), dtype=bool)
    start = 0
    for i in range(1, len(flag) + 1):
        if i == len(flag) or flag[i] != flag[start]:
            if not flag[start] and i - start >= min_length:
                out[start:i] = True
            start = i
    return out


def off_spell_lengths(*, flag: np.ndarray, min_length: int) -> list[int]:
    """Return the lengths of the runs of False in `flag` that are at least `min_length` long."""
    flag = np.asarray(flag, dtype=bool)
    lengths: list[int] = []
    start = 0
    for i in range(1, len(flag) + 1):
        if i == len(flag) or flag[i] != flag[start]:
            if not flag[start] and i - start >= min_length:
                lengths.append(i - start)
            start = i
    return lengths


def whole_day_rows(*, rows: np.ndarray, local_date: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Return the rows of every local date whose half-hours are all in `rows` and all observed.

    Args:
        rows: A boolean mask, such as the rows inside off-spells.
        local_date: The local date of each row, as a number.
        observed: Whether each row's reading is present.
    """
    dates, inverse = np.unique(local_date, return_inverse=True)
    total = np.bincount(inverse, minlength=len(dates))
    good = np.bincount(inverse, weights=(rows & observed).astype(float), minlength=len(dates))
    return (good == total)[inverse]

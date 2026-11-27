# ruff: noqa: E501
"""Experiment 733: the synthetic forecasting world built from the real files.

This module turns experiment 730's banks into `e00733_synth.SegmentArrays`: one per segment, on the
segment's dense axis of calendar days (padding days included). The pure array arithmetic of the
world lives in `e00733_synth`; this module reads the files:

- the history weather and date-time columns of D8 on experiment 422's grid, standardised as
  experiment 713's fold of 2025-11 standardises them (`weather_cache`, built once through the real
  pipeline of experiment 713 and stored in `world/weather_d8.npz`);
- the ensemble rows of every origin: the 00Z run of the origin's own base date, from the run
  summaries of experiments 607 and 707 (`RunCache`), standardised with the same fold's constants;
- the untangled channels of every segment day, from the frozen networks of experiment 732 (the
  stored outputs of days up to D - 3) and from the re-run of days D - 2 and D - 1 with every input
  after the visibility limit hidden (`attach_channels`).

A base date without a 00Z run is a `coverage_gap` (data, never an error): the origin is not covered
and the segment is left out of the world when no origin is covered.

The target-row columns that experiment 713 builds from observed weather take, in the synthetic
world, the values of the origin day's own start (00:00 UTC on the base date), so that every
target-row column other than the ensemble run reads a valid time before the native origin. No
function prints a date.
"""

import json
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import e00733_channels as ch
import e00733_core as core
import e00733_rerun as rr
import e00733_synth as sy
import numpy as np
from e00733_leakage import CorrectnessFailureError

GRID_START_US: Final = 1_567_296_000_000_000
STEP_US: Final = 1_800_000_000
US_PER_DAY: Final = 86_400 * 1_000_000
CLIP_SD: Final = 5.0
"""Experiment 397's input clip, in standard deviations."""
N_ENS_WEATHER: Final = 7
WEATHER_NAMES: Final = (
    "shortwave", "longwave", "temperature_2m", "dew_point_temperature_2m", "pressure_surface",
    "precipitation", "wind_speed_10m",
)  # fmt: skip
"""The ensemble summaries' seven weather variables, in experiment 87's order."""
SHORTWAVE_COLUMN: Final = 0
SKY_COLUMN: Final = 0
"""Column 0 of the weather columns is the shortwave (the ensemble's in a target row, the observed
sky in a history token)."""
SUN_UP_WATTS: Final = 10.0
"""A shortwave above this many watts per square metre counts as sun-up."""
DAY_US: Final = 86_400_000_000
WORLD_DIR: Final = "world"
WEATHER_FILE: Final = "weather_d8.npz"
META_FILE: Final = "meta.json"
KINDS: Final = ("train_s3", "train_s4", "validation", "test")
"""The banks of the synthetic forecasting world."""


class WorldError(CorrectnessFailureError):
    """The synthetic world met an input it cannot use (a correctness failure, verdict row 1)."""


def world_dir(*, root: Path) -> Path:
    """Return (and create) the folder of the world's cached files."""
    path = root / WORLD_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- the weather cache


@dataclass(frozen=True)
class WeatherCache:
    """D8's weather and date-time columns on experiment 422's grid, and the fold's constants.

    Attributes:
        weather: `(G, 21)` float16: the seven weather columns, the twelve valid-time B1 columns, and
            the two CAMS day columns, standardised (experiment 713's `dense_weather`).
        datetime: `(G, 21)` float16 the date-time columns (experiment 713's `dense_datetime`).
        day_type: `(G,)` int8 the day type of each grid half-hour.
        local_hh: `(G,)` int8 the local half-hour of the day (0 to 47) of each half-hour.
        local_month: `(G,)` int8 the local calendar month.
        mean: `(14,)` the mean of the seven weather and seven spread columns of an ensemble row.
        std: `(14,)` their standard deviation.
    """

    weather: np.ndarray
    datetime: np.ndarray
    day_type: np.ndarray
    local_hh: np.ndarray
    local_month: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @property
    def n_grid(self) -> int:
        """Return the number of grid half-hours."""
        return int(self.weather.shape[0])


def local_time_columns(*, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the local half-hour of the day and the local month of every grid half-hour.

    The stamp is the period end, so the local half-hour is that of the period's start (stamp minus
    30 minutes) on the clock of Europe/London.
    """
    import polars as pl

    start_us = GRID_START_US + (np.arange(n_grid, dtype=np.int64) - 1) * STEP_US
    series = pl.Series(start_us).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    local = series.dt.convert_time_zone("Europe/London")
    hh = (local.dt.hour() * 2 + local.dt.minute() // 30).to_numpy().astype(np.int8)
    return hh, local.dt.month().to_numpy().astype(np.int8)


def write_weather_cache(*, root: Path, data: Any, standardiser: Any, d8_index: int) -> Path:
    """Write the weather cache from experiment 713's fold data and its row standardiser.

    Args:
        root: This experiment's data folder.
        data: Experiment 713's `FoldData` of one fold (its source holds the dense arrays).
        standardiser: The fold's `Standardiser` over the 37 input columns.
        d8_index: The Primary index of D8.

    Returns:
        The path of `world/weather_d8.npz`.
    """
    src = data.source
    weather = src.dense_weather[d8_index].cpu().numpy().astype(np.float16)
    dtime = src.dense_datetime[d8_index].cpu().numpy().astype(np.float16)
    day_type = src.day_type.cpu().numpy().astype(np.int8)
    day_type = day_type[d8_index] if day_type.ndim == 2 else day_type
    n_grid = weather.shape[0]
    hh, month = local_time_columns(n_grid=n_grid)
    mean = np.asarray(standardiser.mean[:14], dtype=np.float64)
    std = np.asarray(standardiser.std[:14], dtype=np.float64)
    path = world_dir(root=root) / WEATHER_FILE
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp, weather=weather, datetime=dtime, day_type=day_type, local_hh=hh, local_month=month,
        mean=mean, std=std,
    )  # fmt: skip
    tmp.replace(path)
    return path


def read_weather_cache(*, root: Path) -> WeatherCache:
    """Read `world/weather_d8.npz`."""
    with np.load(world_dir(root=root) / WEATHER_FILE) as z:
        return WeatherCache(**{k: z[k] for k in z.files})


# --------------------------------------------------------------------------- ensemble rows


@dataclass(frozen=True)
class EnsRows:
    """One 00Z run's standardised ensemble rows at D8's cell.

    Attributes:
        valid_us: `(n,)` the valid-time stamps (period ends) in microseconds, sorted.
        weather: `(n, 7)` the standardised ensemble-mean weather columns.
        spread: `(n, 7)` the standardised ensemble spread columns.
        run_init_us: The run's initialisation time (00:00 UTC of the file's date) in microseconds.
        run_avail_us: The file's own `power_fcst_init_time` column in microseconds: the origin the
            summary was built for, so the time at which the run is available.
    """

    valid_us: np.ndarray
    weather: np.ndarray
    spread: np.ndarray
    run_init_us: int
    run_avail_us: int


def read_ens_rows(
    *, summary_dir: Path | Sequence[Path], run_epoch_day: int, series_id: int, cache: WeatherCache
) -> EnsRows | None:
    """Read and standardise one run's rows; None when the run has no summary (a coverage gap).

    Precipitation goes through `log1p` as experiment 397's feature matrix does, and every column is
    standardised with the fold's constants, clipped at 5 standard deviations, and NaN becomes 0.
    """
    from datetime import date, timedelta

    import polars as pl

    day = date(1970, 1, 1) + timedelta(days=run_epoch_day)
    folders = [summary_dir] if isinstance(summary_dir, Path) else list(summary_dir)
    path = next(
        (
            f / f"init={day.isoformat()}.parquet"
            for f in folders
            if (f / f"init={day.isoformat()}.parquet").exists()
        ),
        None,
    )
    if path is None:
        return None
    wanted = [f"{k}_{w}" for k in ("mean", "std") for w in WEATHER_NAMES]
    frame = (
        pl.read_parquet(
            path, columns=["time_series_id", "power_fcst_init_time", "valid_time", *wanted]
        )
        .filter(pl.col("time_series_id") == series_id)
        .sort("valid_time")
    )
    if frame.is_empty():
        return None
    avail = frame["power_fcst_init_time"].dt.epoch("us").unique().to_numpy().astype(np.int64)
    if len(avail) != 1:
        msg = f"the summary of run {day} holds {len(avail)} power_fcst_init_time values, not one"
        raise WorldError(msg)
    raw = np.stack([frame[c].to_numpy().astype(np.float64) for c in wanted], axis=1)
    precip = [
        WEATHER_NAMES.index("precipitation"),
        len(WEATHER_NAMES) + WEATHER_NAMES.index("precipitation"),
    ]
    raw[:, precip] = np.log1p(np.clip(raw[:, precip], 0.0, None))
    z = np.nan_to_num(np.clip((raw - cache.mean) / cache.std, -CLIP_SD, CLIP_SD), nan=0.0)
    return EnsRows(
        valid_us=frame["valid_time"].dt.epoch("us").to_numpy().astype(np.int64),
        weather=z[:, :N_ENS_WEATHER].astype(np.float32), spread=z[:, N_ENS_WEATHER:].astype(np.float32),
        run_init_us=int(run_epoch_day) * DAY_US, run_avail_us=int(avail[0]),
    )  # fmt: skip


@dataclass
class RunCache:
    """A cache of standardised ensemble runs by epoch day (None for a missing run)."""

    summary_dir: Path | Sequence[Path]
    series_id: int
    weather: WeatherCache
    runs: dict[int, EnsRows | None] = field(default_factory=dict)

    def get(self, *, epoch_day: int) -> EnsRows | None:
        """Return the run of one epoch day."""
        if epoch_day not in self.runs:
            self.runs[epoch_day] = read_ens_rows(
                summary_dir=self.summary_dir, run_epoch_day=epoch_day, series_id=self.series_id,
                cache=self.weather,
            )  # fmt: skip
        return self.runs[epoch_day]


# --------------------------------------------------------------------------- one segment


def epoch_of(*, date64: np.datetime64) -> int:
    """Return the epoch day of a `datetime64[D]`."""
    return int(np.asarray(date64).astype("datetime64[D]").astype(np.int64))


def grid_of_slots(*, first_epoch_day: int, n_days: int) -> np.ndarray:
    """Return the grid index (the period-end stamp's index) of each dense slot of a segment."""
    return (
        (first_epoch_day - ch.GRID_START_EPOCH_DAY) * core.STEPS_PER_DAY
        + np.arange(n_days * core.STEPS_PER_DAY, dtype=np.int64)
        + 1
    )


def reading_constants(*, segments: Sequence[Any]) -> dict[str, float]:
    """Return the mean and standard deviation of D8's folded reading, in fractions of capacity.

    Read over the valid base half-hours of the given segments (the training banks of seeds 3 and 4
    when the world is built, so no validation or test half-hour enters).
    """
    total = 0.0
    sq = 0.0
    n = 0
    for s in segments:
        v = np.asarray(s.valid) & np.isfinite(s.r)
        r = np.asarray(s.r)[v].astype(np.float64) / 100.0
        total += float(r.sum())
        sq += float((r * r).sum())
        n += int(r.size)
    if n == 0:
        msg = "no valid half-hour in the training banks, so the reading constants cannot be counted"
        raise WorldError(msg)
    mean = total / n
    return {"mean": mean, "std": float(np.sqrt(max(sq / n - mean * mean, 1e-12))), "n": float(n)}


def build_segment(  # noqa: PLR0915
    *, seg: Any, runs: RunCache, run_days: frozenset[Any] | set[int], constants: Mapping[str, float]
) -> sy.SegmentArrays:
    """Build one segment's arrays (the untangled channels are attached later).

    Args:
        seg: Experiment 729's simulated `Segment` (dates are consecutive, padding days included).
        runs: The ensemble run cache.
        run_days: The epoch days that have a 00Z run summary.
        constants: The reading's `mean` and `std` (fractions of capacity).

    Returns:
        The segment's arrays: `covered` marks the base days whose own 00Z run exists.

    Raises:
        WorldError: When the segment's dates are not consecutive days.
    """
    dates = np.asarray(seg.dates).astype("datetime64[D]")
    if len(dates) > 1 and not (np.diff(dates.astype(np.int64)) == 1).all():
        msg = "a segment's dates are not consecutive days"
        raise WorldError(msg)
    n = len(dates)
    steps = core.STEPS_PER_DAY
    wx = runs.weather
    first = epoch_of(date64=dates[0])
    grid = grid_of_slots(first_epoch_day=first, n_days=n)
    if grid.min() < 0 or grid.max() >= wx.n_grid:
        msg = "a segment lies outside experiment 422's grid"
        raise WorldError(msg)
    present = np.asarray(seg.cal_index) >= 0
    epoch = dates.astype(np.int64)
    covered = present & np.isin(epoch, np.fromiter(run_days, dtype=np.int64))
    valid = np.asarray(seg.valid) & np.isfinite(seg.r) & np.isfinite(seg.x)
    r_frac = np.asarray(seg.r, dtype=np.float64) / 100.0
    reading_z = np.where(valid, (r_frac - constants["mean"]) / constants["std"], np.nan).reshape(-1)
    x_neg = np.where(valid, (np.asarray(seg.x) < 0).astype(np.float64), np.nan).reshape(-1)
    top_weather = np.zeros((n, sy.TOP_ROWS, steps, sy.N_WEATHER_COLUMNS), dtype=np.float16)
    top_spread = np.zeros((n, sy.TOP_ROWS, steps, sy.N_SPREAD_COLUMNS), dtype=np.float16)
    top_datetime = np.zeros((n, sy.TOP_ROWS, steps, sy.N_WEATHER_COLUMNS), dtype=np.float16)
    top_day_type = np.zeros((n, sy.TOP_ROWS, steps), dtype=np.int8)
    top_filled = np.zeros((n, sy.TOP_ROWS, steps), dtype=bool)
    run_init = np.zeros(n, dtype=np.int64)
    run_avail = np.zeros(n, dtype=np.int64)
    other_valid = np.zeros(n, dtype=np.int64)
    c_grid = np.zeros(n, dtype=np.int64)
    for d in np.flatnonzero(covered):
        run = runs.get(epoch_day=int(epoch[d]))
        if run is None:
            covered[d] = False
            continue
        c = int(d) * steps + 24
        stamp = c + np.arange(sy.TOP_ROWS)[:, None] * steps + np.arange(steps)[None, :]
        g = (first - ch.GRID_START_EPOCH_DAY) * steps + stamp
        inside = (g >= 0) & (g < wx.n_grid)
        gc = np.clip(g, 0, wx.n_grid - 1)
        valid_us = GRID_START_US + g.astype(np.int64) * STEP_US
        pos = np.clip(np.searchsorted(run.valid_us, valid_us), 0, len(run.valid_us) - 1)
        found = (run.valid_us[pos] == valid_us) & inside
        top_weather[d, ..., :N_ENS_WEATHER] = np.where(found[..., None], run.weather[pos], 0.0)
        top_spread[d] = np.where(found[..., None], run.spread[pos], 0.0)
        day_start = (first - ch.GRID_START_EPOCH_DAY + int(d)) * steps
        top_filled[d] = found
        run_init[d], run_avail[d] = run.run_init_us, run.run_avail_us
        other_valid[d] = GRID_START_US + day_start * STEP_US
        c_grid[d] = day_start + 24
        top_weather[d, ..., N_ENS_WEATHER:] = wx.weather[day_start, N_ENS_WEATHER:]
        top_datetime[d] = np.where(inside[..., None], wx.datetime[gc], 0.0)
        top_day_type[d] = np.where(inside, wx.day_type[gc], 0)
    month = wx.local_month[np.minimum(grid.reshape(n, steps)[:, 24], wx.n_grid - 1)].astype(
        np.int64
    )
    return sy.SegmentArrays(
        present=present, covered=covered, reading_z=reading_z.astype(np.float32),
        x_neg=x_neg.astype(np.float32), solar_points=np.asarray(seg.solar, dtype=np.float32).reshape(-1),
        u_raw=np.zeros((n * steps, core.N_CHANNELS), dtype=np.float32), u_ok=np.zeros(n * steps, dtype=bool),
        hist_weather=wx.weather[grid].astype(np.float32), hist_datetime=wx.datetime[grid].astype(np.float32),
        day_type=wx.day_type[grid].astype(np.int64), local_hh=wx.local_hh[grid].astype(np.int64), month=month,
        top_weather=top_weather, top_spread=top_spread, top_datetime=top_datetime,
        top_day_type=top_day_type, top_run_init_us=run_init, top_run_avail_us=run_avail,
        top_other_valid_us=other_valid, top_c_grid=c_grid, top_filled=top_filled,
    )  # fmt: skip


# --------------------------------------------------------------------------- the untangled channels


@dataclass
class ChannelEnv:
    """What the channel builder needs from experiments 729 and 732.

    Attributes:
        arms: Experiment 732's `e00732_arms` module.
        spec: Experiment 732's arm specification of `U*`.
        stats: Experiment 729's `InputStats`.
        holidays: Bank-holiday dates.
        medians: The training median of each folded neighbour.
        neighbours: The neighbours' series on D8's base calendar.
        predict: Windows `(n, C, 96)` to the decoded outputs `(seeds, n, 3, 48)`.
        n_seeds: The number of seeds.
    """

    arms: Any
    spec: Any
    stats: Any
    holidays: np.ndarray
    medians: Mapping[str, float]
    neighbours: Mapping[str, np.ndarray]
    predict: Callable[[np.ndarray], np.ndarray]
    n_seeds: int


def attach_channels(
    *, arrays: sy.SegmentArrays, seg: Any, prepared: Any, env: ChannelEnv, runs: RunCache,
    run_days: set[int],
) -> dict[str, int]:  # fmt: skip
    """Fill `u_raw`, `u_ok`, `u_rerun`, and `u_rerun_ok` of one segment from the frozen networks.

    The stored outputs are those of the windows of every present day (nothing hidden); the re-run
    outputs of an origin on day D are those of days D - 2 and D - 1 with every input after the
    visibility limit hidden. An origin whose runs of days D - 1 and D are not both available has no
    re-run, and its two history days are flagged absent.

    Args:
        arrays: The segment's arrays, filled in place.
        seg: The 730 segment.
        prepared: The untangler's prepared inputs of the segment.
        env: The frozen networks and the 729 context.
        runs: The ensemble runs read so far.
        run_days: The epoch days that have an ensemble run.

    Returns:
        Counts: `windows`, `origins_rerun`, `origins_without_rerun`.
    """
    import e00733_real as rl

    steps = core.STEPS_PER_DAY
    n = arrays.n_days
    dates = np.asarray(seg.dates).astype("datetime64[D]")
    epoch = dates.astype(np.int64)
    days = [d for d in range(1, n - 1) if arrays.present[d]]
    counts = {"windows": 0, "origins_rerun": 0, "origins_without_rerun": 0}
    if days:
        w = np.stack(
            [env.arms.plain_window(p=prepared, day=d, spec=env.spec, hidden=None) for d in days]
        ).astype(np.float32)
        out = env.predict(w).mean(axis=0)  # (len(days), 3, 48)
        for i, d in enumerate(days):
            sl = slice(d * steps, (d + 1) * steps)
            arrays.u_raw[sl] = out[i].T
            arrays.u_ok[sl] = np.isfinite(arrays.reading_z[sl])
        counts["windows"] = len(days)
    arrays.u_rerun = np.zeros((n, 2, steps, core.N_CHANNELS), dtype=np.float32)
    arrays.u_rerun_ok = np.zeros((n, 2, steps), dtype=bool)
    origins = [
        int(d)
        for d in np.flatnonzero(arrays.covered)
        if d >= 3 and d <= n - 2 and int(epoch[d]) - 1 in run_days and runs.get(epoch_day=int(epoch[d]) - 1) is not None
    ]  # fmt: skip
    counts["origins_without_rerun"] = int(arrays.covered.sum()) - len(origins)
    if origins:
        windows = rl.RealWindows(
            prepared=prepared, spec=env.spec, first_epoch_day=int(epoch[1]), arms=env.arms,
            stats=env.stats, summary_dir=runs.summary_dir, series_id=runs.series_id,
        )  # fmt: skip
        grid_days = [int(epoch[d]) - ch.GRID_START_EPOCH_DAY for d in origins]
        got = rr.rerun_origins(
            origin_days=grid_days, build_window=windows.build_window, predict=env.predict,
            n_seeds=env.n_seeds,
        )  # fmt: skip
        for i, d in enumerate(origins):
            arrays.u_rerun[d] = got.values[i]
            arrays.u_rerun_ok[d] = got.ok[i]
        counts["origins_rerun"] = len(origins)
    return counts


# --------------------------------------------------------------------------- the files of the world


def world_path(*, root: Path, kind: str) -> Path:
    """Return the path of one cached bank of the world."""
    return world_dir(root=root) / f"{kind}.pkl"


def write_world(
    *, root: Path, kind: str, segments: Sequence[sy.SegmentArrays], extra: Mapping[str, Any]
) -> Path:
    """Write the cached segments of one bank atomically."""
    path = world_path(root=root, kind=kind)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(
            {"segments": list(segments), "extra": dict(extra)}, f, protocol=pickle.HIGHEST_PROTOCOL
        )
    tmp.replace(path)
    return path


def read_world(*, root: Path, kind: str) -> tuple[list[sy.SegmentArrays], dict[str, Any]]:
    """Read one cached bank of the world (written by `write_world` of this code)."""
    with world_path(root=root, kind=kind).open("rb") as f:
        blob = pickle.load(f)
    return blob["segments"], blob["extra"]


def write_meta(*, root: Path, payload: Mapping[str, Any]) -> None:
    """Write `world/meta.json` (the reading constants and the counts)."""
    (world_dir(root=root) / META_FILE).write_text(json.dumps(payload, indent=1, default=float))


def read_meta(*, root: Path) -> dict[str, Any]:
    """Read `world/meta.json`."""
    return json.loads((world_dir(root=root) / META_FILE).read_text())


def sun_up_threshold(*, cache: WeatherCache) -> float:
    """Return the standardised shortwave above which a half-hour counts as sun-up (10 W/m2)."""
    return float((SUN_UP_WATTS - cache.mean[SHORTWAVE_COLUMN]) / cache.std[SHORTWAVE_COLUMN])


# --------------------------------------------------------------------------- one bank


def make_channel_env(
    *, root_732: Path, root_730: Path, log: Callable[[str], None] = print
) -> tuple[ChannelEnv, Any]:
    """Return the channel environment of experiments 729 and 732 and experiment 729's context.

    Builds the context from the real files (CPU), reads the neighbours and their medians that
    experiment 732 stored, and loads the three networks of the headline untangler `U*`.
    """
    import e00729_stages as stages729
    import e00732_arms as arms
    import e00732_cli as cli732
    import e00732_core as core732
    import e00732_stages as st732

    ctx = stages729.build_context()
    base = cli732.real_base(ctx=ctx)
    neighbours, medians = st732.read_neighbour_inputs(root=root_732, base=base)
    nets = [
        rr.load_network(source=core.SOURCE_732, seed=s, root_732=root_732, root_730=root_730, device="cpu")
        for s in core.SOURCE_SEEDS
    ]  # fmt: skip

    def predict(w: np.ndarray) -> np.ndarray:
        return rr.predict_windows(nets=nets, windows=w, source=core.SOURCE_732, n_base=w.shape[1])

    log(f"channel environment ready: {len(nets)} networks")
    env = ChannelEnv(
        arms=arms, spec=core732.ARMS["U*"], stats=ctx.stats, holidays=ctx.cal.holidays,
        medians=medians, neighbours=neighbours, predict=predict, n_seeds=len(nets),
    )  # fmt: skip
    return env, ctx


def build_bank_world(
    *, bank: Any, env: ChannelEnv, runs: RunCache, run_days: set[int], constants: Mapping[str, float],
    limit: int | None = None, log: Callable[[str], None] = print,
) -> tuple[list[sy.SegmentArrays], dict[str, int]]:  # fmt: skip
    """Build the arrays of every kept segment of one bank, with their untangled channels.

    A segment with no covered origin is left out. `limit` keeps only the first segments (the CPU
    smoke test uses it; the official run passes None).

    Args:
        bank: The 730 bank.
        env: The frozen networks and the 729 context.
        runs: The ensemble runs read so far.
        run_days: The epoch days that have an ensemble run.
        constants: The reading's mean and standard deviation.
        limit: The number of kept segments after which to stop, or None.
        log: Receives progress lines.

    Returns:
        The kept segments and the counts (segments, left out, covered origins, windows, re-runs).
    """
    segments = list(bank.segments)
    counts = {"n_segments": len(segments), "n_left_out": 0, "n_origins_covered": 0, "windows": 0, "origins_rerun": 0, "origins_without_rerun": 0}  # fmt: skip
    prepared = env.arms.prepare_plain(
        segments=segments, holidays=env.holidays, stats=env.stats, medians=env.medians, neighbours=env.neighbours,
    )  # fmt: skip
    out: list[sy.SegmentArrays] = []
    for i, (seg, prep) in enumerate(zip(segments, prepared, strict=True)):
        if limit is not None and len(out) >= limit:
            break
        arrays = build_segment(seg=seg, runs=runs, run_days=run_days, constants=constants)
        if not arrays.covered.any():
            counts["n_left_out"] += 1
            continue
        got = attach_channels(
            arrays=arrays, seg=seg, prepared=prep, env=env, runs=runs, run_days=run_days
        )
        arrays.check_shapes()
        for k, v in got.items():
            counts[k] += v
        counts["n_origins_covered"] += int(arrays.covered.sum())
        out.append(arrays)
        if (i + 1) % 100 == 0:
            log(f"  {i + 1} of {len(segments)} segments")
    return out, counts

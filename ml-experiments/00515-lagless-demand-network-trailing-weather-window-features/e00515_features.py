# ruff: noqa: E501
r"""Trailing weather-window columns for experiment 515, for ensemble rows and ERA5-and-CAMS rows.

Nine columns, in `ALL_COLUMNS` order, grouped into the arms of the plan:

- `W-T`: means of temperature over the 6, 24, and 72 hours ending at the valid time;
- `W-S`: means of shortwave over the 6 and 24 hours ending at the valid time, and over the 24 hours
  ending 24 hours before the valid time;
- `W-DH`: heating degree-hours below 15.5 C and cooling degree-hours above 20 C over 24 hours (the
  mean of the excess over the 48 half-hours, times 24 hours);
- `W-X`: (24-hour mean temperature minus 15.5 C) times the 24-hour mean shortwave.

**One window rule at every lead.** Every window is built on the half-hourly series, which is the
linear interpolation of the weather model's steps: 3-hourly up to 144 hours after the run's 00 UTC
start and 6-hourly after that.

- *Ensemble rows* (`ensemble_table`): the series of a run `O` (09:00 UTC) holds the run's own
  ensemble mean after `O`, and the first day of the freshest earlier run for half-hours at or before
  `O` (experiment 437's "analysis proxy", leads of about 9.5 to 33.5 hours). No later run is used.
  With `seam="persistence"` every half-hour at or before `O` holds the run's first value instead
  (the seam control).
- *ERA5-and-CAMS rows* (`DenseWindows`): ERA5 temperature (the `:00` rows) and CAMS shortwave
  (hour-end stamps) are subsampled to the step grid (3-hourly UTC stamps up to the run's 144th hour,
  6-hourly after) and linearly interpolated to half-hours, so a pretraining window is as smooth as a
  serving window.

**Missing earlier runs.** `diag_missed_runs` counts, among the three days before a run's origin,
those whose run is absent from the summaries. A window that reaches before the first run (April
2024) uses what exists, and `diag_short_72h` marks it.

Commands (run with `PYTHONPATH` as `e00515_fit.py` shows): `build` writes the table of every Primary
and seam variant; `check` runs the causality check (later runs removed) and range checks on one
Primary.
"""

import argparse
import time
from itertools import pairwise
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00515-lagless-demand-network-trailing-weather-window-features"
    / "work"
)
TABLE_DIR: Final[Path] = WORK_DIR / "windows"
ARM_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "T": ("t_mean_6h", "t_mean_24h", "t_mean_72h"),
    "S": ("sw_mean_6h", "sw_mean_24h", "sw_mean_prev_24h"),
    "DH": ("heat_dh_24h", "cool_dh_24h"),
    "X": ("t_sw_product_24h",),
}
ALL_COLUMNS: Final[tuple[str, ...]] = tuple(c for cols in ARM_COLUMNS.values() for c in cols)
DIAGNOSTICS: Final[tuple[str, ...]] = ("diag_missed_runs", "diag_short_72h")
SEAMS: Final[tuple[str, ...]] = ("previous", "persistence")
STEP_US: Final[int] = f437.STEP_US
STEPS_PER_DAY: Final[int] = 48
HEATING_BASE_C: Final[float] = 15.5
COOLING_BASE_C: Final[float] = 20.0
INIT_TO_ORIGIN_STEPS: Final[int] = 18
"""A run starts at 00 UTC and its origin is 09:00 UTC, 18 half-hours later."""
FINE_GRID_STEPS: Final[int] = 288
"""The 3-hourly steps last to 144 hours after the run's start, which is 288 half-hours."""
WINDOWS: Final[dict[str, int]] = {"6h": 12, "24h": 48, "72h": 144}
LOOKBACK_STEPS: Final[int] = 4 * STEPS_PER_DAY
"""History kept before the origin: the 72-hour window plus the previous-day shortwave window."""
PREVIOUS_RUN_DAYS: Final[int] = 3


def table_path(*, site: str, seam: str) -> Path:
    """Return where a Primary's ensemble-row window table is stored."""
    return TABLE_DIR / seam / f"{site}.parquet"


def _derived_series(*, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return heating and cooling excess of a temperature series (NaN stays NaN)."""
    heat = np.where(np.isfinite(t), np.maximum(HEATING_BASE_C - t, 0.0), np.nan)
    cool = np.where(np.isfinite(t), np.maximum(t - COOLING_BASE_C, 0.0), np.nan)
    return heat, cool


def run_columns(
    *, t_all: np.ndarray, s_all: np.ndarray, p: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the nine window columns at positions `p` of one run's combined series.

    Args:
        t_all: Temperature on the run's half-hourly series (NaN where nothing exists).
        s_all: Shortwave on the same series.
        p: Positions of the run's rows in the series; each must be at least 96 steps in.

    Returns:
        The columns by name, and a boolean per row that is True where the 72-hour temperature window
        held fewer than 144 half-hours.
    """
    out: dict[str, np.ndarray] = {}
    for label, n_steps in WINDOWS.items():
        out[f"t_mean_{label}"], found = f437.trailing_mean(a=t_all, n=n_steps, at=p)
        if label == "72h":
            short = found < n_steps
        if label != "72h":
            out[f"sw_mean_{label}"], _ = f437.trailing_mean(a=s_all, n=n_steps, at=p)
    previous, found_previous = f437.trailing_mean(a=s_all, n=STEPS_PER_DAY, at=p - STEPS_PER_DAY)
    out["sw_mean_prev_24h"] = np.where(found_previous > 0, previous, out["sw_mean_24h"])
    heat, cool = _derived_series(t=t_all)
    out["heat_dh_24h"] = f437.trailing_mean(a=heat, n=STEPS_PER_DAY, at=p)[0] * 24.0
    out["cool_dh_24h"] = f437.trailing_mean(a=cool, n=STEPS_PER_DAY, at=p)[0] * 24.0
    out["t_sw_product_24h"] = (out["t_mean_24h"] - HEATING_BASE_C) * out["sw_mean_24h"]
    return out, short


def ensemble_table(*, row: dict, seam: str) -> pl.DataFrame:
    """Return one Primary's window columns for every (run, valid time) of its ensemble summary.

    Args:
        row: The Primary's series row.
        seam: `"previous"` (freshest earlier run before the origin) or `"persistence"` (the run's
            first value held before the origin).

    Returns:
        A frame keyed by `origin_us` and `valid_us` with `ALL_COLUMNS` and `DIAGNOSTICS`.

    Raises:
        ValueError: If two runs' first days overlap on the grid, or `seam` is unknown.
    """
    if seam not in SEAMS:
        msg = f"unknown seam {seam!r}"
        raise ValueError(msg)
    summary = (
        prep.load_summary(series_id=int(row["time_series_id"]))
        .select(
            origin=pl.col("power_fcst_init_time").dt.epoch("us"),
            valid=pl.col("valid_time").dt.epoch("us"),
            temperature=pl.col("mean_temperature_2m").cast(pl.Float64),
            shortwave=pl.col("mean_shortwave").cast(pl.Float64),
        )
        .sort("origin", "valid")
    )
    origin_us, valid_us = summary["origin"].to_numpy(), summary["valid"].to_numpy()
    temperature, shortwave = summary["temperature"].to_numpy(), summary["shortwave"].to_numpy()
    oi, vi = f437.grid_of(us=origin_us), f437.grid_of(us=valid_us)
    size = common.grid_times().len()
    history_t, history_s = np.full(size, np.nan), np.full(size, np.nan)
    first_day = (vi - oi >= 1) & (vi - oi <= STEPS_PER_DAY)
    if len(np.unique(vi[first_day])) != int(first_day.sum()):
        msg = "two runs' first days overlap on the grid"
        raise ValueError(msg)
    history_t[vi[first_day]] = temperature[first_day]
    history_s[vi[first_day]] = shortwave[first_day]
    run_days = set(np.unique(oi // STEPS_PER_DAY).tolist())
    n_rows = len(summary)
    out = {c: np.full(n_rows, np.nan) for c in (*ALL_COLUMNS, *DIAGNOSTICS)}
    boundaries = np.flatnonzero(np.r_[True, oi[1:] != oi[:-1], True])
    for start, stop in pairwise(boundaries):
        rows = np.arange(start, stop)
        origin_index = int(oi[start])
        lo = origin_index - LOOKBACK_STEPS
        hi = int(vi[rows].max()) + 1
        t_all, s_all = history_t[lo:hi].copy(), history_s[lo:hi].copy()
        after = origin_index + 1 - lo
        t_all[after:], s_all[after:] = np.nan, np.nan  # nothing after the origin from a later run
        if seam == "persistence":
            s_first = summary["shortwave"].to_numpy()[rows]
            s_first = s_first[np.isfinite(s_first)]
            t_first = temperature[rows][np.isfinite(temperature[rows])]
            t_all[:after] = t_first[0]
            s_all[:after] = s_first[0] if len(s_first) else 0.0
        p = vi[rows] - lo
        t_all[p], s_all[p] = temperature[rows], shortwave[rows]
        columns, short = run_columns(t_all=t_all, s_all=s_all, p=p)
        for name, values in columns.items():
            out[name][rows] = values
        out["diag_short_72h"][rows] = short
        missed = sum(
            (origin_index // STEPS_PER_DAY - d) not in run_days
            for d in range(1, PREVIOUS_RUN_DAYS + 1)
        )
        out["diag_missed_runs"][rows] = missed
    return pl.DataFrame(
        {"origin_us": origin_us, "valid_us": valid_us}
        | {c: v.astype(np.float32) for c, v in out.items()}
    )


# --- ERA5-and-CAMS rows ---------------------------------------------------------------------


def _interpolated(*, hourly: np.ndarray, stamp_hour: np.ndarray, step_hours: int) -> np.ndarray:
    """Return a half-hourly series from `hourly` sampled at UTC stamps that are multiples of `step_hours`."""
    index = np.flatnonzero((stamp_hour % step_hours == 0) & np.isfinite(hourly))
    return np.interp(np.arange(len(hourly)), index, hourly[index])


def _prefix(*, a: np.ndarray) -> np.ndarray:
    """Return the zero-led cumulative sum, so a window sum is a difference of two entries."""
    return np.concatenate([[0.0], np.cumsum(a)])


class DenseWindows:
    """A Primary's ERA5-and-CAMS windows on the 3-hourly and 6-hourly step grids, as prefix sums."""

    def __init__(self, *, row: dict) -> None:
        """Build the subsampled series and their prefix sums for one Primary."""
        dense = prep418.dense_weather(series_id=int(row["time_series_id"]), site=row["site"])
        times = common.grid_times()
        on_hour = (times.dt.minute() == 0).to_numpy()
        hour = times.dt.hour().to_numpy()
        stamp_hour = np.where(on_hour, hour, -1)  # half-hour stamps never sit on a step grid
        temperature = dense[:, prep418.WEATHER.index("temperature_2m")].astype(np.float64)
        shortwave = dense[:, prep418.WEATHER.index("shortwave")].astype(np.float64)
        temperature = np.where(on_hour, temperature, np.nan)
        shortwave = np.where(on_hour, shortwave, np.nan)
        self.prefix: dict[tuple[str, str], np.ndarray] = {}
        for grid, step_hours in (("A", 3), ("B", 6)):
            t = _interpolated(hourly=temperature, stamp_hour=stamp_hour, step_hours=step_hours)
            s = _interpolated(hourly=shortwave, stamp_hour=stamp_hour, step_hours=step_hours)
            heat, cool = _derived_series(t=t)
            for name, series in (("t", t), ("s", s), ("heat", heat), ("cool", cool)):
                self.prefix[(grid, name)] = _prefix(a=series)

    def _window_mean(
        self, *, name: str, end: np.ndarray, n_steps: int, cut: np.ndarray
    ) -> np.ndarray:
        """Return the mean of the `n_steps` half-hours ending at `end`, grid A up to `cut` and grid B after."""
        pa, pb = self.prefix[("A", name)], self.prefix[("B", name)]
        start = np.maximum(end - n_steps + 1, 0)
        a_stop = np.minimum(end, cut)
        a_sum = np.where(start <= a_stop, pa[a_stop + 1] - pa[start], 0.0)
        b_start = np.maximum(start, cut + 1)
        b_sum = np.where(end >= b_start, pb[end + 1] - pb[b_start], 0.0)
        return (a_sum + b_sum) / np.maximum(end - start + 1, 1)

    def columns(self, *, origin_idx: np.ndarray, valid_idx: np.ndarray) -> np.ndarray:
        """Return the nine window columns of pseudo-runs, shape (rows, 9) float32, in `ALL_COLUMNS` order.

        Args:
            origin_idx: The grid index of each row's run origin (09:00 UTC).
            valid_idx: The grid index of each row's valid time.
        """
        cut = origin_idx - INIT_TO_ORIGIN_STEPS + FINE_GRID_STEPS
        out = {}
        for label, n_steps in WINDOWS.items():
            out[f"t_mean_{label}"] = self._window_mean(
                name="t", end=valid_idx, n_steps=n_steps, cut=cut
            )
            if label != "72h":
                out[f"sw_mean_{label}"] = self._window_mean(
                    name="s", end=valid_idx, n_steps=n_steps, cut=cut
                )
        out["sw_mean_prev_24h"] = self._window_mean(
            name="s", end=np.maximum(valid_idx - STEPS_PER_DAY, 0), n_steps=STEPS_PER_DAY, cut=cut
        )
        out["heat_dh_24h"] = (
            self._window_mean(name="heat", end=valid_idx, n_steps=STEPS_PER_DAY, cut=cut) * 24.0
        )
        out["cool_dh_24h"] = (
            self._window_mean(name="cool", end=valid_idx, n_steps=STEPS_PER_DAY, cut=cut) * 24.0
        )
        out["t_sw_product_24h"] = (out["t_mean_24h"] - HEATING_BASE_C) * out["sw_mean_24h"]
        return np.stack([out[c] for c in ALL_COLUMNS], axis=1).astype(np.float32)


# --- commands -------------------------------------------------------------------------------


def build_all() -> None:
    """Write the ensemble-row table of every Primary and seam variant that has none.

    The `persistence` variant is written for every Primary, but only its three `W-T` columns are
    used; the other columns are written too so that the table has one layout.
    """
    for seam in SEAMS:
        (TABLE_DIR / seam).mkdir(parents=True, exist_ok=True)
        for row in prep.primaries().iter_rows(named=True):
            target = table_path(site=row["site"], seam=seam)
            if target.exists():
                print(f"{seam} {row['site']}: exists", flush=True)
                continue
            started = time.monotonic()
            table = ensemble_table(row=row, seam=seam)
            bad = {
                c: int(table[c].null_count() + table[c].is_nan().sum())
                for c in ALL_COLUMNS
                if table[c].null_count() + table[c].is_nan().sum()
            }
            if bad:
                msg = f"{row['site']}: window columns with missing values: {bad}"
                raise ValueError(msg)
            temporary = target.with_name(target.name + ".tmp")
            table.write_parquet(temporary)
            temporary.rename(target)
            print(
                f"{seam} {row['site']}: {table.height:,} rows, {time.monotonic() - started:.0f} s",
                flush=True,
            )


def check(*, site: str = "D2") -> None:
    """Print range checks, rebuild sampled runs with every later run deleted, and compare grids.

    The causality check rebuilds the table from a summary truncated at a sampled run's origin and
    requires that run's rows to be identical. The grid check compares ERA5-window columns built by
    `DenseWindows` against the ensemble table's columns for the same (origin, valid) pairs, which
    differ only by the ensemble's forecast error.
    """
    row = next(r for r in prep.primaries().iter_rows(named=True) if r["site"] == site)
    table = pl.read_parquet(table_path(site=site, seam="previous"))
    print(table.select(*ALL_COLUMNS, *DIAGNOSTICS).describe())
    print(
        "rows with a 72 h window shorter than 144 half-hours:",
        int(table["diag_short_72h"].sum()),
        "of",
        table.height,
    )
    print(
        "rows by missed earlier runs:",
        table["diag_missed_runs"].value_counts().sort("diag_missed_runs").to_dicts(),
    )
    full = prep.load_summary(series_id=int(row["time_series_id"]))
    origins = full["power_fcst_init_time"].unique().sort()
    rng = np.random.default_rng(0)
    original = prep.load_summary
    for pick in rng.choice(len(origins) - 3, size=3, replace=False) + 8:
        cut_origin = origins[int(pick)]
        truncated = full.filter(pl.col("power_fcst_init_time") <= cut_origin)
        prep.load_summary = lambda *, series_id, t=truncated: t
        try:
            rebuilt = ensemble_table(row=row, seam="previous")
        finally:
            prep.load_summary = original
        key = int(cut_origin.timestamp() * 1_000_000)
        a = table.filter(pl.col("origin_us") == key).sort("valid_us")
        b = rebuilt.filter(pl.col("origin_us") == key).sort("valid_us")
        worst = max(
            float(np.nanmax(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in ALL_COLUMNS
        )
        print(f"run {cut_origin}: largest difference with all later runs deleted: {worst:.2e}")
        if worst > 1e-5:
            msg = "a window column depends on a later run"
            raise AssertionError(msg)
    dense = DenseWindows(row=row)
    sample = table.filter(pl.col("diag_short_72h") == 0).sample(n=20_000, seed=0)
    era5 = dense.columns(
        origin_idx=f437.grid_of(us=sample["origin_us"].to_numpy()),
        valid_idx=f437.grid_of(us=sample["valid_us"].to_numpy()),
    )
    diff = era5 - sample.select(ALL_COLUMNS).to_numpy()
    print("ERA5-window minus ensemble-window column, mean and standard deviation on 20,000 rows:")
    for j, name in enumerate(ALL_COLUMNS):
        print(f"  {name}: {np.nanmean(diff[:, j]):+.3f} {np.nanstd(diff[:, j]):.3f}")


def main() -> None:
    """Build the tables or run the checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--site", default="D2")
    args = parser.parse_args()
    if args.command == "build":
        build_all()
    else:
        check(site=args.site)


if __name__ == "__main__":
    main()

"""Row-level cloud (and positive-control temperature) inputs for the Primary demand arms of 711.

**The store holds every run's summaries as one array per Primary.** `work/raw_cloud/<site>.npz`
holds `values` of shape (runs, 702, 9): one slice per daily 00 UTC run from 2020-10-15 to
2026-06-30, one row per half-hour lead (0.5 to 351 hours after the run's origin), and the channels
of `CHANNELS`. The channels are the four cloud summaries of the cache (`e00711_cloud.py`), the
daylight mean of the ensemble-mean cloud (summary T3), and the mean and spread of 2 m temperature
and of 2 m dew point from experiments 707 and 397 (the positive control). A run that carries no
cloud (the 19-month gap and 2024-01-03) holds NaN in the cloud channels.

**A row finds its inputs by run and lead.** The row's run is its origin minus 9 hours, and its
lead step is its lead in hours times two, less one. The real arm reads the row's own run. A
placebo arm reads a donor run at the same step (`e00711_placebo` draws the donor map). The missing
pattern is the run-date rule, so a real row and a placebo row are missing together, and a missing
row holds NaN. Standardisation skips NaN, and a NaN input becomes 0 after standardisation, which is
the plan's mask (section 4).

This module imports no `torch` and reads files only in `build` and `load_store`.

```bash
# environment as in e00711_common.py; one thread
POLARS_MAX_THREADS=1 uv run --no-sync python ml-experiments/00711-*/e00711_rowcloud.py build
```
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00711_arms as arms
import e00711_common as common
import e00711_placebo as placebo
import numpy as np
import polars as pl

CHANNELS: Final[tuple[str, ...]] = (
    "cloud_mean",
    "cloud_spread",
    "cloud_clear_share",
    "cloud_overcast_share",
    "cloud_daylight_mean",
    "temp_mean",
    "temp_std",
    "dew_mean",
    "dew_std",
)
"""The store's channels. The first four are the cache's; the daylight mean is summary T3."""
SUMMARY_TEMPERATURE_COLUMNS: Final[dict[str, str]] = {
    "temp_mean": "mean_temperature_2m",
    "temp_std": "std_temperature_2m",
    "dew_mean": "mean_dew_point_temperature_2m",
    "dew_std": "std_dew_point_temperature_2m",
}
"""The channel of the positive control and the column of experiments 707 and 397 it comes from."""
STEPS: Final[int] = 702
"""Half-hour leads 0.5 to 351 hours after the origin."""
LOCAL_TIME_ZONE: Final[str] = "Europe/London"
STORE_DIR: Final[Path] = common.WORK_DIR / "work" / "raw_cloud"
US_PER_DAY: Final[int] = 86_400 * 1_000_000
STEP_TOLERANCE: Final[float] = 1e-6


def channel_indices(*, columns: tuple[str, ...]) -> list[int]:
    """Return the store channel of each named column.

    Raises:
        ValueError: If a column is not a channel.
    """
    unknown = [c for c in columns if c not in CHANNELS]
    if unknown:
        msg = f"not channels of the store: {unknown}"
        raise ValueError(msg)
    return [CHANNELS.index(c) for c in columns]


class RunArrays:
    """One Primary's summaries of every run, as an array indexed by run day and lead step.

    Attributes:
        first_day: The day number (days since the epoch) of `values[0]`.
        values: Shape (runs, `STEPS`, `len(CHANNELS)`), float32, NaN where a run has no value.
    """

    def __init__(self, *, first_day: int, values: np.ndarray) -> None:
        """Hold the array; it must have the shape of the store."""
        if values.ndim != 3 or values.shape[1:] != (STEPS, len(CHANNELS)):
            msg = f"values must have shape (runs, {STEPS}, {len(CHANNELS)}), not {values.shape}"
            raise ValueError(msg)
        self.first_day = int(first_day)
        self.values = values

    @property
    def n_runs(self) -> int:
        """Return how many daily runs the array holds."""
        return int(self.values.shape[0])

    def take(self, *, run_days: np.ndarray, steps: np.ndarray, channels: list[int]) -> np.ndarray:
        """Return `values[run, step, channels]` for each row, shape (rows, len(channels)).

        Raises:
            ValueError: If a run day lies outside the array.
        """
        index = run_days - self.first_day
        if ((index < 0) | (index >= self.n_runs)).any():
            msg = "a row's run lies outside the store"
            raise ValueError(msg)
        return self.values[index[:, None], steps[:, None], np.array(channels)[None, :]]


def row_run_days(*, nwp_init: np.ndarray) -> np.ndarray:
    """Return each row's run as a day number, from its run time in microseconds since the epoch.

    Raises:
        ValueError: If a run is not a 00:00 UTC run.
    """
    if (nwp_init % US_PER_DAY != 0).any():
        msg = "a row's run is not a 00:00 UTC run"
        raise ValueError(msg)
    return (nwp_init // US_PER_DAY).astype(np.int64)


def row_steps(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return each row's lead step: its lead in hours times two, less one (0 to 701).

    Raises:
        ValueError: If a lead is not a whole number of half-hours between 0.5 and 351 hours.
    """
    doubled = np.asarray(lead_hours, dtype=np.float64) * 2.0
    rounded = np.rint(doubled)
    if (np.abs(doubled - rounded) > STEP_TOLERANCE).any() or (rounded < 1).any():
        msg = "a lead is not a whole number of half-hours after the origin"
        raise ValueError(msg)
    steps = rounded.astype(np.int64) - 1
    if (steps >= STEPS).any():
        msg = f"a lead is beyond {STEPS / 2 - 0.5} hours"
        raise ValueError(msg)
    return steps


def missing_run_days() -> np.ndarray:
    """Return the day numbers of the runs that carry no cloud, from 2020-10-15 to 2026-06-30."""
    runs = common.runs_between(first=common.CACHE_FIRST_RUN, last=common.SCORED_LAST_RUN)
    return np.array(
        [placebo.day_number(run=r) for r in runs if common.is_missing_run(run=r)], dtype=np.int64
    )


def missing_flag(*, run_days: np.ndarray) -> np.ndarray:
    """Return each row's `cloud_missing` flag from its run date alone (never from a cloud value).

    Raises:
        ValueError: If a run lies outside 2020-10-15 to 2026-06-30.
    """
    low = placebo.day_number(run=common.CACHE_FIRST_RUN)
    high = placebo.day_number(run=common.SCORED_LAST_RUN)
    if ((run_days < low) | (run_days > high)).any():
        msg = "a row's run lies outside 2020-10-15 to 2026-06-30"
        raise ValueError(msg)
    return np.isin(run_days, missing_run_days())


def real_extras(
    *, store: RunArrays, run_days: np.ndarray, steps: np.ndarray, columns: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the real summaries of each row's own run, and the row's `cloud_missing` flag.

    A flagged row holds NaN in every column, including a temperature column of the positive
    control, so the control's inputs have the cloud's missing pattern.

    Args:
        store: The Primary's `RunArrays`.
        run_days: Each row's run, from `row_run_days`.
        steps: Each row's lead step, from `row_steps`.
        columns: The channels to read.

    Returns:
        The values, shape (rows, len(columns)), and the flag, shape (rows,).
    """
    flag = missing_flag(run_days=run_days)
    values = store.take(run_days=run_days, steps=steps, channels=channel_indices(columns=columns))
    values[flag] = np.nan
    return values, flag


def donor_extras(
    *,
    store: RunArrays,
    run_days: np.ndarray,
    steps: np.ndarray,
    columns: tuple[str, ...],
    donors: np.ndarray,
    donors_first_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the placebo summaries: each row reads its run's donor run at the same lead step.

    Args:
        store: The Primary's `RunArrays`.
        run_days: Each row's run, from `row_run_days`.
        steps: Each row's lead step, from `row_steps`.
        columns: The channels to read.
        donors: The donor day of each run of the map (`placebo.UNMAPPED` where the run carries no
            cloud), indexed by run day minus `donors_first_day`.
        donors_first_day: The day number of the map's first run.

    Returns:
        The values (NaN where the donor is `UNMAPPED`), and the placebo's missing flag.

    Raises:
        ValueError: If a row's run lies outside the map.
    """
    index = run_days - donors_first_day
    if ((index < 0) | (index >= len(donors))).any():
        msg = "a row's run lies outside the placebo map"
        raise ValueError(msg)
    donor_days = donors[index]
    flag = donor_days == placebo.UNMAPPED
    safe = np.where(flag, store.first_day, donor_days)
    values = store.take(run_days=safe, steps=steps, channels=channel_indices(columns=columns))
    values[flag] = np.nan
    return values, flag


def assert_same_flag(*, real_flag: np.ndarray, placebo_flag: np.ndarray) -> None:
    """Assert that the real and the placebo `cloud_missing` flags are equal on every row.

    Raises:
        ValueError: If a row is missing in one and not in the other.
    """
    if (real_flag != placebo_flag).any():
        msg = f"{int((real_flag != placebo_flag).sum())} rows differ in real and placebo flag"
        raise ValueError(msg)


def assert_missing_is_nan(*, values: np.ndarray, flag: np.ndarray) -> None:
    """Assert that a row's values are all NaN if and only if the row is flagged.

    A present run whose store slice holds a NaN (a series missing from a cache file, or a missing
    seven-variable summary) would otherwise become 0 after standardisation with `cloud_missing` = 0,
    so the network could not tell the hole from the mean.

    Raises:
        ValueError: If a flagged row holds a value, or an unflagged row holds a NaN.
    """
    nan_rows = np.isnan(values).any(axis=1)
    if (nan_rows != flag).any():
        msg = "the NaN rows of the cloud inputs differ from the cloud_missing flag"
        raise ValueError(msg)


def row_extras(
    *,
    arm: arms.Arm,
    store: RunArrays,
    nwp_init: np.ndarray,
    lead_hours: np.ndarray,
    donors: np.ndarray | None,
    donors_first_day: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an arm's raw extra summaries for rows, and the real `cloud_missing` flag.

    A real arm reads its own run. A placebo arm reads the donor map, and the code asserts that the
    placebo's missing pattern equals the real one before returning.

    Args:
        arm: The arm.
        store: The Primary's `RunArrays`.
        nwp_init: Each row's run in microseconds since the epoch.
        lead_hours: Each row's lead from the origin in hours.
        donors: The donor map for a placebo arm, else `None`.
        donors_first_day: The day number of the map's first run.

    Returns:
        The summaries, shape (rows, len(SUMMARY_COLUMNS[arm.summary])), and the flag.

    Raises:
        ValueError: If a placebo arm has no map, or a pattern assertion fails.
    """
    columns = arms.SUMMARY_COLUMNS[arm.summary]
    days, steps = row_run_days(nwp_init=nwp_init), row_steps(lead_hours=lead_hours)
    real, flag = real_extras(store=store, run_days=days, steps=steps, columns=columns)
    if columns:
        assert_missing_is_nan(values=real, flag=flag)
    if arm.placebo == "none":
        return real, flag
    if donors is None:
        msg = f"arm {arm.name} needs a donor map"
        raise ValueError(msg)
    values, placebo_flag = donor_extras(
        store=store,
        run_days=days,
        steps=steps,
        columns=columns,
        donors=donors,
        donors_first_day=donors_first_day,
    )
    assert_same_flag(real_flag=flag, placebo_flag=placebo_flag)
    assert_missing_is_nan(values=values, flag=placebo_flag)
    return values, flag


def extend_columns(
    *, base_names: list[str], base_mask: np.ndarray, arm: arms.Arm
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return the arm's input names, its standardise-mask, and the base columns that it keeps.

    The positive control takes `TEMPERATURE_BASE_COLUMNS` out of the base columns and adds them
    back as masked extras. Every summary column is standardised, and the flag is not.

    Args:
        base_names: The 46 base input names of experiment 707's `feature_names`.
        base_mask: Which base columns are standardised.
        arm: The arm.

    Returns:
        The names (kept base columns, then extras, then the flag), the mask, and the indices of
        the kept base columns.
    """
    keep = np.array(
        [not (arm.removes_temperature and n in arms.TEMPERATURE_BASE_COLUMNS) for n in base_names]
    )
    if arm.removes_temperature and int(keep.sum()) != len(base_names) - len(
        arms.TEMPERATURE_BASE_COLUMNS
    ):
        msg = "the base columns do not hold the four temperature inputs"
        raise ValueError(msg)
    summary = list(arms.SUMMARY_COLUMNS[arm.summary])
    flag = [arms.FLAG_COLUMN] if arm.flag else []
    names = [n for n, k in zip(base_names, keep, strict=True) if k] + summary + flag
    mask = np.concatenate([base_mask[keep], np.ones(len(summary), bool), np.zeros(len(flag), bool)])
    return names, mask, np.flatnonzero(keep)


def extend_matrix(
    *,
    x: np.ndarray,
    kept: np.ndarray,
    summaries: np.ndarray,
    flag: np.ndarray | None,
) -> np.ndarray:
    """Return the raw input matrix: the kept base columns, the summaries, and the flag.

    A masked row holds NaN in the summary columns, which the standardiser skips when it fits and
    sets to 0 when it applies.

    Args:
        x: The raw base matrix, float32.
        kept: Indices of the base columns to keep.
        summaries: The raw summaries, shape (rows, k).
        flag: The `cloud_missing` flag, or `None` if the arm has none.

    Returns:
        A float32 matrix of shape (rows, len(kept) + k + (flag is not None)).
    """
    pieces = [x[:, kept], summaries.astype(np.float32)]
    if flag is not None:
        pieces.append(flag.astype(np.float32)[:, None])
    return np.concatenate(pieces, axis=1)


def drop_gap_rows(*, nwp_init: np.ndarray) -> np.ndarray:
    """Return the rows to keep when the 19-month gap is dropped (the run-date rule only)."""
    return ~missing_flag(run_days=row_run_days(nwp_init=nwp_init))


# --- T3: the daylight mean of the ensemble-mean cloud ---------------------------------------------


def daylight_mean_by_step(
    *, cloud_mean: np.ndarray, local_day: np.ndarray, daylight: np.ndarray
) -> np.ndarray:
    """Return, per half-hour of one run, the mean cloud over its local day's daylight half-hours.

    The mean covers the daylight half-hours of the run that fall on the same local calendar day, so
    it reads only the run's own forecast. Where no daylight half-hour of the run falls on the day
    (a run's last day, in winter), the value is the half-hour's own ensemble-mean cloud.

    Args:
        cloud_mean: The ensemble-mean cloud of the run, shape (steps,), NaN for a missing run.
        local_day: The local calendar day of each half-hour, as an integer.
        daylight: Whether the sun is above the horizon at each half-hour.

    Returns:
        An array shaped like `cloud_mean`.
    """
    _, inverse = np.unique(local_day, return_inverse=True)
    usable = daylight & np.isfinite(cloud_mean)
    sums = np.bincount(inverse, weights=np.where(usable, cloud_mean, 0.0))
    counts = np.bincount(inverse, weights=usable.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
    out = np.where(counts[inverse] > 0, mean[inverse], cloud_mean)
    return np.where(np.isfinite(cloud_mean), out, np.nan)


# --- building and loading the store ---------------------------------------------------------------


def store_path(*, site: str) -> Path:
    """Return a Primary's store file."""
    return STORE_DIR / f"{site}.npz"


def load_store(*, site: str) -> RunArrays:
    """Return a Primary's `RunArrays` from its store file.

    Raises:
        FileNotFoundError: If the store has not been built.
    """
    path = store_path(site=site)
    if not path.exists():
        msg = f"{path} does not exist: run `e00711_rowcloud.py build`"
        raise FileNotFoundError(msg)
    with np.load(path) as z:
        if [str(c) for c in z["channels"]] != list(CHANNELS):
            msg = f"{path} holds other channels than {CHANNELS}: rebuild"
            raise ValueError(msg)
        return RunArrays(first_day=int(z["first_day"]), values=z["values"])


def _summary_path(*, run: datetime) -> Path:
    """Return a run's stored seven-variable summary: experiment 707's, or 397's from April 2024."""
    from e00711_cloud import OLD_SUMMARY_DIRS

    folder = OLD_SUMMARY_DIRS[0] if run <= common.BACKFILL_LAST_RUN else OLD_SUMMARY_DIRS[1]
    return folder / f"init={run:%Y-%m-%d}.parquet"


def _site_grid(*, series: pl.DataFrame) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return each series' local day number and daylight flag on a dense half-hourly valid grid."""
    from studies.solar import zenith

    first = common.CACHE_FIRST_RUN + common.PUBLICATION_DELAY + timedelta(minutes=30)
    last = common.SCORED_LAST_RUN + common.PUBLICATION_DELAY + timedelta(hours=351)
    stamps = pl.datetime_range(first, last, interval="30m", time_zone="UTC", eager=True)
    local = (
        stamps.dt.convert_time_zone(LOCAL_TIME_ZONE).dt.date().cast(pl.Int32).to_numpy().astype(int)
    )
    days = {}
    daylight = {}
    for row in series.iter_rows(named=True):
        angle = zenith(stamps=stamps, latitude=row["latitude"], longitude=row["longitude"])
        days[int(row["time_series_id"])] = local
        daylight[int(row["time_series_id"])] = angle < 90.0
    return days, daylight


def build() -> None:
    """Write every Primary's store from the cloud cache and the stored summaries."""
    from e00711_cloud import primaries

    series = primaries()
    ids = [int(i) for i in series["time_series_id"]]
    sites = dict(zip(ids, series["site"].to_list(), strict=True))
    runs = common.runs_between(first=common.CACHE_FIRST_RUN, last=common.SCORED_LAST_RUN)
    values = {i: np.full((len(runs), STEPS, len(CHANNELS)), np.nan, np.float32) for i in ids}
    cloud_channels = channel_indices(columns=arms.SUMMARY_COLUMNS["t2"])
    temp_channels = channel_indices(columns=tuple(SUMMARY_TEMPERATURE_COLUMNS))
    n_cloud = n_temp = 0
    for r, run in enumerate(runs):
        for path, channels, columns in (
            (common.cache_path(run=run), cloud_channels, list(arms.SUMMARY_COLUMNS["t2"])),
            (_summary_path(run=run), temp_channels, list(SUMMARY_TEMPERATURE_COLUMNS.values())),
        ):
            if not path.exists():
                continue
            table = pl.read_parquet(path, columns=["time_series_id", "valid_time", *columns])
            for (sid,), group in table.group_by("time_series_id"):
                if int(sid) not in values:  # ty: ignore[invalid-argument-type]
                    continue
                ordered = group.sort("valid_time")
                if ordered.height != STEPS:
                    msg = f"{path.name}: series {sid} holds {ordered.height} rows, not {STEPS}"
                    raise ValueError(msg)
                values[int(sid)][r][:, channels] = ordered.select(columns).to_numpy()  # ty: ignore[invalid-argument-type]
            if channels is cloud_channels:
                n_cloud += 1
            else:
                n_temp += 1
    local_days, daylight = _site_grid(series=series)
    grid_first = common.CACHE_FIRST_RUN + common.PUBLICATION_DELAY + timedelta(minutes=30)
    daylight_channel = CHANNELS.index("cloud_daylight_mean")
    for sid, array in values.items():
        for r, run in enumerate(runs):
            if np.isnan(array[r, :, 0]).all():
                continue
            offset = int((run - common.CACHE_FIRST_RUN) // timedelta(minutes=30))
            window = slice(offset, offset + STEPS)
            array[r, :, daylight_channel] = daylight_mean_by_step(
                cloud_mean=array[r, :, 0].astype(np.float64),
                local_day=local_days[sid][window],
                daylight=daylight[sid][window],
            )
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        target = store_path(site=sites[sid])
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez(
            temporary,
            values=array,
            first_day=placebo.day_number(run=common.CACHE_FIRST_RUN),
            channels=np.array(CHANNELS),
            grid_first=grid_first.isoformat(),
        )
        temporary.rename(target)
        print(f"{sites[sid]}: wrote {target.name}", flush=True)
    print(f"{n_cloud} runs with cloud files, {n_temp} runs with temperature summaries")
    (STORE_DIR / "manifest.json").write_text(
        json.dumps({"runs": len(runs), "cloud_files": n_cloud, "temperature_files": n_temp})
    )


def check() -> None:
    """Check each store: NaN cloud exactly on the missing runs, values in range, one example."""
    from e00711_cloud import primaries

    for site in primaries()["site"].to_list():
        store = load_store(site=site)
        runs = common.runs_between(first=common.CACHE_FIRST_RUN, last=common.SCORED_LAST_RUN)
        nan_run = np.isnan(store.values[:, :, 0]).all(axis=1)
        expected = np.array([common.is_missing_run(run=r) for r in runs])
        if (nan_run != expected).any():
            msg = f"{site}: cloud is NaN on other runs than the missing ones"
            raise ValueError(msg)
        present = store.values[~nan_run]
        if np.isnan(present).any() or (present[..., [0, 2, 3, 4]] > 1.0).any():
            msg = f"{site}: a present run holds NaN (cloud or temperature) or an out-of-range cloud"
            raise ValueError(msg)
        print(f"{site}: {int((~nan_run).sum())} runs with cloud, {int(nan_run.sum())} missing")


def main() -> None:
    """Build or check the stores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    args = parser.parse_args()
    common.require_one_polars_thread()
    if args.command == "build":
        build()
    else:
        check()


if __name__ == "__main__":
    main()

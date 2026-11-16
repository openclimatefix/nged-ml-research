# ruff: noqa: E501
"""Experiment 701: build the XGBoost inputs of the 123 runs 2026-07-01 to 2026-10-31.

Experiment 87's fit reads three kinds of input that stop at 2026-06-30: the demand frames of
experiment 48 (`frames_demand`, 51 members of every run), the control member's rows of experiment
34 (`ens_after_training`, the training rows of the refits), and the solar offset's ensemble-mean
shape and anomaly. This script rebuilds each kind for the new runs with the builder that made the
stored files (`e00048_build_frames.py`, `e00033_build_ens_after_training.py`, and
`e00087_offset.py`), from the ENS archive and experiment 607's extended observed-power table. It
writes only under this experiment's private folder.

Subcommands, in the order to run them:

- `frames-check` rebuilds 15 stored runs of 2026-05 and 2026-06 and compares each with the stored
  frame (largest absolute difference per column, null positions, row counts). It stops on any
  difference. The rebuilt frames stay in `work/frames_check` for the smoke test.
- `after-check` does the same for 12 runs of the control member's rows.
- `frames`, `after` build the new runs' files (resumable: an existing file is kept).
- `links` makes `work/ens_after_all` and `work/offset/u_mean` hold links to the stored files and
  the new files together.
- `u-mean` builds the ensemble-mean unit PV shape of each new run (experiment 87's builder).
- `anomaly` rebuilds the calibrated anomaly table for every run, checks that it equals experiment
  87's stored table on every stored row, and writes it.
- `lags-check` asserts, for every new frame and series, that every lag is no younger than the newest
  usable observation, and that the weather comes from the run issued at origin minus 9 hours.

    PYTHONPATH=<folders of experiments 87, 33, 48, 701> \
        uv run --no-sync python ml-experiments/00701-*/e00701_build_inputs.py frames-check
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00033_build_ens_after_training as b33  # ty: ignore[unresolved-import]
import e00033_common as c33  # ty: ignore[unresolved-import]
import e00048_build_frames as b48  # ty: ignore[unresolved-import]
import e00048_common as c48  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_offset as offsets  # ty: ignore[unresolved-import]
import e00701_common as mine  # first: patches experiment 87's modules
import numpy as np
import polars as pl

FRAME_CHECK_RUNS: Final[tuple[str, ...]] = (
    # 15 runs of 2026-05 and 2026-06: some are the pilot's stored frames (every fifth day from
    # 2025-07-01), the others experiment 48's frames (`frames_demand`).
    "2026-05-02",
    "2026-05-07",
    "2026-05-12",
    "2026-05-17",
    "2026-05-22",
    "2026-05-27",
    "2026-06-02",
    "2026-06-03",
    "2026-06-04",
    "2026-06-13",
    "2026-06-23",
    "2026-06-28",
    "2026-05-21",
    "2026-06-05",
    "2026-06-20",
)
AFTER_CHECK_RUNS: Final[tuple[str, ...]] = (
    "2026-05-01",
    "2026-05-06",
    "2026-05-11",
    "2026-05-16",
    "2026-05-21",
    "2026-05-26",
    "2026-06-01",
    "2026-06-05",
    "2026-06-10",
    "2026-06-15",
    "2026-06-20",
    "2026-06-29",
)
WORKERS: Final[int] = 4
"""Worker processes of the builds, one Polars thread each: the 4 cores this experiment's non-GPU
work may use."""
SERIES_COLUMNS: Final[list[str]] = ["time_series_id", "h3_res_5", "latitude", "longitude"]
KEYS: Final[list[str]] = ["time_series_id", "ensemble_member", "valid_time"]
MEMBERS: Final[int] = 51
SERIES: Final[int] = 20


def _day(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=mine.STORED_LAST_RUN.tzinfo)


def runs_between(*, first: datetime, last: datetime) -> list[datetime]:
    """Return the 00:00 UTC runs from `first` to `last` inclusive."""
    return [first + timedelta(days=d) for d in range((last - first).days + 1)]


def demand_frame(*, run: datetime, series: pl.DataFrame, observed: pl.DataFrame) -> pl.DataFrame:
    """Return one run's demand frame, built with experiment 48's functions and experiment 607's observed power."""
    rows, _init = b48._half_hourly_rows(run=run, targets=series.select(SERIES_COLUMNS))
    return b48.demand_rows(rows=rows, series=series, observed=observed)


def control_rows(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one run's control-member rows, built with experiment 33's builder."""
    return b33.run_rows(run=run, series=series)


def compare_frames(*, new: pl.DataFrame, stored: pl.DataFrame, keys: list[str]) -> dict[str, float]:
    """Return the largest absolute difference per column of two frames; raise on any structural difference.

    Args:
        new: The rebuilt frame.
        stored: The stored frame.
        keys: The columns that identify a row.

    Returns:
        Each non-key column's largest absolute difference (0 for a string or integer column that is
        equal).

    Raises:
        ValueError: If the row counts, the columns, the keys, or the null positions differ.
    """
    if new.columns != stored.columns:
        msg = f"columns differ: {new.columns} against {stored.columns}"
        raise ValueError(msg)
    if new.height != stored.height:
        msg = f"row counts differ: {new.height} against {stored.height}"
        raise ValueError(msg)
    new = new.sort(keys)
    stored = stored.sort(keys)
    for key in keys:
        if not new[key].equals(stored[key]):
            msg = f"key column {key} differs"
            raise ValueError(msg)
    out: dict[str, float] = {}
    for column in new.columns:
        if column in keys:
            continue
        a, b = new[column], stored[column]
        if a.null_count() != b.null_count() or (a.is_null() != b.is_null()).any():
            msg = f"null positions of {column} differ"
            raise ValueError(msg)
        if a.dtype.is_float():
            both_nan = a.is_nan().fill_null(False) & b.is_nan().fill_null(False)
            if (a.is_nan().fill_null(False) != b.is_nan().fill_null(False)).any():
                msg = f"NaN positions of {column} differ"
                raise ValueError(msg)
            difference = (a.cast(pl.Float64) - b.cast(pl.Float64)).abs().fill_nan(0.0)
            difference = pl.select(
                pl.when(both_nan).then(0.0).otherwise(difference).alias("d")
            ).to_series()
            out[column] = float(np.max(difference.to_numpy(), initial=0.0))
        else:
            out[column] = 0.0 if a.equals(b) else float("inf")
    return out


def _print_comparison(*, label: str, result: dict[str, float]) -> bool:
    worst = max(result.values())
    detail = ", ".join(f"{name} {value:.3g}" for name, value in result.items() if value > 0.0)
    print(
        f"{label}: {len(result)} columns, largest difference {worst:.3g}"
        + (f" | {detail}" if detail else " in every column"),
        flush=True,
    )
    return worst == 0.0


def frames_check() -> None:
    """Rebuild 15 stored demand frames and require them to equal the stored ones bit for bit."""
    series = c48.series_table()
    observed = pl.read_parquet(common.OBSERVED_PATH)
    mine.FRAMES_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    all_equal = True
    for text in FRAME_CHECK_RUNS:
        run = _day(text)
        started = time.monotonic()
        frame = demand_frame(run=run, series=series, observed=observed)
        b48._write_sorted(
            frame=frame, target=mine.FRAMES_CHECK_DIR / f"init={run:%Y-%m-%d}.parquet"
        )
        stored_path = mine.stored_frame_path(run=run)
        stored = pl.read_parquet(stored_path).filter(
            pl.col("time_series_id").is_in(series["time_series_id"].to_list())
        )
        if stored.columns != frame.columns:
            # The pilot's frames may hold other columns; compare the shared ones.
            shared = [c for c in frame.columns if c in stored.columns]
            print(
                f"  {run:%Y-%m-%d}: columns only in the rebuilt frame {sorted(set(frame.columns) - set(shared))}, "
                f"only in the stored frame {sorted(set(stored.columns) - set(shared))}"
            )
            frame, stored = frame.select(shared), stored.select(shared)
        result = compare_frames(new=frame, stored=stored, keys=KEYS)
        kind = "pilot frame" if stored_path.parent == common.VALIDATION_DIR else "frames_demand"
        equal = _print_comparison(
            label=f"{run:%Y-%m-%d} ({kind}, {frame.height:,} rows, {time.monotonic() - started:.0f} s)",
            result=result,
        )
        all_equal &= equal
    if not all_equal:
        msg = "a rebuilt frame differs from the stored frame: stop and explain before building"
        raise SystemExit(msg)
    print(f"frames-check: all {len(FRAME_CHECK_RUNS)} runs bit-equal")


def after_check() -> None:
    """Rebuild 12 stored control-member runs and require them to equal experiment 34's files."""
    series = c33.series_table()
    mine.AFTER_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    all_equal = True
    for text in AFTER_CHECK_RUNS:
        run = _day(text)
        frame = control_rows(run=run, series=series)
        frame.write_parquet(mine.AFTER_CHECK_DIR / f"init={run:%Y-%m-%d}.parquet")
        stored = pl.read_parquet(
            common.EXPERIMENT_00034_WORK_DIR / "ens_after_training" / f"init={run:%Y-%m-%d}.parquet"
        )
        result = compare_frames(
            new=frame,
            stored=stored,
            keys=["time_series_id", "power_fcst_init_time", "valid_time", "ensemble_member"],
        )
        all_equal &= _print_comparison(
            label=f"{run:%Y-%m-%d} ({frame.height:,} rows)", result=result
        )
    if not all_equal:
        msg = "a rebuilt control-member file differs from the stored file: stop and explain"
        raise SystemExit(msg)
    print(f"after-check: all {len(AFTER_CHECK_RUNS)} runs bit-equal")


_SHARED: dict[str, pl.DataFrame] = {}


def _init_worker() -> None:
    os.environ["POLARS_MAX_THREADS"] = "1"
    _SHARED["series48"] = c48.series_table()
    _SHARED["series33"] = c33.series_table()
    _SHARED["observed"] = pl.read_parquet(common.OBSERVED_PATH)


def _build_frame(*, run: datetime) -> str:
    target = mine.FRAMES_DIR / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{target.name}: exists"
    started = time.monotonic()
    frame = demand_frame(run=run, series=_SHARED["series48"], observed=_SHARED["observed"])
    b48._write_sorted(frame=frame, target=target)
    return f"{target.name}: {frame.height:,} rows, {time.monotonic() - started:.0f} s"


def _build_after(*, run: datetime) -> str:
    target = mine.AFTER_NEW_DIR / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{target.name}: exists"
    started = time.monotonic()
    frame = control_rows(run=run, series=_SHARED["series33"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
    return f"{target.name}: {frame.height:,} rows, {time.monotonic() - started:.0f} s"


def _build_many(*, builder: str, runs: list[datetime]) -> None:
    started = time.monotonic()
    function = {"frames": _build_frame, "after": _build_after}[builder]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as pool:
        futures = [pool.submit(function, run=run) for run in runs]
        for done, future in enumerate(futures, start=1):
            print(f"[{done}/{len(futures)}] {future.result()}", flush=True)
    print(f"{builder}: {len(runs)} runs in {time.monotonic() - started:.0f} s")


def make_links() -> None:
    """Link the stored and new control-member files, and the stored ensemble-mean shape files."""
    mine.AFTER_ALL_DIR.mkdir(parents=True, exist_ok=True)
    sources = [
        *sorted((common.EXPERIMENT_00034_WORK_DIR / "ens_after_training").glob("init=*.parquet")),
        *sorted(mine.AFTER_NEW_DIR.glob("init=*.parquet")),
    ]
    names = [p.name for p in sources]
    if len(names) != len(set(names)):
        msg = "a control-member run is in both the stored and the new folder"
        raise ValueError(msg)
    for path in sources:
        link = mine.AFTER_ALL_DIR / path.name
        if not link.exists():
            link.symlink_to(path)
    print(f"ens_after_all: {len(sources)} files ({sources[0].stem} to {sources[-1].stem})")
    u_dir = offsets.U_MEAN_DIR
    u_dir.mkdir(parents=True, exist_ok=True)
    stored = sorted((mine.E87_WORK / "offset" / "u_mean").glob("init=*.parquet"))
    for path in stored:
        link = u_dir / path.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(path)
    print(f"u_mean: {len(stored)} stored files linked into {u_dir}")


def build_anomaly() -> None:
    """Rebuild the anomaly table of every run, check it against experiment 87's, and write it."""
    table = offsets.anomaly_table()
    stored = pl.read_parquet(mine.E87_WORK / "offset" / "anomaly.parquet")
    joined = stored.join(table, on=list(common.KEYS), how="left", suffix="_new", validate="1:1")
    missing = joined["anomaly_new"].null_count()
    gap = float(np.abs((joined["anomaly"] - joined["anomaly_new"]).to_numpy()).max())
    print(
        f"anomaly: {table.height:,} rows to {table['valid_time'].max()}; stored table {stored.height:,} rows: "
        f"{missing} missing in the rebuilt table, largest difference {gap:.3g}"
    )
    if missing or gap > 0.0:
        msg = "the rebuilt anomaly table does not equal experiment 87's stored table"
        raise SystemExit(msg)
    new_rows = table.filter(
        pl.col("power_fcst_init_time") > common.origin_of(run=mine.STORED_LAST_RUN)
    )
    runs = new_rows["power_fcst_init_time"].n_unique()
    print(f"anomaly: {new_rows.height:,} rows of {runs} new runs (expected 123)")
    if runs != (mine.NEW_LAST_RUN - mine.NEW_FIRST_RUN).days + 1:
        msg = "the anomaly table lacks some new runs"
        raise SystemExit(msg)
    common.atomic_write(frame=table, target=offsets.ANOMALY_FILE)


def lags_check() -> None:
    """Assert the lag and weather-timing rules on every new frame (the leakage checks).

    For each new frame and series: the weather comes from the run issued at origin minus 9 hours
    (`nwp_init_time` is the run's day and `power_fcst_init_time` is 9 hours later on every row),
    every valid time is after the origin, and `e00087_common.origin_lags` (which raises if a lag
    index is later than the newest usable observation, 3.5 hours before the origin) runs on every
    case. The check then repeats the lag arithmetic independently: the newest observation any lag
    reads is at least 3.5 hours before the origin.
    """
    series = common.series_table()
    arrays = common.load_power_arrays(series=series)
    youngest_age = 10**9
    total_cases = 0
    for run in runs_between(first=mine.NEW_FIRST_RUN, last=mine.NEW_LAST_RUN):
        frame = pl.read_parquet(mine.FRAMES_DIR / f"init={run:%Y-%m-%d}.parquet")
        origin = common.origin_of(run=run)
        if not (frame["nwp_init_time"] == run).all():
            msg = f"{run:%Y-%m-%d}: nwp_init_time is not the run"
            raise ValueError(msg)
        if not (frame["power_fcst_init_time"] == origin).all():
            msg = f"{run:%Y-%m-%d}: power_fcst_init_time is not nwp_init_time plus 9 hours"
            raise ValueError(msg)
        if not (frame["valid_time"] > origin).all():
            msg = f"{run:%Y-%m-%d}: a valid time is not after the origin"
            raise ValueError(msg)
        if (
            frame["time_series_id"].n_unique() != SERIES
            or frame["ensemble_member"].n_unique() != MEMBERS
        ):
            msg = f"{run:%Y-%m-%d}: not {SERIES} series and {MEMBERS} members"
            raise ValueError(msg)
        times = frame["valid_time"].unique().sort()
        target = np.asarray(common.grid_index(times=times))
        origin_index = np.full(target.shape, int(common.grid_index(times=origin)))
        for site in series["site"]:
            power = arrays[site].power
            common.origin_lags(power=power, target=target, origin=origin_index)
            newest = origin_index - common.NEWEST_AGE
            gap = target - newest
            day = target - common.STEPS_PER_DAY * np.ceil(gap / common.STEPS_PER_DAY).astype(int)
            week = target - 7 * common.STEPS_PER_DAY * np.ceil(
                gap / (7 * common.STEPS_PER_DAY)
            ).astype(int)
            latest = int(max(day.max(), week.max(), newest.max()))
            age = int(origin_index[0] - latest)
            youngest_age = min(youngest_age, age)
            if age < common.NEWEST_AGE:
                msg = f"{run:%Y-%m-%d} {site}: a lag reads an observation {age} half-hours before the origin"
                raise ValueError(msg)
            total_cases += target.size
    print(
        f"lags-check: {total_cases:,} cases pass; the youngest observation any lag reads is "
        f"{youngest_age * 0.5} hours before its origin (3.5 hours required); weather timing holds on all "
        f"{(mine.NEW_LAST_RUN - mine.NEW_FIRST_RUN).days + 1} frames"
    )


def main() -> None:
    """Run one subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "frames-check",
            "after-check",
            "frames",
            "after",
            "links",
            "u-mean",
            "anomaly",
            "lags-check",
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="build only the first N runs")
    args = parser.parse_args()
    mine.WORK.mkdir(parents=True, exist_ok=True)
    runs: Path | list[datetime]
    if args.command == "frames-check":
        frames_check()
    elif args.command == "after-check":
        after_check()
    elif args.command == "frames":
        runs = runs_between(first=mine.NEW_FIRST_RUN, last=mine.NEW_LAST_RUN)[: args.limit]
        _build_many(builder="frames", runs=runs)
    elif args.command == "after":
        runs = runs_between(first=mine.AFTER_FIRST_NEW_RUN, last=mine.AFTER_LAST_NEW_RUN)[
            : args.limit
        ]
        _build_many(builder="after", runs=runs)
    elif args.command == "links":
        make_links()
    elif args.command == "u-mean":
        offsets.build_u_means(workers=WORKERS, limit=args.limit)
    elif args.command == "anomaly":
        build_anomaly()
    else:
        lags_check()


if __name__ == "__main__":
    main()

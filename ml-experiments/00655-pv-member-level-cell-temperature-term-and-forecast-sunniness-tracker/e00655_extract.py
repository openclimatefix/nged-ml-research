# ruff: noqa: E501
"""Experiment 655: extract the 51-member weather of the six MW Primaries from the ensemble archive.

For each daily 00:00 UTC run, the archive (`data/NWP`, 51 members) is cut to each of the six MW
Primaries' H3 cells and resampled to half-hours by experiment 48's `_half_hourly_rows` (the path
that built the cached ensemble summaries). The file `extract/run=<date>.npz` holds, for each
Primary index `p`, the arrays `p<p>_sw`, `p<p>_t`, and `p<p>_w` (shortwave in W/m2, 2 m temperature
in degrees C, 10 m wind speed in m/s), each of shape (702 steps, 51 members), float32. Step `j` is
the valid time `origin + (j + 1) * 30 min`, where `origin` is the run plus 9 hours.

Usage: python e00655_extract.py run [--workers 6] [--first 2024-04-01] [--last 2026-06-30]
       python e00655_extract.py check
"""

import argparse
import multiprocessing as mp
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00655-pv-member-level-cell-temperature-term-and-forecast-sunniness-tracker"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
EXTRACT: Final[Path] = WORK / "extract"
N_STEPS: Final = 702
"""Half-hours from origin + 30 min to origin + 351 h, the longest lead of the cached training rows."""
N_MEMBERS: Final = 51
FIRST_RUN: Final = datetime(2024, 4, 1, tzinfo=UTC)
LAST_RUN: Final = datetime(2026, 6, 30, tzinfo=UTC)
ORIGIN_DELAY: Final = timedelta(hours=9)
STEP: Final = timedelta(minutes=30)


def run_path(*, run: datetime) -> Path:
    """Return the file of one run."""
    return EXTRACT / f"run={run:%Y-%m-%d}.npz"


def _targets() -> pl.DataFrame:
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00634_common as k634  # ty: ignore[unresolved-import]

    six = list(k634.six_indices().values())
    return (
        prep.primaries()
        .filter(pl.col("primary").is_in(six))
        .select("primary", "time_series_id", "h3_res_5", "latitude", "longitude")
    )


def extract_one(run: datetime) -> str:
    """Write one run's file if absent."""
    import e00048_build_frames as build  # ty: ignore[unresolved-import]

    target = run_path(run=run)
    if target.exists():
        return f"{run:%Y-%m-%d}: exists"
    targets = _targets()
    rows, init = build._half_hourly_rows(
        run=run, targets=targets.select("time_series_id", "h3_res_5", "latitude", "longitude")
    )
    assert init == run + ORIGIN_DELAY, init
    rows = rows.filter(pl.col("lead_hours") <= N_STEPS / 2.0).join(
        targets.select("primary", "time_series_id"), on="time_series_id"
    )
    arrays: dict[str, np.ndarray] = {}
    for p in targets["primary"].to_list():
        part = rows.filter(pl.col("primary") == p).sort("valid_time", "ensemble_member")
        valid = part["valid_time"].dt.epoch("us").to_numpy()
        j = (valid - int(init.timestamp() * 1e6)) // int(STEP.total_seconds() * 1e6) - 1
        assert j.min() == 0 and j.max() == N_STEPS - 1, (p, j.min(), j.max())
        assert part["ensemble_member"].n_unique() == N_MEMBERS
        order = np.lexsort((part["ensemble_member"].to_numpy(), j))
        for key, col in (("sw", "shortwave"), ("t", "temperature_2m"), ("w", "wind_speed_10m")):
            a = part[col].to_numpy()[order].reshape(N_STEPS, N_MEMBERS).astype(np.float32)
            arrays[f"p{p}_{key}"] = a
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)  # ty: ignore[invalid-argument-type]
    tmp.rename(target)
    return f"{run:%Y-%m-%d}: written"


def _worker(run: datetime) -> str:
    return extract_one(run)


def runs(*, first: datetime, last: datetime) -> list[datetime]:
    """Return the daily 00:00 UTC runs from `first` to `last` inclusive."""
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def check() -> None:
    """Compare three extracted runs' member means with the cached summaries (exact to float32)."""
    import e00397_prep as prep  # ty: ignore[unresolved-import]

    targets = _targets()
    for day in ("2024-06-15", "2025-03-10", "2026-02-01"):
        run = datetime.fromisoformat(day).replace(tzinfo=UTC)
        extract_one(run)
        z = np.load(run_path(run=run))
        summary = pl.read_parquet(prep.SUMMARY_DIR / f"init={day}.parquet")
        origin = run + ORIGIN_DELAY
        valid = [origin + (j + 1) * STEP for j in range(N_STEPS)]
        worst = 0.0
        for row in targets.iter_rows(named=True):
            s = summary.filter(pl.col("time_series_id") == row["time_series_id"]).sort("valid_time")
            frame = pl.DataFrame({"valid_time": valid}).with_columns(
                pl.col("valid_time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"),
                sw=z[f"p{row['primary']}_sw"].mean(axis=1),
                t=z[f"p{row['primary']}_t"].mean(axis=1),
                w=z[f"p{row['primary']}_w"].mean(axis=1),
            ).join(s, on="valid_time", how="inner")
            assert frame.height > 400, frame.height
            for mine, cached in (("sw", "mean_shortwave"), ("t", "mean_temperature_2m"), ("w", "mean_wind_speed_10m")):
                err = float((frame[mine] - frame[cached]).abs().max())  # ty: ignore[invalid-argument-type]
                scale = float(frame[cached].abs().max())  # ty: ignore[invalid-argument-type]
                worst = max(worst, err / max(scale, 1.0))
        print(f"{day}: worst relative difference {worst:.2e}", flush=True)
        assert worst < 1e-3, worst


def main() -> None:
    """Run the extract or the check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--first", default=FIRST_RUN.strftime("%Y-%m-%d"))
    parser.add_argument("--last", default=LAST_RUN.strftime("%Y-%m-%d"))
    a = parser.parse_args()
    if a.command == "check":
        check()
        return
    todo = runs(
        first=datetime.fromisoformat(a.first).replace(tzinfo=UTC),
        last=datetime.fromisoformat(a.last).replace(tzinfo=UTC),
    )
    assert datetime.fromisoformat(a.last).replace(tzinfo=UTC) < datetime(2026, 7, 1, tzinfo=UTC)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for i, line in enumerate(pool.imap(_worker, todo, chunksize=4)):
            if i % 25 == 0:
                print(line, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

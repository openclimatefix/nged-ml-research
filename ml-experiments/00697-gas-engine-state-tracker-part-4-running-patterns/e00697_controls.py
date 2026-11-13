r"""Experiment 697's controls: a real reading with a known engine injected, then tracked.

    E=ml-experiments
    export PYTHONPATH=$E/00697-gas-engine-state-tracker-part-4-running-patterns:$E/00694-gas-engine-state-tracker-part-1:$E/00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back
    uv run --no-sync python $E/00697-*/e00697_controls.py --list
    uv run --no-sync python $E/00697-*/e00697_controls.py --time-one   # one run, nothing written
    uv run --no-sync python $E/00697-*/e00697_controls.py              # every job, 8 workers

**A job is one series, one span, one kind of truth, and one seed.** The jobs are:

- **C1:** D8 before the engine (the panel's first 28 days are the pre-roll), null and weekday
  truth, seeds 0 to 2.
- **C1s:** C1 with one constant subtracted from the reading before the injection, null truth,
  seeds 0 to 2. The design expects D8's weekday daytime demand to be about 9 MW before the engine
  and about 3 MW in the net era, so that a 5 MW engine pushes the reading through the MVA fold (the
  reading is apparent power and never negative). The constant is C1's weekday 09:00 to 16:00 median
  reading minus the same median of the demand level (`lmean`) that experiment 695 fitted for a
  target era, and the scaled reading is floored at 0.3 MVA. A constant keeps the size of C1's
  evening peak in MW, so it does not reproduce the target era's smaller demand swing or noise.
  **The stored data give two constants, and the script prints both.** For the net era the median
  level is within 0.1 MVA of C1's, so `c1s_net` is nearly C1. For the solar era the level is about
  3 MVA lower, so `c1s_solar` is the control that tests the fold. Both are run and both are
  compared with both eras.
- **C2:** hosts D2, D3, and D4 in the net and the solar span, null and weekday truth, seeds 0 and 1.

Each job injects Gas A's days over the span and the 28 days before it, tracks the span with
experiment 694's M2 (`track_span`), and writes `controls/<job>.npz` atomically. A finished file is
skipped, so a crashed run is rerun without touching the others. The seed of the day draw is the
job's seed plus a per-series offset, so two hosts of one span do not share a truth.
"""

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from typing import Final

import numpy as np
import polars as pl
from e00694_common import ENGINE_CONNECTED, PRE_ROLL, stamp  # ty: ignore[unresolved-import]
from e00697_common import (  # ty: ignore[unresolved-import]
    CONTROLS_DIR,
    DAY,
    PIPELINE_OFFSET,
    PROBABILITY_PATH,
    TruthKindType,
    add_local_fields,
    era_bounds,
    half_hour_starts,
    inject_span,
    load_panel,
    panel_array,
    track_span,
)
from e00697_engine import (  # ty: ignore[unresolved-import]
    DRAW_SEED,
    RecordDays,
    drive_series,
    grid_calendar,
    record_days,
)

N_WORKERS: Final[int] = 8
HOSTS: Final[tuple[str, ...]] = ("D2", "D3", "D4")
"""MVA-only Primaries that experiment 695 used as hosts."""
SERIES_SEED_OFFSET: Final[dict[str, int]] = {"D8": 0, "D2": 100, "D3": 200, "D4": 300}
SCALED_FLOOR_MVA: Final[float] = 0.3
"""C1s floors the scaled reading here, so the reading stays positive."""
DAYTIME_FIRST: Final[int] = 18
DAYTIME_LAST: Final[int] = 31
"""Local half-hours 18 to 31 start at 09:00 to 15:30, so they cover 09:00 to 16:00."""
PRE_FIRST: Final[int] = 28 * DAY
"""C1 starts after the panel's first 28 days, which carry the trailing spread."""
SPAN_NAMES: Final[tuple[str, ...]] = ("pre", "net", "solar")
SeriesName = str


@dataclass(frozen=True)
class Job:
    """One tracker run on an injected reading."""

    name: str
    control: str
    series: SeriesName
    span: str
    truth: TruthKindType
    seed: int
    scale_to: str | None


def jobs() -> list[Job]:
    """Return every job: 6 for C1, 6 for C1s (3 scaled to each era), and 24 for C2."""
    out: list[Job] = []
    for truth in ("null", "weekday"):
        for seed in range(3):
            out.append(Job(f"c1_{truth}_s{seed}", "c1", "D8", "pre", truth, seed, scale_to=None))
    for era in ("net", "solar"):
        out.extend(
            Job(f"c1s_{era}_null_s{seed}", f"c1s_{era}", "D8", "pre", "null", seed, scale_to=era)
            for seed in range(3)
        )
    for host in HOSTS:
        for span in ("net", "solar"):
            for truth in ("null", "weekday"):
                for seed in range(2):
                    name = f"c2_{host.lower()}_{span}_{truth}_s{seed}"
                    out.append(Job(name, "c2", host, span, truth, seed, scale_to=None))
    return out


def span_bounds(*, span: str) -> tuple[int, int]:
    """Return the first panel index of a span and one past its last."""
    if span == "pre":
        return PRE_FIRST, stamp(when=ENGINE_CONNECTED)
    return era_bounds()[span]


def level_shift_constant(*, panel: dict[int, dict], era: str) -> float:
    """Return the constant that C1s subtracts from D8's reading.

    The constant is the median of D8's reading on weekday half-hours that start 09:00 to 16:00
    local in C1's span, minus the median of experiment 695's fitted demand level `lmean` over the
    same local half-hours of the target `era`. The two series are on grids that differ by
    `PIPELINE_OFFSET` half-hours and share one alignment.
    """
    y = panel_array(panel=panel, label="D8", key="power")
    lmean = np.load(PROBABILITY_PATH)["lmean"].astype(float)
    first, last = span_bounds(span="pre")
    era_first, era_last = span_bounds(span=era)

    def weekday_daytime(*, lo: int, hi: int) -> np.ndarray:
        index = np.arange(lo, hi)
        frame = add_local_fields(frame=pl.DataFrame({"start": half_hour_starts(index=index)}))
        keep = (
            ~frame["weekend"].to_numpy()
            & ~frame["clock_change_day"].to_numpy()
            & (frame["local_half_hour"].to_numpy() >= DAYTIME_FIRST)
            & (frame["local_half_hour"].to_numpy() <= DAYTIME_LAST)
        )
        return index[keep]

    reading = y[weekday_daytime(lo=first, hi=last)]
    level = lmean[weekday_daytime(lo=era_first, hi=era_last) + PIPELINE_OFFSET]
    return float(np.nanmedian(reading) - np.nanmedian(level))


PANEL: dict[int, dict] = {}
SERIES: dict[str, np.ndarray] = {}
CALENDAR: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
DAYS: list[RecordDays] = []
SHIFT: dict[str, float] = {}


def init() -> None:
    """Load the panel, the calendar, the record days, and C1s's constant into module globals.

    Forked workers inherit them.
    """
    PANEL.update(load_panel())
    for label in ("D8", *HOSTS):
        SERIES[label] = panel_array(panel=PANEL, label=label, key="power")
    CALENDAR.append(grid_calendar(n_grid=len(SERIES["D8"])))
    DAYS.append(record_days())
    for era in ("net", "solar"):
        SHIFT[era] = level_shift_constant(panel=PANEL, era=era)


def run_job(job: Job, *, write: bool = True) -> str:
    """Inject, track, and store one job, and return a line for the log."""
    path = CONTROLS_DIR / f"{job.name}.npz"
    if write and path.exists():
        return f"{job.name}: exists, skipped"
    started = time.time()
    y = SERIES[job.series].copy()
    shift = 0.0 if job.scale_to is None else SHIFT[job.scale_to]
    if job.scale_to is not None:
        y = np.maximum(y - shift, SCALED_FLOOR_MVA)
    first, last = span_bounds(span=job.span)
    inject_first = max(first - PRE_ROLL, 0)
    g = drive_series(
        calendar=CALENDAR[0],
        first=inject_first,
        last=last,
        kind=job.truth,
        seed=DRAW_SEED + job.seed + SERIES_SEED_OFFSET[job.series],
        days=DAYS[0],
    )
    injected = inject_span(y_all=y, g=g, first=inject_first, last=last)
    track = track_span(y_all=injected, first=first, last=last)
    seconds = time.time() - started
    if write:
        CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                first=first,
                last=last,
                y=injected[first:last],
                g=g[first:last],
                flag=track.flag,
                filtered=track.filtered,
                k_path=track.k_path,
                p_any=track.p_any,
                k_max=track.k_max,
                unit=track.unit,
                seed=job.seed,
                shift=shift,
                seconds=seconds,
            )
        tmp.replace(path)
    return f"{job.name}: {seconds:.0f} s, K={track.k_max}, unit={track.unit:.2f} MW"


def load_control(*, job: Job) -> dict[str, np.ndarray]:
    """Return a stored control run's arrays."""
    with np.load(CONTROLS_DIR / f"{job.name}.npz") as data:
        return {key: data[key] for key in data.files}


def _worker(job: Job) -> str:
    return run_job(job)


def main() -> None:
    """List the jobs, time one, or run them all."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print the jobs and exit")
    parser.add_argument("--time-one", action="store_true", help="run one C1 job, write nothing")
    parser.add_argument("--only", default="", help="run only the jobs whose name has this text")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()
    selected = [j for j in jobs() if args.only in j.name]
    if args.list:
        for job in selected:
            print(job)
        print(f"{len(selected)} jobs")
        return
    init()
    print(f"C1s constants (MVA): {SHIFT}", flush=True)
    if args.time_one:
        print(run_job(selected[0], write=False))
        return
    started = datetime.now(UTC)
    with get_context("fork").Pool(args.workers) as pool:
        for line in pool.imap_unordered(_worker, selected, chunksize=1):
            print(line, flush=True)
    print(f"done in {(datetime.now(UTC) - started).total_seconds():.0f} s")


if __name__ == "__main__":
    main()

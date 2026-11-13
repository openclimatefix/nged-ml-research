r"""Experiment 698's tracker runs: host jobs and step 4 (Gas A in D8's own off-spells).

    E=ml-experiments
    export PYTHONPATH=$(ls -d $E/00698-* $E/00697-* $E/00694-* $E/00695-* | paste -sd:)
    uv run --no-sync python $E/00698-*/e00698_jobs.py --list
    uv run --no-sync python $E/00698-*/e00698_jobs.py --time-one   # one run, nothing written
    uv run --no-sync python $E/00698-*/e00698_jobs.py              # every job, 8 workers

**A host job is one host, one solar capacity, one solar shape, and one seed.** The hosts are D8
before the engine (`d8pre`) and D3 in the solar era (`d3solar`). Each host has a solar-free twin
and, for each capacity and shape that passes the curve gate of `e00698_audit.py`, 3 seeds of a
patternless engine and one run with no engine. The engine is Gas A drawn as whole local days with
no regard to the weekday (experiment 697's `drive_series`, kind `null`, with experiment 697's
seed recipe). The solar and the engine output are subtracted together from the real power behind
the reading, and the sum is read back as `sqrt(real^2 + q^2)` (`fold_reading`). The signed real
power is stored.

**A step 4 job injects Gas A into D8's own reading, on whole local days inside off-spells.** A local
day takes part when all its half-hours are observed and lie in a run of 48 half-hours or more that
experiment 694's smoothed labels call off, in one era. There are 5 seeds in each of the net and the
solar era. The tracker runs over the whole era.

Each job writes `jobs/<name>.npz` atomically. A finished file is skipped, so a crashed run is rerun
without touching the others. Each file records the farm's letter, the shape, and the capacity, and
`e00698_score.py` refuses a file whose farm differs from the audit's. A real run needs a passing
`audit.json` that is newer than the audit's inputs, and `--ignore-gate` works only with `--list` or
`--time-one`. Nothing here is published: the files hold dated output of Gas A and a
farm and stay in the private data folder.
"""

import argparse
import time
from datetime import UTC, datetime
from multiprocessing import get_context

import numpy as np
from e00694_common import PRE_ROLL, load_panel  # ty: ignore[unresolved-import]
from e00697_common import (  # ty: ignore[unresolved-import]
    era_bounds,
    load_d8_eras,
    track_span,
)
from e00697_engine import (  # ty: ignore[unresolved-import]
    DRAW_SEED,
    RecordDays,
    drive_series,
    grid_calendar,
    record_days,
)
from e00698_common import (  # ty: ignore[unresolved-import]
    GATE_PATH,
    HOST_SERIES,
    JOBS_DIR,
    N_WORKERS,
    PRINTED_DIR,
    STEP4_MIN_SPELL,
    STEP4_SEED_OFFSET,
    STEP4_WHOLE_DAYS_ONLY,
    Job,
    all_jobs,
    engine_seed,
    farm_norm_series,
    farm_on_grid,
    fold_reading,
    grid_calendar_fields,
    host_bounds,
    off_spell_rows,
    panel_series,
    read_json,
    require_passing_audit,
    solar_curve,
    whole_day_rows,
)

PANEL: dict[int, dict] = {}
SERIES: dict[str, np.ndarray] = {}
PV_SHAPE: dict[str, np.ndarray] = {}
CALENDAR: list = []
FIELDS: list = []
DAYS: list[RecordDays] = []
FARM: list[np.ndarray] = []
FARM_LETTER: list[str] = []


def init(*, farm: str) -> None:
    """Load the panel, the calendars, Gas A's days, and the farm into module globals.

    Forked workers inherit them.

    Args:
        farm: The farm's letter, chosen by the audit.
    """
    PANEL.update(load_panel())
    for label in ("D8", "D3"):
        SERIES[label] = panel_series(panel=PANEL, label=label, key="power")
        PV_SHAPE[label] = panel_series(panel=PANEL, label=label, key="pv_shape")
    n_grid = len(SERIES["D8"])
    CALENDAR.append(grid_calendar(n_grid=n_grid))
    FIELDS.append(grid_calendar_fields(n_grid=n_grid))
    DAYS.append(record_days())
    FARM.append(farm_on_grid(series=farm_norm_series(farm=farm), n_grid=n_grid))
    FARM_LETTER.append(farm)


def run_host_job(*, job: Job) -> dict[str, np.ndarray | float | int | str]:
    """Inject solar and engine into a host's reading and track the span.

    Returns:
        The arrays of the job's file: `farm` (the farm's letter), `shape`, `capacity_mw`, `y` (the
        reading), `g` (the engine's output, MW), `pv` (the solar output, MW), `signed` (the signed
        real power), `farm_ok`, the labels, and the fit.
    """
    series = HOST_SERIES[job.host]
    y = SERIES[series]
    first, last = host_bounds(host=job.host)
    inject_first = max(first - PRE_ROLL, 0)
    n_grid = len(y)
    if job.capacity_mw is None:
        pv = np.zeros(n_grid)
    else:
        pv = solar_curve(
            shape=job.shape,
            capacity_mw=job.capacity_mw,
            pv_shape=PV_SHAPE[series],
            farm_norm=FARM[0],
            calendar=FIELDS[0],
            first=inject_first,
            last=last,
        ).pv_mw
    if job.seed is None:
        g = np.zeros(n_grid)
    else:
        g = drive_series(
            calendar=CALENDAR[0],
            first=inject_first,
            last=last,
            kind="null",
            seed=engine_seed(host=job.host, seed=job.seed),
            days=DAYS[0],
        )
    reading, signed = fold_reading(y=y, total_mw=pv + g, first=inject_first, last=last)
    track = track_span(y_all=reading, first=first, last=last)
    return {
        "farm": FARM_LETTER[0],
        "shape": job.shape,
        "capacity_mw": float("nan") if job.capacity_mw is None else job.capacity_mw,
        "first": first,
        "last": last,
        "y": reading[first:last],
        "g": g[first:last],
        "pv": pv[first:last],
        "signed": signed[first:last],
        "farm_ok": np.isfinite(FARM[0][first:last]),
        "flag": track.flag,
        "filtered": track.filtered,
        "k_path": track.k_path,
        "p_any": track.p_any,
        "k_max": track.k_max,
        "unit": track.unit,
    }


def step4_days(*, era: str) -> np.ndarray:
    """Return, on the era's half-hours, the rows of the whole local days inside off-spells.

    The off-spells are runs of 48 half-hours or more that experiment 694's smoothed labels of D8
    call off, in the one era.
    """
    record = next(e for e in load_d8_eras() if e["label"] == era)
    flag = np.asarray(record["M2"]["flag"], dtype=bool)
    first, last = era_bounds()[era]
    observed = np.isfinite(SERIES["D8"][first:last])
    rows = off_spell_rows(flag=flag, min_length=STEP4_MIN_SPELL)
    if not STEP4_WHOLE_DAYS_ONLY:
        return rows & observed
    return whole_day_rows(rows=rows, local_date=CALENDAR[0][0][first:last], observed=observed)


def run_step4_job(*, job: Job) -> dict[str, np.ndarray | float | int | str]:
    """Inject Gas A into the selected whole days of one era of D8's reading and track the era."""
    era = job.era
    if era is None or job.seed is None:
        msg = "a step 4 job needs an era and a seed"
        raise ValueError(msg)
    y = SERIES["D8"]
    first, last = era_bounds()[era]
    days = step4_days(era=era)
    g_all = drive_series(
        calendar=CALENDAR[0],
        first=first,
        last=last,
        kind="null",
        seed=DRAW_SEED + job.seed + STEP4_SEED_OFFSET[era],
        days=DAYS[0],
    )
    g = np.zeros(len(y))
    g[first:last] = np.where(days, g_all[first:last], 0.0)
    reading, signed = fold_reading(y=y, total_mw=g, first=first, last=last)
    track = track_span(y_all=reading, first=first, last=last)
    return {
        "first": first,
        "last": last,
        "y": reading[first:last],
        "g": g[first:last],
        "pv": np.zeros(last - first),
        "signed": signed[first:last],
        "selected": days,
        "flag": track.flag,
        "filtered": track.filtered,
        "k_path": track.k_path,
        "p_any": track.p_any,
        "k_max": track.k_max,
        "unit": track.unit,
    }


def run_job(*, job: Job, write: bool = True) -> str:
    """Run and store one job, and return a line for the log."""
    path = JOBS_DIR / f"{job.name}.npz"
    if write and path.exists():
        return f"{job.name}: exists, skipped"
    started = time.time()
    arrays = run_step4_job(job=job) if job.kind == "step4" else run_host_job(job=job)
    seconds = time.time() - started
    line = f"{job.name}: {seconds:.0f} s, K={arrays['k_max']}, unit={float(arrays['unit']):.2f} MW"
    if not write:
        return line + reproduces_experiment_697(job=job, flag=np.asarray(arrays["flag"]))
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, seconds=seconds, **arrays)  # ty: ignore[invalid-argument-type]
    tmp.replace(path)
    return line


def reproduces_experiment_697(*, job: Job, flag: np.ndarray) -> str:
    """Return a text on whether a solar-free run matches experiment 697's stored control.

    The solar-free twin of D8 with seed 0 to 2 draws the days of experiment 697's `c1_null` control
    of the same seed, and the twin of D3 with seed 0 or 1 draws those of `c2_d3_solar_null`, so the
    smoothed labels must equal the stored ones.
    """
    from e00697_controls import CONTROLS_DIR  # ty: ignore[unresolved-import]

    if job.kind != "host" or job.capacity_mw is not None or job.seed is None:
        return ""
    names = {"d8pre": f"c1_null_s{job.seed}", "d3solar": f"c2_d3_solar_null_s{job.seed}"}
    stored = CONTROLS_DIR / f"{names[job.host]}.npz"
    if not stored.exists():
        return f"; no stored control {stored.name} to compare with"
    with np.load(stored) as data:
        same = bool(np.array_equal(data["flag"], flag))
        share = float(np.mean(data["flag"] == flag))
    return f"; smoothed labels equal experiment 697's {stored.name}: {same} (agreement {share:.4f})"


def _worker(job: Job) -> str:
    return run_job(job=job)


def main() -> None:
    """List the jobs, time one, or run them all."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print the jobs and exit")
    parser.add_argument("--time-one", action="store_true", help="run one job, write nothing")
    parser.add_argument("--only", default="", help="run only the jobs whose name has this text")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--ignore-gate",
        action="store_true",
        help="with --list or --time-one only: list every job the lists allow, with farm A",
    )
    args = parser.parse_args()
    if args.ignore_gate and not (args.list or args.time_one):
        parser.error("--ignore-gate is valid only with --list or --time-one, never for a real run")
    if args.ignore_gate:
        gate = None
    else:
        require_passing_audit()
        gate = read_json(path=GATE_PATH)
    selected = [j for j in all_jobs(gate=gate) if args.only in j.name]
    if not selected:
        parser.error(f"no job's name has the text {args.only!r}")
    if args.list:
        for job in selected:
            print(job)
        print(f"{len(selected)} jobs")
        return
    farm = "A" if gate is None else str(gate["farm"])
    init(farm=farm)
    if args.time_one:
        print(run_job(job=selected[0], write=False))
        return
    started = datetime.now(UTC)
    log: list[str] = []
    with get_context("fork").Pool(args.workers) as pool:
        for line in pool.imap_unordered(_worker, selected, chunksize=1):
            print(line, flush=True)
            log.append(line)
    log.append(
        f"{len(selected)} jobs, {args.workers} workers, "
        f"done in {(datetime.now(UTC) - started).total_seconds():.0f} s"
    )
    print(log[-1])
    if not args.only:
        PRINTED_DIR.mkdir(parents=True, exist_ok=True)
        (PRINTED_DIR / "jobs.txt").write_text("\n".join(sorted(log)) + "\n")


if __name__ == "__main__":
    main()

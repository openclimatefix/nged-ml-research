"""Experiment 636: run the four worlds and write every step statistic to a private parquet.

Usage: `uv run --no-sync python e00636_run.py [n_workers]`. Writes
`data/ml-experiments/<slug>/stats.parquet`, one row per (world, farm, loss, start, rep, estimator,
horizon). World `none` is the uninjected series, from which the thresholds come. Worlds
`ac_cloudy` and `dc_cloudy` are worlds `ac` and `dc` at each farm's three cloudiest start months.
"""

import sys
from multiprocessing import Pool

import e00636_common as c
import polars as pl

CURTAILMENT_SHARE = 0.05
CURTAILMENT_REPS = 3
CLOUDY_LOSSES = (0.05, 0.10)
N_CLOUDY_PLACEBO = 10
Task = tuple[str, str, float, int, int]
FARMS: dict[str, c.Farm] = {}


def _init() -> None:
    FARMS.update(c.load_farms())


def run_task(task: Task) -> list[dict]:
    """Return the step statistics of one injected series, for every estimator and horizon."""
    world, site, loss, start, rep = task
    farm = FARMS[site]
    kind = world.removesuffix("_cloudy")
    if kind == "none":
        y = farm.y
    elif kind == "ac":
        y = c.inject_ac_loss(farm, start=start, loss=loss)
    elif kind == "dc":
        y = c.inject_dc_loss(farm, start=start, loss=loss)
    else:
        seed = 1000 * start + 10 * rep + c.TESTABLE.index(site)
        y = c.inject_curtailment(farm, start=start, share=CURTAILMENT_SHARE, seed=seed)
    levels = c.monthly_levels(farm, y)
    return [
        {
            "world": world,
            "farm": site,
            "loss": loss,
            "start": start,
            "rep": rep,
            "est": est,
            "k": k,
            "stat": c.step_statistic(months, level, start=start, horizon=k),
        }
        for est, (months, level) in levels.items()
        for k in c.HORIZONS
    ]


def build_tasks() -> list[Task]:
    """Return one task per injected series."""
    farms = c.load_farms()
    standard = [c.month_index(y, m) for y, m in c.START_MONTHS]
    tasks: list[Task] = []
    for site in c.TESTABLE:
        cloudy = c.cloudiest_months(farms[site], n=N_CLOUDY_PLACEBO)
        tasks += [("none", site, 0.0, s, 0) for s in sorted(set(standard) | set(cloudy))]
        for loss in c.LOSSES:
            for s in standard:
                tasks += [("ac", site, loss, s, 0), ("dc", site, loss, s, 0)]
        for loss in CLOUDY_LOSSES:
            for s in cloudy[:3]:
                tasks += [("ac_cloudy", site, loss, s, 0), ("dc_cloudy", site, loss, s, 0)]
        for s in standard:
            tasks += [("curtail", site, 0.0, s, r) for r in range(CURTAILMENT_REPS)]
    return tasks


def main() -> None:
    """Run every task on a pool of workers and write the statistics table."""
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tasks = build_tasks()
    out: list[dict] = []
    with Pool(n_workers, initializer=_init) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_task, tasks, chunksize=4)):
            out += rows
            if i % 200 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    c.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).write_parquet(c.DATA_DIR / "stats.parquet")


if __name__ == "__main__":
    main()

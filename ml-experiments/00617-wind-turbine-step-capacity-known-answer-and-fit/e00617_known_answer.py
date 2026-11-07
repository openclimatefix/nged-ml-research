"""Experiment 617, part A: known-answer test and placebos of the turbine-step estimator.

For each farm the real daily series (fault days removed) is the base. Each run injects outages of
`m` turbines of a farm with `n_true` turbines, each lasting `duration` days, by multiplying the
daily value by `1 - m / n_true` on the outage days. Writes `known_answer_events.parquet`,
`known_answer_runs.parquet`, and `placebos.parquet` in `DATA_DIR`.

```bash
export DATA_PATH_INTERNAL=...
PYTHONPATH=ml-experiments/00617-wind-turbine-step-capacity-known-answer-and-fit \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00617-*/e00617_known_answer.py
```
"""

import itertools
import multiprocessing as mp
import os
from dataclasses import replace
from typing import Final

import e00617_hmm as h
import numpy as np
import polars as pl

STEP_UP_GRID: Final[tuple[float, ...]] = (0.001, 0.003, 0.01)
N_TRUE: Final[tuple[int, ...]] = (6, 10, 14)
OUTAGE_TURBINES: Final[tuple[int, ...]] = (1, 2, 3)
DURATIONS: Final[tuple[int, ...]] = (3, 7, 14, 28, 60)
EVENTS_PER_RUN: Final[int] = 15
RUNS_PER_CELL: Final[int] = 3
SPACING_DAYS: Final[int] = 130
BEFORE_DAYS: Final[int] = 30
N_PERMUTATIONS: Final[int] = 30
MIN_EPISODE_DAYS: Final[int] = 3
PROCESSES: Final[int] = 4
BLOCK_DAYS_OBS: Final[int] = (
    8  # block permutation: about 8 observed days (30 calendar days) a block
)


def _starts(rng: np.random.Generator, n_days: int, duration: int, n: int) -> np.ndarray:
    """Return `n` start days at least `SPACING_DAYS` apart (after the previous end)."""
    slots = (n_days - BEFORE_DAYS) // (SPACING_DAYS + duration)
    n = min(n, slots)
    gaps = rng.multinomial(
        max(n_days - BEFORE_DAYS - n * (SPACING_DAYS + duration), 0), np.ones(n + 1) / (n + 1)
    )
    return BEFORE_DAYS + np.cumsum(gaps[:n]) + np.arange(n) * (SPACING_DAYS + duration)


def count_steps(deficit: np.ndarray, unit: float) -> int:
    """Count upward steps of whole turbines in the rounded deficit, after a 3-day median filter.

    A step is a rise in the rounded number of turbines out that lasts at least `MIN_EPISODE_DAYS`.
    """
    level = np.round(deficit / unit).astype(int)
    padded = np.pad(level, 1, mode="edge")
    level = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0).astype(int)
    steps, i = 0, 1
    while i < len(level):
        if level[i] > level[i - 1]:
            if (level[i : i + MIN_EPISODE_DAYS] >= level[i]).all():
                steps += level[i] - level[i - 1]
            i += MIN_EPISODE_DAYS
        else:
            i += 1
    return int(steps)


def _run(job: tuple) -> list[dict]:
    site, step_up, n_true, m, duration, seed = job
    rng = np.random.default_rng(seed)
    base = h.remove_fault_days(h.daily_series(site))
    starts = _starts(rng, base.n_days, duration, EVENTS_PER_RUN)
    factor = np.ones(base.n_days)
    truth = np.zeros(base.n_days)
    for s0 in starts:
        factor[s0 : s0 + duration] = 1 - m / n_true
        truth[s0 : s0 + duration] = m * base.ceiling / n_true
    value = base.value * factor[base.day]
    shaped = replace(base, value=value, ceiling=float(np.quantile(value, 0.99)))
    fits = h.fit_grid(shaped, step_up=step_up)
    weights = h.log_n_weights(fits)
    summary = h.deficit_summary(fits, weights)
    n_map = float(h.N_GRID[int(np.argmax(weights))])
    unit_true = base.ceiling / n_true  # injected size uses the clean ceiling
    rows = []
    covered = (summary["q05"] <= truth + 1e-9) & (truth <= summary["q95"] + 1e-9)
    seen = np.zeros(base.n_days, dtype=bool)
    seen[base.day] = True
    for s0 in starts:
        core = slice(s0 + duration // 4, s0 + duration - duration // 4)
        pre = slice(max(s0 - BEFORE_DAYS, 0), s0)
        observed = int(np.isin(np.arange(s0, s0 + duration), base.day).sum())
        step = float(np.median(summary["mean"][core]) - np.median(summary["mean"][pre]))
        rows.append(
            {
                "site": site,
                "step_up": step_up,
                "n_true": n_true,
                "m": m,
                "duration": duration,
                "seed": seed,
                "start_day": int(s0),
                "observed_days": observed,
                "step_mw": step,
                "true_step_mw": m * unit_true,
                "step_in_turbines": step / unit_true,
                "detected": step >= 0.5 * m * unit_true,
                "n_map": n_map,
                "n_weight_true": float(weights[np.argmin(np.abs(h.N_GRID - n_true))]),
                "run_covered_share": float(covered[truth > 0].mean()),
                "run_covered_all": float(covered.mean()),
                "run_covered_seen": float(covered[seen].mean()),
                "run_covered_seen_outage": float(covered[seen & (truth > 0)].mean()),
            }
        )
    return rows


def _placebo(job: tuple) -> dict:
    site, step_up, kind, seed = job
    rng = np.random.default_rng(seed)
    base = h.daily_series(site)
    if kind in ("permuted", "block_permuted"):
        if kind == "permuted":
            order = rng.permutation(len(base.day))
        else:
            blocks = np.array_split(
                np.arange(len(base.day)), max(len(base.day) // BLOCK_DAYS_OBS, 1)
            )
            order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        shaped = replace(base, value=base.value[order])
        true_end = base.ceiling
    else:
        t0, t1 = int(base.n_days * 0.25), int(base.n_days * 0.25) + int(3 * 365.25)
        ramp = np.clip(1 - 0.1 * (base.day - t0) / (t1 - t0), 0.9, 1.0)
        shaped = replace(base, value=base.value * ramp)
        true_end = 0.9 * base.ceiling
    fits = h.fit_grid(shaped, step_up=step_up)
    weights = h.log_n_weights(fits)
    summary = h.deficit_summary(fits, weights)
    unit = base.ceiling / float(h.N_GRID[int(np.argmax(weights))])
    n_steps = count_steps(summary["mean"], unit)
    return {
        "site": site,
        "step_up": step_up,
        "kind": kind,
        "seed": seed,
        "episodes": n_steps,
        "false_per_year": n_steps / base.years,
        "years": base.years,
        "end_capacity_error": float((base.ceiling - summary["mean"][-1]) / true_end - 1),
    }


def main() -> None:
    """Run the known-answer events and the placebos and write them to `DATA_DIR`."""
    h.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cells = list(itertools.product(h.SITES, STEP_UP_GRID, N_TRUE, OUTAGE_TURBINES, DURATIONS))
    jobs = [(*c, 1000 * r + i) for i, c in enumerate(cells) for r in range(RUNS_PER_CELL)]
    placebo_jobs = [
        (site, a, kind, 7000 + 100 * si + 10_000 * ai + i)
        for si, site in enumerate(h.SITES)
        for ai, a in enumerate(STEP_UP_GRID)
        for kind, reps in (
            ("permuted", N_PERMUTATIONS),
            ("block_permuted", N_PERMUTATIONS),
            ("drift", 1),
        )
        for i in range(reps)
    ]
    ctx = mp.get_context("spawn")
    os.environ["OMP_NUM_THREADS"] = "1"
    with ctx.Pool(PROCESSES) as pool:
        placebos = pool.map(_placebo, placebo_jobs, chunksize=4)
        pl.DataFrame(placebos).write_parquet(h.DATA_DIR / "placebos.parquet")
        events = [r for rows in pool.imap_unordered(_run, jobs, chunksize=4) for r in rows]
    pl.DataFrame(events).write_parquet(h.DATA_DIR / "known_answer_events.parquet")


if __name__ == "__main__":
    main()

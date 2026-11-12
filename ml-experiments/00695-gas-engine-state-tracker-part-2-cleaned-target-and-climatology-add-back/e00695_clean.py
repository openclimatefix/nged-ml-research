# ruff: noqa: E501
r"""Build the cleaned power rows of experiment 695's arms, one file per arm and fold.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00695-*/e00695_clean.py ARM [--months M ...] [--workers N]

ARM is one of:

- `A1` (D8, causal): fixed-lag states with a training cut at each fold's refit.
- `A1n` (D8): as A1, cleaned without the fold (reading plus the units' output).
- `AH` (D8, hindsight): the tracker fitted and smoothed to 2026-06-30, the same rows for every fold.
- `W0` (host, injected engine, raw), `W1` (host, injected engine, causal cleaning).

Each fold gets `stacks/<ARM>/<month>.npz` (the series' row of the network's power-over-capacity
stack) and `states/<ARM>/<month>.npz` (state marginals, level means, and the era fits). Era 1 (from
2023-08-15 to 2025-03-23) is fitted once, because it ends before every refit. Era 2 (from
2025-03-24) is refitted per fold on readings that had arrived by the refit, and is never split at
the fade. Readings from 2026-07-01 are set to NaN before anything is fitted or tracked.
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00677_extend as ext  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00695_common import (
    D8_LABEL,
    DATA_DIR,
    ERA1_START,
    ERA2_START,
    FOLD_MONTHS,
    HINDSIGHT_END,
    HOLDOUT,
    READING_DELAY,
    SHARED_DIR,
    fold_refit,
    grid_index,
)
from e00695_engine import drive_series, inject_engine
from e00695_tracker import (
    EraFit,
    States,
    cleaned_series,
    fit_era,
    fold_branch_share,
    spread_levels,
    track_segment,
)

ARMS: Final[tuple[str, ...]] = ("A1", "A1n", "AH", "W0", "W1")


@dataclass(frozen=True)
class SeriesSetup:
    """One series to clean: its label, stack row, capacity, readings, and the engine injected."""

    label: str
    row: int
    capacity: float
    y: np.ndarray
    raw_row: np.ndarray
    truth_g: np.ndarray | None


def load_setup(*, arm: str) -> SeriesSetup:
    """Return the series an arm cleans, with the held-out readings hidden.

    The A arms clean D8's own reading. The W arms clean the host's reading with Gas A's day-block
    output subtracted, the host named in `host.json` by `e00695_phase_a.py`.
    """
    ext.redirect()
    stack = c.clean_stack()
    table = prep.primaries().select("site", "time_series_id")
    capacity = pl.read_parquet(SHARED_DIR / "series_capacity.parquet").select(
        "time_series_id", "capacity"
    )
    joined = table.join(capacity, on="time_series_id")
    caps = dict(zip(joined["site"], joined["capacity"], strict=True))
    label = (
        D8_LABEL
        if arm in {"A1", "A1n", "AH"}
        else json.loads((DATA_DIR / "host.json").read_text())["host"]
    )
    row = c.sites().index(label)
    raw_row = stack[row].astype(np.float64) * float(caps[label])
    y = raw_row.copy()
    y[grid_index(when=HOLDOUT) :] = np.nan
    truth_g = None
    if arm in {"W0", "W1"}:
        truth_g, _ = drive_series(n_grid=len(y))
        y = inject_engine(y=y, g=truth_g)
    return SeriesSetup(
        label=label, row=row, capacity=float(caps[label]), y=y, raw_row=raw_row, truth_g=truth_g
    )


def _last_data(*, y: np.ndarray) -> int:
    """Return one past the last finite reading."""
    return int(np.flatnonzero(np.isfinite(y))[-1]) + 1


def era_bounds(*, y: np.ndarray) -> tuple[int, int, int]:
    """Return the first half-hour of era 1, the first of era 2, and one past the last reading."""
    return grid_index(when=ERA1_START), grid_index(when=ERA2_START), _last_data(y=y)


def era1_states(*, setup: SeriesSetup, hindsight: bool) -> tuple[States, EraFit]:
    """Fit era 1 once and track it (a fixed lag, or the full backward pass for the hindsight arm).

    Era 1 ends before every refit, so one fit and one set of states serve every fold. The result is
    cached in the data folder.
    """
    cache = (
        DATA_DIR
        / "era1"
        / f"{setup.label}_{'AH' if hindsight else 'causal'}_{int(setup.truth_g is not None)}.pkl"
    )
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    first, second, _ = era_bounds(y=setup.y)
    segment = setup.y[first:second]
    fit = fit_era(y=segment, lvl=spread_levels(y=segment))
    states = track_segment(y=segment, fit=fit, refit=None, lag=None if hindsight else 40)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps((states, fit)))
    tmp.replace(cache)
    return states, fit


def era2_fit(*, y: np.ndarray, fit_stop: int) -> EraFit:
    """Fit era 2 on the readings before `fit_stop` (an index into `y`, which starts at era 2)."""
    segment = y[:fit_stop]
    return fit_era(y=segment, lvl=spread_levels(y=y)[:fit_stop])


def fold_states(
    *, setup: SeriesSetup, month: str | None, hindsight: bool, mode: str
) -> tuple[np.ndarray, States, EraFit, dict]:
    """Return the states of every half-hour of D8 or the host for one fold, and the era 2 fit.

    Args:
        setup: The series.
        month: The fold's test month, `None` for the hindsight arm.
        hindsight: Whether to fit and smooth with every reading to 2026-06-30.
        mode: `folded` or `signed` cleaning of the half-hours with a unit running.

    Returns:
        The cleaned reading (MVA) on the whole grid, the states on the whole grid (k = 0 before era 1),
        the era 2 fit, and a small report.
    """
    n_grid = len(setup.y)
    first, second, end = era_bounds(y=setup.y)
    pk = np.zeros((n_grid, 4))
    pk[:, 0] = 1.0
    lmean = setup.y.copy()
    s1, fit1 = era1_states(setup=setup, hindsight=hindsight)
    pk[first:second], lmean[first:second] = s1.pk, s1.lmean
    segment = setup.y[second:end]
    if hindsight:
        stop = min(end, grid_index(when=HINDSIGHT_END)) - second
        fit2 = era2_fit(y=segment, fit_stop=stop)
        s2 = track_segment(y=segment[:stop], fit=fit2, refit=None, lag=None)
        pk[second : second + stop], lmean[second : second + stop] = s2.pk, s2.lmean
    else:
        assert month is not None
        refit = grid_index(when=fold_refit(month=month)) - second
        fit2 = era2_fit(y=segment, fit_stop=refit - READING_DELAY)
        s2 = track_segment(y=segment, fit=fit2, refit=refit, reading_delay=READING_DELAY)
        pk[second:end], lmean[second:end] = s2.pk, s2.lmean
    states = States(pk=pk, lmean=lmean)
    era2 = States(pk=pk[second:end], lmean=lmean[second:end])
    cleaned = setup.y.copy()
    cleaned[first:second] = cleaned_series(y=setup.y[first:second], states=s1, fit=fit1, mode=mode)
    cleaned[second:end] = cleaned_series(y=segment, states=era2, fit=fit2, mode=mode)
    wrong, running = fold_branch_share(y=segment, states=era2, fit=fit2)
    report = {
        "era1_k": fit1.k_max,
        "era1_unit": fit1.unit,
        "era2_k": fit2.k_max,
        "era2_unit": fit2.unit,
        "branch_differs": wrong,
        "branch_running": running,
    }
    return cleaned, states, fit2, report


def build_one(args: tuple[str, str | None]) -> dict:
    """Build and write one fold's stack row and states for one arm."""
    arm, month = args
    setup = load_setup(arm=arm)
    out = DATA_DIR / "stacks" / arm
    out.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "states" / arm).mkdir(parents=True, exist_ok=True)
    name = month if month is not None else "all"
    if arm == "W0":
        cleaned = setup.y
        states = fit2 = None
        report: dict = {}
    else:
        cleaned, states, fit2, report = fold_states(
            setup=setup,
            month=month,
            hindsight=arm == "AH",
            mode="signed" if arm == "A1n" else "folded",
        )
    before = np.arange(len(cleaned)) < grid_index(when=HOLDOUT)
    values = np.where(before, cleaned, setup.raw_row) / setup.capacity
    assert np.array_equal(np.isfinite(values), np.isfinite(setup.raw_row)), (
        "the cleaned row changed which half-hours exist"
    )
    target = out / f"{name}.npz"
    np.savez(out / f"{name}.tmp.npz", row=setup.row, values=values.astype(np.float32))
    (out / f"{name}.tmp.npz").replace(target)
    if states is not None and fit2 is not None:
        folder = DATA_DIR / "states" / arm
        np.savez(
            folder / f"{name}.tmp.npz",
            pk=states.pk.astype(np.float32),
            lmean=states.lmean.astype(np.float32),
            unit=fit2.unit,
            k_max=fit2.k_max,
        )
        (folder / f"{name}.tmp.npz").replace(folder / f"{name}.npz")
    report |= {"arm": arm, "month": name, "label": setup.label}
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    """Build the arm's files for the requested folds, in parallel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--months", nargs="+", default=list(FOLD_MONTHS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.arm == "AH":
        tasks: list[tuple[str, str | None]] = [("AH", None)]
    else:
        tasks = [(args.arm, m) for m in args.months]
    if args.arm in {"A1", "A1n", "W1", "AH"}:
        # fit era 1 once in the parent, so that the workers read the cache
        era1_states(setup=load_setup(arm=args.arm), hindsight=args.arm == "AH")
    with get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        reports = pool.map(build_one, tasks, chunksize=1)
    (DATA_DIR / "stats").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "stats" / f"clean_{args.arm}.json").write_text(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main()

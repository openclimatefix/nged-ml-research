# ruff: noqa: E501
r"""The engine-state columns that the pre-training mixer reads, built from experiment 695's tracker states.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00696-*/e00696_state.py ARM [--months M ...]

A state file holds the three columns on the pipeline grid, one row per half-hour, for one arm and one
fold: the expected engine output as a share of capacity, the probability that one or more units run,
and a hidden flag (always 0 here; the arm S1d sets it on the training rows). The functions below are
pure NumPy, so the tests run on small synthetic inputs.

**Cut.** Experiment 695's per-fold state files hide every reading from the refit minus 3.5 hours onward
for the half-hours before the refit, and pre-training rows end at the refit minus 3.5 hours, so no state
column used in training reads a reading after its cut. The placebo permutes whole days only inside that
cut.
"""

import argparse
import pickle
from pathlib import Path
from typing import Final

import numpy as np
from e00695_common import (
    D8_CAPACITY_MVA,
    ERA1_START,
    ERA2_START,
    FOLD_MONTHS,
    PIPELINE_GRID_START,
    READING_DELAY,
    fold_refit,
    grid_index,
)
from e00696_common import ARMS, DATA_DIR, E695_DIR, N_STATE

DAY: Final[int] = 48
"""Half-hours in a UTC day."""
HIDE_SEED_SPAN: Final[int] = 100_000
"""Days drawn when choosing which origin days S1d hides, far more than the grid holds."""


def unit_vector(*, n_grid: int, era2_first: int, unit1: float, unit2: float) -> np.ndarray:
    """Return the engine unit size (MVA) of every half-hour: era 1's before `era2_first`, era 2's from it."""
    unit = np.full(n_grid, unit1)
    unit[era2_first:] = unit2
    return unit


def tracked_state(*, pk: np.ndarray, unit: np.ndarray, capacity: float) -> np.ndarray:
    """Return the tracked state columns, (half-hours, 3), float32.

    Args:
        pk: The smoothed marginals P(k = j), (half-hours, units + 1).
        unit: The unit size (MVA) of each half-hour.
        capacity: The series' effective capacity (MVA).

    Returns:
        The expected output as a share of capacity (the sum over unit counts j of j times the unit
        size times P(k = j), over the capacity), the probability that one or more units run, and a
        hidden flag of 0.
    """
    expected = (pk * np.arange(pk.shape[1])[None, :]).sum(axis=1) * unit / capacity
    return np.stack([expected, 1.0 - pk[:, 0], np.zeros(len(pk))], axis=1).astype(np.float32)


def true_state(*, output: np.ndarray, units: np.ndarray, capacity: float) -> np.ndarray:
    """Return the true injected engine's state columns: its output over capacity, whether any unit runs, 0."""
    return np.stack(
        [output / capacity, (units > 0).astype(np.float64), np.zeros(len(output))], axis=1
    ).astype(np.float32)


def _day_calendar(*, first_day: int, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the calendar month number and the weekend flag of days `first_day` to `first_day + n_days - 1`."""
    start = np.datetime64(PIPELINE_GRID_START.replace(tzinfo=None), "D")
    days = start + np.arange(first_day, first_day + n_days)
    months = days.astype("datetime64[M]").astype(int) % 12 + 1
    weekday = (days.astype(int) + 3) % 7  # 1970-01-01 was a Thursday; Monday is 0
    return months, weekday >= 5


def permute_days(*, state: np.ndarray, first: int, stop: int, seed: int) -> np.ndarray:
    """Return `state` with whole UTC days shuffled within calendar month and weekday class.

    Only half-hours from `first` (a day boundary) up to `stop` (exclusive) are touched, so no day
    after a fold's cut is read. A complete day takes the content of a day drawn without replacement
    from the same calendar month and the same weekday class (Monday to Friday, or Saturday and Sunday)
    over every year in the span, so the permutation keeps the state's seasonal profile and within-day
    shape and loses which days the engine ran. The trailing partial day takes the first half-hours of
    a complete day of the same group, drawn with replacement. A group of one day keeps its day.

    Args:
        state: The state columns on the grid, (half-hours, columns).
        first: The first half-hour of the span, which must start a UTC day.
        stop: One past the last half-hour that may be read.
        seed: The seed of the permutation.

    Returns:
        A copy of `state` with the span permuted.
    """
    assert first % DAY == 0, "the span must start on a UTC day boundary"
    n_full, partial = divmod(stop - first, DAY)
    first_day = first // DAY
    months, weekend = _day_calendar(first_day=first_day, n_days=n_full + (1 if partial else 0))
    rng = np.random.default_rng(seed)
    out = state.copy()
    blocks = state[first : first + n_full * DAY].reshape(n_full, DAY, -1)
    destination = np.empty_like(blocks)
    for month in range(1, 13):
        for is_weekend in (False, True):
            group = np.flatnonzero((months[:n_full] == month) & (weekend[:n_full] == is_weekend))
            if len(group):
                destination[group] = blocks[rng.permutation(group)]
    out[first : first + n_full * DAY] = destination.reshape(n_full * DAY, -1)
    if partial:
        same_class = (months[:n_full] == months[n_full]) & (weekend[:n_full] == weekend[n_full])
        same_month = months[:n_full] == months[n_full]
        pool = next(
            (np.flatnonzero(mask) for mask in (same_class, same_month) if mask.any()),
            np.arange(n_full),
        )
        out[first + n_full * DAY : stop] = blocks[int(rng.choice(pool))][:partial]
    return out


def rows_state(
    *, grid_state: np.ndarray, row: int, primary: np.ndarray, target_idx: np.ndarray
) -> np.ndarray:
    """Return the (rows, 3) state columns of pre-training rows; other Primaries get zeros."""
    out = np.zeros((len(primary), grid_state.shape[1]), dtype=np.float32)
    mine = primary == row
    out[mine] = grid_state[target_idx[mine]]
    return out


def hide_half_of_origin_days(
    *, state: np.ndarray, origin_idx: np.ndarray, seed: int, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows' state with a random half of origin days hidden, and the hidden mask.

    A hidden row of the series reads zeros in the first two columns and 1 in the third (the hidden
    flag). The draw is one coin per UTC origin day, fixed by `seed`, so every row of a day shares the
    decision. A row of another Primary reads zeros and no flag whatever its day.

    Args:
        state: The rows' state columns, (rows, 3).
        origin_idx: Each row's origin half-hour.
        seed: The seed of the coins.
        rows: Which rows belong to the series that carries the state.

    Returns:
        The state with the hidden rows set, and the mask of hidden rows.
    """
    coins = np.random.default_rng(seed).random(HIDE_SEED_SPAN) < 0.5
    hidden = coins[origin_idx // DAY] & rows
    out = state.copy()
    out[hidden, : N_STATE - 1] = 0.0
    out[hidden, N_STATE - 1] = 1.0
    return out, hidden


def state_path(*, arm: str, month: str) -> Path:
    """Return the state file of an arm and a fold."""
    return DATA_DIR / "states" / arm / f"{month}.npz"


def _tracker_inputs(*, source: str, month: str) -> tuple[np.ndarray, float, float, int]:
    """Return experiment 695's marginals, era 1's and era 2's unit sizes, and the stack row for a source arm."""
    states = np.load(E695_DIR / "states" / source / f"{month}.npz")
    label, injected = ("D8", 0) if source == "A1" else ("D4", 1)
    with (E695_DIR / "era1" / f"{label}_causal_{injected}.pkl").open("rb") as handle:
        _, fit1 = pickle.load(handle)
    row = int(np.load(E695_DIR / "stacks" / source / f"{month}.npz")["row"])
    return states["pk"].astype(np.float64), float(fit1.unit), float(states["unit"]), row


def build_fold(*, arm: str, month: str) -> Path:
    """Write the state file of an arm for one fold and return its path.

    The file holds `state` (grid half-hours, 3, float32), `row` (the stack row of the series that
    carries the state), `refit` and `stop` (the fold's refit half-hour and the first half-hour a
    training row may not read).
    """
    spec = ARMS[arm]
    refit = grid_index(when=fold_refit(month=month))
    stop = refit - READING_DELAY
    first = grid_index(when=ERA1_START)
    era2 = grid_index(when=ERA2_START)
    if spec.series == "host":
        from e00695_engine import drive_series
        from e00695_score import capacity_of, host_series_id

        capacity = capacity_of()[host_series_id()]
        row = int(np.load(E695_DIR / "stacks" / "W0" / f"{month}.npz")["row"])
    else:
        capacity = D8_CAPACITY_MVA
        row = int(np.load(E695_DIR / "stacks" / "A1" / f"{month}.npz")["row"])
    if spec.kind in {"tracked", "placebo", "hidden"}:
        assert spec.source is not None
        pk, unit1, unit2, row_source = _tracker_inputs(source=spec.source, month=month)
        if spec.series == "d8":
            assert row_source == row
        grid = tracked_state(
            pk=pk,
            unit=unit_vector(n_grid=len(pk), era2_first=era2, unit1=unit1, unit2=unit2),
            capacity=capacity,
        )
        if spec.kind == "placebo":
            grid = permute_days(state=grid, first=first, stop=stop, seed=0)
    else:
        n_grid = len(np.load(E695_DIR / "states" / "A1" / f"{month}.npz")["pk"])
        if spec.kind == "true":
            output, units = drive_series(n_grid=n_grid)
            grid = true_state(output=output, units=units, capacity=capacity)
        else:
            grid = np.zeros((n_grid, N_STATE), dtype=np.float32)
    target = state_path(arm=arm, month=month)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.npz")
    np.savez(tmp, state=grid, row=row, refit=refit, stop=stop)
    tmp.replace(target)
    print(
        f"{arm} {month}: row {row}, share of half-hours with P(any unit runs) above 0.5 {float((grid[:, 1] > 0.5).mean()):.3f}"
    )
    return target


def main() -> None:
    """Write the state files of the requested arms and folds."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arms", nargs="+", choices=sorted(a for a, s in ARMS.items() if s.kind != "none")
    )
    parser.add_argument("--months", nargs="+", default=list(FOLD_MONTHS))
    args = parser.parse_args()
    for arm in args.arms:
        for month in args.months:
            build_fold(arm=arm, month=month)


if __name__ == "__main__":
    main()

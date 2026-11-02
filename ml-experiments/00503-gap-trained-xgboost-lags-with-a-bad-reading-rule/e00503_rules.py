"""Serving rules of experiment 503 that set a power lag to missing.

Each rule looks at the reading behind a lag (the reading at the lag's time) using only readings
no younger than the newest usable one, `origin - NEWEST_AGE`, so no rule sees the future:

- (i) **zero:** the reading is exactly 0. A flagged half-hour is already NaN in the readings, and
  a NaN lag is missing without any rule.
- (ii) **far:** the reading is more than `FAR_MADS` times the Primary's stage-1 residual median
  absolute deviation (MAD) away from experiment 391's stage-1 median at the lag's time. Where the
  stage-1 table has no row, the rule is off and the case is counted by `table_gap`.
- (iii) **stuck:** the lag's time lies in a run of at least `STUCK_RUN` identical readings, counting
  only readings up to the newest usable one.
"""

from datetime import datetime
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

STAGE1_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00391-xgboost-lag-surprise-second-stage" / "work" / "stage1" / "ens"
)
FAR_MADS: Final[float] = 6.0
STUCK_RUN: Final[int] = 12
RULE_NAMES: Final[tuple[str, ...]] = ("zero", "far", "stuck")
WEEK_STEPS: Final[int] = 7 * common.STEPS_PER_DAY


def stage1_median(*, site: str, capacity_mw: float, refit: datetime) -> tuple[np.ndarray, float]:
    """Return the stage-1 median on the grid as a fraction of capacity, and the residual MAD.

    The MAD is of the stage-1 residual `r_mw` (the reading minus the median) over the rows before
    `refit`, as a fraction of capacity, so no later reading sets the scale.

    Args:
        site: Site label such as `D5`.
        capacity_mw: The series' effective capacity.
        refit: The refit date; only residuals of earlier valid times set the MAD.

    Returns:
        The grid array (NaN where the table has no row) and the MAD.
    """
    table = pl.read_parquet(STAGE1_DIR / f"{site}.parquet").select(
        "valid_time", "median_mw", "r_mw"
    )
    grid = np.full(common.grid_times().len(), np.nan, dtype=np.float32)
    index = np.asarray(common.grid_index(times=table["valid_time"]))
    grid[index] = (table["median_mw"] / capacity_mw).to_numpy()
    earlier = table.filter(pl.col("valid_time") < refit)["r_mw"].drop_nulls().to_numpy()
    mad = float(np.median(np.abs(earlier - np.median(earlier)))) / capacity_mw
    return grid, mad


def lag_indices(*, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the grid index of the reading behind each lag, shape (cases, 3), in lag order.

    The indices are those of `e00087_common.origin_lags`, which asserts none is younger than the
    newest usable observation.
    """
    newest = origin - common.NEWEST_AGE
    gap = target - newest
    day = target - common.STEPS_PER_DAY * np.ceil(gap / common.STEPS_PER_DAY).astype(int)
    week = target - WEEK_STEPS * np.ceil(gap / WEEK_STEPS).astype(int)
    if (day > newest).any() or (week > newest).any():
        msg = "a lag index is younger than the newest usable observation"
        raise ValueError(msg)
    return np.stack([newest, day, week], axis=1)


def run_counts(*, power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return, per grid index, the identical readings ending there and starting there.

    A NaN reading is never part of a run (both counts are 0).
    """
    size = power.size
    index = np.arange(size)
    finite = np.isfinite(power)
    same_as_previous = np.zeros(size, dtype=bool)
    same_as_previous[1:] = (power[1:] == power[:-1]) & finite[1:]
    starts = np.maximum.accumulate(np.where(finite & ~same_as_previous, index, -1))
    back = np.where(finite, index - starts + 1, 0)
    same_as_next = np.zeros(size, dtype=bool)
    same_as_next[:-1] = same_as_previous[1:]
    ends = np.minimum.accumulate(np.where(finite & ~same_as_next, index, size)[::-1])[::-1]
    forward = np.where(finite, ends - index + 1, 0)
    return back.astype(np.int64), forward.astype(np.int64)


def rule_flags(
    *,
    power: np.ndarray,
    index: np.ndarray,
    newest: np.ndarray,
    median: np.ndarray,
    mad: float,
    runs: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return a boolean (cases, 3) array per rule: True where the rule sets that lag to missing.

    Args:
        power: Readings on the grid, as a fraction of capacity.
        index: Output of `lag_indices`.
        newest: Grid index of each case's newest usable reading, shape (cases,).
        median: Stage-1 median on the grid (NaN where unknown).
        mad: The stage-1 residual MAD.
        runs: Output of `run_counts` for `power`.

    Returns:
        A dict keyed by `RULE_NAMES`, plus `table_gap` (True where rule `far` is off for lack of a
        stage-1 value).
    """
    reading = common.take(values=power, index=index)
    stage1 = common.take(values=median, index=index)
    back, forward = runs
    visible_ahead = np.minimum(common.take(values=forward.astype(np.float32), index=index) - 1,
                               (newest[:, None] - index)).astype(np.int64)
    run = common.take(values=back.astype(np.float32), index=index).astype(np.int64) + visible_ahead
    return {
        "zero": reading == 0.0,
        "far": np.isfinite(stage1) & (np.abs(reading - stage1) > FAR_MADS * mad),
        "stuck": np.isfinite(reading) & (run >= STUCK_RUN),
        "table_gap": ~np.isfinite(stage1),
    }


def rules_to_apply(*, name: str) -> tuple[str, ...]:
    """Return the rules of a rule set: `none`, `iii` (zero and stuck), or `iii2` (all three)."""
    return {"none": (), "iii": ("zero", "stuck"), "iii2": RULE_NAMES}[name]


def apply_rules(
    *, matrix: np.ndarray, lag_positions: list[int], flags: dict[str, np.ndarray], rules: tuple
) -> np.ndarray:
    """Return a copy of `matrix` with each flagged lag value set to NaN."""
    out = matrix.copy()
    for rule in rules:
        for column, position in enumerate(lag_positions):
            out[flags[rule][:, column], position] = np.nan
    return out

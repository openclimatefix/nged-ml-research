# ruff: noqa: E501, PT018
"""Experiment 702: the plan's constants and the pure numpy helpers that the tests cover.

Nothing here imports torch or an earlier experiment, so the tests import it without the chain on
`PYTHONPATH`. The fit driver (`e00702_fit.py`), the export (`e00702_export.py`), and the scoring
(`e00702_score.py`) import it.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00702-per-substation-versus-global-training-of-the-modular-demand-network"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
"""The private data folder of this experiment; every fit, forecast, and score is written here."""
NAME_677: Final[str] = "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
ROOT_677: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME_677
"""Experiment 677's data folder, which this experiment reads and never writes."""

ArmType = Literal["split3", "split2", "split1", "hybrid3"]
"""The four per-substation arms; `global` and `blend` have no fit of their own."""
ARMS: Final[tuple[ArmType, ...]] = ("split3", "split2", "split1", "hybrid3")
FIT_ARMS: Final[tuple[str, ...]] = (*ARMS, "global")
"""The arms `e00702_fit.py` fits. The `global` arm is the control refit: it trains on all 16 Primaries at once."""
ARM_LABEL: Final[dict[str, str]] = {
    "global": "global",
    "split3": "split-at-3",
    "split2": "split-at-2",
    "split1": "split-at-1",
    "hybrid3": "split-at-2-plus-encoders",
    "blend": "blend",
    "global_pair": "global-pair",
}
"""The plan's names of the arms, keyed by the short names the file names use."""
TEST_MONTHS: Final[tuple[str, ...]] = ("2025-11", "2026-01", "2026-04", "2026-07")
PILOT_MONTH: Final[str] = "2026-05"
"""The first run's fold, which is not a test month."""
EXTRA_MONTHS: Final[tuple[str, ...]] = ("2025-12", "2026-02", "2026-03", "2026-05", "2026-06")
"""The five folds the maintainer approved after the four-month screen. 2026-08 to 2026-10 lie in the final-test window and are not approved."""
EXTRA_FOLDER: Final[str] = "extra_months"
"""The subfolder of `backtests/` that holds the final exports of `EXTRA_MONTHS`, one `<arm>.parquet` per arm."""
SCORABLE_MONTHS: Final[tuple[str, ...]] = tuple(sorted((*TEST_MONTHS, *EXTRA_MONTHS)))
"""The nine months that `e00702_score.py` may score."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
GLOBAL_PAIR_SEEDS: Final[tuple[int, ...]] = (3, 4, 5)
ROW_SET_ID: Final[str] = "backtest_four_weather_months_2025_11_2026_01_2026_04_2026_07_daily_12utc"
N_PRIMARIES: Final[int] = 16

BATCH_GLOBAL: Final[int] = 8192
"""The batch of the global arm, and of the diagnostic fits of the first run."""
BATCH_PER_SUBSTATION: Final[int] = 512
"""The batch of every per-substation phase, one sixteenth of the global batch."""
FIXED_EPOCHS: Final[dict[str, int]] = {"1": 60, "2": 20, "3": 10}
"""Epochs of a phase with too few held rows, keyed by the schedule's step."""
MIN_HELD_WEEKS: Final[int] = 2
MIN_ROWS_PER_WEEK: Final[int] = 1000
"""A phase early-stops only when at least `MIN_HELD_WEEKS` held weeks hold `MIN_ROWS_PER_WEEK` rows."""
WINDOW_DAYS: Final[int] = 730
"""The training window of the fine-tuning rows, in days (`prep.WINDOW_DAYS`)."""
STEPS_PER_DAY: Final[int] = 48
CUTOFF_AGE_STEPS: Final[int] = 7
"""Half-hours between the newest training target and the cutoff (3.5 hours)."""
HEADLINE_LEAD_HOURS: Final[float] = 240.0

N_RESAMPLES: Final[int] = 20_000
N_PLANNED_CONTRASTS: Final[int] = 5
LEVEL: Final[float] = 0.99
"""The planned contrasts' level: 95% with a Bonferroni correction across the five contrasts, rounded to 99%."""
BLOCK_DAYS: Final[int] = 7
GAIN_THRESHOLD_POINTS: Final[float] = -0.05
"""The upper bound at or below which a gain counts as `supported`."""


def needs_fixed_epochs(*, origin_day: np.ndarray, is_val: np.ndarray) -> bool:
    """Return whether a phase has too few held weeks to early-stop on.

    Args:
        origin_day: The origin day of each selected row of the phase.
        is_val: Whether each selected row is an early-stopping row.

    Returns:
        True when fewer than `MIN_HELD_WEEKS` held weeks (`origin_day // 7`) hold at least
        `MIN_ROWS_PER_WEEK` early-stopping rows.
    """
    held = np.asarray(origin_day)[np.asarray(is_val, dtype=bool)] // BLOCK_DAYS
    if len(held) == 0:
        return True
    _, counts = np.unique(held, return_counts=True)
    return int((counts >= MIN_ROWS_PER_WEEK).sum()) < MIN_HELD_WEEKS


def first_target_idx(*, first_origin_idx: int, purged_day: np.ndarray) -> int:
    """Return the grid index of the earliest training target that any fold's rows can hold.

    `e00428_data.build_fold` drops every row whose origin day (days since the first ENS run) is
    purged by `e00422_prep.purge_masks`, and the first days of the 84-day cycle are purged
    (`PURGE_AFTER_DAYS`, 9 days). The earliest kept origin is therefore the first unpurged day of
    the cycle at the run's origin time, and its first target is one half-hour later (lead 0.5 h).

    Args:
        first_origin_idx: Grid index of the first ENS run's origin (`FIRST_RUN` plus the 9-hour
            publication delay).
        purged_day: The purge flag of each day of one cycle, from `purge_masks` on days 0 to 83.

    Returns:
        The grid index of the first target of the first unpurged origin.
    """
    kept = np.flatnonzero(~np.asarray(purged_day, dtype=bool))
    assert len(kept), "every day of the cycle is purged"
    return first_origin_idx + int(kept[0]) * STEPS_PER_DAY + 1


def fold_span(*, cutoff_idx: int, first_idx: int) -> tuple[int, int]:
    """Return the first and last grid index of a fold's training targets from the fold's constants.

    The rows of `e00428_data.build_fold` keep `valid > refit - 730 days` and
    `valid <= refit - 3.5 hours`, so on the half-hourly grid the window starts at
    `cutoff_idx - 730 * 48 + 1` and ends at `cutoff_idx - 7`. No row starts before `first_idx`
    (`first_target_idx`), which is 2024-04-10 09:30 UTC, so a fold whose window reaches back past
    that time starts there instead: 2025-11, 2026-01, and 2026-04 do. A window that starts inside a
    purged stretch of origin days still starts on time when an earlier unpurged origin reaches it at
    a longer lead. The longest lead is 351 hours and the longest purged stretch is 15 days, so a
    window starting in the last hours of that stretch would start later than this formula says; the
    assertion in `split3 spans` catches that case. A per-substation arm passes this span to `synthetic_clean`, so that the schedule
    of shifts depends on the fold and not on a Primary's own first and last reading.
    `e00702_fit.py split3 spans` asserts that the result equals the span of the pooled rows.

    Args:
        cutoff_idx: Grid index of the refit time.
        first_idx: Grid index of the earliest target that any fold can hold (`first_target_idx`).

    Returns:
        The pair `(first, last)`.
    """
    window_first = cutoff_idx - WINDOW_DAYS * STEPS_PER_DAY + 1
    return max(window_first, first_idx), cutoff_idx - CUTOFF_AGE_STEPS


def blend_quantiles(*, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the equal-weight mean of two quantile forecasts, sorted within each row.

    Args:
        first: Quantiles of shape (rows, 11).
        second: Quantiles of the same shape.

    Returns:
        The blend, the rule that already averages seeds: mean per quantile, then sort per row.
    """
    assert first.shape == second.shape, (first.shape, second.shape)
    return np.sort((first + second) / 2.0, axis=-1)


def pair_seeds(*, seeds: tuple[int, ...] = SEEDS) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return the seed-level pairs of the blend contrast (C4).

    Args:
        seeds: The seeds of the global control (seeds 0 to 2).

    Returns:
        For each seed `s`, `((global s, split-at-1 s), (global s, global s + 3))`: the blend of
        global and split-at-1 seed `s` against the mean of global seeds `s` and `s + 3`.
    """
    return [((s, s), (s, s + len(seeds))) for s in seeds]


def array_hash(*, arrays: dict[str, np.ndarray]) -> str:
    """Return a SHA-256 hash of named arrays, over their names, dtypes, shapes, and bytes."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}|{a.dtype}|{a.shape}".encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class StatsRecord:
    """The per-fold, per-Primary statistics of a per-substation arm, with a hash for the prediction phase."""

    month: str
    primary: int
    arrays: dict[str, np.ndarray]

    @property
    def digest(self) -> str:
        """The hash of the arrays."""
        return array_hash(arrays=self.arrays)

    def save(self, *, path: Path) -> None:
        """Write the arrays and the hash atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **{**self.arrays, "_hash": np.array(self.digest)})  # ty: ignore[invalid-argument-type]
        tmp.rename(path)

    @staticmethod
    def check(*, path: Path, arrays: dict[str, np.ndarray]) -> None:
        """Raise unless the arrays hash to the hash stored in `path`."""
        stored = str(np.load(path)["_hash"])
        now = array_hash(arrays=arrays)
        assert stored == now, f"statistics changed since the training phase: {path}"


def restrict_series_table(*, series: pl.DataFrame, label_index: int) -> pl.DataFrame:
    """Return the one-row series table of the Primary at position `label_index` of the sorted Primaries.

    Args:
        series: Experiment 87's `series_table()` as `e00397_prep.primaries()` sees it: the Primary
            rows only, in Primary-index order, with or without the `primary` column.
        label_index: The Primary's index in the 16-row order.

    Returns:
        A frame whose single row is that Primary.
    """
    assert 0 <= label_index < series.height, (label_index, series.height)
    return series.slice(label_index, 1)


def block_ids(*, origin_day: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Return one block id per row: 7-day blocks of origin days that do not cross a month.

    Args:
        origin_day: The origin day (an integer) of each row.
        month: The test month label of each row.

    Returns:
        Integer block ids, equal for rows of the same month whose origin days lie in one block.
    """
    out = np.zeros(len(origin_day), dtype=np.int64)
    key = {}
    for m in sorted(set(month.tolist())):
        sel = month == m
        first = int(origin_day[sel].min())
        for b in np.unique((origin_day[sel] - first) // BLOCK_DAYS):
            key[(m, int(b))] = len(key)
        out[sel] = [key[(m, int(b))] for b in (origin_day[sel] - first) // BLOCK_DAYS]
    return out


def block_seed_bootstrap(
    *,
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    block: np.ndarray,
    n_resamples: int = N_RESAMPLES,
    level: float = LEVEL,
    rng_seed: int = 702,
) -> dict[str, float]:
    """Return the mean paired difference `a - b` and its interval, resampling blocks and one seed.

    Each resample draws as many blocks as there are, with replacement, and one seed (the same seed
    for `a` and `b`, which are already paired seed by seed), and takes the ratio of the drawn blocks'
    summed loss difference to the drawn blocks' row count. The sums per seed and block are computed
    once, so a resample costs one gather.

    Args:
        loss_a: Per-row losses, shape (seeds, rows), of the arm being tested.
        loss_b: Per-row losses of the reference, the same shape and the same rows.
        block: The block id of each row, shape (rows,).
        n_resamples: Number of resamples.
        level: Two-sided interval level.
        rng_seed: Seed of the resampling.

    Returns:
        The keys `mean`, `lo`, `hi`, `n_blocks`, and `n_rows`.
    """
    assert loss_a.shape == loss_b.shape and loss_a.shape[1] == len(block)
    n_seeds = loss_a.shape[0]
    _, inverse = np.unique(block, return_inverse=True)
    n_blocks = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_blocks).astype(np.float64)
    sums = np.stack(
        [
            np.bincount(inverse, weights=(loss_a[s] - loss_b[s]), minlength=n_blocks)
            for s in range(n_seeds)
        ]
    )
    rng = np.random.default_rng(rng_seed)
    draws = np.empty(n_resamples)
    chunk = 2000
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        pick = rng.integers(0, n_blocks, size=(size, n_blocks))
        seed = rng.integers(0, n_seeds, size=size)
        draws[start : start + size] = sums[seed[:, None], pick].sum(axis=1) / counts[pick].sum(
            axis=1
        )
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "mean": float((loss_a - loss_b).mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_blocks": float(n_blocks),
        "n_rows": float(len(block)),
    }


def verdict(
    *,
    hi: float,
    lo: float,
    month_means: list[float],
    hi_without_d15: float,
    hi_without_d18: float,
    lo_without_d15: float,
    lo_without_d18: float,
) -> str:
    """Return the plan's verdict of one contrast on its decision rule.

    A contrast is a gain when `hi < 0`, every month's mean difference is negative, and `hi` stays
    below zero with D15 left out and with D18 left out. A loss is the mirror image. A gain is
    `supported` when `hi <= -0.05`, and `small gain` otherwise. A loss is `refuted`.

    Args:
        hi: Upper bound of the contrast's 99% interval.
        lo: Lower bound of the contrast's 99% interval.
        month_means: The contrast's mean in each scored month.
        hi_without_d15: Upper bound with D15 left out.
        hi_without_d18: Upper bound with D18 left out.
        lo_without_d15: Lower bound with D15 left out.
        lo_without_d18: Lower bound with D18 left out.

    Returns:
        One of `supported`, `small gain`, `refuted`, `inconclusive`.
    """
    gain = hi < 0 and all(m < 0 for m in month_means) and hi_without_d15 < 0 and hi_without_d18 < 0
    loss = lo > 0 and all(m > 0 for m in month_means) and lo_without_d15 > 0 and lo_without_d18 > 0
    if gain:
        return "supported" if hi <= GAIN_THRESHOLD_POINTS else "small gain"
    return "refuted" if loss else "inconclusive"


def uses_synthetic_rows(*, command: str) -> bool:
    """Return whether a command builds its rows with injected lag shifts: fit and bank do, pre-training and step 2 do not.

    Experiment 677 installs the synthetic builder for the fine-tuning fit only, so the stage 1 and
    stage 2 networks see the observed lags.
    """
    return command in {"fit", "bank"}


def default_batch(*, arm: str) -> int:
    """Return the plan's batch for an arm: 8,192 for the global arm and 512 for a per-substation arm."""
    return BATCH_GLOBAL if arm == "global" else BATCH_PER_SUBSTATION

# ruff: noqa: E501
"""Scores, contrasts, intervals, and the verdict mapping of experiment 738.

Streaming: `score_fold_seed` reads one fold and one seed at a time and writes small tables of sums
by arm, world, slice, metric, and Primary. Every contrast is a difference of two arms' sums over the
same rows, so the contrasts, groups, leave-one-out versions, and intervals are computed afterwards
from those tables, with nothing of the forecasts in memory. The verdict mapping is a pure function.
All losses are in points of capacity (1 point is 1% of capacity). CRPS is twice the mean pinball
loss over the 11 quantile levels. A contrast is the first arm minus the second, so a negative number
favours the first arm.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00738_common as common
import e00738_world as world
import numpy as np
import polars as pl
from scipy import stats

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
METRICS: Final[tuple[str, ...]] = (
    "crps",
    "crps_scaled",
    "nmae",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "cover90",
    "cover98",
    "follow_num",
    "follow_den",
    "departs",
)
"""The summed per-row values. `follow_num` and `follow_den` are the regression-through-zero parts
of the follow share against the reference arm, and `departs` counts rows where the median differs
from the reference's by more than half a point of capacity."""
IN_FORCE_SLICES: Final[tuple[str, ...]] = world.IN_FORCE
LEAD_SLICES: Final[tuple[str, ...]] = tuple(f"in_force_lead_{d}" for d in range(1, 10))
AGE_SLICES: Final[tuple[str, ...]] = tuple(f"in_force_{name}" for name, _, _ in common.AGE_BANDS)
SCORING_SLICES: Final[tuple[str, ...]] = (
    "all",
    "in_force",
    *world.STRATA,
    "in_force_clipped",
    *AGE_SLICES,
    *LEAD_SLICES,
    "event_free_ended_8_15",
)


# ---------------------------------------------------------------------------------------------
# Per-row losses
# ---------------------------------------------------------------------------------------------
def pinball_points(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the pinball loss in points of capacity, shape (n, 11).

    Args:
        y: Actual power over capacity, (n,).
        q: The 11 quantiles over capacity, (n, 11).
    """
    e = y[:, None] - q
    return np.maximum(LEVELS * e, (LEVELS - 1.0) * e) * 100.0


def crps_points(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return CRPS in points of capacity: twice the mean pinball loss over the 11 levels."""
    return 2.0 * pinball_points(y=y, q=q).mean(axis=1)


def scale_factor(*, y: np.ndarray, q: np.ndarray, target: float = 0.90) -> float:
    """Return the factor that makes the q05 to q95 band cover `target` of the rows, about the median.

    Args:
        y: Actual power over capacity.
        q: The 11 quantiles.
        target: The coverage of the q05 to q95 band.

    Returns:
        The factor, found by bisection.
    """
    med, lo, hi = q[:, 5:6], q[:, 1:2], q[:, 9:10]

    def cover(f: float) -> float:
        return float(
            ((y[:, None] >= med - f * (med - lo)) & (y[:, None] <= med + f * (hi - med))).mean()
        )

    a, b = 0.2, 20.0
    for _ in range(40):
        m = 0.5 * (a + b)
        if cover(m) < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def apply_scale(*, q: np.ndarray, factor: float) -> np.ndarray:
    """Return the quantiles with their spread about the median multiplied by `factor`."""
    med = q[:, 5:6]
    return (med + factor * (q - med)).astype(np.float32)


def follow_share(*, median_arm: np.ndarray, median_ref: np.ndarray, delta: np.ndarray) -> float:
    """Return the follow share: the slope of the arm's median change on the true shift, through zero.

    An oracle that adds the true shift to the reference median returns exactly 1, whatever the
    onset time of day, because the shift at each valid time is the regressor.

    Args:
        median_arm: The arm's median.
        median_ref: The reference arm's median (`M1`).
        delta: The true shift at each valid time, as a share of capacity.

    Returns:
        The share, or NaN when the shift is zero everywhere.
    """
    den = float((delta**2).sum())
    return float(((median_arm - median_ref) * delta).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------------------------
# Row labels
# ---------------------------------------------------------------------------------------------
def row_masks(
    *,
    prim: np.ndarray,
    origin_idx: np.ndarray,
    valid_idx: np.ndarray,
    lead_day: np.ndarray,
    onset: np.ndarray,
    end: np.ndarray,
    clipped: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the boolean mask of each scoring slice for rows of a scoring world.

    Args:
        prim: Each row's Primary index.
        origin_idx: Each row's contract origin index.
        valid_idx: Each row's valid index.
        lead_day: Each row's zero-based lead day.
        onset: Each Primary's shift onset (16,).
        end: Each Primary's shift end (16,).
        clipped: Whether each Primary's shift was clipped (16,).

    Returns:
        A mask per name in `SCORING_SLICES`.
    """
    s = world.strata(
        origin_idx=origin_idx,
        valid_idx=valid_idx,
        onset_idx=onset[prim],
        end_idx=end[prim],
        clipped=clipped[prim],
    )
    age = world.true_age_days(origin_idx=origin_idx, onset_idx=onset[prim])
    since = (origin_idx - end[prim]) / world.PER_DAY
    in_force = np.isin(s, world.IN_FORCE)
    masks: dict[str, np.ndarray] = {"all": np.ones(len(s), bool), "in_force": in_force}
    for name in (*world.STRATA, "in_force_clipped"):
        masks[name] = s == name
    for name, lo, hi in common.AGE_BANDS:
        masks[f"in_force_{name}"] = in_force & (age >= lo) & (age < hi)
    for d in range(1, 10):
        masks[f"in_force_lead_{d}"] = in_force & (lead_day == d)
    masks["event_free_ended_8_15"] = (
        (s == "event_free") & (since > 7.0) & (since <= 15.0) & ~clipped[prim]
    )
    return masks


def accumulate(
    *,
    sums: dict[str, dict[str, np.ndarray]],
    masks: Mapping[str, np.ndarray],
    prim: np.ndarray,
    values: Mapping[str, np.ndarray],
) -> None:
    """Add per-row values into `sums[slice][metric]` (16 Primaries), and counts into `sums[slice]['n']`.

    Args:
        sums: The accumulator, created as needed.
        masks: The mask of each slice.
        prim: Each row's Primary.
        values: Each metric's per-row value.
    """
    for name, mask in masks.items():
        slot = sums.setdefault(name, {})
        slot["n"] = slot.get("n", np.zeros(16)) + np.bincount(prim[mask], minlength=16)
        for metric, v in values.items():
            slot[metric] = slot.get(metric, np.zeros(16)) + np.bincount(
                prim[mask], weights=v[mask], minlength=16
            )


# ---------------------------------------------------------------------------------------------
# Intervals and effects
# ---------------------------------------------------------------------------------------------
def t_interval(*, monthly: np.ndarray, level: float) -> tuple[float, float]:
    """Return the t interval of the mean of monthly differences, at `level` percent (n - 1 degrees of freedom).

    Args:
        monthly: One difference per month.
        level: The coverage in percent (98.33 for the Bonferroni intervals, 95 otherwise).

    Returns:
        The lower and upper limits.
    """
    n = len(monthly)
    half = stats.t.ppf(1.0 - (1.0 - level / 100.0) / 2.0, n - 1) * monthly.std(ddof=1) / np.sqrt(n)
    return float(monthly.mean() - half), float(monthly.mean() + half)


def net_effect(*, c: float, v: float, q: float) -> float:
    """Return q * C + (1 - q) * V, the net all-rows effect, with q the share of real rows in a shift."""
    return q * c + (1.0 - q) * v


def bootstrap_month_means(
    *, per_month: np.ndarray, level: float, n_resamples: int = common.N_RESAMPLES
) -> tuple[float, float]:
    """Bound the mean of the monthly values, resampling whole months.

    Each resample draws the months with replacement and takes the mean of the drawn months' values.
    The monthly values are already averaged over the seeds, so every resample keeps all the seeds
    and the interval holds month-to-month variation only, as the t interval does.

    Args:
        per_month: The value of each month (the seed mean of that month's mean), (n_months,).
        level: The coverage in percent.
        n_resamples: The number of resamples.

    Returns:
        The lower and upper percentiles.
    """
    from studies.bootstrap import BOOTSTRAP_SEED

    gen = np.random.default_rng(BOOTSTRAP_SEED)
    drawn = gen.integers(0, per_month.shape[0], size=(n_resamples, per_month.shape[0]))
    out = per_month[drawn].mean(axis=1)
    tail = (100.0 - level) / 2.0
    return float(np.percentile(out, tail)), float(np.percentile(out, 100.0 - tail))


def month_block_bootstrap_net(
    *,
    c_sums: np.ndarray,
    c_counts: np.ndarray,
    v_sums: np.ndarray,
    v_counts: np.ndarray,
    q: float,
    level: float,
    n_resamples: int = common.N_RESAMPLES,
) -> tuple[float, float]:
    """Bound the net effect q C + (1 - q) V, resampling whole months and keeping every seed.

    C and V are the in-force and the event-free difference. Each month's value of each is the mean
    over the seeds of that month's sum over its row count, and each resample draws the months once
    for both, so the pairing of C and V in a month is kept.

    Args:
        c_sums: Month sums of the in-force difference, (n_seeds, n_months).
        c_counts: Rows of the in-force difference in each month, (n_months,).
        v_sums: Month sums of the event-free difference, (n_seeds, n_months).
        v_counts: Rows of the event-free difference in each month, (n_months,).
        q: The in-force share of real rows.
        level: The coverage in percent.
        n_resamples: The number of resamples.

    Returns:
        The lower and upper percentiles.
    """
    per_month = q * (c_sums / c_counts[None, :]).mean(axis=0) + (1.0 - q) * (
        v_sums / v_counts[None, :]
    ).mean(axis=0)
    return bootstrap_month_means(per_month=per_month, level=level, n_resamples=n_resamples)


@dataclass(frozen=True)
class Reading:
    """One contrast, with its intervals and seed signs.

    Attributes:
        point: The mean of the monthly differences.
        t_lower: Lower limit of the t interval.
        t_upper: Upper limit of the t interval.
        boot_lower: Lower limit of the month-block bootstrap interval.
        boot_upper: Upper limit of the month-block bootstrap interval.
        seed_points: The point estimate of each seed (the mean of that seed's monthly values, as
            `point` is the mean of the seed-averaged monthly values).
    """

    point: float
    t_lower: float
    t_upper: float
    boot_lower: float
    boot_upper: float
    seed_points: tuple[float, ...]

    @property
    def gain(self) -> bool:
        """Both upper limits are below zero."""
        return self.t_upper < 0.0 and self.boot_upper < 0.0

    @property
    def seeds_agree(self) -> bool:
        """Each seed's point estimate has the sign of the pooled point estimate."""
        sign = 1.0 if self.point > 0 else -1.0
        return all(p * sign > 0 for p in self.seed_points)


# ---------------------------------------------------------------------------------------------
# The verdict mapping
# ---------------------------------------------------------------------------------------------
VERDICTS: Final[tuple[str, ...]] = (
    "invalid",
    "refuted",
    "not supported",
    "inconclusive",
    "supported",
    "partly supported",
    "small gain",
)


@dataclass(frozen=True)
class VerdictState:
    """Every input of the verdict table, as booleans.

    Attributes:
        invalid: A leakage test, a generator oracle, the reproduction of stored `M1`, or the known-answer gate failed.
        n1_lower_above_zero: The lower limit of the 98.33% t interval of N1 is above 0.
        v1_fails: V1's point estimate is above +0.02.
        n1_inside_band: The 98.33% t interval of N1 lies inside -0.05 to +0.05.
        n1_gain: N1 is a gain (both upper limits below zero).
        n1_at_threshold: N1's point estimate is at or below -0.05.
        n1_seeds: Each seed has the sign of N1's point estimate.
        n3_gain: N3 is a gain.
        n3_at_threshold: N3's point estimate is at or below -0.03.
        n3_seeds: Each seed has the sign of N3's point estimate.
        c2_gain: C2 is a gain.
        c2_seeds: Each seed has the sign of C2's point estimate.
        held_out_agrees: The point estimates of C2 and C3 on the held-out world have the sign of the training family's.
    """

    invalid: bool
    n1_lower_above_zero: bool
    v1_fails: bool
    n1_inside_band: bool
    n1_gain: bool
    n1_at_threshold: bool
    n1_seeds: bool
    n3_gain: bool
    n3_at_threshold: bool
    n3_seeds: bool
    c2_gain: bool
    c2_seeds: bool
    held_out_agrees: bool


def _row_5(s: VerdictState) -> bool:
    return (
        s.n1_gain
        and s.n1_at_threshold
        and s.n3_gain
        and s.n3_at_threshold
        and s.c2_gain
        and not s.v1_fails
        and s.n1_seeds
        and s.n3_seeds
        and s.c2_seeds
        and s.held_out_agrees
    )


def _row_6(s: VerdictState) -> bool:
    return (
        s.n1_gain
        and s.n1_at_threshold
        and not s.v1_fails
        and s.n1_seeds
        and not (s.n3_gain and s.n3_at_threshold and s.c2_gain and s.held_out_agrees)
    )


def _row_7(s: VerdictState) -> bool:
    return s.n1_gain and not s.n1_at_threshold and not s.v1_fails and s.n1_seeds


ROWS: Final[tuple[tuple[int, str, object], ...]] = (
    (1, "invalid", lambda s: s.invalid),
    (2, "refuted", lambda s: s.n1_lower_above_zero),
    (3, "not supported", lambda s: s.v1_fails or s.n1_inside_band),
    (
        4,
        "inconclusive",
        lambda s: (
            (s.n1_gain and not s.n1_seeds)
            or (s.n3_gain and not s.n3_seeds)
            or (s.c2_gain and not s.c2_seeds)
        ),
    ),
    (5, "supported", _row_5),
    (6, "partly supported", _row_6),
    (7, "small gain", _row_7),
    (8, "inconclusive", lambda s: True),
)
"""The rows of the verdict table, in order: row number, verdict, and the condition."""


def matching_rows(*, state: VerdictState) -> list[int]:
    """Return the numbers of every row of the table whose condition holds (row 8 always does)."""
    return [n for n, _, cond in ROWS if cond(state)]  # ty: ignore[call-non-callable]


def verdict(*, state: VerdictState) -> tuple[int, str]:
    """Return the first row of the verdict table that holds, and its verdict.

    Args:
        state: The inputs.

    Returns:
        The row number and the verdict. Every state maps to exactly one row, because the first
        matching row decides and row 8 holds for every state.
    """
    first = matching_rows(state=state)[0]
    name = next(v for n, v, _ in ROWS if n == first)
    return first, name


def all_states() -> Iterable[VerdictState]:
    """Yield every combination of the 13 boolean inputs (8,192 states)."""
    names = list(VerdictState.__dataclass_fields__)
    for combo in itertools.product([False, True], repeat=len(names)):
        yield VerdictState(**dict(zip(names, combo, strict=True)))


def label_of(*, state: VerdictState) -> str:
    """Return the verdict with its label, `duration-dependent` when row 6 holds because held-out disagrees."""
    row, name = verdict(state=state)
    if row == 6 and not state.held_out_agrees:
        return f"{name} (duration-dependent)"
    return name


# ---------------------------------------------------------------------------------------------
# From the tables of sums to contrasts
# ---------------------------------------------------------------------------------------------
def sums_path(*, month: str, seed: int, out: Path | None = None) -> Path:
    """Return the file of the sums of one fold and one seed."""
    return (out if out is not None else common.OUT) / "scores" / f"sums_{month}_s{seed}.json"


def write_sums(*, sums: dict, month: str, seed: int, out: Path | None = None) -> None:
    """Write the sums of one fold and one seed, atomically (arrays as lists)."""

    def plain(o: object) -> object:
        if isinstance(o, dict):
            return {k: plain(v) for k, v in o.items()}
        return o.tolist() if isinstance(o, np.ndarray) else o

    common.atomic_json(obj=plain(sums), path=sums_path(month=month, seed=seed, out=out))


def read_sums(*, month: str, seed: int, out: Path | None = None) -> dict:
    """Read the sums of one fold and one seed, as arrays."""
    raw = json.loads(sums_path(month=month, seed=seed, out=out).read_text())

    def arr(o: object) -> object:
        if isinstance(o, dict):
            return {k: arr(v) for k, v in o.items()}
        return np.asarray(o, dtype=float) if isinstance(o, list) else o

    return arr(raw)  # ty: ignore[invalid-return-type]


def primary_indices(*, group: str, labels: list[str], drop: tuple[str, ...] = ()) -> np.ndarray:
    """Return the Primary indices of a group.

    Args:
        group: A key of `common.GROUP_OF_PRIMARY`.
        labels: The 16 Primaries' labels in index order.
        drop: Labels to leave out (leave-one-out).

    Returns:
        The indices.
    """
    names = [n for n in common.GROUP_OF_PRIMARY[group] if n not in drop]
    return np.array([labels.index(n) for n in names], dtype=int)


def month_arrays(
    *,
    tables: Mapping[tuple[str, int], dict],
    months: tuple[str, ...],
    seeds: tuple[int, ...],
    arm_a: str,
    arm_b: str | None,
    wld: str,
    slc: str,
    metric: str,
    prims: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the month sums of (arm_a - arm_b) over the Primaries, and the month row counts.

    Args:
        tables: The sums of each (month, seed).
        months: The fold months.
        seeds: The seeds.
        arm_a: The first arm.
        arm_b: The second arm (None for the first arm's own sums).
        wld: The world.
        slc: The slice.
        metric: The metric.
        prims: The Primary indices.

    Returns:
        (n_seeds, n_months) month sums of the difference, and (n_months,) row counts.
    """
    sums = np.zeros((len(seeds), len(months)))
    counts = np.zeros(len(months))
    for j, m in enumerate(months):
        for i, s in enumerate(seeds):
            t = tables[(m, s)]
            a = t[arm_a][wld][slc][metric][prims].sum()
            b = 0.0 if arm_b is None else t[arm_b][wld][slc][metric][prims].sum()
            sums[i, j] = a - b
        counts[j] = tables[(m, seeds[0])][arm_a][wld][slc]["n"][prims].sum()
    return sums, counts


def reading_from_month_arrays(
    *, sums: np.ndarray, counts: np.ndarray, level: float, n_resamples: int = common.N_RESAMPLES
) -> Reading:
    """Return the reading of a contrast from its month sums, with the t and the bootstrap intervals.

    Args:
        sums: Month sums of the difference, (n_seeds, n_months).
        counts: Rows per month.
        level: The coverage in percent.
        n_resamples: Bootstrap resamples.

    Returns:
        The reading. The monthly differences of the t interval are the seed means of each month's mean.
    """
    per_month = (sums / counts[None, :]).mean(axis=0)
    t_lo, t_hi = t_interval(monthly=per_month, level=level)
    b_lo, b_hi = bootstrap_month_means(per_month=per_month, level=level, n_resamples=n_resamples)
    seed_points = tuple(float(v) for v in (sums / counts[None, :]).mean(axis=1))
    return Reading(
        point=float(per_month.mean()),
        t_lower=t_lo,
        t_upper=t_hi,
        boot_lower=b_lo,
        boot_upper=b_hi,
        seed_points=seed_points,
    )


def net_reading(
    *,
    c: tuple[np.ndarray, np.ndarray],
    v: tuple[np.ndarray, np.ndarray],
    q: float,
    level: float,
    n_resamples: int = common.N_RESAMPLES,
) -> Reading:
    """Return the reading of the net effect q C + (1 - q) V from the in-force and event-free month arrays.

    Args:
        c: Month sums and counts of the in-force contrast.
        v: Month sums and counts of the event-free contrast.
        q: The in-force share.
        level: The coverage in percent.
        n_resamples: Bootstrap resamples.

    Returns:
        The reading.
    """
    c_sums, c_counts = c
    v_sums, v_counts = v
    monthly = q * (c_sums / c_counts[None, :]).mean(axis=0) + (1.0 - q) * (
        v_sums / v_counts[None, :]
    ).mean(axis=0)
    t_lo, t_hi = t_interval(monthly=monthly, level=level)
    b_lo, b_hi = month_block_bootstrap_net(
        c_sums=c_sums,
        c_counts=c_counts,
        v_sums=v_sums,
        v_counts=v_counts,
        q=q,
        level=level,
        n_resamples=n_resamples,
    )
    seeds = tuple(
        float(q * (c_sums[i] / c_counts).mean() + (1.0 - q) * (v_sums[i] / v_counts).mean())
        for i in range(c_sums.shape[0])
    )
    return Reading(float(monthly.mean()), t_lo, t_hi, b_lo, b_hi, seeds)


def choose_s_star(
    *, in_force_scaled: Mapping[str, float], margin: float = common.TIE_MARGIN
) -> str:
    """Return S*: the arm of LD, LM, LR with the lowest in-force scaled CRPS, preferring the simpler arm within `margin`.

    Args:
        in_force_scaled: The screen's in-force scaled CRPS of LD, LM, and LR.
        margin: An arm within this of the best loses to a simpler arm (order LD, LM, LR).

    Returns:
        The arm's name.
    """
    best = min(in_force_scaled.values())
    for arm in common.SURVIVAL_ARMS:
        if arm in in_force_scaled and in_force_scaled[arm] <= best + margin:
            return arm
    msg = "no survival arm scored"
    raise ValueError(msg)


# ---------------------------------------------------------------------------------------------
# Streaming one fold and one seed (needs the stored forecasts; not run by the unit tests)
# ---------------------------------------------------------------------------------------------
def world_of(*, name: str) -> str:
    """Return the world a panel name belongs to: `real`, `resampled`, `held_out`, or `train_family`."""
    if name == "real":
        return "real"
    if name.startswith("resampled"):
        return "resampled"
    return "held_out" if int(name.removeprefix("pass")) >= common.HELD_FIRST else "train_family"


def row_indices(*, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the contract origin index, the valid index, and the zero-based lead day of each row.

    Args:
        frame: A forecast frame of `e00738_panel.forecast_pass` (native-origin keys).

    Returns:
        Three integer arrays.
    """
    native = (
        frame["power_fcst_init_time"].dt.epoch("us").to_numpy() - common.GRID_START_US
    ) // common.STEP_US
    valid = (frame["valid_time"].dt.epoch("us").to_numpy() - common.GRID_START_US) // common.STEP_US
    return (
        (native + common.NATIVE_OFFSET_STEPS).astype(np.int64),
        valid.astype(np.int64),
        frame["lead_day"].to_numpy(),
    )


def primary_of(*, time_series_id: np.ndarray | list, ids: np.ndarray) -> np.ndarray:
    """Return the Primary index (the site table's `primary`) of each row's `time_series_id`.

    The shift lists and every per-Primary array are indexed by the site table's `primary` column,
    which is not in the order of the `time_series_id` values, so a caller must go through this
    mapping rather than sort the ids.

    Args:
        time_series_id: The id of each row.
        ids: The `time_series_id` of each Primary, in `primary` order.

    Returns:
        One index per row.
    """
    pos = {int(t): i for i, t in enumerate(ids)}
    return np.array([pos[int(t)] for t in time_series_id], dtype=np.int64)


def score_fold_seed(
    *,
    month: str,
    seed: int,
    labels: list[str],
    ids: np.ndarray,
    shifts_of: Mapping[str, list],
    out: Path | None = None,
) -> dict:
    """Score every arm and panel of one fold and one seed, and write the tables of sums.

    The scale factor of each arm is fitted on the training-family passes of the fold and seed, pooled
    over strata and Primaries, and then applied to every panel of the arm. `M1` is read first, because
    the other arms' follow share and departure count are against `M1`'s median.

    Args:
        month: The fold month.
        seed: The seed.
        labels: The 16 Primaries' labels in index order.
        ids: The `time_series_id` of each Primary.
        shifts_of: For each scoring panel name, the 16 shifts (index = Primary).
        out: The output folder (`OUT` by default).

    Returns:
        The sums, `[arm][world][slice][metric]` as arrays over the 16 Primaries.
    """
    import polars as pl
    from e00738_predict import panel_names, panel_path

    sums: dict = {}
    ref_median: dict[str, np.ndarray] = {}
    real_actual: np.ndarray | None = None
    for arm in common.arms_of(month=month, seed=seed, out=out):
        train_y, train_q = [], []
        for name in panel_names():
            if world_of(name=name) == "train_family":
                f = pl.read_parquet(panel_path(arm=arm, month=month, seed=seed, name=name, out=out))
                train_y.append(f["actual"].to_numpy())
                train_q.append(f.select(*_QCOLS).to_numpy())
        y_all, q_all = np.concatenate(train_y), np.concatenate(train_q)
        keep = np.isfinite(y_all)
        factor = scale_factor(y=y_all[keep], q=q_all[keep])
        del train_y, train_q, y_all, q_all
        arm_sums: dict = {}
        for name in panel_names():
            f = pl.read_parquet(panel_path(arm=arm, month=month, seed=seed, name=name, out=out))
            wld = world_of(name=name)
            y = f["actual"].to_numpy().astype(np.float64)
            q = f.select(*_QCOLS).to_numpy().astype(np.float64)
            ok = np.isfinite(y)
            prim = primary_of(time_series_id=f["time_series_id"].to_list(), ids=ids)
            origin, valid, lead = row_indices(frame=f)
            if wld == "real":
                masks = {"all": np.ones(len(y), bool)}
                if arm == common.ARM_M1:
                    real_actual = y
            else:
                sh = shifts_of[name]
                masks = row_masks(
                    prim=prim,
                    origin_idx=origin,
                    valid_idx=valid,
                    lead_day=lead,
                    onset=np.array([s.onset_idx for s in sh]),
                    end=np.array([s.end_idx for s in sh]),
                    clipped=np.array([s.clipped for s in sh]),
                )
            masks = {k: m & ok for k, m in masks.items()}
            pin = pinball_points(y=np.where(ok, y, 0.0), q=q)
            qs = apply_scale(q=q, factor=factor).astype(np.float64)
            med = q[:, 5]
            if arm == common.ARM_M1:
                ref_median[name] = med
            ref = ref_median[name]
            if wld == "real":
                delta = np.zeros(len(y))
            else:
                base = real_actual if real_actual is not None else y
                assert len(base) == len(y), (
                    f"the real and injected panels differ in length: {len(base)} and {len(y)}"
                )
                delta = np.where(ok, y - base, 0.0)
            values = {
                "crps": 2.0 * pin.mean(axis=1),
                "crps_scaled": crps_points(y=np.where(ok, y, 0.0), q=qs),
                "nmae": np.abs(np.where(ok, y, 0.0) - med) * 100.0,
                "pin01": pin[:, 0],
                "pin05": pin[:, 1],
                "pin95": pin[:, 9],
                "pin99": pin[:, 10],
                "cover90": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(float),
                "cover98": ((y >= q[:, 0]) & (y <= q[:, 10])).astype(float),
                "follow_num": (med - ref) * delta,
                "follow_den": delta**2,
                "departs": (np.abs(med - ref) * 100.0 > common.DEPARTURE_POINTS).astype(float),
            }
            accumulate(sums=arm_sums.setdefault(wld, {}), masks=masks, prim=prim, values=values)
        sums[arm] = arm_sums
        sums[arm]["scale_factor"] = factor
    write_sums(sums=sums, month=month, seed=seed, out=out)
    return sums


_QCOLS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)

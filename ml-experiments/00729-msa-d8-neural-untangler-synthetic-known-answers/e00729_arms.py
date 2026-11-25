# ruff: noqa: E501
"""Experiment 729: the arms, the input channels of each, masking, lags, and window batches.

This module needs no torch. An arm is a row of `ARM_SPECS`: which input groups it reads, its window
length, its output head, and its auxiliary head. A segment is turned once into a channel stack of
shape `(channels, (D + 2) * 48)`, and a window is a slice of the stack, so building a batch is cheap.
No arm reads the month or the day of the year: the calendar inputs of arm D are the local
half-hour, the day of the week, and the bank-holiday flag.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import e00729_core as core
import e00729_sim as sim
import numpy as np

R_SCALE: Final = 50.0
"""Readings, in points, are divided by this before they enter a network."""
BSP_SCALE: Final = 20.0
"""BSP half-hour steps, in points, are divided by this."""
CLEAR_SCALE: Final = 1000.0
"""Clear-sky irradiance in W/m2 is divided by this."""
TARGET_NAMES: Final = ("x", "valid", "engine", "solar")
"""The rows of a window's target array."""
HeadType = Literal["lobe", "quantile"]
AuxType = Literal["none", "solar", "engine"]


@dataclass(frozen=True)
class ArmSpec:
    """The design of one arm.

    Attributes:
        name: The arm's name in `core.ARMS`.
        cams: Reads the CAMS fields at D8 (clear-sky index, clear-sky irradiance, cosine of the
            zenith, and a presence mask).
        bsp: Reads the synthetic BSP steps and their mask.
        calendar: Reads the local half-hour, the day of the week, the bank-holiday flag, and ERA5
            temperature.
        lags: Reads the folded reading 7 and 14 days earlier, each with a mask.
        steps: The window length in half-hours (96, or 48 for the 24-hour window).
        head: "lobe" (two-lobe mixture) or "quantile" (11 quantiles).
        aux: The auxiliary output head: "none", "solar", or "engine".
        mask_training: Half of the training windows have a random span of the reading hidden.
    """

    name: str
    cams: bool = False
    bsp: bool = False
    calendar: bool = False
    lags: bool = False
    steps: int = core.WINDOW_STEPS
    head: HeadType = "lobe"
    aux: AuxType = "none"
    mask_training: bool = False


ARM_SPECS: Final = {
    "A": ArmSpec("A"),
    "B": ArmSpec("B", cams=True),
    "C": ArmSpec("C", cams=True, bsp=True),
    "D": ArmSpec("D", cams=True, bsp=True, calendar=True),
    "H": ArmSpec("H", bsp=True),
    "Dq": ArmSpec("Dq", cams=True, bsp=True, calendar=True, head="quantile"),
    "D24": ArmSpec("D24", cams=True, bsp=True, calendar=True, steps=core.WINDOW_STEPS_24),
    "E": ArmSpec("E", cams=True, bsp=True, calendar=True, mask_training=True),
    "F": ArmSpec("F", cams=True, bsp=True, calendar=True, lags=True),
    "G1": ArmSpec("G1", cams=True, bsp=True, calendar=True, aux="solar"),
    "G2": ArmSpec("G2", cams=True, bsp=True, calendar=True, aux="engine"),
}
"""Every arm of the plan's ladder and variants."""


def stack_names(*, spec: ArmSpec) -> list[str]:
    """Return the names of the rows of a segment's channel stack for an arm (lags excluded)."""
    names = ["r", "r_mask"]
    if spec.cams:
        names += ["csi", "clear", "cosz", "cams_mask"]
    if spec.bsp:
        names += ["d_bsp", "bsp_mask"]
    if spec.calendar:
        names += ["local_sin", "local_cos", "dow_sin", "dow_cos", "holiday", "t2m"]
    return names


def n_channels(*, spec: ArmSpec) -> int:
    """Return the number of input channels of an arm, with the 4 lag channels of arm F."""
    return len(stack_names(spec=spec)) + (4 if spec.lags else 0)


@dataclass(frozen=True)
class InputStats:
    """Standardisation constants taken from the training days.

    Attributes:
        t_mean: Mean of ERA5 2 m temperature in degrees Celsius.
        t_sd: Standard deviation of the same.
    """

    t_mean: float
    t_sd: float


def input_stats(*, cal: sim.Calendar, train_days: np.ndarray) -> InputStats:
    """Return the temperature mean and standard deviation over the training days."""
    t = cal.t2m[train_days]
    t = t[np.isfinite(t)]
    return InputStats(float(t.mean()), float(max(t.std(), 1e-3)))


def diff_along_time(*, x: np.ndarray) -> np.ndarray:
    """Return the first difference of a 1-D series, with NaN in position 0."""
    out = np.full_like(x, np.nan)
    out[1:] = x[1:] - x[:-1]
    return out


def calendar_rows(
    *, seg: sim.Segment, holidays: np.ndarray, stats: InputStats, hh: np.ndarray
) -> list[np.ndarray]:
    """Return the six calendar rows (local half-hour, day of week, holiday, temperature).

    Args:
        seg: The segment.
        holidays: Bank-holiday dates.
        stats: The standardisation constants.
        hh: The half-hour index `0..47` repeated for every dense day, flat.

    Returns:
        Six flat arrays of length `(D + 2) * 48`.
    """
    n_days = len(seg.dates)
    offsets = np.array(
        [sim.utc_offset_hours(day=d.astype("datetime64[D]").astype(date)) for d in seg.dates]
    )
    local = (hh + 2.0 * np.repeat(offsets, 48) + 0.5) % 48.0
    angle = 2.0 * np.pi * local / 48.0
    weekday = (seg.dates.astype("datetime64[D]").astype(int) + 3) % 7
    dow_angle = np.repeat(2.0 * np.pi * weekday / 7.0, 48)
    holiday = np.repeat(np.isin(seg.dates, holidays).astype(float), 48)
    t = (seg.t2m.reshape(-1) - stats.t_mean) / stats.t_sd
    assert len(holiday) == n_days * 48
    return [
        np.sin(angle), np.cos(angle), np.sin(dow_angle), np.cos(dow_angle), holiday,
        np.nan_to_num(t, nan=0.0),
    ]  # fmt: skip


def segment_stack(
    *, seg: sim.Segment, spec: ArmSpec, holidays: np.ndarray, stats: InputStats
) -> np.ndarray:
    """Return the channel stack of a segment for an arm: `(rows, (D + 2) * 48)`, float32.

    Args:
        seg: The segment.
        spec: The arm.
        holidays: Bank-holiday dates.
        stats: Temperature standardisation.

    Returns:
        The stack, with rows in the order of `stack_names`.
    """
    flat = {k: getattr(seg, k).reshape(-1) for k in ("r", "csi", "clear", "cosz", "bsp")}
    valid = seg.valid.reshape(-1)
    r_ok = valid & np.isfinite(flat["r"])
    rows = [np.where(r_ok, flat["r"] / R_SCALE, 0.0), r_ok.astype(float)]
    if spec.cams:
        ok = np.isfinite(flat["csi"])
        rows += [
            np.where(ok, np.clip(flat["csi"], 0.0, 1.5), 0.0),
            np.nan_to_num(flat["clear"] / CLEAR_SCALE, nan=0.0),
            np.nan_to_num(flat["cosz"], nan=0.0),
            ok.astype(float),
        ]
    if spec.bsp:
        d = diff_along_time(x=flat["bsp"])
        ok = np.isfinite(d)
        rows += [np.where(ok, d / BSP_SCALE, 0.0), ok.astype(float)]
    if spec.calendar:
        hh = np.tile(np.arange(48, dtype=float), len(seg.dates))
        rows += calendar_rows(seg=seg, holidays=holidays, stats=stats, hh=hh)
    return np.stack(rows).astype(np.float32)


def target_stack(*, seg: sim.Segment) -> np.ndarray:
    """Return the `(4, (D + 2) * 48)` targets: truth, valid, engine, solar (zero where missing)."""
    return np.stack(
        [
            np.nan_to_num(seg.x.reshape(-1), nan=0.0),
            (seg.valid & np.isfinite(seg.x)).reshape(-1).astype(float),
            np.nan_to_num(seg.engine.reshape(-1), nan=0.0),
            np.nan_to_num(seg.solar.reshape(-1), nan=0.0),
        ]
    ).astype(np.float32)


def window_start(*, dense_day: int, spec: ArmSpec) -> int:
    """Return the first stack column of the window of the scored day at dense index `dense_day`."""
    if spec.steps == core.WINDOW_STEPS_24:
        return dense_day * 48
    return dense_day * 48 - core.MIDDLE_START


def target_start(*, dense_day: int) -> int:
    """Return the first stack column of the scored day (midnight UTC)."""
    return dense_day * 48


def lag_rows(*, stack: np.ndarray, start: int, steps: int, lag_days: int) -> np.ndarray:
    """Return the folded reading and its mask `lag_days` earlier, zeros where outside the month.

    Args:
        stack: The segment stack (rows 0 and 1 are the reading and its mask).
        start: The window's first column.
        steps: The window length.
        lag_days: The lag in days.

    Returns:
        A `(2, steps)` array; a lag that falls before the first padding day is all zeros.
    """
    a = start - lag_days * 48
    if a < 0:
        return np.zeros((2, steps), dtype=np.float32)
    return stack[:2, a : a + steps].copy()


def mask_span(
    *, window: np.ndarray, rng: np.random.Generator, steps: int
) -> np.ndarray:  # fmt: skip
    """Hide a random span of 1 to 24 hours of the reading (and set its mask to zero) in place.

    Args:
        window: One window `(C, steps)`; rows 0 and 1 hold the reading and its mask.
        rng: The generator.
        steps: The window length.

    Returns:
        The same array, modified.
    """
    hours = int(rng.integers(core.MASK_HOURS[0], core.MASK_HOURS[1] + 1))
    length = min(hours * 2, steps)
    a = int(rng.integers(0, steps - length + 1))
    window[0, a : a + length] = 0.0
    window[1, a : a + length] = 0.0
    return window


def build_window(
    *, stack: np.ndarray, targets: np.ndarray, dense_day: int, spec: ArmSpec,
    rng: np.random.Generator | None = None, mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return one window `(C, steps)` and its targets `(4, 48)`.

    Args:
        stack: The segment's channel stack.
        targets: The segment's target stack.
        dense_day: The scored day's dense index (at least 1).
        spec: The arm.
        rng: The generator for masking (needed when `mask` is true).
        mask: Hide a random span of the reading.

    Returns:
        The input window and the 48 scored half-hours' targets.
    """
    a = window_start(dense_day=dense_day, spec=spec)
    window = stack[:, a : a + spec.steps].copy()
    if spec.lags:
        extra = [
            lag_rows(stack=stack, start=a, steps=spec.steps, lag_days=lag) for lag in core.LAG_DAYS
        ]
        window = np.concatenate([window, *extra], axis=0)
    if mask:
        assert rng is not None
        mask_span(window=window, rng=rng, steps=spec.steps)
    t0 = target_start(dense_day=dense_day)
    return window, targets[:, t0 : t0 + 48].copy()


def middle_slice(*, spec: ArmSpec) -> slice:
    """Return the slice of a network's output steps that holds the scored day."""
    if spec.steps == core.WINDOW_STEPS_24:
        return slice(0, 48)
    return slice(core.MIDDLE_START, core.MIDDLE_START + 48)


@dataclass
class PreparedSegment:
    """A segment with its stacks for one arm.

    Attributes:
        stack: The channel stack.
        targets: The target stack.
        days: Dense indices of the scored candidate days (the segment's base days).
        has_bsp: Whether the segment carries real BSP steps (for the sampler).
        source_index: The segment's position in the list it came from.
    """

    stack: np.ndarray
    targets: np.ndarray
    days: np.ndarray
    has_bsp: bool
    source_index: int


def prepare_segments(
    *, segments: Sequence[sim.Segment], spec: ArmSpec, holidays: np.ndarray, stats: InputStats
) -> list[PreparedSegment]:
    """Return the prepared stacks of every segment (segments without a base day are dropped)."""
    out = []
    for index, seg in enumerate(segments):
        days = seg.base_days
        days = days[(days >= 1) & (days <= seg.n_dense)]
        if len(days) == 0:
            continue
        out.append(
            PreparedSegment(
                stack=segment_stack(seg=seg, spec=spec, holidays=holidays, stats=stats),
                targets=target_stack(seg=seg),
                days=days,
                has_bsp=seg.has_bsp,
                source_index=index,
            )
        )
    return out


def sample_picks(
    *, prepared: Sequence[PreparedSegment], n: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Draw `n` (segment, day) picks: half from segments with real BSP steps, half from the others.

    Args:
        prepared: The prepared segments.
        n: The number of windows.
        rng: The generator.

    Returns:
        A list of `(segment index, dense day)` pairs. When one group is empty, all picks come from
        the other.
    """
    bsp = [i for i, p in enumerate(prepared) if p.has_bsp]
    other = [i for i, p in enumerate(prepared) if not p.has_bsp]
    if not bsp or not other:
        pool = bsp or other
        groups = [(pool, n)]
    else:
        n_bsp = round(n * core.BSP_SAMPLER_SHARE)
        groups = [(bsp, n_bsp), (other, n - n_bsp)]
    picks: list[tuple[int, int]] = []
    for pool, count in groups:
        for i in rng.choice(pool, size=count):
            seg = prepared[int(i)]
            picks.append((int(i), int(rng.choice(seg.days))))
    return picks


def make_batch(
    *, prepared: Sequence[PreparedSegment], picks: Sequence[tuple[int, int]], spec: ArmSpec,
    rng: np.random.Generator, training: bool,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return the inputs `(B, C, steps)` and targets `(B, 4, 48)` of a batch of picks.

    Args:
        prepared: The prepared segments.
        picks: `(segment index, dense day)` pairs.
        spec: The arm.
        rng: The generator for masking.
        training: Whether masking applies (arm E only, to half of the windows).

    Returns:
        The two arrays, float32.
    """
    xs, ts = [], []
    for seg_index, day in picks:
        p = prepared[seg_index]
        do_mask = bool(spec.mask_training and training and rng.uniform() < core.MASK_PROBABILITY)
        w, t = build_window(
            stack=p.stack, targets=p.targets, dense_day=day, spec=spec, rng=rng, mask=do_mask
        )
        xs.append(w)
        ts.append(t)
    return np.stack(xs), np.stack(ts)


def all_picks(*, prepared: Sequence[PreparedSegment]) -> list[tuple[int, int]]:
    """Return every (segment, base day) pair in order, the rows that a test bank scores."""
    return [(i, int(d)) for i, p in enumerate(prepared) for d in p.days]

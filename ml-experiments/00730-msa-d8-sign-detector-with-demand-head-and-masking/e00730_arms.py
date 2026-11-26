"""Experiment 730: the arm specifications, the target stack, the encoder channels, and the batches.

This module needs no torch. An arm reads one of three input sets: the reading and its mask alone,
experiment 729's arm D inputs, or arm D's inputs with the 8 calendar and weather rows replaced by
the 28 inputs of experiment 545's weather and datetime encoders. A segment becomes a channel stack
once, and a window is a slice of it. The target stack has experiment 729's five rows (truth, valid,
engine, solar, reading) and a sixth row, the base demand, which no input reads.

A batch builder applies the masking style (hidden reading spans) or the causal cut (every observed
channel after the cut hidden, with the hourly-source channels rebuilt from the visible hours only),
and returns, beside the windows and the targets, which scored half-hours keep the sign loss and
which keep the demand loss.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import e00730_paths  # noqa: F401

# isort: split
import e00729_arms as arms729
import e00729_core as core729
import e00729_sim as sim729
import e00730_core as core
import e00730_masks as masks
import numpy as np

InputSet = Literal["a", "d", "enc"]
BASE_ROW: Final = 5
"""Row of the target stack that holds the base demand in points (never an input)."""
TARGET_ROWS: Final = ("x", "valid", "engine", "solar", "r", "base")
FUTURE_ROW: Final = "future_hidden"
HOURLY_STAMP_SHIFT: Final = {"csi": 0, "t2m": 0, "wind10": 0, "precip": 1}
"""How many hours after the start of its own hour a source value is stamped. Precipitation is an
accumulation stamped at the end of its hour, so under a cut inside hour `h` the accumulation of hour
`h` is hidden as well."""
COPIED_ROWS: Final = ("precip",)
"""Channels that copy each hourly value to both half-hours of its hour; the rest interpolate."""
N_PLAIN_ROWS: Final = 8
"""Rows of the encoder arms that precede the encoder inputs: reading, mask, CAMS (4), BSP (2)."""


@dataclass(frozen=True)
class ArmSpec:
    """The design of one arm of experiment 730.

    Attributes:
        name: The arm's name.
        inputs: "a" (reading and mask), "d" (experiment 729's arm D inputs), or "enc" (arm D's with
            the encoder inputs in place of the calendar and weather rows).
        demand: The arm has the demand output.
        style: The masking style: "0", "K", "W", or "R".
        causal: The arm is trained with the future hidden on half of its windows.
        lam: The weight of the demand loss.
    """

    name: str
    inputs: InputSet
    demand: bool = False
    style: str = "0"
    causal: bool = False
    lam: float = core.LAMBDA

    @property
    def n_outputs(self) -> int:
        """The number of output channels: a sign logit, then a demand mean and spread."""
        return 3 if self.demand else 1

    @property
    def spec729(self) -> arms729.ArmSpec:
        """Experiment 729's arm that supplies the base channel stack of this arm."""
        if self.inputs == "a":
            return arms729.ARM_SPECS["A"]
        if self.inputs == "d":
            return arms729.ARM_SPECS["D"]
        return arms729.ArmSpec(f"{self.name}_enc", cams=True, bsp=True, calendar=False, head="sign")


def arm_specs(*, style: str) -> dict[str, ArmSpec]:
    """Return every arm of the plan, with SD* using the masking style chosen on validation.

    Args:
        style: The chosen style of SD* (one of "0", "K", "W", "R").

    Returns:
        The arms by name.
    """
    return {
        "S-A": ArmSpec("S-A", "a"),
        "S": ArmSpec("S", "d"),
        "SD-0": ArmSpec("SD-0", "d", demand=True, style="0"),
        "SD-K": ArmSpec("SD-K", "d", demand=True, style="K"),
        "SD-W": ArmSpec("SD-W", "d", demand=True, style="W"),
        "SD-R": ArmSpec("SD-R", "d", demand=True, style="R"),
        "SD*": ArmSpec("SD*", "d", demand=True, style=style),
        "SDe-PT": ArmSpec("SDe-PT", "enc", demand=True, style=style),
        "SDe-FR": ArmSpec("SDe-FR", "enc", demand=True, style=style),
        "SD*-C": ArmSpec("SD*-C", "d", demand=True, style=style, causal=True),
        "SD*-lambda": ArmSpec("SD*-lambda", "d", demand=True, style=style, lam=core.LAMBDA_REPORT),
    }


def stack_names(*, spec: ArmSpec) -> list[str]:
    """Return the names of the rows of an arm's channel stack."""
    if spec.inputs == "enc":
        base = arms729.stack_names(spec=spec.spec729)
        names = [*base, *(f"enc_{i}" for i in range(core.ENCODER_INPUT_COLUMNS))]
    else:
        names = arms729.stack_names(spec=spec.spec729)
    if spec.causal:
        names.append(FUTURE_ROW)
    return names


def n_channels(*, spec: ArmSpec) -> int:
    """Return the number of input channels of an arm."""
    return len(stack_names(spec=spec))


# --------------------------------------------------------------------------- encoder inputs


@dataclass(frozen=True)
class EncoderTable:
    """D8's encoder inputs for every half-hour of its base calendar, z-scored.

    Attributes:
        dates: The base days, `datetime64[D]`, sorted.
        values: The 7 weather and 21 datetime columns of every half-hour, `(n, 48, 28)`.
    """

    dates: np.ndarray
    values: np.ndarray

    def found(self, *, seg_dates: np.ndarray) -> np.ndarray:
        """Return whether each of a segment's dense days has a row in the table."""
        pos = np.clip(np.searchsorted(self.dates, seg_dates), 0, len(self.dates) - 1)
        return self.dates[pos] == seg_dates

    def coverage(self, *, segments: Sequence[sim729.Segment]) -> float:
        """Return the share of the segments' base days that have an encoder row."""
        total = hits = 0
        for seg in segments:
            base = seg.cal_index >= 0
            total += int(base.sum())
            hits += int(self.found(seg_dates=seg.dates)[base].sum())
        return hits / total if total else 1.0

    def dense_rows(self, *, seg_dates: np.ndarray) -> np.ndarray:
        """Return the encoder rows of a segment's dense days `(28, (D + 2) * 48)`, else zero."""
        pos = np.clip(np.searchsorted(self.dates, seg_dates), 0, len(self.dates) - 1)
        found = self.found(seg_dates=seg_dates)
        out = np.zeros((len(seg_dates), 48, core.ENCODER_INPUT_COLUMNS), dtype=np.float32)
        out[found] = np.nan_to_num(self.values[pos[found]], nan=0.0)
        return out.reshape(-1, core.ENCODER_INPUT_COLUMNS).T


# --------------------------------------------------------------------------- segments and stacks


def hourly_rows(*, seg: sim729.Segment) -> np.ndarray:
    """Return the raw hourly source values behind a segment's weather channels, `(4, hours)`.

    The rows follow `masks.HOURLY_ROWS`. The interpolated channels are inverted exactly, and
    precipitation, which copies one value to both half-hours of its hour, is read from the first
    half-hour of each pair.
    """
    out = []
    for name in masks.HOURLY_ROWS:
        flat = getattr(seg, name).reshape(-1).astype(np.float64)
        if name in COPIED_ROWS:
            out.append(flat.reshape(-1, 2)[:, 0])
        else:
            out.append(masks.hourly_from_half_hour(values=flat))
    return np.stack(out)


def utc_offsets(*, seg: sim729.Segment) -> np.ndarray:
    """Return local time minus UTC in hours for each dense day of a segment."""
    return np.array(
        [sim729.utc_offset_hours(day=d.astype("datetime64[D]").astype(date)) for d in seg.dates]
    )


@dataclass
class PreparedSegment:
    """A segment with its stacks for one arm.

    Attributes:
        stack: The channel stack, rows in the order of `stack_names` (without the future row).
        targets: The `(6, (D + 2) * 48)` target stack, with the base demand in `BASE_ROW`.
        days: Dense indices of the scored candidate days.
        has_bsp: Whether the segment carries real BSP steps (for the sampler).
        source_index: The segment's position in the list it came from.
        hourly: The raw hourly source values, `(4, hours)`, for the causal cut.
        offsets: Local time minus UTC in hours of each dense day.
        stats: The standardisation constants of the temperature and wind-speed channels.
    """

    stack: np.ndarray
    targets: np.ndarray
    days: np.ndarray
    has_bsp: bool
    source_index: int
    hourly: np.ndarray
    offsets: np.ndarray
    stats: arms729.InputStats


def target_stack(*, seg: sim729.Segment) -> np.ndarray:
    """Return the `(6, (D + 2) * 48)` targets: experiment 729's five rows and the base demand."""
    base = np.nan_to_num(seg.base.reshape(-1), nan=0.0).astype(np.float32)
    return np.concatenate([arms729.target_stack(seg=seg), base[None]], axis=0)


def prepare_segments(
    *,
    segments: Sequence[sim729.Segment],
    spec: ArmSpec,
    holidays: np.ndarray,
    stats: arms729.InputStats,
    encoder: EncoderTable | None = None,
) -> list[PreparedSegment]:
    """Return the prepared stacks of every segment (segments without a base day are dropped).

    Args:
        segments: The segments.
        spec: The arm.
        holidays: Bank-holiday dates.
        stats: The standardisation constants of the temperature and wind speed.
        encoder: D8's encoder input table; required for the "enc" arms.

    Returns:
        The prepared segments.
    """
    if spec.inputs == "enc" and encoder is None:
        msg = "an encoder arm needs D8's encoder input table"
        raise ValueError(msg)
    out = []
    for index, seg in enumerate(segments):
        days = seg.base_days
        days = days[(days >= 1) & (days <= seg.n_dense)]
        if len(days) == 0:
            continue
        stack = arms729.segment_stack(seg=seg, spec=spec.spec729, holidays=holidays, stats=stats)
        if spec.inputs == "enc":
            assert encoder is not None
            stack = np.concatenate([stack, encoder.dense_rows(seg_dates=seg.dates)], axis=0)
        out.append(
            PreparedSegment(
                stack=stack.astype(np.float32),
                targets=target_stack(seg=seg),
                days=days,
                has_bsp=seg.has_bsp,
                source_index=index,
                hourly=hourly_rows(seg=seg),
                offsets=utc_offsets(seg=seg),
                stats=stats,
            )
        )
    return out


# --------------------------------------------------------------------------- the causal cut


def channel_from_raw(*, name: str, raw: np.ndarray, stats: arms729.InputStats) -> np.ndarray:
    """Return a stack row's values from raw half-hour source values (NaN where missing)."""
    ok = np.isfinite(raw)
    if name == "csi":
        return np.where(ok, np.clip(raw, 0.0, 1.5), 0.0)
    if name == "t2m":
        return np.where(ok, (raw - stats.t_mean) / stats.t_sd, 0.0)
    if name == "wind10":
        return np.where(ok, (raw - stats.w_mean) / stats.w_sd, 0.0)
    if name == "precip":
        return np.where(ok, np.log1p(np.clip(raw, 0.0, None)) / arms729.PRECIP_SCALE, 0.0)
    msg = f"{name} is not an hourly-source row"
    raise ValueError(msg)


def apply_cut(
    *, window: np.ndarray, names: Sequence[str], prepared: PreparedSegment, start: int, cut_pos: int
) -> np.ndarray:
    """Hide the future of a window after `cut_pos`, in place.

    The half-hour channels built from hourly sources (the CAMS clear-sky index, the temperature, the
    wind speed, and the precipitation) are rebuilt for the positions up to the cut from the hourly
    source values stamped at or before the start of the cut's hour (precipitation, stamped at the
    end of its hour, from the hours before the cut's hour), before any interpolation or copy. Every
    other observed channel after the cut is zero, and the future-hidden channel holds 1 there.

    Args:
        window: One window `(C, 96)` (modified).
        names: The row names of the window.
        prepared: The segment's prepared stacks (its raw hourly values and constants).
        start: The window's first stack column (even).
        cut_pos: The last visible position, `0..95`; 95 or more hides nothing.

    Returns:
        The same array, modified.
    """
    steps = window.shape[1]
    if cut_pos >= steps - 1:
        return window
    first_hour = start // 2 - 1
    cut_hour = (cut_pos // 2) + 1
    index = {n: i for i, n in enumerate(names)}
    for row, name in enumerate(masks.HOURLY_ROWS):
        if name not in index:
            continue
        hourly = np.full(steps // 2 + 1, np.nan)
        lo = max(first_hour, 0)
        hourly[lo - first_hour :] = prepared.hourly[row, lo : first_hour + steps // 2 + 1]
        last = cut_hour - HOURLY_STAMP_SHIFT[name]
        half = masks.causal_half_hours(
            hourly=hourly, last_visible_hour=last, copy=name in COPIED_ROWS
        )[2:]
        values = channel_from_raw(name=name, raw=half, stats=prepared.stats)
        window[index[name], : cut_pos + 1] = values[: cut_pos + 1]
        if name == "csi":
            window[index["cams_mask"], : cut_pos + 1] = np.isfinite(half[: cut_pos + 1])
    for name in masks.OBSERVED_ROWS:
        if name in index:
            window[index[name], cut_pos + 1 :] = 0.0
    if FUTURE_ROW in index:
        window[index[FUTURE_ROW], cut_pos + 1 :] = 1.0
    return window


# --------------------------------------------------------------------------- windows and batches


def build_one(
    *,
    prepared: PreparedSegment,
    dense_day: int,
    spec: ArmSpec,
    hidden: np.ndarray | None = None,
    cut_pos: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one window, its 6 target rows, and the loss keeps.

    Args:
        prepared: The segment's prepared stacks.
        dense_day: The scored day's dense index (at least 1).
        spec: The arm.
        hidden: Positions of the window whose reading is hidden (`(96,)`), or None.
        cut_pos: The last visible position of a causal cut, or None for the two-sided window.

    Returns:
        The window `(C, 96)`, the targets `(6, 48)`, and the keeps `(2, 48)`: row 0 is where the
        sign loss applies (valid and visible) and row 1 where the demand loss applies (valid).
    """
    names = stack_names(spec=spec)
    start = dense_day * 48 - core729.MIDDLE_START
    window = prepared.stack[:, start : start + core729.WINDOW_STEPS].copy()
    if spec.causal:
        window = np.concatenate([window, np.zeros((1, window.shape[1]), dtype=np.float32)], axis=0)
    target = prepared.targets[:, dense_day * 48 : dense_day * 48 + 48].copy()
    sign_keep = target[1].copy()
    demand_keep = target[1].copy() if spec.demand else np.zeros(48, dtype=np.float32)
    if hidden is not None:
        masks.apply_hidden(window=window, hidden=hidden)
        sign_keep[hidden[core729.MIDDLE_START : core729.MIDDLE_START + 48]] = 0.0
    if cut_pos is not None:
        apply_cut(window=window, names=names, prepared=prepared, start=start, cut_pos=cut_pos)
        last_scored = cut_pos - core729.MIDDLE_START
        sign_keep[max(last_scored + 1, 0) :] = 0.0
    return window, target, np.stack([sign_keep, demand_keep])


def make_batch(
    *,
    prepared: Sequence[PreparedSegment],
    picks: Sequence[tuple[int, int]],
    spec: ArmSpec,
    rng: np.random.Generator,
    training: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the inputs `(B, C, 96)`, targets `(B, 6, 48)`, and keeps `(B, 2, 48)` of a batch.

    In training, an arm trained with the future hidden cuts the future on half of its windows, with
    a cut drawn uniformly from the scored day's 48 half-hours. The other windows are masked in the
    arm's style with probability `MASK_PROBABILITY` (style "0" masks none), so no window carries
    both. Evaluation windows are never masked or cut here.
    """
    xs, ts, ks = [], [], []
    for seg_index, day in picks:
        p = prepared[seg_index]
        hidden = cut = None
        if training and spec.causal and rng.uniform() < core.MASK_PROBABILITY:
            cut = core729.MIDDLE_START + int(rng.integers(0, core729.WINDOW_STEPS_24))
        elif training and spec.style != "0" and rng.uniform() < core.MASK_PROBABILITY:
            hidden = masks.hidden_positions(style=spec.style, rng=rng, utc_offset=p.offsets[day])
        w, t, k = build_one(prepared=p, dense_day=day, spec=spec, hidden=hidden, cut_pos=cut)
        xs.append(w)
        ts.append(t)
        ks.append(k)
    return np.stack(xs), np.stack(ts), np.stack(ks)


def make_eval_batch(
    *,
    prepared: Sequence[PreparedSegment],
    picks: Sequence[tuple[int, int]],
    spec: ArmSpec,
    hidden: np.ndarray | None = None,
    cut_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the evaluation batch of the given picks with fixed masks or cuts.

    Args:
        prepared: The prepared segments.
        picks: `(segment index, dense day)` pairs.
        spec: The arm.
        hidden: The hidden positions of every pick, `(B, 96)`, or None.
        cut_pos: The last visible position of every pick, `(B,)`, or None.

    Returns:
        The inputs, targets, and keeps, as in `make_batch`.
    """
    xs, ts, ks = [], [], []
    for i, (seg_index, day) in enumerate(picks):
        w, t, k = build_one(
            prepared=prepared[seg_index],
            dense_day=day,
            spec=spec,
            hidden=None if hidden is None else hidden[i],
            cut_pos=None if cut_pos is None else int(cut_pos[i]),
        )
        xs.append(w)
        ts.append(t)
        ks.append(k)
    return np.stack(xs), np.stack(ts), np.stack(ks)


def middle() -> slice:
    """Return the slice of a network's output steps that holds the scored day."""
    return slice(core729.MIDDLE_START, core729.MIDDLE_START + 48)


def input_equals_target_rows(*, window: np.ndarray, targets: np.ndarray) -> bool:
    """Return whether any input channel equals a target row on the scored day's half-hours.

    The leakage check: no input channel may equal the truth row, the engine row, or the base-demand
    row, exactly or after scaling by the row's maximum, and none may equal the solar-output row
    exactly. The solar row is compared exactly only, because the CAMS clear-sky channels are
    legitimately proportional to the solar output on a day with a constant clear-sky index.
    Constant channels are skipped.
    """
    scored = window[:, core729.MIDDLE_START : core729.MIDDLE_START + 48]
    for channel in scored:
        if np.ptp(channel) == 0.0:
            continue
        for index in (0, 2, 3, BASE_ROW):
            row = targets[index]
            if np.ptp(row) == 0.0:
                continue
            scale = (np.abs(row).max() or 1.0) if index != 3 else 1.0
            if np.allclose(channel, row, atol=1e-6) or np.allclose(channel, row / scale, atol=1e-6):
                return True
    return False

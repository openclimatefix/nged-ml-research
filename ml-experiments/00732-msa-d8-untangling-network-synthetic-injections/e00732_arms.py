# ruff: noqa: E501
"""Experiment 732: the target stack, the windows of a training triple, and the batches.

This module needs no torch. A segment (with its load transfers applied) becomes a channel stack once,
through experiment 730's `prepare_segments`, and a window is a slice of it. The target stack has eight
rows: experiment 730's six (truth, valid, engine, solar, reading, base demand) and the step event and
the signed step size. A **triple** is the plain window `w`, the injected window `w+` (the same window
with an extra solar injection delta from a sibling segment), and the raised-sky window `wc`. The
three members share one hidden span of the reading. The inputs of `w+` differ from `w` only in the
reading channel and the BSP-step channel; the inputs of `wc` differ only in the clear-sky index.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import e00729_arms as arms729
import e00729_core as core729
import e00730_arms as arms730
import e00730_masks as masks
import e00732_core as core
import e00732_overlays as ov
import numpy as np

STACK_SPEC: Final = arms730.ArmSpec("S732", "d", demand=False, style=core.STYLE)
"""The channel stack of every arm: experiment 730's arm S inputs."""
NAMES: Final = tuple(arms730.stack_names(spec=STACK_SPEC))
ROW: Final = {name: i for i, name in enumerate(NAMES)}
N_CHANNELS: Final = len(NAMES)
TARGET_ROWS: Final = ("x", "valid", "engine", "solar", "r", "base", "event", "size")
T: Final = {name: i for i, name in enumerate(TARGET_ROWS)}
KEEP_ROWS: Final = ("sign", "solar", "step", "size", "injection", "night", "raisable")
K: Final = {name: i for i, name in enumerate(KEEP_ROWS)}
"""Rows of the keeps: where each loss term applies on the scored day."""
STEPS: Final = core729.WINDOW_STEPS
MID: Final = core729.MIDDLE_START


@dataclass
class Prepared:
    """A transfer-overlaid segment with its stack, targets, and the arrays the pairs need.

    Attributes:
        base: Experiment 730's prepared segment of the overlaid segment (stack and six target rows).
        q: The reactive floor, flat over the dense axis.
        bsp: The overlaid BSP meter, flat (NaN where the meter has no data).
        beta: The segment's BSP coupling.
        event: The step event, flat.
        size: The signed step size, flat.
    """

    base: arms730.PreparedSegment
    q: np.ndarray
    bsp: np.ndarray
    beta: float
    event: np.ndarray
    size: np.ndarray

    @property
    def days(self) -> np.ndarray:
        """Dense indices of the scored candidate days."""
        return self.base.days

    @property
    def has_bsp(self) -> bool:
        """Whether the segment carries real BSP steps (for the sampler)."""
        return self.base.has_bsp

    @property
    def source_index(self) -> int:
        """The segment's position in the bank's list."""
        return self.base.source_index

    @property
    def offsets(self) -> np.ndarray:
        """Local time minus UTC in hours of each dense day."""
        return self.base.offsets


def prepare(
    *, overlays: Sequence[ov.TransferOverlay], holidays: np.ndarray, stats: arms729.InputStats
) -> list[Prepared]:
    """Return the prepared segments of a bank's transfer overlays (segments without a base day are dropped).

    Args:
        overlays: The overlays, in the order of the bank's segments.
        holidays: Bank-holiday dates.
        stats: The standardisation constants of the temperature and wind speed.

    Returns:
        The prepared segments; `source_index` is the segment's position in the bank.
    """
    base = arms730.prepare_segments(segments=[o.seg for o in overlays], spec=STACK_SPEC, holidays=holidays, stats=stats)
    return [
        Prepared(
            base=b, q=overlays[b.source_index].q.reshape(-1), bsp=overlays[b.source_index].seg.bsp.reshape(-1),
            beta=overlays[b.source_index].beta, event=overlays[b.source_index].event.reshape(-1),
            size=overlays[b.source_index].size.reshape(-1),
        )
        for b in base
    ]  # fmt: skip


def prepare_plain(
    *, segments: Sequence, holidays: np.ndarray, stats: arms729.InputStats
) -> list[Prepared]:
    """Return the prepared segments of a bank or real set with no load transfer (all zeros)."""
    overlays = [ov.apply_transfers(seg=s, transfers=[]) if "beta" in s.params else _no_transfer(seg=s) for s in segments]
    return prepare(overlays=overlays, holidays=holidays, stats=stats)


def _no_transfer(*, seg: object) -> ov.TransferOverlay:
    """Return an overlay of a real pseudo-segment (no BSP coupling, no transfer)."""
    shape = seg.x.shape  # ty: ignore[unresolved-attribute]
    zeros = np.zeros(shape, dtype=np.float32)
    return ov.TransferOverlay(
        seg=seg, q=zeros, beta=0.0, transfers=[], t=zeros, t_bsp=zeros, event=zeros, size=zeros, n_failed=0, rate=0.0,  # ty: ignore[invalid-argument-type]
    )  # fmt: skip


def window_start(*, day: int) -> int:
    """Return the first stack column of the window of the scored day at dense index `day`."""
    return day * 48 - MID


def target_window(*, p: Prepared, day: int) -> np.ndarray:
    """Return the `(8, 48)` targets of the scored day."""
    a = day * 48
    rows = [p.base.targets[:, a : a + 48], p.event[None, a : a + 48], p.size[None, a : a + 48]]
    return np.concatenate(rows, axis=0).astype(np.float32).copy()


def injected_window(
    *, p: Prepared, day: int, plain: np.ndarray, delta_flat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the injected member `w+` of a window, its `(8, 48)` targets, and the scored delta.

    The reading channel is re-folded from the truth less delta and the segment's reactive floor
    (zero where the reading is hidden or missing), and the BSP-step channel is the half-hour change
    of the BSP meter less `beta * delta`. No other channel changes.

    Args:
        p: The prepared segment.
        day: The scored day's dense index.
        plain: The plain window `(C, 96)`, after the hidden span is applied.
        delta_flat: The extra injection over the whole dense axis, in points.

    Returns:
        The window, the targets of `w+` (truth less delta, solar plus delta, re-folded reading), and
        the scored delta `(48,)`.
    """
    s = window_start(day=day)
    out = plain.copy()
    x = p.base.targets[T["x"], s : s + STEPS]
    delta = delta_flat[s : s + STEPS].astype(np.float64)
    r_new = ov.refold(x=x - delta, q=p.q[s : s + STEPS])
    shown = plain[ROW["r_mask"]] > 0.5
    out[ROW["r"]] = np.where(shown, r_new / arms729.R_SCALE, 0.0)
    ext = p.bsp[s - 1 : s + STEPS] - p.beta * delta_flat[s - 1 : s + STEPS]
    d = np.diff(ext)
    ok = np.isfinite(d)
    out[ROW["d_bsp"]] = np.where(ok, d / arms729.BSP_SCALE, 0.0)
    out[ROW["bsp_mask"]] = ok.astype(np.float32)
    target = target_window(p=p, day=day)
    scored = delta_flat[day * 48 : day * 48 + 48]
    target[T["x"]] -= scored
    target[T["solar"]] += scored
    target[T["r"]] = np.nan_to_num(ov.refold(x=target[T["x"]], q=p.q[day * 48 : day * 48 + 48]), nan=0.0)
    return out, target, scored.astype(np.float32)


def raised_window(*, plain: np.ndarray, eps: float) -> np.ndarray:
    """Return the raised-sky member `wc`: the clear-sky index times `1 + eps` on sun-up half-hours."""
    out = plain.copy()
    out[ROW["csi"]] = ov.raise_sky(csi=plain[ROW["csi"]], cosz=plain[ROW["cosz"]], eps=eps)
    return out


def keeps_of(*, p: Prepared, day: int, window: np.ndarray, hidden: np.ndarray | None, target: np.ndarray) -> np.ndarray:
    """Return the `(7, 48)` keeps of the scored day, in the order of `KEEP_ROWS`.

    The sign, step, size, and injection terms drop hidden half-hours; the solar term keeps them.
    """
    valid = target[T["valid"]] > 0.5
    shown = np.ones(48, dtype=bool) if hidden is None else ~hidden[MID : MID + 48]
    scored = slice(MID, MID + 48)
    cosz = window[ROW["cosz"], scored]
    csi = window[ROW["csi"], scored]
    out = np.zeros((len(KEEP_ROWS), 48), dtype=np.float32)
    out[K["sign"]] = valid & shown
    out[K["solar"]] = valid
    out[K["step"]] = valid & shown
    out[K["size"]] = valid & shown & (target[T["event"]] > 0.5)
    out[K["injection"]] = valid & shown
    out[K["night"]] = valid & (cosz <= core.NIGHT_COSZ)
    out[K["raisable"]] = valid & ov.raisable(csi=csi, cosz=cosz)
    return out


@dataclass
class Batch:
    """A batch of triples.

    Attributes:
        x: Plain windows `(B, C, 96)`.
        xi: Injected windows `(B, C, 96)` (a copy of `x` where there is no pair), or None.
        xc: Raised-sky windows `(B, C, 96)`, or None.
        t: Plain targets `(B, 8, 48)`.
        ti: Injected targets `(B, 8, 48)`, or None.
        keeps: `(B, 7, 48)`.
        delta: The scored delta `(B, 48)` (zero where there is no pair).
        pair: 1.0 where the window has a pair (a sibling with output that day), else 0.0.
        amount: The injected amount of each window (0 without a pair).
    """

    x: np.ndarray
    xi: np.ndarray | None
    xc: np.ndarray | None
    t: np.ndarray
    ti: np.ndarray | None
    keeps: np.ndarray
    delta: np.ndarray
    pair: np.ndarray
    amount: np.ndarray


def build_triple(
    *, p: Prepared, day: int, spec: core.ArmSpec732, hidden: np.ndarray | None,
    delta_flat: np.ndarray | None, eps: float | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:  # fmt: skip
    """Return one window and, when the arm needs them, its injected and raised members.

    Args:
        p: The prepared segment.
        day: The scored day's dense index (at least 1).
        spec: The arm.
        hidden: The hidden positions `(96,)` shared by the three members, or None.
        delta_flat: The extra injection over the dense axis, or None when the window has no pair.
        eps: The raised-sky factor, or None.

    Returns:
        `w`, `w+` (a copy of `w` where `delta_flat` is None), `wc`, the targets of `w`, the targets
        of `w+`, the keeps, and the scored delta.
    """
    s = window_start(day=day)
    plain = p.base.stack[:, s : s + STEPS].copy()
    if hidden is not None:
        masks.apply_hidden(window=plain, hidden=hidden)
    target = target_window(p=p, day=day)
    keeps = keeps_of(p=p, day=day, window=plain, hidden=hidden, target=target)
    inj = inj_target = None
    scored = np.zeros(48, dtype=np.float32)
    if spec.inject:
        if delta_flat is None:
            inj, inj_target = plain.copy(), target.copy()
        else:
            inj, inj_target, scored = injected_window(p=p, day=day, plain=plain, delta_flat=delta_flat)
    raised = None
    if spec.raised:
        raised = raised_window(plain=plain, eps=core.TEST_EPS if eps is None else eps)
    return plain, inj, raised, target, inj_target, keeps, scored


def collate(*, parts: Sequence[tuple], pair: Sequence[bool], amount: Sequence[float], spec: core.ArmSpec732) -> Batch:
    """Stack the triples of a batch."""
    xs, xis, xcs, ts, tis, ks, ds = zip(*parts, strict=True)
    return Batch(
        x=np.stack(xs), xi=np.stack(xis) if spec.inject else None, xc=np.stack(xcs) if spec.raised else None,
        t=np.stack(ts), ti=np.stack(tis) if spec.inject else None, keeps=np.stack(ks), delta=np.stack(ds),
        pair=np.asarray(pair, dtype=np.float32), amount=np.asarray(amount, dtype=np.float32),
    )  # fmt: skip


def make_batch(
    *, prepared: Sequence[Prepared], picks: Sequence[tuple[int, int]], spec: core.ArmSpec732,
    rng: np.random.Generator, index: ov.PairIndex | None,
) -> Batch:  # fmt: skip
    """Return a training batch: hidden spans on half the windows, a sibling pair, and a raised sky.

    Args:
        prepared: The prepared segments of the training bank.
        picks: `(segment index in prepared, dense day)` pairs.
        spec: The arm.
        rng: The generator (one draw stream per fit, so a batch is the same whatever the schedule).
        index: The pair index of the training bank, or None for an arm without the injected member.

    Returns:
        The batch. A window whose month has no other solar segment with output that day takes no pair.
    """
    parts, pair, amount = [], [], []
    for seg_i, day in picks:
        p = prepared[seg_i]
        hidden = None
        if rng.uniform() < core729.MASK_PROBABILITY:
            hidden = masks.hidden_positions(style=core.STYLE, rng=rng, utc_offset=p.offsets[day])
        delta_flat = None
        a = 0.0
        if spec.inject and index is not None:
            sib = ov.pick_sibling(index=index, segment=p.source_index, day=day, rng=rng)
            a = ov.draw_amount(rng=rng)
            if sib is not None:
                delta_flat = ov.delta_flat(index=index, sibling=sib, amount=a)
        eps = float(rng.uniform(*core.EPS_RANGE)) if spec.raised else None
        parts.append(build_triple(p=p, day=day, spec=spec, hidden=hidden, delta_flat=delta_flat, eps=eps))
        pair.append(delta_flat is not None)
        amount.append(a if delta_flat is not None else 0.0)
    return collate(parts=parts, pair=pair, amount=amount, spec=spec)


def make_eval_batch(
    *, prepared: Sequence[Prepared], picks: Sequence[tuple[int, int]], spec: core.ArmSpec732,
    index: ov.PairIndex | None = None, amount: float | None = None, eps: float | None = None,
    hidden: np.ndarray | None = None,
) -> Batch:  # fmt: skip
    """Return an evaluation batch: never masked, with the fixed test pair of amount `amount` when given.

    Args:
        prepared: The prepared segments.
        picks: `(segment index in prepared, dense day)` pairs.
        spec: The arm (its `inject` and `raised` flags decide which members are built).
        index: The pair index (needed with `amount`); the sibling is the window's fixed sibling.
        amount: The injected amount in points, or None for no injected member's delta.
        eps: The raised-sky factor of the raised member (0.2 by default).
        hidden: Fixed hidden positions `(B, 96)`, or None.

    Returns:
        The batch.
    """
    parts, pair, amt = [], [], []
    for i, (seg_i, day) in enumerate(picks):
        p = prepared[seg_i]
        delta_flat = None
        if amount is not None and index is not None:
            sib = ov.fixed_sibling(index=index, segment=p.source_index, day=day)
            if sib is not None:
                delta_flat = ov.delta_flat(index=index, sibling=sib, amount=amount)
        parts.append(
            build_triple(p=p, day=day, spec=spec, hidden=None if hidden is None else hidden[i], delta_flat=delta_flat, eps=eps)
        )
        pair.append(delta_flat is not None)
        amt.append(amount if delta_flat is not None and amount is not None else 0.0)
    return collate(parts=parts, pair=pair, amount=amt, spec=spec)


def eval_spec(*, inject: bool = False, raised: bool = False) -> core.ArmSpec732:
    """Return an arm that builds the members an evaluation needs (the weights play no role)."""
    return core.ArmSpec732("eval", "unet", True, core.BASE_WEIGHTS, inject=inject, raised=raised)


def input_equals_target(*, window: np.ndarray, targets: np.ndarray, delta: np.ndarray | None = None) -> bool:
    """Return whether any input channel equals a target row of the scored day (the leakage check).

    The truth, engine, base-demand, and solar rows are compared as experiment 730 does; the step
    event and the step size are compared exactly and after scaling by their maximum, and delta is
    compared exactly only, because the clear-sky channels are legitimately proportional to a solar
    shape on a day with a constant clear-sky index. Constant channels are skipped.
    """
    if arms730.input_equals_target_rows(window=window, targets=targets):
        return True
    scored = window[:, MID : MID + 48]
    delta_row = delta
    extras = [targets[T["event"]], targets[T["size"]]] + ([] if delta is None else [delta])
    for channel in scored:
        if np.ptp(channel) == 0.0:
            continue
        for row in extras:
            if np.ptp(row) == 0.0:
                continue
            if np.allclose(channel, row, atol=1e-6):
                return True
            if row is not delta_row and np.allclose(channel, row / (np.abs(row).max() or 1.0), atol=1e-6):
                return True
    return False


def all_picks(*, prepared: Sequence[Prepared]) -> list[tuple[int, int]]:
    """Return every (segment, base day) pair in order, the rows that a bank scores."""
    return [(i, int(d)) for i, p in enumerate(prepared) for d in p.days]


def sample_picks(*, prepared: Sequence[Prepared], n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Draw `n` picks, half from segments with real BSP steps (experiment 729's sampler)."""
    return arms729.sample_picks(prepared=prepared, n=n, rng=rng)  # ty: ignore[invalid-argument-type]

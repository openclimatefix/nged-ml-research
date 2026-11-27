# ruff: noqa: E501
"""Experiment 732: the target stack, the windows of a training triple, the batches, and the tensor checks.

This module needs no torch. A segment (with its load transfers applied) becomes a channel stack once,
through experiment 730's `prepare_segments`, and a window is a slice of it. The target stack has eight
rows: experiment 730's six (truth, valid, engine, solar, reading, base demand) and the step event and
the signed step size. A **triple** is the plain window `w`, the injected window `w+` (the same window
with an extra solar injection delta from a sibling segment), and the raised-sky window `wc`. The
three members share one hidden span of the reading. The inputs of `w+` differ from `w` only in the
reading channel, the BSP-step channel, and the sign-aligned neighbour channels (which read the
change of the reading); the inputs of `wc` differ only in the clear-sky index.

Every arm except U-N appends 12 neighbour channels to the 730 stack: for each of D3, D5, and D14, the
folded reading over the neighbour's training median, its half-hour change, that change times the
sign of D8's half-hour change of reading, and a presence mask. The neighbours are hidden wherever
D8's reading is hidden. U-noBSP keeps the stack's shape and sets the BSP-step channel and its mask to
zero. `tensor_checks` reads a built batch, not an overlay, and fails when a component went silent
between the overlay and the tensor.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import e00729_arms as arms729
import e00729_core as core729
import e00730_arms as arms730
import e00730_masks as masks
import e00732_core as core
import e00732_neighbours as nb
import e00732_overlays as ov
import numpy as np

STACK_SPEC: Final = arms730.ArmSpec("S732", "d", demand=False, style=core.STYLE)
"""The channel stack of every arm: experiment 730's arm S inputs."""
NAMES: Final = tuple(arms730.stack_names(spec=STACK_SPEC))
ROW: Final = {name: i for i, name in enumerate(NAMES)}
N_BASE: Final = len(NAMES)
"""The channels of experiment 730's arm S stack (the channels of a batch of U-N)."""
NEIGHBOUR_KINDS: Final = ("value", "change", "aligned", "mask")
NEIGHBOUR_NAMES: Final = tuple(f"{n}_{kind}" for n in nb.NEIGHBOURS for kind in NEIGHBOUR_KINDS)
NB_ROW: Final = {name: N_BASE + i for i, name in enumerate(NEIGHBOUR_NAMES)}
ALL_NAMES: Final = (*NAMES, *NEIGHBOUR_NAMES)
N_CHANNELS: Final = len(ALL_NAMES)
"""The channels of a batch of every arm except U-N: the stack and the 12 neighbour channels."""
TARGET_ROWS: Final = ("x", "valid", "engine", "solar", "r", "base", "event", "size")
T: Final = {name: i for i, name in enumerate(TARGET_ROWS)}
KEEP_ROWS: Final = ("sign", "solar", "step", "size", "injection", "night", "raisable")
K: Final = {name: i for i, name in enumerate(KEEP_ROWS)}
"""Rows of the keeps: where each loss term applies on the scored day."""
STEPS: Final = core729.WINDOW_STEPS
MID: Final = core729.MIDDLE_START


@dataclass
class Prepared:
    """A transfer-overlaid segment with its stack, targets, neighbours, and the arrays the pairs need.

    Attributes:
        base: Experiment 730's prepared segment of the overlaid segment (stack and six target rows).
        q: The reactive floor, flat over the dense axis.
        bsp: The overlaid BSP meter, flat (NaN where the meter has no data).
        beta: The segment's BSP coupling.
        event: The step event, flat.
        size: The signed step size, flat.
        nb_folded: The three neighbours' folded readings after the mirrors, `(3, L)` in points, NaN
            where absent.
        nb_folded_real: The same without the mirrors (the stored neighbours).
        medians: The neighbours' training medians, `(3,)`, the scale of the value channels.
        transfers: The segment's transfers (the tensor checks read their edges and partners).
        world: The simulated world type of the segment ("none" for a real segment).
        unreliable: The half-hours of S1 solar output that are zero-filled gaps, flat (None: none).
        dates: The date of each dense day (`datetime64[D]`, None when the segment has none).
    """

    base: arms730.PreparedSegment
    q: np.ndarray
    bsp: np.ndarray
    beta: float
    event: np.ndarray
    size: np.ndarray
    nb_folded: np.ndarray
    nb_folded_real: np.ndarray
    medians: np.ndarray
    transfers: list[ov.Transfer] = field(default_factory=list)
    world: str = "none"
    unreliable: np.ndarray | None = None
    dates: np.ndarray | None = None

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


def n_channels(*, spec: core.ArmSpec732) -> int:
    """Return the number of input channels of an arm's windows."""
    return N_CHANNELS if spec.neighbours else N_BASE


def _folded(
    *, series: Mapping[str, np.ndarray], floors: Mapping[str, float], length: int
) -> np.ndarray:
    """Return the three neighbours' folded readings `(3, L)`, all NaN for a neighbour with no series."""
    out = np.full((len(nb.NEIGHBOURS), length), np.nan, dtype=np.float32)
    for i, name in enumerate(nb.NEIGHBOURS):
        if name in series:
            out[i] = nb.fold(signed=series[name].reshape(-1), floor=floors.get(name, 0.0))
    return out


def prepare(
    *,
    overlays: Sequence[ov.TransferOverlay],
    holidays: np.ndarray,
    stats: arms729.InputStats,
    medians: Mapping[str, float],
) -> list[Prepared]:
    """Return the prepared segments of a bank's transfer overlays (segments without a base day are dropped).

    Args:
        overlays: The overlays, in the order of the bank's segments.
        holidays: Bank-holiday dates.
        stats: The standardisation constants of the temperature and wind speed.
        medians: The training median of each folded neighbour, by name (the scale of its channels).

    Returns:
        The prepared segments; `source_index` is the segment's position in the bank.
    """
    base = arms730.prepare_segments(
        segments=[o.seg for o in overlays], spec=STACK_SPEC, holidays=holidays, stats=stats
    )
    scale = np.asarray([medians[n] for n in nb.NEIGHBOURS], dtype=np.float64)
    out = []
    for b in base:
        o = overlays[b.source_index]
        length = o.seg.x.size
        out.append(
            Prepared(
                base=b, q=o.q.reshape(-1), bsp=o.seg.bsp.reshape(-1), beta=o.beta,
                event=o.event.reshape(-1), size=o.size.reshape(-1),
                nb_folded=_folded(series=o.neighbours, floors=o.floors, length=length),
                nb_folded_real=_folded(series=o.neighbours_real, floors=o.floors, length=length),
                medians=scale, transfers=list(o.transfers), world=str(getattr(o.seg, "world", "none")),
                unreliable=ov.unreliable_solar(seg=o.seg).reshape(-1),
                dates=np.asarray(o.seg.dates).astype("datetime64[D]") if hasattr(o.seg, "dates") else None,
            )
        )  # fmt: skip
    return out


def prepare_plain(
    *,
    segments: Sequence,
    holidays: np.ndarray,
    stats: arms729.InputStats,
    medians: Mapping[str, float],
    neighbours: Mapping[str, np.ndarray] | None = None,
) -> list[Prepared]:
    """Return the prepared segments of a bank or real set with no load transfer (all zeros).

    Args:
        segments: Simulated segments, or real pseudo-segments (which have no BSP coupling).
        holidays: Bank-holiday dates.
        stats: The standardisation constants of the temperature and wind speed.
        medians: The training median of each folded neighbour.
        neighbours: The neighbours' stored series on D8's base calendar, or None when the
            neighbours hold no reading (their channels are then all zero with a zero mask).

    Returns:
        The prepared segments.
    """
    simulated = all("beta" in s.params for s in segments)
    overlays = (
        ov.plain_overlays(segments=segments, neighbours=neighbours)
        if simulated
        else [_no_transfer(seg=s, neighbours=neighbours, index=i) for i, s in enumerate(segments)]
    )
    return prepare(overlays=overlays, holidays=holidays, stats=stats, medians=medians)


def _no_transfer(
    *, seg: object, neighbours: Mapping[str, np.ndarray] | None, index: int
) -> ov.TransferOverlay:
    """Return an overlay of a real pseudo-segment: no BSP coupling, no transfer, real neighbours.

    The neighbours' series are placed on the segment's dense axis with the segment's `cal_index`, so
    they must be on the same calendar as the pseudo-segment's real days. Their fold floors are drawn
    as `plain_overlays` draws them. Without `neighbours`, every neighbour channel is zero.
    """
    shape = seg.x.shape  # ty: ignore[unresolved-attribute]
    zeros = np.zeros(shape, dtype=np.float32)
    dense = None if neighbours is None else ov.dense_neighbours(seg=seg, series=neighbours)  # ty: ignore[invalid-argument-type]
    real = {} if dense is None else {k: v.astype(np.float32) for k, v in dense.items()}
    floors = ov.draw_floors(rng=np.random.default_rng(np.random.SeedSequence([0, 3, index])))
    return ov.TransferOverlay(
        seg=seg, q=zeros, beta=0.0, transfers=[], t=zeros, t_bsp=zeros, event=zeros, size=zeros, n_failed=0, rate=0.0,  # ty: ignore[invalid-argument-type]
        neighbours_real=real, neighbours=dict(real), floors=floors if real else {},
    )  # fmt: skip


def window_start(*, day: int) -> int:
    """Return the first stack column of the window of the scored day at dense index `day`."""
    return day * 48 - MID


def target_window(*, p: Prepared, day: int) -> np.ndarray:
    """Return the `(8, 48)` targets of the scored day."""
    a = day * 48
    rows = [p.base.targets[:, a : a + 48], p.event[None, a : a + 48], p.size[None, a : a + 48]]
    return np.concatenate(rows, axis=0).astype(np.float32).copy()


# --------------------------------------------------------------------------- neighbour channels


def reading_change(*, window: np.ndarray) -> np.ndarray:
    """Return D8's half-hour change of reading `(96,)`, zero wherever either half-hour is not shown.

    A hidden or missing half-hour has a zero reading and a zero mask, so a change that touches one
    is zero, and the sign of a hidden step never reaches the sign-aligned channel.
    """
    reading = window[ROW["r"]].astype(np.float64)
    shown = window[ROW["r_mask"]] > 0.5
    change = np.zeros_like(reading)
    change[1:] = np.where(shown[1:] & shown[:-1], reading[1:] - reading[:-1], 0.0)
    return change


def neighbour_rows(
    *, p: Prepared, start: int, window: np.ndarray, hidden: np.ndarray | None, real: bool = False
) -> np.ndarray:
    """Return the 12 neighbour channels `(12, 96)` of the window that starts at stack column `start`.

    Args:
        p: The prepared segment.
        start: The window's first stack column.
        window: The window's base channels `(N_BASE, 96)` after the hidden span is applied (the
            reading and its mask give the sign-aligned channel).
        hidden: The hidden positions `(96,)`, or None. A neighbour is hidden where D8 is.
        real: Build from the stored neighbours, without the transfers' mirrors (a check reads this).

    Returns:
        The channels in the order of `NEIGHBOUR_NAMES`.
    """
    folded = (p.nb_folded_real if real else p.nb_folded)[:, start : start + STEPS]
    change = reading_change(window=window)
    rows = [
        nb.channels(folded=folded[i], median=float(p.medians[i]), d8_change=change, hidden=hidden)
        for i in range(len(nb.NEIGHBOURS))
    ]
    return np.concatenate(rows, axis=0).astype(np.float32)


def plain_window(
    *, p: Prepared, day: int, spec: core.ArmSpec732, hidden: np.ndarray | None, real: bool = False
) -> np.ndarray:
    """Return the plain window `w` `(C, 96)`: the 730 stack with the hidden span, then the neighbours."""
    s = window_start(day=day)
    out = p.base.stack[:, s : s + STEPS].copy()
    if hidden is not None:
        masks.apply_hidden(window=out, hidden=hidden)
    if spec.neighbours:
        rows = neighbour_rows(p=p, start=s, window=out, hidden=hidden, real=real)
        out = np.concatenate([out, rows], axis=0)
    return out


def _zero_bsp(*, window: np.ndarray) -> None:
    """Set the BSP-step channel and its mask to zero, in place."""
    window[ROW["d_bsp"]] = 0.0
    window[ROW["bsp_mask"]] = 0.0


# --------------------------------------------------------------------------- the members


def injected_window(
    *, p: Prepared, day: int, plain: np.ndarray, delta: np.ndarray, hidden: np.ndarray | None,
    spec: core.ArmSpec732,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Return the injected member `w+` of a window, its `(8, 48)` targets, and the scored delta.

    The reading channel is re-folded from the truth less delta and the segment's reactive floor on
    the shown half-hours where delta is above zero, and the BSP-step channel is the half-hour change
    of the BSP meter less `beta * delta`, so both channels equal the plain window's elsewhere. The
    sign-aligned neighbour channels are rebuilt from the injected reading. No other channel changes.

    Args:
        p: The prepared segment.
        day: The scored day's dense index.
        plain: The plain window `(C, 96)`, after the hidden span is applied.
        delta: The extra injection over the window's 97 half-hours (the half-hour before it
            first), in points.
        hidden: The hidden positions of the window, or None.
        spec: The arm.

    Returns:
        The window, the targets of `w+` (truth less delta, solar plus delta, re-folded reading), and
        the scored delta `(48,)`.
    """
    s = window_start(day=day)
    out = plain.copy()
    here = delta[1:].astype(np.float64)
    x = p.base.targets[T["x"], s : s + STEPS]
    r_new = ov.refold(x=np.nan_to_num(x - here, nan=0.0), q=p.q[s : s + STEPS])
    moved = (plain[ROW["r_mask"]] > 0.5) & (here != 0.0)
    out[ROW["r"]] = np.where(moved, r_new / arms729.R_SCALE, plain[ROW["r"]])
    step = np.diff(p.beta * delta.astype(np.float64))
    shifted = (plain[ROW["bsp_mask"]] > 0.5) & (step != 0.0)
    out[ROW["d_bsp"]] = np.where(
        shifted, plain[ROW["d_bsp"]] - step / arms729.BSP_SCALE, plain[ROW["d_bsp"]]
    )
    if spec.neighbours:
        out[N_BASE:] = neighbour_rows(p=p, start=s, window=out[:N_BASE], hidden=hidden)
    target = target_window(p=p, day=day)
    scored = delta[1 + MID : 1 + MID + 48].astype(np.float32)
    target[T["x"]] -= scored
    target[T["solar"]] += scored
    target[T["r"]] = np.nan_to_num(
        ov.refold(x=target[T["x"]], q=p.q[day * 48 : day * 48 + 48]), nan=0.0
    )
    return out, target, scored


def raised_window(*, plain: np.ndarray, eps: float) -> np.ndarray:
    """Return the raised-sky member `wc`: the clear-sky index times `1 + eps` on sun-up half-hours."""
    out = plain.copy()
    out[ROW["csi"]] = ov.raise_sky(csi=plain[ROW["csi"]], cosz=plain[ROW["cosz"]], eps=eps)
    return out


def unreliable_day(*, p: Prepared, day: int) -> np.ndarray:
    """Return the unreliable S1 solar half-hours of the scored day `(48,)`."""
    if p.unreliable is None:
        return np.zeros(48, dtype=bool)
    return p.unreliable[day * 48 : day * 48 + 48]


def keeps_of(
    *, p: Prepared, day: int, window: np.ndarray, hidden: np.ndarray | None, target: np.ndarray
) -> np.ndarray:
    """Return the `(7, 48)` keeps of the scored day, in the order of `KEEP_ROWS`.

    The sign, step, size, and injection terms drop hidden half-hours; the solar term keeps them.
    """
    valid = target[T["valid"]] > 0.5
    gap = unreliable_day(p=p, day=day)
    shown = np.ones(48, dtype=bool) if hidden is None else ~hidden[MID : MID + 48]
    scored = slice(MID, MID + 48)
    cosz = window[ROW["cosz"], scored]
    csi = window[ROW["csi"], scored]
    out = np.zeros((len(KEEP_ROWS), 48), dtype=np.float32)
    out[K["sign"]] = valid & shown
    out[K["solar"]] = valid & ~gap
    out[K["step"]] = valid & shown
    out[K["size"]] = valid & shown & (target[T["event"]] > 0.5)
    out[K["injection"]] = valid & shown & ~gap
    out[K["night"]] = valid & (cosz <= core.NIGHT_COSZ)
    out[K["raisable"]] = valid & ov.raisable(csi=csi, cosz=cosz)
    return out


@dataclass
class Triple:
    """One window with its injected and raised members.

    Attributes:
        x: The plain window `(C, 96)`.
        xi: The injected window (a copy of `x` where there is no pair), or None.
        xc: The raised-sky window, or None.
        t: The plain targets `(8, 48)`.
        ti: The injected targets, or None.
        keeps: `(7, 48)`.
        delta: The scored delta `(48,)`.
        delta_win: The delta over the window's 97 half-hours (zero without a pair).
        hidden: The hidden positions `(96,)`.
    """

    x: np.ndarray
    xi: np.ndarray | None
    xc: np.ndarray | None
    t: np.ndarray
    ti: np.ndarray | None
    keeps: np.ndarray
    delta: np.ndarray
    delta_win: np.ndarray
    hidden: np.ndarray


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
        delta_win: The delta over each window's 97 half-hours `(B, 97)`.
        hidden: The hidden positions of each window `(B, 96)`.
        picks: The `(segment index in prepared, dense day)` of each window.
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
    delta_win: np.ndarray
    hidden: np.ndarray
    picks: list[tuple[int, int]]


def build_triple(
    *, p: Prepared, day: int, spec: core.ArmSpec732, hidden: np.ndarray | None,
    delta: np.ndarray | None, eps: float | None,
) -> Triple:  # fmt: skip
    """Return one window and, when the arm needs them, its injected and raised members.

    Args:
        p: The prepared segment.
        day: The scored day's dense index (at least 1).
        spec: The arm.
        hidden: The hidden positions `(96,)` shared by the three members, or None.
        delta: The extra injection over the window's 97 half-hours (`window_delta`), or None when
            the window has no pair.
        eps: The raised-sky factor, or None.

    Returns:
        The triple; its injected member is a copy of the plain window where `delta` is None.
    """
    plain = plain_window(p=p, day=day, spec=spec, hidden=hidden)
    target = target_window(p=p, day=day)
    keeps = keeps_of(p=p, day=day, window=plain, hidden=hidden, target=target)
    inj = inj_target = None
    scored = np.zeros(48, dtype=np.float32)
    if spec.inject:
        if delta is None:
            inj, inj_target = plain.copy(), target.copy()
        else:
            inj, inj_target, scored = injected_window(
                p=p, day=day, plain=plain, delta=delta, hidden=hidden, spec=spec
            )
    raised = None
    if spec.raised:
        raised = raised_window(plain=plain, eps=core.TEST_EPS if eps is None else eps)
    if spec.bsp_zeroed:
        for member in (plain, inj, raised):
            if member is not None:
                _zero_bsp(window=member)
    return Triple(
        x=plain, xi=inj, xc=raised, t=target, ti=inj_target, keeps=keeps, delta=scored,
        delta_win=np.zeros(ov.WINDOW_DELTA_LEN, np.float32) if delta is None else delta.astype(np.float32),
        hidden=np.zeros(STEPS, dtype=bool) if hidden is None else hidden,
    )  # fmt: skip


def collate(
    *, parts: Sequence[Triple], pair: Sequence[bool], amount: Sequence[float],
    picks: Sequence[tuple[int, int]], spec: core.ArmSpec732,
) -> Batch:  # fmt: skip
    """Stack the triples of a batch."""

    def stack(name: str) -> np.ndarray:
        return np.stack([getattr(part, name) for part in parts])

    return Batch(
        x=stack("x"), xi=stack("xi") if spec.inject else None, xc=stack("xc") if spec.raised else None,
        t=stack("t"), ti=stack("ti") if spec.inject else None, keeps=stack("keeps"),
        delta=stack("delta"), pair=np.asarray(pair, dtype=np.float32),
        amount=np.asarray(amount, dtype=np.float32), delta_win=stack("delta_win"),
        hidden=stack("hidden"), picks=list(picks),
    )  # fmt: skip


def make_batch(
    *, prepared: Sequence[Prepared], picks: Sequence[tuple[int, int]], spec: core.ArmSpec732,
    rng: np.random.Generator, index: ov.PairIndex | None,
) -> Batch:  # fmt: skip
    """Return a training batch: hidden spans on half the windows, a sibling pair, and a raised sky.

    The generator draws, in this order, one hidden-span decision per window, then the siblings and
    amounts of all windows together (`draw_pairs`, one operation), then one raised-sky factor per
    window, so a batch is the same whatever the schedule.

    Args:
        prepared: The prepared segments of the training bank.
        picks: `(segment index in prepared, dense day)` pairs.
        spec: The arm.
        rng: The generator (one draw stream per fit).
        index: The pair index of the training bank, or None for an arm without the injected member.

    Returns:
        The batch. A window whose month has no other solar segment with output that day takes no pair.
    """
    hidden: list[np.ndarray | None] = []
    for seg_i, day in picks:
        p = prepared[seg_i]
        masked = rng.uniform() < core729.MASK_PROBABILITY
        hidden.append(
            masks.hidden_positions(style=core.STYLE, rng=rng, utc_offset=p.offsets[day])
            if masked
            else None
        )
    paired = spec.inject and index is not None
    draw = (
        ov.draw_pairs(
            index=index,
            segments=[prepared[i].source_index for i, _ in picks],
            days=[d for _, d in picks],
            rng=rng,
        )
        if paired and index is not None
        else None
    )
    eps = rng.uniform(*core.EPS_RANGE, size=len(picks)) if spec.raised else None
    parts, pair, amount = [], [], []
    for k, (seg_i, day) in enumerate(picks):
        has_pair = draw is not None and draw.sibling[k] >= 0
        parts.append(
            build_triple(
                p=prepared[seg_i], day=day, spec=spec, hidden=hidden[k],
                delta=draw.delta[k] if draw is not None and has_pair else None,
                eps=None if eps is None else float(eps[k]),
            )
        )  # fmt: skip
        pair.append(has_pair)
        amount.append(float(draw.amount[k]) if draw is not None and has_pair else 0.0)
    return collate(parts=parts, pair=pair, amount=amount, picks=picks, spec=spec)


def make_eval_batch(
    *, prepared: Sequence[Prepared], picks: Sequence[tuple[int, int]], spec: core.ArmSpec732,
    index: ov.PairIndex | None = None, amount: float | None = None, eps: float | None = None,
    hidden: np.ndarray | None = None, hide_bsp: bool = False,
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
        hide_bsp: Whether to set the BSP step channel and its presence mask to zero in every member
            (table T5's check that the solar head does not read the BSP meter).

    Returns:
        The batch.
    """
    parts, pair, amt = [], [], []
    for i, (seg_i, day) in enumerate(picks):
        p = prepared[seg_i]
        delta = None
        if amount is not None and index is not None:
            sib = ov.fixed_sibling(index=index, segment=p.source_index, day=day)
            if sib is not None:
                delta = ov.window_delta(index=index, sibling=sib, amount=amount, day=day)
        parts.append(
            build_triple(
                p=p, day=day, spec=spec, hidden=None if hidden is None else hidden[i],
                delta=delta, eps=eps,
            )
        )  # fmt: skip
        pair.append(delta is not None)
        amt.append(amount if delta is not None and amount is not None else 0.0)
    batch = collate(parts=parts, pair=pair, amount=amt, picks=picks, spec=spec)
    if hide_bsp:
        for member in (batch.x, batch.xi, batch.xc):
            if member is not None:
                member[:, ROW["d_bsp"]] = 0.0
                member[:, ROW["bsp_mask"]] = 0.0
    return batch


def eval_spec(
    *, inject: bool = False, raised: bool = False, neighbours: bool = True, bsp_zeroed: bool = False
) -> core.ArmSpec732:
    """Return an arm that builds the members an evaluation needs (the weights play no role)."""
    return core.ArmSpec732(
        "eval", "unet", True, core.BASE_WEIGHTS, inject=inject, raised=raised,
        neighbours=neighbours, bsp_zeroed=bsp_zeroed,
    )  # fmt: skip


def input_equals_target(
    *, window: np.ndarray, targets: np.ndarray, delta: np.ndarray | None = None
) -> bool:
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
            if row is not delta_row and np.allclose(
                channel, row / (np.abs(row).max() or 1.0), atol=1e-6
            ):
                return True
    return False


def all_picks(*, prepared: Sequence[Prepared]) -> list[tuple[int, int]]:
    """Return every (segment, base day) pair in order, the rows that a bank scores."""
    return [(i, int(d)) for i, p in enumerate(prepared) for d in p.days]


def sample_picks(
    *, prepared: Sequence[Prepared], n: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Draw `n` picks, half from segments with real BSP steps (experiment 729's sampler)."""
    return arms729.sample_picks(prepared=prepared, n=n, rng=rng)  # ty: ignore[invalid-argument-type]


# --------------------------------------------------------------------------- checks on the tensors

TENSOR_BARS: Final = {
    "mirror_edges": core.EXACT_SHARE,
    "aligned_nonzero": core.EXACT_SHARE,
    "aligned_exact": core.EXACT_SHARE,
    "aligned_reading_not_truth": core.EXACT_SHARE,
    "hidden_zero": core.EXACT_SHARE,
    "injected_reading_on_delta": core.EXACT_SHARE,
    "injected_reading_elsewhere": core.EXACT_SHARE,
    "injected_bsp_on_delta": core.EXACT_SHARE,
    "injected_bsp_elsewhere": core.EXACT_SHARE,
    "raised_above": core.EXACT_SHARE,
    "injection_only_allowed_channels": core.EXACT_SHARE,
    "neighbours_unsigned_binary": core.EXACT_SHARE,
    "solar_target_positive": core.MIN_SOLAR_TARGET_SHARE,
    "channel_count": core.EXACT_SHARE,
    "bsp_zero": core.EXACT_SHARE,
}
"""The bar of each check on the tensors: all need 1.0 except the solar target (99%)."""
DETECTABLE: Final = 1e-5
"""The smallest change of a scaled channel that a check requires to be present."""


def _share(*, passed: int, n: int) -> ov.Share:
    """Return a share, with an empty counter as `n` of 0 (which fails)."""
    return ov.Share(passed / n if n else 0.0, n)


def _members(*, batch: Batch) -> list[np.ndarray]:
    """Return the batch's window tensors (the plain, injected, and raised members that exist)."""
    return [m for m in (batch.x, batch.xi, batch.xc) if m is not None]


def check_mirror_edges(
    *, batch: Batch, prepared: Sequence[Prepared], spec: core.ArmSpec732
) -> ov.Share:
    """Check that only the partner's channels differ at a mirrored transfer's edge.

    At each scored edge of a mirrored transfer, only the partner's channels differ from the window
    built without the overlay.

    The partner's value channel (at the edge half-hour or the one before) or its change channel
    differs, and the other two neighbours' value and change channels do not.
    """
    passed = n = 0
    for k, (seg_i, day) in enumerate(batch.picks):
        p = prepared[seg_i]
        start = window_start(day=day)
        hidden = batch.hidden[k]
        real = plain_window(
            p=p, day=day, spec=spec, hidden=hidden if hidden.any() else None, real=True
        )
        for tr in p.transfers:
            if not tr.mirrored:
                continue
            i = nb.NEIGHBOURS.index(tr.partner)
            for hh in (tr.start_half_hour, tr.end_half_hour):
                col = hh - start
                if not (MID <= col < MID + 48) or hidden[col] or hidden[col - 1]:
                    continue
                others = [
                    o
                    for o in p.transfers
                    if o is not tr and o.start_half_hour <= hh and o.end_half_hour >= hh - 1
                ]
                if others or not np.isfinite(p.nb_folded[i, hh - 1 : hh + 1]).all():
                    continue
                n += 1
                ok = True
                for j, name in enumerate(nb.NEIGHBOURS):
                    # a transfer that ends on a half-hour boundary leaves the value of its end
                    # half-hour alone, so the value is read at the edge and the half-hour before it
                    value = NB_ROW[f"{name}_value"]
                    changed = (
                        batch.x[k, value, col - 1 : col + 1] != real[value, col - 1 : col + 1]
                    ).any() or batch.x[k, NB_ROW[f"{name}_change"], col] != real[
                        NB_ROW[f"{name}_change"], col
                    ]
                    ok &= bool(changed) if j == i else not bool(changed)
                passed += ok
    return _share(passed=passed, n=n)


def check_aligned(*, batch: Batch) -> dict[str, ov.Share]:
    """Check the sign-aligned channel against D8's change of reading.

    The sign-aligned channel is the neighbour's change times the sign of D8's change of reading,
    never the truth's.
    """
    nonzero = [0, 0]
    exact = [0, 0]
    discriminating = [0, 0]
    truth_step = np.sign(np.diff(batch.t[:, T["x"]], axis=1))
    for member in (m for m in (batch.x, batch.xi) if m is not None):
        for b in range(len(member)):
            sign = np.sign(reading_change(window=member[b][:N_BASE]))
            for name in nb.NEIGHBOURS:
                change = member[b, NB_ROW[f"{name}_change"]]
                aligned = member[b, NB_ROW[f"{name}_aligned"]]
                want = change * sign
                right = np.isclose(aligned, want, atol=1e-6)
                exact[0] += int(right.sum())
                exact[1] += right.size
                both = (change != 0) & (sign != 0)
                nonzero[0] += int((right & (aligned != 0) & both).sum())
                nonzero[1] += int(both.sum())
                scored = slice(MID + 1, MID + 48)
                differs = (
                    both[scored] & (sign[scored] != truth_step[b, :47]) & (truth_step[b, :47] != 0)
                )
                discriminating[0] += int((right[scored] & differs).sum())
                discriminating[1] += int(differs.sum())
    return {
        "aligned_nonzero": _share(passed=nonzero[0], n=nonzero[1]),
        "aligned_exact": _share(passed=exact[0], n=exact[1]),
        "aligned_reading_not_truth": _share(passed=discriminating[0], n=discriminating[1]),
    }


def check_hidden(*, batch: Batch) -> ov.Share:
    """All 12 neighbour channels are zero on every hidden half-hour of every member."""
    passed = n = 0
    for member in _members(batch=batch):
        rows = member[:, N_BASE:]
        for b in range(len(member)):
            cols = np.flatnonzero(batch.hidden[b])
            n += cols.size
            passed += int((np.abs(rows[b][:, cols]).sum(axis=0) == 0).sum())
    return _share(passed=passed, n=n)


def check_injected(*, batch: Batch, prepared: Sequence[Prepared]) -> dict[str, ov.Share]:
    """Check what the injection changes in the injected member.

    The injected member's reading and BSP channels differ from the plain window's on half-hours with
    extra output, and nowhere else.
    """
    counts = {k: [0, 0] for k in ("reading_on", "reading_off", "bsp_on", "bsp_off")}
    if batch.xi is not None:
        for k in np.flatnonzero(batch.pair > 0.5):
            seg_i, day = batch.picks[k]
            p = prepared[seg_i]
            s = window_start(day=day)
            delta = batch.delta_win[k].astype(np.float64)
            x, xi = batch.x[k], batch.xi[k]
            shown = x[ROW["r_mask"]] > 0.5
            old = ov.refold(
                x=np.nan_to_num(p.base.targets[T["x"], s : s + STEPS]), q=p.q[s : s + STEPS]
            )
            new = ov.refold(
                x=np.nan_to_num(p.base.targets[T["x"], s : s + STEPS]) - delta[1:],
                q=p.q[s : s + STEPS],
            )
            moves = shown & (np.abs(new - old) / arms729.R_SCALE > DETECTABLE)
            differs = xi[ROW["r"]] != x[ROW["r"]]
            counts["reading_on"][0] += int((differs & moves).sum())
            counts["reading_on"][1] += int(moves.sum())
            still = delta[1:] == 0.0
            counts["reading_off"][0] += int((~differs & still).sum())
            counts["reading_off"][1] += int(still.sum())
            step = np.diff(p.beta * delta)
            present = x[ROW["bsp_mask"]] > 0.5
            bsp_moves = present & (np.abs(step) / arms729.BSP_SCALE > DETECTABLE)
            bsp_differs = xi[ROW["d_bsp"]] != x[ROW["d_bsp"]]
            counts["bsp_on"][0] += int((bsp_differs & bsp_moves).sum())
            counts["bsp_on"][1] += int(bsp_moves.sum())
            flat = step == 0.0
            counts["bsp_off"][0] += int((~bsp_differs & flat).sum())
            counts["bsp_off"][1] += int(flat.sum())
    return {
        "injected_reading_on_delta": _share(passed=counts["reading_on"][0], n=counts["reading_on"][1]),
        "injected_reading_elsewhere": _share(passed=counts["reading_off"][0], n=counts["reading_off"][1]),
        "injected_bsp_on_delta": _share(passed=counts["bsp_on"][0], n=counts["bsp_on"][1]),
        "injected_bsp_elsewhere": _share(passed=counts["bsp_off"][0], n=counts["bsp_off"][1]),
    }  # fmt: skip


INJECTION_MAY_CHANGE: Final = (
    ROW["r"],
    ROW["d_bsp"],
    *(NB_ROW[f"{n}_aligned"] for n in nb.NEIGHBOURS),
)
"""The channels an injected member may differ from its plain window in: the reading, the BSP step, and the three sign-aligned changes."""


def check_injection_only_allowed_channels(*, batch: Batch) -> ov.Share:
    """Check that every channel of w+ outside the reading, BSP, and sign-aligned channels equals w.

    The share counts windows with a sibling pair. A batch with no such window has an empty counter,
    which fails.
    """
    if batch.xi is None:
        return _share(passed=0, n=0)
    paired = np.flatnonzero(batch.pair > 0.5)
    other = [c for c in range(batch.x.shape[1]) if c not in INJECTION_MAY_CHANGE]
    passed = sum(int((batch.x[b][other] == batch.xi[b][other]).all()) for b in paired)
    return _share(passed=passed, n=int(paired.size))


def check_neighbours_unsigned_binary(*, batch: Batch) -> ov.Share:
    """Check that every neighbour value channel is non-negative and every neighbour mask is 0 or 1.

    The share counts the neighbour channels of every member of the batch.
    """
    passed = n = 0
    for member in _members(batch=batch):
        for name in nb.NEIGHBOURS:
            value, mask = member[:, NB_ROW[f"{name}_value"]], member[:, NB_ROW[f"{name}_mask"]]
            n += 1
            passed += int(bool((value >= 0.0).all() and np.isin(mask, (0.0, 1.0)).all()))
    return _share(passed=passed, n=n)


def check_raised(*, batch: Batch) -> ov.Share:
    """The raised member's clear-sky index is above the plain window's on every raisable half-hour."""
    if batch.xc is None:
        return _share(passed=0, n=0)
    plain = batch.x[:, ROW["csi"]]
    mask = ov.raisable(csi=plain, cosz=batch.x[:, ROW["cosz"]])
    return _share(
        passed=int((batch.xc[:, ROW["csi"]][mask] > plain[mask]).sum()), n=int(mask.sum())
    )


def check_solar_target(*, batch: Batch, prepared: Sequence[Prepared]) -> dict[str, ov.Share]:
    """Return the solar-target check on the sun-up half-hours of clear days in solar-world windows.

    The check excludes the unreliable half-hours (zero-filled gaps of S1 solar output). The
    unmasked share is returned as information and has no bar.

    Args:
        batch: The batch.
        prepared: The prepared segments.

    Returns:
        "solar_target_positive" (the bar's share) and "info_solar_target_positive_unmasked".
    """
    passed = n = passed_all = n_all = 0
    for k, (seg_i, day) in enumerate(batch.picks):
        if prepared[seg_i].world not in ov.SOLAR_WORLDS:
            continue
        scored = slice(MID, MID + 48)
        up = (batch.x[k, ROW["cosz"], scored] > core.SUN_UP_COSZ) & (batch.t[k, T["valid"]] > 0.5)
        csi = batch.x[k, ROW["csi"], scored]
        if not up.any() or csi[up].mean() < core.CLEAR_DAY_CSI:
            continue
        positive = batch.t[k, T["solar"]] > 0.0
        n_all += int(up.sum())
        passed_all += int(positive[up].sum())
        kept = up & ~unreliable_day(p=prepared[seg_i], day=day)
        n += int(kept.sum())
        passed += int(positive[kept].sum())
    return {
        "solar_target_positive": _share(passed=passed, n=n),
        "info_solar_target_positive_unmasked": _share(passed=passed_all, n=n_all),
    }


def check_arm_inputs(*, batch: Batch, spec: core.ArmSpec732) -> dict[str, ov.Share]:
    """Check that a batch holds the channels its arm reads.

    A batch has no neighbour channel for U-N, and a zero BSP channel and mask for U-noBSP.
    """
    members = _members(batch=batch)
    out = {
        "channel_count": _share(
            passed=sum(m.shape[1] == n_channels(spec=spec) for m in members), n=len(members)
        )
    }
    if spec.bsp_zeroed:
        zero = sum(
            int(not m[:, [ROW["d_bsp"], ROW["bsp_mask"]]].any(axis=(1, 2)).any()) for m in members
        )
        out["bsp_zero"] = _share(passed=zero, n=len(members))
    return out


def tensor_checks(
    *, batch: Batch, prepared: Sequence[Prepared], spec: core.ArmSpec732
) -> dict[str, ov.Share]:
    """Return every check on a built batch that applies to the arm.

    A failure stops the line at verdict row 0.

    Args:
        batch: A batch built by `make_batch` or `make_eval_batch`.
        prepared: The prepared segments the batch was built from.
        spec: The arm the batch was built for.

    Returns:
        The share of cases that pass each check. A check with no case fails, so a batch that never
        exercises a component is a failure.
    """
    out: dict[str, ov.Share] = check_arm_inputs(batch=batch, spec=spec)
    if out["channel_count"].value < 1.0:
        return out  # the other checks index channels that a wrong-shaped batch does not have
    if spec.neighbours:
        out["mirror_edges"] = check_mirror_edges(batch=batch, prepared=prepared, spec=spec)
        out.update(check_aligned(batch=batch))
        out["hidden_zero"] = check_hidden(batch=batch)
        out["neighbours_unsigned_binary"] = check_neighbours_unsigned_binary(batch=batch)
    if spec.inject:
        out["injection_only_allowed_channels"] = check_injection_only_allowed_channels(batch=batch)
        injected = check_injected(batch=batch, prepared=prepared)
        if spec.bsp_zeroed:
            injected = {k: v for k, v in injected.items() if "bsp" not in k}
        out.update(injected)
    if spec.raised:
        out["raised_above"] = check_raised(batch=batch)
    out.update(check_solar_target(batch=batch, prepared=prepared))
    return out


def tensor_problems(*, checks: Mapping[str, ov.Share]) -> list[str]:
    """Return the checks below their bar (an empty counter fails)."""
    return [
        name
        for name, share in checks.items()
        if not name.startswith("info_") and not ov.share_passes(share=share, bar=TENSOR_BARS[name])
    ]

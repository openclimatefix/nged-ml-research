# ruff: noqa: E501
"""Experiment 732: the load-transfer generator, the re-fold, the pairing of extra solar, and the raised sky.

This module needs no torch. An overlay is a pure function of a stored bank, a segment's position in
it, the neighbours' stored readings, and a fixed seed: it never edits a bank, and it builds new
arrays from a segment's stored arrays. A load transfer adds a block of load `T` (in points,
positive when D8 gains load) to D8's demand for a span, so the signed truth becomes `x + T`, and the
reading is re-folded from the segment's own reactive floor, recovered exactly as
`q = sqrt(max(r**2 - x**2, 0))`. The partner of a transfer is one of the read neighbours D3, D5, and
D14 (whose series then changes by minus `j T`), another Primary under BSP A that the neural network
does not read, or a Primary under another BSP (the BSP meter then moves by `beta T`). A contrastive
pair is a window and the same window with an extra solar injection delta taken from a sibling
segment of the same month. The injection overlay and the raised-sky overlay carry no transfer. No
function prints a month, a date, or a real value.
"""

import hashlib
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, NamedTuple

import e00732_paths  # noqa: F401

# isort: split
import e00729_core as core729
import e00729_sim as sim729
import e00732_core as core
import e00732_neighbours as nb
import numpy as np

SOLAR_WORLDS: Final = ("solar", "both")
CAPACITY_KEY: Final = "solar_capacity_points"
BETA_KEY: Final = "beta"
TOLERANCE: Final = 1e-4


class Share(NamedTuple):
    """A component check: the share of cases that pass, and how many cases there were.

    Attributes:
        value: The share that passed.
        n: The number of cases (an empty counter has `n` of 0 and fails).
    """

    value: float
    n: int


def share_passes(*, share: Share, bar: float = core.MIN_OUTPUT_SHARE) -> bool:
    """Return whether a share has cases and reaches the bar (an empty counter always fails)."""
    return share.n > 0 and bool(np.isfinite(share.value)) and share.value >= bar


# --------------------------------------------------------------------------- the re-fold


def reactive_floor(*, seg: sim729.Segment) -> np.ndarray:
    """Return the segment's reactive term `q = sqrt(max(r**2 - x**2, 0))`, NaN where the reading is.

    Experiment 729's fold draws `q` per half-hour and stores the truth `x` and the reading `r`, so
    the term is recovered exactly (up to float32 rounding) from the stored arrays.
    """
    r = seg.r.astype(np.float64)
    x = np.nan_to_num(seg.x.astype(np.float64), nan=0.0)
    return np.sqrt(np.maximum(r**2 - x**2, 0.0)).astype(np.float32)


def refold(*, x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the reading `sqrt(x**2 + q**2)` of a new truth and a reactive floor (NaN where `q` is)."""
    return np.sqrt(x.astype(np.float64) ** 2 + q.astype(np.float64) ** 2).astype(np.float32)


# --------------------------------------------------------------------------- load transfers


@dataclass(frozen=True, eq=False)
class Transfer:
    """One load transfer on a segment's dense axis.

    Attributes:
        start_tick: The first 5-minute tick of the transferred load.
        ticks: The duration in ticks.
        size: The signed mean size T0 in points (positive when D8 gains load).
        partner: "D3", "D5", or "D14" (a read neighbour, mirrored), "unread" (another Primary under
            BSP A), or "other_bsp" (a Primary under another BSP).
        shape: "constant", "demand", or "solar".
        jitter: The factor j of the partner's mirror (1.0 for an exact mirror).
        profile: The unit shape of the load on each half-hour from `start_half_hour` to
            `end_half_hour` inclusive. Its mean over the transfer's ticks is 1, so the mean of T is
            T0. None means a constant block.
    """

    start_tick: int
    ticks: int
    size: float
    partner: str = core.UNREAD_PARTNER
    shape: str = "constant"
    jitter: float = 1.0
    profile: np.ndarray | None = None

    @property
    def end_tick(self) -> int:
        """The first tick after the transfer."""
        return self.start_tick + self.ticks

    @property
    def start_half_hour(self) -> int:
        """The half-hour that holds the start edge."""
        return self.start_tick // core.TICKS_PER_HALF_HOUR

    @property
    def end_half_hour(self) -> int:
        """The half-hour that holds the end edge."""
        return self.end_tick // core.TICKS_PER_HALF_HOUR

    @property
    def inside_bsp_a(self) -> bool:
        """The partner sits under BSP A, so the BSP meter stays unchanged."""
        return self.partner != core.OTHER_BSP_PARTNER

    @property
    def mirrored(self) -> bool:
        """The partner is a neighbour that the neural network reads."""
        return self.partner in core.READ_PARTNERS

    def used_weights(self) -> np.ndarray:
        """Return the number of the transfer's ticks in each half-hour of `start_half_hour..end_half_hour`."""
        n = self.end_half_hour - self.start_half_hour + 1
        lo = (self.start_half_hour + np.arange(n)) * core.TICKS_PER_HALF_HOUR
        hi = lo + core.TICKS_PER_HALF_HOUR
        return np.clip(np.minimum(self.end_tick, hi) - np.maximum(self.start_tick, lo), 0, None)

    def mean_levels(self) -> np.ndarray:
        """Return the mean transferred load T on each half-hour from the start edge to the end edge.

        The load is built on the 5-minute grid and averaged over each half-hour, so a half-hour
        that holds an edge carries a partial mean.
        """
        n = self.end_half_hour - self.start_half_hour + 1
        profile = np.ones(n) if self.profile is None else np.asarray(self.profile, dtype=np.float64)
        weights = self.used_weights()
        return self.size * profile * weights / core.TICKS_PER_HALF_HOUR


@dataclass
class TransferOverlay:
    """A segment with its load transfers applied, and the mirrored neighbours.

    Attributes:
        seg: The transfer-applied copy of the segment (truth, reading, base demand, and BSP meter).
        q: The reactive floor of the stored segment, `(D + 2, 48)`.
        beta: The segment's BSP coupling.
        transfers: The accepted transfers.
        t: The mean transferred load in each half-hour, in points (a partial half-hour at an edge).
        t_bsp: The part of `t` that moves the BSP meter (transfers to a Primary under another BSP).
        event: 1 on a half-hour that holds a transfer edge.
        size: The signed change of D8's demand at each edge half-hour (plus or minus T0).
        n_failed: Transfers that could not be placed after `MAX_REDRAWS` redraws.
        rate: The drawn transfers per 30 days.
        n_dropped: Transfers dropped by the donor rule after `DONOR_REDRAWS` redraws of the size.
        neighbours_real: The neighbours' stored series on the segment's dense axis in points
            (D5 and D14 signed, D3 a magnitude), NaN where absent.
        neighbours: The same after the mirrors of the transfers (before any fold).
        floors: The reactive floor of each neighbour's fold in points (0 for D3, drawn per segment
            for D5 and D14).
    """

    seg: sim729.Segment
    q: np.ndarray
    beta: float
    transfers: list[Transfer]
    t: np.ndarray
    t_bsp: np.ndarray
    event: np.ndarray
    size: np.ndarray
    n_failed: int
    rate: float
    n_dropped: int = 0
    neighbours_real: dict[str, np.ndarray] = field(default_factory=dict)
    neighbours: dict[str, np.ndarray] = field(default_factory=dict)
    floors: dict[str, float] = field(default_factory=dict)


class TransferArrays(NamedTuple):
    """The arrays of a segment's transfers on its dense axis.

    Attributes:
        t: The load that D8 gains in each half-hour.
        t_bsp: The part of `t` that moves the BSP meter.
        event: 1 on a half-hour that holds an edge.
        size: The signed change of D8's demand at each edge half-hour.
        mirror: For each read neighbour, the load the partner loses (j times `T`).
    """

    t: np.ndarray
    t_bsp: np.ndarray
    event: np.ndarray
    size: np.ndarray
    mirror: dict[str, np.ndarray]


def scored_half_hours(*, seg: sim729.Segment) -> np.ndarray:
    """Return the flat mask of half-hours that a bank scores: base days with a valid truth."""
    on_base = np.repeat(seg.cal_index >= 0, 48)
    return on_base & (seg.valid & np.isfinite(seg.x)).reshape(-1)


def dense_neighbours(
    *, seg: sim729.Segment, series: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Place the neighbours' calendar series on a segment's dense axis.

    Args:
        seg: The segment (`cal_index` maps its dense days to calendar rows, -1 off base days).
        series: Each neighbour's series on D8's base calendar `(n_calendar_days, 48)` in points, as
            experiment 732's `neighbours.build_arrays` returns them (D5 and D14 signed).

    Returns:
        `(D + 2, 48)` arrays, NaN on padding days, non-base days, and where the neighbour is absent.
    """
    rows = np.where(seg.cal_index >= 0, seg.cal_index, 0)
    out = {}
    for name in nb.NEIGHBOURS:
        dense = np.asarray(series[name], dtype=np.float64)[rows].copy()
        dense[seg.cal_index < 0] = np.nan
        out[name] = dense
    return out


def draw_floors(*, rng: np.random.Generator) -> dict[str, float]:
    """Draw the reactive floors of the neighbours' fold in points (D3 is a magnitude and has none)."""
    per_mva = core729.POINTS / core729.CAPACITY_MVA
    return {
        "D3": 0.0,
        "D5": nb.draw_floor(rng=rng, points_per_mva=per_mva),
        "D14": nb.draw_floor(rng=rng, points_per_mva=per_mva),
    }


def _draw_partner(*, rng: np.random.Generator) -> str:
    """Draw a partner: a read neighbour (equal shares), an unread Primary, or a Primary under another BSP."""
    u = rng.uniform()
    if u < core.PARTNER_MIX["read"]:
        return core.READ_PARTNERS[int(rng.integers(0, len(core.READ_PARTNERS)))]
    if u < core.PARTNER_MIX["read"] + core.PARTNER_MIX[core.UNREAD_PARTNER]:
        return core.UNREAD_PARTNER
    return core.OTHER_BSP_PARTNER


def _draw_shape(*, rng: np.random.Generator) -> str:
    """Draw a shape: constant, demand-shaped, or solar-shaped."""
    u = rng.uniform()
    if u < core.SHAPE_MIX["constant"]:
        return "constant"
    if u < core.SHAPE_MIX["constant"] + core.SHAPE_MIX["demand"]:
        return "demand"
    return "solar"


def _draw_jitter(*, rng: np.random.Generator) -> float:
    """Draw the mirror factor: 1.0 in 70% of the mirrored transfers, uniform from 0.7 to 1.3 in the rest."""
    if rng.uniform() < core.MIRROR_JITTER_SHARE:
        return float(rng.uniform(*core.MIRROR_JITTER_RANGE))
    return 1.0


def shape_profile(
    *, probe: Transfer, shape: str, donor: np.ndarray, sun: np.ndarray, kappa: float
) -> np.ndarray | None:
    """Return the unit shape of a transfer on the half-hours from its start edge to its end edge.

    Args:
        probe: A transfer with the span (`start_tick`, `ticks`) that the shape covers.
        shape: "constant", "demand", or "solar".
        donor: The donor's demand over the same half-hours (NaN where absent): D8's base demand
            when D8 loses the load or the partner is not read, else the partner's series.
        sun: The CAMS clear-sky index times the clear-sky irradiance over 1,000 at D8 over the same
            half-hours (zero where absent).
        kappa: The depth of a solar-shaped transfer; it is halved until no half-hour holds less than
            `SHAPE_FLOOR` of T0.

    Returns:
        The profile, whose mean over the transfer's ticks is 1, or None when a demand-shaped load
        falls below the floor or the donor has no positive demand.
    """
    w = probe.used_weights().astype(np.float64)
    if shape == "constant":
        return np.ones(w.size)
    if shape == "demand":
        series = np.where(np.isfinite(donor), np.maximum(donor, 0.0), np.nan)
        if not np.isfinite(series).any():
            return None
        series = np.where(np.isfinite(series), series, np.nanmean(series))
        mean = float((w * series).sum() / w.sum())
        if not mean > 0.0:
            return None
        profile = series / mean
        return profile if profile[w > 0].min() >= core.SHAPE_FLOOR else None
    g = np.nan_to_num(sun, nan=0.0)
    g_bar = float((w * g).sum() / w.sum())
    for _ in range(60):
        profile = 1.0 + kappa * (g_bar - g)
        if profile[w > 0].min() >= core.SHAPE_FLOOR:
            return profile
        kappa /= 2.0
    return np.ones(w.size)


def donor_cap_t0(*, probe: Transfer, donor: np.ndarray, factor: float) -> float:
    """Return the largest T0 that the donor rule allows: `DONOR_CAP * min(donor / (factor * profile))`.

    Args:
        probe: The transfer with its profile.
        donor: The donor's demand on each half-hour from the start edge to the end edge (NaN where
            absent; absent half-hours do not bind).
        factor: The factor that multiplies the load on the donor's side (the mirror's j when the
            partner is the donor, else 1).

    Returns:
        The cap in points (zero or negative when the donor has no positive demand to give).
    """
    w = probe.used_weights()
    profile = np.ones(w.size) if probe.profile is None else np.asarray(probe.profile)
    use = (w > 0) & np.isfinite(donor)
    if not use.any():
        return float("inf")
    return float(core.DONOR_CAP * np.min(donor[use] / (factor * profile[use])))


class Drawn(NamedTuple):
    """The result of drawing a segment's transfers.

    Attributes:
        transfers: The accepted transfers.
        n_failed: Transfers that could not be placed.
        n_dropped: Transfers dropped by the donor rule.
        n_draws: The number of transfers that the rate asked for.
    """

    transfers: list[Transfer]
    n_failed: int
    n_dropped: int
    n_draws: int


@dataclass
class _SegmentArrays:
    """A segment's flat arrays that a transfer draw reads."""

    scored: np.ndarray
    on_scored: np.ndarray
    n_ticks: int
    neighbours: dict[str, np.ndarray]
    base: np.ndarray
    sun: np.ndarray


def _segment_arrays(
    *, seg: sim729.Segment, neighbours: Mapping[str, np.ndarray] | None
) -> _SegmentArrays:
    """Flatten the arrays of a segment that the transfer draw reads."""
    scored = np.flatnonzero(scored_half_hours(seg=seg))
    n_half = len(seg.dates) * 48
    on_scored = np.zeros(n_half, dtype=bool)
    on_scored[scored] = True
    return _SegmentArrays(
        scored=scored, on_scored=on_scored, n_ticks=n_half * core.TICKS_PER_HALF_HOUR,
        neighbours={name: np.asarray(v).reshape(-1) for name, v in (neighbours or {}).items()},
        base=np.nan_to_num(seg.base.reshape(-1), nan=0.0).astype(np.float64),
        sun=(np.nan_to_num(seg.csi) * np.nan_to_num(seg.clear)).reshape(-1),
    )  # fmt: skip


def _place(
    *, arr: _SegmentArrays, accepted: Sequence[Transfer], rng: np.random.Generator
) -> tuple[Transfer, float] | None:
    """Draw one transfer's placement, direction, partner, shape, and mirror, or None when it does not fit.

    Args:
        arr: The segment's arrays (scored half-hours, base demand, sun, and neighbours).
        accepted: The transfers already placed on the segment (a new one may not overlap them).
        rng: The segment's generator.

    Returns:
        The transfer (with a unit size and its profile) and the largest T0 that the donor rule
        allows, or None when the placement, the shape, or the partner's coverage fails.
    """
    lo_h, hi_h = (np.log(v * 2 * core.TICKS_PER_HALF_HOUR) for v in core.TRANSFER_HOURS)
    ticks = round(np.exp(rng.uniform(lo_h, hi_h)))
    first = int(rng.choice(arr.scored)) * core.TICKS_PER_HALF_HOUR + int(
        rng.integers(0, core.TICKS_PER_HALF_HOUR)
    )
    probe = Transfer(start_tick=first, ticks=ticks, size=1.0)
    overlaps = any(
        probe.start_tick < o.end_tick and o.start_tick < probe.end_tick for o in accepted
    )
    if probe.end_tick > arr.n_ticks - 1 or not arr.on_scored[probe.end_half_hour] or overlaps:
        return None
    direction = 1.0 if rng.uniform() < 0.5 else -1.0
    partner = _draw_partner(rng=rng)
    shape = _draw_shape(rng=rng)
    jitter = _draw_jitter(rng=rng) if partner in core.READ_PARTNERS else 1.0
    kappa = float(rng.uniform(*core.SOLAR_SHAPE_KAPPA))
    span = slice(probe.start_half_hour, probe.end_half_hour + 1)
    series = arr.neighbours.get(partner) if partner in core.READ_PARTNERS else None
    if partner in core.READ_PARTNERS and (
        series is None or np.isfinite(series[span]).mean() < core.MIN_PARTNER_COVERAGE
    ):
        return None
    partner_donates = direction > 0
    donor = series[span] if partner_donates and series is not None else arr.base[span]
    profile = shape_profile(probe=probe, shape=shape, donor=donor, sun=arr.sun[span], kappa=kappa)
    if profile is None:
        return None
    shaped = replace(
        probe, size=direction, profile=profile, shape=shape, partner=partner, jitter=jitter
    )
    if partner_donates and series is not None:
        cap = donor_cap_t0(probe=shaped, donor=series[span], factor=jitter)
    elif partner_donates:
        cap = float("inf")
    else:
        cap = donor_cap_t0(probe=shaped, donor=arr.base[span], factor=1.0)
    return shaped, cap


def draw_transfers(
    *, seg: sim729.Segment, rate: float, rng: np.random.Generator,
    neighbours: Mapping[str, np.ndarray] | None = None,
) -> Drawn:  # fmt: skip
    """Draw a segment's load transfers.

    A transfer is placed only when both its edges fall on scored half-hours, its span lies inside the
    segment's dense axis, it overlaps no accepted transfer, its shape holds the floor, and a read
    partner holds a reading on at least `MIN_PARTNER_COVERAGE` of its half-hours; otherwise the whole
    transfer is redrawn, up to `MAX_REDRAWS` times. The size is then redrawn up to `DONOR_REDRAWS`
    times until the donor keeps at least 20% of its demand on every half-hour, and the transfer is
    dropped (and counted) when no size does.

    Args:
        seg: The segment.
        rate: Transfers per 30 days.
        rng: The generator of this segment alone.
        neighbours: The neighbours' series on the segment's dense axis (`dense_neighbours`); None
            means no neighbour holds a reading, so no read partner can be placed.

    Returns:
        The accepted transfers and the counts of failed and dropped draws.
    """
    count = round(rate * seg.n_dense / 30.0)
    arr = _segment_arrays(seg=seg, neighbours=neighbours)
    lo_s, hi_s = (np.log(v) for v in core.TRANSFER_POINTS)
    accepted: list[Transfer] = []
    failed = dropped = 0
    for _ in range(count):
        placed = None
        for _attempt in range(core.MAX_REDRAWS if arr.scored.size else 0):
            placed = _place(arr=arr, accepted=accepted, rng=rng)
            if placed is not None:
                break
        if placed is None:
            failed += 1
            continue
        shaped, cap = placed
        sizes = np.exp(rng.uniform(lo_s, hi_s, core.DONOR_REDRAWS))
        fits = np.flatnonzero(sizes <= cap)
        if fits.size == 0:
            dropped += 1
            continue
        accepted.append(replace(shaped, size=shaped.size * float(sizes[fits[0]])))
    return Drawn(accepted, failed, dropped, count)


def transfer_arrays(*, transfers: Sequence[Transfer], shape: tuple[int, int]) -> TransferArrays:
    """Return `t`, `t_bsp`, `event`, `size`, and the mirrors of a segment's transfers on its dense axis.

    The load is built on the 5-minute grid and averaged over each half-hour, so a half-hour that
    holds an edge carries a partial mean.
    """
    n = shape[0] * shape[1]
    t = np.zeros(n)
    t_bsp = np.zeros(n)
    event = np.zeros(n)
    size = np.zeros(n)
    mirror = {name: np.zeros(n) for name in core.READ_PARTNERS}
    for tr in transfers:
        span = slice(tr.start_half_hour, tr.end_half_hour + 1)
        level = tr.mean_levels()
        t[span] += level
        if not tr.inside_bsp_a:
            t_bsp[span] += level
        if tr.mirrored:
            mirror[tr.partner][span] += tr.jitter * level
        event[tr.start_half_hour] = 1.0
        event[tr.end_half_hour] = 1.0
        size[tr.start_half_hour] += tr.size
        size[tr.end_half_hour] -= tr.size
    cast = {"t": t, "t_bsp": t_bsp, "event": event, "size": size}
    out = {k: v.astype(np.float32).reshape(shape) for k, v in cast.items()}
    return TransferArrays(
        out["t"], out["t_bsp"], out["event"], out["size"],
        {k: v.astype(np.float32).reshape(shape) for k, v in mirror.items()},
    )  # fmt: skip


def apply_transfers(
    *, seg: sim729.Segment, transfers: Sequence[Transfer], n_failed: int = 0, rate: float = 0.0,
    neighbours: Mapping[str, np.ndarray] | None = None, floors: Mapping[str, float] | None = None,
    n_dropped: int = 0,
) -> TransferOverlay:  # fmt: skip
    """Return the segment with its transfers applied: `x + T`, the re-folded reading, `base + T`, the BSP meter, and the mirrors.

    Args:
        seg: The stored segment (never edited).
        transfers: The transfers.
        n_failed: Transfers that could not be placed.
        rate: The drawn transfers per 30 days.
        neighbours: The neighbours' series on the segment's dense axis (`dense_neighbours`), before
            any mirror; None means the neighbours hold no reading.
        floors: The neighbours' fold floors in points (`draw_floors`); zero floors when None.
        n_dropped: Transfers dropped by the donor rule.

    Returns:
        The overlay; its `neighbours` hold each neighbour's series after the mirrors.
    """
    shape = seg.x.shape
    arrays = transfer_arrays(transfers=transfers, shape=shape)
    q = reactive_floor(seg=seg)
    x = (seg.x + arrays.t).astype(np.float32)
    beta = float(seg.params[BETA_KEY])
    changed = replace(
        seg,
        base=(seg.base + arrays.t).astype(np.float32),
        x=x,
        r=np.where(np.isfinite(seg.r), refold(x=np.nan_to_num(x, nan=0.0), q=q), np.nan).astype(
            np.float32
        ),
        bsp=(seg.bsp + beta * arrays.t_bsp).astype(np.float32),
    )
    real = {
        name: np.full(shape, np.nan, dtype=np.float32) if neighbours is None else
        np.asarray(neighbours[name]).astype(np.float32)
        for name in nb.NEIGHBOURS
    }  # fmt: skip
    after = {name: (real[name] - arrays.mirror[name]).astype(np.float32) for name in nb.NEIGHBOURS}
    return TransferOverlay(
        seg=changed, q=q, beta=beta, transfers=list(transfers), t=arrays.t, t_bsp=arrays.t_bsp,
        event=arrays.event, size=arrays.size, n_failed=n_failed, rate=rate, n_dropped=n_dropped,
        neighbours_real=real, neighbours=after,
        floors=dict.fromkeys(nb.NEIGHBOURS, 0.0) if floors is None else dict(floors),
    )  # fmt: skip


def overlay_seed(*, kind: str, seed: int | None) -> int:
    """Return the first seed-sequence entry of a bank's overlay (training: 732,100 plus the seed)."""
    if kind == "train":
        assert seed is not None
        return core.SEED_TRAIN_OVERLAY + seed
    return {"validation": core.SEED_VALIDATION_OVERLAY, "test": core.SEED_TEST_OVERLAY}[kind]


def overlay_rng(*, kind: str, seed: int | None, index: int) -> np.random.Generator:
    """Return the generator of segment `index` of a bank's transfer overlay."""
    return np.random.default_rng(
        np.random.SeedSequence([overlay_seed(kind=kind, seed=seed), 1, index])
    )


def floor_rng(*, kind: str, seed: int | None, index: int) -> np.random.Generator:
    """Return the generator of segment `index`'s neighbour fold floors (a stream apart from the transfers)."""
    return np.random.default_rng(
        np.random.SeedSequence([overlay_seed(kind=kind, seed=seed), 2, index])
    )


def overlay_segments(
    *, segments: Sequence[sim729.Segment], kind: str, neighbours: Mapping[str, np.ndarray],
    seed: int | None = None, first_index: int = 0,
) -> list[TransferOverlay]:  # fmt: skip
    """Return the transfer overlay of a bank: one overlay per segment, from the bank's fixed seed.

    Args:
        segments: The bank's segments.
        kind: "train", "validation", or "test".
        neighbours: The neighbours' series on D8's base calendar `(n_calendar_days, 48)` in points.
        seed: The training seed (None for validation and test).
        first_index: The position of the first segment in its bank (a leakage check builds one segment alone).

    Returns:
        The overlays, in the order of `segments`. The overlay of a segment depends only on the
        segment, its position, the neighbours, the kind, and the seed.
    """
    out = []
    for index, seg in enumerate(segments, start=first_index):
        rng = overlay_rng(kind=kind, seed=seed, index=index)
        rate = (
            float(rng.uniform(*core.TRANSFER_RATE_TRAIN))
            if kind == "train"
            else core.TRANSFER_RATE_FIXED
        )
        dense = dense_neighbours(seg=seg, series=neighbours)
        drawn = draw_transfers(seg=seg, rate=rate, rng=rng, neighbours=dense)
        out.append(
            apply_transfers(
                seg=seg, transfers=drawn.transfers, n_failed=drawn.n_failed, rate=rate,
                neighbours=dense, floors=draw_floors(rng=floor_rng(kind=kind, seed=seed, index=index)),
                n_dropped=drawn.n_dropped,
            )
        )  # fmt: skip
    return out


def plain_overlays(
    *, segments: Sequence[sim729.Segment], neighbours: Mapping[str, np.ndarray] | None,
    seed: int = 0,
) -> list[TransferOverlay]:  # fmt: skip
    """Return overlays with no transfer (the plain bank, the injection overlay, and the raised-sky overlay).

    The neighbours still enter every window as their stored readings, and their fold floors are
    drawn from `seed` and the segment's position, so a plain window and its transfer-overlaid twin
    share the floors of the same overlay kind only when the caller passes the same seed.
    """
    out = []
    for index, seg in enumerate(segments):
        dense = None if neighbours is None else dense_neighbours(seg=seg, series=neighbours)
        floors = draw_floors(rng=np.random.default_rng(np.random.SeedSequence([seed, 3, index])))
        out.append(apply_transfers(seg=seg, transfers=[], neighbours=dense, floors=floors))
    return out


def overlay_digest(*, overlays: Sequence[TransferOverlay]) -> str:
    """Return the SHA-256 of the overlays' truth, reading, BSP meter, edges, sizes, and neighbours."""
    h = hashlib.sha256()
    for o in overlays:
        arrays = (
            o.seg.x,
            o.seg.r,
            o.seg.bsp,
            o.event,
            o.size,
            *(o.neighbours[n] for n in nb.NEIGHBOURS),
        )
        for arr in arrays:
            h.update(np.ascontiguousarray(np.nan_to_num(arr, nan=-1.0)).tobytes())
        h.update(np.asarray([o.floors[n] for n in nb.NEIGHBOURS], dtype=np.float64).tobytes())
    return h.hexdigest()


def cached_overlays(
    *, path: Path, build: Callable[[], list[TransferOverlay]]
) -> list[TransferOverlay]:
    """Return the overlays stored at `path`, building and storing them once when the file is missing.

    The training overlay of a bank and seed is drawn once and read back by every fit that needs it.
    The file is written to a temporary name and renamed, so a crash leaves no half-written file.
    """
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    overlays = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("wb") as handle:
        pickle.dump(overlays, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return overlays


# --------------------------------------------------------------------------- extra solar and pairs


def solar_unit(*, seg: sim729.Segment) -> np.ndarray:
    """Return a solar-world segment's output divided by its drawn solar capacity, flat (zeros otherwise)."""
    capacity = float(seg.params.get(CAPACITY_KEY, 0.0))
    if seg.world not in SOLAR_WORLDS or capacity <= 0.0:
        return np.zeros(seg.solar.size, dtype=np.float32)
    return (np.nan_to_num(seg.solar.reshape(-1)) / capacity).astype(np.float32)


@dataclass
class PairIndex:
    """Which sibling segments can lend an extra injection to each window of a bank.

    Attributes:
        siblings: For each (segment, dense day) of a base day, the sibling segments of the same month
            with solar output on that day.
        units: The solar output over capacity of every solar-world segment, flat, by segment.
        windows: For each month, the number of base-day windows.
        unpaired: For each month, the number of base-day windows with no sibling.
        months: The month label of each segment (held in memory only).
        matrix: The `units` of the solar-world segments as rows of one array, zero-padded to the
            longest segment, so that a batch's pairs are one gather.
        row_of: The row of `matrix` of each solar-world segment's position in the bank.
        segment_of_row: The segment position of each row of `matrix`.
        flat: The siblings of every window, concatenated (as rows of `matrix`).
        span: For each (segment, dense day) with a sibling, the offset and count in `flat`.
    """

    siblings: dict[tuple[int, int], np.ndarray]
    units: dict[int, np.ndarray]
    windows: dict[str, int]
    unpaired: dict[str, int]
    months: list[str]
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    row_of: dict[int, int] = field(default_factory=dict)
    segment_of_row: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    flat: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    span: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)

    def share_unpaired(self) -> dict[str, float]:
        """Return the share of each month's windows with no sibling, in time order of the month label."""
        return {
            m: self.unpaired[m] / self.windows[m] for m in sorted(self.windows) if self.windows[m]
        }


def build_pair_index(
    *, segments: Sequence[sim729.Segment], sources: Sequence[str] = ("S1", "S2")
) -> PairIndex:
    """Find, for every base-day window, the sibling segments that can lend it an extra injection.

    Args:
        segments: The bank's segments.
        sources: The solar sources a sibling may have (training: S1 and S2; the published test
            overlay: S2 only).

    Returns:
        The index. A sibling is another segment of the same month and bank, of a solar world, whose
        source is in `sources`, with solar output above zero on the window's day.
    """
    by_month: dict[str, list[int]] = {}
    units: dict[int, np.ndarray] = {}
    day_max: dict[int, np.ndarray] = {}
    for j, seg in enumerate(segments):
        if seg.world in SOLAR_WORLDS and seg.source in sources:
            by_month.setdefault(seg.month, []).append(j)
            units[j] = solar_unit(seg=seg)
            day_max[j] = units[j].reshape(-1, 48).max(axis=1)
    siblings: dict[tuple[int, int], np.ndarray] = {}
    windows: dict[str, int] = {}
    unpaired: dict[str, int] = {}
    for i, seg in enumerate(segments):
        pool = by_month.get(seg.month, [])
        for d in range(1, seg.n_dense + 1):
            if seg.cal_index[d] < 0:
                continue
            windows[seg.month] = windows.get(seg.month, 0) + 1
            eligible = np.array([j for j in pool if j != i and day_max[j][d] > 0.0], dtype=np.int64)
            if eligible.size:
                siblings[i, d] = eligible
            else:
                unpaired[seg.month] = unpaired.get(seg.month, 0) + 1
    for m in windows:
        unpaired.setdefault(m, 0)
    row_of = {j: r for r, j in enumerate(sorted(units))}
    width = max((u.size for u in units.values()), default=0)
    matrix = np.zeros((len(row_of), width), dtype=np.float32)
    for j, r in row_of.items():
        matrix[r, : units[j].size] = units[j]
    flat_parts: list[int] = []
    span: dict[tuple[int, int], tuple[int, int]] = {}
    for key in sorted(siblings):
        span[key] = (len(flat_parts), siblings[key].size)
        flat_parts.extend(row_of[int(j)] for j in siblings[key])
    return PairIndex(
        siblings=siblings, units=units, windows=windows, unpaired=unpaired,
        months=[s.month for s in segments], matrix=matrix, row_of=row_of,
        segment_of_row=np.asarray(sorted(units), dtype=np.int64),
        flat=np.asarray(flat_parts, dtype=np.int64), span=span,
    )  # fmt: skip


def pick_sibling(
    *, index: PairIndex, segment: int, day: int, rng: np.random.Generator
) -> int | None:
    """Return a sibling segment for a window, or None when no sibling has output that day."""
    eligible = index.siblings.get((segment, day))
    if eligible is None:
        return None
    return int(rng.choice(eligible))


def fixed_sibling(*, index: PairIndex, segment: int, day: int) -> int | None:
    """Return the test overlay's sibling of a window, picked from seed 732,000 by the window alone."""
    rng = np.random.default_rng(np.random.SeedSequence([core.SEED_TEST_OVERLAY, 7, segment, day]))
    return pick_sibling(index=index, segment=segment, day=day, rng=rng)


def delta_flat(*, index: PairIndex, sibling: int, amount: float) -> np.ndarray:
    """Return the extra injection `amount * unit` of a sibling, flat over the dense axis, in points."""
    return (amount * index.units[sibling]).astype(np.float32)


def draw_amount(*, rng: np.random.Generator) -> float:
    """Draw an extra-injection amount a (log-uniform from 2 to 60 points)."""
    lo, hi = np.log(core.INJECTION_POINTS[0]), np.log(core.INJECTION_POINTS[1])
    return float(np.exp(rng.uniform(lo, hi)))


WINDOW_DELTA_LEN: Final = core729.WINDOW_STEPS + 1
"""A window's delta spans its 96 half-hours and the half-hour before (the BSP step reads two)."""


def window_columns(*, days: np.ndarray) -> np.ndarray:
    """Return the dense columns `(B, 97)` of each window's delta: from the half-hour before the window."""
    start = np.asarray(days, dtype=np.int64) * 48 - core729.MIDDLE_START
    return (start - 1)[:, None] + np.arange(WINDOW_DELTA_LEN)[None, :]


class PairDraw(NamedTuple):
    """The extra injections of a batch of windows, drawn together.

    Attributes:
        sibling: The segment position of each window's sibling, -1 where the window has no pair.
        amount: The amount a of each window in points (0 where there is no pair).
        delta: The injection over each window's 97 half-hours, `(B, 97)` in points (zero without a pair).
    """

    sibling: np.ndarray
    amount: np.ndarray
    delta: np.ndarray


def draw_pairs(
    *, index: PairIndex, segments: Sequence[int], days: Sequence[int], rng: np.random.Generator
) -> PairDraw:
    """Draw every window's sibling and amount, and gather the extra injections in one operation.

    The sibling of each window is a uniform draw among the eligible siblings, and the amount is
    log-uniform from 2 to 60 points. One uniform and one amount are drawn per window, so the stream
    does not depend on which windows have a sibling.

    Args:
        index: The pair index of the training bank.
        segments: The segment position of each window.
        days: The dense day of each window.
        rng: The fit's generator.

    Returns:
        The draw; a window without a sibling gets sibling -1, amount 0, and delta 0.
    """
    n = len(segments)
    u = rng.random(n)
    lo, hi = np.log(core.INJECTION_POINTS[0]), np.log(core.INJECTION_POINTS[1])
    amount = np.exp(rng.uniform(lo, hi, n))
    offset = np.full(n, -1, dtype=np.int64)
    count = np.zeros(n, dtype=np.int64)
    for k, key in enumerate(zip(segments, days, strict=True)):
        found = index.span.get((int(key[0]), int(key[1])))
        if found is not None:
            offset[k], count[k] = found
    paired = count > 0
    pick = np.minimum((u * np.maximum(count, 1)).astype(np.int64), np.maximum(count, 1) - 1)
    row = np.zeros(n, dtype=np.int64)
    if index.flat.size:
        row = np.where(paired, index.flat[np.where(paired, offset + pick, 0)], 0)
    sibling = np.where(paired, index.segment_of_row[row] if index.flat.size else -1, -1)
    if index.matrix.size == 0:
        delta = np.zeros((n, WINDOW_DELTA_LEN), dtype=np.float32)
    else:
        cols = window_columns(days=np.asarray(days))
        delta = index.matrix[row[:, None], cols] * (amount * paired)[:, None]
    return PairDraw(
        sibling, np.where(paired, amount, 0.0).astype(np.float32), delta.astype(np.float32)
    )


def window_delta(*, index: PairIndex, sibling: int, amount: float, day: int) -> np.ndarray:
    """Return one window's extra injection `amount * unit` over its 97 half-hours, in points."""
    cols = window_columns(days=np.asarray([day]))[0]
    return (amount * index.matrix[index.row_of[sibling], cols]).astype(np.float32)


def inject(
    *, x: np.ndarray, q: np.ndarray, delta: np.ndarray, bsp: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the injected truth, re-folded reading, and BSP meter: `x - delta`, `sqrt(.)`, `bsp - beta delta`."""
    x_new = (x - delta).astype(np.float32)
    return (
        x_new,
        refold(x=np.nan_to_num(x_new, nan=0.0), q=q),
        (bsp - beta * delta).astype(np.float32),
    )


# --------------------------------------------------------------------------- the raised sky


def raisable(*, csi: np.ndarray, cosz: np.ndarray) -> np.ndarray:
    """Return the raisable half-hours: sun-up with a clear-sky index above 0 and below 1.25."""
    return (cosz > core.SUN_UP_COSZ) & (csi > 0.0) & (csi < core.RAISABLE_MAX_CSI)


def raise_sky(*, csi: np.ndarray, cosz: np.ndarray, eps: float) -> np.ndarray:
    """Return the clear-sky index multiplied by `1 + eps` on sun-up half-hours, clipped at 1.5.

    Nothing changes where the index is 0, and the CAMS presence mask and every other channel are
    left to the caller (they stay as they were).
    """
    raised = np.where(cosz > core.SUN_UP_COSZ, np.minimum(csi * (1.0 + eps), core.CSI_CLIP), csi)
    return raised.astype(csi.dtype)


# --------------------------------------------------------------------------- component checks


def _ratio(counter: list[int]) -> Share:
    """Return a counter `[passed, total]` as a share (NaN and an empty count when no case existed)."""
    return Share(counter[0] / counter[1] if counter[1] else float("nan"), counter[1])


def transfer_checks(  # noqa: PLR0915
    *, overlays: Sequence[TransferOverlay], originals: Sequence[sim729.Segment]
) -> dict[str, Share]:
    """Return the component checks of the transfer overlay, mirror checks included.

    Args:
        overlays: The overlays.
        originals: The stored segments they were built from, in the same order.

    Returns:
        For each check, the share of cases that pass and the number of cases:

        - `transfer_exact`: a transfer changes the truth by its own load on every scored half-hour
          inside it, and its mean load over its ticks is T0.
        - `transfer_edges_scored`: both edges fall on scored half-hours that carry the event.
        - `bsp_inside_unchanged` and `bsp_outside_moves`: the BSP meter stays unchanged for a
          transfer inside BSP A and moves by beta times the load for any other.
        - `shape_floor`: the load of every solar-shaped or demand-shaped transfer is at least
          `SHAPE_FLOOR` of T0 on every half-hour that holds a tick of it, so T never changes sign.
        - `solar_shape_differs`: a solar-shaped transfer that holds a sun-up half-hour differs from
          T0 on one (a transfer that lies wholly at night cannot).
        - `donor_dropped`: the share of the transfers that the rate asked for that the donor rule
          did not drop (`MAX_DROPPED_SHARE` bounds the dropped share).
        - `mirror_exact`: for every transfer whose partner is D3, D5, or D14, the mean change of the
          partner's series before the fold, over the transfer's half-hours with a reading, equals
          minus j times the mean of T within `MIRROR_TOLERANCE_POINTS`.
        - `mirror_nonzero`: that mean change is nonzero on every mirrored transfer.
        - `mirror_nowhere_else`: a neighbour's series changes nowhere outside the spans of the
          transfers whose partner it is, so no neighbour moves for a partner that is not read, at
          the wrong neighbour, or outside a transfer.
    """
    exact, edges, inside, outside = [0, 0], [0, 0], [0, 0], [0, 0]
    floor, sun, mirror_exact, mirror_nonzero, nowhere = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    drawn = dropped = 0
    for ov_, seg in zip(overlays, originals, strict=True):
        scored = scored_half_hours(seg=seg)
        diff = (ov_.seg.x - seg.x).reshape(-1)
        bsp_diff = (ov_.seg.bsp - seg.bsp).reshape(-1)
        ok_bsp = np.isfinite(seg.bsp.reshape(-1))
        cosz = np.nan_to_num(seg.cosz).reshape(-1)
        t_flat = ov_.t.reshape(-1)
        drawn += len(ov_.transfers) + ov_.n_dropped + ov_.n_failed
        dropped += ov_.n_dropped
        own_span = {name: np.zeros(scored.size, dtype=bool) for name in nb.NEIGHBOURS}
        for tr in ov_.transfers:
            lo, hi = tr.start_half_hour, tr.end_half_hour
            level = tr.mean_levels()
            inner = np.zeros_like(scored)
            inner[lo + 1 : hi] = True
            inner &= scored
            want = np.zeros(scored.size)
            want[lo : hi + 1] = level
            if inner.any():
                exact[1] += 1
                mean_t = float(level.sum() * core.TICKS_PER_HALF_HOUR / tr.ticks)
                ok_mean = abs(mean_t - tr.size) <= TOLERANCE * max(1.0, abs(tr.size))
                exact[0] += int(
                    ok_mean
                    and np.allclose(
                        diff[inner], want[inner], atol=TOLERANCE * max(1.0, abs(tr.size))
                    )
                )
            for hh in (lo, hi):
                edges[1] += 1
                edges[0] += int(scored[hh] and ov_.event[hh // 48, hh % 48] > 0.0)
            bsp_span = inner & ok_bsp
            if bsp_span.any():
                if tr.inside_bsp_a:
                    inside[1] += 1
                    inside[0] += int(np.allclose(bsp_diff[bsp_span], 0.0, atol=TOLERANCE))
                else:
                    outside[1] += 1
                    expected = ov_.beta * want[bsp_span]
                    outside[0] += int(
                        np.allclose(
                            bsp_diff[bsp_span], expected, atol=TOLERANCE * max(1.0, abs(tr.size))
                        )
                    )
            if tr.shape in ("demand", "solar"):
                used = tr.used_weights() > 0
                floor[1] += 1
                profile = np.asarray(tr.profile)
                floor[0] += int(bool((profile[used] >= core.SHAPE_FLOOR - 1e-9).all()))
            up = (cosz[lo : hi + 1] > core.SUN_UP_COSZ) & (tr.used_weights() > 0)
            if tr.shape == "solar" and up.any():
                sun[1] += 1
                sun[0] += int(bool((np.abs(np.asarray(tr.profile)[up] - 1.0) > 1e-9).any()))
            if tr.mirrored:
                own_span[tr.partner][lo : hi + 1] = True
                series = ov_.neighbours[tr.partner].reshape(-1)
                real = ov_.neighbours_real[tr.partner].reshape(-1)
                seen = np.zeros(scored.size, dtype=bool)
                seen[lo : hi + 1] = np.isfinite(real[lo : hi + 1])
                if not seen.any():
                    continue
                change = float(np.mean((series - real)[seen]))
                expected_change = -tr.jitter * float(np.mean(t_flat[seen]))
                mirror_exact[1] += 1
                mirror_exact[0] += int(
                    abs(change - expected_change) <= core.MIRROR_TOLERANCE_POINTS
                )
                mirror_nonzero[1] += 1
                mirror_nonzero[0] += int(abs(change) > core.MIRROR_TOLERANCE_POINTS)
        for name in nb.NEIGHBOURS:
            moved = np.nan_to_num(ov_.neighbours[name] - ov_.neighbours_real[name]).reshape(-1)
            nowhere[1] += 1
            nowhere[0] += int(not (np.abs(moved) > TOLERANCE)[~own_span[name]].any())
    out = {
        "transfer_exact": exact,
        "transfer_edges_scored": edges,
        "bsp_inside_unchanged": inside,
        "bsp_outside_moves": outside,
        "shape_floor": floor,
        "solar_shape_differs": sun,
        "mirror_exact": mirror_exact,
        "mirror_nonzero": mirror_nonzero,
        "mirror_nowhere_else": nowhere,
    }
    shares = {k: _ratio(v) for k, v in out.items()}
    shares["donor_dropped"] = Share(1.0 - dropped / drawn if drawn else float("nan"), drawn)
    return shares


CHECK_BARS: Final = {
    "transfer_exact": core.MIN_OUTPUT_SHARE,
    "transfer_edges_scored": core.MIN_OUTPUT_SHARE,
    "bsp_inside_unchanged": core.MIN_OUTPUT_SHARE,
    "bsp_outside_moves": core.MIN_OUTPUT_SHARE,
    "shape_floor": core.EXACT_SHARE,
    "solar_shape_differs": core.MIN_OUTPUT_SHARE,
    "donor_dropped": 1.0 - core.MAX_DROPPED_SHARE,
    "mirror_exact": core.EXACT_SHARE,
    "mirror_nonzero": core.EXACT_SHARE,
    "mirror_nowhere_else": core.EXACT_SHARE,
}
"""The bar of each overlay component check; the mirror and shape-floor checks need 1.0."""


def check_problems(*, checks: Mapping[str, Share]) -> list[str]:
    """Return the checks below their bar (an empty counter fails), which stop the line at verdict row 0."""
    return [
        name
        for name, share in checks.items()
        if not share_passes(share=share, bar=CHECK_BARS.get(name, core.MIN_OUTPUT_SHARE))
    ]


def shape_counts(*, overlays: Sequence[TransferOverlay]) -> dict[str, dict[str, int]]:
    """Return how many transfers of each shape, partner, and mirror kind the overlays hold (for the registered shares)."""
    shapes = dict.fromkeys(core.SHAPES, 0)
    partners = dict.fromkeys((*core.READ_PARTNERS, core.UNREAD_PARTNER, core.OTHER_BSP_PARTNER), 0)
    mirrors = {"exact": 0, "jittered": 0}
    for o in overlays:
        for tr in o.transfers:
            shapes[tr.shape] += 1
            partners[tr.partner] += 1
            if tr.mirrored:
                mirrors["exact" if tr.jitter == 1.0 else "jittered"] += 1
    return {"shape": shapes, "partner": partners, "mirror": mirrors}


def pair_checks(*, segments: Sequence[sim729.Segment], index: PairIndex, amount: float, n_pairs: int = 200,
                seed: int = 0) -> dict[str, Share]:  # fmt: skip
    """Return the component checks of the extra-injection pairs on a sample of windows.

    Args:
        segments: The bank's segments.
        index: The pair index of the bank.
        amount: The extra-injection amount a.
        n_pairs: The number of windows sampled.
        seed: The sampling seed.

    Returns:
        The share of pairs whose delta is above zero on at least one sun-up half-hour, and the share
        whose re-folded reading equals the original where delta is zero (1.0 is required).
    """
    keys = sorted(index.siblings)
    rng = np.random.default_rng(seed)
    chosen = (
        [keys[int(k)] for k in rng.choice(len(keys), size=min(n_pairs, len(keys)), replace=False)]
        if keys
        else []
    )
    active = [0, 0]
    equal = [0, 0]
    for i, d in chosen:
        seg = segments[i]
        j = fixed_sibling(index=index, segment=i, day=d)
        if j is None:
            continue
        delta = delta_flat(index=index, sibling=j, amount=amount)
        q = reactive_floor(seg=seg)
        _, r_new, _ = inject(
            x=seg.x,
            q=q,
            delta=delta.reshape(seg.x.shape),
            bsp=seg.bsp,
            beta=float(seg.params[BETA_KEY]),
        )
        day = slice(d * 48, d * 48 + 48)
        dd = delta[day]
        cosz = seg.cosz.reshape(-1)[day]
        active[1] += 1
        active[0] += int(((dd > 0.0) & (cosz > 0.0)).any())
        zero = (dd == 0.0) & np.isfinite(seg.r.reshape(-1)[day])
        equal[1] += 1
        equal[0] += int(np.array_equal(r_new.reshape(-1)[day][zero], seg.r.reshape(-1)[day][zero]))
    return {
        "pair_delta_active": Share(active[0] / active[1] if active[1] else float("nan"), active[1]),
        "pair_refold_equal": Share(equal[0] / equal[1] if equal[1] else float("nan"), equal[1]),
    }


def raised_checks(*, segments: Sequence[sim729.Segment], eps: float = core.TEST_EPS) -> Share:
    """Return the share of raised-sky windows whose index is above the original on every raisable half-hour."""
    ok = n = 0
    for seg in segments:
        for d in range(1, seg.n_dense + 1):
            if seg.cal_index[d] < 0:
                continue
            csi = np.nan_to_num(seg.csi[d])
            cosz = np.nan_to_num(seg.cosz[d])
            mask = raisable(csi=csi, cosz=cosz)
            if not mask.any():
                continue
            new = raise_sky(csi=np.clip(csi, 0.0, core.CSI_CLIP), cosz=cosz, eps=eps)
            n += 1
            ok += int((new[mask] > np.clip(csi, 0.0, core.CSI_CLIP)[mask]).all())
    return Share(ok / n if n else float("nan"), n)


def reachability(
    *,
    overlays: Sequence[TransferOverlay],
    index: PairIndex | None,
    n_windows_by_amount: Mapping[float, int] | None,
) -> dict[str, tuple[int, int, bool]]:
    """Return the reachability counts (count, bar, passes) of the test overlay."""
    steps = 0
    for ov in overlays:
        scored = scored_half_hours(seg=ov.seg).reshape(ov.event.shape)
        steps += int(
            ((ov.event > 0) & scored & (np.abs(ov.size) >= core.MIN_STEP_POINTS_REACHABLE)).sum()
        )
    out = {
        "transfer_steps_5_points": (
            steps,
            core.MIN_TRANSFER_STEPS_TEST,
            steps >= core.MIN_TRANSFER_STEPS_TEST,
        )
    }
    for a, n in (n_windows_by_amount or {}).items():
        out[f"windows_at_{a:g}"] = (
            n,
            core.MIN_WINDOWS_PER_AMOUNT,
            n >= core.MIN_WINDOWS_PER_AMOUNT,
        )
    return out


def sibling_windows(*, segments: Sequence[sim729.Segment], sources: Sequence[str] = ("S2",)) -> int:
    """Return how many base-day windows of the bank have a sibling of the given sources."""
    return len(build_pair_index(segments=segments, sources=sources).siblings)


def unpaired_problems(
    *, share_unpaired: Mapping[str, float], bar: float = core.MAX_UNPAIRED_SHARE
) -> list[str]:
    """Return the months (by position in time order) whose unpaired share exceeds the bar."""
    return [f"month {i}" for i, (_, v) in enumerate(sorted(share_unpaired.items())) if not v <= bar]


Rng = Callable[[], np.random.Generator]

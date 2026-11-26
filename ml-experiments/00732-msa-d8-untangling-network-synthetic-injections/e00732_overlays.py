# ruff: noqa: E501
"""Experiment 732: the load-transfer generator, the re-fold, the pairing of extra solar, and the raised sky.

This module needs no torch. An overlay is a pure function of a stored bank, a segment's position in
it, and a fixed seed: it never edits a bank, and it builds new arrays from a segment's stored arrays.
A load transfer adds a block of load `T` (in points, positive when D8 gains load) to D8's demand for
a span, so the signed truth becomes `x + T`, and the reading is re-folded from the segment's own
reactive floor, recovered exactly as `q = sqrt(max(r**2 - x**2, 0))`. A contrastive pair is a window
and the same window with an extra solar injection delta taken from a sibling segment of the same
month. No function prints a month, a date, or a real value.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, NamedTuple

import e00732_paths  # noqa: F401

# isort: split
import e00729_sim as sim729
import e00732_core as core
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


@dataclass(frozen=True)
class Transfer:
    """One load transfer on a segment's dense axis.

    Attributes:
        start_tick: The first 5-minute tick of the transferred load.
        ticks: The duration in ticks.
        size: The signed size in points (positive when D8 gains load).
        inside_bsp_a: The partner Primary sits under BSP A, so the BSP meter stays unchanged.
    """

    start_tick: int
    ticks: int
    size: float
    inside_bsp_a: bool

    @property
    def end_tick(self) -> int:
        """The first tick after the transfer."""
        return self.start_tick + self.ticks

    @property
    def start_half_hour(self) -> int:
        """The half-hour that first shows the new level (the half-hour that holds the start edge)."""
        return self.start_tick // core.TICKS_PER_HALF_HOUR

    @property
    def end_half_hour(self) -> int:
        """The half-hour that holds the end edge."""
        return self.end_tick // core.TICKS_PER_HALF_HOUR


@dataclass
class TransferOverlay:
    """A segment with its load transfers applied.

    Attributes:
        seg: The transfer-applied copy of the segment (truth, reading, base demand, and BSP meter).
        q: The reactive floor of the stored segment, `(D + 2, 48)`.
        beta: The segment's BSP coupling.
        transfers: The accepted transfers.
        t: The mean transferred load in each half-hour, in points (a partial half-hour at an edge).
        t_bsp: The part of `t` that moves the BSP meter (transfers to a Primary under another BSP).
        event: 1 on a half-hour that holds a transfer edge.
        size: The signed change of D8's demand at each edge half-hour.
        n_failed: Transfers that could not be placed after `MAX_REDRAWS` redraws.
        rate: The drawn transfers per 30 days.
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


def scored_half_hours(*, seg: sim729.Segment) -> np.ndarray:
    """Return the flat mask of half-hours that a bank scores: base days with a valid truth."""
    on_base = np.repeat(seg.cal_index >= 0, 48)
    return on_base & (seg.valid & np.isfinite(seg.x)).reshape(-1)


def draw_transfers(
    *, seg: sim729.Segment, rate: float, rng: np.random.Generator
) -> tuple[list[Transfer], int]:
    """Draw a segment's load transfers.

    A transfer is accepted only when both its edges fall on scored half-hours, its span lies inside
    the segment's dense axis, and it overlaps no accepted transfer; otherwise it is redrawn, up to
    `MAX_REDRAWS` times.

    Args:
        seg: The segment.
        rate: Transfers per 30 days.
        rng: The generator of this segment alone.

    Returns:
        The accepted transfers and the number that could not be placed.
    """
    n_days = seg.n_dense
    count = int(round(rate * n_days / 30.0))
    scored = np.flatnonzero(scored_half_hours(seg=seg))
    n_ticks = len(seg.dates) * 48 * core.TICKS_PER_HALF_HOUR
    on_scored = np.zeros(len(seg.dates) * 48, dtype=bool)
    on_scored[scored] = True
    accepted: list[Transfer] = []
    failed = 0
    lo_h, hi_h = (np.log(v * 2 * core.TICKS_PER_HALF_HOUR) for v in core.TRANSFER_HOURS)
    lo_s, hi_s = (np.log(v) for v in core.TRANSFER_POINTS)
    for _ in range(count):
        for _attempt in range(core.MAX_REDRAWS):
            if scored.size == 0:
                break
            ticks = int(round(np.exp(rng.uniform(lo_h, hi_h))))
            first = int(rng.choice(scored)) * core.TICKS_PER_HALF_HOUR + int(
                rng.integers(0, core.TICKS_PER_HALF_HOUR)
            )
            cand = Transfer(
                start_tick=first,
                ticks=ticks,
                size=float(np.exp(rng.uniform(lo_s, hi_s))) * (1.0 if rng.uniform() < 0.5 else -1.0),
                inside_bsp_a=bool(rng.uniform() < core.TRANSFER_INSIDE_BSP_A),
            )
            ok_end = cand.end_tick <= n_ticks - 1 and on_scored[cand.end_half_hour]
            overlaps = any(
                cand.start_tick < o.end_tick and o.start_tick < cand.end_tick for o in accepted
            )
            if ok_end and not overlaps:
                accepted.append(cand)
                break
        else:
            failed += 1
            continue
        if scored.size == 0:
            failed += 1
    return accepted, failed


def transfer_arrays(
    *, transfers: Sequence[Transfer], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `t`, `t_bsp`, `event`, and `size` of a segment's transfers on its dense axis.

    The load is built on the 5-minute grid and averaged over each half-hour, so a half-hour that
    holds an edge carries a partial mean.
    """
    n = shape[0] * shape[1]
    ticks = np.zeros(n * core.TICKS_PER_HALF_HOUR)
    ticks_bsp = np.zeros_like(ticks)
    event = np.zeros(n, dtype=np.float32)
    size = np.zeros(n, dtype=np.float32)
    for tr in transfers:
        ticks[tr.start_tick : tr.end_tick] += tr.size
        if not tr.inside_bsp_a:
            ticks_bsp[tr.start_tick : tr.end_tick] += tr.size
        event[tr.start_half_hour] = 1.0
        event[tr.end_half_hour] = 1.0
        size[tr.start_half_hour] += tr.size
        size[tr.end_half_hour] -= tr.size
    t = ticks.reshape(n, core.TICKS_PER_HALF_HOUR).mean(axis=1)
    t_bsp = ticks_bsp.reshape(n, core.TICKS_PER_HALF_HOUR).mean(axis=1)
    out = [a.astype(np.float32).reshape(shape) for a in (t, t_bsp, event, size)]
    return out[0], out[1], out[2], out[3]


def apply_transfers(
    *, seg: sim729.Segment, transfers: Sequence[Transfer], n_failed: int = 0, rate: float = 0.0
) -> TransferOverlay:
    """Return the segment with its transfers applied: `x + T`, the re-folded reading, `base + T`, and the BSP meter.

    Args:
        seg: The stored segment (never edited).
        transfers: The transfers.
        n_failed: Transfers that could not be placed.
        rate: The drawn transfers per 30 days.

    Returns:
        The overlay.
    """
    shape = seg.x.shape
    t, t_bsp, event, size = transfer_arrays(transfers=transfers, shape=shape)
    q = reactive_floor(seg=seg)
    x = (seg.x + t).astype(np.float32)
    beta = float(seg.params[BETA_KEY])
    changed = replace(
        seg,
        base=(seg.base + t).astype(np.float32),
        x=x,
        r=np.where(np.isfinite(seg.r), refold(x=np.nan_to_num(x, nan=0.0), q=q), np.nan).astype(np.float32),
        bsp=(seg.bsp + beta * t_bsp).astype(np.float32),
    )
    return TransferOverlay(
        seg=changed, q=q, beta=beta, transfers=list(transfers), t=t, t_bsp=t_bsp, event=event,
        size=size, n_failed=n_failed, rate=rate,
    )  # fmt: skip


def overlay_seed(*, kind: str, seed: int | None) -> int:
    """Return the first seed-sequence entry of a bank's overlay (training: 732,100 plus the seed)."""
    if kind == "train":
        assert seed is not None
        return core.SEED_TRAIN_OVERLAY + seed
    return {"validation": core.SEED_VALIDATION_OVERLAY, "test": core.SEED_TEST_OVERLAY}[kind]


def overlay_rng(*, kind: str, seed: int | None, index: int) -> np.random.Generator:
    """Return the generator of segment `index` of a bank's transfer overlay."""
    return np.random.default_rng(np.random.SeedSequence([overlay_seed(kind=kind, seed=seed), 1, index]))


def overlay_segments(
    *, segments: Sequence[sim729.Segment], kind: str, seed: int | None = None
) -> list[TransferOverlay]:
    """Return the transfer overlay of a bank: one overlay per segment, from the bank's fixed seed.

    Args:
        segments: The bank's segments.
        kind: "train", "validation", or "test".
        seed: The training seed (None for validation and test).

    Returns:
        The overlays, in the order of `segments`. The overlay of a segment depends only on the
        segment, its position, the kind, and the seed.
    """
    out = []
    for index, seg in enumerate(segments):
        rng = overlay_rng(kind=kind, seed=seed, index=index)
        rate = float(rng.uniform(*core.TRANSFER_RATE_TRAIN)) if kind == "train" else core.TRANSFER_RATE_FIXED
        transfers, failed = draw_transfers(seg=seg, rate=rate, rng=rng)
        out.append(apply_transfers(seg=seg, transfers=transfers, n_failed=failed, rate=rate))
    return out


def overlay_digest(*, overlays: Sequence[TransferOverlay]) -> str:
    """Return the SHA-256 of the overlays' truth, reading, BSP meter, edges, and sizes."""
    h = hashlib.sha256()
    for ov in overlays:
        for arr in (ov.seg.x, ov.seg.r, ov.seg.bsp, ov.event, ov.size):
            h.update(np.ascontiguousarray(np.nan_to_num(arr, nan=-1.0)).tobytes())
    return h.hexdigest()


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
    """

    siblings: dict[tuple[int, int], np.ndarray]
    units: dict[int, np.ndarray]
    windows: dict[str, int]
    unpaired: dict[str, int]
    months: list[str]

    def share_unpaired(self) -> dict[str, float]:
        """Return the share of each month's windows with no sibling, in time order of the month label."""
        return {m: self.unpaired[m] / self.windows[m] for m in sorted(self.windows) if self.windows[m]}


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
    return PairIndex(siblings=siblings, units=units, windows=windows, unpaired=unpaired, months=[s.month for s in segments])


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


def inject(
    *, x: np.ndarray, q: np.ndarray, delta: np.ndarray, bsp: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the injected truth, re-folded reading, and BSP meter: `x - delta`, `sqrt(.)`, `bsp - beta delta`."""
    x_new = (x - delta).astype(np.float32)
    return x_new, refold(x=np.nan_to_num(x_new, nan=0.0), q=q), (bsp - beta * delta).astype(np.float32)


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


def transfer_checks(*, overlays: Sequence[TransferOverlay], originals: Sequence[sim729.Segment]) -> dict[str, Share]:
    """Return the component checks of the transfer overlay.

    Args:
        overlays: The overlays.
        originals: The stored segments they were built from, in the same order.

    Returns:
        For each check, the share of cases that pass and the number of cases: whether a transfer
        changes the truth by its size on every scored half-hour inside it, whether its edges fall on
        scored half-hours, whether the BSP meter stays unchanged for a transfer inside BSP A, and
        whether it moves by beta times the transfer for any other.
    """
    exact = [0, 0]
    edges = [0, 0]
    inside = [0, 0]
    outside = [0, 0]
    for ov, seg in zip(overlays, originals, strict=True):
        scored = scored_half_hours(seg=seg)
        diff = (ov.seg.x - seg.x).reshape(-1)
        bsp_diff = (ov.seg.bsp - seg.bsp).reshape(-1)
        ok_bsp = np.isfinite(seg.bsp.reshape(-1))
        for tr in ov.transfers:
            first = -(-tr.start_tick // core.TICKS_PER_HALF_HOUR)
            last = tr.end_tick // core.TICKS_PER_HALF_HOUR
            span = np.zeros_like(scored)
            span[first:last] = True
            span &= scored
            if span.any():
                exact[1] += 1
                exact[0] += int(np.allclose(diff[span], tr.size, atol=TOLERANCE * max(1.0, abs(tr.size))))
            for hh in (tr.start_half_hour, tr.end_half_hour):
                edges[1] += 1
                edges[0] += int(scored[hh] and ov.event[hh // 48, hh % 48] > 0.0)
            bsp_span = span & ok_bsp
            if bsp_span.any():
                if tr.inside_bsp_a:
                    inside[1] += 1
                    inside[0] += int(np.allclose(bsp_diff[bsp_span], 0.0, atol=TOLERANCE))
                else:
                    outside[1] += 1
                    want = ov.beta * tr.size
                    outside[0] += int(np.allclose(bsp_diff[bsp_span], want, atol=TOLERANCE * max(1.0, abs(want))))
    out = {"transfer_exact": exact, "transfer_edges_scored": edges, "bsp_inside_unchanged": inside, "bsp_outside_moves": outside}
    return {k: Share(v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in out.items()}


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
    chosen = [keys[int(k)] for k in rng.choice(len(keys), size=min(n_pairs, len(keys)), replace=False)] if keys else []
    active = [0, 0]
    equal = [0, 0]
    for i, d in chosen:
        seg = segments[i]
        j = fixed_sibling(index=index, segment=i, day=d)
        if j is None:
            continue
        delta = delta_flat(index=index, sibling=j, amount=amount)
        q = reactive_floor(seg=seg)
        _, r_new, _ = inject(x=seg.x, q=q, delta=delta.reshape(seg.x.shape), bsp=seg.bsp, beta=float(seg.params[BETA_KEY]))
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
    *, overlays: Sequence[TransferOverlay], index: PairIndex | None, n_windows_by_amount: Mapping[float, int] | None
) -> dict[str, tuple[int, int, bool]]:
    """Return the reachability counts (count, bar, passes) of the test overlay."""
    steps = 0
    for ov in overlays:
        scored = scored_half_hours(seg=ov.seg).reshape(ov.event.shape)
        steps += int(((ov.event > 0) & scored & (np.abs(ov.size) >= core.MIN_STEP_POINTS_REACHABLE)).sum())
    out = {"transfer_steps_5_points": (steps, core.MIN_TRANSFER_STEPS_TEST, steps >= core.MIN_TRANSFER_STEPS_TEST)}
    for a, n in (n_windows_by_amount or {}).items():
        out[f"windows_at_{a:g}"] = (n, core.MIN_WINDOWS_PER_AMOUNT, n >= core.MIN_WINDOWS_PER_AMOUNT)
    return out


def sibling_windows(*, segments: Sequence[sim729.Segment], sources: Sequence[str] = ("S2",)) -> int:
    """Return how many base-day windows of the bank have a sibling of the given sources."""
    return len(build_pair_index(segments=segments, sources=sources).siblings)


def unpaired_problems(*, share_unpaired: Mapping[str, float], bar: float = core.MAX_UNPAIRED_SHARE) -> list[str]:
    """Return the months (by position in time order) whose unpaired share exceeds the bar."""
    return [f"month {i}" for i, (_, v) in enumerate(sorted(share_unpaired.items())) if not v <= bar]


Rng = Callable[[], np.random.Generator]

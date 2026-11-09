"""Penalised segmentation of the residuals into events, and pairing of events across neighbours."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from e00653_common import BLOCK, DAY, GRID_LENGTH, N_BLOCKS, band_of

N_LEVELS: Final[int] = 12
MIN_UNITS: Final[float] = 1.0
MAX_UNITS: Final[float] = 20.0
NU: Final[float] = 4.0
"""Degrees of freedom of the Student-t emission."""
PERMANENT_DAYS: Final[int] = 180
RAMP_FRACTION: Final[float] = 0.25
"""Share of a step that both flanks (12 half-hours either side of it) must hold for a ramp."""


@dataclass(frozen=True)
class Params:
    """Penalties and pairing windows (tuned on 2022 and 2023 only)."""

    on_cost: float = 6.0
    off_cost: float = 6.0
    stage_cost: float = 8.0
    stay_cost: float = 0.05
    start_window: int = 12
    """Half-hours within which paired starts must fall (6 hours)."""
    end_window: int = 24
    share_low: float = 0.3
    share_high: float = 1.5
    conservation_max: float = 2.5
    extra_edge_cost: float = 1.0
    routine_rate: int = 4
    """Unpaired short events within 30 days that make a Primary's steps routine."""
    min_confidence: float = 0.0


@dataclass
class Event:
    """One decoded event at one Primary."""

    primary: str
    start: int
    end: int | None
    amp_mw: float
    units: float
    confidence: float
    ramp_fraction: float
    kind: str = "temporary"
    cls: str = "shift_no_partner"
    partners: list[str] = field(default_factory=list)
    share: float | None = None
    partner_amp: dict[str, float] = field(default_factory=dict)

    @property
    def days(self) -> float | None:
        """Length in days, or None if the end lies beyond the record."""
        return None if self.end is None else (self.end - self.start) / DAY

    @property
    def band(self) -> str:
        """Duration band."""
        return band_of(days=self.days)


def level_grid(*, sigma: float) -> np.ndarray:
    """Zero plus 12 log-spaced amplitudes of each sign, in megawatts."""
    a = np.geomspace(MIN_UNITS, MAX_UNITS, N_LEVELS) * sigma
    return np.concatenate([[0.0], a, -a])


def transition_costs(*, p: Params) -> np.ndarray:
    """Cost matrix T[from, to] of the hidden semi-Markov chain's states."""
    s = 1 + 2 * N_LEVELS
    t = np.full((s, s), p.stage_cost)
    t[0, :] = p.on_cost
    t[:, 0] = p.off_cost
    t[0, 0] = 0.0
    np.fill_diagonal(t, 0.0)
    return t


def decode(*, x: np.ndarray, sigma: float, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi path (state index per block) and forward-backward P(shifted) per block."""
    levels = level_grid(sigma=sigma)
    s = len(levels)
    z = (x[:, None] - levels[None, :]) / sigma
    emit = 0.5 * (NU + 1) * np.log1p(z * z / NU)
    emit[~np.isfinite(x)] = 0.0
    emit[:, 1:] += p.stay_cost
    t = transition_costs(p=p)
    n = len(x)
    dp = np.full(s, 1e9)
    dp[0] = 0.0
    dp = np.minimum(dp, 0.0 + p.on_cost + 0)  # any state may start, at the onset price
    dp[0] = 0.0
    dp = dp + emit[0]
    back = np.zeros((n, s), dtype=np.int16)
    for b in range(1, n):
        m = dp[:, None] + t
        back[b] = m.argmin(axis=0)
        dp = m[back[b], np.arange(s)] + emit[b]
    path = np.zeros(n, dtype=np.int16)
    path[-1] = int(dp.argmin())
    for b in range(n - 1, 0, -1):
        path[b - 1] = back[b, path[b]]
    return path, posterior_shifted(emit=emit, t=t)


def posterior_shifted(*, emit: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Forward-backward posterior that a block is in a shifted state."""
    n, s = emit.shape
    trans = np.exp(-t)
    trans /= trans.sum(axis=1, keepdims=True)
    like = np.exp(-(emit - emit.min(axis=1, keepdims=True)))
    alpha = np.zeros((n, s))
    a = np.zeros(s)
    a[0] = 1.0
    a = a * like[0]
    alpha[0] = a / a.sum()
    for b in range(1, n):
        a = (alpha[b - 1] @ trans) * like[b]
        alpha[b] = a / a.sum()
    beta = np.ones((n, s))
    for b in range(n - 2, -1, -1):
        v = trans @ (like[b + 1] * beta[b + 1])
        beta[b] = v / v.sum()
    post = alpha * beta
    post /= post.sum(axis=1, keepdims=True)
    return 1.0 - post[:, 0]


def refine(*, xh: np.ndarray, boundary: int, width: int = 24, reach: int = 18) -> tuple[int, float]:
    """Move a block boundary to the half-hour of the largest step; return it and the edge width."""
    best, best_t = -1.0, boundary
    lo, hi = max(width, boundary - reach), min(len(xh) - width, boundary + reach)
    for tt in range(lo, hi + 1):
        a, b = xh[tt - width : tt], xh[tt : tt + width]
        if np.isfinite(a).sum() < width // 2 or np.isfinite(b).sum() < width // 2:
            continue
        d = abs(np.nanmean(b) - np.nanmean(a))
        if d > best:
            best, best_t = d, tt
    return best_t, ramp_fraction(xh=xh, t=best_t)


def ramp_fraction(*, xh: np.ndarray, t: int) -> float:
    """Smaller of the shares of a step done just before it and still to come just after it."""
    if t < 48 or t + 48 > len(xh):
        return 0.0
    pre = np.nanmedian(xh[t - 48 : t - 18])
    post = np.nanmedian(xh[t + 18 : t + 48])
    d = post - pre
    a, b = np.nanmean(xh[t - 12 : t]), np.nanmean(xh[t : t + 12])
    if not np.isfinite(d) or abs(d) < 1e-9 or not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    return float(max(0.0, min((a - pre) / d, 1 - (b - pre) / d)))


def step_width(*, xh: np.ndarray, t: int, span: int = 48) -> float:
    """Half-hours a step takes to go from 10% to 90% of its size (NaN if it is too small)."""
    lo, hi = max(0, t - span), min(len(xh), t + span)
    seg = xh[lo:hi].copy()
    if np.isfinite(seg).sum() < span:
        return float("nan")
    k = t - lo
    pre = np.nanmedian(seg[: max(1, k - 12)])
    post = np.nanmedian(seg[min(len(seg) - 1, k + 12) :])
    d = post - pre
    if not np.isfinite(d) or d == 0:
        return float("nan")
    sm = np.convolve(np.nan_to_num(seg, nan=pre), np.ones(5) / 5, mode="same")
    frac = (sm - pre) / d
    i10 = np.argmax(frac[max(0, k - 24) :] > 0.1) + max(0, k - 24)
    if not (frac[max(0, k - 24) :] > 0.9).any() or not (frac[max(0, k - 24) :] > 0.1).any():
        return float("nan")
    i90 = np.argmax(frac[max(0, k - 24) :] > 0.9) + max(0, k - 24)
    return float(max(0, i90 - i10))


def extract(
    *, primary: str, path: np.ndarray, post: np.ndarray, x: np.ndarray, xh: np.ndarray, sigma: float
) -> list[Event]:
    """Turn runs of non-zero states into events with half-hour starts and ends."""
    events: list[Event] = []
    nz = path != 0
    b = 0
    while b < len(path):
        if not nz[b]:
            b += 1
            continue
        e = b
        while e + 1 < len(path) and nz[e + 1] and (path[e + 1] > N_LEVELS) == (path[b] > N_LEVELS):
            e += 1
        seg = x[b : e + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            b = e + 1
            continue
        amp = float(np.median(seg))
        start, width = refine(xh=xh, boundary=b * BLOCK)
        open_end = e == len(path) - 1
        end = None if open_end else refine(xh=xh, boundary=(e + 1) * BLOCK)[0]
        conf = float(np.mean(post[b : e + 1]))
        events.append(Event(primary, start, end, amp, abs(amp) / sigma, conf, width))
        b = e + 1
    return events


MERGE_GAP: Final[int] = 3 * DAY
"""Largest gap in half-hours across which two same-sign events at one Primary are joined."""


def merge_fragments(*, events: list[Event]) -> list[Event]:
    """Join same-sign events at one Primary within `MERGE_GAP` that differ by under 2 times."""
    out: list[Event] = []
    for ev in sorted(events, key=lambda e: (e.primary, e.start)):
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.primary == ev.primary
            and prev.end is not None
            and ev.start - prev.end <= MERGE_GAP
            and prev.amp_mw * ev.amp_mw > 0
            and 0.5 <= abs(ev.amp_mw / prev.amp_mw) <= 2.0
        ):
            span_a, span_b = (
                prev.end - prev.start,
                (ev.end if ev.end is not None else prev.end) - ev.start,
            )
            prev.amp_mw = (prev.amp_mw * span_a + ev.amp_mw * max(span_b, 1)) / max(
                span_a + span_b, 1
            )
            prev.units = max(prev.units, ev.units)
            prev.end = ev.end
            prev.confidence = max(prev.confidence, ev.confidence)
        else:
            out.append(ev)
    return out


def classify(  # noqa: PLR0912 - one long decision cascade over event classes; not refactored
    *,
    events: list[Event],
    sigmas: dict[str, float],
    neighbours: dict[str, list[str]],
    p: Params,
    record_end: int = GRID_LENGTH,
    edge_ok: set[tuple[str, str]] | None = None,
) -> list[Event]:
    """Pair events across connected Primaries and assign each event a class."""
    for ev in events:
        ev.cls, ev.partners, ev.share, ev.partner_amp = "shift_no_partner", [], None, {}
        if ev.end is None:
            ev.kind = "permanent" if (record_end - ev.start) / DAY >= PERMANENT_DAYS else "open"
        else:
            ev.kind = "temporary"
        if ev.ramp_fraction >= RAMP_FRACTION:
            ev.cls = "ramp"
    free = [e for e in events if e.cls != "ramp"]
    donors = sorted((e for e in free if e.amp_mw < 0), key=lambda e: e.amp_mw)
    used: set[int] = set()
    for d in donors:
        if id(d) in used:
            continue
        cands = [
            r
            for r in free
            if r.amp_mw > 0
            and id(r) not in used
            and r.primary in neighbours[d.primary]
            and (edge_ok is None or tuple(sorted((d.primary, r.primary))) in edge_ok)
            and abs(r.start - d.start) <= p.start_window
            and (
                d.end is None
                or r.end is None
                or abs(r.end - d.end) <= max(p.end_window, 0.25 * (d.end - d.start))
            )
        ]
        best, best_cost = None, float("inf")
        for k in range(1, min(3, len(cands)) + 1):
            for subset in itertools.combinations(cands, k):
                total = sum(r.amp_mw for r in subset)
                share = total / abs(d.amp_mw)
                if not (p.share_low <= share <= p.share_high):
                    continue
                var = sigmas[d.primary] ** 2 + sum(sigmas[r.primary] ** 2 for r in subset)
                cost = (d.amp_mw + total) ** 2 / (var + (0.5 * abs(d.amp_mw)) ** 2)
                cost += p.extra_edge_cost * (k - 1)
                if cost < best_cost:
                    best, best_cost = subset, cost
        if best is not None and best_cost <= p.conservation_max:
            for r in best:
                used.add(id(r))
                r.cls, r.partners, r.share = "edge_transfer", [d.primary], None
                r.partner_amp = {d.primary: d.amp_mw}
            used.add(id(d))
            d.cls = "edge_transfer"
            d.partners = [r.primary for r in best]
            d.partner_amp = {r.primary: r.amp_mw for r in best}
            d.share = sum(r.amp_mw for r in best) / abs(d.amp_mw)
    for ev in free:
        if ev.cls == "shift_no_partner" and ev.kind == "permanent":
            ev.cls = "permanent_shift"
        if ev.cls == "edge_transfer" and ev.kind == "permanent":
            ev.cls = "permanent_transfer"
    solo = {id(e) for e in free if e.cls == "shift_no_partner"}
    for ev in free:
        if id(ev) in solo and ev.days is not None and ev.days <= 3:
            near = [
                o
                for o in free
                if o.primary == ev.primary
                and id(o) in solo
                and o.days is not None
                and o.days <= 3
                and abs(o.start - ev.start) <= 30 * DAY
            ]
            if len(near) >= p.routine_rate:
                ev.cls = "routine"
    return events


def segment_primary(
    *, name: str, y: np.ndarray, xh: np.ndarray, slow: np.ndarray, sigma: float, p: Params
) -> list[Event]:
    """Decode one Primary's block series (slow level already removed in `xh`)."""
    x = y - slow
    valid = np.flatnonzero(np.isfinite(x))
    if len(valid) == 0:
        return []
    x = x[: valid[-1] + 1]
    path, post = decode(x=x, sigma=sigma, p=p)
    ev = merge_fragments(
        events=extract(primary=name, path=path, post=post, x=x, xh=xh, sigma=sigma)
    )
    return [e for e in ev if e.confidence >= p.min_confidence]


assert N_BLOCKS * BLOCK == GRID_LENGTH

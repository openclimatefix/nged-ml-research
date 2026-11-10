"""E1-F: the event table rebuilt by choosing, with exact cost changes, which steps are edge flows.

Experiment 653's table (E1) lists steps at each Primary, every one standing alone. This module
asks, for every pair of opposite-sign steps at two connected Primaries that start within a few
hours of each other, whether one edge flow explains both better than two steps that each stand
alone. A step may be a step of E1 or a step proposed by its partner, which is how either end of a
transfer lifts the other.

The cost of a state is the change cost of each segment plus the misfit of the data to the
segments. A step that stands alone costs `c_edge + delta`; an edge flow costs `c_edge` however
many ends it serves, plus `extra` for each recipient beyond the first. The misfit of an edge flow
is half the squared distance, in noise units, of the donor's step and the recipients' summed step
from the best common amplitude `a` and share `rho` (donor `-a`, recipients `+rho * a`), plus half
the squared normalised distance of `log rho` from a normal prior. Amplitudes are closed form, and
nothing shrinks them. A proposed step that is not merged is dropped, and costs the evidence the
data hold for it, half its squared size in noise units.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Final

import numpy as np
from e00653_segment import Event
from e00670_search import UNPAIRED, step_size

RHO_GRID: Final[np.ndarray] = np.geomspace(0.3, 2.0, 24)
LOG_RHO_MEAN: Final[float] = float(np.log(0.8))
LOG_RHO_SD: Final[float] = 0.5
LAG_LO: Final[int] = -6
"""Earliest recipient start relative to the donor's, in half-hours (the start error)."""
LAG_HI: Final[int] = 42
"""Latest recipient start relative to the donor's (36 half-hours plus the start error)."""
MATCH_HH: Final[int] = 12
"""Two steps of one Primary and one sign this close are one step."""
END_HH: Final[int] = 48
"""Ends of the two steps of a pair differ by at most this, or one end is a proposal."""
MIN_UNITS: Final[float] = 2.0
MAX_RECIPIENTS: Final[int] = 3


@dataclass(frozen=True)
class FlowCosts:
    """Change costs, tuned on validation worlds from a grid fixed in the plan."""

    c_edge: float
    delta: float
    extra: float = 1.0


@dataclass
class Step:
    """One candidate step: an E1 event or a step proposed by a partner."""

    primary: str
    start: int
    end: int | None
    amp: float
    sigma: float
    event: Event | None = None
    """The E1 event this step is, or None for a proposal."""

    @property
    def z(self) -> float:
        """Size in noise units, signed."""
        return self.amp / self.sigma


def best_step(
    *, xh: np.ndarray, anchor: Step, lo: int, hi: int, sign: int, sigma: float
) -> tuple[int, float] | None:
    """Start in `[anchor.start + lo, anchor.start + hi]` with the largest step of `sign`."""
    best: tuple[int, float] | None = None
    for s in range(anchor.start + lo, anchor.start + hi + 1):
        if s < 0 or s >= len(xh):
            continue
        d = step_size(xh=xh, start=s, end=anchor.end)
        if (
            np.isfinite(d)
            and sign * d >= MIN_UNITS * sigma
            and (best is None or sign * d > sign * best[1])
        ):
            best = (s, d)
    return best


def candidates(
    *,
    events: list[Event],
    xh: dict[str, np.ndarray],
    sigma: dict[str, float],
    neighbours: dict[str, list[str]],
) -> list[Step]:
    """E1's steps of 2 noise units or more, plus the opposite step each proposes at a neighbour."""
    seeds: list[Step] = []
    for e in events:
        sg = sigma.get(e.primary, float("nan"))
        if e.primary not in xh or not np.isfinite(sg) or e.units < MIN_UNITS:
            continue
        if e.cls not in (*UNPAIRED, "routine", "edge_transfer", "permanent_transfer"):
            continue
        amp = step_size(xh=xh[e.primary], start=e.start, end=e.end)
        seeds.append(Step(e.primary, e.start, e.end, amp if np.isfinite(amp) else e.amp_mw, sg, e))
    out = list(seeds)
    for s in seeds:
        sign = -1 if s.amp > 0 else 1
        lo, hi = (LAG_LO, LAG_HI) if s.amp < 0 else (-LAG_HI, -LAG_LO)
        for n in neighbours.get(s.primary, []):
            sg = sigma.get(n, float("nan"))
            if n not in xh or not np.isfinite(sg):
                continue
            hit = best_step(xh=xh[n], anchor=s, lo=lo, hi=hi, sign=sign, sigma=sg)
            if hit is None:
                continue
            if any(
                c.primary == n and c.amp * hit[1] > 0 and abs(c.start - hit[0]) <= MATCH_HH
                for c in out
            ):
                continue
            out.append(Step(n, hit[0], s.end, hit[1], sg))
    return out


def misfit(*, donor_z: float, recip_z: float, sd: float, sr: float) -> float:
    """Half the squared distance, in noise units, from the best common flow, plus the rho prior.

    Args:
        donor_z: the donor's step in megawatts (negative).
        recip_z: the recipients' summed step in megawatts (positive).
        sd: the donor's noise unit in megawatts.
        sr: the recipients' combined noise unit in megawatts.

    Returns:
        The smallest misfit over the share grid.
    """
    d = -donor_z
    best = float("inf")
    for rho in RHO_GRID:
        a = (d / sd**2 + rho * recip_z / sr**2) / (1 / sd**2 + rho**2 / sr**2)
        fit = 0.5 * ((d - a) ** 2 / sd**2 + (recip_z - rho * a) ** 2 / sr**2)
        prior = 0.5 * ((np.log(rho) - LOG_RHO_MEAN) / LOG_RHO_SD) ** 2
        best = min(best, fit + prior)
    return float(best)


def compatible(*, d: Step, r: Step) -> bool:
    """Start lag within the window, and ends close when both are E1 events."""
    if not LAG_LO <= r.start - d.start <= LAG_HI:
        return False
    if d.event is not None and r.event is not None:
        if (d.end is None) != (r.end is None):
            return False
        if d.end is not None and r.end is not None and abs(d.end - r.end) > END_HH:
            return False
    return True


def gain(*, d: Step, rs: tuple[Step, ...], costs: FlowCosts) -> float:
    """Cost saved by serving donor `d` and recipients `rs` with one edge flow."""
    stand = costs.c_edge + costs.delta
    members = (d, *rs)
    # Without the flow, each E1 step stands alone and each proposed step is left unexplained.
    base = sum(stand if m.event is not None else 0.5 * m.z**2 for m in members)
    joint = costs.c_edge + costs.extra * (len(rs) - 1)
    sr = float(np.sqrt(sum(r.sigma**2 for r in rs)))
    recip = sum(r.amp for r in rs)
    cost = joint + misfit(donor_z=d.amp, recip_z=recip, sd=d.sigma, sr=sr)
    return base - cost


def merge(
    *, cands: list[Step], neighbours: dict[str, list[str]], costs: FlowCosts
) -> list[tuple[Step, tuple[Step, ...], float]]:
    """Greedy: accept the pair or group with the largest positive gain, until none is left."""
    downs = [c for c in cands if c.amp < 0]
    ups = [c for c in cands if c.amp > 0]
    options: list[tuple[float, int, Step, tuple[Step, ...]]] = []
    for i, d in enumerate(downs):
        near = [
            r for r in ups if r.primary in neighbours.get(d.primary, []) and compatible(d=d, r=r)
        ]
        for k in range(1, min(MAX_RECIPIENTS, len(near)) + 1):
            for sub in itertools.combinations(near, k):
                if len({r.primary for r in sub}) < k:
                    continue
                g = gain(d=d, rs=sub, costs=costs)
                if g > 0:
                    options.append((g, i, d, sub))
    options.sort(key=lambda o: -o[0])
    used: set[int] = set()
    out = []
    for g, _, d, sub in options:
        ids = {id(d), *(id(r) for r in sub)}
        if ids & used:
            continue
        used |= ids
        out.append((d, sub, g))
    return out


def reconstruct(
    *,
    events: list[Event],
    xh: dict[str, np.ndarray],
    sigma: dict[str, float],
    neighbours: dict[str, list[str]],
    costs: FlowCosts,
    cands: list[Step] | None = None,
) -> list[Event]:
    """Return E1's events with the accepted edge flows marked as transfers (a copy)."""
    out = copy.deepcopy(events)
    clone = {id(a): b for a, b in zip(events, out, strict=True)}
    for e in out:
        if e.cls in ("edge_transfer", "permanent_transfer"):
            e.cls, e.partners, e.partner_amp, e.share = "shift_no_partner", [], {}, None
    cands = cands or candidates(events=events, xh=xh, sigma=sigma, neighbours=neighbours)
    added: list[Event] = []

    def event_of(s: Step, kind: str) -> Event:
        if s.event is not None:
            return clone[id(s.event)]
        ev = Event(
            primary=s.primary,
            start=s.start,
            end=s.end,
            amp_mw=s.amp,
            units=abs(s.amp) / s.sigma,
            confidence=0.5,
            ramp_fraction=0.0,
            kind=kind,
        )
        added.append(ev)
        return ev

    for d, rs, _ in merge(cands=cands, neighbours=neighbours, costs=costs):
        kind = (d.event or next(r.event for r in rs if r.event is not None)).kind
        de = event_of(d, kind)
        res = [event_of(r, kind) for r in rs]
        cls = "permanent_transfer" if kind == "permanent" else "edge_transfer"
        de.cls, de.partners = cls, [r.primary for r in rs]
        de.partner_amp = {r.primary: r.amp for r in rs}
        for re_ in res:
            re_.cls, re_.partners, re_.partner_amp = cls, [d.primary], {d.primary: d.amp}
    return out + added

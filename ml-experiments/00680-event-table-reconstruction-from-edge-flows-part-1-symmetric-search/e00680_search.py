"""E1-J-sym: experiment 670's recipient search, plus a search for the donor of each unpaired rise.

Experiment 670's search starts at a large downward step (the donor) and looks for a rise at a
neighbour. Experiment 675 found that the donor is the bottleneck: most recipient megawatts that
E1-J located belong to transfers whose donor E1-J did not find. This module runs E1-J first, then
starts from each rise of `recipient_units` noise units or more that is still unpaired and looks
at every connected Primary for a fall of `donor_units` noise units or more that starts 0 to 36
half-hours before the rise (with the same start error as experiment 670), accepting the one donor
whose step best conserves the rise under experiment 653's loose conservation cost.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Final

import numpy as np
from e00653_segment import Event
from e00670_search import UNPAIRED, SearchParams, search_recipients, step_size

__all__ = ["SymParams", "search_both_ends", "search_donors"]


@dataclass(frozen=True)
class SymParams:
    """Thresholds of the donor search, fixed before any score is read."""

    recipient_units: float = 4.0
    """Smallest unpaired rise that starts a donor search (E1-J uses 4 for its donor)."""
    donor_units: float = 2.0
    """Smallest fall accepted as a donor (E1-J uses 2 for its recipient)."""
    lag_lo: int = -42
    """Earliest donor start relative to the rise's, in half-hours (36 plus the start error)."""
    lag_hi: int = 6
    """Latest donor start relative to the rise's (the start error)."""
    share_low: float = 0.3
    share_high: float = 2.0
    kappa: float = 0.5
    conservation_max: float = 2.5


MATCH_HH: Final[int] = 12
"""An existing event this close to a found step is reused, as in experiment 670."""


def best_donor_step(*, xh: np.ndarray, rise: Event, params: SymParams) -> tuple[int, float] | None:
    """Start in the search window with the largest downward step, and the size of that step."""
    best: tuple[int, float] | None = None
    for s in range(rise.start + params.lag_lo, rise.start + params.lag_hi + 1):
        if s < 0 or s >= len(xh):
            continue
        d = step_size(xh=xh, start=s, end=rise.end)
        if np.isfinite(d) and (best is None or d < best[1]):
            best = (s, d)
    return best


def search_donors(
    *,
    events: list[Event],
    xh: dict[str, np.ndarray],
    sigma: dict[str, float],
    neighbours: dict[str, list[str]],
    params: SymParams | None = None,
) -> list[Event]:
    """Return a copy of `events` in which unpaired rises have the donor the search finds."""
    params = params or SymParams()
    out = copy.deepcopy(events)
    used: set[int] = set()
    rises = sorted(
        (
            e
            for e in out
            if e.amp_mw > 0 and e.units >= params.recipient_units and e.cls in UNPAIRED
        ),
        key=lambda e: -e.amp_mw,
    )
    added: list[Event] = []
    for r in rises:
        if r.primary not in xh:
            continue
        r_step = step_size(xh=xh[r.primary], start=r.start, end=r.end)
        if not np.isfinite(r_step) or r_step <= 0:
            r_step = r.amp_mw
        best: tuple[str, int, float] | None = None
        best_cost = float("inf")
        for n in neighbours.get(r.primary, []):
            if n not in xh or not np.isfinite(sigma.get(n, float("nan"))):
                continue
            hit = best_donor_step(xh=xh[n], rise=r, params=params)
            if hit is None or -hit[1] < params.donor_units * sigma[n]:
                continue
            taken = any(
                e.primary == n
                and abs(e.start - hit[0]) <= MATCH_HH
                and (id(e) in used or e.cls.endswith("transfer"))
                for e in out + added
            )
            share = r_step / abs(hit[1])
            if taken or not params.share_low <= share <= params.share_high:
                continue
            var = sigma[r.primary] ** 2 + sigma[n] ** 2
            cost = (hit[1] + r_step) ** 2 / (var + (params.kappa * abs(r_step)) ** 2)
            if cost < best_cost:
                best, best_cost = (n, hit[0], hit[1]), cost
        if best is None or best_cost > params.conservation_max:
            continue
        n, s, d_step = best
        reuse = min(
            (
                e
                for e in out + added
                if e.primary == n
                and e.amp_mw < 0
                and id(e) not in used
                and abs(e.start - s) <= MATCH_HH
                and e.cls in (*UNPAIRED, "routine")
            ),
            key=lambda e: abs(e.start - s),
            default=None,
        )
        if reuse is None:
            reuse = Event(
                primary=n,
                start=s,
                end=r.end,
                amp_mw=d_step,
                units=-d_step / sigma[n],
                confidence=0.5,
                ramp_fraction=0.0,
                kind=r.kind,
            )
            added.append(reuse)
        used.add(id(reuse))
        cls = "permanent_transfer" if r.kind == "permanent" else "edge_transfer"
        reuse.cls, reuse.partners, reuse.partner_amp = cls, [r.primary], {r.primary: r_step}
        reuse.share = r_step / abs(d_step)
        r.cls, r.partners, r.partner_amp = cls, [n], {n: reuse.amp_mw}
    return out + added


def search_both_ends(
    *,
    events: list[Event],
    xh: dict[str, np.ndarray],
    sigma: dict[str, float],
    neighbours: dict[str, list[str]],
) -> list[Event]:
    """E1-J (donor to recipient), then the donor search from each rise still unpaired."""
    j = search_recipients(
        events=events, xh=xh, sigma=sigma, neighbours=neighbours, params=SearchParams()
    )
    return search_donors(events=j, xh=xh, sigma=sigma, neighbours=neighbours)

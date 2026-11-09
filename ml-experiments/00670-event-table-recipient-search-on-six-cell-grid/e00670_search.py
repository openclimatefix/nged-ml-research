"""E1-J: experiment 653's event table plus a search for the recipient of each unpaired donor step.

Experiment 653 pairs a donor step with recipient steps only when its own segmentation has found
both ends, which needs every end to clear the full detection penalty. This module keeps that
segmentation and adds one conditional search. For each donor step of `donor_units` noise units or
more that experiment 653 left unpaired, it looks at every connected Primary for a step of the
opposite sign of `recipient_units` noise units or more that starts within `lag_lo` to `lag_hi`
half-hours of the donor's start, and it accepts a set of up to three recipients when their
combined step conserves the donor's step under experiment 653's loose conservation cost, with a
recipient share between `share_low` and `share_high`.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Final

import numpy as np
from e00653_common import DAY
from e00653_segment import Event

WINDOW_DAYS: Final[int] = 3
"""Days before a step, and at most days after it, over which the step's size is measured."""
MIN_AFTER: Final[int] = 12
"""Half-hours with a reading needed after a step to measure it."""
MIN_BEFORE: Final[int] = 48
"""Half-hours with a reading needed before a step to measure it."""
SKIP_RAMP: Final[int] = 4
"""Half-hours after a step's start that the size measurement leaves out (the longest ramp)."""
UNPAIRED: Final[tuple[str, str]] = ("shift_no_partner", "permanent_shift")


@dataclass(frozen=True)
class SearchParams:
    """Thresholds of the recipient search, fixed before any score is read."""

    donor_units: float = 4.0
    recipient_units: float = 2.0
    lag_lo: int = -6
    """Earliest recipient start relative to the donor's, in half-hours (allows start error)."""
    lag_hi: int = 42
    """Latest recipient start relative to the donor's (36 half-hours plus the start error)."""
    share_low: float = 0.3
    share_high: float = 2.0
    kappa: float = 0.5
    conservation_max: float = 2.5
    extra_edge_cost: float = 1.0
    max_recipients: int = 3


def step_size(*, xh: np.ndarray, start: int, end: int | None) -> float:
    """Median level after a step (to its end or `WINDOW_DAYS`) minus the median before it."""
    stop = start + WINDOW_DAYS * DAY if end is None else min(start + WINDOW_DAYS * DAY, end)
    after = xh[start + SKIP_RAMP : stop]
    before = xh[max(0, start - WINDOW_DAYS * DAY) : start]
    if np.isfinite(after).sum() < MIN_AFTER or np.isfinite(before).sum() < MIN_BEFORE:
        return float("nan")
    return float(np.nanmedian(after) - np.nanmedian(before))


def best_recipient_step(
    *, xh: np.ndarray, donor: Event, params: SearchParams
) -> tuple[int, float] | None:
    """Start in the search window with the largest upward step, and the size of that step."""
    best: tuple[int, float] | None = None
    for s in range(donor.start + params.lag_lo, donor.start + params.lag_hi + 1):
        if s < 0 or s >= len(xh):
            continue
        d = step_size(xh=xh, start=s, end=donor.end)
        if np.isfinite(d) and (best is None or d > best[1]):
            best = (s, d)
    return best


def search_recipients(  # noqa: PLR0912 - one decision cascade, as experiment 653's classify
    *,
    events: list[Event],
    xh: dict[str, np.ndarray],
    sigma: dict[str, float],
    neighbours: dict[str, list[str]],
    params: SearchParams | None = None,
) -> list[Event]:
    """Return a copy of `events` in which unpaired donors have the recipients the search finds.

    A found recipient reuses an event of experiment 653 at that Primary when one starts within 12
    half-hours of the found start with a positive sign, and is added as a new event otherwise.
    """
    params = params or SearchParams()
    out = copy.deepcopy(events)
    used: set[int] = set()
    donors = sorted(
        (e for e in out if e.amp_mw < 0 and e.units >= params.donor_units and e.cls in UNPAIRED),
        key=lambda e: e.amp_mw,
    )
    added: list[Event] = []
    for d in donors:
        if d.primary not in xh:
            continue
        d_step = step_size(xh=xh[d.primary], start=d.start, end=d.end)
        if not np.isfinite(d_step) or d_step >= 0:
            d_step = d.amp_mw
        cands: list[tuple[str, int, float]] = []
        for n in neighbours.get(d.primary, []):
            if n not in xh or not np.isfinite(sigma.get(n, float("nan"))):
                continue
            hit = best_recipient_step(xh=xh[n], donor=d, params=params)
            taken = hit is not None and any(
                e.primary == n
                and abs(e.start - hit[0]) <= 12
                and (id(e) in used or e.cls.endswith("transfer"))
                for e in out + added
            )
            if hit is not None and not taken and hit[1] >= params.recipient_units * sigma[n]:
                cands.append((n, hit[0], hit[1]))
        best, best_cost = None, float("inf")
        for k in range(1, min(params.max_recipients, len(cands)) + 1):
            for subset in itertools.combinations(cands, k):
                total = sum(c[2] for c in subset)
                share = total / abs(d_step)
                if not params.share_low <= share <= params.share_high:
                    continue
                var = sigma[d.primary] ** 2 + sum(sigma[c[0]] ** 2 for c in subset)
                cost = (d_step + total) ** 2 / (var + (params.kappa * abs(d_step)) ** 2)
                cost += params.extra_edge_cost * (k - 1)
                if cost < best_cost:
                    best, best_cost = subset, cost
        if best is None or best_cost > params.conservation_max:
            continue
        permanent = d.kind == "permanent"
        cls = "permanent_transfer" if permanent else "edge_transfer"
        partner_amp = {}
        for n, s, amp in best:
            reuse = min(
                (
                    e
                    for e in out + added
                    if e.primary == n
                    and e.amp_mw > 0
                    and id(e) not in used
                    and abs(e.start - s) <= 12
                    and e.cls in (*UNPAIRED, "routine")
                ),
                key=lambda e: abs(e.start - s),
                default=None,
            )
            if reuse is None:
                reuse = Event(
                    primary=n,
                    start=s,
                    end=d.end,
                    amp_mw=amp,
                    units=amp / sigma[n],
                    confidence=0.5,
                    ramp_fraction=0.0,
                    kind=d.kind,
                )
                added.append(reuse)
            used.add(id(reuse))
            reuse.cls, reuse.partners, reuse.partner_amp = cls, [d.primary], {d.primary: d_step}
            partner_amp[n] = reuse.amp_mw
        d.cls, d.partners, d.partner_amp = cls, [c[0] for c in best], partner_amp
        d.share = sum(c[2] for c in best) / abs(d_step)
    return out + added

"""A plain penalised changepoint search (PELT) per Primary, the second benchmark of experiment 734.

[Killick, Fearnhead, and Eckley (2012)](https://doi.org/10.1080/01621459.2012.737745) give the
algorithm: exact optimal partitioning with pruning, here with a Gaussian mean-change cost, which is
the sum of squared deviations of a segment from its own mean.
"""

from __future__ import annotations

import itertools
from typing import Final

import numpy as np
from e00653_segment import Event, refine  # ty: ignore[unresolved-import]

BLOCK: Final[int] = 12
"""Half-hours per 6-hour block."""
PENALTY_GRID: Final[tuple[float, ...]] = tuple(float(v) for v in np.geomspace(2.0, 200.0, 12))
"""Twelve penalties, geometrically spaced, in units of the noise variance."""
EVENT_UNITS: Final[float] = 1.0
"""A segment mean at least this many noise units from zero belongs to an event."""


def _segment_cost(*, c1: np.ndarray, c2: np.ndarray, taus: np.ndarray, t: int) -> np.ndarray:
    """Sum of squared deviations from the mean of each segment `(tau, t]` (cumulative sums)."""
    n = t - taus
    s1 = c1[t] - c1[taus]
    return (c2[t] - c2[taus]) - s1 * s1 / n


def pelt(*, y: np.ndarray, penalty: float, min_size: int = 1) -> list[int]:
    """Exact penalised changepoints of the finite values of `y`.

    NaN values are skipped, so a segment may span a gap, and the returned indices refer to `y`.

    Args:
        y: The series, with NaN where missing.
        penalty: The penalty per changepoint, in the squared unit of `y`.
        min_size: The fewest finite values in a segment.

    Returns:
        The indices into `y` of the first value of every segment after the first, ascending.
    """
    keep = np.flatnonzero(np.isfinite(y))
    v = y[keep]
    n = len(v)
    if n < 2 * min_size:
        return []
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    candidates = np.array([0])
    for t in range(1, n + 1):
        usable = candidates[t - candidates >= min_size]
        if len(usable) == 0:
            candidates = np.append(candidates, t)
            continue
        cost = _segment_cost(c1=c1, c2=c2, taus=usable, t=t)
        total = best[usable] + cost + penalty
        k = int(np.argmin(total))
        best[t] = total[k]
        last[t] = int(usable[k])
        pruned = usable[best[usable] + cost <= best[t]]
        waiting = candidates[t - candidates < min_size]
        candidates = np.concatenate([pruned, waiting, [t]]).astype(int)
    cuts: list[int] = []
    t = n
    while t > 0:
        t = int(last[t])
        if t > 0:
            cuts.append(int(keep[t]))
    return sorted(cuts)


def pelt_events(
    *,
    site: str,
    y: np.ndarray,
    slow: np.ndarray,
    sigma: float,
    xh: np.ndarray,
    penalty_units: float,
    record_end: int,
) -> list[Event]:
    """Events of one Primary from PELT segments of the block residual in noise units.

    An event is a maximal run of consecutive segments whose means all lie at least 1 noise unit
    from zero with one sign. Its amplitude is the length-weighted mean of the segment means times
    `sigma`, and its end is the first segment back within 1 unit, or None if the run reaches the
    last segment. Every event has confidence 1.0.

    Args:
        site: The Primary.
        y: Block means of the input in megawatts, NaN where missing.
        slow: The slow level on the block grid in megawatts.
        sigma: The noise unit of the input in megawatts.
        xh: The half-hourly input minus the slow level, in megawatts.
        penalty_units: The PELT penalty in units of the noise variance.
        record_end: One past the last half-hour with a reading at any Primary.

    Returns:
        The events.
    """
    z = (y - slow) / sigma
    cuts = pelt(y=z, penalty=penalty_units)
    keep = np.flatnonzero(np.isfinite(z))
    if len(keep) == 0:
        return []
    bounds = [int(keep[0]), *cuts, int(keep[-1]) + 1]
    segs: list[tuple[int, int, float, int]] = []
    for a, b in itertools.pairwise(bounds):
        vals = z[a:b][np.isfinite(z[a:b])]
        if len(vals):
            segs.append((a, b, float(vals.mean()), len(vals)))
    events: list[Event] = []
    i = 0
    while i < len(segs):
        mean = segs[i][2]
        if abs(mean) < EVENT_UNITS:
            i += 1
            continue
        j = i
        while (
            j + 1 < len(segs)
            and abs(segs[j + 1][2]) >= EVENT_UNITS
            and np.sign(segs[j + 1][2]) == np.sign(mean)
        ):
            j += 1
        weights = np.array([s[3] for s in segs[i : j + 1]], dtype=float)
        amp = float(np.average([s[2] for s in segs[i : j + 1]], weights=weights)) * sigma
        start, width = refine(xh=xh, boundary=segs[i][0] * BLOCK)
        last_block = segs[j][1]
        reaches_end = j == len(segs) - 1 or last_block * BLOCK >= record_end
        end = None if reaches_end else refine(xh=xh, boundary=last_block * BLOCK)[0]
        events.append(Event(site, start, end, amp, abs(amp) / sigma, 1.0, width))
        i = j + 1
    return events

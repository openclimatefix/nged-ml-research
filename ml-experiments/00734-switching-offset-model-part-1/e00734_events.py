"""Events and grades of the offset model (experiment 734).

An event table is read from the Viterbi path. An offset event (o-event) is a maximal run of blocks
with a nonzero offset of one sign. A permanent-level event (l-event) starts at each jump of the
permanent level. Both kinds use experiment 653's `Event` dataclass, so experiment 653's `match`
scores them unchanged. The grade of an event is held in `Event.kind`.
"""

from __future__ import annotations

from typing import Final

import e00734_hmm as hmm
import numpy as np
from e00653_core import match  # ty: ignore[unresolved-import]
from e00653_inject import Truth  # ty: ignore[unresolved-import]
from e00653_segment import Event, refine  # ty: ignore[unresolved-import]

BLOCK: Final[int] = 12
"""Half-hours per 6-hour block."""
GRADES: Final[tuple[str, ...]] = (
    "mirrored and returned",
    "mirrored, not yet returned",
    "returned, no visible partner",
    "neither",
)
QUANTILES: Final[tuple[float, ...]] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
LEVEL_CLS: Final[str] = "level_change"
SPLIT_LEVELS: Final[int] = 2
"""A settled run of one sign splits where the offset level changes by this many levels or more."""
L_RETURN_UNITS: Final[int] = 2
"""An l-event ends when the level is back within this many noise units of the level before it."""
MIN_UNITS_OF_RATE: Final[float] = 2.0
"""Threshold of the second called rate, in multiples of experiment 653's noise unit."""


def block_quantiles(*, post: hmm.Posterior, sigma: float) -> np.ndarray:
    """Quantiles of the switched megawatts of every block, read off the 41-level grid.

    Each quantile is the first level whose cumulative probability reaches it (no interpolation).

    Args:
        post: The posterior of the Primary.
        sigma: The noise unit of the input in megawatts.

    Returns:
        An array of shape `(n_blocks, 7)` in megawatts, in the order of `QUANTILES`.
    """
    cum = np.cumsum(post.o_marg.astype(np.float64), axis=1)
    levels = np.array(hmm.FULL_GRID.o_levels, dtype=float)
    out = np.zeros((cum.shape[0], len(QUANTILES)))
    for j, q in enumerate(QUANTILES):
        out[:, j] = levels[np.argmax(cum >= q - 1e-9, axis=1)] * sigma
    return out


def _o_runs(*, o_path: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive block ranges of the o-events on an offset path."""
    m = hmm.move_flags(o_path=o_path)
    runs: list[tuple[int, int]] = []
    b = 0
    n = len(o_path)
    while b < n:
        if o_path[b] == 0:
            b += 1
            continue
        sign = np.sign(o_path[b])
        start = b
        e = b
        while e + 1 < n and o_path[e + 1] != 0 and np.sign(o_path[e + 1]) == sign:
            nxt = e + 1
            big = abs(int(o_path[nxt]) - int(o_path[e])) >= SPLIT_LEVELS
            if big and m[e] == 0:
                runs.append((start, e))
                start = nxt
            e = nxt
        runs.append((start, e))
        b = e + 1
    return runs


def _end_of(*, last_block: int, n_blocks: int, xh: np.ndarray, record_end: int) -> int | None:
    """Half-hour end of a run ending at `last_block`, or None if it reaches the record end."""
    if (last_block + 1) * BLOCK >= min(record_end, n_blocks * BLOCK):
        return None
    return refine(xh=xh, boundary=(last_block + 1) * BLOCK)[0]


def extract(
    *,
    site: str,
    o_path: np.ndarray,
    l_path: np.ndarray,
    post: hmm.Posterior,
    xh: np.ndarray,
    sigma: float,
    record_end: int,
) -> list[Event]:
    """Read the o-events and l-events of one Primary from its Viterbi path.

    An o-event is a maximal run of nonzero offset of one sign, split where the path's offset level
    changes by 2 or more levels after at least one settled block. A run of consecutive moves in one
    direction never splits, so a cascade, or the partial first block of a mid-block step, stays one
    event. Its amplitude is the mean offset over the run times `sigma`, and its confidence is the
    mean posterior probability that the offset is nonzero. An l-event starts at each jump of the
    permanent level and ends at the next jump that brings the level back within 2 noise units of
    the level before the start, else it stays open. Start and end are refined to the half-hour
    with experiment 653's `refine`.

    Args:
        site: The Primary.
        o_path: The offset level path in noise units, `(n_blocks,)`.
        l_path: The permanent level path in noise units, `(n_blocks,)`.
        post: The posterior of the Primary.
        xh: The half-hourly input minus the slow level, in megawatts.
        sigma: The noise unit of the input in megawatts.
        record_end: One past the last half-hour with a reading at any Primary.

    Returns:
        The events, o-events first.
    """
    n = len(o_path)
    zero = hmm.FULL_GRID.zero
    events: list[Event] = []
    for first, last in _o_runs(o_path=o_path):
        amp = float(np.mean(o_path[first : last + 1])) * sigma
        start, width = refine(xh=xh, boundary=first * BLOCK)
        end = _end_of(last_block=last, n_blocks=n, xh=xh, record_end=record_end)
        conf = float(np.mean(1.0 - post.o_marg[first : last + 1, zero]))
        events.append(Event(site, start, end, amp, abs(amp) / sigma, conf, width))
    jumps = np.flatnonzero(np.diff(l_path) != 0) + 1
    for k, t in enumerate(jumps):
        before = int(l_path[t - 1])
        jump = int(l_path[t]) - before
        stop_block = None
        for t2 in jumps[k + 1 :]:
            if abs(int(l_path[t2]) - before) <= L_RETURN_UNITS:
                stop_block = int(t2)
                break
        start, width = refine(xh=xh, boundary=int(t) * BLOCK)
        end = None if stop_block is None else refine(xh=xh, boundary=stop_block * BLOCK)[0]
        conf = float(np.max(post.l_jump[max(0, int(t) - 1) : int(t) + 2]))
        amp = jump * sigma
        events.append(
            Event(
                site,
                start,
                end,
                amp,
                abs(amp) / sigma,
                conf,
                width,
                kind="permanent" if end is None else "temporary",
                cls=LEVEL_CLS,
            )
        )
    return events


def grade(
    *, events: list[Event], neighbours: dict[str, tuple[str, ...]], window_blocks: int = 3
) -> list[Event]:
    """Grade events in place, given the list of called events of all Primaries.

    An o-event is mirrored when a neighbour has an opposite-sign o-event in the list whose start
    lies within `window_blocks` blocks. A mirrored event gets `cls = "edge_transfer"` and its
    partners and their amplitudes. The grade goes into `Event.kind`. An l-event is always
    "neither" and is never a partner.

    Args:
        events: The called events of every Primary.
        neighbours: The neighbours of every Primary.
        window_blocks: The width of the mirror window in blocks.

    Returns:
        The same list, with grades set.
    """
    reach = window_blocks * BLOCK
    for e in events:
        if e.cls == LEVEL_CLS:
            e.kind = "neither"
            continue
        partners = [
            n
            for n in events
            if n.cls != LEVEL_CLS
            and n.primary in neighbours.get(e.primary, ())
            and np.sign(n.amp_mw) == -np.sign(e.amp_mw)
            and abs(n.start - e.start) <= reach
        ]
        returned = e.end is not None
        if partners:
            e.cls = "edge_transfer"
            e.partners = sorted({n.primary for n in partners})
            e.partner_amp = {n.primary: n.amp_mw for n in partners}
            e.kind = GRADES[0] if returned else GRADES[1]
        else:
            e.kind = GRADES[2] if returned else GRADES[3]
    return events


def called(*, events: list[Event], tau: float) -> list[Event]:
    """The events whose confidence reaches `tau`."""
    return [e for e in events if e.confidence >= tau]


def match_called(
    *, truth: list[Truth], events: list[Event], tau: float, world_end: int
) -> list[dict]:
    """Experiment 653's `match` applied to the called events only.

    An injected event is found when a called event at the same Primary has the same sign, a start
    within 24 half-hours, and an intersection over union of the two intervals of at least 0.5.
    """
    return match(truth=truth, events=called(events=events, tau=tau), world_end=world_end)


RAMP_BLOCKS: Final[tuple[int, ...]] = (3, 12, 24)
RAMP_HEIGHT: Final[float] = 4.0
RAMP_HOLD: Final[int] = 20


def ramp_series(*, n_ramp: int, seed: int) -> np.ndarray:
    """Student-t noise plus a ramp of +4 noise units over `n_ramp` blocks, held 5 days, and back."""
    pad = 120
    shape = np.concatenate(
        [
            np.zeros(pad),
            np.linspace(0.0, RAMP_HEIGHT, n_ramp + 1)[1:],
            np.full(RAMP_HOLD, RAMP_HEIGHT),
            np.linspace(RAMP_HEIGHT, 0.0, n_ramp + 1)[1:],
            np.zeros(pad),
        ]
    )
    return shape + np.random.default_rng(seed).standard_t(4, len(shape))


def ramp_diagnostic(*, priors: dict[str, hmm.Priors], tau: float, seed: int = 734) -> dict:
    """Called offset events on linear ramps, per prior setting and ramp length (never a gate).

    Args:
        priors: Named prior settings, such as H-nbr and H-nbr-noramp.
        tau: The confidence threshold of a called event.
        seed: The seed of the noise.

    Returns:
        For each setting name, the number of called o-events and the largest called amplitude in
        noise units, for each ramp length in blocks.
    """
    out: dict[str, dict[int, dict[str, float]]] = {}
    for name, p in priors.items():
        out[name] = {}
        for n_ramp in RAMP_BLOCKS:
            x = ramp_series(n_ramp=n_ramp, seed=seed)
            post, o_path, l_path = hmm.decode(
                x=x, f=np.zeros(len(x)), sigma=1.0, scale=np.ones(4), p=p
            )
            ev = extract(
                site="ramp",
                o_path=o_path,
                l_path=l_path,
                post=post,
                xh=np.repeat(x, BLOCK),
                sigma=1.0,
                record_end=len(x) * BLOCK,
            )
            ok = [e for e in called(events=ev, tau=tau) if e.cls != LEVEL_CLS]
            out[name][n_ramp] = {
                "called": float(len(ok)),
                "max_units": max((e.units for e in ok), default=0.0),
            }
    return out


def called_rates(
    *,
    events: list[Event],
    sigma653: dict[str, float],
    primary_years: float,
    lo: int,
    hi: int,
) -> tuple[float, float]:
    """The two called rates per Primary-year of the events that start in `[lo, hi)`.

    The second rate counts events whose absolute amplitude is at least `2 * sigma653[site]`
    megawatts, one threshold per Primary from experiment 653's noise units, whatever the method.

    Args:
        events: The called events.
        sigma653: Experiment 653's noise unit of every Primary in megawatts.
        primary_years: The Primary-years the interval covers.
        lo: First half-hour of the interval.
        hi: One past the last half-hour of the interval.

    Returns:
        `(all called events, called events of 2 noise units or more)` per Primary-year.
    """
    inside = [e for e in events if lo <= e.start < hi]
    big = [e for e in inside if abs(e.amp_mw) >= MIN_UNITS_OF_RATE * sigma653[e.primary]]
    return len(inside) / primary_years, len(big) / primary_years

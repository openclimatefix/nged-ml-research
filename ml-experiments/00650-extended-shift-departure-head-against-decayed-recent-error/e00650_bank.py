# ruff: noqa: E501
"""Experiment 650: the extended shift generator and the held-out shift family (pure numpy).

Same call shape as experiment 581's `contaminate`: a pass `r` and a month give one shift per
Primary. Unlike experiment 581's bank, shifts here have random ends, constant or proportional
shapes, step or ramp onsets, depths from 3 to 50 points of capacity (1 point = 1% of effective
capacity), and a partial-conservation receiver at one or two real electrical neighbours.

Two families, selected by the pass number:

- training passes `0 <= r < N_TRAIN_PASSES` (`family="train"`);
- held-out passes `HELD_OUT_FIRST <= r < HELD_OUT_FIRST + N_HELD_OUT_PASSES` (`family="held_out"`), whose
  ramps (72 to 120 h), depth (0.65), and durations (45 days) include values outside the training
  support.

`build(...)` returns the Primary's own shift `own` (added to the lag inputs and to the target) and the
receivers' mirrored shift `mirror` (added only to the head's neighbour signals, as experiment 647
did with `mirrored_power`), both (16, grid), plus the event table. Power is never summed across the
MW and MVA groups with the same shape: a receiver in the other group gets the donor's mean shift as
a constant and the stated power factor.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

PER_DAY: Final[int] = 48
N_TRAIN_PASSES: Final[int] = 12
HELD_OUT_FIRST: Final[int] = 200
N_HELD_OUT_PASSES: Final[int] = 4
POWER_FACTOR: Final[float] = 0.97
"""Power factor converting between a signed-MW Primary and an MVA-only Primary (MVA = MW / pf)."""
DEPTH_RANGE: Final[tuple[float, float]] = (0.03, 0.50)
SHORT_DAYS: Final[tuple[int, int]] = (3, 14)
LONG_DAYS: Final[tuple[int, int]] = (28, 112)
TRAIN_RAMP_HOURS: Final[tuple[int, int]] = (6, 48)
HELD_RAMP_HOURS: Final[tuple[int, int]] = (72, 120)
HELD_DEPTHS: Final[tuple[float, ...]] = (0.10, 0.20, 0.40, 0.65)
HELD_DAYS: Final[tuple[int, ...]] = (5, 10, 21, 45)
R_RANGE: Final[tuple[float, float]] = (0.5, 1.0)


@dataclass(frozen=True)
class Event:
    """One shift of one Primary in one pass (`own` is the share of capacity added to its power)."""

    r: int
    primary: int
    kind: str  # "constant" or "proportional"
    onset_idx: int
    ramp_half_hours: int
    end_idx: int
    depth: float  # unsigned share of capacity (mean over the event for a proportional shape)
    sign: int
    gain: float  # receiver gain r
    receivers: tuple[int, ...]
    out_of_support: str  # comma-separated cells outside the training support: ramp, depth, days

    def row(self) -> dict:
        """Return the event as a flat dict for a table."""
        return {
            "pass": self.r, "primary": self.primary, "kind": self.kind, "onset_idx": self.onset_idx,
            "ramp_half_hours": self.ramp_half_hours, "end_idx": self.end_idx, "depth": self.depth,
            "sign": self.sign, "length_days": (self.end_idx - self.onset_idx) / PER_DAY,
            "gain": self.gain, "n_receivers": len(self.receivers),
            "out_of_support": self.out_of_support,
        }  # fmt: skip


def family_of(*, r: int) -> str:
    """Return `"train"` or `"held_out"` for pass `r`."""
    if 0 <= r < N_TRAIN_PASSES:
        return "train"
    if HELD_OUT_FIRST <= r < HELD_OUT_FIRST + N_HELD_OUT_PASSES:
        return "held_out"
    msg = f"unknown pass {r}"
    raise ValueError(msg)


def _profile(*, n: int, onset: int, ramp: int, end: int) -> np.ndarray:
    """Return the 0..1 envelope on grid indices [0, n): a step or linear ramp from `onset`, zero from `end`."""
    i = np.arange(n)
    up = np.clip((i - onset + 1) / max(ramp, 1), 0.0, 1.0)
    return np.where((i >= onset) & (i < end), up, 0.0)


def draw(*, month_idx: int, r: int, primary: int, nb_of: dict[int, list[int]], month_start_idx: int,
         window_end_idx: int, is_mw: np.ndarray) -> Event:  # fmt: skip
    """Draw one event; the draw depends only on (month index, pass, Primary), never on the data."""
    rng = np.random.default_rng([650, month_idx, r, primary])
    fam = family_of(r=r)
    nbs = nb_of.get(primary, [])
    if fam == "train":
        kind = "constant" if (rng.random() < 0.5 or not bool(is_mw[primary])) else "proportional"
        depth = float(np.exp(rng.uniform(np.log(DEPTH_RANGE[0]), np.log(DEPTH_RANGE[1]))))
        sign = 1 if rng.random() < 0.5 else -1
        if rng.random() < 0.5:
            days = int(rng.integers(SHORT_DAYS[0], SHORT_DAYS[1] + 1))
        else:
            days = int(rng.integers(LONG_DAYS[0], LONG_DAYS[1] + 1))
        ramp = 1 if rng.random() < 0.5 else int(rng.integers(*TRAIN_RAMP_HOURS)) * 2
        # onset between 10 days before the month and day 20, so ends fall before, inside, and after the horizon
        onset = month_start_idx + int(rng.integers(-10 * PER_DAY, 20 * PER_DAY))
    else:
        k = r - HELD_OUT_FIRST
        cell = (primary + 5 * k) % 16
        sign = (
            1 if k % 2 == 0 else -1
        )  # each cell meets both signs, twice each, across the 4 passes
        depth = HELD_DEPTHS[cell % 4]
        days = HELD_DAYS[cell // 4]
        kind = "constant"
        long_ramp = (primary + k) % 2 == 0  # half the events have a ramp outside the training range
        ramp = int(rng.integers(*(HELD_RAMP_HOURS if long_ramp else TRAIN_RAMP_HOURS))) * 2
        onset = month_start_idx + int(rng.integers(0, 10 * PER_DAY))
    end = min(onset + days * PER_DAY, window_end_idx)
    n_rec = 0 if not nbs else int(rng.integers(1, min(2, len(nbs)) + 1))
    if nbs and rng.random() < 0.15:
        n_rec = 0  # some transfers leave the set of 16 Primaries
    receivers = tuple(int(x) for x in rng.permutation(nbs)[:n_rec]) if n_rec else ()
    gain = float(rng.uniform(*R_RANGE))
    in_days = SHORT_DAYS[0] <= days <= SHORT_DAYS[1] or LONG_DAYS[0] <= days <= LONG_DAYS[1]
    oos = ",".join(
        c for c, hit in (("ramp", fam == "held_out" and ramp // 2 > TRAIN_RAMP_HOURS[1]),
                         ("depth", depth > DEPTH_RANGE[1]), ("days", not in_days)) if hit
    )  # fmt: skip
    return Event(r=r, primary=primary, kind=kind, onset_idx=onset, ramp_half_hours=ramp, end_idx=end,
                 depth=depth, sign=sign, gain=gain, receivers=receivers, out_of_support=oos)  # fmt: skip


def build(
    *, clean: np.ndarray, month_idx: int, month_start_idx: int, r: int, nb_of: dict[int, list[int]],
    capacity: np.ndarray, is_mw: np.ndarray, window_end_idx: int,
) -> tuple[np.ndarray, np.ndarray, list[Event]]:  # fmt: skip
    """Return (`own`, `mirror`, events) for pass `r`: both arrays are (16, grid) shifts as shares of capacity.

    A proportional shape scales the Primary's own clean series so that its mean over the event
    equals the drawn depth. `mirror` carries each receiver's gain-weighted opposite shift in
    megawatts divided by the receiver's capacity. NaN power stays NaN where the shifts are later
    added by the callers.
    """
    n_prim, n = clean.shape
    own = np.zeros((n_prim, n), np.float32)
    mirror = np.zeros((n_prim, n), np.float32)
    events = []
    for p in range(n_prim):
        ev = draw(month_idx=month_idx, r=r, primary=p, nb_of=nb_of, month_start_idx=month_start_idx,
                  window_end_idx=window_end_idx, is_mw=is_mw)  # fmt: skip
        events.append(ev)
        if ev.end_idx <= ev.onset_idx:
            continue
        env = _profile(n=n, onset=ev.onset_idx, ramp=ev.ramp_half_hours, end=ev.end_idx)
        span = slice(max(ev.onset_idx, 0), ev.end_idx)
        if ev.kind == "constant":
            shape = np.ones(n, np.float32)
        else:
            seg = clean[p, span]
            mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 1.0
            shape = (np.nan_to_num(clean[p], nan=mean) / max(abs(mean), 1e-6)).astype(np.float32)
        delta = (ev.sign * ev.depth * env * shape).astype(np.float32)
        if not bool(
            is_mw[p]
        ):  # an MVA-only Primary's power is a magnitude and stays at or above zero
            delta = np.where(
                np.isfinite(clean[p]), np.maximum(clean[p] + delta, 0.0) - clean[p], delta
            ).astype(np.float32)
        own[p] = delta
        if not ev.receivers:
            continue
        per = ev.gain / len(ev.receivers)
        for q in ev.receivers:
            mw = (
                delta * capacity[p] / capacity[q]
            )  # donor's megawatts as a share of the receiver's capacity
            if bool(is_mw[p]) != bool(is_mw[q]):
                mean_mw = np.full(
                    n, float(delta[span].mean()) if ev.end_idx > ev.onset_idx else 0.0
                )
                mean_mw = mean_mw * (env > 0) * capacity[p] / capacity[q]
                mw = mean_mw * (1.0 / POWER_FACTOR if bool(is_mw[p]) else POWER_FACTOR)
            mirror[q] -= (per * mw).astype(np.float32)
    return own, mirror, events

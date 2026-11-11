"""Experiment 693: the oracle and placebo transfer flags (pure numpy, no torch).

The oracle is a perfect telemetry detector. It knows whether a reading lay inside a load transfer,
and, for a transfer already in progress, when the transfer began. It knows neither the end nor the
depth of the transfer.

Two families of flags are built here.

- **Lag flags for the Candidate's lag encoder** (`lag_flags`): per row, four columns `own_day`,
  `nbr_day`, `own_week`, `nbr_week`. `own_*` says that the row's Primary was inside a transfer at the
  lag's time, and `nbr_*` that an electrical neighbour was. Every lag time is at least 24 hours
  before the 09:00 origin, and `lag_flags` asserts that.
- **Origin flags for the prevailing head** (`origin_flags`): per (Primary, origin), whether a transfer
  is known to be in progress and its age since onset. A transfer is known only if its onset is at or
  before the origin minus 6 hours, the age of the freshest reading the head reads (experiment 681).

The placebo (`roll_mask`) rolls every Primary's true spans along the time axis by one fixed offset.
Span lengths, overlaps, and per-Primary counts stay exactly as in the true flags, and the link to
the real power and weather is broken.
"""

from typing import Final

import numpy as np

PER_DAY: Final[int] = 48
"""Half-hours per day."""
ORIGIN_DELAY: Final[int] = 12
"""The newest reading the prevailing head reads is the origin minus 6 hours, 12 half-hours."""
LAG_MIN_AGE: Final[int] = 48
"""Every lag of the Candidate is at least 24 hours (48 half-hours) older than the origin."""
FLAG_NAMES: Final = ("own_day", "nbr_day", "own_week", "nbr_week")
"""The four lag-encoder columns, in order."""
N_FLAGS: Final[int] = len(FLAG_NAMES)
SHIFT_EPS: Final[float] = 1e-6
"""A half-hour counts as inside an injected shift where the shift's magnitude exceeds this (a share of capacity)."""


def active_from_shift(*, shift: np.ndarray) -> np.ndarray:
    """Return the (Primaries, grid) mask of half-hours whose power carries an injected shift."""
    return np.abs(shift) > SHIFT_EPS


def active_from_windows(*, windows: dict[int, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Return the (Primaries, grid) mask from per-Primary boolean event-window masks."""
    out = np.zeros(shape, bool)
    for prim, mask in windows.items():
        out[prim, : len(mask)] = mask[: shape[1]]
    return out


def neighbour_active(
    *, active: np.ndarray, nb_of: dict[int, list[int]], choice: dict[int, int] | None
) -> np.ndarray:
    """Return, per Primary and grid time, whether an electrical neighbour was inside a transfer.

    `choice` maps each shifted Primary to the neighbour that receives its mirrored shift, as in the
    fine-tune's synthetic events. A mirror receiver's own power stays clean, yet its neighbour
    summaries carry the mirrored shift, so its neighbour flag is set whenever the donor is active.
    Pass `None` where the transfer sits in the neighbours' own series (the experiment 660
    generator, where a recipient's own power is shifted).
    """
    out = np.zeros_like(active)
    for p, qs in nb_of.items():
        for q in qs:
            out[p] |= active[q]
    if choice is not None:
        for donor, receiver in choice.items():
            out[receiver] |= active[donor]
    return out


def roll_mask(*, mask: np.ndarray, offset: int) -> np.ndarray:
    """Return `mask` rolled by `offset` half-hours along time: the placebo spans."""
    assert 0 < offset < mask.shape[1], offset
    return np.roll(mask, offset, axis=1)


def placebo_offset(*, month: str, seed: int, grid_length: int) -> int:
    """Return the placebo's fixed roll in half-hours: 60 to 200 days, never a whole number of weeks."""
    rng = np.random.default_rng([693, int(month.replace("-", "")), seed])
    while True:
        offset = int(rng.integers(60 * PER_DAY, 200 * PER_DAY))
        if offset % (7 * PER_DAY) != 0 and offset < grid_length:
            return offset


def lag_flags(
    *,
    own: np.ndarray,
    nbr: np.ndarray,
    prim: np.ndarray,
    origin_idx: np.ndarray,
    day_idx: np.ndarray,
    week_idx: np.ndarray,
    day_missing: np.ndarray,
    week_missing: np.ndarray,
) -> np.ndarray:
    """Return the (rows, 4) flag block: own_day, nbr_day, own_week, nbr_week.

    A flag is 0 where its lag is missing or masked. Asserts that every flagged lag time is at least
    24 hours before the origin, so no flag reads a half-hour newer than the lag the network sees.
    """
    out = np.zeros((len(prim), N_FLAGS), np.float32)
    for j, (idx, miss) in enumerate(((day_idx, day_missing), (week_idx, week_missing))):
        live = ~miss & (idx >= 0)
        assert (idx[live] <= origin_idx[live] - LAG_MIN_AGE).all(), "a flagged lag is under 24 h old"
        i = np.clip(idx, 0, own.shape[1] - 1)
        out[:, 2 * j] = np.where(live, own[prim, i], False)
        out[:, 2 * j + 1] = np.where(live, nbr[prim, i], False)
    return out


def origin_flags(
    *, onsets: dict[int, list[tuple[int, int | None]]], prim: np.ndarray, oi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (in_progress, age_days) per (Primary, origin) row for the prevailing head.

    `onsets[p]` lists the (onset, end) half-hour indices of Primary `p`'s transfers; `end` is
    exclusive and `None` means still running. A transfer is known at origin `oi` only if
    `onset <= oi - ORIGIN_DELAY`, and it is in progress if `end` is `None` or later than
    `oi - ORIGIN_DELAY` (the freshest reading lay inside it). With several transfers in progress the
    earliest onset sets the age. The age is in days from the onset to the origin; 0 outside a
    transfer. The end never enters the result for a transfer still running at the freshest reading.
    """
    in_progress = np.zeros(len(prim), bool)
    age = np.zeros(len(prim), np.float32)
    first_onset = np.full(len(prim), np.iinfo(np.int64).max, np.int64)
    for p, spans in onsets.items():
        rows = prim == p
        if not rows.any():
            continue
        fresh = oi[rows] - ORIGIN_DELAY
        for onset, end in spans:
            known = (onset <= fresh) & (True if end is None else end > fresh)
            sel = np.flatnonzero(rows)[known]
            first_onset[sel] = np.minimum(first_onset[sel], onset)
            in_progress[sel] = True
    age[in_progress] = (oi[in_progress] - first_onset[in_progress]) / PER_DAY
    assert (age[in_progress] >= ORIGIN_DELAY / PER_DAY - 1e-6).all(), "a flag uses an onset after the freshest reading"
    return in_progress, age


def known_answer() -> None:
    """Check the flags on a hand-built world; raise on any failure (run by the stage scripts)."""
    n = 10 * PER_DAY
    shift = np.zeros((4, n), np.float32)
    shift[1, 3 * PER_DAY : 6 * PER_DAY] = 0.2
    active = active_from_shift(shift=shift)
    nb_of = {0: [1], 1: [0, 2], 2: [1], 3: []}
    choice = {1: 2}
    nbr = neighbour_active(active=active, nb_of=nb_of, choice=choice)
    assert nbr[0, 4 * PER_DAY] and nbr[2, 4 * PER_DAY] and not nbr[3, 4 * PER_DAY]
    assert not nbr[1, 4 * PER_DAY], "a Primary is not its own neighbour"
    prim = np.array([1, 2, 0, 3])
    origin = np.full(4, 8 * PER_DAY)
    day = np.full(4, 4 * PER_DAY)
    week = np.full(4, 0)
    miss = np.zeros(4, bool)
    flags = lag_flags(own=active, nbr=nbr, prim=prim, origin_idx=origin, day_idx=day, week_idx=week, day_missing=miss, week_missing=miss)
    assert flags[:, 0].tolist() == [1, 0, 0, 0], flags
    assert flags[:, 1].tolist() == [0, 1, 1, 0], flags
    assert flags[:, 2:].sum() == 0
    masked = lag_flags(own=active, nbr=nbr, prim=prim, origin_idx=origin, day_idx=day, week_idx=week, day_missing=np.ones(4, bool), week_missing=miss)
    assert masked[:, :2].sum() == 0, "a masked lag keeps its flag"
    rolled = roll_mask(mask=active, offset=2 * PER_DAY + 5)
    assert rolled.sum() == active.sum() and not np.array_equal(rolled, active)
    inp, age = origin_flags(onsets={1: [(3 * PER_DAY, 6 * PER_DAY)]}, prim=np.array([1, 1, 1, 1]), oi=np.array([3 * PER_DAY + 5, 3 * PER_DAY + 12, 5 * PER_DAY, 7 * PER_DAY]))
    assert inp.tolist() == [False, True, True, False], inp
    assert abs(age[2] - 2.0) < 1e-6 and age[0] == 0 and age[3] == 0

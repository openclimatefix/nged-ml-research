# ruff: noqa: E501, ANN001
"""Experiment 581: the lag-only injected event bank (pure numpy; no fitting code is imported).

A pass `r` (0 to 23) adds a constant shift to the cleaned power of every Primary over the half-hours
`[end - length, end)`, where `end` is day 14 of the fold month. Primary `p` gets combination
`(p + r) % 12` of depth in {0.15, 0.30, 0.50} (share of effective capacity) and length in {3, 7, 15,
30} days, with sign +1 in passes 0 to 11 and -1 in passes 12 to 23. Across the 24 passes every
Primary meets every combination with both signs. The shift is added to the lag inputs only, never to
the targets, so the uninjected power is the normal-running-arrangement (NRA) reference.
"""

from typing import Final

import numpy as np

DEPTHS: Final[tuple[float, ...]] = (0.15, 0.30, 0.50)
LENGTHS_DAYS: Final[tuple[int, ...]] = (3, 7, 15, 30)
COMBOS: Final[tuple[tuple[float, int], ...]] = tuple((d, n) for d in DEPTHS for n in LENGTHS_DAYS)
N_PASSES: Final[int] = 2 * len(COMBOS)
END_DAY: Final[int] = 14
PER_DAY: Final[int] = 48


def event(*, primary: int, r: int) -> tuple[float, int, int]:
    """Return (signed depth as a share of capacity, length in days, sign) of Primary `primary` in pass `r`."""
    depth, length = COMBOS[(primary + r) % len(COMBOS)]
    sign = 1 if r < len(COMBOS) else -1
    return sign * depth, length, sign


def contaminate(
    *, clean: np.ndarray, month_start_idx: int, r: int
) -> tuple[np.ndarray, list[dict]]:
    """Return a copy of `clean` (primaries x grid) with pass `r`'s shifts added, and the event rows."""
    out = clean.copy()
    end = month_start_idx + END_DAY * PER_DAY
    rows = []
    for p in range(clean.shape[0]):
        shift, length, sign = event(primary=p, r=r)
        on = end - length * PER_DAY
        seg = out[p, on:end]
        out[p, on:end] = np.where(np.isfinite(seg), seg + shift, seg)
        rows.append(
            {"pass": r, "primary": p, "depth": abs(shift), "sign": sign, "length_days": length,
             "onset_idx": on, "end_idx": end}
        )  # fmt: skip
    return out, rows


REAL_PASS: int = 100
"""Pass number of the corrected-lag twin of the real verified events (experiment 580)."""


def verified_events(
    *, capacity_mw: dict[str, float], label_to_primary: dict[str, int], table
) -> list[dict]:
    """Return one row per (transfer, Primary side) of experiment 580's `events_verified.parquet`.

    `depth` is the signed share of capacity (positive: the Primary carries more power after onset);
    the counterfactual without the event is observed minus depth inside `[onset_idx, end_idx)`.
    """
    rows = []
    for q in table.iter_rows(named=True):
        for side in ("a", "b"):
            lab = q[f"primary_{side}"]
            rows.append(
                {"transfer_id": q["transfer_id"], "side": side, "label": lab,
                 "primary": label_to_primary[lab], "onset_idx": q[f"onset_{side}_idx"],
                 "end_idx": q[f"end_{side}_idx"], "length_days": q["length_days"],
                 "depth": q[f"depth_{side}_mw"] / capacity_mw[lab]}
            )  # fmt: skip
    return rows


def corrected_clean(*, clean: np.ndarray, events: list[dict]) -> np.ndarray:
    """Return a copy of `clean` with every verified event removed (observed minus depth in its span)."""
    out = clean.copy()
    for q in events:
        seg = out[q["primary"], q["onset_idx"] : q["end_idx"]]
        out[q["primary"], q["onset_idx"] : q["end_idx"]] = np.where(
            np.isfinite(seg), seg - q["depth"], seg
        )
    return out


SYN_SHARE: Final[float] = 0.05
"""Share of each Primary's time covered by synthetic lag-only events in fine-tuning (X3-syn)."""


def synthetic_clean(
    *, clean: np.ndarray, first: int, last: int, seed: int
) -> tuple[np.ndarray, int]:
    """Return a copy of `clean` with random lag-only events covering `SYN_SHARE` of `[first, last)` per Primary.

    Each event has a depth from `DEPTHS`, a length from `LENGTHS_DAYS`, and a random sign, and starts at
    a random half-hour; the schedule depends only on `seed`, never on the data. Also returns the number
    of events.
    """
    rng = np.random.default_rng(seed)
    out = clean.copy()
    n_events = 0
    for p in range(clean.shape[0]):
        covered = 0
        while covered < SYN_SHARE * (last - first):
            length = int(rng.choice(LENGTHS_DAYS)) * PER_DAY
            on = int(rng.integers(first, max(first + 1, last - length)))
            shift = float(rng.choice(DEPTHS)) * float(rng.choice([-1.0, 1.0]))
            seg = out[p, on : on + length]
            out[p, on : on + length] = np.where(np.isfinite(seg), seg + shift, seg)
            covered += length
            n_events += 1
    return out, n_events

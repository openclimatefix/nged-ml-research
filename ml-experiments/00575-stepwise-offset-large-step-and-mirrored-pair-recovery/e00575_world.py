# ruff: noqa: E501
"""Experiment 575: the known-answer worlds Kl (large single steps) and Kp (mirrored pairs).

An event is a level change on one Primary that starts on `onset` and reverses on `end` (days since
2019-09-01, the day index of experiment 573). Sizes are fractions of the Primary's effective capacity.
The sampling ranges come from the labelled events (`label_quartiles`): depth 10% to 50% of capacity,
duration log-uniform from 1 day to 180 days.

Magnitude-only (MVA) Primaries cannot go below zero, so at an MVA Primary an event is never a loss:
a Kl event there is always a gain, and in Kp a pair is oriented so that no MVA Primary loses load.
"""

import itertools
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00573_world as w573  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

INJECTED: Final = w573.INJECTED
MVA_INJECTED: Final = w573.MVA_INJECTED
PER_DAY: Final = w573.PER_DAY
LABELS: Final = {0: "D2", 2: "D4", 8: "D12", 5: "D9", 7: "D11", 6: "D10"}
"""The injected Primary indices and the labels used in the edge list."""
EDGES: Final[dict[int, tuple[int, ...]]] = {
    0: (2, 6),
    2: (0,),
    8: (6, 7),
    5: (),
    7: (6, 8),
    6: (0, 7, 8),
}
"""Neighbours among the six injected Primaries, from `Primary Substation Interconnections.csv` (experiment 400's `nged_neighbours`); D9's only neighbour is not injected."""
SIZE_RANGE: Final = (0.10, 0.50)
DURATION_DAYS: Final = (1.0, 180.0)
RHO_RANGE: Final = (0.6, 1.0)
NO_RECIPIENT_SHARE: Final = 0.3
MIN_GAP: Final = 30
SEEDS: Final = 5
"""World draws per kind (Kl and Kp), to reach counts per bin that the binomial interval can resolve."""
FIRST_DAY: Final = w573.day_of(t=datetime(2022, 1, 1, tzinfo=UTC))
LAST_DAY: Final = w573.day_of(t=datetime(2026, 5, 31, tzinfo=UTC))
EVENTS_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored/work/events_corrected.parquet"
)


@dataclass(frozen=True)
class Event:
    """One event's change at one Primary: it begins on `onset` and reverses on `end`."""

    primary: int
    onset: int
    end: int
    size: float
    pair_id: int
    role: str
    rho: float = 1.0


def label_quartiles() -> dict[str, list[float]]:
    """Return the 0, 25, 50, 75, 100 percentiles of the labelled events' depth share and length in days."""
    e = pl.read_parquet(EVENTS_PATH)
    depth = e["depth_share"].abs().to_numpy() / 100.0
    days = e["length_days"].to_numpy()
    q = [0, 25, 50, 75, 100]
    return {
        "depth_share": np.percentile(depth, q).tolist(),
        "length_days": np.percentile(days, q).tolist(),
        "n_events": [float(len(e))],
        "share_depth_10_50": [float(((depth >= 0.1) & (depth <= 0.5)).mean())],
        "share_length_1d_or_more": [float((days >= 1.0).mean())],
        "share_length_over_30d": [float((days > 30.0).mean())],
    }


def capacities() -> dict[int, float]:
    """Return the effective capacity in MW of each injected Primary."""
    e = pl.read_parquet(EVENTS_PATH).unique("label")
    cap = {r["label"]: r["capacity_mw"] for r in e.iter_rows(named=True)}
    return {p: float(cap[lab]) for p, lab in LABELS.items()}


def _durations(*, rng: np.random.Generator) -> int:
    return int(np.exp(rng.uniform(np.log(DURATION_DAYS[0]), np.log(DURATION_DAYS[1]))))


def draw_large(*, primary: int, seed: int) -> list[Event]:
    """Return a Primary's events in world Kl.

    1 to 3 events a year on average: the next onset is at least 30 days after the last event ended.
    """
    rng = np.random.default_rng([575, 1, primary, seed])
    day = FIRST_DAY + int(rng.integers(0, 90))
    out: list[Event] = []
    while day <= LAST_DAY:
        size = float(np.exp(rng.uniform(np.log(SIZE_RANGE[0]), np.log(SIZE_RANGE[1]))))
        sign = 1.0 if primary in MVA_INJECTED else float(rng.choice([-1.0, 1.0]))
        end = day + max(_durations(rng=rng), 1)
        out.append(Event(primary, day, end, sign * size, len(out), "single"))
        day = end + MIN_GAP + int(rng.integers(0, 250))
    return out


def draw_pairs(*, seed: int) -> list[Event]:
    """Return the events of world Kp: for each donor, a sequence of events, each mirrored at a neighbour.

    The donor changes by `sign * size`; the recipient (an injected neighbour, absent in about 30% of
    events and always for D9) changes by `-rho * sign * size * donor capacity / recipient capacity`.
    No MVA Primary loses load: a donor MVA event is a gain, and a recipient that would lose is dropped.
    Events at one Primary keep at least 30 days between the end of one and the start of the
    next, whichever role.
    """
    cap = capacities()
    busy: dict[int, list[tuple[int, int]]] = {p: [] for p in INJECTED}
    out: list[Event] = []
    pair = 0
    for donor in INJECTED:
        rng = np.random.default_rng([575, 2, donor, seed])
        day = FIRST_DAY + int(rng.integers(0, 90))
        while day <= LAST_DAY:
            size = float(np.exp(rng.uniform(np.log(SIZE_RANGE[0]), np.log(SIZE_RANGE[1]))))
            sign = 1.0 if donor in MVA_INJECTED else float(rng.choice([-1.0, 1.0]))
            length = max(_durations(rng=rng), 1)
            clashes = [
                b for a, b in busy[donor] if day < b + MIN_GAP and a < day + length + MIN_GAP
            ]
            while clashes:
                day = max(clashes) + MIN_GAP
                clashes = [
                    b for a, b in busy[donor] if day < b + MIN_GAP and a < day + length + MIN_GAP
                ]
            if day > LAST_DAY:
                break
            end = day + length
            nxt = end + MIN_GAP + int(rng.integers(0, 250))
            options = EDGES[donor]
            has_recipient = bool(options) and rng.random() >= NO_RECIPIENT_SHARE
            recipient = int(rng.choice(options)) if has_recipient else None
            rho = float(rng.uniform(*RHO_RANGE))
            if recipient is not None:
                gain = -rho * sign * size * cap[donor] / cap[recipient]
                clash = any(day < b + MIN_GAP and a < end + MIN_GAP for a, b in busy[recipient])
                if clash or (recipient in MVA_INJECTED and gain < 0):
                    recipient = None
            out.append(Event(donor, day, end, sign * size, pair, "donor", rho))
            busy[donor].append((day, end))
            if recipient is not None:
                out.append(Event(recipient, day, end, gain, pair, "recipient", rho))
                busy[recipient].append((day, end))
            pair += 1
            day = nxt
    return out


def events(*, world: str, seed: int = 0) -> list[Event]:
    """Return the events of a world: `Kl`, `Kp`, or `K0` (none)."""
    if world == "Kl":
        return [e for p in INJECTED for e in draw_large(primary=p, seed=seed)]
    if world == "Kp":
        return draw_pairs(seed=seed)
    if world == "K0":
        return []
    raise ValueError(world)


def injection(*, world: str, grid: int, seed: int = 0) -> np.ndarray:
    """Return the world's added component (16 Primaries, `grid` half-hours), a fraction of capacity."""
    out = np.zeros((16, grid), dtype=np.float32)
    days = np.arange(grid) // PER_DAY
    for e in events(world=world, seed=seed):
        out[e.primary, (days >= e.onset) & (days < e.end)] += e.size
    return out


def truth_changes(*, world: str, seed: int = 0) -> dict[int, list[Event]]:
    """Return the events per injected Primary, sorted by onset."""
    out: dict[int, list[Event]] = {p: [] for p in INJECTED}
    for e in events(world=world, seed=seed):
        out[e.primary].append(e)
    return {p: sorted(v, key=lambda e: e.onset) for p, v in out.items()}


def _selftest() -> None:
    print(label_quartiles())
    for world in ("Kl", "Kp"):
        for seed in range(SEEDS):
            ev = events(world=world, seed=seed)
            for p in INJECTED:
                mine = sorted((e for e in ev if e.primary == p), key=lambda e: e.onset)
                for a, b in itertools.pairwise(mine):
                    assert b.onset >= a.end + MIN_GAP, (world, seed, p, a, b)
                if p in MVA_INJECTED:
                    assert all(e.size > 0 for e in mine), (world, seed, p)
            n_pairs = sum(e.role == "recipient" for e in ev)
            print(world, seed, len(ev), "events;", n_pairs, "recipients")
    print("capacities", capacities())


if __name__ == "__main__":
    _selftest()

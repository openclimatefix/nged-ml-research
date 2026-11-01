# ruff: noqa: E501, PLR0915
"""Training and held-out worlds with genuine level changes that a neighbour does and does not mirror.

**What the world holds.** Experiment 446's genuine level change adds one step to a Primary's lags
and target together. This module draws the same events (it calls experiment 446's `draw_events`
with the same seeds, so the Primary's own events are the events of arm U1n-2) and then flips a fair
coin per event. On a heads event it chooses 1 or 2 of the Primary's neighbours at random
(the Primary is the donor, the neighbours the recipients) and adds `-step * capacity_donor / k /
capacity_recipient` to each recipient over the same half-hours, in lags and target alike. That is
the signature a load shift leaves in experiment 402's generator, so a mirrored genuine change looks
like a load shift at the neighbours and differs only in what the target does. On a tails event no
neighbour changes.

**Files.** One `.npz` per (rate, seed) in this experiment's private work folder, holding `shift`
(primaries, grid) the share of capacity added by own events and mirrors together, `onset` (primaries,
grid) the grid index of the first half-hour of the own event covering each half-hour (-1 where none),
and `code` (primaries, grid) int8: 0 none, 1 own event with no mirror, 2 own event with a mirror, 3
mirror received from a neighbour (no own event). A half-hour that holds both an own event and a
mirror received keeps code 1 or 2 and the sum of the two steps in `shift`.

**Seeds.** Training worlds 0 to 3 use experiment 446's seeds 300 to 303 for the own events and
`seed + MIRROR_SEED_OFFSET` for the coins and the recipient choice. The three held-out worlds draw
own events with seeds 500 to 502 at the 3% held-out rate of experiment 446.

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00446-*:ml-experiments/00462-* DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00462-*/e00462_worlds.py selftest
uv run --no-sync python ml-experiments/00462-*/e00462_worlds.py build
uv run --no-sync python ml-experiments/00462-*/e00462_worlds.py rates
```
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00446_worlds as worlds446  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00462-lag-trust-mirrored-genuine-level-change-worlds" / "work"
)
WORLD_DIR: Final[Path] = WORK_DIR / "worlds"
NEIGHBOUR_FILE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00383-load-shift-xgboost-oracle-first-part-1"
    / "work"
    / "neighbours_nged.json"
)
TRAIN_SEEDS: Final[tuple[int, ...]] = worlds446.TRAIN_SEEDS
"""Experiment 446's training seeds, so the own events are the events of U1n-2."""
HELD_OUT_SEEDS: Final[tuple[int, ...]] = (500, 501, 502)
RATE: Final[float] = 0.02
HELD_OUT_RATE: Final[float] = worlds446.HELD_OUT_RATE
MIRROR_SEED_OFFSET: Final[int] = 10_000
MIRROR_PROBABILITY: Final[float] = 0.5
"""Held-out worlds: the share of genuine changes with a mirror at their recipients."""
TRAIN_MIRROR_PROBABILITY: Final[float] = 0.0
"""Training worlds: genuine changes have no mirror, so a mirror always means a load shift."""
MAX_RECIPIENTS: Final[int] = 2
"""Experiment 402's generator picks 1 or 2 recipients."""
NONE, OWN_PLAIN, OWN_MIRRORED, RECEIVED = 0, 1, 2, 3
"""Codes of the `code` array."""


@dataclass(frozen=True)
class Events:
    """The own events of one world: `(primary, start, stop, step)` per event, step a capacity share."""

    primary: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    step: np.ndarray


def events_of(*, shift: np.ndarray, onset: np.ndarray) -> Events:
    """Return the events that `worlds446.draw_events` placed (one per distinct onset per Primary)."""
    p_list, start_list, stop_list, step_list = [], [], [], []
    for p in range(shift.shape[0]):
        inside = np.nonzero(onset[p] >= 0)[0]
        if len(inside) == 0:
            continue
        starts = onset[p, inside]
        # Events do not overlap, so each distinct onset value labels one contiguous run.
        for s in np.unique(starts):
            run = inside[starts == s]
            p_list.append(p)
            start_list.append(int(s))
            stop_list.append(int(run.max()) + 1)
            step_list.append(float(shift[p, s]))
    return Events(
        primary=np.array(p_list, dtype=np.int64), start=np.array(start_list, dtype=np.int64),
        stop=np.array(stop_list, dtype=np.int64), step=np.array(step_list, dtype=np.float64),
    )  # fmt: skip


def add_mirrors(
    *, own_shift: np.ndarray, onset: np.ndarray, neighbours: list[list[int]],
    capacity: np.ndarray, seed: int, probability: float = MIRROR_PROBABILITY,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return `(shift, code)`: own events plus the load-conserving mirror on a share of them.

    Args:
        own_shift: (primaries, grid) share of capacity added by the own events.
        onset: (primaries, grid) grid index of the first half-hour of the own event (-1 outside).
        neighbours: Neighbour indices of each Primary (the possible recipients).
        capacity: Effective capacity in MW of each Primary.
        seed: Seed of the coin flips and of the recipient choice.
        probability: Chance that an own event is mirrored (if the Primary has any neighbour).

    Returns:
        The merged share-of-capacity shift (float32) and the int8 `code` array.
    """
    rng = np.random.default_rng(seed)
    shift = own_shift.astype(np.float64).copy()
    code = np.zeros(own_shift.shape, dtype=np.int8)
    events = events_of(shift=own_shift, onset=onset)
    mirrored = np.zeros(len(events.primary), dtype=bool)
    for i in range(len(events.primary)):
        p, a, b, step = (
            int(events.primary[i]), int(events.start[i]), int(events.stop[i]), events.step[i],
        )  # fmt: skip
        candidates = neighbours[p]
        heads = rng.random() < probability
        if not heads or not candidates:
            code[p, a:b] = OWN_PLAIN
            continue
        k = int(rng.integers(1, min(MAX_RECIPIENTS, len(candidates)) + 1))
        recipients = rng.choice(candidates, size=k, replace=False)
        code[p, a:b] = OWN_MIRRORED
        mirrored[i] = True
        for q in recipients:
            shift[int(q), a:b] += -step * capacity[p] / k / capacity[int(q)]
            free = code[int(q), a:b] == NONE
            code[int(q), a:b][free] = RECEIVED
    return shift.astype(np.float32), code


def neighbour_lists(*, sites: list[str], path: Path = NEIGHBOUR_FILE) -> list[list[int]]:
    """Return each Primary's neighbours (indices into `sites`) from the adjacency file."""
    import json

    graph = json.loads(path.read_text())
    index = {s: i for i, s in enumerate(sites)}
    return [[index[n] for n in graph.get(s, []) if n in index] for s in sites]


def world_file(*, seed: int) -> Path:
    """Return the file of the world whose own events use `seed`."""
    return WORLD_DIR / f"mirrored_seed{seed}.npz"


def load_world(*, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(shift, onset, code)` of a stored world, or fail with the command that builds it."""
    path = world_file(seed=seed)
    if not path.exists():
        msg = f"{path} is missing: run `e00462_worlds.py build`"
        raise FileNotFoundError(msg)
    z = np.load(path)
    return z["shift"], z["onset"], z["code"]


def build_one(
    *, ranges: worlds446.LabelRanges, observed: np.ndarray, rate: float, seed: int,
    neighbours: list[list[int]], capacity: np.ndarray, probability: float,
) -> str:  # fmt: skip
    """Draw and store one world unless it exists; return a progress line."""
    path = world_file(seed=seed)
    if path.exists():
        return f"{path.name}: exists"
    own, onset = worlds446.draw_events(ranges=ranges, observed=observed, rate=rate, seed=seed)
    shift, code = add_mirrors(
        own_shift=own, onset=onset, neighbours=neighbours, capacity=capacity,
        seed=seed + MIRROR_SEED_OFFSET, probability=probability,
    )  # fmt: skip
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, shift=shift, onset=onset, code=code)
    temporary.rename(path)
    inside = (shift != 0) & observed
    share = inside.sum(axis=1) / observed.sum(axis=1)
    return (
        f"{path.name}: share of observed half-hours with a level change per Primary: "
        + " ".join(f"{v:.2%}" for v in share)
    )


def context() -> tuple[list[str], np.ndarray, list[list[int]], worlds446.LabelRanges, np.ndarray]:
    """Return `(sites, capacity, neighbours, ranges, observed)` of the real clean series."""
    import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
    import e00428_data as data  # ty: ignore[unresolved-import]

    ctx = worlds402.load_context()
    sites = data.sites()
    stack, _ = data.load_stack(include_eval=False)
    capacity = np.array([ctx.capacity[s] for s in sites], dtype=np.float64)
    ranges = worlds446.label_ranges(capacity=ctx.capacity, trimmed=False)
    return sites, capacity, neighbour_lists(sites=sites), ranges, np.isfinite(stack[0])


def build_all() -> list[str]:
    """Build the four training worlds and the three held-out worlds."""
    _, capacity, neighbours, ranges, observed = context()
    lines = [
        build_one(
            ranges=ranges, observed=observed, rate=RATE, seed=s, neighbours=neighbours,
            capacity=capacity, probability=TRAIN_MIRROR_PROBABILITY,
        )
        for s in TRAIN_SEEDS
    ]  # fmt: skip
    lines += [
        build_one(
            ranges=ranges, observed=observed, rate=HELD_OUT_RATE, seed=s, neighbours=neighbours,
            capacity=capacity, probability=MIRROR_PROBABILITY,
        )
        for s in HELD_OUT_SEEDS
    ]  # fmt: skip
    return lines


def base_rates() -> list[str]:
    """Print the Bayes-optimal shares of a jump that moves the target, with and without a mirror.

    A half-hour at a Primary is a *jump* if the Primary's power differs from the clean series in a
    training world (a load-shift donor or recipient from experiment 428's worlds 1 to 3, or a level
    change from this experiment's worlds). The *target moved* if the level change is non-zero at the
    Primary at that half-hour. The jump has a *mirror* if any neighbour's power differs from the
    clean series at the same half-hour. The share of jumps that moved the target, within each of the
    two groups, is the share of the step an optimal forecast follows. It is read per half-hour,
    a proxy for the per-row share.
    """
    import e00428_data as data  # ty: ignore[unresolved-import]

    sites, _, neighbours, _, _ = context()
    stack, _ = data.load_stack(include_eval=False)
    clean = stack[0]
    out = []
    totals = {"mirror": [0, 0], "no mirror": [0, 0]}
    for w, seed in enumerate(TRAIN_SEEDS):
        shift, _, _ = load_world(seed=seed)
        # power in world w is the injected stack plus the level change (before any clipping).
        injected = np.nan_to_num(stack[w] - clean) + shift
        target_moved = shift != 0
        jump = injected != 0
        neighbour_jump = np.zeros_like(jump)
        for p, nbrs in enumerate(neighbours):
            for q in nbrs:
                neighbour_jump[p] |= injected[q] != 0
        for name, sel in (("mirror", jump & neighbour_jump), ("no mirror", jump & ~neighbour_jump)):
            totals[name][0] += int((sel & target_moved).sum())
            totals[name][1] += int(sel.sum())
        share = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in totals.items()}
        out.append(
            f"through training world {w}: share of jump half-hours whose target moved: "
            f"with a mirror {share['mirror']:.3f} ({totals['mirror'][1]:,} half-hours), "
            f"without {share['no mirror']:.3f} ({totals['no mirror'][1]:,})"
        )
    shifted = sum(
        int(((load_world(seed=s)[0] != 0) & np.isfinite(clean)).sum()) for s in TRAIN_SEEDS
    )
    observed = int(np.isfinite(clean).sum()) * len(TRAIN_SEEDS)
    out.append(f"share of observed half-hours inside a level change: {shifted / observed:.3%}")
    return out


def selftest() -> None:
    """Check the draw, the mirror's conservation, and the codes on a toy grid (no data needed)."""
    ranges = worlds446.LabelRanges(depth=np.array([0.1, 0.3]), steps=np.array([10, 40]))
    n_p, grid = 4, 4_000
    observed = np.ones((n_p, grid), dtype=bool)
    capacity = np.array([10.0, 20.0, 30.0, 40.0])
    neighbours = [[1, 2], [0], [0, 3], []]  # Primary 3 has no neighbour, so it cannot be mirrored
    own, onset = worlds446.draw_events(ranges=ranges, observed=observed, rate=0.05, seed=1)
    events = events_of(shift=own, onset=onset)
    assert len(events.primary) > 10, "the toy draw must place events"
    rebuilt = np.zeros_like(own)
    for p, a, b, s in zip(events.primary, events.start, events.stop, events.step, strict=True):
        rebuilt[p, a:b] = s
    assert np.allclose(rebuilt, own), "events_of must recover every own event"
    # Probability 1: every event of a Primary with neighbours is mirrored. Total MW is conserved
    # except for the events of Primary 3, which has nobody to mirror to.
    shift, code = add_mirrors(
        own_shift=own, onset=onset, neighbours=neighbours, capacity=capacity, seed=2,
        probability=1.0,
    )  # fmt: skip
    unmirrored_mw = np.zeros(grid)
    for p, a, b, s in zip(events.primary, events.start, events.stop, events.step, strict=True):
        if neighbours[int(p)]:
            assert (code[int(p), a:b] == OWN_MIRRORED).all(), "an event with neighbours is mirrored"
        else:
            assert (code[int(p), a:b] == OWN_PLAIN).all(), "an event without neighbours is plain"
            unmirrored_mw[a:b] += s * capacity[p]
    total_mw = (shift * capacity[:, None]).sum(axis=0)
    assert np.allclose(total_mw, unmirrored_mw, atol=1e-3), "a mirror must conserve total MW"
    # A mirror has the opposite sign to the step it mirrors, wherever only one event touches it.
    first = int(np.nonzero(events.primary == 0)[0][0])
    p, a, b, s = (int(events.primary[first]), int(events.start[first]), int(events.stop[first]),
                  events.step[first])  # fmt: skip
    lone = [q for q in neighbours[p] if (own[q, a:b] == 0).all() and (code[q, a:b] == RECEIVED).all()]
    assert lone, "the toy draw must give a recipient with no event of its own"
    assert all(np.sign(shift[q, a:b]).max() == -np.sign(s) for q in lone[:1]) or lone
    # The half-hours that no event touches have code 0 and shift 0; mirror-only half-hours are 3.
    assert (shift[code == NONE] == 0).all() and (code[(own == 0) & (shift != 0)] == RECEIVED).all()
    # Probability 0 is experiment 446's world: no mirror and the same shift.
    plain, plain_code = add_mirrors(
        own_shift=own, onset=onset, neighbours=neighbours, capacity=capacity, seed=2,
        probability=0.0,
    )  # fmt: skip
    assert np.array_equal(plain, own) and not np.isin(plain_code, (OWN_MIRRORED, RECEIVED)).any()
    # About half of the mirrorable events are mirrored at probability 0.5, and the draw repeats.
    half, half_code = add_mirrors(
        own_shift=own, onset=onset, neighbours=neighbours, capacity=capacity, seed=3,
    )  # fmt: skip
    mirrorable = [i for i, q in enumerate(events.primary) if neighbours[int(q)]]
    mirrored = np.array(
        [half_code[int(events.primary[i]), int(events.start[i])] == OWN_MIRRORED for i in mirrorable]
    )
    assert 0.3 < mirrored.mean() < 0.7, mirrored.mean()
    again, _ = add_mirrors(
        own_shift=own, onset=onset, neighbours=neighbours, capacity=capacity, seed=3,
    )  # fmt: skip
    assert np.array_equal(again, half)
    print("selftest passed")


def main() -> None:
    """Build the worlds, print the base rates, or run the selftest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "rates", "selftest"))
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "build":
        for line in build_all():
            print(line, flush=True)
    else:
        for line in base_rates():
            print(line, flush=True)


if __name__ == "__main__":
    main()

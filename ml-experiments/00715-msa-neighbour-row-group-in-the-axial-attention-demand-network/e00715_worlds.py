# ruff: noqa: E501
"""Experiment 715: the mirror world of arms `F4-W` and `F4-WR`, and the synthetic worlds W3 and W3p.

**The mirror world (design section 5.3).** Experiment 713's stage 3 draws experiment 581's lag-only
shifts (`e00581_bank.synthetic_clean`) at each Primary independently, and every shift sits in the
history and never in the target, so a network trained on them learns "do not follow a step" whatever
the neighbours show. The mirror world keeps 581's schedule (the same seed, so the same onsets,
lengths, depths, and signs at each host as in `M*`) and uses a second random stream, seeded by the
fold month, to give each event a kind:

- **A temporary move (70% of events).** The shift stays in the host's history and out of the target.
  Two thirds of the moves are mirrored with the opposite sign at one listed neighbour, drawn
  uniformly, with a mirrored share drawn uniformly from 0.2 to 1.0 of the moved megawatts, converted
  with the two capacities; an MVA-only neighbour is floored at zero. One third have no mirror.
  The mirrored step sits in the neighbour's history only, never in the neighbour's target.
- **A permanent change (30% of events).** The shift keeps its onset and depth, lasts 60 days, and sits
  in the host's history and in the host's target. One in ten has a chance mirror at a neighbour,
  which sits in the neighbour's history only.

`mirror_world` returns two (16, G) arrays: `history` (what the history rows of every Primary read, in
capacity shares) and `target` (the cleaned power plus the permanent changes only), and a table of events.

**The synthetic worlds (design section 8.1).** `build_w3` is experiment 713's world W0 with level steps
whose kind only the neighbour reveals: steps of +0.30 or -0.30 of capacity that start at a random
half-hour and together cover about 15% of the history; half are temporary (4 to 12 days, uniform) and
mirrored in full, with the opposite sign and the same megawatts, at the Primary's pair partner, and half
are permanent (45 days) with no mirror. `build_w3` takes a `share_range`; `W3p` draws it from 0.3 to 1.0.
"""

from dataclasses import dataclass
from typing import Final

import e00581_bank as bank581  # ty: ignore[unresolved-import]
import e00713_layout as lay
import e00715_neighbours as nb
import numpy as np

PER_DAY: Final = lay.STEPS_PER_DAY
MIRROR_KEY: Final = 5_815
TEMPORARY_SHARE: Final = 0.70
MIRRORED_OF_TEMPORARY: Final = 2.0 / 3.0
MIRROR_RANGE: Final = (0.2, 1.0)
PERMANENT_DAYS: Final = 60
CHANCE_MIRROR: Final = 0.1
W3_STEP: Final = 0.30
W3_SHARE: Final = 0.15
W3_TEMPORARY_DAYS: Final = (4, 12)
W3_PERMANENT_DAYS: Final = 45
W3_SEED: Final = 3
NOISE_SD: Final = 0.01


@dataclass(frozen=True)
class Event:
    """One event of experiment 581's schedule, in the order the schedule draws them.

    Attributes:
        primary: The host Primary index.
        onset: First grid index of the event.
        length: Length in half-hours (before the permanent changes are lengthened).
        shift: The signed depth in shares of the host's capacity.
    """

    primary: int
    onset: int
    length: int
    shift: float


@dataclass
class MirrorWorld:
    """The two arrays of the mirror world and its event table.

    Attributes:
        history: (16, G) power over capacity that the history rows read, all shifts and mirrors in.
        target: (16, G) power over capacity that the targets read: the cleaned power plus the
            permanent changes only.
        events: One dict per event: host, onset, length, kind, neighbour, share, megawatts,
            floored, chance.
    """

    history: np.ndarray
    target: np.ndarray
    events: list[dict]


def schedule(*, shape: tuple[int, int], first: int, last: int, seed: int) -> list[Event]:
    """Return experiment 581's schedule of lag-only events, drawing in the order `synthetic_clean` draws.

    The function copies the random draws of `e00581_bank.synthetic_clean` (a length, an onset, a depth,
    and a sign per event, Primary by Primary, until `SYN_SHARE` of `[first, last)` is covered), so that
    `apply_all` of the schedule equals `synthetic_clean` of the same seed. `shape` is the shape of
    `clean`.
    """
    rng = np.random.default_rng(seed)
    events = []
    for p in range(shape[0]):
        covered = 0
        while covered < bank581.SYN_SHARE * (last - first):
            length = int(rng.choice(bank581.LENGTHS_DAYS)) * PER_DAY
            on = int(rng.integers(first, max(first + 1, last - length)))
            shift = float(rng.choice(bank581.DEPTHS)) * float(rng.choice([-1.0, 1.0]))
            events.append(Event(primary=p, onset=on, length=length, shift=shift))
            covered += length
    return events


def add_shift(
    *, array: np.ndarray, primary: int, onset: int, length: int, shift: float, floor: bool = False
) -> None:
    """Add `shift` to the finite half-hours of `array[primary, onset : onset + length]` in place."""
    seg = array[primary, onset : onset + length]
    new = seg + shift
    if floor:
        new = np.maximum(new, 0.0)
    array[primary, onset : onset + length] = np.where(np.isfinite(seg), new, seg)


def apply_all(*, clean: np.ndarray, events: list[Event]) -> np.ndarray:
    """Return `clean` with every event of the schedule applied as experiment 581 applies it."""
    out = clean.copy()
    for e in events:
        add_shift(array=out, primary=e.primary, onset=e.onset, length=e.length, shift=e.shift)
    return out


def mirror_world(
    *, clean: np.ndarray, first: int, last: int, seed: int, month_key: int,
    table: nb.NeighbourTable, capacity_mw: np.ndarray, mva_only: np.ndarray,
) -> MirrorWorld:  # fmt: skip
    """Return the mirror world of one fold.

    Args:
        clean: (16, G) cleaned power over capacity, NaN where flagged or missing.
        first: First grid index of the schedule (experiment 713's `c_idx.min() - 768`).
        last: End of the schedule (`c_idx.max() + 672`).
        seed: Experiment 581's seed (`int(month.replace("-", ""))`).
        month_key: The seed of the second stream, the fold month as an integer.
        table: The neighbour table (the mirror lands on a listed neighbour).
        capacity_mw: (16,) effective capacity in megawatts.
        mva_only: (16,) bool, True for MVA-only Primaries (a mirrored step is floored at zero).
    """
    history = clean.copy()
    target = clean.copy()
    rng2 = np.random.default_rng([MIRROR_KEY, month_key])
    out_events = []
    for e in schedule(shape=clean.shape, first=first, last=last, seed=seed):
        # every draw is made for every event, so the stream does not depend on the kind
        u_kind, u_mirror, u_nbr, share, u_chance = (
            rng2.random(), rng2.random(), rng2.random(), rng2.uniform(*MIRROR_RANGE), rng2.random(),
        )  # fmt: skip
        temporary = u_kind < TEMPORARY_SHARE
        length = e.length if temporary else PERMANENT_DAYS * PER_DAY
        add_shift(array=history, primary=e.primary, onset=e.onset, length=length, shift=e.shift)
        if not temporary:
            add_shift(array=target, primary=e.primary, onset=e.onset, length=length, shift=e.shift)
        near = table.neighbours[e.primary]
        mirrored = bool(near) and (
            u_mirror < MIRRORED_OF_TEMPORARY if temporary else u_chance < CHANCE_MIRROR
        )
        neighbour, megawatts, floored = -1, 0.0, False
        if mirrored:
            neighbour = near[min(int(u_nbr * len(near)), len(near) - 1)]
            megawatts = -e.shift * capacity_mw[e.primary] * share
            floored = bool(mva_only[neighbour])
            add_shift(
                array=history, primary=neighbour, onset=e.onset, length=length,
                shift=megawatts / capacity_mw[neighbour], floor=floored,
            )  # fmt: skip
        out_events.append(
            {
                "host": e.primary, "onset": e.onset, "length": length,
                "kind": "temporary" if temporary else "permanent", "neighbour": neighbour,
                "share": float(share) if mirrored else 0.0, "megawatts": float(megawatts),
                "floored": floored, "chance": bool(mirrored and not temporary), "shift": e.shift,
            }
        )  # fmt: skip
    return MirrorWorld(history=history, target=target, events=out_events)


def describe(
    *, world: MirrorWorld, target_range: tuple[int, int], clean: np.ndarray
) -> dict[str, float]:
    """Return the shares that design section 5.3 states: the kinds, the mirror classes, and the shifted target share."""
    events = world.events
    n = len(events)
    temporary = [e for e in events if e["kind"] == "temporary"]
    permanent = [e for e in events if e["kind"] == "permanent"]
    lo, hi = target_range
    finite = np.isfinite(clean[:, lo:hi])
    shifted = (np.abs(world.target[:, lo:hi] - clean[:, lo:hi]) > 1e-9) & finite
    return {
        "n_events": n,
        "temporary_share": len(temporary) / n,
        "permanent_share": len(permanent) / n,
        "temporary_mirrored_share": sum(e["neighbour"] >= 0 for e in temporary)
        / max(len(temporary), 1),
        "temporary_hidden_share": sum(e["neighbour"] < 0 for e in temporary)
        / max(len(temporary), 1),
        "permanent_chance_mirror_share": sum(e["chance"] for e in permanent)
        / max(len(permanent), 1),
        "target_shifted_share": float(shifted.sum() / max(finite.sum(), 1)),
    }


# ---------------------------------------------------------------------- synthetic worlds W3 and W3p


@dataclass
class W3:
    """A synthetic world W3 (shares of capacity), shape (16, G); only the paired Primaries are filled.

    Attributes:
        y: The power, steps and mirrors included, plus noise.
        base: The power without any step or mirror (the counterfactual inputs).
        shift: Each Primary's own steps.
        mirror: The mirrored steps that each Primary receives from its partner.
        kind: (16, G) int, 0 outside a step, 1 inside a temporary step, 2 inside a permanent step
            of the Primary's own.
        onset: (16, G) grid index where the Primary's own step covering the half-hour began, -1 if none.
        end: (16, G) grid index one past the last half-hour of that step, -1 if none.
    """

    y: np.ndarray
    base: np.ndarray
    shift: np.ndarray
    mirror: np.ndarray
    kind: np.ndarray
    onset: np.ndarray
    end: np.ndarray
    pairs: tuple[tuple[int, int], ...]


def pair_table(*, pairs: tuple[tuple[int, int], ...]) -> nb.NeighbourTable:
    """Return a neighbour table whose only links are the `pairs`, with every Primary signed-MW under one BSP per pair."""
    neighbours: list[tuple[int, ...]] = [() for _ in range(nb.N_PRIMARIES)]
    bsp = list(range(100, 100 + nb.N_PRIMARIES))
    for a, b in pairs:
        neighbours[a], neighbours[b] = (b,), (a,)
        bsp[a] = bsp[b] = 1 + min(a, b)
    return nb.NeighbourTable(
        neighbours=tuple(neighbours), bsp=tuple(bsp), is_mw=(True,) * nb.N_PRIMARIES
    )


def build_w3(
    *, base: np.ndarray, pairs: tuple[tuple[int, int], ...], first_valid: int, share_range: tuple[float, float] = (1.0, 1.0),
    seed: int = W3_SEED,
) -> W3:  # fmt: skip
    """Return world W3 on top of `base` (16, G), the noise-free baseline power over capacity.

    Each paired Primary draws steps of +-0.30 starting at random half-hours until they cover about 15%
    of `[first_valid, G)`. Half are temporary (4 to 12 days) and mirrored at the partner with the
    opposite sign, a share drawn uniformly from `share_range` of the same megawatts (equal capacities,
    so the same share of capacity), and half are permanent (45 days) with no mirror.
    """
    n_grid = base.shape[1]
    rng = np.random.default_rng([seed, 3])
    shift = np.zeros(base.shape)
    mirror = np.zeros(base.shape)
    kind = np.zeros(base.shape, dtype=np.int64)
    onset = np.full(base.shape, -1, dtype=np.int64)
    end = np.full(base.shape, -1, dtype=np.int64)
    partner = dict(pairs) | {b: a for a, b in pairs}
    for p in sorted(partner):
        covered = 0
        while covered < W3_SHARE * (n_grid - first_valid):
            temporary = bool(rng.random() < 0.5)
            days = (
                int(rng.integers(W3_TEMPORARY_DAYS[0], W3_TEMPORARY_DAYS[1] + 1))
                if temporary
                else W3_PERMANENT_DAYS
            )
            length = days * PER_DAY
            on = int(rng.integers(first_valid, n_grid - length))
            sign = float(rng.choice([-1.0, 1.0]))
            share = (
                float(rng.uniform(*share_range))
                if share_range[0] != share_range[1]
                else share_range[0]
            )
            shift[p, on : on + length] += sign * W3_STEP
            kind[p, on : on + length] = 1 if temporary else 2
            onset[p, on : on + length] = on
            end[p, on : on + length] = on + length
            if temporary:
                mirror[partner[p], on : on + length] += -sign * W3_STEP * share
            covered += length
    noise = rng.normal(0.0, NOISE_SD, size=base.shape)
    filled = np.isin(np.arange(base.shape[0]), sorted(partner))
    y = np.where(filled[:, None], base + shift + mirror + noise, np.nan)
    return W3(
        y=y, base=base, shift=shift, mirror=mirror, kind=kind, onset=onset, end=end, pairs=pairs
    )


def follow_share_w3(
    *, diff: np.ndarray, idx: np.ndarray, prim: np.ndarray, c_idx: np.ndarray, world: W3,
) -> dict[str, float]:  # fmt: skip
    """Return the follow share after permanent and after temporary steps, and their gap (design section 8.1).

    A target token qualifies if it lies in a step of its own Primary's (`kind` 1 or 2 at the origin)
    whose onset is 3 to 10 days before the origin, with the step still in force at the origin, and the
    target half-hour is at least 13 days after the onset. The follow share of a token is `diff` (the
    trained median on W3 minus the trained median on the same inputs with the step removed from the
    host's history and the mirror from the partner's) times the step's sign, over the step size.
    """
    o_on = world.onset[prim, c_idx - 1]
    o_kind = world.kind[prim, c_idx - 1]
    age_days = (c_idx - o_on) / PER_DAY
    in_window = (o_kind > 0) & (age_days >= 3) & (age_days <= 10)
    far = (idx - o_on[:, None]) >= 13 * PER_DAY
    qualifies = in_window[:, None] & far
    sign = np.sign(world.shift[prim, o_on.clip(0)])[:, None]
    share = diff * sign / W3_STEP
    out: dict[str, float] = {}
    for name, value in (("permanent", 2), ("temporary", 1)):
        sel = qualifies & (o_kind == value)[:, None]
        out[name] = float(share[sel].mean()) if sel.any() else float("nan")
        out[f"n_{name}"] = int(sel.sum())
    out["gap"] = out["permanent"] - out["temporary"]
    return out

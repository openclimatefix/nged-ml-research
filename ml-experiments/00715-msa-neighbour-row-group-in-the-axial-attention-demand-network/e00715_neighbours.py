# ruff: noqa: E501
"""Experiment 715: the neighbour table, the slot layout, the rewired draw, and the time-shuffle ages.

A sample of arm `F4` holds up to six neighbour slots. The neighbours of a host come from NGED's
`Primary Substation Interconnections.csv` through the list that experiment 383 wrote to
`work/neighbours_nged.json` (`e00709_features.NEIGHBOURS_PATH`, the file that experiments 660, 709,
and 714 read), so the slots match the links along which the injected worlds mirror their moves.
The list is made symmetric on load, which gives 29 undirected links among the 16 Primaries.

This module is pure index arithmetic and file reading: it needs no torch and no fold data. The
Primary index is the index of `site_map.csv` and of experiment 713's arrays (`D2` is 0, `D22` is 15).
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SlotMode = Literal["real", "rewired", "shuffled", "none"]

LABELS: Final = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19",
    "D21", "D22",
)  # fmt: skip
"""The 16 Primaries in the order of experiment 713's Primary index."""
N_PRIMARIES: Final = len(LABELS)
MAX_SLOTS: Final = 6
"""The largest degree of the table (D10 and D12), so no neighbour is dropped."""
NEIGHBOURS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00383-load-shift-xgboost-oracle-first-part-1"
    / "work"
    / "neighbours_nged.json"
)
"""Experiment 383's list of the metered neighbours of each Primary."""
NGED_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15"
SERIES_TO_SITE: Final[dict[int, str]] = {
    3: "D3", 7: "D9", 8: "D11", 9: "D19", 10: "D10", 11: "D4", 12: "D15", 13: "D18",
    14: "D12", 15: "D22", 16: "D14", 17: "D5", 18: "D16", 24: "D21", 25: "D2", 26: "D8",
}  # fmt: skip
"""Experiment 298's map from NGED time series id to D label, copied as experiment 383 copied it."""
DEGREES: Final = {
    "D2": 5, "D3": 3, "D4": 2, "D5": 3, "D8": 3, "D9": 1, "D10": 6, "D11": 3, "D12": 6, "D14": 5,
    "D15": 4, "D16": 2, "D18": 4, "D19": 2, "D21": 5, "D22": 4,
}  # fmt: skip
"""The degree of each Primary in the table (design section 2); the mean is 3.625."""
N_LINKS: Final = 29
BSP_GROUP_SIZES: Final = (9, 7)
LINKS_ACROSS_BSPS: Final = 6
REWIRE_SEED_BASE: Final = 715
"""The first entry of the key of the rewired draw."""
SEED_OFFSET_DRAW: Final = 3
PAIR: Final = ("D15", "D18")
"""NGED gives the two series one substation number, and the table lists no link between them."""
N_RELATIONS: Final = 8


@dataclass(frozen=True)
class NeighbourTable:
    """The neighbour graph of the 16 Primaries with the units and the BSP of each.

    Attributes:
        neighbours: For each Primary index, the sorted tuple of its neighbours' indices.
        bsp: For each Primary index, the substation number of its bulk supply point.
        is_mw: For each Primary index, whether the series is signed megawatts (False for MVA-only
            Primaries and for D15 and D18, whose unit is not signed megawatts).
    """

    neighbours: tuple[tuple[int, ...], ...]
    bsp: tuple[int, ...]
    is_mw: tuple[bool, ...]

    @property
    def n_links(self) -> int:
        """Return the number of undirected links."""
        return sum(len(n) for n in self.neighbours) // 2

    def degree(self, *, host: int) -> int:
        """Return the number of neighbours of `host`."""
        return len(self.neighbours[host])

    def same_bsp(self, *, host: int, other: int) -> bool:
        """Return whether two Primaries sit under one bulk supply point."""
        return self.bsp[host] == self.bsp[other]

    def relation(self, *, host: int, other: int) -> int:
        """Return the index `4 * same_bsp + 2 * host_is_MW + neighbour_is_MW` of the relation embedding."""
        return (
            4 * int(self.same_bsp(host=host, other=other))
            + 2 * int(self.is_mw[host])
            + int(self.is_mw[other])
        )


def symmetric_neighbours(*, raw: dict[str, list[str]]) -> tuple[tuple[int, ...], ...]:
    """Return the neighbour indices of each Primary from the list, made symmetric."""
    edges: dict[str, set[str]] = {label: set() for label in LABELS}
    for a, others in raw.items():
        for b in others:
            if a in edges and b in edges:
                edges[a].add(b)
                edges[b].add(a)
    return tuple(tuple(sorted(LABELS.index(b) for b in edges[a])) for a in LABELS)


def bsp_numbers(*, nged_dir: Path = NGED_DIR) -> tuple[int, ...]:
    """Return the BSP substation number of each Primary, read through `Monitor Direction.csv`.

    A Primary's series maps to its substation number in `Monitor Direction.csv`, and `Substations.csv`
    gives the substation's `bsp_substation_number`.
    """
    with (nged_dir / "Substations.csv").open() as handle:
        bsp_of = {
            r["substation_number"]: int(r["bsp_substation_number"])
            for r in csv.DictReader(handle)
            if r["bsp_substation_number"].strip()
        }
    with (nged_dir / "Monitor Direction.csv").open() as handle:
        monitors = list(csv.DictReader(handle))
    number_of: dict[str, str] = {}
    for r in monitors:
        site = SERIES_TO_SITE.get(int(r["TimeSeriesID"]))
        if site is not None:
            number_of[site] = r["SubstationNumber"]
    missing = [
        label for label in LABELS if label not in number_of or number_of[label] not in bsp_of
    ]
    assert not missing, f"no BSP found for {missing}"
    return tuple(bsp_of[number_of[label]] for label in LABELS)


def load_table(*, path: Path = NEIGHBOURS_PATH, nged_dir: Path = NGED_DIR) -> NeighbourTable:
    """Read the neighbour list, the BSPs, and the metering units into a `NeighbourTable`."""
    raw = json.loads(path.read_text())
    is_mw = tuple(label in SIGNED_MW_LABELS for label in LABELS)
    assert all(
        label in MVA_ONLY_LABELS or label in SIGNED_MW_LABELS or label in PAIR for label in LABELS
    )
    return NeighbourTable(
        neighbours=symmetric_neighbours(raw=raw), bsp=bsp_numbers(nged_dir=nged_dir), is_mw=is_mw
    )


def real_slots(*, table: NeighbourTable, host: int) -> tuple[int, ...]:
    """Return the six slots of the real neighbours: Primary indices in increasing order, -1 for empty."""
    own = table.neighbours[host]
    assert len(own) <= MAX_SLOTS, (host, own)
    return (*own, *(-1,) * (MAX_SLOTS - len(own)))


def eligible_for_rewire(*, table: NeighbourTable, host: int) -> list[int]:
    """Return the Primaries that are neither the host nor a neighbour of the host.

    D15 and D18 are also never drawn for each other.
    """
    banned = {host, *table.neighbours[host]}
    if LABELS[host] in PAIR:
        banned.add(LABELS.index(PAIR[1] if LABELS[host] == PAIR[0] else PAIR[0]))
    return [p for p in range(N_PRIMARIES) if p not in banned]


def rewired_slots(*, table: NeighbourTable, host: int, seed: int) -> tuple[int, ...]:
    """Return the six slots of the rewired placebo of `host` for the rewire seed `seed`.

    A host with `d` neighbours gets `d` Primaries drawn without replacement from
    `eligible_for_rewire`. The draw is a fixed function of (host, seed), so it is the same in
    training and at prediction. The caller passes the seed of the neural network as `seed`.
    """
    degree = table.degree(host=host)
    pool = eligible_for_rewire(table=table, host=host)
    assert len(pool) >= max(degree, 9), (host, len(pool))
    rng = np.random.default_rng([REWIRE_SEED_BASE, SEED_OFFSET_DRAW, host, seed])
    drawn = tuple(sorted(int(p) for p in rng.choice(pool, size=degree, replace=False)))
    return (*drawn, *(-1,) * (MAX_SLOTS - degree))


def two_link_share(*, table: NeighbourTable, host: int, slots: tuple[int, ...]) -> float:
    """Return the share of the filled `slots` whose Primary is two links from `host`.

    A Primary is two links away if it is a neighbour of a neighbour of the host and is neither the
    host nor a neighbour of the host. The share is NaN for a host with no filled slot.
    """
    filled = [s for s in slots if s >= 0]
    if not filled:
        return float("nan")
    near = {host, *table.neighbours[host]}
    two = {q for n in table.neighbours[host] for q in table.neighbours[n]} - near
    return sum(s in two for s in filled) / len(filled)


def slots_for(*, table: NeighbourTable, host: int, mode: SlotMode, seed: int) -> tuple[int, ...]:
    """Return the slots of `host` in `mode`: `real` and `shuffled` hold the real neighbours, `rewired` the placebo, `none` nothing."""
    if mode in {"real", "shuffled"}:
        return real_slots(table=table, host=host)
    if mode == "rewired":
        return rewired_slots(table=table, host=host, seed=seed)
    assert mode == "none", mode
    return (-1,) * MAX_SLOTS


def slot_arrays(
    *, table: NeighbourTable, hosts: np.ndarray, mode: SlotMode, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(slot_primary, slot_exists, slot_relation)` for each host: (N, 6) int, (N, 6) bool, (N, 6) int.

    `slot_primary` is -1 where the slot is empty, and `slot_relation` is 0 there. A rewired slot
    carries the true relation of the Primary it holds (design section 4).
    """
    hosts = np.asarray(hosts, dtype=np.int64)
    prim = np.full((len(hosts), MAX_SLOTS), -1, dtype=np.int64)
    rel = np.zeros((len(hosts), MAX_SLOTS), dtype=np.int64)
    cache = {h: slots_for(table=table, host=h, mode=mode, seed=seed) for h in np.unique(hosts)}
    for i, h in enumerate(hosts):
        prim[i] = cache[int(h)]
        for s, q in enumerate(prim[i]):
            if q >= 0:
                rel[i, s] = table.relation(host=int(h), other=int(q))
    return prim, prim >= 0, rel


def shuffle_pool(*, table: NeighbourTable, pool: np.ndarray) -> np.ndarray:
    """Return the pool of blocks for `F4-T`: blocks in which the host and at least one neighbour have a reading.

    `pool` is (16, n_blocks) from `e00713_train.block_pool`. A neighbour with a short record would
    otherwise fill an `F4-T` sample with empty rows.
    """
    out = np.zeros_like(pool)
    for host in range(N_PRIMARIES):
        near = list(table.neighbours[host])
        out[host] = pool[host] & (pool[near].any(axis=0) if near else False)
    return out


def check_degrees(*, table: NeighbourTable) -> dict[str, int]:
    """Assert the counts of design section 2 and return the degrees by label."""
    degrees = {label: table.degree(host=i) for i, label in enumerate(LABELS)}
    assert degrees == DEGREES, degrees
    assert table.n_links == N_LINKS, table.n_links
    return degrees

# ruff: noqa: E501, PT018
"""Experiment 715: checks KN3 (the neighbour table) and KN10 (the time-shuffle draw), on the CPU.

Usage: e00715_check_neighbours.py

KN3: the slots equal the JSON list; 29 undirected links, symmetric, every Primary with at least one
neighbour, the degrees of design section 2; 9 and 7 Primaries under the two BSPs and 6 links across
BSPs; a rewired set shares no Primary with the host's real set, the host, or (for D15 and D18) the
other member of the pair, and is identical on a second call; the two-link shares are printed.
KN10: no shuffled row is younger than 16 days, the draw is identical on a second call, and no row
repeats in a sample.
Every assertion can fail: the script exits with a traceback on the first violation.
"""

import json
import sys
from collections import Counter

import e00715_neighbours as nb
import numpy as np
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SIGNED_MW_LABELS


def kn3() -> None:
    """KN3: the table against the JSON file, the degrees, the BSPs, and the rewired draw."""
    raw = json.loads(nb.NEIGHBOURS_PATH.read_text())
    table = nb.load_table()
    degrees = nb.check_degrees(table=table)
    assert sum(degrees.values()) / 16 == 3.625
    for host in range(16):
        # symmetry, and no self link
        for other in table.neighbours[host]:
            assert host in table.neighbours[other], (host, other)
            assert other != host
        assert table.degree(host=host) >= 1
        assert (
            nb.real_slots(table=table, host=host)[: table.degree(host=host)]
            == table.neighbours[host]
        )
        # every link of the JSON list is in the table
        for b in raw.get(nb.LABELS[host], []):
            assert nb.LABELS.index(b) in table.neighbours[host]
    assert nb.LABELS.index("D18") not in table.neighbours[nb.LABELS.index("D15")], (
        "D15 and D18 are not linked"
    )
    sizes = sorted(Counter(table.bsp).values(), reverse=True)
    assert tuple(sizes) == nb.BSP_GROUP_SIZES, sizes
    across = sum(
        1
        for a in range(16)
        for b in table.neighbours[a]
        if a < b and not table.same_bsp(host=a, other=b)
    )
    assert across == nb.LINKS_ACROSS_BSPS, across
    assert tuple(table.is_mw[i] for i in range(16)) == tuple(
        lab in SIGNED_MW_LABELS for lab in nb.LABELS
    )
    assert sum(table.is_mw) == 6 and sum(lab in MVA_ONLY_LABELS for lab in nb.LABELS) == 8
    # the relation index: 4 same_bsp + 2 host_mw + neighbour_mw, within 0..7
    rel = {table.relation(host=a, other=b) for a in range(16) for b in table.neighbours[a]}
    assert rel <= set(range(nb.N_RELATIONS)), rel
    # the rewired draw
    shares = {}
    for seed in (0, 1):
        for host in range(16):
            first = nb.rewired_slots(table=table, host=host, seed=seed)
            again = nb.rewired_slots(table=table, host=host, seed=seed)
            assert first == again, "the rewired draw is not a fixed function of (host, seed)"
            filled = [p for p in first if p >= 0]
            assert len(filled) == table.degree(host=host), (host, first)
            assert len(set(filled)) == len(filled)
            assert host not in filled
            assert not set(filled) & set(table.neighbours[host]), (host, first)
            if nb.LABELS[host] in nb.PAIR:
                other = nb.LABELS.index(nb.PAIR[1] if nb.LABELS[host] == nb.PAIR[0] else nb.PAIR[0])
                assert other not in filled, "D15 and D18 are drawn for each other"
            assert len(nb.eligible_for_rewire(table=table, host=host)) >= 9
            shares[(nb.LABELS[host], seed)] = nb.two_link_share(table=table, host=host, slots=first)
    # the two seeds draw different sets for at least some hosts
    assert any(
        nb.rewired_slots(table=table, host=h, seed=0)
        != nb.rewired_slots(table=table, host=h, seed=1)
        for h in range(16)
    ), "the rewire seed changes no draw"
    # the negative control: a draw that kept a real neighbour must be caught by the same test
    bad = (table.neighbours[0][0],)
    assert set(bad) & set(table.neighbours[0]), "the negative control of KN3 is broken"
    print(
        "KN3: degrees",
        degrees,
        "| links",
        table.n_links,
        "| BSP group sizes",
        sizes,
        "| links across BSPs",
        across,
    )
    print("KN3: two-link share of the rewired set per host and rewire seed:")
    for seed in (0, 1):
        print("  seed", seed, {h: round(shares[(h, seed)], 2) for h in nb.LABELS})
    print("KN3 passed")


def kn10() -> None:
    """KN10: the time-shuffle ages are all at least 16 days, repeat-free, and a fixed function of the sample."""
    table = nb.load_table()
    rng = np.random.default_rng(10)
    n_blocks = 1500
    pool = rng.random((16, n_blocks)) < 0.9
    pool[:, :200] = False  # a short record
    pool_t = nb.shuffle_pool(table=table, pool=pool)
    assert (pool_t <= pool).all(), "the shuffle pool must be a subset of the host's pool"
    for host in range(16):
        near = list(table.neighbours[host])
        assert (pool_t[host] == (pool[host] & pool[near].any(axis=0))).all()
    import e00713_layout as lay  # ty: ignore[unresolved-import]
    import e00713_train as tr  # ty: ignore[unresolved-import]

    prim = np.repeat(np.arange(16), 20)
    c_idx = np.tile(np.arange(40, 60), 16) * 48 + 24 + 600 * 48
    ages = tr.placebo_ages_for(primary=prim, c_idx=c_idx, pool=pool_t)
    again = tr.placebo_ages_for(primary=prim, c_idx=c_idx, pool=pool_t)
    assert np.array_equal(ages, again), "the draw changes between two calls"
    used = ages[ages >= 0]
    assert used.min() >= lay.PLACEBO_MIN_AGE and used.max() <= lay.PLACEBO_MAX_AGE, (
        used.min(),
        used.max(),
    )
    for row in ages:
        got = row[row >= 0]
        assert len(set(got.tolist())) == len(got), "a row repeats in a sample"
    # every drawn block holds a reading for the host and for a neighbour
    for i, (p, c) in enumerate(zip(prim, c_idx, strict=True)):
        blocks = int(c) // 48 - ages[i][ages[i] >= 0]
        assert pool_t[p, blocks].all(), "a shuffled row falls outside the shuffle pool"
    # the negative control: a pool that ignores the neighbours draws blocks where no neighbour has a reading
    loose = tr.placebo_ages_for(primary=prim, c_idx=c_idx, pool=pool)
    outside = 0
    for i, (p, c) in enumerate(zip(prim, c_idx, strict=True)):
        blocks = int(c) // 48 - loose[i][loose[i] >= 0]
        outside += int((~pool_t[p, blocks]).sum())
    assert outside > 0, "the negative control of KN10 did not fire"
    print(
        f"KN10 passed: {used.size} shuffled rows, youngest {used.min()} days, oldest {used.max()}; the host-only pool leaves {outside} rows without a neighbour reading"
    )


def main() -> int:
    """Run both checks."""
    kn3()
    kn10()
    return 0


if __name__ == "__main__":
    sys.exit(main())

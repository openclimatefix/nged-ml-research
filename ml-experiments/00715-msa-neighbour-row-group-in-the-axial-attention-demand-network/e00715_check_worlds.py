# ruff: noqa: E501, PT018, PLR0915
"""Experiment 715: checks of the mirror world (design section 5.3) on synthetic arrays, on the CPU.

Usage: e00715_check_worlds.py

The schedule equals experiment 581's `synthetic_clean` (the same onsets, depths, and signs as `M*`
sees); the kinds, mirror classes, and shifted target share match section 5.3 (70% and 30%; two thirds
of the temporary moves mirrored, one third hidden; 1 in 10 permanent changes with a chance mirror;
about 6.5% of target half-hours shifted); each mirror conserves the moved megawatts times the mirrored
share; a mirrored step sits in the neighbour's history and never in its target; a permanent change sits
in the host's history and target. W3's formula and follow share are checked in `e00715_known_answer.py`.
"""

import sys

import e00581_bank as bank581  # ty: ignore[unresolved-import]
import e00715_neighbours as nb
import e00715_worlds as wd
import numpy as np


def main() -> int:
    """Run the checks."""
    table = nb.load_table()
    n_grid = 48 * 1200
    rng = np.random.default_rng(0)
    clean = 0.5 + 0.1 * rng.standard_normal((16, n_grid))
    clean[rng.random((16, n_grid)) < 0.02] = np.nan
    capacity = rng.uniform(3.0, 18.0, 16)
    mva = np.array(
        [lab in {"D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"} for lab in nb.LABELS]
    )
    first, last, seed = 48 * 100, n_grid - 48 * 30, 202601
    # 1. the schedule equals experiment 581's
    events = wd.schedule(shape=clean.shape, first=first, last=last, seed=seed)
    ours = wd.apply_all(clean=clean, events=events)
    theirs, n_events = bank581.synthetic_clean(clean=clean, first=first, last=last, seed=seed)
    assert n_events == len(events) and np.array_equal(ours, theirs, equal_nan=True), (
        "the schedule differs from experiment 581's"
    )
    # 2. the mirror world
    world = wd.mirror_world(
        clean=clean, first=first, last=last, seed=seed, month_key=202601, table=table,
        capacity_mw=capacity, mva_only=mva,
    )  # fmt: skip
    again = wd.mirror_world(
        clean=clean, first=first, last=last, seed=seed, month_key=202601, table=table,
        capacity_mw=capacity, mva_only=mva,
    )  # fmt: skip
    assert np.array_equal(world.history, again.history, equal_nan=True), (
        "the mirror world is not a fixed function of its seeds"
    )
    shares = wd.describe(world=world, target_range=(first, last), clean=clean)
    print("mirror world:", {k: round(v, 3) for k, v in shares.items()})
    assert abs(shares["temporary_share"] - 0.70) < 0.10, shares
    assert abs(shares["temporary_mirrored_share"] - 2 / 3) < 0.12, shares
    assert abs(shares["temporary_hidden_share"] - 1 / 3) < 0.12, shares
    assert shares["permanent_chance_mirror_share"] < 0.25, shares
    assert 0.03 < shares["target_shifted_share"] < 0.11, shares
    # the same onsets, depths, and signs as the schedule
    assert [(e["host"], e["onset"], e["shift"]) for e in world.events] == [
        (e.primary, e.onset, e.shift) for e in events
    ]
    # 3. one isolated event of each kind: the history, the target, and the megawatt balance
    for kind in ("temporary", "permanent"):
        iso = np.full((16, 48 * 200), 0.5)
        host, nbr_idx = 6, table.neighbours[6][0]  # D10 and its first neighbour
        history, target = iso.copy(), iso.copy()
        length = 7 * 48 if kind == "temporary" else wd.PERMANENT_DAYS * 48
        shift, share = -0.30, 0.8
        wd.add_shift(array=history, primary=host, onset=1000, length=length, shift=shift)
        if kind == "permanent":
            wd.add_shift(array=target, primary=host, onset=1000, length=length, shift=shift)
        mw = -shift * capacity[host] * share
        wd.add_shift(
            array=history,
            primary=nbr_idx,
            onset=1000,
            length=length,
            shift=mw / capacity[nbr_idx],
            floor=False,
        )
        moved_host_mw = (history[host] - iso[host]) * capacity[host]
        moved_nbr_mw = (history[nbr_idx] - iso[nbr_idx]) * capacity[nbr_idx]
        assert np.allclose(moved_host_mw[1000 : 1000 + length], shift * capacity[host]), (
            "the host's megawatts are wrong"
        )
        assert np.allclose(
            moved_nbr_mw[1000 : 1000 + length], -share * moved_host_mw[1000 : 1000 + length]
        ), "a mirror does not conserve the moved megawatts times the mirrored share"
        assert np.allclose(history[host, :1000], 0.5) and np.allclose(
            history[host, 1000 + length :], 0.5
        )
        assert np.array_equal(target[nbr_idx], iso[nbr_idx]), (
            "a mirrored step reached the neighbour's target"
        )
        assert (np.array_equal(target[host], iso[host])) == (kind == "temporary")
    # in the full world: no target array carries a mirror, and a temporary move never reaches the target
    temp_hosts = {e["host"] for e in world.events if e["kind"] == "temporary"}
    perm = [e for e in world.events if e["kind"] == "permanent"]
    differs_from_clean = np.abs(np.nan_to_num(world.target - clean)) > 1e-12
    expected = np.zeros(clean.shape, bool)
    for e in perm:
        expected[e["host"], e["onset"] : e["onset"] + e["length"]] = True
    assert (differs_from_clean & ~expected).sum() == 0, (
        "the target holds a shift that is not a permanent change"
    )
    assert temp_hosts, "no temporary event"
    # an MVA-only neighbour is floored at zero
    floored = [e for e in world.events if e["floored"]]
    assert floored, "no floored mirror in the draw"
    assert np.nanmin(world.history[mva][:, :]) > -2.0
    print(
        f"mirror world checks passed: {len(world.events)} events, {len(floored)} mirrors floored at an MVA-only neighbour"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

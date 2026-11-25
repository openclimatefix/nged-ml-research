# ruff: noqa: E402, PT018, E501
"""Tests of experiment 728's training world and scorer. Synthetic data only: no panel, no network.

Run: `PYTHONPATH=<frozen 713 folder>:<the folders of experiments 581, 660, 709, 714, 715, 728>
pytest ml-experiments/00728-*/test_e00728.py --run-studies` (the repository's `run.sh` sets the path).
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
for name in ("00581-", "00660-", "00709-", "00714-", "00715-"):
    for folder in HERE.parent.glob(f"{name}*"):
        sys.path.insert(0, str(folder))
sys.path.insert(0, str(HERE))

import e00715_neighbours as nb  # ty: ignore[unresolved-import]
import e00728_score as sc
import e00728_worlds as wd
import numpy as np
import polars as pl
import pytest

N_GRID = 48 * 900
FIRST, LAST = 48 * 100, 48 * 800


def ring_table() -> nb.NeighbourTable:
    """Return a table in which each Primary neighbours the two before and the two after it."""
    near = tuple(
        tuple(sorted({(i + d) % nb.N_PRIMARIES for d in (-2, -1, 1, 2)}))
        for i in range(nb.N_PRIMARIES)
    )
    return nb.NeighbourTable(
        neighbours=near, bsp=tuple(range(nb.N_PRIMARIES)), is_mw=(True,) * nb.N_PRIMARIES
    )


def small_world(*, seed: int) -> tuple[wd.TrainingWorld, np.ndarray, np.ndarray]:
    """Return a training draw on a flat synthetic panel, with the panel and the MVA-only flags."""
    rng = np.random.default_rng(0)
    clean = 0.5 + 0.05 * rng.standard_normal((nb.N_PRIMARIES, N_GRID))
    mva = np.arange(nb.N_PRIMARIES) % 2 == 0
    world = wd.training_world(
        clean=clean, first=FIRST, last=LAST, world_seed=seed, table=ring_table(),
        capacity_mw=np.full(nb.N_PRIMARIES, 30.0), mva_only=mva, vis=np.full(nb.N_PRIMARIES, 0.12),
    )  # fmt: skip
    return world, clean, mva


def test_training_world_limits_and_clean_targets() -> None:
    """GH7 holds on a draw: clean targets, at most two permanent changes, a stack within 0.50."""
    world, clean, mva = small_world(seed=1)
    stats = wd.check_targets_and_limits(world=world, clean=clean, mva_only=mva)
    assert stats["n_events"] > 100 and 0.0 < stats["permanent_share"] < 0.6
    cover = wd.realised_cover(world=world, clean=clean, first=FIRST, last=LAST)
    assert 0.08 < cover < 0.45, cover


def test_training_world_is_deterministic_and_seeded() -> None:
    """The same seed gives the same arrays (GH9), and another seed another draw."""
    a, _, _ = small_world(seed=3)
    b, _, _ = small_world(seed=3)
    c, _, _ = small_world(seed=4)
    wd.check_shared_draw(host_a=a.history, host_b=b.history)
    assert not np.array_equal(a.history, c.history, equal_nan=True)


def test_mirror_classes_and_sign_balance() -> None:
    """A third of temporary moves have each mirror class, and the sign does not depend on the kind."""
    world, _, _ = small_world(seed=5)
    temporary = [e for e in world.events if e["kind"] == "temporary"]
    classes = np.array([min(e["mirror_class"], 2) for e in temporary])
    shares = np.bincount(classes, minlength=3) / len(classes)
    assert np.all(np.abs(shares - 1 / 3) < 0.12), shares
    wd.independence_tables(world=world)


def test_visible_depth_and_seed_disjointness() -> None:
    """The visible depth is the larger of 0.10 and twice the error; seed lists are disjoint."""
    vis = wd.visible_depth(reference_error_share=np.array([0.02, 0.08, np.nan]))
    assert np.allclose(vis, [0.10, 0.16, 0.10])
    wd.assert_disjoint(train=[1, 2], g2=[3], held_out=[4])
    with pytest.raises(AssertionError):
        wd.assert_disjoint(train=[1, 2], g2=[2], held_out=[4])


def test_target_index_and_slice() -> None:
    """The target row starts at the contract origin plus 48 per lead day; the slice ends 7 days after the end."""
    idx = sc.target_index(origin_day=10, lead_day=2)
    assert idx[0] == 10 * 48 + 24 + 48 and idx[-1] == idx[0] + 47
    assert sc.counted_slice(onset_day=5, end_day=6) == (5 * 48, 13 * 48)
    assert sc.counted_slice(onset_day=5, end_day=-1)[1] > 10**7
    assert sc.rows_in_slice(onset_day=5, end_day=6) > 0
    assert sc.rows_in_slice(onset_day=5, end_day=-1) == len(sc.ORIGIN_DAYS) * 14 * 48


def test_persistence_index_is_the_newest_visible_same_half_hour() -> None:
    """The persistence index is at or before the newest visible half-hour, a whole number of days back."""
    valid = sc.target_index(origin_day=30, lead_day=3)
    back = sc.persistence_index(valid=valid, origin_day=30)
    assert (back <= sc.newest_visible(origin_day=30)).all()
    assert ((valid - back) % 48 == 0).all()
    assert (back > sc.newest_visible(origin_day=30) - 48).all()


def make_rows(*, truth: str, end_day: int) -> pl.DataFrame:
    """Return the rows of one event with a flat step of 0.2 from day 100."""
    clean = np.full(48 * 400, 0.5)
    injected = clean.copy()
    injected[100 * 48 : 100 * 48 + (end_day - 100 if end_day > 0 else 10**6) * 48] += 0.2
    event = {
        "truth": truth,
        "kind": "flat",
        "mirror_state": "none",
        "onset_day": 100,
        "end_day": end_day,
    }
    return sc.rows_for_event(event_id="e0", primary=0, event=event, injected=injected, clean=clean)


def test_truth_oracle_scores_zero_and_follows_only_permanent_changes() -> None:
    """GH1 on a toy event: zero score, a follow share of 1 for a permanent change, 0 for a move."""
    for truth, end, expect in (("era", -1, 1.0), ("switch", 130, 0.0)):
        rows = make_rows(truth=truth, end_day=end)
        reading = sc.read_oracle(
            injected=sc.truth_oracle(rows=rows, world="injected"),
            real=sc.truth_oracle(rows=rows, world="real"),
        )
        assert reading.max_abs_score < 1e-12
        got = reading.follow_permanent if truth == "era" else reading.follow_temporary
        assert abs(got - expect) < 1e-9, (truth, got)


def test_persistence_oracle_follows_a_move_in_force_and_scores_worse() -> None:
    """GH2 on a toy move that is still in force at every origin: follow share 1, CRPS above 0."""
    rows = make_rows(truth="switch", end_day=160)
    reading = sc.read_oracle(
        injected=sc.persistence_oracle(rows=rows, world="injected"),
        real=sc.persistence_oracle(rows=rows, world="real"),
    )
    assert abs(reading.follow_temporary - 1.0) < 1e-9 and reading.mean_counted_crps > 1.0


def test_event_values_weight_events_equally() -> None:
    """An event with many rows counts as much as an event with few."""
    scored = pl.DataFrame(
        {"event_id": ["a"] * 10 + ["b"], "primary": [0] * 10 + [1], "crps": [1.0] * 10 + [3.0]}
    )
    values = sc.event_values(scored=scored)
    assert values["crps"].mean() == pytest.approx(2.0)

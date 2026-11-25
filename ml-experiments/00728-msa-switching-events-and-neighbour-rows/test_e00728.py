# ruff: noqa: E402, PT018, E501
"""Tests of experiment 728's training world and scorer. Synthetic data only: no panel, no network.

The scorer and the worlds import experiment 713's layout, which is not on the repository's test
path. The module looks for experiment 713's folder in the checkout, then in the frozen checkout next
to the worktrees, and skips the whole file when neither exists.
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
FROZEN_713 = HERE.parents[2] / "frozen-713-7c29114a8" / "ml-experiments"
for root, name in (
    (HERE.parent, "00581-"), (HERE.parent, "00660-"), (HERE.parent, "00709-"),
    (HERE.parent, "00713-"), (HERE.parent, "00714-"), (HERE.parent, "00715-"),
    (FROZEN_713, "00713-"),
):  # fmt: skip
    for folder in root.glob(f"{name}*"):
        sys.path.insert(0, str(folder))
sys.path.insert(0, str(HERE))
pytest.importorskip("e00713_layout", reason="experiment 713's folder is not available")

import e00715_neighbours as nb  # ty: ignore[unresolved-import]
import e00728_score as sc
import e00728_worlds as wd
import numpy as np
import polars as pl

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
    others = [small_world(seed=seed)[0] for seed in (6, 7)]
    mva = np.arange(nb.N_PRIMARIES) % 2 == 0
    tables = wd.independence_tables(worlds=[world, *others], mva_only=mva, level=1e-4)
    assert "kind_by_sign_mva_only" in tables and "kind_by_sign_signed_mw" in tables


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
    clean = np.full(48 * 200, 0.5)
    assert sc.slice_rows(clean=clean, onset_day=5, end_day=6).scorable > 0
    assert (
        sc.slice_rows(clean=clean, onset_day=5, end_day=-1).nominal == len(sc.ORIGIN_DAYS) * 14 * 48
    )


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


def test_history_of_mva_only_primaries_stays_at_or_above_zero() -> None:
    """GH7 on the history: no MVA-only history goes below zero, whatever the kind of event."""
    for seed in (11, 12, 13):
        world, clean, mva = small_world(seed=seed)
        wd.check_targets_and_limits(world=world, clean=clean, mva_only=mva)
        for p in np.nonzero(mva)[0]:
            assert np.nanmin(world.history[p]) >= -1e-9


def test_floor_rule_does_not_tie_group_and_sign_to_kind() -> None:
    """On a panel with low MVA-only power, falls of both kinds are rejected alike at MVA-only hosts."""
    rng = np.random.default_rng(1)
    clean = 0.5 + 0.02 * rng.standard_normal((nb.N_PRIMARIES, N_GRID))
    mva = np.arange(nb.N_PRIMARIES) % 2 == 0
    clean[mva] = 0.2 + 0.02 * rng.standard_normal((int(mva.sum()), N_GRID))
    worlds = [
        wd.training_world(
            clean=clean, first=FIRST, last=LAST, world_seed=s, table=ring_table(),
            capacity_mw=np.full(nb.N_PRIMARIES, 30.0), mva_only=mva, vis=np.full(nb.N_PRIMARIES, 0.12),
        )
        for s in range(20, 26)
    ]  # fmt: skip
    events = pl.DataFrame([e for w in worlds for e in w.events]).with_columns(
        mva=pl.col("host").map_elements(lambda h: bool(mva[h]), return_dtype=pl.Boolean)
    )
    share = (
        events.filter(pl.col("mva"))
        .group_by("kind")
        .agg((pl.col("sign") < 0).mean().alias("falls"))
    )
    falls = dict(zip(share["kind"].to_list(), share["falls"].to_list(), strict=True))
    assert abs(falls["temporary"] - falls["permanent"]) < 0.12, falls


def test_donor_step_recovers_the_donor_from_a_receiving_row() -> None:
    """A row scored at the donor keeps its size; a receiving row is scaled by capacity and share."""
    truth = pl.DataFrame(
        {
            "truth": ["switch", "switch", "era"], "receiving_host": [False, True, False],
            "size": [-0.2, 0.05, 0.3], "share": [0.9, 0.5, None], "host": [0, 1, 2],
            "recipient": [1, 0, -1],
        }
    )  # fmt: skip
    capacity = np.array([40.0, 20.0, 10.0])
    out = wd.donor_step(truth=truth, capacity_mw=capacity)
    assert out[0] == pytest.approx(0.2)
    assert out[1] == pytest.approx(0.05 * 20.0 / (0.5 * 40.0))
    assert np.isnan(out[2])


def make_events() -> pl.DataFrame:
    """Return a small table of events with the columns the counts read."""
    return pl.DataFrame(
        {
            "truth": ["switch"] * 4 + ["era"], "metered": [True, True, True, False, False],
            "donor_step": [0.30, 0.12, 0.05, 0.40, float("nan")],
            "visible": [True, False, False, True, True], "month": ["2025-11"] * 2 + ["2026-01"] * 3,
            "mirror_state": ["full", "partial", "full", "hidden", "none"], "band": ["<60"] * 5,
            "duration_days": [10.0, 3.0, 20.0, 5.0, None],
        }
    )  # fmt: skip


def test_counted_moves_and_floors() -> None:
    """The counted set needs a metered mirror and visible depth; the floor-only set only 0.10."""
    events = make_events()
    assert wd.counted_moves(events=events).height == 1
    assert wd.counted_moves(events=events, floor_only=True).height == 2
    report = wd.count_report(events=events)
    assert (report.all_temporary, report.metered, report.floor_only, report.visible_depth) == (
        4,
        3,
        2,
        1,
    )
    assert not wd.pooled_floor_ok(report=report, pooled=1, per_month=1)  # 2026-01 and others hold 0
    assert not wd.pooled_floor_ok(report=report, pooled=2, per_month=0)


def test_rows_past_the_grid_end_are_nan_and_counted() -> None:
    """A target past the end of the grid has a NaN truth, and the slice count reports it."""
    clean = np.full(48 * 112, 0.5)
    counts = sc.slice_rows(clean=clean, onset_day=100, end_day=-1)
    assert (
        counts.past_grid > 0
        and counts.scorable + counts.past_grid + counts.no_truth == counts.nominal
    )
    event = {
        "truth": "switch",
        "kind": "transfer",
        "mirror_state": "full",
        "onset_day": 100,
        "end_day": 120,
    }
    rows = sc.rows_for_event(
        event_id="x", primary=0, event=event, injected=clean.copy(), clean=clean
    )
    assert rows["clean_share"].is_nan().sum() > 0
    scored = sc.scoring_rows(forecast=sc.truth_oracle(rows=rows, world="injected"), what="counted")
    assert scored.height > 0 and scored["crps"].is_finite().all()


def test_follow_share_ignores_nan_truth() -> None:
    """A NaN median drops the row and does not turn the follow share into NaN."""
    clean = np.full(48 * 112, 0.5)
    injected = clean.copy()
    injected[100 * 48 :] += 0.2
    event = {
        "truth": "era",
        "kind": "flat",
        "mirror_state": "none",
        "onset_day": 100,
        "end_day": -1,
    }
    rows = sc.rows_for_event(event_id="y", primary=0, event=event, injected=injected, clean=clean)
    follow = sc.follow_shares(
        injected=sc.truth_oracle(rows=rows, world="injected"),
        real=sc.truth_oracle(rows=rows, world="real"),
    ).drop_nulls("follow")
    assert follow.height > 0 and follow["follow"].is_finite().all()
    assert follow["follow"].mean() == pytest.approx(1.0)

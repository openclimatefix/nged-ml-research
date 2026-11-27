"""Tests of the per-world outputs that scoring reads without decoding again (experiment 734)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import e00734_hmm as hmm
import e00734_store as store

N_O = hmm.N_O
N_L = hmm.N_L


def point_mass(*, n: int, o: int, level: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Marginals with all mass on offset level `o` and permanent level `level`, `n` blocks."""
    om = np.zeros((n, N_O), dtype=np.float32)
    om[:, hmm.FULL_GRID.o_levels.index(o)] = 1.0
    lm = np.zeros((n, N_L), dtype=np.float32)
    lm[:, hmm.FULL_GRID.l_levels.index(level)] = 1.0
    return om, lm


def blocks_of(*, n: int, o: int, level: int = 0, sigma: float = 2.0) -> store.SiteBlocks:
    """Summaries of a posterior that is a point mass."""
    om, lm = point_mass(n=n, o=o, level=level)
    post = hmm.Posterior(
        o_marg=om,
        l_jump=np.zeros(n),
        up_jump=np.zeros(n),
        down_jump=np.zeros(n),
        loglik=0.0,
        l_marg=lm,
    )
    return store.site_blocks(post=post, sigma=sigma)


def test_block_slice_covers_the_overlapping_blocks_and_open_ends_run_to_the_record_end() -> None:
    """Test a span that starts and ends mid-block covers both partial blocks."""
    assert store.block_slice(start=13, end=37, n_blocks=100) == slice(1, 4)
    assert store.block_slice(start=24, end=None, n_blocks=10) == slice(2, 10)
    assert store.block_slice(start=24, end=-1, n_blocks=10) == slice(2, 10)
    assert store.block_slice(start=24, end=24, n_blocks=10) == slice(2, 3)


def test_block_mean_of_averages_each_block_of_the_window() -> None:
    """Test a step that starts mid-block gives a half-height first block."""
    delta = np.zeros(120)
    delta[6:] = 2.0
    win = store.block_slice(start=6, end=48, n_blocks=10)
    np.testing.assert_allclose(store.block_mean_of(delta=delta, window=win), [1.0, 2.0, 2.0, 2.0])


def test_site_blocks_give_the_posterior_mean_the_median_and_the_probability_of_zero() -> None:
    """Test a point mass at +3 noise units of 2 MW reads 6 MW at every quantile."""
    b = blocks_of(n=5, o=3)
    np.testing.assert_allclose(b.mean_mw, 6.0)
    np.testing.assert_allclose(b.quantiles, 6.0)
    np.testing.assert_allclose(b.p_zero, 0.0)


def test_the_offset_plus_level_quantiles_add_the_two_marginals_on_the_one_unit_grid() -> None:
    """Test an offset of 3 and a level of 4 gives 7 noise units at every quantile."""
    om, lm = point_mass(n=2, o=3, level=4)
    q = store.sum_quantiles(o_marg=om, l_marg=lm, sigma=0.5)
    np.testing.assert_allclose(q, 3.5)


def test_the_window_table_holds_one_row_per_block_with_permanent_columns_only_for_shifts() -> None:
    """Test rows, truth megawatts, and the level quantiles of a permanent shift."""
    n = 40
    blocks = {"A": blocks_of(n=n, o=0, level=6)}
    truths = [
        {"site": "A", "start": 24, "end": 48, "kind": "edge", "truth_mw": np.array([1.0, 2.0])},
        {"site": "A", "start": 24, "end": None, "kind": "permanent_shift", "truth_mw": np.ones(38)},
        {"site": "Z", "start": 24, "end": 48, "kind": "solo", "truth_mw": np.array([1.0, 2.0])},
    ]
    t = store.window_table(world=7, arm="H-nbr", truths=truths, blocks=blocks)
    edge = t.filter(pl.col("kind") == "edge")
    assert edge["block"].to_list() == [2, 3]
    assert edge["truth_mw"].to_list() == [1.0, 2.0]
    assert edge["s50"].null_count() == 2
    assert edge["l_marg"].null_count() == 2
    perm = t.filter(pl.col("kind") == "permanent_shift")
    assert perm.height == 38
    assert perm["s50"].to_list()[0] == pytest.approx(12.0)
    assert len(perm["l_marg"][0]) == N_L
    assert "Z" not in t["site"].to_list()


def test_the_mean_shift_over_a_window_averages_the_posterior_mean_blocks() -> None:
    """Test the mean and the mean absolute shift of a window."""
    b = blocks_of(n=10, o=-2, sigma=1.0)
    out = store.window_mean_shift(
        windows=[{"site": "A", "start": 12, "end": 36, "event_class": "ramp"}], blocks={"A": b}
    )
    assert out["mean_shift_mw"][0] == pytest.approx(-2.0)
    assert out["mean_abs_shift_mw"][0] == pytest.approx(2.0)
    assert out["event_class"][0] == "ramp"


def test_call_shift_columns_sum_the_partners_and_flag_pairs_of_signed_primaries() -> None:
    """Test a donor at -3 and a recipient at +3 conserve, and an MVA partner breaks the pair."""
    blocks = {"A": blocks_of(n=20, o=-3, sigma=1.0), "B": blocks_of(n=20, o=3, sigma=1.0)}
    e = SimpleNamespace(primary="A", start=12, end=60, partners=["B"])
    lone = SimpleNamespace(primary="A", start=12, end=60, partners=[])
    out = store.call_shift_columns(called=[e, lone], blocks=blocks, is_mva={"A": False, "B": False})
    assert out["post_mean_mw"] == [pytest.approx(-3.0)] * 2
    assert out["partner_post_mean_mw"] == [pytest.approx(3.0), None]
    assert out["signed_pair"] == [True, False]
    mva = store.call_shift_columns(called=[e], blocks=blocks, is_mva={"A": False, "B": True})
    assert mva["signed_pair"] == [False]


def test_the_four_scenario_masks_split_quiet_inside_and_after_a_called_event() -> None:
    """Test the masks of an event from half-hour 100 to 200 in a record of 1000 half-hours."""
    m = store.scenario_masks(length=1000, called=[(100, 200)], lo=50)
    assert m["inside called events"].sum() == 100
    assert m["7 days after"].sum() == 7 * 48
    assert not (m["7 days after"] & m["inside called events"]).any()
    assert m["quiet rows"].sum() == 1000 - 50 - 100 - 7 * 48
    assert m["all rows"].sum() == 950
    open_ended = store.scenario_masks(length=1000, called=[(900, None)], lo=0)
    assert open_ended["inside called events"].sum() == 100
    assert open_ended["7 days after"].sum() == 0


def test_the_scenario_error_falls_inside_an_event_when_the_shift_matches_the_step() -> None:
    """Test the error after subtracting a correct shift is zero inside the event."""
    length = 600
    power = np.zeros(length)
    power[100:200] = 4.0
    shift = np.zeros(length)
    shift[100:200] = 4.0
    out = store.scenario_rows(
        world=1,
        arm="H-nbr",
        power={"A": power},
        forecast={"A": np.zeros(length)},
        capacity={"A": 10.0},
        shift_mw={"A": shift},
        called={"A": [(100, 200)]},
        lo=0,
    )
    inside = out.filter(pl.col("scenario") == "inside called events")
    assert inside["nmae_before"][0] == pytest.approx(0.4)
    assert inside["nmae_after"][0] == pytest.approx(0.0)
    quiet = out.filter(pl.col("scenario") == "quiet rows")
    assert quiet["nmae_before"][0] == pytest.approx(0.0)
    assert out.filter(pl.col("scenario") == "all rows")["n"][0] == length


def test_overlap_iou_treats_an_open_end_as_the_record_end() -> None:
    """Test the overlap of spans, one of them open."""
    assert store.overlap_iou(a=(0, 10), b=(5, 15), length=100) == pytest.approx(5 / 15)
    assert store.overlap_iou(a=(0, None), b=(50, -1), length=100) == pytest.approx(0.5)
    assert store.overlap_iou(a=(0, 10), b=(20, 30), length=100) == 0.0


def test_false_events_are_calls_at_grown_primaries_that_overlap_no_uninjected_event() -> None:
    """Test only a late call at a grown Primary with no uninjected partner is false."""
    before = pl.DataFrame({"primary": ["A"], "start": [100], "end": [200]})
    after = [
        SimpleNamespace(primary="A", start=110, end=200),  # overlaps the uninjected event
        SimpleNamespace(primary="A", start=500, end=600),  # false
        SimpleNamespace(primary="B", start=500, end=600),  # not a grown Primary
        SimpleNamespace(primary="A", start=10, end=20),  # before the scored years
    ]
    out = store.false_events(after=after, before=before, grown=["A"], lo=50, length=1000)
    assert [(e.primary, e.start) for e in out] == [("A", 500)]


def test_the_uninjected_table_has_one_row_per_primary_and_block_with_the_level_median() -> None:
    """Test the table's columns and the median of the offset plus the level."""
    t = store.uninjected_table(blocks={"A": blocks_of(n=4, o=1, level=2, sigma=1.0)})
    assert t.height == 4
    assert t["s50"].to_list() == [3.0] * 4
    assert t["q50"].to_list() == [1.0] * 4


def test_the_neighbourhood_share_baseline_removes_a_stable_share_and_keeps_a_short_step() -> None:
    """Test a constant share of the neighbourhood leaves no residual, and a short step shows."""
    rng = np.random.default_rng(0)
    total_other = 10 + rng.normal(0, 0.1, 400)
    own = 0.25 * (own_plus := total_other / 0.75)
    resid = store.neighbourhood_share_residual(own=own, others=[total_other], window=112)
    assert np.nanmax(np.abs(resid)) < 1e-9
    stepped = own.copy()
    stepped[200:230] += 2.0
    r2 = store.neighbourhood_share_residual(own=stepped, others=[total_other], window=112)
    assert np.nanmean(r2[205:225]) > 1.0
    assert own_plus.shape == own.shape


def test_event_key_mismatches_name_the_events_found_in_only_one_table() -> None:
    """Test the comparison on primary, start, and end, not on the count."""
    stored = pl.DataFrame({"primary": ["A", "B"], "start": [1, 2], "end": [5, -1]})
    same = store.event_key_mismatches(mine=[("B", 2, -1), ("A", 1, 5)], stored=stored)
    assert same["equal"]
    other = store.event_key_mismatches(mine=[("A", 1, 5), ("B", 2, 9)], stored=stored)
    assert not other["equal"]
    assert other["n_mine"] == other["n_stored"] == 2
    assert other["only_mine"] == [("B", 2, 9)]
    assert other["only_stored"] == [("B", 2, -1)]


def test_cascaded_keys_hold_the_earliest_step_of_each_cascade_and_primary() -> None:
    """Test the key of a cascade is its first start, and a table without cascades has none."""
    table = pl.DataFrame(
        {
            "cascade_id": [7, 7, None, 9],
            "site": ["A", "A", "A", "B"],
            "start": [300, 100, 500, 40],
        }
    )
    assert store.cascaded_keys(table=table) == {("A", 100), ("B", 40)}
    assert store.cascaded_keys(table=table.drop("cascade_id")) == set()
    assert store.cascaded_keys(table=table.filter(pl.col("cascade_id").is_null())) == set()


def test_blocks_to_half_hours_handles_a_tuning_world_shorter_than_the_record() -> None:
    """Test a 100-block level (a tuning world) maps to 1,200 half-hours, with the 653 centres."""
    level = np.arange(100, dtype=float)
    got = store.blocks_to_half_hours(level=level)
    assert got.shape == (100 * store.BLOCK,)
    # The centre of block 3 is half-hour 3 * 12 + 6 = 42.
    assert got[42] == pytest.approx(3.0)
    # Before the first centre and after the last, the level is held constant.
    assert got[0] == 0.0
    assert got[-1] == 99.0

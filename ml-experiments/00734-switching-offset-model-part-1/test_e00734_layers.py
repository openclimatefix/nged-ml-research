"""Tests of the cascade and cold-spell layers, the world bank guard, and the scoring functions."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import e00660_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import e00734_bank as bank
import e00734_layers as layers
import e00734_score as score

DAY = layers.DAY


def table(*, n_groups: int, with_recipient: bool = True) -> pl.DataFrame:
    """A table of `n_groups` edge-transfer groups (donor and recipient), each 30 days long."""
    rows = []
    for g in range(n_groups):
        for k, (site, role, sign) in enumerate((("A", "donor", -1.0), ("B", "recipient", 1.0))):
            if k == 1 and not with_recipient:
                continue
            rows.append(
                {
                    "event_id": 2 * g + k,
                    "group_id": g,
                    "site": site,
                    "role": role,
                    "event_class": "edge_transfer",
                    "label": "edge_transfer",
                    "start": 1000 + 100 * g,
                    "end": 1000 + 100 * g + 30 * DAY,
                    "residual_frac": 0.0,
                    "delta": sign * (1.0 + g),
                    "units": 5.0,
                    "shape": "constant",
                    "prop_share": 0.0,
                    "shape_source": "A",
                    "ramp_in_hh": 0,
                    "ramp_out_hh": 0,
                    "drift": 0.0,
                    "mid_factor": 1.0,
                    "staged_hh": 0,
                    "share": 1.0,
                    "period_hh": 0.0,
                    "phase": 0.0,
                    "partners": ["B"] if role == "donor" else ["A"],
                    "chain_parent": None,
                }
            )
    return pl.DataFrame(rows, schema=gen.EVENT_COLUMNS)


def test_a_cascade_keeps_the_total_change_and_moves_starts_by_less_than_a_day() -> None:
    """Test 15% of the groups are cascaded into 2 or 3 steps with the same total change."""
    base = table(n_groups=40)
    out = layers.cascade_events(events=base, rng=np.random.default_rng(0))
    chosen = out.filter(pl.col("cascade_id").is_not_null())
    assert chosen["group_id"].n_unique() == 6
    for (cid, _site), g in chosen.group_by(["cascade_id", "site"]):
        orig = base.filter(pl.col("event_id") == cid).row(0, named=True)
        assert g.height in (2, 3)
        assert g["delta"].sum() == pytest.approx(orig["delta"])
        assert (g["delta"] * orig["delta"] > 0).all()
        assert g["start"].min() == orig["start"]
        assert g["start"].max() - orig["start"] <= layers.MAX_OFFSET_HH
        assert set(g["cascade_end"].to_list()) == {orig["end"]}
    assert out["event_id"].n_unique() == out.height
    untouched = out.filter(pl.col("cascade_id").is_null())
    assert untouched.height == base.height - 2 * 6
    assert any(g["end"].n_unique() > 1 for _, g in chosen.group_by(["cascade_id", "site"]))


def test_a_donor_and_its_recipient_are_cascaded_at_the_same_times() -> None:
    """Test every row of a cascaded group takes the same start offsets."""
    out = layers.cascade_events(events=table(n_groups=40), rng=np.random.default_rng(1))
    for _, g in out.filter(pl.col("cascade_id").is_not_null()).group_by("group_id"):
        a = sorted(g.filter(pl.col("site") == "A")["start"].to_list())
        b = sorted(g.filter(pl.col("site") == "B")["start"].to_list())
        assert a == b


def test_cascade_is_deterministic_and_the_merged_table_has_one_row_per_cascade() -> None:
    """Test the same generator gives the same cascade, and merging restores the original rows."""
    base = table(n_groups=20)
    a = layers.cascade_events(events=base, rng=np.random.default_rng(5))
    b = layers.cascade_events(events=base, rng=np.random.default_rng(5))
    assert a.equals(b)
    merged = layers.merged_cascades(events=a).sort("event_id")
    assert merged.height == base.height
    np.testing.assert_allclose(
        merged["delta"].to_numpy(), base.sort("event_id")["delta"].to_numpy()
    )
    assert merged["start"].to_list() == base.sort("event_id")["start"].to_list()
    assert merged["end"].to_list() == base.sort("event_id")["end"].to_list()


def test_a_short_event_is_never_cascaded() -> None:
    """Test a group whose event ends under 2 days after its start is not eligible."""
    base = table(n_groups=10).with_columns(end=pl.col("start") + 60)
    assert layers.eligible_groups(events=base) == []
    out = layers.cascade_events(events=base, rng=np.random.default_rng(0))
    assert out["cascade_id"].null_count() == out.height


def test_the_cascaded_table_rebuilds_a_panel_with_the_same_missing_half_hours() -> None:
    """Test the generator applies a cascaded table, and sums the rows of a group independently."""
    length = 300 * DAY
    t = np.arange(length)
    power = {s: 5.0 + np.sin(2 * np.pi * t / DAY) for s in ("A", "B")}
    power["A"][5000:5100] = np.nan
    panel = gen.PowerPanel(
        power=power,
        is_mva={"A": False, "B": False},
        noise={"A": 0.1, "B": 0.1},
        neighbours={"A": ["B"], "B": ["A"]},
        origin=gen.datetime(2022, 1, 1, tzinfo=gen.UTC),
    )
    base = table(n_groups=40)
    base = base.with_columns(start=pl.col("start") + 1000)
    out = layers.cascade_events(events=base, rng=np.random.default_rng(2))
    cascaded = out.drop("cascade_id", "cascade_end")
    world = gen.apply_events(panel=panel, events=cascaded)
    plain = gen.apply_events(panel=panel, events=base)
    for s in ("A", "B"):
        np.testing.assert_array_equal(np.isnan(world[s]), np.isnan(plain[s]))
    mid = slice(1000 + 40 * 100 + 200, 1000 + 40 * 100 + 300)
    assert np.isfinite(world["A"][mid]).any()


def make_rig() -> SimpleNamespace:
    """A rig stub with a positive forecast at 6 Primaries."""
    sites = [f"S{i}" for i in range(6)]
    forecast = {s: np.full(layers.GRID_LENGTH, 10.0 + i) for i, s in enumerate(sites)}
    return SimpleNamespace(panel=SimpleNamespace(sites=sites), forecast=forecast)


def test_cold_spells_are_nonzero_exactly_inside_the_spells_and_zero_at_two_primaries() -> None:
    """Test the additions sit inside the spells, scale with loading, and vanish at 2 Primaries."""
    rig = make_rig()
    add, spells = layers.cold_spells(rig=rig, rng=np.random.default_rng(3))
    assert spells.height == 5 * layers.COLD_SPELLS_PER_WINTER
    zero_sites = [s for s in rig.panel.sites if not add[s].any()]
    assert len(zero_sites) == layers.N_ZERO_LOADING
    inside = np.zeros(layers.GRID_LENGTH, dtype=bool)
    for r in spells.iter_rows(named=True):
        assert 4 * DAY <= r["end"] - r["start"] <= 14 * DAY or r["end"] == layers.GRID_LENGTH
        assert 0.03 <= r["share"] <= 0.10
        inside[r["start"] + 1 : r["end"]] = True
    for s in rig.panel.sites:
        if s in zero_sites:
            continue
        assert (add[s][~inside] == 0).all()
        assert (add[s] >= 0).all()
        assert (add[s][inside] > 0).mean() > 0.99
    live = [s for s in rig.panel.sites if s not in zero_sites]
    ratio = add[live[0]].max() / (add[live[1]].max() or 1.0)
    assert ratio > 0.0
    spell = spells.row(0, named=True)
    mid = (spell["start"] + spell["end"]) // 2
    assert add[live[0]][mid] == pytest.approx(
        spell["share"] * spell[f"lambda_{live[0]}"] * rig.forecast[live[0]][mid], rel=0.05
    )


def test_cold_spells_use_the_magnitude_at_an_mva_primary() -> None:
    """Test an MVA Primary gets |power + addition| and a missing half-hour stays missing."""
    power = {"A": np.array([1.0, -2.0, np.nan, 3.0]), "B": np.array([1.0, -2.0, 4.0, 3.0])}
    add = {"A": np.array([0.0, 1.0, 1.0, -5.0]), "B": np.array([0.0, 1.0, 1.0, -5.0])}
    out = layers.apply_additions(power=power, additions=add, is_mva={"A": True, "B": False})
    np.testing.assert_array_equal(out["A"][[0, 1, 3]], [1.0, 1.0, 2.0])
    assert np.isnan(out["A"][2])
    np.testing.assert_array_equal(out["B"], [1.0, -1.0, 5.0, -2.0])


def test_the_bank_split_never_lets_a_scoring_seed_reach_the_tuning_code() -> None:
    """Test the tuning guard rejects every off-limits seed and accepts the 30 tuning seeds."""
    assert bank.guard_tuning(seeds=bank.TUNING_SEEDS) == bank.TUNING_SEEDS
    assert len(bank.TUNING_SEEDS) == 30
    solar = (686001, 686002)
    forbidden = [
        *bank.SCORED_DEFAULT_SEEDS,
        *bank.CASCADE_SEEDS,
        *bank.COLD_SEEDS,
        *bank.SENSITIVITY_SEEDS,
        *bank.BIAS_SEEDS,
        900,
        939,
        734100,
        734219,
        1000,
        1009,
        *solar,
    ]
    for seed in forbidden:
        with pytest.raises(ValueError, match="not a tuning seed"):
            bank.guard_tuning(seeds=[bank.TUNING_SEEDS[0], seed], extra=solar)
    assert not set(bank.TUNING_SEEDS) & bank.off_limits(extra=solar)
    with pytest.raises(ValueError, match="not a tuning seed"):
        bank.guard_tuning(seeds=[12345])


def rows(
    *, world: int, found: list[bool], start: list[int], cell_days: float = 20.0
) -> pl.DataFrame:
    """Truth rows of a long, shallow cell for one world."""
    n = len(found)
    return pl.DataFrame(
        {
            "world": [world] * n,
            "site": ["A"] * n,
            "kind": ["edge"] * n,
            "units": [4.0] * n,
            "amp_mw": [1.0] * n,
            "start": start,
            "days": [cell_days] * n,
            "found": found,
            "right_class": found,
        }
    )


def test_the_paired_contrast_counts_only_when_both_intervals_exclude_zero() -> None:
    """Test the contrast of a clearly better method counts, and a tie does not."""
    month_starts = [i * 30 * DAY for i in range(30)]
    worlds_a = [rows(world=w, found=[True] * 30, start=month_starts) for w in range(40)]
    worlds_b = [
        rows(world=w, found=[w % 2 == 0 and i % 2 == 0 for i in range(30)], start=month_starts)
        for w in range(40)
    ]
    a = score.add_cells(rows=pl.concat(worlds_a))
    b = score.add_cells(rows=pl.concat(worlds_b))
    c = score.contrast(a=a, b=b, cell="long, shallow", rng=np.random.default_rng(0))
    assert c.point == pytest.approx(0.75)
    assert c.counts
    tie = score.contrast(a=a, b=a, cell="long, shallow", rng=np.random.default_rng(0))
    assert tie.point == 0.0
    assert not tie.counts
    assert set(a["cell"].to_list()) == {"long, shallow"}
    assert a["month"].n_unique() >= 28


def test_the_verdict_follows_the_pre_registered_rule() -> None:
    """Test the verdict words for each combination of C1 and the two vetoes."""
    big = score.Contrast(0.12, 0.05, 0.2, 0.04, 0.21)
    small = score.Contrast(0.06, 0.02, 0.1, 0.01, 0.11)
    null = score.Contrast(0.01, -0.03, 0.05, -0.04, 0.06)
    assert score.verdict(c1=big, v1=True, v2=True, c1_upper_below_zero=False) == "supported"
    assert score.verdict(c1=small, v1=True, v2=True, c1_upper_below_zero=False) == "small gain"
    assert score.verdict(c1=big, v1=False, v2=True, c1_upper_below_zero=False) == "partly supported"
    assert score.verdict(c1=null, v1=True, v2=True, c1_upper_below_zero=False) == "not supported"
    assert score.verdict(c1=null, v1=True, v2=True, c1_upper_below_zero=True) == "refuted"


def test_the_veto_needs_both_rates_within_ten_percent() -> None:
    """Test a veto passes only when each of the two rate ratios is at most 1.10."""
    ratios, ok = score.veto_ratios(h_rates=(1.0, 0.5), base_rates=(1.0, 0.5))
    assert ratios == (1.0, 1.0)
    assert ok
    _, ok = score.veto_ratios(h_rates=(1.09, 0.56), base_rates=(1.0, 0.5))
    assert not ok


def test_calibration_scores_pinball_and_the_80_percent_interval() -> None:
    """Test pinball loss and the coverage of q10 to q90 on a known case."""
    truth = np.array([1.0, 2.0, 3.0, 10.0])
    q = np.tile(np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]), (4, 1))
    out = score.calibration(truth=truth, quantiles=q)
    assert out["coverage_80"] == pytest.approx(0.75)
    assert out["pinball_0.50"] == pytest.approx(np.mean(0.5 * np.abs(truth - 2.0)))


def test_candidates_that_recur_across_ten_worlds_are_dropped() -> None:
    """Test a candidate at one Primary and day in 10 or more worlds is a real-record event."""
    recurring = pl.DataFrame(
        {"world": list(range(10)), "primary": ["A"] * 10, "start": [5 * DAY + w for w in range(10)]}
    )
    rare = pl.DataFrame({"world": [1, 2], "primary": ["A", "B"], "start": [99 * DAY, 99 * DAY]})
    kept = score.drop_recurring(candidates=pl.concat([recurring, rare]))
    assert kept.height == 2
    assert set(kept["primary"].to_list()) == {"A", "B"}

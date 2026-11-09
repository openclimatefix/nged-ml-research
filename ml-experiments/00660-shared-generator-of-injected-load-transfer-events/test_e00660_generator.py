"""Tests of the shared injected-event generator (experiment 660). Synthetic panel, no data files."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import e00660_generator as gen

ORIGIN = datetime(2022, 1, 1, tzinfo=UTC)
YEARS = 3
LENGTH = YEARS * 365 * gen.DAY


def make_panel(*, mva: tuple[str, ...] = ("B", "D"), cutoff_gap: bool = False) -> gen.PowerPanel:
    """Six Primaries on a daily profile plus noise, a ring of neighbours, and two MVA series."""
    rng = np.random.default_rng(0)
    t = np.arange(LENGTH)
    sites = ["A", "B", "C", "D", "E", "F"]
    power: dict[str, np.ndarray] = {}
    for k, site in enumerate(sites):
        daily = 3 + 1.5 * np.sin(2 * np.pi * (t % gen.DAY) / gen.DAY - 1.0 - 0.2 * k)
        series = daily + 0.1 * rng.standard_normal(LENGTH)
        series[(t // gen.DAY) % 97 == 5] = np.nan
        power[site] = np.abs(series) if site in mva else series - 2.0
    if cutoff_gap:
        stop = int((datetime(2024, 3, 1, tzinfo=UTC) - ORIGIN).total_seconds() // 1800)
        for series in power.values():
            series[stop:] = np.nan
    ring = {s: [sites[(i - 1) % 6], sites[(i + 1) % 6]] for i, s in enumerate(sites)}
    return gen.PowerPanel(
        power=power,
        is_mva={s: s in mva for s in sites},
        noise=dict.fromkeys(sites, 0.2),
        neighbours=ring,
        origin=ORIGIN,
    )


def positives_only() -> gen.GeneratorConfig:
    """Positive events only, no overlap or chain, so groups can be checked one at a time."""
    return replace(
        gen.default_config(),
        hard_negatives=False,
        overlap_prob=0.0,
        chain_prob=0.0,
        events_per_primary_year=4.0,
    )


def diff_of(world: gen.InjectedWorld, site: str) -> np.ndarray:
    """The change the injection made to one series, with NaN counted as no change."""
    return np.nan_to_num(world.power[site] - world.base.power[site])


def test_same_seed_gives_the_same_world_and_other_seeds_differ() -> None:
    """Test same seed gives the same world and other seeds differ."""
    panel = make_panel()
    a = gen.inject(panel=panel, seed=3)
    b = gen.inject(panel=panel, seed=3)
    c = gen.inject(panel=panel, seed=4)
    assert a.events.equals(b.events)
    for s in panel.sites:
        np.testing.assert_array_equal(a.power[s], b.power[s])
    assert not a.events.equals(c.events)


def test_table_rebuilds_the_injected_panel_exactly() -> None:
    """Test table rebuilds the injected panel exactly."""
    panel = make_panel()
    world = gen.inject(panel=panel, seed=1)
    rebuilt = gen.apply_events(panel=panel, events=world.events)
    for s in panel.sites:
        np.testing.assert_array_equal(rebuilt[s], world.power[s])
    rows = world.events.to_dicts()
    for s in panel.sites:
        if panel.is_mva[s]:
            continue
        total = sum(
            (gen.event_delta(row=r, panel=panel) for r in rows if r["site"] == s),
            start=np.zeros(panel.length),
        )
        np.testing.assert_allclose(
            diff_of(world, s), np.where(np.isfinite(panel.power[s]), total, 0)
        )


def test_nothing_changes_outside_event_windows() -> None:
    """Test nothing changes outside event windows."""
    panel = make_panel()
    world = gen.inject(panel=panel, seed=2)
    windows = gen.event_windows(events=world.events, length=panel.length)
    for s in panel.sites:
        outside = ~windows.get(s, np.zeros(panel.length, dtype=bool))
        np.testing.assert_array_equal(world.power[s][outside], panel.power[s][outside])


def test_nan_readings_stay_nan() -> None:
    """Test nan readings stay nan."""
    panel = make_panel()
    world = gen.inject(panel=panel, seed=2)
    for s in panel.sites:
        np.testing.assert_array_equal(np.isnan(world.power[s]), np.isnan(panel.power[s]))


def test_recipients_receive_the_stated_share_of_the_donors_change() -> None:
    """Test recipients receive the stated share of the donors change."""
    panel = make_panel(mva=())
    world = gen.inject(panel=panel, config=positives_only(), seed=5)
    groups = 0
    for gid, rows in world.events.filter(pl.col("event_class") == "edge_transfer").group_by(
        "group_id"
    ):
        donor = rows.filter(pl.col("role") == "donor")
        recipients = rows.filter(pl.col("role") == "recipient")
        if donor.height != 1 or recipients.height == 0:
            continue
        groups += 1
        share = donor["share"][0]
        assert 0.5 <= share <= 1.1
        donor_change = diff_of(world, donor["site"][0])
        received = sum(diff_of(world, r) for r in recipients["site"])
        # A site can sit in two groups, so test the table instead of the panel when it does.
        assert gid is not None
        np.testing.assert_allclose(recipients["delta"].sum(), -share * donor["delta"][0], rtol=1e-9)
        if (
            world.events.filter(pl.col("site").is_in([donor["site"][0], *recipients["site"]]))
            .get_column("group_id")
            .n_unique()
            == 1
        ):
            ok = np.isfinite(panel.power[donor["site"][0]])
            for r in recipients["site"]:
                ok &= np.isfinite(panel.power[r])
            np.testing.assert_allclose(received[ok], -share * donor_change[ok], atol=1e-9)
    assert groups >= 5


def test_mva_series_hold_the_magnitude_and_are_never_negative() -> None:
    """Test mva series hold the magnitude and are never negative."""
    panel = make_panel()
    world = gen.inject(panel=panel, config=positives_only(), seed=6)
    for s in panel.sites:
        if not panel.is_mva[s]:
            continue
        assert np.nanmin(world.power[s]) >= 0
        acc = sum(
            (
                gen.event_delta(row=r, panel=panel)
                for r in world.events.to_dicts()
                if r["site"] == s
            ),
            start=np.zeros(panel.length),
        )
        moved = np.where(acc != 0, np.abs(panel.power[s] + acc), panel.power[s])
        np.testing.assert_allclose(world.power[s], moved, equal_nan=True)


def test_a_reading_at_or_after_the_cutoff_is_refused() -> None:
    """Test a reading at or after the cutoff is refused."""
    panel = make_panel()
    bad = gen.PowerPanel(
        power=panel.power,
        is_mva=panel.is_mva,
        noise=panel.noise,
        neighbours=panel.neighbours,
        origin=datetime(2025, 3, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="cutoff"):
        gen.inject(panel=bad, seed=1)


def test_no_event_starts_at_or_after_the_cutoff_and_the_cutoff_data_stays_empty() -> None:
    """Test no event starts at or after the cutoff and the cutoff data stays empty."""
    panel = make_panel(cutoff_gap=True)
    cutoff = datetime(2024, 3, 1, tzinfo=UTC)
    config = replace(gen.default_config(), cutoff=cutoff)
    world = gen.inject(panel=panel, config=config, seed=7)
    stop = panel.time_index(when=cutoff)
    assert world.events["start"].max() < stop
    for s in panel.sites:
        assert np.isnan(world.power[s][stop:]).all()


def test_labels_are_edge_shift_or_hard_negative() -> None:
    """Test labels are edge shift or hard negative."""
    world = gen.inject(panel=make_panel(), seed=8)
    assert set(world.events["label"]) <= {gen.POSITIVE_EDGE, gen.POSITIVE_SHIFT, gen.HARD_NEGATIVE}
    assert set(world.events["event_class"]) <= set(
        gen.EDGE_CLASSES + gen.SHIFT_CLASSES + gen.NEGATIVE_CLASSES
    )
    for cls in gen.NEGATIVE_CLASSES:
        assert world.events.filter(pl.col("event_class") == cls).height > 0


def test_topology_classes_have_the_rows_their_names_promise() -> None:
    """Test topology classes have the rows their names promise."""
    panel = make_panel()
    config = replace(positives_only(), events_per_primary_year=20.0)
    world = gen.inject(panel=panel, config=config, seed=9)
    seen: set[str] = set()
    for gid, rows in world.events.group_by("group_id"):
        cls = rows["event_class"][0]
        seen.add(cls)
        roles = set(rows["role"])
        if cls == "edge_transfer_unseen_donor":
            assert roles == {"recipient"}
        elif cls == "edge_transfer_unseen_recipient":
            assert roles == {"donor"}
        elif cls == "edge_transfer":
            assert "donor" in roles
            donor = rows.filter(pl.col("role") == "donor")["site"][0]
            for r in rows.filter(pl.col("role") == "recipient")["site"]:
                assert r in panel.neighbours[donor]
        else:
            assert roles == {"solo"}
        assert gid is not None
    assert seen == set(gen.EDGE_CLASSES + gen.SHIFT_CLASSES)


def test_permanent_shifts_have_no_end_and_temporary_ones_do() -> None:
    """Test permanent shifts have no end and temporary ones do."""
    world = gen.inject(
        panel=make_panel(), config=replace(positives_only(), events_per_primary_year=20.0), seed=10
    )
    perm = world.events.filter(pl.col("event_class") == "permanent_shift")
    assert perm.height > 0
    assert perm["end"].is_null().all()
    temp = world.events.filter(pl.col("event_class") == "temporary_shift")
    assert temp.filter(pl.col("end").is_not_null()).height > 0


def test_start_times_follow_the_logged_hours_and_weekdays() -> None:
    """Test start times follow the logged hours and weekdays."""
    starts = gen.StartTimes(half_hour_of_day=np.array([20, 21]), weekday=np.array([1, 1]))
    panel = make_panel()
    world = gen.inject(
        panel=panel, config=positives_only(), seed=11, start_times=starts
    ).events.filter(pl.col("event_class") != "common_mode")
    when = [ORIGIN.timestamp() + 1800 * s for s in world["start"]]
    hours = {int((w % 86400) // 1800) for w in when}
    days = {datetime.fromtimestamp(w, tz=UTC).weekday() for w in when}
    assert hours <= {20, 21}
    assert days == {1}


def test_overlap_and_chaining_happen_when_asked_for() -> None:
    """Test overlap and chaining happen when asked for."""
    config = replace(
        positives_only(), overlap_prob=0.4, chain_prob=0.4, events_per_primary_year=15.0
    )
    world = gen.inject(panel=make_panel(), config=config, seed=12)
    assert (world.events["chain_parent"] >= 0).sum() > 3


def test_step_ramp_drift_staged_and_partial_return_profiles() -> None:
    """Test step ramp drift staged and partial return profiles."""
    panel = make_panel(mva=())

    def row(**kw: object) -> dict:
        """A table row with every column, overridden by `kw`."""
        base: dict = {
            "event_id": 0,
            "group_id": 0,
            "site": "A",
            "role": "solo",
            "event_class": "temporary_shift",
            "label": gen.POSITIVE_SHIFT,
            "start": 1000,
            "end": 1100,
            "residual_frac": 0.0,
            "delta": 2.0,
            "units": 10.0,
            "shape": "constant",
            "shape_source": "A",
            "ramp_in_hh": 0,
            "ramp_out_hh": 0,
            "drift": 0.0,
            "mid_factor": 1.0,
            "staged_hh": 0,
            "share": 1.0,
            "period_hh": 0.0,
            "phase": 0.0,
            "partners": [],
            "chain_parent": -1,
        }
        base.update(kw)
        return base

    square = gen.event_delta(row=row(), panel=panel)
    assert square[999] == 0
    assert square[1000] == 2.0
    assert square[1099] == 2.0
    assert square[1100] == 0
    ramp = gen.event_delta(row=row(ramp_in_hh=3, ramp_out_hh=3), panel=panel)
    np.testing.assert_allclose(ramp[1000:1004], 2.0 * np.array([0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(ramp[1096:1100], 2.0 * np.array([1.0, 0.75, 0.5, 0.25]))
    partial = gen.event_delta(row=row(residual_frac=0.4), panel=panel)
    assert partial[1100] == pytest.approx(0.8)
    assert partial[-1] == pytest.approx(0.8)
    staged = gen.event_delta(row=row(staged_hh=20), panel=panel)
    assert staged[1079] == 2.0
    assert staged[1080] == 1.0
    drift = gen.event_delta(row=row(drift=0.2), panel=panel)
    assert drift[1099] > drift[1000]
    assert drift[1050] == pytest.approx(2.0, rel=0.01)
    open_end = gen.event_delta(row=row(end=None), panel=panel)
    assert open_end[-1] == 2.0
    prop = gen.event_delta(row=row(shape="proportional"), panel=panel)
    assert prop[1000:1100].mean() == pytest.approx(2.0, rel=0.05)
    assert prop[1000:1100].std() > 0.1


def test_config_rejects_probabilities_that_do_not_sum_to_one() -> None:
    """Test config rejects probabilities that do not sum to one."""
    with pytest.raises(ValueError, match="sum to 1"):
        gen.GeneratorConfig(class_probs=(("edge_transfer", 0.5), ("permanent_shift", 0.4)))


def test_sensitivity_constructors_change_only_what_they_name() -> None:
    """Test sensitivity constructors change only what they name."""
    base = gen.default_config()
    assert gen.short_heavy().duration_bands != base.duration_bands
    assert gen.long_heavy().duration_bands != base.duration_bands
    square = gen.constant_mw_square()
    assert (square.proportional_prob, square.ramp_hh, square.staged_return_prob) == (
        0.0,
        (0, 0),
        0.0,
    )
    shaped = gen.shaped_transfers()
    assert shaped.proportional_prob == 1.0
    assert shaped.staged_return_prob > 0
    assert shaped.units_range == base.units_range


def test_a_later_cutoff_cannot_switch_off_the_final_test_guard() -> None:
    """A later cutoff cannot switch off the final test guard."""
    panel = gen.PowerPanel(
        power=make_panel().power,
        is_mva=make_panel().is_mva,
        noise=make_panel().noise,
        neighbours=make_panel().neighbours,
        origin=datetime(2025, 3, 1, tzinfo=UTC),
    )
    config = replace(gen.default_config(), cutoff=datetime(2030, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="cutoff"):
        gen.inject(panel=panel, config=config, seed=1)


def test_no_event_of_any_class_sits_where_its_primary_has_no_data() -> None:
    """No event of any class sits where its Primary has no data."""
    panel = make_panel()
    power = {s: v.copy() for s, v in panel.power.items()}
    power["C"][: 2 * 365 * gen.DAY] = np.nan
    gappy = gen.PowerPanel(
        power=power,
        is_mva=panel.is_mva,
        noise=panel.noise,
        neighbours=panel.neighbours,
        origin=ORIGIN,
    )
    world = gen.inject(panel=gappy, seed=13)
    on_c = world.events.filter((pl.col("site") == "C") & (pl.col("event_class") != "common_mode"))
    assert on_c["start"].min() >= 2 * 365 * gen.DAY - gen.DAY
    cm = world.events.filter((pl.col("site") == "C") & (pl.col("event_class") == "common_mode"))
    assert cm["start"].min() >= 2 * 365 * gen.DAY


def test_table_conserves_megawatts_even_when_gaps_differ() -> None:
    """The table conserves megawatts even when gaps differ; only the panel diff is masked."""
    panel = make_panel(mva=())
    power = {s: v.copy() for s, v in panel.power.items()}
    power["B"][5000:5400] = np.nan
    gappy = gen.PowerPanel(
        power=power,
        is_mva=panel.is_mva,
        noise=panel.noise,
        neighbours=panel.neighbours,
        origin=ORIGIN,
    )
    world = gen.inject(panel=gappy, config=positives_only(), seed=5)
    for _gid, rows in world.events.filter(pl.col("event_class") == "edge_transfer").group_by(
        "group_id"
    ):
        donor = rows.filter(pl.col("role") == "donor")
        rec = rows.filter(pl.col("role") == "recipient")
        if donor.height == 1 and rec.height:
            assert rec["delta"].sum() == pytest.approx(-donor["share"][0] * donor["delta"][0])

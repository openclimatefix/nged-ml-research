# ruff: noqa: E501
"""Known-answer tests of the real-day readings P1 to P4 and K3, and of the real-day loaders (no torch)."""

from datetime import date, datetime

import e00729_arms as arms729
import e00732_arms as arms
import e00732_core as core
import e00732_neighbours as nb
import e00732_real as real
import numpy as np
import polars as pl
import pytest
from world732 import make_neighbours


def days(n: int = 20):
    cosz = np.tile(np.concatenate([np.full(10, -0.5), np.full(28, 0.6), np.full(10, -0.5)]), (n, 1))
    return cosz, np.ones_like(cosz, bool)


def test_p1_night_splits_the_in_sample_part_and_fails_on_a_leak() -> None:
    cosz, valid = days()
    pv = np.where(cosz > 0, 5.0, 0.0)
    in_sample = np.zeros_like(valid)
    in_sample[:5] = True
    out = real.p1_night(
        pv=pv.ravel(), cosz=cosz.ravel(), valid=valid.ravel(), in_sample=in_sample.ravel()
    )
    assert out["all"]["pass"]
    assert out["held_out"]["pass"]
    assert out["in_sample"]["pass"]
    leak = pv.copy()
    leak[2, :3] = 3.0  # only in-sample days leak at night
    out = real.p1_night(
        pv=leak.ravel(), cosz=cosz.ravel(), valid=valid.ravel(), in_sample=in_sample.ravel()
    )
    assert out["held_out"]["pass"]
    assert not out["in_sample"]["pass"]
    assert not out["all"]["pass"]


def test_p2_prints_a_rounded_mean_and_fails_a_farm() -> None:
    cosz, valid = days()
    assert real.p2_no_farm(pv=np.full(cosz.shape, 0.2), cosz=cosz, valid=valid)["pass"]
    out = real.p2_no_farm(pv=np.full(cosz.shape, 0.26), cosz=cosz, valid=valid)
    assert out["mean_rounded"] == 0.3
    assert not real.p2_no_farm(pv=np.full(cosz.shape, 6.0), cosz=cosz, valid=valid)["pass"]
    assert not real.p2_no_farm(pv=np.zeros(cosz.shape), cosz=-np.ones(cosz.shape), valid=valid)[
        "pass"
    ]


def sky_world(follow: bool):
    rng = np.random.default_rng(0)
    cosz, valid = days(40)
    csi = np.repeat(np.linspace(0.05, 1.0, 40)[:, None], 48, axis=1)
    pv = (
        np.where(cosz > 0, 20.0 * csi * cosz, 0.0)
        if follow
        else np.where(cosz > 0, rng.random(cosz.shape), 0.0)
    )
    return pv, csi, cosz, valid


def test_p3_passes_a_head_that_follows_the_sky_and_fails_one_that_ignores_it() -> None:
    pv, csi, cosz, valid = sky_world(follow=True)
    ok = real.p3_follows_sky(pv=pv, csi=csi, cosz=cosz, valid=valid)
    assert ok["pass"]
    assert ok["spearman"] > 0.99
    assert ok["ratio"] > 2
    pv, csi, cosz, valid = sky_world(follow=False)
    assert not real.p3_follows_sky(pv=pv, csi=csi, cosz=cosz, valid=valid)["pass"]
    assert not real.p3_follows_sky(pv=pv[:2], csi=csi[:2], cosz=cosz[:2], valid=valid[:2])["pass"]


def test_p4_fails_a_head_that_absorbs_the_engine_and_returns_no_level() -> None:
    n = 400
    hh = np.tile(np.arange(48), n // 48 + 1)[:n]
    running = (np.arange(n) // 48) % 2 == 0
    valid = np.ones(n, bool)
    quiet = real.p4_absorb(pv=np.full(n, 3.0), running=running, hh=hh, valid=valid)
    assert quiet["pass"]
    assert set(quiet) == {"pass", "n_running", "void"}
    assert not real.p4_absorb(pv=np.where(running, 6.0, 3.0), running=running, hh=hh, valid=valid)[
        "pass"
    ]
    assert real.p4_absorb(pv=np.zeros(n), running=np.ones(n, bool), hh=hh, valid=valid)["void"]


def test_spearman_handles_ties_and_constants() -> None:
    assert real.spearman(
        a=np.array([1.0, 2.0, 3.0, 4.0]), b=np.array([2.0, 4.0, 6.0, 8.0])
    ) == pytest.approx(1.0)
    assert real.spearman(
        a=np.array([1.0, 1.0, 2.0, 3.0]), b=np.array([1.0, 2.0, 2.0, 3.0])
    ) == pytest.approx(0.8, abs=0.05)
    assert np.isnan(real.spearman(a=np.ones(5), b=np.arange(5.0)))


def test_closure_edges_carry_their_times_and_k3_reads_them() -> None:
    sheet = pl.DataFrame(
        {
            "Substation": ["7", "7", "9"],
            "Start": [
                datetime(2023, 3, 1, 9, 0),
                datetime(2023, 3, 5, 14, 30),
                datetime(2023, 3, 6, 1, 0),
            ],
            "End": [
                datetime(2023, 3, 1, 17, 0),
                datetime(2023, 3, 6, 0, 0),
                datetime(2023, 3, 6, 5, 0),
            ],
        }
    )
    edges = real.closure_edges_from_sheets(sheets={"s": sheet}, substation_number=7)
    assert [e.has_time for e in edges] == [True, True, True, True]
    dated = pl.DataFrame(
        {"Substation": ["7"], "Start": [date(2023, 3, 2)], "End": [date(2023, 3, 3)]}
    )
    assert [
        e.has_time for e in real.closure_edges_from_sheets(sheets={"s": dated}, substation_number=7)
    ] == [False, False]
    assert (
        real.closure_edges_from_sheets(sheets={"s": pl.DataFrame({"x": [1]})}, substation_number=7)
        == []
    )
    assert real.closure_edges_from_sheets(sheets={"s": sheet}, substation_number=8) == []
    n_days = 12
    index = {date(2023, 3, 1 + i): i for i in range(n_days)}
    step = np.zeros((n_days, 48))
    valid = np.ones((n_days, 48), bool)
    many = [real.Edge(when=datetime(2023, 3, 1 + i, 9, 0), has_time=True) for i in range(10)]
    step[:, 18] = 0.9  # a flag at 09:00 on every day
    hit = real.k3_hit_rate(edges=many, step_p=step, day_index=index, valid=valid)
    assert hit["hit_share"] == 1.0
    assert not hit["void"]
    step[:, 18], step[:, 40] = 0.0, 0.9  # the flag moves 11 hours away
    assert (
        real.k3_hit_rate(edges=many, step_p=step, day_index=index, valid=valid)["hit_share"] == 0.0
    )
    assert real.k3_hit_rate(edges=many[:3], step_p=step, day_index=index, valid=valid)["void"]
    no_time = [real.Edge(when=datetime(2023, 3, 1 + i, 0, 0), has_time=False) for i in range(10)]
    assert (
        real.k3_hit_rate(edges=no_time, step_p=step, day_index=index, valid=valid)["hit_share"]
        == 1.0
    )
    off_days = [real.Edge(when=datetime(2024, 1, 1 + i), has_time=False) for i in range(10)]
    assert real.k3_hit_rate(edges=off_days, step_p=step, day_index=index, valid=valid)["void"]


def test_k3_false_steps_count_flags_per_day_outside_closures() -> None:
    step = np.zeros((4, 48))
    step[:, 5] = 0.9
    step[3, 6:10] = 0.9
    valid = np.ones((4, 48), bool)
    closure = np.array([False, False, False, True])
    out = real.k3_false_steps(step_p=step, valid=valid, closure_days=closure)
    assert out["n_days"] == 3
    assert out["flagged_per_day"] == 1.0
    assert real.k3_false_steps(step_p=step, valid=valid, closure_days=np.ones(4, bool))["void"]


def test_every_real_check_fails_on_a_nan_output() -> None:
    cosz, valid = days()
    nan = np.full(cosz.shape, np.nan)
    assert not real.p1_night(pv=nan.ravel(), cosz=cosz.ravel(), valid=valid.ravel())["all"]["pass"]
    assert not real.p2_no_farm(pv=nan, cosz=cosz, valid=valid)["pass"]
    pv, csi, cosz3, valid3 = sky_world(follow=True)
    assert not real.p3_follows_sky(pv=np.full(pv.shape, np.nan), csi=csi, cosz=cosz3, valid=valid3)[
        "pass"
    ]
    n = 400
    hh = np.tile(np.arange(48), n // 48 + 1)[:n]
    running = (np.arange(n) // 48) % 2 == 0
    out = real.p4_absorb(pv=np.full(n, np.nan), running=running, hh=hh, valid=np.ones(n, bool))
    assert not out["pass"]
    assert set(out) == {"pass", "n_running", "void"}


def test_the_real_constants_are_the_cores() -> None:
    assert slice(20, 28) == real.MIDDAY
    assert real.K3_WINDOW_HALF_HOURS == 4
    cosz, valid = days()
    just_over = np.full(cosz.shape, core.P2_MEAN_POINTS + 0.2)
    assert not real.p2_no_farm(pv=just_over, cosz=cosz, valid=valid)["pass"]
    assert real.p2_no_farm(pv=just_over - 0.3, cosz=cosz, valid=valid)["pass"]


def test_k3_reads_each_rule_separately_with_counts_and_shares_only() -> None:
    n_days = 12
    index = {date(2023, 3, 1 + i): i for i in range(n_days)}
    valid = np.ones((n_days, 48), bool)
    edges = [real.Edge(when=datetime(2023, 3, 1 + i, 9, 0), has_time=True) for i in range(10)]
    good, silent = np.zeros((n_days, 48)), np.zeros((n_days, 48))
    good[:, 18] = 0.9
    closure = np.zeros(n_days, bool)
    closure[:10] = True
    out = real.k3_rules(
        edges=edges,
        steps={"U*": good, "B-TN": silent},
        day_index=index,
        valid=valid,
        closure_days=closure,
    )
    assert out["U*"]["hit"]["hit_share"] == 1.0
    assert out["B-TN"]["hit"]["hit_share"] == 0.0
    assert out["U*"]["false"]["flagged_per_day"] == 1.0
    assert out["B-TN"]["false"]["flagged_per_day"] == 0.0
    assert set(out["U*"]["hit"]) == {"void", "n_edges", "hit_share"}
    assert set(out["U*"]["false"]) == {"void", "n_days", "flagged_per_day"}


def test_by_hour_breakdown_covers_only_night_and_pre_engine_half_hours() -> None:
    cosz, valid = days(4)
    hh = np.tile(np.arange(48), (4, 1))
    pv = np.where(cosz > 0, 9.0, 0.0)
    night = real.night_by_hour(pv=pv.ravel(), cosz=cosz.ravel(), valid=valid.ravel(), hh=hh.ravel())
    assert set(night) == {
        h for h in range(24) if (cosz[0, 2 * h : 2 * h + 2] <= core.NIGHT_COSZ).any()
    }
    assert all(v["share_above"] == 0.0 for v in night.values())
    leak = pv.copy()
    leak[:, 0] = 2.0
    assert (
        real.night_by_hour(pv=leak.ravel(), cosz=cosz.ravel(), valid=valid.ravel(), hh=hh.ravel())[
            0
        ]["share_above"]
        == 0.5
    )
    pre = np.zeros(cosz.shape, bool)
    pre[:2] = True
    out = real.pre_engine_by_hour(
        pv=pv.ravel(), cosz=cosz.ravel(), valid=valid.ravel(), hh=hh.ravel(), pre_engine=pre.ravel()
    )
    assert out
    assert all(v["share_above"] == 1.0 and v["n"] == 4.0 for v in out.values())
    assert (
        real.pre_engine_by_hour(
            pv=pv.ravel(),
            cosz=cosz.ravel(),
            valid=valid.ravel(),
            hh=hh.ravel(),
            pre_engine=np.zeros(pv.size, bool),
        )
        == {}
    )


def test_negative_power_shares_count_the_export_half_hours_and_skip_missing_ones() -> None:
    signed = {
        "D5": np.array([[1.0, -2.0], [3.0, np.nan]]),
        "D14": np.full((2, 2), np.nan),
        "D3": np.ones(3),
    }
    out = real.negative_power_shares(signed=signed)
    assert out["D5"] == pytest.approx(1 / 3)
    assert np.isnan(out["D14"])
    assert set(out) == {"D5", "D14"}


def test_real_day_windows_carry_the_real_neighbours_through_the_real_loader() -> None:
    from world732 import make_calendar

    cal = make_calendar(n_days=62, seed=0)
    series = make_neighbours(cal=cal)
    use = np.ones(len(cal.dates), bool)
    reading = np.nan_to_num(cal.reading, nan=5.0)
    everyone = np.arange(len(cal.dates))
    stats = arms729.input_stats(cal=cal, train_days=everyone)
    medians = {
        n: nb.training_median(folded=nb.fold(signed=series[n], floor=0.0), train_rows=everyone)
        for n in nb.NEIGHBOURS
    }
    kw = {
        "cal": cal,
        "use": use,
        "reading": reading,
        "truth": reading,
        "stats": stats,
        "medians": medians,
    }
    with_nb, segs = real.real_day_windows(neighbours=series, **kw)
    without, _ = real.real_day_windows(neighbours=None, **kw)
    assert len(with_nb) == len(segs) == len({str(d)[:7] for d in cal.dates})
    day = int(with_nb[0].days[3])
    spec = arms.eval_spec()
    shown = arms.plain_window(p=with_nb[0], day=day, spec=spec, hidden=None, real=True)
    absent = arms.plain_window(p=without[0], day=day, spec=spec, hidden=None, real=True)
    assert shown[[arms.NB_ROW[f"{n}_mask"] for n in nb.NEIGHBOURS]].min() == 1.0
    assert not absent[arms.N_BASE :].any()
    assert shown.shape[0] == arms.N_CHANNELS

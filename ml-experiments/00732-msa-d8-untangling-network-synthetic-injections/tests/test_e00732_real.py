"""Known-answer tests of the real-day readings P1 to P4 and K3 on invented arrays (no torch)."""

from datetime import date, datetime

import e00732_real as real
import numpy as np
import polars as pl
import pytest


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
    assert out["all"]["pass"] and out["held_out"]["pass"] and out["in_sample"]["pass"]
    leak = pv.copy()
    leak[2, :3] = 3.0  # only in-sample days leak at night
    out = real.p1_night(
        pv=leak.ravel(), cosz=cosz.ravel(), valid=valid.ravel(), in_sample=in_sample.ravel()
    )
    assert out["held_out"]["pass"] and not out["in_sample"]["pass"] and not out["all"]["pass"]


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
    assert ok["pass"] and ok["spearman"] > 0.99 and ok["ratio"] > 2
    pv, csi, cosz, valid = sky_world(follow=False)
    assert not real.p3_follows_sky(pv=pv, csi=csi, cosz=cosz, valid=valid)["pass"]
    assert not real.p3_follows_sky(pv=pv[:2], csi=csi[:2], cosz=cosz[:2], valid=valid[:2])["pass"]


def test_p4_fails_a_head_that_absorbs_the_engine_and_returns_no_level() -> None:
    n = 400
    hh = np.tile(np.arange(48), n // 48 + 1)[:n]
    running = (np.arange(n) // 48) % 2 == 0
    valid = np.ones(n, bool)
    quiet = real.p4_absorb(pv=np.full(n, 3.0), running=running, hh=hh, valid=valid)
    assert quiet["pass"] and set(quiet) == {"pass", "n_running", "void"}
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
    assert hit["hit_share"] == 1.0 and not hit["void"]
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
    assert out["n_days"] == 3 and out["flagged_per_day"] == 1.0
    assert real.k3_false_steps(step_p=step, valid=valid, closure_days=np.ones(4, bool))["void"]

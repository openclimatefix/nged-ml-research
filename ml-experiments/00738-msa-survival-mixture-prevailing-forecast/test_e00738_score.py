"""Tests of the scoring, the contrasts, the verdict table, and the gate."""
# ruff: noqa: D103, E402, PT018

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
# the tests never write to the experiment's data folder
os.environ.setdefault("E00738_OUT", str(Path(tempfile.gettempdir()) / "scratch" / "e738_tests"))
os.environ.setdefault("E00738_SCRATCH", "1")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 378, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))

import e00738_common as common
import e00738_gate as gate
import e00738_members as members
import e00738_score as score
import e00738_world as world


def _quantiles(centre: np.ndarray, width: float = 0.05) -> np.ndarray:
    return centre[:, None] + width * np.linspace(-1.0, 1.0, 11)[None, :]


def test_pinball_and_crps_are_zero_for_a_perfect_point_forecast() -> None:
    y = np.array([0.3, 0.5])
    q = np.repeat(y[:, None], 11, axis=1)
    assert np.all(score.crps_points(y=y, q=q) == 0.0)
    assert score.pinball_points(y=y, q=q).shape == (2, 11)


def test_crps_in_points_of_capacity() -> None:
    y = np.array([0.5])
    q = np.repeat(np.array([[0.4]]), 11, axis=1)  # a point forecast 0.1 below
    # twice the mean pinball loss over levels with mean level 0.5 is 0.1 * 100
    assert score.crps_points(y=y, q=q)[0] == pytest.approx(10.0, rel=0.05)


def test_scale_factor_widens_an_overconfident_forecast_to_90_percent_cover() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 4000)
    q = np.zeros((4000, 11)) + np.array(
        [-0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5]
    )
    f = score.scale_factor(y=y, q=q)
    scaled = score.apply_scale(q=q, factor=f)
    cover = ((y >= scaled[:, 1]) & (y <= scaled[:, 9])).mean()
    assert cover == pytest.approx(0.90, abs=0.01)
    assert f > 1.0
    assert score.apply_scale(q=q, factor=1.0)[:, 5].tolist() == q[:, 5].astype(np.float32).tolist()


def test_follow_share_of_an_oracle_is_one_whatever_the_onset_time_of_day() -> None:
    for half_hour in (0, 24, 47):
        assert gate.oracle_follow_synthetic(onset_half_hour_of_day=half_hour) == pytest.approx(1.0)


def test_follow_share_of_no_response_and_half_response() -> None:
    delta = np.array([0.0, 0.1, 0.1, 0.0])
    ref = np.zeros(4)
    assert score.follow_share(median_arm=ref, median_ref=ref, delta=delta) == 0.0
    assert score.follow_share(median_arm=0.5 * delta, median_ref=ref, delta=delta) == (
        pytest.approx(0.5)
    )
    assert np.isnan(score.follow_share(median_arm=ref, median_ref=ref, delta=np.zeros(4)))


def _masks() -> tuple[dict[str, np.ndarray], np.ndarray]:
    d = world.PER_DAY
    onset = np.full(16, 0)
    end = np.full(16, 30 * d)
    clipped = np.zeros(16, bool)
    prim = np.array([0, 1, 2, 3, 4])
    origin = np.array([2 * d, 25 * d, 29 * d + 24, 31 * d, 40 * d])
    valid = origin + 5 * d
    lead = np.array([4, 4, 4, 4, 4])
    m = score.row_masks(prim=prim, origin_idx=origin, valid_idx=valid, lead_day=lead,
                        onset=onset, end=end, clipped=clipped)  # fmt: skip
    return m, origin


def test_row_masks_cover_each_row_once_and_in_force_is_the_union() -> None:
    m, _ = _masks()
    names = [*world.STRATA, "in_force_clipped"]
    stack = np.stack([m[n] for n in names])
    assert np.all(stack.sum(axis=0) == 1)
    assert np.array_equal(m["in_force"], m["far"] | m["middle"] | m["near"])
    assert m["all"].all()
    assert m["far"].tolist() == [True, False, False, False, False]
    assert m["middle"].tolist() == [False, True, False, False, False]
    assert m["near"].tolist() == [False, False, True, False, False]
    assert m["ended_24h"].tolist() == [False, False, False, True, False]
    assert m["in_force_lead_4"].tolist() == m["in_force"].tolist()
    assert m["in_force_lead_5"].sum() == 0
    assert set(m) >= {f"in_force_{n}" for n, _, _ in common.AGE_BANDS}


def test_accumulate_sums_by_primary_and_counts_rows() -> None:
    sums: dict = {}
    prim = np.array([0, 0, 3, 3, 3])
    values = {"crps": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    masks = {"all": np.ones(5, bool), "some": np.array([True, False, True, False, False])}
    score.accumulate(sums=sums, masks=masks, prim=prim, values=values)
    score.accumulate(sums=sums, masks=masks, prim=prim, values=values)
    assert sums["all"]["crps"][0] == 6.0 and sums["all"]["crps"][3] == 24.0
    assert sums["all"]["n"][3] == 6
    assert sums["some"]["crps"][3] == 6.0
    assert sums["some"]["n"].sum() == 4


def test_t_interval_matches_the_textbook_formula() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    lo, hi = score.t_interval(monthly=x, level=95.0)
    half = 2.364624 * x.std(ddof=1) / np.sqrt(8)
    assert lo == pytest.approx(x.mean() - half, abs=1e-4)
    assert hi == pytest.approx(x.mean() + half, abs=1e-4)
    wider = score.t_interval(monthly=x, level=common.LEVEL_BONFERRONI)
    assert wider[0] < lo and wider[1] > hi


def test_net_effect_and_its_bootstrap_agree_when_months_are_identical() -> None:
    assert score.net_effect(c=-0.2, v=0.01, q=0.05) == pytest.approx(0.05 * -0.2 + 0.95 * 0.01)
    c_sums = np.array([[-2.0, -2.0, -2.0, -2.0]])
    v_sums = np.array([[1.0, 1.0, 1.0, 1.0]])
    ones = np.full(4, 10.0)
    lo, hi = score.month_block_bootstrap_net(
        c_sums=c_sums, c_counts=ones, v_sums=v_sums, v_counts=ones, q=0.05, level=95.0,
        n_resamples=200,
    )  # fmt: skip
    want = 0.05 * -0.2 + 0.95 * 0.1
    assert lo == pytest.approx(want) and hi == pytest.approx(want)


def _tables(*, delta: float, months: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    t = {}
    for m_i, m in enumerate(months):
        for s in seeds:

            def arm(level: float, m_i: int = m_i) -> dict:
                body = {"crps": np.full(16, (level + 0.01 * m_i) * 100.0), "n": np.full(16, 100.0)}
                return {"train_family": {"in_force": body, "event_free": body}}

            t[(m, s)] = {"M1": arm(1.0), "LD": arm(1.0 + delta)}
    return t


def test_contrast_readings_have_the_sign_of_the_difference() -> None:
    months = ("2025-01", "2025-02", "2025-03", "2025-04")
    tables = _tables(delta=-0.2, months=months, seeds=(0, 1))
    prims = np.arange(16)
    sums, counts = score.month_arrays(
        tables=tables, months=months, seeds=(0, 1), arm_a="LD", arm_b="M1",
        wld="train_family", slc="in_force", metric="crps", prims=prims,
    )  # fmt: skip
    assert sums.shape == (2, 4) and counts.shape == (4,)
    r = score.reading_from_month_arrays(sums=sums, counts=counts, level=95.0, n_resamples=500)
    assert r.point == pytest.approx(-0.2)
    assert r.seed_points == pytest.approx((-0.2, -0.2))
    assert r.t_lower <= r.point <= r.t_upper or r.t_lower == r.t_upper


def test_choose_s_star_prefers_the_simpler_arm_within_the_margin() -> None:
    assert score.choose_s_star(in_force={"LD": 5.0, "LM": 4.95, "LR": 4.9}) == "LD"
    assert score.choose_s_star(in_force={"LD": 5.0, "LM": 4.7, "LR": 4.9}) == "LM"
    assert score.choose_s_star(in_force={"LD": 5.0, "LM": 4.9, "LR": 4.2}) == "LR"


def test_world_of_panel_names() -> None:
    assert score.world_of(name="real") == "real"
    assert score.world_of(name="resampled00") == "resampled"
    assert score.world_of(name="pass00") == "train_family"
    assert score.world_of(name=f"pass{common.HELD_FIRST}") == "held_out"


def test_sums_round_trip_through_json(tmp_path: Path) -> None:
    sums = {"M1": {"real": {"all": {"crps": np.arange(16.0), "n": np.ones(16)}}}}
    score.write_sums(sums=sums, month="2025-11", seed=0, out=tmp_path)
    back = score.read_sums(month="2025-11", seed=0, out=tmp_path)
    np.testing.assert_array_equal(back["M1"]["real"]["all"]["crps"], np.arange(16.0))


# --- The verdict table ---------------------------------------------------------------------------


def test_every_state_of_the_verdict_table_maps_to_exactly_one_row() -> None:
    counts: dict[str, int] = {}
    n = 0
    for state in score.all_states():
        n += 1
        rows = score.matching_rows(state=state)
        assert rows, "row 8 must always hold"
        assert rows[-1] == 8
        row, name = score.verdict(state=state)
        assert row == rows[0], "the first matching row decides"
        assert 1 <= row <= 8
        counts[name] = counts.get(name, 0) + 1
    assert n == 2**14
    assert sum(counts.values()) == 2**14
    assert set(counts) == {
        "invalid", "refuted", "not supported", "inconclusive", "supported", "partly supported",
        "small gain",
    }  # fmt: skip


def _state(**kw: bool) -> score.VerdictState:
    base = dict.fromkeys(score.VerdictState.__dataclass_fields__, False)
    return score.VerdictState(**{**base, **kw})


def test_verdict_rows_in_priority_order() -> None:
    assert score.verdict(state=_state(invalid=True, n1_lower_above_zero=True)) == (1, "invalid")
    assert score.verdict(state=_state(n1_lower_above_zero=True, n1_gain=True)) == (2, "refuted")
    assert score.verdict(state=_state(n1_inside_band=True))[1] == "not supported"
    assert score.verdict(state=_state(v1_fails=True, n1_gain=True))[1] == "not supported"
    assert score.verdict(state=_state(n1_gain=True, n1_seeds=False))[1] == "inconclusive"
    full = _state(
        n1_gain=True, n1_at_threshold=True, n1_seeds=True, n3_gain=True, n3_at_threshold=True,
        n3_seeds=True, c3_below_zero=True, c2_gain=True, c2_seeds=True, held_out_agrees=True,
    )  # fmt: skip
    assert score.verdict(state=full) == (5, "supported")
    # N3 passes only with an in-force C3 interval below zero: without it the verdict is row 6
    no_c3 = _state(
        **{**{k: getattr(full, k) for k in full.__dataclass_fields__}, "c3_below_zero": False}
    )
    assert score.verdict(state=no_c3) == (6, "partly supported")
    partly = _state(n1_gain=True, n1_at_threshold=True, n1_seeds=True)
    assert score.verdict(state=partly) == (6, "partly supported")
    assert score.label_of(state=partly) == "partly supported (duration-dependent)"
    small = _state(n1_gain=True, n1_seeds=True)
    assert score.verdict(state=small) == (7, "small gain")
    assert score.verdict(state=_state()) == (8, "inconclusive")


# --- The gate -----------------------------------------------------------------------------------


def test_gate_decisions_and_files(tmp_path: Path) -> None:
    assert gate.decide(passed=True, remedy_done=False) == "pass"
    assert gate.decide(passed=False, remedy_done=False) == "remedy"
    assert gate.decide(passed=False, remedy_done=True) == "stop"
    bad = gate.evaluate(oracle={"midday": 1.0, "midnight": 1.0}, lp_follow=0.5, depth_ratio=0.9)
    assert not bad["passed"]
    assert gate.write_gate(record=bad, remedy_done=False, code_hash="x", out=tmp_path) == "remedy"
    assert gate.remedy_active(out=tmp_path)
    assert not (tmp_path / gate.GATE_FILE).exists()
    ok = gate.evaluate(oracle={"midday": 1.0, "midnight": 0.99}, lp_follow=0.8, depth_ratio=0.75)
    assert ok["passed"]
    assert gate.write_gate(record=ok, remedy_done=True, code_hash="x", out=tmp_path) == "pass"
    assert (tmp_path / gate.GATE_FILE).exists()


def test_depth_ratio_table_takes_the_median_by_age_band() -> None:
    age = np.array([3.5, 4.0, 7.0, 12.0, 4.0])
    true = np.full(5, 0.1)
    share = np.array([0.08, 0.06, 0.09, 0.05, 0.02])
    seen = np.array([True, True, True, True, False])
    t = gate.depth_ratio_table(depth_share=share, true_depth=true, age=age, seen=seen)
    assert t["3-5"] == pytest.approx(0.7)
    assert t["6-10"] == pytest.approx(0.9)
    assert t["11-15"] == pytest.approx(0.5)
    assert t["3-10"] == pytest.approx(0.8)


# --- The 51-path check --------------------------------------------------------------------------


def test_end_times_condition_on_the_age_and_are_ordered() -> None:
    curve = score_curve()
    t = members.end_times(curve=curve, age_days=10.0, n=51)
    assert t.shape == (51,) and np.all(t >= 10.0)
    assert np.all(np.diff(t) <= 1e-9), "longest first"


def score_curve():  # noqa: ANN201
    import e00738_survival as surv

    return surv.GeneratorCurve()


def test_paths_end_share_matches_one_minus_the_ratio() -> None:
    import e00738_survival as surv

    curve = score_curve()
    base_q = np.tile(np.linspace(0.1, 0.5, 11), (6, 1))
    leads = np.array([0.5, 1.5, 3.0, 5.0, 8.0, 12.0])
    paths, ended = members.paths_at_leads(base_q=base_q, depth=0.1, age_days=5.0,
                                         lead_days=leads, curve=curve)  # fmt: skip
    p = surv.ratio(curve=curve, age_days=np.full(6, 5.0), lead_days=leads)
    np.testing.assert_allclose(ended, 1.0 - p, atol=0.03)
    assert paths.shape == (6, members.N_PATHS)


def test_fair_crps_energy_and_variogram_scores_are_zero_or_small_for_a_perfect_ensemble() -> None:
    paths = np.zeros((1, 51))
    assert members.fair_crps(paths=paths, y=np.zeros(1))[0] == pytest.approx(0.0)
    assert members.fair_crps(paths=paths, y=np.ones(1))[0] == pytest.approx(1.0)
    assert members.energy_score(paths=np.zeros((51, 3)), y=np.zeros(3)) == pytest.approx(0.0)
    assert members.variogram_score(paths=np.zeros((51, 3)), y=np.zeros(3)) == pytest.approx(0.0)
    assert members.passes(ended_gap=0.01, quantile_gap=0.1, crps_gap=0.01)
    assert not members.passes(ended_gap=0.5, quantile_gap=0.1, crps_gap=0.01)


def test_interp_quantile_and_isf() -> None:
    q = np.linspace(0.0, 1.0, 11)[None, :]
    levels = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
    got = members.interp_quantile(q=np.tile(levels, (1, 1)), tau=np.array([0.5, 0.0, 1.0]))
    np.testing.assert_allclose(got[0], [0.5, 0.01, 0.99])
    del q
    t = members.isf(curve=score_curve(), s=np.array([1.0, 0.5, 0.25]))
    assert t[0] == 0.0

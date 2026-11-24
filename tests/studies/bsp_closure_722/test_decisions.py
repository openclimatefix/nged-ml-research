"""Every bar and decision rule of section 8, the bootstrap, de-meaning, and detection limits."""

import e00722_core as core
import numpy as np
from e00722_testkit import block_flow, calendar, make_weights


def test_every_bar_constant_matches_the_specification():
    assert core.N_BOOT == 1000
    assert core.SE_MULT == 2.8
    assert core.ENGINE_DETECT_MAX == 0.5
    assert core.D15_DETECT_MAX == 0.3
    assert core.ENGINE_SLOPE_BAND == (-1.3, -0.7)
    assert core.ENGINE_PLACEBO_SHIFT_DAYS == (7, 14)
    assert core.D15_CLOSURE_BAND == (0.7, 1.3)
    assert core.D15_AUC_MIN == 0.85
    assert core.D15_SIZE_NOISE_MULT == 2.0
    assert core.D15_MIN_HOURS == 200
    assert core.PLACEBO_SHIFT_DAYS == 7
    assert core.AUC_NULL == 0.5
    assert core.U2B_DETECT_MAX == 1.0
    assert core.CALIBRATION_LOWER_ABOVE == 0.5
    assert core.TEST_REVERSE_UPPER_BELOW == -0.5
    assert core.TEST_REFUTE_LOWER_ABOVE == 0.5
    assert core.ENGINE_STEP_DROP_HALF_HOURS == 2
    assert core.ADDBACK_NULL_BAND == (-0.2, 0.2)
    assert core.EMBEDDED_BAND == (-1.3, -0.7)
    assert core.ENGINE_BELOW_POINTS == -2.0
    assert core.MIN_ENGINE_COVERAGE == 0.9
    assert core.MIN_SAME_SKY_PAIRS == 4
    assert core.SAME_SKY_CORRELATION == 0.5
    assert core.DOMINANT_FARM_SHARE == 1 / 3
    assert core.CORE_SECOND_CAP == 3600.0
    assert core.HOLDOUT.isoformat() == "2026-07-01T00:00:00+00:00"


def test_a_detection_limit_over_the_cap_is_void_and_a_nan_is_void():
    assert not core.is_void(se=0.1, max_limit=0.5)
    assert core.is_void(se=0.2, max_limit=0.5)
    assert core.is_void(se=float("nan"), max_limit=0.5)
    assert core.detection_limit(se=0.25) == 0.7


def test_the_engine_bar():
    good_shift = {7: (-0.2, 0.2), 14: (-0.3, 0.1)}
    assert core.engine_bar(lo=-1.1, hi=-0.9, se=0.05, shifted=good_shift) == "pass"
    assert core.engine_bar(lo=-1.1, hi=-0.9, se=0.2, shifted=good_shift) == "void"
    assert core.engine_bar(lo=-1.4, hi=-0.9, se=0.05, shifted=good_shift) == "fail"
    assert core.engine_bar(lo=-0.4, hi=0.1, se=0.05, shifted=good_shift) == "fail"
    assert (
        core.engine_bar(lo=-1.1, hi=-0.9, se=0.05, shifted={7: (0.1, 0.4), 14: (-0.3, 0.1)})
        == "fail"
    )


def test_the_d15_bars():
    assert core.d15_closure_bar(lo=0.8, hi=1.2, se=0.05) == "pass"
    assert core.d15_closure_bar(lo=0.6, hi=1.2, se=0.05) == "fail"
    assert core.d15_closure_bar(lo=0.8, hi=1.2, se=0.2) == "void"
    assert core.d15_sign_bar(auc=0.9, n_hours=250) == "pass"
    assert core.d15_sign_bar(auc=0.8, n_hours=250) == "fail"
    assert core.d15_sign_bar(auc=0.99, n_hours=199) == "void"
    assert core.d15_placebo_bar(sign_outcome="void", lo=0.4, hi=0.6, n_hours=300) == "void"
    assert core.d15_placebo_bar(sign_outcome="pass", lo=0.4, hi=0.6, n_hours=300) == "pass"
    assert core.d15_placebo_bar(sign_outcome="pass", lo=0.6, hi=0.8, n_hours=300) == "fail"


def test_the_d15_placebo_needs_200_scorable_hours():
    """B8: a placebo on 199 hours is void whatever its interval, and 200 hours is enough."""
    assert core.d15_placebo_bar(sign_outcome="pass", lo=0.45, hi=0.55, n_hours=199) == "void"
    assert core.d15_placebo_bar(sign_outcome="pass", lo=0.45, hi=0.55, n_hours=200) == "pass"
    assert core.d15_placebo_bar(sign_outcome="pass", lo=0.45, hi=0.55, n_hours=0) == "void"
    assert core.d15_placebo_bar(sign_outcome="fail", lo=0.45, hi=0.55, n_hours=30) == "void"


def test_a_passed_sign_test_with_a_void_placebo_is_not_supported():
    """B8: the pair is unreadable, so the verdict needs another readable test."""
    only_sign: dict[str, core.Outcome] = {
        "engine": "void",
        "d15_closure": "void",
        "d15_sign": "pass",
        "d15_placebo": "void",
    }
    assert core.verdict_u2a(outcomes=only_sign) == "inconclusive"
    with_closure: dict[str, core.Outcome] = {**only_sign, "d15_closure": "pass"}
    assert core.verdict_u2a(outcomes=with_closure) == "supported"
    failing: dict[str, core.Outcome] = {**only_sign, "engine": "fail"}
    assert core.verdict_u2a(outcomes=failing) == "not supported"


def test_the_u2b_bars():
    assert core.calibration_bar(lo=0.6, se=0.2) == "pass"
    assert core.calibration_bar(lo=0.4, se=0.2) == "fail"
    assert core.calibration_bar(lo=0.6, se=0.5) == "void"
    assert core.reverse_flow_test(lo=-2.0, hi=-0.6, se=0.2) == "supports_reverse_flow"
    assert core.reverse_flow_test(lo=0.6, hi=2.0, se=0.2) == "refutes_reverse_flow"
    assert core.reverse_flow_test(lo=-1.0, hi=0.2, se=0.2) == "unresolved"
    assert core.reverse_flow_test(lo=-1.0, hi=0.2, se=0.5) == "void"
    assert core.positive_control_bar(lo=0.1, se=0.2) == "pass"
    assert core.positive_control_bar(lo=-0.1, se=0.2) == "fail"
    assert core.shifted_date_bar(lo=-0.3, hi=0.3, se=0.2) == "pass"
    assert core.shifted_date_bar(lo=0.1, hi=0.3, se=0.2) == "fail"


def test_u2a_maps_every_combination_to_exactly_one_verdict():
    names = ["engine", "d15_closure", "d15_sign", "d15_placebo"]
    seen = set()
    for outcomes in core.all_outcome_combinations(names=names):
        effective = dict(outcomes)
        if effective["d15_sign"] == "pass" and effective["d15_placebo"] == "void":
            effective["d15_sign"] = "void"
        values = list(effective.values())
        if "fail" in values:
            expected = "not supported"
        elif "pass" in values:
            expected = "supported"
        else:
            expected = "inconclusive"
        assert core.verdict_u2a(outcomes=outcomes) == expected
        seen.add(expected)
    assert seen == {"supported", "not supported", "inconclusive"}


def test_u2b_maps_every_combination_to_exactly_one_verdict():
    outcomes: tuple[core.Outcome, ...] = ("pass", "fail", "void")
    tests: tuple[core.TestOutcome, ...] = (
        "supports_reverse_flow",
        "refutes_reverse_flow",
        "unresolved",
        "void",
    )
    seen = set()
    for cal in outcomes:
        for test in tests:
            for c1 in outcomes:
                for c2 in outcomes:
                    controls = {"d3": c1, "shifted": c2}
                    got = core.verdict_u2b(calibration=cal, test=test, controls=controls)
                    if cal == "fail" or test == "refutes_reverse_flow" or "fail" in (c1, c2):
                        expected = "not supported"
                    elif (
                        cal == "pass"
                        and test == "supports_reverse_flow"
                        and (c1, c2) == ("pass", "pass")
                    ):
                        expected = "supported"
                    else:
                        expected = "inconclusive"
                    assert got == expected
                    seen.add(got)
    assert seen == {"supported", "not supported", "inconclusive"}


def test_u1_verdict_rule():
    for audit in (True, False):
        for align in (True, False):
            for pairs in (3, 4):
                for bars in (True, False):
                    got = core.verdict_u1(
                        audit_ok=audit, alignment_ok=align, n_pairs=pairs, bars_pass=bars
                    )
                    if not (audit and align):
                        assert got == "invalid"
                    elif pairs < 4:
                        assert got == "inconclusive"
                    else:
                        assert got == ("supported" if bars else "not supported")


def test_the_whole_day_bootstrap_recovers_a_slope_and_its_interval_covers_the_truth():
    n_days = 120
    cal = calendar(n_days=n_days)
    rng = np.random.default_rng(9)
    n = n_days * 48
    x = rng.normal(0, 1, (n, 2))
    day_effect = np.repeat(rng.normal(0, 0.5, n_days), 48)
    y = 1.5 * x[:, 0] - 0.5 * x[:, 1] + day_effect + rng.normal(0, 1, n)
    fit = core.bootstrap_ols(
        y=y, x=x, day=cal["day"], weights=make_weights(n_days=n_days), names=["a", "b"]
    )
    assert fit.lo[0] < 1.5 < fit.hi[0]
    assert fit.lo[1] < -0.5 < fit.hi[1]
    assert fit.n_rows == n
    assert fit.n_days == n_days
    assert 0.0 < fit.se[0] < 0.1


def test_the_bootstrap_of_a_subset_of_days_uses_only_those_days_and_shares_the_draw():
    n_days = 100
    cal = calendar(n_days=n_days)
    rng = np.random.default_rng(10)
    n = n_days * 48
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 1, n)
    keep = cal["day"] < 10
    w = make_weights(n_days=n_days)
    fit = core.bootstrap_ols(
        y=np.where(keep, y, np.nan),
        x=np.where(keep, x, np.nan),
        day=cal["day"],
        weights=w,
        names=["s"],
    )
    assert fit.n_days == 10
    sub = core.subset_weights(weights=w, present=np.arange(10))
    assert np.allclose(sub.sum(axis=1)[sub.sum(axis=1) > 0], 10.0)
    assert fit.lo[0] < 2.0 < fit.hi[0]


def test_the_weighted_area_under_the_curve_handles_ties_and_the_bootstrap_interval():
    assert (
        core.weighted_auc(
            score=np.array([1.0, 2.0, 3.0, 4.0]), label=np.array([0, 0, 1, 1]), w=np.ones(4)
        )
        == 1.0
    )
    assert core.weighted_auc(score=np.ones(4), label=np.array([0, 1, 0, 1]), w=np.ones(4)) == 0.5
    rng = np.random.default_rng(11)
    n_days = 80
    cal = calendar(n_days=n_days)
    label = rng.integers(0, 2, n_days * 48).astype(float)
    score = rng.normal(0, 1, n_days * 48) + label
    point, se, lo, hi = core.bootstrap_auc(
        score=score, label=label, day=cal["day"], weights=make_weights(n_days=n_days, n_boot=100)
    )
    assert lo < point < hi
    assert 0.6 < point < 0.85
    assert se > 0


def test_matched_demeaning_uses_quiet_rows_only_and_needs_enough_of_them():
    n = 48 * 40
    key = (np.arange(n) % 48).astype(np.int64)
    base = 3.0 * (np.arange(n) % 48 == 5)
    x = base.astype(float) + 1.0
    quiet = np.ones(n, dtype=bool)
    out = core.demean_matched(x=x, key=key, quiet=quiet)
    assert np.allclose(out, 0.0)
    x2 = x.copy()
    x2[5 + 48 * 3] = 100.0
    quiet2 = quiet.copy()
    quiet2[5 + 48 * 3] = False
    out2 = core.demean_matched(x=x2, key=key, quiet=quiet2)
    assert np.isclose(out2[5 + 48 * 3], 96.0)
    assert np.allclose(np.delete(out2, 5 + 48 * 3), 0.0)
    sparse = core.demean_matched(x=x, key=key, quiet=np.arange(n) < 48 * 3)
    assert np.isnan(sparse).all()


def test_quiet_mask_keeps_half_hours_far_from_engine_changes():
    change = np.zeros(100, dtype=bool)
    change[50] = True
    q = core.quiet_mask(engine_change=change, calendar_ok=np.ones(100, dtype=bool))
    assert not q[44:57].any()
    assert q[43]
    assert q[57]
    assert not core.quiet_mask(engine_change=change, calendar_ok=np.zeros(100, dtype=bool)).any()


def test_hourly_sign_rows_exclude_crossing_hours_and_small_changes():
    n = 48 * 100
    rng = np.random.default_rng(12)
    flow = block_flow(n=n, rng=rng)
    folded = np.abs(flow)
    d_m = core.change(x=folded, lag=2)
    d_r = core.change(x=flow, lag=2)
    hour_end = np.arange(n) % 2 == 1
    table = core.hourly_sign_rows(d_residual=d_r, d_folded=d_m, flow=flow, hour_end=hour_end)
    assert table["scorable"].sum() > 50
    assert table["crossing"].sum() > 10
    scored = table["scorable"]
    assert np.all(table["label"][scored] == (flow[scored] > 0))
    assert np.all(np.abs(table["ratio"][scored]) > 0.9)

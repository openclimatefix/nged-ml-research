"""Known-answer tests of the scores, the five solar bars, the baselines, and the injection rows (no torch)."""

import e00729_score as score729
import e00732_baselines as bl
import e00732_core as core
import e00732_score as sc
import numpy as np
import pytest
from world732 import make_bank


@pytest.fixture(scope="module")
def rows():
    cal, segs = make_bank(draws_per_world=3)
    return score729.rows_from_segments(segments=segs, holidays=cal.holidays)


def test_recovery_bar_known_answers_and_empty_fails() -> None:
    delta = np.full((2, 48), 20.0)
    keep = np.ones((2, 48), bool)
    pv = np.zeros((2, 48))
    ok = sc.recovery_bar(pv_inj=pv + 20.0, pv_plain=pv, delta=delta, amount=20.0, keep=keep)
    assert ok["pass"] and ok["rho"] == pytest.approx(1.0)
    under = sc.recovery_bar(pv_inj=pv + 15.0, pv_plain=pv, delta=delta, amount=20.0, keep=keep)
    assert not under["pass"] and under["rho"] == pytest.approx(0.75)
    wide = np.where(np.arange(48) % 2 == 0, 25.0, 15.0) * np.ones(
        (2, 48)
    )  # rho 1.0 but median error 0.25 a
    assert not sc.recovery_bar(pv_inj=wide, pv_plain=pv, delta=delta, amount=20.0, keep=keep)[
        "pass"
    ]
    assert not sc.recovery_bar(
        pv_inj=pv, pv_plain=pv, delta=delta, amount=20.0, keep=np.zeros((2, 48), bool)
    )["pass"]


def test_injection_keep_needs_a_pair_sun_and_five_percent_of_the_amount() -> None:
    delta = np.array([[0.5, 1.0, 1.0, 1.0]])
    cosz = np.array([[0.5, 0.5, 0.0, 0.5]])
    keep = sc.injection_keep(delta=delta, amount=20.0, cosz=cosz, pair=np.array([1.0]))
    assert keep.tolist() == [[False, True, False, True]]
    assert not sc.injection_keep(delta=delta, amount=20.0, cosz=cosz, pair=np.array([0.0])).any()


def test_night_bar() -> None:
    cosz = np.array([[-0.5] * 10 + [0.5] * 10])
    valid = np.ones_like(cosz, bool)
    quiet = np.zeros_like(cosz)
    assert sc.night_bar(pv=quiet, cosz=cosz, valid=valid)["pass"]
    leak = quiet.copy()
    leak[0, 0] = 1.0  # 1 of 10 night half-hours above 0.5 points: share 0.1
    assert not sc.night_bar(pv=leak, cosz=cosz, valid=valid)["pass"]
    assert not sc.night_bar(pv=quiet, cosz=np.ones_like(cosz), valid=valid)[
        "pass"
    ]  # no night rows: fails


def test_monotone_and_response_bars() -> None:
    raisable = np.ones((1, 20), bool)
    pv = np.full((1, 20), 5.0)
    assert sc.monotone_bar(pv=pv, pv_raised=pv + 1.0, raisable=raisable)["pass"]
    bad = sc.monotone_bar(pv=pv, pv_raised=pv - 1.0, raisable=raisable)
    assert not bad["pass"] and bad["share"] == 1.0
    assert not sc.monotone_bar(pv=pv, pv_raised=pv, raisable=np.zeros_like(raisable))["pass"]
    target = np.full((1, 20), 3.0)
    assert sc.response_bar(pv=pv, pv_raised=pv + 1.0, raisable=raisable, solar_target=target)[
        "pass"
    ]
    assert not sc.response_bar(pv=pv, pv_raised=pv, raisable=raisable, solar_target=target)["pass"]
    assert not sc.response_bar(
        pv=pv, pv_raised=pv + 1.0, raisable=raisable, solar_target=target * 0.1
    )["pass"]  # no rows


def test_step_f1_tolerates_one_half_hour_and_empty_truth_is_nan() -> None:
    event = np.zeros((1, 48))
    size = np.zeros((1, 48))
    event[0, 10], size[0, 10] = 1.0, 10.0
    event[0, 30], size[0, 30] = 1.0, 2.0  # small step: not in the F1 positives
    p = np.zeros((1, 48))
    p[0, 11] = 0.9  # one half-hour late: still found
    p[0, 40] = 0.9  # a false call
    valid = np.ones((1, 48), bool)
    r = sc.step_f1(step_p=p, event=event, size=size, valid=valid)
    assert r["recall"] == 1.0 and r["precision"] == 0.5 and r["f1"] == pytest.approx(2 / 3)
    p2 = np.zeros((1, 48))
    p2[0, 13] = 0.9
    assert sc.step_f1(step_p=p2, event=event, size=size, valid=valid)["recall"] == 0.0
    assert np.isnan(sc.step_f1(step_p=p, event=np.zeros((1, 48)), size=size, valid=valid)["f1"])


def test_transfer_scenarios_partition_the_valid_rows() -> None:
    inside = np.zeros((1, 6), bool)
    inside[0, 2] = True
    since = np.array([[np.inf, 3.0, 0.0, 0.0, 10.0, 6.9]])
    valid = np.ones((1, 6), bool)
    m = sc.transfer_scenarios(inside=inside, since_end_days=since, valid=valid)
    assert m["inside"].sum() == 1 and m["after"][0].tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert m["outside"][0].tolist() == [True, False, False, False, True, False]
    assert (m["outside"] & m["after"]).sum() == 0 and (m["outside"] & m["inside"]).sum() == 0


def test_size_error_reads_only_true_edges() -> None:
    event = np.array([[0.0, 1.0, 0.0]])
    out = sc.size_error(
        size_hat=np.array([[9.0, 4.0, 9.0]]),
        size=np.array([[0.0, 6.0, 0.0]]),
        event=event,
        valid=np.ones((1, 3), bool),
    )
    assert out == 2.0
    assert np.isnan(
        sc.size_error(size_hat=event, size=event, event=event * 0, valid=np.ones((1, 3), bool))
    )


def test_harm_by_band_uses_each_bands_own_limit(rows) -> None:
    limits = core.band_limits(x=rows.x, r=rows.r)
    p = np.zeros(rows.n)
    out = sc.harm_by_band(rows=rows, p=p, limits=limits)
    assert set(out) == set(core.BANDS)
    assert all(v["mean_p"] == 0.0 for v in out.values() if v["n"])
    masks = sc.band_masks(rows=rows)
    assert not (masks["1-3MVA"] & masks["3MVA+"]).any()
    assert (masks["1-3MVA"] | masks["3MVA+"]).sum() == masks["pooled"].sum()
    assert sc.band_boundary(band={"pass": True}, b13_lower=1.0) == "pass"
    assert sc.band_boundary(band={"pass": False}, b13_lower=0.1) == "fail"
    assert sc.band_boundary(band={"pass": False}, b13_lower=-0.1) == "at the calibrated boundary"


def test_injection_rows_are_keyed_by_amount_and_duplicates_raise(rows) -> None:
    stacked = sc.stack_amounts(rows=rows, amounts=core.TEST_INJECTION_POINTS)
    assert stacked.n == 3 * rows.n and sc.keys_are_unique(rows=stacked)
    dup = sc.stack_amounts(rows=rows, amounts=(5.0, 5.0))
    assert not sc.keys_are_unique(rows=dup)
    a = np.zeros(dup.n)
    with pytest.raises(ValueError, match="repeat"):
        sc.injection_contrast(rows=dup, a=a, b=a, keep=np.ones(dup.n, bool), n_resamples=50)
    a = np.zeros(stacked.n)
    reading = sc.injection_contrast(
        rows=stacked, a=a, b=a + 1.0, keep=np.ones(stacked.n, bool), n_resamples=200
    )
    assert reading.reading.point == pytest.approx(-1.0) and reading.reading.upper < 0


def test_contrast_of_reads_a_known_shift(rows) -> None:
    a = np.zeros(rows.n)
    r = sc.contrast_of(rows=rows, a=a, b=a + 0.5, n_resamples=200)
    assert r.reading.point == pytest.approx(-0.5) and r.n_rows == rows.n
    assert sc.log_loss_opposes(brier_point=-1.0, log_loss_point=0.5) and not sc.log_loss_opposes(
        brier_point=-1.0, log_loss_point=-0.5
    )


def test_c4_subsets_are_by_world(rows) -> None:
    sub = sc.c4_subsets(rows=rows)
    assert sub["all"].all() and sub["solar_only"].any() and sub["no_injection"].any()
    assert not (sub["solar_only"] & sub["no_injection"]).any()


def test_signed_reading_estimate_and_b_pv2() -> None:
    r = np.array([5.0, 5.0, 0.05])
    p = np.array([0.9, 0.1, 0.9])
    x_hat = bl.signed_reading_estimate(p=p, r=r)
    assert x_hat[0] < 0 < x_hat[1] and x_hat[2] == 0.0  # R below Q-hat: M is zero
    cosz = np.array([0.5, 0.5, 0.5])
    est = bl.solar_from_demand(demand=np.array([20.0, 20.0, 20.0]), p=p, r=r, cosz=cosz)
    assert est[0] == pytest.approx(20.0 + np.sqrt(25 - core.Q_HAT_POINTS**2)) and est[
        1
    ] == pytest.approx(20.0 - np.sqrt(25 - core.Q_HAT_POINTS**2))
    assert (
        bl.solar_from_demand(
            demand=np.array([20.0]), p=np.array([0.1]), r=np.array([5.0]), cosz=np.array([-0.2])
        )[0]
        == 0.0
    )
    assert (
        bl.solar_from_demand(
            demand=np.array([1.0]), p=np.array([0.1]), r=np.array([5.0]), cosz=np.array([0.5])
        )
        >= 0
    ).all()
    with pytest.raises(ValueError, match="three seeds"):
        bl.b_pv2(demand_by_seed=[r], p_by_seed=[p], r=r, cosz=cosz)
    seeds = [r * 4] * 3
    both = bl.b_pv2(demand_by_seed=seeds, p_by_seed=[p] * 3, r=r, cosz=cosz)
    np.testing.assert_allclose(both, bl.solar_from_demand(demand=r * 4, p=p, r=r, cosz=cosz))
    with pytest.raises(ValueError, match="three seeds"):
        bl.b_pv1(demand_median=r, p_by_seed=[p], r=r, cosz=cosz)


def test_logistic_regression_recovers_a_planted_rule() -> None:
    rng = np.random.default_rng(0)
    f = rng.normal(size=(4000, 5))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(2.0 * f[:, 0] - 1.0)))).astype(float)
    w = bl.fit_logistic(features=f, y=y)
    assert (
        w[1] == pytest.approx(2.0, abs=0.3)
        and w[0] == pytest.approx(-1.0, abs=0.3)
        and abs(w[3]) < 0.3
    )
    p = bl.predict_logistic(features=f, weights=w)
    assert float(np.mean((p - y) ** 2)) < float(np.mean((y.mean() - y) ** 2))
    with pytest.raises(ValueError, match="with and without"):
        bl.fit_logistic(features=f, y=np.zeros(4000))


def test_bt_features_shape_and_zero_edge_rows() -> None:
    r = np.tile(np.arange(48.0), (2, 1))
    f = bl.bt_features(
        r=r, bsp=np.zeros((2, 48)), cosz=np.ones((2, 48)), valid=np.ones((2, 48), bool)
    )
    assert f.shape == (2, 48, len(bl.BT_FEATURES))
    assert (
        (f[:, 0, 0] == 0).all()
        and (f[:, 1:, 0] == 1.0).all()
        and (f[:, 1:, 1] == pytest.approx(1.0))
    )
    bad = np.ones((2, 48), bool)
    bad[:, 10] = False
    g = bl.bt_features(r=r, bsp=np.zeros((2, 48)), cosz=np.ones((2, 48)), valid=bad)
    assert (g[:, 10, 0] == 0).all() and (g[:, 11, 0] == 0).all()

# ruff: noqa: E501
"""Known-answer tests of the scores, the eight solar bars, the readings, the stored-prediction
links, and the verdict (no torch).
"""

import e00729_arms as arms729
import e00729_score as score729
import e00732_arms as arms
import e00732_core as core
import e00732_neighbours as nb
import e00732_overlays as ov
import e00732_score as sc
import numpy as np
import pytest
from world732 import make_bank, make_neighbours


@pytest.fixture(scope="module")
def rows():
    cal, segs = make_bank(draws_per_world=3)
    return score729.rows_from_segments(segments=segs, holidays=cal.holidays)


def test_recovery_bar_known_answers_and_empty_fails() -> None:
    delta = np.full((2, 48), 20.0)
    keep = np.ones((2, 48), bool)
    pv = np.zeros((2, 48))
    ok = sc.recovery_bar(pv_inj=pv + 20.0, pv_plain=pv, delta=delta, amount=20.0, keep=keep)
    assert ok["pass"]
    assert ok["rho"] == pytest.approx(1.0)
    under = sc.recovery_bar(pv_inj=pv + 15.0, pv_plain=pv, delta=delta, amount=20.0, keep=keep)
    assert not under["pass"]
    assert under["rho"] == pytest.approx(0.75)
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
    assert not bad["pass"]
    assert bad["share"] == 1.0
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
    assert r["recall"] == 1.0
    assert r["precision"] == 0.5
    assert r["f1"] == pytest.approx(2 / 3)
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
    assert m["inside"].sum() == 1
    assert m["after"][0].tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert m["outside"][0].tolist() == [True, False, False, False, True, False]
    assert (m["outside"] & m["after"]).sum() == 0
    assert (m["outside"] & m["inside"]).sum() == 0


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


def test_harm_by_band_uses_each_bands_own_limit(rows: score729.Rows) -> None:
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


def test_injection_rows_are_keyed_by_amount_and_duplicates_raise(rows: score729.Rows) -> None:
    stacked = sc.stack_amounts(rows=rows, amounts=core.TEST_INJECTION_POINTS)
    assert stacked.n == 3 * rows.n
    assert sc.keys_are_unique(rows=stacked)
    dup = sc.stack_amounts(rows=rows, amounts=(5.0, 5.0))
    assert not sc.keys_are_unique(rows=dup)
    a = np.zeros(dup.n)
    with pytest.raises(ValueError, match="repeat"):
        sc.injection_contrast(rows=dup, a=a, b=a, keep=np.ones(dup.n, bool), n_resamples=50)
    a = np.zeros(stacked.n)
    reading = sc.injection_contrast(
        rows=stacked, a=a, b=a + 1.0, keep=np.ones(stacked.n, bool), n_resamples=200
    )
    assert reading.reading.point == pytest.approx(-1.0)
    assert reading.reading.upper < 0


def test_contrast_of_reads_a_known_shift(rows: score729.Rows) -> None:
    a = np.zeros(rows.n)
    r = sc.contrast_of(rows=rows, a=a, b=a + 0.5, n_resamples=200)
    assert r.reading.point == pytest.approx(-0.5)
    assert r.n_rows == rows.n
    assert sc.log_loss_opposes(brier_point=-1.0, log_loss_point=0.5)
    assert not sc.log_loss_opposes(brier_point=-1.0, log_loss_point=-0.5)


def test_c4_subsets_are_by_world(rows: score729.Rows) -> None:
    sub = sc.c4_subsets(rows=rows)
    assert sub["all"].all()
    assert sub["solar_only"].any()
    assert sub["no_injection"].any()
    assert not (sub["solar_only"] & sub["no_injection"]).any()


# --------------------------------------------------------------------------- the real overlay path


class World:
    """The invented bank pushed through the real overlay, prepare, and row builders."""

    def __init__(self) -> None:
        cal, segs = make_bank(draws_per_world=3)
        series = make_neighbours(cal=cal)
        self.rows = score729.rows_from_segments(segments=segs, holidays=cal.holidays)
        overlays = ov.overlay_segments(segments=segs, kind="test", seed=None, neighbours=series)
        everyone = np.arange(len(cal.dates))
        stats = arms729.input_stats(cal=cal, train_days=everyone)
        medians = {
            n: nb.training_median(folded=nb.fold(signed=series[n], floor=0.0), train_rows=everyone)
            for n in nb.NEIGHBOURS
        }
        self.prepared = arms.prepare(
            overlays=overlays, holidays=cal.holidays, stats=stats, medians=medians
        )
        self.picks = arms.all_picks(prepared=self.prepared)
        self.valid = (
            np.stack(
                [
                    arms.target_window(p=self.prepared[i], day=d)[arms.T["valid"]]
                    for i, d in self.picks
                ]
            )
            > 0.5
        )
        self.tr = sc.transfer_rows(
            transfers=[p.transfers for p in self.prepared],
            picks=self.picks,
            valid=self.valid,
            noise_d8=2.0,
        )


@pytest.fixture(scope="module")
def world() -> World:
    return World()


def test_transfer_rows_follow_the_real_transfers(world: World) -> None:
    tr = world.tr
    assert tr.edge.shape[0] == world.rows.n == int(world.valid.sum())
    assert tr.edge.any()
    assert tr.inside.any()
    assert (tr.edge_points[tr.edge] > 0).all()
    assert (tr.edge_points[~tr.edge] == 0).all()
    assert set(np.unique(tr.edge_case[tr.edge])) <= {0, 1, 2, 3}
    assert (tr.edge_case[~tr.edge] == -1).all()
    assert np.allclose(tr.edge_noise, tr.edge_points / 2.0)
    assert set(np.unique(tr.edge_direction)) <= {-1.0, 0.0, 1.0}
    # an edge half-hour of a transfer lies inside its own window
    assert tr.win_edge[tr.edge].all()
    # a half-hour inside a transfer carries that transfer's partner case
    assert (tr.win_case[tr.inside] >= 0).all()
    # the edge count equals the count of transfer edges whose day is scored and whose half-hour is valid
    expected = 0
    where = {pick: k for k, pick in enumerate(world.picks)}
    for i, p in enumerate(world.prepared):
        for t in p.transfers:
            for e in (t.start_half_hour, t.end_half_hour):
                day, hh = divmod(e, 48)
                if (i, day) in where and world.valid[where[i, day], hh]:
                    expected += 1
    assert int(tr.edge.sum()) <= expected  # two edges that share a half-hour count once


def test_partner_case_reads_the_partner_and_the_jitter(world: World) -> None:
    for p in world.prepared:
        for t in p.transfers:
            case = sc.partner_case(transfer=t)
            if t.mirrored:
                assert case == (0 if t.jitter == 1.0 else 1)
            else:
                assert case == (2 if t.inside_bsp_a else 3)


def test_step_table_counts_a_planted_edge_in_its_band_and_fails_when_silent(world: World) -> None:
    tr = world.tr
    event = tr.edge.astype(float)
    size = np.where(tr.edge, tr.edge_direction * tr.edge_points, 0.0)
    perfect = sc.step_table(step_p=event, size_hat=size, event=event, size=size, tr=tr)
    silent = sc.step_table(
        step_p=np.zeros_like(event), size_hat=None, event=event, size=size, tr=tr
    )
    for group, bands in perfect.items():
        for label, reading in bands.items():
            if reading["n_true"]:
                assert reading["recall"] == 1.0, (group, label)
                assert reading["size_error"] == 0.0
                assert silent[group][label]["recall"] == 0.0
                assert np.isnan(silent[group][label]["size_error"])
    total = sum(r["n_true"] for r in perfect["direction"].values())
    assert total == int(tr.edge.sum())
    assert sc.precision_all(step_p=event, event=event) == 1.0
    assert sc.precision_all(step_p=np.zeros_like(event), event=event) == 0.0


def test_detectability_is_void_under_100_steps_a_side_and_names_the_bands() -> None:
    table = {"2_to_4": {"recall": 0.7}, "4_to_8": {"recall": 0.9}, "under_2": {"recall": 0.1}}
    out = sc.detectability(table=table, precision=0.8, counts={"below": 150, "above": 150})
    assert not out["void"]
    assert out["smallest_detectable"] == "2_to_4"
    assert (
        sc.detectability(table=table, precision=0.3, counts={"below": 150, "above": 150})[
            "detectable"
        ]
        == []
    )
    assert sc.detectability(table=table, precision=0.8, counts={"below": 99, "above": 150})["void"]
    assert sc.detectability(table=table, precision=0.8, counts={})["smallest_detectable"] is None


def make_truth(*, world: World) -> sc.Truth:
    rows = world.rows
    rng = np.random.default_rng(3)
    cosz = np.where(np.arange(rows.n) % 48 < 12, -0.3, 0.6)
    delta = {a: np.where(rng.random(rows.n) < 0.5, a, 0.0) for a in core.TEST_INJECTION_POINTS}
    return sc.Truth(
        rows=rows,
        x_tr=rows.x,
        event=world.tr.edge.astype(float),
        size=np.where(world.tr.edge, world.tr.edge_direction * world.tr.edge_points, 0.0),
        cosz=cosz,
        raisable=cosz > 0.1,
        delta=delta,
        pair={a: np.ones(rows.n) for a in core.TEST_INJECTION_POINTS},
        tr=world.tr,
    )


def make_arm(*, truth: sc.Truth, better: bool) -> sc.ArmScores:
    rows = truth.rows
    sharp = 0.05 if better else 0.3
    p = np.clip((rows.x < 0).astype(float) * (1 - 2 * sharp) + sharp, 0, 1)
    step = np.clip(truth.event * (1 - 2 * sharp) + sharp, 0, 1)
    pv = np.where(truth.cosz > 0.1, rows.solar + (0.0 if better else 2.0), 0.0)
    return sc.ArmScores(
        p=p,
        p_tr=p,
        p_nobsp=p,
        pv=pv,
        pv_tr=pv,
        pv_inj={
            a: pv + truth.delta[a] * (1.0 if better else 0.5) for a in core.TEST_INJECTION_POINTS
        },
        pv_raised=pv + 1.0,
        step=step,
        size_hat=truth.size,
    )


def test_compute_readings_reads_each_contrast_on_the_same_rows_and_voids_missing_arms(
    world: World,
) -> None:
    truth = make_truth(world=world)
    arms_ = {
        "U*": make_arm(truth=truth, better=True),
        "S": make_arm(truth=truth, better=False),
        "S+": make_arm(truth=truth, better=False),
        "U-NC": make_arm(truth=truth, better=False),
        "B-PV2": make_arm(truth=truth, better=False),
        "B-TN": make_arm(truth=truth, better=False),
        "U-N": make_arm(truth=truth, better=False),
    }
    out = sc.compute_readings(truth=truth, arms=arms_, n_resamples=200)
    counted = out["counted"]
    assert set(counted) == {f"C{k}" for k in range(1, 8)}  # ty: ignore[invalid-argument-type]
    for name in counted:  # ty: ignore[not-iterable]
        assert counted[name] is not None, name  # ty: ignore[not-subscriptable]
        assert counted[name].reading.point < 0, name  # U* is better in every contrast  # ty: ignore[not-subscriptable]
    assert counted["C3"].n_rows == int(sc.injection_keep_stack(truth=truth).sum())  # ty: ignore[not-subscriptable]
    assert counted["C1"].n_rows == world.rows.n  # ty: ignore[not-subscriptable]
    assert out["hb"] is not None
    report = out["report"]
    assert report["R1"] is None  # U-SG is absent: void, not zero  # ty: ignore[not-subscriptable]
    assert report["R19"] is None  # U-noBSP is absent  # ty: ignore[not-subscriptable]
    with_bsp = sc.compute_readings(
        truth=truth, arms={**arms_, "U-noBSP": make_arm(truth=truth, better=False)}, n_resamples=100
    )
    assert with_bsp["report"]["R19"] is not None  # ty: ignore[not-subscriptable]
    assert with_bsp["report"]["R19"].reading.point < 0  # ty: ignore[not-subscriptable]
    assert counted["C4"].n_rows == int((sc.sun_up(cosz=truth.cosz)).sum())  # ty: ignore[not-subscriptable]
    # a NaN output on a kept row makes the contrast refuse, never read zero
    bad = make_arm(truth=truth, better=True)
    bad.pv = bad.pv.copy()  # ty: ignore[unresolved-attribute]
    bad.pv[np.flatnonzero(sc.sun_up(cosz=truth.cosz))[0]] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sc.compute_readings(truth=truth, arms={**arms_, "U*": bad}, n_resamples=50)


def test_hb_uses_the_one_sided_level_and_is_narrower_than_the_counted_level(world: World) -> None:
    rng = np.random.default_rng(5)
    p_s = rng.random(world.rows.n) * 0.2
    p_u = p_s + rng.normal(0, 0.05, world.rows.n)
    hb = sc.harm_hb(rows=world.rows, p_u=p_u, p_s=p_s, n_resamples=300)
    keep = sc.band_masks(rows=world.rows)["1-3MVA"]
    counted = sc.contrast_at_level(rows=world.rows, a=p_u, b=p_s, keep=keep, n_resamples=300)
    assert pytest.approx(1 - 2 * (1 - core.LEVEL)) == sc.HB_LEVEL
    assert hb.n_rows == int(keep.sum())
    assert hb.reading.lower >= counted.reading.lower
    assert hb.reading.point == pytest.approx(counted.reading.point)


def test_contrasts_use_generator_seed_732_and_refuse_non_finite_differences(world: World) -> None:
    rows = world.rows
    rng = np.random.default_rng(1)
    a, b = rng.random(rows.n), rng.random(rows.n)
    keep = np.ones(rows.n, bool)
    got = sc.contrast_at_level(rows=rows, a=a, b=b, keep=keep, n_resamples=200)
    table = score729.cell_table(rows=rows, diff=a - b, keep=keep)
    want = score729.read_contrast(table=table, n_resamples=200, seed=732, level=core.LEVEL)
    assert got.reading == want.reading
    other = score729.read_contrast(table=table, n_resamples=200, seed=730, level=core.LEVEL)
    assert got.reading != other.reading
    b_bad = b.copy()
    b_bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sc.contrast_at_level(rows=rows, a=a, b=b_bad, keep=keep, n_resamples=50)
    keep[3] = False  # the NaN row is not read
    sc.contrast_at_level(rows=rows, a=a, b=b_bad, keep=keep, n_resamples=50)


# --------------------------------------------------------------------------- the eight bars


def test_every_bar_fails_on_a_nan_and_on_an_empty_input() -> None:
    n = 20
    ones, up = np.ones((1, n)), np.full((1, n), 0.6)
    nan = np.full((1, n), np.nan)
    keep = np.ones((1, n), bool)
    assert not sc.recovery_bar(
        pv_inj=nan, pv_plain=ones * 0, delta=ones * 5, amount=5.0, keep=keep
    )["pass"]
    assert not sc.night_bar(pv=nan, cosz=up * -1, valid=keep)["pass"]
    assert not sc.monotone_bar(pv=nan, pv_raised=ones, raisable=keep)["pass"]
    assert not sc.response_bar(pv=ones, pv_raised=nan, raisable=keep, solar_target=ones * 3)["pass"]
    assert not sc.absolute_bar(pv=nan, solar=ones, cosz=up, world_is_solar=keep)["pass"]
    assert not sc.recovery_reverse_bar(
        pv_inj={5.0: nan}, pv_plain=ones * 0, delta={5.0: ones * 5}, keep={5.0: keep}, reverse=keep
    )["pass"]
    assert not sc.recovery_reverse_bar(
        pv_inj={5.0: ones},
        pv_plain=ones * 0,
        delta={5.0: ones * 5},
        keep={5.0: keep},
        reverse=~keep,
    )["pass"]  # no reverse-flow half-hour: nothing read
    assert not sc.gain_bar(reading=None)["pass"]


def test_absolute_bar_known_answers_and_a_silent_solar_world_fails() -> None:
    n = 10
    solar, cosz, world = np.full(n, 10.0), np.full(n, 0.6), np.ones(n, bool)
    assert sc.absolute_bar(pv=solar + 4.0, solar=solar, cosz=cosz, world_is_solar=world)["pass"]
    assert not sc.absolute_bar(pv=solar + 6.0, solar=solar, cosz=cosz, world_is_solar=world)["pass"]
    zero = np.zeros(n)
    assert not sc.absolute_bar(pv=zero, solar=zero, cosz=cosz, world_is_solar=world)["pass"]
    assert not sc.absolute_bar(pv=solar, solar=solar, cosz=cosz, world_is_solar=~world)["pass"]


def test_recovery_reverse_bar_reads_only_reverse_flow_half_hours() -> None:
    n = 40
    reverse = np.arange(n) % 2 == 0
    keep = {a: np.ones(n, bool) for a in (5.0, 20.0)}
    delta = {a: np.full(n, a) for a in keep}
    plain = np.zeros(n)
    exact = {a: plain + np.where(reverse, a, 0.0) for a in keep}
    assert sc.recovery_reverse_bar(
        pv_inj=exact, pv_plain=plain, delta=delta, keep=keep, reverse=reverse
    )["pass"]
    weak = {
        a: plain + np.where(reverse, 0.5 * a, a) for a in keep
    }  # fine off reverse flow, 0.5 on it
    r = sc.recovery_reverse_bar(
        pv_inj=weak, pv_plain=plain, delta=delta, keep=keep, reverse=reverse
    )
    assert not r["pass"]
    assert r["rho"] == pytest.approx(0.5)


def test_solar_bars_need_all_eight_and_a_missing_amount_fails() -> None:
    good = {"pass": True}
    recovery = dict.fromkeys(core.TEST_INJECTION_POINTS, good)
    kw = {
        "recovery": recovery,
        "recovery_reverse": good,
        "absolute": good,
        "sky": good,
        "night": good,
        "monotone": good,
        "response": good,
        "demand": good,
    }
    flags = sc.solar_bars(**kw)  # ty: ignore[invalid-argument-type]
    assert list(flags) == list(core.SOLAR_BARS)
    assert all(flags.values())
    assert core.solar_word(bars=flags, reachable=True) == "reasonable"
    for name in kw:
        flags = sc.solar_bars(**{**kw, name: None})  # ty: ignore[invalid-argument-type]
        assert not all(flags.values()), name
    short = sc.solar_bars(**{**kw, "recovery": {5.0: good, 20.0: good}})  # ty: ignore[invalid-argument-type]
    assert not short["recovery"]
    assert not sc.solar_bars(**{**kw, "recovery": {**recovery, 50.0: {"pass": False}}})["recovery"]  # ty: ignore[invalid-argument-type]


def test_gain_bar_reads_the_upper_end(world: World) -> None:
    a = np.zeros(world.rows.n)
    gain = sc.contrast_of(rows=world.rows, a=a, b=a + 0.5, n_resamples=100)
    flat = sc.contrast_of(rows=world.rows, a=a, b=a, n_resamples=100)
    assert sc.gain_bar(reading=gain)["pass"]
    assert not sc.gain_bar(reading=flat)["pass"]


# --------------------------------------------------------------------------- links to the stored predictions


def stored(*, rows: score729.Rows, p: np.ndarray) -> dict[str, list[dict[str, np.ndarray]]]:
    return {"S": [{"p": p.astype(np.float32)} for _ in range(3)]}


def test_stored_prediction_links_fail_on_rows_digest_and_canary(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = world.rows
    p = np.full(rows.n, 0.25)
    digest = sc.row_key_digest(rows=rows)
    assert sc.row_key_digest(rows=rows) == digest
    assert sc.row_key_digest(rows=sc.with_overrides(rows=rows, hh=(rows.hh + 1) % 48)) != digest
    monkeypatch.setattr(core, "N_TEST_ROWS", rows.n)
    brier = round(float(np.mean((p - (rows.x < 0)) ** 2)), core.BRIER_CANARY_DECIMALS)
    monkeypatch.setattr(core, "BRIER_CANARY", brier)
    assert (
        sc.stored_prediction_problems(
            arrays=stored(rows=rows, p=p), rows=rows, recorded_digest=digest
        )
        == []
    )
    bad_digest = sc.stored_prediction_problems(
        arrays=stored(rows=rows, p=p), rows=rows, recorded_digest="0" * 64
    )
    assert any("digest" in m for m in bad_digest)
    assert any(
        "digest" in m
        for m in sc.stored_prediction_problems(
            arrays=stored(rows=rows, p=p), rows=rows, recorded_digest=None
        )
    )
    short = sc.stored_prediction_problems(
        arrays=stored(rows=rows, p=p[:-1]), rows=rows, recorded_digest=digest
    )
    assert any("row count" in m for m in short)
    drifted = sc.stored_prediction_problems(
        arrays=stored(rows=rows, p=p * 0.5), rows=rows, recorded_digest=digest
    )
    assert any("canary" in m for m in drifted)
    assert any(
        "no stored" in m
        for m in sc.stored_prediction_problems(arrays={"S": []}, rows=rows, recorded_digest=digest)
    )
    monkeypatch.setattr(core, "N_TEST_ROWS", rows.n + 1)
    assert any(
        "row count" in m
        for m in sc.stored_prediction_problems(
            arrays=stored(rows=rows, p=p), rows=rows, recorded_digest=digest
        )
    )


def test_overlay_reachability_voids_but_never_stops() -> None:
    ok = sc.overlay_reachability(
        n_transfer_steps=core.MIN_TRANSFER_STEPS_TEST,
        windows_by_amount=dict.fromkeys(core.TEST_INJECTION_POINTS, core.MIN_WINDOWS_PER_AMOUNT),
    )
    assert all(ok.values())
    short = sc.overlay_reachability(
        n_transfer_steps=core.MIN_TRANSFER_STEPS_TEST - 1,
        windows_by_amount={
            5.0: core.MIN_WINDOWS_PER_AMOUNT,
            20.0: 0,
            50.0: core.MIN_WINDOWS_PER_AMOUNT,
        },
    )
    assert not short["transfer_reachable"]
    assert not short["injection_reachable"]
    assert not sc.overlay_reachability(n_transfer_steps=10**6, windows_by_amount={5.0: 10**6})[
        "injection_reachable"
    ]


# --------------------------------------------------------------------------- the verdict


def reading(*, point: float, lower: float, upper: float) -> score729.ContrastReading:
    return score729.ContrastReading(
        reading=core.core729.Reading(point, lower, upper), raw_lower=lower, raw_upper=upper,
        kish_blocks=5.0, kish_segments=5.0, widening=1.0, n_rows=100, n_resamples=10,
    )  # fmt: skip


PASS = {"pass": True, "n": 10}


def facts(**over: object) -> sc.VerdictFacts:
    base: dict[str, object] = {
        "margin": 0.001,
        "h1": {"pooled": PASS, "3MVA+": PASS, "1-3MVA": PASS},
        "h2": PASS,
        "hb": reading(point=0.0, lower=-0.01, upper=0.01),
        "c": {f"C{k}": reading(point=-0.1, lower=-0.2, upper=-0.05) for k in range(1, 8)},
        "r4": reading(point=0.0, lower=-0.01, upper=0.01),
        "r19": reading(point=-0.1, lower=-0.2, upper=-0.05),
        "solar_bars": dict.fromkeys(core.SOLAR_BARS, True),
        "injection_reachable": True,
        "transfer_reachable": True,
        "f1_big": 0.9,
        "n_bsp_reverse": 10_000,
    }
    base.update(over)
    return sc.VerdictFacts(**base)  # ty: ignore[invalid-argument-type]


def test_verdict_rows_follow_the_table_in_order() -> None:
    c = lambda **k: {**facts().c, "C1": reading(**k)}  # noqa: E731
    assert sc.build_verdict(facts=facts(c=c(point=-0.1, lower=-0.2, upper=0.0)))["row"] == 4
    assert sc.build_verdict(facts=facts(c=c(point=0.0, lower=-0.1, upper=0.1)))["row"] == 5
    assert sc.build_verdict(facts=facts(c=c(point=0.2, lower=0.1, upper=0.3)))["row"] == 6
    assert sc.build_verdict(facts=facts(stopped="a stop"))["row"] == 0
    assert sc.build_verdict(facts=facts(invalid="leak", stopped=None))["row"] == 1
    assert sc.build_verdict(facts=facts(silent_test_missed=True))["row"] == 1
    assert sc.build_verdict(facts=facts(failed="a fit raised"))["row"] == 2
    harm = {"pooled": {"pass": False, "n": 10}, "3MVA+": PASS, "1-3MVA": PASS}
    out = sc.build_verdict(facts=facts(h1=harm))
    assert out["row"] == 3
    assert out["reason"] == "H1 pooled"
    assert "c6" in out
    assert sc.build_verdict(facts=facts(h2={"pass": False, "n": 5}))["row"] == 3
    assert sc.build_verdict(facts=facts(stopped="a stop"))["front_matter_verdict"] == "inconclusive"


def test_a_band_above_its_limit_fails_only_with_hb_lower_above_zero() -> None:
    band = {"pooled": PASS, "3MVA+": PASS, "1-3MVA": {"pass": False, "n": 10}}
    boundary = sc.build_verdict(
        facts=facts(h1=band, hb=reading(point=0.0, lower=-0.01, upper=0.02))
    )
    assert boundary["row"] != 3
    assert boundary["band_1_3_mva"] == "at the calibrated boundary"
    failing = sc.build_verdict(facts=facts(h1=band, hb=reading(point=0.02, lower=0.01, upper=0.03)))
    assert failing["row"] == 3
    assert failing["band_1_3_mva"] == "fail"


def test_a_band_above_its_limit_with_an_unreadable_hb_is_unread_row_2() -> None:
    nan = reading(point=float("nan"), lower=float("nan"), upper=float("nan"))
    band = {"pooled": PASS, "3MVA+": PASS, "1-3MVA": {"pass": False, "n": 10}}
    for hb in (nan, None):
        out = sc.build_verdict(facts=facts(h1=band, hb=hb))
        assert out["row"] == 2
        assert out["band_1_3_mva"] == "unread"
        assert "HB" in out["reason"]  # ty: ignore[unsupported-operator]
    # a band that passes does not need HB
    assert sc.build_verdict(facts=facts(hb=nan))["row"] != 2


def test_unread_c1_or_harm_checks_are_row_2_never_a_silent_row_6() -> None:
    nan = reading(point=float("nan"), lower=float("nan"), upper=float("nan"))
    out = sc.build_verdict(facts=facts(c={**facts().c, "C1": nan}))
    assert out["row"] == 2
    assert "C1" in out["reason"]  # ty: ignore[unsupported-operator]
    assert (
        sc.build_verdict(facts=facts(c={k: v for k, v in facts().c.items() if k != "C1"}))["row"]
        == 2
    )
    empty = {"pooled": {"pass": True, "n": 0}, "3MVA+": PASS, "1-3MVA": PASS}
    assert sc.build_verdict(facts=facts(h1=empty))["row"] == 2
    assert sc.build_verdict(facts=facts(h2=None))["row"] == 2
    assert sc.build_verdict(facts=facts(h1={"pooled": PASS}))["row"] == 2


def test_side_readings_never_change_a_row_and_a_short_count_voids_its_word() -> None:
    base = sc.build_verdict(facts=facts())
    other = sc.build_verdict(
        facts=facts(
            solar_bars=dict.fromkeys(core.SOLAR_BARS, False),
            r19=None,
            f1_big=0.0,
            c={**facts().c, "C5": reading(point=0.1, lower=0.05, upper=0.2), "C7": None},
        )
    )
    assert base["row"] == other["row"]
    assert base["verdict"] == other["verdict"]
    assert other["words"]["solar"] == "not shown"  # ty: ignore[not-subscriptable]
    assert other["words"]["bsp"] == "void"  # ty: ignore[not-subscriptable]
    assert other["words"]["neighbour"] == "void"  # ty: ignore[not-subscriptable]
    assert other["words"]["transfer"] == "worse than the rule"  # ty: ignore[not-subscriptable]
    short = sc.build_verdict(
        facts=facts(transfer_reachable=False, injection_reachable=False, n_bsp_reverse=0)
    )
    assert short["row"] == base["row"]
    assert short["words"] == {
        "solar": "void",
        "transfer": "void",
        "neighbour": "void",
        "bsp": "void",
    }


def test_step_matching_is_one_to_one_so_two_calls_at_one_edge_halve_the_precision() -> None:
    event = np.zeros((1, 48))
    event[0, 10] = 1.0
    size = event * 8.0
    valid = np.ones((1, 48), dtype=bool)
    p = np.zeros((1, 48))
    p[0, 9] = p[0, 10] = 0.9
    r = sc.step_f1(step_p=p, event=event, size=size, valid=valid)
    assert r["recall"] == 1.0
    assert r["precision"] == 0.5
    assert sc.precision_all(step_p=p[0], event=event[0]) == 0.5


def test_step_matching_does_not_cross_rows() -> None:
    called = np.zeros((2, 48), dtype=bool)
    true = np.zeros((2, 48), dtype=bool)
    called[0, 47] = True
    true[1, 0] = True
    call_ok, true_ok = sc.match_steps(called=called, true=true)
    assert not call_ok.any()
    assert not true_ok.any()

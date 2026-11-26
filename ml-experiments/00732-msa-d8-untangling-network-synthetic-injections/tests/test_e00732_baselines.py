# ruff: noqa: E501
"""Known-answer tests of B-PV0, B-PV1, B-PV2, B-T, and B-TN, through the real overlay and window builders."""

import dataclasses
from pathlib import Path
from typing import NamedTuple

import e00729_arms as arms729
import e00729_sim as sim
import e00732_arms as arms
import e00732_baselines as bl
import e00732_core as core
import e00732_neighbours as nb
import e00732_overlays as ov
import numpy as np
import pytest
from world732 import make_bank, make_neighbours


class World(NamedTuple):
    """The invented bank, the neighbours, a test overlay, its prepared segments, and the scales."""

    cal: sim.Calendar
    segs: list[sim.Segment]
    series: dict[str, np.ndarray]
    overlays: list[ov.TransferOverlay]
    prepared: list[arms.Prepared]
    scales: bl.StepScales
    steps: list[bl.StepSeries]


@pytest.fixture(scope="module")
def world() -> World:
    cal, segs = make_bank(draws_per_world=3)
    series = make_neighbours(cal=cal)
    overlays = ov.overlay_segments(segments=segs, kind="test", seed=None, neighbours=series)
    everyone = np.arange(len(cal.dates))
    stats = arms729.input_stats(cal=cal, train_days=everyone)
    medians = {
        n: nb.training_median(folded=nb.fold(signed=series[n], floor=0.0), train_rows=everyone)
        for n in nb.NEIGHBOURS
    }
    prepared = arms.prepare(overlays=overlays, holidays=cal.holidays, stats=stats, medians=medians)
    scales = bl.compute_scales(
        reading=cal.reading,
        bsp=cal.bsp,
        neighbours=series,
        dates=cal.dates,
        weekend=bl.weekend_flags(dates=cal.dates, holidays=cal.holidays),
        train_rows=np.ones(len(cal.dates), bool),
    )
    steps = [bl.series_of_overlay(overlay=o, holidays=cal.holidays) for o in overlays]
    return World(cal, segs, series, overlays, prepared, scales, steps)


def edge_rows(*, world: World, mirrored: bool, big: float = 10.0) -> list[tuple[int, int, int]]:
    """Return (row of all_picks, half-hour) of the start edges of mirrored (or unmirrored) transfers of at least `big` points."""
    picks = arms.all_picks(prepared=world.prepared)
    where = {pick: k for k, pick in enumerate(picks)}
    out = []
    for i, p in enumerate(world.prepared):
        for tr in p.transfers:
            day, hh = divmod(tr.start_half_hour, 48)
            if tr.mirrored == mirrored and abs(tr.size) >= big and (i, day) in where:
                out.append((where[i, day], hh, i))
    return out


# --------------------------------------------------------------------------- B-PV1 and B-PV2


def test_signed_reading_estimate_and_b_pv2_known_answers() -> None:
    r = np.array([5.0, 5.0, 0.05])
    p = np.array([0.9, 0.1, 0.9])
    x_hat = bl.signed_reading_estimate(p=p, r=r)
    assert x_hat[0] < 0 < x_hat[1]
    assert x_hat[2] == 0.0
    cosz = np.array([0.5, 0.5, 0.5])
    est = bl.solar_from_demand(demand=np.array([20.0] * 3), p=p, r=r, cosz=cosz)
    m = np.sqrt(25 - core.Q_HAT_POINTS**2)
    assert est[0] == pytest.approx(20.0 + m)
    assert est[1] == pytest.approx(20.0 - m)
    night = bl.solar_from_demand(
        demand=np.array([20.0]), p=np.array([0.1]), r=np.array([5.0]), cosz=np.array([-0.2])
    )
    assert night[0] == 0.0
    with pytest.raises(ValueError, match="three seeds"):
        bl.b_pv2(demand_by_seed=[r], p_by_seed=[p], r=r, cosz=cosz)
    with pytest.raises(ValueError, match="three seeds"):
        bl.b_pv1(demand_median=r, p_by_seed=[p], r=r, cosz=cosz)
    both = bl.b_pv2(demand_by_seed=[r * 4] * 3, p_by_seed=[p] * 3, r=r, cosz=cosz)
    np.testing.assert_allclose(both, bl.solar_from_demand(demand=r * 4, p=p, r=r, cosz=cosz))


def test_a_rule_recovers_an_injection_in_both_signs_only_with_the_right_sign() -> None:
    """B-PV2 recovers delta whether the plain member imports or exports; a head that reads the reading as import falls in reverse flow."""
    delta = np.array([5.0, 5.0])
    x = np.array([30.0, -12.0])  # one import row, one reverse-flow row
    r_plain = np.sqrt(x**2 + core.Q_HAT_POINTS**2)  # so that M is exactly |x|
    r_inj = np.sqrt((x - delta) ** 2 + core.Q_HAT_POINTS**2)
    demand = np.full(2, 40.0)
    cosz = np.full(2, 0.6)
    right = (x < 0).astype(float)
    pv = lambda p, r: bl.b_pv_member(  # noqa: E731
        outputs=bl.Outputs730(p=[p] * 3, demand=[demand] * 3, r=r, cosz=cosz)
    )
    np.testing.assert_allclose(pv(right, r_inj) - pv(right, r_plain), delta)
    blind = pv(np.zeros(2), r_inj) - pv(np.zeros(2), r_plain)
    assert blind[0] == pytest.approx(5.0)
    assert blind[1] == pytest.approx(-5.0)


def test_b_pv1_needs_the_climatology_and_b_pv2_the_demand_output() -> None:
    p, r, cosz = np.zeros(3), np.full(3, 4.0), np.full(3, 0.5)
    outputs = bl.Outputs730(p=[p] * 3, demand=None, r=r, cosz=cosz)
    with pytest.raises(ValueError, match="climatology"):
        bl.b_pv_member(outputs=outputs)
    out = bl.b_pv_member(outputs=outputs, demand_median=np.full(3, 10.0))
    assert out == pytest.approx(10.0 - np.sqrt(16 - core.Q_HAT_POINTS**2))


def test_infer_730_runs_a_stored_network_of_730_on_this_experiments_windows(
    world: World, tmp_path: Path
) -> None:
    """The inference reads a state file of 730's shape, builds windows without neighbours, and returns plain and injected outputs."""
    torch = pytest.importorskip("torch")
    import e00730_arms as arms730
    import e00730_nets as nets730

    spec730 = arms730.ArmSpec("SD-R", "d", demand=True, style="R")
    net = nets730.build_net(arch="unet", spec=spec730)
    state = tmp_path / "SD-R.pt"
    torch.save(net.state_dict(), state)
    index = ov.build_pair_index(segments=world.segs, sources=("S2",))
    out = bl.infer_730(
        state_path=state, demand=True, prepared=world.prepared, index=index, amount=20.0
    )
    n = len(arms.all_picks(prepared=world.prepared))
    plain, inj = out["plain"], out["inj"]
    assert isinstance(plain, dict)
    assert isinstance(inj, dict)
    assert plain["p"].shape == (n, 48)
    assert plain["demand_mu"].shape == (n, 48)
    assert inj["p"].shape == (n, 48)
    assert out["pair"].sum() > 0  # type: ignore[union-attr]
    # a window with a pair changes the injected member's outputs; a window without one does not
    paired = np.asarray(out["pair"]) > 0.5
    assert not np.allclose(plain["p"][paired], inj["p"][paired])
    np.testing.assert_allclose(plain["p"][~paired], inj["p"][~paired], atol=1e-6)


# --------------------------------------------------------------------------- B-PV0


def test_b_pv0_reads_the_sky_alone_and_fails_on_a_silent_target() -> None:
    rng = np.random.default_rng(0)
    cosz = np.clip(rng.uniform(-0.5, 1.0, 4000), -0.5, 1.0)
    clear = np.clip(cosz, 0, None) * 0.9
    csi = rng.uniform(0.0, 1.2, 4000)
    target = 30.0 * csi * clear * (cosz > 0)
    w = bl.fit_pv0(csi=csi, clear=clear, cosz=cosz, target=target)
    est = bl.b_pv0(weights=w, csi=csi, clear=clear, cosz=cosz)
    assert (est >= 0).all()
    assert (est[cosz <= core.NIGHT_COSZ] == 0).all()
    assert float(np.abs(est - target).mean()) < 0.1 * float(target.mean())
    # it reads no reading, so a window's reading cannot change it: it takes only the sky
    assert bl.PV0_NAMES
    assert bl.pv0_features(csi=csi, clear=clear, cosz=cosz).shape == (4000, 6)
    with pytest.raises(ValueError, match="solar output"):
        bl.fit_pv0(csi=csi, clear=clear, cosz=cosz, target=np.zeros(4000))


def test_b_pv0_fits_on_the_real_window_channels_and_follows_the_sky(world: World) -> None:
    picks = arms.all_picks(prepared=world.prepared)
    plain = np.stack(
        [
            arms.plain_window(p=world.prepared[i], day=d, spec=arms.eval_spec(), hidden=None)
            for i, d in picks
        ]
    )
    csi, clear, cosz = bl.sky_of_windows(x=plain)
    assert csi.shape == (len(picks), 48)
    assert (cosz > 0).any()
    solar = np.stack(
        [arms.target_window(p=world.prepared[i], day=d)[arms.T["solar"]] for i, d in picks]
    )
    w = bl.fit_pv0(csi=csi.ravel(), clear=clear.ravel(), cosz=cosz.ravel(), target=solar.ravel())
    est = bl.b_pv0(weights=w, csi=csi, clear=clear, cosz=cosz)
    assert est.shape == solar.shape
    assert (est[cosz <= 0] == 0).all()
    assert est.max() > 0, "B-PV0 predicts no solar output on a bank with solar worlds"


# --------------------------------------------------------------------------- the step statistics


def test_the_step_features_come_out_in_noise_units_with_the_daily_shape_removed(
    world: World,
) -> None:
    u = world.scales.units
    assert set(u) == {"D8", "D3", "D5", "D14"}
    assert all(v > 0 for v in u.values())
    shape = world.scales.shapes["D8"]
    assert shape.shape == (2, 48)
    # a series equal to its own daily shape has no step, so the ramps are not steps
    n = 6
    dates = world.cal.dates[:n]
    weekend = bl.weekend_flags(dates=dates, holidays=world.cal.holidays)
    values = shape[weekend.astype(int)]
    stat = nb.step_statistic(values=values, dates=dates, weekend=weekend, shape=shape)
    assert float(np.nanmax(np.abs(stat))) < 1e-9
    # a unit step of one noise unit gives a step statistic of 1 after scaling
    stepped = values.copy().reshape(-1)
    stepped[3 * 48 + 20 :] += u["D8"]
    stat = nb.step_statistic(
        values=stepped.reshape(n, 48), dates=dates, weekend=weekend, shape=shape
    )
    assert float(np.nanmax(stat / u["D8"])) == pytest.approx(1.0)


def test_weekend_flags_follow_the_calendar() -> None:
    dates = np.arange(np.datetime64("2022-03-04"), np.datetime64("2022-03-10"))  # Fri to Wed
    flags = bl.weekend_flags(dates=dates, holidays=np.array([np.datetime64("2022-03-07")]))
    assert flags.tolist() == [False, True, True, True, False, False]  # Mon is a holiday here


def test_step_features_at_a_mirrored_edge_carry_the_mirror_cue(world: World) -> None:
    """On the real overlay path, a mirrored transfer's start shows a large step, a negative aligned step, and an opposite-sign cue."""
    f = bl.step_feature_rows(
        series=world.steps, prepared=world.prepared, scales=world.scales, neighbours=True
    )
    picks = arms.all_picks(prepared=world.prepared)
    assert f.shape == (len(picks), 48, len(bl.BTN_FEATURES))
    assert np.isfinite(f).all()
    mirrored = edge_rows(world=world, mirrored=True)
    unmirrored = edge_rows(world=world, mirrored=False)
    assert mirrored, "the fixture must draw large mirrored transfers"
    assert unmirrored, "the fixture must draw large unmirrored transfers"
    names = {n: i for i, n in enumerate(bl.BTN_FEATURES)}
    at = lambda rows: np.array([f[k, hh] for k, hh, _ in rows])  # noqa: E731
    m, u = at(mirrored), at(unmirrored)
    typical = np.median(f[..., names["abs_step"]])
    assert m[:, names["abs_step"]].mean() > 5 * typical  # the edge is a step in noise units
    assert (
        m[:, names["largest_opposite_step"]].mean()
        > 3 * u[:, names["largest_opposite_step"]].mean()
    )
    assert m[:, names["largest_opposite_step"]].mean() > 2 * typical
    aligned = m[:, [names[f"aligned_{n}"] for n in nb.NEIGHBOURS]]
    assert aligned.min(axis=1).mean() < -1.0  # the partner's step has the opposite sign to D8's


def test_b_tn_with_the_neighbours_removed_equals_b_t_and_zero_neighbours_give_zero_cues(
    world: World,
) -> None:
    kwargs = {"series": world.steps, "prepared": world.prepared, "scales": world.scales}
    plain = bl.step_feature_rows(neighbours=False, **kwargs)
    full = bl.step_feature_rows(neighbours=True, **kwargs)
    np.testing.assert_allclose(full[..., : len(bl.BT_FEATURES)], plain)
    gone = [
        bl.StepSeries(
            reading=s.reading, bsp=s.bsp, cosz=s.cosz, weekend=s.weekend, dates=s.dates,
            neighbours={n: np.full_like(s.reading, np.nan) for n in nb.NEIGHBOURS},
        )
        for s in world.steps
    ]  # fmt: skip
    none = bl.step_feature_rows(
        series=gone, prepared=world.prepared, scales=world.scales, neighbours=True
    )
    assert not none[..., len(bl.BT_FEATURES) :].any()


def test_the_bsp_inputs_are_zero_where_the_meter_has_no_data(world: World) -> None:
    names = {n: i for i, n in enumerate(bl.BT_FEATURES)}
    silent = len(world.steps) // 2  # the first segments get no BSP meter
    steps = [
        dataclasses.replace(s, bsp=np.full_like(s.bsp, np.nan)) if k < silent else s
        for k, s in enumerate(world.steps)
    ]
    f = bl.step_feature_rows(
        series=steps, prepared=world.prepared, scales=world.scales, neighbours=False
    )
    picks = arms.all_picks(prepared=world.prepared)
    absent = np.array([world.prepared[i].source_index < silent for i, _ in picks])
    assert absent.any()
    assert (~absent).any()
    assert not f[absent][..., names["abs_bsp_step"]].any()
    np.testing.assert_allclose(  # no meter: the difference of the two steps is D8's step alone
        f[absent][..., names["abs_step_minus_bsp_step"]], f[absent][..., names["abs_step"]]
    )
    assert f[~absent][..., names["abs_bsp_step"]].any()


def test_the_rules_are_fitted_on_one_overlay_and_beat_the_prior_on_another(world: World) -> None:
    """B-T and B-TN learn the step from the validation overlay and predict a held-out overlay."""
    cal = world.cal
    everyone = np.arange(len(cal.dates))
    stats = arms729.input_stats(cal=cal, train_days=everyone)
    medians = {
        n: nb.training_median(
            folded=nb.fold(signed=world.series[n], floor=0.0), train_rows=everyone
        )
        for n in nb.NEIGHBOURS
    }

    def bank(kind: str) -> tuple[list[arms.Prepared], list[bl.StepSeries], np.ndarray]:
        overlays = ov.overlay_segments(
            segments=world.segs, kind=kind, seed=None, neighbours=world.series
        )
        prepared = arms.prepare(
            overlays=overlays, holidays=cal.holidays, stats=stats, medians=medians
        )
        steps = [bl.series_of_overlay(overlay=o, holidays=cal.holidays) for o in overlays]
        targets = np.stack([arms.target_window(p=p, day=int(d)) for p in prepared for d in p.days])
        return prepared, steps, targets

    fit_p, fit_s, fit_t = bank("validation")
    test_p, test_s, test_t = bank("test")
    scores = {}
    for with_neighbours in (False, True):
        kw = {"scales": world.scales, "neighbours": with_neighbours}
        weights = bl.fit_step_rule(
            features=bl.step_feature_rows(series=fit_s, prepared=fit_p, **kw),
            event=fit_t[:, arms.T["event"]],
            valid=fit_t[:, arms.T["valid"]] > 0.5,
        )
        pred = bl.predict_step_rule(
            features=bl.step_feature_rows(series=test_s, prepared=test_p, **kw), weights=weights
        )
        valid = test_t[:, arms.T["valid"]] > 0.5
        event = test_t[:, arms.T["event"]]
        scores[with_neighbours] = float(((pred - event)[valid] ** 2).mean())
    prior = float(event[valid].mean())
    assert scores[False] < prior * (1 - prior)
    assert scores[True] < prior * (1 - prior)


def test_the_rule_refuses_a_validation_overlay_with_no_step() -> None:
    f = np.random.default_rng(0).normal(size=(2, 48, 4))
    with pytest.raises(ValueError, match="with and without"):
        bl.fit_step_rule(features=f, event=np.zeros((2, 48)), valid=np.ones((2, 48), bool))


def test_logistic_regression_recovers_a_planted_rule() -> None:
    rng = np.random.default_rng(0)
    f = rng.normal(size=(4000, 5))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(2.0 * f[:, 0] - 1.0)))).astype(float)
    w = bl.fit_logistic(features=f, y=y)
    assert w[1] == pytest.approx(2.0, abs=0.3)
    assert w[0] == pytest.approx(-1.0, abs=0.3)
    assert abs(w[3]) < 0.3
    p = bl.predict_logistic(features=f, weights=w)
    assert float(np.mean((p - y) ** 2)) < float(np.mean((y.mean() - y) ** 2))

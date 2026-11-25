"""Tests of the networks, the losses, and a tiny fit. They need `torch` and run on the CPU."""

import pytest

torch = pytest.importorskip("torch")

import e00729_arms as arms  # noqa: E402
import e00729_core as core  # noqa: E402
import e00729_dist as dist  # noqa: E402
import e00729_nets as nets  # noqa: E402
import e00729_train as train  # noqa: E402
import numpy as np  # noqa: E402
from synthetic_world import make_calendar  # noqa: E402

import e00729_sim as sim  # noqa: E402  # isort: skip


@pytest.mark.parametrize("arch", core.ARCHITECTURES)
def test_every_architecture_keeps_the_time_axis_and_emits_the_head_channels(arch: str) -> None:
    spec = arms.ARM_SPECS["D"]
    c = arms.n_channels(spec=spec)
    n_out = nets.n_outputs(head=spec.head, aux=spec.aux)
    net = nets.build_net(arch=arch, c_in=c, steps=spec.steps, n_out=n_out)
    out = net(torch.zeros(3, c, spec.steps))
    assert out.shape == (3, n_out, spec.steps)
    assert nets.count_weights(net=net) > 0
    with pytest.raises(ValueError, match="unknown architecture"):
        nets.build_net(arch="mlp", c_in=c, steps=spec.steps, n_out=n_out)


def test_head_sizes() -> None:
    assert nets.n_outputs(head="lobe", aux="none") == 5
    assert nets.n_outputs(head="lobe", aux="solar") == 7
    assert nets.n_outputs(head="quantile", aux="none") == len(core.QUANTILE_LEVELS)


def test_lobes_respect_the_sign_and_the_spread_floor() -> None:
    raw = torch.randn(4, 5, 48) * 5
    q = nets.lobe_parameters(raw=raw)
    assert (q["mu_i"] >= 0).all()
    assert (q["mu_r"] <= 0).all()
    assert (q["s_i"] >= core.FLOOR_POINTS).all()
    assert (q["s_r"] >= core.FLOOR_POINTS).all()
    assert ((q["p"] >= 0) & (q["p"] <= 1)).all()


def test_mixture_density_integrates_to_one_and_matches_the_numpy_pdf() -> None:
    raw = torch.tensor([[[0.3], [1.0], [0.0], [-0.5], [0.4]]])
    grid = torch.linspace(-400, 400, 160_001)
    nll = torch.stack([nets.mixture_nll(raw=raw, x=g.view(1, 1)) for g in grid[::20]])
    step = float(grid[20] - grid[0])
    assert float(torch.exp(-nll).sum() * step) == pytest.approx(1.0, abs=2e-3)
    q = nets.lobe_parameters(raw=raw)
    x = 3.0
    p = float(q["p"])
    expected = (1 - p) * np.exp(-0.5 * ((x - float(q["mu_i"])) / float(q["s_i"])) ** 2) / (
        float(q["s_i"]) * np.sqrt(2 * np.pi)
    ) + p * np.exp(-0.5 * ((x - float(q["mu_r"])) / float(q["s_r"])) ** 2) / (
        float(q["s_r"]) * np.sqrt(2 * np.pi)
    )
    assert float(torch.exp(-nets.mixture_nll(raw=raw, x=torch.tensor([[x]])))) == pytest.approx(
        expected, rel=1e-4
    )


def test_quantiles_are_sorted_and_pinball_is_minimised_at_the_truth() -> None:
    q = nets.quantile_values(raw=torch.randn(2, 11, 48))
    assert (q[:, 1:] >= q[:, :-1]).all()
    x = torch.zeros(1, 4)
    near = torch.zeros(1, 11, 4)
    far = near + 3.0
    assert float(nets.pinball(q=near, x=x).sum()) < float(nets.pinball(q=far, x=x).sum())


def test_batch_loss_ignores_invalid_half_hours_and_uses_the_right_aux_target() -> None:
    raw = torch.randn(2, 7, 48)
    t = torch.zeros(2, 5, 48)
    t[:, 0] = 5.0
    t[:, 1] = 1.0
    base = nets.batch_loss(raw=raw, targets=t, head="lobe", aux="none")
    t_bad = t.clone()
    t_bad[:, 0, 10:20] = 1e6
    t_bad[:, 1, 10:20] = 0.0
    assert float(nets.batch_loss(raw=raw, targets=t_bad, head="lobe", aux="none")) == pytest.approx(
        float(nets.batch_loss(raw=raw[:, :, :], targets=t_bad, head="lobe", aux="none"))
    )
    assert np.isfinite(float(base))
    t_eng, t_sol = t.clone(), t.clone()
    t_eng[:, 2] = 30.0
    t_sol[:, 3] = 30.0
    on_eng = float(nets.batch_loss(raw=raw, targets=t_eng, head="lobe", aux="engine"))
    on_sol = float(nets.batch_loss(raw=raw, targets=t_sol, head="lobe", aux="solar"))
    assert on_eng == pytest.approx(
        on_sol
    )  # each reads its own row, so swapping rows and aux gives equal loss
    assert float(
        nets.batch_loss(raw=raw, targets=t_eng, head="lobe", aux="solar")
    ) != pytest.approx(on_eng)
    assert float(nets.batch_loss(raw=raw, targets=t, head="lobe", aux="engine")) != pytest.approx(
        on_eng
    )


def test_the_loss_matches_the_exact_crps_ordering_numerically() -> None:
    lobes = dist.Lobes(
        np.zeros(1), np.array([10.0]), np.array([1.0]), np.array([-10.0]), np.array([1.0])
    )
    assert (
        dist.mixture_crps(lobes=lobes, y=np.array([10.0]))[0]
        < dist.mixture_crps(lobes=lobes, y=np.array([0.0]))[0]
    )


def tiny_banks() -> list[arms.PreparedSegment]:
    cal = make_calendar()
    use = np.array([str(d)[:7] == "2022-03" for d in cal.dates])
    segs = [
        sim.simulate_segment(
            cal=cal, month="2022-03", use=use, world=w, rng=np.random.default_rng(i)
        )
        for i, w in enumerate(["both", "none", "engine"])
    ]
    stats = arms.InputStats(10.0, 3.0, 5.0, 2.0)
    return arms.prepare_segments(
        segments=segs, spec=arms.ARM_SPECS["A"], holidays=cal.holidays, stats=stats
    )


def test_a_tiny_cpu_fit_lowers_the_loss_and_saves_the_best_weights(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 64)
    monkeypatch.setattr(core, "BATCH_SIZE", 32)
    prepared = tiny_banks()
    net, result = train.fit(
        spec=arms.ARM_SPECS["A"],
        arch="dae",
        seed=0,
        train=prepared,
        val=prepared,
        device="cpu",
        max_epochs=4,
        out_path=tmp_path / "w.pt",
    )
    assert 1 <= result.epochs <= 4
    assert result.best_val_crps == min(result.history)
    assert result.history[result.best_epoch - 1] == result.best_val_crps
    assert (tmp_path / "w.pt").exists()
    flat = train.predictions(net=net, prepared=prepared, spec=arms.ARM_SPECS["A"], device="cpu")
    assert all(np.isfinite(v).all() for v in flat.values())
    again = train.fit(
        spec=arms.ARM_SPECS["A"],
        arch="dae",
        seed=0,
        train=prepared,
        val=prepared,
        device="cpu",
        max_epochs=2,
    )[1]
    assert again.history[0] == pytest.approx(result.history[0], rel=1e-4)


def test_timing_projection_uses_the_slowest_architecture_for_the_ladder() -> None:
    per = {"dae": 10.0, "unet": 20.0, "bilstm": 30.0}
    hours = train.projected_hours(seconds_per_epoch=per, epochs=100)
    assert hours == pytest.approx((6 * 60 + 32 * 30) * 100 / 3600)
    rep = train.timing_gate_report(seconds_per_epoch=per)
    assert rep["within_budget"] == (rep["hours_at_cap"] <= core.GPU_BUDGET_HOURS)
    slow = train.timing_gate_report(
        seconds_per_epoch={"dae": 100.0, "unet": 100.0, "bilstm": 100.0}
    )
    assert not slow["within_budget"]
    quick = train.timing_gate_report(seconds_per_epoch={"dae": 1.0, "unet": 1.0, "bilstm": 1.0})
    assert quick["within_budget"]


def test_the_job_lists_total_50_fits() -> None:
    assert len(train.screen_jobs()) == 18
    ladder = train.ladder_jobs(arch="unet")
    assert len(ladder) == 32
    assert train.all_jobs_count(arch="unet") == core.N_FITS == 50
    assert len({*train.screen_jobs(), *ladder}) == len(train.screen_jobs()) + 32 - len(
        set(train.screen_jobs()) & set(ladder)
    )


def test_the_sign_head_has_one_output_and_is_trained_on_the_event_x_below_zero() -> None:
    spec = arms.ARM_SPECS["Dsign"]
    assert spec.head == "sign"
    assert nets.n_outputs(head=spec.head, aux=spec.aux) == nets.SIGN_OUTPUTS == 1
    x = torch.tensor([[-5.0, 5.0, 0.0]])
    # a large positive logit means "X < 0": it is right for the first half-hour only
    right = torch.tensor([[[8.0, -8.0, -8.0]]])
    assert float(nets.sign_bce(raw=right, x=x).max()) < 1e-3  # X = 0 is not below zero
    wrong = nets.sign_bce(raw=-right, x=x)
    assert float(wrong.min()) > 7.0


def test_the_sign_head_loss_ignores_invalid_half_hours() -> None:
    raw = torch.randn(2, 1, 48)
    t = torch.zeros(2, 5, 48)
    t[:, 0] = -3.0
    t[:, 1] = 1.0
    t_bad = t.clone()
    t_bad[:, 0, 5:9] = 1e6
    t_bad[:, 1, 5:9] = 0.0
    t_ref = t.clone()
    t_ref[:, 1, 5:9] = 0.0
    assert float(nets.batch_loss(raw=raw, targets=t_bad, head="sign", aux="none")) == pytest.approx(
        float(nets.batch_loss(raw=raw, targets=t_ref, head="sign", aux="none"))
    )


def test_the_sign_lobes_sit_at_plus_and_minus_the_reading_without_the_reactive_floor() -> None:
    reading = torch.tensor([[0.2, core.QHAT_POINTS, 5.0, 40.0]])
    raw = torch.tensor([[[0.0, 1.0, -1.0, 2.0]]])
    q = nets.sign_lobes(raw=raw, reading=reading)
    magnitude = np.sqrt(np.maximum(reading.numpy()[0] ** 2 - core.QHAT_POINTS**2, 0.0))
    assert np.allclose(q["mu_i"].numpy()[0], magnitude, atol=1e-6)
    assert np.allclose(q["mu_r"].numpy()[0], -magnitude, atol=1e-6)
    assert q["mu_i"][0, 0] == 0.0
    assert q["mu_i"][0, 2] == pytest.approx(float(np.sqrt(25.0 - core.QHAT_POINTS**2)))
    assert (q["s_i"] == core.FLOOR_POINTS).all()
    assert (q["s_r"] == core.FLOOR_POINTS).all()
    assert np.allclose(q["p"].numpy()[0], 1 / (1 + np.exp(-raw.numpy()[0, 0])), atol=1e-6)
    # the weight of the negative lobe is P: a high logit moves the CRPS of a negative truth down
    lobes = dist.Lobes(*(q[k].numpy().reshape(-1) for k in ("p", "mu_i", "s_i", "mu_r", "s_r")))
    y = np.full(4, -5.0)
    low = dist.Lobes(
        *(
            np.full(4, v) if k == "p" else getattr(lobes, k)
            for k, v in (("p", 0.01), ("mu_i", 0), ("s_i", 0), ("mu_r", 0), ("s_r", 0))
        )
    )
    high = dist.Lobes(
        *(
            np.full(4, v) if k == "p" else getattr(lobes, k)
            for k, v in (("p", 0.99), ("mu_i", 0), ("s_i", 0), ("mu_r", 0), ("s_r", 0))
        )
    )
    assert dist.mixture_crps(lobes=high, y=y)[2] < dist.mixture_crps(lobes=low, y=y)[2]


def test_the_sign_arm_decodes_to_the_two_lobe_format_and_fits_on_cpu(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 64)
    monkeypatch.setattr(core, "BATCH_SIZE", 32)
    spec = arms.ARM_SPECS["Dsign"]
    cal = make_calendar()
    use = np.array([str(d)[:7] == "2022-03" for d in cal.dates])
    segs = [
        sim.simulate_segment(
            cal=cal, month="2022-03", use=use, world=w, rng=np.random.default_rng(i)
        )
        for i, w in enumerate(["both", "none", "engine"])
    ]
    prepared = arms.prepare_segments(
        segments=segs, spec=spec, holidays=cal.holidays, stats=arms.InputStats(10.0, 3.0, 5.0, 2.0)
    )
    net, result = train.fit(
        spec=spec, arch="dae", seed=0, train=prepared, val=prepared, device="cpu", max_epochs=2,
    )  # fmt: skip
    assert result.n_weights > 0
    flat = train.predictions(net=net, prepared=prepared, spec=spec, device="cpu")
    assert set(flat) == {"p", "mu_i", "s_i", "mu_r", "s_r"}
    assert (flat["mu_r"] <= 0).all()
    assert (flat["mu_i"] >= 0).all()
    assert np.allclose(flat["mu_i"], -flat["mu_r"])
    assert ((flat["p"] > 0) & (flat["p"] < 1)).all()


def sign_world() -> tuple[list[sim.Segment], list[arms.PreparedSegment], sim.Calendar]:
    cal = make_calendar()
    use = np.array([str(d)[:7] == "2022-03" for d in cal.dates])
    segs = [
        sim.simulate_segment(
            cal=cal, month="2022-03", use=use, world=w, rng=np.random.default_rng(i)
        )
        for i, w in enumerate(["both", "engine", "solar"])
    ]
    prepared = arms.prepare_segments(
        segments=segs, spec=arms.ARM_SPECS["Dsign"], holidays=cal.holidays,
        stats=arms.InputStats(10.0, 3.0, 5.0, 2.0),
    )  # fmt: skip
    return segs, prepared, cal


def expected_magnitudes(
    *, segs: list[sim.Segment], prepared: list[arms.PreparedSegment]
) -> np.ndarray:
    """Return sqrt(max(r^2 - QHAT^2, 0)) of the valid half-hours, in the order of `Rows`."""
    out = []
    for pick_seg, day in arms.all_picks(prepared=prepared):
        seg = segs[prepared[pick_seg].source_index]
        keep = seg.valid[day] & np.isfinite(seg.x[day])
        out.append(
            np.sqrt(np.maximum(seg.r[day][keep].astype(np.float64) ** 2 - core.QHAT_POINTS**2, 0.0))
        )
    return np.concatenate(out)


def test_the_stored_sign_lobes_sit_at_the_folded_reading_less_the_reactive_floor() -> None:
    segs, prepared, _ = sign_world()
    spec = arms.ARM_SPECS["Dsign"]
    net = nets.build_net(arch="dae", c_in=arms.n_channels(spec=spec), steps=spec.steps, n_out=1)
    flat = train.predictions(net=net, prepared=prepared, spec=spec, device="cpu")
    want = expected_magnitudes(segs=segs, prepared=prepared)
    assert np.allclose(flat["mu_i"], want, atol=1e-3)
    assert np.allclose(flat["mu_r"], -want, atol=1e-3)
    truth = np.concatenate([
        s.x[d][s.valid[d] & np.isfinite(s.x[d])] for s, d in
        ((segs[prepared[i].source_index], day) for i, day in arms.all_picks(prepared=prepared))
    ])  # fmt: skip
    assert not np.allclose(flat["mu_i"], np.abs(truth), atol=1e-2)


def test_the_validation_score_places_the_sign_lobes_at_the_reading(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 32)
    monkeypatch.setattr(core, "BATCH_SIZE", 16)
    segs, prepared, _ = sign_world()
    spec = arms.ARM_SPECS["Dsign"]
    seen: list[np.ndarray] = []
    original = train.decode

    def spy(*, raw: np.ndarray, spec: arms.ArmSpec, reading: np.ndarray) -> dict[str, np.ndarray]:
        seen.append(reading.copy())
        return original(raw=raw, spec=spec, reading=reading)

    monkeypatch.setattr(train, "decode", spy)
    train.fit(
        spec=spec, arch="dae", seed=0, train=prepared, val=prepared, device="cpu", max_epochs=1
    )
    picks = arms.all_picks(prepared=prepared)
    expected = np.stack([
        np.nan_to_num(segs[prepared[i].source_index].r[day]) for i, day in picks
    ]).astype(np.float32)  # fmt: skip
    assert len(seen) == 1
    assert np.allclose(seen[0], expected)
    assert not np.allclose(seen[0], np.abs(np.stack([
        np.nan_to_num(segs[prepared[i].source_index].x[day]) for i, day in picks
    ])), atol=1e-4)  # fmt: skip


def test_the_training_batches_carry_the_reading_in_the_fifth_target_row() -> None:
    segs, prepared, _ = sign_world()
    spec = arms.ARM_SPECS["Dsign"]
    picks = arms.all_picks(prepared=prepared)[:6]
    _, t = arms.make_batch(
        prepared=prepared, picks=picks, spec=spec, rng=np.random.default_rng(0), training=True
    )
    assert t.shape[1] == 5
    for k, (i, day) in enumerate(picks):
        assert np.allclose(t[k, 4], np.nan_to_num(segs[prepared[i].source_index].r[day]), atol=1e-5)

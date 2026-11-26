"""Tests of the heads, the loss terms, the stop-gradient, and a tiny fit on the CPU (need torch)."""

import pytest

torch = pytest.importorskip("torch")

import e00729_arms as arms729  # noqa: E402
import e00729_nets as nets729  # noqa: E402
import e00732_arms as arms  # noqa: E402
import e00732_core as core  # noqa: E402
import e00732_losses as losses  # noqa: E402
import e00732_nets as nets  # noqa: E402
import e00732_overlays as ov  # noqa: E402
import e00732_train as train  # noqa: E402
import numpy as np  # noqa: E402
from world732 import make_bank  # noqa: E402


@pytest.fixture(scope="module")
def world():
    cal, segs = make_bank(draws_per_world=3)
    stats = arms729.input_stats(cal=cal, train_days=np.arange(len(cal.dates)))
    overlays = ov.overlay_segments(segments=segs, kind="train", seed=0)
    prepared = arms.prepare(overlays=overlays, holidays=cal.holidays, stats=stats)
    plain = arms.prepare_plain(segments=segs, holidays=cal.holidays, stats=stats)
    return cal, segs, prepared, plain, ov.build_pair_index(segments=segs)


@pytest.mark.parametrize("arch", ["unet", "bilstm"])
def test_the_sign_channel_is_experiment_729s_backbone_output(arch) -> None:
    torch.manual_seed(0)
    net = nets.Net(arch=arch, c_in=arms.N_CHANNELS, heads=True)
    x = torch.randn(3, arms.N_CHANNELS, 96)
    out = net(x)
    assert out.shape == (3, 4, 96)
    torch.testing.assert_close(out[:, :1], net.backbone(x), rtol=1e-5, atol=1e-5)
    plus = nets.Net(arch=arch, c_in=arms.N_CHANNELS, heads=False)
    assert plus(x).shape == (3, 1, 96) and plus.extra is None


def test_unet_weight_count_is_the_plans() -> None:
    n = nets.count_weights(net=nets.Net(arch="unet", c_in=arms.N_CHANNELS, heads=True))
    assert 1.0e6 < n < 1.2e6
    assert nets.count_weights(net=nets.Net(arch="unet", c_in=arms.N_CHANNELS, heads=True)) - nets.count_weights(
        net=nets.Net(arch="unet", c_in=arms.N_CHANNELS, heads=False)
    ) == 3 * (32 + 1)
    assert 0.4e6 < nets.count_weights(net=nets.Net(arch="bilstm", c_in=arms.N_CHANNELS, heads=True)) < 0.7e6


@pytest.mark.parametrize("arch", ["unet", "bilstm"])
def test_changing_the_extra_heads_does_not_change_the_sign_logits(arch) -> None:
    torch.manual_seed(1)
    net = nets.Net(arch=arch, c_in=arms.N_CHANNELS, heads=True)
    x = torch.randn(2, arms.N_CHANNELS, 96)
    before = net(x)[:, 0].detach().clone()
    with torch.no_grad():
        for p in net.extra.parameters():
            p.add_(5.0 * torch.randn_like(p))
    after = net(x)
    torch.testing.assert_close(after[:, 0], before)
    assert not torch.allclose(after[:, 1:], net(x)[:, 1:] * 0 + before[:, None].expand(-1, 3, -1))


def fake_batch(n: int = 4):
    t = np.zeros((n, 8, 48), np.float32)
    t[:, arms.T["valid"]] = 1.0
    keeps = np.ones((n, 7, 48), np.float32)
    return t, keeps


def test_stop_gradient_blocks_every_extra_loss_from_the_backbone_and_only_those() -> None:
    torch.manual_seed(2)
    t, keeps = fake_batch()
    tt = torch.from_numpy(t)
    kk = torch.from_numpy(keeps)
    delta = torch.ones(4, 48)
    pair = torch.ones(4)
    x = torch.randn(4, arms.N_CHANNELS, 96)
    for sg, expect_backbone_grad in ((True, False), (False, True)):
        net = nets.Net(arch="unet", c_in=arms.N_CHANNELS, heads=True, stop_gradient=sg)
        raw = net(torch.cat([x, x, x]))[:, :, 24:72]
        raw_w, raw_i, raw_c = torch.split(raw, 4)
        values = losses.terms(raw_w=raw_w, raw_i=raw_i, raw_c=raw_c, t=tt, ti=tt, keeps=kk, delta=delta, pair=pair)
        extras = sum(values[k] for k in ("solar", "step", "size", "injection", "night", "monotone"))
        net.zero_grad()
        extras.backward(retain_graph=True)
        grads = [p.grad for p in net.backbone.parameters() if p.grad is not None]
        has = any(float(g.abs().sum()) > 0 for g in grads)
        assert has == expect_backbone_grad
        assert net.extra.weight.grad is not None and float(net.extra.weight.grad.abs().sum()) > 0
        net.zero_grad()
        values["sign"].backward()
        assert any(float(p.grad.abs().sum()) > 0 for p in net.backbone.parameters() if p.grad is not None)
        assert net.extra.weight.grad is None or float(net.extra.weight.grad.abs().sum()) == 0


def raw_with(pv: float, n: int = 2, step: float = 0.0, size: float = 0.0, sign: float = 0.0) -> torch.Tensor:
    """Return raw outputs whose solar output is `pv` points: raw = softplus^-1(pv / scale)."""
    raw_pv = float(np.log(np.expm1(pv / core.SOLAR_SCALE))) if pv > 0 else -50.0
    raw = torch.zeros(n, 4, 48)
    raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3] = sign, raw_pv, step, size / core.SIZE_SCALE
    return raw


def test_loss_terms_known_answers() -> None:
    t, keeps = fake_batch(2)
    t[:, arms.T["solar"]] = 4.0
    t[:, arms.T["event"]] = 0.0
    tt, kk = torch.from_numpy(t), torch.from_numpy(keeps)
    delta = torch.full((2, 48), 3.0)
    pair = torch.ones(2)
    raw_w = raw_with(2.0)  # solar error 2 -> Huber(2) = 1.5
    raw_i = raw_with(5.0)  # injected solar target is 4 + 0 (ti = t here): error 1 -> Huber(1) = 0.5
    raw_c = raw_with(1.0)  # monotone: relu(2 - 1) = 1
    v = losses.terms(raw_w=raw_w, raw_i=raw_i, raw_c=raw_c, t=tt, ti=tt, keeps=kk, delta=delta, pair=pair)
    assert float(v["sign"]) == pytest.approx(np.log(2.0), abs=1e-6)
    assert float(v["solar"]) == pytest.approx((1.5 + 0.5) / 2, abs=1e-5)
    assert float(v["step"]) == pytest.approx(np.log(2.0), abs=1e-6)
    assert float(v["injection"]) == pytest.approx(float(losses.huber(error=torch.tensor(3.0 - 3.0))), abs=1e-5)  # (5 - 2) - 3 = 0
    assert float(v["night"]) == pytest.approx((2.0 + 5.0) / 2, abs=1e-5)
    assert float(v["monotone"]) == pytest.approx(1.0, abs=1e-5)
    raw_i2 = raw_with(6.5)  # (6.5 - 2) - 3 = 1.5 -> Huber = 1.0
    v2 = losses.terms(raw_w=raw_w, raw_i=raw_i2, raw_c=raw_c, t=tt, ti=tt, keeps=kk, delta=delta, pair=pair)
    assert float(v2["injection"]) == pytest.approx(1.0, abs=1e-5)
    # a large injection error is linear in the Huber loss (error 10 -> 9.5), not quadratic
    v3 = losses.terms(raw_w=raw_w, raw_i=raw_with(15.0), raw_c=raw_c, t=tt, ti=tt, keeps=kk, delta=delta, pair=pair)
    assert float(v3["injection"]) == pytest.approx(9.5, abs=1e-4)


def test_unpaired_windows_add_nothing_to_the_injected_terms() -> None:
    t, keeps = fake_batch(2)
    tt, kk = torch.from_numpy(t), torch.from_numpy(keeps)
    delta = torch.full((2, 48), 3.0)
    raw_w, raw_i = raw_with(2.0), raw_with(50.0)
    none = losses.terms(raw_w=raw_w, raw_i=raw_i, raw_c=None, t=tt, ti=tt, keeps=kk, delta=delta, pair=torch.zeros(2))
    assert float(none["injection"]) == 0.0
    only_plain = losses.terms(raw_w=raw_w, raw_i=None, raw_c=None, t=tt, ti=None, keeps=kk, delta=delta, pair=torch.zeros(2))
    assert float(none["solar"]) == pytest.approx(float(only_plain["solar"]), abs=1e-6)
    assert float(none["night"]) == pytest.approx(float(only_plain["night"]), abs=1e-6)


def test_size_term_reads_only_edge_half_hours_and_the_sign_term_drops_hidden_ones() -> None:
    t, keeps = fake_batch(1)
    t[0, arms.T["x"], :10] = -1.0
    keeps[0, arms.K["size"]] = 0.0
    keeps[0, arms.K["size"], 7] = 1.0
    t[0, arms.T["size"], 7] = 20.0
    tt, kk = torch.from_numpy(t), torch.from_numpy(keeps)
    v = losses.terms(raw_w=raw_with(0.0, n=1, size=5.0), raw_i=None, raw_c=None, t=tt, ti=None, keeps=kk, delta=torch.zeros(1, 48), pair=torch.ones(1))
    assert float(v["size"]) == pytest.approx(14.5, abs=1e-4)  # Huber(5 - 20)
    keeps2 = keeps.copy()
    keeps2[0, arms.K["sign"]] = 0.0
    keeps2[0, arms.K["sign"], 20:] = 1.0  # only import half-hours kept: the reverse-flow ones are hidden
    big = raw_with(0.0, n=1, sign=-5.0)  # confident "import" is right on the kept half-hours
    v2 = losses.terms(raw_w=big, raw_i=None, raw_c=None, t=tt, ti=None, keeps=torch.from_numpy(keeps2), delta=torch.zeros(1, 48), pair=torch.ones(1))
    assert float(v2["sign"]) < 0.01


def test_total_uses_only_positive_weights_and_s_plus_has_only_the_sign() -> None:
    values = {"sign": torch.tensor(2.0), "solar": torch.tensor(10.0), "injection": torch.tensor(7.0)}
    w = core.ARMS["U-NC"].weights
    assert float(losses.total(values=values, weights={**w, "injection": 0.0})) == pytest.approx(2.0 + 0.02 * 10.0)
    assert float(losses.total(values={"sign": torch.tensor(3.0)}, weights=core.ARMS["S+"].weights)) == 3.0
    with pytest.raises(ValueError, match="no loss term"):
        losses.total(values={"solar": torch.tensor(1.0)}, weights=core.ARMS["S+"].weights)


def test_predict_rows_match_rows_from_segments_and_members_are_built(world) -> None:
    import e00729_score as score729

    cal, segs, _, plain, index = world
    net = nets.Net(arch="unet", c_in=arms.N_CHANNELS, heads=True)
    pred = train.predict(net=net, prepared=plain, device="cpu", index=index, amount=20.0, eps=0.2)
    rows = score729.rows_from_segments(segments=segs, holidays=cal.holidays)
    flat = train.valid_flat(values=pred.plain["p"], targets=pred.t)
    assert len(flat) == rows.n
    np.testing.assert_allclose(train.valid_flat(values=pred.t[:, arms.T["x"]], targets=pred.t), rows.x, atol=1e-5)
    assert pred.inj is not None and pred.raised is not None and pred.pair.sum() > 0
    assert not np.allclose(pred.inj["pv"][pred.pair == 1], pred.plain["pv"][pred.pair == 1])  # the injected window is a different input


def test_a_tiny_fit_runs_records_every_epoch_and_is_repeatable(world, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 48)
    monkeypatch.setattr(core, "BATCH_SIZE", 16)
    _, _, prepared, plain, index = world
    spec = core.ARMS["U*"]
    kwargs = {"spec": spec, "seed": 0, "train": prepared, "index": index, "val_plain": plain, "val_transfer": prepared, "device": "cpu", "max_epochs": 2}
    net, rec = train.fit(**kwargs, out_path=tmp_path / "n.pt")
    _, rec2 = train.fit(**kwargs)
    assert rec.epochs == 2 and len(rec.history) == len(rec.solar_history) == len(rec.step_history) == 2
    assert rec.history == rec2.history and rec.solar_history == rec2.solar_history
    for terms_by_epoch in (rec.train_terms, rec.val_terms):
        assert {"total", "sign", "solar", "step", "size", "injection", "night", "monotone"} <= set(terms_by_epoch)
        assert all(len(v) == 2 and np.isfinite(v).all() for v in terms_by_epoch.values())
    assert rec.train_terms == rec2.train_terms and rec.val_terms == rec2.val_terms
    assert (tmp_path / "n.pt").exists() and 0.0 <= rec.unpaired_share <= 1.0
    state = torch.load(tmp_path / "n.pt")
    assert set(state) == set(net.state_dict())
    _, rec3 = train.fit(**{**kwargs, "seed": 1})
    assert rec3.history != rec.history


def test_a_tiny_fit_of_every_arm_runs(world, monkeypatch) -> None:
    monkeypatch.setattr(core, "EPOCH_WINDOWS", 32)
    monkeypatch.setattr(core, "BATCH_SIZE", 16)
    _, _, prepared, plain, index = world
    for name, spec in core.ARMS.items():
        _, rec = train.fit(spec=spec, seed=0, train=prepared, index=index, val_plain=plain, val_transfer=prepared, device="cpu", max_epochs=1)
        assert rec.arm == name and np.isfinite(rec.best_val_brier)
        assert set(rec.train_terms) == set(rec.val_terms) and "total" in rec.train_terms
        assert ("solar" in rec.train_terms) == spec.heads  # a sign-only arm has the sign term alone
        assert np.isfinite(rec.solar_history[0]) == spec.heads


def test_the_unet_of_a_plus_arm_is_experiment_729s(world) -> None:
    net = nets.build_net(spec=core.ARMS["S+"], c_in=arms.N_CHANNELS)
    ref = nets729.UNet1d(arms.N_CHANNELS, 1)
    assert nets.count_weights(net=net) == nets729.count_weights(net=ref)

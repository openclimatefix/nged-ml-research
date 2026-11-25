"""Tests that need torch (skipped in CI): shapes, loss masks, and the sign logit's independence."""

import pytest

torch = pytest.importorskip("torch")

import e00730_arms as arms  # noqa: E402
import e00730_core as core  # noqa: E402
import e00730_nets as nets  # noqa: E402
import numpy as np  # noqa: E402


@pytest.mark.parametrize("arch", core.ARCHITECTURES)
def test_every_architecture_maps_a_window_to_the_outputs(arch: str) -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    net = nets.build_net(arch=arch, spec=spec)
    x = torch.randn(2, arms.n_channels(spec=spec), 96)
    assert net(x).shape == (2, 3, 96)
    assert nets.count_weights(net=net) > 0


def test_transformer_shape_follows_the_plan() -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    net = nets.build_net(arch="transformer", spec=spec)
    assert core.TRANSFORMER_LAYERS == 3
    assert core.TRANSFORMER_WIDTH == 128
    assert core.TRANSFORMER_HEADS == 4
    assert core.TRANSFORMER_FF == 512
    assert len([m for m in net.backbone.modules() if isinstance(m, nets.Block)]) == 3


@pytest.mark.parametrize("arch", core.ARCHITECTURES)
def test_the_sign_logit_does_not_depend_on_the_demand_rows(arch: str) -> None:
    spec = arms.arm_specs(style="K")["SD*"]
    net = nets.build_net(arch=arch, spec=spec).eval()
    x = torch.randn(1, arms.n_channels(spec=spec), 96)
    before = net(x)[:, 0].detach().clone()
    with torch.no_grad():
        net.head.weight[1:] += 10.0
        net.head.bias[1:] += 10.0
    after = net(x)[:, 0]
    assert torch.allclose(before, after)


def test_masked_loss_keeps_demand_and_drops_sign() -> None:
    raw = torch.zeros(1, 3, 48)
    targets = torch.zeros(1, 6, 48)
    targets[0, 0] = -1.0
    targets[0, 1] = 1.0
    keeps = torch.zeros(1, 2, 48)
    keeps[0, 1] = 1.0
    sign_dropped = nets.batch_loss(raw=raw, targets=targets, keeps=keeps, lam=0.1)
    keeps[0, 0] = 1.0
    sign_kept = nets.batch_loss(raw=raw, targets=targets, keeps=keeps, lam=0.1)
    assert sign_kept > sign_dropped
    only_sign = nets.batch_loss(raw=raw[:, :1], targets=targets, keeps=keeps, lam=0.1)
    assert only_sign == pytest.approx(float(np.log(2.0)), abs=1e-4)


def test_decode_gives_a_probability_and_a_positive_spread() -> None:
    out = nets.decode(raw=torch.randn(2, 3, 48))
    assert ((out["p"] >= 0) & (out["p"] <= 1)).all()
    assert (out["demand_s"] > 0).all()

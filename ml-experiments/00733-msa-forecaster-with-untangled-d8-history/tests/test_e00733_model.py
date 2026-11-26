# ruff: noqa: E501, PT018, C408
"""Tests of `e00733_model`: the zero-initialised path, the read-out features, and the sign head's start."""

import e00733_model as model
import pytest

torch = pytest.importorskip("torch")
import e00713_model as mdl  # noqa: E402  # ty: ignore[unresolved-import]
from torch_inputs import make_inputs  # noqa: E402


def _m_star(seed: int = 0) -> mdl.MsaNet:
    torch.manual_seed(seed)
    net = mdl.MsaNet()
    with torch.no_grad():  # the head's output layer starts at zero; a trained one does not
        for p in net.head.parameters():
            p.normal_(std=0.1)
    return net.eval()


def _untangled(*, state: dict, **kwargs: object) -> model.UntangledNet:
    net = model.UntangledNet(seed=1, **kwargs)  # ty: ignore[invalid-argument-type]
    net.load_m_star(state=state)
    return net.eval()


def test_absent_channels_give_m_stars_output_bit_for_bit() -> None:
    """A network that loads M*'s state and reads every channel as absent equals M* exactly (verdict row 1 check)."""
    m = _m_star()
    inp = make_inputs(channels=False)
    net = _untangled(state=m.state_dict())
    with torch.no_grad():
        want = m(inp).out
        got = net(inp).out
    assert torch.equal(want, got)


def test_present_channels_change_nothing_while_the_gain_and_head_map_are_zero() -> None:
    """The gain and the head map start at zero, so even present channels leave the output unchanged."""
    m = _m_star()
    inp = make_inputs(channels=True)
    assert bool(inp.hist_u_ok.any())
    net = _untangled(state=m.state_dict())
    with torch.no_grad():
        assert torch.equal(m(inp).out, net(inp).out)


def test_a_trained_gain_makes_the_channels_matter() -> None:
    """With a nonzero gain, present channels move the output and absent ones do not."""
    m = _m_star()
    inp = make_inputs(channels=True)
    net = _untangled(state=m.state_dict())
    with torch.no_grad():
        net.u_gamma.fill_(0.5)
        net.u_head_in.weight.normal_(std=0.1)
        with_ch = net(inp).out
        without = net(inp.absent()).out
        base = m(inp).out
    assert not torch.equal(with_ch, base)
    assert torch.equal(without, base)


def test_missing_keys_are_only_the_new_ones() -> None:
    """Loading a state with a key of the trunk missing fails the assertion."""
    m = _m_star()
    state = dict(m.state_dict())
    state.pop(next(iter(state)))
    net = model.UntangledNet(seed=1)
    with pytest.raises(AssertionError):
        net.load_m_star(state=state)


def test_calibrate_scale_sets_r_and_refuses_an_empty_arm() -> None:
    """`r` becomes the root mean square of the path over flagged tokens; an arm with no flagged token raises."""
    net = model.UntangledNet(seed=2)
    inp = make_inputs(channels=True)
    rms = net.calibrate_scale(batches=[inp])
    assert rms > 0 and float(net.u_r) == pytest.approx(rms)
    with pytest.raises(ValueError, match="no history token"):
        net.calibrate_scale(batches=[make_inputs(channels=False)])


def test_parameter_counts_match_the_plan() -> None:
    """The history path has 4*32+32 + 32*64 + 64 weights, the head map 4*64, and the sign head 80*32+32 + 32+1."""
    net = model.UntangledNet(seed=0, sign_head=False)
    path = (4 * 32 + 32) + 32 * 64 + 64 + 4 * 64
    assert model.new_parameter_count(model=net) == path
    both = model.UntangledNet(seed=0, sign_head=True)
    assert model.new_parameter_count(model=both) == path + (80 * 32 + 32) + (32 + 1)


def test_sign_head_starts_as_the_persistence_of_the_history() -> None:
    """Where the history covers a column fully, the untrained head's logit is the logit of the weighted mean of P(X<0)."""
    inp = make_inputs(b=2, channels=True)
    ok = torch.ones_like(inp.hist_u_ok)
    u = inp.hist_u.clone()
    u[..., 0] = 0.3
    inp = inp.with_channels(hist_u=u, hist_u_ok=ok)
    inp = type(inp)(**{**inp.__dict__, "hist_visible": torch.ones_like(inp.hist_visible)})
    net = model.UntangledNet(seed=3, sign_head=True, clim_logit=-3.0).eval()
    with torch.no_grad():
        out = net(inp)
    assert out.sign_logit is not None
    p = torch.sigmoid(out.sign_logit)
    # the weights sum to one on a fully covered column, so the weighted mean of a constant is the constant
    assert torch.allclose(p, torch.full_like(p, 0.3), atol=1e-4)


def test_sign_head_falls_back_to_climatology_with_no_history() -> None:
    """With every flag off, the cover is zero and the head gives the climatology."""
    inp = make_inputs(b=2, channels=False)
    net = model.UntangledNet(seed=3, sign_head=True, clim_logit=-2.0).eval()
    with torch.no_grad():
        out = net(inp)
    assert out.sign_logit is not None
    assert torch.allclose(
        torch.sigmoid(out.sign_logit),
        torch.sigmoid(torch.tensor(-2.0)).expand_as(out.sign_logit),
        atol=1e-6,
    )


def test_sign_loss_ignores_unknown_targets() -> None:
    """The cross-entropy counts only the weighted tokens with a known truth."""
    logit = torch.zeros(1, 4)
    truth = torch.tensor([[1.0, 0.0, float("nan"), 1.0]])
    weight = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    got = model.sign_loss(logit_=logit, truth=truth, weight=weight)
    assert float(got) == pytest.approx(0.6931, abs=1e-3)


def test_attention_flag_does_not_change_the_forecast() -> None:
    """Recording attention returns the same mixture outputs."""
    net = model.UntangledNet(seed=4, sign_head=True).eval()
    with torch.no_grad():
        net.u_gamma.fill_(0.3)
        inp = make_inputs()
        a = net(inp)
        b = net(inp, return_attention=True)
    assert torch.equal(a.out, b.out)
    assert b.attention is not None and "within_0" in b.attention

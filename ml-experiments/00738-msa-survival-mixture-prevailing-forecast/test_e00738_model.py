"""Tests of the level head: zero start, the arms' outputs, and the visibility limit."""
# ruff: noqa: D103, E402

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
torch = pytest.importorskip("torch", reason="the level head tests need torch (workstation only)")
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 378, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))

import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00738_common as common
import e00738_model as model
import e00738_survival as surv


def make_inputs(*, batch: int = 4, seed: int = 0) -> mdl.MsaInputs:
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    visible = torch.rand(batch, 14, 48, generator=g) > 0.3
    visible[:, 0, :] = True  # at least one visible row, so A1 sees history
    return mdl.MsaInputs(
        primary=torch.arange(batch) % 16,
        lead_day=torch.arange(batch) % mdl.N_LEAD_DAYS,
        top_weather=r(batch, 48, 21),
        top_spread=r(batch, 48, 7).abs(),
        top_datetime=r(batch, 48, 21),
        hist_weather=r(batch, 14, 48, 21),
        hist_datetime=r(batch, 14, 48, 21),
        hist_power=torch.where(visible, r(batch, 14, 48), torch.zeros(())),
        hist_visible=visible,
        row_exists=torch.ones(batch, 14, dtype=torch.bool),
        day_type_top=torch.zeros(batch, 48, dtype=torch.long),
        day_type_hist=torch.zeros(batch, 14, 48, dtype=torch.long),
    )


def make_net(*, mode: str) -> model.LevelNet:
    curve = surv.GeneratorCurve() if mode in {"LD", "LM"} else None
    torch.manual_seed(0)
    net = model.LevelNet(mode=mode, target_std=np.full(16, 0.2), noise=0.05, curve=curve)
    net.eval()
    return net


@pytest.mark.parametrize("mode", common.HEAD_ARMS)
def test_untrained_head_arm_equals_the_base(mode: str) -> None:
    net = make_net(mode=mode)
    inp = make_inputs()
    out = net.level_forward(inp)
    base_out = net.base_forward(inp)[3]
    q_head = model.quantiles_of(out=out.out, components=out.components)
    k = out.components // 2 if mode in {"LM", "LR"} else out.components
    q_base = model.quantiles_of(out=base_out, components=k)
    assert torch.allclose(q_head, q_base, atol=1e-4)


def test_freeze_base_trains_only_the_head() -> None:
    net = make_net(mode="LD")
    net.freeze_base()
    trainable = {id(p) for p in net.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in net.level_parameters()}
    net.train()
    assert not net.blocks.training, "a frozen base must stay in evaluation mode"
    assert net.level_head.training


def test_head_arm_output_shapes_and_weights() -> None:
    inp = make_inputs()
    for mode, components in (("LP", 3), ("LD", 3), ("LM", 6), ("LR", 6)):
        out = make_net(mode=mode).level_forward(inp)
        assert out.out.shape == (4, 48, 3 * components)
        w = model.mixture_weights(out=out.out, components=components)
        assert torch.allclose(w.sum(-1), torch.ones(4, 48), atol=1e-5)


def test_a_signed_depth_moves_the_means_of_lp_and_ld() -> None:
    inp = make_inputs()
    net = make_net(mode="LP")
    torch.nn.init.constant_(net.level_head[-1].bias, 0.5)  # a depth of tanh(0.5) * limit
    out = net.level_forward(inp)
    seen = out.age > 0
    base = net.base_forward(inp)[3]
    k = 3
    moved = out.out[..., k : 2 * k] - base[..., k : 2 * k]
    assert torch.all(moved[~seen] == 0), "an unseen shift must leave the means alone"
    if seen.any():
        assert torch.allclose(moved[seen][:, 0, 0], out.depth[seen], atol=1e-6)


def test_ld_adds_depth_times_survival_ratio() -> None:
    inp = make_inputs()
    net = make_net(mode="LD")
    torch.nn.init.constant_(net.level_head[-1].bias, 0.5)
    out = net.level_forward(inp)
    base = net.base_forward(inp)[3]
    moved = out.out[..., 3:6] - base[..., 3:6]
    want = out.depth[:, None, None] * out.p[..., None]
    assert torch.allclose(moved, want.expand_as(moved), atol=1e-6)


def test_visibility_limit_leaves_level_features_bit_identical() -> None:
    """Perturbing every history value that is not visible changes nothing the head reads."""
    net = make_net(mode="LR")
    inp = make_inputs()
    h0, base, spread, _ = net.base_forward(inp)
    before = net.features_of(inp=inp, h0=h0, base=base, spread=spread)
    age_before = net.a1_age(inp=inp)
    hidden = ~inp.hist_visible
    assert hidden.any()
    noisy = torch.where(hidden, torch.full_like(inp.hist_power, 123.0), inp.hist_power)
    inp2 = inp.with_power(power=noisy, visible=inp.hist_visible)
    h02, base2, spread2, _ = net.base_forward(inp2)
    after = net.features_of(inp=inp2, h0=h02, base=base2, spread=spread2)
    assert torch.equal(before, after)
    assert torch.equal(age_before, net.a1_age(inp=inp2))


def test_features_have_the_planned_width() -> None:
    net = make_net(mode="LP")
    inp = make_inputs()
    h0, base, spread, _ = net.base_forward(inp)
    assert net.features_of(inp=inp, h0=h0, base=base, spread=spread).shape == (4, model.N_FEATURES)
    assert model.N_FEATURES == 108


def test_torch_hazard_ratio_matches_numpy() -> None:
    hz = surv.daily_hazards(curve=surv.GeneratorCurve())
    age = np.array([2.0, 5.0, 20.0])
    lead = np.array([0.5, 3.0, 9.5])
    want = surv.hazard_ratio(hazards=hz, age_days=age, lead_days=lead)
    h = torch.from_numpy(hz.astype(np.float32))[None].expand(3, -1)
    t = torch.from_numpy(age.astype(np.float32))
    a = t[:, None]
    ld = torch.from_numpy(lead.astype(np.float32))[:, None]
    s_a = torch.exp(-model.torch_cumulative_hazard(hazards=h, t_days=a))
    s_ah = torch.exp(-model.torch_cumulative_hazard(hazards=h, t_days=a + ld))
    got = (s_ah / s_a.clamp_min(common.SURVIVAL_FLOOR)).clamp(0, 1)[:, 0].numpy()
    np.testing.assert_allclose(got, want, atol=1e-5)

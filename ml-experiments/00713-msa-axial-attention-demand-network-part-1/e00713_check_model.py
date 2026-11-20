# ruff: noqa: E501, PT018, D103, PLR0915
"""Experiment 713: CPU checks of the network (WP2, KA1). Exit code 0 means pass.

The script follows the pattern of `e00651_test_model.py`. It is a check of this experiment's build
and not a test in `tests/`. It skips (exit code 0) when `torch` is missing.

1. Parameter counts of section 2.6 for every arm; the built module is the reference.
2. KA1: an untrained network's median equals `base`, computed independently in NumPy, for `M`, `E`,
   and `G`.
3. The output of a sample does not change when another sample of the batch changes.
4. Changing a masked power value changes no output (and the test can fail: changing a visible value
   changes the output once the read-out is non-zero).
5. The switches of arms `N`, `A`, `K`, `E`, and `G`; an untrained `E` or `G` equals an untrained `M`.
6. An untrained trunk is the identity.
7. Loading a Candidate `pretrained_P2` state leaves no unexpected key and no missing encoder key.
8. One backward pass gives finite gradients.
9. The read-out's `W_q` and `W_k` both receive a gradient within two steps (they are not a saddle).
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    print("SKIP: torch is not installed")
    sys.exit(0)

import e00713_model as mdl
import numpy as np
import torch

B = 6
DESIGN_TOTAL = 128_655


def random_batch(*, seed: int = 0, n: int = B) -> mdl.MsaInputs:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    visible = torch.rand(n, 14, 48, generator=g) < 0.8
    visible[:, 0, 43:] = False
    exists = torch.ones(n, 14, dtype=torch.bool)
    exists[:, -1] = torch.rand(n, generator=g) < 0.5
    visible &= exists[:, :, None]
    return mdl.MsaInputs(
        primary=torch.randint(0, 16, (n,), generator=g),
        lead_day=torch.randint(0, 14, (n,), generator=g),
        top_weather=r(n, 48, 21), top_spread=r(n, 48, 7).abs(), top_datetime=r(n, 48, 21),
        hist_weather=r(n, 14, 48, 21), hist_datetime=r(n, 14, 48, 21),
        hist_power=torch.where(visible, r(n, 14, 48), torch.zeros(n, 14, 48)),
        hist_visible=visible, row_exists=exists,
        day_type_top=torch.randint(0, 8, (n, 48), generator=g),
        day_type_hist=torch.randint(0, 8, (n, 14, 48), generator=g),
    )  # fmt: skip


def build(arm: str, seed: int = 0) -> mdl.MsaNet:
    torch.manual_seed(seed)
    return mdl.MsaNet(switches=mdl.ARM_SWITCHES[arm]).eval()


def numpy_base(inp: mdl.MsaInputs) -> np.ndarray:
    """The untrained read-out, independently: softmax of the day-type prior over visible tokens."""
    prior = mdl.beta_day_start().numpy()
    out = np.zeros((len(inp.primary), 48))
    for b in range(len(inp.primary)):
        for j in range(48):
            logit, z = [], []
            for r in range(14):
                if inp.hist_visible[b, r, j]:
                    logit.append(prior[inp.day_type_top[b, j], inp.day_type_hist[b, r, j]])
                    z.append(float(inp.hist_power[b, r, j]))
            if logit:
                w = np.exp(np.array(logit) - max(logit))
                out[b, j] = float((w / w.sum()) @ np.array(z))
    return out


def check_readout_gradients(inp: mdl.MsaInputs) -> None:
    """The read-out projections are not a saddle point.

    `W_k` gets a gradient at step 0, and after one optimiser step `W_q` gets one too. With both
    matrices at zero, both gradients stay zero for ever.
    """
    net = build("M")
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
    target = torch.randn(B, 48, generator=torch.Generator().manual_seed(5))
    norms: list[tuple[float, float]] = []
    for _ in range(2):
        opt.zero_grad()
        mdl.per_token_nll(out=net(inp).out, target=target).mean().backward()
        assert net.w_q.weight.grad is not None and net.w_k.weight.grad is not None
        norms.append((float(net.w_q.weight.grad.norm()), float(net.w_k.weight.grad.norm())))
        opt.step()
    print("read-out gradient norms (W_q, W_k) at steps 0 and 1:", norms)
    assert norms[0][1] > 0.0 and norms[1][0] > 0.0, norms
    assert float(net.w_k.weight.abs().max()) > 0.0 and float(net.w_q.weight.abs().max()) > 0.0


def main() -> int:
    inp = random_batch()
    totals: dict[str, int] = {}
    for arm in ("M", "A", "K", "N", "E", "G"):
        totals[arm] = mdl.parameter_count(model=build(arm))
    print("parameter counts:", totals, "(design table total for M:", DESIGN_TOTAL, ")")
    m1 = build("M")
    m1.freeze_encoders()
    trainable = mdl.parameter_count(model=m1, trainable_only=True)
    print("M1 trainable:", trainable, "(design: 120,527)")
    assert totals["E"] - totals["M"] == 2_560 and totals["G"] - totals["M"] == 4_032
    # the design total differs from the built module by this much; the built module is the reference
    print("difference of the built M from the design table:", totals["M"] - DESIGN_TOTAL)
    assert totals["M"] - mdl.parameter_count(model=m1, trainable_only=True) == 8_128
    print(
        "A fewer than M:", totals["M"] - totals["A"], "(33,280 attention weights + 256 LayerNorm)"
    )
    print("K fewer than M:", totals["M"] - totals["K"], "(66,560 + 120 key bias + 512 LayerNorm)")
    assert totals["M"] - totals["A"] == 33_280 + 2 * 128
    assert totals["M"] - totals["K"] == 66_560 + 2 * 60 + 4 * 128

    # KA1
    expected = numpy_base(inp)
    for arm in ("M", "E", "G"):
        net = build(arm)
        with torch.no_grad():
            res = net(inp)
            median = mdl.standardised_quantiles(out=res.out)[..., 5]
        diff = float((median - torch.from_numpy(expected).float()).abs().max())
        print(f"KA1 arm {arm}: max |median - base| = {diff:.2e}")
        assert diff <= 1e-5, (arm, diff)
        assert float((res.base - torch.from_numpy(expected).float()).abs().max()) <= 1e-5

    # E and G start from the same random weights as M at the same seed, for every shared parameter
    base_state = build("M", seed=3).state_dict()
    for arm in ("E", "G"):
        other = build(arm, seed=3).state_dict()
        shared = [k for k in other if not k.startswith("stem.")]
        assert set(shared) == set(base_state), arm
        for k in shared:
            assert torch.equal(other[k], base_state[k]), (arm, k)
    print("E and G draw the same initial weights as M for every shared parameter")

    # the prior start: a Monday-morning token puts about 50% / 45% / 5% on the three groups
    prior = mdl.beta_day_start()
    w = torch.exp(prior[0, [0] * 2 + [1, 2, 3, 4] * 2 + [5, 5, 6, 6]])
    w = w / w.sum()
    assert abs(float(w[:2].sum()) - 0.50) < 0.01 and abs(float(w[-4:].sum()) - 0.05) < 0.01

    # untrained trunk is the identity
    net = build("M")
    x = torch.randn(2, 15, 48, 64)
    for block in net.blocks:
        assert torch.allclose(
            block(x, row_exists=torch.ones(2, 15, dtype=torch.bool)), x, atol=1e-6
        )

    # independence of the samples of a batch
    net = build("M")
    for p in net.parameters():
        if p.requires_grad and float(p.detach().abs().sum()) == 0.0:
            torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        full = net(inp).out
        other = random_batch(seed=5)
        mixed = mdl.MsaInputs(
            **{k: torch.cat([getattr(inp, k)[:3], getattr(other, k)[3:]]) for k in inp.__dict__}
        )
        changed = net(mixed).out
    assert torch.allclose(full[:3], changed[:3], atol=1e-5), "a sample depends on another sample"
    assert not torch.allclose(full[3:], changed[3:], atol=1e-5), "the test cannot fail"

    # masked power changes nothing; visible power does change the output (positive control)
    with torch.no_grad():
        hidden = torch.where(
            inp.hist_visible, inp.hist_power, torch.full_like(inp.hist_power, 123.0)
        )
        same = net(inp.with_power(power=hidden, visible=inp.hist_visible)).out
        moved = net(
            inp.with_power(power=inp.hist_power + 0.5 * inp.hist_visible, visible=inp.hist_visible)
        ).out
    assert torch.allclose(full, same, atol=1e-6), "a masked power value reached the output"
    assert not torch.allclose(full, moved, atol=1e-4), (
        "visible power has no effect: the check cannot fail"
    )

    # switches
    n_out = build("N")
    with torch.no_grad():
        a = n_out(inp).out
        b = n_out(inp.with_power(power=inp.hist_power + 3.0, visible=inp.hist_visible)).out
    assert torch.equal(a, b), "arm N must not read history power"
    assert float(n_out(inp).no_history.min()) == 1.0
    assert build("A").blocks[0].within is None and build("A").blocks[0].across is not None
    assert build("K").blocks[0].within is None and build("K").blocks[0].across is None
    m = build("M")
    for arm in ("E", "G"):
        s = build(arm)
        s.load_state_dict(m.state_dict(), strict=False)
        with torch.no_grad():
            assert torch.allclose(m(inp).out, s(inp).out, atol=1e-6), arm
    # a stem changes the output once its last layer is non-zero, and only through the history power
    e = build("E")
    torch.nn.init.normal_(e.stem.out.weight, std=0.1)  # ty: ignore[invalid-argument-type, unresolved-attribute]
    with torch.no_grad():
        o1 = e(inp).out
        o2 = e(
            inp.with_power(power=inp.hist_power + 0.5 * inp.hist_visible, visible=inp.hist_visible)
        ).out
    assert not torch.allclose(o1, o2, atol=1e-4)

    # loading a Candidate pretrained state
    import e00574_model as cand  # ty: ignore[unresolved-import]

    torch.manual_seed(3)
    source = cand.PartDMdn(n_primaries=16, stage="pre", n_knots=2, n_b1=14)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pretrained.pt"
        torch.save(source.state_dict(), path)
        target = mdl.MsaNet()
        target.load_encoders(path=path)
    for key, value in source.state_dict().items():
        if key.startswith(mdl.ENCODER_PREFIXES):
            assert torch.equal(target.state_dict()[key], value), key
    print(
        "loaded encoder keys:",
        sorted(
            {k.split(".")[0] for k in source.state_dict() if k.startswith(mdl.ENCODER_PREFIXES)}
        ),
    )

    # one backward pass: finite gradients (not a fit)
    net = build("M")
    net.train()
    res = net(inp)
    loss = mdl.per_token_nll(out=res.out, target=torch.randn(B, 48)).mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    check_readout_gradients(inp)
    # the attention hook: off by default, the forecast is bit-identical with it on, weights sum to 1
    net = build("M")
    for q in net.parameters():
        if float(q.detach().abs().sum()) == 0.0:
            torch.nn.init.normal_(q, std=0.05)
    with torch.no_grad():
        plain = net(inp)
        hooked = net(inp, return_attention=True)
    assert plain.attention is None and plain.target_vector is None
    assert torch.equal(plain.out, hooked.out), "return_attention changed the forecast"
    att = hooked.attention
    assert att is not None and sorted(att) == ["across_0", "across_1", "within_0", "within_1"]
    b = len(inp.primary)
    assert att["across_0"].shape == (b, 48, mdl.HEADS, 15)
    assert att["within_0"].shape == (b, 15, mdl.HEADS, 48, 48)
    assert hooked.target_vector is not None and hooked.target_vector.shape == (b, 48, 64)
    assert torch.allclose(att["across_1"].sum(-1), torch.ones(b, 48, mdl.HEADS), atol=1e-5)
    assert torch.allclose(att["within_1"].sum(-1), torch.ones(b, 15, mdl.HEADS, 48), atol=1e-5)
    assert all(m.record is None for m in net.modules() if isinstance(m, mdl.Attention))
    print(
        "attention hook: forecast bit-identical, weights sum to 1, shapes",
        {k: tuple(v.shape) for k, v in att.items()},
    )
    # the report-only bank read-outs
    import e00713_variants as var

    sh = var.apply_variant(inputs=inp, variant="shuffle")
    assert not torch.equal(sh.hist_power, inp.hist_power)
    assert torch.allclose(sh.hist_power.sort(dim=1).values, inp.hist_power.sort(dim=1).values)
    assert torch.equal(sh.hist_visible.sum(1), inp.hist_visible.sum(1))
    assert torch.equal(sh.top_weather, inp.top_weather) and torch.equal(sh.primary, inp.primary)
    m9 = var.apply_variant(inputs=inp, variant="mask9")
    assert not bool(m9.hist_visible[:, 7:].any()) and not bool(m9.row_exists[:, 7:].any())
    assert torch.equal(m9.hist_visible[:, :7], inp.hist_visible[:, :7])
    assert var.apply_variant(inputs=inp, variant="plain") is inp
    with torch.no_grad():
        assert net(sh).out.shape == net(m9).out.shape == plain.out.shape
    print("bank read-outs: shuffle keeps each row's content, mask9 hides slots 7 to 13")
    print("model checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

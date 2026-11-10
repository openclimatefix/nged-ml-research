"""Leak tests of experiment 676 (run in the code review, on CPU, before any fit).

1. Masking regression (vacuous for fine-tuning, which masks nothing; kept as experiment 675's check
   of `node_inputs`). For 200 seeded samples a panel and a copy that differs only under random
   masks (Gaussian noise of 10 noise units, then a step of +20) give bit-identical node inputs, link
   features, and network outputs (float64, `torch.equal`) when the mask is applied before any
   statistic is computed. A positive control that computes the statistics on the unmasked copy must
   fail the same comparison.
2. Scored-target leak. `labels()` is patched to raise while a world is predicted and decoded, so no
   truth table can reach inference.
4. Receptive field (at the production window of 2,880 hours). Replacing the hours after a cut with noise instead of NaN leaves the posterior
   unchanged more than 800 hours before the cut (a window's quarter margin plus 80), and changes it just before the cut. The routine-step rate channel (a share over the whole 2,880-hour
   window, as in experiment 675) is zeroed for this test: it is the one input that reads the whole
   window, so hidden hours inside a window reach every hour of it. Training windows never contain
   hidden hours, and validation and scored worlds hold them as NaN.
3. Antisymmetry. Reversing a link's orientation swaps its two donating logits and its two lag heads,
   and leaves its off logit unchanged.

    uv run --no-sync python e00676_leaktests.py [1|2|3]   (PYTHONPATH as in `e00676_train.py`)
"""

# ruff: noqa: ANN201, E501, PT018

from __future__ import annotations

import sys

import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00675_inputs import node_inputs
from e00676_net import EdgeStateNet

N, SPAN = 6, 480
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]


def panel(gen: torch.Generator):
    """A random residual panel with gaps, noise units, and megawatt flags."""
    x = torch.randn(N, SPAN, generator=gen, dtype=torch.float64) * 2
    x[torch.rand(N, SPAN, generator=gen, dtype=torch.float64) < 0.02] = float("nan")
    sigma = 0.2 + torch.rand(N, generator=gen, dtype=torch.float64)
    mva = (torch.rand(N, generator=gen) < 0.5).to(torch.float64)
    return x, sigma, mva


def masks(gen: torch.Generator) -> torch.Tensor:
    """1 to 3 windows of 3 to 24 hours at about 40% of the Primaries, co-masked at neighbours."""
    m = torch.zeros(N, SPAN, dtype=torch.bool)
    for i in range(N):
        if torch.rand(1, generator=gen).item() < 0.4:
            for _ in range(int(torch.randint(1, 4, (1,), generator=gen))):
                ln = int(torch.randint(3, 25, (1,), generator=gen))
                s = int(torch.randint(60, SPAN - 60 - ln, (1,), generator=gen))
                m[i, s : s + ln] = True
                m[(i + 1) % N, s : s + ln] = True
    return m


def forward(net: EdgeStateNet, h: torch.Tensor, d: torch.Tensor, mva: torch.Tensor):
    adj = torch.full((N, N), 1 / N, dtype=torch.float64)
    with torch.no_grad():
        return net(h[None], d[None], adj, mva)


def test1(*, n_samples: int = 200) -> None:
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    net = EdgeStateNet(edges=EDGES).double().eval()
    for _ in range(n_samples):
        x, sigma, mva = panel(gen)
        m = masks(gen) & torch.isfinite(x)
        outs = []
        for variant in ("clean", "noise", "step"):
            xv = x.clone()
            if variant == "noise":
                xv[m] = 10.0 * torch.randn(int(m.sum()), generator=gen, dtype=torch.float64)
            if variant == "step":
                xv[m] = 20.0
            seen = torch.where(
                m, torch.full_like(xv, float("nan")), xv
            )  # mask first, then statistics
            h, d = node_inputs(x=seen, sigma=sigma, is_mva=mva, hour0=0)
            outs.append((h, d, forward(net, h, d, mva)))
        for h, d, o in outs[1:]:
            assert torch.equal(h, outs[0][0]) and torch.equal(d, outs[0][1])
            assert all(torch.equal(a, b) for a, b in zip(o, outs[0][2], strict=True))
        # positive control: statistics on the unmasked copy must differ
        bad = []
        for variant in ("clean", "step"):
            xv = x.clone()
            if variant == "step":
                xv[m] = 20.0
            bad.append(node_inputs(x=xv, sigma=sigma, is_mva=mva, hour0=0)[0])
        if m.any():
            assert not torch.equal(bad[0], bad[1]), "positive control failed to detect the leak"
    print("test 1 passed:", n_samples, "samples")


def test2() -> None:
    """No truth reaches inference: `labels` is patched to raise while a world is predicted and decoded."""
    import e00676_data as data
    import e00676_decode as dec

    rng = np.random.default_rng(0)
    resid = rng.normal(size=(N, SPAN * 2)).astype(np.float32)
    sigma = 0.2 + rng.random(N)
    mva = torch.zeros(N, dtype=torch.float64)
    torch.manual_seed(0)
    net = EdgeStateNet(edges=EDGES).float().eval()
    adj = torch.full((N, N), 1 / N, dtype=torch.float64)
    dec.SPAN = SPAN

    def boom(**_: object) -> None:
        raise AssertionError("labels() was called at inference")

    real = data.labels
    data.labels = boom
    try:
        post = dec.predict_world(net=net, resid=resid, sigma=sigma, mva=mva, adj=adj, device=torch.device("cpu"))  # fmt: skip
        names = [f"s{i}" for i in range(N)]
        xh = {s: np.repeat(resid[i].astype(np.float64) * sigma[i], 2) for i, s in enumerate(names)}
        dec.decode(post=post, edges=EDGES, sites=names, xh=xh, sigma=dict(zip(names, sigma, strict=True)), onset=2.0, cutoff=0.0, lo=0, hi=post.link.shape[1])  # fmt: skip
    finally:
        data.labels = real
    print("test 2 passed")


def test4() -> None:
    """Nothing beyond the receptive field leaks: hours after a cut change only nearby posteriors."""
    import e00676_decode as dec

    rng = np.random.default_rng(1)
    hours = 3 * 2880
    base = rng.normal(size=(N, hours))
    sigma = 0.2 + rng.random(N)
    mva = torch.zeros(N, dtype=torch.float64)
    torch.manual_seed(0)
    net = EdgeStateNet(edges=EDGES).float().eval()
    adj = torch.full((N, N), 1 / N, dtype=torch.float64)
    dec.SPAN = 2880
    cut = 5000
    hidden, noisy = base.copy(), base.copy()
    hidden[:, cut:] = np.nan
    noisy[:, cut:] = 10.0 * rng.normal(size=(N, hours - cut))
    from e00675_inputs import N_CHANNELS

    orig = dec.window_inputs

    def no_rate(**kwargs):
        h, d = orig(**kwargs)
        h[..., N_CHANNELS - 7] = 0.0  # the routine-step rate is a whole-window statistic by design
        return h, d

    dec.window_inputs = no_rate
    kw = {"net": net, "sigma": sigma, "mva": mva, "adj": adj, "device": torch.device("cpu")}
    a = dec.predict_world(resid=hidden.astype(np.float32), **kw)
    b = dec.predict_world(resid=noisy.astype(np.float32), **kw)
    safe = slice(0, cut - 2880 // 4 - 80)
    assert np.allclose(a.link[:, safe], b.link[:, safe], atol=1e-4), "future hours leaked"
    assert not np.allclose(a.link[:, cut - 100 : cut], b.link[:, cut - 100 : cut], atol=1e-4)
    dec.window_inputs = orig
    print("test 4 passed")


def test3() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    x, sigma, mva = panel(gen)
    h, d = node_inputs(x=x, sigma=sigma, is_mva=mva, hour0=0)
    n1 = EdgeStateNet(edges=EDGES).double().eval()
    n2 = EdgeStateNet(edges=[(b, a) for a, b in EDGES]).double().eval()
    n2.load_state_dict(n1.state_dict(), strict=False)
    n2.ia, n2.ib = n1.ib.clone(), n1.ia.clone()
    l1, g1, _ = forward(n1, h, d, mva)
    l2, g2, _ = forward(n2, h, d, mva)
    assert torch.allclose(l1[..., 0], l2[..., 0], atol=1e-10)
    assert torch.allclose(l1[..., 1], l2[..., 2], atol=1e-10) and torch.allclose(l1[..., 2], l2[..., 1], atol=1e-10)  # fmt: skip
    assert torch.allclose(g1[..., 0, :], g2[..., 1, :], atol=1e-10)
    print("test 3 passed")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for w in which:
        {"1": test1, "2": test2, "3": test3, "4": test4}[w]()

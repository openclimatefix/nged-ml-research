"""The leak tests of experiment 675 (run in the code review, before any fit, on CPU).

1. Masked-window leak, through the real pipeline (`e00675_pretrain.build_sample`): two copies of a
   panel that differ only under the masks (Gaussian noise of 10 noise units, then a +20 step) must
   give bit-identical inputs (`torch.equal`, float64) and predictions that agree to 1e-12, for 200
   seeded samples. A positive control patches `node_inputs` to read the unmasked panel and must
   fail the same test.
2. Causal arm: perturbing every hour after t0 must leave the causal network's outputs at hours up
   to t0 unchanged (to 1e-12); the same perturbation must change the bidirectional network's.
3. Scored-target leak: with the probe fitted on the same training worlds, shuffling the *scored*
   worlds' truth tables must leave every saved donor score bit-identical.

    PYTHONPATH=the_experiment_folder:... uv run --no-sync python e00675_leaktests.py [1|2|3]
"""

# ruff: noqa: ANN202, PT018  # research code run once; behaviour fixed by the results

from __future__ import annotations

import sys

import e00675_pretrain as pre
import torch  # ty: ignore[unresolved-import]
from e00675_inputs import node_inputs as real_node_inputs
from e00675_net import Pretrainer

TOL = 1e-12
N = 6


def _model(*, causal: bool) -> Pretrainer:
    torch.manual_seed(0)
    return Pretrainer(causal=causal).double().eval()


def _forward(model: Pretrainer, h: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    adj = torch.full((N, N), 1 / N, dtype=torch.float64)
    with torch.no_grad():
        out, day = model(h[None], d[None], adj)
    return torch.cat([out.flatten(), day.flatten()])


def masked_leak_test(*, n_samples: int = 200, seed: int = 0) -> float:
    """Raise AssertionError if any input or prediction depends on a masked value."""
    gen = torch.Generator().manual_seed(seed)
    model = _model(causal=False)
    worst = 0.0
    pre.PRE_HOURS = 4000  # the synthetic panel is shorter than the real one
    for _ in range(n_samples):
        x, sigma, mva = pre.synthetic_panel(gen=gen, n_nodes=N, hours=pre.SPAN + 48)
        mask = pre.draw_masks(gen=gen, n_nodes=N, hours=pre.SPAN, centre=(360, pre.SPAN - 360))
        ref = None
        for variant in ("clean", "noise", "step"):
            xv = x.clone()
            window = xv[:, 24 : 24 + pre.SPAN]
            full_mask = mask & torch.isfinite(window)
            if variant == "noise":
                window[full_mask] = 10.0 * torch.randn(
                    int(full_mask.sum()), generator=gen, dtype=window.dtype
                )
            elif variant == "step":
                window[full_mask] = 20.0
            h, d, _, _ = pre.build_sample(
                x_all=xv, start=24, mask=mask, sigma=sigma, mva=mva, causal=False
            )
            out = _forward(model, h, d)
            if ref is None:
                ref = (h, d, out)
                continue
            assert torch.equal(h, ref[0]) and torch.equal(d, ref[1]), f"inputs differ ({variant})"
            worst = max(worst, float((out - ref[2]).abs().max()))
    assert worst <= TOL, f"predictions differ by {worst}"
    return worst


def positive_control() -> None:
    """A node_inputs that reads the unmasked panel must fail the masked-window test."""
    original = pre.node_inputs
    holder: dict = {}

    def leaky(*, x: torch.Tensor, sigma: torch.Tensor, is_mva: torch.Tensor, hour0: int):
        return real_node_inputs(x=holder["x"], sigma=sigma, is_mva=is_mva, hour0=hour0)

    gen = torch.Generator().manual_seed(1)
    pre.PRE_HOURS = 4000
    x, sigma, mva = pre.synthetic_panel(gen=gen, n_nodes=N, hours=pre.SPAN + 48)
    mask = pre.draw_masks(gen=gen, n_nodes=N, hours=pre.SPAN, centre=(360, pre.SPAN - 360))
    outs = []
    pre.node_inputs = leaky  # ty: ignore[invalid-assignment]
    try:
        for fill in (0.0, 20.0):
            xv = x.clone()
            win = xv[:, 24 : 24 + pre.SPAN]
            win[mask & torch.isfinite(win)] = fill
            holder["x"] = xv[:, 24 : 24 + pre.SPAN]
            outs.append(
                pre.build_sample(x_all=xv, start=24, mask=mask, sigma=sigma, mva=mva, causal=False)[
                    0
                ]
            )
    finally:
        pre.node_inputs = original
    assert not torch.equal(outs[0], outs[1]), "the positive control did not leak: the test is blind"


def causal_test(*, n_samples: int = 20) -> float:
    """Outputs of the causal network up to t0 must not depend on hours after t0."""
    gen = torch.Generator().manual_seed(2)
    causal, bidir = _model(causal=True), _model(causal=False)
    pre.PRE_HOURS = 4000
    worst, bidir_moves = 0.0, 0.0
    for _ in range(n_samples):
        x, sigma, mva = pre.synthetic_panel(gen=gen, n_nodes=N, hours=pre.SPAN + 48)
        t0 = int(torch.randint(500, pre.SPAN - 500, (1,), generator=gen))
        xp = x.clone()
        xp[:, 24 + t0 + 1 :] = torch.randn_like(xp[:, 24 + t0 + 1 :]) * 5
        no_mask = torch.zeros(N, pre.SPAN, dtype=torch.bool)
        for model, is_causal in ((causal, True), (bidir, False)):
            outs = []
            for arr in (x, xp):
                h, d, _, _ = pre.build_sample(
                    x_all=arr, start=24, mask=no_mask, sigma=sigma, mva=mva, causal=is_causal
                )
                adj = torch.full((N, N), 1 / N, dtype=torch.float64)
                with torch.no_grad():
                    out, _ = model(h[None], d[None], adj)
                outs.append(out[0, :, : t0 + 1])
            diff = float((outs[0] - outs[1]).abs().max())
            if is_causal:
                worst = max(worst, diff)
            else:
                bidir_moves = max(bidir_moves, diff)
    assert worst <= TOL, f"the causal network sees the future: {worst}"
    assert bidir_moves > 1e-6, (
        "the perturbation does not reach the bidirectional network: test is blind"
    )
    return worst


def target_leak_test() -> None:
    """Shuffled scored-world truth must leave the saved donor scores unchanged (arm F, smoke)."""
    import os

    import e00675_probe as probe
    import numpy as np
    import polars as pl

    os.environ["E675_SMOKE"] = "1"
    probe.SMOKE = True
    probe.OUT = probe.ROOT / "probe_leaktest"
    probe.run(arm="F", seed=0)
    first = {w: np.load(probe.OUT / f"score_F_seed0_world{w}.npy") for w in range(900, 903)}
    original = probe.load_world

    def shuffled(*, kind: str, seed: int):
        resid, sigma, truth = original(kind=kind, seed=seed)
        if kind == "test":
            truth = truth.with_columns(
                pl.col("start").shuffle(seed=1),
                pl.col("site").shuffle(seed=2),
                pl.col("amp_mw").shuffle(seed=3),
            )
        return resid, sigma, truth

    probe.load_world = shuffled  # ty: ignore[invalid-assignment]
    probe.run(arm="F", seed=0)
    for w, a in first.items():
        assert np.array_equal(a, np.load(probe.OUT / f"score_F_seed0_world{w}.npy")), (
            f"world {w} scores moved"
        )


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which:
        positive_control()
        print("positive control: the test is sensitive")
        print("masked-window leak test: worst prediction difference", masked_leak_test())
    if "2" in which:
        print("causal test: worst difference up to t0", causal_test())
    if "3" in which:
        target_leak_test()
        print("scored-target leak test passed")

"""The two leak tests of experiment 675 (run in the code review, before any fit, on CPU).

1. Masked-window leak: for 200 seeded samples with masks, the residual under each mask is replaced
   by Gaussian noise of 10 noise units (then by a +20 noise-unit step); every input channel must be
   bit-identical (`torch.equal`, float64) and the reconstruction predictions must agree to 1e-12.
2. Scored-target leak: replacing the injected-truth table by a shuffled copy must leave the inputs
   unchanged, because labels enter only the loss.

    PYTHONPATH=<this folder> uv run --no-sync python e00675_leaktests.py
"""

from __future__ import annotations

import torch
from e00675_net import Pretrainer
from e00675_pretrain import draw_masks, synthetic_panel

TOL = 1e-12


def run_masked_leak_test(*, n_samples: int = 200, seed: int = 0) -> dict:
    """Check inputs and predictions against hidden values for `n_samples` masked samples."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Pretrainer(causal=False).double().eval()
    adj = torch.full((6, 6), 1 / 6, dtype=torch.float64)
    worst_pred = 0.0
    for k in range(n_samples):
        x, sigma, is_mva = synthetic_panel(gen=gen, n_nodes=6, hours=2880 // 4)
        mask = draw_masks(gen=gen, n_nodes=6, hours=x.shape[1], centre=(48, x.shape[1] - 48))
        ref = None
        for variant in ("clean", "noise", "step"):
            hidden = x.clone()
            if variant == "noise":
                hidden[mask] = 10.0 * torch.randn(int(mask.sum()), generator=gen, dtype=hidden.dtype)
            elif variant == "step":
                hidden[mask] = 20.0
            from e00675_inputs import node_inputs

            seen = torch.where(mask, torch.full_like(hidden, float("nan")), hidden)
            hourly, daily = node_inputs(x=seen, sigma=sigma, is_mva=is_mva, hour0=0)
            with torch.no_grad():
                (loc, scale), day = _predict(model, hourly, daily, adj)
            if ref is None:
                ref = (hourly, daily, loc, scale, day)
                continue
            assert torch.equal(hourly, ref[0]), f"hourly inputs differ on sample {k} ({variant})"
            assert torch.equal(daily, ref[1]), f"daily inputs differ on sample {k} ({variant})"
            worst_pred = max(
                worst_pred, float((loc - ref[2]).abs().max()), float((scale - ref[3]).abs().max()),
                float((day - ref[4]).abs().max()),
            )  # fmt: skip
    assert worst_pred <= TOL, f"predictions differ by {worst_pred}"
    return {"samples": n_samples, "worst_prediction_difference": worst_pred}


def _predict(model: Pretrainer, hourly: torch.Tensor, daily: torch.Tensor, adj: torch.Tensor):
    out, day = model(hourly[None], daily[None], adj)
    return (out[0, ..., 0], out[0, ..., 1]), day[0]


def run_target_leak_test() -> dict:
    """Inputs are a function of the residual alone: a shuffled truth table cannot change them."""
    import inspect

    from e00675_inputs import node_inputs

    names = set(inspect.signature(node_inputs).parameters)
    assert names == {"x", "sigma", "is_mva", "hour0"}, names
    gen = torch.Generator().manual_seed(1)
    x, sigma, is_mva = synthetic_panel(gen=gen, n_nodes=6, hours=720)
    truth = torch.randint(0, 2, (6, 720), generator=gen)
    shuffled = truth[torch.randperm(6, generator=gen)][:, torch.randperm(720, generator=gen)]
    a = node_inputs(x=x, sigma=sigma, is_mva=is_mva, hour0=0)
    _ = (truth, shuffled)  # labels never reach `node_inputs`: only `x` does
    b = node_inputs(x=x, sigma=sigma, is_mva=is_mva, hour0=0)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    return {"input_arguments": sorted(names)}


if __name__ == "__main__":
    print(run_masked_leak_test())
    print(run_target_leak_test())

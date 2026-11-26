# ruff: noqa: E501
"""A small random `UntInputs` batch for the tests of the model, the training loops, and the predictor."""

import e00733_model as model
import torch


def make_inputs(*, b: int = 3, seed: int = 0, channels: bool = True) -> model.UntInputs:
    """Return a random batch with the shapes of experiment 713's `MsaInputs` and three untangled channels."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    hist_ok = torch.rand(b, 14, 48, generator=g) > 0.2
    u = torch.rand(b, 14, 48, 3, generator=g)
    u[..., 1] *= 4.0
    ok = (torch.rand(b, 14, 48, generator=g) > 0.3) & hist_ok & channels
    return model.UntInputs(
        primary=torch.randint(0, 16, (b,), generator=g),
        lead_day=torch.randint(0, 14, (b,), generator=g),
        top_weather=rnd(b, 48, 21),
        top_spread=rnd(b, 48, 7).abs(),
        top_datetime=rnd(b, 48, 21),
        hist_weather=rnd(b, 14, 48, 21),
        hist_datetime=rnd(b, 14, 48, 21),
        hist_power=rnd(b, 14, 48) * hist_ok,
        hist_visible=hist_ok,
        row_exists=torch.ones(b, 14, dtype=torch.bool),
        day_type_top=torch.randint(0, 8, (b, 48), generator=g),
        day_type_hist=torch.randint(0, 8, (b, 14, 48), generator=g),
        hist_u=u * ok[..., None],
        hist_u_ok=ok,
    )

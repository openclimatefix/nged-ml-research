# ruff: noqa: E501
"""Experiment 554's modular lag-free network M0 with a monotone shortwave path (stage `pre` only).

Net demand, in standardised target units, is made non-increasing in the shortwave input (column 0)
with every other input held fixed. The construction has three parts:

- the weather encoder sees the other six weather columns, with column 0 set to zero, so no free path
  carries shortwave irradiance;
- a context network reads the datetime encoder's output and the Primary embedding (the context holds
  no weather) and returns, for each of `HIDDEN` units j, an amplitude `a_j`, a slope `s_j`, and an
  offset `b_j`;
- the shortwave effect is `h = sum_j softplus(a_j) * softplus(softplus(s_j) * sw + b_j)`. Each term
  is non-decreasing in `sw`, because `softplus(a_j) >= 0`, `softplus(s_j) >= 0`, and softplus is
  increasing. The sum `h` is therefore non-decreasing in `sw`.

`h` is subtracted from every mixture component's mean, so the mixture mean falls as `sw` rises,
whatever the other inputs are. The mixture's weights and scales do not read `sw`.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

HIDDEN: Final[int] = 16
CONTEXT_HIDDEN: Final[int] = 32
AMPLITUDE_INIT: Final[float] = -4.0
"""Initial amplitude pre-activation, so that `softplus(a)` starts near 0.02 and the path starts quiet."""


class MonotoneModularMdn(m.ModularMdn):
    """M0 (stage `pre`) plus a shortwave path that is non-increasing in net demand."""

    def __init__(self, *, n_primaries: int) -> None:
        """Build the network."""
        super().__init__(n_primaries=n_primaries, stage="pre")
        self.sw_context = nn.Sequential(
            nn.Linear(m.ENCODER_OUT + m.EMBEDDING, CONTEXT_HIDDEN),
            nn.GELU(),
            nn.Linear(CONTEXT_HIDDEN, 3 * HIDDEN),
        )
        last = self.sw_context[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.zero_()
            last.bias[:HIDDEN] = AMPLITUDE_INIT
            last.bias[HIDDEN : 2 * HIDDEN] = 0.0
            last.bias[2 * HIDDEN :] = torch.linspace(-2.0, 2.0, HIDDEN)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the mixture's 9 raw outputs with the monotone shortwave shift applied."""
        assert primary is not None
        sw = x[:, 0]
        emb = self.embedding(primary)
        weather = x[:, m.WEATHER_COLS].clone()
        weather[:, 0] = 0.0
        hw = nn.functional.gelu(self.weather(torch.cat([weather, emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        out = self.mixer_pre(core)
        amp, slope, offset = self.sw_context(torch.cat([hd, emb], dim=-1)).chunk(3, dim=-1)
        h = (
            nn.functional.softplus(amp)
            * nn.functional.softplus(nn.functional.softplus(slope) * sw[:, None] + offset)
        ).sum(dim=-1)
        k = m.COMPONENTS
        shift = torch.zeros_like(out)
        shift[:, k : 2 * k] = h[:, None]
        return out - shift

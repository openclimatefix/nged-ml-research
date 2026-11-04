# ruff: noqa: E501
"""M0 with the three trailing-temperature columns (experiment 515's `W-T`) added to the weather encoder.

The 37 non-lag columns of `e00559_model` are followed by 3 more columns (37 to 39): the mean forecast
temperature over the 6, 24, and 72 hours before the valid time, taken from the latest weather run
issued at or before the forecast origin (experiment 515's `previous` seam), standardised with the
fold's training rows. Arm A9 feeds the real columns and arm A9-null feeds experiment 526's matched
shuffled-day placebo of the same three columns.
"""

from typing import Final

import e00559_model as m
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

WT_COLS: Final = slice(37, 40)
N_INPUTS_WT: Final[int] = 40


class ModularMdnWt(m.ModularMdn):
    """`ModularMdn` whose weather encoder also reads the three trailing-temperature columns."""

    def __init__(self, *, n_primaries: int, stage: str) -> None:
        """Build the network with a 10-column weather encoder input."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        self.weather = m.base.mlp(
            n_in=7 + 3 + m.EMBEDDING,
            hidden=m.ENCODER_HIDDEN,
            layers=1,
            n_out=m.ENCODER_OUT,
            dropout=m.DROPOUT,
        )

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        weather_in = torch.cat([x[:, m.WEATHER_COLS], x[:, WT_COLS], emb], dim=-1)
        hw = nn.functional.gelu(self.weather(weather_in))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))

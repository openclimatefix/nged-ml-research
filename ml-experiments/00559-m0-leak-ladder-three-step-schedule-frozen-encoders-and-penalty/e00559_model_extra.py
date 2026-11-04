# ruff: noqa: E501
"""M0 with extra columns: `n_weather` ensemble-mean columns in the weather encoder and `n_spread` ensemble-spread columns in the fine-tuning mixer.

The extra columns follow the 37 columns of `e00559_model`: columns 37 to 37 + n_weather - 1 go to the
weather encoder, the next n_spread columns to the fine-tuning mixer (the pretraining mixer reads no spread).
"""

import e00559_model as m
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]


class ModularMdnExtra(m.ModularMdn):
    """`ModularMdn` with extra weather-encoder and spread inputs."""

    def __init__(self, *, n_primaries: int, stage: str, n_weather: int, n_spread: int) -> None:
        """Build the network with a widened weather encoder and fine-tuning mixer."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        self.n_weather, self.n_spread = n_weather, n_spread
        self.weather = m.base.mlp(
            n_in=7 + n_weather + m.EMBEDDING,
            hidden=m.ENCODER_HIDDEN,
            layers=1,
            n_out=m.ENCODER_OUT,
            dropout=m.DROPOUT,
        )
        wide = 2 * m.ENCODER_OUT + m.EMBEDDING
        self.mixer_ft = m.base.mlp(
            n_in=wide + 7 + 2 + n_spread,
            hidden=m.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        m.base.init_head(head=self.mixer_ft, components=m.COMPONENTS)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        extra_w = x[:, m.N_INPUTS : m.N_INPUTS + self.n_weather]
        extra_s = x[:, m.N_INPUTS + self.n_weather : m.N_INPUTS + self.n_weather + self.n_spread]
        hw = nn.functional.gelu(
            self.weather(torch.cat([x[:, m.WEATHER_COLS], extra_w, emb], dim=-1))
        )
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(
            torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS], extra_s], dim=-1)
        )

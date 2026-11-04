# ruff: noqa: E501
"""The modular lag-free demand network of experiment 545 (arms M0 and M1).

Three parts share the 8-wide Primary embedding of experiment 397's `PlainMdn`:

- a weather encoder over the 7 ensemble-mean weather columns (columns 0-6 of the 37 lag-free inputs);
- a datetime encoder over the 21 solar-geometry, calendar, and holiday columns (columns 14-34);
- a mixer over both encoders' outputs. Pretraining on ERA5 and CAMS rows uses `mixer_pre`, which sees
  nothing else. Fine-tuning on ENS rows discards it and attaches a fresh `mixer_ft`, which also
  sees the 7 ensemble-spread columns (7-13) and the 2 lead columns (35-36).

M1 adds an auxiliary mixture head after each encoder (given the encoder output and the embedding),
trained with weight `AUX_WEIGHT` on the same standardised target in pretraining and in fine-tuning.
Serving (`forward`) returns only the main mixture's 9 raw outputs.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

WEATHER_COLS: Final = slice(0, 7)
SPREAD_COLS: Final = slice(7, 14)
DATETIME_COLS: Final = slice(14, 35)
LEAD_COLS: Final = slice(35, 37)
N_INPUTS: Final[int] = 37
EMBEDDING: Final[int] = 8
ENCODER_HIDDEN: Final[int] = 64
ENCODER_OUT: Final[int] = 32
MIXER_HIDDEN: Final[int] = 64
AUX_HIDDEN: Final[int] = 32
COMPONENTS: Final[int] = 3
DROPOUT: Final[float] = 0.1
AUX_WEIGHT: Final[float] = 0.25
"""Weight of each auxiliary head's negative log-likelihood beside the main mixture's (M1 only)."""


class ModularMdn(nn.Module):
    """Weather encoder, datetime encoder, and a mixer, with optional auxiliary heads."""

    def __init__(self, *, n_primaries: int, aux: bool, stage: str) -> None:
        """Build the network; `stage` is `pre` (ERA5/CAMS pretraining) or `ft` (ENS fine-tuning)."""
        super().__init__()
        assert stage in {"pre", "ft"}
        self.stage, self.aux = stage, aux
        self.embedding = nn.Embedding(n_primaries, EMBEDDING)
        self.weather = base.mlp(
            n_in=7 + EMBEDDING, hidden=ENCODER_HIDDEN, layers=1, n_out=ENCODER_OUT, dropout=DROPOUT
        )
        self.datetime = base.mlp(
            n_in=21 + EMBEDDING, hidden=ENCODER_HIDDEN, layers=1, n_out=ENCODER_OUT, dropout=DROPOUT
        )
        wide = 2 * ENCODER_OUT + EMBEDDING
        self.mixer_pre = base.mlp(
            n_in=wide, hidden=MIXER_HIDDEN, layers=2, n_out=3 * COMPONENTS, dropout=DROPOUT
        )
        self.mixer_ft = base.mlp(
            n_in=wide + 7 + 2, hidden=MIXER_HIDDEN, layers=2, n_out=3 * COMPONENTS, dropout=DROPOUT
        )
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=COMPONENTS)
        if aux:
            self.head_weather = base.mlp(
                n_in=ENCODER_OUT + EMBEDDING, hidden=AUX_HIDDEN, layers=1, n_out=3 * COMPONENTS
            )
            self.head_datetime = base.mlp(
                n_in=ENCODER_OUT + EMBEDDING, hidden=AUX_HIDDEN, layers=1, n_out=3 * COMPONENTS
            )
            base.init_head(head=self.head_weather, components=COMPONENTS)
            base.init_head(head=self.head_datetime, components=COMPONENTS)

    def forward_all(self, x: torch.Tensor, primary: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the main mixture's raw outputs, then (with `aux`) the weather and datetime heads'."""
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        if self.stage == "pre":
            main = self.mixer_pre(core)
        else:
            main = self.mixer_ft(torch.cat([core, x[:, SPREAD_COLS], x[:, LEAD_COLS]], dim=-1))
        if not self.aux:
            return (main,)
        return (
            main,
            self.head_weather(torch.cat([hw, emb], dim=-1)),
            self.head_datetime(torch.cat([hd, emb], dim=-1)),
        )

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's raw outputs."""
        assert primary is not None
        return self.forward_all(x, primary)[0]

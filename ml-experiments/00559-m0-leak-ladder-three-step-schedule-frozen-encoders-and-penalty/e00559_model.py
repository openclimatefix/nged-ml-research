# ruff: noqa: E501
"""The modular lag-free demand network of experiment 545 (arm M0), without the auxiliary heads.

Three parts share the 8-wide Primary embedding of experiment 397's `PlainMdn`:

- a weather encoder over the 7 ensemble-mean weather columns (columns 0-6 of the 37 lag-free inputs);
- a datetime encoder over the 21 solar-geometry, calendar, and holiday columns (columns 14-34);
- a mixer over both encoders' outputs. Pretraining on ERA5 and CAMS rows uses `mixer_pre`, which sees
  nothing else (stage `pre`). Fine-tuning on ENS rows normally discards it and attaches a fresh
  `mixer_ft`, which also sees the 7 ensemble-spread columns (7-13) and the 2 lead columns (35-36)
  (stage `ft`). Arm M0-sep fine-tunes the stage `pre` network instead, so its pretrained mixer is kept.
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
COMPONENTS: Final[int] = 3
DROPOUT: Final[float] = 0.1


class ModularMdn(nn.Module):
    """Weather encoder, datetime encoder, and a mixer."""

    def __init__(self, *, n_primaries: int, stage: str) -> None:
        """Build the network; `stage` is `pre` (ERA5/CAMS pretraining, or M0-sep fine-tuning) or `ft`."""
        super().__init__()
        assert stage in {"pre", "ft"}
        self.stage = stage
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

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, SPREAD_COLS], x[:, LEAD_COLS]], dim=-1))

# ruff: noqa: D205, E501
"""Experiment 554's modular network (arm M0) with extra input columns that enter pre-training and
fine-tuning.

The extra columns are the last `n_extra` entries of the input row, after the 37 lag-free inputs.
Three routes, chosen by which encoder matches the columns' content:

- `weather`: the weather encoder reads the columns beside its 7 weather inputs and the Primary
  embedding.
- `datetime`: the datetime encoder reads them beside its 21 inputs and the Primary embedding.
- `parallel`: a small separate encoder (columns plus the Primary embedding, 32 hidden units, 16
  outputs) feeds both mixers by concatenation with the two encoders' outputs.

Unlike experiment 562, the added input units are randomly initialised (as the rest of the encoder
is)
because the network is pre-trained with them.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_BASE: Final[int] = m.N_INPUTS
PARALLEL_HIDDEN: Final[int] = 32
PARALLEL_OUT: Final[int] = 16


class ExtraMdn(nn.Module):
    """`m.ModularMdn` whose weather encoder, datetime encoder, or a parallel encoder also reads
    `n_extra` columns.
    """

    def __init__(self, *, n_primaries: int, stage: str, route: str, n_extra: int) -> None:
        """Build the network; `stage` is `pre` or `ft`, `route` is `weather`, `datetime`, or
        `parallel`.
        """
        super().__init__()
        assert stage in {"pre", "ft"}
        assert route in {"weather", "datetime", "parallel"}
        self.stage, self.route, self.n_extra = stage, route, n_extra
        self.embedding = nn.Embedding(n_primaries, m.EMBEDDING)
        n_weather = 7 + m.EMBEDDING + (n_extra if route == "weather" else 0)
        n_datetime = 21 + m.EMBEDDING + (n_extra if route == "datetime" else 0)
        self.weather = base.mlp(n_in=n_weather, hidden=m.ENCODER_HIDDEN, layers=1, n_out=m.ENCODER_OUT, dropout=m.DROPOUT)  # fmt: skip
        self.datetime = base.mlp(n_in=n_datetime, hidden=m.ENCODER_HIDDEN, layers=1, n_out=m.ENCODER_OUT, dropout=m.DROPOUT)  # fmt: skip
        wide = 2 * m.ENCODER_OUT + m.EMBEDDING
        if route == "parallel":
            self.parallel = base.mlp(n_in=n_extra + m.EMBEDDING, hidden=PARALLEL_HIDDEN, layers=1, n_out=PARALLEL_OUT, dropout=m.DROPOUT)  # fmt: skip
            wide += PARALLEL_OUT
        self.mixer_pre = base.mlp(n_in=wide, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT)  # fmt: skip
        self.mixer_ft = base.mlp(n_in=wide + 7 + 2, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT)  # fmt: skip
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=m.COMPONENTS)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        extra = x[:, N_BASE : N_BASE + self.n_extra]
        weather_in = torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)
        datetime_in = torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)
        if self.route == "weather":
            weather_in = torch.cat([weather_in, extra], dim=-1)
        elif self.route == "datetime":
            datetime_in = torch.cat([datetime_in, extra], dim=-1)
        hw = nn.functional.gelu(self.weather(weather_in))
        hd = nn.functional.gelu(self.datetime(datetime_in))
        parts = [hw, hd, emb]
        if self.route == "parallel":
            parts.append(nn.functional.gelu(self.parallel(torch.cat([extra, emb], dim=-1))))
        core = torch.cat(parts, dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))

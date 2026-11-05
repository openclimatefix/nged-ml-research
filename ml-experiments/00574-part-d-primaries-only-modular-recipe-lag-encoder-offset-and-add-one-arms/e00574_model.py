# ruff: noqa: E501
"""Experiment 574: experiment 572's v3+D5 network with the optional add-ons of Part D.

The base is the lag encoder plus the D5 output offset. Input row layout: 37 lag-free columns, 8 lag
columns (37-44), the knot-position column (45), then the optional B1 columns (arm P2: `n_b1` thermal-lag
and solar-gain columns that the weather encoder reads, in pre-training and fine-tuning), then the
optional spread block (arm P1: experiment 568's seven relative-spread columns, fine-tuning only).

- `n_b1`: the weather encoder reads the B1 columns beside its 7 weather inputs. At a lag's own time
  (the lag encoder's expected embedding) the B1 slots read 0, the standardised column mean.
- `spread`: the fine-tuning mixer also reads the 7 spread columns (pre-training has none, as in
  experiment 568's T4).
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00572_model as m572  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

TAU: Final[int] = m572.TAU_COL_LAG
"""Knot-position column (45)."""
assert m560.N_INPUTS == 45


def b1_cols(*, n_b1: int) -> slice:
    """Return the B1 columns (46 onwards)."""
    return slice(TAU + 1, TAU + 1 + n_b1)


def spread_cols(*, n_b1: int) -> slice:
    """Return the seven spread-block columns (after the B1 columns)."""
    return slice(TAU + 1 + n_b1, TAU + 8 + n_b1)


class PartDMdn(m572.LagOffsetMdn):
    """Experiment 572's `LagOffsetMdn` plus the optional B1 columns and spread block."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, spread: bool = False, n_b1: int = 0
    ) -> None:
        """Build the network; with `spread` the fine-tuning mixer's input is 7 columns wider."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        self.spread = spread and stage == "ft"
        self.n_b1 = n_b1
        if n_b1:
            self.weather = base.mlp(
                n_in=7 + m.EMBEDDING + n_b1, hidden=m.ENCODER_HIDDEN, layers=1,
                n_out=m.ENCODER_OUT, dropout=m.DROPOUT,
            )  # fmt: skip
        if self.spread:
            old = self.mixer_ft[0].in_features  # ty: ignore[not-subscriptable,unresolved-attribute]
            self.mixer_ft = base.mlp(
                n_in=old + 7,
                hidden=m.MIXER_HIDDEN,
                layers=2,
                n_out=3 * m.COMPONENTS,
                dropout=m.DROPOUT,
            )
            base.init_head(head=self.mixer_ft, components=m.COMPONENTS)

    def _expected(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return (weather, datetime) embeddings at each lag's own time; B1 slots read 0 there."""
        if not self.n_b1:
            return super()._expected(x=x, primary=primary, emb=emb)
        assert self.table is not None, "attach the expected-input table first"
        out = []
        for j in range(2):
            idx = x[:, 43 + j].long()
            row = self.table[primary, idx.clamp(min=0)].float()
            row = torch.where((idx >= 0)[:, None], row, torch.zeros_like(row))
            pad = torch.zeros(len(row), self.n_b1, device=row.device)
            hw = nn.functional.gelu(self.weather(torch.cat([row[:, :7], emb, pad], dim=-1)))
            hd = nn.functional.gelu(self.datetime(torch.cat([row[:, 7:], emb], dim=-1)))
            out.append((hw, hd))
        return out

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs, with the offset added to the component means."""
        assert primary is not None
        emb = self.embedding(primary)
        weather_in = [x[:, m.WEATHER_COLS], emb]
        if self.n_b1:
            weather_in.append(x[:, b1_cols(n_b1=self.n_b1)])
        hw = nn.functional.gelu(self.weather(torch.cat(weather_in, dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        if self.stage == "pre":
            out = self.mixer_pre(core)
        else:
            parts = [core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]]
            if self.spread:
                parts.append(x[:, spread_cols(n_b1=self.n_b1)])
            out = self.mixer_ft(torch.cat(parts, dim=-1))
        off = self.offset(x[:, TAU], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)

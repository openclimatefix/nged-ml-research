# ruff: noqa: E501
"""Experiment 574: experiment 572's v3+D5 network with the optional add-ons of Part D.

The base is the lag encoder plus the D5 output offset.

Input row layout: 37 lag-free columns, 8 lag columns (37-44), the knot-position column (45), then
the optional spread block (46-52, arm P1: experiment 568's seven relative-spread columns).

- `spread`: the fine-tuning mixer also reads the 7 spread columns (pre-training has none, as in
  experiment 568's T4).
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00572_model as m572  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

SPREAD_COLS: Final[slice] = slice(m572.TAU_COL_LAG + 1, m572.TAU_COL_LAG + 8)
"""The seven spread-block columns (46-52)."""


class PartDMdn(m572.LagOffsetMdn):
    """Experiment 572's `LagOffsetMdn` plus the optional spread block in the fine-tuning mixer."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int, spread: bool = False) -> None:
        """Build the network; with `spread` the fine-tuning mixer's input is 7 columns wider."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        self.spread = spread and stage == "ft"
        if self.spread:
            old = self.mixer_ft[0].in_features  # ty: ignore[not-subscriptable,unresolved-attribute]
            self.mixer_ft = base.mlp(
                n_in=old + 7, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT
            )
            base.init_head(head=self.mixer_ft, components=m.COMPONENTS)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs, with the offset added to the component means."""
        assert primary is not None
        nn = torch.nn
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        if self.stage == "pre":
            out = self.mixer_pre(core)
        else:
            parts = [core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]]
            if self.spread:
                parts.append(x[:, SPREAD_COLS])
            out = self.mixer_ft(torch.cat(parts, dim=-1))
        off = self.offset(x[:, m572.TAU_COL_LAG], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)


assert m560.N_INPUTS == 45

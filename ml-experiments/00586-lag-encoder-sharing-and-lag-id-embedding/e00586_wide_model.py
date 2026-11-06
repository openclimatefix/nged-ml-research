# ruff: noqa: E501
"""Experiment 586, stage 2: the wide layout (day, week, 2-week, and year-ago lags) in three wirings.

The input columns and lag block are experiment 585's with `periods = (-7, 365)`: 37 lag-free columns,
the 8 columns of the day and week lags, 3 columns each for the 2-week lag (a second weekly lag, 14 days
before the target's newest same weekday) and the year-ago lag (365 days, the same calendar date;
experiment 585's Y1c), the grid indices of the four lag times, and the D5 knot position in column 53.

- `s0w`: experiment 585's `MultiLagOffsetMdn` unchanged: one encoder reads the four lags and their
  eight expected embeddings together and gives one 32-number embedding to the mixer;
- `s1w`: one encoder with weights shared across the four lags reads one lag at a time (value, missing
  flag, age, that lag's expected weather and datetime embeddings, the Primary embedding) plus a learnt
  8-number embedding of the lag kind (day, week, 2-week, year-ago); the four 32-number outputs are
  concatenated in that fixed order, so the mixer's input is 96 numbers wider than `s0w`'s;
- `s3w`: as `s1w` with a separate encoder (own weights) for each lag and no lag ID.

Block dropout is experiment 585's: whole origin days mask the day lag with the week lag or alone, and
each extra lag has its own independent whole-day mask. A masked lag's three columns carry the absent
code, and its encoder output is still concatenated.
"""

import e00397_models as base  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00585_model as m585  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

KINDS = ("s0w", "s1w", "s3w")
N_LAGS = 4
LAG_ID_DIM = 8


class WideLagMdn(m585.MultiLagOffsetMdn):
    """The wide layout with one lag at a time through a shared or per-lag encoder."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int, kind: str) -> None:
        """Build the network, replace the lag encoder, and widen both mixers."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_extra=2)
        assert kind in {"s1w", "s3w"}
        self.kind = kind
        e = m545.EMBEDDING
        id_dim = LAG_ID_DIM if kind == "s1w" else 0
        n_enc = N_LAGS if kind == "s3w" else 1
        self.lag = nn.ModuleList(
            [
                base.mlp(
                    n_in=3 + 2 * m545.ENCODER_OUT + e + id_dim,
                    hidden=m560.LAG_HIDDEN,
                    layers=1,
                    n_out=m560.LAG_OUT,
                    dropout=m545.DROPOUT,
                )
                for _ in range(n_enc)
            ]
        )
        if kind == "s1w":
            self.lag_id = nn.Embedding(N_LAGS, LAG_ID_DIM)
        wide = 2 * m545.ENCODER_OUT + N_LAGS * m560.LAG_OUT + e
        self.mixer_pre = base.mlp(
            n_in=wide, hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT,
        )  # fmt: skip
        self.mixer_ft = base.mlp(
            n_in=wide + 7 + 2, hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT,
        )  # fmt: skip
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=m.COMPONENTS)

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the four lags' encoder outputs in the order day, week, 2-week, year-ago."""
        k = self.n_extra
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        expected = [(dw, dd), (ww, wd)]
        for j in range(k):
            idx = x[:, 45 + 3 * k + j].long()
            table = self.table_extra if self.stage == "ft" and self.extra_uses_pre[j] else None
            expected.append(self._expected_at(idx=idx, primary=primary, emb=emb, table=table))
        cols = [m560.DAY, m560.WEEK] + [slice(45 + 3 * j, 48 + 3 * j) for j in range(k)]
        outs = []
        for j, (c, (hw, hd)) in enumerate(zip(cols, expected, strict=True)):
            parts = [x[:, c], hw, hd, emb]
            if self.kind == "s1w":
                parts.append(
                    self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
                )
            outs.append(self.lag[j if self.kind == "s3w" else 0](torch.cat(parts, dim=-1)))
        return torch.cat(outs, dim=-1)

# ruff: noqa: E501
"""Experiment 586: experiment 572's v3+D5 network with the lag encoder wired in four other ways.

All four read the same 45 + 1 input columns as v3+D5 (37 lag-free columns, the day and week lag
value, missing flag, and age, the grid indices of the two lag times, and the D5 knot position). They
differ in how the two lags reach the mixer. Each lag is read with its own expected weather and
datetime embeddings (the weather and datetime encoders applied at the lag's own time), as in
experiment 560's `b_shared` and `b_unshared`.

- `s1`: one encoder with weights shared across the two lags, plus a learnt 8-number embedding of
  the lag kind (row 0 is the day lag, row 1 the week lag);
- `s2`: the shared encoder with no lag identity (experiment 560's `b_shared` on the D5 recipe);
- `s3`: a separate encoder per lag (experiment 560's `b_unshared` on the D5 recipe);
- `s4`: the shared encoder plus a fixed 9-number encoding of the lag's nominal period (1 or 7 days):
  log2 of the period, then its sine and cosine at 4 frequencies. Nothing in it is learnt.

The two encoder outputs are concatenated, day lag first, so the mixer's input is 32 numbers wider
than v3+D5's.
"""

import math
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00572_model as mod  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

KINDS: Final[tuple[str, ...]] = ("s1", "s2", "s3", "s4")
LAG_ID_DIM: Final[int] = 8
PERIOD_DAYS: Final[tuple[float, float]] = (1.0, 7.0)
PERIOD_FREQUENCIES: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 4.0)
PERIOD_DIM: Final[int] = 1 + 2 * len(PERIOD_FREQUENCIES)


def period_code() -> torch.Tensor:
    """Return the fixed (2, 9) encoding of the day and week lag periods."""
    rows = []
    for days in PERIOD_DAYS:
        lg = math.log2(days)
        row = [lg]
        for f in PERIOD_FREQUENCIES:
            row += [math.sin(2 * math.pi * f * lg), math.cos(2 * math.pi * f * lg)]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


class SharedLagOffsetMdn(mod.LagOffsetMdn):
    """v3+D5 whose lag encoder reads one lag at a time (see the module docstring)."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int, kind: str) -> None:
        """Build the network, then replace the lag encoder and widen both mixers."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        assert kind in KINDS
        self.kind = kind
        e = m545.EMBEDDING
        id_dim = {"s1": LAG_ID_DIM, "s2": 0, "s3": 0, "s4": PERIOD_DIM}[kind]
        n_enc = 2 if kind == "s3" else 1
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
        if kind == "s1":
            self.lag_id = nn.Embedding(2, LAG_ID_DIM)
        if kind == "s4":
            self.register_buffer("period", period_code())
        wide = 2 * m545.ENCODER_OUT + 2 * m560.LAG_OUT + e
        self.mixer_pre = base.mlp(
            n_in=wide, hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT,
        )  # fmt: skip
        self.mixer_ft = base.mlp(
            n_in=wide + 7 + 2, hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT,
        )  # fmt: skip
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=m.COMPONENTS)

    def _identity(self, *, j: int, rows: int, device: torch.device) -> list[torch.Tensor]:
        """Return the lag-identity columns for lag `j` (empty for s2 and s3)."""
        if self.kind == "s1":
            idx = torch.full((rows,), j, dtype=torch.long, device=device)
            return [self.lag_id(idx)]
        if self.kind == "s4":
            return [self.period[j][None, :].expand(rows, -1)]  # ty: ignore[not-subscriptable]
        return []

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the two lags' encoder outputs, day lag first."""
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        outs = []
        for j, (cols, hw, hd) in enumerate(((m560.DAY, dw, dd), (m560.WEEK, ww, wd))):
            inp = torch.cat(
                [x[:, cols], hw, hd, emb, *self._identity(j=j, rows=len(x), device=x.device)],
                dim=-1,
            )
            outs.append(self.lag[-1 if j == 1 else 0](inp))
        return torch.cat(outs, dim=-1)

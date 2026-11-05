"""Experiment 581: experiment 572's v3+D5 network whose lag encoder also reads seven robust columns.

The input has 37 lag-free columns, the 8 lag columns of experiment 560 (the week lag replaced by the
5-week median), 7 robust columns (45-51, see `e00581_lags`), and the D5 knot position in column 52.
The lag encoder is experiment 560's mode-v3 encoder with 7 more inputs.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00572_model as mod  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_ROBUST: Final[int] = 7
ROBUST: Final[slice] = slice(45, 52)
TAU_COL: Final[int] = 52
"""Knot-position column of the robust network."""


class RobustLagOffsetMdn(mod.LagOffsetMdn):
    """v3+D5 with seven robust lag columns in the lag encoder's input."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network, then widen the lag encoder by `N_ROBUST` inputs."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        e = m545.EMBEDDING
        expected = 4 * m545.ENCODER_OUT
        self.lag = nn.ModuleList(
            [
                base.mlp(
                    n_in=6 + N_ROBUST + expected + e,
                    hidden=m560.LAG_HIDDEN,
                    layers=1,
                    n_out=m560.LAG_OUT,
                    dropout=m545.DROPOUT,
                )
            ]
        )

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the lag encoder's output (mode v3 plus the robust columns)."""
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        return self.lag[0](
            torch.cat([x[:, m560.LAG_COLS], x[:, ROBUST], dw, dd, ww, wd, emb], dim=-1)
        )

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs with the offset added to the component means."""
        assert primary is not None
        out = m560.LagModularMdn.forward(self, x, primary)
        off = self.offset(x[:, TAU_COL], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)

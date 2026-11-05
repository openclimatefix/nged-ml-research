"""Experiment 585: experiment 572's v3+D5 network whose lag encoder also reads `K` extra lags.

The input has 37 lag-free columns, the 8 lag columns of experiment 560, 4 columns per extra lag (see
`e00585_lags`), and the D5 knot position after them. The lag encoder is experiment 560's mode-v3
encoder with, for each extra lag, 3 more input columns and that lag's expected weather and datetime
embeddings (the same weather and datetime encoders, so the same weights as for the day and week lags).
"""

import e00397_models as base  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00572_model as mod  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]


class MultiLagOffsetMdn(mod.LagOffsetMdn):
    """v3+D5 with `n_extra` extra lags in the lag encoder's input."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_extra: int, drop_day: bool = False
    ) -> None:
        """Build the network, then widen the lag encoder by the extra lags' inputs."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        self.n_extra, self.drop_day = n_extra, drop_day
        e = m545.EMBEDDING
        expected = 2 * m545.ENCODER_OUT * (2 + n_extra)
        self.lag = nn.ModuleList(
            [
                base.mlp(
                    n_in=6 + 3 * n_extra + expected + e,
                    hidden=m560.LAG_HIDDEN,
                    layers=1,
                    n_out=m560.LAG_OUT,
                    dropout=m545.DROPOUT,
                )
            ]
        )

    @property
    def tau_col(self) -> int:
        """Return the knot-position column."""
        return 45 + 4 * self.n_extra

    def _expected_at(
        self, *, idx: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weather and datetime embeddings at grid indices `idx` (zeros where idx < 0)."""
        assert self.table is not None, "attach the expected-input table first"
        row = self.table[primary, idx.clamp(min=0)].float()
        row = torch.where((idx >= 0)[:, None], row, torch.zeros_like(row))
        hw = nn.functional.gelu(self.weather(torch.cat([row[:, :7], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([row[:, 7:], emb], dim=-1)))
        return hw, hd

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the lag encoder's output (mode v3 plus the extra lags)."""
        k = self.n_extra
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        if self.drop_day:
            dw, dd = torch.zeros_like(dw), torch.zeros_like(dd)
        parts = [x[:, m560.LAG_COLS], x[:, 45 : 45 + 3 * k], dw, dd, ww, wd]
        for j in range(k):
            idx = x[:, 45 + 3 * k + j].long()
            parts.extend(self._expected_at(idx=idx, primary=primary, emb=emb))
        return self.lag[0](torch.cat([*parts, emb], dim=-1))

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs with the offset added to the component means."""
        assert primary is not None
        out = m560.LagModularMdn.forward(self, x, primary)
        off = self.offset(x[:, self.tau_col], primary)
        c = m.COMPONENTS
        return torch.cat([out[:, :c], out[:, c : 2 * c] + off, out[:, 2 * c :]], dim=-1)

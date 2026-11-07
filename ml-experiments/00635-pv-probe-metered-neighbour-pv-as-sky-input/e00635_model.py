"""Experiment 554's modular lag-free network M0 (stage `pre`) with a metered-farm sky input.

The metered column rides in input column 35, which stage `pre` otherwise ignores. Arm `b` appends
it to the seven weather columns. Arm `c` appends it too and sets the CAMS shortwave (column 0) to
zero, so the metered farm is the only sky signal.
"""

import e00397_models as base  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

METERED_COL = 35


class MeteredMdn(m.ModularMdn):
    """M0 (stage `pre`) whose weather encoder also reads the metered column."""

    def __init__(self, *, n_primaries: int, arm: str) -> None:
        """Build the network for arm `b` or `c`."""
        super().__init__(n_primaries=n_primaries, stage="pre")
        assert arm in {"b", "c"}
        self.arm = arm
        self.weather = base.mlp(
            n_in=8 + m.EMBEDDING, hidden=m.ENCODER_HIDDEN, layers=1, n_out=m.ENCODER_OUT,
            dropout=m.DROPOUT,
        )  # fmt: skip

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        weather = x[:, m.WEATHER_COLS].clone()
        if self.arm == "c":
            weather[:, 0] = 0.0
        weather = torch.cat([weather, x[:, METERED_COL : METERED_COL + 1]], dim=-1)
        hw = nn.functional.gelu(self.weather(torch.cat([weather, emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        return self.mixer_pre(torch.cat([hw, hd, emb], dim=-1))

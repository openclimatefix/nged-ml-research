# ruff: noqa: E501
"""Experiment 608: experiment 591's A1 network with one group of inputs removed.

Removal multiplies the inputs by zero, in pre-training, step 2, and the daily fine-tune alike (the
networks are trained from scratch with the input absent), so the layer shapes stay those of A1. As in
experiment 606's `PermLagMdn`, from which the lag-encoder logic is copied.

- `drop_id`: the learnt lag-ID embedding reads zero;
- `drop_age`: the lag age column of each lag reads zero;
- `drop_expected`: the expected-weather and expected-datetime embeddings of the lag's own time read zero;
- `drop_spread`: the seven ensemble-spread columns of the fine-tuning mixer read zero.
"""

from dataclasses import dataclass

import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

AGE_OFFSET = 2
"""Position of the lag age inside a lag's three columns (value, missing flag, age)."""


@dataclass(frozen=True)
class Drops:
    """Which inputs of A1 are removed."""

    drop_id: bool = False
    drop_age: bool = False
    drop_expected: bool = False
    drop_spread: bool = False


class DropMdn(aux.AuxLagMdn):
    """A1's network with the inputs of `drops` set to zero."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, drops: Drops) -> None:
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind="shared")
        self.drops = drops

    def lag_embedding(self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Return the two lags' encoder outputs, day lag first, without the removed inputs."""
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        outs = []
        for j, (cols, hw, hd) in enumerate(((m560.DAY, dw, dd), (m560.WEEK, ww, wd))):
            lag = x[:, cols]
            if self.drops.drop_age:
                lag = torch.cat([lag[:, :AGE_OFFSET], torch.zeros_like(lag[:, AGE_OFFSET:])], dim=1)
            if self.drops.drop_expected:
                hw, hd = torch.zeros_like(hw), torch.zeros_like(hd)
            ident = self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
            if self.drops.drop_id:
                ident = torch.zeros_like(ident)
            outs.append(self.lag[0](torch.cat([lag, hw, hd, emb, ident], dim=-1)))
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs, with the spread columns zeroed when removed."""
        if self.drops.drop_spread:
            x = x.clone()
            x[:, m.SPREAD_COLS] = 0.0
        return super().forward(x, primary)

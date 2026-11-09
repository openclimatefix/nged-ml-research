# ruff: noqa: D107
"""Experiment 677: experiment 591's A1 network with whole input groups removed, for the ladder steps 1 to 4.

Removal multiplies the inputs by zero in pre-training, stage 2, and stage 3 alike (the networks are
trained from scratch with the group absent), so the layer shapes stay those of A1 and the stored
code paths of experiments 574 and 591 run unchanged. The pattern is experiment 612's `DropMdn`.

- `drop_weather`: the 7 weather means, the 7 spread columns, and the 14 B1 columns read zero (the
  weather encoder sees only the Primary embedding);
- `drop_datetime`: the 21 date-time columns read zero (the date-time encoder sees only the Primary
  embedding);
- `drop_lead`: the 2 lead columns read zero (ladder steps 1 and 2; the lead columns enter at step 3);
- `drop_day_lag`, `drop_week_lag`: the lag encoder's 32-wide output for that lag reads zero.
"""

from dataclasses import dataclass

import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00574_model as mod  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]


@dataclass(frozen=True)
class Masks:
    """Which input groups of A1 are removed."""

    drop_weather: bool = False
    drop_datetime: bool = False
    drop_lead: bool = False
    drop_day_lag: bool = False
    drop_week_lag: bool = False


STEP_MASKS: dict[str, Masks] = {
    "1": Masks(drop_weather=True, drop_lead=True, drop_day_lag=True, drop_week_lag=True),
    "2": Masks(drop_datetime=True, drop_lead=True, drop_day_lag=True, drop_week_lag=True),
    "3": Masks(drop_day_lag=True, drop_week_lag=True),
    "4": Masks(drop_week_lag=True),
    "5": Masks(),
}
"""The removed groups of ladder steps 1 to 5 (step 5 is experiment 591's A1 network, nothing removed)."""


class LadderMdn(aux.AuxLagMdn):
    """A1's network with the input groups of `masks` set to zero."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, masks: Masks
    ) -> None:
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind="shared"
        )
        self.masks = masks

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the two lags' encoder outputs, with a removed lag's 32 columns set to zero."""
        out = super().lag_embedding(x=x, primary=primary, emb=emb)
        keep = torch.ones(2 * m560.LAG_OUT, dtype=out.dtype, device=out.device)
        if self.masks.drop_day_lag:
            keep[: m560.LAG_OUT] = 0.0
        if self.masks.drop_week_lag:
            keep[m560.LAG_OUT :] = 0.0
        return out * keep

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs, with the removed input columns zeroed."""
        if self.masks.drop_weather or self.masks.drop_datetime or self.masks.drop_lead:
            x = x.clone()
            if self.masks.drop_weather:
                x[:, m.WEATHER_COLS] = 0.0
                x[:, m.SPREAD_COLS] = 0.0
                if self.n_b1:
                    x[:, mod.b1_cols(n_b1=self.n_b1)] = 0.0
            if self.masks.drop_datetime:
                x[:, m.DATETIME_COLS] = 0.0
            if self.masks.drop_lead:
                x[:, m.LEAD_COLS] = 0.0
        return super().forward(x, primary)

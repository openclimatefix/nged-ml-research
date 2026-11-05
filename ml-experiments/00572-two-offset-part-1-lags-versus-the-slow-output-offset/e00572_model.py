# ruff: noqa: E501
"""Experiment 572: the lag network of experiment 560 with the D5 output offset, and D5 learnt in fine-tuning.

- `LagOffsetMdn` (arm v3+D5): experiment 560's `LagModularMdn` (mode v3) plus experiment 565's
  design 5 offset. The knot position `tau` is input column 45 (after the 37 lag-free columns and the
  8 lag columns). As in `OutputOffsetDrift`, pre-training learns every monthly knot, fine-tuning
  keeps the older knots as fixed buffers and trains the newest knot, and a position after the newest
  knot reads the newest knot (flat extrapolation).
- `OffsetFtOnly` (arm E2-ft): experiment 565's `OutputOffsetDrift` network whose every knot starts at
  zero in fine-tuning and is learnt there; the encoders are experiment 554's M0 pre-training, which
  has no offset.
"""

from typing import Final

import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00565_model_variants as var  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

TAU_COL_LAG: Final[int] = m560.N_INPUTS
"""Knot-position column of the lag network (45)."""
PENALTY_LAMBDA: Final[float] = var.PENALTY_LAMBDA


class LagOffsetMdn(m560.LagModularMdn):
    """Experiment 560's v3 network plus the design 5 offset on the three component means."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; the offset knots start at zero."""
        super().__init__(n_primaries=n_primaries, mode="v3", stage=stage)
        assert n_knots >= 2
        self.n_knots = n_knots
        if stage == "pre":
            self.knots = nn.Parameter(torch.zeros(n_primaries, n_knots, 1))
        else:
            self.register_buffer("knots_old", torch.zeros(n_primaries, n_knots - 1, 1))
            self.knot_new = nn.Parameter(torch.zeros(n_primaries, 1))

    def all_knots(self) -> torch.Tensor:
        """Return every knot, (Primaries, knots, 1)."""
        if self.stage == "pre":
            return self.knots
        return torch.cat([self.knots_old, self.knot_new[:, None, :]], dim=1)  # ty: ignore[invalid-argument-type]

    def penalty(self) -> torch.Tensor:
        """Return `PENALTY_LAMBDA` times the mean squared first difference of the knots."""
        k = self.all_knots()
        return PENALTY_LAMBDA * ((k[:, 1:] - k[:, :-1]) ** 2).mean()

    def offset(self, tau: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the interpolated offset of each row, (rows, 1); the position is clamped to the knots."""
        knots = self.all_knots()
        tau = tau.clamp(0.0, float(self.n_knots - 1))
        lo = tau.floor().long().clamp(max=self.n_knots - 2)
        frac = (tau - lo.to(tau.dtype))[:, None]
        return (1.0 - frac) * knots[primary, lo] + frac * knots[primary, lo + 1]

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs with the offset added to the component means."""
        assert primary is not None
        out = super().forward(x, primary)
        off = self.offset(x[:, TAU_COL_LAG], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)


def load_lag_pre_into_ft(*, model: LagOffsetMdn, state: dict[str, torch.Tensor]) -> None:
    """Load a stage-`pre` state dict into the stage-`ft` network; `mixer_ft` stays as built."""
    assert model.stage == "ft"
    knots = state["knots"]
    rest = {a: v for a, v in state.items() if a != "knots" and not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(rest, strict=False)
    assert all(
        a.startswith("mixer_ft.") or a in {"knots_old", "knot_new"} for a in missing.missing_keys
    ), missing
    assert not missing.unexpected_keys, missing
    with torch.no_grad():
        model.knots_old.copy_(knots[:, :-1])  # ty: ignore[call-non-callable]
        model.knot_new.copy_(knots[:, -1])


class OffsetFtOnly(var.OutputOffsetDrift):
    """Design 5 with every knot a fine-tuning parameter (zero at the start of each fit)."""

    def __init__(self, *, n_primaries: int, n_knots: int) -> None:
        """Build the fine-tuning network, then replace its knot buffers by one full parameter."""
        super().__init__(n_primaries=n_primaries, stage="ft", n_knots=n_knots)
        del self.knots_old, self.knot_new
        self.knots = nn.Parameter(torch.zeros(n_primaries, n_knots, 1))

    def all_knots(self) -> torch.Tensor:
        """Return every knot, (Primaries, knots, 1)."""
        return self.knots

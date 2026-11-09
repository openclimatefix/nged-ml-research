# ruff: noqa: D107, E501
"""Experiment 667, prevailing rows: the survival-hazard head of experiment 652, 14 daily hazards.

A copy of `e00652_head` with `N_DAYS = 14`, so that the survival probability covers leads to 14
days (the contract's longest valid time is 339 h after the network origin). Nothing else changes.
"""

from typing import Final

import e00647_head as h647  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_DAYS: Final[int] = 14
HAZARD_OFFSET: Final[float] = 3.0
"""The hazard is softplus(output - offset), about 0.05 per day at initialisation."""


def survival(
    *, hazard_logits: torch.Tensor, lead_h: torch.Tensor, age_days: torch.Tensor
) -> torch.Tensor:
    """Return P(shift still on at the valid time) from per-day hazard logits (B, 14) and the lead in hours."""
    haz = nn.functional.softplus(hazard_logits - HAZARD_OFFSET)
    days = torch.arange(N_DAYS, dtype=lead_h.dtype, device=lead_h.device)[None, :]
    share = torch.clamp(lead_h[:, None] / 24.0 - days, 0.0, 1.0)
    p = torch.exp(-(haz * share).sum(dim=1))
    return torch.where(age_days < h647.AGE_FLOOR_DAYS, torch.zeros_like(p), p)


class HazardHead(nn.Module):
    """Two hidden layers of 64 units; outputs (depth, spread, 14 daily hazards)."""

    def __init__(self, *, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2 + N_DAYS)
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)  # ty: ignore[call-non-callable]
            self.net[-1].bias.zero_()  # ty: ignore[call-non-callable]

    def forward(
        self, *, x: torch.Tensor, q: torch.Tensor, age: torch.Tensor, lead_h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (prevailing quantiles (B, 11), survival probability (B,))."""
        out = self.net(x)
        d = h647.DEPTH_SCALE * torch.tanh(out[:, 0])
        sigma = h647.SIGMA_FLOOR + h647.SIGMA_SCALE * nn.functional.softplus(out[:, 1])
        p = survival(hazard_logits=out[:, 2:], lead_h=lead_h, age_days=age)
        return h647.mixture_quantiles(q=q, d=d, sigma=sigma, p=p), p

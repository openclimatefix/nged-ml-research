# ruff: noqa: D103, E501
"""Experiment 647, stage 2: the departure heads (H1 survival mixture, H2 free quantile head) and the mixture quantiles.

K1's 11 stored quantiles define a piecewise-linear distribution (two tail knots are added at 1.5
times the outermost quantile spacing). The prevailing distribution of H1 is the mixture
`(1 - p) K1 + p K1 shifted by d`, with the depth `d` uncertain (a 3-point Gauss-Hermite spread of
standard deviation `sigma`), and `p` the survival probability of the departure given its age (a fixed
function of the age and the lead, from the stated duration prior). The mixture's quantiles are exact
(the mixture CDF is piecewise linear, so they are found by interpolating between the sorted
breakpoints) and differentiable in `d`, `sigma`, and `p`. H2 adds 11 free offsets to K1's quantiles and
sorts them. The head's loss is the mean pinball loss over the 11 levels.
"""

from typing import Final

import numpy as np
import torch
from torch import nn

TAUS: Final = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99], np.float32)
F_KNOTS: Final = np.concatenate([[0.0], TAUS, [1.0]]).astype(np.float32)
GH_NODES: Final = (-(3.0**0.5), 0.0, 3.0**0.5)
GH_WEIGHTS: Final = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
TAIL_SPACINGS: Final[float] = 1.5
SHORT_DAYS: Final = (3.0, 14.0)
LONG_DAYS: Final = (28.0, 112.0)
AGE_FLOOR_DAYS: Final[float] = 0.5
AGE_CAP_DAYS: Final[float] = 100.0
DEPTH_SCALE: Final[float] = 0.5
SIGMA_FLOOR: Final[float] = 0.01
SIGMA_SCALE: Final[float] = 0.05
OFFSET_SCALE: Final[float] = 0.1


def _uniform_survival(*, t: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Return P(duration > t) for a duration uniform on [lo, hi] days."""
    return torch.clamp((hi - t) / (hi - lo), 0.0, 1.0)


def survival_probability(*, age_days: torch.Tensor, lead_h: torch.Tensor, short_weight: float) -> torch.Tensor:
    """Return p: the probability that a departure of `age_days` at the newest lag still holds at the valid time.

    The valid time is `lead_h + 24` hours after the newest allowed lag, which sits 24 h before the
    origin. The duration prior is `short_weight` uniform on 3 to 14 days and the rest uniform on 28
    to 112 days.
    """

    def surv(t: torch.Tensor) -> torch.Tensor:
        return short_weight * _uniform_survival(t=t, lo=SHORT_DAYS[0], hi=SHORT_DAYS[1]) + (
            1.0 - short_weight
        ) * _uniform_survival(t=t, lo=LONG_DAYS[0], hi=LONG_DAYS[1])

    a = torch.clamp(age_days, AGE_FLOOR_DAYS, AGE_CAP_DAYS)
    h = (lead_h + 24.0) / 24.0
    return torch.clamp(surv(a + h) / torch.clamp(surv(a), min=1e-3), 0.0, 1.0)


def _knots(*, q: torch.Tensor) -> torch.Tensor:
    """Return the 13 sorted knots (two tail knots added) of each row's K1 distribution."""
    lo = q[:, 0] - TAIL_SPACINGS * torch.clamp(q[:, 1] - q[:, 0], min=1e-3)
    hi = q[:, -1] + TAIL_SPACINGS * torch.clamp(q[:, -1] - q[:, -2], min=1e-3)
    x = torch.cat([lo[:, None], q, hi[:, None]], dim=1)
    return x + torch.arange(x.shape[1], dtype=x.dtype)[None, :] * 2e-6


def _cdf(*, x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Return the piecewise-linear CDF of the distribution with `knots` (B, 13) at the points `x` (B, m)."""
    f = torch.from_numpy(F_KNOTS).to(x.device)
    idx = torch.clamp(torch.searchsorted(knots.detach().contiguous(), x.detach().contiguous(), right=True) - 1, 0, knots.shape[1] - 2)
    x0, x1 = torch.gather(knots, 1, idx), torch.gather(knots, 1, idx + 1)
    frac = torch.clamp((x - x0) / (x1 - x0), 0.0, 1.0)
    return f[idx] + frac * (f[idx + 1] - f[idx])


def mixture_quantiles(*, q: torch.Tensor, d: torch.Tensor, sigma: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return the 11 quantiles of `(1 - p) K1 + p (K1 shifted by d, spread by sigma)`, shape (B, 11)."""
    knots = _knots(q=q)
    shifts = torch.stack([torch.zeros_like(d)] + [d + g * sigma for g in GH_NODES], dim=1)  # (B, 4)
    weights = torch.stack([1.0 - p] + [p * w for w in GH_WEIGHTS], dim=1)  # (B, 4)
    pts = (knots[:, None, :] + shifts[:, :, None]).reshape(len(q), -1)  # (B, 52)
    xs, _ = torch.sort(pts, dim=1)
    fm = sum(weights[:, c : c + 1] * _cdf(x=xs - shifts[:, c : c + 1], knots=knots) for c in range(4))
    tau = torch.from_numpy(TAUS).to(q.device)[None, :].expand(len(q), -1).contiguous()
    j = torch.clamp(torch.searchsorted(fm.detach().contiguous(), tau, right=False), 1, xs.shape[1] - 1)
    f0, f1 = torch.gather(fm, 1, j - 1), torch.gather(fm, 1, j)
    x0, x1 = torch.gather(xs, 1, j - 1), torch.gather(xs, 1, j)
    frac = torch.clamp((tau - f0) / torch.clamp(f1 - f0, min=1e-9), 0.0, 1.0)
    return x0 + frac * (x1 - x0)


def pinball(*, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return the per-row mean pinball loss over the 11 levels."""
    tau = torch.from_numpy(TAUS).to(q.device)[None, :]
    e = y[:, None] - q
    return torch.maximum(tau * e, (tau - 1.0) * e).mean(dim=1)


class Head(nn.Module):
    """Two hidden layers of 64 units. H1 outputs (depth, spread); H2 outputs 11 quantile offsets."""

    def __init__(self, *, n_in: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        n_out = 2 if kind == "mix" else len(TAUS)
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, *, x: torch.Tensor, q: torch.Tensor, age: torch.Tensor, lead_h: torch.Tensor, short_weight: float) -> tuple[torch.Tensor, ...]:
        out = self.net(x)
        if self.kind == "mix":
            d = DEPTH_SCALE * torch.tanh(out[:, 0])
            sigma = SIGMA_FLOOR + SIGMA_SCALE * nn.functional.softplus(out[:, 1])
            p = survival_probability(age_days=age, lead_h=lead_h, short_weight=short_weight)
            return mixture_quantiles(q=q, d=d, sigma=sigma, p=p), d, sigma
        qq, _ = torch.sort(q + OFFSET_SCALE * out, dim=1)
        zero = torch.zeros(len(q))
        return qq, zero, zero

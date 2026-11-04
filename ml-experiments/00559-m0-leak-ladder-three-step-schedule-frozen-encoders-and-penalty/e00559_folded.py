# ruff: noqa: E501
"""Arm A8: a folded-normal mixture head for the MVA-only Primaries.

An MVA-only Primary's apparent power cannot be negative. Write `u = z + m` for the standardised target
`z` shifted by `m = target_mean / target_std` of the Primary, so that `u >= 0` is the load in units of
its standard deviation. The head's three components describe `u` (component mean = raw mean + `m`).
A folded component has density `N(u; mu, sigma) + N(u; -mu, sigma)` on `u >= 0`, and its CDF is
`Phi((u - mu) / sigma) - Phi((-u - mu) / sigma)`. The signed-MW Primaries keep the usual mixture, so a
row's loss and quantiles depend on its Primary's group.
"""

import math
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

LOG_SQRT_2PI: Final[float] = 0.5 * math.log(2 * math.pi)


def folded_nll(
    *,
    out: torch.Tensor,
    z: torch.Tensor,
    offset: torch.Tensor,
    folded: torch.Tensor,
    components: int,
) -> torch.Tensor:
    """Return each row's negative log-likelihood: folded for rows with `folded` True, usual otherwise.

    Args:
        out: Raw network outputs, (rows, 3 * components).
        z: Standardised targets, (rows,).
        offset: Each row's `m = target_mean / target_std`, (rows,).
        folded: Whether each row's Primary is MVA-only, (rows,), bool.
        components: Number of mixture components.
    """
    log_w, mean, std = base.mixture_parts(out=out, components=components)
    plain = (
        -torch.logsumexp(log_w - 0.5 * ((z[..., None] - mean) / std) ** 2 - torch.log(std), dim=-1)
        + LOG_SQRT_2PI
    )
    u = (z + offset)[..., None]
    mu = mean + offset[..., None]
    direct = -0.5 * ((u - mu) / std) ** 2
    mirror = -0.5 * ((u + mu) / std) ** 2
    log_pdf = torch.logaddexp(direct, mirror) - torch.log(std)
    fold = -torch.logsumexp(log_w + log_pdf, dim=-1) + LOG_SQRT_2PI
    return torch.where(folded, fold, plain)


def folded_quantiles(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """Return the 11 quantiles of `z` for folded rows (rows, 11), by bisection on the folded CDF."""
    levels = torch.tensor(base.LEVELS, dtype=mean.dtype, device=mean.device)
    w = log_w.exp()
    mu = (mean + offset[:, None])[:, None, :]
    sd = std[:, None, :]
    lo = torch.zeros((len(w), len(base.LEVELS)), dtype=mean.dtype, device=mean.device)
    hi = torch.full_like(lo, base.Z_LIMIT)
    root2 = math.sqrt(2.0)
    for _ in range(base.BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        u = mid[..., None]
        cdf = (
            w[:, None, :]
            * 0.5
            * (torch.erf((u - mu) / (sd * root2)) - torch.erf((-u - mu) / (sd * root2)))
        ).sum(-1)
        below = cdf < levels
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi) - offset[:, None]

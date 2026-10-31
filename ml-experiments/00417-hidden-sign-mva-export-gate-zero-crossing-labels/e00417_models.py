# ruff: noqa: E501
"""The likelihood of experiment 417: experiment 412's folded likelihood with a per-row mask.

`SignedMdn` and the mixture functions are experiment 412's. The negative log-likelihood changes in
two ways:

- **`open_mask`** says where the folded density applies. Experiment 412 opened it where the scaled
  CAMS unit shape is at least 0.3. Here it is also closed at every row of a Primary whose
  Primary-level gate is shut (design A) and at rows below the high-irradiance threshold (design H).
  A closed row has the density of the reading truncated to `x > 0`.
- **`label`** is a signed label (-1 reverse, +1 forward, 0 none) from the zero-crossing labeller
  (design C). A labelled row has the signed density at `label * y'` with `y' = sqrt(max(r^2 -
  Q0^2, 0))`, or the censored probability when `r <= Q0`, with no folding.
"""

from typing import Final

import e00412_models as m412  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

SignedMdn = m412.SignedMdn
LEVELS: Final[tuple[float, ...]] = m412.LEVELS
Q0_POINTS: Final[float] = m412.Q0_POINTS
GATE_U: Final[float] = m412.GATE_U
TINY: Final[float] = m412.TINY
mixture_cdf = m412.mixture_cdf
quantiles = m412.quantiles
abs_quantiles = m412.abs_quantiles


def nll(
    *,
    log_w: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    r_points: torch.Tensor,
    folded: torch.Tensor,
    open_mask: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    """Return each row's negative log-likelihood of the reading `r_points` (rows,)."""
    log_pdf = m412._log_pdf
    plain = log_pdf(x=r_points, log_w=log_w, mean=mean, std=std)
    q0 = Q0_POINTS
    above = r_points > q0
    y_prime = torch.sqrt(torch.clamp(r_points**2 - q0**2, min=0.0))
    forward = log_pdf(x=y_prime, log_w=log_w, mean=mean, std=std)
    backward = log_pdf(x=-y_prime, log_w=log_w, mean=mean, std=std)
    band = (
        log_w.exp()
        * (torch.special.ndtr((q0 - mean) / std) - torch.special.ndtr((-q0 - mean) / std))
    ).sum(-1)
    censored = torch.log(torch.clamp(band, min=TINY))
    open_ll = torch.where(above, torch.logaddexp(forward, backward), censored)
    log_positive = torch.logsumexp(log_w + torch.special.log_ndtr(mean / std), dim=-1)
    closed_ll = plain - log_positive
    folded_ll = torch.where(open_mask, open_ll, closed_ll)
    signed_ll = torch.where(above, torch.where(label < 0, backward, forward), censored)
    folded_ll = torch.where(label != 0, signed_ll, folded_ll)
    return -torch.where(folded, folded_ll, plain)


def signed_quantiles(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, open_mask: torch.Tensor
) -> torch.Tensor:
    """Return the 11 signed quantiles (rows, 11), truncated to `x > 0` where `open_mask` is false."""
    free = quantiles(log_w=log_w, mean=mean, std=std)
    truncated = quantiles(
        log_w=log_w,
        mean=mean,
        std=std,
        lower=torch.zeros(len(mean), device=mean.device, dtype=mean.dtype),
    )
    return torch.where(open_mask[:, None], free, truncated)


def p_negative(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, open_mask: torch.Tensor
) -> torch.Tensor:
    """Return `P(X < 0)`, zero where the row is closed."""
    zero = torch.zeros(len(mean), 1, device=mean.device, dtype=mean.dtype)
    p = mixture_cdf(x=zero, log_w=log_w, mean=mean, std=std)[:, 0]
    return torch.where(open_mask, p, torch.zeros_like(p))

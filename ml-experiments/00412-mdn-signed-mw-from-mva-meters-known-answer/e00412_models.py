# ruff: noqa: E501
"""The structural signed mixture density network of experiment 412 and its folded likelihood.

Signed power is `X = D - C_p u`. `D` is experiment 397's pooled plain mixture (3 Gaussian
components, a learned Primary embedding), `u` is the CAMS unit PV shape scaled to a 99th percentile
of 1, and `C_p >= 0` is a learned capacity per Primary in points of effective capacity (percent).
Everything the likelihood and the quantiles compute is in points: the folded likelihood and the
reactive floor `Q0` act on physical values, never on the standardised units of `D`.

Likelihood of a reading `r` (points):

- **Signed Primary:** the mixture density `f(r)`.
- **Folded Primary, gate open (`u_gate >= GATE_U`):** `y' = sqrt(max(r^2 - Q0^2, 0))`. For `r > Q0`
  the density is `f(y') + f(-y')`. For `r <= Q0` it is the probability `P(|X| <= Q0)`, because a
  reading at or below `Q0` bounds `|P|` by `Q0`. The Jacobian of the transform is left out, so the
  log-likelihood is not comparable across arms.
- **Folded Primary, gate closed:** `f(r)` truncated to `x > 0` and renormalised, so the mixture gains
  nothing by putting mass on negative `x` off the gate.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

Q0_POINTS: Final[float] = 2.0
"""The reactive floor of the likelihood and of the reading forecast, in points of capacity."""
GATE_U: Final[float] = 0.3
"""The gate: the folded likelihood applies where the scaled CAMS unit shape is at least this."""
PRIOR_LOG_SD: Final[float] = 0.7
LEVELS: Final[tuple[float, ...]] = base.LEVELS
BISECTION_STEPS: Final[int] = 40
X_LIMIT_POINTS: Final[float] = 4_000.0
TINY: Final[float] = 1e-12
LOG_SQRT_2PI: Final[float] = 0.5 * float(np.log(2 * np.pi))


class SignedMdn(nn.Module):
    """The demand mixture minus a learned capacity times the CAMS unit shape."""

    mean_points: torch.Tensor
    std_points: torch.Tensor
    folded: torch.Tensor
    prior_centre: torch.Tensor

    def __init__(
        self,
        *,
        n_inputs: int,
        n_primaries: int,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        folded: np.ndarray,
        capacity_start: np.ndarray,
        prior_centre_points: float,
        u_forced_zero: bool = False,
        components: int = 3,
        hidden: int = 64,
        layers: int = 2,
        embedding: bool = True,
    ) -> None:
        """Build the network.

        Args:
            n_inputs: Width of the demand module's input.
            n_primaries: Number of Primaries.
            target_mean: (P,) mean of the training target, a fraction of effective capacity.
            target_std: (P,) standard deviation of the training target, a fraction of capacity.
            folded: (P,) whether each Primary's target is folded.
            capacity_start: (P,) starting capacity in points of effective capacity.
            prior_centre_points: The centre of the shared log-normal prior, in points.
            u_forced_zero: Fix `C_p` at zero, so `u` has no effect (the identifiability check).
            components: Mixture components.
            hidden: Width of the demand module.
            layers: Hidden layers of the demand module.
            embedding: Whether the demand module sees the Primary embedding.
        """
        super().__init__()
        self.components, self.u_forced_zero = components, u_forced_zero
        self.demand = base.PlainMdn(
            n_inputs=n_inputs,
            hidden=hidden,
            layers=layers,
            components=components,
            n_primaries=n_primaries if embedding else 0,
        )
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self.register_buffer("mean_points", t(target_mean) * 100.0)
        self.register_buffer("std_points", t(target_std) * 100.0)
        self.register_buffer("folded", torch.as_tensor(np.asarray(folded), dtype=torch.bool))
        self.register_buffer("prior_centre", torch.tensor(float(prior_centre_points)))
        self.log_c = nn.Parameter(torch.log(t(capacity_start)), requires_grad=not u_forced_zero)

    def capacity(self) -> torch.Tensor:
        """Return the (P,) capacity in points (zero when `u` is forced to zero)."""
        c = torch.exp(self.log_c)
        return torch.zeros_like(c) if self.u_forced_zero else c

    def parts(
        self, x: torch.Tensor, primary: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the signed mixture as log weights, means (points), and standard deviations (points)."""
        log_w, mu, sd = base.mixture_parts(out=self.demand(x, primary), components=self.components)
        mean = self.mean_points[primary][:, None] + self.std_points[primary][:, None] * mu
        mean = mean - (self.capacity()[primary] * u)[:, None]
        return log_w, mean, self.std_points[primary][:, None] * sd

    def penalty(self) -> torch.Tensor:
        """Return the shared log-normal prior on the capacities, in nats."""
        if self.u_forced_zero:
            return torch.zeros((), device=self.log_c.device)
        return (0.5 * ((self.log_c - torch.log(self.prior_centre)) / PRIOR_LOG_SD) ** 2).sum()


def _log_pdf(
    *, x: torch.Tensor, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Return the log mixture density at `x` (rows,) for mixtures (rows, K)."""
    z = (x[:, None] - mean) / std
    return torch.logsumexp(log_w - 0.5 * z**2 - torch.log(std) - LOG_SQRT_2PI, dim=-1)


def mixture_cdf(
    *, x: torch.Tensor, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Return the mixture CDF at `x` of shape (rows, m) for mixtures (rows, K)."""
    z = (x[..., None] - mean[:, None, :]) / std[:, None, :]
    return (log_w.exp()[:, None, :] * torch.special.ndtr(z)).sum(-1)


def nll(
    *,
    log_w: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    r_points: torch.Tensor,
    folded: torch.Tensor,
    gate_u: torch.Tensor,
) -> torch.Tensor:
    """Return each row's negative log-likelihood of the reading `r_points` (rows,)."""
    plain = _log_pdf(x=r_points, log_w=log_w, mean=mean, std=std)
    open_gate = gate_u >= GATE_U
    q0 = Q0_POINTS
    above = r_points > q0
    y_prime = torch.sqrt(torch.clamp(r_points**2 - q0**2, min=0.0))
    folded_density = torch.logaddexp(
        _log_pdf(x=y_prime, log_w=log_w, mean=mean, std=std),
        _log_pdf(x=-y_prime, log_w=log_w, mean=mean, std=std),
    )
    band = (
        log_w.exp()
        * (torch.special.ndtr((q0 - mean) / std) - torch.special.ndtr((-q0 - mean) / std))
    ).sum(-1)
    censored = torch.log(torch.clamp(band, min=TINY))
    gate_open_ll = torch.where(above, folded_density, censored)
    log_positive = torch.logsumexp(log_w + torch.special.log_ndtr(mean / std), dim=-1)
    gate_closed_ll = plain - log_positive
    folded_ll = torch.where(open_gate, gate_open_ll, gate_closed_ll)
    return -torch.where(folded, folded_ll, plain)


def quantiles(
    *,
    log_w: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the 11 quantiles (rows, 11) in points, by bisection on the CDF.

    With `lower` (rows,) the mixture is truncated to `x >= lower` and renormalised (used for the
    gate-closed reading forecast, `lower = 0`).
    """
    levels = torch.tensor(LEVELS, dtype=mean.dtype, device=mean.device)
    if lower is None:
        base_cdf = torch.zeros(len(mean), 1, dtype=mean.dtype, device=mean.device)
        scale = torch.ones_like(base_cdf)
    else:
        base_cdf = mixture_cdf(x=lower[:, None], log_w=log_w, mean=mean, std=std)
        scale = 1.0 - base_cdf
    target = base_cdf + levels[None, :] * scale
    lo = torch.full((len(mean), len(LEVELS)), -X_LIMIT_POINTS, dtype=mean.dtype, device=mean.device)
    hi = torch.full_like(lo, X_LIMIT_POINTS)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        below = mixture_cdf(x=mid, log_w=log_w, mean=mean, std=std) < target
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def abs_quantiles(*, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Return the 11 quantiles (rows, 11) of `|X|` in points, by bisection on `F(a) - F(-a)`."""
    levels = torch.tensor(LEVELS, dtype=mean.dtype, device=mean.device)
    lo = torch.zeros((len(mean), len(LEVELS)), dtype=mean.dtype, device=mean.device)
    hi = torch.full_like(lo, X_LIMIT_POINTS)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        h = mixture_cdf(x=mid, log_w=log_w, mean=mean, std=std) - mixture_cdf(
            x=-mid, log_w=log_w, mean=mean, std=std
        )
        below = h < levels
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def reading_quantiles(
    *,
    log_w: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    gate_u: torch.Tensor,
) -> torch.Tensor:
    """Return the reading forecast (rows, 11) in points for a folded Primary.

    Gate open: `sqrt(q^2 + Q0^2)` for each quantile `q` of `|X|`. Gate closed: the quantiles of
    `X` truncated to `x > 0`.
    """
    folded = torch.sqrt(abs_quantiles(log_w=log_w, mean=mean, std=std) ** 2 + Q0_POINTS**2)
    truncated = quantiles(
        log_w=log_w,
        mean=mean,
        std=std,
        lower=torch.zeros(len(mean), device=mean.device, dtype=mean.dtype),
    )
    return torch.where((gate_u >= GATE_U)[:, None], folded, truncated)

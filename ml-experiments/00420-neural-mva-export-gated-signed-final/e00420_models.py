# ruff: noqa: E501
"""The signed mixture density network of experiment 420: experiment 417's network plus a variance term.

`SignedMdn`, the mixture functions, the folded likelihood (`nll`), and the signed quantiles are
experiment 417's and experiment 412's. This module adds:

- **`SignedMdnAv`** (arm AV): each mixture component's standard deviation `std` becomes
  `sqrt(std^2 + (s * C_p * u)^2)`, where `C_p` is the Primary's learned capacity in points of
  effective capacity, `u` is the scaled CAMS unit PV shape, and `s >= 0` is one scalar shared by all
  16 Primaries. A site with a large PV term gets a wider signed distribution where `u` is large.
- **`reading_quantiles`**: the reading forecast with a per-row open mask, used for the MVA-only
  eight, where the reading (not the signed power) is the truth.
"""

from typing import Final

import e00412_models as m412  # ty: ignore[unresolved-import]
import e00417_models as m417  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

LEVELS: Final[tuple[float, ...]] = m417.LEVELS
Q0_POINTS: Final[float] = m417.Q0_POINTS
GATE_U: Final[float] = m417.GATE_U
nll = m417.nll
signed_quantiles = m417.signed_quantiles
p_negative = m417.p_negative
quantiles = m417.quantiles
abs_quantiles = m417.abs_quantiles
S_START: Final[float] = 0.1
"""The starting value of the shared variance scale `s`."""


class SignedMdnAv(m412.SignedMdn):
    """Experiment 412's signed network with a variance that grows with the PV term."""

    def __init__(self, *, variance_scale_start: float = S_START, **kwargs) -> None:  # noqa: ANN003
        """Build the network; every other argument is `SignedMdn`'s."""
        super().__init__(**kwargs)
        start = torch.tensor(float(variance_scale_start))
        self.raw_s = nn.Parameter(torch.log(torch.expm1(start)))

    def variance_scale(self) -> torch.Tensor:
        """Return `s >= 0`, the shared scale of the PV-proportional standard deviation."""
        return torch.nn.functional.softplus(self.raw_s)

    def parts(
        self, x: torch.Tensor, primary: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log weights, means, and standard deviations (points) with the PV variance added."""
        log_w, mean, std = super().parts(x, primary, u)
        extra = self.variance_scale() * self.capacity()[primary] * u
        return log_w, mean, torch.sqrt(std**2 + extra[:, None] ** 2)


def reading_quantiles(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, open_mask: torch.Tensor
) -> torch.Tensor:
    """Return the reading forecast (rows, 11) in points for a folded Primary.

    Open rows: `sqrt(q^2 + Q0^2)` for each quantile `q` of `|X|`. Closed rows: the quantiles of `X`
    truncated to `x > 0`.
    """
    folded = torch.sqrt(abs_quantiles(log_w=log_w, mean=mean, std=std) ** 2 + Q0_POINTS**2)
    truncated = quantiles(
        log_w=log_w,
        mean=mean,
        std=std,
        lower=torch.zeros(len(mean), device=mean.device, dtype=mean.dtype),
    )
    return torch.where(open_mask[:, None], folded, truncated)

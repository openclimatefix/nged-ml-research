"""Output families for the tail arms of experiment 407.

Three ways to turn a network's raw outputs into 11 quantiles of the standardised power `z`:

- `gauss`: a mixture of Gaussians (experiment 397's family; `K` components of 3 outputs each);
- `split`: a mixture of split-normal components, each with its own standard deviation below and
  above its mode (4 outputs per component), so a component can be skewed;
- `quantile`: 11 monotone quantiles trained directly with the pinball loss (no density).
"""

from typing import Final

import e00397_models as models  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

LEVELS: Final[tuple[float, ...]] = models.LEVELS
FAMILIES: Final[tuple[str, ...]] = ("gauss", "split", "quantile")
MEDIAN_INDEX: Final[int] = 5
N_UP: Final[int] = 5
"""Levels above the median: 0.6, 0.75, 0.9, 0.95, 0.99."""


def outputs_per_component(*, family: str) -> int:
    """Return the number of raw outputs per mixture component."""
    return 4 if family == "split" else 3


def n_outputs(*, family: str, components: int) -> int:
    """Return the number of raw outputs of a network."""
    if family == "quantile":
        return len(LEVELS)
    return outputs_per_component(family=family) * components


def split_parts(
    *, out: torch.Tensor, components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split raw outputs (..., 4 K) into log weights, modes, and lower and upper deviations."""
    k = components
    return (
        torch.log_softmax(out[..., :k], dim=-1),
        out[..., k : 2 * k],
        torch.exp(out[..., 2 * k : 3 * k].clamp(-3.0, 3.0)),
        torch.exp(out[..., 3 * k :].clamp(-3.0, 3.0)),
    )


def split_nll(*, out: torch.Tensor, z: torch.Tensor, components: int) -> torch.Tensor:
    """Return the negative log-likelihood of `z` under a split-normal mixture, per row."""
    log_w, mode, sd_low, sd_high = split_parts(out=out, components=components)
    diff = z[..., None] - mode
    sd = torch.where(diff < 0, sd_low, sd_high)
    log_pdf = (
        float(np.log(2.0 / np.sqrt(2.0 * np.pi)))
        - torch.log(sd_low + sd_high)
        - 0.5 * (diff / sd) ** 2
    )
    return -torch.logsumexp(log_w + log_pdf, dim=-1)


def split_quantiles(*, out: torch.Tensor, components: int) -> torch.Tensor:
    """Return the 11 quantiles (rows, 11) of split-normal mixtures, by bisection on the CDF."""
    log_w, mode, sd_low, sd_high = split_parts(out=out.double(), components=components)
    w = log_w.exp()
    levels = torch.tensor(LEVELS, dtype=mode.dtype, device=mode.device)
    lo = torch.full((len(w), len(LEVELS)), -models.Z_LIMIT, dtype=mode.dtype, device=mode.device)
    hi = torch.full_like(lo, models.Z_LIMIT)
    total = sd_low + sd_high
    root2 = float(np.sqrt(2.0))
    for _ in range(models.BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        diff = mid[..., None] - mode[:, None, :]
        below = (
            2.0
            * sd_low[:, None, :]
            / total[:, None, :]
            * 0.5
            * (1.0 + torch.erf(diff / (sd_low[:, None, :] * root2)))
        )
        above = sd_low[:, None, :] / total[:, None, :] + 2.0 * sd_high[:, None, :] / total[
            :, None, :
        ] * 0.5 * torch.erf(diff / (sd_high[:, None, :] * root2))
        cdf = (w[:, None, :] * torch.where(diff < 0, below, above)).sum(-1)
        is_below = cdf < levels
        lo, hi = torch.where(is_below, mid, lo), torch.where(is_below, hi, mid)
    return 0.5 * (lo + hi)


def monotone_quantiles(*, out: torch.Tensor) -> torch.Tensor:
    """Return 11 ascending quantiles from raw outputs (rows, 11).

    Output 5 is the median; outputs 6 to 10 are positive steps up from it (softplus, cumulated);
    outputs 0 to 4 are positive steps down from it, ordered from the 0.4 level to the 0.01 level.
    """
    median = out[..., MEDIAN_INDEX : MEDIAN_INDEX + 1]
    up = median + torch.cumsum(nn.functional.softplus(out[..., MEDIAN_INDEX + 1 :]), dim=-1)
    down = median - torch.cumsum(nn.functional.softplus(out[..., :MEDIAN_INDEX]), dim=-1)
    return torch.cat([down.flip(-1), median, up], dim=-1)


def pinball_loss(*, out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over the 11 levels, per row."""
    q = monotone_quantiles(out=out)
    levels = torch.tensor(LEVELS, dtype=q.dtype, device=q.device)
    err = z[..., None] - q
    return torch.maximum(levels * err, (levels - 1.0) * err).mean(dim=-1)


def loss(*, family: str, components: int, out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return the per-row training loss of a family."""
    if family == "quantile":
        return pinball_loss(out=out, z=z)
    if family == "split":
        return split_nll(out=out, z=z, components=components)
    return models.mixture_nll(out=out, z=z, components=components)


def quantiles(*, family: str, components: int, out: torch.Tensor) -> torch.Tensor:
    """Return the 11 standardised quantiles (rows, 11) of raw outputs."""
    if family == "quantile":
        return monotone_quantiles(out=out.double())
    if family == "split":
        return split_quantiles(out=out, components=components)
    log_w, mean, std = models.mixture_parts(out=out.double(), components=components)
    return models.mixture_quantiles(log_w=log_w, mean=mean, std=std)


def init_output_layer(*, head: nn.Sequential, family: str, components: int) -> None:
    """Zero the output layer's weights and set a bias that starts the network at a spread guess."""
    last = head[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    if family == "quantile":
        bias = [0.0] * len(LEVELS)
        bias[MEDIAN_INDEX] = 0.0
        steps = [-0.69] * N_UP
        bias[MEDIAN_INDEX + 1 :] = steps
        bias[:MEDIAN_INDEX] = steps
    elif family == "split":
        offsets = np.linspace(-0.5, 0.5, components) if components > 1 else np.zeros(1)
        bias = [0.0] * components + [float(o) for o in offsets] + [0.0] * (2 * components)
    else:
        bias = models.mixture_head_bias(components=components)
    with torch.no_grad():
        last.bias.copy_(torch.tensor(bias))


class PooledNetwork(nn.Module):
    """A pooled multilayer perceptron with a Primary embedding and a family-specific output."""

    def __init__(
        self,
        *,
        n_inputs: int,
        family: str,
        components: int = 3,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        n_primaries: int = 16,
        embedding: int = 8,
    ) -> None:
        """Build the network; the output layer starts at zero weights."""
        super().__init__()
        self.family, self.components = family, components
        self.embedding = nn.Embedding(n_primaries, embedding)
        self.net = models.mlp(
            n_in=n_inputs + embedding,
            hidden=hidden,
            layers=layers,
            n_out=n_outputs(family=family, components=components),
            dropout=dropout,
        )
        init_output_layer(head=self.net, family=family, components=components)

    def forward(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the raw outputs."""
        return self.net(torch.cat([x, self.embedding(primary)], dim=-1))

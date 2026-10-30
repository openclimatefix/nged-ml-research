# ruff: noqa: E501
"""The pooled plain mixture density network with an explicit solar term (405).

The demand module is experiment 397's `PlainMdn` (pooled, Primary embedding, 3 components). The
solar module subtracts a solar output with a mean and a spread from the demand mixture: component
`k` of the net-demand mixture has mean `mu_k - m` and variance `sigma_k^2 + s^2` (independent
errors). All quantities are in the standardised units `z = (power / capacity - mean) / std` of the
demand module.
"""

from dataclasses import dataclass
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_ADAPTER_INPUTS: Final[int] = 9
ADAPTER_HIDDEN: Final[int] = 16
ADAPTER_LOG_RANGE: Final[float] = 0.5
"""The adapter multiplier is `exp(0.5 * tanh(a))`, between 0.61 and 1.65."""
RISE_SD: Final[float] = 0.25
FALL_SD: Final[float] = 0.25 / float(np.sqrt(10.0))
"""Persistence penalty: a fall costs 10 times as much as a rise of the same size."""
MONTH_SMOOTH_SD: Final[float] = 0.05
INCREMENT_SD: Final[float] = 0.1
"""Experiment 129's increment penalty, as a relative step between neighbouring cells."""

ARMS: Final[tuple[str, ...]] = ("A0", "A0-noirr", "C1", "C2", "C3", "C4", "C5a", "C5b", "C2-inc")


@dataclass(frozen=True)
class ArmSpec:
    """What an arm changes."""

    solar: bool
    adapter: bool
    learn_capacity: bool
    prior: bool
    cells: str
    penalty: str
    irradiance_free: bool


SPECS: Final[dict[str, ArmSpec]] = {
    "A0": ArmSpec(False, False, False, False, "year", "none", False),
    "A0-noirr": ArmSpec(False, False, False, False, "year", "none", True),
    "C1": ArmSpec(True, False, False, False, "year", "none", True),
    "C2": ArmSpec(True, False, True, True, "year", "none", True),
    "C3": ArmSpec(True, True, True, True, "year", "none", True),
    "C4": ArmSpec(True, True, True, False, "year", "none", True),
    "C5a": ArmSpec(True, True, True, True, "month", "smooth", True),
    "C5b": ArmSpec(True, True, True, True, "year", "persist", True),
    "C2-inc": ArmSpec(True, False, True, True, "year", "increment", True),
}


class PvMdn(nn.Module):
    """The demand mixture minus a physical solar term with learned capacity."""

    def __init__(
        self,
        *,
        spec: ArmSpec,
        n_inputs: int,
        n_primaries: int,
        solar_mask: np.ndarray,
        capacity_mw: np.ndarray,
        prior_centre: np.ndarray,
        prior_sd: np.ndarray,
        scale: np.ndarray,
        components: int = 3,
        hidden: int = 64,
        layers: int = 2,
    ) -> None:
        """Build the network.

        Args:
            spec: The arm.
            n_inputs: Demand-module input width.
            n_primaries: Number of Primaries.
            solar_mask: (P,) True where the solar term runs.
            capacity_mw: (P, T) starting (or pinned) capacity in MW; a dummy 1 where masked.
            prior_centre: (P, T) prior centre in MW.
            prior_sd: (P, T) prior standard deviation in MW.
            scale: (P,) 1 / (effective capacity in MW * target standard deviation).
            components: Mixture components.
            hidden: Demand-module width.
            layers: Demand-module hidden layers.
        """
        super().__init__()
        self.spec, self.components = spec, components
        self.demand = base.PlainMdn(
            n_inputs=n_inputs,
            hidden=hidden,
            layers=layers,
            components=components,
            n_primaries=n_primaries,
        )
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        self.register_buffer("solar_mask", t(solar_mask))
        self.register_buffer("scale", t(scale))
        self.register_buffer("prior_centre", t(prior_centre))
        self.register_buffer("prior_sd", t(prior_sd))
        log_c = torch.log(t(capacity_mw))
        if spec.learn_capacity:
            self.log_c = nn.Parameter(log_c)
        else:
            self.register_buffer("log_c", log_c)
        self.adapter = None
        if spec.adapter:
            self.adapter = base.mlp(n_in=N_ADAPTER_INPUTS, hidden=ADAPTER_HIDDEN, layers=1, n_out=1)
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def capacity(self) -> torch.Tensor:
        """Return the (P, T) capacity in MW."""
        return torch.exp(self.log_c)

    def parts(
        self, x: torch.Tensor, primary: torch.Tensor, e: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the net-demand mixture as log weights, means, and standard deviations."""
        log_w, mean, std = base.mixture_parts(out=self.demand(x, primary), components=self.components)
        if not self.spec.solar:
            return log_w, mean, std
        u, u_sd = e[:, 0], e[:, 1]
        if self.adapter is not None:
            g = torch.exp(ADAPTER_LOG_RANGE * torch.tanh(self.adapter(e[:, 2:]).squeeze(-1)))
        else:
            g = torch.ones_like(u)
        c = self.capacity()[primary, cell]
        k = self.scale[primary] * self.solar_mask[primary] * c
        m = k * torch.clamp(u * g, 0.0, 1.0)
        s = k * torch.clamp(u_sd * g, 0.0, 1.0)
        return log_w, mean - m[:, None], torch.sqrt(std**2 + (s**2)[:, None])

    def penalty(self) -> torch.Tensor:
        """Return the capacity penalties in nats (zero for an arm without them)."""
        spec = self.spec
        zero = torch.zeros((), device=self.log_c.device)
        if not (spec.solar and spec.learn_capacity):
            return zero
        c = self.capacity()
        mask = self.solar_mask[:, None]
        total = zero
        if spec.prior:
            total = total + (0.5 * mask * ((c - self.prior_centre) / self.prior_sd) ** 2).sum()
        if spec.penalty == "none":
            return total
        step = (c[:, 1:] - c[:, :-1]) / c[:, :-1]
        if spec.penalty == "persist":
            total = total + (
                0.5 * mask * (torch.relu(step) / RISE_SD) ** 2
                + 0.5 * mask * (torch.relu(-step) / FALL_SD) ** 2
            ).sum()
        elif spec.penalty == "smooth":
            total = total + (0.5 * mask * (step / MONTH_SMOOTH_SD) ** 2).sum()
        elif spec.penalty == "increment":
            total = total + (0.5 * mask * (step / INCREMENT_SD) ** 2).sum()
        return total


def nll_parts(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Return the negative log-likelihood of `z` under mixtures given as parts."""
    log_pdf = -0.5 * ((z[..., None] - mean) / std) ** 2 - torch.log(std)
    return -torch.logsumexp(log_w + log_pdf, dim=-1) + 0.5 * float(np.log(2 * np.pi))

# ruff: noqa: D102, D107, E501
"""The shared pooled mixture-density network of experiment 430, with one switch per ablation.

With every switch off, `UnifiedNet` is experiment 310's `MixtureHead` (inputs, two hidden layers
of 64, three Gaussians on the capacity-normalised output) with the 345 column list, which is what
the reproduction gate checks. The switches:

- `member_route`: a small shared encoder runs on each of the 51 members' shortwave, beam, diffuse
  and temperature (and the plant output when the plant is on); the 51 encodings are pooled by
  mean, standard deviation, 10th and 90th percentile and joined to the one-pass inputs, so the
  member route sees everything the one-pass network sees.
- `plant`: a PyTorch fleet plant (`e00430_plant`) adds a power channel, per member on the member
  route and on the member mean otherwise. Frozen keeps its scale and clip fixed; end to end also
  learns tilt and panel azimuth.
- `atoms`: the head adds point masses at zero and at the farm's output ceiling, and the Gaussians
  are truncated to the open interval between them.
"""

from dataclasses import dataclass
from typing import Final

import e00430_common as common
import e00430_plant as plant_module
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

COMPONENTS: Final[int] = 3
ENCODER_WIDTH: Final[int] = 16
ENCODER_HIDDEN: Final[int] = 32
LOG_SQRT_2PI: Final[float] = float(0.5 * np.log(2 * np.pi))
PLANT_INPUT_SCALE: Final[float] = 3.0
"""The plant output is a fraction of capacity (0 to about 1); this gives the input roughly unit size."""
LEVELS: Final = torch.tensor(
    [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99], dtype=torch.float64
)


@dataclass
class Batch:
    """One batch: standardised static inputs, sun geometry, member weather, and the target."""

    x: torch.Tensor
    geo: torch.Tensor
    """Columns: zenith (degrees), sun azimuth (degrees), day of year."""
    members: torch.Tensor
    """Shape (rows, members or 1, 2): shortwave (W/m2) and temperature (degC). One member is repeated."""
    y: torch.Tensor
    ceiling: torch.Tensor


class UnifiedNet(nn.Module):
    """The pooled mixture-density network with the switches of `common.Config`."""

    def __init__(
        self, *, n_static: int, config: common.Config, plant: plant_module.Plant | None
    ) -> None:
        super().__init__()
        self.config = config
        self.plant = plant
        n_in = n_static
        if config.member_route:
            channels = 4 + (1 if plant is not None else 0)
            self.encoder = nn.Sequential(
                nn.Linear(channels, ENCODER_HIDDEN),
                nn.GELU(),
                nn.Linear(ENCODER_HIDDEN, ENCODER_WIDTH),
                nn.GELU(),
            )
            n_in += 4 * ENCODER_WIDTH
        elif plant is not None:
            n_in += 1
        n_out = 3 * COMPONENTS + (2 if config.atoms else 0)
        self.trunk = nn.Sequential(
            nn.Linear(n_in, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, n_out)
        )
        last = self.trunk[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        bias = [0.0] * 3 + [-0.1, 0.0, 0.1] + [float(np.log(0.08))] * 3
        if config.atoms:
            bias += [float(np.log(0.03)), float(np.log(0.005))]
        with torch.no_grad():
            last.bias.copy_(torch.tensor(bias))

    def _member_inputs(self, batch: Batch) -> torch.Tensor:
        """Return the (rows, members, channels) tensor the encoder reads."""
        ghi, temp = batch.members[..., 0], batch.members[..., 1]
        zenith, azimuth, doy = (batch.geo[:, i : i + 1] for i in range(3))
        dni, dhi = plant_module.erbs(ghi=ghi, zenith_deg=zenith, doy=doy)
        scale = common.MEMBER_CHANNEL_SCALE
        channels = [ghi / scale[0], dni / scale[1], dhi / scale[2], (temp - 10.0) / scale[3]]
        if self.plant is not None:
            power = self.plant(
                ghi=ghi, temperature_c=temp, zenith_deg=zenith, azimuth_deg=azimuth, doy=doy
            )
            channels.append(power * PLANT_INPUT_SCALE)
        return torch.stack(channels, dim=2)

    def _trunk_input(self, batch: Batch) -> torch.Tensor:
        parts = [batch.x]
        if self.config.member_route:
            encoded = self.encoder(self._member_inputs(batch))
            low_high = torch.quantile(
                encoded, torch.tensor([0.1, 0.9], device=encoded.device), dim=1
            )
            parts += [
                encoded.mean(dim=1),
                encoded.std(dim=1, unbiased=False),
                low_high[0],
                low_high[1],
            ]
        elif self.plant is not None:
            mean = batch.members.mean(dim=1)
            power = self.plant(
                ghi=mean[:, 0],
                temperature_c=mean[:, 1],
                zenith_deg=batch.geo[:, 0],
                azimuth_deg=batch.geo[:, 1],
                doy=batch.geo[:, 2],
            )
            parts.append(power[:, None] * PLANT_INPUT_SCALE)
        return torch.cat(parts, dim=1)

    def parameters_of(
        self, batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return log weights, means, standard deviations, and (atoms) log [continuous, zero, ceiling]."""
        out = self.trunk(self._trunk_input(batch))
        log_w = torch.log_softmax(out[:, :COMPONENTS], dim=1)
        mean = out[:, COMPONENTS : 2 * COMPONENTS]
        std = torch.exp(out[:, 2 * COMPONENTS : 3 * COMPONENTS].clamp(-6.0, 1.0))
        atoms = None
        if self.config.atoms:
            zeros = torch.zeros_like(out[:, :1])
            atoms = torch.log_softmax(torch.cat([zeros, out[:, 3 * COMPONENTS :]], dim=1), dim=1)
        return log_w, mean, std, atoms

    def nll(self, batch: Batch) -> torch.Tensor:
        return self.nll_per_row(batch).mean()

    def nll_per_row(self, batch: Batch) -> torch.Tensor:
        log_w, mean, std, atoms = self.parameters_of(batch)
        y = batch.y[:, None]
        log_pdf = -0.5 * ((y - mean) / std) ** 2 - torch.log(std) - LOG_SQRT_2PI
        log_mix = torch.logsumexp(log_w + log_pdf, dim=1)
        if atoms is None:
            return -log_mix
        c = batch.ceiling[:, None]
        mass = (
            log_w.exp() * (torch.special.ndtr((c - mean) / std) - torch.special.ndtr(-mean / std))
        ).sum(dim=1)
        log_density = atoms[:, 0] + log_mix - torch.log(mass.clamp(min=1e-6))
        value = batch.y
        log_lik = torch.where(
            value <= 0.0,
            atoms[:, 1],
            torch.where(value >= batch.ceiling, atoms[:, 2], log_density),
        )
        return -log_lik


@torch.no_grad()
def quantiles(*, model: UnifiedNet, batch: Batch) -> torch.Tensor:
    """Return the 11 quantiles per row by bisection on the mixture CDF, in [0, ceiling]."""
    levels = LEVELS.to(batch.x.device)
    log_w, mean, std, atoms = model.parameters_of(batch)
    w, mean, std = log_w.exp().double(), mean.double(), std.double()
    c = batch.ceiling.double()
    if atoms is None:
        lo = torch.full((len(w), levels.numel()), -2.0, dtype=torch.float64, device=w.device)
        hi = torch.full_like(lo, 2.0)
        target = levels[None, :].expand_as(lo)
        base = torch.zeros_like(lo)
        scale = torch.ones_like(lo)
    else:
        p0, p1 = atoms[:, 1].exp().double()[:, None], atoms[:, 2].exp().double()[:, None]
        lower = (w * torch.special.ndtr(-mean / std)).sum(dim=1, keepdim=True)
        upper = (w * torch.special.ndtr((c[:, None] - mean) / std)).sum(dim=1, keepdim=True)
        scale = (upper - lower).clamp(min=1e-9)
        base = lower
        target = (levels[None, :] - p0) / (1.0 - p0 - p1).clamp(min=1e-9)
        lo = torch.zeros((len(w), levels.numel()), dtype=torch.float64, device=w.device)
        hi = c[:, None].expand_as(lo).clone()
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        z = (mid[:, :, None] - mean[:, None, :]) / std[:, None, :]
        cdf = ((w[:, None, :] * torch.special.ndtr(z)).sum(dim=2) - base) / scale
        below = cdf < target
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    result = 0.5 * (lo + hi)
    if atoms is not None:
        result = torch.where(levels[None, :] <= p0, torch.zeros_like(result), result)
        result = torch.where(levels[None, :] >= 1.0 - p1, c[:, None].expand_as(result), result)
    return torch.minimum(result.clamp(min=0.0), c[:, None])

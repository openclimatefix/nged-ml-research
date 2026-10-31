# ruff: noqa: D102, D107
"""A differentiable PyTorch copy of experiment 310's fleet-shape plant, with learnable parameters.

`erbs` copies `pvlib.irradiance.erbs`, `unit_response` copies the plane-of-array and derating
steps of `e00310_members.unit_response`, and `Plant` adds the four parameters the end-to-end arm
learns (tilt, panel azimuth, scale, clip). `tests/test_e00430_plant.py` checks each against the
NumPy original on random inputs.
"""

import math
from typing import Final

import torch  # ty: ignore[unresolved-import]
from studies.pv_physics import GROUND_ALBEDO
from torch import nn  # ty: ignore[unresolved-import]

SOLAR_CONSTANT: Final[float] = 1366.1
MIN_COS_ZENITH: Final[float] = 0.065
MAX_ZENITH_DEG: Final[float] = 87.0
HEAT_PER_KELVIN: Final[float] = 0.004
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0


def erbs(
    *, ghi: torch.Tensor, zenith_deg: torch.Tensor, doy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return beam (DNI) and diffuse (DHI) irradiance from shortwave, as `pvlib.irradiance.erbs`."""
    b = 2.0 * math.pi * (doy - 1.0) / 365.0
    extra = SOLAR_CONSTANT * (
        1.00011
        + 0.034221 * torch.cos(b)
        + 0.00128 * torch.sin(b)
        + 0.000719 * torch.cos(2 * b)
        + 0.000077 * torch.sin(2 * b)
    )
    cos_zenith = torch.cos(torch.deg2rad(zenith_deg))
    kt = (ghi / (extra * torch.clamp(cos_zenith, min=MIN_COS_ZENITH))).clamp(0.0, 1.0)
    diffuse_fraction = torch.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        torch.where(
            kt <= 0.8,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            torch.full_like(kt, 0.165),
        ),
    )
    dhi = diffuse_fraction * ghi
    dni = (ghi - dhi) / torch.clamp(cos_zenith, min=1e-6)
    bad = (zenith_deg > MAX_ZENITH_DEG) | (ghi < 0) | (dni < 0)
    dni = torch.where(bad, torch.zeros_like(dni), dni)
    dhi = torch.where(bad, ghi, dhi)
    return dni, dhi


def unit_response(
    *,
    ghi: torch.Tensor,
    temperature_c: torch.Tensor,
    zenith_deg: torch.Tensor,
    azimuth_deg: torch.Tensor,
    doy: torch.Tensor,
    tilt_deg: torch.Tensor | float,
    panel_azimuth_deg: torch.Tensor | float,
) -> torch.Tensor:
    """Return the derated plane-of-array output per unit of capacity (before any scale or clip)."""
    dni, dhi = erbs(ghi=ghi, zenith_deg=zenith_deg, doy=doy)
    zenith = torch.deg2rad(zenith_deg)
    sun_azimuth = torch.deg2rad(azimuth_deg)
    tilt = torch.deg2rad(torch.as_tensor(tilt_deg, dtype=ghi.dtype, device=ghi.device))
    panel_azimuth = torch.deg2rad(
        torch.as_tensor(panel_azimuth_deg, dtype=ghi.dtype, device=ghi.device)
    )
    cos_incidence = torch.cos(zenith) * torch.cos(tilt) + torch.sin(zenith) * torch.sin(
        tilt
    ) * torch.cos(sun_azimuth - panel_azimuth)
    beam = dni * torch.clamp(cos_incidence, min=0.0)
    sky_diffuse = dhi * (1.0 + torch.cos(tilt)) / 2.0
    ground = ghi * GROUND_ALBEDO * (1.0 - torch.cos(tilt)) / 2.0
    unit = torch.clamp(beam + sky_diffuse + ground, min=0.0) / 1000.0
    cell = temperature_c + CELL_TEMPERATURE_RISE_K * unit
    derating = torch.clamp(1.0 - HEAT_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), min=0.0)
    return unit * derating


class Plant(nn.Module):
    """The fleet plant as a module: output = min(scale x unit response, clip), in capacity units."""

    def __init__(
        self, *, tilt_deg: float, azimuth_deg: float, scale: float, clip: float, learn: bool
    ) -> None:
        super().__init__()
        self.learn = learn
        tilt = torch.tensor(float(tilt_deg))
        azimuth = torch.tensor(float(azimuth_deg))
        log_scale = torch.tensor(math.log(max(scale, 1e-3)))
        clip_t = torch.tensor(float(clip))
        if learn:
            self.tilt, self.azimuth = nn.Parameter(tilt), nn.Parameter(azimuth)
            self.log_scale, self.clip = nn.Parameter(log_scale), nn.Parameter(clip_t)
        else:
            self.register_buffer("tilt", tilt)
            self.register_buffer("azimuth", azimuth)
            self.register_buffer("log_scale", log_scale)
            self.register_buffer("clip", clip_t)

    def forward(
        self,
        *,
        ghi: torch.Tensor,
        temperature_c: torch.Tensor,
        zenith_deg: torch.Tensor,
        azimuth_deg: torch.Tensor,
        doy: torch.Tensor,
    ) -> torch.Tensor:
        unit = unit_response(
            ghi=ghi,
            temperature_c=temperature_c,
            zenith_deg=zenith_deg,
            azimuth_deg=azimuth_deg,
            doy=doy,
            tilt_deg=self.tilt,
            panel_azimuth_deg=self.azimuth,
        )
        return torch.minimum(self.log_scale.exp() * unit, self.clip)

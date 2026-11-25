# ruff: noqa: E501
"""Experiment 721: the arms, the sign head, and the folded loss.

`SgNet` is experiment 716's `PvNet` of arm `HT` (history and target-row PV paths) plus one sign
head. The head reads the history token after the axial blocks and gives `ρ`, the probability that
the meter's reading at that half-hour was an export. The formula is
`ρ = e_i * gate(u) * sigmoid(w . LayerNorm(x) + b)`, where `e_i` is the per-Primary export gate,
`gate(u)` is 1 where the shortwave likelihood `u` is at least `u_min` and 0 elsewhere, and the bias
`b` starts at -4. The read-out then uses the signed-expected history `(1 - 2ρ) v` and adds the sign
variance `4 ρ (1 - ρ) v²` to its spread, where `v` is the folded reading in units of the Primary's
standard deviation. `SG-0` has `ρ` fixed at 0.

The folded loss is the negative log of the density of the reading `r = sqrt(X² + Q²)` under the
signed mixture for `X`. It sums both signs of `X`, multiplies by the Jacobian `r / |X|`, and has a
censored branch for readings at the floor, which takes the mixture mass inside `|X| <= Q`.
"""

from dataclasses import dataclass, replace
from typing import Final

import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00716_model as pvm  # ty: ignore[unresolved-import]
import e00721_core as core
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

ARM_SPECS: Final = core.ARM_SPECS
ArmSpec = core.ArmSpec

POWER_CLIP: Final = 5.0
"""History is clipped to plus or minus 5 standard deviations before the sign step."""
SKY_WEATHER_COLUMNS: Final = (0, 15, 16, 17, 18, 19, 20)
"""Weather columns that carry the sky: the shortwave mean and the six B1 solar columns."""
SKY_DATETIME_COLUMNS: Final = (0, 1, 2)
"""Datetime columns that carry the sun's position."""
SKY_SPREAD_COLUMNS: Final = (0,)
"""Spread columns that carry the shortwave spread."""
CENSOR_TOLERANCE: Final = 1e-4
"""A reading within this relative margin of the floor `Q` is censored."""
SAFE_X: Final = 1e-6


@dataclass
class SgInputs(pvm.PvInputs):
    """716's batch plus `hist_u`, the shortwave likelihood of each history token.

    `hist_u` is 0 where the history is signed.
    """

    hist_u: torch.Tensor  # (B, 14, 48)

    def to(self, device: str | torch.device) -> SgInputs:
        """Return the batch on `device`."""
        return SgInputs(**{k: v.to(device) for k, v in self.__dict__.items()})

    def with_power(self, *, power: torch.Tensor, visible: torch.Tensor) -> SgInputs:
        """Return the batch with another history power and visibility."""
        return replace(self, hist_power=power, hist_visible=visible)

    def with_pv(
        self, *, hist_pv: torch.Tensor | None = None, hist_pv_ok: torch.Tensor | None = None,
        top_pv: torch.Tensor | None = None, top_pv_ok: torch.Tensor | None = None,
    ) -> SgInputs:  # fmt: skip
        """Return the batch with some of the PV tensors replaced."""
        changes = {
            k: v
            for k, v in (
                ("hist_pv", hist_pv), ("hist_pv_ok", hist_pv_ok), ("top_pv", top_pv), ("top_pv_ok", top_pv_ok)
            )
            if v is not None
        }  # fmt: skip
        return replace(self, **changes)


def blank_sky(*, inp: SgInputs) -> SgInputs:
    """Return the batch with every sky column zeroed.

    The sky-blind network of `SG-0`'s gate check.
    """
    top_w, hist_w = inp.top_weather.clone(), inp.hist_weather.clone()
    top_d, hist_d = inp.top_datetime.clone(), inp.hist_datetime.clone()
    top_s = inp.top_spread.clone()
    top_w[..., list(SKY_WEATHER_COLUMNS)] = 0.0
    hist_w[..., list(SKY_WEATHER_COLUMNS)] = 0.0
    top_d[..., list(SKY_DATETIME_COLUMNS)] = 0.0
    hist_d[..., list(SKY_DATETIME_COLUMNS)] = 0.0
    top_s[..., list(SKY_SPREAD_COLUMNS)] = 0.0
    return replace(
        inp,
        top_weather=top_w,
        hist_weather=hist_w,
        top_datetime=top_d,
        hist_datetime=hist_d,
        top_spread=top_s,
    )


def signed_expected_history(
    *, z: torch.Tensor, rho: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the signed-expected history in z units and the sign variance in v units squared.

    Args:
        z: Folded history in standardised units, shape `(B, 14, 48)`, already clipped.
        rho: Probability of an export at each history token, same shape.
        shift: `(B,)` the Primary's mean over standard deviation, so that `v = z + shift`.

    Returns:
        `z_exp = (1 - 2 rho) v - shift` and `4 rho (1 - rho) v²`.
    """
    v = z + shift[:, None, None]
    return (1.0 - 2.0 * rho) * v - shift[:, None, None], 4.0 * rho * (1.0 - rho) * v**2


class SgNet(pvm.PvNet):
    """`PvNet` of arm `HT` with the sign head."""

    def __init__(self, *, seed: int = 0, sign_head: bool = True) -> None:
        """Build the network with the sign head's weights at zero and its bias at -4.

        The head's `ρ` therefore starts near 0.018.
        """
        super().__init__(arm="HT", seed=seed)
        self.sign_head = sign_head
        self.sign_lin = nn.Linear(mdl.WIDTH, 1)
        nn.init.zeros_(self.sign_lin.weight)
        nn.init.constant_(self.sign_lin.bias, core.HEAD_BIAS_START)
        self.register_buffer("export_gate", torch.ones(mdl.N_PRIMARIES))
        self.register_buffer("v_shift", torch.zeros(mdl.N_PRIMARIES))
        self.last_rho: torch.Tensor | None = None

    def set_gate(self, *, gate: np.ndarray, force_open: bool = False) -> None:
        """Set the per-Primary export gate `e_i` (all ones when forced open)."""
        values = np.ones(mdl.N_PRIMARIES) if force_open else np.asarray(gate, dtype=np.float64)
        self.export_gate.copy_(torch.as_tensor(values, dtype=self.export_gate.dtype))

    def set_shift(self, *, shift: np.ndarray) -> None:
        """Set each Primary's mean over standard deviation (`v = z + shift`)."""
        self.v_shift.copy_(
            torch.as_tensor(np.asarray(shift, dtype=np.float64), dtype=self.v_shift.dtype)
        )

    def history_rho(self, *, h: torch.Tensor, inp: SgInputs) -> torch.Tensor:
        """Return `ρ` for each history token, shape `(B, 14, 48)`.

        From the normalised history tokens `h`.
        """
        if not self.sign_head:
            return torch.zeros_like(inp.hist_u)
        logit = self.sign_lin(h).squeeze(-1)
        gate_u = (inp.hist_u >= core.U_MIN).to(logit.dtype)
        return self.export_gate[inp.primary][:, None, None] * gate_u * torch.sigmoid(logit)

    def _readout(  # ty: ignore[invalid-method-override]
        self, x: torch.Tensor, inp: SgInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `h0`, the signed-expected `base`, the `spread` with the sign variance, and the weights."""
        h0, _, _, no_history, alpha = super()._readout(x, inp)
        if self.switches.top_row_only:
            return h0, torch.zeros_like(h0[..., 0]), torch.zeros_like(h0[..., 0]), no_history, alpha
        h = self.ln_readout(x)[:, 1:]
        rho = self.history_rho(h=h, inp=inp)
        self.last_rho = rho.detach()
        z = inp.hist_power.clamp(-POWER_CLIP, POWER_CLIP)
        z_exp, sign_var = signed_expected_history(z=z, rho=rho, shift=self.v_shift[inp.primary])
        base = (alpha * z_exp).sum(dim=1)
        var = (alpha * (z_exp - base[:, None, :]) ** 2).sum(dim=1) + (alpha * sign_var).sum(dim=1)
        any_visible = no_history < 0.5
        spread = torch.where(any_visible, torch.sqrt(var + 1e-6), torch.zeros_like(var))
        return h0, base, spread, no_history, alpha


# --------------------------------------------------------------------------- the folded loss


def _normal_cdf(*, z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / float(np.sqrt(2.0))))


def folded_nll(
    *, out: torch.Tensor, reading_z: torch.Tensor, shift: torch.Tensor, q_v: torch.Tensor
) -> torch.Tensor:
    """Return the folded negative log-likelihood of each target token, shape `(B, 48)`.

    Args:
        out: Raw mixture outputs `(B, 48, 9)`.
        reading_z: The folded reading in standardised units, `(B, 48)`.
        shift: `(B,)` mean over standard deviation of each sample's Primary.
        q_v: `(B,)` the reactive floor `Q` in standard deviations of the Primary.

    Returns:
        The negative log-likelihood of each target token, `(B, 48)`.
    """
    log_w, mean, std = mdl.base_models.mixture_parts(out=out, components=mdl.m559.COMPONENTS)
    mean_v = mean + shift[:, None, None]
    r = reading_z + shift[:, None]
    q = q_v[:, None]
    censored = r <= q * (1.0 + CENSOR_TOLERANCE)
    x = torch.sqrt(torch.clamp(r**2 - q**2, min=SAFE_X**2))
    log_norm = -torch.log(std) - 0.5 * float(np.log(2 * np.pi))
    pos = -0.5 * ((x[..., None] - mean_v) / std) ** 2
    neg = -0.5 * ((-x[..., None] + 0.0 - mean_v) / std) ** 2
    log_pdf = torch.logsumexp(torch.stack([pos, neg], dim=-1), dim=-1) + log_norm
    log_density = torch.logsumexp(log_w + log_pdf, dim=-1) + torch.log(r.clamp_min(SAFE_X) / x)
    mass = (
        log_w.exp()
        * (
            _normal_cdf(z=(q[..., None] - mean_v) / std)
            - _normal_cdf(z=(-q[..., None] - mean_v) / std)
        )
    ).sum(-1)
    log_mass = torch.log(mass.clamp_min(1e-12))
    return -torch.where(censored, log_mass, log_density)


def folded_quantiles(
    *,
    out: torch.Tensor,
    shift: torch.Tensor,
    q_v: torch.Tensor,
    steps: int = mdl.base_models.BISECTION_STEPS,
) -> torch.Tensor:
    """Return the 11 quantiles `(B, 48, 11)` of the folded reading, in standardised units.

    The reading `r = sqrt(X² + Q²)` is at most `c` exactly when `|X| <= sqrt(c² - Q²)`, so each
    quantile is
    found by bisection on `c` with the mixture's two-sided mass.
    """
    shape = out.shape[:-1]
    log_w, mean, std = mdl.base_models.mixture_parts(
        out=out.reshape(-1, out.shape[-1]), components=mdl.m559.COMPONENTS
    )
    mean = mean + shift.repeat_interleave(shape[1])[:, None]
    q = q_v.repeat_interleave(shape[1])[:, None]
    levels = torch.tensor(mdl.base_models.LEVELS, dtype=mean.dtype, device=mean.device)
    w = log_w.exp()
    lo = q.expand(-1, len(levels)).clone()
    hi = torch.full_like(lo, mdl.base_models.Z_LIMIT)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        x = torch.sqrt(torch.clamp(mid**2 - q**2, min=0.0))
        mass = (
            w[:, None, :]
            * (_normal_cdf(z=(x[..., None] - mean[:, None, :]) / std[:, None, :])
               - _normal_cdf(z=(-x[..., None] - mean[:, None, :]) / std[:, None, :]))
        ).sum(-1)  # fmt: skip
        below = mass < levels
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    r = 0.5 * (lo + hi) - shift.repeat_interleave(shape[1])[:, None]
    return r.view(*shape, len(levels))

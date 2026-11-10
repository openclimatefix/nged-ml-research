# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, E501, PT018
"""Experiment 684: replace the Gaussian mixture of experiment 397's head with a positive family.

Importing this module patches `mixture_parts`, `mixture_nll`, and `mixture_quantiles` of
`e00397_models`, which every stage of experiment 667's chain reaches through the module attribute
(`base.mixture_nll`, `base.mixture_quantiles`). The patched functions need the Primary of each row
and its target mean and standard deviation, which the original signatures do not carry, so:

- the `forward` of the four network classes records the batch's Primary index in `STATE.primary`;
- `Rows.__init__` of `e00428_data` (the fold's rows, `mr.base`) records the fold's per-Primary
  target mean and standard deviation in `STATE`.

`E00684_FAMILY` is `gauss` (nothing is patched), `tgauss` (the Gaussian mixture conditioned on
power at or above zero), or `gamma` (three gamma components on power over the Primary's standard
deviation). `E00684_POSITIVE_IDS` lists the `time_series_id`s that use the family; every other
Primary stays Gaussian. Row-wise, a boolean mask on the Primary index selects the family.

In z space, `z = (power - mean) / std`, so power zero sits at `z0 = -mean / std`. For gamma rows,
`w = z - z0 = power / std` is the gamma variable. The head's raw outputs are read as
`mean_k = exp(raw_mean_k) * (-z0)` (the standardised mean level of the Primary is `-z0`) and
`cv_k = 0.3 * exp(raw_cv_k)`, which start the components near the Primary's mean level with a
coefficient of variation near 0.3.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

FAMILY: Final = os.environ.get("E00684_FAMILY", "gauss")
assert FAMILY in {"gauss", "tgauss", "gamma"}, FAMILY
POSITIVE_IDS: Final = tuple(
    int(v) for v in os.environ.get("E00684_POSITIVE_IDS", "14,15,26,17").split(",")
)
W_FLOOR_POWER: Final = 1e-4
"""Smallest power, as a share of capacity, in the gamma density (an observed zero is raised to it)."""
CV_BASE: Final = 0.3
BISECTION_STEPS: Final = 50


@dataclass
class State:
    """What the patched functions need beyond their arguments."""

    primary: torch.Tensor | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    cache: dict = field(default_factory=dict)


STATE: Final = State()
_orig_parts, _orig_nll, _orig_quantiles = (
    base.mixture_parts,
    base.mixture_nll,
    base.mixture_quantiles,
)


def positive_primaries() -> np.ndarray:
    """Return the Primary indices of `POSITIVE_IDS`."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sm = k578.sites()
    sel = sm.filter(sm["time_series_id"].is_in(list(POSITIVE_IDS)))
    assert sel.height == len(POSITIVE_IDS), (POSITIVE_IDS, sel)
    return np.sort(sel["primary"].to_numpy())


def row_info(*, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row `(is_positive, z0)` for the batch whose Primary the last forward recorded."""
    p = STATE.primary
    assert p is not None and STATE.mean is not None and STATE.std is not None, "no batch recorded"
    key = (str(device), dtype)
    if key not in STATE.cache or STATE.cache[key][0] is not STATE.mean:
        positive = np.zeros(len(STATE.mean), dtype=bool)
        positive[positive_primaries()] = True
        z0 = -STATE.mean / STATE.std
        STATE.cache[key] = (
            STATE.mean,
            torch.tensor(positive, device=device),
            torch.tensor(z0, device=device, dtype=dtype),
        )
    _, positive_t, z0_t = STATE.cache[key]
    return positive_t[p], z0_t[p]


def floor_w(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return each row's floor of `power / std`: `W_FLOOR_POWER` over the Primary's std (capacity units)."""
    assert STATE.std is not None and STATE.primary is not None
    std = torch.tensor(STATE.std, dtype=dtype, device=device)[STATE.primary]
    return W_FLOOR_POWER / std


def gamma_parts(
    *, out: torch.Tensor, z0: torch.Tensor, components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return log weights, component means (in `w` units), and coefficients of variation."""
    k = components
    log_w = torch.log_softmax(out[..., :k], dim=-1)
    w_scale = (-z0).clamp_min(0.05)[..., None]
    mean = torch.exp(out[..., k : 2 * k].clamp(-6.0, 6.0)) * w_scale
    cv = CV_BASE * torch.exp(out[..., 2 * k : 3 * k].clamp(-3.0, 2.0))
    return log_w, mean, cv


def gamma_log_pdf(*, w: torch.Tensor, mean: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
    """Return the gamma log density of `w` for components with the given mean and CV."""
    shape = 1.0 / cv**2
    rate = shape / mean
    return shape * torch.log(rate) - torch.lgamma(shape) + (shape - 1.0) * torch.log(w) - rate * w


def gauss_log_pdf(*, z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Return the Gaussian log density of `z` for components with the given mean and std."""
    return -0.5 * ((z - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)


def mixture_nll(*, out: torch.Tensor, z: torch.Tensor, components: int) -> torch.Tensor:
    """Return each row's negative log likelihood under its Primary's family."""
    positive, z0 = row_info(device=out.device, dtype=out.dtype)
    log_w, mean, std = _orig_parts(out=out, components=components)
    gaussian = -torch.logsumexp(log_w + gauss_log_pdf(z=z[..., None], mean=mean, std=std), dim=-1)
    if FAMILY == "tgauss":
        z_obs = torch.maximum(z, z0)[..., None]
        log_mass = torch.logsumexp(
            log_w + torch.special.log_ndtr((mean - z0[..., None]) / std), dim=-1
        )
        family = (
            -torch.logsumexp(log_w + gauss_log_pdf(z=z_obs, mean=mean, std=std), dim=-1) + log_mass
        )
    else:
        g_log_w, g_mean, g_cv = gamma_parts(out=out, z0=z0, components=components)
        w = torch.maximum(z - z0, floor_w(dtype=out.dtype, device=out.device))[..., None]
        family = -torch.logsumexp(g_log_w + gamma_log_pdf(w=w, mean=g_mean, cv=g_cv), dim=-1)
    return torch.where(positive, family, gaussian)


def mixture_parts(
    *, out: torch.Tensor, components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Gaussian parts, with gamma parts (mean, CV) in place of them for the family's rows."""
    log_w, mean, std = _orig_parts(out=out, components=components)
    if FAMILY != "gamma":
        return log_w, mean, std
    positive, z0 = row_info(device=out.device, dtype=out.dtype)
    g_log_w, g_mean, g_cv = gamma_parts(out=out, z0=z0, components=components)
    flag = positive[..., None]
    return (
        torch.where(flag, g_log_w, log_w),
        torch.where(flag, g_mean, mean),
        torch.where(flag, g_cv, std),
    )


def mixture_quantiles(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Return the 11 quantiles (rows, 11) in z space, by bisection on each row's cumulative distribution."""
    positive, z0 = row_info(device=mean.device, dtype=mean.dtype)
    levels = torch.tensor(base.LEVELS, dtype=mean.dtype, device=mean.device)[None, :]
    w = log_w.exp()
    sqrt2 = math.sqrt(2.0)

    def gauss_cdf(x: torch.Tensor) -> torch.Tensor:
        z = (x[..., None] - mean[:, None, :]) / (std[:, None, :] * sqrt2)
        return (w[:, None, :] * 0.5 * (1.0 + torch.erf(z))).sum(-1)

    pos = positive[:, None]
    lo = torch.full((len(w), levels.shape[1]), -base.Z_LIMIT, dtype=mean.dtype, device=mean.device)
    hi = torch.full_like(lo, base.Z_LIMIT)
    lo = torch.where(pos, z0[:, None].expand_as(lo), lo)
    if FAMILY == "tgauss":
        f0 = gauss_cdf(z0[:, None])
        target = torch.where(pos, f0 + levels * (1.0 - f0), levels.expand_as(lo))
        cdf_of = gauss_cdf
    else:
        shape = 1.0 / std**2
        rate = shape / mean.clamp_min(1e-6)

        def gamma_cdf(x: torch.Tensor) -> torch.Tensor:
            xw = (x - z0[:, None]).clamp_min(0.0)[..., None]
            parts = torch.special.gammainc(shape[:, None, :], rate[:, None, :] * xw)
            return (w[:, None, :] * parts).sum(-1)

        def cdf_of(x: torch.Tensor) -> torch.Tensor:
            return torch.where(pos, gamma_cdf(x), gauss_cdf(x))

        target = levels.expand_as(lo)
        upper = z0 + 60.0 * torch.maximum(-z0, torch.ones_like(z0))
        hi = torch.where(pos, upper[:, None].expand_as(lo), hi)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        below = cdf_of(mid) < target
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def record_primary(forward):
    """Return `forward` wrapped so that it records the batch's Primary index in `STATE`."""

    def wrapped(self, x, primary=None, *args, **kwargs):
        STATE.primary = primary
        return forward(self, x, primary, *args, **kwargs)

    return wrapped


def install() -> None:
    """Patch the mixture functions, the networks' `forward`, and the fold's target statistics."""
    if FAMILY == "gauss":
        return
    base.mixture_nll, base.mixture_parts, base.mixture_quantiles = (
        mixture_nll,
        mixture_parts,
        mixture_quantiles,
    )
    import e00545_model as m545  # ty: ignore[unresolved-import]
    import e00560_model as m560  # ty: ignore[unresolved-import]
    import e00591_model as m591  # ty: ignore[unresolved-import]
    import e00613_nbmodel as m613  # ty: ignore[unresolved-import]

    for cls in (m545.ModularMdn, m560.LagModularMdn, m591.AuxLagMdn, m613.PermLagMdn):
        if "forward" in cls.__dict__:
            cls.forward = record_primary(cls.__dict__["forward"])
    init = data.Rows.__init__

    def record_stats(self, *args, **kwargs):
        init(self, *args, **kwargs)
        mean, std = np.asarray(self.target_mean), np.asarray(self.target_std)
        if STATE.mean is not None:
            assert np.array_equal(STATE.mean, mean) and np.array_equal(STATE.std, std), (  # ty: ignore[invalid-argument-type]
                "a second Rows with other target statistics in one process"
            )
        STATE.mean, STATE.std = mean, std

    data.Rows.__init__ = record_stats


install()

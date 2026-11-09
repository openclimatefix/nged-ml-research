"""Node inputs of the pretrained encoder of experiment 675, built from a masked hourly residual.

Every statistic is computed from the readings that are present (not masked, not missing) and
carries a coverage channel. A masked hour is set to NaN *before* any statistic is computed, so no
channel at any hour can depend on a hidden value; `e00675_leaktests.py` checks this bit for bit.

Channels per Primary and hour (`CHANNELS`):

- the residual in noise units clipped at 30, scaled by 0.2, and a present flag (2);
- for d in 24, 72, and 168 hours: the mean residual over the past [t - d, t) and over the future
  [t + 1, t + d], the gap of the hour's residual to each mean, and each window's coverage (18);
- for w in 6 and 24 hours: the forward step (mean of [t, t + w) minus mean of [t - w, t)) and the
  backward step (mean of [t - w, t) minus mean of [t - 2w, t - w)) with their coverages (8);
- the share of days of the sample with a daily step of 4 noise units or more (1);
- the noise unit as log megawatts, a megawatt-or-megavolt-ampere flag, and hour of day and weekday
  as sine and cosine (6).

The expected-power q05 to q95 width of the specification is not a channel: the convex demand model
stores no quantiles. Past statistics use only the past, so the causal arm zeroes the channels that
use the future (`future_channels`).
"""

from __future__ import annotations

import math
from typing import Final

import torch

CLIP: Final[float] = 30.0
SCALE: Final[float] = 0.2
GAP_WINDOWS: Final[tuple[int, ...]] = (24, 72, 168)
STEP_WINDOWS: Final[tuple[int, ...]] = (6, 24)
N_CHANNELS: Final[int] = 2 + 6 * len(GAP_WINDOWS) + 4 * len(STEP_WINDOWS) + 1 + 6
D_CHANNELS: Final[int] = 7
"""Channels of the daily stream."""


def future_channels() -> list[int]:
    """Indices of the channels that use hours after t (the causal arm zeroes them)."""
    out = []
    for k in range(len(GAP_WINDOWS)):
        base = 2 + 6 * k
        out += [base + 1, base + 3, base + 5]  # future mean, future gap, future coverage
    base = 2 + 6 * len(GAP_WINDOWS)
    for k in range(len(STEP_WINDOWS)):
        out += [base + 4 * k, base + 4 * k + 2]  # forward step and its coverage
    return out


def _window(cs: torch.Tensor, cn: torch.Tensor, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and coverage over [t + lo, t + hi) for every t (zero mean where nothing is present)."""
    n = cs.shape[-1] - 1
    t = torch.arange(n, device=cs.device)
    a = (t + lo).clamp(0, n)
    b = (t + hi).clamp(0, n)
    s = cs[..., b] - cs[..., a]
    c = cn[..., b] - cn[..., a]
    mean = torch.where(c > 0, s / c.clamp(min=1), torch.zeros_like(s))
    return mean, c / (hi - lo)


def node_inputs(
    *,
    x: torch.Tensor,
    sigma: torch.Tensor,
    is_mva: torch.Tensor,
    hour0: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hourly and daily inputs of a (masked) residual.

    Args:
        x: The residual in noise units, shape (nodes, hours), NaN where missing or masked. The
            length is a multiple of 24 and `hour0` is a multiple of 24.
        sigma: Noise unit in megawatts per node, shape (nodes,).
        is_mva: 1.0 for a node reported in megavolt-amperes, shape (nodes,).
        hour0: Hours from the grid start (2022-01-01 00:00) to the first column.

    Returns:
        The hourly inputs, shape (nodes, hours, `N_CHANNELS`), and the daily inputs, shape
        (nodes, days, `D_CHANNELS`).
    """
    n_nodes, n_hours = x.shape
    dtype = x.dtype
    present = torch.isfinite(x)
    r = torch.where(present, x.clamp(-CLIP, CLIP), torch.zeros_like(x))
    pad = torch.zeros(n_nodes, 1, dtype=dtype, device=x.device)
    cs = torch.cat([pad, r.cumsum(-1)], -1)
    cn = torch.cat([pad, present.to(dtype).cumsum(-1)], -1)
    ch = [r * SCALE, present.to(dtype)]
    for d in GAP_WINDOWS:
        for lo, hi in ((-d, 0), (1, d + 1)):
            m, cov = _window(cs, cn, lo, hi)
            ch += [m * SCALE, torch.where(present, (r - m) * SCALE, torch.zeros_like(r)), cov]
    for w in STEP_WINDOWS:
        m_f, c_f = _window(cs, cn, 0, w)
        m_p, c_p = _window(cs, cn, -w, 0)
        m_pp, c_pp = _window(cs, cn, -2 * w, -w)
        ok_f, ok_b = (c_f > 0) & (c_p > 0), (c_p > 0) & (c_pp > 0)
        ch += [
            torch.where(ok_f, m_f - m_p, torch.zeros_like(r)) * SCALE,
            torch.minimum(c_f, c_p),
            torch.where(ok_b, m_p - m_pp, torch.zeros_like(r)) * SCALE,
            torch.minimum(c_p, c_pp),
        ]
    # daily means, coverage, and the share of days with a daily step of 4 units or more
    days = n_hours // 24
    rd = r.reshape(n_nodes, days, 24)
    cnt = present.reshape(n_nodes, days, 24).to(dtype).sum(-1)
    dmean = torch.where(cnt > 0, rd.sum(-1) / cnt.clamp(min=1), torch.zeros_like(cnt))
    dcov = cnt / 24.0
    ok = (dcov[:, 1:] > 0.5) & (dcov[:, :-1] > 0.5)
    dstep = torch.zeros_like(dmean)
    dstep[:, 1:] = torch.where(ok, dmean[:, 1:] - dmean[:, :-1], torch.zeros_like(cnt[:, 1:]))
    big = (dstep.abs() >= 4.0) & (dcov > 0.5)
    valid = (dcov > 0.5).to(dtype).sum(-1).clamp(min=1)
    rate = (big.to(dtype).sum(-1) / valid)[:, None].expand(n_nodes, n_hours)
    ch.append(rate)
    t = torch.arange(n_hours, device=x.device, dtype=dtype) + hour0
    hod = 2 * math.pi * (t % 24) / 24
    dow = 2 * math.pi * (((t // 24) + 5) % 7) / 7
    static = [
        torch.log(sigma)[:, None].expand(n_nodes, n_hours).to(dtype),
        is_mva[:, None].expand(n_nodes, n_hours).to(dtype),
        torch.sin(hod)[None].expand(n_nodes, n_hours),
        torch.cos(hod)[None].expand(n_nodes, n_hours),
        torch.sin(dow)[None].expand(n_nodes, n_hours),
        torch.cos(dow)[None].expand(n_nodes, n_hours),
    ]
    hourly = torch.stack([*ch, *static], -1)
    assert hourly.shape[-1] == N_CHANNELS
    p7 = _past_future_days(dmean, dcov)
    daily = torch.stack(
        [
            dmean * SCALE,
            dcov,
            dstep * SCALE,
            p7[0] * SCALE,
            p7[1] * SCALE,
            torch.log(sigma)[:, None].expand(n_nodes, days).to(dtype),
            is_mva[:, None].expand(n_nodes, days).to(dtype),
        ],
        -1,
    )
    return hourly, daily


def _past_future_days(dmean: torch.Tensor, dcov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of the daily means over the 7 days before and the 7 days after each day."""
    n_nodes, days = dmean.shape
    w = dmean * dcov
    pad = torch.zeros(n_nodes, 1, dtype=dmean.dtype, device=dmean.device)
    cs = torch.cat([pad, w.cumsum(-1)], -1)
    cn = torch.cat([pad, dcov.cumsum(-1)], -1)
    out = []
    for lo, hi in ((-7, 0), (1, 8)):
        m, _ = _window(cs, cn, lo, hi)
        out.append(m)
    return out[0], out[1]

"""The edge-state neural network of experiment 676.

The node encoder is experiment 675's bidirectional encoder from a random initialisation. For every
link (a, b) of the graph and every hour, the link head scores three states, off, a donates to b,
and b donates to a, and the recipient lag (0 to 18 hours in 1-hour steps) given the donating
orientation. The link features are the two Primaries' encoder outputs, the cross-correlation of the
two Primaries' 6-hour forward steps at 19 recipient lags, the amplitude ratio, and the two
megawatt-or-megavolt-ampere flags. A node head scores no step, a downward step, and an upward step.
"""

from __future__ import annotations

from typing import Final

import torch  # ty: ignore[unresolved-import]
from e00675_net import Encoder
from torch import nn  # ty: ignore[unresolved-import]

N_LAGS: Final[int] = 19
STEP_CHANNEL: Final[int] = 20
"""Index of the 6-hour forward step in `e00675_inputs.node_inputs` (two plus three gap windows)."""
FEAT: Final[int] = 64 * 2 + N_LAGS + 1 + 2


class EdgeStateNet(nn.Module):
    """Encoder plus link head plus node head."""

    def __init__(self, *, edges: list[tuple[int, int]], no_links: bool = False) -> None:
        """Build the network; `edges` index the Primaries, `no_links` builds the G-0 arm."""
        super().__init__()
        self.register_buffer("ia", torch.tensor([a for a, _ in edges]))
        self.register_buffer("ib", torch.tensor([b for _, b in edges]))
        self.no_links = no_links
        self.encoder = Encoder(causal=False)
        self.don = nn.Sequential(nn.Linear(FEAT, 64), nn.GELU(), nn.Linear(64, 1 + N_LAGS))
        self.off = nn.Sequential(nn.Linear(FEAT, 64), nn.GELU(), nn.Linear(64, 1))
        self.node = nn.Linear(64, 3)

    def link_features(
        self, *, h: torch.Tensor, s: torch.Tensor, mva: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Oriented features of every link: (a, b) and (b, a), each (batch, links, hours, FEAT)."""
        ia, ib = self.ia, self.ib
        hi, hj, si, sj = h[:, ia], h[:, ib], s[:, ia], s[:, ib]

        def xcorr(donor: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
            shifted = torch.stack(
                [nn.functional.pad(rec[..., k:], (0, k)) for k in range(N_LAGS)], -1
            )
            return torch.asinh(25.0 * -donor[..., None] * shifted)

        def ratio(donor: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
            peak = (
                torch.stack([nn.functional.pad(rec[..., k:], (0, k)) for k in range(N_LAGS)], -1)
                .abs()
                .amax(-1)
            )
            return torch.asinh(5.0 * peak / (donor.abs() + 0.02))[..., None]

        t = h.shape[2]
        mi = mva[ia][None, :, None, None].expand(h.shape[0], -1, t, 1)
        mj = mva[ib][None, :, None, None].expand(h.shape[0], -1, t, 1)
        f_ab = torch.cat([hi, hj, xcorr(si, sj), ratio(si, sj), mi, mj], -1)
        f_ba = torch.cat([hj, hi, xcorr(sj, si), ratio(sj, si), mj, mi], -1)
        if self.no_links:  # only the donating Primary's own output remains
            keep = torch.zeros_like(f_ab)
            keep[..., :64] = 1.0
            f_ab, f_ba = f_ab * keep, f_ba * keep
        return f_ab, f_ba

    def forward(
        self, hourly: torch.Tensor, daily: torch.Tensor, adj: torch.Tensor, mva: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Link logits (b, links, hours, 3), lag logits (b, links, hours, 2, 19), node logits."""
        if self.no_links:
            adj = torch.zeros_like(adj)
        h, _ = self.encoder(hourly, daily, adj)  # (b, nodes, hours, 64)
        s = hourly[..., STEP_CHANNEL]
        f_ab, f_ba = self.link_features(h=h, s=s, mva=mva)
        a, b = self.don(f_ab), self.don(f_ba)
        off = self.off(f_ab) + self.off(f_ba)
        link = torch.cat([off, a[..., :1], b[..., :1]], -1)
        lag = torch.stack([a[..., 1:], b[..., 1:]], -2)
        return link, lag, self.node(h)

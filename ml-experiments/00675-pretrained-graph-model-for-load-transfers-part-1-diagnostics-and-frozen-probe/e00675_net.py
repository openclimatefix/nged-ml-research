"""The bidirectional temporal convolutional encoder of experiment 675 and its pretraining heads.

Two streams share one encoder per Primary: an hourly stream of 6 dilated layers with 64 channels
(receptive field about 15 days each side) and a daily stream of 4 dilated layers with 32 channels.
The daily features are projected, repeated 24 times, and added to the hourly features. One round of
neighbour averaging over the connection graph follows. The causal variant pads on the left only.
"""

from __future__ import annotations

import torch
from e00675_inputs import D_CHANNELS, N_CHANNELS
from torch import nn

HOURLY_DILATIONS = (1, 3, 9, 27, 81, 243)
DAILY_DILATIONS = (1, 2, 4, 8)


class ChannelNorm(nn.Module):
    """Layer norm over channels at each time step (no statistics across time)."""

    def __init__(self, ch: int) -> None:
        """Build the norm."""
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise (batch, channels, time) over channels."""
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    """A residual dilated convolution (kernel 3), bidirectional or causal."""

    def __init__(self, *, ch: int, dilation: int, causal: bool) -> None:
        """Build the block."""
        super().__init__()
        self.dilation, self.causal = dilation, causal
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.norm = ChannelNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to (batch, channels, time)."""
        pad = (2 * self.dilation, 0) if self.causal else (self.dilation, self.dilation)
        y = self.conv(nn.functional.pad(x, pad))
        return x + self.mix(nn.functional.gelu(self.norm(y)))


class Encoder(nn.Module):
    """Hourly plus daily encoder with one round of neighbour averaging."""

    def __init__(self, *, causal: bool = False, ch: int = 64, dch: int = 32) -> None:
        """Build the encoder."""
        super().__init__()
        self.ch, self.causal = ch, causal
        self.h_in = nn.Conv1d(N_CHANNELS, ch, 1)
        self.h_blocks = nn.ModuleList(Block(ch=ch, dilation=d, causal=causal) for d in HOURLY_DILATIONS)
        self.d_in = nn.Conv1d(D_CHANNELS, dch, 1)
        self.d_blocks = nn.ModuleList(Block(ch=dch, dilation=d, causal=causal) for d in DAILY_DILATIONS)
        self.fuse = nn.Conv1d(dch, ch, 1)
        self.nb = nn.Conv1d(ch, ch, 1)
        self.out_norm = ChannelNorm(ch)

    def forward(
        self, hourly: torch.Tensor, daily: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one sample.

        Args:
            hourly: (batch, nodes, hours, N_CHANNELS).
            daily: (batch, nodes, days, D_CHANNELS).
            adj: Row-normalised neighbour matrix (nodes, nodes).

        Returns:
            Hourly features (batch, nodes, hours, ch) and daily features (batch, nodes, days, dch).
        """
        b, n, t, _ = hourly.shape
        h = self.h_in(hourly.reshape(b * n, t, -1).transpose(1, 2))
        for blk in self.h_blocks:
            h = blk(h)
        d = self.d_in(daily.reshape(b * n, daily.shape[2], -1).transpose(1, 2))
        for blk in self.d_blocks:
            d = blk(d)
        if self.causal:  # a day's features see the whole day, so use the previous day's
            d_use = nn.functional.pad(d, (1, 0))[..., :-1]
        else:
            d_use = d
        h = h + self.fuse(d_use).repeat_interleave(24, dim=-1)
        h = h.reshape(b, n, self.ch, t)
        m = torch.einsum("ij,bjct->bict", adj, h)
        h = h + self.nb(m.reshape(b * n, self.ch, t)).reshape(b, n, self.ch, t)
        h = self.out_norm(h.reshape(b * n, self.ch, t)).reshape(b, n, self.ch, t)
        return h.permute(0, 1, 3, 2), d.reshape(b, n, -1, d.shape[-1]).permute(0, 1, 3, 2)


class Pretrainer(nn.Module):
    """Encoder plus the reconstruction head (masked arm) or the next-hours head (causal arm)."""

    def __init__(self, *, causal: bool) -> None:
        """Build the heads."""
        super().__init__()
        self.causal = causal
        self.encoder = Encoder(causal=causal)
        self.hour_head = nn.Linear(64, 2)
        self.day_head = nn.Linear(32, 1)

    def forward(
        self, hourly: torch.Tensor, daily: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (location, log scale) per hour and the daily mean per day."""
        h, d = self.encoder(hourly, daily, adj)
        out = self.hour_head(h)
        return out, self.day_head(d).squeeze(-1)

"""A one-dimensional convolutional encoder over a 24 h weather window, feeding the lag-free network.

The encoder reads the seven ensemble-mean weather variables (shortwave, longwave, temperature, dew
point, surface pressure, precipitation, 10 m wind speed) at 48 half-hourly positions, either
centred on the valid time (12 h before to 11.5 h after) or trailing (24 h ending at the valid
time), and returns a small vector. `WindowedMdn` concatenates that vector to the 37 lag-free
columns and passes the row to experiment 397's `PlainMdn`, so everything else about the network
is unchanged. The pretrained weights of the 37-column network do not load: the first layer is
wider, so each arm is pretrained afresh.
"""

from typing import Final

import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_VARIABLES: Final[int] = 7
N_POSITIONS: Final[int] = 48
CHANNELS: Final[tuple[int, int]] = (16, 32)
KERNEL: Final[int] = 5
OUT_DIM: Final[int] = 8


class WindowEncoder(nn.Module):
    """Two strided convolutions over the window, then a linear layer to a small vector.

    The window is flattened after the second convolution, not pooled, because the position of a
    change inside the window (before or after the valid time) is what the encoder must keep.
    """

    def __init__(self, *, out_dim: int = OUT_DIM, dropout: float = 0.0) -> None:
        """Build the encoder; the input is (batch, `N_VARIABLES`, `N_POSITIONS`)."""
        super().__init__()
        c1, c2 = CHANNELS
        pad = KERNEL // 2
        self.conv = nn.Sequential(
            nn.Conv1d(N_VARIABLES, c1, KERNEL, stride=2, padding=pad),
            nn.GELU(),
            nn.Conv1d(c1, c2, KERNEL, stride=2, padding=pad),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(c2 * (N_POSITIONS // 4), out_dim)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """Return the window's vector, shape (batch, `out_dim`)."""
        if window.shape[1:] != (N_VARIABLES, N_POSITIONS):
            msg = f"window must be (batch, {N_VARIABLES}, {N_POSITIONS}), got {tuple(window.shape)}"
            raise ValueError(msg)
        return self.project(self.dropout(self.conv(window).flatten(start_dim=1)))


class WindowedMdn(nn.Module):
    """The lag-free mixture network with a window encoder in front of its inputs."""

    def __init__(self, *, mdn: nn.Module, encoder: WindowEncoder) -> None:
        """Wrap `mdn` (a `PlainMdn` built for `37 + encoder out_dim` inputs) and its encoder."""
        super().__init__()
        self.mdn = mdn
        self.encoder = encoder

    def forward(
        self, x: torch.Tensor, window: torch.Tensor, primary: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the raw mixture outputs for rows `x` (batch, 37) with windows (batch, 7, 48)."""
        return self.mdn(torch.cat([x, self.encoder(window)], dim=-1), primary)

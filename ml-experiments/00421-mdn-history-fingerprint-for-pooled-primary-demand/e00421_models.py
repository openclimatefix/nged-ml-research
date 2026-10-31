"""The pooled plain mixture density network with an optional 7-day history encoder (421).

`HistoryMdn` is experiment 397's `PlainMdn` (same multilayer perceptron and mixture head) with two
independent switches: the Primary embedding, and a 1D convolutional encoder over the last 7 days of
the Primary's own power. The encoder reads two channels, the standardised power and a flag that is 1
where the reading is missing, and its flattened output keeps the time of day because every history
ends at the same time of day.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

HISTORY_STEPS: Final[int] = 336
"""Seven days of half-hours."""
ENCODER_WIDTH: Final[int] = 32
CHANNELS: Final[int] = 16
EMBEDDING: Final[int] = 8


class HistoryEncoder(nn.Module):
    """Two stride-2 convolutions over (value, missing flag), then one linear layer."""

    def __init__(self) -> None:
        """Build the encoder."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, CHANNELS, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(CHANNELS, CHANNELS, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.linear = nn.Linear(CHANNELS * (HISTORY_STEPS // 4), ENCODER_WIDTH)
        self.norm = nn.LayerNorm(ENCODER_WIDTH)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return the 32-wide encoding of histories shaped (rows, 2, 336)."""
        h = self.conv(history).flatten(start_dim=1)
        return self.norm(torch.nn.functional.gelu(self.linear(h)))


class HistoryMdn(nn.Module):
    """A mixture density network over one concatenated row, with optional embedding and encoder."""

    def __init__(
        self,
        *,
        n_inputs: int,
        n_primaries: int,
        use_embedding: bool,
        use_history: bool,
        hidden: int,
        layers: int,
        components: int,
        dropout: float,
    ) -> None:
        """Build the network; the output layer starts at zero weights."""
        super().__init__()
        self.components = components
        self.embedding = nn.Embedding(n_primaries, EMBEDDING) if use_embedding else None
        self.encoder = HistoryEncoder() if use_history else None
        width = (
            n_inputs + (EMBEDDING if use_embedding else 0) + (ENCODER_WIDTH if use_history else 0)
        )
        self.net = base.mlp(
            n_in=width, hidden=hidden, layers=layers, n_out=3 * components, dropout=dropout
        )
        base.init_head(head=self.net, components=components)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """Return the encoder output; raises if the network has no encoder."""
        assert self.encoder is not None
        return self.encoder(history)

    def forward(
        self, x: torch.Tensor, primary: torch.Tensor, history: torch.Tensor | None
    ) -> torch.Tensor:
        """Return the raw mixture outputs."""
        parts = [x]
        if self.embedding is not None:
            parts.append(self.embedding(primary))
        if self.encoder is not None:
            assert history is not None
            parts.append(self.encoder(history))
        return self.net(torch.cat(parts, dim=-1))

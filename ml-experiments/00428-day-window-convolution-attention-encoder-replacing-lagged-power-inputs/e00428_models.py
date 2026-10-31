"""Networks of experiment 428: a day-window convolution and attention encoder, and its references.

`LaggedMdn` is the pooled plain mixture density network of experiment 397 given the three origin
lags (arm R and R-inj) or none (arm L0). `DayEncoderMdn` replaces the lag columns with up to three
windows of 48 half-hours of raw power: a one-dimensional convolution turns each window into 48
position vectors, and one attention head, whose query is built from the half-hour being forecast,
reads the position vectors of every window shown plus a learned "no day" token. Both end in the
same mixture head as experiment 397 (`mlp`, `init_head`, `mixture_*`).
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

WINDOW: Final[int] = 48
"""Half-hours in one day window."""
CHANNELS: Final[int] = 5
"""Power, observed flag, sin and cos of the UTC time of day, age before the origin."""
TOKEN: Final[int] = 32
WIDEN_STATS: Final[int] = 3
WIDEN_INIT: Final[float] = -10.0
"""Start of the raw widening weights: their softplus is about 5e-5, so the path starts near zero."""
"""Numbers the widening-only path reads: three summaries of the newest day's |surprise|."""
CONV_WIDTHS: Final[tuple[int, int, int]] = (16, 32, TOKEN)


class LaggedMdn(nn.Module):
    """Experiment 397's pooled plain network on a row of inputs (lag columns included or not)."""

    def __init__(
        self,
        *,
        n_inputs: int,
        n_primaries: int,
        hidden: int = 64,
        layers: int = 2,
        components: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Build the network; the output layer starts at zero weights."""
        super().__init__()
        self.components = components
        self.net = base.PlainMdn(
            n_inputs=n_inputs,
            hidden=hidden,
            layers=layers,
            components=components,
            dropout=dropout,
            n_primaries=n_primaries,
        )

    def forward(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the raw mixture outputs."""
        return self.net(x, primary)


class DayEncoderMdn(nn.Module):
    """Convolution over day windows, then attention from the forecast half-hour, then the head.

    Inputs to `forward`: `x` (rows, n_inputs) the standardised non-lag inputs, `primary` (rows,),
    `windows` (rows, days, channels, 48) the channels of each window (power already standardised, 0
    where unobserved, the observed flag in channel 1). `days` is 1, 2, or 3.
    """

    def __init__(
        self,
        *,
        n_inputs: int,
        n_primaries: int,
        days: int,
        hidden: int = 64,
        layers: int = 2,
        components: int = 3,
        dropout: float = 0.1,
        embedding: int = 8,
        channels: int = CHANNELS,
        widen_path: bool = False,
    ) -> None:
        """Build the encoder, the attention layer, and the head."""
        super().__init__()
        if days not in (1, 2, 3):
            msg = "days must be 1, 2, or 3"
            raise ValueError(msg)
        self.days, self.components, self.channels = days, components, channels
        self.embedding = nn.Embedding(n_primaries, embedding)
        a, b, c = CONV_WIDTHS
        self.conv = nn.Sequential(
            nn.Conv1d(channels, a, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(a, b, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(b, c, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.day_embedding = nn.Embedding(days, TOKEN)
        query_width = n_inputs + embedding
        self.query = nn.Linear(query_width, TOKEN)
        self.key = nn.Linear(TOKEN, TOKEN)
        self.value = nn.Linear(TOKEN, TOKEN)
        self.null_key = nn.Parameter(torch.zeros(TOKEN))
        self.null_value = nn.Parameter(torch.zeros(TOKEN))
        self.head = base.mlp(
            n_in=query_width + TOKEN, hidden=hidden, layers=layers, n_out=3 * components,
            dropout=dropout,
        )  # fmt: skip
        base.init_head(head=self.head, components=components)
        self.widen_raw = (
            nn.Parameter(torch.full((components, WIDEN_STATS), WIDEN_INIT)) if widen_path else None
        )

    def widen_stats(self, windows: torch.Tensor) -> torch.Tensor:
        """Return (rows, 3): the newest-12 maximum, mean, and maximum of D1's absolute surprise."""
        surprise = windows[:, 0, 5, :].abs()
        observed = windows[:, 0, 1, :]
        newest = (surprise * observed)[:, -12:].amax(dim=1)
        mean = (surprise * observed).sum(1) / observed.sum(1).clamp(min=1.0)
        return torch.stack([newest, mean, (surprise * observed).amax(dim=1)], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        primary: torch.Tensor,
        windows: torch.Tensor,
        *,
        details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the raw mixture outputs; with `details`, also the attention mass per window.

        The mass tensor has shape (rows, days + 1): the mass on each window, then on the "no day"
        token.
        """
        rows, days = windows.shape[0], windows.shape[1]
        flat = windows.reshape(rows * days, self.channels, WINDOW)
        encoded = self.conv(flat).reshape(rows, days, TOKEN, WINDOW).transpose(2, 3)
        encoded = encoded + self.day_embedding.weight[None, :days, None, :]
        observed = windows[:, :, 1, :] > 0.5
        keys = self.key(encoded)
        values = self.value(encoded)
        context_in = torch.cat([x, self.embedding(primary)], dim=-1)
        query = self.query(context_in)
        scale = float(TOKEN) ** -0.5
        scores = torch.einsum("rt,rdpt->rdp", query, keys) * scale
        scores = scores.masked_fill(~observed, float("-inf")).reshape(rows, days * WINDOW)
        null_score = (query * self.null_key).sum(-1, keepdim=True) * scale
        weights = torch.softmax(torch.cat([scores, null_score], dim=-1), dim=-1)
        flat_values = values.reshape(rows, days * WINDOW, TOKEN)
        context = (weights[:, :-1, None] * flat_values).sum(1) + weights[:, -1:] * self.null_value
        out = self.head(torch.cat([context_in, context], dim=-1))
        if self.widen_raw is not None:
            k = self.components
            term = self.widen_stats(windows) @ nn.functional.softplus(self.widen_raw).T
            out = torch.cat([out[:, : 2 * k], out[:, 2 * k :] + term], dim=-1)
        if details:
            mass = weights[:, :-1].reshape(rows, days, WINDOW).sum(-1)
            return out, torch.cat([mass, weights[:, -1:]], dim=-1)
        return out


def count_parameters(*, model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return base.count_parameters(model=model)

"""Experiment 729: the three architectures, the output heads, and the losses (torch).

Every network maps `(batch, channels, steps)` to `(batch, outputs, steps)`; the loss and the
predictor crop the scored day's 48 half-hours with `e00729_arms.middle_slice`. The two-lobe head has
five outputs per half-hour (a sign logit, and a mean and a spread for each lobe), the solar and
engine heads add two outputs each (a mean and a spread), and the quantile head has 11 outputs.
"""

from typing import Final

import e00729_core as core
import torch  # ty: ignore[unresolved-import]
import torch.nn.functional as F  # noqa: N812
from e00729_arms import AuxType, HeadType
from torch import nn  # ty: ignore[unresolved-import]

MEAN_SCALE: Final = 20.0
"""A lobe's mean is `MEAN_SCALE * softplus(raw)` points in magnitude."""
SPREAD_SCALE: Final = 5.0
"""A lobe's spread is `FLOOR_POINTS + SPREAD_SCALE * softplus(raw)` points."""
QUANTILE_SCALE: Final = 10.0
LOBE_OUTPUTS: Final = 5
AUX_OUTPUTS: Final = 2
LOG_SQRT_2PI: Final = 0.9189385332046727


def n_outputs(*, head: HeadType, aux: AuxType) -> int:
    """Return the number of output channels of a head and its auxiliary head."""
    if head == "quantile":
        return len(core.QUANTILE_LEVELS)
    return LOBE_OUTPUTS + (AUX_OUTPUTS if aux != "none" else 0)


class ConvBlock(nn.Module):
    """Two convolutions of width 5, each with group normalisation and a GELU."""

    def __init__(self, c_in: int, c_out: int) -> None:
        """Build the block."""
        super().__init__()
        groups = 8 if c_out % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, 5, padding=2), nn.GroupNorm(groups, c_out), nn.GELU(),
            nn.Conv1d(c_out, c_out, 5, padding=2), nn.GroupNorm(groups, c_out), nn.GELU(),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return self.net(x)


class UNet1d(nn.Module):
    """A 1-D U-net of 4 levels (32, 64, 128, 256 channels) with skip connections."""

    def __init__(self, c_in: int, n_out: int) -> None:
        """Build the network."""
        super().__init__()
        widths = (32, 64, 128, 256)
        self.down = nn.ModuleList()
        prev = c_in
        for w in widths:
            self.down.append(ConvBlock(prev, w))
            prev = w
        self.up = nn.ModuleList(
            nn.ConvTranspose1d(widths[i + 1], widths[i], 2, stride=2) for i in range(3)
        )
        self.dec = nn.ModuleList(ConvBlock(2 * widths[i], widths[i]) for i in range(3))
        self.head = nn.Conv1d(widths[0], n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `(B, C, L)` to `(B, n_out, L)`; `L` must be divisible by 8."""
        skips = []
        h = x
        for i, block in enumerate(self.down):
            h = block(h)
            if i < len(self.down) - 1:
                skips.append(h)
                h = F.max_pool1d(h, 2)
        for i in (2, 1, 0):
            h = self.up[i](h)
            h = self.dec[i](torch.cat([h, skips[i]], dim=1))
        return self.head(h)


class Dae(nn.Module):
    """The denoising autoencoder of Kelly and Knottenbelt, with a 128-unit bottleneck."""

    def __init__(self, c_in: int, n_out: int, steps: int) -> None:
        """Build the network for windows of `steps` half-hours."""
        super().__init__()
        self.pad = nn.ConstantPad1d((1, 2), 0.0)
        self.conv_in = nn.Conv1d(c_in, 8, 4)
        self.enc = nn.Linear(8 * steps, 128)
        self.dec = nn.Linear(128, steps)
        self.conv_out = nn.Conv1d(1, 16, 5, padding=2)
        self.head = nn.Conv1d(16, n_out, 1)
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `(B, C, steps)` to `(B, n_out, steps)`."""
        h = F.relu(self.conv_in(self.pad(x)))
        h = F.relu(self.enc(h.flatten(1)))
        h = F.relu(self.dec(h)).unsqueeze(1)
        return self.head(F.relu(self.conv_out(h)))


class BiLstm(nn.Module):
    """A convolution of 16 filters and two bidirectional layers of 128 units."""

    def __init__(self, c_in: int, n_out: int) -> None:
        """Build the network."""
        super().__init__()
        self.conv = nn.Conv1d(c_in, 16, 5, padding=2)
        self.lstm = nn.LSTM(16, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(256, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `(B, C, L)` to `(B, n_out, L)`."""
        h = F.relu(self.conv(x)).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out).transpose(1, 2)


def build_net(*, arch: str, c_in: int, steps: int, n_out: int) -> nn.Module:
    """Build one of the three architectures by name ("dae", "unet", or "bilstm")."""
    if arch == "dae":
        return Dae(c_in, n_out, steps)
    if arch == "unet":
        return UNet1d(c_in, n_out)
    if arch == "bilstm":
        return BiLstm(c_in, n_out)
    msg = f"unknown architecture {arch!r}"
    raise ValueError(msg)


def count_weights(*, net: nn.Module) -> int:
    """Return the number of trainable weights."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


# --------------------------------------------------------------------------- heads and losses


def lobe_parameters(*, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode the first five outputs into the two-lobe distribution.

    Args:
        raw: Network output, `(B, outputs, steps)`.

    Returns:
        `p` (probability of reverse flow), `mu_i` (positive), `s_i`, `mu_r` (negative), `s_r`, each
        `(B, steps)`.
    """
    return {
        "p": torch.sigmoid(raw[:, 0]),
        "mu_i": MEAN_SCALE * F.softplus(raw[:, 1]),
        "s_i": core.FLOOR_POINTS + SPREAD_SCALE * F.softplus(raw[:, 2]),
        "mu_r": -MEAN_SCALE * F.softplus(raw[:, 3]),
        "s_r": core.FLOOR_POINTS + SPREAD_SCALE * F.softplus(raw[:, 4]),
    }


def gaussian_logpdf(*, y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Return the log density of a Gaussian."""
    z = (y - mu) / sigma
    return -0.5 * z * z - torch.log(sigma) - LOG_SQRT_2PI


def mixture_nll(*, raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return the negative log likelihood of `x` under the two-lobe mixture, per half-hour."""
    q = lobe_parameters(raw=raw)
    log_p = torch.log(q["p"].clamp_min(1e-6))
    log_1p = torch.log((1.0 - q["p"]).clamp_min(1e-6))
    a = log_1p + gaussian_logpdf(y=x, mu=q["mu_i"], sigma=q["s_i"])
    b = log_p + gaussian_logpdf(y=x, mu=q["mu_r"], sigma=q["s_r"])
    return -torch.logsumexp(torch.stack([a, b]), dim=0)


def aux_nll(*, raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the Gaussian negative log likelihood of an auxiliary head's target (non-negative)."""
    mu = MEAN_SCALE * 5.0 * F.softplus(raw[:, 0])
    sigma = core.FLOOR_POINTS + SPREAD_SCALE * F.softplus(raw[:, 1])
    return -gaussian_logpdf(y=target, mu=mu, sigma=sigma)


def quantile_values(*, raw: torch.Tensor) -> torch.Tensor:
    """Return sorted quantiles `(B, 11, steps)`: a free first value plus positive increments."""
    first = QUANTILE_SCALE * raw[:, :1]
    steps = QUANTILE_SCALE * F.softplus(raw[:, 1:])
    return torch.cat([first, first + torch.cumsum(steps, dim=1)], dim=1)


def pinball(*, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over the 11 levels, per half-hour."""
    taus = torch.tensor(core.QUANTILE_LEVELS, device=q.device, dtype=q.dtype).view(1, -1, 1)
    err = x.unsqueeze(1) - q
    return torch.maximum(taus * err, (taus - 1.0) * err).mean(dim=1)


def batch_loss(
    *, raw: torch.Tensor, targets: torch.Tensor, head: HeadType, aux: AuxType
) -> torch.Tensor:
    """Return the loss of a batch: the mean over valid half-hours of the main and auxiliary terms.

    Args:
        raw: The network's output cropped to the scored day, `(B, outputs, 48)`.
        targets: `(B, 4, 48)`: truth, valid, engine, solar.
        head: The main head.
        aux: The auxiliary head.

    Returns:
        A scalar tensor.
    """
    x, valid = targets[:, 0], targets[:, 1]
    per = (
        pinball(q=quantile_values(raw=raw), x=x)
        if head == "quantile"
        else mixture_nll(raw=raw, x=x)
    )
    if aux != "none":
        target = targets[:, 3] if aux == "solar" else targets[:, 2]
        per = per + core.AUX_WEIGHT * aux_nll(raw=raw[:, LOBE_OUTPUTS:], target=target)
    return (per * valid).sum() / valid.sum().clamp_min(1.0)


def predict_lobes(*, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the two-lobe parameters of a cropped output, for scoring."""
    return lobe_parameters(raw=raw)

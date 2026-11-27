# ruff: noqa: E501
"""Experiment 732: the four heads on experiment 729's backbones, the softplus solar output, and the stop-gradient.

This module needs torch. Every network maps `(batch, channels, steps)` to `(batch, outputs, steps)`.
Output channel 0 is the logit of P(X < 0), exactly the head of experiment 729's backbone (so arm S+
is experiment 730's arm S). Channels 1 to 3, present when the arm has the extra heads, are the raw
solar output, the step logit, and the raw step size. Each extra head is a separate linear layer on
the backbone's last hidden layer, and nothing reads an output, so the sign logit does not depend on
the extra heads' weights. In arm U-SG the extra heads read the hidden layer through `detach()`, so
their losses train only their own layers.
"""

from typing import Final

import e00732_paths  # noqa: F401

# isort: split
import e00729_nets as nets729
import e00732_core as core
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

EXTRA_OUTPUTS: Final = 3
"""The solar output, the step logit, and the step size."""
HIDDEN: Final = {"unet": 32, "bilstm": 256}
"""The width of each backbone's last hidden layer."""


def unet_features(*, net: nets729.UNet1d, x: torch.Tensor) -> torch.Tensor:
    """Return the U-net's last hidden layer `(B, 32, L)`: experiment 729's forward without the final layer."""
    skips = []
    h = x
    for i, block in enumerate(net.down):
        h = block(h)
        if i < len(net.down) - 1:
            skips.append(h)
            h = F.max_pool1d(h, 2)
    for i in (2, 1, 0):
        h = net.up[i](h)
        h = net.dec[i](torch.cat([h, skips[i]], dim=1))
    return h


def bilstm_features(*, net: nets729.BiLstm, x: torch.Tensor) -> torch.Tensor:
    """Return the bidirectional LSTM's last hidden layer `(B, 256, L)`."""
    h = F.relu(net.conv(x)).transpose(1, 2)
    out, _ = net.lstm(h)
    return out.transpose(1, 2)


def apply_head(*, head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Apply a convolutional or linear head to channels-first features, returning `(B, n, L)`."""
    if isinstance(head, nn.Linear):
        return head(features.transpose(1, 2)).transpose(1, 2)
    return head(features)


class Net(nn.Module):
    """A backbone with the sign head and, optionally, the three extra heads."""

    def __init__(self, *, arch: str, c_in: int, heads: bool, stop_gradient: bool = False) -> None:
        """Build the network.

        Args:
            arch: "unet" or "bilstm".
            c_in: The number of input channels.
            heads: Whether the solar, step, and size heads exist.
            stop_gradient: The extra heads read the hidden layer through a stop-gradient.
        """
        super().__init__()
        self.arch = arch
        self.stop_gradient = stop_gradient
        self.backbone: nn.Module = nets729.build_net(arch=arch, c_in=c_in, steps=96, n_out=1)
        self.extra: nn.Module | None = None
        if heads:
            width = HIDDEN[arch]
            self.extra = (
                nn.Conv1d(width, EXTRA_OUTPUTS, 1)
                if arch == "unet"
                else nn.Linear(width, EXTRA_OUTPUTS)
            )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the backbone's last hidden layer."""
        if self.arch == "unet":
            return unet_features(net=self.backbone, x=x)  # ty: ignore[invalid-argument-type]
        return bilstm_features(net=self.backbone, x=x)  # ty: ignore[invalid-argument-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `(B, C, L)` to `(B, 1 or 4, L)`."""
        f = self.features(x)
        sign = apply_head(head=self.backbone.head, features=f)  # ty: ignore[invalid-argument-type]
        if self.extra is None:
            return sign
        g = f.detach() if self.stop_gradient else f
        return torch.cat([sign, apply_head(head=self.extra, features=g)], dim=1)


def build_net(*, spec: core.ArmSpec732, c_in: int) -> Net:
    """Build the network of an arm."""
    return Net(arch=spec.arch, c_in=c_in, heads=spec.heads, stop_gradient=spec.stop_gradient)


def count_weights(*, net: nn.Module) -> int:
    """Return the number of trainable weights."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def solar_of(*, raw: torch.Tensor) -> torch.Tensor:
    """Return the solar output in points: a softplus of the raw channel, so it is never negative."""
    return core.SOLAR_SCALE * F.softplus(raw)


def decode(*, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode raw outputs `(N, outputs, 48)` into P(X<0) and, with the extra heads.

    The solar output, step probability, and step size.
    """
    out = {"p": torch.sigmoid(raw[:, 0])}
    if raw.shape[1] > 1:
        out["pv"] = solar_of(raw=raw[:, 1])
        out["step_p"] = torch.sigmoid(raw[:, 2])
        out["size"] = core.SIZE_SCALE * raw[:, 3]
    return out

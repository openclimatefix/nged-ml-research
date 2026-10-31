# Copied unchanged from experiment 397 (PR 50, branch mdn-attention-demand).
# ruff: noqa: E501
"""The plain mixture density network and the group-attention mixture density network (397).

Both predict the per-Primary standardised power `z = (power / capacity - mean) / std` as a mixture
of `components` Gaussians. `PlainMdn` is experiment 334's `MixtureMlp` with a configurable number of
hidden layers and components and an optional Primary embedding. `GroupAttentionMdn` encodes each
input group to a token, mixes the tokens with a softmax gate, and feeds the mixed token to the same
kind of head.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)
Z_LIMIT: Final[float] = 40.0
BISECTION_STEPS: Final[int] = 40
GROUPS: Final[tuple[str, ...]] = ("weather", "solar", "calendar", "holidays", "lags")
"""The five gated input groups, in token order."""
ARMS: Final[tuple[str, ...]] = ("A0", "A1", "A2", "A3")
"""Gate conditioning: A0 uniform weights, A1 lead, A2 lead + calendar + holidays, A3 adds weather."""
QUERY_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "A0": (),
    "A1": (),
    "A2": ("calendar", "holidays"),
    "A3": ("weather", "calendar", "holidays"),
}
"""The gated groups whose raw columns enter the gate's query, besides the lead (A1 to A3)."""


@dataclass(frozen=True)
class Layout:
    """Where each input group sits in the input matrix.

    Attributes:
        slices: Column range of each group in `GROUPS`, and of `lead` (never gated).
        n_inputs: Number of columns.
    """

    slices: dict[str, slice]
    n_inputs: int

    def columns_of(self, *, groups: Sequence[str]) -> list[int]:
        """Return the column indices of the named groups, in the order named."""
        return [i for g in groups for i in range(self.n_inputs)[self.slices[g]]]


def mixture_head_bias(*, components: int) -> list[float]:
    """Return the output bias that starts the mixture as evenly spread components."""
    offsets = np.linspace(-0.5, 0.5, components) if components > 1 else np.zeros(1)
    return [0.0] * components + [float(o) for o in offsets] + [0.0] * components


def mixture_parts(
    *, out: torch.Tensor, components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split raw outputs (..., 3 K) into log weights, means, and standard deviations."""
    k = components
    return (
        torch.log_softmax(out[..., :k], dim=-1),
        out[..., k : 2 * k],
        torch.exp(out[..., 2 * k :].clamp(-3.0, 3.0)),
    )


def mixture_nll(*, out: torch.Tensor, z: torch.Tensor, components: int) -> torch.Tensor:
    """Return the negative log-likelihood of `z` for each row."""
    log_w, mean, std = mixture_parts(out=out, components=components)
    log_pdf = -0.5 * ((z[..., None] - mean) / std) ** 2 - torch.log(std)
    return -torch.logsumexp(log_w + log_pdf, dim=-1) + 0.5 * float(np.log(2 * np.pi))


def mixture_quantiles(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Return the 11 quantiles (rows, 11) of mixtures (rows, components), by bisection on the CDF."""
    levels = torch.tensor(LEVELS, dtype=mean.dtype, device=mean.device)
    w = log_w.exp()
    lo = torch.full((len(w), len(LEVELS)), -Z_LIMIT, dtype=mean.dtype, device=mean.device)
    hi = torch.full_like(lo, Z_LIMIT)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        z = (mid[..., None] - mean[:, None, :]) / (std[:, None, :] * float(np.sqrt(2.0)))
        cdf = (w[:, None, :] * 0.5 * (1.0 + torch.erf(z))).sum(-1)
        below = cdf < levels
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def init_head(*, head: nn.Sequential, components: int) -> None:
    """Zero the output layer's weights and set its bias to spread the components."""
    last = head[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    with torch.no_grad():
        last.bias.copy_(torch.tensor(mixture_head_bias(components=components)))


def mlp(*, n_in: int, hidden: int, layers: int, n_out: int, dropout: float = 0.0) -> nn.Sequential:
    """Return `layers` hidden GELU layers of width `hidden`, then a linear output layer."""
    modules: list[nn.Module] = []
    width = n_in
    for _ in range(layers):
        modules += [nn.Linear(width, hidden), nn.GELU()]
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        width = hidden
    modules.append(nn.Linear(width, n_out))
    return nn.Sequential(*modules)


class PlainMdn(nn.Module):
    """A multilayer perceptron with a mixture output; the inputs are one concatenated row.

    With `n_primaries > 0` the Primary's embedding is appended to the row (the pooled network).
    """

    def __init__(
        self,
        *,
        n_inputs: int,
        hidden: int = 64,
        layers: int = 2,
        components: int = 3,
        dropout: float = 0.0,
        n_primaries: int = 0,
        embedding: int = 8,
    ) -> None:
        """Build the network; the output layer starts at zero weights."""
        super().__init__()
        self.components = components
        self.embedding = nn.Embedding(n_primaries, embedding) if n_primaries > 0 else None
        width = n_inputs + (embedding if n_primaries > 0 else 0)
        self.net = mlp(
            n_in=width, hidden=hidden, layers=layers, n_out=3 * components, dropout=dropout
        )
        init_head(head=self.net, components=components)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the raw mixture outputs."""
        if self.embedding is not None:
            assert primary is not None
            x = torch.cat([x, self.embedding(primary)], dim=-1)
        return self.net(x)


class GroupAttentionMdn(nn.Module):
    """Five group tokens mixed by a softmax gate, then the plain mixture head.

    Each group g gives a token `LayerNorm(GELU(W_g x_g))` of width `token`. The gate's logits come
    from a small network on the query, which is built from the lead features, the raw columns of the
    groups the arm names in `QUERY_GROUPS`, and (with `primary_in_query`) the Primary embedding. Arm
    A0 has no query and weights every token 0.2. The mixed token is `5 sum_g a_g h_g`, so uniform
    weights reduce to the plain sum. The head sees the mixed token, the lead features, and the
    Primary embedding.
    """

    def __init__(
        self,
        *,
        layout: Layout,
        arm: str,
        token: int = 32,
        hidden: int = 64,
        layers: int = 2,
        components: int = 3,
        dropout: float = 0.0,
        n_primaries: int = 16,
        embedding: int = 8,
        primary_in_query: bool = True,
    ) -> None:
        """Build the encoders, the gate, the Primary embedding and the head."""
        super().__init__()
        if arm not in ARMS:
            msg = f"unknown arm {arm!r}"
            raise ValueError(msg)
        self.layout, self.arm, self.components = layout, arm, components
        self.encoders = nn.ModuleDict(
            {g: nn.Linear(len(layout.columns_of(groups=[g])), token) for g in GROUPS}
        )
        self.norms = nn.ModuleDict({g: nn.LayerNorm(token) for g in GROUPS})
        self.embedding = nn.Embedding(n_primaries, embedding)
        self.query_columns = layout.columns_of(groups=["lead", *QUERY_GROUPS[arm]])
        self.primary_in_query = primary_in_query
        self.lead_columns = layout.columns_of(groups=["lead"])
        if arm == "A0":
            self.gate = None
        else:
            width = len(self.query_columns) + (embedding if primary_in_query else 0)
            self.gate = nn.Sequential(nn.Linear(width, 16), nn.GELU(), nn.Linear(16, len(GROUPS)))
        self.head = mlp(
            n_in=token + len(self.lead_columns) + embedding,
            hidden=hidden,
            layers=layers,
            n_out=3 * components,
            dropout=dropout,
        )
        init_head(head=self.head, components=components)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return the group tokens, shape (rows, 5, token)."""
        return torch.stack(
            [
                self.norms[g](
                    torch.nn.functional.gelu(
                        self.encoders[g](x[:, self.layout.columns_of(groups=[g])])
                    )
                )
                for g in GROUPS
            ],
            dim=1,
        )

    def weights(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the gate weights, shape (rows, 5)."""
        if self.gate is None:
            return torch.full((len(x), len(GROUPS)), 1.0 / len(GROUPS), device=x.device)
        query = x[:, self.query_columns]
        if self.primary_in_query:
            query = torch.cat([query, self.embedding(primary)], dim=-1)
        return torch.softmax(self.gate(query), dim=-1)

    def forward(
        self, x: torch.Tensor, primary: torch.Tensor, *, details: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the raw mixture outputs; with `details`, also the weights and token norms."""
        tokens = self.tokens(x)
        weights = self.weights(x, primary)
        mixed = len(GROUPS) * (weights[..., None] * tokens).sum(dim=1)
        head_in = torch.cat([mixed, x[:, self.lead_columns], self.embedding(primary)], dim=-1)
        out = self.head(head_in)
        if details:
            return out, weights, tokens.norm(dim=-1)
        return out


def count_parameters(*, model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Experiment 518: the history encoder H, its three stage-0 losses, and the step read-out.

H reads one window: the 672 half-hours (14 days) of the reading, as a surprise against the
lagless median, beside that median. It writes, for every half-hour of the window, the probability
that a step is in force, the size of the step in spreads (a spread is the Primary's residual
standard deviation), and, once per window, the probability that the step in force at the newest
reading is a genuine level change rather than a temporary shift.

Architecture (about 100,000 parameters): a dilated convolution stem of width 32, a strided
convolution to 168 two-hour tokens of width 64, two transformer layers of width 64 with 4 heads and
a learned relative-position bias, and a head that combines each token with the stem features of its
four half-hours.

Losses (weights fixed before any fit; the reading-class loss L_cls of the plan is dropped):

- `L_u`: for the step head, a non-negative positive-unlabelled risk over every half-hour (an
  injected half-hour is a labelled positive; every other half-hour is unlabelled and may sit inside
  an unlogged real event), plus a Huber loss on the size at injected half-hours only. Weight 0.5.
- `L_g`: cross-entropy of the genuine-change head, on windows with a step in force at the newest
  reading whose onset lies inside the window. Weight 0.2.
- `L_tv`: L1 on the first differences of the estimated step. Weight 1e-3.
"""

from dataclasses import dataclass
from typing import Final

import torch  # ty: ignore[unresolved-import]
import torch.nn.functional as F  # noqa: N812  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

HISTORY: Final[int] = 672
TOKEN: Final[int] = 4
"""Half-hours per token (two hours)."""
TOKENS: Final[int] = HISTORY // TOKEN
CHANNELS: Final[int] = 2
"""The input channels: the surprise, and the lagless median standardised within the window."""
WIDTH: Final[int] = 64
STEM: Final[int] = 32
HEADS: Final[int] = 4
LAYERS: Final[int] = 2
SIZE_SCALE: Final[float] = 3.0
"""The size head predicts the step in spreads divided by this number."""
SURPRISE_CLIP: Final[float] = 30.0
"""Surprises beyond this many spreads are clipped before scaling."""
SURPRISE_SCALE: Final[float] = 10.0

WEIGHT_U: Final[float] = 0.5
WEIGHT_G: Final[float] = 0.2
WEIGHT_TV: Final[float] = 1e-3
PU_PRIOR: Final[float] = 0.03
"""Assumed share of half-hours with no injection that sit in a real, unlogged step (a guess)."""


class RelativeBiasAttention(nn.Module):
    """Multi-head self-attention with a learned bias per head and relative offset."""

    def __init__(self) -> None:
        """Create the projections and the bias table."""
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)
        self.bias = nn.Parameter(torch.zeros(HEADS, 2 * TOKENS - 1))
        self.offsets: torch.Tensor
        offsets = torch.arange(TOKENS)[None, :] - torch.arange(TOKENS)[:, None] + TOKENS - 1
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attended features (batch, tokens, width)."""
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, HEADS, WIDTH // HEADS).permute(2, 0, 3, 1, 4)
        bias = self.bias[:, self.offsets]  # (heads, t, t)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None].expand(b, -1, -1, -1))
        return self.out(y.transpose(1, 2).reshape(b, t, WIDTH))


class Layer(nn.Module):
    """One pre-norm transformer layer."""

    def __init__(self) -> None:
        """Create the layer."""
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.attention = RelativeBiasAttention()
        self.mlp = nn.Sequential(
            nn.Linear(WIDTH, 2 * WIDTH), nn.GELU(), nn.Linear(2 * WIDTH, WIDTH)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the layer's output."""
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@dataclass(frozen=True)
class Output:
    """What H writes for a batch of windows.

    Attributes:
        step_logit: Log-odds that a step is in force at each half-hour (batch, 672).
        size: The step in force in spreads, if there is one (batch, 672).
        genuine_logit: Log-odds that the step in force at the newest reading is genuine (batch,).
    """

    step_logit: torch.Tensor
    size: torch.Tensor
    genuine_logit: torch.Tensor

    @property
    def level(self) -> torch.Tensor:
        """Return the estimated step in spreads: probability of a step times its size."""
        return torch.sigmoid(self.step_logit) * self.size


class HistoryEncoder(nn.Module):
    """The history encoder H."""

    def __init__(self) -> None:
        """Create the stem, the layers, and the heads."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(CHANNELS, STEM, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(STEM, STEM, 5, padding=4, dilation=2),
            nn.GELU(),
        )
        self.down = nn.Conv1d(STEM, WIDTH, TOKEN, stride=TOKEN)
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(WIDTH)
        self.point = nn.Sequential(nn.Linear(WIDTH + STEM, 32), nn.GELU(), nn.Linear(32, 2))
        self.genuine = nn.Linear(WIDTH, 1)

    def forward(self, x: torch.Tensor) -> Output:
        """Return H's output for inputs of shape (batch, CHANNELS, 672)."""
        low = self.stem(x)  # (b, STEM, 672)
        tokens = self.down(low).transpose(1, 2)  # (b, 168, WIDTH)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        up = tokens.repeat_interleave(TOKEN, dim=1)  # (b, 672, WIDTH)
        out = self.point(torch.cat([up, low.transpose(1, 2)], dim=2))
        return Output(
            step_logit=out[..., 0],
            size=SIZE_SCALE * out[..., 1],
            genuine_logit=self.genuine(tokens[:, -1]).squeeze(-1),
        )


def parameter_count(*, model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pu_step_loss(
    *, logit: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor, prior: float = PU_PRIOR
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the non-negative positive-unlabelled risk, and the plain cross-entropy beside it.

    Injected half-hours (`positive`) are labelled positives. Every other valid half-hour is
    unlabelled, and a share `prior` of those sits in a real step. With the batch share `rho` of
    positives, the risk is `rho * E_P[l(+)] + (1 - rho) * max(0, E_U[l(-)] - prior * E_P[l(-)]) /
    (1 - prior)`, where `l(+) = softplus(-logit)` and `l(-) = softplus(logit)`. At `prior = 0` it is
    the plain cross-entropy, which is returned as the second value.

    Args:
        logit: Step log-odds (batch, 672).
        positive: True at injected half-hours.
        valid: True where the half-hour is observed and not masked.
        prior: The assumed share of unlabelled half-hours that are real steps.

    Returns:
        The risk and the plain cross-entropy, both scalars.
    """
    pos = positive & valid
    unl = ~positive & valid
    n_pos, n_unl = pos.sum().clamp(min=1), unl.sum().clamp(min=1)
    rho = pos.sum() / (pos.sum() + unl.sum()).clamp(min=1)
    pos_as_pos = (F.softplus(-logit) * pos).sum() / n_pos
    pos_as_neg = (F.softplus(logit) * pos).sum() / n_pos
    unl_as_neg = (F.softplus(logit) * unl).sum() / n_unl
    neg_part = torch.clamp(unl_as_neg - prior * pos_as_neg, min=0.0) / (1.0 - prior)
    risk = rho * pos_as_pos + (1 - rho) * neg_part
    plain = rho * pos_as_pos + (1 - rho) * unl_as_neg
    return risk, plain


@dataclass(frozen=True)
class Losses:
    """The stage-0 losses of one batch (each a scalar tensor)."""

    total: torch.Tensor
    step: torch.Tensor
    size: torch.Tensor
    genuine: torch.Tensor
    smooth: torch.Tensor
    plain_step: torch.Tensor


def stage0_loss(
    *,
    out: Output,
    step_true: torch.Tensor,
    valid: torch.Tensor,
    genuine_target: torch.Tensor,
    genuine_valid: torch.Tensor,
    prior: float = PU_PRIOR,
) -> Losses:
    """Return the weighted stage-0 loss and its parts.

    Args:
        out: H's output.
        step_true: The injected step in spreads (batch, 672); zero where none.
        valid: False where the half-hour is unobserved or masked from the step loss.
        genuine_target: 1 for a genuine change, 0 for a temporary shift (batch,).
        genuine_valid: True for windows that carry a label for the genuine head (batch,).
        prior: The positive-unlabelled prior.

    Returns:
        The losses.
    """
    positive = step_true.abs() > 1e-6
    risk, plain = pu_step_loss(logit=out.step_logit, positive=positive, valid=valid, prior=prior)
    pos = positive & valid
    size = (
        F.huber_loss(out.size / SIZE_SCALE, step_true / SIZE_SCALE, reduction="none") * pos
    ).sum() / pos.sum().clamp(min=1)
    gv = genuine_valid.float()
    genuine = (
        F.binary_cross_entropy_with_logits(out.genuine_logit, genuine_target, reduction="none") * gv
    ).sum() / gv.sum().clamp(min=1.0)
    smooth = (out.level[:, 1:] - out.level[:, :-1]).abs().mean()
    u = risk + size
    total = WEIGHT_U * u + WEIGHT_G * genuine + WEIGHT_TV * smooth
    return Losses(
        total=total, step=risk, size=size, genuine=genuine, smooth=smooth, plain_step=plain
    )

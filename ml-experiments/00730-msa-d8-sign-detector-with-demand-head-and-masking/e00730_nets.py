"""Experiment 730: the transformer architectures, the encoder front, the heads, and the loss.

This module needs torch.

Every network maps `(batch, channels, steps)` to `(batch, outputs, steps)`. Output channel 0 is the
logit of P(X < 0). Channels 1 and 2, when the arm has the demand output, are the raw mean and spread
of a Gaussian over the clean demand, decoded as in experiment 729's auxiliary head. The demand rows
are separate rows of each backbone's final linear layer (`head`), so no later layer reads them and
the sign logit does not depend on the demand weights.
"""

import math
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00729_core as core729
import e00729_nets as nets729
import e00730_arms as arms
import e00730_core as core
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

EMBEDDING: Final = 8
"""The width of the Primary embedding of experiment 545's encoders."""
ENCODER_HIDDEN: Final = 64
ENCODER_OUT: Final = 32
ENCODER_DROPOUT: Final = 0.1
"""Each encoder is experiment 397's two-layer perceptron (`e00397_models.mlp`) and a GELU."""


def sinusoid_table(*, steps: int, width: int) -> torch.Tensor:
    """Return the fixed sinusoidal position code, `(steps, width)`."""
    position = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    table = torch.zeros(steps, width)
    table[:, 0::2] = torch.sin(position * freq)
    table[:, 1::2] = torch.cos(position * freq)
    return table


class Block(nn.Module):
    """A pre-norm transformer layer: self-attention then a feed-forward block, both residual."""

    def __init__(self, width: int, heads: int, hidden: int, dropout: float) -> None:
        """Build the layer."""
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width)
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, want_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the layer; the second value is the head-averaged attention `(B, L, L)` if asked."""
        h = self.ln1(x)
        a, weights = self.attn(h, h, h, need_weights=want_attention, average_attn_weights=True)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.ln2(x))), weights


class Transformer(nn.Module):
    """A 1-D convolution of width 5 as the token embedding, a fixed position code, and 3 layers.

    With `recurrent` set, the token embedding is a convolution to 64 channels and one bidirectional
    gated recurrent unit layer of 64 units in each direction (the GRU-transformer).
    """

    def __init__(self, c_in: int, n_out: int, *, steps: int = 96, recurrent: bool = False) -> None:
        """Build the network."""
        super().__init__()
        width = core.TRANSFORMER_WIDTH
        self.recurrent = recurrent
        if recurrent:
            self.embed = nn.Conv1d(c_in, core.GRU_TRANSFORMER_TOKEN, 5, padding=2)
            self.gru = nn.GRU(
                core.GRU_TRANSFORMER_TOKEN, core.GRU_TRANSFORMER_TOKEN, batch_first=True,
                bidirectional=True,
            )  # fmt: skip
        else:
            self.embed = nn.Conv1d(c_in, width, 5, padding=2)
        self.register_buffer("position", sinusoid_table(steps=steps, width=width))
        self.blocks = nn.ModuleList(
            Block(width, core.TRANSFORMER_HEADS, core.TRANSFORMER_FF, core.TRANSFORMER_DROPOUT)
            for _ in range(core.TRANSFORMER_LAYERS)
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_out)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Map `(B, C, L)` to `(B, n_out, L)`, and return each layer's attention if asked."""
        h = self.embed(x).transpose(1, 2)
        if self.recurrent:
            h, _ = self.gru(h)
        h = h + self.position[: h.shape[1]]
        attention = []
        for block in self.blocks:
            h, weights = block(h, return_attention)
            if weights is not None:
                attention.append(weights)
        out = self.head(self.norm(h)).transpose(1, 2)
        return (out, attention) if return_attention else out


class EncoderFront(nn.Module):
    """Experiment 545's weather and datetime encoders, with D8's embedding row, as an input stage.

    The window's rows `N_PLAIN_ROWS` onward hold the 7 weather and 21 datetime inputs of each
    half-hour. The front passes the first rows through and replaces the rest with the encoders' 64
    outputs.
    """

    def __init__(self, *, d8_index: int, n_primaries: int) -> None:
        """Build the front with random weights; `load_pretrained` fills the pre-trained arms."""
        super().__init__()
        import e00397_models as base  # ty: ignore[unresolved-import]

        self.d8_index = d8_index
        self.embedding = nn.Embedding(n_primaries, EMBEDDING)
        self.weather = base.mlp(
            n_in=core.ENCODER_WEATHER_COLUMNS + EMBEDDING, hidden=ENCODER_HIDDEN, layers=1,
            n_out=ENCODER_OUT, dropout=ENCODER_DROPOUT,
        )  # fmt: skip
        self.datetime = base.mlp(
            n_in=core.ENCODER_DATETIME_COLUMNS + EMBEDDING, hidden=ENCODER_HIDDEN, layers=1,
            n_out=ENCODER_OUT, dropout=ENCODER_DROPOUT,
        )  # fmt: skip

    @property
    def out_channels(self) -> int:
        """The number of channels this front hands to the backbone."""
        return arms.N_PLAIN_ROWS + core.ENCODER_OUTPUT_CHANNELS

    def load_pretrained(self, *, state: dict[str, torch.Tensor]) -> None:
        """Load the embedding and the two encoders from a pre-training state dictionary."""
        keep = {
            k: v
            for k, v in state.items()
            if k.split(".")[0] in ("embedding", "weather", "datetime")
        }
        result = self.load_state_dict(keep, strict=False)
        assert not result.unexpected_keys, result
        assert not result.missing_keys, result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map `(B, 8 + 28, L)` to `(B, 8 + 64, L)`."""
        plain = x[:, : arms.N_PLAIN_ROWS]
        enc = x[:, arms.N_PLAIN_ROWS :]
        batch, _, steps = enc.shape
        rows = enc.transpose(1, 2).reshape(batch * steps, -1)
        index = torch.full((rows.shape[0],), self.d8_index, dtype=torch.long, device=x.device)
        emb = self.embedding(index)
        weather = nn.functional.gelu(
            self.weather(torch.cat([rows[:, : core.ENCODER_WEATHER_COLUMNS], emb], dim=-1))
        )
        datetime = nn.functional.gelu(
            self.datetime(torch.cat([rows[:, core.ENCODER_WEATHER_COLUMNS :], emb], dim=-1))
        )
        coded = torch.cat([weather, datetime], dim=-1).reshape(batch, steps, -1).transpose(1, 2)
        return torch.cat([plain, coded], dim=1)


class Net(nn.Module):
    """An optional encoder front and a backbone, with the sign and demand outputs."""

    def __init__(self, front: EncoderFront | None, backbone: nn.Module) -> None:
        """Join the two parts."""
        super().__init__()
        self.front = front
        self.backbone = backbone

    @property
    def head(self) -> nn.Module:
        """The backbone's final layer, whose row 0 is the sign logit and rows 1 and 2 the demand."""
        return self.backbone.head

    def forward(self, x: torch.Tensor, **kwargs: bool) -> torch.Tensor | tuple:
        """Map `(B, C, L)` to `(B, n_out, L)`."""
        if self.front is not None:
            x = self.front(x)
        return self.backbone(x, **kwargs)


def build_backbone(*, arch: str, c_in: int, n_out: int) -> nn.Module:
    """Build one of the four architectures by name."""
    if arch == "bilstm":
        return nets729.BiLstm(c_in, n_out)
    if arch == "unet":
        return nets729.UNet1d(c_in, n_out)
    if arch == "transformer":
        return Transformer(c_in, n_out)
    if arch == "gru_transformer":
        return Transformer(c_in, n_out, recurrent=True)
    msg = f"unknown architecture {arch!r}"
    raise ValueError(msg)


def build_net(*, arch: str, spec: arms.ArmSpec, d8_index: int = 0, n_primaries: int = 16) -> Net:
    """Build the network of an arm: the front (encoder arms only) and the backbone."""
    front = None
    c_in = arms.n_channels(spec=spec)
    if spec.inputs == "enc":
        front = EncoderFront(d8_index=d8_index, n_primaries=n_primaries)
        c_in = front.out_channels + (1 if spec.causal else 0)
    return Net(front, build_backbone(arch=arch, c_in=c_in, n_out=spec.n_outputs))


def count_weights(*, net: nn.Module) -> int:
    """Return the number of trainable weights."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


# --------------------------------------------------------------------------- loss and decoding


def masked_mean(*, per: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Return the mean of `per` over the half-hours where `keep` is 1 (0 when none is kept)."""
    return (per * keep).sum() / keep.sum().clamp_min(1.0)


def batch_loss(
    *, raw: torch.Tensor, targets: torch.Tensor, keeps: torch.Tensor, lam: float
) -> torch.Tensor:
    """Return the sign cross-entropy plus `lam` times the demand negative log likelihood.

    The sign loss averages over the half-hours that `keeps[:, 0]` marks (valid and not hidden), and
    the demand loss over those that `keeps[:, 1]` marks (valid; all zero for an arm without the
    demand output). A masked half-hour therefore keeps the demand loss and drops the sign loss.

    Args:
        raw: The network's output cropped to the scored day, `(B, outputs, 48)`.
        targets: `(B, 6, 48)`: truth, valid, engine, solar, reading, base demand.
        keeps: `(B, 2, 48)`: where the sign loss and the demand loss apply.
        lam: The demand loss weight.

    Returns:
        A scalar tensor.
    """
    sign = masked_mean(per=nets729.sign_bce(raw=raw[:, :1], x=targets[:, 0]), keep=keeps[:, 0])
    if raw.shape[1] < 3:
        return sign
    demand = nets729.aux_nll(raw=raw[:, 1:3], target=targets[:, arms.BASE_ROW])
    return sign + lam * masked_mean(per=demand, keep=keeps[:, 1])


def decode(*, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode raw outputs `(N, outputs, 48)` into P(X<0) and, with the demand head, a Gaussian."""
    out = {"p": torch.sigmoid(raw[:, 0])}
    if raw.shape[1] >= 3:
        out["demand_mu"] = nets729.MEAN_SCALE * 5.0 * nn.functional.softplus(raw[:, 1])
        out["demand_s"] = core729.FLOOR_POINTS + nets729.SPREAD_SCALE * nn.functional.softplus(
            raw[:, 2]
        )
    return out

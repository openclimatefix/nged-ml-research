# ruff: noqa: E501
"""Experiment 713: the axial-attention "MSA" demand network.

`MsaNet` reads a matrix of 15 rows by 48 half-hours (section 2 of the design). Row 0 is the target
row (forecast-time weather and date-time, no power). Rows 1 to 14 are the history rows of ages 2 to
15 (weather and date-time at each token's own time, and the cleaned power where it is visible). Two
axial blocks apply within-day attention (the 48 tokens of a row), across-day attention (the 15
tokens of a column), and a transition. A similar-day read-out turns the history power of one column
into a weighted mean `base`, and the Candidate's three-component mixture head forecasts the
standardised power of the 48 target tokens.

The encoders keep the Candidate's module names (`embedding`, `weather`, `datetime`) so that the
`pretrained_P2` state of the Candidate chain loads into them by key.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

import e00397_models as base_models  # ty: ignore[unresolved-import]
import e00559_model as m559  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]
from torch.nn import functional  # ty: ignore[unresolved-import]

WIDTH: Final = 64
HEADS: Final = 4
BLOCKS: Final = 2
DROPOUT: Final = m559.DROPOUT
N_ROWS: Final = 15
N_COLS: Final = 48
N_HISTORY: Final = 14
N_PRIMARIES: Final = 16
N_LEAD_DAYS: Final = 14
N_DAY_TYPES: Final = 8
N_WEATHER: Final = 7
N_B1: Final = 14
N_DATETIME: Final = 21
N_SPREAD: Final = 7
POWER_CLIP: Final = 5.0
READOUT_WIDTH: Final = 32
SAME_DAY_TYPE_PRIOR: Final = 3.0
WEEKDAY_PRIOR: Final = 1.5
ENCODER_PREFIXES: Final = ("weather.", "datetime.", "embedding.")
StemKind = Literal["none", "conv", "gru"]


@dataclass(frozen=True)
class ArmSwitches:
    """The architectural switches of the arms of section 1.2.

    Attributes:
        within_day: Apply the within-day attention (False in arm `A` and `K`).
        across_day: Apply the across-day attention (False in arm `K`).
        top_row_only: Build the target row only (arm `N`).
        stem: A temporal stem on each history row: `conv` (arm `E`), `gru` (arm `G`), or `none`.
    """

    within_day: bool = True
    across_day: bool = True
    top_row_only: bool = False
    stem: StemKind = "none"


ARM_SWITCHES: Final[Mapping[str, ArmSwitches]] = {
    "M": ArmSwitches(),
    "P": ArmSwitches(),
    "A": ArmSwitches(within_day=False),
    "K": ArmSwitches(within_day=False, across_day=False),
    "N": ArmSwitches(top_row_only=True),
    "E": ArmSwitches(stem="conv"),
    "G": ArmSwitches(stem="gru"),
}
"""Per arm letter (the MSA arms `M1`, `M2`, `M3` and `M*` are all `M`), the switches."""


@dataclass
class MsaInputs:
    """One batch of `B` samples, all tensors on the device (section 2.2)."""

    primary: torch.Tensor  # (B,) int
    lead_day: torch.Tensor  # (B,) int, k - 1
    top_weather: torch.Tensor  # (B, 48, 21): 7 ENS-mean weather columns, 14 B1 columns
    top_spread: torch.Tensor  # (B, 48, 7)
    top_datetime: torch.Tensor  # (B, 48, 21)
    hist_weather: torch.Tensor  # (B, 14, 48, 21)
    hist_datetime: torch.Tensor  # (B, 14, 48, 21)
    hist_power: torch.Tensor  # (B, 14, 48), 0 where not visible
    hist_visible: torch.Tensor  # (B, 14, 48) bool
    row_exists: torch.Tensor  # (B, 14) bool
    day_type_top: torch.Tensor  # (B, 48) int
    day_type_hist: torch.Tensor  # (B, 14, 48) int

    def to(self, device: str | torch.device) -> MsaInputs:
        """Return the batch on `device`."""
        return MsaInputs(**{k: v.to(device) for k, v in self.__dict__.items()})

    def with_power(self, *, power: torch.Tensor, visible: torch.Tensor) -> MsaInputs:
        """Return the batch with another history power and visibility."""
        return replace(self, hist_power=power, hist_visible=visible)


@dataclass
class MsaOutput:
    """The raw mixture outputs and the read-out of one forward pass."""

    out: torch.Tensor  # (B, 48, 9): log-weight logits, means, log standard deviations
    base: torch.Tensor  # (B, 48) weighted mean of the visible same-half-hour history readings
    spread: torch.Tensor  # (B, 48)
    no_history: torch.Tensor  # (B, 48)
    alpha: torch.Tensor  # (B, 14, 48) read-out weights (0 where not visible)
    attention: dict[str, torch.Tensor] | None = None  # only with `return_attention`
    target_vector: torch.Tensor | None = (
        None  # (B, 48, 64) the head's input, only with `return_attention`
    )


def beta_day_start() -> torch.Tensor:
    """Return the 8 x 8 day-type prior table of the read-out at its start (section 2.5)."""
    table = torch.zeros(N_DAY_TYPES, N_DAY_TYPES)
    for i in range(5):
        for j in range(5):
            table[i, j] = SAME_DAY_TYPE_PRIOR if i == j else WEEKDAY_PRIOR
    for i in (5, 6, 7):
        table[i, i] = SAME_DAY_TYPE_PRIOR
    table[6, 7] = table[7, 6] = WEEKDAY_PRIOR
    return table


class Attention(nn.Module):
    """Multi-head self-attention with a fused projection and an optional learned key-row bias."""

    def __init__(self, *, key_rows: int = 0) -> None:
        """Build the projection (16,640 weights) and a zero key bias of `key_rows` rows per head."""
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)
        self.key_bias = nn.Parameter(torch.zeros(HEADS, key_rows)) if key_rows else None
        self.record: dict[str, torch.Tensor] | None = None
        """When a dict, `forward` stores the softmax weights (N, heads, L, L) under `weights`."""
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, *, key_ok: torch.Tensor | None = None) -> torch.Tensor:
        """Attend over the length-L axis of `x` (N, L, 64); `key_ok` (N, L) excludes keys."""
        n, length, _ = x.shape
        q, k, v = self.qkv(x).view(n, length, 3, HEADS, WIDTH // HEADS).permute(2, 0, 3, 1, 4)
        bias = torch.zeros(n, HEADS, 1, length, device=x.device, dtype=x.dtype)
        if self.key_bias is not None:
            bias = bias + self.key_bias[None, :, None, :length]
        if key_ok is not None:
            bias = bias.masked_fill(~key_ok[:, None, None, :], float("-inf"))
        if self.record is not None:
            with torch.no_grad():
                logits = q @ k.transpose(-1, -2) / float(np.sqrt(WIDTH // HEADS)) + bias
                self.record["weights"] = torch.softmax(logits, dim=-1)
        y = functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=DROPOUT if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(n, length, WIDTH))


class AxialBlock(nn.Module):
    """Within-day attention, across-day attention, and a transition; pre-LayerNorm and residual."""

    def __init__(self, *, within_day: bool, across_day: bool) -> None:
        """Build the block; a switched-off attention and its LayerNorm are absent."""
        super().__init__()
        self.ln_within = nn.LayerNorm(WIDTH) if within_day else None
        self.within = Attention() if within_day else None
        self.ln_across = nn.LayerNorm(WIDTH) if across_day else None
        self.across = Attention(key_rows=N_ROWS) if across_day else None
        self.ln_transition = nn.LayerNorm(WIDTH)
        self.transition = nn.Sequential(
            nn.Linear(WIDTH, 2 * WIDTH), nn.GELU(), nn.Linear(2 * WIDTH, WIDTH)
        )
        self.drop = nn.Dropout(DROPOUT)
        last = self.transition[-1]
        nn.init.zeros_(last.weight)  # ty: ignore[invalid-argument-type]
        nn.init.zeros_(last.bias)  # ty: ignore[invalid-argument-type]

    def forward(self, x: torch.Tensor, *, row_exists: torch.Tensor) -> torch.Tensor:
        """Return `x` (B, R, 48, 64) after the block; `row_exists` is (B, R)."""
        b, r, c, w = x.shape
        if self.within is not None and self.ln_within is not None:
            y = self.within(self.ln_within(x).reshape(b * r, c, w))
            x = x + self.drop(y.view(b, r, c, w))
        if self.across is not None and self.ln_across is not None:
            cols = self.ln_across(x).permute(0, 2, 1, 3).reshape(b * c, r, w)
            ok = row_exists[:, None, :].expand(b, c, r).reshape(b * c, r)
            y = self.across(cols, key_ok=ok).view(b, c, r, w).permute(0, 2, 1, 3)
            x = x + self.drop(y)
        return x + self.drop(self.transition(self.ln_transition(x)))


class ConvStem(nn.Module):
    """Arm `E`: a 1-D convolution over the 48 half-hours of each history row (2,560 weights)."""

    def __init__(self) -> None:
        """Build the stem; the last linear layer starts at zero."""
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=2)
        self.out = nn.Linear(16, WIDTH)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """Map (N, 2, 48) (power, visibility) to (N, 48, 64)."""
        h = functional.gelu(self.conv2(functional.gelu(self.conv1(series))))
        return self.out(h.transpose(1, 2))


class GruStem(nn.Module):
    """Arm `G`: a bidirectional gated recurrent unit of width 16 per direction (4,032 weights)."""

    def __init__(self) -> None:
        """Build the stem; the last linear layer starts at zero."""
        super().__init__()
        self.gru = nn.GRU(2, 16, batch_first=True, bidirectional=True)
        self.out = nn.Linear(32, WIDTH)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        """Map (N, 2, 48) (power, visibility) to (N, 48, 64)."""
        h, _ = self.gru(series.transpose(1, 2))
        return self.out(h)


class MsaNet(nn.Module):
    """The axial-attention demand network."""

    def __init__(self, *, switches: ArmSwitches = ARM_SWITCHES["M"]) -> None:
        """Build the network for the arm `switches`."""
        super().__init__()
        self.switches = switches
        self.frozen_encoders = False
        # the Candidate's encoders, unchanged in shape (module names match `pretrained_P2`)
        self.embedding = nn.Embedding(N_PRIMARIES, m559.EMBEDDING)
        self.weather = base_models.mlp(
            n_in=N_WEATHER + m559.EMBEDDING + N_B1, hidden=m559.ENCODER_HIDDEN, layers=1,
            n_out=m559.ENCODER_OUT, dropout=DROPOUT,
        )  # fmt: skip
        self.datetime = base_models.mlp(
            n_in=N_DATETIME + m559.EMBEDDING, hidden=m559.ENCODER_HIDDEN, layers=1,
            n_out=m559.ENCODER_OUT, dropout=DROPOUT,
        )  # fmt: skip
        # token maps and embeddings
        self.proj_w = nn.Linear(m559.ENCODER_OUT, WIDTH, bias=False)
        self.proj_d = nn.Linear(m559.ENCODER_OUT, WIDTH, bias=False)
        self.proj_s = nn.Linear(N_SPREAD, WIDTH, bias=False)
        nn.init.zeros_(self.proj_s.weight)
        self.tok_emb = nn.Embedding(N_PRIMARIES, WIDTH)
        self.col_emb = nn.Embedding(N_COLS, WIDTH)
        self.lead_emb = nn.Embedding(N_LEAD_DAYS, WIDTH)
        self.age_emb = nn.Embedding(N_HISTORY, WIDTH)
        for emb in (self.tok_emb, self.col_emb, self.lead_emb, self.age_emb):
            nn.init.normal_(emb.weight, std=0.02)
        self.power_u = nn.Parameter(torch.randn(WIDTH) * 0.02)
        self.power_b = nn.Parameter(torch.randn(WIDTH) * 0.02)
        self.m_target = nn.Parameter(torch.randn(WIDTH) * 0.02)
        self.m_hist = nn.Parameter(torch.randn(WIDTH) * 0.02)
        self.blocks = nn.ModuleList(
            AxialBlock(within_day=switches.within_day, across_day=switches.across_day)
            for _ in range(BLOCKS)
        )
        # the similar-day read-out
        self.ln_readout = nn.LayerNorm(WIDTH)
        self.w_q = nn.Linear(WIDTH, READOUT_WIDTH, bias=False)
        self.w_k = nn.Linear(WIDTH, READOUT_WIDTH, bias=False)
        # W_k starts at zero, so every logit starts at zero and the untrained read-out is the
        # day-type-weighted mean of the history (KA1). W_q keeps PyTorch's default random
        # initialisation: with both matrices at zero, each gradient is proportional to the other
        # matrix, so neither would ever move.
        nn.init.zeros_(self.w_k.weight)
        self.beta_day = nn.Parameter(beta_day_start())
        self.beta_age = nn.Parameter(torch.zeros(N_HISTORY))
        # the Candidate's mixture head
        self.head = base_models.mlp(
            n_in=WIDTH + 3, hidden=m559.MIXER_HIDDEN if hasattr(m559, "MIXER_HIDDEN") else 64,
            layers=1, n_out=3 * m559.COMPONENTS, dropout=DROPOUT,
        )  # fmt: skip
        base_models.init_head(head=self.head, components=m559.COMPONENTS)
        # built last, so that arms `E` and `G` draw the same initial weights as `M` for every shared parameter
        self.stem: nn.Module | None = (
            ConvStem() if switches.stem == "conv" else GruStem() if switches.stem == "gru" else None
        )

    # ------------------------------------------------------------------ encoders

    def encoder_parameters(self) -> list[nn.Parameter]:
        """Return the parameters of the loaded encoders (`weather`, `datetime`, `embedding`)."""
        return [p for m in (self.weather, self.datetime, self.embedding) for p in m.parameters()]

    def freeze_encoders(self) -> None:
        """Freeze the encoders (arm `M1`): no gradient, and evaluation mode so that dropout is off."""
        self.frozen_encoders = True
        for p in self.encoder_parameters():
            p.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True) -> MsaNet:
        """Set the mode; frozen encoders stay in evaluation mode, as in `e00574_fit.train`."""
        super().train(mode)
        if self.frozen_encoders:
            for m in (self.weather, self.datetime, self.embedding):
                m.eval()
        return self

    def load_encoders(self, *, path: Path, map_location: str | torch.device = "cpu") -> None:
        """Load `weather.*`, `datetime.*`, and `embedding.*` from a Candidate `pretrained_P2` state.

        The check is exact: the encoder keys of the file match the encoder keys of this module, and
        nothing else of this module is touched.
        """
        state = torch.load(path, map_location=map_location)
        wanted = {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIXES)}
        mine = {k for k in self.state_dict() if k.startswith(ENCODER_PREFIXES)}
        assert set(wanted) == mine, (sorted(set(wanted) ^ mine), path)
        result = self.load_state_dict(wanted, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        assert set(result.missing_keys).isdisjoint(mine), result.missing_keys

    # ------------------------------------------------------------------ forward

    def _tokens(self, inp: MsaInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the tokens (B, R, 48, 64) and the (B, R) row-exists mask."""
        b = len(inp.primary)
        emb = self.embedding(inp.primary)  # (B, 8)
        rows = 1 if self.switches.top_row_only else N_ROWS
        weather = inp.top_weather[:, None]
        datetime_ = inp.top_datetime[:, None]
        if rows > 1:
            weather = torch.cat([weather, inp.hist_weather], dim=1)
            datetime_ = torch.cat([datetime_, inp.hist_datetime], dim=1)
        e = emb[:, None, None, :].expand(b, rows, N_COLS, emb.shape[-1])
        w_in = torch.cat([weather[..., :N_WEATHER], e, weather[..., N_WEATHER:]], dim=-1)
        e_w = functional.gelu(self.weather(w_in))
        e_d = functional.gelu(self.datetime(torch.cat([datetime_, e], dim=-1)))
        x = self.proj_w(e_w) + self.proj_d(e_d)
        x = x + self.tok_emb(inp.primary)[:, None, None, :] + self.col_emb.weight[None, None]
        top = (
            self.lead_emb(inp.lead_day)[:, None, :]
            + self.proj_s(inp.top_spread)
            + self.m_target[None, None, :]
        )
        parts = [x[:, :1] + top[:, None]]
        if rows > 1:
            z = inp.hist_power.clamp(-POWER_CLIP, POWER_CLIP)
            power = torch.where(
                inp.hist_visible[..., None],
                z[..., None] * self.power_u + self.power_b,
                self.m_hist.expand_as(x[:, 1:]),
            )
            hist = x[:, 1:] + self.age_emb.weight[None, :, None, :] + power
            if self.stem is not None:
                series = torch.stack([z * inp.hist_visible, inp.hist_visible.to(z.dtype)], dim=2)
                stem = self.stem(series.reshape(b * N_HISTORY, 2, N_COLS))
                hist = hist + stem.view(b, N_HISTORY, N_COLS, WIDTH)
            parts.append(hist)
        row_exists = torch.ones(b, rows, dtype=torch.bool, device=x.device)
        if rows > 1:
            row_exists[:, 1:] = inp.row_exists
        return torch.cat(parts, dim=1), row_exists

    def _readout(
        self, x: torch.Tensor, inp: MsaInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `h0` and the read-out `base`, `spread`, `no_history`, and the weights `alpha`."""
        b = x.shape[0]
        h = self.ln_readout(x)
        h0 = h[:, 0]
        if self.switches.top_row_only:
            zero = torch.zeros(b, N_COLS, device=x.device, dtype=x.dtype)
            return (
                h0,
                zero,
                zero,
                torch.ones_like(zero),
                torch.zeros(b, N_HISTORY, N_COLS, device=x.device),
            )
        q = self.w_q(h0)  # (B, 48, 32)
        k = self.w_k(h[:, 1:])  # (B, 14, 48, 32)
        logit = torch.einsum("bjd,brjd->brj", q, k) / float(np.sqrt(READOUT_WIDTH))
        day = self.beta_day[inp.day_type_top[:, None, :], inp.day_type_hist]
        logit = logit + day + self.beta_age[None, :, None]
        visible = inp.hist_visible & inp.row_exists[:, :, None]
        any_visible = visible.any(dim=1)
        logit = logit.masked_fill(~visible, -1e9)
        alpha = torch.softmax(logit, dim=1) * any_visible[:, None, :]
        z = inp.hist_power.clamp(-POWER_CLIP, POWER_CLIP)
        base = (alpha * z).sum(dim=1)
        var = (alpha * (z - base[:, None, :]) ** 2).sum(dim=1)
        spread = torch.where(any_visible, torch.sqrt(var + 1e-6), torch.zeros_like(var))
        return h0, base, spread, (~any_visible).to(x.dtype), alpha

    def forward(self, inp: MsaInputs, *, return_attention: bool = False) -> MsaOutput:
        """Return the raw mixture outputs of the 48 target tokens of each sample.

        With `return_attention` the output also holds the attention weights of every block and the
        64-number vector that enters the head. The flag only records: the forecast is computed by the
        same operations and is bit-identical, and training never sets it.
        """
        x, row_exists = self._tokens(inp)
        attention_modules: list[tuple[str, Attention]] = []
        if return_attention:
            for i, block in enumerate(self.blocks):
                for kind in ("within", "across"):
                    module = getattr(block, kind)
                    if module is not None:
                        module.record = {}
                        attention_modules.append((f"{kind}_{i}", module))
        for block in self.blocks:
            x = block(x, row_exists=row_exists)
        h0, base, spread, no_history, alpha = self._readout(x, inp)
        out = self.head(
            torch.cat([h0, base[..., None], spread[..., None], no_history[..., None]], dim=-1)
        )
        k = m559.COMPONENTS
        out = torch.cat(
            [out[..., :k], out[..., k : 2 * k] + base[..., None], out[..., 2 * k :]], dim=-1
        )
        attention = None
        if return_attention:
            attention = {}
            b = len(inp.primary)
            for name, module in attention_modules:
                assert module.record is not None
                w = module.record.pop("weights")
                module.record = None
                if name.startswith("within"):  # (B*R, heads, 48, 48) -> (B, R, heads, 48, 48)
                    attention[name] = w.view(b, -1, *w.shape[1:])
                else:  # (B*48, heads, R, R): the target row's query -> (B, 48, heads, R)
                    attention[name] = w[:, :, 0, :].view(b, N_COLS, *w.shape[1:2], w.shape[-1])
        return MsaOutput(
            out=out, base=base, spread=spread, no_history=no_history, alpha=alpha,
            attention=attention, target_vector=h0.detach() if return_attention else None,
        )  # fmt: skip


def per_token_nll(*, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mixture negative log-likelihood of every target token, shape (B, 48)."""
    return base_models.mixture_nll(out=out, z=target, components=m559.COMPONENTS)


def standardised_quantiles(*, out: torch.Tensor) -> torch.Tensor:
    """Return the 11 quantiles (B, 48, 11) of the mixture, in standardised units."""
    shape = out.shape[:-1]
    log_w, mean, std = base_models.mixture_parts(
        out=out.reshape(-1, out.shape[-1]), components=m559.COMPONENTS
    )
    q = base_models.mixture_quantiles(log_w=log_w, mean=mean, std=std)
    return q.view(*shape, len(base_models.LEVELS))


def standardised_mixture_mean(*, out: torch.Tensor) -> torch.Tensor:
    """Return the mean (B, 48) of the mixture, in standardised units: the weights times the component means."""
    shape = out.shape[:-1]
    log_w, mean, _ = base_models.mixture_parts(
        out=out.reshape(-1, out.shape[-1]), components=m559.COMPONENTS
    )
    return (torch.softmax(log_w, dim=-1) * mean).sum(dim=-1).view(*shape)


def parameter_count(*, model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of parameters of `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)

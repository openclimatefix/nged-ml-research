# ruff: noqa: E501, B007, RET504
"""Experiment 568: the modular lag-free demand network with an attention mixer, and its MLP control.

The weather encoder, datetime encoder, and Primary embedding are experiment 554's M0 (`e00559_model`).
Only the mixer differs. The input row has the 37 columns of M0, then a block of 7 relative-spread
columns (see `e00568_spread`):

- 37: relative 2 m temperature spread; 38: wind speed; 39: shortwave radiation; 40: u of 10 m wind;
  41: v of 10 m wind; 42: total cloud cover (0 where the run has none); 43: cloud missing flag.

`AttentionMdn` reads the encoders' outputs as tokens. Fine-tuning (`ft`) adds a lead-and-spread token
built from the 2 lead columns (35-36) and the spread columns the arm selects. `MlpMdn` is the control:
M0's mixer with a chosen hidden width, optionally with the whole spread block appended (arm T4).
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_BASE: Final[int] = m.N_INPUTS
SPREAD_BLOCK: Final[slice] = slice(37, 44)
N_INPUTS: Final[int] = 44
TEMP_ONLY: Final[slice] = slice(37, 38)
D_MODEL: Final[int] = 32
HEADS: Final[int] = 4
LAYERS: Final[int] = 2
DROPOUT: Final[float] = 0.1
TOKEN_DROPOUT: Final[float] = 0.1
KINDS: Final[tuple[str, ...]] = ("weather", "datetime", "primary", "spread")
"""Token kinds in the order of their type embeddings; the query token has the type after these."""
SPREADS: Final[tuple[str, ...]] = ("multi", "temp", "none")


class Block(nn.Module):
    """A pre-norm self-attention block."""

    def __init__(self) -> None:
        """Build one block of width `D_MODEL`."""
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, HEADS, dropout=DROPOUT, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2 * D_MODEL, D_MODEL),
        )
        self.drop = nn.Dropout(DROPOUT)

    def forward(
        self, h: torch.Tensor, pad: torch.Tensor | None, record: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the block output and, when `record`, the head-averaged attention weights."""
        z = self.norm1(h)
        out, w = self.attn(
            z, z, z, key_padding_mask=pad, need_weights=record, average_attn_weights=True
        )
        h = h + self.drop(out)
        h = h + self.drop(self.ff(self.norm2(h)))
        return h, w


class AttnMixer(nn.Module):
    """Tokens in, 9 raw mixture outputs out: a learned query token attends over the other tokens."""

    def __init__(self, *, kinds: tuple[str, ...], spread_in: int, lead_on_query: bool) -> None:
        """Build the mixer for the token kinds in `kinds` (a subset of `KINDS`, weather first)."""
        super().__init__()
        assert kinds[0] == "weather"
        self.kinds, self.lead_on_query = kinds, lead_on_query
        width = {"weather": m.ENCODER_OUT, "datetime": m.ENCODER_OUT, "primary": m.EMBEDDING}
        self.project = nn.ModuleDict(
            {k: nn.Linear(width[k], D_MODEL) for k in kinds if k != "spread"}
        )
        if "spread" in kinds:
            assert spread_in > 0
            self.project["spread"] = base.mlp(
                n_in=spread_in + 2, hidden=2 * D_MODEL, layers=1, n_out=D_MODEL, dropout=DROPOUT
            )
        self.type_embedding = nn.Parameter(0.02 * torch.randn(len(KINDS) + 1, D_MODEL))
        self.query = nn.Parameter(0.02 * torch.randn(D_MODEL))
        self.lead_to_query = nn.Linear(2, D_MODEL) if lead_on_query else None
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = base.mlp(
            n_in=D_MODEL, hidden=2 * D_MODEL, layers=1, n_out=3 * m.COMPONENTS, dropout=DROPOUT
        )
        base.init_head(head=self.head, components=m.COMPONENTS)
        self.record = False
        self.last_weights: torch.Tensor | None = None
        """When `record` is set: the last layer's weights from the query to each token, shape (rows, tokens)."""

    def forward(
        self, *, tokens: dict[str, torch.Tensor], lead: torch.Tensor, spread: torch.Tensor | None
    ) -> torch.Tensor:
        """Return the 9 raw outputs; `lead` is the 2 lead columns and `spread` the selected spread columns."""
        seq = []
        for i, kind in enumerate(self.kinds):
            if kind == "spread":
                assert spread is not None
                x = self.project["spread"](torch.cat([lead, spread], dim=-1))
            else:
                x = self.project[kind](tokens[kind])
            seq.append(x + self.type_embedding[KINDS.index(kind)])
        q = self.query + self.type_embedding[len(KINDS)]
        q = q.expand(len(seq[0]), -1)
        if self.lead_to_query is not None:
            q = q + self.lead_to_query(lead)
        h = torch.stack([*seq, q], dim=1)
        pad = None
        if self.training:
            # Each non-weather token is dropped with probability TOKEN_DROPOUT; weather and the query never are.
            pad = torch.rand(h.shape[0], h.shape[1], device=h.device) < TOKEN_DROPOUT
            pad[:, 0] = False
            pad[:, -1] = False
        for j, block in enumerate(self.blocks):
            h, w = block(h, pad, self.record and j == len(self.blocks) - 1)
            if w is not None:
                self.last_weights = w[:, -1, :-1].detach()
        return self.head(self.norm(h[:, -1]))


class AttentionMdn(nn.Module):
    """M0's encoders and an attention mixer; `spread` is `multi`, `temp`, or `none` (arms T1/T5, T3, T2)."""

    def __init__(self, *, n_primaries: int, stage: str, spread: str) -> None:
        """Build the network; `stage` is `pre` (ERA5/CAMS pretraining, no spread token) or `ft`."""
        super().__init__()
        assert stage in {"pre", "ft"}
        assert spread in SPREADS
        self.stage, self.spread = stage, spread
        self.embedding = nn.Embedding(n_primaries, m.EMBEDDING)
        self.weather = base.mlp(
            n_in=7 + m.EMBEDDING,
            hidden=m.ENCODER_HIDDEN,
            layers=1,
            n_out=m.ENCODER_OUT,
            dropout=m.DROPOUT,
        )
        self.datetime = base.mlp(
            n_in=21 + m.EMBEDDING,
            hidden=m.ENCODER_HIDDEN,
            layers=1,
            n_out=m.ENCODER_OUT,
            dropout=m.DROPOUT,
        )
        self.mixer_pre = AttnMixer(
            kinds=("weather", "datetime", "primary"), spread_in=0, lead_on_query=False
        )
        self.mixer_ft = self.make_mixer_ft()

    def make_mixer_ft(self) -> AttnMixer:
        """Return a new fine-tuning mixer (also used for the fresh mixer of step 2)."""
        spread_in = {"multi": 7, "temp": 1, "none": 0}[self.spread]
        kinds = ("weather", "datetime", "primary") + (("spread",) if spread_in else ())
        return AttnMixer(kinds=kinds, spread_in=spread_in, lead_on_query=spread_in == 0)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        tokens = {"weather": hw, "datetime": hd, "primary": emb}
        mixer = self.mixer_pre if self.stage == "pre" else self.mixer_ft
        lead = x[:, m.LEAD_COLS]
        spread = None
        if self.stage == "ft":
            if self.spread == "multi":
                spread = x[:, SPREAD_BLOCK]
            elif self.spread == "temp":
                spread = x[:, TEMP_ONLY]
        return mixer(tokens=tokens, lead=lead, spread=spread)


class MlpMdn(m.ModularMdn):
    """M0 with a mixer of chosen hidden width; `with_block` appends the 7 spread-block columns (arm T4)."""

    def __init__(
        self,
        *,
        n_primaries: int,
        stage: str,
        hidden: int,
        with_block: bool,
        zero_stored: bool = False,
    ) -> None:
        """Build M0 and replace the fine-tuning mixer."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        self.hidden, self.with_block, self.zero_stored = hidden, with_block, zero_stored
        self.mixer_ft = self.make_mixer_ft()

    def make_mixer_ft(self) -> nn.Sequential:
        """Return a new fine-tuning mixer."""
        wide = 2 * m.ENCODER_OUT + m.EMBEDDING + 7 + 2 + (7 if self.with_block else 0)
        head = base.mlp(
            n_in=wide, hidden=self.hidden, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT
        )
        base.init_head(head=head, components=m.COMPONENTS)
        return head

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        stored = x[:, m.SPREAD_COLS]
        parts = [core, torch.zeros_like(stored) if self.zero_stored else stored, x[:, m.LEAD_COLS]]
        if self.with_block:
            parts.append(x[:, SPREAD_BLOCK])
        return self.mixer_ft(torch.cat(parts, dim=-1))


def n_parameters(module: nn.Module) -> int:
    """Return the number of parameters of a module."""
    return sum(p.numel() for p in module.parameters())


def matched_hidden(*, target: int, with_block: bool) -> int:
    """Return the MLP mixer's hidden width whose parameter count is closest to `target`."""
    best = min(
        range(32, 512),
        key=lambda h: abs(
            n_parameters(
                MlpMdn(n_primaries=1, stage="ft", hidden=h, with_block=with_block).mixer_ft
            )
            - target
        ),
    )
    return best

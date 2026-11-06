# ruff: noqa: E501, RUF100, UP037, FBT001, FBT002
# ruff: noqa: E501
"""Experiment 603: experiment 574's network with its mixer split into a kept stage 1 and a replaced stage 2.

The core is `[hw, hd, hl, emb]`, the weather, datetime, and lag encoder outputs and the Primary
embedding. Stage 1 (`mixer1`) maps the core to `h1` and is learnt in pre-training and kept (frozen)
afterwards. Stage 2 (`mixer_pre` in pre-training, `mixer_ft` in steps 2 and 3) is a hidden layer and
the Gaussian-mixture head; it is fresh in step 2. Kinds:

- `j3`: stage 2 reads `h1` (and, in steps 2 and 3, the spread and lead columns).
- `j4`: stage 2 also reads the raw weather, datetime, and lag encoder outputs.
- `j4b`: as `j4` without the lag encoder output.
- `j5`: stage 1 is residual, `h1 = core + g(core)`, so the raw embeddings reach stage 2.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00574_model as mod  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

KINDS: Final[tuple[str, ...]] = ("j3", "j4", "j4b", "j5")
H1: Final[int] = m.MIXER_HIDDEN


class Stage1(nn.Module):
    """Stage 1: `Linear`, GELU, dropout; with `residual` the output is the input plus `g(input)`."""

    def __init__(self, *, wide: int, residual: bool) -> None:
        """Build stage 1; `hold_eval` keeps it in evaluation mode while it is frozen."""
        super().__init__()
        self.residual, self.hold_eval = residual, False
        if residual:
            self.net = nn.Sequential(
                nn.Linear(wide, H1), nn.GELU(), nn.Dropout(m.DROPOUT), nn.Linear(H1, wide)
            )
        else:
            self.net = nn.Sequential(nn.Linear(wide, H1), nn.GELU(), nn.Dropout(m.DROPOUT))

    def forward(self, core: torch.Tensor) -> torch.Tensor:
        """Return `h1`."""
        out = self.net(core)
        return core + out if self.residual else out

    def train(self, mode: bool = True) -> "Stage1":  # noqa: FBT001, FBT002
        """Stay in evaluation mode while frozen, so dropout stays off."""
        return super().train(False if self.hold_eval else mode)  # ty: ignore[invalid-return-type]

    def freeze(self) -> None:
        """Freeze the parameters and hold evaluation mode."""
        self.hold_eval = True
        for q in self.parameters():
            q.requires_grad_(False)
        self.eval()


class SplitMdn(mod.PartDMdn):
    """`PartDMdn` whose mixer is stage 1 plus a fresh stage 2 (kind in `KINDS`)."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, kind: str, n_b1: int = 0
    ) -> None:
        """Build the network; stage 2 has one hidden layer, so the depth equals the unsplit mixer's."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1)
        assert kind in KINDS
        self.kind = kind
        wide = 2 * m.ENCODER_OUT + self.n_lag_out + m.EMBEDDING
        self.mixer1 = Stage1(wide=wide, residual=kind == "j5")
        h1_width = wide if kind == "j5" else H1
        raw = {
            "j3": 0,
            "j5": 0,
            "j4": 2 * m.ENCODER_OUT + self.n_lag_out,
            "j4b": 2 * m.ENCODER_OUT,
        }[kind]
        n_in = h1_width + raw
        self.mixer_pre = base.mlp(
            n_in=n_in, hidden=m.MIXER_HIDDEN, layers=1, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT
        )
        self.mixer_ft = base.mlp(
            n_in=n_in + 7 + 2,
            hidden=m.MIXER_HIDDEN,
            layers=1,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=m.COMPONENTS)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs, with the offset added to the component means."""
        assert primary is not None
        emb = self.embedding(primary)
        weather_in = [x[:, m.WEATHER_COLS], emb]
        if self.n_b1:
            weather_in.append(x[:, mod.b1_cols(n_b1=self.n_b1)])
        hw = nn.functional.gelu(self.weather(torch.cat(weather_in, dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        h1 = self.mixer1(core)
        raw = {"j3": [], "j5": [], "j4": [hw, hd, hl], "j4b": [hw, hd]}[self.kind]
        parts = [h1, *raw]
        if self.stage == "pre":
            out = self.mixer_pre(torch.cat(parts, dim=-1))
        else:
            out = self.mixer_ft(torch.cat([*parts, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))
        off = self.offset(x[:, mod.TAU], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)

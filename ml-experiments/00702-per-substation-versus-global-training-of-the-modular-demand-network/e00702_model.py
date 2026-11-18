# ruff: noqa: E501
"""Experiment 702: the two network variants that differ from experiment 591's `AuxLagMdn`.

- `ZeroEmbedding` replaces a network's Primary embedding in split-at-1. It returns zeros and has no
  parameter, so no part of the network can read a Primary identity, the layer shapes stay those of
  K1, and nothing can unfreeze it (experiment 574's `train` re-enables every parameter at its end).
- `Hybrid3Mdn` is split-at-2-plus-encoders: K1's network plus a second weather encoder (the 7
  weather means and the 14 thermal-lag and solar-gain columns, to 64 hidden units, to 32 outputs)
  and a second date-time encoder (21 columns, to 64, to 32), with no embedding input. Their 64
  outputs enter the fine-tuning mixer after the 145 existing inputs, so the mixer has 209 inputs.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00574_model as mod  # ty: ignore[unresolved-import]
import e00591_model as m591  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_B1: Final[int] = 14
"""Thermal-lag and solar-gain columns that the weather encoder reads beside its 7 weather means."""
MIXER_IN_K1: Final[int] = 145
MIXER_IN_HYBRID3: Final[int] = 209
NEW_ENCODER_OUT: Final[int] = m.ENCODER_OUT
FRESH_MODULES: Final[tuple[str, ...]] = ("weather2", "datetime2")


class ZeroEmbedding(nn.Module):
    """A Primary embedding that is always zero and holds no parameter."""

    def __init__(self, *, width: int = m.EMBEDDING) -> None:
        """Remember the embedding width."""
        super().__init__()
        self.width = width

    def forward(self, primary: torch.Tensor) -> torch.Tensor:
        """Return one row of zeros per input row."""
        return torch.zeros(len(primary), self.width, device=primary.device)


def zero_primary_embedding(*, model: nn.Module) -> None:
    """Replace `model.embedding` by `ZeroEmbedding`, so the network reads no Primary identity."""
    model.embedding = ZeroEmbedding()


class Hybrid3Mdn(m591.AuxLagMdn):
    """K1's fine-tuning network plus two encoders that the pre-training never saw."""

    fresh_modules = FRESH_MODULES

    def __init__(self, *, n_primaries: int, n_knots: int, n_b1: int = N_B1) -> None:
        """Build K1's `ft` network, the two new encoders, and the 209-input mixer."""
        super().__init__(
            n_primaries=n_primaries, stage="ft", n_knots=n_knots, n_b1=n_b1, kind="shared"
        )
        self.weather2 = base.mlp(
            n_in=7 + n_b1,
            hidden=m.ENCODER_HIDDEN,
            layers=1,
            n_out=NEW_ENCODER_OUT,
            dropout=m.DROPOUT,
        )
        self.datetime2 = base.mlp(
            n_in=21, hidden=m.ENCODER_HIDDEN, layers=1, n_out=NEW_ENCODER_OUT, dropout=m.DROPOUT
        )
        old = self.mixer_ft[0].in_features  # ty: ignore[not-subscriptable,unresolved-attribute]
        assert old == MIXER_IN_K1, old
        self.mixer_ft = base.mlp(
            n_in=old + 2 * NEW_ENCODER_OUT, hidden=m.MIXER_HIDDEN, layers=2,
            n_out=3 * m.COMPONENTS, dropout=m.DROPOUT,
        )  # fmt: skip
        base.init_head(head=self.mixer_ft, components=m.COMPONENTS)
        assert self.mixer_ft[0].in_features == MIXER_IN_HYBRID3  # ty: ignore[not-subscriptable,unresolved-attribute]

    def reset_new_encoders(self, *, seed: int) -> None:
        """Redraw the two new encoders from `seed`, leaving the global random state as it was."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for part in (self.weather2, self.datetime2):
                for module in part.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs; the mixer also reads the two new encoders' outputs."""
        assert primary is not None
        labels = None
        if x.shape[1] == self.label_col + 2:
            labels, x = x[:, self.label_col :], x[:, : self.label_col]
        emb = self.embedding(primary)
        b1 = x[:, mod.b1_cols(n_b1=self.n_b1)]
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb, b1], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        if labels is not None and self.training:
            aux = self.aux_from(hl=hl)
            ok = torch.isfinite(labels)
            err = torch.where(ok, (aux - torch.nan_to_num(labels)) ** 2, torch.zeros_like(aux))
            m591.STASH["aux"] = err.sum(dim=1) / 2.0
        hw2 = nn.functional.gelu(self.weather2(torch.cat([x[:, m.WEATHER_COLS], b1], dim=-1)))
        hd2 = nn.functional.gelu(self.datetime2(x[:, m.DATETIME_COLS]))
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        out = self.mixer_ft(
            torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS], hw2, hd2], dim=-1)
        )
        off = self.offset(x[:, mod.TAU], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)


def load_pretrained_into_hybrid3(
    *, model: Hybrid3Mdn, state: dict[str, torch.Tensor], seed: int
) -> None:
    """Load a pooled `pre`-stage state dict into `model`, then draw the two new encoders from `seed`.

    The stored pre-training has no `mixer_ft`, `weather2`, or `datetime2`; those start from their own
    draws (the mixer is redrawn by `fresh_mixer`). `knots` becomes the old knots plus the newest one.
    """
    knots = state["knots"]
    rest = {a: v for a, v in state.items() if a != "knots" and not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(rest, strict=False)
    allowed = (*(f"{n}." for n in ("mixer_ft", *FRESH_MODULES)), "knots_old", "knot_new")
    assert all(
        a.startswith(allowed) or a in {"knots_old", "knot_new"} for a in missing.missing_keys
    ), missing
    assert not missing.unexpected_keys, missing
    with torch.no_grad():
        model.knots_old.copy_(knots[:, :-1])  # ty: ignore[call-non-callable]
        model.knot_new.copy_(knots[:, -1])
    model.reset_new_encoders(seed=seed)

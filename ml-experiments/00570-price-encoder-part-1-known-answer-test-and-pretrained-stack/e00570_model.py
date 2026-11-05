# ruff: noqa: E501, ANN001, ANN201
"""Experiment 570's modular lag-free network with a learnt price encoder beside the weather and datetime encoders.

Two extra input columns follow the 37 lag-free columns: the (bundle-offset) grid index of the target
half-hour, and a 0/1 presence flag. The price encoder reads its windows from `self.bundle`, a
`e00570_price.Bundle` set by the fit script. Where the flag is 0 the encoder returns its learned null
vector. Stage `pre` feeds the latent to `mixer_pre`; stage `ft` feeds it to `mixer_ft`.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00570_price as price  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_BASE: Final[int] = m.N_INPUTS
IDX_COL: Final[int] = N_BASE
PRESENT_COL: Final[int] = N_BASE + 1
N_EXTRA: Final[int] = 2
ENCODER_SEED: Final[int] = 570


class PriceMdn(m.ModularMdn):
    """`ModularMdn` whose mixer also reads the price encoder's 16-number latent."""

    def __init__(self, *, n_primaries: int, stage: str) -> None:
        """Build the network; the price encoder is initialised from a fixed seed."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        wide = 2 * m.ENCODER_OUT + m.EMBEDDING + price.LATENT + (0 if stage == "pre" else 7 + 2)
        mixer = base.mlp(n_in=wide, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT)
        base.init_head(head=mixer, components=m.COMPONENTS)
        if stage == "pre":
            self.mixer_pre = mixer
        else:
            self.mixer_ft = mixer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(ENCODER_SEED)
            self.price = price.PriceEncoder()
        self.bundle: price.Bundle | None = None

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return the price latent (B, 16) of the rows."""
        assert self.bundle is not None
        idx = x[:, IDX_COL].long()
        return self.price(self.bundle.windows(idx), idx, x[:, PRESENT_COL])

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb, self.latent(x)], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))


class PlainWithExtras(m.ModularMdn):
    """`ModularMdn` that ignores the two extra columns (arms M0 and A6, which discard the price encoder)."""

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs, reading only the 37 lag-free columns."""
        return super().forward(x[:, :N_BASE], primary)


def load_pretrained(*, model: nn.Module, path: str, fresh_seed: int, keep_price: bool) -> None:
    """Load a pretrained state (weather, datetime, embedding, and optionally the price encoder); give `mixer_ft` a fresh start."""
    raw = torch.load(path, map_location="cpu")
    own = model.state_dict()
    state = {
        a: v
        for a, v in raw.items()
        if a in own
        and not a.startswith(("mixer_ft.", "mixer_pre."))
        and (keep_price or not a.startswith("price."))
    }
    for a, v in state.items():
        assert v.shape == own[a].shape, (a, v.shape, own[a].shape)
    res = model.load_state_dict(state, strict=False)
    assert not res.unexpected_keys, res
    assert all(a.startswith(("mixer_ft.", "mixer_pre.", "price.")) for a in res.missing_keys), res.missing_keys
    if keep_price:
        assert not any(a.startswith("price.") for a in res.missing_keys), res.missing_keys
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fresh_seed)
        fresh = base.mlp(
            n_in=model.mixer_ft[0].in_features,  # ty: ignore[not-subscriptable,unresolved-attribute]
            hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT,
        )  # fmt: skip
        base.init_head(head=fresh, components=m.COMPONENTS)
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]

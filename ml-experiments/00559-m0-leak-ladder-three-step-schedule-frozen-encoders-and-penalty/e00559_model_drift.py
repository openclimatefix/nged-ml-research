# ruff: noqa: E501
"""M0 with a per-Primary drift embedding (arm A10).

Each Primary has one `DIM`-wide embedding at the first of every month from 2019-09 to the fold's month
(`n_knots` knots). A row's drift embedding is the linear interpolation of its Primary's knots at the
row's valid time, with the position clamped to the newest knot (so a forecast for a valid time after
the fold's cutoff reads the newest knot: flat extrapolation). The drift embedding joins the Primary
embedding in the input of the mixer. Column 37 of the input is the row's knot position `tau`.

Pretraining (stage `pre`) learns every knot. Fine-tuning (stage `ft`) keeps the older knots as fixed
buffers and trains only the newest knot, the fresh mixer, and the Primary embedding (the caller freezes
the two encoders). A second-difference penalty keeps the knot trajectory smooth.
"""

from typing import Final

import e00559_model as m
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

DIM: Final[int] = 4
SMOOTH_LAMBDA: Final[float] = 10.0
"""Weight of the mean squared second difference of the knots (fixed in the plan, not tuned)."""
TAU_COL: Final[int] = m.N_INPUTS


class ModularMdnDrift(m.ModularMdn):
    """`ModularMdn` plus a drift embedding read from the knot position in column 37."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; `n_knots` is the number of monthly knots (at least 2)."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        assert n_knots >= 2
        self.n_knots = n_knots
        wide = 2 * m.ENCODER_OUT + m.EMBEDDING + DIM
        self.mixer_pre = m.base.mlp(
            n_in=wide, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT
        )
        self.mixer_ft = m.base.mlp(
            n_in=wide + 7 + 2,
            hidden=m.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        for head in (self.mixer_pre, self.mixer_ft):
            m.base.init_head(head=head, components=m.COMPONENTS)
        if stage == "pre":
            self.knots = nn.Parameter(torch.zeros(n_primaries, n_knots, DIM))
        else:
            self.register_buffer("knots_old", torch.zeros(n_primaries, n_knots - 1, DIM))
            self.knot_new = nn.Parameter(torch.zeros(n_primaries, DIM))

    def all_knots(self) -> torch.Tensor:
        """Return every knot, (Primaries, knots, DIM)."""
        if self.stage == "pre":
            return self.knots
        return torch.cat([self.knots_old, self.knot_new[:, None, :]], dim=1)  # ty: ignore[invalid-argument-type]

    def drift(self, tau: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the interpolated knot embedding of each row (position clamped to the knot range)."""
        knots = self.all_knots()
        tau = tau.clamp(0.0, float(self.n_knots - 1))
        lo = tau.floor().long().clamp(max=self.n_knots - 2)
        frac = (tau - lo.to(tau.dtype))[:, None]
        return (1.0 - frac) * knots[primary, lo] + frac * knots[primary, lo + 1]

    def penalty(self) -> torch.Tensor:
        """Return the smoothness penalty: `SMOOTH_LAMBDA` times the mean squared second difference of the knots."""
        k = self.all_knots()
        return SMOOTH_LAMBDA * ((k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]) ** 2).mean()

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb, self.drift(x[:, TAU_COL], primary)], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))


def load_pre_into_ft(*, model: ModularMdnDrift, state: dict[str, torch.Tensor]) -> None:
    """Load a stage-`pre` state dict into a stage-`ft` network; `mixer_ft` stays as built."""
    assert model.stage == "ft"
    knots = state["knots"]
    rest = {a: v for a, v in state.items() if a != "knots" and not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(rest, strict=False)
    assert all(
        a.startswith("mixer_ft.") or a in {"knots_old", "knot_new"} for a in missing.missing_keys
    ), missing
    assert not missing.unexpected_keys, missing
    with torch.no_grad():
        model.knots_old.copy_(knots[:, :-1])  # ty: ignore[call-non-callable]
        model.knot_new.copy_(knots[:, -1])

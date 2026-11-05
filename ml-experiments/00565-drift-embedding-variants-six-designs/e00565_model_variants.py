# ruff: noqa: E501
"""Experiment 565: the slow-signal network variants of designs 3, 4, and 5.

All three start from experiment 559's `ModularMdn` family and read the knot position `tau` (months
from 2019-09-01, fractional) from column 37, as arm A10 does. Pretraining learns every knot; fine-tuning
keeps the older knots as fixed buffers and trains only the newest knot, as arm A10 does. Every
position is clamped to the newest knot, so a forecast for a valid time after the cutoff reads the
newest knot (flat extrapolation).

- `RandomWalkDrift` (design 3): experiment 559's per-Primary embedding knots, with an L2 penalty on
  the first differences of the knots (a Gaussian random walk) instead of the second differences. With
  `knot_months=3` the knots are quarterly.
- `GlobalTimeDrift` (design 4): one `EMBEDDING`-wide vector per month, shared by all Primaries, added
  to the Primary embedding (before both encoders and the mixer) with a per-Primary scale.
- `OutputOffsetDrift` (design 5): one scalar per Primary and month, added to the three component
  means of the mixture (in standardised units, which are the Primary's target standard deviation),
  with an L2 penalty on the first differences (a slope penalty).
"""

from typing import Final

import e00559_model as m  # ty: ignore[unresolved-import]
import e00559_model_drift as mdr  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

PENALTY_LAMBDA: Final[float] = 10.0
"""Weight of each variant's penalty, the same as arm A10's second-difference weight (fixed in the plan, not tuned)."""
TAU_COL: Final[int] = m.N_INPUTS


class RandomWalkDrift(mdr.ModularMdnDrift):
    """Design 3: per-Primary knots with an L2 penalty on their first differences."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int, knot_months: int = 1) -> None:
        """Build the network; `n_knots` counts monthly knots, `knot_months` is the knot spacing."""
        self.knot_months = knot_months
        self.n_monthly = n_knots
        coarse = -(-(n_knots - 1) // knot_months) + 1
        self.grid_offset = float((n_knots - 1) - knot_months * (coarse - 1))
        """Month of knot 0: knot j sits at month `grid_offset + knot_months * j`, so the newest knot is the cutoff month."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=coarse)

    def drift(self, tau: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the interpolated knot embedding at the position `tau` months."""
        tau = tau.clamp(max=float(self.n_monthly - 1))
        return super().drift((tau - self.grid_offset) / float(self.knot_months), primary)

    def penalty(self) -> torch.Tensor:
        """Return `PENALTY_LAMBDA` times the mean squared first difference of the knots."""
        k = self.all_knots()
        return PENALTY_LAMBDA * ((k[:, 1:] - k[:, :-1]) ** 2).mean()


class _Interpolated(m.ModularMdn):
    """Knots on a month grid, shared code of designs 4 and 5 (`knots` is (rows of knots, n_knots, width))."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, shape: tuple[int, ...]
    ) -> None:
        """Add the knot parameters: all learned in stage `pre`, or old buffers plus one new parameter in `ft`."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        assert n_knots >= 2
        self.n_knots = n_knots
        if stage == "pre":
            self.knots = nn.Parameter(torch.zeros(*shape[:-1], n_knots, shape[-1]))
        else:
            self.register_buffer("knots_old", torch.zeros(*shape[:-1], n_knots - 1, shape[-1]))
            self.knot_new = nn.Parameter(torch.zeros(*shape[:-1], shape[-1]))

    def all_knots(self) -> torch.Tensor:
        """Return every knot, (..., n_knots, width)."""
        if self.stage == "pre":
            return self.knots
        return torch.cat([self.knots_old, self.knot_new[..., None, :]], dim=-2)  # ty: ignore[invalid-argument-type]

    def interpolate(self, tau: torch.Tensor, primary: torch.Tensor | None) -> torch.Tensor:
        """Return the linearly interpolated knot value of each row (position clamped to the knots)."""
        knots = self.all_knots()
        tau = tau.clamp(0.0, float(self.n_knots - 1))
        lo = tau.floor().long().clamp(max=self.n_knots - 2)
        frac = (tau - lo.to(tau.dtype))[:, None]
        if primary is None:
            return (1.0 - frac) * knots[lo] + frac * knots[lo + 1]
        return (1.0 - frac) * knots[primary, lo] + frac * knots[primary, lo + 1]


def _load(*, model: _Interpolated, state: dict[str, torch.Tensor]) -> None:
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
        model.knots_old.copy_(knots[..., :-1, :])  # ty: ignore[call-non-callable]
        model.knot_new.copy_(knots[..., -1, :])


class GlobalTimeDrift(_Interpolated):
    """Design 4: one shared month embedding, added to the Primary embedding with a per-Primary scale."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; the month embedding is `EMBEDDING` wide and the scales start at 1."""
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, shape=(m.EMBEDDING,)
        )
        self.scale = nn.Parameter(torch.ones(n_primaries, 1))

    def penalty(self) -> torch.Tensor:
        """Return `PENALTY_LAMBDA` times the mean squared second difference of the shared knots."""
        k = self.all_knots()
        return PENALTY_LAMBDA * ((k[2:] - 2 * k[1:-1] + k[:-2]) ** 2).mean()

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary) + self.scale[primary] * self.interpolate(x[:, TAU_COL], None)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))


class OutputOffsetDrift(_Interpolated):
    """Design 5: a per-Primary slow offset added to the three component means."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; offsets start at zero."""
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, shape=(n_primaries, 1)
        )

    def penalty(self) -> torch.Tensor:
        """Return `PENALTY_LAMBDA` times the mean squared first difference of the offsets."""
        k = self.all_knots()
        return PENALTY_LAMBDA * ((k[:, 1:] - k[:, :-1]) ** 2).mean()

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs with the offset added to the component means."""
        assert primary is not None
        out = super().forward(x, primary)
        offset = self.interpolate(x[:, TAU_COL], primary)  # (rows, 1)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + offset, out[:, 2 * k :]], dim=-1)


class HybridDrift(mdr.ModularMdnDrift):
    """Designs 1 and 5 together: arm A10's drift embedding plus design 5's per-Primary output offset."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; the offset knots start at zero."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        if stage == "pre":
            self.off_knots = nn.Parameter(torch.zeros(n_primaries, n_knots))
        else:
            self.register_buffer("off_old", torch.zeros(n_primaries, n_knots - 1))
            self.off_new = nn.Parameter(torch.zeros(n_primaries))

    def all_offsets(self) -> torch.Tensor:
        """Return every offset knot, (Primaries, knots)."""
        if self.stage == "pre":
            return self.off_knots
        return torch.cat([self.off_old, self.off_new[:, None]], dim=1)  # ty: ignore[invalid-argument-type]

    def offset(self, tau: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the interpolated offset of each row (position clamped to the knot range), (rows, 1)."""
        knots = self.all_offsets()
        tau = tau.clamp(0.0, float(self.n_knots - 1))
        lo = tau.floor().long().clamp(max=self.n_knots - 2)
        frac = tau - lo.to(tau.dtype)
        return ((1.0 - frac) * knots[primary, lo] + frac * knots[primary, lo + 1])[:, None]

    def penalty(self) -> torch.Tensor:
        """Return arm A10's second-difference penalty plus design 5's first-difference penalty."""
        k = self.all_offsets()
        return super().penalty() + PENALTY_LAMBDA * ((k[:, 1:] - k[:, :-1]) ** 2).mean()

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs with the offset added to the component means."""
        assert primary is not None
        out = super().forward(x, primary)
        offset = self.offset(x[:, TAU_COL], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + offset, out[:, 2 * k :]], dim=-1)


def load_pre_into_ft(*, model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a stage-`pre` state dict into the matching stage-`ft` variant."""
    if isinstance(model, HybridDrift):
        assert model.stage == "ft"
        knots, off = state["knots"], state["off_knots"]
        skip = {"knots", "off_knots"}
        rest = {a: v for a, v in state.items() if a not in skip and not a.startswith("mixer_ft.")}
        missing = model.load_state_dict(rest, strict=False)
        assert all(
            a.startswith("mixer_ft.") or a in {"knots_old", "knot_new", "off_old", "off_new"}
            for a in missing.missing_keys
        ), missing
        assert not missing.unexpected_keys, missing
        with torch.no_grad():
            model.knots_old.copy_(knots[:, :-1])  # ty: ignore[call-non-callable]
            model.knot_new.copy_(knots[:, -1])
            model.off_old.copy_(off[:, :-1])  # ty: ignore[call-non-callable]
            model.off_new.copy_(off[:, -1])
    elif isinstance(model, mdr.ModularMdnDrift):
        mdr.load_pre_into_ft(model=model, state=state)
    else:
        assert isinstance(model, _Interpolated)
        _load(model=model, state=state)

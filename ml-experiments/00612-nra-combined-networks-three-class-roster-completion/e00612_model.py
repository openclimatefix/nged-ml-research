# ruff: noqa: E501, D107, E402, PLW2901
"""Experiment 612: experiment 591's A1 network with one group of inputs removed.

Removal multiplies the inputs by zero, in pre-training, step 2, and the daily fine-tune alike (the
networks are trained from scratch with the input absent), so the layer shapes stay those of A1. As in
experiment 606's `PermLagMdn`, from which the lag-encoder logic is copied.

- `drop_id`: the learnt lag-ID embedding reads zero;
- `drop_age`: the lag age column of each lag reads zero;
- `drop_expected`: the expected-weather and expected-datetime embeddings of the lag's own time read zero;
- `drop_spread`: the seven ensemble-spread columns of the fine-tuning mixer read zero.
"""

from dataclasses import dataclass

import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

AGE_OFFSET = 2
"""Position of the lag age inside a lag's three columns (value, missing flag, age)."""


@dataclass(frozen=True)
class Drops:
    """Which inputs of A1 are removed."""

    drop_id: bool = False
    drop_age: bool = False
    drop_expected: bool = False
    drop_spread: bool = False


class DropMdn(aux.AuxLagMdn):
    """A1's network with the inputs of `drops` set to zero."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, drops: Drops
    ) -> None:
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind="shared"
        )
        self.drops = drops

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the two lags' encoder outputs, day lag first, without the removed inputs."""
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        outs = []
        for j, (cols, hw, hd) in enumerate(((m560.DAY, dw, dd), (m560.WEEK, ww, wd))):
            lag = x[:, cols]
            if self.drops.drop_age:
                lag = torch.cat([lag[:, :AGE_OFFSET], torch.zeros_like(lag[:, AGE_OFFSET:])], dim=1)
            if self.drops.drop_expected:
                hw, hd = torch.zeros_like(hw), torch.zeros_like(hd)
            ident = self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
            if self.drops.drop_id:
                ident = torch.zeros_like(ident)
            outs.append(self.lag[0](torch.cat([lag, hw, hd, emb, ident], dim=-1)))
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs, with the spread columns zeroed when removed."""
        if self.drops.drop_spread:
            x = x.clone()
            x[:, m.SPREAD_COLS] = 0.0
        return super().forward(x, primary)


# ---------------------------------------------------------------- the steadier offset (O2) inside A1

import e00577_model_variants as mv  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

O2_LAMBDA = 3000.0
"""Experiment 577's picked penalty weight for O2 (nats per standardised unit, divided by the training rows)."""


class O2Mdn(DropMdn):
    """A1's network with experiment 577's O2 offset (one knot per day, L1 total-variation penalty).

    The monthly knots of experiment 574's D5 offset stay in the module but are unused. In fine-tuning
    the newest `TAIL` daily knots are trained and the older ones are fixed.
    """

    n_rows: float = 1.0

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, drops: Drops
    ) -> None:
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, drops=drops
        )
        self.n_days = int(mv.MONTH_FIRST_DAY[n_knots - 1]) + 1
        if stage == "pre":
            self.dknots = nn.Parameter(torch.zeros(n_primaries, self.n_days, 1))
        else:
            self.register_buffer("dknots_old", torch.zeros(n_primaries, self.n_days - mv.TAIL, 1))
            self.dknot_new = nn.Parameter(torch.zeros(n_primaries, mv.TAIL, 1))

    def all_dknots(self) -> torch.Tensor:
        """Return every daily knot, (Primaries, days, 1)."""
        if self.stage == "pre":
            return self.dknots
        return torch.cat([self.dknots_old, self.dknot_new], dim=1)  # ty: ignore[invalid-argument-type]

    def penalty(self) -> torch.Tensor:
        """Return `O2_LAMBDA x total variation / n_rows`."""
        k = self.all_dknots()
        return O2_LAMBDA * (k[:, 1:] - k[:, :-1]).abs().sum() / self.n_rows

    def offset(self, tau: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return each row's daily knot, (rows, 1); positions after the newest read the newest knot."""
        idx = mv.day_index(tau=tau).clamp(0, self.n_days - 1)
        return self.all_dknots()[primary, idx]


def load_o2_pre_into_ft(*, model: O2Mdn, state: dict[str, torch.Tensor]) -> None:
    """Load a stage-`pre` O2 state dict into the stage-`ft` network; `mixer_ft` stays as built."""
    assert model.stage == "ft"
    knots = state["dknots"]
    rest = {
        a: v
        for a, v in state.items()
        if a not in {"dknots", "knots"} and not a.startswith("mixer_ft.")
    }
    missing = model.load_state_dict(rest, strict=False)
    allowed = {"knots_old", "knot_new", "dknots_old", "dknot_new"}
    assert all(a.startswith("mixer_ft.") or a in allowed for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    with torch.no_grad():
        model.dknots_old.copy_(knots[:, : -mv.TAIL])  # ty: ignore[call-non-callable]
        model.dknot_new.copy_(knots[:, -mv.TAIL :])

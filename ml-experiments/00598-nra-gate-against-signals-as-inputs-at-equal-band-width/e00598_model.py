# ruff: noqa: E501, D205
"""Experiment 598: experiment 591's A1 network with the gate's signals as extra lag-encoder inputs.

The base is `e00591_model.AuxLagMdn` with the shared lag encoder and no auxiliary loss (arm A1). The
input row is A1's (`label_col` columns: 37 lag-free, 8 lag, the knot position, the B1 columns) followed
by `n_sig` signal columns (`e00598_signals`). The lag encoder reads, beside A1's inputs, the `n_sig`
signal columns and the two lag ages (input columns 39 and 42). The first layer of the lag encoder is
widened by `n_sig + 2` columns that are zero-initialised, and `load_state_dict` pads an A1 state dict
the same way (as experiment 591's A5 does for its detector input), so a freshly loaded network gives
exactly A1's output whatever the signal columns hold.

`n_sig` must not be 2: experiment 591's forward reads two trailing columns as depth labels when the
row is `label_col + 2` wide.
"""

from typing import Final

import e00591_model as aux  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

AGE_COLS: Final[tuple[int, int]] = (39, 42)
"""Input columns holding the day and week lag ages."""
ARMS: Final[tuple[str, ...]] = ("S", "S-g", "S-p")
"""S: signals; S-g: signals plus the G4 state bit; S-p: signals replaced by noise of the same variance."""
LAG_KEY: Final[str] = "lag.0.0.weight"


class SignalLagMdn(aux.AuxLagMdn):
    """A1's network whose lag encoder also reads `n_sig` signal columns and the two lag ages."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, n_sig: int
    ) -> None:  # fmt: skip
        """Build A1's network (shared encoder, no detector), then widen the lag encoder's first layer."""
        assert n_sig != 2, "n_sig == 2 would be read as depth labels"
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind="shared"
        )
        self.n_sig = n_sig
        self.extra = n_sig + len(AGE_COLS)
        old = self.lag[0][0]
        wide = nn.Linear(old.in_features + self.extra, old.out_features)
        with torch.no_grad():
            wide.weight.zero_()
            wide.weight[:, : old.in_features] = old.weight
            wide.bias.copy_(old.bias)
        self.lag[0][0] = wide

    def signal_inputs(self, *, x: torch.Tensor) -> torch.Tensor:
        """Return the signal columns followed by the two lag ages, (rows, n_sig + 2)."""
        sig = x[:, self.label_col : self.label_col + self.n_sig]
        assert sig.shape[1] == self.n_sig, "the signal columns are missing from the input row"
        return torch.cat([sig, x[:, list(AGE_COLS)]], dim=1)

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the two lags' encoder outputs, day lag first, each also reading the signals."""
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        sig = self.signal_inputs(x=x)
        outs = []
        for j, (cols, hw, hd) in enumerate(((m560.DAY, dw, dd), (m560.WEEK, ww, wd))):
            ident = self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
            outs.append(self.lag[0](torch.cat([x[:, cols], hw, hd, emb, ident, sig], dim=-1)))
        return torch.cat(outs, dim=-1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: ANN001, ANN201
        """Load a state dict; an A1 state dict's narrower first lag layer is padded with zero columns."""
        w = state_dict.get(LAG_KEY)
        if w is not None and w.shape[1] == self.lag[0][0].in_features - self.extra:
            state_dict = {**state_dict, LAG_KEY: torch.cat([w, w.new_zeros(w.shape[0], self.extra)], dim=1)}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def shapes_from_state(*, state: dict[str, torch.Tensor]) -> dict[str, int]:
    """Return `n_primaries`, `n_knots`, and `n_b1` of the network that wrote an A1 or S state dict."""
    n_primaries = state["embedding.weight"].shape[0]
    n_knots = state["knots_old"].shape[1] + 1
    n_b1 = state["weather.0.weight"].shape[1] - 7 - m545.EMBEDDING
    return {"n_primaries": n_primaries, "n_knots": n_knots, "n_b1": n_b1}

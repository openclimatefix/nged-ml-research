# ruff: noqa: E501
"""Experiment 606: experiment 591's A1 network with a chosen set of inputs to the shared lag encoder.

The base is `e00591_model.AuxLagMdn` with the shared lag encoder and no auxiliary loss (arm A1). The
lag encoder reads, for each of the two lags: the lag's value, missing flag, and age
(`x[:, DAY]` or `x[:, WEEK]`), the expected-weather and expected-datetime embeddings of the lag's own
time, the Primary embedding, and the learnt 8-number lag-ID embedding. An `InputSet` removes inputs
from that list (by multiplying them by zero, so the A1 weights and the layer shape stay) and appends
`n_extra` columns of `e00610_signals` to the encoder's input. The first layer is widened by `n_extra`
columns that start at zero, and `load_state_dict` pads an A1 state dict the same way, so a freshly
loaded network with no input removed gives exactly A1's output whatever the extra columns hold.

`n_extra` must not be 2: experiment 591's forward reads two trailing columns as depth labels when the
row is `label_col + 2` wide.
"""

from dataclasses import dataclass
from typing import Final

import e00560_model as m560  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

LAG_KEY: Final[str] = "lag.0.0.weight"
AGE_OFFSET: Final[int] = 2
"""Position of the lag age inside a lag's three columns (value, missing flag, age)."""


@dataclass(frozen=True)
class InputSet:
    """Which of A1's lag-encoder inputs are kept, and how many extra columns are appended."""

    use_id: bool = True
    use_age: bool = True
    use_expected: bool = True
    n_extra: int = 0


class PermLagMdn(aux.AuxLagMdn):
    """A1's network whose lag encoder reads the inputs of `inputs`."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, inputs: InputSet
    ) -> None:  # fmt: skip
        """Build A1's network (shared encoder, no detector), then widen the lag encoder's first layer."""
        assert inputs.n_extra != 2, "n_extra == 2 would be read as depth labels"
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind="shared"
        )
        self.inputs = inputs
        if inputs.n_extra:
            old = self.lag[0][0]
            wide = nn.Linear(old.in_features + inputs.n_extra, old.out_features)
            with torch.no_grad():
                wide.weight.zero_()
                wide.weight[:, : old.in_features] = old.weight
                wide.bias.copy_(old.bias)
            self.lag[0][0] = wide

    def extra_inputs(self, *, x: torch.Tensor) -> torch.Tensor:
        """Return the extra columns, (rows, n_extra), which follow the A1 row (`label_col` columns)."""
        extra = x[:, self.label_col : self.label_col + self.inputs.n_extra]
        assert extra.shape[1] == self.inputs.n_extra, "the extra columns are missing from the row"
        return extra

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the two lags' encoder outputs, day lag first, each reading the chosen inputs."""
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        extra = self.extra_inputs(x=x) if self.inputs.n_extra else None
        outs = []
        for j, (cols, hw, hd) in enumerate(((m560.DAY, dw, dd), (m560.WEEK, ww, wd))):
            lag = x[:, cols]
            if not self.inputs.use_age:
                lag = torch.cat([lag[:, :AGE_OFFSET], torch.zeros_like(lag[:, AGE_OFFSET:])], dim=1)
            w_in, d_in = (
                (hw, hd)
                if self.inputs.use_expected
                else (torch.zeros_like(hw), torch.zeros_like(hd))
            )
            ident = self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
            if not self.inputs.use_id:
                ident = torch.zeros_like(ident)
            parts = [lag, w_in, d_in, emb, ident]
            if extra is not None:
                parts.append(extra)
            outs.append(self.lag[0](torch.cat(parts, dim=-1)))
        return torch.cat(outs, dim=-1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: ANN001, ANN201
        """Load a state dict; an A1 state dict's narrower first lag layer is padded with zero columns."""
        w = state_dict.get(LAG_KEY)
        n = self.inputs.n_extra
        if n and w is not None and w.shape[1] == self.lag[0][0].in_features - n:
            state_dict = {**state_dict, LAG_KEY: torch.cat([w, w.new_zeros(w.shape[0], n)], dim=1)}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

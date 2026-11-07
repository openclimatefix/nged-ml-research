# ruff: noqa: E501
"""Experiment 612 (copy of 610): experiment 591's A1 network with extra lag-encoder columns and two options.

The options are a scalar bottleneck and a load-shift label head.

Copied from experiment 606's `e00606_model.py`. The base is `e00591_model.AuxLagMdn` with the shared
lag encoder and no auxiliary loss (arm A1). The lag encoder reads, for each of the two lags: the lag's
value, missing flag, and age, the expected-weather and expected-datetime embeddings of the lag's own
time, the Primary embedding, and the learnt 8-number lag-ID embedding. The first layer is widened by
the extra inputs, which start at zero weight, and `load_state_dict` pads an A1 state dict the same
way, so a freshly loaded network gives exactly A1's output whatever the extra columns hold.

What this file adds to 606's network:

- `bottleneck`: the `n_extra` signal columns pass through a small network that returns one number, and
  the lag encoder reads only that number (one extra input per lag).
- `label_head`: a head on each lag encoder's (activated) output that predicts whether the lag lies
  inside a synthetic load-shift (a logit) and the shift depth as a share of capacity. Fine-tuning rows
  carry four trailing label columns (day active, day depth, week active, week depth; NaN where a lag
  has no label). The network reads them only to set `STASH["shift"]`, the per-row label loss, and a
  forecast row has no label columns, so the labels cannot reach a forecast.

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
BOTTLENECK_HIDDEN: Final[int] = 16
LABEL_COLUMNS: Final[int] = 4
STASH: dict[str, torch.Tensor] = {}
"""Per-row label loss of the last training-mode forward pass that carried label columns."""


@dataclass(frozen=True)
class InputSet:
    """How many extra columns are appended, and the optional bottleneck and label head."""

    n_extra: int = 0
    bottleneck: bool = False
    label_head: bool = False

    @property
    def n_encoder_extra(self) -> int:
        """Return how many extra inputs the lag encoder's first layer reads."""
        return 1 if self.bottleneck else self.n_extra


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
            wide = nn.Linear(old.in_features + inputs.n_encoder_extra, old.out_features)
            with torch.no_grad():
                wide.weight.zero_()
                wide.weight[:, : old.in_features] = old.weight
                wide.bias.copy_(old.bias)
            self.lag[0][0] = wide
        if inputs.bottleneck:
            self.bottle = nn.Sequential(
                nn.Linear(inputs.n_extra, BOTTLENECK_HIDDEN),
                nn.GELU(),
                nn.Linear(BOTTLENECK_HIDDEN, 1),
            )
        if inputs.label_head:
            self.shift_head = nn.Sequential(
                nn.Linear(m560.LAG_OUT, aux.HEAD_HIDDEN), nn.GELU(), nn.Linear(aux.HEAD_HIDDEN, 2)
            )
        self._hl_raw: torch.Tensor | None = None

    def extra_inputs(self, *, x: torch.Tensor) -> torch.Tensor:
        """Return what the lag encoder reads beyond A1's row: the extra columns, or their bottleneck scalar."""
        extra = x[:, self.label_col : self.label_col + self.inputs.n_extra]
        assert extra.shape[1] == self.inputs.n_extra, "the extra columns are missing from the row"
        return self.bottle(extra) if self.inputs.bottleneck else extra

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the two lags' encoder outputs, day lag first, each reading A1's inputs and the extras."""
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        extra = self.extra_inputs(x=x) if self.inputs.n_extra else None
        outs = []
        for j, (cols, hw, hd) in enumerate(((m560.DAY, dw, dd), (m560.WEEK, ww, wd))):
            ident = self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
            parts = [x[:, cols], hw, hd, emb, ident]
            if extra is not None:
                parts.append(extra)
            outs.append(self.lag[0](torch.cat(parts, dim=-1)))
        out = torch.cat(outs, dim=-1)
        self._hl_raw = out
        return out

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs; with label columns in training mode also set `STASH['shift']`."""
        width = self.label_col + self.inputs.n_extra
        labels = None
        if self.inputs.label_head and x.shape[1] == width + LABEL_COLUMNS:
            labels, x = x[:, width:], x[:, :width]
        out = super().forward(x, primary)
        if labels is not None and self.training:
            assert self._hl_raw is not None
            STASH["shift"] = self.label_loss(hl=nn.functional.gelu(self._hl_raw), labels=labels)
        return out

    def label_loss(self, *, hl: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return the per-row label loss: cross-entropy of "active" plus squared depth error on shifted lags."""
        w = m560.LAG_OUT
        total = hl.new_zeros(len(hl))
        for j in range(2):
            pred = self.shift_head(hl[:, j * w : (j + 1) * w])
            active, depth = labels[:, 2 * j], labels[:, 2 * j + 1]
            ok = torch.isfinite(active)
            bce = nn.functional.binary_cross_entropy_with_logits(
                pred[:, 0], torch.nan_to_num(active), reduction="none"
            )
            shifted = ok & (active > 0.5)
            sq = (pred[:, 1] - torch.nan_to_num(depth)) ** 2
            total = total + torch.where(ok, bce, torch.zeros_like(bce))
            total = total + torch.where(shifted, sq, torch.zeros_like(sq))
        return total / 2.0

    def shift_probability(self, *, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the head's probability that each lag is inside a shift, (rows, 2), for rows without labels."""
        assert self.inputs.label_head
        emb = self.embedding(primary)
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        w = m560.LAG_OUT
        return torch.stack(
            [torch.sigmoid(self.shift_head(hl[:, j * w : (j + 1) * w])[:, 0]) for j in range(2)],
            dim=1,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: ANN001, ANN201
        """Load a state dict; an A1 state dict's narrower first lag layer is padded with zero columns."""
        w = state_dict.get(LAG_KEY)
        n = self.inputs.n_encoder_extra if self.inputs.n_extra else 0
        if n and w is not None and w.shape[1] == self.lag[0][0].in_features - n:
            state_dict = {**state_dict, LAG_KEY: torch.cat([w, w.new_zeros(w.shape[0], n)], dim=1)}
            missing = super().load_state_dict(state_dict, strict=False, assign=assign)
            allowed = ("bottle.", "shift_head.")
            assert all(m.startswith(allowed) for m in missing.missing_keys), missing
            assert not missing.unexpected_keys, missing
            return missing
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

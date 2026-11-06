# ruff: noqa: E501
"""Experiment 591: experiment 574's P5 network with a shared (or separate) lag encoder and an auxiliary depth head.

The base is `e00574_model.PartDMdn` (lag encoder, D5 output offset, B1 weather columns). As in
experiment 586, the lag encoder is replaced by one that reads one lag at a time with that lag's
expected-weather embeddings, and both mixers are widened to take the two encoder outputs.

- `shared`: one encoder with shared weights and a learnt 8-number lag-ID embedding (experiment 586's S1);
- `separate`: one encoder per lag, no lag ID (experiment 586's S3).

Each encoder has an auxiliary head: a small network that reads the encoder's (activated) output and
predicts the signed depth of the synthetic load-shift in that lag, as a share of capacity. The head is
used only by the training loss (and by the probe), and by the mixer in the detector arm A5.

Fine-tuning tensors carry two extra trailing columns (the day and week depth labels, NaN where a lag
has no label). The network reads them only in training mode, to build `last_aux_loss`; at forecast time
the extra columns are absent, so the labels cannot reach a forecast.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00574_model as mod  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

KINDS: Final[tuple[str, ...]] = ("shared", "separate")
LAG_ID_DIM: Final[int] = 8
HEAD_HIDDEN: Final[int] = 32
STASH: dict[str, torch.Tensor] = {}
"""Per-row auxiliary squared error of the last training-mode forward pass that carried labels."""


class AuxLagMdn(mod.PartDMdn):
    """P5's network with the lag encoder of `kind`, auxiliary depth heads, and an optional detector input."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, kind: str,
        detector: bool = False,
    ) -> None:  # fmt: skip
        """Build the network, then replace the lag encoder and widen both mixers."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1)
        assert kind in KINDS
        self.kind, self.detector = kind, detector
        e = m545.EMBEDDING
        id_dim = LAG_ID_DIM if kind == "shared" else 0
        n_enc = 1 if kind == "shared" else 2
        self.lag = nn.ModuleList(
            [
                base.mlp(
                    n_in=3 + 2 * m545.ENCODER_OUT + e + id_dim, hidden=m560.LAG_HIDDEN, layers=1,
                    n_out=m560.LAG_OUT, dropout=m545.DROPOUT,
                )
                for _ in range(n_enc)
            ]
        )  # fmt: skip
        if kind == "shared":
            self.lag_id = nn.Embedding(2, LAG_ID_DIM)
        self.head = nn.ModuleList(
            [
                base.mlp(n_in=m560.LAG_OUT, hidden=HEAD_HIDDEN, layers=1, n_out=1, dropout=0.0)
                for _ in range(n_enc)
            ]
        )
        wide = 2 * m545.ENCODER_OUT + 2 * m560.LAG_OUT + e
        self.mixer_pre = base.mlp(
            n_in=wide, hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT,
        )  # fmt: skip
        self.mixer_ft = base.mlp(
            n_in=wide + 7 + 2 + (2 if detector else 0), hidden=m545.MIXER_HIDDEN, layers=2,
            n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT,
        )  # fmt: skip
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=m.COMPONENTS)
        self.label_col = mod.TAU + 1 + n_b1

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the two lags' encoder outputs, day lag first."""
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        outs = []
        for j, (cols, hw, hd) in enumerate(((m560.DAY, dw, dd), (m560.WEEK, ww, wd))):
            parts = [x[:, cols], hw, hd, emb]
            if self.kind == "shared":
                parts.append(
                    self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
                )
            outs.append(self.lag[0 if self.kind == "shared" else j](torch.cat(parts, dim=-1)))
        return torch.cat(outs, dim=-1)

    def aux_from(self, *, hl: torch.Tensor) -> torch.Tensor:
        """Return the two heads' depth predictions, (rows, 2), from the activated encoder outputs."""
        w = m560.LAG_OUT
        outs = [
            self.head[0 if self.kind == "shared" else j](hl[:, j * w : (j + 1) * w])
            for j in range(2)
        ]
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs; in training mode with label columns also set `STASH['aux']`."""
        assert primary is not None
        labels = None
        if x.shape[1] == self.label_col + 2:
            labels, x = x[:, self.label_col :], x[:, : self.label_col]
        emb = self.embedding(primary)
        weather_in = [x[:, m.WEATHER_COLS], emb]
        if self.n_b1:
            weather_in.append(x[:, mod.b1_cols(n_b1=self.n_b1)])
        hw = nn.functional.gelu(self.weather(torch.cat(weather_in, dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        need_aux = self.detector or (labels is not None and self.training)
        aux = self.aux_from(hl=hl) if need_aux else None
        if labels is not None and self.training:
            ok = torch.isfinite(labels)
            err = torch.where(ok, (aux - torch.nan_to_num(labels)) ** 2, torch.zeros_like(aux))  # ty: ignore[unsupported-operator, invalid-argument-type]
            STASH["aux"] = err.sum(dim=1) / 2.0
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        if self.stage == "pre":
            out = self.mixer_pre(core)
        else:
            parts = [core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]]
            if self.detector:
                parts.append(aux)  # ty: ignore[invalid-argument-type]
            out = self.mixer_ft(torch.cat(parts, dim=-1))  # ty: ignore[invalid-argument-type]
        off = self.offset(x[:, mod.TAU], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: ANN001, ANN201
        """Load a state dict; a detector network pads a narrower first mixer layer with zero columns.

        This starts arm A5 from arm A2's step-2 weights with two zero-initialised detector inputs, so it
        begins as the same network as A2.
        """
        key = "mixer_ft.0.weight"
        if (
            self.detector
            and key in state_dict
            and state_dict[key].shape[1] == self.mixer_ft[0].in_features - 2
        ):  # ty: ignore[unresolved-attribute]
            w = state_dict[key]
            state_dict = {**state_dict, key: torch.cat([w, w.new_zeros(w.shape[0], 2)], dim=1)}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def predict_aux(self, *, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the heads' depth predictions for rows without label columns (probe)."""
        assert x.shape[1] == self.label_col
        emb = self.embedding(primary)
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        return self.aux_from(hl=hl)

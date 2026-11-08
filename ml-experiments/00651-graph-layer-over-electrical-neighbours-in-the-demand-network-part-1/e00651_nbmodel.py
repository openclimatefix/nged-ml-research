# ruff: noqa: E501
"""Experiment 651 (copy of 613): experiment 591's A1 network with extra lag-encoder columns and two options.

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


# ======================================================================= experiment 651: the graph layer

import numpy as np  # noqa: E402

MSG_HEADS: Final[int] = 2
MSG_KEY: Final[int] = 16
STATE_DIM: Final[int] = 2 * 32 + 2 * m560.LAG_OUT + 8
"""A node's state: weather (32), date-time (32), the two lag-encoder outputs (2 x 32), Primary embedding (8)."""
UPDATE_DIM: Final[int] = STATE_DIM - 8
"""The graph layer updates weather, date-time, and lag parts, never the Primary embedding."""
GRAPH_KINDS: Final[tuple[str, ...]] = ("none", "gcn", "gat", "flow")
MESSAGE_DROP: Final[float] = 0.5
FLOW_HIDDEN: Final[int] = 32
FLOW_CLIP: Final[float] = 8.0
AUX_FLOW_W: Final[float] = 0.1
FLOW_DEPTH_SCALE: Final[float] = 0.2
"""Typical synthetic shift depth (share of capacity); the auxiliary squared error is divided by its square."""


class Bank:
    """The neighbour-visible view of the rows a forward pass addresses (set by the arm script).

    `x[i]` is row i's input as its neighbours see it (a mirrored shift in the lags and signals),
    `p[i]` its Primary, and `nbr[i, k]` the bank row of row i's k-th neighbour at the same origin and
    valid time, or -1. Rows of the batch carry their own bank row number in the last input column.
    """

    x: torch.Tensor | None = None
    p: torch.Tensor | None = None
    nbr: torch.Tensor | None = None


BANK = Bank()


class GraphLagMdn(PermLagMdn):
    """The class 2 network with a graph layer between the encoders and the mixer.

    The input row is `[A1 columns, extra columns, (4 label columns, FLOW only, training only), row id]`.
    `kind` is `gcn`, `gat`, or `flow`. The layer's output weights start at zero, so the network begins
    as the own-signal network. Messages are hidden on a share `MESSAGE_DROP` of rows in training mode.
    """

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, inputs: InputSet,
        kind: str, degree: np.ndarray, capacity: np.ndarray, target_std: np.ndarray,
        gap_cols: tuple[int, int] = (0, 0), n_neighbours: int = 0,
        lag_std: np.ndarray = np.ones(2, np.float32),
    ) -> None:  # fmt: skip
        assert kind in GRAPH_KINDS and kind != "none"
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, inputs=inputs
        )
        self.kind = kind
        self.register_buffer("degree", torch.as_tensor(degree, dtype=torch.float32))
        self.register_buffer("capacity", torch.as_tensor(capacity, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))
        self.register_buffer("lag_std", torch.as_tensor(lag_std, dtype=torch.float32))
        self.gap_cols = gap_cols
        self.n_neighbours = n_neighbours
        self._corr: torch.Tensor | None = None
        if kind == "gcn":
            self.graph_out = nn.Linear(STATE_DIM, UPDATE_DIM, bias=False)
            nn.init.zeros_(self.graph_out.weight)
        elif kind == "gat":
            self.graph_q = nn.Linear(UPDATE_DIM, MSG_HEADS * MSG_KEY)
            self.graph_k = nn.Linear(STATE_DIM, MSG_HEADS * MSG_KEY)
            self.graph_v = nn.Linear(STATE_DIM, MSG_HEADS * 32)
            self.graph_gate = nn.Linear(UPDATE_DIM + MSG_HEADS * 32, 1)
            self.graph_out = nn.Linear(MSG_HEADS * 32, UPDATE_DIM, bias=False)
            nn.init.zeros_(self.graph_out.weight)
        else:
            self.flow_net = nn.Sequential(
                nn.Linear(2 + 2 + 1, FLOW_HIDDEN), nn.GELU(), nn.Linear(FLOW_HIDDEN, 2)
            )
            self.flow_scale = nn.Parameter(torch.zeros(1))
            old = self.lag[0][0]
            wide = nn.Linear(old.in_features + 1, old.out_features)
            with torch.no_grad():
                wide.weight.zero_()
                wide.weight[:, : old.in_features] = old.weight
                wide.bias.copy_(old.bias)
            self.lag[0][0] = wide

    # ------------------------------------------------------------------ states

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the two lags' encoder outputs; FLOW reads the corrected lag value and the correction."""
        if self.kind != "flow" or self._corr is None:
            return super().lag_embedding(x=x, primary=primary, emb=emb)
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        extra = self.extra_inputs(x=x) if self.inputs.n_extra else None
        outs = []
        for j, (cols, hw, hd) in enumerate(((m560.DAY, dw, dd), (m560.WEEK, ww, wd))):
            lag = x[:, cols].clone()
            lag[:, 0] = lag[:, 0] + self._corr[:, j] * (1.0 - lag[:, 1])  # no change where missing
            ident = self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
            parts = [lag, hw, hd, emb, ident]
            if extra is not None:
                parts.append(extra)
            parts.append(self._corr[:, j : j + 1])
            outs.append(self.lag[0](torch.cat(parts, dim=-1)))
        return torch.cat(outs, dim=-1)

    def states(self, *, x: torch.Tensor, primary: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return a row's (weather, date-time, lag, embedding) states, as experiment 591's forward builds them."""
        emb = self.embedding(primary)
        weather_in = [x[:, aux.m.WEATHER_COLS], emb]
        if self.n_b1:
            weather_in.append(x[:, aux.mod.b1_cols(n_b1=self.n_b1)])
        hw = nn.functional.gelu(self.weather(torch.cat(weather_in, dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, aux.m.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        return hw, hd, hl, emb

    def neighbour_states(
        self, *, rid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the neighbours' (states, primaries, valid mask), each (batch, K, ...), from the bank."""
        assert BANK.x is not None and BANK.p is not None and BANK.nbr is not None
        idx = BANK.nbr[rid]
        valid = idx >= 0
        flat = idx[valid]
        xn, pn = BANK.x[flat], BANK.p[flat]
        saved = self._corr
        self._corr = None  # a neighbour's own flow correction is not part of its message
        hn = torch.cat(self.states(x=xn, primary=pn), dim=-1)
        self._corr = saved
        out = hn.new_zeros(idx.shape[0], idx.shape[1], hn.shape[-1])
        out[valid] = hn
        prim = pn.new_zeros(idx.shape[0], idx.shape[1])
        prim[valid] = pn
        return out, prim, valid

    # ------------------------------------------------------------------ the layers

    def update_gcn(
        self, *, p: torch.Tensor, hn: torch.Tensor, pn: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Return U x (sum over neighbours of h_j / sqrt(d_i d_j))."""
        di, dj = self.degree[p][:, None], self.degree[pn]
        w = valid.float() / torch.sqrt(di.clamp(min=1.0) * dj.clamp(min=1.0))
        return self.graph_out((w[..., None] * hn).sum(dim=1))

    def update_gat(self, *, h: torch.Tensor, hn: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Return U x (sigmoid gate x multi-head attention over the valid neighbours)."""
        b, k, _ = hn.shape
        q = self.graph_q(h[:, :UPDATE_DIM]).view(b, 1, MSG_HEADS, MSG_KEY)
        key = self.graph_k(hn).view(b, k, MSG_HEADS, MSG_KEY)
        val = self.graph_v(hn).view(b, k, MSG_HEADS, 32)
        score = (q * key).sum(-1) / MSG_KEY**0.5
        score = score.masked_fill(~valid[..., None], -1e9)
        attn = torch.softmax(score, dim=1) * valid[..., None].float()
        agg = (attn[..., None] * val).sum(dim=1).reshape(b, MSG_HEADS * 32)
        gate = torch.sigmoid(self.graph_gate(torch.cat([h[:, :UPDATE_DIM], agg], dim=-1)))
        return self.graph_out(gate * agg)

    def flow_correction(
        self, *, x: torch.Tensor, p: torch.Tensor, rid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-node lag correction in capacity share, (batch, 2), and the flows' antisymmetry error.

        Each edge's flow in MW is `g(i, j) cap_i - g(j, i) cap_j`, so a flow leaves one node and arrives at
        the other. The correction of node i is minus its net outflow over its capacity, times a learnt scale
        that starts at zero.
        """
        assert BANK.x is not None and BANK.p is not None and BANK.nbr is not None
        idx = BANK.nbr[rid]
        valid = idx >= 0
        xn, pn = BANK.x[idx.clamp(min=0)], BANK.p[idx.clamp(min=0)]  # (b, k, w), (b, k)
        c0, c1 = (self.label_col + g for g in self.gap_cols)
        gi = x[:, c0], x[:, c1]
        gj0, gj1 = xn[..., c0], xn[..., c1]
        ci, cj = self.capacity[p][:, None], self.capacity[pn]
        ratio = torch.log(ci / cj)
        gi_b = torch.stack([gi[0], gi[1]], dim=-1)[:, None, :].expand(-1, idx.shape[1], -1)
        gj = torch.stack([gj0, gj1], dim=-1)
        fwd = self.flow_net(torch.cat([gi_b, gj, ratio[..., None]], dim=-1))
        rev = self.flow_net(torch.cat([gj, gi_b, -ratio[..., None]], dim=-1))
        mw = fwd * ci[..., None] - rev * cj[..., None]
        mw = torch.where(valid[..., None], mw, torch.zeros_like(mw))
        share = -mw.sum(dim=1) / self.capacity[p][:, None]
        return self.flow_scale * share, mw

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs; the last input column is the bank row number."""
        assert primary is not None
        rid = x[:, -1].long()
        x = x[:, :-1]
        width = self.label_col + self.inputs.n_extra
        labels = None
        if x.shape[1] == width + LABEL_COLUMNS:
            labels, x = x[:, width:], x[:, :width]
        assert x.shape[1] == width, (x.shape, width)
        drop = None
        if self.training:
            drop = (torch.rand(len(x), 1, device=x.device) >= MESSAGE_DROP).float()
        self._corr = None
        if self.kind == "flow":
            corr_full, _ = self.flow_correction(x=x, p=primary, rid=rid)
            corr = corr_full * drop if drop is not None else corr_full
            self._corr = (corr / self.lag_std[None, :]).clamp(-FLOW_CLIP, FLOW_CLIP)
            if labels is not None and self.training:
                depth = labels[:, 1::2]  # the shift added to each lag, as a share of capacity
                ok = torch.isfinite(depth)
                err = torch.where(
                    ok,
                    (corr_full + torch.nan_to_num(depth)) ** 2 / FLOW_DEPTH_SCALE**2,
                    torch.zeros_like(corr),
                )  # on every row, messages dropped or not
                STASH["shift"] = err.sum(dim=1) / 2.0  # weighted by `AUX_FLOW_W` in `row_nll`
        hw, hd, hl, emb = self.states(x=x, primary=primary)
        if self.kind in ("gcn", "gat"):
            hn, pn, valid = self.neighbour_states(rid=rid)
            h = torch.cat([hw, hd, hl], dim=-1)
            if self.kind == "gcn":
                upd = self.update_gcn(p=primary, hn=hn, pn=pn, valid=valid)
            else:
                upd = self.update_gat(h=h, hn=hn, valid=valid)
            if drop is not None:
                upd = upd * drop
            h = h + upd
            hw, hd, hl = h[:, :32], h[:, 32:64], h[:, 64:]
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        parts = [core, x[:, aux.m.SPREAD_COLS], x[:, aux.m.LEAD_COLS]]
        out = self.mixer_ft(torch.cat(parts, dim=-1))
        off = self.offset(x[:, aux.mod.TAU], primary)
        k = aux.m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: ANN001, ANN201
        """Load an A1 or own-signal state dict: narrower lag columns are padded with zeros, graph keys stay as built."""
        w = state_dict.get(LAG_KEY)
        have = self.lag[0][0].in_features
        if w is not None and w.shape[1] < have:
            state_dict = {
                **state_dict,
                LAG_KEY: torch.cat([w, w.new_zeros(w.shape[0], have - w.shape[1])], dim=1),
            }
        result = nn.Module.load_state_dict(self, state_dict, strict=False, assign=assign)
        allowed = (
            "bottle.",
            "shift_head.",
            "graph_",
            "flow_",
            "degree",
            "capacity",
            "target_std",
            "lag_std",
        )
        assert all(m.startswith(allowed) for m in result.missing_keys), result
        assert not result.unexpected_keys, result
        if strict and w is not None and w.shape[1] == have:
            pass  # a state saved from this network has every key, which the check above allows
        return result

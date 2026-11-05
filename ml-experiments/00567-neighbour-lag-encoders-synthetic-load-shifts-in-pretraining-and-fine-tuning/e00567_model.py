# ruff: noqa: E501, ANN001, ANN201
"""The modular demand network of experiment 567: experiment 560's lag encoder (variant 3) plus neighbour lag encoders.

Input columns follow experiment 560's 45 (37 lag-free columns, then the own day and week lag value,
missing flag, and age in 37-42, then the grid index of the day and week lag times in 43-44). New:

- 45-68: for each of 4 neighbour slots `j`, six columns at `45 + 6 j`: the neighbour's day lag value,
  missing flag, age, then its week lag value, missing flag, age (same lag times as the own lags);
- 69-72: the neighbour's Primary index as a float, -1 where the slot is empty.

Flags (the arms):

- `use_nb=False`: the network of experiment 560 (variant 3); columns 45 and up are ignored (arm N0);
- `use_nb=True`: each present neighbour's lags, the expected weather and datetime embeddings at the
  lag's own time and place (copies of the weather and datetime encoders, same weights, run on that
  neighbour's table row and Primary embedding), the neighbour's Primary embedding, and the target
  Primary's embedding go through one neighbour encoder; the K outputs are pooled by mean and by
  attention over the target's embedding, masked for empty slots, and joined to the mixer's inputs;
- `shared=False`: one neighbour encoder per target Primary (arm N2);
- `aux=True`: a linear head on the mixer's last hidden layer gives, for each of the 10 lag slots (own day
  and week, then each neighbour's day and week), one flag logit and one signed fraction (arm N1-aux).
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

K: Final[int] = 4
NB0: Final[int] = 45
NB_ID0: Final[int] = 69
N_INPUTS: Final[int] = 73
SLOTS: Final[int] = 2 + 2 * K
NB_OUT: Final[int] = 32
NB_HIDDEN: Final[int] = 64


class NeighbourLagMdn(m560.LagModularMdn):
    """Experiment 560's variant 3 with optional neighbour lag encoders and an auxiliary flag head."""

    def __init__(
        self, *, n_primaries: int, stage: str, use_nb: bool, shared: bool = True, aux: bool = False
    ) -> None:
        """Build the network; `stage` is `pre` or `ft` as in experiment 560."""
        super().__init__(n_primaries=n_primaries, mode="v3", stage=stage)
        self.use_nb, self.shared, self.aux = use_nb, shared, aux
        e = m545.EMBEDDING
        if use_nb:
            tok = 6 + 4 * m545.ENCODER_OUT + 2 * e
            n_enc = 1 if shared else n_primaries
            self.nb_enc = nn.ModuleList(
                [
                    base.mlp(n_in=tok, hidden=NB_HIDDEN, layers=1, n_out=NB_OUT, dropout=m545.DROPOUT)
                    for _ in range(n_enc)
                ]
            )
            self.nb_att = nn.Linear(NB_OUT + e, 1)
            wide = 2 * m545.ENCODER_OUT + self.n_lag_out + e + 2 * NB_OUT
            self.mixer_pre = base.mlp(
                n_in=wide, hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT
            )
            self.mixer_ft = base.mlp(
                n_in=wide + 7 + 2, hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS, dropout=m545.DROPOUT
            )
            for head in (self.mixer_pre, self.mixer_ft):
                base.init_head(head=head, components=m545.COMPONENTS)
        if aux:
            self.aux_head = nn.Linear(m545.MIXER_HIDDEN, 2 * SLOTS)
            nn.init.zeros_(self.aux_head.weight)
            nn.init.zeros_(self.aux_head.bias)

    def neighbour_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the pooled neighbour encoding (rows x 2 * NB_OUT), zero for rows with no neighbour."""
        ids = x[:, NB_ID0 : NB_ID0 + K].round().long()
        present = ids >= 0
        tokens = []
        for j in range(K):
            nid = ids[:, j].clamp(min=0)
            emb_n = self.embedding(nid)
            (dw, dd), (ww, wd) = self._expected(x=x, primary=nid, emb=emb_n)
            lag = x[:, NB0 + 6 * j : NB0 + 6 * j + 6]
            tokens.append(torch.cat([lag, dw, dd, ww, wd, emb_n, emb], dim=-1))
        tok = torch.stack(tokens, dim=1)
        rows = tok.reshape(-1, tok.shape[-1])
        if self.shared:
            h = self.nb_enc[0](rows)
        else:
            h = torch.zeros(len(rows), NB_OUT, device=rows.device, dtype=rows.dtype)
            owner = primary.repeat_interleave(K)
            for q in torch.unique(owner).tolist():
                sel = owner == q
                h[sel] = self.nb_enc[q](rows[sel])
        h = nn.functional.gelu(h).reshape(len(x), K, NB_OUT)
        h = h * present[:, :, None]
        n = present.sum(dim=1, keepdim=True).clamp(min=1)
        mean = h.sum(dim=1) / n
        score = self.nb_att(torch.cat([h, emb[:, None, :].expand(-1, K, -1)], dim=-1)).squeeze(-1)
        score = score.masked_fill(~present, float("-inf"))
        weight = torch.softmax(score.masked_fill(~present.any(dim=1, keepdim=True), 0.0), dim=1)
        weight = weight * present
        attended = (weight[:, :, None] * h).sum(dim=1)
        return torch.cat([mean, attended], dim=-1)

    def _core(self, x: torch.Tensor, primary: torch.Tensor) -> tuple[torch.Tensor, nn.Sequential]:
        """Return the mixer's input and the mixer for the stage."""
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m545.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m545.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        parts = [hw, hd, hl, emb]
        if self.use_nb:
            parts.append(self.neighbour_embedding(x=x, primary=primary, emb=emb))
        core = torch.cat(parts, dim=-1)
        if self.stage == "pre":
            return core, self.mixer_pre
        return torch.cat([core, x[:, m545.SPREAD_COLS], x[:, m545.LEAD_COLS]], dim=-1), self.mixer_ft

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        inp, mixer = self._core(x, primary)
        return mixer(inp)

    def forward_aux(
        self, x: torch.Tensor, primary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the mixture outputs, the per-slot flag logits, and the per-slot signed fractions."""
        assert self.aux
        inp, mixer = self._core(x, primary)
        hidden = mixer[:-1](inp)
        out = mixer[-1](hidden)
        both = self.aux_head(hidden)
        return out, both[:, :SLOTS], both[:, SLOTS:]

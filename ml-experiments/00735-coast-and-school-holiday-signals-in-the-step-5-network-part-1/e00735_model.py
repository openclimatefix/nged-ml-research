# ruff: noqa: E501
"""Experiment 735: the step-5 network with extra mixer columns and an optional coast term in the embedding.

README. `CoastLagMdn` is a subclass of experiment 591's `AuxLagMdn` (the network of arm 9a, kind
`shared`, no detector). It relies on experiment 591 (`e00591_model`), 574 (`e00574_model`: `TAU`,
`b1_cols`), 559 (`e00559_model`: the column slices and widths), and 397 (`e00397_models`: `mlp`,
`init_head`). PYTHONPATH: source `ml-experiments/00524*/env.sh` from the worktree root.

- **Extra mixer columns.** The fine-tuning mixer's first layer gets `n_new` more input columns, which
  hold the new calendar, coast, and weather columns. They sit after the B1 columns of the input row
  (input index `TAU + 1 + n_b1` onwards), and the two injection-label columns that stage 3 appends
  sit after them, so `label_col` moves past the new columns.
- **Zero start.** `fresh_mixer` draws the control's mixer from seed 12345 at the control's width and
  pads the first layer with zero columns, so the new columns' weights start at zero and every arm
  starts stage 2 as exactly the control's network. `load_state_dict` pads a narrower first layer in
  the same way.
- **Coast in the embedding (SE and SEP).** The embedding that every encoder and the mixer read for
  Primary i is `E[i] + c_i * u`, where `u` is a learned 8-number vector that starts at zero and `c`
  is the arm's closeness (true or permuted). `c` is a non-persistent buffer, so a saved state holds
  `u` only.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00574_model as mod  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

FRESH_SEED: Final = 12345
"""Initialisation seed of every fresh mixer, as in experiments 545, 554, and 574."""
COAST_TERM: Final = "coast_u"
"""The name of the embedding term's parameter in a state dict."""


class CoastLagMdn(aux.AuxLagMdn):
    """Experiment 591's shared-lag-encoder network with `n_new` extra mixer columns and a coast term."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, n_new: int,
        coast: np.ndarray | None,
    ) -> None:  # fmt: skip
        """Build the network.

        Args:
            n_primaries: Number of Primaries.
            stage: `pre` or `ft`.
            n_knots: Number of output-offset knots.
            n_b1: Number of B1 columns.
            n_new: Number of new columns that the fine-tuning mixer reads.
            coast: The closeness per Primary for the embedding term, or None for no term.
        """
        super().__init__(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind="shared",
            detector=False,
        )  # fmt: skip
        self.n_new = n_new
        if n_new:
            old = self.mixer_ft[0].in_features  # ty: ignore[not-subscriptable,unresolved-attribute]
            self.mixer_ft = base.mlp(
                n_in=old + n_new, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS,
                dropout=m.DROPOUT,
            )  # fmt: skip
            base.init_head(head=self.mixer_ft, components=m.COMPONENTS)
        first = mod.TAU + 1 + n_b1
        self.new_cols = slice(first, first + n_new)
        self.label_col = first + n_new
        self.coast_embedding = coast is not None
        if coast is not None:
            assert len(coast) == n_primaries
            self.register_buffer(
                "coast_c", torch.tensor(np.asarray(coast), dtype=torch.float32), persistent=False
            )
            setattr(self, COAST_TERM, nn.Parameter(torch.zeros(m.EMBEDDING)))

    def embedding_of(self, *, primary: torch.Tensor) -> torch.Tensor:
        """Return the Primary embedding, with the coast term `c * u` added in the SE and SEP arms."""
        emb = self.embedding(primary)
        if self.coast_embedding:
            emb = emb + self.coast_c[primary][:, None] * getattr(self, COAST_TERM)  # ty: ignore[invalid-argument-type]
        return emb

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs; as `AuxLagMdn.forward`, plus the new columns and the coast term."""
        assert primary is not None
        labels = None
        if x.shape[1] == self.label_col + 2:
            labels, x = x[:, self.label_col :], x[:, : self.label_col]
        emb = self.embedding_of(primary=primary)
        weather_in = [x[:, m.WEATHER_COLS], emb]
        if self.n_b1:
            weather_in.append(x[:, mod.b1_cols(n_b1=self.n_b1)])
        hw = nn.functional.gelu(self.weather(torch.cat(weather_in, dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        if labels is not None and self.training:
            aux_pred = self.aux_from(hl=hl)
            ok = torch.isfinite(labels)
            err = torch.where(
                ok, (aux_pred - torch.nan_to_num(labels)) ** 2, torch.zeros_like(aux_pred)
            )
            aux.STASH["aux"] = err.sum(dim=1) / 2.0
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        if self.stage == "pre":
            out = self.mixer_pre(core)
        else:
            parts = [core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]]
            if self.n_new:
                parts.append(x[:, self.new_cols])
            out = self.mixer_ft(torch.cat(parts, dim=-1))
        off = self.offset(x[:, mod.TAU], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: ANN001, ANN201
        """Load a state dict; a first mixer layer narrower by `n_new` columns is padded with zeros."""
        key = "mixer_ft.0.weight"
        if (
            self.n_new
            and key in state_dict
            and state_dict[key].shape[1] == self.mixer_ft[0].in_features - self.n_new  # ty: ignore[unresolved-attribute]
        ):
            w = state_dict[key]
            state_dict = {
                **state_dict,
                key: torch.cat([w, w.new_zeros(w.shape[0], self.n_new)], dim=1),
            }
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def predict_aux(self, *, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the heads' depth predictions for rows without label columns."""
        assert x.shape[1] == self.label_col
        emb = self.embedding_of(primary=primary)
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        return self.aux_from(hl=hl)


def make_network(
    *, n_primaries: int, stage: str, n_knots: int, n_b1: int, n_new: int, coast: np.ndarray | None
) -> aux.AuxLagMdn:
    """Return the control's own class for an arm with no extra column and no coast term, else `CoastLagMdn`."""
    if n_new == 0 and coast is None:
        return aux.AuxLagMdn(
            n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, kind="shared",
            detector=False,
        )  # fmt: skip
    return CoastLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, n_new=n_new, coast=coast
    )


def fresh_mixer(*, model: nn.Module) -> None:
    """Give `mixer_ft` the control's seed-12345 draw at the control's width, with zero columns for the new inputs.

    Experiment 574's `fresh_mixer` draws every weight for the layer's width, so a wider mixer
    would change the old columns' weights as well. This draws the control's width and pads.
    """
    n_new = int(getattr(model, "n_new", 0))
    first = model.mixer_ft[0]  # ty: ignore[not-subscriptable]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FRESH_SEED)
        fresh = base.mlp(
            n_in=first.in_features - n_new, hidden=m.MIXER_HIDDEN, layers=2,
            n_out=3 * m.COMPONENTS, dropout=m.DROPOUT,
        )  # fmt: skip
        base.init_head(head=fresh, components=m.COMPONENTS)
    state = fresh.state_dict()
    if n_new:
        w = state["0.weight"]
        state["0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], n_new)], dim=1)
    model.mixer_ft.load_state_dict(state)  # ty: ignore[unresolved-attribute]


def load_pretrained_fresh(*, model: nn.Module, path) -> None:  # noqa: ANN001
    """Load experiment 677's stage-1 weights (without `mixer_ft`), then the zero-padded fresh mixer.

    A copy of `e00574_fit.load_pretrained_fresh` (variant `lagoff`) and
    `e00572_model.load_lag_pre_into_ft`, whose missing-key assertion (line 75) allows only
    `mixer_ft.*` and two knot keys and so would reject the embedding term `coast_u`.
    """
    raw = torch.load(path, map_location="cpu")
    knots = raw["knots"]
    rest = {a: v for a, v in raw.items() if a != "knots" and not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(rest, strict=False)
    assert all(
        a.startswith("mixer_ft.") or a in {"knots_old", "knot_new", COAST_TERM}
        for a in missing.missing_keys
    ), missing
    assert not missing.unexpected_keys, missing
    with torch.no_grad():
        model.knots_old.copy_(knots[:, :-1])  # ty: ignore[call-non-callable]
        model.knot_new.copy_(knots[:, -1])  # ty: ignore[call-non-callable]
    fresh_mixer(model=model)

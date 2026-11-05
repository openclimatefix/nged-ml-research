# ruff: noqa: E501
"""Experiment 571: the oracle network and the multiplicative-embedding hybrids.

All networks extend experiment 559's `ModularMdn` and read the knot position `tau` (months from
2019-09-01) from column 37, as experiments 559 and 565 do.

- `OracleMdn`: the base network plus one extra input, the true injected component in the Primary's
  target-standard-deviation units (column 37), fed to both mixers. It is an oracle by construction
  and is only used to say how much of an injected component a network could recover.
- `FilmHybrid` (arm H-film): the drift embedding multiplies the weather encoder's output channels by
  1 plus a linear map of the embedding (FiLM). The embedding has no additive path, so it has no
  direct level channel. The per-Primary monthly output offset of experiment 565's D5 is fitted jointly.
- `FilmHybrid(orth=True)` (arm H-orth): the same, with two penalties computed on a fixed probe sample
  of training rows. The level penalty is the mean squared change of the mixture location when the
  embedding moves from its Primary-average to each monthly knot (the embedding's level path). The
  covariance penalty is the squared covariance of that level path with the offset's path.

Pretraining learns every knot. Fine-tuning keeps the older knots as buffers and trains only the
newest knot of each (and the fresh mixer, and the Primary embedding); the FiLM map is frozen with the
encoders. A position is clamped to the newest knot (flat extrapolation).
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

TAU_COL: Final[int] = m.N_INPUTS
U_COL: Final[int] = m.N_INPUTS
FILM_DIM: Final[int] = 3
"""Width of the FiLM embedding (the brief asks for 2 to 4)."""
SMOOTH_LAMBDA: Final[float] = 10.0
"""Second-difference penalty on the embedding knots, the same as arm A10's."""
OFFSET_LAMBDA: Final[float] = 10.0
"""First-difference penalty on the offset knots, the same as D5's."""
FILM_L2: Final[float] = 10.0
"""Strong L2 shrinkage of the FiLM knots and map (mean squared value), fixed in the plan, not tuned."""
ORTH_LEVEL: Final[float] = 100.0
"""Weight of the mean squared level path of the embedding (standardised units squared)."""
ORTH_COV: Final[float] = 1e4
"""Weight of the squared covariance of the embedding's level path with the offset path."""
PROBE_ROWS: Final[int] = 64
PROBE_PRIMARIES: Final[int] = 2


class OracleMdn(m.ModularMdn):
    """`ModularMdn` plus the true injected component as one extra input to the mixers."""

    def __init__(self, *, n_primaries: int, stage: str) -> None:
        """Build the network with mixers one input wider."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        wide = 2 * m.ENCODER_OUT + m.EMBEDDING + 1
        self.mixer_pre = base.mlp(
            n_in=wide, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT
        )
        self.mixer_ft = base.mlp(
            n_in=wide + 7 + 2,
            hidden=m.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m.COMPONENTS,
            dropout=m.DROPOUT,
        )
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=m.COMPONENTS)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb, x[:, U_COL : U_COL + 1]], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))


def interpolate(*, table: torch.Tensor, tau: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
    """Return each row's linearly interpolated knot value, the position clamped to the knots.

    `table` is (Primaries, knots) or (Primaries, knots, width).
    """
    n_knots = table.shape[1]
    tau = tau.clamp(0.0, float(n_knots - 1))
    lo = tau.floor().long().clamp(max=n_knots - 2)
    frac = (tau - lo.to(tau.dtype)).reshape(-1, *([1] * (table.dim() - 2)))
    return (1.0 - frac) * table[primary, lo] + frac * table[primary, lo + 1]


class FilmHybrid(m.ModularMdn):
    """The drift embedding as a multiplicative modulation of the weather encoder, plus the D5 offset."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int, orth: bool = False) -> None:
        """Build the network; `n_knots` monthly knots (at least 2); `orth` adds the two penalties."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        assert n_knots >= 2
        self.n_knots = n_knots
        self.orth = orth
        self.film = nn.Linear(FILM_DIM, m.ENCODER_OUT, bias=False)
        nn.init.normal_(self.film.weight, std=0.1)
        if stage == "pre":
            self.knots = nn.Parameter(torch.zeros(n_primaries, n_knots, FILM_DIM))
            self.off_knots = nn.Parameter(torch.zeros(n_primaries, n_knots))
        else:
            self.register_buffer("knots_old", torch.zeros(n_primaries, n_knots - 1, FILM_DIM))
            self.knot_new = nn.Parameter(torch.zeros(n_primaries, FILM_DIM))
            self.register_buffer("off_old", torch.zeros(n_primaries, n_knots - 1))
            self.off_new = nn.Parameter(torch.zeros(n_primaries))
            self.film.requires_grad_(False)
        self.probe_x: torch.Tensor | None = None
        self.probe_primaries: torch.Tensor | None = None

    def all_knots(self) -> torch.Tensor:
        """Return every FiLM knot, (Primaries, knots, FILM_DIM)."""
        if self.stage == "pre":
            return self.knots
        return torch.cat([self.knots_old, self.knot_new[:, None, :]], dim=1)  # ty: ignore[invalid-argument-type]

    def all_offsets(self) -> torch.Tensor:
        """Return every offset knot, (Primaries, knots)."""
        if self.stage == "pre":
            return self.off_knots
        return torch.cat([self.off_old, self.off_new[:, None]], dim=1)  # ty: ignore[invalid-argument-type]

    def set_probe(self, *, x: torch.Tensor, p: torch.Tensor, seed: int = 0) -> None:
        """Store a fixed sample of `PROBE_ROWS` training rows per Primary for the orthogonality penalty."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows, have = [], []
        for q in range(self.embedding.num_embeddings):
            idx = torch.nonzero(p == q).squeeze(1)
            if len(idx) < PROBE_ROWS:
                continue
            pick = torch.randperm(len(idx), generator=generator)[:PROBE_ROWS].to(idx.device)
            rows.append(x[idx[pick]])
            have.append(q)
        self.probe_x = torch.stack(rows)  # (Primaries with rows, PROBE_ROWS, columns)
        self.probe_primaries = torch.tensor(have, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        primary: torch.Tensor | None = None,
        *,
        film_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs; `film_vec` overrides each row's embedding."""
        assert primary is not None
        emb = self.embedding(primary)
        vec = (
            interpolate(table=self.all_knots(), tau=x[:, TAU_COL], primary=primary)
            if film_vec is None
            else film_vec
        )
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hw = hw * (1.0 + self.film(vec))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        if self.stage == "pre":
            out = self.mixer_pre(core)
        else:
            out = self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))
        offset = interpolate(table=self.all_offsets(), tau=x[:, TAU_COL], primary=primary)[:, None]
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + offset, out[:, 2 * k :]], dim=-1)

    def level_path(self, *, primary: int) -> torch.Tensor:
        """Return the embedding's level path of one Primary on the probe rows, (knots,).

        The mean change of the mixture location when the embedding moves from the Primary-average
        (mean of the Primary's knots) to each monthly knot. Differentiable.
        """
        assert self.probe_x is not None
        assert self.probe_primaries is not None
        row = int(torch.nonzero(self.probe_primaries == primary).squeeze())
        xs = self.probe_x[row]
        knots = self.all_knots()[primary]
        vecs = torch.cat([knots, knots.mean(0, keepdim=True)], dim=0)  # (knots + 1, FILM_DIM)
        n = len(vecs)
        x_rep = xs.repeat(n, 1)
        v_rep = vecs.repeat_interleave(len(xs), dim=0)
        p_rep = torch.full((len(x_rep),), primary, device=xs.device, dtype=torch.long)
        out = self.forward(x_rep, p_rep, film_vec=v_rep)
        log_w, mean, _ = base.mixture_parts(out=out, components=m.COMPONENTS)
        loc = (log_w.exp() * mean).sum(-1).reshape(n, len(xs)).mean(1)
        return loc[:-1] - loc[-1]

    def penalty(self) -> torch.Tensor:
        """Return the smoothness and shrinkage penalties, plus the two orthogonality penalties if `orth`."""
        k = self.all_knots()
        off = self.all_offsets()
        pen = (
            SMOOTH_LAMBDA * ((k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]) ** 2).mean()
            + FILM_L2 * (k**2).mean()
            + FILM_L2 * (self.film.weight**2).mean()
            + OFFSET_LAMBDA * ((off[:, 1:] - off[:, :-1]) ** 2).mean()
        )
        if self.orth and self.probe_x is not None and self.training:
            assert self.probe_primaries is not None
            chosen = self.probe_primaries[
                torch.randperm(len(self.probe_primaries), device=k.device)[:PROBE_PRIMARIES]
            ]
            for q in chosen.tolist():
                e = self.level_path(primary=q)
                o = off[q] - off[q].mean()
                cov = ((e - e.mean()) * o).mean()
                pen = pen + (ORTH_LEVEL * (e**2).mean() + ORTH_COV * cov**2) / len(chosen)
        return pen


def load_pre_into_ft(*, model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a stage-`pre` state dict into the matching stage-`ft` network; `mixer_ft` stays as built."""
    assert getattr(model, "stage", None) == "ft"
    if isinstance(model, FilmHybrid):
        knots, off = state["knots"], state["off_knots"]
        rest = {
            a: v
            for a, v in state.items()
            if a not in {"knots", "off_knots"} and not a.startswith("mixer_ft.")
        }
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
        return
    rest = {a: v for a, v in state.items() if not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(rest, strict=False)
    assert all(a.startswith("mixer_ft.") for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing

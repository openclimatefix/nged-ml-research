# ruff: noqa: D205, E501
"""Experiment 579: the v3+D5 modular network of experiment 572 with configurable parts.

`ArmMdn` computes the same function class as experiment 572's `LagOffsetMdn` when `Cfg()` holds its
defaults (the state-dict keys of the embedding, weather encoder, datetime encoder, and lag encoder
match, so a stored 572 pre-training loads into it; `tests`-free equivalence is asserted by
`e00579_check.py`). Each field of `Cfg` changes one part:

- encoder size (`w`, `d`: layers, hidden width, outputs), the Primary embedding width, the mixer
  (layers, width, mixture components K), block type (`mlp`, `res` pre-layer-norm residual, `geglu`),
- FiLM conditioning of the encoders by the Primary embedding (`film`), a separate datetime embedding
  (`emb_dt`), a learnt per-Primary half-hour-of-day embedding (`p2`), extra datetime columns
  (`n_extra`),
- a rank-r per-Primary adapter on the mixer's last layer (`adapter_rank`).

Input columns: 0-36 lag-free, 37-44 lag block, 45 the D5 knot position, 46 onward the `n_extra`
extra datetime columns (when `p2`, the last extra column is the half-hour index and is not a network
input).
"""

from dataclasses import dataclass, field
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_LAG_FREE: Final[int] = 37
TAU_COL: Final[int] = 45
EXTRA0: Final[int] = 46
WEATHER_COLS: Final = slice(0, 7)
SPREAD_COLS: Final = slice(7, 14)
DATETIME_COLS: Final = slice(14, 35)
LEAD_COLS: Final = slice(35, 37)
LAG_COLS: Final = slice(37, 43)
DROPOUT: Final[float] = 0.1
LAG_HIDDEN: Final[int] = 64
LAG_OUT: Final[int] = 32
HALF_HOURS: Final[int] = 48


@dataclass(frozen=True)
class Part:
    """Shape of an encoder: hidden layers, hidden width, outputs."""

    layers: int = 1
    hidden: int = 64
    out: int = 32
    tied: bool = False  # C-tied: the wide layers are weight-tied copies of 64-unit layers


@dataclass(frozen=True)
class Cfg:
    """Every knob of an arm; the defaults are experiment 572's v3+D5."""

    w: Part = field(default_factory=Part)
    d: Part = field(default_factory=Part)
    emb: int = 8
    emb_dt: int = 0  # 0: the datetime encoder shares `emb`; else a separate embedding of this width
    mixer_layers: int = 2
    mixer_hidden: int = 64
    k: int = 3
    block: str = "mlp"  # "mlp", "res", or "geglu" (encoders and mixer)
    film: bool = False
    p2: bool = False
    n_extra: int = 0
    adapter_rank: int = 0


class Geglu(nn.Module):
    """GEGLU: a linear layer to twice the width, gated by GELU of one half."""

    def __init__(self, *, n_in: int, hidden: int) -> None:
        """Build the gate."""
        super().__init__()
        self.proj = nn.Linear(n_in, 2 * hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the gated activation."""
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * nn.functional.gelu(b)


class ResBlock(nn.Module):
    """Pre-layer-norm residual block: x + dropout(linear(gelu(linear(norm(x)))))."""

    def __init__(self, *, hidden: int, dropout: float) -> None:
        """Build the block."""
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, hidden)
        self.down = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the block output."""
        return x + self.drop(self.down(nn.functional.gelu(self.up(self.norm(x)))))


def trunk(*, n_in: int, part: Part, block: str) -> tuple[nn.Module, int]:
    """Return the hidden stack (everything before the output layer) and its output width."""
    if block == "mlp":
        modules: list[nn.Module] = []
        width = n_in
        for _ in range(part.layers):
            modules += [nn.Linear(width, part.hidden), nn.GELU(), nn.Dropout(DROPOUT)]
            width = part.hidden
        return nn.Sequential(*modules), width
    if block == "geglu":
        modules = []
        width = n_in
        for _ in range(part.layers):
            modules += [Geglu(n_in=width, hidden=part.hidden), nn.Dropout(DROPOUT)]
            width = part.hidden
        return nn.Sequential(*modules), width
    assert block == "res"
    modules = [nn.Linear(n_in, part.hidden)]
    modules += [ResBlock(hidden=part.hidden, dropout=DROPOUT) for _ in range(part.layers)]  # ty: ignore[unsupported-operator]
    modules.append(nn.LayerNorm(part.hidden))  # ty: ignore[invalid-argument-type]
    return nn.Sequential(*modules), part.hidden


class TiedEncoder(nn.Module):
    """C-tied: the shapes of a wide encoder whose units are copies of 64-unit layers.

    Each layer's output is the same 64 values repeated, so the encoder computes a function of a
    base-size network with the wide encoder's depth.
    """

    def __init__(self, *, n_in: int, part: Part) -> None:
        """Build the tied encoder; the base unit width is 64 and the output unit width is 32."""
        super().__init__()
        self.rep, self.rep_out = part.hidden // 64, part.out // 32
        self.l1 = nn.Linear(n_in, 64)
        self.rest = nn.ModuleList([nn.Linear(64, 64) for _ in range(part.layers - 1)])
        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output with each of the 32 base outputs repeated."""
        h = self.drop(nn.functional.gelu(self.l1(x)))
        for lin in self.rest:
            h = self.drop(nn.functional.gelu(lin(h)))
        return self.out(h).repeat(1, self.rep_out)


def encoder(*, n_feats: int, emb: int, part: Part, block: str, film: bool) -> nn.Module:
    """Return an encoder reading `[features, embedding]`. With the plain `mlp` block and no FiLM it is
    experiment 545's `base.mlp` (same keys), so stored weights load.
    """
    if part.tied:
        return TiedEncoder(n_in=n_feats + emb, part=part)
    if film:
        return FilmEncoder(n_feats=n_feats, emb=emb, part=part)
    if block == "mlp":
        return base.mlp(
            n_in=n_feats + emb,
            hidden=part.hidden,
            layers=part.layers,
            n_out=part.out,
            dropout=DROPOUT,
        )
    stack, width = trunk(n_in=n_feats + emb, part=part, block=block)
    return nn.Sequential(stack, nn.Linear(width, part.out))


class FilmEncoder(nn.Module):
    """Encoder over the features alone; the Primary embedding scales and shifts each hidden layer (FiLM)."""

    def __init__(self, *, n_feats: int, emb: int, part: Part) -> None:
        """Build the encoder; the input is `[features, embedding]` and the last `emb` columns are the embedding."""
        super().__init__()
        self.emb = emb
        widths = [n_feats] + [part.hidden] * part.layers
        self.lin = nn.ModuleList([nn.Linear(widths[i], widths[i + 1]) for i in range(part.layers)])
        self.film = nn.ModuleList([nn.Linear(emb, 2 * part.hidden) for _ in range(part.layers)])
        for f in self.film:
            nn.init.zeros_(f.weight)  # ty: ignore[invalid-argument-type]
            nn.init.zeros_(f.bias)  # ty: ignore[invalid-argument-type]
        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(part.hidden, part.out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the encoder output."""
        feats, emb = x[:, : -self.emb], x[:, -self.emb :]
        h = feats
        for lin, film in zip(self.lin, self.film, strict=True):
            gamma, beta = film(emb).chunk(2, dim=-1)
            h = self.drop(nn.functional.gelu(lin(h)) * (1.0 + gamma) + beta)
        return self.out(h)


class Mixer(nn.Module):
    """Hidden stack, output layer (3 K raw outputs), and an optional rank-r per-Primary adapter on the output."""

    def __init__(self, *, n_in: int, cfg: Cfg, n_primaries: int) -> None:
        """Build the mixer with the base-style output initialisation."""
        super().__init__()
        self.n_in = n_in
        self.trunk, width = trunk(
            n_in=n_in,
            part=Part(layers=cfg.mixer_layers, hidden=cfg.mixer_hidden),
            block=cfg.block,
        )
        self.out = nn.Linear(width, 3 * cfg.k)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor(base.mixture_head_bias(components=cfg.k)))
        self.rank = cfg.adapter_rank
        if self.rank:
            gen = torch.Generator().manual_seed(7)
            self.adapt_a = nn.Parameter(
                0.1 * torch.randn(n_primaries, width, self.rank, generator=gen)
            )
            self.adapt_b = nn.Parameter(torch.zeros(n_primaries, self.rank, 3 * cfg.k))

    def forward(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the raw mixture outputs."""
        h = self.trunk(x)
        out = self.out(h)
        if self.rank:
            out = (
                out
                + torch.bmm(torch.bmm(h[:, None, :], self.adapt_a[primary]), self.adapt_b[primary])[
                    :, 0
                ]
            )
        return out


class ArmMdn(nn.Module):
    """The v3+D5 modular network with the parts of `Cfg` (see the module docstring)."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int, cfg: Cfg) -> None:
        """Build the network; `stage` is `pre` or `ft`; the knots start at zero."""
        super().__init__()
        assert stage in {"pre", "ft"}
        assert n_knots >= 2
        self.cfg, self.stage, self.n_knots = cfg, stage, n_knots
        self.table: torch.Tensor | None = None
        e = cfg.emb
        self.embedding = nn.Embedding(n_primaries, e)
        e_dt = cfg.emb_dt or e
        if cfg.emb_dt:
            self.embedding_dt = nn.Embedding(n_primaries, cfg.emb_dt)
        if cfg.p2:
            self.hh = nn.Embedding(n_primaries * HALF_HOURS, 8)
        n_dt = (
            21 + cfg.n_extra + (7 if cfg.p2 else 0)
        )  # p2: the index column becomes an 8-wide embedding
        self.weather = encoder(n_feats=7, emb=e, part=cfg.w, block=cfg.block, film=cfg.film)
        self.datetime = encoder(n_feats=n_dt, emb=e_dt, part=cfg.d, block=cfg.block, film=cfg.film)
        expected = 2 * (cfg.w.out + cfg.d.out)
        self.lag = nn.ModuleList(
            [
                base.mlp(
                    n_in=6 + expected + e,
                    hidden=LAG_HIDDEN,
                    layers=1,
                    n_out=LAG_OUT,
                    dropout=DROPOUT,
                )
            ]
        )
        wide = cfg.w.out + cfg.d.out + LAG_OUT + e
        self.mixer_ft = Mixer(n_in=wide + 7 + 2, cfg=cfg, n_primaries=n_primaries)
        if stage == "pre":
            self.mixer_pre = Mixer(n_in=wide, cfg=cfg, n_primaries=n_primaries)
            self.knots = nn.Parameter(torch.zeros(n_primaries, n_knots, 1))
        else:
            self.register_buffer("knots_old", torch.zeros(n_primaries, n_knots - 1, 1))
            self.knot_new = nn.Parameter(torch.zeros(n_primaries, 1))

    # --- offset (as experiment 572) -------------------------------------------------------------
    def all_knots(self) -> torch.Tensor:
        """Return every knot, (Primaries, knots, 1)."""
        if self.stage == "pre":
            return self.knots
        return torch.cat([self.knots_old, self.knot_new[:, None, :]], dim=1)  # ty: ignore[invalid-argument-type]

    def penalty(self) -> torch.Tensor:
        """Return the mean squared first difference of the knots times experiment 565's lambda."""
        import e00565_model_variants as var  # ty: ignore[unresolved-import]

        k = self.all_knots()
        return var.PENALTY_LAMBDA * ((k[:, 1:] - k[:, :-1]) ** 2).mean()

    def offset(self, tau: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the interpolated offset of each row, (rows, 1)."""
        knots = self.all_knots()
        tau = tau.clamp(0.0, float(self.n_knots - 1))
        lo = tau.floor().long().clamp(max=self.n_knots - 2)
        frac = (tau - lo.to(tau.dtype))[:, None]
        return (1.0 - frac) * knots[primary, lo] + frac * knots[primary, lo + 1]

    # --- encoders -------------------------------------------------------------------------------
    def encode_weather(self, *, feats: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Return the weather encoder's activated output."""
        return nn.functional.gelu(self.weather(torch.cat([feats, emb], dim=-1)))

    def encode_datetime(
        self, *, feats: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the datetime encoder's activated output; `feats` is the 21 columns plus the extras."""
        if self.cfg.p2:
            idx = feats[:, -1].round().long().clamp(0, HALF_HOURS - 1)
            feats = torch.cat([feats[:, :-1], self.hh(primary * HALF_HOURS + idx)], dim=-1)
        if self.cfg.emb_dt:
            emb = self.embedding_dt(primary)
        return nn.functional.gelu(self.datetime(torch.cat([feats, emb], dim=-1)))

    def datetime_feats(self, x: torch.Tensor) -> torch.Tensor:
        """Return the datetime encoder's feature columns of the rows."""
        if self.cfg.n_extra:
            return torch.cat(
                [x[:, DATETIME_COLS], x[:, EXTRA0 : EXTRA0 + self.cfg.n_extra]], dim=-1
            )
        return x[:, DATETIME_COLS]

    def _expected(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return (weather, datetime) embeddings at each lag's own time, day lag first."""
        assert self.table is not None, "attach the expected-input table first"
        out = []
        for j in range(2):
            idx = x[:, 43 + j].long()
            row = self.table[primary, idx.clamp(min=0)].float()
            row = torch.where((idx >= 0)[:, None], row, torch.zeros_like(row))
            hw = self.encode_weather(feats=row[:, :7], emb=emb)
            hd = self.encode_datetime(feats=row[:, 7:], primary=primary, emb=emb)
            out.append((hw, hd))
        return out

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 3 K raw outputs with the offset added to the component means."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = self.encode_weather(feats=x[:, WEATHER_COLS], emb=emb)
        hd = self.encode_datetime(feats=self.datetime_feats(x), primary=primary, emb=emb)
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        lag_in = torch.cat([x[:, LAG_COLS], dw, dd, ww, wd, emb], dim=-1)
        hl = nn.functional.gelu(self.lag[0](lag_in))
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        if self.stage == "pre":
            out = self.mixer_pre(core, primary)
        else:
            out = self.mixer_ft(
                torch.cat([core, x[:, SPREAD_COLS], x[:, LEAD_COLS]], dim=-1), primary
            )
        k = self.cfg.k
        off = self.offset(x[:, TAU_COL], primary)
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)

    def parameter_counts(self) -> dict[str, int]:
        """Return the number of weights in each component (the offset knots and the stage's mixer included)."""
        n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        out = {"weather encoder": n(self.weather), "datetime encoder": n(self.datetime), "embedding": n(self.embedding),
               "lag encoder": n(self.lag), "mixer": n(self.mixer_pre if self.stage == "pre" else self.mixer_ft)}  # fmt: skip
        if self.cfg.emb_dt:
            out["datetime embedding"] = n(self.embedding_dt)
        if self.cfg.p2:
            out["half-hour embedding"] = n(self.hh)
        return out


def load_pre_into_ft(
    *, model: ArmMdn, state: dict[str, torch.Tensor], skip_mixer_pre: bool
) -> None:
    """Load a stage-`pre` state dict into the stage-`ft` network; `mixer_ft` stays as built."""
    assert model.stage == "ft"
    knots = state["knots"]
    rest = {
        a: v
        for a, v in state.items()
        if a != "knots" and not a.startswith(("mixer_ft.", "mixer_pre."))
    }
    missing = model.load_state_dict(rest, strict=False)
    assert all(
        a.startswith("mixer_ft.") or a in {"knots_old", "knot_new"} for a in missing.missing_keys
    ), missing
    assert not missing.unexpected_keys, missing
    _ = skip_mixer_pre
    with torch.no_grad():
        model.knots_old.copy_(knots[:, :-1])  # ty: ignore[call-non-callable]
        model.knot_new.copy_(knots[:, -1])

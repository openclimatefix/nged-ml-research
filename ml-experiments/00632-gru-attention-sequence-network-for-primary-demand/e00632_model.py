# ruff: noqa: E501, D102, D107
"""Experiment 632: the GRU plus self-attention network (Seq) and its ablations.

Input layout of a batch is `e00632_data.Batch`. Per token: a weather encoder over the 7 weather
columns and the 14 B1 columns (the experiment 574 weather encoder with B1), a datetime encoder over
the 21 datetime columns, a position embedding and a segment embedding. A history token adds a lag
encoder over own power (value, missing flag, age) with the token's own weather and datetime
embeddings and the Primary embedding (and, in arm A-NB, the neighbour summaries); a future token adds
a learnt "no lag" vector. An encoder GRU reads the history tokens, a decoder GRU reads the future
tokens from a start state built from the encoder's last state and its state at origin minus 24 h,
and `N_LAYERS` joint self-attention layers run over all tokens. The head is a three-component
Gaussian mixture per future token plus the D5 offset on the component means.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

WIDTH: Final[int] = 64
ENC_HIDDEN: Final[int] = 64
ENC_OUT: Final[int] = 32
EMB: Final[int] = 8
POS_EMB: Final[int] = 16
LAG_HIDDEN: Final[int] = 64
LAG_OUT: Final[int] = 32
N_LAYERS: Final[int] = 3
HEADS: Final[int] = 4
FF: Final[int] = 256
DROPOUT: Final[float] = 0.1
COMPONENTS: Final[int] = 3
PENALTY_LAMBDA: Final[float] = 10.0
N_WEATHER: Final[int] = 7
N_DATETIME: Final[int] = 21
N_B1_GRID: Final[int] = 12
FIXED_STATE: Final[int] = 48
"""The decoder also sees the encoder state this many tokens before the last one (origin minus 24 h)."""
ARMS: Final = ("seq", "seq_nb", "no_attn", "no_gru")


class Seq(nn.Module):
    """The sequence network; `arm` selects the main model or an ablation."""

    def __init__(
        self, *, n_primaries: int, n_knots: int, n_hist: int, n_fut: int, arm: str = "seq"
    ) -> None:
        super().__init__()
        assert arm in ARMS
        self.arm, self.n_hist, self.n_fut, self.n_knots = arm, n_hist, n_fut, n_knots
        self.use_nb = arm == "seq_nb"
        self.embedding = nn.Embedding(n_primaries, EMB)
        self.weather = base.mlp(
            n_in=N_WEATHER + N_B1_GRID + 2 + EMB,
            hidden=ENC_HIDDEN,
            layers=1,
            n_out=ENC_OUT,
            dropout=DROPOUT,
        )
        self.datetime = base.mlp(
            n_in=N_DATETIME + EMB, hidden=ENC_HIDDEN, layers=1, n_out=ENC_OUT, dropout=DROPOUT
        )
        self.lag = base.mlp(
            n_in=3 + (2 if self.use_nb else 0) + 2 * ENC_OUT + EMB,
            hidden=LAG_HIDDEN,
            layers=1,
            n_out=LAG_OUT,
            dropout=DROPOUT,
        )
        self.no_lag = nn.Parameter(torch.zeros(LAG_OUT))
        self.pos = nn.Embedding(n_hist + n_fut, POS_EMB)
        self.seg = nn.Embedding(2, EMB)
        self.proj = nn.Linear(2 * ENC_OUT + LAG_OUT + EMB + POS_EMB + EMB, WIDTH)
        self.enc_gru = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.dec_gru = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.link = nn.Linear(2 * WIDTH, WIDTH)
        layer = nn.TransformerEncoderLayer(
            WIDTH, HEADS, FF, DROPOUT, activation="gelu", batch_first=True, norm_first=True
        )
        self.attn = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.head = base.mlp(n_in=WIDTH, hidden=WIDTH, layers=1, n_out=3 * COMPONENTS, dropout=0.0)
        base.init_head(head=self.head, components=COMPONENTS)
        self.knots = nn.Parameter(torch.zeros(n_primaries, n_knots))

    def offset(self, tau: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the interpolated D5 offset (B, F); positions beyond the newest knot read the newest knot."""
        tau = tau.clamp(0.0, float(self.n_knots - 1))
        lo = tau.floor().long().clamp(max=self.n_knots - 2)
        frac = tau - lo.to(tau.dtype)
        k = self.knots[primary[:, None], lo], self.knots[primary[:, None], lo + 1]
        return (1.0 - frac) * k[0] + frac * k[1]

    def penalty(self) -> torch.Tensor:
        return PENALTY_LAMBDA * ((self.knots[:, 1:] - self.knots[:, :-1]) ** 2).mean()

    def encode(
        self, x: torch.Tensor, cams: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weather and datetime embeddings of tokens x (B, T, 40)."""
        t = x.shape[1]
        w_in = torch.cat(
            [
                x[..., :N_WEATHER],
                x[..., N_WEATHER + N_DATETIME :],
                cams[:, None, :].expand(-1, t, -1),
                emb[:, None, :].expand(-1, t, -1),
            ],
            dim=-1,
        )
        d_in = torch.cat(
            [x[..., N_WEATHER : N_WEATHER + N_DATETIME], emb[:, None, :].expand(-1, t, -1)], dim=-1
        )
        return nn.functional.gelu(self.weather(w_in)), nn.functional.gelu(self.datetime(d_in))

    def forward(self, b) -> torch.Tensor:  # noqa: ANN001
        """Return the raw mixture outputs (B, F, 9) with the D5 offset added to the component means."""
        emb = self.embedding(b.primary)
        x_h, x_f = b.x_hist.float(), b.x_fut.float()
        hw, hd = self.encode(x_h, b.cams, emb)
        fw, fd = self.encode(x_f, b.cams, emb)
        bs = len(emb)
        lag_in = [b.power[..., None], b.power_missing[..., None], b.age[..., None]]
        if self.use_nb:
            lag_in += [b.nb_mean[..., None], b.nb_share[..., None]]
        lag = nn.functional.gelu(
            self.lag(
                torch.cat([*lag_in, hw, hd, emb[:, None, :].expand(-1, self.n_hist, -1)], dim=-1)
            )
        )
        pos = self.pos(torch.arange(self.n_hist + self.n_fut, device=emb.device))[None].expand(
            bs, -1, -1
        )
        seg = torch.cat(
            [
                self.seg(torch.zeros(self.n_hist, dtype=torch.long, device=emb.device)),
                self.seg(torch.ones(self.n_fut, dtype=torch.long, device=emb.device)),
            ]
        )[None].expand(bs, -1, -1)
        e = emb[:, None, :].expand(-1, self.n_hist + self.n_fut, -1)
        no_lag = self.no_lag[None, None, :].expand(bs, self.n_fut, -1)
        tok = torch.cat(
            [
                torch.cat([hw, fw], 1),
                torch.cat([hd, fd], 1),
                torch.cat([lag, no_lag], 1),
                e,
                pos,
                seg,
            ],
            dim=-1,
        )
        tok = self.proj(tok)
        th, tf = tok[:, : self.n_hist], tok[:, self.n_hist :]
        if self.arm == "no_gru":
            h = torch.cat([th, tf], 1)
        else:
            eo, h_last = self.enc_gru(th)
            start = torch.tanh(self.link(torch.cat([eo[:, -1], eo[:, -1 - FIXED_STATE]], dim=-1)))[
                None
            ]
            do, _ = self.dec_gru(tf, start.contiguous())
            h = torch.cat([eo, do], 1)
        if self.arm != "no_attn":
            h = self.attn(h)
        out = self.head(h[:, self.n_hist :]).float()
        k = COMPONENTS
        off = self.offset(b.tau, b.primary)[..., None]
        return torch.cat([out[..., :k], out[..., k : 2 * k] + off, out[..., 2 * k :]], dim=-1)


def masked_nll(*, out: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the summed negative log-likelihood over future tokens with a target, and their count."""
    ok = torch.isfinite(target)
    nll = base.mixture_nll(
        out=out.reshape(-1, 3 * COMPONENTS),
        z=torch.nan_to_num(target).reshape(-1),
        components=COMPONENTS,
    )
    nll = nll.reshape(target.shape)
    return (nll * ok).sum(), ok.sum()

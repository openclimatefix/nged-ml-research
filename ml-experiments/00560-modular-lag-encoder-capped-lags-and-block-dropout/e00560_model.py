# ruff: noqa: E501
"""The modular demand network of experiment 560: weather, datetime, and lag encoders plus a mixer.

The weather encoder, datetime encoder, mixer, and Primary embedding are experiment 545's M0
(`e00545_model`). The lag encoder is new. It reads the capped day and week lags (value, missing flag,
age) in these input columns, which follow the 37 lag-free columns:

- 37-39: day lag value, missing flag, age; 40-42: the same for the week lag;
- 43, 44: the grid index of the day and week lag times, as floats (not inputs; they address `table`).

`table` is a (primaries, grid, 28) float16 tensor of standardised ensemble-mean weather (7 columns) and
datetime (21 columns) inputs at each grid time, taken from the freshest weather run issued at or
before that time. A copy of the weather encoder and a copy of the datetime encoder (the same
modules, so the same weights) turn the table row at a lag's own time into "expected" embeddings.

Modes (the `mode` argument):

- `v1`: the lag encoder reads the six lag columns and the Primary embedding;
- `v2`: `v1` plus the 21 datetime columns of the target time;
- `v3`: `v1` plus the expected embeddings of both lag times;
- `v4`: the expected embeddings alone (no lagged power);
- `b_shared`: one encoder, with shared weights, reads one lag at a time with that lag's expected
  embeddings; its two outputs are concatenated;
- `b_unshared`: as `b_shared` with one encoder per lag.
"""

from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_BASE: Final[int] = 37
LAG_COLS: Final[slice] = slice(37, 43)
DAY: Final[slice] = slice(37, 40)
WEEK: Final[slice] = slice(40, 43)
IDX_COLS: Final[slice] = slice(43, 45)
N_INPUTS: Final[int] = 45
N_EXPECTED: Final[int] = 28
MODES: Final[tuple[str, ...]] = ("v1", "v2", "v3", "v4", "b_shared", "b_unshared")
LAG_OUT: Final[int] = 32
LAG_HIDDEN: Final[int] = 64


class LagModularMdn(nn.Module):
    """Experiment 545's M0 plus a lag encoder (see the module docstring)."""

    def __init__(self, *, n_primaries: int, mode: str, stage: str) -> None:
        """Build the network; `stage` is `pre` (ERA5/CAMS pretraining) or `ft` (ENS fine-tuning)."""
        super().__init__()
        assert mode in MODES
        assert stage in {"pre", "ft"}
        self.mode, self.stage = mode, stage
        self.table: torch.Tensor | None = None
        e = m545.EMBEDDING
        self.embedding = nn.Embedding(n_primaries, e)
        self.weather = base.mlp(
            n_in=7 + e,
            hidden=m545.ENCODER_HIDDEN,
            layers=1,
            n_out=m545.ENCODER_OUT,
            dropout=m545.DROPOUT,
        )
        self.datetime = base.mlp(
            n_in=21 + e,
            hidden=m545.ENCODER_HIDDEN,
            layers=1,
            n_out=m545.ENCODER_OUT,
            dropout=m545.DROPOUT,
        )
        expected = 4 * m545.ENCODER_OUT
        if mode == "v1":
            lag_in, n_lag = 6 + e, 1
        elif mode == "v2":
            lag_in, n_lag = 6 + 21 + e, 1
        elif mode == "v3":
            lag_in, n_lag = 6 + expected + e, 1
        elif mode == "v4":
            lag_in, n_lag = expected + e, 1
        else:
            lag_in, n_lag = 3 + 2 * m545.ENCODER_OUT + e, 2
        n_enc = 2 if mode == "b_unshared" else 1
        self.lag = nn.ModuleList(
            [
                base.mlp(
                    n_in=lag_in, hidden=LAG_HIDDEN, layers=1, n_out=LAG_OUT, dropout=m545.DROPOUT
                )
                for _ in range(n_enc)
            ]
        )
        self.n_lag_out = LAG_OUT * n_lag
        wide = 2 * m545.ENCODER_OUT + self.n_lag_out + e
        self.mixer_pre = base.mlp(
            n_in=wide,
            hidden=m545.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m545.COMPONENTS,
            dropout=m545.DROPOUT,
        )
        self.mixer_ft = base.mlp(
            n_in=wide + 7 + 2,
            hidden=m545.MIXER_HIDDEN,
            layers=2,
            n_out=3 * m545.COMPONENTS,
            dropout=m545.DROPOUT,
        )
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=m545.COMPONENTS)

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
            hw = nn.functional.gelu(self.weather(torch.cat([row[:, :7], emb], dim=-1)))
            hd = nn.functional.gelu(self.datetime(torch.cat([row[:, 7:], emb], dim=-1)))
            out.append((hw, hd))
        return out

    def lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the lag encoder's output."""
        lags = x[:, LAG_COLS]
        if self.mode == "v1":
            return self.lag[0](torch.cat([lags, emb], dim=-1))
        if self.mode == "v2":
            return self.lag[0](torch.cat([lags, x[:, m545.DATETIME_COLS], emb], dim=-1))
        (dw, dd), (ww, wd) = self._expected(x=x, primary=primary, emb=emb)
        if self.mode == "v3":
            return self.lag[0](torch.cat([lags, dw, dd, ww, wd, emb], dim=-1))
        if self.mode == "v4":
            return self.lag[0](torch.cat([dw, dd, ww, wd, emb], dim=-1))
        day = torch.cat([x[:, DAY], dw, dd, emb], dim=-1)
        week = torch.cat([x[:, WEEK], ww, wd, emb], dim=-1)
        enc_week = self.lag[-1]
        return torch.cat([self.lag[0](day), enc_week(week)], dim=-1)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m545.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m545.DATETIME_COLS], emb], dim=-1)))
        hl = nn.functional.gelu(self.lag_embedding(x=x, primary=primary, emb=emb))
        core = torch.cat([hw, hd, hl, emb], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(
            torch.cat([core, x[:, m545.SPREAD_COLS], x[:, m545.LEAD_COLS]], dim=-1)
        )

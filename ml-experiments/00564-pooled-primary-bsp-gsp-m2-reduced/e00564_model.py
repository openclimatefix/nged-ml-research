# ruff: noqa: E501
"""Experiment 545's modular lag-free network (M0) with a series-type input, for 20 series.

The network is `e00554_model.ModularMdn` with one change: the datetime encoder also reads a fixed
one-hot vector saying whether the series is a Primary, a bulk supply point (BSP), or a grid supply
point (GSP). The vector is a buffer indexed by the series number, not a column of the input matrix,
so the rows are the same as M0's. The learned 8-wide embedding is kept.
"""

import e00554_model as m0  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

import e00397_models as base  # ty: ignore[unresolved-import]  # isort: skip

KINDS = ("Primary", "BSP", "GSP")
N_KINDS = len(KINDS)


class PooledModularMdn(nn.Module):
    """Weather encoder, datetime encoder (with the series type), and a mixer."""

    def __init__(self, *, kinds: list[str], stage: str) -> None:
        """Build the network; `kinds` holds each series' type in series-index order."""
        super().__init__()
        assert stage in {"pre", "ft"}
        self.stage = stage
        n_series = len(kinds)
        one_hot = torch.zeros(n_series, N_KINDS)
        for i, kind in enumerate(kinds):
            one_hot[i, KINDS.index(kind)] = 1.0
        self.register_buffer("kind", one_hot)
        self.embedding = nn.Embedding(n_series, m0.EMBEDDING)
        self.weather = base.mlp(
            n_in=7 + m0.EMBEDDING, hidden=m0.ENCODER_HIDDEN, layers=1, n_out=m0.ENCODER_OUT, dropout=m0.DROPOUT
        )
        self.datetime = base.mlp(
            n_in=21 + m0.EMBEDDING + N_KINDS, hidden=m0.ENCODER_HIDDEN, layers=1, n_out=m0.ENCODER_OUT, dropout=m0.DROPOUT
        )
        wide = 2 * m0.ENCODER_OUT + m0.EMBEDDING
        self.mixer_pre = base.mlp(
            n_in=wide, hidden=m0.MIXER_HIDDEN, layers=2, n_out=3 * m0.COMPONENTS, dropout=m0.DROPOUT
        )
        self.mixer_ft = base.mlp(
            n_in=wide + 7 + 2, hidden=m0.MIXER_HIDDEN, layers=2, n_out=3 * m0.COMPONENTS, dropout=m0.DROPOUT
        )
        for head in (self.mixer_pre, self.mixer_ft):
            base.init_head(head=head, components=m0.COMPONENTS)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m0.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(
            self.datetime(torch.cat([x[:, m0.DATETIME_COLS], emb, self.kind[primary]], dim=-1))
        )
        core = torch.cat([hw, hd, emb], dim=-1)
        if self.stage == "pre":
            return self.mixer_pre(core)
        return self.mixer_ft(torch.cat([core, x[:, m0.SPREAD_COLS], x[:, m0.LEAD_COLS]], dim=-1))


def expand_m0_state(*, raw: dict[str, torch.Tensor], n_series: int) -> dict[str, torch.Tensor]:
    """Return experiment 545's 16-series M0 weights fitted to the 20-series network.

    The embedding gets the mean of the 16 Primary rows for each new series. The datetime encoder's
    first layer gets three zero columns for the series-type input, so the pretrained network starts
    by ignoring the type. Everything else is copied.
    """
    out = dict(raw)
    weight = raw["embedding.weight"]
    extra = weight.mean(dim=0, keepdim=True).expand(n_series - len(weight), -1)
    out["embedding.weight"] = torch.cat([weight, extra], dim=0)
    first = raw["datetime.0.weight"]
    out["datetime.0.weight"] = torch.cat([first, torch.zeros(first.shape[0], N_KINDS)], dim=1)
    return out

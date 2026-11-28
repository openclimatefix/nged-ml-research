# ruff: noqa: E501
"""Experiment 735 part 2: the step-5 network with 8 extra inputs to the date-time encoder.

README. `SchoolLagMdn` is a subclass of experiment 591's `AuxLagMdn` (the network of arm 9a, kind
`shared`, no detector, which is also experiment 677's step-5 network). It relies on experiment 591
(`e00591_model`), 574 (`e00574_model`: `TAU`, `b1_cols`), 559 (`e00559_model`: the column slices and
widths), and 397 (`e00397_models`: `mlp`). PYTHONPATH: source `ml-experiments/00524*/env.sh` from the
worktree root.

- **n_new = 0 is experiment 677's network.** `forward` hands the call to `AuxLagMdn.forward`, so the
  control R1 has no extra input and the same state-dict keys, shapes, and outputs as experiment 677's
  step-5 network.
- **n_new = 8 widens the date-time encoder's first layer from 29 to 37 inputs.** The encoder reads
  the 21 date-time columns, then the 8 new columns, then the Primary embedding. In the row path the 8
  columns are the 8 trailing columns of `x` (input index 60 to 67, after the B1 columns). The two
  injection-label columns of stage 3 sit after them, so `label_col` is 68.
- **The lag path reads the 8 columns at each lag's own valid time.** `_expected` takes the
  date-time encoder's input at a lag from the expected-input table (`table`, 21 date-time columns at
  the grid time `x[:, 43 + j]`). The extra table `table_new` (Primaries, grid, 8) holds the 8 columns
  at every grid time (`e00735b_columns.grid_table`), and `_expected` reads it at the same grid index
  and passes the date-time encoder `[21 table columns, 8 table_new columns, embedding]`. A missing lag
  (index below 0) reads zeros, as the 21 columns do.
- **Paired initialisation.** The 37-input first layer copies the 29-input layer that `AuxLagMdn`
  drew (every other weight and bias too), and draws only the 8 new columns' weights, from a
  separate generator seeded with `EXTRA_COLUMN_SEED`, uniform in plus or minus 1 over the square root of
  37 (PyTorch's default for a linear layer). The widened layer is built after every original weight
  has been drawn, so every original weight equals the control's for the same seed. Building the
  widened layer does consume the global random stream, so the stream differs from the control's
  after construction. `e00574_fit.train` reseeds with `torch.manual_seed(seed)`, so the data order
  and the dropout masks stay paired across arms.
"""

import math
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00574_model as mod  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_NEW: Final = 8
"""The number of columns the date-time encoder gains."""
EXTRA_COLUMN_SEED: Final = 735
"""Seed of the generator that draws the new columns' first-layer weights."""
N_DATETIME: Final = 21
"""The width of `m.DATETIME_COLS`."""


class SchoolLagMdn(aux.AuxLagMdn):
    """Experiment 591's shared-lag-encoder network with `n_new` extra date-time encoder inputs."""

    def __init__(
        self, *, n_primaries: int, stage: str, n_knots: int, n_b1: int, n_new: int
    ) -> None:
        """Build the network.

        Args:
            n_primaries: Number of Primaries.
            stage: `pre` or `ft`.
            n_knots: Number of output-offset knots.
            n_b1: Number of B1 columns.
            n_new: 0 for experiment 677's network, 8 for the arms with the school and coast columns.
        """
        super().__init__(
            n_primaries=n_primaries,
            stage=stage,
            n_knots=n_knots,
            n_b1=n_b1,
            kind="shared",
            detector=False,
        )
        assert n_new in {0, N_NEW}, n_new
        self.n_new = n_new
        self.table_new: torch.Tensor | None = None
        """(Primaries, grid, 8): the new columns at every grid time; the lag path reads it."""
        first = mod.TAU + 1 + n_b1
        self.new_cols = slice(first, first + n_new)
        self.label_col = first + n_new
        if n_new:
            self._widen_datetime()

    def _widen_datetime(self) -> None:
        """Replace the date-time encoder by one with 8 more inputs, keeping the old weights."""
        old = self.datetime
        wide = base.mlp(
            n_in=N_DATETIME + self.n_new + m.EMBEDDING,
            hidden=m.ENCODER_HIDDEN,
            layers=1,
            n_out=m.ENCODER_OUT,
            dropout=m.DROPOUT,
        )
        state = old.state_dict()
        w_old = state["0.weight"]
        generator = torch.Generator().manual_seed(EXTRA_COLUMN_SEED)
        bound = 1.0 / math.sqrt(w_old.shape[1] + self.n_new)
        fresh = (torch.rand(w_old.shape[0], self.n_new, generator=generator) * 2.0 - 1.0) * bound
        state["0.weight"] = torch.cat(
            [w_old[:, :N_DATETIME], fresh, w_old[:, N_DATETIME:]], dim=1
        ).contiguous()
        wide.load_state_dict(state)
        self.datetime = wide

    def _expected(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return (weather, datetime) embeddings at each lag's own time; the datetime input holds the 8 columns."""
        if not self.n_new:
            return super()._expected(x=x, primary=primary, emb=emb)
        assert self.table is not None, "attach the expected-input table first"
        assert self.table_new is not None, "attach the new-column table first"
        out = []
        for j in range(2):
            idx = x[:, 43 + j].long()
            keep = (idx >= 0)[:, None]
            safe = idx.clamp(min=0)
            row = self.table[primary, safe].float()
            row = torch.where(keep, row, torch.zeros_like(row))
            new = self.table_new[primary, safe].float()
            new = torch.where(keep, new, torch.zeros_like(new))
            weather_in = [row[:, :7], emb]
            if self.n_b1:
                weather_in.append(torch.zeros(len(row), self.n_b1, device=row.device))
            hw = nn.functional.gelu(self.weather(torch.cat(weather_in, dim=-1)))
            hd = nn.functional.gelu(self.datetime(torch.cat([row[:, 7:], new, emb], dim=-1)))
            out.append((hw, hd))
        return out

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs; as `AuxLagMdn.forward`, with the 8 columns in the date-time input."""
        if not self.n_new:
            return super().forward(x, primary)
        assert primary is not None
        labels = None
        if x.shape[1] == self.label_col + 2:
            labels, x = x[:, self.label_col :], x[:, : self.label_col]
        emb = self.embedding(primary)
        weather_in = [x[:, m.WEATHER_COLS], emb]
        if self.n_b1:
            weather_in.append(x[:, mod.b1_cols(n_b1=self.n_b1)])
        hw = nn.functional.gelu(self.weather(torch.cat(weather_in, dim=-1)))
        hd = nn.functional.gelu(
            self.datetime(torch.cat([x[:, m.DATETIME_COLS], x[:, self.new_cols], emb], dim=-1))
        )
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
            out = self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS]], dim=-1))
        off = self.offset(x[:, mod.TAU], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)


def make_network(
    *, n_primaries: int, stage: str, n_knots: int, n_b1: int, n_new: int
) -> SchoolLagMdn:
    """Return the network of an arm: experiment 677's (`n_new` 0) or the one with the 8 columns."""
    return SchoolLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=n_knots, n_b1=n_b1, n_new=n_new
    )

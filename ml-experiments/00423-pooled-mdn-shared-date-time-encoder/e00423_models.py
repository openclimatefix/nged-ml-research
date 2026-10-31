"""The calendar encoder and the pooled network that carries it, for experiment 423.

`CalendarEncoder` embeds the calendar indices of a row, concatenates the embeddings, and maps them
through Linear 32, GELU, Linear 16. `EncodedPooledNetwork` is experiment 407's `PooledNetwork` whose
trunk input is `[kept columns, 16-wide encoder output, Primary embedding]`. The integer indices
travel as the last columns of the float input matrix (exact for values below 2^24), so experiment
407's training loop, which calls `model(x, primary)`, runs unchanged.

The arms differ in which indices the encoder reads (`ENCODERS`) and in whether one encoder serves
all Primaries or each Primary has its own (E3).
"""

from dataclasses import dataclass
from typing import Final

import e00407_models as fam  # ty: ignore[unresolved-import]
import e00423_features as feats  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

HIDDEN_WIDTH: Final[int] = 32
OUTPUT_WIDTH: Final[int] = 16
HALF_HOUR_EMBEDDING: Final[int] = 8
SMALL_EMBEDDING: Final[int] = 4


@dataclass(frozen=True)
class EncoderSpec:
    """Which calendar indices an encoder reads, and each index's embedding width."""

    widths: dict[str, int]
    keep_holiday_inputs: bool
    per_primary: bool = False

    @property
    def index_columns(self) -> tuple[str, ...]:
        """Return the encoder's index columns, in the order of `feats.INDEX_COLUMNS`."""
        return tuple(c for c in feats.INDEX_COLUMNS if c in self.widths)

    @property
    def input_width(self) -> int:
        """Return the width of the concatenated embeddings."""
        return sum(self.widths.values())


_E0_WIDTHS: Final[dict[str, int]] = {"half_hour": HALF_HOUR_EMBEDDING, "weekday": SMALL_EMBEDDING}
_E2_WIDTHS: Final[dict[str, int]] = {
    **_E0_WIDTHS,
    "day_of_month": SMALL_EMBEDDING,
    "day_of_year": SMALL_EMBEDDING,
}
_E1_WIDTHS: Final[dict[str, int]] = {**_E2_WIDTHS, "holiday_name": SMALL_EMBEDDING}
ENCODERS: Final[dict[str, EncoderSpec]] = {
    "E0": EncoderSpec(widths=_E0_WIDTHS, keep_holiday_inputs=False),
    "E1n": EncoderSpec(widths=_E0_WIDTHS, keep_holiday_inputs=True),
    "E1": EncoderSpec(widths=_E1_WIDTHS, keep_holiday_inputs=True),
    "E2": EncoderSpec(widths=_E2_WIDTHS, keep_holiday_inputs=False),
    "E3": EncoderSpec(widths=_E1_WIDTHS, keep_holiday_inputs=True, per_primary=True),
}
"""The encoder of each arm. R0 has none: it is experiment 407's network on the 18 columns."""


class CalendarEncoder(nn.Module):
    """Embeddings of the calendar indices, concatenated, then Linear 32, GELU, Linear 16."""

    def __init__(self, *, spec: EncoderSpec) -> None:
        """Build the embedding tables and the two linear layers."""
        super().__init__()
        self.columns = spec.index_columns
        self.tables = nn.ModuleList(
            [nn.Embedding(feats.INDEX_SIZES[c], spec.widths[c]) for c in self.columns]
        )
        self.mlp = nn.Sequential(
            nn.Linear(spec.input_width, HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, OUTPUT_WIDTH),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the encoding (rows, 16) of calendar indices (rows, len(columns)), int64."""
        parts = [table(indices[:, j]) for j, table in enumerate(self.tables)]
        return self.mlp(torch.cat(parts, dim=-1))


class EncodedPooledNetwork(fam.PooledNetwork):
    """Experiment 407's pooled network with a calendar encoder in front of the trunk.

    The input matrix holds `n_dense` standardised columns followed by the calendar indices the
    encoder reads, in `EncoderSpec.index_columns` order. With `spec.per_primary` there is one
    encoder per Primary, chosen by the row's Primary id (arm E3); otherwise one encoder is shared.
    """

    def __init__(
        self,
        *,
        n_dense: int,
        spec: EncoderSpec,
        hidden: int,
        layers: int,
        dropout: float,
        n_primaries: int,
    ) -> None:
        """Build the network; the trunk reads the dense columns, the encoding, and the embedding."""
        super().__init__(
            n_inputs=n_dense + OUTPUT_WIDTH,
            family="gauss",
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            n_primaries=n_primaries,
        )
        self.n_dense = n_dense
        self.n_indices = len(spec.index_columns)
        self.n_primaries = n_primaries
        self.encoders = nn.ModuleList(
            [CalendarEncoder(spec=spec) for _ in range(n_primaries if spec.per_primary else 1)]
        )

    def encode(self, indices: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the encoding (rows, 16), from the shared encoder or each row's own."""
        if len(self.encoders) == 1:
            return self.encoders[0](indices)
        out = torch.zeros(len(indices), OUTPUT_WIDTH, dtype=torch.float32, device=indices.device)
        for p, encoder in enumerate(self.encoders):
            rows = torch.nonzero(primary == p).squeeze(1)
            if len(rows):
                out = out.index_copy(0, rows, encoder(indices[rows]))
        return out

    def forward(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the raw outputs of rows `x` (dense columns, then indices) of `primary`."""
        dense = x[:, : self.n_dense]
        indices = x[:, self.n_dense :].round().long()
        return self.net(
            torch.cat([dense, self.encode(indices, primary), self.embedding(primary)], -1)
        )

    def encoder_parameter_counts(self) -> dict[str, int]:
        """Return the encoder's embedding-table and linear-layer parameter counts (all encoders)."""
        tables = sum(p.numel() for e in self.encoders for p in e.tables.parameters())
        mlp = sum(p.numel() for e in self.encoders for p in e.mlp.parameters())
        return {"embedding tables": tables, "linear layers": mlp, "encoder total": tables + mlp}

# ruff: noqa: E501
"""Experiment 706, stage 2: the two learned arms as small neural networks, and their loss (needs `torch`).

Both arms add one correction, in points of capacity, to all 11 stored quantiles of a row. Both read
the look-back features of the row's entry (`e00706_s2_data`) and train on the CRPS of the corrected
quantiles, which is twice the pinball loss averaged over the 11 levels, against the target power.

- Arm N-gate (`GateNet`): reads the 16 look-back offsets of the row and outputs one gate in [0, 1]
  per offset. The 16 offsets are, for each of the windows of 1, 3, 7, and 14 days (exponentially
  weighted): the global offset of the row's lead band, the irradiance slope times the row's
  irradiance, the cold slope times the row's cold driver, and the neighbours' offset of the lead
  band. The network also reads the lead, the row's two drivers, and the no-neighbour flag, so a gate
  can fade with lead. The correction is the sum of gate times offset. The loss is the CRPS alone.
- Arm N-flag (`FlagNet`): reads the 11 switching features of the entry and outputs the probability
  `p` that the step in the window is a transfer. The correction is the slow part of arm G-lb's
  offset plus `1 - p` times the step part. The loss is the CRPS plus 0.05 times the binary
  cross-entropy of `p` against the generator's transfer flag, on the rows whose flag is known.

Each network has two hidden layers of 32 units. The gate network starts with the gate of the 14-day
global offset at 0.75, the factor that stage 1 applies, and the other 15 gates near 0.02. The flag
network starts with `p` near 0.12.

Two caveats limit the reading of the contrasts. Arm N-flag has no lead input, so a contrast of
N-gate with N-flag on real-row CRPS partly measures the fading with lead that only N-gate can
learn. And the lead input of arm N-gate, with every forecast issued at 12:00 UTC, also encodes the
hour of day of the target; that is allowed, and the fading with lead cannot be separated from a
daily profile of the gates.
"""

from typing import Final, NamedTuple

import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.eval_protocol.metrics import LEVELS
from torch import nn  # ty: ignore[unresolved-import]
from torch.nn import functional  # ty: ignore[unresolved-import]

type Arm = str
HIDDEN: Final[int] = 32
N_GATE_INPUTS: Final[int] = 16
N_GATE_CONTEXT: Final[int] = 4
N_FLAG_INPUTS: Final[int] = 11
N_FLAG_OWN: Final[int] = 6
NO_NEIGHBOUR_COLUMN: Final[int] = 10
OFFSET_SCALE: Final[float] = 5.0
"""Offsets are divided by this many points before the gate network reads them."""
LEAD_SCALE: Final[float] = 672.0
BCE_WEIGHT: Final[float] = 0.05
GATE_BIAS_START: Final[float] = -4.0
"""The starting bias of the 15 gates other than the 14-day global gate (a gate of about 0.02)."""
GATE_14D_GLOBAL_INDEX: Final[int] = 12
"""Position of the 14-day global offset among the 16 offsets (window 3 of 4, offset type 0 of 4)."""
GATE_14D_GLOBAL_START: Final[float] = 0.75
"""The starting gate of the 14-day global offset, the factor of stage 1's arm G-lb."""
FLAG_BIAS_START: Final[float] = -2.0


class Batch(NamedTuple):
    """A batch of rows, all tensors on one device."""

    entry: torch.Tensor
    """Index of each row's entry in its source's table, long, (B,)."""
    band: torch.Tensor
    """Lead band of each row, long, (B,)."""
    h: torch.Tensor
    """Row of the forecast, 0 to 671, long, (B,)."""
    q: torch.Tensor
    """The base forecast's 11 quantiles in points, (B, 11)."""
    y: torch.Tensor
    """The target power in points, (B,)."""
    x: torch.Tensor
    """The row's irradiance and cold drivers (NaN replaced by 0), (B, 2)."""
    flag_y: torch.Tensor
    """The generator's transfer flag, long: 1, 0, or -1 when unknown, (B,)."""


class Output(NamedTuple):
    """What a network returns for a batch."""

    correction: torch.Tensor
    """The correction added to all 11 quantiles, in points, (B,)."""
    gates: torch.Tensor | None
    """The gates of arm N-gate, (B, 16); None for arm N-flag."""
    logit: torch.Tensor | None
    """The logit of arm N-flag's switching probability, (B,); None for arm N-gate."""


class GateNet(nn.Module):
    """Arm N-gate: a gate in [0, 1] per look-back offset."""

    def __init__(self) -> None:
        """Build the network: two hidden layers of 32 units and one gate per offset."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(N_GATE_INPUTS + N_GATE_CONTEXT, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN, N_GATE_INPUTS)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.fill_(GATE_BIAS_START)
            self.head.bias[GATE_14D_GLOBAL_INDEX] = float(
                np.log(GATE_14D_GLOBAL_START / (1.0 - GATE_14D_GLOBAL_START))
            )

    def forward(self, *, offsets: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return the gates, (B, 16), from the offsets (B, 16) in points and the context (B, 4)."""
        features = torch.cat([offsets / OFFSET_SCALE, context], dim=1)
        return torch.sigmoid(self.head(self.body(features)))


class FlagNet(nn.Module):
    """Arm N-flag: the logit of the probability that the step in the window is a transfer."""

    def __init__(self) -> None:
        """Build the network: two hidden layers of 32 units and one logit."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(N_FLAG_INPUTS, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN, 1)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.fill_(FLAG_BIAS_START)

    def forward(self, *, flags: torch.Tensor) -> torch.Tensor:
        """Return the logits, (B,), from the 11 switching features (B, 11)."""
        return self.head(self.body(flags))[:, 0]


def new_network(*, arm: Arm) -> nn.Module:
    """Return a freshly initialised network for `arm`."""
    return GateNet() if arm == "n-gate" else FlagNet()


def levels_tensor(*, device: torch.device) -> torch.Tensor:
    """Return the 11 quantile levels as a float32 tensor on `device`."""
    return torch.as_tensor(LEVELS.astype(np.float32), device=device)


def crps_per_row(
    *, y: torch.Tensor, base: torch.Tensor, correction: torch.Tensor, levels: torch.Tensor
) -> torch.Tensor:
    """Return each row's CRPS in points: twice the pinball loss averaged over the 11 levels."""
    error = y[:, None] - (base + correction[:, None])
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    return 2.0 * pinball.mean(dim=1)


def apply_network(
    *,
    arm: Arm,
    network: nn.Module,
    table: dict[str, torch.Tensor],
    batch: Batch,
    zero_neighbours: bool = False,
) -> Output:
    """Return the correction of `network` for a batch of rows whose entries are in `table`.

    Args:
        arm: Which arm the network is.
        network: The network.
        table: The entries' tensors: `gate` (E, 4, 6, 3), `nb_gate` (E, 4, 6), `box` (E, 6),
            `slow` (E, 6), and `flag` (E, 11), with the neighbour arrays of one graph.
        batch: The rows.
        zero_neighbours: Remove the neighbours: the neighbours' offsets and the neighbour switching
            features (the neighbour count included) are set to zero, and the no-neighbour flag is
            set to 1, as for a series without neighbours.

    Returns:
        The correction, and the gates or the logit.
    """
    if arm == "n-gate":
        own = table["gate"][batch.entry, :, batch.band]
        neighbours = table["nb_gate"][batch.entry, :, batch.band]
        no_neighbour = table["flag"][batch.entry, NO_NEIGHBOUR_COLUMN]
        if zero_neighbours:
            neighbours = torch.zeros_like(neighbours)
            no_neighbour = torch.ones_like(no_neighbour)
        offsets = torch.stack(
            [own[..., 0], own[..., 1] * batch.x[:, 0:1], own[..., 2] * batch.x[:, 1:2], neighbours],
            dim=2,
        ).reshape(len(batch.entry), N_GATE_INPUTS)
        context = torch.stack(
            [batch.h.float() / LEAD_SCALE, batch.x[:, 0], batch.x[:, 1], no_neighbour], dim=1
        )
        gates = network(offsets=offsets, context=context)
        return Output(correction=(gates * offsets).sum(dim=1), gates=gates, logit=None)
    flags = table["flag"][batch.entry]
    if zero_neighbours:
        flags = torch.cat([flags[:, :N_FLAG_OWN], torch.zeros_like(flags[:, N_FLAG_OWN:])], dim=1)
        flags[:, NO_NEIGHBOUR_COLUMN] = 1.0
    slow = table["slow"][batch.entry, batch.band]
    box = table["box"][batch.entry, batch.band]
    logit = network(flags=flags)
    switching = torch.sigmoid(logit)
    return Output(correction=slow + (1.0 - switching) * (box - slow), gates=None, logit=logit)


def training_loss(
    *,
    arm: Arm,
    network: nn.Module,
    table: dict[str, torch.Tensor],
    batch: Batch,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the training loss of a batch, and its CRPS part.

    The loss is the mean CRPS, plus for arm N-flag `BCE_WEIGHT` times the mean binary cross-entropy
    over the rows whose transfer flag is known.
    """
    output = apply_network(arm=arm, network=network, table=table, batch=batch)
    crps = crps_per_row(y=batch.y, base=batch.q, correction=output.correction, levels=levels).mean()
    loss = crps
    if arm == "n-flag" and output.logit is not None:
        known = batch.flag_y >= 0
        if bool(known.any()):
            entropy = functional.binary_cross_entropy_with_logits(
                output.logit[known], batch.flag_y[known].float()
            )
            loss = loss + BCE_WEIGHT * entropy
    return loss, crps

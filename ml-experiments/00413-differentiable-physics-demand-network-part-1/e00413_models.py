"""The shape-and-response-library network of experiment 413.

`PhysicsMdn` writes the Primary's standardised power as a mixture of Gaussians whose component means
are a physics output `mu` plus learned offsets. `mu` is a Primary-specific mix of shared learned
daily shapes, plus a Primary-specific mix of shared monotone response curves of weather (scaled by a
time-of-day gate), plus yearly harmonics. `PlainWrapper` is the pooled plain network of experiment
397, reading the same input matrix without the yearly columns.

The input matrix is experiment 397's 46 `summary` columns followed by 4 yearly columns (sine and
cosine of the day of the year at one and two cycles), `N_YEARLY` in all.
"""

from dataclasses import dataclass
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_YEARLY: Final[int] = 4
ARMS: Final[tuple[str, ...]] = ("D0", "D0Y", "D1", "D2", "D3")
"""D0Y is D0 plus the four yearly columns, the control for the yearly term of D1 to D3."""
K_SHAPES: Final[int] = 8
N_KNOTS: Final[int] = 16
EMBEDDING: Final[int] = 8
SHAPE_HIDDEN: Final[int] = 32
REFERENCE_ROWS: Final[int] = 4096
"""Training rows whose shapes set each shape's mean and root-mean-square."""
RMS_FLOOR: Final[float] = 1e-2
GATE_LIMIT: Final[float] = 1.0
"""Each gate coefficient is bounded to plus or minus this by a tanh."""
GATE_EXPONENT_LIMIT: Final[float] = 3.0
"""The gate's exponent `c . shape` is clamped to plus or minus this."""


@dataclass(frozen=True)
class Columns:
    """Column indices of the input matrix that the physics network reads.

    Attributes:
        calendar_holidays: The calendar and holiday columns (the shape network's input).
        lags_lead_spread: The lag, lead, and ensemble-spread columns (the residual mixture's input
            in arms D2 and D3).
        temperature: The ensemble-mean temperature column (standardised).
        shortwave: The ensemble-mean shortwave column (standardised).
        cos_zenith: The cosine of the solar zenith (not standardised).
        yearly: The yearly harmonic columns, at the end of the matrix.
        n_base: The number of columns of experiment 397's matrix.
        temperature_reference: The reference temperature in standardised units.
        shortwave_zero: Zero shortwave in standardised units.
        temperature_per_std: Degrees Celsius per standardised unit, to label plots.
    """

    calendar_holidays: list[int]
    lags_lead_spread: list[int]
    temperature: int
    shortwave: int
    cos_zenith: int
    yearly: list[int]
    n_base: int
    temperature_reference: float
    shortwave_zero: float
    temperature_per_std: float


class MonotoneRamps(nn.Module):
    """Monotone curves of one variable, each zero at a reference value and of unit root-mean-square.

    A curve is `sum_j softplus(a_j) * clamp(sign * (v - k_j), 0, gap)` over knots `k_j` that start
    at `reference` and run away from it by `span`, so the curve is non-decreasing in the distance
    from the reference, zero on the other side, and piecewise linear. `direction` is +1 for curves
    that rise above the reference and -1 for curves that rise below it. `sign` multiplies the whole
    curve, so a decreasing response (solar lowering net demand) has sign -1.
    """

    knots: torch.Tensor
    grid: torch.Tensor

    def __init__(
        self,
        *,
        n_curves: int,
        reference: float,
        span: float,
        direction: int,
        sign: float,
        grid_points: int = 101,
    ) -> None:
        """Set the knots, the slopes (zero logits give slope 0.69), and the normalising grid."""
        super().__init__()
        gap = span / N_KNOTS
        self.direction, self.sign, self.gap = direction, sign, gap
        self.register_buffer(
            "knots", reference + direction * gap * torch.arange(N_KNOTS, dtype=torch.float32)
        )
        self.logits = nn.Parameter(torch.randn(n_curves, N_KNOTS) * 0.5)
        self.register_buffer("grid", reference + direction * torch.linspace(0.0, span, grid_points))

    def raw(self, v: torch.Tensor) -> torch.Tensor:
        """Return the un-normalised curves at `v`, shape (rows, n_curves)."""
        ramps = torch.clamp(self.direction * (v[:, None] - self.knots), min=0.0, max=self.gap)
        return self.sign * (ramps @ nn.functional.softplus(self.logits).T)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Return the curves at `v`, each divided by its root-mean-square over the grid."""
        grid_rms = self.raw(self.grid).pow(2).mean(0).sqrt().clamp_min(RMS_FLOOR)
        return self.raw(v) / grid_rms


class PhysicsMdn(nn.Module):
    """The library network; the arm sets which libraries and which residual inputs it uses.

    Arm D1 has the shape library only and a residual mixture that reads every input. Arms D2 and D3
    have both libraries and a residual mixture that reads only the lags, the lead, the ensemble
    spread, and `mu`. Arm D3 has one fleet-wide mix (every row uses the Primary index 0).
    """

    reference_x: torch.Tensor

    def __init__(
        self,
        *,
        arm: str,
        columns: Columns,
        n_primaries: int,
        reference_x: torch.Tensor,
        hidden: int = 64,
        layers: int = 2,
        components: int = 3,
    ) -> None:
        """Build the libraries, the mixing heads, and the residual mixture."""
        super().__init__()
        if arm not in ("D1", "D2", "D3"):
            msg = f"unknown arm {arm!r}"
            raise ValueError(msg)
        self.arm, self.cols, self.components = arm, columns, components
        self.shared = arm == "D3"
        self.with_responses = arm in ("D2", "D3")
        n_mix = 1 if self.shared else n_primaries
        self.register_buffer("reference_x", reference_x)
        n_calendar = len(columns.calendar_holidays)
        self.shape_net = base.mlp(n_in=n_calendar, hidden=SHAPE_HIDDEN, layers=2, n_out=K_SHAPES)
        self.embedding = nn.Embedding(n_mix, EMBEDDING)
        self.to_shape_weights = nn.Linear(EMBEDDING, K_SHAPES)
        self.to_amplitude = nn.Linear(EMBEDDING, 2)
        self.level = nn.Embedding(n_mix, 1)
        nn.init.zeros_(self.level.weight)
        self.yearly_coefficients = nn.Parameter(torch.zeros(n_mix, N_YEARLY))
        if self.with_responses:
            self.heating = MonotoneRamps(
                n_curves=3,
                reference=columns.temperature_reference,
                span=5.0 + columns.temperature_reference,
                direction=-1,
                sign=1.0,
            )
            self.cooling = MonotoneRamps(
                n_curves=1,
                reference=columns.temperature_reference,
                span=5.0 - columns.temperature_reference,
                direction=1,
                sign=1.0,
            )
            self.shortwave = MonotoneRamps(
                n_curves=1,
                reference=columns.shortwave_zero,
                span=5.0 - columns.shortwave_zero,
                direction=1,
                sign=-1.0,
            )
            self.zenith = MonotoneRamps(n_curves=1, reference=0.0, span=1.0, direction=1, sign=-1.0)
            self.to_response_weights = nn.Linear(EMBEDDING, 6)
            self.to_gate = nn.Linear(EMBEDDING, K_SHAPES)
            width = len(columns.lags_lead_spread) + 1
        else:
            width = columns.n_base + 1 + EMBEDDING
        self.head = base.mlp(n_in=width, hidden=hidden, layers=layers, n_out=3 * components)
        base.init_head(head=self.head, components=components)
        nn.init.zeros_(self.to_shape_weights.weight)
        nn.init.zeros_(self.to_shape_weights.bias)
        with torch.no_grad():
            self.to_amplitude.weight.zero_()
            self.to_amplitude.bias.fill_(0.0)
        if self.with_responses:
            nn.init.zeros_(self.to_response_weights.weight)
            nn.init.zeros_(self.to_response_weights.bias)
            nn.init.zeros_(self.to_gate.weight)
            nn.init.zeros_(self.to_gate.bias)

    def shapes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shape curves at the rows of `x` and at the reference rows.

        Both are normalised to zero mean and unit root-mean-square over the reference rows, so the
        scale of a shape never trades with its weight.
        """
        calendar = self.cols.calendar_holidays
        ref = self.shape_net(self.reference_x[:, calendar])
        mean = ref.mean(0)
        rms = (ref - mean).pow(2).mean(0).sqrt().clamp_min(RMS_FLOOR)
        return (self.shape_net(x[:, calendar]) - mean) / rms, (ref - mean) / rms

    def responses(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 6 response curves at the rows of `x`, shape (rows, 6)."""
        c = self.cols
        temperature = x[:, c.temperature]
        return torch.cat(
            [
                self.heating(temperature),
                self.cooling(temperature),
                self.shortwave(x[:, c.shortwave]),
                self.zenith(x[:, c.cos_zenith].clamp_min(0.0)),
            ],
            dim=1,
        )

    def physics(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return `mu`, the physics output in standardised units, shape (rows,)."""
        index = torch.zeros_like(primary) if self.shared else primary
        mix = self.embedding(index)
        shape, reference_shape = self.shapes(x)
        a = torch.softmax(self.to_shape_weights(mix), dim=-1)
        amplitude = nn.functional.softplus(self.to_amplitude(mix) + 0.5413)
        mu = self.level(index).squeeze(-1) + amplitude[:, 0] * (a * shape).sum(-1)
        mu = mu + (x[:, self.cols.yearly] * self.yearly_coefficients[index]).sum(-1)
        if self.with_responses:
            b = torch.softmax(self.to_response_weights(mix), dim=-1)
            c = GATE_LIMIT * torch.tanh(self.to_gate(mix))
            gate = torch.exp((c * shape).sum(-1).clamp(-GATE_EXPONENT_LIMIT, GATE_EXPONENT_LIMIT))
            all_c = GATE_LIMIT * torch.tanh(
                self.to_gate(
                    self.embedding(torch.arange(len(self.embedding.weight), device=x.device))
                )
            )
            norm = torch.exp(
                (reference_shape @ all_c.T).clamp(-GATE_EXPONENT_LIMIT, GATE_EXPONENT_LIMIT)
            ).mean(0)
            mu = mu + amplitude[:, 1] * (b * self.responses(x)).sum(-1) * gate / norm[index]
        return mu

    def forward(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the raw mixture outputs (log weight logits, means, log widths)."""
        mu = self.physics(x, primary)
        if self.with_responses:
            head_in = torch.cat([x[:, self.cols.lags_lead_spread], mu[:, None]], dim=1)
        else:
            index = torch.zeros_like(primary) if self.shared else primary
            head_in = torch.cat(
                [x[:, : self.cols.n_base], self.embedding(index), mu[:, None]], dim=1
            )
        out = self.head(head_in)
        k = self.components
        return torch.cat([out[:, :k], out[:, k : 2 * k] + mu[:, None], out[:, 2 * k :]], dim=1)


class PlainWrapper(base.PlainMdn):
    """Experiment 397's pooled plain network, reading only the first `n_base` columns."""

    def __init__(
        self, *, n_base: int, n_primaries: int, hidden: int, layers: int, components: int
    ) -> None:
        """Build the pooled plain network of `n_base` inputs with a Primary embedding."""
        super().__init__(
            n_inputs=n_base,
            hidden=hidden,
            layers=layers,
            components=components,
            n_primaries=n_primaries,
        )
        self.n_base = n_base

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the raw mixture outputs, ignoring the yearly columns."""
        return super().forward(x[:, : self.n_base], primary)

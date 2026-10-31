# ruff: noqa: E501, PLR0915, PT018
"""Networks and loss terms of experiment 446's Brief B arms (U2, U2i, U2c) on the S1 base.

**S1 base.** The location path is experiment 397's pooled plain network on the S1 inputs: the
non-lag columns, the nine lag columns (three lag values, three missing flags, three lag ages), and
the three single-step surprises at the lags (experiment 428's `S1`). Its input row is therefore
`[x | lag values (3) | missing flags (3) | ages (3) | surprise (3)]`.

**Scale-only path.** A small multilayer perceptron reads eleven numbers: the absolute surprise at
each lag (3), the lag ages (3), the missing flags (3), and the two lead columns `lead` and
`lead_sqrt`. It returns one number `g` per mixture component, and the standard deviation of that
component is multiplied by `1 + softplus(g)`. The last layer starts at zero weights with bias
`SCALE_INIT`, so every factor starts within 0.01 of 1 and the factor can only widen. The path
touches only the log standard deviation outputs: the mixture weights and means come from the
location path alone, whose parameters receive no gradient from the scale path (checked in the
selftest, along with the scale features' gradient).

**Loss terms.** `interval_score` is the interval score of the central interval between two
quantiles (Gneiting and Raftery's interval score). `tail_quantiles_st` returns the mixture's 1%, 5%,
95%, and 99% quantiles by bisection with an implicit-function gradient, so the interval score and
the coverage-matching penalty are differentiable in the mixture parameters. `coverage_penalty` is
the squared gap between a smooth coverage of each tail quantile and its nominal level: it is not a
proper scoring rule, and arm U2c is labelled improper because of it. `log_factor_penalty` is the
mean log factor on clean rows.

```bash
uv run --no-sync python ml-experiments/00446-*/e00446_models.py selftest
```
"""

import math
import sys
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

SCALE_INIT: Final[float] = -5.0
"""Bias of the scale path's last layer: softplus(-5) = 0.0067, so factors start at 1.0067."""
SCALE_HIDDEN: Final[int] = 16
LAG_COLUMNS: Final[int] = 12
"""Lag values, missing flags, ages, and surprises (three lags each)."""
N_SCALE_FEATURES: Final[int] = 11
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
TAIL_INDEX: Final[dict[float, int]] = {lv: base.LEVELS.index(lv) for lv in TAIL_LEVELS}
INTERVAL_ALPHAS: Final[tuple[tuple[float, float, float], ...]] = (
    (0.05, 0.95, 0.10),
    (0.01, 0.99, 0.02),
)
"""The 90% and the 98% central intervals: lower level, upper level, and alpha."""
SQRT2: Final[float] = math.sqrt(2.0)


def lead_columns() -> tuple[int, int]:
    """Return the positions of `lead` and `lead_sqrt` among the non-lag columns."""
    import e00428_data as data  # ty: ignore[unresolved-import]

    _, names, _ = data.non_lag_layout()
    return names.index("lead"), names.index("lead_sqrt")


class ScaleMdn(nn.Module):
    """S1's location network plus a scale-only path that multiplies each component's spread."""

    def __init__(
        self,
        *,
        n_inputs: int,
        n_primaries: int,
        lead_cols: tuple[int, int],
        hidden: int = 64,
        layers: int = 2,
        components: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Build the location network (over `n_inputs` non-lag columns plus 12) and the scale path."""
        super().__init__()
        self.components = components
        self.lead_cols = lead_cols
        self.location = base.PlainMdn(
            n_inputs=n_inputs + LAG_COLUMNS,
            hidden=hidden,
            layers=layers,
            components=components,
            dropout=dropout,
            n_primaries=n_primaries,
        )
        last = nn.Linear(SCALE_HIDDEN, components)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, SCALE_INIT)
        self.scale_net = nn.Sequential(nn.Linear(N_SCALE_FEATURES, SCALE_HIDDEN), nn.GELU(), last)

    def scale_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the eleven scale-path inputs (rows, 11) from an S1 input row."""
        surprise = x[:, -3:].abs()
        ages = x[:, -6:-3]
        missing = x[:, -9:-6]
        lead = x[:, list(self.lead_cols)]
        return torch.cat([surprise, ages, missing, lead], dim=1)

    def factors(self, x: torch.Tensor, scale_x: torch.Tensor | None = None) -> torch.Tensor:
        """Return the standard-deviation factor of each component, shape (rows, components)."""
        features = self.scale_features(x) if scale_x is None else scale_x
        return 1.0 + nn.functional.softplus(self.scale_net(features))

    def forward_parts(
        self, x: torch.Tensor, primary: torch.Tensor, scale_x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the raw mixture outputs (with the factor in the spread) and the factors."""
        out = self.location(x, primary)
        factor = self.factors(x, scale_x)
        k = self.components
        scaled = torch.cat([out[:, : 2 * k], out[:, 2 * k :] + torch.log(factor)], dim=-1)
        return scaled, factor

    def forward(
        self, x: torch.Tensor, primary: torch.Tensor, scale_x: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the raw mixture outputs (same layout as experiment 397's networks)."""
        return self.forward_parts(x, primary, scale_x)[0]


def tail_quantiles_st(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> dict[float, torch.Tensor]:
    """Return the 1%, 5%, 95%, and 99% quantiles (rows,) of mixtures, differentiable in the parts.

    The value is found by bisection without gradient. The gradient is the implicit-function one,
    `dq = -dF / f` where `F` is the mixture CDF and `f` its density at the found quantile.
    """
    with torch.no_grad():
        found = base.mixture_quantiles(log_w=log_w.detach(), mean=mean.detach(), std=std.detach())
    out = {}
    w = log_w.exp()
    for level in TAIL_LEVELS:
        q = found[:, TAIL_INDEX[level]].detach()
        z = (q[:, None] - mean) / std
        cdf = (w * 0.5 * (1.0 + torch.erf(z / SQRT2))).sum(-1)
        pdf = (w * torch.exp(-0.5 * z**2) / (std * math.sqrt(2.0 * math.pi))).sum(-1)
        out[level] = q - (cdf - level) / pdf.detach().clamp(min=1e-3)
    return out


def interval_score(
    *, lower: torch.Tensor, upper: torch.Tensor, y: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Return the interval score per row: width plus 2/alpha times the miss distance."""
    return (
        (upper - lower)
        + (2.0 / alpha) * torch.relu(lower - y)
        + (2.0 / alpha) * torch.relu(y - upper)
    )


def interval_loss(
    *, quantiles: dict[float, torch.Tensor], y: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Return the mean 90% plus 98% interval score over the rows flagged in `rows`."""
    if not bool(rows.any()):
        return y.sum() * 0.0
    total = torch.zeros((), device=y.device)
    for low, high, alpha in INTERVAL_ALPHAS:
        score = interval_score(lower=quantiles[low], upper=quantiles[high], y=y, alpha=alpha)
        total = total + score[rows].mean()
    return total


def coverage_penalty(
    *, quantiles: dict[float, torch.Tensor], y: torch.Tensor, rows: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:  # fmt: skip
    """Return the sum over tails of (smooth coverage - nominal level)^2 on the flagged rows.

    The smooth coverage of a quantile is the mean of `sigmoid((q - y) / temperature)`, which tends
    to the share of targets at or below the quantile as the temperature falls. Not a proper
    scoring rule.
    """
    if not bool(rows.any()):
        return y.sum() * 0.0
    total = torch.zeros((), device=y.device)
    for level in TAIL_LEVELS:
        covered = torch.sigmoid((quantiles[level][rows] - y[rows]) / temperature).mean()
        total = total + (covered - level) ** 2
    return total


def log_factor_penalty(*, factor: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Return the mean log factor (over components, then over the flagged rows)."""
    if not bool(rows.any()):
        return factor.sum() * 0.0
    return torch.log(factor).mean(dim=1)[rows].mean()


def count_parameters(*, model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return base.count_parameters(model=model)


def selftest() -> None:
    """Check the initial factor, the gradient wall, the interval score, and the penalties."""
    torch.manual_seed(0)
    model = ScaleMdn(n_inputs=6, n_primaries=4, lead_cols=(2, 3), components=3)
    rows = 64
    x = torch.randn(rows, 6 + LAG_COLUMNS)
    x[:, -3:] = torch.randn(rows, 3) * 4.0  # large surprises
    primary = torch.randint(0, 4, (rows,))
    out, factor = model.forward_parts(x, primary)
    assert float((factor - 1.0).abs().max()) < 0.01, "factors must start within 0.01 of 1"
    clean_x = x.clone()
    clean_x[:, -3:] = 0.0
    assert float((model.factors(clean_x) - 1.0).abs().max()) < 0.01, "clean rows start at 1"
    assert out.shape == (rows, 9)
    # Make the scale path non-trivial so its gradient is non-zero.
    last_layer = model.scale_net[-1]
    assert isinstance(last_layer, nn.Linear)
    nn.init.normal_(last_layer.weight, std=0.5)
    features = model.scale_features(x).detach().requires_grad_(True)
    out, _ = model.forward_parts(x, primary, scale_x=features)
    k = 3
    location_part = out[:, : 2 * k].sum()
    spread_part = out[:, 2 * k :].sum()
    (g_loc,) = torch.autograd.grad(location_part, features, allow_unused=True, retain_graph=True)
    assert g_loc is None or float(g_loc.abs().max()) == 0.0, (
        "location outputs depend on scale input"
    )
    (g_spread,) = torch.autograd.grad(spread_part, features)
    assert float(g_spread.abs().max()) > 0.0, "spread outputs must depend on the scale input"
    model.zero_grad()
    out, _ = model.forward_parts(x, primary)
    out[:, : 2 * k].sum().backward()
    assert all(
        p.grad is None or float(p.grad.abs().max()) == 0.0 for p in model.scale_net.parameters()
    ), "scale-path parameters must receive no gradient from the location outputs"
    reference = model.location(x, primary)
    perturbed = ScaleMdn(n_inputs=6, n_primaries=4, lead_cols=(2, 3), components=3)
    perturbed.location.load_state_dict(model.location.state_dict())
    out_a, _ = model.forward_parts(x, primary)
    out_b, _ = perturbed.forward_parts(x, primary)
    assert torch.equal(out_a[:, : 2 * k], reference[:, : 2 * k]), "location differs from base"
    assert torch.equal(out_a[:, : 2 * k], out_b[:, : 2 * k]), "scale path leaked into location"
    # Interval score on a toy interval [-1, 1] at alpha 0.1.
    lo, hi = torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0])
    score = interval_score(lower=lo, upper=hi, y=torch.tensor([0.0, 2.0, -3.0]), alpha=0.1)
    assert torch.allclose(score, torch.tensor([2.0, 22.0, 42.0])), score
    # Coverage penalty: every quantile at 0, targets half below and half above.
    flat = {lv: torch.zeros(4) for lv in TAIL_LEVELS}
    y = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    penalty = coverage_penalty(
        quantiles=flat, y=y, rows=torch.ones(4, dtype=torch.bool), temperature=1e-3
    )
    expected = sum((0.5 - lv) ** 2 for lv in TAIL_LEVELS)
    assert abs(float(penalty) - expected) < 1e-4, (float(penalty), expected)
    # Implicit quantile gradient on one Gaussian: dq/dmean = 1, dq/dstd = z_level.
    mean = torch.tensor([[0.3]], requires_grad=True)
    std = torch.tensor([[1.7]], requires_grad=True)
    quantiles = tail_quantiles_st(log_w=torch.zeros(1, 1), mean=mean, std=std)
    z95 = 1.6448536269514722
    assert abs(float(quantiles[0.95]) - (0.3 + 1.7 * z95)) < 1e-4
    quantiles[0.95].sum().backward()
    assert mean.grad is not None and std.grad is not None
    assert abs(float(mean.grad) - 1.0) < 1e-3 and abs(float(std.grad) - z95) < 1e-3, (
        mean.grad, std.grad)  # fmt: skip
    # Mean log factor on the flagged rows only.
    f = torch.tensor([[math.e, math.e], [1.0, 1.0], [math.e**2, math.e**2]])
    mask = torch.tensor([True, True, False])
    assert abs(float(log_factor_penalty(factor=f, rows=mask)) - 0.5) < 1e-6
    print("selftest passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()

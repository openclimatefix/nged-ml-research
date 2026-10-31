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
location path alone. The scale path adds its own inputs; the location path is S1's network,
unchanged (it takes the signed surprise directly), and receives no gradient from the scale-only loss
terms. The interval-score and coverage-penalty terms of arms U2i and U2c are computed with
`scale_only_parts`, which detaches the mixture weights, means, and base spread so that only the
factor `g` receives gradient from them (checked in the selftest, along with the scale features'
gradient). The log-likelihood and the factor penalty are the same for every scale arm.

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
import numpy as np
import torch  # ty: ignore[unresolved-import]
from scipy.special import gammaln, stdtr, stdtrit
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
DF_MIN: Final[float] = 2.5
"""Smallest degrees of freedom a Student-t component can take (arm T1)."""
DF_MAX: Final[float] = 1_000.0
DF_INIT: Final[float] = 30.0
MAX_NEWTON_STEPS: Final[int] = 100


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


def scale_only_parts(
    *, out: torch.Tensor, factor: torch.Tensor, components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return log weights, means, and standard deviations with only `factor` carrying gradient.

    The mixture weights, means, and base spread (the spread output before the factor) are detached.
    """
    k = components
    log_factor = torch.log(factor)
    base_spread = (out[:, 2 * k :] - log_factor).detach()
    spread = base_spread + log_factor
    return base.mixture_parts(
        out=torch.cat([out[:, : 2 * k].detach(), spread], dim=-1), components=k
    )


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


class StudentMdn(nn.Module):
    """S1's location network with Student-t mixture components (arm T1).

    The output row is `[logits (K) | means (K) | log scales (K) | raw degrees of freedom (K)]`. The
    first `3 K` columns are read exactly as in experiment 397's Gaussian head (`base.mixture_parts`),
    with the third block a scale rather than a standard deviation. The degrees of freedom of each
    component are `DF_MIN + softplus(raw)`, so they cannot fall below `DF_MIN`; they start at
    `DF_INIT`, where the density is within a few percent of Gaussian.
    """

    def __init__(
        self,
        *,
        n_inputs: int,
        n_primaries: int,
        hidden: int = 64,
        layers: int = 2,
        components: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Build S1's network (over `n_inputs` non-lag columns plus 12) with a wider output layer."""
        super().__init__()
        self.components = components
        self.location = base.PlainMdn(
            n_inputs=n_inputs + LAG_COLUMNS,
            hidden=hidden,
            layers=layers,
            components=components,
            dropout=dropout,
            n_primaries=n_primaries,
        )
        last = nn.Linear(hidden, 4 * components)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(
                torch.tensor(
                    [
                        *base.mixture_head_bias(components=components),
                        *([_raw_df(df=DF_INIT)] * components),
                    ]
                )
            )
        self.location.net[-1] = last

    def forward(self, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the raw outputs, shape (rows, 4 K)."""
        return self.location(x, primary)


def _raw_df(*, df: float) -> float:
    """Return the raw output whose degrees of freedom are `df` (the inverse of `student_parts`)."""
    return math.log(math.expm1(df - DF_MIN))


def student_parts(
    *, out: torch.Tensor, components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split raw outputs (..., 4 K) into log weights, means, scales, and degrees of freedom."""
    k = components
    log_w, mean, scale = base.mixture_parts(out=out[..., : 3 * k], components=k)
    df = (DF_MIN + nn.functional.softplus(out[..., 3 * k :])).clamp(max=DF_MAX)
    return log_w, mean, scale, df


def student_log_pdf(*, z: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """Return the log density of the standard Student-t distribution at `z` (same shape as `z`)."""
    half = 0.5 * (df + 1.0)
    return (
        torch.lgamma(half)
        - torch.lgamma(0.5 * df)
        - 0.5 * torch.log(df * math.pi)
        - half * torch.log1p(z**2 / df)
    )


def student_mixture_nll(*, out: torch.Tensor, z: torch.Tensor, components: int) -> torch.Tensor:
    """Return the negative log-likelihood of `z` under the Student-t mixture, for each row."""
    log_w, mean, scale, df = student_parts(out=out, components=components)
    log_pdf = student_log_pdf(z=(z[..., None] - mean) / scale, df=df) - torch.log(scale)
    return -torch.logsumexp(log_w + log_pdf, dim=-1)


def student_mixture_quantiles(
    *,
    log_w: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    df: torch.Tensor,
    tolerance: float = 1e-9,
) -> torch.Tensor:
    """Return the 11 quantiles (rows, 11) of Student-t mixtures (rows, components); no gradient.

    The mixture CDF is evaluated on the CPU with `scipy.special.stdtr`. A mixture's quantile lies
    between the smallest and the largest of its components' quantiles, which gives the starting
    bracket. Each quantile is then found by Newton steps that fall back to bisection whenever a step
    leaves the bracket, and only the quantiles whose CDF is still more than `tolerance` from its
    level are iterated.
    """
    device = mean.device
    n_rows = log_w.shape[0]
    levels = np.asarray(base.LEVELS, dtype=np.float64)
    n_levels = len(levels)
    w = log_w.detach().double().exp().cpu().numpy()
    m = mean.detach().double().cpu().numpy()
    s = scale.detach().double().cpu().numpy()
    v = df.detach().double().cpu().numpy()
    component_q = m[:, None, :] + s[:, None, :] * stdtrit(v[:, None, :], levels[None, :, None])
    lo, hi = component_q.min(-1).ravel(), component_q.max(-1).ravel()
    x = (w[:, None, :] * component_q).sum(-1).ravel()
    level = np.tile(levels, n_rows)
    row = np.repeat(np.arange(n_rows), n_levels)
    log_norm = gammaln(0.5 * (v + 1.0)) - gammaln(0.5 * v) - 0.5 * np.log(v * np.pi)
    active = np.arange(len(x))
    for _ in range(MAX_NEWTON_STEPS):
        r = row[active]
        wa, ma, sa, va, na = w[r], m[r], s[r], v[r], log_norm[r]
        z = (x[active, None] - ma) / sa
        err = (wa * stdtr(va, z)).sum(-1) - level[active]
        pdf = (wa * np.exp(na - 0.5 * (va + 1.0) * np.log1p(z**2 / va)) / sa).sum(-1)
        below = err < 0.0
        lo[active] = np.where(below, x[active], lo[active])
        hi[active] = np.where(below, hi[active], x[active])
        step = x[active] - err / np.maximum(pdf, 1e-300)
        inside = (step > lo[active]) & (step < hi[active])
        x[active] = np.where(inside, step, 0.5 * (lo[active] + hi[active]))
        active = active[np.abs(err) >= tolerance]
        if len(active) == 0:
            break
    return torch.from_numpy(x.reshape(n_rows, n_levels)).to(device=device, dtype=mean.dtype)


def pinball(*, quantile: torch.Tensor, y: torch.Tensor, level: float) -> torch.Tensor:
    """Return the pinball loss per row of a forecast quantile at `level`."""
    err = y - quantile
    return torch.maximum(level * err, (level - 1.0) * err)


def tail_pinball(*, quantiles: dict[float, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """Return, per row, the sum of the pinball losses at the four tail levels (q01, q05, q95, q99)."""
    return sum(
        (pinball(quantile=quantiles[lv], y=y, level=lv) for lv in TAIL_LEVELS),
        start=torch.zeros_like(y),
    )


def tail_weights(
    *, cold: torch.Tensor | None, weight: float, slope: float, like: torch.Tensor
) -> torch.Tensor:
    """Return the per-row weight of the tail term: `weight * (1 + slope * cold)`.

    With `cold` unset (arm T2) every row gets `weight`.
    """
    if cold is None:
        return torch.full_like(like, weight)
    return weight * (1.0 + slope * cold.to(like.dtype))


def output_layer(*, model: nn.Module) -> nn.Linear:
    """Return the final linear layer (the mixture head) of S1's network or any arm built on it."""
    plain = model.location if hasattr(model, "location") else model.net  # ty: ignore[unresolved-attribute]
    assert isinstance(plain, base.PlainMdn)
    last = plain.net[-1]
    assert isinstance(last, nn.Linear)
    return last


def freeze_trunk(*, model: nn.Module) -> list[nn.Parameter]:
    """Stop gradients to every parameter except the mixture head's; return the head's parameters.

    The trunk is the Primary embedding and the hidden layers (and, for arms with a scale path, the
    scale path). The head is the final linear layer that maps the hidden features to the mixture
    parameters.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    head = list(output_layer(model=model).parameters())
    for p in head:
        p.requires_grad_(True)
    return head


def pool_mixtures(
    *, pieces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool mixtures (log weights, means, standard deviations) with equal weight.

    Each model's components are kept and its log weights are shifted by `-log(n)`, so the pooled
    CDF is the average of the models' CDFs. Pooling copies of one mixture reproduces that mixture.
    """
    shift = math.log(len(pieces))
    return (
        torch.cat([p[0] - shift for p in pieces], dim=1),
        torch.cat([p[1] for p in pieces], dim=1),
        torch.cat([p[2] for p in pieces], dim=1),
    )


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
    # Scale-only terms: loss built from `scale_only_parts` reaches only the scale path.
    model.zero_grad()
    out, factor = model.forward_parts(x, primary)
    log_w, mean, std = scale_only_parts(out=out, factor=factor, components=k)
    quantiles = tail_quantiles_st(log_w=log_w, mean=mean, std=std)
    y = torch.randn(rows)
    flagged = torch.ones(rows, dtype=torch.bool)
    scale_loss = interval_loss(quantiles=quantiles, y=y, rows=flagged) + coverage_penalty(
        quantiles=quantiles, y=y, rows=flagged
    )
    scale_loss.backward()
    scale_grad = max(
        float(p.grad.abs().max()) for p in model.scale_net.parameters() if p.grad is not None
    )
    assert scale_grad > 0.0, "scale-only terms must reach the scale path"
    assert all(
        p.grad is None or float(p.grad.abs().max()) == 0.0 for p in model.location.parameters()
    ), "scale-only terms must give the location path no gradient"
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
    selftest_brief_c()
    print("selftest passed")


def selftest_brief_c() -> None:
    """Check the Student-t density and quantiles, the pinball term, the frozen trunk, and pooling."""
    from scipy import stats

    torch.manual_seed(1)
    # Density against scipy.stats.t, including a location and a scale.
    z = torch.linspace(-9.0, 9.0, 41, dtype=torch.float64)
    for df_value in (2.5, 4.0, 17.3, 400.0):
        df = torch.full_like(z, df_value)
        ours = student_log_pdf(z=z, df=df).numpy()
        assert np.allclose(ours, stats.t.logpdf(z.numpy(), df_value), atol=1e-10), df_value
    out = torch.zeros(1, 4, dtype=torch.float64)  # one component, location 0.4, scale 1.7, df 6.2
    out[0, 1], out[0, 2] = 0.4, math.log(1.7)
    out[0, 3] = _raw_df(df=6.2)
    y = torch.tensor([1.3], dtype=torch.float64)
    nll = student_mixture_nll(out=out, z=y, components=1)
    expected = -stats.t.logpdf(1.3, 6.2, loc=0.4, scale=1.7)
    assert abs(float(nll) - expected) < 1e-10, (float(nll), expected)
    log_w, mean, scale, df = student_parts(out=out, components=1)
    assert abs(float(df) - 6.2) < 1e-9 and float(df) >= DF_MIN
    assert float(student_parts(out=torch.full((1, 4), -50.0), components=1)[3]) >= DF_MIN
    # Quantile function against scipy.stats.t.ppf (one component, then the CDF of a mixture).
    q = student_mixture_quantiles(log_w=log_w, mean=mean, scale=scale, df=df)
    ref = stats.t.ppf(base.LEVELS, 6.2, loc=0.4, scale=1.7)
    assert np.allclose(q.numpy()[0], ref, atol=1e-7), (q.numpy()[0], ref)
    w2 = torch.tensor([[0.7, 0.3]], dtype=torch.float64)
    m2 = torch.tensor([[-0.5, 1.5]], dtype=torch.float64)
    s2 = torch.tensor([[0.8, 2.5]], dtype=torch.float64)
    v2 = torch.tensor([[3.0, 9.0]], dtype=torch.float64)
    q2 = student_mixture_quantiles(log_w=w2.log(), mean=m2, scale=s2, df=v2).numpy()[0]
    cdf = 0.7 * stats.t.cdf(q2, 3.0, loc=-0.5, scale=0.8) + 0.3 * stats.t.cdf(
        q2, 9.0, loc=1.5, scale=2.5
    )
    assert np.allclose(cdf, base.LEVELS, atol=1e-8), cdf
    # Initial head: within a few percent of Gaussian in the 1% to 99% quantiles.
    head = StudentMdn(n_inputs=6, n_primaries=4, components=3)
    x0 = torch.randn(8, 6 + LAG_COLUMNS)
    out0 = head(x0, torch.randint(0, 4, (8,))).double()
    lw0, m0, s0, v0 = student_parts(out=out0, components=3)
    assert float((v0 - DF_INIT).abs().max()) < 1e-3 and out0.shape == (8, 12)
    qt = student_mixture_quantiles(log_w=lw0, mean=m0, scale=s0, df=v0)
    qg = base.mixture_quantiles(log_w=lw0, mean=m0, std=s0)
    assert float(((qt - qg).abs() / qg.abs().clamp(min=1.0)).max()) < 0.08
    # Pinball loss, numerically: level 0.95, quantile 1, y 3 gives 0.95 * 2; y -1 gives 0.05 * 2.
    one = torch.tensor([1.0, 1.0])
    pin = pinball(quantile=one, y=torch.tensor([3.0, -1.0]), level=0.95)
    assert torch.allclose(pin, torch.tensor([1.9, 0.1])), pin
    flat = {lv: torch.zeros(2) for lv in TAIL_LEVELS}
    total = tail_pinball(quantiles=flat, y=torch.tensor([2.0, -2.0]))
    assert torch.allclose(total, torch.full((2,), 4.0)), total
    # Tail weights: 0.2 off the cold nights, 0.2 * (1 + slope) on them.
    cold = torch.tensor([False, True, False, True])
    weights = tail_weights(cold=cold, weight=0.2, slope=1.0, like=torch.zeros(4))
    assert torch.allclose(weights, torch.tensor([0.2, 0.4, 0.2, 0.4])), weights
    flat_w = tail_weights(cold=None, weight=0.2, slope=1.0, like=torch.zeros(4))
    assert torch.allclose(flat_w, torch.full((4,), 0.2))
    # Frozen trunk: only the final linear layer receives gradient, for both network classes.
    rows = 32
    x = torch.randn(rows, 6 + LAG_COLUMNS)
    primary = torch.randint(0, 4, (rows,))
    y = torch.randn(rows)
    for model in (
        ScaleMdn(n_inputs=6, n_primaries=4, lead_cols=(2, 3), components=3),
        StudentMdn(n_inputs=6, n_primaries=4, components=3),
    ):
        head_parameters = freeze_trunk(model=model)
        out = model(x, primary)
        lw, mu, sd = base.mixture_parts(out=out[:, :9], components=3)
        loss = tail_pinball(quantiles=tail_quantiles_st(log_w=lw, mean=mu, std=sd), y=y).mean()
        loss.backward()
        head_ids = {id(p) for p in head_parameters}
        for p in model.parameters():
            if id(p) in head_ids:
                assert p.grad is not None and float(p.grad.abs().max()) > 0.0, "head gets gradient"
            else:
                assert p.grad is None or float(p.grad.abs().max()) == 0.0, "trunk got gradient"
    # Pinball term reaches the mixture weights, means, and spreads (not only the scale path).
    model = ScaleMdn(n_inputs=6, n_primaries=4, lead_cols=(2, 3), components=3)
    out, _ = model.forward_parts(x, primary)
    lw, mu, sd = base.mixture_parts(out=out, components=3)
    tail_pinball(quantiles=tail_quantiles_st(log_w=lw, mean=mu, std=sd), y=y).mean().backward()
    head_grad = output_layer(model=model).weight.grad
    assert head_grad is not None and float(head_grad.abs().max()) > 0.0
    # Pooling copies of one forecast reproduces that forecast's quantiles.
    lw1, mu1, sd1 = (
        torch.log_softmax(torch.randn(16, 3), dim=-1).double(),
        torch.randn(16, 3).double(),
        torch.rand(16, 3).double() + 0.3,
    )
    single = base.mixture_quantiles(log_w=lw1, mean=mu1, std=sd1)
    pooled = base.mixture_quantiles(
        log_w=pool_mixtures(pieces=[(lw1, mu1, sd1)] * 5)[0],
        mean=pool_mixtures(pieces=[(lw1, mu1, sd1)] * 5)[1],
        std=pool_mixtures(pieces=[(lw1, mu1, sd1)] * 5)[2],
    )
    assert torch.allclose(single, pooled, atol=1e-9), (single - pooled).abs().max()
    print("selftest brief C passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()

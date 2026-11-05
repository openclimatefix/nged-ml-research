# ruff: noqa: E501
"""Experiment 577: output-side offsets that stay constant between sparse steps, and offsets with a posterior.

All networks are experiment 565's `OutputOffsetDrift` (design 5, D5) with a different knot grid,
penalty, or posterior. The offset is added to the three component means in standardised units. The
knot position `tau` (months from 2019-09-01, fractional) is read from column 37, and every position
is clamped to the newest knot, so a forecast for a valid time after the cutoff reads the newest knot.

- `TvMonthlyOffset` (O1): monthly knots, L1 penalty on the first differences (fused lasso).
- `TvDailyOffset` (O2): daily knots with zero-order hold, the same L1 penalty. Fine-tuning trains the
  newest `TAIL` daily knots and fixes the older ones.
- `GatedMonthlyOffset` (O3): monthly knots that are the cumulative sum of hard-concrete-gated jumps,
  with a penalty on the expected number of open gates.
- `VariationalMixin` (Q2 and Q4): the newest knot has a Gaussian posterior in fine-tuning. Forecasts
  use its mean and add its variance to every mixture component.
"""

from typing import Final

import e00559_model as m  # ty: ignore[unresolved-import]
import e00565_model_variants as var  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

TAU_COL: Final[int] = m.N_INPUTS
TAIL: Final[int] = 28
"""Daily knots that fine-tuning trains (a step inside the last four weeks can be learnt)."""
HC_BETA, HC_GAMMA, HC_ZETA = 2.0 / 3.0, -0.1, 1.1
"""Hard-concrete constants of experiment 573's S2."""
GATE_INIT: Final[float] = 2.0
"""Initial log-odds of an open gate (gates start open, so the penalty closes them)."""
LOG_S_INIT: Final[float] = float(np.log(0.05))
"""Initial posterior standard deviation of the newest knot, 0.05 standardised units."""
MONTH_DAYS: Final[float] = 30.4375
_N_MONTHS: Final[int] = 160
_MONTH_START = np.datetime64("2019-09", "M") + np.arange(_N_MONTHS + 1)
MONTH_FIRST_DAY: Final[np.ndarray] = (
    _MONTH_START.astype("datetime64[D]") - np.datetime64("2019-09-01", "D")
).astype(np.int64)
"""Day index (from 2019-09-01) of the first day of month `i` after 2019-09."""


def day_index(*, tau: torch.Tensor) -> torch.Tensor:
    """Return the day index (from 2019-09-01) of positions `tau` in months."""
    first = torch.as_tensor(MONTH_FIRST_DAY, device=tau.device)
    month = tau.floor().long().clamp(0, _N_MONTHS - 1)
    length = (first[month + 1] - first[month]).to(tau.dtype)
    frac = tau - month.to(tau.dtype)
    return first[month] + torch.floor(frac * length + 1e-3).long()


def _newest_weight_monthly(*, tau: torch.Tensor, n_knots: int) -> torch.Tensor:
    """Return the interpolation weight of the newest of `n_knots` monthly knots."""
    return (tau.clamp(0.0, float(n_knots - 1)) - float(n_knots - 2)).clamp(0.0, 1.0)


class TvMonthlyOffset(var.OutputOffsetDrift):
    """O1: monthly knots, L1 total-variation penalty `lam x sum |first difference| / n_rows`."""

    lam: float = 0.0
    n_rows: float = 1.0

    def configure(self, *, lam: float, n_rows: float, prior_sd: float | None = None) -> None:
        """Set the penalty weight (nats per standardised unit) and the training-row count."""
        self.lam, self.n_rows = float(lam), float(n_rows)
        if prior_sd is not None:
            self.prior_sd = float(prior_sd)

    def penalty(self) -> torch.Tensor:
        """Return `lam x total variation / n_rows` (experiment 573's scaling)."""
        k = self.all_knots()
        return self.lam * (k[:, 1:] - k[:, :-1]).abs().sum() / self.n_rows

    def newest_weight(self, tau: torch.Tensor) -> torch.Tensor:
        """Return the interpolation weight of the newest knot."""
        return _newest_weight_monthly(tau=tau, n_knots=self.n_knots)


class TvDailyOffset(m.ModularMdn):
    """O2: one knot per day, held constant within the day, L1 total-variation penalty."""

    lam: float = 0.0
    n_rows: float = 1.0

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; `n_knots` counts monthly knots."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        self.n_monthly = n_knots
        self.n_days = int(MONTH_FIRST_DAY[n_knots - 1]) + 1
        if stage == "pre":
            self.knots = nn.Parameter(torch.zeros(n_primaries, self.n_days, 1))
        else:
            self.register_buffer("knots_old", torch.zeros(n_primaries, self.n_days - TAIL, 1))
            self.knot_new = nn.Parameter(torch.zeros(n_primaries, TAIL, 1))

    def all_knots(self) -> torch.Tensor:
        """Return every daily knot, (Primaries, days, 1)."""
        if self.stage == "pre":
            return self.knots
        return torch.cat([self.knots_old, self.knot_new], dim=1)  # ty: ignore[invalid-argument-type]

    def configure(self, *, lam: float, n_rows: float, prior_sd: float | None = None) -> None:
        """Set the penalty weight and the training-row count."""
        self.lam, self.n_rows = float(lam), float(n_rows)
        if prior_sd is not None:
            self.prior_sd = float(prior_sd)

    def penalty(self) -> torch.Tensor:
        """Return `lam x total variation / n_rows`."""
        k = self.all_knots()
        return self.lam * (k[:, 1:] - k[:, :-1]).abs().sum() / self.n_rows

    def newest_weight(self, tau: torch.Tensor) -> torch.Tensor:
        """Return 1 for rows that read the newest knot."""
        return (day_index(tau=tau) >= self.n_days - 1).to(tau.dtype)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs with the offset added to the component means."""
        assert primary is not None
        out = super().forward(x, primary)
        idx = day_index(tau=x[:, TAU_COL]).clamp(0, self.n_days - 1)
        offset = self.all_knots()[primary, idx]
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + offset, out[:, 2 * k :]], dim=-1)


class GatedMonthlyOffset(m.ModularMdn):
    """O3: monthly knots as the cumulative sum of hard-concrete-gated jumps."""

    lam: float = 0.0
    n_rows: float = 1.0

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; every gate starts open with amplitude zero."""
        super().__init__(n_primaries=n_primaries, stage=stage)
        self.n_knots = n_knots
        if stage == "pre":
            self.log_alpha = nn.Parameter(torch.full((n_primaries, n_knots), GATE_INIT))
            self.amp = nn.Parameter(torch.zeros(n_primaries, n_knots))
        else:
            self.register_buffer("knots_old", torch.zeros(n_primaries, n_knots - 1))
            self.log_alpha_new = nn.Parameter(torch.full((n_primaries,), GATE_INIT))
            self.amp_new = nn.Parameter(torch.zeros(n_primaries))

    def configure(self, *, lam: float, n_rows: float, prior_sd: float | None = None) -> None:
        """Set the penalty weight (nats per open gate) and the training-row count."""
        self.lam, self.n_rows = float(lam), float(n_rows)
        _ = prior_sd

    @staticmethod
    def gate(log_alpha: torch.Tensor, *, noisy: bool) -> torch.Tensor:
        """Return the hard-concrete gate (noisy while training)."""
        if noisy:
            u = torch.rand_like(log_alpha).clamp(1e-4, 1 - 1e-4)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / HC_BETA)
        else:
            s = torch.sigmoid(log_alpha)
        return (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0.0, 1.0)

    def all_knots(self) -> torch.Tensor:
        """Return every knot, (Primaries, knots, 1); noisy gates while training."""
        noisy = self.training
        if self.stage == "pre":
            return torch.cumsum(self.gate(self.log_alpha, noisy=noisy) * self.amp, dim=1)[..., None]
        last = self.knots_old[:, -1] + self.gate(self.log_alpha_new, noisy=noisy) * self.amp_new  # ty: ignore[not-subscriptable]
        return torch.cat([self.knots_old, last[:, None]], dim=1)[..., None]  # ty: ignore[invalid-argument-type]

    def penalty(self) -> torch.Tensor:
        """Return `lam x expected number of open gates / n_rows`."""
        offset = HC_BETA * float(np.log(-HC_GAMMA / HC_ZETA))
        alpha = self.log_alpha if self.stage == "pre" else self.log_alpha_new
        return self.lam * torch.sigmoid(alpha - offset).sum() / self.n_rows

    def newest_weight(self, tau: torch.Tensor) -> torch.Tensor:
        """Return the interpolation weight of the newest knot."""
        return _newest_weight_monthly(tau=tau, n_knots=self.n_knots)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the main mixture's 9 raw outputs with the offset added to the component means."""
        assert primary is not None
        out = super().forward(x, primary)
        knots = self.all_knots()[..., 0]
        tau = x[:, TAU_COL].clamp(0.0, float(self.n_knots - 1))
        lo = tau.floor().long().clamp(max=self.n_knots - 2)
        frac = tau - lo.to(tau.dtype)
        offset = ((1.0 - frac) * knots[primary, lo] + frac * knots[primary, lo + 1])[:, None]
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + offset, out[:, 2 * k :]], dim=-1)


class VariationalMixin:
    """Gives a host network a Gaussian posterior on its newest knot (stage `ft` only).

    The host supplies `all_knots`, `newest_weight`, `n_rows`, and `knot_new`. During training one
    reparameterised draw per Primary per batch is added to the newest knot. The Kullback-Leibler
    term to a Gaussian centred on the previous knot, divided by `n_rows`, is added to the host's
    penalty. At evaluation the knots are the posterior mean.
    """

    prior_sd: float = 0.1
    mean_only: bool = False

    def add_posterior(self) -> None:
        """Create the posterior's log standard deviation, one per Primary."""
        self.log_s = nn.Parameter(torch.full((self.knot_new.shape[0], 1), LOG_S_INIT))  # ty: ignore[unresolved-attribute]

    def all_knots(self) -> torch.Tensor:
        """Return the knots with the newest knot sampled while training."""
        k = super().all_knots()  # ty: ignore[unresolved-attribute]
        if not self.training or self.mean_only:  # ty: ignore[unresolved-attribute]
            return k
        noise = torch.exp(self.log_s) * torch.randn_like(self.log_s)  # ty: ignore[unresolved-attribute]
        return torch.cat([k[:, :-1], k[:, -1:] + noise[:, None]], dim=1)

    def penalty(self) -> torch.Tensor:
        """Return the host's penalty plus the Kullback-Leibler term divided by `n_rows`."""
        mean_knots = super().all_knots()  # ty: ignore[unresolved-attribute]
        m_new, prev = mean_knots[:, -1], mean_knots[:, -2].detach()
        s = torch.exp(self.log_s)  # ty: ignore[unresolved-attribute]
        kl = (
            torch.log(self.prior_sd / s)
            + (s**2 + (m_new - prev) ** 2) / (2 * self.prior_sd**2)
            - 0.5
        )
        self.mean_only = True
        host = super().penalty()  # ty: ignore[unresolved-attribute]
        self.mean_only = False
        return host + kl.sum() / self.n_rows  # ty: ignore[unresolved-attribute]

    def extra_variance(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Return each row's variance of the offset (standardised units squared)."""
        w = self.newest_weight(x[:, TAU_COL])  # ty: ignore[unresolved-attribute]
        return (w * torch.exp(self.log_s)[p, 0]) ** 2  # ty: ignore[unresolved-attribute]


class OutputOffsetVar(VariationalMixin, var.OutputOffsetDrift):
    """Q2: D5's monthly offset (L2 slope penalty) with a posterior on the newest knot."""

    n_rows: float = 1.0

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; the posterior exists in fine-tuning only."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        if stage == "ft":
            self.add_posterior()

    def configure(self, *, lam: float, n_rows: float, prior_sd: float | None = None) -> None:
        """Set the training-row count and the prior standard deviation (D5's penalty has no weight)."""
        _ = lam
        self.n_rows = float(n_rows)
        if prior_sd is not None:
            self.prior_sd = float(prior_sd)

    def newest_weight(self, tau: torch.Tensor) -> torch.Tensor:
        """Return the interpolation weight of the newest knot."""
        return _newest_weight_monthly(tau=tau, n_knots=self.n_knots)


class TvMonthlyVar(VariationalMixin, TvMonthlyOffset):
    """Q4 with O1."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; the posterior exists in fine-tuning only."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        if stage == "ft":
            self.add_posterior()


class TvDailyVar(VariationalMixin, TvDailyOffset):
    """Q4 with O2."""

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; the posterior exists in fine-tuning only."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        if stage == "ft":
            self.add_posterior()


class ScaleOffsetDrift(var.OutputOffsetDrift):
    """Q5: D5's location offset plus a slow per-Primary offset on the log standard deviation of the components.

    Scale knots follow D5's schedule and penalty: all learnt in pretraining, only the newest in
    fine-tuning, flat extrapolation. `perm` (set after training, for the placebo) makes each Primary
    read another Primary's scale knots.
    """

    perm: torch.Tensor | None = None

    def __init__(self, *, n_primaries: int, stage: str, n_knots: int) -> None:
        """Build the network; scale knots start at zero."""
        super().__init__(n_primaries=n_primaries, stage=stage, n_knots=n_knots)
        if stage == "pre":
            self.s_knots = nn.Parameter(torch.zeros(n_primaries, n_knots, 1))
        else:
            self.register_buffer("s_old", torch.zeros(n_primaries, n_knots - 1, 1))
            self.s_new = nn.Parameter(torch.zeros(n_primaries, 1))

    def configure(self, *, lam: float, n_rows: float, prior_sd: float | None = None) -> None:
        """Nothing to set: the penalty is D5's."""
        _ = (lam, n_rows, prior_sd)

    def all_scale_knots(self) -> torch.Tensor:
        """Return every scale knot, (Primaries, knots, 1)."""
        if self.stage == "pre":
            return self.s_knots
        return torch.cat([self.s_old, self.s_new[:, None]], dim=1)  # ty: ignore[invalid-argument-type]

    def penalty(self) -> torch.Tensor:
        """Return D5's penalty on the location knots plus the same penalty on the scale knots."""
        k = self.all_scale_knots()
        return super().penalty() + var.PENALTY_LAMBDA * ((k[:, 1:] - k[:, :-1]) ** 2).mean()

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs with the scale offset added to the log standard deviations."""
        assert primary is not None
        out = super().forward(x, primary)
        knots = self.all_scale_knots()
        tau = x[:, TAU_COL].clamp(0.0, float(self.n_knots - 1))
        lo = tau.floor().long().clamp(max=self.n_knots - 2)
        frac = (tau - lo.to(tau.dtype))[:, None]
        who = primary if self.perm is None else self.perm[primary]
        scale = (1.0 - frac) * knots[who, lo] + frac * knots[who, lo + 1]
        k = m.COMPONENTS
        return torch.cat([out[:, : 2 * k], out[:, 2 * k :] + scale], dim=-1)


def load_pre_into_ft(*, model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a stage-`pre` state dict into the matching stage-`ft` network; `mixer_ft` stays as built."""
    assert getattr(model, "stage", None) == "ft"
    rest = {a: v for a, v in state.items() if not a.startswith("mixer_ft.")}
    s_knots = rest.pop("s_knots", None)
    if isinstance(model, GatedMonthlyOffset):
        la, amp = rest.pop("log_alpha"), rest.pop("amp")
        extra = {"knots_old", "log_alpha_new", "amp_new"}
    else:
        knots = rest.pop("knots")
        extra = {"knots_old", "knot_new", "log_s", "s_old", "s_new"}
    missing = model.load_state_dict(rest, strict=False)
    assert all(a.startswith("mixer_ft.") or a in extra for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    with torch.no_grad():
        if s_knots is not None:
            model.s_old.copy_(s_knots[:, :-1])  # ty: ignore[call-non-callable]
            model.s_new.copy_(s_knots[:, -1])  # ty: ignore[call-non-callable]
        if isinstance(model, GatedMonthlyOffset):
            gate = model.gate(la, noisy=False)
            cum = torch.cumsum(gate * amp, dim=1)
            model.knots_old.copy_(cum[:, :-1])  # ty: ignore[call-non-callable]
            model.log_alpha_new.copy_(la[:, -1])
            model.amp_new.copy_(amp[:, -1])
        elif isinstance(model, TvDailyOffset):
            model.knots_old.copy_(knots[:, :-TAIL])  # ty: ignore[call-non-callable]
            model.knot_new.copy_(knots[:, -TAIL:])
        else:
            model.knots_old.copy_(knots[..., :-1, :])  # ty: ignore[call-non-callable]
            model.knot_new.copy_(knots[..., -1, :])  # ty: ignore[call-non-callable]

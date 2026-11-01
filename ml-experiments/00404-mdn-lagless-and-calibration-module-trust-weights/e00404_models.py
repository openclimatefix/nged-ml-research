"""Networks of experiment 404: the plain lag network, the trust network, and its controls.

The mixture helpers are copied from experiment 397's `e00397_models.py`, whose branch has not
merged. All networks predict the per-Primary standardised power `z` as a mixture of Gaussians.

Inputs, all float32 tensors:

- `base` (rows, n_base): weather, solar geometry, calendar, holidays, and the lead features.
- `lags` (rows, 9): for each of the 3 lags, the standardised value, an is-missing flag, and the age.
- `ll` (rows, 9): the lagless mixture (3 log weights, 3 means, 3 log standard deviations) in
  standardised units, from the cross-fitted table in training and from the further lagless module
  at test time. It is a constant (stop-gradient) for every network here.
- `trust` (rows, 3, n_trust): the trust inputs at the row's valid time and at the 2 previous
  half-hours (the causal moving average of 3 half-hours of the trust weights averages their
  weights).
- `primary` (rows,) int64 and `metered` (rows, 1) float32 (1 for signed-MW, 0 for MVA-only).
- `lead_band` (rows,) int64: 0 for days 1 to 2, 1 for days 3 to 5, 2 for days 6 to 10 and beyond.

Arms: `plain` (A0, A0-inj, and B2 with `lag_dropout`), `b1`, `b1m`, `b0`, and `b3`.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)
Z_LIMIT: Final[float] = 40.0
BISECTION_STEPS: Final[int] = 40
COMPONENTS: Final[int] = 3
N_LAGS: Final[int] = 3
CNN_CHANNELS: Final[int] = 16
HISTORY_STEPS: Final[int] = 336
"""Half-hours of actual demand (7 days) that arm b7 reads up to the newest usable reading."""
KEPT_PER_LAG: Final[int] = 3
"""Columns kept per lag when the learned windows replace the statistics: single-step surprise,
band score, and neighbour mean."""
LAG_FEATURES: Final[int] = 3
"""Value, is-missing flag, and age of each lag."""
LEAD_BANDS: Final[int] = 3
SMOOTHING_STEPS: Final[int] = 3
ARMS: Final[tuple[str, ...]] = ("plain", "b1", "b1m", "b1g", "b1o", "b6", "b7", "b0", "b3")


def mixture_parts(
    *, out: torch.Tensor, components: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split raw outputs (..., 3 K) into log weights, means, and standard deviations."""
    k = components
    return (
        torch.log_softmax(out[..., :k], dim=-1),
        out[..., k : 2 * k],
        torch.exp(out[..., 2 * k :].clamp(-3.0, 3.0)),
    )


def mixture_nll_parts(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Return the negative log-likelihood of `z` for each row, for any number of components."""
    log_pdf = -0.5 * ((z[..., None] - mean) / std) ** 2 - torch.log(std)
    return -torch.logsumexp(log_w + log_pdf, dim=-1) + 0.5 * float(np.log(2 * np.pi))


def mixture_quantiles(
    *, log_w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Return the 11 quantiles (rows, 11) of mixtures (rows, components), by CDF bisection."""
    levels = torch.tensor(LEVELS, dtype=mean.dtype, device=mean.device)
    w = log_w.exp()
    lo = torch.full((len(w), len(LEVELS)), -Z_LIMIT, dtype=mean.dtype, device=mean.device)
    hi = torch.full_like(lo, Z_LIMIT)
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        z = (mid[..., None] - mean[:, None, :]) / (std[:, None, :] * float(np.sqrt(2.0)))
        cdf = (w[:, None, :] * 0.5 * (1.0 + torch.erf(z))).sum(-1)
        below = cdf < levels
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def init_head(*, head: nn.Sequential, components: int = COMPONENTS) -> None:
    """Zero the output layer's weights and set its bias to spread the components."""
    last = head[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    offsets = np.linspace(-0.5, 0.5, components) if components > 1 else np.zeros(1)
    bias = [0.0] * components + [float(o) for o in offsets] + [0.0] * components
    with torch.no_grad():
        last.bias.copy_(torch.tensor(bias))


def mlp(*, n_in: int, hidden: int, layers: int, n_out: int) -> nn.Sequential:
    """Return `layers` hidden GELU layers of width `hidden`, then a linear output layer."""
    modules: list[nn.Module] = []
    width = n_in
    for _ in range(layers):
        modules += [nn.Linear(width, hidden), nn.GELU()]
        width = hidden
    modules.append(nn.Linear(width, n_out))
    return nn.Sequential(*modules)


def causal_average(*, weights: torch.Tensor) -> torch.Tensor:
    """Average trust weights (rows, 3 steps, 3 lags) over the current and 2 previous half-hours."""
    return weights.mean(dim=1)


def _gap_features(
    ll_lw: torch.Tensor,
    ll_mean: torch.Tensor,
    ll_std: torch.Tensor,
    lag_lw: torch.Tensor,
    lag_mean: torch.Tensor,
    lag_std: torch.Tensor,
) -> torch.Tensor:
    """Return the gap between the lagged and the lagless mixture means, and the gap over the spread.

    The gradient is stopped, so the trust head reads the gap but cannot move either forecast to
    change it.
    """
    del lag_std
    w_ll, w_lag = ll_lw.exp(), lag_lw.exp()
    mean_ll = (w_ll * ll_mean).sum(-1)
    mean_lag = (w_lag * lag_mean).sum(-1)
    spread = ((w_ll * (ll_std**2 + ll_mean**2)).sum(-1) - mean_ll**2).clamp(min=1e-4).sqrt()
    gap = (mean_lag - mean_ll).detach()
    return torch.stack([gap, gap / spread.detach().clamp(min=0.05)], dim=-1)


class LaglessModule(nn.Module):
    """A mixture density network on the base inputs and the Primary embedding, with no power lag."""

    def __init__(self, *, n_base: int, n_primaries: int, hidden: int = 64, layers: int = 2) -> None:
        """Build the network; the output layer starts at zero weights."""
        super().__init__()
        self.embedding = nn.Embedding(n_primaries, 8)
        self.net = mlp(n_in=n_base + 8, hidden=hidden, layers=layers, n_out=3 * COMPONENTS)
        init_head(head=self.net)

    def forward(self, base: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
        """Return the raw lagless mixture outputs (rows, 9)."""
        return self.net(torch.cat([base, self.embedding(primary)], dim=-1))


class PlainLagMdn(nn.Module):
    """The pooled plain network with the 3 lags (A0, A0-inj) and optional token-level lag dropout.

    With `lag_dropout > 0` (arm B2), each lag's three features are zeroed with that probability
    during training and the lag is flagged missing. There is no lagless module.
    """

    def __init__(
        self,
        *,
        n_base: int,
        n_primaries: int,
        hidden: int = 64,
        layers: int = 2,
        lag_dropout: float = 0.0,
        n_extra: int = 0,
    ) -> None:
        """Build the network; the output layer starts at zero weights.

        `n_extra` counts columns after the 9 lag columns (arm S1: the surprise at each lag).
        """
        super().__init__()
        self.lag_dropout = lag_dropout
        self.embedding = nn.Embedding(n_primaries, 8)
        n_in = n_base + N_LAGS * LAG_FEATURES + n_extra + 8
        self.net = mlp(n_in=n_in, hidden=hidden, layers=layers, n_out=3 * COMPONENTS)
        init_head(head=self.net)

    def drop_lags(self, lags: torch.Tensor) -> torch.Tensor:
        """Zero each lag with probability `lag_dropout` and flag it missing (training only)."""
        if not self.training or self.lag_dropout <= 0:
            return lags
        view = lags.reshape(len(lags), N_LAGS, LAG_FEATURES)
        keep = (torch.rand(len(lags), N_LAGS, 1, device=lags.device) >= self.lag_dropout).float()
        missing = view[..., 1:2].clamp(0, 1)
        flagged = torch.cat(
            [view[..., :1] * keep, torch.maximum(missing, 1.0 - keep), view[..., 2:] * keep], -1
        )
        return flagged.reshape(len(lags), -1)

    def forward(
        self, base: torch.Tensor, lags: torch.Tensor, primary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the log weights, means, and standard deviations of the 3-component mixture."""
        x = torch.cat([base, self.drop_lags(lags), self.embedding(primary)], dim=-1)
        return mixture_parts(out=self.net(x), components=COMPONENTS)


class TrustMdn(nn.Module):
    """The calibration module and a trust head (B1, B1m, B0) or an attention gate (B3).

    The calibration module sees the lagless mixture `ll` (constant), the 3 lags, the Primary
    embedding, the metering indicator, and the lead features, and never the trust inputs. The
    final output is a mixture of the lagless and the lagged mixtures (6 components). The weight on
    the lagged mixture is the product of the 3 per-lag trust weights (`b1`), their mean (`b1m`),
    or comes from a learned weight per lead band and lag that ignores the trust inputs (`b0`).
    Arm `b3` replaces the trust head and the product rule with a softmax attention gate over 4
    tokens (the 3 lags and the lagless forecast) feeding a 3-component head.
    """

    def __init__(
        self,
        *,
        arm: str,
        n_base: int,
        n_trust: int,
        n_primaries: int,
        n_lead: int = 2,
        hidden: int = 64,
        layers: int = 2,
        token: int = 16,
        n_extra: int = 0,
    ) -> None:
        """Build the modules; each trust weight starts near 0.5.

        `n_extra` counts extra weekly-lag columns that follow the 9 lag columns (arms B4a-c).
        """
        super().__init__()
        if arm not in ("b1", "b1m", "b1g", "b1o", "b6", "b7", "b0", "b3"):
            msg = f"unknown arm {arm!r}"
            raise ValueError(msg)
        self.arm, self.n_lead = arm, n_lead
        self.embedding = nn.Embedding(n_primaries, 8)
        shared = 3 * COMPONENTS + N_LAGS * LAG_FEATURES + n_extra + 8 + 1 + n_lead
        if arm == "b7":
            shared += CNN_CHANNELS
            self.history_cnn = nn.Sequential(
                nn.Conv1d(2, CNN_CHANNELS, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
            )
        if arm == "b3":
            self.lag_encoders = nn.ModuleList(
                [nn.Linear(LAG_FEATURES + n_trust, token) for _ in range(N_LAGS)]
            )
            self.ll_encoder = nn.Linear(3 * COMPONENTS, token)
            self.gate = nn.Sequential(
                nn.Linear(n_lead + 8 + 1, 16), nn.GELU(), nn.Linear(16, N_LAGS + 1)
            )
            self.head = mlp(n_in=token + n_lead + 8, hidden=hidden, layers=layers, n_out=9)
            init_head(head=self.head)
            return
        self.calibration = mlp(n_in=shared, hidden=hidden, layers=layers, n_out=3 * COMPONENTS)
        init_head(head=self.calibration)
        if arm == "b0":
            self.band_logits = nn.Parameter(0.02 * torch.randn(2, LEAD_BANDS, N_LAGS))
        else:
            n_gap = {"b1g": 2, "b1o": N_LAGS}.get(arm, 0)
            n_trust_in = n_trust
            if arm == "b6":
                self.cnn = nn.Sequential(
                    nn.Conv1d(2, CNN_CHANNELS, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=5, padding=2),
                    nn.GELU(),
                )
                n_trust_in = N_LAGS * (CNN_CHANNELS + KEPT_PER_LAG)
            self.trust_head = mlp(
                n_in=n_trust_in + n_lead + 8 + 1 + n_gap, hidden=32, layers=2, n_out=N_LAGS
            )
            last = self.trust_head[-1]
            assert isinstance(last, nn.Linear)
            # Small random weights break the symmetry between the 3 lags: with zero weights the
            # product rule gives every lag the same gradient, so the 3 weights would stay equal.
            nn.init.normal_(last.weight, std=0.02)
            nn.init.zeros_(last.bias)

    def trust_weights(
        self,
        trust: torch.Tensor,
        lead: torch.Tensor,
        primary: torch.Tensor,
        metered: torch.Tensor,
        lead_band: torch.Tensor,
        gap: torch.Tensor | None = None,
        window: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the smoothed per-lag trust weights (rows, 3) in [0, 1].

        `gap` (rows, 2), used by arm `b1g`, is the lagged minus the lagless mean and the same in
        units of the lagless spread, both with the gradient stopped.
        """
        if self.arm == "b0":
            return torch.sigmoid(self.band_logits[metered[:, 0].long(), lead_band])
        rows, steps, _ = trust.shape
        if self.arm == "b6":
            trust = self._learned_windows(trust=trust, window=window)
        side = torch.cat([lead, self.embedding(primary), metered], dim=-1)
        if gap is not None:
            side = torch.cat([side, gap], dim=-1)
        side = side[:, None, :].expand(rows, steps, side.shape[-1])
        logits = self.trust_head(torch.cat([trust, side], dim=-1).reshape(rows * steps, -1))
        return causal_average(weights=torch.sigmoid(logits.reshape(rows, steps, N_LAGS)))

    def _learned_windows(self, *, trust: torch.Tensor, window: torch.Tensor | None) -> torch.Tensor:
        """Replace each lag's hand-built window statistics by pooled convolution outputs.

        `window` is (rows, steps, 3 lags, 2 channels, 97): the surprise and an observed flag. The
        three columns after the statistics (single-step surprise, band score, neighbour mean) are
        kept for every lag.
        """
        assert window is not None
        rows, steps = trust.shape[:2]
        features = self.cnn(window.reshape(rows * steps * N_LAGS, 2, -1)).mean(-1)
        features = features.reshape(rows, steps, N_LAGS, CNN_CHANNELS)
        kept = trust.reshape(rows, steps, N_LAGS, -1)[..., -KEPT_PER_LAG:]
        return torch.cat([features, kept], dim=-1).reshape(rows, steps, -1)

    def forward(
        self,
        *,
        base: torch.Tensor,
        lags: torch.Tensor,
        ll: torch.Tensor,
        trust: torch.Tensor,
        lead: torch.Tensor,
        primary: torch.Tensor,
        metered: torch.Tensor,
        lead_band: torch.Tensor,
        details: bool = False,
        oracle: torch.Tensor | None = None,
        window: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        """Return the final mixture (log weights, means, stds); with `details`, also the weights.

        `base` is accepted for a uniform call signature; the trust arms do not use it because the
        lagless forecast `ll` already carries the base inputs. `oracle` (rows, 3) holds the true
        event flag at each lag's valid time; only the positive-control arm `b1o` reads it.
        """
        del base
        ll = ll.detach()
        emb = self.embedding(primary)
        if self.arm == "b3":
            return self._attention(
                lags=lags, ll=ll, trust=trust, lead=lead, emb=emb, metered=metered, details=details
            )
        inputs = [ll, lags, emb, metered, lead]
        if self.arm == "b7":
            assert history is not None
            inputs.append(self.history_cnn(history).mean(-1))
        out = self.calibration(torch.cat(inputs, dim=-1))
        lag_lw, lag_mean, lag_std = mixture_parts(out=out, components=COMPONENTS)
        ll_lw, ll_mean, ll_std = mixture_parts(out=ll, components=COMPONENTS)
        gap = None
        if self.arm == "b1o":
            gap = oracle
        if self.arm == "b1g":
            gap = _gap_features(ll_lw, ll_mean, ll_std, lag_lw, lag_mean, lag_std)
        t = self.trust_weights(trust, lead, primary, metered, lead_band, gap, window)
        w = t.mean(dim=-1) if self.arm == "b1m" else t.prod(dim=-1)
        w = w[:, None]
        # The offsets keep both logs finite without a clamp, which would zero the gradient.
        log_w = torch.cat(
            [ll_lw + torch.log1p(-w * (1.0 - 1e-6)), lag_lw + torch.log(w + 1e-6)], dim=-1
        )
        final = (log_w, torch.cat([ll_mean, lag_mean], -1), torch.cat([ll_std, lag_std], -1))
        return (final, t) if details else final

    def _attention(
        self,
        *,
        lags: torch.Tensor,
        ll: torch.Tensor,
        trust: torch.Tensor,
        lead: torch.Tensor,
        emb: torch.Tensor,
        metered: torch.Tensor,
        details: bool,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        """Gate over 4 tokens: the 3 lag tokens and the lagless token."""
        view = lags.reshape(len(lags), N_LAGS, LAG_FEATURES)
        tokens = [
            torch.nn.functional.gelu(enc(torch.cat([view[:, i], trust[:, 0, :]], dim=-1)))
            for i, enc in enumerate(self.lag_encoders)
        ]
        tokens.append(torch.nn.functional.gelu(self.ll_encoder(ll)))
        weights = torch.softmax(self.gate(torch.cat([lead, emb, metered], dim=-1)), dim=-1)
        mixed = (weights[..., None] * torch.stack(tokens, dim=1)).sum(dim=1)
        out = self.head(torch.cat([mixed, lead, emb], dim=-1))
        final = mixture_parts(out=out, components=COMPONENTS)
        return (final, weights) if details else final


def count_parameters(*, model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lead_band_of(*, lead_hours: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the lead band: 0 for days 1 to 2, 1 for days 3 to 5, 2 for later leads."""
    hours = np.asarray(lead_hours, dtype=np.float64)
    return np.digitize(hours, [48.0, 120.0]).astype(np.int64)


def selftest() -> None:
    """Check shapes, the trust weight at initialisation, and that `ll` receives no gradient."""
    torch.manual_seed(0)
    rows, n_base, n_trust, n_primaries = 32, 20, 3 * (KEPT_PER_LAG + 14), 4
    base = torch.randn(rows, n_base)
    lags = torch.randn(rows, 9)
    ll = torch.randn(rows, 9, requires_grad=True)
    trust = torch.randn(rows, SMOOTHING_STEPS, n_trust)
    lead = torch.rand(rows, 2)
    primary = torch.randint(0, n_primaries, (rows,))
    metered = torch.randint(0, 2, (rows, 1)).float()
    band = torch.randint(0, LEAD_BANDS, (rows,))
    z = torch.randn(rows)
    for arm in ("b1", "b1m", "b1g", "b1o", "b6", "b7", "b0", "b3"):
        model = TrustMdn(arm=arm, n_base=n_base, n_trust=n_trust, n_primaries=n_primaries)
        (lw, mean, std), weights = model(
            base=base, lags=lags, ll=ll, trust=trust, lead=lead, primary=primary,
            metered=metered, lead_band=band, details=True,
            oracle=torch.zeros(rows, N_LAGS),
            window=torch.randn(rows, SMOOTHING_STEPS, N_LAGS, 2, 97),
            history=torch.randn(rows, 2, HISTORY_STEPS),
        )  # fmt: skip
        assert lw.shape[0] == rows
        assert mean.shape == std.shape == lw.shape
        loss = mixture_nll_parts(log_w=lw, mean=mean, std=std, z=z).mean()
        loss.backward()
        assert ll.grad is None or float(ll.grad.abs().sum()) == 0.0, "ll must be constant"
        if arm in ("b1", "b1m"):
            assert torch.allclose(weights, torch.full_like(weights, 0.5), atol=0.1), (
                "trust must start near 0.5"
            )
        print(
            arm,
            tuple(lw.shape),
            f"loss {float(loss):.3f}",
            f"params {count_parameters(model=model):,}",
        )
    plain = PlainLagMdn(n_base=n_base, n_primaries=n_primaries, lag_dropout=0.2)
    plain.train()
    lw, mean, std = plain(base, lags, primary)
    q = mixture_quantiles(log_w=lw.double(), mean=mean.double(), std=std.double())
    assert bool((q[:, 1:] >= q[:, :-1]).all()), "quantiles must be sorted"
    lagless = LaglessModule(n_base=n_base, n_primaries=n_primaries)
    assert lagless(base, primary).shape == (rows, 9)
    print("selftest passed")


if __name__ == "__main__":
    selftest()

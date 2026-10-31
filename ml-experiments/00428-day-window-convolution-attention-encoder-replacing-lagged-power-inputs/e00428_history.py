"""The 14-day history arms of experiment 428 (design iteration 2): inputs and networks.

A history window is the 672 half-hours ending at the newest usable reading (the origin minus 3.5
hours), shifted back by 28 days for the other-week control. Each position carries eight channels:
the standardised power, the observed flag, the standardised cross-fitted lagless median (experiment
404's table), the surprise (power minus lagless median over the half band width, clipped at 5), and
the sine and cosine of the time of day and of the day of week. Five arms read it:

- `s1`: R-inj plus the single-step surprise at the three origin lags (no window).
- `hsmooth`: R-inj plus four learned smoothings of the surprise series (aim b).
- `hlvl`: R-inj plus a learned level offset and a soft regime vector (aim a).
- `h14s`: no lags; a dilated convolutional network and one attention head with a null token.
- `h14s_ow`: `h14s` reading the window from 28 days earlier.
"""

import math
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_models as models  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]
from torch.utils.checkpoint import checkpoint  # ty: ignore[unresolved-import]

HISTORY: Final[int] = 672
"""Half-hours in the history window (14 days)."""
OTHER_WEEK_SHIFT: Final[int] = 4 * win.STEPS_PER_WEEK
"""How far the other-week control moves the window back (28 days)."""
CHANNELS: Final[int] = 8
WIDTH: Final[int] = 32
DILATIONS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32)
KERNELS: Final[int] = 4
LEAD_BANDS: Final[int] = 4
LEAD_EDGES_DAYS: Final[tuple[float, ...]] = (2.0, 5.0, 10.0)
CLIP_SD: Final[float] = 5.0
POSITION_DROPOUT: Final[float] = 0.1
ATTENTION_DROPOUT: Final[float] = 0.1
REGIMES: Final[int] = 4
EXTRA_SMOOTH: Final[int] = KERNELS + 1
"""Inputs H-smooth adds: four smoothed surprises and the observed share of the window."""
MASS_CLASSES: Final[int] = 3
"""Attention classes reported besides the null token: newest 6 hours, same time of day, other."""


def history_indices(*, origin: torch.Tensor, shift: int) -> torch.Tensor:
    """Return the grid index of the 672 positions of each row's window: (rows, 672)."""
    newest = origin - win.NEWEST_AGE - shift
    offsets = torch.arange(HISTORY, device=origin.device) - (HISTORY - 1)
    return newest[:, None] + offsets[None, :]


def lead_band(*, origin: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the lead band (0 to 3: under 2, 2 to 5, 5 to 10, and over 10 days) of each row."""
    lead_days = (target - origin).to(torch.float32) / win.STEPS_PER_DAY
    edges = torch.tensor(LEAD_EDGES_DAYS, device=origin.device)
    return torch.bucketize(lead_days, edges, right=True)


def surprise_at(
    *, power: torch.Tensor, median: torch.Tensor, half: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the clipped surprise (0 where either input is unobserved) and the usable mask."""
    usable = torch.isfinite(power) & torch.isfinite(median) & torch.isfinite(half)
    z = ((power - median) / half).clamp(-CLIP_SD, CLIP_SD)
    return torch.where(usable, z, torch.zeros_like(z)), usable


def build_history(
    *, raw: torch.Tensor, index: torch.Tensor, median: torch.Tensor, half: torch.Tensor,
    power_mean: float, power_std: float, origin: torch.Tensor, target: torch.Tensor,
    primary: torch.Tensor, target_std: torch.Tensor,
) -> dict:  # fmt: skip
    """Return the window dict the history networks read.

    Args:
        raw: (rows, 672) raw power over capacity, NaN where unobserved.
        index: (rows, 672) grid indices.
        median: (rows, 672) lagless median over capacity, NaN where the table has no row.
        half: (rows, 672) half width of the lagless 5%-95% band, NaN where the table has no row.
        power_mean: Standardiser mean of the power.
        power_std: Standardiser standard deviation of the power.
        origin: (rows,) grid index of the origin.
        target: (rows,) grid index of the target.
        primary: (rows,) Primary number.
        target_std: (primaries,) standard deviation of each Primary's target.

    Returns:
        The window dict (power, surprise, and mask tensors) the history networks read.
    """
    assert bool((index[:, -1] <= origin - win.NEWEST_AGE).all()), (
        "history reads after the newest reading"
    )
    observed = torch.isfinite(raw) & (index <= (origin - win.NEWEST_AGE)[:, None])
    power = torch.where(
        observed, ((raw - power_mean) / power_std).clamp(-CLIP_SD, CLIP_SD), torch.zeros_like(raw)
    )
    has_median = torch.isfinite(median)
    median_z = torch.where(
        has_median, ((median - power_mean) / power_std).clamp(-CLIP_SD, CLIP_SD),
        torch.zeros_like(median),
    )  # fmt: skip
    surprise, usable = surprise_at(
        power=torch.where(observed, raw, torch.full_like(raw, float("nan"))),
        median=median,
        half=half,
    )
    tod = 2.0 * math.pi * (index % win.STEPS_PER_DAY).to(torch.float32) / win.STEPS_PER_DAY
    dow = 2.0 * math.pi * (index % win.STEPS_PER_WEEK).to(torch.float32) / win.STEPS_PER_WEEK
    chan = torch.stack(
        [power, observed.to(torch.float32), median_z, surprise,
         torch.sin(tod), torch.cos(tod), torch.sin(dow), torch.cos(dow)], dim=1,
    )  # fmt: skip
    level = torch.where(usable, raw - median, torch.zeros_like(raw))
    return {
        "chan": chan,
        "index": index,
        "observed": observed,
        "usable": usable,
        "surprise": surprise,
        "level": (level / target_std[primary][:, None].clamp(min=1e-6)).clamp(-CLIP_SD, CLIP_SD),
        "origin": origin,
        "target": target,
        "band": lead_band(origin=origin, target=target),
    }


class CnnEncoder(nn.Module):
    """A non-causal dilated residual convolutional network; every position keeps its own vector."""

    def __init__(self, *, dropout: float = 0.1) -> None:
        """Build the input projection and the six residual blocks."""
        super().__init__()
        self.project = nn.Conv1d(CHANNELS, WIDTH, kernel_size=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(WIDTH, WIDTH, kernel_size=3, padding=d, dilation=d),
                nn.GroupNorm(1, WIDTH),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for d in DILATIONS
        )

    def forward(self, chan: torch.Tensor) -> torch.Tensor:
        """Return (rows, 672, WIDTH) from the (rows, CHANNELS, 672) channels."""
        h = self.project(chan)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                h = h + checkpoint(block, h, use_reentrant=False)
            else:
                h = h + block(h)
        return h.transpose(1, 2)


def drop_positions(*, window: dict, probability: float) -> dict:
    """Return `window` with a random share of the observed positions hidden (training only)."""
    keep = torch.rand_like(window["chan"][:, 0, :]) >= probability
    observed = window["observed"] & keep
    chan = window["chan"].clone()
    for c in (0, 3):
        chan[:, c, :] = torch.where(observed, chan[:, c, :], torch.zeros_like(chan[:, c, :]))
    chan[:, 1, :] = observed.to(torch.float32)
    out = dict(window)
    out.update(chan=chan, observed=observed, usable=window["usable"] & keep, surprise=chan[:, 3, :])
    return out


def kernel_weights(
    *, window: dict, a: torch.Tensor, c: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Return (rows, KERNELS, 672) softmax weights over the observed positions.

    The score of position j is a cos(time of day of j minus the target's) + c cos(day of week of j
    minus the target's) - b age of j in days, with a, c, b read from the row's lead band. The
    weights never read the surprise values.
    """
    index, target, origin = window["index"], window["target"], window["origin"]
    band = window["band"]
    delta = (index - target[:, None]).to(torch.float32)
    tod = torch.cos(2.0 * math.pi * delta / win.STEPS_PER_DAY)
    dow = torch.cos(2.0 * math.pi * delta / win.STEPS_PER_WEEK)
    age = (origin[:, None] - index).to(torch.float32) / win.STEPS_PER_DAY
    score = (
        a[band][:, :, None] * tod[:, None, :]
        + c[band][:, :, None] * dow[:, None, :]
        - nn.functional.softplus(b[band])[:, :, None] * age[:, None, :]
    )
    score = score.masked_fill(~window["usable"][:, None, :], float("-inf"))
    weights = torch.softmax(score, dim=-1)
    return torch.nan_to_num(weights, nan=0.0)


class HSmoothMdn(nn.Module):
    """R-inj's network plus four learned smoothings of the surprise series (aim b)."""

    takes_windows = True
    attends = False

    def __init__(self, *, n_inputs: int, n_primaries: int, **head: float) -> None:
        """Build the kernels (per lead band) and R's network with the extra inputs."""
        super().__init__()
        self.a = nn.Parameter(torch.randn(LEAD_BANDS, KERNELS) * 0.1)
        self.c = nn.Parameter(torch.randn(LEAD_BANDS, KERNELS) * 0.1)
        self.b = nn.Parameter(torch.full((LEAD_BANDS, KERNELS), -1.0))
        self.core = models.LaggedMdn(
            n_inputs=n_inputs + EXTRA_SMOOTH,
            n_primaries=n_primaries,
            **head,  # ty: ignore[invalid-argument-type]
        )

    def smoothed(self, window: dict) -> torch.Tensor:
        """Return (rows, 4) smoothed surprises and the observed share, as (rows, 5)."""
        weights = kernel_weights(window=window, a=self.a, c=self.c, b=self.b)
        smooth = (weights * window["surprise"][:, None, :]).sum(-1)
        share = window["usable"].to(torch.float32).mean(-1, keepdim=True)
        return torch.cat([smooth, share], dim=1)

    def forward(self, x: torch.Tensor, primary: torch.Tensor, window: dict) -> torch.Tensor:
        """Return the raw mixture outputs."""
        return self.core(torch.cat([x, self.smoothed(window)], dim=1), primary)


class HLevelMdn(nn.Module):
    """R-inj's network plus a learned level offset and a soft regime vector (aim a)."""

    takes_windows = True
    attends = False

    def __init__(self, *, n_inputs: int, n_primaries: int, **head: float) -> None:
        """Build the encoder, the two poolings, the persistence per lead band, and R's network."""
        super().__init__()
        self.components = int(head.get("components", 3))
        self.encoder = CnnEncoder()
        self.embedding = nn.Embedding(n_primaries, 8)
        self.summary_query = nn.Linear(8 + 2, WIDTH)
        self.level_score = nn.Linear(WIDTH, 1)
        self.regime = nn.Linear(WIDTH, REGIMES)
        self.persistence = nn.Parameter(torch.zeros(LEAD_BANDS))
        self.core = models.LaggedMdn(n_inputs=n_inputs + REGIMES, n_primaries=n_primaries, **head)  # ty: ignore[invalid-argument-type]

    def forward(self, x: torch.Tensor, primary: torch.Tensor, window: dict) -> torch.Tensor:
        """Return the raw mixture outputs; the three means shift by g(L) times the level."""
        if self.training:
            window = drop_positions(window=window, probability=POSITION_DROPOUT)
        encoded = self.encoder(window["chan"])
        usable = window["usable"]
        tod = 2.0 * math.pi * (window["origin"] % win.STEPS_PER_DAY).to(torch.float32)
        tod = tod / win.STEPS_PER_DAY
        query = self.summary_query(
            torch.cat(
                [self.embedding(primary), torch.sin(tod)[:, None], torch.cos(tod)[:, None]], 1
            )
        )
        scores = (encoded * query[:, None, :]).sum(-1) * float(WIDTH) ** -0.5
        scores = scores.masked_fill(~window["observed"], float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        summary = (weights[:, :, None] * encoded).sum(1)
        pool = self.level_score(encoded).squeeze(-1).masked_fill(~usable, float("-inf"))
        pool = torch.nan_to_num(torch.softmax(pool, dim=-1), nan=0.0)
        level = (pool * window["level"]).sum(-1)
        regime = torch.softmax(self.regime(summary), dim=-1)
        out = self.core(torch.cat([x, regime], dim=1), primary)
        k = self.components
        gain = torch.sigmoid(self.persistence)[window["band"]]
        shift = torch.zeros_like(out)
        shift[:, k : 2 * k] = (gain * level)[:, None]
        return out + shift


class H14sMdn(nn.Module):
    """Dilated convolutions over the 14-day window, then one attention head with a null token."""

    takes_windows = True
    attends = True

    def __init__(
        self, *, n_inputs: int, n_primaries: int, hidden: int = 64, layers: int = 2,
        components: int = 3, dropout: float = 0.1, position_dropout: float = POSITION_DROPOUT,
    ) -> None:  # fmt: skip
        """Build the encoder, the attention, and R's mixture head."""
        super().__init__()
        self.position_dropout = position_dropout
        self.components = components
        self.encoder = CnnEncoder()
        self.embedding = nn.Embedding(n_primaries, 8)
        query_width = n_inputs + 8
        self.query = nn.Linear(query_width, WIDTH)
        self.key = nn.Linear(WIDTH, WIDTH)
        self.value = nn.Linear(WIDTH, WIDTH)
        self.null_key = nn.Parameter(torch.zeros(WIDTH))
        self.null_value = nn.Parameter(torch.zeros(WIDTH))
        self.tod_gain = nn.Parameter(torch.zeros(1))
        self.dow_gain = nn.Parameter(torch.zeros(1))
        self.age_gain = nn.Parameter(torch.full((1,), -2.0))
        self.attention_dropout = nn.Dropout(ATTENTION_DROPOUT)
        self.head = base.mlp(
            n_in=query_width + WIDTH, hidden=hidden, layers=layers, n_out=3 * components,
            dropout=dropout,
        )  # fmt: skip
        base.init_head(head=self.head, components=components)

    def forward(
        self, x: torch.Tensor, primary: torch.Tensor, window: dict, *, details: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the raw mixture outputs; with `details`, the attention mass by class."""
        if self.training:
            window = drop_positions(window=window, probability=self.position_dropout)
        encoded = self.encoder(window["chan"])
        context_in = torch.cat([x, self.embedding(primary)], dim=-1)
        query = self.query(context_in)
        keys, values = self.key(encoded), self.value(encoded)
        scale = float(WIDTH) ** -0.5
        scores = torch.einsum("rw,rpw->rp", query, keys) * scale
        index, target, origin = window["index"], window["target"], window["origin"]
        delta = (index - target[:, None]).to(torch.float32)
        age = (origin[:, None] - index).to(torch.float32) / win.STEPS_PER_DAY
        scores = (
            scores
            + self.tod_gain * torch.cos(2.0 * math.pi * delta / win.STEPS_PER_DAY)
            + self.dow_gain * torch.cos(2.0 * math.pi * delta / win.STEPS_PER_WEEK)
            - nn.functional.softplus(self.age_gain) * age
        )
        scores = scores.masked_fill(~window["observed"], float("-inf"))
        null_score = (query * self.null_key).sum(-1, keepdim=True) * scale
        weights = torch.softmax(torch.cat([scores, null_score], dim=-1), dim=-1)
        weights = self.attention_dropout(weights)
        context = (weights[:, :-1, None] * values).sum(1) + weights[:, -1:] * self.null_value
        out = self.head(torch.cat([context_in, context], dim=-1))
        if not details:
            return out
        newest = window["index"] > (origin - win.NEWEST_AGE - 12)[:, None]
        phase = torch.remainder(delta, win.STEPS_PER_DAY)
        same_tod = (phase <= 2) | (phase >= win.STEPS_PER_DAY - 2)
        inner = weights[:, :-1]
        recent = (inner * newest).sum(-1)
        match = (inner * (same_tod & ~newest)).sum(-1)
        other = inner.sum(-1) - recent - match
        return out, torch.stack([recent, match, other, weights[:, -1]], dim=1)

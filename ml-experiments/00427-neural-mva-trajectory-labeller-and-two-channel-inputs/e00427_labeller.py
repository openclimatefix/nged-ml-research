# ruff: noqa: E501
"""Step 0 of experiment 427: a whole-day sign labeller for folded MVA readings. No mixture density network is fitted.

The labeller reads one UTC day of 48 half-hourly readings and outputs, per half-hour, the probability
`pi` that the signed power `P'` is negative (reverse flow). It is trained on whole days of the
visible signed-MW Primaries with a solar farm injected at a random level, so the sign is known. The
hidden pair's signed truth and the hidden pair's own farm are never seen in training (see
`train_farms`).

Inputs per half-hour: the reading `r`, `y' = sqrt(max(r^2 - Q0^2, 0))` with `Q0 = 2` points, CAMS
`u` and `k`, ERA5 temperature, a demand reference (the median reading by half-hour over overcast days
of the same 28-day block and day type), the day type, the position in the day, and a mask. The
demand reference of a training copy is made from the uninjected reading with `Q = 2`, and the
reference of a scored Primary is made from the scored reading itself: a farm's output is near zero on
an overcast day, so the two agree to within the floor.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00427-*/e00427_labeller.py pilot
```
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_prep as prep405  # ty: ignore[unresolved-import]
import e00412_fit as f412  # ty: ignore[unresolved-import]
import e00412_inputs as i412  # ty: ignore[unresolved-import]
import e00417_diagnostics as diag  # ty: ignore[unresolved-import]
import e00417_inputs as inputs  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from scipy.stats import rankdata
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from torch import nn

SLUG: Final[str] = "neural-mva-trajectory-labeller-and-two-channel-inputs"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00427-{SLUG}" / "work" / "labeller"
TEST_MONTHS: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-09",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
PILOT_MONTHS: Final[tuple[str, ...]] = ("2025-09", "2026-06")
SIGNED_SIX: Final[tuple[str, ...]] = inputs.SIGNED_SIX
ROTATIONS: Final[dict[str, tuple[str, str]]] = inputs.ROTATIONS
FARM_OF_SITE: Final[dict[str, str]] = inputs.FARM_OF_SITE
TRAIN_FARMS: Final[dict[str, tuple[str, ...]]] = {
    "r1": ("B", "D"),
    "r2": ("A", "C", "D"),
    "r3": ("B", "D"),
}
"""Farms that a rotation's labeller may inject: the farms that the hidden pair does not use."""
GAP_FARMS: Final[tuple[str, ...]] = ("A", "B", "C")
"""A half-hour with a gap in any of these farms is dropped, as in experiment 420's harness."""
US_PER_DAY: Final[int] = i412.US_PER_DAY
US_PER_HALF_HOUR: Final[int] = i412.US_PER_HALF_HOUR
STEPS: Final[int] = 48
Q0_POINTS: Final[float] = 2.0
SCALE_POINTS: Final[float] = 50.0
"""Readings and powers are divided by this before they enter the network."""
GATE_U: Final[float] = 0.3
SUNNY_K: Final[float] = 0.7
OVERCAST_K: Final[float] = 0.4
LEVEL_RANGE: Final[tuple[float, float]] = (0.0, 2.5)
PLACEBO_SHARE: Final[float] = 0.15
"""Share of training copies with no injected farm (the placebo days)."""
Q_CHOICES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0)
CAP_SHARE: Final[float] = 0.3
CAP_FRACTION: Final[float] = 0.7
CUT_SHARE: Final[float] = 0.5
MAX_CUT_HALF_HOURS: Final[int] = 16
"""A cut copy hides the last 1 to 16 half-hours (8 hours) of the day."""
DEAD_BAND_POINTS: Final[float] = 2.0
DEAD_BAND_WEIGHT: Final[float] = 0.25
REFERENCE_HALF_WINDOW_DAYS: Final[int] = 14
N_FEATURES: Final[int] = 12
EPOCHS: Final[int] = 25
PATIENCE: Final[int] = 5
BATCH: Final[int] = 128
LEARNING_RATE: Final[float] = 2e-3
MIN_AVAILABLE_GIB: Final[float] = 12.0
"""The stop line: the `available` column of `free -g`."""
FAR_FUTURE_US: Final[int] = 2**62


# --- one Primary's days -------------------------------------------------------------------------


@dataclass
class PrimaryDays:
    """One Primary's uninjected signed power, weather, and farm outputs on a (day, half-hour) grid.

    Every array has shape `(n_days, 48)`; a missing half-hour is NaN in `truth`. `increment` is the
    size of an injected farm at level 1.0, in points of capacity.
    """

    site: str
    day: np.ndarray
    truth: np.ndarray
    u: np.ndarray
    k: np.ndarray
    temp: np.ndarray
    daytype: np.ndarray
    reference: np.ndarray
    farm: dict[str, np.ndarray]
    gap: np.ndarray
    increment: float
    valid_us: np.ndarray


def _grid(*, start_us: np.ndarray, values: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Scatter per-half-hour values onto a `(len(days), 48)` grid (NaN where absent)."""
    out = np.full((len(days), STEPS), np.nan)
    day = start_us // US_PER_DAY
    row = np.searchsorted(days, day)
    col = ((start_us - day * US_PER_DAY) // US_PER_HALF_HOUR).astype(np.int64)
    out[row, col] = values
    return out


def day_type(*, dow: np.ndarray, bank_holiday: np.ndarray) -> np.ndarray:
    """Return 0 for a weekday, 1 for a Saturday, and 2 for a Sunday or bank holiday."""
    out = np.zeros(len(dow), dtype=np.int64)
    out[dow == 6] = 1
    out[(dow == 7) | bank_holiday] = 2
    return out


def demand_reference(
    *, reading: np.ndarray, k: np.ndarray, daytype: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Return, per (day, half-hour), the median reading over overcast days of the same type nearby.

    Neighbours are the other days within 14 days either side, of the same day type, whose mean `k`
    over their daylight half-hours is under 0.4. A cell with no neighbour takes the Primary's median
    for that half-hour and day type over all overcast days.
    """
    n = len(reading)
    daylight = np.nan_to_num(k) > 0.0
    k_day = np.where(daylight & mask, k, np.nan)
    with np.errstate(all="ignore"):
        overcast = np.nanmean(k_day, axis=1) < OVERCAST_K
    r = np.where(mask & np.isfinite(reading), reading, np.nan)
    out = np.full((n, STEPS), np.nan)
    overall = np.full((3, STEPS), np.nan)
    for t in range(3):
        sel = overcast & (daytype == t)
        if sel.any():
            with np.errstate(all="ignore"):
                overall[t] = np.nanmedian(r[sel], axis=0)
    for i in range(n):
        lo, hi = max(0, i - REFERENCE_HALF_WINDOW_DAYS), min(n, i + REFERENCE_HALF_WINDOW_DAYS + 1)
        near = np.arange(lo, hi)
        near = near[(near != i) & overcast[near] & (daytype[near] == daytype[i])]
        if len(near):
            with np.errstate(all="ignore"):
                out[i] = np.nanmedian(r[near], axis=0)
    fill = overall[daytype]
    out = np.where(np.isfinite(out), out, fill)
    return np.where(np.isfinite(out), out, np.nanmedian(r) if np.isfinite(r).any() else 0.0)


def load_primary(*, site: str) -> PrimaryDays:
    """Return one Primary's uninjected days (experiment 417's harness at level 0 and `Q = 0`)."""
    t = diag.load_table(
        site=site, setting=inputs.Setting(0.0, 0.0), start_us=0, end_us=FAR_FUTURE_US
    )
    valid = t.start_us + US_PER_HALF_HOUR
    days = np.unique(t.start_us // US_PER_DAY)
    truth = _grid(start_us=t.start_us, values=t.truth, days=days)
    daytype_rows = day_type(dow=t.dow, bank_holiday=t.bank_holiday)
    first = np.searchsorted(t.start_us // US_PER_DAY, days)
    daytype = daytype_rows[first]
    u, k = (
        _grid(start_us=t.start_us, values=t.u, days=days),
        _grid(start_us=t.start_us, values=t.k, days=days),
    )
    temp = _grid(start_us=t.start_us, values=t.temp, days=days)
    mask = np.isfinite(truth)
    reading0 = np.sqrt(truth**2 + Q0_POINTS**2)
    reference = demand_reference(reading=reading0, k=k, daytype=daytype, mask=mask)
    grid_valid = (
        days[:, None] * US_PER_DAY + (np.arange(STEPS)[None, :] + 1) * US_PER_HALF_HOUR
    ).astype(np.int64)
    farm: dict[str, np.ndarray] = {}
    gap = np.zeros(grid_valid.shape, dtype=bool)
    for name in ("A", "B", "C", "D"):
        series = prep405.farm_norm_series(farm=name)
        values = prep405.farm_at(times_us=grid_valid.ravel(), series=series, placebo=False)
        farm[name] = values.reshape(grid_valid.shape)
        if name in GAP_FARMS:
            gap |= np.isnan(farm[name])
    increment = (
        i412.level_scale(site=site, y_signed=t.truth / 100.0, valid_us=valid, level=1.0) * 100.0
    )
    return PrimaryDays(
        site=site,
        day=days,
        truth=truth,
        u=u,
        k=k,
        temp=temp,
        daytype=daytype,
        reference=reference,
        farm=farm,
        gap=gap,
        increment=float(increment),
        valid_us=grid_valid,
    )


# --- features -----------------------------------------------------------------------------------


def features(
    *,
    reading: np.ndarray,
    u: np.ndarray,
    k: np.ndarray,
    temp: np.ndarray,
    reference: np.ndarray,
    daytype: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return the `(n, 48, N_FEATURES)` input; missing half-hours are zero with mask 0."""
    n = len(reading)
    y_dash = np.sqrt(np.maximum(reading**2 - Q0_POINTS**2, 0.0))
    onehot = np.eye(3)[daytype][:, None, :].repeat(STEPS, axis=1)
    position = np.broadcast_to(np.arange(STEPS)[None, :] / STEPS, (n, STEPS))
    cols = [
        reading / SCALE_POINTS,
        y_dash / SCALE_POINTS,
        u,
        k / 1.5,
        (temp - 10.0) / 6.0,
        reference / SCALE_POINTS,
        (reading - reference) / SCALE_POINTS,
    ]
    stack = np.stack(cols, axis=-1)
    stack = np.nan_to_num(stack, nan=0.0)
    stack = np.concatenate(
        [stack, onehot, position[..., None], mask[..., None].astype(np.float64)], axis=-1
    )
    stack[~mask] = 0.0
    return stack.astype(np.float32)


# --- the network --------------------------------------------------------------------------------


class Labeller(nn.Module):
    """A width-5 convolution feeding a recurrent layer or a dilated convolution stack."""

    def __init__(self, *, arch: str, hidden: int = 32) -> None:
        """Build the layers for `arch` (`gru` or `conv`) with `hidden` channels."""
        super().__init__()
        self.arch = arch
        self.conv = nn.Conv1d(N_FEATURES, hidden, kernel_size=5, padding=2)
        if arch == "gru":
            self.rnn = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
            width = 2 * hidden
        else:
            self.stack = nn.ModuleList(
                [
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=d, dilation=d)
                    for d in (1, 2, 4, 8)
                ]
            )
            width = hidden
        self.head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the logit of `P' < 0` and the estimate of `P' / 50`, both `(batch, 48)`."""
        h = torch.relu(self.conv(x.transpose(1, 2)))
        if self.arch == "gru":
            h, _ = self.rnn(h.transpose(1, 2))
        else:
            for layer in self.stack:
                h = h + torch.relu(layer(h))
            h = h.transpose(1, 2)
        out = self.head(h)
        return out[..., 0], out[..., 1]


# --- training copies ----------------------------------------------------------------------------


@dataclass
class Copies:
    """A batch of training copies: features, signed truth, and the loss mask."""

    x: np.ndarray
    p_prime: np.ndarray
    mask: np.ndarray


def make_copies(
    *,
    primaries: list[PrimaryDays],
    farms: tuple[str, ...],
    rng: np.random.Generator,
    rows: np.ndarray | None = None,
) -> Copies:
    """Return one injected copy of every training day (or of the `rows` of the stacked days)."""
    stack = {
        name: np.concatenate([getattr(p, name) for p in primaries])
        for name in ("truth", "u", "k", "temp", "daytype", "reference", "gap")
    }
    farm = {f: np.concatenate([p.farm[f] for p in primaries]) for f in farms}
    increment = np.concatenate([np.full(len(p.day), p.increment) for p in primaries])
    peak = {f: float(np.nanmax(prep405.farm_norm_series(farm=f)[1])) for f in farms}
    n = len(increment)
    if rows is None:
        rows = np.arange(n)
    m = len(rows)
    which = rng.integers(0, len(farms), size=m)
    level = rng.uniform(*LEVEL_RANGE, size=m)
    level[rng.random(m) < PLACEBO_SHARE] = 0.0
    q = rng.choice(np.array(Q_CHOICES), size=m)
    capped = rng.random(m) < CAP_SHARE
    injected = np.zeros((m, STEPS))
    farm_ok = np.ones((m, STEPS), dtype=bool)
    for j, f in enumerate(farms):
        sel = which == j
        out = farm[f][rows[sel]]
        out = np.where(capped[sel, None], np.minimum(out, CAP_FRACTION * peak[f]), out)
        injected[sel] = out
        farm_ok[sel] = np.isfinite(farm[f][rows[sel]])
    truth = stack["truth"][rows]
    p_prime = truth - level[:, None] * increment[rows, None] * np.nan_to_num(injected)
    reading = np.sqrt(p_prime**2 + q[:, None] ** 2)
    mask = np.isfinite(truth) & ~stack["gap"][rows] & farm_ok
    cut = rng.random(m) < CUT_SHARE
    hide = rng.integers(1, MAX_CUT_HALF_HOURS + 1, size=m)
    position = np.arange(STEPS)[None, :]
    mask &= ~(cut[:, None] & (position >= STEPS - hide[:, None]))
    x = features(
        reading=reading,
        u=stack["u"][rows],
        k=stack["k"][rows],
        temp=stack["temp"][rows],
        reference=stack["reference"][rows],
        daytype=stack["daytype"][rows],
        mask=mask,
    )
    return Copies(x=x, p_prime=np.nan_to_num(p_prime), mask=mask)


def loss_of(
    *, model: Labeller, copies: Copies, device: torch.device, index: np.ndarray
) -> torch.Tensor:
    """Return the weighted cross-entropy on the sign plus the Huber loss on `P'`."""
    x = torch.from_numpy(copies.x[index]).to(device)
    p = torch.from_numpy(copies.p_prime[index]).to(device, dtype=torch.float32)
    mask = torch.from_numpy(copies.mask[index]).to(device, dtype=torch.float32)
    logit, estimate = model(x)
    weight = torch.where(p.abs() < DEAD_BAND_POINTS, DEAD_BAND_WEIGHT, 1.0) * mask
    sign = nn.functional.binary_cross_entropy_with_logits(
        logit, (p < 0).to(torch.float32), weight=weight, reduction="sum"
    )
    huber = nn.functional.huber_loss(estimate, p / SCALE_POINTS, reduction="none", delta=0.2)
    return (sign + (huber * mask).sum()) / mask.sum().clamp(min=1.0)


@dataclass
class Trained:
    """A fitted labeller and how long it took."""

    model: Labeller
    seconds: float
    epochs: int
    best_validation: float


def held_out_rows(*, primaries: list[PrimaryDays]) -> np.ndarray:
    """Return a boolean per stacked day: every eighth block of seven days since the epoch is held out for early stopping."""
    day = np.concatenate([p.day for p in primaries])
    return (day // 7) % 8 == 0


def train_labeller(
    *,
    primaries: list[PrimaryDays],
    farms: tuple[str, ...],
    arch: str,
    seed: int,
    device: torch.device,
) -> Trained:
    """Fit a labeller on injected copies of the Primaries' days, with early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held = held_out_rows(primaries=primaries)
    train_rows, val_rows = np.flatnonzero(~held), np.flatnonzero(held)
    val = make_copies(
        primaries=primaries, farms=farms, rng=np.random.default_rng(seed + 10_000), rows=val_rows
    )
    model = Labeller(arch=arch).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best, best_state, since, done = float("inf"), None, 0, 0
    start = time.time()
    for epoch in range(EPOCHS):
        copies = make_copies(primaries=primaries, farms=farms, rng=rng, rows=train_rows)
        order = rng.permutation(len(train_rows))
        model.train()
        for i in range(0, len(order), BATCH):
            optimiser.zero_grad()
            loss_of(
                model=model, copies=copies, device=device, index=order[i : i + BATCH]
            ).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            total = sum(
                float(
                    loss_of(
                        model=model,
                        copies=val,
                        device=device,
                        index=np.arange(i, min(i + 512, len(val_rows))),
                    )
                )
                * min(512, len(val_rows) - i)
                for i in range(0, len(val_rows), 512)
            ) / len(val_rows)
        done = epoch + 1
        if total < best - 1e-4:
            best, since = total, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return Trained(model=model, seconds=time.time() - start, epochs=done, best_validation=best)


# --- scoring ------------------------------------------------------------------------------------


def window_us(*, month: str) -> tuple[int, int]:
    """Return the labeller's whole-day training window `[start, end]` for a test month's refit."""
    refit = f412.refit_time(month=month)
    end = i412._us(t=refit - regimes.REFIT_CUTOFF_AGE)
    start = i412._us(t=refit - timedelta(days=base.WINDOW_DAYS))
    start = -(-start // US_PER_DAY) * US_PER_DAY
    end = end - end % US_PER_DAY
    return start, end


def test_window_us(*, month: str) -> tuple[int, int]:
    """Return `[start, end)` of a test month in microseconds (valid times of its rows)."""
    year, mon = (int(x) for x in month.split("-"))
    following = f"{year + (mon == 12)}-{mon % 12 + 1:02d}"
    return i412._us(t=f412.refit_time(month=month)), i412._us(t=f412.refit_time(month=following))


def predict_table(*, model: Labeller, t: diag.Table, device: torch.device) -> pl.DataFrame:
    """Return `pi`, the signed truth, `u`, and `k` for every half-hour of a scored Primary table."""
    valid = t.start_us + US_PER_HALF_HOUR
    days = np.unique(t.start_us // US_PER_DAY)
    reading = _grid(start_us=t.start_us, values=t.r, days=days)
    mask = np.isfinite(reading)
    u = _grid(start_us=t.start_us, values=t.u, days=days)
    k = _grid(start_us=t.start_us, values=t.k, days=days)
    temp = _grid(start_us=t.start_us, values=t.temp, days=days)
    dtype_rows = day_type(dow=t.dow, bank_holiday=t.bank_holiday)
    daytype = dtype_rows[np.searchsorted(t.start_us // US_PER_DAY, days)]
    reference = demand_reference(reading=reading, k=k, daytype=daytype, mask=mask)
    x = features(
        reading=reading, u=u, k=k, temp=temp, reference=reference, daytype=daytype, mask=mask
    )
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            logit, _ = model(torch.from_numpy(x[i : i + 256]).to(device))
            out.append(torch.sigmoid(logit).cpu().numpy())
    pi = np.concatenate(out)
    pos = np.searchsorted(days, t.start_us // US_PER_DAY) * STEPS + (
        (t.start_us % US_PER_DAY) // US_PER_HALF_HOUR
    ).astype(np.int64)
    return pl.DataFrame(
        {
            "site": [t.site] * len(valid),
            "valid": valid,
            "pi": pi.ravel()[pos].astype(np.float32),
            "truth_points": t.truth.astype(np.float32),
            "u_gate": t.u.astype(np.float32),
            "k": t.k.astype(np.float32),
        }
    )


def score_primary(
    *, model: Labeller, site: str, level: float, month: str, window: str, device: torch.device
) -> pl.DataFrame:
    """Score a Primary at an injection level (its own farm, `Q = 2`) in the `train` or `test` window.

    The `train` window is the labeller's whole training days. The `test` window is the test month:
    the table spans 15 days either side so that the retrospective demand reference has neighbours,
    and the rows are cut back to the month afterwards. The reference is a laboratory choice for
    this diagnostic; the network stage uses trailing days only.
    """
    if window == "train":
        start, end = window_us(month=month)
        keep_lo, keep_hi = start, end + 1
    else:
        keep_lo, keep_hi = test_window_us(month=month)
        start, end = keep_lo - 15 * US_PER_DAY, keep_hi + 15 * US_PER_DAY
    t = diag.load_table(
        site=site, setting=inputs.Setting(level, 2.0), start_us=start, end_us=end + 1
    )
    frame = predict_table(model=model, t=t, device=device)
    frame = frame.filter((pl.col("valid") > keep_lo) & (pl.col("valid") <= keep_hi))
    return frame.with_columns(level=pl.lit(level), window=pl.lit(window))


# --- statistics ---------------------------------------------------------------------------------


def auc(*, positive: np.ndarray, negative: np.ndarray) -> float:
    """Return the area under the receiver operating characteristic curve (ties count half)."""
    if not len(positive) or not len(negative):
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    return float(
        (ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def slice_metrics(*, frame: pl.DataFrame) -> dict:
    """Return the area and the recall at 5% false reverse on the sunny slice of injected rows."""
    s = frame.filter((pl.col("u_gate") >= GATE_U) & (pl.col("k") >= SUNNY_K))
    if not len(s):
        return {}
    truth = s["truth_points"].to_numpy() < 0.0
    p = s["pi"].to_numpy()
    threshold = float(np.percentile(p[~truth], 95.0)) if (~truth).any() else float("nan")
    return {
        "rows": len(s),
        "rows_reverse": int(truth.sum()),
        "auc": auc(positive=p[truth], negative=p[~truth]),
        "recall_at_5pc_false_reverse": float((p[truth] > threshold).mean())
        if truth.any()
        else float("nan"),
        "recall_at_0p5": float((p[truth] > 0.5).mean()) if truth.any() else float("nan"),
        "false_at_0p5": float((p[~truth] > 0.5).mean()) if (~truth).any() else float("nan"),
    }


def placebo_rate(*, frame: pl.DataFrame) -> dict:
    """Return the share of sunny-midday half-hours with `pi > 0.5` when nothing is injected."""
    frame = frame.filter((pl.col("level") == 0.0) & (pl.col("truth_points") >= 0.0))
    valid = frame["valid"].to_numpy()
    mid = inputs.is_midday(valid_us=valid)
    sel = mid & (frame["u_gate"].to_numpy() >= GATE_U) & (frame["k"].to_numpy() >= SUNNY_K)
    p = frame["pi"].to_numpy()[sel]
    return {
        "n": int(sel.sum()),
        "share_above_half": float((p > 0.5).mean()) if len(p) else float("nan"),
    }


def day_level_auc(*, injected: pl.DataFrame, placebo: pl.DataFrame) -> dict:
    """Return the area for injected against placebo days: the score of a day is its largest `pi`.

    Days are the clear-ish days with a sunny half-hour (`u >= 0.3`, `k >= 0.7`) in the table, and an
    injected day is positive when it holds at least one sunny half-hour with `P' < 0`.
    """

    def per_day(f: pl.DataFrame) -> pl.DataFrame:
        s = f.filter((pl.col("u_gate") >= GATE_U) & (pl.col("k") >= SUNNY_K))
        return s.group_by("site", (pl.col("valid") // US_PER_DAY).alias("day")).agg(
            score=pl.col("pi").max(), reverse=(pl.col("truth_points") < 0).any()
        )

    a, b = per_day(injected), per_day(placebo)
    pos = a.filter(pl.col("reverse"))["score"].to_numpy()
    neg = b.filter(~pl.col("reverse"))["score"].to_numpy()
    return {
        "auc": auc(positive=pos, negative=neg),
        "n_positive_days": len(pos),
        "n_negative_days": len(neg),
    }


# --- the runs -----------------------------------------------------------------------------------


def available_gib() -> float:
    """Return the `available` column of `free -g`."""
    out = subprocess.run(["free", "-g"], capture_output=True, text=True, check=True).stdout
    return float(out.splitlines()[1].split()[-1])


def atomic_write(*, path: Path, data: bytes) -> None:
    """Write bytes through a temporary file and a rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


def write_frame(*, frame: pl.DataFrame, path: Path) -> None:
    """Write a parquet file atomically."""
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(path)


@dataclass(frozen=True)
class Job:
    """One labeller: a rotation, a test month, the left-out visible Primary, and the architecture."""

    rotation: str
    month: str
    left_out: str
    """A visible Primary left out of training, or `none` for the labeller trained on all four."""
    arch: str
    seed: int = 0

    @property
    def stem(self) -> str:
        """Return the file stem of the job's outputs."""
        return f"{self.rotation}_{self.month}_{self.left_out}_{self.arch}_s{self.seed}"


def visible(*, rotation: str) -> tuple[str, ...]:
    """Return the four visible signed-MW Primaries of a rotation."""
    return tuple(s for s in SIGNED_SIX if s not in ROTATIONS[rotation])


def window_days(*, p: PrimaryDays, month: str) -> PrimaryDays:
    """Return the Primary restricted to the days that lie wholly inside a month's training window."""
    start, end = window_us(month=month)
    keep = (p.valid_us[:, 0] - US_PER_HALF_HOUR >= start) & (p.valid_us[:, -1] <= end)
    truth, k, daytype = p.truth[keep], p.k[keep], p.daytype[keep]
    reference = demand_reference(
        reading=np.sqrt(truth**2 + Q0_POINTS**2), k=k, daytype=daytype, mask=np.isfinite(truth)
    )
    return PrimaryDays(
        site=p.site,
        day=p.day[keep],
        truth=truth,
        u=p.u[keep],
        k=k,
        temp=p.temp[keep],
        daytype=daytype,
        reference=reference,
        farm={f: v[keep] for f, v in p.farm.items()},
        gap=p.gap[keep],
        increment=p.increment,
        valid_us=p.valid_us[keep],
    )


def run_job(
    *, job: Job, cache: dict[str, PrimaryDays], device: torch.device, out_dir: Path
) -> dict:
    """Train one labeller and write its scored tables; skip it if the outputs exist."""
    done = out_dir / f"{job.stem}.json"
    if done.exists():
        return json.loads(done.read_text())
    if available_gib() < MIN_AVAILABLE_GIB:
        msg = f"stop line: under {MIN_AVAILABLE_GIB} GiB available before {job.stem}"
        raise SystemExit(msg)
    train_sites = [s for s in visible(rotation=job.rotation) if s != job.left_out]
    primaries = [window_days(p=cache[s], month=job.month) for s in train_sites]
    farms = tuple(
        f
        for f in TRAIN_FARMS[job.rotation]
        if job.left_out == "none" or f != FARM_OF_SITE[job.left_out]
    )
    trained = train_labeller(
        primaries=primaries, farms=farms, arch=job.arch, seed=job.seed, device=device
    )
    if job.left_out == "none":
        targets = list(ROTATIONS[job.rotation])
        levels = (2.0, 1.0, 0.0)
    else:
        targets = [job.left_out]
        levels = (1.0, 0.0)
    frames = [
        score_primary(
            model=trained.model, site=s, level=lv, month=job.month, window=w, device=device
        )
        for s in targets
        for lv in levels
        for w in (("train", "test") if job.left_out == "none" else ("train",))
    ]
    frame = pl.concat(frames)
    write_frame(frame=frame, path=out_dir / f"{job.stem}.parquet")
    torch.save(trained.model.state_dict(), out_dir / f"{job.stem}.pt")
    scored = {
        "job": job.stem,
        "seconds": trained.seconds,
        "epochs": trained.epochs,
        "best_validation": trained.best_validation,
        "train_sites": train_sites,
        "n_train_days": int(sum(len(p.day) for p in primaries)),
    }
    atomic_write(path=done, data=json.dumps(scored, indent=1).encode())
    return scored


def pilot_jobs() -> list[Job]:
    """Return the architecture pilot: leave-one-Primary-out labellers of rotation r1, two months."""
    return [
        Job("r1", m, s, arch)
        for m in PILOT_MONTHS
        for s in visible(rotation="r1")
        for arch in ("gru", "conv")
    ]


def full_jobs(*, arch: str) -> list[Job]:
    """Return the 105 labellers: per rotation and test month, one on all four and four leave-one-out."""
    return [
        Job(rot, m, left, arch)
        for rot in ROTATIONS
        for m in TEST_MONTHS
        for left in ("none", *visible(rotation=rot))
    ]


def load_cache() -> dict[str, PrimaryDays]:
    """Load the six signed-MW Primaries' days."""
    return {s: load_primary(site=s) for s in SIGNED_SIX}


def pilot_report(*, out_dir: Path) -> dict:
    """Return each architecture's pooled leave-one-out area at level 1.0 and its mean fit time."""
    report: dict = {}
    for arch in ("gru", "conv"):
        frames, secs = [], []
        for job in pilot_jobs():
            if job.arch != arch:
                continue
            f = out_dir / f"{job.stem}.parquet"
            if f.exists():
                frames.append(pl.read_parquet(f).filter(pl.col("level") == 1.0))
                secs.append(json.loads((out_dir / f"{job.stem}.json").read_text())["seconds"])
        if frames:
            report[arch] = {
                **slice_metrics(frame=pl.concat(frames)),
                "mean_fit_seconds": float(np.mean(secs)),
                "n_labellers": len(frames),
            }
    return report


A_FITS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00420-neural-mva-export-gated-signed-final" / "work" / "fits" / "A"
)


def ece(*, p: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    """Return the expected calibration error of probabilities `p` of the boolean `truth`."""
    edges = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = edges == b
        if sel.any():
            total += sel.mean() * abs(float(p[sel].mean()) - float(truth[sel].mean()))
    return total


def with_arm_a(*, frame: pl.DataFrame, rotation: str, month: str, level: float) -> pl.DataFrame:
    """Join experiment 420's arm A (seed 0) probability to the labeller's rows on (site, valid).

    Arm A's file holds one row per run, so a valid time appears about 15 times; the join keeps
    every such row, which makes the labeller and arm A score the identical rows.
    """
    tag = inputs.Setting(level, 2.0).tag
    a = pl.read_parquet(A_FITS / f"{tag}_{rotation}_{month}_s0.parquet").select(
        "site", "valid", "p_negative"
    )
    lab = frame.filter((pl.col("level") == level) & (pl.col("window") == "test"))
    return a.join(lab, on=["site", "valid"], how="inner", validate="m:1")


def metrics_with_ece(*, frame: pl.DataFrame, column: str) -> dict:
    """Return area, recall at 5% false reverse, and calibration error on the sunny slice."""
    s = frame.filter((pl.col("u_gate") >= GATE_U) & (pl.col("k") >= SUNNY_K))
    truth = s["truth_points"].to_numpy() < 0.0
    p = s[column].to_numpy().astype(np.float64)
    threshold = float(np.percentile(p[~truth], 95.0)) if (~truth).any() else float("nan")
    return {
        "rows": len(s),
        "rows_reverse": int(truth.sum()),
        "auc": auc(positive=p[truth], negative=p[~truth]),
        "recall_at_5pc_false_reverse": float((p[truth] > threshold).mean())
        if truth.any()
        else float("nan"),
        "ece": ece(p=p, truth=truth),
    }


def gate_report(*, out_dir: Path, arch: str) -> dict:
    """Return the step 0 gate: L1 and L2 pooled and as a Primary-month macro-average, beside arm A."""
    pooled: dict[str, list[pl.DataFrame]] = {"l1": [], "l2_placebo": []}
    per_block: list[dict] = []
    day_pos, day_neg = [], []
    for job in full_jobs(arch=arch):
        if job.left_out != "none":
            continue
        frame = pl.read_parquet(out_dir / f"{job.stem}.parquet")
        l1 = with_arm_a(frame=frame, rotation=job.rotation, month=job.month, level=1.0)
        pooled["l1"].append(l1)
        for site in ROTATIONS[job.rotation]:
            block = l1.filter(pl.col("site") == site)
            per_block.append(
                {
                    "site": site,
                    "month": job.month,
                    "labeller": metrics_with_ece(frame=block, column="pi"),
                    "arm_a": metrics_with_ece(frame=block, column="p_negative"),
                }
            )
        pooled["l2_placebo"].append(
            with_arm_a(frame=frame, rotation=job.rotation, month=job.month, level=0.0)
        )
        day_pos.append(frame.filter((pl.col("level") == 1.0) & (pl.col("window") == "test")))
        day_neg.append(frame.filter((pl.col("level") == 0.0) & (pl.col("window") == "test")))
    l1_all = pl.concat(pooled["l1"])
    report = {
        "l1_pooled": {
            "labeller": metrics_with_ece(frame=l1_all, column="pi"),
            "arm_a": metrics_with_ece(frame=l1_all, column="p_negative"),
        },
        "l1_macro": {
            who: {
                key: float(np.nanmean([b[who][key] for b in per_block]))
                for key in ("auc", "recall_at_5pc_false_reverse", "ece")
            }
            for who in ("labeller", "arm_a")
        },
        "day_level_l1": day_level_auc(injected=pl.concat(day_pos), placebo=pl.concat(day_neg)),
        "l2": {},
    }
    placebo = pl.concat(pooled["l2_placebo"]).filter(pl.col("truth_points") >= 0.0)
    for site in ("D5", "D9", "D11", "D14", "D16"):
        s = placebo.filter(pl.col("site") == site)
        mid = inputs.is_midday(valid_us=s["valid"].to_numpy())
        sel = mid & (s["u_gate"].to_numpy() >= GATE_U) & (s["k"].to_numpy() >= SUNNY_K)
        report["l2"][site] = {
            "n_rows": int(sel.sum()),
            "labeller_share_above_half": float((s["pi"].to_numpy()[sel] > 0.5).mean())
            if sel.any()
            else None,
            "arm_a_share_above_half": float((s["p_negative"].to_numpy()[sel] > 0.5).mean())
            if sel.any()
            else None,
        }
    return report


def main() -> None:
    """Run the pilot or the 105 labellers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "pilot-report", "run", "gate-report"))
    parser.add_argument("--arch", choices=("gru", "conv"), default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = WORK_DIR / ("pilot" if args.stage.startswith("pilot") else f"full_{args.arch}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "gate-report":
        print(json.dumps(gate_report(out_dir=out_dir, arch=args.arch), indent=1))
        return
    if args.stage == "pilot-report":
        print(json.dumps(pilot_report(out_dir=out_dir), indent=1))
        return
    if args.stage == "run" and args.arch is None:
        msg = "pass --arch (the pilot's choice)"
        raise SystemExit(msg)
    cache = load_cache()
    jobs = pilot_jobs() if args.stage == "pilot" else full_jobs(arch=args.arch)
    for i, job in enumerate(jobs):
        r = run_job(job=job, cache=cache, device=device, out_dir=out_dir)
        print(
            f"[{i + 1}/{len(jobs)}] {job.stem}: {r['seconds']:.0f}s, {r['epochs']} epochs",
            flush=True,
        )


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Shared pieces of experiment 510: rows by cutoff, the lag-free network, its loop, and forecasts.

The lag-free network is experiment 397's pooled `PlainMdn` (setting `g09`) on the 37 non-lag columns
of the `summary` scheme, which is the network of experiment 448's `L0PT` arm. Rows are experiment
428's lag-free rows (`e00428_data.build_fold`): the 24 months before the month's refit, the month's
standardiser and target statistics, experiment 422's early-stopping weeks. Weekly refits extend the
month's rows to the last refit and mask them by cutoff, so a month's rows are built once.

A refit at `cutoff` trains on rows with valid time at most `cutoff - 3.5 h`, origin at most `cutoff`,
and valid time within 730 days before the cutoff, weighted by `0.5 ** (age / half_life)` with age
measured from the cutoff. `leak_report` prints the check before any score.
"""

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00510-lagless-demand-network-weekly-finetune-and-recency-weights"
    / "work"
)
E448_PRETRAINED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00448-s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool/work/pretrained/L0PT"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
REFIT_DAYS: Final[tuple[int, ...]] = (0, 7, 14, 21, 28)
"""Weekly refit offsets in days from the first day of the month; week 1 is the monthly refit."""
STRESS_WEEK: Final[int] = 2
"""Zero-based refit week of the stress run (week 3: the injected 7 days end at day 14)."""
HALF_LIFE_425: Final[float] = common.SHORT_HALF_LIFE_DAYS
HALF_LIFE_90: Final[float] = 90.0
FINE_TUNE_LR: Final[float] = 2e-4
HIDDEN, LAYERS, COMPONENTS, DROPOUT = 64, 2, 3, 0.1
PRETRAIN_LR: Final[float] = 5e-4
WEIGHT_DECAY: Final[float] = 1e-3
BATCH: Final[int] = 8_192
MAX_EPOCHS: Final[int] = 60
PATIENCE: Final[int] = 8
ROW_FRACTION: Final[float] = 0.2
VALIDATION_ROWS: Final[int] = 200_000
GRAD_CLIP: Final[float] = 1.0
WINDOW_DAYS: Final[int] = 730
DAY_US: Final[int] = 86_400 * 1_000_000
CUTOFF_AGE_US: Final[int] = int(regimes.REFIT_CUTOFF_AGE.total_seconds() * 1_000_000)
STEPS_PER_WEEK: Final[int] = 336
EVENT_PAD_STEPS: Final[int] = 96
"""Two days of half-hours: the pad of L-Ww-D."""
BETA_A, BETA_B = 3.0, 97.0
OUTLIER_MIN_SCALE: Final[float] = 3.0
SHIFT_SDS: Final[float] = 3.0


def month_start(*, month: str) -> datetime:
    """Return 00:00 UTC on the first day of a month."""
    return prep.refit_of(month=month)


def month_stop(*, month: str) -> datetime:
    """Return 00:00 UTC on the first day of the next month."""
    return (month_start(month=month) + timedelta(days=32)).replace(day=1)


def cutoffs(*, month: str) -> list[datetime]:
    """Return the month's weekly refit times (day 1, 8, 15, 22, 29), all inside the month."""
    start = month_start(month=month)
    return [start + timedelta(days=d) for d in REFIT_DAYS]


def us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds."""
    return int(t.timestamp() * 1_000_000)


def sites() -> list[str]:
    """Return the Primary site names in Primary-index order."""
    return data.sites()


def clean_stack() -> np.ndarray:
    """Return the cleaned power over capacity (primaries, grid), NaN where flagged or cut."""
    clean = prep422.cleaned_arrays()
    return np.stack([clean[s] for s in sites()]).astype(np.float32)


@dataclass
class MonthRows:
    """All lag-free rows of a month's weekly refits, standardised with the month's refit statistics."""

    month: str
    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    lead: np.ndarray
    valid_us: np.ndarray
    origin_us: np.ndarray
    target_idx: np.ndarray
    is_val: np.ndarray
    base: data.Rows
    n_base: int


def _extension(*, refit: datetime, last: datetime, clean: np.ndarray, rows: data.Rows) -> dict:
    """Return the rows with valid time after the month's refit cutoff and up to the last cutoff."""
    keep_cols, _, _ = data.non_lag_layout()
    low = us(t=refit - regimes.REFIT_CUTOFF_AGE)
    high = us(t=last - regimes.REFIT_CUTOFF_AGE)
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (z["valid"] > low) & (z["valid"] <= high) & (z["origin"] <= us(t=last))
        if not keep.any():
            continue
        target_idx = ((z["valid"][keep] - prep422.GRID_START_US) // prep422.STEP_US).astype(
            np.int64
        )
        y = clean[row["primary"]][target_idx]
        usable = np.isfinite(y)
        parts.append(
            {
                "x": z["x"][keep][usable][:, keep_cols],
                "y": y[usable],
                "origin": z["origin"][keep][usable],
                "valid": z["valid"][keep][usable],
                "target_idx": target_idx[usable],
                "lead": z["lead"][keep][usable],
                "primary": np.full(int(usable.sum()), row["primary"], dtype=np.int64),
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    origin_day = ((cat["origin"] - us(t=prep.FIRST_RUN)) // DAY_US).astype(np.int64)
    held, purged = prep422.purge_masks(origin_day=origin_day)
    kept = np.nonzero(held | ~purged)[0]
    cat = {k: v[kept] for k, v in cat.items()}
    cat["is_val"] = held[kept]
    cat["x"] = rows.standardiser.apply(x=cat["x"]).astype(np.float32)
    return cat


def build_month(*, month: str, clean: np.ndarray) -> MonthRows:
    """Return a month's rows: experiment 428's fold at the monthly refit plus the later weeks' rows."""
    refit = month_start(month=month)
    rows = data.build_fold(refit=refit, clean=clean)
    ext = _extension(refit=refit, last=cutoffs(month=month)[-1], clean=clean, rows=rows)
    grid_us = prep422.GRID_START_US
    step = prep422.STEP_US
    base_valid = grid_us + rows.target_idx.astype(np.int64) * step
    base_origin = grid_us + rows.origin_idx.astype(np.int64) * step
    return MonthRows(
        month=month,
        x=np.concatenate([rows.x, ext["x"]]),
        y=np.concatenate([rows.y, ext["y"].astype(np.float32)]),
        primary=np.concatenate([rows.primary, ext["primary"]]),
        lead=np.concatenate([rows.lead, ext["lead"]]),
        valid_us=np.concatenate([base_valid, ext["valid"]]),
        origin_us=np.concatenate([base_origin, ext["origin"]]),
        target_idx=np.concatenate([rows.target_idx.astype(np.int64), ext["target_idx"]]),
        is_val=np.concatenate([rows.is_val, ext["is_val"]]),
        base=rows,
        n_base=len(rows.x),
    )


def labelled_events() -> dict[int, list[tuple[int, int]]]:
    """Return each Primary's labelled events as (onset, end) grid indices (hindsight labels)."""
    table = pl.read_parquet(worlds402.cf.EVENTS)
    names = sites()
    out: dict[int, list[tuple[int, int]]] = {}
    for r in table.iter_rows(named=True):
        out.setdefault(names.index(r["label"]), []).append((int(r["onset_idx"]), int(r["end_idx"])))
    return out


def event_drop_mask(
    *, rows: MonthRows, cutoff: datetime, events: dict[int, list[tuple[int, int]]]
) -> np.ndarray:
    """Return True for rows inside a labelled event known at `cutoff`, padded two days before and after.

    An event is known if it started at or before the cutoff. The drop runs from two days before its
    onset to the earlier of two days after its end and the cutoff, so an event that ends after the
    cutoff is dropped from its start to the cutoff and its future end is not used.
    """
    cut_idx = (us(t=cutoff) - prep422.GRID_START_US) // prep422.STEP_US
    drop = np.zeros(len(rows.y), dtype=bool)
    for p, spans in events.items():
        sel = rows.primary == p
        t = rows.target_idx[sel]
        hit = np.zeros(len(t), dtype=bool)
        for onset, end in spans:
            if onset > cut_idx:
                continue
            hit |= (t >= onset - EVENT_PAD_STEPS) & (t <= min(end + EVENT_PAD_STEPS, cut_idx))
        drop[sel] = hit
    return drop


def shift_sizes(*, clean: np.ndarray, cutoff: datetime) -> np.ndarray:
    """Return 3 x the standard deviation of each Primary's non-overlapping 7-day mean target (capacity fraction).

    Uses only cleaned data in the 730 days before the cutoff. A week counts if at least 80% of its
    half-hours are observed.
    """
    cut_idx = int((us(t=cutoff) - prep422.GRID_START_US) // prep422.STEP_US)
    out = np.zeros(len(clean))
    for p in range(len(clean)):
        series = clean[p][max(0, cut_idx - WINDOW_DAYS * 48) : cut_idx]
        n = len(series) // STEPS_PER_WEEK
        weeks = series[len(series) - n * STEPS_PER_WEEK :].reshape(n, STEPS_PER_WEEK)
        ok = np.isfinite(weeks).mean(axis=1) >= 0.8
        means = np.nanmean(weeks[ok], axis=1)
        out[p] = SHIFT_SDS * float(np.std(means)) if len(means) > 10 else 0.0
    return out


@dataclass
class Refit:
    """The selected rows of one refit, as arrays ready for the GPU."""

    index: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    is_val: np.ndarray
    report: dict


def select_rows(
    *, rows: MonthRows, cutoff: datetime, half_life_days: float, drop: np.ndarray | None = None,
    shift: np.ndarray | None = None, sign: int = 0, clean: np.ndarray | None = None,
) -> Refit:  # fmt: skip
    """Return the rows of a refit at `cutoff`, their (optionally shifted) targets, and recency weights.

    Args:
        rows: The month's rows.
        cutoff: The refit time.
        half_life_days: Half-life of the recency weight, with age measured from the cutoff.
        drop: Rows to leave out (L-Ww-D), in both the training and the early-stopping set.
        shift: Per-Primary shift size for the stress run, or None.
        sign: +1 or -1, the sign of the injected shift.
        clean: Unused unless `shift` is given.

    Returns:
        The refit's rows, targets, weights, early-stopping flags, and a report of the leak check.
    """
    c = us(t=cutoff)
    keep = (
        (rows.valid_us <= c - CUTOFF_AGE_US)
        & (rows.origin_us <= c)
        & (rows.valid_us > c - WINDOW_DAYS * DAY_US)
    )
    n_before_drop = int(keep.sum())
    if drop is not None:
        keep &= ~drop
    idx = np.nonzero(keep)[0]
    y = rows.y[idx].astype(np.float32).copy()
    injected = 0
    if shift is not None:
        lo = c - 7 * DAY_US
        inj = (rows.valid_us[idx] >= lo) & ~rows.is_val[idx]
        y[inj] += sign * shift[rows.primary[idx][inj]].astype(np.float32)
        injected = int(inj.sum())
    age_days = (c - rows.valid_us[idx]) / DAY_US
    weight = (0.5 ** (age_days / half_life_days)).astype(np.float32)
    report = {
        "cutoff": cutoff.isoformat(),
        "rows": len(idx),
        "val_rows": int(rows.is_val[idx].sum()),
        "latest_valid": datetime.fromtimestamp(rows.valid_us[idx].max() / 1e6, UTC).isoformat(),
        "latest_origin": datetime.fromtimestamp(rows.origin_us[idx].max() / 1e6, UTC).isoformat(),
        "rows_past_valid_cut": int((rows.valid_us[idx] > c - CUTOFF_AGE_US).sum()),
        "rows_past_origin_cut": int((rows.origin_us[idx] > c).sum()),
        "dropped_rows": n_before_drop - len(idx),
        "injected_rows": injected,
    }
    return Refit(index=idx, y=y, weight=weight, is_val=rows.is_val[idx], report=report)


class OutlierMdn(base.PlainMdn):
    """`PlainMdn` with two more outputs: the outlier logit and the offset of its scale multiple."""

    def __init__(self, *, n_inputs: int, n_primaries: int) -> None:
        """Build the network; the extra outputs start at an outlier weight of 3% and a multiple of 3 + softplus(0)."""
        super().__init__(
            n_inputs=n_inputs, hidden=HIDDEN, layers=LAYERS, components=COMPONENTS,
            dropout=DROPOUT, n_primaries=n_primaries,
        )  # fmt: skip
        last = self.net[-1]
        assert isinstance(last, torch.nn.Linear)
        head = torch.nn.Linear(last.in_features, 3 * COMPONENTS + 2)
        torch.nn.init.zeros_(head.weight)
        with torch.no_grad():
            head.bias.copy_(
                torch.tensor(
                    [*base.mixture_head_bias(components=COMPONENTS), math.log(0.03 / 0.97), 0.0]
                )
            )
        self.net[-1] = head


def new_model(*, outlier: bool, n_inputs: int, n_primaries: int) -> torch.nn.Module:
    """Return an untrained lag-free network (3 components, or 3 plus the outlier component)."""
    if outlier:
        return OutlierMdn(n_inputs=n_inputs, n_primaries=n_primaries)
    return base.PlainMdn(
        n_inputs=n_inputs, hidden=HIDDEN, layers=LAYERS, components=COMPONENTS,
        dropout=DROPOUT, n_primaries=n_primaries,
    )  # fmt: skip


def load_pretrained(*, model: torch.nn.Module, path: Path) -> None:
    """Load a pretrained state dict, stripping the `net.` wrapper prefix of experiment 428's files."""
    state = torch.load(path, map_location="cpu")
    if any(k.startswith("net.net.") or k == "net.embedding.weight" for k in state):
        state = {k[4:]: v for k, v in state.items()}
    model.load_state_dict(state)


def outlier_parts(*, out: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Split the 11 outputs into main log weights, means, stds, the outlier weight, mean, and std."""
    k = COMPONENTS
    log_w, mean, std = base.mixture_parts(out=out[..., : 3 * k], components=k)
    p_out = torch.sigmoid(out[..., 3 * k])
    mu_out = (log_w.exp() * mean).sum(-1)
    sd_out = std.max(dim=-1).values * (
        OUTLIER_MIN_SCALE + torch.nn.functional.softplus(out[..., 3 * k + 1])
    )
    return log_w, mean, std, p_out, mu_out, sd_out


def outlier_nll(*, out: torch.Tensor, z: torch.Tensor, prior: bool) -> torch.Tensor:
    """Return the four-component negative log-likelihood per row (plus the Beta prior if `prior`)."""
    log_w, mean, std, p_out, mu_out, sd_out = outlier_parts(out=out)
    p_out = p_out.clamp(1e-6, 1 - 1e-6)
    log_main = torch.logsumexp(
        log_w - 0.5 * ((z[..., None] - mean) / std) ** 2 - torch.log(std), dim=-1
    ) + torch.log1p(-p_out)
    log_out = -0.5 * ((z - mu_out) / sd_out) ** 2 - torch.log(sd_out) + torch.log(p_out)
    nll = -torch.logaddexp(log_main, log_out) + 0.5 * float(np.log(2 * np.pi))
    if prior:
        nll = nll - (BETA_A - 1.0) * torch.log(p_out) - (BETA_B - 1.0) * torch.log1p(-p_out)
    return nll


def row_nll(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    p: torch.Tensor,
    z: torch.Tensor,
    outlier: bool,
    prior: bool = False,
    main_only: bool = False,
) -> torch.Tensor:
    """Return the negative log-likelihood of each row (`main_only` scores the served 3 components)."""
    out = model(x, p)
    if outlier and main_only:
        return base.mixture_nll(out=out[..., : 3 * COMPONENTS], z=z, components=COMPONENTS)
    if outlier:
        return outlier_nll(out=out, z=z, prior=prior)
    return base.mixture_nll(out=out, z=z, components=COMPONENTS)


def train(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, p: torch.Tensor,
    is_val: torch.Tensor, seed: int, lr: float, outlier: bool, prior: bool, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Train with weighted NLL and early stopping (experiment 428's loop for a lag-free arm).

    The Beta prior of the outlier weight enters the training loss only (once per row, a per-row
    regulariser); the early-stopping validation scores the served three main components alone. Returns the epoch count and the best validation NLL.
    """
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[:VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(ROW_FRACTION * len(order))]
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            nll = row_nll(model=model, x=x[b], p=p[b], z=z[b], outlier=outlier, prior=prior)
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        row_nll(
                            model=model, x=x[c], p=p[c], z=z[c], outlier=outlier, main_only=True
                        )
                        for c in val_idx.split(BATCH * 2)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def serving_quantiles(
    *, model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor, outlier: bool
) -> torch.Tensor:
    """Return the 11 standardised quantiles of the three main components (the outlier is dropped)."""
    with torch.no_grad():
        out = model(x, p)
        if outlier:
            out = out[..., : 3 * COMPONENTS]
        log_w, mean, std = base.mixture_parts(out=out.double(), components=COMPONENTS)
        return base.mixture_quantiles(log_w=log_w, mean=mean, std=std)


def forecast_frame(
    *, model: torch.nn.Module, outlier: bool, test: data.TestRows, select: np.ndarray, rows: data.Rows,
    widen: float, device: str,
) -> pl.DataFrame:  # fmt: skip
    """Return raw (or widened) quantiles, as a fraction of capacity, for the selected test rows.

    Quantiles are the three main components' mixture quantiles, sorted; `widen` scales the distance
    from the median (1.0 stores the raw quantiles).
    """
    table = prep.primaries().select("primary", "time_series_id")
    parts = []
    idx = np.nonzero(select)[0]
    for a in range(0, len(idx), 65_536):
        sl = idx[a : a + 65_536]
        x = torch.from_numpy(test.x[sl]).to(device)
        p = torch.from_numpy(test.primary[sl]).to(device)
        q = serving_quantiles(model=model, x=x, p=p, outlier=outlier)
        mu = torch.from_numpy(rows.target_mean).to(device)[p][:, None]
        sd = torch.from_numpy(rows.target_std).to(device)[p][:, None]
        q = torch.sort(q * sd + mu, dim=1).values
        if widen != 1.0:
            med = q[:, 5:6]
            q = med + widen * (q - med)
        parts.append(q.float().cpu().numpy())
    q = np.concatenate(parts)
    frame = pl.DataFrame(
        {
            "primary": test.primary[idx],
            "power_fcst_init_time": test.origin[idx],
            "valid_time": test.valid[idx],
        }
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )
    frame = frame.join(table, on="primary", how="left", maintain_order="left").drop("primary")
    levels = pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
    return frame.select(
        pl.col("time_series_id").cast(pl.Int32), "power_fcst_init_time", "valid_time"
    ).hstack(levels)


def write_json(*, obj: object, path: Path) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)

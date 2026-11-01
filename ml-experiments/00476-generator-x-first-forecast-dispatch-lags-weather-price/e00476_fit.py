# ruff: noqa: E501
"""Fit the arms of experiment 476 for Generator X and write their quantile forecasts.

One series, so the network is experiment 397's plain mixture density network without a series
embedding. Rows are the (forecast run, valid time) pairs of experiment 397's ensemble summaries at
the Primary nearest to the site, so every arm trains and is scored on the same rows. Training is
experiment 397's loop (weighted mixture negative log-likelihood, AdamW, early stopping on held-out
weeks, setting g09). Arms:

- `A0`: calendar and the three origin lags (values, missing flags, ages).
- `A1`: `A0` plus the ensemble mean and standard deviation of the seven weather variables.
- `A2`: `A0` plus the market index price at the three lag times (values and missing flags).
- `A3`: calendar and weather, no lags.

The target is output over the series' maximum absolute output, which is read once from the cleaned
record before the hold-out (a scale only). No clip at zero and no folded likelihood.

```bash
source env; PYTHONPATH=ml-experiments/00087-*:ml-experiments/00397-*:ml-experiments/00476-* \
  OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00476-*/e00476_fit.py --arms A0 --months 2026-01 --seeds 0
```
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

DATA: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"])
OUT: Final[Path] = (
    DATA / "ml-experiments" / "00476-generator-x-first-forecast-dispatch-lags-weather-price"
)
LABEL_MAP: Final[Path] = (
    DATA / "ml-experiments" / "00473-three-generators-data-look" / "label_map_PRIVATE.json"
)
PRICE_PATH: Final[Path] = (
    DATA / "ml-experiments/00010-elexon-price-lags-stage1a-demand/prices/market_index.parquet"
)
MODELS_DIR: Final[Path] = OUT / "work" / "models"
PRED_DIR: Final[Path] = OUT / "work" / "predictions"
CACHE: Final[Path] = OUT / "work" / "cache.npz"
WEATHER_CACHE: Final[Path] = OUT / "work" / "weather.parquet"
SCREEN_MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[tuple[str, ...]] = ("A0", "A1", "A2", "A3")
MAX_LEAD_HOURS: Final[float] = 240.0
BATCH: Final[int] = 8_192
MAX_EPOCHS: Final[int] = 200
PATIENCE: Final[int] = 6
GRAD_CLIP: Final[float] = 1.0
CLIP_SD: Final[float] = 5.0
SETTING: Final[fit397.Setting] = fit397.setting_by_name(name="g09")
LEVELS: Final[tuple[float, ...]] = base.LEVELS
WEATHER_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{k}_{w}" for k in ("mean", "std") for w in prep.WEATHER
)
STEPS_PER_DAY: Final[int] = 48
STEPS_PER_WEEK: Final[int] = 336
AGE_UNIT: Final[float] = 672.0
"""Half-hours in 336 hours, the unit of the lag ages."""


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


def site_series_id() -> int:
    """Return Generator X's time series id (the mapping is private)."""
    kind = json.loads(LABEL_MAP.read_text())["X"]
    meta = pl.read_parquet(DATA / "NGED" / "metadata.parquet").filter(
        pl.col("time_series_type") == kind
    )
    assert meta.height == 1
    return int(meta["time_series_id"][0])


def weather_series_id() -> int:
    """Return the time series id of the Primary nearest to the site (its ensemble summary is used)."""
    kind = json.loads(LABEL_MAP.read_text())["X"]
    meta = pl.read_parquet(DATA / "NGED" / "metadata.parquet")
    site = meta.filter(pl.col("time_series_type") == kind)
    lat, lon = site["latitude"][0], site["longitude"][0]
    near = (
        prep.primaries()
        .with_columns(
            d=(
                ((pl.col("latitude") - lat) * 111.0) ** 2
                + ((pl.col("longitude") - lon) * 111.0 * math.cos(math.radians(lat))) ** 2
            ).sqrt()
        )
        .sort("d")
    )
    assert float(near["d"][0]) < 1.0, "the nearest Primary is not within 1 km of the site"
    return int(near["time_series_id"][0])


def grid_arrays() -> tuple[np.ndarray, np.ndarray, float]:
    """Return the site's output over its scale and the market price on the dense grid, and the scale.

    Output and price are NaN where unobserved. Output comes from the cleaned table with flagged
    rows nulled (none are flagged for Generator X, so none is dropped); exact zeros are kept.
    """
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        z = np.load(CACHE)
        return z["power"], z["price"], float(z["scale"])
    tsid = site_series_id()
    raw = (
        pl.scan_delta(str(DATA / "NGED" / "cleaned_power_time_series.delta"))
        .filter(pl.col("time_series_id") == tsid)
        .select("time", "power", "drop_reason")
        .collect()
    )
    assert raw["drop_reason"].null_count() == raw.height, "a half-hour of the site is flagged"
    raw = raw.filter(pl.col("time") < common.HOLDOUT).drop_nulls("power")
    size = len(common.grid_times())
    power = np.full(size, np.nan, dtype=np.float64)
    idx = np.asarray(common.grid_index(times=raw["time"]))
    power[idx] = raw["power"].to_numpy()
    scale = float(np.nanmax(np.abs(power)))
    price = np.full(size, np.nan, dtype=np.float64)
    prices = (
        pl.read_parquet(PRICE_PATH)
        .filter((pl.col("dataProvider") == "APXMIDP") & (pl.col("volume") > 0))
        .sort("time", "price")
        .unique(subset="time", keep="first", maintain_order=True)
    )
    pidx = np.asarray(common.grid_index(times=prices["time"]))
    ok = (pidx >= 0) & (pidx < size)
    price[pidx[ok]] = prices["price"].to_numpy()[ok]
    np.savez(
        CACHE, power=(power / scale).astype(np.float32), price=price.astype(np.float32), scale=scale
    )
    z = np.load(CACHE)
    return z["power"], z["price"], float(z["scale"])


def weather_table() -> pl.DataFrame:
    """Return the nearest Primary's ensemble summaries over all runs, cached."""
    if WEATHER_CACHE.exists():
        return pl.read_parquet(WEATHER_CACHE)
    tsid = weather_series_id()
    table = (
        pl.scan_parquet(prep.SUMMARY_DIR / "init=*.parquet")
        .filter(pl.col("time_series_id") == tsid)
        .select("power_fcst_init_time", "valid_time", *WEATHER_COLUMNS)
        .collect()
        .sort("power_fcst_init_time", "valid_time")
    )
    table.write_parquet(WEATHER_CACHE)
    return table


def lag_indices(*, origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the grid index of the newest, daily, and weekly lag (rows, 3), as `common.origin_lags`."""
    newest = origin - common.NEWEST_AGE
    gap = target - newest
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(np.int64)
    week = target - STEPS_PER_WEEK * np.ceil(gap / STEPS_PER_WEEK).astype(np.int64)
    assert (day <= newest).all(), "a daily lag is younger than the newest reading"
    assert (week <= newest).all(), "a weekly lag is younger than the newest reading"
    return np.stack([newest, day, week], axis=1)


@dataclass
class Frame:
    """The rows of a set of forecast runs: keys, target, and the unstandardised input blocks."""

    origin: np.ndarray
    valid: np.ndarray
    lead: np.ndarray
    y: np.ndarray
    calendar: np.ndarray
    lags: np.ndarray
    ages: np.ndarray
    price: np.ndarray
    weather: np.ndarray


def build_frame(*, power: np.ndarray, price: np.ndarray, weather: pl.DataFrame) -> Frame:
    """Return every row of the summaries with lead 0 to 240 hours, valid before the hold-out."""
    t = weather.with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds() / 3600.0
    ).filter((pl.col("lead") > 0) & (pl.col("lead") <= MAX_LEAD_HOURS))
    t = t.filter(pl.col("valid_time") < common.HOLDOUT)
    origin = np.asarray(common.grid_index(times=t["power_fcst_init_time"]))
    target = np.asarray(common.grid_index(times=t["valid_time"]))
    local = t["valid_time"].dt.convert_time_zone("Europe/London")
    minute = (local.dt.hour() * 60 + local.dt.minute()).to_numpy() / 1440.0
    weekday = local.dt.weekday().to_numpy() / 7.0
    lead = t["lead"].to_numpy()
    calendar = np.stack(
        [
            np.sin(2 * np.pi * minute), np.cos(2 * np.pi * minute),
            np.sin(2 * np.pi * weekday), np.cos(2 * np.pi * weekday),
            (local.dt.weekday().to_numpy() >= 6).astype(np.float64), lead / 336.0,
        ],
        axis=1,
    )  # fmt: skip
    idx = lag_indices(origin=origin, target=target)
    lags = np.stack([common.take(values=power, index=idx[:, k]) for k in range(3)], axis=1)
    prices = np.stack([common.take(values=price, index=idx[:, k]) for k in range(3)], axis=1)
    ages = (target[:, None] - idx) / AGE_UNIT
    return Frame(
        origin=t["power_fcst_init_time"].dt.epoch("us").to_numpy(),
        valid=t["valid_time"].dt.epoch("us").to_numpy(),
        lead=lead,
        y=common.take(values=power, index=target),
        calendar=calendar,
        lags=lags,
        ages=ages,
        price=prices,
        weather=t.select(WEATHER_COLUMNS).to_numpy().astype(np.float64),
    )


def _standardise(*, a: np.ndarray, mean: np.ndarray | float, std: np.ndarray | float) -> np.ndarray:
    """Return `a` standardised and clipped, NaN to 0."""
    z = np.clip((a - mean) / std, -CLIP_SD, CLIP_SD)
    return np.where(np.isfinite(z), z, 0.0)


@dataclass
class Prep:
    """Training-window statistics that standardise the inputs."""

    power_mean: float
    power_std: float
    price_mean: float
    price_std: float
    weather_mean: np.ndarray
    weather_std: np.ndarray
    target_mean: float
    target_std: float


def fit_prep(*, frame: Frame, train: np.ndarray) -> Prep:
    """Return the statistics of the training rows."""
    y = frame.y[train]
    price = frame.price[train]
    return Prep(
        power_mean=float(y.mean()),
        power_std=float(max(y.std(), 1e-3)),
        price_mean=float(np.nanmean(price)),
        price_std=float(max(np.nanstd(price), 1e-3)),
        weather_mean=frame.weather[train].mean(axis=0),
        weather_std=np.maximum(frame.weather[train].std(axis=0), 1e-6),
        target_mean=float(y.mean()),
        target_std=float(max(y.std(), 1e-3)),
    )


def matrix(*, frame: Frame, rows: np.ndarray, arm: str, prep_: Prep) -> np.ndarray:
    """Return the standardised input matrix of an arm for `rows`."""
    blocks = [frame.calendar[rows]]
    if arm != "A3":
        lags = frame.lags[rows]
        blocks += [
            _standardise(a=lags, mean=prep_.power_mean, std=prep_.power_std),
            (~np.isfinite(lags)).astype(np.float64),
            frame.ages[rows],
        ]
    if arm in ("A1", "A3"):
        blocks.append(
            _standardise(a=frame.weather[rows], mean=prep_.weather_mean, std=prep_.weather_std)
        )
    if arm == "A2":
        price = frame.price[rows]
        blocks += [
            _standardise(a=price, mean=prep_.price_mean, std=prep_.price_std),
            (~np.isfinite(price)).astype(np.float64),
        ]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def fold_masks(*, frame: Frame, refit: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the train, early-stopping, and test row masks of the fold refitted at `refit`."""
    cutoff = _us(t=refit - timedelta(hours=3.5))
    window = _us(t=refit - timedelta(days=prep.WINDOW_DAYS))
    in_window = (
        (frame.valid <= cutoff)
        & (frame.origin <= _us(t=refit))
        & (frame.valid > window)
        & np.isfinite(frame.y)
    )
    origin_day = ((frame.origin - _us(t=prep.FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
    held, purged = prep.held_out_weeks(origin_day=origin_day)
    end = (refit + timedelta(days=32)).replace(day=1)
    test = (frame.origin >= _us(t=refit)) & (frame.origin < _us(t=end))
    return in_window & ~held & ~purged, in_window & held, test


def train_network(
    *, model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, w: torch.Tensor,
    is_val: torch.Tensor, seed: int,
) -> tuple[int, float]:  # fmt: skip
    """Train with weighted negative log-likelihood; early stopping on `is_val`; return epochs, loss."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=SETTING.lr, weight_decay=SETTING.weight_decay
    )
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    k = SETTING.components
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=x.device)]
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            nll = base.mixture_nll(out=model(x[b]), z=z[b], components=k)
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
                        base.mixture_nll(out=model(x[c]), z=z[c], components=k)
                        for c in val_idx.split(BATCH * 4)
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


def predict(
    *, model: torch.nn.Module, x: torch.Tensor, prep_: Prep, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 11 quantiles (rows, 11) in output over scale, and the probability of a negative output."""
    k = SETTING.components
    quantiles, negative = [], []
    zero_z = (0.0 - prep_.target_mean) / prep_.target_std
    with torch.no_grad():
        for chunk in x.split(BATCH * 4):
            log_w, mean, std = base.mixture_parts(out=model(chunk), components=k)
            q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std)
            quantiles.append(q.cpu().numpy())
            cdf = (
                log_w.exp() * 0.5 * (1 + torch.erf((zero_z - mean) / (std * math.sqrt(2.0))))
            ).sum(-1)
            negative.append(cdf.cpu().numpy())
    q = np.sort(np.concatenate(quantiles) * prep_.target_std + prep_.target_mean, axis=1)
    return q.astype(np.float32), np.concatenate(negative).astype(np.float32)


def fit_one(*, arm: str, month: str, seed: int, frame: Frame, device: str) -> str:
    """Fit one arm, fold, and seed, save the network, and write the fold's forecasts."""
    target = MODELS_DIR / arm / month / f"s{seed}"
    pred_path = PRED_DIR / arm / f"{month}_s{seed}.parquet"
    if pred_path.exists():
        return f"{arm} {month} s{seed}: exists"
    started = time.monotonic()
    refit = prep.refit_of(month=month)
    train, held, test = fold_masks(frame=frame, refit=refit)
    assert np.isfinite(frame.weather).all(), "the ensemble summary holds a NaN"
    prep_ = fit_prep(frame=frame, train=train)
    fit_rows = np.nonzero(train | held)[0]
    x = torch.from_numpy(matrix(frame=frame, rows=fit_rows, arm=arm, prep_=prep_)).to(device)
    z = torch.from_numpy(
        ((frame.y[fit_rows] - prep_.target_mean) / prep_.target_std).astype(np.float32)
    ).to(device)
    age_days = (_us(t=refit) - frame.valid[fit_rows]) / (86_400 * 1_000_000)
    w = torch.from_numpy((0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)).astype(np.float32)).to(
        device
    )
    is_val = torch.from_numpy(held[fit_rows]).to(device)
    torch.manual_seed(seed)
    model = base.PlainMdn(
        n_inputs=x.shape[1], hidden=SETTING.hidden, layers=SETTING.layers,
        components=SETTING.components, dropout=SETTING.dropout,
    ).to(device)  # fmt: skip
    epochs, best = train_network(model=model, x=x, z=z, w=w, is_val=is_val, seed=seed)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    rows = np.nonzero(test)[0]
    xt = torch.from_numpy(matrix(frame=frame, rows=rows, arm=arm, prep_=prep_)).to(device)
    q, p_neg = predict(model=model, x=xt, prep_=prep_, device=device)
    table = pl.DataFrame(
        {
            "origin_us": frame.origin[rows], "valid_us": frame.valid[rows],
            "lead": frame.lead[rows], "actual": frame.y[rows], "p_neg": p_neg,
            **{f"q{round(level * 100):02d}": q[:, i] for i, level in enumerate(LEVELS)},
        }
    )  # fmt: skip
    (target / "meta.json").write_text(
        json.dumps(
            {
                "arm": arm, "month": month, "seed": seed, "epochs": epochs, "val_nll": best,
                "train_rows": int(train.sum()), "held_rows": int(held.sum()),
                "test_rows": int(test.sum()), "test_rows_with_actual": int(np.isfinite(frame.y[rows]).sum()),
                "n_inputs": int(x.shape[1]),
            }
        )
    )  # fmt: skip
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pred_path.with_name(pred_path.name + ".tmp")
    table.write_parquet(temporary)
    temporary.rename(pred_path)
    return (
        f"{arm} {month} s{seed}: {epochs} epochs, val nll {best:.4f}, {int(train.sum()):,} train rows, "
        f"{time.monotonic() - started:.0f} s"
    )


def selftest(*, frame: Frame, power: np.ndarray) -> None:
    """Check the lag indices against experiment 87's `origin_lags` on a sample of rows."""
    rng = np.random.default_rng(0)
    rows = rng.choice(len(frame.y), 2_000, replace=False)
    origin = np.asarray(
        common.grid_index(times=pl.Series(frame.origin[rows]).cast(pl.Datetime("us", "UTC")))
    )
    target = np.asarray(
        common.grid_index(times=pl.Series(frame.valid[rows]).cast(pl.Datetime("us", "UTC")))
    )
    expected = common.origin_lags(power=power, target=target, origin=origin)
    np.testing.assert_array_equal(
        np.nan_to_num(expected, nan=-9.0), np.nan_to_num(frame.lags[rows], nan=-9.0)
    )


def main() -> None:
    """Fit the requested arms, months, and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--months", nargs="+", default=list(SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    power, price, _ = grid_arrays()
    frame = build_frame(power=power, price=price, weather=weather_table())
    selftest(frame=frame, power=power)
    print(f"{len(frame.y):,} rows on the grid", flush=True)
    for month in args.months:
        for seed in args.seeds:
            for arm in args.arms:
                print(
                    fit_one(arm=arm, month=month, seed=seed, frame=frame, device=args.device),
                    flush=True,
                )


if __name__ == "__main__":
    main()

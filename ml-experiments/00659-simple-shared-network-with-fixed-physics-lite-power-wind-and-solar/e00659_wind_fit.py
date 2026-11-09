# ruff: noqa: E501
"""Fit the simple shared wind networks of experiment 659 at day 0 and day 14 of each development month.

One pooled network per (arm, seed, refit): two hidden layers of 64 units (experiment 310's
`MixtureHead`), a three-component Gaussian mixture, 11 quantiles read from the mixture and capped
at capacity. No member encoder, no pretraining, no learned plant, equal row weights, one-hot farm.

Arms:

- `S`: mean and standard deviation across the 51 members of the raw weather (10 m and 100 m wind
  speed, sine and cosine of the 100 m direction, air temperature).
- `PLP`: `S` plus the mean and standard deviation of the fixed physics-lite power.
- `PL`: the mean and standard deviation of the physics-lite power only.
- `S_cal`, `PLP_cal`: `S` and `PLP` plus sine and cosine of the hour of day and day of year.

Capacity is the 99.9th percentile of the farm's power over the 730 days before the refit. The
target is the observed power over that capacity with the idle-mask rows replaced by zero ("cleaned
zeros"), in training and in scoring. The mixture's centre is the mean physics-lite power (`PL`,
`PLP`) or the training mean of the target (`S`), so `S` has no physics-lite input anywhere.
A refit at `R` trains on runs with `init` and valid time before `R - 62 days` and stops early on the
runs from that date to `R`, as experiment 615 did.

    source PYTHONPATH from e00659_run_wind.sh
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00659-*/e00659_wind_fit.py ARM [--seeds 0 1 2]
"""

import argparse
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00461_common as common461  # ty: ignore[unresolved-import]
import e00615_wind_fit as w615  # ty: ignore[unresolved-import]
import e00659_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS: Final[int] = int(os.environ.get("E00659_EPOCHS", "40"))
BATCH: Final[int] = 2048
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4
STOP_WINDOW_DAYS: Final[int] = 62
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
CALENDAR: Final[tuple[str, ...]] = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")
ARMS: Final[dict[str, tuple[tuple[str, ...], bool]]] = {
    "S": (common.RAW_NAMES, False),
    "PLP": ((*common.RAW_NAMES, *common.PL_NAMES), False),
    "PL": (common.PL_NAMES, False),
    "S_cal": (common.RAW_NAMES, True),
    "PLP_cal": ((*common.RAW_NAMES, *common.PL_NAMES), True),
}
"""Arm -> (weather columns, whether the calendar columns are added)."""


def load_cases() -> pl.DataFrame:
    """Return every daily case with its 10 raw-weather and 2 physics-lite summaries and the target."""
    frame, _ = w615.all_cases()
    features = pl.concat([pl.read_parquet(common.FEATURE_DIR / f"{f}.parquet") for f in FARMS])
    out = frame.join(features, on=KEYS, how="left", validate="1:1")
    names = [*common.RAW_NAMES, *common.PL_NAMES]
    missing = sum(out[n].null_count() + out[n].is_nan().sum() for n in names)
    if missing:
        msg = f"{missing} cases have no member summaries"
        raise ValueError(msg)
    # Cleaned zeros: the idle-mask rows hold zero power.
    out = out.filter(pl.col("valid_time") < common.VALID_LIMIT)
    return out.with_columns(
        actual_clean=pl.when(pl.col("masked")).then(0.0).otherwise(pl.col("actual_norm"))
    ).sort("site", *KEYS)


def design(*, frame: pl.DataFrame, weather: tuple[str, ...], calendar: bool) -> np.ndarray:
    """Return the network inputs: weather summaries, lead terms, one-hot farm, optional calendar."""
    hours = frame["lead_hours"].to_numpy().astype(np.float64)
    angle = 2 * np.pi * hours / 24.0
    parts = [
        frame.select(list(weather)).to_numpy().astype(np.float64),
        np.stack([np.sin(angle), np.cos(angle), hours / 100.0], axis=1),
        np.stack([(frame["site"] == f).to_numpy().astype(np.float64) for f in FARMS], axis=1),
    ]
    if calendar:
        valid = frame["valid_time"]
        hour = (valid.dt.hour() + valid.dt.minute() / 60.0).to_numpy()
        day = valid.dt.ordinal_day().to_numpy()
        parts.append(
            np.stack(
                [
                    np.sin(2 * np.pi * hour / 24.0),
                    np.cos(2 * np.pi * hour / 24.0),
                    np.sin(2 * np.pi * day / 366.0),
                    np.cos(2 * np.pi * day / 366.0),
                ],
                axis=1,
            )
        )
    return np.concatenate(parts, axis=1)


def centre(*, frame: pl.DataFrame, arm: str, constant: float) -> np.ndarray:
    """Return the mixture's centre: the physics-lite mean, or the constant for the arm `S`."""
    if arm.startswith("S"):
        return np.full(frame.height, constant)
    return frame["pl_mean"].to_numpy().astype(np.float64)


def capacity_730(*, site_label: str, refit: datetime) -> dict[str, float]:
    """Return old and new capacity (MW) with the new one from the 730 days before `refit`."""
    series = pl.read_parquet(common461.SERIES_PATH).filter(pl.col("site") == site_label)
    start = refit - timedelta(days=common.CAPACITY_DAYS)
    earlier = series.filter((pl.col("time") >= start) & (pl.col("time") < refit))["power"]
    values = earlier.drop_nulls().to_numpy()
    old = float(series["cap"][0])
    new = float(np.quantile(values, common.CAPACITY_QUANTILE))
    return {"old_mw": old, "new_mw": new, "ratio": new / old, "n_history": len(values)}


def train_and_predict(
    *,
    train: pl.DataFrame,
    target: pl.DataFrame,
    arm: str,
    refit: datetime,
    seed: int,
) -> np.ndarray:
    """Train one network on `train` (stopping on the last 62 days) and return `target`'s quantiles."""
    weather, calendar = ARMS[arm]
    torch.manual_seed(seed)
    cutoff = refit - timedelta(days=STOP_WINDOW_DAYS)
    is_fit = ((train["power_fcst_init_time"] < cutoff) & (train["valid_time"] < cutoff)).to_numpy()
    is_stop = (train["power_fcst_init_time"] >= cutoff).to_numpy() & (
        train["valid_time"] < refit
    ).to_numpy()
    x_train = design(frame=train, weather=weather, calendar=calendar)
    mean, std = x_train[is_fit].mean(axis=0), np.maximum(x_train[is_fit].std(axis=0), 1e-6)
    y = train["target"].to_numpy()
    constant = float(y[is_fit].mean())
    mu = centre(frame=train, arm=arm, constant=constant)

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)

    x_t, mu_t = tensor(x_train), torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    model = f310.MixtureHead(x_train.shape[1]).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    fit_i = torch.tensor(np.flatnonzero(is_fit), device=DEVICE)
    stop_i = torch.tensor(np.flatnonzero(is_stop), device=DEVICE)
    generator = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        order = fit_i[torch.randperm(len(fit_i), generator=generator).to(DEVICE)]
        for start in range(0, len(order), BATCH):
            i = order[start : start + BATCH]
            loss = model.nll({"x": x_t[i], "mu": mu_t[i], "actual": y_t[i]})
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(model.nll({"x": x_t[stop_i], "mu": mu_t[stop_i], "actual": y_t[stop_i]}))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    common.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = common.MODEL_DIR / f"wind_{arm}_{refit:%Y-%m-%d}_seed{seed}.tmp"
    torch.save({"state": best_state, "mean": mean.tolist(), "std": std.tolist(), "constant": constant}, tmp)
    tmp.rename(tmp.with_suffix(".pt"))
    xs_t = tensor(design(frame=target, weather=weather, calendar=calendar))
    mu_s = centre(frame=target, arm=arm, constant=constant)
    result = np.empty((target.height, 11), dtype=np.float32)
    for start in range(0, target.height, 50_000):
        sl = slice(start, start + 50_000)
        batch = {"x": xs_t[sl], "mu": torch.tensor(mu_s[sl], dtype=torch.float32, device=DEVICE)}
        result[sl] = f310.mixture_quantiles(model=model, batch=batch, ceiling=1.0)
    return np.sort(result, axis=1)


def main() -> None:
    """Fit every refit and seed of the named arm, skipping finished parts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--first-only", action="store_true")
    args = parser.parse_args()
    common.FIT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_cases()
    labels = common461.WIND_LABEL
    dates = w615.refit_dates()
    started = time.time()
    for index, refit in enumerate(dates):
        upper = (
            dates[index + 1]
            if index + 1 < len(dates) and dates[index + 1].month == refit.month
            else w615.month_end(refit=refit)
        )
        cap = {f: capacity_730(site_label=labels[f], refit=refit) for f in FARMS}
        ratio = {f: cap[f]["ratio"] for f in FARMS}
        scale = pl.col("site").replace_strict(ratio, return_dtype=pl.Float64)
        before = frame.filter(
            (pl.col("power_fcst_init_time") < refit) & pl.col("actual_clean").is_finite()
        ).with_columns(target=pl.col("actual_clean") / scale)
        target = frame.filter(
            (pl.col("power_fcst_init_time") >= refit) & (pl.col("power_fcst_init_time") < upper)
        )
        for seed in args.seeds:
            path = common.FIT_DIR / f"{args.arm}_refit={refit:%Y-%m-%d}_seed={seed}.parquet"
            if path.exists():
                continue
            out = train_and_predict(train=before, target=target, arm=args.arm, refit=refit, seed=seed)
            mw = out * target["site"].replace_strict(
                {f: cap[f]["new_mw"] for f in FARMS}, return_dtype=pl.Float64
            ).to_numpy()[:, None]
            table = target.select(
                "site", *KEYS, "lead_hours", "masked", "actual_norm", "actual_clean"
            ).with_columns(
                refit=pl.lit(refit),
                seed=pl.lit(seed),
                new_mw=target["site"].replace_strict(
                    {f: cap[f]["new_mw"] for f in FARMS}, return_dtype=pl.Float64
                ),
                old_mw=target["site"].replace_strict(
                    {f: cap[f]["old_mw"] for f in FARMS}, return_dtype=pl.Float64
                ),
                **{f"q{i}": pl.Series(mw[:, i]) for i in range(11)},
            )
            table.write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            print(
                f"{args.arm} refit {refit:%Y-%m-%d} seed {seed}: {table.height:,} rows, "
                f"{time.time() - started:.0f} s",
                flush=True,
            )
            if args.first_only:
                return


if __name__ == "__main__":
    main()

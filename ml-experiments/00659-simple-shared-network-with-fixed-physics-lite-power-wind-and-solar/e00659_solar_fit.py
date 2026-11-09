# ruff: noqa: E402
"""Fit the simple shared solar networks of experiment 659 at the four quarterly refits.

Rows, refits, and scoring follow experiment 430's `a2_member` job: farms A to F pooled over the six
day bands, four refits (2025-09, 2025-12, 2026-03, 2026-06), each training on rows whose valid time
is at least one day before the refit and stopping early on the last 61 days. The network is the
same two hidden layers of 64 units and three Gaussian components. No member encoder, no
pretraining, no learned plant, no farm column (experiment 430's network has none), and no calendar
column; the sun position (elevation and the sine and cosine of the azimuth) and the lead are inputs.

Arms:

- `S`: mean and standard deviation across the 51 members of shortwave, air temperature, and 10 m
  wind speed, plus the ensemble means of relative humidity, longwave radiation, precipitation, and
  surface pressure (experiment 430 stored only their means).
- `PLP`: `S` plus the mean and standard deviation of the fixed physics-lite power and of the
  Faiman cell temperature.
- `PL`: the mean and standard deviation of the physics-lite power and of the cell temperature only.

Physics-lite per member: `clip(G / 1000 * (1 + gamma * (T_cell - 25)), 0, 1)` with
`gamma = -0.004 / K` and `T_cell = T_air + G / (U0 + U1 * wind)`, `U0 = 25 W/m2/K`,
`U1 = 6.84 W/m2/K per m/s`. Capacity is the 99.9th percentile of the farm's power over the 730
days before the refit, and the target is the output over the old capacity divided by
`new / old`. Stored quantiles are fractions of the old capacity, the scoring capacity of
experiment 615's solar candidates.

    PYTHONPATH=<310>:<430>:<659 folders> OMP_NUM_THREADS=1 systemd-run --user --scope \
        -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00659-*/e00659_solar_fit.py ARM [--seeds 0 1 2]
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00430_fit as f430  # ty: ignore[unresolved-import]
import e00461_common as common461  # ty: ignore[unresolved-import]
import e00659_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS: Final[int] = 20
BATCH: Final[int] = 2048
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4
VALIDATION_DAYS: Final[int] = 61
GAMMA: Final[float] = -0.004
U0: Final[float] = 25.0
U1: Final[float] = 6.84
OUT_DIR: Final = common.DATA_DIR / "solar_fits"
SUN: Final[tuple[str, ...]] = ("elevation_deg", "azimuth_sin", "azimuth_cos", "lead_hours")
RAW: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_sd",
    "temp_mean",
    "temp_sd",
    "wind_mean",
    "wind_sd",
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
)
PHYSICS: Final[tuple[str, ...]] = ("pl_mean", "pl_sd", "cell_mean", "cell_sd")
ARMS: Final[dict[str, tuple[str, ...]]] = {
    "S": (*RAW, *SUN),
    "PLP": (*RAW, *PHYSICS, *SUN),
    "PL": (*PHYSICS, *SUN),
}
SERIES_LABEL: Final[dict[str, str]] = {f: f"Solar {f}" for f in c310.FARMS}
REFITS: Final = f310.REFIT_STARTS


def wind_array(*, farm: str, band: str) -> np.ndarray:
    """Return the (rows, 51) 10 m wind speed in `solar_base` row order."""
    return np.concatenate(
        [np.load(common.DATA_DIR / "solar_wind" / f"{farm}_d{day}.npy") for day in c310.BANDS[band]]
    )


def physics_lite(
    *, ghi: np.ndarray, temp: np.ndarray, wind: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-member physics-lite power (unit capacity) and the Faiman cell temperature."""
    cell = temp + ghi / (U0 + U1 * wind)
    power = np.clip(ghi / 1000.0 * (1.0 + GAMMA * (cell - 25.0)), 0.0, 1.0)
    return power, cell


def farm_band_frame(*, farm: str, band: str) -> pl.DataFrame:
    """Return one farm-band's rows with the arm inputs, before the valid limit."""
    base = f430.solar_base(farm, band)
    members = f430.member_array(farm, band).astype(np.float64)
    wind = wind_array(farm=farm, band=band).astype(np.float64)
    if members.shape[0] != base.height or wind.shape[0] != base.height:
        msg = f"{farm} {band}: member arrays do not match the rows"
        raise ValueError(msg)
    ghi, temp = members[:, :, 0], members[:, :, 1]
    power, cell = physics_lite(ghi=ghi, temp=temp, wind=wind)
    frame = base.select(
        "time",
        "init_time",
        "lead_hours",
        "output",
        "elevation_deg",
        "azimuth_sin",
        "azimuth_cos",
        "relative_humidity",
        "longwave",
        "precipitation",
        "pressure_surface",
    ).with_columns(
        series=pl.lit(farm),
        band=pl.lit(band),
        ghi_mean=pl.Series(ghi.mean(axis=1)),
        ghi_sd=pl.Series(ghi.std(axis=1)),
        temp_mean=pl.Series(temp.mean(axis=1)),
        temp_sd=pl.Series(temp.std(axis=1)),
        wind_mean=pl.Series(wind.mean(axis=1)),
        wind_sd=pl.Series(wind.std(axis=1)),
        pl_mean=pl.Series(power.mean(axis=1)),
        pl_sd=pl.Series(power.std(axis=1)),
        cell_mean=pl.Series(cell.mean(axis=1)),
        cell_sd=pl.Series(cell.std(axis=1)),
    )
    return frame.filter(pl.col("time") < common.VALID_LIMIT)


def load_cases() -> pl.DataFrame:
    """Return every farm and band's rows, sorted by series, band, and time."""
    return pl.concat([farm_band_frame(farm=f, band=b) for f in c310.FARMS for b in c310.BAND_ORDER])


def capacity_730(*, farm: str, refit: datetime) -> dict[str, float]:
    """Return old and new capacity (MW), the new one from the 730 days before the refit."""
    series = pl.read_parquet(common461.SERIES_PATH).filter(pl.col("site") == SERIES_LABEL[farm])
    start = refit - timedelta(days=common.CAPACITY_DAYS)
    values = (
        series.filter((pl.col("time") >= start) & (pl.col("time") < refit))["power"]
        .drop_nulls()
        .to_numpy()
    )
    old = float(series["cap"][0])
    new = float(np.quantile(values, common.CAPACITY_QUANTILE))
    return {"old_mw": old, "new_mw": new, "ratio": new / old, "n_history": len(values)}


def train_and_predict(
    *, train: pl.DataFrame, target: pl.DataFrame, arm: str, seed: int, refit: datetime
) -> np.ndarray:
    """Train one network and return `target`'s 11 quantiles as fractions of the new capacity."""
    columns = ARMS[arm]
    torch.manual_seed(seed)
    cutoff = refit - timedelta(days=f310.REFIT_GAP_DAYS)
    is_stop = (train["time"] >= cutoff - timedelta(days=VALIDATION_DAYS)).to_numpy()
    is_fit = ~is_stop
    x = train.select(list(columns)).to_numpy().astype(np.float64)
    mean, std = x[is_fit].mean(axis=0), np.maximum(x[is_fit].std(axis=0), 1e-6)
    y = train["scaled"].to_numpy()
    constant = float(y[is_fit].mean())

    def centre(frame: pl.DataFrame) -> np.ndarray:
        if arm == "S":
            return np.full(frame.height, constant)
        return frame["pl_mean"].to_numpy().astype(np.float64)

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.tensor((array - mean) / std, dtype=torch.float32, device=DEVICE)

    x_t = tensor(x)
    mu_t = torch.tensor(centre(train), dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    model = f310.MixtureHead(x.shape[1]).to(DEVICE)
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
    tmp = common.MODEL_DIR / f"solar_{arm}_{refit:%Y-%m}_seed{seed}.tmp"
    torch.save(
        {"state": best_state, "mean": mean.tolist(), "std": std.tolist(), "constant": constant}, tmp
    )
    tmp.rename(tmp.with_suffix(".pt"))
    xs_t = tensor(target.select(list(columns)).to_numpy().astype(np.float64))
    mu_s = centre(target)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_cases()
    started = time.time()
    for refit in REFITS:
        end = f310.month_add(start=refit, months=f310.REFIT_MONTHS)
        cutoff = refit - timedelta(days=f310.REFIT_GAP_DAYS)
        cap = {f: capacity_730(farm=f, refit=refit) for f in c310.FARMS}
        ratio = pl.col("series").replace_strict(
            {f: cap[f]["ratio"] for f in c310.FARMS}, return_dtype=pl.Float64
        )
        train = frame.filter(pl.col("time") < cutoff).with_columns(
            scaled=pl.col("output").cast(pl.Float64) / ratio
        )
        target = frame.filter((pl.col("init_time") >= refit) & (pl.col("init_time") < end))
        for seed in args.seeds:
            path = OUT_DIR / f"{args.arm}_refit={refit:%Y-%m}_seed={seed}.parquet"
            if path.exists():
                continue
            q = train_and_predict(train=train, target=target, arm=args.arm, seed=seed, refit=refit)
            r = (
                target["series"]
                .replace_strict({f: cap[f]["ratio"] for f in c310.FARMS}, return_dtype=pl.Float64)
                .to_numpy()[:, None]
            )
            table = target.select(
                "series", "band", "init_time", "time", "lead_hours", actual=pl.col("output")
            ).with_columns(
                refit=pl.lit(refit),
                seed=pl.lit(seed),
                **{f"q{i}": pl.Series((q * r)[:, i]) for i in range(11)},
            )
            table.write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            print(
                f"{args.arm} refit {refit:%Y-%m} seed {seed}: {table.height:,} rows, "
                f"{time.time() - started:.0f} s",
                flush=True,
            )
            if args.first_only:
                return


if __name__ == "__main__":
    main()

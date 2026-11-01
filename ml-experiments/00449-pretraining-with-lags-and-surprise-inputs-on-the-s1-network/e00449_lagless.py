# ruff: noqa: E501
"""The cross-fitted lagless forecast table on ERA5 and CAMS weather, for the pretraining years.

Experiment 404's table forecasts each half-hour from ensemble weather and covers only the ensemble
years. Pretraining in experiment 449 starts in 2019-09, so the surprise at a lag there needs a
lagless forecast built from ERA5 and CAMS weather. For one scored month's refit, the half-hours from
`PRETRAIN_START` to the refit are split into `FOLDS` contiguous blocks. A pooled lagless module
(experiment 404's `LaglessModule`: all 16 Primaries, a Primary embedding, no lag, no lead) fitted on
the other blocks forecasts each held-out block, so no half-hour is forecast by a module that saw it.
Nothing at or after the refit enters any fit. The inputs are the 7 ERA5 and CAMS weather means, the
solar position, the calendar, and the holiday columns of experiment 397; the ensemble spread columns
are dropped because ERA5 has none. Rows have equal weights.

```bash
source <paths>; PYTHONPATH=$PP OMP_NUM_THREADS=1 \
  uv run --no-sync python ml-experiments/00449-*/e00449_lagless.py build --month 2026-06 --seed 0
  uv run --no-sync python ml-experiments/00449-*/e00449_lagless.py smoke
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_lagless as lag  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00449-pretraining-with-lags-and-surprise-inputs-on-the-s1-network"
    / "work"
)
TABLE_DIR: Final[Path] = WORK_DIR / "lagless_era5"
FOLDS: Final[int] = lag.FOLDS
GROUPS: Final[tuple[str, ...]] = lag.data.LAGLESS_GROUPS


def lagless_keep() -> tuple[list[str], np.ndarray]:
    """Return the lagless columns without the spread columns, and their indices in the full matrix."""
    names, _, layout = prep.feature_names(scheme="summary")
    keep = [i for i in layout.columns_of(groups=GROUPS) if not names[i].startswith("std_")]
    return [names[i] for i in keep], np.array(keep)


def dense_era5_inputs(*, site_row: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """Return `(first, x, valid)`: a Primary's raw lagless inputs on the dense grid from PRETRAIN_START.

    A half-hour with a missing ERA5 or CAMS value gets zero weather and `valid == 0`.
    """
    weather = prep418.dense_weather(series_id=site_row["time_series_id"], site=site_row["site"])
    first = int(common.grid_index(times=prep418.PRETRAIN_START))
    weather = weather[first:]
    valid = np.isfinite(weather).all(axis=1)
    weather = np.where(valid[:, None], weather, 0.0)
    grid = common.grid_times().slice(first, len(weather))
    columns: dict[str, np.ndarray] = {}
    for column, name in enumerate(prep418.WEATHER):
        columns[f"mean_{name}"] = weather[:, column]
        columns[f"std_{name}"] = np.zeros(len(weather), dtype=np.float32)
    frame = pl.DataFrame({"valid_time": grid, **columns})
    frame = common.with_derived(
        frame=frame, latitude=site_row["latitude"], longitude=site_row["longitude"]
    ).with_columns(pl.lit(None, dtype=pl.Float32).alias(c) for c in common.LAG_COLUMNS)
    frame = frame.with_columns(lead_hours=pl.lit(24.0))
    raw = prep.feature_matrix(frame=frame, scheme="summary")
    _, keep = lagless_keep()
    return first, raw[:, keep], valid.astype(np.int8)


def cut_early(*, site: str, grid: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return `y` with the targets before experiment 418's `FIRST_USABLE` date of the site missing."""
    cut = prep418.FIRST_USABLE.get(site)
    if cut is None:
        return y
    return np.where(grid >= int(common.grid_index(times=cut)), y, np.nan)


def build_pool(*, lo: int, hi: int, clean: np.ndarray) -> lag.Pool:
    """Return the dense rows of every Primary with grid index in `[lo, hi)` and finite weather.

    `clean` is world 0 (primaries, grid): cleaned power over capacity, NaN where flagged or cut.
    """
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        first, x, valid = dense_era5_inputs(site_row=row)
        grid = first + np.arange(len(x))
        keep = (grid >= lo) & (grid < hi) & (valid == 1)
        parts.append(
            lag.Pool(
                site=np.full(int(keep.sum()), row["primary"], dtype=np.int64),
                grid=grid[keep],
                x=x[keep],
                y=cut_early(site=row["site"], grid=grid[keep], y=clean[row["primary"]][grid[keep]]),
            )
        )
    return lag.Pool(
        site=np.concatenate([p.site for p in parts]),
        grid=np.concatenate([p.grid for p in parts]),
        x=np.concatenate([p.x for p in parts]),
        y=np.concatenate([p.y for p in parts]),
    )


def crossfit_table(*, month: str, seed: int, device: str, clean: np.ndarray) -> pl.DataFrame:
    """Return the cross-fitted ERA5 table for a scored month's refit."""
    refit = prep.refit_of(month=month)
    end = int(common.grid_index(times=refit))
    start = int(common.grid_index(times=prep418.PRETRAIN_START))
    pool = build_pool(lo=start, hi=end, clean=clean)
    assert pool.grid.max() < end, "the table must stop before the refit"
    observed = np.isfinite(pool.y)
    block = lag.block_of(grid=pool.grid, lo=start, hi=end, folds=FOLDS)
    parts = []
    for k in range(FOLDS):
        held = np.flatnonzero(block == k)
        fit_mask = observed & (block != k)
        scaler = lag.Scaler(
            x=pool.x[fit_mask], site=pool.site[fit_mask], y=pool.y[fit_mask], n_primaries=16
        )
        module = lag.fit_module(
            x=torch.tensor(scaler.inputs(x=pool.x[fit_mask]), device=device),
            z=torch.tensor(
                scaler.target(y=pool.y[fit_mask], site=pool.site[fit_mask]), device=device
            ),
            site=torch.tensor(pool.site[fit_mask], device=device),
            weight=torch.ones(int(fit_mask.sum()), dtype=torch.float32, device=device),
            seed=seed,
        )
        _, quant = lag.predict(
            module=module,
            x=torch.tensor(scaler.inputs(x=pool.x[held]), device=device),
            site=torch.tensor(pool.site[held], device=device),
        )
        site = pool.site[held]
        q = quant * scaler.y_std[site][:, None] + scaler.y_mean[site][:, None]
        parts.append(
            pl.DataFrame(
                {
                    "primary": site.astype(np.int16), "grid": pool.grid[held].astype(np.int32),
                    "block": np.full(len(held), k, dtype=np.int8),
                    "q05": q[:, 0].astype(np.float32), "q50": q[:, 1].astype(np.float32),
                    "q95": q[:, 2].astype(np.float32),
                }
            )
        )  # fmt: skip
    table = pl.concat(parts)
    return table.join(half_scales(month=month, seed=seed, table=table, clean=clean), on="primary")


def surprise_sd(
    *, median: np.ndarray, half: np.ndarray, clean: np.ndarray, lo: int, hi: int
) -> np.ndarray:
    """Return each Primary's surprise standard deviation over grid `[lo, hi)`, NaN where no overlap."""
    z = (clean[:, lo:hi] - median[:, lo:hi]) / half[:, lo:hi]
    out = np.full(len(z), np.nan)
    for p in range(len(z)):
        ok = np.isfinite(z[p])
        if ok.sum() > 1000:
            out[p] = z[p][ok].std()
    return out


def half_scales(*, month: str, seed: int, table: pl.DataFrame, clean: np.ndarray) -> pl.DataFrame:
    """Return the per-Primary factor that gives the ERA5 surprise the ensemble table's spread.

    Over the half-hours before the refit where both tables exist, the factor is the ERA5 surprise
    standard deviation over the ensemble one. A factor outside 0.25 to 4 raises.
    """
    median = np.full((len(data.sites()), data.grid_length()), np.nan, dtype=np.float32)
    half = np.full_like(median, np.nan)
    p, g = table["primary"].to_numpy(), table["grid"].to_numpy()
    median[p, g] = table["q50"].to_numpy()
    half[p, g] = np.maximum((table["q95"] - table["q05"]).to_numpy() / 2.0, 1e-3)
    ens = pl.read_parquet(fit_lagless_dir() / f"month={month}" / f"seed={seed}.parquet").filter(
        pl.col("source") == "crossfit"
    )
    e_median = np.full_like(median, np.nan)
    e_half = np.full_like(median, np.nan)
    e_median[ens["primary"].to_numpy(), ens["grid"].to_numpy()] = ens["q50"].to_numpy()
    e_half[ens["primary"].to_numpy(), ens["grid"].to_numpy()] = np.maximum(
        (ens["q95"] - ens["q05"]).to_numpy() / 2.0, 1e-3
    )
    both = np.isfinite(median) & np.isfinite(e_median)
    # Compare on the same half-hours: blank both tables outside the overlap.
    for m in (median, half, e_median, e_half):
        m[~both] = np.nan
    end = int(common.grid_index(times=prep.refit_of(month=month)))
    sd_era5 = surprise_sd(median=median, half=half, clean=clean, lo=0, hi=end)
    sd_ens = surprise_sd(median=e_median, half=e_half, clean=clean, lo=0, hi=end)
    ratio = sd_era5 / sd_ens
    print(
        f"  {month} s{seed} half-width factors: " + " ".join(f"{r:.2f}" for r in ratio), flush=True
    )
    if not ((ratio > 0.25) & (ratio < 4.0)).all():
        msg = f"{month} s{seed}: a half-width factor is outside 0.25 to 4: {ratio}"
        raise ValueError(msg)
    return pl.DataFrame(
        {"primary": np.arange(len(ratio), dtype=np.int16), "half_scale": ratio.astype(np.float32)}
    )


def fit_lagless_dir() -> Path:
    """Return the folder of experiment 404's ensemble-based tables (read by experiment 428)."""
    import e00428_fit as fit  # ty: ignore[unresolved-import]

    return fit.LAGLESS_DIR


def table_path(*, month: str, seed: int) -> Path:
    """Return where a scored month's ERA5 table is stored."""
    return TABLE_DIR / f"month={month}" / f"seed={seed}.parquet"


def surprise_tables(*, month: str, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the table as `(median, half band width)` tensors (primaries, grid), NaN where absent."""
    table = pl.read_parquet(table_path(month=month, seed=seed))
    median = np.full((len(data.sites()), data.grid_length()), np.nan, dtype=np.float32)
    half = np.full_like(median, np.nan)
    p, g = table["primary"].to_numpy(), table["grid"].to_numpy()
    median[p, g] = table["q50"].to_numpy()
    half[p, g] = np.maximum(
        (table["q95"] - table["q05"]).to_numpy() / 2.0 * table["half_scale"].to_numpy(), 1e-3
    )
    return torch.from_numpy(median).to(device), torch.from_numpy(half).to(device)


def build(*, month: str, seed: int, device: str, clean: np.ndarray) -> str:
    """Build and write a table; return a progress line."""
    target = table_path(month=month, seed=seed)
    if target.exists():
        return f"{month} s{seed}: exists"
    started = time.monotonic()
    table = crossfit_table(month=month, seed=seed, device=device, clean=clean)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    table.write_parquet(temporary)
    temporary.rename(target)
    return f"{month} s{seed}: {table.height:,} rows, {time.monotonic() - started:.0f} s"


def smoke(*, device: str) -> None:
    """Check input width, no use of the refit or later, and the surprise spread by source."""
    stack, _ = data.load_stack(include_eval=False)
    names, _ = lagless_keep()
    first, _, valid = dense_era5_inputs(site_row=next(prep.primaries().iter_rows(named=True)))
    ens_names = lag.data.lagless_columns()
    print("ERA5 lagless columns", len(names), "ensemble lagless columns", len(ens_names))
    assert [n for n in ens_names if not n.startswith("std_")] == names
    print(f"first grid {first}, {int(valid.sum()):,} of {len(valid):,} half-hours with weather")
    month = "2026-01"
    print(build(month=month, seed=0, device=device, clean=stack[0]))
    table = pl.read_parquet(table_path(month=month, seed=0))
    end = int(common.grid_index(times=prep.refit_of(month=month)))
    assert table["grid"].max() < end
    median, half = surprise_tables(month=month, seed=0, device=device)
    ens_median, ens_half = lagless_ens_tables(month=month, seed=0, device=device)
    for label, (m, h) in (("ERA5", (median, half)), ("ensemble", (ens_median, ens_half))):
        sl = slice(int(common.grid_index(times=prep.refit_of(month="2025-01"))), end)
        z = ((torch.from_numpy(stack[0]).to(device)[:, sl] - m[:, sl]) / h[:, sl]).flatten()
        z = z[torch.isfinite(z)]
        print(
            f"{label}: surprise sd {float(z.std()):.3f}, mean {float(z.mean()):.3f}, n {len(z):,}"
        )
    print("smoke passed")


def lagless_ens_tables(*, month: str, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return experiment 428's ensemble-based tables for comparison in the smoke check."""
    import e00428_fit as fit  # ty: ignore[unresolved-import]

    table = pl.read_parquet(fit.LAGLESS_DIR / f"month={month}" / f"seed={seed}.parquet")
    median = np.full((len(data.sites()), data.grid_length()), np.nan, dtype=np.float32)
    half = np.full_like(median, np.nan)
    p, g = table["primary"].to_numpy(), table["grid"].to_numpy()
    median[p, g] = table["q50"].to_numpy()
    half[p, g] = np.maximum((table["q95"] - table["q05"]).to_numpy() / 2.0, 1e-3)
    return torch.from_numpy(median).to(device), torch.from_numpy(half).to(device)


def main() -> None:
    """Build tables or run the smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "smoke"))
    parser.add_argument(
        "--months", nargs="+", default=["2026-06", "2026-01", "2025-10", "2026-03", "2025-12"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(device=args.device)
        return
    stack, _ = data.load_stack(include_eval=False)
    for month in args.months:
        for seed in args.seeds:
            print(build(month=month, seed=seed, device=args.device, clean=stack[0]), flush=True)


if __name__ == "__main__":
    main()

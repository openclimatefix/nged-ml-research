# ruff: noqa: E501
"""Build the joint run tables of experiment 436: 51 values per node, half-hour, and run.

No fit. For each Tuesday run (09:00 UTC power forecast from the 00:00 UTC ensemble run) that
experiment 425 (demand), experiment 429 (wind), and experiment 430 (solar) all scored, writes
`work/runs/<date>.npz` with, on the 702 half-hours after the run and the 25 nodes in the order
16 Primaries, wind farms W1 to W3, solar farms A to F:

- `values`: (25, 702, 51) points of capacity at the levels k/52 (k = 1 to 51) by linear
  interpolation of the stored 11 quantiles, ascending along the last axis. Half-hours a node did
  not score hold the node's median (the value is never used: `actual` is NaN there);
- `actual`: (25, 702) points of capacity, NaN where the node has no scored actual;
- `ecc`: (25, 702, 51) uint8, the rank (0 = lowest value) that ensemble copula coupling gives each
  member, by temperature (shortwave where the median member's shortwave exceeds 20 W/m2) for demand,
  100 m wind for wind, and shortwave for solar;
- `kind`: the node's family, so that scorers need no other file.

It also writes `work/observed.npz`: the observed history of all 25 nodes on one half-hourly grid
from 2024-04-01, which the Schaake templates read, and `work/runs_found.txt`, the count of runs.

```bash
source ml-experiments/00436-*/env.sh
nice -n 19 uv run --no-sync python ml-experiments/00436-*/e00436_build.py
```
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00087_common as common  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_features as feats310  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # noqa: E402  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # noqa: E402  # ty: ignore[unresolved-import]
import e00381_score as s381  # noqa: E402  # ty: ignore[unresolved-import]
import e00429_data as data429  # noqa: E402  # ty: ignore[unresolved-import]
import e00434_score as s434  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

SLUG: Final[str] = "00436-joint-scenarios-across-demand-wind-solar-and-leads"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
WIND_PARTS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00429-unified-forecaster-wind-ablations" / "parts"
)
SOLAR_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00430-unified-forecaster-solar-and-pretraining-ablations"
    / "predictions"
    / "g0_onepass"
)
WIND_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
SOLAR_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
WIND_FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
SOLAR_FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
N_PRIMARIES: Final[int] = 16
N_NODES: Final[int] = N_PRIMARIES + len(WIND_FARMS) + len(SOLAR_FARMS)
N_STEPS: Final[int] = s434.N_STEPS
N_MEMBERS: Final[int] = s434.N_MEMBERS
STEP: Final[timedelta] = timedelta(minutes=30)
GRID_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 9, 12, tzinfo=UTC)
FIRST_RUN: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, 23, tzinfo=UTC)
WIND_100M: Final[int] = 1
DAYLIGHT_SHORTWAVE: Final[float] = 20.0
"""Median member shortwave (W/m2) above which demand's ECC ranks follow shortwave, not temperature."""
KIND: Final[np.ndarray] = np.array(
    [0] * N_PRIMARIES + [1] * len(WIND_FARMS) + [2] * len(SOLAR_FARMS), dtype=np.int8
)
"""Family of each node: 0 demand, 1 wind, 2 solar."""


def grid_index(*, t: datetime) -> int:
    """Return the index of a half-hour on the observed grid."""
    return int((t - GRID_START) / STEP)


def quantile_values(*, quantiles: np.ndarray) -> np.ndarray:
    """Return the 51 values at the levels k/52 from 11 quantiles (..., 11), sorted ascending."""
    return s434.fifty_one_values(quantiles=np.sort(quantiles, axis=-1))


def observed_grid(*, series: pl.DataFrame, actuals: pl.DataFrame) -> np.ndarray:
    """Return the observed history (25, n) in points of capacity; NaN is missing.

    Primaries come from experiment 434's cleaned grid; wind from the hindcast and scored frames
    (idle-band rows masked); solar from the farm output over capacity.
    """
    n = int((GRID_END - GRID_START) / STEP)
    out = np.full((N_NODES, n), np.nan)
    demand = s434.Grid(series=series, actuals=actuals)
    out[:N_PRIMARIES, : demand.n] = demand.values
    frames = pl.concat(
        [
            data347.hindcast_frame().select(
                "site", "valid_time", pl.col("actual_norm").cast(pl.Float64), "masked"
            ),
            data347.scored_frame().select(
                "site", "valid_time", pl.col("actual_norm").cast(pl.Float64), "masked"
            ),
        ]
    ).filter(~pl.col("masked"))
    for i, farm in enumerate(WIND_FARMS):
        sub = frames.filter(pl.col("site") == farm).unique(subset="valid_time")
        idx = ((sub["valid_time"] - GRID_START).dt.total_minutes() // 30).to_numpy()
        ok = (idx >= 0) & (idx < n)
        out[N_PRIMARIES + i, idx[ok]] = sub["actual_norm"].to_numpy()[ok] * 100.0
    for i, farm in enumerate(SOLAR_FARMS):
        sub = feats310.output_series(farm)
        idx = ((sub["time"] - GRID_START).dt.total_minutes() // 30).to_numpy()
        ok = (idx >= 0) & (idx < n)
        out[N_PRIMARIES + len(WIND_FARMS) + i, idx[ok]] = sub["output"].to_numpy()[ok] * 100.0
    return out


def wind_rows() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return the scored wind rows, their seed-mean quantiles (rows, 11), and 100 m wind (rows, 51)."""
    scored = data347.scored_frame().with_row_index("row")
    parts = np.mean(
        [np.load(WIND_PARTS / f"one_pass_ms_h{seed}.npy") for seed in WIND_SEEDS], axis=0
    )
    weather, _ = data429.load(set_name="scored")
    return scored, parts.astype(np.float64), weather[:, :, WIND_100M].astype(np.float64)


def solar_rows() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return the solar rows, their seed-mean quantiles (rows, 11), and shortwave ranks (rows, 51).

    The seeds' tables have the same rows in the same order (checked); the ranks come from seed 0.
    """
    levels = common.LEVEL_COLUMNS
    tables, ranks = [], []
    for path in sorted(SOLAR_DIR.glob("s0_*.parquet")):
        base = pl.read_parquet(path)
        stack = [base.select(levels).to_numpy().astype(np.float64)]
        for seed in SOLAR_SEEDS[1:]:
            other = pl.read_parquet(path.with_name(path.name.replace("s0_", f"s{seed}_")))
            if not other.select("time", "init_time", "time_series_id").equals(
                base.select("time", "init_time", "time_series_id")
            ):
                msg = f"{path.name}: seed {seed} has different rows from seed 0"
                raise ValueError(msg)
            stack.append(other.select(levels).to_numpy().astype(np.float64))
        z = np.load(path.with_name("paths_" + path.name.replace(".parquet", ".npz")))
        tables.append(base.select("time", "init_time", "lead_hours", "time_series_id", "actual"))
        tables[-1] = tables[-1].with_columns(
            *[pl.Series(c, np.mean(stack, axis=0)[:, i]) for i, c in enumerate(levels)]
        )
        ranks.append(z["rank"])
    table = pl.concat(tables).with_row_index("row")
    quantiles = table.select(levels).to_numpy()
    return table.drop(levels), quantiles, np.concatenate(ranks)


def ecc_ranks(*, key: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return each member's rank along the last axis, ties broken at random."""
    return s434.ranks_by_key(key=key, rng=rng)


def main() -> None:  # noqa: PLR0915
    """Write the observed grid and one table per joint run."""
    (WORK_DIR / "runs").mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ctx = s434.cleaned_context(series=series)
    ids = series["time_series_id"].to_list()
    observed = observed_grid(series=series, actuals=ctx["actuals"])
    np.savez_compressed(WORK_DIR / "observed.npz", observed=observed)
    wind, wind_q, wind_ws = wind_rows()
    solar, solar_q, solar_rank = solar_rows()
    wind_runs = {t for t in wind["power_fcst_init_time"].unique().to_list()}
    solar_inits = {t for t in solar["init_time"].unique().to_list()}
    demand_runs: set[datetime] = set()
    for month in s434.MONTHS:
        demand_runs |= set(
            s434.stored_arm_a(month=month)["power_fcst_init_time"].unique().to_list()
        )
    joint = sorted(
        t
        for t in wind_runs
        if FIRST_RUN <= t <= LAST_RUN and t in demand_runs and t.replace(hour=0) in solar_inits
    )
    print(
        f"joint runs: {len(joint)} (wind {len(wind_runs)}, demand {len(demand_runs)})", flush=True
    )
    (WORK_DIR / "runs_found.txt").write_text(
        f"{len(joint)} joint runs\n" + "\n".join(f"{t:%Y-%m-%d}" for t in joint) + "\n"
    )
    stored_by_month: dict[str, pl.DataFrame] = {}
    n_wind_rows = wind.height
    for init in joint:
        out = WORK_DIR / "runs" / f"{init:%Y-%m-%d}.npz"
        if out.exists():
            continue
        month = f"{init:%Y-%m}"
        if month not in stored_by_month:
            stored_by_month = {month: s434.stored_arm_a(month=month)}
        sub = stored_by_month[month].filter(pl.col("power_fcst_init_time") == init)
        order = {sid: i for i, sid in enumerate(ids)}
        sub = sub.with_columns(
            p=pl.col("time_series_id").replace_strict(order, return_dtype=pl.Int64)
        ).sort("p", "valid_time")
        if sub.height != len(ids) * N_STEPS:
            msg = f"{init}: demand rows {sub.height}"
            raise ValueError(msg)
        rng = np.random.default_rng(int(init.timestamp()) % (2**32))
        values = np.zeros((N_NODES, N_STEPS, N_MEMBERS))
        actual = np.full((N_NODES, N_STEPS), np.nan)
        ecc = np.zeros((N_NODES, N_STEPS, N_MEMBERS), dtype=np.uint8)
        # demand
        quantiles = sub.select(common.LEVEL_COLUMNS).to_numpy().reshape(N_PRIMARIES, N_STEPS, 11)
        values[:N_PRIMARIES] = quantile_values(quantiles=quantiles) * 100.0
        t_member, sw_member = s434.run_weather(init=init, ids=ids)
        day = np.median(sw_member, axis=-1, keepdims=True) > DAYLIGHT_SHORTWAVE
        key = np.where(day, sw_member, -t_member)
        ecc[:N_PRIMARIES] = ecc_ranks(key=key, rng=rng)
        i0 = grid_index(t=init) + 1
        actual[:N_PRIMARIES] = observed[:N_PRIMARIES, i0 : i0 + N_STEPS]
        # wind: the stored values are fractions of capacity; rows are in scored order
        for f, farm in enumerate(WIND_FARMS):
            node = N_PRIMARIES + f
            rows = wind.filter((pl.col("power_fcst_init_time") == init) & (pl.col("site") == farm))
            step = np.rint(rows["lead_hours"].to_numpy() * 2).astype(int) - 1
            ridx = rows["row"].to_numpy()
            if ridx.max() >= n_wind_rows:
                msg = "wind row index outside the stored parts"
                raise ValueError(msg)
            values[node] = np.median(wind_q[ridx], axis=0)[5] * 100.0  # fallback level
            values[node, step] = quantile_values(quantiles=wind_q[ridx]) * 100.0
            ecc[node, step] = ecc_ranks(key=wind_ws[ridx], rng=rng)
            ok = ~rows["masked"].to_numpy()
            actual[node, step[ok]] = rows["actual_norm"].to_numpy()[ok] * 100.0
        # solar: valid times map onto the 09:00 grid
        init00 = init.replace(hour=0)
        for f, farm in enumerate(SOLAR_FARMS):
            node = N_PRIMARIES + len(WIND_FARMS) + f
            rows = solar.filter(
                (pl.col("init_time") == init00) & (pl.col("time_series_id") == farm)
            )
            step = ((rows["time"] - init).dt.total_minutes() // 30).to_numpy().astype(int) - 1
            ok = (step >= 0) & (step < N_STEPS)
            ridx = rows["row"].to_numpy()[ok]
            step = step[ok]
            values[node] = 0.0
            values[node, step] = quantile_values(quantiles=solar_q[ridx]) * 100.0
            ecc[node, step] = solar_rank[ridx].astype(np.uint8)
            actual[node, step] = rows["actual"].to_numpy()[ok] * 100.0
        np.savez_compressed(
            out,
            values=values.astype(np.float32),
            actual=actual.astype(np.float32),
            ecc=ecc,
            kind=KIND,
            init=np.datetime64(init.replace(tzinfo=None), "us"),
        )
        print(f"{init:%Y-%m-%d} written", flush=True)


if __name__ == "__main__":
    main()

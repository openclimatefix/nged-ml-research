# ruff: noqa: E501
"""Score experiment 454's monthly lag-free references and L-1d's two seeds on experiment 511's rows.

Experiment 511's row table holds per-row scores for K2-S1, L-1d, and L-Ww. It holds none for
experiment 454's monthly lag-free arms (`LL`, `LL-ens3`, `LL-ens5`), and none for L-1d's individual
seeds. This script scores them from the stored quantiles with experiment 470's per-row CRPS formula
(eleven quantiles, atom masses of `e00087_common`, 1.15 widening about the median), joined to
experiment 511's rows on `(primary, origin, target)`. No network is fitted and no GPU is used.

K2-S1 is scored cell by cell (`K2-S1-c0` to `K2-S1-c2`) and as the mean of the three cells
(`K2-S1-3cell`). Experiment 511's own `K2-S1__crps` column is the score of cell c0 alone, because its
scorer was configured with no multi-cell arms; `K2-S1-c0` reproduces it. L-1d recomputed from its two
stored seeds (`L-1d-454rows`) is the second reproduction check.
The script writes `refs_{month}.parquet` and `flags_{month}.parquet` (experiment 454's `lag_real` and
`win_real` flags) under `work/` of this experiment's private data folder.

```bash
uv run --no-sync python ml-experiments/00521-*/e00521_refs.py
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00435_tailcore as tail  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free"
)
WORK: Final[Path] = DATA / "work"
E454: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00454-lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads"
    / "work"
    / "predictions"
)
E511: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining"
    / "work"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
KEYS: Final[list[str]] = ["primary", "origin", "target"]
LEVELS: Final[tuple[str, ...]] = common.LEVEL_COLUMNS
POINTS: Final[float] = 100.0
WIDENING: Final[float] = tail.WIDENING
STEP_US: Final[int] = 1_800_000_000
GRID_START_US: Final[int] = int(common.GRID_START.timestamp() * 1_000_000)
CELLS_454: Final[dict[str, tuple[int, ...]]] = {
    "LL": (0, 1, 2),
    "LL-ens3": (0,),
    "LL-ens5": (0,),
    "K2-S1": (0, 1, 2),
}
"""Arm -> stored cells. An arm with several cells is scored cell by cell and the CRPS averaged."""


def row_metrics(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row CRPS and tail pinball (points of capacity) and tail coverage (0 or 1).

    `frame` carries `actual` and the eleven quantile columns, already widened.
    """
    q = np.sort(frame.select(list(LEVELS)).to_numpy().astype(np.float64), axis=1)
    y = frame["actual"].to_numpy().astype(np.float64)[:, None]
    lev = common.LEVELS[None, :]
    pin = np.maximum(lev * (y - q), (lev - 1.0) * (y - q))
    out = {"crps": 2.0 * POINTS * (pin * common.ATOM_MASSES[None, :]).sum(axis=1)}
    for name in tail.TAIL_LEVELS:
        i = LEVELS.index(name)
        out[f"pin_{name}"] = POINTS * pin[:, i]
        out[f"cov_{name}"] = (y[:, 0] <= q[:, i]).astype(np.float64)
    return out


def score_cells(
    *, rows: pl.DataFrame, cells: list[pl.DataFrame], name: str, capacity: dict[int, float]
) -> pl.DataFrame:
    """Return `rows`' keys with the per-row metrics of `name`, averaged over its cells.

    A row that any cell does not hold gets NaN, so no arm is scored on a different row set.
    """
    per_cell = []
    for cell in cells:
        joined = rows.select(*KEYS).join(cell, on=KEYS, how="left", maintain_order="left")
        ok = np.isfinite(joined["q50"].to_numpy()) & np.isfinite(joined["actual"].to_numpy())
        widened = tail.widen(frame=joined.fill_null(0.0), factor=WIDENING)
        metrics = row_metrics(frame=widened)
        cap = np.array([capacity[p] for p in joined["primary"].to_list()])
        q = widened.select(["q01", "q05", "q95", "q99"]).to_numpy().astype(np.float64)
        metrics["w90"] = POINTS * (q[:, 2] - q[:, 1]) / cap
        metrics["w98"] = POINTS * (q[:, 3] - q[:, 0]) / cap
        per_cell.append({k: np.where(ok, v, np.nan) for k, v in metrics.items()})
    out = {f"{name}__{k}": np.mean([c[k] for c in per_cell], axis=0) for k in per_cell[0]}
    return pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in out.items()})


def read_454(*, arm: str, month: str, cell: int) -> pl.DataFrame:
    """Return one stored cell of an experiment 454 arm: the keys, `actual`, and the quantiles."""
    path = E454 / arm / f"{month}_s{cell}_w-1.parquet"
    return pl.read_parquet(path).select(*KEYS, "actual", *LEVELS)


def read_seed(*, month: str, seed: int, sites: pl.DataFrame, actual: pl.DataFrame) -> pl.DataFrame:
    """Return L-1d's seed table on the grid keys, with `actual` joined from experiment 454's rows."""
    frame = pl.read_parquet(E511 / "seed_preds" / "L-1d" / f"{month}_s{seed}.parquet")
    frame = frame.join(sites.select("time_series_id", "primary"), on="time_series_id").with_columns(
        primary=pl.col("primary").cast(pl.Int16),
        origin=((pl.col("power_fcst_init_time").dt.epoch("us") - GRID_START_US) // STEP_US).cast(
            pl.Int32
        ),
        target=((pl.col("valid_time").dt.epoch("us") - GRID_START_US) // STEP_US).cast(pl.Int32),
    )
    return frame.select(*KEYS, *LEVELS).join(actual, on=KEYS, how="inner")


def pooled_seeds(*, tables: list[pl.DataFrame]) -> pl.DataFrame:
    """Return the quantile average of seed tables on their common keys (experiment 511's L-1d)."""
    base = tables[0].select(*KEYS, "actual").sort(KEYS)
    stacked = np.stack(
        [
            base.select(*KEYS)
            .join(t.select(*KEYS, *LEVELS), on=KEYS, how="left", maintain_order="left")
            .select(list(LEVELS))
            .to_numpy()
            .astype(np.float64)
            for t in tables
        ]
    )
    return base.hstack(pl.DataFrame(stacked.mean(axis=0), schema=list(LEVELS), orient="row"))


def build_month(*, month: str, sites: pl.DataFrame) -> None:
    """Write one month's `refs_{month}.parquet` and `flags_{month}.parquet`."""
    capacity = dict(zip(sites["primary"].to_list(), sites["capacity_mw"].to_list(), strict=True))
    rows = pl.read_parquet(E511 / "scores" / f"rows_{month}.parquet", columns=KEYS)
    parts = [rows]
    reference = pl.read_parquet(E454 / "K2-S1" / f"{month}_s0_w-1.parquet")
    actual = reference.select(*KEYS, "actual")
    for arm, cells in CELLS_454.items():
        name = "K2-S1-3cell" if arm == "K2-S1" else arm
        loaded = [read_454(arm=arm, month=month, cell=c) for c in cells]
        parts.append(score_cells(rows=rows, cells=loaded, name=name, capacity=capacity))
        if len(loaded) > 1:
            for c, one in zip(cells, loaded, strict=True):
                parts.append(
                    score_cells(rows=rows, cells=[one], name=f"{arm}-c{c}", capacity=capacity)
                )
    seeds = [read_seed(month=month, seed=s, sites=sites, actual=actual) for s in (0, 1)]
    for s, table in enumerate(seeds):
        parts.append(score_cells(rows=rows, cells=[table], name=f"L-1d-seed{s}", capacity=capacity))
    parts.append(
        score_cells(
            rows=rows, cells=[pooled_seeds(tables=seeds)], name="L-1d-454rows", capacity=capacity
        )
    )
    refs = pl.concat(parts, how="horizontal")
    refs.write_parquet(WORK / f"refs_{month}.parquet")
    flags = (
        reference.select(*KEYS, "lag_real", "win_real", "tgt_real")
        .join(rows, on=KEYS, how="semi")
        .unique(KEYS)
    )
    flags.write_parquet(WORK / f"flags_{month}.parquet")
    print(month, "rows", rows.height, "flags", flags.height, flush=True)


def main() -> None:
    """Write the reference scores of every month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    sites = pl.read_csv(DATA / "site_map.csv").with_columns(pl.col("primary").cast(pl.Int16))
    for month in MONTHS:
        build_month(month=month, sites=sites)


if __name__ == "__main__":
    main()

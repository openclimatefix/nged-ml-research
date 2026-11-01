"""Put every candidate and reference of experiment 470 on experiment 454's rows.

Rows are experiment 454's scored rows (`primary`, `origin`, `target` on experiment 404's grid), which
are a subset of experiment 425's rows. Each arm is written to
`work/predictions/<arm>/<month>_c<cell>.parquet` with the keys and the 11 quantiles in capacity
fractions, raw (the scorer applies the 1.15 widening). An arm with several cells (seeds or seed pairs)
is scored cell by cell and averaged by row.

Arms: `B-mix`, `K2-S1`, `S1-ens3` (copied from experiment 454); `A-pool` (arm A of experiment 425,
five-seed quantile average); `A-gam-in` (experiment 457's `Gin`, five-seed quantile average, the 14
Primaries that have a GAM); `K2-425A` (2 arm A seeds plus 2 lagless heads, density mixture),
`K2-425A+S1` (1 arm A seed, 1 S1 seed, and 2 lagless seeds), `A-ens2` (2 arm A seeds); and the
references `BG2`, `manual_heuristic`, `climatology`, and `XGB-monthly` (MW divided by capacity).

```bash
source .e470env.sh
uv run --no-sync python ml-experiments/00470-*/e00470_build.py --month 2025-10
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00448_refs as refs448  # ty: ignore[unresolved-import]
import e00454_build as build454  # ty: ignore[unresolved-import]
import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

MONTHS: Final[tuple[str, ...]] = build454.MONTHS
KEYS: Final[list[str]] = ["primary", "origin", "target"]
LEVELS: Final[tuple[str, ...]] = score.LEVEL_NAMES
E425: Final[Path] = heads470.E425
E457: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00457-gam-stage-1-forecast-as-an-input-to-the-pooled-mixture-density-network"
    / "work"
    / "seed_preds"
)
POOL_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
COPIED: Final[tuple[str, ...]] = ("B-mix", "K2-S1", "S1-ens3")
type Head = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def prediction_path(*, arm: str, month: str, cell: int) -> Path:
    """Return where an arm's cell is stored."""
    return heads470.WORK_DIR / "predictions" / arm / f"{month}_c{cell}.parquet"


def write_arm(*, arm: str, month: str, cell: int, frame: pl.DataFrame) -> None:
    """Write an arm's keys and quantiles atomically."""
    target = prediction_path(arm=arm, month=month, cell=cell)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.select(*KEYS, *LEVELS).write_parquet(temporary)
    temporary.rename(target)


def site_map() -> pl.DataFrame:
    """Return the Primary index, time_series_id, and capacity in MW of the 16 Primaries."""
    return refs448.sites_table().select(
        pl.col("primary").cast(pl.Int16), "time_series_id", "capacity_mw"
    )


def grid_keys(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add experiment 404's `origin` and `target` grid indices to a (init, valid) keyed frame."""
    grid = lambda c: ((pl.col(c).dt.epoch("us") - heads470.GRID_START_US) // heads470.STEP_US).cast(  # noqa: E731
        pl.Int32
    )
    return frame.with_columns(origin=grid("power_fcst_init_time"), target=grid("valid_time"))


def on_rows(*, rows: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `frame`'s quantiles left-joined onto the base rows' keys (null where missing)."""
    return rows.select(*KEYS).join(
        frame.select(*KEYS, *LEVELS), on=KEYS, how="left", maintain_order="left"
    )


def pool_stored(
    *, folder: Path, month: str, sites: pl.DataFrame, seeds: tuple[int, ...] = POOL_SEEDS
) -> pl.DataFrame:
    """Return the quantile average over `seeds` of an experiment 425-layout arm, keyed on the grid."""
    parts = [
        pl.read_parquet(folder / f"{month}_s{s}.parquet").join(
            pl.read_parquet(folder / f"{month}_s{seeds[0]}.parquet").select(
                "time_series_id", "power_fcst_init_time", "valid_time"
            ),
            on=["time_series_id", "power_fcst_init_time", "valid_time"],
            how="semi",
        )
        for s in seeds
    ]
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    base = parts[0].select(keys).sort(keys)
    stacked = np.stack(
        [
            base.join(p, on=keys, how="left", maintain_order="left").select(list(LEVELS)).to_numpy()
            for p in parts
        ]
    )
    mean = stacked.astype(np.float64).mean(axis=0)
    frame = base.hstack(pl.DataFrame(mean, schema=list(LEVELS), orient="row"))
    return grid_keys(frame=frame.join(sites.select("time_series_id", "primary"), on="time_series_id"))


def a_heads(*, month: str, keys: dict[str, np.ndarray]) -> dict[int, Head]:
    """Return arm A's heads (seeds 0 to 2) aligned to the 454 row order."""
    base = pl.DataFrame({k: keys[k].astype(np.int64) for k in KEYS}).with_columns(
        pl.col("primary").cast(pl.Int16), pl.col("origin").cast(pl.Int32), pl.col("target").cast(pl.Int32)
    )
    out: dict[int, Head] = {}
    for seed in build454.CELL_SEEDS:
        frame = pl.read_parquet(heads470.head_path(month=month, seed=seed))
        joined = base.join(frame, on=KEYS, how="left", maintain_order="left")
        assert joined["lw0"].null_count() == 0, f"arm A s{seed} {month} misses 454 rows"
        t = torch.tensor(joined.select([f"{n}{i}" for n in ("lw", "mu", "sd") for i in range(3)]).to_numpy())
        out[seed] = (t[:, :3], t[:, 3:6], t[:, 6:9])
    return out


def mixtures(*, heads: dict[str, dict[int, Head]], a: dict[int, Head]) -> dict[str, list[list[Head]]]:
    """Return the cells of the arm A mixtures (equal weights); cell `c` of `K2-425A+S1` uses seed c."""
    s, ll = heads["S"], heads["L"]
    pairs = build454.PAIRS
    return {
        "A-ens2": [[a[i], a[j]] for i, j in pairs],
        "K2-425A": [[a[i], a[j], ll[i], ll[j]] for i, j in pairs],
        "K2-425A+S1": [[a[c], s[c], ll[c], ll[(c + 1) % 3]] for c in build454.CELL_SEEDS],
    }


def build_singles(*, month: str) -> None:
    """Write arm A's five single seeds, which give the minimum detectable effect."""
    sites = site_map()
    heads, keys = build454.load_heads(month=month)
    del heads
    rows = pl.DataFrame({k: keys[k] for k in KEYS})
    for seed in POOL_SEEDS:
        frame = pool_stored(folder=E425 / "seed_preds" / "A", month=month, sites=sites, seeds=(seed,))
        write_arm(arm=f"A-s{seed}", month=month, cell=0, frame=on_rows(rows=rows, frame=frame))


def build_month(*, month: str) -> None:
    """Write every arm of a month on experiment 454's rows."""
    sites = site_map()
    heads, keys = build454.load_heads(month=month)
    rows = pl.DataFrame({k: keys[k] for k in KEYS})
    a = a_heads(month=month, keys=keys)
    # Experiment 454's own constructs, copied unchanged.
    for arm in COPIED:
        cells = score454.construct_cells()[arm]
        for cell in range(cells):
            stored = pl.read_parquet(build454.prediction_path(arm=arm, month=month, cell=cell))
            write_arm(arm=arm, month=month, cell=cell, frame=on_rows(rows=rows, frame=stored))
    # Mixtures that include arm A.
    for arm, cells in mixtures(heads=heads, a=a).items():
        for cell, members in enumerate(cells):
            q = build454.mix_equal(heads=members)
            frame = rows.hstack(pl.DataFrame(q, schema=list(LEVELS), orient="row"))
            write_arm(arm=arm, month=month, cell=cell, frame=frame)
    # Stored five-seed pools.
    write_arm(arm="A-pool", month=month, cell=0, frame=on_rows(
        rows=rows, frame=pool_stored(folder=E425 / "seed_preds" / "A", month=month, sites=sites)))  # fmt: skip
    write_arm(arm="A-gam-in", month=month, cell=0, frame=on_rows(
        rows=rows, frame=pool_stored(folder=E457 / "Gin", month=month, sites=sites)))  # fmt: skip
    # References (MW over capacity).
    series = adapter.series_table()
    for name, frame in refs448.reference_frames(series=series).items():
        if name == "XGB-monthly-w115":
            continue
        f = grid_keys(frame=frame).join(sites, on="time_series_id")
        f = f.with_columns(*[(pl.col(c) / pl.col("capacity_mw")).alias(c) for c in LEVELS])
        write_arm(arm=name, month=month, cell=0, frame=on_rows(rows=rows, frame=f))
    print(f"{month}: wrote arms on {rows.height} rows", flush=True)


def main() -> None:
    """Build one month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=MONTHS)
    parser.add_argument("--singles-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    _ = models
    if args.singles_only:
        build_singles(month=args.month)
    else:
        build_month(month=args.month)


if __name__ == "__main__":
    main()

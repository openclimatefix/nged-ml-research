# ruff: noqa: E501, PERF401, B026
"""Build one per-month table of raw stored quantiles for every arm experiment 523 compares.

Nothing is fitted. Each month's table is keyed on experiment 511's scored rows
(`primary`, `origin`, `target`, with `lead_hours`, `group`, `cls`, `main16`) and holds `actual` and the
eleven raw (unwidened) quantiles of:

- `L1d`: experiment 511's L-1d, the quantile average of its two seeds (as in experiment 521);
- `K2c0` to `K2c2`: experiment 454's K2-S1, one column set per cell;
- `LLc0` to `LLc2`: experiment 454's monthly lag-free heads (ensemble only, not pretrained);
- `PTs0` to `PTs2`: experiment 448's L0PT (ERA5-and-CAMS pretrained, then fine-tuned on ensemble rows,
  world 0), one per seed;
- `L0s0` to `L0s2`: experiment 448's L0 (the same lag-free network and fine-tuning recipe, no
  pretraining), one per seed.

A row an arm does not hold is NaN for that arm. Experiment 448 holds no 2025-12 fold, so `PTs*` is
NaN there. Every column is `{arm}__{level}`.

```bash
source /tmp/e523_env.sh   # DATA_PATH_INTERNAL, PYTHONPATH of the ml-experiments folders
uv run --no-sync python ml-experiments/00523-*/e00523_panel.py
```
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00521_refs as refs  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00523-lead-awareness-of-the-lag-free-network-widening-and-pretraining-by-lead"
)
WORK: Final[Path] = DATA / "work"
E448: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00448-s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool"
    / "work"
    / "predictions"
)
KEYS: Final[list[str]] = refs.KEYS
LEVELS: Final[tuple[str, ...]] = refs.LEVELS
SITE_MAP: Final[Path] = refs.DATA / "site_map.csv"
META: Final[tuple[str, ...]] = ("lead_hours", "group", "cls", "tgt_real", "main16", "cold", "xmas")


def panel_path(*, month: str) -> Path:
    """Return the month's panel file."""
    return WORK / f"panel_{month}.parquet"


def aligned(*, rows: pl.DataFrame, table: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return `table`'s quantiles on `rows`' keys, columns `{name}__{level}`, NaN where absent."""
    joined = rows.select(*KEYS).join(
        table.select(*KEYS, *LEVELS), on=KEYS, how="left", maintain_order="left"
    )
    return joined.select(
        [pl.col(q).cast(pl.Float32).fill_null(float("nan")).alias(f"{name}__{q}") for q in LEVELS]
    )


def read_448(*, arm: str, month: str, seed: int) -> pl.DataFrame | None:
    """Return experiment 448's L0PT seed table on the grid keys, or None when the fold is absent."""
    path = E448 / arm / f"{month}_s{seed}_w0.parquet"
    if not path.exists():
        return None
    frame = pl.read_parquet(path)
    return frame.select(
        pl.col("primary").cast(pl.Int16),
        origin=((pl.col("origin_us") - refs.GRID_START_US) // refs.STEP_US).cast(pl.Int32),
        target=((pl.col("valid_us") - refs.GRID_START_US) // refs.STEP_US).cast(pl.Int32),
        *LEVELS,
    )


def aligned_448(
    *, rows: pl.DataFrame, arm: str, prefix: str, month: str, seed: int
) -> pl.DataFrame:
    """Return an experiment 448 arm's seed on `rows`' keys, all NaN when 448 stored no such fold."""
    table = read_448(arm=arm, month=month, seed=seed)
    if table is None:
        table = (
            rows.head(0)
            .select(*KEYS)
            .with_columns([pl.lit(None, dtype=pl.Float32).alias(q) for q in LEVELS])
        )
    return aligned(rows=rows, table=table, name=f"{prefix}{seed}")


def build_month(*, month: str, sites: pl.DataFrame) -> None:
    """Write one month's panel."""
    rows = pl.read_parquet(refs.E511 / "scores" / f"rows_{month}.parquet", columns=[*KEYS, *META])
    reference = pl.read_parquet(refs.E454 / "K2-S1" / f"{month}_s0_w-1.parquet")
    actual = reference.select(*KEYS, "actual")
    parts = [
        rows,
        rows.select(*KEYS)
        .join(actual, on=KEYS, how="left", maintain_order="left")
        .select("actual"),
    ]
    seeds = [refs.read_seed(month=month, seed=s, sites=sites, actual=actual) for s in (0, 1)]
    parts.append(aligned(rows=rows, table=refs.pooled_seeds(tables=seeds), name="L1d"))
    for arm, prefix in (("K2-S1", "K2c"), ("LL", "LLc")):
        for cell in (0, 1, 2):
            parts.append(
                aligned(
                    rows=rows,
                    table=refs.read_454(arm=arm, month=month, cell=cell),
                    name=f"{prefix}{cell}",
                )
            )
    for arm, prefix in (("L0PT", "PTs"), ("L0", "L0s")):
        for seed in (0, 1, 2):
            parts.append(aligned_448(rows=rows, arm=arm, prefix=prefix, month=month, seed=seed))
    out = pl.concat(parts, how="horizontal").with_columns(
        month=pl.lit(month),
        day=(pl.col("origin") // 48).cast(pl.Int32),
        half_hour=(pl.col("target") % 48).cast(pl.Int32),
    )
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = panel_path(month=month).with_suffix(".tmp")
    out.write_parquet(tmp)
    tmp.rename(panel_path(month=month))
    held = {
        arm: int(np.isfinite(out[f"{arm}__q50"].to_numpy()).sum())
        for arm in ("L1d", "K2c0", "LLc0", "PTs0", "PTs1", "PTs2", "L0s0", "L0s1", "L0s2")
    }
    print(f"{month}: {out.height} rows, finite q50 per arm {held}", flush=True)


def main() -> None:
    """Build all five months."""
    assert len(common.LEVEL_COLUMNS) == 11
    sites = pl.read_csv(SITE_MAP)
    for month in refs.MONTHS:
        build_month(month=month, sites=sites)


if __name__ == "__main__":
    main()

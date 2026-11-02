# ruff: noqa: E501
"""Report-only slices of experiment 510: week of the month, Christmas week, and January.

Reads the row-level table written by `e00510_score.py rows`. Slices (rowset `main16`, the eight and
the signed-MW six on rows with no labelled event):

- `week_of_month`: the scored refit week (origin day 1-7, 8-14, 15-21, 22-28, 29+), all five folds;
  tests whether the weekly-over-monthly gain grows through the month (week 1 shares one fit).
- `christmas`: target half-hours on 22 to 31 December 2025 (fold 2025-12).
- `jan_1_14` and `jan_25_31`: forecast origins on 1 to 14 and 25 to 31 January 2026 (fold 2026-01).

Each cell is the mean CRPS (points of effective capacity) per arm and the contrast of two arms with
an origin-day block bootstrap (2,000 resamples, 95%, no multiplicity correction: descriptive).

```bash
source <paths>; uv run --no-sync python ml-experiments/00510-*/e00510_slices.py
```
"""

from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00510-lagless-demand-network-weekly-finetune-and-recency-weights/work/scores"
)
ARMS: Final[tuple[str, ...]] = ("L-M", "L-W", "L-Ww", "L-Ww-D", "L-Ww-O", "K2-S1", "A-pool")
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("L-W", "L-M"), ("L-Ww", "L-W"), ("L-Ww", "K2-S1"), ("L-W", "K2-S1"), ("L-Ww-D", "L-Ww"), ("L-Ww-O", "L-Ww"),
)  # fmt: skip
GROUPS: Final[dict[str, pl.Expr]] = {
    "eight": pl.col("group") == "mva_eight",
    "six": (pl.col("group") == "signed_six") & (pl.col("cls") == "clean"),
}
STEP_PER_DAY: Final[int] = 48


def load() -> pl.DataFrame:
    """Return the row table of all months with the origin and target dates."""
    frames = [pl.read_parquet(p) for p in sorted(SCORES.glob("rows_*.parquet"))]
    t = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("main16"))
    start = common.GRID_START
    return t.with_columns(
        origin_date=(pl.lit(start) + pl.duration(days=pl.col("origin") // STEP_PER_DAY)).dt.date(),
        target_date=(pl.lit(start) + pl.duration(days=pl.col("target") // STEP_PER_DAY)).dt.date(),
    )


def boot(
    *, frame: pl.DataFrame, a: str, b: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the mean CRPS difference a - b and its origin-day block-bootstrap 95% interval."""
    d = (
        frame.select("origin_date", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps"))
        .drop_nulls()
        .drop_nans()
    )
    by_day = d.group_by("origin_date").agg(s=pl.col("d").sum(), n=pl.len())
    s, n = by_day["s"].to_numpy(), by_day["n"].to_numpy()
    est = float(s.sum() / n.sum())
    draws = np.empty(2_000)
    for i in range(len(draws)):
        pick = rng.integers(0, len(s), len(s))
        draws[i] = s[pick].sum() / n[pick].sum()
    return est, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def slices(*, t: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the named slices."""
    day = pl.col("origin_date").dt.day()
    week = (
        pl.when(day <= 7)
        .then(1)
        .when(day <= 14)
        .then(2)
        .when(day <= 21)
        .then(3)
        .when(day <= 28)
        .then(4)
        .otherwise(5)
    )
    t = t.with_columns(week_of_month=week)
    out = {f"week_{w}": t.filter(pl.col("week_of_month") == w) for w in range(1, 6)}
    out["christmas"] = t.filter((pl.col("month") == "2025-12") & pl.col("xmas"))
    jan = t.filter(pl.col("month") == "2026-01")
    out["jan_1_14"] = jan.filter(day <= 14)
    out["jan_25_31"] = jan.filter(day >= 25)
    out["jan_15_24"] = jan.filter((day > 14) & (day < 25))
    return out


def main() -> None:
    """Write `slices.csv` and print it."""
    t = load()
    rng = np.random.default_rng(0)
    records = []
    for name, frame in slices(t=t).items():
        for gname, cond in GROUPS.items():
            part = frame.filter(cond)
            if part.height == 0:
                continue
            for arm in ARMS:
                records.append(  # noqa: PERF401
                    {
                        "slice": name,
                        "group": gname,
                        "kind": "mean",
                        "arm": arm,
                        "n": part.height,
                        "value": part[f"{arm}__crps"].mean(),
                    }
                )
            for a, b in CONTRASTS:
                est, lo, hi = boot(frame=part, a=a, b=b, rng=rng)
                records.append(
                    {
                        "slice": name,
                        "group": gname,
                        "kind": f"{a} - {b}",
                        "arm": "",
                        "n": part.height,
                        "value": est,
                        "low": lo,
                        "high": hi,
                    }
                )
    table = pl.DataFrame(records, infer_schema_length=None)
    table.write_csv(SCORES / "slices.csv")
    with pl.Config(tbl_rows=300, tbl_width_chars=160):
        print(table.with_columns(pl.col(pl.Float64).round(3)))
    _ = timedelta


if __name__ == "__main__":
    main()

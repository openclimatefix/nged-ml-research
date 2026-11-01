"""Score the constructs of experiment 454 by row class and group, with tails and widening.

Reads the forecasts `e00454_build.py` wrote (real world only) and writes
`work/scores/cells.parquet`: one row per construct, month, cell, variant, basis, group, and row
class, with the row count, the mean score, and coverage and pinball loss at q01, q05, q95, and q99.

- **Row classes** follow experiment 404 with experiment 455's reconciled harm-row definition:
  `event`
  (target inside a padded real event or in the 96 half-hours after it), `harm` (target outside
  those, but a day window of the lags inside a padded event), and `clean` (neither: rows with no
  labelled event, which may still hold unlabelled events).
- **Variants**: `raw`; `w1.05`, `w1.10`, `w1.15` (the four tail quantiles' distance from the median
  scaled); `P`, a table P fitted per construct and cell on earlier months' clean rows whose target
  precedes the test month's first forecast origin, one factor per group, lead band, and tail (so
  the first month has no `P`).
- **Bases**: `metered` (the stored actual) and, for event rows only, `nra` (the actual minus the
  normal-running-arrangement correction of experiment 435 at depth factor 1.0), on rows whose
  target lies inside a padded real event (not the 96 half-hours after it, which have no correction).
- **Rows** are limited to leads from 0 to 240 hours.

```bash
<PYTHONPATH as e00454_build.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00454-*/e00454_score.py cells
```
"""

import argparse
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import e00454_build as build  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import TAIL_LEVELS, lead_band_expr, widen  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORE_DIR: Final = build.WORK_DIR / "scores"
NRA_FILE: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
    / "nra_correction_share.npz"
)
WIDEN_SWEEP: Final[tuple[float, ...]] = (1.05, 1.10, 1.15)
LEVELS: Final[tuple[str, ...]] = score.LEVEL_NAMES


def construct_cells() -> dict[str, int]:
    """Return the number of cells (seeds or seed pairs) of each construct."""
    heads = {k: dict.fromkeys(build.SEEDS, (None, None, None)) for k in "ASL"}
    return {arm: len(cells) for arm, cells in build.constructs(heads=heads).items()}  # ty: ignore[invalid-argument-type]


def month_base(*, month: str) -> pl.DataFrame:
    """Return a month's row keys, actuals, row class, group, lead band, and NRA correction."""
    frame = pl.read_parquet(build.prediction_path(arm="B-mix", month=month, cell=0))
    sites = fit.prep.primaries()["site"].to_list()
    groups = report.group_of(sites=sites)
    nra = np.load(NRA_FILE)["f1.0"]
    primary = frame["primary"].to_numpy().astype(np.int64)
    target = frame["target"].to_numpy().astype(np.int64)
    frame = frame.with_columns(
        group=pl.Series(groups[primary]),
        nra=pl.Series(nra[primary, target].astype(np.float64)),
        lead_band=lead_band_expr(column="lead_hours"),
        cls=pl.when(pl.col("tgt_real") | pl.col("tgt_post"))
        .then(pl.lit("event"))
        .when(pl.col("win_real"))
        .then(pl.lit("harm"))
        .otherwise(pl.lit("clean")),
    )
    return frame.filter(pl.col("lead_band").is_not_null() & pl.col("actual").is_finite()).select(
        "primary", "origin", "target", "lead_hours", "actual", "nra", "group", "lead_band", "cls",
        "tgt_real",
    )  # fmt: skip


def load_cell(*, base: pl.DataFrame, arm: str, month: str, cell: int) -> pl.DataFrame:
    """Return a construct's quantiles joined (by position) to a month's base rows."""
    frame = pl.read_parquet(build.prediction_path(arm=arm, month=month, cell=cell))
    keys = frame.select("primary", "origin", "target")
    joined = base.join(frame.select(*keys.columns, *LEVELS), on=list(keys.columns), how="left")
    assert joined.height == base.height, "rows differ"
    assert joined["q50"].null_count() == 0, "rows differ"
    return joined.with_columns(pl.col("actual", *LEVELS).cast(pl.Float64))


def summarise(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the row count, mean score, and tail coverage and pinball loss by group and class."""
    aggs: list[pl.Expr] = [
        pl.len().alias("n"),
        score.pinball_points(frame=frame).mean().alias("crps"),
    ]
    for name, tau in TAIL_LEVELS.items():
        error = pl.col("actual") - pl.col(name)
        aggs.append((pl.col("actual") <= pl.col(name)).mean().alias(f"cov_{name}"))
        aggs.append(
            pl.when(error >= 0)
            .then(tau * error)
            .otherwise((tau - 1.0) * error)
            .mean()
            .alias(f"pin_{name}")
        )
    return frame.group_by("group", "cls").agg(*aggs)


def score_arm_cell(*, arm: str, cell: int, bases: dict[str, pl.DataFrame]) -> list[pl.DataFrame]:
    """Return the summaries of every month, variant, and basis for one construct cell."""
    frames = {m: load_cell(base=b, arm=arm, month=m, cell=cell) for m, b in bases.items()}
    out = []
    for i, month in enumerate(build.MONTHS):
        frame = frames[month]
        variants = {"raw": frame}
        for factor in WIDEN_SWEEP:
            variants[f"w{factor:.2f}"] = widen(frame=frame, factor=factor)
        if i > 0:
            first_origin = int(frame["origin"].min())  # ty: ignore[invalid-argument-type]
            earlier = pl.concat([frames[m] for m in build.MONTHS[:i]]).filter(
                (pl.col("cls") == "clean") & (pl.col("target") < first_origin)
            )
            variants["P"] = prov.apply_factors(frame=frame, factors=prov.fit_factors(fit=earlier))
        for variant, data in variants.items():
            metered = summarise(frame=data).with_columns(basis=pl.lit("metered"))
            events = data.filter(pl.col("tgt_real")).with_columns(
                actual=pl.col("actual") - pl.col("nra")
            )
            nra = summarise(frame=events).with_columns(basis=pl.lit("nra"))
            out.append(
                pl.concat([metered, nra]).with_columns(
                    arm=pl.lit(arm), month=pl.lit(month), cell=pl.lit(cell), variant=pl.lit(variant)
                )
            )
    return out


def cells() -> None:
    """Score every construct cell and write the long table."""
    bases = {m: month_base(month=m) for m in build.MONTHS}
    for m, b in bases.items():
        counts = b.group_by("group", "cls").len().sort("group", "cls")
        print(m, {(r["group"], r["cls"]): r["len"] for r in counts.iter_rows(named=True)})
    parts: list[pl.DataFrame] = []
    for arm, n_cells in construct_cells().items():
        for cell in range(n_cells):
            parts.extend(score_arm_cell(arm=arm, cell=cell, bases=bases))
        print("scored", arm, flush=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(parts).write_parquet(SCORE_DIR / "cells.parquet")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cells",))
    parser.parse_args()
    cells()


if __name__ == "__main__":
    main()

"""Tail calibration of experiment 470's candidates: raw, widened, and after table P.

Reuses experiment 454's `score_arm_cell` (group-by-class summaries with coverage and pinball loss at
q01, q05, q95, and q99, the widening sweep 1.05, 1.10, 1.15, and table P of experiment 435 fitted on
earlier months' rows with no labelled event) with this experiment's arm loader. Rows are the
`gam14` set (the eight and the six, every candidate present), so every arm sees identical rows.
Writes `work/scores/tails_cells.parquet`.

```bash
source .e470env.sh
uv run --no-sync python ml-experiments/00470-*/e00470_tails.py
```
"""

import e00404_score as score  # ty: ignore[unresolved-import]
import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import polars as pl

ARMS = (*score470.CANDIDATES, *score470.CONTROLS)


def main() -> None:
    """Write the per-cell tail summaries."""
    series = score470.adapter.series_table()
    ctx = score470.prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    rows = score470.load_rows()
    bases = {}
    for month in score470.MONTHS:
        base = score470.month_context(month=month, sites=sites, ctx=ctx)
        keep = rows.filter((pl.col("month") == month) & pl.col("gam14")).select(build470.KEYS)
        bases[month] = base.join(keep, on=build470.KEYS, how="semi")
    # score454.load_cell reads this experiment's stored cells instead of experiment 454's.
    def load_cell(*, base: pl.DataFrame, arm: str, month: str, cell: int) -> pl.DataFrame:
        frame = pl.read_parquet(build470.prediction_path(arm=arm, month=month, cell=cell))
        joined = base.join(
            frame.select(*build470.KEYS, *score.LEVEL_NAMES), on=build470.KEYS, how="left"
        )
        assert joined.height == base.height and joined["q50"].null_count() == 0, "rows differ"
        return joined.with_columns(pl.col("actual", *score.LEVEL_NAMES).cast(pl.Float64))

    score454.load_cell = load_cell  # ty: ignore[invalid-assignment]
    parts: list[pl.DataFrame] = []
    for arm in ARMS:
        for cell in range(score470.arm_cells(arm=arm)):
            parts.extend(score454.score_arm_cell(arm=arm, cell=cell, bases=bases))
        print("scored", arm, flush=True)
    out = score470.SCORE_DIR / "tails_cells.parquet"
    pl.concat(parts).write_parquet(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

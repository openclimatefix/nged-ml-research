"""Print the checks of the audit triggers for experiment 439 (no fit).

- The error of the median forecast (mean absolute error of q50, points of capacity) of S1 and
  R-inj on clean rows, per group, and its difference: trigger 3 compares its sign with the CRPS
  contrast. The stored forecasts hold 11 quantiles and no mean, so the mean forecast is taken as the
  average of the 11 quantiles (a proxy).
- The median minus that mean forecast per Primary, on clean rows.
- S1's and R-inj's tail pinball loss and coverage at q01, q05, q95, and q99 per group and row set.

```bash
PYTHONPATH=<the folders of e00439_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00439-*/e00439_checks.py
```
"""

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00439_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

POINTS = 100.0
SEEDS = (0, 1, 2, 3, 4, 5)
MONTHS = ("2026-06", "2026-01", "2025-10", "2026-03")


def main() -> None:
    """Print the median-error contrast and the per-Primary median minus mean."""
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    rows = []
    for arm in run.ARMS:
        for month in MONTHS:
            f = score.read_arm(arm=arm, month=month, world=0, seeds=SEEDS, factor=score.WIDENING)
            keep = score.row_mask(frame=f, rule=score.ROW_SETS["clean"][1])
            q = f.select(list(pr.LEVEL_NAMES)).to_numpy()
            sites = np.array(data.sites())[f["primary"].to_numpy()]
            frame = pl.DataFrame(
                {
                    "arm": arm, "site": sites, "keep": keep,
                    "abs_median": np.abs(f["actual"].to_numpy() - f["q50"].to_numpy()) * POINTS,
                    "median_minus_mean": (f["q50"].to_numpy() - q.mean(axis=1)) * POINTS,
                }
            ).filter(pl.col("keep") & pl.col("abs_median").is_finite())  # fmt: skip
            rows.append(frame)
    t = pl.concat(rows)
    t = t.with_columns(
        group=pl.when(pl.col("site").is_in(list(score.SIGNED_MW_LABELS)))
        .then(pl.lit("signed-MW six"))
        .when(pl.col("site").is_in(list(score.MVA_ONLY_LABELS)))
        .then(pl.lit("MVA-only eight"))
        .otherwise(pl.lit("D15 and D18"))
    )
    with pl.Config(tbl_rows=60):
        mae = t.group_by("group", "arm").agg(pl.col("abs_median").mean().alias("mae_q50"))
        print(mae.sort("group", "arm"))
        print(
            t.group_by("arm", "site")
            .agg(pl.col("median_minus_mean").mean())
            .pivot(on="arm", index="site", values="median_minus_mean")
            .sort("site")
        )


if __name__ == "__main__":
    main()

# ruff: noqa: E501
r"""Checks bullets for experiment 497: median-forecast error and median minus quantile-mean forecast.

For H0, WO, and RO (seeds averaged, 1.15 widening as the scorer), per group, on the gate rows
(outside 22 December to 2 January) and the Christmas rows (22 to 31 December 2025): the mean
absolute error of the median forecast in points of capacity (the error of the point forecast beside
each CRPS contrast), and the mean of (q50 minus the mean of the 11 quantiles) in points (the
shift the spread puts into the forecast).

```bash
source .xmasenv.sh
uv run --no-sync python ml-experiments/00497-*/e00497_checks.py
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as s470  # ty: ignore[unresolved-import]
import e00485_regime as regime  # ty: ignore[unresolved-import]
import e00497_score as score497  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def main() -> None:
    """Print the checks table."""
    score497.configure()
    ctx = prep425.cleaned_scoring_context(series=adapter.series_table())
    sites = build470.site_map()
    records = []
    for month in sorted(score497.score485.MONTHS):
        base = s470.month_context(month=month, sites=sites, ctx=ctx)
        day = base["valid_time"].dt.date().dt.epoch("d").to_numpy().astype(np.int64)
        window = regime.in_window_day(day=day)
        xmas = window & (month == "2025-12") & (base["valid_time"].dt.month().to_numpy() == 12)
        for arm in score497.LOGICAL_ARMS:
            maes, shifts = [], []
            for seed in score497.SEEDS:
                stored = score497.score485.seed_frame(arm=arm, month=month, seed=seed, sites=sites)
                frame = base.select(*s470.KEYS, "actual").join(
                    stored.select(*s470.KEYS, *s470.LEVELS),
                    on=s470.KEYS,
                    how="left",
                    maintain_order="left",
                )
                frame = s470.widen_arm(arm=arm, frame=frame, factor=s470.HEADLINE)
                q = np.sort(frame.select(list(s470.LEVELS)).to_numpy().astype(np.float64), axis=1)
                q50 = frame["q50"].to_numpy().astype(np.float64)
                y = frame["actual"].to_numpy().astype(np.float64)
                maes.append(s470.POINTS * np.abs(q50 - y))
                shifts.append(s470.POINTS * (q50 - q.mean(axis=1)))
            mae, shift = np.mean(maes, axis=0), np.mean(shifts, axis=0)
            ok = np.isfinite(mae)
            for slice_name, mask in (("gate", ~window), ("xmas", xmas)):
                for group in ("mva_eight", "signed_six"):
                    m = mask & ok & (base["group"] == group).to_numpy()
                    records.append(
                        {"arm": arm, "slice": slice_name, "group": group, "n": int(m.sum()),
                         "mae": float(mae[m].sum()), "shift": float(shift[m].sum())}
                    )  # fmt: skip
    table = (
        pl.DataFrame(records)
        .group_by("arm", "slice", "group")
        .agg(pl.col("n").sum(), pl.col("mae").sum(), pl.col("shift").sum())
        .with_columns(mae=pl.col("mae") / pl.col("n"), shift=pl.col("shift") / pl.col("n"))
        .sort("slice", "group", "arm")
    )
    table.write_csv(score497.SCORE_DIR / "checks.csv")
    with pl.Config(tbl_rows=50):
        print(table.with_columns(pl.col("mae", "shift").round(3)))


if __name__ == "__main__":
    main()

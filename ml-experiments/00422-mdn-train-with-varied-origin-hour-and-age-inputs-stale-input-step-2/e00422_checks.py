"""Print the checks of experiment 422 (no fit).

The checks are the error of the median forecast beside the CRPS contrasts, and the median minus
the mean forecast per group.

```bash
PYTHONPATH=<the experiment folders, as in e00422_score.py> \
uv run --no-sync python ml-experiments/00422-*/e00422_checks.py
```
"""

from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00410_stale_predict as stale  # ty: ignore[unresolved-import]
import e00410_stale_score as score410  # ty: ignore[unresolved-import]
import e00422_score as score422  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SOURCES: Final[tuple[str, ...]] = ("T0", "T1", "T2", "T2n", "T3", "T4")
ARMS: Final[tuple[str, ...]] = ("P12", "P24", "W24", "W24_P24", "DROP")
SHOWN: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight")


def main() -> None:
    """Print MAE of the median (fresh and stale) and median minus mean per group."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(stale.FLAGGED_KEYS).select("time_series_id", "valid_time")
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    months = list(prep.SCREEN_MONTHS)
    label_of = {r[0]: r[1] for r in series.iter_rows()}
    masses = np.asarray(common.ATOM_MASSES)
    rows = []
    for source in SOURCES:
        fresh = score422.forecast(source=source, test_arm="FRESH", months=months, series=series)
        for arm in ARMS:
            stale_frame = score422.forecast(
                source=source, test_arm=arm, months=months, series=series
            )
            _table, scored, _ = s381.score_arms(
                arms={"fresh": fresh, "stale": stale_frame}, series=series, ctx=ctx
            )
            for group in SHOWN:
                labels = score410.GROUPS[group]
                rec = {"source": source, "arm": arm, "group": group}
                for tag in ("fresh", "stale"):
                    rec[f"mae_median_{tag}"] = score410.group_mean(
                        scored=scored[tag], metric="mae_median", labels=labels
                    )
                rows.append(rec)
        # median minus mean of the stored fresh forecast, per group, in points of capacity
        frame = score422.seed_average(
            folder=score422.predict422.OUT_DIR / "preds" / source / "FRESH",
            months=months,
            seeds=score422.SEEDS,
        )
        quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy()
        mean = (quantiles * masses).sum(axis=1) / masses.sum()
        gap = frame.with_columns(
            gap=pl.Series((frame["q50"].to_numpy() - mean) * 100.0),
            label=pl.col("time_series_id").replace_strict(label_of),
        )
        for group in SHOWN:
            value = gap.filter(pl.col("label").is_in(list(score410.GROUPS[group])))["gap"].mean()
            print(f"{source} median minus mean forecast, {group}: {value:+.3f} points")
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()

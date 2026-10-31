"""Score each training arm's FRESH forecast per seed, to report the seed spread (no fit).

The spread is the standard deviation across the three seeds of the group CRPS at days 1 to 10 of a
single seed's forecast (before averaging seeds in quantile space). Twice the standard deviation of
the difference of two such scores is a rough minimum detectable effect for a one-seed contrast.

```bash
PYTHONPATH=<the experiment folders, as in e00422_score.py> \
uv run --no-sync python ml-experiments/00422-*/e00422_seed_spread.py
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00410_stale_predict as stale  # ty: ignore[unresolved-import]
import e00410_stale_score as score410  # ty: ignore[unresolved-import]
import e00422_score as score422  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def main() -> None:
    """Print each arm's per-seed FRESH CRPS per group and the spread across seeds."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(stale.FLAGGED_KEYS).select("time_series_id", "valid_time")
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    months = list(prep.SCREEN_MONTHS)
    for source in ("T0", "T1", "T2n", "T3"):
        scores: dict[str, list[float]] = {"signed-MW six": [], "MVA-only eight": []}
        for seed in score422.SEEDS:
            stored = score422.seed_average(
                folder=score422.predict422.OUT_DIR / "preds" / source / "FRESH",
                months=months,
                seeds=(seed,),
            )
            frame = score410.widen(
                frame=adapter.forecast_frame(stored=stored, series=series, unit="fraction")
            )
            _table, scored, _ = s381.score_arms(arms={"x": frame}, series=series, ctx=ctx)
            for group, values in scores.items():
                values.append(
                    score410.group_mean(
                        scored=scored["x"], metric="crps", labels=score410.GROUPS[group]
                    )
                )
        for group, values in scores.items():
            print(
                f"{source} {group}: seeds {np.round(values, 3).tolist()}, "
                f"sd {np.std(values, ddof=1):.3f}"
            )


if __name__ == "__main__":
    main()

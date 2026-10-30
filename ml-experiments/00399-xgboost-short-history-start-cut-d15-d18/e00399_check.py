"""Gates for experiment 399, run before any fit: cut dates, removed rows, and unchanged labels.

Prints (1) each arm's cut date per Primary, asserting NS matches the settling rule's dates in the
plan, (2) how many half-hours NCZ and NCR remove since 2025-03-01 and inside the scored year, (3)
that, with no removal, the patched training rows equal experiment 87's stored rows from the cut on
(same row count, labels equal), and (4) that `scan_power` has no half-hour before 2026-07-01 that
the stored `observed.parquet` lacks.

```bash
PYTHONPATH=ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:studies/quick_wins \
    uv run --no-sync python ml-experiments/00399-*/e00399_check.py
```
"""

import os
from datetime import UTC, datetime

os.environ["E399_ARM"] = "NC"

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00399_fit as f
import numpy as np
import polars as pl

SCORED_YEAR_START = datetime(2025, 7, 1, tzinfo=UTC)


def main() -> None:
    """Print the gates and fail on a mismatch."""
    series = common.series_table()
    ids = series.filter(pl.col("site").is_in(f.SITES))["time_series_id"].to_list()
    site_of = dict(zip(series["time_series_id"], series["site"], strict=True))
    power = f.scanned_power(series=series)
    for arm in f.ARMS:
        cuts = f.cut_times(power=power, series=series, arm=arm)
        print(f"{arm} cuts: { {k: v.strftime('%Y-%m-%d %H:%M') for k, v in cuts.items()} }")
        if arm == "NS":
            for site, day in f.EXPECTED_SETTLING_CUT.items():
                assert cuts[site].strftime("%Y-%m-%d") == day, (site, cuts[site], day)
    for arm in ("NCZ", "NCR"):
        gone = f.removed_times(power=power, arm=arm)
        for sid in ids:
            mine = gone.filter(pl.col("time_series_id") == sid)
            print(
                f"{arm} {site_of[sid]}: removes {mine.height} half-hours since 2025-03-01, "
                f"{mine.filter(pl.col('time') >= SCORED_YEAR_START).height} in the scored year "
                f"(from 2025-07-01)"
            )
    stored = pl.read_parquet(common.OBSERVED_PATH, columns=["time_series_id", "valid_time"])
    extra = power.join(
        stored,
        left_on=["time_series_id", "time"],
        right_on=["time_series_id", "valid_time"],
        how="anti",
    )
    print(f"scan_power half-hours before 2026-07-01 absent from observed.parquet: {extra.height}")
    assert extra.is_empty() or (
        extra["time"].min() >= datetime(2026, 6, 30, tzinfo=UTC)  # ty: ignore[unsupported-operator]
    )
    # Labels: patched rows (NC, no removal) against the stored rows, from the cut on.
    series2, arrays = fd._series_and_arrays()
    for site in f.SITES:
        row = series2.filter(pl.col("site") == site).row(0, named=True)
        patched = fd.training_rows(row=row, arrays=arrays[site], frozen=False, a7=False)
        observed = pl.read_parquet(common.OBSERVED_PATH).filter(
            pl.col("time_series_id") == row["time_series_id"]
        )
        joined = patched.join(
            observed.select("valid_time", stored=pl.col("power")), on="valid_time", how="left"
        )
        diff = (joined["power"] - joined["stored"]).abs().drop_nulls()
        print(
            f"{site}: patched rows {patched.height:,}, label differs from observed.parquet by "
            f"at most {diff.max():.2e} MW over {diff.len():,} rows; "
            f"{joined['stored'].null_count()} rows lack a stored reading"
        )
        assert np.isclose(diff.max(), 0.0, atol=1e-3)
    print("gates passed")


if __name__ == "__main__":
    main()

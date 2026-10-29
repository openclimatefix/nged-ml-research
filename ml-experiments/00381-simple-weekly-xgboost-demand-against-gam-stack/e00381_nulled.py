"""Interval width and coverage of the weekly bundle where a power lag is missing (experiment 381).

A row is "lag-missing" when any of the three origin lags (the newest usable reading, the reading a
day earlier, and the reading a week earlier) is NaN because the telemetry has a gap. The script
prints, per group and for the lag-missing and lag-present rows at days 1 to 10 in the scored year,
the mean width of the 5%-to-95% interval (points of capacity), the coverage of that interval (the
nominal value is 0.90), the coverage below q05 and above q95, and the number of rows.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
    ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack \
    uv run --no-sync python ml-experiments/00381-*/e00381_nulled.py 'bundle_solar+sunny+hl1'
```
"""

import sys
from datetime import UTC, datetime

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00381_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SCORED_FIRST = datetime(2025, 7, 1, tzinfo=UTC)
MAX_LEAD_HOURS = 240.0
GROUPS = {
    "signed-MW": ("D5", "D9", "D10", "D11", "D14", "D16"),
    "MVA-only": ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
}


def main() -> None:
    """Print the width and coverage table."""
    arm = sys.argv[1]
    _, arrays = fd._series_and_arrays()
    pieces = []
    for site in [s for sites in GROUPS.values() for s in sites]:
        files = sorted((fit.ARM_DIR / "weekly" / arm).glob(f"{site}_s0_*.parquet"))
        frame = pl.concat([pl.read_parquet(f) for f in files]).filter(
            (pl.col("power_fcst_init_time") >= SCORED_FIRST)
            & (pl.col("valid_time") <= datetime(2026, 7, 1, tzinfo=UTC))
        )
        power = arrays[site].power
        target = np.asarray(common.grid_index(times=frame["valid_time"]))
        origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
        lags = common.origin_lags(power=power, target=target, origin=origin)
        actual = common.take(values=power, index=target)
        group = next(g for g, sites in GROUPS.items() if site in sites)
        pieces.append(
            frame.with_columns(
                lag_missing=pl.Series(np.isnan(lags).any(axis=1)),
                actual=pl.Series(actual).cast(pl.Float32),
                group=pl.lit(group),
                site=pl.lit(site),
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
            ).filter(pl.col("actual").is_finite() & (pl.col("lead_hours") <= MAX_LEAD_HOURS))
        )
    table = pl.concat(pieces)
    out = (
        table.group_by("group", "lag_missing")
        .agg(
            n=pl.len(),
            width_points=((pl.col("q95") - pl.col("q05")) * 100).mean(),
            coverage_90=((pl.col("actual") >= pl.col("q05")) & (pl.col("actual") <= pl.col("q95")))
            .cast(pl.Float64)
            .mean(),
            below_q05=(pl.col("actual") < pl.col("q05")).cast(pl.Float64).mean(),
            above_q95=(pl.col("actual") > pl.col("q95")).cast(pl.Float64).mean(),
        )
        .sort("group", "lag_missing")
    )
    pl.Config.set_tbl_cols(20)
    print(out)
    print(
        table.group_by("site", "lag_missing")
        .agg(n=pl.len())
        .pivot(on="lag_missing", index="site", values="n")
        .sort("site")
    )


if __name__ == "__main__":
    main()

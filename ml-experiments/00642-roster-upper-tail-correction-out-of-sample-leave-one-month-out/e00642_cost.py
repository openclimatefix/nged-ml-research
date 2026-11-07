"""Experiment 642: avoidable curtailment cost of each arm at the six signed-MW Primaries.

Uses `metrics.avoidable_curtailment_cost` (node-only mode, lead window 12 to 36 hours), with the
limit set from the readings before 2025-10-01, and sums the cost over the five held-out months.

```bash
E=ml-experiments
PYTHONPATH=$E/00377-demand-v1-first-run-feasibility:$E/00642-roster-upper-tail-correction-\
out-of-sample-leave-one-month-out \
    uv run --no-sync python $E/00642-*/e00642_cost.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00642-roster-upper-tail-correction-out-of-sample-leave-one-month-out"
)
SIX: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
SCORED_START: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
NETWORKS: Final[tuple[str, ...]] = ("K1", "Candidate", "K3")


def main() -> None:
    """Score the cost of every arm and write one table."""
    series = adapter.series_table().filter(pl.col("label").is_in(SIX))
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=OUT / "cache")
    flows, _ = adapter.actuals_and_context(
        cache=cache, series=series, first=datetime(2024, 4, 1, tzinfo=UTC), last=SCORED_START
    )
    flows = flows.rename({"actual": "flow_mw"}).filter(pl.col("flow_mw").is_finite())
    rows = []
    for network in NETWORKS:
        frame = pl.read_parquet(OUT / f"cost_quantiles_{network}.parquet").with_columns(
            pl.col("time_series_id").cast(series.schema["time_series_id"]),
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        )
        for month in sorted(frame["month"].unique().to_list()):
            this = frame.filter(pl.col("month") == month)
            arms = {
                a: this.filter(pl.col("arm") == a).select(
                    *KEYS, "actual", *metrics.QUANTILE_COLUMNS
                )
                for a in this["arm"].unique().to_list()
            }
            for fraction in FRACTIONS:
                out = metrics.avoidable_curtailment_cost(
                    forecasts=arms,
                    series=series,
                    flows=flows,
                    scored_start=SCORED_START,
                    fraction=fraction,
                )
                rows.append(
                    out.with_columns(
                        network=pl.lit(network), month=pl.lit(month), fraction=pl.lit(fraction)
                    )
                )
    table = pl.concat(rows)
    table.write_csv(OUT / "tables" / "curtailment_cost_by_month.csv")
    total = (
        table.group_by("network", "arm", "label", "fraction")
        .agg(
            pl.col("avoidable_cost_gbp").sum().alias("cost_gbp"),
            pl.col("n_node_half_hours").sum(),
            pl.col("status").first(),
        )
        .sort("network", "fraction", "label", "arm")
    )
    total.write_csv(OUT / "tables" / "curtailment_cost_total.csv")
    pooled = (
        total.group_by("network", "arm", "fraction")
        .agg(pl.col("cost_gbp").sum())
        .sort("network", "fraction", "arm")
    )
    pooled.write_csv(OUT / "tables" / "curtailment_cost_pooled.csv")
    with pl.Config(tbl_rows=100):
        print(pooled.pivot(on="arm", index=["network", "fraction"], values="cost_gbp"))


if __name__ == "__main__":
    main()

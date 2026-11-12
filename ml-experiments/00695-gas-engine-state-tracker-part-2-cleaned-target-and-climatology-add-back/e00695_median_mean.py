# ruff: noqa: E501
"""Experiment 695: the median-minus-mean-forecast check and the median error by tracker state.

    source ml-experiments/00524*/env.sh; export PYTHONPATH=src:$PYTHONPATH
    uv run --no-sync python ml-experiments/00695-*/e00695_median_mean.py

The networks emit 11 quantiles and no mean, so "the mean forecast" here is the mean of the 11
quantile values (levels 0.01 to 0.99). A spread that shifts the forecast would show as a gap
between that mean and the median (q50). Everything is in points of effective capacity. Nothing is
fitted. It reads arm A0's raw fits for all 16 Primaries, and the stored prediction sets for D8
(a0-d8, a1-d8, ah-d8) and for the host (w0-host, w1-host). The median error is the forecast median
minus the metered reading, over the scored rows.
"""

import e00695_score as sc
import polars as pl
from e00695_common import D8_CAPACITY_MVA, D8_SERIES, DATA_DIR, QUANTILE_COLUMNS, e694_flags
from studies.experiment_store import read_predictions


def main() -> None:
    """Print the check per Primary for A0 and per system at D8 and the host."""
    caps = sc.capacity_of()
    print("MEDIAN MINUS MEAN OF THE 11 QUANTILES, arm A0 raw network output (points of capacity)")
    raw = sc.scored_window(frame=sc.read_fits(arm="A0"))
    table = raw.group_by("time_series_id").agg(
        gap=(pl.mean_horizontal(*QUANTILE_COLUMNS) - pl.col("q50")).mean() * 100.0
    )
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00677_extend as ext  # ty: ignore[unresolved-import]

    ext.redirect()
    sites = prep.primaries().select("site", "time_series_id")
    label_of = dict(zip(sites["time_series_id"].cast(pl.Int64), sites["site"], strict=True))
    for row in table.sort("time_series_id").iter_rows(named=True):
        label = label_of.get(row["time_series_id"], row["time_series_id"])
        print(f"  {label}: mean minus median {row['gap']:+.3f}")
    print(
        "MEDIAN MINUS MEAN, stored systems (points of capacity); MEDIAN ERROR = median minus reading"
    )
    flags = e694_flags().select("valid_time", "flag")
    d8_truth = sc.d8_truth().rename({"power": "reading"})
    for name, series, capacity in (
        ("a0-d8", D8_SERIES, D8_CAPACITY_MVA),
        ("a1-d8", D8_SERIES, D8_CAPACITY_MVA),
        ("ah-d8", D8_SERIES, D8_CAPACITY_MVA),
        ("w0-host", sc.host_series_id(), caps[sc.host_series_id()]),
        ("w1-host", sc.host_series_id(), caps[sc.host_series_id()]),
    ):
        frame = read_predictions(experiment_id=695, name=name, row_set_id=sc.ROW_SET_ID)  # ty: ignore[invalid-argument-type]
        frame = sc.scored_window(frame=frame.filter(pl.col("time_series_id") == series))
        gap = (pl.mean_horizontal(*QUANTILE_COLUMNS) - pl.col("q50")) / capacity * 100.0
        print(f"  {name}: mean minus median {frame.select(gap.mean()).item():+.3f}")
        if name.endswith("d8"):
            joined = frame.join(d8_truth, on="valid_time").join(flags, on="valid_time", how="left")
            joined = joined.filter(pl.col("reading").is_not_null()).with_columns(
                err=(pl.col("q50") - pl.col("reading")) / capacity * 100.0
            )
            overall = joined.select(pl.col("err").mean()).item()
            on = joined.filter(pl.col("flag")).select(pl.col("err").mean()).item()
            off = joined.filter(~pl.col("flag")).select(pl.col("err").mean()).item()
            print(
                f"    median error (mean over rows): all {overall:+.2f}, tracked on {on:+.2f}, "
                f"tracked off {off:+.2f}"
            )
    print(f"data folder name check: {DATA_DIR.name}")


if __name__ == "__main__":
    main()

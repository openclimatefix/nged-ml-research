"""Checks of experiment 503: median minus mean forecast per Primary, and the median's error.

Prints, per arm, for the six Primaries on rows with no labelled event and leads up to 240 hours:
the mean absolute error of the median forecast (points of capacity), and per Primary the median
forecast minus the mean of the 11 quantiles (points of capacity).
"""

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00503_fit as fit  # ty: ignore[unresolved-import]
import e00503_score as score  # ty: ignore[unresolved-import]
import polars as pl

ARMS = ("a", "g", "gb", "b")


def main() -> None:
    """Print the checks."""
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    series = series.filter(pl.col("label").is_in(list(fit.SIGNED_MW_LABELS)))
    actuals, _, _ = s392.load_actuals(source="observed", series=series)
    flags, _ = s392.event_tables(series=series)
    pieces = []
    for fold in range(len(s392.FOLD_NAMES)):
        for arm in ARMS:
            stored = s392.read_fold(directory=score.SOURCES[arm], sites=list(fit.SIGNED_MW_LABELS), fold=fold)
            frame = adapter.forecast_frame(stored=stored, series=series, unit="fraction").filter(
                s392.lead_expression() <= score.MAX_LEAD_HOURS
            )
            pieces.append(
                frame.join(actuals, on=["time_series_id", "valid_time"])
                .join(flags.select("time_series_id", "valid_time", "in_event", "post48"), on=["time_series_id", "valid_time"], how="left")
                .with_columns(pl.col("in_event", "post48").fill_null(value=False))
                .filter(~pl.col("in_event") & ~pl.col("post48"))
                .join(series.select("time_series_id", "label", "capacity_mw"), on="time_series_id")
                .with_columns(
                    arm=pl.lit(arm),
                    error=(pl.col("q50") - pl.col("actual")).abs() * 100.0 / pl.col("capacity_mw"),
                    median_minus_mean=(pl.col("q50") - sum(pl.col(c) for c in s392.QUANTILES) / len(s392.QUANTILES)) * 100.0 / pl.col("capacity_mw"),
                )
                .select("arm", "label", "error", "median_minus_mean")
            )
    table = pl.concat(pieces)
    print("-- mean absolute error of the median (points of capacity), rows with no labelled event --")
    print(table.group_by("arm").agg(mae=pl.col("error").mean()).sort("arm"))
    print("-- median minus mean of the 11 quantiles, by Primary (points of capacity) --")
    print(table.group_by("label", "arm").agg(d=pl.col("median_minus_mean").mean()).pivot(on="arm", index="label", values="d").sort("label"))


if __name__ == "__main__":
    main()

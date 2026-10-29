r"""Event-level views of experiment 383: per-event table, leave-one-event-out, onset bands.

Reads `work/scores/rows.parquet` (from `e00383_score.py`) and the event list of the feature cache.
A contaminated-lag row outside the events is attributed to the event of its own Primary whose window
`[start, end + 8 days]` holds the row's valid time (the longest lag is a week). Prints, per event:
the length, the contaminated rows, and the CRPS of d, c1, c2, and c2 with widening; then the
leave-one-event-out contrast for the D5 and D16 events; then the gain by days between the event's
onset and the forecast origin (0 to 1, 1 to 3, over 3).

```bash
PYTHONPATH=ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-* DATA_PATH_INTERNAL=... \\
    uv run --no-sync python ml-experiments/00383-*/e00383_events.py
```
"""

from datetime import UTC, datetime
from typing import Final

import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00379_score as ref379  # ty: ignore[unresolved-import]
import polars as pl
from e00383_score import HEADLINE_BAND, OUT
from studies.dp_features import default_input_paths, load_or_build_feature_cache

AFTER: Final[pl.Expr] = pl.duration(days=8)
ARMS: Final[tuple[str, ...]] = ("d", "c1", "c2", "c2_w", "c1_w", "d_w")
ONSET_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-1 days", 0.0, 1.0),
    ("1-3 days", 1.0, 3.0),
    ("over 3 days", 3.0, 1e9),
)


def main() -> None:
    """Print the event tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=ref379.CACHE_DIR)
    events = (
        adapter.events_from_masks(cache=cache, series=series)
        .join(series.select("time_series_id", "label"), on="time_series_id")
        .filter(pl.col("end") >= datetime(2025, 7, 1, tzinfo=UTC))
        .sort("label", "start")
        .with_row_index("event")
    )
    rows = pl.read_parquet(OUT / "rows.parquet").filter(
        pl.col("arm").is_in(list(ARMS)), pl.col(HEADLINE_BAND), ~pl.col("event_rows")
    )
    contaminated = rows.filter(pl.col("lag_in_event"))
    joined = contaminated.join(events.drop("label"), on="time_series_id").filter(
        (pl.col("valid_time") >= pl.col("start")) & (pl.col("valid_time") <= pl.col("end") + AFTER)
    )
    joined = joined.with_columns(
        onset_days=(pl.col("power_fcst_init_time") - pl.col("start")).dt.total_hours() / 24.0,
        group=pl.col("group"),
    )
    print(f"events ending after 2025-07-01: {events.height}")
    table = (
        joined.group_by("event", "label", "start", "end", "arm")
        .agg(pl.col("crps").mean(), n=pl.len())
        .pivot(on="arm", index=["event", "label", "start", "end", "n"], values="crps")
        .with_columns(
            days=(pl.col("end") - pl.col("start")).dt.total_hours() / 24.0,
            c2_minus_d=pl.col("c2") - pl.col("d"),
            c1_minus_d=pl.col("c1") - pl.col("d"),
            c2w_minus_d=pl.col("c2_w") - pl.col("d"),
        )
        .sort("label", "start")
    )
    print("\n== Per event: CRPS on the contaminated-lag rows attributed to the event ==")
    print(table.drop("event"))
    print(f"\ncontaminated rows attributed to an event: {joined.height // len(ARMS):,}")
    print("\n== Leave one event out: signed-MW and MVA-only contaminated-lag CRPS, arm minus d ==")
    for left_out in [None, *events["event"].to_list()]:
        drop = (
            joined.filter(pl.col("event") == left_out).select(
                "time_series_id", "power_fcst_init_time", "valid_time"
            )
            if left_out is not None
            else None
        )
        part = (
            contaminated
            if drop is None
            else contaminated.join(
                drop, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="anti"
            )
        )
        wide = (
            part.group_by("group", "arm")
            .agg(pl.col("crps").mean())
            .pivot(on="arm", index="group", values="crps")
        )
        name = (
            "none"
            if left_out is None
            else str(
                events.filter(pl.col("event") == left_out)
                .select(
                    pl.concat_str("label", pl.col("start").dt.strftime("%Y-%m-%d"), separator=" ")
                )
                .item()
            )
        )
        out = wide.with_columns(
            *[
                (pl.col(a) - pl.col("d")).alias(f"{a}-d")
                for a in ("c1", "c2", "c1_w", "c2_w", "d_w")
            ]
        ).select("group", pl.col("^.*-d$"))
        print(f"left out: {name}")
        print(out.filter(pl.col("group") != "short-history"))
    print("\n== Gain by days from event onset to forecast origin (contaminated rows) ==")
    for label, low, high in ONSET_BANDS:
        band = joined.filter((pl.col("onset_days") >= low) & (pl.col("onset_days") < high))
        wide = (
            band.group_by("group", "arm")
            .agg(pl.col("crps").mean(), n=pl.len())
            .pivot(on="arm", index="group", values="crps")
        )
        counts = band.filter(pl.col("arm") == "d").group_by("group").agg(n=pl.len())
        print(label)
        print(
            wide.join(counts, on="group")
            .with_columns(
                *[(pl.col(a) - pl.col("d")).alias(f"{a}-d") for a in ("c1", "c2", "c2_w")]
            )
            .select("group", "n", "d", pl.col("^.*-d$"))
            .filter(pl.col("group") != "short-history")
        )


if __name__ == "__main__":
    main()

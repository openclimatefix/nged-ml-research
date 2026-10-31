"""Part H of experiment 395, rescore step: the median's share of event depth with corrected signs.

Reads `sw_block_detector_label_table_corrected.csv` (from `e00395_part_h_audit.py`) and the Part F
tables. Each named Primary's event sign is the sign measured at that Primary where it could be
measured, and otherwise the corrected sign of the first-named Primary (first-named) or its opposite
(other Primaries). The size stays the labelled `|depth_mw|`. The share followed is
`1 - (actual - q50) / signed depth` per row, so 0 means the median sits at normal running and 1
means it sits at the event reading; the table takes the median over rows of each event, and then
over events.
"""

import csv
import re
import sys
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00395_part_f_build as build  # ty: ignore[unresolved-import]
import polars as pl

HERE: Final[Path] = Path(__file__).parent
CORRECTED: Final[Path] = HERE / "sw_block_detector_label_table_corrected.csv"
FMT: Final[str] = "%Y-%m-%d %H:%M"
ARMS: Final[tuple[str, ...]] = (
    "bundle",
    "monthly_control",
    "stack",
    "w383_d",
    "w383_c1",
    "w383_c2",
)


def corrected_events(*, events: pl.DataFrame) -> pl.DataFrame:
    """Return the merged events with the corrected signed depth."""
    labels = set(events["label"].to_list())
    items: list[tuple[str, int, int, float]] = []
    with CORRECTED.open() as handle:
        for row in csv.DictReader(handle):
            named = list(dict.fromkeys(re.findall(r"D\d+", row["pair"])))
            if not named or not row["corrected_sign_first"]:
                continue
            measured = dict(re.findall(r"(D\d+)=(-?[\d.]+)", row["measured_depth_others_mw"]))
            size = abs(float(row["depth_mw"]))
            onset = int(
                common.grid_index(
                    times=pl.Series([row["onset_utc"]]).str.to_datetime(FMT, time_zone="UTC")[0]
                )
            )
            end = int(
                common.grid_index(
                    times=pl.Series([row["end_utc"]]).str.to_datetime(FMT, time_zone="UTC")[0]
                )
            )
            first_sign = int(float(row["corrected_sign_first"]))
            for i, label in enumerate(named):
                if label not in labels:
                    continue
                if i == 0:
                    sign = first_sign
                elif label in measured:
                    sign = 1 if float(measured[label]) > 0 else -1
                else:
                    sign = -first_sign
                items.append((label, onset, end, sign * size))
    rows = []
    for event in events.iter_rows(named=True):
        inside = [
            d
            for lab, o, e, d in items
            if lab == event["label"] and event["onset_idx"] <= o <= event["end_idx"]
        ]
        rows.append(
            pl.DataFrame(
                {
                    "event_id": [event["event_id"]],
                    "corrected_depth_mw": [max(inside, key=abs) if inside else None],
                }
            )
        )
    return events.join(pl.concat(rows), on="event_id")


def main() -> None:
    """Print the share of depth followed by the median, old signs against corrected signs."""
    events = pl.read_parquet(build.WORK / "events_unpadded.parquet")
    events = corrected_events(events=events.with_columns(pl.col("event_id").cast(pl.Int64)))
    old = events.filter(pl.col("corrected_depth_mw").is_not_null())
    print(
        "events",
        events.height,
        "with corrected depth",
        old.height,
        "sign changed",
        (old["depth_mw"] * old["corrected_depth_mw"] < 0).sum(),
    )
    tags = pl.read_parquet(build.WORK / "tags.parquet")
    keys = build.KEYS
    cols = [*keys, "lead_hours", "flagged", "q50_mw", "actual_mw"]
    for name in ARMS:
        rows = (
            pl.read_parquet(build.WORK / f"rows_{name}.parquet", columns=cols)
            .filter(pl.col("lead_hours") < 240, ~pl.col("flagged"))
            .join(tags, on=keys)
            .filter(pl.col("target_event_id") >= 0)
            .join(
                events.select("event_id", "depth_mw", "corrected_depth_mw"),
                left_on="target_event_id",
                right_on="event_id",
            )
            .with_columns(
                group=pl.when(pl.col("lag_count") == 0)
                .then(pl.lit("0 lags"))
                .when(pl.col("lag_count") == 3)
                .then(pl.lit("3 lags"))
                .otherwise(pl.lit("1 or 2 lags")),
                res=pl.col("actual_mw") - pl.col("q50_mw"),
            )
            .with_columns(
                old_share=1 - pl.col("res") / pl.col("depth_mw"),
                new_share=1 - pl.col("res") / pl.col("corrected_depth_mw"),
            )
        )
        per_event = (
            rows.group_by("group", "target_event_id")
            .agg(
                old=pl.col("old_share").median(),
                new=pl.col("new_share").median(),
                n=pl.len(),
                res=pl.col("res").median(),
                d=pl.col("corrected_depth_mw").first(),
            )
            .filter(pl.col("n") >= 10)
        )
        print(
            name,
            per_event.group_by("group")
            .agg(
                events=pl.len(),
                old_share=pl.col("old").median(),
                new_share=pl.col("new").median(),
                same_sign=((pl.col("res") * pl.col("d")) > 0).mean(),
            )
            .sort("group")
            .with_columns(pl.selectors.float().round(3))
            .rows(),
        )
        if name == "bundle":
            z = per_event.filter(pl.col("group") == "0 lags").join(
                events.select(pl.col("event_id").alias("target_event_id"), "label"),
                on="target_event_id",
            )
            print(
                "sign agreement per Primary, 0 lags, corrected:",
                z.group_by("label")
                .agg(n=pl.len(), same=((pl.col("res") * pl.col("d")) > 0).mean())
                .sort("label")
                .rows(),
            )


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()

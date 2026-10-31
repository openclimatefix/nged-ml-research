"""Experiment 400, tag step: unpadded corrected events and the lag and target tags of every row.

Reads the corrected label table (a copy of `sw_block_detector_label_table_corrected.csv` from the
head of nged-ml-research PR 49, saved as `work/label_table_corrected.csv`). The first-named Primary
of a row takes `corrected_depth_first_mw` and any other named Primary the opposite sign, as in
experiment 395's Part F. Tagging uses Part F's `tag_rows`.
"""

import csv
import re
import sys
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00400_blend as blend
import e00400_partf as pf  # ty: ignore[unresolved-import]
import polars as pl

LABELS = blend.WORK / "label_table_corrected.csv"


def corrected_events(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return merged unpadded events per Primary with the corrected signed depth in MW."""
    known = set(series["label"].to_list())
    per: dict[str, list[tuple[int, int, float]]] = {}
    with LABELS.open() as handle:
        for r in csv.DictReader(handle):
            named = re.findall(r"D\d+", r["pair"])
            onset = pl.Series([r["onset_utc"]]).str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC")[
                0
            ]
            end = pl.Series([r["end_utc"]]).str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC")[0]
            depth = float(r["corrected_depth_first_mw"])
            for i, label in enumerate(named):
                if label in known:
                    per.setdefault(label, []).append(
                        (
                            int(common.grid_index(times=onset)),
                            int(common.grid_index(times=end)),
                            depth if i == 0 else -depth,
                        )
                    )
    out = []
    for label, items in per.items():
        items.sort()
        merged: list[list] = []
        for onset, end, depth in items:
            if merged and onset <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
                if abs(depth) > abs(merged[-1][2]):
                    merged[-1][2] = depth
            else:
                merged.append([onset, end, depth])
        out.extend(
            {"label": label, "onset_idx": o, "end_idx": e, "depth_mw": d} for o, e, d in merged
        )
    table = pl.DataFrame(out).join(
        series.select("label", "time_series_id", "capacity_mw"), on="label"
    )
    return (
        table.sort("label", "onset_idx")
        .with_row_index("event_id")
        .with_columns(
            length_days=(pl.col("end_idx") - pl.col("onset_idx") + 1) * 0.5 / 24,
            depth_share=pl.col("depth_mw").abs() / pl.col("capacity_mw") * 100,
        )
    )


def main() -> None:
    """Write the corrected events and the tags."""
    series = adapter.series_table()
    events = corrected_events(series=series)
    events.write_parquet(blend.WORK / "events_corrected.parquet")
    print("events", events.height, flush=True)
    keys = pl.read_parquet(pf.WORK / "slice_table.parquet").select(*blend.KEYS)
    tags = pf.tag_rows(keys=keys, events=events)
    tags.write_parquet(blend.WORK / "tags_corrected.parquet")
    print(tags.group_by("lag_count").len().sort("lag_count"), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

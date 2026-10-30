"""Experiment 401, Job 1: the matched events, with corrected and measured depths.

Events are experiment 395 Part F's merged unpadded intervals (`events_unpadded.parquet`) with
Part H's corrected signed depth (a copy of `sw_block_detector_label_table_corrected.csv` from the
head of nged-ml-research PR 49, at `work/label_table_corrected.csv` of experiment 400). Each named
Primary's event sign is the sign measured at that Primary where it could be measured, and otherwise
the corrected sign of the first-named Primary (first-named) or its opposite (other Primaries). The
size stays the labelled `|depth_mw|`. The measured depth is the label table's measured step at that
Primary (`measured_depth_first_mw` or the entry in `measured_depth_others_mw`), signed.
"""

import csv
import re
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00400_blend as blend
import e00400_partf as pf  # ty: ignore[unresolved-import]
import polars as pl

CORRECTED: Final = blend.WORK / "label_table_corrected.csv"
FMT: Final[str] = "%Y-%m-%d %H:%M"


def events_with_depths() -> pl.DataFrame:
    """Return Part F's 95 unpadded events with `corrected_depth_mw` and `measured_depth_mw`."""
    events = pl.read_parquet(pf.WORK / "events_unpadded.parquet").with_columns(
        pl.col("event_id").cast(pl.Int64)
    )
    labels = set(events["label"].to_list())
    items: list[tuple[str, int, int, float, float | None]] = []
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
                    meas = (
                        float(row["measured_depth_first_mw"])
                        if row["measured_depth_first_mw"]
                        else None
                    )
                elif label in measured:
                    sign = 1 if float(measured[label]) > 0 else -1
                    meas = float(measured[label])
                else:
                    sign = -first_sign
                    meas = None
                items.append((label, onset, end, sign * size, meas))
    rows = []
    for event in events.iter_rows(named=True):
        inside = [
            (d, m)
            for lab, o, e, d, m in items
            if lab == event["label"] and event["onset_idx"] <= o <= event["end_idx"]
        ]
        best = max(inside, key=lambda t: abs(t[0])) if inside else (None, None)
        rows.append(
            {
                "event_id": event["event_id"],
                "corrected_depth_mw": best[0],
                "measured_depth_mw": best[1],
            }
        )
    return events.join(pl.DataFrame(rows), on="event_id").filter(
        pl.col("corrected_depth_mw").is_not_null()
    )

"""Part H of experiment 395: audit the sign convention of the hand-labelled load-shift events.

For each labelled event and each Primary it names, the measured step is the mean cleaned power in
the event's first 48 hours (or the whole event if shorter) minus the mean in the 48 hours before the
onset. Power is read through `studies.power.scan_power`, so flagged zeros are dropped. The measured
step is a plain difference of means, so weather and the daily cycle (the same hours enter both
windows) are not removed; the size comparison is therefore only as good as the 48-hour windows.
Writes `sw_block_detector_label_table_corrected.csv` beside this script.
"""

import csv
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import polars as pl
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR

LABELS: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_block_detector_label_table.csv"
OUT: Final[Path] = Path(__file__).parent / "sw_block_detector_label_table_corrected.csv"
WINDOW: Final[timedelta] = timedelta(hours=48)
SIZE_TOLERANCE: Final[float] = 2.0
"""A measured step more than this factor from |depth_mw| (either way) is a size disagreement."""
FMT: Final[str] = "%Y-%m-%d %H:%M"


def read_events() -> list[dict]:
    """Return the label-table rows that carry an end time and a depth, with parsed times."""
    out = []
    with LABELS.open() as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if not row["end_utc"] or not row["depth_mw"]:
                continue
            row["row"] = index
            row["named"] = list(dict.fromkeys(re.findall(r"D\d+", row["pair"])))
            out.append(row)
    return out


def step_at(*, power: pl.DataFrame, onset, end) -> tuple[float | None, int]:  # noqa: ANN001
    """Return the measured step and the number of readings in the event window."""
    stop = min(end, onset + WINDOW)
    before = power.filter((pl.col("time") > onset - WINDOW) & (pl.col("time") <= onset))
    during = power.filter((pl.col("time") > onset) & (pl.col("time") <= stop))
    if before.height < 24 or during.height < 4:
        return None, during.height
    return float(during["power"].mean() - before["power"].mean()), during.height  # ty: ignore[unsupported-operator]


def main() -> None:
    """Measure every event and write the corrected table."""
    events = read_events()
    series = adapter.series_table().select("label", "time_series_id")
    ids = dict(zip(series["label"], series["time_series_id"], strict=True))
    parse = lambda text: pl.Series([text]).str.to_datetime(FMT, time_zone="UTC")[0]  # noqa: E731
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(list(ids.values())))
        .select("time_series_id", "time", "power")
        .collect()
    )
    by_id = {i: power.filter(pl.col("time_series_id") == i) for i in ids.values()}
    rows = []
    for event in events:
        onset, end = parse(event["onset_utc"]), parse(event["end_utc"])
        depth = float(event["depth_mw"])
        steps = {}
        for label in event["named"]:
            if label in ids:
                steps[label] = step_at(power=by_id[ids[label]], onset=onset, end=end)[0]
        first = event["named"][0] if event["named"] else None
        measured_first = steps.get(first)
        others = [v for k, v in steps.items() if k != first and v is not None]
        sign_ok = measured_first is not None and measured_first * depth > 0
        size_ok = measured_first is not None and (
            abs(depth) / SIZE_TOLERANCE <= abs(measured_first) <= abs(depth) * SIZE_TOLERANCE
        )
        pair_opposite = (
            None
            if measured_first is None or not others
            else all(measured_first * o < 0 for o in others)
        )
        rows.append(
            {
                **{
                    k: event[k]
                    for k in (
                        "onset_utc",
                        "end_utc",
                        "depth_mw",
                        "duration_hours",
                        "pair",
                        "source",
                        "confidence",
                    )
                },
                "measured_depth_first_mw": None
                if measured_first is None
                else round(measured_first, 3),
                "measured_depth_others_mw": ";".join(
                    f"{k}={v:.3f}" for k, v in steps.items() if k != first and v is not None
                ),
                "measured_sign_first": None
                if measured_first is None
                else int(measured_first > 0) * 2 - 1,
                "depth_sign_ok": sign_ok if measured_first is not None else None,
                "depth_size_ok": size_ok if measured_first is not None else None,
                "pair_opposite": pair_opposite,
                "n_named": len(event["named"]),
                "first_primary": first,
            }
        )
    table = pl.DataFrame(rows)
    table = table.with_columns(
        disagreement=pl.when(pl.col("measured_depth_first_mw").is_null())
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(~pl.col("depth_sign_ok") | ~pl.col("depth_size_ok"))
    )
    table = (
        table.with_columns(
            rule_sign_first=pl.when(pl.col("pair").str.contains(" to"))
            .then(-1)
            .otherwise(pl.col("depth_mw").cast(pl.Float64).sign().cast(pl.Int64))
        )
        .with_columns(
            corrected_sign_first=pl.coalesce("measured_sign_first", "rule_sign_first"),
            sign_source=pl.when(pl.col("measured_sign_first").is_null())
            .then(pl.lit("rule"))
            .otherwise(pl.lit("measured")),
        )
        .with_columns(
            corrected_depth_first_mw=pl.col("corrected_sign_first")
            * pl.col("depth_mw").cast(pl.Float64).abs(),
            sign_corrected=pl.col("corrected_sign_first")
            != pl.col("depth_mw").cast(pl.Float64).sign().cast(pl.Int64),
        )
    )
    table.write_csv(OUT)
    print(
        "rule agrees with the measured sign on",
        table.filter(pl.col("measured_sign_first").is_not_null())
        .select((pl.col("rule_sign_first") == pl.col("measured_sign_first")).sum())
        .item(),
        "of",
        table["measured_sign_first"].is_not_null().sum(),
        "events",
    )
    print(
        "events whose sign is corrected:",
        table["sign_corrected"].sum(),
        "of",
        table.height,
        "; of those with a measured sign:",
        table.filter(pl.col("sign_source") == "measured")["sign_corrected"].sum(),
    )
    print(
        "size: median measured/|depth_mw| among events with a measured step:",
        table.filter(pl.col("measured_depth_first_mw").is_not_null())
        .select(
            (
                pl.col("measured_depth_first_mw").abs() / pl.col("depth_mw").cast(pl.Float64).abs()
            ).median()
        )
        .item(),
    )
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(250)
    print(
        "events with end and depth:",
        table.height,
        "measurable:",
        table["measured_depth_first_mw"].is_not_null().sum(),
    )
    print(table.group_by("depth_sign_ok", "depth_size_ok").len())
    print(
        "pair opposite (events with a measurable second Primary):",
        table.filter(pl.col("pair_opposite").is_not_null()).group_by("pair_opposite").len().rows(),
    )
    print(
        table.filter(pl.col("disagreement")).select(
            "onset_utc",
            "pair",
            "depth_mw",
            "measured_depth_first_mw",
            "measured_depth_others_mw",
            "depth_sign_ok",
            "depth_size_ok",
            "pair_opposite",
            "confidence",
        )
    )
    print(
        table.group_by("first_primary")
        .agg(
            events=pl.len(),
            measurable=pl.col("depth_sign_ok").is_not_null().sum(),
            wrong_sign=(~pl.col("depth_sign_ok")).sum(),
            wrong_size=(~pl.col("depth_size_ok")).sum(),
        )
        .sort("first_primary")
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

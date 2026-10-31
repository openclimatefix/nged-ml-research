"""Copy of experiment 395 Part F build script (PR 49, unmerged): row tags and flagged keys.

Re-scores stored forecasts with no refit. The events are the hand-labelled rows of
`briefs/sw_block_detector_label_table.csv` with an end time, unpadded, one interval per named
Primary; overlapping intervals of one Primary are merged. The sign of an event at a Primary is the
sign of `depth_mw` for the first-named Primary and the opposite sign for any other named Primary
(load moved between Primaries; the label table does not state signs per Primary, so this is an
assumption). Each scored half-hour is tagged with the number of its three lagged-power inputs that
lie inside an event, and the position of its target relative to the event.

Scoring actuals: experiment 103's `rows.parquet` through the feature cache. Half-hours flagged in
the production `cleaned_power_time_series` table are tagged `flagged`; the report gives each number
with and without them.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\
ml-experiments/00390-xgboost-d11-d14-gap-and-load-shift-event-lags:\
ml-experiments/00395-xgboost-level-state-winter-spread-anchor-stored-forecasts \
    uv run --no-sync python ml-experiments/00395-*/e00395_part_f_build.py
```
"""

import csv
import gc
import re
import sys
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00390_build as b390  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = (
    DATA / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts" / "work" / "part_f"
)
E383_ARMS: Final[Path] = DATA / "00383-load-shift-xgboost-oracle-first-part-1" / "work" / "arms"
CLEANED: Final[Path] = DATA.parent / "NGED" / "cleaned_power_time_series.delta"
LABELS: Final[Path] = DATA / "briefs" / "sw_block_detector_label_table.csv"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
STEP_HOURS: Final[float] = 0.5
PART_A_WORK: Final[Path] = WORK.parent


def event_intervals(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return merged unpadded events per Primary: onset and end grid indices, signed depth in MW."""
    labels = set(series["label"].to_list())
    per: dict[str, list[tuple[int, int, float]]] = {}
    with LABELS.open() as handle:
        for r in csv.DictReader(handle):
            if not r["end_utc"] or not r["depth_mw"]:
                continue
            named = re.findall(r"D\d+", r["pair"])
            onset = pl.Series([r["onset_utc"]]).str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC")[
                0
            ]
            end = pl.Series([r["end_utc"]]).str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC")[0]
            depth = float(r["depth_mw"])
            for i, label in enumerate(named):
                if label in labels:
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
        for onset, end, depth in merged:
            out.append({"label": label, "onset_idx": onset, "end_idx": end, "depth_mw": depth})
    table = pl.DataFrame(out).join(
        series.select("label", "time_series_id", "capacity_mw"), on="label"
    )
    return (
        table.sort("label", "onset_idx")
        .with_row_index("event_id")
        .with_columns(
            length_days=(pl.col("end_idx") - pl.col("onset_idx") + 1) * STEP_HOURS / 24,
            depth_share=pl.col("depth_mw").abs() / pl.col("capacity_mw") * 100,
        )
        .with_columns(
            length_class=pl.when(pl.col("length_days") < 1).then(pl.lit("under 1 day"))
            .when(pl.col("length_days") <= 7).then(pl.lit("1 to 7 days"))
            .otherwise(pl.lit("over 7 days"))
        )
    )  # fmt: skip


def tag_rows(*, keys: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Return the keys with the unpadded lag count, event ids, and target position."""
    size = common.grid_times().len()
    pieces = []
    for sid in keys["time_series_id"].unique().to_list():
        part = keys.filter(pl.col("time_series_id") == sid)
        mine = events.filter(pl.col("time_series_id") == sid).sort("onset_idx")
        cell = np.full(size, -1, dtype=np.int64)
        for r in mine.iter_rows(named=True):
            cell[r["onset_idx"] : r["end_idx"] + 1] = r["event_id"]
        onsets = mine["onset_idx"].to_numpy()
        ends = mine["end_idx"].to_numpy()
        ids = mine["event_id"].to_numpy().astype(np.int64)
        target = np.asarray(common.grid_index(times=part["valid_time"]))
        origin = np.asarray(common.grid_index(times=part["power_fcst_init_time"]))
        lag_index = common.origin_lags(
            power=np.arange(size, dtype=np.float32), target=target, origin=origin
        ).astype(np.int64)
        lag_event = cell[lag_index]
        in_lag = lag_event >= 0
        chosen = np.where(
            in_lag[:, 0],
            lag_event[:, 0],
            np.where(in_lag[:, 1], lag_event[:, 1], np.where(in_lag[:, 2], lag_event[:, 2], -1)),
        )
        target_event = cell[target]
        hours_since_onset = np.full(len(target), np.nan)
        hours_after_end = np.full(len(target), np.nan)
        if len(onsets):
            inside = target_event >= 0
            at = np.searchsorted(onsets, target, side="right") - 1
            hours_since_onset[inside] = (target[inside] - onsets[at[inside]]) * STEP_HOURS
            prev = np.searchsorted(ends, target, side="left") - 1
            after = ~inside & (prev >= 0)
            hours_after_end[after] = (target[after] - ends[prev[after]]) * STEP_HOURS
            after_event = np.where(after, ids[np.maximum(prev, 0)], -1)
        else:
            after_event = np.full(len(target), -1)
        pieces.append(
            part.select(*KEYS).with_columns(
                lag_count=pl.Series(in_lag.sum(axis=1)),
                lag_event_id=pl.Series(chosen),
                target_event_id=pl.Series(target_event),
                hours_since_onset=pl.Series(hours_since_onset),
                hours_after_end=pl.Series(hours_after_end),
                after_event_id=pl.Series(after_event),
            )
        )
    return pl.concat(pieces)


def flagged_keys(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the (series, valid time) pairs flagged in the cleaned power table."""
    return (
        pl.scan_delta(str(CLEANED))
        .filter(pl.col("drop_reason").is_not_null())
        .select(
            pl.col("time_series_id").cast(series.schema["time_series_id"]),
            valid_time=pl.col("time"),
        )
        .collect()
        .with_columns(flagged=pl.lit(True))
    )


def load_383(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return experiment 383's plain weekly member (`d`), `c1`, and `c2`, raw, in MW."""
    out = {}
    for name in ("d", "c1", "c2"):
        stored = pl.read_parquet(E383_ARMS / f"{name}.parquet").filter(pl.col("lead_hours") <= 336)
        out[f"w383_{name}"] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return out


def arm_frames(*, series: pl.DataFrame):  # noqa: ANN201
    """Yield (name, frame) one stored forecast at a time, so only one is in memory."""
    yield "bundle", s381.arm_frame(regime="weekly", arm="bundle_solar+sunny+hl1", series=series)
    yield "monthly_control", s381.arm_frame(regime="monthly", arm="control", series=series)
    yield "stack", base.stack_frames(series=series)["stack"]
    yield from load_383(series=series).items()


def main() -> None:
    """Build the tagged per-row tables, one stored forecast at a time (low memory)."""
    WORK.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    events = event_intervals(series=series)
    events.write_parquet(WORK / "events_unpadded.parquet")
    print("events", events.height, flush=True)
    table = pl.read_parquet(PART_A_WORK / "slice_table.parquet")
    table.write_parquet(WORK / "slice_table.parquet")
    tag_rows(keys=table.select(*KEYS), events=events).write_parquet(WORK / "tags.parquet")
    flags = flagged_keys(series=series)
    for name, frame in arm_frames(series=series):
        frame = frame.join(table.select(*KEYS), on=KEYS, how="semi")  # noqa: PLW2901
        use = table.join(frame.select(*KEYS), on=KEYS, how="semi")
        rows = b390.trimmed_rows(name=name, frame=frame, table=use, ctx=ctx, series=series)
        rows = rows.join(flags, on=["time_series_id", "valid_time"], how="left").with_columns(
            pl.col("flagged").fill_null(value=False)
        )
        extra = frame.join(ctx["actuals"], on=["time_series_id", "valid_time"]).select(
            *KEYS,
            q50_mw=pl.col("q50"),
            actual_mw=pl.col("actual"),
            q95_mw=pl.col("q95"),
            q99_mw=pl.col("q99"),
        )
        rows.join(extra, on=KEYS, how="left", validate="1:1").write_parquet(
            WORK / f"rows_{name}.parquet"
        )
        print("wrote", name, rows.height, flush=True)
        del frame, rows, extra, use
        gc.collect()
    padded_events = b390.event_table(cache_events=ctx["events"], series=series)
    b390.row_event_flags(
        keys=table.select(*KEYS), events=padded_events, series=series
    ).write_parquet(WORK / "padded_flags.parquet")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

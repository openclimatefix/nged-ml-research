"""Build the row tables experiment 390 analyses: stored forecasts re-scored, with event flags.

Nothing is refitted. The arms are the weekly bundle of experiment 381, its plain weekly member
(experiment 379 `only_W`), the GAM stack of experiment 377, BG2 and its other members (experiment
379), the stack without the Kalman filter (experiment 380 `drop_k`), and the weekly bundle with a
causal level offset added (A3). Every arm is scored on the half-hours they all share, with
`studies.eval_protocol`, and written as a parquet file of per-row losses in the private work folder.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00390-*/e00390_build.py
```
"""
# ruff: noqa: PERF401, RET504

import csv
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = DATA / "00390-xgboost-d11-d14-gap-and-load-shift-event-lags" / "work"
E380_ARMS: Final[Path] = DATA / "00380-gam-stack-ablation-and-jaggedness-at-16-primaries/work/arms"
LABEL_TABLE: Final[Path] = DATA / "briefs" / "sw_block_detector_label_table.csv"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
OFFSET_DAYS: Final[tuple[int, ...]] = (3, 7, 14)
NEWEST_AGE_HOURS: Final[float] = 3.5
SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
BG2_MEMBERS: Final[tuple[str, ...]] = ("only_W", "only_M", "only_AN", "only_PT", "only_ST", "full")
QUANTILES: Final[list[str]] = [
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
]  # fmt: skip


def kalman_off_frame(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 380's `drop_k` arm (the stack without its Kalman filter) in MW."""
    pieces = []
    for label in SIGNED_SIX:
        one = series.filter(pl.col("label") == label)
        stored = pl.read_parquet(E380_ARMS / f"drop_k_{label}.parquet")
        pieces.append(adapter.forecast_frame(stored=stored, series=one, unit="points"))
    return pl.concat(pieces)


def level_offset_frame(*, bundle: pl.DataFrame, actuals: pl.DataFrame, days: int) -> pl.DataFrame:
    """Return the bundle with a causal level offset added to every quantile.

    The offset of a forecast origin is the median, over the previous `days` days, of the bundle's
    own residuals (actual minus median forecast) at readings at least `NEWEST_AGE_HOURS` before the
    origin. A reading's residual comes from the forecast with the shortest lead that targets it.
    Only readings that predate the origin enter, so the offset uses no later information.
    """
    window = timedelta(days=days)
    latest = timedelta(hours=NEWEST_AGE_HOURS)
    pieces = []
    for series_id in bundle["time_series_id"].unique().sort().to_list():
        one = bundle.filter(pl.col("time_series_id") == series_id)
        residual = (
            one.sort("valid_time", "power_fcst_init_time", descending=[False, True])
            .group_by("valid_time", maintain_order=True)
            .agg(pl.col("q50").first())
            .join(
                actuals.filter(pl.col("time_series_id") == series_id).select(
                    "valid_time", "actual"
                ),
                on="valid_time",
            )
            .with_columns(residual=pl.col("actual") - pl.col("q50"))
            .filter(pl.col("residual").is_finite())
            .sort("valid_time")
        )
        times = residual["valid_time"].to_numpy()
        values = residual["residual"].to_numpy()
        offsets = {}
        for origin in one["power_fcst_init_time"].unique().to_list():
            hi = np.searchsorted(times, np.datetime64(origin - latest, "us"), side="right")
            lo = np.searchsorted(times, np.datetime64(origin - latest - window, "us"), side="left")
            offsets[origin] = float(np.median(values[lo:hi])) if hi > lo else 0.0
        lookup = pl.DataFrame(
            {"power_fcst_init_time": list(offsets), "_offset": list(offsets.values())},
            schema={
                "power_fcst_init_time": one.schema["power_fcst_init_time"],
                "_offset": pl.Float64,
            },
        )
        pieces.append(
            one.join(lookup, on="power_fcst_init_time", how="left")
            .with_columns(pl.col(*QUANTILES) + pl.col("_offset").cast(pl.Float32))
            .drop("_offset")
        )
    return pl.concat(pieces)


def label_table() -> pl.DataFrame:
    """Return the hand label table with `onset`, `end`, `depth_mw`, and the Primaries named."""
    rows = []
    with LABEL_TABLE.open() as handle:
        for r in csv.DictReader(handle):
            rows.append(
                {
                    "onset": r["onset_utc"],
                    "end": r["end_utc"] or None,
                    "depth_mw": float(r["depth_mw"]) if r["depth_mw"] else None,
                    "named": sorted(set(re.findall(r"D\d+", r["pair"]))),
                    "source": r["source"],
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("onset").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC"),
        pl.col("end").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC"),
    )


def event_table(*, cache_events: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Return the padded hand-filtered events with their length, matched depth, and an id.

    Depth comes from the label table: the row that names the event's Primary and whose onset lies
    inside the event or within 2 days before it (the larger absolute depth if several). An event
    with no such row has a null depth.
    """
    labels = label_table()
    out = []
    for i, row in enumerate(
        cache_events.join(
            series.select("time_series_id", "label", "capacity_mw"), on="time_series_id"
        )
        .sort("time_series_id", "start")
        .iter_rows(named=True)
    ):
        match = labels.filter(
            pl.col("named").list.contains(row["label"]),
            pl.col("depth_mw").is_not_null(),
            pl.col("onset") >= row["start"] - timedelta(days=2),
            pl.col("onset") <= row["end"],
        )
        depth = (
            float(match.sort(pl.col("depth_mw").abs(), descending=True)["depth_mw"][0])
            if match.height
            else None
        )
        out.append({**row, "event_id": i, "depth_mw": depth})
    table = pl.DataFrame(out).with_columns(
        duration_hours=(pl.col("end") - pl.col("start")).dt.total_minutes() / 60 + 0.5,
        depth_share=pl.col("depth_mw").abs() / pl.col("capacity_mw") * 100,
    )
    return table


def row_event_flags(
    *, keys: pl.DataFrame, events: pl.DataFrame, series: pl.DataFrame
) -> pl.DataFrame:
    """Return the keys with event flags: lags inside an event, target inside an event, event id.

    `event_id` is the event holding the newest usable lag, else the day lag, else the week lag, else
    (for a target inside an event with clean lags) the event holding the target. `lag_clean` is true
    where none of the three lags lies inside any event.
    """
    cache = s381.load_or_build_feature_cache(
        paths=s381.default_input_paths(), cache_dir=base.CACHE_DIR
    )
    size = common.grid_times().len()
    pieces = []
    for series_id, label, _ in series.iter_rows():
        part = keys.filter(pl.col("time_series_id") == series_id)
        if not part.height:
            continue
        mask = np.zeros(size, dtype=bool)
        idx = np.asarray(common.grid_index(times=cache.times_series()))
        inside = (idx >= 0) & (idx < size)
        mask[idx[inside]] = cache.sites[label].switching_mask[inside]
        # Event id per grid cell.
        edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        cell_event = np.full(size, -1, dtype=np.int64)
        mine = events.filter(pl.col("time_series_id") == series_id).sort("start")
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for s, e, eid in zip(starts, ends, mine["event_id"].to_list(), strict=True):
            cell_event[s:e] = eid
        target = np.asarray(common.grid_index(times=part["valid_time"]))
        origin = np.asarray(common.grid_index(times=part["power_fcst_init_time"]))
        lag_index = common.origin_lags(
            power=np.arange(size, dtype=np.float32), target=target, origin=origin
        ).astype(np.int64)
        lag_event = cell_event[lag_index]
        in_lag = lag_event >= 0
        chosen = np.where(
            in_lag[:, 0],
            lag_event[:, 0],
            np.where(in_lag[:, 1], lag_event[:, 1], np.where(in_lag[:, 2], lag_event[:, 2], -1)),
        )
        target_event = cell_event[target]
        pieces.append(
            part.select(*KEYS).with_columns(
                lag_origin_in_event=pl.Series(in_lag[:, 0]),
                lag_day_in_event=pl.Series(in_lag[:, 1]),
                lag_week_in_event=pl.Series(in_lag[:, 2]),
                lag_in_event=pl.Series(in_lag.any(axis=1)),
                target_in_event=pl.Series(target_event >= 0),
                lag_event_id=pl.Series(chosen),
                target_event_id=pl.Series(target_event),
            )
        )
    return pl.concat(pieces)


def trimmed_rows(
    *,
    name: str,
    frame: pl.DataFrame,
    table: pl.DataFrame,
    ctx: dict,
    series: pl.DataFrame,
) -> pl.DataFrame:
    """Return an arm's per-row losses, plus the signed median error and the 90% width, in points."""
    kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
    rows = metrics.scored_rows(
        forecast=kept,
        actuals=ctx["actuals"],
        series=series,
        thresholds=ctx["thresholds"],
        slice_table=table,
    )
    extra = (
        kept.join(ctx["actuals"], on=["time_series_id", "valid_time"])
        .join(series.select("time_series_id", "capacity_mw"), on="time_series_id")
        .select(
            *KEYS,
            q50_error=(pl.col("q50") - pl.col("actual")) / pl.col("capacity_mw") * 100,
            width90=(pl.col("q95") - pl.col("q05")) / pl.col("capacity_mw") * 100,
        )
    )
    slices = [c for c in table.columns if c not in KEYS]
    return (
        rows.join(extra, on=KEYS, how="left", validate="1:1")
        .drop(*slices)
        .with_columns(arm=pl.lit(name))
    )


def main() -> None:
    """Build and write every table."""
    WORK.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    print("building the arms")
    bundle = s381.arm_frame(regime="weekly", arm="bundle_solar+sunny+hl1", series=series)
    arms = {
        "bundle": bundle,
        "stack": base.stack_frames(series=series)["stack"],
        **base.reference_frames(series=series),
        **{f"bg2_{m}": s381.stored_frame(name=m, series=series) for m in BG2_MEMBERS},
        "stack_kalman_off": kalman_off_frame(series=series),
    }
    for days in OFFSET_DAYS:
        arms[f"bundle_offset_{days}d"] = level_offset_frame(
            bundle=bundle, actuals=ctx["actuals"], days=days
        )
    # Only the half-hours that the bundle, the stack, and BG2 share (the stack-off arm covers six
    # Primaries and is scored on its own table below).
    core = {k: arms[k] for k in ("bundle", "stack", "bg2", "bg2_only_W", "bg2_full")}
    table, _, _ = s381.score_arms(arms=core, series=series, ctx=ctx)
    table.write_parquet(WORK / "slice_table.parquet")
    events = event_table(cache_events=ctx["events"], series=series)
    events.write_parquet(WORK / "events.parquet")
    flags = row_event_flags(keys=table.select(*KEYS), events=events, series=series)
    flags.write_parquet(WORK / "event_flags.parquet")
    for name, frame in arms.items():
        if name in core or name.startswith("bundle_offset"):
            use_table = table
        else:
            use_table = table.join(frame.select(*KEYS), on=KEYS, how="inner")
        rows = trimmed_rows(name=name, frame=frame, table=use_table, ctx=ctx, series=series)
        rows.write_parquet(WORK / f"rows_{name}.parquet")
        print(f"wrote rows_{name}: {rows.height} rows")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

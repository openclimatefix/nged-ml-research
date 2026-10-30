"""Part A of experiment 395: level-state arms K0 to K4 on the stored weekly bundle and GAM stack.

Nothing is refitted. Writes per-row losses of every arm to the private work folder, the monthly
filter settings, and the shared slice table.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00390-xgboost-d11-d14-gap-and-load-shift-event-lags:\
ml-experiments/00395-xgboost-level-state-winter-spread-anchor-stored-forecasts \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00395-*/e00395_build.py
```
"""

import json
import sys
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_run as run377  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00390_build as b390  # ty: ignore[unresolved-import]
import e00395_levels as lv  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts"
    / "work"
)
KEYS: Final[list[str]] = lv.KEYS
WIDEN: Final[float] = 1.15
SIX: Final[tuple[str, ...]] = b390.SIGNED_SIX


def to_points(*, frame: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Return a forecast in MW as points of effective capacity, with `lead_hours`."""
    return (
        frame.join(series.select("time_series_id", "capacity_mw"), on="time_series_id")
        .with_columns(pl.col(*lv.QUANTILES) / pl.col("capacity_mw") * 100)
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60
        )
        .drop("capacity_mw")
    )


def to_mw(*, frame: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Return a forecast in points as MW, with the key columns and quantiles only."""
    return (
        frame.join(series.select("time_series_id", "capacity_mw"), on="time_series_id")
        .with_columns(pl.col(*lv.QUANTILES) * pl.col("capacity_mw") / 100)
        .select(*KEYS, *lv.QUANTILES)
    )


def actual_points(*, ctx: dict, series: pl.DataFrame) -> pl.DataFrame:
    """Return the actuals in points: `time_series_id`, `valid_time`, `actual_points`."""
    return (
        ctx["actuals"]
        .join(series.select("time_series_id", "capacity_mw"), on="time_series_id")
        .select(
            "time_series_id",
            "valid_time",
            actual_points=pl.col("actual") / pl.col("capacity_mw") * 100,
        )
    )


def streams(*, pts: pl.DataFrame, actuals: pl.DataFrame) -> dict[int, np.ndarray]:
    """Return each Primary's residual stream on the grid."""
    out = {}
    for sid in pts["time_series_id"].unique().sort().to_list():
        out[sid] = lv.residual_stream(
            forecast=pts.filter(pl.col("time_series_id") == sid),
            actual_points=actuals.filter(pl.col("time_series_id") == sid).drop("time_series_id"),
        )
    return out


def k1(*, pts: pl.DataFrame, streams_: dict[int, np.ndarray]) -> pl.DataFrame:
    """Return the stack's filter (K1) added to the bundle's quantiles, in points."""
    settings = run377.settings_of_hyper(hyper=run377.noise_hyper(noise="pool", site="any"))
    pieces = []
    for sid, residual in streams_.items():
        part = pts.filter(pl.col("time_series_id") == sid)
        correction = lv.stack_filter_correction(residual=residual, cases=part, settings=settings)
        pieces.append(part.with_columns(*[pl.col(q) + pl.Series(correction) for q in lv.QUANTILES]))
    return pl.concat(pieces)


def k_one_state(
    *,
    pts: pl.DataFrame,
    actuals: pl.DataFrame,
    streams_: dict[int, np.ndarray],
    pooled: bool,
    labels: dict[int, str],
) -> tuple[pl.DataFrame, list[dict]]:
    """Return K2 (per Primary) or K3 (pooled) corrected quantiles in points, and the settings."""
    months = pts["power_fcst_init_time"].dt.truncate("1mo").unique().sort().to_list()
    settings: list[dict] = []
    corrected = []
    for month in months:
        start = int(common.grid_index(times=month))
        fits: dict[int, tuple[float, float, int]] = {}
        train_by_sid, resid_by_sid = {}, {}
        for sid, residual in streams_.items():
            masked = residual.copy()
            masked[start:] = np.nan
            resid_by_sid[sid] = masked
            train_by_sid[sid] = (
                pts.filter(pl.col("time_series_id") == sid, pl.col("valid_time") < month)
                .join(
                    actuals.filter(pl.col("time_series_id") == sid).drop("time_series_id"),
                    on="valid_time",
                )
                .select("power_fcst_init_time", "valid_time", "q50", "lead_hours", "actual_points")
            )
        if pooled:
            shared = lv.fit_gain_and_decay(
                residuals=list(resid_by_sid.values()), training=list(train_by_sid.values())
            )
            fits = dict.fromkeys(streams_, shared)
        else:
            for sid in streams_:
                fits[sid] = lv.fit_gain_and_decay(
                    residuals=[resid_by_sid[sid]], training=[train_by_sid[sid]]
                )
        for sid, residual in streams_.items():
            gain, decay, n_train = fits[sid]
            settings.append(
                {"label": labels[sid], "month": month.strftime("%Y-%m"), "gain": gain,
                 "decay": decay, "n_train_cases": n_train}
            )  # fmt: skip
            part = pts.filter(
                pl.col("time_series_id") == sid,
                pl.col("power_fcst_init_time").dt.truncate("1mo") == month,
            )
            if not part.height:
                continue
            positions, level = lv.one_state_level(residual=residual, gain=gain)
            cutoff, target = lv.case_positions(cases=part)
            correction = lv.level_correction(
                positions=positions, level=level, cutoff=cutoff, target=target, decay=decay
            )
            corrected.append(
                part.with_columns(*[pl.col(q) + pl.Series(correction) for q in lv.QUANTILES])
            )
        print(f"  {'K3' if pooled else 'K2'} {month:%Y-%m} done", flush=True)
    return pl.concat(corrected), settings


def main() -> None:
    """Build every Part A arm, score it, and write the tables."""
    WORK.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    labels = dict(zip(series["time_series_id"].to_list(), series["label"].to_list(), strict=True))
    ctx = s381.build_context(series=series)
    actuals = actual_points(ctx=ctx, series=series)
    print("loading the stored forecasts")
    bundle_mw = s381.arm_frame(regime="weekly", arm="bundle_solar+sunny+hl1", series=series)
    pts = to_points(frame=bundle_mw, series=series)
    streams_ = streams(pts=pts, actuals=actuals)
    print("K1")
    arms_pts = {"K0": pts, "K1": k1(pts=pts, streams_=streams_)}
    print("K2")
    arms_pts["K2"], set2 = k_one_state(
        pts=pts, actuals=actuals, streams_=streams_, pooled=False, labels=labels
    )
    print("K3")
    arms_pts["K3"], set3 = k_one_state(
        pts=pts, actuals=actuals, streams_=streams_, pooled=True, labels=labels
    )
    (WORK / "filter_settings.json").write_text(json.dumps({"K2": set2, "K3": set3}, indent=1))
    arms = {}
    for name, frame in arms_pts.items():
        arms[name] = to_mw(frame=frame, series=series)
        arms[f"{name}_w"] = to_mw(frame=lv.widen(frame=frame, factor=WIDEN), series=series)
    core = {
        **arms,
        "stack": base.stack_frames(series=series)["stack"],
        **base.reference_frames(series=series),
        "bg2_only_W": s381.stored_frame(name="only_W", series=series),
    }
    table, _, _ = s381.score_arms(arms=core, series=series, ctx=ctx)
    table.write_parquet(WORK / "slice_table.parquet")
    events = b390.event_table(cache_events=ctx["events"], series=series)
    events.write_parquet(WORK / "events.parquet")
    b390.row_event_flags(keys=table.select(*KEYS), events=events, series=series).write_parquet(
        WORK / "event_flags.parquet"
    )
    for name, frame in core.items():
        rows = b390.trimmed_rows(name=name, frame=frame, table=table, ctx=ctx, series=series)
        rows.write_parquet(WORK / f"rows_{name}.parquet")
        print(f"wrote rows_{name}: {rows.height}")
    # K4 and the six-Primary references.
    six = series.filter(pl.col("label").is_in(list(SIX)))
    off_mw = b390.kalman_off_frame(series=series)
    off_pts = to_points(frame=off_mw, series=six)
    off_streams = streams(pts=off_pts, actuals=actuals)
    k4_pts, set4 = k_one_state(
        pts=off_pts, actuals=actuals, streams_=off_streams, pooled=False, labels=labels
    )
    (WORK / "filter_settings_k4.json").write_text(json.dumps(set4, indent=1))
    six_arms = {
        "stack_off": off_mw,
        "K4": to_mw(frame=k4_pts, series=six),
        "K4_w": to_mw(frame=lv.widen(frame=k4_pts, factor=WIDEN), series=six),
        "stack": core["stack"].filter(pl.col("time_series_id").is_in(six["time_series_id"])),
        "K0": arms["K0"].filter(pl.col("time_series_id").is_in(six["time_series_id"])),
        "K2": arms["K2"].filter(pl.col("time_series_id").is_in(six["time_series_id"])),
    }
    table6, _, _ = s381.score_arms(arms=six_arms, series=series, ctx=ctx)
    table6.write_parquet(WORK / "slice_table_six.parquet")
    for name, frame in six_arms.items():
        rows = b390.trimmed_rows(name=name, frame=frame, table=table6, ctx=ctx, series=series)
        rows.write_parquet(WORK / f"rows6_{name}.parquet")
        print(f"wrote rows6_{name}: {rows.height}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

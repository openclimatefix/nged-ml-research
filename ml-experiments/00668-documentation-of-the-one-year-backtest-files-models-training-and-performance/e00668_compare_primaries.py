"""Score the stored Primary forecasts beside the one-year backtest, on common rows, by the protocol.

PYTHONPATH=<folder of 377>:<folder of 378> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00668-*/e00668_compare_primaries.py

Arms (all 16 Primaries, quantiles q01 to q99, leads to 240 hours): Primary demand research default
3 and default 1 and the stack without the solar summary (experiment 378), and BG2, the manual
heuristic, and climatology (experiment 87). Every stored forecast has a 09:00 UTC origin. It is
relabelled to the 12:00 UTC origin of the same day, with the lead 3 hours shorter, and joined to the
backtest on series and valid time. The slices, the twCRPS thresholds, the shared-row rule, and the
losses are those of `studies.eval_protocol`, unchanged. The backtest file `primaries.parquet`, when
it exists, joins as the arms `backtest_NRA` and `backtest_prevailing`.

Writes `<backtest folder>/scores_668/primaries_compare/` (private): `rows.parquet`, the per-row
losses of every arm, and `scores.csv`, the pooled means and day-block bootstrap intervals.
"""

from datetime import UTC, datetime, timedelta

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00378_common as common  # ty: ignore[unresolved-import]
import e00668_score_lib as lib
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics

KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
Q = list(common.QUANTILES)
FIRST_VALID = datetime(2025, 11, 1, 12, tzinfo=UTC)
ORIGIN_SHIFT = timedelta(hours=3)
CACHE_DIR = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
GROUPS = {
    "signed-MW six": ["D5", "D9", "D10", "D11", "D14", "D16"],
    "MVA-only seven": ["D2", "D3", "D4", "D12", "D19", "D21", "D22"],
}
SLICES = [
    "all",
    "lead_day_1",
    "lead_day_2",
    "lead_days_3_4",
    "lead_days_5_7",
    "lead_days_8_10",
    "winter",
    "winter_nights",
    "cold_nights",
    "economy_7_step",
    "christmas",
    "bank_holidays",
    "sunny_unit_shape",
]
METRICS = ["mae_median", "crps", "twcrps"]


def stored_arms(*, site: str) -> dict[str, pl.DataFrame]:
    """Return the three experiment-378 arms of one site in points of capacity, leads to 240 h."""
    path = common.stage_path(site=site, stage="dp_preds_s2")
    if site == "D8":
        path = path.with_name("stack_d_donor_D8.parquet")
    base = pl.read_parquet(path).filter(
        (pl.col("seed") == 0) & (pl.col("lead_hours") <= common.HORIZON_HOURS)
    )
    source = "control" if site == "D15" else common.SOURCE_OF_SITE[site]
    low = pl.read_parquet(common.ARMS_DIR / f"low_lead_{source}.parquet").filter(
        pl.col("site") == site
    )
    long = pl.read_parquet(common.ARMS_DIR / f"long_lead_{source}.parquet").filter(
        pl.col("site") == site
    )
    time_keys = KEYS[1:]
    low_x1 = low.select(*time_keys, *[pl.col(f"x1_{q}").alias(f"low_{q}") for q in Q])
    long_d3 = long.select(*time_keys, *[pl.col(f"d3_{q}") for q in Q])
    joined = base.join(low_x1, on=time_keys, how="left").join(long_d3, on=time_keys, how="left")
    short = pl.col("lead_hours") <= common.STORED_HORIZON_HOURS
    out = {"stack_without_s": base.select(*KEYS, *Q)}
    out["default_1"] = joined.select(
        *KEYS, *[pl.when(short).then(pl.col(f"low_{q}")).otherwise(pl.col(q)).alias(q) for q in Q]
    )
    out["default_3"] = joined.select(
        *KEYS,
        *[pl.when(short).then(pl.col(f"low_{q}")).otherwise(pl.col(f"d3_{q}")).alias(q) for q in Q],
    )
    return {n: f.drop_nulls(subset=Q) for n, f in out.items()}


def relabel(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return a stored forecast with its 09:00 origin relabelled 12:00 and the lead 3 hours less."""
    return frame.with_columns(
        power_fcst_init_time=pl.col("power_fcst_init_time") + ORIGIN_SHIFT
    ).filter(
        pl.col("valid_time") > pl.col("power_fcst_init_time"),
        pl.col("valid_time") >= FIRST_VALID,
    )


def main() -> None:
    """Build every arm, score on the shared rows, and write the tables."""
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=metrics.HOLDOUT_START,
    )
    events = adapter.events_from_masks(cache=cache, series=series)
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}),
        series=series,
        before=datetime(2025, 7, 1, tzinfo=UTC),
    )
    per_arm: dict[str, list[pl.DataFrame]] = {
        "default_3": [],
        "default_1": [],
        "stack_without_s": [],
    }
    for site in common.SOURCE_OF_SITE:
        one = series.filter(pl.col("label") == site)
        for name, stored in stored_arms(site=site).items():
            per_arm[name].append(adapter.forecast_frame(stored=stored, series=one, unit="points"))
    forecasts = {name: pl.concat(frames) for name, frames in per_arm.items()}
    for arm, stored_name in (
        ("bg2", "p_s6_bg2"),
        ("climatology", "ref_climatology"),
        ("manual_heuristic", "ref_manual_heuristic"),
    ):
        stored = pl.read_parquet(
            adapter.stored_path(
                root=common.ML_EXPERIMENTS_DATA_DIR,
                pattern=f"00087*/predictions/{stored_name}/predictions.parquet",
            )
        ).filter(pl.col("time_series_id").is_in(series["time_series_id"].to_list()))
        forecasts[arm] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    forecasts = {name: relabel(frame=frame) for name, frame in forecasts.items()}
    backtest = lib.FOLDER / "primaries.parquet"
    if backtest.exists():
        file = pl.read_parquet(backtest)
        ids = pl.Series(series["time_series_id"]).cast(pl.Int8)
        for kind in ("NRA", "prevailing"):
            part = file.filter(pl.col("NRA_or_prevailing") == kind).filter(
                pl.col("time_series_id").is_in(ids.to_list())
            )
            frame = adapter.forecast_frame(stored=part, series=series, unit="mw")
            forecasts[f"backtest_{kind}"] = frame.filter(pl.col("valid_time") >= FIRST_VALID)
    table = metrics.shared_slice_table(
        forecasts=forecasts,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print("arms", list(forecasts), "shared half-hours", table.height, flush=True)
    out = lib.FOLDER / "scores_668" / "primaries_compare"
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for name, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        rows = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        ).with_columns(arm=pl.lit(name))
        rows_out.append(rows.select(*KEYS, "label", "arm", *METRICS, *SLICES))
    all_rows = pl.concat(rows_out)
    all_rows.write_parquet(out / "rows.parquet")
    day = (all_rows["power_fcst_init_time"].dt.epoch("s") // 86400).to_numpy()
    arms = all_rows["arm"].unique(maintain_order=True).to_list()
    records = []
    for group_name, labels in GROUPS.items():
        in_group = all_rows["label"].is_in(labels).to_numpy()
        for slice_name in SLICES:
            mask = in_group & all_rows[slice_name].to_numpy()
            for metric in METRICS:
                for arm in arms:
                    pick = mask & (all_rows["arm"] == arm).to_numpy()
                    if not pick.any():
                        continue
                    v = all_rows[metric].to_numpy().astype(np.float64)[pick]
                    draws = lib.day_bootstrap(day=day[pick], values={arm: v})[arm]
                    records.append(
                        {
                            "group": group_name,
                            "slice": slice_name,
                            "metric": metric,
                            "arm": arm,
                            "mean": float(v.mean()),
                            "lo": float(np.quantile(draws, 0.025)),
                            "hi": float(np.quantile(draws, 0.975)),
                            "half_hours": int(pick.sum()),
                        }
                    )
    pl.DataFrame(records).write_csv(out / "scores.csv")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

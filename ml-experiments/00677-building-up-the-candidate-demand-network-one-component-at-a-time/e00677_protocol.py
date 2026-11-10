# ruff: noqa: D103, E501
"""Experiment 677: score the ladder, the stored standard methods, and experiment 667's file by the protocol.

Usage: python e00677_protocol.py TAG STEP[:FILE_SPAN:SEEDS] ...   (for example `3 1 2 3b 4 5w 8w 8wo`)

Builds the arms of `e00668_compare_primaries` (Primary demand research default 3 as "GAM", BG2 as
"XGBoost", the manual heuristic, climatology, and experiment 667's NRA output), adds one arm per ladder
step (the mean of its seeds' quantiles, sorted within each row), and scores every arm on the rows that all
arms share, with `studies.eval_protocol.metrics`: MAE of the median, CRPS, and twCRPS, by the slices
of the protocol (lead days, winter, cold nights, the Economy 7 step, sunny shape, bank holidays) for the six
MW and the seven MVA Primaries separately. Steps 1 to 4 use seeds 0 to 2 and steps 5 to 8 use seeds 0 to 4
unless the argument gives `STEP:SPAN:SEEDS` (for example `8w:1:012` for monthly refits).
Writes `scores/protocol_<TAG>/{rows.parquet,scores.csv}` (private).
"""

import sys
from datetime import UTC, datetime

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00378_common as common  # ty: ignore[unresolved-import]
import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00677_export as exp
import e00677_score as sc
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics

RENAME = {
    "default_3": "GAM",
    "bg2": "XGBoost",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
    "backtest_NRA": "Experiment 667 (monthly refits)",
}


def main() -> None:
    tag = sys.argv[1]
    args = sys.argv[2:]
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=cp.CACHE_DIR)
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
    per_arm: dict[str, list[pl.DataFrame]] = {"default_3": []}
    for site in common.SOURCE_OF_SITE:
        one = series.filter(pl.col("label") == site)
        per_arm["default_3"].append(
            adapter.forecast_frame(
                stored=cp.stored_arms(site=site)["default_3"], series=one, unit="points"
            )
        )
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
    forecasts = {name: cp.relabel(frame=frame) for name, frame in forecasts.items()}
    ids = pl.Series(series["time_series_id"]).cast(pl.Int8).to_list()
    file = pl.read_parquet(lib.FOLDER / "primaries.parquet").filter(
        (pl.col("NRA_or_prevailing") == "NRA") & pl.col("time_series_id").is_in(ids)
    )
    forecasts["backtest_NRA"] = adapter.forecast_frame(
        stored=file, series=series, unit="mw"
    ).filter(pl.col("valid_time") >= cp.FIRST_VALID)
    forecasts = {RENAME[k]: v for k, v in forecasts.items()}
    for arg in args:
        step, _, rest = arg.partition(":")
        span, seeds = (
            (int(rest.split(":")[0]), tuple(int(c) for c in rest.split(":")[1]))
            if rest
            else (3, (0, 1, 2, 3, 4) if step[0] in "5678" else (0, 1, 2))
        )
        name = f"step{step}_{exp.VERSION[step]}_sp{span}"
        label = f"Step {step}" + (" monthly" if span == 1 else "")
        mean = sc.seed_mean(name=name, seeds=seeds).filter(pl.col("time_series_id").is_in(ids))
        forecasts[label] = adapter.forecast_frame(stored=mean, series=series, unit="mw").filter(
            pl.col("valid_time") >= cp.FIRST_VALID
        )
        print("built", label, forecasts[label].height, flush=True)
    table = metrics.shared_slice_table(
        forecasts=forecasts,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print("arms", list(forecasts), "shared half-hours", table.height, flush=True)
    out = sc.ROOT / "scores" / f"protocol_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    signed = series.filter(pl.col("label").is_in(cp.GROUPS["signed-MW six"]))
    flows = actuals.rename({"actual": "flow_mw"})
    cost_inputs: dict[str, pl.DataFrame] = {}
    for name, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        cost_inputs[name] = kept.filter(
            pl.col("time_series_id").is_in(signed["time_series_id"].to_list())
        ).join(actuals, on=["time_series_id", "valid_time"])
        rows = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        ).with_columns(arm=pl.lit(name))
        rows_out.append(rows.select(*cp.KEYS, "label", "arm", *cp.METRICS, *cp.SLICES))
        del kept, rows
    all_rows = pl.concat(rows_out)
    all_rows.write_parquet(out / "rows.parquet")
    day = (all_rows["power_fcst_init_time"].dt.epoch("s") // 86400).to_numpy()
    arms = all_rows["arm"].unique(maintain_order=True).to_list()
    records = []
    for group_name, labels in cp.GROUPS.items():
        in_group = all_rows["label"].is_in(labels).to_numpy()
        for slice_name in cp.SLICES:
            mask = in_group & all_rows[slice_name].to_numpy()
            for metric in cp.METRICS:
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
    costs = []
    for fraction in (0.5, 0.9):
        cost = metrics.avoidable_curtailment_cost(
            forecasts=cost_inputs,
            series=signed,
            flows=flows,
            scored_start=datetime(2025, 7, 1, tzinfo=UTC),
            fraction=fraction,
        )
        costs.append(cost.with_columns(limit_fraction=pl.lit(fraction)))
    pl.concat(costs).write_csv(out / "curtailment_cost.csv")
    print("wrote", out)


if __name__ == "__main__":
    main()

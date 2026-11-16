# ruff: noqa: D103, E501
"""Experiment 700: score the origin months 2025-11 to 2026-06 of the GAM and XGBoost beside steps 5 and 8, by the protocol.

    PYTHONPATH=<folders of experiments 377, 378, 668, 677> E00677_ROOT=<this experiment's private folder> \
        uv run --no-sync python ml-experiments/00700-*/e00700_protocol.py

Adapts `e00677_protocol.py`: every arm is a file of the layout of experiment 677 (12:00 UTC contract
origin, 672 half-hours, megawatts), and the arms are scored on the rows that all of them share with
`studies.eval_protocol.metrics` (MAE of the median, CRPS, twCRPS, by the slices of the protocol),
for the six signed-MW and the seven MVA-only Primaries. The manual heuristic and climatology are the
stored arms of experiment 87 (`work/arms/ref_*`, all 365 daily runs), relabelled from the 09:00 to the
12:00 origin. Writes `scores/protocol_year/{rows.parquet,scores.csv}` and `curtailment_cost.csv`.
"""

from datetime import UTC, datetime

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00677_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

START = datetime(2025, 11, 1, tzinfo=UTC)
END = datetime(2026, 7, 1, tzinfo=UTC)
FIRST_VALID = datetime(2025, 11, 1, 12, tzinfo=UTC)
FILES = {
    "GAM": ("gam_year_sp1", (0,)),
    "XGBoost BG2 (weekly W member)": ("xgb_bg2_year_sp1", (0,)),
    "XGBoost with monthly refits only": ("xgb_monthly_year_sp1", (0,)),
    "Step 5 (quarterly refits)": ("step5_year_sp3", (0, 1, 2)),
    "Step 8 (quarterly refits)": ("step8_year_sp3", (0, 1, 2)),
    "Step 5 monthly (9a)": ("step5_year_sp1", (0, 1, 2)),
    "Step 8 monthly (9b)": ("step8_year_sp1", (0, 1, 2)),
}
REFERENCES = {"Manual heuristic": "ref_manual_heuristic", "Climatology": "ref_climatology"}


def main() -> None:
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
    ids = pl.Series(series["time_series_id"]).cast(pl.Int8).to_list()
    forecasts: dict[str, pl.DataFrame] = {}
    for label, (name, seeds) in FILES.items():
        mean = sc.seed_mean(name=name, seeds=seeds).filter(pl.col("time_series_id").is_in(ids))
        forecasts[label] = adapter.forecast_frame(stored=mean, series=series, unit="mw")
        print("built", label, forecasts[label].height, flush=True)
    label_of = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for label, arm in REFERENCES.items():
        stored = (
            pl.scan_parquet(
                next(ML_EXPERIMENTS_DATA_DIR.glob("00087-*")) / "work" / "arms" / f"{arm}.parquet"
            )
            .filter(
                (pl.col("group") == "Primary")
                & (pl.col("power_fcst_init_time") >= START)
                & (pl.col("power_fcst_init_time") < END)
            )
            .collect()
            .with_columns(
                time_series_id=pl.col("site").replace_strict(label_of, return_dtype=pl.Int32)
            )
        )
        frame = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
        forecasts[label] = cp.relabel(frame=frame).filter(pl.col("valid_time") >= FIRST_VALID)
    table = metrics.shared_slice_table(
        forecasts=forecasts,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print("arms", list(forecasts), "shared half-hours", table.height, flush=True)
    out = sc.ROOT / "scores" / "protocol_year"
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for name, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        rows = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        ).with_columns(arm=pl.lit(name))
        rows_out.append(rows.select(*cp.KEYS, "label", "arm", *cp.METRICS, *cp.SLICES))
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
    print("wrote", out)


if __name__ == "__main__":
    main()

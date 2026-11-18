# ruff: noqa: D103, E501
"""Experiment 701: score the GAM and XGBoost beside steps 5 and 8, for origins 2026-07 to 2026-10.

The scoring follows the protocol of experiments 677 and 700.

    PYTHONPATH=<experiment folders> \
        uv run --no-sync python ml-experiments/00701-*/e00701_protocol.py [--smoke]

Adapts `e00700_protocol.py`: every arm is a file of the layout of experiment 677 (12:00 UTC contract
origin, 672 half-hours, megawatts), and the nine arms are scored on the rows that all of them share
with `studies.eval_protocol.metrics` (MAE of the median, CRPS, twCRPS, by the slices of the protocol).
Four things differ from experiment 700:

- **Dates.** Origins run from 2026-07-01 to 2026-10-31, so valid times reach 2026-11-14 12:00. The
  scorer moves the holdout of `studies.eval_protocol.metrics` (its constant and the keyword
  defaults of `row_losses` and `shared_slice_table`) from 2026-07-01 to 2026-11-15, in memory only.
- **Inputs.** The feature cache is the GAM side's extended cache (`e00701_gam_extend`), which holds
  the actual power and the ERA5 temperature to 2026-11-14 23:30. The twCRPS thresholds still use the
  history before 2025-07-01.
- **Files.** The GAM, XGBoost, and ladder files are those of `backtests/` under this experiment's
  private folder (`gam_final_test_sp1`, `xgb_bg2_year_sp1`, `xgb_monthly_year_sp1`, and
  `step{5,8}_final_sp{3,1}` from `e00701_ladder.py`).
- **References.** The manual heuristic and climatology are `refs/arms/ref_*.parquet` of
  `e00701_refs.py`, relabelled from the 09:00 to the 12:00 origin as in experiment 700.

`--smoke` scores the origins 2026-06-01 to 2026-06-30 from experiment 700's own files (read only) and
experiment 87's stored reference arms, and writes to `scores/smoke_protocol_final/`, so the scorer
can be checked against experiment 700's rows: the run ends by comparing every metric and slice flag
of the rows both runs hold, key by key.

Writes `scores/protocol_final/{rows.parquet,scores.csv}`.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00677_score as sc  # ty: ignore[unresolved-import]
import e00701_gam_extend as ext
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics

ROOT: Final[Path] = ext.ROOT
EXPERIMENT_700: Final[Path] = (
    ext.REAL_DATA / "ml-experiments" / "00700-timing-run-gam-and-xgboost-14-day-monthly-refit"
)
EXPERIMENT_87: Final[Path] = next((ext.REAL_DATA / "ml-experiments").glob("00087-*"))
HOLDOUT: Final[datetime] = ext.ROWS_END_EXCLUSIVE
"""No valid time on or after this instant is scored; the extended cache holds none either."""

metrics.HOLDOUT_START = HOLDOUT  # ty: ignore[invalid-assignment]
for _function in (metrics.row_losses, metrics.shared_slice_table):
    _defaults = _function.__kwdefaults__
    assert _defaults is not None
    _defaults["holdout_start"] = HOLDOUT

ARM_NAMES: Final[tuple[str, ...]] = (
    "GAM",
    "XGBoost BG2 (weekly W member)",
    "XGBoost with monthly refits only",
    "Step 5 (quarterly refits)",
    "Step 8 (quarterly refits)",
    "Step 5 monthly (9a)",
    "Step 8 monthly (9b)",
    "Manual heuristic",
    "Climatology",
)
"""The nine arms, named exactly as in experiment 700's `rows.parquet`."""
REFERENCES: Final[dict[str, str]] = {
    "Manual heuristic": "ref_manual_heuristic",
    "Climatology": "ref_climatology",
}


@dataclass(frozen=True)
class Setting:
    """What a scoring run reads and writes."""

    start: datetime
    end: datetime
    first_valid: datetime
    backtests: Path
    files: dict[str, tuple[str, tuple[int, ...]]]
    reference_arms: Path
    out: Path


FINAL = Setting(
    start=datetime(2026, 7, 1, tzinfo=UTC),
    end=datetime(2026, 11, 1, tzinfo=UTC),
    first_valid=datetime(2026, 7, 1, 12, tzinfo=UTC),
    backtests=ROOT / "backtests",
    files={
        "GAM": ("gam_final_test_sp1", (0,)),
        "XGBoost BG2 (weekly W member)": ("xgb_bg2_year_sp1", (0,)),
        "XGBoost with monthly refits only": ("xgb_monthly_year_sp1", (0,)),
        "Step 5 (quarterly refits)": ("step5_final_sp3", (0, 1, 2)),
        "Step 8 (quarterly refits)": ("step8_final_sp3", (0, 1, 2)),
        "Step 5 monthly (9a)": ("step5_final_sp1", (0, 1, 2)),
        "Step 8 monthly (9b)": ("step8_final_sp1", (0, 1, 2)),
    },
    reference_arms=ROOT / "refs" / "arms",
    out=ROOT / "scores" / "protocol_final",
)
SMOKE = Setting(
    start=datetime(2026, 6, 1, tzinfo=UTC),
    end=datetime(2026, 7, 1, tzinfo=UTC),
    first_valid=datetime(2026, 6, 1, 12, tzinfo=UTC),
    backtests=EXPERIMENT_700 / "backtests",
    files={
        "GAM": ("gam_year_sp1", (0,)),
        "XGBoost BG2 (weekly W member)": ("xgb_bg2_year_sp1", (0,)),
        "XGBoost with monthly refits only": ("xgb_monthly_year_sp1", (0,)),
        "Step 5 (quarterly refits)": ("step5_year_sp3", (0, 1, 2)),
        "Step 8 (quarterly refits)": ("step8_year_sp3", (0, 1, 2)),
        "Step 5 monthly (9a)": ("step5_year_sp1", (0, 1, 2)),
        "Step 8 monthly (9b)": ("step8_year_sp1", (0, 1, 2)),
    },
    reference_arms=EXPERIMENT_87 / "work" / "arms",
    out=ROOT / "scores" / "smoke_protocol_final",
)


def seed_mean(*, path: Path, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return a ladder or model file with each quantile averaged over `seeds` and sorted within the row.

    The same as `e00677_score.seed_mean`, which reads its folder from an environment variable, but
    for a file path.
    """
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    raw = pl.read_parquet(path).filter(pl.col("seed").is_in(list(seeds)))
    assert set(raw["seed"].unique().to_list()) == set(seeds), (path.name, seeds)
    assert raw.group_by(keys).len()["len"].unique().to_list() == [len(seeds)], (
        "a seed lacks some rows"
    )
    mean = raw.group_by(keys).agg(pl.col(list(sc.QCOLS)).mean()).sort(keys)
    q = np.sort(mean.select(list(sc.QCOLS)).to_numpy().astype(np.float32), axis=1)
    return mean.with_columns([pl.Series(c, q[:, i]) for i, c in enumerate(sc.QCOLS)])


def load_inputs() -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict, pl.DataFrame
]:
    """Return the series table, actuals, context, events, first readings, and twCRPS thresholds.

    The actuals and context come from the extended feature cache, from 2024-04-01 to the holdout.
    """
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=ext.CACHE_DIR)
    assert cache.end_times[-1] == np.datetime64(ext.GRID_END.replace(tzinfo=None)), cache.end_times[
        -1
    ]
    actuals, context = adapter.actuals_and_context(
        cache=cache, series=series, first=datetime(2024, 4, 1, tzinfo=UTC), last=HOLDOUT
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
    return series, actuals, context, events, first_readings, thresholds


def load_forecasts(*, setting: Setting, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the nine arms' forecast frames (megawatts) for the origins of the setting."""
    ids = pl.Series(series["time_series_id"]).cast(pl.Int8).to_list()
    forecasts: dict[str, pl.DataFrame] = {}
    for label, (name, seeds) in setting.files.items():
        mean = seed_mean(path=setting.backtests / f"{name}.parquet", seeds=seeds).filter(
            pl.col("time_series_id").is_in(ids)
        )
        forecasts[label] = adapter.forecast_frame(stored=mean, series=series, unit="mw")
        print("built", label, forecasts[label].height, flush=True)
    label_of = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for label, arm in REFERENCES.items():
        stored = (
            pl.scan_parquet(setting.reference_arms / f"{arm}.parquet")
            .filter(
                (pl.col("group") == "Primary")
                & (pl.col("power_fcst_init_time") >= setting.start)
                & (pl.col("power_fcst_init_time") < setting.end)
            )
            .collect()
            .with_columns(
                time_series_id=pl.col("site").replace_strict(label_of, return_dtype=pl.Int32)
            )
        )
        frame = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
        forecasts[label] = cp.relabel(frame=frame).filter(
            pl.col("valid_time") >= setting.first_valid
        )
        print("built", label, forecasts[label].height, flush=True)
    assert list(forecasts) == [*setting.files, *REFERENCES]
    return forecasts


def score_records(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the pooled scores by group, slice, metric, and arm, with day-bootstrap 95% intervals."""
    day = (rows["power_fcst_init_time"].dt.epoch("s") // 86400).to_numpy()
    arms = rows["arm"].unique(maintain_order=True).to_list()
    records = []
    for group_name, labels in cp.GROUPS.items():
        in_group = rows["label"].is_in(labels).to_numpy()
        for slice_name in cp.SLICES:
            mask = in_group & rows[slice_name].to_numpy()
            for metric in cp.METRICS:
                for arm in arms:
                    pick = mask & (rows["arm"] == arm).to_numpy()
                    if not pick.any():
                        continue
                    v = rows[metric].to_numpy().astype(np.float64)[pick]
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
    return pl.DataFrame(records)


def main(*, setting: Setting) -> None:
    series, actuals, context, events, first_readings, thresholds = load_inputs()
    forecasts = load_forecasts(setting=setting, series=series)
    table = metrics.shared_slice_table(
        forecasts=forecasts,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print("arms", list(forecasts), "shared half-hours", table.height, flush=True)
    print("shared half-hours per Primary (a Primary with few rows is partly missing from an arm):")
    print(table.group_by("time_series_id").len().sort("time_series_id"), flush=True)
    setting.out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for name, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        rows = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        ).with_columns(arm=pl.lit(name))
        rows_out.append(rows.select(*cp.KEYS, "label", "arm", *cp.METRICS, *cp.SLICES))
    all_rows = pl.concat(rows_out)
    all_rows.write_parquet(setting.out / "rows.parquet")
    score_records(rows=all_rows).write_csv(setting.out / "scores.csv")
    print("wrote", setting.out)


def compare_with_experiment_700() -> None:
    """Print how far the smoke rows are from experiment 700's rows of the same keys and arms.

    Experiment 700 scored eight months at once, so its shared rows and the per-Primary cut of the
    `sunny_unit_shape` slice differ slightly. The metrics and every other slice flag are properties
    of one row and must agree exactly.
    """
    keys = [*cp.KEYS, "arm"]
    new = pl.read_parquet(SMOKE.out / "rows.parquet")
    old = pl.read_parquet(EXPERIMENT_700 / "scores" / "protocol_year" / "rows.parquet").filter(
        (pl.col("power_fcst_init_time") >= SMOKE.start)
        & (pl.col("power_fcst_init_time") < SMOKE.end)
    )
    joined = new.join(old, on=keys, how="full", coalesce=True, suffix="_700")
    only_new = joined.filter(pl.col("label_700").is_null()).height
    only_old = joined.filter(pl.col("label").is_null()).height
    both = joined.drop_nulls(subset=["label", "label_700"])
    print(
        f"smoke rows {new.height:,}, experiment 700 rows of June origins {old.height:,}, shared keys "
        f"{both.height:,}; only in smoke {only_new:,}, only in experiment 700 {only_old:,}"
    )
    for column in [*cp.METRICS, *[s for s in cp.SLICES if s != "sunny_unit_shape"], "label"]:
        left, right = both[column], both[f"{column}_700"]
        differing = (
            int((left != right).sum())
            if left.dtype != pl.Float64
            else int(((left - right).abs() > 0).sum())
        )
        gap = (
            f", largest difference {np.nanmax(np.abs(left.to_numpy() - right.to_numpy())):.3g}"
            if left.dtype == pl.Float64
            else ""
        )
        print(f"  {column:18s} rows that differ: {differing:,}{gap}")
    flag = int((both["sunny_unit_shape"] != both["sunny_unit_shape_700"]).sum())
    print(f"  sunny_unit_shape   rows that differ: {flag:,} (a per-Primary cut of the shared rows)")
    for arm in ARM_NAMES:
        count = both.filter(pl.col("arm") == arm).height
        print(f"  {arm:34s} shared keys {count:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="score June 2026 from experiment 700's files"
    )
    smoke = parser.parse_args().smoke
    main(setting=SMOKE if smoke else FINAL)
    if smoke:
        compare_with_experiment_700()

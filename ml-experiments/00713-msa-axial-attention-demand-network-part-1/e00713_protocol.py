# ruff: noqa: E501
"""Experiment 713: score the arms by the evaluation protocol (design 6.5), with the climatology row.

Usage: e00713_protocol.py TAG ARM[,ARM...] [--seeds ...] [--months 2025-11 ...] [--no-cost]

Adapted from `e00677_protocol.py`. Each arm is the seed mean of its quantiles (sorted within the row)
in the layout of experiment 677: the contract origin (12:00 UTC), 672 half-hours, megawatts. The
arm `C` is the stored Candidate (`e00713_score.load_arm`). The climatology and the manual-heuristic
rows are the stored reference arms of experiment 87, relabelled from the 09:00 to the 12:00 origin.
The climatology is the 51-member climatology of experiment 48's weekly walk-forward run, as experiment 87
recomputed it at 11 levels (`e00087_assemble.py`); it is not refitted per fold, and its training window is
that of experiment 48's run, which this experiment did not re-derive. The row is report-only.
Every arm is scored on the half-hours that all of them share, with `studies.eval_protocol.metrics`:
the mean absolute error of the median, CRPS, threshold-weighted CRPS, and the coverage and pinball
loss at q01, q05, q95, and q99, by the slices of the protocol. The slices include the cold nights, the
winter, the time of day, and the four load-shift scenarios (all rows; outside events and more than 7
days after one; inside events; the 7 days after an event).

The table of each slice and metric gives the mean per Primary group with a 95% interval from
resampling whole origin months. Three further tables are written: the leave-one-Primary-out
contrast of each arm against `C`, the leave-one-month-out contrast, and the avoidable curtailment
cost of the signed-MW six at the scaled limits 0.9 and 0.5, in the three row sets of the common brief
(inside events, the 7 days after, and all rows).

Writes `score/protocol_<TAG>/{rows.parquet,scores.csv,leave_one_primary.csv,leave_one_month.csv,
curtailment_cost.csv}` in the experiment's data folder.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
import e00713_score as score
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCENARIOS: Final = {
    "all rows": "all",
    "outside events": "outside_events",
    "inside events": "event_rows",
    "7 days after an event": "event_following_7_days",
}
SLICES: Final = (
    "all", "lead_day_1", "lead_day_2", "lead_days_3_4", "lead_days_5_7", "lead_days_8_10",
    "lead_days_11_14", "winter", "winter_nights", "cold_nights", "economy_7_step", "christmas",
    "bank_holidays", "sunny_unit_shape", "time_night", "time_morning", "time_midday",
    "time_evening", "event_rows", "event_following_7_days", "outside_events",
)  # fmt: skip
SCORED_METRICS: Final = (
    "mae_median", "crps", "twcrps", "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99",
    "pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99",
)  # fmt: skip
N_BOOT: Final = 10_000
REFERENCES: Final = {"Climatology": "ref_climatology", "Manual heuristic": "ref_manual_heuristic"}
GROUPS: Final = {
    "signed-MW six": ("D5", "D9", "D10", "D11", "D14", "D16"),
    "MVA-only eight": ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
    "short-history two": ("D15", "D18"),
}


def month_interval(
    *, month: np.ndarray, values: np.ndarray, seed: int = 713
) -> tuple[float, float, float]:
    """Return the mean and the 95% interval of a mean that resamples whole months."""
    labels, inv = np.unique(month, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv).astype(np.float64)
    pick = np.random.default_rng(seed).integers(0, len(labels), size=(N_BOOT, len(labels)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def arm_frame(
    *, arm: str, seeds: tuple[int, ...], months: tuple[str, ...], series: pl.DataFrame
) -> pl.DataFrame:
    """Return an arm's seed mean in megawatts, from the contract origin."""
    mean = score.seed_average(frame=score.load_arm(arm=arm, seeds=seeds, months=months))
    return adapter.forecast_frame(stored=mean, series=series, unit="fraction").filter(
        pl.col("valid_time") >= cp.FIRST_VALID
    )


def main() -> int:
    """Build the arms, score them on the shared half-hours, and write the tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("arms")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="default: seeds 0 to 4 for M*, P, and C, and seeds 0 and 1 for every other arm",
    )
    parser.add_argument("--months", nargs="+", default=list(score.arms.ALL_FOLDS))
    parser.add_argument("--no-cost", action="store_true")
    args = parser.parse_args()
    months = tuple(args.months)
    star = (
        None if args.seeds is not None else score.arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    )  # selection.json is read only for the default seeds

    def seeds_for(arm: str) -> tuple[int, ...]:
        if args.seeds is not None:
            return tuple(args.seeds)
        return score.arms.HEADLINE_SEEDS if arm in {star, "P", "C"} else score.arms.SCREEN_SEEDS

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
    forecasts: dict[str, pl.DataFrame] = {}
    for arm in args.arms.split(","):
        forecasts[arm] = arm_frame(arm=arm, seeds=seeds_for(arm), months=months, series=series)
        print("built", arm, forecasts[arm].height, flush=True)
    ids = series["time_series_id"].to_list()
    for (
        name,
        stored_name,
    ) in (
        REFERENCES.items()
    ):  # stored experiment 87 arms; Climatology is experiment 48's, not refitted per fold
        stored = pl.read_parquet(
            adapter.stored_path(
                root=ML_EXPERIMENTS_DATA_DIR,
                pattern=f"00087*/predictions/{stored_name}/predictions.parquet",
            )
        ).filter(pl.col("time_series_id").is_in(ids))
        frame = cp.relabel(
            frame=adapter.forecast_frame(stored=stored, series=series, unit="fraction")
        )
        forecasts[name] = frame.filter(pl.col("valid_time") >= cp.FIRST_VALID)
    table = metrics.shared_slice_table(
        forecasts=forecasts,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    table = table.with_columns(
        outside_events=~pl.col("event_rows") & ~pl.col("event_following_7_days")
    )
    print("arms", list(forecasts), "shared half-hours", table.height, flush=True)
    out = score.SCORE_DIR / f"protocol_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    kept_rows = []
    cost_inputs: dict[str, pl.DataFrame] = {}
    signed = series.filter(pl.col("label").is_in(GROUPS["signed-MW six"]))
    for name, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        cost_inputs[name] = kept.filter(
            pl.col("time_series_id").is_in(signed["time_series_id"].to_list())
        ).join(actuals, on=["time_series_id", "valid_time"])
        rows = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        ).with_columns(
            arm=pl.lit(name),
            outside_events=~pl.col("event_rows") & ~pl.col("event_following_7_days"),
        )
        kept_rows.append(rows.select(*score.KEYS, "label", "arm", *SCORED_METRICS, *SLICES))
        del kept, rows
    all_rows = pl.concat(kept_rows).with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
    )
    all_rows.write_parquet(out / "rows.parquet")
    write_scores(all_rows=all_rows, out=out)
    write_leave_out(all_rows=all_rows, out=out)
    if not args.no_cost:
        write_cost(cost_inputs=cost_inputs, table=table, signed=signed, actuals=actuals, out=out)
    print("wrote", out)
    return 0


def write_scores(*, all_rows: pl.DataFrame, out: Path) -> None:
    """Write the mean and the month-block interval of every (group, slice, metric, arm)."""
    month = all_rows["month"].to_numpy()
    records = []
    for group, labels in GROUPS.items():
        in_group = all_rows["label"].is_in(list(labels)).to_numpy()
        for slice_name in SLICES:
            mask = in_group & all_rows[slice_name].to_numpy()
            for arm in all_rows["arm"].unique(maintain_order=True).to_list():
                pick = mask & (all_rows["arm"] == arm).to_numpy()
                if not pick.any():
                    continue
                for metric in SCORED_METRICS:
                    v = all_rows[metric].to_numpy().astype(np.float64)[pick]
                    ok = np.isfinite(v)
                    if not ok.any():
                        continue
                    mean, lo, hi = month_interval(month=month[pick][ok], values=v[ok])
                    records.append(
                        {"group": group, "slice": slice_name, "metric": metric, "arm": arm, "mean": mean,
                         "lo": lo, "hi": hi, "half_hours": int(ok.sum())}
                    )  # fmt: skip
    pl.DataFrame(records).write_csv(out / "scores.csv")


def write_leave_out(*, all_rows: pl.DataFrame, out: Path) -> None:
    """Write the CRPS contrast of each arm against `C` with each Primary, then each month, left out."""
    if "C" not in set(all_rows["arm"].to_list()):
        return
    headline = all_rows.filter(pl.col("all"))
    wide = (
        headline.select("arm", "label", "month", "crps")
        .group_by("arm", "label", "month")
        .agg(pl.col("crps").mean(), n=pl.len())
    )
    base = wide.filter(pl.col("arm") == "C").rename({"crps": "crps_c", "n": "n_c"}).drop("arm")
    by_primary, by_month = [], []
    for arm in wide["arm"].unique().to_list():
        if arm == "C":
            continue
        d = wide.filter(pl.col("arm") == arm).join(base, on=["label", "month"])
        d = d.with_columns(diff=pl.col("crps") - pl.col("crps_c"))
        for group, labels in GROUPS.items():
            g = d.filter(pl.col("label").is_in(list(labels)))
            full = float(g["diff"].mean())  # ty: ignore[invalid-argument-type]
            for label in labels:
                rest = g.filter(pl.col("label") != label)
                by_primary.append(
                    {
                        "arm": arm,
                        "group": group,
                        "left_out": label,
                        "all": full,
                        "contrast": float(rest["diff"].mean()),  # ty: ignore[invalid-argument-type]
                    }
                )
            for m in sorted(g["month"].unique().to_list()):
                rest = g.filter(pl.col("month") != m)
                by_month.append(
                    {
                        "arm": arm,
                        "group": group,
                        "left_out": m,
                        "all": full,
                        "contrast": float(rest["diff"].mean()),  # ty: ignore[invalid-argument-type]
                    }
                )
    pl.DataFrame(by_primary).write_csv(out / "leave_one_primary.csv")
    pl.DataFrame(by_month).write_csv(out / "leave_one_month.csv")


def write_cost(
    *,
    cost_inputs: dict[str, pl.DataFrame],
    table: pl.DataFrame,
    signed: pl.DataFrame,
    actuals: pl.DataFrame,
    out: Path,
) -> None:
    """Write the avoidable curtailment cost in the three row sets of the common brief."""
    flows = actuals.rename({"actual": "flow_mw"})
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    pieces = []
    for row_set, flag in (
        ("inside events", "event_rows"),
        ("7 days after", "event_following_7_days"),
        ("all rows", "all"),
    ):
        sel = table.filter(pl.col(flag)).select(keys)
        restricted = {
            name: frame.join(sel, on=keys, how="semi") for name, frame in cost_inputs.items()
        }
        for fraction in (0.9, 0.5):
            try:
                cost = metrics.avoidable_curtailment_cost(
                    forecasts=restricted, series=signed, flows=flows,
                    scored_start=datetime(2025, 7, 1, tzinfo=UTC), fraction=fraction,
                )  # fmt: skip
            except ValueError as err:  # a row set without a node half-hour at the limit has no cost
                print(f"curtailment cost undefined for {row_set!r} at {fraction}: {err}")
                continue
            pieces.append(
                cost.with_columns(limit_fraction=pl.lit(fraction), row_set=pl.lit(row_set))
            )
    if not pieces:
        print("curtailment cost undefined for every row set: no curtailment_cost.csv written")
        return
    pl.concat(pieces).write_csv(out / "curtailment_cost.csv")


if __name__ == "__main__":
    sys.exit(main())

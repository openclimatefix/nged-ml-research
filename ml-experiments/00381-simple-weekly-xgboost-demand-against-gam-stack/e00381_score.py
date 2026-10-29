"""Score experiment 381's XGBoost arms under demand-v1 and print every number the write-up quotes.

Two modes:

- `screen`: the monthly arms (control plus the candidates) against the monthly control. Prints the
  reference gate (the refitted control against the stored monthly member `only_M` of experiment
  379), then each arm's CRPS minus the control's at days 1 to 10, per group (99.86% interval,
  20,000 resamples, the Bonferroni level for k = 7) and per Primary (95%), and the slices the arm
  aims at (Economy 7 step hours, sunny daylight half-hours).
- `final`: the weekly bundle against the stored weekly member `only_W`, the GAM stack of experiment
  377, BG2, `manual_heuristic`, and `climatology`. Prints the per-Primary table, the Primaries
  matching the stack, leave-one-out contrasts, tails, slices, load-shift rows, and the avoidable
  curtailment cost.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
    ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00381-*/e00381_score.py screen
```
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = DATA / "00381-simple-weekly-xgboost-demand-against-gam-stack" / "work"
E379_ARMS: Final[Path] = DATA / "00379-bg2-components-all-16-primaries-to-14-days" / "work" / "arms"
SCREEN_LEVEL: Final[float] = 100 * (1 - 0.01 / 7)
SCREEN_RESAMPLES: Final[int] = 20_000
HEADLINE: Final[str] = base.HEADLINE_BAND
MATCH_TOLERANCE: Final[float] = 0.05
COUNTABLE_EXCLUDED: Final[tuple[str, ...]] = ("D15", "D18")
GROUPS: Final[dict[str, pl.Expr]] = {
    "signed-MW": pl.col("group") == "signed-MW",
    "MVA-only with D8": pl.col("group") == "MVA-only",
    "MVA-only without D8": (pl.col("group") == "MVA-only") & (pl.col("label") != "D8"),
}
LABEL_GROUPS: Final[dict[str, str]] = {
    **dict.fromkeys(("D5", "D9", "D10", "D11", "D14", "D16"), "signed-MW"),
    **dict.fromkeys(("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"), "MVA-only"),
}
QUANTILE_COLUMNS: Final[list[str]] = [
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
]  # fmt: skip


def arm_frame(*, regime: str, arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's forecast in MW: every refit file of every Primary, concatenated."""
    folder = WORK / "arm_preds" / regime / arm
    files = sorted(folder.glob("*.parquet"))
    if not files:
        msg = f"no files in {folder}"
        raise FileNotFoundError(msg)
    stored = pl.concat([pl.read_parquet(f) for f in files]).filter(
        pl.col("power_fcst_init_time") >= base.SCORED_FIRST
    )
    return adapter.forecast_frame(stored=stored, series=series, unit="fraction")


def stored_frame(*, name: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an experiment 379 arm (`only_W` or `only_M`) in MW."""
    stored = pl.read_parquet(E379_ARMS / f"{name}.parquet").filter(
        pl.col("time_series_id").is_in(series["time_series_id"].to_list())
    )
    return adapter.forecast_frame(stored=stored, series=series, unit="fraction")


def build_context(*, series: pl.DataFrame) -> dict:
    """Return the actuals, slice context, events, first readings, and twCRPS thresholds."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=base.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in series.iter_rows()
    }
    return {
        "actuals": actuals,
        "context": context,
        "events": adapter.events_from_masks(cache=cache, series=series),
        "first_readings": first_readings,
        "thresholds": metrics.twcrps_thresholds(
            history=actuals.rename({"actual": "power_mw"}),
            series=series,
            before=base.SCORED_FIRST,
        ),
    }


def score_arms(*, arms: dict[str, pl.DataFrame], series: pl.DataFrame, ctx: dict) -> tuple:
    """Return the shared slice table, each arm's scores, and each arm's row losses."""
    table = metrics.shared_slice_table(
        forecasts=arms,
        actuals=ctx["actuals"],
        context=ctx["context"],
        events=ctx["events"],
        first_readings=ctx["first_readings"],
    )
    print(f"shared half-hours: {table.height}")
    scored, rows = {}, {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept,
            actuals=ctx["actuals"],
            series=series,
            slice_table=table,
            thresholds=ctx["thresholds"],
        )
        rows[name] = metrics.scored_rows(
            forecast=kept,
            actuals=ctx["actuals"],
            series=series,
            thresholds=ctx["thresholds"],
            slice_table=table,
        )
    return table, scored, rows


def print_gate(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print the reference gate: the refitted control against the stored monthly member."""
    stored = arms["stored_monthly"].select(
        "time_series_id", "power_fcst_init_time", "valid_time", "q50"
    )
    joined = (
        arms["control"]
        .select("time_series_id", "power_fcst_init_time", "valid_time", "q50")
        .join(stored, on=["time_series_id", "power_fcst_init_time", "valid_time"], suffix="_stored")
    )
    gap = (joined["q50"] - joined["q50_stored"]).abs()
    print(
        f"GATE: {joined.height} shared rows, mean |median gap| {gap.mean():.4f} MW, "
        f"max {gap.max():.4f} MW"
    )


def screen(*, series: pl.DataFrame, ctx: dict, candidate_arms: list[str]) -> None:
    """Print the screening contrasts of each candidate arm minus the control."""
    arms = {"control": arm_frame(regime="monthly", arm="control", series=series)}
    for a in candidate_arms:
        arms[a] = arm_frame(regime="monthly", arm=a, series=series)
    arms["stored_monthly"] = stored_frame(name="only_M", series=series)
    print_gate(arms=arms)
    arms.pop("stored_monthly")
    table, scored, rows = score_arms(arms=arms, series=series, ctx=ctx)
    for a in candidate_arms:
        print(f"\n== {a} minus control, CRPS, points, level {SCREEN_LEVEL:.2f}% ==")
        for sl in (HEADLINE, "economy_7_step", "sunny_unit_shape", "sunny_april_june",
                   "cold_nights"):  # fmt: skip
            where_slice = pl.col(HEADLINE) if sl == HEADLINE else pl.col(sl) & pl.col(HEADLINE)
            for gname, where in GROUPS.items():
                r = metrics.month_bootstrap_contrast(
                    treatment=rows[a],
                    reference=rows["control"],
                    metric="crps",
                    where=where & where_slice,
                    level_percent=SCREEN_LEVEL,
                    n_resamples=SCREEN_RESAMPLES,
                    slice_table=table,
                )
                print(
                    f"{sl:18s} {gname:20s} {r['difference']:+.3f} "
                    f"({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
                )
        print("per Primary, 95%, days 1 to 10")
        print(
            metrics.reported_contrast_by_primary(
                treatment=rows[a],
                reference=rows["control"],
                metric="crps",
                where=pl.col(HEADLINE),
                slice_table=table,
            )
        )
    print("\n== absolute CRPS, days 1 to 10, per Primary ==")
    print(
        pl.concat(
            [
                t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
                    "label", "group", pl.lit(n).alias("arm"), "value"
                )
                for n, t in scored.items()
            ]
        )
        .pivot(on="arm", index=["label", "group"], values="value")
        .sort("group", "label")
    )


def diagnose(*, series: pl.DataFrame, ctx: dict, arm_names: list[str]) -> None:
    """Print the median-minus-actual bias (points of capacity) on sunny daylight half-hours."""
    arms = {a: arm_frame(regime="monthly", arm=a, series=series) for a in arm_names}
    table, _, _ = score_arms(arms=arms, series=series, ctx=ctx)
    flags = table.select(
        "time_series_id", "power_fcst_init_time", "valid_time", HEADLINE, "sunny_unit_shape"
    )
    parts = []
    for name, frame in arms.items():
        joined = (
            frame.join(
                flags, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="inner"
            )
            .join(ctx["actuals"], on=["time_series_id", "valid_time"])
            .join(series.select("time_series_id", "label", "capacity_mw"), on="time_series_id")
            .filter(pl.col(HEADLINE))
            .with_columns(
                group=pl.col("label").replace_strict(LABEL_GROUPS, default="short-history"),
                err=(pl.col("q50") - pl.col("actual")) / pl.col("capacity_mw") * 100,
                sunny=pl.col("sunny_unit_shape"),
            )
        )
        parts.append(
            joined.group_by("label", "group", "sunny")
            .agg(pl.col("err").mean().alias(f"bias_{name}"))
            .sort("group", "label", "sunny")
        )
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, on=["label", "group", "sunny"])
    print("\n== bias of the median (median minus actual), points of capacity, days 1 to 10 ==")
    print(out.sort("group", "label", "sunny"))


def final(*, series: pl.DataFrame, ctx: dict, bundle: str) -> None:
    """Print the weekly bundle against the stored weekly member and the references."""
    stack = base.stack_frames(series=series)["stack"]
    refs = base.reference_frames(series=series)
    arms = {
        "bundle": arm_frame(regime="weekly", arm=bundle, series=series),
        "weekly_baseline": stored_frame(name="only_W", series=series),
        "stack": stack,
        **refs,
    }
    table, scored, rows = score_arms(arms=arms, series=series, ctx=ctx)
    out = WORK / "scores"
    out.mkdir(exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        out / "final_scores.parquet"
    )
    countable = lambda t: t.filter(~pl.col("label").is_in(COUNTABLE_EXCLUDED))  # noqa: E731
    for ref in ("stack", "weekly_baseline", "bg2", "manual_heuristic", "climatology"):
        print(f"\n== bundle minus {ref}, days 1 to 10, 99% headline, pooled by group ==")
        for gname, where in GROUPS.items():
            r = metrics.headline_month_bootstrap_contrast(
                treatment=rows["bundle"],
                reference=rows[ref],
                metric="crps",
                where=where & pl.col(HEADLINE),
                slice_table=table,
            )
            print(f"{gname:20s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})")
        print("per Primary, 95%")
        print(
            metrics.reported_contrast_by_primary(
                treatment=rows["bundle"],
                reference=rows[ref],
                metric="crps",
                where=pl.col(HEADLINE),
                slice_table=table,
            )
        )
    diffs = (
        countable(scored["bundle"].filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE))
        .select("label", pl.col("value").alias("bundle"))
        .join(
            countable(
                scored["stack"].filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE)
            ).select("label", pl.col("value").alias("stack")),
            on="label",
        )
        .with_columns(gap=pl.col("bundle") - pl.col("stack"))
        .sort("gap")
    )
    print("\n== bundle minus stack per Primary (point estimate), sorted ==")
    print(diffs)
    print(
        f"matching the stack (gap <= {MATCH_TOLERANCE:+.2f}): "
        f"{(diffs['gap'] <= MATCH_TOLERANCE).sum()} of {diffs.height}; worst: "
        f"{diffs.tail(1)['label'][0]} ({diffs.tail(1)['gap'][0]:+.3f})"
    )
    print(
        "Primaries won (bundle vs stack):",
        metrics.primaries_won(
            treatment=countable(scored["bundle"]),
            reference=countable(scored["stack"]),
            slice_name=HEADLINE,
        ),
    )
    print("\n== leave-one-out bundle minus stack (95%) ==")
    for gname, labels in (
        ("signed-MW", ["D5", "D9", "D10", "D11", "D14", "D16"]),
        ("MVA-only", ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]),
    ):
        for left_out in labels:
            keep = pl.col("label").is_in([x for x in labels if x != left_out])
            r = metrics.month_bootstrap_contrast(
                treatment=rows["bundle"],
                reference=rows["stack"],
                metric="crps",
                where=keep & pl.col(HEADLINE),
                slice_table=table,
            )
            print(
                f"{gname:10s} without {left_out:4s} {r['difference']:+.3f} "
                f"({r['lower']:+.3f} to {r['upper']:+.3f})"
            )
    tails = ["pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99",
             "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99", "twcrps"]  # fmt: skip
    print("\n== tails and twCRPS, days 1 to 10, mean over Primaries per group ==")
    for name in arms:
        t = scored[name].filter(pl.col("slice") == HEADLINE, pl.col("metric").is_in(tails))
        print(name)
        print(
            t.group_by("group", "metric")
            .agg(pl.col("value").mean())
            .pivot(on="metric", index="group", values="value")
            .sort("group")
        )
    print("\n== slice CRPS by group: bundle, weekly_baseline, stack, bg2 ==")
    for sl in ("sunny_unit_shape", "sunny_april_june", "winter_nights", "cold_nights",
               "economy_7_step", "christmas", "bank_holidays", "event_rows",
               "event_following_7_days", "cold_start_months"):  # fmt: skip
        parts = [
            rows[n]
            .filter(pl.col(sl), pl.col(HEADLINE))
            .group_by("group")
            .agg(pl.col("crps").mean().alias(n), n_half_hours=pl.len())
            for n in ("bundle", "weekly_baseline", "stack", "bg2")
        ]
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("n_half_hours"), on="group")
        print(sl)
        print(joined.sort("group"))
    print("\n== MAE of the median, days 1 to 10 (beside the quantile-score contrasts) ==")
    print(
        pl.concat(
            [
                t.filter(pl.col("metric") == "mae_median", pl.col("slice") == HEADLINE).select(
                    "label", pl.lit(n).alias("arm"), "value"
                )
                for n, t in scored.items()
            ]
        )
        .pivot(on="arm", index="label", values="value")
        .sort("label")
    )
    print("\n== avoidable curtailment cost, signed-MW six, pounds, scored year ==")
    flows = ctx["actuals"].rename({"actual": "flow_mw"})
    cost_forecasts = {
        n: metrics.restrict_to_slice_table(forecast=arms[n], slice_table=table).join(
            ctx["actuals"], on=["time_series_id", "valid_time"]
        )
        for n in ("bundle", "weekly_baseline", "stack", "bg2", "manual_heuristic")
    }
    signed = series.filter(pl.col("label").is_in(["D5", "D9", "D10", "D11", "D14", "D16"]))
    for fraction in (0.5, 0.9):
        cost = metrics.avoidable_curtailment_cost(
            forecasts=cost_forecasts,
            series=signed,
            flows=flows,
            scored_start=base.SCORED_FIRST,
            fraction=fraction,
        )
        print(f"limit fraction {fraction}")
        print(cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label"))


def main() -> None:
    """Run the requested mode."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screen", "final", "diagnose"))
    parser.add_argument(
        "--arms", nargs="+", default=["e7", "solar", "leadw", "sunny", "hl025", "hl1", "win12"]
    )
    parser.add_argument("--bundle", default="bundle_solar+sunny+hl1")
    args = parser.parse_args()
    series = adapter.series_table()
    ctx = build_context(series=series)
    if args.mode == "diagnose":
        diagnose(series=series, ctx=ctx, arm_names=["control", "sunny", "solar"])
    elif args.mode == "screen":
        screen(series=series, ctx=ctx, candidate_arms=args.arms)
    else:
        final(series=series, ctx=ctx, bundle=args.bundle)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

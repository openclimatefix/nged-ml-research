"""Score experiment 388's lagged-power arms under demand-v1; print every number the write-up quotes.

Two modes:

- `gate`: the `base` arm, refitted at every 13th weekly refit, against experiment 379's stored
  weekly member `only_W` on the origins both cover. Passes when the mean CRPS (days 1 to 10)
  differs by at most 0.01 points of effective capacity at every Primary (experiment 383's gate).
- `final`: each lag arm minus the stored weekly baseline `only_W`, per group, per lead band, per
  Primary, with leave-one-out, tails, slices, load-shift rows, and the avoidable curtailment cost,
  beside the GAM stack of experiment 377, BG2, `manual_heuristic`, and `climatology`.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00388-*/e00388_score.py gate
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
WORK: Final[Path] = DATA / "00388-xgboost-lagged-power-column-sets-weekly-demand" / "work"
E379_ARMS: Final[Path] = DATA / "00379-bg2-components-all-16-primaries-to-14-days" / "work" / "arms"
CONTRAST_COUNT: Final[int] = 4
LEVEL: Final[float] = 100 * (1 - 0.01 / CONTRAST_COUNT)
RESAMPLES: Final[int] = 20_000
HEADLINE: Final[str] = base.HEADLINE_BAND
GATE_POINTS: Final[float] = 0.01
COUNTABLE_EXCLUDED: Final[tuple[str, ...]] = ("D15", "D18")
SIGNED: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
MVA_ONLY: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
GROUPS: Final[dict[str, pl.Expr]] = {
    "signed-MW": pl.col("group") == "signed-MW",
    "MVA-only with D8": pl.col("group") == "MVA-only",
    "MVA-only without D8": (pl.col("group") == "MVA-only") & (pl.col("label") != "D8"),
}
LEAD_BANDS_HOURS: Final[dict[str, tuple[float, float]]] = {
    "0-24 h": (0.0, 24.0),
    "24-72 h": (24.0, 72.0),
    "72-168 h": (72.0, 168.0),
    "168-240 h": (168.0, 240.0),
}
"""The brief's lead bands; a row belongs to a band when low < lead_hours <= high."""
SLICES: Final[tuple[str, ...]] = (
    "sunny_unit_shape", "sunny_april_june", "winter_nights", "cold_nights", "economy_7_step",
    "christmas", "bank_holidays", "event_rows", "event_following_7_days", "cold_start_months",
)  # fmt: skip
TAILS: Final[list[str]] = [
    "pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99",
    "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99", "twcrps",
]  # fmt: skip


def arm_frame(*, arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's forecast in MW: every refit file of every Primary, concatenated."""
    folder = WORK / "arm_preds" / "weekly" / arm
    files = sorted(folder.glob("*.parquet"))
    if not files:
        msg = f"no files in {folder}"
        raise FileNotFoundError(msg)
    stored = pl.concat([pl.read_parquet(f) for f in files]).filter(
        pl.col("power_fcst_init_time") >= base.SCORED_FIRST
    )
    return adapter.forecast_frame(stored=stored, series=series, unit="fraction")


def stored_frame(*, name: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an experiment 379 arm (`only_W`) in MW."""
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


def gate(*, series: pl.DataFrame, ctx: dict) -> None:
    """Print the reproduction gate and raise if any Primary's CRPS gap exceeds the tolerance."""
    arms = {
        "base": arm_frame(arm="base", series=series),
        "stored_weekly": stored_frame(name="only_W", series=series),
    }
    _table, scored, _rows = score_arms(arms=arms, series=series, ctx=ctx)
    crps = {
        n: t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
            "label", "value"
        )
        for n, t in scored.items()
    }
    gap = (
        crps["base"]
        .join(crps["stored_weekly"], on="label", suffix="_stored")
        .with_columns(gap=pl.col("value") - pl.col("value_stored"))
        .sort("label")
    )
    print(gap)
    worst = gap["gap"].abs().max()
    print(f"worst |CRPS gap| = {worst:.4f} points (gate {GATE_POINTS})")
    if worst > GATE_POINTS:
        msg = "reproduction gate failed"
        raise SystemExit(msg)
    print("reproduction gate passed")


def band(name: str) -> pl.Expr:
    """Return the expression selecting rows of one of the brief's lead bands."""
    low, high = LEAD_BANDS_HOURS[name]
    return (pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high)


def final(*, series: pl.DataFrame, ctx: dict, lag_arms: list[str]) -> None:
    """Print each lag arm against the stored weekly baseline and the references."""
    refs = base.reference_frames(series=series)
    arms = {
        **{a: arm_frame(arm=a, series=series) for a in lag_arms},
        "baseline": stored_frame(name="only_W", series=series),
        "stack": base.stack_frames(series=series)["stack"],
        **refs,
    }
    table, scored, rows = score_arms(arms=arms, series=series, ctx=ctx)
    out = WORK / "scores"
    out.mkdir(exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        out / "final_scores.parquet"
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
    for a in lag_arms:
        print(f"\n===== {a} minus baseline (CRPS, points, level {LEVEL:.2f}%) =====")
        for gname, where in GROUPS.items():
            r = metrics.month_bootstrap_contrast(
                treatment=rows[a],
                reference=rows["baseline"],
                metric="crps",
                where=where & pl.col(HEADLINE),
                level_percent=LEVEL,
                n_resamples=RESAMPLES,
                slice_table=table,
            )
            print(
                f"days 1-10  {gname:20s} {r['difference']:+.3f} "
                f"({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
            )
            for bname in LEAD_BANDS_HOURS:
                r = metrics.month_bootstrap_contrast(
                    treatment=rows[a],
                    reference=rows["baseline"],
                    metric="crps",
                    where=where & pl.col(HEADLINE) & band(bname),
                    level_percent=LEVEL,
                    n_resamples=RESAMPLES,
                    slice_table=table,
                )
                print(
                    f"{bname:10s} {gname:20s} {r['difference']:+.3f} "
                    f"({r['lower']:+.3f} to {r['upper']:+.3f})"
                )
        print("per Primary, 95%, days 1 to 10")
        print(
            metrics.reported_contrast_by_primary(
                treatment=rows[a],
                reference=rows["baseline"],
                metric="crps",
                where=pl.col(HEADLINE),
                slice_table=table,
            )
        )
        print("leave-one-out (95%)")
        for gname, labels in (("signed-MW", SIGNED), ("MVA-only", MVA_ONLY)):
            for left_out in labels:
                keep = pl.col("label").is_in([x for x in labels if x != left_out])
                r = metrics.month_bootstrap_contrast(
                    treatment=rows[a],
                    reference=rows["baseline"],
                    metric="crps",
                    where=keep & pl.col(HEADLINE),
                    slice_table=table,
                )
                print(
                    f"{gname:10s} without {left_out:4s} {r['difference']:+.3f} "
                    f"({r['lower']:+.3f} to {r['upper']:+.3f})"
                )
        print("placebos D2 and D4 and target D8 (95%)")
        for lab in ("D2", "D4", "D8"):
            r = metrics.month_bootstrap_contrast(
                treatment=rows[a],
                reference=rows["baseline"],
                metric="crps",
                where=(pl.col("label") == lab) & pl.col(HEADLINE),
                slice_table=table,
            )
            print(f"{lab} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})")
    for ref in ("stack", "bg2", "manual_heuristic", "climatology"):
        print(f"\n== baseline minus {ref}, days 1 to 10, 99% headline, pooled by group ==")
        for gname, where in GROUPS.items():
            r = metrics.headline_month_bootstrap_contrast(
                treatment=rows["baseline"],
                reference=rows[ref],
                metric="crps",
                where=where & pl.col(HEADLINE),
                slice_table=table,
            )
            print(f"{gname:20s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})")
    print("\n== tails and twCRPS, days 1 to 10, mean over Primaries per group ==")
    for name in arms:
        t = scored[name].filter(pl.col("slice") == HEADLINE, pl.col("metric").is_in(TAILS))
        print(name)
        print(
            t.group_by("group", "metric")
            .agg(pl.col("value").mean())
            .pivot(on="metric", index="group", values="value")
            .sort("group")
        )
    print("\n== slice CRPS by group ==")
    names = [*lag_arms, "baseline", "stack", "bg2"]
    for sl in SLICES:
        parts = [
            rows[n]
            .filter(pl.col(sl), pl.col(HEADLINE))
            .group_by("group")
            .agg(pl.col("crps").mean().alias(n), n_half_hours=pl.len())
            for n in names
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
        for n in (*lag_arms, "baseline", "stack", "bg2", "manual_heuristic")
    }
    signed = series.filter(pl.col("label").is_in(SIGNED))
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
    parser.add_argument("mode", choices=("gate", "final"))
    parser.add_argument("--arms", nargs="+", default=["a", "b", "c"])
    args = parser.parse_args()
    series = adapter.series_table()
    ctx = build_context(series=series)
    if args.mode == "gate":
        gate(series=series, ctx=ctx)
    else:
        final(series=series, ctx=ctx, lag_arms=args.arms)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

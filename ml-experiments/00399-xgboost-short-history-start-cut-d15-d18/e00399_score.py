# ruff: noqa: E501, PLR0915
"""Score experiment 399's arms at D15 and D18 under demand-v1 and print every number quoted.

Arms: `N0` (the monthly control refitted in this environment, the reference for every contrast),
`N0_stored` (experiment 381's stored `monthly/control`), `N3`, `N6`, `NS`, `NC`, `NCZ`, `NCR`, each
raw and widened about the median by 1.15 (`w115`), and the references `stack`, `bg2`,
`manual_heuristic`, and `climatology`. Only the refits the protocol scores are used: D18 from
2026-01-01 and D15 from 2026-03-01. Two rows sets are scored: every row (`all`), and the rows that
remain after removing NCZ's near-zero half-hours from the actuals (`ncz_common`).

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00399-xgboost-short-history-start-cut-d15-d18:studies/quick_wins \
    uv run --no-sync python ml-experiments/00399-*/e00399_score.py
```
"""

from datetime import UTC, datetime
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00399_fit as fit
import polars as pl
from studies.eval_protocol import metrics

ARMS: Final[tuple[str, ...]] = ("N0", "N3", "N6", "NS", "NC", "NCZ", "NCR")
SPREAD: Final[float] = 1.15
HEADLINE: Final[str] = base.HEADLINE_BAND
QUANTILES: Final[list[str]] = s381.QUANTILE_COLUMNS


def keep_scored_refits(*, frame: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Keep each Primary's forecasts from its first scored refit on."""
    first = pl.DataFrame(
        {
            "label": list(fit.FIRST_REFIT),
            "first": list(fit.FIRST_REFIT.values()),
        }
    ).join(series.select("label", "time_series_id"), on="label")
    return (
        frame.join(first.select("time_series_id", "first"), on="time_series_id")
        .filter(pl.col("power_fcst_init_time") >= pl.col("first"))
        .drop("first")
    )


def arm_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return every XGBoost arm in MW, raw."""
    out = {}
    for arm in ARMS:
        files = sorted((fit.ARM_DIR / arm / "control").glob("*.parquet"))
        stored = pl.concat([pl.read_parquet(f) for f in files])
        out[arm] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    out["N0_stored"] = s381.arm_frame(regime="monthly", arm="control", series=series)
    return {name: keep_scored_refits(frame=f, series=series) for name, f in out.items()}


def widen(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Widen the quantiles about the median by `SPREAD`."""
    return frame.with_columns(
        [
            (pl.col("q50") + SPREAD * (pl.col(q) - pl.col("q50"))).alias(q)
            for q in QUANTILES
            if q != "q50"
        ]
    )


def removed_actuals(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return NCZ's removed `(time_series_id, valid_time)` half-hours."""
    power = fit.scanned_power(series=series.rename({"label": "site"}))
    return fit.removed_times(power=power, arm="NCZ").rename({"time": "valid_time"})


def main() -> None:
    """Print the tables."""
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(260)
    series = adapter.series_table(labels=tuple(fit.SITES))
    ctx = s381.build_context(series=series)
    gone = removed_actuals(series=series)
    in_year = gone.filter(pl.col("valid_time") >= datetime(2025, 7, 1, tzinfo=UTC))
    print(f"NCZ removes {gone.group_by('time_series_id').len().to_dicts()} half-hours in all")
    raw = arm_frames(series=series)
    refs = base.reference_frames(series=series)
    refs["stack"] = base.stack_frames(series=series)["stack"]
    refs = {k: keep_scored_refits(frame=v, series=series) for k, v in refs.items()}
    arms = {**raw, **{f"{k}_w115": widen(frame=v) for k, v in raw.items()}, **refs}
    for rows_set in ("all", "ncz_common"):
        actuals = ctx["actuals"]
        if rows_set == "ncz_common":
            actuals = actuals.join(gone, on=["time_series_id", "valid_time"], how="anti")
        context = {**ctx, "actuals": actuals}
        print(
            f"\n######## rows set: {rows_set} ({actuals.filter(pl.col('actual').is_finite()).height} actuals)"
        )
        table, scored, rows = s381.score_arms(arms=arms, series=series, ctx=context)
        head = pl.concat(
            [
                t.filter(pl.col("slice") == HEADLINE).select(
                    "label", "metric", pl.lit(n).alias("arm"), "value"
                )
                for n, t in scored.items()
            ]
        )
        for metric in ("crps", "mae_median", "bias_median", "coverage_q01", "coverage_q05",
                       "coverage_q95", "coverage_q99", "pinball_q01", "pinball_q99"):  # fmt: skip
            sub = head.filter(pl.col("metric") == metric)
            if sub.is_empty():
                print(f"(no metric {metric})")
                continue
            print(f"\n== {metric}, days 1 to 10 ==")
            print(sub.pivot(on="arm", index="label", values="value").sort("label"))
        print("\n== scored metrics available ==", sorted(head["metric"].unique().to_list()))
        month = pl.concat(
            [
                r.filter(pl.col(HEADLINE))
                .with_columns(month=pl.col("valid_time").dt.truncate("1mo"), arm=pl.lit(n))
                .group_by("label", "month", "arm")
                .agg(pl.col("crps").mean(), n=pl.len())
                for n, r in rows.items()
            ]
        )
        for suffix in ("", "_w115"):
            sub = month.filter(pl.col("arm").str.ends_with("_w115") == (suffix == "_w115"))
            sub = sub.with_columns(pl.col("arm").str.replace("_w115", ""))
            print(f"\n== CRPS by valid-time month, days 1 to 10, spread {suffix or 'raw'} ==")
            print(
                sub.pivot(on="arm", index=["label", "month"], values="crps").sort("label", "month")
            )
            wide = sub.pivot(on="arm", index=["label", "month"], values="crps")
            for arm in ("N3", "N6", "NS", "NC", "NCZ", "NCR", "N0_stored"):
                d = (
                    wide.with_columns(diff=pl.col(arm) - pl.col("N0"))
                    .group_by("label")
                    .agg(
                        months_lower=(pl.col("diff") < 0).sum(),
                        months=pl.len(),
                        mean_month_diff=pl.col("diff").mean(),
                    )
                )
                print(f"-- {arm} minus N0 ({suffix or 'raw'}) --")
                print(d.sort("label"))
        print("\n== by lead band (CRPS) ==")
        bands = [x for x in scored["N0"]["slice"].unique().to_list() if "lead" in x or "day" in x]
        for n in ("N0", "N0_stored", "N3", "N6", "NS", "NC", "NCZ", "NCR", "bg2", "stack",
                  "manual_heuristic", "climatology"):  # fmt: skip
            print(n)
            print(
                scored[n]
                .filter(pl.col("metric") == "crps", pl.col("slice").is_in(bands))
                .pivot(on="slice", index="label", values="value")
                .sort("label")
            )
        flags = table.select("time_series_id", "power_fcst_init_time", "valid_time", HEADLINE)
        bias_parts = []
        for n, frame in arms.items():
            joined = (
                frame.select("time_series_id", "power_fcst_init_time", "valid_time", "q50")
                .join(flags, on=["time_series_id", "power_fcst_init_time", "valid_time"])
                .join(actuals, on=["time_series_id", "valid_time"])
                .filter(pl.col(HEADLINE) & pl.col("actual").is_finite())
                .join(series.select("time_series_id", "label", "capacity_mw"), on="time_series_id")
                .group_by("label")
                .agg(
                    ((pl.col("q50") - pl.col("actual")) / pl.col("capacity_mw") * 100)
                    .mean()
                    .alias("bias")
                )
                .with_columns(arm=pl.lit(n))
            )
            bias_parts.append(joined)
        print(
            "\n== signed bias of the median (forecast minus actual), points of capacity, days 1 to 10 =="
        )
        print(pl.concat(bias_parts).pivot(on="arm", index="label", values="bias").sort("label"))
        fit.WORK_DIR.joinpath("scores").mkdir(exist_ok=True)
        pl.concat(bias_parts).write_parquet(fit.WORK_DIR / "scores" / f"bias_{rows_set}.parquet")
        head.write_parquet(fit.WORK_DIR / "scores" / f"headline_{rows_set}.parquet")
        month.write_parquet(fit.WORK_DIR / "scores" / f"monthly_{rows_set}.parquet")
        flows = actuals.rename({"actual": "flow_mw"})
        try:
            cost_fc = {
                n: metrics.restrict_to_slice_table(forecast=arms[n], slice_table=table).join(
                    actuals, on=["time_series_id", "valid_time"]
                )
                for n in (
                    "N0",
                    "N3",
                    "N6",
                    "NS",
                    "NC",
                    "NCZ",
                    "NCR",
                    "bg2",
                    "stack",
                    "manual_heuristic",
                )
            }
            for fraction in (0.5, 0.9):
                cost = metrics.avoidable_curtailment_cost(
                    forecasts=cost_fc,
                    series=series,
                    flows=flows,
                    scored_start=base.SCORED_FIRST,
                    fraction=fraction,
                )
                print(f"\n== avoidable curtailment cost, pounds, limit fraction {fraction} ==")
                print(
                    cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label")
                )
        except Exception as error:  # noqa: BLE001
            print(f"curtailment cost unavailable: {type(error).__name__}: {error}")
        if rows_set == "all":
            print("\n== median minus mean forecast, per site (MW), N0 and NC ==")
            for n in ("N0", "NC"):
                f = arms[n].with_columns(mean_q=sum(pl.col(q) for q in QUANTILES) / len(QUANTILES))
                print(
                    n,
                    f.group_by("time_series_id")
                    .agg((pl.col("q50") - pl.col("mean_q")).mean())
                    .to_dicts(),
                )
    print(
        f"\nNCZ half-hours removed in the scored year per series: "
        f"{in_year.group_by('time_series_id').len().to_dicts()}"
    )


if __name__ == "__main__":
    main()

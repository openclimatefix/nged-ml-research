# ruff: noqa: E501, PLR0915
"""Score experiment 408's A0 and A1 arms under demand-v1 and print every number the page quotes.

Arms: `A0` (the monthly control refitted here on cleaned data, D15 and D18 cut), `A1` (A0 plus the
CAMS columns `cams_u` and `cams_k`), each raw and widened about the median by 1.15 (`w115`), and
the references `bg2`, `stack`, `manual_heuristic`, and `climatology`. The scoring actuals are the
cleaned power of `studies.power.scan_power`; the half-hours the cleaning dropped are counted per
refit month. Contrasts (paired month bootstrap, 20,000 resamples): C1 = A1 - A0, pooled over the
eight MVA-only Primaries at 99.67% (Bonferroni k = 3, of which only C1 is run because the
known-answer test failed), and per Primary at 95%. A code check compares A0 at D5 and D12 with
experiment 381's stored monthly control.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00408-xgboost-signed-label-and-cams-columns-at-mva-only-primaries \
    uv run --no-sync python ml-experiments/00408-*/e00408_score.py --sites D2 D3 D4 D8 D12 D19 D21 D22 D5
```
"""

import argparse
from datetime import UTC, datetime
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00408_common as c
import e00408_fit as fit
import polars as pl
from studies.eval_protocol import metrics
from studies.power import scan_power

SPREAD: Final[float] = 1.15
HEADLINE: Final[str] = base.HEADLINE_BAND
QUANTILES: Final[list[str]] = s381.QUANTILE_COLUMNS
C1_LEVEL: Final[float] = 100 * (1 - 0.01 / 3)
RESAMPLES: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = ("A0", "A1")
SHORT_HISTORY_FIRST_REFIT: Final[dict[str, datetime]] = {
    "D15": datetime(2026, 3, 1, tzinfo=UTC),
    "D18": datetime(2026, 1, 1, tzinfo=UTC),
}
"""The first monthly refit the protocol scores at the two short-history Primaries."""


def arm_frame(*, arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's forecast in MW over the scored refits of the series in `series`."""
    folder = fit.mode_dir(mode="real") / "preds" / arm
    labels = series["label"].to_list()
    files = [f for f in sorted(folder.glob("*.parquet")) if f.name.split("_")[0] in labels]
    stored = pl.concat(
        [pl.read_parquet(f).with_columns(site=pl.lit(f.name.split("_")[0])) for f in files]
    )
    first = pl.col("site").replace_strict(
        SHORT_HISTORY_FIRST_REFIT, default=base.SCORED_FIRST, return_dtype=pl.Datetime("us", "UTC")
    )
    stored = stored.filter(pl.col("power_fcst_init_time") >= first).drop("site")
    return adapter.forecast_frame(stored=stored, series=series, unit="fraction")


def widen(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Widen the quantiles about the median by `SPREAD`."""
    return frame.with_columns(
        [
            (pl.col("q50") + SPREAD * (pl.col(q) - pl.col("q50"))).alias(q)
            for q in QUANTILES
            if q != "q50"
        ]
    )


def cleaned_actuals(*, series: pl.DataFrame, first: datetime) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the scoring actuals from the cleaned power, and the half-hours cleaning dropped.

    The second frame holds the `(time_series_id, valid_time)` half-hours that experiment 87's stored
    observed power holds and the cleaned power does not, counted by refit month in `main`.
    """
    ids = series["time_series_id"].to_list()
    cleaned = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids) & (pl.col("time") < c.POWER_END))
        .select(
            "time_series_id", pl.col("time").alias("valid_time"), pl.col("power").alias("actual")
        )
        .collect()
    )
    cuts = {site: c.cut_of(site=site) for site in c.CUT_SITES}
    for label, cut in cuts.items():
        sid = series.filter(pl.col("label") == label)["time_series_id"]
        if sid.len():
            cleaned = cleaned.filter(
                ~((pl.col("time_series_id") == sid[0]) & (pl.col("valid_time") < cut))
            )
    return cleaned.filter(pl.col("valid_time") >= first), cleaned.head(0)


def print_sunny_bias(
    *,
    arms: dict[str, pl.DataFrame],
    table: pl.DataFrame,
    series: pl.DataFrame,
    actuals: pl.DataFrame,
) -> None:
    """Print D12's bias of the median (forecast minus actual, points) on sunny half-hours by lead.

    Sunny is shown three ways: the table's realised-index slice if present, the table's
    `sunny_unit_shape` slice, and CAMS `k` at least 0.7 (computed here from experiment 408's
    CAMS arrays).
    """
    row = series.filter(pl.col("label") == "D12")
    if row.is_empty():
        return
    sid, capacity = row["time_series_id"][0], row["capacity_mw"][0]
    full = c.cams_arrays(
        row=next(
            fit.fd._series_and_arrays()[0].filter(pl.col("site") == "D12").iter_rows(named=True)
        )
    )
    grid_k = pl.DataFrame({"valid_time": fit.common.grid_times(), "cams_k": full.k})
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    bands = [col for col in table.columns if col.startswith("lead_day")]
    sunny = [col for col in table.columns if col.startswith("sunny")]
    for name in ("A0", "A1"):
        joined = (
            arms[name]
            .filter(pl.col("time_series_id") == sid)
            .select(*keys, "q50")
            .join(table.filter(pl.col("time_series_id") == sid), on=keys)
            .join(
                actuals.filter(pl.col("time_series_id") == sid), on=["time_series_id", "valid_time"]
            )
            .join(grid_k, on="valid_time", how="left")
            .filter(pl.col("actual").is_finite())
            .with_columns(
                bias=(pl.col("q50") - pl.col("actual")) / capacity * 100,
                cams_sunny=pl.col("cams_k") >= c.GATE_K_MIN,
            )
        )
        for flag in [*sunny, "cams_sunny"]:
            sel = joined.filter(pl.col(flag))
            by = {
                b: float(sel.filter(pl.col(b))["bias"].mean())  # ty: ignore[invalid-argument-type]
                for b in bands
                if sel.filter(pl.col(b)).height
            }
            print(
                f"{name} D12 sunny={flag}: n={sel.height} mean bias {float(sel['bias'].mean()):+.2f}; by lead band { ({k: round(v, 2) for k, v in by.items()}) }"  # ty: ignore[invalid-argument-type]
            )


def main() -> None:
    """Print the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", required=True)
    args = parser.parse_args()
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(260)
    series = adapter.series_table(labels=tuple(args.sites))
    ctx = s381.build_context(series=series)
    old = ctx["actuals"]
    new, _ = cleaned_actuals(series=series, first=old["valid_time"].min())
    dropped = (
        old.join(
            new.select("time_series_id", "valid_time"),
            on=["time_series_id", "valid_time"],
            how="anti",
        )
        .filter(pl.col("actual").is_finite() & (pl.col("valid_time") >= base.SCORED_FIRST))
        .with_columns(month=pl.col("valid_time").dt.truncate("1mo"))
        .join(series.select("time_series_id", "label"), on="time_series_id")
    )
    print(
        "== half-hours dropped by cleaning (in the old actuals, not in the cleaned ones), scored year =="
    )
    print(dropped.group_by("label").len().sort("label"))
    print(dropped.group_by("month").len().sort("month"))
    ctx = {
        **ctx,
        "actuals": new.join(
            old.select("time_series_id", "valid_time"),
            on=["time_series_id", "valid_time"],
            how="semi",
        ),
    }
    raw = {a: arm_frame(arm=a, series=series) for a in ARMS}
    refs = base.reference_frames(series=series)
    refs["stack"] = base.stack_frames(series=series)["stack"]
    arms = {**raw, **{f"{k}_w115": widen(frame=v) for k, v in raw.items()}, **refs}
    table, scored, rows = s381.score_arms(arms=arms, series=series, ctx=ctx)
    head = pl.concat(
        [
            t.filter(pl.col("slice") == HEADLINE).select(
                "label", "metric", pl.lit(n).alias("arm"), "value"
            )
            for n, t in scored.items()
        ]
    )
    for metric in (
        "crps",
        "mae_median",
        "bias_median",
        "coverage_q01",
        "coverage_q05",
        "coverage_q95",
        "coverage_q99",
        "pinball_q01",
        "pinball_q99",
    ):
        sub = head.filter(pl.col("metric") == metric)
        if not sub.is_empty():
            print(f"\n== {metric}, days 1 to 10 ==")
            print(sub.pivot(on="arm", index="label", values="value").sort("label"))
    print("\n== scored metrics available ==", sorted(head["metric"].unique().to_list()))
    mva = pl.col("label").is_in(list(c.MVA_ONLY))
    for spread in ("A", "A_w115"):
        t, r = (f"{spread[0]}1{spread[1:]}", f"{spread[0]}0{spread[1:]}")
        for slice_name in (HEADLINE, "sunny_unit_shape", "cold_nights", "economy_7_step"):
            where_slice = (
                pl.col(HEADLINE)
                if slice_name == HEADLINE
                else pl.col(slice_name) & pl.col(HEADLINE)
            )
            for name, where in (
                ("MVA-only eight", mva),
                ("other eight", ~mva),
                ("D12", pl.col("label") == "D12"),
            ):
                try:
                    res = metrics.month_bootstrap_contrast(
                        treatment=rows[t],
                        reference=rows[r],
                        metric="crps",
                        where=where & where_slice,
                        level_percent=C1_LEVEL,
                        n_resamples=RESAMPLES,
                        slice_table=table,
                    )
                except ValueError as error:
                    print(f"C1 {t}-{r} {slice_name} {name}: {error}")
                    continue
                print(
                    f"C1 {t}-{r} {slice_name:18s} {name:15s} {res['difference']:+.3f} ({res['lower']:+.3f} to {res['upper']:+.3f}) n={res['n_half_hours']}"
                )
        print(f"\nper Primary, 95%, days 1 to 10, {t} minus {r}")
        print(
            metrics.reported_contrast_by_primary(
                treatment=rows[t],
                reference=rows[r],
                metric="crps",
                where=pl.col(HEADLINE),
                slice_table=table,
            )
        )
    print("\n== by lead band, CRPS ==")
    bands = [x for x in scored["A0"]["slice"].unique().to_list() if "lead" in x or "day" in x]
    for n in ("A0", "A1", "bg2", "stack", "manual_heuristic", "climatology"):
        print(n)
        print(
            scored[n]
            .filter(pl.col("metric") == "crps", pl.col("slice").is_in(bands))
            .pivot(on="slice", index="label", values="value")
            .sort("label")
        )
    print("\n== slice columns of the shared table ==", table.columns)
    print_sunny_bias(arms=arms, table=table, series=series, actuals=ctx["actuals"])
    flags = table.select("time_series_id", "power_fcst_init_time", "valid_time", HEADLINE)
    print("\n== median minus mean forecast per site (MW) ==")
    for n in ("A0", "A1"):
        f = arms[n].with_columns(mean_q=sum(pl.col(q) for q in QUANTILES) / len(QUANTILES))
        print(
            n,
            f.group_by("time_series_id").agg((pl.col("q50") - pl.col("mean_q")).mean()).to_dicts(),
        )
    del flags
    flows = ctx["actuals"].rename({"actual": "flow_mw"})
    cost_fc = {
        n: metrics.restrict_to_slice_table(forecast=arms[n], slice_table=table).join(
            ctx["actuals"], on=["time_series_id", "valid_time"]
        )
        for n in ("A0", "A1", "bg2", "manual_heuristic")
    }
    for fraction in (0.5, 0.9):
        cost = metrics.avoidable_curtailment_cost(
            forecasts=cost_fc,
            series=series,
            flows=flows,
            scored_start=base.SCORED_FIRST,
            fraction=fraction,
        )
        print(
            f"\n== avoidable curtailment cost, pounds, limit fraction {fraction} (signed-MW series only) =="
        )
        print(cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label"))
    stored = s381.arm_frame(
        regime="monthly", arm="control", series=adapter.series_table(labels=("D5", "D12"))
    )
    s_table, s_scored, _ = s381.score_arms(
        arms={
            "A0": raw["A0"].filter(
                pl.col("time_series_id").is_in(stored["time_series_id"].unique())
            ),
            "stored_381": stored,
        },
        series=adapter.series_table(labels=("D5", "D12")),
        ctx=ctx,
    )
    print(
        "\n== code check: A0 against experiment 381's stored monthly control, CRPS days 1 to 10 =="
    )
    print(
        pl.concat(
            [
                t.filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE).select(
                    "label", pl.lit(n).alias("arm"), "value"
                )
                for n, t in s_scored.items()
            ]
        ).pivot(on="arm", index="label", values="value")
    )
    del s_table
    head.write_parquet(fit.mode_dir(mode="real").parent.parent / "headline.parquet")


if __name__ == "__main__":
    main()

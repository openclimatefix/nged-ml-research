"""Score the stale-input forecasts of experiment 410 against the fresh case (no fit).

For each stale arm the script averages the three seeds in quantile space, applies the 1.15
widening, and scores the arm and the fresh arm together on their shared half-hours (so the older
weather runs, which do not reach the last day of the horizon, are compared on equal rows). It
prints, per group, the CRPS at days 1 to 10, the difference from the fresh arm with a month-block
95% bootstrap interval, the q95 and q99 coverage, the q05/q95 and q01/q99 interval widths, and the
number of flagged `substation_zero` half-hours dropped from the actuals per fold.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand \
OMP_NUM_THREADS=2 systemd-run --user --scope -p MemoryMax=4G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00410-*/e00410_stale_score.py --network pooled
```
"""

import argparse
from datetime import timedelta
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00410_stale_predict import ARMS, FLAGGED_KEYS, OUT_DIR, SEEDS  # ty: ignore[unresolved-import]
from studies.eval_protocol import metrics

HEADLINE: Final[str] = base.HEADLINE_BAND
WIDEN: Final[float] = common.SPREAD_FACTOR
OUTER: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
SIGNED: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "signed-MW six": SIGNED,
    "MVA-only eight": MVA_ONLY,
    "MVA-only seven (no D8)": tuple(s for s in MVA_ONLY if s != "D8"),
    "D15 and D18": ("D15", "D18"),
}
THRESHOLDS: Final[dict[str, float]] = {
    "P6": 0.05,
    "P12": 0.05,
    "W24": 0.10,
    "P24": 0.10,
    "W24_P12": 0.10,
    "W24_P24": 0.10,
}
"""Pre-registered CRPS-loss limits in points; the other arms are report-only."""
COVERAGE_LOSS_LIMIT: Final[float] = 0.02
RESAMPLES: Final[int] = 20_000


def widen(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Widen q01, q05, q95, and q99 about the median by the spread factor 1.15."""
    return frame.with_columns(
        *[(pl.col("q50") + WIDEN * (pl.col(c) - pl.col("q50"))).alias(c) for c in OUTER]
    )


def arm_forecast(*, network: str, arm: str, months: list[str]) -> pl.DataFrame:
    """Return an arm's seed-averaged forecast (fraction of capacity) over the months, sorted."""
    keys = list(common.KEYS)
    pieces = []
    for month in months:
        seeds = (0,) if network == "xgb_s1m" else SEEDS
        files = [OUT_DIR / "preds" / network / arm / f"{month}_s{s}.parquet" for s in seeds]
        stack = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(stack.group_by(keys).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS]))
    frame = pl.concat(pieces).sort(keys)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    return frame.with_columns(
        *[pl.Series(c, values[:, i]).cast(pl.Float32) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )


def group_mean(*, scored: pl.DataFrame, metric: str, labels: tuple[str, ...]) -> float:
    """Return the equal-weight mean over Primaries of a headline-band metric."""
    table = scored.filter(
        pl.col("metric") == metric, pl.col("slice") == HEADLINE, pl.col("label").is_in(list(labels))
    )
    return float(table["value"].to_numpy().mean())


def widths(
    *, frame: pl.DataFrame, series: pl.DataFrame, table: pl.DataFrame, labels: tuple[str, ...]
) -> dict[str, float]:
    """Return the mean q05-to-q95 and q01-to-q99 widths in points of capacity, days 1 to 10."""
    label_of = {r[0]: r[1] for r in series.iter_rows()}
    scale_of = {r[0]: 100.0 / r[2] for r in series.iter_rows()}
    headline = table.select(*metrics.KEY_COLUMNS, HEADLINE)
    kept = (
        frame.join(headline, on=list(metrics.KEY_COLUMNS), how="inner")
        .filter(pl.col(HEADLINE))
        .with_columns(
            label=pl.col("time_series_id").replace_strict(label_of),
            scale=pl.col("time_series_id").replace_strict(scale_of),
        )
        .filter(pl.col("label").is_in(list(labels)))
        .with_columns(
            width_90=(pl.col("q95") - pl.col("q05")) * pl.col("scale"),
            width_98=(pl.col("q99") - pl.col("q01")) * pl.col("scale"),
        )
        .group_by("label")
        .agg(pl.col("width_90").mean(), pl.col("width_98").mean())
    )
    return {
        "width_90": float(kept["width_90"].to_numpy().mean()),
        "width_98": float(kept["width_98"].to_numpy().mean()),
    }


def decision_table(*, results: pl.DataFrame) -> None:
    """Print each limit arm's pass or fail against the pre-registered rule, per group."""
    rows = []
    for record in results.filter(pl.col("arm").is_in(list(THRESHOLDS))).iter_rows(named=True):
        if record["group"] not in ("signed-MW six", "MVA-only eight"):
            continue
        limit = THRESHOLDS[record["arm"]]
        loss95 = record["coverage_q95_fresh"] - record["coverage_q95_arm"]
        loss99 = record["coverage_q99_fresh"] - record["coverage_q99_arm"]
        rows.append(
            {
                "arm": record["arm"],
                "group": record["group"],
                "limit": limit,
                "loss": record["diff"],
                "loss_upper95": record["upper"],
                "crps_ok": record["diff"] < limit,
                "crps_ok_upper": record["upper"] < limit,
                "cov95_loss": loss95,
                "cov99_loss": loss99,
                "coverage_ok": max(loss95, loss99) < COVERAGE_LOSS_LIMIT,
            }
        )
    table = pl.DataFrame(rows)
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200):
        print("\n== pre-registered rule: point estimate and 95% upper bound against the limit ==")
        print(table)
    ok = table.group_by("arm").agg((pl.col("crps_ok") & pl.col("coverage_ok")).all().alias("holds"))
    print(f"Rule holds for every arm and both groups (point estimates): {bool(ok['holds'].all())}")


def main() -> None:
    """Score every stale arm against the fresh arm and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="pooled", choices=("pooled", "per_primary", "xgb_s1m"))
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--arms", nargs="+", default=[a.name for a in ARMS if a.name != "FRESH"])
    args = parser.parse_args()
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(FLAGGED_KEYS).select("time_series_id", "valid_time")
    for month in args.months:
        origins = prep.test_origins(month=month)
        window = flagged.filter(
            pl.col("valid_time").is_between(origins[0], origins[-1] + timedelta(days=15))
        )
        count = ctx["actuals"].join(window, on=["time_series_id", "valid_time"], how="semi").height
        print(f"{month}: flagged substation_zero half-hours dropped from the actuals: {count}")
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    fresh = widen(
        frame=adapter.forecast_frame(
            stored=arm_forecast(network=args.network, arm="FRESH", months=args.months),
            series=series,
            unit="fraction",
        )
    )
    results = []
    for arm in args.arms:
        stale = widen(
            frame=adapter.forecast_frame(
                stored=arm_forecast(network=args.network, arm=arm, months=args.months),
                series=series,
                unit="fraction",
            )
        )
        table, scored, rows = s381.score_arms(
            arms={"FRESH": fresh, arm: stale}, series=series, ctx=ctx
        )
        for name, labels in GROUPS.items():
            where = pl.col("label").is_in(list(labels)) & pl.col(HEADLINE)
            r = metrics.month_bootstrap_contrast(
                treatment=rows[arm],
                reference=rows["FRESH"],
                metric="crps",
                where=where,
                level_percent=95.0,
                n_resamples=RESAMPLES,
                slice_table=table,
            )
            record = {
                "arm": arm,
                "group": name,
                "n_half_hours": r["n_half_hours"],
                "crps_fresh": group_mean(scored=scored["FRESH"], metric="crps", labels=labels),
                "crps_arm": group_mean(scored=scored[arm], metric="crps", labels=labels),
                "diff": r["difference"],
                "lower": r["lower"],
                "upper": r["upper"],
            }
            for tag, frame in (("fresh", fresh), ("arm", stale)):
                for key, value in widths(
                    frame=frame, series=series, table=table, labels=labels
                ).items():
                    record[f"{key}_{tag}"] = value
            for metric in ("coverage_q95", "coverage_q99", "coverage_q01", "coverage_q05"):
                record[f"{metric}_fresh"] = group_mean(
                    scored=scored["FRESH"], metric=metric, labels=labels
                )
                record[f"{metric}_arm"] = group_mean(
                    scored=scored[arm], metric=metric, labels=labels
                )
            for metric in ("pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99"):
                record[f"{metric}_fresh"] = group_mean(
                    scored=scored["FRESH"], metric=metric, labels=labels
                )
                record[f"{metric}_arm"] = group_mean(
                    scored=scored[arm], metric=metric, labels=labels
                )
            results.append(record)
        print(f"{arm}: scored", flush=True)
    out = pl.DataFrame(results)
    target = OUT_DIR / "scores" / f"{args.network}_contrasts.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(target)
    with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=200):
        for name in ("signed-MW six", "MVA-only eight"):
            sub = out.filter(pl.col("group") == name)
            print(f"\n== {name}: CRPS days 1 to 10 (points), interim ==")
            print(
                sub.select(
                    "arm", "n_half_hours", "crps_fresh", "crps_arm", "diff", "lower", "upper"
                )
            )
            print(f"== {name}: coverage q95 and q99, fresh then stale ==")
            print(
                sub.select(
                    "arm",
                    "coverage_q95_fresh",
                    "coverage_q95_arm",
                    "coverage_q99_fresh",
                    "coverage_q99_arm",
                )
            )
    decision_table(results=out)
    print(
        f"\nPre-registered CRPS-loss limits (points): {THRESHOLDS}; "
        f"coverage loss < {COVERAGE_LOSS_LIMIT}"
    )


if __name__ == "__main__":
    main()

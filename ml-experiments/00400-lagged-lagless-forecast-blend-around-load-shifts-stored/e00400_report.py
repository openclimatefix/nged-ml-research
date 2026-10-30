"""Experiment 400, report step: print every number the write-up quotes.

Builds one wide per-row table (`work/wide.parquet`: keys, corrected tags, and for each arm the CRPS,
coverages, 90% width, median, and q95) and prints the seven checks of the plan. Headline rows are
days 1 to 10 (lead under 240 hours), valid times from 2025-10-01 (the warm-up months July to
September 2025 are printed apart), with the flagged exact-zero half-hours dropped. Intervals resample
whole events (event rows) or calendar months (pooled rows).

```bash
PYTHONPATH=<as in e00400_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00400-*/e00400_report.py [wide|report|curtailment]
```
"""
# ruff: noqa: E501, PERF401

import sys
from datetime import datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00400_blend as blend
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import KEY_COLUMNS, SIGNED_MW_LABELS

WORK: Final[Path] = blend.WORK
KEYS: Final[list[str]] = blend.KEYS
ARMS: Final[tuple[str, ...]] = (
    "lags", "lagless", "mix_fixed", "avg_fixed", "mix_b1", "avg_b1", "mix_b2", "avg_b2", "mix_b3", "avg_b3",
)  # fmt: skip
BLENDS: Final[tuple[str, ...]] = ARMS[2:]
NAMES: Final[dict[str, str]] = {
    "lags": "forecast with lags (experiment 381 weekly bundle)",
    "lagless": "forecast without lags (experiment 383 raw lagless arm)",
}
SIX: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
EIGHT: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
GROUPS: Final[dict[str, list[str]]] = {"six": SIX, "eight": EIGHT}
HEADLINE_LEAD: Final[float] = 240.0
N_RESAMPLES: Final[int] = 4000
SCORED_FROM: Final[datetime] = blend.WARMUP_END


def build_wide() -> None:
    """Write `wide.parquet`: one row per scored half-hour, the arms' columns side by side."""
    tags = pl.read_parquet(WORK / "tags_corrected.parquet")
    slices = pl.read_parquet(
        blend.PART_F / "slice_table.parquet",
        columns=[*KEYS, "cold_nights", "event_rows", "event_following_7_days"],
    )
    weights = pl.read_parquet(
        WORK / "weights.parquet", columns=[*KEYS, "w_b1", "w_b2", "w_b3", "s_own", "s_nb"]
    )
    wide: pl.DataFrame | None = None
    for arm in ARMS:
        rows = pl.read_parquet(WORK / f"rows_{arm}.parquet").filter(
            pl.col("lead_hours") < HEADLINE_LEAD
        )
        part = rows.select(
            *KEYS,
            pl.col("crps").alias(f"crps_{arm}"),
            pl.col("coverage_q95").alias(f"c95_{arm}"),
            pl.col("coverage_q99").alias(f"c99_{arm}"),
            pl.col("coverage_q05").alias(f"c05_{arm}"),
            pl.col("coverage_q01").alias(f"c01_{arm}"),
            pl.col("width90").alias(f"w90_{arm}"),
            pl.col("q50_mw").alias(f"q50_{arm}"),
            pl.col("q95_mw").alias(f"q95_{arm}"),
        )
        if wide is None:
            wide = rows.select(
                *KEYS, "lead_hours", "label", "group", "capacity_mw", "flagged", "actual_mw"
            ).join(part, on=KEYS)
        else:
            wide = wide.join(part, on=KEYS, how="left", validate="1:1")
        print("wide", arm, wide.width, flush=True)
    if wide is None:
        msg = "no arms"
        raise ValueError(msg)
    wide = (
        wide.join(tags, on=KEYS, how="left", validate="1:1")
        .join(slices, on=KEYS, how="left", validate="1:1")
        .join(weights, on=KEYS, how="left", validate="1:1")
    )
    wide.write_parquet(WORK / "wide.parquet")


def event_interval(*, sums: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    """Return the pooled mean and a 95% interval from resampling whole units (events or months)."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(sums), size=(N_RESAMPLES, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(sums.sum() / counts.sum()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def fmt(t: tuple[float, float, float]) -> str:
    """Format a mean and interval."""
    return f"{t[0]:+.3f} ({t[1]:+.3f} to {t[2]:+.3f})"


def contrast(*, rows: pl.DataFrame, arm: str, ref: str, unit: str) -> str:
    """Return arm minus ref CRPS with an interval from resampling `unit` (a column)."""
    d = rows.with_columns(diff=pl.col(f"crps_{arm}") - pl.col(f"crps_{ref}"))
    by = d.group_by(unit).agg(s=pl.col("diff").sum(), n=pl.len())
    if by.height < 2:
        return f"n/a ({by.height} units)"
    return (
        fmt(event_interval(sums=by["s"].to_numpy(), counts=by["n"].to_numpy()))
        + f" [{by.height} units, {d.height} rows]"
    )


def absolute(*, rows: pl.DataFrame, title: str) -> None:
    """Print each arm's absolute CRPS, coverages, and width on the rows, per group."""
    print(f"\n-- {title}: rows {rows.height}")
    out = []
    for gname, labs in {**GROUPS, "all 16": None}.items():
        sub = rows if labs is None else rows.filter(pl.col("label").is_in(labs))
        for arm in ARMS:
            out.append(
                {
                    "group": gname,
                    "arm": arm,
                    "n": sub.height,
                    "crps": sub[f"crps_{arm}"].mean(),
                    "q95": sub[f"c95_{arm}"].mean(),
                    "q99": sub[f"c99_{arm}"].mean(),
                    "w90": sub[f"w90_{arm}"].mean(),
                }
            )
    print(pl.DataFrame(out).with_columns(pl.selectors.float().round(3)))


def checks(*, wide: pl.DataFrame, tag: str) -> None:
    """Print checks 1 and 2 (harm table and cost outside events) on the given rows."""
    print(f"\n===== CHECK 1: harm, target outside every event, by lags in the event [{tag}] =====")
    outside = wide.filter(pl.col("target_event_id") < 0)
    for gname, labs in GROUPS.items():
        sub = outside.filter(pl.col("label").is_in(labs))
        rows = []
        for arm in ARMS:
            by = (
                sub.group_by("lag_count")
                .agg(
                    n=pl.len(),
                    crps=pl.col(f"crps_{arm}").mean(),
                    q95=pl.col(f"c95_{arm}").mean(),
                    q99=pl.col(f"c99_{arm}").mean(),
                )
                .sort("lag_count")
            )
            for r in by.iter_rows(named=True):
                rows.append({"group": gname, "arm": arm, **r})
        print(
            pl.DataFrame(rows)
            .with_columns(pl.selectors.float().round(3))
            .pivot(on="lag_count", index=["group", "arm"], values=["n", "crps", "q95", "q99"])
        )
        hit = sub.filter(pl.col("lag_count") > 0)
        print(
            f"{gname}: contrasts on rows with 1 to 3 lags inside an event (whole-event resampling)"
        )
        for arm in ARMS:
            if arm != "lags":
                print(
                    f"   {arm} minus lags: {contrast(rows=hit, arm=arm, ref='lags', unit='lag_event_id')}"
                )
        for arm in ("mix_b2", "mix_b3", "mix_b1"):
            print(
                f"   {arm} minus mix_fixed: {contrast(rows=hit, arm=arm, ref='mix_fixed', unit='lag_event_id')}"
            )
            print(
                f"   {arm} minus lagless: {contrast(rows=hit, arm=arm, ref='lagless', unit='lag_event_id')}"
            )
    print(
        f"\n===== CHECK 2: cost outside events (no lag in an event, target outside, not in the 7 days after) [{tag}] ====="
    )
    clean = wide.filter(
        pl.col("lag_count") == 0,
        pl.col("target_event_id") < 0,
        pl.col("hours_after_end").is_null() | (pl.col("hours_after_end") > 168),
    )
    clean = clean.with_columns(month=pl.col("valid_time").dt.truncate("1mo"))
    absolute(rows=clean, title=f"absolute scores outside events [{tag}]")
    for arm in ARMS:
        if arm != "lags":
            print(
                f"   {arm} minus lags, all 16, month resampling: {contrast(rows=clean, arm=arm, ref='lags', unit='month')}"
            )
    for arm in ("mix_b2", "mix_b3", "mix_b1", "mix_fixed"):
        per = (
            clean.group_by("label")
            .agg(d=(pl.col(f"crps_{arm}") - pl.col("crps_lags")).mean())
            .sort("label")
        )
        print(
            f"   {arm} minus lags per Primary:", [(r[0], round(r[1], 3)) for r in per.iter_rows()]
        )
    for arm in ("mix_b2", "mix_b3"):
        print(
            f"   {arm} minus mix_fixed, all 16: {contrast(rows=clean, arm=arm, ref='mix_fixed', unit='month')}"
        )
        print(
            f"   {arm} minus lagless, all 16: {contrast(rows=clean, arm=arm, ref='lagless', unit='month')}"
        )


def report() -> None:
    """Print the seven checks."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(250)
    wide_all = pl.read_parquet(WORK / "wide.parquet")
    events = pl.read_parquet(WORK / "events_corrected.parquet").with_columns(
        pl.col("event_id").cast(pl.Int64)
    )
    warm = wide_all.filter(pl.col("valid_time") < SCORED_FROM)
    wide_flag = wide_all.filter(pl.col("valid_time") >= SCORED_FROM)
    print(
        "flagged half-hours dropped per Primary:",
        wide_flag.filter(pl.col("flagged")).group_by("label").len().sort("label").rows(),
    )
    wide = wide_flag.filter(~pl.col("flagged"))
    absolute(
        rows=warm.filter(~pl.col("flagged")), title="WARM-UP months (w = 0.5 in all blend arms)"
    )
    absolute(rows=wide, title="ALL scored rows, days 1 to 10")
    checks(wide=wide, tag="all rows")
    print("\n===== CHECK 7: cold-night rows only =====")
    checks(wide=wide.filter(pl.col("cold_nights")), tag="cold nights")

    print(
        "\n===== CHECK 3: depth missed by the median per event (|median(actual - q50)| / |depth|), corrected signs ====="
    )
    inside = wide.filter(pl.col("target_event_id") >= 0).join(
        events.select("event_id", "depth_mw"), left_on="target_event_id", right_on="event_id"
    )
    inside = inside.with_columns(
        grp=pl.when(pl.col("lag_count") == 0)
        .then(pl.lit("0 lags"))
        .when(pl.col("lag_count") == 3)
        .then(pl.lit("3 lags"))
        .otherwise(pl.lit("1 or 2 lags"))
    )
    for arm in ARMS:
        per = (
            inside.with_columns(res=pl.col("actual_mw") - pl.col(f"q50_{arm}"))
            .group_by("grp", "target_event_id")
            .agg(m=pl.col("res").median(), d=pl.col("depth_mw").first(), n=pl.len())
            .filter(pl.col("n") >= 10)
            .with_columns(missed=pl.col("m").abs() / pl.col("d").abs())
        )
        print(
            arm,
            per.group_by("grp")
            .agg(events=pl.len(), missed=pl.col("missed").median())
            .sort("grp")
            .with_columns(pl.selectors.float().round(3))
            .rows(),
        )

    print("\n===== CHECK 4: tails during events (target inside a corrected event) =====")
    inside = inside.with_columns(neg=pl.col("depth_mw") < 0)
    for arm in ARMS:
        t = (
            inside.group_by("grp")
            .agg(
                n=pl.len(),
                in_1_99=(pl.col(f"c99_{arm}") - pl.col(f"c01_{arm}")).mean(),
                in_5_95=(pl.col(f"c95_{arm}") - pl.col(f"c05_{arm}")).mean(),
                beyond_event_side_1=pl.when(pl.col("neg"))
                .then(pl.col(f"c01_{arm}"))
                .otherwise(1 - pl.col(f"c99_{arm}"))
                .mean(),
                width90=pl.col(f"w90_{arm}").mean(),
            )
            .sort("grp")
        )
        print(arm, t.with_columns(pl.selectors.float().round(3)).rows())

    print(
        "\n===== CHECK 5: smoothness at event onset and end (largest half-hour change, points of capacity, window +-6 h) ====="
    )
    grid = pl.datetime_range(
        blend.common.GRID_START,
        blend.common.GRID_END,
        interval="30m",
        time_unit="us",
        time_zone="UTC",
        eager=True,
    )
    summary = []
    for ev in events.iter_rows(named=True):
        for kind, idx in (("onset", ev["onset_idx"]), ("end", ev["end_idx"])):
            t = grid[idx]
            win = wide.filter(
                pl.col("label") == ev["label"],
                pl.col("valid_time") >= t - pl.duration(hours=6),
                pl.col("valid_time") <= t + pl.duration(hours=6),
            ).sort("power_fcst_init_time", "valid_time")
            if win.is_empty():
                continue
            for arm in ARMS:
                for col, name in ((f"q50_{arm}", "median"), (f"q95_{arm}", "q95")):
                    step = win.group_by("power_fcst_init_time", maintain_order=True).agg(
                        s=(pl.col(col).diff().abs().max() / pl.col("capacity_mw").first() * 100)
                    )
                    summary.append(
                        pl.DataFrame(
                            {"arm": arm, "stat": name, "kind": kind, "step": step["s"].drop_nulls()}
                        )
                    )
    if summary:
        sm = pl.concat(summary)
        print(
            sm.group_by("arm", "stat", "kind")
            .agg(
                n=pl.len(),
                median=pl.col("step").median(),
                p95=pl.col("step").quantile(0.95),
                max=pl.col("step").max(),
            )
            .sort("stat", "kind", "arm")
            .with_columns(pl.selectors.float().round(3))
        )
    print(
        "weight on the forecast with lags (arm B2) by hours since onset (target inside an event), and outside:"
    )
    w = wide.with_columns(
        b=pl.when(pl.col("target_event_id") < 0)
        .then(pl.lit("outside"))
        .when(pl.col("hours_since_onset") < 6)
        .then(pl.lit("0-6 h"))
        .when(pl.col("hours_since_onset") < 24)
        .then(pl.lit("6-24 h"))
        .otherwise(pl.lit("over 24 h"))
    )
    print(
        w.group_by("b")
        .agg(
            n=pl.len(),
            w_b1=pl.col("w_b1").mean(),
            w_b2=pl.col("w_b2").mean(),
            w_b3=pl.col("w_b3").mean(),
            p05=pl.col("w_b2").quantile(0.05),
            p95=pl.col("w_b2").quantile(0.95),
        )
        .with_columns(pl.selectors.float().round(3))
    )
    print(
        "weight (B2) by number of lags inside an event, target outside:",
        w.filter(pl.col("target_event_id") < 0)
        .group_by("lag_count")
        .agg(
            n=pl.len(),
            w_b2=pl.col("w_b2").mean(),
            s_own=pl.col("s_own").mean(),
            s_nb=pl.col("s_nb").mean(),
        )
        .sort("lag_count")
        .with_columns(pl.selectors.float().round(3))
        .rows(),
    )


def curtailment() -> None:
    """Print avoidable curtailment (experiment 383's definition) as differences against the references."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    actuals = ctx["actuals"]
    flows = actuals.rename({"actual": "flow_mw"})
    signed = series.filter(pl.col("label").is_in(list(SIGNED_MW_LABELS)))
    table = pl.read_parquet(blend.PART_F / "slice_table.parquet")
    table = table.filter(pl.col("valid_time") >= SCORED_FROM)
    arms = {
        a: pl.read_parquet(WORK / f"arm_{a}.parquet")
        for a in ("lags", "lagless", "mix_fixed", "mix_b1", "mix_b2", "mix_b3", "avg_b2", "avg_b3")
    }
    arms["stack"] = base.stack_frames(series=series)["stack"]
    sets = {
        "during events": pl.col("event_rows"),
        "7 days after events": pl.col("event_following_7_days"),
        "all rows": pl.lit(value=True),
    }
    for name, cond in sets.items():
        keys = table.filter(cond).select(*KEY_COLUMNS)
        forecasts = {
            n: f.join(keys, on=list(KEY_COLUMNS), how="semi").join(
                actuals, on=["time_series_id", "valid_time"]
            )
            for n, f in arms.items()
        }
        parts = []
        for label in SIGNED_MW_LABELS:
            one = signed.filter(pl.col("label") == label)
            try:
                parts.append(
                    metrics.avoidable_curtailment_cost(
                        forecasts=forecasts,
                        series=one,
                        flows=flows,
                        scored_start=base.SCORED_FIRST,
                        fraction=0.9,
                    )
                )
            except ValueError as error:
                print(f"{label}: {name}: no cost ({error})")
        cost = pl.concat(parts)
        wide = cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label")
        print(f"\n== avoidable curtailment cost, GBP, limit 0.9 x max, {name} ==")
        print(wide)
        totals = {c: float(wide[c].sum()) for c in wide.columns if c != "label"}
        print(
            "total:",
            {k: round(v) for k, v in totals.items()},
            "| minus lags:",
            {k: round(v - totals["lags"]) for k, v in totals.items()},
            "| minus stack:",
            {k: round(v - totals["stack"]) for k, v in totals.items()},
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"wide": build_wide, "report": report, "curtailment": curtailment}[what]()

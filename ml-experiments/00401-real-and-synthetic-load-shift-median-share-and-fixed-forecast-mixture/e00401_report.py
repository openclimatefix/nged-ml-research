"""Experiment 401, Job 2, report step: print every number the write-up quotes.

Builds `work/wide.parquet` (keys, experiment 400's corrected tags, and for each arm the CRPS,
coverages, 90% width, median, and q95), then prints: absolute scores and contrasts on the three
row sets (rows with 1 to 3 lags inside a corrected event, rows with no lag in an event and outside
the 7 days after, all rows) and on cold nights; per-Primary contrasts; smoothness at event onset
and end; the chosen per-lead weights; and avoidable curtailment cost. Headline rows are days 1 to
10 (lead under 240 hours), valid times from 2025-10-01, flagged exact-zero half-hours dropped.
Intervals resample whole events (event rows) or calendar months (pooled rows).

```bash
PYTHONPATH=<as in e00401_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00401-*/e00401_report.py [wide|report|curtailment]
```
"""
# ruff: noqa: E501, PERF401

import json
import sys
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00400_blend as blend
import e00400_report as r400  # ty: ignore[unresolved-import]
import e00401_arms as arms
import e00401_score as score
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import KEY_COLUMNS, SIGNED_MW_LABELS

WORK: Final[Path] = arms.WORK
KEYS: Final[list[str]] = arms.KEYS
B2: Final[tuple[str, ...]] = ("mix_b2", "avg_b2")
ARMS: Final[tuple[str, ...]] = (*score.NAMES, *B2)
REFERENCE: Final[tuple[str, ...]] = ("stack",)
SIX = r400.SIX
EIGHT = r400.EIGHT
GROUPS = r400.GROUPS
HEADLINE_LEAD = r400.HEADLINE_LEAD
SCORED_FROM = r400.SCORED_FROM
fmt = r400.fmt


def arm_path(arm: str) -> Path:
    """Return the scored-rows file of an arm (experiment 400's for B2 and the stack)."""
    if arm in B2:
        return blend.WORK / f"rows_{arm}.parquet"
    if arm == "stack":
        return blend.PART_F / "rows_stack.parquet"
    return WORK / f"rows_{arm}.parquet"


def build_wide() -> None:
    """Write `wide.parquet`: one row per scored half-hour, the arms' columns side by side."""
    tags = pl.read_parquet(blend.WORK / "tags_corrected.parquet")
    slices = pl.read_parquet(
        blend.PART_F / "slice_table.parquet",
        columns=[*KEYS, "cold_nights", "event_rows", "event_following_7_days"],
    )
    wide: pl.DataFrame | None = None
    for arm in (*ARMS, "stack"):
        rows = pl.read_parquet(arm_path(arm)).filter(pl.col("lead_hours") < HEADLINE_LEAD)
        part = rows.select(
            *KEYS,
            pl.col("crps").cast(pl.Float32).alias(f"crps_{arm}"),
            pl.col("coverage_q95").cast(pl.Float32).alias(f"c95_{arm}"),
            pl.col("coverage_q99").cast(pl.Float32).alias(f"c99_{arm}"),
            pl.col("coverage_q05").cast(pl.Float32).alias(f"c05_{arm}"),
            pl.col("coverage_q01").cast(pl.Float32).alias(f"c01_{arm}"),
            pl.col("width90").cast(pl.Float32).alias(f"w90_{arm}"),
            pl.col("q50_mw").cast(pl.Float32).alias(f"q50_{arm}"),
            pl.col("q95_mw").cast(pl.Float32).alias(f"q95_{arm}"),
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
    wide.join(tags, on=KEYS, how="left", validate="1:1").join(
        slices, on=KEYS, how="left", validate="1:1"
    ).write_parquet(WORK / "wide.parquet")


def pair_for(*, arm: str) -> str:
    """Return the same-widening forecast with lags to compare an arm with."""
    return "lags_x115" if arm.endswith("_x115") else "lags"


def row_sets(*, wide: pl.DataFrame) -> dict[str, tuple[pl.DataFrame, str]]:
    """Return the row sets and the resampling unit of each."""
    outside = wide.filter(pl.col("target_event_id") < 0)
    clean = wide.filter(
        pl.col("lag_count") == 0,
        pl.col("target_event_id") < 0,
        pl.col("hours_after_end").is_null() | (pl.col("hours_after_end") > 168),
    ).with_columns(month=pl.col("valid_time").dt.truncate("1mo"))
    return {
        "event-lag rows (1 to 3 lags in an event, target outside)": (
            outside.filter(pl.col("lag_count") > 0),
            "lag_event_id",
        ),
        "outside events (no lag in an event, not in the 7 days after)": (clean, "month"),
        "all rows": (wide.with_columns(month=pl.col("valid_time").dt.truncate("1mo")), "month"),
    }


def table(*, wide: pl.DataFrame, tag: str) -> None:
    """Print absolute scores and contrasts on each row set, per group."""
    for title, (rows, unit) in row_sets(wide=wide).items():
        for gname, labs in GROUPS.items():
            sub = rows.filter(pl.col("label").is_in(labs))
            print(
                f"\n== {title} | {gname} | {tag} | rows {sub.height}, units {sub[unit].n_unique()}"
            )
            out = []
            for arm in (*REFERENCE, *ARMS):
                out.append(
                    {
                        "arm": arm,
                        "crps": sub[f"crps_{arm}"].mean(),
                        "q95": sub[f"c95_{arm}"].mean(),
                        "q99": sub[f"c99_{arm}"].mean(),
                        "q05": sub[f"c05_{arm}"].mean(),
                        "q01": sub[f"c01_{arm}"].mean(),
                        "w90": sub[f"w90_{arm}"].mean(),
                    }
                )
            print(pl.DataFrame(out).with_columns(pl.selectors.float().round(3)))
            if sub.height == 0:
                continue
            for arm in ARMS:
                ref = pair_for(arm=arm)
                if arm == ref:
                    continue
                print(
                    f"   {arm} minus {ref}: {r400.contrast(rows=sub, arm=arm, ref=ref, unit=unit)}"
                )
            for arm in ARMS:
                if (
                    arm in B2
                    or arm.endswith("_x115")
                    or arm.startswith("avg_")
                    or arm in ("lags", "lagless")
                ):
                    continue
                print(
                    f"   {arm} minus mix_b2 (raw): {r400.contrast(rows=sub, arm=arm, ref='mix_b2', unit=unit)}"
                )
            print(
                f"   mix_w50 minus avg_w50: {r400.contrast(rows=sub, arm='mix_w50', ref='avg_w50', unit=unit)}"
            )
            print(
                f"   mix_w50 minus lagless: {r400.contrast(rows=sub, arm='mix_w50', ref='lagless', unit=unit)}"
            )


def per_primary(*, wide: pl.DataFrame) -> None:
    """Print each Primary's CRPS difference against the forecast with lags and against B2."""
    for title, (rows, _unit) in row_sets(wide=wide).items():
        print(f"\n== per Primary, {title}")
        for arm in ("mix_w30", "mix_w50", "mix_w70", "mix_lead", "avg_w50", "mix_b2"):
            per = (
                rows.group_by("label")
                .agg(d=(pl.col(f"crps_{arm}") - pl.col("crps_lags")).mean(), n=pl.len())
                .sort("label")
            )
            print(f"   {arm} minus lags:", [(r[0], round(r[1], 3), r[2]) for r in per.iter_rows()])
        per = (
            rows.group_by("label")
            .agg(d=(pl.col("crps_mix_w50") - pl.col("crps_mix_b2")).mean())
            .sort("label")
        )
        print("   mix_w50 minus mix_b2:", [(r[0], round(r[1], 3)) for r in per.iter_rows()])


def headline_99(*, wide: pl.DataFrame) -> None:
    """Print the two headline contrasts at the 99.58% level (Bonferroni, k = 12)."""
    rng = np.random.default_rng(0)
    sets = row_sets(wide=wide)
    for gname, labs in GROUPS.items():
        for key, arm, ref in (
            ("event-lag rows (1 to 3 lags in an event, target outside)", "mix_w50", "mix_b2"),
            ("outside events (no lag in an event, not in the 7 days after)", "mix_w50", "lags"),
            ("outside events (no lag in an event, not in the 7 days after)", "mix_w50", "mix_b2"),
        ):
            rows, unit = sets[key]
            sub = rows.filter(pl.col("label").is_in(labs))
            d = sub.with_columns(diff=pl.col(f"crps_{arm}") - pl.col(f"crps_{ref}"))
            by = d.group_by(unit).agg(s=pl.col("diff").sum(), n=pl.len())
            sums, counts = by["s"].to_numpy(), by["n"].to_numpy()
            draws = rng.integers(0, len(sums), size=(20000, len(sums)))
            means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            lo, hi = np.percentile(means, [2.5 / 12, 100 - 2.5 / 12])
            print(
                f"HEADLINE 99.58% | {gname} | {key[:20]} | {arm} minus {ref}: "
                f"{sums.sum() / counts.sum():+.3f} ({lo:+.3f} to {hi:+.3f}) [{by.height} units]"
            )


def median_error_check() -> None:
    """Print the error of the median forecast beside the CRPS contrast (audit trigger 3)."""
    tags = pl.read_parquet(blend.WORK / "tags_corrected.parquet")
    loaded = {}
    for arm in ("lags", "mix_w50", "mix_b2"):
        loaded[arm] = (
            pl.read_parquet(arm_path(arm), columns=[*KEYS, "label", "lead_hours", "flagged", "crps", "mae_median", "q50_mw"])
            .filter(pl.col("lead_hours") < HEADLINE_LEAD, pl.col("valid_time") >= SCORED_FROM, ~pl.col("flagged"))
            .join(tags, on=KEYS)
            .filter(pl.col("target_event_id") < 0, pl.col("lag_count") > 0)
        )  # fmt: skip
    for gname, labs in GROUPS.items():
        for arm, ref in (("mix_w50", "lags"), ("mix_b2", "lags"), ("mix_w50", "mix_b2")):
            a = (
                loaded[arm]
                .filter(pl.col("label").is_in(labs))
                .select(*KEYS, "lag_event_id", crps_a="crps", mae_a="mae_median", q50_a="q50_mw")
            )
            b = (
                loaded[ref]
                .filter(pl.col("label").is_in(labs))
                .select(*KEYS, crps_b="crps", mae_b="mae_median", q50_b="q50_mw")
            )
            j = a.join(b, on=KEYS)
            print(
                f"MEDIAN CHECK | {gname} | event-lag rows | {arm} minus {ref}: CRPS {(j['crps_a'] - j['crps_b']).mean():+.3f}, "
                f"median absolute error {(j['mae_a'] - j['mae_b']).mean():+.3f}, rows {j.height}"
            )


def leave_one_out(*, wide: pl.DataFrame) -> None:
    """Print the six's event-lag contrasts with each Primary left out (audit trigger 4)."""
    rows, unit = row_sets(wide=wide)["event-lag rows (1 to 3 lags in an event, target outside)"]
    for left in (None, *SIX):
        sub = rows.filter(pl.col("label").is_in([x for x in SIX if x != left]))
        for arm, ref in (("mix_w50", "lags"), ("mix_w50", "mix_b2")):
            print(
                f"LEAVE-ONE-OUT | six without {left} | {arm} minus {ref}: {r400.contrast(rows=sub, arm=arm, ref=ref, unit=unit)}"
            )


def smoothness(*, wide: pl.DataFrame, events: pl.DataFrame) -> None:
    """Print the largest half-hour change of the median and q95 across event onsets and ends."""
    grid = pl.datetime_range(
        blend.common.GRID_START, blend.common.GRID_END, interval="30m",
        time_unit="us", time_zone="UTC", eager=True,
    )  # fmt: skip
    shown = ("lags", "lagless", "mix_w30", "mix_w50", "mix_w70", "mix_lead", "avg_w50", "mix_b2")
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
            for arm in shown:
                for col, name in ((f"q50_{arm}", "median"), (f"q95_{arm}", "q95")):
                    step = win.group_by("power_fcst_init_time", maintain_order=True).agg(
                        s=(pl.col(col).diff().abs().max() / pl.col("capacity_mw").first() * 100)
                    )
                    summary.append(
                        pl.DataFrame(
                            {"arm": arm, "stat": name, "kind": kind, "step": step["s"].drop_nulls()}
                        )
                    )
    print("\n== smoothness (largest half-hour change, points of capacity, window +-6 h) ==")
    print(
        pl.concat(summary)
        .group_by("arm", "stat", "kind")
        .agg(
            n=pl.len(),
            median=pl.col("step").median(),
            p95=pl.col("step").quantile(0.95),
            max=pl.col("step").max(),
        )
        .sort("stat", "kind", "arm")
        .with_columns(pl.selectors.float().round(3))
    )


def report() -> None:
    """Print the tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(250)
    wide_all = pl.read_parquet(WORK / "wide.parquet")
    events = pl.read_parquet(blend.WORK / "events_corrected.parquet").with_columns(
        pl.col("event_id").cast(pl.Int64)
    )
    scored = wide_all.filter(pl.col("valid_time") >= SCORED_FROM)
    print(
        "flagged half-hours dropped per Primary (scored months, days 1 to 10):",
        scored.filter(pl.col("flagged")).group_by("label").len().sort("label").rows(),
        "total",
        scored.filter(pl.col("flagged")).height,
    )
    wide = scored.filter(~pl.col("flagged"))
    headline_99(wide=wide)
    leave_one_out(wide=wide)
    median_error_check()
    table(wide=wide, tag="all rows")
    table(wide=wide.filter(pl.col("cold_nights")), tag="cold nights")
    per_primary(wide=wide)
    smoothness(wide=wide, events=events)
    chosen = json.loads((WORK / "per_lead_weights.json").read_text())
    print("\n== per-lead weights on the forecast with lags (lead days 1 to 10), by scored month ==")
    for month, w in chosen.items():
        print(month, w)
    print(
        "mean over months:",
        np.round(np.mean(list(chosen.values()), axis=0), 3).tolist(),
    )


def curtailment() -> None:
    """Print avoidable curtailment (experiment 383's definition) against the references."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    actuals = ctx["actuals"]
    flows = actuals.rename({"actual": "flow_mw"})
    signed = series.filter(pl.col("label").is_in(list(SIGNED_MW_LABELS)))
    tbl = pl.read_parquet(blend.PART_F / "slice_table.parquet").filter(
        pl.col("valid_time") >= SCORED_FROM
    )
    frames = {
        a: pl.read_parquet(WORK / f"arm_{a}.parquet")
        for a in ("lags", "lagless", "mix_w30", "mix_w50", "mix_w70", "mix_lead", "avg_w50",
                  "mix_w50_x115", "mix_lead_x115", "lags_x115")
    }  # fmt: skip
    frames["mix_b2"] = pl.read_parquet(blend.WORK / "arm_mix_b2.parquet")
    frames["stack"] = base.stack_frames(series=series)["stack"]
    sets = {
        "during events": pl.col("event_rows"),
        "7 days after events": pl.col("event_following_7_days"),
        "all rows": pl.lit(value=True),
    }
    for name, cond in sets.items():
        keys = tbl.filter(cond).select(*KEY_COLUMNS)
        forecasts = {
            n: f.join(keys, on=list(KEY_COLUMNS), how="semi").join(
                actuals, on=["time_series_id", "valid_time"]
            )
            for n, f in frames.items()
        }
        parts = []
        for label in SIGNED_MW_LABELS:
            one = signed.filter(pl.col("label") == label)
            try:
                parts.append(
                    metrics.avoidable_curtailment_cost(
                        forecasts=forecasts, series=one, flows=flows,
                        scored_start=base.SCORED_FIRST, fraction=0.9,
                    )
                )  # fmt: skip
            except ValueError as error:
                print(f"{label}: {name}: no cost ({error})")
        wide = (
            pl.concat(parts)
            .pivot(on="arm", index="label", values="avoidable_cost_gbp")
            .sort("label")
        )
        print(f"\n== avoidable curtailment cost, GBP, limit 0.9 x max, {name} ==")
        print(wide)
        totals = {c: float(wide[c].sum()) for c in wide.columns if c != "label"}
        print(
            "total:", {k: round(v) for k, v in totals.items()},
            "| minus lags:", {k: round(v - totals["lags"]) for k, v in totals.items()},
            "| minus stack:", {k: round(v - totals["stack"]) for k, v in totals.items()},
        )  # fmt: skip


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"wide": build_wide, "report": report, "curtailment": curtailment}[what]()

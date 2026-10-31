"""Part F of experiment 395, report step: print every number the write-up quotes.

Reads the tagged per-row tables of `e00395_part_f_build.py`, one forecast at a time. Every table is
given with the flagged exact-zero half-hours dropped (`filtered`, the headline) and kept
(`unfiltered`, comparable with experiment 381). Intervals resample whole events.
"""
# ruff: noqa: E501

import gc
import sys
from pathlib import Path
from typing import Final

import e00395_part_f_build as build  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

WORK: Final[Path] = build.WORK
KEYS: Final[list[str]] = build.KEYS
ARMS: Final[tuple[str, ...]] = (
    "bundle",
    "monthly_control",
    "stack",
    "w383_d",
    "w383_c1",
    "w383_c2",
)
SIX: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
EIGHT: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
GROUPS: Final[dict[str, list[str]]] = {"six": SIX, "eight": EIGHT}
COLUMNS: Final[list[str]] = [
    *KEYS,
    "label",
    "crps",
    "coverage_q95",
    "coverage_q99",
    "coverage_q05",
    "coverage_q01",
    "lead_hours",
    "flagged",
    "q50_mw",
    "actual_mw",
    "width90",
]
HEADLINE_LEAD: Final[float] = 240.0
N_RESAMPLES: Final[int] = 2000


def load(name: str, tags: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's headline-lead rows joined to the tags."""
    return (
        pl.read_parquet(WORK / f"rows_{name}.parquet", columns=COLUMNS)
        .filter(pl.col("lead_hours") < HEADLINE_LEAD)
        .join(tags, on=KEYS, how="left", validate="1:1")
    )


def event_interval(*, sums: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    """Return the pooled mean and a 95% interval from resampling whole events."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(sums), size=(N_RESAMPLES, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(sums.sum() / counts.sum()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def harm(*, rows: pl.DataFrame, labels: list[str]) -> dict:
    """Return the harm of lag-in-event rows (target outside any event) over clean-lag rows."""
    outside = rows.filter(pl.col("label").is_in(labels), pl.col("target_event_id") < 0)
    clean = (
        outside.filter(pl.col("lag_count") == 0).group_by("label").agg(base=pl.col("crps").mean())
    )
    hit = (
        outside.filter(pl.col("lag_count") > 0)
        .join(clean, on="label")
        .with_columns(diff=pl.col("crps") - pl.col("base"))
    )
    per_event = hit.group_by("lag_event_id").agg(s=pl.col("diff").sum(), n=pl.len())
    if per_event.height < 2:
        return {"events": per_event.height}
    mean, lo, hi = event_interval(sums=per_event["s"].to_numpy(), counts=per_event["n"].to_numpy())
    return {
        "harm": round(mean, 3),
        "ci": f"({lo:+.3f} to {hi:+.3f})",
        "events": per_event.height,
        "rows": hit.height,
    }


def report() -> None:
    """Print the Part F tables."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(220)
    tags = pl.read_parquet(WORK / "tags.parquet")
    events = pl.read_parquet(WORK / "events_unpadded.parquet").with_columns(
        pl.col("event_id").cast(pl.Int64)
    )
    print(
        "== F0: unpadded events per Primary (merged intervals), all and those with scored rows =="
    )
    scored_events = set(
        tags.filter(pl.col("target_event_id") >= 0)["target_event_id"].unique().to_list()
    ) | set(tags.filter(pl.col("lag_event_id") >= 0)["lag_event_id"].unique().to_list())
    print(
        events.with_columns(scored=pl.col("event_id").is_in(list(scored_events)))
        .group_by("label")
        .agg(
            events=pl.len(),
            scored=pl.col("scored").sum(),
            median_days=pl.col("length_days").median(),
            median_depth_share=pl.col("depth_share").median(),
        )
        .sort("label")
    )
    print(
        "length classes:",
        events.group_by("length_class").len().to_dicts(),
        "depth share quartiles:",
        events["depth_share"].quantile(0.25),
        events["depth_share"].median(),
        events["depth_share"].quantile(0.75),
    )
    base_rows = load("bundle", tags)
    print(
        "\n== F0b: rows by lag count and target status (bundle rows, days 1-10, all 16 Primaries) =="
    )
    print(
        base_rows.group_by(pl.col("lag_count"), inside=pl.col("target_event_id") >= 0)
        .len()
        .sort("lag_count", "inside")
    )
    print(
        "rows with target inside an event, by hours since onset:",
        base_rows.filter(pl.col("target_event_id") >= 0)
        .group_by(h=pl.col("hours_since_onset") < 24)
        .len()
        .to_dicts(),
    )
    del base_rows
    gc.collect()
    print(
        "\n== F1: harm of a lag inside an unpadded event, rows with target outside every event =="
    )
    for flt in ("filtered", "unfiltered"):
        for name in ARMS:
            rows = load(name, tags)
            if flt == "filtered":
                rows = rows.filter(~pl.col("flagged"))
            for gname, labs in GROUPS.items():
                sub = rows.filter(pl.col("label").is_in(labs), pl.col("target_event_id") < 0)
                by = (
                    sub.group_by("lag_count")
                    .agg(
                        n=pl.len(),
                        crps=pl.col("crps").mean(),
                        q95=pl.col("coverage_q95").mean(),
                        q99=pl.col("coverage_q99").mean(),
                    )
                    .sort("lag_count")
                    .with_columns(pl.selectors.float().round(3))
                )
                print(flt, name, gname, by.rows(), harm(rows=rows, labels=labs))
            del rows
            gc.collect()
    print("\n== F2: depth missed by the median per event, |median(actual - q50)| / |depth| ==")
    depth = events.select("event_id", "depth_mw", "label")
    for name in ARMS:
        rows = (
            load(name, tags)
            .filter(~pl.col("flagged"), pl.col("target_event_id") >= 0)
            .join(
                depth.select("event_id", "depth_mw"), left_on="target_event_id", right_on="event_id"
            )
            .with_columns(
                res=pl.col("actual_mw") - pl.col("q50_mw"),
                cov90=pl.col("coverage_q95") - pl.col("coverage_q05"),
                group=pl.when(pl.col("lag_count") == 0)
                .then(pl.lit("0 lags"))
                .when(pl.col("lag_count") == 3)
                .then(pl.lit("3 lags"))
                .otherwise(pl.lit("1 or 2 lags")),
            )
        )
        per_event = (
            rows.group_by("group", "target_event_id")
            .agg(
                m=pl.col("res").median(),
                d=pl.col("depth_mw").first(),
                n=pl.len(),
                cov90=pl.col("cov90").mean(),
                width90=pl.col("width90").mean(),
            )
            .filter(pl.col("n") >= 10)
            .with_columns(missed=pl.col("m").abs() / pl.col("d").abs())
        )
        by = per_event.group_by("group").agg(
            events=pl.len(),
            missed=pl.col("missed").median(),
            cov90=pl.col("cov90").mean(),
            width90=pl.col("width90").median(),
        )
        print(name, by.sort("group").with_columns(pl.selectors.float().round(3)).rows())
        if name == "bundle":
            zero = per_event.filter(pl.col("group") == "0 lags").join(
                depth.select("event_id", "label"), left_on="target_event_id", right_on="event_id"
            )
            print(
                "F2b sign agreement per Primary, 0 lags:",
                zero.group_by("label")
                .agg(events=pl.len(), same=(pl.col("m") * pl.col("d") > 0).mean())
                .sort("label")
                .rows(),
            )
        del rows
        gc.collect()
    print(
        "\n== F3: unpadded against padded (bundle, signed-MW six): q95 coverage on lag-contaminated rows with target outside the event =="
    )
    padded = pl.read_parquet(WORK / "padded_flags.parquet").select(
        *KEYS, "lag_in_event", "target_in_event"
    )
    rows = (
        load("bundle", tags)
        .join(padded, on=KEYS, how="left", validate="1:1")
        .filter(pl.col("label").is_in(SIX), ~pl.col("flagged"))
    )
    p = rows.filter(pl.col("lag_in_event"), ~pl.col("target_in_event"))
    u = rows.filter(pl.col("lag_count") > 0, pl.col("target_event_id") < 0)
    print(
        "padded mask: rows",
        p.height,
        "q95",
        round(p["coverage_q95"].mean(), 3),  # ty: ignore[no-matching-overload]
        "q99",
        round(p["coverage_q99"].mean(), 3),  # ty: ignore[no-matching-overload]
        "crps",
        round(p["crps"].mean(), 3),  # ty: ignore[no-matching-overload]
    )
    print(
        "unpadded:   rows",
        u.height,
        "q95",
        round(u["coverage_q95"].mean(), 3),  # ty: ignore[no-matching-overload]
        "q99",
        round(u["coverage_q99"].mean(), 3),  # ty: ignore[no-matching-overload]
        "crps",
        round(u["crps"].mean(), 3),  # ty: ignore[no-matching-overload]
    )
    print(
        "padded-contaminated rows with no lag in the unpadded event:",
        rows.filter(pl.col("lag_in_event"), pl.col("lag_count") == 0).height,
        "of",
        rows.filter(pl.col("lag_in_event")).height,
    )
    pe = p.group_by("lag_event_id").agg(s=pl.col("coverage_q95").sum(), n=pl.len())
    ue = u.group_by("lag_event_id").agg(s=pl.col("coverage_q95").sum(), n=pl.len())
    print(
        "padded q95, whole-event interval:",
        event_interval(sums=pe["s"].to_numpy(), counts=pe["n"].to_numpy()),
        "events",
        pe.height,
    )
    print(
        "unpadded q95, whole-event interval:",
        event_interval(sums=ue["s"].to_numpy(), counts=ue["n"].to_numpy()),
        "events",
        ue.height,
    )
    print(
        "\n== F4: q95 coverage by lag count and event length, bundle (target outside events, filtered) =="
    )
    rows = (
        load("bundle", tags)
        .filter(~pl.col("flagged"), pl.col("target_event_id") < 0)
        .join(
            events.select("event_id", "length_class", "depth_share"),
            left_on="lag_event_id",
            right_on="event_id",
            how="left",
        )
    )
    print(
        rows.filter(pl.col("label").is_in(SIX))
        .group_by("length_class", "lag_count")
        .agg(
            n=pl.len(), q95=pl.col("coverage_q95").mean(), events=pl.col("lag_event_id").n_unique()
        )
        .sort("length_class", "lag_count")
        .with_columns(pl.selectors.float().round(3))
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    report()

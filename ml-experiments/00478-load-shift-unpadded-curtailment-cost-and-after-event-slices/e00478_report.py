"""Experiment 478 (Part F2 of experiment 395), report step: print the numbers and draw the charts.

Prints every number the write-up quotes and draws the per-forecast coverage chart.

Reads experiment 395's Part F row files and tags, experiment 478's row files for experiment 391's
forecasts, and the cost table of `e00478_build.py`. Flagged exact-zero half-hours are dropped.
"Rows with no labelled event" means rows with no event in NGED's (incomplete) labels. Intervals
resample whole events.

```bash
D=ml-experiments
PYTHONPATH=$D/00377-demand-v1-first-run-feasibility:\
$D/00381-simple-weekly-xgboost-demand-against-gam-stack:\
$D/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
$D/00390-xgboost-d11-d14-gap-and-load-shift-event-lags:\
$D/00395-xgboost-level-state-winter-spread-anchor-stored-forecasts \
    uv run --no-sync python $D/00478-*/e00478_report.py
```
"""
# ruff: noqa: E501

import gc
import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00395_part_f_build as pf  # ty: ignore[unresolved-import]
import e00395_part_h_rescore as ph  # ty: ignore[unresolved-import]
import e00478_build as build
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

KEYS: Final[list[str]] = pf.KEYS
SIX: Final[list[str]] = build.SIX
EIGHT: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
GROUPS: Final[dict[str, list[str]]] = {"six": SIX, "eight": EIGHT}
ARMS: Final[tuple[str, ...]] = (
    "bundle", "monthly_control", "stack", "w383_d", "w383_c1", "w383_c2",
    "b391_b0", "b391_n1", "b391_o",
)  # fmt: skip
NAMES: Final[dict[str, str]] = {
    "bundle": "Weekly bundle (381)", "monthly_control": "Monthly control (381)",
    "stack": "GAM stack (377)", "w383_d": "Plain weekly member (383 d)",
    "w383_c1": "383 c1", "w383_c2": "383 c2", "b391_b0": "391 control (b0)",
    "b391_n1": "391 surprise (n1)", "b391_o": "391 oracle (o)",
}  # fmt: skip
COLUMNS: Final[list[str]] = [
    *KEYS, "label", "crps", "coverage_q95", "coverage_q99", "lead_hours", "flagged", "q50_mw", "actual_mw",
]  # fmt: skip
HEADLINE_LEAD: Final[float] = 240.0
N_RESAMPLES: Final[int] = 2000
LEVEL_PRIMARY: Final[float] = 95.0
LEVEL_POOLED: Final[float] = 100 * (1 - 0.05 / 6)
REFERENCE: Final[str] = "w383_d"
OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / "00478-load-shift-unpadded-curtailment-cost-and-after-event-slices"
)  # fmt: skip
POSITIONS: Final[tuple[str, ...]] = (
    "0-24 h after the lag's event",
    "1-3 days after",
    "3-7 days after",
    "7-21 days after",
)


def rows_path(name: str) -> Path:
    """Return the row file of a forecast (experiment 391's are in this experiment's folder)."""
    return (build.WORK if name.startswith("b391_") else pf.WORK) / f"rows_{name}.parquet"


def load(name: str, tags: pl.DataFrame) -> pl.DataFrame:
    """Return a forecast's headline-lead rows without flagged half-hours, joined to the tags."""
    return (
        pl.read_parquet(rows_path(name), columns=COLUMNS)
        .filter(pl.col("lead_hours") < HEADLINE_LEAD, ~pl.col("flagged"))
        .join(tags, on=KEYS, how="inner", validate="1:1")
    )


def interval(
    *, sums: np.ndarray, counts: np.ndarray, level: float = LEVEL_PRIMARY
) -> tuple[float, float, float]:
    """Return the pooled mean and an interval from resampling whole events."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(sums), size=(N_RESAMPLES, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100 - level) / 2
    return (
        float(sums.sum() / counts.sum()),
        float(np.percentile(means, tail)),
        float(np.percentile(means, 100 - tail)),
    )


def position_expr() -> pl.Expr:
    """Return the hours from the end of the lag's own event to the target, as a position bucket."""
    h = pl.col("hours_after_lag_event_end")
    return (
        pl.when(h <= 24).then(pl.lit(POSITIONS[0]))
        .when(h <= 72).then(pl.lit(POSITIONS[1]))
        .when(h <= 168).then(pl.lit(POSITIONS[2]))
        .otherwise(pl.lit(POSITIONS[3]))
    )  # fmt: skip


def harm_slices(
    *, rows: pl.DataFrame, labels: list[str], events: pl.DataFrame, by: str
) -> list[tuple]:
    """Return (slice, harm, interval, events, rows) of lag-in-event rows, targets outside events.

    The baseline is each Primary's mean CRPS on rows with three clean lags and a target outside every
    labelled event (the baseline of experiment 395 Part F).
    """
    outside = rows.filter(pl.col("label").is_in(labels), pl.col("target_event_id") < 0)
    base = (
        outside.filter(pl.col("lag_count") == 0).group_by("label").agg(base=pl.col("crps").mean())
    )
    hit = (
        outside.filter(pl.col("lag_count") > 0)
        .join(base, on="label")
        .join(
            events.select("event_id", "length_class", "depth_quartile", "end_idx"),
            left_on="lag_event_id",
            right_on="event_id",
            how="left",
        )
        .with_columns(diff=pl.col("crps") - pl.col("base"))
    )
    hit = hit.with_columns(
        hours_after_lag_event_end=(
            pl.Series(common.grid_index(times=hit["valid_time"])) - hit["end_idx"]
        )
        * pf.STEP_HOURS
    ).with_columns(position=position_expr())
    if by == "lag_count":
        hit = hit.with_columns(slice=pl.col("lag_count").cast(pl.String))
    else:
        hit = hit.with_columns(slice=pl.col(by))
    out = []
    for key in sorted(hit["slice"].unique().to_list()):
        sub = hit.filter(pl.col("slice") == key)
        per = sub.group_by("lag_event_id").agg(s=pl.col("diff").sum(), n=pl.len())
        if per.height < 5:
            out.append((key, None, None, per.height, sub.height))
            continue
        m, lo, hi = interval(sums=per["s"].to_numpy(), counts=per["n"].to_numpy())
        out.append((key, round(m, 3), f"({lo:+.3f} to {hi:+.3f})", per.height, sub.height))
    return out


def cost_tables(*, events: pl.DataFrame) -> None:
    """Print each forecast's avoidable-curtailment cost minus two references, with event intervals."""
    cost = pl.read_parquet(build.WORK / "cost.parquet").filter(pl.col("status") == "ok")
    print("\n== C: avoidable-curtailment cost (pounds), signed-MW six ==")
    pooled = cost.filter(pl.col("event_id").is_null())
    for frac in build.FRACTIONS:
        print(f"\n-- f = {frac}: cost per Primary, all arms --")
        print(
            pooled.filter(pl.col("fraction") == frac)
            .pivot(on="arm", index=["row_set", "label"], values="avoidable_cost_gbp")
            .sort("row_set", "label")
            .with_columns(pl.selectors.float().round(0))
        )
    for ref in (REFERENCE, "stack"):
        for frac in build.FRACTIONS:
            print(
                f"\n-- f = {frac}: cost minus {NAMES[ref]}, per-event mean and interval (events resampled) --"
            )
            for row_set in ("during events", "7 days after events"):
                ev = cost.filter(
                    pl.col("event_id").is_not_null(),
                    pl.col("row_set") == row_set,
                    pl.col("fraction") == frac,
                )
                wide = ev.pivot(on="arm", index=["label", "event_id"], values="avoidable_cost_gbp")
                if ref not in wide.columns:
                    continue
                for arm in ARMS:
                    if arm == ref or arm not in wide.columns:
                        continue
                    d = wide.filter(
                        pl.col(arm).is_not_null(), pl.col(ref).is_not_null()
                    ).with_columns(diff=pl.col(arm) - pl.col(ref))
                    line = []
                    for label in [*SIX, "pooled"]:
                        sub = d if label == "pooled" else d.filter(pl.col("label") == label)
                        if sub.height < 3:
                            line.append(f"{label}: {sub.height} events")
                            continue
                        level = LEVEL_POOLED if label == "pooled" else LEVEL_PRIMARY
                        m, lo, hi = interval(
                            sums=sub["diff"].to_numpy(), counts=np.ones(sub.height), level=level
                        )
                        line.append(f"{label}: {m:+.0f} ({lo:+.0f} to {hi:+.0f}; {sub.height} ev)")
                    print(f"{row_set} | {arm} | " + " | ".join(line))
            print("all rows, per Primary difference (no interval):")
            wide = pooled.filter(pl.col("fraction") == frac, pl.col("row_set") == "all rows").pivot(
                on="arm", index="label", values="avoidable_cost_gbp"
            )
            if ref in wide.columns:
                print(
                    wide.with_columns(
                        *[
                            (pl.col(a) - pl.col(ref)).alias(f"{a}-{ref}")
                            for a in ARMS
                            if a in wide.columns and a != ref
                        ]
                    )
                    .select("label", pl.selectors.ends_with(f"-{ref}"))
                    .sort("label")
                    .with_columns(pl.selectors.float().round(0))
                )


def coverage_chart(*, name: str, rows: pl.DataFrame, events: pl.DataFrame) -> None:
    """Draw q95 coverage by lag count and length of the lag's event for one forecast."""
    sub = (
        rows.filter(pl.col("label").is_in(SIX), pl.col("target_event_id") < 0)
        .join(
            events.select("event_id", "length_class"),
            left_on="lag_event_id",
            right_on="event_id",
            how="left",
        )
        .with_columns(pl.col("length_class").fill_null("rows with no labelled event"))
        .group_by("length_class", "lag_count")
        .agg(q95=pl.col("coverage_q95").mean(), n=pl.len())
        .filter(pl.col("n") >= 30)
    )
    order = ["rows with no labelled event", "under 1 day", "1 to 7 days", "over 7 days"]
    chart = (
        alt.Chart(sub)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("lag_count:O", title="Power lags inside an event (of 3)"),
            y=alt.Y("q95:Q", title="Share of readings below q95", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "lag_count:O",
                legend=None,
                scale=alt.Scale(
                    range=[ocf.BRAND_ORANGE if hasattr(ocf, "BRAND_ORANGE") else "#FF4901"] * 4
                ),
            ),
            column=alt.Column(
                "length_class:N", sort=order, title="Length of the event holding the lag"
            ),
        )
        .properties(
            width=130,
            height=160,
            title=alt.TitleParams(
                f"{NAMES[name]}: q95 coverage at the signed-MW six, targets outside events (nominal 0.95)",
                anchor="start",
                fontSize=11,
            ),
        )
    )  # ty: ignore[unresolved-attribute]
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(OUT / f"coverage_{name}.svg")


def report() -> None:
    """Print the Part F2 tables."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(220)
    tags = pl.read_parquet(pf.WORK / "tags.parquet")
    events = pl.read_parquet(pf.WORK / "events_unpadded.parquet").with_columns(
        pl.col("event_id").cast(pl.Int64)
    )
    events = ph.corrected_events(events=events)
    quart = (
        events["depth_share"].quantile(0.25),
        events["depth_share"].median(),
        events["depth_share"].quantile(0.75),
    )
    print("depth share quartile cuts (% of capacity):", quart)
    events = events.with_columns(
        depth_quartile=pl.when(pl.col("depth_share") <= quart[0]).then(pl.lit("depth Q1 (smallest)"))
        .when(pl.col("depth_share") <= quart[1]).then(pl.lit("depth Q2"))
        .when(pl.col("depth_share") <= quart[2]).then(pl.lit("depth Q3"))
        .otherwise(pl.lit("depth Q4 (largest)"))
    )  # fmt: skip
    print(
        "events",
        events.height,
        "with corrected depth",
        events["corrected_depth_mw"].is_not_null().sum(),
    )
    print("events per Primary:", events.group_by("label").len().sort("label").rows())
    for name in ARMS:
        rows = load(name, tags)
        print(f"\n===== {NAMES[name]}: rows {rows.height} =====")
        for gname, labs in GROUPS.items():
            for by in ("lag_count", "position", "length_class", "depth_quartile"):
                print(
                    f"H harm [{name}] [{gname}] by {by}:",
                    harm_slices(rows=rows, labels=labs, events=events, by=by),
                )
        # share of the corrected depth followed by the median, rows inside events
        inside = (
            rows.filter(pl.col("target_event_id") >= 0)
            .join(events.select("event_id", "corrected_depth_mw"), left_on="target_event_id", right_on="event_id")
            .filter(pl.col("corrected_depth_mw").is_not_null())
            .with_columns(
                group=pl.when(pl.col("lag_count") == 0).then(pl.lit("0 lags"))
                .when(pl.col("lag_count") == 3).then(pl.lit("3 lags")).otherwise(pl.lit("1 or 2 lags")),
                share=1 - (pl.col("actual_mw") - pl.col("q50_mw")) / pl.col("corrected_depth_mw"),
            )
        )  # fmt: skip
        per = (
            inside.group_by("group", "target_event_id")
            .agg(share=pl.col("share").median(), n=pl.len())
            .filter(pl.col("n") >= 10)
        )
        print(
            f"S share of depth followed [{name}]:",
            per.group_by("group")
            .agg(events=pl.len(), share=pl.col("share").median())
            .sort("group")
            .with_columns(pl.selectors.float().round(2))
            .rows(),
        )
        coverage_chart(name=name, rows=rows, events=events)
        del rows, inside
        gc.collect()
    cost_tables(events=events)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    report()

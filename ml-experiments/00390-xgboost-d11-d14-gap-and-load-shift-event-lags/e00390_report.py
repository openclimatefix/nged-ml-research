"""Print every number experiment 390's write-up quotes, from the tables `e00390_build.py` wrote.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00390-*/e00390_report.py a|b
```
"""
# ruff: noqa: E501, PLR0915

import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00390-xgboost-d11-d14-gap-and-load-shift-event-lags" / "work"
)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
HEADLINE: Final[str] = "lead_days_1_10"
SIGNED_SIX: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
GAP_PRIMARIES: Final[list[str]] = ["D11", "D14"]
N_RESAMPLES: Final[int] = 2_000
RNG_SEED: Final[int] = 0
TAILS: Final[list[str]] = ["coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99"]
PINBALLS: Final[list[str]] = ["pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99"]
LEAD_BANDS: Final[list[tuple[str, float, float]]] = [
    ("days 1-3", 0.0, 72.0),
    ("days 4-7", 72.0, 168.0),
    ("days 8-10", 168.0, 240.0),
]


def load(name: str, table: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's rows joined to the slice table and the event flags."""
    rows = pl.read_parquet(WORK / f"rows_{name}.parquet")
    flags = pl.read_parquet(WORK / "event_flags.parquet")
    return rows.join(table, on=KEYS, how="left", validate="1:1").join(
        flags, on=KEYS, how="left", validate="1:1"
    )


def band_expr() -> pl.Expr:
    """Return an expression naming the lead band of days 1 to 10."""
    return (
        pl.when(pl.col("lead_hours") < 72)
        .then(pl.lit("days 1-3"))
        .when(pl.col("lead_hours") < 168)
        .then(pl.lit("days 4-7"))
        .otherwise(pl.lit("days 8-10"))
    )


def part_a(table: pl.DataFrame) -> None:
    """Print A1 to A5."""
    names = ["bundle", "stack", "bg2", "bg2_only_W", "bundle_offset_3d", "bundle_offset_7d",
             "bundle_offset_14d"]  # fmt: skip
    arms = {n: load(n, table).filter(pl.col(HEADLINE)) for n in names}
    print("\n== A1: CRPS (points), days 1 to 10, per Primary ==")
    table_a1 = None
    for n, a in arms.items():
        piece = a.group_by("label").agg(pl.col("crps").mean().alias(n))
        table_a1 = piece if table_a1 is None else table_a1.join(piece, on="label")
    table_a1 = table_a1.with_columns(gap=pl.col("bundle") - pl.col("stack")).sort("label")  # ty: ignore[unresolved-attribute]
    print(table_a1)
    for lab in GAP_PRIMARIES:
        rows = arms["bundle"].filter(pl.col("label") == lab).height
        print(f"{lab}: rows {rows}")
    print("\n== A1: bundle minus stack, 95% month bootstrap, signed-MW group and per Primary ==")
    for lab in [*GAP_PRIMARIES, "all signed-MW"]:
        where = (
            pl.col("label").is_in(SIGNED_SIX) if lab == "all signed-MW" else pl.col("label") == lab
        ) & pl.col(HEADLINE)
        r = metrics.month_bootstrap_contrast(
            treatment=load("bundle", table), reference=load("stack", table), metric="crps",
            where=where, slice_table=table,
        )  # fmt: skip
        print(f"{lab:14s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) "
              f"n={r['n_half_hours']}")  # fmt: skip

    print("\n== A2: split of the bundle-minus-stack gap ==")
    key = ["time_series_id", "power_fcst_init_time", "valid_time"]
    for lab in [*GAP_PRIMARIES, "signed-MW six"]:
        sel = SIGNED_SIX if lab == "signed-MW six" else [lab]
        b = arms["bundle"].filter(pl.col("label").is_in(sel))
        s = arms["stack"].filter(pl.col("label").is_in(sel)).select(*key, stack_crps="crps")
        j = b.join(s, on=key).with_columns(
            gap=pl.col("crps") - pl.col("stack_crps"),
            band=band_expr(),
            hour=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour(),
            season=pl.col("valid_time")
            .dt.convert_time_zone("Europe/London")
            .dt.month()
            .replace_strict(
                {
                    12: "winter",
                    1: "winter",
                    2: "winter",
                    3: "spring",
                    4: "spring",
                    5: "spring",
                    6: "summer",
                    7: "summer",
                    8: "summer",
                    9: "autumn",
                    10: "autumn",
                    11: "autumn",
                }
            ),
            sunny=pl.when(pl.col("sunny_unit_shape"))
            .then(pl.lit("sunny"))
            .otherwise(pl.lit("not sunny")),
            after_event=pl.when(pl.col("event_following_7_days") | pl.col("event_rows"))
            .then(pl.lit("event or 7 days after"))
            .otherwise(pl.lit("other")),
            hour_block=(pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour() // 6)
            * 6,
        )
        j = j.with_columns(
            demand_quartile=pl.col("actual_points")
            .qcut(  # ty: ignore[deprecated]
                [0.25, 0.5, 0.75],
                labels=["Q1 low", "Q2", "Q3", "Q4 high"],
                allow_duplicates=True,
            )
            .over("time_series_id")
            .cast(pl.String)
        )
        total = j["gap"].sum()
        print(f"\n-- {lab}: rows {j.height}, mean gap {j['gap'].mean():+.3f}, total {total:.0f}")
        for col in ("band", "hour_block", "season", "sunny", "demand_quartile", "after_event"):
            t = (
                j.group_by(col)
                .agg(
                    rows=pl.len(), mean_gap=pl.col("gap").mean(), share=pl.col("gap").sum() / total
                )
                .sort(col)
            )
            print(t)

    print("\n== A3: level-tracking offsets, CRPS days 1 to 10 and share of the gap closed ==")
    for lab in [*GAP_PRIMARIES, "signed-MW six"]:
        sel = SIGNED_SIX if lab == "signed-MW six" else [lab]
        base_b = arms["bundle"].filter(pl.col("label").is_in(sel))["crps"].mean()
        base_s = arms["stack"].filter(pl.col("label").is_in(sel))["crps"].mean()
        print(f"{lab}: bundle {base_b:.3f}, stack {base_s:.3f}, gap {base_b - base_s:+.3f}")  # ty: ignore[unsupported-operator]
        for n in ("bundle_offset_3d", "bundle_offset_7d", "bundle_offset_14d"):
            v = arms[n].filter(pl.col("label").is_in(sel))["crps"].mean()
            closed = (base_b - v) / (base_b - base_s) if base_b != base_s else float("nan")  # ty: ignore[unsupported-operator]  # ty: ignore[unsupported-operator]
            where = pl.col("label").is_in(sel) & pl.col(HEADLINE)
            r = metrics.month_bootstrap_contrast(
                treatment=load(n, table), reference=load("bundle", table), metric="crps",
                where=where, slice_table=table,
            )  # fmt: skip
            print(f"   {n}: {v:.3f} ({v - base_b:+.3f} vs bundle, 95% {r['lower']:+.3f} to "  # ty: ignore[unsupported-operator]
                  f"{r['upper']:+.3f}); closes {closed:.0%} of the gap")  # fmt: skip
    try:
        off = load("stack_kalman_off", table).filter(pl.col(HEADLINE))
        print("\n-- stack without the Kalman filter (experiment 380 drop_k), same rows --")
        for lab in [*GAP_PRIMARIES, "signed-MW six"]:
            sel = SIGNED_SIX if lab == "signed-MW six" else [lab]
            o = off.filter(pl.col("label").is_in(sel))["crps"].mean()
            s_ = arms["stack"].filter(pl.col("label").is_in(sel))["crps"].mean()
            b_ = arms["bundle"].filter(pl.col("label").is_in(sel))["crps"].mean()
            print(f"{lab}: stack {s_:.3f}, stack without Kalman {o:.3f}, bundle {b_:.3f}")
    except FileNotFoundError:
        print("stack_kalman_off not built")

    print("\n== A4: BG2 members at D11 and D14, CRPS days 1 to 10 ==")
    members = ["bg2_only_W", "bg2_only_M", "bg2_only_AN", "bg2_only_PT", "bg2_only_ST", "bg2_full"]
    out = None
    for n in ["bundle", "stack", *members]:
        a = arms[n] if n in arms else load(n, table).filter(pl.col(HEADLINE))
        piece = (
            a.filter(pl.col("label").is_in(GAP_PRIMARIES))
            .group_by("label")
            .agg(pl.col("crps").mean().alias(n))
        )
        out = piece if out is None else out.join(piece, on="label")
    print(out.sort("label"))

    print("\n== A5: upper-tail coverage (target 0.95 and 0.99) and pinball at D11 and D14 ==")
    for lab in GAP_PRIMARIES:
        for n in ("bundle", "stack", "bg2"):
            a = (
                arms[n]
                .filter(pl.col("label") == lab)
                .with_columns(
                    band=band_expr(),
                    hour_block=(
                        pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour() // 6
                    )
                    * 6,
                )
            )
            print(f"-- {lab} {n}")
            print(
                a.group_by("band")
                .agg(
                    pl.col("coverage_q95").mean(),
                    pl.col("coverage_q99").mean(),
                    pl.col("pinball_q99").mean(),
                    pl.col("width90").mean(),
                )
                .sort("band")
            )
            print(
                a.group_by("hour_block")
                .agg(
                    pl.col("coverage_q95").mean(),
                    pl.col("coverage_q99").mean(),
                    pl.col("pinball_q99").mean(),
                )
                .sort("hour_block")
            )


def cluster_interval(
    rows: pl.DataFrame, columns: list[str], cluster: str = "event_id"
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    """Return slice means with 95% intervals from resampling whole events."""
    g = rows.group_by(cluster).agg(
        [pl.col(c).sum().alias(f"{c}_s") for c in columns] + [pl.len().alias("n")]
    )
    n_events = g.height
    if n_events == 0:
        return {}, {}, 0
    sums = g.select([f"{c}_s" for c in columns]).to_numpy().astype(float)
    counts = g["n"].to_numpy().astype(float)
    means = {c: float(sums[:, i].sum() / counts.sum()) for i, c in enumerate(columns)}
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, n_events, size=(N_RESAMPLES, n_events))
    tot = counts[draws].sum(axis=1)
    intervals = {}
    for i, c in enumerate(columns):
        est = sums[:, i][draws].sum(axis=1) / tot
        intervals[c] = (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))
    return means, intervals, n_events


def part_b(table: pl.DataFrame) -> None:
    """Print B1 to B4."""
    events = pl.read_parquet(WORK / "events.parquet").select(
        "event_id", "label", "start", "end", "duration_hours", "depth_share", "depth_mw"
    )
    scored = events.filter(pl.col("end") >= pl.datetime(2025, 7, 1, time_zone="UTC"))
    print(f"\nevents in the padded mask: {events.height}; ending in or after the scored year: "
          f"{scored.height}; with a matched depth: {scored['depth_share'].is_not_null().sum()}")  # fmt: skip
    cuts = (
        events["depth_share"].drop_nulls().quantile(0.25),
        events["depth_share"].drop_nulls().quantile(0.5),
        events["depth_share"].drop_nulls().quantile(0.75),
    )
    print("depth quartile cuts (share of capacity, points):", cuts)
    ev = events.with_columns(
        depth_q=pl.when(pl.col("depth_share").is_null())
        .then(pl.lit("unmatched"))
        .when(pl.col("depth_share") <= cuts[0])
        .then(pl.lit("Q1 shallow"))
        .when(pl.col("depth_share") <= cuts[1])
        .then(pl.lit("Q2"))
        .when(pl.col("depth_share") <= cuts[2])
        .then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4 deep")),
        dur_class=pl.when(pl.col("duration_hours") < 24)
        .then(pl.lit("under 24 h"))
        .when(pl.col("duration_hours") <= 168)
        .then(pl.lit("1 to 7 days"))
        .otherwise(pl.lit("over 7 days")),
    )
    cols = [*TAILS, *PINBALLS, "width90", "crps", "q50_error", "mae_median"]
    for name in ("bundle", "bg2_only_W", "stack"):
        rows = load(name, table).filter(pl.col(HEADLINE))
        # Event of the row: the lag's event; hours since onset from the origin.
        with_lag = (
            rows.filter(pl.col("lag_in_event"))
            .join(
                ev.rename({"event_id": "lag_event_id"}), on="lag_event_id", how="left", suffix="_ev"
            )
            .with_columns(
                hours_since_onset=(
                    pl.col("power_fcst_init_time") - pl.col("start")
                ).dt.total_minutes()
                / 60,
                after_end=pl.col("power_fcst_init_time") > pl.col("end"),
                event_id=pl.col("lag_event_id"),
            )
            .with_columns(
                phase=pl.when(pl.col("after_end"))
                .then(pl.lit("after the end, up to 7 days"))
                .when(pl.col("hours_since_onset") < 24)
                .then(pl.lit("first 24 h"))
                .otherwise(pl.lit("1 to 7 days in")),
            )
        )
        clean = rows.filter(~pl.col("lag_in_event"))
        print(
            f"\n===== B1: {name}: rows with a lag in an event {with_lag.height}, clean {clean.height}"
        )
        c_mean = clean.select([pl.col(c).mean() for c in cols]).row(0, named=True)
        e_mean = with_lag.select([pl.col(c).mean() for c in cols]).row(0, named=True)
        print("columns:", cols)
        print("clean :", {k: round(v, 3) for k, v in c_mean.items()})
        print("event :", {k: round(v, 3) for k, v in e_mean.items()})
        for col in ("depth_q", "dur_class", "phase"):
            print(f"-- by {col} (events resampled whole; rows, events, 95% for coverage_q95 / q99)")
            for value in sorted(with_lag[col].unique().to_list()):
                part = with_lag.filter(pl.col(col) == value)
                means, ints, n_ev = cluster_interval(
                    part,
                    [
                        "coverage_q95",
                        "coverage_q99",
                        "coverage_q05",
                        "coverage_q01",
                        "width90",
                        "crps",
                    ],
                )
                desc = " (descriptive: under 5 events)" if n_ev < 5 else ""
                print(f"{value:28s} rows {part.height:6d} events {n_ev:3d}{desc}: "
                      f"q01 {means['coverage_q01']:.3f}, q05 {means['coverage_q05']:.3f}, "
                      f"q95 {means['coverage_q95']:.3f} ({ints['coverage_q95'][0]:.3f}-{ints['coverage_q95'][1]:.3f}), "
                      f"q99 {means['coverage_q99']:.3f} ({ints['coverage_q99'][0]:.3f}-{ints['coverage_q99'][1]:.3f}), "
                      f"width {means['width90']:.1f}, crps {means['crps']:.2f}")  # fmt: skip

        # B2: target in event, three lags clean.
        t = (
            rows.filter(pl.col("target_in_event"), ~pl.col("lag_in_event"))
            .join(ev.rename({"event_id": "target_event_id"}), on="target_event_id", how="left")
            .with_columns(
                event_id=pl.col("target_event_id"),
                inside_band=pl.col("coverage_q95") - pl.col("coverage_q05"),
                below_q01=pl.col("coverage_q01"),
                above_q99=1 - pl.col("coverage_q99"),
            )
        )
        print(f"-- B2 {name}: target inside an event with clean lags: {t.height} rows")
        for value in ["all", *sorted(t["depth_q"].unique().to_list())]:
            part = t if value == "all" else t.filter(pl.col("depth_q") == value)
            means, ints, n_ev = cluster_interval(
                part, ["inside_band", "below_q01", "above_q99", "q50_error"]
            )
            if not n_ev:
                continue
            print(f"{value:12s} rows {part.height:6d} events {n_ev:3d}: inside 5-95 band {means['inside_band']:.3f} "
                  f"({ints['inside_band'][0]:.3f}-{ints['inside_band'][1]:.3f}), below q01 {means['below_q01']:.3f}, "
                  f"above q99 {means['above_q99']:.3f}, signed median error {means['q50_error']:+.2f}")  # fmt: skip

        # B3: median behaviour in events, with event lag against clean lag.
        tin = (
            rows.filter(pl.col("target_in_event"))
            .join(ev.rename({"event_id": "target_event_id"}), on="target_event_id", how="left")
            .with_columns(event_id=pl.col("target_event_id"))
        )
        print(f"-- B3 {name}: target inside an event; signed median error (median minus actual, points) "
              f"and mean absolute error of the median")  # fmt: skip
        for col in ("depth_q", "dur_class"):
            for lagged in (True, False):
                for value in sorted(tin[col].unique().to_list()):
                    part = tin.filter(pl.col(col) == value, pl.col("lag_in_event") == lagged)
                    means, ints, n_ev = cluster_interval(part, ["q50_error", "mae_median"])
                    if not n_ev:
                        continue
                    flag = "lag in event" if lagged else "clean lags  "
                    print(f"{col:9s} {value:14s} {flag}: rows {part.height:6d} events {n_ev:3d}: "
                          f"signed {means['q50_error']:+.2f} ({ints['q50_error'][0]:+.2f} to "
                          f"{ints['q50_error'][1]:+.2f}), MAE {means['mae_median']:.2f}")  # fmt: skip


def main() -> None:
    """Print the requested part."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    table = pl.read_parquet(WORK / "slice_table.parquet")
    if sys.argv[1] == "a":
        part_a(table)
    else:
        part_b(table)


if __name__ == "__main__":
    main()

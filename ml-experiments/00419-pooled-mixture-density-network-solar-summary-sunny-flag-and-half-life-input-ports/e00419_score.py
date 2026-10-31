# ruff: noqa: E501
"""Score experiment 419's input-port arms against experiment 407's stored widened control.

Reads the per-seed forecasts `e00419_fit.py` wrote (and 407's `ctrl` and `hl91`), averages the seeds
in quantile space, widens every arm by the fixed factor 1.15 about the median (`_w115`), and prints:

- the match of every arm's keys and per-fold row counts to the reference, and the per-fold counts of
  rows the fit dropped as flagged exact zeros;
- the minimum detectable effect: 2 times the standard deviation, over the five folds, of the
  signed-MW six CRPS difference between `ctrl_w115` (407, seeds 0 to 2) and `ctrl_s3_w115`;
- CRPS at days 1 to 10 per Primary and per group, CRPS on the sunny April-to-June, cold-night, and
  Economy 7 step slices, the sunny April-to-June difference as a point estimate only, tails on all
  half-hours and on sunny low-demand half-hours, and the avoidable curtailment cost;
- the registered contrasts S1 to S4 with the month-block bootstrap of `studies.eval_protocol` at
  95% (20,000 resamples);
- example forecasts of the best arm (lowest signed-MW six CRPS).

Run in passes: name the arms on the command line (`ctrl_s3_w115 sol_w115`); the reference
`ctrl_w115` is always loaded. With no names, every complete arm is scored.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights \
uv run --no-sync python ml-experiments/00419-*/e00419_score.py sol_w115 ctrl_s3_w115
```
"""

import json
import sys
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT_419: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00419-pooled-mixture-density-network-solar-summary-sunny-flag-and-half-life-input-ports"
)
SEED_419: Final[Path] = ROOT_419 / "work" / "seed_preds"
FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00419-pooled-mixture-density-network-solar-summary-sunny-flag-and-half-life-input-ports"
)
HEADLINE: Final[str] = s407.HEADLINE
SUBSETS: Final[dict[str, tuple[str, ...]]] = s407.SUBSETS
SIGNED: Final[tuple[str, ...]] = s407.SIGNED
REFERENCE: Final[str] = "ctrl_w115"
"""Experiment 407's `ctrl` (seeds 0 to 2) widened by 1.15."""
NEW_ARMS: Final[tuple[str, ...]] = ("ctrl_s3", "sol", "bundle", "hl91r", "sol_hl91")
STORED_407_ARMS: Final[tuple[str, ...]] = ("ctrl", "hl91")
SLICES: Final[tuple[str, ...]] = (
    "sunny_april_june",
    "cold_nights",
    "economy_7_step",
    "sunny_unit_shape",
)
REPORT_ONLY_SLICES: Final[tuple[tuple[str, str | None], ...]] = (
    ("days 1 to 10", None),
    ("cold nights", "cold_nights"),
    ("Economy 7 step", "economy_7_step"),
)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "S0 ctrl_s3 (seeds 3-5) minus reference (seeds 0-2)": ("ctrl_s3_w115", REFERENCE),
    "S1 sol minus reference": ("sol_w115", REFERENCE),
    "S2 bundle minus sol": ("bundle_w115", "sol_w115"),
    "S3 hl91r minus ctrl_s3": ("hl91r_w115", "ctrl_s3_w115"),
    "S3 sanity: 407 hl91 minus reference": ("hl91_w115", REFERENCE),
    "S4 sol_hl91 minus sol": ("sol_hl91_w115", "sol_w115"),
}
CURTAILMENT_FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
EXAMPLES: Final[dict[str, str]] = s407.EXAMPLES
EXAMPLE_ORIGINS: Final[tuple[str, ...]] = s407.EXAMPLE_ORIGINS
KEYS: Final[list[str]] = list(common.KEYS)


def load_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the scored forecasts (MW, one row per key); all but `per_primary_cut_406` are widened by 1.15."""
    dirs: dict[str, Path] = {f"{arm}_w115": s407.SEED_407 / arm for arm in STORED_407_ARMS}
    dirs.update({f"{arm}_w115": SEED_419 / arm for arm in NEW_ARMS})
    dirs["per_primary_cut_406"] = s407.SEED_406 / "per_primary"
    found = {n: d for n, d in dirs.items() if s407.is_complete(arm_dir=d)}
    wanted = set(sys.argv[1:])
    if wanted:
        found = {n: d for n, d in found.items() if n in wanted or n == REFERENCE}
    if REFERENCE not in found:
        msg = f"the reference {REFERENCE} is missing"
        raise ValueError(msg)
    print("arms found:", ", ".join(found))
    return {
        n: adapter.forecast_frame(
            stored=(
                s407.seed_average(arm_dir=d)
                if n == "per_primary_cut_406"
                else s407.widen_global(frame=s407.seed_average(arm_dir=d))
            ),
            series=series,
            unit="fraction",
        )
        for n, d in found.items()
    }


def check_keys(*, frames: dict[str, pl.DataFrame]) -> None:
    """Print per-fold row counts and raise if any arm's keys differ from the reference's."""
    reference = frames[REFERENCE].select(KEYS)
    counts = {
        n: f.group_by(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").alias("fold"))
        .len()
        .sort("fold")
        for n, f in frames.items()
    }
    print("\n== rows per fold (month of forecast origin) ==")
    print(
        pl.concat([c.with_columns(arm=pl.lit(n)) for n, c in counts.items()]).pivot(
            on="arm", index="fold", values="len"
        )
    )
    for name, frame in frames.items():
        keys = frame.select(KEYS)
        extra = keys.join(reference, on=KEYS, how="anti").height
        missing = reference.join(keys, on=KEYS, how="anti").height
        if extra or missing:
            msg = f"{name}: {extra} keys not in the reference and {missing} reference keys missing"
            raise ValueError(msg)
    print("every arm's keys match the reference's")
    path = ROOT_419 / "work" / "dropped_rows.json"
    if path.exists():
        print(
            "training rows dropped as flagged exact zeros, per fold:", json.loads(path.read_text())
        )


def group_means(*, crps: pl.DataFrame, arm: str, members: tuple[str, ...]) -> float:
    """Return the equal-weight mean over Primaries of an arm's headline value in a pivoted table."""
    return float(np.mean(crps.filter(pl.col("label").is_in(list(members)))[arm].to_numpy()))


def headline_table(*, scored: dict[str, pl.DataFrame], metric: str) -> pl.DataFrame:
    """Return one column per arm of a metric at the headline band, one row per Primary."""
    return pl.concat(
        [
            t.filter(pl.col("metric") == metric, pl.col("slice") == HEADLINE).select(
                "label", pl.lit(n).alias("arm"), "value"
            )
            for n, t in scored.items()
        ]
    ).pivot(on="arm", index="label", values="value")


def draw_arm_examples(*, arm: str, frame: pl.DataFrame, ctx: dict, series: pl.DataFrame) -> None:
    """Draw 407's example charts for one arm, with the arm name in each SVG filename."""
    scratch = FIGURE_DIR / f"_scratch_{arm}"
    s407.FIGURE_DIR = scratch  # ty: ignore[invalid-assignment]
    s407.draw_examples(frame=frame, ctx=ctx, series=series)
    for path in scratch.glob("figure_example_*.svg"):
        path.replace(FIGURE_DIR / path.name.replace("figure_example_", f"figure_example_{arm}_", 1))
    scratch.rmdir()


def minimum_detectable_effect(*, rows: dict[str, pl.DataFrame]) -> None:
    """Print 2 times the SD over the five folds of the signed-MW six CRPS gap between two fits."""
    if "ctrl_s3_w115" not in rows:
        print("\nMDE not computed: ctrl_s3_w115 is not in this pass")
        return
    per_fold = []
    for name in (REFERENCE, "ctrl_s3_w115"):
        part = rows[name].filter(pl.col("label").is_in(list(SIGNED)) & pl.col(HEADLINE))
        per_fold.append(
            part.with_columns(fold=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
            .group_by("label", "fold")
            .agg(pl.col("crps").mean())
            .group_by("fold")
            .agg(pl.col("crps").mean().alias(name))
        )
    gap = per_fold[0].join(per_fold[1], on="fold").sort("fold")
    diff = (gap[REFERENCE] - gap["ctrl_s3_w115"]).to_numpy()
    print("\n== minimum detectable effect, signed-MW six, days 1 to 10 ==")
    print(gap.with_columns(difference=pl.col(REFERENCE) - pl.col("ctrl_s3_w115")))
    print(
        f"SD over {len(diff)} folds {np.std(diff, ddof=1):.4f}; MDE (2 x SD) {2 * np.std(diff, ddof=1):.4f}"
    )


def slice_report(*, scored: dict[str, pl.DataFrame]) -> None:
    """Print group means of CRPS on the slices, the sunny April-to-June gap, and the metric names."""
    one = next(iter(scored.values()))
    print("\n== metrics available:", sorted(one["metric"].unique().to_list()))
    print("== slices available:", sorted(one["slice"].unique().to_list()))
    for slice_name in SLICES:
        for group, members in SUBSETS.items():
            means = {}
            for arm, t in scored.items():
                sub = t.filter(
                    pl.col("metric") == "crps",
                    pl.col("slice") == slice_name,
                    pl.col("label").is_in(list(members)),
                )
                if sub.height:
                    means[arm] = float(np.nanmean(sub["value"].to_numpy()))
            if means:
                print(
                    f"CRPS on {slice_name}, {group}: { {a: round(v, 3) for a, v in means.items()} }"
                )
            if slice_name == "sunny_april_june" and REFERENCE in means:
                print(
                    f"  point estimates against the reference (no interval): "
                    f"{ {a: round(v - means[REFERENCE], 3) for a, v in means.items()} }"
                )


def curtailment(
    *, frames: dict[str, pl.DataFrame], series: pl.DataFrame, ctx: dict, table: pl.DataFrame
) -> None:
    """Print the avoidable curtailment cost per series at each limit fraction, if it can be computed."""
    flows = ctx["actuals"].rename({"actual": "flow_mw"})
    try:
        forecasts = {
            n: metrics.restrict_to_slice_table(forecast=f, slice_table=table).join(
                ctx["actuals"], on=["time_series_id", "valid_time"]
            )
            for n, f in frames.items()
        }
        for fraction in CURTAILMENT_FRACTIONS:
            cost = metrics.avoidable_curtailment_cost(
                forecasts=forecasts,
                series=series,
                flows=flows,
                scored_start=base.SCORED_FIRST,
                fraction=fraction,
            )
            print(f"\n== avoidable curtailment cost, pounds, limit fraction {fraction} ==")
            print(cost.pivot(on="arm", index="label", values="avoidable_cost_gbp").sort("label"))
            print(
                "total:", cost.group_by("arm").agg(pl.col("avoidable_cost_gbp").sum()).sort("arm")
            )
    except Exception as error:  # noqa: BLE001
        print(f"curtailment cost unavailable: {type(error).__name__}: {error}")


def main() -> None:
    """Print the checks, tables, and contrasts, and draw the example charts."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(s407.FLAGGED_KEYS).select("time_series_id", "valid_time")
    before = ctx["actuals"].height
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    print(f"scoring actuals dropped for a flagged exact zero: {before - ctx['actuals'].height}")
    frames = load_frames(series=series)
    check_keys(frames=frames)
    table, scored, rows = s381.score_arms(arms=frames, series=series, ctx=ctx)
    crps = headline_table(scored=scored, metric="crps")
    mae = headline_table(scored=scored, metric="mae_median")
    with pl.Config(tbl_rows=40, tbl_cols=40):
        print("\n== CRPS days 1 to 10 per Primary (points of capacity) ==")
        print(crps.sort(pl.col("label").str.slice(1).cast(pl.Int32)))
    print("\n== group means of CRPS at days 1 to 10 (equal weight per Primary) ==")
    for group, members in SUBSETS.items():
        print(group, {a: round(group_means(crps=crps, arm=a, members=members), 3) for a in frames})
    print("\n== group means of median MAE at days 1 to 10 (equal weight per Primary) ==")
    for group, members in SUBSETS.items():
        print(group, {a: round(group_means(crps=mae, arm=a, members=members), 3) for a in frames})
    if len(sys.argv) == 1 or "mde" in sys.argv[1:] or "ctrl_s3_w115" in frames:
        minimum_detectable_effect(rows=rows)
    slice_report(scored=scored)
    low = s407.sunny_low_demand(table=table, reference=frames[REFERENCE])
    print(f"\nsunny low-demand half-hours: {low.height}")
    s407.tail_table(rows=rows, low=low)
    curtailment(frames=frames, series=series, ctx=ctx, table=table)
    print("\n== contrasts (treatment minus reference, month-block bootstrap, interim) ==")
    for name, (treatment, reference) in CONTRASTS.items():
        if treatment not in rows or reference not in rows:
            continue
        for slice_label, slice_name in REPORT_ONLY_SLICES:
            for group, members in SUBSETS.items():
                where = pl.col("label").is_in(list(members)) & pl.col(HEADLINE)
                if slice_name is not None:
                    where &= pl.col(slice_name)
                for metric in ("crps", "mae_median"):
                    r = metrics.month_bootstrap_contrast(
                        treatment=rows[treatment],
                        reference=rows[reference],
                        metric=metric,
                        where=where,
                        level_percent=95.0,
                        n_resamples=20_000,
                        slice_table=table,
                    )
                    print(
                        f"{name:52s} {slice_label:14s} {group:15s} {metric:10s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
                    )
    candidates = [a for a in (REFERENCE, *(f"{n}_w115" for n in NEW_ARMS)) if a in frames]
    best = min(candidates, key=lambda a: group_means(crps=crps, arm=a, members=SIGNED))
    print(f"\nbest arm by signed-MW six CRPS among the new arms and the reference: {best}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    draw_arm_examples(arm=best, frame=frames[best], ctx=ctx, series=series)


if __name__ == "__main__":
    main()

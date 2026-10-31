# ruff: noqa: E501
"""Print every number experiment 434's page quotes, from the files `e00434_score.py` wrote.

Printed in this order: rows and complete vectors per fold, the Schaake pool size, the marginal
check (the 51-value CRPS against the stored 11-quantile CRPS), the share of actuals beyond the
smallest and largest of the 51 values, the coherence scores per arm, and the four tested
contrasts with month-block bootstrap intervals. Tables for the figures go to
`work/figure_tables/`.

```bash
DATA_PATH_INTERNAL=... PYTHONPATH=<folders of experiments 87, 377, 381, 434> \
  nice -n 19 uv run --no-sync python ml-experiments/00434-*/e00434_report.py
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00434_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

SIGNED = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SHORT = ("D15", "D18")
GROUPS = {"signed-MW six": SIGNED, "MVA-only eight": MVA_ONLY, "D15 and D18": SHORT}
N_RESAMPLES = 20_000
LEVEL_PERCENT = 100.0 - 5.0 / 4
MARGIN = 0.05
CONTRASTS = {
    "C1 ECC-T minus IND": ("ECC-T", "IND"),
    "C2 SCH minus IND": ("SCH", "IND"),
    "C3 SCH minus ECC-T": ("SCH", "ECC-T"),
    "C4 ECC-SW minus ECC-T": ("ECC-SW", "ECC-T"),
}
FIGURE_TABLE_DIR = sc.WORK_DIR / "figure_tables"


def load(*, prefix: str) -> pl.DataFrame:
    """Return all months of one output file kind, with a `fold` column."""
    frames = [
        pl.read_parquet(sc.WORK_DIR / f"{prefix}_{m}.parquet").with_columns(fold=pl.lit(m))
        for m in sc.MONTHS
        if (sc.WORK_DIR / f"{prefix}_{m}.parquet").exists()
    ]
    return pl.concat(frames)


def fold_means(*, coherence: pl.DataFrame) -> pl.DataFrame:
    """Return, per block, window, slice, arm, and fold: the mean score (Primaries then runs)."""
    return (
        coherence.group_by("block", "window", "slice", "arm", "fold", "label")
        .agg(pl.col("energy").mean(), pl.col("variogram").mean(), pl.col("n_dim").mean())
        .group_by("block", "window", "slice", "arm", "fold")
        .agg(pl.col("energy").mean(), pl.col("variogram").mean(), pl.col("n_dim").mean())
    )


def relative_contrast(
    *, folds: pl.DataFrame, treat: str, ref: str, score: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the relative difference of the mean over folds, with a month-block bootstrap interval."""
    wide = folds.pivot(on="arm", index="fold", values=score).sort("fold")
    a, b = wide[treat].to_numpy(), wide[ref].to_numpy()
    point = a.mean() / b.mean() - 1.0
    idx = rng.integers(0, len(a), size=(N_RESAMPLES, len(a)))
    boot = a[idx].mean(1) / b[idx].mean(1) - 1.0
    tail = (100.0 - LEVEL_PERCENT) / 2
    return float(point), float(np.percentile(boot, tail)), float(np.percentile(boot, 100 - tail))


def stored_crps(*, series: pl.DataFrame, ctx: dict, months: list[str]) -> pl.DataFrame:
    """Return row CRPS (points) of the stored widened 11-quantile arm A on the scored half-hours."""
    pieces = []
    for month in months:
        frame = adapter.forecast_frame(
            stored=sc.stored_arm_a(month=month), series=series, unit="fraction"
        )
        table = metrics.shared_slice_table(
            forecasts={"A": frame}, actuals=ctx["actuals"], context=ctx["context"],
            events=ctx["events"], first_readings=ctx["first_readings"],
        )  # fmt: skip
        rows = metrics.scored_rows(
            forecast=metrics.restrict_to_slice_table(forecast=frame, slice_table=table),
            actuals=ctx["actuals"], series=series, thresholds=ctx["thresholds"], slice_table=table,
        )  # fmt: skip
        pieces.append(
            rows.select("time_series_id", "label", "power_fcst_init_time", "valid_time", "crps")
        )
    return pl.concat(pieces)


def marginal_tables(*, marginal: pl.DataFrame, stored: pl.DataFrame) -> None:
    """Print the 51-value CRPS against the stored CRPS, and the share beyond the extreme values."""
    marginal = marginal.with_columns(
        valid_time=pl.col("power_fcst_init_time")
        + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64)),
    ).join(
        stored,
        on=["time_series_id", "power_fcst_init_time", "valid_time"],
        how="inner",
        suffix="_s",
    )
    print(f"\n== marginal rows joined to the stored CRPS: {marginal.height} ==")
    for window, (low, high) in sc.WINDOWS.items():
        for slice_name, col in (
            ("all", None),
            ("daylight", "daylight"),
            ("cold nights", "cold_night"),
        ):
            sub = marginal.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))
            if col:
                sub = sub.filter(pl.col(col))
            per = sub.group_by("label").agg(pl.col("crps51").mean(), pl.col("crps").mean())
            for group, members in GROUPS.items():
                g = per.filter(pl.col("label").is_in(members))
                if g.height:
                    print(
                        f"{window:10s} {slice_name:12s} {group:15s} CRPS51 {g['crps51'].mean():.3f}  "
                        f"stored {g['crps'].mean():.3f}  diff {g['crps51'].mean() - g['crps'].mean():+.3f}"  # ty: ignore[unsupported-operator]
                    )
    print(
        "\n== share of actuals above the largest / below the smallest of the 51 values (days 1-10) =="
    )
    sub = marginal.filter(pl.col("lead_hours") <= 240.0)
    per = sub.group_by("label").agg(pl.col("above_max").mean(), pl.col("below_min").mean())
    for group, members in GROUPS.items():
        g = per.filter(pl.col("label").is_in(members))
        print(
            f"{group:15s} above {g['above_max'].mean():.4f}  below {g['below_min'].mean():.4f}  (calibrated: {1 / 52:.4f})"
        )


def coherence_tables(*, folds: pl.DataFrame, rng: np.random.Generator) -> None:
    """Print absolute scores per arm and the four contrasts for every block, window, and slice."""
    print("\n== coherence scores: mean over folds (points of capacity, divided as in the plan) ==")
    keys = folds.select("block", "window", "slice").unique().sort("block", "window", "slice")
    for block, window, slice_name in keys.iter_rows():
        sub = folds.filter(
            (pl.col("block") == block)
            & (pl.col("window") == window)
            & (pl.col("slice") == slice_name)
        )
        line = "  ".join(
            f"{arm} E {sub.filter(pl.col('arm') == arm)['energy'].mean():.4f} V {sub.filter(pl.col('arm') == arm)['variogram'].mean():.4f}"
            for arm in (*sc.ARMS,)
        )
        print(
            f"{block:8s} {window:9s} {slice_name:12s} n_folds {sub['fold'].n_unique():2d}  {line}"
        )
    print(
        f"\n== contrasts: relative difference, {LEVEL_PERCENT:.2f}% month-block interval, margin {MARGIN:.0%} =="
    )
    for block, window, slice_name in keys.iter_rows():
        sub = folds.filter(
            (pl.col("block") == block)
            & (pl.col("window") == window)
            & (pl.col("slice") == slice_name)
        )
        for score in ("variogram", "energy"):
            parts = []
            for name, (treat, ref) in CONTRASTS.items():
                point, lo, hi = relative_contrast(
                    folds=sub, treat=treat, ref=ref, score=score, rng=rng
                )
                parts.append(f"{name[:2]} {point:+.1%} ({lo:+.1%} to {hi:+.1%})")
            co, lo, hi = relative_contrast(folds=sub, treat="CO", ref="IND", score=score, rng=rng)
            parts.append(f"CO-IND {co:+.1%} ({lo:+.1%} to {hi:+.1%})")
            print(f"{block:8s} {window:9s} {slice_name:12s} {score:9s} " + "  ".join(parts))


def main() -> None:
    """Print the report and write the figure tables."""
    series = adapter.series_table()
    ctx = sc.cleaned_context(series=series)
    coherence, marginal, pool = (
        load(prefix="coherence"),
        load(prefix="marginal"),
        load(prefix="pool"),
    )
    months = sorted(coherence["fold"].unique().to_list())
    print("== folds scored:", months)
    print("\n== scored half-hours per fold ==")
    print(marginal.group_by("fold").len().sort("fold"))
    print("\n== Schaake dates per fold (min, mean, max) ==")
    print(
        pool.group_by("fold")
        .agg(
            pl.col("schaake_dates").min().alias("min"),
            pl.col("schaake_dates").mean().alias("mean"),
            pl.col("schaake_dates").max().alias("max"),
        )
        .sort("fold")
    )
    print("\n== joint-block vectors scored per fold and mean dimension (days 1-10) ==")
    print(
        coherence.filter(
            (pl.col("block") == "joint")
            & (pl.col("window") == "days_1_10")
            & (pl.col("arm") == "IND")
        )
        .group_by("fold")
        .agg(pl.len().alias("runs"), pl.col("n_dim").mean().alias("mean_dim"))
        .sort("fold")
    )
    marginal_tables(marginal=marginal, stored=stored_crps(series=series, ctx=ctx, months=months))
    folds = fold_means(coherence=coherence)
    FIGURE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    folds.write_parquet(FIGURE_TABLE_DIR / "fold_means.parquet")
    coherence_tables(folds=folds, rng=np.random.default_rng(434))


if __name__ == "__main__":
    main()

"""Score the library networks of experiment 413 on the 2026-01 fold.

It scores arms D0 to D3 and the stored references on the same half-hours with
`studies.eval_protocol` (through experiment 381's scoring context), and prints:

- CRPS at days 1 to 10 per group, raw and with the 1.15 widening, and the four contrasts C1 to C4;
- C1 per Primary;
- coverage at q95 and q99 and pinball loss at q01, q05, q95, and q99 per group;
- the checks: the error of the median forecast, and the gap between the median and the mean of the
  11 quantiles (a trapezoid approximation of the mean forecast) per group.

One fold and one seed give no interval, so every number is interim.

```bash
PYTHONPATH=<the folders `e00413_fit.py` lists, plus experiments 377 and 381> \
    uv run --no-sync python ml-experiments/00413-*/e00413_score.py
```
"""

import argparse
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_score as score397  # ty: ignore[unresolved-import]
import e00413_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

HEADLINE: Final[str] = score397.HEADLINE
GROUPS: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight")
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 D2 minus D0": ("D2", "D0"),
    "C2 D1 minus D0": ("D1", "D0"),
    "C3 D2 minus D1": ("D2", "D1"),
    "C4 D3 minus D2": ("D3", "D2"),
}
TAIL_METRICS: Final[tuple[str, ...]] = (
    "coverage_q95",
    "coverage_q99",
    "pinball_q01",
    "pinball_q05",
    "pinball_q95",
    "pinball_q99",
)


def arm_frame(*, arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's forecast for the fold in the scorer's units."""
    path = fit.OUT_DIR / "seed_preds" / arm / f"{fit.FOLD_MONTH}_s0.parquet"
    return adapter.forecast_frame(stored=pl.read_parquet(path), series=series, unit="fraction")


def approximate_mean(*, frame: pl.DataFrame) -> pl.Series:
    """Return the trapezoid mean of the 11 quantiles (a mean forecast to a few per cent)."""
    levels = np.array([float(c[1:]) / 100.0 for c in common.LEVEL_COLUMNS])
    values = frame.select(common.LEVEL_COLUMNS).to_numpy()
    edges = np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2.0, [1.0]])
    return pl.Series("mean", values @ np.diff(edges))


def run() -> None:
    """Score every arm and print the tables."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    refs = score397.references(series=series)
    keep = ("stack", "xgb_monthly_e389", "xgb_monthly_e389_widened")
    arms = {a: arm_frame(arm=a, series=series) for a in fit.models.ARMS}
    stored_397 = adapter.forecast_frame(
        stored=score397.average_arm(scheme="summary", arm="plain_pooled", months=[fit.FOLD_MONTH]),
        series=series,
        unit="fraction",
    )
    everything = {
        **arms,
        **{f"{a}_w": score397.widen(frame=f) for a, f in arms.items()},
        "e397_plain_pooled_3seeds_no_cut": stored_397,
        **{k: refs[k] for k in keep},
    }
    everything = {
        k: score397.only_month(frame=v, month=fit.FOLD_MONTH) for k, v in everything.items()
    }
    _, scored, _ = s381.score_arms(arms=everything, series=series, ctx=ctx)
    rows = []
    for name, frame in scored.items():
        for metric in ("crps", *TAIL_METRICS):
            for group, value in score397.group_means(
                scored=frame, metric=metric, slice_name=HEADLINE
            ).items():
                rows.append({"arm": name, "group": group, "metric": metric, "value": value})
    table = pl.DataFrame(rows)
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=60):
        for group in GROUPS:
            print(f"\n== CRPS (days 1 to 10), {group}, points of capacity ==")
            print(
                table.filter(pl.col("group") == group, pl.col("metric") == "crps")
                .select("arm", "value")
                .sort("value")
            )
        print("\n== contrasts (CRPS, days 1 to 10; raw arms) ==")
        for label, (left, right) in CONTRASTS.items():
            gap = {
                g: float(
                    table.filter(
                        pl.col("arm") == left, pl.col("group") == g, pl.col("metric") == "crps"
                    )["value"][0]
                    - table.filter(
                        pl.col("arm") == right, pl.col("group") == g, pl.col("metric") == "crps"
                    )["value"][0]
                )
                for g in GROUPS
            }
            print(f"{label}: " + ", ".join(f"{g} {v:+.3f}" for g, v in gap.items()))
        print("\n== C1 (D2 minus D0) per Primary, CRPS days 1 to 10 ==")
        per = {
            a: scored[a]
            .filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE)
            .select("label", "group", pl.col("value").alias(a))
            for a in ("D0", "D2")
        }
        print(
            per["D0"]
            .join(per["D2"], on=["label", "group"])
            .with_columns((pl.col("D2") - pl.col("D0")).alias("D2_minus_D0"))
            .sort("label")
        )
        print("\n== tails, days 1 to 10 (coverage nominal 0.95 and 0.99; pinball in points) ==")
        print(
            table.filter(pl.col("metric").is_in(TAIL_METRICS))
            .pivot(on="metric", index=["arm", "group"], values="value")
            .sort("group", "arm")
        )
    print("\n== checks: median minus approximate mean forecast, fraction of capacity ==")
    for name in ("D0", "D1", "D2", "D3"):
        frame = everything[name]
        gap = frame["q50"] - approximate_mean(frame=frame)
        print(f"{name}: mean {gap.mean():+.5f}, mean absolute {gap.abs().mean():.5f}")
    for metric in sorted({m for m in scored["D0"]["metric"].unique() if str(m).startswith("mae")}):
        for group, value in score397.group_means(
            scored=scored["D0"], metric=metric, slice_name=HEADLINE
        ).items():
            print(f"D0 {metric} {group}: {value:.4f}")


def main() -> None:
    """Run the scorer."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()

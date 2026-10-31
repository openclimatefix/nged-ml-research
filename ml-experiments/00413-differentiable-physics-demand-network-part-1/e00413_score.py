# ruff: noqa: E501, PLR0912, PERF401
"""Score the library networks of experiment 413 on the 2026-01 fold.

It scores arms D0 to D3 and the stored references on the same half-hours with
`studies.eval_protocol` (through experiment 381's scoring context), and prints:

- CRPS at days 1 to 10 per group, raw and with the 1.15 widening, and the four contrasts C1 to C4;
- C1 per Primary;
- coverage at q95 and q99 and pinball loss at q01, q05, q95, and q99 per group;
- the checks: the error of the median forecast, and the gap between the median and the mean of the
  11 quantiles (weighted by `common.ATOM_MASSES`) per group.

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
    "C5 D2 minus D0Y": ("D2", "D0Y"),
    "C6 D0Y minus D0": ("D0Y", "D0"),
}
TAIL_METRICS: Final[tuple[str, ...]] = (
    "coverage_q95",
    "coverage_q99",
    "pinball_q01",
    "pinball_q05",
    "pinball_q95",
    "pinball_q99",
)


def arm_frame(*, arm: str, series: pl.DataFrame, months: list[str]) -> pl.DataFrame:
    """Return an arm's seed-averaged forecast over the months, in the scorer's units."""
    keys = list(common.KEYS)
    pieces = []
    for month in months:
        files = sorted((fit.OUT_DIR / "seed_preds" / arm).glob(f"{month}_s*.parquet"))
        stack = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(
            stack.group_by(keys).agg(*[pl.col(c).mean() for c in common.LEVEL_COLUMNS]).sort(keys)
        )
    frame = pl.concat(pieces)
    values = np.sort(frame.select(common.LEVEL_COLUMNS).to_numpy(), axis=1)
    frame = frame.with_columns(
        *[pl.Series(c, values[:, i]) for i, c in enumerate(common.LEVEL_COLUMNS)]
    )
    return adapter.forecast_frame(stored=frame, series=series, unit="fraction")


def approximate_mean(*, frame: pl.DataFrame) -> pl.Series:
    """Return the mean of the 11 quantiles weighted by `common.ATOM_MASSES` (MW, as the frames)."""
    values = frame.select(common.LEVEL_COLUMNS).to_numpy()
    return pl.Series("mean", values @ common.ATOM_MASSES)


def run(*, months: list[str], arm_names: list[str]) -> None:
    """Score every arm fold by fold and print the tables."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    refs = score397.references(series=series)
    arms = {a: arm_frame(arm=a, series=series, months=months) for a in arm_names}
    stored_397 = adapter.forecast_frame(
        stored=score397.average_arm(scheme="summary", arm="plain_pooled", months=months),
        series=series,
        unit="fraction",
    )
    everything = {
        **arms,
        **{f"{a}_w": score397.widen(frame=f) for a, f in arms.items()},
        "e397_plain_pooled_3seeds_no_cut": stored_397,
        **{k: refs[k] for k in ("stack", "xgb_monthly_e389", "xgb_monthly_e389_widened")},
    }
    rows = []
    scored_by_month = {}
    for month in months:
        month_arms = {k: score397.only_month(frame=v, month=month) for k, v in everything.items()}
        _, scored, _ = s381.score_arms(arms=month_arms, series=series, ctx=ctx)
        scored_by_month[month] = scored
        for name, frame in scored.items():
            for metric in ("crps", *TAIL_METRICS):
                for group, value in score397.group_means(
                    scored=frame, metric=metric, slice_name=HEADLINE
                ).items():
                    rows.append(
                        {
                            "fold": month,
                            "arm": name,
                            "group": group,
                            "metric": metric,
                            "value": value,
                        }
                    )
    table = pl.DataFrame(rows)
    table.write_parquet(fit.OUT_DIR / "score_table.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60):
        for group in GROUPS:
            print(f"\n== CRPS (days 1 to 10), {group}, points of capacity ==")
            print(
                table.filter(pl.col("group") == group, pl.col("metric") == "crps")
                .pivot(on="fold", index="arm", values="value")
                .with_columns(mean=pl.mean_horizontal(pl.exclude("arm")))
                .sort("mean")
            )
        print("\n== contrasts per fold (CRPS, raw arms) and folds won by the first arm ==")
        for label, (left, right) in CONTRASTS.items():
            for group in GROUPS:
                gaps = []
                for month in months:
                    sel = table.filter(
                        pl.col("fold") == month,
                        pl.col("group") == group,
                        pl.col("metric") == "crps",
                    )
                    gaps.append(
                        float(
                            sel.filter(pl.col("arm") == left)["value"][0]
                            - sel.filter(pl.col("arm") == right)["value"][0]
                        )
                    )
                print(
                    f"{label}, {group}: "
                    + ", ".join(f"{m} {g:+.3f}" for m, g in zip(months, gaps, strict=True))
                    + f"; mean {np.mean(gaps):+.3f}; first arm lower in {sum(g < 0 for g in gaps)} of {len(gaps)}"
                )
        print("\n== tails, mean over folds (coverage nominal 0.95 and 0.99; pinball in points) ==")
        print(
            table.filter(pl.col("metric").is_in(TAIL_METRICS))
            .group_by("arm", "group", "metric")
            .agg(pl.col("value").mean())
            .pivot(on="metric", index=["arm", "group"], values="value")
            .sort("group", "arm")
        )
        print("\n== C1 per Primary, mean over folds, CRPS days 1 to 10 ==")
        per = []
        for month, scored in scored_by_month.items():
            for a in ("D0", "D2"):
                per.append(
                    scored[a]
                    .filter(pl.col("metric") == "crps", pl.col("slice") == HEADLINE)
                    .select("label", "group", pl.lit(a).alias("arm"), "value")
                    .with_columns(fold=pl.lit(month))
                )
        wide = (
            pl.concat(per)
            .group_by("label", "group", "arm")
            .agg(pl.col("value").mean())
            .pivot(on="arm", index=["label", "group"], values="value")
            .with_columns((pl.col("D2") - pl.col("D0")).alias("D2_minus_D0"))
            .sort("label")
        )
        print(wide)
    print("\n== checks: median minus approximate mean forecast, MW ==")
    for name in arm_names:
        gap = everything[name]["q50"] - approximate_mean(frame=everything[name])
        print(f"{name}: mean {gap.mean():+.5f}, mean absolute {gap.abs().mean():.5f}")
    for metric in sorted(
        {
            m
            for m in next(iter(scored_by_month.values()))["D0"]["metric"].unique()
            if str(m).startswith("mae")
        }
    ):
        for arm in arm_names:
            vals = [
                score397.group_means(scored=sc[arm], metric=metric, slice_name=HEADLINE)
                for sc in scored_by_month.values()
            ]
            print(
                f"{arm} {metric}: "
                + ", ".join(f"{g} {np.mean([v[g] for v in vals]):.4f}" for g in GROUPS)
            )


def main() -> None:
    """Run the scorer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(fit.base.SCREEN_MONTHS))
    parser.add_argument("--arms", nargs="+", default=["D0", "D0Y", "D1", "D2"])
    args = parser.parse_args()
    run(months=args.months, arm_names=args.arms)


if __name__ == "__main__":
    main()
